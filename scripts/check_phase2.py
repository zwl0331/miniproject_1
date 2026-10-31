"""Self-checks of the phase 2 harness (task P2.1, doc/plans/phase2-models-features.md section 4.8), without pytest.

From the project root: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.check_phase2
Everything runs in memory and writes nothing to the project (libraries may use the system's temporary directory,
as EBM's joblib backend does). Checks that need a package not installed yet are reported as
SKIPPED, and checks of code that later tasks build are reported as STAGED, never as passed. Exit status 0 only if
no check failed.
"""
import builtins
import copy
import importlib.metadata
import io
import itertools
import json
import math
import pickle
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from songpop import artifacts
from songpop.data import AUDIO, DATE
from songpop.evaluate import PredictionBank
from songpop.metrics import paired_cluster_bootstrap
from songpop.phase2 import allocation, banks, batches, feature_stage, protocol, refine, runs, searches
from songpop.phase2.encoders import DateCount, DateTargetEncoder, EraRelative, FitContext, inner_group_splits
from songpop.phase2.engine import fit_recipe, run_recipe_cv, training_weights
from songpop.phase2.ensembles import blend_banks, select_seeds
from songpop.phase2.features import FeaturePipeline, feature_columns
from songpop.phase2.models import (EBM_FIXED, EBM_PAIRS, THREADS, ebm_pairs, make_estimator, make_model,
                                   make_preprocessing,
                                   prediction_threads)
from songpop.phase2.recipes import load_recipe, recipe_hash, recipe_id, validate_recipe
from songpop.phase2.submission import members_of, repeated_uri_pairs, validate_candidate, write_candidate
from songpop.splits import SPLIT_SEEDS
from scripts import install_locks, run_experiments  # loaded before the source manifest is taken
from analysis import p23_summary  # likewise

ROOT = Path(".").resolve()
RESULTS = []
ANCHOR = "configs/phase2/recipes/lightgbm_F2_anchor.json"
PHASE1_LGBM_F2 = "artifacts/bank/production_lightgbm_F2_9c37e77d549e.npz"
P13_CSV = "submissions/2026-10-29_p1.3_lightgbm_F2.csv"


def check(name, status="run"):
    def wrap(fn):
        if status != "run":
            RESULTS.append((name, status, fn.__doc__ or ""))
            return fn
        try:
            note = fn()
            RESULTS.append((name, "PASS", note or ""))
        except Skip as e:
            RESULTS.append((name, "SKIPPED", str(e)))
        except Exception as e:  # report every failure, keep going
            RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        return fn
    return wrap


class Skip(Exception):
    pass


def raises(fn, exc=Exception, match=None):
    try:
        fn()
    except exc as e:
        if match is not None and match not in str(e):
            raise AssertionError(f"raised {type(e).__name__} without {match!r}: {e}")
        return
    raise AssertionError("did not raise")


class MemFS:
    """An in-memory file system for paths under a fake root, so checks of writing code run the real code and
    write nothing; every other path uses the real file system."""

    def __init__(self, root="/memfs-check-phase2"):
        self.root, self.files = root, {}

    def owns(self, path):
        return str(path).startswith(self.root)

    def __enter__(self):
        fs, real_open = self, builtins.open
        real = {n: getattr(Path, n) for n in ("exists", "read_text", "read_bytes", "mkdir")}

        class Writer:
            def __init__(self, key, start, binary):
                self.key, self.buf, self.binary = key, bytearray(start), binary

            def write(self, s):
                self.buf += s if self.binary else s.encode()
                return len(s)

            def close(self):
                fs.files[self.key] = bytes(self.buf)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        def fake_open(file, mode="r", *a, **k):
            if not fs.owns(file):
                return real_open(file, mode, *a, **k)
            key, binary = str(file), "b" in mode
            if "r" in mode:
                if key not in fs.files:
                    raise FileNotFoundError(key)
                return io.BytesIO(fs.files[key]) if binary else io.StringIO(fs.files[key].decode())
            if "x" in mode and key in fs.files:
                raise FileExistsError(key)
            return Writer(key, fs.files.get(key, b"") if "a" in mode else b"", binary)

        def owned(name, fake):
            return lambda p, *a, **k: fake(p) if fs.owns(p) else real[name](p, *a, **k)

        self.patches = [mock.patch("builtins.open", fake_open),
                        mock.patch.object(Path, "exists", owned("exists", lambda p: str(p) in fs.files)),
                        mock.patch.object(Path, "read_bytes", owned("read_bytes", lambda p: fs.files[str(p)])),
                        mock.patch.object(Path, "read_text", owned("read_text", lambda p: fs.files[str(p)].decode())),
                        mock.patch.object(Path, "mkdir", owned("mkdir", lambda p: None))]
        for x in self.patches:
            x.start()
        return self

    def __exit__(self, *exc):
        for x in reversed(self.patches):
            x.stop()


class Patches:
    """Several mock patches as one context."""

    def __init__(self, patches):
        self.patches = patches

    def __enter__(self):
        for x in self.patches:
            x.start()

    def __exit__(self, *exc):
        for x in reversed(self.patches):
            x.stop()


CTX = runs.Context.load(ROOT)
MANIFEST = banks.source_manifest(ROOT)
INV = {"argv": ["check_phase2"]}


def recipe(**over):
    r = load_recipe(ROOT / ANCHOR)
    r = copy.deepcopy(r)
    r.update(over)
    return validate_recipe(r)


def make_features_has_nan(rows):
    return FeaturePipeline(recipe()).fit_transform(rows, None, None, None).isna().any().any()


SMALL_LGBM = {"n_estimators": 60, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 20,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}
EVERY_BLOCK = ["date_count", "date_te10", "interactions", "cyclic", "era_relative"]


def frame(dates, **audio):
    """Raw rows with the given dates and defaults for the audio columns."""
    n = len(dates)
    d = {c: np.full(n, 0.5) for c in AUDIO}
    d.update({"key": np.full(n, 5.0), "mode": np.ones(n), "time_signature": np.full(n, 4.0), "loudness": np.full(n, -6.0),
              "tempo": np.full(n, 120.0), "duration_ms": np.full(n, 200000.0)})
    d.update({k: np.asarray(v, dtype="float64") for k, v in audio.items()})
    d[DATE] = list(dates)
    return pd.DataFrame(d, index=np.arange(n))[AUDIO + [DATE]]


@check("recipes: validation rejects malformed recipes; ids ignore key order")
def _():
    base = load_recipe(ROOT / ANCHOR)
    bad = [dict(base, family="tabpfn"), dict(base, base="F3"), dict(base, blocks=["date_te10", "date_te10"]),
           dict(base, blocks=["date_te10", "date_te30"]), dict(base, blocks=["lookup_uri"]),
           dict(base, estimator_seeds=[]), dict(base, estimator_seeds=[0, 0]), dict(base, estimator_seeds=[True]),
           dict(base, deployment="full_refit"), dict(base, iterations="early_stopping"),
           dict(base, training_weights="balanced"), {**base, "extra": 1}, dict(base, name="bad name"),
           dict(base, preprocessing={"name": "pca"}), dict(base, preprocessing={"name": "spline"}),
           dict(base, preprocessing={"name": "spline", "n_knots": 2}),
           dict(base, preprocessing={"name": "none", "n_knots": 4}),
           dict(base, family="logistic", params={"C": 1.0}), dict(base, drop=["year", "year"]),
           dict(base, drop=["date_count"]), dict(base, drop=list(feature_columns(base)))]
    for r in bad:
        raises(lambda r=r: validate_recipe(r), ValueError)
    shuffled = dict(reversed(list(base.items())))
    assert recipe_id(shuffled) == recipe_id(base)
    assert recipe_id(dict(base, estimator_seeds=[1])) != recipe_id(base)
    raises(lambda: make_model(dict(base, params={**base["params"], "class_weight": "balanced"}), 0,
                              feature_columns(base)), ValueError)


@check("searches: anchors first and exact; draws seeded, deduplicated, reproducible")
def _():
    spec = {"schema": "search_v1", "name": "probe", "family": "lightgbm", "base": "F2", "blocks": [], "drop": [],
            "preprocessing": {"name": "none"}, "training_weights": "none", "estimator_seeds": [0],
            "fixed": {"subsample_freq": 1},
            "space": {"num_leaves": [7, 15], "learning_rate": [0.02, 0.05], "n_estimators": [300, 600]},
            "anchors": [dict(load_recipe(ROOT / ANCHOR)["params"])], "draws": 7, "sampler_seed": 1}
    a, b = searches.expand(spec), searches.expand(spec)
    assert [recipe_id(r) for r in a] == [recipe_id(r) for r in b]
    assert a[0]["params"] == spec["anchors"][0] and len(a) == 8
    keys = [artifacts.dumps(r["params"]) for r in a]
    assert len(set(keys)) == len(keys)
    raises(lambda: searches.expand(dict(spec, draws=9)), ValueError, "fewer than")
    assert [r["params"] for r in searches.expand(dict(spec, sampler_seed=2))] != [r["params"] for r in a]
    raises(lambda: searches.validate_search(dict(spec, fixed={"num_leaves": 7})), ValueError)
    spline = dict(spec, family="logistic", base="F1", preprocessing={"name": "spline", "n_knots": 4}, fixed={},
                  space={"C": [0.1, 1.0, 10.0], "preprocessing.n_knots": [4, 7]}, anchors=[], draws=6)
    got = sorted((r["params"]["C"], r["preprocessing"]["n_knots"]) for r in searches.expand(spline))
    assert got == sorted((c, k) for c in (0.1, 1.0, 10.0) for k in (4, 7)), got


@check("searches v2: derived parameters, admissible regions, rejection counts; v1 expansions unchanged")
def _():
    v1 = {"schema": "search_v1", "name": "probe", "family": "lightgbm", "base": "F2", "blocks": [], "drop": [],
          "preprocessing": {"name": "none"}, "training_weights": "none", "estimator_seeds": [0],
          "fixed": {"subsample_freq": 1},
          "space": {"num_leaves": [7, 15], "learning_rate": [0.02, 0.05], "n_estimators": [300, 600]},
          "anchors": [dict(load_recipe(ROOT / ANCHOR)["params"])], "draws": 7, "sampler_seed": 1}
    # The expansion of this specification under the committed P2.1 code (281a014), pinned: v2 leaves v1 untouched.
    assert artifacts.sha256_bytes(artifacts.dumps([recipe_id(r) for r in searches.expand(v1)]).encode()) == \
        "4731448fa1b8469a6bbabf5738f62605fe78551fb107c47a23ebacd14779a4eb"
    space = {"num_leaves": [7, 15], "learning_rate": [0.01, 0.1], "n_estimators": [300, 1600],
             "subsample": [0.7, 1.0]}
    v2 = dict(v1, schema="search_v2", fixed={}, space=space, draws=8,
              derived=[{"when": {"param": "subsample", "op": "lt", "value": 1}, "set": {"subsample_freq": 1}}],
              admissible=[{"product": ["learning_rate", "n_estimators"], "min": 5, "max": 100}])
    recipes, rejected = searches.expand_v2(v2)
    assert recipes[0]["params"] == v1["anchors"][0]  # anchors exact and exempt
    for r in recipes[1:]:
        q = r["params"]
        assert 5 <= q["learning_rate"] * q["n_estimators"] <= 100, q
        assert ("subsample_freq" in q) == (q["subsample"] < 1) and q.get("subsample_freq", 1) == 1, q
    # 2 x 2 x 2 x 2 = 16 combinations; lr x trees of 3 (0.01 x 300) and 160 (0.1 x 1600) are inadmissible: 8 remain.
    assert len(recipes) == 9 and rejected["inadmissible"] > 0, (len(recipes), rejected)
    raises(lambda: searches.expand_v2(dict(v2, draws=9)), ValueError, "no new admissible configuration")
    got, info = searches.expand_v2(dict(v2, draws=9), allow_fewer=True)
    assert len(got) == 9 and info["termination"] == "enumerated_exhaustion"
    # A sparse admissible region inside a space too large to enumerate fails instead of passing for exhaustion.
    sparse = dict(v2, space={f"p{i}": list(range(12)) for i in range(5)}, admissible=[], derived=[], anchors=[],
                  draws=5, family="random_forest", fixed={})
    with mock.patch.object(searches, "MAX_ATTEMPTS", 3), mock.patch.object(searches, "admissible", lambda q, a: False):
        raises(lambda: searches.expand_v2(sparse, allow_fewer=True), ValueError, "not proven exhausted")
    assert searches.expand(v2) == recipes and searches.expand_v2(v2) == (recipes, rejected)
    raises(lambda: searches.validate_search(dict(v2, fixed={"subsample_freq": 1})), ValueError, "derived")
    raises(lambda: searches.validate_search(dict(v2, admissible=[{"product": ["learning_rate"], "min": 5,
                                                                      "max": 100}])), ValueError)
    raises(lambda: searches.validate_search(dict(v1, derived=[])), ValueError)
    return f"{len(recipes) - 1} admissible draws, rejections {rejected}; exhaustion proven by enumeration, a sparse " \
           "region refused"


@check("kNN: the block-weighted representation of plan section 6.2, fitted on the training part only")
def _():
    knn = recipe(family="knn", params={"n_neighbors": 25, "weights": "distance", "metric": "euclidean"}, base="F1",
                 preprocessing={"name": "knn_blocks", "alpha": 0.7, "category_weight": 0.5, "normalization": "l2"})
    tr = np.flatnonzero(CTX.fold_id([155])[0] != 0)
    f = fit_recipe(knn, CTX.rows.iloc[tr], CTX.data.y[tr], CTX.groups[tr], FitContext(155, 0))
    Z = f.model_inputs(CTX.rows.iloc[tr])
    assert Z.shape[1] == 10 + 1 + 18, Z.shape  # ten continuous, the year, one-hot key (12), mode (2), metre (4)
    assert np.allclose(Z[:, :10].std(axis=0), 1 / np.sqrt(10), atol=1e-9)
    assert np.allclose(Z[:, :10].mean(axis=0), 0, atol=1e-9) and abs(Z[:, 10].std() - 0.7) < 1e-9
    assert set(np.round(Z[:, 11:].ravel(), 12)) == {0.0, round(0.5 / np.sqrt(3), 12)}
    m = recipe(**{**knn, "params": dict(knn["params"], metric="manhattan"),
                  "preprocessing": dict(knn["preprocessing"], normalization="l1")})
    Zm = fit_recipe(m, CTX.rows.iloc[tr], CTX.data.y[tr], CTX.groups[tr], FitContext(155, 0)).model_inputs(
        CTX.rows.iloc[tr])
    assert np.allclose(Zm[:, :10].std(axis=0), 1 / 10, atol=1e-9)
    assert set(np.round(Zm[:, 11:].ravel(), 12)) == {0.0, round(0.5 / 6, 12)}
    for bad in (dict(base="F2"), dict(estimator_seeds=[0, 1]), dict(training_weights="inverse_group_size"),
                dict(blocks=["cyclic"]), dict(preprocessing={"name": "log_standard"}),
                dict(params=dict(knn["params"], metric="manhattan")), dict(params=dict(knn["params"], metric="cosine")),
                dict(drop=["key"]), dict(params=dict(knn["params"], n_neighbors=0)),
                dict(preprocessing={"name": "knn_blocks", "alpha": 0, "category_weight": 0.5, "normalization": "l2"})):
        raises(lambda: validate_recipe(dict(knn, **bad)), ValueError)
    raises(lambda: validate_recipe(dict(knn, family="logistic")), ValueError)
    # Fitted on the training part only: held-out rows never enter the fit; a changed training row changes the fitted
    # statistics (positive control) while a changed held-out row cannot.
    rows = CTX.rows.iloc[tr].copy()
    rows.iloc[0, rows.columns.get_loc("loudness")] += 30.0
    g = fit_recipe(knn, rows, CTX.data.y[tr], CTX.groups[tr], FitContext(155, 0))
    va = np.flatnonzero(CTX.fold_id([155])[0] == 0)
    assert not np.allclose(g.model_inputs(CTX.rows.iloc[va]), f.model_inputs(CTX.rows.iloc[va]))
    heldout = CTX.rows.copy()
    heldout.iloc[va, heldout.columns.get_loc("loudness")] += 30.0
    f2 = fit_recipe(knn, heldout.iloc[tr], CTX.data.y[tr], CTX.groups[tr], FitContext(155, 0))
    assert np.array_equal(f2.model_inputs(CTX.test_rows), f.model_inputs(CTX.test_rows))
    tie = KNeighborsClassifier(n_neighbors=1, algorithm="brute").fit(np.array([[1.0], [-1.0], [2.0]]), np.array([0, 1, 1]))
    assert len({float(tie.predict_proba([[0.0]])[0, 1]) for _ in range(5)}) == 1  # a k-th-neighbor tie is repeatable
    p1, p2 = f.predict_high(CTX.test_rows), f.predict_high(CTX.test_rows)
    assert np.array_equal(p1, p2) and np.all((p1 >= 0) & (p1 <= 1))
    # Exact matches and ties (synthetic): zero-distance neighbors take the weight; identical queries, alone or in a
    # batch at any position, get identical predictions; an unseen category is all zeros.
    Xs, ys = np.array([[0.0], [0.0], [2.0], [4.0]]), np.array([0, 1, 1, 0])
    kd = KNeighborsClassifier(n_neighbors=3, weights="distance", algorithm="brute").fit(Xs, ys)
    ku = KNeighborsClassifier(n_neighbors=3, weights="uniform", algorithm="brute").fit(Xs, ys)
    assert kd.predict_proba([[0.0]])[0, 1] == 0.5 and abs(ku.predict_proba([[0.0]])[0, 1] - 2 / 3) < 1e-12
    q = CTX.test_rows.iloc[[5, 1, 5, 2, 5]]
    batch, alone = f.predict_high(q), f.predict_high(CTX.test_rows.iloc[[5]])
    assert batch[0] == batch[2] == batch[4] == alone[0]
    odd = CTX.test_rows.iloc[[0]].copy()
    odd["key"] = 99.0
    zi = f.model_inputs(odd)[0, 11:]
    assert np.isfinite(f.predict_high(odd)).all() and np.count_nonzero(zi) == 2  # mode and metre known, key unseen
    return "29 columns; block weights 1/sqrt(10), alpha, beta/sqrt(3) (1/10, alpha, beta/6 under Manhattan); " \
           "exact matches, ties, batch positions and unseen categories behave as declared"


@check("anchor: the LightGBM F2 recipe through the new engine equals the phase 1 bank exactly")
def _():
    b = runs.fitted_run(CTX, load_recipe(ROOT / ANCHOR), [155, 2026, 1029], "P2.1", MANIFEST, INV)
    p1 = artifacts.load_bank(ROOT / PHASE1_LGBM_F2)
    diff = [f for f in PredictionBank.ARRAYS if not np.array_equal(getattr(b, f), getattr(p1, f))]
    assert not diff, f"arrays differ: {diff}"
    assert b.metadata["config"]["resolved_parameters"][0] == p1.metadata["parameters"]
    assert artifacts.submission_bytes(b.test_id, b.test_mean) == (ROOT / P13_CSV).read_bytes()
    one = runs.fitted_run(CTX, load_recipe(ROOT / ANCHOR), [2026], "P2.1", MANIFEST, INV)
    assert np.array_equal(one.fold_id[0], p1.fold_id[1]) and np.array_equal(one.oof[0], p1.oof[1])
    assert runs.fitted_config(CTX, load_recipe(ROOT / ANCHOR), [2026, 155], MANIFEST) == \
        runs.fitted_config(CTX, load_recipe(ROOT / ANCHOR), [155, 2026], MANIFEST)
    return (f"mean fold AUC {b.fold_auc.mean():.10f}, all {len(PredictionBank.ARRAYS)} arrays bit-identical; its "
            f"submission bytes equal the P1.3 file; seed 2026 alone uses the saved second repeat")


@check("label isolation: changing one held-out fold's labels changes no prediction of that fold's model")
def _():
    r = recipe(name="probe_isolation", params=SMALL_LGBM, blocks=EVERY_BLOCK, training_weights="inverse_group_size")
    fold_id = CTX.fold_id([155])
    k = 2
    va = fold_id[0] == k
    y2 = CTX.data.y.copy()
    y2[va] = 1 - y2[va]
    a, b = [run_recipe_cv(r, CTX.rows, yy, CTX.data.row_ids, CTX.groups, fold_id, [155], CTX.test_rows,
                          CTX.data.test_id) for yy in (CTX.data.y, y2)]
    assert np.array_equal(a.oof[0, va], b.oof[0, va]), "held-out predictions changed"
    assert np.array_equal(a.test_by_fold[0, k], b.test_by_fold[0, k]), "the fold model's test predictions changed"
    others = [j for j in range(5) if j != k]
    assert not np.array_equal(a.test_by_fold[0, others], b.test_by_fold[0, others]), "positive control: no effect"
    return "every block and inverse-group weights; the other folds' models do change (positive control)"


@check("preprocessing isolation: an extreme value in one held-out row changes nothing else that fold's model predicts")
def _():
    r = recipe(name="probe_preprocessing", family="logistic", params={"C": 1.0}, blocks=["date_count", "era_relative"],
               preprocessing={"name": "spline", "n_knots": 4})
    fold_id = CTX.fold_id([155])
    k = 3
    va = np.flatnonzero(fold_id[0] == k)
    rows2 = CTX.rows.copy()
    rows2.iloc[va[0], rows2.columns.get_loc("loudness")] = 1e6
    a, b = [run_recipe_cv(r, rr, CTX.data.y, CTX.data.row_ids, CTX.groups, fold_id, [155], CTX.test_rows,
                          CTX.data.test_id) for rr in (CTX.rows, rows2)]
    assert np.array_equal(a.oof[0, va[1:]], b.oof[0, va[1:]]) and np.array_equal(a.test_by_fold[0, k], b.test_by_fold[0, k])
    others = [j for j in range(5) if j != k]
    assert not np.array_equal(a.test_by_fold[0, others], b.test_by_fold[0, others]), "positive control: no effect"
    return "imputation, splines, scaling and the era means are fitted on the training part only"


@check("date target encoding: hand-computed cross-fitting, mixed-label groups, unseen dates, the prior")
def _():
    dates = ["A", "A", "A", "B", "B", "C", "C", "C", "A", "B", "D", "D"]
    y = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
    groups = np.array([0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9])  # group 0 repeats with one label; group 5 mixed
    rows = frame(dates)
    m = 10
    enc = DateTargetEncoder(m)
    ctx = FitContext(155, 0)
    out = enc.fit_transform(rows, y, groups, ctx)["date_te10"].to_numpy()
    expected = np.full(len(y), np.nan)
    for tr, te in enc.splits_:
        assert not set(groups[tr]) & set(groups[te])
        prior = y[tr].mean()
        for i in te:
            same = [j for j in tr if dates[j] == dates[i]]
            expected[i] = (y[same].sum() + m * prior) / (len(same) + m)
    assert np.allclose(out, expected, rtol=0, atol=1e-12), (out, expected)
    ext = enc.transform(frame(["A", "C", "Z"]))["date_te10"].to_numpy()
    prior = y.mean()
    want = [(y[[i for i, d in enumerate(dates) if d == c]].sum() + m * prior) / (dates.count(c) + m) for c in "AC"]
    assert np.allclose(ext, want + [prior], rtol=0, atol=1e-12), (ext, want, prior)


@check("date target encoding on the real data: with inner folds held fixed, a row's group never encodes it")
def _():
    rows, y, groups = CTX.rows, CTX.data.y, CTX.groups
    sizes = pd.Series(groups).value_counts()
    dates = rows[DATE].to_numpy()
    g = next(g for g in sizes.index[sizes.values > 1]
             if ((dates == dates[groups == g][0]) & (groups != g)).sum() >= 3)
    in_g = groups == g
    ctx = FitContext(155, 1)
    splits = inner_group_splits(y, groups, ctx.inner_seed("date_te"))  # held fixed for both label sets
    a = DateTargetEncoder(10).fit_transform(rows, y, groups, ctx, splits).to_numpy()[:, 0]
    assert np.array_equal(a, DateTargetEncoder(10).fit_transform(rows, y, groups, ctx).to_numpy()[:, 0])
    y2 = y.copy()
    y2[in_g] = 1 - y2[in_g]
    b = DateTargetEncoder(10).fit_transform(rows, y2, groups, ctx, splits).to_numpy()[:, 0]
    assert np.array_equal(a[in_g], b[in_g]), "flipping a group's labels changed its own encoding"
    same_date = (rows[DATE] == rows[DATE][in_g].iloc[0]).to_numpy() & ~in_g
    assert same_date.any() and not np.array_equal(a[same_date], b[same_date]), "positive control: no effect"
    assert sorted(np.concatenate([te for _, te in splits]).tolist()) == list(range(len(y)))
    every = np.arange(len(y))
    for bad in ([(every, every)], [(every[:-10], every[-10:])], [(s[0], s[1][:-1]) for s in splits],
                [(s[0], np.r_[s[1], s[1][:1]]) for s in splits]):
        raises(lambda bad=bad: DateTargetEncoder(10).fit_transform(rows, y, groups, ctx, bad), ValueError)
    enc = DateTargetEncoder(10)
    enc.fit_transform(rows, y, groups, ctx)
    assert enc.receipt["inner_seed"] == ctx.inner_seed("date_te") and len(enc.receipt["inner_splits_sha256"]) == 64
    return f"group {g} of size {int(sizes[g])}; other rows of its date do change (positive control)"


@check("date count: the own group left out for fitting rows; zero for an unseen date")
def _():
    rows = frame(["A", "A", "A", "B", "C"])
    groups = np.array([1, 1, 2, 3, 3 + 1])
    enc = DateCount()
    tr = enc.fit_transform(rows, None, groups, FitContext(1, 0))["date_count"].to_numpy()
    assert np.allclose(tr, np.log1p([1, 1, 1, 0, 0]))
    ext = enc.transform(frame(["A", "B", "Z"]))["date_count"].to_numpy()
    assert np.allclose(ext, np.log1p([2, 1, 0]))
    multi = DateCount().fit_transform(frame(["A", "B", "A"]), None, np.array([7, 7, 8]), FitContext(1, 0))
    assert np.allclose(multi["date_count"].to_numpy(), np.log1p([1, 0, 1]))  # group 7 spans two dates


@check("era-relative audio: era means shrunk with 20 pseudo-rows; an unseen era uses the fitting mean")
def _():
    rows = frame(["1985", "1986-02-01", "2021-01-01", "2022", "2023-05-05"], loudness=[-10, -12, -5, -6, -4],
                 acousticness=[0.8, 0.6, 0.1, 0.2, 0.3])
    enc = EraRelative()
    out = enc.fit_transform(rows, None, None, None)
    g = np.mean([-10, -12, -5, -6, -4])
    old = (-22 + 20 * g) / (2 + 20)
    assert math.isclose(out["loudness_era_rel"].iloc[0], -10 - old, abs_tol=1e-12)
    ext = enc.transform(frame(["1995"], loudness=[-7]))
    assert math.isclose(ext["loudness_era_rel"].iloc[0], -7 - g, abs_tol=1e-12)


@check("cyclic and interaction blocks: 2 pi phases, wraparound, NaN where unobserved, formulas")
def _():
    fp = FeaturePipeline(recipe(blocks=["cyclic", "interactions"]))
    X = fp.fit_transform(frame(["2023-01-15", "2023-02-15", "2023-12-15", "2024", "2024-12-31", "1979-06"],
                               tempo=[120] * 6, duration_ms=[180000] * 6, loudness=[-6] * 6), None, None, None)
    v = X[["month_sin", "month_cos"]].to_numpy()
    assert math.isclose(np.linalg.norm(v[0] - v[1]), np.linalg.norm(v[2] - v[0]), rel_tol=1e-12)
    assert X.loc[3, ["month_sin", "weekday_cos", "doy_sin"]].isna().all()
    assert X.loc[5, ["weekday_sin", "doy_cos"]].isna().all() and not math.isnan(X.loc[5, "month_sin"])
    assert math.isclose(X.loc[4, "doy_sin"], math.sin(2 * math.pi * 365 / 366), abs_tol=1e-12)
    assert math.isclose(X.loc[0, "beats"], 360.0) and math.isclose(X.loc[0, "ix_year_loudness"], 23 * -6.0)
    assert set(fp.lineage_) == set(X.columns) and fp.lineage_["beats"] == ["tempo", "duration_ms"]


@check("training weights: inverse group size, mean one in the fitting rows; they change the fit")
def _():
    groups = np.array([1, 1, 2, 3, 3, 3])
    w = training_weights(recipe(training_weights="inverse_group_size"), groups)
    assert math.isclose(w.mean(), 1.0) and math.isclose(w[0] / w[2], 0.5) and math.isclose(w[3] / w[2], 1 / 3)
    assert training_weights(recipe(), groups) is None
    tr = np.flatnonzero(CTX.fold_id([155])[0] != 0)[:800]
    a = fit_recipe(recipe(params=SMALL_LGBM), CTX.rows.iloc[tr], CTX.data.y[tr], CTX.groups[tr], FitContext(155, 0))
    b = fit_recipe(recipe(params=SMALL_LGBM, training_weights="inverse_group_size"), CTX.rows.iloc[tr],
                   CTX.data.y[tr], CTX.groups[tr], FitContext(155, 0))
    assert not np.array_equal(a.predict_high(CTX.test_rows), b.predict_high(CTX.test_rows))


def describe_matches(fitted, r):
    """The fitted estimator is the recipe's preprocessing (if any) followed by its family's estimator."""
    from sklearn.pipeline import Pipeline
    est = fitted.estimators[0]
    if r["preprocessing"]["name"] == "none":
        return not isinstance(est, Pipeline)
    return isinstance(est, Pipeline) and [n for n, _ in est.steps] == ["prep", "model"]


@check("recipes: the model, the feature blocks and the preprocessing vary independently")
def _():
    base = recipe(params=SMALL_LGBM)
    variants = {"model": dict(base, family="random_forest", params={"n_estimators": 50, "min_samples_leaf": 3}),
                "blocks": dict(base, blocks=["date_count", "cyclic"], drop=["weekday"]),
                "preprocessing": dict(base, preprocessing={"name": "log_standard"})}
    tr = np.flatnonzero(CTX.fold_id([155])[0] != 0)
    fits = {k: fit_recipe(v, CTX.rows.iloc[tr], CTX.data.y[tr], CTX.groups[tr], FitContext(155, 0))
            for k, v in [("base", base), *variants.items()]}
    ids = {k: recipe_id(v) for k, v in [("base", base), *variants.items()]}
    assert len(set(ids.values())) == 4
    assert fits["model"].features.columns_ == fits["base"].features.columns_ == fits["preprocessing"].features.columns_
    assert fits["blocks"].features.columns_ == feature_columns(variants["blocks"]) != fits["base"].features.columns_
    assert "weekday" not in fits["blocks"].features.columns_ and "weekday" not in fits["blocks"].features.lineage_
    assert type(fits["model"].estimators[0]).__name__ == "RandomForestClassifier"
    assert type(fits["base"].estimators[0]).__name__ == "LGBMClassifier"
    prep = fits["preprocessing"].estimators[0]
    assert [n for n, _ in prep.steps] == ["prep", "model"] and type(prep.named_steps["model"]).__name__ == "LGBMClassifier"
    return ("family, blocks (with a dropped column) and preprocessing each changed alone: four recipe ids, the other "
            "two parts unchanged")


@check("engine: no screening, promotion or selection rule in the fitting code")
def _():
    engine_files = ["recipes.py", "features.py", "encoders.py", "models.py", "engine.py"]
    banned = ["experiments.jsonl", "LOG_FILE", "read_experiments", "mean_fold_auc", "0.003", "promot", "select_candidate"]
    hits = [(f, b) for f in engine_files for b in banned
            if b in (ROOT / "songpop" / "phase2" / f).read_text()]
    assert not hits, hits
    r = recipe(params=SMALL_LGBM, blocks=["era_relative"])
    tr = np.flatnonzero(CTX.fold_id([155])[0] != 0)[:500]
    fit_recipe(r, CTX.rows.iloc[tr], CTX.data.y[tr], CTX.groups[tr], FitContext(155, 0))
    return "the fitting modules read no log and hold no threshold; any valid recipe fits"


@check("threads: HistGB fits and predicts under an OpenMP limit of four; the other families set their own")
def _():
    from threadpoolctl import threadpool_info
    from songpop.phase2.models import THREADS, thread_limit
    with thread_limit("hist_gb"):
        inside = [p["num_threads"] for p in threadpool_info() if p["user_api"] == "openmp"]
    assert inside and all(n == THREADS for n in inside), inside
    seen = []
    real_fit = __import__("sklearn.ensemble", fromlist=["HistGradientBoostingClassifier"]).HistGradientBoostingClassifier.fit

    def spy(self, *a, **k):
        seen.append([p["num_threads"] for p in threadpool_info() if p["user_api"] == "openmp"])
        return real_fit(self, *a, **k)
    tr = np.flatnonzero(CTX.fold_id([155])[0] != 0)[:600]
    with mock.patch("sklearn.ensemble.HistGradientBoostingClassifier.fit", spy):
        fit_recipe(recipe(family="hist_gb", params={"max_iter": 20, "max_leaf_nodes": 7, "min_samples_leaf": 20,
                                                    "learning_rate": 0.1}), CTX.rows.iloc[tr], CTX.data.y[tr],
                   CTX.groups[tr], FitContext(155, 0))
    assert seen and all(n == THREADS for n in seen[0]), seen
    return f"OpenMP pools inside the HistGB fit: {seen[0]}"


@check("model factories: every installed family fits F2 with missing values and returns P(class 1)")
def _():
    tr = np.flatnonzero(CTX.fold_id([155])[0] != 0)
    sub = CTX.rows.iloc[tr]
    assert make_features_has_nan(sub)
    families = {"lightgbm": SMALL_LGBM, "random_forest": {"n_estimators": 50, "min_samples_leaf": 3},
                "extra_trees": {"n_estimators": 50, "min_samples_leaf": 3},
                "xgboost": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
                "hist_gb": {"max_iter": 50, "max_leaf_nodes": 15, "min_samples_leaf": 20, "learning_rate": 0.05},
                "logistic_spline": {"C": 1.0}, "logistic_log_standard": {"C": 1.0}}
    preps = {"logistic_spline": ("F1", {"name": "spline", "n_knots": 4}),
             "logistic_log_standard": ("F2", {"name": "log_standard"})}
    out = []
    for fam, params in families.items():
        base, prep = preps.get(fam, ("F2", {"name": "none"}))
        r = recipe(family=fam.split("_")[0] if fam in preps else fam, params=params, base=base, preprocessing=prep)
        f = fit_recipe(r, sub, CTX.data.y[tr], CTX.groups[tr], FitContext(155, 0))
        p = f.predict_high(CTX.test_rows)
        va = np.flatnonzero(CTX.fold_id([155])[0] == 0)
        auc = roc_auc_score(CTX.data.y[va], f.predict_high(CTX.rows.iloc[va]))
        assert list(f.estimators[0].classes_) == [0, 1] and np.all((p >= 0) & (p <= 1)) and auc > 0.6, (fam, auc)
        assert describe_matches(f, r)
        out.append(f"{fam} {auc:.3f}")
    return "held-out AUC on one fold: " + ", ".join(out)


@check("forest prediction: one thread at prediction, four at fit, restored, bare and in a pipeline (plan section 16.4)")
def _():
    tr = np.flatnonzero(CTX.fold_id([155])[0] != 0)
    sub, y, g = CTX.rows.iloc[tr], CTX.data.y[tr], CTX.groups[tr]
    seen = []
    real = {cls: cls.predict_proba for cls in (RandomForestClassifier, ExtraTreesClassifier)}

    def spy(cls):
        def wrapped(self, X):
            seen.append((cls.__name__, self.n_jobs))
            return real[cls](self, X)
        return wrapped
    with mock.patch.object(RandomForestClassifier, "predict_proba", spy(RandomForestClassifier)), \
            mock.patch.object(ExtraTreesClassifier, "predict_proba", spy(ExtraTreesClassifier)):
        for fam in ("random_forest", "extra_trees"):
            assert make_estimator(fam, {"n_estimators": 30, "min_samples_leaf": 3}, 0).n_jobs == THREADS
            f = fit_recipe(recipe(family=fam, params={"n_estimators": 30, "min_samples_leaf": 3}), sub, y, g,
                           FitContext(155, 0))
            outs = {f.predict_high(CTX.test_rows).tobytes() for _ in range(10)}
            assert len(outs) == 1 and f.estimators[0].n_jobs == THREADS, fam
            assert f.receipt["forest_prediction_threads"] == 1, fam
    assert seen and all(n == 1 for _, n in seen), seen
    lg = fit_recipe(recipe(params=SMALL_LGBM), sub, y, g, FitContext(155, 0))
    assert "forest_prediction_threads" not in lg.receipt
    with prediction_threads("lightgbm", lg.estimators):
        assert lg.estimators[0].n_jobs == THREADS  # other families untouched
    pipe = Pipeline([("prep", StandardScaler()), ("model", ExtraTreesClassifier(n_estimators=20, n_jobs=THREADS,
                                                                               random_state=0))])
    pipe.fit(sub.drop(columns=[DATE]).fillna(0.0), y)
    with prediction_threads("extra_trees", [pipe]):
        assert pipe.steps[-1][1].n_jobs == 1
    assert pipe.steps[-1][1].n_jobs == THREADS
    try:
        with prediction_threads("extra_trees", [pipe]):
            raise KeyError("inside")
    except KeyError:
        pass
    assert pipe.steps[-1][1].n_jobs == THREADS  # restored after an error too
    return f"{len(seen)} forest predictions, all with one thread; 10 repeated predictions identical; pipeline restored"


@check("music block: circle-of-fifths key and tempo octave, raw key and tempo kept, lineage (plan section 16.3)")
def _():
    rows = frame(["2020-01-01"] * 4, key=[0, 7, 2, 11], tempo=[60.0, 120.0, 240.0, 90.0])
    r = recipe(blocks=["music"])
    X = FeaturePipeline(r).fit_transform(rows, np.array([0, 1, 0, 1]), np.arange(4), FitContext(155, 0))
    assert list(X.columns) == feature_columns(r) and list(X.columns[-4:]) == [
        "key_fifths_sin", "key_fifths_cos", "tempo_octave_sin", "tempo_octave_cos"]
    want = 2 * np.pi * np.array([0, 1, 2, 5]) / 12  # (7 key) mod 12 for keys 0, 7, 2, 11
    assert np.allclose(X["key_fifths_sin"], np.sin(want)) and np.allclose(X["key_fifths_cos"], np.cos(want))
    assert np.allclose(X.loc[[0, 1, 2], ["tempo_octave_sin", "tempo_octave_cos"]].to_numpy(),
                       X.loc[[0, 0, 0], ["tempo_octave_sin", "tempo_octave_cos"]].to_numpy(), rtol=0, atol=1e-12)
    assert not np.allclose(X.loc[3, "tempo_octave_sin"], X.loc[0, "tempo_octave_sin"])
    assert X["key"].tolist() == [0, 7, 2, 11] and X["tempo"].tolist() == [60.0, 120.0, 240.0, 90.0]
    pipe = FeaturePipeline(r)
    pipe.fit_transform(rows, np.array([0, 1, 0, 1]), np.arange(4), FitContext(155, 0))
    assert pipe.lineage_["key_fifths_cos"] == ["key"] and pipe.lineage_["tempo_octave_sin"] == ["tempo"]
    full = FeaturePipeline(r).fit_transform(CTX.rows, CTX.data.y, CTX.groups, FitContext("all", "all"))
    assert np.isfinite(full[["key_fifths_sin", "key_fifths_cos", "tempo_octave_sin", "tempo_octave_cos"]]).all().all()
    return "fifths mapping, octave equality for 60, 120 and 240, finite on every eligible row"


def _runtime_fixture():
    """Fitting rows: the first 1,500 of seed 155's fold-0 training part (full dates and year-only dates, so missing
    date parts). Prediction rows: the first 400 test rows and the one month-precision training row (1979-06), so all
    three date precisions are predicted. The fitted recipes see nothing else."""
    tr = np.flatnonzero(CTX.fold_id([155])[0] != 0)[:1500]
    rows = CTX.rows.iloc[tr]
    dates = CTX.rows["track_album_release_date"].astype(str)
    month = CTX.rows[dates.str.len() == 7]
    pred = pd.concat([CTX.test_rows.iloc[:400], month])
    lengths = set(pred["track_album_release_date"].astype(str).str.len())
    assert make_features_has_nan(rows) and lengths == {4, 7, 10}, lengths
    return rows, CTX.data.y[tr], CTX.groups[tr], pred


def _runtime(r, fixture):
    """Two fits of the recipe (identical predictions), the class-1 column, a pickle round trip, no new top-level
    entry in the project root and no catboost_info directory (CatBoost's file writing and snapshots are disabled in
    the factory); returns the first fit and its predictions."""
    import pickle
    rows, y, groups, test = fixture
    before = sorted(p.name for p in ROOT.iterdir())
    a = fit_recipe(r, rows, y, groups, FitContext(155, 0))
    b = fit_recipe(r, rows, y, groups, FitContext(155, 0))
    pa = a.predict_high(test)
    assert np.array_equal(pa, b.predict_high(test)), "two fits with the same seed differ"
    assert pa.shape == (len(test),) and np.all((pa >= 0) & (pa <= 1)) and np.unique(pa).size > 10
    est, X = a.estimators[0], a.features.transform(test)
    assert list(est.classes_) == [0, 1] and np.array_equal(pa, est.predict_proba(X)[:, 1])
    assert np.array_equal(pa, pickle.loads(pickle.dumps(a)).predict_high(test)), "pickle round trip differs"
    assert sorted(p.name for p in ROOT.iterdir()) == before, "a new entry appeared in the project root"
    assert not (ROOT / "catboost_info").exists(), "CatBoost wrote its training directory into the working directory"
    return a, pa


@check("CatBoost runtime: fits F2 with missing values, predicts all three date precisions, repeatable, pickle round "
       "trip, no new project entry, P(class 1); settings and refusals (plan sections 5.3, 16.3, 17.2)")
def _():
    import catboost
    if importlib.metadata.version("catboost") != "1.2.10":
        raise AssertionError(f"catboost {importlib.metadata.version('catboost')} is not the locked 1.2.10")
    fx = _runtime_fixture()
    notes = []
    for bt in ("Plain", "Ordered"):
        r = recipe(name=f"probe_catboost_{bt.lower()}", family="catboost",
                   params={"iterations": 60, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 3.0,
                           "random_strength": 1.0, "boosting_type": bt})
        a, _ = _runtime(r, fx)
        est = a.estimators[0]
        got = est.get_all_params()
        assert got["boosting_type"] == bt and got["bootstrap_type"] == "MVS" and abs(got["subsample"] - 0.8) < 1e-6
        assert got["rsm"] == 1 and got["nan_mode"] == "Min" and got["auto_class_weights"] == "None" \
            and got["loss_function"] == "Logloss" and "od_type" not in got  # no stopping
        gp = est.get_params()
        assert (gp["thread_count"], gp["allow_writing_files"], gp["save_snapshot"], gp["use_best_model"],
                gp["has_time"]) == (THREADS, False, False, False, False)
        notes.append(f"{bt}: MVS bootstrap, subsample 0.8, rsm 1, depth {got['depth']}, border count "
                     f"{got['border_count']}, leaf estimation {got['leaf_estimation_method']} x "
                     f"{got['leaf_estimation_iterations']}")
    # Prediction runs with THREADS threads (CatBoost's own default is every core), and identical rows, NaNs
    # included, get identical predictions.
    from catboost import CatBoostClassifier
    rows, y, groups, test = fx
    nan_rows = rows.index[a.features.transform(rows).isna().any(axis=1).to_numpy()]
    twin = pd.concat([rows.loc[nan_rows[:3]]] * 2)
    calls = []
    original = CatBoostClassifier.predict_proba

    def spy(self, X, *args, **kw):
        calls.append(kw.get("thread_count"))
        return original(self, X, *args, **kw)
    with mock.patch.object(CatBoostClassifier, "predict_proba", spy):
        tw = a.predict_high(twin)
    assert calls == [THREADS] and len(nan_rows) >= 3 and np.array_equal(tw[:3], tw[3:])
    w = np.linspace(0.5, 1.5, len(y))
    make_model(r, 0, a.features.columns_).fit(a.features.transform(rows), y, sample_weight=w)  # P2.4's weight route
    c31 = recipe(name="probe_catboost_c31", family="catboost",
                 params={"iterations": 60, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 4.757135294733201,
                         "random_strength": 0.0, "rsm": 0.8880009208730744, "bootstrap_type": "Bayesian",
                         "bagging_temperature": 1.793473187168845, "boosting_type": "Plain"})
    a, _ = _runtime(c31, fx)
    got = a.estimators[0].get_all_params()
    assert got["bootstrap_type"] == "Bayesian" and abs(got["bagging_temperature"] - 1.793473187168845) < 1e-6 \
        and abs(got["rsm"] - 0.8880009208730744) < 1e-6
    for bad in ({"bootstrap_type": "MVS", "bagging_temperature": 1.0}, {"bagging_temperature": 1.0},
                {"bootstrap_type": "Bayesian"}, {"rsm": 1}, {"rsm": 1.5}, {"bootstrap_type": "Bernoulli"},
                {"border_count": 128}):
        raises(lambda bad=bad: make_estimator("catboost", dict({"iterations": 10}, **bad), 0), ValueError)
    raises(lambda: make_model(recipe(family="catboost", params={"iterations": 10},
                                     preprocessing={"name": "log_standard"}), 0, ["year"]), ValueError)
    return "; ".join(notes) + "; prediction with 4 threads; identical NaN rows predicted identically; weights " \
        "accepted; candidate 31's Bayesian bootstrap, temperature and rsm in effect; eight refusals"


@check("EBM runtime: fits F2 with missing values, predicts all three date precisions, max_rounds x terms boosting "
       "updates per stage, declared pairs, repeatable, pickle round trip, no new project entry, P(class 1); refusals "
       "(plan sections 5.2, 17.2)")
def _():
    if importlib.metadata.version("interpret-core") != "0.7.8":
        raise AssertionError(f"interpret-core {importlib.metadata.version('interpret-core')} is not the locked 0.7.8")
    fx = _runtime_fixture()
    notes = []
    for inter, n_pairs in (("none", 0), ("declared3", 3)):
        r = recipe(name=f"probe_ebm_{inter}", family="ebm",
                   params={"max_rounds": 120, "learning_rate": 0.03, "interactions": inter})
        a, _ = _runtime(r, fx)
        est = a.estimators[0]
        n_main = len(a.features.columns_)
        names = list(est.term_names_)
        assert len(names) == n_main + n_pairs and names[:n_main] == list(a.features.columns_)
        if n_pairs:
            assert sorted(names[n_main:]) == sorted(["year & loudness", "year & acousticness", "energy & loudness"])
        # With no validation split and no early stopping, each stage makes max_rounds x (its number of terms)
        # boosting updates; greedy selection may spread them unevenly across terms, so only stage totals are checked.
        assert np.all(np.asarray(est.best_iteration_) == 120 * np.array([[n_main]] + ([[n_pairs]] if n_pairs else []))), \
            f"boosting stopped early: {est.best_iteration_}"
        params = est.get_params()
        assert all(params[k] == v for k, v in EBM_FIXED.items()) and params["n_jobs"] == THREADS
        assert params["feature_types"] == ["continuous"] * n_main and set(est.feature_types_in_) == {"continuous"}
        notes.append(f"{inter}: {len(names)} terms, 120 x terms boosting updates per stage")
    rows, y, groups, test = fx
    nan_rows = rows.index[a.features.transform(rows).isna().any(axis=1).to_numpy()]
    tw = a.predict_high(pd.concat([rows.loc[nan_rows[:3]]] * 2))
    assert len(nan_rows) >= 3 and np.array_equal(tw[:3], tw[3:])
    w = np.linspace(0.5, 1.5, len(y))
    make_model(r, 0, a.features.columns_).fit(a.features.transform(rows), y, sample_weight=w)  # P2.4's weight route
    cols = list(reversed(a.features.columns_))  # pairs follow the names, wherever the columns stand
    assert [(cols[i], cols[j]) for i, j in ebm_pairs("declared3", cols)] == EBM_PAIRS["declared3"]
    raises(lambda: make_estimator("ebm", {"max_rounds": 10, "interactions": 3}, 0, ["year"]), ValueError)
    raises(lambda: make_estimator("ebm", {"max_rounds": 10, "interactions": "declared3"}, 0, ["year", "energy"]),
           ValueError, "need the columns")
    raises(lambda: make_estimator("ebm", {"max_rounds": 10}, 0), ValueError, "columns")
    raises(lambda: make_model(recipe(family="ebm", params={"max_rounds": 10},
                                     preprocessing={"name": "log_standard"}), 0, ["year"]), ValueError)
    return "; ".join(notes) + "; every feature continuous; identical NaN rows predicted identically; weights " \
        "accepted; pairs resolved by name on reordered columns; four refusals"


@check("banks: bank_v2 round trip; metadata validation rejects every inconsistency")
def _():
    b = runs.fitted_run(CTX, recipe(name="probe_bank", params=SMALL_LGBM), [155], "P2.1", MANIFEST, INV)
    back = banks.load_bank_v2(artifacts.bank_bytes(b))
    assert not banks.compare_v2(b, back)
    raises(lambda: banks.load_bank_v2(artifacts.bank_bytes(b), n_splits=3), ValueError)
    for key, val in [("schema", "bank_v1"), ("run_id", "p2_x_seeds155_000000000000"), ("prediction_kind", "blend"),
                     ("n_repeats", 3), ("estimator_seeds", [1]), ("recipe_sha256", "0" * 64),
                     ("evaluation_population", {"rows": 1, "row_ids_sha256": "0"}), ("config_sha256", "0" * 64)]:
        bad = copy.deepcopy(b)
        bad.metadata[key] = val
        raises(lambda bad=bad: banks.validate_v2(bad), ValueError)
    bad = copy.deepcopy(b)
    del bad.metadata["feature_lineage"]
    raises(lambda: banks.validate_v2(bad), ValueError)
    forged = copy.deepcopy(b)
    forged.metadata["selection_eligible"] = True
    raises(lambda: banks.validate_v2(forged), ValueError, "selection_eligible")
    corrupt = copy.deepcopy(b)
    corrupt.fold_auc[0, 2] += 0.01
    raises(lambda: banks.validate_v2(corrupt), ValueError)
    assert not b.metadata["selection_eligible"] and b.metadata["stage"] == "seeds155"

    def rehashed(edit):
        """A copy with one contradiction whose config hash and run id are recomputed, so only that link can fail."""
        x = copy.deepcopy(b)
        edit(x.metadata)
        x.metadata["config_sha256"] = banks.run_identity(x.metadata["config"])
        x.metadata["run_id"] = banks.make_run_id("probe_bank", x.metadata["stage"], x.metadata["config"])
        return x
    contradictions = {
        "recipe_sha256": lambda md: md["config"].update(recipe_sha256="0" * 64),
        "fold_ids_sha256": lambda md: md["config"].update(fold_ids_sha256="0" * 64),
        "estimator_seeds": lambda md: md["config"].update(estimator_seeds=[1]),
        "split_seeds": lambda md: md["config"].update(split_seeds=[2026]),
        "evaluation_population": lambda md: md["config"].update(evaluation_population={"rows": 1}),
        "source_manifest_sha256": lambda md: md["config"].update(source_manifest_sha256="0" * 64),
        "source manifest digest": lambda md: md["source_manifest"]["files"].update({"songpop/data.py": "0" * 64}),
        "parents": lambda md: md["config"].update(parents=[{"run_id": "x"}]),
        "features": lambda md: md.update(features=md["features"][::-1]),
        "fit_receipts": lambda md: md.update(fit_receipts=md["fit_receipts"][:-1]),
        "stage": lambda md: md.update(stage="seeds2026")}
    for name, edit in contradictions.items():
        raises(lambda x=rehashed(edit): banks.validate_v2(x), ValueError)
    banks.validate_v2(rehashed(lambda md: None))
    three = copy.deepcopy(b)  # a valid three-fold fixture, made by merging folds 0-1, 2-3 and 4
    merge = np.array([0, 0, 1, 1, 2])
    three.fold_id = merge[three.fold_id]
    three.test_by_fold = np.stack([three.test_by_fold[:, merge == j].mean(axis=1) for j in range(3)], axis=1)
    three.test_mean = three.test_by_fold.mean(axis=(0, 1))
    three.fold_auc = np.array([[roc_auc_score(three.y[three.fold_id[0] == j], three.oof[0, three.fold_id[0] == j])
                                for j in range(3)]])
    md = three.metadata
    md.update(n_splits=3, fit_receipts=md["fit_receipts"][:3])
    md["config"]["fold_ids_sha256"] = artifacts.sha256_bytes(np.ascontiguousarray(three.fold_id, np.int64).tobytes())
    md["config_sha256"] = banks.run_identity(md["config"])
    md["run_id"] = banks.make_run_id("probe_bank", md["stage"], md["config"])
    back3 = banks.load_bank_v2(artifacts.bank_bytes(three), n_splits=3)
    assert not banks.compare_v2(three, back3)
    raises(lambda: banks.load_bank_v2(artifacts.bank_bytes(three)), ValueError)
    assert b.metadata["fit_receipts"][0]["inner"] == [] and len(b.metadata["fit_receipts"]) == 5
    return f"{len(contradictions)} single contradictions refused; a three-fold bank round-trips with n_splits=3"


@check("verification: a source mismatch is refused and names the replay commit")
def _():
    stored = MANIFEST
    current = copy.deepcopy(MANIFEST)
    current["files"]["songpop/phase2/engine.py"] = "0" * 64
    raises(lambda: runs.check_sources(CTX, stored, current), runs.SourceMismatch)
    runs.check_sources(CTX, stored, copy.deepcopy(stored))
    p1 = artifacts.load_bank(ROOT / PHASE1_LGBM_F2).metadata["source_manifest"]
    commit = runs.find_replay_commit(ROOT, p1)
    assert commit is not None
    fake = {"files": {"songpop/data.py": "0" * 64}, "manifest_sha256": "x"}
    assert runs.find_replay_commit(ROOT, fake) is None
    cfg = {"python": artifacts.environment(ROOT)["python"], "packages": artifacts.packages()}
    runs.check_sources(CTX, stored, None, cfg)
    raises(lambda: runs.check_sources(CTX, stored, None, dict(cfg, packages={**cfg["packages"], "lightgbm": "0.0"})),
           runs.SourceMismatch, "packages")
    return f"the phase 1 LightGBM F2 manifest replays at {commit[:7]} (its recorded head is " \
           f"{artifacts.load_bank(ROOT / PHASE1_LGBM_F2).metadata['environment']['git_head'][:7]})"


def _two_banks():
    a = runs.fitted_run(CTX, recipe(name="probe_a", params=SMALL_LGBM), [155, 2026, 1029], "P2.1", MANIFEST, INV)
    b = runs.fitted_run(CTX, recipe(name="probe_b", params=dict(SMALL_LGBM, num_leaves=7)), [155, 2026, 1029], "P2.1",
                        MANIFEST, INV)
    return a, b


BANK_A, BANK_B = None, None


@check("blends: identity, member order, weights, identities, recomputed AUCs")
def _():
    global BANK_A, BANK_B
    BANK_A, BANK_B = _two_banks()
    a, b = BANK_A, BANK_B
    twin = copy.deepcopy(a)
    twin.metadata["run_id"] = "p2_twin"
    same = blend_banks([(a, "h1"), (twin, "h2")], [0.5, 0.5])
    assert np.allclose(same.oof, a.oof, rtol=0, atol=1e-15) and np.allclose(same.fold_auc, a.fold_auc, rtol=0, atol=1e-15)
    ab = blend_banks([(a, "h1"), (b, "h2")], [0.25, 0.75])
    ba = blend_banks([(b, "h2"), (a, "h1")], [0.75, 0.25])
    assert np.allclose(ab.oof, ba.oof, rtol=0, atol=banks.TOL) and \
        np.allclose(ab.test_mean, ba.test_mean, rtol=0, atol=banks.TOL)
    m = ab.fold_id[1] == 3
    assert math.isclose(ab.fold_auc[1, 3], roc_auc_score(ab.y[m], 0.25 * a.oof[1, m] + 0.75 * b.oof[1, m]))
    for w in ([0.6, 0.6], [1.1, -0.1], [1.0, 0.0], [0.5]):
        raises(lambda w=w: blend_banks([(a, "h1"), (b, "h2")], w), ValueError)
    raises(lambda: blend_banks([(a, "h1"), (a, "h1")], [0.5, 0.5]), ValueError, "twice")
    shifted = copy.deepcopy(b)
    shifted.fold_id = np.roll(shifted.fold_id, 1, axis=1)
    raises(lambda: blend_banks([(a, "h1"), (shifted, "h2")], [0.5, 0.5]), ValueError)
    other_test = copy.deepcopy(b)
    other_test.test_id = other_test.test_id[::-1].copy()
    raises(lambda: blend_banks([(a, "h1"), (other_test, "h2")], [0.5, 0.5]), ValueError, "test_id")
    singles = [runs.fitted_run(CTX, recipe(name="probe_seed", params=SMALL_LGBM, estimator_seeds=[s]), [155], "P2.1",
                               MANIFEST, INV) for s in (0, 1, 2)]
    three = runs.fitted_run(CTX, recipe(name="probe_seed", params=SMALL_LGBM, estimator_seeds=[0, 1, 2]), [155], "P2.1",
                            MANIFEST, INV)
    assert three.metadata["prediction_kind"] == "seed_average"
    assert np.array_equal(three.oof, np.mean([s.oof for s in singles], axis=0))
    assert np.array_equal(three.test_by_fold, np.mean([s.test_by_fold for s in singles], axis=0))
    mixed = blend_banks([(three, "h3"), (select_seeds(a, [155]), "h1")], [0.75, 0.25])
    assert np.allclose(mixed.oof, 0.75 * three.oof + 0.25 * a.oof[:1], rtol=0, atol=1e-15)
    return "estimator seeds [0, 1, 2] equal the mean of three single-seed runs exactly; a blend keeps recipe weights"


@check("matched comparisons and declared blends, also on a declared subset of split seeds")
def _():
    a, b = BANK_A, BANK_B
    res = paired_cluster_bootstrap(a, b, n_boot=200)
    assert res["valid"] == 200 and math.isclose(res["observed_difference"], a.fold_auc.mean() - b.fold_auc.mean(),
                                                abs_tol=1e-12)
    one = runs.fitted_run(CTX, recipe(name="probe_one_seed", params=dict(SMALL_LGBM, num_leaves=31)), [155], "P2.1",
                          MANIFEST, INV)
    raises(lambda: paired_cluster_bootstrap(a, one, n_boot=10), ValueError)
    raises(lambda: blend_banks([(a, "ha"), (one, "h1")], [0.75, 0.25]), ValueError)
    a155 = select_seeds(a, [155])
    assert np.array_equal(a155.oof[0], a.oof[0]) and np.array_equal(a155.test_by_fold[0], a.test_by_fold[0])
    assert np.allclose(a155.test_mean, a.test_by_fold[0].mean(axis=0), rtol=0, atol=1e-15)
    blend = runs.blend_run(CTX, "probe_contribution", [(a, "ha"), (one, "h1")], [0.75, 0.25], "P2.1", MANIFEST, INV,
                           seeds=[155])
    assert blend.metadata["split_seeds"] == [155] and not blend.metadata["selection_eligible"]
    assert np.allclose(blend.oof[0], 0.75 * a.oof[0] + 0.25 * one.oof[0], rtol=0, atol=1e-15)
    gain = paired_cluster_bootstrap(blend, a155, n_boot=200)
    assert gain["valid"] == 200
    raises(lambda: select_seeds(a, [7]), ValueError)
    return "a three-seed leader blended 0.75/0.25 with a seed-155 challenger on seed 155 and compared, paired"


@check("submissions: a valid candidate passes; every defect, a partial bank and an altered parent are refused")
def _():
    a, b = BANK_A, BANK_B
    if a is None:
        raise AssertionError("the blend check did not build its banks")
    content = artifacts.submission_bytes(a.test_id, a.test_mean)
    report = validate_candidate(content, CTX, a, [(a, None)])
    assert report["repeated_uri_pairs_checked"] == 11 and len(repeated_uri_pairs(CTX.data.test)) == 11
    df = pd.read_csv(io.BytesIO(content))
    pair = repeated_uri_pairs(CTX.data.test)[0]
    defects = {
        "column order": df[["Popularity_Type", "ID"]], "missing row": df.iloc[:-1],
        "ID order": df.iloc[::-1], "above one": df.assign(Popularity_Type=df["Popularity_Type"] + 1.0),
        "NaN": df.assign(Popularity_Type=df["Popularity_Type"].where(df.index != 3)),
        "differs from bank": df.assign(Popularity_Type=df["Popularity_Type"] * 0.999)}
    for what, d in defects.items():
        raises(lambda d=d: validate_candidate(d.to_csv(index=False).encode(), CTX, a, [(a, None)]), ValueError)
    tampered = copy.deepcopy(a)
    tampered.test_by_fold[0, 0, pair[1]] += 0.01
    tampered.test_mean = tampered.test_by_fold.mean(axis=(0, 1))
    raises(lambda: validate_candidate(artifacts.submission_bytes(tampered.test_id, tampered.test_mean), CTX, tampered,
                                      [(tampered, None)]), ValueError, "identical model inputs")
    partial = runs.fitted_run(CTX, recipe(name="probe_partial", params=SMALL_LGBM), [155], "P2.1", MANIFEST, INV)
    raises(lambda: validate_candidate(artifacts.submission_bytes(partial.test_id, partial.test_mean), CTX, partial,
                                      [(partial, None)]), ValueError, "three-repeat")
    blend = runs.blend_run(CTX, "probe_blend", [(a, "ha"), (b, "hb")], [0.5, 0.5], "P2.1", MANIFEST, INV)
    store = {a.metadata["run_id"]: (a, "ha"), b.metadata["run_id"]: (b, "hb")}
    members = members_of(CTX, blend, lambda r: store[r])
    validate_candidate(artifacts.submission_bytes(blend.test_id, blend.test_mean), CTX, blend, members)
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
    a2.test_by_fold[:, :, pair[1]] += 0.02
    b2.test_by_fold[:, :, pair[1]] -= 0.02
    for m in (a2, b2):
        m.test_mean = m.test_by_fold.mean(axis=(0, 1))
    swapped = {a.metadata["run_id"]: (a2, "ha"), b.metadata["run_id"]: (b2, "hb")}
    tm = members_of(CTX, blend, lambda r: swapped[r])
    assert np.allclose(blend_banks(tm, [0.5, 0.5]).test_mean, blend.test_mean, rtol=0, atol=1e-12)
    raises(lambda: validate_candidate(artifacts.submission_bytes(blend.test_id, blend.test_mean), CTX, blend, tm),
           ValueError, "member")
    store[b.metadata["run_id"]] = (b, "altered")
    raises(lambda: members_of(CTX, blend, lambda r: store[r]), ValueError, "recorded hash")
    month = runs.fitted_run(CTX, recipe(name="probe_month", family="logistic", params={"C": 1.0},
                                        drop=[c for c in feature_columns(recipe()) if c != "month"],
                                        preprocessing={"name": "log_standard"}), [155, 2026, 1029], "P2.1", MANIFEST, INV)
    year_only = CTX.data.test["track_album_release_date"].str.len().to_numpy() == 4
    bent = copy.deepcopy(month)
    bent.test_by_fold[:, :, year_only] += 0.01  # missing months become the median month after imputation
    bent.test_mean = bent.test_by_fold.mean(axis=(0, 1))
    raises(lambda: validate_candidate(artifacts.submission_bytes(bent.test_id, bent.test_mean), CTX, bent,
                                      [(bent, None)]), ValueError, "identical model inputs")
    with MemFS() as fs:
        target = Path(fs.root) / "candidate.csv"
        write_candidate(target, CTX, a, [(a, None)])
        raises(lambda: write_candidate(target, CTX, a, [(a, None)]), FileExistsError)
        other = Path(fs.root) / "candidate2.csv"
        current = Path.read_bytes  # the in-memory version; only the target's read-back is corrupted
        with mock.patch.object(Path, "read_bytes",
                               lambda p, *x, **k: b"ID,Popularity_Type\n" if str(p) == str(other) else current(p)):
            raises(lambda: write_candidate(other, CTX, a, [(a, None)]), ValueError, "read back")
    return f"member duplicate classes {report['member_fold_duplicate_classes']}, raw classes {report['raw_duplicate_classes']}"


@check("searches: the configuration list is saved before the first fit; a failure is logged and stops the search")
def _():
    rx = run_experiments
    spec = {"schema": "search_v1", "name": "probe_spy", "family": "lightgbm", "base": "F2", "blocks": [], "drop": [],
            "preprocessing": {"name": "none"}, "training_weights": "none", "estimator_seeds": [0],
            "fixed": {"subsample_freq": 1}, "space": {"num_leaves": [7, 15, 31]}, "anchors": [], "draws": 3,
            "sampler_seed": 3}
    with MemFS() as fs:
        tmp = Path(fs.root)
        seen = []

        def spy(recipe, sid):
            key = str(tmp / "searches" / f"{sid}.json")
            assert key in fs.files, "a fit started before the list was saved"
            assert len(run_experiments.strict_loads(fs.files[key].decode())["recipes"]) == 3, "the saved list is incomplete"
            seen.append(recipe_id(recipe))
        sid = rx.run_search(CTX, spec, [155], "P2.1", MANIFEST, INV, directory=tmp / "searches", log_path=tmp / "log.jsonl",
                            runner=spy)
        assert seen == [recipe_id(r) for r in searches.expand(spec)]

        def boom(recipe, sid):
            raise RuntimeError("probe failure")
        raises(lambda: rx.run_search(CTX, dict(spec, sampler_seed=4), [155], "P2.1", MANIFEST, INV,
                                     directory=tmp / "searches", log_path=tmp / "log.jsonl", runner=boom), RuntimeError)
        recs = artifacts.read_experiments(tmp / "log.jsonl")
        assert [r["record"] for r in recs] == ["search", "search", "run_failure"], [r["record"] for r in recs]
    dup = dict(spec, family="logistic", base="F1", fixed={}, preprocessing={"name": "spline", "n_knots": 4},
               anchors=[{"C": 1.0}], space={"C": [1.0, 10.0], "preprocessing.n_knots": [4, 7]}, draws=3)
    got = searches.expand(dup)
    assert len({r["preprocessing"]["n_knots"] * 100 + r["params"]["C"] for r in got}) == 4
    raises(lambda: searches.expand(dict(dup, draws=4)), ValueError, "fewer than")
    return f"search {sid} in an in-memory file system; an implicit and an explicit n_knots=4 count once"


@check("verification writes nothing: a run under the current sources recomputes with every writer disabled")
def _():
    """A probe run outside the search spaces, logged in an in-memory file system under the current sources. The
    historical anchor's refusal under changed sources is the check "verification: a source mismatch is refused"."""
    probe = recipe(name="probe_verify", params=dict(SMALL_LGBM, n_estimators=20))
    with MemFS() as fs, Patches([mock.patch.object(runs, "LOG_FILE", Path(fs.root) / "log.jsonl"),
                                 mock.patch.object(runs, "BANK_DIR", Path(fs.root) / "bank")]):
        rid = runs.save_run(CTX, runs.fitted_run(CTX, probe, list(SPLIT_SEEDS), "P9.9", MANIFEST, INV))["run_id"]
        boom = mock.Mock(side_effect=AssertionError("a writer was called during verification"))
        with mock.patch.object(artifacts, "append_experiment", boom), mock.patch.object(banks, "save_bank_v2", boom), \
                mock.patch.object(searches, "save_expansion", boom), mock.patch.object(runs, "save_run", boom):
            bad = runs.verify_run(CTX, rid, MANIFEST, INV)
        assert not bad and not boom.called, bad
        logged = runs.logged(CTX)
        tampered = [dict(r, mean_fold_auc_unrounded=0.123, repeat_mean_auc_unrounded=[0.123] * 3) for r in logged]
        with mock.patch.object(runs, "logged", lambda ctx, record="run": tampered if record == "run" else []):
            bad = runs.verify_run(CTX, rid, MANIFEST, INV)
        assert "logged mean_fold_auc_unrounded" in bad and "logged repeat_mean_auc_unrounded" in bad, bad
        fields = {"family": "random_forest", "stage": "seeds155", "task_id": "P9.8", "search_id": "none",
                  "source_manifest_hash": "0" * 64, "n_rows": 1, "n_groups": 1, "n_features": 1,
                  "execution_status": "failed", "warnings": ["probe"], "bank_sha256": "0" * 64, "recipe_id": "x",
                  "record": "other"}
        for k, v in fields.items():
            one = [dict(r, **{k: v}) if r["run_id"] == rid else r for r in logged]
            with mock.patch.object(runs, "logged", lambda ctx, record="run", one=one: one if record == "run" else []):
                try:
                    bad = runs.verify_run(CTX, rid, MANIFEST, INV)
                except ValueError:
                    bad = [f"logged {k}"]  # a record the loader itself refuses
            assert f"logged {k}" in bad, (k, bad)
    return f"a probe run equals its saved bank and log with every writer disabled; tampered unrounded scores and " \
           f"{len(fields)} other single record fields are caught"


@check("comparisons: seed-restricted, matched, logged once, verified")
def _():
    recs = [r for r in runs.logged(CTX) if r.get("recipe_id", "").startswith("lightgbm_F2_anchor_")]
    if not recs:
        raise Skip("no logged anchor run yet")
    anchor = recs[-1]["run_id"]
    probe = runs.fitted_run(CTX, recipe(name="probe_compare", params=SMALL_LGBM), [155], "P2.1", MANIFEST, INV)
    with MemFS() as fs:
        log = Path(fs.root) / "experiments.jsonl"
        fs.files[str(log)] = (ROOT / runs.LOG_FILE).read_bytes()
        with mock.patch.object(runs, "LOG_FILE", log), mock.patch.object(runs, "BANK_DIR", Path(fs.root) / "bank"):
            rec = runs.save_run(CTX, probe)
            before = len(runs.logged(CTX, "comparison"))  # the live log may already hold comparisons of real tasks
            raises(lambda: run_experiments.compare(CTX, anchor, rec["run_id"], "P2.1"), ValueError)
            run_experiments.compare(CTX, anchor, rec["run_id"], "P2.1", seeds=[155])
            run_experiments.compare(CTX, anchor, rec["run_id"], "P2.1", seeds=[155])
            cmp = [r for r in runs.logged(CTX, "comparison") if r["b"] == rec["run_id"]]
            assert len(runs.logged(CTX, "comparison")) == before + 1  # the repeated call logs nothing new
            assert len(cmp) == 1 and cmp[0]["seeds"] == [155] and len(cmp[0]["repeat_differences"]) == 1
            assert run_experiments.compare(CTX, anchor, rec["run_id"], None, seeds=cmp[0]["seeds"], verify=True,
                                           comparison_id=cmp[0]["comparison_id"])
    assert not (ROOT / "artifacts/phase2/bank" / f"{rec['run_id']}.npz").exists()
    return "a three-seed and a one-seed bank compared on seed 155, logged once, verified; nothing written"


@check("installation checks: the after-inventory must equal the before-inventory plus the lock")
def _():
    protected, lock = {"numpy": "2.5.4"}, {"interpret-core": "0.7.8"}
    before = {"numpy": "2.5.4", "catboost": "1.2.10", "graphviz": "0.21", "plotly": "7.1.0"}
    others = {"catboost": "1.2.10", "graphviz": "0.21", "plotly": "7.1.0"}
    assert not install_locks.inventory_problems(protected, before, lock, others)
    assert not install_locks.inventory_problems(protected, {**before, **lock}, lock, others, before)
    removed = {"numpy": "2.5.4", "interpret-core": "0.7.8"}  # CatBoost and its dependencies vanished
    assert install_locks.inventory_problems(protected, removed, lock, others, before)
    assert install_locks.inventory_problems(protected, {**before, "numpy": "2.6.0"}, lock, others)
    assert install_locks.inventory_problems(protected, {**before, "mystery": "1.0"}, lock, others)
    for route in install_locks.ROUTES:
        entries = install_locks.read_lock(ROOT, route)
        assert entries and all(len(h) == 64 for _, h in entries.values())
    return "a vanished earlier addition, a changed protected version and an unapproved addition are refused"


PROTO = protocol.load_protocol(ROOT / run_experiments.PROTOCOLS["P2.2"])
FAKE_ID = lambda r: f"run_{r['name']}_{recipe_hash(r)[:8]}"


def finding010_recipes():
    text = (ROOT / "analysis/phase2_timing_output.txt").read_text()
    return [c["recipe"] for c in run_experiments.strict_loads(text[text.index("\n{") + 1:])["cases"] if c["recipe"]]


@check("protocol: the wave 1 manifests hold 276 configurations, anchors first and exact, admissible, deterministic")
def _():
    ms = protocol.plan_wave(PROTO, protocol.Log([], "P2.2", None), 1, FAKE_ID)
    counts = {m["family"]: len(m["recipes"]) for m in ms}
    assert counts == {"lightgbm": 64, "random_forest": 48, "xgboost": 48, "extra_trees": 32, "hist_gb": 12,
                      "logistic": 24, "knn": 48}, counts
    first = {m["family"]: m["recipes"][0] for m in ms}
    assert recipe_hash(first["lightgbm"]) == recipe_hash(load_recipe(ROOT / ANCHOR))
    assert first["random_forest"]["params"] == {"n_estimators": 500, "min_samples_leaf": 3}
    f10 = {(r["family"], artifacts.dumps(r["params"])) for r in finding010_recipes()}
    for fam in ("xgboost", "extra_trees", "hist_gb"):
        assert (fam, artifacts.dumps(first[fam]["params"])) in f10 and first[fam]["base"] == "F2", fam
    assert first["logistic"]["params"] == {"C": 1} and first["logistic"]["preprocessing"] == {"name": "log_standard"}
    assert first["knn"]["params"] == {"metric": "euclidean", "n_neighbors": 60, "weights": "distance"}
    for m in ms:
        hashes = [recipe_hash(r) for r in m["recipes"]]
        assert len(set(hashes)) == len(hashes) and m["run_ids"] == [FAKE_ID(r) for r in m["recipes"]]
        it = {"lightgbm": "n_estimators", "xgboost": "n_estimators", "hist_gb": "max_iter"}.get(m["family"])
        for r in m["recipes"][1:]:
            q = r["params"]
            if it:
                assert 5 <= q["learning_rate"] * q[it] <= 100, q
            if m["family"] == "lightgbm":
                assert ("subsample_freq" in q) == (q["subsample"] < 1), q
    log_ = {m["family"]: m for m in ms}["logistic"]["recipes"]
    kinds = sorted((r["base"], r["preprocessing"]["name"]) for r in log_)
    assert kinds.count(("F1", "log_standard")) == 6 and kinds.count(("F2", "log_standard")) == 6 \
        and kinds.count(("F1", "spline")) == 12
    pairs = sorted((r["params"]["C"], r["preprocessing"]["n_knots"]) for r in log_ if r["preprocessing"]["name"] == "spline")
    assert pairs == sorted([(0.001, 3), (0.001, 5), (0.01, 4), (0.01, 7), (0.1, 5), (0.1, 10), (1, 7), (1, 3), (10, 10),
                            (10, 4), (100, 3), (100, 5)])
    knn = {m["family"]: m for m in ms}["knn"]["recipes"]
    assert sorted((r["params"]["n_neighbors"], r["params"]["weights"], r["preprocessing"]["alpha"]) for r in knn) == \
        sorted((k, w, a) for k in (15, 30, 60, 120, 240, 480) for w in ("uniform", "distance") for a in (0.25, 0.5, 1, 2))
    again = protocol.plan_wave(PROTO, protocol.Log([], "P2.2", None), 1, FAKE_ID)
    assert artifacts.dumps(again) == artifacts.dumps(ms)
    # The comparison key: equivalent numbers and omitted factory defaults are the same computation.
    fam = {f["family"]: f for f in PROTO["families"]}
    kl, kr = protocol.search_key(fam["logistic"]), protocol.search_key(fam["random_forest"])
    mk = lambda f, v: protocol.make_recipe(fam[f]["template"], f, "k", v)
    assert kl(mk("logistic", {"C": 100})) == kl(mk("logistic", {"C": 100.0})) != kl(mk("logistic", {"C": 10.0}))
    assert kr(first["random_forest"]) == kr(mk("random_forest", {"n_estimators": 500, "min_samples_leaf": 3,
                                                                 "max_features": "sqrt", "max_depth": None,
                                                                 "max_samples": 1.0}))
    assert kr(first["random_forest"]) != kr(mk("random_forest", {"n_estimators": 500, "min_samples_leaf": 3,
                                                                 "max_features": 0.3}))
    for m in ms:
        k = protocol.search_key(fam[m["family"]])
        assert len({k(r) for r in m["recipes"]}) == len(m["recipes"]), m["family"]
    raises(lambda: protocol.make_recipe(fam["random_forest"]["template"], "random_forest", "k",
                                        {"n_estimators": 500, "max_samples": 1}), ValueError, "out of domain")
    return "64, 48, 48, 32, 12, 24, 48; anchors as declared (Finding 010 recipes exact); deterministic; " \
           "equivalent configurations share one key; an integer fraction refused"


@check("refinement: typed local moves, zero steps, sqrt and None, active bounds, tree variants, deduplication")
def _():
    rng = np.random.default_rng(0)
    log_t = {"type": "log", "min": 1, "max": 100, "integer": False, "zero_step": 0.1, "default": 0}
    assert refine.move(0, log_t, rng) == 0.1
    assert all(refine.move(0.12, log_t, np.random.default_rng(i)) in (0,) or refine.move(0.12, log_t, np.random.default_rng(i)) >= 1
               for i in range(20))  # below the step lands on zero, otherwise clipped up to the active minimum
    ordinal = {"type": "ordinal", "values": ["sqrt", 0.3, 0.5, None], "default": "sqrt"}
    assert refine.move("sqrt", ordinal, rng) == 0.3 and refine.move(None, ordinal, rng) == 0.5
    assert {refine.move(0.3, ordinal, np.random.default_rng(i)) for i in range(20)} == {"sqrt", 0.5}
    integer = {"type": "log", "min": 7, "max": 63, "integer": True, "zero_step": None, "default": 31}
    assert all(isinstance(refine.move(15, integer, np.random.default_rng(i)), int) and
               7 <= refine.move(15, integer, np.random.default_rng(i)) <= 63 for i in range(50))
    fam = next(f for f in PROTO["families"] if f["family"] == "lightgbm")
    parent = protocol.anchor_recipe(PROTO, fam)
    other = protocol.make_recipe(fam["template"], "lightgbm", "other", dict(parent["params"], num_leaves=31))
    space = fam["space"]
    spec = {"schema": refine.REFINE_SCHEMA, "name": "probe_r", "family": "lightgbm", "parents": [parent, other],
            "local_per_parent": [5, 4], "trees": "n_estimators", "broad": 3,
            "broad_template": protocol.make_recipe(fam["template"], "lightgbm", "t", searches.apply_derived(
                {n: v[0] for n, v in space.items()}, fam["derived"])),
            "broad_space": space, "types": protocol.local_types(fam, space), "derived": fam["derived"],
            "admissible": fam["admissible"], "exclude_recipe_sha256": [], "sampler_seed": 5, "max_dimensions": 3}
    got, rej = refine.propose(spec)
    assert len(got) == 12 and [r["params"]["n_estimators"] for r in got[:2]] == [400, 900], [r["params"] for r in got[:2]]
    assert all({k: v for k, v in r["params"].items() if k != "n_estimators"} ==
               {k: v for k, v in parent["params"].items() if k != "n_estimators"} for r in got[:2])
    for r in got:
        q = r["params"]
        assert 5 <= q["learning_rate"] * q["n_estimators"] <= 100 and 7 <= q["num_leaves"] <= 63 \
            and 300 <= q["n_estimators"] <= 1600 and ("subsample_freq" in q) == (q.get("subsample", 1.0) < 1), q
    assert len({recipe_hash(r) for r in got} | {recipe_hash(parent), recipe_hash(other)}) == 14
    again, _ = refine.propose(spec)
    assert artifacts.dumps(again) == artifacts.dumps(got)
    excl, _ = refine.propose(dict(spec, exclude_recipe_sha256=[recipe_hash(r) for r in got]))
    assert not {recipe_hash(r) for r in excl} & {recipe_hash(r) for r in got} and len(excl) == 12
    lfam = next(f for f in PROTO["families"] if f["family"] == "logistic")
    lp = protocol.anchor_recipe(PROTO, lfam)
    assert refine.applicable(lp, sorted(lfam["types"])) == ["C"]
    one, _ = refine.propose(dict(spec, parents=[parent], local_per_parent=[9]))
    assert len(one) == 12
    # An exhausted local neighborhood (review round 1, F1): the Extra Trees wave 1 recipe with leaf 1, "sqrt" and no
    # depth limit has six new neighbors; its other three slots become broad draws.
    et = next(f for f in PROTO["families"] if f["family"] == "extra_trees")
    ket = protocol.search_key(et)
    w1 = protocol.initial_recipes(PROTO, et)
    small = next(r for r in w1 if r["params"] == {"n_estimators": 500, "min_samples_leaf": 1, "max_features": "sqrt",
                                                  "max_depth": None})
    tmpl = protocol.make_recipe(et["space_template"], "extra_trees", "t", dict(et["fixed"], **{
        n: v[0] for n, v in et["space"].items()}))
    espec = dict(spec, family="extra_trees", parents=[small], local_per_parent=[9], trees=None, broad=3,
                 broad_template=tmpl, broad_space=et["space"], types=protocol.local_types(et, et["space"]), derived=[],
                 admissible=[], exclude_recipe_sha256=sorted({ket(r) for r in w1}))
    got_et, info = refine.propose(espec, key=ket)
    assert len(got_et) == 12 and info["local_shortfall"] == [3] and info["termination"] == "requested_count", info
    # Everything reachable already seen but two broad configurations: a smaller batch, exhaustion proven.
    allc = [protocol.make_recipe(et["space_template"], "extra_trees", "a", dict(et["fixed"], **dict(zip(sorted(
        et["space"]), c)))) for c in itertools.product(*(et["space"][n] for n in sorted(et["space"])))]
    keep = {ket(allc[0]), ket(allc[1])} - {ket(small)}
    gone = sorted({ket(r) for r in allc + got_et + w1} - keep)
    got2, info2 = refine.propose(dict(espec, exclude_recipe_sha256=gone), key=ket)
    assert len(got2) == len(keep) and info2["termination"] == "enumerated_exhaustion", (len(got2), info2)
    with mock.patch.object(refine, "SLOT_ATTEMPTS", 2), mock.patch.object(refine, "ENUMERATE_LIMIT", 10):
        raises(lambda: refine.propose(dict(espec, exclude_recipe_sha256=gone), key=ket), ValueError,
               "not proven exhausted")
    return f"12 proposals, tree variants 400 and 900 first, rejections {rej}; deterministic; exclusions honoured; " \
           f"an exhausted neighborhood passes its slots to broad draws; exhaustion proven or refused"


@check("allocation: the four routes at their margins, positive repeats, means-only proximity, the counter")
def _():
    inc, top = [0.8000] * 3, [0.8300] * 3
    R = lambda s, b=((0.83,) * 3, (0.83,) * 3): allocation.routes({"standalone": list(s), "blends": [list(x) for x in b]},
                                                               inc, top, [0.83] * 3, [0.83] * 3)
    assert R([0.8031] * 3) == ["clear_standalone"]
    assert R([0.8100, 0.8100, 0.7999]) == []  # a mean gain of 0.0066 with one negative repeat
    near = allocation.routes({"standalone": [0.8265] * 3, "blends": [[0.83] * 3] * 2}, [0.8250] * 3, top, [0.83] * 3,
                             [0.83] * 3)
    assert near == ["smaller_useful"], near
    close = allocation.routes({"standalone": [0.8290] * 3, "blends": [[0.83] * 3] * 2}, [0.8284] * 3, top, [0.83] * 3,
                              [0.83] * 3)
    assert close == ["competitive_refinement"], close
    assert allocation.routes({"standalone": [0.8290, 0.8310, 0.8270], "blends": [[0.83] * 3] * 2}, [0.8284] * 3,
                             [0.8310, 0.8290, 0.8300], [0.83] * 3, [0.83] * 3) == []  # repeat 3 below the incumbent
    assert R([0.79] * 3, ([0.8306] * 3, [0.83] * 3)) == ["ensemble_contribution"]
    assert R([0.79] * 3, ([0.8306, 0.8306, 0.8300], [0.83] * 3)) == []
    assert allocation.routes({"standalone": [0.79] * 3, "blends": [[0.8306] * 3] * 2}, inc, top, [0.83] * 3,
                             [0.8303] * 3) == []  # does not beat the previous best blend by 0.0005
    n = allocation.next_allowance
    assert (n(None, True, True), n(None, False, True), n(2, False, False), n(1, False, False), n(1, True, False)) == \
        (2, 0, 1, 0, 2)
    return "clear, smaller, competitive and ensemble routes at their margins; the counter 2, 0, 1, 0, 2"


def synthetic_log(score, blend):
    """A log of wave 1 with synthetic scores: score(family, recipe) and blend(family, recipe) give the standalone and
    the better contribution blend means (equal on every repeat); the other blend is H's mean."""
    ms = protocol.plan_wave(PROTO, protocol.Log([], "P2.2", None), 1, FAKE_ID)
    files, records = {}, []
    h = ms[0]["run_ids"][0]
    for m in ms:
        bid = f"{m['family']}_w1"
        files[bid] = {"recipes": m["recipes"]}
        records.append({"record": "batch", "task_id": "P2.2", "batch_id": bid, "family": m["family"], "wave": 1,
                        "kind": "initial", "file": bid, "run_ids": m["run_ids"], "state": m["state"],
                        "sources": m["sources"]})
        for r, rid in zip(m["recipes"], m["run_ids"]):
            v = 0.8314739289 if rid == h else score(m["family"], r)
            records.append({"record": "run", "run_id": rid, "recipe_id": recipe_id(r), "selection_eligible": True,
                            "split_seeds": list(SPLIT_SEEDS), "n_rows": 3863, "bank_sha256": "bank_" + rid,
                            "estimator_seeds": [0], "mean_fold_auc_unrounded": v, "repeat_mean_auc_unrounded": [v] * 3})
            b = 0.8314739289 if rid == h else blend(m["family"], r)
            for w, val in ((0.1, b), (0.25, 0.8314739289)):
                records.append({"record": "contribution", "task_id": "P2.2", "run_id": rid, "h_run_id": h, "weight": w,
                                "bank_sha256": "bank_" + rid, "h_sha256": "bank_" + h, "identity": rid == h,
                                "mean_fold_auc_unrounded": val, "repeat_mean_auc_unrounded": [val] * 3})
    log = protocol.Log(records, "P2.2", files.__getitem__)
    for b in log.batches:
        res = protocol.judge(b, log)
        records.append({"record": "allocation", "task_id": "P2.2", "batch_id": b["batch_id"], **res})
    return protocol.Log(records, "P2.2", files.__getitem__), records


def _score(family, r):
    q, j = r["params"], int(recipe_hash(r)[:4], 16) / 65536 * 1e-5
    return {"lightgbm": 0.8320 + 1e-5 * q.get("num_leaves", 0) - 1e-6 * q.get("min_child_samples", 0),
            "random_forest": 0.8270, "xgboost": 0.8300 + j, "extra_trees": 0.8200, "hist_gb": 0.8240 + j,
            "logistic": 0.7850, "knn": 0.7700}[family] + (0.0 if family == "lightgbm" else j) \
        - (0.004 if family in ("xgboost", "hist_gb") and r["name"].endswith("_anchor") else 0.0)


def _blend(family, r):
    if r["name"].endswith("_anchor"):
        return 0.8314739289
    return {"random_forest": 0.8323, "logistic": 0.8321, "hist_gb": 0.83155}.get(family, 0.8314739289) \
        + int(recipe_hash(r)[:4], 16) / 65536 * 1e-6


@check("protocol: judging and wave 2 planning on a synthetic log: routes, counter, refinement families, local split")
def _():
    log, records = synthetic_log(_score, _blend)
    alloc = {r["batch_id"].split("_w")[0]: r for r in records if r["record"] == "allocation"}
    after = {f: a["allowance_after"] for f, a in alloc.items()}
    assert after == {"lightgbm": 2, "random_forest": 2, "xgboost": 2, "extra_trees": 0, "hist_gb": 2, "logistic": 2,
                     "knn": 0}, after
    h = log.batches[0]["run_ids"][0]
    assert h not in alloc["lightgbm"]["routes"]  # the anchor judges its batch, it is not judged
    assert protocol.refinement_families(PROTO, log, PROTO["families"]) == ["lightgbm", "xgboost", "random_forest",
                                                                           "logistic"]
    w2 = protocol.plan_wave(PROTO, log, 2, FAKE_ID)
    kinds = {m["family"]: (m["kind"], len(m["recipes"])) for m in w2}
    assert kinds == {"lightgbm": ("refinement", 12), "random_forest": ("refinement", 12), "xgboost": ("refinement", 12),
                     "hist_gb": ("expansion", 12), "logistic": ("refinement", 12)}, kinds
    lgb = next(m for m in w2 if m["family"] == "lightgbm")
    assert lgb["sources"]["local_per_parent"] == [5, 4] and lgb["sources"]["broad"] == 3
    inc = lgb["state"]["incumbent"]["run_id"]
    t = log.manifest_recipe(inc)["params"]["n_estimators"]
    assert [r["params"]["n_estimators"] for r in lgb["recipes"][:2]] == [int(2 * t / 3 + 0.5), int(1.5 * t + 0.5)] or \
        lgb["sources"]["rejected"]["inadmissible"] + lgb["sources"]["rejected"]["out_of_bounds"] > 0
    assert lgb["state"]["strongest"]["run_id"] == inc and lgb["state"]["allowance_before"] == 2
    prior = {r for b in log.batches if b["family"] == "lightgbm" for r in log.batches[0]["run_ids"]}
    assert not {FAKE_ID(r) for r in lgb["recipes"]} & prior
    assert artifacts.dumps(protocol.plan_wave(PROTO, log, 2, FAKE_ID)) == artifacts.dumps(w2)
    # Range extension: the LightGBM incumbent has 63 leaves (the top of its list): 63 x 63 / 31 clipped to 127.
    fam = PROTO["families"][0]
    ext = protocol.new_extensions(fam, fam["space"], log.manifest_recipe(inc), [])
    assert {"parameter": "num_leaves", "edge": 63, "value": 127, "hard_bounds": fam["hard_bounds"]["num_leaves"]} in ext
    assert not protocol.new_extensions(fam, fam["space"], log.manifest_recipe(inc), ext)  # once per parameter
    rf = next(f for f in PROTO["families"] if f["family"] == "random_forest")
    assert all(e["parameter"] not in ("max_features", "max_depth") for e in protocol.new_extensions(
        rf, rf["space"], dict(log.manifest_recipe(inc), params={"max_features": 1.0, "max_depth": None}), []))
    # Ties (review round 1, F4): equal contribution blends resolve by the challenger's canonical recipe hash.
    tied, _ = synthetic_log(_score, lambda f, r: 0.8314739289 if r["name"].endswith("_anchor") else
                            {"random_forest": 0.8323, "logistic": 0.8323, "hist_gb": 0.8323}.get(f, 0.8314739289))
    rows = {}
    h_t = tied.batches[0]["run_ids"][0]
    for f in ("random_forest", "logistic", "hist_gb"):
        cs = tied.family_contributions(f, 2, h_t)
        pick = protocol.best_blend(cs)
        top = [c for c in cs if not c.get("identity") and c["mean_fold_auc_unrounded"] == pick["mean_fold_auc_unrounded"]]
        assert pick["run_id"] == min(top, key=lambda c: (c["recipe_sha256"], c["weight"]))["run_id"]
        rows[f] = tied.runs[pick["run_id"]]["recipe_sha256"]
    order = sorted(rows, key=lambda f: (rows[f], f))
    assert protocol.refinement_families(PROTO, tied, PROTO["families"]) == ["lightgbm", "xgboost"] + order[:2], order
    b = next(x for x in tied.batches if x["family"] == "random_forest")
    res = protocol.judge(b, tied)
    ens = [w for w in protocol.witnesses(b, res, tied) if w[0] == "ensemble_contribution"]
    met = [r for r, v in res["routes"].items() if "ensemble_contribution" in v]
    assert ens and ens[0][1][1] == min(met, key=lambda r: tied.runs[r]["recipe_sha256"])
    return "allowances 2, 2, 2, 0, 2, 2, 0; refinement lightgbm, xgboost, random_forest, logistic; " \
           "local 5/4 + 3 broad; num_leaves extends to 127 once; tied contributions resolve by recipe hash"


PROTO23 = protocol.load_protocol(ROOT / run_experiments.PROTOCOLS["P2.3"])


def synthetic_log23(score, blend, prior_score=0.8354212897300688):
    """P2.3's wave 1 on synthetic scores, as synthetic_log does for P2.2, with the protocol's prior run (G's
    external member) logged as a complete P2.8 run with the given score."""
    prior = PROTO23["prior_strongest"]
    pv = prior_score
    base = [{"record": "run", "run_id": prior["run_id"], "task_id": "P2.8", "recipe_id": recipe_id(prior["recipe"]),
             "selection_eligible": True, "split_seeds": list(SPLIT_SEEDS), "n_rows": 3863,
             "bank_sha256": prior["bank_sha256"], "estimator_seeds": [0], "mean_fold_auc_unrounded": pv,
             "repeat_mean_auc_unrounded": [pv] * 3}]
    ms = protocol.plan_wave(PROTO23, protocol.Log(base, "P2.3", None, prior), 1, FAKE_ID)
    files, records = {}, list(base)
    h = ms[0]["run_ids"][0]
    for m in ms:
        bid = f"{m['family']}_w1"
        files[bid] = {"recipes": m["recipes"]}
        records.append({"record": "batch", "task_id": "P2.3", "batch_id": bid, "family": m["family"], "wave": 1,
                        "kind": "initial", "file": bid, "run_ids": m["run_ids"], "state": m["state"],
                        "sources": m["sources"]})
        for r, rid in zip(m["recipes"], m["run_ids"]):
            v = 0.8314739289 if rid == h else score(m["family"], r)
            records.append({"record": "run", "run_id": rid, "task_id": "P2.3", "recipe_id": recipe_id(r),
                            "selection_eligible": True, "split_seeds": list(SPLIT_SEEDS), "n_rows": 3863,
                            "bank_sha256": "bank_" + rid, "estimator_seeds": [0], "mean_fold_auc_unrounded": v,
                            "repeat_mean_auc_unrounded": [v] * 3})
            b = 0.8314739289 if rid == h else blend(m["family"], r)
            for w, val in ((0.1, b), (0.25, 0.8314739289)):
                records.append({"record": "contribution", "task_id": "P2.3", "run_id": rid, "h_run_id": h, "weight": w,
                                "bank_sha256": "bank_" + rid, "h_sha256": "bank_" + h, "identity": rid == h,
                                "mean_fold_auc_unrounded": val, "repeat_mean_auc_unrounded": [val] * 3})
    log = protocol.Log(records, "P2.3", files.__getitem__, prior)
    for b in log.batches:
        records.append({"record": "allocation", "task_id": "P2.3", "batch_id": b["batch_id"], **protocol.judge(b, log)})
    return protocol.Log(records, "P2.3", files.__getitem__, prior), records, ms


@check("P2.3 protocol: reference family, prior run in G, wave 1 manifests (1, 48, 12), candidate 31, one refinement "
       "family, inherited off-list values (plan section 17)")
def _():
    names = [f["family"] for f in PROTO23["families"]]
    assert names == ["lightgbm", "catboost", "ebm"] and PROTO23["refinement"]["families"] == 1
    p22_lgb = next(f for f in PROTO["families"] if f["family"] == "lightgbm")
    ref = PROTO23["families"][0]
    assert ref["anchor"] == p22_lgb["anchor"] and not ref["space"] and ref["initial"] == ref["ceiling"] == 1
    init = {f["family"]: protocol.initial_recipes(PROTO23, f) for f in PROTO23["families"]}
    assert [len(init[n]) for n in names] == [1, 48, 12]
    cb, eb = init["catboost"], init["ebm"]
    assert cb[0]["name"] == "p23_catboost_anchor" and cb[0]["params"] == PROTO23["families"][1]["anchor"]
    c31 = {"depth": 6, "iterations": 1944, "learning_rate": 0.009535096982557112, "l2_leaf_reg": 4.757135294733201,
           "random_strength": 0.0, "rsm": 0.8880009208730744, "bootstrap_type": "Bayesian",
           "bagging_temperature": 1.793473187168845, "boosting_type": "Plain"}
    assert cb[1]["params"] == c31
    assert sum(r["params"]["boosting_type"] == "Ordered" for r in cb) == 24
    assert all(5 <= r["params"]["learning_rate"] * r["params"]["iterations"] <= 100 for r in cb)
    settings = {(r["params"]["max_rounds"], r["params"]["learning_rate"]) for r in eb}
    assert len(settings) == 6 and all({x["params"]["interactions"] for x in eb if
                                       (x["params"]["max_rounds"], x["params"]["learning_rate"]) == st} ==
                                      {"none", "declared3"} for st in settings)
    raises(lambda: run_experiments.load_protocol(CTX, "P2.8"), ValueError, "no wave protocol")
    with mock.patch.dict(run_experiments.PROTOCOLS, {"P2.3": run_experiments.PROTOCOLS["P2.2"]}):
        raises(lambda: run_experiments.load_protocol(CTX, "P2.3"), ValueError, "belongs to task P2.2")
    real = run_experiments.load_log(CTX, "P2.3")  # the prior binds to the logged P2.8 record
    assert real.prior == PROTO23["prior_strongest"]["run_id"]
    for bad in ({"bank_sha256": "0" * 64}, {"task_id": "P2.2"}, {"recipe": dict(PROTO23["prior_strongest"]["recipe"],
                                                                               blocks=[])}):
        raises(lambda bad=bad: protocol.Log(artifacts.read_experiments(ROOT / runs.LOG_FILE), "P2.3",
                                            lambda rel: run_experiments.read_manifest(CTX, rel),
                                            dict(PROTO23["prior_strongest"], **bad)), ValueError, "prior_strongest")

    # Synthetic wave 1: candidate 31 leads CatBoost by +0.004 over the anchor; EBM's best pair is +0.0015 over its
    # anchor, 0.0054 below the prior run but only 0.0015 below H.
    def score(family, r):
        j = int(recipe_hash(r)[:4], 16) / 65536 * 1e-5
        if family == "catboost":
            return 0.8320 if r["params"].get("rsm") else (0.8280 if r["name"].endswith("_anchor") else 0.8270 + j)
        return 0.8300 if r["params"]["max_rounds"] == 1500 and r["params"]["learning_rate"] == 0.06 else 0.8285 + j
    log, records, ms = synthetic_log23(score, lambda f, r: 0.8314739289)
    alloc = {r["batch_id"].split("_w")[0]: r for r in records if r["record"] == "allocation"}
    assert alloc["lightgbm"]["routes"] == {} and alloc["lightgbm"]["allowance_after"] == 0
    st = alloc["catboost"]["resolved_state"]["strongest"]
    assert st["run_id"] == PROTO23["prior_strongest"]["run_id"]  # G: the better of H and the prior run
    c31_id = ms[1]["run_ids"][1]
    assert alloc["catboost"]["routes"][c31_id] == ["clear_standalone", "smaller_useful"]  # 0.0034 below G
    assert alloc["ebm"]["routes"] and all(v == [] for v in alloc["ebm"]["routes"].values())  # 0.0054 below G
    assert alloc["ebm"]["allowance_after"] == 0 and alloc["catboost"]["allowance_after"] == 2
    _, records_h, _ = synthetic_log23(score, lambda f, r: 0.8314739289, prior_score=0.8300)
    alloc_h = {r["batch_id"].split("_w")[0]: r for r in records_h if r["record"] == "allocation"}
    assert alloc_h["catboost"]["resolved_state"]["strongest"]["run_id"] == ms[0]["run_ids"][0]  # weaker prior: G = H
    assert alloc_h["ebm"]["allowance_after"] == 2 and any(alloc_h["ebm"]["routes"].values())  # within 0.005 of H
    assert protocol.refinement_families(PROTO23, log, PROTO23["families"]) == ["catboost"]
    w2 = protocol.plan_wave(PROTO23, log, 2, FAKE_ID)
    assert [(m["family"], m["kind"], len(m["recipes"])) for m in w2] == [("catboost", "refinement", 12)]
    m = w2[0]
    assert m["state"]["incumbent"]["run_id"] == c31_id and m["sources"]["local_per_parent"] == [9]
    local = m["recipes"][:9]
    assert local[0]["params"]["iterations"] == 1296 and all(r["params"]["iterations"] != 2916 for r in local)
    kept = [r for r in local if r["params"]["learning_rate"] == c31["learning_rate"]]
    moved = [r for r in local if r["params"]["learning_rate"] != c31["learning_rate"]]
    assert kept and all(0.015 <= r["params"]["learning_rate"] <= 0.1 for r in moved)
    assert all(r["params"]["bootstrap_type"] == "Bayesian" and r["params"]["rsm"] == c31["rsm"] for r in local)
    assert artifacts.dumps(protocol.plan_wave(PROTO23, log, 2, FAKE_ID)) == artifacts.dumps(w2)
    return (f"the reference batch judged with no configuration, allowance 0; G the prior run, or H when the prior "
            f"is weaker, which changes EBM's smaller-useful route; one refinement family; {len(kept)} local proposals keep its "
            f"learning rate 0.0095, {len(moved)} move it inside [0.015, 0.1]; its 1.5T variant (2916) dropped")


def extend_log23(records, files, manifests, wave, score):
    """Append synthetic batch, run and contribution records for planned later-wave manifests (states complete)."""
    h = next(r for r in records if r["record"] == "batch" and r["family"] == "lightgbm")["run_ids"][0]
    for m in manifests:
        bid = f"{m['family']}_w{wave}"
        files[bid] = {"recipes": m["recipes"]}
        records.append({"record": "batch", "task_id": "P2.3", "batch_id": bid, "family": m["family"], "wave": wave,
                        "kind": m["kind"], "file": bid, "run_ids": m["run_ids"], "state": m["state"],
                        "sources": m["sources"]})
        for r, rid in zip(m["recipes"], m["run_ids"]):
            v = score(m["family"], r)
            records.append({"record": "run", "run_id": rid, "task_id": "P2.3", "recipe_id": recipe_id(r),
                            "selection_eligible": True, "split_seeds": list(SPLIT_SEEDS), "n_rows": 3863,
                            "bank_sha256": "bank_" + rid, "estimator_seeds": [0], "mean_fold_auc_unrounded": v,
                            "repeat_mean_auc_unrounded": [v] * 3})
            for w in (0.1, 0.25):
                records.append({"record": "contribution", "task_id": "P2.3", "run_id": rid, "h_run_id": h, "weight": w,
                                "bank_sha256": "bank_" + rid, "h_sha256": "bank_" + h, "identity": False,
                                "mean_fold_auc_unrounded": 0.8314739289,
                                "repeat_mean_auc_unrounded": [0.8314739289] * 3})


@check("P2.3 protocol, other paths: EBM takes the refinement slot and CatBoost expands; a P2.3 run overtakes the "
       "prior member of G; current-wave results leave the wave's comparators unchanged; EBM's exhausted space gives "
       "a smaller batch; a corrupted prior member is refused (plan section 17)")
def _():
    prior = PROTO23["prior_strongest"]
    j = lambda r: int(recipe_hash(r)[:4], 16) / 65536 * 1e-5  # noqa: E731

    # (a) EBM's incumbent (0.8310) beats CatBoost's (0.8300): EBM refines, CatBoost gets a broad expansion of 24.
    def score_a(family, r):
        if family == "catboost":
            return 0.8300 if r["params"].get("rsm") else 0.8265 + j(r)
        return 0.8310 if (r["params"]["max_rounds"], r["params"]["learning_rate"]) == (1500, 0.01) else 0.8270 + j(r)
    log, _, _ = synthetic_log23(score_a, lambda f, r: 0.8314739289)
    assert protocol.refinement_families(PROTO23, log, PROTO23["families"]) == ["ebm"]
    w2 = {m["family"]: m for m in protocol.plan_wave(PROTO23, log, 2, FAKE_ID)}
    assert (w2["ebm"]["kind"], len(w2["ebm"]["recipes"])) == ("refinement", 12)
    assert (w2["catboost"]["kind"], len(w2["catboost"]["recipes"])) == ("expansion", 24) and "lightgbm" not in w2
    assert w2["catboost"]["state"]["strongest"]["run_id"] == prior["run_id"]

    # (b) Candidate 31 at 0.8360 overtakes the prior member (0.8354); EBM's best (+0.004) is interior (700 rounds,
    # 0.03), so its expansion has no range extension.
    def score_b(family, r):
        if family == "catboost":
            return 0.8360 if r["params"].get("rsm") else 0.8265 + j(r)
        q = r["params"]
        return 0.8310 if (q["max_rounds"], q["learning_rate"], q["interactions"]) == (700, 0.03, "declared3") \
            else 0.8270 + j(r)
    log, records, ms = synthetic_log23(score_b, lambda f, r: 0.8314739289)
    c31_id = ms[1]["run_ids"][1]
    w2 = protocol.plan_wave(PROTO23, log, 2, FAKE_ID)
    by = {m["family"]: m for m in w2}
    assert by["catboost"]["kind"] == "refinement" and by["catboost"]["state"]["strongest"]["run_id"] == c31_id
    e = by["ebm"]
    assert e["kind"] == "expansion" and len(e["recipes"]) == 4 and not e["sources"]["extensions"] \
        and e["sources"]["rejected"]["termination"] == "enumerated_exhaustion"
    assert {(r["params"]["max_rounds"], r["params"]["learning_rate"]) for r in e["recipes"]} == {(700, 0.06), (1500, 0.03)}

    # (c) A wave 2 EBM run far above everything changes nothing in the wave 2 CatBoost batch's judgment.
    files = {b["batch_id"]: {"recipes": [log.manifest_recipe(r) for r in b["run_ids"]]} for b in log.batches}
    recs = list(records)
    extend_log23(recs, files, w2, 2, lambda f, r: 0.8270 + j(r))
    before = protocol.judge(next(b for b in protocol.Log(recs, "P2.3", files.__getitem__, prior).batches
                                 if b["batch_id"] == "catboost_w2"), protocol.Log(recs, "P2.3", files.__getitem__, prior))
    hot = [dict(r, mean_fold_auc_unrounded=0.90, repeat_mean_auc_unrounded=[0.90] * 3)
           if r["record"] == "run" and r["run_id"] == e["run_ids"][0] else r for r in recs]
    log_hot = protocol.Log(hot, "P2.3", files.__getitem__, prior)
    after = protocol.judge(next(b for b in log_hot.batches if b["batch_id"] == "catboost_w2"), log_hot)
    assert artifacts.dumps(before) == artifacts.dumps(after)
    assert protocol.strongest(log_hot, 3, log_hot.batches[0]["run_ids"][0])["run_id"] == e["run_ids"][0]  # next wave

    # (d) The prior member's bank must match the protocol, and its record must match its bank.
    rx = run_experiments
    rx.validate_prior(CTX, PROTO23)
    raises(lambda: rx.validate_prior(CTX, dict(PROTO23, prior_strongest=dict(prior, bank_sha256="0" * 64))),
           ValueError, "bank hash")
    raises(lambda: rx.validate_prior(CTX, dict(PROTO23, prior_strongest=dict(prior, task_id="P2.2"))),
           ValueError, "task")
    real = runs.logged

    def forged(ctx, record="run"):
        return [dict(r, mean_fold_auc_unrounded=r["mean_fold_auc_unrounded"] + 0.01)
                if record == "run" and r["run_id"] == prior["run_id"] else r for r in real(ctx, record)]
    with mock.patch.object(runs, "logged", forged):
        raises(lambda: rx.validate_prior(CTX, PROTO23), ValueError, "fails its validation")
    return ("EBM refines while CatBoost expands by 24; candidate 31 at 0.8360 becomes G for wave 2; EBM's interior "
            "incumbent leaves 4 admissible configurations, enumerated; a wave 2 score of 0.90 leaves the wave 2 "
            "judgment unchanged and becomes G for wave 3; a wrong bank hash, task or forged score is refused")


@check("P2.3 handoff: unstarted last-wave batches after the cutoff are reported as omitted; before the cutoff, or "
       "with a partial batch, the handoff is refused (plan section 17.6)")
def _():
    rx = run_experiments

    def score(family, r):
        j = int(recipe_hash(r)[:4], 16) / 65536 * 1e-5
        if family == "catboost":
            return 0.8320 if r["params"].get("rsm") else (0.8280 if r["name"].endswith("_anchor") else 0.8270 + j)
        return 0.8300 if r["params"]["max_rounds"] == 1500 and r["params"]["learning_rate"] == 0.06 else 0.8285 + j
    log, records, _ = synthetic_log23(score, lambda f, r: 0.8314739289)
    files = {b["batch_id"]: {"recipes": [log.manifest_recipe(r) for r in b["run_ids"]]} for b in log.batches}
    w2 = protocol.plan_wave(PROTO23, log, 2, rx.run_id_of(CTX, list(SPLIT_SEEDS), MANIFEST))
    recs = list(records)
    for m in w2:  # wave 2 planned, nothing of it started
        files[f"{m['family']}_w2"] = {"recipes": m["recipes"]}
        recs.append({"record": "batch", "task_id": "P2.3", "batch_id": f"{m['family']}_w2", "family": m["family"],
                     "wave": 2, "kind": m["kind"], "file": f"{m['family']}_w2", "run_ids": m["run_ids"],
                     "state": m["state"], "sources": m["sources"]})
    log2 = protocol.Log(recs, "P2.3", files.__getitem__, PROTO23["prior_strongest"])
    seeds = list(SPLIT_SEEDS)
    past, future = dict(PROTO23, cutoff_utc="2000-01-01T00:00:00+00:00"), dict(PROTO23, cutoff_utc="2100-01-01T00:00:00+00:00")
    assert p23_summary.handoff_gate(log2, rx.completion_problems(CTX, past, log2, seeds, MANIFEST)) == ["catboost_w2"]
    raises(lambda: p23_summary.handoff_gate(log2, rx.completion_problems(CTX, future, log2, seeds, MANIFEST)),
           SystemExit, "not complete")
    partial = recs + [{"record": "batch_start", "task_id": "P2.3", "batch_id": "catboost_w2"}]
    log3 = protocol.Log(partial, "P2.3", files.__getitem__, PROTO23["prior_strongest"])
    raises(lambda: p23_summary.handoff_gate(log3, rx.completion_problems(CTX, past, log3, seeds, MANIFEST)),
           SystemExit, "partial")
    raises(lambda: p23_summary.handoff_gate(log, rx.completion_problems(CTX, future, log, seeds, MANIFEST)),
           SystemExit, "still due")
    return ("after the cutoff an unstarted wave 2 batch is reported as omitted; before it, a started (partial) batch "
            "or a wave still due refuses the handoff")


@check("P2.4 blocks: date_te3 and date_te100 by hand, one shared inner split for every smoothing value, at most one "
       "target encoding per recipe (plan section 18)")
def _():
    dates = ["A", "A", "A", "B", "B", "C", "C", "C", "A", "B", "D", "D"]
    y = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
    groups = np.array([0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
    rows = frame(dates)
    splits = {}
    for m in (3, 10, 30, 100):
        enc = DateTargetEncoder(m)
        out = enc.fit_transform(rows, y, groups, FitContext(155, 0))[f"date_te{m}"].to_numpy()
        splits[m] = [(list(a), list(b)) for a, b in enc.splits_]
        expected = np.full(len(y), np.nan)
        for tr, te in enc.splits_:
            prior = y[tr].mean()
            for i in te:
                same = [j for j in tr if dates[j] == dates[i]]
                expected[i] = (y[same].sum() + m * prior) / (len(same) + m)
        assert np.allclose(out, expected, rtol=0, atol=1e-12), (m, out, expected)
        assert np.allclose(enc.transform(frame(["Z"]))[f"date_te{m}"], y.mean(), rtol=0, atol=1e-12)
    assert all(splits[m] == splits[10] for m in (3, 30, 100))
    for blocks in (["date_te3", "date_te100"], ["date_te10", "date_te3"], ["date_te30", "date_te100"]):
        raises(lambda b=blocks: recipe(blocks=b), ValueError, "at most one date target encoding")
    r = recipe(blocks=["date_count", "date_te3"])
    assert feature_columns(r)[-2:] == ["date_count", "date_te3"]
    return "m = 3, 10, 30, 100 by hand, unseen dates get the fitting prior, identical inner splits; two encodings refused"


@check("P2.4 preprocessing: spline_calendar reproduces spline on F1 and keeps calendar fields linear on F2; "
       "knn_blocks_v2 standard reproduces knn_blocks, robust changes only the audio block (plan section 18)")
def _():
    tr = np.flatnonzero(CTX.fold_id([155])[0] != 0)[:900]
    rows, y, groups = CTX.rows.iloc[tr], CTX.data.y[tr], CTX.groups[tr]
    test = CTX.test_rows.iloc[:200]
    old = recipe(name="probe_spline", family="logistic", base="F1", params={"C": 0.05},
                 preprocessing={"name": "spline", "n_knots": 5})
    new = recipe(name="probe_spline", family="logistic", base="F1", params={"C": 0.05},
                 preprocessing={"name": "spline_calendar", "n_knots": 5})
    a, b = (fit_recipe(r, rows, y, groups, FitContext(155, 0)) for r in (old, new))
    assert np.array_equal(a.model_inputs(test), b.model_inputs(test)) and np.array_equal(a.predict_high(test),
                                                                                         b.predict_high(test))
    f2 = recipe(name="probe_spline_f2", family="logistic", base="F2", blocks=["cyclic", "date_count"], params={"C": 0.05},
                preprocessing={"name": "spline_calendar", "n_knots": 5})
    fit = fit_recipe(f2, rows, y, groups, FitContext(155, 0))
    ct = fit.estimators[0].named_steps["prep"].named_steps["columns"]
    names = {n: list(c) for n, _, c in ct.transformers_ if n != "remainder"}
    assert names["rest"] == [c for c in feature_columns(f2) if c in
                             ("speechiness", "danceability", "duration_ms", "energy", "instrumentalness", "valence",
                              "tempo", "loudness", "acousticness", "liveness", "year")]
    assert names["categories"] == ["time_signature", "mode", "key", "date_precision"] or \
        sorted(names["categories"]) == sorted(["time_signature", "mode", "key", "date_precision"])
    assert set(names["linear"]) == {"month", "day", "day_of_year", "weekday", "days", "date_count"}
    assert len(names["cyclic"]) == 6 and names["cyclic_missing"] == names["cyclic"]
    Z = fit.model_inputs(rows)
    miss = fit.features.transform(rows)["month_sin"].isna().to_numpy()
    assert miss.any() and np.array_equal(Z[:, -3], miss.astype(float))  # one indicator per phase pair
    raises(lambda: recipe(family="lightgbm", preprocessing={"name": "spline_calendar", "n_knots": 5}), ValueError)
    k1 = recipe(name="probe_knn", family="knn", base="F1", params={"n_neighbors": 30, "weights": "distance",
                "metric": "euclidean"}, preprocessing={"name": "knn_blocks", "alpha": 1.0, "category_weight": 0.5,
                                                       "normalization": "l2"})
    k2 = dict(k1, preprocessing=dict(k1["preprocessing"], name="knn_blocks_v2", audio_scaler="standard"))
    k3 = dict(k1, preprocessing=dict(k1["preprocessing"], name="knn_blocks_v2", audio_scaler="robust"))
    f1, f2_, f3 = (fit_recipe(validate_recipe(r), rows, y, groups, FitContext(155, 0)) for r in (k1, k2, k3))
    z1, z2, z3 = (f.model_inputs(test) for f in (f1, f2_, f3))
    assert np.array_equal(z1, z2) and np.array_equal(f1.predict_high(test), f2_.predict_high(test))
    assert not np.allclose(z1[:, :10], z3[:, :10]) and np.array_equal(z1[:, 10:], z3[:, 10:])
    raises(lambda: validate_recipe(dict(k1, preprocessing=dict(k3["preprocessing"], audio_scaler="minmax"))), ValueError)
    return "F1 matrices and predictions identical; F2: 11 splined, 4 one-hot, 6 linear, 6 cyclic with 3 indicators; " \
        "kNN standard identical, robust changes only the ten audio columns"


def _fs_proto():
    return run_experiments.load_feature_protocol(CTX, "P2.4")[0]


def _entry(name, family, rec, mean, blend=None, task="P2.2"):
    b = blend if blend is not None else 0.8314739289
    return {"run_id": f"run_{name}", "task_id": task, "family": family, "recipe": rec, "recipe_id": f"{name}_x",
            "recipe_hash": recipe_hash(rec), "bank_sha256": "b_" + name, "features": feature_columns(rec), "mean": mean,
            "repeats": [mean] * 3, "blends": {0.1: {"mean": b, "repeats": [b] * 3}, 0.25: {"mean": b, "repeats": [b] * 3}}}


@check("P2.4 feature stage: declared changes, judgments against frozen references, the counter with and without "
       "imports, witnesses, expansion queues by hand (released slots, exhausted bases and holders, matched menu "
       "comparators), units and the family cap, normalized re-tuning keys (plan section 18)")
def _():
    fs = feature_stage
    base = recipe(name="probe_base", blocks=["date_count", "date_te30"])
    assert fs.changed(base, {"add": ["date_te10"]}, "x")["blocks"] == ["date_count", "date_te10"]
    assert fs.changed(base, {"remove": ["date_te30"]}, "x")["blocks"] == ["date_count"]
    assert fs.changed(base, {"drop": ["weekday", "days"]}, "x")["drop"] == ["days", "weekday"]
    sp = recipe(name="s", family="logistic", base="F1", params={"C": 1.0}, preprocessing={"name": "spline", "n_knots": 7})
    assert fs.changed(sp, {"base": "F2", "preprocessing": {"name": "spline_calendar"}}, "x")["preprocessing"] == \
        {"name": "spline_calendar", "n_knots": 7}
    h = _entry("h", "lightgbm", recipe(name="h"), 0.8314739289)
    lgb = _entry("lgb", "lightgbm", recipe(name="lgb", params=dict(SMALL_LGBM, num_leaves=31)), 0.8339)
    et = _entry("et", "extra_trees", recipe(name="et", family="extra_trees", params={"n_estimators": 50}), 0.8280, 0.8351)
    pool = [h, lgb, et]
    st = fs.state(pool, h, {"lightgbm", "extra_trees"}, None)
    assert st["strongest"]["run_id"] == "run_lgb" and st["incumbents"]["extra_trees"]["run_id"] == "run_et"
    assert st["previous_best_blends"]["extra_trees"]["weight"] == 0.1 and st["previous_best_blends"]["lightgbm"]["weight"] == 0.0
    ch = _entry("c", "lightgbm", fs.changed(lgb["recipe"], {"add": ["date_te30"]}, "c"), 0.8372, 0.8360)
    m = {"kind": "initial", "state": st, "h": {"run_id": "run_h"},
         "comparisons": [{"id": "B20", "family": "lightgbm", "comparator_run_id": "run_lgb", "challenger_run_id": "run_c"}]}
    j = fs.judge(m, {"run_lgb": lgb, "run_c": ch}, imports=[{"routes": ["ensemble_contribution"]}])
    assert j["comparisons"]["B20"]["routes"] == ["clear_standalone", "smaller_useful", "competitive_refinement",
                                                 "ensemble_contribution"]
    assert abs(j["comparisons"]["B20"]["matched_difference"] - 0.0033) < 1e-12 and j["allowance_after"] == 2
    w = fs.witnesses(m, j, {"run_lgb": lgb, "run_c": ch})
    assert [x[0] for x in w] == ["clear_standalone", "smaller_useful", "competitive_refinement", "ensemble_contribution"]
    assert w[0][3] == ("run", "run_lgb") and w[3][2] == ("blend", "run_c", 0.1) and w[3][3] == ("run", "run_h")
    weak = _entry("w", "lightgbm", ch["recipe"], 0.8300)
    j2 = fs.judge(m, {"run_lgb": lgb, "run_c": weak}, imports=[{"routes": ["ensemble_contribution"]}])
    assert not j2["comparisons"]["B20"]["routes"] and j2["progress"] and j2["allowance_after"] == 2  # imports count
    j3 = fs.judge(dict(m, kind="expansion", state=dict(st, allowance_before=2)), {"run_lgb": lgb, "run_c": weak},
                  imports=[{"routes": ["ensemble_contribution"]}])
    assert not j3["progress"] and j3["allowance_after"] == 1  # later batches count only their own comparisons
    later = dict(st, strongest=dict(st["strongest"], repeat_mean_auc_unrounded=[0.90] * 3))
    assert fs.judge(dict(m, state=later), {"run_lgb": lgb, "run_c": ch}, [])["comparisons"]["B20"]["routes"] == \
        ["clear_standalone", "ensemble_contribution"]  # the routes follow the batch's own frozen references
    # Expansion by hand: two date bases, the first exhausted; a removal holder without the column skipped; menu
    # comparators; released slots go to the first queue with candidates.
    full = [(c, t) for c, t in fs.DATE_VARIANTS]
    imports = []
    for k, (c, t) in enumerate(full):  # every date variant of lgb already declared
        rec = fs.changed(lgb["recipe"], {"add": (["date_count"] if c else []) + ([t] if t else [])}, f"i{k}")
        imports.append({"id": f"B{k + 1}", "comparator_recipe": lgb["recipe"], "challenger_recipe": rec,
                        "challenger_recipe_sha256": recipe_hash(rec), "routes": ["smaller_useful"] if k == 0 else []})
    f0 = _entry("f0", "lightgbm", recipe(name="f0", base="F0"), 0.8200)
    et_rec = et["recipe"]
    proto = {"expansion": {"size": 6, "max_comparisons": 50, "date_families": ["extra_trees"], "date_bases_per_batch": 1,
                           "queue": [["dates", 3], ["combinations", 1], ["smooth_knn", 0], ["removals", 1],
                                     ["other_trees", 1], ["weights", 0]], "smooth_knn": [],
                           "other_trees": [{"id": "OT1", "base": "E", "change": {"add": ["interactions"]},
                                            "comparator_change": {"add": ["music"]}}]}}
    aliases = {"E": et}
    got = fs.plan_expansion(proto, [h, lgb, et, f0], aliases, [], [], imports, 2)
    qs = [(c["generation"]["queue"], c["comparator"]["name"], c["challenger"]["blocks"], c["challenger"]["drop"])
          for c in got]
    want_dates = [("dates", fs.base_name(et_rec), b, []) for b in (["date_count", "date_te30"], ["date_count", "date_te10"],
                                                                     ["date_te10"])]
    assert qs[:3] == want_dates, qs  # lgb exhausted, so the declared family's base takes the place
    assert qs[3][0] == "removals" and qs[3][1] == fs.base_name(lgb["recipe"]) and qs[3][3] == ["days"]
    ot = got[4]
    assert ot["generation"]["queue"] == "other_trees" and ot["comparator"]["blocks"] == ["music"] \
        and ot["challenger"]["blocks"] == ["music", "interactions"]  # the declared comparator plus the change
    assert got[5]["generation"]["queue"] == "dates"  # the empty combinations queue released its slot to dates
    assert fs.plan_expansion(proto, [h, lgb, et, f0], aliases, [], [], imports, 2) == got
    # Units: qualifying feature recipes only; the family cap; distinct parents by the normalized key.
    rp = {"retune": {"max_units": 3, "max_per_family": 1, "by_standalone": 2,
                     "families": {"lightgbm": {"types": {"num_leaves": {"type": "log", "integer": True, "zero_step": None,
                                                                          "default": 31}}}, "extra_trees": {"types": {}}}}}
    u_a = _entry("ua", "lightgbm", fs.changed(lgb["recipe"], {"add": ["date_count"]}, "ua"), 0.8360)
    u_b = _entry("ub", "lightgbm", fs.changed(lgb["recipe"], {"add": ["music"]}, "ub"), 0.8350)
    u_c = _entry("uc", "extra_trees", fs.changed(et_rec, {"add": ["date_count"]}, "uc"), 0.8290, 0.8370)
    routes = {"A": ["smaller_useful"], "B": ["smaller_useful"], "C": ["ensemble_contribution"], "D": []}
    recs = {"A": u_a["recipe"], "B": u_b["recipe"], "C": u_c["recipe"], "D": lgb["recipe"]}
    chosen = fs.choose_units(rp, [u_a, u_b, u_c, lgb], routes, recs)
    assert [v["unit"]["unit_id"] for v in chosen] == [fs.unit_of(u_a["recipe"])["unit_id"],
                                                      fs.unit_of(u_c["recipe"])["unit_id"]]  # one LightGBM unit only
    k = fs.family_key(rp, "lightgbm")
    one = recipe(name="p", params=dict(SMALL_LGBM, reg_lambda=1))
    two = recipe(name="q", params=dict(SMALL_LGBM, reg_lambda=1.0))
    assert recipe_hash(one) != recipe_hash(two) and k(one) == k(two)
    p1, p2 = _entry("p1", "lightgbm", one, 0.84), _entry("p2", "lightgbm", two, 0.83)
    parents, excl = fs.retune_parents(rp, fs.unit_of(one), [p1, p2])
    assert [x["run_id"] for x in parents] == ["run_p1"] and excl == {k(one)}  # the same computation is one parent
    return ("changes; routes and witnesses against frozen references; the counter; a hand-built expansion with an "
            "exhausted base, a holder without the column, a matched menu comparator and a released slot; the family "
            "cap; integer and float parameters one computation")


@check("P2.4 imports: P2.8's receipt validates against P2.8's manifest; a changed route flag, bank hash, file hash, "
       "substituted comparator, swapped ids or another H is refused (plan section 18.2)")
def _():
    rx = run_experiments
    proto = _fs_proto()
    got = rx.validate_imports(CTX, proto)
    assert [c["id"] for c in got] == [f"B{i}" for i in range(1, 15)] and [c["id"] for c in got if c["routes"]] == \
        ["B12", "B14"]
    imp = proto["imports"]
    cs = imp["comparisons"]
    swapped = [dict(cs[1], id="B1"), dict(cs[0], id="B2")] + cs[2:]
    other = dict(cs[12], comparator_run_id=cs[0]["comparator_run_id"], comparator_bank_sha256=cs[0]["comparator_bank_sha256"],
                 comparator_recipe=cs[0]["comparator_recipe"])  # a valid P2.8 run, but not B13's comparator
    for bad in (dict(imp, comparisons=[dict(cs[0], routes=["clear_standalone"])] + cs[1:]),
                dict(imp, comparisons=[cs[0], dict(cs[1], challenger_bank_sha256="0" * 64)] + cs[2:]),
                dict(imp, judge_sha256="0" * 64), dict(imp, comparisons=swapped),
                dict(imp, comparisons=cs[:12] + [other] + cs[13:]),
                dict(imp, h_run_id=cs[13]["comparator_run_id"])):
        raises(lambda bad=bad: rx.validate_imports(CTX, dict(proto, imports=bad)), ValueError)
    rx.validate_handoff(CTX, proto)
    bad_alias = dict(proto, aliases=dict(proto["aliases"], C=dict(proto["aliases"]["C"], run_id=proto["aliases"]["X"]["run_id"])))
    raises(lambda: rx.validate_handoff(CTX, bad_alias), ValueError, "handoff")
    raises(lambda: rx.validate_handoff(CTX, dict(proto, p23_handoff=dict(proto["p23_handoff"], sha256="0" * 64))),
           ValueError, "hash")
    hb, hs = rx.reference_bank(CTX, proto)
    inv0 = proto["reference_inventory"][0]
    rx.pool_entry(CTX, inv0["run_id"], hb, hs, inv0["bank_sha256"])  # warms the cache
    raises(lambda: rx.pool_entry(CTX, inv0["run_id"], hb, hs, "0" * 64), ValueError, "expected")
    raises(lambda: feature_stage.load_protocol(dict(proto, imports=dict(imp, comparisons=cs[:13]))), ValueError)
    raises(lambda: feature_stage.load_protocol(dict(proto, imports=dict(imp, comparisons=cs[:13] + [cs[12]]))), ValueError)
    return "14 imports, B12 and B14 with routes; tampered routes, bank hash, judge hash, swapped ids, a substituted " \
        "comparator, another H, a missing and a duplicated import refused; the aliases follow P2.3's handoff, a " \
        "changed alias or handoff hash refused"


def _tiny_feature_protocol(fs_root, runs_of, task):
    rx, fs = run_experiments, feature_stage
    L, Ldc, E, ref_rec = runs_of["L"], runs_of["Ldc"], runs_of["E"], runs_of["ref"]
    imports = []
    for i in range(1, 15):
        imports.append({"id": f"B{i}", "family": "lightgbm", "comparator_run_id": L["run_id"],
                        "comparator_bank_sha256": L["bank_sha256"], "comparator_recipe": L["recipe"],
                        "challenger_run_id": Ldc["run_id"], "challenger_bank_sha256": Ldc["bank_sha256"],
                        "challenger_recipe": Ldc["recipe"], "challenger_recipe_sha256": recipe_hash(Ldc["recipe"]),
                        "routes": ["smaller_useful"] if i == 1 else []})
    L_ = lambda: {"type": "log", "integer": True, "zero_step": None, "default": 31}  # noqa: E731
    return {"schema": fs.PROTOCOL_SCHEMA, "task_id": task, "name": "tinyfeat", "split_seeds": list(SPLIT_SEEDS),
            "cutoff_utc": "2100-01-01T00:00:00+00:00", "retune_budget_end_utc": "2100-01-01T00:00:00+00:00",
            "reference_p21_run_id": ref_rec["run_id"], "h_recipe": dict(ref_rec["recipe"], name="tinyfeat_h"),
            "reference_inventory": [{"run_id": x["run_id"], "bank_sha256": x["bank_sha256"]} for x in (L, Ldc, E)],
            "p23_handoff": {"path": None, "sha256": None, "families": {}},
            "imports": {"task": "P9.7", "manifest": None, "manifest_sha256": None, "judge": None, "judge_sha256": None,
                        "output": None, "output_sha256": None, "h_run_id": ref_rec["run_id"], "comparisons": imports},
            "aliases": {"L": {"rule": {"rule": "named_run"}, "run_id": L["run_id"], "omitted_reason": None},
                        "E": {"rule": {"rule": "named_run"}, "run_id": E["run_id"], "omitted_reason": None},
                        "Z": {"rule": {"rule": "best_of_family"}, "run_id": None, "omitted_reason": "test omission"}},
            "initial": [{"id": "T1", "base": "L", "change": {"add": ["date_te30"]}},
                        {"id": "T2", "base": "E", "change": {"add": ["date_count"]}},
                        {"id": "T3", "base": "Z", "change": {"add": ["music"]}}],
            "expansion": {"size": 4, "max_comparisons": 40, "date_families": [], "date_bases_per_batch": 1,
                          "queue": [["dates", 1], ["combinations", 1], ["smooth_knn", 0], ["removals", 1],
                                    ["other_trees", 1], ["weights", 0]], "smooth_knn": [],
                          "other_trees": [{"id": "OT1", "base": "E", "change": {"add": ["interactions"]}}]},
            "retune": {"max_units": 2, "max_per_family": 2, "by_standalone": 1, "batch": 4, "local": 3, "broad": 1,
                       "max_batches": 2, "seed_base": 1,
                       "families": {"lightgbm": {"space": {"num_leaves": [3, 5, 7], "n_estimators": [10, 20, 30]},
                                                 "fixed": {}, "derived": [], "admissible": [],
                                                 "types": {"num_leaves": L_(), "n_estimators": dict(L_(), default=100)},
                                                 "trees": "n_estimators"},
                                    "extra_trees": {"space": {"min_samples_leaf": [1, 3, 9]}, "fixed": {},
                                                    "derived": [], "admissible": [],
                                                    "types": {"min_samples_leaf": dict(L_(), default=1)},
                                                    "trees": None}}}}


@check("P2.4 feature commands in memory: plan, an interrupted run and its resume, expansions until closed, both "
       "re-tuning rounds, the full audit; a missing contribution, a missing comparison, a duplicate allocation and an "
       "empty task are refused; a later reference-task run changes no declared batch (plan section 18)")
def _():
    rx, seeds, task = run_experiments, list(SPLIT_SEEDS), "P9.8"
    tiny = dict(SMALL_LGBM, n_estimators=15, num_leaves=5)
    with MemFS() as fs_:
        tmp = Path(fs_.root)
        patches = [mock.patch.object(runs, "LOG_FILE", tmp / "log.jsonl"), mock.patch.object(runs, "BANK_DIR", tmp / "bank"),
                   mock.patch.object(rx, "BATCH_DIR", tmp / "batches"),
                   mock.patch.dict(rx.FEATURE_PROTOCOLS, {task: tmp / "fp.json", "P9.5": tmp / "fp_empty.json"}),
                   mock.patch.object(rx, "validate_imports", lambda ctx, proto: proto["imports"]["comparisons"])]
        with Patches(patches):
            mk = lambda name, **o: validate_recipe(dict(recipe(name=name, base="F0", params=tiny), **o))  # noqa: E731
            made = {}
            for key, rec, t in (("ref", mk("tinyfeat_ref"), "P9.6"), ("L", mk("tinyfeat_L"), "P9.7"),
                                ("Ldc", mk("tinyfeat_Ldc", blocks=["date_count"]), "P9.7"),
                                ("E", mk("tinyfeat_E", family="extra_trees", params={"n_estimators": 20}), "P9.7")):
                r = rx.run_one(CTX, rec, seeds, t, MANIFEST, INV)
                made[key] = dict(r, recipe=rec)
            proto = _tiny_feature_protocol(fs_.root, made, task)
            fs_.files[str(tmp / "fp.json")] = artifacts.dumps(proto).encode()
            fs_.files[str(tmp / "fp_empty.json")] = artifacts.dumps(dict(proto, task_id="P9.5",
                                                                       cutoff_utc="2000-01-01T00:00:00+00:00")).encode()
            m1 = rx.plan_features(CTX, "features", 1, seeds, task, MANIFEST)
            assert [c["id"] for c in m1["comparisons"]] == ["T1", "T2"] and m1["omitted"][0]["id"] == "T3"
            calls = {"n": 0}
            real_run_one = rx.run_one

            def flaky(*a, **k):
                calls["n"] += 1
                if calls["n"] == 3:
                    raise RuntimeError("interrupted")
                return real_run_one(*a, **k)
            with mock.patch.object(rx, "run_one", flaky):
                raises(lambda: rx.run_features(CTX, "features", 1, seeds, task, MANIFEST, INV), RuntimeError)
            log_key = str(tmp / "log.jsonl")
            assert b'"record": "run_failure"' in fs_.files[log_key]
            assert rx.run_features(CTX, "features", 1, seeds, task, MANIFEST, INV)  # resumes, reusing logged runs
            assert rx.audit_features(CTX, seeds, task, MANIFEST, allow_incomplete=True)
            n = 2
            while rx.plan_features(CTX, "features", n, seeds, task, MANIFEST) is not None:
                assert rx.run_features(CTX, "features", n, seeds, task, MANIFEST, INV)
                n += 1
            expansions = n - 2
            for k in (1, 2):
                mr = rx.plan_features(CTX, "retune", k, seeds, task, MANIFEST)
                assert mr is not None and mr["units"] and all(0 < len(u["recipes"]) <= 4 for u in mr["units"])
                assert rx.run_features(CTX, "retune", k, seeds, task, MANIFEST, INV)
            assert rx.plan_features(CTX, "retune", 3, seeds, task, MANIFEST) is None
            assert rx.audit_features(CTX, seeds, task, MANIFEST)
            # A later run of a reference task changes no declared batch: the inventory is frozen.
            rx.run_one(CTX, mk("tinyfeat_late", params=dict(tiny, num_leaves=7)), seeds, "P9.7", MANIFEST, INV)
            assert rx.audit_features(CTX, seeds, task, MANIFEST)
            good = fs_.files[log_key]
            lines = good.decode().splitlines()

            def audited(new_lines, incomplete=False):
                fs_.files[log_key] = ("\n".join(new_lines) + "\n").encode()
                try:
                    return rx.audit_features(CTX, seeds, task, MANIFEST, allow_incomplete=incomplete)
                finally:
                    fs_.files[log_key] = good
            mine = [i for i, l in enumerate(lines) if f'"task_id": "{task}"' in l]
            for kind in ('"record": "contribution"', '"record": "comparison"', '"record": "bootstrap"',
                         '"record": "h_check"'):
                i = next(i for i in mine if kind in lines[i])
                assert not audited(lines[:i] + lines[i + 1:]), kind  # a missing record
                assert not audited(lines + [lines[i]]), kind  # a duplicated record
            i = next(i for i in mine if '"record": "feature_allocation"' in lines[i])
            assert not audited(lines + [lines[i]])
            assert not audited([l for l in lines if '"record": "batch_start"' not in l])  # starts removed
            b = json.loads(lines[next(i for i in mine if '"record": "bootstrap"' in lines[i])])
            assert not audited(lines + [json.dumps(dict(b, comparison_id="0" * 16))])  # an orphan witness
            for bad in ({"repeat_differences": [999.0, 999.0, 999.0]}, {"draws_positive_share": -1.0}):
                k = next(i for i in mine if '"record": "bootstrap"' in lines[i])
                assert not audited(lines[:k] + [json.dumps(dict(json.loads(lines[k]), **bad))] + lines[k + 1:]), bad
            # The stop record: a valid one after the last feature batch passes; an empty reason, an unknown or
            # earlier batch, a wrong omitted expansion, or a stop after re-tuning is refused.
            retune0 = next(i for i in mine if '"record": "feature_batch"' in lines[i] and '"stage": "retune"' in lines[i])
            head = lines[:retune0]
            feats = [json.loads(lines[i])["batch_id"] for i in mine if '"record": "feature_batch"' in lines[i]
                     and '"stage": "features"' in lines[i]]
            stop = {"record": "feature_stop", "task_id": task, "batch_id": feats[-1], "reason": "schedule",
                    "omitted_expansion": [], "timestamp_utc": "2026-10-30T20:00:00+00:00"}
            assert audited(head + [json.dumps(stop)], incomplete=True)
            for bad in ({"reason": " "}, {"batch_id": "nonexistent"}, {"batch_id": feats[0]},
                        {"omitted_expansion": ["X9_01"]}):
                assert not audited(head + [json.dumps(dict(stop, **bad))], incomplete=True), bad
            assert not audited(lines + [json.dumps(stop)])
            # Witnesses moved after their allocation are refused (they must close the batch before it).
            boot = [i for i in mine if '"record": "bootstrap"' in lines[i]]
            assert not audited([l for i, l in enumerate(lines) if i not in boot] + [lines[i] for i in boot])
            # The second feature batch interrupted after its fits, before its contributions, with the cutoff passed:
            # refused with its start, and refused without it (its logged runs are execution evidence).
            rec2 = next(i for i in mine if '"record": "feature_batch"' in lines[i] and '"stage": "features"' in lines[i]
                        and '"index": 2' in lines[i])
            bid2 = json.loads(lines[rec2])["batch_id"]
            start2 = next(i for i in mine if '"record": "batch_start"' in lines[i] and bid2 in lines[i])
            alloc2 = next(i for i in mine if '"record": "feature_allocation"' in lines[i] and bid2 in lines[i])
            cut = lines[:start2 + 1] + [lines[i] for i in range(start2 + 1, alloc2)
                                        if '"record": "run"' in lines[i] and f'"task_id": "{task}"' in lines[i]]
            with mock.patch.object(rx, "past_cutoff", lambda proto: True):
                assert not audited(cut)
                assert not audited([l for i, l in enumerate(cut) if i != start2])

            def moved(i, j):  # line i moved to just before line j
                rest = [l for k, l in enumerate(lines) if k != i]
                at = j - (i < j)
                return rest[:at] + [lines[i]] + rest[at:]
            # Review round 3, F2: the second batch record logged before the first batch's start, or after its fits
            # and before its allocation (both consume an allocation not yet logged), is refused; so is H's check
            # logged after the other fits of its batch, or before H's run.
            bid1 = feats[0]
            start1 = next(i for i in mine if '"record": "batch_start"' in lines[i] and bid1 in lines[i])
            alloc1 = next(i for i in mine if '"record": "feature_allocation"' in lines[i] and bid1 in lines[i])
            assert not audited(moved(rec2, start1)) and not audited(moved(rec2, alloc1))
            assert not audited(moved(rec2, start1), incomplete=True)
            hc = next(i for i in mine if '"record": "h_check"' in lines[i])
            hrun = next(i for i in mine if '"record": "run"' in lines[i] and m1["h"]["run_id"] in lines[i])
            assert hrun < hc < alloc1
            assert not audited(moved(hc, alloc1)) and not audited(moved(hc, alloc1), incomplete=True)
            assert not audited(moved(hc, hrun))
            # A matched comparison logged before a run it uses, and a contribution logged after the batch's first
            # witness, are refused.
            comp1 = next(i for i in range(start1, alloc1) if '"record": "comparison"' in lines[i] and i in mine)
            ch1 = json.loads(lines[comp1])["a"]
            run1 = next(i for i in range(start1, alloc1) if '"record": "run"' in lines[i] and ch1 in lines[i])
            assert not audited(moved(comp1, run1))
            boot1 = [i for i in range(start1, alloc1) if '"record": "bootstrap"' in lines[i] and i in mine]
            if boot1:
                con1 = next(i for i in range(start1, alloc1) if '"record": "contribution"' in lines[i] and i in mine)
                assert not audited(moved(con1, alloc1))
            assert audited(lines)  # the unmodified log still passes
            # Review round 3, F6: a refused stop writes nothing (a blank reason; a stop after re-tuning); a valid
            # stop through the command re-plans its omitted expansion from the log as it stood, and passes.
            head_bytes = ("\n".join(head) + "\n").encode()
            fs_.files[log_key] = head_bytes
            for reason in ("   ", "", None):
                raises(lambda: rx.stop_features(CTX, task, reason, seeds, MANIFEST), ValueError, "reason")
                assert fs_.files[log_key] == head_bytes
            fs_.files[log_key] = good
            raises(lambda: rx.stop_features(CTX, task, "schedule", seeds, MANIFEST), ValueError, "feature_stop")
            assert fs_.files[log_key] == good
            fs_.files[log_key] = head_bytes
            rx.stop_features(CTX, task, "  schedule  ", seeds, MANIFEST)
            stops = [json.loads(l) for l in fs_.files[log_key].decode().splitlines() if '"record": "feature_stop"' in l]
            assert len(stops) == 1 and stops[0]["reason"] == "schedule" and stops[0]["batch_id"] == feats[-1]
            assert rx.audit_features(CTX, seeds, task, MANIFEST, allow_incomplete=True)
            fs_.files[log_key] = good
            assert not rx.audit_features(CTX, seeds, "P9.5", MANIFEST)  # nothing executed is not complete
    return (f"initial batch interrupted at its third run and resumed; {expansions} expansion batches until closed; two "
            f"re-tuning rounds; the full audit passes and refuses a missing or duplicated contribution, comparison, "
            f"witness or H check, a duplicated allocation, removed starts, an orphan witness, corrupted witness "
            f"statistics, witnesses after their allocation, an interrupted batch after the cutoff with or without its "
            f"start, a batch record before an allocation it uses, an H check after its batch's other fits or before "
            f"H's run, a comparison before its run{', a contribution after a witness' if boot1 else ''}, four invalid "
            f"stop records and an empty task after its cutoff; refused stop commands write "
            f"nothing; a valid stop passes; a later reference-task run changes nothing")


def tiny_protocol():
    """Two families of tiny models outside the P2.2 spaces (F0, a few trees), for an end-to-end run in memory."""
    T = lambda prep: {"base": "F0", "blocks": [], "drop": [], "preprocessing": prep, "training_weights": "none",
                      "estimator_seeds": [0]}
    L = lambda i, d: {"type": "log", "integer": i, "zero_step": None, "default": d}
    lgb_space = {"num_leaves": [3, 5], "n_estimators": [20, 30]}
    return {"schema": "protocol_v1", "task_id": "P9.9", "name": "tiny", "reference": "probe",
            "reference_p21_run_id": None, "split_seeds": list(SPLIT_SEEDS),
            "seed_base": 1, "cutoff_utc": "2100-01-01T00:00:00+00:00",
            "refinement": {"families": 4, "by_standalone": 2, "batch": 12, "broad": 3, "local": [[5, 4], [4, 5]],
                           "max_dimensions": 3, "max_per_family": 24},
            "families": [
                {"family": "lightgbm", "index": 0, "initial": 3, "expansion": 1, "ceiling": 4, "template": T({"name": "none"}),
                 "anchor": {"n_estimators": 20, "learning_rate": 0.3, "num_leaves": 4, "min_child_samples": 20},
                 "initial_branches": [{"kind": "draws", "template": T({"name": "none"}),
                                       "fixed": {"learning_rate": 0.3, "min_child_samples": 20}, "space": lgb_space, "draws": 2}],
                 "space_template": T({"name": "none"}), "fixed": {"learning_rate": 0.3, "min_child_samples": 20},
                 "space": lgb_space, "derived": [], "admissible": [{"product": ["learning_rate", "n_estimators"], "min": 5, "max": 100}],
                 "types": {"num_leaves": L(True, 31), "n_estimators": L(True, 100)},
                 "hard_bounds": {"num_leaves": {"min": 2, "max": 8, "integer": True}}, "trees": "n_estimators"},
                {"family": "logistic", "index": 1, "initial": 3, "expansion": 1, "ceiling": 4,
                 "template": T({"name": "log_standard"}), "anchor": {"C": 0.5},
                 "initial_branches": [{"kind": "explicit", "template": T({"name": "log_standard"}),
                                       "values": [{"C": 0.2}, {"C": 2.0}]}],
                 "space_template": T({"name": "log_standard"}), "fixed": {}, "space": {"C": [0.2, 0.5, 2.0, 5.0]},
                 "derived": [], "admissible": [], "types": {"C": L(False, 1.0)},
                 "hard_bounds": {"C": {"min": 0.01, "max": 100, "integer": False}}, "trees": None}]}


def tiny_files(fs, proto):
    tmp = Path(fs.root)
    lgb = next(f for f in proto["families"] if f["family"] == "lightgbm")
    proto = dict(proto, reference_p21_run_id=run_experiments.run_id_of(CTX, list(SPLIT_SEEDS), MANIFEST)(
        protocol.anchor_recipe(proto, lgb)))  # the tiny H checked against itself
    fs.files[str(tmp / "protocol.json")] = artifacts.dumps(proto).encode()
    return [mock.patch.object(runs, "LOG_FILE", tmp / "log.jsonl"), mock.patch.object(runs, "BANK_DIR", tmp / "bank"),
            mock.patch.object(run_experiments, "BATCH_DIR", tmp / "batches"),
            mock.patch.dict(run_experiments.PROTOCOLS, {proto["task_id"]: tmp / "protocol.json"})]


@check("wave commands in memory: preflight, initialization, H check, run, judge, bootstraps, resume, audit, refit")
def _():
    rx, seeds, task = run_experiments, list(SPLIT_SEEDS), "P9.9"
    with MemFS() as fs, Patches(tiny_files(fs, tiny_protocol())):
        tmp = Path(fs.root)
        log_key = str(tmp / "log.jsonl")
        w1 = rx.plan_wave(CTX, 1, seeds, task, MANIFEST)
        assert [len(m["recipes"]) for m in w1] == [3, 3]
        raises(lambda: rx.plan_wave(CTX, 1, [155], task, MANIFEST), ValueError, "exactly the split seeds")
        # Resume: a manifest saved without its log record is reused and logged once.
        lines = fs.files[log_key].decode().splitlines()
        fs.files[log_key] = ("\n".join(lines[:1]) + "\n").encode()
        rx.plan_wave(CTX, 1, seeds, task, MANIFEST)
        assert sum(r["record"] == "batch" for r in artifacts.read_experiments(tmp / "log.jsonl")) == 2
        # Preflight: any mismatch refuses before a fit.
        spy = mock.Mock(side_effect=AssertionError("fitted before the preflight passed"))
        with mock.patch.object(runs, "fitted_run", spy):
            raises(lambda: rx.run_wave(CTX, 1, seeds, task, dict(MANIFEST, manifest_sha256="0" * 64), INV,
                                       anchors_only=True), ValueError, "preflight")
            raises(lambda: rx.run_wave(CTX, 1, seeds, task, MANIFEST, INV), ValueError, "initialization is incomplete")
            raises(lambda: rx.run_wave(CTX, 2, seeds, task, MANIFEST, INV, anchors_only=True), ValueError, "wave 1 only")
            key = str(tmp / "batches" / Path(w1[0]["run_ids"] and artifacts.read_experiments(tmp / "log.jsonl")[0]["file"]).name)
            good = fs.files[key]
            fs.files[key] = good.replace(b'"wave": 1', b'"wave": 1 ')
            raises(lambda: rx.run_wave(CTX, 1, seeds, task, MANIFEST, INV, anchors_only=True), ValueError, "preflight")
            fs.files[key] = good
        assert not spy.called
        # Initialization, with the H check, then the search fits.
        assert rx.run_wave(CTX, 1, seeds, task, MANIFEST, INV, anchors_only=True)
        recs = artifacts.read_experiments(tmp / "log.jsonl")
        assert sum(r["record"] == "run" for r in recs) == 2 and [r["passed"] for r in recs if r["record"] == "h_check"] == [True]
        assert not any(r["record"] in ("allocation", "batch_start") for r in recs)
        # An interrupted append: the bank is finalized, its record missing; the rerun recovers it without fitting.
        h = w1[0]["run_ids"][0]
        lines = fs.files[log_key].decode().splitlines()
        fs.files[log_key] = ("\n".join(l for l in lines if f'"run_id": "{w1[1]["run_ids"][0]}"' not in l
                                       or '"record": "run"' not in l) + "\n").encode()
        with mock.patch.object(runs, "fitted_run", spy):
            rx.run_wave(CTX, 1, seeds, task, MANIFEST, INV, anchors_only=True)
        assert not spy.called
        assert rx.run_wave(CTX, 1, seeds, task, MANIFEST, INV)
        recs = artifacts.read_experiments(tmp / "log.jsonl")
        runs_ = [r for r in recs if r["record"] == "run"]
        assert len(runs_) == 6 and len({r["run_id"] for r in runs_}) == 6  # anchors reused, not refitted
        assert sum(r["record"] == "contribution" for r in recs) == 12 and sum(r["record"] == "allocation" for r in recs) == 2
        assert sum(r["record"] == "batch_start" for r in recs) == 2
        ident = [r for r in recs if r["record"] == "contribution" and r["run_id"] == h]
        assert all(r["identity"] for r in ident)
        boots = [r for r in recs if r["record"] == "bootstrap"]
        progress = [r for r in recs if r["record"] == "allocation" and r["progress"]]
        assert bool(boots) == bool(progress)
        w2 = rx.plan_wave(CTX, 2, seeds, task, MANIFEST) if progress else []
        if w2:
            assert rx.run_wave(CTX, 2, seeds, task, MANIFEST, INV)
        assert rx.audit_task(CTX, seeds, task, MANIFEST, allow_incomplete=True)
        # The writers stay untouched while every run of the task is refitted and compared.
        boom = mock.Mock(side_effect=AssertionError("a writer was called during verification"))
        with mock.patch.object(artifacts, "append_experiment", boom), mock.patch.object(banks, "save_bank_v2", boom):
            assert rx.verify_task(CTX, task, MANIFEST, INV)
        with mock.patch.object(rx, "past_cutoff", lambda proto: True):
            raises(lambda: rx.plan_wave(CTX, 9, seeds, task, MANIFEST), ValueError)
        # Tampering: an edited manifest, an edited allocation, a missing run are each refused.
        bkey = str(tmp / "batches" / Path(next(r for r in recs if r["record"] == "batch")["file"]).name)
        good = fs.files[bkey]
        fs.files[bkey] = good.replace(b'"C": 0.2', b'"C": 0.3') if b'"C": 0.2' in good else good + b" "
        assert not rx.audit_task(CTX, seeds, task, MANIFEST, allow_incomplete=True)
        fs.files[bkey] = good
        log_bytes = fs.files[log_key]
        fs.files[log_key] = log_bytes.replace(b'"progress": true', b'"progress": false', 1) \
            if b'"progress": true' in log_bytes else log_bytes.replace(b'"progress": false', b'"progress": true', 1)
        assert not rx.audit_task(CTX, seeds, task, MANIFEST, allow_incomplete=True)
        lines = log_bytes.decode().splitlines()
        drop = next(i for i, l in enumerate(lines) if '"record": "run"' in l and h not in l)
        fs.files[log_key] = ("\n".join(lines[:drop] + lines[drop + 1:]) + "\n").encode()
        assert not rx.audit_task(CTX, seeds, task, MANIFEST, allow_incomplete=True)
        fs.files[log_key] = log_bytes + log_bytes.splitlines(keepends=True)[-1]
        raises(lambda: rx.load_log(CTX, task), ValueError, "duplicate")
        fs.files[log_key] = log_bytes
        assert rx.audit_task(CTX, seeds, task, MANIFEST, allow_incomplete=True)
    with MemFS() as fs, Patches(tiny_files(fs, tiny_protocol())):
        assert not rx.verify_task(CTX, task, MANIFEST, INV)  # nothing to verify is not a pass
        rx.plan_wave(CTX, 1, seeds, task, MANIFEST)
        with mock.patch.object(rx, "past_cutoff", lambda proto: True):
            assert not rx.run_wave(CTX, 1, seeds, task, MANIFEST, INV, anchors_only=True)  # the cutoff covers init
        # Completion before the cutoff (review round 1, F2): manifests only, then anchors only, are incomplete.
        assert not rx.audit_task(CTX, seeds, task, MANIFEST) and rx.audit_task(CTX, seeds, task, MANIFEST, True)
        assert rx.run_wave(CTX, 1, seeds, task, MANIFEST, INV, anchors_only=True)
        assert not rx.audit_task(CTX, seeds, task, MANIFEST) and rx.audit_task(CTX, seeds, task, MANIFEST, True)
        assert not rx.verify_task(CTX, task, MANIFEST, INV) and rx.verify_task(CTX, task, MANIFEST, INV, seeds, True)
        # An interrupted bootstrap (review round 1, F3): the allocation waits for its bootstraps; resume finishes both.
        calls = {"n": 0}
        real = rx.paired_cluster_bootstrap

        def flaky(a, b, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("interrupted at the first bootstrap")
            return real(a, b, **k)
        with mock.patch.object(rx, "paired_cluster_bootstrap", flaky), \
                mock.patch.object(protocol, "witnesses", lambda b, res, log: [
                    ("clear_standalone", ("run", b["run_ids"][1]), ("run", b["run_ids"][0])),
                    ("smaller_useful", ("run", b["run_ids"][2]), ("run", b["run_ids"][0]))]):
            raises(lambda: rx.run_wave(CTX, 1, seeds, task, MANIFEST, INV), RuntimeError, "interrupted")
            recs = artifacts.read_experiments(Path(fs.root) / "log.jsonl")
            assert not any(r["record"] == "allocation" for r in recs)
            assert rx.run_wave(CTX, 1, seeds, task, MANIFEST, INV)
            recs = artifacts.read_experiments(Path(fs.root) / "log.jsonl")
            boot_ids = [r["comparison_id"] for r in recs if r["record"] == "bootstrap"]
            assert sum(r["record"] == "allocation" for r in recs) == 2 and len(boot_ids) == 4 == len(set(boot_ids))
            assert rx.audit_task(CTX, seeds, task, MANIFEST, allow_incomplete=True)
            # A lost bootstrap record of a closed batch is finished on resume, not skipped.
            log_key = str(Path(fs.root) / "log.jsonl")
            lines = fs.files[log_key].decode().splitlines()
            drop = max(i for i, l in enumerate(lines) if '"record": "bootstrap"' in l)
            fs.files[log_key] = ("\n".join(lines[:drop] + lines[drop + 1:]) + "\n").encode()
            assert not rx.audit_task(CTX, seeds, task, MANIFEST, allow_incomplete=True)
            assert rx.run_wave(CTX, 1, seeds, task, MANIFEST, INV)
            assert rx.audit_task(CTX, seeds, task, MANIFEST, allow_incomplete=True)
    return f"6 runs, 12 contributions, 2 allocations, {len(boots)} bootstraps, wave 2 with {len(w2)} batch(es); " \
           "preflight refusals before any fit; resume after a lost manifest record and a lost run record; " \
           "the full refit writes nothing; tampering, a missing run and duplicates refused; the cutoff covers init; " \
           "completion refused before the cutoff; an interrupted or lost bootstrap finished on resume"


@check("H check: H must equal the P2.1 anchor's arrays; a different bank stops the search")
def _():
    rx, seeds, task = run_experiments, list(SPLIT_SEEDS), "P9.9"
    with MemFS() as fs, Patches(tiny_files(fs, tiny_protocol())):
        rx.plan_wave(CTX, 1, seeds, task, MANIFEST)
        assert rx.run_wave(CTX, 1, seeds, task, MANIFEST, INV, anchors_only=True)
        proto, _ = rx.load_protocol(CTX, task)
        logistic_anchor = rx.load_log(CTX, task).batches[1]["run_ids"][0]
        h = rx.load_log(CTX, task).batches[0]["run_ids"][0]
        assert not rx.h_check(CTX, dict(proto, reference_p21_run_id=logistic_anchor), h, task)
    return "a mismatched reference bank fails the check"


@check("timing: fit and predict with every scorer disabled; records hold times, never a score")
def _():
    boom = mock.Mock(side_effect=AssertionError("a score was computed during timing"))
    probe = recipe(name="probe_time", params=dict(SMALL_LGBM, n_estimators=10))
    with MemFS() as fs, mock.patch("sklearn.metrics.roc_auc_score", boom), \
            mock.patch("songpop.phase2.engine.roc_auc_score", boom), mock.patch("songpop.metrics.roc_auc_score", boom):
        recs = run_experiments.time_cases(CTX, {"schema": "timing_v1", "cases": [{"case": "light", "recipe": probe}]},
                                          [1029, 155], "P9.9", MANIFEST, INV, "0" * 64, log_path=Path(fs.root) / "t.jsonl")
    assert not boom.called and recs[0]["split_seeds"] == [155, 1029] and recs[0]["completed_fits"] == 10
    assert not [k for k in artifacts.dumps(recs[0]).lower().split('"') if "auc" in k or "score" in k]
    return f"{recs[0]['elapsed_seconds']} s for 10 fits; no scorer called; seeds canonicalized"


def _evidence(cid, y, g, F, oof):
    from songpop.phase2.selection import Evidence
    auc = np.array([[roc_auc_score(y[F[r] == k], oof[r, F[r] == k]) for k in range(5)] for r in range(len(F))])
    return Evidence(cid, np.arange(len(y)), y, g, (155, 2026, 1029)[:len(F)], F, oof, auc)


@check("selection (P2.6a): exactly the menu's candidates, complete on three repeats of five folds, on one population "
       "and its folds, fold AUCs recomputed; the tie set is measured from the single maximum, not chained; ties go "
       "to fewer atomic recipes, then supervised transformations, learned combiners, complexity rank, id")
def _():
    from songpop.phase2 import procedures as P
    from songpop.phase2.selection import select_candidate
    rng = np.random.default_rng(0)
    n = 600
    y, g = rng.integers(0, 2, n), np.arange(n) // 2
    F = np.vstack([P.population_folds(y, g, np.arange(n), 5, s) for s in (155, 2026, 1029)])
    pool = [(1.0 + 0.002 * i) * y + rng.normal(0, 1.6, (3, n)) for i in range(40)]
    ev = [_evidence(f"c{i:02d}", y, g, F, 1 / (1 + np.exp(-o))) for i, o in enumerate(pool)]
    m = {e.candidate_id: float(e.fold_auc.mean()) for e in ev}
    # a chain a > b > c: b within 0.001 of a, c within 0.001 of b but more than 0.001 below a
    trip = next((a, b, c) for a in ev for b in ev for c in ev
                if 0 < m[a.candidate_id] - m[b.candidate_id] < 0.001 and 0 < m[b.candidate_id] - m[c.candidate_id] < 0.001
                and m[a.candidate_id] - m[c.candidate_id] > 0.001)
    a, b, c = trip
    top = max(ev, key=lambda e: m[e.candidate_id])
    chosen = [top] if top.candidate_id not in (a.candidate_id, b.candidate_id, c.candidate_id) else []
    evs = {e.candidate_id: e for e in [a, b, c]}
    menu = {"candidates": [{"id": a.candidate_id, "atomic_recipes": 3, "supervised_transformations": 0,
                            "learned_combiners": 0, "complexity_rank": 1},
                           {"id": b.candidate_id, "atomic_recipes": 2, "supervised_transformations": 1,
                            "learned_combiners": 1, "complexity_rank": 9},
                           {"id": c.candidate_id, "atomic_recipes": 1, "supervised_transformations": 0,
                            "learned_combiners": 0, "complexity_rank": 0}]}
    sel = select_candidate(evs, menu)
    assert sel["tied"] == sorted([a.candidate_id, b.candidate_id]) and sel["selected"] == b.candidate_id, sel
    menu2 = {"candidates": [dict(x, atomic_recipes=2) for x in menu["candidates"]]}
    assert select_candidate(evs, menu2)["selected"] == a.candidate_id  # equal atoms: a has fewer transformations
    menu3 = {"candidates": [dict(x, atomic_recipes=2, supervised_transformations=0, learned_combiners=0,
                                 complexity_rank=5) for x in menu["candidates"]]}
    assert select_candidate(evs, menu3)["selected"] == min(a.candidate_id, b.candidate_id)  # then the id
    # refusals: a partial bank, forged fold AUCs, other folds, a menu mismatch
    # each tie key decides on its own when the earlier ones are equal
    flat = {"atomic_recipes": 1, "supervised_transformations": 0, "learned_combiners": 0, "complexity_rank": 0}
    for key in ("atomic_recipes", "supervised_transformations", "learned_combiners", "complexity_rank"):
        m4 = {"candidates": [dict(flat, id=a.candidate_id, **{key: 2}), dict(flat, id=b.candidate_id, **{key: 1}),
                             dict(flat, id=c.candidate_id)]}
        assert select_candidate(evs, m4)["selected"] == b.candidate_id, key
    # refusals: a partial bank, forged fold AUCs, other folds, a menu mismatch, a key filed under another id, a group
    # split across folds, a sixth fold
    from dataclasses import replace
    part = evs[a.candidate_id]
    bad = {**evs, a.candidate_id: replace(part, fold_id=F[:2], oof=part.oof[:2], fold_auc=part.fold_auc[:2],
                                          split_seeds=(155, 2026))}
    raises(lambda: select_candidate(bad, menu), ValueError, "complete")
    forged = {**evs, a.candidate_id: replace(part, fold_auc=part.fold_auc + 0.01)}
    raises(lambda: select_candidate(forged, menu), ValueError, "fold AUCs")
    other = {**evs, a.candidate_id: _evidence(a.candidate_id, y, g, F[::-1].copy(), part.oof)}
    raises(lambda: select_candidate(other, menu), ValueError, "population")
    raises(lambda: select_candidate({k: v for k, v in evs.items() if k != c.candidate_id}, menu), ValueError, "menu")
    raises(lambda: select_candidate({**evs, a.candidate_id: replace(part, candidate_id="x")}, menu), ValueError, "filed")
    split = F.copy()
    split[:, 0] = (split[:, 1] + 1) % 5  # rows 0 and 1 share group 0
    first = menu["candidates"][0]["id"]
    sp = {k: _evidence(k, y, g, split, v.oof) for k, v in evs.items()}
    raises(lambda: select_candidate(sp, menu), ValueError, "split across")
    six = F.copy()
    six[:, g == g[-1]] = 5
    raises(lambda: select_candidate({k: replace(v, fold_id=six) for k, v in evs.items()}, menu), ValueError, "folds")
    return f"chain {m[a.candidate_id]:.5f} > {m[b.candidate_id]:.5f} > {m[c.candidate_id]:.5f}: tie set of two"


@check("procedures (P2.6a): grouped population folds cover each row once with whole groups; constrained weights are "
       "nonnegative, sum to one and fail without falling back; the stacker refuses a convergence warning; the cache "
       "refuses an entry for another request")
def _():
    from songpop.phase2 import procedures as P
    rng = np.random.default_rng(1)
    n = 500
    y, g = rng.integers(0, 2, n), np.arange(n) // 3
    f = P.population_folds(y, g, np.arange(n) * 7, 5, 11)
    P.check_folds(y, g, f, 5)
    assert all(len(set(f[g == k])) == 1 for k in np.unique(g))
    broken = f.copy()
    broken[0] = (broken[0] + 1) % 5  # row 0's group now spans two folds
    raises(lambda: P.check_folds(y, g, broken, 5), ValueError, "split across")
    good = np.clip(0.8 * y + 0.1 + rng.normal(0, 0.05, n), 0.01, 0.99)
    noise = rng.uniform(0.01, 0.99, n)
    Z = np.column_stack([good, noise])
    st = P.fit_weights(Z, y, 0.0)
    w = np.asarray(st["weights"])
    assert (w >= 0).all() and abs(w.sum() - 1) < 1e-12 and w[0] > 0.9 and st["loss"] <= st["equal_weight_loss"]
    heavy = np.asarray(P.fit_weights(Z, y, 1e6)["weights"])
    assert np.allclose(heavy, 0.5, atol=1e-3)
    assert np.array_equal(P.apply_combiner(st, np.array([[1.0, 1.0]])), np.array([1.0]))  # the combination, unclipped
    from types import SimpleNamespace
    with mock.patch.object(P, "minimize", lambda *a, **k: SimpleNamespace(success=False, x=np.array([0.5, 0.5]),
                                                                           message="forced", nit=0)):
        raises(lambda: P.fit_weights(Z, y, 0.0), ValueError, "failed")
    # round V: a point on the clipping boundary must not pass as converged when a feasible move improves the loss
    edge = P.fit_weights(np.array([[0.0, 2e-6], [0.5, 0.5]]), np.array([1, 0]), 0.0)
    assert edge["loss"] < 7.25, edge
    raises(lambda: P.fit_combiner(("misspelled", 0.1), Z, y), ValueError, "undeclared")
    raises(lambda: P.fit_weights(np.column_stack([good, noise * 2]), y, 0.0), ValueError, "probabilities")
    raises(lambda: P.check_channels([{"id": "a", "atoms": [recipe()]}, {"id": "a", "atoms": [recipe()]}]),
           ValueError, "distinct")
    s2 = P.fit_stacker(Z, y, 0.1)
    assert s2["coef"][0] > 0 and len(s2["mean"]) == 2
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    real = P.LogisticRegression.fit

    def warn_fit(self, X, yy):
        warnings.warn("forced", ConvergenceWarning)
        return real(self, X, yy)
    with mock.patch.object(P.LogisticRegression, "fit", warn_fit):
        raises(lambda: P.fit_stacker(Z, y, 0.1), ConvergenceWarning)
    cache = P.Cache()
    base = {"fitting": {"rows": 3}, "folds_shape": [1, 3], "n_folds": 2, "prediction": {"rows": 1}}
    ra, rb = dict(base, x=1), dict(base, x=2)
    cache.put(ra, np.full((1, 3), 0.5), np.full((1, 2, 1), 0.5))
    assert cache.get(rb) is None and cache.get(ra) is not None
    got = cache.get(ra)[0]
    got[:] = 0.9  # a returned array is a copy: mutating it changes nothing stored
    assert (cache.get(ra)[0] == 0.5).all()
    raises(lambda: cache.put(ra, np.full((1, 3), 0.4), np.full((1, 2, 1), 0.5)), ValueError, "other predictions")
    raises(lambda: cache.put(rb, np.full((1, 3), 2.0), np.full((1, 2, 1), 0.5)), ValueError, "malformed")
    raises(lambda: cache.put(rb, np.full((1, 3), 0.5), np.full((1,), np.nan)), ValueError, "malformed")
    cache.memory[P.Cache.key(rb)] = cache.memory[P.Cache.key(ra)]  # an entry filed under another request's key
    raises(lambda: cache.get(rb), ValueError, "another request")
    import pandas as pd_
    assert P.frame_digest(pd_.DataFrame({"d": [None]}, index=[1])) != P.frame_digest(pd_.DataFrame({"d": ["NA"]}, index=[1]))
    return f"weights {np.round(w, 4).tolist()}; heavy lambda {np.round(heavy, 4).tolist()}"


def _tiny_population(n=480):
    from songpop.phase2 import procedures as P
    idx = np.arange(n)
    return P.Population(CTX.rows.iloc[idx], CTX.data.y[idx].copy(), CTX.groups[idx].copy(), "tiny")


TINY_LGBM = dict(SMALL_LGBM, n_estimators=15, num_leaves=5)


@check("learned procedures (P2.6a): every fit inside an outer fold uses only that fold's training rows; changing the "
       "labels of a fold's held-out rows changes none of its predictions; a meta-held-out group's base predictions "
       "ignore its own labels (positive control: others change); all six settings share one set of base fits")
def _():
    from songpop.phase2 import procedures as P
    pop = _tiny_population()
    F = P.population_folds(pop.y, pop.groups, pop.row_ids, 5, 155)[None, :]
    chans = [{"id": "a", "atoms": [recipe(name="pa", params=TINY_LGBM, blocks=["date_te30"])]},  # target-encoded
             {"id": "b", "atoms": [recipe(name="pb", params=dict(TINY_LGBM, num_leaves=3))]}]
    pk = lambda r: {}  # noqa: E731
    seen, fits = [], []
    real = P.run_recipe_cv
    import songpop.phase2.engine as E
    real_fit = E.fit_recipe

    def spy(rec, rows, *a, **k):
        seen.append(set(np.asarray(rows.index).tolist()))
        return real(rec, rows, *a, **k)

    def spy_fit(rec, rows, yy, gg, ctx):
        fits.append((str(ctx.split_seed), ctx.fold, frozenset(np.asarray(rows.index).tolist())))
        return real_fit(rec, rows, yy, gg, ctx)

    def run(pp, cache):
        outer = {ch["id"]: P.channel_cv(ch, pp, F, ["prod"], pp.rows.iloc[:5], cache, "m", pk) for ch in chans}
        sets = [("constrained", l) for l in P.LAMBDAS] + [("stacker", c) for c in P.CS]
        return P.evaluate_learned(chans, sets, outer, pp, F, [155], cache, "m", pk)
    cache = P.Cache()
    with mock.patch.object(P, "run_recipe_cv", spy), mock.patch.object(E, "fit_recipe", spy_fit):
        got, rcpt = run(pop, cache)
    ids = pop.row_ids
    assert sum(1 for s_ in seen if s_ != set(ids.tolist())) == 2 * 5, "one base cross-fit per channel and outer fold"
    # every innermost fit, by its context: a meta fit of outer fold k and meta fold j saw exactly D_k minus meta fold j
    n_meta = 0
    for name, j, rows in fits:
        if name.startswith(f"{P.META_PROTOCOL}:tiny:155:"):
            k = int(name.rsplit(":", 1)[1])
            D = pop.subset(np.flatnonzero(F[0] != k), "D")
            meta = P.population_folds(D.y, D.groups, D.row_ids, 5, P.coordinate_seed(P.META_PROTOCOL, "tiny", 155, k))
            assert rows == frozenset(D.row_ids[meta != j].tolist()), (name, j)
            n_meta += 1
        else:
            assert name == "prod" and rows == frozenset(ids[F[0] != j].tolist()), (name, j)  # the outer fits
    assert n_meta == 2 * 5 * 5, n_meta
    # flip the labels of outer fold 0's held-out rows: fold 0's predictions are unchanged, others move
    y2 = pop.y.copy()
    y2[F[0] == 0] = 1 - y2[F[0] == 0]
    got2, _ = run(P.Population(pop.rows, y2, pop.groups, "tiny"), P.Cache())
    s0 = ("stacker", 0.1)
    assert np.array_equal(got[s0]["oof"][0, F[0] == 0], got2[s0]["oof"][0, F[0] == 0])
    assert np.array_equal(got[s0]["test"][0, 0], got2[s0]["test"][0, 0])
    assert not np.array_equal(got[s0]["test"][0, 1], got2[s0]["test"][0, 1])
    # meta level: inside D (fold 0's training part), flipping one meta-held-out group's labels leaves its base
    # predictions unchanged and moves other rows'
    D = pop.subset(np.flatnonzero(F[0] != 0), "D")
    meta = P.population_folds(D.y, D.groups, D.row_ids, 5, 3)
    o1, _ = P.atomic_cv(chans[0]["atoms"][0], D, meta[None, :], ["m0"], D.rows.iloc[:1], P.Cache(), "m", {})
    yd = D.y.copy()
    yd[meta == 2] = 1 - yd[meta == 2]
    o2, _ = P.atomic_cv(chans[0]["atoms"][0], P.Population(D.rows, yd, D.groups, "D"), meta[None, :], ["m0"],
                        D.rows.iloc[:1], P.Cache(), "m", {})
    assert np.array_equal(o1[0, meta == 2], o2[0, meta == 2]) and not np.array_equal(o1[0, meta != 2], o2[0, meta != 2])
    assert all(r["meta_rows"] == int((F[0] != r["fold"]).sum()) for r in rcpt)
    return f"{len(seen)} cross-fits for 6 settings; cache hits {cache.hits}"


@check("nested evaluation (P2.6a): every fit stays inside its outer training rows; changing the outer validation "
       "labels changes no evidence, selection or prediction of that fold; the decision function applies the fallback "
       "before the deployment rule; the reduced path decides nothing")
def _():
    from songpop.phase2 import nested as Nn, procedures as P
    pop = _tiny_population(360)
    a1 = recipe(name="na", params=TINY_LGBM)
    a2 = recipe(name="nb", params=dict(TINY_LGBM, num_leaves=3))
    chans = [{"id": "ca", "atoms": [a1]}, {"id": "cb", "atoms": [a2]}]
    menu = {"candidates": [
        {"id": "anchor", "kind": "atomic", "recipe": a1, "atomic_recipes": 1, "supervised_transformations": 0,
         "learned_combiners": 0, "complexity_rank": 0},
        {"id": "other", "kind": "atomic", "recipe": a2, "atomic_recipes": 1, "supervised_transformations": 0,
         "learned_combiners": 0, "complexity_rank": 1},
        {"id": "blend", "kind": "blend", "members": [{"channel": chans[0], "weight": 0.5},
                                                     {"channel": chans[1], "weight": 0.5}],
         "atomic_recipes": 2, "supervised_transformations": 0, "learned_combiners": 0, "complexity_rank": 2},
        {"id": "stack", "kind": "learned", "channels": chans, "setting": ["stacker", 0.1], "atomic_recipes": 2,
         "supervised_transformations": 0, "learned_combiners": 1, "complexity_rank": 3}]}
    pk = lambda r: {}  # noqa: E731
    seen = []
    real = P.run_recipe_cv
    real_fit = Nn.fit_recipe

    def spy(rec, rows, *a, **k):
        seen.append(set(np.asarray(rows.index).tolist()))
        return real(rec, rows, *a, **k)

    def spy_fit(rec, rows, *a, **k):
        seen.append(set(np.asarray(rows.index).tolist()))
        return real_fit(rec, rows, *a, **k)
    fold = Nn.outer_folds(pop)  # the outer split stratifies on labels, so the isolation test holds it fixed
    with mock.patch.object(P, "run_recipe_cv", spy), mock.patch.object(Nn, "fit_recipe", spy_fit):
        res = Nn.run(menu, pop, P.Cache(), "m", pk, "anchor", fold=fold)
    trains = [set(pop.row_ids[fold != o].tolist()) for o in range(3)]
    assert all(any(s_ <= t for t in trains) for s_ in seen), "a fit used rows outside every outer training part"
    y2 = pop.y.copy()
    y2[fold == 0] = 1 - y2[fold == 0]
    res2 = Nn.run(menu, P.Population(pop.rows, y2, pop.groups, "tiny"), P.Cache(), "m", pk, "anchor", fold=fold)
    f0, g0 = res["folds"][0], res2["folds"][0]
    assert f0["selected"] == g0["selected"] and f0["selection"]["means"] == g0["selection"]["means"]
    for key in ("fold_average", "full_refit", "anchor"):
        assert np.array_equal(f0[key], g0[key]), key
    dec = Nn.decide(res, pop)
    assert dec["path"] == "full" and len(dec["fold_average"]) == 3
    # an incomplete, duplicated or substituted outer evaluation is refused before any decision
    raises(lambda: Nn.decide(dict(res, folds=res["folds"][:1]), pop), ValueError, "exactly once")
    raises(lambda: Nn.decide(dict(res, folds=[res["folds"][0]] * 3), pop), ValueError, "exactly once")
    swapped = [dict(res["folds"][0], validation_row_ids=res["folds"][1]["validation_row_ids"])] + res["folds"][1:]
    raises(lambda: Nn.decide(dict(res, folds=swapped), pop), ValueError, "saved validation rows")
    raises(lambda: Nn.run(dict(menu, candidates=menu["candidates"] + [dict(menu["candidates"][1], id="dup",
                                                                                kind="mystery")]), pop, P.Cache(), "m",
                          pk, "anchor", fold=fold), ValueError, "unknown candidate kind")
    fake = dict(res, folds=[dict(r) for r in res["folds"]])
    for r in fake["folds"]:  # the selection far worse than the anchor, and its refit better: the fallback wins
        r["fold_average"], r["full_refit"] = 1 - r["anchor"], r["anchor"]
    assert Nn.decide(fake, pop)["decision"].startswith("fallback")
    red = Nn.run(menu, pop, P.Cache(), "m", pk, "anchor", reduced="blend")
    assert Nn.decide(red, pop)["decision"] is None and all(r["selection"] is None for r in red["folds"])
    return f"selected per fold {dec['selected']}; {len(seen)} fits, all inside their outer training parts"


@check("P2.6b pool rule on synthetic banks: H* the strongest; three family representatives, two contribution recipes "
       "(another configuration of a represented family allowed), the anchor, leading-family configurations within 0.002, "
       "at most eight atoms; one channel per family with the within-family average for H*'s family; at most 32 "
       "procedures; complexity ranks by kind, atoms, transformations, regularization, id; a family left out of learned "
       "channels keeps its fixed blends")
def _():
    sys.path.insert(0, str(ROOT / "analysis"))
    import p26_pool as Pp
    from songpop.phase2 import procedures as P
    rng = np.random.default_rng(5)
    n = 300
    y, g = rng.integers(0, 2, n), np.arange(n)
    F = np.vstack([P.population_folds(y, g, g, 5, s_) for s_ in (155, 2026, 1029)])
    base = rng.normal(0, 1, (3, n))

    near = rng.normal(0, 1, (3, n))

    def entry(name, family, strength, own, roles=(), blocks=(), like=None):
        z = strength * y + 0.6 * base + (own * near + 0.02 * rng.normal(0, 1, (3, n)) if like else
                                         own * rng.normal(0, 1, (3, n)))
        oof = 1 / (1 + np.exp(-z))
        auc = np.array([[roc_auc_score(y[F[r] == k], oof[r, F[r] == k]) for k in range(5)] for r in range(3)])
        b = PredictionBank(g, y, g, np.array([155, 2026, 1029]), F, oof, np.arange(4), np.full((3, 5, 4), 0.5),
                           np.full(4, 0.5), auc, {"run_id": f"run_{name}"})
        rec = recipe(name=name, params=dict(TINY_LGBM, num_leaves=3 + sum(map(ord, name)) % 50), blocks=list(blocks))
        return {"name": name, "run_id": f"run_{name}", "family": family, "roles": list(roles),
                "atom_id": name, "recipe": rec, "recipe_id": f"{name}_x", "bank_sha256": name,
                "mean": float(auc.mean()), "bank": b}
    fin = [entry("h", "lightgbm", 1.3, 0.3, ("reference",), ("date_te30",), like=True),
           entry("l2", "lightgbm", 1.3, 0.3, (), ("date_te10",), like=True), entry("l3", "lightgbm", 1.3, 0.3, like=True),
           entry("lmusic", "lightgbm", 1.0, 1.2),
           entry("anchor", "lightgbm", 1.1, 0.3, ("anchor",)), entry("anchor_s3", "lightgbm", 1.1, 0.3,
                                                                   ("anchor_three_seeds",)),
           entry("cb", "catboost", 1.2, 0.6), entry("xg", "xgboost", 1.15, 0.6), entry("et", "extra_trees", 1.0, 1.4),
           entry("ebm", "ebm", 0.9, 1.5), entry("knn", "knn", 0.6, 1.6)]
    hstar, contrib, rep, cwin, pool, why, fam_avg, anc, a3 = Pp.build(fin)
    m = {e["name"]: e["mean"] for e in fin}
    assert hstar["name"] == max((e for e in fin if "anchor" not in e["name"]), key=lambda e: e["mean"])["name"]
    names = [e["name"] for e in pool]
    assert len(names) <= 8 and len({e["atom_id"] for e in pool}) == len(names) and "anchor" in names
    reps3 = sorted(rep.values(), key=lambda e: (-e["mean"], e["recipe_id"]))[:3]
    assert names[:3] == [e["name"] for e in reps3]
    assert all(why[e["run_id"]].startswith("contribution") for e in pool[3:5] if e["name"] != "anchor")
    assert all(hstar["mean"] - e["mean"] <= 0.002 for e in fam_avg) and fam_avg[0] is hstar and len(fam_avg) > 1
    chans = Pp.channels_of(pool, rep, cwin, hstar, fam_avg, anc)
    fams = [c["family"] for c in chans]
    assert len(set(fams)) == len(fams) and "anchor" not in [r for c in chans for r in c["run_ids"]]
    lead = next(c for c in chans if c["family"] == "lightgbm")
    assert lead["run_ids"] == [e["run_id"] for e in fam_avg]
    procs = Pp.procedures(pool, chans, hstar, fam_avg, anc, a3, [c for c in chans if c["family"] != "catboost"])
    learned = [p for p in procs if p["kind"] == "learned"]
    assert sum(p["kind"] != "atomic" for p in procs) <= 32 and len(learned) == 6
    assert all(p["atomic_recipes"] == sum(len(c["run_ids"]) for c in chans if c["family"] != "catboost") for p in learned)
    ranks = sorted(procs, key=lambda p: p["complexity_rank"])
    kinds = [Pp.KIND_ORDER[p["kind"] if p["kind"] != "learned" else p["setting"][0]] for p in ranks]
    assert kinds == sorted(kinds) and [p["id"] for p in ranks if p["kind"] == "learned"][:3] == [
        "constrained_l0.1", "constrained_l0.01", "constrained_l0"]
    assert any(p["kind"] == "blend" and any(mm["run_id"] == "run_cb" for mm in p["members"]) for p in procs)
    fa = next(p for p in procs if p["id"] == "family_average")
    assert fa["supervised_transformations"] == len({b_ for e in fam_avg for b_ in e["recipe"]["blocks"]
                                                    if b_.startswith("date_te")})
    return f"pool {names}; {len(procs)} candidates"


@check("deployment (P2.6a): a learned procedure's bank is its own schema, refused as a fitted bank; its test mean "
       "is the average of its fold procedures' test predictions")
def _():
    from songpop.phase2 import procedures as P
    pop = _tiny_population(300)
    F = P.population_folds(pop.y, pop.groups, pop.row_ids, 5, 155)[None, :]
    arrays = {"oof": np.clip(pop.y * 0.6 + 0.2, 0, 1)[None, :].astype(float), "test": np.full((1, 5, 4), 0.3)}
    arrays["test"][0, 2] = 0.5
    b = P.procedure_bank(arrays, pop.y, pop.row_ids, pop.groups, F, [155], np.arange(4), {"run_id": "x"})
    assert b.metadata["schema"] == P.PROCEDURE_SCHEMA and np.allclose(b.test_mean, 0.34)
    raises(lambda: banks.validate_v2_metadata(b), Exception)
    raises(lambda: P.save_procedure_bank(b, Path("/nonexistent/x.npz")), ValueError, "missing")  # no identity, states
    return "procedure banks stay apart from fitted banks and are refused without their identity and states"


TABM_TINY = {"n_blocks": 2, "d_block": 32, "dropout": 0.1, "learning_rate": 0.003, "weight_decay": 1e-4,
             "batch_size": 64}


def tabm_recipe(**over):
    base = dict(recipe(), name="probe_tabm", family="tabm", params=dict(TABM_TINY), base="F2",
                preprocessing={"name": "tabm_numeric", "scaler": "standard"}, iterations="early_stop_refit_v1")
    return validate_recipe(dict(base, **over))


TABM_CHILD = """
import json, pickle, sys
from unittest import mock
import numpy as np
from songpop.phase2 import tabm as T
X, y, g, p, seeds = pickle.load(sys.stdin.buffer)
assert not {"lightgbm", "xgboost"} & set(sys.modules)
seen, trained, nets = [], [], []
real_fit, real_train, real_network = T.TabMPreprocessor.fit, T.train, T.network
def spy_fit(self, Xf):
    seen.append(sorted(int(i) for i in Xf.index))
    return real_fit(self, Xf)
def spy_train(torch, q, Z, yy, epochs, init_seed, batch_seed, Zv=None, yv=None):
    trained.append([len(Z), None if Zv is None else len(Zv), epochs, init_seed, batch_seed])
    return real_train(torch, q, Z, yy, epochs, init_seed, batch_seed, Zv, yv)
def spy_network(torch, q, d_in, init_seed):
    net = real_network(torch, q, d_in, init_seed)
    nets.append(id(net))
    return net
with mock.patch.object(T.TabMPreprocessor, "fit", spy_fit), mock.patch.object(T, "train", spy_train), \\
        mock.patch.object(T, "network", spy_network):
    got = T.fit_state(p, X, y, g, seeds)
with T.torch_settings(4) as torch:
    torch.rand(1000)  # unrelated random numbers consumed before a repeat
    again = T.fit_state(p, X, y, g, seeds)
    net = T.network(torch, p, got["d_in"], 0)
    net.load_state_dict({k: torch.from_numpy(v) for k, v in got["state"].items()})
    Z = got["prep"].transform(X)
    z = T.logits(torch, net, Z).numpy()
q = T.predict_state(p, got["state"], got["d_in"], Z)
dup = np.vstack([Z[:300], Z[:1]])  # row 0 again at position 300, in another prediction batch
qd = T.predict_state(p, got["state"], got["d_in"], dup)
print(json.dumps({"seen": seen, "trained": trained, "fresh_networks": len(set(nets)) == 2, "receipt": got["receipt"],
                  "repeat": bool(np.array_equal(q, T.predict_state(p, again["state"], again["d_in"], Z))),
                  "mean_prob": bool(np.allclose(q, (1 / (1 + np.exp(-z))).mean(axis=1), rtol=0, atol=1e-12)),
                  "mean_logit": bool(np.allclose(q, 1 / (1 + np.exp(-z.mean(axis=1))), rtol=0, atol=1e-6)),
                  "duplicate": bool(qd[0] == qd[300] == q[0]), "digest": q.tobytes().hex()}))
"""


@check("neural early stopping (P2.5): preprocessing fitted on inner training rows only, then on all fitting rows; "
       "the inner split keeps groups whole; the stopping stage trains on inner training rows and validates on the "
       "rest; a new network trains for the chosen epoch on all fitting rows; members' probabilities averaged; "
       "repeatable after unrelated random draws and across prediction batches; torch only in the worker; the stop "
       "rule by hand; outer labels never reach a fit; unseen categories all zero; weights, missing groups, fitted "
       "blocks and foreign preprocessing refused")
def _():
    import subprocess
    from songpop.phase2 import tabm as T
    from songpop.models import LOG100
    from songpop.phase2.models import CATEGORICAL
    assert T.LOG100 == list(LOG100) and T.CATEGORICAL == list(CATEGORICAL)
    idx = np.arange(400)  # positions among the eligible rows
    rows, y, g = CTX.rows.iloc[idx], CTX.data.y[idx], CTX.groups[idx]
    X = FeaturePipeline(tabm_recipe()).fit_transform(rows, y, g, FitContext(155, 0))
    est = T.TabMClassifier(**TABM_TINY, scaler="standard", random_state=0, max_epochs=12, patience=3)
    seeds = T.stage_seeds(FitContext(155, 0), 0)
    out = subprocess.run([sys.executable, "-c", TABM_CHILD], cwd=ROOT,
                         input=pickle.dumps((X, y, g, est.get_params(), seeds)), capture_output=True, check=True)
    c = json.loads(out.stdout.decode().strip().splitlines()[-1])
    tr, va = T.inner_split(y, g, 5, seeds["inner_split"])
    ids = np.asarray(X.index)
    assert set(g[tr]).isdisjoint(g[va]) and len(tr) + len(va) == len(y)
    assert c["seen"][0] == sorted(int(i) for i in ids[tr]), "stopping preprocessing saw rows outside the inner training part"
    assert c["seen"][1] == sorted(int(i) for i in ids) and len(c["seen"]) == 2
    rc = c["receipt"]
    assert c["trained"] == [[len(tr), len(va), 12, seeds["stop_init"], seeds["stop_batches"]],
                            [len(y), None, rc["chosen_epoch"], seeds["final_init"], seeds["final_batches"]]]
    assert c["fresh_networks"] and c["repeat"] and c["duplicate"]
    assert c["mean_prob"] and not c["mean_logit"], "members' logits averaged instead of their probabilities"
    assert T.stop_rule(rc["validation_losses"], 1e-6, 3)[1] == rc["chosen_epoch"]
    assert rc["epochs_evaluated"] == len(rc["validation_losses"]) and rc["stopped_by"] in ("patience", "max_epochs")
    assert rc["stopping_preprocessing_rows_sha256"] == T.rows_sha256(ids, tr) == rc["inner_train_rows_sha256"]
    assert rc["final_preprocessing_rows_sha256"] == T.rows_sha256(ids, np.arange(len(ids)))
    # The stop rule by hand: patience counts epochs since an improvement larger than the tolerance; the chosen epoch
    # is the earliest within the tolerance of the lowest.
    assert T.stop_rule([0.5, 0.4, 0.4000005, 0.3999995, 0.41], 1e-6, 20) == (False, 2)
    assert T.stop_rule([0.5, 0.4, 0.4000005, 0.3999985, 0.41], 1e-6, 20) == (False, 4)
    assert T.stop_rule([0.5, 0.4, 0.3999995, 0.3999991, 0.41], 1e-6, 3) == (True, 2)  # 0.4 is within 1e-6 of the lowest
    raises(lambda: T.stop_rule([0.5, float("nan")], 1e-6, 3), ValueError, "non-finite")
    # Through the persistent worker from this process (LightGBM loaded): torch never loaded here; the same fit in
    # this process equals the child's; another estimator seed differs.
    assert "lightgbm" in sys.modules and "torch" not in sys.modules
    a = T.TabMClassifier(**TABM_TINY, scaler="standard", random_state=0, max_epochs=12, patience=3).fit(
        X, y, groups=g, context=FitContext(155, 0))
    o = T.TabMClassifier(**TABM_TINY, scaler="standard", random_state=1, max_epochs=12, patience=3).fit(
        X, y, groups=g, context=FitContext(155, 0))
    assert {k: v for k, v in a.receipt_.items() if k != "estimator_seed"} == rc and "torch" not in sys.modules
    pa = a.predict_proba(X)[:, 1]
    assert pa.tobytes().hex() == c["digest"], "the worker's predictions differ from the child's"
    assert not np.array_equal(pa, o.predict_proba(X)[:, 1])
    T.Worker.close()  # after a restart the same state predicts the same bytes
    assert np.array_equal(a.predict_proba(X)[:, 1], pa) and T.Worker.hello["source_sha256"] == T.source_digest()
    # a failed exchange discards the worker: a timeout, and a reply to another request
    with mock.patch.object(T, "read_frame", side_effect=TimeoutError("stalled")):
        raises(lambda: a.predict_proba(X.iloc[:3]), TimeoutError)
    assert T.Worker.proc is None
    real_read = T.read_frame
    with mock.patch.object(T, "read_frame", lambda fd, dl: dict(real_read(fd, dl), id="someone-else")):
        T.Worker.start() if T.Worker.proc is None else None
        raises(lambda: a.predict_proba(X.iloc[:3]), RuntimeError, "answered")
    assert T.Worker.proc is None and np.array_equal(a.predict_proba(X)[:, 1], pa)
    # the submission path rebuilds the same inputs from the recipe's preprocessing
    rebuilt = make_preprocessing({"name": "tabm_numeric", "scaler": "standard"}, list(X.columns)).fit(X, y)
    assert np.array_equal(rebuilt.transform(X).astype("float64"), a.transform_inputs(X))
    # Unseen categories are all zero; the indicator columns come from the recipe's columns, not from the fold.
    pre = T.TabMPreprocessor("standard").fit(X)
    Xu = X.iloc[:3].copy()
    Xu["key"] = 99.0
    cols = pre.output_columns()
    Zu = pre.transform(Xu)
    assert (Zu[:, [k for k, c_ in enumerate(cols) if c_.startswith("key=")]] == 0).all()
    assert [c_ for c_ in cols if c_.startswith("missing_")] == [f"missing_{c_}" for c_ in T.MISSING_FLAGS]
    qpre = T.TabMPreprocessor("quantile").fit(X)
    assert qpre.quant_ == [c_ for c_ in X.columns if c_ in T.AUDIO_CONTINUOUS] and len(qpre.quant_) == 10
    raises(lambda: T.TabMClassifier(**TABM_TINY, scaler="standard", random_state=0).fit(X, y), ValueError, "groups")
    raises(lambda: T.TabMClassifier(**TABM_TINY, scaler="standard", random_state=0).fit(
        X, y, groups=g, context=FitContext(155, 0), sample_weight=np.ones(len(y))), ValueError, "weights")
    raises(lambda: tabm_recipe(preprocessing={"name": "none"}), ValueError, "tabm_numeric")
    raises(lambda: tabm_recipe(training_weights="inverse_group_size"), ValueError)
    raises(lambda: tabm_recipe(blocks=["date_te30"]), ValueError, "leak")
    raises(lambda: tabm_recipe(iterations="fixed"), ValueError, "early_stop_refit_v1")
    assert tabm_recipe(blocks=["cyclic", "music"])  # stateless blocks are allowed
    raises(lambda: recipe(preprocessing={"name": "tabm_numeric", "scaler": "standard"}), ValueError, "tabm_numeric")
    raises(lambda: make_model(tabm_recipe(params={k_: v for k_, v in TABM_TINY.items() if k_ != "dropout"}), 0,
                              X.columns), ValueError, "exactly")
    return (f"stopped by {rc['stopped_by']} at {rc['epochs_evaluated']} of 12 epochs, chosen {rc['chosen_epoch']}; "
            f"inner {len(tr)}/{len(va)} rows, groups disjoint")


@check("P2.5 budget: no wave after the first is planned when the search runs so far plus the wave at its slowest "
       "run would exceed the allowance; H and the reference family are not counted; protocols without a budget "
       "are unchanged")
def _():
    from types import SimpleNamespace
    rx = run_experiments
    proto = {"task_id": "P2.5", "budget": {"fitting_seconds": 1000},
             "families": [{"family": "lightgbm", "ceiling": 1}, {"family": "tabm", "ceiling": 16}]}
    runs_ = {"h": {"task_id": "P2.5", "family": "lightgbm", "elapsed_seconds": 5000.0},
             "a": {"task_id": "P2.5", "family": "tabm", "elapsed_seconds": 300.0},
             "b": {"task_id": "P2.5", "family": "tabm", "elapsed_seconds": 200.0},
             "x": {"task_id": "P2.3", "family": "tabm", "elapsed_seconds": 9000.0}}
    log = SimpleNamespace(runs=runs_)
    wave = [{"run_ids": ["c", "d"]}]
    over = rx.budget_exceeded(proto, log, wave, 2)
    assert over == {"spent_seconds": 500.0, "slowest_run_seconds": 300.0, "projected_seconds": 600.0,
                    "allowance_seconds": 1000}
    assert rx.budget_exceeded(proto, log, [{"run_ids": ["c"]}], 2) is None  # 500 + 300 <= 1000
    assert rx.budget_exceeded(proto, log, wave, 1) is None  # the first wave is fixed by the timing
    assert rx.budget_exceeded({k: v for k, v in proto.items() if k != "budget"}, log, wave, 2) is None
    return "500 s spent + 2 x 300 s projected > 1000 s: stop"


@check("TabM through the engine (P2.5): fold receipts carry the stopping record; a bit-identical repeat; changing "
       "the labels of each fold's held-out rows changes none of that fold's predictions")
def _():
    from songpop.phase2 import models as M
    idx = np.arange(600)
    rows, y, g = CTX.rows.iloc[idx], CTX.data.y[idx], CTX.groups[idx]
    fold = CTX.fold_id([155])[:, idx]
    test = CTX.test_rows.iloc[:40]
    rec = tabm_recipe()
    with mock.patch.dict(M.TABM_FIXED, {"max_epochs": 6, "patience": 2}):
        b1 = run_recipe_cv(rec, rows, y, CTX.data.row_ids[idx], g, fold, [155], test, CTX.data.test_id[:40])
        b2 = run_recipe_cv(rec, rows, y, CTX.data.row_ids[idx], g, fold, [155], test, CTX.data.test_id[:40])
        k = 0  # fold 0's held-out labels flipped: its fit sees the same rows and labels, so its predictions agree
        y2 = y.copy()
        y2[fold[0] == k] = 1 - y2[fold[0] == k]
        b3 = run_recipe_cv(rec, rows, y2, CTX.data.row_ids[idx], g, fold, [155], test, CTX.data.test_id[:40])
    assert np.array_equal(b1.oof, b2.oof) and np.array_equal(b1.test_by_fold, b2.test_by_fold)
    assert np.array_equal(b1.oof[0, fold[0] == k], b3.oof[0, fold[0] == k])
    assert np.array_equal(b1.test_by_fold[0, k], b3.test_by_fold[0, k])
    assert not np.array_equal(b1.test_by_fold[0, 1], b3.test_by_fold[0, 1])  # positive control: other folds see them
    rcs = b1.metadata["fit_receipts"]
    assert len(rcs) == 5 and all(len(r["tabm"]) == 1 and 1 <= r["tabm"][0]["chosen_epoch"] <= 6 for r in rcs)
    return f"chosen epochs {[r['tabm'][0]['chosen_epoch'] for r in rcs]}"


def main():
    failed = 0
    for name, status, note in RESULTS:
        print(f"{status:8s}{name}" + (f"\n        {note}" if note else ""))
        failed += status == "FAIL"
    counts = {s: sum(1 for _, st, _ in RESULTS if st == s) for s in ("PASS", "FAIL", "SKIPPED", "STAGED")}
    print(f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIPPED']} skipped, {counts['STAGED']} staged")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
