"""Self-checks of the phase 2 harness (task P2.1, doc/plans/phase2-models-features.md section 4.8), without pytest.

From the project root: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.check_phase2
Everything runs in memory and writes nothing. Checks that need a package not installed yet are reported as
SKIPPED, and checks of code that later tasks build are reported as STAGED, never as passed. Exit status 0 only if
no check failed.
"""
import builtins
import copy
import io
import itertools
import math
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
from songpop.phase2 import allocation, banks, batches, protocol, refine, runs, searches
from songpop.phase2.encoders import DateCount, DateTargetEncoder, EraRelative, FitContext, inner_group_splits
from songpop.phase2.engine import fit_recipe, run_recipe_cv, training_weights
from songpop.phase2.ensembles import blend_banks, select_seeds
from songpop.phase2.features import FeaturePipeline, feature_columns
from songpop.phase2.models import THREADS, make_estimator, make_model, prediction_threads
from songpop.phase2.recipes import load_recipe, recipe_hash, recipe_id, validate_recipe
from songpop.phase2.submission import members_of, repeated_uri_pairs, validate_candidate, write_candidate
from songpop.splits import SPLIT_SEEDS
from scripts import install_locks, run_experiments  # loaded before the source manifest is taken

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


@check("CatBoost runtime: fits F2 with missing values, writes no file, returns P(class 1)", status="STAGED")
def _():
    """Built and run in P2.3, after the reviewed installation; an installed package never turns it into a pass."""


@check("EBM runtime: fits F2 with missing values, writes no file, returns P(class 1)", status="STAGED")
def _():
    """Built and run in P2.3 if EBM is used, after the reviewed installation."""


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


PROTO = protocol.load_protocol(ROOT / run_experiments.PROTOCOL)
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
            mock.patch.object(run_experiments, "PROTOCOL", tmp / "protocol.json")]


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
        proto, _ = rx.load_protocol(CTX)
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


@check("nested evaluation: fitting ids stay inside their outer training rows", status="STAGED")
def _():
    """songpop/phase2/nested.py arrives with P2.7, which first uses it; its check arrives with it."""


@check("deployment: full-refit artifacts stay apart from fold averages", status="STAGED")
def _():
    """Full-refit deployment artifacts arrive with P2.7."""


@check("selection: the frozen rule rejects partial-seed candidates and breaks ties as declared", status="STAGED")
def _():
    """songpop/phase2/selection.py arrives with P2.6; partial-seed submissions are already refused above."""


@check("neural early stopping: preprocessing fitted on inner training rows only", status="STAGED")
def _():
    """The TabM wrapper arrives with P2.5."""


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
