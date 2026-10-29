"""Self-checks of the phase 2 harness (task P2.1, doc/plans/phase2-models-features.md section 4.8), without pytest.

From the project root: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.check_phase2
Everything runs in memory and writes nothing. Checks that need a package not installed yet are reported as
SKIPPED, and checks of code that later tasks build are reported as STAGED, never as passed. Exit status 0 only if
no check failed.
"""
import builtins
import copy
import io
import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from songpop import artifacts
from songpop.data import AUDIO, DATE
from songpop.evaluate import PredictionBank
from songpop.metrics import paired_cluster_bootstrap
from songpop.phase2 import banks, runs, searches
from songpop.phase2.encoders import DateCount, DateTargetEncoder, EraRelative, FitContext, inner_group_splits
from songpop.phase2.engine import fit_recipe, run_recipe_cv, training_weights
from songpop.phase2.ensembles import blend_banks, select_seeds
from songpop.phase2.features import FeaturePipeline, feature_columns
from songpop.phase2.models import make_model
from songpop.phase2.recipes import load_recipe, recipe_id, validate_recipe
from songpop.phase2.submission import members_of, repeated_uri_pairs, validate_candidate, write_candidate
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


@check("verification writes nothing: the logged anchor recomputes with every writer disabled")
def _():
    recs = [r for r in runs.logged(CTX) if r.get("recipe_id", "").startswith("lightgbm_F2_anchor_")]
    if not recs:
        raise Skip("no logged anchor run yet; P2.1 logs it with scripts.run_experiments")
    stored = runs.load_logged(CTX, recs[-1]["run_id"])[0].metadata["source_manifest"]
    boom = mock.Mock(side_effect=AssertionError("a writer was called during verification"))
    with mock.patch.object(artifacts, "append_experiment", boom), mock.patch.object(banks, "save_bank_v2", boom), \
            mock.patch.object(searches, "save_expansion", boom), mock.patch.object(runs, "save_run", boom):
        bad = runs.verify_run(CTX, recs[-1]["run_id"], stored, INV)
    assert not bad, bad
    tampered = [dict(r, mean_fold_auc_unrounded=0.123, repeat_mean_auc_unrounded=[0.123] * 3) for r in runs.logged(CTX)]
    with mock.patch.object(runs, "logged", lambda ctx, record="run": tampered if record == "run" else []):
        bad = runs.verify_run(CTX, recs[-1]["run_id"], stored, INV)
    assert "logged mean_fold_auc_unrounded" in bad and "logged repeat_mean_auc_unrounded" in bad, bad
    fields = {"family": "random_forest", "stage": "seeds155", "task_id": "P9.9", "search_id": "none",
              "source_manifest_hash": "0" * 64, "n_rows": 1, "n_groups": 1, "n_features": 1, "execution_status": "failed",
              "warnings": ["probe"], "bank_sha256": "0" * 64, "recipe_id": "x", "record": "other"}
    for k, v in fields.items():
        one = [dict(r, **{k: v}) if r["run_id"] == recs[-1]["run_id"] else r for r in runs.logged(CTX)]
        with mock.patch.object(runs, "logged", lambda ctx, record="run", one=one: one if record == "run" else []):
            try:
                bad = runs.verify_run(CTX, recs[-1]["run_id"], stored, INV)
            except ValueError:
                bad = [f"logged {k}"]  # a record the loader itself refuses
        assert f"logged {k}" in bad, (k, bad)
    return f"{recs[-1]['run_id']} equals its saved bank and log (sources checked on disk against its manifest); " \
           f"tampered unrounded scores and {len(fields)} other single record fields are caught"


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
            raises(lambda: run_experiments.compare(CTX, anchor, rec["run_id"], "P2.1"), ValueError)
            run_experiments.compare(CTX, anchor, rec["run_id"], "P2.1", seeds=[155])
            run_experiments.compare(CTX, anchor, rec["run_id"], "P2.1", seeds=[155])
            cmp = [r for r in runs.logged(CTX, "comparison")]
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
