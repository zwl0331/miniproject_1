"""Self-checks of the harness (task P1.2, doc/plans/phase1-data-baseline.md section 4.9), without pytest.

From the project root: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.check_foundation
Everything runs in memory and writes nothing; no check passes a data file to a writer. Exit status 0 only if
every check passes.
"""
import io
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from songpop import artifacts
from songpop.data import AUDIO, DATE, encode_target, load_dataset
from songpop.evaluate import BANK_SCHEMA, PredictionBank, run_cv, validate_bank
from songpop.features import F2_DATE, make_features, parse_date
from songpop.metrics import paired_cluster_bootstrap, weighted_auc
from songpop.models import describe, log_transform, make_estimator, predict_high
from songpop.splits import FoldSet, assign_folds, build_foldset, fold_report, incomplete_fold, make_groups

ROOT = Path(".").resolve()
RESULTS = []


def check(name):
    def wrap(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
        except Exception as e:  # report every failure, keep going
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return wrap


def raises(fn, exc=Exception, match=None):
    try:
        fn()
    except exc as e:
        if match is not None and match not in str(e):
            raise AssertionError(f"raised {type(e).__name__} without {match!r}: {e}")
        return True
    raise AssertionError(f"expected {exc.__name__}")


DATA = load_dataset(ROOT)
FS = build_foldset(DATA)
E = FS.eligible


@check("target: High is 1 and Low is 0; missing and unknown labels rejected")
def _():
    assert encode_target(["High", "Low", "High"]).tolist() == [1, 0, 1]
    raises(lambda: encode_target(["High", None]), ValueError)
    raises(lambda: encode_target(["High", "Medium"]), ValueError)
    assert DATA.y_all.sum() == 1359 and len(DATA.y_all) == 3864


@check("rows: permanent row ids survive filtering; only row 425 is ineligible; feature index is the row id")
def _():
    assert DATA.row_ids.tolist() == [i for i in range(3864) if i != 425]
    assert DATA.y.sum() == 1359 and len(DATA.y) == 3863
    assert make_features(DATA.train_eligible, "F2").index.tolist() == DATA.row_ids.tolist()


def _synthetic():
    base = {c: 0.5 for c in AUDIO}
    rows = []
    for i, (uri, shift, date) in enumerate([("u1", 0, "2020"), ("u1", 0, "2020"), ("u2", 1, "2020"),
                                            ("u3", 1, "2020"), ("u4", 1, "2021"), ("u5", 2, "2020"),
                                            (None, None, "2020")]):
        r = {c: (np.nan if shift is None else v + shift) for c, v in base.items()}
        r.update({"uri": uri, DATE: date, "train_row_id": 10 + i, "Popularity_Type": "High" if i % 2 else "Low"})
        rows.append(r)
    return pd.DataFrame(rows)


@check("groups: uri, signature, transitive chain, singleton, strict coarsening, label and order invariance")
def _():
    t = _synthetic()
    g = make_groups(t).tolist()
    # 10, 11 share u1; 12 and 13 share audio + date; 14 matches 12 and 13 on audio only; 16 is incomplete
    assert g == [10, 10, 12, 12, 14, 15, 16], g
    assert make_groups(t, strict=True).tolist() == [10, 10, 12, 12, 12, 15, 16]
    chain = t.copy()
    chain.loc[3, "uri"] = "u5"  # 13 joins 15 by uri and 12 by signature: one component 12, 13, 15
    assert make_groups(chain).tolist() == [10, 10, 12, 12, 14, 12, 16]
    flipped = t.assign(Popularity_Type=t["Popularity_Type"][::-1].to_numpy())
    assert make_groups(flipped).tolist() == g
    perm = t.sample(frac=1, random_state=1)
    assert dict(zip(perm["train_row_id"], make_groups(perm))) == dict(zip(t["train_row_id"], g))


@check("groups: real data counts equal Finding 002")
def _():
    p, s = FS.primary_group, FS.strict_group
    assert len(set(p)) == 3638 and len(set(s)) == 3630
    assert int(pd.DataFrame({"g": p, "y": DATA.y_all}).groupby("g")["y"].nunique().gt(1).sum()) == 35
    assert pd.DataFrame({"p": p, "s": s}).groupby("p")["s"].nunique().max() == 1
    assert p[425] == 425 and (p == 425).sum() == 1


@check("folds: the saved split file is valid and equals the recomputed folds")
def _():
    df = pd.read_csv(ROOT / "artifacts/splits/production_v1.csv")
    assert FS.equals(FoldSet.from_frame(df))
    bad = df.copy()
    bad.loc[0, "eligible"] = 2
    raises(lambda: FoldSet.from_frame(bad), ValueError, "eligibility")
    bad = df.copy()
    bad.loc[1, "train_row_id"] = 0
    raises(lambda: FoldSet.from_frame(bad), ValueError, "row ids")
    bad = df.copy()
    bad.loc[0, "primary_fold_155"] = 5
    raises(lambda: FoldSet.from_frame(bad), ValueError, "fold ids")
    raises(lambda: FoldSet.from_frame(df.drop(columns=["strict_group"])), ValueError, "columns")


@check("folds: groups in one fold, both classes on both sides, all folds present; tables for both groupings")
def _():
    for kind in ("primary", "strict"):
        folds, groups = getattr(FS, kind), getattr(FS, f"{kind}_group")
        for r in range(3):
            f = folds[r][E]
            assert set(f) == set(range(5))
            assert pd.DataFrame({"g": groups[E], "f": f}).groupby("g")["f"].nunique().max() == 1
        table, _ = fold_report(DATA, FS, kind)  # raises on a split group or a missing class
        assert len(table) == 30 and set(table["side"]) == {"training", "validation"}
        assert (table["shared_groups"] == 0).all()


@check("folds: the incomplete row gets 4, 3, 0 and cannot move any eligible row's fold")
def _():
    assert FS.primary[:, 425].tolist() == [4, 3, 0] == [incomplete_fold(s, 425) for s in (155, 2026, 1029)]
    without = DATA.train[E].reset_index(drop=True)
    g = make_groups(without)
    folds = assign_folds(DATA.y, g, np.ones(len(without), bool), without["train_row_id"].to_numpy())
    assert np.array_equal(folds, FS.primary[:, E])
    relabeled = DATA.y_all.copy()
    relabeled[425] = 1 - relabeled[425]
    again = assign_folds(relabeled, FS.primary_group, E, FS.train_row_id)
    assert np.array_equal(again[:, E], FS.primary[:, E])


@check("dates: edge cases, rejections of malformed strings, F2 columns")
def _():
    assert np.isnan(parse_date("1984")["month"]) and parse_date("1984")["date_precision"] == 4
    d = parse_date("1979-06")
    assert d["month"] == 6 and d["date_precision"] == 7 and all(np.isnan(d[k]) for k in
                                                                ["day", "day_of_year", "weekday", "days"])
    e = parse_date("1970-01-01")
    assert (e["day_of_year"], e["weekday"], e["days"], e["date_precision"]) == (1, 3, 0, 10)
    assert parse_date("1969-12-31")["days"] == -1
    assert parse_date("2024-02-29")["day_of_year"] == 60
    for bad in ["2023-02-30", "2023-13", "2023-1-5", "0000", "0000-01", "2020\n", "２０２０", "", None]:
        raises(lambda b=bad: parse_date(b), ValueError)
    X = make_features(DATA.train_eligible, "F2")
    assert list(X.columns) == AUDIO + F2_DATE and X.shape == (3863, 20) and (X.dtypes == "float64").all()
    assert int(X["month"].isna().sum()) == 111 and int(X["days"].isna().sum()) == 112
    assert list(make_features(DATA.test, "F1").columns) == AUDIO + ["year"]


def _one_seed_inputs(fset="F2"):
    X, X_test = make_features(DATA.train_eligible, fset), make_features(DATA.test, fset)
    return X, X_test, FS.primary_group[E], FS.primary[:1, E]


@check("preprocessing: each fold's logistic statistics are those of its training rows; held-out values never enter")
def _():
    X, X_test, groups, folds = _one_seed_inputs()
    fitted = []

    def recording():
        est = make_estimator("logistic")
        fitted.append(est)
        return est

    run_cv(X, DATA.y, DATA.row_ids, groups, folds, (155,), X_test, DATA.test_id, recording)
    k = 0
    tr = folds[0] != k
    Xt = log_transform(X.iloc[np.flatnonzero(tr)])
    med = np.nanmedian(Xt.to_numpy(), axis=0)
    est = fitted[k]
    assert np.allclose(est.named_steps["impute"].statistics_, med, rtol=0, atol=1e-12)
    assert np.allclose(est.named_steps["scale"].mean_, np.where(np.isnan(Xt), med, Xt).mean(axis=0), rtol=0, atol=1e-9)
    X2 = X.copy()
    held = np.flatnonzero(folds[0] == k)
    X2.iloc[held[0], X2.columns.get_loc("loudness")] = 1e9
    X2.iloc[held[1], X2.columns.get_loc("month")] = np.nan
    fitted2 = []

    def recording2():
        est = make_estimator("logistic")
        fitted2.append(est)
        return est

    run_cv(X2, DATA.y, DATA.row_ids, groups, folds, (155,), X_test, DATA.test_id, recording2)
    a, b = fitted[k], fitted2[k]
    assert np.array_equal(a.named_steps["impute"].statistics_, b.named_steps["impute"].statistics_)
    assert np.array_equal(a.named_steps["scale"].mean_, b.named_steps["scale"].mean_)
    assert np.array_equal(a.named_steps["model"].coef_, b.named_steps["model"].coef_)


@check("run_cv: malformed inputs are rejected before any estimator is created")
def _():
    X, X_test, groups, folds = _one_seed_inputs("F0")
    y, rid = DATA.y, DATA.row_ids

    def never():
        raise AssertionError("the estimator factory was called")

    cases = {
        "one group across folds": dict(groups=np.zeros(len(y), int)),
        "missing group ids": dict(groups=np.full(len(y), np.nan)),
        "one missing group id": dict(groups=np.where(np.arange(len(y)) == 0, np.nan, groups.astype(float))),
        "two-dimensional groups": dict(groups=np.c_[groups, groups]),
        "row ids shorter": dict(row_ids=rid[:-1]),
        "no split seed": dict(seeds=()),
        "duplicate row ids": dict(row_ids=np.r_[rid[:-1], rid[0]]),
        "index not row ids": dict(X=X.reset_index(drop=True)),
        "fold id out of range": dict(fold_id=np.where(folds == 4, 7, folds)),
        "labels not 0 and 1": dict(y=y + 1),
        "test ids duplicated": dict(test_id=np.r_[DATA.test_id[:-1], DATA.test_id[0]]),
        "columns differ": dict(X_test=X_test.iloc[:, :-1]),
    }
    for name, override in cases.items():
        args = dict(X=X, y=y, row_ids=rid, groups=groups, fold_id=folds, seeds=(155,), X_test=X_test,
                    test_id=DATA.test_id, make=never)
        args.update(override)
        raises(lambda a=args: run_cv(**a), ValueError)


class _Fake:
    def __init__(self, classes, proba):
        self.classes_, self._p = classes, np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return self._p


@check("probabilities: class 1 located in classes_; the whole matrix checked")
def _():
    X = pd.DataFrame({"a": [0, 1]})
    assert predict_high(_Fake([1, 0], [[0.9, 0.1], [0.2, 0.8]]), X).tolist() == [0.9, 0.2]
    for classes, proba in [([0, 2], [[0.5, 0.5], [0.5, 0.5]]), ([0, 1], [[0.5, 0.5]]), ([0, 1], [0.5, 0.5]),
                           ([0, 1], [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]), ([0, 1], [[np.nan, 0.3], [0.5, 0.5]]),
                           ([0, 1], [[0.5, np.inf], [0.5, 0.5]]), ([0, 1], [[-0.1, 1.1], [0.5, 0.5]]),
                           ([0, 1], [[0.4, 0.4], [0.5, 0.5]])]:
        raises(lambda c=classes, p=proba: predict_high(_Fake(c, p), X), ValueError)


@check("AUC: perfect, reversed, constant, weighted ties and zero weights equal scikit-learn, one class is NaN")
def _():
    y = np.array([0, 0, 1, 1])
    assert weighted_auc(y, [0.1, 0.2, 0.3, 0.4], np.ones(4)) == 1.0
    assert weighted_auc(y, [0.4, 0.3, 0.2, 0.1], np.ones(4)) == 0.0
    assert weighted_auc(y, [0.5] * 4, np.ones(4)) == 0.5
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(5, 60))
        yy = rng.integers(0, 2, n)
        yy[:2] = [0, 1]
        s = rng.integers(0, 6, n) / 5.0  # many ties
        w = rng.integers(0, 4, n).astype(float)
        w[:2] = 1
        worst = max(worst, abs(weighted_auc(yy, s, w) - roc_auc_score(yy, s, sample_weight=w)))
    assert worst < 1e-12, worst
    assert np.isnan(weighted_auc(np.array([1, 1, 1]), [0.1, 0.2, 0.3], np.ones(3)))


def _bank(oof_seed, n_groups=120, R=3, pos_groups=None, seed=0):
    """A valid, group-safe bank: groups of one or two rows, folds assigned by group, both classes in every
    fold, fold AUCs computed from the predictions, test arrays consistent."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 3, n_groups)
    group = np.repeat(np.arange(n_groups), sizes)
    n = len(group)
    y = rng.integers(0, 2, n)  # drawn either way, so the folds below do not depend on pos_groups
    if pos_groups is not None:
        y = np.isin(group, pos_groups).astype(int)
    fold = np.vstack([rng.permutation(np.arange(n_groups) % 5)[group] for _ in range(R)])
    for r in range(R):
        for k in range(5):
            m = fold[r] == k
            y[np.flatnonzero(m)[:2]] = [0, 1] if pos_groups is None else y[np.flatnonzero(m)[:2]]
    oof = np.random.default_rng(oof_seed).random((R, n))
    auc = np.array([[roc_auc_score(y[fold[r] == k], oof[r, fold[r] == k]) for k in range(5)] for r in range(R)])
    tbf = np.random.default_rng(oof_seed + 1).random((R, 5, 4))
    return PredictionBank(np.arange(n), y, group, np.arange(R) + 1, fold, oof, np.arange(4), tbf,
                          tbf.mean(axis=(0, 1)), auc, {})


def _reference_draws(a, b, n_boot, seed):
    """Independent bootstrap with scikit-learn weights: one multiplicity vector per draw for both banks and
    every repeat; a draw with a single-class fold is skipped."""
    codes = np.unique(a.group_id, return_inverse=True)[1]
    G = codes.max() + 1
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        w = rng.multinomial(G, np.full(G, 1.0 / G))[codes].astype(float)
        diffs = []
        for r in range(a.fold_id.shape[0]):
            for k in range(5):
                m = a.fold_id[r] == k
                if w[m & (a.y == 1)].sum() == 0 or w[m & (a.y == 0)].sum() == 0:
                    diffs = None
                    break
                diffs.append(roc_auc_score(a.y[m], a.oof[r, m], sample_weight=w[m])
                             - roc_auc_score(b.y[m], b.oof[r, m], sample_weight=w[m]))
            if diffs is None:
                break
        if diffs is not None:
            out.append(np.mean(diffs))
    return np.array(out)


@check("bootstrap: matches an independent scikit-learn reference draw by draw; identity, repeat, sign, validity")
def _():
    a, b = _bank(1), _bank(2)
    for f in ("y", "group_id", "fold_id", "train_row_id", "split_seeds"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    validate_bank(a)
    validate_bank(b)
    r1 = paired_cluster_bootstrap(a, b, n_boot=300, seed=7, return_draws=True)
    ref = _reference_draws(a, b, 300, 7)
    assert len(ref) == r1["valid"] and np.allclose(r1["draws"], ref, rtol=0, atol=1e-12)
    assert np.array_equal(r1["draws"], paired_cluster_bootstrap(a, b, n_boot=300, seed=7, return_draws=True)["draws"])
    r3 = paired_cluster_bootstrap(b, a, n_boot=300, seed=7, return_draws=True)
    assert np.allclose(r1["draws"], -r3["draws"], rtol=0, atol=1e-15)
    same = paired_cluster_bootstrap(a, a, n_boot=100)
    assert same["observed_difference"] == 0 and same["interval_95"] == [0.0, 0.0]
    raises(lambda: paired_cluster_bootstrap(a, b, n_boot=0), ValueError)
    for f in ("y", "group_id", "fold_id", "train_row_id", "split_seeds"):
        c = _bank(2)
        v = getattr(c, f).copy()
        v[0] = v[0] + 1 if f != "fold_id" else v[0]
        if f == "fold_id":
            v = np.roll(v, 1, axis=1)
        setattr(c, f, v)
        raises(lambda c=c: paired_cluster_bootstrap(a, c, n_boot=5), ValueError, "cannot be paired")


@check("bootstrap: a replicate with a single-class fold is rejected as a whole")
def _():
    # one positive group per fold in the first repeat: dropping it in a draw leaves that fold without positives
    base = _bank(3, n_groups=40, R=1, seed=5)
    folds_of_group = {g: base.fold_id[0][base.group_id == g][0] for g in np.unique(base.group_id)}
    pos = [min(g for g in folds_of_group if folds_of_group[g] == k) for k in range(5)]
    a = _bank(3, n_groups=40, R=1, pos_groups=pos, seed=5)
    b = _bank(4, n_groups=40, R=1, pos_groups=pos, seed=5)
    res = paired_cluster_bootstrap(a, b, n_boot=200, seed=3, return_draws=True)
    ref = _reference_draws(a, b, 200, 3)
    assert res["invalid"] > 0 and res["valid"] + res["invalid"] == 200 and len(ref) == res["valid"]
    assert np.allclose(res["draws"], ref, rtol=0, atol=1e-12)


@check("banks: validation rejects bad schema, inconsistent deployment mean, wrong fold AUC, split groups")
def _():
    a = _bank(5)
    a.metadata = {"schema": BANK_SCHEMA, "run_id": "x", "recipe_id": "x", "feature_set": "F0", "features": [],
                  "deployment": "mean", "parameters": {}, "split_seeds": [int(s) for s in a.split_seeds]}
    validate_bank(a, require_metadata=True)
    b = artifacts.load_bank(artifacts.bank_bytes(a))
    assert artifacts.compare_banks(a, b) == [] and b.metadata == a.metadata
    c = artifacts.load_bank(artifacts.bank_bytes(a))
    c.metadata["schema"] = "INVALID"
    raises(lambda: validate_bank(c, require_metadata=True), ValueError, "schema")
    raises(lambda: artifacts.load_bank(artifacts.bank_bytes(c)), ValueError, "schema")
    assert artifacts.compare_banks(a, c) == ["metadata:schema"]
    d = artifacts.load_bank(artifacts.bank_bytes(a))
    d.test_mean = 1 - d.test_mean
    raises(lambda: validate_bank(d), ValueError, "test_mean")
    d = artifacts.load_bank(artifacts.bank_bytes(a))
    d.fold_auc = d.fold_auc + 0.01
    raises(lambda: validate_bank(d), ValueError, "fold AUC")
    for v in (np.nan, np.inf):
        d = artifacts.load_bank(artifacts.bank_bytes(a))
        d.fold_auc[0, 0] = v
        raises(lambda d=d: validate_bank(d), ValueError, "fold AUC")
        raises(lambda d=d: artifacts.load_bank(artifacts.bank_bytes(d)), ValueError, "fold AUC")
    d = artifacts.load_bank(artifacts.bank_bytes(a))
    d.group_id = np.zeros_like(d.group_id)
    raises(lambda: validate_bank(d), ValueError, "group")
    d = artifacts.load_bank(artifacts.bank_bytes(a))
    d.group_id = np.full(len(d.group_id), np.nan)
    raises(lambda: validate_bank(d), ValueError, "missing group ids")
    d = artifacts.load_bank(artifacts.bank_bytes(a))
    d.oof = d.oof + 1e-6
    d.fold_auc = np.array([[roc_auc_score(d.y[d.fold_id[r] == k], d.oof[r, d.fold_id[r] == k]) for k in range(5)]
                           for r in range(3)])
    assert artifacts.compare_banks(a, d) == ["oof"]


@check("banks: an existing path is never written, and an exclusive-create race is not overwritten")
def _():
    a = _bank(6)
    a.metadata = {"schema": BANK_SCHEMA, "run_id": "x", "recipe_id": "x", "feature_set": "F0", "features": [],
                  "deployment": "mean", "parameters": {}, "split_seeds": [int(s) for s in a.split_seeds]}
    fake = Path("/nonexistent-songpop-check/bank.npz")
    with mock.patch.object(Path, "exists", return_value=True), mock.patch("builtins.open") as op, \
            mock.patch.object(Path, "mkdir") as mk:
        raises(lambda: artifacts.save_bank(a, fake), FileExistsError)
        assert not op.called and not mk.called
    with mock.patch.object(Path, "exists", return_value=False), mock.patch.object(Path, "mkdir"), \
            mock.patch("builtins.open", side_effect=FileExistsError("race")) as op:
        raises(lambda: artifacts.save_bank(a, fake), FileExistsError, "race")
        assert op.call_args[0][1] == "xb"


@check("parameters: described without process addresses, callables by name, JSON without NaN")
def _():
    d = describe(make_estimator("logistic"))
    s = artifacts.dumps(d)
    assert "0x" not in s and "songpop.models:log_transform" in s and '"nan"' in s
    assert artifacts.dumps(describe(make_estimator("logistic"))) == s
    raises(lambda: artifacts.dumps({"x": float("nan")}), ValueError)


def _small_bank():
    """A real bank for the submission checks: logistic on F2 with one split seed (fast)."""
    X, X_test, groups, folds = _one_seed_inputs()
    bank = run_cv(X, DATA.y, DATA.row_ids, groups, folds, (155,), X_test, DATA.test_id,
                  lambda: make_estimator("logistic"))
    bank.metadata.update({"schema": BANK_SCHEMA, "run_id": "check", "recipe_id": "logistic_F2", "feature_set": "F2",
                          "features": list(X.columns), "deployment": "mean of fold models", "parameters": {},
                          "split_seeds": [155]})
    return bank


BANK = _small_bank()


@check("submission: a valid file is accepted; every defect is rejected; repeated pairs are really checked")
def _():
    good = artifacts.submission_bytes(DATA.test_id, BANK.test_mean)
    rep = artifacts.validate_submission(good, DATA, BANK)
    assert rep["rows"] == 863 and rep["repeated_uri_pairs_checked"] == 11 and rep["identical_feature_groups"] >= 11
    df = pd.read_csv(io.BytesIO(good))

    def bad(frame, match=None):
        b = frame.to_csv(index=False, float_format="%.17g").encode()
        raises(lambda: artifacts.validate_submission(b, DATA, BANK), ValueError, match)

    bad(df.rename(columns={"Popularity_Type": "target"}), "columns")
    bad(df.assign(ID=df["ID"].iloc[np.r_[1, 0, 2:len(df)]].to_numpy()), "order")
    bad(df.assign(ID=np.r_[df["ID"].iloc[:-1], df["ID"].iloc[0]]), "duplicate")
    bad(df.iloc[:-1], "rows")
    for v in [np.nan, np.inf]:
        bad(df.assign(Popularity_Type=np.r_[v, df["Popularity_Type"].iloc[1:]]), "non-finite")
    bad(df.assign(Popularity_Type=np.r_[1.5, df["Popularity_Type"].iloc[1:]]), "outside")
    bad(df.assign(Popularity_Type=df["Popularity_Type"] + 1e-6), "differ from the bank")
    # one member of a real repeated pair moved by 0.01, with a bank whose fold predictions agree with the file
    t = DATA.test.reset_index(drop=True)
    pair = next(list(g.index) for _, g in t.groupby("uri") if len(g) > 1)
    alt = artifacts.load_bank(artifacts.bank_bytes(BANK))
    delta = -0.01 if alt.test_by_fold[:, :, pair[0]].max() > 0.98 else 0.01
    alt.test_by_fold[:, :, pair[0]] += delta
    alt.test_mean = alt.test_by_fold.mean(axis=(0, 1))
    raises(lambda: artifacts.validate_submission(artifacts.submission_bytes(DATA.test_id, alt.test_mean), DATA, alt),
           ValueError, "identical features")
    wrong = artifacts.load_bank(artifacts.bank_bytes(BANK))
    wrong.metadata["features"] = []
    raises(lambda: artifacts.validate_submission(good, DATA, wrong), ValueError, "recorded features")


class _FakeFiles:
    """An in-memory stand-in for open(): exclusive writes store bytes, reads return them; corrupt changes the
    first prediction on the way to storage, to show that the read-back bytes are the ones validated."""

    def __init__(self, corrupt=False):
        self.files, self.corrupt, self.modes = {}, corrupt, []

    def open(self, path, mode="r", *args, **kwargs):
        self.modes.append(mode)
        store = self

        class _Writer:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def write(self, b):
                if store.corrupt:
                    lines = b.split(b"\n")
                    lines[1] = lines[1].split(b",")[0] + b",0.5"
                    b = b"\n".join(lines)
                store.files[str(path)] = b

        return _Writer() if "x" in mode else io.BytesIO(self.files[str(path)])


@check("submission writer: exclusive write, read-back validated, never over an existing file")
def _():
    fake = Path("/nonexistent-songpop-check/submission.csv")
    fs = _FakeFiles()
    with mock.patch.object(Path, "exists", return_value=False), mock.patch.object(Path, "mkdir"), \
            mock.patch("builtins.open", side_effect=fs.open):
        rep = artifacts.write_submission(fake, DATA, BANK)
    assert fs.modes == ["xb", "rb"] and rep["path"] == str(fake)
    assert rep["sha256"] == artifacts.sha256_bytes(fs.files[str(fake)])
    assert fs.files[str(fake)] == artifacts.submission_bytes(DATA.test_id, BANK.test_mean)
    bad = _FakeFiles(corrupt=True)
    with mock.patch.object(Path, "exists", return_value=False), mock.patch.object(Path, "mkdir"), \
            mock.patch("builtins.open", side_effect=bad.open):
        raises(lambda: artifacts.write_submission(fake, DATA, BANK), ValueError, "differ from the bank")
    with mock.patch.object(Path, "exists", return_value=True), mock.patch("builtins.open") as op:
        raises(lambda: artifacts.write_submission(fake, DATA, BANK), FileExistsError)
        assert not op.called


def main():
    width = max(len(n) for n, _, _ in RESULTS)
    for name, ok, msg in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:{width}s}  {msg}")
    failed = sum(not ok for _, ok, _ in RESULTS)
    print(f"{len(RESULTS) - failed} of {len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
