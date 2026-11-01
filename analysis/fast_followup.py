"""Bounded follow-up to the first fast pass; uses the existing CV runner."""
import copy
import json

import numpy as np

from analysis.fast_exploration import ROOT, OUT, data_inputs, model, write_json
from analysis.fast_features import FastFeatures
from songpop.phase2.encoders import FitContext
from songpop.phase2.models import prediction_threads, prediction_view

CUTOFF = "2026-11-01T06:31:00+00:00"


def configurations():
    lgb = json.loads((OUT / "lgb_year_counts_seeds012/config.json").read_text())
    mlp = json.loads((OUT / "mlp_w128_a1.0_e300_standard_seeds012/config.json").read_text())
    queue = []

    def add(name, family, params, features, **extra):
        queue.append({"id": "f2_" + name, "family": family, "params": copy.deepcopy(params),
                      "features": copy.deepcopy(features), "estimator_seeds": [0],
                      "cutoff_utc": CUTOFF, **extra})

    changes = [
        {"num_leaves": 31}, {"num_leaves": 127},
        {"min_child_samples": 5}, {"min_child_samples": 20},
        {"reg_lambda": 0.0}, {"reg_lambda": 2.0}, {"reg_lambda": 10.0},
        {"colsample_bytree": .5}, {"colsample_bytree": .9},
        {"subsample": 1.0}, {"n_estimators": 1000}, {"n_estimators": 1500},
        {"learning_rate": .02, "n_estimators": 600},
        {"learning_rate": .005, "n_estimators": 1600},
        {"num_leaves": 127, "min_child_samples": 5, "reg_lambda": 2.0},
        {"num_leaves": 31, "min_child_samples": 5, "n_estimators": 1000},
    ]
    for i, change in enumerate(changes):
        add(f"lgb_tune{i:02d}", "lightgbm", dict(lgb["params"], **change), lgb["features"])
    variations = [("shrink5", {"year_shrinkage": 5}), ("shrink100", {"year_shrinkage": 100}),
                  ("window1", {"year_window": 1}), ("window2", {"year_window": 2}),
                  ("zscore", {"year_standardize": True}),
                  ("window2_zscore", {"year_window": 2, "year_standardize": True})]
    for name, change in variations:
        add("lgb_" + name, "lightgbm", lgb["params"], dict(lgb["features"], **change))

    menu = json.loads((ROOT / "configs/phase2/menu.json").read_text())
    for family in ("xgboost", "catboost", "extra_trees"):
        recipe = next(v for v in menu["atom_recipes"].values() if v["family"] == family)
        features = dict(lgb["features"], blocks=list(dict.fromkeys(recipe["blocks"] + ["date_count", "date_te3"])))
        if family == "xgboost":
            changes = [{}, {"max_depth": 3}, {"max_depth": 5, "reg_lambda": 5.0},
                       {"max_depth": 4, "n_estimators": 1500, "learning_rate": .02}]
        elif family == "catboost":
            changes = [{}, {"depth": 4, "learning_rate": .03},
                       {"depth": 5, "l2_leaf_reg": 6.0, "learning_rate": .03},
                       {"depth": 8, "l2_leaf_reg": 6.0}]
        else:
            changes = [{}, {"min_samples_leaf": 1, "max_features": .7},
                       {"min_samples_leaf": 4}, {"min_samples_leaf": 2, "max_features": .7}]
        for i, change in enumerate(changes):
            add(f"{family}_{i}", family, dict(recipe["params"], **change), features)

    for width in (64, 128):
        for epochs, lr in ((600, .001), (1000, .0003)):
            p = dict(mlp["params"], hidden_layer_sizes=[width, width // 2], max_iter=epochs, learning_rate_init=lr)
            add(f"mlp_w{width}_e{epochs}", "mlp", p, mlp["features"], scaler="standard")
        p = dict(mlp["params"], hidden_layer_sizes=[width, width // 2], activation="tanh", max_iter=600)
        add(f"mlp_tanh{width}", "mlp", p, mlp["features"], scaler="standard")
        p = dict(mlp["params"], hidden_layer_sizes=[width, width // 2], max_iter=600)
        add(f"mlp_year{width}", "mlp", p, dict(mlp["features"], **lgb["features"]), scaler="standard")
    for width, activation, alpha in ((32, "tanh", 1.), (64, "tanh", 1.),
                                     (64, "relu", 1.), (64, "tanh", 10.)):
        p = {"hidden_layer_sizes": [width], "activation": activation, "solver": "lbfgs",
             "alpha": alpha, "max_iter": 600, "max_fun": 6000, "tol": 1e-5}
        add(f"mlp_lbfgs{width}_{activation}_a{alpha:g}", "mlp", p, mlp["features"], scaler="standard")
    # Sam asked about SVM, sophisticated neural nets and Gaussian processes.
    # Trade eight tuning cases for ten broader cases, retaining the time cap.
    omit = {f"f2_lgb_tune{i:02d}" for i in (3, 6, 8, 9, 12, 13)}
    omit |= {"f2_catboost_3", "f2_extra_trees_2"}
    queue = [c for c in queue if c["id"] not in omit]
    for c, gamma in ((3., .001), (30., .001), (30., .003), (300., .001)):
        add(f"svm_c{c:g}_g{gamma:g}", "svm", {"C": c, "gamma": gamma},
            {"blocks": ["cyclic", "music"], "round": True})
    for kernel in ("rbf", "matern"):
        for length in (3., 10.):
            add(f"gp_{kernel}{length:g}", "gaussian_process", {"kernel": kernel, "length_scale": length},
                dict(mlp["features"], **lgb["features"]))
    for blocks, dropout, lr in ((3, .1, .001), (2, .1, .0003)):
        add(f"tabm_b{blocks}_lr{lr:g}", "tabm", {"batch_size": 128, "d_block": 128,
            "dropout": dropout, "learning_rate": lr, "n_blocks": blocks, "weight_decay": .01},
            {"blocks": [], "round": True}, scaler="quantile")
    assert len(queue) == 48 and len({c["id"] for c in queue}) == 48
    # Start longer neural runs first, leaving four slots for the other families.
    return sorted(queue, key=lambda c: c["family"] != "tabm")


def check_boundary(queue):
    data, rows, test, groups, folds = data_inputs()
    tr = np.flatnonzero(folds[0] != 0)
    ctx = FitContext(155, 0)
    variants = [c for c in queue if any(k in c["features"] for k in
                                     ("year_shrinkage", "year_window", "year_standardize"))]
    assert len(variants) == 6
    for config in variants:
        f = FastFeatures(config["features"])
        X = f.fit_transform(rows.iloc[tr], data.y[tr], groups[tr], ctx)
        before = f.transform(test.iloc[:10])
        poisoned = test.iloc[:10].copy()
        poisoned.loc[:, "duration_ms"] = 1e12
        f.transform(poisoned)
        assert f.transform(test.iloc[:10]).equals(before)
        assert list(X.columns) == list(before.columns)
        assert not np.isinf(X.to_numpy()).any()
        assert not np.isinf(before.to_numpy()).any()
        # Alter training moments as a positive control, with unchanged queries.
        changed = rows.iloc[tr].copy()
        changed.loc[:, "duration_ms"] *= 2
        g = FastFeatures(config["features"])
        g.fit_transform(changed, data.y[tr], groups[tr], ctx)
        assert not np.array_equal(before["duration_ms_year_relative"].to_numpy(),
                                  g.transform(test.iloc[:10])["duration_ms_year_relative"].to_numpy())
    # Only a few trees: check existing factory and prediction-thread routing.
    for family in ("xgboost", "catboost", "extra_trees"):
        c = copy.deepcopy(next(c for c in queue if c["family"] == family))
        c["params"]["iterations" if family == "catboost" else "n_estimators"] = 2
        f = FastFeatures(c["features"])
        X = f.fit_transform(rows.iloc[tr], data.y[tr], groups[tr], ctx)
        V = f.transform(rows.iloc[np.flatnonzero(folds[0] == 0)])
        est = model(c, list(X.columns), groups[tr], data.y[tr], ctx, 0)
        est.fit(X, data.y[tr])
        with prediction_threads(family, [est]):
            p = prediction_view(family, est).predict_proba(V)
        assert list(est.classes_) == [0, 1] and p.shape == (len(V), 2)
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    c = next(c for c in queue if c["family"] == "gaussian_process")
    f = FastFeatures(c["features"])
    X = f.fit_transform(rows.iloc[tr], data.y[tr], groups[tr], ctx)
    est = model(c, list(X.columns), groups[tr], data.y[tr], ctx, 0)
    est.fit(X.iloc[:200], data.y[tr][:200])
    p = est.predict_proba(X.iloc[200:210])
    assert p.shape == (10, 2) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    print("PASS: saved folds; six training-only year-statistic variants; query poisoning and positive controls; three existing tree factories")


if __name__ == "__main__":
    queue = configurations()
    check_boundary(queue)
    write_json(OUT / "followup_queue.json", queue)
    print(f"Saved {len(queue)} configurations; cutoff {CUTOFF}")
