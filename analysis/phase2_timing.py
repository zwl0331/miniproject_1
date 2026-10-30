"""Representative runtimes for the phase 2 search budgets (task P2.0b, Finding 010). Runs recipes through the reviewed
P2.1 engine in memory on the saved folds and all three split seeds, and writes nothing into the project. -B keeps
Python from writing bytecode (-I ignores PYTHONDONTWRITEBYTECODE). To verify, from the project root:

    .venv/bin/python -B -I analysis/phase2_timing.py .

The saved output was made by redirecting that command into analysis/phase2_timing_output.txt. The printed mean fold
AUCs are probe exposures, kept because they were already seen; they select no recipe and allocate no batch.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

from songpop import artifacts  # noqa: E402
from songpop.features import make_features  # noqa: E402
from songpop.phase2.banks import source_manifest  # noqa: E402
from songpop.phase2.engine import run_recipe_cv  # noqa: E402
from songpop.phase2.models import log_columns  # noqa: E402
from songpop.phase2.recipes import load_recipe, validate_recipe  # noqa: E402
from songpop.phase2.runs import Context  # noqa: E402
from songpop.splits import SPLIT_SEEDS  # noqa: E402

ctx = Context.load(ROOT)
anchor = load_recipe(ROOT / "configs/phase2/recipes/lightgbm_F2_anchor.json")


def r(**over):
    x = dict(anchor)
    x.update(over)
    return validate_recipe(x)


LGB = anchor["params"]
CASES = [
    ("lightgbm anchor (600 trees, 15 leaves)", r()),
    ("lightgbm 300 trees, 7 leaves, lr 0.05", r(params=dict(LGB, n_estimators=300, num_leaves=7, learning_rate=0.05))),
    ("lightgbm 1200 trees, 31 leaves", r(params=dict(LGB, n_estimators=1200, num_leaves=31))),
    ("lightgbm anchor + date_count + date_te10", r(blocks=["date_count", "date_te10"])),
    ("lightgbm anchor + every block", r(blocks=["date_count", "date_te10", "interactions", "cyclic", "era_relative"])),
    ("lightgbm anchor, 3 estimator seeds", r(estimator_seeds=[0, 1, 2])),
    ("random forest 500 trees, leaf 3, sqrt", r(family="random_forest", params={"n_estimators": 500, "min_samples_leaf": 3})),
    ("random forest 500 trees, leaf 1, all features", r(family="random_forest",
                                                        params={"n_estimators": 500, "min_samples_leaf": 1, "max_features": 1.0})),
    ("extra trees 500 trees, leaf 2, all features", r(family="extra_trees",
                                                      params={"n_estimators": 500, "min_samples_leaf": 2, "max_features": 1.0})),
    ("xgboost 1000 trees, depth 4, lr 0.03", r(family="xgboost", params={"n_estimators": 1000, "max_depth": 4,
                                                                          "learning_rate": 0.03, "subsample": 0.8,
                                                                          "colsample_bytree": 0.8})),
    ("histgb 300 iterations, 31 leaves", r(family="hist_gb", params={"max_iter": 300, "max_leaf_nodes": 31,
                                                                       "min_samples_leaf": 20, "learning_rate": 0.05})),
    ("logistic, spline 7 knots, F1", r(family="logistic", params={"C": 1.0}, base="F1",
                                       preprocessing={"name": "spline", "n_knots": 7})),
    ("logistic, log_standard, F2", r(family="logistic", params={"C": 1.0}, preprocessing={"name": "log_standard"})),
]


def fits(recipe):
    """Outer fit positions (3 split seeds x 5 folds) and estimator fits (times the estimator seeds)."""
    return {"outer_fit_positions": 15, "estimator_fits": 15 * len(recipe["estimator_seeds"])}


def time_case(recipe):
    t0 = time.perf_counter()
    bank = run_recipe_cv(recipe, ctx.rows, ctx.data.y, ctx.data.row_ids, ctx.groups, ctx.fold_id(list(SPLIT_SEEDS)),
                         list(SPLIT_SEEDS), ctx.test_rows, ctx.data.test_id)
    return time.perf_counter() - t0, float(bank.fold_auc.mean())


def knn_case(k, weights):
    """kNN on the ten continuous audio features and year, log transforms, imputation and scaling fitted per fold."""
    cols = ["year", "speechiness", "danceability", "duration_ms", "energy", "instrumentalness", "valence", "tempo",
            "loudness", "acousticness", "liveness"]
    X = make_features(ctx.data.train_eligible, "F1")[cols]
    y, fold_id = ctx.data.y, ctx.fold_id(list(SPLIT_SEEDS))
    t0, aucs = time.perf_counter(), []
    for rep in range(3):
        for f in range(5):
            tr, va = fold_id[rep] != f, fold_id[rep] == f
            m = Pipeline([("log", FunctionTransformer(log_columns)), ("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights,
                                                                                      n_jobs=4))])
            m.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
            m.predict_proba(make_features(ctx.data.test, "F1")[cols])
    return time.perf_counter() - t0, float(np.mean(aucs))


out = {"python": platform.python_version(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
       "threads_per_model": 4, "split_seeds": list(SPLIT_SEEDS), "data_hashes": ctx.data.hashes,
       "split_sha256": ctx.split_sha256, "packages": artifacts.packages(), "source_manifest": source_manifest(ROOT),
       "note": "probe_auc_exposure values select nothing (Finding 010)", "cases": []}
print("seconds  probe AUC  case (15 outer fit positions; estimator fits in the JSON)")
for name, recipe in CASES:
    secs, auc = time_case(recipe)
    out["cases"].append({"case": name, "recipe": recipe, **fits(recipe), "seconds": round(secs, 2),
                         "probe_auc_exposure": round(auc, 4)})
    print(f"{secs:7.2f}  {auc:.4f}  {name}", flush=True)
for k, w in [(50, "distance"), (100, "uniform")]:
    secs, auc = knn_case(k, w)
    name = (f"knn k={k}, {w} weights, eleven standardized continuous columns (a proxy: not the block-weighted "
            f"representation of plan section 6.2)")
    out["cases"].append({"case": name, "recipe": None, "outer_fit_positions": 15, "estimator_fits": 15,
                         "seconds": round(secs, 2), "probe_auc_exposure": round(auc, 4)})
    print(f"{secs:7.2f}  {auc:.4f}  {name}", flush=True)
print(json.dumps(out, indent=1))
