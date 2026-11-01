"""Small reproducible trials on saved grouped folds. No P2.9 integration.

Run: .venv/bin/python -m analysis.fast_exploration --batch features --workers 4
Each result includes OOF/test arrays, its config, warnings and runtime.
"""
import argparse
import concurrent.futures
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from analysis.fast_features import FastFeatures
from songpop.data import load_dataset, sha256_file
from songpop.evaluate import PredictionBank, validate_bank, validate_cv_inputs
from songpop.phase2.encoders import FitContext, inner_group_splits
from songpop.phase2.engine import raw_rows
from songpop.phase2.models import make_preprocessing, make_estimator, prediction_threads, prediction_view
from songpop.phase2.tabm import TabMClassifier
from songpop.splits import FoldSet, SPLIT_SEEDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/fast"
NOTES = ROOT / "notes/2026-11-01-fast-exploration.md"
CUTOFF = datetime(2026, 11, 1, 20, tzinfo=timezone.utc)
BEST_LGB = "p2_p24_rt_0ec2239a_b1_006_seeds155-2026-1029_af202d6f369b"
WEIGHTS = (0.10, 0.25, 0.50)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in PredictionBank.ARRAYS}


def references():
    menu = json.loads((ROOT / "configs/phase2/menu.json").read_text())
    refs = {c["id"]: load_arrays(ROOT / "artifacts/phase2/bank" / (c["run_id"] + ".npz"))
            for c in menu["candidates"] if c["id"] in ("blend_eq_all", "constrained_l0.1")}
    current = OUT / "final_candidates/combined/predictions.npz"
    if current.exists():
        refs["fast_combined"] = load_arrays(current)
    return refs


def data_inputs():
    data = load_dataset(ROOT)
    fs = FoldSet.from_frame(pd.read_csv(ROOT / "artifacts/splits/production_v1.csv"))
    rows = raw_rows(data.train_eligible, data.row_ids)
    test = raw_rows(data.test, data.test_id)
    groups, folds = fs.primary_group[data.eligible], fs.primary[:, data.eligible]
    if not np.array_equal(fs.eligible, data.eligible):
        raise ValueError("saved eligible rows differ from data")
    validate_cv_inputs(rows, data.y, data.row_ids, groups, folds, SPLIT_SEEDS, test, data.test_id)
    for ref in references().values():
        for key, actual in {"train_row_id": data.row_ids, "y": data.y, "group_id": groups,
                            "fold_id": folds, "split_seeds": SPLIT_SEEDS, "test_id": data.test_id}.items():
            if not np.array_equal(ref[key], actual):
                raise ValueError(f"new and historical banks differ in {key}")
    return data, rows, test, groups, folds


def score(y, folds, predictions):
    return np.array([[roc_auc_score(y[folds[r] == k], predictions[r, folds[r] == k])
                      for k in range(5)] for r in range(3)])


def summaries(arrays, refs):
    out = []
    baseline = float(refs["blend_eq_all"]["fold_auc"].mean())
    for ref_id, ref in [("standalone", None)] + list(refs.items()):
        for weight in ((1.0,) if ref is None else WEIGHTS):
            pred = arrays["oof"] if ref is None else (1 - weight) * ref["oof"] + weight * arrays["oof"]
            auc = score(arrays["y"], arrays["fold_id"], pred)
            base_auc = baseline if ref is None else float(ref["fold_auc"].mean())
            out.append({"reference": ref_id, "weight": weight, "cv": float(auc.mean()),
                        "per_seed": auc.mean(axis=1).tolist(), "fold_auc": auc.tolist(),
                        "delta_E": float(auc.mean() - baseline), "delta_reference": float(auc.mean() - base_auc)})
    return out


def model(config, columns, groups, y, ctx, seed):
    family, params = config["family"], config["params"]
    if family == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params, random_state=seed, n_jobs=4, deterministic=True,
                              force_col_wise=True, verbosity=-1)
    if family == "hist_gb":
        return HistGradientBoostingClassifier(**params, random_state=seed, early_stopping=False)
    if family in ("xgboost", "catboost", "extra_trees"):
        return make_estimator(family, params, seed, columns)
    if family == "gaussian_process":
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, Matern
        scale = params["length_scale"]
        kernel = RBF(scale, length_scale_bounds="fixed") if params["kernel"] == "rbf" else \
            Matern(scale, length_scale_bounds="fixed", nu=1.5)
        # Fixed kernels bound the work; Laplace inference uses training rows only.
        prep = make_preprocessing({"name": "log_standard"}, columns)
        return make_pipeline(prep, GaussianProcessClassifier(kernel=kernel, optimizer=None,
                             max_iter_predict=50, random_state=seed, n_jobs=1))
    if family == "mlp":
        if config.get("scaler") == "quantile":
            prep = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                 QuantileTransformer(n_quantiles=256, output_distribution="normal", random_state=0),
                                 StandardScaler())
        else:
            prep = make_preprocessing({"name": "log_standard"}, columns)
        return make_pipeline(prep, MLPClassifier(**params, random_state=seed, early_stopping=False))
    if family == "spline":
        prep = make_preprocessing({"name": "spline_calendar", "n_knots": config["knots"]}, columns)
        return make_pipeline(prep, LogisticRegression(**params, solver="lbfgs", max_iter=3000, tol=1e-6))
    if family == "svm":
        # Only stateless features enter this pipeline. Calibration refits the
        # imputer/scaler and SVC inside each grouped calibration-training fold.
        opts = config["features"]
        if set(opts.get("blocks", [])) & {"date_count", "date_te3", "date_te10", "date_te30", "era_relative"} \
                or any(opts.get(k) for k in ("hier", "date_counts", "year_relative")):
            raise ValueError("SVM calibration requires stateless input features")
        prep = make_preprocessing({"name": "log_standard"}, columns)
        base = make_pipeline(prep, SVC(**params, kernel="rbf", probability=False,
                                      cache_size=512, max_iter=100000, random_state=seed))
        cv = inner_group_splits(y, groups, ctx.inner_seed("fast_svm_calibration"), n_splits=3)
        return CalibratedClassifierCV(base, method="sigmoid", cv=cv, ensemble=False, n_jobs=1)
    if family == "tabm":
        if config["features"].get("blocks") or any(config["features"].get(k) for k in ("hier", "date_counts", "year_relative")):
            raise ValueError("legacy TabM stopping receives only stateless features")
        return TabMClassifier(**params, scaler=config.get("scaler", "standard"), random_state=seed,
                              threads=4, max_epochs=300, patience=20, k=16, inner_folds=5, tolerance=1e-6)
    raise ValueError(f"unknown family {family}")


def run_trial(config):
    trial_id = config["id"]
    folder = OUT / trial_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        existing = json.loads((folder / "result.json").read_text())
        if existing["config"] != config:
            raise ValueError(f"refusing to reuse {trial_id} with a different config")
        return existing
    cutoff = min(CUTOFF, datetime.fromisoformat(config.get("cutoff_utc", CUTOFF.isoformat())))
    if datetime.now(timezone.utc) >= cutoff:
        return {"id": trial_id, "status": "not_started_cutoff"}
    start = time.monotonic()
    write_json(folder / "config.json", config)
    sources = {str(p.relative_to(ROOT)): sha256_file(p) for p in
               (Path(__file__), ROOT / "analysis/fast_features.py")}
    print(f"START {trial_id} {datetime.now(timezone.utc).isoformat()}", flush=True)
    data, rows, test, groups, folds = data_inputs()
    oof = np.full(folds.shape, np.nan)
    test_by_fold = np.full((3, 5, len(test)), np.nan)
    fit_info, warning_text = [], set()
    with threadpool_limits(limits=4):
        for r, split_seed in enumerate(SPLIT_SEEDS):
            for k in range(5):
                if datetime.now(timezone.utc) >= cutoff:
                    raise TimeoutError("experiment cutoff reached; incomplete result is not scored")
                tr, va = np.flatnonzero(folds[r] != k), np.flatnonzero(folds[r] == k)
                assert not set(groups[tr]) & set(groups[va])
                ctx = FitContext(split_seed, k)
                features = FastFeatures(config["features"])
                X = features.fit_transform(rows.iloc[tr], data.y[tr], groups[tr], ctx)
                V, T = features.transform(rows.iloc[va]), features.transform(test)
                vp, tp = [], []
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    for seed in config.get("estimator_seeds", [0]):
                        if datetime.now(timezone.utc) >= cutoff:
                            raise TimeoutError("experiment cutoff reached before the next fit")
                        est = model(config, list(X.columns), groups[tr], data.y[tr], ctx, seed)
                        if config["family"] == "tabm":
                            est.fit(X, data.y[tr], groups=groups[tr], context=ctx)
                        else:
                            est.fit(X, data.y[tr])
                        if list(est.classes_) != [0, 1]:
                            raise ValueError("unexpected class orientation")
                        with prediction_threads(config["family"], [est]):
                            predictor = prediction_view(config["family"], est)
                            vp.append(predictor.predict_proba(V)[:, 1])
                            tp.append(predictor.predict_proba(T)[:, 1])
                        inner = est.steps[-1][1] if hasattr(est, "steps") else est
                        info = {"split_seed": split_seed, "fold": k, "estimator_seed": seed,
                                "train_rows": len(tr), "validation_rows": len(va)}
                        if hasattr(inner, "n_iter_"):
                            info["n_iter"] = np.asarray(inner.n_iter_).tolist()
                        if hasattr(inner, "receipt_"):
                            info["tabm_stopping"] = inner.receipt_
                        fit_info.append(info)
                    warning_text.update(f"{type(w.message).__name__}: {w.message}" for w in caught)
                oof[r, va], test_by_fold[r, k] = np.mean(vp, axis=0), np.mean(tp, axis=0)
                write_json(folder / "progress.json", {"completed_folds": r * 5 + k + 1,
                           "elapsed_seconds": time.monotonic() - start, "pid": os.getpid()})
    auc = score(data.y, folds, oof)
    bank = PredictionBank(data.row_ids, data.y, groups, np.array(SPLIT_SEEDS), folds, oof,
                          data.test_id, test_by_fold, test_by_fold.mean(axis=(0, 1)), auc)
    validate_bank(bank)  # cheap shapes, group separation, probabilities and score check
    arrays = {key: getattr(bank, key) for key in PredictionBank.ARRAYS}
    np.savez_compressed(folder / "predictions.npz", **arrays)
    result = {"id": trial_id, "status": "complete", "config": config,
              "cv": float(auc.mean()), "per_seed": auc.mean(axis=1).tolist(),
              "elapsed_seconds": time.monotonic() - start, "fit_info": fit_info,
              "feature_columns": list(X.columns), "warnings": sorted(warning_text),
              "data_hashes": data.hashes, "split_sha256": sha256_file(ROOT / "artifacts/splits/production_v1.csv"),
              "code_sha256": sources, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("numpy", "pandas", "scikit-learn", "lightgbm") +
                           ((config["family"],) if config["family"] in ("xgboost", "catboost") else ())},
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "prediction_sha256": sha256_file(folder / "predictions.npz"),
              "comparisons": summaries(arrays, references())}
    write_json(folder / "result.json", result)
    best = max(result["comparisons"], key=lambda x: x["cv"])
    print(f"DONE {trial_id} CV={result['cv']:.6f} best={best['cv']:.6f} delta_E={best['delta_E']:+.6f} seconds={result['elapsed_seconds']:.1f}", flush=True)
    return result


def feature_configs():
    with np.load(ROOT / "artifacts/phase2/bank" / (BEST_LGB + ".npz")) as z:
        recipe = json.loads(str(z["metadata_json"]))["recipe"]
    cases = [("control", {}), ("round", {"round": True}),
             ("round_detail", {"round": True, "round_detail": True}),
             ("hier3", {"hier": [30, 10, 3]}), ("hier10", {"hier": [100, 30, 10]}),
             ("hier_replace", {"blocks": ["date_count"], "hier": [30, 10, 3]}),
             ("date_counts", {"date_counts": True}), ("year_relative", {"year_relative": True}),
             ("audio_extra", {"audio_extra": True}),
             ("cyclic_music_round", {"blocks": ["date_count", "date_te3", "cyclic", "music"], "round": True})]
    out = [{"id": "lgb_" + name, "family": "lightgbm", "params": recipe["params"],
            "features": opts, "estimator_seeds": [0]} for name, opts in cases]
    for leaf, bins in [(3, 255), (10, 1023)]:
        out.append({"id": f"lgb_round_leaf{leaf}_bins{bins}", "family": "lightgbm",
                    "params": dict(recipe["params"], min_child_samples=leaf, max_bin=bins),
                    "features": {"round": True}, "estimator_seeds": [0]})
    return out


def model_configs():
    out = []
    for width, alpha, epochs, scaler in [(64, .1, 100, "standard"), (64, 1., 300, "standard"),
                                        (128, .1, 100, "standard"), (128, 1., 300, "standard"),
                                        (64, .1, 200, "quantile"), (128, 1., 200, "quantile")]:
        out.append({"id": f"mlp_w{width}_a{alpha}_e{epochs}_{scaler}", "family": "mlp",
                    "params": {"hidden_layer_sizes": [width, width // 2], "alpha": alpha,
                               "max_iter": epochs, "learning_rate_init": .001, "batch_size": 128,
                               "tol": 1e-5, "n_iter_no_change": 30},
                    "features": {"blocks": ["date_count", "date_te3", "cyclic"], "round": True},
                    "scaler": scaler, "estimator_seeds": [0]})
    for c in (.3, 3., 30.):
        for gamma in (.01, .05):
            out.append({"id": f"svm_c{c}_g{gamma}", "family": "svm", "params": {"C": c, "gamma": gamma},
                        "features": {"blocks": ["cyclic", "music"], "round": True}, "estimator_seeds": [0]})
    for knots in (5, 8):
        for c in (.01, .1, 1.):
            out.append({"id": f"spline_k{knots}_c{c}", "family": "spline", "params": {"C": c},
                        "knots": knots, "features": {"blocks": ["date_count", "date_te3", "cyclic"], "round": True},
                        "estimator_seeds": [0]})
    for leaves in (15, 31):
        for l2 in (1., 10.):
            out.append({"id": f"hist_leaves{leaves}_l2{l2}", "family": "hist_gb",
                        "params": {"max_iter": 400, "learning_rate": .03, "max_leaf_nodes": leaves,
                                   "min_samples_leaf": 20, "l2_regularization": l2},
                        "features": {"round": True}, "estimator_seeds": [0]})
    for scaler, width in [("standard", 128), ("quantile", 128), ("quantile", 256)]:
        out.append({"id": f"tabm_{scaler}_w{width}", "family": "tabm",
                    "params": {"batch_size": 128, "d_block": width, "dropout": .2,
                               "learning_rate": .001, "n_blocks": 2, "weight_decay": .01},
                    "features": {"blocks": [], "round": True}, "scaler": scaler, "estimator_seeds": [0]})
    return out


def render_table():
    rows = []
    result_files = list(OUT.glob("*/result.json")) + list((ROOT / "artifacts/neural").glob("f2_*/result.json"))
    for file in sorted(result_files):
        result = json.loads(file.read_text())
        for c in result["comparisons"]:
            label = result["id"] if c["reference"] == "standalone" else f"{c['reference']} + {c['weight']:.2f} {result['id']}"
            rows.append((label, c, result["elapsed_seconds"]))
    text = NOTES.read_text()
    before, after = text.split("## Results\n", 1)
    _, checkpoint = after.split("## Checkpoint\n", 1)
    refs = references()
    current_cv = float(refs.get("fast_combined", refs["blend_eq_all"])["fold_auc"].mean())
    table = "\n| Model or feature change | CV mean | Seed 155 | Seed 2026 | Seed 1029 | Change vs E | Change vs combined | Runtime |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for name, a in refs.items():
        cv = a["fold_auc"]
        delta = float(cv.mean() - references()["blend_eq_all"]["fold_auc"].mean())
        table += f"| Reference {name} | {cv.mean():.6f} | " + " | ".join(f"{v:.6f}" for v in cv.mean(axis=1)) + f" | {delta:+.6f} | {cv.mean()-current_cv:+.6f} | existing |\n"
    for label, c, seconds in rows:
        table += f"| {label} | {c['cv']:.6f} | " + " | ".join(f"{v:.6f}" for v in c["per_seed"]) + f" | {c['delta_E']:+.6f} | {c['cv']-current_cv:+.6f} | {seconds:.1f} s |\n"
    NOTES.write_text(before + "## Results\n" + table + "\nRuntime is the model's complete 15-fold fit, shared by its blend rows.\n\n## Checkpoint\n" + checkpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=["features", "models"])
    ap.add_argument("--queue", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        render_table()
        return
    configs = json.loads(args.queue.read_text()) if args.queue else feature_configs() if args.batch == "features" else model_configs()
    write_json(OUT / ((args.batch or args.queue.stem) + "_configs.json"), configs)
    data_inputs()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_trial, c): c for c in configs}
        for f in concurrent.futures.as_completed(futures):
            config = futures[f]
            try:
                f.result()
            except Exception:
                error = traceback.format_exc()
                write_json(OUT / config["id"] / "error.json", {"config": config, "error": error})
                print(f"FAILED {config['id']}\n{error}", flush=True)
            render_table()


if __name__ == "__main__":
    main()
