"""Broader EBM and training-only neighborhood feature trials for fast takeover."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import subprocess
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from analysis.fast_exploration import data_inputs, references, load_arrays, score, summaries, write_json, model
from analysis.fast_features import FastFeatures
from songpop.data import AUDIO, sha256_file
from songpop.evaluate import PredictionBank, validate_bank
from songpop.features import make_features
from songpop.phase2.encoders import FitContext, inner_group_splits
from songpop.phase2.models import prediction_threads, prediction_view
from songpop.splits import SPLIT_SEEDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/extended"
CUTOFF = datetime(2026, 11, 1, 20, tzinfo=timezone.utc)


def current_references():
    return dict(references(), followup_trees=load_arrays(ROOT / "artifacts/fast/f2_blend_z5_et25/predictions.npz"))


class Neighbors:
    """OOF local target means and learned-cluster summaries. All fits use raw rows."""
    def __init__(self, view="audio", clusters=16):
        self.view, self.clusters = view, clusters

    def matrix(self, rows):
        return rows[AUDIO] if self.view == "audio" else make_features(rows, "F2")

    def fit(self, rows, y):
        self.prep = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        Z = self.prep.fit_transform(self.matrix(rows))
        self.nn = NearestNeighbors(n_neighbors=min(100, len(Z)), n_jobs=1).fit(Z)
        self.y = np.asarray(y)
        self.km = KMeans(n_clusters=self.clusters, random_state=155, n_init=5).fit(Z)
        self.count = np.bincount(self.km.labels_, minlength=self.clusters)
        total = np.bincount(self.km.labels_, weights=y, minlength=self.clusters)
        self.posterior = (total + 20 * np.mean(y)) / (self.count + 20)
        return self

    def transform(self, rows):
        Z = self.prep.transform(self.matrix(rows))
        distance, indices = self.nn.kneighbors(Z)
        out = {}
        for k in (10, 30, 100):
            k = min(k, len(self.y))
            out[f"neighbor_mean{k}"] = self.y[indices[:, :k]].mean(axis=1)
            weights = 1 / (distance[:, :k] + 0.1)
            out[f"neighbor_weighted{k}"] = (self.y[indices[:, :k]] * weights).sum(axis=1) / weights.sum(axis=1)
            out[f"neighbor_distance{k}"] = distance[:, :k].mean(axis=1)
        cluster = self.km.predict(Z)
        out.update(cluster_posterior=self.posterior[cluster], cluster_log_count=np.log1p(self.count[cluster]),
                   cluster_distance=np.linalg.norm(Z - self.km.cluster_centers_[cluster], axis=1))
        return pd.DataFrame(out, index=rows.index)

    def fit_transform(self, rows, y, groups, context):
        values = None
        for tr, va in inner_group_splits(y, groups, context.inner_seed("extended_neighbors")):
            fitted = Neighbors(self.view, self.clusters).fit(rows.iloc[tr], np.asarray(y)[tr])
            block = fitted.transform(rows.iloc[va])
            if values is None:
                values = pd.DataFrame(np.nan, index=rows.index, columns=block.columns)
            values.iloc[va] = block.to_numpy()
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("neighborhood cross-fitting did not cover every row")
        self.fit(rows, y)
        return values


def fit_predict(config, rows, y, groups, context, queries, seed):
    features = FastFeatures(config["features"])
    X = features.fit_transform(rows, y, groups, context)
    query_X = [features.transform(q) for q in queries]
    if config.get("neighbors"):
        neighbors = Neighbors(**config["neighbors"])
        X = pd.concat([X, neighbors.fit_transform(rows, y, groups, context)], axis=1)
        query_X = [pd.concat([x, neighbors.transform(q)], axis=1) for x, q in zip(query_X, queries)]
    family, params = config["family"], dict(config["params"])
    info = {}
    if family == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        types = ["nominal" if c in ("key", "mode", "time_signature") else "continuous" for c in X]
        estimator = ExplainableBoostingClassifier(**params, feature_types=types, validation_size=0,
            early_stopping_rounds=0, n_jobs=4, random_state=seed)
        # Bootstrap whole tracks; no OOB stopping or OOB score is used. Every
        # transform and bag is confined to the permitted outer fitting rows.
        unique, inverse = np.unique(groups, return_inverse=True)
        rng = np.random.default_rng(context.inner_seed("ebm_bags") + seed)
        bags = np.column_stack([np.bincount(rng.integers(len(unique), size=len(unique)),
                   minlength=len(unique))[inverse] for _ in range(params["outer_bags"])]).astype(np.int8)
        estimator.fit(X, y, bags=bags)
        info = {"terms": len(estimator.term_features_), "bags": params["outer_bags"]}
        output = [estimator.predict_proba(q)[:, 1] for q in query_X]
    elif family == "pairlogit":
        from catboost import CatBoostRanker
        estimator = CatBoostRanker(**params, loss_function="PairLogit:max_pairs=50000", random_seed=seed,
            thread_count=4, verbose=False, allow_writing_files=False)
        # A ranking query covers this training partition. Track-group exclusion
        # is imposed by the outer CV split; no pairs contain held-out rows.
        estimator.fit(X, y, group_id=np.zeros(len(y), dtype=int))
        output = [expit(estimator.predict(q)) for q in query_X]
        info = {"score_link": "fixed sigmoid, no fitted calibration"}
    else:
        estimator = model(config, list(X.columns), groups, y, context, seed)
        estimator.fit(X, y)
        with prediction_threads(family, [estimator]):
            estimator = prediction_view(family, estimator)
            output = [estimator.predict_proba(q)[:, 1] for q in query_X]
    if any(not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in output):
        raise ValueError("invalid model probabilities")
    return output, info


def run_trial(config):
    folder = OUT / config["id"]
    if (folder / "result.json").exists():
        old = json.loads((folder / "result.json").read_text())
        if old["config"] != config:
            raise ValueError("trial id already belongs to another config")
        return old
    if datetime.now(timezone.utc) >= CUTOFF:
        return {"id": config["id"], "status": "not_started_cutoff"}
    start = time.monotonic()
    print(f"START {config['id']}", flush=True)
    write_json(folder / "config.json", config)
    data, rows, test, groups, folds = data_inputs()
    oof, test_by_fold = np.full(folds.shape, np.nan), np.full((3, 5, len(test)), np.nan)
    warnings_seen, fit_info = set(), []
    with threadpool_limits(limits=4):
        for r, split_seed in enumerate(SPLIT_SEEDS):
            for fold in range(5):
                if datetime.now(timezone.utc) >= CUTOFF:
                    raise TimeoutError("experiment cutoff; partial run is not scored")
                tr, va = np.flatnonzero(folds[r] != fold), np.flatnonzero(folds[r] == fold)
                predictions = []
                for seed in config.get("estimator_seeds", [0]):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        p, info = fit_predict(config, rows.iloc[tr], data.y[tr], groups[tr],
                            FitContext(split_seed, fold), [rows.iloc[va], test], seed)
                        warnings_seen.update(str(w.message) for w in caught)
                    predictions.append(p)
                    fit_info.append(dict(split_seed=split_seed, fold=fold, estimator_seed=seed, **info))
                # Validation and test populations differ in length.
                oof[r, va] = np.mean([p[0] for p in predictions], axis=0)
                test_by_fold[r, fold] = np.mean([p[1] for p in predictions], axis=0)
                write_json(folder / "progress.json", {"completed_folds": r * 5 + fold + 1,
                           "elapsed_seconds": time.monotonic() - start})
    auc = score(data.y, folds, oof)
    bank = PredictionBank(data.row_ids, data.y, groups, np.array(SPLIT_SEEDS), folds, oof,
        data.test_id, test_by_fold, test_by_fold.mean(axis=(0, 1)), auc)
    validate_bank(bank)
    arrays = {key: getattr(bank, key) for key in PredictionBank.ARRAYS}
    np.savez_compressed(folder / "predictions.npz", **arrays)
    result = {"id": config["id"], "status": "complete", "config": config, "cv": float(auc.mean()),
        "per_seed": auc.mean(axis=1).tolist(), "elapsed_seconds": time.monotonic() - start,
        "fit_info": fit_info, "warnings": sorted(warnings_seen), "data_hashes": data.hashes,
        "split_sha256": sha256_file(ROOT / "artifacts/splits/production_v1.csv"),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                          (Path(__file__), ROOT / "analysis/fast_features.py")},
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "scikit-learn", "interpret-core", "catboost", "lightgbm")},
        "prediction_sha256": sha256_file(folder / "predictions.npz"),
        "comparisons": summaries(arrays, current_references())}
    write_json(folder / "result.json", result)
    best = max((c for c in result["comparisons"] if c["reference"] == "followup_trees"), key=lambda c:c["cv"])
    print(f"DONE {config['id']} CV={auc.mean():.6f} blend={best['cv']:.6f} seconds={result['elapsed_seconds']:.1f}", flush=True)
    return result


def safe_trial(config):
    try:
        return run_trial(config)
    except Exception:
        error = {"id": config["id"], "status": "error", "traceback": traceback.format_exc()}
        write_json(OUT / config["id"] / "error.json", error)
        print(error["traceback"], flush=True)
        return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(safe_trial, queue))
    write_json(args.queue.with_name(args.queue.stem + "_outcomes.json"), results)
    if any(r["status"] == "error" for r in results):
        raise SystemExit(1)
