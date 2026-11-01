"""Fast learned ensembles with base-model cross-fitting inside every outer fold.

Global OOF predictions are used only for outer validation, never meta training.
All six existing regularized combiners share the same rebuilt training matrix.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from analysis.extended_cpu import fit_predict, current_references, CUTOFF
from analysis.fast_exploration import ROOT, data_inputs, load_arrays, score, summaries, write_json
from songpop.data import sha256_file
from songpop.evaluate import PredictionBank, validate_bank
from songpop.phase2.encoders import FitContext
from songpop.phase2.procedures import population_folds, coordinate_seed, fit_combiner, apply_combiner
from songpop.phase2.raw_neural import fit_raw, Worker
from songpop.splits import SPLIT_SEEDS

OUT = ROOT / "artifacts/extended/stacking"


def resolve(path):
    """Resolve a saved fixed seed average to its original fitting configurations."""
    path = ROOT / path
    config = json.loads(path.with_name("config.json").read_text())
    if "components" not in config:
        return [(config, 1.)]
    if "estimator_seeds" not in config:
        raise ValueError("stacking channels must be atoms or fixed estimator-seed averages")
    result = []
    for component in config["components"]:
        result += [(child, weight * component["weight"]) for child, weight in resolve(component["path"])]
    if abs(sum(weight for _, weight in result) - 1) > 1e-12:
        raise ValueError("channel weights must sum to one")
    return result


def predict_channel(channel, rows, y, groups, context, query):
    predictions = []
    for config, weight in channel["components"]:
        if "recipe" in config:
            fitted = fit_raw(config["recipe"], rows, y, groups, context)
            predictions.append(weight * fitted.predict_high(query))
        else:
            seeds = config.get("estimator_seeds", [0])
            for seed in seeds:
                p, _ = fit_predict(config, rows, y, groups, context, [query], seed)
                predictions.append(weight / len(seeds) * p[0])
    return np.sum(predictions, axis=0)


def meta_predictions(channel, rows, y, groups, meta_fold, context_name, predictor=predict_channel):
    """Only the outer training population is an argument to this function."""
    result = np.full(len(y), np.nan)
    for fold in range(5):
        if datetime.now(timezone.utc) >= CUTOFF:
            raise TimeoutError("experiment cutoff before meta fit")
        tr, va = np.flatnonzero(meta_fold != fold), np.flatnonzero(meta_fold == fold)
        if set(groups[tr]) & set(groups[va]):
            raise ValueError("meta fitting and validation track groups overlap")
        result[va] = predictor(channel, rows.iloc[tr], y[tr], groups[tr],
                               FitContext(context_name, fold), rows.iloc[va])
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("invalid meta predictions")
    return result


def run_task(task):
    channel, repeat, fold = task["channel"], task["repeat"], task["fold"]
    seed = SPLIT_SEEDS[repeat]
    folder = OUT / channel["id"] / f"s{seed}_f{fold}"
    if (folder / "result.json").exists():
        old = json.loads((folder / "result.json").read_text())
        if old["task"] != task:
            raise ValueError("meta cache belongs to another channel")
        return old
    started = time.monotonic()
    data, rows, _, groups, folds = data_inputs()
    tr = np.flatnonzero(folds[repeat] != fold)
    rows, y, groups = rows.iloc[tr], data.y[tr], groups[tr]
    context = f"x3_meta:{seed}:{fold}"
    meta_fold = population_folds(y, groups, rows.index, 5, coordinate_seed(context))
    print(f"META START {channel['id']} {seed}/{fold}", flush=True)
    with threadpool_limits(limits=4), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions = meta_predictions(channel, rows, y, groups, meta_fold, context)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "predictions.npz", train_row_id=np.asarray(rows.index), y=y,
                        group_id=groups, meta_fold=meta_fold, oof=predictions)
    result = dict(status="complete", task=task, elapsed_seconds=time.monotonic()-started,
                  warnings=sorted({str(w.message) for w in caught}),
                  prediction_sha256=sha256_file(folder / "predictions.npz"),
                  source_sha256=sha256_file(Path(__file__)))
    write_json(folder / "result.json", result)
    print(f"META DONE {channel['id']} {seed}/{fold} {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def safe_task(task):
    try:
        return run_task(task)
    except Exception:
        error = dict(status="error", task=task, traceback=traceback.format_exc())
        print(error["traceback"], flush=True)
        Worker.close_all()
        return error


def make_channels(paths):
    channels = []
    for path in paths:
        path = str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else str(path)
        channels.append(dict(id=Path(path).parent.name, path=path,
                             sha256=sha256_file(ROOT / path), components=resolve(path)))
    return channels


def combine(spec):
    channels = spec["channels"]
    data, _, _, groups, folds = data_inputs()
    base = current_references()["followup_trees"]
    banks = []
    for channel in channels:
        path = ROOT / channel["path"]
        if sha256_file(path) != channel["sha256"]:
            raise ValueError("outer channel bank changed")
        bank = load_arrays(path)
        for key in ("train_row_id", "y", "group_id", "fold_id", "split_seeds", "test_id"):
            if not np.array_equal(bank[key], base[key]):
                raise ValueError(f"channel mismatch in {key}")
        banks.append(bank)
    results = []
    settings = [("constrained", x) for x in (0., .01, .1)] + [("stacker", x) for x in (.01, .1, 1.)]
    for setting in settings:
        name = f"x3_learned_{spec['id']}_{setting[0]}_{setting[1]:g}".replace(".", "p")
        folder = ROOT / "artifacts/extended" / name
        config = dict(id=name, family="learned_ensemble", setting=list(setting), pool=spec)
        if (folder / "result.json").exists():
            old = json.loads((folder / "result.json").read_text())
            if old["config"] != config:
                raise ValueError("learned ensemble id already belongs to another specification")
            results.append(old)
            continue
        started = time.monotonic()
        oof, test = np.full_like(base["oof"], np.nan), np.full_like(base["test_by_fold"], np.nan)
        states = []
        for r, seed in enumerate(SPLIT_SEEDS):
            for fold in range(5):
                tr, va = np.flatnonzero(folds[r] != fold), np.flatnonzero(folds[r] == fold)
                Z, shared_meta = [], None
                for channel in channels:
                    path = OUT / channel["id"] / f"s{seed}_f{fold}" / "predictions.npz"
                    with np.load(path, allow_pickle=False) as z:
                        for key, expected in (("train_row_id", data.row_ids[tr]), ("y", data.y[tr]), ("group_id", groups[tr])):
                            if not np.array_equal(z[key], expected):
                                raise ValueError("meta bank fitting population differs")
                        if shared_meta is not None and not np.array_equal(shared_meta, z["meta_fold"]):
                            raise ValueError("meta channel folds differ")
                        shared_meta = z["meta_fold"].copy()
                        Z.append(z["oof"].copy())
                state = fit_combiner(setting, np.column_stack(Z), data.y[tr])
                oof[r, va] = apply_combiner(state, np.column_stack([b["oof"][r, va] for b in banks]))
                test[r, fold] = apply_combiner(state, np.column_stack([b["test_by_fold"][r, fold] for b in banks]))
                states.append(dict(split_seed=seed, fold=fold, state=state))
        auc = score(data.y, folds, oof)
        arrays = dict(base, oof=oof, test_by_fold=test, test_mean=test.mean(axis=(0, 1)), fold_auc=auc)
        validate_bank(PredictionBank(**arrays))
        write_json(folder / "config.json", config)
        np.savez_compressed(folder / "predictions.npz", **arrays)
        result = dict(id=name, status="complete", config=config, cv=float(auc.mean()),
            per_seed=auc.mean(axis=1).tolist(), elapsed_seconds=time.monotonic()-started,
            fit_info=states, comparisons=summaries(arrays, current_references()),
            prediction_sha256=sha256_file(folder / "predictions.npz"), source_sha256=sha256_file(Path(__file__)))
        write_json(folder / "result.json", result)
        results.append(result)
        print(name, result["cv"], result["per_seed"], flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--queue", type=Path)
    action.add_argument("--combine", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.combine:
        combine(json.loads(args.combine.read_text()))
    else:
        queue = json.loads(args.queue.read_text())
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(safe_task, queue))
        write_json(args.queue.with_name(args.queue.stem + "_outcomes.json"), results)
        if any(r["status"] == "error" for r in results):
            raise SystemExit(1)
