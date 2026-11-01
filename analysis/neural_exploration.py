"""Fast raw-neural trials on the existing 3 x 5 grouped folds.

Write a small starter queue:
  python -m analysis.neural_exploration --write-queue artifacts/neural/queue.json
Run an explicit queue, one GPU worker:
  python -m analysis.neural_exploration --queue artifacts/neural/queue.json
"""
import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import traceback
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from analysis.fast_exploration import data_inputs, references, score, summaries, write_json
from songpop.data import sha256_file
from songpop.evaluate import PredictionBank, validate_bank
from songpop.phase2.encoders import FitContext
from songpop.phase2.raw_neural import Worker, fit_raw, sources
from songpop.phase2.raw_recipes import make_recipe, validate
from songpop.splits import SPLIT_SEEDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/neural"
CUTOFF = "2026-11-01T20:00:00+00:00"


def starter_queue(device="cuda"):
    cases = [("tabm", "scalar"), ("tabm", "ple"), ("tabm", "periodic"),
             ("mlp", "ple"), ("resnet", "ple"), ("ft_transformer", "scalar")]
    return [{"id": f"neural_{family}_{rep}", "recipe": make_recipe(family,
             representation=rep, device=device, blocks=["date_count", "date_te3", "cyclic"],
             params={"max_epochs": 200, "patience": 20}), "cutoff_utc": CUTOFF}
            for family, rep in cases]


def check_config(config):
    if set(config) - {"id", "recipe", "cutoff_utc"} or not {"id", "recipe"} <= set(config):
        raise ValueError("trial config needs id, recipe and optional cutoff_utc")
    if not isinstance(config["id"], str) or not config["id"] or \
            not all(c.isalnum() or c in "_-" for c in config["id"]):
        raise ValueError("unsafe trial id")
    validate(config["recipe"])
    cutoff = datetime.fromisoformat(config.get("cutoff_utc", CUTOFF))
    if cutoff.tzinfo is None:
        raise ValueError("cutoff_utc must include its timezone")
    return min(cutoff, datetime(2026, 11, 1, 20, tzinfo=timezone.utc))


def run_trial(config, *, output=OUT):
    """Fit raw training folds only; save complete OOF/test banks for blending."""
    config = copy.deepcopy(config)
    cutoff = check_config(config)
    folder = Path(output) / config["id"]
    result_path = folder / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["config"] != config:
            raise ValueError("refusing to reuse a neural trial id with a different config")
        return result
    if datetime.now(timezone.utc) >= cutoff:
        return {"id": config["id"], "status": "not_started_cutoff"}
    folder.mkdir(parents=True, exist_ok=True)
    saved_config = folder / "config.json"
    if saved_config.exists() and json.loads(saved_config.read_text()) != config:
        raise ValueError("refusing to overwrite a different interrupted neural trial")
    write_json(saved_config, config)
    start = time.monotonic()
    print(f"START {config['id']} {datetime.now(timezone.utc).isoformat()}", flush=True)
    data, rows, test, groups, folds = data_inputs()
    recipe = config["recipe"]
    oof = np.full(folds.shape, np.nan)
    test_by_fold = np.full((len(SPLIT_SEEDS), 5, len(test)), np.nan)
    records, warning_text = [], set()
    with threadpool_limits(limits=4):
        for repeat, split_seed in enumerate(SPLIT_SEEDS):
            for fold in range(5):
                if datetime.now(timezone.utc) >= cutoff:
                    np.savez_compressed(folder / "partial.npz", oof=oof, test_by_fold=test_by_fold)
                    write_json(folder / "interrupted.json", {"id": config["id"], "status": "cutoff",
                        "completed_folds": len(records), "elapsed_seconds": time.monotonic() - start})
                    raise TimeoutError("cutoff reached; incomplete neural trial is not scored")
                tr, va = np.flatnonzero(folds[repeat] != fold), np.flatnonzero(folds[repeat] == fold)
                if set(groups[tr]) & set(groups[va]):
                    raise ValueError("outer train/validation groups overlap")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted = fit_raw(recipe, rows.iloc[tr], data.y[tr], groups[tr], FitContext(split_seed, fold))
                    oof[repeat, va] = fitted.predict_high(rows.iloc[va])
                    test_by_fold[repeat, fold] = fitted.predict_high(test)
                    warning_text.update(f"{type(w.message).__name__}: {w.message}" for w in caught)
                records.append(fitted.receipt)
                write_json(folder / "progress.json", {"completed_folds": len(records),
                    "elapsed_seconds": time.monotonic() - start})
                print(f"FOLD {config['id']} {len(records)}/15 {time.monotonic() - start:.1f}s", flush=True)
    auc = score(data.y, folds, oof)
    bank = PredictionBank(data.row_ids, data.y, groups, np.array(SPLIT_SEEDS), folds, oof,
        data.test_id, test_by_fold, test_by_fold.mean(axis=(0, 1)), auc)
    validate_bank(bank)
    arrays = {key: getattr(bank, key) for key in PredictionBank.ARRAYS}
    np.savez_compressed(folder / "predictions.npz", **arrays)
    result = {"id": config["id"], "status": "complete", "config": config, "cv": float(auc.mean()),
        "per_seed": auc.mean(axis=1).tolist(), "elapsed_seconds": time.monotonic() - start,
        "fit_info": records, "warnings": sorted(warning_text), "data_hashes": data.hashes,
        "split_sha256": sha256_file(ROOT / "artifacts/splits/production_v1.csv"),
        "code_sha256": dict(sources(), **{"analysis/neural_exploration.py": sha256_file(Path(__file__))}),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "packages": {p: importlib.metadata.version(p) for p in
            ("numpy", "pandas", "scikit-learn", "torch", "tabm", "rtdl-num-embeddings", "rtdl-revisiting-models")},
        "prediction_sha256": sha256_file(folder / "predictions.npz"),
        "finished_utc": datetime.now(timezone.utc).isoformat(), "comparisons": summaries(arrays, comparison_references())}
    write_json(result_path, result)
    print(f"DONE {config['id']} CV={result['cv']:.6f} seconds={result['elapsed_seconds']:.1f}", flush=True)
    return result


def comparison_references():
    from analysis.fast_exploration import load_arrays
    path = ROOT / "artifacts/fast/f2_blend_z5_et25/predictions.npz"
    return dict(references(), followup_trees=load_arrays(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--write-queue", type=Path)
    choice.add_argument("--queue", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="device for generated starter recipes")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.write_queue:
        write_json(args.write_queue, starter_queue(args.device))
        return
    queue = json.loads(args.queue.read_text())
    if not isinstance(queue, list) or not queue or len({c["id"] for c in queue}) != len(queue):
        raise ValueError("queue must contain distinct trial ids")
    for config in queue:
        check_config(config)
    failed = False
    try:
        for config in queue:
            try:
                result = run_trial(config, output=args.output)
                if result["status"] == "not_started_cutoff":
                    break
            except TimeoutError:
                raise
            except Exception:
                failed = True
                write_json(args.output / config["id"] / "error.json", {"traceback": traceback.format_exc()})
                traceback.print_exc()
                Worker.close_all()
    finally:
        Worker.close_all()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
