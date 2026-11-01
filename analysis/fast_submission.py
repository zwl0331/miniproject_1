"""Prepare an owner-chosen fixed blend and apply fast-mode automatic file checks.

The input JSON lists saved prediction-bank paths and nonnegative weights.
No fitting, test-feature statistics, or public scores enter this operation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.fast_exploration import ROOT, load_arrays, references, score, write_json
from songpop.data import TARGET, load_dataset, sha256_file
from songpop.evaluate import PredictionBank, validate_bank


def check_file(path, data, current_best):
    frame = pd.read_csv(path)
    if list(frame.columns) != ["ID", TARGET] or len(frame) != len(data.test):
        raise ValueError("submission columns or row count differ from the sample")
    if not np.array_equal(frame.ID.to_numpy(), data.test_id) or not frame.ID.is_unique:
        raise ValueError("submission IDs differ from test.csv order")
    p = frame[TARGET].to_numpy(float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("submission probabilities are not finite in [0, 1]")
    reference = pd.read_csv(current_best)
    if not np.array_equal(reference.ID.to_numpy(), frame.ID.to_numpy()):
        raise ValueError("correlation reference has different IDs")
    corr = float(np.corrcoef(reference[TARGET].to_numpy(float), p)[0, 1])
    if not np.isfinite(corr):
        raise ValueError("undefined correlation")
    return {"rows": len(frame), "ids_match_test_in_order": True, "finite_in_unit_interval": True,
            "min": float(p.min()), "max": float(p.max()), "correlation_current_best": corr,
            "current_best_file": str(Path(current_best).relative_to(ROOT)), "sha256": sha256_file(path)}


def prepare(spec):
    data = load_dataset(ROOT)
    ref = references()["blend_eq_all"]
    weights = np.array([c["weight"] for c in spec["components"]], dtype=float)
    if not len(weights) or not np.isfinite(weights).all() or (weights < 0).any() or abs(weights.sum() - 1) > 1e-12:
        raise ValueError("blend weights must be nonnegative and sum to one")
    oof, test = np.zeros_like(ref["oof"]), np.zeros_like(ref["test_by_fold"])
    sources = []
    for component, weight in zip(spec["components"], weights):
        path = ROOT / component["path"]
        arrays = load_arrays(path)
        for key in ("train_row_id", "y", "group_id", "split_seeds", "fold_id", "test_id"):
            if not np.array_equal(arrays[key], ref[key]):
                raise ValueError(f"component {path} differs in {key}")
        validate_bank(PredictionBank(**arrays))
        oof += weight * arrays["oof"]
        test += weight * arrays["test_by_fold"]
        sources.append(dict(component, sha256=sha256_file(path)))
    auc = score(ref["y"], ref["fold_id"], oof)
    arrays = dict(ref, oof=oof, test_by_fold=test, test_mean=test.mean(axis=(0, 1)), fold_auc=auc)
    validate_bank(PredictionBank(**arrays))
    folder = ROOT / "artifacts/fast/final_candidates" / spec["name"]
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "predictions.npz", **arrays)
    output = ROOT / "submissions" / ("2026-11-01_fast_" + spec["name"] + ".csv")
    frame = pd.DataFrame({"ID": data.test_id, TARGET: arrays["test_mean"]})
    content = frame.to_csv(index=False, float_format="%.17g")
    if output.exists():
        if output.read_text() != content:
            raise FileExistsError(f"refusing to overwrite different submission {output}")
    else:
        with output.open("x") as f:
            f.write(content)
    best = ROOT / spec.get("current_best_file", "submissions/2026-10-31_p2.6b_blend_eq_all.csv")
    checks = check_file(output, data, best)
    result = {"name": spec["name"], "purpose": spec["purpose"], "status": "prepared for Sam; not uploaded or final-selected",
              "cv_mean": float(auc.mean()), "cv_per_seed": auc.mean(axis=1).tolist(),
              "fold_auc": auc.tolist(), "delta_E": float(auc.mean() - ref["fold_auc"].mean()),
              "components": sources, "checks": checks, "public_score": None,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "prediction_path": str((folder / "predictions.npz").relative_to(ROOT)),
              "prediction_sha256": sha256_file(folder / "predictions.npz"),
              "llm_use": "Codex authored the fast exploration, blending and file checks under Sam's 2026-11-01 instruction."}
    write_json(folder / "result.json", result)
    write_json(output.with_suffix(".json"), result)
    log = ROOT / "submissions/README.md"
    marker = f"`{output.name}` | `{checks['sha256']}`"
    if marker not in log.read_text():
        with log.open("a") as f:
            f.write(f"| 2026-11-01 | {marker} | Prepared for public check under fast mode. CV {auc.mean():.6f}; "
                    f"per seed " + "/".join(f"{v:.6f}" for v in auc.mean(axis=1)) +
                    f". Automatic IDs/count/probability checks passed; correlation with {best.name}: {checks['correlation_current_best']:.6f}. "
                    f"Purpose: {spec['purpose']}. Sam decides and uploads; public score pending. |\n")
    print(json.dumps({"file": str(output.relative_to(ROOT)), "cv": result["cv_mean"], "per_seed": result["cv_per_seed"],
                      "delta_E": result["delta_E"], **checks}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    prepare(json.loads(args.spec.read_text()))
