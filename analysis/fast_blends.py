"""Save a few explicitly listed fixed blends; no weight optimization."""
import argparse
import json
import time

import numpy as np

from analysis.fast_exploration import ROOT, OUT, load_arrays, references, score, write_json
from songpop.data import sha256_file
from songpop.evaluate import PredictionBank, validate_bank


def run(spec):
    started = time.monotonic()
    refs = references()
    e = refs["blend_eq_all"]
    weights = [x["weight"] for x in spec["components"]]
    if min(weights) < 0 or abs(sum(weights) - 1) > 1e-12:
        raise ValueError("weights must be nonnegative and sum to one")
    oof, test = np.zeros_like(e["oof"]), np.zeros_like(e["test_by_fold"])
    for component in spec["components"]:
        a = load_arrays(ROOT / component["path"])
        for key in ("train_row_id", "y", "group_id", "fold_id", "split_seeds", "test_id"):
            if not np.array_equal(e[key], a[key]):
                raise ValueError(f"blend component differs in {key}")
        oof += component["weight"] * a["oof"]
        test += component["weight"] * a["test_by_fold"]
    auc = score(e["y"], e["fold_id"], oof)
    arrays = dict(e, oof=oof, test_by_fold=test, test_mean=test.mean(axis=(0, 1)), fold_auc=auc)
    validate_bank(PredictionBank(**arrays))
    folder = OUT / spec["id"]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "config.json").exists() and json.loads((folder / "config.json").read_text()) != spec:
        raise ValueError("refusing to overwrite a different blend")
    write_json(folder / "config.json", spec)
    np.savez_compressed(folder / "predictions.npz", **arrays)
    result = {"id": spec["id"], "status": "complete_fixed_blend", "config": spec,
              "cv": float(auc.mean()), "per_seed": auc.mean(axis=1).tolist(),
              "elapsed_seconds": time.monotonic() - started,
              "prediction_sha256": sha256_file(folder / "predictions.npz"),
              "comparisons": [{"reference": "standalone", "weight": 1., "cv": float(auc.mean()),
                               "per_seed": auc.mean(axis=1).tolist(), "fold_auc": auc.tolist(),
                               "delta_E": float(auc.mean() - e["fold_auc"].mean()),
                               "delta_reference": float(auc.mean() - e["fold_auc"].mean())}]}
    write_json(folder / "result.json", result)
    print(spec["id"], result["cv"], result["per_seed"], "delta E", result["comparisons"][0]["delta_E"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("specs")
    args = ap.parse_args()
    for spec in json.loads(open(args.specs).read()):
        run(spec)
