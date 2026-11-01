"""Small stateless log-odds blends; no calibration or query-population fitting."""
import json
import time

import numpy as np
from scipy.special import expit, logit

from analysis.fast_exploration import ROOT, OUT, load_arrays, references, score, write_json, render_table
from songpop.data import sha256_file
from songpop.evaluate import PredictionBank, validate_bank


def main():
    baseline_path = OUT / "f2_blend_z5_et25/predictions.npz"
    baseline = load_arrays(baseline_path)
    e = references()["blend_eq_all"]
    candidates = list((ROOT / "artifacts/neural").glob("f2_*/predictions.npz"))
    candidates += [OUT / "f2_gp_matern10/predictions.npz"]
    candidates += list(OUT.glob("f2_neural_avg_*/predictions.npz"))
    for path in candidates:
        a = load_arrays(path)
        for key in ("train_row_id", "y", "group_id", "fold_id", "split_seeds", "test_id"):
            if not np.array_equal(a[key], baseline[key]):
                raise ValueError(f"{path} differs in {key}")
        for w in (.1, .25):
            name = "f2_logit_" + path.parent.name.removeprefix("f2_") + f"_w{round(w*100)}"
            folder = OUT / name
            if (folder / "result.json").exists():
                continue
            started = time.monotonic()
            def blend(key):
                return expit((1-w)*logit(np.clip(baseline[key], 1e-6, 1-1e-6)) +
                             w*logit(np.clip(a[key], 1e-6, 1-1e-6)))
            oof, test = blend("oof"), blend("test_by_fold")
            auc = score(a["y"], a["fold_id"], oof)
            arrays = dict(a, oof=oof, test_by_fold=test, test_mean=test.mean(axis=(0, 1)), fold_auc=auc)
            validate_bank(PredictionBank(**arrays))
            spec = {"id": name, "method": "sigmoid_of_weighted_logits", "clip": 1e-6,
                    "components": [{"path": str(p.relative_to(ROOT)), "weight": weight, "sha256": sha256_file(p)}
                                   for p, weight in ((baseline_path, 1-w), (path, w))]}
            write_json(folder / "config.json", spec)
            np.savez_compressed(folder / "predictions.npz", **arrays)
            result = {"id": name, "status": "complete_fixed_blend", "config": spec,
                "cv": float(auc.mean()), "per_seed": auc.mean(axis=1).tolist(),
                "elapsed_seconds": time.monotonic()-started,
                "prediction_sha256": sha256_file(folder / "predictions.npz"),
                "code_sha256": {str((ROOT / "analysis/fast_logit_blends.py").relative_to(ROOT)):
                                sha256_file(ROOT / "analysis/fast_logit_blends.py")},
                "comparisons": [{"reference": "standalone", "weight": 1., "cv": float(auc.mean()),
                    "per_seed": auc.mean(axis=1).tolist(), "fold_auc": auc.tolist(),
                    "delta_E": float(auc.mean()-e["fold_auc"].mean()),
                    "delta_reference": float(auc.mean()-baseline["fold_auc"].mean())}]}
            write_json(folder / "result.json", result)
            print(name, f"CV={auc.mean():.6f}", f"delta_tree={auc.mean()-baseline['fold_auc'].mean():+.6f}")
    render_table()


if __name__ == "__main__":
    main()
