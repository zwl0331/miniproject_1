"""A few cheap blend checks using already fitted non-menu candidates."""
import json
import time
import numpy as np

from analysis.fast_exploration import ROOT, OUT, load_arrays, references, summaries, write_json
from songpop.data import sha256_file
from songpop.evaluate import PredictionBank, validate_bank

RUNS = {
    "reuse_knn_best": "p2_p26_knn_230fcf76_seeds155-2026-1029_b0d6ed8364c2",
    "reuse_knn_anchor": "p2_p22_knn_anchor_seeds155-2026-1029",
    "reuse_catboost_dates": "p2_p26_catboost_cf304bd3_seeds155-2026-1029_c293905d8c54",
    "reuse_spline_best": "p2_p26_logistic_3040d97c_seeds155-2026-1029_d2b995aa2c47",
}


def main():
    refs = references()
    for name, run in RUNS.items():
        start = time.monotonic()
        paths = list((ROOT / "artifacts/phase2/bank").glob(run + "*.npz"))
        if len(paths) != 1:
            raise ValueError(f"expected one source for {run}, found {len(paths)}")
        path = paths[0]
        arrays = load_arrays(path)
        validate_bank(PredictionBank(**arrays))
        for key in ("train_row_id", "y", "group_id", "split_seeds", "fold_id", "test_id"):
            if not np.array_equal(arrays[key], refs["blend_eq_all"][key]):
                raise ValueError(f"reused bank differs in {key}")
        with np.load(path) as z:
            original = json.loads(str(z["metadata_json"]))
        comparisons = summaries(arrays, refs)
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "predictions.npz", **arrays)
        config = {"id": name, "family": original["recipe"]["family"], "recipe": original["recipe"],
                  "source_path": str(path.relative_to(ROOT)), "source_sha256": sha256_file(path),
                  "note": "Reused historical fit, same saved folds, no new training."}
        write_json(folder / "config.json", config)
        result = {"id": name, "status": "reused_historical", "config": config,
                  "cv": float(arrays["fold_auc"].mean()), "per_seed": arrays["fold_auc"].mean(axis=1).tolist(),
                  "elapsed_seconds": time.monotonic() - start, "comparisons": comparisons,
                  "prediction_sha256": sha256_file(folder / "predictions.npz")}
        write_json(folder / "result.json", result)
        best = max(comparisons, key=lambda x: x["cv"])
        print(f"{name}: standalone {result['cv']:.6f}, best addition {best['cv']:.6f}, delta E {best['delta_E']:+.6f}")


if __name__ == "__main__":
    main()
