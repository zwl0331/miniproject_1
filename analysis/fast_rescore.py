"""Recheck saved atomic candidates against today's ensemble, without retraining."""
import json
import time

import numpy as np
import pandas as pd

from analysis.fast_exploration import ROOT, OUT, load_arrays, references, score, summaries, write_json
from songpop.data import sha256_file


def main():
    start = time.monotonic()
    refs = references()
    e = refs["blend_eq_all"]
    records = {}
    for line in (ROOT / "artifacts/phase2/experiments.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("record") == "run" and r.get("prediction_kind") == "fitted" \
                and r.get("split_seeds") == [155, 2026, 1029] and r.get("execution_status") == "ok":
            records[r["run_id"]] = r
    results, signatures = [], set()
    for i, (run_id, r) in enumerate(records.items()):
        path = ROOT / r["bank_path"]
        a = load_arrays(path)
        for key in ("train_row_id", "y", "group_id", "split_seeds", "fold_id", "test_id"):
            if not np.array_equal(a[key], e[key]):
                raise ValueError(f"{run_id}: mismatched {key}")
        # Identical predictions, including copies under new provenance, are one
        # diagnostic candidate. Do not discard distinct seed averages.
        import hashlib
        signature = hashlib.sha256(a["oof"].tobytes() + a["test_mean"].tobytes()).hexdigest()
        if signature in signatures:
            continue
        signatures.add(signature)
        for weight in (.1, .25, .5):
            auc = score(e["y"], e["fold_id"], (1 - weight) * e["oof"] + weight * a["oof"])
            diff = auc.mean(axis=1) - e["fold_auc"].mean(axis=1)
            results.append({"run_id": run_id, "family": r["family"], "source": r["bank_path"], "weight": weight,
                            "standalone_cv": float(a["fold_auc"].mean()), "cv": float(auc.mean()),
                            "seed155": float(auc[0].mean()), "seed2026": float(auc[1].mean()), "seed1029": float(auc[2].mean()),
                            "delta_E": float(auc.mean() - e["fold_auc"].mean()), "all_splits_positive": bool(np.all(diff > 0))})
        if (i + 1) % 100 == 0:
            print(f"checked {i + 1}/{len(records)} saved banks", flush=True)
    frame = pd.DataFrame(results).sort_values("cv", ascending=False)
    frame.to_csv(OUT / "historical_rescore.csv", index=False)
    winners = frame.drop_duplicates("family")
    print(winners[["family", "cv", "delta_E", "weight", "all_splits_positive", "run_id"]].to_string(index=False))
    # Retain each positive family winner as a named result in the single notes
    # table. The full scan is a diagnostic CSV; original banks remain intact.
    for _, row in winners[winners.delta_E > 0].iterrows():
        path = ROOT / row.source
        a = load_arrays(path)
        with np.load(path) as z:
            original = json.loads(str(z["metadata_json"]))
        name = "rescored_" + row.family
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "predictions.npz", **a)
        config = {"id": name, "family": row.family, "recipe": original["recipe"],
                  "source_path": row.source, "source_sha256": sha256_file(path),
                  "note": "Historical saved-bank contribution winner; no new training. Selected exploratorily on existing CV."}
        write_json(folder / "config.json", config)
        write_json(folder / "result.json", {"id": name, "status": "reused_historical", "config": config,
                   "cv": float(a["fold_auc"].mean()), "per_seed": a["fold_auc"].mean(axis=1).tolist(),
                   "elapsed_seconds": 0., "comparisons": summaries(a, refs),
                   "prediction_sha256": sha256_file(folder / "predictions.npz")})
    write_json(OUT / "historical_rescore.json", {"saved_run_count": len(records), "distinct_predictions": len(signatures),
               "elapsed_seconds": time.monotonic() - start, "top_by_family": winners.to_dict(orient="records"),
               "purpose": "Revisit potentially overlooked ensemble contributions without applying old allocation thresholds."})


if __name__ == "__main__":
    main()
