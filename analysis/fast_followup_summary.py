"""Compare follow-up additions and fixed component replacements using saved OOF."""
import json
from pathlib import Path

import pandas as pd

from analysis.fast_exploration import OUT, load_arrays, references, score, write_json


def summarize():
    refs = references()
    e, current = refs["blend_eq_all"], refs["fast_combined"]
    old_lgb = load_arrays(OUT / "lgb_year_counts_seeds012/predictions.npz")
    old_mlp = load_arrays(OUT / "mlp_w128_a1.0_e300_standard_seeds012/predictions.npz")
    rows = []
    for p in sorted(OUT.glob("f2_*/result.json")):
        r = json.loads(p.read_text())
        if r["status"] not in ("complete", "complete_fixed_blend"):
            continue
        if r["config"].get("family") not in ("lightgbm", "mlp", "extra_trees", "catboost",
                                            "xgboost", "svm", "gaussian_process", "tabm"):
            continue
        a = load_arrays(p.parent / "predictions.npz")
        variants = [(f"add_{w:g}", (1-w)*current["oof"] + w*a["oof"]) for w in (.1, .25, .5)]
        if r["config"]["family"] == "lightgbm":
            variants.append(("replace_lgb", .65*e["oof"] + .25*a["oof"] + .1*old_mlp["oof"]))
        if r["config"]["family"] == "mlp":
            variants.append(("replace_mlp", .65*e["oof"] + .25*old_lgb["oof"] + .1*a["oof"]))
        for name, pred in variants:
            auc = score(a["y"], a["fold_id"], pred)
            diff = auc.mean(axis=1) - current["fold_auc"].mean(axis=1)
            rows.append({"id": r["id"], "family": r["config"]["family"], "comparison": name,
                         "cv": float(auc.mean()), "standalone_cv": r["cv"],
                         "delta_combined": float(diff.mean()), "all_splits_up": bool((diff > 0).all()),
                         "seed155": float(auc[0].mean()), "seed2026": float(auc[1].mean()),
                         "seed1029": float(auc[2].mean()), "elapsed_seconds": r["elapsed_seconds"]})
    if not rows:
        print("No follow-up results yet")
        return
    frame = pd.DataFrame(rows).sort_values("cv", ascending=False)
    frame.to_csv(OUT / "followup_comparisons.csv", index=False)
    write_json(OUT / "followup_comparisons.json", frame.to_dict(orient="records"))
    best = frame.drop_duplicates("id")
    print(best.head(12).to_string(index=False))
    print("Completed:", len(best), "Best by family:")
    print(best.drop_duplicates("family").to_string(index=False))


if __name__ == "__main__":
    summarize()
