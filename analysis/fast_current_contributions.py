"""Rescore completed follow-up candidates against the improved tree blend."""
import json

import numpy as np
import pandas as pd

from analysis.fast_exploration import ROOT, OUT, load_arrays, score, write_json, render_table


def main():
    base = load_arrays(OUT / "f2_blend_z5_et25/predictions.npz")
    mlp = load_arrays(OUT / "mlp_w128_a1.0_e300_standard_seeds012/predictions.npz")
    files = list(OUT.glob("f2_*/result.json")) + list((ROOT / "artifacts/neural").glob("f2_*/result.json"))
    cache = OUT / "followup_current_contributions.json"
    records = json.loads(cache.read_text()) if cache.exists() else []
    if any("comparison" not in r for r in records):
        records = []
    # Trial IDs and the tree-reference bank are immutable in this follow-up.
    done = {r["id"] for r in records}
    for path in files:
        r = json.loads(path.read_text())
        if r["id"] in done:
            continue
        recipe = r["config"].get("recipe", r["config"])
        if "family" not in recipe:
            continue
        a = load_arrays(path.parent / "predictions.npz")
        for key in ("train_row_id", "y", "group_id", "fold_id", "split_seeds", "test_id"):
            if not np.array_equal(a[key], base[key]):
                raise ValueError(f"{r['id']} differs in {key}")
        variants = [(f"add_{w:g}", w, (1-w)*base["oof"] + w*a["oof"]) for w in (.1, .25, .5)]
        variants.append(("replace_mlp", .1, base["oof"] - .1*mlp["oof"] + .1*a["oof"]))
        for comparison, w, pred in variants:
            auc = score(a["y"], a["fold_id"], pred)
            diff = auc.mean(axis=1) - base["fold_auc"].mean(axis=1)
            records.append({"id": r["id"], "family": recipe["family"], "comparison": comparison, "weight": w,
                "cv": float(auc.mean()), "standalone_cv": r["cv"], "delta_tree_blend": float(diff.mean()),
                "per_seed": auc.mean(axis=1).tolist(), "all_splits_up": bool((diff > 0).all()),
                "source": str((path.parent / "predictions.npz").relative_to(ROOT))})
    frame = pd.DataFrame(records).sort_values("cv", ascending=False)
    write_json(OUT / "followup_current_contributions.json", frame.to_dict(orient="records"))
    frame.to_csv(OUT / "followup_current_contributions.csv", index=False)
    print(frame.drop_duplicates("id").head(10).to_string(index=False))
    neural = frame[frame.id.str.startswith("f2_neural_")].drop_duplicates("id")
    if len(neural):
        print("Neural candidates:")
        print(neural.to_string(index=False))
    render_table()


if __name__ == "__main__":
    main()
