"""Re-score every phase 2 bank on a test-like validation population: rows whose primary group has one row
(the test file was built by deleting every test row whose track also occurs in train, Agent B's Q5), beside the
official all-rows mean fold AUC. Read-only: banks are loaded, nothing is written into the project."""
import glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = "/Users/zwl0331/Desktop/Mini_project_1"
OUT = os.path.dirname(os.path.abspath(__file__))

folds = pd.read_csv(f"{ROOT}/artifacts/splits/production_v1.csv")
gsize = folds.groupby("primary_group")["train_row_id"].transform("size")
singleton_ids = set(folds.loc[(folds["eligible"] == 1) & (gsize == 1), "train_row_id"])
print(f"singleton eligible rows: {len(singleton_ids)} of {int(folds['eligible'].sum())}")

# recipe metadata from the experiment log
meta = {}
for line in open(f"{ROOT}/artifacts/phase2/experiments.jsonl"):
    r = json.loads(line)
    if r.get("record") == "run":
        meta[r["run_id"]] = r

rows = []
for f in sorted(glob.glob(f"{ROOT}/artifacts/phase2/bank/*.npz")):
    z = np.load(f, allow_pickle=True)
    ids, y, fold_id, oof = z["train_row_id"], z["y"], z["fold_id"], z["oof"]
    single = np.array([i in singleton_ids for i in ids])
    all_auc, single_auc = [], []
    for r in range(fold_id.shape[0]):
        for k in range(5):
            m = fold_id[r] == k
            all_auc.append(roc_auc_score(y[m], oof[r, m]))
            ms = m & single
            single_auc.append(roc_auc_score(y[ms], oof[r, ms]))
    all_auc, single_auc = np.array(all_auc).reshape(-1, 5), np.array(single_auc).reshape(-1, 5)
    run_id = os.path.basename(f)[:-4]
    md = meta.get(run_id, {})
    rec = md.get("recipe") or (md.get("config") or {}).get("recipe") or {}
    rows.append(dict(run_id=run_id, task=md.get("task_id"), family=md.get("family") or rec.get("family"),
                     blocks=",".join(rec.get("blocks", [])) if rec else "", drop=",".join(rec.get("drop", [])) if rec else "",
                     kind=md.get("prediction_kind"), n_repeats=all_auc.shape[0],
                     all_mean=all_auc.mean(), all_by_seed=",".join(f"{v:.4f}" for v in all_auc.mean(axis=1)),
                     single_mean=single_auc.mean(), single_by_seed=",".join(f"{v:.4f}" for v in single_auc.mean(axis=1)),
                     logged=md.get("mean_fold_auc")))
df = pd.DataFrame(rows)
df["gap"] = df["all_mean"] - df["single_mean"]
df.to_csv(f"{OUT}/rescore_singletons.csv", index=False)
full = df[df["n_repeats"] == 3]
print(f"\n{len(df)} banks, {len(full)} with all three split seeds")
print("\nlogged vs recomputed all-rows mean: max abs diff",
      (full["all_mean"] - full["logged"].astype(float)).abs().max())
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
cols = ["run_id", "task", "family", "blocks", "drop", "all_mean", "single_mean", "gap", "single_by_seed"]
print("\nTop 25 by ALL-ROWS mean fold AUC:")
print(full.sort_values("all_mean", ascending=False).head(25)[cols].to_string(index=False))
print("\nTop 25 by SINGLETON mean fold AUC:")
print(full.sort_values("single_mean", ascending=False).head(25)[cols].to_string(index=False))
print("\nRank correlation (Spearman) between the two scores over all 3-seed banks:",
      round(full[["all_mean", "single_mean"]].corr(method="spearman").iloc[0, 1], 4))
print("Gap (all minus singleton): mean %.4f, min %.4f, max %.4f" % (full["gap"].mean(), full["gap"].min(), full["gap"].max()))
print("\nGap by family:")
print(full.groupby("family")["gap"].agg(["mean", "min", "max", "count"]).round(4).to_string())
# named references
for pat in ("lightgbm_F2_anchor", "p22_lightgbm_b3_001", "p24_B14", "p24_B24", "p24_B20", "p24_B22", "p24_B23", "p22_extra_trees_anchor", "p28_B9", "p24_B32"):
    hit = full[full["run_id"].str.contains(pat)]
    for _, r in hit.iterrows():
        print(f"{pat:28s} all {r['all_mean']:.4f}  singleton {r['single_mean']:.4f} ({r['single_by_seed']})  blocks [{r['blocks']}] drop [{r['drop']}]")
