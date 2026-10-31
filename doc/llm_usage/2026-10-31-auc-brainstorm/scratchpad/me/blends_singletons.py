"""Fixed blends of stored out-of-fold predictions, scored on all rows and on singleton (test-like) rows.
Base: B24 (tuned LightGBM + date_count + date_te3). Partners: the strongest or best-contributing member of each
other family from the logged banks. Probability average and rank average; weights 0.1, 0.25, 0.4; two 3-way
blends. Read-only."""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = "/Users/zwl0331/Desktop/Mini_project_1"
BANK = f"{ROOT}/artifacts/phase2/bank"
folds = pd.read_csv(f"{ROOT}/artifacts/splits/production_v1.csv")
gsize = folds.groupby("primary_group")["train_row_id"].transform("size")
singleton_ids = set(folds.loc[(folds["eligible"] == 1) & (gsize == 1), "train_row_id"])

NAMES = {
    "B24": "p2_p24_B24_seeds155-2026-1029_*",
    "ET_anchor": "p2_p22_extra_trees_anchor_seeds155-2026-1029_*",
    "ET_b3_001": "p2_p22_extra_trees_b3_001_seeds155-2026-1029_*",
    "RF_040": "p2_p22_random_forest_b1_0_040_seeds155-2026-1029_*",
    "kNN_anchor": "p2_p22_knn_anchor_seeds155-2026-1029_*",
    "XGB_020": "p2_p22_xgboost_b1_0_020_seeds155-2026-1029_*",
    "HGB_009": "p2_p22_hist_gb_b1_0_009_seeds155-2026-1029_*",
    "CatB_X2_04": "p2_p24_X2_04_seeds155-2026-1029_*",
    "Logit_B29": "p2_p24_B29_seeds155-2026-1029_*",
    "EBM_best": "p2_p23_ebm_b2_001_seeds155-2026-1029_*",
}
banks = {}
for name, pat in NAMES.items():
    hits = glob.glob(f"{BANK}/{pat}.npz")
    if not hits:
        print("missing", name); continue
    z = np.load(hits[0], allow_pickle=True)
    banks[name] = dict(ids=z["train_row_id"], y=z["y"], fold=z["fold_id"], oof=z["oof"])
ref = banks["B24"]
for n, b in banks.items():
    assert (b["ids"] == ref["ids"]).all() and (b["fold"] == ref["fold"]).all(), n
y, fold, ids = ref["y"], ref["fold"], ref["ids"]
single = np.array([i in singleton_ids for i in ids])


def score(pred):
    a, s = [], []
    for r in range(3):
        for k in range(5):
            m = fold[r] == k
            a.append(roc_auc_score(y[m], pred[r, m])); s.append(roc_auc_score(y[m & single], pred[r, m & single]))
    a, s = np.array(a).reshape(3, 5), np.array(s).reshape(3, 5)
    return a.mean(), s.mean(), a.mean(axis=1), s.mean(axis=1)


def ranks(pred):
    out = np.empty_like(pred)
    for r in range(3):
        for k in range(5):
            m = fold[r] == k
            out[r, m] = rankdata(pred[r, m]) / m.sum()
    return out


base_all, base_single, base_all_seed, base_single_seed = score(ref["oof"])
print(f"B24 alone: all {base_all:.4f}  singleton {base_single:.4f}")
rows = []
for name, b in banks.items():
    if name == "B24":
        continue
    pa, ps, _, _ = score(b["oof"])
    rows.append(dict(blend=f"{name} alone", w=None, geom="", all=pa, single=ps, d_all=pa - base_all, d_single=ps - base_single, single_seed=""))
    for w in (0.1, 0.25, 0.4):
        for geom in ("prob", "rank"):
            p = (1 - w) * ref["oof"] + w * b["oof"] if geom == "prob" else (1 - w) * ranks(ref["oof"]) + w * ranks(b["oof"])
            pa, ps, _, pss = score(p)
            rows.append(dict(blend=f"B24 + {name}", w=w, geom=geom, all=pa, single=ps, d_all=pa - base_all, d_single=ps - base_single,
                             single_seed=",".join(f"{v - bs:+.4f}" for v, bs in zip(pss, base_single_seed))))
for combo, ws in ((("ET_anchor", "kNN_anchor"), (0.2, 0.1)), (("ET_anchor", "kNN_anchor"), (0.25, 0.15)),
                  (("ET_anchor", "RF_040"), (0.15, 0.15)), (("ET_anchor", "CatB_X2_04"), (0.2, 0.15)),
                  (("ET_anchor", "kNN_anchor", "CatB_X2_04"), (0.2, 0.1, 0.1))):
    for geom in ("prob", "rank"):
        f = (lambda x: x) if geom == "prob" else ranks
        p = (1 - sum(ws)) * f(ref["oof"]) + sum(w * f(banks[c]["oof"]) for c, w in zip(combo, ws))
        pa, ps, _, pss = score(p)
        rows.append(dict(blend="B24 + " + " + ".join(f"{w}x{c}" for c, w in zip(combo, ws)), w=None, geom=geom, all=pa, single=ps,
                         d_all=pa - base_all, d_single=ps - base_single,
                         single_seed=",".join(f"{v - bs:+.4f}" for v, bs in zip(pss, base_single_seed))))
df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blends_singletons.csv"), index=False)
print(df.sort_values("single", ascending=False).head(30).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nstandalone partners:")
print(df[df["w"].isna() & df["blend"].str.endswith("alone")].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
