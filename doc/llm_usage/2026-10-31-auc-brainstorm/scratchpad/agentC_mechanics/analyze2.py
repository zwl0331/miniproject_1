"""Extended tables: adds the queue-3/4 rows, seed averages for L, L+, mcs=5 variants, and the L+ / L blend."""
import json, os, numpy as np
import pipe, analyze
from analyze import HEADER, row, table, oof_auc_rec, S
from scipy.special import logit
from scipy.stats import rankdata

def seed_avgs(recs):
    out = {}
    for base, pat, pre in [("L+", "G_Lp_seed{}", "G_Lp_avg"), ("L", "G_L_seed{}", "G_L_avg"),
                           ("B_Lp_mcs5", "B_Lp_mcs5_seed{}", "B_Lp_mcs5_avg"), ("B_L_mcs5", "B_L_mcs5_seed{}", "B_L_mcs5_avg")]:
        try: seeds = {0: pipe.load_oof(base)}
        except FileNotFoundError: continue
        for s in [1, 2, 3, 4]:
            try: seeds[s] = pipe.load_oof(pat.format(s))
            except FileNotFoundError: pass
        for name, ids in [(pre + "012", [0, 1, 2]), (pre + "01234", [0, 1, 2, 3, 4])]:
            if all(i in seeds for i in ids):
                out[name] = oof_auc_rec(name, {s: np.mean([seeds[i][s] for i in ids], axis=0) for s in S}, note=f"{base} seeds {ids}")
    return out

def blends(a_name, b_name, prefix):
    out = {}
    try: a, b = pipe.load_oof(a_name), pipe.load_oof(b_name)
    except FileNotFoundError: return out
    fi = pipe.fold_indices(); eps = 1e-6
    for kind in ["prob", "rank", "logit"]:
        for w in [0.75, 0.5]:
            oof = {s: np.full(len(a[s]), np.nan) for s in S}
            for s in S:
                for k in range(5):
                    _, va = fi[(s, k)]; pa, pb = a[s][va], b[s][va]
                    if kind == "prob": oof[s][va] = w * pa + (1 - w) * pb
                    elif kind == "rank": oof[s][va] = w * rankdata(pa) / len(va) + (1 - w) * rankdata(pb) / len(va)
                    else: oof[s][va] = w * logit(np.clip(pa, eps, 1 - eps)) + (1 - w) * logit(np.clip(pb, eps, 1 - eps))
            name = f"{prefix}_{kind}_{w}"
            out[name] = oof_auc_rec(name, oof, note=f"{a_name} x {b_name}, {kind}, w={w}")
    return out

recs = analyze.load_results()
recs.update(seed_avgs(recs)); recs.update(blends("L+", "H_ET", "H")); recs.update(blends("L+", "L", "HL"))
parts = ["## Reproduction", "", HEADER,
         row("L (reference recipe)", recs.get("L"), None, recs),
         row("L+ (L + date_count + date_te3), inner cross-fit seed 0", recs.get("L+"), "L", recs),
         row("L+ with inner cross-fit seed 1 (noise calibration; d vs L+)", recs.get("L+_inner1"), "L+", recs), ""]
parts.append(table("A. Histogram resolution (reference L)", [
    ("max_bin=1023, min_data_in_bin=3", "A_L_mb1023_mdb3", "L"), ("max_bin=4095, min_data_in_bin=3", "A_L_mb4095_mdb3", "L"),
    ("max_bin=255, min_data_in_bin=1", "A_L_mb255_mdb1", "L"), ("max_bin=4095, min_data_in_bin=1", "A_L_mb4095_mdb1", "L"),
    ("max_bin_by_feature days=4095 (others 255), min_data_in_bin=1", "A_L_mbf_days4095_mdb1", "L"),
    ("best of the above (days=4095, mdb=1) applied to L+ (reference L+)", "A_Lp_best", "L+")], recs))
parts.append(table("B. Leaf size (reference L unless stated)", [
    ("min_child_samples=3", "B_L_mcs3", "L"), ("min_child_samples=5", "B_L_mcs5", "L"), ("min_child_samples=20", "B_L_mcs20", "L"),
    ("min_child_samples=3 + days=4095/mdb=1 bins", "B_L_mcs3_bestbins", "L"), ("min_child_samples=5 + days=4095/mdb=1 bins", "B_L_mcs5_bestbins", "L"),
    ("L, mcs=5, estimator seed 1 (reference L seed 1)", "B_L_mcs5_seed1", "G_L_seed1"), ("L, mcs=5, estimator seed 2 (reference L seed 2)", "B_L_mcs5_seed2", "G_L_seed2"),
    ("L, mcs=5, mean of estimator seeds {0,1,2} (reference L seeds {0,1,2})", "B_L_mcs5_avg012", "G_L_avg012"),
    ("L+ with min_child_samples=5 (reference L+)", "B_Lp_mcs5", "L+"), ("L+ with min_child_samples=3 (reference L+)", "B_Lp_mcs3", "L+"),
    ("L+, mcs=5, estimator seed 1 (reference L+ seed 1)", "B_Lp_mcs5_seed1", "G_Lp_seed1"), ("L+, mcs=5, estimator seed 2 (reference L+ seed 2)", "B_Lp_mcs5_seed2", "G_Lp_seed2"),
    ("L+, mcs=5, mean of seeds {0,1,2} (reference L+ seeds {0,1,2})", "B_Lp_mcs5_avg012", "G_Lp_avg012")], recs))
parts.append(table("C. Exact date as LightGBM categorical (min_data_in_bin=1, 4095 bins for the date column; reference L with min_data_in_bin=1 = A_L_mb255_mdb1 unless stated)", [
    ("added: cat_smooth=1, min_data_per_group=5", "C_L_cat_cs1_mdpg5", "A_L_mb255_mdb1"), ("added: cat_smooth=1, min_data_per_group=20", "C_L_cat_cs1_mdpg20", "A_L_mb255_mdb1"),
    ("added: cat_smooth=10, min_data_per_group=5", "C_L_cat_cs10_mdpg5", "A_L_mb255_mdb1"), ("added: cat_smooth=10, min_data_per_group=20", "C_L_cat_cs10_mdpg20", "A_L_mb255_mdb1"),
    ("added: cat_smooth=30, min_data_per_group=5", "C_L_cat_cs30_mdpg5", "A_L_mb255_mdb1"),
    ("replacing days: cat_smooth=1, min_data_per_group=5", "C_L_catREPL_cs1_mdpg5", "A_L_mb255_mdb1"),
    ("added, vs plain L: cat_smooth=10, min_data_per_group=5", "C_L_cat_cs10_mdpg5", "L"), ("added, vs plain L: cat_smooth=10, min_data_per_group=20", "C_L_cat_cs10_mdpg20", "L"),
    ("L+ + categorical, cat_smooth=1, mdpg=5 (reference L+)", "C_Lp_cat_cs1_mdpg5", "L+"), ("L+ + categorical, cat_smooth=10, mdpg=5 (reference L+)", "C_Lp_cat_cs10_mdpg5", "L+")], recs))
parts.append(table("D. CatBoost (depth 6, 1000 it, lr 0.03, l2 3), raw date string as categorical", [
    ("CatBoost on F2 (reference)", "D_CB_F2", None), ("CatBoost on F2 + date string categorical", "D_CB_F2_cat", "D_CB_F2"),
    ("CatBoost on F2 + date string, vs L", "D_CB_F2_cat", "L")], recs))
parts.append(table("E. Within-date nearest-neighbour block (cross-fitted; 4 columns)", [
    ("L + NN block (reference L)", "E_L_nn", "L"), ("L+ + NN block (reference L+)", "E_Lp_nn", "L+")], recs))
parts.append(table("F. Exact-date target-encoding smoothing (reference L+, m=3)", [
    ("m=1", "F_Lp_te_m1", "L+"), ("m=0.5", "F_Lp_te_m0.5", "L+"), ("unsmoothed mean + raw count column", "F_Lp_te_m0_cnt", "L+"),
    ("m=3, one label per (date, group)", "F_Lp_te_m3_grp", "L+")], recs))
parts.append(table("G. Estimator-seed averaging (reference seed 0 of the same recipe)", [
    ("L+ seed 1 alone", "G_Lp_seed1", "L+"), ("L+ seed 2 alone", "G_Lp_seed2", "L+"), ("L+ seed 3 alone", "G_Lp_seed3", "L+"), ("L+ seed 4 alone", "G_Lp_seed4", "L+"),
    ("L+ mean of seeds {0,1,2}", "G_Lp_avg012", "L+"), ("L+ mean of seeds {0..4}", "G_Lp_avg01234", "L+"),
    ("L seed 1 alone", "G_L_seed1", "L"), ("L seed 2 alone", "G_L_seed2", "L"), ("L mean of seeds {0,1,2}", "G_L_avg012", "L")], recs))
parts.append(table("H. Blend geometry (reference L+ seed 0); weights are w_L+ / w_other, ranks within each validation fold", [
    ("ExtraTrees(500, min_samples_leaf=2, max_features=1.0) alone", "H_ET", "L+"),
    ("L+/ET probability 0.75/0.25", "H_prob_0.75", "L+"), ("L+/ET rank 0.75/0.25", "H_rank_0.75", "L+"), ("L+/ET logit 0.75/0.25", "H_logit_0.75", "L+"),
    ("L+/ET probability 0.5/0.5", "H_prob_0.5", "L+"), ("L+/ET rank 0.5/0.5", "H_rank_0.5", "L+"), ("L+/ET logit 0.5/0.5", "H_logit_0.5", "L+"),
    ("incidental: L+/L probability 0.75/0.25", "HL_prob_0.75", "L+"), ("incidental: L+/L rank 0.75/0.25", "HL_rank_0.75", "L+"),
    ("incidental: L+/L probability 0.5/0.5", "HL_prob_0.5", "L+"), ("incidental: L+/L logit 0.5/0.5", "HL_logit_0.5", "L+")], recs))
txt = "\n".join(parts)
open(os.path.join(pipe.HERE, "tables.md"), "w").write(txt)
json.dump({k: v for k, v in recs.items() if k.startswith(("G_", "H", "B_L_mcs5_avg", "B_Lp_mcs5_avg"))}, open(os.path.join(pipe.HERE, "derived.json"), "w"), indent=1)
print(txt)
