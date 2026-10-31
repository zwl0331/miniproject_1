"""Build the per-family tables (mean, per-seed means, per-seed deltas vs the matched
reference) from results.jsonl and the cached OOF predictions. Writes tables.md."""
import json
import os

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata

import pipe

HERE = pipe.HERE
S = pipe.SEEDS


def load_results():
    recs = {}
    if os.path.exists(pipe.RESULTS):
        for line in open(pipe.RESULTS):
            r = json.loads(line)
            recs[r["name"]] = r  # last wins
    return recs


def per_seed(rec):
    return [rec["per_seed"][str(s)] for s in S]


def row(label, rec, ref, recs):
    if rec is None:
        return f"| {label} | not run | | | | | | | |"
    ps = per_seed(rec)
    cells = [f"{rec['mean']:.4f}"] + [f"{v:.4f}" for v in ps]
    if ref is not None and ref in recs:
        rp = per_seed(recs[ref])
        d = [a - b for a, b in zip(ps, rp)]
        dm = rec["mean"] - recs[ref]["mean"]
        stable = "yes" if all(x > 0 for x in d) else ("no (all <=0)" if all(x <= 0 for x in d) else "mixed")
        cells += [f"{dm:+.4f}"] + [f"{x:+.4f}" for x in d] + [stable]
    else:
        cells += ["ref", "", "", "", ""]
    return f"| {label} | " + " | ".join(cells) + " |"


HEADER = ("| run | mean | s155 | s2026 | s1029 | d mean | d155 | d2026 | d1029 | all seeds up |\n"
          "|---|---|---|---|---|---|---|---|---|---|")


def table(title, rows, recs):
    out = [f"### {title}", "", HEADER]
    for label, name, ref in rows:
        out.append(row(label, recs.get(name), ref, recs))
    return "\n".join(out) + "\n"


def oof_auc_rec(name, oof, note=""):
    fa = pipe.auc_from_oof(name, oof)
    per = {str(s): float(np.mean([fa[(s, k)] for k in range(5)])) for s in S}
    return dict(name=name, mean=float(np.mean(list(fa.values()))), per_seed=per, note=note)


def seed_avg(recs):
    """G: probability average of the L+ estimator seeds."""
    out = {}
    try:
        base = pipe.load_oof("L+")
    except FileNotFoundError:
        return out
    seeds = {0: base}
    for s in [1, 2, 3, 4]:
        try:
            seeds[s] = pipe.load_oof(f"G_Lp_seed{s}")
        except FileNotFoundError:
            pass
    for name, ids in [("G_Lp_avg012", [0, 1, 2]), ("G_Lp_avg01234", [0, 1, 2, 3, 4])]:
        if all(i in seeds for i in ids):
            oof = {s: np.mean([seeds[i][s] for i in ids], axis=0) for s in S}
            out[name] = oof_auc_rec(name, oof, note=f"seeds {ids}")
    return out


def blends(recs):
    """H: geometry of the L+ / ExtraTrees blend on OOF predictions, per fold."""
    out = {}
    try:
        a = pipe.load_oof("L+")
        b = pipe.load_oof("H_ET")
    except FileNotFoundError:
        return out
    fi = pipe.fold_indices()
    eps = 1e-6

    def blend(kind, w):
        oof = {s: np.full(len(a[s]), np.nan) for s in S}
        for s in S:
            for k in range(5):
                _, va = fi[(s, k)]
                pa, pb = a[s][va], b[s][va]
                if kind == "prob":
                    oof[s][va] = w * pa + (1 - w) * pb
                elif kind == "rank":
                    oof[s][va] = w * rankdata(pa) / len(va) + (1 - w) * rankdata(pb) / len(va)
                elif kind == "logit":
                    oof[s][va] = w * logit(np.clip(pa, eps, 1 - eps)) + (1 - w) * logit(np.clip(pb, eps, 1 - eps))
        return oof

    for kind in ["prob", "rank", "logit"]:
        for w in [0.75, 0.5]:
            name = f"H_{kind}_{w}"
            out[name] = oof_auc_rec(name, blend(kind, w), note=f"{kind} average, w_L+={w}")
    return out


def main():
    recs = load_results()
    recs.update(seed_avg(recs))
    recs.update(blends(recs))
    parts = ["## Reproduction", "", HEADER,
             row("L (reference recipe)", recs.get("L"), None, recs),
             row("L+ (team best: L + date_count + date_te3)", recs.get("L+"), "L", recs),
             row("L+ with inner cross-fit seed 1 (noise calibration)", recs.get("L+_inner1"), "L+", recs),
             ""]
    parts.append(table("A. Histogram resolution (reference L)", [
        ("max_bin=1023, min_data_in_bin=3", "A_L_mb1023_mdb3", "L"),
        ("max_bin=4095, min_data_in_bin=3", "A_L_mb4095_mdb3", "L"),
        ("max_bin=255, min_data_in_bin=1", "A_L_mb255_mdb1", "L"),
        ("max_bin=4095, min_data_in_bin=1", "A_L_mb4095_mdb1", "L"),
        ("max_bin_by_feature: days=4095, others 255, min_data_in_bin=1", "A_L_mbf_days4095_mdb1", "L"),
        ("best A setting applied to L+ (reference L+)", "A_Lp_best", "L+"),
    ], recs))
    parts.append(table("B. Leaf size (reference L)", [
        ("min_child_samples=3", "B_L_mcs3", "L"),
        ("min_child_samples=5", "B_L_mcs5", "L"),
        ("min_child_samples=20", "B_L_mcs20", "L"),
        ("min_child_samples=3 + best bins", "B_L_mcs3_bestbins", "L"),
        ("min_child_samples=5 + best bins", "B_L_mcs5_bestbins", "L"),
    ], recs))
    parts.append(table("C. Exact date as LightGBM categorical, added to F2 (reference L with min_data_in_bin=1; L in the second block)", [
        ("cat_smooth=1, min_data_per_group=5", "C_L_cat_cs1_mdpg5", "A_L_mb255_mdb1"),
        ("cat_smooth=10, min_data_per_group=5", "C_L_cat_cs10_mdpg5", "A_L_mb255_mdb1"),
        ("cat_smooth=1, min_data_per_group=20", "C_L_cat_cs1_mdpg20", "A_L_mb255_mdb1"),
        ("cat_smooth=10, min_data_per_group=20", "C_L_cat_cs10_mdpg20", "A_L_mb255_mdb1"),
        ("replace days by categorical, cs=1, mdpg=5", "C_L_catREPL_cs1_mdpg5", "A_L_mb255_mdb1"),
        ("vs L: cat_smooth=1, min_data_per_group=5", "C_L_cat_cs1_mdpg5", "L"),
        ("vs L: cat_smooth=10, min_data_per_group=20", "C_L_cat_cs10_mdpg20", "L"),
        ("L+ + categorical date, cs=1, mdpg=5 (reference L+)", "C_Lp_cat_cs1_mdpg5", "L+"),
    ], recs))
    parts.append(table("D. CatBoost, raw date string as categorical (reference CatBoost on F2)", [
        ("CatBoost F2", "D_CB_F2", None),
        ("CatBoost F2 + date string categorical", "D_CB_F2_cat", "D_CB_F2"),
    ], recs))
    parts.append(table("E. Within-date nearest-neighbour block (cross-fitted)", [
        ("L + NN block (reference L)", "E_L_nn", "L"),
        ("L+ + NN block (reference L+)", "E_Lp_nn", "L+"),
    ], recs))
    parts.append(table("F. Exact-date target-encoding smoothing (reference L+, m=3)", [
        ("m=1", "F_Lp_te_m1", "L+"),
        ("m=0.5", "F_Lp_te_m0.5", "L+"),
        ("unsmoothed mean + raw count", "F_Lp_te_m0_cnt", "L+"),
        ("m=3, one label per group", "F_Lp_te_m3_grp", "L+"),
    ], recs))
    parts.append(table("G. Estimator-seed averaging of L+ (reference L+ seed 0)", [
        ("L+ seed 1 alone", "G_Lp_seed1", "L+"),
        ("L+ seed 2 alone", "G_Lp_seed2", "L+"),
        ("L+ seed 3 alone", "G_Lp_seed3", "L+"),
        ("L+ seed 4 alone", "G_Lp_seed4", "L+"),
        ("mean of seeds {0,1,2}", "G_Lp_avg012", "L+"),
        ("mean of seeds {0..4}", "G_Lp_avg01234", "L+"),
    ], recs))
    parts.append(table("H. Blend geometry, L+ with ExtraTrees (reference L+)", [
        ("ExtraTrees alone", "H_ET", "L+"),
        ("probability 0.75/0.25", "H_prob_0.75", "L+"),
        ("rank 0.75/0.25", "H_rank_0.75", "L+"),
        ("logit 0.75/0.25", "H_logit_0.75", "L+"),
        ("probability 0.5/0.5", "H_prob_0.5", "L+"),
        ("rank 0.5/0.5", "H_rank_0.5", "L+"),
        ("logit 0.5/0.5", "H_logit_0.5", "L+"),
    ], recs))
    txt = "\n".join(parts)
    open(os.path.join(HERE, "tables.md"), "w").write(txt)
    print(txt)
    # dump derived records too
    with open(os.path.join(HERE, "derived.json"), "w") as fh:
        json.dump({k: v for k, v in recs.items() if k.startswith(("G_Lp_avg", "H_"))}, fh, indent=1)


if __name__ == "__main__":
    main()
