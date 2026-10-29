"""Scores and the paired cluster bootstrap (doc/plans/modeling_strategy.md section 4, steps 7 and 8;
doc/plans/phase1-data-baseline.md section 4.7)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BOOTSTRAP_SEED = 155001


def prepare_auc(y, score, code):
    """Sort once by score; the weighted AUC of any weighting then needs only sums over tied blocks."""
    order = np.argsort(score, kind="stable")
    s = score[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    return y[order].astype("float64"), code[order], starts


def cached_weighted_auc(cache, weight_by_code):
    """Exact weighted ROC AUC with ties counted as one half; NaN when a class has zero weight."""
    y, code, starts = cache
    w = weight_by_code[code]
    pos = np.add.reduceat(w * y, starts)
    neg = np.add.reduceat(w * (1.0 - y), starts)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return np.nan
    neg_below = np.cumsum(neg) - neg
    return float(np.sum(pos * (neg_below + 0.5 * neg)) / (P * N))


def weighted_auc(y, score, weight):
    """Weighted AUC of one score vector, through the cached implementation."""
    code = np.arange(len(y))
    return cached_weighted_auc(prepare_auc(np.asarray(y), np.asarray(score, dtype="float64"), code),
                               np.asarray(weight, dtype="float64"))


def summarize(bank):
    """The selection metric (mean fold AUC) and its companions, rounded to 6 decimals for the log."""
    R = bank.oof.shape[0]
    return {
        "fold_auc": bank.fold_auc.round(6).tolist(),
        "repeat_mean_auc": bank.fold_auc.mean(axis=1).round(6).tolist(),
        "mean_fold_auc": round(float(bank.fold_auc.mean()), 6),
        "pooled_oof_auc": [round(float(roc_auc_score(bank.y, bank.oof[r])), 6) for r in range(R)],
        "repeat_averaged_oof_auc": round(float(roc_auc_score(bank.y, bank.oof.mean(axis=0))), 6),
    }


def diagnostics(bank, era):
    """Not for selection (strategy section 4, step 7): mean fold AUC with each row weighted by 1 / its group's
    size, and the pooled out-of-fold AUC per era and repeat; null for an era with a single class."""
    sizes = pd.Series(bank.group_id).map(pd.Series(bank.group_id).value_counts()).to_numpy()
    w = 1.0 / sizes
    R, K = bank.fold_auc.shape
    gw = [roc_auc_score(bank.y[m], bank.oof[r, m], sample_weight=w[m])
          for r in range(R) for k in range(K) for m in [bank.fold_id[r] == k]]
    eras = {}
    for name in pd.unique(era):
        m = era == name
        both = len(np.unique(bank.y[m])) == 2
        eras[str(name)] = {"rows": int(m.sum()), "high": int(bank.y[m].sum()),
                           "pooled_oof_auc_by_repeat": [round(float(roc_auc_score(bank.y[m], bank.oof[r, m])), 6)
                                                        for r in range(R)] if both else None,
                           "note": None if both else "single class: AUC undefined"}
    return {"selection_eligible": False, "group_weighted_mean_fold_auc": round(float(np.mean(gw)), 6),
            "era_pooled_oof_auc": eras}


def check_pairable(a, b):
    for f in ["train_row_id", "y", "group_id", "split_seeds", "fold_id"]:
        if not np.array_equal(getattr(a, f), getattr(b, f)):
            raise ValueError(f"banks differ in {f}; they cannot be paired")


def paired_cluster_bootstrap(a, b, n_boot=2000, seed=BOOTSTRAP_SEED, return_draws=False):
    """Difference in mean fold AUC, a minus b. Each replicate samples the groups with replacement; every row
    gets its group's multiplicity as a weight, shared by both banks and all repeats; folds are kept.
    A replicate with a single-class fold is invalid as a whole. Conditional on the fixed predictions."""
    if not isinstance(n_boot, (int, np.integer)) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    check_pairable(a, b)
    codes = np.unique(a.group_id, return_inverse=True)[1]
    G = int(codes.max()) + 1
    R, K = a.fold_auc.shape
    caches = []
    for r in range(R):
        for k in range(K):
            m = a.fold_id[r] == k
            caches.append((prepare_auc(a.y[m], a.oof[r, m], codes[m]), prepare_auc(b.y[m], b.oof[r, m], codes[m])))
    ones = np.ones(G)
    observed = float(np.mean([cached_weighted_auc(ca, ones) - cached_weighted_auc(cb, ones) for ca, cb in caches]))
    rng = np.random.default_rng(seed)
    draws, invalid = [], 0
    for _ in range(n_boot):
        mult = rng.multinomial(G, np.full(G, 1.0 / G)).astype("float64")
        diffs = [cached_weighted_auc(ca, mult) - cached_weighted_auc(cb, mult) for ca, cb in caches]
        if np.any(np.isnan(diffs)):
            invalid += 1
            continue
        draws.append(np.mean(diffs))
    if not draws:
        raise RuntimeError(f"no valid bootstrap replicate among {n_boot}; the comparison is unusable")
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out = {"observed_difference": observed, "interval_95": [float(lo), float(hi)], "n_boot": int(n_boot),
           "valid": int(len(draws)), "invalid": invalid, "seed": seed, "groups": G,
           "draws_positive_share": float((draws > 0).mean()),
           "interpretation": "conditional on fixed predictions; excludes retraining and selection variability"}
    if return_draws:
        out["draws"] = draws
    return out
