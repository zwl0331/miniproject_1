"""Exact-date coverage and distinct-group support under the actual fitting dictionaries (plan sections 6.4 and 18),
descriptive only. For each split seed and outer fold: the outer validation rows and the test rows against the outer
fitting rows' dates; and, inside the outer fitting rows, each inner held-out part against its inner fitting part,
with the inner splits the date target encoding uses (FitContext(seed, fold).inner_seed("date_te")). Support is the
number of distinct primary groups of the fitting part holding the exact raw date string. Fits no model and
computes no score; writes nothing but its standard output. From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -I analysis/p24_date_coverage.py . > analysis/p24_date_coverage_output.txt
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))

from songpop.data import DATE  # noqa: E402
from songpop.phase2 import runs  # noqa: E402
from songpop.phase2.encoders import FitContext, inner_group_splits  # noqa: E402
from songpop.splits import SPLIT_SEEDS  # noqa: E402

BINS = (("0", 0, 0), ("1", 1, 1), ("2-4", 2, 4), ("5+", 5, 10 ** 9))


def support(fit_dates, fit_groups):
    out = {}
    for d, g in zip(fit_dates, fit_groups):
        out.setdefault(d, set()).add(g)
    return {d: len(s) for d, s in out.items()}


def summary(query_dates, sup):
    s = np.array([sup.get(d, 0) for d in query_dates])
    rows = {name: float(np.mean((s >= lo) & (s <= hi))) for name, lo, hi in BINS}
    uniq = sorted(set(query_dates))
    su = np.array([sup.get(d, 0) for d in uniq])
    dates = {name: float(np.mean((su >= lo) & (su <= hi))) for name, lo, hi in BINS}
    return float(np.mean(s > 0)), rows, float(np.mean(su > 0)), dates


def fmt(cov, rows, dcov, dates):
    return (f"rows covered {cov:.4f} (support {', '.join(f'{k}: {v:.3f}' for k, v in rows.items())}); "
            f"distinct dates covered {dcov:.4f} (support {', '.join(f'{k}: {v:.3f}' for k, v in dates.items())})")


def main():
    ctx = runs.Context.load(ROOT)
    dates = ctx.rows[DATE].astype(str).to_numpy()
    test_dates = ctx.test_rows[DATE].astype(str).to_numpy()
    y, groups = np.asarray(ctx.data.y), np.asarray(ctx.groups)
    means = {"validation": [], "test": [], "inner": []}
    pooled = {"validation": [0, 0], "inner": [0, 0]}  # covered rows, rows
    print("Exact-date coverage under the actual fitting dictionaries (descriptive; no model, no score)")
    for seed in SPLIT_SEEDS:
        fold_id = ctx.fold_id([seed])[0]
        seed_pooled = [0, 0]
        for k in range(5):
            tr, va = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            sup = support(dates[tr], groups[tr])
            v, t = summary(dates[va], sup), summary(test_dates, sup)
            covered = int(sum(sup.get(d, 0) > 0 for d in dates[va]))
            seed_pooled[0] += covered
            seed_pooled[1] += len(va)
            means["validation"].append(v[0])
            means["test"].append(t[0])
            print(f"seed {seed} fold {k}: validation ({covered} of {len(va)} rows) {fmt(*v)}")
            print(f"                 test {fmt(*t)}")
            fc = FitContext(int(seed), k)
            inner = []
            for j, (itr, ite) in enumerate(inner_group_splits(y[tr], groups[tr], fc.inner_seed("date_te"))):
                isup = support(dates[tr][itr], groups[tr][itr])
                q = dates[tr][ite]
                c = int(sum(isup.get(d, 0) > 0 for d in q))
                pooled["inner"][0] += c
                pooled["inner"][1] += len(q)
                inner.append(summary(q, isup)[0])
                print(f"                 inner part {j} ({c} of {len(q)} rows) {fmt(*summary(q, isup))}")
            means["inner"].append(float(np.mean(inner)))
        pooled["validation"][0] += seed_pooled[0]
        pooled["validation"][1] += seed_pooled[1]
        print(f"seed {seed}: validation rows covered, pooled over its five folds: {seed_pooled[0]} of {seed_pooled[1]} "
              f"({seed_pooled[0] / seed_pooled[1]:.4f})")
    print()
    for name, xs in means.items():
        print(f"mean over the 15 outer fits of the per-fit share, {name} rows covered: {np.mean(xs):.4f} "
              f"(min {min(xs):.4f}, max {max(xs):.4f})")
    for name, (c, n) in pooled.items():
        print(f"pooled {name} rows covered: {c} of {n} ({c / n:.4f})")


if __name__ == "__main__":
    main()
