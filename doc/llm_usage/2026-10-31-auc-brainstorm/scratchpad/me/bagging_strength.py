"""Follow-up to refit_vs_foldavg.py: is more bagging a lever, and where does the fold-average gain come from?
Inside each outer fold (same 15 folds, recipe L+), score on the outer validation fold:
  full        one model on the whole outer training part (reference)
  te3avg      three full models differing ONLY in the target encoder's inner cross-fit seed, averaged
              (isolates the encoder's cross-fit diversity)
  fold_avg5   five inner-fold models (80% of the outer training part each), averaged (as before)
  fold_avg10  two inner-split realizations x five folds = ten models, averaged
  bag10       ten group-subsample bags (random 80% of the outer training groups each, encoders refitted per bag)
Nothing is written into the project; 2 threads."""
import os, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import refit_vs_foldavg as base


def model_on(rows, va, X_all, y_all, dates_all, groups_all, enc_seed, est_seed=0):
    c1, t1, c2, t2 = base.encode(dates_all[rows], y_all[rows], groups_all[rows], dates_all[va], seed=enc_seed)
    Xtr = X_all.iloc[rows].reset_index(drop=True).assign(date_count=c1, date_te3=t1)
    Xva = X_all.iloc[va].reset_index(drop=True).assign(date_count=c2, date_te3=t2)
    return base.fit_predict(Xtr, y_all[rows], Xva, seed=est_seed)


def main():
    train = pd.read_csv(f"{base.ROOT}/data/train.csv")
    folds = pd.read_csv(f"{base.ROOT}/artifacts/splits/production_v1.csv")
    elig = folds["eligible"].to_numpy() == 1
    y_all = (train["Popularity_Type"] == "High").astype(int).to_numpy()
    X_all = base.f2(train)
    dates_all = train[base.DATE].to_numpy()
    groups_all = folds["primary_group"].to_numpy()
    results, t0 = [], time.time()
    for seed in (155, 2026, 1029):
        fold = folds[f"primary_fold_{seed}"].to_numpy()
        for k in range(5):
            tr = np.flatnonzero(elig & (fold != k)); va = np.flatnonzero(elig & (fold == k))
            yva, gtr = y_all[va], groups_all[tr]
            p_full = model_on(tr, va, X_all, y_all, dates_all, groups_all, enc_seed=1000 + seed * 10 + k)
            p_te3 = np.mean([p_full] + [model_on(tr, va, X_all, y_all, dates_all, groups_all, enc_seed=7000 + seed * 100 + k * 10 + j)
                                         for j in (1, 2)], axis=0)
            fa = []
            for rep in (0, 1):
                for j, (itr, ite) in enumerate(base.inner_splits(y_all[tr], gtr, seed=2000 + seed * 10 + k + 500 * rep)):
                    rows = tr[itr]
                    fa.append(model_on(rows, va, X_all, y_all, dates_all, groups_all,
                                       enc_seed=3000 + seed * 100 + k * 10 + j + 5000 * rep))
            p_fa5, p_fa10 = np.mean(fa[:5], axis=0), np.mean(fa, axis=0)
            bags = []
            gss = GroupShuffleSplit(n_splits=10, train_size=0.8, random_state=4000 + seed * 10 + k)
            for j, (btr, _) in enumerate(gss.split(np.zeros((len(tr), 1)), y_all[tr], gtr)):
                rows = tr[btr]
                bags.append(model_on(rows, va, X_all, y_all, dates_all, groups_all, enc_seed=6000 + seed * 100 + k * 10 + j))
            p_bag = np.mean(bags, axis=0)
            r = dict(seed=seed, fold=k, full=roc_auc_score(yva, p_full), te3avg=roc_auc_score(yva, p_te3),
                     fold_avg5=roc_auc_score(yva, p_fa5), fold_avg10=roc_auc_score(yva, p_fa10), bag10=roc_auc_score(yva, p_bag))
            results.append(r)
            print(f"seed {seed} fold {k}: " + " ".join(f"{c} {r[c]:.4f}" for c in ("full", "te3avg", "fold_avg5", "fold_avg10", "bag10"))
                  + f" ({time.time()-t0:.0f} s)", flush=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bagging_strength_results.csv"), index=False)
    print("\nMean over 15 folds (per seed 155, 2026, 1029):")
    for c in ("full", "te3avg", "fold_avg5", "fold_avg10", "bag10"):
        print(f"  {c:10s} {df[c].mean():.4f}   " + ", ".join(f"{v:.4f}" for v in df.groupby('seed')[c].mean().reindex([155, 2026, 1029])))
    for c in ("te3avg", "fold_avg5", "fold_avg10", "bag10"):
        d = df[c] - df["full"]
        print(f"  {c} minus full: {d.mean():+.4f}  per seed " + ", ".join(f"{v:+.4f}" for v in d.groupby(df['seed']).mean().reindex([155, 2026, 1029]))
              + f"  folds positive {int((d > 0).sum())}/15")
    d = df["fold_avg10"] - df["fold_avg5"]
    print(f"  fold_avg10 minus fold_avg5: {d.mean():+.4f}  folds positive {int((d > 0).sum())}/15")


if __name__ == "__main__":
    main()
