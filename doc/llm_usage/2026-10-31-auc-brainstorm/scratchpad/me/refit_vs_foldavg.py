"""Deployment check inside the CV protocol: for each outer fold, compare
(a) one model fitted on the whole outer training part (the 'full refit' deployment) with
(b) the average of five inner-fold models each fitted on 80% of the outer training part (the project's
    'fold_average' deployment, mirrored one level down),
both scored on the outer validation fold. Recipe: the project's current best L+ (tuned LightGBM on F2 with
date_count and date_te3). Nothing is written into the project; 2 threads."""
import os, sys, time, re
from datetime import date
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import TargetEncoder

ROOT = "/Users/zwl0331/Desktop/Mini_project_1"
AUDIO = ["time_signature", "speechiness", "danceability", "duration_ms", "energy", "mode", "instrumentalness",
         "valence", "key", "tempo", "loudness", "acousticness", "liveness"]
DATE = "track_album_release_date"
PARAMS = dict(n_estimators=600, learning_rate=0.01, num_leaves=63, min_child_samples=10, colsample_bytree=0.7,
              subsample=0.7, subsample_freq=1, reg_alpha=0.0714, reg_lambda=1)
EPOCH = date(1970, 1, 1)
PAT = re.compile(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?")


def parse(s):
    m = PAT.fullmatch(s)
    y, mo, d = int(m.group(1)), m.group(2), m.group(3)
    out = dict(year=float(y), month=np.nan, day=np.nan, day_of_year=np.nan, weekday=np.nan,
               date_precision=float(len(s)), days=np.nan)
    if mo is not None:
        out["month"] = float(mo)
    if d is not None:
        dd = date(y, int(mo), int(d))
        out["day"], out["day_of_year"], out["weekday"] = float(dd.day), float(dd.timetuple().tm_yday), float(dd.weekday())
        out["days"] = float((dd - EPOCH).days)
    return out


def f2(frame):
    audio = frame[AUDIO].astype("float64").reset_index(drop=True)
    dates = pd.DataFrame([parse(s) for s in frame[DATE]])
    return pd.concat([audio, dates], axis=1)


def inner_splits(y, groups, seed, n=5):
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in sgkf.split(np.zeros((len(y), 1)), y, groups)]


def encode(train_dates, y, groups, other_dates, seed, m=3.0):
    """date_count and date_te{m} fitted on the training rows (cross-fitted for them), transformed for others."""
    # date_count: distinct groups per date, own group excluded for the training rows
    pairs = pd.DataFrame({"d": train_dates, "g": groups}).drop_duplicates()
    cnt = pairs.groupby("d").size().to_dict()
    tr_count = np.log1p(np.array([cnt[d] for d in train_dates], dtype=float) - 1)
    ot_count = np.log1p(np.array([cnt.get(d, 0) for d in other_dates], dtype=float))
    # date_te: smoothed mean label of the date string, cross-fitted on inner grouped folds
    splits = inner_splits(y, groups, seed)
    enc = TargetEncoder(categories="auto", target_type="binary", smooth=m, cv=splits)
    tr_te = enc.fit_transform(np.asarray(train_dates, dtype=object).reshape(-1, 1), y)[:, 0]
    ot_te = enc.transform(np.asarray(other_dates, dtype=object).reshape(-1, 1))[:, 0]
    return tr_count, tr_te, ot_count, ot_te


def fit_predict(Xtr, ytr, Xte, seed=0, n_estimators=None):
    p = dict(PARAMS)
    if n_estimators:
        p["n_estimators"] = n_estimators
    model = lgb.LGBMClassifier(**p, random_state=seed, n_jobs=2, deterministic=True, force_col_wise=True, verbose=-1)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def main():
    train = pd.read_csv(f"{ROOT}/data/train.csv")
    folds = pd.read_csv(f"{ROOT}/artifacts/splits/production_v1.csv")
    elig = folds["eligible"].to_numpy() == 1
    y_all = (train["Popularity_Type"] == "High").astype(int).to_numpy()
    X_all = f2(train)
    dates_all = train[DATE].to_numpy()
    groups_all = folds["primary_group"].to_numpy()
    results = []
    t0 = time.time()
    for seed in (155, 2026, 1029):
        fold = folds[f"primary_fold_{seed}"].to_numpy()
        for k in range(5):
            tr = np.flatnonzero(elig & (fold != k)); va = np.flatnonzero(elig & (fold == k))
            ytr, yva = y_all[tr], y_all[va]
            gtr = groups_all[tr]
            # (a) full refit on the outer training part
            c1, t1, c2, t2 = encode(dates_all[tr], ytr, gtr, dates_all[va], seed=1000 + seed * 10 + k)
            Xtr = X_all.iloc[tr].reset_index(drop=True).assign(date_count=c1, date_te3=t1)
            Xva = X_all.iloc[va].reset_index(drop=True).assign(date_count=c2, date_te3=t2)
            p_full = fit_predict(Xtr, ytr, Xva)
            p_full750 = fit_predict(Xtr, ytr, Xva, n_estimators=750)
            p_full3 = np.mean([fit_predict(Xtr, ytr, Xva, seed=s) for s in (0, 1, 2)], axis=0)
            # (b) fold average: five inner models on 80% of the outer training part each
            preds = []
            for j, (itr, ite) in enumerate(inner_splits(ytr, gtr, seed=2000 + seed * 10 + k)):
                rows = tr[itr]
                c1, t1, c2, t2 = encode(dates_all[rows], y_all[rows], groups_all[rows], dates_all[va],
                                        seed=3000 + seed * 100 + k * 10 + j)
                Xi = X_all.iloc[rows].reset_index(drop=True).assign(date_count=c1, date_te3=t1)
                Xv = X_all.iloc[va].reset_index(drop=True).assign(date_count=c2, date_te3=t2)
                preds.append(fit_predict(Xi, y_all[rows], Xv))
            p_avg = np.mean(preds, axis=0)
            r = dict(seed=seed, fold=k, full=roc_auc_score(yva, p_full), full750=roc_auc_score(yva, p_full750),
                     full3seeds=roc_auc_score(yva, p_full3), fold_avg=roc_auc_score(yva, p_avg),
                     corr_full_avg=np.corrcoef(p_full, p_avg)[0, 1])
            results.append(r)
            print(f"seed {seed} fold {k}: full {r['full']:.4f} full750 {r['full750']:.4f} full3 {r['full3seeds']:.4f} "
                  f"fold_avg {r['fold_avg']:.4f} corr {r['corr_full_avg']:.3f} ({time.time()-t0:.0f} s)", flush=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.path.dirname(__file__), "refit_vs_foldavg_results.csv"), index=False)
    print("\nMean over 15 folds:")
    for c in ("full", "full750", "full3seeds", "fold_avg"):
        print(f"  {c:12s} {df[c].mean():.4f}   per seed " + ", ".join(f"{v:.4f}" for v in df.groupby('seed')[c].mean().reindex([155, 2026, 1029])))
    for c in ("full", "full750", "full3seeds"):
        d = df[c] - df["fold_avg"]
        print(f"  {c} minus fold_avg: {d.mean():+.4f}  per seed " + ", ".join(f"{v:+.4f}" for v in d.groupby(df['seed']).mean().reindex([155, 2026, 1029]))
              + f"  folds positive {int((d > 0).sum())}/15")


if __name__ == "__main__":
    main()
