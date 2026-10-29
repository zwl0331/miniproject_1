"""Scoping run for the modeling strategy (task P0.2, doc/plans/modeling_strategy.md section 2).

Sizing only: these numbers set priorities. They are not findings; phase 1 tasks
re-establish the facts they depend on.

Usage, from the project root: .venv/bin/python analysis/scoping.py
Output: printed; the run of 2026-10-29 is kept in analysis/scoping_output.txt.
"""
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = ["time_signature", "speechiness", "danceability", "duration_ms", "energy", "mode", "instrumentalness",
         "valence", "key", "tempo", "loudness", "acousticness", "liveness"]
DATE = "track_album_release_date"


def date_feats(df):
    d = df[DATE].astype(str)
    out = pd.DataFrame(index=df.index)
    out["year"] = d.str[:4].astype(float)
    out["month"] = pd.to_numeric(d.str[5:7], errors="coerce")
    out["day"] = pd.to_numeric(d.str[8:10], errors="coerce")
    out["date_precision"] = d.str.len()
    full = pd.to_datetime(d.where(d.str.len() == 10), errors="coerce")
    days = (full - pd.Timestamp("1970-01-01")).dt.days
    out["days"] = days.fillna((out["year"] - 1970) * 365.25 + 182)
    return out


class LogLR:
    """Logistic regression on standardized features: log1p(x / 1000) for duration_ms, log1p(100 x) for the
    other skewed features, then median imputation with medians of the transformed training columns."""
    SKEWED = ["duration_ms", "instrumentalness", "speechiness", "liveness", "acousticness"]

    def _log(self, X):
        X = X.copy()
        for c in self.SKEWED:
            if c in X:
                X[c] = np.log1p(X[c] * (1e-3 if c == "duration_ms" else 100))
        return X

    def fit(self, X, y):
        Xt = self._log(X)
        self.med = Xt.median()
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xt.fillna(self.med), y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(self._log(X).fillna(self.med))


MODELS = {
    "logreg_logfeat": lambda: LogLR(),
    "random_forest": lambda: RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=0),
    "extra_trees": lambda: ExtraTreesClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=0),
    "hist_gb": lambda: HistGradientBoostingClassifier(random_state=0),
    "lgbm_default": lambda: lgb.LGBMClassifier(verbose=-1, random_state=0),
    "lgbm_slow": lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=20,
                                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                                            random_state=0),
    "xgb_default": lambda: xgb.XGBClassifier(eval_metric="auc", random_state=0, n_jobs=-1),
}


def cv_auc(make, X, y, groups, grouped=True, n_rep=3):
    """Mean fold AUC over n_rep repeats of 5 folds, and a naive standard error of that mean that treats the
    5 x n_rep folds as independent; repeated folds share rows, so it understates the uncertainty."""
    scores = []
    for rep in range(n_rep):
        if grouped:
            splits = StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, groups)
        else:
            splits = StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y)
        for tri, vai in splits:
            m = make().fit(X.iloc[tri], y[tri])
            scores.append(roc_auc_score(y[vai], m.predict_proba(X.iloc[vai])[:, 1]))
    return np.mean(scores), np.std(scores) / np.sqrt(len(scores))


def hanley_mcneil_se(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))


def main():
    tr = pd.read_csv(f"{ROOT}/data/train.csv")
    te = pd.read_csv(f"{ROOT}/data/test.csv")
    n_raw = len(tr)
    tr = tr[tr["uri"].notna()].reset_index(drop=True)  # drops the one row without audio features
    y = (tr["Popularity_Type"] == "High").astype(int).values
    groups = tr["uri"].values
    X_audio = tr[AUDIO].copy()
    X_full = pd.concat([X_audio, date_feats(tr)], axis=1)
    print(f"[1] rows {n_raw} raw, {len(tr)} used; High rate {y.mean():.4f}; distinct uri {len(set(groups))}")

    print("[2] 5-fold StratifiedGroupKFold by uri, 3 repeats: mean fold AUC +/- naive SE (folds as independent)")
    for name, make in MODELS.items():
        t = time.time()
        mf, sf = cv_auc(make, X_full, y, groups)
        ma, sa = cv_auc(make, X_audio, y, groups)
        print(f"    {name:15s} audio+date {mf:.4f} +/- {sf:.4f}   audio only {ma:.4f} +/- {sa:.4f}   ({time.time() - t:.0f} s)")

    print("[3] leakage check: ungrouped vs grouped folds, audio+date")
    for name in ["random_forest", "lgbm_default"]:
        mp, sp = cv_auc(MODELS[name], X_full, y, groups, grouped=False)
        mg, sg = cv_auc(MODELS[name], X_full, y, groups)
        print(f"    {name:15s} ungrouped {mp:.4f} +/- {sp:.4f}   grouped {mg:.4f} +/- {sg:.4f}   difference {mp - mg:+.4f}")

    # Descriptive only: the modeling strategy (section 3) excludes fitting anything on test rows. An earlier
    # version of this block fitted a train-versus-test classifier; it was removed in P0.2, review round 2.
    te_year = date_feats(te)["year"]
    print("[4] release year percentiles 10/25/50/75/90 (descriptive), train:",
          np.nanpercentile(X_full["year"], [10, 25, 50, 75, 90]), "test:", np.nanpercentile(te_year, [10, 25, 50, 75, 90]))

    raw = pd.read_csv(f"{ROOT}/data/train.csv")
    yr = (raw["Popularity_Type"] == "High").astype(int).values
    print(f"[5] row order: AUC of the train row index for High {roc_auc_score(yr, np.arange(len(raw))):.4f}; "
          f"corr(test ID, year) {np.corrcoef(te['ID'], te[DATE].str[:4].astype(float))[0, 1]:+.4f}")

    d_tr, d_te = tr[DATE], te[DATE]
    oof_aucs, year_aucs = [], []
    for rep in range(3):
        oof = np.zeros(len(tr))
        for tri, vai in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(tr, y, groups):
            prior, alpha = y[tri].mean(), 10
            st = pd.DataFrame({"d": d_tr.iloc[tri].values, "y": y[tri]}).groupby("d")["y"].agg(["sum", "count"])
            s = d_tr.iloc[vai].map(st["sum"]).fillna(0).values
            c = d_tr.iloc[vai].map(st["count"]).fillna(0).values
            oof[vai] = (s + alpha * prior) / (c + alpha)
        oof_aucs.append(roc_auc_score(y, oof))
    year_auc = roc_auc_score(y, -X_full["year"])
    print(f"[6] exact release date: {d_te.isin(set(d_tr)).mean():.3f} of test rows share a date with a train row; "
          f"OOF smoothed target encoding AUC {np.mean(oof_aucs):.4f}; year alone (older ranks higher) {year_auc:.4f}")

    print("[7] public leaderboard noise, Hanley-McNeil SE of one AUC of 0.83 at the train High rate "
          "(hypothetical public shares; not the SE of a difference between two models):")
    for share in [0.3, 0.5, 1.0]:
        n = int(round(863 * share))
        n_pos = int(round(n * y.mean()))
        print(f"    {share:.0%} of 863 test rows ({n_pos} High, {n - n_pos} Low): SE {hanley_mcneil_se(0.83, n_pos, n - n_pos):.4f}")

    X_year = pd.concat([X_audio, date_feats(tr)[["year"]]], axis=1)
    my, sy = cv_auc(MODELS["lgbm_slow"], X_year, y, groups)
    mf, sf = cv_auc(MODELS["lgbm_slow"], X_full, y, groups)
    print(f"[8] lgbm_slow, same folds: audio + year {my:.4f}; audio + all scoping date fields {mf:.4f}; "
          f"fields beyond year add {mf - my:+.4f}")


if __name__ == "__main__":
    main()
