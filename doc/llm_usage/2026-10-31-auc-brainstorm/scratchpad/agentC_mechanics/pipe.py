"""Compact matched-experiment pipeline (agent C, mechanics).

Everything is evaluated on the fixed production_v1 folds (3 seeds x 5 stratified
group folds). A recipe is: base feature columns + a list of fold-fitted blocks +
a model factory. evaluate() returns the 15 fold AUCs and caches OOF predictions.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = "/Users/zwl0331/Desktop/Mini_project_1"
HERE = os.path.dirname(os.path.abspath(__file__))
OOF_DIR = os.path.join(HERE, "oof")
os.makedirs(OOF_DIR, exist_ok=True)
RESULTS = os.path.join(HERE, "results.jsonl")

AUDIO = ["time_signature", "speechiness", "danceability", "duration_ms", "energy",
         "mode", "instrumentalness", "valence", "key", "tempo", "loudness",
         "acousticness", "liveness"]
DATE_PARTS = ["year", "month", "day", "day_of_year", "weekday", "days", "date_precision"]
F2_COLS = AUDIO + DATE_PARTS
SEEDS = [155, 2026, 1029]
INNER_SEED = 0

LGB_BASE = dict(n_estimators=600, learning_rate=0.01, num_leaves=63, min_child_samples=10,
                colsample_bytree=0.7, subsample=0.7, subsample_freq=1, reg_alpha=0.0714,
                reg_lambda=1, random_state=0, n_jobs=2, deterministic=True,
                force_col_wise=True, verbose=-1)

_cache = {}


def f2(df):
    out = df[AUDIO].astype(float).copy()
    s = df["track_album_release_date"].astype(str)
    n = s.str.len()
    out["year"] = pd.to_numeric(s.str[:4], errors="coerce")
    out["month"] = pd.to_numeric(s.str[5:7], errors="coerce")
    out["day"] = pd.to_numeric(s.str[8:10], errors="coerce")
    full = pd.to_datetime(s.where(n == 10), format="%Y-%m-%d", errors="coerce")
    out["day_of_year"] = full.dt.dayofyear.astype(float)
    out["weekday"] = full.dt.weekday.astype(float)
    out["days"] = (full - pd.Timestamp("1970-01-01")).dt.days.astype(float)
    out["date_precision"] = n.astype(float)
    return out[F2_COLS]


def log_audio(df):
    """Audio space for the NN block: log1p(duration/1000), log1p(100 x) for the four
    bounded-skewed features, others raw. Standardisation happens inside the fold."""
    z = df[AUDIO].astype(float).copy()
    z["duration_ms"] = np.log1p(z["duration_ms"] / 1000.0)
    for c in ["speechiness", "instrumentalness", "acousticness", "liveness"]:
        z[c] = np.log1p(100.0 * z[c])
    return z.values


def load():
    if "data" in _cache:
        return _cache["data"]
    tr = pd.read_csv(f"{ROOT}/data/train.csv")
    sp = pd.read_csv(f"{ROOT}/artifacts/splits/production_v1.csv")
    assert (sp["train_row_id"].values == np.arange(len(tr))).all()
    keep = (sp["eligible"] == 1).values
    assert keep.sum() == 3863 and not keep[425]
    tr = tr[keep].reset_index(drop=True)
    sp = sp[keep].reset_index(drop=True)
    y = (tr["Popularity_Type"] == "High").astype(int).values
    data = dict(
        tr=tr,
        y=y,
        groups=sp["primary_group"].values,
        folds={s: sp[f"primary_fold_{s}"].values for s in SEEDS},
        dates=tr["track_album_release_date"].astype(str).values,
        X=f2(tr),
        Zlog=log_audio(tr),
    )
    _cache["data"] = data
    return data


def fold_indices():
    """Cached (seed, k) -> (tr_idx, va_idx)."""
    if "folds" in _cache:
        return _cache["folds"]
    d = load()
    out = {}
    for s in SEEDS:
        f = d["folds"][s]
        for k in range(5):
            out[(s, k)] = (np.where(f != k)[0], np.where(f == k)[0])
    _cache["folds"] = out
    return out


def inner_splits(tr_idx, seed=None):
    """5 StratifiedGroupKFold splits inside the training fold (positions into tr_idx)."""
    if seed is None:
        seed = INNER_SEED
    key = ("inner", tuple(tr_idx[:5]), len(tr_idx), seed)
    if key in _cache:
        return _cache[key]
    d = load()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    sp = list(sgkf.split(np.zeros(len(tr_idx)), d["y"][tr_idx], d["groups"][tr_idx]))
    _cache[key] = sp
    return sp


def crossfit(tr_idx, va_idx, fit_fn, apply_fn, k):
    """Training rows encoded by inner group cross-fitting; held-out rows by the whole
    training fold. fit_fn(ref_idx) -> state; apply_fn(state, query_idx) -> (n, k)."""
    tr_enc = np.full((len(tr_idx), k), np.nan)
    for in_tr, in_va in inner_splits(tr_idx):
        tr_enc[in_va] = apply_fn(fit_fn(tr_idx[in_tr]), tr_idx[in_va])
    va_enc = apply_fn(fit_fn(tr_idx), va_idx)
    return tr_enc, va_enc


# ---------------------------------------------------------------- blocks
# A block is f(tr_idx, va_idx) -> (tr_block, va_block[, meta]) with meta.get("categorical")

def block_date_count(tr_idx, va_idx):
    d = load()
    dates, groups = d["dates"], d["groups"]
    df = pd.DataFrame({"d": dates[tr_idx], "g": groups[tr_idx]}).drop_duplicates()
    cnt = df.groupby("d").size()
    tr_c = cnt.reindex(dates[tr_idx]).values.astype(float) - 1.0  # own group left out
    va_c = cnt.reindex(dates[va_idx]).fillna(0).values.astype(float)
    return np.log1p(tr_c)[:, None], np.log1p(va_c)[:, None]


def _te_stats(idx, per_group=False):
    d = load()
    df = pd.DataFrame({"d": d["dates"][idx], "y": d["y"][idx], "g": d["groups"][idx]})
    if per_group:  # one label per (date, group): duplicates count once
        df = df.groupby(["d", "g"], as_index=False)["y"].mean()
    stats = df.groupby("d")["y"].agg(["sum", "count"])
    return stats, df["y"].mean()


def make_block_date_te(m=3.0, per_group=False, with_count=False):
    """Smoothed mean label of the exact date string, (sum + m*prior)/(n + m), cross-fitted
    inside the training fold. m=0 gives the unsmoothed mean (prior for unseen dates).
    with_count adds the raw count n as a second column."""
    def fit_fn(ref):
        return _te_stats(ref, per_group)

    def apply_fn(state, q):
        stats, prior = state
        s = stats.reindex(load()["dates"][q])
        sm = s["sum"].fillna(0).values
        ct = s["count"].fillna(0).values
        if m > 0:
            enc = (sm + m * prior) / (ct + m)
        else:
            enc = np.where(ct > 0, sm / np.maximum(ct, 1), prior)
        return np.column_stack([enc, ct]) if with_count else enc[:, None]

    def block(tr_idx, va_idx):
        return crossfit(tr_idx, va_idx, fit_fn, apply_fn, 2 if with_count else 1)
    block.__name__ = f"date_te_m{m}{'_grp' if per_group else ''}{'_cnt' if with_count else ''}"
    return block


def block_date_nn(tr_idx, va_idx):
    """Within-date nearest-neighbour features in standardised log-audio space, cross-fitted
    like the target encoding: nn_label, nn_dist, kernel-weighted mean label (h = median
    training-row NN distance), n_same_date. NaN when no same-date neighbour."""
    d = load()
    y, dates = d["y"], d["dates"]
    Z = d["Zlog"]
    mu = Z[tr_idx].mean(0)
    sd = Z[tr_idx].std(0)
    sd[sd == 0] = 1.0
    Zs = (Z - mu) / sd

    def fit_fn(ref):
        by = {}
        for i in ref:
            by.setdefault(dates[i], []).append(i)
        return {k: np.asarray(v) for k, v in by.items()}

    def apply_fn(state, q):
        out = np.full((len(q), 3), np.nan)
        raw = [None] * len(q)
        for j, i in enumerate(q):
            cand = state.get(dates[i])
            if cand is None:
                continue
            dist = np.sqrt(((Zs[cand] - Zs[i]) ** 2).sum(1))
            k = int(np.argmin(dist))
            out[j] = (y[cand[k]], dist[k], len(cand))
            raw[j] = (dist, y[cand])
        return out, raw

    tr3 = np.full((len(tr_idx), 3), np.nan)
    tr_raw = [None] * len(tr_idx)
    for in_tr, in_va in inner_splits(tr_idx):
        o, r = apply_fn(fit_fn(tr_idx[in_tr]), tr_idx[in_va])
        tr3[in_va] = o
        for pos, rr in zip(in_va, r):
            tr_raw[pos] = rr
    va3, va_raw = apply_fn(fit_fn(tr_idx), va_idx)
    h = np.nanmedian(tr3[:, 1])
    if not np.isfinite(h) or h <= 0:
        h = 1.0

    def kern(raw):
        out = np.full(len(raw), np.nan)
        for j, rr in enumerate(raw):
            if rr is None:
                continue
            dist, lab = rr
            w = np.exp(-dist / h)
            out[j] = (w * lab).sum() / w.sum()
        return out

    trb = np.column_stack([tr3[:, 0], tr3[:, 1], kern(tr_raw), tr3[:, 2]])
    vab = np.column_stack([va3[:, 0], va3[:, 1], kern(va_raw), va3[:, 2]])
    return trb, vab


def block_date_cat(tr_idx, va_idx):
    """Exact date string as an integer code built on the training fold; unseen held-out
    dates get a code the model never saw. Declared categorical by the recipe."""
    d = load()
    dates = d["dates"]
    uniq = np.unique(dates[tr_idx])
    code = {s: i for i, s in enumerate(uniq)}
    unseen = len(uniq)
    tr_c = np.array([code[s] for s in dates[tr_idx]], dtype=float)
    va_c = np.array([code.get(s, unseen) for s in dates[va_idx]], dtype=float)
    return tr_c[:, None], va_c[:, None], {"categorical": True}


def block_date_str(tr_idx, va_idx):
    """Raw date string (object) for CatBoost."""
    dates = load()["dates"]
    return dates[tr_idx][:, None], dates[va_idx][:, None], {"categorical": True, "string": True}


# ---------------------------------------------------------------- models

def lgb_factory(**over):
    import lightgbm as lgb
    params = dict(LGB_BASE)
    params.update(over)

    def make(seed):
        p = dict(params)
        p["random_state"] = seed
        return lgb.LGBMClassifier(**p)
    make.params = params
    return make


# ---------------------------------------------------------------- evaluation

def build_fold(tr_idx, va_idx, base_cols, blocks):
    d = load()
    Xb = d["X"][list(base_cols)].values.astype(float)
    Xtr = [Xb[tr_idx]]
    Xva = [Xb[va_idx]]
    cat_cols = []
    names = list(base_cols)
    col = Xb.shape[1]
    string_cols = []
    for b in blocks:
        res = b(tr_idx, va_idx)
        btr, bva = res[0], res[1]
        meta = res[2] if len(res) > 2 else {}
        if meta.get("categorical"):
            cat_cols.extend(range(col, col + btr.shape[1]))
        if meta.get("string"):
            string_cols.extend(range(col, col + btr.shape[1]))
        names.extend([f"{b.__name__}_{j}" for j in range(btr.shape[1])])
        col += btr.shape[1]
        Xtr.append(btr)
        Xva.append(bva)
    if string_cols:
        Xtr = pd.DataFrame(np.column_stack([np.asarray(a, dtype=object) for a in Xtr]), columns=names)
        Xva = pd.DataFrame(np.column_stack([np.asarray(a, dtype=object) for a in Xva]), columns=names)
        for j, c in enumerate(names):
            if j not in string_cols:
                Xtr[c] = Xtr[c].astype(float)
                Xva[c] = Xva[c].astype(float)
    else:
        Xtr = np.hstack(Xtr)
        Xva = np.hstack(Xva)
    return Xtr, Xva, cat_cols, names


def evaluate(name, model_fn, blocks=(), base_cols=F2_COLS, model_seeds=(0,), fit_kwargs=None,
             fit_hook=None, note="", save_oof=True, verbose=True):
    """Return dict with per-fold AUCs, per-seed means and the grand mean; append to results."""
    d = load()
    y = d["y"]
    n = len(y)
    fi = fold_indices()
    fold_auc = {}
    oof = {s: np.full(n, np.nan) for s in SEEDS}
    t0 = time.time()
    for s in SEEDS:
        for k in range(5):
            tr_idx, va_idx = fi[(s, k)]
            Xtr, Xva, cat_cols, names = build_fold(tr_idx, va_idx, base_cols, blocks)
            p = np.zeros(len(va_idx))
            for ms in model_seeds:
                m = model_fn(ms)
                kw = dict(fit_kwargs or {})
                if fit_hook is not None:
                    kw = fit_hook(kw, cat_cols, names, Xtr)
                m.fit(Xtr, y[tr_idx], **kw)
                p += m.predict_proba(Xva)[:, 1]
            p /= len(model_seeds)
            oof[s][va_idx] = p
            fold_auc[(s, k)] = float(roc_auc_score(y[va_idx], p))
    elapsed = time.time() - t0
    rec = summarize(name, fold_auc, note=note, elapsed=elapsed,
                    params=getattr(model_fn, "params", None),
                    blocks=[b.__name__ for b in blocks], base_cols=list(base_cols),
                    model_seeds=list(model_seeds))
    if save_oof:
        np.savez(os.path.join(OOF_DIR, f"{name}.npz"), **{str(s): oof[s] for s in SEEDS})
    if verbose:
        print(fmt(rec), flush=True)
    return rec


def summarize(name, fold_auc, **extra):
    per_seed = {str(s): float(np.mean([fold_auc[(s, k)] for k in range(5)])) for s in SEEDS}
    rec = dict(name=name, mean=float(np.mean(list(fold_auc.values()))), per_seed=per_seed,
               folds={f"{s}_{k}": v for (s, k), v in fold_auc.items()}, **extra)
    rec = json.loads(json.dumps(rec, default=str))
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def fmt(rec):
    ps = rec["per_seed"]
    return (f"{rec['name']:<34s} mean={rec['mean']:.4f}  "
            f"s155={ps['155']:.4f} s2026={ps['2026']:.4f} s1029={ps['1029']:.4f}  "
            f"t={rec.get('elapsed', 0):.0f}s")


def auc_from_oof(name, oof):
    """Per-fold AUCs from an OOF dict {seed: array}."""
    d = load()
    fi = fold_indices()
    out = {}
    for s in SEEDS:
        for k in range(5):
            _, va_idx = fi[(s, k)]
            out[(s, k)] = float(roc_auc_score(d["y"][va_idx], oof[s][va_idx]))
    return out


def load_oof(name):
    z = np.load(os.path.join(OOF_DIR, f"{name}.npz"))
    return {s: z[str(s)] for s in SEEDS}


def lgb_cat_hook(kw, cat_cols, names, Xtr):
    if cat_cols:
        kw["categorical_feature"] = list(cat_cols)
    return kw
