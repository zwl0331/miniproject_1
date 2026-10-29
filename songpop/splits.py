"""Track groups and saved folds (doc/plans/phase1-data-baseline.md section 4.3; Finding 002)."""
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from songpop.data import AUDIO, DATE

SPLIT_SEEDS = (155, 2026, 1029)
N_SPLITS = 5
SPLIT_VERSION = "production_v1"
ERAS = [(-np.inf, 1989, "before 1990"), (1990, 1999, "1990s"), (2000, 2009, "2000s"), (2010, 2019, "2010s"),
        (2020, np.inf, "2020 on")]


def _components(n, key_lists):
    """Union-find over positions 0..n-1, joining positions that share a non-None key in any list.
    Each component is labeled by its smallest position."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for keys in key_lists:
        first = {}
        for i, k in enumerate(keys):
            if k is None:
                continue
            j = first.setdefault(k, i)
            a, b = find(i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(n)])


def _keys(train, cols):
    complete = train[cols].notna().all(axis=1).to_numpy()
    values = zip(*[train[c].astype("float64") if c in AUDIO else train[c].astype(str) for c in cols])
    return [k if ok else None for k, ok in zip(values, complete)]


def make_groups(train, strict=False):
    """Group id per training row: the smallest train_row_id of its connected component. Rows are joined by
    the same nonmissing uri and by identical audio values plus date string; strict also joins identical audio
    values alone. Labels, URLs and test rows play no part."""
    order = np.argsort(train["train_row_id"].to_numpy(), kind="stable")
    t = train.iloc[order]
    key_lists = [[None if pd.isna(u) else u for u in t["uri"]], _keys(t, AUDIO + [DATE])]
    if strict:
        key_lists.append(_keys(t, AUDIO))
    comp = _components(len(t), key_lists)
    ids = t["train_row_id"].to_numpy()[comp]
    out = np.empty(len(train), dtype=np.int64)
    out[order] = ids
    return out


def incomplete_fold(seed, row_id):
    token = f"incomplete-v1:{seed}:{row_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % N_SPLITS


def assign_folds(y_all, groups, eligible, row_ids, seeds=SPLIT_SEEDS):
    """Fold id per seed and training row: StratifiedGroupKFold on the eligible rows; each ineligible row gets
    the hashed fold of incomplete_fold without changing any eligible row's fold."""
    folds = np.full((len(seeds), len(y_all)), -1, dtype=np.int64)
    idx = np.flatnonzero(eligible)
    for r, seed in enumerate(seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for k, (_, va) in enumerate(sgkf.split(np.zeros((len(idx), 1)), y_all[idx], groups[idx])):
            folds[r, idx[va]] = k
        for i in np.flatnonzero(~eligible):
            folds[r, i] = incomplete_fold(seed, int(row_ids[i]))
    return folds


@dataclass
class FoldSet:
    train_row_id: np.ndarray
    eligible: np.ndarray
    primary_group: np.ndarray
    strict_group: np.ndarray
    seeds: tuple
    primary: np.ndarray  # (len(seeds), n_rows)
    strict: np.ndarray

    def to_frame(self):
        d = {"train_row_id": self.train_row_id, "eligible": self.eligible.astype(int),
             "primary_group": self.primary_group, "strict_group": self.strict_group}
        for name, folds in [("primary", self.primary), ("strict", self.strict)]:
            for r, s in enumerate(self.seeds):
                d[f"{name}_fold_{s}"] = folds[r]
        return pd.DataFrame(d)

    @classmethod
    def columns(cls, seeds=SPLIT_SEEDS):
        return (["train_row_id", "eligible", "primary_group", "strict_group"]
                + [f"{k}_fold_{s}" for k in ("primary", "strict") for s in seeds])

    @classmethod
    def from_frame(cls, df, seeds=SPLIT_SEEDS):
        """Validate the split table before converting anything: the exact columns, integer values, unique row
        ids 0..n-1, eligibility exactly 0 or 1, group ids that are row ids, folds in 0..N_SPLITS-1."""
        if list(df.columns) != cls.columns(seeds):
            raise ValueError(f"split columns {list(df.columns)}, expected {cls.columns(seeds)}")
        for c in df.columns:
            if not pd.api.types.is_integer_dtype(df[c]):
                raise ValueError(f"split column {c} is not integer")
        n = len(df)
        if not np.array_equal(df["train_row_id"].to_numpy(), np.arange(n)):
            raise ValueError("split row ids are not 0..n-1 in order")
        if not set(df["eligible"].unique()) <= {0, 1}:
            raise ValueError("split eligibility values are not 0 or 1")
        for c in ("primary_group", "strict_group"):
            if df[c].min() < 0 or df[c].max() >= n:
                raise ValueError(f"split column {c} holds values that are not row ids")
        for c in cls.columns(seeds)[4:]:
            if not set(df[c].unique()) <= set(range(N_SPLITS)):
                raise ValueError(f"split column {c} holds fold ids outside 0..{N_SPLITS - 1}")
        return cls(df["train_row_id"].to_numpy(), df["eligible"].to_numpy() == 1,
                   df["primary_group"].to_numpy(), df["strict_group"].to_numpy(), tuple(seeds),
                   np.vstack([df[f"primary_fold_{s}"].to_numpy() for s in seeds]),
                   np.vstack([df[f"strict_fold_{s}"].to_numpy() for s in seeds]))

    def equals(self, other):
        return (self.seeds == other.seeds and all(np.array_equal(getattr(self, f), getattr(other, f)) for f in
                ["train_row_id", "eligible", "primary_group", "strict_group", "primary", "strict"]))


def build_foldset(data, seeds=SPLIT_SEEDS):
    row_ids = data.train["train_row_id"].to_numpy()
    primary_group = make_groups(data.train)
    strict_group = make_groups(data.train, strict=True)
    return FoldSet(row_ids, data.eligible, primary_group, strict_group, tuple(seeds),
                   assign_folds(data.y_all, primary_group, data.eligible, row_ids, seeds),
                   assign_folds(data.y_all, strict_group, data.eligible, row_ids, seeds))


def era_of(years):
    out = np.empty(len(years), dtype=object)
    for lo, hi, name in ERAS:
        out[(years >= lo) & (years <= hi)] = name
    return out


def fold_report(data, fs, kind="primary"):
    """One row per seed, fold and side (training, validation) over the eligible rows: rows, groups, High and
    Low counts, High rate, era shares, mixed-label groups, groups shared with the other side. Returns the table
    and a list of warnings (validation High rate off by more than 0.03, an era share off by more than 0.05);
    raises if a group is on both sides or a side lacks a class."""
    folds = getattr(fs, kind)
    groups = getattr(fs, f"{kind}_group")
    e = fs.eligible
    y = data.y_all
    era = era_of(data.train["track_album_release_date"].str[:4].astype(int).to_numpy())
    era_all = pd.Series(era[e]).value_counts(normalize=True)
    rate_all = y[e].mean()
    rows, warnings = [], []
    for r, seed in enumerate(fs.seeds):
        for k in range(N_SPLITS):
            sides = {"training": e & (folds[r] != k), "validation": e & (folds[r] == k)}
            shared = len(set(groups[sides["training"]]) & set(groups[sides["validation"]]))
            if shared:
                raise AssertionError(f"{kind} seed {seed} fold {k}: {shared} groups on both sides")
            for side, m in sides.items():
                if len(set(y[m])) < 2:
                    raise AssertionError(f"{kind} seed {seed} fold {k}: {side} side lacks a class")
                d = pd.DataFrame({"g": groups[m], "y": y[m]})
                shares = pd.Series(era[m]).value_counts(normalize=True)
                row = {"seed": seed, "fold": k, "side": side, "rows": int(m.sum()), "groups": int(d["g"].nunique()),
                       "high": int(y[m].sum()), "low": int((1 - y[m]).sum()), "high_rate": round(float(y[m].mean()), 4),
                       "mixed_groups": int(d.groupby("g")["y"].nunique().gt(1).sum()), "shared_groups": shared}
                for _, _, name in ERAS:
                    row[name] = round(float(shares.get(name, 0.0)), 4)
                    if side == "validation" and abs(shares.get(name, 0.0) - era_all.get(name, 0.0)) > 0.05:
                        warnings.append(f"{kind} seed {seed} fold {k}: era {name} share off by more than 0.05")
                if side == "validation" and abs(y[m].mean() - rate_all) > 0.03:
                    warnings.append(f"{kind} seed {seed} fold {k}: High rate off by more than 0.03")
                rows.append(row)
    return pd.DataFrame(rows), warnings
