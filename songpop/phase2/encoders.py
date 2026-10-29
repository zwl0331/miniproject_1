"""Fold-fitted date and era encodings (plan sections 6.3 and 6.4). Each is fitted on its fitting population only:
fit_transform encodes the fitting rows (cross-fitted where labels are used), transform encodes any other rows."""
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import TargetEncoder

from songpop.data import DATE
from songpop.splits import era_of

INNER_FOLDS = 5
ERA_PSEUDO_COUNT = 20


@dataclass(frozen=True)
class FitContext:
    """Where a fit happens: the outer split seed and fold ("all" for a fit on every eligible row). Inner seeds
    derive from it deterministically, so a fit is reproducible without any global random state."""
    split_seed: object
    fold: object

    def inner_seed(self, purpose):
        token = f"{purpose}-v1:{self.split_seed}:{self.fold}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") % (2 ** 31)


def check_inner_splits(y, groups, splits):
    """Explicit (train, test) positional index pairs over n fitting rows: integer indices in range, no repeats on
    a side, sides disjoint in rows and groups, every row held out exactly once, both classes in every fitting
    part. The held-out parts need not hold both classes: they are only encoded. Raises ValueError."""
    y, groups, n = np.asarray(y), np.asarray(groups), len(y)
    held = []
    for tr, te in splits:
        tr, te = np.asarray(tr), np.asarray(te)
        for side in (tr, te):
            if not np.issubdtype(side.dtype, np.integer) or side.ndim != 1 or len(np.unique(side)) != len(side) \
                    or (len(side) and (side.min() < 0 or side.max() >= n)):
                raise ValueError("inner split indices must be distinct integers in range")
        if set(tr.tolist()) & set(te.tolist()) or set(groups[tr].tolist()) & set(groups[te].tolist()):
            raise ValueError("an inner split shares a row or a group between its sides")
        if len(np.unique(y[tr])) < 2:
            raise ValueError("an inner fitting part lacks a class")
        held += te.tolist()
    if sorted(held) != list(range(n)):
        raise ValueError("inner splits do not hold out every fitting row exactly once")
    return splits


def inner_group_splits(y, groups, seed, n_splits=INNER_FOLDS):
    """Explicit group-disjoint (train, test) positional index pairs over the fitting rows, stratified by label."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return check_inner_splits(y, groups, [(tr, te) for tr, te in sgkf.split(np.zeros((len(y), 1)), y, groups)])


def splits_sha256(row_ids, splits):
    """Hash of inner splits as permanent row ids, sorted within each side, in split order."""
    ids = np.asarray(row_ids)
    canon = [[sorted(int(i) for i in ids[tr]), sorted(int(i) for i in ids[te])] for tr, te in splits]
    return hashlib.sha256(json.dumps(canon, separators=(",", ":")).encode()).hexdigest()


class DateCount:
    """log1p of the number of distinct fitting groups sharing the raw date string, counted over distinct (date,
    group) pairs. A fitting row's own group is left out (log1p(count - 1)); any other row gets log1p(count), zero
    for an unseen date."""
    columns = ["date_count"]
    sources = {"date_count": [DATE]}
    receipt = None

    def fit(self, rows, groups):
        dates = rows[DATE].to_numpy()
        self.count_ = pd.DataFrame({"date": dates, "group": groups}).drop_duplicates().groupby("date").size().to_dict()
        return self

    def fit_transform(self, rows, y, groups, ctx):
        self.fit(rows, groups)
        counts = np.array([self.count_[d] for d in rows[DATE]], dtype="float64")
        if counts.min() < 1:
            raise AssertionError("a fitting row's date has no group")
        return pd.DataFrame({"date_count": np.log1p(counts - 1)}, index=rows.index)

    def transform(self, rows):
        counts = np.array([self.count_.get(d, 0) for d in rows[DATE]], dtype="float64")
        return pd.DataFrame({"date_count": np.log1p(counts)}, index=rows.index)


class DateTargetEncoder:
    """The mean label of the exact raw date string, smoothed toward the fitting prior with m pseudo-rows:
    (sum of labels + m x prior) / (rows + m). Fitting rows are encoded by cross-fitting over explicit inner
    group folds, so a row's whole group is outside the rows that encode it; other rows use the encoder fitted
    on all fitting rows; an unseen date gets the fitting prior."""

    def __init__(self, m):
        self.m, self.receipt = m, None
        self.columns = [f"date_te{m}"]
        self.sources = {f"date_te{m}": [DATE]}

    def fit_transform(self, rows, y, groups, ctx, splits=None):
        """splits: explicit inner (train, test) pairs, checked like generated ones; by default the stratified group
        folds of the context's inner seed. Stratification uses the fitting labels to place groups, as the outer
        folds do. The receipt records the seed and the splits as permanent row ids."""
        dates = rows[DATE].to_numpy(dtype=object).reshape(-1, 1)
        seed = None if splits is not None else ctx.inner_seed("date_te")
        self.splits_ = check_inner_splits(y, groups, splits) if splits is not None else \
            inner_group_splits(np.asarray(y), np.asarray(groups), seed)
        self.receipt = {"block": self.columns[0], "inner_seed": seed,
                        "inner_splits_sha256": splits_sha256(rows.index, self.splits_)}
        self.encoder_ = TargetEncoder(categories="auto", target_type="binary", smooth=float(self.m),
                                      cv=self.splits_)
        values = self.encoder_.fit_transform(dates, np.asarray(y))
        if list(self.encoder_.classes_) != [0, 1]:
            raise ValueError(f"target encoder classes {list(self.encoder_.classes_)}, expected [0, 1]")
        return pd.DataFrame({self.columns[0]: values[:, 0]}, index=rows.index)

    def transform(self, rows):
        values = self.encoder_.transform(rows[DATE].to_numpy(dtype=object).reshape(-1, 1))
        return pd.DataFrame({self.columns[0]: values[:, 0]}, index=rows.index)


class EraRelative:
    """Loudness and acousticness minus their era mean (the five eras of songpop.splits), each era mean shrunk
    toward the fitting mean with ERA_PSEUDO_COUNT pseudo-rows; an era absent from the fitting rows uses the
    fitting mean. Unsupervised, so fit_transform needs no cross-fitting."""
    features = ["loudness", "acousticness"]
    columns = ["loudness_era_rel", "acousticness_era_rel"]
    receipt = None
    sources = {"loudness_era_rel": ["loudness", DATE], "acousticness_era_rel": ["acousticness", DATE]}

    @staticmethod
    def _eras(rows):
        return era_of(rows[DATE].str[:4].astype(int).to_numpy())

    def fit(self, rows):
        eras = self._eras(rows)
        self.global_ = {f: float(rows[f].mean()) for f in self.features}
        self.era_mean_ = {}
        for f in self.features:
            g = pd.Series(rows[f].to_numpy(dtype="float64")).groupby(eras).agg(["sum", "count"])
            self.era_mean_[f] = {str(e): float((s + ERA_PSEUDO_COUNT * self.global_[f]) / (n + ERA_PSEUDO_COUNT))
                                 for e, (s, n) in g.iterrows()}
        return self

    def fit_transform(self, rows, y, groups, ctx):
        return self.fit(rows).transform(rows)

    def transform(self, rows):
        eras = self._eras(rows)
        out = {}
        for f, col in zip(self.features, self.columns):
            means = np.array([self.era_mean_[f].get(str(e), self.global_[f]) for e in eras])
            out[col] = rows[f].to_numpy(dtype="float64") - means
        return pd.DataFrame(out, index=rows.index)
