"""Identifier and order feature blocks of the P2.8 probe (plan section 16.2). Row-local blocks are plain functions;
fitted blocks follow the phase 2 encoder convention: fit_transform encodes the fitting rows (leaving out a row's own
group, or cross-fitted over inner group folds where labels are used), transform encodes any other rows."""
import hashlib
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import TargetEncoder

from songpop.data import AUDIO
from songpop.phase2.encoders import check_inner_splits, inner_group_splits, splits_sha256

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CODE = {c: i for i, c in enumerate(ALPHABET)}
TOKEN_LENGTH = 22
URI = re.compile(r"spotify:track:([0-9A-Za-z]{22})", re.ASCII)
CHUNKS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 22)]
COUNT_FRAGMENTS = ("p1", "p2", "p3", "s1", "s2", "s3")
TE_FRAGMENTS = ("p1", "p2", "s1", "s2")
TE_SMOOTHING = 30
HASH_FEATURES = 16384


def tokens(uris):
    """The 22-character token of each uri, parsed strictly; raises on any other form."""
    out = []
    for u in uris:
        m = URI.fullmatch(u) if isinstance(u, str) else None
        if m is None:
            raise ValueError(f"not a spotify track uri: {u!r}")
        out.append(m.group(1))
    return np.array(out, dtype=object)


def pseudotokens(toks, salt):
    """A control token per token: the first 16 bytes of sha256(salt:token) written in the alphabet with 22 digits,
    so equal tokens stay equal, the format is kept and every fragment's link to the real token is destroyed."""
    out = []
    for t in toks:
        n = int.from_bytes(hashlib.sha256(f"{salt}:{t}".encode("ascii")).digest()[:16], "big")
        digits = []
        for _ in range(TOKEN_LENGTH):
            n, d = divmod(n, len(ALPHABET))
            digits.append(ALPHABET[d])
        out.append("".join(reversed(digits)))
    return np.array(out, dtype=object)


def _longest_run(t):
    best = run = 1
    for a, b in zip(t, t[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def _entropy(t):
    counts = pd.Series(list(t)).value_counts().to_numpy(dtype="float64")
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def shape_frame(toks, index):
    """id_shape_v1, row-local: the 22 character codes, six numeric chunks read in the alphabet's radix, digit and
    uppercase counts, distinct characters, the longest identical-character run and the character entropy."""
    rows = []
    for t in toks:
        row = [float(CODE[c]) for c in t]
        for a, b in CHUNKS:
            row.append(float(sum(CODE[c] * len(ALPHABET) ** (b - a - 1 - i) for i, c in enumerate(t[a:b]))))
        row += [float(sum(c.isdigit() for c in t)), float(sum(c.isupper() for c in t)), float(len(set(t))),
                float(_longest_run(t)), _entropy(t)]
        rows.append(row)
    cols = ([f"id_char{i:02d}" for i in range(TOKEN_LENGTH)] + [f"id_chunk{a}_{b}" for a, b in CHUNKS]
            + ["id_digits", "id_upper", "id_distinct", "id_longest_run", "id_entropy"])
    return pd.DataFrame(rows, index=index, columns=cols, dtype="float64")


def fragments(toks, names):
    """Prefixes pN and suffixes sN of each token, as a frame of strings."""
    out = {}
    for n in names:
        k = int(n[1:])
        out[n] = [t[:k] if n[0] == "p" else t[-k:] for t in toks]
    return pd.DataFrame(out)


class GroupCount:
    """log1p of the number of distinct fitting groups holding each key of each column, over distinct (key, group)
    pairs. A fitting row's own group is left out (log1p(count - 1)); any other row gets log1p(count), zero for an
    unseen key. The phase 2 date_count rule, generalized to several key columns."""

    def __init__(self, names):
        self.names = dict(names)  # key column -> output column

    def fit(self, keys, groups):
        self.count_ = {c: pd.DataFrame({"k": keys[c].to_numpy(), "g": np.asarray(groups)}).drop_duplicates()
                       .groupby("k").size().to_dict() for c in keys.columns}
        return self

    def fit_transform(self, keys, groups, index):
        self.fit(keys, groups)
        out = {}
        for c in keys.columns:
            n = np.array([self.count_[c][k] for k in keys[c]], dtype="float64")
            if n.min() < 1:
                raise AssertionError("a fitting key has no group")
            out[self.names[c]] = np.log1p(n - 1)
        return pd.DataFrame(out, index=index)

    def transform(self, keys, index):
        return pd.DataFrame({self.names[c]: np.log1p(np.array([self.count_[c].get(k, 0) for k in keys[c]],
                                                              dtype="float64"))
                             for c in keys.columns}, index=index)


def _inner_splits(y, groups, seed, splits):
    """Explicit inner (train, test) pairs, checked; by default the stratified group folds of the seed."""
    if splits is not None:
        return check_inner_splits(y, groups, splits)
    return inner_group_splits(np.asarray(y), np.asarray(groups), seed)


class CrossFittedTE:
    """The smoothed mean label of each key of each column, (sum of labels + m x prior) / (rows + m), by
    scikit-learn's TargetEncoder: fitting rows cross-fitted over explicit inner stratified group folds of the
    fitting population (a row's whole group outside the rows encoding it), other rows by the encoder fitted on
    all fitting rows, an unseen key the fitting prior. The phase 2 date_te rule, for several key columns."""

    def __init__(self, names, block, m=TE_SMOOTHING):
        self.names, self.block, self.m = dict(names), block, m

    def fit_transform(self, keys, y, groups, seed, index, splits=None):
        y = np.asarray(y)
        self.splits_ = _inner_splits(y, groups, seed, splits)
        self.receipt = {"block": self.block, "inner_seed": None if splits is not None else seed,
                        "inner_splits_sha256": splits_sha256(index, self.splits_)}
        self.columns_ = list(keys.columns)
        self.encoder_ = TargetEncoder(categories="auto", target_type="binary", smooth=float(self.m), cv=self.splits_)
        values = self.encoder_.fit_transform(keys.to_numpy(dtype=object), y)
        if list(self.encoder_.classes_) != [0, 1]:
            raise ValueError(f"target encoder classes {list(self.encoder_.classes_)}, expected [0, 1]")
        return self._frame(values, index)

    def transform(self, keys, index):
        return self._frame(self.encoder_.transform(keys[self.columns_].to_numpy(dtype=object)), index)

    def _frame(self, values, index):
        return pd.DataFrame(values, index=index, columns=[self.names[c] for c in self.columns_])


class AudioDeviation:
    """The smoothed deviation of a key's mean label from the prior, (S - n mu) / (n + m), which equals
    (S + m mu) / (n + m) - mu and is zero for an unsupported key (n = 0). For a fitting row, S, n and mu come from
    the inner training part of its inner group fold; for any other row, from the whole fitting population."""
    column = "audio_te_dev"

    def __init__(self, m=TE_SMOOTHING):
        self.m = m

    @staticmethod
    def _stats(keys, y):
        g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
        return {k: (float(s), float(n)) for k, (s, n) in g.iterrows()}

    def _deviation(self, keys, stats, mu):
        return np.array([(stats[k][0] - stats[k][1] * mu) / (stats[k][1] + self.m) if k in stats else 0.0
                         for k in keys], dtype="float64")

    def fit_transform(self, keys, y, groups, seed, index, splits=None):
        keys, y = np.asarray(keys, dtype=object), np.asarray(y, dtype="float64")
        self.splits_ = _inner_splits(y, groups, seed, splits)
        self.receipt = {"block": "audio_recurrence", "inner_seed": None if splits is not None else seed,
                        "inner_splits_sha256": splits_sha256(index, self.splits_)}
        out = np.full(len(y), np.nan)
        for tr, te in self.splits_:
            out[te] = self._deviation(keys[te], self._stats(keys[tr], y[tr]), float(y[tr].mean()))
        self.stats_, self.mu_ = self._stats(keys, y), float(y.mean())
        return pd.DataFrame({self.column: out}, index=index)

    def transform(self, keys, index):
        return pd.DataFrame({self.column: self._deviation(np.asarray(keys, dtype=object), self.stats_, self.mu_)},
                            index=index)


def audio_keys(frame):
    """The exact 13-value audio tuple of each row as one string key (repr of each float64 value)."""
    a = frame[AUDIO].to_numpy(dtype="float64")
    return pd.DataFrame({"audio": ["|".join(repr(float(v)) for v in row) for row in a]})


def hashed_identifier_features(toks):
    """The input of the identifier-only linear model L: position-tagged characters, unpositioned adjacent
    character bigrams and the tagged two-character prefix and suffix, hashed to HASH_FEATURES columns without
    alternating signs, each row L2-normalized. Row-local: nothing is fitted."""
    dicts = []
    for t in toks:
        d = {f"c{i}={c}": 1.0 for i, c in enumerate(t)}
        for a, b in zip(t, t[1:]):
            d[f"b={a}{b}"] = d.get(f"b={a}{b}", 0.0) + 1.0
        d[f"p2={t[:2]}"] = 1.0
        d[f"s2={t[-2:]}"] = 1.0
        dicts.append(d)
    X = FeatureHasher(n_features=HASH_FEATURES, input_type="dict", alternate_sign=False).transform(dicts)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    return sparse.csr_matrix(sparse.diags(1.0 / norms) @ X)


class FittedBlocks:
    """The probe's feature blocks fitted on one fitting population (plan section 16.2). blocks: names among
    id_shape_v1, id_frequency_v1, id_te30_v1, row_position, audio_recurrence."""

    def __init__(self, blocks):
        self.blocks = list(blocks)

    def fit_transform(self, raw, toks, row_ids, y, groups, ctx):
        """raw: the fitting rows' raw frame; toks: their tokens; row_ids: their permanent ids (the frame index);
        ctx: the phase 2 FitContext of the outer fold, from which the inner seeds derive."""
        index = pd.Index(np.asarray(row_ids))
        parts, self.receipts_ = [], []
        for b in self.blocks:
            if b == "id_shape_v1":
                parts.append(shape_frame(toks, index))
            elif b == "id_frequency_v1":
                self.freq_ = GroupCount({f: f"id_freq_{f}" for f in COUNT_FRAGMENTS})
                parts.append(self.freq_.fit_transform(fragments(toks, COUNT_FRAGMENTS), groups, index))
            elif b == "id_te30_v1":
                self.te_ = CrossFittedTE({f: f"id_te30_{f}" for f in TE_FRAGMENTS}, "id_te30_v1")
                parts.append(self.te_.fit_transform(fragments(toks, TE_FRAGMENTS), y, groups,
                                                    ctx.inner_seed("id_te30"), index))
                self.receipts_.append(self.te_.receipt)
            elif b == "row_position":
                parts.append(pd.DataFrame({"row_position": np.asarray(row_ids, dtype="float64")}, index=index))
            elif b == "audio_recurrence":
                keys = audio_keys(raw)
                self.rec_count_ = GroupCount({"audio": "audio_support"})
                self.rec_dev_ = AudioDeviation()
                parts.append(self.rec_count_.fit_transform(keys, groups, index))
                parts.append(self.rec_dev_.fit_transform(keys["audio"], y, groups, ctx.inner_seed("audio_recurrence"),
                                                         index))
                self.receipts_.append(self.rec_dev_.receipt)
            else:
                raise ValueError(f"unknown probe block {b!r}")
        self.columns_ = [c for p in parts for c in p.columns]
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=index)

    def transform(self, raw, toks, row_ids, positions_known):
        """Other rows: held-out training rows (positions_known True) or test rows (row position missing)."""
        index = pd.Index(np.asarray(row_ids))
        parts = []
        for b in self.blocks:
            if b == "id_shape_v1":
                parts.append(shape_frame(toks, index))
            elif b == "id_frequency_v1":
                parts.append(self.freq_.transform(fragments(toks, COUNT_FRAGMENTS), index))
            elif b == "id_te30_v1":
                parts.append(self.te_.transform(fragments(toks, TE_FRAGMENTS), index))
            elif b == "row_position":
                pos = np.asarray(row_ids, dtype="float64") if positions_known else np.full(len(index), np.nan)
                parts.append(pd.DataFrame({"row_position": pos}, index=index))
            elif b == "audio_recurrence":
                keys = audio_keys(raw)
                parts.append(self.rec_count_.transform(keys, index))
                parts.append(self.rec_dev_.transform(keys["audio"], index))
        out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=index)
        if list(out.columns) != self.columns_:
            raise AssertionError("transformed block columns differ from the fitted ones")
        return out

