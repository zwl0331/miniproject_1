"""Strict population and partition checks for P2.9 raw-row adapters.

Safe to import in neural workers: no model factory, artifacts module or torch.
Positions are local to the supplied population. Returned positions preserve their
input order; a caller must not sort a training population after deriving seeds.
Legacy split generation and recipe_v1 behavior are deliberately untouched.
"""
import numpy as np


def population(y, groups, row_ids=None):
    """Validate a nonempty binary fitting population; return labels and groups."""
    y, groups = np.asarray(y), np.asarray(groups)
    if y.ndim != 1 or y.dtype.kind not in "biuf" or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be a one-dimensional binary array")
    if len(np.unique(y)) != 2:
        raise ValueError("fitting population must contain both classes")
    if groups.ndim != 1 or len(groups) != len(y):
        raise ValueError("groups must align with fitting labels")
    # Missing values and mixed representations can silently alter group equality.
    values = groups.tolist()
    strings = all(isinstance(g, str) and bool(g) for g in values)
    integers = all(isinstance(g, int) and not isinstance(g, bool) for g in values)
    if not (strings or integers):
        raise ValueError("groups must be nonempty strings or integers, without missing values")
    if row_ids is not None:
        ids = np.asarray(row_ids)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) != len(y)
                or len(np.unique(ids)) != len(ids) or np.any(ids < 0)
                or np.any(ids > np.iinfo(np.int64).max)):
            raise ValueError("row ids must be unique nonnegative int64-compatible integers aligned with labels")
    return y, groups


def positions(values, n, name):
    """Copy a nonempty unique positional index vector without coercing floats."""
    a = np.asarray(values)
    if (a.ndim != 1 or a.dtype.kind not in "iu" or not len(a)
            or len(np.unique(a)) != len(a) or np.any(a < 0) or np.any(a >= n)):
        raise ValueError(f"{name} must contain distinct integer positions in range")
    return a.astype(np.int64, copy=True)


def partition(y, groups, split, *, query_needs_both_classes=True):
    """A complete, group-disjoint (fit, query) partition of exactly this population."""
    y, groups = population(y, groups)
    if not isinstance(split, (tuple, list)) or len(split) != 2:
        raise ValueError("a split must have exactly a fitting side and a query side")
    fit = positions(split[0], len(y), "fitting side")
    query = positions(split[1], len(y), "query side")
    if len(np.intersect1d(fit, query)) or len(fit) + len(query) != len(y):
        raise ValueError("split sides must partition every row exactly once")
    if set(groups[fit].tolist()) & set(groups[query].tolist()):
        raise ValueError("split sides must not share a primary group")
    if len(np.unique(y[fit])) != 2:
        raise ValueError("fitting side must contain both classes")
    if query_needs_both_classes and len(np.unique(y[query])) != 2:
        raise ValueError("query side must contain both classes")
    return fit, query


def crossfit(y, groups, splits, *, n_folds=5, query_needs_both_classes=False):
    """Complete complementary folds, each row held out once, whole groups only.

Encoder/calibration query folds may contain one class; every training fold must
contain both. Stopping validation uses partition() with its stricter default.
"""
    y, groups = population(y, groups)
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least two")
    if not isinstance(splits, (tuple, list)) or len(splits) != n_folds:
        raise ValueError(f"expected exactly {n_folds} cross-fitting folds")
    checked = [partition(y, groups, split, query_needs_both_classes=query_needs_both_classes)
               for split in splits]
    held = np.concatenate([query for _, query in checked])
    if not np.array_equal(np.sort(held), np.arange(len(y))):
        raise ValueError("cross-fitting must hold out every row exactly once")
    return checked
