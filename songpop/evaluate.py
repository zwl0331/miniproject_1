"""Cross-validation into a prediction bank (doc/plans/phase1-data-baseline.md sections 4.6 and 4.8)."""
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from songpop.models import predict_high

BANK_SCHEMA = "bank_v1"
N_SPLITS = 5


@dataclass
class PredictionBank:
    train_row_id: np.ndarray  # (N,) eligible rows
    y: np.ndarray  # (N,)
    group_id: np.ndarray  # (N,)
    split_seeds: np.ndarray  # (R,)
    fold_id: np.ndarray  # (R, N)
    oof: np.ndarray  # (R, N)
    test_id: np.ndarray  # (T,)
    test_by_fold: np.ndarray  # (R, K, T)
    test_mean: np.ndarray  # (T,) deployment: equal average of the R x K fold models
    fold_auc: np.ndarray  # (R, K)
    metadata: dict = field(default_factory=dict)

    ARRAYS = ("train_row_id", "y", "group_id", "split_seeds", "fold_id", "oof", "test_id", "test_by_fold",
              "test_mean", "fold_auc")


def _check_folds(y, groups, fold_id, n_splits):
    """One-dimensional groups with no missing id; integral fold ids 0..n_splits-1, every fold present, every
    group in one fold, both classes on each side. Missing ids are rejected first because groupby drops them."""
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("group ids must be one-dimensional")
    if pd.isna(groups).any():
        raise ValueError("missing group ids")
    if not np.issubdtype(np.asarray(fold_id).dtype, np.integer):
        raise ValueError("fold ids are not integers")
    for r in range(fold_id.shape[0]):
        f = fold_id[r]
        if set(np.unique(f)) != set(range(n_splits)):
            raise ValueError(f"repeat {r}: fold ids {sorted(set(np.unique(f)))}, expected 0 to {n_splits - 1}")
        if pd.DataFrame({"g": groups, "f": f}).groupby("g")["f"].nunique().max() > 1:
            raise ValueError(f"repeat {r}: a group is assigned to more than one fold")
        for k in range(n_splits):
            for side, m in [("validation", f == k), ("training", f != k)]:
                if len(np.unique(y[m])) < 2:
                    raise ValueError(f"repeat {r} fold {k}: the {side} side lacks a class")


def validate_cv_inputs(X, y, row_ids, groups, fold_id, seeds, X_test, test_id, n_splits=N_SPLITS):
    """The contract of run_cv, checked before anything is fitted. Raises ValueError."""
    y, row_ids, groups, fold_id = map(np.asarray, (y, row_ids, groups, fold_id))
    n = len(X)
    if fold_id.ndim != 2 or not (len(y) == len(row_ids) == len(groups) == fold_id.shape[1] == n):
        raise ValueError("X, y, row ids, groups and fold ids differ in length")
    if len(seeds) < 1 or fold_id.shape[0] != len(seeds):
        raise ValueError(f"{fold_id.shape[0]} fold repeats for {len(seeds)} split seeds")
    if len(np.unique(row_ids)) != n:
        raise ValueError("row ids are not unique")
    if not np.array_equal(np.asarray(X.index), row_ids):
        raise ValueError("the feature frame's index is not the row ids")
    if len(test_id) != len(X_test) or len(np.unique(test_id)) != len(test_id):
        raise ValueError("test ids are not unique or differ in length from the test features")
    if list(X.columns) != list(X_test.columns):
        raise ValueError("training and test feature columns differ")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("labels are not exactly 0 and 1")
    _check_folds(y, groups, fold_id, n_splits)


def validate_bank(bank, n_splits=N_SPLITS, require_metadata=False, tol=1e-12):
    """Shapes, identities, folds, probabilities, the deployment mean and the stored fold AUCs; with
    require_metadata also the schema and the consistency of the metadata with the arrays. Raises ValueError."""
    N, R, T = len(bank.train_row_id), len(bank.split_seeds), len(bank.test_id)
    shapes = {"y": (N,), "group_id": (N,), "fold_id": (R, N), "oof": (R, N), "test_by_fold": (R, n_splits, T),
              "test_mean": (T,), "fold_auc": (R, n_splits)}
    for f, s in shapes.items():
        if getattr(bank, f).shape != s:
            raise ValueError(f"bank field {f} has shape {getattr(bank, f).shape}, expected {s}")
    if R < 1 or len(np.unique(bank.train_row_id)) != N or len(np.unique(bank.test_id)) != T:
        raise ValueError("no repeat, or row ids or test ids not unique")
    if set(np.unique(bank.y)) != {0, 1}:
        raise ValueError("bank labels are not exactly 0 and 1")
    _check_folds(bank.y, bank.group_id, bank.fold_id, n_splits)
    for f in ("oof", "test_by_fold", "test_mean"):
        a = getattr(bank, f)
        if not np.all(np.isfinite(a)) or a.min() < 0 or a.max() > 1:
            raise ValueError(f"bank field {f} not finite or outside [0, 1]")
    if not np.allclose(bank.test_mean, bank.test_by_fold.mean(axis=(0, 1)), rtol=0, atol=tol):
        raise ValueError("test_mean is not the mean of test_by_fold")
    if not np.all(np.isfinite(bank.fold_auc)) or bank.fold_auc.min() < 0 or bank.fold_auc.max() > 1:
        raise ValueError("stored fold AUCs not finite or outside [0, 1]")
    for r in range(R):
        for k in range(n_splits):
            m = bank.fold_id[r] == k
            if abs(roc_auc_score(bank.y[m], bank.oof[r, m]) - bank.fold_auc[r, k]) > tol:
                raise ValueError(f"stored fold AUC of repeat {r} fold {k} does not match its predictions")
    if require_metadata:
        md = bank.metadata
        if md.get("schema") != BANK_SCHEMA:
            raise ValueError(f"bank schema {md.get('schema')!r}, expected {BANK_SCHEMA!r}")
        if md.get("split_seeds") != [int(s) for s in bank.split_seeds]:
            raise ValueError("metadata split seeds differ from the bank's")
        for key in ("run_id", "recipe_id", "feature_set", "features", "deployment", "parameters"):
            if key not in md:
                raise ValueError(f"bank metadata lacks {key}")


def run_cv(X, y, row_ids, groups, fold_id, seeds, X_test, test_id, make, n_splits=N_SPLITS):
    """Fit make() on each training part, predict the held-out part and the test rows. X's index must be the
    row ids; fold_id is (R, N) over the rows of X. The inputs are validated before anything is fitted."""
    validate_cv_inputs(X, y, row_ids, groups, fold_id, seeds, X_test, test_id, n_splits)
    y, fold_id = np.asarray(y), np.asarray(fold_id)
    R, N = fold_id.shape
    oof = np.full((R, N), np.nan)
    count = np.zeros((R, N), dtype=int)
    test_by_fold = np.full((R, n_splits, len(X_test)), np.nan)
    fold_auc = np.full((R, n_splits), np.nan)
    caught = []
    for r in range(R):
        for k in range(n_splits):
            va = np.flatnonzero(fold_id[r] == k)
            tr = np.flatnonzero(fold_id[r] != k)
            est = make()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                est.fit(X.iloc[tr], y[tr])
                p = predict_high(est, X.iloc[va])
                test_by_fold[r, k] = predict_high(est, X_test)
            caught += [f"{type(x.message).__name__}: {x.message}" for x in w]
            oof[r, va] = p
            count[r, va] += 1
            fold_auc[r, k] = roc_auc_score(y[va], p)
    if not (count == 1).all():
        raise AssertionError("a row was not held out exactly once per repeat")
    bank = PredictionBank(np.asarray(row_ids), y, np.asarray(groups), np.asarray(seeds), fold_id, oof,
                          np.asarray(test_id), test_by_fold, test_by_fold.mean(axis=(0, 1)), fold_auc)
    bank.metadata["warnings"] = sorted(set(caught))
    bank.metadata["completed_fits"] = R * n_splits
    validate_bank(bank, n_splits)
    return bank
