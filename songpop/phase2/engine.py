"""Fitting a recipe and cross-validating it into a prediction bank (plan section 4.3)."""
import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from songpop.evaluate import N_SPLITS, PredictionBank, validate_bank, validate_cv_inputs
from songpop.models import predict_high
from songpop.phase2.encoders import FitContext
from songpop.phase2.features import RAW_COLUMNS, FeaturePipeline
from songpop.phase2.models import FORESTS, make_model, prediction_threads, thread_limit, weight_argument
from songpop.phase2.recipes import validate_recipe

TOL = 1e-12  # equal predictions for identical model inputs


def training_weights(recipe, groups):
    """None, or inverse group size within the fitting rows, normalized to mean one (plan section 6.3)."""
    if recipe["training_weights"] == "none":
        return None
    sizes = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy(dtype="float64")
    w = 1.0 / sizes
    return w * len(w) / w.sum()


@dataclass
class FittedRecipe:
    features: FeaturePipeline
    estimators: list  # one per estimator seed, in the recipe's order
    receipt: dict  # where it was fitted, the fitting rows' hash, the inner splits of its encodings
    family: str

    def model_inputs(self, raw_rows):
        """The matrix entering the final estimator: the fitted feature blocks, then the recipe's fitted
        preprocessing, which is deterministic and identical for every estimator seed."""
        X = self.features.transform(raw_rows)
        est = self.estimators[0]
        return est.named_steps["prep"].transform(X) if isinstance(est, Pipeline) else X.to_numpy(dtype="float64")

    def predict_each(self, raw_rows):
        """The probability of numeric class 1 from each estimator seed, in the recipe's order; forests predict with
        one thread (plan section 16.4)."""
        X = self.features.transform(raw_rows)
        with thread_limit(self.family), prediction_threads(self.family, self.estimators):
            return [predict_high(est, X) for est in self.estimators]

    def predict_high(self, raw_rows):
        """Probability of numeric class 1, averaged with equal weights over the estimator seeds."""
        return np.mean(self.predict_each(raw_rows), axis=0)


def input_classes(Z):
    """Row positions grouped by identical model-input vectors (NaN equal to NaN), classes of two or more rows."""
    out = {}
    for i, row in enumerate(np.asarray(Z, dtype="float64")):
        out.setdefault(tuple("nan" if v != v else v for v in row), []).append(i)
    return [v for v in out.values() if len(v) > 1]


def fit_recipe(recipe, rows, y, groups, ctx):
    """Fit on the fitting population only: raw rows (exactly RAW_COLUMNS, indexed by permanent row id), their
    labels and groups. Nothing else is visible to the fit."""
    validate_recipe(recipe)
    if list(rows.columns) != RAW_COLUMNS:
        raise ValueError(f"fitting rows must have exactly the raw columns {RAW_COLUMNS}")
    y, groups = np.asarray(y), np.asarray(groups)
    if not (len(rows) == len(y) == len(groups)):
        raise ValueError("rows, labels and groups differ in length")
    features = FeaturePipeline(recipe)
    X = features.fit_transform(rows, y, groups, ctx)
    w = training_weights(recipe, groups)
    estimators = []
    for seed in recipe["estimator_seeds"]:
        est = make_model(recipe, seed, X.columns)
        with thread_limit(recipe["family"]):
            est.fit(X, y, **({} if w is None else {weight_argument(est): w}))
        estimators.append(est)
    ids = np.asarray(rows.index, dtype=np.int64)
    receipt = {"split_seed": ctx.split_seed, "fold": ctx.fold,
               "fitting_rows_sha256": hashlib.sha256(np.sort(ids).tobytes()).hexdigest(), "inner": features.receipts_}
    if recipe["family"] in FORESTS:
        receipt["forest_prediction_threads"] = 1
    return FittedRecipe(features, estimators, receipt, recipe["family"])


def raw_rows(frame, index):
    """The raw recipe columns of a frame, indexed by the given ids."""
    out = frame[RAW_COLUMNS].copy()
    out.index = pd.Index(np.asarray(index), name=None)
    return out


def run_recipe_cv(recipe, rows, y, row_ids, groups, fold_id, seeds, test_rows, test_id, n_splits=N_SPLITS):
    """Fit the recipe on each training part and predict the held-out part and the test rows. rows and
    test_rows hold exactly RAW_COLUMNS; rows is indexed by row_ids; fold_id is (R, N). The inputs are validated
    as songpop.evaluate.run_cv validates them, before anything is fitted."""
    validate_recipe(recipe)
    validate_cv_inputs(rows, y, row_ids, groups, fold_id, seeds, test_rows, test_id, n_splits)
    y, groups, fold_id = np.asarray(y), np.asarray(groups), np.asarray(fold_id)
    R, N = fold_id.shape
    oof = np.full((R, N), np.nan)
    count = np.zeros((R, N), dtype=int)
    test_by_fold = np.full((R, n_splits, len(test_rows)), np.nan)
    fold_auc = np.full((R, n_splits), np.nan)
    caught, columns, lineage, receipts = [], None, None, []
    for r in range(R):
        for k in range(n_splits):
            va = np.flatnonzero(fold_id[r] == k)
            tr = np.flatnonzero(fold_id[r] != k)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fitted = fit_recipe(recipe, rows.iloc[tr], y[tr], groups[tr], FitContext(int(seeds[r]), k))
                p = fitted.predict_high(rows.iloc[va])
                each = fitted.predict_each(test_rows)
                classes = input_classes(fitted.model_inputs(test_rows))
            # Fit-time check of every estimator seed: identical model inputs get identical test predictions.
            for s, q in zip(recipe["estimator_seeds"], each):
                if max([float(np.ptp(q[c])) for c in classes], default=0.0) > TOL:
                    raise AssertionError(f"repeat {r} fold {k} seed {s}: identical model inputs predicted differently")
            test_by_fold[r, k] = np.mean(each, axis=0)
            receipts.append(dict(fitted.receipt, test_input_duplicate_classes=len(classes),
                                 estimator_seeds_checked=list(recipe["estimator_seeds"])))
            caught += [f"{type(x.message).__name__}: {x.message}" for x in w]
            if columns is None:
                columns, lineage = fitted.features.columns_, fitted.features.lineage_
            elif fitted.features.columns_ != columns or fitted.features.lineage_ != lineage:
                raise AssertionError("feature columns or lineage differ between folds")
            oof[r, va] = p
            count[r, va] += 1
            fold_auc[r, k] = roc_auc_score(y[va], p)
    if not (count == 1).all():
        raise AssertionError("a row was not held out exactly once per repeat")
    bank = PredictionBank(np.asarray(row_ids), y, groups, np.asarray(seeds), fold_id, oof, np.asarray(test_id),
                          test_by_fold, test_by_fold.mean(axis=(0, 1)), fold_auc)
    bank.metadata["features"], bank.metadata["feature_lineage"] = list(columns), dict(lineage)
    bank.metadata["fit_receipts"] = receipts
    bank.metadata["warnings"] = sorted(set(caught))
    bank.metadata["completed_fits"] = R * n_splits * len(recipe["estimator_seeds"])
    validate_bank(bank, n_splits)
    return bank
