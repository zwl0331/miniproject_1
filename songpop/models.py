"""Baseline recipes (doc/plans/phase1-data-baseline.md section 4.5)."""
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

MODELS = ("lightgbm", "random_forest", "logistic")
LOG100 = ["instrumentalness", "speechiness", "liveness", "acousticness"]
ESTIMATOR_SEED = 0


def log_transform(X):
    """log1p(duration_ms / 1000) and log1p(100 x) for the skewed bounded features; other columns unchanged."""
    X = X.copy()
    if "duration_ms" in X:
        X["duration_ms"] = np.log1p(X["duration_ms"] / 1000)
    for c in LOG100:
        if c in X:
            X[c] = np.log1p(100 * X[c])
    return X


def make_estimator(model):
    if model == "lightgbm":
        return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=20,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=ESTIMATOR_SEED, n_jobs=4, deterministic=True, force_col_wise=True,
                                  verbose=-1)
    if model == "random_forest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=ESTIMATOR_SEED, n_jobs=4)
    if model == "logistic":
        # Imputation follows the transform, so medians are those of the transformed training columns.
        return Pipeline([("log", FunctionTransformer(log_transform)),
                         ("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("model", LogisticRegression(C=1.0, max_iter=2000))])
    raise ValueError(f"unknown model {model!r}")


def describe(obj):
    """A JSON-serializable description of a parameter value that does not depend on the process:
    estimators as class and parameters, callables by qualified name, non-finite floats as strings."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else repr(float(obj))
    if isinstance(obj, (list, tuple)):
        return [describe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): describe(v) for k, v in sorted(obj.items())}
    if hasattr(obj, "get_params"):
        return {"class": f"{type(obj).__module__}.{type(obj).__qualname__}",
                "params": {k: describe(v) for k, v in sorted(obj.get_params(deep=False).items())}}
    if callable(obj):
        return f"{getattr(obj, '__module__', '?')}:{getattr(obj, '__qualname__', type(obj).__qualname__)}"
    return {"class": f"{type(obj).__module__}.{type(obj).__qualname__}"}


def predict_high(estimator, X):
    """Probability of High (class 1), located in classes_. The whole probability matrix is checked: shape
    (len(X), 2), finite, in [0, 1], rows summing to one within 1e-6 (XGBoost returns float32 probabilities)."""
    classes = list(getattr(estimator, "classes_", []))
    if sorted(classes) != [0, 1]:
        raise ValueError(f"estimator classes are {classes}, expected 0 and 1")
    proba = np.asarray(estimator.predict_proba(X), dtype="float64")
    if proba.ndim != 2 or proba.shape != (len(X), 2):
        raise ValueError(f"probability matrix shape {proba.shape} for {len(X)} rows, expected ({len(X)}, 2)")
    if not np.all(np.isfinite(proba)) or proba.min() < 0 or proba.max() > 1:
        raise ValueError("probabilities not finite or outside [0, 1]")
    if not np.allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("probability rows do not sum to one")
    return proba[:, classes.index(1)].copy()
