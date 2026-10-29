"""Estimator factories for the phase 2 families (plan sections 5.1 and 5.3), composed with the recipe's named
preprocessing, so the model, the feature blocks and the preprocessing vary independently (plan section 4.2).
Preprocessing lives inside the estimator and is fitted on the training part only. Optional packages are imported
inside their factories, so the infrastructure runs and verifies without them (plan section 4.8)."""
import contextlib
import importlib.metadata

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from songpop.models import LOG100

THREADS = 4
CATEGORICAL = ["key", "mode", "time_signature"]  # one-hot under log_standard and spline, never splined
OPTIONAL_PACKAGES = {"catboost": "catboost", "ebm": "interpret-core"}

# Parameters a recipe may set, per family; everything else is fixed by the factory.
ALLOWED = {
    "lightgbm": {"n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "subsample_freq",
                 "colsample_bytree", "reg_alpha", "reg_lambda"},
    "random_forest": {"n_estimators", "min_samples_leaf", "max_features", "max_depth", "max_samples"},
    "extra_trees": {"n_estimators", "min_samples_leaf", "max_features"},
    "xgboost": {"n_estimators", "learning_rate", "max_depth", "min_child_weight", "reg_lambda", "subsample",
                "colsample_bytree"},
    "hist_gb": {"max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf"},
    "logistic": {"C"},
    "catboost": {"iterations", "learning_rate", "depth", "l2_leaf_reg", "random_strength", "boosting_type"},
    "ebm": {"max_rounds", "interactions"},
}


def log_columns(X):
    """log1p(duration_ms / 1000) and log1p(100 x) for the skewed bounded audio features present (as phase 1)."""
    X = X.copy()
    if "duration_ms" in X:
        X["duration_ms"] = np.log1p(X["duration_ms"] / 1000)
    for c in LOG100:
        if c in X:
            X[c] = np.log1p(100 * X[c])
    return X


def make_preprocessing(prep, columns):
    """None for "none"; otherwise the phase 1 log transforms, then one-hot categories (CATEGORICAL) and, for every
    other column, median imputation followed by scaling ("log_standard") or by cubic splines and scaling
    ("spline"). Every statistic is fitted on the rows the estimator is fitted on."""
    name = prep["name"]
    if name == "none":
        return None
    cats = [c for c in columns if c in CATEGORICAL]
    rest = [c for c in columns if c not in CATEGORICAL]
    steps = [("impute", SimpleImputer(strategy="median"))]
    if name == "spline":
        steps.append(("spline", SplineTransformer(n_knots=prep["n_knots"], degree=3)))
    elif name != "log_standard":
        raise ValueError(f"unknown preprocessing {name!r}")
    steps.append(("scale", StandardScaler()))
    parts = [("rest", Pipeline(steps), rest)]
    if cats:
        parts.append(("categories", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                                       sparse_output=False))]), cats))
    return Pipeline([("log", FunctionTransformer(log_columns)),
                     ("columns", ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False))])


def make_estimator(family, p, seed):
    """A new, unfitted estimator of the family with the recipe's parameters and the given estimator seed."""
    extra = set(p) - ALLOWED[family]
    if extra:
        raise ValueError(f"parameters {sorted(extra)} are not allowed for {family}")
    if family == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**p, random_state=seed, n_jobs=THREADS, deterministic=True, force_col_wise=True,
                                  verbose=-1)
    if family == "random_forest":
        return RandomForestClassifier(**p, bootstrap=True, random_state=seed, n_jobs=THREADS)
    if family == "extra_trees":
        return ExtraTreesClassifier(**p, bootstrap=False, random_state=seed, n_jobs=THREADS)
    if family == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**p, tree_method="hist", random_state=seed, n_jobs=THREADS, eval_metric="logloss")
    if family == "hist_gb":
        return HistGradientBoostingClassifier(**p, early_stopping=False, random_state=seed)
    if family == "logistic":
        return LogisticRegression(**p, max_iter=5000, random_state=seed)
    if family == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**p, random_seed=seed, thread_count=THREADS, allow_writing_files=False,
                                  use_best_model=False, task_type="CPU", grow_policy="SymmetricTree", verbose=False)
    if family == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(**p, validation_size=0, outer_bags=1, inner_bags=0, n_jobs=THREADS,
                                             random_state=seed)
    raise ValueError(f"unknown family {family!r}")


def make_model(recipe, seed, columns):
    """The recipe's preprocessing (if any) followed by its family's estimator, for feature frames with these columns."""
    est = make_estimator(recipe["family"], dict(recipe["params"]), seed)
    prep = make_preprocessing(recipe["preprocessing"], list(columns))
    return est if prep is None else Pipeline([("prep", prep), ("model", est)])


def thread_limit(family):
    """A context that holds OpenMP to THREADS for families whose estimators take no thread argument (HistGB);
    the others set their threads in the factory."""
    return threadpool_limits(limits=THREADS, user_api="openmp") if family == "hist_gb" else contextlib.nullcontext()


def weight_argument(model):
    """The fit keyword that carries training weights to the estimator."""
    return "model__sample_weight" if isinstance(model, Pipeline) else "sample_weight"


def family_packages(recipe):
    """Versions of the optional package a family needs, without importing it; None when absent."""
    name = OPTIONAL_PACKAGES.get(recipe["family"])
    if name is None:
        return {}
    try:
        return {name: importlib.metadata.version(name)}
    except importlib.metadata.PackageNotFoundError:
        return {name: None}
