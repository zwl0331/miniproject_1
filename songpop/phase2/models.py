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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from songpop.models import LOG100

THREADS = 4
CATEGORICAL = ["key", "mode", "time_signature"]  # one-hot under log_standard, spline and knn_blocks, never splined
OPTIONAL_PACKAGES = {"catboost": "catboost", "ebm": "interpret-core"}

# Parameters a recipe may set, per family; everything else is fixed by the factory.
ALLOWED = {
    "lightgbm": {"n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "subsample_freq",
                 "colsample_bytree", "reg_alpha", "reg_lambda"},
    "random_forest": {"n_estimators", "min_samples_leaf", "max_features", "max_depth", "max_samples"},
    "extra_trees": {"n_estimators", "min_samples_leaf", "max_features", "max_depth"},
    "xgboost": {"n_estimators", "learning_rate", "max_depth", "min_child_weight", "reg_alpha", "reg_lambda", "gamma",
                "subsample", "colsample_bytree"},
    "hist_gb": {"max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"},
    "logistic": {"C"},
    "catboost": {"iterations", "learning_rate", "depth", "l2_leaf_reg", "random_strength", "boosting_type"},
    "ebm": {"max_rounds", "interactions"},
    "knn": {"n_neighbors", "weights", "metric"},
}
KNN_METRICS = {"euclidean": "l2", "manhattan": "l1"}  # each metric with the block normalization it requires


def _int(v, lo=1):
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def _fraction(v):
    """A float in (0, 1]: an integer 1 would mean one row or one feature to scikit-learn, not all of them."""
    return isinstance(v, float) and np.isfinite(v) and 0 < v <= 1


DOMAINS = {  # parameter: predicate on its value; a recipe's parameters are checked before anything is fitted
    "n_estimators": _int, "max_iter": _int, "iterations": _int, "max_rounds": _int, "n_neighbors": _int,
    "num_leaves": lambda v: _int(v, 2), "max_leaf_nodes": lambda v: _int(v, 2),
    "min_child_samples": _int, "min_samples_leaf": _int, "subsample_freq": lambda v: _int(v, 0),
    "max_depth": lambda v: v is None or _int(v), "depth": _int,  # XGBoost: an integer only (FAMILY_DOMAINS)
    "learning_rate": lambda v: _num(v) and v > 0, "C": lambda v: _num(v) and v > 0,
    "reg_alpha": lambda v: _num(v) and v >= 0, "reg_lambda": lambda v: _num(v) and v >= 0,
    "gamma": lambda v: _num(v) and v >= 0, "l2_regularization": lambda v: _num(v) and v >= 0,
    "l2_leaf_reg": lambda v: _num(v) and v >= 0, "random_strength": lambda v: _num(v) and v >= 0,
    "min_child_weight": lambda v: _num(v) and v >= 0,
    "subsample": _fraction, "colsample_bytree": _fraction, "max_samples": _fraction,
    "max_features": lambda v: v == "sqrt" or _fraction(v),
    "weights": lambda v: v in ("uniform", "distance"), "metric": lambda v: v in KNN_METRICS,
    "boosting_type": lambda v: v in ("Plain", "Ordered"), "interactions": lambda v: True,
}


FAMILY_DOMAINS = {("xgboost", "max_depth"): _int}
# Parameters whose values are counts: compared as integers; every other numeric parameter is compared as a float.
INTEGER_PARAMS = {"n_estimators", "max_iter", "iterations", "max_rounds", "n_neighbors", "num_leaves", "max_leaf_nodes",
                  "min_child_samples", "min_samples_leaf", "subsample_freq", "max_depth", "depth", "n_knots"}


def validate_model_params(family, params):
    """Names allowed for the family and values in their domains; raises ValueError."""
    extra = set(params) - ALLOWED[family]
    if extra:
        raise ValueError(f"parameters {sorted(extra)} are not allowed for {family}")
    bad = sorted(k for k, v in params.items() if not FAMILY_DOMAINS.get((family, k), DOMAINS[k])(v))
    if bad:
        raise ValueError(f"{family} parameters out of domain: { {k: params[k] for k in bad} }")


def log_columns(X):
    """log1p(duration_ms / 1000) and log1p(100 x) for the skewed bounded audio features present (as phase 1)."""
    X = X.copy()
    if "duration_ms" in X:
        X["duration_ms"] = np.log1p(X["duration_ms"] / 1000)
    for c in LOG100:
        if c in X:
            X[c] = np.log1p(100 * X[c])
    return X


def scale_by(X, factor):
    """Multiply a block by its fixed weight (the knn_blocks representation)."""
    return np.asarray(X, dtype="float64") * factor


def knn_blocks(prep, columns):
    """The block-weighted kNN representation of plan section 6.2 (on F1), fitted on the rows the estimator is
    fitted on: the phase 1 log transforms; the continuous audio columns median-imputed and standardized; the year
    median-imputed, standardized and multiplied by alpha; key, mode and time signature one-hot (categories from the
    fitting rows, unseen ones all zero), not standardized. Block weights by normalization, with d the number of
    continuous columns (ten) and beta the category weight: "l2" (Euclidean) 1/sqrt(d) and beta/sqrt(3); "l1"
    (Manhattan) 1/d and beta/6, so a row differing in one of the three categorical fields adds beta/3 to the
    distance (plan section 15.4)."""
    l2 = prep["normalization"] == "l2"
    cont = [c for c in columns if c not in CATEGORICAL and c != "year"]
    year = [c for c in columns if c == "year"]
    cats = [c for c in columns if c in CATEGORICAL]
    if sorted(cont + year + cats) != sorted(columns) or len(cont) != 10 or len(year) != 1 or len(cats) != 3:
        raise ValueError(f"knn_blocks needs the F1 columns (ten continuous audio columns, year, three categories), "
                         f"got {list(columns)}")
    w_cont = 1.0 / np.sqrt(len(cont)) if l2 else 1.0 / len(cont)
    w_cat = float(prep["category_weight"]) / (np.sqrt(len(cats)) if l2 else 2.0 * len(cats))

    def weighted(steps, factor):
        return Pipeline(steps + [("weight", FunctionTransformer(scale_by, kw_args={"factor": float(factor)}))])

    numeric = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]
    parts = [("continuous", weighted(list(numeric), w_cont), cont),
             ("year", weighted(list(numeric), prep["alpha"]), year),
             ("categories", weighted([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))], w_cat),
              cats)]
    return Pipeline([("log", FunctionTransformer(log_columns)),
                     ("columns", ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False))])


def make_preprocessing(prep, columns):
    """None for "none"; the kNN representation for "knn_blocks"; otherwise the phase 1 log transforms, then one-hot categories (CATEGORICAL) and, for every other column, median imputation followed by
    scaling ("log_standard") or by cubic splines and scaling ("spline"). Every statistic is fitted on the rows the
    estimator is fitted on."""
    name = prep["name"]
    if name == "none":
        return None
    if name == "knn_blocks":
        return knn_blocks(prep, columns)
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
    validate_model_params(family, p)
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
    if family == "knn":
        # Deterministic: brute-force distances, no random state; the seed is unused (recipes allow only [0]).
        # Training rows use permanent row-id order. Tied-neighbor behavior is that of the pinned scikit-learn
        # implementation; repeatability and identical-query predictions are tested.
        return KNeighborsClassifier(**p, algorithm="brute", n_jobs=1)
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
    """A context that holds OpenMP to THREADS for families whose estimators take no thread argument (HistGB, and
    kNN, whose distance reductions use OpenMP); the others set their threads in the factory."""
    return threadpool_limits(limits=THREADS, user_api="openmp") if family in ("hist_gb", "knn") \
        else contextlib.nullcontext()


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
