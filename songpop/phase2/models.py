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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from songpop.models import LOG100
from songpop.phase2.tabm import TabMClassifier  # imported here, so its source is in every run's manifest from the start

THREADS = 4
CATEGORICAL = ["key", "mode", "time_signature"]  # one-hot under log_standard, spline and knn_blocks, never splined
OPTIONAL_PACKAGES = {"catboost": "catboost", "ebm": "interpret-core", "tabm": ("tabm", "torch")}

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
    "catboost": {"iterations", "learning_rate", "depth", "l2_leaf_reg", "random_strength", "boosting_type",
                 "bootstrap_type", "bagging_temperature", "rsm"},
    "ebm": {"max_rounds", "learning_rate", "interactions"},
    "knn": {"n_neighbors", "weights", "metric"},
    "tabm": {"n_blocks", "d_block", "dropout", "learning_rate", "weight_decay", "batch_size"},  # P2.5
}
# TabM settings fixed by the factory (plan section 5.5), written out so an omitted argument cannot change the method.
TABM_FIXED = {"k": 16, "max_epochs": 300, "patience": 20, "tolerance": 1e-6, "inner_folds": 5}
KNN_METRICS = {"euclidean": "l2", "manhattan": "l1"}  # each metric with the block normalization it requires
# EBM pair terms (plan section 5.2): "none" fits main effects only; "declared3" adds these three pairs, resolved to
# column indices after the recipe's feature construction. interpret's default ("3x", automatic pairs) is never used.
EBM_PAIRS = {"none": [], "declared3": [("year", "loudness"), ("year", "acousticness"), ("energy", "loudness")]}
# EBM settings fixed by the factory (plan section 17.2): no validation split and no early stopping, one bag; the
# rest are interpret-core 0.7.8's defaults, written out so that an omitted argument cannot change the method.
# Every feature is binned as continuous, which is also what 0.7.8 infers for the float64 frames it receives.
EBM_FIXED = {"validation_size": 0, "early_stopping_rounds": 0, "outer_bags": 1, "inner_bags": 0,
             "objective": "log_loss", "max_bins": 1024, "max_interaction_bins": 64, "greedy_ratio": 10.0,
             "cyclic_progress": False, "smoothing_rounds": 75, "interaction_smoothing_rounds": 75, "max_leaves": 2,
             "min_samples_leaf": 4, "missing": "separate"}


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
    "boosting_type": lambda v: v in ("Plain", "Ordered"), "interactions": lambda v: v in EBM_PAIRS,
    "bootstrap_type": lambda v: v in ("MVS", "Bayesian"),
    "bagging_temperature": lambda v: _num(v) and v >= 0, "rsm": _fraction,
    "n_blocks": _int, "d_block": _int, "batch_size": _int, "dropout": lambda v: _num(v) and 0 <= v < 1,
    "weight_decay": lambda v: _num(v) and v >= 0,
}


FAMILY_DOMAINS = {("xgboost", "max_depth"): _int}
# Parameters whose values are counts: compared as integers; every other numeric parameter is compared as a float.
INTEGER_PARAMS = {"n_blocks", "d_block", "batch_size", "n_estimators", "max_iter", "iterations", "max_rounds", "n_neighbors", "num_leaves", "max_leaf_nodes",
                  "min_child_samples", "min_samples_leaf", "subsample_freq", "max_depth", "depth", "n_knots"}


def validate_model_params(family, params):
    """Names allowed for the family and values in their domains; raises ValueError."""
    extra = set(params) - ALLOWED[family]
    if extra:
        raise ValueError(f"parameters {sorted(extra)} are not allowed for {family}")
    bad = sorted(k for k, v in params.items() if not FAMILY_DOMAINS.get((family, k), DOMAINS[k])(v))
    if bad:
        raise ValueError(f"{family} parameters out of domain: { {k: params[k] for k in bad} }")
    if ("bagging_temperature" in params) != (params.get("bootstrap_type") == "Bayesian"):
        # CatBoost 1.2.10 accepts a temperature with MVS and ignores it: refuse the silent no-op, and an implicit one
        raise ValueError("bagging_temperature is set exactly when bootstrap_type is 'Bayesian'")


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
    robust = prep.get("audio_scaler", "standard") == "robust"  # knn_blocks_v2 (P2.4): robust audio scaling
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
    audio = [numeric[0], ("scale", RobustScaler(quantile_range=(25, 75), unit_variance=True))] if robust else numeric
    parts = [("continuous", weighted(list(audio), w_cont), cont),
             ("year", weighted(list(numeric), prep["alpha"]), year),
             ("categories", weighted([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))], w_cat),
              cats)]
    return Pipeline([("log", FunctionTransformer(log_columns)),
                     ("columns", ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False))])


SPLINE_CORE = ["speechiness", "danceability", "duration_ms", "energy", "instrumentalness", "valence", "tempo",
               "loudness", "acousticness", "liveness", "year"]  # plan section 6.2's splined columns
CALENDAR_CATEGORICAL = CATEGORICAL + ["date_precision"]
CYCLIC_PAIRS = [("month_sin", "month_cos"), ("weekday_sin", "weekday_cos"), ("doy_sin", "doy_cos")]


def missing_pairs(X):
    """One indicator per cyclic phase pair: 1 where the phase is unobserved (its sine is missing)."""
    return np.isnan(np.asarray(X, dtype="float64")[:, ::2]).astype("float64")


def spline_calendar(prep, columns):
    """Spline logistic preprocessing for frames beyond F1 (plan section 18; P2.4): the phase 1 log transforms; cubic
    splines with scaling on the eleven core columns of section 6.2 only; key, mode, time signature and date precision
    one-hot; cyclic coordinates zero-imputed, as linear inputs, with one missingness indicator per phase pair; every
    other column (the further F2 date parts, the date blocks) median-imputed and standardized as a linear input.
    Every statistic is fitted on the estimator's rows. On F1 it builds exactly the spline preprocessing's steps."""
    columns = list(columns)
    core = [c for c in columns if c in SPLINE_CORE]
    cats = [c for c in columns if c in CALENDAR_CATEGORICAL]
    cyc = [c for pair in CYCLIC_PAIRS for c in pair if c in columns]
    if cyc and len(cyc) != 6:
        raise ValueError("spline_calendar needs all six cyclic coordinates or none")
    linear = [c for c in columns if c not in core and c not in cats and c not in cyc]
    if len(core) != len(SPLINE_CORE):
        raise ValueError(f"spline_calendar needs the core columns {SPLINE_CORE}")
    parts = [("rest", Pipeline([("impute", SimpleImputer(strategy="median")),
                                ("spline", SplineTransformer(n_knots=prep["n_knots"], degree=3)),
                                ("scale", StandardScaler())]), core)]
    if cats:
        parts.append(("categories", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                                       sparse_output=False))]), cats))
    if linear:
        parts.append(("linear", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                                          ("scale", StandardScaler())]), linear))
    if cyc:
        parts.append(("cyclic", SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True), cyc))
        parts.append(("cyclic_missing", FunctionTransformer(missing_pairs), cyc))
    return Pipeline([("log", FunctionTransformer(log_columns)),
                     ("columns", ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False))])


def make_preprocessing(prep, columns):
    """None for "none"; the kNN representation for "knn_blocks"; otherwise the phase 1 log transforms, then one-hot categories (CATEGORICAL) and, for every other column, median imputation followed by
    scaling ("log_standard") or by cubic splines and scaling ("spline"). Every statistic is fitted on the rows the
    estimator is fitted on."""
    name = prep["name"]
    if name == "none":
        return None
    if name in ("knn_blocks", "knn_blocks_v2"):
        return knn_blocks(prep, columns)
    if name == "spline_calendar":
        return spline_calendar(prep, columns)
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


def ebm_pairs(name, columns):
    """The declared EBM pairs as column indices of the frame the estimator receives."""
    columns = list(columns)
    missing = sorted({c for pair in EBM_PAIRS[name] for c in pair} - set(columns))
    if missing:
        raise ValueError(f"EBM pairs {name!r} need the columns {missing}")
    return [(columns.index(a), columns.index(b)) for a, b in EBM_PAIRS[name]]


def make_estimator(family, p, seed, columns=None):
    """A new, unfitted estimator of the family with the recipe's parameters and the given estimator seed. columns:
    the frame's columns, needed only to resolve EBM's declared pairs."""
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
        # Numeric features only (no cat_features), no eval set, no class weights, no stopping. A recipe without
        # bootstrap_type gets CatBoost 1.2.10's own CPU choice for this objective, made explicit: MVS with
        # subsample 0.8, for Plain and Ordered alike; rsm 1.0 unless set (plan section 17.2).
        from catboost import CatBoostClassifier
        q = dict(p)
        q.setdefault("bootstrap_type", "MVS")
        if q["bootstrap_type"] == "MVS":
            q["subsample"] = 0.8
        q.setdefault("rsm", 1.0)
        return CatBoostClassifier(**q, loss_function="Logloss", eval_metric="Logloss", random_seed=seed,
                                  thread_count=THREADS, allow_writing_files=False, save_snapshot=False,
                                  use_best_model=False, auto_class_weights="None", nan_mode="Min", has_time=False,
                                  task_type="CPU", grow_policy="SymmetricTree", verbose=False)
    if family == "knn":
        # Deterministic: brute-force distances, no random state; the seed is unused (recipes allow only [0]).
        # Training rows use permanent row-id order. Tied-neighbor behavior is that of the pinned scikit-learn
        # implementation; repeatability and identical-query predictions are tested.
        return KNeighborsClassifier(**p, algorithm="brute", n_jobs=1)
    if family == "ebm":
        # No validation split and no early stopping, so each stage makes max_rounds x (its number of terms) boosting
        # updates (greedy selection may spread them unevenly across terms); one bag; pairs only as declared.
        from interpret.glassbox import ExplainableBoostingClassifier
        if columns is None:
            raise ValueError("an EBM estimator needs the frame's columns to resolve its pairs")
        pairs = ebm_pairs(p.get("interactions", "none"), columns)
        q = dict(p, interactions=pairs if pairs else 0)
        return ExplainableBoostingClassifier(**q, **EBM_FIXED, feature_types=["continuous"] * len(columns),
                                             n_jobs=THREADS, random_state=seed)
    raise ValueError(f"unknown family {family!r}")


def make_model(recipe, seed, columns):
    """The recipe's preprocessing (if any) followed by its family's estimator, for feature frames with these columns.
    TabM owns its preprocessing (songpop.phase2.tabm): its stopping stage and its final stage fit it on different
    rows, so the recipe names it ("tabm_numeric" with its scaler) and the estimator builds it."""
    if recipe["family"] == "tabm":
        validate_model_params("tabm", recipe["params"])
        if set(recipe["params"]) != ALLOWED["tabm"]:
            raise ValueError(f"a tabm recipe sets exactly {sorted(ALLOWED['tabm'])}")
        return TabMClassifier(**recipe["params"], scaler=recipe["preprocessing"]["scaler"], random_state=seed,
                              **TABM_FIXED, threads=THREADS)
    if recipe["family"] in ("catboost", "ebm") and recipe["preprocessing"]["name"] != "none":
        raise ValueError(f"{recipe['family']} takes the feature frame as it is (EBM resolves its pairs on those "
                         f"columns; CatBoost predicts through prediction_view)")
    est = make_estimator(recipe["family"], dict(recipe["params"]), seed, list(columns))
    prep = make_preprocessing(recipe["preprocessing"], list(columns))
    return est if prep is None else Pipeline([("prep", prep), ("model", est)])


def thread_limit(family):
    """A context that holds OpenMP to THREADS for families whose estimators take no thread argument (HistGB, and
    kNN, whose distance reductions use OpenMP); the others set their threads in the factory."""
    return threadpool_limits(limits=THREADS, user_api="openmp") if family in ("hist_gb", "knn") \
        else contextlib.nullcontext()


FORESTS = ("random_forest", "extra_trees")


@contextlib.contextmanager
def prediction_threads(family, estimators):
    """One thread for forest prediction (Finding 013, plan section 16.4): a forest's predict_proba adds its trees'
    probabilities from parallel threads in a varying order, so predictions tied in exact arithmetic can come out in
    either order between calls. Every forest, bare or the final step of a pipeline, predicts with n_jobs=1 inside
    the context; fitting keeps THREADS, and each previous setting is restored on exit. Other families are untouched."""
    if family not in FORESTS:
        yield
        return
    forests = [e.steps[-1][1] if isinstance(e, Pipeline) else e for e in estimators]
    saved = [f.n_jobs for f in forests]
    try:
        for f in forests:
            f.n_jobs = 1
        yield
    finally:
        for f, n in zip(forests, saved):
            f.n_jobs = n


class _ThreadedCatBoost:
    """A CatBoost model seen through predict_proba with THREADS threads: its own default is every CPU core."""

    def __init__(self, est):
        self.est, self.classes_ = est, est.classes_

    def predict_proba(self, X):
        return self.est.predict_proba(X, thread_count=THREADS)


def prediction_view(family, est):
    """The object whose predict_proba gives a fitted estimator's predictions under the thread limit: CatBoost
    through _ThreadedCatBoost (plan section 17.2), every other family unchanged (forests: prediction_threads)."""
    return _ThreadedCatBoost(est) if family == "catboost" else est


def weight_argument(model):
    """The fit keyword that carries training weights to the estimator."""
    return "model__sample_weight" if isinstance(model, Pipeline) else "sample_weight"


def family_packages(recipe):
    """Versions of the optional packages a family needs, without importing them; None when absent."""
    names = OPTIONAL_PACKAGES.get(recipe["family"])
    out = {}
    for name in ((names,) if isinstance(names, str) else names or ()):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out
