"""Strict, worker-safe recipe_v2 validation for implemented raw-row adapters.

Only the executable scalar TabM route is admitted initially. Later architecture
extensions must add explicit fields and their implementation together. This
module never imports model factories, artifacts, or Torch.
"""
import copy
import math

from songpop.phase2.features import BASE_COLUMNS, BLOCK_COLUMNS, feature_columns, undropped_columns

SCHEMA = "recipe_v2"
KEYS = {"schema", "name", "family", "params", "base", "blocks", "drop", "preprocessing", "training_weights",
        "estimator_seeds", "iterations", "deployment", "adapter"}
PARAMS = {"n_blocks", "d_block", "dropout", "learning_rate", "weight_decay", "batch_size", "k", "max_epochs",
          "patience", "tolerance", "inner_folds", "threads"}
ADAPTER = {
    "name": "raw_neural_v1", "numerical_policy": "linux_ieee_deterministic_v1",
    "layout": "legacy_tabm_v1", "representation": "scalar", "initialization": "random-signs",
    "optimizer": "legacy_adamw_v1", "stopping_metric": "log_loss", "seed_policy": "legacy_tabm_v1",
    "last_batch": "keep", "prediction_batch_rows": 256,
}
FIXED = {"k": 16, "max_epochs": 300, "patience": 20, "tolerance": 1e-6, "inner_folds": 5, "threads": 4}


def exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


def integer(value, minimum=1):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(recipe):
    exact_keys(recipe, KEYS, "recipe_v2")
    if recipe["schema"] != SCHEMA or recipe["family"] != "tabm":
        raise ValueError("recipe_v2 currently implements only the raw scalar TabM adapter")
    name = recipe["name"]
    if not isinstance(name, str) or not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError("recipe name must use letters, digits, underscores or hyphens")
    p = recipe["params"]
    exact_keys(p, PARAMS, "raw TabM parameters")
    for field in ("n_blocks", "d_block", "batch_size", "k", "max_epochs", "patience", "inner_folds", "threads"):
        if not integer(p[field]):
            raise ValueError(f"{field} must be a positive integer")
    if p["inner_folds"] != 5 or p["threads"] != 4:
        raise ValueError("the declared route requires five stopping folds and four CPU threads")
    for field in ("dropout", "learning_rate", "weight_decay", "tolerance"):
        if not number(p[field]):
            raise ValueError(f"{field} must be a finite number")
    if not (0 <= p["dropout"] < 1 and p["learning_rate"] > 0 and p["weight_decay"] >= 0 and p["tolerance"] >= 0):
        raise ValueError("raw TabM optimization parameters are outside their valid domains")
    adapter = recipe["adapter"]
    exact_keys(adapter, set(ADAPTER) | {"device"}, "raw neural adapter")
    if adapter["device"] not in ("cpu", "cuda") or any(adapter[k] != v for k, v in ADAPTER.items()):
        raise ValueError("unimplemented raw neural representation, policy or device")
    if not integer(adapter["prediction_batch_rows"]) or adapter["prediction_batch_rows"] != 256:
        raise ValueError("prediction batch size must be exactly 256")
    if recipe["base"] != "F2" or recipe["base"] not in BASE_COLUMNS:
        raise ValueError("the legacy TabM input layout requires base F2")
    blocks = recipe["blocks"]
    if (not isinstance(blocks, list) or not all(isinstance(b, str) and b in BLOCK_COLUMNS for b in blocks)
            or len(set(blocks)) != len(blocks)):
        raise ValueError("raw feature blocks must be distinct implemented names")
    if sum(b.startswith("date_te") for b in blocks) > 1:
        raise ValueError("at most one date target encoding is allowed")
    available, drop = undropped_columns(recipe), recipe["drop"]
    if (not isinstance(drop, list) or not all(isinstance(c, str) and c in available for c in drop)
            or len(set(drop)) != len(drop) or not feature_columns(recipe)):
        raise ValueError("drop must contain distinct available columns and retain a feature")
    if drop:
        raise ValueError("column drops are not implemented for the initial raw scalar layout")
    # The historical preprocessor needs continuous and categorical columns.
    # General layouts and all-column removal cases are not silently admitted.
    if any(c in drop for c in ("key", "mode", "time_signature")) or not set(feature_columns(recipe)) - {
            "key", "mode", "time_signature"}:
        raise ValueError("the legacy layout retains all categorical fields and a continuous field")
    prep = recipe["preprocessing"]
    exact_keys(prep, {"name", "scaler"}, "raw neural preprocessing")
    if prep["name"] != "tabm_numeric" or prep["scaler"] not in ("standard", "quantile"):
        raise ValueError("raw scalar TabM requires its declared standard or quantile preprocessing")
    seeds = recipe["estimator_seeds"]
    if (not isinstance(seeds, list) or not seeds or not all(integer(s, 0) for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("estimator seeds must be distinct nonnegative integers")
    if recipe["training_weights"] != "none" or recipe["iterations"] != "raw_stop_refit_v1" \
            or recipe["deployment"] != "fold_average":
        raise ValueError("raw neural recipes require unweighted stop/refit and fold-average deployment")
    return recipe


def from_legacy(recipe, *, device="cpu", name=None):
    """Declare an explicit v2 equivalence control without altering its v1 input."""
    if recipe.get("schema") != "recipe_v1" or recipe.get("family") != "tabm":
        raise ValueError("equivalence controls require an existing recipe_v1 TabM recipe")
    result = copy.deepcopy(recipe)
    result.update(schema=SCHEMA, name=name or recipe["name"] + "_raw", iterations="raw_stop_refit_v1",
                  params={**FIXED, **recipe["params"]}, adapter={**ADAPTER, "device": device})
    return validate(result)
