"""Recipes: JSON descriptions of a model, its features and its policies (plan section 4.2)."""
import json
from pathlib import Path

from songpop import artifacts

RECIPE_SCHEMA = "recipe_v1"
FAMILIES = ("lightgbm", "random_forest", "extra_trees", "xgboost", "hist_gb", "logistic", "catboost", "ebm")
NEEDS_IMPUTATION = ("logistic",)  # families without native missing-value support
PREPROCESSING = {"none": set(), "log_standard": set(), "spline": {"n_knots"}}  # name: its parameters
BASES = ("F0", "F1", "F2")
BLOCKS = ("date_count", "date_te10", "date_te30", "interactions", "cyclic", "era_relative")
WEIGHTS = ("none", "inverse_group_size")
KEYS = ("schema", "name", "family", "params", "base", "blocks", "drop", "preprocessing", "training_weights",
        "estimator_seeds", "iterations", "deployment")


def validate_recipe(recipe):
    """The exact keys and allowed values; raises ValueError."""
    if not isinstance(recipe, dict) or sorted(recipe) != sorted(KEYS):
        raise ValueError(f"recipe keys {sorted(recipe) if isinstance(recipe, dict) else recipe!r}, "
                         f"expected {sorted(KEYS)}")
    if recipe["schema"] != RECIPE_SCHEMA:
        raise ValueError(f"recipe schema {recipe['schema']!r}, expected {RECIPE_SCHEMA!r}")
    name = recipe["name"]
    if not isinstance(name, str) or not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError(f"recipe name {name!r} must be letters, digits, underscores or hyphens")
    if recipe["family"] not in FAMILIES:
        raise ValueError(f"unknown family {recipe['family']!r}")
    if not isinstance(recipe["params"], dict) or any(isinstance(v, bool) for v in recipe["params"].values()):
        raise ValueError("params must be an object without boolean values")
    if recipe["base"] not in BASES:
        raise ValueError(f"unknown base feature set {recipe['base']!r}")
    blocks = recipe["blocks"]
    if not isinstance(blocks, list) or len(set(blocks)) != len(blocks) or set(blocks) - set(BLOCKS):
        raise ValueError(f"blocks {blocks!r} must be distinct names from {BLOCKS}")
    if "date_te10" in blocks and "date_te30" in blocks:
        raise ValueError("at most one date target encoding per recipe")
    from songpop.phase2.features import undropped_columns  # features does not import recipes
    drop, available = recipe["drop"], undropped_columns(recipe)
    if not isinstance(drop, list) or len(set(drop)) != len(drop) or set(drop) - set(available) \
            or len(drop) >= len(available):
        raise ValueError(f"drop {drop!r} must name distinct feature columns of the base and blocks, leaving at least one")
    prep = recipe["preprocessing"]
    if not isinstance(prep, dict) or prep.get("name") not in PREPROCESSING \
            or set(prep) - {"name"} != PREPROCESSING[prep["name"]]:
        raise ValueError(f"preprocessing {prep!r} must name one of {sorted(PREPROCESSING)} with exactly its parameters")
    if prep["name"] == "spline" and not (isinstance(prep["n_knots"], int) and not isinstance(prep["n_knots"], bool)
                                         and prep["n_knots"] >= 3):
        raise ValueError("spline preprocessing needs an integer n_knots of at least 3")
    if recipe["family"] in NEEDS_IMPUTATION and prep["name"] == "none":
        raise ValueError(f"{recipe['family']} has no native missing-value support; it needs a preprocessing that imputes")
    if recipe["training_weights"] not in WEIGHTS:
        raise ValueError(f"training_weights {recipe['training_weights']!r} not in {WEIGHTS}")
    seeds = recipe["estimator_seeds"]
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds)):
        raise ValueError(f"estimator_seeds {seeds!r} must be distinct non-negative integers")
    if recipe["iterations"] != "fixed":
        raise ValueError("only fixed iteration counts are implemented (plan section 5.2)")
    if recipe["deployment"] != "fold_average":
        raise ValueError("only fold-average deployment is implemented; full refits arrive with P2.7")
    return recipe


def canonical(recipe):
    """Canonical JSON of the computation: sorted keys, no NaN or infinity, the human name left out. Lists such as
    the blocks keep their order."""
    validate_recipe(recipe)
    return artifacts.dumps({k: v for k, v in recipe.items() if k != "name"})


def recipe_hash(recipe):
    return artifacts.sha256_bytes(canonical(recipe).encode())


def recipe_id(recipe):
    return f"{recipe['name']}_{recipe_hash(recipe)[:12]}"


def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate JSON keys {sorted(k for k in set(keys) if keys.count(k) > 1)}")
    return dict(pairs)


def _no_constant(name):
    raise ValueError(f"non-finite number {name} in JSON")


def strict_loads(text):
    """JSON that rejects duplicate keys and NaN or infinity."""
    return json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_no_constant)


def load_recipe(path):
    return validate_recipe(strict_loads(Path(path).read_text()))
