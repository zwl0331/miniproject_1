"""Strict, worker-safe recipe_v2 validation for implemented raw-row adapters.

The legacy scalar control and the small fast-mode architecture menu share the
same raw-row stopping/refit boundary. This module never imports Torch.
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
    advanced = isinstance(recipe, dict) and recipe.get("adapter", {}).get("name") == "raw_neural_v2"
    extension_keys = {"extensions"} if advanced and "extensions" in recipe else set()
    exact_keys(recipe, KEYS | ({"embedding"} if advanced else set()) | extension_keys, "recipe_v2")
    if recipe["schema"] != SCHEMA or recipe["family"] not in ("tabm", "mlp", "resnet", "ft_transformer", "retrieval"):
        raise ValueError("unimplemented raw neural family")
    if not advanced and recipe["family"] != "tabm":
        raise ValueError("legacy adapter supports only scalar TabM")
    name = recipe["name"]
    if not isinstance(name, str) or not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError("recipe name must use letters, digits, underscores or hyphens")
    p = recipe["params"]
    family_params = {"attention_n_heads"} if recipe["family"] == "ft_transformer" else \
                    {"context_size", "context_dropout"} if recipe["family"] == "retrieval" else set()
    exact_keys(p, PARAMS | family_params,
               "raw neural parameters")
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
    expected_adapter = dict(ADAPTER)
    if advanced:
        ext = recipe.get("extensions", {})
        if ext:
            exact_keys(ext, {"features", "layout", "pairwise_weight", "stopping_metric"}, "neural extensions")
            if ext["layout"] not in ("legacy_tabm_v1", "mixed_neural_v1") or ext["stopping_metric"] not in ("log_loss", "auc"):
                raise ValueError("unimplemented neural layout or stopping metric")
            if not number(ext["pairwise_weight"]) or not 0 <= ext["pairwise_weight"] <= 1:
                raise ValueError("pairwise_weight must be in [0,1]")
            features = ext["features"]
            if not isinstance(features, dict) or set(features) - {"year_relative", "year_standardize", "date_counts",
                "year_shrinkage", "year_window", "round", "round_detail", "audio_extra"}:
                raise ValueError("unimplemented neural extra features")
            expected_adapter.update(layout=ext["layout"], stopping_metric=ext["stopping_metric"])
        representation = adapter["representation"]
        if representation not in ("scalar", "ple", "periodic"):
            raise ValueError("unimplemented neural representation")
        expected_adapter.update(name="raw_neural_v2", representation=representation,
            initialization=("random-signs" if representation == "scalar" else "normal")
                if recipe["family"] == "tabm" else "library_default",
            last_batch="singleton_bn_eval" if recipe["family"] == "resnet" else "keep",
            optimizer="ft_adamw_v1" if recipe["family"] == "ft_transformer" else "legacy_adamw_v1")
        emb = recipe["embedding"]
        fields = {"d_embedding", "n_bins"} if representation == "ple" else \
            {"d_embedding", "n_frequencies", "frequency_init_scale"} if representation == "periodic" else set()
        exact_keys(emb, fields, "embedding")
        for field in fields - {"frequency_init_scale"}:
            if not integer(emb[field], 2 if field == "n_bins" else 1):
                raise ValueError(f"invalid embedding {field}")
        if "frequency_init_scale" in emb and (not number(emb["frequency_init_scale"]) or emb["frequency_init_scale"] <= 0):
            raise ValueError("frequency_init_scale must be positive")
        if recipe["family"] != "tabm" and p["k"] != 1:
            raise ValueError("single-network architectures require k=1")
        if recipe["family"] == "ft_transformer":
            if representation != "scalar":
                raise ValueError("FT-Transformer uses its native feature tokenizer")
            if not integer(p["attention_n_heads"]) or p["d_block"] % p["attention_n_heads"]:
                raise ValueError("Transformer width must divide into its attention heads")
        if recipe["family"] == "retrieval":
            if not integer(p["context_size"]) or not number(p["context_dropout"]) or not 0 <= p["context_dropout"] < 1:
                raise ValueError("invalid retrieval context parameters")
            if any(b.startswith("date_te") for b in recipe["blocks"]):
                raise ValueError("retrieval forbids supervised features that could carry a query label into candidates")
    if adapter["device"] not in ("cpu", "cuda") or any(adapter[k] != v for k, v in expected_adapter.items()):
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


def make_recipe(family="tabm", *, representation="scalar", device="cuda", blocks=None,
                scaler="standard", estimator_seeds=(0,), name=None, params=None, embedding=None,
                features=None, layout="legacy_tabm_v1", pairwise_weight=0., stopping_metric="log_loss"):
    """Construct an executable recipe. Overrides are validated, never ignored."""
    p = dict(FIXED, n_blocks=2, d_block=128, dropout=0.1, learning_rate=0.001,
             weight_decay=0.0001, batch_size=128)
    if family != "tabm":
        p["k"] = 1
    if family == "ft_transformer":
        p.update(d_block=64, attention_n_heads=4)
    if family == "retrieval":
        p.update(d_block=64, context_size=32, context_dropout=.2)
    p.update(params or {})
    emb = {"d_embedding": 8, "n_bins": 16} if representation == "ple" else \
        {"d_embedding": 8, "n_frequencies": 16, "frequency_init_scale": 0.01} if representation == "periodic" else {}
    emb.update(embedding or {})
    adapter = dict(ADAPTER, name="raw_neural_v2", device=device, representation=representation,
        initialization=("random-signs" if representation == "scalar" else "normal")
            if family == "tabm" else "library_default",
        last_batch="singleton_bn_eval" if family == "resnet" else "keep",
        optimizer="ft_adamw_v1" if family == "ft_transformer" else "legacy_adamw_v1")
    recipe = {"schema": SCHEMA, "name": name or f"{family}_{representation}", "family": family,
        "params": p, "base": "F2", "blocks": list(blocks or []), "drop": [],
        "preprocessing": {"name": "tabm_numeric", "scaler": scaler}, "training_weights": "none",
        "estimator_seeds": list(estimator_seeds), "iterations": "raw_stop_refit_v1",
        "deployment": "fold_average", "adapter": adapter, "embedding": emb}
    if features or layout != "legacy_tabm_v1" or pairwise_weight or stopping_metric != "log_loss":
        recipe["extensions"] = dict(features=features or {}, layout=layout,
                                    pairwise_weight=pairwise_weight, stopping_metric=stopping_metric)
        recipe["adapter"].update(layout=layout, stopping_metric=stopping_metric)
    return validate(recipe)
