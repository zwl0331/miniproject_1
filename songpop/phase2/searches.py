"""Search specifications and their deterministic expansion into recipes (plan sections 4.2 and 5.1). The ordered,
deduplicated list is saved before anything is fitted."""
import json
from pathlib import Path

import numpy as np

from songpop import artifacts
from songpop.phase2.recipes import RECIPE_SCHEMA, recipe_hash, validate_recipe

SEARCH_SCHEMA = "search_v1"
KEYS = ("schema", "name", "family", "base", "blocks", "drop", "preprocessing", "training_weights", "estimator_seeds",
        "fixed", "space", "anchors", "draws", "sampler_seed")
PREP = "preprocessing."  # a space or fixed key with this prefix sets a preprocessing parameter
MAX_ATTEMPTS = 10000


def validate_search(spec):
    if not isinstance(spec, dict) or sorted(spec) != sorted(KEYS):
        raise ValueError(f"search keys {sorted(spec) if isinstance(spec, dict) else spec!r}, expected {sorted(KEYS)}")
    if spec["schema"] != SEARCH_SCHEMA:
        raise ValueError(f"search schema {spec['schema']!r}, expected {SEARCH_SCHEMA!r}")
    if not isinstance(spec["space"], dict) or not all(isinstance(v, list) and v for v in spec["space"].values()):
        raise ValueError("every space entry must be a non-empty list of values")
    if set(spec["space"]) & set(spec["fixed"]):
        raise ValueError("a parameter is both fixed and searched")
    if not isinstance(spec["draws"], int) or spec["draws"] < 0 or not isinstance(spec["anchors"], list):
        raise ValueError("draws must be a non-negative integer and anchors a list of parameter objects")
    return spec


def _recipe(spec, i, values):
    params = {k: v for k, v in values.items() if not k.startswith(PREP)}
    prep = dict(spec["preprocessing"], **{k[len(PREP):]: v for k, v in values.items() if k.startswith(PREP)})
    return validate_recipe({"schema": RECIPE_SCHEMA, "name": f"{spec['name']}_{i:03d}", "family": spec["family"],
                            "params": params, "base": spec["base"], "blocks": list(spec["blocks"]),
                            "drop": list(spec["drop"]),
                            "preprocessing": prep, "training_weights": spec["training_weights"],
                            "estimator_seeds": list(spec["estimator_seeds"]), "iterations": "fixed",
                            "deployment": "fold_average"})


def expand(spec):
    """Anchors first, exactly as given; then draws from the seeded sampler, each parameter uniform over its list,
    skipping any draw whose canonical recipe is already present (recipe_hash ignores the name), until `draws` new
    recipes exist."""
    validate_search(spec)
    rng = np.random.default_rng(spec["sampler_seed"])
    seen, out = set(), []
    for params in spec["anchors"]:
        r = _recipe(spec, len(out), params)
        if recipe_hash(r) in seen:
            raise ValueError("duplicate anchors: two anchors give the same recipe")
        seen.add(recipe_hash(r))
        out.append(r)
    names = sorted(spec["space"])
    drawn, attempts = 0, 0
    while drawn < spec["draws"]:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            raise ValueError(f"the space holds fewer than {spec['draws']} distinct new configurations")
        params = dict(spec["fixed"])
        for n in names:
            v = spec["space"][n][int(rng.integers(len(spec["space"][n])))]
            params[n] = v.item() if isinstance(v, np.generic) else v
        r = _recipe(spec, len(out), params)
        if recipe_hash(r) in seen:
            continue
        seen.add(recipe_hash(r))
        out.append(r)
        drawn += 1
    return out


def search_id(spec):
    return f"{spec['name']}_{artifacts.sha256_bytes(artifacts.dumps(validate_search(spec)).encode())[:12]}"


def save_expansion(root, directory, spec):
    """Write the ordered configurations once, before any fit; an existing file must hold the same content.
    Returns (search id, recipes, relative path)."""
    sid, recipes = search_id(spec), expand(spec)
    content = json.dumps({"search_id": sid, "spec": spec, "recipes": recipes,
                          "recipe_sha256": [recipe_hash(r) for r in recipes]}, indent=1, sort_keys=True,
                         allow_nan=False) + "\n"
    rel = Path(directory) / f"{sid}.json"
    path = Path(root) / rel
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"{rel} exists with different content")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x") as f:
            f.write(content)
    return sid, recipes, rel
