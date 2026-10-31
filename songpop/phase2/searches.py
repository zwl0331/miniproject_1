"""Search specifications and their deterministic expansion into recipes (plan sections 4.2 and 5.1). The ordered,
deduplicated list is saved before anything is fitted."""
import itertools
import json
from pathlib import Path

import numpy as np

from songpop import artifacts
from songpop.phase2.recipes import RECIPE_SCHEMA, iteration_policy, recipe_hash, validate_recipe

SEARCH_SCHEMA = "search_v1"
SEARCH_SCHEMA_V2 = "search_v2"  # adds declarative derived parameters and admissible regions (P2.2)
KEYS = ("schema", "name", "family", "base", "blocks", "drop", "preprocessing", "training_weights", "estimator_seeds",
        "fixed", "space", "anchors", "draws", "sampler_seed")
KEYS_V2 = KEYS + ("derived", "admissible")
OPS = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
       "eq": lambda a, b: a == b, "ne": lambda a, b: a != b}
PREP = "preprocessing."  # a space or fixed key with this prefix sets a preprocessing parameter
MAX_ATTEMPTS = 10000


def validate_rules(derived, admissible):
    """derived: [{"when": {"param", "op", "value"}, "set": {name: value}}], applied in order to a draw; a set
    parameter is present only when its condition holds. admissible: [{"product": [names], "min", "max"}], each a
    closed interval for the product of the drawn values."""
    if not isinstance(derived, list) or not isinstance(admissible, list):
        raise ValueError("derived and admissible must be lists")
    for d in derived:
        if not isinstance(d, dict) or sorted(d) != ["set", "when"] or not isinstance(d["set"], dict) or not d["set"] \
                or not isinstance(d["when"], dict) or sorted(d["when"]) != ["op", "param", "value"] \
                or d["when"]["op"] not in OPS:
            raise ValueError(f"malformed derived rule {d!r}")
    for a in admissible:
        if not isinstance(a, dict) or sorted(a) != ["max", "min", "product"] or not isinstance(a["product"], list) \
                or len(a["product"]) < 2 or not a["min"] <= a["max"]:
            raise ValueError(f"malformed admissible region {a!r}")


def validate_search(spec):
    if not isinstance(spec, dict) or spec.get("schema") not in (SEARCH_SCHEMA, SEARCH_SCHEMA_V2):
        raise ValueError(f"search schema {spec.get('schema') if isinstance(spec, dict) else spec!r}, expected "
                         f"{SEARCH_SCHEMA!r} or {SEARCH_SCHEMA_V2!r}")
    keys = KEYS if spec["schema"] == SEARCH_SCHEMA else KEYS_V2
    if sorted(spec) != sorted(keys):
        raise ValueError(f"search keys {sorted(spec)}, expected {sorted(keys)}")
    if not isinstance(spec["space"], dict) or not all(isinstance(v, list) and v for v in spec["space"].values()):
        raise ValueError("every space entry must be a non-empty list of values")
    if spec["schema"] == SEARCH_SCHEMA_V2:
        validate_rules(spec["derived"], spec["admissible"])
        targets = {k for d in spec["derived"] for k in d["set"]}
        if targets & (set(spec["space"]) | set(spec["fixed"])):
            raise ValueError("a derived parameter is also fixed or searched")
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
                            "estimator_seeds": list(spec["estimator_seeds"]),
                            "iterations": iteration_policy(spec["family"]),
                            "deployment": "fold_average"})


def apply_derived(params, derived):
    out = dict(params)
    for d in derived:
        w = d["when"]
        if w["param"] in out and OPS[w["op"]](out[w["param"]], w["value"]):
            out.update(d["set"])
    return out


def admissible(params, regions):
    """Whether every declared product lies in its closed interval; a region whose parameters are not all present
    does not apply."""
    for a in regions:
        if all(n in params for n in a["product"]):
            prod = float(np.prod([params[n] for n in a["product"]]))
            if not a["min"] <= prod <= a["max"]:
                return False
    return True


def expand(spec):
    """Anchors first, exactly as given; then draws from the seeded sampler, each parameter uniform over its list,
    skipping any draw whose canonical recipe is already present (recipe_hash ignores the name), until `draws` new
    recipes exist. search_v2 then applies the derived rules to each draw and rejects draws outside the admissible
    regions (anchors are exempt from both); expand_v2 also returns the rejection counts."""
    if validate_search(spec)["schema"] == SEARCH_SCHEMA_V2:
        return expand_v2(spec)[0]
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


ENUMERATE_LIMIT = 200000  # a space this small is enumerated to prove exhaustion


def _draw(spec, names, rng):
    params = dict(spec["fixed"])
    for n in names:
        v = spec["space"][n][int(rng.integers(len(spec["space"][n])))]  # indexed: no dtype coercion of mixed lists
        params[n] = v.item() if isinstance(v, np.generic) else v
    return apply_derived(params, spec["derived"])


def _exhausted(spec, names, seen, key):
    """Whether every admissible configuration of the space is already seen, by enumeration; None if the space is
    too large to enumerate."""
    sizes = [len(spec["space"][n]) for n in names]
    if int(np.prod(sizes, dtype=np.int64)) > ENUMERATE_LIMIT:
        return None
    for combo in itertools.product(*(spec["space"][n] for n in names)):
        params = apply_derived(dict(spec["fixed"], **dict(zip(names, combo))), spec["derived"])
        if admissible(params, spec["admissible"]) and key(_recipe(spec, 0, params)) not in seen:
            return False
    return True


def expand_v2(spec, exclude=(), allow_fewer=False, key=recipe_hash):
    """search_v2: (recipes, info). Anchors first, exact and exempt from the rules; then draws from the seeded
    sampler, each parameter uniform over its list, the derived rules applied, draws outside the admissible regions
    rejected and draws whose key (by default the canonical recipe hash) is seen redrawn; rejection sampling from
    the uniform product distribution gives the uniform distribution over the admissible new configurations.
    exclude: keys of earlier configurations, never drawn again. allow_fewer: when MAX_ATTEMPTS draws find no
    more, an enumeration must prove the space exhausted (info["termination"] = "enumerated_exhaustion");
    otherwise, or without allow_fewer, the expansion fails. info also counts inadmissible and duplicate draws."""
    validate_search(spec)
    rng = np.random.default_rng(spec["sampler_seed"])
    seen, out = set(exclude), []
    for params in spec["anchors"]:
        r = _recipe(spec, len(out), params)
        if key(r) in seen:
            raise ValueError("duplicate anchors: an anchor repeats another or an excluded recipe")
        seen.add(key(r))
        out.append(r)
    names = sorted(spec["space"])
    drawn, attempts = 0, 0
    info = {"inadmissible": 0, "duplicate": 0, "termination": "requested_count"}
    while drawn < spec["draws"]:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            done = _exhausted(spec, names, seen, key) if allow_fewer else None
            if done:
                info["termination"] = "enumerated_exhaustion"
                break
            raise ValueError(f"no new admissible configuration in {MAX_ATTEMPTS} draws after {drawn} of "
                             f"{spec['draws']}" + (", and the space is not proven exhausted" if allow_fewer else ""))
        params = _draw(spec, names, rng)
        if not admissible(params, spec["admissible"]):
            info["inadmissible"] += 1
            continue
        r = _recipe(spec, len(out), params)
        if key(r) in seen:
            info["duplicate"] += 1
            continue
        seen.add(key(r))
        out.append(r)
        drawn += 1
    return out, info


def search_id(spec):
    return f"{spec['name']}_{artifacts.sha256_bytes(artifacts.dumps(validate_search(spec)).encode())[:12]}"


def save_expansion(root, directory, spec):
    """Write the ordered configurations once, before any fit; an existing file must hold the same content.
    Returns (search id, recipes, relative path)."""
    sid = search_id(spec)
    if spec["schema"] == SEARCH_SCHEMA_V2:
        recipes, info = expand_v2(spec)
        extra = {"rejected_draws": info}
    else:
        recipes, extra = expand(spec), {}
    content = json.dumps({"search_id": sid, "spec": spec, "recipes": recipes,
                          "recipe_sha256": [recipe_hash(r) for r in recipes], **extra}, indent=1, sort_keys=True,
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
