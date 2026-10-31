"""Local refinement proposals (plan section 5.2): new configurations around declared parents, plus broad draws,
deduplicated against every earlier configuration of the family. Deterministic from the saved specification, so
the proposal list is saved before the first fit and re-derived by verification."""
import itertools
import math

import numpy as np

from songpop import artifacts
from songpop.phase2.recipes import RECIPE_SCHEMA, recipe_hash, validate_recipe
from songpop.phase2.searches import PREP, admissible, apply_derived, validate_rules

REFINE_SCHEMA = "refine_v1"
KEYS = ("schema", "name", "family", "parents", "local_per_parent", "trees", "broad", "broad_template", "broad_space",
        "types", "derived", "admissible", "exclude_recipe_sha256", "sampler_seed", "max_dimensions")
FACTOR = (0.5, 2.0)  # multiplicative range of a scale parameter's move, drawn log-uniformly
SIGNIFICANT = 3  # moved floats are rounded to three significant digits
SLOT_ATTEMPTS = 500  # draws one proposal slot may take before its neighborhood counts as exhausted
ENUMERATE_LIMIT = 200000  # a broad space this small is enumerated to prove exhaustion


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_types(types):
    """types: {name: spec}; spec is {"type": "log", "min", "max", "integer", "zero_step", "default"} for a
    positive scale parameter (zero_step: where a zero value moves, or null if zero is not allowed), {"type":
    "ordinal", "values": [...], "default"} for an ordered list moved to a neighbor (values may mix "sqrt" or null
    with numbers, in the declared order), or {"type": "categorical", "values": [...], "default"}. default is the
    factory's value when a parent leaves the parameter out."""
    for n, t in types.items():
        kind = t.get("type") if isinstance(t, dict) else None
        if kind == "log":
            if sorted(t) != ["default", "integer", "max", "min", "type", "zero_step"] \
                    or not (_number(t["min"]) and _number(t["max"]) and 0 < t["min"] < t["max"]):
                raise ValueError(f"malformed log type for {n}: {t!r}")
        elif kind in ("ordinal", "categorical"):
            if sorted(t) != ["default", "type", "values"] or not isinstance(t["values"], list) or len(t["values"]) < 2 \
                    or len({artifacts.dumps(v) for v in t["values"]}) != len(t["values"]):
                raise ValueError(f"malformed {kind} type for {n}: {t!r}")
        else:
            raise ValueError(f"unknown parameter type for {n}: {t!r}")


def validate_refine(spec):
    if not isinstance(spec, dict) or sorted(spec) != sorted(KEYS) or spec["schema"] != REFINE_SCHEMA:
        raise ValueError(f"refinement keys {sorted(spec) if isinstance(spec, dict) else spec!r}, expected {sorted(KEYS)}")
    validate_types(spec["types"])
    validate_rules(spec["derived"], spec["admissible"])
    parents = spec["parents"]
    if not parents or len(parents) != len(spec["local_per_parent"]) or any(validate_recipe(p)["family"] != spec["family"]
                                                                          for p in parents):
        raise ValueError("one local count per parent recipe, every parent of the family")
    if not all(isinstance(k, int) and k >= 0 for k in spec["local_per_parent"]) or not isinstance(spec["broad"], int) \
            or spec["broad"] < 0 or not isinstance(spec["max_dimensions"], int) or spec["max_dimensions"] < 1:
        raise ValueError("counts must be non-negative integers and max_dimensions positive")
    if spec["trees"] is not None and not all(spec["trees"] in p["params"] for p in parents[:1]):
        raise ValueError("trees must name an iteration parameter the first parent sets, or be null")
    if validate_recipe(spec["broad_template"])["family"] != spec["family"]:
        raise ValueError("the broad template must be a recipe of the family")
    return spec


def _round(v, integer):
    if integer:
        return int(round(v))
    return float(f"{v:.{SIGNIFICANT}g}")


def move(value, t, rng):
    """One move of a parameter value under its declared type. A scale value moves by a log-uniform factor in
    FACTOR, clipped to its active bounds [min, max]; with a declared zero_step, zero moves to zero_step and a move
    below zero_step lands on zero (an explicit move to zero)."""
    if t["type"] == "log":
        if value == 0:
            if t["zero_step"] is None:
                raise ValueError("a zero value without a declared zero_step")
            return t["zero_step"]
        v = value * math.exp(rng.uniform(math.log(FACTOR[0]), math.log(FACTOR[1])))
        if t["zero_step"] is not None and v < t["zero_step"]:
            return 0
        return _round(min(max(v, t["min"]), t["max"]), t["integer"])
    keys = [artifacts.dumps(v) for v in t["values"]]
    if t["type"] == "categorical":
        others = [v for v, k in zip(t["values"], keys) if k != artifacts.dumps(value)]
        return others[int(rng.integers(len(others)))]
    if artifacts.dumps(value) in keys:
        i = keys.index(artifacts.dumps(value))
    else:  # a value off the ordinal list (an anchor's) moves from the nearest numeric neighbor
        nums = [(abs(v - value), j) for j, v in enumerate(t["values"]) if _number(v) and _number(value)]
        if not nums:
            raise ValueError(f"value {value!r} is not on the ordinal list {t['values']!r}")
        i = min(nums)[1]
    steps = [j for j in (i - 1, i + 1) if 0 <= j < len(t["values"])]
    return t["values"][steps[int(rng.integers(len(steps)))]]


def _with_params(parent, values, derived, name):
    """The parent with the given values (model parameters, or preprocessing parameters with the PREP prefix);
    derived parameters are recomputed from the result, so a derived rule may set either kind."""
    targets = {k for d in derived for k in d["set"]}
    flat = {k: v for k, v in parent["params"].items()}
    flat.update({PREP + k: v for k, v in parent["preprocessing"].items() if k != "name"})
    flat = {k: v for k, v in flat.items() if k not in targets}
    flat = apply_derived(dict(flat, **values), derived)
    params = {k: v for k, v in flat.items() if not k.startswith(PREP)}
    prep = {"name": parent["preprocessing"]["name"], **{k[len(PREP):]: v for k, v in flat.items() if k.startswith(PREP)}}
    return validate_recipe(dict(parent, name=name, params=params, preprocessing=prep, schema=RECIPE_SCHEMA))


def applicable(parent, names):
    """The declared parameters a parent can move: model parameters, and preprocessing parameters its own
    preprocessing has (a log_standard logistic parent has no knots to move)."""
    return [n for n in names if not n.startswith(PREP) or n[len(PREP):] in parent["preprocessing"]]


def tree_variants(parent, trees, derived, name_of):
    """The {0.67T, 1.5T} comparisons of plan section 5.2 around a boosted parent with T iterations: round(2T/3) and
    round(1.5T), every other field fixed (the parent itself is the centre and is not refitted)."""
    t = parent["params"][trees]
    for v in (int(math.floor(2 * t / 3 + 0.5)), int(math.floor(1.5 * t + 0.5))):
        yield _with_params(parent, {trees: v}, derived, name_of())


def _value(parent, name, t):
    """The parent's value of a parameter, or the declared factory default when the parent leaves it out."""
    held = parent["preprocessing"] if name.startswith(PREP) else parent["params"]
    return held.get(name[len(PREP):] if name.startswith(PREP) else name, t["default"])


def propose(spec, key=recipe_hash):
    """The ordered proposals: each parent's local slots, in parent order, then the broad draws. A boosted family
    (trees names its iteration parameter) fills the first two slots of the first parent with its tree variants,
    when they are admissible and new. Every other local proposal changes one to max_dimensions applicable
    parameters of its parent. Each broad draw takes every broad_space parameter uniformly from its list, the rest
    from broad_template. Proposals outside the active bounds or admissible regions, or whose key is a parent's, an
    excluded earlier configuration's or an earlier proposal's, are redrawn (a tree variant is dropped and its slot
    filled by a local move). key compares configurations (by default the canonical recipe hash; the protocol passes
    a key that treats equivalent numbers and omitted defaults as equal), so every accepted proposal changes the
    computation; exclude_recipe_sha256 then holds keys.

    Exhausted neighborhoods (plan section 15.4): a local slot that finds no new proposal in SLOT_ATTEMPTS draws ends
    its parent's local proposals, and the parent's remaining slots become broad draws. When broad draws find no new
    configuration in SLOT_ATTEMPTS draws, an enumeration of the broad space must prove it exhausted, and the batch is
    smaller; otherwise the proposal fails. Returns (recipes, info): rejection counts, each parent's shortfall, the
    broad shortfall and the termination."""
    validate_refine(spec)
    rng = np.random.default_rng(spec["sampler_seed"])
    seen = set(spec["exclude_recipe_sha256"]) | {key(p) for p in spec["parents"]}
    out = []
    info = {"inadmissible": 0, "duplicate": 0, "out_of_bounds": 0, "local_shortfall": [], "broad_shortfall": 0,
            "termination": "requested_count"}

    def name_of():
        return f"{spec['name']}_{len(out):03d}"

    def value(r, n):
        held = r["preprocessing"] if n.startswith(PREP) else r["params"]
        return held.get(n[len(PREP):] if n.startswith(PREP) else n)

    def in_bounds(r, parent=None):
        """Every scale value inside its active bounds, except a value kept unchanged from the parent: a declared
        configuration off the space's lists (P2.3's candidate 31) keeps its own values when others move."""
        for n, t in spec["types"].items():
            v = value(r, n)
            if v is None or t["type"] != "log" or v == 0 or (parent is not None and v == value(parent, n)):
                continue
            if not t["min"] <= v <= t["max"]:
                return False
        return True

    def new(r, count=True, parent=None):
        if not in_bounds(r, parent):
            info["out_of_bounds"] += count
            return False
        if not admissible(r["params"], spec["admissible"]):
            info["inadmissible"] += count
            return False
        if key(r) in seen:
            info["duplicate"] += count
            return False
        return True

    def accept(r, parent=None):
        if not new(r, parent=parent):
            return False
        seen.add(key(r))
        out.append(r)
        return True

    def fill(count, draw, parent=None):
        """Up to count accepted proposals from draw(); stops at the first slot without one in SLOT_ATTEMPTS draws."""
        made = 0
        while made < count:
            for _ in range(SLOT_ATTEMPTS):
                if accept(draw(), parent):
                    made += 1
                    break
            else:
                return made
        return made

    leftover = 0
    for i, (parent, count) in enumerate(zip(spec["parents"], spec["local_per_parent"])):
        made = 0
        if i == 0 and spec["trees"] is not None:
            for r in tree_variants(parent, spec["trees"], spec["derived"], name_of):
                if made < count:
                    made += accept(r, parent)
        names = applicable(parent, sorted(spec["types"]))

        def local(parent=parent, names=names):
            d = int(rng.integers(1, min(spec["max_dimensions"], len(names)) + 1))
            chosen = sorted(rng.choice(len(names), size=d, replace=False).tolist())
            values = {names[j]: move(_value(parent, names[j], spec["types"][names[j]]), spec["types"][names[j]], rng)
                      for j in chosen}
            return _with_params(parent, values, spec["derived"], name_of())
        made += fill(count - made, local, parent)
        info["local_shortfall"].append(count - made)
        leftover += count - made
    broad = sorted(spec["broad_space"])

    def values_of(combo):
        return {n: (v.item() if isinstance(v, np.generic) else v) for n, v in zip(broad, combo)}

    def broad_draw():
        return _with_params(spec["broad_template"], values_of(
            [spec["broad_space"][n][int(rng.integers(len(spec["broad_space"][n])))] for n in broad]),
            spec["derived"], name_of())
    want = spec["broad"] + leftover
    made = fill(want, broad_draw)
    if made < want:
        sizes = [len(spec["broad_space"][n]) for n in broad]
        if int(np.prod(sizes, dtype=np.int64)) > ENUMERATE_LIMIT or any(
                new(_with_params(spec["broad_template"], values_of(c), spec["derived"], "probe"), count=False)
                for c in itertools.product(*(spec["broad_space"][n] for n in broad))):
            raise ValueError(f"{want - made} broad draws found nothing new in {SLOT_ATTEMPTS} draws, and the broad "
                             f"space is not proven exhausted")
        info["broad_shortfall"] = want - made
        info["termination"] = "enumerated_exhaustion"
    return out, info
