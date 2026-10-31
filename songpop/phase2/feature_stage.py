"""The P2.4 feature stage (plan section 18): matched feature comparisons in batches declared before their scores and
judged by the allocation rule of plan section 5.6 against references frozen in each batch's manifest, then the
re-tuning of up to four model-and-feature combinations. Every function here is a pure function of the protocol, the
reference pool (validated runs with their recipes, scores and contribution blends with H) and the earlier batches;
nothing fits a model, reads a file or writes one."""
import copy
import itertools

from songpop import artifacts
from songpop.phase2 import allocation, protocol as waves, refine, searches
from songpop.phase2.recipes import DATE_TE_BLOCKS, RECIPE_SCHEMA, recipe_hash, validate_recipe
from songpop.splits import SPLIT_SEEDS

PROTOCOL_SCHEMA = "feature_protocol_v1"
BATCH_SCHEMA = "feature_batch_v1"
KEYS = ("schema", "task_id", "name", "split_seeds", "cutoff_utc", "retune_budget_end_utc", "reference_p21_run_id",
        "h_recipe", "reference_inventory", "p23_handoff", "imports", "aliases", "initial", "expansion", "retune")
CHANGE_KEYS = ("add", "remove", "drop", "base", "preprocessing", "params", "training_weights")
DATE_BLOCKS = {"date_count"} | set(DATE_TE_BLOCKS)
DATE_VARIANTS = [(True, "date_te30"), (True, "date_te10"), (False, "date_te10"), (False, "date_te30"),
                 (True, "date_te3"), (True, "date_te100"), (False, "date_te3"), (False, "date_te100"), (True, None)]
REMOVALS = [["days"], ["day", "day_of_year"], ["weekday"], ["key", "mode", "time_signature"]]  # plan section 6.6
QUEUES = ("dates", "combinations", "smooth_knn", "removals", "other_trees", "weights")
IMPORT_IDS = [f"B{i}" for i in range(1, 15)]
N_ROWS = 3863


# ---- protocol ------------------------------------------------------------------------------------------------

def load_protocol(p):
    """Checks a parsed protocol: exact keys, the official split seeds, the H recipe, the frozen reference inventory
    and aliases, exactly the imports B1 to B14, the initial comparisons, the expansion queues and the re-tuning."""
    if not isinstance(p, dict) or p.get("schema") != PROTOCOL_SCHEMA or sorted(p) != sorted(KEYS):
        raise ValueError(f"feature protocol keys {sorted(p) if isinstance(p, dict) else p!r}, expected {sorted(KEYS)}")
    if p["split_seeds"] != list(SPLIT_SEEDS):
        raise ValueError("the feature protocol runs on the official split seeds")
    validate_recipe(p["h_recipe"])
    inv = [e["run_id"] for e in p["reference_inventory"]]
    if not inv or len(set(inv)) != len(inv) or any(sorted(e) != ["bank_sha256", "run_id"] for e in p["reference_inventory"]):
        raise ValueError("the reference inventory lists distinct runs, each with its bank hash")
    for name, a in p["aliases"].items():
        if sorted(a) != ["omitted_reason", "rule", "run_id"] or (a["run_id"] is None) == (a["omitted_reason"] is None):
            raise ValueError(f"alias {name}: a run id or an omission reason, exactly one")
        if a["run_id"] is not None and a["run_id"] not in inv:
            raise ValueError(f"alias {name}: its run is not in the reference inventory")
    if [c["id"] for c in p["imports"]["comparisons"]] != IMPORT_IDS:
        raise ValueError(f"the imports must be exactly {IMPORT_IDS}, in order")
    ids = IMPORT_IDS + [c["id"] for c in p["initial"]]
    if len(set(ids)) != len(ids):
        raise ValueError("comparison ids must be distinct")
    for c in p["initial"] + p["expansion"]["smooth_knn"] + p["expansion"]["other_trees"]:
        if c["base"] not in p["aliases"] or not c["change"] or set(c["change"]) - set(CHANGE_KEYS) \
                or set(c.get("comparator_change", {})) - set(CHANGE_KEYS) or set(c) - {"id", "base", "change",
                                                                                       "comparator_change"}:
            raise ValueError(f"malformed comparison {c!r}")
    e = p["expansion"]
    if [q for q, _ in e["queue"]] != list(QUEUES) or sum(n for _, n in e["queue"]) != e["size"]:
        raise ValueError("the expansion queue must list every queue once, its slots summing to the batch size")
    r = p["retune"]
    if r["local"] + r["broad"] != r["batch"] or r["max_per_family"] > r["max_units"] or r["max_batches"] != 2:
        raise ValueError("malformed re-tuning settings")
    return p


def renamed(recipe, name):
    return validate_recipe(dict(copy.deepcopy(recipe), name=name, schema=RECIPE_SCHEMA))


def changed(recipe, change, name):
    """The recipe with a declared change: blocks added (appended in order; a new target encoding replaces the old)
    or removed, columns dropped, the base set, the preprocessing (its name and extra parameters; parameters of the old
    preprocessing that the new one declares are kept), model parameters, the training weights."""
    from songpop.phase2.recipes import PREPROCESSING
    r = copy.deepcopy(recipe)
    if "base" in change:
        r["base"] = change["base"]
    r["blocks"] = [b for b in r["blocks"] if b not in change.get("remove", [])] + \
        [b for b in change.get("add", []) if b not in r["blocks"]]
    if set(change.get("add", [])) & set(DATE_TE_BLOCKS):
        r["blocks"] = [b for b in r["blocks"] if b not in DATE_TE_BLOCKS or b in change["add"]]
    r["drop"] = sorted(set(r["drop"]) | set(change.get("drop", [])))
    if "preprocessing" in change:
        new = dict(change["preprocessing"])
        keep = {k: v for k, v in r["preprocessing"].items() if k != "name" and k in PREPROCESSING[new["name"]]}
        r["preprocessing"] = {**keep, **new}
    if "params" in change:
        r["params"] = dict(r["params"], **change["params"])
    if "training_weights" in change:
        r["training_weights"] = change["training_weights"]
    return renamed(r, name)


def base_name(recipe):
    """A comparator from outside P2.4 is refitted under P2.4's sources with a name fixed by its computation."""
    return f"p24_base_{recipe_hash(recipe)[:10]}"


# ---- the reference pool ----------------------------------------------------------------------------------------

def ref(e):
    return {"run_id": e["run_id"], "bank_sha256": e["bank_sha256"], "mean_fold_auc_unrounded": e["mean"],
            "repeat_mean_auc_unrounded": e["repeats"]}


def best(entries):
    """The highest unrounded mean fold AUC, ties to the lower recipe id."""
    entries = list(entries)
    return min(entries, key=lambda e: (-e["mean"], e["recipe_id"])) if entries else None


def best_blend(entries, h):
    """A family's best contribution blend with H over the given pool entries, floored at H: the highest unrounded
    mean, ties to the lower recipe id, then the lower weight."""
    out = {"run_id": h["run_id"], "weight": 0.0, "bank_sha256": h["bank_sha256"], "mean_fold_auc_unrounded": h["mean"],
           "repeat_mean_auc_unrounded": h["repeats"]}
    cands = [(e, w, b) for e in entries for w, b in e["blends"].items()]
    if cands:
        e, w, b = min(cands, key=lambda x: (-x[2]["mean"], x[0]["recipe_id"], float(x[1])))
        if b["mean"] > h["mean"]:
            out = {"run_id": e["run_id"], "weight": float(w), "bank_sha256": e["bank_sha256"],
                   "mean_fold_auc_unrounded": b["mean"], "repeat_mean_auc_unrounded": b["repeats"]}
    return out


def alias_rule(rule, pool):
    """The pool entry a declared alias rule selects (plan section 18.4), or None: named_run (the run of the named
    tasks whose recipe has that name), best_of_family, best_with_preprocessing (family, base and preprocessing
    name), best_with_column (features holding the column); best by unrounded mean, ties to the lower recipe id."""
    cand = [e for e in pool if e["task_id"] in set(rule["tasks"])]
    if rule["rule"] == "named_run":
        cand = [e for e in cand if e["recipe"]["name"] == rule["recipe_name"]]
        if len(cand) != 1:
            raise ValueError(f"{len(cand)} runs of {rule['recipe_name']}")
    elif rule["rule"] == "best_of_family":
        cand = [e for e in cand if e["family"] == rule["family"]]
    elif rule["rule"] == "best_with_preprocessing":
        cand = [e for e in cand if e["family"] == rule["family"] and e["recipe"]["base"] == rule["base"]
                and e["recipe"]["preprocessing"]["name"] == rule["preprocessing"]]
    elif rule["rule"] == "best_with_column":
        cand = [e for e in cand if rule["column"] in e["features"]]
    else:
        raise ValueError(f"unknown alias rule {rule['rule']!r}")
    return best(cand)


def resolve_aliases(protocol, pool):
    """The frozen aliases of the protocol (resolved from the reference inventory and P2.3's handoff when the
    protocol was written): name -> pool entry, or None for a declared omission."""
    by_id = {e["run_id"]: e for e in pool}
    return {k: (by_id[a["run_id"]] if a["run_id"] is not None else None) for k, a in protocol["aliases"].items()}


def state(pool, h, families, allowance_before):
    """The frozen references of a batch (plan section 5.6): H, G (the strongest standalone pool entry), and for each
    family the incumbent (its best pool entry) and its best contribution blend with H (floored at H)."""
    return {"reference_h": ref(h), "strongest": ref(best(pool)),
            "incumbents": {f: ref(best(e for e in pool if e["family"] == f)) for f in sorted(families)
                           if any(e["family"] == f for e in pool)},
            "previous_best_blends": {f: best_blend([e for e in pool if e["family"] == f], h) for f in sorted(families)},
            "allowance_before": allowance_before}


# ---- comparisons -----------------------------------------------------------------------------------------------

def comparison(cid, comparator, challenger, generation):
    return {"id": cid, "family": challenger["family"], "comparator": comparator, "challenger": challenger,
            "generation": generation}


def declared_challengers(batches, imports):
    """Recipe hash -> the declared recipe (with its name) of every challenger so far, imports included."""
    out = {i["challenger_recipe_sha256"]: i["challenger_recipe"] for i in imports}
    for b in batches:
        out.update({recipe_hash(c["challenger"]): c["challenger"] for c in b["comparisons"]})
    return out


def as_comparator(recipe, declared):
    """A comparator under its P2.4 name: a declared P2.4 challenger with the same computation keeps its recipe, so
    its run is reused; any other recipe from outside P2.4 is renamed by its computation."""
    d = declared.get(recipe_hash(recipe))
    if d is not None and d["name"].startswith("p24_"):
        return d
    return recipe if recipe["name"].startswith("p24_") else renamed(recipe, base_name(recipe))


def menu_pair(item, aliases, name):
    """A declared item's (comparator, challenger): the comparator is the alias's recipe with comparator_change (or
    unchanged), and the challenger is that comparator with change, so the two differ only by the declared change."""
    a = aliases[item["base"]]
    if a is None:
        return None
    comp = changed(a["recipe"], item["comparator_change"], a["recipe"]["name"]) if item.get("comparator_change") \
        else a["recipe"]
    return comp, changed(comp, item["change"], name)


def initial_comparisons(protocol, aliases):
    """The declared initial comparisons; a comparison whose alias is a declared omission is recorded as omitted."""
    out, omitted, declared = [], [], declared_challengers([], protocol["imports"]["comparisons"])
    for c in protocol["initial"]:
        pair = menu_pair(c, aliases, f"p24_{c['id']}")
        if pair is None:
            omitted.append({"id": c["id"], "reason": protocol["aliases"][c["base"]]["omitted_reason"]})
            continue
        out.append(comparison(c["id"], as_comparator(pair[0], declared), pair[1],
                              {"queue": "initial", "base_alias": c["base"], "base_run_id": aliases[c["base"]]["run_id"],
                               "change": c["change"]}))
    return out, omitted


def met_routes(judged, imports):
    """Comparison id -> routes met, over the judged batches and the imports."""
    out = {i["id"]: i["routes"] for i in imports}
    for j in judged:
        out.update({cid: r["routes"] for cid, r in j["comparisons"].items()})
    return out


def _new(base, change, seen):
    """The changed recipe if the change applies and is not already declared, else None."""
    try:
        ch = changed(base, change, "probe")
    except ValueError:
        return None  # an inapplicable change: a column the recipe does not hold, an invalid combination
    if recipe_hash(ch) == recipe_hash(base) or recipe_hash(ch) in seen:
        return None
    return ch


def expansion_candidates(protocol, pool, aliases, batches, judged, imports, seen):
    """Every queue's ordered candidates for the next expansion batch (plan section 18.5): (comparator recipe,
    challenger change, generation). Only applicable changes whose challenger is not declared yet (seen: declared
    challenger hashes) are candidates, so an exhausted base or holder never occupies a place."""
    routes = met_routes(judged, imports)
    pairs = {i["id"]: (i["comparator_recipe"], i["challenger_recipe"]) for i in imports}
    comps = {c["id"]: c for b in batches for c in b["comparisons"]}
    pairs.update({cid: (c["comparator"], c["challenger"]) for cid, c in comps.items()})
    score = {e["recipe_hash"]: e for e in pool}
    out = {q: [] for q in QUEUES}

    def added(cid):
        a, c = pairs[cid]
        return [b for b in c["blocks"] if b not in a["blocks"]]

    def mean_of(recipe):
        e = score.get(recipe_hash(recipe))
        return e["mean"] if e else float("-inf")
    # Dates: comparators of date comparisons that met a route, by their best qualifying challenger's score; then for
    # each declared family without a date comparison, its best pool recipe without date blocks. A base enters only
    # with an untried variant; at most date_bases_per_batch bases.
    date_cids = sorted(cid for cid in pairs if set(added(cid)) & DATE_BLOCKS)
    qual = {}
    for cid in date_cids:
        if routes.get(cid):
            k = recipe_hash(pairs[cid][0])
            qual[k] = max(qual.get(k, float("-inf")), mean_of(pairs[cid][1]))
    by_hash = {recipe_hash(pairs[c][0]): pairs[c][0] for c in date_cids}
    bases = [by_hash[k] for k, _ in sorted(qual.items(), key=lambda kv: (-kv[1], kv[0]))]
    tried = {pairs[cid][0]["family"] for cid in date_cids}
    for fam in protocol["expansion"]["date_families"]:
        if fam not in tried:
            e = best(x for x in pool if x["family"] == fam and not set(x["recipe"]["blocks"]) & DATE_BLOCKS)
            if e is not None:
                bases.append(e["recipe"])
    taken = 0
    for base in bases:
        if taken == protocol["expansion"]["date_bases_per_batch"]:
            break
        variants = [{"add": (["date_count"] if c else []) + ([t] if t else [])} for c, t in DATE_VARIANTS]
        fresh = [v for v in variants if _new(base, v, seen) is not None]
        if fresh:
            taken += 1
            out["dates"] += [(base, v, {"date_base": recipe_hash(base)[:12]}) for v in fresh]
    # Combinations: two comparisons on one comparator, each adding blocks and each meeting a route, their additions
    # united (at most one target encoding).
    single = sorted(cid for cid in pairs if routes.get(cid) and added(cid))
    for x, y in itertools.combinations(single, 2):
        ax, ay = pairs[x][0], pairs[y][0]
        union = added(x) + [b for b in added(y) if b not in added(x)]
        if recipe_hash(ax) == recipe_hash(ay) and len(set(union) & set(DATE_TE_BLOCKS)) <= 1:
            out["combinations"].append((ax, {"add": union}, {"from": [x, y]}))
    # Smooth and kNN preprocessing, and other tree families: the protocol's ordered menus, with their declared
    # comparators.
    for q in ("smooth_knn", "other_trees"):
        for item in protocol["expansion"][q]:
            pair = menu_pair(item, aliases, "probe")
            if pair is not None:
                out[q].append((pair[0], item["change"], {"menu": item["id"], "base_alias": item["base"]}))
    # Removals: for each drop of section 6.6, the two strongest pool recipes for which it is new; then, for a
    # qualifying combination, the combination without each block it added.
    ranked = sorted(pool, key=lambda e: (-e["mean"], e["recipe_id"]))
    for cols in REMOVALS:
        holders = []
        for e in ranked:
            if len(holders) == 2:
                break
            if recipe_hash(e["recipe"]) not in {recipe_hash(h) for h in holders} and \
                    _new(e["recipe"], {"drop": cols}, seen) is not None:
                holders.append(e["recipe"])
        out["removals"] += [(h, {"drop": cols}, {"removal": cols}) for h in holders]
    for cid, c in sorted(comps.items()):
        if c["generation"].get("queue") == "combinations" and routes.get(cid):
            out["removals"] += [(c["challenger"], {"remove": [b]}, {"removal_of_block": b}) for b in added(cid)]
    # Training weights: inverse-group-size weights on challengers that met a route, by score; never kNN.
    for cid in sorted((c for c in pairs if routes.get(c)), key=lambda c: (-mean_of(pairs[c][1]), c)):
        r = pairs[cid][1]
        if r["family"] != "knn" and r["training_weights"] == "none":
            out["weights"].append((r, {"training_weights": "inverse_group_size"}, {"of": cid}))
    return out


def plan_expansion(protocol, pool, aliases, batches, judged, imports, index):
    """The next expansion batch: each queue's slots filled in queue order from its candidates, unused slots released
    to the first queue in order with candidates left, every challenger new; at most the comparison ceiling. Returns
    its comparisons (empty when every queue is exhausted)."""
    e = protocol["expansion"]
    used = len(imports) + sum(len(b["comparisons"]) + len(b.get("omitted", [])) for b in batches)
    size = max(0, min(e["size"], e["max_comparisons"] - used))
    declared = declared_challengers(batches, imports)
    seen = set(declared)
    cands = expansion_candidates(protocol, pool, aliases, batches, judged, imports, seen)
    picks = []

    def take(q):
        while cands[q]:
            base, change, gen = cands[q].pop(0)
            ch = _new(base, change, seen)
            if ch is not None:
                seen.add(recipe_hash(ch))
                return base, ch, dict(gen, queue=q, change=change)
        return None
    quota = dict(e["queue"])
    for q in QUEUES:
        for _ in range(quota[q]):
            if len(picks) < size and (got := take(q)):
                picks.append(got)
    for q in QUEUES:  # released slots, in queue order
        while len(picks) < size and (got := take(q)):
            picks.append(got)
    out = []
    for k, (base, ch, gen) in enumerate(picks, 1):
        cid = f"X{index}_{k:02d}"
        out.append(comparison(cid, as_comparator(base, declared), renamed(ch, f"p24_{cid}"), gen))
    return out


def routes_of(entry, family, st):
    inc = st["incumbents"][family]
    return allocation.routes({"standalone": entry["repeats"],
                              "blends": [entry["blends"][w]["repeats"] for w in sorted(entry["blends"])]},
                             inc["repeat_mean_auc_unrounded"], st["strongest"]["repeat_mean_auc_unrounded"],
                             st["reference_h"]["repeat_mean_auc_unrounded"],
                             st["previous_best_blends"][family]["repeat_mean_auc_unrounded"])


def judge(manifest, entries, imports=None):
    """A closed feature batch's allocation: each comparison's matched repeat differences (challenger minus
    comparator) and the routes its challenger meets against the frozen state of its family; progress; the counter.
    entries: run id -> entry for every run of the batch. The initial batch's progress also counts the imported route
    flags (plan section 16.3); later batches count only their own comparisons."""
    st = manifest["state"]
    out = {}
    for c in manifest["comparisons"]:
        ch, cp = entries[c["challenger_run_id"]], entries[c["comparator_run_id"]]
        out[c["id"]] = {"routes": routes_of(ch, c["family"], st), "matched_difference": ch["mean"] - cp["mean"],
                        "matched_repeat_differences": [a - b for a, b in zip(ch["repeats"], cp["repeats"])],
                        "challenger_mean": ch["mean"],
                        "blend_means": {str(w): ch["blends"][w]["mean"] for w in sorted(ch["blends"])}}
    progress = any(v["routes"] for v in out.values())
    if manifest["kind"] == "initial":
        progress = progress or any(i["routes"] for i in imports or [])
    after = allocation.next_allowance(st["allowance_before"], progress, manifest["kind"] == "initial")
    return {"comparisons": out, "progress": progress, "allowance_after": after}


def judge_retune(manifest, entries):
    """A re-tuning round: each configuration's routes against the round's frozen state of its family; a unit's
    progress is recorded (it does not gate its second batch: plan section 18.6)."""
    st, units = manifest["state"], {}
    for u in manifest["units"]:
        res = {rid: routes_of(entries[rid], u["unit"]["family"], st) for rid in u["run_ids"]}
        units[u["unit"]["unit_id"]] = {"routes": res, "progress": any(res.values())}
    return {"units": units, "progress": any(v["progress"] for v in units.values())}


def witnesses(manifest, judgment, entries):
    """The comparisons behind a batch's allocation (plan sections 5.6 and 18.3), in the manner of section 15.5: per
    family and route met, the qualifying run with the highest relevant score (ties to the lower recipe id) against
    the frozen incumbent (standalone routes), or its better blend with H against H and against the family's previous
    best blend (ensemble route). Each: (route, family, a, b), a and b ("run", id) or ("blend", id, weight)."""
    st, met = manifest["state"], []
    if manifest["kind"] == "retune":
        for u in manifest["units"]:
            met += [(u["unit"]["family"], rid, rts) for rid, rts in judgment["units"][u["unit"]["unit_id"]]["routes"].items()]
    else:
        met = [(c["family"], c["challenger_run_id"], judgment["comparisons"][c["id"]]["routes"])
               for c in manifest["comparisons"]]
    out = []
    for f in sorted({m[0] for m in met}):
        for route in ("clear_standalone", "smaller_useful", "competitive_refinement"):
            q = [entries[rid] for fam, rid, rts in met if fam == f and route in rts]
            if q:
                w = best(q)
                out.append((route, f, ("run", w["run_id"]), ("run", st["incumbents"][f]["run_id"])))
        q = [entries[rid] for fam, rid, rts in met if fam == f and "ensemble_contribution" in rts]
        if q:
            e, w = min(((e, w) for e in q for w in e["blends"]),
                       key=lambda x: (-x[0]["blends"][x[1]]["mean"], x[0]["recipe_id"], float(x[1])))
            a = ("blend", e["run_id"], float(w))
            out.append(("ensemble_contribution", f, a, ("run", manifest["h"]["run_id"])))
            pb = st["previous_best_blends"][f]
            if pb["weight"] != 0.0:
                out.append(("ensemble_contribution", f, a, ("blend", pb["run_id"], pb["weight"])))
    return out


# ---- re-tuning -----------------------------------------------------------------------------------------------

def unit_of(recipe):
    """A re-tuning unit: the family and everything about the recipe except its model parameters, its name and
    spline knots."""
    prep = {k: v for k, v in recipe["preprocessing"].items() if k != "n_knots"}
    body = {"family": recipe["family"], "base": recipe["base"], "blocks": recipe["blocks"], "drop": recipe["drop"],
            "preprocessing": prep, "training_weights": recipe["training_weights"],
            "estimator_seeds": recipe["estimator_seeds"]}
    return {"unit_id": artifacts.sha256_bytes(artifacts.dumps(body).encode())[:12], **body}


def family_key(protocol, family):
    """The family-normalized computation key of section 15.4 (omitted defaults and number types normalized)."""
    return waves.search_key(protocol["retune"]["families"][family])


def choose_units(protocol, pool, routes, recipes):
    """Up to max_units combinations that met a route of section 5.6 (imports included), carry a feature change and
    have a declared re-tuning space: up to by_standalone by the best standalone score of a qualifying configuration,
    then by the best contribution blend; at most max_per_family per family; ties by unit id. recipes: comparison id
    -> challenger recipe."""
    r = protocol["retune"]
    by_hash = {e["recipe_hash"]: e for e in pool}
    units = {}
    for cid in sorted(routes):
        if not routes[cid]:
            continue
        rec = recipes[cid]
        if rec["family"] not in r["families"]:
            continue
        if not rec["blocks"] and not rec["drop"] and rec["training_weights"] == "none" \
                and rec["preprocessing"]["name"] in ("none", "spline", "log_standard", "knn_blocks"):
            continue  # a base recipe is not a feature combination
        u = unit_of(rec)
        e = by_hash[recipe_hash(rec)]
        blend = max(b["mean"] for b in e["blends"].values())
        cur = units.setdefault(u["unit_id"], {"unit": u, "standalone": e["mean"], "blend": blend})
        cur["standalone"], cur["blend"] = max(cur["standalone"], e["mean"]), max(cur["blend"], blend)
    order_s = sorted(units.values(), key=lambda v: (-v["standalone"], v["unit"]["unit_id"]))
    order_b = sorted(units.values(), key=lambda v: (-v["blend"], v["unit"]["unit_id"]))
    chosen, per = [], {}

    def add(v):
        f = v["unit"]["family"]
        if any(c is v for c in chosen) or per.get(f, 0) >= r["max_per_family"] or len(chosen) >= r["max_units"]:
            return False
        chosen.append(v)
        per[f] = per.get(f, 0) + 1
        return True
    n = 0
    for v in order_s:
        if n == r["by_standalone"]:
            break
        n += add(v)
    for v in order_b:
        add(v)
    for v in order_s:
        add(v)
    return chosen


def retune_parents(protocol, unit, pool):
    """The unit's best standalone configuration, then its best contributor whose computation differs (family key)."""
    key = family_key(protocol, unit["family"])
    members = [e for e in pool if unit_of(e["recipe"])["unit_id"] == unit["unit_id"]]
    top = best(members)
    rest = sorted((e for e in members if key(e["recipe"]) != key(top["recipe"])),
                  key=lambda e: (-max(b["mean"] for b in e["blends"].values()), e["recipe_id"]))
    return [top] + rest[:1], {key(e["recipe"]) for e in members}


def plan_retune(protocol, unit, parents, exclude, batch_index, seed):
    """One re-tuning batch for a unit (plan section 18.6): P2.2's refine code with the unit's features fixed; local
    proposals around the parents and broad draws from the family's space, both keeping the unit's features (broad
    draws keep the first parent's parameters outside the space); deduplicated by the family-normalized key against
    every earlier configuration of the unit."""
    r = protocol["retune"]
    fam = r["families"][unit["family"]]
    name = f"p24_rt_{unit['unit_id'][:8]}_b{batch_index}"
    local = [r["local"]] if len(parents) == 1 else [r["local"] - r["local"] // 2, r["local"] // 2]
    space = fam["space"]
    held = {k: v for k, v in parents[0]["params"].items() if k not in space}  # e.g. kNN's metric, candidate 31's rsm
    broad = searches.apply_derived(dict(fam["fixed"], **held, **{n: v[0] for n, v in space.items()}), fam["derived"])
    spec = {"schema": refine.REFINE_SCHEMA, "name": name, "family": unit["family"], "parents": parents,
            "local_per_parent": local, "trees": fam["trees"], "broad": r["broad"],
            "broad_template": validate_recipe(dict(copy.deepcopy(parents[0]), name=f"{name}_template", params=broad)),
            "broad_space": space, "types": local_types(fam), "derived": fam["derived"],
            "admissible": fam["admissible"], "exclude_recipe_sha256": sorted(exclude), "sampler_seed": seed,
            "max_dimensions": 3}
    return refine.propose(spec, key=family_key(protocol, unit["family"]))


def local_types(fam):
    out = {}
    for n, t in fam["types"].items():
        if t["type"] == "log":
            nums = [v for v in fam["space"].get(n, []) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0]
            out[n] = dict(t, min=min(nums), max=max(nums))
        else:
            out[n] = dict(t)
    return out
