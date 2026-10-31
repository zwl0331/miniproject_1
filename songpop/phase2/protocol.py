"""The wave execution protocol of P2.2 and P2.3 (plan sections 15 and 17): families advance in waves separated by
barriers. Every manifest of wave N is a pure function of the protocol file and the records of the waves before N, so
verification re-plans each wave from the log and compares; each batch is judged by the allocation rule of plan
section 5.6 against the pre-batch state its manifest names. Nothing here fits a model or writes a file."""
import copy
import itertools

from songpop import artifacts
from songpop.phase2 import allocation, refine, searches
from songpop.phase2.models import INTEGER_PARAMS
from songpop.phase2.recipes import RECIPE_SCHEMA, iteration_policy, recipe_hash, recipe_id, strict_loads, validate_recipe
from songpop.splits import SPLIT_SEEDS

PROTOCOL_SCHEMA = "protocol_v1"
KEYS = ("schema", "task_id", "name", "reference", "reference_p21_run_id", "split_seeds", "seed_base", "cutoff_utc",
        "refinement", "families")
# prior_strongest (P2.3, plan section 17): a complete run of an earlier converged task that joins G, the strongest
# standalone recipe, in every wave: {"run_id", "task_id", "recipe", "bank_sha256"}. P2.2's protocol has none.
OPTIONAL_KEYS = ("prior_strongest",)
PRIOR_KEYS = ("bank_sha256", "recipe", "run_id", "task_id")
FAMILY_KEYS = ("family", "index", "initial", "expansion", "ceiling", "template", "anchor", "initial_branches",
               "space_template", "fixed", "space", "derived", "admissible", "types", "hard_bounds", "trees")
REFINEMENT_KEYS = ("families", "by_standalone", "batch", "broad", "local", "max_dimensions", "max_per_family")
N_ROWS = 3863  # the eligible training rows (Finding 001)


def _posint(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def load_protocol(path):
    """The protocol file, with its keys, the official split seeds, unique families in index order, positive counts
    with initial and expansion within the ceiling, hard bounds, types and rules checked."""
    p = strict_loads(path.read_text())
    if not isinstance(p, dict) or p.get("schema") != PROTOCOL_SCHEMA \
            or not set(KEYS) <= set(p) <= set(KEYS) | set(OPTIONAL_KEYS):
        raise ValueError(f"protocol keys {sorted(p) if isinstance(p, dict) else p!r}, expected {sorted(KEYS)} "
                         f"and optionally {list(OPTIONAL_KEYS)}")
    prior = p.get("prior_strongest")
    if prior is not None:
        if not isinstance(prior, dict) or sorted(prior) != sorted(PRIOR_KEYS) or prior["task_id"] == p["task_id"]:
            raise ValueError(f"malformed prior_strongest {prior!r}")
        validate_recipe(prior["recipe"])
    if p["split_seeds"] != list(SPLIT_SEEDS):
        raise ValueError(f"protocol split seeds {p['split_seeds']}, expected the official {list(SPLIT_SEEDS)}")
    r = p["refinement"]
    if sorted(r) != sorted(REFINEMENT_KEYS) or not all(_posint(r[k]) for k in REFINEMENT_KEYS if k != "local") \
            or any(r["broad"] + sum(split) != r["batch"] for split in r["local"]) \
            or r["max_per_family"] != r["batch"] * len(r["local"]) or r["by_standalone"] > r["families"]:
        raise ValueError(f"malformed refinement settings {r!r}")
    names = [f.get("family") for f in p["families"]]
    if len(set(names)) != len(names) or "lightgbm" not in names:
        raise ValueError("families must be distinct and include lightgbm, whose anchor is H")
    for i, f in enumerate(p["families"]):
        if sorted(f) != sorted(FAMILY_KEYS):
            raise ValueError(f"protocol family {f.get('family')!r} keys {sorted(f)}, expected {sorted(FAMILY_KEYS)}")
        if f["index"] != i or not all(_posint(f[k]) for k in ("initial", "expansion", "ceiling")) \
                or not f["initial"] <= f["ceiling"] or not f["expansion"] <= f["ceiling"]:
            raise ValueError(f"{f['family']}: index or counts malformed")
        searches.validate_rules(f["derived"], f["admissible"])
        refine.validate_types({n: dict(t, min=1, max=2) if t["type"] == "log" else t for n, t in f["types"].items()})
        for n, b in f["hard_bounds"].items():
            if sorted(b) != ["integer", "max", "min"] or not b["min"] <= b["max"] or not isinstance(b["integer"], bool):
                raise ValueError(f"{f['family']}: malformed hard bounds for {n}")
        if f["trees"] is not None and f["trees"] not in f["anchor"]:
            raise ValueError(f"{f['family']}: trees must name an iteration parameter the anchor sets")
    return p


def make_recipe(template, family, name, values):
    """A recipe from a template and flat values (model parameters, or PREP-prefixed preprocessing parameters)."""
    params = {k: v for k, v in values.items() if not k.startswith(searches.PREP)}
    prep = dict(template["preprocessing"], **{k[len(searches.PREP):]: v for k, v in values.items()
                                             if k.startswith(searches.PREP)})
    return validate_recipe({"schema": RECIPE_SCHEMA, "name": name, "family": family, "params": params,
                            "base": template["base"], "blocks": list(template["blocks"]), "drop": list(template["drop"]),
                            "preprocessing": prep, "training_weights": template["training_weights"],
                            "estimator_seeds": list(template["estimator_seeds"]), "iterations": iteration_policy(family),
                            "deployment": "fold_average"})


def search_key(fam):
    """The comparison key of plan section 15.4: two recipes with the same key are the same computation. Model
    parameters a recipe leaves out take the factory default the protocol declares; counts compare as integers
    and every other number as a float (C 100 and 100.0, reg_alpha 0 and 0.0). The name is ignored."""
    defaults = {n: t["default"] for n, t in fam["types"].items() if not n.startswith(searches.PREP)}

    def norm(k, v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return v
        return int(v) if k in INTEGER_PARAMS else float(v)

    def key(r):
        params = dict(defaults, **r["params"])
        body = {k: v for k, v in r.items() if k not in ("name", "params", "preprocessing")}
        body["params"] = {k: norm(k, v) for k, v in params.items()}
        body["preprocessing"] = {k: norm(k, v) for k, v in r["preprocessing"].items()}
        return artifacts.sha256_bytes(artifacts.dumps(body).encode())
    return key


def anchor_recipe(protocol, fam):
    return make_recipe(fam["template"], fam["family"], f"{protocol['name']}_{fam['family']}_anchor", fam["anchor"])


def sampler_seed(protocol, fam, batch_index):
    return protocol["seed_base"] + 100 * fam["index"] + batch_index


def _spec(protocol, fam, template, fixed, space, draws, seed, name):
    return {"schema": searches.SEARCH_SCHEMA_V2, "name": name, "family": fam["family"], "base": template["base"],
            "blocks": list(template["blocks"]), "drop": list(template["drop"]),
            "preprocessing": dict(template["preprocessing"]), "training_weights": template["training_weights"],
            "estimator_seeds": list(template["estimator_seeds"]), "fixed": dict(fixed), "space": space, "anchors": [],
            "draws": draws, "sampler_seed": seed, "derived": fam["derived"], "admissible": fam["admissible"]}


def initial_recipes(protocol, fam):
    """The family anchor, then each initial branch in order: seeded draws (search_v2), an explicit list, or a full
    grid in the lexicographic order of the sorted parameter names. Explicit and grid members obey the admissible
    regions like draws; a configuration equivalent to an earlier one (search_key) is not repeated."""
    key = search_key(fam)
    out = [anchor_recipe(protocol, fam)]
    seen = {key(out[0])}
    prefix = f"{protocol['name']}_{fam['family']}_b1"
    for j, br in enumerate(fam["initial_branches"]):
        if br["kind"] == "draws":
            spec = _spec(protocol, fam, br["template"], br["fixed"], br["space"], br["draws"],
                         sampler_seed(protocol, fam, 1) + 10 * j, f"{prefix}_{j}")
            got, _ = searches.expand_v2(spec, exclude=seen, key=key)
        elif br["kind"] in ("explicit", "grid"):
            if br["kind"] == "explicit":
                values = [dict(v) for v in br["values"]]
            else:
                names = sorted(br["space"])
                values = [dict(br["fixed"], **dict(zip(names, c)))
                          for c in itertools.product(*(br["space"][n] for n in names))]
            got = [make_recipe(br["template"], fam["family"], f"{prefix}_{j}_{i:03d}",
                               searches.apply_derived(v, fam["derived"])) for i, v in enumerate(values)]
            bad = [r["name"] for r in got if not searches.admissible(r["params"], fam["admissible"])]
            if bad:
                raise ValueError(f"{fam['family']}: initial configurations outside the admissible regions: {bad}")
        else:
            raise ValueError(f"unknown initial branch kind {br['kind']!r}")
        for r in got:
            if key(r) not in seen:
                seen.add(key(r))
                out.append(r)
    if len(out) != fam["initial"]:
        raise ValueError(f"{fam['family']}: {len(out)} initial configurations, the plan declares {fam['initial']}")
    return out


# ---- state from the log -------------------------------------------------------------------------------------

def complete(rec):
    """A run record the allocation rule may use: the eligible rows on all three official split seeds."""
    return rec.get("record") == "run" and rec["selection_eligible"] and rec["split_seeds"] == list(SPLIT_SEEDS) \
        and rec["n_rows"] == N_ROWS and len(rec["repeat_mean_auc_unrounded"]) == len(SPLIT_SEEDS)


def best(records):
    """The record with the highest unrounded mean fold AUC, ties to the lower canonical recipe hash."""
    recs = [r for r in records if complete(r)]
    return min(recs, key=lambda r: (-r["mean_fold_auc_unrounded"], r["recipe_sha256"])) if recs else None


def ref(rec):
    return {"run_id": rec["run_id"], "recipe_sha256": rec["recipe_sha256"], "bank_sha256": rec["bank_sha256"],
            "estimator_seeds": rec["estimator_seeds"], "mean_fold_auc_unrounded": rec["mean_fold_auc_unrounded"],
            "repeat_mean_auc_unrounded": rec["repeat_mean_auc_unrounded"]}


def blend_ref(c):
    return {"run_id": c["run_id"], "weight": c["weight"], "bank_sha256": c["bank_sha256"],
            "mean_fold_auc_unrounded": c["mean_fold_auc_unrounded"],
            "repeat_mean_auc_unrounded": c["repeat_mean_auc_unrounded"]}


def best_blend(contribs):
    """The highest contribution blend, by unrounded mean fold AUC; ties to the lower canonical recipe hash of its
    challenger (attached by Log), then the lower weight. H's identity records (H with itself) are not blends: H
    enters as the floor wherever a blend is compared."""
    contribs = [c for c in contribs if not c.get("identity")]
    if not contribs:
        return None
    return blend_ref(min(contribs, key=lambda c: (-c["mean_fold_auc_unrounded"], c["recipe_sha256"], c["weight"])))


class Log:
    """The task's records of the phase 2 log, indexed once and checked: one record per run id, batch id,
    allocation and contribution identity; every batch member's run record bound to its manifest recipe.
    read_manifest(path) returns a saved manifest."""

    def __init__(self, records, task_id, read_manifest, prior=None):
        """prior: the protocol's prior_strongest entry, or None."""
        self.task_id, self.read_manifest = task_id, read_manifest

        def unique(kind, ident):
            out = {}
            for r in records:
                if r.get("record") == kind and (kind == "run" or r.get("task_id") == task_id):
                    k = ident(r)
                    if k in out:
                        raise ValueError(f"duplicate {kind} record {k}")
                    out[k] = r
            return out
        self.runs = unique("run", lambda r: r["run_id"])
        self.batches = list(unique("batch", lambda r: r["batch_id"]).values())
        self.allocations = unique("allocation", lambda r: r["batch_id"])
        self.starts = unique("batch_start", lambda r: r["batch_id"])
        self.contributions = list(unique("contribution", lambda r: (r["run_id"], r["h_run_id"], r["weight"])).values())
        self.h_checks = list(unique("h_check", lambda r: r["h_run_id"]).values())
        self.bootstraps = unique("bootstrap", lambda r: r["comparison_id"])
        self.recipes = {}
        for b in self.batches:
            m = read_manifest(b["file"])
            for rid, r in zip(b["run_ids"], m["recipes"]):
                if rid in self.recipes and recipe_hash(self.recipes[rid]) != recipe_hash(r):
                    raise ValueError(f"run id {rid} is bound to two different recipes")
                self.recipes[rid] = r
        for rid, r in self.recipes.items():  # bind run records to their recipes; add the canonical hash for tie-breaks
            if rid in self.runs:
                if self.runs[rid]["recipe_id"] != recipe_id(r):
                    raise ValueError(f"run {rid} records recipe {self.runs[rid]['recipe_id']}, its manifest {recipe_id(r)}")
                self.runs[rid] = dict(self.runs[rid], recipe_sha256=recipe_hash(r))
        self.contributions = [dict(c, recipe_sha256=recipe_hash(self.recipes[c["run_id"]])) if c["run_id"] in self.recipes
                              else c for c in self.contributions]
        self.prior = None
        if prior is not None:  # bound to its logged record: task, recipe, bank and completeness must all agree
            rec = self.runs.get(prior["run_id"])
            if rec is None or rec["task_id"] != prior["task_id"] or rec["recipe_id"] != recipe_id(prior["recipe"]) \
                    or rec["bank_sha256"] != prior["bank_sha256"] or not complete(rec):
                raise ValueError(f"prior_strongest {prior['run_id']} does not match a complete logged run")
            self.runs[prior["run_id"]] = dict(rec, recipe_sha256=recipe_hash(prior["recipe"]))
            self.prior = prior["run_id"]

    def manifest_recipe(self, run_id):
        return self.recipes[run_id]

    def batches_before(self, wave):
        return [b for b in self.batches if b["wave"] < wave]

    def family_runs(self, family, wave):
        """Run records of the family's batches before the wave (each run once)."""
        ids = []
        for b in self.batches_before(wave):
            if b["family"] == family:
                ids += [i for i in b["run_ids"] if i in self.runs and i not in ids]
        return [self.runs[i] for i in ids]

    def contribution_pair(self, run_id, h_id):
        """The run's two contribution records with H, one per declared weight, bound to both banks' hashes."""
        cs = {c["weight"]: c for c in self.contributions if c["run_id"] == run_id and c["h_run_id"] == h_id}
        if sorted(cs) != sorted(allocation.CONTRIBUTION_WEIGHTS):
            raise ValueError(f"run {run_id} has contribution weights {sorted(cs)}, expected "
                             f"{sorted(allocation.CONTRIBUTION_WEIGHTS)}")
        for c in cs.values():
            if c["bank_sha256"] != self.runs[run_id]["bank_sha256"] or c["h_sha256"] != self.runs[h_id]["bank_sha256"]:
                raise ValueError(f"contribution of {run_id} is not bound to the logged banks")
        return [cs[w] for w in allocation.CONTRIBUTION_WEIGHTS]

    def family_contributions(self, family, wave, h_id):
        return [c for r in self.family_runs(family, wave) for c in self.contribution_pair(r["run_id"], h_id)]


def h_run_id(protocol, log):
    """H: the LightGBM family anchor's run, named in the wave 1 LightGBM manifest."""
    for b in log.batches:
        if b["wave"] == 1 and b["family"] == "lightgbm":
            return b["state"]["family_anchor"]["run_id"]
    raise ValueError("no wave 1 LightGBM manifest is logged")


def allowance_after(log, family, wave):
    """The family's counter after its last judged batch before the wave; None before any judgment."""
    judged = [log.allocations[b["batch_id"]] for b in log.batches_before(wave) if b["family"] == family
              and b["batch_id"] in log.allocations]
    return judged[-1]["allowance_after"] if judged else None


def strongest(log, wave, h):
    """G: the strongest complete run of the closed waves before this one, H and the protocol's prior run (if any)
    included. Results inside a wave never update its comparators."""
    return best([log.runs[h]] + ([log.runs[log.prior]] if log.prior else [])
                + [r for b in log.batches_before(wave) for r in log.family_runs(b["family"], wave)])


def qualifying(log, family, wave):
    """Run ids of the family that met a route of plan section 5.6 in a judged batch before the wave."""
    return {rid for b in log.batches_before(wave) if b["family"] == family and b["batch_id"] in log.allocations
            for rid, met in log.allocations[b["batch_id"]]["routes"].items() if met}


def refinement_families(protocol, log, families):
    """At the barrier after wave 1, at most four families with allowance: the two strongest by standalone
    unrounded mean fold AUC, then the others by their best contribution to H; ties by the canonical recipe hash,
    then the family name."""
    h = h_run_id(protocol, log)
    rows = []
    for fam in families:
        if not allowance_after(log, fam["family"], 2):
            continue
        inc = best(log.family_runs(fam["family"], 2))
        blend = best_blend(log.family_contributions(fam["family"], 2, h))
        rows.append((fam["family"], inc, blend))
    k, s = protocol["refinement"]["families"], protocol["refinement"]["by_standalone"]
    by_standalone = sorted(rows, key=lambda r: (-r[1]["mean_fold_auc_unrounded"], r[1]["recipe_sha256"], r[0]))
    chosen = [r[0] for r in by_standalone[:s]]
    rest = sorted([r for r in rows if r[0] not in chosen],
                  key=lambda r: (-(r[2]["mean_fold_auc_unrounded"] if r[2] else 0.0),
                                 log.runs[r[2]["run_id"]]["recipe_sha256"] if r[2] else "", r[0]))
    return chosen + [r[0] for r in rest[:k - len(chosen)]]


def refinement_parents(log, family, wave, inc, h):
    """The standalone parent (the incumbent) and the contribution parent: the qualifying run (one that met a route)
    with the best contribution blend with H, other than the incumbent; the incumbent alone if none exists."""
    q = qualifying(log, family, wave) - {inc["run_id"]}
    cands = [c for c in log.family_contributions(family, wave, h) if c["run_id"] in q]
    if not cands:
        return [inc]
    return [inc, log.runs[best_blend(cands)["run_id"]]]


def active_space(fam, extensions):
    """The family's expansion space with every range extension so far appended."""
    return {n: list(v) + [e["value"] for e in extensions if e["parameter"] == n] for n, v in fam["space"].items()}


def new_extensions(fam, space, incumbent, extensions):
    """Range extension (plan section 5.2): for each ordered numeric parameter of the active space (an all-numeric
    list) whose incumbent value is an end of the list and which has not been extended in P2.2: one value one
    geometric step beyond that end, clipped to its hard bounds, rounded for integers; none at a hard bound or
    below a zero end."""
    done = {e["parameter"] for e in extensions}
    held = dict(incumbent["params"], **{searches.PREP + k: v for k, v in incumbent["preprocessing"].items()})
    out = []
    for n in sorted(space):
        values, bound = space[n], fam["hard_bounds"].get(n)
        if n in done or bound is None or len(values) < 2 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            continue
        v, ordered = held.get(n), sorted(values)
        if v == ordered[-1] and ordered[-2] > 0:
            new = ordered[-1] ** 2 / ordered[-2]
        elif v == ordered[0] and ordered[0] > 0:
            new = ordered[0] ** 2 / ordered[1]
        else:
            continue
        new = min(max(new, bound["min"]), bound["max"])  # clipped to the hard bounds
        new = int(new + 0.5) if bound["integer"] else float(f"{new:.3g}")
        if new not in values and (new > ordered[-1] or new < ordered[0]):
            out.append({"parameter": n, "edge": v, "value": new, "hard_bounds": bound})
    return out


def local_types(fam, space):
    """Refinement types with the active bounds of the space (a local move never extends a range)."""
    out = {}
    for n, t in fam["types"].items():
        if t["type"] == "log":
            nums = [v for v in space.get(n, []) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0]
            if not nums:
                raise ValueError(f"{fam['family']}: no positive active values for {n}")
            out[n] = dict(t, min=min(nums), max=max(nums))
        else:
            out[n] = dict(t)
    return out


def plan_wave(protocol, log, wave, run_id_of):
    """The manifests of a wave, in family order. Wave 1: every family's initial batch, its state naming the runs
    that will supply it (the family anchor as incumbent, H as reference, and as strongest H or, when the protocol
    names a prior run, the better of H and that run; the previous best blend is the better of the anchor's two blends
    with H, floored at H). Later waves: for each family with allowance below
    its ceiling, a local refinement batch (a chosen family's first two batches after wave 1) or a broad expansion
    batch, with the full pre-batch state. run_id_of(recipe) gives the run id the recipe will have."""
    fams = protocol["families"]
    out = []
    if wave == 1:
        lgb = next(f for f in fams if f["family"] == "lightgbm")
        h = run_id_of(anchor_recipe(protocol, lgb))
        g = {"run_id": h} if log.prior is None else {"run_id": h, "prior_run_id": log.prior}
        for fam in fams:
            recipes = initial_recipes(protocol, fam)
            a = run_id_of(recipes[0])
            out.append({"family": fam["family"], "wave": 1, "kind": "initial", "recipes": recipes,
                        "run_ids": [run_id_of(r) for r in recipes],
                        "state": {"family_anchor": {"run_id": a}, "incumbent": {"run_id": a},
                                  "strongest": dict(g), "reference_h": {"run_id": h},
                                  "previous_best_blend": {"of_run_id": a, "floor_run_id": h},
                                  "allowance_before": None},
                        "sources": {"sampler_seed": sampler_seed(protocol, fam, 1), "extensions": []}})
        return out
    if any(b["batch_id"] not in log.allocations for b in log.batches_before(wave)):
        raise ValueError(f"wave {wave} needs every earlier batch judged")
    h = h_run_id(protocol, log)
    g = strongest(log, wave, h)
    chosen = refinement_families(protocol, log, fams)
    rp = protocol["refinement"]
    for fam in fams:
        f, key = fam["family"], search_key(fam)
        allowance = allowance_after(log, f, wave)
        runs = log.family_runs(f, wave)
        prior = [b for b in log.batches_before(wave) if b["family"] == f]
        used = sum(len(b["run_ids"]) for b in prior)
        if not allowance or used >= fam["ceiling"]:
            continue
        inc = best(runs)
        inc_recipe = log.manifest_recipe(inc["run_id"])
        extensions = [e for b in prior for e in b["sources"].get("extensions", [])]
        space = active_space(fam, extensions)
        exclude = sorted({key(log.manifest_recipe(r)) for b in prior for r in b["run_ids"]})
        h_rec = log.runs[h]
        prev = best_blend(log.family_contributions(f, wave, h))
        floor = {"run_id": h, "weight": 0.0, "bank_sha256": h_rec["bank_sha256"],
                 "mean_fold_auc_unrounded": h_rec["mean_fold_auc_unrounded"],
                 "repeat_mean_auc_unrounded": h_rec["repeat_mean_auc_unrounded"]}
        if prev is None or prev["mean_fold_auc_unrounded"] <= floor["mean_fold_auc_unrounded"]:
            prev = floor
        local_done = sum(1 for b in prior if b["kind"] == "refinement")
        batch_index = len(prior) + 1
        seed = sampler_seed(protocol, fam, batch_index)
        state = {"family_anchor": prior[0]["state"]["family_anchor"], "incumbent": ref(inc), "strongest": ref(g),
                 "reference_h": ref(h_rec), "previous_best_blend": prev, "allowance_before": allowance}
        name = f"{protocol['name']}_{f}_b{batch_index}"
        sources = {"sampler_seed": seed, "exclude_digest": artifacts.sha256_bytes(artifacts.dumps(exclude).encode())}
        if f in chosen and local_done < len(rp["local"]):
            size = min(rp["batch"], fam["ceiling"] - used, rp["max_per_family"] - rp["batch"] * local_done)
            parents = refinement_parents(log, f, wave, inc, h)
            split = rp["local"][local_done]
            n_local = max(size - rp["broad"], 0)
            first = (n_local * split[0] + sum(split) // 2) // sum(split)
            local = [n_local] if len(parents) == 1 else [first, n_local - first]
            spec = {"schema": refine.REFINE_SCHEMA, "name": name, "family": f,
                    "parents": [log.manifest_recipe(p["run_id"]) for p in parents], "local_per_parent": local,
                    "trees": fam["trees"], "broad": size - sum(local),
                    "broad_template": make_recipe(fam["space_template"], f, f"{name}_template",
                                                  searches.apply_derived(dict(fam["fixed"], **{
                                                      n: v[0] for n, v in space.items()}), fam["derived"])),
                    "broad_space": space, "types": local_types(fam, space), "derived": fam["derived"],
                    "admissible": fam["admissible"], "exclude_recipe_sha256": exclude, "sampler_seed": seed,
                    "max_dimensions": rp["max_dimensions"]}
            recipes, rejected = refine.propose(spec, key=key)
            sources.update({"extensions": [], "parents": [p["run_id"] for p in parents], "local_per_parent": local,
                            "broad": size - sum(local), "rejected": rejected})
            kind = "refinement"
        else:
            ext = new_extensions(fam, space, inc_recipe, extensions)
            space = active_space(fam, extensions + ext)
            size = min(fam["expansion"], fam["ceiling"] - used)
            spec = _spec(protocol, fam, fam["space_template"], fam["fixed"], space, size, seed, name)
            recipes, info = searches.expand_v2(spec, exclude=exclude, allow_fewer=True, key=key)
            sources.update({"extensions": ext, "rejected": info})
            kind = "expansion"
        if recipes:
            out.append({"family": f, "wave": wave, "kind": kind, "recipes": recipes,
                        "run_ids": [run_id_of(r) for r in recipes], "state": state, "sources": sources})
    return out


def resolved_state(state, log):
    """A wave 1 state with its values filled from the logged runs and contributions it names; later states are
    complete already."""
    st = copy.deepcopy(state)
    for k in ("family_anchor", "incumbent", "strongest", "reference_h"):
        if "prior_run_id" in st[k]:  # wave 1 with a prior run: G is the better of H and that run
            st[k] = ref(best([log.runs[st[k]["run_id"]], log.runs[st[k]["prior_run_id"]]]))
        elif "repeat_mean_auc_unrounded" not in st[k]:
            st[k] = ref(log.runs[st[k]["run_id"]])
    pb = st["previous_best_blend"]
    if "of_run_id" in pb:
        h = log.runs[pb["floor_run_id"]]
        floor = {"run_id": h["run_id"], "weight": 0.0, "bank_sha256": h["bank_sha256"],
                 "mean_fold_auc_unrounded": h["mean_fold_auc_unrounded"],
                 "repeat_mean_auc_unrounded": h["repeat_mean_auc_unrounded"]}
        mine = best_blend(log.contribution_pair(pb["of_run_id"], pb["floor_run_id"]))
        st["previous_best_blend"] = mine if mine and mine["mean_fold_auc_unrounded"] > floor["mean_fold_auc_unrounded"] \
            else floor
    return st


def judge(batch, log):
    """The allocation of one logged batch (plan section 5.6): for each new configuration, the routes it meets
    against the pre-batch state; the progress flag; the family counter after the batch. Every member must be
    logged and complete."""
    st = resolved_state(batch["state"], log)
    anchor_id, h_id = st["family_anchor"]["run_id"], st["reference_h"]["run_id"]
    out = {}
    for rid in batch["run_ids"]:
        rec = log.runs.get(rid)
        if rec is None or not complete(rec):
            raise ValueError(f"run {rid} of batch {batch['batch_id']} is not logged as a complete run")
        if batch["kind"] == "initial" and rid == anchor_id:
            continue  # the anchor is the comparator of its own initial batch, not a new configuration
        blends = [c["repeat_mean_auc_unrounded"] for c in log.contribution_pair(rid, h_id)]
        out[rid] = allocation.routes({"standalone": rec["repeat_mean_auc_unrounded"], "blends": blends},
                                     st["incumbent"]["repeat_mean_auc_unrounded"],
                                     st["strongest"]["repeat_mean_auc_unrounded"],
                                     st["reference_h"]["repeat_mean_auc_unrounded"],
                                     st["previous_best_blend"]["repeat_mean_auc_unrounded"])
    progress = any(out.values())
    after = allocation.next_allowance(batch["state"]["allowance_before"], progress, batch["kind"] == "initial")
    return {"routes": out, "progress": progress, "allowance_after": after, "resolved_state": st}


STANDALONE_ROUTES = ("clear_standalone", "smaller_useful", "competitive_refinement")


def witnesses(batch, res, log):
    """The comparisons behind a batch's allocation (plan section 15.5): for each route met, the qualifying
    configuration with the highest relevant unrounded score (standalone for the standalone routes, its better
    blend with H for the ensemble route), ties to the lower canonical recipe hash. A standalone witness is compared
    with the pre-batch incumbent; an ensemble witness's blend with H and with the previous best blend. Each entry:
    (route, a, b), where a and b are ("run", run_id) or ("blend", run_id, weight)."""
    st, h = res["resolved_state"], res["resolved_state"]["reference_h"]["run_id"]
    out = []
    for route in STANDALONE_ROUTES:
        met = [rid for rid, rs in res["routes"].items() if route in rs]
        if met:
            w = min(met, key=lambda r: (-log.runs[r]["mean_fold_auc_unrounded"], log.runs[r]["recipe_sha256"]))
            out.append((route, ("run", w), ("run", st["incumbent"]["run_id"])))
    met = [rid for rid, rs in res["routes"].items() if "ensemble_contribution" in rs]
    if met:
        best_c = best_blend([c for r in met for c in log.contribution_pair(r, h)])
        a = ("blend", best_c["run_id"], best_c["weight"])
        pb = st["previous_best_blend"]
        out.append(("ensemble_contribution", a, ("run", h)))
        if pb["weight"] != 0.0:
            out.append(("ensemble_contribution", a, ("blend", pb["run_id"], pb["weight"])))
    seen, unique = set(), []
    for w in out:
        k = artifacts.dumps(w[1:])
        if k not in seen:
            seen.add(k)
            unique.append(w)
    return unique
