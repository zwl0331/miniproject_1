"""P2.6b pool and procedure manifest (plan section 20.3), read-only: from the nominees' final-policy runs, H*, the
contributions against it, the pool of at most eight atomic recipes, the channels, and the enumerated procedures with
their structural counts and complexity ranks. Fits nothing and scores no procedure; writes
configs/phase2/p2.6/procedures.json. From the project root, after the final-policy runs:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p26_pool.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "analysis")
from p26_nominate import ANCHOR_RUN, WEIGHTS, atom_id  # noqa: E402

from songpop import artifacts  # noqa: E402
from songpop.phase2 import protocol, runs  # noqa: E402
from songpop.phase2.ensembles import blend_banks  # noqa: E402
from songpop.phase2.recipes import DATE_TE_BLOCKS, recipe_hash, recipe_id  # noqa: E402

OUT = Path("configs/phase2/p2.6")
LEAD_BAND = 0.002
MAX_ATOMS = 8
MAX_PROCEDURES = 32
LAMBDAS = (0.1, 0.01, 0.0)  # stronger regularization first: the complexity-rank order within a learned kind
CS = (0.01, 0.1, 1.0)
KIND_ORDER = {"atomic": 0, "blend": 1, "constrained": 2, "stacker": 3}


def final_runs(ctx, nom):
    """Each nominee's final-policy run: the existing run when its policy changed nothing, else the P2.6b run of its
    recipe file; each audited against its bank and complete."""
    by_recipe = {}
    for r in runs.logged(ctx):
        if r["task_id"] == "P2.6b" and r["prediction_kind"] in ("fitted", "seed_average"):
            by_recipe.setdefault(r["recipe_id"], []).append(r)
    out = []
    for n in nom["nominees"]:
        rid = n["run_existing"]
        if rid is None:
            got = by_recipe.get(recipe_id(n["recipe"]), [])
            if len(got) != 1:
                raise ValueError(f"{n['recipe']['name']}: {len(got)} final-policy runs, expected 1")
            rid = got[0]["run_id"]
        rec = next(r for r in runs.logged(ctx) if r["run_id"] == rid)
        if runs.audit_run(ctx, rid) or not protocol.complete(rec):
            raise ValueError(f"{rid} is not a complete run that audits against its bank")
        bank, sha, _ = runs.load_logged(ctx, rid)
        if recipe_hash(bank.metadata["recipe"]) != recipe_hash(n["recipe"]) and rid != n["run_existing"]:
            raise ValueError(f"{rid} is not the nominee's recipe")
        out.append({"name": n["recipe"]["name"], "run_id": rid, "family": n["family"], "roles": n["roles"],
                    "atom_id": atom_id(bank.metadata["recipe"]), "recipe": bank.metadata["recipe"],
                    "recipe_id": recipe_id(bank.metadata["recipe"]), "bank_sha256": sha,
                    "mean": rec["mean_fold_auc_unrounded"], "bank": bank})
    return out


def supervised(recipes):
    """Distinct label-fitted feature maps reachable in a procedure: each date target encoding counts once however many
    atoms share it (same input, same cross-fitting policy); counts, imputation, scaling, splines, fixed interactions,
    era-relative columns and estimator training count zero (round U, item 14)."""
    return len({b for r in recipes for b in r["blocks"] if b in DATE_TE_BLOCKS})


def build(fin):
    rank = lambda e: (-e["mean"], e["recipe_id"])  # noqa: E731
    hstar = min(fin, key=rank)
    contrib = {}
    for e in fin:
        if e["run_id"] == hstar["run_id"]:
            continue
        for w in WEIGHTS:
            b = blend_banks([(hstar["bank"], hstar["bank_sha256"]), (e["bank"], e["bank_sha256"])], [1 - w, w])
            contrib[(e["run_id"], w)] = float(b.fold_auc.mean())
    fams = sorted({e["family"] for e in fin})
    rep = {f: min([e for e in fin if e["family"] == f and "anchor" not in e["roles"]
                   and "anchor_three_seeds" not in e["roles"]] or [e for e in fin if e["family"] == f], key=rank)
           for f in fams}
    cwin = {}
    for f in fams:
        c = [(e, w) for e in fin if e["family"] == f and e["run_id"] != hstar["run_id"] for w in WEIGHTS]
        if c:
            cwin[f] = min(c, key=lambda t: (-contrib[(t[0]["run_id"], t[1])], t[0]["recipe_id"], t[1]))
    pool, atoms, why = [], set(), {}

    def add(e, reason):
        if e["atom_id"] in atoms or len(atoms) >= MAX_ATOMS:
            return False
        pool.append(e)
        atoms.add(e["atom_id"])
        why[e["run_id"]] = reason
        return True
    for e in sorted(rep.values(), key=rank)[:3]:
        add(e, "standalone representative")
    extra = sorted([t for f, t in cwin.items() if t[0]["atom_id"] not in atoms],
                   key=lambda t: (-contrib[(t[0]["run_id"], t[1])], t[0]["recipe_id"], t[1]))
    n = 0
    for e, w in extra:
        if n < 2 and add(e, f"contribution to H* at weight {w}"):
            n += 1
    anchor = next(e for e in fin if "anchor" in e["roles"])
    add(anchor, "anchor")
    lead = sorted([e for e in fin if e["family"] == hstar["family"] and e["atom_id"] != hstar["atom_id"]
                   and "anchor" not in e["roles"] and "anchor_three_seeds" not in e["roles"]
                   and hstar["mean"] - e["mean"] <= LEAD_BAND], key=rank)
    family_avg = [hstar]
    for e in lead:
        if len(family_avg) < 3 and (e["atom_id"] in atoms or add(e, "leading-family configuration")):
            family_avg.append(e)
    anchor_s3 = next(e for e in fin if "anchor_three_seeds" in e["roles"])
    return hstar, contrib, rep, cwin, pool, why, family_avg, anchor, anchor_s3


def channels_of(pool, rep, cwin, hstar, family_avg, anchor):
    """One channel per represented family: its contribution representative when that is in the pool, else its
    standalone representative; the within-family average replaces the leading family's channel; the anchor is a
    standalone candidate, not a channel. Ordered by the representative's standalone mean, then id: fixed before any
    procedure is scored."""
    in_pool = {e["run_id"] for e in pool}
    chans = []
    for f in sorted({e["family"] for e in pool if e["run_id"] != anchor["run_id"]}):
        if f == hstar["family"] and len(family_avg) > 1:
            members = family_avg
        else:
            c = cwin.get(f)
            pick = c[0] if c and c[0]["run_id"] in in_pool else rep[f]
            if pick["run_id"] not in in_pool:
                pick = next(e for e in pool if e["family"] == f and e["run_id"] != anchor["run_id"])
            members = [pick]
        chans.append({"id": f"ch_{f}", "family": f, "run_ids": [m["run_id"] for m in members],
                      "atoms": [m["recipe"] for m in members], "rank_mean": max(m["mean"] for m in members),
                      "rank_id": members[0]["recipe_id"]})
    chans.sort(key=lambda c: (-c["rank_mean"], c["rank_id"]))
    return chans


def procedures(pool, chans, hstar, family_avg, anchor, anchor_s3):
    procs = []

    def counts(recipes, learned):
        return {"atomic_recipes": len({atom_id(r) for r in recipes}), "supervised_transformations": supervised(recipes),
                "learned_combiners": int(learned)}
    for e in [hstar] + [p for p in pool if p["run_id"] != hstar["run_id"]] + [anchor_s3]:
        procs.append({"id": f"atom_{e['name']}", "kind": "atomic", "run_id": e["run_id"], "recipe": e["recipe"],
                      **counts([e["recipe"]], False)})

    def blend(pid, parts):  # parts: [(channel, weight)] expanded to atom runs with equal weights inside a channel
        members = {}
        for ch, w in parts:
            for rid, r in zip(ch["run_ids"], ch["atoms"]):
                members[rid] = members.get(rid, 0.0) + w / len(ch["run_ids"])
        recipes = [r for ch, _ in parts for r in ch["atoms"]]
        procs.append({"id": pid, "kind": "blend", "members": [{"run_id": k, "weight": v} for k, v in members.items()],
                      "channels": [{"id": ch["id"], "weight": w} for ch, w in parts], **counts(recipes, False)})
    if len(chans) >= 2:
        blend("blend_eq2", [(c, 0.5) for c in chans[:2]])
    if len(chans) >= 3:
        blend("blend_eq3", [(c, 1 / 3) for c in chans[:3]])
    if len(chans) >= 4:
        blend("blend_eq_all", [(c, 1 / len(chans)) for c in chans])
    hch = {"id": "H*", "run_ids": [hstar["run_id"]], "atoms": [hstar["recipe"]]}
    for e in pool:
        if e["run_id"] == hstar["run_id"]:
            continue
        ch = {"id": e["name"], "run_ids": [e["run_id"]], "atoms": [e["recipe"]]}
        for w in (0.1, 0.25):
            blend(f"blend_h_{e['name']}_{int(w * 100):02d}", [(hch, 1 - w), (ch, w)])
    if len(family_avg) > 1:
        fa = next(c for c in chans if c["family"] == hstar["family"])
        procs.append({"id": "family_average", "kind": "blend",
                      "members": [{"run_id": r, "weight": 1 / len(fa["run_ids"])} for r in fa["run_ids"]],
                      "channels": [{"id": fa["id"], "weight": 1.0}], **counts(fa["atoms"], False)})
    recipes = [r for c in chans for r in c["atoms"]]
    for lam in LAMBDAS:
        procs.append({"id": f"constrained_l{lam:g}", "kind": "learned", "setting": ["constrained", lam],
                      **counts(recipes, True)})
    for C in CS:
        procs.append({"id": f"stacker_c{C:g}", "kind": "learned", "setting": ["stacker", C], **counts(recipes, True)})
    order = lambda p: (KIND_ORDER["atomic" if p["kind"] == "atomic" else "blend" if p["kind"] == "blend"  # noqa: E731
                                  else p["setting"][0]], p["atomic_recipes"], p["supervised_transformations"],
                       ((LAMBDAS if p["setting"][0] == "constrained" else CS).index(p["setting"][1])
                        if p["kind"] == "learned" else 0), p["id"])
    for i, p in enumerate(sorted(procs, key=order)):
        p["complexity_rank"] = i
    n_proc = sum(1 for p in procs if p["kind"] != "atomic")
    if n_proc > MAX_PROCEDURES:
        raise ValueError(f"{n_proc} procedures, at most {MAX_PROCEDURES}")
    return procs


def main():
    ctx = runs.Context.load(Path("."))
    raw = (OUT / "nominations.json").read_bytes()
    nom = json.loads(raw)
    fin = final_runs(ctx, nom)
    hstar, contrib, rep, cwin, pool, why, family_avg, anchor, anchor_s3 = build(fin)
    chans = channels_of(pool, rep, cwin, hstar, family_avg, anchor)
    procs = procedures(pool, chans, hstar, family_avg, anchor, anchor_s3)
    doc = {"schema": "p26_procedures_v1", "nominations_sha256": artifacts.sha256_bytes(raw),
           "final_runs": [{k: e[k] for k in ("name", "run_id", "family", "roles", "atom_id", "bank_sha256", "mean")}
                          for e in fin],
           "h_star": {k: hstar[k] for k in ("name", "run_id", "family", "atom_id", "mean")},
           "contributions": [{"run_id": k[0], "weight": k[1], "mean": v} for k, v in sorted(contrib.items())],
           "pool": [{"name": e["name"], "run_id": e["run_id"], "family": e["family"], "atom_id": e["atom_id"],
                     "reason": why[e["run_id"]]} for e in pool],
           "anchor_three_seeds": anchor_s3["run_id"],
           "channels": [{k: c[k] for k in ("id", "family", "run_ids", "atoms")} for c in chans],
           "meta_protocol": {"folds": 5, "seed": "coordinate_seed('meta-v1', 'production', split seed, fold)",
                             "clip": 1e-6, "slsqp": {"ftol": 1e-12, "maxiter": 2000}, "weight_cleanup": 1e-9,
                             "stacker": {"solver": "lbfgs", "tol": 1e-10, "max_iter": 5000}},
           "procedures": procs}
    (OUT / "procedures.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"H*: {hstar['run_id']} ({hstar['family']}) {hstar['mean']:.6f}")
    print(f"pool ({len(pool)} atoms):")
    for e in pool:
        print(f"  {e['name']:30s} {e['family']:13s} {e['mean']:.6f}  {why[e['run_id']]}")
    print("channels:", [(c["id"], len(c["run_ids"])) for c in chans])
    print(f"{len(procs)} candidates ({sum(p['kind'] != 'atomic' for p in procs)} procedures):")
    for p in sorted(procs, key=lambda p: p["complexity_rank"]):
        print(f"  {p['complexity_rank']:2d} {p['id']:44s} {p['kind']:8s} atoms {p['atomic_recipes']} "
              f"supervised {p['supervised_transformations']} learned {p['learned_combiners']}")


if __name__ == "__main__":
    main()
