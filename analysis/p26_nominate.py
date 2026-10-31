"""P2.6b nominations (plan section 20.2), read-only: from the frozen discovery inventory, the nominees whose final
seed policies are then run. Fits nothing; writes configs/phase2/p2.6/nominations.json and one recipe file per
nominee under configs/phase2/p2.6/recipes/. From the project root, after P2.4 and P2.5 have run:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p26_nominate.py

The discovery inventory: every complete atomic run (fitted or seed-averaged, the 3,863 eligible rows, all three
split seeds) of P2.1, P2.2, P2.8, P2.3, P2.4 and P2.5, each validated against its bank, one per canonical recipe
(the earliest task, then the lowest run id). The discovery reference R0: the strongest by unrounded mean fold AUC,
ties to the lower recipe id. Contributions: the blends 0.9 R0 + 0.1 x and 0.75 R0 + 0.25 x on identical folds, for
every other inventory run against this one R0 (never against earlier references). Nominees: R0; each family's
standalone winner and contribution winner (best blend AUC, ties to the lower recipe id, then the lower weight); up
to two further configurations of R0's family within 0.002 of R0, distinct atoms, by mean AUC then recipe id; the
anchor (the phase 1 recipe, estimator seed 0) and its three-seed version. Final seed policies: estimator seeds 0, 1
and 2 for every family except kNN (deterministic, one fit) and TabM (seed 0, declared from its cost in P2.5); the
anchor keeps seed 0 and its three-seed version is a separate nominee. Every nominee, the seed-0 anchor and the
single-seed families included, is run again under the current sources (P2.6b), so the menu is regenerated under one
final implementation revision (plan section 11; R2 review round 1, F3); no historical bank enters the pool."""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

from songpop import artifacts  # noqa: E402
from songpop.phase2 import protocol, runs  # noqa: E402
from songpop.phase2.ensembles import blend_banks  # noqa: E402
from songpop.phase2.recipes import recipe_hash, recipe_id, validate_recipe  # noqa: E402

TASKS = ("P2.1", "P2.2", "P2.8", "P2.3", "P2.4", "P2.5")
ANCHOR_RUN = "p2_lightgbm_F2_anchor_seeds155-2026-1029_e00287a531af"
WEIGHTS = (0.1, 0.25)
LEAD_BAND = 0.002
SINGLE_SEED = ("knn", "tabm")
OUT = Path("configs/phase2/p2.6")


def atom_id(recipe):
    """The atomic recipe: its exact features, preprocessing, weighting, model and training configuration, without
    the name, the estimator seeds or the deployment wrapper (round U, item 14)."""
    validate_recipe(recipe)
    body = {k: v for k, v in recipe.items() if k not in ("name", "estimator_seeds", "deployment")}
    return artifacts.sha256_bytes(artifacts.dumps(body).encode())[:16]


def final_policy(recipe, name):
    seeds = [0] if recipe["family"] in SINGLE_SEED else [0, 1, 2]
    return validate_recipe(dict(recipe, name=name, estimator_seeds=seeds))


def inventory(ctx):
    order = {t: i for i, t in enumerate(TASKS)}
    recs = [r for r in runs.logged(ctx) if r["task_id"] in TASKS and protocol.complete(r)
            and r["prediction_kind"] in ("fitted", "seed_average")]
    recs.sort(key=lambda r: (order[r["task_id"]], r["run_id"]))
    out, seen = [], set()
    for r in recs:
        problems = runs.audit_run(ctx, r["run_id"])
        if problems:
            raise ValueError(f"{r['run_id']} does not audit against its bank: {problems}")
        bank, sha, _ = runs.load_logged(ctx, r["run_id"])
        rh = recipe_hash(bank.metadata["recipe"])
        if rh in seen:
            continue
        seen.add(rh)
        out.append({"run_id": r["run_id"], "task_id": r["task_id"], "family": r["family"], "bank_sha256": sha,
                    "recipe": bank.metadata["recipe"], "recipe_id": recipe_id(bank.metadata["recipe"]),
                    "recipe_sha256": rh, "atom_id": atom_id(bank.metadata["recipe"]),
                    "mean": r["mean_fold_auc_unrounded"], "repeats": r["repeat_mean_auc_unrounded"], "bank": bank})
    return out


def contributions(ref, inv):
    out = {}
    for e in inv:
        if e["run_id"] == ref["run_id"]:
            continue
        for w in WEIGHTS:
            b = blend_banks([(ref["bank"], ref["bank_sha256"]), (e["bank"], e["bank_sha256"])], [1 - w, w])
            out[(e["run_id"], w)] = {"mean": float(b.fold_auc.mean()), "repeats": b.fold_auc.mean(axis=1).tolist()}
    return out


def nominate(inv):
    ref = min(inv, key=lambda e: (-e["mean"], e["recipe_id"]))
    contrib = contributions(ref, inv)
    fams = sorted({e["family"] for e in inv})
    per_family = {}
    for f in fams:
        mine = [e for e in inv if e["family"] == f]
        sw = min(mine, key=lambda e: (-e["mean"], e["recipe_id"]))
        cands = [(e, w) for e in mine for w in WEIGHTS if e["run_id"] != ref["run_id"]]
        cw = min(cands, key=lambda c: (-contrib[(c[0]["run_id"], c[1])]["mean"], c[0]["recipe_id"], c[1])) if cands else None
        per_family[f] = {"standalone": sw["run_id"], "standalone_mean": sw["mean"],
                         "contribution": cw[0]["run_id"] if cw else None, "contribution_weight": cw[1] if cw else None,
                         "contribution_mean": contrib[(cw[0]["run_id"], cw[1])]["mean"] if cw else None}
    lead = [e for e in inv if e["family"] == ref["family"] and e["atom_id"] != ref["atom_id"]
            and ref["mean"] - e["mean"] <= LEAD_BAND]
    extras, atoms = [], {ref["atom_id"]}
    for e in sorted(lead, key=lambda e: (-e["mean"], e["recipe_id"])):
        if e["atom_id"] not in atoms and len(extras) < 2:
            extras.append(e["run_id"])
            atoms.add(e["atom_id"])
    by_run = {e["run_id"]: e for e in inv}
    picked = [ref["run_id"]] + [v[k] for v in per_family.values() for k in ("standalone", "contribution") if v[k]] \
        + extras + [ANCHOR_RUN]
    nominees, seen = [], set()
    for rid in picked:
        e = by_run[rid]
        if e["atom_id"] in seen and rid != ANCHOR_RUN:
            continue
        seen.add(e["atom_id"])
        roles = [r for r, ok in (("reference", rid == ref["run_id"]), ("anchor", rid == ANCHOR_RUN),
                                 ("leading_extra", rid in extras),
                                 ("standalone_winner", any(v["standalone"] == rid for v in per_family.values())),
                                 ("contribution_winner", any(v["contribution"] == rid for v in per_family.values())))
                 if ok]
        if rid == ANCHOR_RUN:
            base = dict(e["recipe"], name="p26_anchor")
            nominees.append({"source_run_id": rid, "family": e["family"], "atom_id": e["atom_id"], "roles": roles,
                             "policy": "anchor, estimator seed 0", "recipe": validate_recipe(base), "run_existing": None})
            nominees.append({"source_run_id": rid, "family": e["family"], "atom_id": e["atom_id"],
                             "roles": ["anchor_three_seeds"], "policy": "estimator seeds 0, 1, 2",
                             "recipe": validate_recipe(dict(base, name="p26_anchor_s3", estimator_seeds=[0, 1, 2])),
                             "run_existing": None})
            continue
        name = f"p26_{e['family']}_{e['atom_id'][:8]}"
        rec = final_policy(e["recipe"], name)
        nominees.append({"source_run_id": rid, "family": e["family"], "atom_id": e["atom_id"], "roles": roles,
                         "policy": f"estimator seeds {rec['estimator_seeds']}", "recipe": rec, "run_existing": None})
    return ref, contrib, per_family, extras, nominees


def main():
    ctx = runs.Context.load(Path("."))
    inv = inventory(ctx)
    ref, contrib, per_family, extras, nominees = nominate(inv)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recipes").mkdir(exist_ok=True)
    for n in nominees:
        if n["run_existing"] is None:
            (OUT / "recipes" / f"{n['recipe']['name']}.json").write_text(json.dumps(n["recipe"], indent=1) + "\n")
    doc = {"schema": "p26_nominations_v1",
           "inventory": [{k: e[k] for k in ("run_id", "task_id", "family", "bank_sha256", "recipe_sha256", "atom_id",
                                             "mean")} for e in inv],
           "reference": {k: ref[k] for k in ("run_id", "family", "recipe_id", "atom_id", "mean", "repeats")},
           "per_family": per_family, "leading_extras": extras, "nominees": nominees}
    (OUT / "nominations.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"inventory: {len(inv)} distinct complete atomic runs of {', '.join(TASKS)}")
    print(f"R0: {ref['run_id']} ({ref['family']}) {ref['mean']:.6f}")
    for f, v in per_family.items():
        print(f"  {f:14s} standalone {v['standalone']} {v['standalone_mean']:.6f}; contribution "
              f"{v['contribution']} at {v['contribution_weight']} -> {v['contribution_mean']:.6f}")
    print(f"leading extras: {extras}")
    print(f"nominees: {len(nominees)}")
    for n in nominees:
        print(f"  {n['recipe']['name']:34s} {n['family']:13s} {n['policy']:26s} {','.join(n['roles'])}")


if __name__ == "__main__":
    main()
