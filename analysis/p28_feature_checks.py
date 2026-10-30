"""Part B of P2.8 (doc/plans/phase2-models-features.md section 16.3): the feature checks moved from P2.4, read-only
from the logged P2.8 runs and the allocation state saved in configs/phase2/p2.8/manifest.json before any P2.8 score.

For each declared comparison: the matched difference of challenger and comparator (unrounded mean fold AUC, the
three repeat differences, the paired cluster bootstrap with the manifest's draws and seed, descriptive), the
challenger's two contribution blends with H, and the section 5.6 routes it meets against the saved state (family
incumbent, strongest standalone recipe, H, the family's previous best blend). Then the two fixed-blend diagnostics,
and the reference checks: P2.8's H equal to the P2.1 anchor bank, and part A's H equal to P2.8's H once part A has
run. Writes nothing but its standard output; rerunning it reproduces that output; exits 1 if a check fails. From
the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -I analysis/p28_feature_checks.py . > analysis/p28_feature_checks_output.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "identifier_probe"))

import probe_bank  # noqa: E402
from songpop.evaluate import PredictionBank  # noqa: E402
from songpop.metrics import paired_cluster_bootstrap  # noqa: E402
from songpop.phase2 import allocation, runs  # noqa: E402
from songpop.phase2.ensembles import blend_banks  # noqa: E402
from songpop.phase2.recipes import load_recipe, recipe_hash, recipe_id  # noqa: E402

MANIFEST = "configs/phase2/p2.8/manifest.json"
PROBE_H = "artifacts/p2.8/identifier_probe/banks/H.npz"
TASK = "P2.8"


def equal_arrays(a, b):
    return [f for f in PredictionBank.ARRAYS if not np.array_equal(getattr(a, f), getattr(b, f))]


def stats(challenger, comparator, bs):
    res = paired_cluster_bootstrap(challenger, comparator, n_boot=bs["draws"], seed=bs["seed"])
    rep = (challenger.fold_auc.mean(axis=1) - comparator.fold_auc.mean(axis=1)).tolist()
    return float(challenger.fold_auc.mean() - comparator.fold_auc.mean()), rep, res


def line(label, diff, rep, res):
    return (f"{label}: {diff:+.6f}, by repeat {', '.join(f'{x:+.6f}' for x in rep)}; "
            f"95% [{res['interval_95'][0]:+.6f}, {res['interval_95'][1]:+.6f}], "
            f"share above zero {res['draws_positive_share']:.3f}, draws {res['valid']}/{res['n_boot']}")


def main():
    ctx = runs.Context.load(ROOT)
    man = json.loads((ROOT / MANIFEST).read_text())
    b = man["part_b"]
    ok = True
    logged = [r for r in runs.logged(ctx) if r.get("task_id") == TASK]
    by_name = {}
    for spec in b["recipes"]:
        recipe = load_recipe(ROOT / spec["path"])
        recs = [r for r in logged if r["recipe_id"] == recipe_id(recipe)]
        if len(recs) != 1:
            print(f"{spec['name']}: {len(recs)} logged P2.8 runs, expected 1")
            ok = False
            continue
        bank, sha, rec = runs.load_logged(ctx, recs[0]["run_id"])
        if not recipe_hash(bank.metadata["recipe"]) == spec["recipe_sha256"] == recipe_hash(recipe):
            print(f"{spec['name']}: recipe differs from the manifest")
            ok = False
        by_name[spec["name"]] = (bank, sha, rec)
    if len(by_name) != len(b["recipes"]) or len(logged) != len(b["recipes"]):
        print(f"P2.8 runs logged: {len(logged)}; declared recipes: {len(b['recipes'])}")
        sys.exit(1)
    state = b["allocation_state"]
    hb, hsha, _ = by_name[state["H"]]
    ref = runs.load_logged(ctx, b["h_must_equal"]["run_id"])[0]
    bad = equal_arrays(hb, ref)
    print(f"P2.8's H against the P2.1 anchor bank: {'all ten arrays equal' if not bad else 'DIFFERS in ' + str(bad)}")
    ok &= not bad
    if (ROOT / PROBE_H).exists():
        bad = equal_arrays(probe_bank.from_bytes((ROOT / PROBE_H).read_bytes()), hb)
        print(f"part A's H against P2.8's H: {'all ten arrays equal' if not bad else 'DIFFERS in ' + str(bad)}")
        ok &= not bad
    else:
        print("part A's H against P2.8's H: part A has not run yet")
    h_rep = allocation.repeat_means(hb)
    strongest = state["strongest_standalone"]["repeat_mean_auc_unrounded"]
    inc = ", ".join(f"{f} {v['mean_fold_auc_unrounded']:.6f}" for f, v in state["family_incumbents"].items())
    pbb = ", ".join(f"{f} {v['mean_fold_auc_unrounded']:.6f}" for f, v in state["previous_best_blends"].items())
    print(f"saved state: incumbents {inc}; strongest standalone "
          f"{state['strongest_standalone']['mean_fold_auc_unrounded']:.6f}; previous best blends {pbb}")
    progress = []
    for c in b["comparisons"]:
        chal, csha, _ = by_name[c["challenger"]]
        comp = by_name[c["comparator"]][0]
        diff, rep, res = stats(chal, comp, b["bootstrap"])
        blends = [allocation.contribution(hb, hsha, chal, csha, w) for w in state["contribution_weights"]]
        met = allocation.routes({"standalone": allocation.repeat_means(chal).tolist(),
                                 "blends": [x["repeat_mean_auc"] for x in blends]},
                                state["family_incumbents"][c["family"]]["repeat_mean_auc_unrounded"], strongest,
                                h_rep.tolist(), state["previous_best_blends"][c["family"]]["repeat_mean_auc_unrounded"])
        progress += met
        print(line(f"{c['id']} {c['challenger']} - {c['comparator']}", diff, rep, res))
        shown = ", ".join(f"{x['weight']}: {x['mean_fold_auc']:.6f}" for x in blends)
        print(f"    challenger {chal.fold_auc.mean():.6f}; blends with H {shown}; routes met: {met or 'none'}")
    print(f"routes met by the batch: {sorted(set(progress)) or 'none'}")
    for d in b["blend_diagnostics"]:
        members = [(by_name[n][0], by_name[n][1]) for n, _ in d["members"]]
        bl = blend_banks(members, [w for _, w in d["members"]])
        diff, rep, res = stats(bl, by_name[d["comparator"]][0], b["bootstrap"])
        label = " + ".join(f"{w} {n}" for n, w in d["members"])
        print(line(f"{d['id']} {label} - {d['comparator']}", diff, rep, res) + f"; blend {bl.fold_auc.mean():.6f}")
    rf_new = by_name["p28_random_forest_b1_0_040"][0]
    rf_old = state["family_incumbents"]["random_forest"]
    print(f"random forest base refitted with deterministic prediction: {rf_new.fold_auc.mean():.10f}; its logged P2.2 "
          f"score {rf_old['mean_fold_auc_unrounded']:.10f} (the allocation state keeps the logged score, section 15.7)")
    print("reference checks:", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
