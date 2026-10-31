"""Summary of the P2.3 searches (doc/plans/phase2-models-features.md section 17) behind Findings 019 and 020, and the
handoff to P2.4 (section 17.6).

Reads the phase 2 log, the banks, the batch manifests and the P2.3 protocol; fits nothing and writes nothing but its
standard output. It fails closed: no handoff without a passed H check, a valid prior member of G, the task complete by
the runner's own rule (every batch closed, except last-wave batches left unstarted because the cutoff passed, which
are reported as omitted; no further wave due before the cutoff), and every exported bank audited against its record
(runs.audit_run). From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -I analysis/p23_summary.py . > analysis/p23_summary_output.txt
All scores are exploratory: each is a matched evaluation on the repeats that also chose it (plan section 5.7).
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))

from scripts import run_experiments  # noqa: E402
from songpop import artifacts  # noqa: E402
from songpop.phase2 import banks, protocol, runs  # noqa: E402
from songpop.phase2.recipes import recipe_hash, strict_loads  # noqa: E402
from songpop.splits import SPLIT_SEEDS  # noqa: E402

TASK = "P2.3"
PROTOCOL = "configs/phase2/p2.3/protocol.json"


def fmt(xs):
    return "[" + ", ".join(f"{x:+.4f}" for x in xs) + "]"


def handoff_best(done, log):
    """The best complete run by unrounded mean fold AUC, ties to the lower recipe id (plan section 16.3; the
    contribution choice keeps section 15.4's rule, the canonical recipe hash and then the weight)."""
    recs = [r for r in done if protocol.complete(r)]
    return min(recs, key=lambda r: (-r["mean_fold_auc_unrounded"], r["recipe_id"])) if recs else None


def handoff_gate(log, problems):
    """Refuses a handoff while the runner's completion rule (run_experiments.completion_problems) reports a problem;
    otherwise returns the batches left unstarted at the cutoff, the only unjudged batches that rule allows."""
    if problems:
        raise SystemExit(f"no handoff: the task is not complete: {problems}")
    return [b["batch_id"] for b in log.batches if b["batch_id"] not in log.allocations]


def audited(ctx, run_id):
    problems = runs.audit_run(ctx, run_id)
    if problems:
        raise SystemExit(f"no handoff: run {run_id} fails its audit: {problems}")


def entry(ctx, rec, log):
    audited(ctx, rec["run_id"])
    recipe = log.manifest_recipe(rec["run_id"])
    return {"run_id": rec["run_id"], "recipe_id": rec["recipe_id"], "recipe_sha256": recipe_hash(recipe),
            "recipe": recipe, "bank_path": rec["bank_path"], "bank_sha256": rec["bank_sha256"],
            "source_manifest_hash": rec["source_manifest_hash"], "split_seeds": rec["split_seeds"],
            "estimator_seeds": rec["estimator_seeds"], "mean_fold_auc_unrounded": rec["mean_fold_auc_unrounded"],
            "repeat_mean_auc_unrounded": rec["repeat_mean_auc_unrounded"]}


def main():
    proto = protocol.load_protocol(ROOT / PROTOCOL)
    ctx = runs.Context.load(ROOT)
    run_experiments.validate_prior(ctx, proto)
    records = artifacts.read_experiments(ROOT / "artifacts/phase2/experiments.jsonl")
    log = protocol.Log(records, TASK, lambda rel: strict_loads((ROOT / rel).read_text()), proto["prior_strongest"])
    h = protocol.h_run_id(proto, log)
    if h not in log.runs:
        raise SystemExit("no handoff: H is not logged")
    audited(ctx, h)
    H, G0 = log.runs[h], log.runs[log.prior]
    hm = H["repeat_mean_auc_unrounded"]
    print(f"P2.3 summary. H (the reference family's anchor): {h}, bank {H['bank_sha256']}")
    print(f"  mean fold AUC {H['mean_fold_auc_unrounded']:.10f}, repeats {[round(x, 6) for x in hm]}")
    checks = [r for r in log.h_checks if r["h_run_id"] == h]
    if not (len(checks) == 1 and checks[0]["passed"]):
        raise SystemExit("no handoff: H has no passed H check")
    print(f"  H check against {proto['reference_p21_run_id']}: passed")
    omitted = handoff_gate(log, run_experiments.completion_problems(ctx, proto, log, list(SPLIT_SEEDS),
                                                                    banks.source_manifest(ROOT)))
    print(f"  completion: every batch closed{'; unstarted at the cutoff ' + str(omitted) if omitted else ''}")
    print(f"  G's prior member: {log.prior} ({G0['task_id']}), {G0['mean_fold_auc_unrounded']:.10f}")
    print(f"Waves planned: {sorted({b['wave'] for b in log.batches})}; batches {len(log.batches)}, "
          f"judged {len(log.allocations)}")
    handoff = {"task_id": TASK, "h_run_id": h, "h_bank_sha256": H["bank_sha256"],
               "batches_unstarted_at_cutoff": omitted,
               "prior_member_of_g": {"run_id": log.prior, "bank_sha256": G0["bank_sha256"],
                                     "mean_fold_auc_unrounded": G0["mean_fold_auc_unrounded"]}, "families": {}}
    for fam in proto["families"]:
        f = fam["family"]
        if not fam["space"]:
            continue
        fb = [b for b in log.batches if b["family"] == f]
        ids = []
        for b in fb:
            ids += [r for r in b["run_ids"] if r not in ids]
        done = [log.runs[r] for r in ids if r in log.runs]
        secs = sum(r["elapsed_seconds"] for r in done)
        print()
        print(f"== {f}: {len(done)} configurations run of {len(ids)} planned (ceiling {fam['ceiling']}); "
              f"{secs / 60:.1f} min of fitting")
        if not done:
            handoff["families"][f] = {"available": False, "reason": "no logged run of the family"}
            continue
        anchor = log.runs.get(fb[0]["state"]["family_anchor"]["run_id"])
        if anchor:
            print(f"  family anchor  {anchor['run_id']}  {anchor['mean_fold_auc_unrounded']:.4f}  "
                  f"minus H by repeat {fmt([a - b for a, b in zip(anchor['repeat_mean_auc_unrounded'], hm)])}")
        ranked = sorted(done, key=lambda r: (-r["mean_fold_auc_unrounded"], r["recipe_id"]))
        print("  top five by unrounded mean fold AUC (minus H by repeat; parameters):")
        for r in ranked[:5]:
            print(f"    {r['mean_fold_auc_unrounded']:.4f}  {fmt([a - b for a, b in zip(r['repeat_mean_auc_unrounded'], hm)])}"
                  f"  {log.manifest_recipe(r['run_id'])['name']}  {json.dumps(log.manifest_recipe(r['run_id'])['params'])}")
        for b in fb:
            a = log.allocations.get(b["batch_id"])
            met = {k: v for k, v in (a["routes"].items() if a else []) if v}
            print(f"  batch {b['batch_id']} ({b['kind']}, wave {b['wave']}, {len(b['run_ids'])} configurations): "
                  + (f"progress {a['progress']}, allowance {b['state']['allowance_before']} -> {a['allowance_after']}, "
                     f"{len(met)} meeting a route {sorted({x for v in met.values() for x in v})}" if a
                     else f"unstarted, the cutoff {proto['cutoff_utc']} passed"))
        best = handoff_best(done, log)
        blends = [c for r in done for c in log.contribution_pair(r["run_id"], h)]
        bb = protocol.best_blend(blends)
        print(f"  best standalone (handoff): {log.manifest_recipe(best['run_id'])['name']} "
              f"{best['mean_fold_auc_unrounded']:.10f}, minus H by repeat "
              f"{fmt([a - b for a, b in zip(best['repeat_mean_auc_unrounded'], hm)])}, minus G's prior member "
              f"{best['mean_fold_auc_unrounded'] - G0['mean_fold_auc_unrounded']:+.4f}")
        print(f"  best contribution to H: {log.manifest_recipe(bb['run_id'])['name']} at weight {bb['weight']}, "
              f"{bb['mean_fold_auc_unrounded']:.10f}, minus H by repeat "
              f"{fmt([a - b for a, b in zip(bb['repeat_mean_auc_unrounded'], hm)])}")
        handoff["families"][f] = {"available": True, "configurations": len(done),
                                  "best_standalone": entry(ctx, best, log),
                                  "best_contribution": {**entry(ctx, log.runs[bb["run_id"]], log), "weight": bb["weight"],
                                                        "h_bank_sha256": H["bank_sha256"],
                                                        "blend_mean_fold_auc_unrounded": bb["mean_fold_auc_unrounded"],
                                                        "blend_repeat_mean_auc_unrounded":
                                                            bb["repeat_mean_auc_unrounded"]}}
    print()
    print("Handoff to P2.4 (plan sections 6.5, 16.3 and 17.6):")
    print(json.dumps(handoff, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
