"""Summary of the P2.2 searches (doc/plans/phase2-models-features.md section 15) behind Findings 011 and 012.

Reads the phase 2 log, the batch manifests and the protocol; fits nothing and writes nothing but its standard
output. From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -I analysis/p22_summary.py . > analysis/p22_summary_output.txt
All scores are exploratory: each is a matched evaluation on the repeats that also chose it (plan section 5.7).
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))

from songpop import artifacts  # noqa: E402
from songpop.phase2 import protocol  # noqa: E402
from songpop.phase2.recipes import strict_loads  # noqa: E402

TASK = "P2.2"


def fmt(xs):
    return "[" + ", ".join(f"{x:+.4f}" for x in xs) + "]"


def main():
    proto = protocol.load_protocol(ROOT / "configs/phase2/p2.2/protocol.json")
    records = artifacts.read_experiments(ROOT / "artifacts/phase2/experiments.jsonl")
    log = protocol.Log(records, TASK, lambda rel: strict_loads((ROOT / rel).read_text()))
    h = protocol.h_run_id(proto, log)
    H = log.runs[h]
    hm = H["repeat_mean_auc_unrounded"]
    print(f"P2.2 summary. H (the LightGBM anchor's P2.2 run): {h}")
    print(f"  mean fold AUC {H['mean_fold_auc_unrounded']:.10f}, repeats {[round(x, 6) for x in hm]}")
    checks = [r for r in log.h_checks if r["h_run_id"] == h]
    print(f"  H check against {proto['reference_p21_run_id']}: {'passed' if checks and checks[0]['passed'] else 'MISSING'}")
    waves = sorted({b["wave"] for b in log.batches})
    print(f"Waves planned: {waves}; batches {len(log.batches)}, judged {len(log.allocations)}")
    print()
    totals = {"configs": 0, "seconds": 0.0}
    rows = []
    for fam in proto["families"]:
        f = fam["family"]
        fb = [b for b in log.batches if b["family"] == f]
        ids = []
        for b in fb:
            ids += [r for r in b["run_ids"] if r not in ids]
        done = [log.runs[r] for r in ids if r in log.runs]
        secs = sum(r["elapsed_seconds"] for r in done)
        totals["configs"] += len(done)
        totals["seconds"] += secs
        inc = protocol.best(done)
        blends = [c for r in done for c in log.contributions if c["run_id"] == r["run_id"] and c["h_run_id"] == h]
        bb = protocol.best_blend(blends)
        print(f"== {f}: {len(done)} configurations run of {len(ids)} planned (ceiling {fam['ceiling']}); "
              f"{secs / 60:.1f} min of fitting")
        anchor = log.runs.get(fb[0]["state"]["family_anchor"]["run_id"]) if fb else None
        if anchor:
            print(f"  family anchor  {anchor['run_id']}  {anchor['mean_fold_auc_unrounded']:.4f}  "
                  f"minus H by repeat {fmt([a - b for a, b in zip(anchor['repeat_mean_auc_unrounded'], hm)])}")
        if inc:
            d = [a - b for a, b in zip(inc["repeat_mean_auc_unrounded"], hm)]
            print(f"  best standalone {inc['run_id']}  {inc['mean_fold_auc_unrounded']:.4f}  minus H {sum(d) / 3:+.4f} "
                  f"by repeat {fmt(d)}")
            print(f"    recipe: base {log.manifest_recipe(inc['run_id'])['base']}, "
                  f"preprocessing {json.dumps(log.manifest_recipe(inc['run_id'])['preprocessing'], sort_keys=True)}, "
                  f"params {json.dumps(log.manifest_recipe(inc['run_id'])['params'], sort_keys=True)}")
        if bb:
            d = [a - b for a, b in zip(bb["repeat_mean_auc_unrounded"], hm)]
            print(f"  best blend with H  {bb['run_id']} at weight {bb['weight']}  {bb['mean_fold_auc_unrounded']:.4f}  "
                  f"minus H {sum(d) / 3:+.4f} by repeat {fmt(d)}")
        for b in fb:
            a = log.allocations.get(b["batch_id"])
            if a is None:
                status = "not judged" + (" (started)" if b["batch_id"] in log.starts else " (unstarted)")
            else:
                routes = {}
                for rs in a["routes"].values():
                    for r in rs:
                        routes[r] = routes.get(r, 0) + 1
                status = (f"progress {a['progress']}, allowance {b['state']['allowance_before']} -> "
                          f"{a['allowance_after']}, routes met {routes or 'none'}")
            ext = b["sources"].get("extensions") or []
            extra = f"; range extensions {[(e['parameter'], e['value']) for e in ext]}" if ext else ""
            print(f"  wave {b['wave']} {b['kind']:10s} {len(b['run_ids']):3d} configurations: {status}{extra}")
        rows.append((f, inc, bb))
        print()
    print("Bootstraps of the decision-driving comparisons (descriptive; conditional on the fixed predictions):")
    for r in sorted(log.bootstraps.values(), key=lambda r: (r["batch_id"], r["route"], r["comparison_id"])):
        print(f"  {r['batch_id']:42s} {r['route']:24s} {r['observed_difference']:+.4f} "
              f"[{r['interval_95'][0]:+.4f}, {r['interval_95'][1]:+.4f}] share>0 {r['draws_positive_share']:.3f} "
              f"repeats {fmt(r['repeat_differences'])}")
    print()
    print(f"Totals: {totals['configs']} configurations run, {totals['seconds'] / 60:.1f} min of fitting")
    print("Ranking by best standalone mean fold AUC (exploratory):")
    for f, inc, bb in sorted(rows, key=lambda x: -(x[1]["mean_fold_auc_unrounded"] if x[1] else 0)):
        if inc:
            print(f"  {f:14s} {inc['mean_fold_auc_unrounded']:.4f}   best blend with H "
                  f"{bb['mean_fold_auc_unrounded'] if bb else float('nan'):.4f}")


if __name__ == "__main__":
    main()
