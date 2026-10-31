"""Summary of the P2.4 feature stage (doc/plans/phase2-models-features.md section 18) behind Findings 022 and 023,
and the handoff index for P2.6b (section 18.10).

Reads the phase 2 log, the banks, the batch manifests and the P2.4 protocol; fits nothing and writes nothing but its
standard output. It refuses to summarize an audit-incomplete log: the imports and P2.3's handoff must validate and
every logged batch must be judged or unstarted at the cutoff. From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -I analysis/p24_summary.py . > analysis/p24_summary_output.txt
All scores are exploratory: adaptively chosen comparisons on repeatedly used folds (plan sections 5.7 and 18.10).
"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))

from scripts import run_experiments as rx  # noqa: E402
from songpop.phase2 import feature_stage as fs, runs  # noqa: E402

TASK = "P2.4"


def fmt(xs):
    return "[" + ", ".join(f"{x:+.4f}" for x in xs) + "]"


def main():
    ctx = runs.Context.load(ROOT)
    proto, _ = rx.load_feature_protocol(ctx, TASK)
    imports = rx.validate_imports(ctx, proto)
    rx.validate_handoff(ctx, proto)
    flog = rx.FeatureLog(ctx, TASK)
    for b in flog.batches:
        if b["batch_id"] not in flog.allocations and (b["batch_id"] in flog.starts or not rx.past_cutoff(proto)):
            raise SystemExit(f"no summary: {b['batch_id']} is not judged")
    comps = {r["comparison_id"]: r for r in runs.logged(ctx, "comparison") if r.get("task_id") == TASK}
    by_pair = {(r["a"], r["b"]): r for r in comps.values()}
    hb, hs = rx.reference_bank(ctx, proto)
    print(f"P2.4 summary. {len(flog.of('features'))} feature batches, {len(flog.of('retune'))} re-tuning rounds; "
          f"stop record: {flog.stop['reason'] if flog.stop else 'none'}")
    print("Imported from P2.8 (routes as judged there):")
    for i in imports:
        print(f"  {i['id']}: {i['challenger_recipe']['name']} against {i['comparator_recipe']['name']}: "
              f"{i['routes'] or 'no route'}")
    index, qualified = [], []
    for b in flog.of("features"):
        m, j = flog.manifests[b["batch_id"]], flog.allocations.get(b["batch_id"], {}).get("judgment")
        print()
        print(f"== {b['batch_id']} ({m['kind']}, {len(m['comparisons'])} comparisons"
              + (f", omitted {[o['id'] for o in m['omitted']]}" if m["omitted"] else "") + ")")
        if j is None:
            print("  unstarted at the cutoff")
            continue
        for c in m["comparisons"]:
            v = j["comparisons"][c["id"]]
            r = by_pair.get((c["challenger_run_id"], c["comparator_run_id"]))
            ci = f"[{r['interval_95'][0]:+.4f}, {r['interval_95'][1]:+.4f}]" if r else "missing"
            g = c["generation"]
            what = g.get("change")
            print(f"  {c['id']} {g['queue']:12s} {c['family']:12s} {c['challenger']['name']} vs {c['comparator']['name']}"
                  f" {json.dumps(what)}: matched {v['matched_difference']:+.4f} {fmt(v['matched_repeat_differences'])} "
                  f"{ci}; challenger {v['challenger_mean']:.4f}, blends {', '.join(f'{w}: {x:.4f}' for w, x in v['blend_means'].items())}; "
                  f"routes {v['routes'] or 'none'}")
            e = rx.pool_entry(ctx, c["challenger_run_id"], hb, hs)
            index.append({"run_id": e["run_id"], "bank_sha256": e["bank_sha256"], "recipe": e["recipe"],
                          "features": e["features"], "mean_fold_auc_unrounded": e["mean"],
                          "repeat_mean_auc_unrounded": e["repeats"],
                          "blends": {str(w): x["mean"] for w, x in e["blends"].items()},
                          "lineage": {"comparison": c["id"], "comparator_run_id": c["comparator_run_id"],
                                      "queue": g["queue"]}, "routes": v["routes"]})
            if v["routes"]:
                qualified.append((c["id"], v["challenger_mean"], v["routes"]))
        print(f"  progress {j['progress']}, allowance {m['state']['allowance_before']} -> {j['allowance_after']}")
    for b in flog.of("retune"):
        m, j = flog.manifests[b["batch_id"]], flog.allocations.get(b["batch_id"], {}).get("judgment")
        print()
        print(f"== {b['batch_id']} (re-tuning round {m['index']}, units {len(m['units'])}, omitted units "
              f"{[o['unit_id'] for o in m['omitted_units']]})")
        if j is None:
            print("  unstarted at the cutoff")
            continue
        for u in m["units"]:
            res = j["units"][u["unit"]["unit_id"]]
            es = [rx.pool_entry(ctx, rid, hb, hs) for rid in u["run_ids"]]
            top = fs.best(es)
            print(f"  unit {u['unit']['unit_id']} {u['unit']['family']} blocks {u['unit']['blocks']} drop "
                  f"{u['unit']['drop']}: {len(es)} configurations, best {top['mean']:.4f} ({top['recipe']['name']}), "
                  f"progress {res['progress']}")
            for e in es:
                index.append({"run_id": e["run_id"], "bank_sha256": e["bank_sha256"], "recipe": e["recipe"],
                              "features": e["features"], "mean_fold_auc_unrounded": e["mean"],
                              "repeat_mean_auc_unrounded": e["repeats"],
                              "blends": {str(w): x["mean"] for w, x in e["blends"].items()},
                              "lineage": {"retune_unit": u["unit"]["unit_id"], "round": m["index"]},
                              "routes": res["routes"][e["run_id"]]})
    print()
    top = sorted(index, key=lambda x: -x["mean_fold_auc_unrounded"])[:8]
    print("Best P2.4 standalone runs:")
    for x in top:
        print(f"  {x['mean_fold_auc_unrounded']:.4f}  {x['recipe']['name']}  {x['recipe']['family']}  "
              f"{x['recipe']['blocks']}  drop {x['recipe']['drop']}")
    tb = sorted(index, key=lambda x: -max(x["blends"].values()))[:8]
    print("Best P2.4 contribution blends with H:")
    for x in tb:
        w, v = max(x["blends"].items(), key=lambda kv: kv[1])
        print(f"  {v:.4f} at weight {w}  {x['recipe']['name']}  {x['recipe']['family']}  {x['recipe']['blocks']}")
    print(f"Comparisons meeting a route: {[q[0] for q in qualified]}")
    print()
    print("Handoff index for P2.6b (plan section 18.10):")
    print(json.dumps({"task_id": TASK, "runs": index}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
