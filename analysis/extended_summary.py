"""Summarize complete extended trials against the preserved follow-up blend."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from analysis.fast_exploration import load_arrays, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/extended"
BASE = ROOT / "artifacts/fast/f2_blend_z5_et25/predictions.npz"


def collect():
    baseline = load_arrays(BASE)["fold_auc"].mean(axis=1)
    results = []
    for parent in (OUT, ROOT / "artifacts/neural"):
        for path in sorted(parent.glob("x3_*/result.json")):
            result = json.loads(path.read_text())
            comparisons = [c for c in result["comparisons"] if c["reference"] == "followup_trees"]
            if not comparisons:
                continue
            best = max(comparisons, key=lambda c: c["cv"])
            config = result["config"]
            family = config.get("family", config.get("recipe", {}).get("family"))
            row = dict(id=result["id"], family=family, cv=result["cv"],
                best_blend=best["cv"], weight=best["weight"], delta=best["delta_reference"],
                elapsed_seconds=result["elapsed_seconds"], path=str(path.relative_to(ROOT)))
            row.update({f"delta_{seed}": float(value - base) for seed, value, base in
                        zip((155, 2026, 1029), best["per_seed"], baseline)})
            row["positive_splits"] = sum(row[f"delta_{seed}"] > 0 for seed in (155, 2026, 1029))
            row["epoch_cap_hits"] = sum(item["stopped_by"] == "max_epochs" for fit in
                result.get("fit_info", []) for item in fit.get("tabm", []))
            results.append(row)
    return sorted(results, key=lambda row: row["delta"], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = collect()
    print(f"Completed: {len(rows)}; positive additions: {sum(row['delta'] > 0 for row in rows)}")
    for row in rows[:12]:
        print(f"{row['id']:42} CV={row['cv']:.6f} blend={row['best_blend']:.6f} "
              f"delta={row['delta']:+.6f} w={row['weight']:.2f} splits={row['positive_splits']}/3")
    errors = [str(path.relative_to(ROOT)) for parent in (OUT, ROOT / "artifacts/neural")
              for path in parent.glob("x3_*/error.json") if not path.with_name("result.json").exists()]
    print("Errors:", errors)
    if not args.write:
        return
    write_json(OUT / "summary.json", dict(updated_utc=datetime.now(timezone.utc).isoformat(),
               reference=str(BASE.relative_to(ROOT)), results=rows, unresolved_errors=errors))
    with (OUT / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
    notes = ROOT / "notes/2026-11-01-fast-exploration.md"
    marker = "<!-- extended-results -->"
    prefix = notes.read_text().split(marker)[0].rstrip()
    table = ["| Trial | Standalone CV | Best addition CV | Weight | Delta | Positive splits |",
             "|---|---:|---:|---:|---:|---:|"]
    table += [f"| {r['id']} | {r['cv']:.6f} | {r['best_blend']:.6f} | {r['weight']:.2f} | "
              f"{r['delta']:+.6f} | {r['positive_splits']}/3 |" for r in rows]
    notes.write_text(prefix + "\n\n" + marker + "\n\n## Extended exploration after takeover\n\n"
        "Sam reopened the remaining method tests. The preserved follow-up tree blend is the reference "
        "(CV 0.843814; public 0.86153, reported by Sam). Each complete run uses all 15 saved grouped folds. "
        "The best addition below is selected from fixed 10%, 25%, 50% probability blends. "
        "These are exploratory comparisons on reused folds, not independent significance tests. "
        "Initial configurations use estimator seed 0; promising candidates need further seeds.\n\n"
        f"Completed: {len(rows)}. Unresolved errors: {len(errors)}. Full configs, runtime, warnings, "
        "OOF/test arrays and fit records are under artifacts/extended and artifacts/neural. "
        "Machine-readable table: artifacts/extended/summary.csv.\n\n" + "\n".join(table) + "\n")


if __name__ == "__main__":
    main()
