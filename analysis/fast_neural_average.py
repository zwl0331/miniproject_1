"""Two simple neural averages after the entire fixed eight-model screen."""
import json

from analysis.fast_blends import run
from analysis.fast_exploration import ROOT, OUT, write_json


def main():
    queue = []
    for i in (0, 1):
        queue.extend(json.loads((ROOT / f"artifacts/neural/followup_queue_{i}.json").read_text()))
    results = []
    for config in queue:
        path = ROOT / "artifacts/neural" / config["id"] / "result.json"
        if not path.exists():
            print("Waiting for the complete eight-model screen")
            return
        results.append(json.loads(path.read_text()))
    families = {r["config"]["recipe"]["family"] for r in results}
    counts = {f: sum(r["config"]["recipe"]["family"] == f for r in results) for f in families}
    balanced = {"id": "f2_neural_avg_family", "family": "neural_ensemble", "components": []}
    for r in results:
        family = r["config"]["recipe"]["family"]
        balanced["components"].append({"path": f"artifacts/neural/{r['id']}/predictions.npz",
                                       "weight": 1 / len(families) / counts[family]})
    top = sorted(results, key=lambda r: r["cv"], reverse=True)[:3]
    best = {"id": "f2_neural_avg_top3", "family": "neural_ensemble",
            "selection": "Three best standalone mean-fold CV scores among the eight completed configs; exploratory.",
            "components": [{"path": f"artifacts/neural/{r['id']}/predictions.npz", "weight": 1/3} for r in top]}
    for spec in (balanced, best):
        if not (OUT / spec["id"] / "result.json").exists():
            run(spec)
    write_json(OUT / "followup_neural_average_specs.json", [balanced, best])


if __name__ == "__main__":
    main()
