"""Save the explicitly requested seed-confirmation averages as complete banks."""
import json
from analysis.fast_blends import run
from analysis.fast_exploration import ROOT, load_arrays, summaries, write_json
from analysis.extended_cpu import current_references


def main():
    candidates = [
        ("extended", "x3_ebm_short_i60_lr0.03_r100", 5, "ebm"),
        ("extended", "x3_ebm_short_i30_lr0.03_r100", 5, "ebm"),
        ("extended", "x3_ebm_short_i30_lr0.003_r1000", 5, "ebm"),
        ("extended", "x3_ebm_short_i10_lr0.003_r1000", 5, "ebm"),
        ("neural", "x3_tabm_auc", 3, "tabm"),
    ]
    for parent, name, count, family in candidates:
        paths = [f"artifacts/{parent}/{name}{'_seed'+str(seed) if seed else ''}/predictions.npz"
                 for seed in range(count)]
        if not all((ROOT / p).with_name("result.json").exists() for p in paths):
            continue
        spec = dict(id="x3_avg_" + name.removeprefix("x3_"), family=family,
                    estimator_seeds=list(range(count)),
                    components=[dict(path=p, weight=1/count) for p in paths])
        destination = ROOT / "artifacts/fast" / spec["id"]
        if (destination / "result.json").exists():
            continue
        run(spec)
        result = json.loads((destination / "result.json").read_text())
        result["comparisons"] = summaries(load_arrays(destination / "predictions.npz"), current_references())
        write_json(destination / "result.json", result)


if __name__ == "__main__":
    main()
