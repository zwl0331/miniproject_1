"""Save the explicitly requested seed-confirmation averages as complete banks."""
import json
from analysis.fast_blends import run
from analysis.fast_exploration import ROOT, load_arrays, summaries, write_json
from analysis.extended_cpu import current_references


def save_average(name, family, paths):
    if not all((ROOT / p).with_name("result.json").exists() for p in paths):
        return
    spec = dict(id=name, family=family, estimator_seeds=list(range(len(paths))),
                components=[dict(path=p, weight=1/len(paths)) for p in paths])
    destination = ROOT / "artifacts/fast" / spec["id"]
    if (destination / "result.json").exists():
        return
    run(spec)
    result = json.loads((destination / "result.json").read_text())
    result["comparisons"] = summaries(load_arrays(destination / "predictions.npz"), current_references())
    write_json(destination / "result.json", result)


def main():
    candidates = [
        ("extended", "x3_ebm_short_i60_lr0.03_r100", 5, "ebm"),
        ("extended", "x3_ebm_short_i30_lr0.03_r100", 5, "ebm"),
        ("extended", "x3_ebm_short_i30_lr0.003_r1000", 5, "ebm"),
        ("extended", "x3_ebm_short_i10_lr0.003_r1000", 5, "ebm"),
        ("neural", "x3_tabm_auc", 3, "tabm"),
        ("neural", "x3_retrieval_c16_w64_scalar", 3, "retrieval"),
        ("extended", "x3_pairlogit_d4_lr0.03_l23.0", 3, "pairlogit"),
        ("extended", "x3_pairlogit_d4_lr0.03_l210.0", 3, "pairlogit"),
        ("extended", "x3_pairlogit_d6_lr0.1_l210.0", 3, "pairlogit"),
        ("extended", "x3_neighbors_calendar_c64_lightgbm", 5, "lightgbm"),
        ("extended", "x3_neighbors_calendar_c8_lightgbm", 5, "lightgbm"),
    ]
    for parent, name, count, family in candidates:
        paths = [f"artifacts/{parent}/{name}{'_seed'+str(seed) if seed else ''}/predictions.npz"
                 for seed in range(count)]
        save_average("x3_avg_" + name.removeprefix("x3_"), family, paths)
    save_average("x3_avg_xgb3", "xgboost", ["artifacts/fast/f2_xgboost_3/predictions.npz"] +
                 [f"artifacts/extended/x3_xgb_trees_seed{s}/predictions.npz" for s in (1, 2)])


if __name__ == "__main__":
    main()
