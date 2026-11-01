"""Save available five-estimator-seed averages and a few fixed comparisons."""
import json

from analysis.fast_blends import run
from analysis.fast_exploration import ROOT, OUT, write_json, render_table


def component(name, weight):
    return {"path": f"artifacts/fast/{name}/predictions.npz", "weight": weight}


def save(spec):
    if not all((OUT.parents[1] / c["path"]).exists() for c in spec["components"]):
        return False
    dest = OUT / spec["id"] / "result.json"
    if not dest.exists():
        run(spec)
    return True


def main():
    specs = []
    for label, family, old, extra, weight in (
        ("lgb_base5", "lightgbm", "lgb_year_counts_seeds012", "f2_lgb_incumbent_extra34", .6),
        ("mlp_base5", "mlp", "mlp_w128_a1.0_e300_standard_seeds012", "f2_mlp_incumbent_extra34", .6),
        ("lgb_zscore5", "lightgbm", "f2_lgb_zscore", "f2_lgb_zscore_extra1234", .2),
        ("lgb_zero_l2_5", "lightgbm", "f2_lgb_tune04", "f2_lgb_tune04_extra1234", .2),
        ("et5", "extra_trees", "f2_extra_trees_3", "f2_extra_trees_3_extra1234", .2),
        ("cat5", "catboost", "f2_catboost_2", "f2_catboost_2_extra1234", .2),
    ):
        spec = {"id": "f2_avg_" + label, "family": family, "estimator_seeds": [0, 1, 2, 3, 4],
                "components": [component(old, weight), component(extra, 1-weight)]}
        if save(spec):
            specs.append(spec)
    write_json(OUT / "followup_seed_average_specs.json", specs)
    menu = json.loads((ROOT / "configs/phase2/menu.json").read_text())
    e = next(c for c in menu["candidates"] if c["id"] == "blend_eq_all")
    e_path = f"artifacts/phase2/bank/{e['run_id']}.npz"
    z = "f2_avg_lgb_zscore5"
    m = "mlp_w128_a1.0_e300_standard_seeds012"
    blends = [
        {"id": "f2_blend_z5", "components": [{"path": e_path, "weight": .65}, component(z, .25), component(m, .1)]},
        {"id": "f2_blend_z5_et10", "components": [{"path": e_path, "weight": .55}, component(z, .25), component(m, .1), component("f2_avg_et5", .1)]},
        {"id": "f2_blend_z5_et25", "components": [{"path": e_path, "weight": .4}, component(z, .25), component(m, .1), component("f2_avg_et5", .25)]},
        {"id": "f2_blend_lgbmix", "components": [{"path": e_path, "weight": .65}, component(z, .125), component("f2_avg_lgb_zero_l2_5", .125), component(m, .1)]},
        {"id": "f2_blend_z5_mlp5", "components": [{"path": e_path, "weight": .65}, component(z, .25), component("f2_avg_mlp_base5", .1)]},
    ]
    for label, replace in (("replace_et", {"extra_trees": "f2_avg_et5"}),
                           ("replace_et_cat", {"extra_trees": "f2_avg_et5", "catboost": "f2_avg_cat5"})):
        components = [component(z, .25), component(m, .1)]
        for member in e["members"]:
            family = menu["atom_recipes"][member["run_id"]]["family"]
            if family in replace:
                components.append(component(replace[family], .65 * member["weight"]))
            else:
                components.append({"path": f"artifacts/phase2/bank/{member['run_id']}.npz",
                                   "weight": .65 * member["weight"]})
        blends.append({"id": "f2_blend_z5_" + label, "components": components})
    for spec in blends:
        save(spec)
    write_json(OUT / "followup_fixed_blend_specs.json", blends)
    render_table()


if __name__ == "__main__":
    main()
