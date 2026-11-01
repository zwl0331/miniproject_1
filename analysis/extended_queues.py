"""Concrete initial CPU comparisons for Sam's broader AUC exploration."""
import copy
import json

from analysis.extended_cpu import ROOT, OUT, write_json


def cpu_queue():
    lgb = json.loads((ROOT / "artifacts/fast/f2_lgb_zscore/config.json").read_text())
    et = json.loads((ROOT / "artifacts/fast/f2_extra_trees_3/config.json").read_text())
    queue = []
    def add(name, family, params, features, **extras):
        queue.append(dict(id="x3_" + name, family=family, params=copy.deepcopy(params),
                          features=copy.deepcopy(features), estimator_seeds=[0], **extras))
    base = dict(outer_bags=4, inner_bags=0, learning_rate=.03, max_bins=256,
                max_interaction_bins=32, min_samples_leaf=4)
    for interactions in (0, 10, 30, 60):
        for rounds in (1000, 3000, 6000):
            add(f"ebm_i{interactions}_r{rounds}", "ebm", dict(base, interactions=interactions,
                max_rounds=rounds), lgb["features"])
    for i, change in enumerate(({"outer_bags":8}, {"max_bins":64}, {"max_interaction_bins":64},
                               {"min_samples_leaf":10}, {"reg_lambda":1.}, {"reg_lambda":10.},
                               {"learning_rate":.01}, {"greedy_ratio":0.})):
        add(f"ebm_variant{i}", "ebm", dict(base, interactions=30, max_rounds=3000, **change)
            if not set(base) & set(change) else {**base, "interactions":30, "max_rounds":3000, **change}, lgb["features"])
    for depth in (4, 6):
        for lr in (.03, .1):
            for l2 in (3., 10.):
                add(f"pairlogit_d{depth}_lr{lr}_l2{l2}", "pairlogit", dict(depth=depth,
                    learning_rate=lr, l2_leaf_reg=l2, iterations=1500), lgb["features"])
    for view in ("audio", "calendar"):
        for clusters in (8, 32, 64):
            for config in (lgb, et):
                add(f"neighbors_{view}_c{clusters}_{config['family']}", config["family"],
                    config["params"], config["features"], neighbors={"view":view,"clusters":clusters})
    assert len(queue) == 40
    return queue


def shorter_ebm_queue():
    """Follow-up to the measured deterioration from 1,000 to 6,000 rounds."""
    template = cpu_queue()[0]
    queue = []
    for interactions in (0, 10, 30, 60):
        for lr, rounds in ((.03, 100), (.03, 300), (.03, 500),
                           (.003, 300), (.003, 1000), (.003, 3000)):
            config = copy.deepcopy(template)
            config["id"] = f"x3_ebm_short_i{interactions}_lr{lr}_r{rounds}"
            config["params"].update(interactions=interactions, learning_rate=lr, max_rounds=rounds)
            queue.append(config)
    return queue


if __name__ == "__main__":
    write_json(OUT / "cpu_queue.json", cpu_queue())
    print("Saved 40 configurations: 20 EBM, 8 pairwise CatBoost, 12 neighborhood-feature comparisons")
