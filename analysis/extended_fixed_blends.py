"""A few explicit probability/logit combinations of confirmed saved channels."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit, logit

from analysis.extended_cpu import current_references
from analysis.fast_exploration import ROOT, load_arrays, score, summaries, write_json
from songpop.data import sha256_file
from songpop.evaluate import PredictionBank, validate_bank


def mixture(kind, *parts):
    return dict(kind=kind, parts=[dict(weight=w, node=n) for w, n in parts])


def save(spec):
    folder = ROOT / "artifacts/fast" / spec["id"]
    if (folder / "result.json").exists():
        old = json.loads((folder / "result.json").read_text())
        if old["config"] != spec:
            raise ValueError("fixed blend id already belongs to another specification")
        return old
    started = time.monotonic()
    base = current_references()["followup_trees"]
    banks, hashes = {}, {}
    for label, path in spec["inputs"].items():
        a = load_arrays(ROOT / path)
        for key in ("train_row_id", "y", "group_id", "fold_id", "split_seeds", "test_id"):
            if not np.array_equal(a[key], base[key]):
                raise ValueError(f"{label} differs in {key}")
        validate_bank(PredictionBank(**a))
        banks[label], hashes[label] = a, sha256_file(ROOT / path)
    def evaluate(node, key):
        if isinstance(node, str):
            return banks[node][key]
        weights = np.array([p["weight"] for p in node["parts"]])
        if not np.isfinite(weights).all() or weights.min() < 0 or abs(weights.sum()-1) > 1e-12:
            raise ValueError("mixture weights must be nonnegative and sum to one")
        values = [evaluate(p["node"], key) for p in node["parts"]]
        if node["kind"] == "probability":
            return sum(w*v for w, v in zip(weights, values))
        if node["kind"] == "logit":
            return expit(sum(w*logit(np.clip(v, 1e-6, 1-1e-6)) for w, v in zip(weights, values)))
        raise ValueError("unknown fixed mixture operation")
    # Nonlinear combinations act per fold before test predictions are averaged.
    oof, test = (evaluate(spec["expression"], key) for key in ("oof", "test_by_fold"))
    auc = score(base["y"], base["fold_id"], oof)
    arrays = dict(base, oof=oof, test_by_fold=test, test_mean=test.mean(axis=(0, 1)), fold_auc=auc)
    validate_bank(PredictionBank(**arrays))
    write_json(folder / "config.json", spec)
    np.savez_compressed(folder / "predictions.npz", **arrays)
    result = dict(id=spec["id"], status="complete_fixed_blend", config=spec, cv=float(auc.mean()),
                  per_seed=auc.mean(axis=1).tolist(), elapsed_seconds=time.monotonic()-started,
                  input_sha256=hashes, prediction_sha256=sha256_file(folder / "predictions.npz"),
                  code_sha256=sha256_file(Path(__file__)), comparisons=summaries(arrays, current_references()))
    write_json(folder / "result.json", result)
    print(spec["id"], result["cv"], result["per_seed"], flush=True)
    return result


def confirmed_specs(ranking):
    inputs = dict(base="artifacts/fast/f2_blend_z5_et25/predictions.npz", ranking=ranking,
        neighbors="artifacts/fast/x3_avg_neighbors_calendar_c64_lightgbm/predictions.npz",
        ebm="artifacts/fast/x3_avg_ebm_short_i30_lr0.03_r100/predictions.npz")
    hybrid = mixture("probability", (.9, mixture("logit", (.9, "base"), (.1, "neighbors"))), (.1, "ranking"))
    expressions = dict(probability=mixture("probability", (.8, "base"), (.1, "neighbors"), (.1, "ranking")),
                       hybrid=hybrid, hybrid_ebm5=mixture("logit", (.95, hybrid), (.05, "ebm")),
                       hybrid_ebm10=mixture("logit", (.9, hybrid), (.1, "ebm")))
    suffix = Path(ranking).parent.name.removeprefix("x3_avg_").replace(".", "p")
    return [dict(id=f"x3_fixed_{label}_{suffix}", family="fixed_ensemble", inputs=inputs, expression=node)
            for label, node in expressions.items()]


def learned_specs():
    """Four prespecified conservative additions to the confirmed fixed blend."""
    base = "artifacts/fast/x3_fixed_hybrid_pairlogit_d4_lr0p03_l210p0/predictions.npz"
    choices = [
        ("core_stacker", "trees_mlp_xgb_stacker_0p01", "probability"),
        ("expanded_pairwise", "trees_mlp_xgb_ebm_tabm_pairwise_0p1", "logit"),
        ("final_stacker", "final9_stacker_0p01", "probability"),
        ("final_pairwise", "final9_pairwise_0p1", "logit"),
    ]
    return [dict(id=f"x3_fixed_hybrid_{label}25", family="fixed_ensemble",
                 inputs=dict(base=base, learned=f"artifacts/extended/x3_learned_{name}/predictions.npz"),
                 expression=mixture(kind, (.75, "base"), (.25, "learned")))
            for label, name, kind in choices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--ranking")
    action.add_argument("--learned", action="store_true")
    args = parser.parse_args()
    specs = learned_specs() if args.learned else confirmed_specs(args.ranking)
    name = "learned_blend_specs" if args.learned else "fixed_blend_specs"
    write_json(ROOT / f"artifacts/extended/{name}.json", specs)
    for spec in specs:
        save(spec)
