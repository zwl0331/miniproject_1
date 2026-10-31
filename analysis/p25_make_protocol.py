"""Writes configs/phase2/p2.5/protocol.json and configs/phase2/p2.5/timing.json (plan section 19), from the project
root, after P2.4 has run: the wave protocol of P2.2 and P2.3 with the LightGBM reference family (H) and the tabm
family; prior_strongest is the strongest complete run of P2.2, P2.8, P2.3 and P2.4 in the log, by unrounded mean
fold AUC, ties to the lower run id. Fits nothing.
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p25_make_protocol.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from songpop import artifacts  # noqa: E402
from songpop.phase2 import protocol, runs  # noqa: E402
from songpop.phase2.recipes import recipe_hash  # noqa: E402

ROOT = Path(".")
TEMPLATE = {"base": "F2", "blocks": [], "drop": [], "preprocessing": {"name": "tabm_numeric", "scaler": "standard"},
            "training_weights": "none", "estimator_seeds": [0]}
REF_TEMPLATE = {"base": "F2", "blocks": [], "drop": [], "preprocessing": {"name": "none"}, "training_weights": "none",
                "estimator_seeds": [0]}
SPACE = {"n_blocks": [2, 3], "d_block": [64, 128, 256], "dropout": [0.0, 0.1, 0.2],
         "learning_rate": [0.0003, 0.0005, 0.001, 0.002, 0.003], "weight_decay": [1e-06, 1e-05, 0.0001, 0.001, 0.01],
         "batch_size": [128, 256], "preprocessing.scaler": ["standard", "quantile"]}
# The family anchor N0 and seven variants around it (six change one setting, one a regularization trio) (round S of the consultation): width 64 and 256,
# three blocks, learning rates 3e-4 and 3e-3, stronger regularization with batch 128, and the matched preprocessing
# comparison (quantile against standard), the one neural feature comparison P2.4 omitted.
ANCHOR = {"n_blocks": 2, "d_block": 128, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0001,
          "batch_size": 256, "preprocessing.scaler": "standard"}
VARIANTS = [{"d_block": 64}, {"d_block": 256}, {"n_blocks": 3}, {"learning_rate": 0.0003}, {"learning_rate": 0.003},
            {"dropout": 0.2, "weight_decay": 0.01, "batch_size": 128}, {"preprocessing.scaler": "quantile"}]
# Every TabM recipe sets all six parameters (the estimator has no defaults), so "default" is never used to complete
# a recipe; it is the anchor's value.
TYPES = {"n_blocks": {"type": "ordinal", "values": [2, 3], "default": 2},
         "d_block": {"type": "ordinal", "values": [64, 128, 256], "default": 128},
         "dropout": {"type": "ordinal", "values": [0.0, 0.1, 0.2], "default": 0.1},
         "learning_rate": {"type": "log", "integer": False, "zero_step": None, "default": 0.001},
         "weight_decay": {"type": "log", "integer": False, "zero_step": None, "default": 0.0001},
         "batch_size": {"type": "ordinal", "values": [128, 256], "default": 256},
         "preprocessing.scaler": {"type": "categorical", "values": ["standard", "quantile"], "default": "standard"}}
# Hard bounds at the declared space's edges, so no range extension ever leaves section 5.5's space (round T).
HARD = {"learning_rate": {"min": 0.0003, "max": 0.003, "integer": False},
        "weight_decay": {"min": 1e-06, "max": 0.01, "integer": False}}
BUDGET = {"fitting_seconds": 10800}  # section 5.5: three hours for fitting and confirmation
P21_ANCHOR = "p2_lightgbm_F2_anchor_seeds155-2026-1029_e00287a531af"
CUTOFF = "2026-10-31T15:00:00+00:00"  # Saturday 08:00 PDT, the schedule guard: no exploratory batch starts later


def strongest(ctx):
    """The strongest complete standalone recipe (fitted or seed-averaged, never a blend) of the earlier tasks, ties to
    the lower canonical recipe hash (protocol.best's rule), audited against its bank."""
    recs = []
    for r in runs.logged(ctx):
        if r.get("task_id") in ("P2.2", "P2.8", "P2.3", "P2.4") and protocol.complete(r) \
                and r["prediction_kind"] in ("fitted", "seed_average"):
            bank, sha, _ = runs.load_logged(ctx, r["run_id"])
            recs.append((r, bank, sha))
    r, bank, sha = min(recs, key=lambda t: (-t[0]["mean_fold_auc_unrounded"], recipe_hash(t[1].metadata["recipe"])))
    if runs.audit_run(ctx, r["run_id"]):
        raise ValueError(f"{r['run_id']} does not audit against its bank")
    return {"run_id": r["run_id"], "task_id": r["task_id"], "recipe": bank.metadata["recipe"], "bank_sha256": sha}


def main():
    ctx = runs.Context.load(ROOT)
    p23 = json.loads((ROOT / "configs/phase2/p2.3/protocol.json").read_text())
    ref = dict(p23["families"][0])
    tabm = {"family": "tabm", "index": 1, "initial": 8, "expansion": 4, "ceiling": 16, "template": TEMPLATE,
            "anchor": ANCHOR, "initial_branches": [{"kind": "explicit", "template": TEMPLATE,
                                                    "values": [dict(ANCHOR, **v) for v in VARIANTS]}],
            "space_template": TEMPLATE, "fixed": {}, "space": SPACE, "derived": [], "admissible": [], "types": TYPES,
            "hard_bounds": HARD, "trees": None}
    assert ref["family"] == "lightgbm" and ref["template"] == REF_TEMPLATE and ref["ceiling"] == 1
    proto = {"schema": protocol.PROTOCOL_SCHEMA, "task_id": "P2.5", "name": "p25",
             "reference": p23["reference"].replace("P2.3", "P2.5"), "reference_p21_run_id": P21_ANCHOR,
             "split_seeds": [155, 2026, 1029], "seed_base": 2026103100, "cutoff_utc": CUTOFF,
             "refinement": {"families": 1, "by_standalone": 1, "batch": 4, "broad": 1, "local": [[2, 1], [1, 2]],
                            "max_dimensions": 2, "max_per_family": 8},
             "prior_strongest": strongest(ctx), "budget": BUDGET, "families": [ref, tabm]}
    if any(r.get("task_id") == "P2.5" and r.get("record") in ("timing", "batch", "run")
           for r in artifacts.read_experiments(ROOT / runs.LOG_FILE)):
        raise SystemExit("P2.5 records exist: the protocol is bound to them and is not rewritten")
    out = ROOT / "configs/phase2/p2.5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(proto, indent=1) + "\n")
    protocol.load_protocol(out / "protocol.json")
    def corner(name, **p):
        v = dict(ANCHOR, **p)
        return protocol.make_recipe(TEMPLATE, "tabm", name, v)
    cases = [("light", corner("p25_time_tabm_light", d_block=64, learning_rate=0.003)),
             ("representative", corner("p25_time_tabm_representative")),
             ("expensive", corner("p25_time_tabm_expensive", d_block=256, n_blocks=3, learning_rate=0.0003,
                                  batch_size=128, dropout=0.2))]
    (out / "timing.json").write_text(json.dumps({"schema": "timing_v1", "cases": [{"case": c, "recipe": r}
                                                                                   for c, r in cases]}, indent=1) + "\n")
    print("prior_strongest", proto["prior_strongest"]["run_id"], proto["prior_strongest"]["task_id"])
    print("initial", [r["name"] for r in protocol.initial_recipes(proto, tabm)])


if __name__ == "__main__":
    main()
