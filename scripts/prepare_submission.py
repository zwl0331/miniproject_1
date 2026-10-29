"""The first submission (task P1.3, doc/plans/phase1-data-baseline.md section 5).

From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.prepare_submission
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.prepare_submission --verify
The rule: among the production F2 banks logged in artifacts/experiments.jsonl, the highest unrounded mean
fold AUC; a tie within 1e-12 goes to the order LightGBM, random forest, logistic. The file is written once by
songpop.artifacts.write_submission (exclusive create, read back, validated), with a sidecar JSON holding its
provenance and the description for Kaggle. --verify rebuilds both in memory, compares, and writes nothing.
"""
import argparse
import json
import sys
from pathlib import Path

from songpop import artifacts
from songpop.data import load_dataset
from songpop.models import MODELS

TASK = "P1.3"
LOG_FILE = Path("artifacts/experiments.jsonl")
OUT_DIR = Path("submissions")
PURPOSE = "verify the end-to-end submission pipeline"


def select(root):
    """The P1.3 rule over the logged production F2 runs; returns (record, bank)."""
    runs = [r for r in artifacts.read_experiments(root / LOG_FILE)
            if r.get("record") == "run" and r.get("protocol") == "production" and r.get("feature_set") == "F2"
            and r.get("selection_eligible")]
    if sorted(r["model"] for r in runs) != sorted(MODELS):
        raise ValueError(f"expected one production F2 run per model, found {[r['run_id'] for r in runs]}")
    banks = {r["model"]: artifacts.load_bank(root / r["bank_path"]) for r in runs}
    for r in runs:
        if artifacts.sha256_bytes((root / r["bank_path"]).read_bytes()) != r["bank_sha256"]:
            raise ValueError(f"bank {r['bank_path']} differs from its logged hash")
    top = max(float(b.fold_auc.mean()) for b in banks.values())
    model = next(m for m in MODELS if float(banks[m].fold_auc.mean()) >= top - 1e-12)
    return next(r for r in runs if r["model"] == model), banks[model]


def describe_submission(rec, bank, sha):
    auc = float(bank.fold_auc.mean())
    long = (f"{TASK} pipeline baseline: {rec['model']}, F2 audio and release-date features, High = 1. "
            f"Five-fold StratifiedGroupKFold over track groups, split seeds {', '.join(map(str, rec['split_seeds']))}, "
            f"estimator seed {rec['estimator_seeds'][0]}. Equal average of the 15 fold models' probabilities. "
            f"Mean fold CV AUC {auc:.4f}. Provided training data only; no pretrained models; nothing fitted on test "
            f"rows. Recipe {rec['recipe_id']}, run {rec['run_id']}, file sha256 {sha}. Purpose: {PURPOSE}.")
    short = (f"{TASK} pipeline check: {rec['model']} F2 baseline, grouped 5-fold CV AUC {auc:.4f} "
             f"(split seeds 155/2026/1029), mean of 15 fold models; sha256 {sha[:12]}")
    return long, short


def build(root, data, verify):
    rec, bank = select(root)
    name = f"2026-10-29_p1.3_{rec['recipe_id']}"
    csv_path, side_path = root / OUT_DIR / f"{name}.csv", root / OUT_DIR / f"{name}.json"
    if verify:
        content = csv_path.read_bytes()
        if content != artifacts.submission_bytes(bank.test_id, bank.test_mean):
            raise AssertionError(f"{csv_path.name} differs from the bytes rebuilt from the bank")
        report = artifacts.validate_submission(content, data, bank)
        report["path"] = str(csv_path.relative_to(root))
    else:
        if side_path.exists():
            raise FileExistsError(f"{side_path} exists; submission records are never overwritten")
        report = artifacts.write_submission(csv_path, data, bank)
        report["path"] = str(csv_path.relative_to(root))
    long, short = describe_submission(rec, bank, report["sha256"])
    sidecar = {"task": TASK, "file": report["path"], "sha256": report["sha256"], "recipe_id": rec["recipe_id"],
               "run_id": rec["run_id"], "bank_path": rec["bank_path"], "bank_sha256": rec["bank_sha256"],
               "mean_fold_auc": float(bank.fold_auc.mean()), "deployment": rec["deployment"],
               "selection_rule": "highest unrounded mean fold AUC among the production F2 banks; tie within 1e-12 "
                                 "to the order LightGBM, random forest, logistic",
               "purpose": PURPOSE, "description": long, "kaggle_description": short, "validation": report}
    if verify:
        saved = json.loads(side_path.read_text())
        if saved != json.loads(artifacts.dumps(sidecar)):
            raise AssertionError(f"{side_path.name} differs from the rebuilt record")
        print(f"verified: {csv_path.name} and {side_path.name} equal the records rebuilt from the bank")
    else:
        with open(side_path, "x") as f:
            f.write(json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(f"written: {csv_path.relative_to(root)} and {side_path.relative_to(root)}")
    print(json.dumps({k: sidecar[k] for k in ("file", "sha256", "run_id", "mean_fold_auc")}, indent=2))
    print("validation:", json.dumps(report, sort_keys=True))
    print("description:", long)
    print("kaggle description:", short)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="rebuild in memory, write nothing, compare")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    build(root, load_dataset(root), args.verify)
    sys.exit(0)


if __name__ == "__main__":
    main()
