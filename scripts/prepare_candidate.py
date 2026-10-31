"""A phase 2 submission from an explicit selection manifest (doc/plans/phase2-models-features.md sections 4.7 and 10).

From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.prepare_candidate --manifest artifacts/phase2/selections/<name>.json
Add --verify to rebuild in memory, write nothing, and compare with the saved file and sidecar.
The manifest (artifacts/phase2/selections/<name>.json) holds exactly: schema "selection_v1", name, task_id, run_id
(one logged run: a fitted recipe or a declared blend), purpose and date. The script never searches the log for a
maximum. The file is written once (exclusive create), read back and validated; a submission record with the
manifest, bank, CSV and sidecar hashes is appended to artifacts/phase2/experiments.jsonl. --verify writes nothing.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from songpop import artifacts
from songpop.phase2 import runs
from songpop.phase2.submission import members_of, validate_candidate, write_candidate

MANIFEST_KEYS = ["schema", "name", "task_id", "run_id", "purpose", "date"]
DEPLOYMENT = ("within each base recipe, probabilities averaged equally over its estimator seeds, its 5 folds and "
              "its 3 split repeats; a blend then combines the base-recipe averages with its recorded weights")


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    if sorted(m) != sorted(MANIFEST_KEYS) or m["schema"] != "selection_v1":
        raise ValueError(f"selection manifest keys {sorted(m)}, expected {sorted(MANIFEST_KEYS)} and schema selection_v1")
    return m


def describe_candidate(m, bank, rec, sha):
    auc = float(bank.fold_auc.mean())
    md = bank.metadata
    if md["prediction_kind"] == "blend":
        what = "blend of " + ", ".join(f"{p['run_id']} ({p['weight']:g})" for p in md["parents"])
    else:
        what = f"{md['recipe']['family']} recipe {md['recipe_id']}"
    long = (f"{m['task_id']} candidate {m['name']}: {what}, High = 1. Five-fold StratifiedGroupKFold over track "
            f"groups, split seeds 155, 2026, 1029. Deployment: {DEPLOYMENT}. Mean fold CV AUC {auc:.4f}. Provided "
            f"training data only; no pretrained models; nothing fitted on test rows. Run {md['run_id']}, file "
            f"sha256 {sha}. Purpose: {m['purpose']}.")
    short = f"{m['date']}_{m['task_id'].lower()}_{m['name']}: grouped 5-fold CV AUC {auc:.4f}; sha256 {sha[:12]}"
    return long, short


def build(root, manifest_path, verify):
    ctx = runs.Context.load(root)
    m = load_manifest(root / manifest_path)
    bank, bank_sha, rec = runs.load_logged(ctx, m["run_id"])
    members = members_of(ctx, bank, lambda r: runs.load_logged(ctx, r)[:2])
    name = f"{m['date']}_{m['task_id'].lower()}_{m['name']}"
    csv_path, side_path = root / "submissions" / f"{name}.csv", root / "submissions" / f"{name}.json"
    if verify:
        content = csv_path.read_bytes()
        if content != artifacts.submission_bytes(bank.test_id, bank.test_mean):
            raise AssertionError(f"{csv_path.name} differs from the bytes rebuilt from the bank")
        report = validate_candidate(content, ctx, bank, members)
    else:
        if side_path.exists():
            raise FileExistsError(f"{side_path} exists; submission records are never overwritten")
        report = write_candidate(csv_path, ctx, bank, members)
    report["path"] = str(csv_path.relative_to(root))
    long, short = describe_candidate(m, bank, rec, report["sha256"])
    sidecar = {"task": m["task_id"], "file": report["path"], "sha256": report["sha256"], "run_id": m["run_id"],
               "bank_path": rec["bank_path"], "bank_sha256": bank_sha, "manifest": str(manifest_path),
               "manifest_sha256": artifacts.sha256_bytes((root / manifest_path).read_bytes()),
               "mean_fold_auc": float(bank.fold_auc.mean()), "deployment": DEPLOYMENT, "purpose": m["purpose"],
               "description": long, "kaggle_description": short, "validation": report}
    if bank.metadata["prediction_kind"] == "blend":
        sidecar["weights"] = {p["run_id"]: p["weight"] for p in bank.metadata["parents"]}
    side_text = json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n"
    record = {"record": "submission", "task_id": m["task_id"], "manifest": str(manifest_path),
              "manifest_sha256": sidecar["manifest_sha256"], "run_id": m["run_id"], "bank_sha256": bank_sha,
              "csv": report["path"], "csv_sha256": report["sha256"], "sidecar": str(side_path.relative_to(root)),
              "sidecar_sha256": artifacts.sha256_bytes(side_text.encode())}
    if verify:
        if side_path.read_text() != side_text:
            raise AssertionError(f"{side_path.name} differs from the rebuilt record")
        logged = [r for r in runs.logged(ctx, "submission") if r["csv"] == record["csv"]]
        if len(logged) != 1 or {k: logged[0].get(k) for k in record} != record:
            raise AssertionError(f"the submission record of {csv_path.name} is missing or differs")
        print(f"verified: {csv_path.name}, {side_path.name} and the submission record equal those rebuilt from the bank")
    else:
        with open(side_path, "x") as f:
            f.write(side_text)
        artifacts.append_experiment(root / runs.LOG_FILE,
                                    dict(record, timestamp_utc=datetime.now(timezone.utc).isoformat()))
        print(f"written: {csv_path.relative_to(root)} and {side_path.relative_to(root)}; submission record logged")
    print("validation:", json.dumps(report, sort_keys=True))
    print("description:", long)
    print("kaggle description:", short)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--verify", action="store_true", help="rebuild in memory, write nothing, compare")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    build(Path(args.root).resolve(), Path(args.manifest), args.verify)
    sys.exit(0)


if __name__ == "__main__":
    main()
