"""Running, logging and verifying phase 2 runs (plan sections 4.2 to 4.5): the data and saved folds, the run
identity, the bank_v2 metadata, the phase 2 experiment log, and replay lookups for a changed source."""
import hashlib
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from songpop import artifacts
from songpop.data import load_dataset
from songpop.evaluate import N_SPLITS
from songpop.metrics import diagnostics, summarize
from songpop.models import describe
from songpop.phase2 import banks
from songpop.phase2.engine import raw_rows, run_recipe_cv
from songpop.phase2.ensembles import blend_banks, select_seeds
from songpop.phase2.features import feature_columns
from songpop.phase2.models import family_packages, make_model
from songpop.phase2.recipes import recipe_hash, recipe_id, validate_recipe
from songpop.splits import SPLIT_SEEDS, SPLIT_VERSION, FoldSet, build_foldset, era_of

ARTIFACTS = Path("artifacts/phase2")
BANK_DIR = ARTIFACTS / "bank"
LOG_FILE = ARTIFACTS / "experiments.jsonl"
SEARCH_DIR = ARTIFACTS / "searches"
SPLIT_FILE = Path("artifacts/splits/production_v1.csv")
SPLIT_SHA256 = "b2cba98d49b27c7c57e18cb211c3cd4a2a96509a58203a6214a51f2eb22939ee"  # Finding 007
FITTING_POPULATION = "the eligible training rows outside the held-out fold, for each repeat and fold"


class SourceMismatch(Exception):
    """A run's recorded source files differ from the current ones; it replays only at a matching commit."""


@dataclass
class Context:
    root: Path
    data: object
    foldset: FoldSet
    split_sha256: str

    @classmethod
    def load(cls, root):
        """The data and the saved primary folds, checked against their recorded hash and against a recomputation."""
        root = Path(root).resolve()
        data = load_dataset(root)
        raw = (root / SPLIT_FILE).read_bytes()
        sha = artifacts.sha256_bytes(raw)
        if sha != SPLIT_SHA256:
            raise ValueError(f"{SPLIT_FILE} has sha256 {sha}, expected {SPLIT_SHA256}")
        fs = FoldSet.from_frame(pd.read_csv(root / SPLIT_FILE))
        if not fs.equals(build_foldset(data)):
            raise AssertionError(f"recomputed folds differ from {SPLIT_FILE}")
        return cls(root, data, fs, sha)

    @property
    def rows(self):
        return raw_rows(self.data.train_eligible, self.data.row_ids)

    @property
    def test_rows(self):
        return raw_rows(self.data.test, self.data.test_id)

    @property
    def groups(self):
        return self.foldset.primary_group[self.data.eligible]

    def fold_id(self, seeds):
        """The saved primary folds of the eligible rows for the requested split seeds, selected by seed value."""
        seeds = banks.canonical_seeds(seeds)
        return np.vstack([self.foldset.primary[SPLIT_SEEDS.index(s), self.data.eligible] for s in seeds])


def packages_for(recipe):
    return {**artifacts.packages(), **family_packages(recipe)}


def fitted_config(ctx, recipe, seeds, manifest):
    """The identity fields of a fitted run (plan section 4.2)."""
    validate_recipe(recipe)
    seeds = banks.canonical_seeds(seeds)
    env = artifacts.environment(ctx.root)
    return {"kind": "fitted", "recipe_sha256": recipe_hash(recipe), "data_hashes": ctx.data.hashes,
            "evaluation_population": banks.evaluation_population(ctx.data.row_ids),
            "fitting_population": FITTING_POPULATION, "split_version": SPLIT_VERSION, "split_sha256": ctx.split_sha256,
            "grouping": "primary", "split_seeds": [int(s) for s in seeds],
            "fold_ids_sha256": artifacts.sha256_bytes(np.ascontiguousarray(ctx.fold_id(seeds), dtype=np.int64).tobytes()),
            "resolved_parameters": [describe(make_model(recipe, s, feature_columns(recipe)))
                                    for s in recipe["estimator_seeds"]],
            "estimator_seeds": list(recipe["estimator_seeds"]), "source_manifest_sha256": manifest["manifest_sha256"],
            "python": env["python"], "packages": packages_for(recipe), "requirements_sha256": env["requirements_sha256"],
            "parents": []}


def _common_metadata(ctx, manifest, invocation, task_id, stage, kind, seeds, config):
    return {"schema": banks.BANK_SCHEMA_V2, "task_id": task_id, "stage": stage, "prediction_kind": kind,
            "selection_eligible": sorted(int(s) for s in seeds) == sorted(SPLIT_SEEDS), "n_splits": N_SPLITS,
            "n_repeats": len(seeds), "split_seeds": [int(s) for s in seeds],
            "evaluation_population": banks.evaluation_population(ctx.data.row_ids),
            "fitting_population": FITTING_POPULATION, "data_hashes": ctx.data.hashes, "split_file": str(SPLIT_FILE),
            "split_sha256": ctx.split_sha256, "config": config, "config_sha256": banks.run_identity(config),
            "source_manifest": manifest, "environment": artifacts.environment(ctx.root), "invocation": invocation,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()}


def fitted_run(ctx, recipe, seeds, task_id, manifest, invocation, search_id=None):
    """Cross-validate one recipe on the requested split seeds into a bank with complete bank_v2 metadata."""
    seeds = banks.canonical_seeds(seeds)
    config = fitted_config(ctx, recipe, seeds, manifest)
    stage = banks.stage_of(seeds)
    t0 = time.time()
    bank = run_recipe_cv(recipe, ctx.rows, ctx.data.y, ctx.data.row_ids, ctx.groups, ctx.fold_id(seeds), list(seeds),
                         ctx.test_rows, ctx.data.test_id)
    elapsed = round(time.time() - t0, 1)
    kind = "fitted" if len(recipe["estimator_seeds"]) == 1 else "seed_average"
    bank.metadata.update(_common_metadata(ctx, manifest, invocation, task_id, stage, kind, seeds, config))
    bank.metadata.update({"run_id": banks.make_run_id(recipe["name"], stage, config), "recipe": recipe,
                          "recipe_id": recipe_id(recipe), "recipe_sha256": recipe_hash(recipe),
                          "estimator_seeds": list(recipe["estimator_seeds"]), "parents": [],
                          "search_id": search_id, "elapsed_seconds": elapsed})
    banks.validate_v2(bank)
    return bank


def blend_run(ctx, name, members, weights, task_id, manifest, invocation, seeds=None):
    """A declared blend of logged banks: members are (bank, bank_sha256) pairs, weights positive, summing to one.
    seeds, if given, restricts every member to those split seeds first, so a challenger screened on one split
    seed can be blended with a three-seed recipe on identical folds."""
    if seeds is not None:
        seeds = banks.canonical_seeds(seeds)
        members = [(select_seeds(b, seeds), sha) for b, sha in members]
    bank = blend_banks(members, weights)
    seeds = [int(s) for s in bank.split_seeds]
    parents = bank.metadata["parents"]
    env = artifacts.environment(ctx.root)
    config = {"kind": "blend", "parents": parents, "parent_seeds": seeds, "data_hashes": ctx.data.hashes,
              "evaluation_population": banks.evaluation_population(ctx.data.row_ids), "split_seeds": seeds,
              "source_manifest_sha256": manifest["manifest_sha256"], "python": env["python"],
              "packages": artifacts.packages()}
    stage = banks.stage_of(seeds)
    features, lineage = [], {}
    for b, _ in members:
        features += [f for f in b.metadata["features"] if f not in lineage]
        lineage.update(b.metadata["feature_lineage"])
    bank.metadata.update(_common_metadata(ctx, manifest, invocation, task_id, stage, "blend", seeds, config))
    bank.metadata.update({"run_id": banks.make_run_id(name, stage, config), "blend_name": name, "parents": parents,
                          "features": features, "feature_lineage": lineage, "elapsed_seconds": 0.0})
    banks.validate_v2(bank)
    return bank


def run_record(ctx, bank, bank_path, bank_sha256):
    md = bank.metadata
    era = era_of(ctx.data.train_eligible["track_album_release_date"].str[:4].astype(int).to_numpy())
    rec = {"record": "run", "run_id": md["run_id"], "task_id": md["task_id"], "stage": md["stage"],
           "prediction_kind": md["prediction_kind"], "selection_eligible": md["selection_eligible"],
           "recipe_id": md.get("recipe_id"), "family": md.get("recipe", {}).get("family"),
           "blend_name": md.get("blend_name"), "parents": md["parents"], "search_id": md.get("search_id"),
           "features": md["features"], "split_seeds": md["split_seeds"], "estimator_seeds": md.get("estimator_seeds"),
           "config_sha256": md["config_sha256"], "source_manifest_hash": md["source_manifest"]["manifest_sha256"],
           "n_rows": int(len(bank.y)), "n_groups": int(len(set(bank.group_id))), "n_features": len(md["features"]),
           "n_completed_fits": md["completed_fits"], **summarize(bank),
           "mean_fold_auc_unrounded": float(bank.fold_auc.mean()),
           "repeat_mean_auc_unrounded": bank.fold_auc.mean(axis=1).tolist(), "diagnostics": diagnostics(bank, era),
           "elapsed_seconds": md["elapsed_seconds"], "warnings": md["warnings"], "execution_status": "ok",
           "timestamp_utc": md["timestamp_utc"], "invocation": md["invocation"],
           "bank_path": str(bank_path), "bank_sha256": bank_sha256}
    return rec


def logged(ctx, record="run"):
    return [r for r in artifacts.read_experiments(ctx.root / LOG_FILE) if r.get("record") == record]


def save_run(ctx, bank):
    """Write the bank once and append its record; returns the record. The project sources loaded now, and their
    files on disk, must still be those the run started with: a module imported during the fit, or a file edited
    meanwhile, refuses the bank."""
    stored = bank.metadata["source_manifest"]
    if banks.source_manifest(ctx.root) != stored or artifacts.source_manifest(ctx.root, list(stored["files"])) != stored:
        raise SourceMismatch("project sources changed while the run was computed; the bank is not saved")
    path = BANK_DIR / f"{bank.metadata['run_id']}.npz"
    sha = banks.save_bank_v2(bank, ctx.root / path)
    rec = run_record(ctx, bank, path, sha)
    artifacts.append_experiment(ctx.root / LOG_FILE, rec)
    return rec


def load_logged(ctx, run_id):
    """A logged bank, its hash checked against the log; returns (bank, sha256, record)."""
    recs = [r for r in logged(ctx) if r["run_id"] == run_id]
    if len(recs) != 1:
        raise ValueError(f"run {run_id} has {len(recs)} log records, expected 1")
    rec = recs[0]
    raw = (ctx.root / rec["bank_path"]).read_bytes()
    sha = artifacts.sha256_bytes(raw)
    if sha != rec["bank_sha256"]:
        raise ValueError(f"bank {rec['bank_path']} differs from its logged hash")
    bank = banks.load_bank_v2(raw)
    if bank.metadata["run_id"] != run_id:
        raise ValueError(f"bank {rec['bank_path']} holds run {bank.metadata['run_id']}, not {run_id}")
    return bank, sha, rec


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def find_replay_commit(root, manifest):
    """The newest commit whose versions of every manifest file have the recorded hashes, or None."""
    try:
        commits = _git(root, "rev-list", "HEAD").decode().split()
    except (OSError, subprocess.CalledProcessError):
        return None
    for c in commits:
        try:
            if all(hashlib.sha256(_git(root, "show", f"{c}:{p}")).hexdigest() == h
                   for p, h in manifest["files"].items()):
                return c
        except subprocess.CalledProcessError:
            continue
    return None


def check_sources(ctx, stored, current, config=None):
    """Raise SourceMismatch, naming a replay commit when one exists, unless the stored manifest equals both the
    files on disk and the manifest of the code running now (current=None checks the files on disk only), and the
    recorded Python and package versions are the ones installed now."""
    on_disk = artifacts.source_manifest(ctx.root, list(stored["files"]))
    if on_disk != stored or (current is not None and current != stored):
        changed = sorted(set(stored["files"]) ^ set((current or stored)["files"]) |
                         {p for p in stored["files"] if on_disk["files"].get(p) != stored["files"][p]})
        commit = find_replay_commit(ctx.root, stored)
        where = f"it replays at commit {commit}" if commit else "no commit in this repository matches it"
        raise SourceMismatch(f"source manifest differs from the run's (files: {changed}); {where}")
    if config is not None:
        now = {"python": artifacts.environment(ctx.root)["python"],
               "packages": packages_for(config["recipe"]) if "recipe" in config else artifacts.packages()}
        diff = [k for k in ("python", "packages") if config.get(k) != now[k]]
        if diff:
            raise SourceMismatch(f"recorded {diff} differ from the installed ones: recorded "
                                 f"{ {k: config.get(k) for k in diff} }, now { {k: now[k] for k in diff} }")


def verify_run(ctx, run_id, manifest, invocation):
    """Recompute a logged run in memory and compare it with its bank and record; returns a list of problems.
    The bank's envelope is read first and its sources and dependencies compared, so a run made by other code
    raises SourceMismatch with its replay commit before current semantics are applied to it."""
    recs = [r for r in logged(ctx) if r["run_id"] == run_id]
    if len(recs) != 1:
        raise ValueError(f"run {run_id} has {len(recs)} log records, expected 1")
    raw = (ctx.root / recs[0]["bank_path"]).read_bytes()
    envelope = banks.load_bank_v2(raw, validate=False).metadata
    config = dict(envelope["config"], **({"recipe": envelope["recipe"]} if "recipe" in envelope else {}))
    check_sources(ctx, envelope["source_manifest"], manifest, config)
    saved, sha, rec = load_logged(ctx, run_id)
    md = saved.metadata
    if md["prediction_kind"] == "blend":
        members = [load_logged(ctx, p["run_id"])[:2] for p in md["parents"]]
        for (b, s), p in zip(members, md["parents"]):
            if s != p["bank_sha256"]:
                return [f"parent {p['run_id']} differs from its recorded hash"]
        bank = blend_run(ctx, md["blend_name"], members, [p["weight"] for p in md["parents"]], md["task_id"], manifest,
                         invocation, seeds=md["config"]["parent_seeds"])
    else:
        bank = fitted_run(ctx, md["recipe"], md["split_seeds"], md["task_id"], manifest, invocation, md.get("search_id"))
    bad = banks.compare_v2(bank, saved)
    fresh = run_record(ctx, bank, rec["bank_path"], sha)
    return bad + record_problems(rec, fresh)


ROUNDED = ("fold_auc", "repeat_mean_auc", "mean_fold_auc", "pooled_oof_auc", "repeat_averaged_oof_auc")
UNROUNDED = ("mean_fold_auc_unrounded", "repeat_mean_auc_unrounded")
VOLATILE = ("timestamp_utc", "elapsed_seconds", "invocation")  # legitimately differ on replay


def record_problems(rec, fresh):
    """Every field of a logged run record against the record rebuilt from the recomputed bank: the scores within
    their tolerances, the volatile fields of VOLATILE skipped, every other field exactly."""
    bad = []
    for k in sorted(set(rec) | set(fresh)):
        if k in VOLATILE:
            continue
        if k in ROUNDED or k in UNROUNDED:
            tol = 1e-6 if k in ROUNDED else banks.TOL
            ok = k in rec and k in fresh and np.allclose(np.asarray(rec[k], dtype=float),
                                                         np.asarray(fresh[k], dtype=float), rtol=0, atol=tol)
        else:
            ok = artifacts.dumps(rec.get(k)) == artifacts.dumps(fresh.get(k))
        if not ok:
            bad.append(f"logged {k}")
    return bad
