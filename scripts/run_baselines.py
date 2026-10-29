"""Day 1 baselines (task P1.2, doc/plans/phase1-data-baseline.md section 4).

From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.run_baselines --protocol legacy
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.run_baselines --protocol production
Add --verify to recompute in memory, write nothing, and compare with the saved split file, banks and log.

legacy: the scoping protocol of analysis/scoping.py (uri groups, split seeds 0 to 2, its date fields and its
estimator factories) run through this harness's folds, prediction and metric code; the result must equal
analysis/scoping_output.txt to four decimals. No bank is written; it is not selection-eligible.
production: the saved primary folds, the nine baselines (LightGBM, random forest, logistic on F0, F1, F2)
as banks, the diagnostics, and the paired comparisons of section 4.7.
"""
import argparse
import importlib.util
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from songpop import artifacts
from songpop.data import AUDIO, INCOMPLETE_ROW_IDS, load_dataset
from songpop.evaluate import BANK_SCHEMA, run_cv
from songpop.features import FEATURE_DEFINITION, FEATURE_SETS, make_features
from songpop.metrics import BOOTSTRAP_SEED, diagnostics, paired_cluster_bootstrap, summarize
from songpop.models import ESTIMATOR_SEED, MODELS, describe, make_estimator
from songpop.splits import SPLIT_SEEDS, SPLIT_VERSION, FoldSet, build_foldset, era_of, fold_report

LEGACY_SEEDS = (0, 1, 2)
LEGACY_MODELS = ["logreg_logfeat", "random_forest", "extra_trees", "hist_gb", "lgbm_default", "lgbm_slow",
                 "xgb_default"]
COMPARISONS = [(f"{m}_F1", f"{m}_F0") for m in MODELS] + [(f"{m}_F2", f"{m}_F1") for m in MODELS] + \
              [("random_forest_F2", "lightgbm_F2")]
DEPLOYMENT = "equal average of the test predictions of the 15 fold models (3 split seeds x 5 folds)"
SPLIT_FILE = Path("artifacts/splits/production_v1.csv")
BANK_DIR = Path("artifacts/bank")
LOG_FILE = Path("artifacts/experiments.jsonl")
TOL = 1e-9  # verification tolerance for recomputed metrics and bootstrap results


def sources(root):
    """The executing source files, relative to the project root."""
    return sorted(str(p.relative_to(root)) for p in (root / "songpop").glob("*.py")) + ["scripts/run_baselines.py"]


def invocation(root):
    return {"argv": [sys.executable, "-m", "scripts.run_baselines", *sys.argv[1:]], "cwd": str(root),
            "env": {v: os.environ.get(v) for v in artifacts.THREAD_VARS + ["PYTHONDONTWRITEBYTECODE"]}}


class LegacyAdapter:
    """Wraps a scoping factory's estimator so that classes_ is visible to predict_high; the scoping LogLR keeps
    it on its inner pipeline. Fitting and predictions are unchanged."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(X, y)
        self.classes_ = getattr(self.est, "classes_", None)
        if self.classes_ is None:
            self.classes_ = self.est.m.classes_
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(X)


def load_scoping(root):
    """Import analysis/scoping.py without letting its global warning filter leak into this process."""
    spec = importlib.util.spec_from_file_location("scoping", root / "analysis" / "scoping.py")
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        spec.loader.exec_module(mod)
    return mod


def expected_scoping(root):
    """The committed scoping results: {(model, 'audio'|'audio+date'): '0.xxxx'} and the block [8] value."""
    text = (root / "analysis" / "scoping_output.txt").read_text()
    exp = {}
    for m, full, audio in re.findall(r"^\s+(\w+)\s+audio\+date (\d\.\d{4}) .*?audio only (\d\.\d{4})", text, re.M):
        exp[(m, "audio+date")], exp[(m, "audio")] = full, audio
    exp[("lgbm_slow", "audio+year")] = re.search(r"\[8\] lgbm_slow, same folds: audio \+ year (\d\.\d{4})", text).group(1)
    return exp


def run_record(bank, s, **fields):
    """One experiment-log record of a run, the same fields for both protocols."""
    md = bank.metadata
    rec = {"record": "run", "run_id": md["run_id"], "recipe_id": md["recipe_id"], "task_id": "P1.2",
           "protocol": md["protocol"], "selection_eligible": md["selection_eligible"], "model": md["model"],
           "feature_set": md["feature_set"], "parameters": md["parameters"], "preprocessing": md["preprocessing"],
           "source_manifest_hash": md["source_manifest"]["manifest_sha256"], "data_hashes": md["data_hashes"],
           "split": md["split"], "split_seeds": md["split_seeds"], "estimator_seeds": [md["estimator_seed"]],
           "deployment": md["deployment"], "n_rows": int(len(bank.y)), "n_groups": int(len(set(bank.group_id))),
           "n_features": len(md["features"]), "n_completed_fits": md["completed_fits"], **s,
           "elapsed_seconds": md["elapsed_seconds"], "warnings": md["warnings"], "execution_status": "ok",
           "timestamp_utc": md["timestamp_utc"]}
    rec.update(fields)
    return rec


def legacy(root, data, verify):
    sc = load_scoping(root)
    tr = data.train_eligible
    y, groups = data.y, tr["uri"].to_numpy()
    if not np.array_equal(data.row_ids, np.array([i for i in range(len(data.train)) if i not in INCOMPLETE_ROW_IDS])):
        raise AssertionError("eligible rows are not the scoping rows")
    fold_id = np.full((len(LEGACY_SEEDS), len(tr)), -1)
    for r, seed in enumerate(LEGACY_SEEDS):
        for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, y, groups)):
            fold_id[r, va] = k
    with warnings.catch_warnings():
        frames, tests = {}, {}
        for name, frame in [("train", tr), ("test", data.test)]:
            d = sc.date_feats(frame)
            out = frames if name == "train" else tests
            out["audio"] = frame[AUDIO].copy()
            out["audio+date"] = pd.concat([frame[AUDIO], d], axis=1)
            out["audio+year"] = pd.concat([frame[AUDIO], d[["year"]]], axis=1)
    exp = expected_scoping(root)
    reference = {"file": "analysis/scoping_output.txt",
                 "sha256": artifacts.sha256_bytes((root / "analysis/scoping_output.txt").read_bytes())}
    runs = [(m, f) for m in LEGACY_MODELS for f in ("audio", "audio+date")] + [("lgbm_slow", "audio+year")]
    manifest = artifacts.source_manifest(root, sources(root) + ["analysis/scoping.py"])
    print(f"reference {reference['file']} sha256 {reference['sha256']}")
    print("legacy reproduction through the harness: model, features, mean fold AUC, committed value, match")
    mismatches = 0
    for model, feats in runs:
        t0 = time.time()
        bank = run_cv(frames[feats], y, data.row_ids, groups, fold_id, LEGACY_SEEDS, tests[feats],
                      data.test_id, lambda: LegacyAdapter(sc.MODELS[model]()))
        got = f"{bank.fold_auc.mean():.4f}"
        ok = got == exp[(model, feats)]
        mismatches += not ok
        print(f"  {model:15s} {feats:11s} {got}  {exp[(model, feats)]}  {'ok' if ok else 'MISMATCH'}  "
              f"({time.time() - t0:.0f} s)")
        bank.metadata.update({
            "run_id": f"legacy_{model}_{feats}_{manifest['manifest_sha256'][:10]}", "recipe_id": f"{model}_{feats}",
            "protocol": "legacy", "selection_eligible": False, "model": model, "feature_set": feats,
            "features": list(frames[feats].columns), "parameters": describe(sc.MODELS[model]()),
            "preprocessing": "as in analysis/scoping.py", "source_manifest": manifest, "data_hashes": data.hashes,
            "split": {"grouping": "uri", "seeds": list(LEGACY_SEEDS)}, "split_seeds": list(LEGACY_SEEDS),
            "estimator_seed": 0, "deployment": "none (no bank)", "elapsed_seconds": round(time.time() - t0, 1),
            "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        if not verify:
            artifacts.append_experiment(root / LOG_FILE, run_record(
                bank, summarize(bank), reference=reference, expected_mean_fold_auc=exp[(model, feats)],
                reproduced=ok, bank_path=None, bank_sha256=None, invocation=invocation(root)))
    print(f"legacy reproduction: {len(runs) - mismatches} of {len(runs)} match")
    return mismatches == 0


def production(root, data, verify):
    fs = build_foldset(data)
    split_path = root / SPLIT_FILE
    if split_path.exists():
        if not fs.equals(FoldSet.from_frame(pd.read_csv(split_path))):
            raise AssertionError(f"recomputed folds differ from {SPLIT_FILE}")
        print(f"split file {SPLIT_FILE} equals the recomputed folds")
    elif verify:
        raise FileNotFoundError(f"{SPLIT_FILE} missing; run without --verify first")
    else:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "x") as f:
            fs.to_frame().to_csv(f, index=False, lineterminator="\n")
        print(f"split file {SPLIT_FILE} written")
    split_hash = artifacts.sha256_bytes(split_path.read_bytes())
    e = fs.eligible
    print(f"split file sha256 {split_hash}; groups among eligible rows: primary {len(set(fs.primary_group[e]))}, "
          f"strict {len(set(fs.strict_group[e]))}; incomplete row folds {fs.primary[:, ~e].ravel().tolist()}")
    for kind in ("primary", "strict"):
        table, warns = fold_report(data, fs, kind)
        va = table[table["side"] == "validation"]
        print(f"{kind} folds{' (reporting only, never for selection)' if kind == 'strict' else ''}: validation rows "
              f"{va['rows'].min()} to {va['rows'].max()}, High rate {va['high_rate'].min()} to {va['high_rate'].max()}, "
              f"mixed-label groups {va['mixed_groups'].min()} to {va['mixed_groups'].max()}; warnings: {warns or 'none'}")
        print(table.to_string(index=False))

    manifest = artifacts.source_manifest(root, sources(root))
    env = artifacts.environment(root)
    log = artifacts.read_experiments(root / LOG_FILE)
    logged_runs = {r["run_id"]: r for r in log if r.get("record") == "run"}
    logged_cmp = {(r["a"], r["b"]): r for r in log if r.get("record") == "comparison"}
    fold_id, groups = fs.primary[:, e], fs.primary_group[e]
    era = era_of(data.train_eligible["track_album_release_date"].str[:4].astype(int).to_numpy())
    banks, problems = {}, []
    print("production baselines: recipe, mean fold AUC, repeat means, pooled OOF AUC per repeat, group-weighted")
    for model in MODELS:
        for fset in FEATURE_SETS:
            recipe = f"{model}_{fset}"
            X, X_test = make_features(data.train_eligible, fset), make_features(data.test, fset)
            config = {"source_manifest": manifest["manifest_sha256"], "data_hashes": data.hashes,
                      "split_version": SPLIT_VERSION, "split_sha256": split_hash, "grouping": "primary",
                      "feature_definition": FEATURE_DEFINITION, "feature_set": fset, "features": list(X.columns),
                      "parameters": describe(make_estimator(model)), "split_seeds": list(SPLIT_SEEDS),
                      "estimator_seed": ESTIMATOR_SEED, "deployment": DEPLOYMENT, "packages": env["packages"],
                      "requirements_sha256": env["requirements_sha256"]}
            chash = artifacts.config_hash(config)
            run_id = f"production_{recipe}_{chash[:12]}"
            t0 = time.time()
            bank = run_cv(X, data.y, data.row_ids, groups, fold_id, SPLIT_SEEDS, X_test, data.test_id,
                          lambda: make_estimator(model))
            bank.metadata.update({
                "schema": BANK_SCHEMA, "run_id": run_id, "recipe_id": recipe, "task_id": "P1.2",
                "protocol": "production", "selection_eligible": True, "model": model, "feature_set": fset,
                "feature_definition": FEATURE_DEFINITION, "features": list(X.columns),
                "parameters": config["parameters"], "preprocessing": "log transforms, imputation and scaling inside "
                "the logistic pipeline, fitted per fold" if model == "logistic" else "none",
                "data_hashes": data.hashes, "split_version": SPLIT_VERSION, "split_file": str(SPLIT_FILE),
                "split_sha256": split_hash, "grouping": "primary", "excluded_row_ids": list(INCOMPLETE_ROW_IDS),
                "split": {"grouping": "primary", "file": str(SPLIT_FILE), "sha256": split_hash},
                "split_seeds": list(SPLIT_SEEDS), "estimator_seed": ESTIMATOR_SEED, "deployment": DEPLOYMENT,
                "source_manifest": manifest, "config": config, "config_sha256": chash, "environment": env,
                "invocation": invocation(root), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.time() - t0, 1)})
            s, diag = summarize(bank), diagnostics(bank, era)
            path = root / BANK_DIR / f"{run_id}.npz"
            if verify:
                saved = artifacts.load_bank(path)
                bad = artifacts.compare_banks(bank, saved)
                rec = logged_runs.get(run_id)
                if rec is None:
                    bad.append("no log record")
                else:
                    if rec["bank_sha256"] != artifacts.sha256_bytes(path.read_bytes()):
                        bad.append("bank file hash differs from the log")
                    for k, v in s.items():
                        if not np.allclose(np.asarray(rec[k], dtype=float), np.asarray(v, dtype=float), rtol=0, atol=1e-6):
                            bad.append(f"logged {k}")
                    if artifacts.dumps(rec["diagnostics"]) != artifacts.dumps(diag):
                        bad.append("logged diagnostics")
                for k in ("python", "platform", "machine"):
                    if saved.metadata["environment"][k] != env[k]:
                        print(f"  note: {recipe} environment {k} differs: saved {saved.metadata['environment'][k]}, "
                              f"now {env[k]}")
                if bad:
                    problems.append(f"{recipe}: {bad}")
                status = "equals saved bank and log" if not bad else f"DIFFERS: {bad}"
                print(f"  {recipe:18s} {s['mean_fold_auc']:.4f}  {s['repeat_mean_auc']}  {s['pooled_oof_auc']}  "
                      f"{diag['group_weighted_mean_fold_auc']:.4f}  {status} (max OOF difference "
                      f"{float(np.max(np.abs(bank.oof - saved.oof))):.1e})")
            else:
                digest = artifacts.save_bank(bank, path)
                artifacts.append_experiment(root / LOG_FILE, run_record(
                    bank, s, diagnostics=diag, config_sha256=chash, split_sha256=split_hash,
                    bank_path=str(BANK_DIR / path.name), bank_sha256=digest, invocation=bank.metadata["invocation"]))
                print(f"  {recipe:18s} {s['mean_fold_auc']:.4f}  {s['repeat_mean_auc']}  {s['pooled_oof_auc']}  "
                      f"{diag['group_weighted_mean_fold_auc']:.4f}  ({bank.metadata['elapsed_seconds']} s, "
                      f"{len(bank.metadata['warnings'])} warnings)")
            if fset == "F2":
                eras = "; ".join(f"{k} {v['pooled_oof_auc_by_repeat'] or v['note']}" for k, v in
                                 diag["era_pooled_oof_auc"].items())
                print(f"    per-era pooled OOF AUC by repeat (diagnostic): {eras}")
            banks[recipe] = bank
    print("paired cluster bootstrap, a minus b: observed difference in mean fold AUC, 95% interval, "
          "share of draws above 0, valid draws")
    for a, b in COMPARISONS:
        res = paired_cluster_bootstrap(banks[a], banks[b])
        key = (banks[a].metadata["run_id"], banks[b].metadata["run_id"])
        line = (f"  {a:18s} - {b:18s} {res['observed_difference']:+.4f}  [{res['interval_95'][0]:+.4f}, "
                f"{res['interval_95'][1]:+.4f}]  {res['draws_positive_share']:.3f}  {res['valid']}/{res['n_boot']}")
        if verify:
            rec = logged_cmp.get(key)
            ok = rec is not None and all(rec[k] == res[k] for k in ("n_boot", "valid", "invalid", "seed", "groups")) \
                and all(abs(rec[k] - res[k]) <= TOL for k in ("observed_difference", "draws_positive_share")) \
                and np.allclose(rec["interval_95"], res["interval_95"], rtol=0, atol=TOL)
            if not ok:
                problems.append(f"comparison {a} - {b} differs from the log")
            line += "  equals log" if ok else "  DIFFERS from log"
        else:
            artifacts.append_experiment(root / LOG_FILE, {
                "record": "comparison", "task_id": "P1.2", "a": key[0], "b": key[1], "a_recipe": a, "b_recipe": b,
                **res, "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        print(line)
    # P1.3 rule (phase plan section 5): highest unrounded mean fold AUC on F2; a tie within 1e-12 goes to the
    # order LightGBM, random forest, logistic.
    f2 = [f"{m}_F2" for m in MODELS]
    top = max(float(banks[r].fold_auc.mean()) for r in f2)
    best = next(r for r in f2 if float(banks[r].fold_auc.mean()) >= top - 1e-12)
    print(f"highest mean fold AUC on F2 (the P1.3 rule): {best} {float(banks[best].fold_auc.mean()):.10f} "
          f"run {banks[best].metadata['run_id']}")
    if problems:
        print("verification problems:\n  " + "\n  ".join(problems))
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["legacy", "production"], required=True)
    ap.add_argument("--verify", action="store_true", help="recompute in memory, write nothing, compare")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    data = load_dataset(root)
    ok = legacy(root, data, args.verify) if args.protocol == "legacy" else production(root, data, args.verify)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
