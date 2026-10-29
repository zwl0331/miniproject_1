"""Phase 2 runs, searches, blends and comparisons (doc/plans/phase2-models-features.md sections 4 and 5).

From the project root, with PYTHONDONTWRITEBYTECODE=1 and .venv/bin/python -m scripts.run_experiments:
    --recipe configs/phase2/recipes/<name>.json --task P2.x [--seeds 155 2026 1029]
    --search configs/phase2/searches/<name>.json --task P2.x [--seeds ...]
    --blend <name> --members RUN_ID:WEIGHT RUN_ID:WEIGHT ... --task P2.x [--seeds 155]   (members restricted
                                             to those split seeds; by default each member's own)
    --compare RUN_A RUN_B --task P2.x [--seeds 155]    (both banks restricted to those split seeds first)
    --verify --run-id RUN_ID [RUN_ID ...]    (recompute in memory, write nothing, compare)
    --verify-all                             (every run, comparison, search and submission in the phase 2 log)
Runs go to artifacts/phase2/bank/ and artifacts/phase2/experiments.jsonl, never to the phase 1 log. A run already
logged under the same run id is not run again. A verification whose sources differ from the run's exits 1 and
names the commit at which it replays.
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from songpop import artifacts
from songpop.metrics import BOOTSTRAP_SEED, paired_cluster_bootstrap
from songpop.phase2 import banks, runs, searches
from songpop.phase2.ensembles import select_seeds
from songpop.phase2.recipes import load_recipe, recipe_id, strict_loads
from songpop.splits import SPLIT_SEEDS

TOL = 1e-9  # verification tolerance for recomputed bootstrap results, as phase 1


def invocation(root):
    return {"argv": [sys.executable, "-m", "scripts.run_experiments", *sys.argv[1:]], "cwd": str(root),
            "env": {v: os.environ.get(v) for v in artifacts.THREAD_VARS + ["PYTHONDONTWRITEBYTECODE"]}}


def show(rec, note=""):
    print(f"  {rec['run_id']:60s} {rec['mean_fold_auc']:.4f}  repeats {rec['repeat_mean_auc']}  "
          f"({rec['elapsed_seconds']} s, {len(rec['warnings'])} warnings){note}")


def run_one(ctx, recipe, seeds, task, manifest, inv, search_id=None):
    config = runs.fitted_config(ctx, recipe, seeds, manifest)
    run_id = banks.make_run_id(recipe["name"], banks.stage_of(seeds), config)
    done = {r["run_id"]: r for r in runs.logged(ctx)}
    if run_id in done:
        show(done[run_id], "  already logged")
        return done[run_id]
    rec = runs.save_run(ctx, runs.fitted_run(ctx, recipe, seeds, task, manifest, inv, search_id))
    show(rec)
    return rec


def run_search(ctx, spec, seeds, task, manifest, inv, directory=runs.SEARCH_DIR, log_path=runs.LOG_FILE, runner=None):
    """Save the ordered, deduplicated configurations before any fit, log the search, then run them in that order.
    A failing configuration is logged as a run_failure record and the search stops: a failure is investigated,
    never skipped."""
    runner = runner or (lambda recipe, sid: run_one(ctx, recipe, seeds, task, manifest, inv, search_id=sid))
    sid, recipes, rel = searches.save_expansion(ctx.root, directory, spec)
    print(f"search {sid}: {len(recipes)} configurations, saved in {rel}")
    artifacts.append_experiment(ctx.root / log_path, {
        "record": "search", "task_id": task, "search_id": sid, "file": str(rel),
        "file_sha256": artifacts.sha256_bytes((ctx.root / rel).read_bytes()), "n_configurations": len(recipes),
        "recipe_ids": [recipe_id(r) for r in recipes], "split_seeds": seeds,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    for recipe in recipes:
        try:
            runner(recipe, sid)
        except Exception as e:
            artifacts.append_experiment(ctx.root / log_path, {
                "record": "run_failure", "task_id": task, "search_id": sid, "recipe_id": recipe_id(recipe),
                "error": f"{type(e).__name__}: {e}", "timestamp_utc": datetime.now(timezone.utc).isoformat()})
            raise
    return sid


def parse_members(items):
    out = []
    for it in items:
        run_id, _, w = it.rpartition(":")
        if not run_id:
            raise ValueError(f"member {it!r} must be RUN_ID:WEIGHT")
        out.append((run_id, float(w)))
    return out


def compare(ctx, a, b, task, seeds=None, verify=False, comparison_id=None):
    """The paired cluster bootstrap of a minus b on identical folds; with seeds, both banks are first restricted to
    those split seeds. A comparison is identified by both bank hashes, the seeds and the bootstrap settings, so
    running it again logs nothing new."""
    ba, sa, _ = runs.load_logged(ctx, a)
    bb, sb, _ = runs.load_logged(ctx, b)
    if seeds is not None:
        seeds = banks.canonical_seeds(seeds)
        ba, bb = select_seeds(ba, seeds), select_seeds(bb, seeds)
    used = [int(s) for s in ba.split_seeds]
    ident = {"a": a, "b": b, "a_sha256": sa, "b_sha256": sb, "seeds": used, "n_boot": 2000, "seed": BOOTSTRAP_SEED}
    cid = artifacts.config_hash(ident)[:16]
    res = paired_cluster_bootstrap(ba, bb, n_boot=ident["n_boot"], seed=ident["seed"])
    repeats = (ba.fold_auc.mean(axis=1) - bb.fold_auc.mean(axis=1)).tolist()
    line = (f"  {a} - {b} on seeds {used}: {res['observed_difference']:+.4f}  [{res['interval_95'][0]:+.4f}, "
            f"{res['interval_95'][1]:+.4f}]  {res['draws_positive_share']:.3f}  {res['valid']}/{res['n_boot']}")
    recs = [r for r in runs.logged(ctx, "comparison") if r.get("comparison_id") == cid]
    if verify:
        ok = cid == comparison_id and len(recs) == 1 \
            and all(recs[0][k] == res[k] for k in ("n_boot", "valid", "invalid", "seed", "groups")) \
            and all(abs(recs[0][k] - res[k]) <= TOL for k in ("observed_difference", "draws_positive_share")) \
            and np.allclose(recs[0]["interval_95"], res["interval_95"], rtol=0, atol=TOL) \
            and np.allclose(recs[0]["repeat_differences"], repeats, rtol=0, atol=TOL)
        print(line + ("  equals log" if ok else "  DIFFERS from log"))
        return ok
    if recs:
        print(line + "  already logged")
        return True
    artifacts.append_experiment(ctx.root / runs.LOG_FILE, {
        "record": "comparison", "comparison_id": cid, "task_id": task, **ident, **res, "repeat_differences": repeats,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    print(line)
    return True


def verify_searches(ctx):
    """Each logged search file has its recorded hash, and re-expanding its specification gives the same ordered
    recipes, with the same ids as the log."""
    ok = True
    for rec in runs.logged(ctx, "search"):
        path = ctx.root / rec["file"]
        saved = strict_loads(path.read_text())
        recipes = searches.expand(saved["spec"])
        good = artifacts.sha256_bytes(path.read_bytes()) == rec["file_sha256"] and saved["recipes"] == recipes \
            and [recipe_id(r) for r in recipes] == rec["recipe_ids"] and searches.search_id(saved["spec"]) == rec["search_id"]
        print(f"  search {rec['search_id']}: {'equals its file and log' if good else 'DIFFERS'}")
        ok &= good
    return ok


def verify_submissions(ctx):
    """Each logged submission's manifest, CSV and sidecar still have their recorded hashes."""
    ok = True
    for rec in runs.logged(ctx, "submission"):
        good = all(artifacts.sha256_bytes((ctx.root / rec[k]).read_bytes()) == rec[f"{k}_sha256"]
                   for k in ("manifest", "csv", "sidecar"))
        print(f"  submission {rec['csv']}: {'files equal their recorded hashes' if good else 'DIFFERS'}")
        ok &= good
    return ok


def verify(ctx, run_ids, manifest, inv):
    ok = True
    for run_id in run_ids:
        try:
            bad = runs.verify_run(ctx, run_id, manifest, inv)
        except (runs.SourceMismatch, ValueError) as e:
            print(f"  {run_id}: NOT VERIFIED: {type(e).__name__}: {e}")
            ok = False
            continue
        print(f"  {run_id}: {'equals saved bank and log' if not bad else f'DIFFERS: {bad}'}")
        ok &= not bad
    return ok


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--recipe")
    mode.add_argument("--search")
    mode.add_argument("--blend")
    mode.add_argument("--compare", nargs=2)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--verify-all", action="store_true")
    ap.add_argument("--members", nargs="+")
    ap.add_argument("--run-id", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--task")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    ctx = runs.Context.load(root)
    manifest, inv = banks.source_manifest(root), invocation(root)
    if not (args.verify or args.verify_all) and not args.task:
        ap.error("--task is required for anything that writes")
    seeds = banks.canonical_seeds(args.seeds or list(SPLIT_SEEDS))
    if args.recipe:
        run_one(ctx, load_recipe(root / args.recipe), seeds, args.task, manifest, inv)
    elif args.search:
        spec = searches.validate_search(strict_loads((root / args.search).read_text()))
        run_search(ctx, spec, seeds, args.task, manifest, inv)
    elif args.blend:
        members = parse_members(args.members or [])
        loaded = [runs.load_logged(ctx, r)[:2] for r, _ in members]
        bank = runs.blend_run(ctx, args.blend, loaded, [w for _, w in members], args.task, manifest, inv,
                              seeds=args.seeds)
        if bank.metadata["run_id"] in {r["run_id"] for r in runs.logged(ctx)}:
            print(f"  {bank.metadata['run_id']} already logged")
        else:
            show(runs.save_run(ctx, bank))
    elif args.compare:
        compare(ctx, *args.compare, args.task, seeds=args.seeds)
    else:
        if args.verify_all:
            ids = [r["run_id"] for r in runs.logged(ctx)]
            comparisons = runs.logged(ctx, "comparison")
        else:
            if not args.run_id:
                ap.error("--verify needs --run-id")
            ids, comparisons = args.run_id, []
        ok = verify(ctx, ids, manifest, inv)
        for r in comparisons:
            ok &= compare(ctx, r["a"], r["b"], None, seeds=r["seeds"], verify=True, comparison_id=r["comparison_id"])
        if args.verify_all:
            ok &= verify_searches(ctx)
            ok &= verify_submissions(ctx)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
