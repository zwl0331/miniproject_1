"""Phase 2 runs, searches, blends and comparisons (doc/plans/phase2-models-features.md sections 4 and 5).

From the project root, with PYTHONDONTWRITEBYTECODE=1 and .venv/bin/python -m scripts.run_experiments:
    --recipe configs/phase2/recipes/<name>.json --task P2.x [--seeds 155 2026 1029]
    --search configs/phase2/searches/<name>.json --task P2.x [--seeds ...]
    --blend <name> --members RUN_ID:WEIGHT RUN_ID:WEIGHT ... --task P2.x [--seeds 155]   (members restricted
                                             to those split seeds; by default each member's own)
    --compare RUN_A RUN_B --task P2.x [--seeds 155]    (both banks restricted to those split seeds first)
    --verify --run-id RUN_ID [RUN_ID ...]    (recompute in memory, write nothing, compare)
    --verify --task P2.x                     (every run and comparison the log records for that task)
    --verify-all                             (every run, comparison, search and submission in the phase 2 log)
Runs go to artifacts/phase2/bank/ and artifacts/phase2/experiments.jsonl, never to the phase 1 log. A run already
logged under the same run id is not run again. A verification whose sources differ from the run's exits 1 and
names the commit at which it replays.
"""
import argparse
import copy
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from songpop import artifacts
from songpop.metrics import BOOTSTRAP_SEED, paired_cluster_bootstrap
from songpop.phase2 import allocation, banks, batches, feature_stage, protocol, runs, searches
from songpop.phase2.encoders import FitContext
from songpop.phase2.engine import fit_recipe
from songpop.phase2.ensembles import blend_banks, select_seeds
from songpop.phase2.recipes import load_recipe, recipe_id, strict_loads, validate_recipe
from songpop.splits import SPLIT_SEEDS

TOL = 1e-9  # verification tolerance for recomputed bootstrap results, as phase 1


def invocation(root):
    return {"argv": [sys.executable, "-m", "scripts.run_experiments", *sys.argv[1:]], "cwd": str(root),
            "env": {v: os.environ.get(v) for v in artifacts.THREAD_VARS + ["PYTHONDONTWRITEBYTECODE"]}}


def show(rec, note=""):
    print(f"  {rec['run_id']:60s} {rec['mean_fold_auc']:.4f}  repeats {rec['repeat_mean_auc']}  "
          f"({rec['elapsed_seconds']} s, {len(rec['warnings'])} warnings){note}")


def run_one(ctx, recipe, seeds, task, manifest, inv, search_id=None, quiet_scores=False):
    """Run one recipe unless its run id is logged; a logged run is reused only after its bank is checked against
    its logged hash. A bank finalized without its log record (an interrupted append) is recovered once, after its
    identity, sources and arrays are validated, instead of being refitted."""
    config = runs.fitted_config(ctx, recipe, seeds, manifest)
    run_id = banks.make_run_id(recipe["name"], banks.stage_of(seeds), config)
    done = {r["run_id"]: r for r in runs.logged(ctx)}
    if run_id in done:
        runs.load_logged(ctx, run_id)  # raises if the bank differs from its logged hash
        show(done[run_id], "  already logged")
        return done[run_id]
    path = runs.BANK_DIR / f"{run_id}.npz"
    if (ctx.root / path).exists():
        raw = (ctx.root / path).read_bytes()
        bank = banks.load_bank_v2(raw)
        md = bank.metadata
        if md["run_id"] != run_id or md["config_sha256"] != banks.run_identity(config) \
                or md["source_manifest"] != manifest or md["recipe_sha256"] != searches.recipe_hash(recipe):
            raise ValueError(f"{path} exists but is not this run's bank; inspect it before anything else")
        rec = runs.run_record(ctx, bank, path, artifacts.sha256_bytes(raw))
        artifacts.append_experiment(ctx.root / runs.LOG_FILE, rec)
        show(rec, "  recovered from its finalized bank")
        return rec
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


TIMING_SCHEMA = "timing_v1"
TIMING_CASES = ("light", "representative", "expensive")
OPTIONAL_TIMING_CASES = ("extended",)  # beyond the space's timed corner (P2.3's range extensions); not averaged
PROTOCOLS = {"P2.2": Path("configs/phase2/p2.2/protocol.json"),  # plan section 15
             "P2.3": Path("configs/phase2/p2.3/protocol.json"),  # plan section 17
             "P2.5": Path("configs/phase2/p2.5/protocol.json")}  # plan section 19
BATCH_DIR = runs.ARTIFACTS / "batches"


def time_recipe(ctx, recipe, seeds):
    """Fit and predict exactly as a cross-validated run does (every repeat and fold: fit on the training part,
    predict the held-out part and the test rows, build the model inputs of the duplicate check), timing it;
    nothing is scored, so no result is exposed. Returns (elapsed seconds, completed fits, warnings)."""
    recipe, seeds = validate_recipe(recipe), banks.canonical_seeds(seeds)
    fold_id = ctx.fold_id(seeds)
    y, groups, rows, test = ctx.data.y, ctx.groups, ctx.rows, ctx.test_rows
    caught = set()
    t0 = time.perf_counter()
    for r, seed in enumerate(seeds):
        for k in range(5):
            tr, va = np.flatnonzero(fold_id[r] != k), np.flatnonzero(fold_id[r] == k)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fitted = fit_recipe(recipe, rows.iloc[tr], y[tr], groups[tr], FitContext(int(seed), k))
                fitted.predict_high(rows.iloc[va])
                fitted.predict_each(test)
                fitted.model_inputs(test)
            caught |= {f"{type(x.message).__name__}: {x.message}" for x in w}
    return round(time.perf_counter() - t0, 2), len(seeds) * 5 * len(recipe["estimator_seeds"]), sorted(caught)


def time_cases(ctx, spec, seeds, task, manifest, inv, spec_sha256, log_path=runs.LOG_FILE):
    """Time each declared case (plan section 5.1) and log a timing record: elapsed seconds, fits, warnings and
    their provenance, never a score. The sources must be unchanged when each record is written."""
    if not isinstance(spec, dict) or spec.get("schema") != TIMING_SCHEMA or not spec.get("cases"):
        raise ValueError(f"a timing specification has schema {TIMING_SCHEMA!r} and a non-empty list of cases")
    seeds = banks.canonical_seeds(seeds)
    env = artifacts.environment(ctx.root)
    out = []
    for case in spec["cases"]:
        if sorted(case) != ["case", "recipe"] or case["case"] not in TIMING_CASES + OPTIONAL_TIMING_CASES:
            raise ValueError(f"malformed timing case {case!r}")
        recipe = validate_recipe(case["recipe"])
        load = [round(x, 2) for x in os.getloadavg()]
        elapsed, fits, caught = time_recipe(ctx, recipe, seeds)
        if banks.source_manifest(ctx.root) != manifest:
            raise runs.SourceMismatch("project sources changed while timing; the record is not written")
        rec = {"record": "timing", "task_id": task, "family": recipe["family"], "case": case["case"],
               "recipe": recipe, "recipe_id": recipe_id(recipe), "timing_spec_sha256": spec_sha256,
               "split_seeds": seeds, "fold_ids_sha256": artifacts.sha256_bytes(
                   np.ascontiguousarray(ctx.fold_id(seeds), dtype=np.int64).tobytes()),
               "data_hashes": ctx.data.hashes, "completed_fits": fits, "elapsed_seconds": elapsed,
               "warnings": caught, "source_manifest": manifest, "python": env["python"],
               "packages": runs.packages_for(recipe), "threads": env["threads"], "machine": platform.machine(),
               "cpu_count": os.cpu_count(), "load_average_before": load, "invocation": inv, "timestamp_utc": now()}
        artifacts.append_experiment(ctx.root / log_path, rec)
        print(f"  {recipe['family']:14s} {case['case']:15s} {elapsed:8.2f} s  {fits} fits  "
              f"{len(caught)} warnings  {recipe_id(recipe)}")
        out.append(rec)
    return out


def timing_set(ctx, task, manifest):
    """The task's timing records made under the running sources, one per (family, case), all from one timing
    specification; older records stay in the log as history. Raises on duplicates or a mixture of specifications."""
    t = [r for r in runs.logged(ctx, "timing") if r["task_id"] == task
         and r["source_manifest"]["manifest_sha256"] == manifest["manifest_sha256"]]
    specs = sorted({r["timing_spec_sha256"] for r in t})
    if len(specs) != 1:
        raise ValueError(f"{task} needs timing records from exactly one specification under the running sources, "
                         f"found {len(specs)}")
    out = {}
    for r in t:
        k = (r["family"], r["case"])
        if k in out:
            raise ValueError(f"duplicate timing records for {k} under the running sources")
        out[k] = r
    return out


def reservation(ctx, task, proto, psha, manifest):
    """The compute reservation of plan sections 15.6 and 17.5 from the timing set of the running sources (one record
    per family and case): per family, the mean of its light, representative and expensive cases times its initial
    count and its ceiling; an extended case, if timed, is reported and not averaged. The record names the timing
    records and the protocol hash. A projection, not an upper bound."""
    t = timing_set(ctx, task, manifest)
    rows, init, ceil, used = [], 0.0, 0.0, []
    for fam in proto["families"]:
        if not fam["space"]:
            continue  # a reference family (P2.3's LightGBM): its anchor H alone, one run of a few seconds
        cases = {c: t.get((fam["family"], c)) for c in TIMING_CASES + OPTIONAL_TIMING_CASES}
        missing = [c for c in TIMING_CASES if cases[c] is None]
        if missing:
            raise ValueError(f"{fam['family']} lacks the timing cases {missing}")
        secs = [cases[c]["elapsed_seconds"] for c in TIMING_CASES]
        mean = sum(secs) / len(secs)
        ext = cases["extended"]
        rows.append({"family": fam["family"], "mean_case_seconds": round(mean, 2), "max_case_seconds": max(secs),
                     "extended_case_seconds": ext["elapsed_seconds"] if ext else None,
                     "initial_minutes": round(mean * fam["initial"] / 60, 1),
                     "ceiling_minutes": round(mean * fam["ceiling"] / 60, 1)})
        used += [{"family": r["family"], "case": r["case"], "recipe_id": r["recipe_id"],
                  "elapsed_seconds": r["elapsed_seconds"], "timestamp_utc": r["timestamp_utc"]}
                 for r in cases.values() if r is not None]
        init, ceil = init + mean * fam["initial"], ceil + mean * fam["ceiling"]
    spec = next(iter(t.values()))["timing_spec_sha256"]
    rec = {"record": "reservation", "task_id": task, "families": rows,
           "initial_minutes": round(init / 60, 1), "ceiling_minutes": round(ceil / 60, 1),
           "full_refit_minutes": "projected to repeat the fitting actually completed; the ceiling projection is not "
                                 "a runtime bound",
           "bootstraps": "2,000 replicates per decision-driving comparison, a few seconds each",
           "review": "two Codex review checkpoints, about 30 to 60 minutes each",
           "note": "a projection from three timed cases per family, not an upper bound",
           "protocol_sha256": psha, "timing_spec_sha256": spec, "source_manifest_sha256": manifest["manifest_sha256"],
           "timing_records": used, "timestamp_utc": now()}
    artifacts.append_experiment(ctx.root / runs.LOG_FILE, rec)
    for r in rows:
        print(f"  {r['family']:14s} mean case {r['mean_case_seconds']:7.2f} s  initial {r['initial_minutes']:6.1f} min  "
              f"ceiling {r['ceiling_minutes']:6.1f} min  extended case {r['extended_case_seconds']} s")
    print(f"  all families: initial {rec['initial_minutes']} min, ceilings {rec['ceiling_minutes']} min; "
          f"the full refit adds the fitting done")
    return rec


def contribution_key(rec):
    return (rec["run_id"], rec["bank_sha256"], rec["h_run_id"], rec["h_sha256"], rec["weight"])


def contributions(ctx, run_id, h_run_id, task, verify=False):
    """The two declared contribution blends of plan section 5.6, (1 - w) H + w c for w in
    allocation.CONTRIBUTION_WEIGHTS, computed in memory from the two logged banks and logged as records (no bank
    is saved: each is a deterministic function of two logged banks). With verify, recompute and compare instead."""
    c, csha, _ = runs.load_logged(ctx, run_id)
    h, hsha, _ = runs.load_logged(ctx, h_run_id)
    if [int(x) for x in c.split_seeds] != list(SPLIT_SEEDS) or [int(x) for x in h.split_seeds] != list(SPLIT_SEEDS):
        raise ValueError("contribution blends use complete banks on the official split seeds only")
    have = {contribution_key(r): r for r in runs.logged(ctx, "contribution")}
    ok, out = True, []
    for w in allocation.CONTRIBUTION_WEIGHTS:
        if run_id == h_run_id:  # H blended with itself is H: an explicit identity, not a two-member blend
            res = {"mean_fold_auc": float(h.fold_auc.mean()), "repeat_mean_auc": allocation.repeat_means(h).tolist()}
        else:
            res = allocation.contribution(h, hsha, c, csha, w)
        rec = {"record": "contribution", "task_id": task, "run_id": run_id, "bank_sha256": csha, "h_run_id": h_run_id,
               "h_sha256": hsha, "weight": w, "split_seeds": [int(x) for x in c.split_seeds],
               "identity": run_id == h_run_id, "mean_fold_auc_unrounded": res["mean_fold_auc"],
               "repeat_mean_auc_unrounded": res["repeat_mean_auc"]}
        old = have.get(contribution_key(rec))
        if verify:
            good = old is not None and abs(old["mean_fold_auc_unrounded"] - rec["mean_fold_auc_unrounded"]) <= banks.TOL \
                and np.allclose(old["repeat_mean_auc_unrounded"], rec["repeat_mean_auc_unrounded"], rtol=0, atol=banks.TOL)
            ok &= good
        elif old is None:
            artifacts.append_experiment(ctx.root / runs.LOG_FILE,
                                        dict(rec, timestamp_utc=datetime.now(timezone.utc).isoformat()))
        out.append(rec)
    return ok if verify else out


def read_manifest(ctx, rel):
    return strict_loads((ctx.root / rel).read_text())


def load_log(ctx, task):
    """The task's log index; a task with a wave protocol binds that protocol's prior run, if it names one."""
    prior = load_protocol(ctx, task)[0].get("prior_strongest") if task in PROTOCOLS else None
    return protocol.Log(artifacts.read_experiments(ctx.root / runs.LOG_FILE), task, lambda rel: read_manifest(ctx, rel),
                        prior)


PRIOR_ARRAYS = ("train_row_id", "y", "group_id", "split_seeds", "fold_id", "test_id")


def validate_prior(ctx, proto):
    """Plan section 17.3: the protocol's prior member of G, read from its bank and not only from its log entry: the
    bank's bytes against the protocol's hash; its record's task and recipe; a complete population whose rows, labels,
    groups, split seeds, folds and test ids equal the P2.1 anchor bank's; and runs.audit_run, which rebuilds the
    record's scores from the stored predictions. Raises before anything is planned, fitted, judged or audited. A
    protocol without a prior member passes."""
    prior = proto.get("prior_strongest")
    if prior is None:
        return
    bank, sha, rec = runs.load_logged(ctx, prior["run_id"])
    ref_bank = runs.load_logged(ctx, proto["reference_p21_run_id"])[0]
    problems = [k for k, ok in {
        "bank hash": sha == prior["bank_sha256"], "task": rec["task_id"] == prior["task_id"],
        "recipe": bank.metadata["recipe_sha256"] == searches.recipe_hash(prior["recipe"]),
        "complete": protocol.complete(rec),
        **{a: bool(np.array_equal(getattr(bank, a), getattr(ref_bank, a))) for a in PRIOR_ARRAYS}}.items() if not ok]
    problems += runs.audit_run(ctx, prior["run_id"])
    if problems:
        raise ValueError(f"the prior member of G, {prior['run_id']}, fails its validation: {problems}")


def run_id_of(ctx, seeds, manifest):
    return lambda r: banks.make_run_id(r["name"], banks.stage_of(seeds), runs.fitted_config(ctx, r, seeds, manifest))


def past_cutoff(proto):
    return datetime.now(timezone.utc) > datetime.fromisoformat(proto["cutoff_utc"])


def load_protocol(ctx, task):
    """The task's protocol file and its hash; a task without one has no waves."""
    if task not in PROTOCOLS:
        raise ValueError(f"task {task} has no wave protocol (known: {sorted(PROTOCOLS)})")
    raw = (ctx.root / PROTOCOLS[task]).read_bytes()
    proto = protocol.load_protocol(ctx.root / PROTOCOLS[task])
    if proto["task_id"] != task:
        raise ValueError(f"the protocol file of {task} belongs to task {proto['task_id']}")
    return proto, artifacts.sha256_bytes(raw)


def full_manifest(m, task, psha, manifest):
    return {"schema": batches.BATCH_SCHEMA, "task_id": task, "protocol_sha256": psha,
            "source_manifest_sha256": manifest["manifest_sha256"], "family": m["family"], "wave": m["wave"],
            "kind": m["kind"], "recipes": m["recipes"], "recipe_sha256": [searches.recipe_hash(r) for r in m["recipes"]],
            "run_ids": m["run_ids"], "state": m["state"], "sources": m["sources"]}


def protocol_seeds(proto, seeds):
    if list(seeds) != proto["split_seeds"]:
        raise ValueError(f"the {proto['task_id']} protocol runs on exactly the split seeds {proto['split_seeds']}, "
                         f"not {list(seeds)}")


def batch_record(m, bid, rel, fsha, task, psha):
    return {"record": "batch", "task_id": task, "batch_id": bid, "family": m["family"], "wave": m["wave"],
            "kind": m["kind"], "file": str(rel), "file_sha256": fsha, "run_ids": m["run_ids"], "state": m["state"],
            "sources": m["sources"], "n_configurations": len(m["recipes"]), "protocol_sha256": psha}


STABLE_BATCH_FIELDS = ("batch_id", "family", "wave", "kind", "file", "file_sha256", "run_ids", "state", "sources",
                       "n_configurations", "protocol_sha256")


def plan_wave(ctx, wave, seeds, task, manifest):
    """Save and log the manifests of a wave (plan section 15), before any of its fits. Resumable: the wave is
    re-planned deterministically; manifests already saved or logged must equal the re-planning and are reused,
    and only missing ones are written and logged."""
    proto, psha = load_protocol(ctx, task)
    protocol_seeds(proto, seeds)
    validate_prior(ctx, proto)
    log = load_log(ctx, task)
    if wave > 1 and not any(b["wave"] == wave - 1 for b in log.batches):
        raise ValueError(f"wave {wave - 1} is not planned")
    logged = {b["family"]: b for b in log.batches if b["wave"] == wave}
    if past_cutoff(proto) and not logged:
        raise ValueError(f"the {task} cutoff {proto['cutoff_utc']} has passed: no new batch")
    planned = protocol.plan_wave(proto, log, wave, run_id_of(ctx, seeds, manifest))
    if set(logged) - {m["family"] for m in planned}:
        raise ValueError(f"wave {wave} has logged batches the re-planning does not produce: {sorted(logged)}")
    for m in planned:
        full = full_manifest(m, task, psha, manifest)
        bid, rel, fsha = batches.save_batch(ctx.root, BATCH_DIR, full)  # an existing file must be identical
        rec = batch_record(m, bid, rel, fsha, task, psha)
        if m["family"] in logged:
            if any(artifacts.dumps(logged[m["family"]].get(k)) != artifacts.dumps(rec[k]) for k in STABLE_BATCH_FIELDS):
                raise ValueError(f"the logged {m['family']} batch of wave {wave} differs from its re-planning")
            print(f"  {bid}: already planned")
            continue
        artifacts.append_experiment(ctx.root / runs.LOG_FILE, dict(rec, timestamp_utc=now()))
        print(f"  {bid}: {len(m['recipes'])} configurations, saved in {rel}")
    if not planned:
        print(f"  wave {wave}: no family has allowance left below its ceiling; {task}'s search is complete")
    return planned


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(ctx, wave, seeds, task, manifest, proto, psha, log):
    """Before any fit of a wave: every batch's file against its hash, id, schema and record; the current protocol
    hash, source manifest, task and split seeds; run ids recomputed from the recipes; and the whole wave against
    a re-planning from the log. Returns the logged batches with their manifests."""
    protocol_seeds(proto, seeds)
    todo = [b for b in log.batches if b["wave"] == wave]
    if not todo:
        raise ValueError(f"wave {wave} is not planned")
    replanned = {m["family"]: full_manifest(m, task, psha, manifest)
                 for m in protocol.plan_wave(proto, log, wave, run_id_of(ctx, seeds, manifest))}
    out = []
    for b in todo:
        text = (ctx.root / b["file"]).read_text()
        m = strict_loads(text)
        bid = m.pop("batch_id")
        problems = [k for k, ok in {
            "file hash": artifacts.sha256_bytes(text.encode()) == b["file_sha256"],
            "batch id": bid == b["batch_id"] == batches.batch_id(batches.validate_batch(m)),
            "task": m["task_id"] == task == proto["task_id"], "protocol": m["protocol_sha256"] == psha,
            "sources": m["source_manifest_sha256"] == manifest["manifest_sha256"],
            "run ids": m["run_ids"] == [run_id_of(ctx, seeds, manifest)(r) for r in m["recipes"]],
            "record": all(artifacts.dumps(b.get(k)) == artifacts.dumps(batch_record(
                m, bid, b["file"], b["file_sha256"], task, psha)[k]) for k in STABLE_BATCH_FIELDS),
            "re-planning": artifacts.dumps(replanned.get(b["family"])) == artifacts.dumps(m)}.items() if not ok]
        if problems:
            raise ValueError(f"batch {b['batch_id']} fails its preflight: {problems}; nothing was fitted")
        out.append((b, m))
    if sorted(replanned) != sorted(b["family"] for b in todo):
        raise ValueError(f"wave {wave} differs from its re-planning in its families; nothing was fitted")
    return out


H_ARRAYS = ("train_row_id", "y", "group_id", "split_seeds", "fold_id", "oof", "test_id", "test_by_fold", "test_mean",
            "fold_auc")


def h_check(ctx, proto, h, task, verify=False):
    """Plan section 15.2: H's arrays must equal the P2.1 anchor bank's exactly. Logged once (h_check record) with
    both bank hashes; verify recomputes and compares with the record."""
    new, nsha, _ = runs.load_logged(ctx, h)
    old, osha, _ = runs.load_logged(ctx, proto["reference_p21_run_id"])
    equal = {a: bool(np.array_equal(getattr(new, a), getattr(old, a))) for a in H_ARRAYS}
    rec = {"record": "h_check", "task_id": task, "h_run_id": h, "h_sha256": nsha,
           "p21_run_id": proto["reference_p21_run_id"], "p21_sha256": osha, "arrays_equal": equal,
           "passed": all(equal.values())}
    if verify:
        logged = [r for r in runs.logged(ctx, "h_check") if r["h_run_id"] == h]
        return len(logged) == 1 and all(artifacts.dumps(logged[0][k]) == artifacts.dumps(rec[k]) for k in rec)
    if not [r for r in runs.logged(ctx, "h_check") if r["h_run_id"] == h]:
        artifacts.append_experiment(ctx.root / runs.LOG_FILE, dict(rec, timestamp_utc=now()))
    print(f"  H check: {h} against {proto['reference_p21_run_id']}: "
          f"{'all ten arrays equal' if rec['passed'] else 'DIFFERS in ' + str([a for a, ok in equal.items() if not ok])}")
    return rec["passed"]


def virtual(ctx, ref_, h):
    """A bank for a bootstrap comparison: a logged run, or a contribution blend rebuilt from its two logged banks."""
    if ref_[0] == "run":
        return runs.load_logged(ctx, ref_[1])[:2]
    c, csha, _ = runs.load_logged(ctx, ref_[1])
    hb, hsha, _ = runs.load_logged(ctx, h)
    b = blend_banks([(hb, hsha), (c, csha)], [1.0 - ref_[2], ref_[2]])
    return b, f"blend:{hsha}:{csha}:{ref_[2]}"


def bootstrap_records(ctx, batch, res, log, task, verify=False):
    """The paired cluster bootstraps of the comparisons that drove a batch's allocation (protocol.witnesses),
    logged with their settings, parent hashes and repeat differences; descriptive, never a gate. Idempotent: a
    comparison already logged is not logged again. With verify, every one must be logged and recompute equal."""
    h = res["resolved_state"]["reference_h"]["run_id"]
    have = {r["comparison_id"]: r for r in runs.logged(ctx, "bootstrap")}
    ok = True
    for route, a, b in protocol.witnesses(batch, res, log):
        (ba, ia), (bb, ib) = virtual(ctx, a, h), virtual(ctx, b, h)
        ident = {"batch_id": batch["batch_id"], "route": route, "a": list(a), "b": list(b), "a_ident": ia, "b_ident": ib,
                 "n_boot": 2000, "seed": BOOTSTRAP_SEED}
        cid = artifacts.config_hash(ident)[:16]
        out = paired_cluster_bootstrap(ba, bb, n_boot=ident["n_boot"], seed=ident["seed"])
        rec = {"record": "bootstrap", "task_id": task, "comparison_id": cid, **ident, **out,
               "repeat_differences": (ba.fold_auc.mean(axis=1) - bb.fold_auc.mean(axis=1)).tolist()}
        if verify:
            old = have.get(cid)
            good = old is not None and all(old[k] == rec[k] for k in ("n_boot", "valid", "invalid", "seed", "groups")) \
                and all(abs(old[k] - rec[k]) <= TOL for k in ("observed_difference", "draws_positive_share")) \
                and np.allclose(old["interval_95"], rec["interval_95"], rtol=0, atol=TOL) \
                and np.allclose(old["repeat_differences"], rec["repeat_differences"], rtol=0, atol=TOL)
            ok &= good
        elif cid not in have:
            artifacts.append_experiment(ctx.root / runs.LOG_FILE, dict(rec, timestamp_utc=now()))
            print(f"  bootstrap {route}: {rec['observed_difference']:+.4f} [{rec['interval_95'][0]:+.4f}, "
                  f"{rec['interval_95'][1]:+.4f}]")
    return ok


def run_wave(ctx, wave, seeds, task, manifest, inv, anchors_only=False):
    """Run the logged batches of a wave in family order (plan section 15.3), after a preflight of the whole wave.
    anchors_only (wave 1 only): the initialization, every family's anchor, then H's reproduction check; no
    allocation. A search fit of wave 1 needs every anchor and a passed H check. Each batch logs a batch_start
    record before its first search fit; after the cutoff, only batches with a start record continue. Each run logs
    its contribution records; a failed fit logs a run_failure record and stops. Each completed batch is judged
    (plan section 5.6), its allocation and the bootstraps of its witnesses logged."""
    proto, psha = load_protocol(ctx, task)
    if anchors_only and wave != 1:
        raise ValueError("--anchors-only is the initialization of wave 1 only")
    validate_prior(ctx, proto)
    log = load_log(ctx, task)
    todo = preflight(ctx, wave, seeds, task, manifest, proto, psha, log)
    h = protocol.h_run_id(proto, log)
    if wave == 1 and not anchors_only:
        missing = [b["batch_id"] for b, m in todo if b["run_ids"][0] not in log.runs]
        if missing or not any(c["h_run_id"] == h and c["passed"] for c in log.h_checks):
            raise ValueError(f"the initialization is incomplete (anchors missing in {missing} or no passed H check); "
                             f"run --run-wave 1 --anchors-only first")
    for b, m in todo:
        if b["batch_id"] in log.allocations:
            bootstrap_records(ctx, b, protocol.judge(b, log), log, task)  # idempotent: finishes any missing one
            print(f"  {b['batch_id']}: already judged")
            continue
        first = 0 if anchors_only or b["kind"] != "initial" else 1
        n = 1 if anchors_only else len(m["recipes"])
        if anchors_only:
            if b["run_ids"][0] not in log.runs and past_cutoff(proto):
                print(f"  {b['batch_id']}: anchor not fitted, the cutoff {proto['cutoff_utc']} has passed")
                return False
        elif b["batch_id"] not in log.starts:
            if past_cutoff(proto):
                print(f"  {b['batch_id']}: not started, the cutoff {proto['cutoff_utc']} has passed")
                return False
            artifacts.append_experiment(ctx.root / runs.LOG_FILE, {
                "record": "batch_start", "task_id": task, "batch_id": b["batch_id"], "timestamp_utc": now()})
        print(f"batch {b['batch_id']} ({b['kind']}, {n - first if not anchors_only else 1} of {len(m['recipes'])} "
              f"configurations to fit)")
        for i, (recipe, rid) in enumerate(zip(m["recipes"][:n], m["run_ids"][:n])):
            try:
                rec = run_one(ctx, recipe, seeds, task, manifest, inv, search_id=b["batch_id"])
            except Exception as e:
                artifacts.append_experiment(ctx.root / runs.LOG_FILE, {
                    "record": "run_failure", "task_id": task, "batch_id": b["batch_id"], "planned_run_id": rid,
                    "recipe_sha256": searches.recipe_hash(recipe), "error": f"{type(e).__name__}: {e}",
                    "timestamp_utc": now()})
                raise
            if rec["run_id"] != rid:  # unreachable after the preflight; kept as a guard
                raise AssertionError(f"run id {rec['run_id']} differs from the planned {rid}")
            if rid == h and anchors_only and not h_check(ctx, proto, h, task):
                raise AssertionError(f"H differs from the P2.1 anchor bank; {task} stops (plan sections 15.2, 17)")
            contributions(ctx, rid, h, task)
        if anchors_only:
            continue
        log = load_log(ctx, task)
        res = protocol.judge(b, log)
        bootstrap_records(ctx, b, res, log, task)  # persisted before the allocation record closes the batch
        artifacts.append_experiment(ctx.root / runs.LOG_FILE, {
            "record": "allocation", "task_id": task, "batch_id": b["batch_id"], "family": b["family"], "wave": wave,
            "kind": b["kind"], **res, "timestamp_utc": now()})
        met = sum(1 for v in res["routes"].values() if v)
        print(f"  allocation: {met} configurations meet a route; progress {res['progress']}; "
              f"allowance {b['state']['allowance_before']} -> {res['allowance_after']}")
    return True


ALLOCATION_FIELDS = ("routes", "progress", "allowance_after", "resolved_state")


def audit_task(ctx, seeds, task, manifest, allow_incomplete=False):
    """Without refitting (plan section 15.5): every manifest against its file, record and a re-planning of its
    wave; every task run reconciled with the manifests; every bank against its hash and record; every
    contribution, allocation, bootstrap and the H check recomputed. Completion (the default) also requires every
    planned batch to be fully run and judged, or unstarted at the cutoff with every earlier batch closed;
    allow_incomplete audits the integrity of an unfinished task. Run at the sources the runs were made with."""
    proto, psha = load_protocol(ctx, task)
    try:
        validate_prior(ctx, proto)
    except ValueError as e:
        print(f"  {e}")
        return False
    log = load_log(ctx, task)
    ok, planned_ids = True, []
    if not log.batches:
        print("  no batch is logged")
        return False
    for wave in sorted({b["wave"] for b in log.batches}):
        try:
            preflight(ctx, wave, seeds, task, manifest, proto, psha, log)
            print(f"  wave {wave}: every manifest equals its file, record and re-planning")
        except (ValueError, KeyError) as e:
            print(f"  wave {wave}: DIFFERS: {type(e).__name__}: {e}")
            ok = False
        for b in log.batches:
            if b["wave"] == wave:
                planned_ids += [r for r in b["run_ids"] if r not in planned_ids]
    task_runs = [r["run_id"] for r in runs.logged(ctx) if r["task_id"] == task]
    stray = sorted(set(task_runs) - set(planned_ids))
    if stray:
        print(f"  runs of {task} outside every manifest: {stray}")
        ok = False
    done = [rid for rid in planned_ids if rid in log.runs]
    bad = {rid: v for rid in done if (v := runs.audit_run(ctx, rid))}
    print(f"  runs: {len(done)} audited against their banks and records, {len(bad)} with problems"
          + (f": {bad}" if bad else ""))
    ok &= not bad
    h = protocol.h_run_id(proto, log)
    if h in log.runs:
        good = h_check(ctx, proto, h, task, verify=True)
        print(f"  H check: {'recomputed equal' if good else 'DIFFERS or missing'}")
        ok &= good
    contrib_ok = all(contributions(ctx, rid, h, task, verify=True) for rid in done)
    print(f"  contributions of {len(done)} runs: {'recomputed equal' if contrib_ok else 'DIFFER'}")
    ok &= contrib_ok
    for b in log.batches:
        if b["batch_id"] in log.allocations:
            missing = [r for r in b["run_ids"] if r not in log.runs]
            try:
                res, rec = protocol.judge(b, log), log.allocations[b["batch_id"]]
                good = all(artifacts.dumps(res[k]) == artifacts.dumps(rec[k]) for k in ALLOCATION_FIELDS) \
                    and bootstrap_records(ctx, b, res, log, task, verify=True)
            except (ValueError, KeyError) as e:
                good = False
                missing.append(f"{type(e).__name__}: {e}")
            print(f"  allocation {b['batch_id']}: "
                  f"{'re-judged equal, bootstraps recomputed' if good else f'DIFFERS {missing}'}")
            ok &= good
    if not allow_incomplete:
        problems = completion_problems(ctx, proto, log, seeds, manifest)
        for msg in problems:
            print(f"  incomplete: {msg}")
        ok &= not problems
    return ok


def completion_problems(ctx, proto, log, seeds, manifest):
    """Why the task is not complete (plan section 15.5), or nothing: every planned batch must be closed (all its
    runs, contributions, bootstraps and allocation logged), except batches of the last wave left unstarted because
    the cutoff passed; before the cutoff, no further wave may be due."""
    out = []
    if not log.batches:
        return ["no batch is logged"]
    last = max(b["wave"] for b in log.batches)
    cutoff = past_cutoff(proto)
    for b in log.batches:
        if b["batch_id"] in log.allocations:
            continue
        unstarted = b["batch_id"] not in log.starts and not any(
            r in log.runs for r in b["run_ids"][1 if b["kind"] == "initial" else 0:])
        if not unstarted:
            out.append(f"batch {b['batch_id']} is partial")
        elif b["wave"] != last:
            out.append(f"batch {b['batch_id']} is unstarted before the last wave")
        elif not cutoff:
            out.append(f"batch {b['batch_id']} is unstarted and the cutoff {proto['cutoff_utc']} has not passed")
    if not out and not cutoff and all(b["batch_id"] in log.allocations for b in log.batches if b["wave"] == last):
        due = protocol.plan_wave(proto, log, last + 1, run_id_of(ctx, seeds, manifest))
        if due:
            out.append(f"wave {last + 1} is still due ({len(due)} batches) before the cutoff")
    return out


def verify_task(ctx, task, manifest, inv, seeds=SPLIT_SEEDS, allow_incomplete=False):
    """Refit every distinct run of the task's batches once and compare it with its bank and record. An empty set
    fails, and so does an incomplete task unless allow_incomplete."""
    proto, _ = load_protocol(ctx, task)
    validate_prior(ctx, proto)
    log = load_log(ctx, task)
    problems = [] if allow_incomplete else completion_problems(ctx, proto, log, list(seeds), manifest)
    if problems:
        print(f"  {task} is not complete: {problems}")
        return False
    ids = []
    for b in log.batches:
        ids += [r for r in b["run_ids"] if r in log.runs and r not in ids]
    if not ids:
        print(f"  no run of {task} to verify")
        return False
    return verify(ctx, ids, manifest, inv)


# ---- the P2.4 feature stage (plan section 18) -----------------------------------------------------------------

FEATURE_PROTOCOLS = {"P2.4": Path("configs/phase2/p2.4/protocol.json")}
FEATURE_RECORDS = ("feature_batch", "feature_allocation", "feature_stop", "batch_start", "run", "contribution",
                   "comparison", "bootstrap", "h_check", "run_failure", "timing")
_ENTRIES = {}  # validated entries of this invocation, by (run id, bank hash, H bank hash)


def load_feature_protocol(ctx, task):
    if task not in FEATURE_PROTOCOLS:
        raise ValueError(f"task {task} has no feature protocol (known: {sorted(FEATURE_PROTOCOLS)})")
    raw = (ctx.root / FEATURE_PROTOCOLS[task]).read_bytes()
    proto = feature_stage.load_protocol(strict_loads(raw.decode()))
    if proto["task_id"] != task:
        raise ValueError(f"the feature protocol of {task} belongs to task {proto['task_id']}")
    return proto, artifacts.sha256_bytes(raw)


def pool_entry(ctx, run_id, h_bank, h_sha, expect_sha=None):
    """A run as the feature stage sees it, read from its validated bank (plan section 18.3): the bank's bytes against
    its logged hash (and the expected hash, if given), its record against the bank (runs.audit_run); recipe,
    features, task and scores taken from the bank; its two contribution blends with H recomputed."""
    bank, sha, rec = runs.load_logged(ctx, run_id)
    if expect_sha is not None and sha != expect_sha:
        raise ValueError(f"run {run_id}: bank hash {sha} differs from the expected {expect_sha}")
    key = (run_id, sha, h_sha)
    if key in _ENTRIES:
        return _ENTRIES[key]
    problems = runs.audit_run(ctx, run_id)
    if problems:
        raise ValueError(f"run {run_id} fails its audit: {problems}")
    md = bank.metadata
    if not (md["selection_eligible"] and [int(s) for s in bank.split_seeds] == list(SPLIT_SEEDS)
            and len(bank.y) == feature_stage.N_ROWS and md["prediction_kind"] == "fitted"):
        raise ValueError(f"run {run_id} is not a complete fitted run on the eligible rows and official seeds")
    repeats = allocation.repeat_means(bank).tolist()
    blends = {}
    for w in allocation.CONTRIBUTION_WEIGHTS:
        if sha == h_sha:  # H blended with itself is H (an identity, as in contributions())
            blends[w] = {"mean": float(bank.fold_auc.mean()), "repeats": repeats}
            continue
        c = allocation.contribution(h_bank, h_sha, bank, sha, w)
        blends[w] = {"mean": c["mean_fold_auc"], "repeats": c["repeat_mean_auc"]}
    e = {"run_id": run_id, "task_id": md["task_id"], "family": md["recipe"]["family"], "recipe": md["recipe"],
         "recipe_id": md["recipe_id"], "recipe_hash": md["recipe_sha256"], "bank_sha256": sha,
         "features": list(md["features"]), "mean": float(bank.fold_auc.mean()), "repeats": repeats, "blends": blends}
    _ENTRIES[key] = e
    return e


class FeatureLog:
    """The P2.4 records of the phase 2 log, checked: one feature_batch record per (stage, index) and per batch id,
    at most one allocation, start and stop record per batch, no start or allocation without its batch. extra are
    records not yet appended, checked as if they were (a command validates its record before writing it)."""

    def __init__(self, ctx, task, extra=()):
        recs = artifacts.read_experiments(ctx.root / runs.LOG_FILE) + list(extra)
        mine = [r for r in recs if r.get("task_id") == task]
        self._pos = {id(r): i for i, r in enumerate(mine)}
        self.batches = sorted((r for r in mine if r.get("record") == "feature_batch"),
                              key=lambda r: (r["stage"] != "features", r["index"]))
        keys = [(b["stage"], b["index"]) for b in self.batches]
        ids = [b["batch_id"] for b in self.batches]
        if len(set(keys)) != len(keys) or len(set(ids)) != len(ids):
            raise ValueError("duplicate feature batch records")

        def one(kind):
            out = {}
            for r in mine:
                if r.get("record") == kind:
                    if r["batch_id"] in out:
                        raise ValueError(f"duplicate {kind} records for {r['batch_id']}")
                    if r["batch_id"] not in ids:
                        raise ValueError(f"a {kind} record names an unknown batch {r['batch_id']}")
                    out[r["batch_id"]] = r
            return out
        self.allocations, self.starts_rec = one("feature_allocation"), one("batch_start")
        self.starts = set(self.starts_rec)
        stops = [r for r in mine if r.get("record") == "feature_stop"]
        if len(stops) > 1:
            raise ValueError("more than one feature_stop record")
        self.stop = stops[0] if stops else None
        if self.stop is not None:
            feats = [b for b in self.batches if b["stage"] == "features"]
            st = self.stop
            if not isinstance(st.get("reason"), str) or not st["reason"].strip() or not feats \
                    or st.get("batch_id") != feats[-1]["batch_id"] or st["batch_id"] not in self.allocations \
                    or mine.index(st) < mine.index(self.allocations[st["batch_id"]]) \
                    or any(b["stage"] == "retune" and mine.index(b) < mine.index(st) for b in self.batches):
                raise ValueError("the feature_stop record must give a reason, follow the last feature batch's "
                                 "allocation, name that batch, and precede every re-tuning round")
        self.manifests = {b["batch_id"]: read_manifest(ctx, b["file"]) for b in self.batches}
        self.records = mine

    def of(self, stage):
        return [b for b in self.batches if b["stage"] == stage]

    def judged(self, stage):
        return [self.allocations[b["batch_id"]]["judgment"] for b in self.of(stage) if b["batch_id"] in self.allocations]

    def position(self, r):
        """A record's position among the task's records, in log order."""
        return self._pos[id(r)]

    def upto(self, pos):
        """The log as it stood before the task's record at position pos: only the records logged before it."""
        view = copy.copy(self)
        kept = lambda r: self._pos[id(r)] < pos  # noqa: E731
        view.records = self.records[:pos]
        view.batches = [b for b in self.batches if kept(b)]
        view.allocations = {k: v for k, v in self.allocations.items() if kept(v)}
        view.starts_rec = {k: v for k, v in self.starts_rec.items() if kept(v)}
        view.starts = set(view.starts_rec)
        view.stop = self.stop if self.stop is not None and kept(self.stop) else None
        return view

    def before(self, stage, index):
        """The log as it stood when (stage, index) was planned: the records logged before its feature_batch record
        (all of them when it is not logged), and of their batches only those before it (feature batches come before
        every re-tuning round), with their allocations and starts. Re-planning from this view is how a batch record
        logged before an allocation its manifest uses fails its preflight."""
        rec = next((b for b in self.batches if (b["stage"], b["index"]) == (stage, index)), None)
        view = self.upto(self.position(rec) if rec is not None else len(self.records))
        key = (stage != "features", index)
        view.batches = [b for b in view.batches if (b["stage"] != "features", b["index"]) < key]
        ids = {b["batch_id"] for b in view.batches}
        view.allocations = {k: v for k, v in view.allocations.items() if k in ids}
        view.starts_rec = {k: v for k, v in view.starts_rec.items() if k in ids}
        view.starts = set(view.starts_rec)
        return view


def reference_bank(ctx, proto):
    bank, sha, _ = runs.load_logged(ctx, proto["reference_p21_run_id"])
    return bank, sha


def feature_pool(ctx, proto, flog):
    """The reference pool (plan section 18.3): the protocol's frozen reference inventory, each run validated against
    its bank and expected hash, and the runs of the log's judged P2.4 batches; blends with H from the P2.1 anchor
    bank, whose ten arrays every H check requires H to equal. Later runs of the reference tasks never enter."""
    hb, hs = reference_bank(ctx, proto)
    out = [pool_entry(ctx, e["run_id"], hb, hs, e["bank_sha256"]) for e in proto["reference_inventory"]]
    have = {e["run_id"] for e in out}
    for b in flog.batches:
        if b["batch_id"] in flog.allocations:
            for r in b["run_ids"]:
                if r not in have:
                    have.add(r)
                    out.append(pool_entry(ctx, r, hb, hs))
    return out


def validate_imports(ctx, proto):
    """Plan section 18.2: P2.8's 14 comparisons enter with their original definitions and decisions. The receipt's
    files must have their hashes; each entry must be the hashed P2.8 manifest's comparison of that id (family,
    challenger and comparator recipe names and recipe hashes), its runs P2.8 runs of exactly those recipes with
    banks matching their hashes and records; its H the manifest's named H, audited; and every route flag must
    recompute from P2.8's own frozen allocation state with that H. Returns the receipt's comparisons."""
    imp = proto["imports"]
    for k in ("manifest", "judge", "output"):
        if artifacts.sha256_bytes((ctx.root / imp[k]).read_bytes()) != imp[f"{k}_sha256"]:
            raise ValueError(f"the import receipt's {k} {imp[k]} differs from its hash")
    pb = read_manifest(ctx, imp["manifest"])["part_b"]
    defs = {c["id"]: c for c in pb["comparisons"]}
    rhash = {r["name"]: r["recipe_sha256"] for r in pb["recipes"]}
    st = pb["allocation_state"]
    hb, hs, hrec = runs.load_logged(ctx, imp["h_run_id"])
    if hb.metadata["recipe"]["name"] != st["H"] or hb.metadata["recipe_sha256"] != rhash[st["H"]] \
            or hrec["task_id"] != imp["task"] or runs.audit_run(ctx, imp["h_run_id"]):
        raise ValueError("the import receipt's H is not P2.8's named H, or fails its audit")
    if sorted(defs) != sorted(c["id"] for c in imp["comparisons"]):
        raise ValueError("the receipt's comparisons are not exactly the P2.8 manifest's")
    for c in imp["comparisons"]:
        d = defs[c["id"]]
        if c["family"] != d["family"]:
            raise ValueError(f"imported comparison {c['id']}: family")
        for side in ("comparator", "challenger"):
            bank, sha, rec = runs.load_logged(ctx, c[f"{side}_run_id"])
            md = bank.metadata
            if c[f"{side}_recipe"]["name"] != d[side] or md["recipe"]["name"] != d[side] \
                    or md["recipe_sha256"] != rhash[d[side]] \
                    or searches.recipe_hash(c[f"{side}_recipe"]) != rhash[d[side]] \
                    or sha != c[f"{side}_bank_sha256"] or rec["task_id"] != imp["task"] \
                    or runs.audit_run(ctx, c[f"{side}_run_id"]):
                raise ValueError(f"imported comparison {c['id']}: the {side} is not the P2.8 manifest's, or fails "
                                 f"its bank or audit")
        if c["challenger_recipe_sha256"] != rhash[d["challenger"]]:
            raise ValueError(f"imported comparison {c['id']}: challenger hash")
        e = pool_entry(ctx, c["challenger_run_id"], hb, hs)
        f = c["family"]
        rts = allocation.routes({"standalone": e["repeats"], "blends": [e["blends"][w]["repeats"]
                                                                      for w in allocation.CONTRIBUTION_WEIGHTS]},
                                st["family_incumbents"][f]["repeat_mean_auc_unrounded"],
                                st["strongest_standalone"]["repeat_mean_auc_unrounded"],
                                hrec["repeat_mean_auc_unrounded"],
                                st["previous_best_blends"][f]["repeat_mean_auc_unrounded"])
        if rts != c["routes"]:
            raise ValueError(f"imported comparison {c['id']}: routes {rts} recompute differently from {c['routes']}")
    return imp["comparisons"]


def validate_handoff(ctx, proto):
    """Plan section 18.4: the CatBoost and EBM aliases come from P2.3's converged handoff. The handoff file must have
    its hash; an available family's alias must be its best standalone run; an unavailable family's alias must be
    the declared fallback (C) or a declared omission (M). A protocol without a handoff path (a test) passes."""
    h = proto["p23_handoff"]
    if h["path"] is None:
        return
    text = (ctx.root / h["path"]).read_text()
    if artifacts.sha256_bytes(text.encode()) != h["sha256"]:
        raise ValueError(f"P2.3's handoff {h['path']} differs from its hash")
    data = strict_loads(text[text.index("Handoff to P2.4"):].split("\n", 1)[1])
    for alias, fam in (("C", "catboost"), ("M", "ebm")):
        v, a = data["families"][fam], proto["aliases"][alias]
        ok = (a["run_id"] == v["best_standalone"]["run_id"]) if v["available"] else \
            (("used_fallback" in a["rule"]) if alias == "C" else a["run_id"] is None)
        if not ok:
            raise ValueError(f"alias {alias} does not follow P2.3's handoff for {fam}")


def used_comparisons(proto, flog):
    return len(proto["imports"]["comparisons"]) + sum(
        len(flog.manifests[b["batch_id"]]["comparisons"]) + len(flog.manifests[b["batch_id"]]["omitted"])
        for b in flog.of("features"))


def feature_closed(proto, flog):
    """Why the feature stage takes no further expansion, or None: a recorded stop, the counter at zero, or the
    comparison ceiling (an exhausted queue shows as an expansion that plans nothing)."""
    judged = flog.judged("features")
    if not judged or len(judged) != len(flog.of("features")):
        return None
    if flog.stop is not None:
        return "stopped"
    if judged[-1]["allowance_after"] <= 0:
        return "allowance"
    return "ceiling" if used_comparisons(proto, flog) >= proto["expansion"]["max_comparisons"] else None


def plan_feature_batch(ctx, proto, flog, stage, index, run_id_of_, planned_at):
    """The manifest of a feature batch (stage "features") or a re-tuning round (stage "retune"), a pure function of
    the protocol, the log before it, the validated banks and planned_at (the planning time, which only re-tuning's
    admission uses); None when nothing is due."""
    imports = proto["imports"]["comparisons"]
    flog = flog.before(stage, index)
    unjudged = [b["batch_id"] for b in flog.batches if b["batch_id"] not in flog.allocations]
    if unjudged:
        raise ValueError(f"{stage} {index} needs every earlier batch judged first: {unjudged}")
    pool = feature_pool(ctx, proto, flog)
    hb, hs = reference_bank(ctx, proto)
    h_entry = pool_entry(ctx, proto["reference_p21_run_id"], hb, hs)
    h = {"recipe": proto["h_recipe"], "run_id": run_id_of_(proto["h_recipe"])}
    aliases = feature_stage.resolve_aliases(proto, pool)
    prior = [flog.manifests[b["batch_id"]] for b in flog.of("features")]
    out = {"schema": feature_stage.BATCH_SCHEMA, "task_id": proto["task_id"], "stage": stage, "index": index, "h": h,
           "aliases": {k: (v["run_id"] if v else None) for k, v in aliases.items()},
           "prior_batches": [{"batch_id": b["batch_id"], "file_sha256": b["file_sha256"],
                              "allocation_sha256": artifacts.sha256_bytes(artifacts.dumps(
                                  flog.allocations[b["batch_id"]]["judgment"]).encode())} for b in flog.batches]}
    if stage == "features":
        if index == 1:
            comps, omitted = feature_stage.initial_comparisons(proto, aliases)
            kind, before = "initial", None
        else:
            judged = flog.judged("features")
            if len(judged) != index - 1 or len(prior) != index - 1:
                raise ValueError(f"feature batch {index} needs batches 1 to {index - 1} judged")
            if feature_closed(proto, flog):
                return None
            comps = feature_stage.plan_expansion(proto, pool, aliases, prior, judged, imports, index)
            omitted, kind, before = [], "expansion", judged[-1]["allowance_after"]
            if not comps:
                return None
        for c in comps:
            c["comparator_run_id"], c["challenger_run_id"] = run_id_of_(c["comparator"]), run_id_of_(c["challenger"])
        out.update({"kind": kind, "comparisons": comps, "omitted": omitted,
                    "state": feature_stage.state(pool, h_entry, {c["family"] for c in comps}, before)})
        return out
    # Re-tuning round index (1 or 2): round 1 chooses the units once the feature stage is closed; round 2 gives each
    # round 1 unit its second batch (plan section 18.6). Units are admitted in order while their projected fitting
    # and replay, from the parents' measured times, end before the protocol's re-tuning budget end.
    rp = proto["retune"]
    if index > rp["max_batches"] or len(flog.judged("features")) != len(flog.of("features")):
        return None
    if index == 1:
        recipes = {i["id"]: i["challenger_recipe"] for i in imports}
        for m in prior:
            recipes.update({c["id"]: c["challenger"] for c in m["comparisons"]})
        routes = feature_stage.met_routes(flog.judged("features"), imports)
        units = [v["unit"] for v in feature_stage.choose_units(proto, pool, routes, recipes)]
    else:
        rounds = flog.of("retune")
        if len(rounds) != index - 1 or rounds[-1]["batch_id"] not in flog.allocations:
            return None
        last = flog.manifests[rounds[-1]["batch_id"]]
        units = [u["unit"] for u in last["units"]]
    budget = (datetime.fromisoformat(proto["retune_budget_end_utc"]) - datetime.fromisoformat(planned_at)).total_seconds()
    planned, omitted, spent = [], [], 0.0
    for k, u in enumerate(units):
        parents, exclude = feature_stage.retune_parents(proto, u, pool)
        recs_by_id = {r["run_id"]: r for r in runs_by_id(ctx, [p["run_id"] for p in parents])}
        cost = 2 * rp["batch"] * max(recs_by_id[p["run_id"]]["elapsed_seconds"] for p in parents)
        if spent + cost > budget:
            omitted.append({"unit_id": u["unit_id"], "reason": f"projected {cost:.0f} s of fitting and replay would "
                                                              f"end after {proto['retune_budget_end_utc']}"})
            continue
        spent += cost
        seed = rp["seed_base"] + 100 * k + index
        recs, info = feature_stage.plan_retune(proto, u, [p["recipe"] for p in parents], exclude, index, seed)
        planned.append({"unit": u, "parents": [p["run_id"] for p in parents], "recipes": recs,
                        "run_ids": [run_id_of_(r) for r in recs], "rejected": info, "sampler_seed": seed,
                        "projected_seconds": cost})
    if not planned and not omitted:
        return None
    out.update({"kind": "retune", "units": planned, "omitted_units": omitted, "planned_at_utc": planned_at,
                "state": feature_stage.state(pool, h_entry, {u["unit"]["family"] for u in planned} or {"lightgbm"},
                                             None)})
    return out


def runs_by_id(ctx, ids):
    recs = {r["run_id"]: r for r in runs.logged(ctx)}
    return [recs[i] for i in ids]


def feature_run_ids(m):
    ids = [m["h"]["run_id"]]
    for c in m.get("comparisons", []):
        ids += [c["comparator_run_id"], c["challenger_run_id"]]
    for u in m.get("units", []):
        ids += u["run_ids"]
    return list(dict.fromkeys(ids))


def feature_text(m, psha, manifest):
    return artifacts.dumps(dict(m, protocol_sha256=psha, source_manifest_sha256=manifest["manifest_sha256"]))


def feature_batch_id(m, text):
    return f"p24_{m['stage']}_{m['index']}_{artifacts.sha256_bytes(text.encode())[:12]}"


def feature_record(m, text, task, psha, planned_at):
    bid = feature_batch_id(m, text)
    return {"record": "feature_batch", "task_id": task, "batch_id": bid, "stage": m["stage"], "index": m["index"],
            "kind": m["kind"], "file": str(BATCH_DIR / f"{bid}.json"),
            "file_sha256": artifacts.sha256_bytes(text.encode()), "run_ids": feature_run_ids(m),
            "protocol_sha256": psha, "planned_at_utc": planned_at}


def write_once(ctx, rel, text):
    target = ctx.root / rel
    if target.exists():
        if target.read_text() != text:
            raise ValueError(f"{rel} exists with other content")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x") as f:
        f.write(text)


def plan_features(ctx, stage, index, seeds, task, manifest):
    """Save and log a feature batch or re-tuning round before any of its fits (plan section 18). Resumable: an
    existing record is re-planned with its own planning time and must equal the re-planning before anything is
    written."""
    if index < 1:
        raise ValueError("batch indices start at 1")
    proto, psha = load_feature_protocol(ctx, task)
    protocol_seeds(proto, seeds)
    validate_imports(ctx, proto)
    validate_handoff(ctx, proto)
    flog = FeatureLog(ctx, task)
    logged = [b for b in flog.batches if b["stage"] == stage and b["index"] == index]
    if not logged:
        if past_cutoff(proto):
            raise ValueError(f"the {task} cutoff {proto['cutoff_utc']} has passed: no new batch")
        prev = flog.of(stage)
        if len(prev) != index - 1 or (prev and prev[-1]["batch_id"] not in flog.allocations):
            raise ValueError(f"{stage} {index} needs {stage} 1 to {index - 1} planned and judged")
    if stage == "features" and flog.stop is not None and not logged:
        raise ValueError("the feature stage is stopped (feature_stop record)")
    if stage == "retune" and not logged:
        if not flog.of("features") or flog.of("features")[-1]["batch_id"] not in flog.allocations:
            raise ValueError("re-tuning starts after the last feature batch is judged")
        nxt = len(flog.of("features")) + 1
        if not feature_closed(proto, flog) and plan_feature_batch(
                ctx, proto, flog, "features", nxt, run_id_of(ctx, seeds, manifest), now()) is not None:
            raise ValueError("the feature stage still has an expansion due; re-tuning waits (or log a feature_stop)")
    planned_at = logged[0]["planned_at_utc"] if logged else now()
    m = plan_feature_batch(ctx, proto, flog, stage, index, run_id_of(ctx, seeds, manifest), planned_at)
    if m is None:
        if logged:
            raise ValueError(f"the logged {stage} {index} no longer re-plans")
        print(f"  {stage} {index}: nothing is due ({feature_closed(proto, flog) or 'no candidates or units'})")
        return None
    text = feature_text(m, psha, manifest)
    rec = feature_record(m, text, task, psha, planned_at)
    if logged:
        if any(artifacts.dumps(logged[0].get(k)) != artifacts.dumps(v) for k, v in rec.items()):
            raise ValueError(f"the logged {stage} {index} differs from its re-planning; nothing was written")
        write_once(ctx, rec["file"], text)  # restores a lost manifest file; an existing one must be identical
        print(f"  {rec['batch_id']}: already planned")
        return m
    write_once(ctx, rec["file"], text)
    artifacts.append_experiment(ctx.root / runs.LOG_FILE, dict(rec, timestamp_utc=now()))
    n = len(m.get("comparisons", [])) or sum(len(u["recipes"]) for u in m["units"])
    print(f"  {rec['batch_id']}: {m['kind']}, {n} {'comparisons' if stage == 'features' else 'configurations'}"
          f"{', omitted ' + str(m.get('omitted') or m.get('omitted_units')) if m.get('omitted') or m.get('omitted_units') else ''}"
          f"; saved in {rec['file']}")
    return m


def stop_features(ctx, task, reason, seeds, manifest):
    """Plan section 18.9: a recorded stop of the feature expansions (the cut order's first items), after the last
    feature batch is judged: the reason, the last batch, and the expansion it omits (re-planned, not run). The record
    is checked as part of the log before it is appended, so a refused stop writes nothing."""
    reason = reason.strip() if isinstance(reason, str) else ""
    if not reason:
        raise ValueError("a feature stop needs a reason")
    proto, _ = load_feature_protocol(ctx, task)
    flog = FeatureLog(ctx, task)
    if flog.stop is not None:
        raise ValueError("the feature stage is already stopped")
    if not flog.of("features") or flog.of("features")[-1]["batch_id"] not in flog.allocations:
        raise ValueError("a stop follows a judged feature batch")
    nxt = len(flog.of("features")) + 1
    due = plan_feature_batch(ctx, proto, flog, "features", nxt, run_id_of(ctx, seeds, manifest), now())
    rec = {"record": "feature_stop", "task_id": task, "batch_id": flog.of("features")[-1]["batch_id"],
           "reason": reason, "omitted_expansion": [c["id"] for c in due["comparisons"]] if due else [],
           "timestamp_utc": now()}
    FeatureLog(ctx, task, extra=[rec])  # raises, before anything is written, on a record the log would refuse
    artifacts.append_experiment(ctx.root / runs.LOG_FILE, rec)
    print(f"  feature stage stopped after {rec['batch_id']}: {reason}; omitted {len(rec['omitted_expansion'])}")


def feature_preflight(ctx, proto, psha, flog, b, seeds, manifest):
    text = (ctx.root / b["file"]).read_text()
    m = strict_loads(text)
    replanned = plan_feature_batch(ctx, proto, flog, b["stage"], b["index"], run_id_of(ctx, seeds, manifest),
                                   b["planned_at_utc"])
    rtext = feature_text(replanned, psha, manifest) if replanned is not None else None
    problems = [k for k, ok in {
        "file hash": artifacts.sha256_bytes(text.encode()) == b["file_sha256"],
        "batch id": rtext is not None and feature_batch_id(m, text) == b["batch_id"],
        "stage and index": (m["stage"], m["index"], m["kind"]) == (b["stage"], b["index"], b["kind"]),
        "protocol": m["protocol_sha256"] == psha == b["protocol_sha256"],
        "sources": m["source_manifest_sha256"] == manifest["manifest_sha256"],
        "run ids": feature_run_ids(m) == b["run_ids"],
        "re-planning": rtext == text}.items() if not ok]
    if problems:
        raise ValueError(f"{b['batch_id']} fails its preflight: {problems}; nothing was fitted")
    return m


def feature_entries(ctx, m):
    hb, hs, _ = runs.load_logged(ctx, m["h"]["run_id"])
    return {r: pool_entry(ctx, r, hb, hs) for r in feature_run_ids(m)}


def feature_bootstraps(ctx, b, m, judgment, entries, task, verify=False):
    """The paired cluster bootstraps of the comparisons behind a batch's allocation (feature_stage.witnesses), 2,000
    draws, logged before the allocation; descriptive, never a gate. With verify, every one must be logged and
    recompute equal."""
    h = m["h"]["run_id"]
    have = {r["comparison_id"]: r for r in runs.logged(ctx, "bootstrap")}
    ok = True
    for route, family, a, c in feature_stage.witnesses(m, judgment, entries):
        (ba, ia), (bb, ib) = virtual(ctx, a, h), virtual(ctx, c, h)
        ident = {"batch_id": b["batch_id"], "route": route, "family": family, "a": list(a), "b": list(c),
                 "a_ident": ia, "b_ident": ib, "n_boot": 2000, "seed": BOOTSTRAP_SEED}
        cid = artifacts.config_hash(ident)[:16]
        out = paired_cluster_bootstrap(ba, bb, n_boot=2000, seed=BOOTSTRAP_SEED)
        rec = {"record": "bootstrap", "task_id": task, "comparison_id": cid, **ident, **out,
               "repeat_differences": (ba.fold_auc.mean(axis=1) - bb.fold_auc.mean(axis=1)).tolist()}
        if verify:
            ok &= cid in have and same_record(have[cid], rec)
        elif cid not in have:
            artifacts.append_experiment(ctx.root / runs.LOG_FILE, dict(rec, timestamp_utc=now()))
    return ok


def execution_evidence(records, b, m, first):
    """Positions (in the task's records) of everything a batch's execution logs: the runs it declares first, H's
    check when H is among them, the contributions and matched comparisons of its challengers and configurations,
    its witness bootstraps and run failures."""
    bid = b["batch_id"]
    mine = set(first)
    need = {c["challenger_run_id"] for c in m.get("comparisons", [])} | {r for u in m.get("units", [])
                                                                         for r in u["run_ids"]}
    pairs = {(c["challenger_run_id"], c["comparator_run_id"]) for c in m.get("comparisons", [])}
    out = []
    for i, r in enumerate(records):
        k = r.get("record")
        if (k == "run" and r["run_id"] in mine) or (k == "h_check" and r["h_run_id"] in mine) \
                or (k == "contribution" and r["run_id"] in need and r["h_run_id"] == m["h"]["run_id"]) \
                or (k == "comparison" and (r["a"], r["b"]) in pairs) \
                or (k in ("bootstrap", "run_failure") and r.get("batch_id") == bid):
            out.append((i, k))
    return out


def evidence_order(flog, b, evidence):
    """A batch with any execution evidence needs a start; the batch record precedes the start, the start precedes
    every piece of evidence, and an allocation follows all of it (plan section 18.10)."""
    bid, recs = b["batch_id"], flog.records
    where = lambda kind: next((i for i, r in enumerate(recs) if r.get("record") == kind  # noqa: E731
                               and r.get("batch_id") == bid), None)
    bpos, spos, apos = where("feature_batch"), where("batch_start"), where("feature_allocation")
    out = []
    if (evidence or apos is not None) and spos is None:
        return [f"{bid}: execution evidence or an allocation without a batch_start record"]
    if spos is not None and not bpos < spos:
        out.append(f"{bid}: its start precedes its batch record")
    late = [k for i, k in evidence if spos is not None and i < spos]
    if late:
        out.append(f"{bid}: {sorted(set(late))} logged before its start")
    if apos is not None:
        after = [k for i, k in evidence if i > apos]
        if after or not spos < apos:
            out.append(f"{bid}: {sorted(set(after)) or 'its start'} logged after its allocation")
    return out


def batch_order(flog):
    """Every batch record follows the batch record and the allocation of every batch before it, the allocations its
    manifest's prior_batches name: a batch is planned only from allocations already logged (plan section 18.10)."""
    out = []
    for k, b in enumerate(flog.batches):
        p = flog.position(b)
        for e in flog.batches[:k]:
            a = flog.allocations.get(e["batch_id"])
            if flog.position(e) > p:
                out.append(f"{b['batch_id']}: logged before the earlier batch {e['batch_id']}")
            elif a is None or flog.position(a) > p:
                out.append(f"{b['batch_id']}: logged before {e['batch_id']} was judged")
    return out


def h_order(records, m, first, evidence):
    """Plan section 18.3: in the batch that first declares H, H's run (when this task logged it, not reused it)
    precedes its check, and the check precedes every other fit and record of the batch's execution (H's own failures
    aside)."""
    h = m["h"]["run_id"]
    if h not in first:
        return []
    own = lambda r: (r["record"] == "run" and r["run_id"] == h) or r["record"] == "h_check" \
        or (r["record"] == "run_failure" and r["planned_run_id"] == h)  # noqa: E731
    run_pos = next((i for i, k in evidence if k == "run" and records[i]["run_id"] == h), None)
    check_pos = next((i for i, k in evidence if k == "h_check"), None)
    others = [i for i, k in evidence if not own(records[i])]
    out = []
    if check_pos is not None and run_pos is not None and run_pos > check_pos:
        out.append(f"{m['stage']} {m['index']}: H's check is not preceded by H's run")
    if others and (check_pos is None or check_pos > min(others)):
        out.append(f"{m['stage']} {m['index']}: fits or records logged before H's check")
    return out


def derived_order(records, m, evidence):
    """Each contribution and matched comparison follows the task's run records of the runs it is computed from, and
    the batch closes with its witnesses' bootstraps: they follow every other record of its execution (plan section
    18.10)."""
    run_at = {r["run_id"]: i for i, r in enumerate(records) if r["record"] == "run"}
    boots = [i for i, k in evidence if k == "bootstrap"]
    out = set()
    for i, k in evidence:
        r = records[i]
        uses = [r["run_id"], r["h_run_id"]] if k == "contribution" else [r["a"], r["b"]] if k == "comparison" else []
        if any(run_at.get(u, -1) > i for u in uses):
            out.add(f"{m['stage']} {m['index']}: a {k} logged before a run it uses")
        if boots and k != "bootstrap" and i > boots[0]:
            out.add(f"{m['stage']} {m['index']}: a {k} logged after the batch's first witness")
    return sorted(out)


def same_record(old, new):
    """Every field of a recomputed record equals the logged one: numbers within TOL, lists elementwise, the rest
    exactly; the timestamp aside."""
    if set(old) - {"timestamp_utc"} != set(new):
        return False
    for k, v in new.items():
        o = old[k]
        if isinstance(v, float) or isinstance(o, float):
            if not (isinstance(o, (int, float)) and abs(o - v) <= TOL):
                return False
        elif isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            if not (isinstance(o, list) and len(o) == len(v) and np.allclose(o, v, rtol=0, atol=TOL)):
                return False
        elif artifacts.dumps(o) != artifacts.dumps(v):
            return False
    return True


def witness_ids(ctx, b, m, judgment, entries):
    """The comparison ids of a batch's witness bootstraps, derived exactly as feature_bootstraps derives them."""
    h = m["h"]["run_id"]
    out = set()
    for route, family, a, c in feature_stage.witnesses(m, judgment, entries):
        ia, ib = virtual(ctx, a, h)[1], virtual(ctx, c, h)[1]
        out.add(artifacts.config_hash({"batch_id": b["batch_id"], "route": route, "family": family, "a": list(a),
                                       "b": list(c), "a_ident": ia, "b_ident": ib, "n_boot": 2000,
                                       "seed": BOOTSTRAP_SEED})[:16])
    return out


def checked_run(ctx, recipe, rid, seeds, task, manifest, inv, bid):
    try:
        rec = run_one(ctx, recipe, seeds, task, manifest, inv, search_id=bid)
    except Exception as e:
        artifacts.append_experiment(ctx.root / runs.LOG_FILE, {
            "record": "run_failure", "task_id": task, "batch_id": bid, "planned_run_id": rid,
            "recipe_sha256": searches.recipe_hash(recipe), "error": f"{type(e).__name__}: {e}", "timestamp_utc": now()})
        raise
    if rec["run_id"] != rid:
        raise AssertionError(f"run id {rec['run_id']} differs from the planned {rid}")
    return rec


def run_features(ctx, stage, index, seeds, task, manifest, inv):
    """Run a logged feature batch or re-tuning round after its preflight: H (and its check), then each comparison's
    comparator and challenger (or each configuration), contributions with H and the matched comparisons; then the
    witnesses' bootstraps and the judgment, logged as a feature_allocation record. Resumable."""
    proto, psha = load_feature_protocol(ctx, task)
    validate_imports(ctx, proto)
    validate_handoff(ctx, proto)
    flog = FeatureLog(ctx, task)
    b = next((x for x in flog.batches if x["stage"] == stage and x["index"] == index), None)
    if b is None:
        raise ValueError(f"{stage} {index} is not planned")
    m = feature_preflight(ctx, proto, psha, flog, b, seeds, manifest)
    if b["batch_id"] in flog.allocations:
        print(f"  {b['batch_id']}: already judged")
        return True
    if b["batch_id"] not in flog.starts:
        if past_cutoff(proto):
            print(f"  {b['batch_id']}: not started, the cutoff {proto['cutoff_utc']} has passed")
            return False
        artifacts.append_experiment(ctx.root / runs.LOG_FILE, {
            "record": "batch_start", "task_id": task, "batch_id": b["batch_id"], "timestamp_utc": now()})
    h = m["h"]["run_id"]
    checked_run(ctx, m["h"]["recipe"], h, seeds, task, manifest, inv, b["batch_id"])
    if not h_check(ctx, proto, h, task):
        raise AssertionError(f"H differs from the P2.1 anchor bank; {task} stops (plan section 18)")
    todo = [(c["comparator"], c["comparator_run_id"]) for c in m.get("comparisons", [])] + \
        [(c["challenger"], c["challenger_run_id"]) for c in m.get("comparisons", [])] + \
        [(r, rid) for u in m.get("units", []) for r, rid in zip(u["recipes"], u["run_ids"])]
    for recipe, rid in dict.fromkeys((artifacts.dumps(r), rid) for r, rid in todo):
        checked_run(ctx, strict_loads(recipe), rid, seeds, task, manifest, inv, b["batch_id"])
    for c in m.get("comparisons", []):
        contributions(ctx, c["challenger_run_id"], h, task)
        compare(ctx, c["challenger_run_id"], c["comparator_run_id"], task)
    for u in m.get("units", []):
        for rid in u["run_ids"]:
            contributions(ctx, rid, h, task)
    entries = feature_entries(ctx, m)
    judgment = feature_stage.judge(m, entries, proto["imports"]["comparisons"]) if stage == "features" \
        else feature_stage.judge_retune(m, entries)
    feature_bootstraps(ctx, b, m, judgment, entries, task)  # persisted before the allocation closes the batch
    artifacts.append_experiment(ctx.root / runs.LOG_FILE, {
        "record": "feature_allocation", "task_id": task, "batch_id": b["batch_id"], "stage": stage, "index": index,
        "judgment": judgment, "timestamp_utc": now()})
    met = sum(1 for v in (judgment.get("comparisons") or {}).values() if v["routes"])
    print(f"  {b['batch_id']}: judged; progress {judgment['progress']}"
          + (f", {met} comparisons meeting a route, allowance -> {judgment['allowance_after']}" if stage == "features"
             else ""))
    return True


def expected_comparison_id(ctx, a, c):
    sa, sb = runs.load_logged(ctx, a)[1], runs.load_logged(ctx, c)[1]
    ident = {"a": a, "b": c, "a_sha256": sa, "b_sha256": sb, "seeds": list(SPLIT_SEEDS), "n_boot": 2000,
             "seed": BOOTSTRAP_SEED}
    return artifacts.config_hash(ident)[:16]


def audit_features(ctx, seeds, task, manifest, allow_incomplete=False):
    """Without refitting (plan section 18.10): the import receipt and P2.3's handoff; the log's structure (one
    initial batch, contiguous indices, re-tuning after the features, a valid stop record whose omitted expansion
    re-plans, at most one allocation, start and stop per batch); every batch's file, record and re-planning from the
    log as it stood; for every judged batch its start, every declared run logged and audited against its bank, every
    matched comparison, contribution pair, witness bootstrap and its H check present and recomputed in every field,
    the judgment re-derived, and the order batch record, start, first fits, allocation; every batch record after the
    batch records and allocations of the batches before it; H's run, then its check, before the other fits of the
    batch that first declares H; contributions and comparisons after the runs they use, and the witnesses after the
    rest of their batch; no P2.4 run, comparison, contribution, bootstrap, H check or failure record outside
    the declared work, and none duplicated. Completion also requires the initial batch judged, no started batch
    unjudged, and nothing still due before the cutoff. Run at the sources the runs were made with."""
    proto, psha = load_feature_protocol(ctx, task)
    ok, problems = True, []
    try:
        validate_imports(ctx, proto)
        validate_handoff(ctx, proto)
        print("  imports: the receipt, the P2.8 definitions, the banks and the route flags recompute; the aliases "
              "follow P2.3's handoff")
    except (ValueError, KeyError) as e:
        print(f"  imports: DIFFERS: {e}")
        ok = False
    try:
        flog = FeatureLog(ctx, task)
    except ValueError as e:
        print(f"  log: {e}")
        return False
    for stage in ("features", "retune"):
        idx = [b["index"] for b in flog.of(stage)]
        if idx != list(range(1, len(idx) + 1)):
            problems.append(f"{stage} indices {idx} are not 1, 2, ...")
    if flog.of("retune") and not flog.of("features"):
        problems.append("re-tuning without a feature batch")
    problems += batch_order(flog)
    rid_of = run_id_of(ctx, seeds, manifest)
    if flog.stop is not None:
        n = len(flog.of("features")) + 1
        due = plan_feature_batch(ctx, proto, flog.upto(flog.position(flog.stop)), "features", n, rid_of,
                                 flog.stop["timestamp_utc"])
        omitted = [c["id"] for c in due["comparisons"]] if due else []
        if omitted != flog.stop["omitted_expansion"]:
            problems.append("the feature_stop record's omitted expansion differs from its re-planning")
    logged_runs = {r["run_id"]: r for r in runs.logged(ctx)}
    planned, hs, evidence = [], set(), {}
    want_comp, want_contrib, want_boot = set(), set(), set()
    for b in flog.batches:
        try:
            m = feature_preflight(ctx, proto, psha, flog, b, seeds, manifest)
        except (ValueError, KeyError) as e:
            print(f"  {b['batch_id']}: DIFFERS: {e}")
            ok = False
            continue
        print(f"  {b['batch_id']}: equals its file, record and re-planning")
        first = [r for r in b["run_ids"] if r not in planned]
        planned += first
        bid = b["batch_id"]
        evidence[bid] = execution_evidence(flog.records, b, m, first)
        problems += evidence_order(flog, b, evidence[bid]) + h_order(flog.records, m, first, evidence[bid]) \
            + derived_order(flog.records, m, evidence[bid])
        if bid not in flog.allocations:
            continue
        hs.add(m["h"]["run_id"])
        missing = [r for r in b["run_ids"] if r not in logged_runs]
        if missing:
            problems.append(f"{bid}: judged with runs missing {missing}")
            continue
        entries = feature_entries(ctx, m)
        j = feature_stage.judge(m, entries, proto["imports"]["comparisons"]) if b["stage"] == "features" \
            else feature_stage.judge_retune(m, entries)
        good = artifacts.dumps(j) == artifacts.dumps(flog.allocations[bid]["judgment"])
        good &= feature_bootstraps(ctx, b, m, j, entries, task, verify=True)
        want_boot |= witness_ids(ctx, b, m, j, entries)
        for c in m.get("comparisons", []):
            want_comp.add(expected_comparison_id(ctx, c["challenger_run_id"], c["comparator_run_id"]))
        need = [c["challenger_run_id"] for c in m.get("comparisons", [])] + [r for u in m.get("units", [])
                                                                            for r in u["run_ids"]]
        for rid in need:
            want_contrib |= {(rid, m["h"]["run_id"], w) for w in allocation.CONTRIBUTION_WEIGHTS}
            if not contributions(ctx, rid, m["h"]["run_id"], task, verify=True):
                problems.append(f"{bid}: contributions of {rid} missing or differ")
        print(f"  allocation {bid}: {'re-judged equal, witnesses recomputed' if good else 'DIFFERS'}")
        ok &= good
    mine = flog.records
    task_runs = [r["run_id"] for r in mine if r.get("record") == "run"]
    stray = sorted(set(task_runs) - set(planned))
    if stray or len(task_runs) != len(set(task_runs)):
        problems.append(f"runs of {task} outside every manifest or duplicated: {stray}")
    comps = [r for r in mine if r.get("record") == "comparison"]
    cids = [r["comparison_id"] for r in comps]
    if set(cids) - want_comp or len(cids) != len(set(cids)) or want_comp - set(cids):
        problems.append(f"matched comparisons: orphan {sorted(set(cids) - want_comp)}, missing "
                        f"{sorted(want_comp - set(cids))}, duplicated {len(cids) != len(set(cids))}")
    contribs = [(r["run_id"], r["h_run_id"], r["weight"]) for r in mine if r.get("record") == "contribution"]
    if set(contribs) - want_contrib or len(contribs) != len(set(contribs)):
        problems.append(f"contributions outside the declared work or duplicated: {sorted(set(contribs) - want_contrib)}")
    boots = [r["comparison_id"] for r in mine if r.get("record") == "bootstrap"]
    if set(boots) - want_boot or len(boots) != len(set(boots)) or want_boot - set(boots):
        problems.append(f"witness bootstraps: orphan {sorted(set(boots) - want_boot)}, missing "
                        f"{sorted(want_boot - set(boots))}, duplicated {len(boots) != len(set(boots))}")
    checks = [r["h_run_id"] for r in mine if r.get("record") == "h_check"]
    if sorted(checks) != sorted(hs):
        problems.append(f"H checks {sorted(checks)} do not match the judged batches' H runs {sorted(hs)}")
    ids = {b["batch_id"] for b in flog.batches}
    if any(r.get("batch_id") not in ids for r in mine if r.get("record") == "run_failure"):
        problems.append("a run_failure record names no batch")
    bad = {r: v for r in task_runs if (v := runs.audit_run(ctx, r))}
    print(f"  runs: {len(task_runs)} audited against their banks and records, {len(bad)} with problems")
    ok &= not bad
    for h in sorted(hs):
        if not h_check(ctx, proto, h, task, verify=True):
            problems.append(f"H check of {h} missing or failed")
    ok &= all(compare(ctx, r["a"], r["b"], None, seeds=r["seeds"], verify=True, comparison_id=r["comparison_id"])
              for r in comps)
    kinds = {r.get("record") for r in mine}
    if kinds - set(FEATURE_RECORDS):
        problems.append(f"unexplained {task} record kinds {sorted(kinds - set(FEATURE_RECORDS))}")
    if not allow_incomplete:
        if not flog.of("features") or flog.of("features")[0]["batch_id"] not in flog.allocations:
            problems.append("the initial feature batch is not judged")
        for b in flog.batches:
            if b["batch_id"] not in flog.allocations:
                if b["batch_id"] in flog.starts or evidence.get(b["batch_id"]):
                    problems.append(f"{b['batch_id']} is partial (started, or with execution evidence)")
                elif not past_cutoff(proto):
                    problems.append(f"{b['batch_id']} is unstarted before the cutoff")
                elif b is not flog.batches[-1]:
                    problems.append(f"{b['batch_id']} is unstarted but not the last batch")
        if not past_cutoff(proto) and all(b["batch_id"] in flog.allocations for b in flog.batches):
            nf, nr = len(flog.of("features")) + 1, len(flog.of("retune")) + 1
            if not flog.of("retune") and not feature_closed(proto, flog) and \
                    plan_feature_batch(ctx, proto, flog, "features", nf, rid_of, now()):
                problems.append("a feature expansion is still due")
            elif plan_feature_batch(ctx, proto, flog, "retune", nr, rid_of, now()):
                problems.append("a re-tuning round is still due")
    for p in problems:
        print(f"  incomplete or inconsistent: {p}")
    return ok and not problems


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
        info = searches.expand_v2(saved["spec"])[1] if saved["spec"]["schema"] == searches.SEARCH_SCHEMA_V2 else None
        good = artifacts.sha256_bytes(path.read_bytes()) == rec["file_sha256"] and saved["recipes"] == recipes \
            and [recipe_id(r) for r in recipes] == rec["recipe_ids"] and searches.search_id(saved["spec"]) == rec["search_id"] \
            and saved["recipe_sha256"] == [searches.recipe_hash(r) for r in recipes] \
            and saved.get("rejected_draws") == info
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


def positive(text):
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("batch indices start at 1")
    return v


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--recipe")
    mode.add_argument("--search")
    mode.add_argument("--blend")
    mode.add_argument("--compare", nargs=2)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--verify-all", action="store_true")
    mode.add_argument("--timing")
    mode.add_argument("--plan-wave", type=int)
    mode.add_argument("--reserve", action="store_true")
    mode.add_argument("--run-wave", type=int)
    mode.add_argument("--audit-task")
    ap.add_argument("--allow-incomplete", action="store_true")
    mode.add_argument("--verify-task")
    mode.add_argument("--plan-features", type=positive)
    mode.add_argument("--run-features", type=positive)
    mode.add_argument("--plan-retune", type=positive)
    mode.add_argument("--run-retune", type=positive)
    mode.add_argument("--audit-features")
    mode.add_argument("--stop-features", action="store_true")
    ap.add_argument("--reason")
    ap.add_argument("--members", nargs="+")
    ap.add_argument("--run-id", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--task")
    ap.add_argument("--root", default=".")
    ap.add_argument("--anchors-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    ctx = runs.Context.load(root)
    manifest, inv = banks.source_manifest(root), invocation(root)
    if not (args.verify or args.verify_all or args.audit_task or args.verify_task or args.audit_features) \
            and not args.task:
        ap.error("--task is required for anything that writes")
    seeds = banks.canonical_seeds(args.seeds or list(SPLIT_SEEDS))
    if args.timing:
        raw = (root / args.timing).read_bytes()
        time_cases(ctx, strict_loads(raw.decode()), seeds, args.task, manifest, inv, artifacts.sha256_bytes(raw))
    elif args.reserve:
        reservation(ctx, args.task, *load_protocol(ctx, args.task), manifest)
    elif args.plan_wave:
        plan_wave(ctx, args.plan_wave, seeds, args.task, manifest)
    elif args.run_wave:
        sys.exit(0 if run_wave(ctx, args.run_wave, seeds, args.task, manifest, inv, args.anchors_only) else 1)
    elif args.plan_features or args.plan_retune:
        plan_features(ctx, "features" if args.plan_features else "retune", args.plan_features or args.plan_retune,
                      seeds, args.task, manifest)
    elif args.run_features or args.run_retune:
        sys.exit(0 if run_features(ctx, "features" if args.run_features else "retune",
                                   args.run_features or args.run_retune, seeds, args.task, manifest, inv) else 1)
    elif args.stop_features:
        stop_features(ctx, args.task, args.reason, seeds, manifest)
    elif args.audit_features:
        sys.exit(0 if audit_features(ctx, seeds, args.audit_features, manifest, args.allow_incomplete) else 1)
    elif args.audit_task:
        sys.exit(0 if audit_task(ctx, seeds, args.audit_task, manifest, args.allow_incomplete) else 1)
    elif args.verify_task:
        sys.exit(0 if verify_task(ctx, args.verify_task, manifest, inv, seeds, args.allow_incomplete) else 1)
    elif args.recipe:
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
            if args.run_id:
                ids, comparisons = args.run_id, []
            elif args.task:
                ids = [r["run_id"] for r in runs.logged(ctx) if r.get("task_id") == args.task]
                if not ids:
                    ap.error(f"the log records no run of task {args.task}")
                comparisons = [r for r in runs.logged(ctx, "comparison") if r.get("task_id") == args.task]
            else:
                ap.error("--verify needs --run-id or --task")
        ok = verify(ctx, ids, manifest, inv)
        for r in comparisons:
            ok &= compare(ctx, r["a"], r["b"], None, seeds=r["seeds"], verify=True, comparison_id=r["comparison_id"])
        if args.verify_all:
            ok &= verify_searches(ctx)
            ok &= verify_submissions(ctx)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
