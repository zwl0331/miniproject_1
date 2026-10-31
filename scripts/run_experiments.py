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
from songpop.phase2 import allocation, banks, batches, protocol, runs, searches
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
             "P2.3": Path("configs/phase2/p2.3/protocol.json")}  # plan section 17
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
    if not (args.verify or args.verify_all or args.audit_task or args.verify_task) and not args.task:
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
