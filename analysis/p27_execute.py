"""P2.7: the restricted nested evaluation of the frozen menu (plan section 8; the evaluator of section 20.1).

Stages, each a separate process from the project root, in this order:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py smoke
        (a) the real recursion with real fits on a 300-row population, every iteration count shrunk to 20, both
        paths, shapes and coverage checked; (b) one full-size learned procedure on the real outer fold 0 training
        part: every frozen combiner setting learned on rebuilt meta predictions, one learned full refit, a disk
        cache round trip in a fresh cache object with equal predictions. Numbers are neither shown nor kept.
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py measure
        with nothing else running: the outer assignment saved; the 8 menu atoms cross-fitted on the 15 production
        folds and timed, each compared with its logged bank; C_all and C_L; the projection by the plan's formula
        and the pilot's multipliers; the path decided and recorded in artifacts/phase2/p27/measure.json before any
        outer score exists
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py run
        nested.run on the eligible rows with the saved assignment and the recorded path, through the on-disk
        namespace artifacts/phase2/cache/p27_build (an interrupted build resumes from it; entries are published by
        atomic rename); the result saved with its provenance; no outer score computed or printed
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py replay
        the same in a new process from the fresh, empty namespace artifacts/phase2/cache/p27_replay; compared with
        the build (arrays and inner means within 1e-12, selections and tie sets exactly) and against the number of
        fresh cross-fits the menu requires
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py decide
        only after a verified replay: nested.decide on the build result (and on the replay result, which must agree),
        the outer report of plan section 8 and a group bootstrap; artifacts/phase2/p27/decision.json

Outer labels are read for scoring in the decide stage only: by nested.decide, and by this driver's pooled AUC and
bootstrap, which need row-level predictions and labels that decide() does not return. No other stage scores
outer predictions.
"""
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from songpop import artifacts  # noqa: E402
from songpop.metrics import BOOTSTRAP_SEED  # noqa: E402
from songpop.phase2 import banks, nested, procedures as P, runs  # noqa: E402
from songpop.splits import SPLIT_SEEDS  # noqa: E402

MENU = Path("configs/phase2/menu.json")
OUT = Path("artifacts/phase2/p27")
SPLIT = Path("artifacts/phase2/splits/nested_outer_v1.csv")
CACHE = Path("artifacts/phase2/cache")
LOG = Path("artifacts/phase2/experiments.jsonl")
SCHEMA = "p27_nested_v1"
PLAN = (3.2, 16.2)  # fit-equivalents of C_all and C_L (plan section 8)
PILOT = (3.04, 14.16)  # the cost pilot's measured multipliers (Finding 026), a cross-check
MARGIN = 1.25
TOL = 1e-12
BOOT = 2000
SMOKE_ROWS = 300
SMOKE_ITERATIONS = 20


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AtomicCache(P.Cache):
    """procedures.Cache whose on-disk entries are published by a temporary file and an atomic rename, so an
    interrupted build cannot leave a truncated entry behind (Codex round W, item 2). Reads are unchanged: a
    malformed or foreign entry is refused, never treated as a miss."""

    def put(self, request, oof, pred):
        if self.directory is None:
            return super().put(request, oof, pred)
        k = self.key(request)
        self.check(request, oof, pred, k)
        oof, pred = np.array(oof, dtype="float64"), np.array(pred, dtype="float64")
        old = self.memory.get(k)
        if old is not None and not (np.array_equal(old["oof"], oof) and np.array_equal(old["pred"], pred)):
            raise ValueError(f"cache entry {k[:12]} exists with other predictions; refused")
        self.memory[k] = {"request": request, "oof": oof, "pred": pred}
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{k}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if not (np.array_equal(z["oof"], oof) and np.array_equal(z["pred"], pred)):
                    raise ValueError(f"cache file {k[:12]} exists with other predictions; refused")
            return
        tmp = self.directory / f"{k}.npz.tmp-{os.getpid()}"
        with open(tmp, "xb") as f:
            np.savez(f, oof=oof, pred=pred, request_json=np.array(artifacts.dumps(request)))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def entries(ns):
    return sorted(p.name for p in ns.iterdir() if p.suffix == ".npz") if ns.exists() else []


def load_menu():
    raw = MENU.read_bytes()
    return json.loads(raw), artifacts.sha256_bytes(raw)


def population(ctx):
    return P.Population(ctx.rows, ctx.data.y, ctx.groups, "production")


def assignment_text(pop, fold):
    return pd.DataFrame({"train_row_id": pop.row_ids, "outer_fold": fold}).to_csv(index=False)


def check_assignment(frame, pop):
    """A saved assignment against this population: the rows in order, integer folds 0 to 2 covering every row once,
    whole groups, both classes in every outer fold and in every training complement."""
    if list(frame.columns) != ["train_row_id", "outer_fold"] or len(frame) != len(pop.y):
        raise ValueError("the saved assignment has the wrong columns or length")
    if not np.array_equal(frame["train_row_id"].to_numpy(), pop.row_ids):
        raise ValueError("the saved assignment's rows differ from the population's")
    fold = frame["outer_fold"].to_numpy()
    if not np.issubdtype(fold.dtype, np.integer):
        raise ValueError("outer folds must be integers")
    P.check_folds(pop.y, pop.groups, fold, nested.OUTER_FOLDS)
    for k in range(nested.OUTER_FOLDS):
        if len(set(pop.y[fold == k].tolist())) < 2:
            raise ValueError(f"outer fold {k} lacks a class")
    return fold


def outer_assignment(pop, write=False):
    """The saved outer assignment, always equal to a recomputation; written once by the measure stage."""
    fold = nested.outer_folds(pop)
    text = assignment_text(pop, fold)
    if SPLIT.exists():
        if SPLIT.read_text() != text:
            raise AssertionError(f"{SPLIT} differs from the recomputed outer assignment")
    elif write:
        SPLIT.parent.mkdir(parents=True, exist_ok=True)
        SPLIT.write_text(text)
    else:
        raise FileNotFoundError(f"{SPLIT} is missing; the measure stage writes it")
    saved = check_assignment(pd.read_csv(SPLIT), pop)
    assert np.array_equal(saved, fold)
    return saved, artifacts.sha256_bytes(text.encode())


def decide_path(plan_seconds, reserved_hours):
    """The frozen gate: full if build plus empty-cache replay with the margin fits the reservation, else reduced."""
    hours = 2 * plan_seconds * MARGIN / 3600
    return ("full" if hours <= reserved_hours else "reduced"), hours


def other_project_processes():
    me = os.getpid()
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout.splitlines()
    root = str(Path(".").resolve())
    return [line.strip() for line in out
            if root in line and ".venv/bin/python" in line and int(line.split()[0]) != me]


def log(record):
    artifacts.append_experiment(LOG, dict(record, timestamp_utc=now()))


def write_once(path, data):
    if path.exists():
        raise FileExistsError(f"{path} exists; P2.7 results are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return artifacts.sha256_bytes(data)


def save_result(result, stem, provenance):
    import io
    arrays = {"outer_fold": np.asarray(result["outer_fold"]), "row_ids": np.asarray(result["row_ids"])}
    meta = {"schema": SCHEMA, "reduced": bool(result["reduced"]), "folds": [], "provenance": provenance}
    for r in result["folds"]:
        o = r["fold"]
        for k in ("fold_average", "full_refit", "anchor", "validation_row_ids"):
            if k in r:
                arrays[f"fold{o}_{k}"] = np.asarray(r[k])
        meta["folds"].append({"fold": o, "selected": r["selected"], "selection": r["selection"],
                              "n_validation": int(len(r["validation_row_ids"]))})
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    sha_npz = write_once(Path(f"{stem}.npz"), buf.getvalue())
    sha_json = write_once(Path(f"{stem}.json"), (artifacts.dumps(meta) + "\n").encode())
    return {"npz_sha256": sha_npz, "json_sha256": sha_json}


def load_result(stem):
    meta = json.loads(Path(f"{stem}.json").read_text())
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"{stem}.json is not a {SCHEMA} result")
    with np.load(f"{stem}.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    folds = []
    for m in meta["folds"]:
        o = m["fold"]
        rec = {"fold": o, "selected": m["selected"], "selection": m["selection"],
               "validation_row_ids": arrays[f"fold{o}_validation_row_ids"]}
        for k in ("fold_average", "full_refit", "anchor"):
            if f"fold{o}_{k}" in arrays:
                rec[k] = arrays[f"fold{o}_{k}"]
        folds.append(rec)
    return {"outer_fold": arrays["outer_fold"], "row_ids": arrays["row_ids"], "reduced": meta["reduced"],
            "folds": folds, "provenance": meta["provenance"]}


def compare_results(a, b):
    """Build against replay: assignment and rows exactly; per fold the selected id, the tie set and the tie keys
    exactly, the inner means and maximum within TOL, every prediction array within TOL."""
    lines, ok = [], True
    same = bool(np.array_equal(a["outer_fold"], b["outer_fold"]) and np.array_equal(a["row_ids"], b["row_ids"]))
    ok &= same and a["reduced"] == b["reduced"] and len(a["folds"]) == len(b["folds"])
    lines.append(f"assignment and rows {'equal' if same else 'DIFFERENT'}; path {'equal' if a['reduced'] == b['reduced'] else 'DIFFERENT'}")
    for ra, rb in zip(a["folds"], b["folds"]):
        parts = {"selected": ra["selected"] == rb["selected"]}
        sa, sb = ra["selection"], rb["selection"]
        if (sa is None) != (sb is None):
            parts["selection"] = False
        elif sa is not None:
            parts["tie set"] = sa["tied"] == sb["tied"] and sa["tie_keys"] == sb["tie_keys"]
            parts["inner means"] = (set(sa["means"]) == set(sb["means"])
                                    and all(abs(sa["means"][c] - sb["means"][c]) <= TOL for c in sa["means"])
                                    and abs(sa["maximum"] - sb["maximum"]) <= TOL)
        for k in ("fold_average", "full_refit", "anchor"):
            if (k in ra) != (k in rb):
                parts[k] = False
            elif k in ra:
                parts[k] = bool(ra[k].shape == rb[k].shape and np.allclose(ra[k], rb[k], rtol=0, atol=TOL))
        parts["validation rows"] = bool(np.array_equal(ra["validation_row_ids"], rb["validation_row_ids"]))
        ok &= all(parts.values())
        lines.append(f"outer fold {ra['fold']}: " + "; ".join(f"{k} {'equal' if v else 'DIFFERENT'}" for k, v in parts.items()))
    return ok, lines


def expected_fresh_cross_fits(ex, result):
    """A cold full run of this menu: every distinct atom once per outer fold on the inner folds; every learned-channel
    atom once per inner training part at the third level; the learned-channel atoms again on the whole outer
    training part for each fold whose selected candidate is learned (its full refit's meta stage). Full-refit
    atom fits bypass the cache. The reduced path cross-fits nothing."""
    if result["reduced"]:
        return 0
    atoms = {P.recipe_hash(a) for c in ex["candidates"] for a in nested.atoms_of(c)}
    learned = [c for c in ex["candidates"] if c["kind"] == "learned"]
    learned_atoms = {P.recipe_hash(a) for c in learned for ch in c["channels"] for a in ch["atoms"]}
    learned_ids = {c["id"] for c in learned}
    per_fold = len(atoms) + nested.INNER_FOLDS * len(nested.INNER_SEEDS) * len(learned_atoms)
    refits = sum(len(learned_atoms) for r in result["folds"] if r["selected"] in learned_ids)
    return nested.OUTER_FOLDS * per_fold + refits


def require_replay_verified():
    path = OUT / "replay.txt"
    if not path.exists() or "replay: every array within" not in path.read_text():
        raise RuntimeError("the decide stage needs a verified replay (artifacts/phase2/p27/replay.txt)")


def print_selections(result):
    """Inner evidence only: the selected candidate per outer fold, its tie set and their inner means."""
    for r in result["folds"]:
        if r["selection"] is None:
            print(f"  outer fold {r['fold']}: fixed candidate {r['selected']} (reduced path)")
            continue
        s = r["selection"]
        tied = ", ".join(f"{c} {s['means'][c]:.4f}" for c in s["tied"])
        print(f"  outer fold {r['fold']}: selected {r['selected']} (inner mean {s['means'][r['selected']]:.4f}; "
              f"inner maximum {s['maximum']:.4f}); tie set: {tied}")


def shrink(recipe):
    p = dict(recipe["params"])
    for k in ("n_estimators", "iterations"):
        if k in p:
            p[k] = SMOKE_ITERATIONS
    return dict(recipe, params=p)


def shrink_candidate(cand):
    c = dict(cand)
    if cand["kind"] == "atomic":
        c["recipe"] = shrink(cand["recipe"])
    elif cand["kind"] == "blend":
        c["members"] = [dict(m, channel={"id": m["channel"]["id"], "atoms": [shrink(a) for a in m["channel"]["atoms"]]})
                        for m in cand["members"]]
    else:
        c["channels"] = [{"id": ch["id"], "atoms": [shrink(a) for a in ch["atoms"]]} for ch in cand["channels"]]
    return c


def peak_memory_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, kilobytes on Linux
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)


def smoke():
    ctx = runs.Context.load(Path("."))
    menu, _ = load_menu()
    ex = nested.executable_menu(menu)
    small = {"candidates": [shrink_candidate(c) for c in ex["candidates"]]}
    nested.check_menu(small)
    pop = P.Population(ctx.rows.iloc[:SMOKE_ROWS], ctx.data.y[:SMOKE_ROWS].copy(), ctx.groups[:SMOKE_ROWS].copy(),
                       "p27-smoke")
    anchor, fixed = menu["nested_protocol"]["anchor"], menu["production_selection"]["selected"]
    t0 = time.time()
    cache = P.Cache()
    full = nested.run(small, pop, cache, "p27-smoke", runs.packages_for, anchor)
    nested.check_result(full, pop)
    d = nested.decide(full, pop)
    assert d["path"] == "full" and d["decision"] and len(d["fold_average"]) == nested.OUTER_FOLDS
    assert cache.misses == expected_fresh_cross_fits(small, full), (cache.misses, expected_fresh_cross_fits(small, full))
    t1 = time.time()
    red = nested.run(small, pop, P.Cache(), "p27-smoke", runs.packages_for, anchor, reduced=fixed)
    nested.check_result(red, pop)
    assert nested.decide(red, pop)["decision"] is None
    print(f"smoke (a): the real recursion on {len(pop.y)} rows with iteration counts {SMOKE_ITERATIONS}: full path "
          f"{t1 - t0:.0f} s ({cache.misses} atomic cross-fits as expected for this menu, {cache.hits} cache hits), "
          f"reduced path {time.time() - t1:.0f} s; results checked, decisions computed and not shown", flush=True)
    # (b) one full-size learned procedure on the real outer fold 0 training part
    prod = population(ctx)
    fold = nested.outer_folds(prod)
    O = prod.subset(np.flatnonzero(fold != 0), "outer0")
    V = prod.subset(np.flatnonzero(fold == 0), "valid0")
    learned = next(c for c in ex["candidates"] if c["kind"] == "learned")
    settings = sorted({tuple(c["setting"]) for c in ex["candidates"] if c["kind"] == "learned"})
    ns = CACHE / "p27_smoke"
    if ns.exists():
        shutil.rmtree(ns)
    tag = "nested-full:0"
    t2 = time.time()
    cache = AtomicCache(ns)
    states, _ = P.learn_on(O, learned["channels"], settings, P.coordinate_seed(P.META_PROTOCOL, tag, O.name),
                           f"{P.META_PROTOCOL}:{tag}:{O.name}", cache, "p27-smoke", runs.packages_for)
    assert set(states) == set(settings)
    q1 = nested.refit_predict(learned, O, V.rows, cache, "p27-smoke", runs.packages_for, tag)
    t3 = time.time()
    fresh = AtomicCache(ns)  # a new cache object: the meta predictions come back from disk
    q2 = nested.refit_predict(learned, O, V.rows, fresh, "p27-smoke", runs.packages_for, tag)
    assert q1.shape == (len(V.y),) and np.isfinite(q1).all() and q1.min() >= 0 and q1.max() <= 1
    assert np.allclose(q1, q2, rtol=0, atol=TOL) and fresh.misses == 0 and fresh.hits > 0
    print(f"smoke (b): a full-size learned procedure on outer fold 0's training part ({len(O.y)} rows): "
          f"{len(settings)} combiner settings learned and one learned full refit in {t3 - t2:.0f} s "
          f"({cache.misses} meta cross-fits, {cache.hits} hits); repeated from the disk cache in a fresh object in "
          f"{time.time() - t3:.0f} s with equal predictions ({fresh.hits} hits, no miss); peak memory {peak_memory_gb():.2f} GB",
          flush=True)
    shutil.rmtree(ns)


def measure():
    busy = other_project_processes()
    if busy:
        raise RuntimeError("other project processes are running; the measurement needs an idle machine:\n" + "\n".join(busy))
    ctx = runs.Context.load(Path("."))
    pop = population(ctx)
    menu, msha = load_menu()
    fold, ssha = outer_assignment(pop, write=True)
    fold_id = ctx.fold_id(list(SPLIT_SEEDS))
    names = [int(s) for s in SPLIT_SEEDS]  # the production fit contexts, FitContext(split seed, fold), so each atom must equal its logged bank
    res = menu["p27_reservation"]
    env = artifacts.environment(ctx.root)
    load_before = os.getloadavg()
    started = now()
    times, equal = {}, {}
    for rid in res["menu_atom_runs"]:
        recipe = menu["atom_recipes"][rid]
        saved = runs.load_logged(ctx, rid)[0]
        t = time.perf_counter()
        oof, pred = P.atomic_cv(recipe, pop, fold_id, names, ctx.test_rows, P.Cache(), "p27-measure",
                                runs.packages_for(recipe))
        times[rid] = time.perf_counter() - t
        equal[rid] = bool(np.allclose(oof, saved.oof, rtol=0, atol=TOL)
                          and np.allclose(pred, saved.test_by_fold, rtol=0, atol=TOL))
        print(f"  {recipe['name']}: 15 fits {times[rid]:.1f} s; {'equals' if equal[rid] else 'DIFFERS FROM'} its logged bank",
              flush=True)
    load_after = os.getloadavg()
    c_all = sum(times.values())
    c_l = sum(times[r] for r in res["learned_channel_runs"])
    plan_s = PLAN[0] * c_all + PLAN[1] * c_l
    pilot_s = PILOT[0] * c_all + PILOT[1] * c_l
    path, hours = decide_path(plan_s, res["reserved_hours"])
    rec = {"record": "nested_measure", "task_id": "P2.7", "menu_sha256": msha, "split_sha256": ssha,
           "split_file": str(SPLIT), "atom_seconds": times, "atoms_equal_logged_banks": equal,
           "c_all_seconds": c_all, "c_l_seconds": c_l, "plan_multipliers": PLAN, "pilot_multipliers": PILOT,
           "margin": MARGIN, "plan_projection_seconds": plan_s, "pilot_projection_seconds": pilot_s,
           "build_and_replay_with_margin_hours": hours, "reserved_hours": res["reserved_hours"], "path": path,
           "reduced_candidate": menu["production_selection"]["selected"] if path == "reduced" else None,
           "load_average_before": load_before, "load_average_after": load_after, "other_project_processes": busy,
           "threads": env["threads"], "threadpools": env["threadpools"], "cpu_count": env["cpu_count"],
           "started_utc": started, "decided_before_any_outer_score": True, "decided_at_utc": now()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_once(OUT / "measure.json", (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode())
    log(rec)
    print(f"C_all {c_all:.1f} s, C_L {c_l:.1f} s; plan projection {plan_s:.0f} s, pilot projection {pilot_s:.0f} s; "
          f"build and replay with margin {hours:.2f} h against {res['reserved_hours']} reserved: path {path}")
    if not all(equal.values()):
        sys.exit(1)


def gate():
    """The recorded path decision, bound to this menu and this outer assignment."""
    raw = (OUT / "measure.json").read_bytes()
    meas = json.loads(raw)
    ctx = runs.Context.load(Path("."))
    pop = population(ctx)
    menu, msha = load_menu()
    fold, ssha = outer_assignment(pop)
    if meas["menu_sha256"] != msha or meas["split_sha256"] != ssha:
        raise AssertionError("measure.json was recorded for another menu or outer assignment")
    return ctx, pop, menu, msha, fold, ssha, meas, artifacts.sha256_bytes(raw)


def evaluate(replay):
    ctx, pop, menu, msha, fold, ssha, meas, measure_sha = gate()
    stage = "replay" if replay else "build"
    ns = CACHE / f"p27_{stage}"
    before = entries(ns)
    if replay and before:
        raise RuntimeError(f"{ns} holds {len(before)} entries; the replay needs a fresh, empty namespace")
    ns.mkdir(parents=True, exist_ok=True)
    ex = nested.executable_menu(menu)
    anchor = menu["nested_protocol"]["anchor"]
    reduced = meas["reduced_candidate"] if meas["path"] == "reduced" else None
    assert (reduced is None) == (meas["path"] == "full")
    manifest = banks.source_manifest(ctx.root)
    env = artifacts.environment(ctx.root)
    cache = AtomicCache(ns)
    started = now()
    t0 = time.time()
    print(f"{stage}: path {meas['path']}, namespace {ns} ({len(before)} entries at start), started {started}", flush=True)
    result = nested.run(ex, pop, cache, manifest["manifest_sha256"], runs.packages_for, anchor, reduced=reduced,
                        fold=fold)
    nested.check_result(result, pop)
    elapsed = time.time() - t0
    provenance = {"stage": stage, "path": meas["path"], "menu_sha256": msha,
                  "executable_sha256": artifacts.sha256_bytes(artifacts.dumps(ex).encode()), "split_sha256": ssha,
                  "measure_sha256": measure_sha, "source_manifest": manifest, "environment": env,
                  "invocation": sys.argv, "started_utc": started, "finished_utc": now(), "elapsed_seconds": elapsed,
                  "cache_namespace": str(ns), "cache_entries_at_start": len(before),
                  "cache_entries_at_end": len(entries(ns)), "cache_hits": cache.hits, "cache_misses": cache.misses,
                  "expected_fresh_cross_fits_cold": expected_fresh_cross_fits(ex, result)}
    hashes = save_result(result, OUT / f"result_{stage}", provenance)
    log({"record": f"nested_{stage}", "task_id": "P2.7", **{k: v for k, v in provenance.items() if k != "environment"},
         "selected": [r["selected"] for r in result["folds"]], **hashes})
    print_selections(result)
    print(f"{stage}: done in {elapsed / 3600:.2f} h; {cache.misses} atomic cross-fits computed, {cache.hits} cache hits "
          f"(a cold run of this menu needs {provenance['expected_fresh_cross_fits_cold']}); peak memory {peak_memory_gb():.2f} GB",
          flush=True)
    if replay:
        a, b = load_result(OUT / "result_build"), load_result(OUT / "result_replay")
        ok, lines = compare_results(a, b)
        fresh = len(before) == 0 and cache.misses == provenance["expected_fresh_cross_fits_cold"]
        lines.append(f"freshness: namespace empty at start {len(before) == 0}; {cache.misses} cross-fits computed against "
                     f"{provenance['expected_fresh_cross_fits_cold']} expected for a cold run: {'equal' if fresh else 'DIFFERENT'}")
        ok &= fresh
        verdict = "replay: every array within 1e-12, every selection equal, every cross-fit fresh" if ok else "replay: DIFFERENCES FOUND"
        text = "\n".join(lines + [verdict]) + "\n"
        write_once(OUT / "replay.txt", text.encode())
        print(text)
        if not ok:
            sys.exit(1)


def fold_auc_table(result, pop, keys, fold):
    y = pop.y
    return {k: [float(roc_auc_score(y[fold == r["fold"]], r[k])) for r in result["folds"]] for k in keys}


def decide():
    require_replay_verified()
    ctx, pop, menu, msha, fold, ssha, meas, measure_sha = gate()
    result = load_result(OUT / "result_build")
    other = load_result(OUT / "result_replay")
    d = nested.decide(result, pop)
    d2 = nested.decide(other, pop)
    keys = ("full_refit", "anchor") if result["reduced"] else ("fold_average", "full_refit", "anchor")
    agree = d["decision"] == d2["decision"] and all(np.allclose(d[k], d2[k], rtol=0, atol=TOL) for k in keys)
    if not agree:
        raise AssertionError("the replay's decision or outer fold AUCs differ from the build's")
    N = len(pop.y)
    pooled = {k: np.full(N, np.nan) for k in keys}
    for r in result["folds"]:
        pos = np.flatnonzero(fold == r["fold"])
        assert np.array_equal(pop.row_ids[pos], r["validation_row_ids"])
        for k in keys:
            pooled[k][pos] = r[k]
    assert all(np.isfinite(v).all() for v in pooled.values())
    report = {"path": d["path"], "decision": d["decision"], "outer_fold_auc": {k: d[k] for k in keys},
              "outer_mean_auc": {k: float(np.mean(d[k])) for k in keys},
              "pooled_outer_auc": {k: float(roc_auc_score(pop.y, pooled[k])) for k in keys},
              "selected_per_fold": [r["selected"] for r in result["folds"]],
              "tie_sets_per_fold": [None if r["selection"] is None else r["selection"]["tied"] for r in result["folds"]],
              "replay_decision_agrees": agree}
    pairs = ({"selected_minus_anchor": ("fold_average", "anchor"), "refit_minus_fold_average": ("full_refit", "fold_average")}
             if not result["reduced"] else {"fixed_refit_minus_anchor_refit": ("full_refit", "anchor")})
    for name, (ka, kb) in pairs.items():
        report[name] = [float(a - b) for a, b in zip(d[ka], d[kb])]
    uniq, inv = np.unique(pop.groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws, invalid = {name: [] for name in pairs}, 0
    for _ in range(BOOT):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        per = {name: [] for name in pairs}
        bad = False
        for o in range(nested.OUTER_FOLDS):
            sel = rows[fold[rows] == o]
            yb = pop.y[sel]
            if yb.min() == yb.max():
                bad = True
                break
            auc = {k: roc_auc_score(yb, pooled[k][sel]) for k in keys}
            for name, (ka, kb) in pairs.items():
                per[name].append(auc[ka] - auc[kb])
        if bad:
            invalid += 1
            continue
        for name in pairs:
            draws[name].append(float(np.mean(per[name])))
    report["group_bootstrap"] = {name: {"observed_mean_difference": float(np.mean(report[name])), "valid": len(v),
                                        "invalid": invalid, "mean": float(np.mean(v)),
                                        "interval_95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
                                        "share_positive": float(np.mean(np.array(v) > 0))} for name, v in draws.items()}
    report["bootstrap_method"] = (f"{BOOT} resamples of the primary groups with replacement, seed {BOOTSTRAP_SEED}; the "
                                  "three outer folds kept and the mean of their paired AUC differences taken per "
                                  "resample; a resample with a single-class fold is invalid as a whole")
    report["bootstrap_caveat"] = ("conditional on the fixed outer predictions and the realized per-fold selections; it "
                                  "excludes retraining and selection variability and does not rerun the selection")
    report.update({"menu_sha256": msha, "split_sha256": ssha, "measure_sha256": measure_sha,
                   "measure": {k: meas[k] for k in ("c_all_seconds", "c_l_seconds", "plan_projection_seconds",
                                                     "build_and_replay_with_margin_hours", "path")},
                   "decided_at_utc": now()})
    write_once(OUT / "decision.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    log(dict(record="nested_decision", task_id="P2.7", **{k: report[k] for k in
             ("path", "decision", "outer_fold_auc", "outer_mean_auc", "pooled_outer_auc", "selected_per_fold",
              "replay_decision_agrees", "menu_sha256", "split_sha256", "measure_sha256")}))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    {"smoke": smoke, "measure": measure, "run": lambda: evaluate(False), "replay": lambda: evaluate(True),
     "decide": decide}[sys.argv[1]]()
