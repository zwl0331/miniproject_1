"""P2.7: the restricted nested evaluation of the frozen menu (plan section 8; the evaluator of section 20.1).

Stages, each a separate process from the project root, in this order:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py smoke
        the real recursion (real fits) on a 300-row population with every iteration count shrunk to 20, both paths;
        shapes and coverage checked; its numbers are not shown or kept
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py measure
        with nothing else running: the outer assignment saved; the 8 menu atoms cross-fitted on the 15 production
        folds and timed (each compared with its logged bank); C_all and C_L; the projection by the plan's formula
        and the pilot's multipliers; the path (full or reduced) decided and recorded in artifacts/phase2/p27/measure.json
        before any outer score exists
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py run
        nested.run on the eligible rows with the saved assignment, through the on-disk cache namespace
        artifacts/phase2/cache/p27_build (an interrupted build resumes from it); the result saved; no outer score printed
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py replay
        the same with the fresh, empty namespace artifacts/phase2/cache/p27_replay; compared with the build
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py decide
        nested.decide, the outer report of plan section 8 and a group bootstrap; artifacts/phase2/p27/decision.json

Outer labels are read by the decide stage only, through nested.decide and the report built from its inputs.
"""
import json
import os
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
from songpop.phase2 import banks, nested, procedures as P, runs  # noqa: E402
from songpop.splits import SPLIT_SEEDS  # noqa: E402

MENU = Path("configs/phase2/menu.json")
OUT = Path("artifacts/phase2/p27")
SPLIT = Path("artifacts/phase2/splits/nested_outer_v1.csv")
CACHE = Path("artifacts/phase2/cache")
LOG = Path("artifacts/phase2/experiments.jsonl")
PLAN = (3.2, 16.2)  # fit-equivalents of C_all and C_L (plan section 8)
PILOT = (3.04, 14.16)  # the measured multipliers of the cost pilot (Finding 026), a cross-check
MARGIN = 1.25
TOL = 1e-12
BOOT = 2000
SMOKE_ROWS = 300
SMOKE_ITERATIONS = 20


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_menu():
    raw = MENU.read_bytes()
    return json.loads(raw), artifacts.sha256_bytes(raw)


def population(ctx):
    return P.Population(ctx.rows, ctx.data.y, ctx.groups, "production")


def outer_assignment(pop, write=False):
    """The saved outer assignment, always equal to a recomputation; written once by the measure stage."""
    fold = nested.outer_folds(pop)
    text = pd.DataFrame({"train_row_id": pop.row_ids, "outer_fold": fold}).to_csv(index=False)
    if SPLIT.exists():
        if SPLIT.read_text() != text:
            raise AssertionError(f"{SPLIT} differs from the recomputed outer assignment")
    elif write:
        SPLIT.parent.mkdir(parents=True, exist_ok=True)
        SPLIT.write_text(text)
    else:
        raise FileNotFoundError(f"{SPLIT} is missing; the measure stage writes it")
    return fold, artifacts.sha256_bytes(text.encode())


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


def save_result(result, stem):
    arrays = {"outer_fold": np.asarray(result["outer_fold"]), "row_ids": np.asarray(result["row_ids"])}
    meta = {"reduced": bool(result["reduced"]), "folds": []}
    for r in result["folds"]:
        o = r["fold"]
        for k in ("fold_average", "full_refit", "anchor", "validation_row_ids"):
            if k in r:
                arrays[f"fold{o}_{k}"] = np.asarray(r[k])
        meta["folds"].append({"fold": o, "selected": r["selected"], "selection": r["selection"],
                              "n_validation": int(len(r["validation_row_ids"]))})
    import io
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    sha_npz = write_once(Path(f"{stem}.npz"), buf.getvalue())
    sha_json = write_once(Path(f"{stem}.json"), (artifacts.dumps(meta) + "\n").encode())
    return {"npz_sha256": sha_npz, "json_sha256": sha_json}


def load_result(stem):
    meta = json.loads(Path(f"{stem}.json").read_text())
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
            "folds": folds}


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
    t1 = time.time()
    red = nested.run(small, pop, P.Cache(), "p27-smoke", runs.packages_for, anchor, reduced=fixed)
    nested.check_result(red, pop)
    assert nested.decide(red, pop)["decision"] is None
    print(f"smoke: the real recursion ran on {len(pop.y)} rows with iteration counts {SMOKE_ITERATIONS}: full path "
          f"{t1 - t0:.0f} s ({cache.misses} atomic cross-fits, {cache.hits} cache hits), reduced path "
          f"{time.time() - t1:.0f} s; results checked, decisions computed and not shown")


def measure():
    busy = other_project_processes()
    if busy:
        raise RuntimeError("other project processes are running; the measurement needs an idle machine:\n" + "\n".join(busy))
    ctx = runs.Context.load(Path("."))
    pop = population(ctx)
    menu, msha = load_menu()
    fold, ssha = outer_assignment(pop, write=True)
    fold_id = ctx.fold_id(list(SPLIT_SEEDS))
    names = [f"p27-measure:{s}" for s in SPLIT_SEEDS]
    res = menu["p27_reservation"]
    load_before = os.getloadavg()
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
    hours = 2 * plan_s * MARGIN / 3600
    path = "full" if hours <= res["reserved_hours"] else "reduced"
    rec = {"record": "nested_measure", "task_id": "P2.7", "menu_sha256": msha, "split_sha256": ssha,
           "split_file": str(SPLIT), "atom_seconds": times, "atoms_equal_logged_banks": equal,
           "c_all_seconds": c_all, "c_l_seconds": c_l, "plan_projection_seconds": plan_s,
           "pilot_projection_seconds": pilot_s, "build_and_replay_with_margin_hours": hours,
           "reserved_hours": res["reserved_hours"], "path": path, "load_average_before": load_before,
           "load_average_after": load_after, "other_project_processes": busy,
           "decided_before_any_outer_score": True, "decided_at_utc": now()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_once(OUT / "measure.json", (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode())
    log(rec)
    print(f"C_all {c_all:.1f} s, C_L {c_l:.1f} s; plan projection {plan_s:.0f} s, pilot projection {pilot_s:.0f} s; "
          f"build and replay with margin {hours:.2f} h against {res['reserved_hours']} reserved: path {path}")
    if not all(equal.values()):
        sys.exit(1)


def gate():
    """The recorded path decision, bound to this menu and this outer assignment."""
    meas = json.loads((OUT / "measure.json").read_text())
    ctx = runs.Context.load(Path("."))
    pop = population(ctx)
    menu, msha = load_menu()
    fold, ssha = outer_assignment(pop)
    if meas["menu_sha256"] != msha or meas["split_sha256"] != ssha:
        raise AssertionError("measure.json was recorded for another menu or outer assignment")
    return ctx, pop, menu, msha, fold, ssha, meas


def evaluate(replay):
    ctx, pop, menu, msha, fold, ssha, meas = gate()
    stage = "replay" if replay else "build"
    ns = CACHE / f"p27_{stage}"
    if replay and ns.exists() and any(ns.iterdir()):
        raise RuntimeError(f"{ns} is not empty; the replay needs a fresh namespace")
    ns.mkdir(parents=True, exist_ok=True)
    entries_before = len(list(ns.iterdir()))
    ex = nested.executable_menu(menu)
    anchor = menu["nested_protocol"]["anchor"]
    reduced = menu["production_selection"]["selected"] if meas["path"] == "reduced" else None
    manifest = banks.source_manifest(ctx.root)
    cache = P.Cache(ns)
    t0 = time.time()
    print(f"{stage}: path {meas['path']}, namespace {ns} ({entries_before} entries at start), started {now()}", flush=True)
    result = nested.run(ex, pop, cache, manifest["manifest_sha256"], runs.packages_for, anchor, reduced=reduced,
                        fold=fold)
    nested.check_result(result, pop)
    elapsed = time.time() - t0
    hashes = save_result(result, OUT / f"result_{stage}")
    rec = {"record": "nested", "task_id": "P2.7", "stage": stage, "path": meas["path"], "menu_sha256": msha,
           "split_sha256": ssha, "source_manifest_sha256": manifest["manifest_sha256"], "cache_namespace": str(ns),
           "cache_entries_at_start": entries_before, "cache_hits": cache.hits, "cache_misses": cache.misses,
           "elapsed_seconds": elapsed, "selected": [r["selected"] for r in result["folds"]], **hashes}
    log(rec)
    print_selections(result)
    print(f"{stage}: done in {elapsed / 3600:.2f} h; {cache.misses} atomic cross-fits computed, {cache.hits} cache hits",
          flush=True)
    if replay:
        compare_replay(pop)


def compare_replay(pop):
    a, b = load_result(OUT / "result_build"), load_result(OUT / "result_replay")
    lines, ok = [], True
    ok &= bool(np.array_equal(a["outer_fold"], b["outer_fold"]) and np.array_equal(a["row_ids"], b["row_ids"]))
    for ra, rb in zip(a["folds"], b["folds"]):
        same_sel = ra["selected"] == rb["selected"] and artifacts.dumps(ra["selection"]) == artifacts.dumps(rb["selection"])
        parts = {k: bool(np.allclose(ra[k], rb[k], rtol=0, atol=TOL)) for k in ("fold_average", "full_refit", "anchor")
                 if k in ra}
        parts["validation_row_ids"] = bool(np.array_equal(ra["validation_row_ids"], rb["validation_row_ids"]))
        ok &= same_sel and all(parts.values())
        lines.append(f"outer fold {ra['fold']}: selection {'equal' if same_sel else 'DIFFERENT'}; "
                     + "; ".join(f"{k} {'equal' if v else 'DIFFERENT'}" for k, v in parts.items()))
    verdict = "replay: every array within 1e-12 and every selection equal" if ok else "replay: DIFFERENCES FOUND"
    text = "\n".join(lines + [verdict]) + "\n"
    write_once(OUT / "replay.txt", text.encode())
    print(text)
    if not ok:
        sys.exit(1)


def decide():
    ctx, pop, menu, msha, fold, ssha, meas = gate()
    result = load_result(OUT / "result_build")
    d = nested.decide(result, pop)
    keys = ("full_refit", "anchor") if result["reduced"] else ("fold_average", "full_refit", "anchor")
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
              "tie_sets_per_fold": [None if r["selection"] is None else r["selection"]["tied"] for r in result["folds"]]}
    if not result["reduced"]:
        report["selected_minus_anchor"] = d["selected_minus_anchor"]
        report["refit_minus_fold_average"] = d["refit_minus_fold_average"]
        groups = pop.groups
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        rng = np.random.default_rng(P.coordinate_seed("p27-bootstrap", ssha))
        diffs = {"selected_minus_anchor": [], "refit_minus_fold_average": []}
        for _ in range(BOOT):
            rows = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
            yb = pop.y[rows]
            if yb.min() == yb.max():
                continue
            fa, fr, an = (roc_auc_score(yb, pooled[k][rows]) for k in ("fold_average", "full_refit", "anchor"))
            diffs["selected_minus_anchor"].append(fa - an)
            diffs["refit_minus_fold_average"].append(fr - fa)
        report["group_bootstrap"] = {
            k: {"resamples": len(v), "mean": float(np.mean(v)), "interval_95": [float(np.quantile(v, 0.025)),
                                                                                float(np.quantile(v, 0.975))],
                "share_positive": float(np.mean(np.array(v) > 0))} for k, v in diffs.items()}
        report["bootstrap_caveat"] = ("a group bootstrap of the pooled outer predictions, conditional on the realized "
                                      "per-fold selections; it does not rerun the selection, so it describes the "
                                      "sampling variability of the chosen predictions, not of the procedure")
    replay_ok = (OUT / "replay.txt").exists() and "every array within" in (OUT / "replay.txt").read_text()
    report.update({"menu_sha256": msha, "split_sha256": ssha, "measure": {k: meas[k] for k in
                   ("c_all_seconds", "c_l_seconds", "plan_projection_seconds", "build_and_replay_with_margin_hours",
                    "path")}, "replay_verified": replay_ok, "decided_at_utc": now()})
    write_once(OUT / "decision.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    log(dict(record="nested_decision", task_id="P2.7", **{k: report[k] for k in
             ("path", "decision", "outer_mean_auc", "pooled_outer_auc", "selected_per_fold", "replay_verified",
              "menu_sha256", "split_sha256")}))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    {"smoke": smoke, "measure": measure, "run": lambda: evaluate(False), "replay": lambda: evaluate(True),
     "decide": decide}[sys.argv[1]]()
