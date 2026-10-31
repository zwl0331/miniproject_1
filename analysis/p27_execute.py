"""P2.7: the restricted nested evaluation of the frozen menu (plan section 8; the evaluator of section 20.1).

Stages, each a separate process from the project root, in this order:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py smoke
        (a) the real recursion with real fits on a 300-row population, every iteration count shrunk to 20, both
        paths, shapes and coverage checked (nested.decide runs on these smoke populations, the one place outside the
        decide stage that scores outer predictions; they are not the evaluation and nothing is kept); (b) one
        full-size learned procedure on the real outer fold 0 training part with a disk cache round trip
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py measure
        with nothing else running: the outer assignment saved; the 8 menu atoms cross-fitted on the 15 production
        folds and timed, each compared with its logged bank; the projection and the path, validated before they
        are recorded in artifacts/phase2/p27/measure.json, before any outer score exists
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py preflight
        the recorded measurement and the saved assignment re-validated against the menu, the population and the
        log, without fitting anything
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py run
        nested.run on the eligible rows with the saved assignment and the recorded path, through the on-disk
        namespace artifacts/phase2/cache/p27_build (resumable; entries published by atomic rename); the result
        saved with its provenance; no outer score computed or printed
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py replay
        the same in a new process from the fresh, empty namespace artifacts/phase2/cache/p27_replay; then both
        results are validated and compared, the fresh cross-fits counted, and a verification receipt written
        (artifacts/phase2/p27/verification.json) that binds both result files by hash to their log records
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p27_execute.py decide
        only with a passing receipt whose hashes still match the files: nested.decide on the build result and on
        the replay result (which must agree), the outer report of plan section 8 and a group bootstrap;
        artifacts/phase2/p27/decision.json, bound to the receipt

Outer labels of the evaluation are read for scoring in the decide stage only: by nested.decide, and by this
driver's pooled AUC and bootstrap, which need row-level predictions and labels that decide() does not return.
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
from songpop.phase2.selection import TIE_BAND, TIE_KEYS  # noqa: E402
from songpop.splits import SPLIT_SEEDS  # noqa: E402

MENU = Path("configs/phase2/menu.json")
OUT = Path("artifacts/phase2/p27")
SPLIT = Path("artifacts/phase2/splits/nested_outer_v1.csv")
CACHE = Path("artifacts/phase2/cache")
LOG = Path("artifacts/phase2/experiments.jsonl")
DRIVER = "analysis/p27_execute.py"
SCHEMA = "p27_nested_v1"
RECEIPT_SCHEMA = "p27_verification_v1"
PLAN = (3.2, 16.2)  # fit-equivalents of C_all and C_L (plan section 8)
PILOT = (3.04, 14.16)  # the cost pilot's measured multipliers (Finding 026), a cross-check
MARGIN = 1.25
TOL = 1e-12
BOOT = 2000
SMOKE_ROWS = 300
SMOKE_ITERATIONS = 20
STABLE = ("path", "menu_sha256", "executable_sha256", "split_sha256", "measure_sha256")
CAVEATS = [
    "exploratory: the menu was informed by earlier exploration on the same production folds (plan section 5.7), "
    "which the nested evaluation does not remove",
    "the inner and third-level fits use smaller populations with thinner date support than production CV, so part "
    "of any decline against the production scores comes from smaller training sets",
    "once the fallback or the deployment rule has used the outer results, they are no longer an untouched estimate "
    "of what they chose",
]


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


def executable(menu):
    ex = nested.executable_menu(menu)
    return ex, artifacts.sha256_bytes(artifacts.dumps(ex).encode())


def candidate_keys(ex):
    """Each candidate's tie-break key, as the selector forms it."""
    return {c["id"]: tuple(c[k] for k in TIE_KEYS) + (c["id"],) for c in ex["candidates"]}


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
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True).stdout.splitlines()
    marks = (".venv/bin/python", "p27_execute", "run_experiments", "p26_execute", "p26_verify_learned")
    return [line.strip() for line in out if any(m in line for m in marks) and int(line.split()[0]) != me]


def log(record):
    artifacts.append_experiment(LOG, dict(record, timestamp_utc=now()))


def records(kind):
    return [json.loads(l) for l in LOG.read_text().splitlines() if json.loads(l).get("record") == kind]


def write_once(path, data):
    """Published by a temporary file and an atomic rename; never overwritten."""
    if path.exists():
        raise FileExistsError(f"{path} exists; P2.7 results are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return artifacts.sha256_bytes(data)


def file_hashes(stem):
    return {"npz_sha256": artifacts.sha256_bytes(Path(f"{stem}.npz").read_bytes()),
            "json_sha256": artifacts.sha256_bytes(Path(f"{stem}.json").read_bytes())}


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


def validate_result(result, pop, fold, keys, reduced_id, stable):
    """A loaded result against the population, the saved assignment, the menu's candidates and the expected stable
    provenance: folds 0, 1, 2 in order holding their saved rows; on the full path each selection complete and
    self-consistent (the selected id, the means over exactly the menu, the maximum, the tie set within TIE_BAND and
    the tie-break winner all recomputed); on the reduced path the fixed candidate and no selection."""
    nested.check_result(result, pop)
    if not np.array_equal(np.asarray(result["outer_fold"]), fold):
        raise ValueError("the result's outer assignment differs from the saved one")
    if [r["fold"] for r in result["folds"]] != list(range(nested.OUTER_FOLDS)):
        raise ValueError("the result's folds are not 0, 1, 2 in order")
    prov = result["provenance"]
    for k, v in stable.items():
        if prov.get(k) != v:
            raise ValueError(f"the result's provenance {k} is {prov.get(k)!r}, expected {v!r}")
    if result["reduced"] != (stable["path"] == "reduced"):
        raise ValueError("the result's path flag contradicts the recorded path")
    for r in result["folds"]:
        if not np.array_equal(np.asarray(r["validation_row_ids"]), pop.row_ids[fold == r["fold"]]):
            raise ValueError(f"outer fold {r['fold']} does not hold its saved validation rows")
        if r["selected"] not in keys:
            raise ValueError(f"outer fold {r['fold']} selected {r['selected']!r}, not a menu candidate")
        s = r["selection"]
        if result["reduced"]:
            if s is not None or r["selected"] != reduced_id or "fold_average" in r:
                raise ValueError(f"outer fold {r['fold']}: a reduced result must fix {reduced_id} without a selection")
            continue
        if s is None or s["selected"] != r["selected"] or set(s["means"]) != set(keys):
            raise ValueError(f"outer fold {r['fold']}: the selection is missing, inconsistent or not over the menu")
        top = max(s["means"].values())
        tied = sorted(c for c, m in s["means"].items() if top - m <= TIE_BAND)
        if abs(s["maximum"] - top) > TOL or s["tied"] != tied or set(s["tie_keys"]) != set(tied):
            raise ValueError(f"outer fold {r['fold']}: the maximum or the tie set does not follow from the means")
        if min(tied, key=lambda c: keys[c]) != r["selected"]:
            raise ValueError(f"outer fold {r['fold']}: the tie-break winner is not the selected candidate")
        for c in tied:
            if list(s["tie_keys"][c]) != list(keys[c]):
                raise ValueError(f"outer fold {r['fold']}: the tie keys of {c} differ from the menu's")


def compare_results(a, b):
    """Build against replay: assignment, rows and fold ids exactly; the stable provenance exactly; the engine
    sources exactly (every manifest entry but this driver); per fold the selected id, the tie set and the tie keys
    exactly, the inner means and maximum within TOL, every prediction array within TOL."""
    lines, ok = [], True
    same = bool(np.array_equal(a["outer_fold"], b["outer_fold"]) and np.array_equal(a["row_ids"], b["row_ids"]))
    same_path = a["reduced"] == b["reduced"] and len(a["folds"]) == len(b["folds"])
    pa, pb = a["provenance"], b["provenance"]
    stable = all(pa.get(k) == pb.get(k) for k in STABLE)
    fa = {p: h for p, h in pa["source_manifest"]["files"].items() if p != DRIVER}
    fb = {p: h for p, h in pb["source_manifest"]["files"].items() if p != DRIVER}
    engine = fa == fb
    ok &= same and same_path and stable and engine
    lines.append(f"assignment and rows {'equal' if same else 'DIFFERENT'}; path {'equal' if same_path else 'DIFFERENT'}; "
                 f"stable provenance {'equal' if stable else 'DIFFERENT'}; engine sources {'equal' if engine else 'DIFFERENT'} "
                 f"(driver {pa['source_manifest']['files'].get(DRIVER, '?')[:12]} in the build, "
                 f"{pb['source_manifest']['files'].get(DRIVER, '?')[:12]} in the replay)")
    for ra, rb in zip(a["folds"], b["folds"]):
        parts = {"fold id": ra["fold"] == rb["fold"], "selected": ra["selected"] == rb["selected"]}
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
    training part for each fold whose selected candidate is learned (its full refit's meta stage). Full-refit atom
    fits bypass the cache. The reduced path of this menu, whose fixed candidate is a fixed blend, cross-fits
    nothing; a learned fixed candidate would need its meta stage per fold."""
    learned = [c for c in ex["candidates"] if c["kind"] == "learned"]
    learned_atoms = {P.recipe_hash(a) for c in learned for ch in c["channels"] for a in ch["atoms"]}
    learned_ids = {c["id"] for c in learned}
    if result["reduced"]:
        fixed = result["folds"][0]["selected"] if result["folds"] else None
        return nested.OUTER_FOLDS * len(learned_atoms) if fixed in learned_ids else 0
    atoms = {P.recipe_hash(a) for c in ex["candidates"] for a in nested.atoms_of(c)}
    per_fold = len(atoms) + nested.INNER_FOLDS * len(nested.INNER_SEEDS) * len(learned_atoms)
    refits = sum(len(learned_atoms) for r in result["folds"] if r["selected"] in learned_ids)
    return nested.OUTER_FOLDS * per_fold + refits


def validate_measure_record(rec, menu):
    """Problems with a measurement record against the menu: the atom sets, finite positive timings, every atom equal
    to its logged bank, the sums, both projections, the margin, the reservation, the path and the reduced candidate
    all recomputed."""
    res, problems = menu["p27_reservation"], []
    atoms, learned = set(res["menu_atom_runs"]), set(res["learned_channel_runs"])
    if set(rec["atom_seconds"]) != atoms or set(rec["atoms_equal_logged_banks"]) != atoms:
        problems.append("the timed atoms are not exactly the menu's atom runs")
    if not learned <= atoms:
        problems.append("the learned-channel runs are not among the menu's atom runs")
    t = rec["atom_seconds"]
    if not all(isinstance(v, float) and np.isfinite(v) and v > 0 for v in t.values()):
        problems.append("a timing is missing, non-finite or not positive")
    if not all(rec["atoms_equal_logged_banks"].values()):
        problems.append("an atom did not reproduce its logged bank")
    c_all, c_l = sum(t.values()), sum(t[r] for r in learned if r in t)
    plan_s, pilot_s = PLAN[0] * c_all + PLAN[1] * c_l, PILOT[0] * c_all + PILOT[1] * c_l
    path, hours = decide_path(plan_s, res["reserved_hours"])
    for name, want, got, tol in (("c_all_seconds", c_all, rec["c_all_seconds"], 1e-9),
                                 ("c_l_seconds", c_l, rec["c_l_seconds"], 1e-9),
                                 ("plan_projection_seconds", plan_s, rec["plan_projection_seconds"], 1e-6),
                                 ("pilot_projection_seconds", pilot_s, rec["pilot_projection_seconds"], 1e-6),
                                 ("build_and_replay_with_margin_hours", hours, rec["build_and_replay_with_margin_hours"], 1e-9)):
        if abs(want - got) > tol:
            problems.append(f"{name} {got} does not follow from the timings ({want})")
    if list(rec["plan_multipliers"]) != list(PLAN) or list(rec["pilot_multipliers"]) != list(PILOT) or rec["margin"] != MARGIN:
        problems.append("the multipliers or the margin differ from the driver's")
    if rec["reserved_hours"] != res["reserved_hours"]:
        problems.append("the reservation differs from the menu's")
    if rec["path"] != path:
        problems.append(f"the recorded path {rec['path']} does not follow from the projection ({path})")
    want_reduced = menu["production_selection"]["selected"] if path == "reduced" else None
    if rec["reduced_candidate"] != want_reduced:
        problems.append(f"the reduced candidate must be {want_reduced!r}")
    return problems


def peak_memory_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, kilobytes on Linux
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)


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
    ex, _ = executable(menu)
    small = {"candidates": [shrink_candidate(c) for c in ex["candidates"]]}
    nested.check_menu(small)
    pop = P.Population(ctx.rows.iloc[:SMOKE_ROWS], ctx.data.y[:SMOKE_ROWS].copy(), ctx.groups[:SMOKE_ROWS].copy(),
                       "p27-smoke")
    anchor, fixed = menu["nested_protocol"]["anchor"], menu["production_selection"]["selected"]
    t0 = time.time()
    cache = P.Cache()
    full = nested.run(small, pop, cache, "p27-smoke", runs.packages_for, anchor)
    nested.check_result(full, pop)
    d = nested.decide(full, pop)  # smoke populations only: not the evaluation; nothing is kept
    assert d["path"] == "full" and d["decision"] and len(d["fold_average"]) == nested.OUTER_FOLDS
    assert cache.misses == expected_fresh_cross_fits(small, full), (cache.misses, expected_fresh_cross_fits(small, full))
    t1 = time.time()
    red = nested.run(small, pop, P.Cache(), "p27-smoke", runs.packages_for, anchor, reduced=fixed)
    nested.check_result(red, pop)
    assert nested.decide(red, pop)["decision"] is None
    print(f"smoke (a): the real recursion on {len(pop.y)} rows with iteration counts {SMOKE_ITERATIONS}: full path "
          f"{t1 - t0:.0f} s ({cache.misses} atomic cross-fits as expected for this menu, {cache.hits} cache hits), "
          f"reduced path {time.time() - t1:.0f} s; results checked, decisions computed and not shown", flush=True)
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
    fresh = AtomicCache(ns)
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
           "c_all_seconds": c_all, "c_l_seconds": c_l, "plan_multipliers": list(PLAN), "pilot_multipliers": list(PILOT),
           "margin": MARGIN, "plan_projection_seconds": plan_s, "pilot_projection_seconds": pilot_s,
           "build_and_replay_with_margin_hours": hours, "reserved_hours": res["reserved_hours"], "path": path,
           "reduced_candidate": menu["production_selection"]["selected"] if path == "reduced" else None,
           "load_average_before": load_before, "load_average_after": load_after, "other_project_processes": busy,
           "threads": env["threads"], "threadpools": env["threadpools"], "cpu_count": env["cpu_count"],
           "started_utc": started, "decided_before_any_outer_score": True, "decided_at_utc": now()}
    problems = validate_measure_record(rec, menu)
    print(f"C_all {c_all:.1f} s, C_L {c_l:.1f} s; plan projection {plan_s:.0f} s, pilot projection {pilot_s:.0f} s; "
          f"build and replay with margin {hours:.2f} h against {res['reserved_hours']} reserved: path {path}")
    if problems:
        print("measurement NOT recorded:\n  " + "\n  ".join(problems))
        sys.exit(1)
    OUT.mkdir(parents=True, exist_ok=True)
    write_once(OUT / "measure.json", (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode())
    log(rec)


def gate():
    """The recorded path decision, re-validated: bound to this menu and this outer assignment, internally consistent,
    and equal to exactly one measurement record in the log."""
    raw = (OUT / "measure.json").read_bytes()
    meas = json.loads(raw)
    ctx = runs.Context.load(Path("."))
    pop = population(ctx)
    menu, msha = load_menu()
    fold, ssha = outer_assignment(pop)
    if meas["menu_sha256"] != msha or meas["split_sha256"] != ssha:
        raise AssertionError("measure.json was recorded for another menu or outer assignment")
    problems = validate_measure_record(meas, menu)
    if problems:
        raise AssertionError("measure.json is not a valid measurement: " + "; ".join(problems))
    logged = [r for r in records("nested_measure")
              if artifacts.dumps({k: v for k, v in r.items() if k != "timestamp_utc"}) == artifacts.dumps(meas)]
    if len(logged) != 1:
        raise AssertionError(f"{len(logged)} log records equal measure.json; exactly one is required")
    return ctx, pop, menu, msha, fold, ssha, meas, artifacts.sha256_bytes(raw)


def preflight():
    ctx, pop, menu, msha, fold, ssha, meas, measure_sha = gate()
    ex, xsha = executable(menu)
    print(f"preflight: measure.json valid (path {meas['path']}, {meas['build_and_replay_with_margin_hours']:.2f} h against "
          f"{meas['reserved_hours']}), bound to menu {msha[:12]} and assignment {ssha[:12]} ({len(fold)} rows, folds "
          f"{[int((fold == k).sum()) for k in range(nested.OUTER_FOLDS)]}); executable menu {xsha[:12]} with "
          f"{len(ex['candidates'])} candidates; exactly one matching log record")


def stable_provenance(meas, msha, xsha, ssha, measure_sha):
    return {"path": meas["path"], "menu_sha256": msha, "executable_sha256": xsha, "split_sha256": ssha,
            "measure_sha256": measure_sha}


def evaluate(replay):
    ctx, pop, menu, msha, fold, ssha, meas, measure_sha = gate()
    stage = "replay" if replay else "build"
    ns = CACHE / f"p27_{stage}"
    before = entries(ns)
    if replay and before:
        raise RuntimeError(f"{ns} holds {len(before)} entries; the replay needs a fresh, empty namespace")
    ns.mkdir(parents=True, exist_ok=True)
    ex, xsha = executable(menu)
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
    if banks.source_manifest(ctx.root) != manifest:
        raise AssertionError("the sources changed while the stage ran; the result is not saved")
    provenance = {"stage": stage, **stable_provenance(meas, msha, xsha, ssha, measure_sha),
                  "source_manifest": manifest, "environment": env,
                  "packages_by_atom": {P.recipe_hash(a): runs.packages_for(a) for c in ex["candidates"] for a in nested.atoms_of(c)},
                  "data_hashes": getattr(ctx.data, "hashes", None), "invocation": sys.argv, "started_utc": started,
                  "finished_utc": now(), "elapsed_seconds": elapsed, "cache_namespace": str(ns),
                  "cache_entries_at_start": len(before), "cache_entries_at_end": len(entries(ns)),
                  "cache_hits": cache.hits, "cache_misses": cache.misses,
                  "expected_fresh_cross_fits_cold": expected_fresh_cross_fits(ex, result)}
    result["provenance"] = provenance
    validate_result(result, pop, fold, candidate_keys(ex), reduced, stable_provenance(meas, msha, xsha, ssha, measure_sha))
    hashes = save_result(result, OUT / f"result_{stage}", provenance)
    log({"record": f"nested_{stage}", "task_id": "P2.7",
         **{k: v for k, v in provenance.items() if k not in ("environment", "packages_by_atom", "source_manifest")},
         "source_manifest_sha256": manifest["manifest_sha256"], "driver_sha256": manifest["files"].get(DRIVER),
         "selected": [r["selected"] for r in result["folds"]], **hashes})
    for r in result["folds"]:
        if r["selection"] is None:
            print(f"  outer fold {r['fold']}: fixed candidate {r['selected']} (reduced path)")
        else:
            s = r["selection"]
            tied = ", ".join(f"{c} {s['means'][c]:.4f}" for c in s["tied"])
            print(f"  outer fold {r['fold']}: selected {r['selected']} (inner mean {s['means'][r['selected']]:.4f}; "
                  f"inner maximum {s['maximum']:.4f}); tie set: {tied}")
    print(f"{stage}: done in {elapsed / 3600:.2f} h; {cache.misses} atomic cross-fits computed, {cache.hits} cache hits "
          f"(a cold run of this menu needs {provenance['expected_fresh_cross_fits_cold']}); peak memory {peak_memory_gb():.2f} GB",
          flush=True)
    if replay:
        verify()


def unique_record(kind, hashes):
    found = [r for r in records(kind) if r.get("npz_sha256") == hashes["npz_sha256"] and r.get("json_sha256") == hashes["json_sha256"]]
    if len(found) != 1:
        raise AssertionError(f"{len(found)} {kind} records carry the hashes of the saved result; exactly one is required")
    return found[0]


def verify():
    """Both saved results validated, compared and bound to their log records; the replay's freshness checked; a
    receipt written that the decide stage requires."""
    ctx, pop, menu, msha, fold, ssha, meas, measure_sha = gate()
    ex, xsha = executable(menu)
    keys, reduced = candidate_keys(ex), (meas["reduced_candidate"] if meas["path"] == "reduced" else None)
    stable = stable_provenance(meas, msha, xsha, ssha, measure_sha)
    stems = {"build": OUT / "result_build", "replay": OUT / "result_replay"}
    hashes = {k: file_hashes(v) for k, v in stems.items()}
    recs = {k: unique_record(f"nested_{k}", hashes[k]) for k in stems}
    results = {k: load_result(v) for k, v in stems.items()}
    for k, r in results.items():
        validate_result(r, pop, fold, keys, reduced, stable)
    ok, lines = compare_results(results["build"], results["replay"])
    rp = results["replay"]["provenance"]
    expected = expected_fresh_cross_fits(ex, results["replay"])
    fresh = rp["cache_entries_at_start"] == 0 and rp["cache_misses"] == expected \
        and recs["replay"]["cache_entries_at_start"] == 0 and recs["replay"]["cache_misses"] == expected
    lines.append(f"freshness: the replay namespace held {rp['cache_entries_at_start']} entries at start and "
                 f"{rp['cache_misses']} cross-fits were computed against {expected} expected for a cold run: "
                 f"{'equal' if fresh else 'DIFFERENT'}")
    ok &= fresh
    verdict = "replay: every array within 1e-12, every selection equal, every cross-fit fresh" if ok else "replay: DIFFERENCES FOUND"
    receipt = {"schema": RECEIPT_SCHEMA, "ok": ok, "verdict": verdict, "lines": lines, **stable,
               "files": {k: {"npz": f"{v}.npz", "json": f"{v}.json", **hashes[k]} for k, v in stems.items()},
               "records": {k: {"timestamp_utc": recs[k]["timestamp_utc"], "driver_sha256": recs[k].get("driver_sha256")}
                           for k in stems},
               "selected_per_fold": [r["selected"] for r in results["build"]["folds"]],
               "freshness": {"replay_entries_at_start": rp["cache_entries_at_start"], "replay_cross_fits": rp["cache_misses"],
                             "expected_cold": expected},
               "engine_sources_equal": bool(lines and "engine sources equal" in lines[0]),
               "verifier_source_manifest_sha256": banks.source_manifest(ctx.root)["manifest_sha256"],
               "verified_at_utc": now()}
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    sha = write_once(OUT / "verification.json", data)
    write_once(OUT / "replay.txt", ("\n".join(lines + [verdict]) + "\n").encode())
    log({"record": "nested_verification", "task_id": "P2.7", "ok": ok, "receipt_sha256": sha, **stable,
         "selected_per_fold": receipt["selected_per_fold"], "freshness": receipt["freshness"]})
    print("\n".join(lines + [verdict]))
    if not ok:
        sys.exit(1)


def require_receipt(stable):
    """The passing receipt, whose hashes must still match the files and whose identity must match the current gate."""
    path = OUT / "verification.json"
    if not path.exists():
        raise RuntimeError("the decide stage needs a verification receipt (artifacts/phase2/p27/verification.json)")
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if receipt.get("schema") != RECEIPT_SCHEMA or not receipt.get("ok"):
        raise RuntimeError("the verification receipt is not a passing receipt")
    for k, v in stable.items():
        if receipt.get(k) != v:
            raise RuntimeError(f"the receipt's {k} does not match the current gate")
    for k, f in receipt["files"].items():
        want = {"npz_sha256": f["npz_sha256"], "json_sha256": f["json_sha256"]}
        if file_hashes(OUT / f"result_{k}") != want:
            raise RuntimeError(f"the {k} result files no longer match the receipt")
        unique_record(f"nested_{k}", want)
    return receipt, artifacts.sha256_bytes(raw)


def decide():
    ctx, pop, menu, msha, fold, ssha, meas, measure_sha = gate()
    ex, xsha = executable(menu)
    stable = stable_provenance(meas, msha, xsha, ssha, measure_sha)
    receipt, receipt_sha = require_receipt(stable)
    keys, reduced = candidate_keys(ex), (meas["reduced_candidate"] if meas["path"] == "reduced" else None)
    result, other = load_result(OUT / "result_build"), load_result(OUT / "result_replay")
    for r in (result, other):
        validate_result(r, pop, fold, keys, reduced, stable)
    ok, _ = compare_results(result, other)
    if not ok:
        raise AssertionError("the build and the replay no longer compare equal")
    d, d2 = nested.decide(result, pop), nested.decide(other, pop)
    keys_auc = ("full_refit", "anchor") if result["reduced"] else ("fold_average", "full_refit", "anchor")
    agree = (d["decision"] == d2["decision"] and d["path"] == d2["path"]
             and [r["selected"] for r in result["folds"]] == [r["selected"] for r in other["folds"]]
             and all(np.allclose(d[k], d2[k], rtol=0, atol=TOL) for k in keys_auc))
    if not agree:
        raise AssertionError("the replay's decision, selections or outer fold AUCs differ from the build's")
    N = len(pop.y)
    pooled = {k: np.full(N, np.nan) for k in keys_auc}
    for r in result["folds"]:
        pos = np.flatnonzero(fold == r["fold"])
        assert np.array_equal(pop.row_ids[pos], r["validation_row_ids"])
        for k in keys_auc:
            pooled[k][pos] = r[k]
    assert all(np.isfinite(v).all() for v in pooled.values())
    report = {"path": d["path"], "decision": d["decision"], "outer_fold_auc": {k: d[k] for k in keys_auc},
              "outer_mean_auc": {k: float(np.mean(d[k])) for k in keys_auc},
              "pooled_outer_auc": {k: float(roc_auc_score(pop.y, pooled[k])) for k in keys_auc},
              "selected_per_fold": [r["selected"] for r in result["folds"]],
              "tie_sets_per_fold": [None if r["selection"] is None else r["selection"]["tied"] for r in result["folds"]],
              "replay_agrees": agree}
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
        per, bad = {name: [] for name in pairs}, False
        for o in range(nested.OUTER_FOLDS):
            sel = rows[fold[rows] == o]
            if len(sel) == 0 or pop.y[sel].min() == pop.y[sel].max():
                bad = True
                break
            auc = {k: roc_auc_score(pop.y[sel], pooled[k][sel]) for k in keys_auc}
            for name, (ka, kb) in pairs.items():
                per[name].append(auc[ka] - auc[kb])
        if bad:
            invalid += 1
            continue
        for name in pairs:
            draws[name].append(float(np.mean(per[name])))
    if any(len(v) == 0 for v in draws.values()):
        raise RuntimeError(f"no valid bootstrap resample among {BOOT}; the intervals are unusable")
    report["group_bootstrap"] = {name: {"observed_mean_difference": float(np.mean(report[name])), "valid": len(v),
                                        "invalid": invalid, "mean": float(np.mean(v)),
                                        "interval_95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
                                        "share_positive": float(np.mean(np.array(v) > 0))} for name, v in draws.items()}
    report["bootstrap_method"] = (f"{BOOT} resamples of the primary groups with replacement (numpy default_rng, seed "
                                  f"{BOOTSTRAP_SEED}; the project's helper is not reused, so the draws differ from its "
                                  "stream); the three outer folds kept and the mean of their paired AUC differences "
                                  "taken per resample; a resample with an empty or single-class fold is invalid as a whole")
    report["bootstrap_caveat"] = ("conditional on the fixed outer predictions and the realized per-fold selections; it "
                                  "excludes retraining and selection variability and does not rerun the selection")
    report["caveats"] = CAVEATS
    report.update({**stable, "receipt_sha256": receipt_sha, "verified_at_utc": receipt["verified_at_utc"],
                   "reporting_source_manifest_sha256": banks.source_manifest(ctx.root)["manifest_sha256"],
                   "measure": {k: meas[k] for k in ("c_all_seconds", "c_l_seconds", "plan_projection_seconds",
                                                     "build_and_replay_with_margin_hours", "path")},
                   "decided_at_utc": now()})
    data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    dsha = write_once(OUT / "decision.json", data)
    log(dict(record="nested_decision", task_id="P2.7", decision_sha256=dsha, **{k: report[k] for k in
             ("path", "decision", "outer_fold_auc", "outer_mean_auc", "pooled_outer_auc", "selected_per_fold",
              "replay_agrees", "receipt_sha256", "menu_sha256", "split_sha256", "measure_sha256")}))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    {"smoke": smoke, "measure": measure, "preflight": preflight, "run": lambda: evaluate(False),
     "replay": lambda: evaluate(True), "verify": verify, "decide": decide}[sys.argv[1]]()
