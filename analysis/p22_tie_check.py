"""The tie sensitivity of the P2.2 random forest and Extra Trees scores, and the amended verification of the runs
whose scores do not reproduce exactly (Finding 013; plan section 15.7).

`scripts.run_experiments --verify-task P2.2` reproduced 329 of the 348 P2.2 runs exactly; the other 19, all random
forest or Extra Trees, differed in their AUC-valued fields only (its output: artifacts/phase2/verify_task_p22.txt).
This script
(1) shows the cause: a forest's predict_proba sums its trees' probabilities in parallel threads (n_jobs 4), so the
summation order, and the last bits of a prediction, vary between calls, and exactly tied predictions can then break
either way; with n_jobs 1 the predictions repeat exactly.
(2) Verifies the 19 runs under the amended criterion of plan section 15.7, REALIZATIONS times each: a refit must
equal its saved bank in every array and stable metadata field except the fold AUCs, the predictions within 1e-12, and
its rebuilt log record must equal the logged one in every field except the AUC-valued ones; and every P2.2 decision
must be unchanged when the refit's scores and contribution blends replace the logged ones. The decisions are rebuilt
as a counterfactual chain: wave 1 is re-judged from the substituted runs; each later wave is re-planned from the
substituted runs and the re-judged allocations of the waves before it (refreshing every comparator: incumbents, G,
previous best blends, refinement families and parents, range extensions), its manifests compared with the logged
ones, and each of its batches judged against its refreshed state; wave 4 must again be empty.
(3) A negative control: the same chain with the Extra Trees wave 3 incumbent's score lowered by 0.01 must detect a
change, so a comparator the chain failed to refresh could not pass unnoticed.
(4) A guaranteed bound, without fitting, for every P2.2 random forest and Extra Trees run: the range of each fold AUC
over every realization whose predictions lie within 1e-12 of the saved ones. The realizations of (2) are samples;
this range bounds any realization that meets the prediction criterion.
Run ids, refits and re-planning use the source manifest recorded by every P2.2 run (that of
scripts.run_experiments), after checking that its files on disk are unchanged: a manifest computed in this process
would name this script and its own module set, and so give every re-planned configuration a different run id.
Fits models; writes nothing but its standard output; exits 1 if a criterion of (2) or (3) fails. From the project
root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -I analysis/p22_tie_check.py . > analysis/p22_tie_check_output.txt
"""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))

from songpop import artifacts  # noqa: E402
from songpop.phase2 import allocation, banks, protocol, runs  # noqa: E402
from songpop.phase2.encoders import FitContext  # noqa: E402
from songpop.phase2.engine import fit_recipe  # noqa: E402
from songpop.phase2.recipes import strict_loads  # noqa: E402
from songpop.splits import SPLIT_SEEDS  # noqa: E402

REALIZATIONS = 3
TASK = "P2.2"
DECISION = ("routes", "progress", "allowance_after")
PREDICTIONS = ("oof", "test_by_fold", "test_mean")
FORESTS = ("random_forest", "extra_trees")
TOL = banks.TOL
BANK_SCORES = {"fold_auc"}
RECORD_SCORES = {f"logged {k}" for k in runs.ROUNDED + runs.UNROUNDED + ("diagnostics",)}


def differing_runs():
    lines = (ROOT / "artifacts/phase2/verify_task_p22.txt").read_text().splitlines()
    return [l.split(":")[0].strip() for l in lines if "DIFFERS" in l]


def logged_manifest(ctx, records, h):
    """The source manifest every P2.2 run recorded, taken from H's bank, its files on disk checked unchanged."""
    hashes = {r["source_manifest_hash"] for r in records if r.get("record") == "run" and r.get("task_id") == TASK}
    m = runs.load_logged(ctx, h)[0].metadata["source_manifest"]
    assert hashes == {m["manifest_sha256"]}, f"P2.2 runs record source manifests {sorted(hashes)}"
    assert artifacts.source_manifest(ctx.root, list(m["files"])) == m, "a source file changed since P2.2 ran"
    return m


def cause(ctx, log, rid):
    """Repeated predictions of one fitted forest of the run, with four threads and with one."""
    recipe = log.manifest_recipe(rid)
    tr = np.flatnonzero(ctx.fold_id([155])[0] != 0)
    f = fit_recipe(recipe, ctx.rows.iloc[tr], ctx.data.y[tr], ctx.groups[tr], FitContext(155, 0))
    est = f.estimators[0]
    out = {}
    for jobs in (4, 1):
        est.set_params(n_jobs=jobs)
        out[jobs] = len({f.predict_high(ctx.test_rows).tobytes() for _ in range(20)})
    return out


def without_aucs(diag):
    """A run's diagnostics without their AUC values: the era rows, High counts and notes."""
    return {"era_pooled_oof_auc": {e: {k: v for k, v in d.items() if k != "pooled_oof_auc_by_repeat"}
                                   for e, d in diag["era_pooled_oof_auc"].items()},
            "selection_eligible": diag["selection_eligible"]}


def outside_criterion(ctx, saved, sha, rec, new):
    """What a refit changes beyond the amended criterion: any bank field but the fold AUCs (the predictions within
    TOL), and any log record field but the AUC-valued ones."""
    bad = [f for f in banks.compare_v2(new, saved) if f not in BANK_SCORES]
    fresh = runs.run_record(ctx, new, rec["bank_path"], sha)
    bad += [f for f in runs.record_problems(rec, fresh) if f not in RECORD_SCORES]
    if artifacts.dumps(without_aucs(rec["diagnostics"])) != artifacts.dumps(without_aucs(fresh["diagnostics"])):
        bad.append("diagnostics beyond their AUC values")
    return bad


def tie_bounds(bank):
    """Each fold AUC's range over every realization whose predictions lie within TOL of the saved ones: a
    positive-negative pair whose saved predictions differ by at most 2 x TOL can then come out in either order or
    tied, every other pair keeps its order. The saved AUC, recomputed with ties as halves, is checked first."""
    R, K = bank.fold_auc.shape
    lo, hi = np.empty((R, K)), np.empty((R, K))
    for r in range(R):
        for k in range(K):
            va = bank.fold_id[r] == k
            p, y = bank.oof[r, va], bank.y[va]
            d = p[y == 1][:, None] - p[y == 0][None, :]
            assert abs((d > 0).mean() + 0.5 * (d == 0).mean() - bank.fold_auc[r, k]) < 1e-12
            lo[r, k], hi[r, k] = (d > 2 * TOL).mean(), (d >= -2 * TOL).mean()
    return lo, hi


def worst_case(ctx, base):
    """Prints the guaranteed ranges of (4) for every P2.2 forest run, and whether each family's best configuration
    stays best over them."""
    print(f"guaranteed ranges over every realization within {TOL:g} of the saved predictions (no fitting):")
    for fam in FORESTS:
        rows = []
        for r in base.runs.values():
            if r["family"] != fam:
                continue
            bank = runs.load_logged(ctx, r["run_id"])[0]
            lo, hi = tie_bounds(bank)
            rows.append((r["run_id"], float(bank.fold_auc.mean()), float(lo.mean()), float(hi.mean()),
                         float(np.max(np.maximum(hi - bank.fold_auc, bank.fold_auc - lo)))))
        rows.sort(key=lambda t: -t[1])
        best, rest = rows[0], rows[1:]
        rival = max(rest, key=lambda t: t[3])
        fold = max(rows, key=lambda t: t[4])
        mean = max(rows, key=lambda t: max(t[3] - t[1], t[1] - t[2]))
        print(f"  {fam}: {len(rows)} runs, {sum(t[3] > t[2] for t in rows)} with tolerance-ambiguous pairs; "
              f"largest possible change of a fold AUC {fold[4]:.2e} ({fold[0]}), of a mean fold AUC "
              f"{max(mean[3] - mean[1], mean[1] - mean[2]):.2e} ({mean[0]})")
        print(f"  {fam}: best {best[0]} at {best[1]:.10f}, guaranteed range [{best[2]:.10f}, {best[3]:.10f}]; "
              f"runner-up {rest[0][0]} at {rest[0][1]:.10f}; highest upper end among the others {rival[3]:.10f} "
              f"({rival[0]}): the best is {'guaranteed' if best[2] > rival[3] else 'NOT guaranteed'} to stay best")


def identity(m):
    """What a manifest decides, without the comparators' numeric values: the configurations, their run ids, the
    sources (seeds, parents, extensions, rejection counts) and which runs the state names."""
    st = m["state"]
    return {"family": m["family"], "kind": m["kind"], "run_ids": m["run_ids"],
            "recipes": [protocol.recipe_hash(r) for r in m["recipes"]], "sources": m["sources"],
            "state_runs": {k: (st[k].get("run_id"), st[k].get("weight")) for k in st if isinstance(st[k], dict)},
            "allowance_before": st["allowance_before"]}


def chain(records, read, proto, run_id_of, orig_alloc):
    """The counterfactual chain over the given records: returns the list of differences from the logged decisions
    (empty when every decision and planned batch is unchanged)."""
    base = protocol.Log(records, TASK, read)
    waves = sorted({b["wave"] for b in base.batches})
    kept = [r for r in records if r.get("record") != "allocation"]
    alloc = {}
    problems = []

    def log_with(allocs):
        return protocol.Log(kept + list(allocs.values()), TASK, read)
    for w in waves + [waves[-1] + 1]:
        log = log_with(alloc)
        logged = [b for b in base.batches if b["wave"] == w]
        if w > 1:
            planned = {m["family"]: m for m in protocol.plan_wave(proto, log, w, run_id_of)}
            if sorted(planned) != sorted(b["family"] for b in logged):
                problems.append(f"wave {w}: families {sorted(planned)} instead of {sorted(b['family'] for b in logged)}")
                continue
            for b in logged:
                m = dict(planned[b["family"]])
                saved = log.read_manifest(b["file"])
                if artifacts.dumps(identity(m)) != artifacts.dumps(identity(saved)):
                    problems.append(f"wave {w} {b['family']}: the re-planned batch differs")
                b_state = m["state"]
                res = protocol.judge(dict(b, state=b_state), log)
                alloc[b["batch_id"]] = {"record": "allocation", "task_id": TASK, "batch_id": b["batch_id"], **res}
        else:
            for b in logged:
                res = protocol.judge(b, log)
                alloc[b["batch_id"]] = {"record": "allocation", "task_id": TASK, "batch_id": b["batch_id"], **res}
        for b in logged:
            got = {k: alloc[b["batch_id"]][k] for k in DECISION}
            if artifacts.dumps(got) != artifacts.dumps(orig_alloc[b["batch_id"]]):
                problems.append(f"wave {w} {b['family']}: allocation changed")
    return problems


def main():
    ctx = runs.Context.load(ROOT)
    proto = protocol.load_protocol(ROOT / "configs/phase2/p2.2/protocol.json")
    records = artifacts.read_experiments(ROOT / "artifacts/phase2/experiments.jsonl")
    read = lambda rel: strict_loads((ROOT / rel).read_text())
    base = protocol.Log(records, TASK, read)
    h = protocol.h_run_id(proto, base)
    hb, hsha, _ = runs.load_logged(ctx, h)
    manifest = logged_manifest(ctx, records, h)
    print(f"source manifest: {manifest['manifest_sha256']}, recorded by every P2.2 run; its "
          f"{len(manifest['files'])} files on disk unchanged")
    seeds = list(SPLIT_SEEDS)
    run_id_of = lambda r: banks.make_run_id(r["name"], banks.stage_of(seeds), runs.fitted_config(ctx, r, seeds, manifest))
    ids = differing_runs()
    families = sorted({base.manifest_recipe(r)["family"] for r in ids})
    print(f"{len(ids)} runs differed in the refit, families {families}")
    probe = max(ids, key=lambda r: 1 - len(np.unique(runs.load_logged(ctx, r)[0].oof[0])) / len(ctx.data.y))
    c = cause(ctx, base, probe)
    print(f"cause, {probe}: distinct outputs of 20 identical predict calls: {c[4]} with n_jobs 4, {c[1]} with n_jobs 1")
    orig_alloc = {b: {k: a[k] for k in DECISION} for b, a in base.allocations.items()}
    assert not chain(records, read, proto, run_id_of, orig_alloc), "the chain must reproduce the logged decisions"
    print("the chain on the logged records reproduces every logged decision and manifest")
    worst_case(ctx, base)
    ok, worst = True, {"pred": 0.0, "fold": 0.0, "mean": 0.0, "blend": 0.0}
    for k in range(REALIZATIONS):
        recs = copy.deepcopy(records)
        byid = {r["run_id"]: r for r in recs if r.get("record") == "run"}
        bad_fields = []
        for rid in ids:
            saved, sha, rec = runs.load_logged(ctx, rid)
            md = saved.metadata
            new = runs.fitted_run(ctx, md["recipe"], md["split_seeds"], md["task_id"], manifest, {"argv": ["tie_check"]},
                                  md.get("search_id"))
            bad = outside_criterion(ctx, saved, sha, rec, new)
            if bad:
                bad_fields.append((rid, bad))
            worst["pred"] = max([worst["pred"]] + [float(np.max(np.abs(getattr(new, f) - getattr(saved, f))))
                                                   for f in PREDICTIONS])
            worst["fold"] = max(worst["fold"], float(np.max(np.abs(new.fold_auc - saved.fold_auc))))
            worst["mean"] = max(worst["mean"], abs(float(new.fold_auc.mean() - saved.fold_auc.mean())))
            byid[rid]["mean_fold_auc_unrounded"] = float(new.fold_auc.mean())
            byid[rid]["repeat_mean_auc_unrounded"] = new.fold_auc.mean(axis=1).tolist()
            for c_ in recs:
                if c_.get("record") == "contribution" and c_["run_id"] == rid and not c_["identity"]:
                    res = allocation.contribution(hb, hsha, new, "realization", c_["weight"])
                    worst["blend"] = max(worst["blend"], abs(res["mean_fold_auc"] - c_["mean_fold_auc_unrounded"]))
                    c_["mean_fold_auc_unrounded"], c_["repeat_mean_auc_unrounded"] = res["mean_fold_auc"], res["repeat_mean_auc"]
        problems = chain(recs, read, proto, run_id_of, orig_alloc)
        print(f"realization {k + 1}: fields outside the criterion: {bad_fields or 'none'}; "
              f"decision or planning differences: {problems or 'none'}")
        ok &= not bad_fields and not problems
        if k == 0:
            control = copy.deepcopy(recs)
            et3 = next(b for b in base.batches if b["family"] == "extra_trees" and b["wave"] == 3)
            inc = et3["state"]["incumbent"]["run_id"]
            for r in control:
                if r.get("record") == "run" and r["run_id"] == inc:
                    r["mean_fold_auc_unrounded"] -= 0.01
                    r["repeat_mean_auc_unrounded"] = [x - 0.01 for x in r["repeat_mean_auc_unrounded"]]
            detected = chain(control, read, proto, run_id_of, orig_alloc)
            print(f"negative control (the Extra Trees wave 3 incumbent {inc} lowered by 0.01): "
                  f"{'detected: ' + '; '.join(detected) if detected else 'NOT DETECTED'}")
            ok &= bool(detected)
    print(f"largest observed differences from the saved banks over {REALIZATIONS} realizations of the {len(ids)} runs "
          f"(observed, not guaranteed bounds): predictions {worst['pred']:.1e}, a fold AUC {worst['fold']:.2e}, "
          f"a mean fold AUC {worst['mean']:.2e}, a contribution blend's mean fold AUC {worst['blend']:.2e}")
    print("amended verification of the 19 runs:", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
