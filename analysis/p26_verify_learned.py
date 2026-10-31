"""The complete replay of P2.6b's learned procedures (plan section 20.4; R2 review round 1, F2), writing nothing but its
standard output. Every atom of every learned channel is refitted on the 15 production outer folds under the current
sources and compared with its logged bank (predictions and fold AUCs within 1e-12); each channel's outer predictions
are then rebuilt from those refits (equal averages inside a within-family channel), never read from a saved bank;
every learned procedure is recomputed from an empty in-memory cache (every meta-fold fit redone) and compared with its
logged procedure bank (arrays within 1e-12, combiner states exactly). From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p26_verify_learned.py > artifacts/phase2/verify_learned_p26.txt
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

from songpop import artifacts  # noqa: E402
from songpop.phase2 import banks, procedures, runs  # noqa: E402
from songpop.phase2.engine import run_recipe_cv  # noqa: E402
from songpop.splits import SPLIT_SEEDS  # noqa: E402

PROCS = Path("configs/phase2/p2.6/procedures.json")
TOL = 1e-12


def close(a, b):
    return np.allclose(a, b, rtol=0, atol=TOL)


def main():
    ctx = runs.Context.load(Path("."))
    raw = PROCS.read_bytes()
    psha = artifacts.sha256_bytes(raw)
    doc = json.loads(raw)
    chans = [c for c in doc["channels"] if c["id"] in doc["learned_channels"]]
    fold_id = ctx.fold_id(SPLIT_SEEDS)
    ok, refit = True, {}
    print(f"P2.6b learned-procedure replay; manifest {psha}")
    for ch in chans:
        for rid, recipe in zip(ch["run_ids"], ch["atoms"]):
            saved, sha, _ = runs.load_logged(ctx, rid)
            t0 = time.time()
            fresh = run_recipe_cv(recipe, ctx.rows, ctx.data.y, ctx.data.row_ids, ctx.groups, fold_id,
                                  list(SPLIT_SEEDS), ctx.test_rows, ctx.data.test_id)
            same = all(close(getattr(fresh, f), getattr(saved, f)) for f in ("oof", "test_by_fold", "fold_auc"))
            same &= saved.metadata["recipe_sha256"] == procedures.recipe_hash(recipe)
            print(f"  atom {rid}: production refit {'equals' if same else 'DIFFERS from'} its bank "
                  f"({time.time() - t0:.0f} s)")
            ok &= same
            refit[rid] = fresh
    outer = {ch["id"]: (np.mean([refit[r].oof for r in ch["run_ids"]], axis=0),
                        np.mean([refit[r].test_by_fold for r in ch["run_ids"]], axis=0)) for ch in chans}
    learned = [p for p in doc["procedures"] if p["kind"] == "learned"]
    settings = [tuple(p["setting"]) for p in learned]
    pop = procedures.Population(ctx.rows, ctx.data.y, ctx.groups, "production")
    cache = procedures.Cache()  # empty: every meta-fold fit is redone
    t0 = time.time()
    got, _ = procedures.evaluate_learned([{"id": c["id"], "atoms": c["atoms"]} for c in chans], settings, outer, pop,
                                         fold_id, list(SPLIT_SEEDS), cache, "replay", runs.packages_for,
                                         procedures.META_PROTOCOL)
    print(f"  meta stage recomputed from an empty cache: {cache.misses} atomic cross-fits, {cache.hits} hits, "
          f"{time.time() - t0:.0f} s")
    recs = {r["procedure_id"]: r for r in runs.logged(ctx, "procedure") if r["procedures_sha256"] == psha}
    for p, st in zip(learned, settings):
        rec = recs.get(p["id"])
        if rec is None:
            print(f"  {p['id']}: no procedure record for this manifest")
            ok = False
            continue
        rawb = (ctx.root / rec["bank_path"]).read_bytes()
        bank = banks.load_bank_v2(rawb, validate=False)
        procedures.validate_procedure_metadata(bank)
        same = artifacts.sha256_bytes(rawb) == rec["bank_sha256"] and close(got[st]["oof"], bank.oof) \
            and close(got[st]["test"], bank.test_by_fold) \
            and artifacts.dumps(got[st]["states"]) == artifacts.dumps(bank.metadata["states"])
        print(f"  {rec['run_id']}: {'equals' if same else 'DIFFERS from'} its logged bank and states "
              f"(mean fold AUC {bank.fold_auc.mean():.6f})")
        ok &= same
    print("replay: every atom and every learned procedure equal" if ok else "replay: DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
