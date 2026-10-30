"""Deterministic forest prediction (Finding 013; doc/plans/phase2-models-features.md section 16.4): the checks that
P2.8's code change fixes the forests' prediction order and touches nothing else.

(1) Repetition: one fitted random forest (P2.2's random forest anchor recipe) and one fitted Extra Trees model
(P2.2's leaf-1 recipe with the most ties), each found as the one logged P2.2 run of its recipe, predict the test
rows 20 times through the engine's FittedRecipe; the 20 outputs must be identical, and n_jobs is restored to THREADS
after prediction.
(2) Non-forest families: each of the five P2.2 family anchors other than the forests, refitted in memory under the
current code, must equal its P2.2 bank in all ten arrays.
(3) Forest compatibility: the random forest and Extra Trees anchors must equal their P2.2 banks in every array
except the predictions and fold AUCs, the predictions within 1e-12; every signed fold AUC difference is printed by
split seed and fold.
(4) Forest replay: each forest anchor refitted twice under the current code must be identical in all ten arrays.
Fits models; writes nothing but its standard output; exits 1 if a check fails. From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -I analysis/p28_forest_check.py . > analysis/p28_forest_check_output.txt
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))

from songpop import artifacts  # noqa: E402
from songpop.evaluate import PredictionBank  # noqa: E402
from songpop.phase2 import banks, protocol, runs  # noqa: E402
from songpop.phase2.encoders import FitContext  # noqa: E402
from songpop.phase2.engine import fit_recipe  # noqa: E402
from songpop.phase2.models import FORESTS, THREADS  # noqa: E402
from songpop.phase2.recipes import strict_loads  # noqa: E402

TASK = "P2.2"
PREDICTIONS = ("oof", "test_by_fold", "test_mean")
REPEAT_RECIPES = ("p22_random_forest_anchor", "p22_extra_trees_b1_0_022")


def differing(a, b):
    return [f for f in PredictionBank.ARRAYS if not np.array_equal(getattr(a, f), getattr(b, f))]


def main():
    ctx = runs.Context.load(ROOT)
    manifest = banks.source_manifest(ctx.root)
    inv = {"argv": ["p28_forest_check"]}
    records = artifacts.read_experiments(ROOT / "artifacts/phase2/experiments.jsonl")
    log = protocol.Log(records, TASK, lambda rel: strict_loads((ROOT / rel).read_text()))
    ok = True

    tr = np.flatnonzero(ctx.fold_id([155])[0] != 0)
    for name in REPEAT_RECIPES:
        rids = [r for r in log.runs if log.runs[r].get("task_id") == TASK and log.manifest_recipe(r)["name"] == name]
        if len(rids) != 1:
            raise ValueError(f"recipe {name} names {len(rids)} logged P2.2 runs, expected 1")
        fitted = fit_recipe(log.manifest_recipe(rids[0]), ctx.rows.iloc[tr], ctx.data.y[tr], ctx.groups[tr],
                            FitContext(155, 0))
        outs = {fitted.predict_high(ctx.test_rows).tobytes() for _ in range(20)}
        restored = all(e.n_jobs == THREADS for e in fitted.estimators)
        print(f"repetition, {name}: {len(outs)} distinct outputs in 20 predictions; n_jobs restored to {THREADS}: "
              f"{restored}")
        ok &= len(outs) == 1 and restored

    anchors = sorted((b["family"], log.read_manifest(b["file"])["run_ids"][0]) for b in log.batches if b["wave"] == 1)
    for family, rid in anchors:
        saved = runs.load_logged(ctx, rid)[0]
        md = saved.metadata
        new = runs.fitted_run(ctx, md["recipe"], md["split_seeds"], "P2.8", manifest, inv)
        differ = differing(new, saved)
        pred = max(float(np.max(np.abs(getattr(new, f) - getattr(saved, f)))) for f in PREDICTIONS)
        auc = np.abs(new.fold_auc - saved.fold_auc)
        if family in FORESTS:
            good = set(differ) <= set(PREDICTIONS + ("fold_auc",)) and pred <= banks.TOL
            again = runs.fitted_run(ctx, md["recipe"], md["split_seeds"], "P2.8", manifest, inv)
            replay = differing(again, new)
            good &= not replay
            print(f"{family} anchor {rid}: arrays differing from P2.2 {differ or 'none'}; largest prediction "
                  f"difference {pred:.1e}; fold AUC differences: {int((auc > 0).sum())} of {auc.size} folds, largest "
                  f"{auc.max():.2e}, mean fold AUC {new.fold_auc.mean() - saved.fold_auc.mean():+.2e}; replay under "
                  f"the current code: {'identical' if not replay else 'DIFFERS in ' + str(replay)}: "
                  f"{'OK' if good else 'FAILED'}")
            for r, seed in enumerate(md["split_seeds"]):
                diffs = ", ".join(f"{new.fold_auc[r, k] - saved.fold_auc[r, k]:+.2e}" for k in range(auc.shape[1]))
                print(f"    split seed {seed}, folds 0 to {auc.shape[1] - 1}, current minus P2.2: {diffs}")
        else:
            good = not differ
            print(f"{family} anchor {rid}: arrays differing from P2.2 {differ or 'none'}: {'OK' if good else 'FAILED'}")
        ok &= good
    print("forest check:", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
