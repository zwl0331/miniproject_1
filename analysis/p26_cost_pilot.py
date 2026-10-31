"""P2.6a cost pilot (plan sections 7.2 and 8), score-free: times the real recursion on the anchor and one cheap
second channel, and projects the production evaluation of learned procedures and the nested evaluation from the
measured unit costs. Prints times only; writes nothing. From the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B analysis/p26_cost_pilot.py

Measured: (a) the anchor's 15-fit production run; (b) one outer fold of the learned-procedure stage (the channels'
meta predictions on the outer training part, five fits each, shared by all six settings) and the six combiner fits;
(c) one outer fold of the nested evaluation for the anchor as an atomic candidate (15 inner fits on the outer
training part predicting the outer validation rows, plus its full refit), and the third level for a learned
candidate on one inner training part. Projections multiply these by the number of folds; they assume the per-fit
cost of other recipes scales with their own measured 15-fit time, as the plan's formulas do."""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

from songpop.phase2 import nested, procedures as P, runs  # noqa: E402
from songpop.phase2.recipes import load_recipe  # noqa: E402

ANCHOR = Path("configs/phase2/recipes/lightgbm_F2_anchor.json")


def main():
    ctx = runs.Context.load(Path("."))
    anchor = load_recipe(ANCHOR)
    logistic = dict(anchor, name="pilot_logistic", family="logistic", params={"C": 0.1}, base="F1",
                    preprocessing={"name": "log_standard"})
    pk, man = runs.packages_for, "pilot"
    pop = P.Population(ctx.rows, ctx.data.y, ctx.groups, "production")
    fold = ctx.fold_id([155])
    t = time.perf_counter()
    P.atomic_cv(anchor, pop, fold, ["pilot-prod"], ctx.test_rows, P.Cache(), man, pk(anchor))
    c_anchor = (time.perf_counter() - t) * 3  # one repeat measured; a 15-fit run has three
    chans = [{"id": "a", "atoms": [anchor]}, {"id": "b", "atoms": [logistic]}]
    D = pop.subset(np.flatnonzero(fold[0] != 0), "pilot-D")
    sets = [("constrained", l) for l in P.LAMBDAS] + [("stacker", c) for c in P.CS]
    t = time.perf_counter()
    meta = P.population_folds(D.y, D.groups, D.row_ids, 5, 7)
    Za = P.atomic_cv(anchor, D, meta[None, :], ["pilot-meta"], D.rows.iloc[:1], P.Cache(), man, pk(anchor))[0][0]
    t_meta_anchor = time.perf_counter() - t
    t = time.perf_counter()
    Zb = P.atomic_cv(logistic, D, meta[None, :], ["pilot-meta"], D.rows.iloc[:1], P.Cache(), man, pk(logistic))[0][0]
    t_meta_logistic = time.perf_counter() - t
    t = time.perf_counter()
    for s in sets:
        P.fit_combiner(s, np.column_stack([Za, Zb]), D.y)
    t_comb = time.perf_counter() - t
    of = nested.outer_folds(pop)
    O, V = pop.subset(np.flatnonzero(of != 0), "pilot-O"), pop.subset(np.flatnonzero(of == 0), "pilot-V")
    F = nested.inner_folds(O)
    t = time.perf_counter()
    P.atomic_cv(anchor, O, F, [f"pilot-n:{s}" for s in nested.INNER_SEEDS], V.rows, P.Cache(), man, pk(anchor))
    t_inner = time.perf_counter() - t
    t = time.perf_counter()
    nested.refit_predict({"id": "a", "kind": "atomic", "recipe": anchor}, O, V.rows, P.Cache(), man, pk, "pilot")
    t_refit = time.perf_counter() - t
    I = O.subset(np.flatnonzero(F[0] != 0), "pilot-I")
    t = time.perf_counter()
    P.learn_on(I, chans[:1], sets, 11, "pilot-third", P.Cache(), man, pk)
    t_third = time.perf_counter() - t
    print(f"anchor 15-fit run (three repeats, from one): {c_anchor:.1f} s")
    print(f"learned stage, one outer fold: anchor channel {t_meta_anchor:.1f} s, logistic channel "
          f"{t_meta_logistic:.1f} s, six combiners {t_comb:.2f} s")
    print(f"  ratio of one outer fold's meta stage to the anchor's 15-fit run: {t_meta_anchor / c_anchor:.3f} "
          f"(15 outer folds: {15 * t_meta_anchor / c_anchor:.2f} x C_L; the plan assumes 6 C_L with its overhead)")
    print(f"nested, one outer fold, atomic anchor: 15 inner fits {t_inner:.1f} s, full refit {t_refit:.1f} s; "
          f"three outer folds {3 * (t_inner + t_refit):.1f} s = {3 * (t_inner + t_refit) / c_anchor:.2f} x C_all")
    print(f"nested third level, one inner training part, anchor channel: {t_third:.1f} s; per learned channel "
          f"atom {3 * 15 * t_third:.1f} s = {3 * 15 * t_third / c_anchor:.2f} x C_L")


if __name__ == "__main__":
    main()
