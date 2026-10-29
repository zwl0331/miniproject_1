# Claude's positions on phase 2 (written before reading Codex round E)

Written 2026-10-29, finished at 08:05 PDT (file time), before the round E prompt was written and sent, so the consultant's independent design can be compared with them. Inputs: the committed project at 185dffe (Findings 000 to 008, `doc/plans/modeling_strategy.md`, `doc/plans/phase1-data-baseline.md`, the five review records) and the logged run times (LightGBM and random forest 6 to 8 s for 15 fits with 4 threads; logistic 0.1 s; Apple M5, 10 cores).

## Schedule

1. Phase 1 finished early on Thursday 2026-10-29, a day ahead of the strategy's Day 1 target. Phase 2 starts today and keeps the strategy's task order; the gained day is buffer before the Sunday freeze, not more search.
2. Kaggle closes Monday 2026-11-02, 5:00 PM PST. Daylight saving time ends Sunday 2026-11-01, so Monday's times are PST.
3. Compute is cheap (a 15-fit LightGBM run costs about 8 s). Review turnaround (one Codex round at xhigh takes 6 to 20 minutes, a task 1 to 3 rounds) and author time are the bottleneck. Tasks stay few and large enough to carry one finding each.

## Tasks

| Task | Content | When |
|---|---|---|
| P2.0 | This phase plan; the stale status lines of `doc/README.md`, `doc/project_plan.md` and the strategy | Thursday morning |
| P2.1 | A general recipe runner (a registry of recipes, any subset of split seeds, `--verify`), a generalized submission script, the tuning driver, the CatBoost installation (a reviewed command), and the leftover date questions: the midpoint fill of `days` for partial dates against missing, and a year spline or era encoding for the linear model | Thursday afternoon and evening |
| P2.2 | Model families: a small random-forest search, LightGBM tuning, CatBoost screening and tuning, one smooth model (spline logistic regression, no new package), the first equal blend of two or three families; an improved-incumbent submission if the promotion rule is met | Friday |
| P2.3 | Focused ablations (a small predeclared F3: transforms, a few interactions, era-relative loudness and acousticness; inverse group-size weights; keeping the incomplete row; an exact-date target encoding with inner group cross-fitting), ensembles from the bank (equal averages, a small fixed grid of weights), estimator-seed averaging, the stress tests (chronological split, strict grouping). Freeze the candidate menu and the selection rule by Saturday evening | Saturday |
| P2.4 | The neural challenger, conditional: only if P2.3 is frozen by Saturday afternoon. Otherwise dropped, as the strategy's slip order allows | Saturday |
| P2.5 | The restricted nested evaluation of the frozen menu; interpretation (grouped permutation importance by source column, directions by ALE or dependence plots), as computation and findings, not report text | Sunday |

## Technical positions

1. Iteration counts. Treat the number of boosting rounds as a hyperparameter chosen by outer CV from one fit per fold (staged predictions at a fixed grid of iteration counts), one global count per recipe, rather than early stopping on an inner split of each training part. It needs no inner split, keeps every fit on the full training part, and is one more tuned parameter that the nested evaluation covers.
2. Tuning. Seeded random search over the strategy's regions, 40 trials per family on the split-seed-155 folds, then the best three configurations confirmed on all three split seeds; the selection-free score is the mean over split seeds 2026 and 1029 for the chosen configuration, reported next to the 15-fold mean. No Optuna (one installation fewer); random search is enough at this budget.
3. Deployment. The equal average of the 15 fold models, as in phase 1, for every recipe. A full-data refit is not compared by CV; it stays out unless the nested evaluation finds a way to evaluate it.
4. Ensembles. Probability averages from the bank; a logit average as the one alternative; rank averaging excluded (it uses test ranks). Weights only from a small predeclared grid, chosen on the 15-fold mean and then checked in the nested evaluation.
5. Provenance across phases. Phase 1 banks stay verifiable at their commit. A phase 2 change to `songpop/` or `requirements.txt` changes the source manifest and the configuration hash, so `scripts.run_baselines --verify` at the new HEAD will report the baselines' run ids as stale. The fix: verify phase 1 at its commit in a temporary worktree, and give new runs a manifest of the files they execute.
6. Installation. CatBoost only, pinned, after a `pip install --dry-run` shows that no pinned package changes. EBM (`interpret`), Optuna, SHAP and `torch` are not installed unless a task needs them and its review approves.
7. Submissions in phase 2: at most three (an improved incumbent after P2.2, a blend after P2.3, perhaps one after P2.5), each reviewed within its task, each with its purpose stated before the score is seen.
8. Interpretation runs on Sunday as computation, because phase 4 has about 31 hours for the report and the Colab. Sam's decision defers report and Colab writing, not the analysis; the plan flags this for Sam.

## Doubts

- Whether the exact-date target encoding is the album identity in disguise, and whether grouped CV then measures it the way the test set will (72% of test rows share an exact date with a training row).
- Whether tuning will move AUC at all: the baselines of three different families sit within 0.004 of each other, and the noise of a difference is about 0.003.
- Whether a neural challenger is worth its installation at 3,863 rows.
