# AUC plateau reassessment for Claude

Date: 2026-10-31, America/Los_Angeles. Author: Codex, with three sub-agents in Sam's separate advisory chat. Status: provisional advice for discussion within this project. This is not an R1 or R2 approval, a final model selection, or authorization to change the frozen menu.

Sam requested a fresh read-only reassessment of the apparent public leaderboard ceiling near 0.854, then explicitly requested this handoff document. The assessment trained no models and changed no project files. Creating this note is the subsequent authorized write. Numerical diagnostics below use existing predictions only.

## Assessment and current authority

The current model pool shows real diminishing returns. The seven recent submissions are highly correlated ways of combining similar predictions. Their narrow public score range is evidence that further small weight changes have limited value. It does not establish an upper bound on what the supplied data can support.

Several earlier suggestions now have negative or weak experimental evidence. In particular, native categorical dates, extra year-month encoding, era recalibration, full-data refitting, and adding familiar omitted models should no longer be presented as leading unexplored opportunities.

If Sam wants a further improvement branch, my first choice is one bounded RBF SVM experiment, with a cheap rounded-duration feature check alongside it. Numerical embeddings for TabM are a higher-effort reserve. There is no demonstrated path from 0.85415 to the main leaderboard group around 0.861 to 0.871, and no promised public gain.

The latest owner decision changed while this note was being prepared: Sam authorized **P2.7** at about 06:35 PDT, as recorded in [decisions.md](../decisions.md) and [STATE.md](../STATE.md). Earlier advice in this chat was given while P2.7 was paused. This note does not revoke that authorization, interrupt P2.7, or authorize new fitting. The inspected project HEAD while preparing the note was `5c2545a426ddf17960b894dad060d9122634041b`; P2.7 files were also being developed by the primary session.

Any reopening of the frozen pool needs Sam's decision and the applicable R2 review. Do not use P2.7 outer evaluation results to tune new candidates and then describe those same results as independent validation. A revised search and evaluation sequence must explicitly preserve, or qualify, that separation.

## The public results and a snapshot error

The detailed per-file entries in [submissions/README.md](../submissions/README.md) report:

| Candidate | Mean fold CV AUC | Public AUC |
|---|---:|---:|
| Constrained weights, lambda 0.1 | 0.8410 | **0.85415** |
| Stacker, C 1 | 0.8402 | 0.85411 |
| LightGBM plus 25% Extra Trees | 0.8394 | 0.85398 |
| Equal average of four channels | 0.8411 | 0.85377 |
| Stacker, C 0.01 | 0.8408 | 0.85246 |
| Standalone LightGBM | 0.8364 | 0.85068 |
| LightGBM family average | 0.8365 | 0.85042 |

The six extra uploads improved the displayed best by **0.00038**. Public scores remain checks, not the frozen selection criterion. The production selection remains the equal blend under the reviewed rule.

In the 06:35 snapshot, STATE's six-file row attaches the scores to the wrong filename order. Use the detailed submission log above. This note flags the discrepancy without editing STATE.

Against the lowest score of the main group in the earlier leaderboard export Sam supplied, 0.86127, the gap is **0.00712**. That is a comparison with that export, not a newly checked live leaderboard. Nothing in the export establishes other teams' methods, true generalization performance, or compliance with the course rules.

## The submissions explore very similar rankings

These are new, provisional calculations from the seven existing `2026-10-31_p2.6b_*.csv` files. Their 863 test IDs were checked for identical order. Spearman correlation compares the full test rankings. Pair reversals count unordered test-row pairs whose score differences have opposite signs in the two files, divided by all unordered pairs. Tied pairs are not counted as reversals.

| Comparison with `blend_eq_all` | Spearman correlation | Test-row pairs reversed |
|---|---:|---:|
| `constrained_l0_1` | 0.999973 | 0.1581% |
| `stacker_c0_01` | 0.999202 | 0.9219% |
| `stacker_c1` | 0.999364 | 0.8869% |
| `blend_h_extra_trees_25` | 0.995198 | 2.8321% |
| `family_average` | 0.990030 | 4.1274% |
| `lightgbm_415b649d` | 0.990032 | 4.1352% |

Across all seven files, every pair's rank correlation exceeds 0.9895. Among the five cross-family ensembles it exceeds 0.9946. These are label-free diagnostics on all test rows, not AUC measurements on the public subset. High correlation does not make a gain impossible, but it explains why these uploads are not seven independent modeling directions.

The current channels are a three-recipe LightGBM average, XGBoost, CatBoost with music features, and Extra Trees with interactions. See [the frozen menu](../configs/phase2/menu.json), [pool construction](../analysis/p26_pool.py), and Findings 027 and 028 in [the evidence ledger](../doc/findings.md).

## Existing omitted models add little to the current ensemble

The older brainstorm measured many contributions against B24, a weaker standalone model. To test whether those opportunities remained, this session calculated four fixed diagnostic blends: **90% of the current equal blend plus 10% of one existing final-seed nominee**. No model was fitted, no coefficient grid was searched, and no submission was produced.

| Added nominee | Change in official mean fold AUC | Change on singleton validation rows |
|---|---:|---:|
| kNN contributor | +0.000047 | +0.000169 |
| Date-encoded CatBoost | +0.000086 | +0.000017 |
| LightGBM without date target encoding, with music features | -0.000146 | -0.000228 |
| EBM with music features | +0.000288 | +0.000225 |

EBM's official difference is positive on all three repeats, but small. These observations weaken the case that merely adding the older brainstorm's suggested channels will close the remaining gap. They do not rule out other weights or representations, and are not approved candidates.

For reproduction, all banks are in `artifacts/phase2/bank/`, with `.npz` appended to these run IDs:

- Reference: `p2_blend_eq_all_seeds155-2026-1029_375767dd7ee9`.
- kNN: `p2_p26_knn_c5b5f01f_seeds155-2026-1029_bd834d977bf1`.
- CatBoost: `p2_p26_catboost_cf304bd3_seeds155-2026-1029_c293905d8c54`.
- LightGBM: `p2_p26_lightgbm_142597fc_seeds155-2026-1029_f665b2f192cc`.
- EBM: `p2_p26_ebm_c24e71e3_seeds155-2026-1029_9b297c552c98`.

The row IDs, labels, primary groups, split seeds and fold IDs were checked for exact equality. For each bank pair, average the OOF probabilities with weights 0.9 and 0.1. Compute AUC separately on each of the 15 production folds, average those AUCs, and subtract the reference's corresponding mean. For the singleton diagnostic, restrict each fold to rows whose primary group has size one in the eligible training population. These in-memory calculations were independently repeated while writing this note. They have not entered the experiment log or undergone formal review.

## Revised interpretation of the earlier brainstorm

Sources: the retained [main report](../doc/llm_usage/2026-10-31-auc-brainstorm/scratchpad/me/REPORT.md), [Agent C report](../doc/llm_usage/2026-10-31-auc-brainstorm/scratchpad/agentC_mechanics/REPORT.md), [Agent A report](../doc/llm_usage/2026-10-31-auc-brainstorm/scratchpad/agentA_fresh/REPORT.md), and their code and outputs.

| Mechanism | Evidence and revised priority |
|---|---|
| Full-data refit | The separate matched test favored five inner-fold models by about 0.0019 over one full fit and 0.0012 over a three-seed full fit. This lowers the expected value of a deployment shortcut. It does not settle the final four-family ensemble's deployment comparison. |
| Native categorical dates in CatBoost | Agent C measured +0.0038 against its own CatBoost baseline, reaching 0.8320. Calling that effect null is inaccurate, but it does not show an advantage over the current ensemble. Agent A's much lower result used a different configuration and feature pipeline. |
| Year-month encoding | Adding a separately encoded year-month feature hurt the tested scratch pipeline by about 0.003. This was not hierarchical shrinkage of exact dates toward month and year. The hierarchical proposal is untested, but no longer my leading recommendation. |
| Era-based recalibration | The retained log reports about -0.0019 for recalibrating B24. Its implementation was not retained. That does not prove that conditional multi-model stacking cannot help, but provides no positive evidence for prioritizing it. |
| Same-date neighbors and histogram resolution | Effects were largely redundant with date encoding, small, or unstable. Lower priority. |
| Agent A's apparent 0.855 CV | Much of that score came from a duplicate-count feature computed from the complete train-plus-test pool. Its singleton score was much lower. It is not evidence of a superior clean production candidate to import. |

Agent C's reimplementation missed production B24 by about 0.0014. Its matched within-pipeline comparisons remain informative, but small effects cannot simply be transferred to the production ensemble. The scratch results are exploratory evidence, not reviewed production runs.

Do not adopt the brainstorm's numerical guesses about competitors' true scores, leaderboard selection inflation, or expected pretrained-model gains as measured facts about this competition. Those depend on assumptions or information not available here.

## One bounded improvement branch worth considering

### First choice is an RBF SVM

No RBF SVM experiment was found in the inspected production or brainstorm records. It offers a different way to learn smooth interactions among features. Additive splines and ordinary kNN do not settle its performance.

If Sam authorizes reopening, propose a small fixed grid over regularization and kernel width. Use fold-fitted audio transformations, scaling, missing-value handling and suitable calendar features. Any date target encoding must be rebuilt inside each relevant fitting population. Evaluate both standalone ranking and a small fixed blend with the current leader on the same grouped folds.

Use decision scores for the first AUC test. If retained for probability blending or submission, probability calibration must use group-safe inner folds rather than an unexamined library default. Judge contribution to the current ensemble, not just improvement over a weak SVM baseline. Measure runtime before committing to a wider search. See the [official SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

### Cheap accompanying check is duration rounding

Agent B reported a small positive exploratory effect for `duration_ms % 1000 == 0` on a weaker baseline. The feature uses only the supplied value of the row being predicted. One matched ablation on the current LightGBM recipe would test whether the effect survives, followed by a contribution check against the current ensemble if warranted. Expect an incremental opportunity, not a demonstrated explanation for the full gap.

### Higher-effort reserve is TabM with numerical embeddings

The current implementation explicitly uses `num_embeddings=None` in [tabm.py](../songpop/phase2/tabm.py), and the eight tested configurations reached only 0.8034. That protocol also excluded the stateful date blocks used by the strongest trees. These results establish the weakness of the tested neural setup, not the ceiling of every TabM representation.

Piecewise-linear or periodic numerical embeddings are a distinct possible follow-up. The [authors' implementation](https://github.com/yandex-research/tabm#creating-tabm-with-feature-embeddings) describes these variants. They can be fitted entirely on the provided training data. Preserve preprocessing and stopping isolation. Given the current neural deficit and added implementation cost, I would put this behind the small kernel experiment.

A further, lower-priority possibility is a strongly regularized pairwise ranking objective for the existing combiner, trained only on correctly nested inner OOF predictions. It differs from rank averaging because it learns to improve High-versus-Low ordering. The very similar current channels limit the case for a large gain, so this should not become an open-ended combiner search.

## Validation findings and the role of P2.7

The proposed 80/20 split followed by removal of test tracks overlapping train is consistent with the observed counts. It is not established without the data-generation code or removed rows. A training singleton could still be one observed copy of a globally repeated track. Therefore singleton validation is a useful sensitivity analysis, not proven test-equivalent validation, and the proposed test High rate near 0.30 is an extrapolation.

There is a second snapshot discrepancy. [brainstorm_checks_output.txt](../analysis/brainstorm_checks_output.txt) says the equal blend has the highest raw singleton AUC, but applying the same tie-breaking rule to singleton scores would select the simpler two-atom Extra Trees blend. Its gap to the equal blend shrinks from about 0.0017 on all rows to 0.0007 on singleton rows. STATE's wording that the production selection is kept is too strong. This is diagnostic only and does not change the adopted rule.

P2.7 evaluates the frozen selection procedure and, on the full path, the full-refit deployment comparison and fallback. It cannot discover a feature or model absent from the menu. Its reduced path cannot trigger the deployment switch or fallback. It should be presented as evidence about reliability and deployment, not a promised improvement in leaderboard score. See [plan section 8](../doc/plans/phase2-models-features.md).

## Requested response from Claude

Please read this as advisory input to Sam's decision. Reconcile its claims with current project evidence, especially any work completed after the inspected revision. Distinguish disagreement with a measured fact from disagreement with the proposed priority.

Explain whether one bounded new-model branch is worthwhile, what it would displace, and how it would coexist with the already authorized P2.7 evaluation. If proposing an experiment, specify its comparison, leakage controls, measured cost pilot, and evidence needed to keep it. Do not start new work from this note alone or interrupt authorized work on its authority.

The course restrictions remain binding: supplied data only, no pretrained models, no identifier lookups, no existing code written to model this Spotify dataset, and disclosure of LLM use. No test-label inference, duplicate-score boosts, or pseudo-labeling is proposed here. If these new numerical diagnostics become dependencies of code, submissions or report claims, first integrate them into the project's evidence ledger and applicable review workflow.
