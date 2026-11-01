# Extended exploration checkpoint

2026-11-01 07:32 UTC. Provisional. Final ranking meta fits and the remaining
neural configurations are running. The existing selected pair is preserved.

Sam reported public AUC **0.86153** for `2026-11-01_fast_followup_trees.csv`.
The new models are selected using the saved track-grouped folds, with split seeds
155, 2026 and 1029. These folds have been reused throughout exploration.
The small CV differences below are selection evidence, not significance tests
or guarantees about the private board.

| Candidate | Mean fold AUC | Split 155 | Split 2026 | Split 1029 |
|---|---:|---:|---:|---:|
| Submitted follow-up trees | 0.843814 | 0.844708 | 0.844520 | 0.842215 |
| Add 10% confirmed PairLogit | 0.844270 | 0.845497 | 0.844881 | 0.842431 |
| Probability neighborhood/ranking blend | 0.844477 | 0.845792 | 0.844990 | 0.842650 |
| Hybrid neighborhood/ranking blend | 0.844494 | 0.845795 | 0.845047 | 0.842640 |
| Hybrid plus 5% EBM in logit space | 0.844508 | 0.845884 | 0.845085 | 0.842554 |
| Hybrid plus 25% expanded learned pairwise blend | 0.844500 | 0.845942 | 0.845122 | 0.842437 |

The hybrid uses 90% of a logit blend of follow-up trees (90%) and the five-seed
calendar-neighborhood LightGBM (10%), plus 10% of the three-seed shallow
PairLogit ranker with learning rate .03 and L2=10. Nonlinear mixtures are applied
inside each fold before test predictions are averaged. Its improvement over the
submitted file is +0.000680, positive on all three split seeds. EBM and the
seven-channel learned blend add almost nothing to this stronger reference;
the simpler hybrid is currently preferred.

## Learned ensembles

Every meta-training column is rebuilt inside its outer training population using
five track-grouped meta folds. The global OOF banks supply only outer validation
predictions. The combiner never receives outer validation labels. Preprocessing,
encoders and neural stopping stay inside each base learner's training population.

The tested combiners are constrained log-loss weights (penalty 0, .01, .1),
L2 logistic stacking on standardized logits (C .01, .1, 1), and nonnegative
simplex pairwise-logistic weights (penalty 0, .01, .1). The last objective is a
smooth ranking surrogate using at most 50,000 deterministic training-only pairs.
It does not directly maximize empirical AUC. The three focused boundary checks
pass, including a deliberately leaky global-label control.

The core pool contains five-seed LightGBM, Extra Trees and CatBoost, three-seed
sklearn MLP and XGBoost. The expanded pool adds five-seed EBM and three-seed
AUC-stopped TabM. The final pool adds the confirmed neighborhood and ranking
channels. Each pool has nine combiner settings. Four fixed 25% learned additions
to the stronger hybrid were specified before the final pool finished.

## Neural adapter and method coverage

The adapter supports TabM, MLP, ResNet, FT-Transformer and a project adaptation
inspired by TabR. Numerical representations include scalar, piecewise-linear
and periodic embeddings, with mixed categorical inputs where supported.
Training includes optional quantile preprocessing, year-relative/count features,
AUC stopping and pairwise loss. No pretrained weights or external data are used.
Usage and focused CPU/GPU checks are in `notes/2026-11-01-neural-adapter.md`.

The bounded run contains 96 CPU configurations including seed confirmations,
64 neural configurations and two additional TabM estimator seeds. Three neural
IDs containing a decimal point failed before fitting; only their IDs were
changed, and the original records are retained. The corrected completion roster
is `artifacts/extended/neural_queue_corrected.json`. These are three repairs
inside the original 64 cases, not extra scientific configurations.

## Prepared file

`submissions/2026-11-01_fast_extended_rank_neighbors.csv` contains 863 rows,
with unique IDs in the exact `test.csv` order and finite probabilities in [0, 1].
Its correlation with the submitted follow-up file is 0.998846761.
SHA256: `e94065dd0834358511bf433e38115f5ae031a93743a58f0ce63bf29f839de975`.
The automatic checks and CV are logged in `submissions/README.md` and the
matching JSON sidecar. Sam selects and uploads. No upload or final selection
has been made by the agent.

## Evidence

Full configurations, fit details, OOF/test banks and runtimes are under
`artifacts/extended/`, `artifacts/neural/x3_*` and `artifacts/fast/x3_*`.
`artifacts/extended/contributions.csv` compares fixed additions/replacements;
`summary.csv` covers the original probability-addition comparisons.
The final learned specifications are `stacking_pool_final.json` and
`learned_blend_specs.json`. LLM use is recorded in
`doc/llm_usage/2026-11-01-extended-exploration/README.md`.
