# Extended exploration closeout

2026-11-01 07:57 UTC. Development is complete. The results remain provisional under fast mode.
All 164 scientific configurations, 135 meta banks and 27 learned combiners
completed. Three pre-fit identifier errors were repaired; no unresolved run
errors remain. The existing selected pair is preserved.

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
| Hybrid plus 5% three-seed retrieval network | 0.844649 | 0.846180 | 0.844871 | 0.842897 |

The hybrid uses 90% of a logit blend of follow-up trees (90%) and the five-seed
calendar-neighborhood LightGBM (10%), plus 10% of the three-seed shallow
PairLogit ranker with learning rate .03 and L2=10. Nonlinear mixtures are applied
inside each fold before test predictions are averaged. Its improvement over the
submitted file is +0.000680, positive on all three split seeds. EBM and the
seven-channel learned blend add almost nothing to this stronger reference;
the simpler hybrid remains a useful candidate. The retrieval addition has the
highest mean CV in the completed comparison, +0.000835 over the submitted model, with all three split
means higher. Its incremental improvement over the hybrid is less consistent:
two split means improve and one worsens; individual estimator seeds 0 and 1
help, while seed 2 hurts. The three-seed average uses all seeds, without selection.

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
to the stronger hybrid were specified before the final pool finished. The strongest final standalone
combiner was also checked at 25% and 50% probability addition after the full
results table was available; neither added a useful gain.

| Pool | Uniform probability | Uniform logit | Best learned standalone |
|---|---:|---:|---:|
| Core five | 0.843331 | 0.843172 | 0.843485 |
| Expanded seven | 0.842983 | 0.843036 | 0.843225 |
| Final nine | 0.843563 | 0.843517 | 0.843574 |

The best learned addition to the old submitted reference is a 50% probability
blend with the final constrained combiner (penalty .1): CV 0.844333, a gain of
0.000518 with all split means positive. On the stronger hybrid, the largest
learned increment among the six checked mixtures is only 0.000006, with mixed
split signs. It does not justify another submission candidate.

## Neural adapter and method coverage

The adapter supports TabM, MLP, ResNet, FT-Transformer and a project adaptation
inspired by TabR. Numerical representations include scalar, piecewise-linear
and periodic embeddings, with mixed categorical inputs where supported.
Training includes optional quantile preprocessing, year-relative/count features,
AUC stopping and pairwise loss. No pretrained weights or external data are used.
Usage and focused CPU/GPU checks are in `notes/2026-11-01-neural-adapter.md`.

The completed bounded run contains 96 CPU configurations including seed
confirmations, 64 neural configurations, two additional TabM estimator seeds
and two additional retrieval estimator seeds. The 64-case neural grid covers
17 TabM, 17 MLP, 11 ResNet, 11 FT-Transformer and eight retrieval variants.
None of its 960 outer fit records hit the maximum epoch cap. Three neural
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

`submissions/2026-11-01_fast_extended_neural_blend.csv` is the optional retrieval
addition. It passes the same 863-row, ordered-ID and probability checks;
correlation with follow-up trees is 0.998613022.
SHA256: `177874ed7e61012ca2493189de05445f90899f1a74b7e4356d2df7976176b2b8`.
Its incremental seed uncertainty is recorded in the CSV's sidecar and log.

## Evidence

Full configurations, fit details, OOF/test banks and runtimes are under
`artifacts/extended/`, `artifacts/neural/x3_*` and `artifacts/fast/x3_*`.
`artifacts/extended/contributions.csv` compares fixed additions/replacements;
`summary.csv` covers the original probability-addition comparisons.
The final learned specifications are `stacking_pool_final.json` and
`learned_blend_specs.json`. LLM use is recorded in
`doc/llm_usage/2026-11-01-extended-exploration/README.md`.

## Recommendation and remaining owner action

Recommend final selections **extended neural blend + extended rank/neighborhood
blend**, by grouped CV. The small EBM and learned increments over the simpler
hybrid are treated as tied, with simplicity preferred. Both new files improve
all split means over the submitted reference. The neural increment is eligible
but small and mixed; it is not a private-score guarantee. The submitted follow-up
file remains preserved, with public 0.86153 recorded as a check only. Subsequent public scores are recorded below.

An interim recommendation kept the incumbent as a hedge based partly on its
public result. The final recommendation above corrects that reasoning to follow
Sam's explicit CV-only selection rule. His actual selections are unchanged.

`artifacts/extended/recommendation.json` records this recommendation separately
from the unchanged `artifacts/fast/final_selection.json`. Sam selects and uploads.
A changed final pair receives the one quick Fable review of leakage and file
correctness, one round, after his selection. The existing pair remains approved.
Report and Colab work is separate. No training or review job remains running.

## Policy clarification and untested coverage

Sam relayed the new Harry Chen policy through the supervising session. Verbatim
text: `doc/llm_usage/2026-11-01-policy-clarification/user_prompt.md`. Its ban on
external/foundation-model-generated data and external validation is consistent
with this pipeline. Augmentation entirely from provided training data is allowed.
We used feature engineering, resampling/bagging and training-only pair sampling;
we did not test synthetic-row augmentation such as mixup/SMOTE, input noise,
denoising or distillation. Broad family coverage does not exhaust every allowed
method. This clarification does not authorize another run block or lift Sam's
explicit no-test-feature-fitting rule. See the accompanying assessment.

## Subsequent public checks reported by Sam

Sam reports both uploads: extended neural blend **0.86047**, and extended
rank/neighborhood blend **0.86136**. These regress by 0.00106 and 0.00017
respectively against follow-up trees **0.86153**. The neural file is 0.00089
below the simpler file on this public subset. The small reused-fold CV gains
did not transfer to that subset; private scores remain unknown. This is logged
as a check, without changing the CV selection rule or final_selection.json.
The separately authorized augmentation queue was specified before these scores
arrived and continues unchanged within its original cap.
