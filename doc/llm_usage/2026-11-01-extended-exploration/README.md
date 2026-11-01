# Extended fast exploration

Owner instruction, 2026-11-01, verbatim:

> Can you test these and take over from the other session, I will close it when it finishes. Note again my goal and the abundant computational resources

Codex session 01a19580-8f99-7492-9bc9-409b9774fb23 takes ownership after the
preceding session's clean handoff at 32e7c5d. The new authorization reopens
development beyond the prior short follow-up. The original experiment deadline
of 20:00 UTC remains the limit. The target is improved AUC, especially on private
data, selected by the existing three grouped split seeds. Existing selected
files and reviews are preserved.

Codex implements and tests the remaining feature-transfer, neural representation,
broader EBM, ranking-objective and learned-neighborhood/retrieval comparisons.
General model APIs are checked against installed code and primary documentation.
No external song data, pretrained weights, fitted test features or pseudo-labels.

Initial CPU queue: 20 EBM configurations with explicit whole-track bootstrap
bags, automatic interactions, no internal validation or early stopping; eight
pairwise CatBoost configurations; twelve LightGBM/Extra Trees comparisons with
group-cross-fitted neighborhood and learned-cluster features. The EBM API smoke
fit took 3.66 seconds for one outer fold at 500 rounds and four bags. Pairwise
probabilities and neighborhood query immutability checks passed before launch.
Full results/configs/OOF/test predictions live under artifacts/extended.

This work is provisional exploration under the owner's fast-mode rules.

Neural extension commit f809c43: fold-fitted year-relative/count/round features,
mixed continuous/categorical representation, quantile scaling, AUC stopping,
pairwise loss and a TabR-inspired project retrieval implementation. The retrieval
network excludes the entire query track group and rejects supervised candidate
features. It is not an exact published TabR reproduction. General algorithm
reference: https://github.com/yandex-research/tabular-dl-tabr. No pretrained weights.
Three focused leakage/category/retrieval checks pass (check_boundaries_final.txt);
six tiny new-route GPU fits pass (neural_smoke.txt). An initial category test
failed because its fixture assumed the wrong category-column order; the fixture
was corrected and rerun. The adapter implementation did not change for that fix.

64 GPU configurations launched at 06:01 UTC with four process-isolated workers.
The first EBM results deteriorated when increasing rounds from 1,000 to 6,000,
so 24 shorter/smaller-learning-rate comparisons were added at 06:04 UTC with
two CPU workers. Full configs are in artifacts/extended/*queue.json. Worker-bound
neural sources are held fixed during the running GPU queue.

Sam reported public AUC 0.86153 for the unchanged fast_followup_trees CSV. The
score is logged; continued model selection uses grouped CV, not public feedback.

At 06:15 UTC Sam asked about learned ensembles. Earlier constrained weights and
logistic stacking used the old pool. Codex adds a comparison using the new
five-seed LightGBM/Extra Trees/CatBoost and three-seed MLP channels, followed by
promising new channels. All five meta folds are rebuilt inside each outer training
fold. Global OOF arrays supply only outer validation predictions. The existing
constrained-log-loss combiners (lambda 0, .01, .1) and L2 logit stackers (C .01, .1, 1)
are reused. This internal cross-fitting is needed to train and validate a stacker;
the dropped outer nested model-selection re-evaluation is not resumed.
Two focused boundary checks pass, including a deliberately leaky global-label
control and rejection of track groups crossing meta folds.

At 06:30 UTC the three-seed AUC-stopped TabM average (CV 0.831625) entered
outer-local meta-training on two GPU workers. Its fixed logit-blend gain was
only +0.000048 with mixed split signs. Five-seed short EBM averages also show
small gains with mixed split signs. At 06:37 UTC the completed short-EBM pool
freed a CPU slot: two XGBoost seed confirmations then the first 32 classical
meta tasks run sequentially in that slot. XGBoost is included because it remains
a distinct component of the incumbent ensemble. The remaining classical meta
tasks launch when the initial CPU queue finishes.

Installed Interpret 0.7.8 source confirms the recorded validation_size=0 warning
is unconditional when outer_bags>1. Explicit supplied bag columns are still used
(_ebm.py lines 898-944), so these fits do use the intended whole-track bootstrap
counts. The other recorded EBM warning concerns plots of missing-value terms,
not invalid predictions. No warnings were suppressed by the scientific runner.

At 06:49 UTC all 16 EBM seed checks were complete. The five-seed 30-interaction,
100-round EBM supplies the strongest confirmed marginal addition, +0.000196
for a 10% logit blend, mixed across split seeds. It enters the seven-channel
learned pool with five-seed LightGBM/Extra Trees/CatBoost, three-seed sklearn MLP,
three-seed XGBoost and three-seed AUC-stopped TabM. Core-five and expanded-seven
pool specs are saved. CPU meta work is split into A=32, B-early=20 (one worker
each, running) and C=38 (four workers after the initial CPU queue completes).
The older full B queue is superseded and must not also be launched.

At 06:42 UTC the other session relayed Sam's request for an ETA. Codex replied
07:45-08:30 UTC for modeling plus submission preparation, most likely near 08:00,
moderate confidence. Late useful candidates needing extra seeds or another meta
channel are the main uncertainty. Report/Colab and the owner-dependent final-pair
review are separate. This is an estimate, not a new owner deadline.

At 07:00 UTC Codex added three convex pairwise-logistic weight fits, with
nonnegative weights summing to one and regularization 0/.01/.1. They reuse the
same outer-local training predictions as the existing six combiners. Each fit
uses at most 50,000 deterministic training-only positive/negative pairs; no new
base-model training is required. Ranking orientation, shrinkage and constant-input
checks pass alongside the leakage controls (three tests in check_stacking.txt).
These optimize a smooth ranking surrogate, not exact empirical AUC.

At 07:03 UTC the completed session relayed Sam's request for light, read-only
supervision across compactions. Primary code/STATE ownership stays here. Codex
sent the current queues, stopping condition, ETA and known limitations. No new
review gate or reruns. While meta workers were active, extended_stacking.py
received changes only to combine and imports (id reuse, ranking comparator and
combiner thread limit); run_task/meta_predictions/predict_channel stayed unchanged.
Meta source_sha256 records the current file including those unrelated edits, not
a byte-exact loaded-module snapshot. Raw neural worker-bound sources stay fixed
since f809c43. No replay claim is made.

Finish scope is fixed: existing 64 neural grid and 14 targeted CPU confirmations;
extra seeds for at most two useful late neural configs; one confirmed neighborhood
and one ranking meta channel; nine regularized combiners plus a few fixed blends;
checked files and recommendations. No further families or broad grids.

At 07:07 UTC meta pool C exited naturally after its completed EBM workers'
300-second joblib idle timeout. It wrote outcomes and the same shell started
final_cpu_confirm_queue.json with four workers. No process signals, duplicate
launches or reruns were needed. The other session's read-only checkpoint is
notes/2026-11-01-supervision-checkpoint.md.

At 07:22 UTC three half-pairwise neural job names containing `0.5` were found
to violate the runner's identifier grammar. They failed before fitting. The
generator now writes `0p5`; all 64 corrected configs pass validation with the
recipes unchanged. Original queue/error records are preserved. Completion uses
neural_queue_corrected.json; only the three renamed cases run in
neural_queue_id_repairs.json after TabM meta completed. This is a launcher fix,
not extra scientific configurations.

At 07:30 UTC all 14 final CPU confirmations were complete. The three-seed shallow
PairLogit ranker with L2=10 gives +0.000456 CV at 10% probability addition, with
all split means positive. Its individual estimator seeds all improve the mean
at the same weight. It joins the final nine-channel pool, with the five-seed
calendar-neighborhood model. Core-five and expanded-seven learned comparisons
are complete; the expanded pairwise lambda=.1 at 25% logit addition gives
+0.000271 across all splits. Four fixed neighborhood/ranker/EBM combinations
are saved before the final learned comparison. No new model families added.

Neighborhood meta fitting completed at 07:30 UTC; the final 15 ranking meta tasks
launched with six CPU workers in stacking_ranking.log.

At 07:36 UTC the retrieval c16/w64/scalar configuration was selected for the
first of at most two allowed late neural seed confirmations. Its fixed 5%
probability addition improves the stronger hybrid by 0.000146 (2/3 splits).
Two other small old-reference neural gains failed against the stronger hybrid.
Seeds 1 and 2 run with one GPU worker in gpu_late_confirm.log.

Uniform logit averages of the three fixed learned pools and a uniform probability
average of the final pool provide the corresponding simple baselines. These
require no fits; saved specifications are uniform_pool_baseline_specs.json.
They distinguish learning weights from merely changing the averaging operation.

A direct browser-tool read of the competition leaderboard failed at 07:42 UTC.
No Kaggle connector or installed Kaggle client was available. The public 0.86153
remains explicitly owner-reported; current rank and private score were not verified.
No credentials were requested and no submission was made.

The first additional retrieval seed confirms the 5% addition to the stronger
hybrid: CV 0.844621, with all three split means higher. The final decision uses
the saved three-seed average at the same fixed 5% weight. The second seed is
still running. This does not add another learned meta channel.

At 07:47 UTC all 64 corrected neural configurations completed. All three ID-only
repairs succeeded; original pre-fit errors remain recorded and are marked resolved.
No second late configuration qualified on the stronger hybrid. Final ranking
meta fits are 12/15 complete; the last retrieval estimator seed is running.

At 07:52 UTC all 96 CPU and 68 neural scientific configurations including
confirmations were complete. The retrieval three-seed average at 5% yields
CV 0.844649, versus hybrid 0.844494. The incremental evidence is mixed: two
of three split means improve; estimator seeds 0/1 improve the hybrid mean,
seed 2 worsens it. The averaged candidate improves all split means over the
submitted follow-up trees. An optional second CSV records this qualification.

At closeout all 135 unique outer-local meta banks and 27 learned combiners
completed. The full table showed that the strongest final-nine standalone
combiner was constrained lambda=.1 (0.843574), correcting an interim log-tail
statement that named pairwise lambda=.1 (0.843497). Two saved-bank additions
of that actual best combiner at 25%/50% yielded 0.844494/0.844391 on the stronger
hybrid. No extra base fits were needed. Across six checked learned-on-hybrid
mixtures, the largest increment was only 0.000006. No third CSV was prepared.

The recommendation is the existing follow-up trees file plus the highest-CV
new neural blend; the simpler hybrid is an alternative because the extra neural
gain is mixed across seeds. Both new files pass automatic file checks and are
logged. final_selection.json and both previously selected CSVs are unchanged.
Sam retains selection/uploads and any changed final pair receives its one quick
Fable leakage/file-correctness review. No review of exploration was introduced.

The supervising session relayed Sam's policy clarification at closeout and saved
its verbatim text under 2026-11-01-policy-clarification. The final recommendation
was corrected to the two new files by grouped CV, rather than retaining the old
file partly because of its public score. The old selected pair stays unchanged
until Sam decides. Provided-data augmentation is allowed but untested beyond
feature engineering/resampling/pair sampling. No new experiment block started.
