# Findings

This file is the evidence ledger for the Miniproject 1 model. New factual conclusions about the data and the models land here before code, Kaggle submissions, the Colab notebook, or the report may depend on them. The convention is ported from the Positron Conversion DIO project (its Finding 000), which borrowed it from `DIO_Shape_Systematic`.

## Review Status Labels

- `draft`: written for discussion, not yet accepted.
- `reviewed`: named by a review record under `../reviews/` that ended in `APPROVED` at a commit with no blocking finding open (`../AGENTS.md`; before 2026-10-30, a converged record under `review_policy.md` section 4.1); the finding cites the record.

`draft` findings are provisional: work may build on them provisionally (`../AGENTS.md`).
- `superseded`: replaced by a later finding.
- `rejected`: kept for audit history, but not used.

## Entry Template

```text
## Finding NNN: Short Title

Question:

Current conclusion:

Evidence:

Uncertainty:

Review status:

Downstream use:
```

## Finding 000: Review-First Convention

Question:

What rule keeps the models, the submissions, and the report auditable under a six-day deadline?

Current conclusion:

Every number that leaves this project, into a Kaggle submission, the Colab notebook, the report, or Piazza, must trace to findings in this ledger and to the reviewed code that produced it, with its inputs recorded: the data files and their hashes, the code revision, the random seeds, and the cross-validation split. If a code change would imply a new conclusion about the data or the models, the finding is written or updated first.

Evidence:

- Ported from the Positron Conversion DIO project's Finding 000 with the review framework, on Sam's instruction of 2026-10-28 (task P0.1).
- The failure surface here: a validation score inflated by leakage (repeated tracks across folds, preprocessing fitted on validation data) or by tuning to the public leaderboard looks like progress until the private leaderboard. Recording each number with its split and inputs makes such errors visible to the reviewer.

Uncertainty:

None. This is a working rule, not a measurement.

Review status:

`reviewed`, by the P0.1 review (`../reviews/2026-10-28-p0.1-bootstrap.md`). This ledger is committed only after that review converges.

Downstream use:

Every later finding.

## Finding 001: Dataset Contract, Target, and Missingness

Question:

What is a training observation, which label is the positive class, and which rows lack usable predictors?

Current conclusion:

- `data/train.csv` has 3,864 rows and 19 columns; `data/test.csv` has 863 rows with `ID` in place of the target; `data/sample_submission.csv` has `ID` and `Popularity_Type`, its IDs equal the test IDs in order (2 to 967, unique), every value 0.5.
- The target `Popularity_Type` holds exactly "High" (1,359) and "Low" (2,505), none missing; the PDF's 1 and 0 do not appear. High is mapped to 1 explicitly.
- One training row, position 425, has every audio feature and identifier missing; only its date (2024-09-08) and its label (Low) are present. No other row has a missing or unparsable audio value, and the test file has none. The 3,863 complete rows hold 1,359 High and 2,504 Low.
- Categorical audio columns take few values: `time_signature` 1, 3, 4, 5; `mode` 0, 1; `key` 0 to 11.

Evidence:

`../analysis/audit.py`, output `../analysis/audit_output.txt`, blocks [P] (input hashes: train d14b372d…, test 70f1502c…, sample 35afd1ab…) and [A1].

Uncertainty:

None for the counts. Why the one row is empty is unknown.

Review status:

`reviewed`, by the P1.1 review (`../reviews/2026-10-29-p1.1-data-audit.md`). This ledger revision is committed only after that review converges.

Downstream use:

The loader, the explicit target map, the eligibility mask (the incomplete row is excluded from training by default, `plans/modeling_strategy.md` section 5.1), the submission schema.

## Finding 002: Track Identity and Repeated Observations

Question:

Which training rows must stay together in cross-validation, and how do groups by `uri` differ from groups by identical features?

Current conclusion:

- `track_href` and `analysis_url` end in the `uri` token on every row that has one; `type` is constant ("audio_features"). These columns add no information beyond `uri`.
- Train has 3,642 distinct `uri` values; 203 of them repeat (185 twice, 18 three times; 424 rows). Within every repeated `uri`, the audio features and the date are identical. 32 repeated `uri` groups carry conflicting labels (66 rows). Test has 852 distinct `uri`, 11 of them repeated twice; no `uri` occurs in both files.
- 11.0% of training rows (424 of 3,864) and 2.5% of test rows (22 of 863) belong to a repeated `uri` group.
- Different `uri` values can share identical audio features and date. Grouping rows by the same `uri` or the same audio-plus-date signature (connected components) gives 3,638 groups, including the incomplete row as a singleton: 5 groups join more than one `uri`, one group has 4 rows, and 35 groups (72 rows) have mixed labels. Also joining identical audio-only signatures (the strict grouping, which coarsens the primary one) gives 3,630 groups, 13 joining more than one `uri`, 38 with mixed labels.
- No audio-plus-date signature occurs in both files; two audio-only signatures do.

Evidence:

`../analysis/audit.py`, output `../analysis/audit_output.txt`, block [A2].

Uncertainty:

The groups capture only observable repetition. Songs by the same artist or from the same album are not identifiable, so independence between different groups is an assumption. Why repeated tracks carry different labels is unknown.

Review status:

`reviewed`, by the P1.1 review (`../reviews/2026-10-29-p1.1-data-audit.md`). This ledger revision is committed only after that review converges.

Downstream use:

The primary grouping and the strict sensitivity grouping of `plans/modeling_strategy.md` section 4, step 1; the duplicate-consistency check of submissions; the inverse-group-size ablation; the leakage explanation in the report.

## Finding 003: Release-Date Precision and Validity

Question:

Which date components are actually observed, and does every date parse without inventing information?

Current conclusion:

- Every date in both files is present and valid in one of three formats: `YYYY-MM-DD` (train 3,752, test 840), `YYYY` (train 111, test 23), `YYYY-MM` (train one row, 1979-06).
- Years run from 1954 to 2024 in train and 1957 to 2024 in test; full dates from 1954-01-01 (5,844 days before 1970-01-01) to 2024-11-22.
- Under the F2 rules of `plans/modeling_strategy.md` section 5.2, month is missing for the 111 training and 23 test rows with a year only; day, day of year, weekday and days since 1970 are missing for those plus the 1979-06 row (112 training rows, 23 test rows).

Evidence:

`../analysis/audit.py`, output `../analysis/audit_output.txt`, block [A3]; each date is matched against an explicit pattern and full dates are parsed with an explicit format.

Uncertainty:

None for the formats. Whether a year-only date means an unknown day or a deliberate annual release is unknown.

Review status:

`reviewed`, by the P1.1 review (`../reviews/2026-10-29-p1.1-data-audit.md`). This ledger revision is committed only after that review converges.

Downstream use:

The F1 and F2 definitions and their edge-case checks in P1.2; derived F2 features are missing even though the raw test file has no missing value.

## Finding 004: Marginal Associations with High Popularity

Question:

Which supplied variables and release eras show a marginal association with High?

Current conclusion:

- High rate by release era (training rows): 1.000 before 1990 (134 rows, 115 distinct `uri`), 0.502 in the 1990s (223), 0.519 in the 2000s (316), 0.422 in the 2010s (810), 0.255 from 2020 on (2,381).
- Univariate AUC with High as the positive class and increasing raw value as the score: loudness 0.681, energy 0.628, duration 0.607, valence 0.571; instrumentalness 0.325, acousticness 0.342, release year 0.359 (older songs rank higher); danceability, tempo, key, liveness, mode, time signature and speechiness between 0.496 and 0.545.

Evidence:

`../analysis/audit.py`, output `../analysis/audit_output.txt`, block [A4].

Uncertainty:

Marginal, univariate and descriptive: no conditional effect, interaction or causal claim. Correlated features (loudness, energy, acousticness) share signal. The cause of the era pattern is unknown; selection of enduring popular songs is one unverified hypothesis.

Review status:

`reviewed`, by the P1.1 review (`../reviews/2026-10-29-p1.1-data-audit.md`). This ledger revision is committed only after that review converges.

Downstream use:

Motivation for the date features and for the baselines; hypotheses for the report's feature-direction analysis. Not used for feature selection.

## Finding 005: Descriptive Comparison of Train and Test; Row Order

Question:

What observable differences exist between the two files, without fitting anything on test rows, and does row order carry information?

Current conclusion:

- Release-year percentiles (10, 25, 50, 75, 90) are 2001, 2015, 2022, 2024, 2024 in train and 2002, 2016, 2022, 2023.5, 2024 in test.
- Two-sample Kolmogorov-Smirnov distances between train and test are at most 0.043 (acousticness) over the 13 audio features and the release year.
- 623 of the 863 test rows (72.2%) have an exact date string that occurs in train.
- The training row position has AUC 0.506 for High; the test ID has correlation -0.027 with the release year.

Evidence:

`../analysis/audit.py`, output `../analysis/audit_output.txt`, block [A5]. The distances are computed with NumPy as descriptive statistics; no model or transformation is fitted on test rows.

Uncertainty:

Marginal comparisons only: they do not establish matching joint distributions or labeling, and repeated rows make significance tests inappropriate, so none is reported. The order checks test monotone association only.

Review status:

`reviewed`, by the P1.1 review (`../reviews/2026-10-29-p1.1-data-audit.md`). This ledger revision is committed only after that review converges.

Downstream use:

The working assumption that grouped random cross-validation is a reasonable proxy for the private leaderboard (`plans/modeling_strategy.md` section 4); row position and test ID are not used as features.

## Finding 006: Legacy Scoping Reference

Question:

What did the scoping run of P0.2 establish, and what must the P1.2 harness reproduce?

Current conclusion:

Under the legacy protocol (the 3,863 complete rows, `StratifiedGroupKFold` with 5 folds grouped by `uri`, split seeds 0, 1 and 2, the legacy date fields year, month, day, date precision and days since 1970 with a midyear fill for partial dates):

- Mean fold AUC with audio and legacy date fields: LightGBM (600 trees, learning rate 0.02, 15 leaves) 0.8268, random forest 0.8257, LightGBM default 0.8217, HistGB 0.8201, Extra Trees 0.8197, XGBoost default 0.8080, logistic regression on log features 0.7812. Audio only: 0.7300 to 0.7557.
- The legacy date fields add +0.05 (logistic) to +0.08 (trees) over audio only. In LightGBM, the fields beyond the year add +0.0125 (audio plus year 0.8143).
- Ungrouped folds scored +0.0158 higher than grouped folds for both random forest and LightGBM.

Evidence:

`../analysis/scoping.py` and `../analysis/scoping_output.txt` (blocks [2], [3] and [8]), reviewed in P0.2 (`../reviews/2026-10-29-p0.2-modeling-strategy.md`) and committed in f29267b. The protocol is unchanged, so the evidence is reused, not rerun.

Uncertainty:

The naive standard errors printed there treat overlapping folds as independent and understate the uncertainty. The legacy protocol differs from the production one (groups, split seeds, date fields), so production results need not match. The ungrouped-versus-grouped difference is observed protocol sensitivity, not a universal correction. The standalone exact-date encoding (scoping [6]) says nothing about its value next to the other features.

Review status:

`reviewed`, by the P1.1 review (`../reviews/2026-10-29-p1.1-data-audit.md`). This ledger revision is committed only after that review converges.

Downstream use:

The reproduction target of the P1.2 harness (the table above, to four decimals); the reference point for the production baselines.

## Finding 007: Harness Reproduction and the Production Baselines

Question:

Does the P1.2 harness reproduce the reviewed scoping results, and what do the three baselines score under the production protocol?

Current conclusion:

- Run through the harness's folds, prediction and metric code, the legacy protocol reproduces all 15 values of Finding 006 to four decimals.
- Production protocol: the 3,863 eligible rows, the primary groups of Finding 002, `StratifiedGroupKFold` with 5 folds, split seeds 155, 2026 and 1029 (`../artifacts/splits/production_v1.csv`, sha256 b2cba98d…); estimator seed 0; the recipes of `plans/phase1-data-baseline.md` section 4.5. Mean fold AUC over the 15 folds:

| Model | F0 (audio) | F1 (+ year) | F2 (+ date parts) |
|---|---|---|---|
| LightGBM | 0.7519 | 0.8149 | 0.8315 |
| Random forest | 0.7555 | 0.8141 | 0.8279 |
| Logistic | 0.7300 | 0.7810 | 0.7819 |

- Every validation fold holds 772 or 773 rows with a High rate of 0.351 to 0.352; no class-rate or era-share warning in either grouping.
- Diagnostics, not for selection: weighting each row by one over its group's size changes the mean fold AUC by at most 0.0022 (largest: logistic F1, 0.00212) (LightGBM F2 0.8297, random forest F2 0.8275). Within a single era the pooled out-of-fold AUC is lower than overall (LightGBM F2: 0.80 to 0.81 from 2020 on, 0.77 in the 2010s, 0.79 to 0.80 in the 2000s, 0.78 to 0.79 in the 1990s; undefined before 1990, where every row is High), because the era itself carries much of the signal.
- The highest F2 mean fold AUC is LightGBM's, 0.8314739289 (run `production_lightgbm_F2_9c37e77d549e`).

Evidence:

`../scripts/run_baselines.py`; outputs `../artifacts/run_baselines_legacy.txt` and `../artifacts/run_baselines_production.txt`; the nine banks in `../artifacts/bank/` with their sha256 in `../artifacts/experiments.jsonl` (for example LightGBM F2 37027940…); `--verify` recomputes every bank, its log record and the legacy table without writing. `../scripts/check_foundation.py` tests the harness itself (its output: `../artifacts/check_foundation.txt`).

Uncertainty:

Cross-validated estimates of untuned baselines, not leaderboard scores. Random forest predictions recompute within 3.3e-16 (thread summation order); LightGBM and logistic recompute exactly. Era AUCs are pooled out-of-fold diagnostics.

Review status:

`reviewed`, by the P1.2 review (`../reviews/2026-10-29-p1.2-harness.md`). This ledger revision is committed only after that review converges.

Downstream use:

The first submission (P1.3, LightGBM F2 by the rule of `plans/phase1-data-baseline.md` section 5); the incumbents of phase 2.

## Finding 008: Date Features and the F2 Model Comparison

Question:

What do the date features add for each model, and do random forest and LightGBM differ on F2?

Current conclusion:

Paired cluster bootstrap of the difference in mean fold AUC (`plans/modeling_strategy.md` section 4, step 8; 2,000 replicates, seed 155001, all valid):

| Comparison | Difference | 95% interval | Draws above 0 |
|---|---|---|---|
| LightGBM F1 minus F0 | +0.0630 | +0.0535 to +0.0726 | 1.000 |
| Random forest F1 minus F0 | +0.0586 | +0.0491 to +0.0677 | 1.000 |
| Logistic F1 minus F0 | +0.0511 | +0.0394 to +0.0636 | 1.000 |
| LightGBM F2 minus F1 | +0.0166 | +0.0113 to +0.0218 | 1.000 |
| Random forest F2 minus F1 | +0.0138 | +0.0092 to +0.0186 | 1.000 |
| Logistic F2 minus F1 | +0.0008 | -0.0015 to +0.0031 | 0.755 |
| Random forest F2 minus LightGBM F2 | -0.0036 | -0.0091 to +0.0023 | 0.104 |

- Under the promotion rule (strategy section 4, step 9), the year and the further date parts both earn their place for the two tree models (gains of at least 0.003 overall, positive on each of the three repeats). For logistic regression the date parts beyond the year do not (+0.0008 overall).
- Random forest and LightGBM are not separated on F2.

Evidence:

The comparison records in `../artifacts/experiments.jsonl`, printed in `../artifacts/run_baselines_production.txt`; `--verify` recomputes each one and compares it with its record.

Uncertainty:

The intervals are conditional on the fixed out-of-fold predictions: they exclude retraining and model-selection variability, are not leaderboard intervals, and carry no multiple-comparison correction.

Review status:

`reviewed`, by the P1.2 review (`../reviews/2026-10-29-p1.2-harness.md`). This ledger revision is committed only after that review converges.

Downstream use:

F2 as the feature baseline for the tree models in phase 2; both tree families kept as candidates; the date-feature ablation of the report.

## Finding 009: The Phase 2 Harness Reproduces the Anchor

Question:

Does the phase 2 engine (`songpop/phase2/`), with its per-fold feature construction, recipes and bank_v2 format, compute exactly what the reviewed phase 1 harness computed, so phase 2 comparisons against the anchor are matched?

Current conclusion:

- The recipe `configs/phase2/recipes/lightgbm_F2_anchor.json` (LightGBM on F2 with the phase 1 parameters, no blocks, no preprocessing, no training weights, estimator seed 0) run through `scripts.run_experiments` on the saved primary folds and all three split seeds gives the bank `p2_lightgbm_F2_anchor_seeds155-2026-1029_e00287a531af`. All ten arrays (row ids, labels, groups, split seeds, folds, out-of-fold predictions, per-fold test predictions, deployment mean, fold AUCs) equal those of the phase 1 bank `production_lightgbm_F2_9c37e77d549e` bit for bit, and the resolved estimator parameters are identical. Mean fold AUC 0.8314739289.
- Its fold-average deployment output gives submission bytes identical to `submissions/2026-10-29_p1.3_lightgbm_F2.csv`.
- Requesting split seed 2026 alone uses the saved second repeat: its folds and out-of-fold predictions equal the phase 1 bank's second repeat.

Evidence:

`../artifacts/phase2/run_anchor.txt`; the bank and its record in `../artifacts/phase2/experiments.jsonl`; `scripts.run_experiments --verify-all` recomputes it and compares; `../artifacts/phase2/check_phase2.txt`, check "anchor", recomputes it in memory and compares every array with the phase 1 bank.

Uncertainty:

None for the equality, on this machine and environment (LightGBM 4.7.0 with `deterministic=True`, four threads). Equality was not tested for the other families, whose phase 1 recipes are not phase 2 anchors.

Review status:

`reviewed`, by the P2.1 review (`../reviews/2026-10-29-p2.1-infrastructure.md`), converged after two rounds and committed (719a934). The bank `p2_lightgbm_F2_anchor_seeds155-2026-1029_e00287a531af` is the one that review approved.

Downstream use:

The anchor of phase 2: the comparator of the searches and feature blocks, and of the nested evaluation's fallback (`plans/phase2-models-features.md` sections 5.4 and 8).

## Finding 010: Phase 2 Runtime Measurements

Question:

What does one cross-validated run (15 outer fit positions: three split seeds by five folds) cost per model family on this machine, so that the phase 2 search budgets rest on measurement?

Current conclusion:

On the Apple M5 (10 cores), one process, four model threads, the committed P2.1 engine (719a934), all three split seeds, nothing else running: LightGBM 2.2 to 27.0 s (anchor 7.5 s; with the date count and target encoding 10.4 s; with three estimator seeds, 45 estimator fits, 23.4 s); random forest 8.2 to 26.9 s; Extra Trees 10.7 s; XGBoost with 1,000 trees 6.8 s; HistGB 4.6 s under its four-thread limit; logistic regression 0.4 to 0.6 s; a kNN proxy 0.5 s. These are representative cases, not the most expensive configurations of the phase 2 spaces (for example 63 leaves and 1,600 trees for LightGBM), and the kNN proxy uses eleven standardized columns, not the block-weighted representation of the plan.

Evidence:

`../analysis/phase2_timing.py`, output `../analysis/phase2_timing_output.txt`, which records the data and split hashes, package versions, the source manifest and every case's full recipe. Verification: `.venv/bin/python -B -I analysis/phase2_timing.py .` (timings vary by a few percent between runs). An earlier run of an earlier version of the script, while a Codex review used the machine, took up to twice as long, and HistGB before P2.1's thread limit took 158 s.

Uncertainty:

Wall-clock times depend on machine load and vary between runs. CatBoost, EBM and TabM are not installed and are timed in their tasks. The script also prints each case's mean fold AUC; these are probe exposures, already seen, and select no recipe and allocate no batch.

Review status:

`reviewed`, by the P2.0b review (`../reviews/2026-10-29-p2.0b-search-amendment.md`), converged in one round and committed (ee557e2).

Downstream use:

The budgets and reservations of `plans/phase2-models-features.md` sections 5 to 8; each later task re-times its actual configurations before sizing a batch.

## Finding 011: The P2.2 Base-Feature Search

Question:

What do broadly searched and locally refined LightGBM, random forest, XGBoost, Extra Trees, HistGB, logistic and kNN configurations score on the base features under the production protocol, and which searches continued under the allocation rule (plan sections 5.6 and 15)?

Current conclusion:

- 348 configurations, all on the 3,863 eligible rows, the saved primary folds, all three split seeds and estimator seed 0: LightGBM 88, random forest 48, XGBoost 48, Extra Trees 56, HistGB 12, logistic 48, kNN 48; 77.2 minutes of fitting. Waves 1 to 3 ran; at wave 4 no family had allowance, so the search ended Friday 2026-10-30 at about 00:00 PDT, five hours before the cutoff.
- H, the LightGBM anchor's P2.2 run, equals the P2.1 anchor bank in all ten arrays (the `h_check` record).
- Best complete configuration per family, mean fold AUC, and its difference from H per repeat (split seeds 155, 2026, 1029):

| Family | Best configuration | Mean fold AUC | Minus H | By repeat |
|---|---|---|---|---|
| LightGBM | `p22_lightgbm_b3_001` (wave 3) | 0.8339 | +0.0024 | +0.0029, +0.0016, +0.0028 |
| XGBoost | its anchor (Finding 010's representative recipe) | 0.8290 | -0.0025 | -0.0031, -0.0017, -0.0027 |
| Extra Trees | `p22_extra_trees_b3_001` (wave 3) | 0.8289 | -0.0026 | -0.0031, -0.0017, -0.0030 |
| Random forest | `p22_random_forest_b1_0_040` | 0.8281 | -0.0034 | -0.0036, -0.0049, -0.0017 |
| HistGB | its anchor | 0.8265 | -0.0050 | -0.0058, -0.0073, -0.0018 |
| Logistic | `p22_logistic_b3_002` (`spline` on F1) | 0.7895 | -0.0420 | -0.0450, -0.0413, -0.0397 |
| kNN | its anchor (k 60, distance weights, alpha 1) | 0.7829 | -0.0486 | -0.0481, -0.0489, -0.0488 |

- The best LightGBM configuration has 63 leaves, 600 trees, learning rate 0.01, `min_child_samples` 10, `colsample_bytree` 0.7, `subsample` 0.7, `reg_alpha` 0.0714, `reg_lambda` 1. Its paired cluster bootstrap against H (2,000 replicates, seed 155001): +0.0024, interval -0.0003 to +0.0054, 0.959 of draws above zero.
- Allocation: in wave 1, LightGBM (smaller useful progress, competitive refinement, ensemble contribution), Extra Trees (smaller useful progress) and logistic (clear standalone progress, against its weak anchor) made progress; random forest, XGBoost, HistGB and kNN met no route against their anchors and stopped after their initial batches. The local refinements of waves 2 and 3 made no progress, and the three remaining families stopped.

Evidence:

The run, contribution, allocation, bootstrap, comparison and `h_check` records in `../artifacts/phase2/experiments.jsonl`; the banks in `../artifacts/phase2/bank/`; the manifests in `../artifacts/phase2/batches/`; `../analysis/p22_summary.py` and its output `../analysis/p22_summary_output.txt`. `scripts.run_experiments --audit-task P2.2` (completion) passes; `--verify-task P2.2` reproduces 329 of the 348 runs exactly, and the other 19 pass the amended criterion of plan section 15.7 (Finding 013).

Uncertainty:

Exploratory: each best configuration was chosen on the same three repeats that score it, among many trials (plan section 5.7); the full three-repeat result is a matched evaluation, not an independent confirmation. Bootstrap intervals are conditional on the fixed predictions. A family that stopped under the allocation rule was not shown to be useless: its configurations remain candidates for the ensembles (Finding 012). The thresholds are allocation heuristics. The random forest's best configuration leads `p22_random_forest_b1_0_041` by 6.1e-5 in the logged scores, inside the guaranteed tie range of Finding 013: it is the best by the logged scores and the declared ranking rules, not under every tie-breaking.

Review status:

`reviewed`, by the P2.2 review (`../reviews/2026-10-29-p2.2-searches.md`): its completion checkpoint converged in round 4 and P2.2 was committed (810d1ce).

Downstream use:

The cross-family checks of plan section 6.5 (the random-forest configuration with the highest mean fold AUC), the strongest recipes of P2.4, the pool of P2.6b, and the optional calibration submission `../submissions/2026-10-30_p2.2_lightgbm_best.csv`.

## Finding 012: Contributions to H and the Allocation Outcomes

Question:

Which P2.2 configurations complement H in the declared 0.9/0.1 and 0.75/0.25 probability blends, and how did contribution enter the allocation decisions?

Current conclusion:

- Best contribution blend with H per family (unrounded mean fold AUC; difference from H per repeat):

| Family | Configuration and weight | Blend AUC | Minus H | By repeat |
|---|---|---|---|---|
| Extra Trees | its anchor (500 trees, leaf 2, all features), 0.25 | 0.8351 | +0.0036 | +0.0032, +0.0042, +0.0035 |
| Random forest | `p22_random_forest_b1_0_040`, 0.25 | 0.8338 | +0.0023 | +0.0021, +0.0021, +0.0028 |
| LightGBM | `p22_lightgbm_b3_004`, 0.25 | 0.8338 | +0.0023 | +0.0020, +0.0027, +0.0023 |
| kNN | its anchor, 0.25 | 0.8325 | +0.0011 | +0.0009, +0.0010, +0.0012 |
| HistGB | `p22_hist_gb_b1_0_009`, 0.25 | 0.8325 | +0.0010 | +0.0006, +0.0023, +0.0000 |
| XGBoost | `p22_xgboost_b1_0_020`, 0.25 | 0.8324 | +0.0009 | +0.0008, +0.0009, +0.0010 |
| Logistic | `p22_logistic_b1_1_004`, 0.1 | 0.8314 | -0.0001 | -0.0005, +0.0002, -0.0001 |

- The bagged tree families complement H more than another boosted family does: an Extra Trees or random forest member at weight 0.25 adds as much as the best single LightGBM recipe alone (+0.0024), and the Extra Trees anchor adds more.
- Contribution counted as progress only when a blend improved H, and the family's previous best blend, by a mean of at least 0.0005 with a positive difference on every repeat (plan section 15.2). Extra Trees: its anchor's 0.75/0.25 blend (0.8351228937) remained the family's best; the best other Extra Trees blend (`p22_extra_trees_b1_0_019`, 0.8350177845) was lower. Random forest: its best blend, `p22_random_forest_b1_0_040` (0.8338065820), improved on its anchor's (0.8333297379) by 0.0004768441, short of the 0.0005 margin, so random forest stopped after its initial batch. Both families' strongest contributors are kept for P2.6b.
- Decision-driving bootstraps (descriptive, 2,000 replicates): LightGBM's ensemble witness against H +0.0021 (interval +0.0012 to +0.0032); LightGBM's smaller-useful witness against its anchor +0.0019 (-0.0008 to +0.0051); Extra Trees' smaller-useful witness +0.0023 (-0.0002 to +0.0048); logistic's clear-standalone witness +0.0067 (-0.0018 to +0.0152).

Evidence:

As Finding 011: the contribution, allocation and bootstrap records, recomputed by `--audit-task P2.2`; `../analysis/p22_summary_output.txt`.

Uncertainty:

Fixed weights only; the learned ensembles of plan section 7 may weigh members differently. The blends with H are diagnostics of complementarity on fixed out-of-fold predictions, not ensemble candidates, and are as exploratory as Finding 011.

Review status:

`reviewed`, by the P2.2 review (`../reviews/2026-10-29-p2.2-searches.md`): its completion checkpoint converged in round 4 and P2.2 was committed (810d1ce).

Downstream use:

The pool rule of plan section 7.1 (standalone and contribution winners per family), the channels of the learned procedures, the report's model comparison.

## Finding 013: Tie Sensitivity of the Random Forest and Extra Trees Scores

Question:

Why do 19 of the 348 P2.2 runs not reproduce their stored scores exactly in the full refit, and could the difference have changed any P2.2 result?

Current conclusion:

- `--verify-task P2.2` reproduced 329 runs exactly. The other 19 are random forest (4) and Extra Trees (15) runs. Their predictions recompute inside the plan's prediction tolerance of 1e-12 (section 4.4), but AUC-valued fields differ: the fold AUCs in 16 runs, only the pooled out-of-fold AUCs or the diagnostics in 3.
- Cause: a forest's `predict_proba` with `n_jobs` 4 sums the trees' probabilities in parallel threads, so the summation order and the last bits of a prediction vary between calls (20 identical calls on one fitted Extra Trees model gave 4 distinct outputs with `n_jobs` 4 and 1 with `n_jobs` 1). Predictions tied in exact arithmetic can then come out in either order, which moves an AUC by multiples of one pair.
- Amended verification (plan section 15.7, Sam's decision), passed. Over three further realizations of the 19 runs, no bank or log field outside the AUC-valued ones differed; with each realization's scores and contribution blends substituted, every P2.2 allocation (routes, progress, counter) and every re-planned wave 2 and 3 manifest was unchanged, and wave 4 stayed empty. A negative control that lowers the Extra Trees wave 3 incumbent by 0.01 was detected. Largest observed differences from the saved banks: predictions 3.3e-16, a fold AUC 2.95e-5, a mean fold AUC 3.19e-6, a contribution blend's mean fold AUC 0. These are observed, not bounds.
- Guaranteed ranges, without fitting, over every realization whose predictions lie within 1e-12 of the saved ones:

| Family | Runs | With tolerance-ambiguous pairs | Largest possible change: a fold AUC | A mean fold AUC | Best configuration stays best |
|---|---|---|---|---|---|
| Random forest | 48 | 48 | 4.64e-4 | 1.93e-4 | Not guaranteed: `b1_0_040` (0.8280717) leads `b1_0_041` (0.8280109) by 6.1e-5; their ranges overlap |
| Extra Trees | 56 | 56 | 5.70e-4 | 4.24e-4 | Guaranteed: `b3_001`'s lower end, 0.8288568, is above every other run's upper end (highest 0.8286345) |

- The worst case is about a hundred times the observed differences, because it lets every ambiguous pair break the same way. Every P2.2 forest run is tie-sensitive in principle, not only the 19; the other 85 reproduced exactly in the refit.
- The P2.2 results stand under the amended criterion: the logged scores are kept, every decision was unchanged over the sampled realizations, and later rankings take the logged scores (plan section 15.7).

Evidence:

`../artifacts/phase2/verify_task_p22.txt` (the refit's output); `../analysis/p22_tie_check.py` and its output `../analysis/p22_tie_check_output.txt` (the cause, the guaranteed ranges, three realizations under the amended criterion, the counterfactual chain and its negative control).

Uncertainty:

Three realizations sample the jitter, they do not bound it. The guaranteed ranges bound any realization within the prediction tolerance, but the decisions were not re-judged at those extremes: a decision whose margin lies inside the guaranteed range of its runs, such as the random forest's best against its runner-up, holds for the logged and the sampled realizations, not for every possible one. The same mechanism affects the phase 1 random forest baseline, whose verification has so far passed.

Review status:

`reviewed`, by the P2.2 review (`../reviews/2026-10-29-p2.2-searches.md`): its completion checkpoint converged in round 4 and P2.2 was committed (810d1ce).

Downstream use:

Later rankings of P2.2 forest scores (the comparator of plan section 6.5, the pool rule of section 7.1) take the logged scores with their own tie rules (plan section 15.7). The next task that changes the phase 2 code makes forest predictions deterministic (prediction with one thread, or an equivalent fixed summation order) and re-checks the other families' exact reproduction, before any random forest or Extra Trees configuration enters a frozen menu; the finalists of P2.6b are refitted under their final seed policy anyway.

## Finding 014: Identifier Structure and Deployment Availability

Question:

What do the identifier columns, the file order and the exact audio signatures contain, and what could a model trained on the provided rows legally carry from them to the test rows (plan section 16.2)?

Current conclusion:

- Every `uri` is `spotify:track:` with a 22-character token in the alphabet `0-9A-Za-z` (3,863 eligible training rows, 863 test rows; the incomplete training row has none). Positions 1 to 21 use all 62 characters in each file; the first character is always 0 to 7.
- `track_href` and `analysis_url` equal their fixed prefix plus the token on every row, and `type` is constant: neither adds information beyond the token.
- No test token occurs in training. Fragment coverage of test rows by the training tokens: prefixes of length 1 and 2 all covered, length 3 12.17%; suffixes of length 1 all covered, length 2 63.50%, length 3 1.74%. A whole token has support from no other primary group, so whole-token identity carries nothing across groups.
- Test `ID` is unique and increasing, 2 to 967, with gaps of 1 (769), 2 (83) and 3 (10); it has no counterpart in the training file. Spearman correlation with release year: training row position -0.0186, test `ID` -0.0007.
- Exact audio signatures: no test row shares audio and date with a training row; 2 test rows share audio only, with 2 distinct training signatures; 8 of the 3,629 training audio signatures are held by more than one primary group.

Evidence:

`../analysis/identifier_probe/probe.py`, mode `audit` (no labels, no fitting), output `../analysis/identifier_probe/audit_output.txt`.

Uncertainty:

Descriptive and exact. It says what the supplied files contain, not how Spotify generates identifiers.

Review status:

`reviewed`, by the P2.8 review (`../reviews/2026-10-30-p2.8-gap-hunt.md`): its completion checkpoint converged in round 5 and P2.8 was committed (5e2d5a0).

Downstream use:

The probe design of Finding 015; the strategy's identifier rows (strategy section 5).

## Finding 015: The Identifier and Order Probes

Question:

Do token-derived features, an identifier-only model, the training row position or exact audio recurrence improve the anchor LightGBM or the phase 1 random forest under the production protocol, by the thresholds declared before any probe score (plan section 16.2)?

Current conclusion:

- The probe reproduces its references: H equals the phase 1 LightGBM F2 bank in all ten arrays; R equals the phase 1 random forest bank except for predictions within 2.2e-16 (one-thread prediction), with identical fold AUCs.
- No test meets its follow-up rule, so no pseudotoken control ran. Mean fold AUC differences against the comparator (split seeds 155, 2026, 1029; 10,000 shared group-bootstrap draws):

| Test | Difference | By repeat | 95% interval | One-sided bounds at 0.05/8 |
|---|---|---|---|---|
| I1, H with `id_shape_v1` | -0.013554 | -0.012306, -0.013893, -0.014461 | -0.018787 to -0.008398 | -0.020047, -0.006878 |
| I2, H with `id_frequency_v1` | -0.001676 | -0.002064, -0.002934, -0.000031 | -0.003683 to +0.000368 | -0.004209, +0.000972 |
| I3, H with frequency and `id_te30_v1` | -0.004111 | -0.004336, -0.004459, -0.003538 | -0.007293 to -0.000881 | -0.008129, -0.000013 |
| I4, R with all three blocks | -0.028337 | -0.029023, -0.027114, -0.028875 | -0.034513 to -0.022254 | -0.036316, -0.020615 |
| I5, 0.9 H + 0.1 L | -0.000289 | -0.000409, -0.000217, -0.000240 | -0.000830 to +0.000246 | -0.000969, +0.000387 |
| I6, 0.75 H + 0.25 L | -0.001801 | -0.001923, -0.001797, -0.001682 | -0.003267 to -0.000349 | -0.003728, +0.000084 |
| I7, H with row position (diagnostic) | -0.001171 | -0.000181, -0.002116, -0.001217 | -0.003039 to +0.000692 | -0.003500, +0.001225 |
| I8, H with audio recurrence | +0.000262 | +0.000785, +0.000000, +0.000000 | -0.000110 to +0.000636 | -0.000211, +0.000736 |

- The identifier-only linear model L alone: mean fold AUC 0.501441, chance level. I3 minus I2, described only: -0.002435.
- Coverage on held-out rows: two-character prefixes 98.97% to 100% of rows per fold, two-character suffixes 49.87% to 55.83%, three-character fragments at most 11.13%; audio recurrence supported 0 to 6 held-out rows (0 to 5 groups) per fold.
- I8's score change comes from one of its 15 folds: split seed 155, fold 0, where all 772 held-out predictions differ from H's and the fold AUC rises by 0.003926, a fold with no supported held-out row. In the other 14 folds its held-out predictions equal H's, although 1 to 6 held-out rows per fold had support. The probe does not establish why that fold changed.
- On the declared grouped folds, these identifier representations did not demonstrate a repeat-stable improvement meeting the practical threshold. This does not establish that every possible identifier transformation is uninformative. The row-position diagnostic found no useful effect on held-out training groups; it says nothing about the test file's order.

Evidence:

`../analysis/identifier_probe/probe.py`, mode `run`, output `../analysis/identifier_probe/run_output.txt`; the menu `../configs/phase2/p2.8/identifier_probe.json`, saved before any probe score; the banks and `results.json` in `../artifacts/p2.8/identifier_probe/`; mode `verify`, output `../analysis/identifier_probe/verify_output.txt`.

Uncertainty:

Bootstrap bounds are conditional on the fitted predictions and exclude retraining variability. I1 and I4 lost AUC under unchanged model settings; why is not established. That the added columns perturb LightGBM's column sampling and the forest's feature sampling is an untested hypothesis: no pseudotoken control or sampling ablation ran, since no test met its follow-up rule. The models were not tuned for the added columns.

Review status:

`reviewed`, by the P2.8 review (`../reviews/2026-10-30-p2.8-gap-hunt.md`): its completion checkpoint converged in round 5 and P2.8 was committed (5e2d5a0).

Downstream use:

No identifier block enters the phase 2 engine; the strategy's identifier contents row records the null; the report's account of tested and rejected inputs.

## Finding 016: The Feature Checks Moved from P2.4

Question:

What do the date encodings, the other declared blocks and the `music` block add to the anchor, and the date encodings to P2.2's best random forest and its strongest LightGBM recipe, under the production protocol; which section 5.6 routes do they meet against P2.2's saved state; and do the Extra Trees blends of Finding 012 reproduce under deterministic forest prediction (plan section 16.3)?

Current conclusion:

- P2.8's anchor run equals the P2.1 anchor bank in all ten arrays.
- Matched differences, each challenger against the identical recipe without its addition (unrounded mean fold AUC; paired cluster bootstrap with 2,000 draws, descriptive):

| Id | Base and addition | Challenger | Difference | By repeat | 95% interval | Routes |
|---|---|---|---|---|---|---|
| B1 | H + `date_count` | 0.832283 | +0.000810 | +0.001730, -0.000582, +0.001281 | -0.001034 to +0.002678 | none |
| B2 | H + `date_te10` | 0.833938 | +0.002464 | +0.002268, +0.002995, +0.002130 | -0.001001 to +0.006249 | none |
| B3 | H + `date_te30` | 0.832662 | +0.001188 | +0.000078, +0.002689, +0.000797 | -0.002113 to +0.004837 | none |
| B4 | H + `date_count`, `date_te10` | 0.833666 | +0.002192 | +0.002899, +0.001330, +0.002345 | -0.001349 to +0.005926 | none |
| B5 | H + `date_count`, `date_te30` | 0.832716 | +0.001242 | +0.001997, +0.000828, +0.000902 | -0.002187 to +0.005135 | none |
| B6 | H + `interactions` | 0.829059 | -0.002415 | -0.000597, -0.003574, -0.003075 | -0.005891 to +0.000998 | none |
| B7 | H + `cyclic` | 0.830125 | -0.001349 | -0.000965, -0.002553, -0.000528 | -0.003577 to +0.001102 | none |
| B8 | H + `era_relative` | 0.830378 | -0.001096 | -0.000999, -0.002101, -0.000189 | -0.002719 to +0.000620 | none |
| B9 | H + inverse-group-size weights | 0.829829 | -0.001645 | -0.000228, -0.002088, -0.002619 | -0.003300 to +0.000049 | none |
| B10 | H + `music` | 0.831788 | +0.000314 | -0.000388, +0.001322, +0.000009 | -0.002505 to +0.002948 | none |
| B11 | Random forest `b1_0_040` + `date_count` | 0.828527 | +0.000456 | +0.000924, +0.001425, -0.000982 | -0.001537 to +0.002333 | none |
| B12 | Random forest `b1_0_040` + `date_count`, `date_te30` | 0.827502 | -0.000570 | -0.001581, +0.001385, -0.001514 | -0.004988 to +0.004044 | ensemble contribution |
| B13 | LightGBM `b3_001` + `date_count` | 0.834333 | +0.000431 | -0.000102, +0.001702, -0.000306 | -0.001462 to +0.002258 | none |
| B14 | LightGBM `b3_001` + `date_count`, `date_te30` | 0.835421 | +0.001519 | +0.000323, +0.003993, +0.000242 | -0.002082 to +0.005176 | smaller useful, competitive refinement, ensemble contribution |

- The exact-date target encodings raise the anchor's mean fold AUC by +0.0012 to +0.0025, positive on every repeat for B2 to B5, with every interval including zero; the date count alone and `music` give smaller differences with a negative repeat, and `interactions`, `cyclic`, `era_relative` and the training weights lower it. On the tuned LightGBM recipe the count with `date_te30` adds +0.0015 (B14), the highest standalone mean fold AUC so far. The batch meets routes through B12 (its 0.25 blend with H, 0.835241) and B14 (against the tuned incumbent 0.833902).
- Fixed-blend diagnostics with the deterministic Extra Trees anchor: 0.75 H + 0.25 Extra Trees, 0.835123, +0.003649 over H (repeats +0.003237, +0.004162, +0.003548; interval +0.002080 to +0.005211), reproducing Finding 012's best contribution; 0.75 `b3_001` + 0.25 Extra Trees, 0.835857, +0.001955 over `b3_001` (repeats +0.001654, +0.002531, +0.001680; interval +0.000503 to +0.003308).
- No checked block shows a large omitted signal: the largest gains are of the same order as P2.2's tuning gain, and the Extra Trees blends remain the largest measured improvements.

Evidence:

The P2.8 run records and banks in `../artifacts/phase2/experiments.jsonl` and `../artifacts/phase2/bank/`; the recipes and saved state in `../configs/phase2/p2.8/`; `../analysis/p28_feature_checks.py` and its output `../analysis/p28_feature_checks_output.txt`; `scripts.run_experiments --verify --task P2.8`, output `../artifacts/phase2/verify_task_p28.txt`.

Uncertainty:

Exploratory (plan section 5.7): every interval of the matched comparisons includes zero; the tuned recipe and its base were chosen on the same repeats that score them; bootstrap intervals are conditional on the fixed predictions; routes are allocation heuristics. The earlier audio-only project's music gain (`prior_project/`) did not appear on the anchor with release dates; CatBoost and EBM remain to be checked (P2.4's reserved checks).

Review status:

`reviewed`, by the P2.8 review (`../reviews/2026-10-30-p2.8-gap-hunt.md`): its completion checkpoint converged in round 5 and P2.8 was committed (5e2d5a0).

Downstream use:

P2.4 imports B1 to B14 as its first 14 comparisons with their routes (plan section 16.3); the date encodings with the tuned recipes and forests; P2.6b's pool (the Extra Trees contribution); the reserved CatBoost and EBM checks.

## Finding 017: Deterministic Forest Prediction

Question:

Does predicting random forests and Extra Trees with one thread make their outputs repeat exactly, and does P2.8's code change leave every other family unchanged (plan section 16.4)?

Current conclusion:

- One fitted random forest (P2.2's anchor recipe) and one fitted Extra Trees model (the leaf-1 recipe of Finding 013) each gave 1 distinct output in 20 repeated predictions, with four threads restored after prediction.
- The five non-forest P2.2 family anchors (LightGBM, XGBoost, HistGB, logistic, kNN), refitted under P2.8's code, equal their P2.2 banks in all ten arrays.
- The forest anchors equal theirs in every array except predictions, within 2.2e-16. The random forest anchor's fold AUCs are identical; the Extra Trees anchor's differ in 10 of 15 folds by at most 7.34e-6, mean fold AUC +4.89e-10, of the size Finding 013 observed for tie breaking. Each forest anchor refitted twice under P2.8's code is identical in all ten arrays.
- Later forest runs use a fixed prediction summation order; exact replay holds under the recorded environment and execution path.

Evidence:

`../analysis/p28_forest_check.py` and its output `../analysis/p28_forest_check_output.txt` (rerun identical at the verification); the self-check "forest prediction" of `../scripts/check_phase2.py` (a pipeline fixture and a thread spy), output `../artifacts/phase2/check_phase2.txt`.

Uncertainty:

The seven anchor refits are regression coverage, not proof for every recipe path; the self-checks cover the dispatch. Exact replay is shown on this machine, environment and thread setting.

Review status:

`reviewed`, by the P2.8 review (`../reviews/2026-10-30-p2.8-gap-hunt.md`): its completion checkpoint converged in round 5 and P2.8 was committed (5e2d5a0).

Downstream use:

Every later random forest and Extra Trees run, including P2.8's (B11, B12, the Extra Trees anchor) and any forest in a frozen menu; Finding 013's follow-up is complete.

## Finding 018: CatBoost and EBM Installation, Runtime Compatibility and Cost

Question:

Are CatBoost and EBM installed exactly as their reviewed route locks specify, do they fit and predict the project's F2 data through the phase 2 engine as declared and repeatably, and what does one 15-fit evaluation cost (plan section 17.2 and 17.5)?

Current conclusion:

- Installed on 2026-10-30 with the commands P2.1 reviewed, unchanged: the CatBoost route added exactly catboost 1.2.10, graphviz 0.21 and plotly 7.1.0; the EBM route exactly interpret-core 0.7.8. No previously installed version changed (`check-before` and `check-after` for each route), `pip check` found no broken requirement, and the phase 1 commands of exit criterion 5 pass afterwards.
- Both libraries fit F2 with its missing date parts and predict full, month-only and year-only dates; two fits with one seed predict identically; a pickle round trip predicts identically; identical rows, missing values included, get identical predictions; training weights are accepted; no new top-level project entry appears and CatBoost writes no `catboost_info` directory.
- Defaults that would change the declared method are pinned in the factories. CatBoost: without a declared bootstrap, CatBoost 1.2.10 uses MVS with subsample 0.8 on the CPU for Plain and Ordered alike; the factory passes these explicitly with `rsm` 1.0, Logloss, `nan_mode` Min, no class weights, no evaluation set and no stopping. Its `predict_proba` uses every CPU core by default; the engine passes four threads, and predictions were identical with one, four and all threads on a probe. EBM: interpret-core 0.7.8 defaults to automatic pairs ("3x"), a 0.15 validation split with early stopping, and 14 outer bags; the factory passes no validation split, no early stopping, one bag and only the declared pairs, writes the remaining defaults out, and bins every feature as continuous (also what it infers for these frames). With validation and early stopping disabled, each EBM stage makes `max_rounds` times its number of terms boosting updates, checked.
- Score-free timing at the readiness sources, one 15-fit run on the three split seeds with four threads, feature construction and prediction included: CatBoost 2.8 s (depth 3, 400 iterations, Plain), 11.0 s (depth 5, 1,200, Plain), 131.2 s (depth 7, 2,000, Ordered) and 371.0 s (depth 8, 3,000, Ordered, the largest range extension allowed); EBM 5.5 s (main effects, 300 rounds), 13.5 s (pairs, 700), 27.0 s (pairs, 1,500) and 54.4 s (pairs, 3,000). Each EBM run records one warning, interpret's notice that missing values are not drawn in its plots. The reservation projects 42 minutes for P2.3's initial batches and 104 minutes at both ceilings, not bounds.

Evidence:

`../artifacts/phase2/install/inventory_before_catboost.json`, `inventory_after_catboost.json`, `inventory_before_ebm.json`, `inventory_after_ebm.json`; the self-checks "CatBoost runtime" and "EBM runtime" of `../scripts/check_phase2.py`, output `../artifacts/phase2/check_phase2.txt`; the P2.3 timing and reservation records in `../artifacts/phase2/experiments.jsonl` (`../configs/phase2/p2.3/timing.json`).

Uncertainty:

The runtime checks use samples of 1,500 fitting rows and small models; they establish compatibility and repeatability of the declared paths, not that every searched configuration replays, which is P2.3's completion evidence (`--verify-task P2.3`). Timings are representative cases on one machine, not bounds; Ordered boosting dominates CatBoost's cost.

Review status:

`reviewed`, by the P2.3 review (`../reviews/2026-10-30-p2.3-catboost-ebm.md`): its readiness checkpoint converged in round 2 and its completion checkpoint in round 3; committed (dca38c6) at the migration of 2026-10-30.

Downstream use:

P2.3's searches (plan section 17) and the CatBoost and EBM checks reserved for P2.4 (sections 6.5 and 16.3).

## Finding 019: The CatBoost and EBM Searches on F2

Question:

How well do CatBoost and EBM do on F2 under P2.3's protocol (plan section 17), against H and the strongest recipe so far?

Current conclusion:

- Exploratory: every score below is a matched evaluation on the three repeats that also chose it (plan section 5.7).
- CatBoost, 48 configurations (the initial batch only, 24 minutes of fitting): the best is the default-inspired anchor (depth 6, 1,000 iterations, learning rate 0.0168, L2 3, random strength 1, Plain, MVS), mean fold AUC 0.8306, below H (0.8315) on every repeat (-0.0006, -0.0010, -0.0010) and 0.0048 below P2.8's B14 (0.8354). The earlier project's candidate 31, ported to F2 without the `music` block, 0.8292. Plain boosting: best 0.8306, median 0.8244 over 24; Ordered: best 0.8266, median 0.8218 over 24. Depth 6 gave the best configuration (0.8306); depths 3, 4, 5 and 7 peaked at 0.8235, 0.8267, 0.8248 and 0.8257.
- EBM, 28 configurations (an initial batch of 12, then two local refinement batches of 12 and 4; 5.6 minutes): the best is the declared pairs with learning rate 0.01 and 603 rounds, 0.8162, 0.0153 below H and 0.0193 below B14. Main effects alone peaked at 0.8064, the declared pairs at 0.8162. The main-effects anchor (700 rounds, learning rate 0.03) scored 0.7999.
- Neither family approaches the tuned LightGBM recipes on these features; CatBoost comes within about 0.001 of H.

Evidence:

`../analysis/p23_summary.py` and its output `../analysis/p23_summary_output.txt`; the P2.3 records and banks in `../artifacts/phase2/experiments.jsonl` and `../artifacts/phase2/bank/`; the manifests `../artifacts/phase2/batches/*_w1_initial_*`, `ebm_w2_refinement_*`, `ebm_w3_refinement_*`; `--audit-task P2.3` (`../artifacts/phase2/audit_task_p23.txt`) and the full refit `--verify-task P2.3` (`../artifacts/phase2/verify_task_p23.txt`).

Uncertainty:

CatBoost searched only its initial batch, because no configuration met a route of section 5.6 (Finding 020); its space around the anchor is otherwise untested. CatBoost's automatic learning rate, ordered target statistics for categorical inputs and early stopping were deliberately not used (sections 5.3 and 17.2). EBM's interpret-core 0.7.8 defaults other than those pinned were not searched. F2 only: date encodings and the `music` block on these families are P2.4's reserved checks.

Review status:

`reviewed`, by the P2.3 review (`../reviews/2026-10-30-p2.3-catboost-ebm.md`): its completion checkpoint converged in round 3; committed (dca38c6) at the migration of 2026-10-30.

Downstream use:

P2.4's reserved checks take the best CatBoost (the anchor) and the best EBM (`p23_ebm_b2_001`) by the handoff rule (sections 6.5, 16.3, 17.6); P2.6b's pool rule.

## Finding 020: CatBoost and EBM Contributions to H and the Allocation Outcomes

Question:

Do the CatBoost and EBM configurations add to H in a fixed blend, and what did section 5.6's allocation decide for them?

Current conclusion:

- CatBoost: no configuration met a route, so the family stopped after its initial batch (allowance 0). Its best blend with H is candidate 31 at weight 0.25, 0.8333, +0.0018 over H with every repeat positive (+0.0017, +0.0017, +0.0020); it fails the ensemble route because it improves the anchor's own best blend, the family's previous best blend, by less than 0.0005.
- EBM: seven of the initial batch's eleven new configurations met the clear standalone route against the weak main-effects anchor (bootstrap witness +0.0162, interval +0.0108 to +0.0215), and two of them also met the ensemble route, at weight 0.1. The ensemble witness follows section 15.5's convention, the qualifying configuration's better blend, here at weight 0.25: +0.0007 over H (interval -0.0009 to +0.0023), below H on the first repeat; it describes the comparison and does not change the route, which the weight 0.1 blend met. The allowance of two then bought two local refinement batches without progress, and EBM stopped. Its best blend with H is `p23_ebm_b1_0_002` at weight 0.25, 0.8321, +0.0007 over H.
- The allocation thresholds are heuristics, not significance tests; the bootstrap intervals are conditional on the fitted predictions.

Evidence:

The P2.3 allocation, contribution and bootstrap records (`../artifacts/phase2/experiments.jsonl`), re-judged and recomputed by `--audit-task P2.3`; `../analysis/p23_summary_output.txt`.

Uncertainty:

The ensemble route compares a blend with H and with the family's previous best blend; a family whose anchor already blends well with H (CatBoost) must beat that blend, not only H. Blends with a stronger reference than H are P2.6b's question.

Review status:

`reviewed`, by the P2.3 review (`../reviews/2026-10-30-p2.3-catboost-ebm.md`): its completion checkpoint converged in round 3; committed (dca38c6) at the migration of 2026-10-30.

Downstream use:

P2.6b's pool (CatBoost and EBM members by their contributions); P2.4's re-tuning candidates include only units that meet a route there.

## Finding 021: Exact-Date Coverage Under the Fitting Dictionaries

Question:

How often does a row's exact release-date string occur among the rows that fit its encoders, at each validation level, and with how many distinct groups behind it (plan sections 6.4 and 18.8)?

Current conclusion:

- Outer validation rows: 69.3% have their exact date among the outer fitting rows (8,030 of 11,589 over the three split seeds; per fit 66.1% to 72.5%). Test rows against each outer fitting population: 68.8% on average (67.3% to 69.8%). Inner held-out parts against their inner fitting parts, with the encoder's own inner splits: 65.1% (30,176 of 46,356).
- Support is thin for many covered rows: over the 15 outer fits, on average 16.3% of test rows (14.0% to 18.1%) have a date held by exactly one fitting group and 29.1% (28.2% to 30.2%) by five or more; distinct dates are covered less often than rows (59.1% of the test rows' distinct dates on average, 57.2% to 60.4%).
- The inner coverage is about four points below the outer one, as expected from the smaller inner fitting parts; the test rows' coverage is close to the validation rows', so the validation measures the encoding at about the coverage it would meet on the test rows.
- Descriptive only: nothing was fitted, no label was summarized, no smoothing value was chosen from it. The encoding is an exact-date association, not album identity.

Evidence:

`../analysis/p24_date_coverage.py` and its output `../analysis/p24_date_coverage_output.txt` (per seed and fold, with support bins and distinct-date coverage; per-fit means and pooled counts).

Uncertainty:

The earlier round E figures used the same definition for the validation rows (each fold's training part), and this recomputation reproduces them (69.87%, 68.65% and 69.35% by repeat); only round E's 72.2% for the test rows was measured against the whole training population, while here the test rows are measured against each outer fitting population, as a model sees them (68.8%). Coverage does not say how informative a covered date is.

Review status:

`draft`, in the P2.4 review (`../reviews/2026-10-30-p2.4-features.md`), readiness checkpoint.

Downstream use:

The interpretation of the date-encoding comparisons in P2.4 (Finding 022) and the report's description of the date features.

## Finding 022: The Matched Feature Effects of P2.4

Question:

Which feature blocks, preprocessing alternatives, removals and training weights change each family's mean fold AUC when everything else is held fixed (plan sections 6 and 18)?

Current conclusion:

Matched effects (challenger minus its own comparator, unrounded mean fold AUC; the three repeat differences; paired cluster bootstrap with 2,000 draws, conditional on the fixed predictions):

- Date encodings help the boosted trees. Tuned LightGBM: `date_count` with `date_te3` +0.0028 (+0.0022, +0.0043, +0.0018) to 0.8367; `date_te3` or `date_te10` alone +0.0020; `date_te30` +0.0016; `date_te100` +0.0005. CatBoost: +0.0030 to +0.0047, best `date_count` with `date_te3` (+0.0045, +0.0055, +0.0042; interval +0.0006 to +0.0088) to 0.8353. XGBoost: +0.0010 to +0.0030, best `date_te3` to 0.8319. Smaller smoothing values did as well or better than larger ones in every boosted family.
- The forests barely respond: random forest -0.0006 to +0.0009 and Extra Trees -0.0007 to +0.0016 across the date variants, no consistent sign.
- The smooth and distance models gain from representation changes: logistic regression with `spline_calendar` and the `cyclic` block +0.0048 (interval +0.0012 to +0.0085), with a date target encoding on top of `spline_calendar` +0.0100 to +0.0117 (to about 0.80, still far below the trees); kNN with robust audio scaling +0.0042, and Manhattan distance on it a further +0.0043.
- The other blocks: interactions +0.0020 for Extra Trees, -0.0011 for CatBoost, -0.0021 for XGBoost; `music` -0.0001 for CatBoost, +0.0008 for EBM; `cyclic` -0.0004 for XGBoost.
- Removals hurt consistently: dropping `days` -0.0011 to -0.0032; dropping `day` and `day_of_year` -0.0027 to -0.0034, most intervals excluding zero. Inverse-group-size training weights -0.0008 to -0.0021.
- P2.8's 14 imported comparisons keep their own judgments (B12 and B14 met routes there).

Evidence:

`../analysis/p24_summary.py` and its output `../analysis/p24_summary_output.txt` (every comparison with its definition, matched effect, repeat differences, bootstrap interval, challenger score, contribution blends and routes; the handoff index); the P2.4 records in `../artifacts/phase2/experiments.jsonl` and the batch manifests `../artifacts/phase2/batches/p24_*.json`; `--audit-features P2.4` passes (220 runs audited against their banks and records).

Uncertainty:

Exploratory in the sense of section 15.6: adaptively chosen comparisons on repeatedly used folds; three repeats and conditional bootstrap intervals do not remove the selection optimism. Effects of 0.001 to 0.003 are of the order of the repeat-to-repeat spread. The date encodings depend on exact-date coverage (Finding 021), which is thinner for many rows.

Review status:

`draft`, provisional (agent-workflow v1); reviewed with the results that feed the final selection (R1).

Downstream use:

P2.6b's nominations draw on these runs; the report's description of which features matter.

## Finding 023: P2.4's Expansions and Re-tuning Outcomes

Question:

How far did the feature stage's allocation rule carry the search, and what did re-tuning add (plan sections 18.3 to 18.6)?

Current conclusion:

- The feature stage ran its full 96 comparisons: the initial 32 (14 imported from P2.8, 18 new) and four expansions of 16; the counter went 2, 2, 2, 1, 2 (the fourth batch made no progress, the fifth did), and the stage closed at the comparison ceiling, with no recorded stop. 18 comparisons met an allocation route.
- Re-tuning: four units chosen by the rule (LightGBM with `date_count` and `date_te3`; LightGBM with `date_te10`; Extra Trees with `date_count` and `date_te30`; CatBoost with `date_count` and `date_te3`), two rounds of 16 each, 128 configurations. No round made progress for any unit. The best re-tuned configuration, 0.8368 (LightGBM, `date_count` and `date_te3`), is within 0.0001 of its untuned recipe B24 (0.8367); CatBoost reached 0.8355, Extra Trees 0.8287.
- The strongest P2.4 standalone, 0.8368, is 0.0014 above P2.8's leader (0.8354) and 0.0053 above the phase 1 anchor (0.8315). The best contribution blend with H is 0.75 H + 0.25 Extra Trees with `date_te3`, 0.8371.
- Execution: 2026-10-30, 20:53 to 23:55 PDT, partly beside P2.5 and full self-checks, so the run times are inflated by load.

Evidence:

`../analysis/p24_summary_output.txt` (batch judgments, re-tuning units, best runs, the handoff index); `../artifacts/phase2/p24_execute_log.txt` is the execution log; the audit output `../artifacts/phase2/audit_features_p24.txt`.

Uncertainty:

As Finding 022. Re-tuning around already tuned parents found nothing measurable, which bounds what local search adds at these features, not what a wider search could find. `--verify --task P2.4`, the full refit at 9731e7f, reproduces all 220 runs and all 82 matched comparisons exactly (`../artifacts/phase2/verify_task_p24.txt`).

Review status:

`draft`, provisional; reviewed with the results that feed the final selection (R1).

Downstream use:

P2.6b's discovery inventory and nominations (plan section 20.2).

## Finding 024: TabM Installation, Process Isolation, Repeatability and Cost

Question:

Can TabM run inside the phase 2 engine on this machine without changing any classical computation, does it repeat exactly, and what does it cost (plan sections 5.5 and 19)?

Current conclusion:

- Installation: torch 2.14.1 and tabm 0.0.3 (with rtdl-num-embeddings 0.0.12 and their dependencies) were installed from the hash-pinned lock reviewed in P2.1 (`../configs/phase2/install/tabm.txt`); `install_locks check-before tabm` and `check-after tabm` passed (the after-inventory equals the before-inventory plus exactly the lock). No installed version changed; the package identity of classical runs (`songpop.artifacts.packages`) is unchanged.
- Process isolation: in a process that has loaded LightGBM or XGBoost, torch's first matrix product crashes the process (exit 139, a segmentation fault); CatBoost, interpret and scikit-learn do not trigger it, and torch loaded first does not crash. LightGBM and XGBoost link Homebrew's OpenMP runtime and torch bundles its own. Every torch computation therefore runs in one persistent worker process that imports neither, and the fitting process never imports torch, so classical runs keep their exact runtime.
- Repeatability: the representative configuration fitted through the worker on one production training part (split seed 155, fold 0) in two separate fitting processes gave byte-identical validation and test predictions and the same chosen epoch (67 of 87 evaluated, stopped by patience). The self-checks also show byte-identical predictions within one process, after unrelated random draws, across prediction batches and after a worker restart. Verification therefore keeps the classical criterion (1e-12).
- Cost, score-free, one 15-fit run on all three split seeds with four threads, measured while P2.4 and a full self-check ran beside it: the light case (two blocks, width 64, learning rate 3e-3) 101.7 s; the representative case (the anchor N0) 155.7 s; the expensive case (three blocks, width 256, learning rate 3e-4, batch 128, dropout 0.2) 759.2 s. The reservation projects 45.2 minutes for the initial eight configurations and 90.4 minutes at the ceiling of sixteen, within section 5.5's three hours; nothing is screened on one split seed.

Evidence:

`../analysis/p25_runtime_pilot.py` and its output `../analysis/p25_runtime_pilot_output.txt`; the installation inventories `../artifacts/phase2/install/inventory_before_tabm.json` and `inventory_after_tabm.json`; the P2.5 timing records and reservation in `../artifacts/phase2/experiments.jsonl`; the checks "neural early stopping (P2.5)", "TabM through the engine (P2.5)" and "P2.5 budget" in `../scripts/check_phase2.py`.

Uncertainty:

Repeatability is shown on this machine, with these package versions and four threads; it is not claimed across machines or library versions. The pilot's times were measured while P2.4 and a full self-check ran beside it, so they are inflated by load. The crash's cause is inferred from the libraries' linked runtimes and the import-order pattern, not traced in the runtimes themselves.

Review status:

`draft`, provisional under agent-workflow v1; reviewed with the results that feed the final selection (R1).

Downstream use:

P2.5's search and every TabM run of P2.6b.

## Finding 025: The TabM Search

Question:

How strong is TabM on F2 under the stopping-and-refit protocol, alone and blended with H, and does quantile scaling help it (plan sections 5.5 and 19.2)?

Current conclusion:

- TabM is far below the tree models: the eight initial configurations score 0.8007 to 0.8034 mean fold AUC (best: two blocks, width 128, dropout 0.2, weight decay 1e-2, batch 128), against 0.8315 for H, the phase 1 recipe, on every repeat (best minus H by repeat -0.0251, -0.0274, -0.0318), and 0.0320 below G's prior member (P2.8's leader, 0.8354).
- Its blends with H barely move H: the best, 0.9 H + 0.1 TabM (learning rate 3e-3), is 0.8320 against H's 0.8315 (+0.0007, +0.0005, +0.0004 by repeat); the 0.25 blends are smaller. No configuration met an allocation route (section 5.6), so the family stopped after its initial batch: 8 configurations, 34.8 minutes of fitting, well within the three-hour allowance.
- The matched preprocessing comparison: quantile scaling of the audio columns against standardization, everything else equal, is -0.0009 mean fold AUC (-0.0017, -0.0034, +0.0026 by repeat): no consistent effect.
- Stopping behaved as declared: every one of the 120 fits stopped by patience, none at the 300-epoch cap; chosen epochs range from 14 (learning rate 3e-3) to 255 (width 64), medians 26 to 164.

Evidence:

`../analysis/p25_summary.py` and its output `../analysis/p25_summary_output.txt` (with every configuration's stopping receipts and the handoff to P2.6b); the P2.5 records in `../artifacts/phase2/experiments.jsonl` (`--audit-task P2.5` passes; `--verify-task P2.5` reproduces all 9 runs exactly, `../artifacts/phase2/verify_task_p25.txt`); the protocol `../configs/phase2/p2.5/protocol.json`.

Uncertainty:

Exploratory in the sense of section 15.6: matched evaluations on the repeats that also chose them. Eight configurations explore a small part of the space; a wider search, numerical embeddings or longer training were not tried, so this establishes TabM's level under this protocol and budget, not its ceiling. TabM saw only F2 with stateless preprocessing: the date encodings that help the trees were refused to it to keep the stopping split clean (section 19.1).

Review status:

`draft`, provisional; reviewed with the results that feed the final selection (R1).

Downstream use:

P2.6b's nominations (TabM's standalone and contribution winners against the discovery reference) and its seed policy (estimator seed 0, section 19.3).

## Finding 026: The Learned-Procedure Engine, Its Isolation Checks and Its Cost

Question:

Does the P2.6a engine evaluate learned procedures and the nested selection without any label crossing its permitted population, and what do its levels cost relative to an ordinary 15-fit run (plan sections 7.2, 8 and 20.1)?

Current conclusion:

- Isolation, by self-checks that fail on the plausible shortcuts: every innermost meta fit, identified by its fit context, saw exactly its outer training part minus its meta fold (a target-encoded channel included); flipping the labels of an outer fold's held-out rows changed none of that fold's procedure predictions while other folds' moved; flipping one meta-held-out group's labels left its own base predictions unchanged while other rows' moved; in the nested evaluator every fit stayed inside its own outer training part, and flipping an outer validation fold's labels (assignment held fixed) changed none of that fold's evidence, selection, fold-average or full-refit predictions. All six learned settings share one set of base fits (12 cross-fits for 2 channels and 6 settings on 5 outer folds).
- Validation boundaries: the selector refuses incomplete evidence, forged fold AUCs, other folds, split groups, a sixth fold, a key filed under another candidate and a menu mismatch, and its tie set is measured from the single maximum; the cache refuses an entry for another request, malformed arrays and a second insertion with other predictions, and returns copies; the optimizer refuses a reported success where a feasible move still improves the loss (Codex round V's example); procedure banks without their identity and per-fold states are refused, as are incomplete nested results.
- Cost, score-free, measured while P2.4 and a full self-check also ran: one outer fold's meta stage for the anchor channel is 0.327 of the anchor's 15-fit run, so the production evaluation of a learned procedure is about 4.9 C_L (the plan assumed 6 C_L with overhead); the nested evaluation of an atomic candidate is about 3.04 C_all (plan: 3.2); the third level for a learned channel atom about 14.2 C_L (plan: 16.2). The plan's formulas are therefore slightly conservative and are used for the reservation, cross-checked by these multipliers.

Evidence:

`../songpop/phase2/procedures.py`, `selection.py`, `nested.py`; the checks marked "(P2.6a)" and "P2.6b pool rule" in `../scripts/check_phase2.py`; `../analysis/p26_cost_pilot.py` and its output `../analysis/p26_cost_pilot_output.txt`; Codex rounds U and V in `../doc/llm_usage/2026-10-30-p2.6-consultation/`.

Uncertainty:

The isolation checks use the first project training rows and tiny LightGBM recipes, not every menu family; TabM inside learned procedures is covered by its own label-isolation check through the engine. The pilot's times are inflated by concurrent load, and its multipliers come from the anchor alone.

Review status:

`reviewed`: R2 review of the P2.6b freeze `APPROVED` at fd48199 in round 2 (`../reviews/2026-10-31-p2.6b-menu.md`); provisional results until they feed the final selection (R1).

Downstream use:

P2.6b's production evaluation of learned procedures, the P2.7 reservation in `../configs/phase2/menu.json`, and P2.7.

## Finding 027: P2.6b Nominations, Final Seed Policies, H* and the Pool

Question:

Which recipes enter P2.6b's ensembles under the frozen nomination and pool rules, and what do the final seed policies change (plan sections 7.1 and 20.2 to 20.3)?

Current conclusion:

- The discovery inventory holds 656 distinct complete atomic runs of P2.1, P2.2, P2.8, P2.3, P2.4 and P2.5. Its reference R0 is P2.4's re-tuned LightGBM with `date_count` and `date_te3` (0.8368 with estimator seed 0). Against this one R0, each of the ten families' standalone and contribution winners, R0's family's two further configurations within 0.002, and the anchor with its three-seed version give 23 nominees.
- Under the final seed policies (estimator seeds 0, 1 and 2, except kNN and TabM with seed 0), R0's recipe scores 0.8366: 0.0002 below its seed-0 score, a small sign that choosing on seed 0 was slightly optimistic. It remains the strongest, so H\* is that recipe (`p26_lightgbm_a8ce5635`). The next are the two leading-family LightGBM configurations (0.8364, 0.8362), CatBoost with `date_count` and `date_te3` (0.8354) and XGBoost with `date_te3` (0.8325). The anchor's three-seed version is 0.8323 against the anchor's 0.8315.
- Contributions against H\*: the best blends are 0.75 H\* + 0.25 Extra Trees with interactions (0.8394), with CatBoost with `music` (0.8389), and with EBM (0.8387). Weight 0.25 beats 0.1 throughout.
- The pool (eight atoms): the three strongest family representatives (LightGBM H\*, CatBoost, XGBoost); the two best contribution recipes not already in it (Extra Trees with interactions; CatBoost with `music`, a second CatBoost configuration, as section 20.3 allows); the anchor; the two leading-family LightGBM configurations, which form the within-family average with H\*. EBM, random forest, HistGB, logistic regression, kNN and TabM did not enter. The four learned channels are the within-family average, XGBoost, CatBoost with `music` and Extra Trees with interactions. TabM is not a channel, so no cost decision about it was needed.
- The frozen manifest has 33 candidates: 9 atomic and 24 procedures (3 equal averages over channels, 14 contribution blends with H\*, the within-family average, 3 constrained-weight settings, 3 stacker settings), first committed at ae4dd90 before any procedure was scored. The R2 review (round 1, F3) found that the seed-0 anchor and the kNN and TabM nominees still used historical banks; all were rerun under the final sources (the anchor's new bank equals the P2.1 anchor in all ten arrays; kNN and TabM reproduce their scores) and the manifest was regenerated by the same rule at 7e404e4, with the same pool, channels and procedures and only the anchor's run id changed.

Evidence:

`../analysis/p26_nominate.py`, `../analysis/p26_pool.py`, `../configs/phase2/p2.6/nominations.json`, `../configs/phase2/p2.6/procedures.json`; the 23 P2.6b final-policy runs in `../artifacts/phase2/experiments.jsonl` (refitted by `--verify`, `../artifacts/phase2/verify_p26b_runs.txt`); the execution logs `../artifacts/phase2/p26_nominees_log.txt` and `p26_nominees_r2_log.txt`.

Uncertainty:

All scores are exploratory matched evaluations on the folds that also chose them. The nomination rule takes one discovery reference, so contributions measured against other references in earlier tasks were not compared. Final-policy scores are a fresh evaluation of the same recipes, not an independent confirmation.

Review status:

`reviewed`: R2 review of the P2.6b freeze `APPROVED` at fd48199 in round 2 (`../reviews/2026-10-31-p2.6b-menu.md`); provisional results until they feed the final selection (R1).

Downstream use:

The procedure evaluation, the menu (Finding 028) and P2.7.

## Finding 028: The P2.6b Procedures and the Frozen Menu

Question:

How do the frozen procedures score on the 15 production folds, and what menu and production selection does the frozen rule give (plan sections 7, 11 and 20.4)?

Current conclusion:

- The equal average of the four channels (the within-family average of three LightGBM recipes, XGBoost, CatBoost with `music`, Extra Trees with interactions) is the strongest procedure: 0.8411 (0.8425, 0.8417, 0.8390 by repeat), 0.0045 above H\* (0.8366) and 0.0096 above the anchor (0.8315). The learned procedures, each evaluated as a whole on held-out groups with its combiner refitted inside every outer training part, score 0.8400 to 0.8410: constrained weights 0.8410 (lambda 0.1), 0.8406 (0.01), 0.8400 (0); the stacker 0.8408 (C 0.01), 0.8405 (0.1), 0.8402 (1). Stronger regularization did as well or better in both kinds, so learning the weights added nothing over equal weights.
- Fixed blends: the equal average of the three strongest channels 0.8388, of the two strongest 0.8362; the best contribution blends with H\* are 0.25 Extra Trees with interactions (0.8394) and 0.25 CatBoost with `music` (0.8389). The within-family average of three LightGBM recipes, 0.8365, is no better than H\* alone: averaging near-identical recipes adds little, while different families do.
- The menu (12 candidates, `configs/phase2/menu.json`): the anchor (0.8315), its three-seed version (0.8323), the standalone leader (LightGBM `p26_lightgbm_415b649d`, 0.8364), the within-family average (0.8365), the best fixed blend (the equal average of all channels, 0.8411), the best constrained-weight procedure (lambda 0.1, 0.8410), the best stacker (C 0.01, 0.8408), then by the rule the other four learned settings and the 0.25 Extra Trees contribution blend.
- Production selection by the frozen rule: the equal average of all channels. Its tie set within 0.001 of the maximum holds it and five learned settings; all have six atomic recipes and one supervised transformation (`date_te3`), and it wins on having no learned combiner. The standalone leader is `p26_lightgbm_415b649d` rather than H\* (0.8366): they are within 0.001 with equal structural counts, and the complexity rank, which orders equal structures by candidate id, decides.
- P2.7's projected cost from the logged run times (C_all 676 s, C_L 620 s): 12,210 s by the plan's formula, 10,837 s by the pilot's multipliers, 8.48 hours for build and empty-cache replay with a 25% margin, above the eight reserved; those times were measured partly beside other fitting jobs, so P2.7 decides the path from its own measurement before any outer score.

Evidence:

`../analysis/p26_menu.py` and its output `../analysis/p26_menu_output.txt`; `../configs/phase2/menu.json`; the P2.6b blend runs and procedure records in `../artifacts/phase2/experiments.jsonl` with their banks (the procedure records of the first manifest, ae4dd90, are superseded: those of the regenerated manifest were recomputed from the same cached meta predictions and are identical in score); the complete replay `../analysis/p26_verify_learned.py`, output `../artifacts/phase2/verify_learned_p26.txt` (every channel atom refitted, every meta-fold fit redone from an empty cache); the execution logs `../artifacts/phase2/p26_evaluate_log.txt` and `p26_evaluate_r2_log.txt`.

Uncertainty:

Exploratory: every procedure was scored on the folds that also chose its ingredients, and the gaps among the top candidates (0.0001 to 0.001) are well inside the repeat-to-repeat spread; the restricted nested evaluation (P2.7, not run) is the declared check of the selection procedure. The public leaderboard plays no part.

Review status:

`reviewed`: R2 review of the P2.6b freeze `APPROVED` at fd48199 in round 2 (`../reviews/2026-10-31-p2.6b-menu.md`); provisional results until they feed the final selection (R1).

Downstream use:

P2.7 and phase 3's two final selections (Sam's decision).

## Finding 029: The Test Population and the Frozen Menu's Scores on Singleton Rows

Question:

What population of tracks does the test file represent, and how do the frozen menu's candidates score on validation rows of that kind, compared with the official all-rows score? Raised by the AUC brainstorm of 2026-10-30/31 (`llm_usage/2026-10-31-auc-brainstorm/`), checked here on the project's own data and banks.

Current conclusion:

- Construction of the test file: its ids run from 2 to 967 with 103 ids missing; 3,864 + 863 + 103 = 4,830 rows and 3,864 / 4,830 = 0.8000 exactly. No test uri occurs in train, and no test row shares its 13 audio values and date string with a train row (2 share the audio values with another date). Within the test file 11 uris are repeated, 22 rows or 2.6% of it; within train 203 uris are repeated (185 twice, 18 three times), 424 rows or 11.0%. These facts fit an 80/20 row split from which every test row whose track occurs in train was removed; the removal step is inferred from the counts, not observed.
- Labels by population in train: rows with a repeated uri are 79.3% High (424 rows), rows with a singleton uri 29.7% (3,440), all rows 35.2%. By the primary groups of Finding 002 (same uri, or identical audio and date), 3,431 of the 3,863 eligible rows are singletons (29.7% High) and 432 are in repeated groups (78.9% High). The test's High rate is therefore expected near 0.30, not 0.35.
- The frozen menu on singleton validation rows (the official 15 folds, each fold scored on its singleton rows only, about 685 per fold): every candidate scores 0.0031 to 0.0043 lower. The production selection `blend_eq_all` goes from 0.8411 to 0.8370 (by repeat 0.8383, 0.8366, 0.8362), the anchor from 0.8315 to 0.8279, the standalone leader from 0.8364 to 0.8331. The ranking is kept: Spearman 0.965 over the 12 candidates, `blend_eq_all` first on both scores, and within the frozen rule's tie set on the official score (six candidates) it also has the highest singleton score. Applied to singleton scores as a diagnostic, the rule's mechanics would widen the tie set to eight candidates and the fewer-atoms key would then pick the two-atom 0.25 Extra Trees contribution blend (0.8363); the equal average leads it by 0.0007 on singleton rows against 0.0017 on all rows.
- The official score exceeds both within-population scores (singleton 0.8370, repeated-group rows 0.8169 for `blend_eq_all`) because repeated-group rows are mostly High and receive higher scores than singleton rows: part of the official score is the separation between the two populations, which the test file lacks.
- The menu's eight atom runs lose by family: LightGBM 0.0034 to 0.0038, XGBoost 0.0051, CatBoost 0.0053, Extra Trees 0.0014. The brainstorm's rescoring of all 576 phase 2 banks (its `rescore_singletons.log`, Spearman 0.993 over the banks) is corroboration from outside the project's engine, not project evidence.

Evidence:

`../analysis/brainstorm_checks.py` and its output `../analysis/brainstorm_checks_output.txt` (the data facts; the 12 candidates and 8 atoms rescored from their logged banks, each bank's hash checked against the menu and its all-rows mean against the menu's before the singleton score is taken; the frozen rule's tie set on both scores). Nothing was fitted.

Uncertainty:

The removal step is an inference from counts that fit it exactly. The singleton population omits the 22 test rows in within-test pairs (2.6%), whose behaviour under the models the brainstorm examined separately (`llm_usage/2026-10-31-auc-brainstorm/scratchpad/me/test_duplicates.log`). Singleton fold AUCs rest on about 685 rows each, and the differences among the top eight candidates on singleton rows (at most 0.001) are inside the repeat spread. The public leaderboard score of 0.85377 on about 40% of the test rows is not evidence about any of this.

Review status:

`draft`, written 2026-10-31 by the primary agent during the pause after P2.6b; no review record names it. It would be reviewed with phase 3's selection (R1) if the singleton score is reported there.

Downstream use:

A diagnostic beside the official metric in phase 3's selection evidence, if Sam decides to report it; the report's description of the test population; no change to the frozen selection rule.
