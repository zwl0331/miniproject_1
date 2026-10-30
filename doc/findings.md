# Findings

This file is the evidence ledger for the Miniproject 1 model. New factual conclusions about the data and the models land here before code, Kaggle submissions, the Colab notebook, or the report may depend on them. The convention is ported from the Positron Conversion DIO project (its Finding 000), which borrowed it from `DIO_Shape_Systematic`.

## Review Status Labels

- `draft`: written for discussion, not yet accepted.
- `reviewed`: named by a converged review record under `../reviews/` (`review_policy.md` section 4.1); the finding cites the record.
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

`draft`, in the P2.2 review (`../reviews/2026-10-29-p2.2-searches.md`), completion checkpoint.

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

`draft`, in the P2.2 review (`../reviews/2026-10-29-p2.2-searches.md`), completion checkpoint.

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

`draft`, in the P2.2 review (`../reviews/2026-10-29-p2.2-searches.md`), completion checkpoint.

Downstream use:

Later rankings of P2.2 forest scores (the comparator of plan section 6.5, the pool rule of section 7.1) take the logged scores with their own tie rules (plan section 15.7). The next task that changes the phase 2 code makes forest predictions deterministic (prediction with one thread, or an equivalent fixed summation order) and re-checks the other families' exact reproduction, before any random forest or Extra Trees configuration enters a frozen menu; the finalists of P2.6b are refitted under their final seed policy anyway.
