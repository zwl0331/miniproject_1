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

`draft`, in the P2.1 review (`../reviews/2026-10-29-p2.1-infrastructure.md`); it becomes `reviewed` when that review converges, and this ledger revision is committed only then.

Downstream use:

The anchor of phase 2: the comparator of the searches and feature blocks, and of the nested evaluation's fallback (`plans/phase2-models-features.md` sections 5.4 and 8).
