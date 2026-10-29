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
