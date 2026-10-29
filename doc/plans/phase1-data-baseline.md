# Phase 1: data audit and baseline

v0.1, 2026-10-29. Owner: Sam Zhou. Status: started 2026-10-29 on Sam's go-ahead, verbatim: "Do the day one foundation part now." and "Don't forget to consult deeply with Codex."

## Goal

The facts of the data in the ledger, a validation harness that every later model uses, reproducible baselines, and a first Kaggle submission that proves the pipeline end to end (`modeling_strategy.md` section 6, Day 1).

## Exit criteria

1. Findings 001 to 006 reviewed (P1.1).
2. The harness of section 4 below in place, passing its self-checks; the legacy scoping table reproduced through it to four decimals; the nine production baselines (LightGBM, random forest, logistic on F0, F1, F2) in the prediction bank and the experiment log (P1.2).
3. The first submission file reviewed and handed to Sam with its description; report and Colab skeletons in place (P1.3). P1.3's external step, Kaggle accepting the file, is complete when Sam reports the submission and its public score.

## Tasks

Each task is one reviewed unit; a task starts only after the one before it converged. Before each task is built, Codex is consulted on its design (Sam's instruction above); the formal reviewer is always a fresh Codex session.

| Task | Deliverables | Record |
|---|---|---|
| P1.1 | `../../analysis/audit.py` and its output; Findings 001 to 006 in `../findings.md`; this plan; status updates in `../README.md`, `../project_plan.md` and `modeling_strategy.md` | `../../reviews/2026-10-29-p1.1-data-audit.md` |
| P1.2 | The package and scripts of section 4; split, bank and log artifacts; the legacy reproduction; the nine production baselines; the paired comparisons of section 4.7 | `../../reviews/2026-10-29-p1.2-harness.md` |
| P1.3 | The first submission file, its check report, hash and description; the report skeleton; the Colab skeleton; the LLM integration index updated | `../../reviews/2026-10-29-p1.3-first-submission.md` |

## Decisions

- Sam uploads every submission. Sam's choice, verbatim: "I upload them (Recommended)" (described as "I prepare and check each reviewed file and give you its path and description; you upload it on Kaggle and tell me the public score."). A reviewed file ready for Sam and a file accepted by Kaggle are recorded as separate events.
- No package is installed in phase 1; everything uses `../../requirements.txt`.
- The P1.2 and P1.3 designs below follow Codex consultation round C (`../llm_usage/2026-10-29-day1-consultation/`), adopted with the changes noted.

## 4. Design of P1.2

### 4.1 Layout

A small package `songpop/` at the project root (`data.py`, `features.py`, `splits.py`, `models.py`, `evaluate.py`, `metrics.py`, `artifacts.py`) and scripts run as modules from the project root, so nothing is installed and no import path is edited: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.run_baselines --protocol legacy|production`, `... -m scripts.check_foundation`. Computation is separated from writing: every script has a mode that recomputes in memory and writes nothing, for the read-only reviewer. Artifacts live in `artifacts/` (`splits/`, `bank/`, `experiments.jsonl`).

### 4.2 Data

- A permanent `train_row_id` (the row position in `train.csv`, 0 to 3,863) is assigned on loading and kept through every filter.
- Labels are validated to be exactly {"High", "Low"} with none missing, then mapped explicitly: Low 0, High 1 (Finding 001).
- Eligibility: the rows with all 13 audio features present; an assertion checks that this excludes exactly row 425 and nothing else.
- Test IDs are kept as supplied; the sample submission supplies only the schema and the ID order.

### 4.3 Groups and folds

- Groups: union-find over training rows, joining rows with the same nonmissing `uri` and rows with the identical tuple of the 13 audio values (as floats) plus the raw date string; the strict variant also joins identical audio-only tuples. The incomplete row stays a singleton. Each group is labeled by its smallest `train_row_id`. Labels, URLs and test rows play no part. The counts must equal Finding 002 (3,638 and 3,630 groups including the singleton).
- Folds: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)` on the 3,863 eligible rows with their own labels, for split seeds s = 155, 2026, 1029; for the primary and the strict grouping. The incomplete row then gets fold `int.from_bytes(hashlib.sha256(f"incomplete-v1:{s}:425".encode("utf-8")).digest()[:8], "big") % 5` (folds 4, 3 and 0 for split seeds 155, 2026 and 1029), without changing any other assignment, for the later keep-row ablation.
- Saved in `artifacts/splits/production_v1.csv`, one row per training row: `train_row_id, eligible, primary_group, strict_group, primary_fold_155, primary_fold_2026, primary_fold_1029, strict_fold_155, strict_fold_2026, strict_fold_1029`; its hash is recorded. The strict folds are for reporting only, never for selection. The legacy protocol (`uri` groups, split seeds 0, 1, 2) is recomputed by the scoping code path and kept apart.
- Checked per fold and seed: rows and groups on each side, High and Low counts and rate, era shares, mixed-label groups, an empty intersection of training and validation groups, both classes on each side. Class-rate deviations above 0.03 and era-share deviations above 0.05 are reported as warnings; partitions are never regenerated by model results.

### 4.4 Features

Float64 frames, NaN for missing, fixed column order; the audio order is that of `../../analysis/scoping.py`.

| Set | Columns |
|---|---|
| F0 | The 13 audio features |
| F1 | F0 + `year` |
| F2 | F1 + `month` (full and year-month dates), `day`, `day_of_year` (January 1 is 1), `weekday` (Monday 0 to Sunday 6) and `days` (days since 1970-01-01, negative before) for full dates only, and `date_precision` (the string length 4, 7 or 10): 20 columns |
| legacy | The scoping fields: F0 + `year`, `month`, `day`, `date_precision`, `days` with partial dates filled as (year - 1970) x 365.25 + 182 |

Dates are matched against an explicit pattern and full dates parsed with an explicit format (Finding 003). Edge cases checked: 1979-06 (month 6, the rest missing), a year-only date, 1970-01-01 (weekday 3, days 0), a date before 1970, a leap day, an invalid date.

### 4.5 Baseline recipes

| Model | Configuration |
|---|---|
| LightGBM | 600 trees, learning rate 0.02, 15 leaves, `min_child_samples` 20, `subsample` 0.8 with `subsample_freq` 1, `colsample_bytree` 0.8, seed 0; production adds `n_jobs=4`, `deterministic=True`, `force_col_wise=True` |
| Random forest | 500 trees, `min_samples_leaf` 3, seed 0, otherwise the defaults of scikit-learn 1.9.1; missing values handled natively |
| Logistic | `log1p(duration_ms / 1000)` and `log1p(100 x)` for instrumentalness, speechiness, liveness and acousticness; median imputation with medians of the transformed training columns; standardization; `C=1`, `max_iter=2000` |

A new estimator per fold; estimator seed 0 throughout, separate from the split seeds; fixed tree counts, so no early stopping on Day 1. The probability of High is read from the column of class 1 in `classes_`, never from the position of a string label.

### 4.6 Prediction bank and experiment log

- One `.npz` file per run in `artifacts/bank/`, read with `allow_pickle=False`, never overwritten: `train_row_id`, `y`, `group_id`, `split_seeds`, `fold_id` (3 x N), `oof` (3 x N), `test_id`, `test_by_fold` (3 x 5 x 863), `test_mean`, `fold_auc` (3 x 5), and `metadata_json` (recipe and resolved parameters, feature names, input hashes, split-file hash, row population, seeds, deployment rule, Python and package versions, platform, thread settings, git commit and the hashes of the executing source files, command, time, warnings, completed fits). Every eligible row gets exactly one held-out prediction per repeat.
- `artifacts/experiments.jsonl`: one record per run (run and recipe ids, task, protocol, whether selection-eligible, features, model, parameters, hashes, seeds, deployment, row, group and feature counts, fold, repeat and mean fold AUC, pooled out-of-fold AUC per repeat, time, warnings, status, bank path and hash). Paired comparisons and submission events are separate records.

### 4.7 Metrics and comparisons

- Primary: the unweighted mean of the 15 fold AUCs. Also the five-fold mean per repeat and the pooled out-of-fold AUC per repeat. Repeats are never pooled as independent observations.
- The paired cluster bootstrap of `modeling_strategy.md` section 4, step 8, with an exact weighted AUC computed from a cached sort of each fold's predictions, checked against scikit-learn's `roc_auc_score` with `sample_weight`. One multiplicity vector per replicate for both models and all repeats; a replicate with a single-class fold is invalid as a whole. 2,000 replicates, seed 155001. Two banks are paired only if their row ids, labels, groups, seeds and folds agree exactly.
- Day 1 comparisons: F1 minus F0 and F2 minus F1 for each model; random forest minus LightGBM on F2.

### 4.8 Deployment and submissions

- Deployment of the baselines, predeclared: the equal average of the 15 fold models' test predictions. No full-data refit on Day 1.
- The submission writer reads the file back and checks: 863 rows; columns `ID, Popularity_Type` in that order; no index column; unique IDs in the order of the sample and the bank; finite probabilities in [0, 1]; equality with the bank's deployment output; predictions equal within 1e-12 for test rows with identical recipe feature vectors (NaN equal to NaN), which covers the 11 repeated `uri` pairs; a summary of the predictions; the hash of the final bytes.

### 4.9 Self-checks

`scripts/check_foundation.py`, in memory and without pytest: the target map and its rejections; permanent row ids under filtering; groups (`uri`, signatures, a transitive chain, label invariance, permutation invariance, the singleton, strict coarsening, the counts of Finding 002); folds (recomputed equal to saved, each eligible row held out once per repeat, no group split, both classes present, the incomplete row's fold fixed); the date edge cases; fold-local preprocessing unaffected by an extreme held-out value; probability extraction and its rejections; AUC (perfect, reversed, constant, weighted ties and zero weights equal to scikit-learn); the bootstrap (identical predictions give zero difference and a [0, 0] interval, a fixed seed repeats, swapping the models negates the draws); bank round trips and mismatches; the submission validator accepting a valid file and rejecting each of the defects in section 4.8. The legacy table is reproduced to four decimals; a mismatch is investigated, never absorbed by changing the expectation.

## 5. Design of P1.3

- The first submission is the F2 production baseline with the highest mean fold AUC; an exact tie within 1e-12 goes to the order LightGBM, random forest, logistic. Its purpose is to verify the pipeline, not to select a model; a margin below 0.003 establishes nothing.
- Its description, filled from the reviewed bank: model, F2, High = 1, five grouped folds with split seeds 155, 2026, 1029, estimator seed 0, the equal average of 15 fold models, mean fold AUC, provided data only, no pretrained models and nothing fitted on test rows, recipe id, file hash, purpose.
- The report skeleton: a working copy of the course template next to the untouched `../../latex/` files, with the example LLM figures and prose removed, the reviewed facts filled in and explicit placeholders for everything else.
- The Colab skeleton: a notebook that takes an uploaded, versioned code-only zip (`songpop/`, `scripts/`, `requirements.txt`, the selected recipe, the split file, a source manifest) and the three CSVs, checks their hashes, and runs the same code path. Its clean run is a later task.
- `../llm_usage/integration_index.md`: one row per LLM-assisted component (role, model and session evidence, prompt and answer location, the adopted excerpt, where it was implemented, what was changed or rejected, how it was verified, the task).

## Open questions

1. Team members and the Kaggle team name (Sam).

## Status

- 2026-10-29: phase started on Sam's go-ahead; P0.2 committed first (f29267b, 5fc0cef). Codex consultation round C on the Day 1 design. P1.1 in review.
- 2026-10-29: P1.1 approved in two rounds and committed (a369262, 34247f5).
- 2026-10-29: P1.2 built to section 4; on Codex consultation round D (`../llm_usage/2026-10-29-day1-consultation/`) the implementation adds, beyond section 4: validation of the cross-validation inputs before any fit and of every bank on saving and loading; run ids from a hash of the full configuration; parameters described without process addresses; `--verify` checking bank hashes, stable metadata, logged metrics and comparisons; split-file validation; fold tables for both sides and both groupings; the diagnostics of `modeling_strategy.md` section 4, step 7 (group-weighted AUC, AUC per era); submission checks that derive the recipe's test features themselves. Findings 007 and 008. In review.
