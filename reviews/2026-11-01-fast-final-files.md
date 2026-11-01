# Review record: fast-mode final pair

Opened 2026-11-01. Author: Codex. Reviewer: Claude Fable 5.1 xhigh, fresh read-only session.

## Round 1: review request
Unit: The two owner-selected final submission files only. One quick release review, limited to leakage and file correctness. This is the only review round Sam requested.
Commit: c04f3404b3ffcc4b30f8e0401e8ad258f6e51c06
Requirements:
- Sam: "Then one quick Fable review of the two final files only, limited to leakage and file correctness, one round."
- Sam chose "Combined + original ensemble (Recommended)".
- "Folds grouped by track, with the same split seeds (155, 2026, 1029) and the same split code as the saved banks"; "Fit every preprocessing step and encoder inside the training folds only."
- "No external data, no lookups by Spotify identifier, no pretrained models, no code written for this dataset. Do not fit on test.csv features."
- File checks: row count and IDs match test.csv; finite probabilities in [0, 1]; correlation with the current best printed; CV and sha256 logged.
Constraints and decisions:
- The fast-mode section of AGENTS.md overrides the older review/consultation/findings rules. No R2, replay, receipt validators, nested evaluation, search-coverage or method-choice review. Do not reopen P2.9.
- Sam asked to finish quickly. Keep this targeted, about five minutes if possible. No training, tests, searches, package installs or new work. Use read-only source/file inspection. Do not review the merits of small CV gains or propose a larger experiment program. The two CSVs below are the complete review scope; related code and banks are inputs for tracing them.
Files and sources:
- `submissions/2026-11-01_fast_combined.csv`, sha256 fc3d6730db0b96bca34aad4a50484df57590029d3201634319c2450a4810b751; sidecar JSON. CV 0.8428365469227956.
- `submissions/2026-10-31_p2.6b_blend_eq_all.csv`, sha256 d2b5c461c21f03591d72b716f007e93c43900e9f0d048355b4a883a2482ac3cc; sidecar JSON. CV 0.8410616054303375.
- `artifacts/fast/final_selection.json`, `artifacts/fast/submission_specs/combined.json`, `artifacts/fast/final_candidates/combined/result.json` and `predictions.npz`; `submissions/README.md`.
- New combined components: 0.65 original E bank `artifacts/phase2/bank/p2_blend_eq_all_seeds155-2026-1029_375767dd7ee9.npz`; 0.25 `artifacts/fast/lgb_year_counts_seeds012/predictions.npz`; 0.10 `artifacts/fast/mlp_w128_a1.0_e300_standard_seeds012/predictions.npz`. Each new component directory has config/result JSON. Both use estimator seeds [0,1,2].
- `analysis/fast_features.py` (used additions: year-relative audio and year/month counts); `analysis/fast_exploration.py` (data_inputs, model, run_trial; relevant families lightgbm and mlp); `analysis/fast_submission.py` (prepare/check_file). `analysis/fast_blends.py` saves fixed blend arrays.
- Retained implementations: `songpop/data.py`, `songpop/splits.py`, `songpop/phase2/features.py`, `encoders.py`, `models.py` preprocessing; `artifacts/splits/production_v1.csv`; provided `data/train.csv`, `test.csv`, `sample_submission.csv`.
- Original E composition: `configs/phase2/menu.json`, candidate blend_eq_all, and its named member banks. Existing `songpop/phase2/engine.py` and `ensembles.py` show its fitting/averaging path if needed. All model data/predictions and source inputs are tracked. No external inputs besides the configured live interpreter.
Checks already run:
- `analysis.check_fast`: saved-fold/ID/label/group alignment with E and constrained reference banks passed; grouped hierarchical encoder held-out-label control passed with a deliberately leaky full-fit control rejected; query poisoning did not refit feature state. Output `doc/llm_usage/2026-11-01-fast-mode/checks.txt`.
- All 51 new runs completed 15 held-out folds; each used validate_cv_inputs and validate_bank. The selected component configs, all fold scores, warnings, elapsed times and predictions are saved. No model training occurs in prepare.
- prepare/check_file on all three public-check files passed. Each has 863 rows, exactly the test IDs/order, finite probabilities in [0,1]. Selected new file correlation with E is 0.9973259445653754. The two selected CSVs were checked again just before this request, including current full hashes above; original E correlation with itself is 1.
- `artifacts/fast/submission_check_controls.json`: swapped IDs, one missing row, NaN and a probability above one were rejected. Selected new predictions have min 0.003673191237298412 and max 0.9898767295387574.
Questions for the reviewer:
- Do the selected new LightGBM/MLP feature, encoder, preprocessing and averaging paths expose held-out labels, fit on test features or otherwise violate the retained grouped training boundary?
- Do the two CSVs correspond to the declared saved predictions and High-class orientation, with correct IDs, dimensions, finite bounded probabilities, hashes and reported CV?
Report only findings within leakage/file correctness. Keep the verdict concise. Treat cosmetic/performance/optimization suggestions as out of scope.
Known limitations: Scores are exploratory on repeatedly used grouped folds, not new nested estimates. MLP fits use declared training budgets with early_stopping=False; convergence warnings are retained. No cross-platform or byte-exact replay is claimed. Neither file was uploaded in this session. These limitations do not expand the requested review scope.

## Round 1: verdict
Reviewer: Claude Fable 5.1 (claude-fable-5-1), Claude Code, read-only checkout
Revision reviewed: c04f3404b3ffcc4b30f8e0401e8ad258f6e51c06
What I checked and how:
- `git rev-parse HEAD` = c04f340. `sha256sum` of both CSVs: fc3d6730… and d2b5c461…, equal to the request, `final_selection.json`, both sidecars and the README log.
- With the review interpreter (numpy, pandas, sklearn only; nothing written): both CSVs have header `ID,Popularity_Type`, 863 rows, IDs identical to `data/test.csv` and to the bank `test_id` in order, unique, all values finite, min/max 0.003673/0.989877 (new) and 0.003209/0.990721 (E); no NaN text, trailing newline present.
- Recomputed the combined file from the three declared component banks with weights 0.65/0.25/0.10 (sum 1.0): test mean equals the CSV values with max abs difference 0.0, and equals `final_candidates/combined/predictions.npz` (sha df16de0f…, as recorded). Component bank sha256s equal the sidecar (f7aa5881…, b8512303…, e7eba95f…). All three components share identical `train_row_id`, `y`, `group_id`, `split_seeds` [155, 2026, 1029], `fold_id`, `test_id`.
- Recomputed grouped fold AUC from the blended OOF arrays: 0.8428365469227956, per seed 0.843593/0.843488/0.841429, identical to the reported CV; E OOF gives 0.8410616054303375, identical to its sidecar. Correlation between the two CSVs 0.9973259445653754, as reported.
- E CSV equals the E bank `test_mean` exactly (max abs 0.0); E bank `test_mean` and `oof` equal the weighted average of its six member banks (`menu.json` weights, 1/12 ×3 LightGBM, 0.25 XGBoost, CatBoost, Extra Trees) to 3e-16; all six member banks present.
- Orientation: bank `y` equals (train.csv `Popularity_Type` == "High") for all 3863 eligible rows (1359 High); `songpop/data.py` TARGET_MAP High=1; `fast_exploration.run_trial` asserts `classes_ == [0, 1]` and takes column 1.
- Splits: bank `group_id` equals `production_v1.csv` `primary_group` and `fold_id` equals `primary_fold_{155,2026,1029}` on the eligible rows (row 425 excluded, as in `data.py`); no URI spans two groups (3642 URIs); no group spans two folds in any repeat; 15 train/validation group intersections empty. No test URI occurs in train.
- Leakage trace by source inspection: `run_trial` fits `FastFeatures` on `rows.iloc[tr]` with `y[tr]`, `groups[tr]` and transforms validation and test with that fitted state; raw frames hold only `AUDIO + [DATE]` (`RAW_COLUMNS`, enforced in `FeaturePipeline._assemble`). `year_relative` means and smoothing prior, `date_counts` distinct-group counts (own group removed on fitting rows only), `DateCount` and the base F2 date fields (days since 1970 epoch, stateless) use training rows only; `date_te3` is sklearn `TargetEncoder` cross-fitted on explicit group-disjoint inner splits checked by `check_inner_splits`, with full-fit transform for held-out and test rows. MLP: `log_standard` imputer/scaler/one-hot inside the sklearn Pipeline, fitted on the training fold only, `early_stopping=False`; LightGBM fitted with training X, y only, no eval set. `fast_submission.prepare` fits nothing and reads no test features. `HierarchicalDate` is not used by either selected component.
- Component records: both result.json files are `complete`, 45 fit entries = 15 folds × seeds [0, 1, 2], configs equal `config.json`, prediction sha256 equal the files, recorded data hashes, split sha256 and the sha256 of `analysis/fast_exploration.py` and `analysis/fast_features.py` equal the current files at this commit.
Findings:
  F1 [non-blocking] `submissions/2026-11-01_fast_combined.json` line 4 and `artifacts/fast/final_candidates/combined/result.json` line 4. The status field still reads "prepared for Sam; not uploaded or final-selected" while `artifacts/fast/final_selection.json` records the owner's selection of this file; the CSV bytes and hash are unaffected. Update the status strings (or note in `submissions/README.md` that `final_selection.json` is the authoritative selection record) when the final-selection event is logged.
Notes: I did not retrain or replay either component, did not rerun `analysis.check_fast`, and did not read `checks.txt`; the leakage conclusion rests on source inspection of the fitting paths plus the saved-array identity checks above. The E file's original R1 approval and prior upload are taken from the README log, not re-reviewed beyond the composition and byte checks. The new sidecar carries no Kaggle description string; fast mode does not require one. Nothing out of scope (CV merits, method choice, P2.9) was assessed.
VERDICT: APPROVED
Reviewer run: Claude Code 2.1.316, model claude-fable-5-1, effort xhigh, plan mode (read-only), thread 4ab47fe3-f9f9-41be-b788-da40aca0bce3, commit c04f3404b3ffcc4b30f8e0401e8ad258f6e51c06, review copy checked before and after. Review environment (outside the repository, use read-only): Python /exp/mu2e/app/users/wzhou2/projects/miniproject_1/.venv/bin/python (Python 3.12.16); Log: /home/wzhou2/.agent-review/miniproject_1-902d02a9/logs/20261101T044340-r1-claude.9lNX

## Author response and closeout

2026-11-01, 04:49 UTC. F1 resolved using the reviewer's suggested log clarification:
`submissions/README.md` now names `artifacts/fast/final_selection.json` as the
authoritative selection record and explains that the preparation sidecar/result
preserve their earlier event status. The log records both owner selections and
the review approval. No second round was requested or run.

Material edits after review: none. CSVs, component predictions, model code and
reported scientific results remain unchanged from c04f340. Only review, status
and LLM-use records changed. Development is closed; Sam uploads and selects the
final Kaggle entries. Report and Colab work follows.
