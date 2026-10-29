Claude, the fitting engine looks sound on the paths I exercised. The main problems are in artifact consistency, verification, and submission auditing. **The current files do not reproduce the reported clean check results.** My additional probes below are unreviewed consultation evidence.

1. **Must-fix: regenerate the stale anchor evidence after the code is final.**  
   [recipes.py:14](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/recipes.py:14), `artifacts/phase2/bank/`, and [findings.md:314](/Users/zwl0331/Desktop/Mini_project_1/doc/findings.md:314).

   The saved anchor recipe has no `drop` field. The current loader requires it. Consequently, `scripts.run_experiments --verify-all` fails before recomputation. Its manifest also differs from the current `features.py`, `recipes.py`, and `searches.py`.

   **Fix:** after the remaining fixes, regenerate the unapproved anchor bank, log, run output, and check output together. Update Finding 009 with the resulting run ID. Rerun both phase 2 commands against that exact revision. Do not carry forward the saved “21 passed” output: my current run returned **19 passed, 2 failed, 1 skipped, 4 staged**. The two failures were the stale bank and the temporary-directory check discussed below.

2. **Must-fix: check equivalence after all estimator preprocessing.**  
   [submission.py:43](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/submission.py:43), [models.py:47](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/models.py:47).

   `member_feature_keys()` stops at `FeaturePipeline.transform()`. It excludes imputation, one-hot encoding, splines, and scaling inside the estimator. Row-wise preprocessing preserves existing equalities, but can also create new equalities.

   I demonstrated this using the actual test rows and a valid month-only recipe. Test positions **15 and 0**, with months **NaN and 7**, became identical after fitted imputation. I added **0.01** to predictions for the missing-month group, updated `test_mean`, and `validate_candidate()` accepted the corrupted submission.

   **Fix:** reconstruct both stages:

   ```python
   X_train = fp.fit_transform(training_rows, y_train, groups_train, context)
   X_test = fp.transform(test_rows)
   prep = make_preprocessing(recipe["preprocessing"], X_train.columns)
   Z_test = X_test if prep is None else prep.fit(X_train, y_train).transform(X_test)
   ```

   Build duplicate keys from `Z_test`. Crucially, fit preprocessing on the returned cross-fitted `X_train`, not on `fp.transform(training_rows)`. Add the missing-month collision as a regression check. Correct the consultation README’s justification for checking only preprocessed feature-block outputs.

3. **Must-fix: preserve the promised checks for individual estimator seeds.**  
   [engine.py:31](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/engine.py:31), [submission.py:111](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/submission.py:111).

   A multi-seed bank retains only the seed average at each fold. Submission validation therefore checks a recipe-fold average, not every fitted estimator as §4.7 promises. Opposite errors in two estimator seeds could cancel, just as they could across blend members.

   **Fix:** keeping one bank per recipe is fine. Check each estimator’s predictions against its actual transformed inputs before averaging. Retain per-seed test predictions in a small audit extension or companion artifact if submission-time checks are to recheck every original member. Otherwise, explicitly distinguish the recorded fit-time per-estimator checks from submission-time checks of their average. Do not describe the current aggregate check as individual-member verification.

4. **Must-fix: enforce consistency between the bank’s duplicated provenance fields.**  
   [banks.py:72](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/banks.py:72).

   `validate_v2_metadata()` validates several hashes independently without connecting them. In memory, it accepted:

   - A changed recipe with updated recipe hash/ID, while `config.recipe_sha256` still described the old recipe.
   - A false `config.fold_ids_sha256`, with a correspondingly recomputed config hash.
   - A changed source-manifest digest inconsistent with the config.

   **Fix:** validate the relationships, not just each object separately:

   ```text
   config.recipe_sha256 == recipe_sha256
   config.estimator_seeds == estimator_seeds
   config.split_seeds == bank.split_seeds
   config.fold_ids_sha256 == hash(actual fold_id)
   config.evaluation_population == evaluation_population
   config.source_manifest_sha256 == recomputed manifest digest
   config.parents == parents
   ```

   Also check canonical, unique production split seeds; stage consistency; feature names against the recipe; and parent uniqueness, finite positive weights, and weight sum. Add separate corruption tests for these relationships.

5. **Must-fix: validate a candidate against the actual production protocol and source revision.**  
   [prepare_candidate.py:44](/Users/zwl0331/Desktop/Mini_project_1/scripts/prepare_candidate.py:44), [submission.py:75](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/submission.py:75).

   Candidate preparation checks test IDs and three seed labels, but does not establish that each leaf bank contains the production evaluation population, labels, groups, and saved fold assignments. It also rebuilds feature pipelines using current code without checking that this is the code that produced the bank.

   **Fix:** before rebuilding any member inputs, require exact equality with `Context` for training row IDs, labels, groups, test IDs, data hashes, and the official fold matrix selected by seed value. Check the source and dependency versions needed for reconstruction. On mismatch, stop and identify the replay revision. For blends, compare the rebuilt OOF and per-fold test arrays as well as `test_mean`.

6. **Must-fix: verify the unrounded scores that later decisions will use.**  
   [runs.py:254](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/runs.py:254).

   Verification compares rounded metrics but ignores `mean_fold_auc_unrounded` and `repeat_mean_auc_unrounded`. I changed both to **0.123** in a mocked log record; `verify_run()` returned an empty problem list.

   **Fix:** compare both unrounded fields to freshly computed bank values with `rtol=0, atol=1e-12`. Check stable record fields such as recipe identity, selection eligibility, seeds, config hash, and completed-fit count against bank metadata too. Selection code should still calculate its scores directly from validated bank arrays.

7. **Must-fix: make `--compare --seeds` perform the requested matched comparison.**  
   [run_experiments.py:87](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:87) and [run_experiments.py:156](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:156).

   The CLI accepts `--seeds` in comparison mode but ignores it. My in-memory CLI probe comparing a three-repeat leader with a seed-155 challenger using `--seeds 155` failed with `banks differ in split_seeds`.

   **Fix:** give `compare()` an explicit seed-subset argument. Apply `select_seeds()` to both banks before bootstrapping. Record the canonical subset and both parent hashes in the comparison record; verification must reconstruct that same comparison.

   Also make comparison recording idempotent. Currently, repeating the same comparison appends a second record, after which verification fails because it requires exactly one. Use an identity based on parent hashes, selected seeds, and bootstrap settings.

8. **Must-fix: deduplicate searches after constructing canonical recipes.**  
   [searches.py:49](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/searches.py:49).

   Deduplication uses the pre-expansion parameter dictionary. Different dictionaries can produce the same recipe. I tested a spline anchor with `C=1` and default `n_knots=4`, followed by a draw explicitly setting `preprocessing.n_knots=4`. Both were emitted, with identical recipe hashes.

   **Fix:** construct and validate each candidate recipe first, then deduplicate by `recipe_hash()`, which already excludes the human name. Count only distinct emitted recipes toward the draw budget. Reject duplicate canonical anchors. Add this exact implicit-versus-explicit preprocessing example to the self-check.

9. **Must-fix: make the search self-check genuinely read-only.**  
   [check_phase2.py:512](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:512).

   The script says everything runs in memory, but this check creates a `TemporaryDirectory` and writes files. It failed in the current read-only sandbox.

   **Fix:** replace temporary filesystem writes with an in-memory filesystem mock. Exercise the real `save_expansion()` and `run_search()` ordering: the spy must observe the complete saved JSON before the first fit, and a failure must append its failure record before propagation. Patching `save_expansion()` to return success without testing serialization would weaken the check.

10. **Must-fix: do not let the optional-family placeholder become a false PASS.**  
    [check_phase2.py:363](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:363).

    Once both packages are installed, this function returns successfully without importing a model, fitting anything, predicting, or checking file writes. If only CatBoost is installed, EBM’s absence skips CatBoost’s check too.

    **Fix:** split the checks by route. Until their bodies exist, report them as **STAGED**, regardless of installation status. In P2.3, replace each with its actual F2-with-NaNs fit, class-orientation and finite-probability assertions, repeat/reload checks, and file-write audit. An installed package must never turn an empty test into PASS.

11. **Must-fix: compare the installation’s after-inventory with its saved before-inventory.**  
    [install_locks.py:156](/Users/zwl0331/Desktop/Mini_project_1/scripts/install_locks.py:156).

    `check-after` checks protected packages and the current route, but allows previously installed additions to disappear. My mocked sequence passed both checks when CatBoost, graphviz, and plotly existed before EBM installation and all disappeared afterward.

    **Fix:** save the accepted before-inventory and its hash. Require:

    ```python
    expected_after = {**before_inventory, **current_route_versions}
    assert after_inventory == expected_after
    ```

    Save the after-inventory too, as §4.9 requires. Add this removal test. The current locks themselves passed my closure, version, hash-reference, and wheel-tag checks.

12. **Must-fix: record the inner-split evidence required by §4.4.**  
    [encoders.py:86](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/encoders.py:86), [engine.py:98](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/engine.py:98).

    Target-encoding splits exist transiently as `splits_`, but their hashes never enter the bank. The plan explicitly promises inner-split hashes where used.

    **Fix:** return a fit receipt containing the outer fit coordinate, fitting-row hash, derived inner seed, and ordered hashes of inner train/validation **permanent row IDs**. Collect those receipts in bank metadata and compare them on replay. Use an empty list for recipes without inner splits.

    Separately, validate supplied explicit splits as well as generated ones. `DateTargetEncoder.fit_transform(..., splits=[(all_rows, all_rows)])` currently succeeds. Reject overlap, out-of-range or repeated indices, split groups, incomplete coverage, and fitting portions lacking a class. Both classes in the inner validation portion are unnecessary for pure target encoding; make the documentation precise about that distinction.

13. **Should-fix: extend verification to the promised search and submission evidence.**  
    [run_experiments.py:159](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:159), [prepare_candidate.py:62](/Users/zwl0331/Desktop/Mini_project_1/scripts/prepare_candidate.py:62).

    `--verify-all` verifies runs and comparisons only. It ignores saved search-list hashes and expansions. `prepare_candidate.py` writes a sidecar but never appends the submission record promised by §4.5.

    **Fix:** verify each search file’s hash, regenerate its expansion in memory, and check ordered recipe hashes and recorded outcomes. Append an immutable submission record containing manifest, bank, CSV, and sidecar references and hashes. Verify those records without appending anything. Selection records can remain explicitly staged to P2.6.

14. **Should-fix: finish the source/dependency replay guards.**  
    [banks.py:29](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/banks.py:29), [runs.py:177](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/runs.py:177), [runs.py:238](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/runs.py:238).

    Three weaknesses remain:

    - The final source check rehashes only the initial file list. A project module imported later during fitting could be omitted.
    - Blend identity omits the Python/package versions used to average predictions and compute AUC.
    - Historical banks are validated with the current recipe schema before source mismatch handling. Schema evolution can therefore produce a loader traceback instead of the promised replay explanation, as the current anchor demonstrates.

    **Fix:** resolve project implementation imports before the initial snapshot and compare the complete loaded-source manifest again afterward. Add relevant runtime versions to blend identity. Read a bank’s envelope and source manifest before applying current recipe semantics, so incompatible historical sources produce a clear nonzero replay diagnostic.

15. **Should-fix: add the missing substantive regression checks.**  
    [check_phase2.py:370](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:370).

    Add these concrete cases:

    | Check | Required assertion |
    |---|---|
    | Estimator-seed averaging | `[0,1,2]` equals the arithmetic mean of three otherwise identical single-seed runs |
    | Recipe weighting | A blend of a three-seed recipe and a one-seed recipe retains the declared recipe weights |
    | Explicit fold count | A valid three-fold bank round-trips with `n_splits=3`; merely rejecting a five-fold bank as three-fold is insufficient |
    | Dependency mismatch | Changed relevant version causes verification failure |
    | Submission writer | Existing-file and corrupted-read-back cases fail through the phase 2 writer |
    | Metadata consistency | Each contradiction in item 4 fails independently |

    My small LightGBM probe found **zero difference** between internal three-seed averaging and averaging separate seed runs. That supports the implementation, but the committed checks currently do not protect it. Also specify `rtol=0` consistently; some blend assertions currently retain NumPy’s default relative tolerance.

16. **Should-fix: reconcile the amendment’s exploration limits with its frozen-menu limits.**  
    [phase2-models-features.md:185](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:185), sections 6.5, 7, and 7.1.

    The numerical budgets remain bounded, but their allocation is ambiguous.

    **Add these rules:**

    > Cross-family comparators are chosen only among complete three-repeat runs, using unrounded mean fold AUC and a deterministic tie-break. The RF and CatBoost comparators consume the two tuned-recipe slots in the eight-run feature budget. If both are used, remaining confirmation runs use those same two recipes.

    > Contribution screening may evaluate the three declared challenger slots. The “at most one other recipe” restriction applies to admission to the frozen menu, not to contribution screening. Every admitted member has complete three-repeat confirmation. The original anchor counts toward the five-base-recipe limit, and the final predictor menu remains capped at eight.

    P2.6 should record which exploratory candidates were omitted to meet those caps. Skip a contribution blend when challenger and leader are the same exact recipe. This resolves the tension without expanding the experiment budget.

17. **Should-fix: correct the deployment description for weighted blends.**  
    [prepare_candidate.py:19](/Users/zwl0331/Desktop/Mini_project_1/scripts/prepare_candidate.py:19).

    The constant description says deployment is the equal average of fold-model predictions. That is incomplete for a 75/25 blend, particularly when members have different estimator-seed counts.

    **Replace it with:**

    > Within each base recipe, probabilities are averaged equally over estimator seeds, folds, and split repeats. The candidate then combines base-recipe averages using its recorded blend weights.

    Keep the actual weights in the description and sidecar.

18. **Should-fix: correct premature status claims and document the small interface changes.**  
    [findings.md:328](/Users/zwl0331/Desktop/Mini_project_1/doc/findings.md:328), [phase2-models-features.md:286](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:286), §13.

    Finding 009 currently says `reviewed` by a review record that does not exist. The phase-plan amendment section still says the current strategy is draft until P2.0 converges, although P2.0 has converged.

    **Fix:** mark Finding 009 as draft pending P2.1 convergence. Make §12 historical and identify §13 as the pending amendment. Document `drop` in the recipe contract, seed-restricted comparisons in §4.6, and the actual selection-manifest schema in §4.7. Complete the consultation archive and integration-index entry before the review fingerprint. Pending paths are acceptable as planned paths, but should not be cited as existing approval evidence.

19. **Optional: remove the unnecessary single-date-per-group restriction.**  
    [encoders.py:30](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/encoders.py:30).

    Counting distinct `(date, group)` pairs and subtracting one works even when a group spans dates. The extra assertion prevents reuse under the stricter audio-only grouping planned for phase 4.

    **Fix:** remove that restriction and replace its rejection test with a positive test for a multi-date group. Keep the count-before-subtraction assertion. This does not change the current primary-group results.

What I checked and found correct:

- Fresh anchor recomputation matched **all ten phase 1 arrays exactly**, including AUC **0.8314739289455241**, and produced identical P1.3 CSV bytes.
- Seed 2026 alone used the saved second repeat. Engine row slicing and prediction placement were consistent.
- The exercised target encoding, date count, era means, preprocessing, and training weights remained confined to their fitting populations. The label-isolation and preprocessing-isolation tests had meaningful positive controls.
- Model family, feature blocks, and preprocessing can vary independently. I found no screening or promotion rule in the fitting engine.
- Probability blending and seed-subset blending work at the library level. Internal estimator-seed averaging matched separate fits in my probe.
- Phase 1 verification passed: **15/15 legacy values**, all production banks and comparisons, **19/19 foundation checks**, and P1.3 submission verification.
- The installation reports and locks agree. The protected inventory still contains the same **50 distributions**. The routes add **3, 1, and 11 distributions**, with no protected-version changes or missing active dependencies in the saved metadata. Runtime compatibility remains untested.
- The kNN replacement preserves the 104-configuration cap. The feature and contribution-blend amendments preserve their numerical caps. Staging selection, nested evaluation, full refits, and TabM is explicitly recorded.
- I found no em-dash or en-dash punctuation in the inspected implementation and changed project documents. I modified no files and ran no reviewer.