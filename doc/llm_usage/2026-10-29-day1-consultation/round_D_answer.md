Claude: I found no numerical defect in the current baseline results. The main problems are missing validation at reusable interfaces, incomplete provenance verification, and self-checks that do not exercise the failure they claim to catch. These should be fixed before the formal review.

1. **Must-fix — Remove the real training CSV from the writer test.**  
   [scripts/check_foundation.py:230](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_foundation.py:230)

   The immutability test calls `save_bank(..., ROOT / "data/train.csv")`. The current guards protect that file, but an overwrite regression could make the test itself destructive.

   **Exact fix:** use standard-library `unittest.mock` with a fictitious path. Test that an existing path raises `FileExistsError` without calling `open` or `mkdir`. Separately simulate a race where `exists()` returns false and exclusive `open(..., "xb")` raises `FileExistsError`. No self-check should pass a competition data file to a writer.

2. **Must-fix — Enforce the CV contract before fitting anything.**  
   [songpop/evaluate.py:29](/Users/zwl0331/Desktop/Mini_project_1/songpop/evaluate.py:29)

   `run_cv` accepts malformed inputs that can produce a plausible but invalid bank. My negative probes succeeded with every row assigned to the same group while that group crossed folds, with 19 row IDs for 20 observations, and with zero split seeds for one repeat. The production caller currently supplies correct inputs; the reusable harness does not enforce them.

   **Exact fix:** before the fitting loop, require:

   - `len(X) == len(y) == len(row_ids) == len(groups) == fold_id.shape[1]`.
   - `len(seeds) == fold_id.shape[0]`, with at least one repeat.
   - Unique row IDs and test IDs; `len(test_id) == len(X_test)`.
   - Binary labels; integral fold IDs in `0..n_splits-1`; every expected fold present.
   - Both classes in each training and validation partition.
   - Every group assigned to exactly one fold per repeat.
   - Feature frames retain original training row IDs as their index, with `X.index` equal to `row_ids`.

   Add negative tests that assert rejection occurs **before the estimator factory or `fit` is called**.

3. **Must-fix — Validate banks, not just their serialization.**  
   [songpop/artifacts.py:63](/Users/zwl0331/Desktop/Mini_project_1/songpop/artifacts.py:63), [songpop/evaluate.py:11](/Users/zwl0331/Desktop/Mini_project_1/songpop/evaluate.py:11)

   `load_bank` accepts an invalid schema and internally inconsistent arrays. I changed a bank’s schema to `"INVALID"` and it loaded successfully. I also inverted `test_mean`, leaving `test_by_fold` unchanged; a matching submission was accepted.

   **Exact fix:** add shared array validation and production-metadata validation. Check dimensions, identity-vector lengths, fold/group consistency, finite probabilities in `[0,1]`, and:

   ```python
   np.allclose(
       bank.test_mean,
       bank.test_by_fold.mean(axis=(0, 1)),
       rtol=0,
       atol=1e-12,
   )
   ```

   Recompute each `fold_auc` from its OOF predictions and labels. Require the supported schema and consistency between metadata seeds/features/deployment and the arrays. Apply validation when saving, loading, comparing, and preparing a submission. Transient CV banks can receive metadata validation after their metadata is populated.

4. **Must-fix — Bind submission validation to the actual recipe features.**  
   [songpop/artifacts.py:106](/Users/zwl0331/Desktop/Mini_project_1/songpop/artifacts.py:106), [scripts/check_foundation.py:266](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_foundation.py:266)

   The duplicate check trusts arbitrary `X_test`. Passing an empty DataFrame bypasses it and returns a successful validation with zero identical-feature groups. Counting 11 repeated URI pairs does not prove those pairs were checked.

   **Exact fix:** derive the expected features internally from `data.test` and the bank’s feature recipe, or compare the supplied frame against that independently derived frame, including column order, row order, values, and NaNs. Assert that every repeated URI pair belongs to an identical-feature group for these baseline recipes.

   Repair the negative self-check too: select indices from **one actual duplicate group**, alter one prediction while keeping it in range, and keep `test_by_fold` and `test_mean` mutually consistent. Assert the specific duplicate-consistency error. Otherwise the new bank validator will reject the fixture before the duplicate check is exercised.

5. **Must-fix — Make production `--verify` verify provenance and logged results.**  
   [songpop/artifacts.py:72](/Users/zwl0331/Desktop/Mini_project_1/songpop/artifacts.py:72), [scripts/run_baselines.py:187](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_baselines.py:187), [scripts/run_baselines.py:214](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_baselines.py:214)

   `compare_banks` ignores metadata completely. Changing a schema or data hash produces no reported difference. Production verification also recomputes bootstrap results without comparing them with the saved comparison records.

   **Exact fix:** verification must check:

   - Saved bank bytes against the SHA-256 recorded in the experiment log.
   - Stable metadata: recipe, parameters, features, data hashes, source manifest, split hash, seeds, deployment, and package versions.
   - Logged metrics against recomputed metrics, respecting the log’s declared rounding.
   - All seven saved comparison records against their recomputed results, including model IDs, bootstrap seed, counts, observed difference, interval, and positive-draw share.

   Exclude timestamps, elapsed time, and the invocation’s `--verify` flag from semantic equality. A later Git commit changing `HEAD` should not invalidate an unchanged source manifest. Report platform differences explicitly rather than silently ignoring them.

6. **Must-fix — Record stable parameters and an executable reproduction command.**  
   [songpop/models.py:43](/Users/zwl0331/Desktop/Mini_project_1/songpop/models.py:43), [scripts/run_baselines.py:183](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_baselines.py:183)

   `repr` inserts process-specific addresses into logistic parameters, including `log__func` and `steps`. These cannot be compared reproducibly. The recorded command is also wrong: it becomes `python /.../scripts/run_baselines.py ...`. Running that direct-script form from the project root failed with `ModuleNotFoundError: No module named 'songpop'`.

   **Exact fix:** serialize typed JSON parameters recursively. Represent callables by qualified names such as `"songpop.models:log_transform"` and estimators by class plus parameters. Use an explicit representation for NaN rather than nonstandard JSON.

   Store the invocation as an argument list:

   ```python
   [sys.executable, "-m", "scripts.run_baselines", *sys.argv[1:]]
   ```

   Record its working directory and relevant environment variables. Keep the source manifest as the evidence for callable implementations.

7. **Should-fix — Make run identity cover the full computational recipe.**  
   [scripts/run_baselines.py:169](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_baselines.py:169), [songpop/artifacts.py:33](/Users/zwl0331/Desktop/Mini_project_1/songpop/artifacts.py:33)

   Run IDs currently depend only on recipe name and source hash. Different data, splits, or installed numerical dependencies can therefore claim the same immutable path.

   **Exact fix:** hash a canonical configuration containing source manifest, input hashes, split hash, feature definition, typed resolved parameters, seeds, deployment rule, and relevant package versions. Exclude timestamps and elapsed time.

   Also record the `requirements.txt` hash and installed `scipy`, `joblib`, and `threadpoolctl` versions. These are relevant dependencies already present; recording them requires no installation. BLAS/OpenMP runtime information from `threadpoolctl.threadpool_info()` would be useful provenance.

8. **Should-fix — Bring legacy experiment records up to the declared schema.**  
   [scripts/run_baselines.py:121](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_baselines.py:121)

   Legacy records omit fields required by phase-plan §4.6, including recipe ID, resolved parameters, feature count, estimator seeds, deployment description, and execution status. They also do not fingerprint the expected scoping-output file.

   **Exact fix:** use a common run-record builder for both protocols. Keep legacy `selection_eligible=False` and `bank_path=None`, but record its actual configuration and status. For wrappers without `get_params`, supply an explicit configuration descriptor. Hash `analysis/scoping_output.txt` as the reproduction reference. Never regenerate that reference from current results.

9. **Should-fix — Replace the vacuous preprocessing-independence test.**  
   [scripts/check_foundation.py:144](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_foundation.py:144), [scripts/check_foundation.py:121](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_foundation.py:121)

   The preprocessing test changes held-out rows and then fits twice on identical training slices. It would still pass if `run_cv` later leaked validation rows into fitting. Likewise, recomputing the same `assign_folds` call does not test that excluding row 425 leaves eligible assignments unchanged.

   **Exact fix:** capture estimators created by `run_cv` using a recording factory. Perturb a validation row, rerun, and compare preprocessing statistics for **the fold holding out that row**. Other folds may legitimately change. Also compare imputer statistics with `nanmedian(log_transform(X_train))` and scaler statistics with the resulting imputed training matrix.

   For the incomplete-row test, physically remove row 425, recompute eligible folds, and compare assignments by permanent row ID. Separately change its label and confirm eligible assignments remain unchanged.

10. **Should-fix — Use valid bootstrap fixtures and test shared multiplicities directly.**  
    [scripts/check_foundation.py:198](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_foundation.py:198), [scripts/check_foundation.py:207](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_foundation.py:207)

    `_bank` assigns folds independently by row while grouping adjacent pairs. Its groups therefore cross folds. It also fills `fold_auc` with `0.5` regardless of OOF scores. These fixtures are invalid banks.

    Identical predictions, repeatability, and sign reversal do not establish that multiplicities are shared across repeats. An implementation resampling separately per repeat could pass those checks.

    **Exact fix:** construct group-safe synthetic banks with both classes in every fold and calculate their actual fold AUCs. Add a small independent reference bootstrap using `roc_auc_score(..., sample_weight=...)`, with one global multiplicity vector per draw. Compare individual draws within `1e-12`.

    Include a fixture producing invalid replicates and verify whole-replicate rejection. Test mismatches in each pairing field, not only `fold_id`.

11. **Should-fix — Validate the complete probability matrix before extracting High.**  
    [songpop/models.py:49](/Users/zwl0331/Desktop/Mini_project_1/songpop/models.py:49)

    A fake estimator with classes `[0,1]` and rows `[NaN, 0.3, 0.7]` is accepted: the code extracts `0.3` and ignores the malformed matrix.

    **Exact fix:** require `predict_proba` to return shape `(len(X), 2)`, with every entry finite and in `[0,1]`, and row sums equal to one within `1e-12`. Then extract the column located through `classes_`. Raise `ValueError` consistently for one-dimensional or wrong-column outputs.

    Extend the self-check to cover those cases, negative probabilities, nonfinite values in the other class, and invalid row sums.

12. **Should-fix — Reject malformed partial dates consistently.**  
    [songpop/features.py:13](/Users/zwl0331/Desktop/Mini_project_1/songpop/features.py:13), [songpop/features.py:20](/Users/zwl0331/Desktop/Mini_project_1/songpop/features.py:20)

    `parse_date("0000")` and `"0000-01"` succeed. A trailing newline also succeeds because `$` can match before it: `"2020\n"` produces precision 5.

    **Exact fix:** use an ASCII-digit pattern with `fullmatch`, then validate `1 <= year <= 9999` even for partial dates. Preserve the existing observed-component semantics. Add those four rejection cases. This does not change features for the current audited data.

13. **Should-fix — Handle an unusable bootstrap explicitly.**  
    [songpop/metrics.py:54](/Users/zwl0331/Desktop/Mini_project_1/songpop/metrics.py:54), [songpop/artifacts.py:88](/Users/zwl0331/Desktop/Mini_project_1/songpop/artifacts.py:88)

    `n_boot=0` returns a result containing NaNs rather than rejecting the request. An all-invalid run similarly looks structurally like an ordinary comparison. `json.dumps` then permits nonstandard `NaN` values.

    **Exact fix:** require a positive integer `n_boot`. If no replicates are valid, raise a clear error or return an explicitly failed comparison with null interval fields. Such a result must never enter a promotion decision. Serialize metadata and logs with `allow_nan=False`.

    Keep the current whole-replicate rejection rule. All seven current comparisons had 2,000 valid draws.

14. **Should-fix — Validate split-file values before coercion and finish the fold report.**  
    [songpop/splits.py:101](/Users/zwl0331/Desktop/Mini_project_1/songpop/splits.py:101), [songpop/splits.py:128](/Users/zwl0331/Desktop/Mini_project_1/songpop/splits.py:128), [scripts/run_baselines.py:153](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_baselines.py:153)

    `eligible.astype(bool)` silently accepts values such as `2`. The report also falls short of phase-plan §4.3: it reports validation group/class/era summaries but not the corresponding training summaries, and suppresses the detailed strict-fold table.

    **Exact fix:** validate the split schema, unique expected row IDs, integral group/fold IDs, eligibility values exactly in `{0,1}`, and fold ranges before conversion.

    Produce one diagnostic row per `(grouping, seed, fold, side)` containing row count, group count, High and Low counts, High rate, era shares, mixed-label groups, and overlap count. Print both primary and strict tables. Keep strict results diagnostic-only.

15. **Should-fix — Add the strategy’s missing diagnostic metrics.**  
    [songpop/metrics.py:37](/Users/zwl0331/Desktop/Mini_project_1/songpop/metrics.py:37)

    Strategy §4 step 7 requests per-era AUC and row-weighted versus group-weighted AUC. Neither is currently produced. These require no additional model fitting.

    **Exact rule:** retain mean fold AUC as the selection metric. Additionally report mean fold AUC with each row weighted by `1 / primary_group_size`, and pooled OOF AUC per era for each repeat. Include population/class counts. For eras containing only one class, report null with an explanation, not `0.5`. Label all these diagnostics as ineligible for selection.

16. **Must-fix before P1.3 depends on the results — Complete the findings and regeneration record.**  
    [doc/findings.md:228](/Users/zwl0331/Desktop/Mini_project_1/doc/findings.md:228), [doc/plans/phase1-data-baseline.md:74](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase1-data-baseline.md:74)

    The ledger currently ends with the legacy reference. The new production results and comparisons need their own draft findings before downstream work uses them.

    **Exact fix:** add a compact finding for harness reproduction and the nine production baselines, and another for the date ablations and F2 model comparison. Cite final run IDs, hashes, split population, and comparison records. State that bootstrap intervals are conditional on fixed predictions and do not include retraining or selection uncertainty. The P1.2 review record should name those findings.

    After the code is frozen, perform the already-planned artifact regeneration, then run self-checks and both verification commands. Update directory/status documentation. The known stale run IDs are expected at present; I am not counting them as a newly discovered defect.

I checked the following without writing files:

- **All 13 self-checks passed; all 15 legacy checkpoints reproduced to four decimals.**
- The actual grouping, union-find transitivity, exact-float keys, mixed-label handling, eligible-row folds, and incomplete-row assignments are correct. Removing row 425 left eligible assignments unchanged.
- F2’s current-data features, logistic transform/impute/scale order, RandomForest NaN handling, positional CV alignment, and averaging of 15 test predictions behaved correctly. I found no prohibited fitting on test rows.
- Every saved bank’s file hash matched its log record. Stored fold AUCs matched recomputation, and stored deployment means matched fold predictions. All seven saved bootstrap summaries reproduced.
- An independent 25-draw bootstrap using sklearn sample weights agreed with the cached implementation to approximately **7×10⁻¹⁷**. Shared multiplicities, weighted ties, and percentile calculation are correct.
- Recomputing RF F0/F1/F2 changed probabilities by at most **3.3×10⁻¹⁶**, with no fold-AUC change. LightGBM F2 matched exactly. Keep `rtol=0, atol=1e-12`; I found no reason to loosen it or require byte-identical regenerated RF banks.
- The P1.3 selection rule correctly uses unrounded F2 mean fold AUC and the declared tie order. It selects **LightGBM F2, 0.8314739289**. Full production `--verify` still needs to pass after the acknowledged regeneration.