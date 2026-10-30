Claude, I would fix the execution and verification paths before the readiness review. The current draft can fail at its first allocation decision, fit a run before detecting that it differs from the manifest, and report successful task verification with no completed runs.

I changed no files and computed no P2.2 search scores. The additional probes below used planning code, fabricated records, or parameter calculations.

1. **Must-fix: allocation expects a field absent from every run record.**  
   **Location:** [protocol.py:96](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/protocol.py:96), `ref()`, and [runs.py:156](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/runs.py:156).

   `best()`, `ref()` and subsequent exclusion lists require `recipe_sha256`. `run_record()` does not write it. I confirmed that applying `protocol.best()` to the existing complete run record raises `KeyError: 'recipe_sha256'`. The initial batch’s `resolved_state()` will encounter the same problem.

   **Exact fix:** Normalize records when constructing `protocol.Log`. Derive each P2.2 run’s recipe hash from its validated batch recipe, verify it against the bank’s `metadata["recipe_sha256"]`, and attach it to an in-memory copy of the record. Preserve historical log records. Alternatively, introduce an explicitly versioned record schema with backward-compatible normalization.

   Add a test using the actual shape returned by `runs.run_record()`, rather than synthetic records that already contain the missing field.

2. **Must-fix: validate the complete planned execution before fitting anything.**  
   **Location:** [run_experiments.py:220](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:220), particularly lines 243–246.

   The run-ID check happens after `run_one()` has fitted, saved and printed the score. Changed sources, different seeds or a different recipe can therefore produce an undeclared score before the command rejects it. `run_wave()` also does not check the manifest’s protocol hash against the current protocol.

   **Exact fix:** Before the first fit, validate every selected batch against:

   - Its file hash, batch ID and complete schema.
   - The current protocol hash and source manifest.
   - Task ID `P2.2`.
   - Exactly the protocol’s split seeds, `[155, 2026, 1029]`.
   - Recomputed run IDs for every recipe.
   - The expected manifests obtained by re-planning that wave.

   Keep seed-subset support in the general runner. Reject it specifically for this declared P2.2 protocol.

   Test each mismatch with the fitting function replaced by a spy that must remain uncalled.

3. **Must-fix: task audit and verification currently accept missing work.**  
   **Location:** [run_experiments.py:264](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:264), especially lines 287, 295 and 308.

   Both commands silently filter out planned runs absent from the log. Missing allocations are also skipped. Using valid fabricated wave-1 manifests with **zero run records**, I obtained `True` from both `audit_task()` and `verify_task()`.

   **Exact fix:** Distinguish integrity auditing of an unfinished task from completion verification. Completion requires every planned batch to be either:

   - Completed, with every run, both contribution records and its allocation present.
   - Explicitly recorded as unstarted and stopped by the cutoff.

   A partially completed batch must not count as complete. An empty verification set must not pass completion verification.

   Also reconcile all `task_id == "P2.2"` run records against batch manifests. Unexpected runs must be reported, not omitted. Refit every distinct completed task run once, and verify the task’s comparisons and any submission records.

4. **Must-fix: make interruption recovery explicit and idempotent.**  
   **Location:** [run_experiments.py:48](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:48), `plan_wave()`, `run_wave()`; [runs.py:179](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/runs.py:179).

   Three interruption windows currently cause trouble:

   - `plan_wave()` refuses to resume if even one manifest of the wave was logged.
   - `run_wave()` does not log failures as `run_search()` does.
   - A bank saved before its log append becomes an orphan. Retrying can refit it and then fail because the bank already exists.

   **Exact fix:**

   - Recompute a wave deterministically. Validate and reuse matching existing manifests, then create only missing ones.
   - Wrap each fit with a failure record containing batch ID, recipe hash, planned run ID and exception. Stop immediately.
   - Finalize bank files atomically. Recover a finalized orphan only after validating its expected identity, source, schema and arrays, then append its record exactly once.
   - Validate an already logged run before reusing it. Do not merely print its record and return.

   Add interruption tests after manifest creation, manifest logging, bank finalization, run logging and contribution logging.

5. **Must-fix: enforce initialization and cutoff semantics.**  
   **Location:** [run_experiments.py:237](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:237); plan §§15.2–15.3.

   `--anchors-only` bypasses the cutoff and is accepted for later waves. Ordinary wave-1 execution does not require initialization to have completed. “Started” is inferred from completed run records, so an interrupted first fit has no reliable start state.

   **Exact fix:**

   - Allow `--anchors-only` only for wave 1.
   - Require all seven anchors and the H reproduction check before any non-anchor search fit.
   - Record an explicit `batch_start` event immediately before the first non-anchor fit of an initial batch, or the first fit of a later batch.
   - After the cutoff, resume only batches with a valid earlier start event.
   - Apply the cutoff to initialization too.

   Do not let fitting an anchor make its entire initial batch count as started.

6. **Must-fix: enforce the actual P2.2 H reproduction requirement.**  
   **Location:** [run_experiments.py:229](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:229); plan §15.2.

   The protocol promises that the new H bank equals the P2.1 anchor’s arrays. The execution path does not perform that comparison. The existing anchor self-check tests the recipe separately, not necessarily the bank that becomes this protocol’s H.

   **Exact fix:** After fitting H during initialization, compare all ten prediction-bank arrays with the identified P2.1 anchor using exact equality. Record both bank hashes and the result. Refuse further search execution if any array differs.

   Loading the historical bank for this comparison does not require replaying it under current sources. Its full historical replay remains a separate check at its matching revision.

7. **Must-fix: validate allocation inputs instead of trusting loosely indexed log records.**  
   **Location:** [protocol.py:91](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/protocol.py:91), `Log`, `family_contributions()` and `judge()`; [run_experiments.py:139](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:139).

   Dictionaries silently overwrite duplicate run or allocation records. Contribution records are not scoped tightly enough. `judge()` checks only that two contribution records exist, not that they are the two declared weights. `complete()` trusts an eligibility flag and a list length.

   **Exact fix:**

   - Reject duplicate identities and conflicting recipe bindings.
   - Require exactly one contribution at each challenger weight, `0.1` and `0.25`.
   - Bind contributions to the task, challenger bank hash, H bank hash and identical official split seeds.
   - Validate complete-run status from the bank, including its evaluation population and folds.
   - Reject an incomplete batch member instead of silently skipping it.
   - Validate all stable batch-record fields, including family, wave, kind and configuration count.

   For wave-1 deferred state, resolving values later is acceptable. The resolution must use uniquely identified, validated banks, and its recorded state should include bank hashes and seed policies.

8. **Must-fix: implement the decision-driving bootstrap records required by §5.6.**  
   **Location:** [run_experiments.py:250](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:250); plan §5.6, line 237.

   Allocation currently proceeds without any paired bootstrap. The existing `--compare` command also cannot directly compare the contribution blends, because those deliberately have no saved bank.

   **Exact fix:** Declare a deterministic witness rule before execution. For example, for each satisfied allocation route, select the qualifying configuration with the highest relevant unrounded score, with recipe hash breaking ties. Deduplicate witnesses.

   Reconstruct virtual blend banks when needed. Log the paired comparisons against the relevant incumbent, H and previous best blend. Record the bootstrap settings, parent hashes and repeat differences. Recompute these records during audit.

   The interval remains descriptive. Do not turn exclusion of zero into a new allocation requirement.

9. **Must-fix: repair the failing self-checks and add tests of the actual orchestration.**  
   **Location:** [check_phase2.py:238](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:238) and [check_phase2.py:772](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:772).

   My authorized run returned **24 passed, 2 failed, 0 skipped, 6 staged**.

   The concrete fixes are:

   - Replace kNN’s obsolete `p=2` with `metric="euclidean"` and `normalization="l2"`.
   - For Manhattan, use `metric="manhattan"` and `normalization="l1"`.
   - Correct its expected categorical multiplier from `0.5/3` to `0.5/6`.
   - Replace the writer-disabled replay fixture with a current-source, off-grid fixture in the in-memory filesystem. Keep a separate test that historical-source mismatch is rejected. Do not weaken source checking to make the old anchor pass.

   Before readiness, add non-vacuous tests for all four allocation routes, threshold boundaries, one negative repeat, counter reset/decrement, missing runs, duplicate contributions, changed manifests, interrupted execution, cutoff behavior and full audit reconstruction.

   The kNN test’s title claims training-part isolation, but its current assertions do not establish that. Perturb held-out/test values and confirm unchanged fitted preprocessing. Include a training-data perturbation as a positive control.

10. **Must-fix: deduplicate equivalent numeric configurations during refinement.**  
    **Location:** [refine.py:63](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/refine.py:63), `accept()`; recipe hashing.

    I reproduced a local move from logistic `C=100` to `C=100.0`. The parameter dictionaries compare equal, but the recipe hashes differ. Clipping a proposal to a boundary can therefore admit an unchanged computation as a new configuration.

    **Exact fix:** Introduce a typed search-comparison key. Normalize scale parameters to their declared numeric type and preserve distinctions where the estimator gives integers and floats different meanings. Use that key for exclusions, parent equality and proposal deduplication. Keep historical recipe hashes unchanged.

    Test boundary clipping, explicit versus omitted defaults where equivalent, and integer/float representations of C and regularization values. Assert that every accepted local proposal actually changes the computation.

11. **Must-fix: resolve the parent-selection and comparator inconsistencies before scores exist.**  
    **Location:** [protocol.py:335](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/protocol.py:335); plan §§5.2, 15.2 and 15.4.

    Two rules differ between the documents and code:

    - Section 5.2 says the second parent is the next distinct **qualifying** contributor. The code accepts the next distinct contributor regardless of qualification.
    - Section 15.2 defines G over all P2.2 runs. Wave-1 manifests actually fix G to H, even after other anchors are available.

    **Exact fix:** Define qualification explicitly and use it in `refinement_parents()`. My recommendation is a recorded satisfied §5.6 route, with the best qualifying contribution recipe selected first. If no distinct qualifying second parent exists, give all local slots to the standalone parent.

    Add this sentence to §15.2:

    > “For wave 1, G is fixed to H. For later waves, G is the strongest complete run from earlier closed waves. Results within the current wave never update its comparators.”

    This preserves the existing manifest design and removes order dependence.

12. **Should-fix: strengthen parameter domains and protocol validation.**  
    **Location:** [models.py:49](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/models.py:49), [protocol.py:17](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/protocol.py:17), `batches.validate_batch()`.

    `_fraction()` admits integer `1`. For random forests, `max_samples=1` means one sampled row, while `max_samples=1.0` means all rows. I verified those outcomes from the installed library’s bootstrap-size helper: **1 versus 100** for a 100-row input. `max_features` has the same integer-versus-fraction distinction.

    **Exact fix:** For the declared fractional search parameters, require floats. If integer forms are supported later, give them an explicit separate domain. Make `max_depth` validation family-specific.

    Also validate protocol-level keys, task, official seeds, unique families, positive counts, initial/expansion/ceiling relationships, hard-bound types and refinement totals. Correct Extra Trees’ declared fallback `max_features` default from `"sqrt"` to its factory default, `1.0`.

    Validate recipes from explicit and grid branches against admissibility rules too, except the designated anchors.

13. **Should-fix: distinguish sampler exhaustion from an attempt limit.**  
    **Location:** [searches.py:125](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/searches.py:125); [protocol.py:316](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/protocol.py:316).

    Reaching 10,000 rejected draws does not prove that the space is exhausted. Returning no recipes can then lead to “no family has allowance left,” although allowance remains.

    **Exact fix:** Return a termination reason such as `requested_count`, `enumerated_exhaustion` or `attempt_limit`. For small finite spaces, use deterministic enumeration to establish exhaustion. Record why a family stops.

    Extend `verify_searches()` to verify the saved recipe-hash list and v2 rejection counts, not just the recipes and search ID. Test an exhausted space separately from a sparse admissible space that hits the attempt limit.

14. **Should-fix: record actual timing reservations and enough provenance to audit them.**  
    **Location:** [run_experiments.py:89](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:89), `time_cases()`; plan §§5.1 and 15.6.

    The timing function correctly avoids scoring. Its records, however, contain only a source digest and limited environment information. No batch reservation links the measurements to the planned workload.

    **Exact fix:** Record the timing-spec hash, full source-manifest reference, data and fold hashes, Python/package versions, thread settings, invocation and canonical seeds. Recheck sources before appending the timing record.

    Save a reservation record covering initial execution, permitted expansions, bootstrap work, full replay and review time. Link it from the execution manifest or readiness record. State explicitly that the current “about two hours” estimate excludes replay and is not an upper bound.

    Canonicalize seeds inside `time_recipe()`, rather than relying solely on the CLI. Add a test that patches scoring functions to raise and confirms timing still succeeds.

15. **Should-fix: test kNN’s difficult cases without promising a tie rule the implementation does not define.**  
    **Location:** [models.py:175](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/models.py:175); kNN self-check.

    “Ties at the k-th neighbor follow the fitting-row order” is too precise. Fixed input order supports repeatability, but does not establish a particular stable ordering among tied neighbors.

    **Exact replacement:**

    > “Training rows use permanent row-ID order. Tied-neighbor behavior is that of the pinned scikit-learn implementation. Repeatability and identical-query predictions are tested.”

    Add synthetic checks for mixed-label zero-distance neighbors, a tie at the k-th boundary, duplicate query rows, unseen categories and query batching.

    Exact F1 matches across different primary groups are possible because F1 omits date precision. That is not newly introduced leakage. Keep the declared primary folds, document this limitation, and do not silently collapse rows or change grouping.

16. **Should-fix: make inexpensive blend auditing reconstruct the blend.**  
    **Location:** [runs.py:276](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/runs.py:276).

    `audit_run()` checks parent hashes for a saved blend, but does not check that the child predictions equal the declared weighted parents. Rebuilding a record from the child’s own arrays cannot establish that relationship.

    **Exact fix:** Load the parents, apply the declared seed restriction and weights, reconstruct the blend arrays, and compare them with the child. This requires no model fitting. Add a test that changes child predictions and consistently updates its metrics and logged hash.

17. **Should-fix: finish the readiness documents with accurate status and explicit commands.**  
    **Location:** phase plan header, §§15.1–15.6 and Status; README/master-plan status lines.

    The phase-plan header already says P2.2 is “in review” and links a record that does not yet exist. Its trailing status still ends with P2.0b in review. Section 5.1 still calls Finding 010 draft.

    **Exact fix:** Until the record opens, say “P2.2 draft under consultation; execution-readiness review pending.” Add the P2.0b approval and P2.2 start events, update Finding 010’s wording, and include the exact ordered readiness/execution/audit/replay commands.

    Explicitly record that serial execution makes log shards unnecessary for P2.2. Keep the full refit in the completion checkpoint. Do not reduce it to a reviewer-selected sample.

    Section 15.4’s long refinement paragraph would be clearer as a small table separating parent choice, local counts, movement rules, tree variants and broad draws.

What I checked and found correct:

- Pure wave-1 planning produces **64, 48, 48, 32, 12, 24 and 48** configurations, totaling **276**.
- With fabricated complete records, each family could produce a 12-configuration first refinement batch with the declared **5/4 local split and three broad draws**.
- The implemented kNN block multipliers match the declared Euclidean and Manhattan formulas. Preprocessing remains inside the estimator fitted on each training part.
- The four allocation formulas and allowance reset/decrement logic match §5.6 when given valid inputs.
- The existing checker passed the v1 expansion compatibility check, v2 admissibility checks and anchor array/CSV reproduction checks.
- Using section 15 instead of a separate plan file, serial execution, omitted `subsample_freq` at full sampling, and compact contribution records are reasonable choices. None requires reversing the design.