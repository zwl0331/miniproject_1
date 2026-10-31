I recommend a staged driver with a persisted timing decision, separate build and replay processes, and outer scoring only after replay verification. One interface gap needs explicit handling: the current `nested.decide()` output cannot support pooled AUC or bootstrap by itself.

I read the requested sources and inspected logged timings only. I did not read Claude’s Round W positions, change files, fit models, compute scores, or run reviewers. `STATE.md:10` still says paused; its Next action and the later owner decision, “Start P2.7,” supersede that sentence.

1. **Use explicit stages: prepare → measure and smoke → record path → build → replay and compare → decide and report → audit.**

   Put the orchestration in a new `analysis/p27_execute.py`, following the sequential subprocess pattern in [p26_execute.py](/Users/zwl0331/Desktop/Mini_project_1/analysis/p26_execute.py:23). Use the same entry point for build and replay, with subcommands, and stop on any nonzero exit. Keep the existing engine files unchanged.

   **Prepare.** Load `runs.Context`, construct:

   ```python
   pop = P.Population(ctx.rows, ctx.data.y, ctx.groups, "production")
   executable = nested.executable_menu(menu)
   ```

   Check the menu’s protocol against the implementation constants, including outer seed, fold counts, inner seeds, thresholds, meta protocol, and selection keys. Do not assume `executable_menu()` checks these: it primarily translates candidate specifications.

   Generate the outer assignment exactly once with `nested.outer_folds(pop)`. This uses `population_folds()`, which sorts by permanent row ID before `StratifiedGroupKFold`. Save `row_id`, primary group, and integer outer fold under `artifacts/phase2/splits/`, together with metadata binding the data hashes, eligible population, seed `20261101`, and three folds. Read it back immediately and require exact equality.

   Every later stage must reload this assignment and pass `fold=saved_fold`. Validate row order, unique IDs, complete coverage, whole groups, integer fold IDs, and both classes in every validation fold and training complement. The existing `check_folds()` checks training complements, but does not explicitly require integer dtype or both classes in validation folds. Never regenerate or overwrite a mismatching assignment silently.

   **Measure.** Measure the eight distinct executable recipe/seed-policy requests represented in `p27_reservation.menu_atom_runs`, including the separate one-seed and three-seed anchors. Six requests belong to the learned-channel union. Do not multiply costs by twelve candidates or six combiner settings.

   I recommend timing all fifteen production-fold fits for each request, serially, at its frozen estimator-seed policy. Time feature construction, preprocessing, encoders, estimator fitting, held-out prediction, query prediction, and the duplicate-input checks used by the engine. Use the provided test rows as the production query population. Load data and imports before starting these per-recipe timers.

   For literal score-free measurement, use a small timing helper around `fit_recipe()` and its prediction methods, with production `FitContext(split_seed, fold)`. Do not call a scoring function. Both `atomic_cv()` and `run_recipe_cv()` calculate AUC internally, even though the existing pilot prints only timings. Mirror the fit-and-predict workload, not its scoring and bank construction.

   Record an exclusive project execution lock, the absence of other fitting jobs, start/end timestamps, thread settings, and elapsed wall time from `perf_counter()`. A lock alone cannot establish that unrelated jobs were absent. If load contaminates the measurement, mark it invalid and repeat before recording the path.

   With measured fifteen-fold times \(c_a\), compute:

   \[
   C_{\rm all}=\sum_{a\in A}c_a,\qquad C_L=\sum_{a\in L}c_a,
   \]
   \[
   T_{\rm plan}=3.2C_{\rm all}+16.2C_L,
   \]
   \[
   T_{\rm pilot}=3.04C_{\rm all}+14.16C_L,
   \]
   \[
   T_{\rm reserved}=2(1.25)\max(T_{\rm plan},T_{\rm pilot}).
   \]

   This is the reservation calculation in [p26_menu.py](/Users/zwl0331/Desktop/Mini_project_1/analysis/p26_menu.py:56). The factor two covers build and empty-cache replay. The pilot’s `4.91` multiplier concerns production learned evaluation, not P2.7’s third level.

   Choose full iff the **unrounded** `T_reserved <= 28_800`; otherwise choose reduced. Save and reload an immutable `path.json`, and append its log record before invoking `nested.run`. Include raw timings, atom membership, formulas, margin, reservation, and the reduced candidate ID. Record measurement and smoke overhead separately in the schedule.

   **Reason:** the existing 8.48-hour estimate is evidence about previous loaded runs, not P2.7’s gate. The gate must be reproducible and independent of outer performance. The formulas remain projections, not runtime bounds.

2. **Call the existing evaluator directly, with stable provenance and separate cache namespaces.**

   The computational call should be essentially:

   ```python
   result = nested.run(
       executable,
       pop,
       cache,
       manifest["manifest_sha256"],
       runs.packages_for,
       menu["nested_protocol"]["anchor"],
       reduced=(
           menu["production_selection"]["selected"]
           if path["path"] == "reduced" else None
       ),
       fold=saved_fold,
   )
   ```

   This preserves the populations, contexts, averaging order, and third level already implemented in [nested.py](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/nested.py:163). Do not implement another outer loop or substitute global banks for inner predictions.

   Use four disjoint cache purposes: timing/smoke, build, replay, and fixture tests. For the long build, a dedicated disk namespace is useful for recovery. Restarting `nested.run()` against that namespace can reuse completed atomic cross-fits. It will redo combiners and full refits: `refit_predict()` fits its atoms directly and does not cache them. The current API does not provide fold-level execution checkpoints.

   Two implementation details in [Cache](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/procedures.py:119) need handling:

   - Disk storage does not bound memory. Newly computed entries remain in `cache.memory`.
   - `put()` writes directly to the final `.npz` filename. A crash can leave a truncated file.

   For reliable recovery, add a driver-local cache adapter that preserves the existing request checks and uses temporary-file publication followed by an atomic rename. Test it. Use one writer per namespace. Refuse malformed existing entries; never convert arbitrary cache exceptions into misses. If retaining the unmodified cache instead, explicitly document that corruption aborts recovery and requires a fresh namespace.

   Obtain the source manifest through `banks.source_manifest(root)` after all project imports needed by every stage are loaded. Save the complete manifest, then check it again at stage completion. Build and replay must have the same manifest and package maps. Use the real `runs.packages_for`, including optional family versions, rather than `{}` or a fabricated `"replay"` digest.

   The P2.7 manifest will include its new driver, so it need not equal a P2.6 bank’s entire manifest. Separately verify that the frozen engine source files retain their recorded hashes. The distinction follows [banks.source_manifest](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/banks.py:31).

   Build may print timings, cache activity, candidate IDs, inner means, tie sets, and selected candidates. It should neither call `decide()` nor calculate or print any outer AUC.

   **Reason:** correct cache reuse depends on the complete request, while an independently named directory alone does not establish either correct provenance or an independent replay.

3. **Save a dedicated nested-result artifact, not a production `bank_v2`.**

   Use an explicit schema such as `p27_nested_v1`, with numeric arrays in an NPZ loaded with `allow_pickle=False` and metadata in canonical JSON.

   Save:

   - Permanent row IDs, outer assignment, and each fold’s validation IDs.
   - Full path: selected fold-average, selected full-refit, and anchor fold-average predictions.
   - Reduced path: fixed-candidate full-refit and anchor full-refit predictions. Do not create a misleading `fold_average` field.
   - Every returned selection dictionary: selected ID, all twelve unrounded inner means, maximum, tie set, and tie keys. Reduced selections must remain `None`.
   - Menu hash, executable-specification hash, path-decision hash, split hash, data/population/group hashes, source manifest, dependencies, Python version, thread settings, invocation, and execution times.
   - Cache namespace identities, request-key inventories, hit/miss counts, completed-work counts, and any restart history.

   Run `nested.check_result()` before publication and after read-back. Also check the result against the recorded path and saved assignment; `check_result()` alone does not establish those external bindings.

   `nested.run()` does **not** return complete candidate inner banks, combiner states, or fit receipts. Do not claim those were saved. If detailed audit evidence is wanted, use narrowly scoped observers that call the original functions unchanged and serialize returned evidence or receipts. Preserve the underlying engine source and include the observer code in P2.7’s manifest.

   Add distinct records to `artifacts/phase2/experiments.jsonl`, for example `nested_path`, `nested_build`, `nested_replay`, and `nested_decision`, each with `task_id: "P2.7"` and a common evaluation identity. Append completion records only after artifacts are safely written and hashed. Make restarts idempotent and reject conflicting duplicate records. Do not use `runs.save_run()` or ordinary `record: "run"` for these artifacts.

   The later phase 3 R1 packet needs the frozen menu and its R2 association, timing gate, saved split, build results, replay comparison, decision report, provenance, driver tests, and a draft finding recording P2.7’s conclusions and limitations. P2.7 evaluates the selection procedure; it does not itself choose Sam’s two Kaggle entries.

   **Reason:** ordinary production-bank conventions assume different dimensions and semantics. A separate schema prevents nested predictions from accidentally entering production selection.

4. **Keep decision rules unchanged, and explicitly resolve the reporting API gap.**

   After the complete build has passed replay comparison, call [nested.decide](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/nested.py:220) on the saved build result.

   From its current output, the driver can calculate, without accessing labels:

   - Each prediction route’s mean of the three returned fold AUCs.
   - Mean selected-minus-anchor difference.
   - Mean refit-minus-fold-average difference.
   - Any additional paired arithmetic, such as refit-minus-anchor.
   - The selected IDs and unchanged decision string.

   **Pooled AUC and group bootstrap cannot be recovered from those scalars.** Pooled AUC depends on cross-fold positive/negative comparisons. Bootstrap needs labels, group membership, and row-level predictions. Neither is contained in `decide()`’s return value.

   My concrete recommendation is a new trusted reporting function, for example `analysis/p27_report.py::decide_and_summarize(result, pop)`. It calls `nested.decide()` for the existing decision and performs the additional scoring within this same, separately gated reporting stage. General orchestration, timing, build, and replay-comparison code never access labels for outer scoring.

   This preserves the engine files and the intended isolation. If “only the existing `nested.decide()` may access outer labels for scoring” is literal, the requested pooled/bootstrap report is impossible under the present immutable API. Claude should state that limitation rather than disguise an additional label read.

   Inside the reporting boundary:

   - Assemble one length-\(N\) prediction vector per route by permanent row ID. Require exactly one prediction per eligible row.
   - Compute pooled AUC separately from mean fold AUC. Never use pooled AUC in the frozen decision rules.
   - Full path: report all three routes, both declared paired comparisons, their means, and per-fold selected IDs.
   - Reduced path: report fixed-candidate refit versus anchor refit, including their paired differences. Keep `decision=None`; refit-versus-fold-average is unavailable.

   Reuse [paired_cluster_bootstrap](/Users/zwl0331/Desktop/Mini_project_1/songpop/metrics.py:76) through an in-memory adapter with one repeat and three outer folds: `oof.shape == (1, N)` and `fold_auc.shape == (1, 3)`. Use 2,000 replicates and seed `155001`. Sample primary groups with replacement, give every row its group’s multiplicity, and use the same multiplicities for both predictions. Keep folds fixed and record invalid single-class replicates. Report mean-fold-AUC difference intervals for selected versus anchor and refit versus fold average; on reduced, report fixed refit versus anchor refit.

   State that these intervals are conditional on fixed predictions and exclude retraining and selection variability. Also state that the menu benefited from earlier exploration, and the third level uses smaller fitting populations with thinner date support. After fallback or deployment is chosen using outer results, those results are not an untouched estimate of that chosen outcome.

   **Reason:** this supplies the requested statistical report without silently changing the estimand, decision thresholds, or label-access contract.

5. **Make replay a complete recomputation with evidence of its independence.**

   Start replay in a new process with the identical menu, assignment, population names, protocol contexts, source manifest, and dependencies. Create a new empty cache namespace. Never copy or prepopulate it from timing, smoke, build, production banks, or another replay.

   Run the same `nested.run()` call. Only after it finishes should the comparison code load build predictions. Following the principle of [p26_verify_learned.py](/Users/zwl0331/Desktop/Mini_project_1/analysis/p26_verify_learned.py:30), all base predictions and learned meta predictions must come from fresh fitting.

   Require:

   - Exact equality of IDs, assignment, path, selected candidate IDs, tie membership, structural keys, and stable provenance.
   - Prediction arrays and numerical selection evidence within `atol=1e-12, rtol=0`.
   - Exact equality of discrete fallback/deployment decisions when both results are later scored.
   - Fold AUCs and reported numerical summaries within `1e-12`.
   - If combiner states and receipts are captured, the same comparison discipline as P2.6: states and structural receipts exact.

   A changed winner or decision fails even when underlying numbers differ by less than tolerance. Do not loosen AUC tolerances because probabilities are close. Do not compare nested anchor predictions with production anchor predictions; their training populations differ.

   Prove freshness with an initially empty namespace inventory, fresh-process identity, a trace showing every successful cache hit follows a computation in that replay, and fit/request counts. Internal replay hits are expected because candidates share atoms and meta predictions.

   As a structural cross-check for this menu, a cold full run should need 24 inner atomic cross-fits plus 270 third-level cross-fits, with six additional meta cross-fits for each outer fold that selects a learned candidate for full refitting. Full-refit atom fits bypass the cache and need separate accounting. The actual reduced path selects a fixed blend, so zero cache misses is legitimate; its fresh refits must be evidenced directly.

   **Reason:** numerical agreement can be manufactured by reloading build predictions. Empty-start evidence plus observed computation establishes what was actually replayed.

6. **Use one real learned-population smoke test to expose failures hidden by stand-ins.**

   The cheapest useful addition to the timing stage is one complete learned refit on a real inner-training population from outer fold 0, using the frozen channels and estimator seeds, and predicting outer validation features without scoring them.

   Use a throwaway cache and a declared smoke context. Run one frozen stacker through `refit_predict()`, then call `learn_on()` for all six frozen settings using the same meta coordinates. This reuses that smoke’s meta predictions while exercising both combiner implementations. Check shapes, finite probabilities, optimizer success, cache serialization/read-back, and repeatability. Do not use the smoke’s outputs for selection or the later build.

   This exercises risks the fixtures do not establish:

   - **Runtime:** the pilot multipliers came from the anchor. Date-encoded finalists dominate the logged costs, and their scaling need not match the anchor.
   - **Third-level work:** six settings share base fits, but the 45 inner-training populations each require rebuilt meta predictions. A learned outer full refit also needs a new meta stage.
   - **Contexts:** `FitContext` seeds encoders inside each base fit. Preserve the evaluator’s context strings; do not add `"build"` or `"replay"` to them. Namespace names may differ without changing fit contexts.
   - **Memory:** measure peak memory with real multi-seed estimators. Disk cache entries also occupy memory, and prediction-return copies add overhead.
   - **Threads:** preserve four-thread fitting, one-thread forest prediction, and bounded CatBoost prediction through the existing engine. Record numerical-library thread settings and avoid parallel outer folds.
   - **Disk recovery:** exercise a real cache hit in a fresh process and test interrupted publication with fixtures.
   - **Smaller populations:** exercise actual date encoding, missing-value handling, and optimizer behavior at nested training sizes.

   The timing stage already exercises every frozen atom on production-sized populations. The smoke adds the missing recursive learned procedure without running an entire outer-fold selection.

   **Reason:** the current actual-menu check proves plumbing with fake fits. It does not establish real-family compatibility, resource use, or disk behavior.

7. **Add driver tests that fail under specific incorrect implementations.**

   Keep these in a new test file so the frozen engine sources remain unchanged.

   - **Saved assignment:** reject reordered or substituted IDs, duplicate IDs, fractional folds, split groups, changed split bytes, and a valid-looking alternative partition. A spy must observe the loaded assignment passed to both build and replay. Make fold generation raise during those stages to catch accidental regeneration.
   - **Path gate:** test just below, exactly at, and just above 28,800 seconds. Reject missing, altered, or mismatched gate records before fitting. Assert full passes `reduced=None`; reduced passes exactly the frozen production selection. Neither path may inspect outer scores to choose itself.
   - **Replay:** corrupt one prediction, one selected ID, one tie set, one assignment entry, and one provenance field independently. Each must fail. Include values that pass NumPy’s default relative tolerance but exceed the required absolute tolerance.
   - **Freshness:** preload a sentinel build cache and make original-bank loading fail inside replay. Require replay to fit. Permit hits created within replay. Include the reduced fixed-blend case, where cache counters alone cannot prove work.
   - **Driver label isolation:** test the build wrapper with a fixed, already validated assignment. Flip one outer fold’s held-out labels and require that fold’s evidence, selected ID, fold-average predictions, and full-refit predictions to remain unchanged. Include a positive control showing that labels can affect results where those rows are training data.
   - **Scoring boundary:** make reporting or `decide()` calls raise if reached from timing, build, or replay comparison. Require complete results and successful replay before the reporting entry point opens.
   - **Reporting:** use synthetic predictions whose pooled AUC differs from mean fold AUC. Check paired bootstrap weights, whole-group resampling, row alignment, and reduced-path terminology.
   - **Recovery:** simulate a crash during cache publication and between artifact publication and log append. Resume must not accept partial output or append conflicting completion records.

   Extend the existing isolation pattern in [check_phase2.py](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:2419), but test the driver’s own wiring. Source inspection alone is insufficient.

   **Reason:** evaluator correctness does not protect against a driver loading the wrong split, bypassing the timing gate, scoring early, or replaying cached build outputs.

8. **When the measured projection exceeds eight hours, take exactly the frozen reduced path.**

   Cut the twelve-candidate inner selection, its three repeats of five folds, and the associated learned-candidate third-level evaluations.

   Keep all three saved outer folds. Fully refit `blend_eq_all` and the original anchor in each outer training part, then perform a complete empty-cache replay and the reduced report. If the frozen production selection had been learned, its own meta-training rebuild would remain mandatory.

   Do not reduce estimator seeds, remove atoms, prune learned settings, change encoders, omit replay, or choose a smaller menu. Do not reinterpret reduced refit comparisons as evidence for the full path’s fallback or deployment rules.

   If a full run exceeds its projection, record the overrun or failure. Do not silently rewrite its path record or introduce a performance-driven shortcut. Keep results provisional until the later R1 review. Phase 3 uses the production selection and the applicable frozen decision rule; it must not elect a candidate by outer-fold selection frequency.

   **Reason:** the reduced path is already part of the reviewed protocol. Other shortcuts would evaluate a different procedure.

Only Sam can decide:

- To raise the eight-hour reservation or authorize substantial additional computation beyond the recorded scope.
- To change the frozen menu, selection rule, deployment rule, or reduced-path meaning.
- To choose the two final Kaggle entries and authorize their external use and upload.
- Whether any broader brainstorm proposal should reopen the project’s methodological or course-rule decisions.