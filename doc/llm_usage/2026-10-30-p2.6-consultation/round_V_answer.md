The nominal nested path keeps outer-validation labels out of fitting and selection. The draft is nevertheless **not ready for P2.6b evaluations**: several validation boundaries accept invalid evidence, and the constrained optimizer has a reproducible false-success case.

I inspected commit `3f75181` and ran only pure-function and tiny synthetic probes. No files were changed and no project-data scores were computed. “Blocking” below means blocking completion of P2.6a.

1. **Blocking: `check_channel_bank` does not establish that production-bank reuse is sound.**

   Location: [procedures.py:259](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/songpop/phase2/procedures.py:259), also `evaluate_learned` at line 288.

   It checks row IDs, fold assignments, and recipe hashes. It does not check labels, groups, split-seed identities, prediction IDs/order, data hashes, sources, dependencies, fitting contexts, or fitting receipts. For an average, it neither verifies parent hashes nor reconstructs predictions from those parents.

   **Probe:** an object containing only `fold_id`, `train_row_id`, and matching recipe metadata passed. It had no predictions, labels, groups, or provenance.

   Reusing full-\(D\) channel predictions is statistically valid, but this guard does not establish its prerequisites. `evaluate_learned` subsequently accepts untyped arrays without invoking the guard.

   **Fix:** return a validated channel-prediction object from a single production adapter. Validate the bank and every parent against the expected population, query population, final recipe, effective contexts, sources, and dependencies. Reconstruct fixed averages. Pass that checked object into `evaluate_learned`, retaining its provenance.

2. **Blocking: cache hits accept corrupt predictions and expose mutable stored arrays.**

   Location: [procedures.py:117](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/songpop/phase2/procedures.py:117).

   `Cache.get` validates only the request string, OOF shape, and OOF finiteness. It checks neither OOF probability bounds nor the shape, finiteness, or bounds of `pred`. Hits return the actual stored arrays. `put` also stores references and can replace an in-memory entry while leaving an existing disk entry unchanged.

   **Probe:** OOF probabilities of `2.0` and a wrongly shaped `pred=[NaN]` were accepted. Mutating the returned OOF array changed the next cache hit.

   **Fix:** validate both arrays on insertion and retrieval, including exact expected shapes and probability bounds. Store defensive copies and return copies or read-only arrays. Make repeat insertion verify equality or refuse it. Bind payload hashes and receipts to the entry, and validate requests before lookup so a cache hit cannot bypass checks required on a fresh fit.

3. **Blocking: selection does not verify the population, split protocol, or even candidate identity.**

   Location: [selection.py:14](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/songpop/phase2/selection.py:14), validation at line 25.

   `Evidence` has no row IDs, split seeds, population identity, or candidate-specification identity. Comparing all evidence with the first candidate establishes only mutual agreement, not agreement with the intended evaluation.

   **Probes:**

   - Evidence with groups split across folds was accepted.
   - Evidence whose `candidate_id` differed from its dictionary/menu key was accepted.
   - A sixth fold was accepted and silently excluded from scoring because score reconstruction loops only over folds `0..4`.

   Three copies of one repeat are also indistinguishable from three properly identified repeats.

   **Fix:** include permanent row IDs, split-seed identities, candidate-specification hash, and an expected evaluation contract. Check exact lengths, uniqueness, binary labels, group integrity, integer fold IDs exactly covering the declared folds, and both classes where AUC is evaluated. Bind each dictionary key to its evidence identity. Keep query predictions excluded as they are now.

4. **Blocking: clipping can make SLSQP report success at a point with an immediate feasible improvement.**

   Location: [procedures.py:197](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/songpop/phase2/procedures.py:197).

   The gradient is correct away from clipping boundaries. At exactly either boundary, the strict comparisons force the data gradient to zero. That can produce false convergence.

   **Synthetic reproduction:**

   ```python
   Z = [[0.0, 2e-6],
        [0.5, 0.5]]
   y = [1, 0]
   ```

   With lambda zero, the implementation returns `[0.5, 0.5]` after one iteration with “Optimization terminated successfully.” Its loss is `7.2543288693`. The feasible weights `[0.499, 0.501]` lower it to `7.2533298679`; `[0, 1]` gives `6.9077552790`.

   **Fix:** add this regression case and explicit feasible-direction checks around clipping boundaries. Reject apparent convergence when the declared objective admits an improving feasible move. Declare deterministic handling of these cases before evaluation. Simply making the mask inclusive fixes this example but does not resolve flat clipped regions generally. A smooth alternative objective would need an explicit specification change, not a silent implementation substitution.

5. **Blocking: malformed procedure specifications can silently change the predictor.**

   Location: [nested.py:25](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/songpop/phase2/nested.py:25), especially line 39; `procedures.py:244`.

   Channels are placed in dictionaries by ID without checking uniqueness. Duplicate IDs overwrite earlier channels. Blend weights are not validated. Unknown combiner kinds silently dispatch to the stacker.

   **Probes:** two equally weighted channels producing `0.2` and `0.8`, but sharing an ID, produced `0.8` instead of `0.5`. `fit_combiner(("misspelled", 0.1), ...)` fitted a stacker.

   **Fix:** add a strict procedure-specification validator before any fit or lookup. Reject duplicate channel IDs, empty channels, unsupported kinds, malformed settings, invalid weights, and inconsistent references. Validate the restricted grammar independently of P2.6b’s future menu-construction code.

6. **Blocking: the procedure artifact is a schema tag, not a verifiable learned-procedure record.**

   Location: [procedures.py:277](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/songpop/phase2/procedures.py:277), bank construction at line 316 and saving at line 328.

   The meta receipt contains a population name, row count, and fold hash. Actual assignments, contexts, fitting receipts, and atomic prediction references are discarded. `atomic_cv` discards the engine’s encoder and stopping receipts. `procedure_bank` does not automatically preserve `arrays["states"]`; arbitrary metadata such as `{"run_id": "x"}` suffices. The saver checks only the schema string.

   There is also no learned-procedure loader/verifier dispatch in the existing run machinery.

   **Fix:** define and enforce required metadata: specification identity, ordered channels, atomic references, actual saved meta folds, effective contexts/seeds, population identities, combiner states, numerical settings, feature lineage, source/dependency identities, and leaf receipts. Validate before saving and after loading. Implement replay that rebuilds atomic predictions from an empty replay cache.

   Also expose a fitted procedure accepting raw rows, or a reproducible reconstruction API. Currently the code retains prediction arrays while discarding the fitted composition required for later deployment reconstruction and raw-feature analysis.

7. **Blocking: recursive duplicate validation and full-refit artifact separation are still missing.**

   Location: [nested.py:53](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/songpop/phase2/nested.py:53); [check_phase2.py:2405](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/scripts/check_phase2.py:2405).

   `refit_predict` calls `fit_recipe(...).predict_high(...)` directly, bypassing the per-estimator duplicate checks in `run_recipe_cv`. Procedure outputs lack recursive leaf/channel checks. Existing submission validation still accepts only fitted banks and flat blends.

   The “deployment” test demonstrates only that `procedure_v1` is rejected by the fitted-bank validator. It does not exercise a full-refit artifact or demonstrate separation from fold-average deployment.

   **Fix:** retain and validate each atomic member before averaging, then each fixed channel, combiner application, and final fold average. Check actual fitted-input equivalence, raw-source duplicates, and repeated URI pairs where applicable. Add distinct full-refit artifacts with fitting-population and deployment identities.

   Required negative tests include opposite corruptions in two leaves that cancel inside a within-family average, opposite fold corruptions that cancel in deployment, and corruption of a zero-weight channel. None should pass merely because the final predictions remain unchanged.

8. **Blocking: `decide` can apply the final rules to an incomplete or malformed outer evaluation.**

   Location: [nested.py:121](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/songpop/phase2/nested.py:121).

   There is no validation of three distinct folds, exactly-once validation-row coverage, group exclusion, prediction lengths/bounds, or consistency with the saved outer assignment.

   **Probe:** a result containing only one outer fold triggered the anchor fallback.

   **Fix:** validate the complete nested result before scoring or deciding. Require folds `{0,1,2}`, each expected held-out population exactly, all eligible rows covered once, and valid aligned predictions. Bind the result to the menu and outer-split hashes. Add missing-fold, duplicate-fold, duplicate-row, and substituted-population rejection tests.

   The fallback-before-refit ordering itself is correct. The reduced path also correctly returns no decision.

9. **Blocking: the five checks overstate their isolation and coverage evidence.**

   Location: [check_phase2.py:2311](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/scripts/check_phase2.py:2311), especially lines 2323–2328 and 2385–2386.

   The learned-procedure spy observes entire CV populations, not the individual innermost fits. `meta_calls` is computed and never asserted. The nested ancestry assertion accepts a fit whenever it lies inside *any* outer-training population, rather than the outer-training population belonging to that fit’s context.

   Both learned isolation fixtures use plain LightGBM without target encoding or context-dependent preprocessing. Thus they cannot expose several context/encoding errors. The selector checks do not separately establish every tie-break stage.

   **Fix:** instrument the actual `engine.fit_recipe` boundary and associate every call with its expected ancestor chain. Assert exact fitting complements, whole-group exclusion, and once-only meta assignments. Include a target-encoded recipe and a context-sensitive spy estimator. Add all rejection cases above, separate tests for each tie-break, and an inner-validation-label isolation test in addition to outer and meta tests.

   `_tiny_population` at line 2289 reads the first project rows. I therefore did not run these five checks. Use genuinely synthetic grouped fixtures for these invariants and keep the anchor regression separate.

10. **Blocking before evaluation: the cost pilot and empty-cache replay contract are not implemented.**

    Location: [nested.py:92](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/songpop/phase2/nested.py:92); cache construction at `procedures.py:110`.

    `run` accepts an arbitrary cache and a caller-selected reduced path. There is no recorded cost-based path decision or enforcement of an empty cache for replay. I found no P2.6a cost pilot in the draft.

    **Fix:** the pilot must measure the actual implementation’s:

    - Atomic fitting and prediction at the relevant nesting sizes.
    - Meta construction and all six combiner settings, with actual fit counts proving reuse.
    - Full-refit execution, including repeated atoms not currently cached there.
    - Serialization, cache loading, validation, and complete replay from an empty namespace.
    - TabM stopping/refit and worker costs if it is admitted to learned channels.
    - Subsequent comparison/bootstrap costs and a runtime margin.

    Use synthetic inputs and the anchor for P2.6a. P2.6b then measures admitted recipes under final seed policies. Reserve build **and** replay; a warm-cache rerun measures neither.

    Record full versus reduced path before outer scoring. The formulas in section 8 are cross-checks, not substitutes for measured request counts and elapsed time.

11. **Non-blocking for the current validated data, but fix the cache serialization and context contract.**

    Location: [procedures.py:42](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/songpop/phase2/procedures.py:42), `frame_digest` at line 95 and `atomic_request` at line 145.

    **Probe:** a missing value and the literal string `"NA"` yield the same frame digest. IDs, labels, and folds also undergo lossy integer coercions before hashing. The request omits explicit Python, data-provenance, and lock identities. Protocol coordinates use delimiter-joined strings instead of typed structured values.

    The supplied dataset’s normal types avoid several of these collisions, but `Population` does not enforce that domain.

    **Fix:** validate types before hashing or use a typed canonical serialization with separate missing-value representation. Include the declared environment/provenance identities. Serialize contexts structurally and retain them in receipts.

    I found no candidate-dependent seed in the current nested callers: their meta and nested contexts are distinct and shared across settings. Preserve that property. Also preserve the original numeric `FitContext(seed, fold)` when reusing production fits; replacing it with a descriptive `"prod:..."` string changes date-encoding and TabM seeds.

12. **Non-blocking numerical hardening: cleanup and application need explicit validity checks.**

    Location: [procedures.py:206](/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/wt/songpop/phase2/procedures.py:206), stacker application at line 235.

    Cleanup removes every positive weight below `1e-9`, then renormalizes without checking displacement or objective degradation relative to the raw solution. The final comparison against equal weights does not establish that cleanup was harmless. Explicit finite-objective checks are absent; with a forced successful solver result, a NaN objective passes because `NaN > threshold` is false.

    **Fix:** validate finite, correctly shaped probabilities and binary labels before fitting; validate finite objectives, states, and outputs afterwards. Keep small positive weights unless a tightly bounded cleanup is part of the declared method. Check cleanup displacement and loss change relative to the raw solution. Use `scipy.special.expit` for stacker application.

    The stacker’s training-matrix scaling, class orientation check, and conversion of convergence warnings into failures are otherwise implemented as intended.