My recommendation is to keep P2.2 as one task, use all three split seeds throughout, and freeze the execution rules before scoring. Run every initial family batch before allocating refinements. Keep contribution checks as compact, reproducible records rather than saving two additional banks per configuration.

I read the requested plan, policy, findings and code. I computed no project-model AUC. My new checks were limited to synthetic kNN behavior, source inspection, bank size and budget arithmetic. Those checks are consultation inputs, not reviewed findings.

1. **Keep one P2.2 record, with an execution checkpoint and a final results review.**

   Use `reviews/2026-10-29-p2.2-searches.md`. Record two distinct review scopes:

   | Checkpoint | What the reviewer receives | What its approval permits |
   |---|---|---|
   | Execution readiness | Code, self-checks, timing records, complete initial configuration lists, execution protocol and exact commands | Executing the declared search procedure |
   | P2.2 completion | Everything above, banks, allocation history, verification, findings and any proposed submission | Committing P2.2 and starting dependent searches |

   The first approval cannot validate future results. New artifacts change the fingerprint and require the second review. Do not describe P2.2 as complete after the first checkpoint.

   Keep adaptive expansion within this task. The approved algorithm determines subsequent batches; each resulting manifest is saved before fitting. An unplanned manual change to sampling or allocation requires another review before execution.

   **P2.3 searches wait for final P2.2 convergence.** Its approved installation and independent timing work can proceed as the task table permits. If releasing shared infrastructure earlier becomes necessary, explicitly re-cut the task in the phase plan and review that change. Do not infer partial task release from an intermediate approval.

   Commit the complete task after convergence and the applicable human go-ahead. Preserve the code unchanged between execution readiness, search and replay.

   **Cost:** At least two review checkpoints. Reserve 60–90 minutes across them, plus correction time. This is more predictable than discovering protocol defects after hundreds of runs.

2. **Write a compact execution protocol and machine-readable state before running.**

   Add:

   - `doc/plans/p2.2-execution.md`: the operational choices below, subordinate to the phase plan.
   - `configs/phase2/searches/p22_protocol.json`: anchors, family order, seeds, budgets, references, refinement rules and cutoff.
   - `artifacts/phase2/searches/`: immutable expanded batch manifests.
   - `artifacts/phase2/timing/`: score-free timing records.
   - `artifacts/phase2/allocations/`: reproducible allocation decisions, or equivalent typed log records.

   The protocol must state that P2.2 uses F2 for tree families, the three declared logistic branches, and F1 for kNN. No engineered date blocks, training weights, estimator-seed search, feature-removal search or learned ensemble belongs in this task.

   Mark Finding 010 reviewed using the P2.0b convergence record. Update the stale “in review” status lines as part of this task. The record explicitly deferred those edits to the next task.

   **Cost:** Small documentation work. It prevents choices from being made implicitly after scores arrive.

3. **Extend factories narrowly, with explicit parameter validation.**

   In `songpop/phase2/models.py`, the installed-family additions actually needed are:

   | Family | Additions beyond P2.1 |
   |---|---|
   | Extra Trees | `max_depth` |
   | XGBoost | `reg_alpha`, `gamma` |
   | HistGB | `l2_regularization` |
   | kNN | New factory with `n_neighbors`, `weights`, `metric` |

   LightGBM, random forest and logistic already expose their required initial parameters. CatBoost and EBM factory changes remain P2.3.

   Add a phase 2 parameter validator, for example:

   ```python
   validate_model_params(family: str, params: dict) -> None
   ```

   Validate types and domains before starting a batch. Examples:

   - Positive integer tree counts and leaf sizes.
   - `max_depth` as `None` or a positive integer where supported.
   - `max_features` as `"sqrt"` or a float in `(0, 1]`.
   - Nonnegative regularization.
   - Positive learning rate and `C`.
   - Fractions in `(0, 1]`.
   - No booleans masquerading as integers.

   Keep `bootstrap`, `early_stopping`, threads and determinism as factory policies, not searchable boolean parameters. Retain the existing LightGBM settings and HistGB thread limit.

   Check the largest spline representation with `C=100` during the score-free pilot. Treat a convergence warning as something to resolve before search, rather than silently accepting incomplete optimization or dropping that configuration afterward.

   **Cost:** Modest code and synthetic factory checks. No dependency installation.

4. **Implement kNN preprocessing as a reusable preprocessing field.**

   Add `songpop/phase2/preprocessing.py`, with a named constructor:

   ```python
   make_knn_preprocessing(prep: dict, columns: list[str]) -> Pipeline
   ```

   Suggested recipe fragment:

   ```json
   {
     "family": "knn",
     "params": {
       "n_neighbors": 60,
       "weights": "distance",
       "metric": "euclidean"
     },
     "base": "F1",
     "blocks": [],
     "drop": [],
     "preprocessing": {
       "name": "knn_blocks",
       "audio_scaler": "standard",
       "alpha": 1.0,
       "category_weight": 0.5,
       "normalization": "l2"
     },
     "training_weights": "none",
     "estimator_seeds": [0]
   }
   ```

   The pipeline is:

   | Block | Fold-fitted processing | Final multiplier |
   |---|---|---|
   | Ten continuous audio columns | Existing log transforms, median imputation, standard scaling | `1 / sqrt(10)` |
   | Year | Median imputation, standard scaling | `alpha` |
   | Key, mode, time signature | Most-frequent imputation, one-hot encoding | `category_weight / sqrt(3)` |

   Use dense output and `OneHotEncoder(handle_unknown="ignore")`. Fit categories only on the training population. Use `keep_empty_features=True` where necessary so an entirely missing training column does not silently change the block definition.

   For the Manhattan alternative, specify the normalization rather than leaving “block normalization” undefined:

   \[
   d=\frac{1}{10}\sum_j|\Delta z_j|
     +\alpha|\Delta z_{\text{year}}|
     +\frac{\beta}{6}\|\Delta\text{onehot}\|_1.
   \]

   Thus, for known categories, the categorical term is `beta × fraction of the three fields that differ`. This deliberately gives beta an explicit Manhattan interpretation. Do not reuse the Euclidean multipliers accidentally.

   Validate:

   - `knn_blocks` consumes exactly its declared columns. Reject unsupported extras rather than silently dropping them.
   - P2.2 uses F1, no blocks and no drops.
   - kNN has one estimator seed, canonically `[0]`, and no training weights.
   - `normalization="l1"` matches Manhattan; `"l2"` matches Euclidean.
   - Alpha and category weight are finite and in the declared domains.
   - `k <= number of fitting rows`.

   Keep preprocessing independent of the estimator factory. A representation compatibility restriction is reasonable; hard-coding the search’s screening decisions into the estimator is not.

   Robust scaling and cyclic calendar representations can remain P2.4. Manhattan and category-weight refinement can be supported now without necessarily running those trials.

   **Cost:** Roughly one focused implementation and test block. The actual representation needs a new timing pilot; Finding 010’s eleven-column proxy is not its timing.

5. **Use ordinary sklearn kNN, with an explicit ordering and tie policy.**

   Start with `algorithm="brute"` and a fixed thread policy. Keep fitting rows in permanent row-ID order, carrying labels with them. Do not deduplicate training rows or replace conflicting labels.

   Important distinctions:

   - With distance weights, zero-distance neighbors receive the weight; nonzero-distance neighbors receive none when an exact match exists.
   - Conflicting exact matches therefore average their labels among the selected neighbors.
   - A tie at the kth-neighbor boundary can depend on training-row order. Fixed order makes that behavior reproducible; it does not make the result invariant to arbitrary training-row permutations. This is documented by [scikit-learn’s KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

   My synthetic check, using training points `0, 0, 2, 4` with labels `0, 1, 1, 0`, gave probability `0.5` at zero for distance-weighted `k=3`, versus `2/3` for uniform weighting. No AUC was involved.

   Retain the engine’s existing fit-time identical-input check. Add tests for identical queries:

   - Together and in separate prediction calls.
   - At different positions in a query batch.
   - With zero-distance conflicting neighbors.
   - With a kth-boundary tie.
   - With unseen categories.

   If those fail under the chosen parallel setting, resolve the implementation or thread policy before search. Do not loosen the duplicate tolerance to hide the problem.

   **Cost:** Tiny synthetic tests. No custom nearest-neighbor implementation is justified initially.

6. **Introduce `search_v2`; preserve `search_v1` behavior exactly.**

   Extend `songpop/phase2/searches.py` by dispatching on schema:

   ```python
   validate_search(spec: dict) -> dict
   expand_v1(spec: dict) -> list[dict]
   expand_v2(spec: dict, prior_recipes: list[dict]) -> dict
   expand(spec: dict, ...) -> ...
   ```

   Keep the existing v1 expansion, ordering and identity semantics unchanged. Do not retroactively add admissibility filtering to v1.

   A v2 specification should contain:

   ```text
   schema, name, family, stage, batch_kind
   protocol_sha256, sampler_seed, count_total
   branches                    # full recipe templates and branch quotas
   parameter_domains
   conditional_rules
   admissibility_rules
   anchors                     # exact full recipes
   prior_recipe_hashes
   parent_refs
   range_extension_refs
   ```

   Use a small declarative rule vocabulary, not Python expressions or `eval`. For example:

   ```json
   {
     "conditional_rules": [
       {
         "if": {"parameter": "params.subsample", "operator": "<", "value": 1.0},
         "then": {"params.subsample_freq": 1},
         "else": {"params.subsample_freq": 0}
       }
     ],
     "admissibility_rules": [
       {
         "kind": "product_interval",
         "parameters": ["params.learning_rate", "params.n_estimators"],
         "minimum": 5.0,
         "maximum": 100.0
       }
     ]
   }
   ```

   Apply conditions before canonical hashing. An anchor exemption permits its declared grid or iteration-product exception, not illegal parameter values.

   Use indexed selection from Python lists. `rng.choice` on a mixed list can coerce numbers and strings into a common dtype.

   Save the accepted ordered recipes, their hashes, rejection counts and the prior-history digest. Bound rejection attempts and fail clearly when the remaining space is exhausted.

   I checked the initial iteration grids: the product rule admits **21/24 LightGBM**, **17/20 XGBoost** and **8/9 HistGB** iteration/rate pairs. These are new arithmetic checks, not model results.

   **Cost:** The largest common infrastructure addition. A versioned schema avoids breaking historical expansion verification.

7. **Make local refinement typed, bounded and reproducible.**

   Add:

   ```python
   propose_local_batch(
       spec: dict,
       parents: list[dict],
       prior_recipe_hashes: set[str],
       rng: np.random.Generator,
   ) -> list[dict]
   ```

   Explicit rules:

   | Parameter type | Proposal rule |
   |---|---|
   | Positive scale | Multiply by a log-uniform factor between `0.5` and `2` |
   | Zero-or-positive scale | At zero, move to a declared positive neighbor; otherwise multiplicative movement or an explicit move to zero |
   | Ordered integer | Adjacent declared values, or rounded multiplicative movement if designated as a scale |
   | Categorical | Choose another declared category |
   | Nullable integer | Treat `None` as a separate branch, never as a numeric edge |
   | `"sqrt"` or fraction | Treat `"sqrt"` as categorical; perturb only the numeric branch |

   Use one documented integer rounding rule, such as `floor(x + 0.5)` for positive values. Fix float serialization precision for generated values. Deduplicate using the full canonical recipe hash, not the human-prefixed `recipe_id`.

   For each batch of 12:

   - Three broad proposals.
   - Nine local proposals.
   - Five local slots from the standalone parent and four from the contribution parent.
   - Reverse the 5/4 allocation in the second batch.
   - If only one distinct parent exists, use that parent for all nine local slots.

   For boosted parents, include the two new tree counts around the existing center, with all other fields fixed. Use an exact declared rule such as `round(2T/3)` and `round(1.5T)`. The center references its existing bank and consumes no new trial. Those proposals occupy local slots; they are not extra trials outside the allowance.

   Every proposal must satisfy the active search bounds and learning-rate product rule. A local perturbation cannot quietly extend a range.

   **Cost:** Up to 24 configurations per selected family, already inside its family ceiling. Four fully refined families add at most 96 configurations.

8. **Use explicit range-extension manifests.**

   Add:

   ```python
   propose_range_extension(state: dict, domains: dict) -> dict | None
   validate_range_extension(manifest: dict, history: list[dict]) -> None
   ```

   Each manifest records:

   ```text
   family, parameter, old_values, old_bounds
   triggering_recipe_hash, triggering_run_id
   edge, proposed_values, new_bounds, legal_hard_bounds
   prior_extension_count, timing_reservation
   parent_state_sha256
   ```

   Resolve “family’s best” to the **best complete standalone recipe** for this trigger. That is the most literal implementation of section 5.2. Contribution-driven parents still receive local refinement inside the active region.

   Predeclare hard bounds. Reasonable P2.2 defaults include:

   | Parameter | Suggested hard bounds |
   |---|---|
   | LGB leaves | 3–127 |
   | LGB minimum leaf size | 5–400 |
   | XGB depth | 1–8 |
   | LGB/XGB trees | 100–3,200 |
   | HistGB iterations | 100–1,200 |
   | Learning rate | 0.005–0.2, still subject to the product rule |
   | Sampling fractions | Their legal domain, never above 1 |
   | Logistic C | `1e-4`–`1e3` |
   | kNN neighbors | 5–960, also bounded by the actual fitting population |

   Keep RF and Extra Trees at 500 trees in this stage.

   Permit only one outward extension per numeric parameter across this base-feature stage. Zero regularization cannot extend downward. Neither `"sqrt"` nor `None` has an edge. The exact new values and their timing reservation must exist before fitting.

   **Cost:** Usually little fitting. The main benefit is preventing an apparently small local search from becoming an unbounded adaptive search.

9. **Fix H, anchors and initial configurations before their scores.**

   Use the unchanged LightGBM F2 recipe, estimator seed 0, as **H throughout P2.2**.

   Reproduce it under the P2.2 source revision and compare its arrays with the approved anchor. Its new run identity is expected to differ because the source manifest changes. Preserve the historical bank.

   Recommended family anchors:

   | Family | Anchor |
   |---|---|
   | LightGBM | Exact approved phase 1 recipe |
   | Random forest | Exact phase 1 F2 recipe: 500 trees, minimum leaf 3, remaining original settings |
   | XGBoost | F2, 1,000 trees, depth 4, rate 0.03, child weight 1, row/column fractions 0.8, L1 0, L2 1, gamma 0 |
   | Extra Trees | F2, 500 trees, minimum leaf 2, all features, unlimited depth, no bootstrap |
   | HistGB | F2, 300 iterations, 31 leaves, minimum leaf 20, rate 0.05, L2 0 |
   | Logistic | F2, phase 2 `log_standard`, C 1 |
   | kNN | F1, k 60, distance weights, alpha 1, category weight 0.5, Euclidean |

   The logistic anchor is a **new phase 2 reference**, not a reproduction of phase 1 logistic. Phase 2 one-hot encoding and its pipeline differ.

   Save the complete initial recipe lists before anchor initialization. The initialization wave fits the named anchors and makes no allocation decision. Then seal each family’s initial batch state using those actual bank references. Anchors count within the initial totals and are reused, not fitted twice.

   For logistic’s 24 configurations, use:

   - Six C values with `log_standard` F1.
   - The same six with `log_standard` F2.
   - Twelve spline F1 configurations, two knot choices per C.

   A concrete balanced spline allocation is:

   | C | Knots |
   |---:|---|
   | 0.001 | 3, 5 |
   | 0.01 | 4, 7 |
   | 0.1 | 5, 10 |
   | 1 | 7, 3 |
   | 10 | 10, 4 |
   | 100 | 3, 5 |

   kNN uses the full declared 48-point grid.

   Use deterministic sampler seeds derived from an explicit family index and batch number, for example `2026102900 + 100 * family_index + batch_index`. Record the index table. Do not use Python’s process-randomized `hash()`.

   **Cost:** Initial P2.2 remains **276 configurations**. No added package or tuning of anchors.

10. **Use batch barriers so execution order does not govern allocation.**

   Execute initial families in a fixed order, for example:

   `LightGBM, random forest, XGBoost, Extra Trees, HistGB, logistic, kNN`.

   All initial batches use the same frozen cross-family comparator snapshot. Complete all initial batches before choosing refinement families.

   Each batch manifest should bind:

   ```text
   protocol and source hashes
   family and batch sequence
   ordered recipes and prior-history hash
   family anchor
   pre-batch incumbent
   strongest standalone reference G
   fixed H
   previous best family blend with H
   run IDs, bank hashes and seed policies of those references
   allowance before the batch
   remaining unique-configuration budget
   timing reservation and cutoff
   ```

   Choose at most four refinement families at the initial barrier:

   1. The two strongest qualifying families by standalone AUC.
   2. Up to two further qualifying families by best contribution to H.
   3. Deduplicate; fill vacant places by the remaining contribution ranking.
   4. Break exact ties by canonical recipe hash, then family name.

   Within each chosen family, freeze the standalone and contribution parents before proposing its batch.

   Run the first local wave. Subsequent waves may contain second local batches and qualifying broad expansions. Freeze the cross-family comparator at each wave boundary. A fast process finishing early must not change another process’s already-declared comparator.

   Count local batches against the same allowance counter. They are not free extra expansions.

   **Cost:** Minimal scheduling code. This avoids order-dependent allocation while permitting serial or parallel execution.

11. **Implement contribution and allocation records as pure, verifiable computations.**

   Add `songpop/phase2/allocation.py` and, if useful, `contributions.py`:

   ```python
   contribution_records(challenger, reference, weights) -> list[dict]
   evaluate_routes(state, candidate_records) -> dict
   advance_state(state, batch_result) -> dict
   verify_allocation(record, resolver) -> list[str]
   ```

   For each configuration, compute the actual blended predictions for:

   - `0.9 H + 0.1 challenger`
   - `0.75 H + 0.25 challenger`

   Store compact records containing parent IDs and hashes, split seeds, weights, 15 fold AUCs, three repeat means, overall mean, differences and a digest of the derived prediction arrays. Recompute them from the banks during verification.

   Materialize a normal blend bank only when needed for a decision artifact or submission. Never represent a contribution record as a fitted bank.

   Handle `challenger == H` as an explicit identity result. The existing blend function correctly rejects duplicate parent run IDs; do not fabricate a second identity to circumvent that check.

   Allocation must implement section 5.6 literally:

   - `+0.003` against the frozen incumbent.
   - `+0.001` and no more than `0.005` below frozen G.
   - `+0.0005` and no more than `0.002` below frozen G.
   - A blend improves H by `0.0005` and the previous best family blend with the same H by `0.0005`.

   Require positive repeat differences for the improvement comparisons. “Within 0.005 of G” does not mean the challenger must beat G on every repeat.

   Initialize previous-best contribution using the family anchor’s two blends, with H itself as a floor. Freeze the winning prior blend’s exact identity and weight.

   Initial progress grants two batches. Later progress resets to two; failure consumes one. Stop at zero, the unique-configuration ceiling or the task cutoff.

   Bootstrap only comparisons that actually support a decision, including the relevant blend comparisons. Do not run 2,000 bootstrap replicates for every candidate merely because the computation is available.

   **Cost:** Prediction arithmetic and AUC calculations are cheap. Bootstrap cost is bounded by decision count rather than trial count.

12. **Build score-free timing separately from the scoring engine.**

   Add a timing function that calls `fit_recipe`, predicts validation and test inputs, and records elapsed time. It must not call `run_recipe_cv`, because that function always computes AUC.

   ```python
   time_recipe(recipe, ctx, seeds, *, score=False) -> dict
   ```

   Prefer an API without a scoring option at all:

   ```python
   time_recipe_without_metrics(recipe, ctx, seeds) -> dict
   ```

   Record recipe, data/fold/source hashes, versions, thread settings, fit and prediction counts, wall time and machine-load context. Use `time.perf_counter()`.

   Time light, representative and expensive configurations, including:

   - LGB 63 leaves, 1,600 trees and small minimum leaf size.
   - RF leaf 1, all features, unrestricted depth.
   - XGB depth 6 and 1,600 trees.
   - The actual kNN representation.
   - The largest spline representation.

   A simple scaling of Finding 010’s 31-leaf, 1,200-tree LGB case gives about **73 seconds** for 63 leaves and 1,600 trees. This is my arithmetic projection, not a timing measurement or upper bound.

   For P2.2, I would retain all three split seeds even for relatively expensive configurations. The plan’s single-seed exception is optional. Activating it adds completion bookkeeping and shortlist selection that this task probably does not need.

   **Cost:** A few minutes of pilots. Using the reviewed representative cases gives approximately **27.3 minutes** for the initial 276 runs. Reserve **60–120 minutes** until the actual expensive corners are measured, plus replay and refinements.

13. **Start serial; support shards without letting workers share an append target.**

   Serial execution is the simplest reliable default. Add explicit artifact/log destinations so the common infrastructure supports later parallel runs:

   ```python
   run_batch(..., log_path: Path, bank_directory: Path) -> BatchResult
   merge_log_shards(..., expected_batch_manifest: dict) -> dict
   ```

   If a score-free serial-versus-two-process benchmark shows a useful gain, for example at least 15% lower total wall time, use two processes with four model threads each. Record that threshold before benchmarking.

   Workers write separate shards. Only the coordinator:

   - Merges records.
   - Computes allocation.
   - Writes the next batch manifest.
   - Appends the validated merged records to the main log.

   Merge checks must detect missing configurations, duplicate run identities, conflicting duplicate records, wrong source revisions, mismatched bank hashes, truncated JSON and unaccounted failures. Merge order follows manifest order, not finish time.

   Preserve shards and a merge receipt with their hashes. Resume must not append another search-start record or duplicate existing run records. A failed configuration stops its batch and remains logged.

   Do not install packages or edit fingerprinted source files during a run. Other task development should use separate worktrees based on approved code.

   **Cost:** Serial avoids concurrency overhead. Shard support is small if implemented as explicit I/O paths; a sophisticated scheduler is unnecessary.

14. **Separate artifact auditing from full computational replay.**

   Add modes such as:

   ```text
   --audit-task P2.2
   --verify-task P2.2
   --verify-batch BATCH_ID
   ```

   The audit checks every hash, schema, metric, expansion, contribution and allocation decision without refitting estimators. It is not a substitute for replay.

   Before final P2.2 review:

   - Audit all P2.2 artifacts.
   - Refit every distinct P2.2 fitted run once.
   - Recompute every contribution and decision from those results.
   - Verify phase 1 using its unchanged commands.
   - Verify the historical P2.1 anchor at its matching source revision.

   Keep `--verify-all` honest. Once phase 2 source changes, the old P2.1 bank cannot pass a current-source replay merely because its prediction arrays happen to match. Do not suppress the source mismatch or rewrite its provenance.

   Load log indexes once rather than reparsing the entire JSONL for each lookup. Reuse verified parent banks in memory when checking contribution records.

   Import all production search modules before taking the source manifest. Otherwise a lazily imported project module can change the manifest halfway through the first run. Use the same production entry point for runs and replay.

   Add golden v1 expansion tests, v2 branch/admissibility tests, zero and mixed-type refinement tests, range-limit tests, allocation boundary tests, resume/merge corruption tests and writer-disabled verification tests.

   **Important existing-code detail:** importing `scripts.check_phase2` executes its decorated checks. Do not import it merely to reuse `MemFS` in a score-free preflight script. Extract shared test support or provide a separate fixture module.

   **Cost:** Full replay costs approximately another search execution. A targeted correction should replay affected runs, followed by a complete structural audit; it need not repeatedly refit unrelated runs.

15. **Keep all fitted banks; avoid unnecessary blend-bank storage.**

   I checked the current anchor file: **441,232 bytes**.

   At approximately 440 KB per bank:

   | Scope | Approximate storage |
   |---|---:|
   | Initial P2.2, 276 banks | 121 MB |
   | P2.2 family ceilings, 644 banks | 283 MB |
   | Initial banks plus two materialized blends each | 364 MB |

   The P2.2 ceiling is **644**, not the full phase’s 794.

   Keep the fitted banks in git for this project. At this scale, changing the storage system would cost more complexity than it saves. Do not delete losing runs or keep only winners. Compact contribution records avoid the main unnecessary multiplication.

   Preserve manifests, logs and verification evidence alongside banks. Avoid committing repeatedly regenerated binaries at many intermediate revisions. No remote or external artifact service is needed.

   **Cost:** Hundreds of megabytes, not a compelling infrastructure problem on its own.

16. **Report a small number of findings and keep submission selection separate.**

   I would use two findings:

   - **Finding 011: P2.2 base-feature search results.** Counts proposed, completed, reused and failed; actual runtimes; best complete recipe per family; three repeat means; comparison with the project anchor; exact bank and manifest references.
   - **Finding 012: Complementary candidates and allocation outcomes.** Best declared contributions to H, decision-driving paired comparisons, allocation routes, stopped families and reasons.

   Label best-of-search scores exploratory. Full three-repeat evaluation is matched evaluation, not an independent confirmation after selection on those same repeats. Bootstrap intervals remain conditional on fixed predictions.

   A below-leader model can remain a useful ensemble candidate even if its family earns no further search. Allocation stopping must not delete it from consideration.

   The optional submission should be decided after complete evaluation. Predeclare:

   > Select the complete single recipe with the highest unrounded mean fold AUC among those exceeding the anchor. Break exact ties by canonical recipe hash. Report all three repeat differences. Submit only to exercise the phase 2 pipeline.

   A contribution-only qualification supports the evaluated blend, not a claim that its standalone member improved the baseline. Prepare the exact file, hash and description within the reviewed task. Sam uploads it.

   **Cost:** Small generated tables and one optional submission artifact. Report prose and Colab work remain deferred.

17. **Resolve these remaining ambiguities explicitly before execution.**

   These are implementation clarifications, not reasons to reopen the approved strategy:

   - Family anchors and the initial anchor-initialization stage.
   - H fixed for P2.2, and the population of runs eligible to define G.
   - Batch barriers and deterministic local-family selection.
   - Which comparisons need positive differences on every repeat.
   - Initialization of the previous-best blend.
   - Whether local batches consume the allocation counter. They should.
   - The exact local 9/3 split and treatment of identical parents.
   - Manhattan normalization.
   - Legal range-extension limits.
   - Counting anchors, reused centers, failures and verification fits.
   - The P2.2 exploration cutoff.
   - Historical-source verification versus current-source verification.

   Count unique proposed search recipes against trial ceilings, including failed attempted configurations. Replays and score-free timing fits are compute costs, not new hyperparameter trials.

   Keep Optuna out of P2.2. The existing random and typed local search can use the approved budget without a new installation route or adaptive-optimizer implementation.

   **Cost:** One execution document and its machine-readable counterpart. These choices should not be scattered through runner conditionals.

18. **Use a Friday-morning completion target with a firm P2.2 cutoff.**

   A workable reservation from the stated 21:05 PDT start is:

   | Time | Work |
   |---|---|
   | Thursday 21:05–23:30 | Execution protocol, common code, kNN and synthetic checks |
   | 23:30–Friday 00:15 | Score-free pilots, initial manifests, execution-readiness review |
   | 00:15–01:45 | Anchor initialization and all initial batches |
   | 01:45–03:00 | Qualifying first refinements and affordable expansion waves |
   | 03:00 onward | No new P2.2 exploratory batch; finish active work, audit and replay |
   | 04:30–06:30 | Findings and final review, with correction reserve |
   | Friday morning | Converged commit; release shared extensions to P2.3 |

   These are reservations, not a claim that implementation will take exactly that long. If readiness slips, preserve the initial family coverage and shorten optional expansion.

   Cut in this order:

   1. Extra broad expansion waves.
   2. Second local batches.
   3. Optional Manhattan/category-weight trials.
   4. Parallel execution optimization.
   5. The optional pipeline submission.

   Preserve the initial XGBoost, Extra Trees and kNN evaluations, grouping, fold-fitted preprocessing, artifact validation and replay. Do not consume Friday’s feature-engineering time trying to reach numerical ceilings.

   The Saturday 08:00 rule is an overall emergency limit. It should not become P2.2’s routine stopping time.

Only Sam needs to decide an actual re-cut of the assigned task if early infrastructure release becomes necessary, any policy waiver or deadline change, and whether to upload a reviewed submission. The technical choices above can be declared and reviewed within P2.2 without another preference round.