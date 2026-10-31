My recommendation is to build one population-local fitting engine, then use it for production CV, learned procedures, and nested evaluation. Keep production eligibility and deployment validation outside that engine.

This is a provisional design consultation. I read the requested sources and inspected logged record counts. I changed nothing, fitted nothing, computed no scores, and did not read the withheld design or the P2.5 consultation. `STATE.md` still carries the earlier pause; the latest entry in [decisions.md](/Users/zwl0331/Desktop/Mini_project_1/decisions.md) confirms the newer instruction to finish P2.4, P2.5, and P2.6, then pause.

1. **Separate fitting populations from production eligibility.**

   Add `songpop/phase2/populations.py` with three small immutable contracts:

   - `TrainingPopulation`: raw feature rows indexed by permanent training-row ID, labels, primary groups, and data provenance.
   - `PredictionPopulation`: namespace, ordered IDs, raw feature rows, and input digest. It has no labels.
   - `FoldContext`: protocol version, full nesting coordinates, split assignments or their references, and deterministic seeds for each fitting purpose.

   Preserve primary group IDs when taking subsets. Do not rebuild groups from a subset or silently renumber permanent row IDs. Distinguish training-row IDs from Kaggle test IDs with namespaces.

   Add a generic CV runner in `engine.py` that receives these objects explicitly. Keep `runs.Context` as the production adapter. The recursive fitting code must not receive `runs.Context`, a repository root, or the full dataset.

   Preserve the existing seed derivation for ordinary production recipe fits. Extend the context for nested and meta fits with the complete ancestry, rather than reusing `(split_seed, fold)` at every depth. Seeds must depend on protocol coordinates and population identity, not candidate ID, lambda, C, or labels outside the permitted fitting population.

   **Reason:** [FitContext](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/encoders.py:20) currently identifies only one CV level, while [runs.Context](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/runs.py:40) exposes the complete production dataset. Also, [bank eligibility](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/banks.py:74) currently follows split-seed completeness. An internal bank can have all three seeds without being a production candidate.

2. **Implement learned procedures with a deliberately narrow grammar.**

   Add `procedures.py` supporting:

   - An atomic recipe.
   - A fixed probability average of atomic recipes.
   - A constrained-weight combiner or logistic stacker over atomic or fixed-average channels.

   Do not support learned-on-learned stacking or arbitrary recursive graphs in this task.

   Use an interface such as:

   ```python
   fit_procedure(spec, D, context, cache) -> FittedProcedure
   fitted.predict_high(raw_rows)
   ```

   For a learned procedure fitted on \(D\):

   1. Obtain one saved five-fold `StratifiedGroupKFold` partition of \(D\). Its construction uses only \(D\)'s labels and groups. All settings and channels fitted on this same \(D\) use the same partition.
   2. Allocate an \(N_D \times J\) meta-feature matrix and an integer coverage array.
   3. For each meta fold, fit each required atomic recipe on its training complement. Rebuild target encoding, preprocessing, and any neural stopping entirely inside that complement.
   4. Average estimator seeds within each atom. Then apply any fixed within-channel averaging.
   5. Write predictions into the held-out rows by permanent ID. Require coverage exactly one for every row and channel.
   6. Fit the combiner on this matrix and \(y_D\).
   7. Fit each channel on all of \(D\), then use those channel predictions for new rows.

   Require each meta training side to be the exact complement of its held-out side. The existing [inner-split validator](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/encoders.py:31) checks disjointness and coverage, but does not explicitly require that complement property.

   The meta matrix has one row per training observation. Do not concatenate the three production repeats into three copies of the observations.

   **Reason:** This implements [section 7.2](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:315) directly and makes the exclusion boundary testable.

3. **Allow exact reuse of the channels’ full-\(D\) fits, but never use production OOF predictions as meta-training data.**

   When \(D\) is one production outer-training population, an atomic production fold fit already trained on exactly \(D\) may supply the learned procedure’s full-\(D\) channel predictions. Reuse requires equality of:

   - Fitting rows, their labels, groups, and raw inputs.
   - Recipe, estimator seeds, stopping policy, and effective fitting context.
   - Prediction populations and their ordered inputs.
   - Executing sources and relevant dependencies.

   Make the standalone fit and the learned channel’s full-\(D\) refit request the same canonical atomic-fit context. The container procedure’s ID must not change the atomic fit.

   The current banks save predictions, not fitted predictors. Thus an old bank can supply only predictions for populations it actually contains, after provenance validation. It cannot supply a raw-input predictor for arbitrary later queries. Prefer sharing through the new cache during final regeneration instead of building a permissive historical-bank importer.

   Never slice the global OOF bank to make \(D\)'s meta matrix. Many of those predictions were produced by models trained using groups outside \(D\).

   **Reason:** Sharing identical full-\(D\) work is valid and valuable. Sharing a superficially similar OOF vector changes the training procedure and leaks across the evaluation boundary. The relevant existing boundary is [fit_recipe](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/engine.py:65).

4. **Freeze the exact combiner mathematics and numerical behavior.**

   For constrained weights, use uniform observation weights, consistent with the row-weighted evaluation target:

   \[
   L(w)=
   -\frac{1}{N_D}\sum_i
   \left[y_i\log \tilde p_i+(1-y_i)\log(1-\tilde p_i)\right]
   +\lambda\sum_j(w_j-1/J)^2,
   \quad
   \tilde p_i=\operatorname{clip}((Zw)_i,10^{-6},1-10^{-6}).
   \]

   Constraints are \(w_j\geq0\) and \(\sum_jw_j=1\). Prediction returns the convex combination, without the objective’s probability clipping.

   My concrete solver proposal is SciPy SLSQP, equal initialization, bounds `[0, 1]`, an explicit equality constraint, analytic gradient consistent with clipping, `ftol=1e-12`, and `maxiter=2000`. Record all these settings. Validate success, finiteness, feasibility, and objective value against the equal-weight starting point. Declare any tolerance-sized cleanup precisely, retain both raw and used weights, and reject larger corrections. Keep zero-weight channels in the artifact and complexity accounting.

   A failed optimizer produces a failed procedure record. It must not silently become an equal blend, change optimizer, or relax tolerances after seeing scores.

   For the stacker:

   - Clip channel probabilities to `[1e-6, 1-1e-6]`.
   - Convert to logits.
   - Fit `StandardScaler` on this meta-training matrix only.
   - Fit L2 logistic regression with intercept, `solver="lbfgs"`, `C` in `{0.01, 0.1, 1}`, no class balancing, uniform row weights, a fixed tolerance and iteration ceiling. I suggest `tol=1e-10`, `max_iter=5000`.
   - Treat convergence warnings as failures.
   - Extract numeric class 1 explicitly.

   Save scaler mean, variance and scale, coefficients, intercept, class ordering, and convergence information. A constant meta column uses the scaler’s declared constant-column convention.

   **Reason:** “Constrained weights” and “regularized stacker” are insufficient reproducibility specifications. These choices must be frozen before their evaluations, as required by section 7.2.

5. **Give learned procedures their own artifact, while retaining comparable prediction arrays.**

   Keep the ten-array `PredictionBank` interface from [evaluate.py](/Users/zwl0331/Desktop/Mini_project_1/songpop/evaluate.py:16). A production learned-procedure bank still has:

   - OOF shape `(3, 3863)`.
   - Fold AUC shape `(3, 5)`.
   - Test predictions shape `(3, 5, 863)`.
   - A test mean equal to the mean of the 15 complete procedure predictions.

   Add explicit schemas for internal evaluations and learned-procedure artifacts. Do not manufacture a fake atomic recipe to satisfy `bank_v2` validation. Use schema dispatch in the phase 2 loader and verifier while retaining support for old banks.

   A `learned_procedure_v1` artifact should contain:

   - Canonical procedure specification and ordered channels.
   - Transitive atomic specifications and final seed policies.
   - Each fitting population and context.
   - Saved meta folds expressed as permanent row IDs.
   - Meta predictions, coverage evidence, and their atomic prediction references.
   - Combiner state and diagnostics for every outer fit.
   - Full-\(D\) channel prediction references.
   - Averaging order, raw-source feature lineage, source and dependency identities.
   - Separate logical fit counts, executed fit counts, and cache-hit counts.

   Keep cache-hit counts and elapsed time out of stable computational identity.

   Validate every fold’s AUC from the procedure’s OOF probabilities. Learned weights vary across outer folds; a learned procedure is not a `blend_banks` call with one global weight vector.

   Default deployment is the average of the 15 fitted procedures. For a stacker, apply each fold’s stacker to that fold’s channels before averaging. Averaging coefficients or averaging channels before applying one stacker is a different predictor.

   **Reason:** This preserves compatibility with paired comparisons while exposing the state needed to verify the whole learned procedure.

6. **Build a content-addressed cache of atomic prediction blocks.**

   Add `cache.py`, with immutable entries under a path such as:

   ```text
   artifacts/phase2/cache/<execution-namespace>/<key>.npz
   ```

   Use separate fit and prediction identities. The complete lookup key must include:

   - Data provenance hashes.
   - Canonically ordered fitting IDs, aligned labels, groups, and raw inputs.
   - Exact atomic recipe and resolved parameters.
   - Estimator seeds and every effective fitting-context seed.
   - Relevant split assignments and stopping policy.
   - Ordered prediction populations, including their namespaces, IDs, and raw-input digests.
   - Source manifest, Python, relevant package versions, and dependency-lock hashes.

   The prediction-input digest matters: a later permutation analysis may use the same IDs with changed raw features.

   Store the canonical request beside the arrays and validate it against the caller’s request on every hit. Also check array shapes, ID order, finite probabilities, hashes, and receipts. A file found under the expected filename is not automatically trusted.

   Reject mismatched or corrupt entries. Do not repair an identity mismatch by slicing, sorting, padding, or relabeling its predictions. A different valid request produces a different key.

   Source, dependency, data, recipe, seed, or context changes invalidate reuse by changing identity. Old entries can remain immutable. A malformed existing entry should produce an explicit failure, not an unexplained successful cache miss.

   Allow an in-memory fitted-object cache for arbitrary raw predictions. Persistent model serialization is optional and can be omitted.

   Full replay gets an empty cache namespace. It may share work generated during that replay, but must not read the original run’s cache.

   **Reason:** The cache is part of the statistical isolation boundary. Population size, fold number, and recipe ID alone are inadequate keys. SciPy is already present in [the package inventory](/Users/zwl0331/Desktop/Mini_project_1/songpop/artifacts.py:46); preserve it and add the transitive optional dependencies of every procedure.

7. **Make selection a pure function over restricted evidence.**

   Implement:

   ```python
   select_candidate(evidence_by_id, menu) -> SelectionResult
   ```

   Each evidence object contains only candidate identity, the permitted evaluation population, labels, groups, folds, OOF predictions, and validated fold scores. It must not expose query predictions, public scores, outer labels, or a loader for global banks.

   The function requires the exact frozen candidate set and complete `(3, 5)` evaluation for each candidate. For production, validate the actual official population and saved folds. For nested selection, validate the actual outer-training population and rebuilt inner folds. Do not use the production `selection_eligible` flag to authorize internal evidence.

   Compute unrounded mean fold AUC. Form the tie set relative to the single maximum:

   ```text
   maximum_mean - candidate_mean <= 0.001
   ```

   Then choose the minimum tuple:

   ```text
   (
       unique_atomic_recipe_count,
       supervised_feature_transformation_count,
       learned_combiner_count,
       complexity_rank,
       candidate_id,
   )
   ```

   Return the maximum, tie set, structural counts, chosen ID, and evidence hashes. Do not implement a pairwise “approximately equal” sort comparator, because the tie relation is not transitive.

   **Reason:** [section 11](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:377) specifies a deterministic rule, not an informal judgment of stability or runtime.

8. **Implement the nested evaluator and the deployment decision separately.**

   `nested.py` should receive the frozen menu and nested protocol. Its coordinator constructs:

   | Level | Permitted training information | Output |
   |---|---|---|
   | Outer evaluation | Three grouped folds, seed `20261101` | Held-out assessment |
   | Inner selection | Only one outer-training population; five folds for each of `155, 2026, 1029` | Candidate selection evidence |
   | Learned-procedure training | Only one inner-training population; five meta folds | Combiner training matrix |
   | Atomic fitting | Only its assigned fitting rows | Encoders, preprocessing, estimator, stopping state |

   Target encoding and TabM stopping remain local within the final row of this table.

   For each outer fold, build every candidate’s inner evidence. Query predictions for the outer validation rows may be computed while models are live, but keep them separate from the selector’s evidence view. After selection, retrieve the chosen candidate’s 15-model inner-fold average.

   Separately refit the selected complete procedure on all outer-training rows. A learned candidate regenerates its meta matrix and combiner there. Also produce the anchor’s inner-fold-average prediction.

   Only after all three folds finish should a scorer join the outer labels and predictions. Then call a pure decision function:

   - Fallback when `mean(selected_fold_average - anchor) <= -0.003` and all three differences are negative.
   - If fallback triggers, recommend the original anchor with fold-average deployment.
   - Otherwise, allow full refit only when `mean(selected_full_refit - selected_fold_average) >= 0.003` and all three differences are positive.
   - Otherwise retain fold averaging.

   Different candidates may be selected in the three outer folds. These are outcomes of the frozen selection procedure, not three votes for a new candidate.

   **Reason:** This matches [section 8](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:336). In particular, a refit advantage cannot be transferred to an anchor selected by fallback.

9. **Implement the reduced path before the expensive path is launched.**

   Record the runtime-based path decision before computing or printing outer scores.

   If the projected complete nested build and replay exceed the eight-hour reservation, use the declared fixed-candidate check:

   - Choose the candidate once using ordinary production CV and the frozen selector.
   - Fully refit that fixed candidate and the anchor separately in each of the three outer-training populations.
   - Rebuild a learned candidate’s meta-training predictions locally.
   - Report this explicitly as a fixed-candidate check.
   - Disable both fallback and deployment-switch decisions.

   A runtime failure after inspecting outer results must not become an opportunity to switch protocols or simplify candidates.

   Report fold AUCs, their mean, pooled outer OOF AUC, selected IDs where applicable, and paired differences. Keep the conditional bootstrap caveat and the earlier menu-adaptation caveat. Once the outer results trigger a deployment or fallback decision, they are not untouched evidence for that resulting choice.

   **Reason:** The reduced path preserves the plan’s evaluation requirement without pretending to evaluate the same selection procedure.

10. **Use measured fit requests to control the eight-hour budget.**

   P2.6a’s pilot should exercise the actual recursion and cache on synthetic fixtures and the anchor, without candidate scores. Measure atomic fits, meta construction, optimizer work, serialization, and duplicate validation. Demonstrate that changing lambda or C creates no additional atomic fitting requests.

   P2.6b must then time the actual admitted recipes at their final estimator-seed policies. The existing logged candidate runs all use seed 0, so their elapsed times are not final-policy cost measurements.

   Retain the plan’s conservative projections:

   \[
   T_{\text{production}}\approx C_{\rm all}+6C_L,
   \qquad
   T_{\text{nested}}\approx3.2C_{\rm all}+16.2C_L.
   \]

   Also enumerate the actual unique fitting requests. That catches costs hidden by approximate formulas, especially stopping-stage fits and distinct meta protocols.

   Budget:

   ```text
   1.25 × (build + empty-cache replay + validation + bootstrap + I/O) <= 8 hours
   ```

   Use measured costs under the intended process and thread settings. Start with one orchestration process and the established thread limits. Avoid adding parallel fitting merely to make an optimistic projection fit.

   Share atomic work across settings, fixed averages, and identical full-\(D\) requests. Before the freeze, reduce expensive learned-channel membership if necessary. After the freeze, candidate ingredients do not change for runtime reasons.

   **Reason:** Findings 010 and 018 show substantial runtime variation and machine-load effects. A reservation based only on nominal fold counts is not a reliable schedule.

11. **Replace the staged checks with tests that demonstrate specific failures.**

   The placeholders at [check_phase2.py:2207](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:2207) must become executable P2.6a checks. Use synthetic data, instrumented estimators, and the existing in-memory filesystem.

   | Test | Deliberately wrong implementation it must reject |
   |---|---|
   | Outer-label isolation, all assignments fixed | Outer labels reach fitting, selection, stopping, or random-seed construction |
   | Inner-validation isolation | An inner held-out label changes that fold’s fitted procedure predictions |
   | Meta-fold isolation | Changing a meta-held-out group’s labels changes that group’s base meta predictions |
   | Fit ancestry | A nested fit contains any ID or group outside its permitted parent training population |
   | Exactly-once meta coverage | Duplicate assignment overwrites another prediction, a row is missing, or a channel uses a different partition |
   | Whole-group exclusion | A duplicate track’s other row appears in the training side |
   | Complement validation | Meta training silently omits extra rows |
   | Cache refusal | Same recipe and fold number reuse a different population, labels, context, query order, or query inputs |
   | Seed and channel averaging | Flattening estimator seeds changes declared recipe weights |
   | Optimizer checks | Negative weights, failed convergence, or an undeclared equal-weight fallback pass |
   | Selector boundary cases | Rounded scores, chained ties, partial banks, or forged eligibility change the winner |
   | Deployment separation | Full-refit predictions are accepted as a fold-average bank, or vice versa |

   Include positive controls showing that changes to permitted training labels can affect predictions. At the meta level, test the excluded group’s base predictions, not the final combiner: the combiner legitimately trains on all labels in \(D\).

   Extend the existing [compensating-corruption test](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:929) through within-family averages and learned procedures. Corrupt two members so their final output stays unchanged. Validation must still reject the individual members. Test cancellation across folds and estimator seeds as well.

   **Reason:** A successful end-to-end run does not establish isolation. These tests fail on plausible shortcuts that otherwise produce credible-looking scores.

12. **Define the transition to final seeds before defining H* and the final pool.**

   There is an important underspecification in section 7.1: the historical inventory has seed-0 recipes, while stochastic classical finalists require seeds `[0, 1, 2]`.

   My recommendation is a bounded, predeclared nomination stage:

   - Freeze the complete usable discovery inventory and its bank hashes.
   - Identify a common discovery reference from those complete results.
   - Recompute discovery contribution comparisons against that one reference. Do not compare contributions against different historical H values.
   - Nominate each family’s standalone winner and contribution winner, the anchor, and the leading-family configurations eligible for the within-family average.
   - Complete these nominated configurations under their declared final seed policies.
   - Define H* as the strongest eligible standalone among this completed final-policy inventory, with the plan’s recipe-ID tie-break.
   - Recompute final contribution comparisons against H*, then construct the eight-atom pool.

   Call H* “the strongest eligible final-policy nominee.” Do not imply every historical configuration was rerun with three estimator seeds. The R2 packet should explicitly approve this eligibility definition.

   Final seed policies:

   - Original anchor: `[0]`.
   - Anchor seed average, if evaluated: a distinct predictor with `[0, 1, 2]`.
   - Other stochastic classical finalists: `[0, 1, 2]`.
   - Demonstrably deterministic estimators: one fit.
   - TabM: its already declared policy, chosen from measured cost, never its best single seed.

   The final pool follows the plan: three strongest standalone family representatives; up to two additional contribution representatives; anchor; up to two additional leading-family configurations within `0.002` of its leader. Deduplicate exact atomic configurations and enforce the transitive limit of eight.

   Allow an additional contribution representative to be a different configuration of an already represented family. Otherwise the rule can discard exactly the distinction illustrated by Finding 012, where Extra Trees’ standalone and contribution winners differ. Record this interpretation explicitly.

   **Reason:** This is finite, reproducible, and affordable. It avoids silently choosing H* under one seed policy and deploying or blending it under another.

13. **Freeze an enumerated procedure manifest before evaluating its procedures.**

   Use a construction protocol and a separate enumerated manifest, for example under `configs/phase2/p2.6/`. The former defines nominations and pool construction; the latter lists the actual procedures.

   Enumerate:

   - Equal averages of the two and three strongest representative channels.
   - Equal average of all representative channels.
   - `0.9 H* + 0.1 challenger` and `0.75 H* + 0.25 challenger` for every distinct admitted challenger.
   - The declared within-family equal average.
   - Constrained weights with lambda `0`, `0.01`, `0.1`.
   - The stacker with C `0.01`, `0.1`, `1`.

   Deduplicate structurally identical procedures and retain aliases explaining which construction rules produced them. Assert the limit of 32.

   Resolve channel membership explicitly. My proposed rule is one learned channel per represented family: use its selected contribution representative when present, otherwise its selected standalone representative. The leading-family average replaces that family’s channel. All admitted atomic challengers still receive their declared fixed contribution comparisons, so the channel reduction does not discard their contribution evidence.

   Fix channel order before evaluation. Rank “top two” and “top three” using the declared representative ranking, not a new ranking discovered from procedure scores.

   TabM may remain in fixed blends while being excluded from learned channels by the pre-score cost decision. The same applies to another expensive channel. Every changed channel list gets a new procedure ID.

   **Reason:** This resolves the plan’s representative/channel ambiguity without an adaptive sequence of ensemble experiments. Findings 012, 016, and 020 justify testing contribution separately from standalone strength; they do not establish the winning pool against the eventual H*.

14. **Make complexity accounting and the frozen menu mechanically auditable.**

   Introduce an `atom_id` separate from the existing recipe hash. The current [recipe hash](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/recipes.py:93) includes estimator seeds. Atomic complexity should identify the exact feature, preprocessing, weighting, model, and training configuration while excluding estimator-seed multiplicity, human name, and deployment wrapper. Fit identity still includes all of them.

   Count supervised feature transformations as distinct reachable, label-fitted feature maps in the declared procedure graph:

   - `date_te30` counts as one.
   - `date_te10` and `date_te30` count as two.
   - An identical date map shared across two models counts once when its input and cross-fitting policy are identical.
   - Date count, imputation, scaling, unsupervised splines, and fixed interactions count zero.
   - A supervised feature selector or label-based binning step would count.
   - Base-estimator training and early stopping are not feature transformations.
   - The stacker or constrained combiner counts under learned combiners, not again under feature transformations.

   Count logical transformations, not repetitions across folds or seeds. Do not decrease counts because a learned coefficient happens to be zero.

   Assign every candidate a fixed integer `complexity_rank` before its procedure scores. A sufficient deterministic policy is a frozen ordering by procedure kind, structural specification, and regularization preference, with stronger regularization first within otherwise identical learned settings. Publish the resulting table. Do not derive rank from runtime or observed AUC.

   Construct the menu with the required anchor, selected standalone leaders, within-family average when present, best fixed blend, best constrained procedure, and best stacker. Use the same declared selection rule within each procedure class. Fill remaining places by repeated application of the frozen rule, to at most 12. There is no need to fill every slot.

   `configs/phase2/menu.json` should contain exact specifications, candidate IDs, structural counts and ranks, seed policies, production-bank hashes, construction-manifest hashes, source and environment identities, the selector, the nested protocol, the reduced-path rule, and the cost reservation.

   **Reason:** The tie-break is consequential. Its inputs must be derived from declared structure and independently checkable, rather than manually assigned after looking at nearly tied scores.

15. **Complete deployment validation and the review packet before the handoff.**

   Extend [submission.py](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/submission.py:71) from flat members to the restricted procedure graph. Validate atomic leaves, fixed channels, learned outputs, and the final average separately.

   Retain per-estimator fit-time duplicate checks. At submission time, validate actual fitted-input equivalence, the independently enumerated 11 repeated-URI pairs, and raw audio/date equality for every applicable member and composite. For TabM, use its actual fitted preprocessing contract; do not assume the existing classical reconstruction in `member_input_classes` is sufficient.

   Full-refit deployment needs a separate `deployment_v1` artifact under `artifacts/phase2/deploy/`. It records the full fitting population, procedure identity, combiner state if applicable, prediction IDs, and provenance. Never replace a bank’s `test_mean` with full-refit predictions.

   Extend [prepare_candidate.py](/Users/zwl0331/Desktop/Mini_project_1/scripts/prepare_candidate.py:26) with a versioned selection manifest that names the deployment kind and exact artifact hash. Keep existing `selection_v1` behavior for historical files. Generate the description from the actual deployment graph.

   Extend [verification dispatch](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/runs.py:249), `--compare`, and `--blend` as appropriate. A learned replay must rebuild its base fits and meta predictions, not merely refit its combiner on saved meta features. Source or dependency mismatches remain failures. Earlier runs retain their replay associations.

   P2.6b prepares at most one ensemble-path submission, selected by the frozen rule, with its purpose recorded before any public score. Omit it if it adds no useful deployment check. Every file still receives R1 before Sam uploads it.

   The R2 review should receive the final source commit; construction and procedure manifests; seed-transition rule; pool and omission records; complete regenerated finalist banks; test and replay evidence; complexity table; measured cost reservation; nested and reduced-path protocols; and the exact fallback/deployment decision function. Include the relevant draft findings and the exploratory-score caveat. An optional submission’s R1 is separate from this R2.

   **Reason:** The freeze is a concrete reproducible package. A document describing the intended menu is insufficient for P2.7 to depend on.

16. **Protect the evaluation and verification work when the schedule slips.**

   Follow Sam’s ordering: finish P2.4 first. Then complete P2.5 or its permitted omission, build P2.6a, and evaluate P2.6b. Keep executing source revisions immutable; integrating P2.5 must precede the final implementation revision and finalist regeneration.

   Within P2.6a, implement in this order: population/context contracts, atomic cache, learned fitting, pure selector, nested runner, deployment separation, failure tests, and cost pilot. Avoid optional generality, persistent model serialization, a parallel scheduler, and learned-on-learned support.

   Within P2.6b, reserve final seed evaluations and learned-procedure replay before spending time on extra combinations. Aim to have input handoffs by Saturday 10:00, the concrete review packet by 14:00, and the frozen menu by 18:00. If those targets slip, preserve verification and review time.

   Apply the existing cut order:

   1. Expansion batches without recent progress.
   2. Weak feature combinations and redundant removal checks.
   3. Extra configurations inside the within-family average.
   4. Expensive learned-combiner members, retaining eligible fixed-blend use.
   5. Unpromising neural refinements.
   6. A learned procedure whose correct complete evaluation cannot fit.

   Never substitute global-bank stacking for a removed learned procedure. Never cut grouped cross-fitting, phase 1 verification, submission validation, or the declared nested evaluation path.

   At the P2.6 pause, leave P2.7 ready to launch with its exact manifest, command, cost projection, empty-cache replay configuration, and expected completion time. Do not launch it automatically.

   **Reason:** The fastest reliable route to Saturday night is a smaller frozen hypothesis set with complete evidence. Sam’s current authorization ends at P2.6.

Only Sam can decide:

- Whether to resume after the P2.6 pause and launch the reserved P2.7 computation.
- Whether to authorize additional large compute or a material departure from the frozen protocol.
- Whether to upload the optional reviewed file, and which two submissions become the final Kaggle selections.