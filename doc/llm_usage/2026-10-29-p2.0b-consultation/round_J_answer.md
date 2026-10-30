Your corrected totals are right. My Round I arithmetic was wrong: **336 initial configurations and 794 at the ceilings**. The clean timings also support broader exploration.

The main problems are operational gaps, rather than a flaw in the basic stacking design. Some gaps originate in my Round I shorthand and should be corrected rather than preserved.

1. **Must-fix: make allocation and stopping a deterministic state machine.**  
   **Location:** [phase plan §5.6, lines 206–217](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:206).

   The current rule can repeatedly reward an old achievement. For example, successive configurations can all beat the original anchor by 0.003 while failing to improve the family’s existing best. Likewise, every batch can produce a blend that beats \(H\), but is worse than the previously found blend. The ceiling prevents an infinite loop, but the stagnation rule need never activate.

   There is also a contradiction between requiring every next batch to qualify and permitting two consecutive unsuccessful expansion batches.

   **Replacement rule:**

   > Before each batch, record the family incumbent, the strongest standalone reference and \(H\), including their run IDs and seed policies. Initial qualification grants up to two expansion batches. Thereafter, progress must come from a newly evaluated configuration. Standalone progress is measured against the pre-batch family incumbent. Ensemble progress must both satisfy the contribution threshold against \(H\) and improve the family’s previous best contribution blend against that same \(H\) by at least 0.0005, positive on every repeat. Progress resets the remaining expansion allowance to two batches. A batch without progress consumes one allowance. Stop at zero allowance, the configuration ceiling or the search cutoff.

   Keep \(H\) fixed within an expansion sequence. If a later reviewed stage changes \(H\), recompute previous contributions against the new reference before crediting further progress. Specify the first-batch family anchor in its manifest.

2. **Must-fix: restore the missing confirmation rule for expensive classical searches.**  
   **Location:** [phase plan §5.1, line 159](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:159).

   “Its finalists completed on all three” does not define which configurations receive confirmation. The explicit standalone-plus-contribution shortlist appears only for TabM. This omits part of Round I and permits a classical challenger to be discarded solely on standalone performance.

   **Add:**

   > For every seed-155 screening batch, complete the union of the best two standalone configurations, the best two by declared blend contribution and the named family anchor. Deduplicate this list. Rank using unrounded scores, with recipe ID as the deterministic tie-breaker. Reserve the completion cost before starting the batch. Apply allocation thresholds only to complete three-repeat results.

   Specify whether later batches shortlist their new configurations or the cumulative search. I recommend cumulative ranking, completing only candidates whose other repeats are missing.

3. **Must-fix: use a common reference when constructing the ensemble pool.**  
   **Location:** [phase plan §7.1, lines 273–279](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:273).

   Contributions measured against different batch-specific \(H\) predictors are not comparable. “Family leaders” could also mean only standalone winners, losing a family’s best ensemble contributor.

   An arbitrary composite \(H\) may introduce additional atomic recipes beyond the stated limit of eight.

   **Replacement rule:**

   > P2.6b chooses one common reference \(H_\ast\): the strongest eligible standalone recipe under a declared ranking rule. It recomputes contribution checks against \(H_\ast\) on identical folds and declared seed policies. For each family, retain separate standalone and contribution winners; these may be different recipes. Choose the three standalone family representatives and up to two additional contribution representatives from these records. Count every transitive ingredient of every procedure, including \(H_\ast\), toward the eight-atomic-recipe limit.

   Replace “Before any ensemble score is computed” with “Before the corresponding P2.6b procedure scores are computed.” P2.2 already computes ensemble scores.

4. **Must-fix: define the channels supplied to learned combiners.**  
   **Location:** [phase plan §7.1–7.2, lines 273–287](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:273).

   The pool is described, but the learned procedures’ input matrix is not. It is unclear whether it contains all eight atomic recipes, five family representatives, the family average, or both the average and its constituents. Those choices change the model and the regularization.

   **Proposed rule:**

   > Learned procedures use one channel per selected family representative. Where the declared within-family average exists, it replaces that family’s single-recipe channel. Its constituents are not also separate channels. The original anchor remains a standalone candidate and is not automatically an extra meta-feature. Freeze the ordered channel definitions and all procedure IDs before evaluating those procedures.

   Also declare how missing optional families and duplicate recipes are handled. The “at most 32” limit should apply to an enumerated manifest, not an open-ended description.

5. **Must-fix: timing-based menu changes must happen before the menu freeze.**  
   **Location:** [phase plan §8, lines 299–301](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:299), versus [§11, lines 322–323](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:322).

   Section 8 allows removing learned-combiner members before seeing outer scores. That could occur Saturday night, after P2.6b has frozen the exact procedures. A timing-based change still changes the predictor.

   **Replace with:**

   > All timing-based ingredient removals occur in P2.6b before its review and the Saturday 18:00 PDT freeze. Changed procedures receive new IDs and complete production evaluations before admission. After the freeze, candidate recipes do not change for runtime reasons. P2.7 may activate only the predeclared reduced evaluation path, selected without inspecting outer scores.

   Define the reduced-path trigger numerically, such as projected build plus replay exceeding the eight-hour reservation. Prevent outer scores from being printed before that decision. “Before results are inspected” should not depend on someone voluntarily ignoring terminal output.

6. **Must-fix: state the cache assumptions underlying the nested cost estimate.**  
   **Location:** [phase plan §7.2 and §8, lines 287–299](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:287).

   The \(3.2C_{\rm all}+16.2C_L\) estimate assumes that different combiner settings share the same underlying meta-training predictions. Merely saying “one explicit partition” does not ensure this. Candidate-specific split seeds could multiply the cost.

   **Add:**

   > All learned candidates evaluated on the same fitting population use the same saved combiner-training folds. These folds depend on the population and protocol coordinates, not the candidate name or regularization setting. \(C_L\) counts the union of atomic recipe and meta-split-protocol pairs.

   Strengthen the cache identity to include data hashes, fitting labels, grouping, ordered prediction populations and all fitting-context seeds. Row IDs alone do not distinguish changed inputs.

   **Verification rule:**

   > A full replay starts with an empty prediction cache. It may share newly recomputed fits across candidates within that replay. Reusing the original run’s cached predictions does not count as recomputation.

   Describe the formula as a fit-equivalent projection, not a runtime bound. My independent arithmetic gives **113–145 minutes**, or **4.71–6.04 hours** with replay and the stated margin. The rounded numbers in the draft are acceptable.

7. **Must-fix: repair the task dependencies created by shared search infrastructure.**  
   **Location:** [task table, lines 28–32](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:28), [line 181](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:181), and [§6.1, line 231](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:231).

   P2.3 is independent of P2.2, but P2.2 owns the search extensions and all additional factory parameters. For example, the committed EBM factory does not yet allow `learning_rate`. P2.3’s declared search therefore needs unreviewed P2.2 work unless ownership changes.

   P2.4 also includes neural preprocessing experiments without depending on P2.5.

   **Exact correction:**

   - P2.2 owns common search extensions and installed-family factory changes.
   - P2.3 owns CatBoost/EBM factory changes. Its installation and timing preparation may proceed independently. Its searches wait for any common P2.2 extensions they use to converge.
   - P2.4 uses the TabM wrapper or results only after P2.5 converges. Otherwise it completes its classical scope and omits neural comparisons.
   - Serialize `.venv` installations. Parallel tasks must not race the before/after inventory checks.

   My Round I recommendation to remove P2.3’s dependency was too broad once shared extensions were assigned to P2.2.

8. **Must-fix: add a later-infrastructure subsection rather than leaving section 4 stale.**  
   **Location:** [phase plan §4.1, lines 55–58](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:55), §4.4–4.8, and §13.

   The current layout still says `nested.py` arrives in P2.7. The new task table puts it in P2.6a. Section 4 still describes only fitted-model and fixed-blend banks and submissions. Several safeguards proposed in Round I were not carried into the amendment.

   Add **“4.10 Extensions after P2.1”**, explicitly preserving P2.1’s approved scope. Assign P2.6a:

   - Learned-procedure schemas, prediction banks and recursive deployment validation.
   - Generic internal populations and fold contexts, separate from production eligibility.
   - The selector and nested evaluator.
   - Raw-input prediction and feature-lineage support.
   - Relevant dependency identity, including SciPy for weight optimization.
   - Label-isolation tests at every level.
   - Rejection of caches from a different fitting population.
   - Exactly-once meta-OOF coverage and group exclusion.
   - Duplicate consistency and rejection of compensating member corruption.
   - Separate full-refit and fold-average artifacts.

   State that section 13 records historical staging, now superseded where section 14 reallocates the work.

9. **Must-fix: close the neural budget’s overflow and double-counting gaps.**  
   **Location:** [phase plan §5.5, lines 196–200](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:196), and [§6.1, lines 231–236](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:231).

   Reducing eight screens to four does not guarantee the three-hour reservation is sufficient. Four screens plus four three-repeat completions can cost \(12q\). The plan also permits neural preprocessing and retuning within P2.4’s 128 configurations, potentially bypassing TabM’s 16-configuration ceiling.

   **Add:**

   > All distinct TabM configurations across P2.4 and P2.5 count toward the same 16-configuration ceiling and compute allowance. Neural trials do not receive an additional 128-configuration retuning allowance. Before starting a batch, reserve its screens, required completions and final estimator-seed evaluation. If four screens plus confirmation exceed the available allowance, choose a smaller declared shortlist or record omission before running the batch. A promising partial-seed result is not a finalist.

   Fix the one-seed policy explicitly to seed `0`; the three-seed policy is `{0,1,2}`.

10. **Must-fix: correct the timing script’s read-only invocation.**  
    **Location:** [analysis/phase2_timing.py, line 4](/Users/zwl0331/Desktop/Mini_project_1/analysis/phase2_timing.py:4).

    `-I` ignores Python environment variables, including `PYTHONDONTWRITEBYTECODE`. I verified:

    ```text
    isolated=1
    ignore_environment=1
    dont_write_bytecode=False
    ```

    Therefore the documented invocation can write bytecode on import.

    **Replace the verification command with:**

    ```bash
    .venv/bin/python -B -I analysis/phase2_timing.py .
    ```

    Show output redirection separately as the author’s artifact-generation command. I used `-B -I` for my rerun.

11. **Must-fix: record timing provenance and the performance information already exposed.**  
    **Location:** [analysis/phase2_timing.py, lines 90–101](/Users/zwl0331/Desktop/Mini_project_1/analysis/phase2_timing.py:90), [phase plan line 155](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:155), and `doc/findings.md`.

    The output records Python and hardware, but not data hashes, split hash, source manifest, package versions or complete resolved recipes. It also publishes new AUCs, including feature-block results, before the planned search manifests exist.

    **Fix:**

    - Add a draft **Finding 010: Phase 2 runtime measurements**. Name it explicitly in the P2.0b review.
    - Record the script and imported-source hashes, data/split hashes, package versions and exact case recipes.
    - Distinguish 15 outer fit positions from estimator fits. The three-estimator-seed case has 45 estimator fits.
    - Label the existing AUCs as timing-probe exposures. Preserve that history and state that they do not select recipes or allocate batches.
    - Suppress AUC output in future timing-only pilots.
    - Do not describe already exposed probe configurations as prospectively unseen experiments.

    The older round-E coverage percentages should remain identified as unreviewed until P2.4 recomputes them. Approval of this amendment should not implicitly convert every cited consultation number into a reviewed finding.

12. **Should-fix: distinguish representative timings from actual planned-recipe costs.**  
    **Location:** [phase plan line 155](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:155), [timing script lines 71–87](/Users/zwl0331/Desktop/Mini_project_1/analysis/phase2_timing.py:71).

    The kNN timing uses eleven standardized continuous columns. It does not include the proposed categorical block or block weighting. Its neighbor counts also differ from the planned grid. It is a useful proxy, not a timing of the proposed representation.

    The most expensive timed LightGBM case has 31 leaves and 1,200 trees. The search allows 63 leaves and 1,600 trees.

    **Replacement wording:**

    > These are representative case measurements. The 30 to 60 minute estimate extrapolates from those cases and is not an upper bound for the enlarged spaces. Each task times its actual preprocessing and its declared expensive configuration before fixing the executable batch reservation.

    Remove or qualify “an earlier run of the same script.” The earlier unrestricted HistGB measurement necessarily used a different threading setup. Cite its exact source instead of implying identical execution.

13. **Should-fix: constrain range extensions and reconcile the Optuna allowance.**  
    **Location:** [phase plan §5.2, lines 177–180](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:177).

    “Within one grid step of a range edge” can trigger extensions almost everywhere in short grids. “One step” is undefined for `None`, `"sqrt"` and irregular lists. It can produce invalid fractions above one. The Optuna clause also proposes 32 evaluations while local refinement is capped at 24.

    **Replace with:**

    > Range extension applies only to ordered numeric parameters. The next batch manifest states its exact new values, legal hard bounds and maximum number of extensions before fitting. Categorical values have no range edge. Fractions remain in their valid domains. At most one outward extension per numeric parameter is allowed during this stage.

    Then choose one unambiguous Optuna rule:

    > An Optuna batch may contain up to 24 refinement configurations, within the existing family ceiling.

    Alternatively, call a 32-trial TPE batch an expansion batch and explicitly exempt it from the separate 24-trial refinement allowance. Do not leave both interpretations.

14. **Should-fix: enumerate the fixed feature checks and EBM interactions.**  
    **Location:** [phase plan line 175](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:175), [line 255](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:255).

    “Each of the first five blocks” is ambiguous because one row represents multiple target encodings. “The three declared pairs of v0.1” requires consulting historical versions.

    **Replace the nine checks with this exact list:**

    ```text
    date_count
    date_te10
    date_te30
    date_te10 + date_count
    date_te30 + date_count
    interactions
    cyclic
    era_relative
    inverse_group_size training weights
    ```

    List EBM’s interaction pairs explicitly: `(year, loudness)`, `(year, acousticness)`, `(energy, loudness)`. Resolve their column indices after the recipe’s feature construction.

    Restore the explicit target-encoding contract: **five** inner grouped folds; `fit_transform` for fitting rows; `transform` for external rows. This is currently delegated to “as in v0.1” despite rewriting the surrounding specification.

15. **Should-fix: finish the numerical and seed specifications for ensembles.**  
    **Location:** [phase plan lines 280–287](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:280).

    The constrained-weight objective lacks a numerical convention for zero or one probabilities. The general three-seed policy also needs an explicit exception for the original anchor.

    **Add:**

    > The constrained objective uses row-weighted mean binary log loss. Its numerical clipping threshold is fixed in the procedure manifest. Prediction output remains the declared convex combination. Optimizer initialization, tolerances and failure handling are fixed before evaluation.

    > The original anchor retains estimator seed 0. Its three-seed version is a separate candidate. Deterministic estimators are fitted once. All other declared seed policies are applied consistently to production CV, nested evaluation and deployment.

    Freeze complexity ranks before the corresponding P2.6 evaluations, not merely before Sunday. Define whether “supervised feature transformations and learned combiners” means their sum or a lexicographic pair.

16. **Should-fix: record the additional departure concerning TabM’s ensemble role.**  
    **Location:** [phase plan §5.5, line 200](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:200).

    The draft excludes TabM from learned combiners unconditionally. Round I advised against automatically placing a *costly* neural model inside one. Those are different rules.

    I would use:

    > TabM may enter a learned combiner only if the measured complete evaluation and replay fit the reserved budget. Otherwise it is evaluated through fixed probability blends. This choice is frozen before the affected procedure scores.

    A blanket exclusion is also defensible, but record it as a deliberate amendment and ensure section 7’s channel construction excludes TabM consistently. It was not adopted “as written.”

17. **Should-fix: separate upload eligibility from exploration allocation.**  
    **Location:** [phase plan §10, line 314](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:314).

    A configuration can qualify for more search through blend contribution while scoring below the anchor alone. Section 10 nevertheless proposes uploading “the best confirmed single recipe” when it improves through a section 5.6 route.

    **Replace with:**

    > After P2.2 or P2.3, an eligible complete three-repeat single recipe may be submitted to exercise the phase 2 pipeline when its standalone mean fold AUC exceeds the anchor’s and its repeat differences are reported. A contribution-only qualification supports submitting the corresponding evaluated blend, not claiming a standalone improvement. Uploads remain optional and purpose-driven.

    This keeps allocation, publication purpose and final selection separate.

18. **Should-fix: add an executable schedule and finish the consistency cleanup.**  
    **Location:** [task table and cut order](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md:23), §8, §11, §14; strategy §§5.5–6.

    No whole model family needs to be cut now on the observed costs. The missing reservation is P2.6b’s production evaluation, final seed policies, regeneration and replay before its Saturday 14:00 review request.

    Add:

    > No new exploratory batch starts after Saturday 08:00 PDT. By 10:00, inputs required by P2.6b have converged or their declared optional omissions are recorded. P2.6b reserves measured time for final seed evaluation, learned-procedure production CV, regeneration and verification before its 14:00 review request.

    With shared fits, a conservative production-evaluation projection is approximately \(C_{\rm all}+6C_L\), before replay and overhead. Include it alongside Sunday’s estimate. If it does not fit, stop expansions earlier rather than compressing review.

    Also make these small corrections:

    - Strategy §5.5 still says averages of “two to four” models, while the new plan permits all selected channels. Point to the new declared pool.
    - Qualify strategy line 163’s author-time statement as historical phase 1 guidance. Phase 2 targets include review.
    - Explicitly mark section 12’s old numerical limits as historical, rather than current executable rules.
    - Update the section 4 staging references from P2.6/P2.7 to P2.6a/P2.6b where appropriate.
    - The P2.0b review-record path does not yet exist. Until it does, use “draft, under consultation” rather than claiming formal review is underway. Link Sam’s full messages to the existing Round I prompt.
    - Add the consultation README, Round J response and integration-index entry before formal review.
    - Split the long algorithm paragraphs in §§7.2 and 8 into numbered steps. I found no en/em dashes in the changed project plans.

What I checked and found correct:

- The counts are **276 + 60 = 336**, with a **794** pre-feature ceiling. Feature comparisons and retuning add at most **224**, giving **1,018** configurations before neural trials and final seed-policy evaluations.
- The full timing rerun completed using `.venv/bin/python -B -I analysis/phase2_timing.py .`. Representative results were anchor **7.41 s**, large LightGBM **26.55 s**, random forest **8.00/26.57 s**, Extra Trees **10.52 s**, XGBoost **6.59 s**, and HistGB **4.55 s**. All printed probe AUCs matched the saved output to four decimals. These are consultation checks, not reviewed findings.
- Section 7.2’s basic learned-procedure construction is sound. Meta-training predictions exclude held-out groups, preprocessing is rebuilt locally, and averaging complete fitted procedures is coherent.
- The third nested level is present. The selection-optimism and training-size limitations are stated correctly.
- The kNN grid has exactly **48** configurations. Its proposed block scaling is sensible.
- Finding 009’s reviewed status and `e00287a531af` bank ID agree with P2.1’s convergence record.
- The data restrictions, pretrained-model exclusion, unchanged anchor, three-repeat finalist requirement, report/Colab deferral and both freezes remain intact.