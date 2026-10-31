The main readiness blockers are the audit’s incomplete coverage, unfrozen reference inputs, and missing runtime-stop controls. The new preprocessing appears consistent with fold-local fitting.

Locations below refer to the supplied `p24dev` copy. I fitted nothing, calculated no model score, and wrote nothing. I used file inspection and pure or mocked control-flow probes. The copy changed during inspection; the newly added exclusion of unsupported re-tuning families is now present, so I have excluded that earlier problem from the findings.

1. **Must change: make the audit require the records promised by each manifest.**  
   **Location:** `scripts/run_experiments.py:1004`, `audit_features`; `FeatureLog:680`.

   The audit verifies comparison and contribution records that exist. It does not establish that all required records exist. Deleting every contribution record, or a matched-comparison record, can therefore escape these loops. `FeatureLog` also silently overwrites duplicate allocation records in its dictionary.

   A mocked, score-free probe confirmed that `audit_features` accepts an empty task after the cutoff.

   **Fix:** derive the expected record sets from manifests and require:

   - Exactly one initial feature batch.
   - Contiguous batch indices and valid stage transitions.
   - One start and one allocation record for each completed batch.
   - Every declared run, matched comparison and pair of contribution records.
   - A passed H check.
   - Explicit, permitted omission records for unstarted batches.
   - No duplicate allocations, orphan starts or unexplained records.

   Check event ordering and start timestamps against the cutoff. Passing the cutoff must not turn “nothing executed” into successful completion. Add deletion and duplication tests for each required record type.

2. **Must change: freeze and audit the external reference pool.**  
   **Location:** `scripts/run_experiments.py:662–728`, `pool_entry`, `FeatureLog.before`, `feature_pool`.

   `before()` filters P2.4 batches, but `feature_pool()` always rereads the current complete runs of P2.2, P2.3 and P2.8. It therefore does not fully reconstruct the information available when a batch was planned.

   Also, `pool_entry()` takes standalone means, repeat means, feature names and task identity from the log without `runs.audit_run`. `load_logged()` verifies bank bytes and run identity, but does not establish that these log fields agree with the bank. Those fields directly affect aliases, `G`, incumbents and unit selection.

   **Fix:**

   - After P2.3 converges, save an immutable reference inventory containing run IDs, bank hashes and bound run-record hashes.
   - Audit every admitted reference against its bank.
   - Derive eligibility, features and metrics from the validated bank.
   - Freeze the prior P2.4 batch and allocation records used by each new manifest.
   - Make replanning consume these inventories, not an unrestricted live task query.

   Add a test that appending a later reference-task run cannot change an earlier manifest.

3. **Must change: validate imports against the original comparison definitions, not only against themselves.**  
   **Location:** `scripts/run_experiments.py:731`, `validate_imports`; `feature_stage.py:29`, `load_protocol`.

   The receipt hashes the original manifest, but the validator does not bind each receipt entry to that manifest’s B1–B14 definition. A different valid comparator, with its matching bank hash and recipe, can pass because route calculation does not use the matched comparator. Missing imports are also not rejected by the schema. A pure probe confirmed that an empty import list loads successfully.

   **Fix:** require exactly B1–B14, once each. Resolve their original comparator and challenger definitions from the hashed P2.8 manifest. Check the receipt’s family, recipes, run IDs and hashes against those definitions. Bind the receipt’s H to the manifest’s named H and audit it too.

   Add tests for a missing comparison, duplicated ID, substituted valid comparator, swapped comparison IDs and substituted H.

4. **Must change: add a recorded runtime stop that can close feature exploration and permit re-tuning.**  
   **Location:** `scripts/run_experiments.py:763`, `feature_closed`; `plan_features:872`; plan §§18.6 and 18.9.

   The stated cut order is not executable. If another expansion is due, `plan_features` refuses re-tuning. There is no recorded budget stop. Waiting until the cutoff does not help because it then refuses to plan re-tuning too.

   Moreover, one retuning “batch” contains all selected units. Starting that round just before 04:00 permits every unit to finish, regardless of its cost. The earlier cutoff alone does not protect verification and review.

   **Fix:** add an immutable `feature_stop` or equivalent record with:

   ```text
   stage, last_closed_batch, reason
   timing-record references
   remaining fitting and replay estimate
   completion-review reservation
   omitted expansions or units
   ```

   Recognize this record in planning, execution and audit. Admit retuning units using their measured costs before saving the round manifest. Require enough remaining time for fitting, complete replay and completion review.

   The referenced `configs/phase2/p2.4/timing.json` is currently absent. Create it with the actual post-P2.3 aliases and record the reservation before readiness. The existing `--reserve` path loads a family-wave protocol and does not implement a P2.4 reservation.

5. **Must change: exclude exhausted date bases before applying the two-base limit.**  
   **Location:** `songpop/phase2/feature_stage.py:219–234`, `expansion_candidates`.

   The code takes the first two bases before checking whether either has any untried variants. Those bases can occupy the limit forever.

   A pure planning fixture confirmed that two exhausted qualifying bases leave the date queue empty while an untried XGBoost base is available.

   **Fix:** generate applicable, unseen variants for each candidate base first. Discard exhausted bases, then choose the first two remaining bases.

   Apply the same principle to removals at lines 250–260. The text promises the two strongest recipes holding the relevant columns. The code chooses two recipes first and only then checks their columns. Select applicable holders separately for each removal specification.

6. **Must change: give the smooth-model expansion checks matched comparators.**  
   **Location:** `configs/phase2/p2.4/protocol.json`, `expansion.smooth_knn`; `feature_stage.py:242–247`.

   SK1–SK3 compare the original F1 spline model with F2 plus another block. They cannot isolate the contribution of date count, target encoding or cyclic features beyond F2. SK4 simultaneously changes scaling and the distance metric relative to the original kNN base.

   **Fix:**

   - Compare SK1–SK3 against the B29 recipe, F2 with `spline_calendar`, keeping model parameters fixed.
   - Compare robust Manhattan kNN against B31, robust Euclidean kNN.
   - Resolve these comparator recipes by declaration even if their earlier scores were poor.
   - Keep any additional comparison against the original F1 model explicitly labelled as a bundled change.

   This requires a comparator recipe or comparison-reference field, rather than restricting every menu item to an original alias.

7. **Must change: record bootstraps for the comparisons that actually authorize further work.**  
   **Location:** `scripts/run_experiments.py:928–996`, `judge_retune`, `run_features`.

   Feature batches bootstrap challenger versus matched base. Their allocation routes can instead depend on a different incumbent or on two blend comparisons. Retuning records no such bootstrap at all.

   Section 5.6 requires descriptive bootstraps for decision-driving comparisons. The existing `bootstrap_records` implementation at line 437 provides the relevant pattern.

   **Fix:** define deterministic feature-stage witnesses and record:

   - Challenger versus frozen incumbent for a standalone route.
   - Challenger blend versus H and versus the previous best family blend for an ensemble route.

   Persist these before closing the allocation. Require and recompute them during audit. Keep the matched-effect bootstrap separately. None becomes a significance gate.

8. **Must change: deduplicate re-tuning by computation, and choose distinct parent configurations.**  
   **Location:** `feature_stage.py:399`, `plan_retune`; `scripts/run_experiments.py:830–837`.

   Re-tuning uses raw recipe hashes, whereas P2.2 uses normalized search keys. A pure probe confirmed that `reg_lambda: 1` and `reg_lambda: 1.0` have different raw hashes but the same existing normalized search key.

   The second parent is also required to have only a different **run ID**. An imported recipe and its P2.4 support refit can become two supposedly distinct parents despite having identical parameters.

   **Fix:** use the existing family-normalized computation key consistently for exclusions, proposal deduplication and parent distinctness. Preserve recipe hashes as provenance identifiers. Do not change historical recipe hashing.

   Add tests covering omitted defaults, integer-versus-float representations and two runs of the same configuration under different sources or names.

9. **Must change: finish the resume and preflight checks before writing or returning success.**  
   **Location:** `scripts/run_experiments.py:891–899`, `plan_features`; `run_features:958–984`; `feature_preflight:906`.

   Three gaps remain:

   - Replanning an existing batch calls `save_feature_batch` before comparing with its logged declaration. A mismatch can write an orphan manifest before failing.
   - An already-judged batch returns before preflight.
   - H’s fit is outside the failure-logging and returned-run-ID checks used for other runs.

   **Fix:** calculate and compare the expected manifest bytes and identity in memory before any save. Run preflight before an “already judged” return. Route H through the same checked run wrapper as other recipes.

   Also validate batch ID derivation, stage, index, kind and official split seeds explicitly. Reject zero or negative CLI indices with an argparse validator; the current truthiness dispatch does not handle zero as an invalid planning request.

10. **Must change: replace the vacuous assertion and test the feature controller end to end.**  
    **Location:** `scripts/check_phase2.py:1665–1745`.

    This assertion cannot fail:

    ```python
    assert queues == sorted(queues, key=list(fs.QUEUES).index) or True
    ```

    Simply removing `or True` is insufficient: released slots can legitimately put an earlier queue after later queues.

    The comment also claims a diversity-cap check, but the test only compares two unit IDs. It does not exercise `choose_units`’ cap. The four checks do not exercise feature-controller execution, interruption recovery, audit completeness or historical replanning.

    **Fix:** use handcrafted fixtures with explicit expected candidate sequences, including released slots. Add an in-memory event-log test covering:

    - Initial planning and execution.
    - Interruption before and after bank finalization.
    - Expansion, retuning and replay.
    - Missing and duplicate records.
    - Future-record isolation.
    - Exhausted leading queues.
    - Two units from one family and the family cap.
    - Unsupported retuning families.

    These control-flow tests can use spy runners and synthetic entries. They need no search scores.

11. **Must change: distinguish a reviewed family omission from an unfinished P2.3 log.**  
    **Location:** `feature_stage.py:120–144`, `resolve_aliases`; plan §18.4.

    The current rules interpret “no CatBoost run” as permission to use XGBoost, and “no EBM run” as permission to omit B19. That is convenient in the copy but unsafe as a production gate. A partial or incorrectly loaded P2.3 log is not a reviewed omission.

    **Fix:** production protocol resolution must consume P2.3’s completed handoff and its convergence reference. Permit fallbacks only when that handoff explicitly records the corresponding omission and reason. Otherwise fail.

    Keep the copy’s fallback behavior confined to synthetic planning fixtures. Regeneration alone does not enforce this distinction because the protocol stores dynamic alias rules.

12. **Should change: preserve the exploration allowance for re-tuning, or record the stricter rule as a departure.**  
    **Location:** plan §18.6; `scripts/run_experiments.py:825–827`.

    Round 2 currently requires round 1 to make progress. This is stricter than section 5.6’s two-opportunity allowance and my round Q recommendation. The departure is not explained.

    My recommendation is:

    > A qualifying feature unit receives up to two retuning batches. An unsuccessful first batch consumes one opportunity. The second remains available unless the recorded runtime guard omits it. No unit exceeds thirty-two configurations.

    If Claude retains the stricter rule, identify it explicitly in section 18 and reconcile it with sections 5.6 and 6.1. Do not describe it as unchanged application of the counter.

13. **Should change: reconsider the blanket exclusion of logistic and kNN re-tuning.**  
    **Location:** plan §18.6; `feature_stage.py:355`, `choose_units`.

    The new family filter correctly prevents unsupported units from crashing. The stated reason for excluding these families is nevertheless weak: selected preprocessing can remain fixed while model parameters are re-tuned.

    **Fix:** allow a qualifying smooth unit to tune `C`, and a qualifying kNN unit to tune neighbor count and weighting while keeping its representation fixed. These can use estimator-only spaces and the same overall allowance.

    If omitted for implementation time, record that as a schedule cut. Do not present preprocessing support as requiring their exclusion. The omitted kNN cyclic alternative should likewise be recorded as deferred, rather than appearing implicitly covered by `knn_blocks_v2`.

14. **Should change: complete the promised coverage output.**  
    **Location:** `analysis/p24_date_coverage.py:59–75`; plan §18.8.

    The outer validation and test summaries include support bins and distinct-date coverage. Inner held-out summaries retain only row coverage, despite the section promising the same support information for inner dictionaries.

    Per-repeat validation coverage is also calculated as an unweighted average of fold percentages. That is a mean-fold statistic, not pooled row coverage when fold sizes differ.

    **Fix:** retain and print the full inner summaries. Save numerators and denominators so pooled row coverage can be reported separately from mean-fold coverage. Preserve the current explicit mean-over-fits label where appropriate.

    The script’s use of the encoder’s inner seed and fitting-only support dictionaries is otherwise consistent with the intended diagnostic.

15. **Should change: correct status text and specify completion evidence before readiness.**  
    **Location:** plan header, task table, §§18.1, 18.9 and 18.10.

    “Started after P2.3 converged” is currently false for this draft. The header remains v0.7 without the P2.4 addition, and the task-table record remains “as above.”

    **Fix:** state that the draft was developed in an isolated copy while P2.3 executed, and that production execution will use P2.3’s converged revision. Update the version history and explicit review-record path when porting.

    Before readiness, include the actual timing file, reservation, finalized reference inventory and handoff gate. At completion, require:

    - All expected manifests, comparisons, contributions, witnesses and allocations.
    - Full replay of every distinct new P2.4 run.
    - H equality and unchanged phase 1 verification.
    - Historical replay associations for imported records.
    - Findings distinguishing matched effects from allocation decisions.
    - The usable-candidate handoff index, including contributors with weaker standalone scores.
    - Recorded omissions and the requested pause.

16. **Optional: cache validated pool entries within an invocation.**  
    **Location:** `feature_pool`, `pool_entry`, `validate_imports`.

    Planning and audit repeatedly load the historical pool and rebuild its two contribution blends. After freezing and validating the inventory, cache entries by bank hash and H hash for that invocation. This reduces repeated work without changing the protocol. Do not cache across changed hashes or skip the initial validation.

The smoothing registrations, generalized one-target-encoding restriction, F1-compatible spline construction and audio-only robust scaling look sound by inspection. The existing preprocessing checks provide useful coverage of matrix and prediction compatibility. I did not run that suite because it fits models.

**Decisions only Sam can make**

No new Sam decision is needed for these fixes. Changing the freezes, overriding an unresolved review issue, or resuming work after the P2.4 pause remains Sam’s decision.