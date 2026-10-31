1. **Must change: audit G’s external bank, not just its log entry.**  
   Locations: [protocol.py:229](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/protocol.py:229), [run_experiments.py:231](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:231), [run_experiments.py:504](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:504).

   `prior_strongest` checks task, recipe ID, the *recorded* bank hash and completeness. Neither planning nor task audit reads that external bank to validate its scores. `audit_task` audits only P2.3’s planned runs.

   In a read-only, in-memory probe, I changed the prior record’s mean and repeat scores while retaining its identifiers and hash. `protocol.Log` accepted it. That corrupted comparator could govern allocation.

   **Fix:** add a runner-level `validate_prior(ctx, proto)` before planning, execution and audit. It should:

   - Load the bank and check its bytes against the protocol’s full hash.
   - Compare the full recipe hash, task, eligible population, groups, folds and seeds.
   - Run `runs.audit_run` to reconcile the log’s scores with stored predictions.
   - Reject missing or corrupted dependencies before fitting.

   Keep `protocol.Log` pure. The external bank needs a read-only audit, not a refit under P2.3’s changed sources. Add tests for corrupted scores, missing bank bytes and changed fold identity.

2. **Must change: refresh the timings and bind reservations to specific timing records.**  
   Locations: [run_experiments.py:160](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:160), plan §17.5.

   All six recorded P2.3 timings have source hashes that differ from the current files for:

   ```text
   songpop/phase2/engine.py
   songpop/phase2/models.py
   songpop/phase2/protocol.py
   ```

   This matters particularly because prediction threading changed. The timing specification itself still matches.

   Also, `reservation()` averages **every** timing record with the matching task and family. Rerunning the pilots would mix old and replacement measurements. Three records with the same case label would satisfy its current count check.

   **Fix:** after computation changes settle, rerun the six score-free pilots. Select exactly one record for each `(family, case)` through an explicit timing-set identity containing the specification hash, source manifest, recipe, seeds and environment. Reject duplicates or mismatched inputs. Store those timing-record identities and the protocol hash in the reservation. Retain old records as history.

3. **Must change: correct the EBM training-count claim.**  
   Locations: [check_phase2.py:721](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:721), [check_phase2.py:738](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:738), plan §17.2.

   “Exactly `max_rounds` rounds for every term” is incorrect with greedy boosting. The installed `_boost.py` sets:

   ```python
   max_steps = max_rounds * len(term_features)
   ```

   Greedy steps can update one term more often than another. The existing assertion checks total steps per stage, which is appropriate; its description overstates what it establishes.

   **Replacement text:**

   > With validation and early stopping disabled, each stage performs `max_rounds × number_of_stage_terms` boosting updates. Greedy selection can distribute those updates unevenly across terms. The runtime check verifies the stage totals.

   Change the test name and printed result accordingly. No model-setting change is needed.

4. **Must change before readiness: put the installation and timing evidence into Finding 018.**  
   Locations: `doc/findings.md`, plan §17.6.

   The ledger currently promotes Findings 014–017 but does not contain the proposed Finding 018. Compatibility and timing conclusions are already being used to justify the executable search.

   **Fix:** add Finding 018 as `draft` before readiness. Cover inventory transitions, pinned versions, runtime checks, current-source timings, warnings and limitations. Have the readiness record explicitly name the finding it reviews. Findings 019 and 020 can remain completion deliverables.

   Separate score-free compatibility evidence from the later claim that every searched configuration replayed successfully.

5. **Should change: define “replay exactly” consistently with the verifier.**  
   Locations: plan §17.6, [banks.py:171](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/banks.py:171).

   The plan promises exact CatBoost and EBM replay. `compare_v2` accepts prediction and AUC differences up to `1e-12`. The tiny runtime tests require bit equality, but they do not establish bit equality for every searched configuration.

   **Recommended replacement:**

   > P2.3 uses the existing classical verification criterion: identities and stable metadata match exactly; predictions and AUCs match within absolute tolerance `1e-12`. H must match all ten reference arrays bit for bit. The P2.2 forest exception does not apply.

   If bit equality is genuinely required for CatBoost and EBM, implement and test that stricter P2.3 check instead. Resolve this before observing replay discrepancies.

6. **Should change: extend the new protocol tests beyond the current favorable branch.**  
   Location: [check_phase2.py:1380](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:1380).

   The synthetic test establishes initial G resolution and CatBoost refinement around candidate 31. It does not cover several important new paths.

   Add score-free synthetic tests for:

   - EBM winning the sole refinement slot while CatBoost receives a broad expansion.
   - A closed-wave P2.3 run overtaking the external prior as G.
   - Current-wave results leaving that wave’s comparators unchanged.
   - The reference family receiving no later batch or refinement slot.
   - EBM exhausting its admissible broad space and producing a smaller batch.
   - External-bank corruption being rejected through the actual runner and audit paths.

   Preserve the historical P2.2 replay comparison as a regression check. I independently re-planned **waves 1, 2 and 3** against their saved manifests and found no differences.

7. **Should change: strengthen the runtime fixture and narrow the filesystem claim.**  
   Locations: [check_phase2.py:635](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:635), [check_phase2.py:649](/Users/zwl0331/Desktop/Mini_project_1/scripts/check_phase2.py:649).

   My descriptive probe found **1,449 full dates and 51 year-only dates** in the training fixture. Neither its training rows nor its prediction rows include `1979-06`.

   **Fix:** explicitly include the month-only row in a prediction fixture, plus full-date and year-only rows. Assert all three precision cases are present. This does not require another CV.

   The file check compares only immediate directory names. It would miss an overwrite or a file created inside an existing directory. Either strengthen it for expected output locations, including `catboost_info`, or describe its actual evidence:

   > No new top-level project entry appeared; CatBoost file writing and snapshots were explicitly disabled.

   Also change the script’s blanket “everything runs in memory and writes nothing” claim to distinguish project artifacts from library temporary files. EBM’s joblib backend attempted to obtain a temporary directory during my permitted check.

8. **Should change: remove the false compute bound and specify admission of expensive expansions.**  
   Locations: [run_experiments.py:179](/Users/zwl0331/Desktop/Mini_project_1/scripts/run_experiments.py:179), plan §§17.4–17.5.

   The reservation says full replay costs “at most the ceiling figure,” while the same record correctly says that figure is not an upper bound. Moreover, permitted extensions reach depth 8 and 3,000 CatBoost iterations, beyond the timed expensive case.

   **Fix the record text:**

   > Full replay is projected to repeat the fitting actually completed. The ceiling projection is not a runtime bound.

   Before admitting an expensive extension, reserve its execution, outstanding replay, checks and completion review. Use a score-free pilot when it materially exceeds the timed configurations.

   If this can stop exploration before noon, declare a logged budget-stop rule and make `completion_problems` recognize it. Otherwise the current verifier treats an otherwise justified early stop as incomplete. Do not introduce an informal stopping exception after seeing scores.

9. **Should change: make the handoff fail closed and carry its provenance.**  
   Locations: [p23_summary.py:34](/Users/zwl0331/Desktop/Mini_project_1/analysis/p23_summary.py:34), [p23_summary.py:49](/Users/zwl0331/Desktop/Mini_project_1/analysis/p23_summary.py:49), [p23_summary.py:86](/Users/zwl0331/Desktop/Mini_project_1/analysis/p23_summary.py:86).

   The summary can print `H check: MISSING` and continue toward a handoff. It also assumes every nonempty family has complete runs and contribution pairs. That is fragile after an interruption.

   **Fix:**

   - Refuse a handoff without a passed H check and valid completion state.
   - Audit H, the external reference and exported member banks without refitting.
   - Reject incomplete contribution pairs explicitly.
   - Include `recipe_sha256`, bank path, source-manifest hash, split seeds and estimator seeds.
   - Include H’s bank hash in the contribution entry.
   - Give unavailable families an explicit reason.

   Keep the declared standalone tie rule, lower recipe ID. Keep the contribution tie rule, canonical recipe hash then weight. They differ deliberately in the existing plan; label them rather than silently unifying them.

10. **Optional: call the CatBoost anchor “default-inspired.”**  
    Location: plan §17.4.

    The fixed `0.0168` rate is a rounded rate associated with an approximate training size. It is a valid anchor, but “CatBoost’s defaults for this training size” suggests more exact reproduction than the record establishes.

    **Replacement:**

    > The CatBoost anchor is a fixed, default-inspired configuration. Its learning rate is fixed at `0.0168` across folds; it is not recalculated from each fold’s training size.

    I accept MVS with explicit subsample `0.8`, the reference-family implementation of H, continuous EBM features, and inherited off-list parent values. None requires reversal merely because it differs from Round O.

I checked the inventory transitions: CatBoost added exactly CatBoost, graphviz and plotly; EBM added exactly interpret-core. No previously installed version changed. In-memory wave-1 planning produced **1, 48 and 12 unique configurations**. The permitted self-check command reported **37 passed, 1 failed, 4 staged**. The failure was EBM’s inability to obtain a temporary directory in this sandbox; I cannot claim its runtime check passed here. CatBoost’s runtime check and the exact anchor reproduction passed. I computed no P2.3 search score and changed no file.

**Only Sam’s decisions:** none is needed for these corrections. Changing preserved deadlines or resuming work after the requested P2.4 pause remains Sam’s decision.