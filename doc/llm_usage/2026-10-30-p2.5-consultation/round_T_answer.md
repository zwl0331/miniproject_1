1. **Blocking: an interrupted worker exchange can return another request’s result.**  
   **`songpop/phase2/tabm.py:306–323`, `serve:273–296`.**

   `Worker.call` leaves the worker and its stream reusable after `KeyboardInterrupt`, a serialization error or an interrupted read. Replies have no request identity.

   I reproduced this with an in-memory fake worker: interrupt the first read, issue a second request, and the second call accepts `reply_to_first_request`. With real requests, this can associate predictions or fitted state with the wrong invocation.

   **Fix:** give every request and response a protocol version and request ID. Serialize exchanges with a lock, or explicitly reject concurrent calls. On any exception during transmission or reception, invalidate and terminate that worker before propagating the exception. Validate the response ID and schema before accepting its contents. Never continue using a partially consumed stream.

2. **Blocking: worker reads and shutdown can hang indefinitely.**  
   **`songpop/phase2/tabm.py:311–316`, `326–329`.**

   Neither receiving a response nor waiting for termination has a timeout. In particular, closing stdin and calling `wait()` can deadlock if the worker is writing a large fitted-state response into an undrained stdout pipe. This matters during cancellation and interpreter shutdown.

   **Fix:**

   - Apply a deadline to the entire exchange, including writes.
   - On timeout or interruption, close the pipes and terminate the worker.
   - Use bounded `wait`, followed by `kill` and another bounded wait.
   - Reap exited workers and close their descriptors before replacing them.
   - Send a formatted traceback with handled worker errors. Preserve the exit code for crashes.
   - Check for a parent torch import before starting an exchange as well as after receiving it.

   Add mocked tests for a stalled worker, EOF, interruption during a response, and shutdown with an outstanding large response. These tests should not require torch.

3. **Blocking for any TabM submission: the submission validator cannot rebuild its preprocessing.**  
   **`songpop/phase2/models.py:201–218`; `songpop/phase2/submission.py:65–66`.**

   `make_model` recognizes TabM, but `make_preprocessing` does not recognize `tabm_numeric`. The submission validator calls the latter.

   My pure probe produced:

   ```text
   ValueError: unknown preprocessing 'tabm_numeric'
   ```

   **Fix:** return `TabMPreprocessor(prep["scaler"])` from `make_preprocessing` for this name. Give its `fit` method the compatible signature `fit(X, y=None)`, ignoring `y`. Keep preprocessing owned by the wrapper during neural training.

   Test the actual `member_input_classes` and submission-validation path. Verify that rebuilt float32 inputs equal the wrapper’s `transform_inputs`, and that tampering with a duplicate prediction is rejected.

4. **Blocking before execution: the neural compute allowance is not enforced by the reused controller.**  
   **`scripts/run_experiments.py:181–221`, `467–530`; plan §19.3, line 826.**

   The reservation prints an initial and ceiling projection. Execution checks the calendar cutoff and allocation counter, but never checks the three-hour neural allowance, a six-hour extension, or reserved replay time.

   Consequently, a productive initial batch can automatically launch both refinements even when their fitting and replay no longer fit the reservation. Section 19 describes an initial-size check, while §5.5 requires reservations before each batch.

   **Fix:** add a P2.5-specific, logged budget decision before every batch. Bind it to the timing evidence, completed elapsed time, planned recipes, remaining fitting allowance, replay reservation and handoff deadline. A six-hour allowance must record the existing productivity and Sunday-feasibility conditions.

   Also provide an audited `budget_stop` outcome. Otherwise the completion audit can demand another wave even after the execution budget has legitimately closed the task. Keep old tasks’ behavior unchanged when this optional policy is absent.

5. **Blocking if the initial batch is reduced: the controller can exceed the declared two refinement batches.**  
   **`analysis/p25_make_protocol.py:56–66`; plan §19.2, line 822; `protocol.plan_wave`.**

   Section 19 permits reducing the initial eight configurations. With an initial four, ceiling sixteen and two four-configuration refinements, twelve configurations have been used. If allowance remains, the generic controller can launch a fourth wave of broad expansion.

   That contradicts “at most two local refinement batches.” It can also activate the generator’s wider hard bounds:

   ```text
   learning_rate: 0.0001 to 0.01
   weight_decay: 1e-7 to 0.1
   ```

   These exceed the declared neural space.

   **Fix:** make the timing reduction an explicit generator option with a predetermined subset. Set the resulting ceiling to `initial_count + 8`, or enforce a separate maximum of two expansion batches. Set hard bounds to the declared neural ranges and disable outward extension for this task. Add a synthetic-log test in which all reduced-batch waves make progress.

6. **Blocking robustness defect in selecting G: a complete blend can be selected as a standalone recipe.**  
   **`analysis/p25_make_protocol.py:44–49`.**

   `protocol.complete` does not distinguish fitted recipes from blends. `strongest` includes every complete run from the listed tasks, then accesses `bank.metadata["recipe"]`.

   A synthetic complete blend ranked above a fitted run causes `KeyError: 'recipe'`. More fundamentally, G is supposed to be the strongest **standalone recipe**.

   **Fix:** filter to fitted or estimator-seed-average runs with an atomic recipe. Validate the selected record against its bank and production identities. Declare and use the established recipe-based tie rule rather than the human run ID.

   Make protocol generation refuse to overwrite a protocol already bound to P2.5 timing or batch records. Validate the generated object before writing it.

7. **Non-blocking: neural receipts are recorded but not structurally validated.**  
   **`songpop/phase2/tabm.py:253–262`; `songpop/phase2/banks.py:127–132`.**

   The bank validator still checks only the number of outer receipts. It accepts a TabM receipt with a missing estimator seed, an impossible chosen epoch, inconsistent update counts or incorrect preprocessing-population hashes.

   Full replay should detect differences, but the lightweight audit currently cannot substantiate these receipt claims.

   **Fix:** add TabM-specific receipt validation:

   - Exactly one receipt per declared estimator seed, in order.
   - Matching stopping-policy version and derived seeds.
   - Nonempty finite loss history, with length equal to `epochs_evaluated`.
   - `chosen_epoch` and `stopped_by` recomputed with `stop_rule`.
   - Legal epoch bounds.
   - Full preprocessing-row hash equal to the enclosing fitting-row hash.
   - Correct stopping and final update counts.
   - Inner population hashes checked against the grouped split when auditing with the dataset context.

   Add deliberate receipt-corruption tests.

8. **Non-blocking: worker provenance omits a dependency and the actual execution settings.**  
   **`songpop/phase2/models.py:23`; `tabm.py:250–262`, `273–277`; `runs.py:82–96`.**

   TabM imports `rtdl_num_embeddings`, but its version is absent from the neural package identity. The receipt records torch and TabM versions, but not the worker’s actual dtype, device, inter-op thread count or deterministic-algorithm setting.

   The parent environment describes the classical process, not the worker that performed the neural calculation.

   **Fix:** include `rtdl-num-embeddings` in the family dependencies. Add a worker initialization handshake reporting plain strings and primitive values for:

   ```text
   protocol version
   tabm.py source digest
   Python and relevant package versions
   device and dtype
   intra-op and inter-op threads
   deterministic-algorithm setting
   ```

   Check these against the expected configuration. Keep PIDs and request IDs out of stable bank identity. Explicitly construct CPU float32 networks.

   The current plain-string conversion of `torch.__version__` is correct.

9. **Non-blocking for the declared F2 search: preprocessing has three reproducible edge defects.**  
   **`songpop/phase2/tabm.py:73–116`.**

   My synthetic preprocessing probes found:

   | Input | Current behavior |
   |---|---|
   | F0 with quantile scaling | Crashes because `std_` is empty and `SimpleImputer` receives zero columns |
   | Entirely missing `key`, observed `mode` | The imputer drops `key`; `output_columns()` incorrectly labels mode’s outputs as `key=0`, `key=1` |
   | Entirely missing quantile-transformed audio column | Outputs approximately −5.199337, contrary to the documented zero behavior |

   **Fix:** build and transform only nonempty numerical branches; retain empty categorical columns with an explicit imputation convention; track constant or empty quantile columns and apply the declared zero-output rule. Assert that `output_columns()` matches the transformed matrix’s width and column ownership.

   Alternatively, explicitly reject unsupported recipe shapes before fitting. The present validator advertises more support than the preprocessor provides.

10. **Non-blocking, but important before accepting the repeatability claim: the cross-process self-check compares receipts, not predictions.**  
    **`scripts/check_phase2.py:2461–2465`, `2509–2517`.**

    The child checks repeatability within itself. The parent compares its receipt with the child’s receipt, then checks that two estimator seeds differ. It never compares the parent worker’s predictions against the child’s predictions.

    Thus a deterministic error in state transport or worker prediction could pass the test described as establishing cross-process equality.

    **Fix:** return the child’s prediction array or its canonical binary digest, and compare it with the parent wrapper’s predictions. Also compare a prediction before and after an unrelated worker fit, then after closing and restarting the worker.

    Add two other targeted tests:

    - A hand-computed member-loss or gradient oracle that fails if training uses the loss of averaged probabilities. The current inference check does not test this.
    - Quantile stopping-stage isolation, with held-out-only extreme values and categories. The present fitting-ID spy exercises standard preprocessing.

    Use synthetic fixtures containing explicit mixed-label duplicate groups.

11. **Non-blocking: training checks the loss before an update, but not the resulting gradients or final state.**  
    **`songpop/phase2/tabm.py:221–227`, `248–249`.**

    An invalid final optimizer update can escape the training-loss check. Some infinite logits produce finite probabilities of zero or one, so checking probability finiteness alone is insufficient.

    **Fix:** reject non-finite gradients after `backward` and non-finite parameters after training. Validate every serialized state array before returning it. This can fail the configuration normally; it should not trigger an undeclared optimizer fallback.

12. **Non-blocking documentation and evidence fixes.**  
    **Plan §19, especially lines 809, 818, 822, 827–829; `STATE.md`.**

    - Finding 024 is cited as evidence, but it is not yet present in the worktree’s findings ledger. Add it as **draft**, with the actual crash, worker-pilot and timing evidence before searches rely on it.
    - “Seven one-factor variants” is inaccurate. The stronger-regularization configuration changes dropout, weight decay and batch size.
    - The worktree’s state and decisions still describe the earlier pause. Refresh them when integrating the authorized work.
    - Phrase the review statement as “provisional execution under the applicable AGENTS.md triggers.” Avoid implying that P2.5 is exempt from R2 or R3 if a later change triggers them.
    - `analysis/p25_summary.py` is still missing. Its handoff must audit the task’s completion, budget-stop status, H check and exported banks, not merely rank log records.

**Checked and found sound:** the stopping preprocessing uses only inner training rows; the final preprocessing and network are rebuilt; stateful blocks are refused; member losses and probability averaging are implemented correctly; stage seeds reset training independently of request order; the parent remained torch-free in my probes.

The old and new pure planners matched for all **13 P2.2 and 5 P2.3 saved batches**, covering **425 classical recipes**. Another **40 stored P2.4 recipe canonicalizations** matched. New source manifests necessarily change new run IDs; historical banks should retain their original identities and replay revisions.

I ran only synthetic preprocessing and mocked-protocol probes. I did not run the project-data self-checks, fit a neural model, compute a project score, or independently reproduce the OpenMP crash.