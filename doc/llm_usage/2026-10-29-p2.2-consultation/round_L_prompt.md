Round L of the consultation: a hard critique of task P2.2's draft before its execution-readiness review. Same role as before: you advise Claude, the author; do not modify files, run reviewers or seek approvals.

Since round K: Claude adopted most of your design and built the draft in the working tree (uncommitted; `git status` shows it; ignore `reviews/2026-10-29-project-plan-assessment.md`, Sam's separate chat). Now Thursday about 21:40 PDT. No search score exists, and none may before the execution-readiness review: do not run any command that fits a configuration of the P2.2 spaces and computes or prints its score. Reading, `git diff`, the existing self-checks (`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.check_phase2`, whose small probe models are outside the P2.2 spaces), and dry runs of the planning code (`songpop.phase2.protocol.plan_wave` for wave 1, which fits nothing) are fine.

The draft:
- `songpop/phase2/models.py`: Extra Trees `max_depth`, XGBoost `reg_alpha` and `gamma`, HistGB `l2_regularization`; the kNN family (`metric`), the `knn_blocks` preprocessing with `normalization` (l2: 1/sqrt(10), alpha, beta/sqrt(3); l1: 1/10, alpha, beta/6), `validate_model_params` with domains; kNN under the OpenMP limit.
- `songpop/phase2/recipes.py`: kNN rules (F1, no blocks or drops, seeds [0], no weights, metric matching normalization); parameter domains checked in `validate_recipe`.
- `songpop/phase2/searches.py`: `search_v2` (derived rules, admissible product regions, exclusion of earlier configurations, `allow_fewer`), `search_v1` unchanged.
- `songpop/phase2/refine.py`: typed local proposals, tree variants in the first standalone slots, broad draws, active bounds.
- `songpop/phase2/protocol.py`: the protocol loader, initial recipes, the `Log` index, wave planning as a pure function of the protocol and earlier records, refinement-family choice, range extension, state resolution, `judge`.
- `songpop/phase2/allocation.py`, `batches.py`; `songpop/phase2/runs.py` (`audit_run`).
- `scripts/run_experiments.py`: `--timing` (fit and predict, no score), `--plan-wave`, `--run-wave` (with `--anchors-only` for wave 0), `--audit-task`, `--verify-task`, contribution records (H with itself as an identity).
- `configs/phase2/p2.2/protocol.json`, `configs/phase2/p2.2/timing.json`.
- `doc/plans/phase2-models-features.md` v0.4: new section 15 (the execution protocol), header.
- `doc/findings.md` (Finding 010 `reviewed`), `doc/plans/modeling_strategy.md` (status line).
- `scripts/check_phase2.py`: two new checks (search_v2, kNN). Checks of the protocol, refinement, allocation, audit and timing are being written now; say which you consider necessary.
- `doc/llm_usage/2026-10-29-p2.2-consultation/` (positions, round K).

Where the draft departs from your round K advice, deliberately:
1. The execution protocol is section 15 of the phase 2 plan, not a separate `doc/plans/p2.2-execution.md`, because `doc/project_plan.md` (Plan structure) gives each phase one plan file.
2. `knn_blocks` lives in `models.py` beside the other preprocessing constructors; no `audio_scaler` field, since only standard scaling is built.
3. LightGBM's `subsample_freq` is absent, not 0, when `subsample` is 1.
4. Log shards are not built: execution is serial (section 15.6).
5. The wave 1 manifests name the runs that supply their state (family anchor, H); the values are resolved from the log when the batch is judged, instead of a separate sealing step after wave 0.
6. The P2.2 cutoff is Friday 05:00 PDT, not 03:00.
7. H is the LightGBM anchor's P2.2 run, whose arrays must equal the P2.1 bank's.

Please critique as hard as you can, reading every changed file in full:
1. Correctness of the code: the kNN representation and its weights; parameter domains; search_v2 and its compatibility with v1 and `verify_searches`; refinement (zero moves, ordinal lists with "sqrt" and None, integer rounding, active bounds, tree variants, deduplication, determinism); range extension; wave planning (state, ceilings, counter, refinement-family choice, parents, the 5/4 then 4/5 split, sizes near a ceiling, exhausted spaces); `judge` against plan section 5.6; the run, audit and verify commands (cutoff handling, failure handling, resume after an interruption, the run-id check, the identity contribution); the timing tool.
2. Leakage and validity: anything fitted outside its training part; kNN with exact matches across primary groups; anything that lets a score influence a choice the protocol fixes.
3. Consistency between section 15, the protocol file, the code and sections 5, 6.2, 10 and 11 of the plan.
4. Verification: whether `--audit-task` really re-derives every manifest and allocation, what the full refit must cover, the P2.1 anchor at its own revision, the self-checks the review will expect.
5. Anything that would make the formal reviewer request changes, including style (short sentences, no dashes as punctuation) and unresolved paths.

Deliver a numbered list of concrete changes, each marked must-fix, should-fix or optional, with file and line or section, the problem and the exact fix. Then a short list of what you checked and found correct.
