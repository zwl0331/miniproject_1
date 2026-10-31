# Codex consultation for P2.7, the restricted nested evaluation of the frozen menu

Evidence for the report's LLM Usage section (`../../../Project_1.pdf` section 6). Verbatim copies of the prompts and the final answers; nothing was edited. Sam, verbatim: "Start P2.7" (2026-10-31, `decisions.md`).

| File | What it is |
|---|---|
| `claude_positions_before_round_W.md` | Claude's own design of the P2.7 driver, written before the round W prompt was sent (2026-10-31, 05:44 PDT) |
| `round_W_prompt.md`, `round_W_answer.md` | Round W: Codex's independent design of the driver, without reading Claude's positions |
| `round_X_prompt.md`, `round_X_answer.md` | Round X: Codex's critique of Claude's driver at commit 948f0d4, 10 numbered findings |

| Round | Run | Session evidence |
|---|---|---|
| W | 2026-10-31, 05:45 to 05:53 PDT, from the project root: `codex --search exec -m gpt-6-astra -c model_reasoning_effort="xhigh" -c sandbox_mode="read-only" --skip-git-repo-check --json -o <answer> "<prompt>"`, Codex CLI 0.160.0 | A new consultant thread, 01a191fe-7631-7833-ab06-1cf8c66ea8b2. The event log counted 42 command-bearing events and no web search. Read-only; the answer says it fitted nothing and computed no score |
| X | 2026-10-31, 06:21 to 06:29 PDT, `codex exec resume 01a191fe-... -c model="gpt-6-astra" -c model_reasoning_effort="xhigh" -c sandbox_mode="read-only" --skip-git-repo-check --json -o <answer> "<prompt>"` | The same thread, resumed. The event log counted 26 command-bearing events and no web search. Read-only; three isolated comparator probes, nothing fitted, no score, results and caches not opened |

Roles: Claude Fable 5.1 (Claude Code) is the author of P2.7. Codex advised and is not its reviewer; the P2.7 result gets its R1 review with phase 3's selection evidence through the cross-review skill. Consultation advice is input, not approval.

From round W, adopted (by its numbering), recorded when the driver was revised:

1. The staged driver with the path decision recorded before any outer score and reloaded by every later stage; the outer assignment saved once with its hash, reloaded and checked by every stage, with both classes required in every outer fold.
2. The evaluator called directly with the saved assignment; separate cache namespaces for smoke, build and replay; the source manifest taken after all imports; the real `packages_for`; a driver-local cache subclass that publishes entries by temporary file and atomic rename, so a crash cannot leave a truncated entry.
3. A dedicated result artifact (`p27_nested_v1`: arrays in an NPZ, metadata in canonical JSON, with the menu, executable-specification, split and measure hashes, the source manifest, packages and invocation); distinct log records `nested_measure`, `nested_build`, `nested_replay`, `nested_decision`.
4. The pooled outer AUC and the group bootstrap computed inside the gated decide stage, which is the one place the driver reads outer labels, stated as such; the project's `paired_cluster_bootstrap` reused where its interface fits; intervals reported as conditional on the realized predictions.
5. The replay as a complete recomputation in a new process from an empty namespace, compared within 1e-12 on arrays and inner means and exactly on selections and tie sets, with a structural count of the fresh cross-fits expected for this menu.
6. A full-size learned-procedure smoke on a real inner population of outer fold 0 (every setting's combiner, one learned full refit), beside the shrunk-iteration smoke of the whole recursion.
7. Driver tests as a separate script: the assignment, the path gate at the boundary, the replay comparison on corrupted copies, the decide gate.
8. The reduced path exactly as frozen; an overrun recorded, never a shortcut.

Not adopted from round W, with reasons: timing by a fit-level helper instead of `atomic_cv` (the production cross-fit computes the production score that is already logged, not an outer score, and comparing it with the logged bank doubles as a reproduction check); fold-level checkpoints (the content-addressed cache already makes an interrupted build resumable); the driver-level label-flip test (the engine's own isolation checks cover the evaluator, and the driver adds no fitting code).

From round X, adopted (by its numbering), in the driver revised at the commit after the build started (the build itself ran under the driver of 948f0d4; the engine is unchanged, and the replay's comparison requires every engine source equal between build and replay while allowing the driver's own hash to differ):

1. A verification receipt (`artifacts/phase2/p27/verification.json`) that binds both result files by hash to exactly one log record each, the stable provenance (path, menu, executable, split and measurement hashes), the freshness evidence and the verifier's source manifest; the decide stage refuses to score unless the receipt passes, its hashes still match the files and its identity matches the current gate, and it records the receipt's hash in `decision.json`.
2. The measurement validated before it is recorded and again whenever it is consumed (atom sets, finite positive timings, every reproduction true, the sums, both projections, the margin, the reservation, the path, the reduced candidate), and bound to exactly one equal log record; a `preflight` stage runs that validation alone.
3. Loaded results validated structurally (fold ids in order, each selection's id, means, maximum, tie set and tie-break winner recomputed, the saved rows, the provenance); the comparison also requires fold ids, the stable provenance and the engine sources equal; the decide stage requires the replay's selections, path, decision and fold AUCs to agree with the build's.
4. Provenance extended (per-atom package maps, data hashes, the source manifest checked unchanged at the end of a stage before saving); the log records carry the driver's own hash.
5. The docstring now names the smoke stage's calls of `nested.decide` on smoke populations as the one scoring outside the decide stage; the unused helper removed.
6. The process check matches the project's scripts by name regardless of path and fails on a `ps` error.
7. The bootstrap handles an empty or single-class fold in a resample (invalid as a whole) and zero valid draws (an error); the report says the project's helper is not reused; plan section 8's two caveats and the "no longer untouched" caveat added.
8. Result files published by temporary file and atomic rename.
9. The freshness count's scope stated (a learned fixed candidate on the reduced path would need its meta stage per fold); tested for 0 to 3 learned selections and both reduced cases.
10. Checks added for the measurement record, the loaded result, the comparator counterexamples, the receipt and the freshness count.

Not adopted from round X, with reasons: an execution lock and idempotent recovery of complete-but-unlogged artifacts (nothing runs concurrently in this session, and a failed attempt is handled by hand and recorded); the driver-level label-isolation and run-wrapper spy tests (the driver passes the population, the assignment and the reduced argument straight to the evaluator, whose own isolation checks cover the recursion; left for the next revision); a protocol-constant check against the menu's `nested_protocol` (the constants are the evaluator's, reviewed in R2 with the menu).
