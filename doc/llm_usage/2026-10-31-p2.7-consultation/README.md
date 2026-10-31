# Codex consultation for P2.7, the restricted nested evaluation of the frozen menu

Evidence for the report's LLM Usage section (`../../../Project_1.pdf` section 6). Verbatim copies of the prompts and the final answers; nothing was edited. Sam, verbatim: "Start P2.7" (2026-10-31, `decisions.md`).

| File | What it is |
|---|---|
| `claude_positions_before_round_W.md` | Claude's own design of the P2.7 driver, written before the round W prompt was sent (2026-10-31, 05:44 PDT) |
| `round_W_prompt.md`, `round_W_answer.md` | Round W: Codex's independent design of the driver, without reading Claude's positions |
| `round_X_prompt.md`, `round_X_answer.md` | Round X: Codex's critique of Claude's driver draft (added when the round runs) |

| Round | Run | Session evidence |
|---|---|---|
| W | 2026-10-31, 05:45 to 05:53 PDT, from the project root: `codex --search exec -m gpt-6-astra -c model_reasoning_effort="xhigh" -c sandbox_mode="read-only" --skip-git-repo-check --json -o <answer> "<prompt>"`, Codex CLI 0.160.0 | A new consultant thread, 01a191fe-7631-7833-ab06-1cf8c66ea8b2. The event log counted 42 command-bearing events and no web search. Read-only; the answer says it fitted nothing and computed no score |

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
