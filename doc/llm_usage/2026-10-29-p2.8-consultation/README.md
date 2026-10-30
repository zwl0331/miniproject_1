# Codex consultation for P2.8, the gap hunt

Evidence for the report's LLM Usage section (`../../../Project_1.pdf` section 6). Verbatim copies of the prompts and the final answers; nothing was edited. Sam's instructions, verbatim: "Yeah, don't assume anything. Use tests and actual data to draw conclusions." (2026-10-29), "Prioritise gap hunt" and "Yes, start P2.8" (2026-10-30).

| File | What it is |
|---|---|
| `claude_positions_before_round_M.md` | Claude's own identifier-probe design, written before the round M prompt was sent (file time 23:40 PDT, 2026-10-29) |
| `round_M_prompt.md`, `round_M_answer.md` | Round M: Codex's independent design of the identifier and order probes, during P2.2's searches |
| `claude_positions_before_round_N.md` | Claude's design of P2.8 as a whole (parts A, B, C), written before the round N prompt was sent (03:39 PDT, 2026-10-30) |
| `round_N_prompt.md`, `round_N_answer.md` | Round N: Codex's critique of that design, 16 numbered findings |

| Round | Run | Session evidence |
|---|---|---|
| M | 2026-10-29, 23:40 to 23:58 PDT, from the project root: `codex --search exec resume 01a1867a-9827-7a43-bb3a-5ffb8a6f7ffb -m gpt-6-astra -c model_reasoning_effort="xhigh" -c sandbox_mode="read-only" --skip-git-repo-check --json -o <answer> "<prompt>"`, Codex CLI 0.160.0 | The consultant thread of P0.2 to P2.2; its session log `~/.codex/sessions/2026/10/29/rollout-2026-10-29T00-05-44-01a1867a-9827-7a43-bb3a-5ffb8a6f7ffb.jsonl` records only model gpt-6-astra and effort xhigh. The event log counted 11 commands and no web search. Read-only sandbox; descriptive reads only, no model fitted |
| N | 2026-10-30, 03:40 to 03:57 PDT, the same command and thread | The same session log. The event log counted 31 commands and no web search. Read-only sandbox; no cross-validation fitted |

Roles: Claude Code (Claude Opus 5.5) is the author of P2.8. Codex advised and is not its reviewer; the formal review is a fresh Codex session. Consultation advice is input, not approval. The round M answer and Claude's positions were kept in the ignored `.review_logs/` until P2.8 opened, so they stayed outside P2.2's fingerprints.

From round M, adopted: the structural audit; the classification of candidate signals; the three identifier blocks, the identifier-only linear model L and the audio-recurrence block; the menu I1 to I8 without hyperparameter search; the follow-up thresholds and the multiplicity family; the pseudotoken controls; the implementation checks; the legality table; integration of a positive result only through a later reviewed engine extension; the null wording; a separate task with its own record. Changed: outputs under `artifacts/p2.8/` rather than inside `analysis/`, since `doc/README.md` assigns produced files to `artifacts/`; fingerprints in git mode, since P2.2 converged and was committed before P2.8 opened.

From round N, adopted (by its numbering):

1. The linear-blend threshold is a mean gain of at least 0.0005 with every repeat positive, as in round M (section 16.2); the code already applied it.
2. Part B's allocation state saved in the manifest before any P2.8 score; routes computed against it, apart from the matched difference; no feature expansion launched; P2.4 imports the records without re-judging them (section 16.3).
3. The tuned LightGBM checks B13 and B14, with the base refitted; the random forest comparator not reranked.
4. An explicit fitting contract for part A (`fit_probe` and `FittedProbe`); nothing global is sliced into an outer fit.
5. The encoding specification, with the audio deviation (S - n mu) / (n + 30) on each inner part's prior, and hand-computed checks with mixed-label groups, several uris in one group, unseen fragments and unsupported audio keys.
6. The anchor-reproduction requirements, and part A's H checked equal to part B's H.
7. One forest-prediction helper for the engine and part A, pipelines included, restored in `finally`, recorded in the fit receipt.
8. Separate checks: non-forest anchors exact, forest anchors compatible within 1e-12 with every AUC difference reported, forest self-replay exact, a pipeline fixture and a thread spy; the wording "a fixed prediction summation order".
9. The bootstrap contract: one global pairability check, the 0.00625 quantile, valid and invalid draws, "approximate multiplicity-adjusted conditional bootstrap bounds".
10. The control rule: ranking by gain over threshold, ties by id, at most two distinct pipelines (I5 and I6 share L), the 0.0005 margin, every control kept.
11. Probe banks marked not selection-eligible, I7 with `no_train_test_position_mapping`.
12. The narrower statement of what the public-score difference does and does not show.
13. The two fixed-blend diagnostics with the deterministic Extra Trees anchor.
14. A task manifest, part A's own verifier, and phase 2 runs through the canonical entry point.
15. The fit inventory, a score-free timing of the runner and the bootstrap, the order C, B, A, and a cut order.
16. The checkpoints, the task table and dependencies (P2.3's searches from P2.8's approved revision), a finding of its own for deterministic prediction, Findings 011 to 013 updated.

Changed from round N, with reasons: item 14, part B is verified by a new `--verify --task P2.8` (every logged P2.8 run) rather than a hand-kept run-id list, and its matched comparisons are computed by a read-only summary rather than logged comparison records, so rerunning the summary reproduces them; item 3, B has 14 comparisons, not 13, because Sam's instruction on the earlier project added the `music` anchor check.

The audit of the earlier audio-only project (`../../prior_project/README.md`) was Claude's own read-only work, not a consultation round; its carried-over evidence shapes section 16.3's `music` block and reserved checks.
