# P2.9a resumed readiness

2026-11-01. Codex resumed after reading the project handoff. Sam asked:

> Can we start training now? What are the next few things?

Then:

> Okay, just continue your work then. Use fable for consult and review.

Codex added optional live progress to `tools/run_checks_filtered.py` without
changing the check assertions, bootstrap counts, model sources or test source
file. An empty filter runs the complete legacy suite. The log records the
original test and wrapper source hashes, process ID and each check's elapsed
time. The previous interrupted log remains under `2026-11-01-p29a-node/`.

The full rerun uses the pinned project interpreter and four-thread environment
limits. Its verdict is pending until the process exits and prints its aggregate.
No scientific search is authorized by this verification launch.

Isolated follow-up: `analysis.p29_trace_checks` records original-line tracebacks
without changing test bodies. `failure_diagnostics.txt` measures the exact
numerical differences and identifies the null-recipe fixture error. The proposed
test corrections pass (`check_fixes.txt`). On the complete validator sources,
nine affected checks pass (`check_affected.txt`), including bank validation,
procedure reconstruction, neural-worker repeatability and held-out-label
isolation. Five new receipt tests pass (`check_receipts.txt`) against historical
records and deliberately corrupted copies. No stored bank or prediction is
modified. Finding 034 is draft pending independent review.
