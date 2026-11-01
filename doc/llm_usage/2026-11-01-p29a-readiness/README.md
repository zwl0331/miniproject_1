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
