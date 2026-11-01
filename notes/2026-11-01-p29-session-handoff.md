# P2.9 session handoff

2026-11-01, about 04:08 UTC. Sam: "Wrap up your work, I want to start a new
session." New development is paused. The earlier request to accelerate and
optimize overall public/private AUC remains the objective after resumption.
No new scientific search, submission or model selection was made this session.

## Durable checkpoints and workspaces

- Main checkout: `/exp/mu2e/app/users/wzhou2/projects/miniproject_1`, master.
  Main model sources and original `scripts/check_phase2.py` remain unchanged
  for the already running legacy suite. Commit 4def036 preserves consultation,
  draft findings and the earlier in-session narrative before handoff edits.
- Development worktree: `/tmp/miniproject_1-p29-adapters-20261101`, detached.
  Shared-repository branch `checkpoint/p29-scalar-20261101` protects commit
  `7dbfc6dae2df17b520b6de86ecf7ee3e248989e9`, so scratch loss does not lose code.
  Do not delete the worktree while its review record is being appended.
- The development chain from shared base 8dab4a6 is d7786f4, 369e762, 7dbfc6d,
  then documentation-only 1b47c17 for the completed review. Local branch
  `checkpoint/p29-review-20261101` protects that latest record and full chain.
  Integrate deliberately after the main suite ends. Main STATE/decisions and
  consultation records are newer than the development copy. Findings 034
  differ slightly between copies; development also contains Finding 035.
  Preserve both histories and reconcile documentation, rather than overwriting
  the main checkout wholesale. No source changes were cherry-picked here yet.
- Immutable correctness snapshot: `/tmp/miniproject_1-p29-scalar-gate-7dbfc6d`.
  It contains committed source/config/data/split files with read-only file modes,
  plus snapshot.json naming the commit. It has no running job.

## Completed checks and limits

| Unit | Evidence | Status |
|---|---|---|
| Eight existing raw-boundary checks | readiness/check_boundaries.txt | Passed |
| Three original legacy failures diagnosed | readiness/failure_diagnostics.txt | kNN two-ULP batch differences, historical anchor rounding, null recipe-id fixture |
| Nine affected legacy checks after corrections | readiness/check_affected.txt | Passed; preserves 1e-12 production tolerance and exact same-node neural checks |
| Legacy receipt/policy/cache controls | readiness/check_receipts_v2.txt | Eight passed in 25.617 s |
| All ten historical TabM banks under updated guards | readiness/receipt_cost_v2.json | Accepted; cold 11.45 s, later loads 0.10 to 0.32 s under concurrent load |
| Standalone raw scalar neural adapter | readiness/check_raw_scalar_v3.txt | Seven passed in 60.022 s on a 400-row/four-epoch fixture |
| Full 15-fold old/new CPU equivalence | analysis/p29_scalar_gate.py | Implemented; launch rejected, no fits started |

`readiness/` above means `doc/llm_usage/2026-11-01-p29a-readiness/`.
Some evidence is committed only on the checkpoint branch/development worktree.
Main's completed logs are also preserved. The full suite is not claimed passed.
The tiny kNN numerical kernel remains untraced; measured differences and the
known-wrong 1e-10 rejection control are retained.

d7786f4 implements the test corrections, legacy TabM stopping-receipt validator,
combiner regularization check and comment correction. 369e762 shares the frozen
legacy policy between factory and validator, adds positive max-epoch/multiple-
seed controls, and memoizes only pure stopping splits by complete immutable
labels/groups/fold-count/seed. Every receipt check still runs.

7dbfc6d adds raw_recipes.py, neural_backend.py, raw_neural.py, focused tests and
the score-free scalar gate script. The adapter fits S transforms before stopping,
then rebuilds on D for a fresh final fit. Torch stays in isolated workers.
Tests cover exact legacy scalar predictions/receipts, seed averaging, fixed-
assignment V-label isolation and S-label positive control, a deliberately leaky
D-wide encoder, query poisoning, duplicate batches, restart and split rejection.
Earlier failed scalar logs are retained. No engine/run/cache/bank integration,
full CUDA gate, expanded architecture, calibration or retrieval is implemented.
Finding 035 remains draft on the development branch. These tests do not establish
new scientific AUC improvement or production readiness.

## Consultation and review

The main directory `doc/llm_usage/2026-11-01-p29b-adapters/` contains the verbatim
Fable consultation, full JSON, original prompt and Codex integration account.
Claude Fable 5.1 xhigh session 2928bc8b-bd04-4142-b6ae-f28a911f16af completed
successfully. It is advice, separate from formal approval.

Formal R2 at d7786f4: `VERDICT: APPROVED`, no blocking finding.
Record: `reviews/2026-11-01-p29a-verification.md` in the DEVELOPMENT worktree.
Fresh Fable thread 805d0fad-9d1f-44d2-a15a-4b86862201e6. Six non-blocking findings
were answered; the kNN kernel is explicitly deferred. Round 2 rechecked material
follow-ups at 369e762 and returned `VERDICT: APPROVED`. Session 28653 completed
during wrap-up. The record and responses are saved in both checkouts.
The raw scalar unit itself has not received formal review. Non-blocking F7 asks
for the two legacy TabM engine checks and a shared-policy comment at the next
verification pass. F8 asks to clarify warm-cache timing: memoization helps only
repeat populations within one process; the first bank still constructs all 15
splits. Both author/reviewer measurements were under concurrent suite load.
These follow-ups are deferred to resumption and do not block the approval.

Review runner: `/home/wzhou2/.codex/skills/cross-review/scripts/review.sh`.
Settings explicitly select claude-fable-5-1, xhigh, read-only plan mode.
Project-context sharing with Claude was already authorized; do not ask again.
Review logs live under
`/home/wzhou2/.agent-review/miniproject_1-p29-adapters-20261101-652afada/logs/`.

## Existing work still running

1. Full original legacy suite in MAIN, session 87959, OS PID 25623, started
   03:25:37 UTC. It was still in the expensive P2.4 audit/rejection fixtures
   after 38 minutes. Log: readiness/check_phase2.txt. The expected internal
   `DIFFERS` diagnostics are not the aggregate verdict. Three initial assertion
   failures are already diagnosed and fixed only in the development worktree.
   Let this existing check complete if it survives the session switch. Inspect
   process identity and final log before restarting or integrating sources.
2. Fable recheck session 28653 completed with approval during wrap-up. No review
   remains active and no additional review or training is launched.

The suite is an existing authorized job. Its status may change after this note.
Tool sessions may not be accessible from a new session; inspect OS processes,
the log and the review record directly. Never infer success from process exit
alone. Main model/test source hashes must stay fixed while its suite is active.

## Approval explicitly pending

The attempted scalar gate launch was rejected by automatic approval review:
"This launches a long 135-fit, multi-worker CPU job consuming substantial shared
compute; the project rules require the owner's approval for large compute jobs,
and no explicit approval for this exact launch is present."

It did not execute. Sam answered "Leave approval pending" to the explicit
question. Do not retry, split or indirectly launch it to bypass this rejection.
On resumption, present the concrete run/resource estimate if seeking approval.
The prepared command, not authorized to run yet, is:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 PYTHONHASHSEED=0 /exp/mu2e/app/users/wzhou2/projects/miniproject_1/.venv/bin/python -u -m analysis.p29_scalar_gate --output /exp/mu2e/app/users/wzhou2/projects/miniproject_1/artifacts/phase2/p29/scalar_gate_7dbfc6d
```

Working directory: the immutable snapshot above. Nine child processes run over
three cases, at most three concurrently, four CPU threads each. Each case first
does two old-adapter 15-fold runs in fresh processes, compares them exactly,
then does the new-adapter 15-fold run. Total 135 outer fits, each with stopping
and refit. Arrays, input/state hashes and receipts are compared; no AUC or
scientific search record is computed. Historical Mac timings are context only,
not a measured forecast for this exact node launch. No scalar_gate result path
should be treated as an executed run without a launch/verification receipt.

## Resume sequence

1. Read AGENTS, STATE and this note. Check the legacy suite and Fable verdict.
   Preserve the latest review record in the shared repository. Finish any
   review responses according to AGENTS, without repeating completed checks.
2. Inspect the protected branch/worktree. Reconcile the main documentation and
   implementation commits after the fixed-source suite ends. Keep evidence for
   partial original-suite failures and passing corrected focused checks distinct.
3. Complete recipe_v2 dispatch before D-wide features, worker/runtime identities,
   run/bank/procedure/cache and submission integration, with legacy hashes intact.
   The consultation's concrete checklist and correction of invalid D-state
   invariance are in integration.md. No Torch in the parent.
4. Obtain approval before the pending full CPU gate. Complete full/nested CUDA
   repeatability and approved architecture/layout/calibration contracts before
   dependent scoring. Implement executable coverage manifests and score-free
   representative cost pilots. Present a measured large-search proposal.
5. Preserve E's fallback, grouped CV selection and course restrictions. The
   priority is useful overall AUC and ensemble contribution. Do not use public
   board scores to choose private-test hypotheses. Report/Colab wait until the
   end of development under Sam's existing instruction.

Environment: .venv links to `/tmp/miniproject_1-p29-venv-20261101`, Python
3.12.16 and locked packages; rebuild after node loss. CPU quota 32 core
equivalents, memory 45 GiB, A100 MIG 2g.20gb. Existing fitted packages unchanged.
Use the main absolute interpreter when running in the development worktree.
Git has no default identity: use command-local `-c user.name='Sam Zhou'
-c user.email='zwl20000331@gmail.com'`. No push and no AI trailers.
