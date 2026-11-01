# P2.9 GPU-node handoff

Saved 2026-10-31T18:55:14-07:00. Status: paused on Sam's instruction. This is a handoff record, not a launch budget or implementation approval.

## Owner instruction

> Okay. Pause and save your current work. I am moving to a gpu node

Stop local development. Resume only when Sam directs the next session. No GPU was rented or connected and no training was launched in this session.

## Completed work

- P2.7 engine baseline: 1ffaddd. Engine sources have not changed since that completion.
- Comprehensive P2.9 plan: `doc/plans/phase2-p29-other-methods.md`, version 1.2. Twelve families, 274 distinct initial configurations, separate feature comparisons, guaranteed core refinements, seed confirmation, expanded candidate menu and revised nested evaluation.
- Substantive plan reviewed at fe9c54aec8c8450d6976d57ea491a214f8e5efe4. Claude Fable 5.1 xhigh, independent fresh review followed by two rechecks. Final verdict APPROVED; F1 to F11 resolved. Record: `reviews/2026-10-31-p2.9-other-methods-plan.md`.
- Final status/pointers committed at 8c54be0. Two deep design consultations and their verbatim responses: `doc/llm_usage/2026-10-31-p2.9-plan/`.
- Sam chose current ensemble E as fallback: selected procedure loses at least 0.003 mean outer AUC to E and loses on all three outer folds. Historical H remains a comparator. Selection tie band remains 0.001.
- No new architecture, adapter, manifest, dependency installation, model fit or submission file has been implemented in this session.

## Resume on the node

1. Read `AGENTS.md`, `STATE.md`, this note, the reviewed P2.9 plan and its review record. Codex remains primary developer; Claude Fable 5.1 xhigh remains independent reviewer.
2. Confirm this checkout includes the final pause/handoff commit, not only the older `p2.9-reopened` worktree. That worktree was last inspected clean at 9693f11 before planning; its state has not been rechecked during this pause.
3. Inventory node OS, GPU, driver, CUDA compatibility, CPU, RAM, disk and allocated runtime. Rebuild a Python 3.12 environment from the pinned requirements and audit a separate GPU dependency route. The macOS `.venv` is not portable. Do not silently upgrade historical dependencies or treat a different device as the same recipe.
4. `.agent/review.conf` contains the laptop's absolute `REVIEW_PYTHON` path. Update it to the node interpreter before review, and make the cross-review skill and authenticated Claude CLI available. Preserve explicit `claude-fable-5-1`, `xhigh`, and read-only review mode. Never copy credentials into project files.
5. Start P2.9a after Sam resumes: source audit, precise adapter contracts, ordered manifests and correctness checks, then score-free full-size and nested-size timing pilots. Current neural implementation is CPU-based; a GPU requires implementation and verification. Historical CPU reproduction and a declared GPU reproducibility criterion are separate checks. Follow the reviewed plan before scoring.
6. Build the measured cost graph including all calibration, stopping/refit, seed, nested and replay work. Present a concrete large-job budget to Sam. Moving to a GPU node does not specify a spending limit or authorize every planned expansion.
7. Recompute schedule from current America/Los_Angeles time. Kaggle closes Monday 2026-11-02 at 17:00 PST; target files ready 15:00 PST. Report and Colab due Tuesday 2026-11-03 at 23:59 PST. The plan's intermediate targets remain provisional.

## Transfer inventory and checks

- Data CSVs, configs, prediction banks, P2.7 result/verification/decision receipts, split assignments and consultation/review records are tracked in git. At pause, `artifacts/phase2/` contains about 317.5 MiB of tracked files.
- The 700 untracked files under `artifacts/phase2/` are all in `cache/` (about 25 MiB); no new scientific result depends on transferring those caches. New empty-cache replay must remain genuinely fresh. They remain untouched locally.
- `.venv` is ignored and must be rebuilt. `data/.DS_Store` is incidental and untracked.
- `reviews/2026-10-29-project-plan-assessment.md` is Sam's pre-existing unrelated untracked note. It remains untouched and outside these commits; copy it separately if wanted.
- At pause, the read-only process check found no matching project training/review commands. All consultation and review sessions started here have completed.
- No push or remote transfer was performed. Transfer the local repository including its latest commit, or explicitly arrange a push later.

## Runtime advice, provisional

Sam asked whether a GPU would enable broader work. The response suggested one RTX 4090 and an initial budget around $50, with a short timing pilot first. These were recommendations, not an owner-approved resource budget. The rough 2 to 4 day laptop allowance was an engineering estimate, not a measured forecast; no GPU speedup or completion time was promised.

Measured context: eight historical TabM configurations took 34.8 minutes, or 4.35 minutes per complete 15-fold run on average. Applying that rate to 245 neural single-seed run-equivalents gives 17.76 hours, solely a scale comparison for initial coverage, core refinement, stopping variants and nominee confirmations/refits. It excludes feature experiments, other families, new nested evaluation, extra refinements, implementation and review. New architectures remain unmeasured. Old P2.7 build plus replay took about 3.5 hours.

Runpod's public pricing page checked 2026-10-31 listed RTX 4090 24 GB at $0.89/hour; 24 to 48 hours would be $21.36 to $42.72 compute, before storage and applicable tax. Prices and availability must be rechecked: https://www.runpod.io/pricing and https://docs.runpod.io/pods/pricing. Additional architectures or speculative objectives remain proposed extensions under the reviewed plan, not completed or automatically admitted work.
