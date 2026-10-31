# Review record: migration to agent-workflow v1

Opened 2026-10-30. Author: Claude Opus 5.5, Claude Code (desktop app). Reviewer: as set in .agent/review.conf.

## Round 1: review request
Unit: the migration of this project from its review-gated rules (the old `CLAUDE.md`, `doc/review_policy.md` v1.0, `tools/codex_review.sh`) to agent-workflow v1, homework profile, together with the two commits that preceded it. Trigger: R2, because these rules govern all later work in the project, including the final selections and the report.
Commit: ad81f92b2b8a81d5ec89d662534a269de750d830
Requirements:
  - Sam, verbatim: "You see the new skills that originated from my new AI workflow, right? Project setup and cross-review, skills. Can you rework the current project so that it will from now on use the new workflow?" Sam's answers to three questions, verbatim, are rows of `decisions.md` dated 2026-10-30: "Migrate now (Recommended)", "Review each one" (no Kaggle submission is pre-authorized), "Fix F2, run provisionally (Recommended)".
  - The workflow's project files: `AGENTS.md` (at most 60 lines, from the template `~/claude-skills/project-setup/templates/AGENTS.md` with the homework profile), the `CLAUDE.md` link, `STATE.md` (at most 80 lines; brief at most 10 lines), `decisions.md` (append-only, the owner's words with a source), `notes/state-archive.md`, `reviews/README.md`, `.agent/review.conf`. Project state must live only in the repository.
  - Migration may not drop or weaken a binding rule. The course rules of the old `CLAUDE.md` (`git show cab0174:CLAUDE.md`, section "Course rules") and the project conventions there (data never modified, `.venv` and `requirements.txt`, the Kaggle limit, Sam's two final selections, the findings ledger's review-first convention, the style and commit rules) must survive, in substance, in the new files.
  - The old rule that `CLAUDE.md` and `doc/review_policy.md` change only with Sam's explicit approval (old rule 9); the approval is the "Migrate now (Recommended)" row.
Constraints and decisions: `decisions.md`, all rows; the 2026-10-30 rows are the ones made today. `reviews/2026-10-29-project-plan-assessment.md` is Sam's separate Codex chat and part of no task; it is deliberately not committed.
Files and sources:
  - The two preceding commits: dca38c6cc35a3ff4c8aceba090ffe42b29aa31c9 (P2.3, built from the snapshot that the P2.3 review approved at fingerprint 42967267a99e755eb7a13c777ff80f3c6b509ad6ae84834aaea4fa52753cb7df) and 74517ab6cc084edeeb6bad68b61690df5e3bab38 (P2.4's tree at its readiness round 4 revision 3d66aed5184adfe6848c3884e8513372761ecef7c365607b97c86f5bcb245e33, provisional).
  - The migration commit ad81f92: `AGENTS.md`, `CLAUDE.md` (link), `STATE.md`, `decisions.md`, `notes/state-archive.md`, `.agent/review.conf`, `.gitignore`, `reviews/README.md`; edits to `doc/README.md`, `doc/project_plan.md`, `doc/findings.md`, `doc/plans/phase2-models-features.md` (one status line), `doc/review_policy.md` (a superseded note at the top), `submissions/README.md`; notes appended to `reviews/2026-10-30-p2.3-catboost-ebm.md` and `reviews/2026-10-30-p2.4-features.md`; `reviews/_TEMPLATE.md` and `tools/codex_review.sh` removed; added from outside the repository: `analysis/p23_execute.py`, `analysis/p24_execute.py`, `analysis/p24_summary.py`, `tools/run_checks_filtered.py`, `artifacts/phase2/stale/p24_r1` to `p24_r3`. `git diff 74517ab ad81f92 -- . ':!reviews'` shows the migration without the records.
  - The workflow itself: `~/claude-skills/project-setup/` (SKILL.md and templates) and `~/claude-skills/cross-review/` (SKILL.md, `scripts/review.sh`), read-only, outside the repository.
  - The files moved into the repository came from `~/.claude/projects/-Users-zwl0331-Desktop-Mini-project-1/handoff-2026-10-30/` (outside the repository; their sha256 below).
Checks already run:
  - P2.3: `git archive dca38c6 | tar -x -C <dir>` then `python3 <dir>/tools/fingerprint.py` printed 42967267a99e755eb7a13c777ff80f3c6b509ad6ae84834aaea4fa52753cb7df.
  - P2.4: the same with 74517ab printed 3d66aed5184adfe6848c3884e8513372761ecef7c365607b97c86f5bcb245e33.
  - Moved files, sha256 at the source and in the repository, equal: `p24_r1/moved_log_lines.jsonl` ee403313…4c10, `p24_r1/p24_features_1_3dad4f2a56bb.json` 3dad4f2a…680a, `p24_r2/moved_log_lines.jsonl` 74ae6be6…ef89, `p24_r2/p24_features_1_d13635711064.json` d1363571…49e0, `p24_r3/moved_log_lines.jsonl` b58f026a…25bd, `p24_r3/p24_features_1_0f21cb40af72.json` 0f21cb40…df50, `analysis/p23_execute.py` a7dea4fb…f00f, `analysis/p24_execute.py` 1881cd56…abaf, `analysis/p24_summary.py` 1cc8fc7b…d183, `tools/run_checks_filtered.py` 112506e3…6863.
  - `wc -l AGENTS.md STATE.md`: 60 and 59. `readlink CLAUDE.md`: `AGENTS.md`. `git diff --check`: clean.
  - The phase 2 source manifest (`songpop/phase2/banks.py`, `source_manifest`) hashes only the loaded modules and the running script, so the added files do not change P2.4's recorded source hashes. No code under `songpop/` or `scripts/` changed in ad81f92.
Questions for the reviewer:
  1. Compare the old `CLAUDE.md` and `doc/review_policy.md` at cab0174 with the new `AGENTS.md`: is any binding rule (course rules, conventions, owner authority) lost, weakened or changed in meaning beyond what Sam's decisions of 2026-10-30 call for?
  2. Does each row of `decisions.md` quote Sam's words exactly as they appear in the source it names, with the right date? Rows whose only source is a review record cannot be checked in your copy, because the records are hidden by design: the two rows of 2026-10-28, "Yeah, but I intend to execute this plan. This is good.", "Yes, continue" and "finish the current work and wrap up. Don't start any new work." List them as not checked. The last three rows quote this session's chat.
  3. Do `STATE.md`'s results, commits, statuses and next action agree with the files and the git history at this commit (findings ledger, plans, `doc/README.md`, `submissions/README.md`, `scripts/run_experiments.py` options)?
  4. Do the two preceding commits reproduce the fingerprints stated above (42967267… for dca38c6, 3d66aed5… for 74517ab), and does each hold only its task's files?
  5. Are the edited documents consistent with each other about which rules apply now, and is the "Older documents" clause of `AGENTS.md` enough to settle the remaining references to `review_policy.md` in the plans (for example `doc/plans/modeling_strategy.md` lines 161 and 220, `doc/plans/phase2-models-features.md` line 122)?
  6. Does `.agent/review.conf` match what `review.sh` reads, and does the review environment it names work?
Known limitations:
  - The plans and the modeling strategy keep their old procedural wording (about 90 references to reviews, convergence and fingerprints); `AGENTS.md` overrides them instead of their being rewritten.
  - The handoff folder outside the repository still exists (the P2.3 snapshot tar and the memory files archived from before the migration); nothing in the repository depends on it.
  - `.review_logs/` (ignored) keeps the old reviewer logs that the old records cite.

## Round 1: verdict
Reviewer: GPT-6 (Codex), local shell verification
Revision reviewed: ad81f92b2b8a81d5ec89d662534a269de750d830

What I checked and how:

- Compared the old rules at `cab0174`, the migration diff, workflow templates, and review runner. Course restrictions, owner authority, submission limits, and project conventions survive. The older-document override adequately supersedes legacy review procedures.
- Checked accessible decision quotations, dates, state entries, and referenced commits against their sources and git history.
- `wc -l AGENTS.md STATE.md`: 60 and 59. `readlink CLAUDE.md`: `AGENTS.md`. `git diff --check`: clean. HEAD and tracked project files remained unchanged.
- Recomputed both historical fingerprints using exported files and independent git-blob manifests. Both exactly match the requested hashes. All ten moved files match their handoff-source checksums.
- Exercised the runner’s configuration parser. Model, effort, sandbox, and Python settings resolve correctly; the runner explicitly passes model and effort. Python is 3.12.16, all eight base package pins match, and `pip check` passes.
- With the configured Python, `-m scripts.check_phase2`: **45 passed, 0 failed, 0 skipped, 4 staged**. `-m scripts.run_experiments --audit-features P2.4 --allow-incomplete`: exit 0.
- Independently recomputed 3,864 training rows, 1,359 High labels, and 3,638 groups. Saved predictions reproduce CatBoost AUC 0.830612659, EBM 0.816164239, and P2.8’s best LightGBM 0.835421290.

Findings:

  F1 [blocking] `STATE.md:55`. The resume instructions name `analysis/p24_make_timing.py` as the timing rerun after removing the seven stale P2.4 records. That script only writes the timing specification; it performs no timing measurements. In an isolated copy under `.review/out/`, removing those records and running the named script succeeded but left the experiment log unchanged, with zero P2.4 timing records. The documented sequence therefore omits regeneration of the required timing evidence. Add `.venv/bin/python -m scripts.run_experiments --timing configs/phase2/p2.4/timing.json --task P2.4` after generating the specification and before replanning.

Notes: Hidden review records and their verdicts were not inspected. Consequently, these decision quotations were not checked: both 2026-10-28 rows; “Yeah, but I intend to execute this plan. This is good.”; “Yes, continue”; and “finish the current work and wrap up. Don't start any new work.” The last three decisions were checked against the supplied request, not an original chat transcript. The full old-project quotation matches phase 2 plan section 16; its cited modeling-strategy source contains only an excerpt. Full historical model searches were not rerun. P2.4’s separately recorded F2 remains provisional and is outside this migration fix.

VERDICT: CHANGES REQUESTED
Reviewer run: Codex CLI 0.160.0, model gpt-6-astra, effort xhigh, sandbox workspace-write (writes confined to the review copy), --ignore-rules, thread 01a18fcb-0090-73e0-a288-4d9b9c4d3c3c, commit ad81f92b2b8a81d5ec89d662534a269de750d830, review copy checked before and after. Review environment (outside the repository, use read-only): Python /Users/zwl0331/Desktop/Mini_project_1/.venv/bin/python (Python 3.12.16); Log: /Users/zwl0331/.agent-review/Mini_project_1-303d1eaf/logs/20261030T193006-r1-codex.4nh2

## Round 2: review request
Commit: 10fa77ac4f19804d605fe08ba07b012881cd586a
Response to findings:
  F1: FIXED. `STATE.md` "Next action" step 2 now reads: rewrite the timing cases with `analysis/p24_make_timing.py` (it writes `configs/phase2/p2.4/timing.json` and measures nothing), measure them with `--timing configs/phase2/p2.4/timing.json --task P2.4`, then `--plan-features 1 --task P2.4` and `--audit-features P2.4 --allow-incomplete`, as in the command table of plan section 18.10. Checked against the log: the 7 P2.4 lines of `artifacts/phase2/experiments.jsonl` are 6 `timing` records and 1 `feature_batch` record.
  Note on the old-project quotation: the source of that `decisions.md` row now names `doc/plans/phase2-models-features.md` section 16 (in full) and `doc/plans/modeling_strategy.md` (in part).
Material edits and their dependents: the F1 edit changes the documented resume procedure in `STATE.md` only; nothing else in the repository depends on that step (no script reads `STATE.md`). The `decisions.md` edit changes a source pointer only. The `Updated` time in `STATE.md` was corrected. `git diff ad81f92 10fa77a` shows all three.

## Round 2: verdict
Reviewer: GPT-6 (Codex), local shell verification
Revision reviewed: 10fa77ac4f19804d605fe08ba07b012881cd586a

What I checked and how:

- `git diff ad81f92 10fa77a -- . ':!reviews'`: only the documented resume procedure, timestamp, and quotation source pointer changed.
- Checked the revised procedure against the generator, CLI parser, `time_cases`, and plan section 18.10.
- Ran `tools/run_checks_filtered.py 'timing: fit and predict'` with the configured Python: **1 passed, 0 failed**. It completed 10 fits, called no scorer, and produced timing records.
- Independently counted **7 P2.4 records: 6 timing records and 1 feature-batch record**. The timing records contain 90 completed fits and all match the specification’s SHA-256.
- Verified the complete old-project quotation in the newly cited phase 2 plan section 16.
- `git diff --check` passed. File lengths remain 60 and 59 lines. HEAD and tracked project files remained unchanged.

Findings:

  F1 [blocking] `STATE.md:55`. **RESOLVED.** The procedure now distinguishes specification generation from measurement and explicitly runs `--timing configs/phase2/p2.4/timing.json --task P2.4` before replanning. This addresses the missing step. No blocking findings remain.

Notes: The full regression suite and production timing were not rerun because code, configurations, and artifacts are unchanged. Hidden review records and original chat transcripts were not inspected. P2.4’s separate F2 remains provisional and is not approved by this migration review.

VERDICT: APPROVED
Reviewer run: Codex CLI 0.160.0, model gpt-6-astra, effort xhigh, sandbox workspace-write (writes confined to the review copy), --ignore-rules, thread 01a18fcb-0090-73e0-a288-4d9b9c4d3c3c, commit 10fa77ac4f19804d605fe08ba07b012881cd586a, review copy checked before and after. Review environment (outside the repository, use read-only): Python /Users/zwl0331/Desktop/Mini_project_1/.venv/bin/python (Python 3.12.16); Log: /Users/zwl0331/.agent-review/Mini_project_1-303d1eaf/logs/20261030T195232-r2-codex.CmQd
