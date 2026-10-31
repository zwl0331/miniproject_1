# State: CS 155 Miniproject 1

Updated 2026-10-30, 23:05 PDT. Keep this file under 80 lines. It is a snapshot and an index, not a log: move old material to `notes/state-archive.md` instead of deleting it. The dated history up to the migration is in `doc/README.md` (status) and the phase plans.

## Brief (at most 10 lines)

- Objective: the strongest well-validated probability model for high popularity; two final Kaggle selections chosen on grouped cross-validation; a report and Colab demo that explain what was tried, why, how models were selected, and which features matter in which direction.
- Constraints: the course rules and Sam's decisions in `AGENTS.md`; folds grouped by track (Finding 002); the public leaderboard is a check only.
- Success criteria: two final selections by Monday 2026-11-02, 5:00 PM PST; report PDF and public Colab link on Gradescope by Tuesday 2026-11-03, 11:59 PM PST.
- Next milestone: P2.4 executed and summarized (plan section 18; feature batches cut off Saturday 2026-10-31, 04:00 PDT), then P2.5 (TabM) and P2.6 (ensembles, menu), then a pause (Sam, 2026-10-30).
- Choices that could invalidate later work: the candidate freeze and the final selection rule (phase 2 plan, P2.6 and phase 3) need an R2 review before the selection is made.

## Decisions in force

Pointers into `decisions.md`: the workflow and its R1 review of every Kaggle submission file (2026-10-30); P2.4 fixes F2 and runs provisionally (2026-10-30); no pretrained models (2026-10-29); report and Colab only after development ends (2026-10-29); Sam uploads every submission (2026-10-29); a thorough search with budgets from measured runtimes (2026-10-29); test, don't assume (2026-10-29); consult Codex in depth before each task (2026-10-29); no dependence on the earlier audio-only project, whose evidence is in `doc/prior_project/` (2026-10-30).

## Results

All phase 2 scores are exploratory matched evaluations (plan section 5.7); mean fold AUC on primary groups, split seeds 155, 2026, 1029.

| ID | Result | Status | Evidence | Commit |
|---|---|---|---|---|
| P1.1 | Data facts: 3,864 rows, 35.2% High, 3,638 track groups | reviewed | Findings 001 to 006 | a369262 |
| P1.2 | Production baselines: LightGBM on F2 0.8315 | reviewed | Findings 007, 008 | 7dd8b3d |
| P1.3 | First submission, public AUC 0.83597 (pipeline check) | released | `submissions/README.md` | f6add21 |
| P2.1 | Phase 2 engine reproduces the anchor bit for bit | reviewed | Finding 009 | 719a934 |
| P2.2 | Broad searches: best single recipe 0.8339; best blend 0.8351 | reviewed | Findings 010 to 013 | 810d1ce |
| P2.2 | Calibration submission file `2026-10-30_p2.2_lightgbm_best.csv`: public 0.84343 (CV 0.8339; a check only) | released (uploaded by Sam, 2026-10-30) | `submissions/README.md` | 810d1ce |
| P2.8 | Best standalone 0.8354 (tuned LightGBM, date count, `date_te30`); best blend 0.8359; identifier probes negative | reviewed | Findings 014 to 017 | 5e2d5a0 |
| P2.3 | CatBoost best 0.8306, EBM best 0.8162, both below the anchor | reviewed (approved at fingerprint 42967267…, whose tree is this commit) | Findings 018 to 020 | dca38c6 |
| P2.4 | F2 fixed with its rejection tests; timing and initial manifest redone; executing | provisional | Finding 021 (draft), `reviews/2026-10-30-p2.4-features.md` | 878a6f7 |
| P2.5 | TabM: 8 configurations, best 0.8034 (H 0.8315); best blend with H 0.8320; no allocation progress; audit passes | provisional; refit running | Findings 024, 025 (drafts in scratchpad); branch `p25-p26-dev` 11dd77b | 11dd77b |
| Workflow | Migration to agent-workflow v1, homework profile | reviewed (`APPROVED` in round 2, no finding open) | `reviews/2026-10-30-agent-workflow-migration.md` | 10fa77a |

Checks: the full `scripts.check_phase2` at the F2 commit 878a6f7 passed (45 passed, 0 failed, 4 staged; 35 min), 2026-10-30. On branch `p26-submission` at 527673c (P2.5 and P2.6 code): 55 passed, 0 failed, 0 staged (51 min). A rehearsal of P2.6b's evaluate stage in `scratchpad/wt26`, with seed-0 stand-ins for the final-policy runs, ran the blends, the learned procedures and the menu without error; its outputs were discarded and its scores are not used.

## Rejected approaches

- Pretrained and tabular foundation models: external data under the course rules (decision of 2026-10-29).
- Identifier and row-order features: no probe met its follow-up rule; the identifier-only model scores 0.5014 (Finding 015).
- CatBoost and EBM as leaders: below the phase 1 anchor on every repeat (Finding 019).

## Open questions

- The course-group arrangement is not recorded (the assignment asks for groups of two to four); the exact Kaggle team-name string is to be confirmed.
- `reviews/2026-10-29-project-plan-assessment.md` comes from Sam's separate Codex chat: a request with no verdict, part of no task, kept uncommitted.

## Running

| What | Job or process ID | Log | Started |
|---|---|---|---|
| P2.4 feature stage (`analysis/p24_execute.py`) | background shell b7b298jiv | `.review_logs/p24_execute.log` | 2026-10-30 20:54 PDT |
| P2.5 full refit (`--verify-task P2.5`, worktree `scratchpad/wt`) | background shell bm0ukcv3l | `scratchpad/p25_verify.log` | 2026-10-30 22:20 PDT |

## Next action

Sam, 2026-10-30: "continue the work finish P2.5 and 2.6 then pause." and "Finished 2.4 before doing".

1. P2.4: when the execution ends, `--audit-features P2.4`, `--verify --task P2.4` (a refit; may run beside P2.5 from a worktree at the P2.4 commit), `analysis/p24_summary.py` into `analysis/p24_summary_output.txt`, Findings 022 and 023 as draft. Commit.
2. P2.5 (TabM, plan section 19; branch `p25-p26-dev` in the worktree `scratchpad/wt`, head bb01429, merges cleanly): torch 2.14.1 and tabm 0.0.3 installed from the P2.1 lock (inventory checks OK); Codex rounds S and T adopted (`doc/llm_usage/2026-10-30-p2.5-consultation/`); TabM runs in a worker process (torch after LightGBM or XGBoost crashes a process); its checks pass. Master was merged into the branch (2e54b9a); the protocol is written (prior member of G: P2.8's leader, since P2.4's runs were not yet logged) and P2.5 runs in the worktree beside P2.4. Then audit, refit, summarize; Findings 024 and 025; merge the branch into master after P2.4 is committed (the log needs P2.4's lines, then the branch's).
3. P2.6a (`procedures.py`, `selection.py`, `nested.py`, plan section 20; Codex rounds U and V adopted; checks pass; cost pilot within the plan's formulas) and P2.6b's scripts (`analysis/p26_nominate.py`, `p26_pool.py`, `p26_menu.py`, `p26_execute.py`) are in the same branch. After P2.5: `p26_execute.py nominees`, commit the procedure manifest, `p26_execute.py evaluate`, the menu; R2 review of the freeze and the selection rule; at most one submission file with R1. Then pause; P2.7 is not launched.

`tools/run_checks_filtered.py "<name substring>"` runs a subset of `scripts.check_phase2` (prints only).
