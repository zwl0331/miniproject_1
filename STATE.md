# State: CS 155 Miniproject 1

Updated 2026-10-31, 01:25 PDT. Keep this file under 80 lines. It is a snapshot and an index, not a log: move old material to `notes/state-archive.md` instead of deleting it. The dated history up to the migration is in `doc/README.md` (status) and the phase plans.

## Brief (at most 10 lines)

- Objective: the strongest well-validated probability model for high popularity; two final Kaggle selections chosen on grouped cross-validation; a report and Colab demo that explain what was tried, why, how models were selected, and which features matter in which direction.
- Constraints: the course rules and Sam's decisions in `AGENTS.md`; folds grouped by track (Finding 002); the public leaderboard is a check only.
- Success criteria: two final selections by Monday 2026-11-02, 5:00 PM PST; report PDF and public Colab link on Gradescope by Tuesday 2026-11-03, 11:59 PM PST.
- Next milestone: P2.6b's frozen menu (`configs/phase2/menu.json`) with its R2 review; then a pause (Sam, 2026-10-30: finish P2.4, P2.5 and P2.6, then pause; P2.7 is not launched).
- Choices that could invalidate later work: the candidate freeze and the final selection rule (phase 2 plan, P2.6 and phase 3) need an R2 review before the selection is made.

## Decisions in force

Pointers into `decisions.md`: the workflow and its R1 review of every Kaggle submission file (2026-10-30); P2.4 fixes F2 and runs provisionally (2026-10-30); no pretrained models (2026-10-29); report and Colab only after development ends (2026-10-29); Sam uploads every submission (2026-10-29); a thorough search with budgets from measured runtimes (2026-10-29); test, don't assume (2026-10-29); consult Codex in depth before each task (2026-10-29); no dependence on the earlier audio-only project, whose evidence is in `doc/prior_project/` (2026-10-30); finish P2.4, then P2.5 and P2.6, then pause (2026-10-30); the reviewer is Codex gpt-6-astra at xhigh again (2026-10-31; Fable 5.1 was set on 2026-10-30).

## Results

All phase 2 scores are exploratory matched evaluations (plan section 5.7); mean fold AUC on primary groups, split seeds 155, 2026, 1029.

| ID | Result | Status | Evidence | Commit |
|---|---|---|---|---|
| P1.1 | Data facts: 3,864 rows, 35.2% High, 3,638 track groups | reviewed | Findings 001 to 006 | a369262 |
| P1.2 | Production baselines: LightGBM on F2 0.8315 | reviewed | Findings 007, 008 | 7dd8b3d |
| P1.3 | First submission, public AUC 0.83597 (pipeline check) | released | `submissions/README.md` | f6add21 |
| P2.1 | Phase 2 engine reproduces the anchor bit for bit | reviewed | Finding 009 | 719a934 |
| P2.2 | Broad searches: best single recipe 0.8339; best blend 0.8351 | reviewed | Findings 010 to 013 | 810d1ce |
| P2.2 | Calibration file `2026-10-30_p2.2_lightgbm_best.csv`: public 0.84343 (CV 0.8339; a check only) | released (Sam uploaded it, 2026-10-30) | `submissions/README.md` | f96870a |
| P2.8 | Best standalone 0.8354 (tuned LightGBM, date count, `date_te30`); best blend 0.8359; identifier probes negative | reviewed | Findings 014 to 017 | 5e2d5a0 |
| P2.3 | CatBoost best 0.8306, EBM best 0.8162, both below the anchor | reviewed | Findings 018 to 020 | dca38c6 |
| P2.4 | 96 feature comparisons, 128 re-tuning configurations; best standalone 0.8368 (LightGBM, `date_count`, `date_te3`); best blend with H 0.8371 | provisional; audit passes; full refit running | Findings 021 to 023 (draft) | 9731e7f |
| P2.5 | TabM: 8 configurations, best 0.8034 (H 0.8315); best blend with H 0.8320; no allocation progress; full refit exact | provisional | Findings 024, 025 (draft) | 80a2635 |
| P2.6a | Learned-procedure engine, cache, selector, nested evaluator; isolation checks; cost pilot within the plan's formulas | provisional | Finding 026 (draft); `scripts/check_phase2.py` "(P2.6a)" | 58f0787 |
| P2.6b | Menu of 12 frozen (`configs/phase2/menu.json`); production selection the equal average of four channels, 0.8411 (H* 0.8366, anchor 0.8315); learned procedures 0.8400 to 0.8410 | provisional; R2 review round 1 running | Findings 027, 028 (draft); `reviews/2026-10-31-p2.6b-menu.md` | 7016142 |
| Workflow | agent-workflow v1, homework profile; reviewer Codex gpt-6-astra xhigh (Fable 5.1 briefly, 2026-10-30) | reviewed | `reviews/2026-10-30-agent-workflow-migration.md`; `decisions.md` | ce04746 |

Checks: full `scripts.check_phase2` on the P2.5 and P2.6 code (527673c): 55 passed, 0 failed, 0 staged (`artifacts/phase2/check_phase2.txt`). Codex consultation rounds S to V adopted (`doc/llm_usage/2026-10-30-p2.5-consultation/`, `-p2.6-consultation/`). A rehearsal of P2.6b's evaluate stage with seed-0 stand-ins ran without error; its outputs were discarded and its scores are not used.

## Rejected approaches

- Pretrained and tabular foundation models: external data under the course rules (decision of 2026-10-29).
- Identifier and row-order features: no probe met its follow-up rule; the identifier-only model scores 0.5014 (Finding 015).
- CatBoost and EBM as leaders: below the phase 1 anchor on every repeat (Finding 019). TabM as a leader: 0.03 below the anchor (Finding 025).

## Open questions

- The course-group arrangement is not recorded (the assignment asks for groups of two to four); the exact Kaggle team-name string is to be confirmed.
- `reviews/2026-10-29-project-plan-assessment.md` comes from Sam's separate Codex chat: a request with no verdict, part of no task, kept uncommitted.
- Commits fe51507 to 21c3c4b carry a Co-Authored-By trailer contrary to `AGENTS.md` ("no AI trailers"); left as they are because P2.4's banks record 878a6f7.

## Running

| What | Job or process ID | Log | Started |
|---|---|---|---|
| P2.4 full refit (`--verify --task P2.4`, worktree `scratchpad/wt_p24v` at 9731e7f) | background shell b67m7gk93 | `scratchpad/p24_verify.log` | 2026-10-31 00:05 PDT |
| P2.6b empty-cache replay (`--verify-learned`) | background shell b1g9e48xc | `.review_logs/p26_evaluate.log` | 2026-10-31 01:17 PDT |
| R2 review of the P2.6b freeze, round 1 (Codex gpt-6-astra xhigh) | background shell bhdr9gk7d | `.review_logs/r2_p26b_round1.out` | 2026-10-31 01:19 PDT |

## Next action

1. Done: nominations (23), final-policy runs, H* (`p26_lightgbm_a8ce5635`, 0.8366), pool of 8 atoms, 4 channels (TabM not in the pool), 24 procedures; the manifest frozen at ae4dd90 before evaluation.
2. Done: blends, learned procedures, the menu (7016142), Findings 027 and 028. The replay from an empty cache is running; record its result in `artifacts/phase2/verify_learned_p26.txt`.
3. R2 review running (round 1 at 7016142). Answer every finding (FIXED, REBUTTED, DEFERRED for non-blocking), recheck in the same thread until APPROVED; quote the verdict to Sam.
4. After R2 converges, at most one submission: the candidate the frozen rule ranks first (now the equal blend `blend_eq_all`; `scripts/prepare_candidate.py` with a selection manifest), its R1 review, then the path and description to Sam.
5. P2.4's refit result into `artifacts/phase2/verify_task_p24.txt`. Then pause; P2.7 is not launched.

Scratchpad (`/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/c06b0ee0-23ee-47b0-8274-723d2960347b/scratchpad/`): worktrees `wt` (branch p25-p26-dev, merged), `wt26` (branch p26-submission, merged), `wt_p24v` (P2.4 refit); `r2_request_draft.md`; `merge_log.py`.
