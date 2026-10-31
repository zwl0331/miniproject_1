# State: CS 155 Miniproject 1

Updated 2026-10-31, 06:05 PDT (earlier stamps today ran up to an hour ahead of the clock; the commit times are authoritative). Keep this file under 80 lines. It is a snapshot and an index, not a log: move old material to `notes/state-archive.md` instead of deleting it. The dated history up to the migration is in `doc/README.md` (status) and the phase plans.

## Brief (at most 10 lines)

- Objective: the strongest well-validated probability model for high popularity; two final Kaggle selections chosen on grouped cross-validation; a report and Colab demo that explain what was tried, why, how models were selected, and which features matter in which direction.
- Constraints: the course rules and Sam's decisions in `AGENTS.md`; folds grouped by track (Finding 002); the public leaderboard is a check only.
- Success criteria: two final selections by Monday 2026-11-02, 5:00 PM PST; report PDF and public Colab link on Gradescope by Tuesday 2026-11-03, 11:59 PM PST.
- Next milestone: paused after P2.6 (Sam, 2026-10-30). Next, on Sam's go-ahead: P2.7 (the restricted nested evaluation of the frozen menu), then phase 3 (two final selections by Monday 17:00 PST).
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
| P2.4 | 96 feature comparisons, 128 re-tuning configurations; best standalone 0.8368 (LightGBM, `date_count`, `date_te3`); best blend with H 0.8371 | provisional; audit passes; full refit exact (220 runs, 82 comparisons) | Findings 021 to 023 (draft) | 4dfd0c8 |
| P2.5 | TabM: 8 configurations, best 0.8034 (H 0.8315); best blend with H 0.8320; no allocation progress; full refit exact | provisional | Findings 024, 025 (draft) | 80a2635 |
| P2.6a | Learned-procedure engine, cache, selector, nested evaluator (with `executable_menu`); isolation checks; cost pilot within the plan's formulas | reviewed with the P2.6b freeze | Finding 026 (reviewed); `scripts/check_phase2.py` "(P2.6a)" | fd48199 |
| P2.6b | Menu of 12 frozen (`configs/phase2/menu.json`); production selection the equal average of four channels, 0.8411 (H* 0.8366, anchor 0.8315); learned procedures 0.8400 to 0.8410 | reviewed (R2 `APPROVED` at fd48199, round 2); every P2.6b run and the learned procedures replay exactly | Findings 026 to 028 (reviewed); `reviews/2026-10-31-p2.6b-menu.md` | fd48199 |
| P2.6b | Submission file `2026-10-31_p2.6b_blend_eq_all.csv` (the frozen rule's first, CV 0.8411; sha256 d2b5c461…); public AUC 0.85377, a check only | released (R1 `APPROVED` at fe70fc4; Sam uploaded it 2026-10-31) | `reviews/2026-10-31-p2.6b-submission.md`; `submissions/README.md` | af7a466 |
| P2.6b | Six more frozen-menu files prepared as public-board checks on Sam's instruction of 2026-10-31: the 0.25 Extra Trees blend (CV 0.8394), `constrained_l0.1` (0.8410), `stacker_c0.01` (0.8408), `stacker_c1` (0.8402), the standalone leader (0.8364), the family average (0.8365); the learned-procedure submission path used on real procedure banks for the first time | released (R1 `APPROVED` at d04d270, round 2; Sam uploaded all six 2026-10-31, about 05:37 PDT): public AUC 0.85398, 0.85415, 0.85246, 0.85411, 0.85068, 0.85042 in the order listed (corrected 2026-10-31 after the Codex handoff note); public minus CV +0.012 to +0.015 for every file, the CV order kept within noise; checks only | `reviews/2026-10-31-p2.6b-public-checks.md`; `submissions/README.md` | 6dbaf1f |
| Brainstorm | A separate read-only Claude Fable 5.1 session with three sub-agents (2026-10-30/31) on raising the AUC; checked here: the test file is an 80/20 row split with every train track removed (no shared uri), and on singleton validation rows every menu candidate scores 0.003 to 0.004 lower; the equal blend stays first on both scores (Spearman 0.965), though the rule's tie-break applied to singleton scores would favor the two-atom Extra Trees blend (Finding 029, a diagnostic; the rule is unchanged) | provisional (recorded; nothing adopted into the models or the frozen menu); Finding 029 draft | `doc/llm_usage/2026-10-31-auc-brainstorm/`; `analysis/brainstorm_checks_output.txt` | af7a466 |
| Workflow | agent-workflow v1, homework profile; reviewer Codex gpt-6-astra xhigh (Fable 5.1 briefly, 2026-10-30); primary agent Claude Opus 5.5 to 308b795, Claude Fable 5.1 from 2026-10-31 | reviewed | `reviews/2026-10-30-agent-workflow-migration.md`; `decisions.md` | ce04746 |

Checks: full `scripts.check_phase2` on the P2.5 and P2.6 code (527673c): 55 passed, 0 failed, 0 staged (`artifacts/phase2/check_phase2.txt`). Codex consultation rounds S to V adopted (`doc/llm_usage/2026-10-30-p2.5-consultation/`, `-p2.6-consultation/`). A rehearsal of P2.6b's evaluate stage with seed-0 stand-ins ran without error; its outputs were discarded and its scores are not used.

## Rejected approaches

- Pretrained and tabular foundation models: external data under the course rules (decision of 2026-10-29).
- Identifier and row-order features: no probe met its follow-up rule; the identifier-only model scores 0.5014 (Finding 015).
- CatBoost and EBM as leaders: below the phase 1 anchor on every repeat (Finding 019). TabM as a leader: 0.03 below the anchor (Finding 025).

## Open questions

- The course-group arrangement is not recorded (the assignment asks for groups of two to four); the exact Kaggle team-name string is to be confirmed.
- `reviews/2026-10-29-project-plan-assessment.md` comes from Sam's separate Codex chat: a request with no verdict, part of no task, kept uncommitted.
- Commits fe51507 to 21c3c4b carry a Co-Authored-By trailer contrary to `AGENTS.md` ("no AI trailers"); left as they are because P2.4's banks record 878a6f7.
- The board showed 5 entries for Sam at 03:22 PDT on 2026-10-31 when the project had logged 3 uploads (now 11 against 9); the two entries not in `submissions/README.md` were made on 2026-10-29 by 03:43 PDT (Sam to identify them; they belong in the report's record).
- The brainstorm's recommendations that need Sam's decision (`doc/llm_usage/2026-10-31-auc-brainstorm/README.md`): report the singleton-row score beside the official metric in phase 3 (a diagnostic; the frozen rule unchanged); reopen the frozen pool for extra channels (kNN anchor, CatBoost with date blocks, LightGBM without date blocks), a score-driven revision that plan section 11 forbids without Sam's decision and an R2; transductive uses of `test.csv` (boosting the 22 duplicate-pair rows, pseudo-labelling), a course-rule call.

## Running

| What | Job or process ID | Log | Started |
|---|---|---|---|
| none | | | |

## Next action

P2.7 started on Sam's go-ahead (2026-10-31, 05:41 PDT, verbatim: "Start P2.7"): the driver is being built after the Codex consultation; the cost is measured without concurrent jobs and the path chosen before any outer score. For Sam:


1. Go-ahead for P2.7: the restricted nested evaluation of `configs/phase2/menu.json` through `nested.executable_menu`, `nested.run` and `decide` (plan section 8). No driver exists yet: the engine is built and checked with stand-ins, so P2.7 starts with the Codex consultation, then a driver that saves the outer split, measures the cost without concurrent jobs and chooses the full or the reduced path before any outer score (the menu's projection, 8.48 hours from times measured under load, is not that decision; the eight-hour reservation was sized for a Sunday slot, and raising it is Sam's call). Then phase 3.
2. Sam's decisions on the brainstorm's three questions (open questions above). The primary agent's recommendation: report the singleton score as a diagnostic only, do not reopen the frozen pool, no transductive use of the test file.

Deferred: the comment on `WEIGHT_CLEANUP` in `songpop/phase2/procedures.py` (R2 F4, non-blocking) at the next source revision; structural validation of TabM receipts (Codex round T item 7); the procedure-bank validator's check of each stored combiner state's regularization value against its declared setting, with rejection tests (R1 public-checks round 1, F3, non-blocking), also at the next source revision.

After the close: Sam asks (2026-10-31) for a read of the class's public Colab notebooks, posted in public Piazza notes from Tuesday 2026-11-03, against the rules (external data, lookups by identifier, code written for this dataset, pretrained models if the staff rule them out) and against each team's numbers on the private board; factual observations for Sam, who decides what to do with them; nothing from those notebooks enters the project's models or report beyond the Challenges discussion.
