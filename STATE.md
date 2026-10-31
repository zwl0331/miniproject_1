# State: CS 155 Miniproject 1

Updated 2026-10-30, 19:52 PDT. Keep this file under 80 lines. It is a snapshot and an index, not a log: move old material to `notes/state-archive.md` instead of deleting it. The dated history up to the migration is in `doc/README.md` (status) and the phase plans.

## Brief (at most 10 lines)

- Objective: the strongest well-validated probability model for high popularity; two final Kaggle selections chosen on grouped cross-validation; a report and Colab demo that explain what was tried, why, how models were selected, and which features matter in which direction.
- Constraints: the course rules and Sam's decisions in `AGENTS.md`; folds grouped by track (Finding 002); the public leaderboard is a check only.
- Success criteria: two final selections by Monday 2026-11-02, 5:00 PM PST; report PDF and public Colab link on Gradescope by Tuesday 2026-11-03, 11:59 PM PST.
- Next milestone: the P2.4 feature stage executed and summarized (plan section 18; feature batches cut off Saturday 2026-10-31, 04:00 PDT, the re-tuning budget at 05:00 PDT).
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
| P2.2 | Calibration submission file `2026-10-30_p2.2_lightgbm_best.csv` | reviewed; no upload recorded | `submissions/README.md` | 810d1ce |
| P2.8 | Best standalone 0.8354 (tuned LightGBM, date count, `date_te30`); best blend 0.8359; identifier probes negative | reviewed | Findings 014 to 017 | 5e2d5a0 |
| P2.3 | CatBoost best 0.8306, EBM best 0.8162, both below the anchor | reviewed (approved at fingerprint 42967267…, whose tree is this commit) | Findings 018 to 020 | dca38c6 |
| P2.4 | Feature stage code, protocol and timing; nothing scored | provisional; readiness round 4 left F2 open | Finding 021 (draft), `reviews/2026-10-30-p2.4-features.md` | 74517ab |

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
| none | | | |

## Next action

Work is paused until Sam resumes it ("Don't start any new work", 2026-10-30). Then, for P2.4, per the decision of 2026-10-30:

1. Fix F2 in `scripts/run_experiments.py` (`audit_features`, about line 1425): validate each `run_failure` record's `planned_run_id` and `recipe_sha256` against the named batch's manifest, unallocated batches too; add rejection tests for an undeclared run id and a wrong recipe hash; legitimate failed-fit and resume records must still pass. Run the full `scripts.check_phase2`.
2. A source edit makes the timing, the batch record and the manifest stale: move the 7 P2.4 lines of `artifacts/phase2/experiments.jsonl` and `artifacts/phase2/batches/p24_features_1_9c42d59efbc9.json` to `artifacts/phase2/stale/p24_r4/` (as `p24_r1` to `p24_r3` were), rewrite the timing cases with `analysis/p24_make_timing.py` (it writes `configs/phase2/p2.4/timing.json` and measures nothing), measure them with `--timing configs/phase2/p2.4/timing.json --task P2.4`, then `--plan-features 1 --task P2.4` and `--audit-features P2.4 --allow-incomplete` (commands of plan section 18.10, run as `.venv/bin/python -m scripts.run_experiments`). Commit.
3. Execute with `analysis/p24_execute.py`, then the audit, `--verify --task P2.4`, `analysis/p24_summary.py` into `analysis/p24_summary_output.txt`, and Findings 022 and 023 as draft. Results stay provisional until they feed the final selection (R1).
4. Then the rest of phase 2 (`doc/plans/phase2-models-features.md`).

`tools/run_checks_filtered.py "<name substring>"` runs a subset of `scripts.check_phase2` (prints only).
