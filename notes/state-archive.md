# State archive: CS 155 Miniproject 1

Append only. Material moved out of STATE.md, with the date it was moved.

## 2026-10-31: snapshot before Codex development handoff

The following snapshot is historical. Sam subsequently handed development to Codex and selected Claude Fable 5.1 xhigh as reviewer.

# State: CS 155 Miniproject 1

Updated 2026-10-31, 09:58 PDT (earlier stamps today ran up to an hour ahead of the clock; the commit times are authoritative). Keep this file under 80 lines. It is a snapshot and an index, not a log: move old material to `notes/state-archive.md` instead of deleting it. The dated history up to the migration is in `doc/README.md` (status) and the phase plans.

## Brief (at most 10 lines)

- Objective: the strongest well-validated probability model for high popularity; two final Kaggle selections chosen on grouped cross-validation; a report and Colab demo that explain what was tried, why, how models were selected, and which features matter in which direction.
- Constraints: the course rules and Sam's decisions in `AGENTS.md`; folds grouped by track (Finding 002); the public leaderboard is a check only.
- Success criteria: two final selections by Monday 2026-11-02, 5:00 PM PST; report PDF and public Colab link on Gradescope by Tuesday 2026-11-03, 11:59 PM PST.
- Next milestone: P2.7 complete (2026-10-31; Finding 030 draft). Next, on Sam's go-ahead: phase 3 (two final selections by Monday 17:00 PST) on the current menu, or on a revised menu from the P2.9 session with its own nested evaluation.
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
| P2.7 | Restricted nested evaluation of the frozen menu (plan section 8), full path: the selected procedure with fold averaging scores outer AUC 0.8418, 0.8377, 0.8399 (mean 0.8398, pooled 0.8391) against the anchor's 0.8334, 0.8251, 0.8334 (mean 0.8306); paired gain +0.0092, positive on every fold, group bootstrap 95% +0.0057 to +0.0127 (the production CV gain is +0.0096); the full refit +0.0010 over fold averaging, not on every fold; the inner rule chose `blend_eq_all` twice and the 0.25 Extra Trees blend once; decision: the selected procedure with fold averaging, no fallback; build 1.74 h and empty-cache replay 1.75 h equal within 1e-12 with byte-identical arrays, 294 fresh cross-fits each; C_all 394.3 s, C_L 364.3 s, projection 4.97 h against 8 | provisional (protocol, menu and rule reviewed in R2 at fd48199; the result is reviewed with phase 3's selection); Finding 030 draft | `artifacts/phase2/p27/` (`measure.json`, `result_build.*`, `result_replay.*`, `verification.json`, `replay.txt`, `decision.json`); `doc/llm_usage/2026-10-31-p2.7-consultation/` | (see git log) |
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
| none (P2.7 finished 2026-10-31, 09:53 PDT) | | | |

## Next action

Development reopened (Sam, 2026-10-31, `decisions.md`), explored in a separate session as P2.9. This session has finished P2.7 (build, empty-cache replay, receipt, decision and Finding 030, all on 2026-10-31; Codex round X's critique adopted in the driver at 157a014) and starts nothing new. The P2.9 session works in the worktree `../Mini_project_1_p29` (branch `p2.9-reopened` at 9693f11, `.venv` linked). P2.7 no longer needs this checkout's engine sources fixed; phase 3's deployment of the frozen menu still relies on the current sources reproducing its banks (verified at d04d270 and in P2.7's measurement), so engine changes from P2.9 land here only with a fresh reproduction check. Candidates P2.9 finds are validated on the 15 production folds, and a revised menu gets its own nested evaluation. Starting points for it: the under-tested items in `notes/2026-10-31-claude-response-to-codex-handoff.md` and the Codex handoff beside it (a hierarchical prior for the exact-date encoding, the duration-rounding flag, an RBF SVM family, a neural channel with numerical embeddings and the date blocks); the brainstorm's negative results in `doc/llm_usage/2026-10-31-auc-brainstorm/`. For Sam:

1. P2.7 is complete (Finding 030, draft): in outer folds that never saw the scored rows, the frozen rule's procedure gains +0.0092 AUC over the anchor against +0.0096 on the production folds, so the production selection `blend_eq_all` is not an artifact of selecting on those folds; the deployment rule keeps fold averaging, which is the already-uploaded file `2026-10-31_p2.6b_blend_eq_all.csv`; no refit file is needed. Next, on Sam's go-ahead: phase 3 on the current menu (the first final selection by the frozen rule; the second is Sam's choice, with Findings 028 to 030 as evidence), or on a revised menu from the P2.9 session with its own nested evaluation.
2. Sam's decisions on the brainstorm's three questions (open questions above). The primary agent's recommendation: report the singleton score as a diagnostic only, do not reopen the frozen pool, no transductive use of the test file.

Deferred: the comment on `WEIGHT_CLEANUP` in `songpop/phase2/procedures.py` (R2 F4, non-blocking) at the next source revision; structural validation of TabM receipts (Codex round T item 7); the procedure-bank validator's check of each stored combiner state's regularization value against its declared setting, with rejection tests (R1 public-checks round 1, F3, non-blocking), also at the next source revision.

After the close: Sam asks (2026-10-31) for a read of the class's public Colab notebooks, posted in public Piazza notes from Tuesday 2026-11-03, against the rules (external data, lookups by identifier, code written for this dataset, pretrained models if the staff rule them out) and against each team's numbers on the private board; factual observations for Sam, who decides what to do with them; nothing from those notebooks enters the project's models or report beyond the Challenges discussion.

## 2026-10-31: state before comprehensive P2.9 planning

# State: CS 155 Miniproject 1

Updated 2026-10-31, Codex development handoff. Keep this file under 80 lines. It is a snapshot and an index, not a log: move old material to `notes/state-archive.md` instead of deleting it. Earlier clock corrections and the pre-handoff snapshot are archived there.

## Brief (at most 10 lines)

- Objective: the strongest well-validated probability model for high popularity; two final Kaggle selections chosen on grouped cross-validation; a report and Colab demo that explain what was tried, why, how models were selected, and which features matter in which direction.
- Constraints: the course rules and Sam's decisions in `AGENTS.md`; folds grouped by track (Finding 002); the public leaderboard is a check only.
- Success criteria: two final selections by Monday 2026-11-02, 5:00 PM PST; report PDF and public Colab link on Gradescope by Tuesday 2026-11-03, 11:59 PM PST.
- Next milestone: P2.9 development under Codex, on Sam's handoff of 2026-10-31. P2.7 is complete (Finding 030 draft). Improve overall validation performance before phase 3; report and Colab remain deferred until development ends.
- Choices that could invalidate later work: the candidate freeze and the final selection rule (phase 2 plan, P2.6 and phase 3) need an R2 review before the selection is made.

## Decisions in force

Pointers into `decisions.md`: Codex takes over development and Claude Fable 5.1 xhigh becomes reviewer (latest decision, 2026-10-31); development reopened as P2.9 (2026-10-31), superseding the earlier pause and candidate freeze for new exploration; every submission file gets R1 review (2026-10-30); no pretrained models, report and Colab only after development ends, Sam uploads every submission, search budgets from measured runtimes, test rather than assume, and consultation before building each task (2026-10-29); no dependence on the earlier audio-only project (2026-10-30).

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
| P2.7 | Restricted nested evaluation of the frozen menu (plan section 8), full path: the selected procedure with fold averaging scores outer AUC 0.8418, 0.8377, 0.8399 (mean 0.8398, pooled 0.8391) against the anchor's 0.8334, 0.8251, 0.8334 (mean 0.8306); paired gain +0.0092, positive on every fold, group bootstrap 95% +0.0057 to +0.0127 (the production CV gain is +0.0096); the full refit +0.0010 over fold averaging, not on every fold; the inner rule chose `blend_eq_all` twice and the 0.25 Extra Trees blend once; decision: the selected procedure with fold averaging, no fallback; build 1.74 h and empty-cache replay 1.75 h equal within 1e-12 with byte-identical arrays, 294 fresh cross-fits each; C_all 394.3 s, C_L 364.3 s, projection 4.97 h against 8 | provisional (protocol, menu and rule reviewed in R2 at fd48199; the result is reviewed with phase 3's selection); Finding 030 draft | `artifacts/phase2/p27/` (`measure.json`, `result_build.*`, `result_replay.*`, `verification.json`, `replay.txt`, `decision.json`); `doc/llm_usage/2026-10-31-p2.7-consultation/` | (see git log) |
| Brainstorm | A separate read-only Claude Fable 5.1 session with three sub-agents (2026-10-30/31) on raising the AUC; checked here: the test file is an 80/20 row split with every train track removed (no shared uri), and on singleton validation rows every menu candidate scores 0.003 to 0.004 lower; the equal blend stays first on both scores (Spearman 0.965), though the rule's tie-break applied to singleton scores would favor the two-atom Extra Trees blend (Finding 029, a diagnostic; the rule is unchanged) | provisional (recorded; nothing adopted into the models or the frozen menu); Finding 029 draft | `doc/llm_usage/2026-10-31-auc-brainstorm/`; `analysis/brainstorm_checks_output.txt` | af7a466 |
| Workflow | agent-workflow v1, homework profile; primary developer Codex after the 2026-10-31 handoff; reviewer Claude Fable 5.1 xhigh, with model and effort passed explicitly; earlier roles archived | migration reviewed; handoff owner-directed | `reviews/2026-10-30-agent-workflow-migration.md`; `decisions.md`; `.agent/review.conf` | ce04746 (migration) |

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
- The brainstorm's remaining questions (`doc/llm_usage/2026-10-31-auc-brainstorm/README.md`): report the singleton-row score beside the official metric in phase 3 (diagnostic only); transductive uses of `test.csv` remain a course-rule question. Reopening development is already authorized by Sam (2026-10-31); the older recommendation against reopening is superseded.

## Running

| What | Job or process ID | Log | Started |
|---|---|---|---|
| none (P2.7 finished 2026-10-31, 09:53 PDT) | | | |

## Next action

Codex owns development and this state file from Sam's handoff of 2026-10-31. Claude Fable 5.1 xhigh is the independent reviewer through `.agent/review.conf`; the runner supplies `--model claude-fable-5-1 --effort xhigh` and uses read-only plan mode. The prior instruction that the finishing P2.7 session start nothing new does not pause this new development session.

1. Continue P2.9 by turning the existing handoffs into a concrete provisional experiment plan: `notes/2026-10-31-codex-neural-models-handoff.md`, `notes/2026-10-31-codex-auc-plateau-handoff.md`, `notes/2026-10-31-claude-response-to-codex-handoff.md`, and the negative results in `doc/llm_usage/2026-10-31-auc-brainstorm/`. Preserve consultation before building and obtain R2 for consequential method choices. Measure representative runtimes before proposing large jobs. Judge new candidates on the 15 production folds and their contribution to the current ensemble; a revised menu needs its own nested evaluation.
2. Checkout baseline: this checkout is at P2.7 completion commit 1ffaddd before the handoff. The separate `../Mini_project_1_p29` worktree is clean at 9693f11 on `p2.9-reopened`, as inspected during the handoff; it contains no subsequent tracked development. Integrate work deliberately from this completed baseline. Engine changes must reproduce the existing banks before the current menu is deployed again.
3. P2.7's saved result keeps fold averaging with no fallback (Finding 030, draft). Phase 3 and the two final selections still require their R1 review and Sam's decision. The singleton diagnostic and transductive proposals remain open as listed above. No fitting job or new review was launched for this administrative handoff.

Deferred: the comment on `WEIGHT_CLEANUP` in `songpop/phase2/procedures.py` (R2 F4, non-blocking) at the next source revision; structural validation of TabM receipts (Codex round T item 7); the procedure-bank validator's check of each stored combiner state's regularization value against its declared setting, with rejection tests (R1 public-checks round 1, F3, non-blocking), also at the next source revision.

After the close: Sam asks (2026-10-31) for a read of the class's public Colab notebooks, posted in public Piazza notes from Tuesday 2026-11-03, against the rules (external data, lookups by identifier, code written for this dataset, pretrained models if the staff rule them out) and against each team's numbers on the private board; factual observations for Sam, who decides what to do with them; nothing from those notebooks enters the project's models or report beyond the Challenges discussion.

## 2026-10-31 P2.9 planning checkpoint

The planning milestone and pending R2 entries are complete. Two deep Fable consultation rounds preceded an independent review at ad1ca96 and rechecks at abb6863 and fe9c54a. Round 3 approved with no findings; F1 to F11 are resolved. The reviewed plan is doc/plans/phase2-p29-other-methods.md and the complete record is reviews/2026-10-31-p2.9-other-methods-plan.md. No fitting job, package installation or submission preparation occurred. Next is P2.9a, followed by a measured large-job proposal.

## 2026-10-31 pause for GPU migration

The prior next action was P2.9a after completion of the reviewed plan at fe9c54a, with final status recorded at 8c54be0. Sam now pauses local work and moves to a GPU node. The handoff is notes/2026-10-31-p29-gpu-handoff.md. No fitting or GPU job has launched. Runtime/rental advice was provisional and did not establish a launch budget.

## 2026-11-01 GPU-node resume: previous snapshot

# State: CS 155 Miniproject 1

Updated 2026-11-01. PAUSED on Sam's instruction for migration to a GPU node. Keep this file under 80 lines. It is a snapshot and an index, not a log: move old material to `notes/state-archive.md` instead of deleting it. Earlier clock corrections and the pre-handoff snapshot are archived there.

## Brief (at most 10 lines)

- Objective: the strongest well-validated probability model for high popularity; two final Kaggle selections chosen on grouped cross-validation; a report and Colab demo that explain what was tried, why, how models were selected, and which features matter in which direction.
- Constraints: the course rules and Sam's decisions in `AGENTS.md`; folds grouped by track (Finding 002); the public leaderboard is a check only.
- Success criteria: two final selections by Monday 2026-11-02, 5:00 PM PST; report PDF and public Colab link on Gradescope by Tuesday 2026-11-03, 11:59 PM PST.
- Next milestone after Sam resumes: GPU environment audit and P2.9a under the reviewed plan, then score-free pilots and a measured launch proposal. Handoff: `notes/2026-10-31-p29-gpu-handoff.md`. Planning is complete; no training launched. P2.7 is complete (Finding 030 draft); report and Colab wait until development ends.
- Choices that could invalidate later work: the candidate freeze and the final selection rule (phase 2 plan, P2.6 and phase 3) need an R2 review before the selection is made.

## Decisions in force

Pointers into `decisions.md`: pause and save for GPU-node migration (latest, 2026-10-31); Codex takes over development and Claude Fable 5.1 xhigh becomes reviewer (2026-10-31); development reopened as P2.9 (2026-10-31), superseding the earlier pause and candidate freeze for new exploration; every submission file gets R1 review (2026-10-30); no pretrained models, report and Colab only after development ends, Sam uploads every submission, search budgets from measured runtimes, test rather than assume, and consultation before building each task (2026-10-29); no dependence on the earlier audio-only project (2026-10-30).

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
| P2.7 | Restricted nested evaluation of the frozen menu (plan section 8), full path: the selected procedure with fold averaging scores outer AUC 0.8418, 0.8377, 0.8399 (mean 0.8398, pooled 0.8391) against the anchor's 0.8334, 0.8251, 0.8334 (mean 0.8306); paired gain +0.0092, positive on every fold, group bootstrap 95% +0.0057 to +0.0127 (the production CV gain is +0.0096); the full refit +0.0010 over fold averaging, not on every fold; the inner rule chose `blend_eq_all` twice and the 0.25 Extra Trees blend once; decision: the selected procedure with fold averaging, no fallback; build 1.74 h and empty-cache replay 1.75 h equal within 1e-12 with byte-identical arrays, 294 fresh cross-fits each; C_all 394.3 s, C_L 364.3 s, projection 4.97 h against 8 | provisional (protocol, menu and rule reviewed in R2 at fd48199; the result is reviewed with phase 3's selection); Finding 030 draft | `artifacts/phase2/p27/` (`measure.json`, `result_build.*`, `result_replay.*`, `verification.json`, `replay.txt`, `decision.json`); `doc/llm_usage/2026-10-31-p2.7-consultation/` | (see git log) |
| Brainstorm | A separate read-only Claude Fable 5.1 session with three sub-agents (2026-10-30/31) on raising the AUC; checked here: the test file is an 80/20 row split with every train track removed (no shared uri), and on singleton validation rows every menu candidate scores 0.003 to 0.004 lower; the equal blend stays first on both scores (Spearman 0.965), though the rule's tie-break applied to singleton scores would favor the two-atom Extra Trees blend (Finding 029, a diagnostic; the rule is unchanged) | provisional (recorded; nothing adopted into the models or the frozen menu); Finding 029 draft | `doc/llm_usage/2026-10-31-auc-brainstorm/`; `analysis/brainstorm_checks_output.txt` | af7a466 |
| P2.9 plan | 12 families, 274 distinct initial configurations, guaranteed core refinement, feature comparisons, seed checks and revised evaluation; E fallback as Sam chose | reviewed (R2 APPROVED, round 3; all 11 findings resolved) | `doc/plans/phase2-p29-other-methods.md`; `reviews/2026-10-31-p2.9-other-methods-plan.md` | fe9c54a |
| Workflow | agent-workflow v1, homework profile; primary developer Codex after the 2026-10-31 handoff; reviewer Claude Fable 5.1 xhigh, with model and effort passed explicitly; earlier roles archived | migration reviewed; handoff owner-directed | `reviews/2026-10-30-agent-workflow-migration.md`; `decisions.md`; `.agent/review.conf` | ce04746 (migration) |

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
- The brainstorm's remaining questions (`doc/llm_usage/2026-10-31-auc-brainstorm/README.md`): report the singleton-row score beside the official metric in phase 3 (diagnostic only); transductive uses of `test.csv` remain a course-rule question. Reopening development is already authorized by Sam (2026-10-31); the older recommendation against reopening is superseded.

## Running

| What | Job or process ID | Log | Started |
|---|---|---|---|
| none; paused for GPU migration; training/review process check found no matches | | `notes/2026-10-31-p29-gpu-handoff.md` | 2026-10-31 |

## Next action

Node setup, 2026-11-01: installed the four shared skills from `../agent-skills` into `/home/wzhou2/.codex/skills` at Sam's request. Links and all 45 file hashes verified; built-in skill files unchanged. Evidence: `doc/llm_usage/2026-11-01-skill-installation/`. No model development resumed.

Codex owns development and this state file from Sam's handoff of 2026-10-31. Claude Fable 5.1 xhigh is the independent reviewer through `.agent/review.conf`; the runner supplies `--model claude-fable-5-1 --effort xhigh` and uses read-only plan mode. Sam explicitly paused this session on 2026-10-31 for GPU migration. Remain paused until Sam resumes; first read `notes/2026-10-31-p29-gpu-handoff.md`.

1. Planning is complete: read the reviewed `doc/plans/phase2-p29-other-methods.md` and `reviews/2026-10-31-p2.9-other-methods-plan.md` (R2 APPROVED at fe9c54a, round 3). Two deep Fable consultations, Codex design/critique and integration are in `doc/llm_usage/2026-10-31-p2.9-plan/`. After Sam resumes, proceed with P2.9a: source audit, implementation contracts, ordered manifests and score-free pilots, then Sam's measured large-job proposal. Coverage targets are not launch authorization. E is the revised fallback; the 0.003/all-three-fold trigger and 0.001 tie band are unchanged.
2. Engine baseline: engine sources remain unchanged since P2.7 completion commit 1ffaddd; subsequent commits record the handoff and P2.9 plan. The separate `../Mini_project_1_p29` worktree is clean at 9693f11 on `p2.9-reopened`, as inspected during the handoff; it contains no subsequent tracked development. Integrate work deliberately from this completed baseline. Engine changes must reproduce the existing banks before the current menu is deployed again.
3. P2.7's saved result keeps fold averaging with no fallback (Finding 030, draft). Phase 3 and the two final selections still require their R1 review and Sam's decision. The singleton diagnostic and transductive proposals remain open as listed above. No fitting job has launched in this planning turn; consultation is distinct from formal approval.

Deferred: the comment on `WEIGHT_CLEANUP` in `songpop/phase2/procedures.py` (R2 F4, non-blocking) at the next source revision; structural validation of TabM receipts (Codex round T item 7); the procedure-bank validator's check of each stored combiner state's regularization value against its declared setting, with rejection tests (R1 public-checks round 1, F3, non-blocking), also at the next source revision.

After the close: Sam asks (2026-10-31) for a read of the class's public Colab notebooks, posted in public Piazza notes from Tuesday 2026-11-03, against the rules (external data, lookups by identifier, code written for this dataset, pretrained models if the staff rule them out) and against each team's numbers on the private board; factual observations for Sam, who decides what to do with them; nothing from those notebooks enters the project's models or report beyond the Challenges discussion.


## 2026-11-01 node checkpoint before session wrap

# State: CS 155 Miniproject 1

Updated 2026-11-01. P2.9 resumed on Sam's instruction, on the GPU node. Keep this file under 80 lines. It is a snapshot and an index, not a log: move old material to `notes/state-archive.md` instead of deleting it. Earlier clock corrections and the pre-handoff snapshot are archived there.

## Brief (at most 10 lines)

- Objective: the strongest well-validated probability model for high popularity; two final Kaggle selections chosen on grouped cross-validation; a report and Colab demo that explain what was tried, why, how models were selected, and which features matter in which direction.
- Constraints: the course rules and Sam's decisions in `AGENTS.md`; folds grouped by track (Finding 002); the public leaderboard is a check only.
- Success criteria: two final selections by Monday 2026-11-02, 5:00 PM PST; report PDF and public Colab link on Gradescope by Tuesday 2026-11-03, 11:59 PM PST.
- Next milestone: P2.9a source audit, implementation contracts and score-free pilots, then a measured launch proposal. Node: A100 MIG 2g.20gb, 32-core CPU quota, 45 GiB RAM. Python 3.12.16 and 85 hash-locked packages installed; .venv links to node-local /tmp after a shared-filesystem quota failure. P2.7 is complete (Finding 030 draft); report and Colab wait until development ends.
- Choices that could invalidate later work: the candidate freeze and the final selection rule (phase 2 plan, P2.6 and phase 3) need an R2 review before the selection is made.

## Decisions in force

Pointers into `decisions.md`: resume on the GPU node, Fable 5.1 xhigh consultant and independent reviewer (2026-11-01), superseding the migration pause; Codex owns development (2026-10-31); P2.9 reopening supersedes the earlier candidate freeze; every submission gets R1 (2026-10-30); no pretrained models, report and Colab after development, Sam uploads, measured budgets and consultation before building (2026-10-29); no dependence on the earlier audio-only project (2026-10-30).

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
| P2.7 | Restricted nested evaluation of the frozen menu (plan section 8), full path: the selected procedure with fold averaging scores outer AUC 0.8418, 0.8377, 0.8399 (mean 0.8398, pooled 0.8391) against the anchor's 0.8334, 0.8251, 0.8334 (mean 0.8306); paired gain +0.0092, positive on every fold, group bootstrap 95% +0.0057 to +0.0127 (the production CV gain is +0.0096); the full refit +0.0010 over fold averaging, not on every fold; the inner rule chose `blend_eq_all` twice and the 0.25 Extra Trees blend once; decision: the selected procedure with fold averaging, no fallback; build 1.74 h and empty-cache replay 1.75 h equal within 1e-12 with byte-identical arrays, 294 fresh cross-fits each; C_all 394.3 s, C_L 364.3 s, projection 4.97 h against 8 | provisional (protocol, menu and rule reviewed in R2 at fd48199; the result is reviewed with phase 3's selection); Finding 030 draft | `artifacts/phase2/p27/` (`measure.json`, `result_build.*`, `result_replay.*`, `verification.json`, `replay.txt`, `decision.json`); `doc/llm_usage/2026-10-31-p2.7-consultation/` | (see git log) |
| Brainstorm | A separate read-only Claude Fable 5.1 session with three sub-agents (2026-10-30/31) on raising the AUC; checked here: the test file is an 80/20 row split with every train track removed (no shared uri), and on singleton validation rows every menu candidate scores 0.003 to 0.004 lower; the equal blend stays first on both scores (Spearman 0.965), though the rule's tie-break applied to singleton scores would favor the two-atom Extra Trees blend (Finding 029, a diagnostic; the rule is unchanged) | provisional (recorded; nothing adopted into the models or the frozen menu); Finding 029 draft | `doc/llm_usage/2026-10-31-auc-brainstorm/`; `analysis/brainstorm_checks_output.txt` | af7a466 |
| P2.9 plan | 12 families, 274 distinct initial configurations, guaranteed core refinement, feature comparisons, seed checks and revised evaluation; E fallback as Sam chose | reviewed (v1.2 at fe9c54a; v1.3 portability amendment R2 APPROVED at c6fa913, round 2) | `doc/plans/phase2-p29-other-methods.md`; `reviews/2026-11-01-p29a-node-portability.md` | c6fa913 |
| P2.9a node | Environment installed; 19 foundation checks pass; six synthetic GPU probes repeat exactly; H/E atoms repeat byte-identically on-node, but several differ from Mac; node E diagnostic 0.840896 vs historical 0.841062 | reviewed, Finding 031 and v1.3 amendment; R2 APPROVED at c6fa913 | `doc/plans/phase2-p29-node-contracts.md`; `doc/llm_usage/2026-11-01-p29a-node/`; `artifacts/phase2/p29/` | 74fc0b3 |
| P2.9a boundary | Raw-feature wrapper and strict group partitions; eight focused tests pass, including a deliberately leaky encoder control and legacy feature equivalence; not connected to the engine | provisional, Finding 032; outside migration R2 snapshot | `songpop/phase2/raw_features.py`; `raw_splits.py`; `scripts/check_p29_boundaries.py` | see git log |
| P2.9a device pilot | Fixed TabM B, 3,091/2,061 fitting rows, two fresh CPU/CUDA repeats each: exact within-device states/predictions/receipts; first-fold CPU equals old adapter; GPU process times about 2.24/2.37 times shorter | provisional, Finding 033; one recipe, no outer AUC or search budget | `analysis/p29_tabm_device_pilot.py`; `doc/llm_usage/2026-11-01-p29a-node/tabm_device_pilot.json` | see git log |
| Workflow | agent-workflow v1, homework profile; primary developer Codex after the 2026-10-31 handoff; reviewer Claude Fable 5.1 xhigh, with model and effort passed explicitly; earlier roles archived | migration reviewed; handoff owner-directed | `reviews/2026-10-30-agent-workflow-migration.md`; `decisions.md`; `.agent/review.conf` | ce04746 (migration) |

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
- The brainstorm's remaining questions (`doc/llm_usage/2026-10-31-auc-brainstorm/README.md`): report the singleton-row score beside the official metric in phase 3 (diagnostic only); transductive uses of `test.csv` remain a course-rule question. Reopening development is already authorized by Sam (2026-10-31); the older recommendation against reopening is superseded.

## Running

| What | Job or process ID | Log | Started |
|---|---|---|---|
| Phase-two checks; migration R2 approved. Consultations, tree replays and CPU/CUDA prototype pilots complete; no search fits | checks 45241 (PID 12288) | `doc/llm_usage/2026-11-01-p29a-node/`; `reviews/2026-11-01-p29a-node-portability.md` | 2026-11-01 |

## Next action

Node setup, 2026-11-01: installed the four shared skills from `../agent-skills` into `/home/wzhou2/.codex/skills` at Sam's request. Links and all 45 file hashes verified; built-in skill files unchanged. Evidence: `doc/llm_usage/2026-11-01-skill-installation/`. This preceded the development resume below.

Sam resumed development on 2026-11-01. Codex owns development and STATE. Claude Fable 5.1 xhigh consults and reviews in separate sessions, with explicit model/effort and read-only plan mode. The old handoff is `notes/2026-10-31-p29-gpu-handoff.md`. Node environment installed and REVIEW_PYTHON corrected; .venv points to `/tmp/miniproject_1-p29-venv-20261101`, so rebuild after node loss. Historical locks remain unchanged.

1. Finish the running legacy phase-two suite. Migration v1.3, node contracts and Finding 031 are R2 APPROVED at c6fa913 (`reviews/2026-11-01-p29a-node-portability.md`). Continue P2.9a/b recipe_v2/raw adapters, complete equivalence/isolation gates, executable manifests and representative cost pilots, followed by Sam's measured large-job proposal. Provisional starting code: boundary f420853; one-recipe device prototype aab2370. Neither substitutes for the full gates. Node consultation/evidence: `doc/llm_usage/2026-11-01-p29a-node/`. Coverage targets are not launch authorization. E's procedure, fallback trigger and tie band are unchanged.
2. Engine baseline: engine sources remain unchanged since P2.7 completion commit 1ffaddd; subsequent commits record the handoff and P2.9 plan. The separate `../Mini_project_1_p29` worktree was clean at 9693f11 on `p2.9-reopened` during the handoff. Integrate work deliberately from this completed baseline. Same-node adapter equivalence stays strict; cross-platform portability differs and awaits the explicit R2 amendment above.
3. P2.7's saved result keeps fold averaging with no fallback (Finding 030, draft). Phase 3 and the two final selections still require their R1 review and Sam's decision. The singleton diagnostic and transductive proposals remain open as listed above. Migration fits are diagnostics; no new scientific search has launched. Consultation is distinct from formal approval.

Deferred: the comment on `WEIGHT_CLEANUP` in `songpop/phase2/procedures.py` (R2 F4, non-blocking) at the next source revision; structural validation of TabM receipts (Codex round T item 7); the procedure-bank validator's check of each stored combiner state's regularization value against its declared setting, with rejection tests (R1 public-checks round 1, F3, non-blocking), also at the next source revision.

After the close: Sam asks (2026-10-31) for a read of the class's public Colab notebooks, posted in public Piazza notes from Tuesday 2026-11-03, against the rules (external data, lookups by identifier, code written for this dataset, pretrained models if the staff rule them out) and against each team's numbers on the private board; factual observations for Sam, who decides what to do with them; nothing from those notebooks enters the project's models or report beyond the Challenges discussion.

## 2026-11-01: resume after node-session handoff

Previous STATE at 94917b4 said work was paused on Sam's wrap request. Its running
table recorded no remaining jobs and an interrupted legacy suite after about
40 minutes, exit 130, with no complete node verdict. The retained output is
`doc/llm_usage/2026-11-01-p29a-node/check_phase2.txt`. Sam resumed development in
the new session, requested Fable consultation/review, and the full suite restarted
with progress output. The stale statement that portability still awaited R2 was
corrected to the existing approval at c6fa913; no method criterion changed.


## 2026-11-01 pre-wrap checkpoint, preserved from 4def036

# State: CS 155 Miniproject 1

Updated 2026-11-01. Resumed readiness and implementation on Sam's instruction. No broad-search launch authorized. Keep this file under 80 lines. It is a snapshot and an index, not a log: move old material to `notes/state-archive.md` instead of deleting it. Earlier clock corrections and the pre-handoff snapshot are archived there.

## Brief (at most 10 lines)

- Objective: the strongest well-validated probability model for high popularity; two final Kaggle selections chosen on grouped cross-validation; a report and Colab demo that explain what was tried, why, how models were selected, and which features matter in which direction.
- Constraints: the course rules and Sam's decisions in `AGENTS.md`; folds grouped by track (Finding 002); the public leaderboard is a check only.
- Success criteria: two final selections by Monday 2026-11-02, 5:00 PM PST; report PDF and public Colab link on Gradescope by Tuesday 2026-11-03, 11:59 PM PST.
- Next milestone: P2.9a source audit, implementation contracts and score-free pilots, then a measured launch proposal. Node: A100 MIG 2g.20gb, 32-core CPU quota, 45 GiB RAM. Python 3.12.16 and 85 hash-locked packages installed; .venv links to node-local /tmp after a shared-filesystem quota failure. P2.7 is complete (Finding 030 draft); report and Colab wait until development ends.
- Choices that could invalidate later work: the candidate freeze and the final selection rule (phase 2 plan, P2.6 and phase 3) need an R2 review before the selection is made.

## Decisions in force

Pointers into `decisions.md`: development resume (2026-11-01), superseding the session-wrap pause; Fable 5.1 xhigh consultant and independent reviewer (2026-11-01); Codex owns development (2026-10-31); P2.9 reopening supersedes the earlier candidate freeze; every submission gets R1 (2026-10-30); no pretrained models, report and Colab after development, Sam uploads, measured budgets and consultation before building (2026-10-29); no dependence on the earlier audio-only project (2026-10-30).

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
| P2.7 | Restricted nested evaluation of the frozen menu (plan section 8), full path: the selected procedure with fold averaging scores outer AUC 0.8418, 0.8377, 0.8399 (mean 0.8398, pooled 0.8391) against the anchor's 0.8334, 0.8251, 0.8334 (mean 0.8306); paired gain +0.0092, positive on every fold, group bootstrap 95% +0.0057 to +0.0127 (the production CV gain is +0.0096); the full refit +0.0010 over fold averaging, not on every fold; the inner rule chose `blend_eq_all` twice and the 0.25 Extra Trees blend once; decision: the selected procedure with fold averaging, no fallback; build 1.74 h and empty-cache replay 1.75 h equal within 1e-12 with byte-identical arrays, 294 fresh cross-fits each; C_all 394.3 s, C_L 364.3 s, projection 4.97 h against 8 | provisional (protocol, menu and rule reviewed in R2 at fd48199; the result is reviewed with phase 3's selection); Finding 030 draft | `artifacts/phase2/p27/` (`measure.json`, `result_build.*`, `result_replay.*`, `verification.json`, `replay.txt`, `decision.json`); `doc/llm_usage/2026-10-31-p2.7-consultation/` | (see git log) |
| Brainstorm | A separate read-only Claude Fable 5.1 session with three sub-agents (2026-10-30/31) on raising the AUC; checked here: the test file is an 80/20 row split with every train track removed (no shared uri), and on singleton validation rows every menu candidate scores 0.003 to 0.004 lower; the equal blend stays first on both scores (Spearman 0.965), though the rule's tie-break applied to singleton scores would favor the two-atom Extra Trees blend (Finding 029, a diagnostic; the rule is unchanged) | provisional (recorded; nothing adopted into the models or the frozen menu); Finding 029 draft | `doc/llm_usage/2026-10-31-auc-brainstorm/`; `analysis/brainstorm_checks_output.txt` | af7a466 |
| P2.9 plan | 12 families, 274 distinct initial configurations, guaranteed core refinement, feature comparisons, seed checks and revised evaluation; E fallback as Sam chose | reviewed (v1.2 at fe9c54a; v1.3 portability amendment R2 APPROVED at c6fa913, round 2) | `doc/plans/phase2-p29-other-methods.md`; `reviews/2026-11-01-p29a-node-portability.md` | c6fa913 |
| P2.9a node | Environment installed; 19 foundation checks pass; six synthetic GPU probes repeat exactly; H/E atoms repeat byte-identically on-node, but several differ from Mac; node E diagnostic 0.840896 vs historical 0.841062 | reviewed, Finding 031 and v1.3 amendment; R2 APPROVED at c6fa913 | `doc/plans/phase2-p29-node-contracts.md`; `doc/llm_usage/2026-11-01-p29a-node/`; `artifacts/phase2/p29/` | 74fc0b3 |
| P2.9a boundary | Raw-feature wrapper and strict group partitions; eight focused tests pass, including a deliberately leaky encoder control and legacy feature equivalence; not connected to the engine | provisional, Finding 032; outside migration R2 snapshot | `songpop/phase2/raw_features.py`; `raw_splits.py`; `scripts/check_p29_boundaries.py` | see git log |
| P2.9a device pilot | Fixed TabM B, 3,091/2,061 fitting rows, two fresh CPU/CUDA repeats each: exact within-device states/predictions/receipts; first-fold CPU equals old adapter; GPU process times about 2.24/2.37 times shorter | provisional, Finding 033; one recipe, no outer AUC or search budget | `analysis/p29_tabm_device_pilot.py`; `doc/llm_usage/2026-11-01-p29a-node/tabm_device_pilot.json` | see git log |
| Workflow | agent-workflow v1, homework profile; primary developer Codex after the 2026-10-31 handoff; reviewer Claude Fable 5.1 xhigh, with model and effort passed explicitly; earlier roles archived | migration reviewed; handoff owner-directed | `reviews/2026-10-30-agent-workflow-migration.md`; `decisions.md`; `.agent/review.conf` | ce04746 (migration) |

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
- The brainstorm's remaining questions (`doc/llm_usage/2026-10-31-auc-brainstorm/README.md`): report the singleton-row score beside the official metric in phase 3 (diagnostic only); transductive uses of `test.csv` remain a course-rule question. Reopening development is already authorized by Sam (2026-10-31); the older recommendation against reopening is superseded.

## Running

| What | Job or process ID | Log | Started |
|---|---|---|---|
| Legacy full suite unchanged; fixes/receipt guards R2 APPROVED at isolated d7786f4. Fable consultation complete. Raw scalar adapter checks and non-blocking review follow-ups running in isolation. | suite 87959/PID 25623; scalar checks 47509; receipt checks see live note | `notes/2026-11-01-p29-live-progress.md` indexes logs and worktree | 2026-11-01 03:25 UTC (suite) |

## Next action

Node setup, 2026-11-01: installed the four shared skills from `../agent-skills` into `/home/wzhou2/.codex/skills` at Sam's request. Links and all 45 file hashes verified; built-in skill files unchanged. Evidence: `doc/llm_usage/2026-11-01-skill-installation/`. This preceded the development resume below.

Sam resumed development on 2026-11-01. Codex owns development and STATE. Claude Fable 5.1 xhigh consults and reviews in separate sessions, with explicit model/effort and read-only plan mode. The old handoff is `notes/2026-10-31-p29-gpu-handoff.md`. Node environment installed and REVIEW_PYTHON corrected; .venv points to `/tmp/miniproject_1-p29-venv-20261101`, so rebuild after node loss. Historical locks remain unchanged.

1. Read `notes/2026-11-01-p29-live-progress.md` for active sessions and the isolated worktree; finish verification, consultation and R2. Migration v1.3, node contracts and Finding 031 remain APPROVED at c6fa913. Continue P2.9a/b recipe_v2/raw adapters, complete equivalence/isolation gates, executable manifests and representative cost pilots, followed by Sam's measured large-job proposal. Provisional starting code: boundary f420853; device prototype aab2370. Coverage targets are not launch authorization. E's procedure, fallback trigger and tie band are unchanged.
2. Engine baseline: engine sources remain unchanged since P2.7 completion commit 1ffaddd; subsequent commits record the handoff and P2.9 plan. The separate `../Mini_project_1_p29` worktree was clean at 9693f11 on `p2.9-reopened` during the handoff. Integrate work deliberately from this completed baseline. Same-node adapter equivalence stays strict; cross-platform portability is reported separately under the approved v1.3 amendment.
3. P2.7's saved result keeps fold averaging with no fallback (Finding 030, draft). Phase 3 and the two final selections still require their R1 review and Sam's decision. The singleton diagnostic and transductive proposals remain open as listed above. Migration fits are diagnostics; no new scientific search has launched. Consultation is distinct from formal approval.

Former deferred items (WEIGHT_CLEANUP comment, TabM receipt structure and combiner regularization) implemented and R2 APPROVED at isolated d7786f4; integration awaits the fixed-source suite. Non-blocking follow-ups: shared legacy policy, max-epoch/multiple-seed positive controls, validation cost and documentation. The tiny kNN batch-position difference remains untraced; Finding 034 reports its magnitude.

After the close: Sam asks (2026-10-31) for a read of the class's public Colab notebooks, posted in public Piazza notes from Tuesday 2026-11-03, against the rules (external data, lookups by identifier, code written for this dataset, pretrained models if the staff rule them out) and against each team's numbers on the private board; factual observations for Sam, who decides what to do with them; nothing from those notebooks enters the project's models or report beyond the Challenges discussion.


## 2026-11-01 snapshot before fast exploration mode, preserved from 30a5494

Sam switched the project to fast exploration mode (`decisions.md`, 2026-11-01). P2.9 is paused on branch `p2.9-paused`.

# State: CS 155 Miniproject 1

Updated 2026-11-01. Development paused at Sam's session-wrap request. CPU gate approval explicitly left pending. No broad-search launch authorized. Keep this file under 80 lines. It is a snapshot and an index, not a log: move old material to `notes/state-archive.md` instead of deleting it. Earlier clock corrections and the pre-handoff snapshot are archived there.

## Brief (at most 10 lines)

- Objective: the strongest well-validated probability model for high popularity; two final Kaggle selections chosen on grouped cross-validation; a report and Colab demo that explain what was tried, why, how models were selected, and which features matter in which direction.
- Constraints: the course rules and Sam's decisions in `AGENTS.md`; folds grouped by track (Finding 002); the public leaderboard is a check only.
- Success criteria: two final selections by Monday 2026-11-02, 5:00 PM PST; report PDF and public Colab link on Gradescope by Tuesday 2026-11-03, 11:59 PM PST.
- Next milestone: P2.9a source audit, implementation contracts and score-free pilots, then a measured launch proposal. Node: A100 MIG 2g.20gb, 32-core CPU quota, 45 GiB RAM. Python 3.12.16 and 85 hash-locked packages installed; .venv links to node-local /tmp after a shared-filesystem quota failure. P2.7 is complete (Finding 030 draft); report and Colab wait until development ends.
- Choices that could invalidate later work: the candidate freeze and the final selection rule (phase 2 plan, P2.6 and phase 3) need an R2 review before the selection is made.

## Decisions in force

Pointers into `decisions.md`: session wrap-up and CPU gate approval pending (2026-11-01), superseding the earlier resume; Fable 5.1 xhigh consultant and independent reviewer (2026-11-01); Codex owns development (2026-10-31); P2.9 reopening supersedes the earlier candidate freeze; every submission gets R1 (2026-10-30); no pretrained models, report and Colab after development, Sam uploads, measured budgets and consultation before building (2026-10-29); no dependence on the earlier audio-only project (2026-10-30).

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
| P2.7 | Restricted nested evaluation of the frozen menu (plan section 8), full path: the selected procedure with fold averaging scores outer AUC 0.8418, 0.8377, 0.8399 (mean 0.8398, pooled 0.8391) against the anchor's 0.8334, 0.8251, 0.8334 (mean 0.8306); paired gain +0.0092, positive on every fold, group bootstrap 95% +0.0057 to +0.0127 (the production CV gain is +0.0096); the full refit +0.0010 over fold averaging, not on every fold; the inner rule chose `blend_eq_all` twice and the 0.25 Extra Trees blend once; decision: the selected procedure with fold averaging, no fallback; build 1.74 h and empty-cache replay 1.75 h equal within 1e-12 with byte-identical arrays, 294 fresh cross-fits each; C_all 394.3 s, C_L 364.3 s, projection 4.97 h against 8 | provisional (protocol, menu and rule reviewed in R2 at fd48199; the result is reviewed with phase 3's selection); Finding 030 draft | `artifacts/phase2/p27/` (`measure.json`, `result_build.*`, `result_replay.*`, `verification.json`, `replay.txt`, `decision.json`); `doc/llm_usage/2026-10-31-p2.7-consultation/` | (see git log) |
| Brainstorm | A separate read-only Claude Fable 5.1 session with three sub-agents (2026-10-30/31) on raising the AUC; checked here: the test file is an 80/20 row split with every train track removed (no shared uri), and on singleton validation rows every menu candidate scores 0.003 to 0.004 lower; the equal blend stays first on both scores (Spearman 0.965), though the rule's tie-break applied to singleton scores would favor the two-atom Extra Trees blend (Finding 029, a diagnostic; the rule is unchanged) | provisional (recorded; nothing adopted into the models or the frozen menu); Finding 029 draft | `doc/llm_usage/2026-10-31-auc-brainstorm/`; `analysis/brainstorm_checks_output.txt` | af7a466 |
| P2.9 plan | 12 families, 274 distinct initial configurations, guaranteed core refinement, feature comparisons, seed checks and revised evaluation; E fallback as Sam chose | reviewed (v1.2 at fe9c54a; v1.3 portability amendment R2 APPROVED at c6fa913, round 2) | `doc/plans/phase2-p29-other-methods.md`; `reviews/2026-11-01-p29a-node-portability.md` | c6fa913 |
| P2.9a node | Environment installed; 19 foundation checks pass; six synthetic GPU probes repeat exactly; H/E atoms repeat byte-identically on-node, but several differ from Mac; node E diagnostic 0.840896 vs historical 0.841062 | reviewed, Finding 031 and v1.3 amendment; R2 APPROVED at c6fa913 | `doc/plans/phase2-p29-node-contracts.md`; `doc/llm_usage/2026-11-01-p29a-node/`; `artifacts/phase2/p29/` | 74fc0b3 |
| P2.9a boundary | Raw-feature wrapper and strict group partitions; eight focused tests pass, including a deliberately leaky encoder control and legacy feature equivalence; not connected to the engine | provisional, Finding 032; outside migration R2 snapshot | `songpop/phase2/raw_features.py`; `raw_splits.py`; `scripts/check_p29_boundaries.py` | see git log |
| P2.9a device pilot | Fixed TabM B, 3,091/2,061 fitting rows, two fresh CPU/CUDA repeats each: exact within-device states/predictions/receipts; first-fold CPU equals old adapter; GPU process times about 2.24/2.37 times shorter | provisional, Finding 033; one recipe, no outer AUC or search budget | `analysis/p29_tabm_device_pilot.py`; `doc/llm_usage/2026-11-01-p29a-node/tabm_device_pilot.json` | see git log |
| Workflow | agent-workflow v1, homework profile; primary developer Codex after the 2026-10-31 handoff; reviewer Claude Fable 5.1 xhigh, with model and effort passed explicitly; earlier roles archived | migration reviewed; handoff owner-directed | `reviews/2026-10-30-agent-workflow-migration.md`; `decisions.md`; `.agent/review.conf` | ce04746 (migration) |

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
- The brainstorm's remaining questions (`doc/llm_usage/2026-10-31-auc-brainstorm/README.md`): report the singleton-row score beside the official metric in phase 3 (diagnostic only); transductive uses of `test.csv` remain a course-rule question. Reopening development is already authorized by Sam (2026-10-31); the older recommendation against reopening is superseded.

## Running

| What | Job or process ID | Log | Started |
|---|---|---|---|
| Existing fixed-source legacy suite only; Fable follow-up APPROVED at 369e762. No new training. Scalar gate launch rejected, approval left pending. | suite 87959/PID 25623 | `notes/2026-11-01-p29-session-handoff.md` | suite 03:25 UTC; handoff 04:08 UTC |

## Next action

Node setup, 2026-11-01: installed the four shared skills from `../agent-skills` into `/home/wzhou2/.codex/skills` at Sam's request. Links and all 45 file hashes verified; built-in skill files unchanged. Evidence: `doc/llm_usage/2026-11-01-skill-installation/`. This preceded the development resume below.

Sam resumed development on 2026-11-01. Codex owns development and STATE. Claude Fable 5.1 xhigh consults and reviews in separate sessions, with explicit model/effort and read-only plan mode. The old handoff is `notes/2026-10-31-p29-gpu-handoff.md`. Node environment installed and REVIEW_PYTHON corrected; .venv points to `/tmp/miniproject_1-p29-venv-20261101`, so rebuild after node loss. Historical locks remain unchanged.

1. On Sam's resume, read `notes/2026-11-01-p29-session-handoff.md`. Inspect the existing suite and recheck before editing their sources. Development is committed through 7dbfc6d and protected by local branch `checkpoint/p29-scalar-20261101`; worktree `/tmp/miniproject_1-p29-adapters-20261101`. Seven scalar and eight receipt checks pass. Full equivalence, CUDA, engine/runtime/cache integration and pilots remain. The 135-fit gate did not start; Sam explicitly left approval pending. No broad scientific search is authorized. E's rule and fallback remain unchanged.
2. Engine baseline: engine sources remain unchanged since P2.7 completion commit 1ffaddd; subsequent commits record the handoff and P2.9 plan. The separate `../Mini_project_1_p29` worktree was clean at 9693f11 on `p2.9-reopened` during the handoff. Integrate work deliberately from this completed baseline. Same-node adapter equivalence stays strict; cross-platform portability is reported separately under the approved v1.3 amendment.
3. P2.7's saved result keeps fold averaging with no fallback (Finding 030, draft). Phase 3 and the two final selections still require their R1 review and Sam's decision. The singleton diagnostic and transductive proposals remain open as listed above. Migration fits are diagnostics; no new scientific search has launched. Consultation is distinct from formal approval.

Former deferred items (WEIGHT_CLEANUP comment, TabM receipts and combiner regularization) implemented and R2 APPROVED at isolated d7786f4; follow-ups APPROVED at 369e762. Integration awaits the fixed-source suite. Non-blocking F7/F8 follow-ups are in the handoff. Standalone scalar adapter at 7dbfc6d is provisional. The tiny kNN kernel remains untraced (Finding 034).

After the close: Sam asks (2026-10-31) for a read of the class's public Colab notebooks, posted in public Piazza notes from Tuesday 2026-11-03, against the rules (external data, lookups by identifier, code written for this dataset, pretrained models if the staff rule them out) and against each team's numbers on the private board; factual observations for Sam, who decides what to do with them; nothing from those notebooks enters the project's models or report beyond the Challenges discussion.


## Fast-mode running snapshot, 2026-11-01

# State: CS 155 Miniproject 1

Updated 2026-11-01, 04:20 UTC. Fast exploration mode, owner-directed. Codex owns development; Sam confirmed the other session is paused. P2.9 is saved and stopped.

## Objective and rules

- Improve generalizing AUC with quick feature and model trials. Select on mean fold AUC over saved grouped splits, seeds 155, 2026, 1029. Small gains remain useful and eligible, per Sam's latest clarification. Check split/model-seed consistency; prefer simplicity when evidence is tied. The 0.002 threshold is an early-update trigger, not an eligibility cutoff.
- Only provided training data. No external data, pretrained models, Spotify lookups, dataset-specific external code, fitting on test features or pseudo-labels. Every fitted transform stays inside its training fold.
- Fast-mode override in `AGENTS.md`; full owner prompt: `doc/llm_usage/2026-11-01-fast-mode/prompt.md`. Exploration compute on this node is authorized. No P2.9 gates, consultation, replay, nested evaluation or coverage targets.
- New experiments stop 2026-11-01 20:00 UTC (12:00 PST). Files ready 2026-11-02 20:00 UTC (12:00 PST). Kaggle closes 2026-11-03 01:00 UTC (Monday 17:00 PST).
- Sam uploads, at most 10 per day, and decides the two final selections. Each file gets automatic ID/count/probability checks, correlation with the current best file, and a CV/hash/public-score log. One quick Fable review of the two owner-selected final files, leakage and file correctness only, one round.
- Report and public Colab follow development, due 2026-11-03 23:59 PST. Preserve brief LLM-use evidence.

## Baselines

| Candidate | Mean CV | Per seed: 155 / 2026 / 1029 | Public | Evidence |
|---|---:|---|---:|---|
| Current equal blend E | 0.841062 | 0.842493 / 0.841728 / 0.838964 | 0.85377 | `configs/phase2/menu.json`, `submissions/README.md` |
| Existing constrained blend, lambda 0.1 | 0.841003 | 0.842426 / 0.841618 / 0.838964 | 0.85415 | same menu, existing learned-procedure bank |
| Strongest earlier single-seed LightGBM | about 0.8368 | see P2.4 bank | not submitted alone | date_count + date_te3; Findings 022/023, draft |

P2.7 restricted nested result: selected procedure 0.8398 versus H 0.8306; saved replay passed. Historical evidence, still provisional. No nested reevaluation in fast mode.

## Saved work and environment

- Main pre-exploration snapshot: `8c03e05`, including pending records and the incomplete legacy-suite log.
- `p2.9-paused` protects `1b47c17`, including d7786f4, 369e762 and the standalone adapter at 7dbfc6d. `/tmp/miniproject_1-p29-adapters-20261101` is clean. Do not continue or integrate it.
- Legacy suite PID 25623 no longer exists. Its saved log has no final verdict and three diagnosed failures. No restart. The 135-fit scalar gate never ran and is abandoned with P2.9.
- `.venv` is node-local Python 3.12.16. Use it for project code. Node allocation: 32 CPU equivalents, 45 GiB RAM, A100 MIG 2g.20gb. No installs planned.
- Previous STATE is preserved in `notes/state-archive.md`. Historical banks, data and submission files stay unchanged.

## Running

2026-11-01T04:37:41.287510+00:00: 48 fresh 15-fold trials complete. Only the three original-adapter TabM runs remain, in session 3532; `artifacts/fast/models.log` and each TabM directory's `progress.json`. No new search batches will be launched.

Seeded LightGBM (year-relative audio plus date counts): 0.839308 standalone. Seeded MLP: 0.810332 standalone, useful as a 10% addition. Combined CV 0.842837; adding the recovered Extra Trees recipe gives 0.842945. All three split means improve versus E. Three automatically checked public-check files are ready under `submissions/2026-11-01_fast_*.csv`, not uploaded or final-selected.

Historical scan: 678 saved runs examined; best recovered addition is Extra Trees, +0.000469 at weight 0.10, positive on all three splits. No training in this scan. All scores remain exploratory.

## Next action

Sam asked to finish quickly. No further search expansion. Finish the already-running batches and the saved-bank rescore, then prepare 2 or 3 files and final recommendations. The earlier action list below describes the work already launched.

1. Read `notes/2026-11-01-fast-exploration.md`. Run feature additions on the best LightGBM recipe using the saved production folds and existing fold-local date encoders.
2. Try small grids of cheap non-menu families or distinct variants, using installed tools and the existing TabM adapter. Skip new framework/adapter work.
3. For every candidate report standalone CV and fixed-weight additions to E and the existing constrained blend; save OOF/test predictions, config and runtime. Keep one results table, commit code with results.
4. Propose 2 or 3 public-check files and two final recommendations. Sam decides. Apply the single final-file review only after his selections.


## Before the authorized fast follow-up, 2026-11-01 05:01 UTC

# State: CS 155 Miniproject 1

Updated 2026-11-01, 04:49 UTC. Development is closed. Sam selected the new combined model and original ensemble. The single Fable leakage/file-correctness review returned APPROVED at c04f340. No training or review jobs remain.

## Rules and ownership

- Codex owns development. Fast-mode override in AGENTS.md and owner prompt in `doc/llm_usage/2026-11-01-fast-mode/prompt.md`.
- Train only on provided data; no external data, pretrained models, identifier lookups, external dataset-specific code, fitted test features or pseudo-labels.
- Retained validation: saved track-grouped folds, split seeds 155, 2026, 1029; fit all transforms inside training folds; choose by mean of 15 fold AUCs. Small gains remain eligible. Public scores are checks only.
- Sam asked to finish quickly: no further search expansion. He selects and uploads, at most 10 per day. Automatic file checks replace per-file exploration R1; only two final files get one quick Fable review, leakage and file correctness only, one round.
- Experiments deadline 2026-11-01 20:00 UTC; files deadline 2026-11-02 20:00 UTC; Kaggle close 2026-11-03 01:00 UTC. Work finished ahead of those limits. Report and public Colab due 2026-11-03 23:59 PST.

## Results and files

All new scores are exploratory; the repeated CV folds have been used for model development.

| Candidate | Mean CV | Per seed 155 / 2026 / 1029 | Gain vs E | Status |
|---|---:|---|---:|---|
| Original E | 0.841062 | 0.842493 / 0.841728 / 0.838964 | 0 | Sam's second final pick; public 0.85377 |
| New combined: 0.65 E + 0.25 year-relative/date-count LightGBM + 0.10 MLP | 0.842837 | 0.843593 / 0.843488 / 0.841429 | +0.001775 | Sam's first final pick; leakage/file review APPROVED |
| Combined with 0.10 extra date-feature Extra Trees, E reduced to 0.55 | 0.842945 | 0.843692 / 0.843725 / 0.841418 | +0.001883 | optional public check |
| 0.90 E + 0.10 MLP | 0.842299 | 0.843366 / 0.843136 / 0.840395 | +0.001237 | optional public check |

- Final pair: `submissions/2026-11-01_fast_combined.csv` (sha256 fc3d6730...) and `submissions/2026-10-31_p2.6b_blend_eq_all.csv` (d2b5c461...). Owner decision and full hashes: `artifacts/fast/final_selection.json`.
- Public-check alternatives: `submissions/2026-11-01_fast_combined_et10.csv`, `submissions/2026-11-01_fast_mlp10.csv`. All three new files passed automatic row/ID/probability checks; correlations with E are 0.997326, 0.996354 and 0.998652. No uploads performed.
- Completed 51 fresh 15-fold runs, four small initial blend checks and five seeded fixed blends. Both selected new components average estimator seeds 0, 1, 2. Standalone new LightGBM 0.839308; selected MLP 0.810332.
- Historical rescore: 678 saved runs, 658 distinct prediction sets, no training. Best recovered contribution: Extra Trees +0.000469 at weight 0.10. No hard gain threshold rejected candidates.
- Single results table: `notes/2026-11-01-fast-exploration.md`; every config, OOF/test array, runtime and warning: `artifacts/fast/`. Code: `analysis/fast_*.py`. Quick validation checks passed; malformed submission controls rejected. No nested reevaluation, byte replay or new framework/adapter.

## Preserved work

- P2.9 stopped, saved on `p2.9-paused` at 1b47c17; main initial snapshot 8c03e05. Adapter worktree is clean. No integration or further P2.9 work.
- Old suite PID 25623 exited without a final verdict. Its log is committed. The 135-fit P2.9 gate never ran.
- `.venv`: node-local Python 3.12.16, 32 CPU equivalents, 45 GiB RAM, A100 MIG 2g.20gb. No new packages installed.
- Older STATE snapshots and outstanding course/team/upload bookkeeping are preserved in `notes/state-archive.md`.

## Next action

1. Sam uploads the checked files and reports public scores; log them without using them to tune models. Final Kaggle selections remain his action. The selected CSVs are unchanged from reviewed commit c04f3404b3ffcc4b30f8e0401e8ad258f6e51c06; `reviews/2026-11-01-fast-final-files.md` records APPROVED and the response to the one non-blocking status-label finding.
2. Move to the report and public Colab. Read `doc/README.md`, `Project_1.pdf`, `latex/project1_template.tex`, the results table and LLM-use integration index. Do not reopen model development or P2.9.
3. LLM-use evidence is in `doc/llm_usage/2026-11-01-fast-mode/` and `doc/llm_usage/integration_index.md`. The authoritative final-pair status is `artifacts/fast/final_selection.json`; preparation sidecars preserve their earlier event status.


## Before follow-up final review, 2026-11-01 05:41 UTC

# State: CS 155 Miniproject 1

Updated 2026-11-01, 05:38 UTC. Authorized fast follow-up is finishing ahead of its 06:31 cutoff. All 48 CPU configurations and six seed-confirmation runs completed. GPU queue 79367 finished; queue 47454 is completing the transformer and final ResNet. Existing final selections are unchanged.

## Rules and ownership

- Codex owns development. Fast-mode override in AGENTS.md and owner prompt in `doc/llm_usage/2026-11-01-fast-mode/prompt.md`.
- Train only on provided data; no external data, pretrained models, identifier lookups, external dataset-specific code, fitted test features or pseudo-labels.
- Retained validation: saved track-grouped folds, split seeds 155, 2026, 1029; fit all transforms inside training folds; choose by mean of 15 fold AUCs. Small gains remain eligible. Public scores are checks only.
- Sam authorized a bounded follow-up after asking what remained worth testing. The fast-mode rules continue for this block. He selects and uploads, at most 10 per day. Automatic file checks replace per-file exploration R1; only two final files get one quick Fable review, leakage and file correctness only, one round.
- Experiments deadline 2026-11-01 20:00 UTC; files deadline 2026-11-02 20:00 UTC; Kaggle close 2026-11-03 01:00 UTC. Work finished ahead of those limits. Report and public Colab due 2026-11-03 23:59 PST.

## Follow-up checkpoint

- Best completed fixed blend: `artifacts/fast/f2_blend_z5_et25/`, CV 0.843814, per split 0.844708 / 0.844520 / 0.842215. +0.000978 versus approved combined, all splits up. Uses five-seed year-standardized LightGBM and new Extra Trees; still exploratory.
- All 48 initial CPU trials and six additional-seed runs are complete, no errors. Six five-seed averages saved. Optional CatBoost addition scores 0.843988 (+0.000174 vs tree blend, mixed split signs). GPU queues are the only running experiments.

## Results and files

All new scores are exploratory; the repeated CV folds have been used for model development.

| Candidate | Mean CV | Per seed 155 / 2026 / 1029 | Gain vs E | Status |
|---|---:|---|---:|---|
| Original E | 0.841062 | 0.842493 / 0.841728 / 0.838964 | 0 | Sam's second final pick; public 0.85377 |
| New combined: 0.65 E + 0.25 year-relative/date-count LightGBM + 0.10 MLP | 0.842837 | 0.843593 / 0.843488 / 0.841429 | +0.001775 | Sam's first final pick; leakage/file review APPROVED |
| Combined with 0.10 extra date-feature Extra Trees, E reduced to 0.55 | 0.842945 | 0.843692 / 0.843725 / 0.841418 | +0.001883 | optional public check |
| 0.90 E + 0.10 MLP | 0.842299 | 0.843366 / 0.843136 / 0.840395 | +0.001237 | optional public check |

- Final pair: `submissions/2026-11-01_fast_combined.csv` (sha256 fc3d6730...) and `submissions/2026-10-31_p2.6b_blend_eq_all.csv` (d2b5c461...). Owner decision and full hashes: `artifacts/fast/final_selection.json`.
- Public-check alternatives: `submissions/2026-11-01_fast_combined_et10.csv`, `submissions/2026-11-01_fast_mlp10.csv`. All three new files passed automatic row/ID/probability checks; correlations with E are 0.997326, 0.996354 and 0.998652. No uploads performed.
- Completed 51 fresh 15-fold runs, four small initial blend checks and five seeded fixed blends. Both selected new components average estimator seeds 0, 1, 2. Standalone new LightGBM 0.839308; selected MLP 0.810332.
- Historical rescore: 678 saved runs, 658 distinct prediction sets, no training. Best recovered contribution: Extra Trees +0.000469 at weight 0.10. No hard gain threshold rejected candidates.
- Single results table: `notes/2026-11-01-fast-exploration.md`; every config, OOF/test array, runtime and warning: `artifacts/fast/`. Code: `analysis/fast_*.py`. Quick validation checks passed; malformed submission controls rejected. No nested reevaluation, byte replay or new framework/adapter.

## Parallel adapter work

- Session `01a19580-8f99-7492-9bc9-409b9774fb23` reports Sam explicitly assigned it to finish neural adapters in fast mode. Adapter commits bf9aa18 and 2b463c1 integrated as e9eb1fe/a85b3ca. Eight scientific neural configurations run on GPU in two processes: sessions 79367 and 47454; queues/logs `artifacts/neural/followup_{0,1}.*`. Includes embedded TabM, MLP, ResNet and FT-Transformer. Cutoff remains 06:31 UTC; max 150 epochs and patience 15. No old P2.9 gates.

## Preserved work

- P2.9 stopped, saved on `p2.9-paused` at 1b47c17; main initial snapshot 8c03e05. Adapter worktree is clean. The paused branch remains unchanged; only the separately authorized neural adapter has now been integrated.
- Old suite PID 25623 exited without a final verdict. Its log is committed. The 135-fit P2.9 gate never ran.
- `.venv`: node-local Python 3.12.16, 32 CPU equivalents, 45 GiB RAM, A100 MIG 2g.20gb. No new packages installed.
- Older STATE snapshots and outstanding course/team/upload bookkeeping are preserved in `notes/state-archive.md`.

## Next action

1. Run `analysis.fast_followup` for the small boundary check and queue. Run the 48 configurations with `analysis.fast_exploration --queue artifacts/fast/followup_queue.json --workers 6`, retaining all three saved split seeds. Cutoff 06:31 UTC.
2. LightGBM confirmation session 14305 (`artifacts/fast/followup_lgb_confirm.log`), one worker: z-score year residuals and zero-L2 variant, estimator seeds 1–4. Initial z-score replacement CV 0.843284, +0.000447 over combined, all split means up; provisional. Compare each addition with the approved 0.842837 combined bank, E and the existing constrained reference. Average additional estimator seeds only for a few promising candidates. No repeated reviews or old P2.9 process gates.
3. Save results in the same notes table and commit code/results. Prepare improved public-check candidates if justified; Sam retains final-file selection. Existing review only covers the unchanged previous pair.
