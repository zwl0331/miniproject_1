# Project plan

v0.3, 2026-10-29. Owner: Sam Zhou. Status: phase 0 (P0.1, P0.2) and phase 1 (P1.1 to P1.3, `plans/phase1-data-baseline.md`) converged and committed 2026-10-29; the first submission uploaded by Sam, public AUC 0.83597. Phase 2 planned in P2.0 (`plans/phase2-models-features.md`), in review.

This is the master plan: goal, phases, roles, risks. The task as assigned is `../Project_1.pdf`; the course rules and rulings are in `../CLAUDE.md` and `README.md`. The methods, validation protocol, experiment order and decision rules for phases 1 to 3 are in `plans/modeling_strategy.md`. The work of each phase is planned in its own file under `plans/` when the phase starts.

## Goal

A well-validated probability model for high popularity, submitted to Kaggle with two well-chosen final selections, and a report and Colab demo that explain the process: what was tried, why, how models were selected, and which features matter and in which direction.

## Phases

| Phase | When | Work | Output |
|---|---|---|---|
| 0. Bootstrap | 2026-10-28 to 10-29 | P0.1: the review framework, this plan, the git repository (done 2026-10-29). P0.2: the modeling strategy | P0.1 and P0.2 converged and committed |
| 1. Data audit and baseline | Day 1: Sam's go-ahead (target Thursday 2026-10-29 evening or Friday 10-30) | Record the facts of the data as findings: the target encoding, repeated tracks and their labels, incomplete rows, the formats of the release date, the identifier columns. A cross-validation harness that keeps every row of a track in one fold. Baseline models. A first Kaggle submission | Findings; cross-validated AUC of the baselines; first public leaderboard score |
| 2. Models and features | Thursday 2026-10-29 to Sunday 11-01 (`plans/phase2-models-features.md`) | Encodings of the release date and transforms of the audio features; gradient-boosted trees, random forests, a smooth model, a bounded neural challenger; tuning; ensembling; a restricted nested evaluation | A frozen candidate menu with its selection rule; the nested evaluation |
| 3. Final selection | by Monday 2026-11-02, 5:00 PM PST | Propose two final submissions on cross-validation, with the public leaderboard as a check only | Two selections, Sam's decision |
| 4. Colab and report | After development ends: from the Kaggle close, Monday 2026-11-02, to Tuesday 2026-11-03, 11:59 PM PST (Sam's decision of 2026-10-29) | Feature influence and its direction (`plans/modeling_strategy.md` section 5.6) and the stress tests; a Colab notebook that runs the best model and draws the report's figures; the report from the template, LLM Usage included; the Piazza note; the Gradescope upload | The ranked features with their direction; report PDF, public Colab link, Piazza note |

Phase 4 starts only after development ends (Sam's decision of 2026-10-29, quoted in the history below). The tasks of phases 1 to 4 and their order are in `plans/modeling_strategy.md` section 6.

## Plan structure

| File | Holds | Changes when |
|---|---|---|
| `project_plan.md` (this file) | Goal, phase table, roles, risks | The strategy changes: phases, dates, roles, risks |
| `plans/modeling_strategy.md` | Constraints, what is allowed, the validation protocol, the method catalog with priorities, the experiment order, decision rules, the final selection | Evidence changes a priority or a rule, or Sam decides; each change is its own reviewed task |
| `plans/phase<N>-<slug>.md` | One file per phase: goal, exit criteria, task table, decisions, open questions, dated status log | A task of that phase starts, changes, or finishes |

Rules:

- A phase plan is written when the phase starts, not before. Until then the phase table above and the modeling strategy are its plan. Phase 0 has the tasks P0.1 and P0.2 and no plan file of its own; the modeling strategy is the strategy for phases 1 to 3, not a phase plan.
- Tasks carry stable IDs `P<phase>.<n>`. Each task is one reviewable unit: one review record under `../reviews/`, with as many review rounds as convergence under `review_policy.md` needs. Review records and commit messages name the task ID, so every result traces back to a reviewed task.
- Status lines in the phase plans are dated, newest last.

## Roles

- Sam: owner of this repository; final authority under `review_policy.md` (waivers, overrides, the two final Kaggle selections, changes to the policy).
- Kaggle membership: Sam alone, as Sam reported on 2026-10-29, verbatim: "Kaggle team is my name . I am the only member: Weilong (Sam) Zhou." The course-group arrangement is not recorded.
- Reviewer: Codex, gpt-6-astra at reasoning effort xhigh, per `review_policy.md`.

## Risks

| Risk | Mitigation |
|---|---|
| Six days from release to the close of the competition, and every task waits for a review round at xhigh effort | Tasks sized for one round; interim numbers labelled unreviewed; Sam may waive review for a named task (`review_policy.md` section 3.3) |
| Rows of the same track in different folds leak into validation and inflate the cross-validated AUC | Folds grouped by track (phase 1) |
| Overfitting the public leaderboard | Model selection on cross-validation; the public leaderboard as a check only |
| 10 submissions a day | Each submission reviewed, with a stated purpose |
| Course rules: no external data, no code written for this dataset, LLM use reported | The rules in `../CLAUDE.md`; the reviewer checks them (`review_policy.md` section 6.3, item 9); the review records as the evidence for the LLM Usage section |
| Pretrained models slip in through a library or an AutoML preset | Sam's decision of 2026-10-29 excludes every pretrained model (`plans/modeling_strategy.md` section 3); AutoML only with an explicit whitelist |
| The public leaderboard is a small, noisy subset of the test rows of unknown size | Selection on grouped CV with paired comparisons; public scores as a secondary diagnostic only (`plans/modeling_strategy.md` section 4) |

## History

- v0.1, 2026-10-28: created with P0.1.
- v0.2, 2026-10-29: P0.1 done; P0.2 added to phase 0; the modeling strategy added to the plan structure, the phases and the risks (pretrained models, leaderboard noise). Sam's decision on pretrained models, verbatim: "pre-trained models certainly include data that is external to the project." Sam's instruction on timing, verbatim: "Okay, don't do development today. Yet. Just finish the plan and then pause." Phase 1 starts on Sam's go-ahead; the phase dates above are targets. Status line updated with P1.1: P0.2 committed, phase 1 started; no strategy change. With P1.3, Sam's decision, verbatim: "No, from now on, don't worry about the collab or the report itself until the very end. Focus on development." Phase 4 moved to after the Kaggle close.
- v0.3, 2026-10-29, with P2.0: status line; phase 2 starts Thursday, a day early, because phase 1 finished early; its plan is `plans/phase2-models-features.md`; feature influence and the stress tests moved from phase 2 to phase 4 under Sam's decision quoted in v0.2; Kaggle membership recorded as Sam alone, as Sam reported.
