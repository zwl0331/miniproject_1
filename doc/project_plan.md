# Project plan

v0.1, 2026-10-28. Owner: Sam Zhou. Status: phase 0 (bootstrap, P0.1) in review.

This is the master plan: goal, phases, roles, risks. The task as assigned is `../Project_1.pdf`; the course rules and rulings are in `../CLAUDE.md` and `README.md`. The work of each phase is planned in its own file under `plans/` when the phase starts.

## Goal

A well-validated probability model for high popularity, submitted to Kaggle with two well-chosen final selections, and a report and Colab demo that explain the process: what was tried, why, how models were selected, and which features matter and in which direction.

## Phases

| Phase | When | Work | Output |
|---|---|---|---|
| 0. Bootstrap | 2026-10-28 | P0.1: the review framework, this plan, the git repository | P0.1 converged; initial commit |
| 1. Data audit and baseline | 2026-10-28 to 10-29 | Record the facts of the data as findings: the target encoding, repeated tracks and their labels, incomplete rows, the formats of the release date, the identifier columns. A cross-validation harness that keeps every row of a track in one fold. Baseline models. A first Kaggle submission | Findings; cross-validated AUC of the baselines; first public leaderboard score |
| 2. Models and features | 2026-10-29 to 11-01 | Features from the release date and transforms of the audio features; gradient-boosted trees, linear and other models; tuning; ensembling; feature influence and its direction | The best cross-validated model; the ranked features with their direction |
| 3. Final selection | by Monday 2026-11-02, 5:00 PM PST | Propose two final submissions on cross-validation, with the public leaderboard as a check only | Two selections, Sam's decision |
| 4. Colab and report | 2026-11-01 to Tuesday 2026-11-03, 11:59 PM PST | A Colab notebook that runs the best model and draws the report's figures; the report from the template, LLM Usage included; the Piazza note; the Gradescope upload | Report PDF, public Colab link, Piazza note |

Phases 2 and 4 may overlap: the report can be drafted while models are tuned.

## Plan structure

| File | Holds | Changes when |
|---|---|---|
| `project_plan.md` (this file) | Goal, phase table, roles, risks | The strategy changes: phases, dates, roles, risks |
| `plans/phase<N>-<slug>.md` | One file per phase: goal, exit criteria, task table, decisions, open questions, dated status log | A task of that phase starts, changes, or finishes |

Rules:

- A phase plan is written when the phase starts, not before. Until then the phase table above is its plan. Phase 0 has the single task P0.1 and no plan file.
- Tasks carry stable IDs `P<phase>.<n>`. Each task is one reviewable unit: one review record under `../reviews/`, with as many review rounds as convergence under `review_policy.md` needs. Review records and commit messages name the task ID, so every result traces back to a reviewed task.
- Status lines in the phase plans are dated, newest last.

## Roles

- Sam: owner of this repository; final authority under `review_policy.md` (waivers, overrides, the two final Kaggle selections, changes to the policy).
- Teammates: to be recorded. Whether their decisions count like Sam's under the policy is Sam's decision.
- Reviewer: Codex, gpt-6-astra at reasoning effort xhigh, per `review_policy.md`.

## Risks

| Risk | Mitigation |
|---|---|
| Six days from release to the close of the competition, and every task waits for a review round at xhigh effort | Tasks sized for one round; interim numbers labelled unreviewed; Sam may waive review for a named task (`review_policy.md` section 3.3) |
| Rows of the same track in different folds leak into validation and inflate the cross-validated AUC | Folds grouped by track (phase 1) |
| Overfitting the public leaderboard | Model selection on cross-validation; the public leaderboard as a check only |
| 10 submissions a day | Each submission reviewed, with a stated purpose |
| Course rules: no external data, no code written for this dataset, LLM use reported | The rules in `../CLAUDE.md`; the reviewer checks them (`review_policy.md` section 6.3, item 9); the review records as the evidence for the LLM Usage section |

## History

- v0.1, 2026-10-28: created with P0.1.
