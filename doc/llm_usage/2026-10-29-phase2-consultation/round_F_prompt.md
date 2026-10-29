Round F of the consultation: a hard critique of task P2.0, the phase 2 plan, before its formal review. Same role as before: you advise Claude, the author; do not modify files, run reviewers or seek approvals.

Since round E: Sam uploaded the P1.3 file at about 04:10 PDT with the Kaggle description `2026-10-29_p1.3_lightgbm_F2`; public AUC 0.83597. Sam reported a one-member Kaggle team named after him. Claude drafted P2.0 from round E and from its own positions, written before round E (`doc/llm_usage/2026-10-29-phase2-consultation/claude_positions_before_round_E.md`).

The P2.0 work product (uncommitted; `git status` shows it):
- New: `doc/plans/phase2-models-features.md` (the phase plan); `doc/llm_usage/2026-10-29-phase2-consultation/` (positions, round E prompt and answer, this prompt).
- Changed: `doc/plans/modeling_strategy.md` (status line; the departures of the plan's section 12 edited into sections 4, 5.2, 5.4, 5.6 and 6; a history note); `doc/project_plan.md` (status line, phase table rows 2 and 4, roles, history); `doc/README.md` (team line, layout row, status lines); `doc/plans/phase1-data-baseline.md` (status, open question 1); `submissions/README.md` (the approval, upload and public-score events).
- Still to come after this round: the consultation README, the integration-index row for P2.0, the review record.

Where the draft departs from your round E advice, deliberately:
1. The plan is its own task, P2.0, reviewed before any phase 2 code, rather than part of P2.1: Sam asked to plan phase 2 first, and a plan reviewed alone lets later tasks be checked against it.
2. One environment instead of a separate phase 2 environment: `CLAUDE.md` says project code runs with `.venv/bin/python`. `requirements.txt` stays unchanged (so the phase 1 configuration hashes, which include its sha256, stay valid) and additions go into `requirements-phase2.txt`, installed only after a constrained dry run shows no installed version changes. Open question 3 of the plan asks Sam about the `CLAUDE.md` wording.
3. The installation manifests for all three routes (CatBoost, EBM, TabM) are reviewed together in P2.1 and executed later, unchanged, in P2.3 and P2.5.

Please critique as hard as you can, reading every changed file in full (`git diff` and the new files):
1. Internal consistency: between the phase plan, the strategy (as edited), the master plan, the README, the phase 1 plan and the submissions log. Anything that now contradicts, is stale, or is promised in one place and missing in another.
2. Conformance to the approved strategy and to `doc/review_policy.md`: are the departures in section 12 complete and justified, and is every strategy edit recorded? Is anything in the plan a change the strategy's change-control rule (master plan, "Plan structure") does not allow in this task?
3. Statistical and machine-learning soundness: the search protocol and its optimism, the exact-date encoding and frequency count (leakage, the own-group exclusion, unseen dates, smoothing), the training weights, the ensembles and seed averaging, the nested evaluation (its design, the fallback rule, the deployment rule, the cost bound), the selection rule and its tie band, the promotion rule's use.
4. Feasibility of the schedule against the deadlines and the review turnaround; whether the cut order is right; anything that should be cut now.
5. The infrastructure design: whether phase 1 really stays verifiable at the phase 2 head (check `run_baselines.sources()`, the configuration hash fields, `artifacts.packages()`, `environment()`, and `prepare_submission.select()`), whether one environment with `requirements-phase2.txt` is sound, and anything missing from sections 4.1 to 4.9 that P2.1's review will need.
6. The P1.3 records: are the upload and score events stated correctly and with the right caveats?
7. Anything that would make the formal reviewer request changes, including the project's style rules (short sentences, tables, no dashes as punctuation) and unresolved paths.

You may run read-only commands and read-only Python. Deliver a numbered list of concrete changes, each marked must-fix, should-fix or optional, with file and line or section, the problem, and the exact fix. Then a short list of what you checked and found correct.
