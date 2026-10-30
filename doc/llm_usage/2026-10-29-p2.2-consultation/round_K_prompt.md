Round K of the consultation: your independent implementation and execution design for task P2.2, before Claude builds it. Same role as before: you advise Claude, the author; do not modify files, run reviewers or seek approvals. Sam asked again, verbatim: "remember to consult deeply with codex".

State: P2.0b (the amendment, plan v0.3 section 14, Finding 010) converged in one formal round and is committed (ee557e2, 281a014) on Sam's go-ahead, verbatim: "Commit and start P2.2". The working tree is clean apart from `reviews/2026-10-29-project-plan-assessment.md` (Sam's separate Codex chat; ignore it). Now: Thursday 2026-10-29, about 21:05 PDT. P2.2's target in the task table is Thursday evening to Friday morning, review included; no new exploratory batch after Saturday 08:00 PDT; menu freeze Saturday 18:00 PDT.

P2.2, as the plan defines it (task table; section 4.10): common search extensions (conditional sampling, admissible regions, local refinement batches, range-extension manifests, timing and allocation records, log shards and their merge); factory parameters of the installed families; the kNN family and its block-weighted representation; the broad searches and first refinements of LightGBM, random forest, XGBoost, Extra Trees, HistGB, the smooth logistic models and kNN (section 5); findings. P2.3 (CatBoost, EBM) waits for any common P2.2 extension it uses.

Read first: `AGENTS.md`; `doc/review_policy.md` sections 4, 6 and 8.1; the plan `doc/plans/phase2-models-features.md`, the task table with its schedule guards and cut order, sections 4 (in particular 4.2 to 4.5 and 4.10), 5, 6.2, 10 and 11; `doc/findings.md` Findings 007 to 010; `analysis/phase2_timing_output.txt`; the P2.1 code: `songpop/phase2/` (recipes, searches, models, engine, runs, banks, ensembles), `scripts/run_experiments.py`, `scripts/check_phase2.py`. You may run read-only commands and read-only Python, including small timing probes; the machine is shared with Claude's work, so keep probes light, and say which numbers you computed. Do not compute or print the AUC of any configuration from the P2.2 search spaces: no search score should exist before the batch manifests are saved.

Please give your own design for P2.2, concretely enough to code and execute from:

1. Scope and review staging. What P2.2 must contain to satisfy the plan. Whether and how to stage its formal review under the policy, for example the code, self-checks, timing records and saved initial manifests reviewed before any search score exists, then the results; how that is recorded (one record with several rounds, or a re-cut of the task in the plan); when commits happen; what lets P2.3 start building on the shared extensions.
2. Code, with module and function names and data formats:
   a. the factory parameters the section 5.2 spaces need beyond P2.1;
   b. the kNN family and its block-weighted representation of section 6.2: the preprocessing as a recipe field, recipe validation (base, seeds, weights), the Manhattan alternative and its block normalization, ties and zero distances, duplicate rows, determinism and the fit-time identical-input check;
   c. conditional sampling and admissible regions in search specifications (for example `subsample_freq` only when `subsample` is below 1; learning rate times iterations within 5 to 100 for boosting, anchors exempt), keeping `search_v1` expansions and `verify_searches` valid;
   d. local refinement proposals: parameter types, zero values, mixed-type parameters (`max_features` "sqrt" or a fraction; `max_depth` None or an integer), integer rounding, hard bounds, deduplication against the family's earlier configurations, the broad quarter, the saved proposal list;
   e. range-extension manifests;
   f. the batch manifest and its pre-batch state (section 5.6);
   g. the two contribution blends per configuration: saved banks or logged records, and their verification;
   h. allocation records: the four routes and the counter as pure functions of logged records, and their verification;
   i. timing records without exposing scores;
   j. log shards and their merge, or serial execution, after section 5.1's benchmark;
   k. verification at hundreds of runs, when `--verify-all` refits every run.
3. The execution protocol P2.2 must declare in writing before its runs, and where to write it: the reference H; the family anchor of each family, including the families without a phase 1 recipe (XGBoost, Extra Trees, HistGB, kNN) and logistic, whose phase 1 recipe the phase 2 `log_standard` preprocessing does not reproduce exactly; the order of batches and families, and whether the order can change allocation outcomes through the cross-family terms of section 5.6; when broad expansion, local refinement and the {0.67T, T, 1.5T} tree comparisons run; how at most four families are chosen for local refinement, their two parents and the split of each batch of 12; the range-extension rule; logistic's variants (`spline` F1 and `log_standard` F1 or F2) inside its initial 24; kNN's 48-point grid; sampler seeds; the stopping cutoff for P2.2.
4. The compute this implies, from Finding 010 and the spaces, including the expensive corners (for example 63 leaves and 1,600 trees), serial or two processes.
5. Storage: a bank is about 440 KB; hundreds of banks in a repository with no remote.
6. Findings: what P2.2 reports and how it is labeled; the optional pipeline submission of section 10.
7. Anything in sections 5.1 to 5.6 or 6.2 that is ambiguous or inconsistent and must be resolved in writing before the runs; anything that would make the formal reviewer request changes.
8. A time plan from now to P2.2's convergence, review turnaround included, and what to cut first if it slips.

Deliver a numbered, concrete design, each item with the recommendation, the reason and the cost. Then a short list of the decisions only Sam can make, if any. Mark anything you are unsure of.
