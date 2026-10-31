Round X of the consultation: a hard critique of Claude's P2.7 driver, written after your round W design. You advise Claude, the author; do not modify files, run reviewers or seek approvals. Read only inside the project root; fit nothing; compute no new score; do not open `artifacts/phase2/p27/result_*` or `artifacts/phase2/cache/` (the build is running in the background and must not be disturbed).

Commit: 948f0d48376664351b77ab2ba443375f3fa24eb4

Context: the driver is `analysis/p27_execute.py` with its checks in `analysis/p27_checks.py`; `doc/llm_usage/2026-10-31-p2.7-consultation/README.md` lists what was adopted from round W and what was not, with reasons. The smoke stage passed: the whole recursion with real fits on 300 rows at 20 iterations (312 atomic cross-fits, the count expected for this menu, 1,476 cache hits), and a full-size learned procedure on the real outer fold 0 training part of 2,575 rows (six combiner settings learned and one learned full refit in 126 s; repeated from the disk cache in a fresh object with equal predictions). The driver checks passed (the path gate at its boundary, assignment refusals, the replay comparison on corrupted copies, the decide gate, the atomic cache). The measurement ran on an idle machine with the production fit contexts: every one of the eight menu atoms equals its logged bank within 1e-12; C_all 394.3 s, C_L 364.3 s; plan projection 7,163 s, pilot projection 6,357 s; build plus empty-cache replay with the 25% margin 4.97 hours against 8 reserved; path full, recorded in `artifacts/phase2/p27/measure.json` before any outer score (a first measurement whose bank comparison used the driver's own context names was retired to `artifacts/phase2/stale/p27_measure_r1/`). The build started at about 06:21 PDT through the namespace `artifacts/phase2/cache/p27_build`; the replay and the decide stages have not run.

Read: `analysis/p27_execute.py`, `analysis/p27_checks.py`, the consultation README, `artifacts/phase2/p27/measure.json`, `artifacts/phase2/splits/nested_outer_v1.csv`, the `nested_measure` record at the end of `artifacts/phase2/experiments.jsonl`, `songpop/phase2/nested.py`, `procedures.py`, `selection.py` and `songpop/metrics.py` as needed, and plan section 8.

Please critique the driver as it stands:
1. Any defect that would invalidate the evaluation or its replay, or make the saved result unusable for the later R1 review of phase 3's selection.
2. Any place the driver reads outer labels for scoring outside the decide stage, or lets inner evidence depend on outer validation rows, or alters the evaluator's populations, contexts or protocol.
3. The path gate and the saved assignment: what a wrong implementation could still slip past.
4. The replay comparison and the freshness count (the expected number of fresh cross-fits for a cold run of this menu): are they right for this menu, including the case where an outer fold selects a learned candidate?
5. The decide report: the pooled outer AUC, the fold-kept group bootstrap with the project's seed 155001, the paired differences, the agreement check between the build's and the replay's decisions, the caveats' wording.
6. Provenance and records: what the later R1 review would miss.
7. Driver checks still missing that would fail on a wrong implementation, in order of value.
8. Anything in the measurement record that undermines the recorded path decision.

Number the findings, mark each blocking or non-blocking, give the file and line, and say what to change. A defect in the run stage leads to a restart from the cache namespace, so say if a finding requires that. Then list anything only Sam can decide.
