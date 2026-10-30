Round M of the consultation: your independent design for a new small task, an identifier probe, before Claude builds it. Same role as before: you advise Claude, the author; do not modify files, run reviewers or seek approvals. Read only inside the project root.

Context: P2.2 passed its execution-readiness review and its searches are running now (wave 1 done, wave 2 running); do not run anything that fits models while you work, beyond tiny probes, and do not touch `artifacts/`. Sam asked whether all features, identifiers included, are used, worried about leaving performance on the table. Claude answered by argument: `uri` is used only as the grouping key for the folds, since no test `uri` occurs in train (Finding 002); `track_href` and `analysis_url` end in the `uri` token; `type` is constant; training row position has AUC 0.506 and test ID correlates -0.027 with release year (Finding 005); a within-file repeat count was left out because computing it for test rows uses the test file jointly and because 11% of train rows but 2.5% of test rows sit in repeated groups. Sam replied, verbatim: "Yeah, don't assume anything. Use tests and actual data to draw conclusions."

Read first: `AGENTS.md`; `doc/findings.md` (Findings 001, 002, 005, 007, 008); `doc/plans/modeling_strategy.md` section 3 (what is allowed); `doc/plans/phase2-models-features.md` sections 4.3, 6 and 15; `songpop/data.py`, `songpop/splits.py`, `songpop/phase2/features.py` (`RAW_COLUMNS`); `data/train.csv` and `data/test.csv` (columns only, or descriptive reads). Do not compute a model score with identifier features yourself; describe the tests instead.

Please give your own design, concretely enough to code from:
1. Every identifier-like or order-like signal worth testing in these files, and why (the `uri` token and anything derived from it, the URL columns, `type`, row order, test `ID`, repetition and group size, overlaps between the files, anything else you see).
2. For each, the exact test: features, model, folds, the null or comparator, the statistics to report, and what result would count as evidence of usable signal (pre-declared thresholds), with multiple-testing awareness.
3. Which uses would be legal under the course rules and the project's strategy (nothing fitted on test rows; no external data; every column allowed by the Piazza ruling), and which need Sam's decision.
4. How a positive result would enter the pipeline (the engine reads only `RAW_COLUMNS` by design) and how a null result should be recorded.
5. How to keep it separate from P2.2's frozen protocol and completion review (separate task id, record, fingerprints), and its cost.
6. Other assumption-based exclusions in the current plan that deserve a cheap test under Sam's instruction.

Deliver a numbered, concrete design with reasons and costs, then a short list of the decisions only Sam can make. Mark anything you are unsure of.
