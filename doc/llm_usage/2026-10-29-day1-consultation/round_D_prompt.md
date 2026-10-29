Round D of the consultation: a hard code critique of task P1.2 before its formal review. Same role as before: you advise Claude, the author; do not modify files, run reviewers or seek approvals.

Since round C: P1.1 was reviewed (two rounds; one non-blocking fix to the incomplete-row fold expression), approved and committed (a369262, 34247f5). Its phase plan, doc/plans/phase1-data-baseline.md, fixes the P1.2 design in sections 4.1 to 4.9; that design is the specification the code must meet.

The P1.2 code (uncommitted; `git status` shows it):
- songpop/: __init__.py, data.py, features.py, splits.py, models.py, evaluate.py, metrics.py, artifacts.py
- scripts/: run_baselines.py (--protocol legacy|production, --verify writes nothing and compares with saved artifacts), check_foundation.py (13 in-memory self-checks)
- artifacts/: splits/production_v1.csv, bank/*.npz (nine production banks), experiments.jsonl, run_baselines_production.txt, run_baselines_legacy.txt, check_foundation.txt

Results so far: the legacy reproduction matches all 15 committed scoping values to four decimals. Production mean fold AUC: LightGBM F0 0.7519, F1 0.8149, F2 0.8315; random forest 0.7555, 0.8141, 0.8279; logistic 0.7300, 0.7810, 0.7819. Bootstrap: LightGBM F2 minus F1 +0.0166 [+0.0113, +0.0218]; random forest F2 minus LightGBM F2 -0.0036 [-0.0091, +0.0023]. All 13 self-checks pass. Note: after the production run, a bug in validate_submission (pandas groupby on a list of tuples) was fixed in songpop/artifacts.py, so the source manifest hash changed and the saved banks' run ids are stale; the author will delete and regenerate artifacts/ after this round. `--protocol legacy --verify` and `scripts.check_foundation` run without writing; `--protocol production --verify` will currently fail on the stale run ids.

Please critique as hard as you can, reading every file:
1. Conformance to the phase plan sections 4.1 to 4.9 and to the strategy (section 3 exclusions, section 4 validation): anything missing, different, or wrong.
2. Correctness bugs: grouping (union-find, keys, float equality), fold assignment (StratifiedGroupKFold on eligible rows, the incomplete row's fold), features (F2 edge cases), models (logistic pipeline, LightGBM determinism, RandomForest NaNs), run_cv (alignment of iloc with fold ids and row ids, test predictions), metrics (cached weighted AUC, the bootstrap's multiplicity sharing, percentile interval, invalid replicates), artifacts (bank schema, immutability, compare tolerances, submission bytes and validation, the identical-feature check), run_baselines (legacy fidelity, verify mode, logging, the P1.3 selection rule).
3. Reproducibility and provenance: is everything needed to reproduce a bank recorded? Is anything nondeterministic (RandomForest predict_proba summation with n_jobs, LightGBM threads) and is the verify tolerance right?
4. Self-checks: which of section 4.9's checks are missing, weak, or vacuous? Which would not catch the bug they target?
5. Anything that would make the formal reviewer request changes.
You may run PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.check_foundation and -m scripts.run_baselines --protocol legacy --verify, and any read-only Python. Deliver a numbered list of concrete changes, each marked must-fix, should-fix or optional, with file and line, the problem, and the exact fix. Then a short list of what you checked and found correct.
