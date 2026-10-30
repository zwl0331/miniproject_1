# CS 155 Miniproject 1: song popularity from audio features

Owner: Weilong (Sam) Zhou. Kaggle team: named after Sam, with Sam as its only member, as Sam reported on 2026-10-29; the exact team-name string is to be confirmed. The assignment asks for groups of two to four (`../Project_1.pdf` section 2); the course-group arrangement is not recorded (`plans/phase2-models-features.md`, open questions).

Caltech CS/CNS/EE 155, Machine Learning & Data Mining, fall 2026. Miniproject 1 (`../Project_1.pdf`, released Tuesday 2026-10-27): predict whether a song is of low or high popularity from its audio features. Kaggle competition: `https://www.kaggle.com/competitions/cs-155-fa-26-miniproject-1`. The metric is AUC, so a submission carries the probability of high popularity for each test song, in the format of `../data/sample_submission.csv`. A TA benchmark sits on the public leaderboard. The private leaderboard decides a small extra credit (top 50% of groups, up to 5 points); the report and the Colab demo carry most of the grade.

## Layout

| Path | Content |
|---|---|
| `doc/` | This README; `project_plan.md` (master plan); `plans/modeling_strategy.md` (methods, validation protocol, experiment order); `plans/phase1-data-baseline.md` (phase 1 plan); `plans/phase2-models-features.md` (phase 2 plan); `findings.md` (evidence ledger, Findings 000 to 017); `review_policy.md` |
| `doc/prior_project/` | Evidence carried over from an earlier audio-only attempt at this miniproject (P2.8): copied result files with their hashes and a README of what is used; the source folder is not read |
| `doc/llm_usage/` | Evidence for the report's LLM Usage section: `integration_index.md` (one row per LLM-assisted component) and the verbatim prompts and answers of the consultations, one directory per task |
| `analysis/` | Analysis code: `scoping.py`, the scoping run behind the modeling strategy, with its output `scoping_output.txt`; `audit.py`, the data audit behind Findings 001 to 005, with its output `audit_output.txt`; `phase2_timing.py`, the runtimes behind the phase 2 budgets (P2.0b), with its output `phase2_timing_output.txt`; `p22_summary.py`, the P2.2 summary behind Findings 011 and 012, with its output `p22_summary_output.txt`; `p22_tie_check.py`, the tie sensitivity and the amended verification behind Finding 013 and plan section 15.7, with its output `p22_tie_check_output.txt`; P2.8 (plan section 16): `identifier_probe/` (the standalone identifier and order probes, `blocks.py`, `probe.py` and `probe_bank.py`, with the outputs of their modes), `p28_forest_check.py` (deterministic forest prediction, Finding 017) and `p28_feature_checks.py` (the feature checks moved from P2.4, Finding 016), each with its output |
| `songpop/` | The validation harness (P1.2): loading, features, groups and folds, models, cross-validation, metrics and the bootstrap, banks and submissions. `songpop/phase2/` (P2.1): recipes, feature blocks and fold-fitted date encodings, model factories with their preprocessing, the engine, bank_v2, blends, runs and searches, submission checks (`plans/phase2-models-features.md` section 4); P2.2 adds the kNN family, parameter domains, `search_v2`, local refinement (`refine.py`), batch manifests (`batches.py`), the allocation rule (`allocation.py`) and the execution protocol (`protocol.py`, section 15) |
| `scripts/` | Run as modules from the project root: `run_baselines.py` (legacy reproduction and production baselines, `--verify` to recompute without writing), `check_foundation.py` (the harness's self-checks); phase 2 (P2.1): `run_experiments.py` (runs, searches, blends, comparisons, verification), `check_phase2.py` (self-checks), `prepare_candidate.py` (submissions from a selection manifest), `install_locks.py` (hashed installation locks) |
| `configs/` | Phase 2 inputs (P2.1): `phase2/recipes/` (recipe JSON), `phase2/searches/` (search specifications), `phase2/install/` (the hashed route locks for CatBoost, EBM and TabM, executed in P2.3 and P2.5; `requirements.txt` stays unchanged); `phase2/p2.2/` (P2.2: the execution protocol and the timing cases); `phase2/p2.8/` (P2.8: the manifest, the identifier-probe menu and the part B recipes) |
| `artifacts/` | Produced by the scripts: `splits/production_v1.csv` (groups and folds), `bank/` (one prediction bank per run, never overwritten), `experiments.jsonl` (the experiment log), and the scripts' outputs. `phase2/` (P2.1): the phase 2 banks, searches and experiment log, and `install/` (the protected inventory and the dry-run reports); `phase2/batches/` (P2.2: the batch manifests, written before each batch's first fit), `phase2/selections/` (selection manifests of phase 2 submissions); `p2.8/identifier_probe/` (P2.8: the probe banks and results, outside the phase 2 log) |
| `submissions/` | Kaggle submission files, each with a sidecar JSON (provenance, description), written by `../scripts/prepare_submission.py`; `README.md` logs every event |
| `reviews/` | Review records, one file per task, required by `review_policy.md` |
| `tools/` | `codex_review.sh` runs the independent Codex reviewer non-interactively; `fingerprint.py` computes revision fingerprints |
| `data/` | `train.csv`, `test.csv`, `sample_submission.csv` as downloaded from Kaggle; never modified |
| `latex/` | The course's report template, `project1_template.tex` with `preamble.tex` |
| `Project_1.pdf` | The assignment |
| `requirements.txt` | Pinned Python packages for the ignored environment `.venv/` (Python 3.12) |
| `CLAUDE.md`, `AGENTS.md` | Rules for AI agents; `AGENTS.md` links to `CLAUDE.md` |

Ignored by git: `.venv/`, `.review_logs/` (reviewer output kept by `tools/codex_review.sh`), Python bytecode, notebook checkpoints, `.DS_Store`.

## Status

- 2026-10-27: assignment released; data downloaded into `data/`.
- 2026-10-28: a read-only first look at the data in conversation (not reviewed; its facts enter the ledger as findings in phase 1). Python environment `.venv` created from `requirements.txt`.
- 2026-10-28: Piazza ruling on the use of the release date and the identifiers (course rulings below).
- 2026-10-28: project made review-controlled on the framework of the Positron Conversion DIO project (P0.1, `../reviews/2026-10-28-p0.1-bootstrap.md`).
- 2026-10-29: P0.1 approved by Codex in one round and committed (4b724f6, 90b52f7), with Sam's approval of the policy and confirmation of the reviewer model.
- 2026-10-29: Sam's decision: no pretrained models of any kind, because "pre-trained models certainly include data that is external to the project."
- 2026-10-29: the modeling strategy drafted with two Codex consultation rounds and a scoping run (P0.2, `../reviews/2026-10-29-p0.2-modeling-strategy.md`), in review. Development starts on Sam's go-ahead.
- 2026-10-29: P0.2 approved after two Codex review rounds and committed (f29267b, 5fc0cef).
- 2026-10-29: Sam's go-ahead for Day 1; phase 1 started (`plans/phase1-data-baseline.md`). Sam uploads every submission. P1.1, the data audit and Findings 001 to 006, in review (`../reviews/2026-10-29-p1.1-data-audit.md`).
- 2026-10-29: P1.1 approved after two Codex rounds and committed (a369262, 34247f5).
- 2026-10-29: P1.2, the harness and the nine baselines (Findings 007 and 008), built after Codex consultation rounds C and D, in review (`../reviews/2026-10-29-p1.2-harness.md`).
- 2026-10-29: P1.2 approved after three Codex rounds and committed (7dd8b3d, 0ba74a1).
- 2026-10-29: P1.3, the first submission file (LightGBM F2) and the LLM integration index, in review (`../reviews/2026-10-29-p1.3-first-submission.md`).
- 2026-10-29: P1.3 approved after three Codex rounds and committed (f6add21, 185dffe). Sam uploaded the file at about 04:10 PDT; public AUC 0.83597 as Sam reported it, a pipeline check, not a selection signal (`../submissions/README.md`). Phase 1 complete.
- 2026-10-29: Sam's go-ahead for phase 2 planning. P2.0, the phase 2 plan (`plans/phase2-models-features.md`), built after Codex consultation rounds E and F, in review (`../reviews/2026-10-29-p2.0-phase2-plan.md`).
- 2026-10-29: P2.0 approved after two Codex rounds and committed (3e24fe6, df11816).
- 2026-10-29: P2.1, the phase 2 infrastructure and the anchor reproduced through it (Finding 009), built after Codex consultation rounds G and H; in review (`../reviews/2026-10-29-p2.1-infrastructure.md`). Sam's refinements to the phase 2 experiments, budgets and ensembles become a separate reviewed amendment, P2.0b.
- 2026-10-29: P2.1 approved after two Codex rounds and committed (719a934, 4e451ec).
- 2026-10-29: P2.0b, the amendment of the phase 2 plan (section 14) and the strategy on Sam's refinements, with the runtimes of `../analysis/phase2_timing.py`, built after Codex consultation rounds I and J; in review (`../reviews/2026-10-29-p2.0b-search-amendment.md`).
- 2026-10-29: P2.0b approved in one Codex round and committed (ee557e2, 281a014), on Sam's go-ahead, verbatim: "Commit and start P2.2".
- 2026-10-29: P2.2, the broad searches of the installed families and kNN (plan section 15, the execution protocol), built after Codex consultation rounds K and L; its execution-readiness checkpoint approved after two Codex rounds (`../reviews/2026-10-29-p2.2-searches.md`).
- 2026-10-30: P2.2 executed: 348 configurations in three waves, ended by the allocation rule before the cutoff (Findings 011 to 013); a calibration submission file prepared; at its completion review.
- 2026-10-30: P2.2 completion review, round 3: changes requested. On Sam's decision, verbatim: "Do B", plan section 15.7 accepts the 19 tie-sensitive forest runs by an amended verification, which passes (Finding 013). Round 4 in review.
- 2026-10-30: P2.2 approved in round 4 and committed (810d1ce, 72a822b). P2.8, the gap hunt (plan section 16), started on Sam's go-ahead, verbatim: "Yes, start P2.8", after Codex consultation rounds M and N and an audit of the earlier audio-only project (`prior_project/`); at its readiness review (`../reviews/2026-10-30-p2.8-gap-hunt.md`).
- 2026-10-30: P2.8's readiness approved in round 3; executed (deterministic forest check, 18 phase 2 runs, the identifier probe) and verified; Findings 014 to 017; at its completion review.

## Key numbers

Every number here cites its finding in `findings.md`.

| Quantity | Value | Source |
|---|---|---|
| Training rows; complete rows; High rate | 3,864; 3,863 (one row has no audio features); 35.2% High | Finding 001 |
| Track groups for cross-validation (same `uri` or identical audio and date) | 3,638, of which 35 have mixed labels | Finding 002 |
| High rate before 1990; from 2020 on | 1.000; 0.255 | Finding 004 |
| Largest train-versus-test KS distance | 0.043 (acousticness) | Finding 005 |
| Best scoping baseline, legacy protocol | 0.8268 mean fold AUC (LightGBM, audio and date) | Finding 006 |
| Date fields over audio only; ungrouped-fold optimism | +0.05 to +0.08; +0.0158 | Finding 006 |
| Best production baseline (mean fold AUC, primary groups, split seeds 155, 2026, 1029) | 0.8315, LightGBM on F2; random forest 0.8279, logistic 0.7819 | Finding 007 |
| Phase 2 anchor through the new engine | bit-identical to the phase 1 LightGBM F2 bank, mean fold AUC 0.8315 | Finding 009 |
| F2 date parts over F1, LightGBM | +0.0166 (paired bootstrap interval +0.0113 to +0.0218) | Finding 008 |
| Best P2.2 single recipe (LightGBM, exploratory) | 0.8339 mean fold AUC, +0.0024 over the anchor (interval -0.0003 to +0.0054) | Finding 011 |
| Best P2.2 blend with the anchor (0.75/0.25, Extra Trees anchor, exploratory) | 0.8351, +0.0036 | Finding 012 |
| Identifier and order probes (P2.8) | No test met its follow-up rule; the identifier-only model 0.5014 | Finding 015 |
| Exact-date target encoding on the anchor (P2.8, exploratory) | +0.0025 with `date_te10` (interval -0.0010 to +0.0062); +0.0012 with `date_te30` | Finding 016 |
| Best standalone recipe so far (P2.8, exploratory) | 0.8354, tuned LightGBM with the date count and `date_te30` | Finding 016 |
| Best fixed blend so far (P2.8, exploratory) | 0.8359, 0.75 tuned LightGBM + 0.25 Extra Trees anchor | Finding 016 |

## Course rulings

| Date | Source | Question | Ruling |
|---|---|---|---|
| 2026-10-28 | Piazza, instructors' answer by Harry Chen to an anonymous question | The assignment says popularity is predicted from audio features alone, without identifying features; may `track_album_release_date` and the Spotify identifiers (`uri`, `track_href`, `analysis_url`) still be used? | "You are allowed to use all columns of the provided csv file." External data is still not allowed, including looking a song up by its identifiers. |

## Deadlines and logistics

| Item | Rule |
|---|---|
| Kaggle | Closes Monday 2026-11-02, 5:00 PM PST; late hours cannot be used. 10 submissions a day per team. Two final selections for the private leaderboard. |
| Report and Colab | Due Tuesday 2026-11-03, 11:59 PM PST, on Gradescope, one submission per group; late hours are taken from every member. |
| Colab demo | Runs the best model and several of the report's figures; public read-only link posted in a public Piazza note with the team name; both links in the report. |
| Report sections (template) | Introduction (15 points, at most 1 page), Overview (15, half a page recommended, at most 1), Approach (20, at most 1), Model Selection (20, at most 1), Conclusion (20, at most 1; the top 10 features, with their direction for bonus points), Additional insights (optional, 5 extra credit), LLM Usage |
| Data | Only the provided training set for training and validation; no external data (course rulings above) |
