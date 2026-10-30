# Phase 2: models and features

v0.3, 2026-10-29. Owner: Sam Zhou. Status: v0.1 drafted in task P2.0 on Sam's go-ahead, verbatim: "Yes, plan phase 2" and "Again, consult deeply with codex"; approved after two review rounds and committed (3e24fe6, df11816; `../../reviews/2026-10-29-p2.0-phase2-plan.md`). v0.2: the infrastructure adjustments of P2.1 (section 13), approved after two review rounds and committed (719a934, 4e451ec; `../../reviews/2026-10-29-p2.1-infrastructure.md`). v0.3: the amendment P2.0b (section 14) on Sam's refinements to the experiments, budgets and ensembles, in review (`../../reviews/2026-10-29-p2.0b-search-amendment.md`); sections 5 to 8, 10 and 11 and the task table are rewritten by it. Built after Codex consultation rounds E (an independent design) and F (a critique of the first draft), verbatim in `../llm_usage/2026-10-29-phase2-consultation/` with what was adopted.

## Goal

From the reviewed F2 baselines (Findings 007 and 008) to the strongest validated result a thorough, evidence-driven search can reach (Sam, section 14): broadly searched tree models, CatBoost, the smooth models, kNN and a neural challenger, cross-fitted date encodings, re-tuned model-and-feature combinations, fixed and learned ensembles, every one scored on the saved grouped folds; then a frozen menu and a restricted nested evaluation of the frozen selection rule. Phase 3 then only reproduces, validates, uploads and selects (`modeling_strategy.md` section 6).

## Exit criteria

1. Tasks P2.0b to P2.4, P2.6a, P2.6b and P2.7 converged; P2.5 converged, or its omission recorded with the reason. P2.3 may converge with a reviewed incompatibility finding and CatBoost or EBM omitted.
2. Every menu candidate has a complete bank on the 3,863 eligible rows, the saved primary folds and all three split seeds.
3. Every learned transformation is fitted only on its permitted training population, shown by self-checks.
4. Phase 2 banks, logs, selection manifests and submission files verify without writing.
5. The phase 1 commands still pass unchanged at the phase 2 head: `scripts.run_baselines --protocol production --verify`, `--protocol legacy --verify`, `scripts.check_foundation`, `scripts.prepare_submission --verify`.
6. Every learned ensemble procedure has been evaluated as a whole on held-out groups (section 7.2), and the restricted nested evaluation, with its third level for learned candidates, is complete, or its declared reduced path (section 8) is documented.
7. The menu, the selection rule, the deployment rule, the source revision, the package locks and the hashes are frozen, and the results are reviewed findings.

## Tasks

Each task is one reviewed unit, started only after the one it depends on has converged. Before each task is built, Codex is consulted on its design or code (Sam's instruction above); the formal reviewer is always a fresh Codex session. Times through Saturday are PDT; from Sunday 2026-11-01, when daylight saving time ends, PST. Target times are converged completion times, review included. Independent tasks may run in parallel from approved source revisions; no source file a run or a review fingerprints is edited while it runs.

| Task | Deliverables | Depends on | Target | Record |
|---|---|---|---|---|
| P2.0 | This plan's v0.1 | P1.3 | Thursday morning | `../../reviews/2026-10-29-p2.0-phase2-plan.md` |
| P2.1 | The infrastructure of section 4; the anchor reproduced (Finding 009) | P2.0 | Thursday midday | `../../reviews/2026-10-29-p2.1-infrastructure.md` |
| P2.0b | The amendment of section 14 to this plan and the strategy | P2.1 | Thursday afternoon | `../../reviews/2026-10-29-p2.0b-search-amendment.md` |
| P2.2 | Search extensions (factory parameters, conditional sampling, local refinement batches, timing and allocation records); the kNN family and its representation; the broad searches and first refinements of LightGBM, random forest, XGBoost, Extra Trees, HistGB, the smooth logistic models and kNN (section 5); findings | P2.1, P2.0b | Thursday evening to Friday morning | record named when opened |
| P2.3 | The CatBoost and EBM factory changes, installations and runtime checks; timing pilots; their searches and refinements; findings. Installation and timing may start before P2.2 converges; the searches wait for any common P2.2 extension they use | P2.1, P2.0b; P2.2 for shared search extensions | Thursday evening to Friday midday | as above |
| P2.4 | The feature stage of section 6: screening, the cross-family checks, preprocessing alternatives, removal checks, then re-tuning of up to four model-and-feature combinations; findings. Neural comparisons only after P2.5 converges; otherwise omitted | P2.2, P2.3; P2.5 for neural comparisons | Friday through Saturday morning | as above |
| P2.5 | The TabM wrapper, its installation, a timing and repeatability pilot, its search and confirmation (section 5.5); findings | P2.1, P2.0b | Thursday evening to Saturday morning | as above |
| P2.6a | The extensions of section 4.10: the learned-procedure engine of section 7.2, the population-keyed prediction cache, the selector, the nested evaluator of section 8 with its isolation tests and a cost pilot, on synthetic inputs and the anchor; no candidate scores | P2.1, P2.0b | Friday | as above |
| P2.6b | The ensemble pool and the declared procedures of section 7, their evaluation, final estimator-seed policies, the frozen menu and selection manifest (section 11), the nested-evaluation manifest; at most one submission | P2.2 to P2.4, P2.6a; P2.5 if done | First review request Saturday 14:00 PDT; menu frozen by 18:00 PDT | as above |
| P2.7 | The restricted nested evaluation of the frozen menu and the deployment comparison (section 8); the final evidence table; the frozen deployment manifests; findings | P2.6b | Saturday night to Sunday; first review request by 14:00 PST, final freeze by 18:00 PST | as above |

Schedule guards: no new exploratory batch starts after Saturday 08:00 PDT; by 10:00 every input P2.6b needs has converged or its optional omission is recorded; P2.6b reserves measured time for final estimator-seed evaluations, the production evaluation of learned procedures, the regeneration of the finalist menu and verification before its 14:00 review request. If that does not fit, expansions stop earlier; review time is not compressed. Package installations into `.venv` are serialized, so parallel tasks never race the inventory checks.

Cut order if a task slips: (1) expansion batches without recent measured progress; (2) further weak feature combinations and redundant removal checks; (3) extra configurations inside a within-family average; (4) expensive members inside a learned combiner; (5) unpromising neural refinements; (6) a learned ensemble procedure whose correct evaluation cannot be completed. The incomplete-row ablation is outside the default scope. XGBoost, Extra Trees and kNN are not cut before their initial evaluation. Never cut: grouped cross-fitting, the phase 1 verification, submission validation, the restricted nested evaluation.

## Decisions

- Sam uploads every submission (phase 1 decision, unchanged).
- Interpretation (strategy section 5.6) and the stress tests (strategy section 4, step 11) move to phase 4, following Sam's decision of 2026-10-29, verbatim: "No, from now on, don't worry about the collab or the report itself until the very end. Focus on development." Phase 2 keeps what they will need (section 9). Sam may move them back.
- Kaggle membership: Sam alone, as Sam reported on 2026-10-29, verbatim: "Kaggle team is my name . I am the only member: Weilong (Sam) Zhou." The course-group arrangement is not recorded.
- One environment, `.venv`, extended only by the reviewed route locks of section 4.9; `requirements.txt` stays unchanged, so the phase 1 configuration hashes, which include its sha256, stay valid. A package that cannot be added without changing an installed version is dropped in phase 2, not given its own environment. `../../CLAUDE.md` is unchanged; `../README.md` lists the lock files once they exist.
- No pretrained model of any kind and nothing fitted on test rows (strategy section 3), unchanged.
- Budgets come from measured costs, not from fixed trial counts; Sam, verbatim: "Use measured runtimes to set sensible budgets; do not assume computation or agent effort is the main bottleneck." (section 14).

## 4. Infrastructure (P2.1)

### 4.1 Layout

Phase 1 files stay unchanged: the top-level `songpop/*.py`, `scripts/run_baselines.py`, `scripts/check_foundation.py`, `scripts/prepare_submission.py`, `requirements.txt`, `artifacts/bank/`, `artifacts/experiments.jsonl` and `artifacts/splits/production_v1.csv` (read and checked against its hash, never written). `run_baselines.sources()` hashes the top-level `songpop/*.py` only, so a subpackage leaves the phase 1 run ids valid. A fix found necessary in a phase 1 file is its own reviewed change; phase 1 is then replayed at its replay commit (section 4.5).

| Path | Content |
|---|---|
| `songpop/phase2/` | `recipes.py` (load and hash recipes), `features.py` (feature blocks), `encoders.py` (fold-fitted date encodings), `models.py` (estimator factories composed with the recipe's preprocessing; optional packages imported inside their factories), `engine.py` (fitting, fit-time checks, cross-validation), `banks.py` (bank_v2 metadata, loading with an explicit fold count, validation), `ensembles.py` (blends, also on a declared subset of split seeds), `runs.py` (data and saved folds, run identities and metadata, the log, verification and replay lookup), `searches.py` (search expansion), `submission.py` (the checks of section 4.7); from P2.6, `selection.py` (the frozen rule); from P2.7, `nested.py` (section 8) |
| `scripts/` | `run_experiments.py` (runs, searches, blends, comparisons, `--verify --run-id`, `--verify-all`), `check_phase2.py` (self-checks), `prepare_candidate.py` (submissions from a selection manifest), `install_locks.py` (section 4.9) |
| `configs/phase2/` | `recipes/*.json`, `searches/*.json`, `install/` (route locks); from P2.6, `menu.json` |
| `artifacts/phase2/` | `bank/`, `searches/` (each search's ordered configurations), `selections/`, `install/` (the protected inventory, the dry-run reports, the inventories before and after each installation), `experiments.jsonl`; from P2.7, `deploy/` (full-refit predictions) and `splits/` (nested-evaluation folds) |

Phase 2 runs are never appended to `artifacts/experiments.jsonl`: the P1.3 selector expects exactly one production F2 run per baseline family there.

### 4.2 Recipes, searches and identities

- A recipe is a JSON file, not executable code: model family and parameters, feature blocks, a drop list, preprocessing, training weights, estimator seeds, the iteration or stopping policy, the deployment policy. The family, the base and blocks with their drop list, and the preprocessing are independent fields. The drop list removes named feature columns for feature-removal checks. Preprocessing is `none`, `log_standard` (the phase 1 log transforms, median imputation, scaling, one-hot key, mode and time signature) or `spline` (the same with cubic splines before scaling, `n_knots` searchable); a family without native missing-value support needs a preprocessing that imputes. Duplicate JSON keys, non-finite numbers and boolean parameters are rejected. `recipe_id` carries the sha256 of its canonical JSON, without the human name.
- `run_id` additionally hashes the data, the evaluation and fitting populations, the fold assignments, the requested split seeds, the resolved estimator parameters, the estimator seeds, the source manifest, the relevant package versions and the parent artifacts. A seed-155 screening run and its three-seed confirmation have different run ids.
- A search file names the family, the space (a `preprocessing.` prefix sets a preprocessing parameter), the number of draws and the sampler seed; the ordered list of configurations is saved before anything is fitted, and draws whose canonical recipe already exists are skipped. A failing configuration is logged and stops the search. Phase 1 anchors are copied exactly, even where they fall outside the sampled grid.

### 4.3 Engine

- `fit_recipe(recipe, rows, y, groups, context)` receives only its fitting population: raw rows with their permanent row ids, labels and groups. It never sees outer validation labels or the full dataset. `context` carries seeds and any permitted inner splits. Every learned step (encodings, imputation, scaling, splines, network preprocessing) is fitted inside it.
- The fitted predictor exposes `predict_high(raw_rows)`, taking raw source columns and returning the probability of numeric class 1, located in `classes_` as `songpop.models.predict_high` does. The phase 4 permutation analysis can then permute an original column and rebuild its derived features without refitting.
- `run_recipe_cv(recipe, data, folds, split_seeds)` loops over repeats and folds, routes groups, training weights and inner splits explicitly, and validates its inputs as `songpop.evaluate.validate_cv_inputs` does. Any subset of the three split seeds is allowed, in the official order and selected by seed value; the bank says which, and selection rejects a candidate without the complete three-repeat bank, whatever its metadata says.
- At fit time every estimator seed is checked: test rows with identical model inputs (after the feature blocks and the recipe's preprocessing) get identical predictions. Each fit leaves a receipt: its split seed and fold, the hash of its fitting rows, and for each target encoding its inner seed and the hash of its inner splits as permanent row ids.

### 4.4 Banks

- The phase 1 array schema is kept (`train_row_id`, `y`, `group_id`, `split_seeds`, `fold_id`, `oof`, `test_id`, `test_by_fold`, `test_mean`, `fold_auc`), with schema `bank_v2` metadata that adds: fold and repeat counts; the evaluation and fitting populations; the recipe, its id and the run id; feature lineage from each derived feature to its source columns; estimator seeds; the prediction kind (fitted model, seed average, blend); parent bank hashes; search and stage identity; selection eligibility; inner-split hashes where used; the source manifest of the executing files (the phase 2 modules and the top-level modules they import); package versions including the new ones.
- Validation: `validate_bank(bank, n_splits=K, require_metadata=False)` from `songpop.evaluate`, unchanged, then `validate_v2_metadata(bank)`, which also checks that the linked fields agree: the config's recipe hash, estimator seeds, split seeds, fold-matrix hash, evaluation population, source-manifest digest and parents against the bank's own; canonical split seeds and stage; the features against the recipe; one fit receipt per fold; for blends, distinct parents with positive weights summing to one. `songpop.artifacts.load_bank` validates with five folds, so phase 2 loads through its own loader with the explicit fold count.
- Tolerances: identity fields and stable metadata exact; predictions of classical models within `rtol=0, atol=1e-12`, as in phase 1; a neural tolerance only after a repeatability pilot, declared before its bank is accepted.
- Banks are never overwritten.

### 4.5 Log, verification and replay

- `artifacts/phase2/experiments.jsonl` holds run, search, comparison, selection and submission records.
- `run_experiments --verify --run-id ID` reads the bank's envelope first, compares its sources and recorded Python and package versions with those present, then recomputes the run in memory and compares it with its bank and record, unrounded scores included. `--verify-all` also checks every comparison, every search file against its re-expansion, and every submission's files against their recorded hashes. A bank is saved only if the project sources loaded at the end of the run equal those at its start.
- The execution head recorded in a bank is context, not proof of the executing source: the phase 1 LightGBM F2 bank records 34247f5, a commit before its harness was committed in 7dd8b3d. Replay uses a commit whose files match the recorded source manifest. That association is recorded after the reviewed code is committed, without rewriting any bank, and checked before a replay command is printed. A source or dependency mismatch exits with a nonzero status and never counts as verification.

### 4.6 Ensembles

`blend_banks(banks, weights)` averages probabilities per repeat and fold, only across banks with identical row ids, labels, groups, split seeds, folds and test ids, and recomputes every AUC from the blended predictions; member AUCs are never averaged. A blend or a paired comparison may first restrict every bank to a declared subset of split seeds, so a challenger screened on one seed meets a three-seed recipe on identical folds; a comparison is logged once, identified by both bank hashes, the seeds and the bootstrap settings. Estimator seeds are averaged within each exact base recipe, then the declared weights apply across base recipes; recipes are never merged because they share a model family. Parent hashes, weights and averaging order are recorded. Rank averaging stays excluded (strategy section 5.5).

### 4.7 Submissions

`prepare_candidate.py` reads an explicit selection manifest (`artifacts/phase2/selections/<name>.json`: schema `selection_v1`, name, task id, the run id of one logged fitted recipe or declared blend, purpose, date), writes the file once (exclusive create), reads it back, validates it, and logs a submission record with the manifest, bank, CSV and sidecar hashes. Before rebuilding anything, each member bank must hold the production rows, labels, groups, test ids, data hashes and saved folds, and its source files and dependencies must be those present. The checks:

- the phase 1 checks: 863 rows, `ID, Popularity_Type` in that order, no index column, the sample's ID order, unique IDs, finite probabilities in [0, 1], the hash;
- the probability of numeric class 1, extracted explicitly for every member;
- equality with the deployment predictions rebuilt from the banks or the deployment artifact;
- for every fitted member and fold, its actual model inputs, rebuilt by refitting the feature pipeline on the fold's training rows and the recipe's preprocessing on that cross-fitted training matrix: identical inputs (NaN equal to NaN) get predictions equal within the declared tolerance; the bank holds each recipe's seed average per fold, so this checks the average, and the fit-time check of section 4.3 covered each estimator seed; the composite is checked on rows equivalent for every member; IDs and provenance columns never enter the keys;
- the 11 repeated test `uri` pairs, enumerated independently: equal predictions within each pair for every member and the composite;
- an additional raw-source invariant over the 13 audio columns and the raw date string.

### 4.8 Self-checks

`scripts/check_phase2.py`, in memory, like `check_foundation.py`; checks of writing code run against an in-memory file system. P2.1 runs every package-independent check; the CatBoost, EBM and TabM runtime checks become mandatory in the task that installs each route. Checks of code built in a later task are reported as STAGED until it is built, never as passed.

- Label isolation, with fold assignments held fixed: changing outer-validation labels changes no outer-validation or test prediction.
- Target encoding: a hand-computed example with mixed-label duplicate groups and unseen dates; a row's whole group excluded in its inner folds; unseen dates get the fitting prior.
- Date count: the own group excluded for training rows; zero for an unseen date.
- Neural early stopping: preprocessing fitted only on the inner training rows.
- Nested evaluation: fitting ids stay inside their outer training rows.
- Rejections: partial-seed candidates, mismatched bank identities, invalid weights, altered parent hashes, wrong deployment artifacts, a submission with one duplicate pair's prediction changed, a source or dependency mismatch in verification (nonzero exit).
- Blends: a bank blended with itself returns it; member order changes nothing beyond the declared tolerance; weights sum to one; seed averaging keeps recipe weights; refit and fold-average deployments stay apart.
- CatBoost writes no file (in P2.3).
- The LightGBM F2 baseline recomputed through the new engine equals the phase 1 bank `production_lightgbm_F2_9c37e77d549e` exactly.

### 4.9 Installations

Each installation is a gated action (`../review_policy.md` section 4.2). P2.1 resolves all three routes together against one compatible set of exact pins and writes one lock per route in `configs/phase2/install/`, each with its complete additive dependency closure and wheel hashes; a dependency shared by two routes has the same pin in both. P2.1's review covers the locks and the exact commands; P2.3 and P2.5 execute them unchanged.

1. Record the protected inventory: every installed distribution and version (`.venv/bin/python -m pip list --format=json`).
2. Resolve all routes jointly with every installed version constrained (`.venv/bin/python -m pip install --dry-run --report <file> --only-binary=:all: -c <inventory constraints> <packages>`), then each route against the protected inventory plus the joint additions. Each lock lists exact wheel URLs with their sha256, checked against the wheel tags this interpreter supports. `scripts/install_locks.py` runs these steps and the checks of steps 4 and 5.
3. Reject a resolution that changes an installed version or pins a shared dependency differently in two routes.
4. Before execution: the installed inventory must equal the protected inventory plus a subset of previously approved additions; it is saved as the route's before-inventory.
5. Execute: `.venv/bin/python -m pip install --no-deps --require-hashes -r configs/phase2/install/<route>.txt`, which resolves nothing further. After it, the inventory must equal the before-inventory plus exactly the route's lock, and is saved.
6. After execution: `pip check`, imports, a small fit on F2 with its missing values, predictions after a reload, and the phase 1 commands of exit criterion 5.

| Route | Packages | Executed in |
|---|---|---|
| CatBoost | `catboost==1.2.10` and its additive closure | P2.3 |
| EBM, optional | `interpret-core==0.7.8` and its additive closure | P2.3 |
| TabM | `torch==2.14.1`, `tabm==0.0.3`, its numerical-embedding dependency, and their additive closure | P2.5 |

Optuna is not installed unless a promoted family demonstrably needs it for refinement, which would be a new reviewed lock. SHAP waits for phase 4; `pytabkit` is not installed. The versions come from consultation round E's reading of the package metadata; compatibility is established only by steps 2 to 6.

### 4.10 Extensions after P2.1

P2.1's approved scope stands. Later tasks add, each under its own review:

| Task | Additions |
|---|---|
| P2.2 | Common search extensions (conditional sampling, admissible regions, local refinement batches, range-extension manifests, timing and allocation records, log shards and their merge); factory parameters of the installed families; the kNN family and its block-weighted representation |
| P2.3 | CatBoost and EBM factory parameters; their runtime checks |
| P2.4 | Further smoothing values and preprocessing alternatives; removal recipes; re-tuning manifests |
| P2.5 | The TabM wrapper, stopping receipts, repeatability checks, budget records |
| P2.6a | Learned-procedure schemas, banks and recursive deployment validation; generic internal populations and fold contexts, separate from production eligibility; the selector (`selection.py`) and the nested evaluator (`nested.py`); raw-input prediction and feature lineage through procedures; dependency identity including SciPy; label-isolation tests at every level; refusal of caches from another fitting population; exactly-once meta out-of-fold coverage with whole groups excluded; duplicate consistency and refusal of compensating member corruption; separate full-refit and fold-average artifacts |
| P2.6b | The pool rule, the enumerated procedure manifest, final seed policies, complexity ranks, the frozen menu |

Section 13's staging of the selector and the nested evaluator to P2.6 and P2.7 is superseded by this table.

## 5. Search and tuning (P2.2, P2.3, P2.5)

### 5.1 Budget and protocol

Budgets come from measured costs, not from a fixed trial count (Sam, section 14). The counts below include anchors. Initial batches run first; an expansion batch runs only under the allocation rule of section 5.6; the ceilings bound broad search and local refinement together on the base features, before the feature stage. They are ceilings, not targets.

| Family | Task | Initial | Expansion batch | Ceiling |
|---|---|---|---|---|
| LightGBM | P2.2 | 64 | 32 | 160 |
| Random forest | P2.2 | 48 | 24 | 120 |
| XGBoost | P2.2 | 48 | 24 | 120 |
| Extra Trees | P2.2 | 32 | 16 | 80 |
| HistGB | P2.2 | 12 | 12 | 36 |
| Logistic, `spline` or `log_standard` | P2.2 | 24 | 12 | 48 |
| kNN | P2.2 | 48 (the full grid of section 6.2) | 16 | 80 |
| CatBoost | P2.3 | 48 | 24 | 120 |
| EBM | P2.3 | 12 | 6 | 30 |
| TabM | P2.5 | section 5.5 | | |

That is 336 initial classical configurations, 276 in P2.2 and 60 in P2.3, and at most 794 with every expansion. Measured costs for one 15-fit run, all three split seeds, four model threads, one process on the Apple M5 (`../../analysis/phase2_timing.py` at the committed P2.1 code, output `../../analysis/phase2_timing_output.txt`; part of this amendment's review): LightGBM 2.2 to 27.0 s (7.5 s for the anchor; 10.4 s with the date count and target encoding; 23.4 s with three estimator seeds); random forest 8.2 to 26.9 s; Extra Trees 10.7 s; XGBoost with 1,000 trees 6.8 s; HistGB 4.6 s under the four-thread limit; logistic regression 0.4 to 0.6 s; a kNN proxy 0.5 s. A run counts 15 outer fit positions; with three estimator seeds it makes 45 estimator fits. These are representative cases (Finding 010, draft until this amendment converges): the 30 to 60 minutes of serial fitting estimated for the initial batches with known costs extrapolates from them and is not an upper bound for the enlarged spaces. Each task times its actual preprocessing and its declared expensive configuration before fixing a batch's reservation; CatBoost and EBM are timed in P2.3 before their batches are sized.

Protocol:

- Every configuration runs on all three split seeds with estimator seed 0. A subspace whose representative 15-fit run costs more than 120 s, timed before any of its scores, may be screened on split seed 155 (section 5.5 for TabM). After each such batch, the union of the best two standalone configurations, the best two by declared blend contribution and the named family anchor, ranked cumulatively over the family's screened configurations by unrounded scores with recipe id breaking ties and deduplicated, is completed on the other two split seeds; only configurations still missing repeats are run. The completion cost is reserved before the batch starts, and the allocation rule of section 5.6 uses only complete three-repeat results.
- Configurations come from a seeded sampler over the spaces of section 5.2: scale parameters on geometric lists; for boosting, combinations with learning rate times iterations outside 5 to 100 are skipped before fitting, anchors excepted. The ordered list of each batch is saved before its first fit.
- Before a family's first batch, a light, a representative and an expensive configuration are timed, feature construction and prediction included; the batch sizes above are then confirmed or reduced, never raised, by that measurement.
- At most two fitting processes of four threads each run at once, after a benchmark against serial execution; each writes its own log shard, merged with a check.

### 5.2 Spaces, iteration counts and refinement

| Family | Initial space |
|---|---|
| LightGBM | `num_leaves` {7, 15, 31, 63}; `min_child_samples` {10, 20, 40, 80, 120, 200}; learning rate {0.01, 0.02, 0.03, 0.05, 0.07, 0.1}; trees {300, 600, 1000, 1600}; L1 {0, 0.01, 0.1, 1, 5, 10}; L2 {0, 0.1, 1, 10, 30, 100}; column fraction {0.6, 0.8, 1}; row fraction {0.7, 0.85, 1}, with `subsample_freq=1` when below 1 |
| Random forest | 500 trees; leaf size {1, 2, 3, 5, 8, 12, 20, 30}; `max_features` {"sqrt", 0.3, 0.5, 0.75, 1.0}; depth {None, 8, 16, 24}; `max_samples` {0.6, 0.8, 1.0} |
| Extra Trees | 500 trees; no bootstrap; leaf size {1, 2, 3, 5, 8, 12, 20}; `max_features` {"sqrt", 0.3, 0.5, 0.75, 1.0}; depth {None, 12, 24} |
| XGBoost | `hist`; depth {2, 3, 4, 5, 6}; child weight {1, 3, 10, 30}; learning rate {0.015, 0.03, 0.05, 0.07, 0.1}; trees {300, 600, 1000, 1600}; L1 {0, 0.1, 1, 10}; L2 {1, 3, 10, 30, 100}; gamma {0, 0.1, 0.5, 1}; row and column fractions {0.65, 0.8, 1} |
| HistGB | leaves {7, 15, 31}; minimum leaf {10, 20, 40, 80}; learning rate {0.02, 0.05, 0.1}; iterations {150, 300, 600}; L2 {0, 1, 10}; `early_stopping=False` |
| Logistic | `spline` with knots {3, 4, 5, 7, 10}, degree 3, or `log_standard`; C {0.001, 0.01, 0.1, 1, 10, 100}; base F1 for `spline` (section 6.2), F1 or F2 for `log_standard` |
| CatBoost | depth {3, 4, 5, 6, 7}; learning rate {0.015, 0.03, 0.05, 0.07, 0.1}; iterations {400, 800, 1200, 2000}; L2 {1, 3, 10, 30, 100}; random strength {0, 0.1, 0.5, 1, 2}; Plain and Ordered boosting; symmetric trees; numeric F2 |
| EBM | main effects, or main effects with the pairs (year, loudness), (year, acousticness) and (energy, loudness), their column indices resolved after the recipe's feature construction; rounds {300, 700, 1500}; learning rate {0.01, 0.03, 0.06}; `validation_size=0`, `outer_bags=1`, `inner_bags=0` |

- Boosted trees keep fixed iteration counts as hyperparameters, with no early stopping. Around a promising configuration, about {0.67T, T, 1.5T} trees are compared with the rest fixed; an optimum at the edge allows one declared extension within the family's ceiling. Outer-fold learning curves are never inspected to choose iterations.
- Range extension applies only to ordered numeric parameters, when a family's best sits at a range edge: the next batch's saved manifest states the exact new values and the legal hard bounds before fitting; fractions stay in their valid domains; categorical values such as "sqrt" or None have no edge; at most one outward extension per numeric parameter in this stage.
- Local refinement, for at most four families that qualify under section 5.6: two parents (the best standalone configuration and the best contributor to a blend, or the next distinct qualifying one); batches of 12, at most 24 configurations per family, counted within its ceiling. Each proposal multiplies positive scale parameters by a factor drawn from about 0.5 to 2, moves discrete parameters to neighboring values, and changes one to three dimensions; a quarter of each batch are broader draws; canonical duplicates are skipped.
- Optuna is not installed now. If a family remains productive after 64 evaluations, a reviewed Optuna route lock may replace its local refinement: at most 24 TPE configurations, within the same refinement allowance and the family's ceiling, with a frozen objective on complete three-repeat results, space, sampler seed and order, sequential trials, no pruning, and every proposal saved before its fit.
- Factory parameters these spaces need beyond P2.1 are added by the task that owns the family (section 4.10): P2.2 for the installed families (for example XGBoost's `reg_alpha` and `gamma`), P2.3 for CatBoost and EBM (for example EBM's `learning_rate`).

### 5.3 Model settings

- LightGBM keeps the phase 1 determinism settings; no `Dataset` is reused across fits, so each fit learns its own bins.
- CatBoost: `thread_count=4`, an explicit seed, `allow_writing_files=False`, CPU, an explicit boosting type, no automatic class weights, numeric F2. Exact-date strings are not passed to its native categorical statistics, which do not keep repeated tracks together. Low-cardinality musical fields, if categorical, are forced to one-hot.
- HistGB runs under the four-thread OpenMP limit of P2.1.
- Library-internal validation is disabled or given explicit grouped inner splits (strategy section 4, step 5).

### 5.4 The anchor

The anchor is the phase 1 recipe exactly as submitted: LightGBM on F2, the phase 1 parameters, estimator seed 0, the equal average of its 15 fold models (`production_lightgbm_F2_9c37e77d549e`; through the new engine, Finding 009). It stays the project reference. A seed-averaged version of it is a separate candidate.

### 5.5 Neural challenger (P2.5)

- TabM only. Every distinct TabM configuration, in P2.5 or P2.4, counts toward one ceiling of 16 and one compute allowance; neural trials get no share of P2.4's re-tuning allowance. Initial target 8 configurations; further batches of 4 only under section 5.6. Compute reserved: 3 hours for fitting and confirmation, extendable to 6 hours if productive and if Sunday's evaluation still fits.
- Before each batch, its screens, the completions they may require and the final estimator-seed evaluation are reserved. If the timing pilot shows that 8 screens with their completions exceed the allowance, the batch is smaller or a smaller declared shortlist is completed, decided from timing before any score is seen; an omission is recorded before the batch runs. A promising partial-seed result is not a finalist.
- Space: two or three blocks; width {64, 128, 256}; 16 members; dropout {0, 0.1, 0.2}; AdamW learning rate from 3e-4 to 3e-3; weight decay from 1e-6 to 1e-2; batch size {128, 256}. A numerical-embedding alternative only if its training-fold construction is audited.
- Training: member losses trained separately, member probabilities averaged at inference. At most 300 epochs, patience 20, inner validation log loss as the stopping metric, the earliest best epoch within a fixed tolerance. Epoch selection uses an inner group-safe split of the outer training part, with imputation, scaling and numerical bins fitted on the inner training rows only; then the stopping-stage model and preprocessing are discarded, preprocessing is refitted on the whole outer training part, and the network is reinitialized and trained for the chosen count. CPU, no compilation, no mixed precision, deterministic seeds, no data-loader workers. Its reviewed configuration fixes the inner split seed, the optimizer and the tie rule before it runs.
- Screened on split seed 155; the completion rule of section 5.1 applies. A five-fit screen of q minutes makes 8 screens plus 4 completions about 16q minutes, 4 screens plus 4 completions about 12q.
- Estimator seeds: seed 0 alone, or seeds 0, 1 and 2, declared from its measured cost; never the best single seed. A neural candidate stays alive through its blend contribution even when its standalone AUC trails. TabM enters a learned combiner only if its measured complete evaluation and replay fit the reserved budget, decided before the affected procedure scores; otherwise it contributes through fixed probability blends.

### 5.6 Allocation rule for exploration

This governs only which search continues; final selection and deployment keep their own rules (sections 8 and 11). It replaces v0.1's rule that a gain below 0.003 authorizes no further search. Every comparison uses unrounded mean fold AUC on identical folds, with a positive difference on every repeat, and only complete three-repeat results.

State recorded in each batch's manifest before it runs: the family anchor (for its first batch), the family incumbent, the strongest standalone recipe across families, and the reference predictor H, each with its run id and seed policy. For each configuration the search logs its standalone result and two fixed blends with H, 0.9 H with 0.1 challenger and 0.75 H with 0.25 challenger.

A batch makes progress when one of its newly evaluated configurations meets any of these routes:

| Route | Rule |
|---|---|
| Clear standalone progress | at least +0.003 against the pre-batch family incumbent |
| Smaller useful progress | at least +0.001 against the pre-batch family incumbent, and within 0.005 of the strongest standalone recipe |
| Competitive refinement | at least +0.0005 against the pre-batch family incumbent, and within 0.002 of the strongest standalone recipe |
| Ensemble contribution | a declared blend with H improves H by at least 0.0005, and improves the family's previous best blend with the same H by at least 0.0005 |

The family's allowance works as a counter. The initial batch, if it makes progress against the family anchor, grants an allowance of two expansion batches; otherwise the family stops. Each later batch that makes progress resets the allowance to two; a batch without progress uses one. The family stops when the allowance reaches zero, at its ceiling, or at the search cutoff of the task table; useful candidates it already produced stay in the bank. H stays fixed within a family's expansion sequence; if a later reviewed stage changes H, earlier contributions are recomputed against the new H before further progress is credited.

These thresholds are allocation heuristics, not significance tests. The paired bootstrap is run for the comparisons that drive a decision, reported with its conditional caveat, and not required to exclude zero before a batch.

### 5.7 Selection optimism

Hundreds of trials and their blend checks increase selection pressure. Three repeated splits reduce sensitivity to one partition; they are not three independent datasets, and the full three-repeat result of a configuration chosen on those repeats is a complete matched evaluation, not an independent confirmation. No untouched holdout is created now: every label has already taken part in exploration, and a new split seed would be a partition-stability diagnostic, not a remedy. Instead: every trial, feature comparison, blend check and allocation decision is logged; the candidate-construction rule is frozen before its evaluations; Sunday's menu and procedures are frozen before outer results; exploratory best scores are reported apart from the restricted nested results; bootstrap intervals are described as conditional on fixed predictions; the three repeat-level differences are kept for every decision-driving comparison; no search reopens after Sunday.

## 6. Features (P2.4)

### 6.1 Protocol

The feature stage starts with 32 comparisons, expandable by 16 at a time to 96 under section 5.6, followed by re-tuning. Each batch's recipes, comparators and allocation are saved before its scores. Tree-oriented blocks are compared first against the unchanged anchor; smooth, distance and neural preprocessing within its own family; a confirmed effect compares otherwise identical recipes. Each block keeps its own name; no single F3 holds every idea. Order of allocation after the nine anchor checks of section 6.3 and the four reserved checks of section 6.5:

1. Date count and date target encoding on the strongest tuned models, smoothing {3, 10, 30, 100} in batches, not a full block-by-family grid.
2. Combinations of individually useful blocks.
3. Preprocessing alternatives for the smooth and kNN models, and selected F2 or cyclic calendar comparisons for the smooth models (Finding 008 concerns ordinary logistic regression only); neural preprocessing alternatives only after P2.5 converges, counted toward TabM's ceiling (section 5.5).
4. The removal checks of section 6.6.
5. Interactions, cyclic calendar and era-relative features on the tree models.
6. Training-weight sensitivity on promising recipes.

Re-tuning: up to four model-and-feature combinations, chosen by section 5.6, are re-tuned in two batches of 16 each, at most 128 configurations, separate from the base-feature ceilings. At 10 to 20 s per run, 96 comparisons and 128 re-tuning configurations cost about 40 to 75 minutes.

### 6.2 Smooth, kNN and neural preprocessing

- Spline logistic regression: cubic splines on year and the ten continuous audio features (speechiness, danceability, duration_ms, energy, instrumentalness, valence, tempo, loudness, acousticness, liveness), after the phase 1 log transforms; one-hot key, mode and time signature, never splined; median imputation before the splines, scaling after; all fitted in the fitting population.
- kNN (built in P2.2), on F1, fitted inside every training population: the phase 1 log transforms; the ten continuous audio features median-imputed, standardized and divided by the square root of 10; the release year standardized and multiplied by alpha; key, mode and time signature one-hot and multiplied by 0.5 divided by the square root of 3, not standardized. The initial 48 configurations are the full grid k {15, 30, 60, 120, 240, 480} by weights {uniform, distance} by alpha {0.25, 0.5, 1, 2}, Euclidean distance. Refinement may vary the category weight, robust scaling, Manhattan distance with block normalization, and a cyclic calendar block. kNN is deterministic, so it is never fitted with several estimator seeds, and it takes no training weights.
- TabM: the same continuous columns, the F2 date parts with missing indicators, and the one-hot categories.

### 6.3 Blocks

| Block | Content |
|---|---|
| `date_count` | the number of distinct fitting groups per raw date string: a fitting row gets log1p(count[date] - 1), its own group left out; any other row gets log1p(count.get(date, 0)), zero for an unseen date |
| `date_te10`, `date_te30`, and the smoothing values of section 6.1 | the cross-fitted exact-date target encoding of section 6.4; each with and without the count |
| `interactions` | (year - 2000) x loudness; (year - 2000) x acousticness; energy x loudness; energy x danceability; acousticness x instrumentalness; tempo x duration_ms / 60000 |
| `cyclic` | for each phase p, sin(2πp) and cos(2πp), with p the month phase (month - 1) / 12, the weekday phase weekday / 7 and the day-of-year phase (day_of_year - 1) / days in that year; both missing where the component is unobserved |
| `era_relative` | loudness and acousticness minus their era mean over the five eras of `songpop.splits`, each era mean shrunk toward the fitting-population mean with 20 pseudo-observations; an era absent from the fitting population uses the fitting mean |
| Training weights | inverse-group-size loss weights, recomputed and normalized to mean one in every actual fitting population; evaluation stays row-weighted; target-encoder counts stay row-weighted |

The nine anchor checks, each against the unchanged anchor: `date_count`; `date_te10`; `date_te30`; `date_te10` with `date_count`; `date_te30` with `date_count`; `interactions`; `cyclic`; `era_relative`; inverse-group-size training weights. These definitions are proposed defaults, not established optimal choices.

### 6.4 Exact-date target encoding

As in v0.1 and P2.1, unchanged: five inner `StratifiedGroupKFold` folds of each fitting population, built from its rows, labels and primary groups; their explicit (train, test) pairs passed as `cv` to scikit-learn 1.9.1's `TargetEncoder` with `target_type="binary"` and the fixed smoothing m, giving (sum of labels + m x prior) / (rows + m) from the permitted fitting partition; `fit_transform` encodes the fitting rows and the fitted encoder's `transform` encodes every other row; raw date strings are the categories; an unseen date gets the fitting prior. A global out-of-fold encoding is never reused inside another validation level. The feature is an exact-date association, not album identity. The exact-date coverage of 69.87%, 68.65% and 69.35% by repeat against 72.2% for the test rows (round E, unreviewed) motivates the feature and proves nothing; P2.4 recomputes coverage and distinct-group support under the actual fitting dictionaries. The primary folds are not regrouped by date.

### 6.5 Cross-family checks

Declared now, before any of these scores exists, and run whatever the LightGBM screening of these blocks shows: on the random-forest configuration with the highest mean fold AUC in P2.2, add `date_count`, and separately `date_te30` with `date_count`; the same two on the CatBoost configuration with the highest in P2.3. If P2.3 cannot install CatBoost, XGBoost's best configuration takes its place, recorded before the comparisons run. Comparators are chosen among complete three-repeat runs by unrounded mean fold AUC, ties to the lower recipe id. Each comparison is an otherwise identical recipe with and without the block. These four are a floor inside the 32 initial comparisons; further cross-family comparisons come only through section 5.6, and a block that qualifies goes through the usual confirmation before re-tuning.

### 6.6 Removal checks

Named recipes using the drop list of section 4.2, on the strongest two recipes: without `days`; without `day` and `day_of_year` together; without `weekday`; without key, mode and time signature; without a newly added block from a successful combination. Audio features are not removed because their marginal AUC is near 0.5. A drop list is a recipe hyperparameter whose choice is adaptive and is reported as such. No data-driven selector or importance ranking is used; any learned transformation stays inside its training population.

## 7. Ensembles and deployment (P2.6a, P2.6b)

### 7.1 Pool and declared procedures

Before the corresponding P2.6b procedure scores are computed, P2.6b saves a deterministic rule and an enumerated manifest. The rule:

1. One common reference H*: the strongest eligible standalone recipe, by unrounded mean fold AUC, ties to the lower recipe id. Every contribution check is recomputed against H* on identical folds and declared seed policies; contributions measured against earlier batch references are not compared.
2. For each family, two records: its standalone winner and its contribution winner against H*, which may be different recipes.
3. The pool: the three strongest family representatives by standalone mean fold AUC; up to two further representatives chosen by contribution to H*; the anchor; and up to two further configurations of the leading family within 0.002 of its leader, by a fixed ranking, for a within-family average.
4. At most eight atomic recipes in all, counting every transitive ingredient of every procedure, H* included. An atomic recipe is one exact feature, preprocessing and model configuration; estimator seeds are not separate atomic recipes. A missing optional family is skipped; a recipe chosen twice counts once.

The enumerated manifest lists at most 32 procedures, each with its id, ingredients and weights or settings:

| Kind | Procedures |
|---|---|
| Fixed probability blends | equal averages of the two and of the three strongest channels; the equal average of every channel; the 0.9 and 0.75 contribution blends of each challenger with H*, the smooth models, the cheap challengers and TabM included; the equal within-family average |
| Constrained weights | three settings, lambda in {0, 0.01, 0.1} (section 7.2) |
| One regularized stacker | three settings, C in {0.01, 0.1, 1} (section 7.2) |

Channels of the learned procedures: one per selected family representative; where the within-family average exists, it replaces that family's single-recipe channel, and its constituents are not separate channels; the anchor is a standalone candidate, not an automatic channel; TabM only under section 5.5. The ordered channel list and every procedure id are frozen before those procedures are evaluated.

Seed policy: the anchor keeps estimator seed 0, and its three-seed version is a separate candidate; other stochastic classical finalists use estimator seeds 0, 1 and 2, averaged within each exact base recipe before any blend; deterministic estimators such as kNN are fitted once; TabM as in section 5.5. The same seed policy holds in production CV, the nested evaluation and deployment. A challenger is never discarded only because its standalone AUC is below the leader's; its blended AUC on identical folds is the criterion, and prediction correlation is context only.

### 7.2 Learned procedures

A learned procedure is fitted only on its permitted training rows D:

1. Take D's saved combiner-training folds: five grouped folds that depend only on D's population and the protocol coordinates, never on the candidate or its setting, so every learned candidate on the same D shares them.
2. For every channel, rebuild every feature, preprocessing step and encoding inside each inner fit, giving exactly one inner out-of-fold probability per row of D, estimator seeds averaged first. The three repeats are never stacked as independent observations.
3. Fit the combiner on that meta-feature matrix and the labels of D.
   - Constrained weights: nonnegative, summing to one, minimizing the row-weighted mean binary log loss plus lambda times the squared distance to equal weights; the loss clips probabilities at a threshold fixed in the manifest, while the prediction stays the declared convex combination; SciPy with a fixed initialization (equal weights), tolerances and failure handling, all in the manifest; convergence, constraints and finite predictions checked; zero weights kept.
   - Stacker: base probabilities clipped to [1e-6, 1 - 1e-6], turned into logits, scaled on the meta-training matrix only, then L2 logistic regression with intercept and the declared C; no original features, no class balancing.
4. Refit every channel on all of D; new rows are predicted by the refitted channels, then the fitted combiner.

Each learned procedure is evaluated on the official 15 outer folds by running these steps on each outer training part, so the whole procedure, not the combiner alone, is scored on held-out groups; deployment averages the 15 fitted procedures' test predictions. A combiner is never fitted on the global bank and scored on that same bank. Refitting the channels on all of D after building the meta-features makes a training-size mismatch, accepted because the whole procedure is evaluated outside it. A learned procedure has its own artifact schema: channels, atomic recipes, averaging order, meta-fold assignments, objective and regularization, clipping, fitted weights or coefficients, preprocessing statistics, receipts.

Production evaluation cost, with fits shared across settings: about C_all + 6 C_L fit-equivalents before replay and overhead (C_all and C_L as in section 8). P2.6b reserves it, measured, before its 14:00 review request.

### 7.3 Deployment and the menu

- Deployment default: the equal average of the fold models, or of the 15 fitted procedures. A full refit on all eligible rows is a separate deployment artifact in `artifacts/phase2/deploy/`, compared only in section 8.
- At most 12 predictor candidates are frozen for Sunday: the anchor, the selected standalone leaders, the within-family average, the best fixed blend, the best constrained-weight procedure and the best stacker, the rest filled by the selection rule of section 11. Every candidate and every member has the complete three-repeat evaluation.
- Every timing-based change to a candidate's ingredients happens in P2.6b, before its review and the Saturday freeze; a changed procedure gets a new id and its complete production evaluation before admission. After the freeze no candidate changes for runtime reasons.

## 8. Restricted nested evaluation (P2.7)

Levels: outer, three `StratifiedGroupKFold` folds on the eligible rows and primary groups, split seed 20261101, saved in `artifacts/phase2/splits/`; inner, five grouped folds with split seeds 155, 2026 and 1029, rebuilt within each outer training part; for a learned candidate, a third level, its combiner-training folds, rebuilt inside each inner training part. Target encoding and neural stopping stay local to their base fits. The global phase 2 banks are never reused inside the evaluation.

In each outer fold:

1. Build every candidate's inner validation predictions.
2. For each learned candidate, rebuild its meta-training predictions inside each inner training part.
3. `select_candidate(inner_banks, menu)` applies section 11 steps 1 to 3; it receives no outer predictions or labels, no public score and no global bank.
4. Predict the outer validation rows with the selected candidate's inner-fold average.
5. Separately refit the complete selected procedure on the whole outer training part; for a learned candidate, regenerate its meta-training predictions and refit its combiner, never reusing global weights.
6. Produce the anchor's fold-average prediction.

After all three outer folds, a separate decision function applies, in this order: the fallback (if the selected procedure loses at least 0.003 mean outer AUC to the anchor and loses on all three outer folds, the anchor is recommended with its original deployment); then, only if the fallback did not trigger, the deployment rule (fold averaging stays unless the full refit improves the paired outer mean by at least 0.003 with a positive difference on all three outer folds). A refit advantage of a selected candidate is never transferred to the anchor. Once the outer results have chosen either, they are no longer an untouched estimate of what they chose.

Cost and cache:

- Base predictions are cached across combiner settings. A cache entry is keyed by the data hashes, the fitting row ids and labels, the grouping, the ordered prediction populations, the split identity, every fitting-context seed, the exact recipe, the estimator seeds, the source manifest and the dependencies. An entry from another fitting population is refused.
- With C_all the summed 15-fit time of every atomic recipe in the frozen menu at its final seed policy, and C_L the same over the union of atomic recipes and meta-split protocols inside learned combiners, the evaluation needs about 3.2 C_all + 16.2 C_L fit-equivalents (round I, an arithmetic projection, not a runtime bound: about 113 to 145 minutes for C_all of 600 to 1,200 s and C_L of 300 s, 4.7 to 6.0 hours with a full replay and a 25% margin). A pilot in P2.6a measures it.
- Eight hours are reserved for Sunday's computation and verification, starting Saturday night after P2.6b converges. A full replay starts with an empty cache; reusing the original run's cache is not a recomputation.

Reduced path, declared now: if P2.7's measured projection of build plus replay exceeds the eight-hour reservation, the fixed-candidate check runs instead (the candidate ordinary CV selects and the anchor, each fully refitted in the three outer folds, a stack rebuilding its own meta-training predictions), labeled as such, unable to trigger the fallback or the deployment rule. The decision is made from the projection, and the evaluator prints no outer score before it is recorded.

Reported: the three outer fold AUCs, their mean, the pooled outer out-of-fold AUC, the selected candidates, and the paired differences against the anchor; a group bootstrap of the outer predictions only with its conditional caveat. The innermost fits of a learned combiner use substantially fewer rows than production CV, so part of any decline comes from smaller training sets and thinner date support. This evaluates a restricted selection procedure whose menu was informed by earlier exploration; it does not remove that adaptation.

## 9. Kept for phase 4

No report text, Colab code, SHAP analysis, importance ranking or explanatory plot in phase 2. Phase 2 keeps: predictors that accept raw source columns; feature lineage to the original columns; a reproducible way to rebuild or load every selected fold predictor, its encoders and preprocessing; training and evaluation ids and hashes; the complete search and selection records; the consultation evidence for the LLM Usage section. Permuting the date later must apply the already fitted date map, never learn a new one from evaluation labels.

## 10. Submissions

Each is a gated action, reviewed with its file hash and purpose inside its task; its purpose is written down before Sam reports the score. The public score may prompt a check for an implementation or distribution mismatch; it never drives a weight grid, a feature search or a swap between nearly tied models. The public 0.83597 of P1.3 is a pipeline observation, not a search objective.

| When | File | Purpose |
|---|---|---|
| After P2.2 or P2.3, optional, if a complete three-repeat single recipe's standalone mean fold AUC exceeds the anchor's (its repeat differences reported) | That recipe | Exercise the phase 2 fitting and submission path; a contribution-only qualification supports submitting the evaluated blend, not a standalone claim |
| After P2.6b | The frozen blend or learned procedure the selection rule ranks first | Exercise ensemble or learned-procedure deployment |
| Only if warranted | A different competitive frozen candidate, or a deployment comparison already justified locally | Stated when proposed |

No submission is made only to fill the schedule; a file already uploaded with identical bytes is not uploaded again. With phase 3's final candidates this keeps the project near the strategy's four to six uploads.

## 11. Freeze, menu and selection rule

- Complexity ranks are fixed before the corresponding P2.6b evaluations. By Saturday 18:00 PDT, P2.6b freezes the exact candidate recipes and procedures (parameters, feature definitions, estimator seeds, blend weights, combiner objectives and regularization, meta-fold policy), each candidate's fixed `complexity_rank`, the selection rule and the nested protocol, in `configs/phase2/menu.json`. P2.7 starts only after P2.6b converges. The small finalist menu is regenerated under one final implementation revision before the freeze.
- Sunday permits evaluation, verification and reviewed correctness fixes, and no score-driven revision of the frozen candidates. A correctness fix invalidates the affected predictions, which are regenerated before use. By 18:00 PST, P2.7 has converged and the deployment manifests are frozen.
- Monday: reproduce, validate, prepare and upload the candidates, select (strategy section 6, P3.1 and P3.2).

Selection rule, declared before the menu is evaluated; `select_candidate` implements steps 1 to 3:

1. Rank candidates by unrounded mean fold AUC over the 15 folds.
2. Candidates within 0.001 of the maximum are practically tied (a concrete form of the strategy's "ties go to the simpler recipe").
3. Break ties lexicographically: fewer unique atomic recipes; then fewer supervised feature transformations; then fewer learned combiners; then the lower fixed `complexity_rank` stored in the menu; then candidate id. Measured wall-clock time never breaks a tie.
4. The fallback and deployment decisions of section 8, after the nested evaluation.

Phase 3 proposes A, the strongest stable candidate under this rule, and B, a competitive alternative with evidenced complementary errors, normally within 0.005 mean fold AUC of A; if no diverse candidate qualifies, the closest strong alternative. A learned procedure may be either, under the same rule. Both bounds are engineering bounds, not uncertainty calculations. Sam chooses.

Phase 2 exits when the frozen candidates have complete matched banks, every learned procedure has been evaluated as a whole on held-out groups, the deployment artifacts verify, the findings are reviewed and every run's source revision is known. A large trial count is not an exit criterion.

## 12. Strategy amendments

P2.0 was the reviewed task for this coherent amendment of `modeling_strategy.md`; each row is edited there, with a history note. P2.0 converged and was committed (3e24fe6). This table is historical: its numerical limits (104 configurations, 12 refinement trials, the split-seed rules) are no longer executable rules; section 14 and the rewritten sections 5 to 8 replace them.

| Strategy | Amendment or clarification | Why |
|---|---|---|
| Section 6: P2.1 compares F1 and F2 on Friday; the Day 1 slip paragraph | P2.1 is infrastructure; phase 2 starts Thursday; the slip paragraph is replaced by this plan's cut order | Finding 008 already settled F1 against F2; phase 1 finished a day early |
| Section 5.4: random search on split seed 155, then Optuna TPE; the search regions | 104 classical configurations on all three split seeds when cheap, in the spaces of section 5.1; Optuna only for a reviewed refinement | A 15-fit baseline run costs 6 to 8 s; one partition invites partition-specific choices |
| Section 5.4: iteration caps of 2,000 to 4,000 with inner early stopping | Fixed iteration counts as hyperparameters for boosted trees; inner early stopping only for the network | Cheap at this size, and it removes a whole inner split from the code and the review |
| Section 5.2: frequency encodings priority D | The exact-date group count becomes priority B and is scheduled in P2.4; other frequency encodings stay D | Cheap, auditable, and it complements the date target encoding |
| Section 6: a package with incompatible requirements gets an isolated environment or is dropped | In phase 2 it is dropped | One environment (Decisions) |
| Section 4, step 9: ties go to the simpler recipe | Clarified: the 0.001 tie band and the tie-breaks of section 11 | A deterministic rule, declared before the menu is evaluated |
| Section 4, step 10: the restricted nested evaluation | Clarified: the design, the anchor fallback, the deployment comparison and the reduced path of section 8 | Rules declared before the evaluation runs |
| Sections 5.6 and 6: interpretation on Sunday; section 4, step 11: stress tests for the report | Both in phase 4 | Sam's decision of 2026-10-29 |

## 13. Amendments in P2.1

Infrastructure only, reviewed in P2.1 before P2.2 builds on it; P2.0's approval does not cover these lines.

- Section 4.1: `runs.py`, `searches.py`, `submission.py` and `install_locks.py` join the layout. `selection.py`, `nested.py`, the full-refit artifacts, the TabM wrapper and their self-checks are built in P2.6, P2.7 and P2.5, the tasks that first use them; `check_phase2.py` reports their checks as STAGED until then.
- Section 4.2: preprocessing is a recipe field of its own, as this section always listed, so spline logistic regression is the logistic family with `spline` preprocessing (section 5.1); a drop list supports feature-removal checks; the recipe hash leaves out the human name.
- Sections 4.3 to 4.7: fit-time checks of every estimator seed and fit receipts; the linked-field checks of bank_v2; envelope-first verification of sources and dependencies; seed-restricted blends and comparisons; the selection-manifest schema and the submission record; submission checks on rebuilt model inputs after preprocessing and against the production protocol.
- Section 4.9: joint resolution, exact wheel URLs, saved before- and after-inventories.

This section records P2.1's staging as approved; where section 14 reallocates the work (section 4.10), the later allocation applies.

## 14. Amendment P2.0b

Reviewed in P2.0b before any later task relies on it; the approvals of P0.2, P2.0 and P2.1 do not cover it. It carries two messages Sam sent during P2.1. The first asked for bounded refinements, verbatim in part: "I want the following bounded refinements incorporated into phase 2 before the P2.6 candidate freeze." The second, verbatim in part: "pursue a thorough, evidence-driven search for the strongest final result." and "Use measured runtimes to set sensible budgets; do not assume computation or agent effort is the main bottleneck." and "This supersedes my earlier instruction to preserve the existing experiment budgets. It also supersedes the requirement that every added experiment must replace another solely to keep those budgets unchanged. Existing useful choices do not need to be reversed." Both messages are quoted in full in the P2.0b review record. The second does not supersede the freezes, grouped validation, fold-fitted transformations, the installation rules, supplied-data-only training or the exclusion of pretrained models.

| Sam's request | Where | What changes from v0.2 |
|---|---|---|
| Broader search and local refinement (second message, item 1) | Sections 5.1, 5.2 | Initial batches, expansion batches and ceilings per family from measured costs; wider spaces; expansion and stopping rules; local random refinement; Optuna only through a reviewed lock if a family stays productive |
| Re-tuning after features (item 2); cross-family date checks (first message, item 2) | Sections 6.1, 6.5, 6.6 | A feature stage of 32 to 96 comparisons, the four reserved cross-family checks as a floor, removal checks through the drop list, preprocessing alternatives, re-tuning of up to four combinations |
| Complementary models (item 3); kNN (first message, item 3) | Sections 5.1, 5.5, 6.2 | XGBoost, Extra Trees, HistGB, the smooth models and kNN get real allocations; kNN is added beside HistGB rather than replacing it; TabM's cap is set from its measured cost |
| Thorough ensembling (item 4); contribution blends (first message, item 1) | Sections 7, 8 | A deterministic pool; fixed blends including contribution blends with H; constrained weights; one L2 stacker; learned procedures fitted on rebuilt inner out-of-fold predictions and evaluated as wholes; a third level in the nested evaluation |
| Allocation rule (item 5) | Section 5.6 | Four routes to further search, distinct from the final selection and deployment rules |
| Schedule and freezes kept | Tasks; sections 4.10, 8, 11 | P2.6 split into P2.6a (engine, selector and evaluator) and P2.6b (evaluation and menu); P2.3's installation may start before P2.2 converges; schedule guards from Saturday 08:00; the Saturday and Sunday freezes unchanged; eight hours reserved for Sunday |
| Exploration kept apart from publication | Section 10 | An optional pipeline upload needs a standalone gain over the anchor; a contribution-only qualification supports only its blend |

Superseded by this amendment: v0.1's caps of 104 classical and two neural configurations, its 12-trial refinement for at most two families, its rule that smaller gains authorize no further search, its at most five declared blends and eight predictor candidates, and section 12's rows on those budgets. The runtimes become Finding 010 with this review; the consultation probe numbers (exact-date coverage, round I's HistGB timing and nested projections) stay unreviewed inputs, and the probe AUCs printed by the timing script select nothing.

## Open questions

1. The exact Kaggle team name as Kaggle displays it, for the report and the Piazza note (Sam).
2. The assignment asks for groups of two to four (`../../Project_1.pdf` section 2); the course-group arrangement is not recorded (Sam).

## Status

- 2026-10-29: Sam's go-ahead for planning phase 2 (header). Codex consultation round E (an independent design, after Claude's positions were written down) and round F (a critique of the first draft; its changes adopted, listed in the consultation README). P2.0 in review.
- 2026-10-29: P2.0 approved in two rounds and committed (3e24fe6, df11816). P2.1 started on Sam's go-ahead, verbatim: "Commit and start P2.1". During P2.1 Sam asked for refinements to the experiments, budgets and ensembles of phase 2; they are a separate reviewed amendment, P2.0b. P2.1 in review.
- 2026-10-29: P2.1 approved in two rounds and committed (719a934, 4e451ec). P2.0b, the amendment of section 14, built after Codex consultation rounds I (an independent design, after Claude's positions were written down) and J (a critique of the draft); in review.
