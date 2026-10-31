# Phase 2: models and features

v0.8, 2026-10-30. Owner: Sam Zhou. Status: v0.1 drafted in task P2.0 on Sam's go-ahead, verbatim: "Yes, plan phase 2" and "Again, consult deeply with codex"; approved after two review rounds and committed (3e24fe6, df11816; `../../reviews/2026-10-29-p2.0-phase2-plan.md`). v0.2: the infrastructure adjustments of P2.1 (section 13), approved after two review rounds and committed (719a934, 4e451ec; `../../reviews/2026-10-29-p2.1-infrastructure.md`). v0.3: the amendment P2.0b (section 14) on Sam's refinements to the experiments, budgets and ensembles, approved in one review round and committed (ee557e2, 281a014; `../../reviews/2026-10-29-p2.0b-search-amendment.md`); sections 5 to 8, 10 and 11 and the task table are rewritten by it. v0.4: the P2.2 execution protocol (section 15), approved at P2.2's execution-readiness checkpoint after two review rounds; P2.2 executed and at its completion review (`../../reviews/2026-10-29-p2.2-searches.md`). v0.5: section 15.7, the verification of the tie-sensitive forest runs, added at P2.2's completion checkpoint on Sam's decision; P2.2 converged and committed (810d1ce, 72a822b). v0.6: section 16, task P2.8 (the gap hunt), with the task table, section 6.3's `music` block and section 6.5's reserved checks; approved after readiness and completion reviews and committed (5e2d5a0, cab0174; `../../reviews/2026-10-30-p2.8-gap-hunt.md`). v0.7: section 17, the P2.3 execution protocol (CatBoost and EBM), with the task table and section 4.10; approved at its readiness and completion checkpoints (`../../reviews/2026-10-30-p2.3-catboost-ebm.md`), not yet committed. v0.8: section 18, the P2.4 execution protocol (the feature stage), with the task table and section 4.10; in review (`../../reviews/2026-10-30-p2.4-features.md`). Built after Codex consultation rounds E (an independent design) and F (a critique of the first draft), verbatim in `../llm_usage/2026-10-29-phase2-consultation/` with what was adopted.

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
| P2.2 | Search extensions (factory parameters, conditional sampling, local refinement batches, timing and allocation records); the kNN family and its representation; the broad searches and first refinements of LightGBM, random forest, XGBoost, Extra Trees, HistGB, the smooth logistic models and kNN (section 5); findings; executed under section 15 | P2.1, P2.0b | Thursday evening to Friday morning | `../../reviews/2026-10-29-p2.2-searches.md` |
| P2.8 | The gap hunt of section 16: identifier and order probes, the feature checks moved from P2.4 with the `music` block, deterministic forest prediction; findings | P2.2 | Friday: readiness by about 10:00 PDT, completion by about 14:00 | `../../reviews/2026-10-30-p2.8-gap-hunt.md` |
| P2.3 | The CatBoost and EBM factory changes, installations and runtime checks; timing pilots; their searches and refinements, including section 16.3's declared CatBoost configuration; findings; executed under section 17. Installation and timing may start before P2.2 converges; the searches start from P2.8's approved completion revision | P2.1, P2.0b; P2.2 for shared search extensions; P2.8 for the searches | Friday: readiness by about 08:00 PDT, last batch start 12:00, completion by about 14:00 | `../../reviews/2026-10-30-p2.3-catboost-ebm.md` |
| P2.4 | The feature stage of section 6: screening, the cross-family checks, preprocessing alternatives, removal checks, then re-tuning of up to four model-and-feature combinations; findings; executed under section 18. It imports P2.8's 14 comparisons as its first (section 16.3). Neural comparisons only after P2.5 converges; otherwise omitted | P2.2, P2.3, P2.8; P2.5 for neural comparisons | Friday through Saturday morning; last batch start Saturday 04:00 PDT | `../../reviews/2026-10-30-p2.4-features.md` |
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
| P2.8 | Deterministic forest prediction; the `music` block; `--verify --task` for tasks of individual recipes; the standalone identifier probe outside the engine (section 16) |
| P2.3 | CatBoost and EBM factory parameters, `bootstrap_type`, `bagging_temperature` and `rsm` included (section 16.3); their runtime checks; CatBoost prediction with four threads; a protocol file per task, a reference family without a space, the optional `prior_strongest` member of G, local proposals that keep a parent's own off-list values (section 17) |
| P2.4 | Further smoothing values (`date_te3`, `date_te100`) and preprocessing alternatives (`spline_calendar`, `knn_blocks_v2`); removal recipes; the feature-stage controller (`feature_stage.py` and the feature commands: frozen reference inventory, import receipt, batch manifests, judgments, witnesses, recorded stop, re-tuning units, audit) (section 18) |
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

That is 336 initial classical configurations, 276 in P2.2 and 60 in P2.3, and at most 794 with every expansion. Measured costs for one 15-fit run, all three split seeds, four model threads, one process on the Apple M5 (`../../analysis/phase2_timing.py` at the committed P2.1 code, output `../../analysis/phase2_timing_output.txt`; part of this amendment's review): LightGBM 2.2 to 27.0 s (7.5 s for the anchor; 10.4 s with the date count and target encoding; 23.4 s with three estimator seeds); random forest 8.2 to 26.9 s; Extra Trees 10.7 s; XGBoost with 1,000 trees 6.8 s; HistGB 4.6 s under the four-thread limit; logistic regression 0.4 to 0.6 s; a kNN proxy 0.5 s. A run counts 15 outer fit positions; with three estimator seeds it makes 45 estimator fits. These are representative cases (Finding 010, reviewed in P2.0b): the 30 to 60 minutes of serial fitting estimated for the initial batches with known costs extrapolates from them and is not an upper bound for the enlarged spaces. Each task times its actual preprocessing and its declared expensive configuration before fixing a batch's reservation; CatBoost and EBM are timed in P2.3 before their batches are sized.

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
| `music` | added in P2.8 (section 16.3): the sine and cosine of 2π((7 key) mod 12)/12 and of 2π log2(tempo); raw key and tempo stay |
| Training weights | inverse-group-size loss weights, recomputed and normalized to mean one in every actual fitting population; evaluation stays row-weighted; target-encoder counts stay row-weighted |

The nine anchor checks, each against the unchanged anchor: `date_count`; `date_te10`; `date_te30`; `date_te10` with `date_count`; `date_te30` with `date_count`; `interactions`; `cyclic`; `era_relative`; inverse-group-size training weights. These definitions are proposed defaults, not established optimal choices. P2.8 runs them, a tenth with `music`, and four tuned-recipe checks (section 16.3).

### 6.4 Exact-date target encoding

As in v0.1 and P2.1, unchanged: five inner `StratifiedGroupKFold` folds of each fitting population, built from its rows, labels and primary groups; their explicit (train, test) pairs passed as `cv` to scikit-learn 1.9.1's `TargetEncoder` with `target_type="binary"` and the fixed smoothing m, giving (sum of labels + m x prior) / (rows + m) from the permitted fitting partition; `fit_transform` encodes the fitting rows and the fitted encoder's `transform` encodes every other row; raw date strings are the categories; an unseen date gets the fitting prior. A global out-of-fold encoding is never reused inside another validation level. The feature is an exact-date association, not album identity. The exact-date coverage of 69.87%, 68.65% and 69.35% by repeat against 72.2% for the test rows (round E, unreviewed) motivates the feature and proves nothing; P2.4 recomputes coverage and distinct-group support under the actual fitting dictionaries. The primary folds are not regrouped by date.

### 6.5 Cross-family checks

Declared now, before any of these scores exists, and run whatever the LightGBM screening of these blocks shows: on the random-forest configuration with the highest mean fold AUC in P2.2, add `date_count`, and separately `date_te30` with `date_count`; the same two on the CatBoost configuration with the highest in P2.3. If P2.3 cannot install CatBoost, XGBoost's best configuration takes its place, recorded before the comparisons run. Comparators are chosen among complete three-repeat runs by unrounded mean fold AUC, ties to the lower recipe id. Each comparison is an otherwise identical recipe with and without the block. These four are a floor inside the 32 initial comparisons; further cross-family comparisons come only through section 5.6, and a block that qualifies goes through the usual confirmation before re-tuning. P2.8 runs the two random-forest checks; section 16.3 reserves three more on CatBoost and EBM (`music`, and `interactions` on CatBoost).

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

## 15. P2.2 execution protocol

Declared in P2.2 before any of its search scores exists, and reviewed at its execution-readiness checkpoint (15.1). It resolves what sections 5.1 to 5.6 and 6.2 leave to the task; where it is silent, they apply. The machine-readable counterpart is `../../configs/phase2/p2.2/protocol.json`; the code is `../../songpop/phase2/protocol.py`, `allocation.py`, `refine.py` and `batches.py`, run by `../../scripts/run_experiments.py`.

### 15.1 Scope and review checkpoints

P2.2 searches the installed families on the base features: F2 for the tree families, three logistic branches (`log_standard` on F1 and on F2, `spline` on F1), F1 for kNN. No feature block, training weight, estimator-seed search, removal check or learned ensemble belongs to it. One review record, two checkpoints, each converging on its own fingerprint:

| Checkpoint | The reviewer receives | Its approval permits |
|---|---|---|
| Execution readiness | The code and self-checks, the timing and reservation records (times only), this section and the protocol file, the saved wave 1 manifests, the exact commands | Executing this protocol unchanged |
| Completion | Everything above, the banks, contribution, allocation and bootstrap records, the audit and the full refit, the findings, any proposed submission | Committing P2.2; P2.3's searches |

The first approval covers no result. The source files stay unchanged from the first checkpoint through the waves and the refit; a source edit voids the runs it affects and needs a new readiness round. P2.3 may install and time before P2.2 converges, as the task table allows; its searches wait for P2.2's completion.

### 15.2 Reference, anchors and comparators

- H, for all of P2.2: the LightGBM family anchor's P2.2 run, the phase 1 recipe with estimator seed 0. Its run id differs from the P2.1 anchor's because the source manifest differs; its ten bank arrays must equal those of the P2.1 anchor bank exactly (Finding 009). The initialization checks this and logs an `h_check` record with both bank hashes; no search fit runs without a passed check.
- Family anchors, the first configuration of each initial batch:

| Family | Anchor |
|---|---|
| LightGBM | The phase 1 recipe, exactly |
| Random forest | The phase 1 recipe: 500 trees, `min_samples_leaf` 3, F2 |
| XGBoost, Extra Trees, HistGB | The representative recipes of Finding 010, exactly |
| Logistic | `log_standard` on F2, C 1.0: a new phase 2 reference (phase 1 had no one-hot categories and `max_iter` 2000) |
| kNN | k 60, distance weights, Euclidean, alpha 1.0, category weight 0.5: a point of its grid |

- The family incumbent: the family's complete three-repeat run with the highest unrounded mean fold AUC, ties to the lower canonical recipe hash.
- G, the strongest standalone recipe: for wave 1, G is fixed to H. For later waves, G is the strongest complete run from earlier closed waves, H included. Results within the current wave never update its comparators.
- The family's previous best blend with H: its best contribution blend from earlier waves by unrounded mean fold AUC, with H itself as a floor; for the initial batch, the better of the anchor's two blends, floored at H.
- A gain of at least m means an unrounded mean difference of at least m and a positive difference on every repeat; "within d of G" compares means only.

### 15.3 Waves

All families advance in waves separated by barriers. Every manifest of wave N is a pure function of the protocol file and the records of earlier waves, so the order of families inside a wave changes nothing, and verification re-plans each wave from the log.

| Wave | Batches |
|---|---|
| 0 | The initialization: after the wave 1 manifests are saved, each family's anchor is fitted, then H is checked. No allocation |
| 1 | The initial batches of section 5.1 (276 configurations); the anchors are reused, not refitted |
| 2, 3 | A local refinement batch for each refinement family (15.4); a broad expansion batch for every other family with allowance |
| 4 and later | Broad expansion for every family with allowance |

- A family stops when its allowance reaches zero, at its ceiling (unique configurations proposed, failures included), or at the cutoff: no batch starts, and no anchor is fitted, after Friday 2026-10-30, 05:00 PDT. A `batch_start` record is logged before a batch's first search fit; after the cutoff only batches with such a record continue.
- Local batches use the same counter as broad ones.
- A failed fit logs a `run_failure` record and stops the wave; it is investigated, never skipped.

### 15.4 Batch contents

- Wave 1, sampler seed 2026102900 + 100 x family index + 1 (index 0 to 6 in the order of section 5.1): LightGBM, the anchor and 63 draws; random forest, the anchor and 47; XGBoost, the anchor and 47; Extra Trees, the anchor and 31; HistGB, the anchor and 11; logistic, the six C values on `log_standard` F1 and on F2 (the anchor among them) and twelve `spline` F1 configurations, two knot counts per C: (0.001: 3, 5), (0.01: 4, 7), (0.1: 5, 10), (1: 7, 3), (10: 10, 4), (100: 3, 5); kNN, its full grid of 48.
- Batch b of a family (b = 1 for the initial batch) uses sampler seed 2026102900 + 100 x family index + b.
- Boosting draws and proposals: learning rate times iterations within 5 to 100, anchors exempt; LightGBM's `subsample_freq` is 1 when `subsample` is below 1 and absent otherwise.
- Two configurations are the same when their comparison key is: model parameters a recipe leaves out take the factory default the protocol declares, counts compare as integers and every other number as a float. Fractions (`subsample`, `colsample_bytree`, `max_samples`, `max_features`) must be floats, since an integer 1 means one row or feature to scikit-learn.
- Broad expansion: the declared batch size, or what the ceiling leaves, of new draws from the family's expansion space, never a configuration equivalent to an earlier one. When draws stop finding new configurations, an enumeration must prove the space exhausted (a smaller batch, its reason recorded); otherwise the planning fails. The expansion spaces are the initial spaces, except logistic (the `spline` F1 branch) and kNN (its grid with category weight 0.25, 0.5 or 1.0 and Euclidean or Manhattan distance).
- Range extension, in broad batches: when the incumbent's value of a parameter with an all-numeric list is an end of that list, one value one geometric step beyond, clipped to the hard bounds of the protocol file, rounded for integers; once per parameter in P2.2; never below a zero end; "sqrt" and None have no edge.
- Refinement families, chosen at the barrier after wave 1 among the families with allowance: the two strongest by standalone unrounded mean fold AUC, then up to two more by their best contribution to H; ties by the canonical recipe hash, then the family name.
- A family's best contribution blend, wherever one is chosen (refinement families, the contribution parent, the previous best blend, bootstrap witnesses): the highest unrounded mean fold AUC; ties to the lower canonical recipe hash of its challenger, then the lower weight.

A local batch:

| Part | Rule |
|---|---|
| Size | 12: 3 broad draws and 9 local proposals, fewer only at the ceiling |
| Parents | The standalone parent, the incumbent; the contribution parent, the qualifying configuration (one that met a route of section 5.6) with the best contribution blend with H, other than the incumbent. Without one, the incumbent takes all 9 local slots |
| Local split | 5 standalone and 4 contribution in the first local batch, 4 and 5 in the second |
| Tree variants | For the boosting families, the first two standalone slots: the incumbent with round(2T/3) and round(1.5T) iterations, the rest fixed, when admissible, inside the active bounds and new |
| Moves | One to three parameters per proposal. A scale parameter moves by a log-uniform factor in 0.5 to 2, three significant digits, integers rounded, clipped to the active bounds: a local move never extends a range. Zero moves to its declared step; a move below that step lands on zero. An ordered list moves to a neighbour ("sqrt" and None are members, never edges). A categorical value changes to another |
| Broad draws | Uniform over the family's active space |
| Exhausted neighbourhoods | A local slot that finds no new proposal in 500 draws ends its parent's local proposals; the parent's remaining slots become broad draws. When broad draws find nothing new in 500 draws, an enumeration of the active space must prove it exhausted (a smaller batch, its reason recorded); otherwise the planning fails |

kNN under Manhattan distance: block weights 1/10 for the continuous audio, alpha for the year, and beta/6 for the one-hot categories, so a row differing in one of the three categorical fields adds beta/3. F1 holds no date precision, so rows of different primary groups can be exact F1 matches; trees share this, the primary folds are kept, and it is reported as a limitation.

### 15.5 Records and verification

- Every manifest (`../../artifacts/phase2/batches/`) is written once before its batch's first fit and logged with its hash. Before any fit of a wave, a preflight checks every manifest against its file, id, schema and record, the current protocol hash, source manifest, task and split seeds, the run ids recomputed from its recipes, and a re-planning of the wave from the log; any difference stops the wave with nothing fitted.
- Each configuration's two contribution blends with H are logged as records computed from the two banks and bound to their hashes; no blend bank is saved. H with itself is an identity record, never a blend.
- Each batch's allocation (routes per configuration, progress, the counter) is logged after the bootstraps of its witnesses, so a logged allocation closes a batch whose bootstraps exist; resuming a closed batch finishes any bootstrap record that is missing. The comparisons that drove it are bootstrapped: for each route met, the qualifying configuration with the highest relevant unrounded score (standalone, or its better blend with H), ties to the lower canonical recipe hash; a standalone witness against the pre-batch incumbent, an ensemble witness against H and against the previous best blend. 2,000 replicates, the seed of phase 1; descriptive, never a gate.
- Interruptions: planning resumes from the saved manifests; a finalized bank without its log record is recovered once after its identity, sources and arrays are checked; a logged run is reused only after its bank matches its hash.
- Timing and reservation records hold times, fits, warnings and their provenance, never a score.
- Verification at completion: `--audit-task P2.2` checks, without refitting, every manifest, every task run against the manifests, every bank against its hash and record, every contribution, allocation and bootstrap and the H check, and that every planned batch is closed, or belongs to the last wave and stayed unstarted because the cutoff passed, with no further wave due before the cutoff; `--verify-task P2.2` refits every P2.2 run once and refuses an incomplete task. The P2.1 anchor is verified at its own source revision; `--verify-all` at the P2.2 head reports it as replaying elsewhere, by design.

### 15.6 Commands, compute and storage

From the project root, with `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.run_experiments`:

| Step | Arguments | When |
|---|---|---|
| Timing | `--timing configs/phase2/p2.2/timing.json --task P2.2`, then `--reserve --task P2.2` | Before readiness |
| Wave 1 manifests | `--plan-wave 1 --task P2.2` | Before readiness |
| Initialization | `--run-wave 1 --anchors-only --task P2.2` | After readiness converges |
| Wave 1 | `--run-wave 1 --task P2.2` | After the initialization |
| Later waves | `--plan-wave N --task P2.2`, then `--run-wave N --task P2.2`, for N = 2, 3, ... until no family remains or the cutoff | After each wave |
| Audit and refit | `--audit-task P2.2`, then `--verify-task P2.2` | Before completion |

- Serial: one process, four model threads. Section 5.1 allows at most two processes after a benchmark; P2.2 does not use the second, so the log shards of section 4.10 are unnecessary and not built.
- Compute: the reservation record projects the initial batches and the ceilings from the timing records; the full refit repeats the fitting done; neither is an upper bound.
- The banks, about 440 KB each, at most 644 for P2.2's ceilings, are committed with P2.2.
- Findings 011 (the search results per family) and 012 (contributions and allocation outcomes), labeled exploratory: a full three-repeat evaluation of a configuration chosen on those repeats is a matched evaluation, not an independent confirmation.
- The optional pipeline submission of section 10: the complete single recipe with the highest unrounded mean fold AUC among those exceeding the anchor's, ties by canonical recipe hash, its three repeat differences reported; prepared and reviewed within P2.2, uploaded by Sam.

### 15.7 Amendment: verification of the tie-sensitive forest runs

Added at P2.2's completion checkpoint, after the searches ran, on Sam's decision of 2026-10-30, verbatim: "Do B". Option B was the explicit acceptance below; option A, a deterministic rerun of all 348 configurations under a versioned protocol (review record, response to round 3). The amendment changes no executed step, record, bank, manifest or ranking rule of sections 15.1 to 15.6; the protocol file is unchanged.

| Item | Rule |
|---|---|
| Scope | The 19 random forest and Extra Trees runs that `--verify-task P2.2` did not reproduce exactly (`../../artifacts/phase2/verify_task_p22.txt`). Their banks and log records stay as they are |
| Reason | A forest predicting with four threads sums its trees' probabilities in a varying order, so predictions that are tied in exact arithmetic can come out in either order. Their AUC-valued fields are one realization of a score defined only up to tie-breaking (Finding 013) |
| Criterion | Replaces the exact score comparison of section 4.5 for these 19 runs only. A refit must equal the saved bank in every array and stable metadata field except `fold_auc`, its predictions within 1e-12 (section 4.4). Its rebuilt log record must equal the logged one in every field except the AUC-valued ones: the rounded and unrounded scores, the pooled out-of-fold AUCs and the AUC values of the diagnostics. Every P2.2 decision must be unchanged when the refit's scores and contribution blends replace the logged ones: wave 1 re-judged, each later wave re-planned from the substituted records and judged against its refreshed comparators, wave 4 empty again |
| Check | `../../analysis/p22_tie_check.py`: three realizations of the 19 runs, run ids and refits under the source manifest the P2.2 runs recorded, a negative control that lowers a later-wave comparator |
| Bounds | Differences observed over the realizations are reported as observed, never as bounds. The script also reports, without fitting, the guaranteed range of every P2.2 forest run's fold AUCs over all realizations within the prediction tolerance. Decisions are shown invariant over the sampled realizations only |
| Rankings downstream | Unchanged. Every later use of a P2.2 forest score (the comparator of section 6.5, the pool rule of section 7.1) takes the logged scores with that section's own tie rule, so each ranking is a deterministic function of the log |
| The other forest runs | The other 85 reproduced exactly in the refit and keep the original criterion. The mechanism is the same, and the guaranteed ranges cover them |
| Follow-up | The next task that changes the phase 2 code makes forest prediction deterministic (one thread, or an equivalent fixed summation order) and re-checks the exact reproduction of the other families, before any random forest or Extra Trees configuration enters a frozen menu |

P2.2's verification at completion is therefore the audit of section 15.5, the full refit (329 runs exact) and this check for the 19.

## 16. P2.8: gap hunt

Added on Sam's instructions, verbatim: "Yeah, don't assume anything. Use tests and actual data to draw conclusions." (2026-10-29, answering whether to test the identifier columns instead of excluding them by argument); "Prioritise gap hunt" (2026-10-30, answering the proposal to run the identifier probe and the date-encoding comparisons right after P2.2, ahead of the broad CatBoost and TabM searches); "Yes, start P2.8"; and, on the earlier audio-only project, "Take what you can get out of the old project and then update your plan and your current working, and from then on just don't depend on the old project anymore. I may remove it later." P2.8 tests concrete feature omissions before the broader model work. The public-score difference Sam reported motivates the investigation; it does not establish a comparable validation deficit or identify its cause. Built after Codex consultation rounds M (an independent design of the identifier probe) and N (a critique of Claude's design), and an audit of the earlier project whose carried-over evidence is `../prior_project/`.

### 16.1 Scope, checkpoints and order

| Part | Content | Machinery |
|---|---|---|
| A | Identifier and order probes (16.2) | A standalone runner, `../../analysis/identifier_probe/`, with its own declared menu, outputs and verifier |
| B | Feature checks moved from P2.4, the new `music` block, two fixed-blend diagnostics, and checks reserved for P2.3 and P2.4 (16.3) | The phase 2 engine through `scripts.run_experiments --recipe`, logged as task P2.8; a read-only summary of the matched comparisons |
| C | Deterministic forest prediction, Finding 013's follow-up (16.4) | One helper in `songpop/phase2/models.py`, used by the engine and by part A |

One review record, `../../reviews/2026-10-30-p2.8-gap-hunt.md`, with two checkpoints in the manner of section 15.1. Readiness covers the source, the self-checks, the structural audit, the manifest and recipes saved before any P2.8 score, a score-free timing and the exact commands; its approval permits executing this section unchanged. Completion covers the results, the findings and the verification; its approval permits committing P2.8 and downstream tasks building on it. Execution order: C's check, then B, then A and its conditional controls. P2.3's searches start from P2.8's approved completion revision, because the shared source changes; P2.3's installation and timing may proceed earlier from an approved revision. Saturday's 08:00 PDT cutoff for exploratory batches and the time reserved for P2.6a and P2.6b are unchanged.

### 16.2 Part A: identifier and order probes

- Structural audit, no labels and no fitting: token format, first-character range, per-position alphabet coverage, the full reconstruction of `track_href` and `analysis_url` as fixed prefix plus token, `type` constancy, missingness, full-token and fragment coverage between the files (prefixes and suffixes of lengths 1 to 3), the structure of the test `ID`, multiplicity distributions, exact audio-plus-date and audio-only overlaps, training row position and test `ID` against release year, and whole-token support across groups; for the eligible rows, all training rows and the test rows.
- Blocks, each a separate namespace per fragment kind and length, case and leading characters kept, fitted inside each outer fitting population only:

| Block | Definition |
|---|---|
| `id_shape_v1` | Row-local: the 22 character codes in the alphabet `0-9A-Za-z`; six chunks (positions 0:4, 4:8, 8:12, 12:16, 16:20, 20:22) read in that radix; digit and uppercase counts, distinct characters, longest identical-character run, character entropy |
| `id_frequency_v1` | For prefixes and suffixes of lengths 1, 2 and 3: log1p of the number of distinct primary groups holding the fragment among the fitting rows, over distinct (fragment, group) pairs; a fitting row gets log1p(count - 1), any other row log1p(count), zero when unseen |
| `id_te30_v1` | Four columns, prefixes and suffixes of lengths 1 and 2: scikit-learn's `TargetEncoder`, binary target, smoothing 30, row-weighted sums and prior; fitting rows cross-fitted over five explicit stratified group folds of the fitting population on primary groups; other rows by the encoder fitted on all fitting rows; an unseen fragment gets that population's prior |
| Audio recurrence | Key: the exact 13-value audio tuple. Support: log1p of the distinct groups holding the key, the query's whole group left out for fitting rows. Deviation: (S - n mu) / (n + 30), with S and n the label sum and row count of the key and mu the mean label, all from the permitted partition: for a fitting row the inner training part of its inner fold, for other rows the whole fitting population; zero when unsupported. Supported validation rows and groups are reported |
| Row position | The original training-file row position, before the incomplete row is removed; missing for test rows |

- Fitting contract: `fit_probe(spec, fitting rows, labels, groups, FitContext)` builds every dictionary, inner split, encoder and estimator from the fitting population alone and returns a predictor whose `predict_high` encodes any rows; only token parsing and F2 are computed beforehand. No global encoding, out-of-fold or not, is sliced into an outer fit. Models: the phase 1 LightGBM and random forest factories unchanged, the forest predicting through part C's helper; the identifier-only linear model L: position-tagged characters, unpositioned adjacent bigrams and the tagged two-character prefix and suffix, hashed to 16,384 columns without alternating signs, rows L2-normalized, `LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=0)`.
- References: H is the phase 1 LightGBM F2 recipe, R the phase 1 random forest F2 recipe, both through the probe runner without any identifier column or placeholder. H must equal the phase 1 bank `production_lightgbm_F2_9c37e77d549e` in all ten arrays (same rows, ids, order, groups, saved folds, split seeds 155, 2026, 1029, estimator seed 0, the 20 F2 columns, the phase 1 parameters with four threads, `deterministic=True` and `force_col_wise=True`). R must equal its phase 1 bank in every array except predictions and fold AUCs, its predictions within 1e-12. Part A's H must also equal part B's H in all ten arrays.
- The declared menu, saved in `../../configs/phase2/p2.8/identifier_probe.json` before any probe score:

| Test | Challenger | Comparator | Kind |
|---|---|---|---|
| I1 | H with `id_shape_v1` | H | block |
| I2 | H with `id_frequency_v1` | H | block |
| I3 | H with `id_frequency_v1` and `id_te30_v1` | H | block |
| I4 | R with all three identifier blocks | R | block |
| I5 | 0.9 H + 0.1 L | H | blend |
| I6 | 0.75 H + 0.25 L | H | blend |
| I7 | H with row position | H | diagnostic, never deployable |
| I8 | H with the audio-recurrence block | H | block |

- Thresholds, applied to unrounded values: a block or the diagnostic earns follow-up when its mean fold AUC gain is at least 0.001 and its difference is positive on every repeat; L earns follow-up when either declared blend gains at least 0.0005 in mean and is positive on every repeat. These are exploration thresholds, not selection rules or significance tests. I3 minus I2 is reported as a description only.
- Bootstrap: every menu bank must be pairable with every other (one global check). Each comparison uses `paired_cluster_bootstrap` with 10,000 draws and seed 155001; with identical rows, labels, groups and folds the common seed gives the same group multiplicities in every comparison, so the draws are shared. Reported: the 95% interval, one-sided bounds at the 0.05/8 = 0.00625 and 0.99375 quantiles, valid and invalid draws. They are approximate, multiplicity-adjusted conditional bootstrap bounds: they exclude retraining variability and the project's earlier selection history.
- Controls: the identifier tests (I1 to I6) meeting their follow-up rule are ranked by mean gain divided by their threshold, ties by test id, and at most two distinct fitted pipelines are controlled (I5 and I6 share L, so they share its control fits). Each is refitted on pseudotokens under the three declared salts: the first 16 bytes of sha256("salt:token") written in the same alphabet with 22 characters, so equal tokens stay equal and the format is kept; same folds, same preprocessing. The real gain is not clearly distinguished from model perturbation when it is at most the largest control gain plus 0.0005, an exploratory convention, not a significance test. Every control result is kept; no salt is replaced.
- Implementation checks, without scores: held-out labels flipped with outer and inner splits fixed change nothing fitted; other test tokens change no fitting or validation input; row order changes no row-local, count or held-out output; unseen tokens and unsupported audio keys; hand-computed counts and encodings with mixed-label groups, several `uri` values in one primary group and unseen fragments; pseudotoken determinism, format and equality; zero whole-token support across groups; one-thread forest repetition; the declared menu.
- Probe banks are diagnostic: every one is marked `selection_eligible: false`, I7's with the reason `no_train_test_position_mapping`, and none is a submission candidate. A positive identifier result enters the phase 2 engine only through a later reviewed extension (`raw_columns_for(recipe)` with lineage, nested fitting, replay and submission checks that recognize the identifier as an input), evaluated against the then-current reviewed model. A null result reads: "On the declared grouped folds, these identifier representations did not demonstrate a repeat-stable improvement meeting the practical threshold. This does not establish that every possible identifier transformation is uninformative." If I7 is positive, the provenance of the file order is a question for Sam; no train-to-test mapping is invented.
- Boundaries: row-local parsing, fold-fitted dictionaries and encodings, descriptive overlap tables and training dictionaries applied to test rows are allowed; anything fitted on test rows (within-test repeat counts included), vocabularies fitted on train and test jointly, lookups by identifier and external data are excluded.

### 16.3 Part B: feature checks moved from P2.4, and the music block

- The `music` block (section 6.3), stateless, carried over from the earlier project (`../prior_project/README.md`): `key_fifths_sin` and `key_fifths_cos`, the sine and cosine of 2π((7 key) mod 12)/12 (key on the circle of fifths), and `tempo_octave_sin` and `tempo_octave_cos`, the sine and cosine of 2π log2(tempo) (tempo T and 2T coincide); raw key and tempo stay. Tempo is positive on every eligible and test row.
- Fourteen matched comparisons, each a challenger against the identical recipe without its addition, refitted under P2.8's code:

| Ids | Base (matched comparator) | Additions |
|---|---|---|
| B1 to B10 | H, the phase 1 LightGBM F2 recipe (the nine anchor checks of section 6.3 and one more) | `date_count`; `date_te10`; `date_te30`; `date_te10` with `date_count`; `date_te30` with `date_count`; `interactions`; `cyclic`; `era_relative`; inverse-group-size training weights; `music` |
| B11, B12 | `p22_random_forest_b1_0_040`'s recipe, P2.2's best random forest by the logged scores (section 15.7; not reranked) | `date_count`; `date_te30` with `date_count` (the random-forest checks of section 6.5) |
| B13, B14 | `p22_lightgbm_b3_001`'s recipe, P2.2's strongest standalone recipe | `date_count`; `date_te30` with `date_count` |

- Allocation state, saved in the P2.8 manifest before any P2.8 score, from P2.2's logged records: H (the anchor; P2.8's own H run must equal the P2.1 anchor bank in all ten arrays); the family incumbents (`p22_lightgbm_b3_001`, `p22_random_forest_b1_0_040`); the strongest standalone recipe (`p22_lightgbm_b3_001`); each family's previous best contribution blend with H (LightGBM `p22_lightgbm_b3_004` at 0.25, random forest `p22_random_forest_b1_0_040` at 0.25). For each challenger the summary computes its two contribution blends with H and the section 5.6 routes it meets against that state, separately from its matched difference: a block can improve its own base without beating the tuned incumbent. P2.8 launches no feature expansion. The 14 comparisons are the first 14 of P2.4's 32 initial comparisons; P2.4 imports their records and route flags without changing their references, charging them again or re-judging them against a newer leader, and its initial feature allocation makes progress if any initial comparison, these included, meets a route.
- Fixed-blend diagnostics, no extra fitting beyond the deterministic Extra Trees anchor refit: 0.75 H + 0.25 Extra Trees anchor against H (Finding 012's best contribution, reproduced under deterministic prediction), and 0.75 `p22_lightgbm_b3_001` + 0.25 Extra Trees anchor against that recipe. No weight search; neither becomes a finalist; P2.6b keeps its own H*, pool and seed rules.
- Statistics: unrounded mean fold AUC differences, the three repeat differences, the paired cluster bootstrap with 2,000 draws (descriptive), on the saved primary folds and the three split seeds.
- Reserved now, before any CatBoost or EBM score: in P2.4, three further reserved checks beside the four of section 6.5: P2.3's best CatBoost configuration (by unrounded mean fold AUC, ties to the lower recipe id) with and without `music`; the same with and without `interactions`; P2.3's best EBM configuration with and without `music`. If P2.3 cannot install CatBoost, XGBoost's best takes its place, as in section 6.5; if EBM is omitted, its check is dropped with the reason recorded. In P2.3: one declared configuration of the initial CatBoost batch is the earlier project's candidate 31 on F2 (depth 6, 1,944 iterations, learning rate 0.009535096982557112, L2 4.757135294733201, random strength 0, `rsm` 0.8880009208730744, Bayesian bootstrap with bagging temperature 1.793473187168845), exempt from the random space like an anchor; P2.3 adds `bootstrap_type`, `bagging_temperature` and `rsm` to the CatBoost factory for it. It is a carried-over starting point, not assumed optimal with release dates.

### 16.4 Part C: deterministic forest prediction

- `prediction_threads(family, estimators)` in `songpop/phase2/models.py`: for random forest and Extra Trees, every estimator's forest, bare or the final step of a pipeline, predicts with `n_jobs=1`; fitting keeps four threads; the previous setting is restored in `finally`. `FittedRecipe.predict_each` uses it for validation, test and every estimator seed; the fit receipt of a forest records `forest_prediction_threads: 1`; part A's runner uses the same helper. Fitted estimators are never predicted concurrently.
- Checks: (1) one fitted random forest and one fitted Extra Trees model (P2.2's most tie-heavy leaf-1 recipe) give identical outputs in 20 repeated predictions, their `n_jobs` restored; (2) the five non-forest P2.2 family anchors refitted under P2.8's code equal their P2.2 banks in all ten arrays; (3) the two forest anchors equal theirs in every array except predictions and fold AUCs, the predictions within 1e-12, every fold AUC difference reported; (4) each forest anchor refitted twice under P2.8's code is identical in all ten arrays; (5) self-checks with a pipeline fixture and a spy show four threads at fit and one at prediction. The seven anchor refits are regression coverage, not proof for every recipe path; the dispatch checks cover the rest.
- Later forest runs use a fixed prediction summation order; exact replay is checked under the recorded environment and execution path. The P2.2 forest runs keep their banks and replay at commit 810d1ce; section 15.7's criterion stays specific to them. P2.2's audit and refit run at 810d1ce, as the P2.1 anchor's run at 281a014: at the P2.8 head, `--audit-task P2.2` audits every bank and record but reports its preflight's source difference, by design.

### 16.5 Records, outputs and verification

- `../../configs/phase2/p2.8/manifest.json`, saved before any P2.8 score: part A's menu file and its hash; part B's recipes, comparisons, allocation state with the referenced run ids and bank hashes, fixed-blend diagnostics and reserved checks; part C's script; expected fits; the commands.
- Part A writes once to `../../artifacts/p2.8/identifier_probe/` (banks of schema `p28_probe_bank_v1`, written and read by `../../analysis/identifier_probe/probe_bank.py`, and `results.json`, never overwritten); `probe.py verify` refits every bank under the code running now, checks the recorded source manifest against its files and the running code, compares every bank's arrays and stable metadata with its refit, recomputes every statistic, the descriptive I3 minus I2 comparison, the controls, the coverage and L's score from the refits, and exits 1 on any difference. Part B's runs go to the phase 2 log as task P2.8 through the canonical entry point, so their source manifests match their replay path; `scripts.run_experiments --verify --task P2.8` refits every logged P2.8 run. `../../analysis/p28_feature_checks.py` computes part B's matched comparisons, bootstraps, contribution blends, routes and diagnostics, and the equality of part A's and part B's H, read-only from the logged banks; rerunning it reproduces its output. Part C's check is `../../analysis/p28_forest_check.py`.

| Step | Command (from the project root, `PYTHONDONTWRITEBYTECODE=1`) |
|---|---|
| Readiness | `.venv/bin/python -m scripts.check_phase2`; `.venv/bin/python -B -I analysis/identifier_probe/probe.py . audit`, `. checks`, `. timing` |
| C | `.venv/bin/python -B -I analysis/p28_forest_check.py .` |
| B | `.venv/bin/python -m scripts.run_experiments --recipe configs/phase2/p2.8/recipes/<name>.json --task P2.8` for each of the 18 recipes of the manifest, then `.venv/bin/python -B -I analysis/p28_feature_checks.py .` |
| A | `.venv/bin/python -B -I analysis/identifier_probe/probe.py . run` |
| Verification | `.venv/bin/python -m scripts.run_experiments --verify --task P2.8`; `probe.py . verify`; the C check and the B summary rerun |

- Fits: part A 9 fifteen-fit evaluations and at most 6 control evaluations; part B 18 (the 14 challengers, the three refitted bases and the Extra Trees anchor); part C 9 (the 7 anchors and the two forest replays) plus the single-fold repetitions. Verification refits parts A and B once more. The score-free timing measures part A's runner, its widest blocks and the bootstrap before any score; the projection is not a bound. If time slips, the fixed-blend diagnostics go first; verification, a triggered control and the reserved checks' declaration are never cut.

### 16.6 Findings

Finding 014, identifier structure and deployment availability (the audit); Finding 015, the identifier predictive probes and their controls; Finding 016, the feature checks moved from P2.4 with their routes and the fixed-blend diagnostics; Finding 017, deterministic forest prediction. Findings 011 to 013 become `reviewed` in this ledger revision, after P2.2's convergence and commit.

## 17. P2.3: CatBoost and EBM

Started on Sam's reply of 2026-10-30, verbatim: "Okay, 2P2.3 and 2.4, and then pause after you finish." Built after Codex consultation rounds O (an independent design, after Claude's positions were written down) and P (a critique of the draft), verbatim in `../llm_usage/2026-10-30-p2.3-consultation/` with what was adopted. It resolves what sections 4.9, 5.1 to 5.3, 5.6 and 16.3 leave to P2.3; where it is silent, section 15 applies with P2.3 in place of P2.2. The machine-readable counterpart is `../../configs/phase2/p2.3/protocol.json`.

### 17.1 Scope and checkpoints

P2.3 searches CatBoost and EBM on F2, with no feature block, training weight, removal check or estimator-seed search (those belong to P2.4 and P2.6b). One review record, `../../reviews/2026-10-30-p2.3-catboost-ebm.md`, with two checkpoints in the manner of section 15.1: execution readiness (the code and self-checks, the installation records, the timing and reservation, this section and the protocol file, the saved wave 1 manifests, the commands; its approval permits executing this section unchanged and covers no result), then completion (the banks, contribution, allocation and bootstrap records, the audit and the full refit, the findings and the handoff to P2.4; its approval permits P2.4 to build on P2.3). The source files stay unchanged from readiness through the refit.

### 17.2 Installation and factories

- Installation: executed on 2026-10-30 at about 05:42 PDT with the commands P2.1 reviewed (section 4.9), unchanged, CatBoost first: `scripts.install_locks check-before <route>`, `pip install --no-deps --require-hashes -r configs/phase2/install/<route>.txt`, `check-after <route>`, `pip check`. P2.1's review covers the locks and these exact commands; a changed wheel, version or command would need a new review. Both routes passed their inventory checks (`../../artifacts/phase2/install/inventory_before_*.json`, `inventory_after_*.json`); the phase 1 commands of exit criterion 5 pass afterwards.
- CatBoost (`songpop/phase2/models.py`). Recipe parameters: `depth`, `iterations`, `learning_rate`, `l2_leaf_reg`, `random_strength`, `boosting_type` (Plain or Ordered), and, for declared configurations, `bootstrap_type` (MVS or Bayesian), `bagging_temperature` (set exactly when the bootstrap is Bayesian; CatBoost 1.2.10 silently ignores it otherwise) and `rsm` (a float in (0, 1]). Fixed: `loss_function` and `eval_metric` Logloss, CPU, `SymmetricTree`, four threads, the estimator seed, no file writing or snapshots, `use_best_model=False`, no class weights, `nan_mode` Min, `has_time=False`, no evaluation set, no categorical features. A recipe without a bootstrap gets CatBoost's own CPU choice for this objective, made explicit: MVS with subsample 0.8, for Plain and Ordered alike; `rsm` 1.0 unless set. CatBoost's `predict_proba` uses every CPU core unless told otherwise, so prediction passes four threads (`prediction_view`); predictions are identical for one, four and all threads on a probe.
- EBM. Recipe parameters: `max_rounds`, `learning_rate`, `interactions`, either "none" (main effects only) or "declared3", the pairs (year, loudness), (year, acousticness), (energy, loudness) of section 5.2, resolved by name to the columns after feature construction. Fixed, so that interpret-core 0.7.8's defaults (automatic pairs "3x", a 0.15 validation split, early stopping, 14 bags) never apply: `validation_size=0`, `early_stopping_rounds=0`, one outer bag, no inner bags, four jobs, the estimator seed, and the remaining 0.7.8 defaults written out (log loss, 1,024 bins, 64 interaction bins, greedy ratio 10, smoothing rounds 75, two leaves, minimum leaf 4, missing values in their own bin); every feature binned as continuous, which is also what 0.7.8 infers for these float frames. With validation and early stopping disabled, each stage (main effects, then pairs) performs `max_rounds` times its number of terms boosting updates; greedy selection can distribute them unevenly across terms. The runtime check verifies the stage totals.
- Both take the feature frame as it is: a recipe with another preprocessing is refused.
- Runtime checks (`scripts/check_phase2.py`), replacing the two staged ones: fits on an F2 sample with missing date parts; predictions for all three date precisions (full dates, year-only dates, and the one month-only training row, 1979-06); two fits with one seed predict identically; the class-1 column; a pickle round trip; identical rows, missing values included, predicted identically; no new top-level project entry and no `catboost_info` directory (CatBoost's file writing and snapshots are disabled); training weights accepted (P2.4's route); CatBoost's resolved settings for Plain and Ordered, candidate 31's Bayesian bootstrap, temperature and `rsm` in effect, prediction with four threads, eight refusals; EBM's terms (20 main effects, plus exactly the three declared pairs), its fixed settings, pairs resolved by name on reordered columns, four refusals.

### 17.3 References and comparators

- H: the phase 1 LightGBM F2 recipe with estimator seed 0, as in P2.2. The protocol declares it as a reference family `lightgbm` with no space: its wave 1 batch holds H alone; the initialization fits H under P2.3's sources and checks its ten arrays against the P2.1 anchor bank before any search fit (the `h_check` of section 15.2); its batch is judged with no configuration, so its allowance is 0 and it never plans again; it takes no reservation and no refinement slot. Contributions are blends with this H.
- G, the strongest standalone recipe: the protocol's `prior_strongest`, P2.8's B14 (`p28_lightgbm_b3_001_date_count_te30`, run `p2_p28_lightgbm_b3_001_date_count_te30_seeds155-2026-1029_44725c40a3b4`, bank sha256 `83e5cbf1...`, unrounded mean fold AUC 0.8354213), the strongest complete run in the log when P2.3 opened. The log index binds it to its record (task, recipe id, bank hash, completeness), and before any planning, execution, audit, refit or handoff the runner validates it from its bank (`validate_prior`): the bank's bytes against the protocol's hash, its task and recipe, rows, labels, groups, split seeds, folds and test ids equal to the P2.1 anchor bank's, and its record's scores rebuilt from the stored predictions (`runs.audit_run`). It is read, never refitted under P2.3's sources. For wave 1, G is the better of H and that run; for later waves, the strongest of H, that run and the complete P2.3 runs of closed waves. B14 holds date encodings while P2.3 searches F2: G asks how close a family comes to the best standalone recipe so far, whatever its features. P2.8's fixed blends are not standalone recipes.
- Family anchors, incumbents, the previous best blend, tie rules and gains as in sections 15.2 and 15.4.

### 17.4 Batches and allocation

| Family | Index | Wave 1 batch | Expansion, ceiling |
|---|---|---|---|
| LightGBM (reference) | 0 | H alone | none |
| CatBoost | 1 | 48: the anchor; candidate 31 (section 16.3), exempt from the random space; 22 Plain and 24 Ordered seeded draws over section 5.2's space | 24, 120 |
| EBM | 2 | 12: the six settings of rounds and learning rate (700, 0.03), (300, 0.03), (300, 0.06), (700, 0.01), (1500, 0.01), (1500, 0.06), each with and without the declared pairs; the anchor is (none, 700, 0.03) | 6, 30 |

- The CatBoost anchor is a fixed, default-inspired configuration: depth 6, 1,000 iterations, L2 3, random strength 1, Plain and MVS are CatBoost 1.2.10's defaults; its learning rate is fixed at 0.0168, the automatic rate CatBoost chose on a probe of 3,090 rows and 1,000 iterations, rounded, and is not recalculated from each fold's training size. Candidate 31 keeps every recorded value; its boosting type, not in its record, is set to Plain. It is a ported starting configuration, not the earlier model reproduced, and it is evaluated on F2 without the `music` block (P2.4's reserved checks add it).
- Sampler seeds: 2026103000 + 100 x family index + batch index, plus 10 x branch index within the initial batch. Learning rate times iterations (CatBoost) or rounds (EBM) stays within 5 to 100 for draws, explicit configurations and proposals, the anchors and candidate 31 included; for EBM this is a search heuristic, since its rounds are not trees.
- Refinement: section 5.2 allows local refinement for at most four families in the base-feature stage, and P2.2 refined three (LightGBM, Extra Trees, logistic). P2.3 therefore refines at most one: at the barrier after wave 1, the family with allowance and the stronger standalone incumbent (tie rule of section 15.4) gets the two local batches of section 15.4; the other family gets broad expansions. Local types: CatBoost depth (ordinal 3 to 7), learning rate, iterations, L2 and random strength (scale; random strength moves from zero to 0.1), boosting type (categorical); EBM rounds and learning rate (scale), interactions (categorical). Candidate 31's bootstrap, temperature and `rsm` are inherited unchanged by its proposals, never moved. Its learning rate, 0.0095, lies below the space's 0.015: a local proposal keeps a parent's own value when it does not move that parameter, and a moved value stays inside the active bounds (`songpop/phase2/refine.py`; P2.2's parents all lay inside their bounds, so its plans are unchanged).
- Range extension (sections 5.2 and 15.4) within these hard bounds: CatBoost depth 3 to 8, iterations 200 to 3,000, learning rate 0.005 to 0.15, L2 0.1 to 300, random strength 0 to 5; EBM rounds 100 to 3,000, learning rate 0.003 to 0.1.
- Cutoff: no P2.3 batch starts, and no anchor is fitted, after Friday 2026-10-30, 12:00 PDT (19:00 UTC), leaving P2.4 its implementation, review and batches before Saturday 08:00. Allocation, the counter, contributions and bootstraps follow sections 5.6 and 15.5 unchanged.

### 17.5 Timing and compute

Score-free timing (`../../configs/phase2/p2.3/timing.json`), one 15-fit run on all three split seeds, four threads, before any P2.3 score:

| Family | Light | Representative | Expensive | Extended (beyond the space, for range extensions) |
|---|---|---|---|---|
| CatBoost | 2.8 s (depth 3, 400 iterations, Plain) | 11.0 s (depth 5, 1,200, Plain) | 131.2 s (depth 7, 2,000, Ordered) | 371.0 s (depth 8, 3,000, Ordered) |
| EBM | 5.5 s (main effects, 300 rounds) | 13.5 s (pairs, 700) | 27.0 s (pairs, 1,500) | 54.4 s (pairs, 3,000) |

The timing set is the one made under the readiness sources: one record per family and case, from one specification; the six records timed earlier under draft sources stay in the log as history and are not used. The reservation record names the timing records and the protocol hash, and projects 42 minutes for the initial batches and 104 minutes at both ceilings, from the mean of the light, representative and expensive cases; the extended case is reported, not averaged. The full refit is projected to repeat the fitting actually completed; the ceiling projection is not a runtime bound. A range extension to depth 8 or 3,000 iterations makes some later CatBoost configurations up to about three times the expensive case; the cutoff bounds the last batch start, and a started batch completes. The representative CatBoost run costs far less than 120 s, so nothing is screened on one split seed: every configuration runs on all three. One process with four threads (the M5 has four performance cores); no log shards.

### 17.6 Records, verification and handoff

- Records, preflight, interruptions and verification as in section 15.5, with the P2.3 protocol file: `--audit-task P2.3` without refitting, then `--verify-task P2.3`, a full refit of every distinct P2.3 run, H included. The verification criterion is the existing classical one (section 4.5): identities and stable metadata match exactly, predictions and AUCs within an absolute 1e-12; H must also equal the ten P2.1 reference arrays bit for bit. Section 15.7's forest exception does not apply to P2.3. The phase 1 commands of exit criterion 5 run again at completion. At the P2.3 head, `--audit-task P2.2` and P2.8's replays report their source differences, by design (section 16.4); P2.2's wave 2 and 3 manifests, re-planned under P2.3's protocol code with their logged run ids, are unchanged.
- Handoff to P2.4, `../../analysis/p23_summary.py` (read-only): per family the best complete configuration by unrounded mean fold AUC, ties to the lower recipe id (section 16.3), and the best contributor to H (section 15.4's rule: the canonical recipe hash, then the weight), with recipes, recipe hashes, run ids, bank paths and hashes, source-manifest hashes, split and estimator seeds, repeat means and H's bank hash; an unavailable family with its reason. It fails closed: no handoff without a passed H check, a valid prior member of G, the task complete by the runner's own rule of section 15.5 (every batch closed, except batches of the last wave left unstarted because the cutoff passed, which the handoff reports as omitted; no further wave due before the cutoff), and every exported bank audited against its record. P2.4's reserved checks (sections 6.5 and 16.3) take these.
- Findings: 018 (installation, runtime compatibility, determinism checks and the score-free timing) as a draft at readiness, the score-free evidence the searches rely on; 019 (the CatBoost and EBM searches, with the replay of every searched configuration) and 020 (their contributions and allocation outcomes) at completion, exploratory in the sense of section 15.6.

| Step | Command (from the project root, `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.run_experiments`) | When |
|---|---|---|
| Timing and reservation | `--timing configs/phase2/p2.3/timing.json --task P2.3`, `--reserve --task P2.3` | Before readiness (done at the readiness sources) |
| Wave 1 manifests | `--plan-wave 1 --task P2.3` | Before readiness |
| Initialization | `--run-wave 1 --anchors-only --task P2.3` | After readiness converges |
| Wave 1 | `--run-wave 1 --task P2.3` | After the initialization |
| Later waves | `--plan-wave N --task P2.3`, then `--run-wave N --task P2.3`, N = 2, 3, ..., until no family remains or the cutoff | After each wave |
| Audit, refit, handoff | `--audit-task P2.3`, `--verify-task P2.3`, then `.venv/bin/python -B -I analysis/p23_summary.py .` | Before completion |

## 18. P2.4: the feature stage

On Sam's instruction of 2026-10-30, verbatim: "Okay, 2P2.3 and 2.4, and then pause after you finish." Built after Codex consultation rounds Q (an independent design, after Claude's positions were written down) and R (a critique of the draft), verbatim in `../llm_usage/2026-10-30-p2.4-consultation/` with what was adopted. The draft was developed in an isolated copy of the project while P2.3 executed, so P2.3's frozen sources were never touched; it entered the project after P2.3 converged, and production execution starts from that revision. It resolves what sections 6 and 16.3 leave to P2.4. The machine-readable counterpart is `../../configs/phase2/p2.4/protocol.json`, written by `../../analysis/p24_make_protocol.py`; the code is `../../songpop/phase2/feature_stage.py` (pure planning and judging) and the feature commands of `../../scripts/run_experiments.py`.

### 18.1 Scope and checkpoints

The feature stage of section 6: 32 initial comparisons (P2.8's 14 imported, 18 new), expansion batches of 16 up to 96 comparisons, then the re-tuning of up to four model-and-feature combinations, each in at most two batches of 16. Neural comparisons are omitted because P2.5 has not run (section 6.1). One review record, `../../reviews/2026-10-30-p2.4-features.md`, with readiness and completion checkpoints in the manner of section 15.1; the sources stay unchanged from readiness through the refit. After P2.4 converges, work pauses (Sam's instruction).

### 18.2 Imports

P2.8's 14 comparisons (section 16.3) enter with their original definitions and decisions. The protocol's import receipt holds exactly B1 to B14, in order: the P2.8 manifest, its judge `../../analysis/p28_feature_checks.py` and its output by sha256, P2.8's H run, and for each comparison its family, comparator and challenger run ids, bank hashes and recipes, and route flags (B12: ensemble contribution; B14: smaller useful, competitive refinement, ensemble contribution; the other twelve none). Before any planning, run or audit, `validate_imports` checks the files' hashes; that each entry is the hashed manifest's comparison of that id (family, recipe names and recipe hashes); that its runs are P2.8 runs of exactly those recipes, banks matching their hashes and records (`runs.audit_run`); that the H is the manifest's named H, audited; and it recomputes every route flag from P2.8's own frozen allocation state with that H. A newer reference never re-judges an import. The imports count in the initial batch's progress only.

### 18.3 References, matched effects, witnesses and the counter

- H: the phase 1 recipe with estimator seed 0, refitted under P2.4's sources as each batch's first run and checked against the P2.1 anchor bank in all ten arrays before any other fit (the `h_check` of section 15.2). Contributions are blends with this H at the challenger weights 0.1 and 0.25.
- The reference pool: the protocol's frozen reference inventory (every complete fitted run of P2.2, P2.3 and P2.8 in the log when the protocol was written after P2.3 converged, each with its bank hash) and the runs of P2.4's judged batches. Every pool run is validated from its bank before use (its bytes against the expected hash, its record against the bank, `runs.audit_run`), and its recipe, features and scores are read from the bank; a later run of those tasks never enters. Each batch's manifest freezes, from the pool as it stands before the batch: G, the strongest standalone pool run; for each family in the batch, its incumbent, the family's best pool run (feature recipes included, so B14 is LightGBM's incumbent when P2.4 opens); and the family's best contribution blend with H, floored at H, computed from the banks with the P2.1 anchor bank's arrays (equal to H's by every H check). Ties go to the lower recipe id, and for blends then to the lower weight. P2.2's forest runs enter with their logged banks (section 15.7). The manifest also names the earlier P2.4 batches and the hashes of their judgments that it used.
- For every comparison the judgment records two results apart: the matched effect (challenger minus comparator, the three repeat differences, and the paired cluster bootstrap with 2,000 draws as a logged comparison record), and the routes of section 5.6 the challenger meets against its family's frozen references. A block can improve its base without beating the incumbent.
- Witnesses, as in section 15.5: per family and route met, the qualifying run with the highest relevant score against the frozen incumbent (standalone routes), or its better blend against H and against the family's previous best blend (ensemble route), bootstrapped with 2,000 draws and logged before the batch's allocation; descriptive, never a gate.
- One counter for the whole feature stage, as in section 5.6: the initial batch grants two expansion batches if any of its comparisons, the imported ones included, meets a route (section 16.3); a later batch resets the counter to two when one of its own comparisons meets a route and uses one otherwise. The stage closes at zero, at 96 comparisons, when every expansion queue is exhausted, at a recorded stop (18.9), or at the cutoff.

### 18.4 The initial comparisons

Aliases, resolved by rule from the reference inventory when the protocol is written and frozen in it; CatBoost and EBM come from P2.3's handoff (`../../analysis/p23_summary_output.txt`, hashed in the protocol), and a fallback or omission is allowed only where that handoff records the family as unavailable:

| Alias | Rule |
|---|---|
| L | `p22_lightgbm_b3_001`, P2.2's strongest recipe on F2 |
| C | P2.3's best complete CatBoost by unrounded mean fold AUC, ties to the lower recipe id (sections 6.5, 16.3; the handoff's best standalone); X if the handoff records CatBoost unavailable |
| M | P2.3's best complete EBM, the same rule; its comparison is omitted, with the reason, only if the handoff records EBM unavailable |
| X | P2.2's best XGBoost |
| E | P2.2's Extra Trees anchor, Finding 012's best contributor to H |
| S | P2.2's best spline logistic recipe on F1 |
| K | P2.2's best kNN recipe |
| HG | P2.2's best HistGB |
| R | the strongest complete run of P2.2, P2.3 and P2.8 whose features hold `days` |

| Ids | Base | Challenger |
|---|---|---|
| B1 to B14 | P2.8's | imported unchanged (18.2) |
| B15, B16, B17, B18 | C | with `date_count`; with `date_count` and `date_te30`; with `music`; with `interactions` (sections 6.5, 16.3) |
| B19 | M | with `music` (section 16.3) |
| B20 to B26 | L | with `date_te10`; `date_te30`; `date_count` and `date_te10`; `date_te3`; `date_count` and `date_te3`; `date_te100`; `date_count` and `date_te100` (B13 and B14 hold the count alone and with `date_te30`) |
| B27, B28 | E | with `date_count`; with `date_count` and `date_te30` |
| B29 | S | on F2 with `spline_calendar` (18.7): a bundled change, F2's date parts with the preprocessing they need |
| B30 | S | with `cyclic` and `spline_calendar` (on F1 `spline_calendar` builds exactly `spline`, so this isolates `cyclic`) |
| B31 | K | `knn_blocks_v2` with robust audio scaling (18.7) |
| B32 | R | without `days` (section 6.6) |

Every comparator is its recipe refitted under P2.4's sources (or, when it equals a declared P2.4 challenger, that challenger's run); every challenger is the comparator with only the declared change. Additional interactions, era-relative features, training weights and most removal checks wait for the expansions.

### 18.5 Expansion batches

Each expansion batch of 16 is a pure function of the protocol, the pool and the closed batches, saved before its first fit. Its slots by queue, in this order: dates 6, combinations 2, smooth and kNN preprocessing 2, removals 4, other tree families 1, training weights 1. A queue without candidates releases its slots to the first queue in this order that still has some. Only applicable changes whose challenger is not declared yet (the imports included) are candidates; nothing is padded.

| Queue | Candidates, in order |
|---|---|
| Dates | Bases: the comparators of date comparisons that met a route, by their best challenger's score; then, for XGBoost, CatBoost, HistGB and EBM in that order, a family without any date comparison yet, its best pool recipe without date blocks. A base enters only with an untried variant; at most two bases per batch. Within a base, the untried variants of `date_count` with `date_te30`, with `date_te10`, `date_te10`, `date_te30`, `date_count` with `date_te3`, with `date_te100`, `date_te3`, `date_te100`, `date_count` alone |
| Combinations | Two comparisons on the same comparator, each adding blocks and each meeting a route: their additions united, at most one target encoding |
| Smooth and kNN | A declared menu with matched comparators: B29's recipe (S on F2 with `spline_calendar`) against it plus `date_count`, plus `date_te30`, plus `cyclic`; B31's recipe (robust Euclidean kNN) against robust Manhattan kNN |
| Removals | For each drop of section 6.6 (`days`; `day` and `day_of_year`; `weekday`; key, mode and time signature), the two strongest pool recipes for which it is new; for a combination that met a route, the combination without each block it added |
| Other trees | A declared menu: E and X with `interactions`, then `cyclic`, then `era_relative`; HG with `interactions`, then `cyclic` (Finding 016's negative anchor checks do not settle other families) |
| Weights | Inverse-group-size training weights on challengers that met a route, by score; never kNN |

### 18.6 Re-tuning

- After the feature stage closes, at most four units are chosen. A unit fixes the family, base, blocks and their order, drop list, preprocessing (all but spline knots), training weights and estimator seeds, and is identified by a hash of those. Candidates are units of challengers that met a route (imports included), carry a feature change, and belong to a family with a declared re-tuning space; up to two are chosen by their best standalone score, then by their best contribution blend, then by standalone score again; at most two per family; ties by unit id.
- Each unit gets two batches, as section 5.6's allowance of two gives a qualifying family two opportunities: round 1, then round 2 for every round 1 unit, whatever round 1 showed (its routes are recorded). A batch is 16 configurations, 12 local proposals and 4 broad draws, through P2.2's refine code with the unit's features fixed: parents the unit's best standalone configuration and its best contributor of a different computation; local moves of one to three parameters with the tree variants first; broad draws from the family's space of sections 15.4 and 17.4 (types and admissible regions as there) keeping the parent's parameters outside that space; every earlier configuration of the unit excluded. Configurations compare by the family-normalized key of section 15.4 (omitted defaults and number types normalized), never by raw recipe hash. Each round's references are frozen before it; routes use the actual family incumbent. Sampler seed 2026103100 + 100 x unit order + round.
- Logistic units re-tune only C, and kNN units only the neighbour count and weighting, with their representation fixed; kNN's cyclic calendar alternative is deferred, not covered by `knn_blocks_v2`.
- Admission: units are admitted in order while their projected fitting and replay (32 times the slower parent's measured 15-fit time per round) end before the protocol's re-tuning budget end, Saturday 2026-10-31, 05:00 PDT (12:00 UTC), counted from the round's planning time, which its record keeps; an omitted unit is listed with the reason.

### 18.7 Blocks and preprocessing added

- `date_te3` and `date_te100`: the encoder of section 6.4 with m = 3 and 100; every smoothing value shares one inner split per fitting population; a recipe holds at most one target encoding.
- `spline_calendar`, for logistic regression: the phase 1 log transforms; splines with scaling on the eleven core columns of section 6.2 only; key, mode, time signature and date precision one-hot; cyclic coordinates zero-imputed as linear inputs with one missingness indicator per phase pair; every other column (the further F2 date parts, the date blocks) median-imputed and standardized. On F1 it builds exactly the `spline` preprocessing.
- `knn_blocks_v2`: `knn_blocks` with an `audio_scaler`, standard (identical to `knn_blocks`) or robust (`RobustScaler` with the interquartile range and unit variance, audio only).
- Old recipes and their hashes are unchanged.

### 18.8 Date coverage

`../../analysis/p24_date_coverage.py`, descriptive: for each split seed and outer fold, the outer validation rows and the test rows against the outer fitting rows' exact dates, and each inner held-out part against its inner fitting part with the encoder's own inner splits; row-weighted and distinct-date coverage and distinct-group support (0, 1, 2 to 4, 5 or more); per-fit shares and their mean apart from pooled counts. Nothing is fitted on test rows and no smoothing value is chosen from it. The encoding is an exact-date association, not album identity.

### 18.9 Timing, schedule, stop and cut order

Score-free timing of the feature recipes before readiness (`../../configs/phase2/p2.4/timing.json`, from the resolved aliases): a date-encoded tuned LightGBM, C with `date_count` and `date_te30`, `spline_calendar` on F2, `knn_blocks_v2`, M with `music`, and, when C is CatBoost, its expensive re-tuning corner. Serial, one process, all three split seeds. No P2.4 batch starts after Saturday 2026-10-31, 04:00 PDT (11:00 UTC), four hours before the schedule guard's 08:00, so that the refit and the completion review can finish before P2.6b's inputs are due at 10:00; a started batch completes, and re-tuning is admitted by measured cost (18.6). The cut order's first items are executable by a recorded stop: `--stop-features` after a judged feature batch logs a `feature_stop` record with its reason and the expansion it omits (re-planned, not run); it closes the feature stage and lets re-tuning begin. Cut order: expansions without recent progress, redundant combinations and removals, round 2 batches and lower-priority units (by the admission rule), a second expensive CatBoost unit.

### 18.10 Records, verification, findings and handoff

- Every batch manifest (`../../artifacts/phase2/batches/p24_<stage>_<index>_<hash>.json`) is written once before its first fit and logged as a `feature_batch` record with its planning time; a re-planning in memory must equal a logged batch before anything is written; before any fit a preflight checks its file, batch id, stage, index and kind, protocol and source hashes, run ids and its re-planning from the log as it stood before the batch; a `batch_start` record precedes the first fit; H runs first through the same checked path as every run, a failed fit logs a `run_failure` record; each batch closes with its witnesses' bootstraps and then a `feature_allocation` record holding its judgment.
- Completion: `--audit-features P2.4` checks the imports; the log's structure (one initial batch, contiguous indices, re-tuning after the features, at most one allocation, start and stop per batch, no orphan record, no unexplained record kind); the log's order (every batch record after the batch records and allocations of the batches before it; within a batch its record, its start, its execution, its allocation; H's run, then its check, before the batch's other fits; contributions and matched comparisons after the runs they use; the witnesses' bootstraps after the rest of the batch); every manifest against its file and re-planning from the log as it stood before its record; every declared run logged and audited against its bank; every matched comparison, contribution pair, witness bootstrap and H check present and recomputed; every allocation re-judged; no P2.4 run outside the manifests; the initial batch judged, no started batch unjudged, and nothing still due before the cutoff. Then `--verify --task P2.4` refits every P2.4 run and every logged comparison. The classical criterion of section 4.5 applies (1e-12); H bit for bit; imported P2.8 runs keep their own replay association.
- Findings 021 (date coverage and support), 022 (the matched feature effects, imports included, with their routes) and 023 (the expansions and the re-tuning outcomes), exploratory in the sense of section 15.6; adaptively chosen comparisons on repeatedly used folds increase selection optimism, which three repeats and conditional bootstrap intervals do not remove.
- Handoff to P2.6b: an index of every usable P2.4 recipe with its run id, bank hash, features, standalone result, contribution blends and lineage, lower-standalone contributors included. P2.6b keeps its own H*, pool and seed rules; the weights 0.1 and 0.25 are exploration checks, not deployment weights.

| Step | Command (from the project root, `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.run_experiments`) |
|---|---|
| Timing | `--timing configs/phase2/p2.4/timing.json --task P2.4` |
| Initial batch | `--plan-features 1 --task P2.4`, then `--run-features 1 --task P2.4` |
| Expansions | `--plan-features N --task P2.4`, then `--run-features N --task P2.4`, N = 2, 3, ..., until nothing is due |
| Recorded stop, if the schedule needs it | `--stop-features --task P2.4 --reason "<reason>"` |
| Re-tuning | `--plan-retune K --task P2.4`, then `--run-retune K --task P2.4`, K = 1, 2 |
| Audit and refit | `--audit-features P2.4`, then `--verify --task P2.4` |

## 19. P2.5: the TabM challenger

On Sam's instruction of 2026-10-30, verbatim: "continue the work finish P2.5 and 2.6 then pause." and "Finished 2.4 before doing". Built after Codex consultation rounds S (an independent design, after Claude's positions were written down) and T (a critique of the draft), verbatim in `../llm_usage/2026-10-30-p2.5-consultation/` with what was adopted. Under agent-workflow v1 (`../../AGENTS.md`) it runs provisionally under that file's review triggers: no readiness or completion review is required; its results get an R1 review when they feed the final selection, and an R2 or R3 review if a later change triggers one. It resolves what section 5.5 leaves to the task; where this section is silent, section 5.5 applies.

### 19.1 The family

- Installation: torch 2.14.1 and tabm 0.0.3 from the lock reviewed in P2.1 (`../../configs/phase2/install/tabm.txt`), with `install_locks check-before tabm` and `check-after tabm` (both OK, 2026-10-30). The package identity of `songpop.artifacts.packages` is unchanged; a TabM run's `packages` add `tabm` and `torch`.
- `songpop/phase2/tabm.py`: the `tabm` family. Parameters `n_blocks`, `d_block`, `dropout`, `learning_rate`, `weight_decay`, `batch_size`, all six set by every recipe; fixed by the factory: 16 members, `arch_type` "tabm" with ReLU and random-sign scaling initialization, no numerical embeddings, at most 300 epochs, patience 20, tolerance 1e-6, a five-fold inner split, four threads, AdamW with betas (0.9, 0.999), epsilon 1e-8 and weight decay on every parameter.
- Preprocessing, owned by the estimator because its two stages fit it on different rows: `tabm_numeric` with `scaler` "standard" (the phase 1 log transforms, median imputation, standardization) or "quantile" (the ten continuous audio columns quantile-transformed to the normal distribution, at most 256 quantiles; the calendar columns standardized). Both add a missing indicator for each of the F2 fields month, day, day of year, weekday and days present, and one-hot encode key, mode and time signature with categories from the fitting rows, an unseen category all zero.
- Stopping and refit (recipe field `iterations`: "early_stop_refit_v1"; every other family keeps "fixed"): the first fold of a `StratifiedGroupKFold` of the fitting rows, seeded from the fit context alone (shared by every configuration and estimator seed on one fitting population); preprocessing fitted on its training part; after each epoch the log loss of the averaged member probabilities on its validation part; a stop after 20 epochs without an improvement larger than the tolerance; the chosen epoch the earliest within the tolerance of the lowest loss. Then preprocessing is refitted on all fitting rows and a new network is trained for exactly that many epochs, with the same batch size (so about a quarter more updates). Initialization and minibatch order have separate seeds for each stage, from the fit context and the estimator seed. Members train on the same batches with the mean of their binary cross-entropies; predictions average the members' probabilities, in fixed batches of 256 rows. Each fit's receipt records the inner split, the preprocessing rows and digests, the seeds, the losses, the chosen epoch, why it stopped, and the update counts.
- Leakage boundary: a TabM recipe uses base F2, takes no training weights and none of the fitted blocks (`date_count`, the date target encodings, `era_relative`), because their output would carry information across the inner stopping split; the stateless blocks are allowed.
- Process isolation (Finding 024): LightGBM and XGBoost load Homebrew's OpenMP runtime and torch its own; using torch after either crashes the process. Every torch computation runs in one persistent worker process (`python -m songpop.phase2.tabm`), which imports neither; the fitting process holds the fitted preprocessing and the weights as arrays. The worker first sends a hello (its protocol, the sha256 of `tabm.py`, Python, one inter-op thread, float32), which must match the fitting process; every exchange is a length-prefixed frame with a request id and a deadline; any failure during an exchange discards the worker. A torch import in the fitting process is refused.

### 19.2 Search

The wave protocol of sections 15 and 17 (`../../configs/phase2/p2.5/protocol.json`, written by `../../analysis/p25_make_protocol.py`): the LightGBM reference family (H, refitted and checked against the P2.1 anchor bank in all ten arrays), `prior_strongest` the strongest complete standalone run of P2.2, P2.8, P2.3 and P2.4 in the log when the protocol was written, and the `tabm` family. P2.5 ran in an isolated worktree of the project while P2.4 finished in the main tree (two fitting processes of four threads, within section 5.1's limit), so no P2.4 run was logged yet and G's prior member is P2.8's leader, as in P2.3; G enters only the two routes that compare with the strongest recipe. Initial batch of 8: the anchor N0 (two blocks, width 128, dropout 0.1, learning rate 1e-3, weight decay 1e-4, batch 256, standard scaling) and seven variants (six change one setting: width 64; width 256; three blocks; learning rate 3e-4; learning rate 3e-3; quantile scaling, the matched preprocessing comparison P2.4 omitted; one changes a regularization trio: dropout 0.2, weight decay 1e-2 and batch 128). Then, under section 5.6, at most two local refinement batches of 4 (two proposals around the standalone parent and one around the contribution parent, then one and two; one broad draw; at most two dimensions moved), ceiling 16; range extension is impossible, because the hard bounds are the space's edges. A budget in the protocol (10,800 s, section 5.5's three hours) stops the search before any wave after the first whose projection (the task's search runs so far plus the wave at its slowest run so far) would exceed it; the stop is logged as a `budget_stop` record and closes the task for the completion audit. Every configuration runs on all three split seeds with estimator seed 0; nothing is screened on one split seed, so no partial-seed candidate exists. This replaces section 5.5's screening sentence for P2.5. The cutoff is Saturday 2026-10-31, 08:00 PDT: no batch starts later.

### 19.3 Timing, records, verification and handoff

- Score-free timing of a light, a representative and an expensive configuration (`../../configs/phase2/p2.5/timing.json`), all three split seeds, before any P2.5 score; the reservation from it. If 8 configurations do not fit section 5.5's three hours, the initial batch is reduced before any score.
- Records, audit and refit as in section 17.6, with the P2.5 protocol: `--audit-task P2.5`, then `--verify-task P2.5`. The criterion is the classical one (1e-12), since the repeatability pilot showed bit-identical predictions across processes.
- Finding 024 (the installation, the process isolation, the repeatability and the timing) and Finding 025 (the search: standalone and contribution results), draft and exploratory.
- Handoff to P2.6b (`../../analysis/p25_summary.py`): every complete TabM run with its recipe, run id and bank hash, its standalone result and its contribution blends with H. Default seed policy for P2.6b: estimator seed 0 (16 members are one fitted model); TabM enters learned combiners only if its measured cost fits (section 5.5).

| Step | Command (from the project root, `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.run_experiments`) |
|---|---|
| Protocol | `.venv/bin/python -B analysis/p25_make_protocol.py` |
| Timing and reservation | `--timing configs/phase2/p2.5/timing.json --task P2.5`, `--reserve --task P2.5` |
| Waves | `--plan-wave 1 --task P2.5`; `--run-wave 1 --anchors-only --task P2.5`; `--run-wave 1 --task P2.5`; then `--plan-wave N`, `--run-wave N`, N = 2, 3, until nothing is planned |
| Audit, refit, handoff | `--audit-task P2.5`, `--verify-task P2.5`, `.venv/bin/python -B -I analysis/p25_summary.py .` |

## 20. P2.6: ensembles, the selector, the nested evaluator and the menu

On Sam's instruction of 2026-10-30 (section 19). Built after Codex consultation rounds U (an independent design, after Claude's positions were written down) and V (a critique of the P2.6a draft), verbatim in `../llm_usage/2026-10-30-p2.6-consultation/` with what was adopted. Under agent-workflow v1, P2.6a and P2.6b's evaluations run provisionally; the frozen menu and the selection rule get an R2 review before P2.7 builds on them; a submission file gets an R1 review before Sam uploads it. After P2.6b, work pauses: P2.7 is prepared, not launched. It resolves what sections 7, 8 and 11 leave to the task.

### 20.1 P2.6a: the engine (`songpop/phase2/procedures.py`, `selection.py`, `nested.py`)

- Populations and folds: a fitting population is raw rows indexed by permanent row id, labels and primary groups; its folds are a `StratifiedGroupKFold` of its rows in row-id order, seeded from protocol coordinates only, checked for exactly-once coverage, whole groups and both classes in every training complement. Every fit carries a `FitContext` naming its place in the protocol (`engine.run_recipe_cv(..., contexts=)`), so target encodings and TabM stopping inside a meta or nested fit never reuse an outer fit's inner split.
- Atomic predictions inside a population (`atomic_cv`) go through a content-addressed cache keyed by the whole request (fitting rows, their raw inputs, labels and groups; the folds and every fit context; the recipe and its estimator seeds; the prediction rows and their raw inputs; the source manifest and packages); a stored request must equal the caller's, so an entry for another population, fold assignment, context, query or source is refused, never repaired. The cache lives in `artifacts/phase2/cache/` (ignored by git); a replay uses an empty namespace.
- Learned procedures (section 7.2): for each outer training part D, its five meta folds (seeded from the protocol, the population, the split seed and the fold, never from the candidate or its setting), each channel's out-of-fold probabilities rebuilt inside D (estimator seeds averaged first; a within-family average's atoms averaged), the combiner fitted on that matrix and D's labels, the channels' fits on D predicting new rows. In production the channels' fits on an outer training part are their production fold fits, so their outer predictions come from their production banks after `check_channel_bank` shows each bank is that channel's computation on the saved folds; the global out-of-fold bank is never meta-training data. All six settings share one set of meta predictions.
- Combiners, fixed before evaluation: constrained weights (nonnegative, summing to one, minimizing the mean log loss of the convex combination clipped at 1e-6 plus lambda times the squared distance to equal weights; SLSQP from equal weights with the analytic gradient, `ftol` 1e-12, at most 2,000 iterations; solver failure, infeasibility beyond 1e-9 or a loss above equal weights' raises, with no fallback; weights below 1e-9 set to zero and renormalized, raw and used weights recorded; the prediction is the unclipped convex combination); the stacker (probabilities clipped to [1e-6, 1 - 1e-6], logits, standardized on the meta-training matrix, L2 logistic regression with an intercept, lbfgs, tolerance 1e-10, at most 5,000 iterations, a convergence warning raises).
- A learned procedure's bank keeps the ten arrays (out-of-fold probabilities, fold AUCs recomputed from them, the 15 fold procedures' test predictions and their mean) with its own schema, `procedure_v1`: the procedure, its channels and their bank hashes, every fold's combiner state and meta-fold receipt, the sources and the cache use. It is refused as a fitted bank.
- The selector (`select_candidate`, section 11 steps 1 to 3): exactly the menu's candidates, each complete on three repeats of five folds of one population, fold AUCs recomputed from the probabilities; the tie set is every candidate within 0.001 of the single maximum (not a chained comparison); within it the lexicographic minimum of (atomic recipes, supervised feature transformations, learned combiners, complexity rank, candidate id).
- The nested evaluator (section 8): three outer folds (seed 20261101), inner folds with seeds 155, 2026 and 1029 rebuilt in each outer training part, every candidate's inner evidence from the outer training rows only, `select_candidate` on that evidence alone, the selected candidate's inner-fold average and, separately, its full refit predicting the outer validation rows, the anchor's inner-fold average; `decide` reads outer labels only after all three folds: the fallback first, then the deployment rule; the reduced path refits a fixed candidate and the anchor and decides nothing.
- Self-checks (`scripts/check_phase2.py`, "(P2.6a)"): the selector's boundary cases (a chained tie, the tie keys, a partial bank, forged fold AUCs, other folds, a menu mismatch); the folds, the combiners' constraints and failures, the cache's refusal; label isolation at the outer, meta and nested levels with positive controls; fits confined to their permitted rows; procedure banks kept apart from fitted banks. A score-free cost pilot (`../../analysis/p26_cost_pilot.py`) measures one outer fold of each level.

### 20.2 P2.6b: nominations and final seed policies (`../../analysis/p26_nominate.py`)

The discovery inventory: every complete atomic run of P2.1, P2.2, P2.8, P2.3, P2.4 and P2.5, audited against its bank, one per canonical recipe. The discovery reference R0 is the strongest; contributions (0.9 and 0.75 blends) are recomputed against this one R0 for every other run. Nominees: R0; each family's standalone winner and contribution winner; up to two further configurations of R0's family within 0.002 of R0 (distinct atoms); the anchor (estimator seed 0) and its three-seed version. Final seed policies (section 7.1): estimator seeds 0, 1 and 2 for every family except kNN (deterministic, one fit) and TabM (seed 0, section 19.3); each nominee runs once under its policy (`--recipe ... --task P2.6b`). An atomic recipe is its exact feature, preprocessing, weighting, model and training configuration without its name, estimator seeds or deployment wrapper.

### 20.3 H\*, the pool and the procedure manifest (`../../analysis/p26_pool.py`)

H\* is the strongest final-policy nominee (ties to the lower recipe id); contributions are recomputed against it. Each family's representative is its strongest final-policy nominee other than the anchor's versions. The pool, in this order, to at most eight atomic recipes: the three strongest family representatives; up to two further recipes by their best contribution blend with H\* (possibly another configuration of a represented family, which keeps Finding 012's distinction); the anchor; up to two further configurations of H\*'s family within 0.002 of it, which form the within-family average with H\*. Channels: one per represented family other than the anchor, its contribution recipe when that is in the pool, else its representative; the within-family average replaces H\*'s family; ordered by the representative's standalone mean before any procedure is scored. A family whose measured cost does not fit the learned-procedure reservation (TabM, section 5.5) is left out of the channels, recorded before any procedure score, and keeps its fixed blends. The manifest (`../../configs/phase2/p2.6/procedures.json`, committed before evaluation): the pool's atoms and the anchor's three-seed version as standalone candidates; equal averages of the two and three strongest channels and of every channel; the 0.9 and 0.75 blends of each pool recipe with H\*; the within-family average; constrained weights with lambda 0.1, 0.01, 0; the stacker with C 0.01, 0.1, 1; at most 32 procedures. Structural counts: distinct atoms; supervised feature transformations (each distinct date target encoding once; counts, scaling, splines, fixed interactions, era-relative columns and estimator training zero); learned combiners. The complexity rank is the order of (kind: atomic, fixed blend, constrained, stacker; atoms; supervised transformations; stronger regularization first; id).

### 20.4 Evaluation, menu and freeze

Fixed blends are logged blend runs (`--blend`); learned procedures run with `--run-learned configs/phase2/p2.6/procedures.json --task P2.6b` and are verified with `--verify-learned` from an empty cache. The menu (`../../analysis/p26_menu.py`, `../../configs/phase2/menu.json`, at most 12): the anchor, its three-seed version, the standalone leader, the within-family average, the best fixed blend, the best constrained-weight procedure and the best stacker, each chosen by the frozen rule within its class, the rest filled by repeated application of the rule; with every candidate's exact specification, bank hash and counts, the selection rule, the nested protocol and its reduced path, and the cost reservation. The R2 review sees the menu, the manifests, the nomination and pool rules, the regenerated banks, the checks and the cost projection, before P2.7. At most one submission: the candidate the frozen rule ranks first, with an R1 review of its file before Sam uploads it.

## Open questions

1. The exact Kaggle team name as Kaggle displays it, for the report and the Piazza note (Sam).
2. The assignment asks for groups of two to four (`../../Project_1.pdf` section 2); the course-group arrangement is not recorded (Sam).

## Status

- 2026-10-29: Sam's go-ahead for planning phase 2 (header). Codex consultation round E (an independent design, after Claude's positions were written down) and round F (a critique of the first draft; its changes adopted, listed in the consultation README). P2.0 in review.
- 2026-10-29: P2.0 approved in two rounds and committed (3e24fe6, df11816). P2.1 started on Sam's go-ahead, verbatim: "Commit and start P2.1". During P2.1 Sam asked for refinements to the experiments, budgets and ensembles of phase 2; they are a separate reviewed amendment, P2.0b. P2.1 in review.
- 2026-10-29: P2.1 approved in two rounds and committed (719a934, 4e451ec). P2.0b, the amendment of section 14, built after Codex consultation rounds I (an independent design, after Claude's positions were written down) and J (a critique of the draft); in review.
- 2026-10-29: P2.0b approved in one round and committed (ee557e2, 281a014). P2.2 started on Sam's go-ahead, verbatim: "Commit and start P2.2", and "remember to consult deeply with codex"; its execution protocol (section 15) and code built after Codex consultation rounds K (an independent design, after Claude's positions were written down) and L (a critique of the draft); at its execution-readiness review.
- 2026-10-29: P2.2's execution-readiness checkpoint approved in its second round (five findings fixed: exhausted local neighbourhoods, completion before the cutoff, bootstraps before allocations, contribution ties, whitespace).
- 2026-10-30: P2.2 executed under section 15: the initialization (H equal to the P2.1 anchor bank), waves 1 to 3, 348 configurations; no family had allowance at wave 4, about 00:00 PDT. The completion audit passes; the full refit reproduces 329 runs exactly and 19 random forest and Extra Trees runs within the tie sensitivity of Finding 013. Findings 011 to 013 and a calibration submission file; at its completion review.
- 2026-10-30: P2.2's completion review, round 3: changes requested (F6 to F9). Sam chose option B for the 19 tie-sensitive forest runs, verbatim: "Do B". Section 15.7 accepts them by an amended verification, which passes (Finding 013). Round 4 in review.
- 2026-10-30: P2.2 approved at its completion checkpoint in round 4 and committed (810d1ce, 72a822b). P2.8, the gap hunt (section 16), started on Sam's go-ahead, verbatim: "Yes, start P2.8", after Codex consultation rounds M and N and an audit of the earlier audio-only project; at its readiness review.
- 2026-10-30: P2.8's readiness checkpoint approved in its third round (six findings fixed: the forest check's run lookup, probe-bank serialization, a complete verifier, the descriptive I3 minus I2 record, fold-level forest differences, a corrected prior-project label). Executed under section 16 (C, B, A) and verified: the forest check passed, the 18 part B runs reproduce exactly, the probe's verifier passed. Findings 014 to 017; at its completion review.
- 2026-10-30: P2.8 approved at its completion checkpoint in round 5 and committed (5e2d5a0, cab0174). P2.3 started on Sam's reply, verbatim: "Okay, 2P2.3 and 2.4, and then pause after you finish." CatBoost 1.2.10 and interpret-core 0.7.8 installed with the commands P2.1 reviewed; the factories, runtime checks, score-free timing and the execution protocol of section 17 built after Codex consultation rounds O and P; at its execution-readiness review.
- 2026-10-30: P2.3's readiness checkpoint approved in its second round (one finding fixed: the handoff follows the runner's completion rule). Executed under section 17: the initialization (H equal to the P2.1 anchor bank), wave 1, two EBM refinement waves; no family had allowance at wave 4, about 07:40 PDT. 77 configurations: CatBoost's best is its anchor, 0.8306; EBM's best 0.8162. The audit passes and the full refit reproduces every run. Findings 018 to 020; at its completion review.
- 2026-10-30: P2.3 approved at its completion checkpoint in round 3; not committed (Sam's instruction: P2.3 and P2.4, then a pause). P2.4, the feature stage (section 18), built after Codex consultation rounds Q and R in an isolated copy while P2.3 ran, and brought into the project after P2.3 converged; at its readiness review.
- 2026-10-30: P2.4 readiness round 4: changes requested, F2 open; paused. The project moved to agent-workflow v1 on Sam's approval (`../../AGENTS.md`): where this plan calls for a readiness or completion review, a convergence record or a fingerprint, the review triggers of `AGENTS.md` apply instead. P2.3 committed (dca38c6); P2.4 committed as provisional (74517ab). Sam's decision for P2.4, verbatim: "Fix F2, run provisionally (Recommended)" (`../../decisions.md`). The current state is in `../../STATE.md`.
