# Phase 2: models and features

v0.2, 2026-10-29. Owner: Sam Zhou. Status: v0.1 drafted in task P2.0 on Sam's go-ahead, verbatim: "Yes, plan phase 2" and "Again, consult deeply with codex"; approved after two review rounds and committed (3e24fe6, df11816; `../../reviews/2026-10-29-p2.0-phase2-plan.md`). v0.2: the infrastructure adjustments of P2.1 (section 13), in review (`../../reviews/2026-10-29-p2.1-infrastructure.md`). Sam's later refinements to the experiments, budgets and ensembles are a separate reviewed amendment, P2.0b; until it converges, sections 5 to 11 stand as approved in P2.0. Built after Codex consultation rounds E (an independent design) and F (a critique of the first draft), verbatim in `../llm_usage/2026-10-29-phase2-consultation/` with what was adopted.

## Goal

From the reviewed F2 baselines (Findings 007 and 008) to a small frozen menu of validated candidates: tuned tree models, CatBoost, a smooth model, a bounded neural challenger, cross-fitted date encodings and small blends, every one scored on the saved grouped folds; then a restricted nested evaluation of the frozen selection rule. Phase 3 then only reproduces, validates, uploads and selects (`modeling_strategy.md` section 6).

## Exit criteria

1. Tasks P2.1 to P2.4, P2.6 and P2.7 converged; P2.5 converged, or its omission recorded with the reason. P2.3 may converge with a reviewed incompatibility finding and CatBoost omitted.
2. Every menu candidate has a complete bank on the 3,863 eligible rows, the saved primary folds and all three split seeds.
3. Every learned transformation is fitted only on its permitted training population, shown by self-checks.
4. Phase 2 banks, logs, selection manifests and submission files verify without writing.
5. The phase 1 commands still pass unchanged at the phase 2 head: `scripts.run_baselines --protocol production --verify`, `--protocol legacy --verify`, `scripts.check_foundation`, `scripts.prepare_submission --verify`.
6. The restricted nested evaluation is complete, or its declared reduced path (section 8) is documented.
7. The menu, the selection rule, the deployment rule, the source revision, the package locks and the hashes are frozen, and the results are reviewed findings.

## Tasks

Each task is one reviewed unit, started only after the one it depends on has converged. Before each task is built, Codex is consulted on its design or code (Sam's instruction above); the formal reviewer is always a fresh Codex session. Times through Saturday are PDT; from Sunday 2026-11-01, when daylight saving time ends, PST. Target times are converged completion times, review included.

| Task | Deliverables | Depends on | Target | Record |
|---|---|---|---|---|
| P2.0 | This plan; the strategy amendments of section 12; the consultation evidence; the P1.3 upload and public score; status lines in `../README.md`, `../project_plan.md`, `phase1-data-baseline.md`, `../../submissions/README.md`; the integration index row | P1.3 | Thursday midday | `../../reviews/2026-10-29-p2.0-phase2-plan.md` |
| P2.1 | The infrastructure of section 4; the installation locks of section 4.9 (reviewed here, executed in P2.3 and P2.5); the package-independent self-checks; the LightGBM F2 baseline reproduced through the new engine | P2.0 | Thursday evening | record named when opened |
| P2.2 | Core: the LightGBM and random-forest searches, spline logistic regression, the equal LightGBM and random-forest blend. Optional controls: Extra Trees, HistGB, XGBoost. Findings | P2.1 | Friday morning | as above |
| P2.3 | The CatBoost installation and its runtime checks; a timing pilot; the CatBoost search; optionally EBM (at most four configurations); findings | P2.1, P2.2 | Friday midday | as above |
| P2.4 | The feature blocks and training weights of section 6, date count and date target encoding first; paired comparisons; findings | P2.2, P2.3 | Friday evening | as above |
| P2.5 | The neural challenger (section 5.5): TabM, its installation and runtime checks, a timing and repeatability pilot, at most two configurations; findings | P2.1 | Saturday morning; 90 minutes of author time | as above |
| P2.6 | Ensembles and estimator-seed averaging (section 7); the frozen menu and selection manifest (section 11); the nested-evaluation manifest; at most one submission | P2.2 to P2.4; P2.5 if done | Saturday; menu frozen by 18:00 PDT | as above |
| P2.7 | The restricted nested evaluation and the deployment comparison (section 8); the final evidence table; the frozen deployment manifests; findings | P2.6 | Sunday; first review request by 14:00 PST, final freeze by 18:00 PST | as above |

Cut order if a task slips: the optional controls (Extra Trees, HistGB, XGBoost) and EBM; the neural challenger, if its setup exceeds 30 minutes or its projected confirmation and nested-evaluation cost is unreasonable; the later feature blocks (interactions, cyclic calendar, era-relative audio); local refinements. The incomplete-row ablation is outside the default scope. Never cut: grouped cross-fitting, the phase 1 verification, submission validation, the restricted nested evaluation.

## Decisions

- Sam uploads every submission (phase 1 decision, unchanged).
- Interpretation (strategy section 5.6) and the stress tests (strategy section 4, step 11) move to phase 4, following Sam's decision of 2026-10-29, verbatim: "No, from now on, don't worry about the collab or the report itself until the very end. Focus on development." Phase 2 keeps what they will need (section 9). Sam may move them back.
- Kaggle membership: Sam alone, as Sam reported on 2026-10-29, verbatim: "Kaggle team is my name . I am the only member: Weilong (Sam) Zhou." The course-group arrangement is not recorded.
- One environment, `.venv`, extended only by the reviewed route locks of section 4.9; `requirements.txt` stays unchanged, so the phase 1 configuration hashes, which include its sha256, stay valid. A package that cannot be added without changing an installed version is dropped in phase 2, not given its own environment. `../../CLAUDE.md` is unchanged; `../README.md` lists the lock files once they exist.
- No pretrained model of any kind and nothing fitted on test rows (strategy section 3), unchanged.

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

## 5. Search and tuning (P2.2, P2.3, P2.5)

### 5.1 Budget and protocol

The initial search has exactly 104 classical configurations (80 core, 24 optional) and at most two neural configurations. Measured cost: 6 to 8 s for a 15-fit run of the reviewed LightGBM and random-forest baselines. Every other family is timed on a representative configuration first. A family whose 15-fit run costs at most about a minute runs every configuration on all three split seeds. Above that, declared before its scores are seen: screen on split seed 155, then confirm the best three plus an anchor on all three seeds. Configurations come from a seeded sampler; one trial at a time, four model threads.

| Family | Task | Configurations | Space |
|---|---|---|---|
| LightGBM | P2.2, core | 32, the phase 1 anchor included | `num_leaves` {7, 15, 31}; `min_child_samples` {10, 20, 40, 80, 120}; learning rate {0.02, 0.05}; trees {300, 600, 1200}; L1 {0, 0.1, 1, 5}; L2 {0, 0.1, 1, 10, 30}; column fraction {0.7, 0.85, 1}; row fraction {0.8, 1}, with `subsample_freq=1` when below 1 |
| Random forest | P2.2, core | 18, the phase 1 anchor included | 500 trees; `bootstrap=True`; leaf size {1, 2, 3, 5, 8, 12, 20}; `max_features` {"sqrt", 0.5, 0.75, 1.0}; depth {None, 12, 24}; `max_samples` {0.7, 1.0} |
| Spline logistic | P2.2, core | 6 | logistic regression with `spline` preprocessing: knots {4, 7}, degree 3; C {0.1, 1, 10}; section 6.2 |
| CatBoost | P2.3, core | 24, anchors included | depth 3 to 7; L2 {3, 10, 30}; random strength {0, 0.5, 1, 2}; learning rate {0.03, 0.06}; iterations {500, 1000, 1600}; Plain and Ordered boosting; symmetric trees |
| Extra Trees | P2.2, optional | 6 | 500 trees; no bootstrap; leaf size {2, 5, 12}; feature fraction {0.5, 1} |
| XGBoost | P2.2, optional | 8 | `hist`; depth 2 to 4; child weight {3, 10, 30}; L2 {1, 10, 30}; learning rate {0.03, 0.07}; trees {500, 1000}; row and column sampling 0.8 |
| HistGB | P2.2, optional | 6 | 300 iterations; leaves {7, 15, 31}; minimum leaf {20, 50}; learning rate 0.05; `early_stopping=False` |
| EBM | P2.3, optional | 4 | main effects or three declared pairs (year and loudness, year and acousticness, energy and loudness); fixed rounds {500, 1500}; `validation_size=0`, `outer_bags=1`, `inner_bags=0`; four threads |

Feature experiments have their own cap (section 6.1).

### 5.2 Iteration counts and refinement

- Boosted trees use fixed iteration counts as hyperparameters, with no early stopping: no outer or inner `eval_set`, CatBoost's `use_best_model` off.
- Local refinement: at most two families, at most 12 trials each, any extension of the iteration grid counted inside those 12. A family qualifies only if its best confirmed configuration meets the promotion rule (section 5.6) against the LightGBM F2 anchor.

### 5.3 Model settings

- LightGBM keeps the phase 1 determinism settings; no `Dataset` is reused across fits, so each fit learns its own bins.
- CatBoost: `thread_count=4`, an explicit seed, `allow_writing_files=False`, CPU, an explicit boosting type, no automatic class weights, numeric F2. Exact-date strings are not passed to its native categorical statistics, which do not keep repeated tracks together. Low-cardinality musical fields, if categorical, are forced to one-hot.
- Library-internal validation is disabled or given explicit grouped inner splits (strategy section 4, step 5).

### 5.4 The anchor

The anchor is the phase 1 recipe exactly as submitted: LightGBM on F2, the phase 1 parameters, estimator seed 0, the equal average of its 15 fold models (`production_lightgbm_F2_9c37e77d549e`). A seed-averaged version of it is a separate candidate.

### 5.5 Neural challenger (P2.5)

- TabM only: two blocks, width 128, 16 members, dropout 0.1, and at most one alternative configuration. Member losses are trained separately; member probabilities are averaged at inference.
- Before P2.5 runs, its reviewed configuration fixes the inner split seed, maximum epochs, patience, stopping metric, epoch tie rule, optimizer, learning rate, batch size and the deterministic settings.
- Epoch selection: an inner group-safe split of the outer training part. Imputation, scaling, numerical bins and any learned transformation are fitted on the inner training rows only; the stopping rows are transformed without fitting. After the epoch count is chosen, the stopping-stage model and preprocessing are discarded; preprocessing is refitted on the whole outer training part, the network reinitialized and trained for the chosen count.
- CPU, no compilation, no mixed precision, deterministic seeds, no data-loader workers. Screened on split seed 155 (strategy section 4, step 3); confirmed on all three only if it is competitive or adds to a blend. Stopped if setup exceeds 30 minutes.

### 5.6 Promotion rule

From strategy section 4, step 9: a change earns further engineering time when its unrounded mean fold AUC improves by at least 0.003 against its named comparator and its mean AUC difference is positive on each of the three repeats. The paired bootstrap interval and the repeat differences are reported with it. This is an engineering threshold, not a significance test. Smaller gains are reported as observed gains and may stay as cheap candidates, but they authorize no further search.

### 5.7 Selection optimism

Every searched maximum is optimistic: the configurations are chosen on the same folds that score them. Three repeats stabilize the estimate; they are not three independent datasets, and the bootstrap intervals do not correct the search. Every attempted configuration is logged, failures included. The restricted nested evaluation (section 8) covers the frozen menu, not the whole exploratory search.

## 6. Features (P2.4)

### 6.1 Protocol

Tree-oriented blocks are first compared against the unchanged reviewed LightGBM F2 recipe; smooth and neural preprocessing is compared within its own family. A confirmed feature effect compares otherwise identical recipes. Only promoted blocks receive further tuning or combinations. Cap: at most 9 screening runs on the LightGBM F2 recipe (the blocks of section 6.3 and the training weights) and at most 8 confirmation runs of promoted blocks on at most two tuned recipes. Each block has its own name; no single F3 holding every idea.

### 6.2 Smooth and neural preprocessing

Spline logistic regression: cubic splines on year and the ten continuous audio features (speechiness, danceability, duration_ms, energy, instrumentalness, valence, tempo, loudness, acousticness, liveness), after the phase 1 log transforms; one-hot key, mode and time signature, never splined; no further date parts, which added +0.0008 for logistic regression (Finding 008). Median imputation before the splines, scaling after, all fitted in the fitting population. TabM gets the same continuous columns, the F2 date parts with missing indicators, and the one-hot categories.

### 6.3 Blocks, in order

| Order | Block | Content |
|---|---|---|
| 1 | `F2_date_count` | the number of distinct fitting groups per raw date string: a training row gets log1p(count[date] - 1), its own group left out; an outer validation or test row gets log1p(count.get(date, 0)), zero for an unseen date |
| 2 | `F2_date_te10`, `F2_date_te30` | F2 plus the cross-fitted exact-date target encoding of section 6.4, smoothing 10 and 30; each with and without the count |
| 3 | `F2_interactions` | (year - 2000) x loudness; (year - 2000) x acousticness; energy x loudness; energy x danceability; acousticness x instrumentalness; tempo x duration_ms / 60000 |
| 4 | `F2_cyclic` | for each phase p, the pair sin(2πp) and cos(2πp), with p the month phase (month - 1) / 12, the weekday phase weekday / 7 and the day-of-year phase (day_of_year - 1) / days in that year; both missing where the component is unobserved |
| 5 | `F2_era_relative` | loudness and acousticness minus their era mean over the five eras of `songpop.splits`, each era mean shrunk toward the fitting-population mean with 20 pseudo-observations; an era absent from the fitting population uses the fitting mean |
| 6 | Training weights | inverse-group-size loss weights, recomputed and normalized to mean one in every actual fitting population; evaluation stays row-weighted; target-encoder counts stay row-weighted |

These definitions are proposed defaults, not established optimal choices.

### 6.4 Exact-date target encoding

For each outer training part:

1. Build five inner `StratifiedGroupKFold` folds from its rows, labels and primary groups only, with a deterministic seed; the splits are explicit local positional indices, and group-disjoint.
2. Pass these (train, test) index pairs as `cv` to scikit-learn 1.9.1's `TargetEncoder` (it accepts an iterable of splits), `target_type="binary"`, the fixed smoothing m. The encoded value of date d is (sum of labels + m x prior) / (rows + m), with row counts and the prior from the permitted fitting partition.
3. `fit_transform` encodes the outer training rows; the fitted encoder's `transform` encodes the outer validation and test rows.
4. Raw date strings are the categories, so "1979", "1979-06" and full dates stay distinct; an unseen date gets the fitting prior.

A global out-of-fold encoding is never computed once and reused inside another validation level. The feature is an exact-date association: album identity is unknown, and different releases can share a date. Round E's unreviewed probe reported validation coverage of 69.87%, 68.65% and 69.35% by repeat, against 623 of 863 test rows (72.2%) sharing a date with the full training set. This motivates testing the feature; it does not establish equal support or album composition. P2.4 recomputes coverage and distinct-group support under the actual fitting dictionaries, including those of the fold models used for deployment. The primary folds are not regrouped by date.

## 7. Ensembles and deployment (P2.6)

- Members: the best confirmed LightGBM, random forest, CatBoost and at most one other competitive recipe.
- Declared blends: equal LightGBM and random forest; equal LightGBM and CatBoost; the equal average of the three leading complementary recipes; at most two declared 75/25 alternatives. Not every subset and weight.
- Probabilities are averaged; no learned stacker, no calibration step (strategy section 5.5).
- Finalists other than the anchor use estimator seeds 0, 1 and 2, averaged within each exact base recipe. A three-seed recipe deploys 45 fitted models (three seeds at each of the 15 split and fold positions); the anchor deploys 15.
- Deployment default: the equal average of the fold models, as in phase 1. A full refit on all eligible rows is a separate deployment artifact in `artifacts/phase2/deploy/`, never written into a bank's `test_mean`. The two are compared only in section 8.

## 8. Restricted nested evaluation (P2.7)

- Menu: at most five base recipes and eight predictor candidates, frozen in P2.6 (section 11), the anchor included. A blend refers to base recipes and adds no fits.
- Outer: three `StratifiedGroupKFold` folds on the eligible rows and primary groups, split seed 20261101, saved in `artifacts/phase2/splits/`.
- Inner: five grouped folds with split seeds 155, 2026 and 1029, rebuilt within each outer training part. Every preprocessing step and encoding is rebuilt at every level. The global phase 2 banks are never reused inside it.
- In each outer fold: fit each base recipe on the inner training parts; build fresh inner out-of-fold predictions and the declared blends; `select_candidate(inner_banks, menu)` applies section 11 steps 1 to 3 only, and receives no outer predictions or labels, no public score and no global bank. Save, for the outer validation rows: the selected candidate's fold-average predictions, its members refitted on the whole outer training part, and the anchor's fold-average predictions; with the selected recipe, fitting ids, seeds and decisions.
- After all three outer folds, a separate decision function applies, in this order:
  1. Fallback: compare the selected fold averages with the anchor's fold averages. If the selected procedure loses at least 0.003 mean outer AUC and loses on all three outer folds, the anchor is recommended with its original deployment, and the disagreement is kept in the evidence.
  2. Deployment, only if the fallback did not trigger: fold averaging stays unless the full refit of the selected procedure improves the paired outer mean by at least 0.003 with a positive difference on all three outer folds. A refit advantage measured for a selected candidate is never transferred to the anchor.
- Once the outer results have chosen the fallback or the deployment, they are no longer an untouched estimate of what they chose.
- Cost: at most 675 inner `fit_recipe` calls (3 outer folds x 15 inner fits x 15 recipe-seed pairs, for five recipes with three seeds each) and 45 outer refit calls. The projection, from measured timings before the freeze, includes internal stopping runs, preprocessing, encoding, prediction and verification; an expensive challenger is removed from the menu before the freeze if needed.
- Reported: the three outer fold AUCs, their mean, the pooled outer out-of-fold AUC, the selected candidates, and the paired differences against the anchor. A group bootstrap of the outer predictions only with the usual conditional caveat.
- Reduced path, declared now: before any outer result is inspected, a declared runtime limit may switch to a fixed-candidate check: the candidate ordinary CV selects and the anchor, each fully refitted in the three outer folds, with no inner selection and no deployment comparison. It is labeled a fixed-candidate full-refit check, not a nested selection estimate, and it cannot trigger the fallback or the deployment rule; the default deployment stays.
- This evaluates a restricted selection procedure whose menu was informed by earlier exploration; it does not remove that adaptation, and part of any drop from ordinary CV comes from smaller training sets.

## 9. Kept for phase 4

No report text, Colab code, SHAP analysis, importance ranking or explanatory plot in phase 2. Phase 2 keeps: predictors that accept raw source columns; feature lineage to the original columns; a reproducible way to rebuild or load every selected fold predictor, its encoders and preprocessing; training and evaluation ids and hashes; the complete search and selection records; the consultation evidence for the LLM Usage section. Permuting the date later must apply the already fitted date map, never learn a new one from evaluation labels.

## 10. Submissions

Each is a gated action, reviewed with its file hash and purpose inside its task; its purpose is written down before Sam reports the score. The public score may prompt a check for an implementation or distribution mismatch; it never drives a weight grid, a feature search or a swap between nearly tied models.

| When | File | Purpose |
|---|---|---|
| After P2.2 or P2.3, only if a recipe meets the promotion rule against the anchor | The best confirmed single recipe | Exercise the phase 2 fitting and submission path |
| After P2.6 | The frozen blend | Exercise ensemble deployment |
| Only if warranted | A different competitive frozen candidate, or a deployment comparison already justified locally | Stated when proposed |

No submission is made only to fill the schedule; a file already uploaded with identical bytes is not uploaded again. With phase 3's final candidates this keeps the project near the strategy's four to six uploads.

## 11. Freeze, menu and selection rule

- By Saturday 18:00 PDT, P2.6 freezes the exact candidate recipes, parameters, feature definitions, estimator seeds, blend weights, stopping policies, each candidate's fixed `complexity_rank`, the selection rule and the nested protocol, in `configs/phase2/menu.json`. P2.7 starts only after P2.6 converges.
- Sunday permits evaluation, verification and reviewed correctness fixes, and no score-driven revision of the frozen candidates. A correctness fix invalidates the affected predictions, which are regenerated before use. By 18:00 PST, P2.7 has converged and the deployment manifests are frozen.
- Monday: reproduce, validate, prepare and upload the candidates, select (strategy section 6, P3.1 and P3.2).

Selection rule, declared before the menu is evaluated; `select_candidate` implements steps 1 to 3:

1. Rank candidates by unrounded mean fold AUC over the 15 folds.
2. Candidates within 0.001 of the maximum are practically tied (a concrete form of the strategy's "ties go to the simpler recipe").
3. Break ties by fewer base recipes, then fewer supervised feature transformations, then the lower fixed `complexity_rank` stored in the menu, then recipe id. Measured wall-clock time never breaks a tie.
4. The fallback and deployment decisions of section 8, after the nested evaluation.

Phase 3 proposes A, the strongest stable candidate under this rule, and B, a competitive alternative with evidenced complementary errors, normally within 0.005 mean fold AUC of A; if no diverse candidate qualifies, the closest strong alternative. Both are engineering bounds, not uncertainty calculations. Sam chooses.

## 12. Strategy amendments

P2.0 was the reviewed task for this coherent amendment of `modeling_strategy.md`; each row is edited there, with a history note. P2.0 converged and was committed (3e24fe6).

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

## Open questions

1. The exact Kaggle team name as Kaggle displays it, for the report and the Piazza note (Sam).
2. The assignment asks for groups of two to four (`../../Project_1.pdf` section 2); the course-group arrangement is not recorded (Sam).

## Status

- 2026-10-29: Sam's go-ahead for planning phase 2 (header). Codex consultation round E (an independent design, after Claude's positions were written down) and round F (a critique of the first draft; its changes adopted, listed in the consultation README). P2.0 in review.
- 2026-10-29: P2.0 approved in two rounds and committed (3e24fe6, df11816). P2.1 started on Sam's go-ahead, verbatim: "Commit and start P2.1". During P2.1 Sam asked for refinements to the experiments, budgets and ensembles of phase 2; they are a separate reviewed amendment, P2.0b. P2.1 in review.
