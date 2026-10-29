Claude, I would build P2.1 as a working experiment path, with tested interfaces for later models. Avoid building the complete Sunday evaluation now.

I read the approved plan and phase 1 implementation. My new read-only probes are **unreviewed consultation evidence**:

| Check | Result |
|---|---|
| Rebuild F2 separately within all 15 LightGBM fits | OOF, per-fold test predictions, and test mean matched the approved bank bit-for-bit |
| Time for that probe | 12.353 seconds in this session |
| Explicit-split `TargetEncoder`, compared with manual smoothing | Maximum difference 0 |
| Change only an inner held-out fold’s labels, keeping splits fixed | Its cross-fitted encodings changed by 0 |
| Unseen category in the synthetic example | Encoding 0.5, equal to the fitting prior |
| Environment | pip 26.2.2; `pip check` found no broken requirements |

I did not resolve or download optional packages. This sandbox does not permit the scratch writes needed for that verification.

1. **Define P2.1’s implementation boundary explicitly.**

   I recommend this ownership table:

   | Component | P2.1 delivers | Later task |
   |---|---|---|
   | Recipes, identities, sources, banks, logs | Complete implementation | Extensions reviewed when introduced |
   | CV engine | Complete for installed classical models and arbitrary supplied fold counts | Neural training adapter in P2.5 |
   | Features | F0/F1/F2 adapter, block interface, date count, grouped target encoding, training weights | Additional feature blocks and experiments in P2.4 |
   | Models | LightGBM, RF, existing logistic; preprocessing interface | Spline implementation in P2.2; optional-package factories in their tasks |
   | Ensembles | Probability blending and seed averaging | Actual candidate combinations in P2.6 |
   | Selection | Implement the small pure selector now | Actual menu and complexity ranks in P2.6 |
   | Nested evaluation | Population-containment and inner-split helpers, tested with synthetic data | Complete orchestration and deployment comparison in P2.7 |
   | Submission path | Fold-average submissions, one recipe or declared blend | Full-refit artifacts in P2.7 |
   | Installations | Protected inventory, resolutions, locks, validators, exact commands | Execute in P2.3/P2.5 |

   Implementing the selector now is inexpensive and avoids an unnecessary interface gap. Deferring the TabM wrapper is already consistent with its assigned task.

   Record this table in the P2.1 review request and a phase-plan implementation note. Say explicitly:

   > Section 4 defines the infrastructure contract. P2.1 implements the common mechanisms and the capabilities listed here. Named later-task implementations remain unavailable until implemented and reviewed. Unsupported capabilities raise an explicit error.

   Do not label a stub as an implemented nested evaluator. If the formal reviewer reads §4 as requiring the entire evaluator in P2.1, this table makes the proposed staging visible for resolution.

2. **Use a strict recipe schema in `recipes.py`.**

   Keep human labels separate from computational identity. For example:

   ```json
   {
     "label": "phase1_lightgbm_F2_anchor",
     "spec": {
       "schema": "recipe_v2",
       "model": {
         "family": "lightgbm",
         "params": {
           "n_estimators": 600,
           "learning_rate": 0.02,
           "num_leaves": 15,
           "min_child_samples": 20,
           "subsample": 0.8,
           "subsample_freq": 1,
           "colsample_bytree": 0.8
         }
       },
       "features": {
         "base": "F2",
         "blocks": []
       },
       "preprocessing": {
         "kind": "native"
       },
       "training_weights": {
         "kind": "none"
       },
       "estimator_seeds": [0],
       "iteration_policy": {
         "kind": "fixed"
       },
       "deployment": {
         "kind": "fold_average"
       }
     }
   }
   ```

   Public functions:

   ```python
   load_recipe(path: Path) -> Recipe
   validate_recipe(spec: dict) -> Recipe
   canonical_json(value: object) -> bytes
   recipe_sha256(recipe: Recipe) -> str
   with_estimator_seeds(recipe: Recipe, seeds: tuple[int, ...]) -> Recipe
   base_recipe_sha256(recipe: Recipe) -> str
   ```

   Rules:

   - Reject duplicate JSON keys, unknown fields, nonfinite numbers, invalid types, and unsupported combinations.
   - Normalize numeric types by schema before hashing. Reject booleans where an integer or float is required.
   - Canonicalize using sorted keys, compact separators, UTF-8, and `allow_nan=False`.
   - Preserve ordered lists such as feature blocks. Do not sort them silently.
   - Expand declared schema defaults before hashing.
   - Hash `spec`, excluding the human label and generated IDs.
   - Use an ID such as `lightgbm_<first16hex>`, retaining the full hash in metadata.
   - `base_recipe_sha256` excludes only `estimator_seeds`. It is the compatibility key for seed averaging.
   - Reserve estimator randomness and thread parameters for the execution policy. Reject conflicting values inside `model.params`.

   A recipe parameter dictionary is constructor input. A resolved parameter description is evidence. Do not pass the output of phase 1’s `describe()` back into constructors, because its `"nan"` representation is descriptive.

3. **Separate feature construction from model-specific preprocessing.**

   In `features.py`, use:

   ```python
   class FeatureBuilder:
       def fit_transform_train(
           self, rows, y, groups, context
       ) -> pd.DataFrame: ...

       def transform(self, rows) -> pd.DataFrame: ...

       def feature_lineage(self) -> dict[str, list[str]]: ...
   ```

   Each block follows the same interface. Stateless blocks can implement `fit_transform_train` by calling their deterministic transform.

   Important contracts:

   - Input rows contain only permitted source columns: the 13 audio columns and raw date string for current recipes.
   - Training row IDs live in the index and context. They never become predictors.
   - Labels, URI, URLs, test IDs, and provenance columns are not passed into model feature construction.
   - Outputs preserve input row order and index.
   - Base F0/F1/F2 construction calls the existing `songpop.features.make_features`.
   - Blocks append columns in recipe order and reject duplicate output names.
   - Every call creates fresh fitted block instances.
   - `transform` never accepts labels, updates counts, discovers new categories, or refits anything.

   Training representations can legitimately differ from `transform(training_rows)`. That is essential for target encoding and own-group-excluded counts. Fit the classifier on the returned `fit_transform_train` matrix, never on a subsequent `transform` of those rows.

   Keep imputation, scaling, splines, and categorical preprocessing in the model’s sklearn pipeline. The feature builder handles date encodings and other explicit feature blocks. This avoids implementing two competing preprocessing systems.

4. **Implement exact-date encoders now, with explicit local splits.**

   In `encoders.py`:

   ```python
   make_inner_splits(
       row_ids, y, groups, *, n_splits, seed
   ) -> tuple[tuple[np.ndarray, np.ndarray], ...]

   class DateTargetEncoderBlock: ...
   class DateGroupCountBlock: ...
   ```

   A target-encoding block should contain its complete split policy:

   ```json
   {
     "kind": "date_target",
     "smooth": 10.0,
     "inner_cv": {
       "kind": "stratified_group_kfold",
       "n_splits": 5,
       "shuffle": true,
       "seed": 155010
     }
   }
   ```

   `155010` is my proposed default, not an existing project requirement. Keep this seed independent of estimator seeds so averaging model seeds does not silently change the encodings.

   Construct:

   ```python
   encoder = TargetEncoder(
       categories="auto",
       target_type="binary",
       smooth=10.0,
       cv=explicit_splits,
   )
   ```

   Use a materialized tuple of local positional index pairs. Validate group separation, both classes in each fitting portion, disjoint training/validation indices, and exactly one validation appearance per row.

   Specific sklearn 1.9.1 points:

   - Do not pass deprecated `shuffle` or `random_state` arguments to `TargetEncoder`; put them on `StratifiedGroupKFold`.
   - Use numeric labels 0 and 1 and assert `classes_ == [0, 1]`.
   - Preserve dates as a two-dimensional object array of strings. Never parse the categories into dates or integers.
   - `smooth=10.0` means ten pseudo-observations at the fitting prior. It is not a ten-group prior.
   - Counts and label sums remain row-weighted.
   - `fit_transform` returns rows in original input order. Restore the DataFrame index explicitly.
   - The stored full-fitting map is used by `transform` for external rows.
   - An unseen date receives the applicable fitting prior.
   - Do not use `smooth="auto"` for these named experiments.

   For date count:

   ```python
   counts = (
       pd.DataFrame({"date": dates, "group": groups})
       .drop_duplicates(["date", "group"])
       .groupby("date")
       .size()
   )
   ```

   Training values are `log1p(counts[date] - 1)`. External values are `log1p(counts.get(date, 0))`. Assert that training counts before subtraction are at least one.

   Duplicate groups, including mixed-label groups, must be excluded together in target encoding. Counting distinct `(date, group)` pairs also makes the count implementation correct on synthetic groups spanning multiple dates.

5. **Make `fit_recipe` own every learned operation.**

   In `engine.py`:

   ```python
   @dataclass(frozen=True)
   class FitContext:
       row_ids: tuple[int, ...]
       allowed_row_ids: frozenset[int]
       split_seed: int
       fold_index: int
       estimator_seed: int
       scope_path: tuple[str, ...]
       inner_splits: tuple | None

   def fit_recipe(
       recipe: Recipe,
       rows: pd.DataFrame,
       y: np.ndarray,
       groups: np.ndarray,
       context: FitContext,
   ) -> FittedRecipe: ...
   ```

   `allowed_row_ids` contains IDs only, never an enclosing dataset or its labels. It lets nested callers enforce containment.

   Execution order:

   1. Validate alignment, labels, row IDs, groups, permitted population, and configuration.
   2. Create fresh feature blocks.
   3. Produce the training matrix through `fit_transform_train`.
   4. Create a fresh estimator or pipeline.
   5. Compute optional loss weights from this actual fitting population.
   6. Fit once, without outer validation data.
   7. Return the predictor and a fit receipt.

   For inverse-group-size weighting:

   ```python
   w_i = 1.0 / size_of_group_i_within_this_fit
   w /= w.mean()
   ```

   Route weights explicitly to the final estimator. For a pipeline named `model`, use `model__sample_weight`. Do not enable global sklearn metadata routing merely to accomplish this.

   For the anchor, omit the `sample_weight` argument entirely. Do not replace “none” with an array of ones.

   `FittedRecipe` should expose:

   ```python
   predict_high(raw_rows) -> np.ndarray
   model_inputs(raw_rows) -> ModelInputs
   predict_with_inputs(raw_rows) -> PredictionBatch
   ```

   `predict_with_inputs` obtains the actual matrix entering the final estimator and predicts from that same matrix. This supports trustworthy duplicate checks. For sklearn pipelines, transform through the fitted preprocessing steps, then call the existing `songpop.models.predict_high` on the final estimator.

   Optional-package imports occur only inside their factories. P2.1 must never import Torch or CatBoost just to load a recipe or inspect a bank.

6. **Keep atomic CV runs simple, and build seed averages from their banks.**

   Implement:

   ```python
   run_recipe_cv(
       recipe, data_view, folds, split_seeds
   ) -> BankBundle
   ```

   I recommend one estimator seed per atomic run. When a recipe requests `[0, 1, 2]`, the runner creates three derived single-seed recipes, obtains their banks, then calls `average_seed_banks`. The aggregate references those immutable parents.

   This preserves each fitted member’s test predictions without adding another prediction axis to the core bank.

   The runner must:

   - Load and validate the saved production split file, including its approved hash.
   - Select repeat rows by **seed value**, not by taking the first `R` rows.
   - Canonicalize requested production seeds to their official order.
   - Preserve permanent row IDs and use positional indices only for slicing aligned arrays.
   - Check every eligible row receives exactly one held-out prediction per repeat.
   - Call `fit_recipe` with training rows only.
   - Store predictions immediately in their original positions.
   - Compute each fold AUC from predictions.
   - Use the existing mean over repeat and fold axes for `test_mean`.
   - Validate inputs before creating an estimator.

   Add a generic fold-view constructor for later nested work. It should accept an explicit row population and fold matrix rather than assume the global 3,863-row population.

   **Anchor acceptance:** compare all core arrays against the approved phase 1 bank. IDs, labels, groups, folds, seeds, predictions, and AUCs should match exactly. Metadata and NPZ bytes will differ.

   Preserve:

   - The 20-column F2 order and float64 representation.
   - NaNs and partial-date rules.
   - Training row order.
   - All phase 1 LightGBM constructor settings.
   - `random_state=0`, `n_jobs=4`, `deterministic=True`, `force_col_wise=True`.
   - No extra seed overrides, imputation, weights, early stopping, or global `Dataset`.

   My probe confirms that regenerating these stateless features within each fold does not itself break equality.

7. **Define bank metadata and run identity separately.**

   Reuse `PredictionBank` for the core arrays. Add a v2 loader:

   ```python
   load_bank_v2(
       source,
       *,
       expected_n_splits: int,
       expected_sha256: str | None = None,
   ) -> BankBundle

   validate_v2_metadata(bank) -> None
   compare_banks_v2(actual, expected) -> list[str]
   save_bank_v2(bundle, destination) -> ArtifactRef
   ```

   Load with `allow_pickle=False`. Then call:

   ```python
   validate_bank(
       bank,
       n_splits=expected_n_splits,
       require_metadata=False,
   )
   validate_v2_metadata(bank)
   ```

   I would use these exact top-level metadata keys:

   ```text
   schema
   recipe_id
   recipe
   run_id
   identity_sha256
   identity
   prediction_kind
   feature_lineage_by_fit
   fit_records
   audit
   selection_eligible
   warnings
   execution
   ```

   The `identity` object contains:

   ```text
   schema
   recipe_sha256
   prediction_kind
   data_hashes
   population
   cv
   estimator_seeds
   resolved_parameters
   deployment
   source_manifest
   dependencies
   runtime
   parents
   ```

   Define its nested fields as follows:

   | Object | Required fields |
   |---|---|
   | `population` | Policy, evaluation-row hash/count, fitting-row hash/count, label hash, group hash, test-ID hash |
   | `cv` | Protocol, grouping, split version, split-file hash or null, fold-matrix hash, repeat count, fold count, split seeds, inner-split manifest hash |
   | `dependencies` | Python version, relevant package versions, original requirements hash, relevant route-lock hashes |
   | `runtime` | Device, model thread count, determinism policy, relevant thread environment |
   | `parents` | Ordered references containing run ID, bank path, bank hash, and weight |
   | `deployment` | Kind and exact averaging order |

   Hash canonical `identity` to obtain `run_id`. Exclude timestamps, elapsed time, absolute working directory, execution HEAD, and human descriptions from identity.

   Each fit receipt records its repeat, fold, estimator seed, fitting-ID hash, inner-split hash, model-input names, source-column lineage, and selected iteration count where applicable.

   Additional validation must check unique split seeds, one-dimensional identity arrays, exact metadata/array consistency, parent hashes, recipe hashes, and supported schemas. The v1 validator does not check every v2 invariant.

8. **Save compact member-input evidence instead of fitted models.**

   For each atomic bank, save a small companion file:

   ```text
   artifacts/phase2/bank/<run_id>.audit.npz
   ```

   Its principal array is:

   ```text
   test_feature_class: int64[R, K, T]
   ```

   For each fitted member, equal class IDs mean its actual transformed test vectors are exactly equal. Canonicalize each class to the smallest test-row position belonging to it.

   Build those classes from the matrix returned by `predict_with_inputs`:

   - Compare numeric values exactly.
   - Treat NaNs as equal.
   - Do not round values.
   - Exclude IDs and provenance.
   - Reject infinities.
   - For sparse preprocessing outputs, a temporary dense representation is reasonable at this dataset size.

   The core bank already preserves that atomic member’s prediction vector in `test_by_fold[r, k]`. Therefore no additional prediction array is needed.

   Store the audit schema, path, and SHA-256 under metadata `audit`. The audit file can contain its run ID, seeds, test IDs, and fit-coordinate arrays for alignment. It must not contain the bank hash, which would create a circular hash dependency.

   At fitting time, verify every equivalence class and all repeated URI pairs. During submission preparation, reload this evidence and recheck it against the saved member predictions.

   For a blend, recursively inspect its leaf banks. Composite equivalence is the intersection of the equivalence relations of all fitted members. Equal raw-source rows and the 11 repeated URI pairs receive separate checks.

   **Limitation to state accurately:** receipt-based submission validation does not refit models. `run_experiments --verify` refits the leaf and reconstructs the audit evidence from actual transformations.

   Serialize and validate the audit and bank in memory first. Write both exclusively, then append the completed run record last. An interrupted partial bundle is an error requiring investigation, not a completed run to reuse.

9. **Implement blending and the pure selector now.**

   In `ensembles.py`:

   ```python
   blend_banks(banks, weights) -> BankBundle
   average_seed_banks(banks) -> BankBundle
   ```

   Reject:

   - Nonfinite or negative weights.
   - A sum differing from one beyond the declared tolerance.
   - Different row IDs, labels, groups, fold matrices, split seeds, or test IDs.
   - Different prediction domains or incompatible deployment kinds.
   - Missing or changed parent artifacts.

   Do not silently normalize arbitrary weights.

   Average float64 probabilities in a deterministic parent order. Recompute fold AUCs and `test_mean`. Preserve the repeat axis.

   Seed averaging additionally requires identical `base_recipe_sha256`, distinct estimator seeds, and the exact declared seed set. A tuned LightGBM and an anchor LightGBM cannot pass that check merely because their family names match.

   In `selection.py`:

   ```python
   require_complete_candidate(bank, protocol) -> None
   select_candidate(banks, menu) -> SelectionDecision
   ```

   Implement approved §11 steps 1–3 only. Read unrounded scores from bank arrays. Reject incomplete seed coverage structurally, even when `selection_eligible=True`.

   The actual frozen menu belongs to P2.6. Use synthetic menus for P2.1 tests.

10. **Give the scripts explicit modes and append-only records.**

   Suggested CLI:

   ```text
   python -m scripts.run_experiments --recipe PATH --split-seeds 155
   python -m scripts.run_experiments --recipe PATH --split-seeds 155 2026 1029
   python -m scripts.run_experiments --search PATH
   python -m scripts.run_experiments --compare RUN_A RUN_B
   python -m scripts.run_experiments --verify --run-id RUN_ID
   ```

   Use mutually exclusive modes. `--verify` must not create directories, append logs, rewrite manifests, or install anything.

   Search behavior:

   1. Validate the space, sampler seed, budget, and anchors.
   2. Generate and deduplicate the complete ordered list.
   3. Save it exclusively before the first fit.
   4. Log a planned-search record.
   5. Execute that saved list.
   6. Record every trial, including failures.

   Bound duplicate resampling so an overlarge requested budget cannot loop forever. Log a failure before exiting. Do not treat programming errors as harmless unsuccessful configurations.

   Suggested JSONL shapes:

   | `record` | Required content |
   |---|---|
   | `run` | Schema, task, stage, run/recipe IDs, status, bank and audit references, raw metrics, warnings, execution information |
   | `search` | Search ID/hash, configuration-list path/hash, ordered recipe IDs, sampler seed, protocol, budget, status |
   | `comparison` | Both bank references, bootstrap parameters/results, repeat differences, named comparator, promotion result |
   | `selection` | Menu hash, candidate references/scores, tie set, chosen ID, rule version |
   | `submission` | Manifest hash, CSV/sidecar hashes, source banks, deployment, purpose, validation result |

   Put all of these in `artifacts/phase2/experiments.jsonl`.

   Reuse the existing bootstrap implementation. Do not use phase 1’s rounded `summarize()` output for ranking or promotion. Either add a v2 unrounded summary function or compute raw metrics directly from bank arrays.

11. **Make submission manifests explicit and verification independent of “current best.”**

   `prepare_candidate.py` should accept a manifest such as:

   ```json
   {
     "schema": "candidate_submission_v1",
     "task_id": "P2.2",
     "purpose": "Exercise the phase 2 submission path",
     "deployment": "fold_average",
     "components": [
       {
         "run_id": "run-id",
         "bank_path": "artifacts/phase2/bank/run-id.npz",
         "bank_sha256": "full-sha256",
         "weight": 1.0
       }
     ],
     "output_csv": "submissions/named-file.csv",
     "description": "Prepared description"
   }
   ```

   Functions:

   ```python
   load_selection_manifest(path) -> SubmissionManifest
   build_candidate(manifest, data, bank_loader) -> Candidate
   validate_candidate(candidate, data) -> dict
   candidate_bytes(candidate) -> bytes
   prepare_candidate(manifest, *, verify=False) -> dict
   ```

   Validate complete production banks, deployment kind, parent references, weights, member audits, raw-source equivalence, and the 11 URI pairs before generating bytes.

   Use the existing `submission_bytes` function for CSV formatting. Read the written file back, validate the read-back bytes, and record its full hash. Include the manifest and producer source hashes in the sidecar.

   In P2.1, reject `deployment="full_refit"` explicitly. Implement its distinct artifact schema in P2.7.

   P2.1 needs no new Kaggle file. Build an anchor candidate in memory and assert that its CSV bytes equal the existing P1.3 CSV. Test exclusive creation and read-back behavior using in-memory filesystem mocks.

12. **Make provenance sufficient for historical replay without changing phase 1.**

   Keep all phase 1 source files, requirements, banks, splits, and log unchanged.

   For phase 2, maintain an explicit source list covering:

   - The executing script.
   - Imported phase 2 implementation modules.
   - Imported top-level `songpop` modules.
   - Any other project code actually used.

   Include `__init__.py` files. Avoid a source list that captures only the top-level runner while omitting its implementation.

   Record the manifest before fitting and compare it again afterward. If it changed during the run, do not accept the bank.

   Relevant dependency versions enter identity. A complete installed inventory belongs in execution evidence. Installing an unrelated optional package should not by itself change an existing LightGBM recipe’s identity.

   Verification order:

   1. Resolve the requested run to its unique completed log record.
   2. Verify stored bank and audit hashes.
   3. Validate schemas and internal consistency.
   4. Check source and relevant dependency matches.
   5. Recompute the run in memory.
   6. Compare core arrays, audit partitions, stable metadata, and metrics.
   7. Verify composite parents recursively, with an in-memory cache.

   Compare saved artifact byte hashes against their recorded hashes. Do not expect a freshly serialized bank with a new timestamp to have the old file hash.

   Store a post-commit manifest-to-replay-commit association separately, for example in the review record with a machine-readable companion index. Validate the commit’s file contents against the source manifest before presenting it as a replay target. Never substitute execution HEAD for that check.

13. **Generate installation locks through joint resolution, then route-specific resolution.**

   Add a small helper such as `scripts/prepare_install_locks.py`. It should parse inventories/reports, produce constraints and locks, and verify them. It should not install packages.

   Proposed commands for the author, after creating the output directories:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pip list --format=json \
     > artifacts/phase2/install/protected_inventory.json

   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pip inspect \
     > artifacts/phase2/install/protected_inspect.json
   ```

   Normalize distribution names with `packaging.utils.canonicalize_name`. Reject duplicate normalized names. Generate `protected.constraints.txt` with every installed version pinned.

   First resolve the union:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pip install \
     --dry-run --no-cache-dir --disable-pip-version-check \
     --only-binary=:all: \
     --report artifacts/phase2/install/joint_report.json \
     -c configs/phase2/install/protected.constraints.txt \
     catboost==1.2.10 interpret-core==0.7.8 torch==2.14.1 tabm==0.0.3
   ```

   Generate `resolved.constraints.txt` containing the protected inventory plus the exact proposed additions. Then resolve each route against that same file, before installing any route:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pip install \
     --dry-run --no-cache-dir --disable-pip-version-check \
     --only-binary=:all: \
     --report artifacts/phase2/install/catboost_report.json \
     -c configs/phase2/install/resolved.constraints.txt \
     catboost==1.2.10
   ```

   Repeat with `interpret-core==0.7.8` for EBM, and `torch==2.14.1 tabm==0.0.3` for TabM.

   Pip’s report is evidence for generating a lock, not itself an installable lock. Extract each distribution’s name, version, artifact URL, and `download_info.archive_info.hashes.sha256`. Check report schema version `"1"`. [pip installation-report specification](https://pip.pypa.io/en/stable/reference/installation-report/)

   For each report:

   - Reject any proposed replacement of a protected distribution.
   - Require each proposed version to match the joint resolution.
   - Require a wheel URL and a SHA-256.
   - Check wheel name/version and compatibility using `parse_wheel_filename` and `packaging.tags.sys_tags()`.
   - Check shared additions agree in version and wheel hash.
   - Preserve the complete route report’s additive closure, including shared dependencies. Do not remove shared dependencies because another route might install first.

   I favor exact wheel URLs in each route lock:

   ```text
   --only-binary=:all:
   --no-index

   distribution-name @ https://.../exact-wheel.whl --hash=sha256:...
   ```

   The real URLs and hashes must come from the actual reports. The requirements format supports these options and direct references. [pip requirements format](https://pip.pypa.io/en/stable/reference/requirements-file-format/)

   Execution remains the approved command:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pip install \
     --no-deps --require-hashes \
     -r configs/phase2/install/catboost.txt
   ```

   Before execution, check the protected inventory plus permitted prior additions. After execution, require precisely the expected inventory change, run `pip check`, then the route’s runtime checks.

   Important pitfalls:

   - Do not use `--ignore-installed` when constructing an additive closure.
   - Do not omit transitive additions because `--no-deps` will not fetch them.
   - Do not include the unhashed original `requirements.txt` in a hash-enforced route lock.
   - Do not permit sdists or a source-build fallback.
   - Do not reject `universal2` by filename intuition; use supported wheel tags.
   - These are Mac/Python-specific additive locks, not clean-environment or Colab lock files.
   - Hash mode does not establish the provenance of an already-installed distribution. Preserve the protected inventory and prior installation evidence.

   Hash checking requires pinned, hashed inputs, and binary-only installation avoids source-build behavior. [pip secure-install documentation](https://pip.pypa.io/en/stable/topics/secure-installs/)

   Save the reports, constraints, normalized inventories, lock-generation command, pip/Python/platform information, route manifests, and hashes. If joint resolution fails, record the incompatible route accurately. Do not invent a successful lock or change protected versions.

14. **Make the self-checks prove the intended property.**

   `check_phase2.py` should have a `main()` guard, run in memory, report every failure, and return nonzero if any required check fails.

   | Check | Non-vacuous construction |
   |---|---|
   | Recipe identity | Reordering JSON keys preserves the hash; changing a parameter changes it; malformed values and duplicate keys fail |
   | Seed subsets | Request seed 2026 alone and verify it uses the saved second repeat, not the first |
   | Outer-label isolation | Hold folds fixed, change one fold’s held-out labels, and compare predictions for that fold and its test predictions |
   | Scope containment | Supply one fitting row outside the permitted population and require rejection before estimator creation |
   | Target encoding | Mixed-label repeated groups, shared dates, singleton dates, explicit splits, and manual expected numbers |
   | Target-encoding isolation | Change only an inner held-out fold’s labels; compare only that fold’s encodings |
   | Count encoding | Two groups share a date, one group repeats; verify group counts, subtraction, and unseen zero |
   | Preprocessing isolation | Put an extreme value only in a stopping/validation subset; fitted imputer and scaler statistics must remain unchanged |
   | Weighting | Unequal group sizes; verify equal total loss weight per group and mean row weight one |
   | Prediction orientation | Fake estimator with `classes_=[1,0]`; verify extraction of class 1 |
   | Bank dimensions | Valid one-repeat and three-fold fixtures; wrong declared fold count rejected |
   | Bank consistency | Corrupt stored AUC, deployment mean, IDs, recipe hash, or parent hash separately |
   | Blending | Different parent predictions, reversed member order, invalid weights, and recomputed AUC |
   | Seed averaging | Three seeds for one recipe plus another recipe; verify the latter does not lose weight |
   | Candidate eligibility | A valid one-repeat bank with a forged eligibility flag is still rejected |
   | Member duplicate checking | Perturb one duplicate in two members with opposite signs so the composite remains unchanged; validation must still reject |
   | Transformed equivalence | Different raw rows become identical after a fitted transform; the input audit must group them |
   | Submission corruption | Change a duplicate’s prediction and update the expected deployment consistently, so failure must come from the duplicate check |
   | Read-only verification | Patch writers to raise if called; verification still succeeds |
   | Exclusive writing | Existing-file and read-back-corruption mocks fail correctly |
   | Source/dependency mismatch | Mock one mismatch and assert a nonzero verification result |
   | Search order | A spy confirms the complete configuration list is saved before the first fit |
   | Anchor | All core arrays match the approved bank exactly |

   For outer-label isolation, do **not** demand that the entire recomputed CV bank remain unchanged. Those labels are legitimate training labels in other folds.

   The neural preprocessing and nested-containment checks can exercise common helpers with sklearn transformers and fake learners now. P2.5 and P2.7 must repeat the checks through their actual wrappers and orchestration.

   CatBoost and TabM runtime checks are explicitly deferred, not reported as passed.

15. **Build and review in a sequence that exposes errors early.**

   I would implement in this order:

   1. Recipe validation, source/dependency identity, and preserved-file checks.
   2. F2 adapter, LightGBM factory, one-fold runner, then exact 15-fit anchor reproduction.
   3. Bank loading/saving, audit evidence, and read-only verification.
   4. Date encoders, weights, and synthetic leakage checks.
   5. Bank blending, seed averaging, and pure selection.
   6. Submission construction and corruption tests.
   7. Search orchestration and comparison logging.
   8. Installation report processing and genuine route locks.
   9. Complete phase 1 verification and P2.1 review evidence.

   P2.1’s persisted evidence should include the new anchor bank and audit, its log record, self-check output, installation evidence, and a concise finding establishing reproduction. It should not include exploratory searches or a redundant Kaggle upload.

   The most likely formal-review problems are missing source dependencies, a seed subset mislabeled as complete, training on `transform(training_rows)` instead of cross-fitted features, audit evidence that only checks composite predictions, rounded selection metrics, and optional capabilities reported as implemented when they are only placeholders.

No new decision from Sam is needed for this design. Staging should be explicit in P2.1’s reviewed scope. Actual package compatibility remains uncertain until Claude generates the resolutions and the later tasks run the installation smoke tests.