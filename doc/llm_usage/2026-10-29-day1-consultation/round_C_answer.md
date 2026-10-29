Claude, I would build Day 1 around **one immutable row index, one saved split table, three baseline factories, and one prediction-bank format**. Predeclare fold-model averaging for deployment. Keep legacy reproduction separate from production results.

I read the committed documents and the converged P0.2 record. The working tree was clean. The additional checks below are consultation evidence; P1.1 and P1.2 should reproduce them independently.

1. **P1.1: six findings, with detailed evidence in the audit output**

   Use `analysis/audit_data.py`, independent of the future harness. Its default behavior should print results without writing files. Give it `--root` and `--include-scoping` options. The latter can execute the already-approved scoping implementation within a scoped warnings context.

   Suggested invocation:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python analysis/audit_data.py --include-scoping
   ```

   Print a provenance header containing the exact command, Python/package versions, audit-script hash, and SHA-256 hashes of all three CSVs. Separate deterministic results from timestamps and timing information so reviewers can compare reruns.

   | Finding | Exact question | Evidence the audit must print | Downstream use |
   |---|---|---|---|
   | **001: Dataset contract, target, and missingness** | What constitutes a training observation, which label is positive, and which rows lack usable predictors? | Shapes; ordered column names and dtypes; target counts including unexpected/missing labels; null counts per column; nonfinite numeric values; exact incomplete-row index and populated fields; test/sample ID uniqueness and ordered equality; sample column names. | Loader, explicit target map, eligibility mask, submission schema. |
   | **002: Observable track identity and repeated observations** | Which observations must remain together, and how do URI groups differ from feature-signature groups? | Distinct nonmissing URIs; group-size histogram; repeated groups/rows; within-URI feature/date consistency; conflicting-label groups/rows; redundant identifier-tail checks; constant-column check; exact audio-plus-date and audio-only signature counts; train/test URI and signature intersections; repeated test groups. | Connected components, duplicate checks, later weighting experiment, leakage explanation. |
   | **003: Release-date precision and validity** | Which date components are actually observed, and can every date be parsed without inventing information? | Counts of exact `YYYY`, `YYYY-MM`, and `YYYY-MM-DD` formats; malformed or impossible dates; partial-date rows; year range; full-date ordinal range; missingness expected in each F2 date field. | F1/F2 definitions and edge-case tests. |
   | **004: Marginal target associations** | Which supplied variables and release eras show marginal associations with High popularity? | For each audio feature and year: usable sample count, class counts, missing count, raw-direction AUC. For fixed eras: row count, distinct URI count, High count/rate. Print numeric ranges and category values alongside this table. | Baseline motivation and report hypotheses, not feature selection or conditional importance. |
   | **005: Descriptive train/test and ordering checks** | What observable differences and overlaps exist between the supplied files, without fitting on test rows? | Train/test year quantiles; per-feature quantiles, missingness and two-sample KS distances; exact-date overlap numerator/denominator; training-row-index AUC; test-ID/year correlation. Explicitly state populations and limitations. | Document the working validation assumption and identifier exclusions. |
   | **006: Legacy scoping reference** | What results does the exact approved scoping protocol produce, and what do its comparisons establish? | Re-executed scoping tables: all seven models, audio versus legacy date fields, ungrouped comparisons, exact-date encoding, and LightGBM year-versus-full-date comparison. Include rows, feature order, seeds, estimator parameters, and metric definitions. | Independent reference for P1.2 harness reproduction. |

   Keep each ledger entry short: conclusion, evidence location, uncertainty, downstream use, and review status. Put long tables and row lists in the audit output, not the ledger.

   **Important count distinction.** My independent connected-component check found:

   | Group definition | Complete-row groups | Groups including incomplete singleton | Mixed-label groups |
   |---|---:|---:|---:|
   | URI only | 3,642 | 3,643 | 32 |
   | URI or identical audio plus date | **3,637** | **3,638** | **35** |
   | Stricter grouping, also identical audio | **3,629** | **3,630** | **38** |

   The production grouping has a group of size **4**, although URI groups have maximum size **3**. These are different facts, not contradictory audit results.

   For P1.1, independently establish the signature counts and within-URI consistency. P1.2 must verify that its generic connected-component implementation reproduces the resulting components.

   **Items that should not become empirical findings:**

   - Playlist survivorship is an unverified explanation.
   - Similar marginals do not establish identical populations or label mechanisms.
   - The hypothetical public-AUC standard errors belong in a methodological note, with their assumptions.
   - The removed adversarial classifier stays excluded.
   - Standalone exact-date encoding does not establish its incremental value.
   - Naive fold SE is historical output, not a defensible confidence interval.

   For KS distances, NumPy is sufficient; do not introduce SciPy as a new direct dependency merely for this audit:

   ```python
   points = np.union1d(x, z)
   distance = np.max(np.abs(
       np.searchsorted(np.sort(x), points, side="right") / len(x)
       - np.searchsorted(np.sort(z), points, side="right") / len(z)
   ))
   ```

   Remove missing values and print the resulting sample sizes. Treat these as descriptive distances. Duplicate rows complicate ordinary independent-sample significance calculations.

   The phase plan should record:

   - Sam’s development go-ahead and his role as uploader.
   - P1.1 → reviewed → P1.2 → reviewed → P1.3.
   - Exact deliverables, commands, interfaces, and exit criteria below.
   - The fixed feature, split, deployment, and first-submission selection rules.
   - A dated status log and links to each review record.

   Update stale “P0.2 in review/development paused” status text within the appropriate reviewed task. Preserve the historical record. Arrange the ledger status transition within the review process; do not silently edit approved findings afterward.

2. **P1.2: package, protocols, artifacts, and numerical implementation**

   **Keep the package small.**

   ```text
   songpop/
       __init__.py
       data.py
       features.py
       splits.py
       models.py
       evaluate.py
       metrics.py
       artifacts.py
   scripts/
       __init__.py
       run_baselines.py
       check_foundation.py
       prepare_submission.py
   artifacts/
       splits/
       bank/
       metrics/
       experiments.jsonl
   ```

   Use root-level packages and run scripts as modules from the repository root:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.run_baselines --protocol legacy
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.run_baselines --protocol production
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.check_foundation
   ```

   This needs neither installation nor scattered `sys.path` modifications. Document the required working directory. Accept `--root` for data/artifact locations.

   Separate computation from writing. Reviewers must be able to recompute everything in memory using `--verify` or `--no-write`.

   Suggested public interfaces:

   ```python
   # data.py
   load_dataset(root: Path) -> Dataset
   validate_dataset(data: Dataset) -> None
   sha256_file(path: Path) -> str

   # features.py
   make_features(
       frame: pd.DataFrame,
       feature_set: Literal["F0", "F1", "F2", "legacy_date"],
   ) -> pd.DataFrame

   # splits.py
   make_groups(
       train: pd.DataFrame, *,
       row_ids: np.ndarray,
       strict_audio: bool = False,
   ) -> np.ndarray

   make_folds(
       y: np.ndarray,
       groups: np.ndarray,
       eligible: np.ndarray,
       row_ids: np.ndarray, *,
       split_seeds: tuple[int, ...],
       n_splits: int = 5,
   ) -> FoldSet

   validate_folds(data: Dataset, folds: FoldSet) -> pd.DataFrame

   # models.py
   make_estimator(recipe: Recipe) -> object
   predict_high(estimator: object, X: pd.DataFrame) -> np.ndarray

   # evaluate.py
   run_cv(data: Dataset, recipe: Recipe, folds: FoldSet) -> PredictionBank

   # metrics.py
   summarize_oof(bank: PredictionBank) -> dict
   paired_cluster_bootstrap(
       a: PredictionBank, b: PredictionBank, *,
       n_boot: int = 2000,
       random_state: int = 155_001,
   ) -> dict

   # artifacts.py
   save_bank(bank: PredictionBank, path: Path) -> None
   load_bank(path: Path) -> PredictionBank
   append_experiment(path: Path, record: dict) -> None
   make_submission(data: Dataset, bank: PredictionBank) -> pd.DataFrame
   validate_submission(
       submission: pd.DataFrame,
       data: Dataset,
       bank: PredictionBank,
   ) -> dict
   ```

   `Dataset`, `Recipe`, `FoldSet`, and `PredictionBank` can be small dataclasses. Avoid a general plugin framework or configuration language.

   **Loading and permanent row identity**

   - Assign `train_row_id = np.arange(len(train))` immediately on loading.
   - Preserve it after filtering. Never use a reset DataFrame index as the permanent identity.
   - Validate labels before mapping: exactly `{"High", "Low"}`, no missing values.
   - Map `{"Low": 0, "High": 1}` explicitly.
   - Define eligibility from the audited missing-feature condition. Assert that it excludes exactly the known incomplete row, rather than silently dropping arbitrary future rows.
   - The current complete-row population has **1,359 High and 2,504 Low**.
   - Preserve test IDs as supplied. They are not a newly generated range.
   - Use the sample submission only for its schema and ID sequence.

   The input hashes I obtained were:

   ```text
   train.csv
   d14b372deff47aa791242875ac5c134ce2497695bb190479b159b8ec49d68407

   test.csv
   70f1502ccae76396b45961a0e0c9f96c5ba213d5b99a101015c32ddc65d020e3

   sample_submission.csv
   35afd1abf23cb89279d1e10ef4d12d27b86d4406ae9b56984c1725b674835ca8
   ```

   P1.1 should establish these itself.

   **Connected components**

   Use union-find over training-row positions:

   1. Initialize every row as a singleton.
   2. For each nonmissing URI, union every occurrence with its first occurrence.
   3. For each exact tuple of the 13 raw numeric audio values plus the raw release-date string, union matching complete rows.
   4. For the stricter variant, additionally union exact audio-only tuples.
   5. Keep the audited incomplete row isolated.
   6. Label each component by the minimum permanent `train_row_id` within it.

   Use tuples and dictionary equality, not rounded floats or an unchecked hash alone. Do not include labels, URLs, or test rows in this graph.

   Required checks:

   - Every URI edge and signature edge joins equal group IDs.
   - The stricter grouping is a coarsening of the primary grouping.
   - Changing labels leaves groups unchanged.
   - Reordering input rows while preserving permanent row IDs leaves the aligned group assignments unchanged.
   - A synthetic transitive chain joins correctly.
   - Counts and memberships agree with the independent audit.
   - Every group belongs wholly to one fold.

   **Fold assignment and the incomplete row**

   Generate production folds on the **3,863 eligible rows only**, with their actual row labels:

   ```python
   StratifiedGroupKFold(
       n_splits=5, shuffle=True, random_state=split_seed
   )
   ```

   Do not assign one majority label to a mixed-label group. The installed implementation accumulates counts of each class within each group; mixed-label groups are supported. Stratification is approximate subject to group constraints. [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)

   After assigning complete rows, assign the incomplete singleton a fold without changing any existing assignments:

   ```python
   token = f"incomplete-v1:{split_seed}:{train_row_id}".encode()
   incomplete_fold = int.from_bytes(
       hashlib.sha256(token).digest()[:8], "big"
   ) % 5
   ```

   This assignment uses neither its label nor model scores.

   For the later keep-row ablation:

   - Validation remains the same eligible rows.
   - Include the incomplete row in training iff its assigned fold differs from the validation fold.
   - Its own held-out prediction, if requested, is a separate diagnostic.
   - The drop baseline excludes it from every fit.

   Save both primary and stricter fold assignments now, but label the stricter protocol **reporting-only, ineligible for selection**.

   A simple split CSV can contain one row per original training row:

   ```text
   train_row_id,eligible,primary_group,strict_group,
   primary_fold_155,primary_fold_2026,primary_fold_1029,
   strict_fold_155,strict_fold_2026,strict_fold_1029
   ```

   Keep legacy URI splits in a separate artifact. Hash both files and record their construction version.

   My primary-fold check produced validation sizes of **772 or 773**, with High rates between **0.3510 and 0.3523**. Reproduce that using the specified component labels and row order.

   Print per-fold:

   - Training/validation rows and distinct groups.
   - High/Low counts and High rate.
   - Counts and shares in `<1990`, `1990–1999`, `2000–2009`, `2010–2019`, `≥2020`.
   - Mixed-label group count.
   - Group intersections, which must be empty.

   Assert coverage, disjointness, and both classes in every train/validation partition. Use balance warnings, not seed hunting: for example, flag class-rate deviations over 0.03 or era-share deviations over 0.05. Do not assert equal era distributions, and do not regenerate partitions according to model performance.

   **Exact feature definitions**

   Preserve this audio order:

   ```python
   AUDIO = [
       "time_signature", "speechiness", "danceability", "duration_ms",
       "energy", "mode", "instrumentalness", "valence", "key", "tempo",
       "loudness", "acousticness", "liveness",
   ]
   ```

   Use float64 feature frames, ordinary `np.nan`, and explicit column order.

   | Field | Rule |
   |---|---|
   | `year` | Observed four-digit year. |
   | `month` | Observed month for year-month and full dates; otherwise NaN. |
   | `day` | Calendar day for full dates; otherwise NaN. |
   | `day_of_year` | Full dates only; January 1 is 1, leap years can reach 366. |
   | `weekday` | Full dates only; Monday 0 through Sunday 6. |
   | `date_precision` | Raw string length: 4, 7, or 10. |
   | `days` | Full-date calendar-day difference from `1970-01-01`; otherwise NaN. Negative values are valid. |

   Thus:

   ```text
   F0: AUDIO                                               13 columns
   F1: AUDIO + year                                        14 columns
   F2: AUDIO + year, month, day, day_of_year, weekday,
       date_precision, days                                20 columns
   ```

   Validate exact string formats and actual calendar validity. Do not let permissive datetime parsing silently turn partial dates into January 1.

   Specific cases:

   ```text
   1979-06:
     year=1979, month=6, precision=7
     day/day_of_year/weekday/days = NaN

   1984:
     year=1984, precision=4
     all other date fields = NaN

   1970-01-01:
     year=1970, month=1, day=1, day_of_year=1,
     weekday=3, precision=10, days=0
   ```

   Do not add cyclical encodings, missingness indicators, interactions, or zero indicators under the same F2 name. Those are later, separately named recipes.

   `legacy_date` must preserve the approved scoping order and midpoint formula exactly:

   ```text
   AUDIO + year, month, day, date_precision, days
   partial-date days = (year - 1970) * 365.25 + 182
   ```

   **Three production estimator recipes**

   | Model | Day 1 configuration |
   |---|---|
   | LightGBM | Scoping core: 600 trees, learning rate 0.02, 15 leaves, minimum child samples 20, subsample 0.8, subsample frequency 1, column sample 0.8, estimator seed 0. |
   | Random forest | 500 trees, minimum leaf size 3, estimator seed 0; otherwise the pinned-version defaults, including Gini criterion and `max_features="sqrt"`. |
   | Logistic | The scoping log transformations, then fold-fitted median imputation, standardization, and logistic regression with `C=1`, `max_iter=2000`. |

   For logistic, transform duration with `log1p(duration_ms / 1000)` and instrumentalness, speechiness, liveness, and acousticness with `log1p(100*x)`. Compute medians **after** transforming. No additional indicator features on Day 1.

   Instantiate a new estimator for every fold. Use estimator seed 0 throughout; split seeds remain separate. Day 1 uses fixed tree counts, so no early-stopping machinery is needed yet.

   `predict_high()` must locate class `1` in `classes_`, validate the class set, and validate prediction shape/range. Never rely on the class ordering of string labels.

   For production LightGBM, I recommend explicit CPU execution, four threads, `deterministic=True`, and `force_col_wise=True`. Record these additions. Keep the legacy reproduction factory unchanged. LightGBM documents the deterministic setting and the accompanying histogram-mode recommendation, while warning that different builds/platforms can still differ. [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html)

   **Legacy reproduction through the harness**

   Reuse the approved scoping estimator factories and legacy feature builder, but run them through the new fold/prediction/metric code. Do not merely call `scoping.cv_auc()` and label that harness verification.

   Import scoping within `warnings.catch_warnings()`: that module installs a global ignore filter, which must not suppress production warnings.

   Reproduce the committed audio/full-date table for all seven existing factories. No new estimator implementations or packages are needed. Also reproduce LightGBM F1 at 0.8143 and its +0.0125 increment.

   The key expected values from the committed output are:

   | Model | Audio | Legacy audio/date |
   |---|---:|---:|
   | Logistic | 0.7300 | 0.7812 |
   | Random forest | 0.7557 | 0.8257 |
   | Extra Trees | 0.7449 | 0.8197 |
   | HistGB | 0.7415 | 0.8201 |
   | LightGBM default | 0.7401 | 0.8217 |
   | LightGBM slow | 0.7513 | 0.8268 |
   | XGBoost default | 0.7331 | 0.8080 |

   Then run the nine production combinations: three models × F0/F1/F2 × 15 fits. Never mix legacy and production banks in an ensemble or paired comparison.

   **Prediction bank**

   Use one `.npz` file per recipe/run, loaded with `allow_pickle=False`. This avoids new Parquet dependencies and preserves float64 scores without CSV rounding.

   Required arrays:

   ```text
   train_row_id        (N,)        eligible original row IDs
   y                   (N,)        integers 0/1
   group_id            (N,)        primary or declared protocol groups
   split_seeds         (R,)
   fold_id             (R,N)
   oof                 (R,N)
   test_id             (863,)
   test_by_fold        (R,5,863)
   test_mean           (863,)
   fold_auc            (R,5)
   metadata_json       scalar Unicode JSON string
   ```

   Initialize OOF arrays to NaN and maintain an assignment counter. Every eligible row must receive exactly one held-out prediction per repeat.

   Metadata should include:

   - Schema, feature-builder, grouping, and split-protocol versions.
   - Recipe name, complete resolved estimator/preprocessing parameters, feature names/order/dtypes.
   - Input hashes, split-file hash, ordered row population and excluded-row IDs.
   - Split seeds, estimator seeds, bootstrap seed where applicable.
   - Deployment rule and fold aggregation weights.
   - Python and installed package versions, OS/architecture, thread settings.
   - Git base commit **and hashes of the actual executing source/config files**.
   - Run command and execution timestamp.
   - Warnings, convergence status, and completed-fit count.

   Git HEAD alone is insufficient while P1.2 code is uncommitted. Do not place the final whole-project fingerprint inside an artifact included in that fingerprint; that creates a self-reference. Associate bank hashes with approval in the review record.

   Keep banks immutable. A changed feature definition or code identity produces a new run identifier. Do not overwrite an existing bank.

   **Experiment log**

   One JSONL record per completed or failed run:

   ```text
   run_id, recipe_id, task_id, protocol, selection_eligible,
   feature_set, model, parameters, preprocessing,
   source_manifest_hash, data_hashes, split_hash,
   split_seeds, estimator_seeds, deployment,
   n_rows, n_groups, n_features, n_completed_fits,
   fold_auc, repeat_mean_auc, mean_fold_auc, pooled_oof_auc,
   elapsed_seconds, warnings, execution_status, error,
   bank_path, bank_sha256, timestamp_utc
   ```

   Put paired comparisons in separate records naming both run IDs and their bank hashes. Keep submission/public-score events separate from immutable training results. A pending public score is not zero.

   **Metrics and paired bootstrap**

   Compute:

   - Fifteen individual fold AUCs.
   - Five-fold mean within each repeat.
   - Unweighted mean of all 15 fold AUCs, the primary metric.
   - Pooled OOF AUC separately for each repeat.
   - Optional AUC of the three-repeat averaged OOF predictions, explicitly secondary.

   Never flatten three repeats into a fictitious sample of `3*N` independent observations.

   Before pairing banks, require exact agreement on row IDs, labels, groups, split seeds, and fold assignments.

   An efficient exact weighted-AUC implementation can cache the prediction ordering and tied-score boundaries:

   ```python
   def prepare_auc(y, score, group_code):
       order = np.argsort(score, kind="stable")
       s = score[order]
       starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
       return y[order].astype(float), group_code[order], starts

   def cached_weighted_auc(cache, group_multiplicity):
       y, g, starts = cache
       w = group_multiplicity[g]

       pos = np.add.reduceat(w * y, starts)
       neg = np.add.reduceat(w * (1.0 - y), starts)
       P, N = pos.sum(), neg.sum()
       if P == 0 or N == 0:
           return np.nan

       neg_below = np.cumsum(neg) - neg
       return np.sum(pos * (neg_below + 0.5 * neg)) / (P * N)
   ```

   Dense `group_code` indexes the same global list of complete-row groups in every fold and repeat.

   For each of 2,000 draws:

   ```python
   multiplicity = rng.multinomial(G, np.full(G, 1.0 / G))
   ```

   Use this one multiplicity vector for both models and every repeat. Calculate all 15 paired fold differences and their mean. If any fold has zero weighted positives or negatives, mark the **whole replicate** invalid, rather than averaging fewer folds.

   Return the observed difference, 2.5/97.5 percentile interval, valid/invalid counts, seed, and fixed-prediction interpretation. On this dataset, invalid draws should be exceptionally unlikely; investigate any occurrence.

   I checked this cached calculation against sklearn on 100 randomized tied-score/zero-weight cases: maximum discrepancy was **1.1×10⁻¹⁶**. A synthetic 2,000-replicate workload at this dataset’s dimensions took approximately **0.9 seconds**. Retain sklearn as the reference implementation in verification.

   Day 1 needs a small predefined comparison set: F1−F0 and F2−F1 for each model, plus RF versus LightGBM on F2. There is no need for every possible pair.

   **Deployment**

   Predeclare:

   ```python
   test_mean = test_by_fold.mean(axis=(0, 1))
   ```

   Thus each baseline submission averages its 15 fitted fold models equally. This uses existing fits and keeps deployment simple. Do not claim that CV directly measures the benefit of this 15-model average: each training row has only three held-out predictions.

   Do not add full-data refits on Day 1. Persisting hundreds of forest trees is also unnecessary; later interpretation can refit from the frozen recipe and saved folds.

   **Submission checks**

   Validate the final CSV after reading it back:

   - Exactly 863 rows and columns `ID,Popularity_Type`, in that order.
   - No index column.
   - Unique IDs, with exact ordered equality to the sample and bank.
   - Finite numeric probabilities in `[0,1]`.
   - Predictions equal the bank’s declared deployment output, within the serialization tolerance.
   - Positive-class provenance comes from the target map and estimator classes, not an attempted inference from the CSV.
   - Print minimum, maximum, mean, quantiles, number of unique predictions, and counts at 0/1.
   - Hash the actual final bytes.

   For duplicate consistency, ignore `ID`:

   1. Check all 11 repeated-URI pairs, after verifying their predictor equality.
   2. Group test rows by identical **recipe feature vectors**, treating corresponding NaNs as equal.
   3. Require each group’s prediction spread to be at most `1e-12`.

   Checking active feature vectors also catches additional equivalent rows under F0. Never require equal predictions merely because two rows share audio while their F2 dates differ. Do not average duplicate predictions afterward to hide an upstream bug.

3. **Verification without pytest**

   Use the standard library’s `unittest`, or explicit functions called by `scripts/check_foundation.py`. Tests should operate in memory using `StringIO`/`BytesIO` where possible, so the reviewer can run them in a read-only sandbox.

   | Area | Required checks |
   |---|---|
   | Loading | Unknown/missing target rejected; High→1 and Low→0; permanent row IDs survive filtering; unexpected CSV/schema/hash differences reported. |
   | Groups | URI equality; signature equality; transitive chain; label invariance; row-permutation invariance; incomplete singleton; strict grouping coarsens primary. |
   | Folds | Exact saved/recomputed equality; each eligible row held out once per repeat; no group split; both classes present; fixed incomplete-row assignment; complete folds unchanged by keep/drop choice. |
   | Dates | Year-only; `1979-06`; epoch date; pre-epoch negative day; leap day; invalid date; malformed precision; exact column order and dimensions. |
   | Preprocessing | A deliberately extreme validation/test value does not affect fitted medians/scales; production transforms preserve column order; no fitting on concatenated train/test. |
   | Probabilities | Correct class-column extraction; missing class rejected; wrong length, NaN, infinity, and out-of-range values rejected. |
   | OOF coverage | Duplicate assignment and missing assignment fail; stored training/validation memberships are disjoint by group. |
   | AUC | Perfect=1; reversed=0; constant=0.5; weighted tied cases match sklearn; zero weights handled; one weighted class flagged. |
   | Bootstrap | Identical predictions give zero observed difference and interval `[0,0]`; repeatability with fixed seed; swapping A/B negates draws; group multiplicities shared across repeats. |
   | Banks | NPZ round trip with `allow_pickle=False`; mismatched data/splits/row order rejected; existing run not overwritten. |
   | Submission | Valid round trip accepted; wrong column, swapped IDs, duplicate ID, missing row, NaN, infinity, altered probability, and one deliberately altered duplicate prediction rejected. |
   | Reproduction | Legacy table matches committed values to four decimals; new production results are recorded separately. |

   Use numerical tolerances for floating calculations, not arbitrary rounding inside the metric. For the legacy checkpoint, compare the displayed four-decimal values. A mismatch triggers investigation of versions, row order, group order, feature order, parameters, or thread/build differences. Do not change the expected number merely to make the check pass.

   Split verification into fast unit checks and slower reproduction checks, but run both before P1.2 review. Do not globally suppress warnings. In particular, a logistic convergence warning needs attention before publishing its baseline result.

4. **P1.3: first file, report skeleton, Colab skeleton, and evidence**

   Select the F2 baseline with the highest production mean-fold AUC. Predeclare an exact numerical tie rule, for example scores within `1e-12` use a fixed model-name ordering. Do not use the public leaderboard or stricter stress-test protocol.

   A margin below 0.003 does not establish superiority; it can still determine which baseline file is used for the initial pipeline check.

   **Submission description**

   Fill this from the reviewed bank, not by hand from memory:

   > P1.3 pipeline baseline: `{model}`, F2 audio/date features, High=1. Five-fold StratifiedGroupKFold, split seeds 155/2026/1029, connected-component track groups, estimator seed 0. Equal average of 15 fold-model probabilities. Mean-fold CV AUC `{auc}`. Provided training data only; no pretrained models or test-fitted preprocessing. Recipe `{recipe_id}`, file SHA-256 `{hash}`. Purpose: verify the end-to-end submission pipeline.

   Keep the full description and hash in a local submission manifest/review record; shorten the Kaggle text if necessary.

   Claude prepares and validates the file. Sam uploads it. Record:

   - Prepared file path, byte hash, recipe and bank hash.
   - Formal review record/fingerprint.
   - Exact description given to Sam.
   - Later, Sam’s reported submission identifier, upload time, acceptance status, and public score.

   Distinguish **“reviewed file ready for Sam”** from **“Kaggle accepted.”** P1.3’s external acceptance remains pending until Sam reports it.

   **Report skeleton**

   Create a working report alongside the unchanged course template, preserving the existing preamble arrangement. Remove the example LLM images and example prose, which are not project evidence.

   | Section | Fill now | Leave explicit placeholders |
   |---|---|---|
   | Introduction | Task, metric, deadlines; confirmed team information only. | Team/Kaggle name if unknown, private rank/AUC, final links, division of labor. |
   | Overview | Day 1 timeline; the nine production baselines; separate legacy-reproduction note. | Later models and decisions. |
   | Approach | Reviewed audit facts, target mapping, groups, missing row, F0/F1/F2, baseline preprocessing. | Later engineering and tuning. |
   | Model Selection | Exact split protocol; primary/secondary metrics; bootstrap limitations; fold-average deployment. | Final restricted nested evaluation and final selections. |
   | Conclusion | Challenges already encountered, clearly provisional. | Top 10 original features, directions, final lessons. |
   | Additional insights | Optional outline on AUC, duplicates, and parallelization. | Unsupported conclusions. |
   | LLM Usage | Actual author/consultant/reviewer roles and evidence references. | Later interactions and integrations. |

   Do not call the baseline the “best final model,” and do not fill the feature-importance table from univariate AUCs.

   **Colab without a git remote**

   For Day 1, use an **uploaded, versioned code-only ZIP** plus the three CSVs. This is simpler than duplicating the harness inside notebook cells or adding Drive mounting.

   The ZIP should contain:

   ```text
   songpop/
   scripts/
   requirements.txt
   selected_recipe.json
   saved split artifact
   source_manifest.json
   ```

   Generate it from the reviewed source and record its hash. Include no virtual environment, caches, credentials, or serialized models.

   Notebook cells should:

   1. Explain the task, constraints, target orientation, and that this is currently a baseline demo.
   2. Inspect Python/runtime versions.
   3. Provide pinned installation instructions, before importing the modeling libraries.
   4. Upload the ZIP and `train.csv`, `test.csv`, `sample_submission.csv` in one upload step.
   5. Verify code-manifest and CSV hashes.
   6. Extract the package into a known runtime directory and add that directory to Python’s import path.
   7. Load the data and selected recipe.
   8. Verify saved folds, then execute the same `run_cv()` path.
   9. Display the baseline metrics and simple audit/date plots.
   10. Validate and download a submission.

   Keep the full search out of the notebook. A quick demonstration mode may use fewer fits only if clearly labeled as **not reproducing the submitted recipe**; the default final demo must reproduce it.

   Day 1 need not install anything or claim a successful Colab run. The clean-runtime run remains a later task. Before final publication, the code ZIP can be supplied through a public read-only Drive download link or generated into the notebook from the same reviewed source. Whichever route is chosen, verify identical source hashes. Viewers should supply the provided competition CSVs rather than rely on Sam’s private Drive layout.

   **LLM evidence**

   Save this consultation prompt/answer verbatim under a Day 1 consultation directory. Maintain a small integration index:

   ```text
   component
   author/consultant/reviewer role
   model identity and session evidence
   prompt/response location and adopted excerpt
   implementation file/function
   what was changed or rejected
   verification performed
   task/review-record reference
   ```

   For example: “bootstrap cached-sort algorithm adopted; implemented in `metrics.py`; checked against sklearn weighted AUC.” Record rejected recommendations too when they explain a material choice. Review records alone are not a substitute for the adopted interaction excerpts required by the PDF.

5. **Library-specific pitfalls and the cut order**

   - **RandomForest NaNs:** scikit-learn 1.9.1 supports native NaNs for this configuration. I fitted a small forest on the actual F2 matrix and obtained finite test predictions with zero spread within repeated URI groups. Keep native missing handling; no extra imputer is needed. If a feature had no missing training values, prediction-time missing values follow the larger child. [RandomForest documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

   - **Mixed-label groups:** preserve row labels. Do not deduplicate with a majority vote before splitting. The 35 production mixed-label groups are not the same population as the 32 URI-conflict groups.

   - **Partial dates create real F2 NaNs even in an otherwise complete test CSV.** “Test has no missing raw values” does not mean the derived feature matrix is complete.

   - **Pandas parsing:** validate string format explicitly and parse full dates with an explicit format. Do not rely on inferred mixed-date parsing. Preserve the known month in `1979-06`.

   - **LightGBM zero values:** retain the normal distinction between zero and missing. Zero instrumentalness is a legitimate observed value. Do not enable `zero_as_missing`.

   - **LightGBM subsampling:** preserve `subsample_freq=1`; setting a subsample fraction alone does not activate the intended row bagging.

   - **Estimator objectives differ:** the forest baseline uses Gini, while logistic and LightGBM use their classification objectives. The report should not claim that every baseline was fitted with log loss.

   - **Threading:** the current machine exposes 10 logical CPUs. Run folds and recipes sequentially, with four model workers/threads for production. Set BLAS limits before Python starts, for example `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1`. Record effective settings. Avoid outer multiprocessing combined with `n_jobs=-1`.

   - **Legacy imports:** restore warning filters after loading scoping code. Do not inherit its global warning suppression into production.

   - **Reproducibility across systems:** identical package versions do not guarantee bitwise-identical LightGBM output across Mac and Colab builds. Preserve exact submitted artifacts; later report the actual clean-Colab reproduction tolerance rather than promising equality now.

   If Day 1 runs long, cut in this order:

   1. Extra plots and polished prose beyond the skeletons.
   2. Model serialization, elaborate configuration machinery, and parallel execution.
   3. Unscheduled models, additional encodings, tuning, and ensembling.
   4. Full Colab execution today; retain the runnable skeleton and code-bundle contract.
   5. Extra pairwise bootstrap comparisons beyond the predefined contrasts.

   Do **not** cut the task dependency gates, input/split provenance, legacy checkpoint, OOF coverage checks, target orientation, duplicate consistency, or final submission read-back validation. If the three required tasks exceed the nominal time box, report that plainly rather than weakening their exit criteria.