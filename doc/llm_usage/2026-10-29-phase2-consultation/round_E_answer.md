Claude, I would make phase 2 a bounded search over strong tree models, a few genuinely different predictors, and carefully cross-fitted date features. Compute is cheap here. The scarce resources are reliable comparisons, implementation complexity, and review turnaround.

One correction to the premise: **LightGBM and random forest are close; logistic regression is not**. Their reviewed F2 scores are 0.8315, 0.8279, and 0.7819. There is still room for a useful smooth or neural challenger, particularly as an ensemble member.

I read the requested plans, findings, package, scripts, baseline output, and all 31 experiment-log records. I changed no files and installed nothing. Three new read-only probes inform the recommendations below; these are **unreviewed consultation inputs**:

| Probe | Result |
|---|---|
| Equal probability average of the existing LightGBM F2 and RF F2 OOF banks | Mean fold AUC **0.833895**, gain **0.002421**; positive gains on all three repeats |
| Equal average of all three F2 baselines | **0.827549**, below LightGBM |
| Exact-date coverage | Validation rows whose date occurs in their fold’s training portion: **69.87%, 68.65%, 69.35%** by repeat; test coverage **623/863 = 72.2%** |
| Date support among eligible training rows | **1,629** distinct dates; **619** dates span multiple primary groups; **2,806** rows belong to those dates |
| Pinned sklearn `TargetEncoder` with explicitly supplied grouped splits | Matched a manual smoothed-mean calculation; changing only an inner held-out fold’s labels left that fold’s encodings unchanged |

The first blend misses the strategy’s 0.003 promotion threshold. Keep it as a cheap fixed candidate; do not call it a promoted improvement.

1. **Organize phase 2 into seven reviewable tasks, with two freezes.**

   Each task should produce its code/configuration, immutable results, concise draft findings, verification output, and an LLM integration-index entry. Review batches of related experiments rather than each configuration separately.

   The schedule below includes room for review. Times through Saturday are **PDT**; Sunday times are **PST**.

   | Task | Content and deliverables | Dependency | Target window |
   |---|---|---|---|
   | **P2.1: Experiment infrastructure** | Phase plan; recipe and run schemas; subset-seed runner; fold-local feature interface; ensemble arithmetic; phase 1 compatibility checks; reviewed installation manifests and commands | Approved P1.2/P1.3 | Thu 08:15–12:30 |
   | **P2.2: Existing-package search** | RF and LightGBM searches; small Extra Trees, HistGB, XGBoost controls; spline logistic; reproduce the fixed LGB/RF blend through the new harness | P2.1 | Thu afternoon |
   | **P2.3: CatBoost and optional EBM** | Execute the reviewed installations; compatibility checks; CatBoost pilot/search; at most four EBM configurations; confirmed banks | P2.1 and reviewed P2.2 reference results | Thu evening–Fri morning |
   | **P2.4: Features and training weights** | Date frequency/target encodings; compact F3 blocks; inverse-group-size training weights; paired comparisons | P2.2/P2.3 | Fri afternoon |
   | **P2.5: Optional neural challenger** | One TabM implementation, one timing pilot, at most two configurations, confirmation if warranted | Reviewed infrastructure and installation manifest | Sat morning; 90 minutes author time |
   | **P2.6: Ensembles and candidate-menu freeze** | Fixed small blend menu; estimator-seed averaging; complete confirmation banks; deterministic selector; nested-evaluation manifest | Reviewed preceding results; P2.5 only if completed | Sat afternoon; menu frozen by **18:00 PDT** |
   | **P2.7: Restricted nested evaluation** | Independent outer evaluation of the frozen selection procedure; bagging/refit comparison; final evidence table and frozen deployment manifests | P2.6 | Sun; final freeze by **18:00 PST** |

   Reserve roughly **one hour per task for review**, plus correction time. That is a planning allowance, not permission to stop before convergence.

   **Cut order:** incomplete-row ablation; neural challenger if setup or projected evaluation cost is excessive; optional EBM after spline logistic exists; secondary controls/local refinements. Do not cut grouped cross-fitting, historical verification, submission validation, or the restricted nested evaluation merely to run more configurations.

   **Departure from strategy §6:** replace its already-completed F1/F2 task with infrastructure and tuning. Start Thursday because phase 1 finished early. Move interpretation and report-oriented stress tests after the close in accordance with Sam’s latest instruction.

2. **Extend the harness without changing phase 1’s computational identity.**

   The simplest solution is an additive subpackage:

   ```text
   songpop/phase2/
       recipes.py
       engine.py
       features.py
       encoders.py
       models.py
       banks.py
       ensembles.py
       selection.py
       nested.py

   scripts/
       run_experiments.py
       check_phase2.py
       evaluate_candidates.py
       prepare_candidate.py

   configs/phase2/
       searches/*.json
       recipes/*.json
       candidate_menu.json

   artifacts/phase2/
       bank/
       searches/
       splits/
       experiments.jsonl
       selections/
   ```

   **Leave the existing top-level `songpop/*.py`, baseline scripts, `requirements.txt`, phase 1 banks, and phase 1 log unchanged.** This matters because `run_baselines.sources()` hashes every top-level `songpop/*.py`. A new subdirectory does not enter that glob; a new top-level module does.

   Keep the original `.venv` unchanged. Use a separate phase 2 environment for new dependencies. Existing commands remain valid:

   ```text
   .venv/bin/python -m scripts.run_baselines --protocol production --verify
   .venv/bin/python -m scripts.prepare_submission --verify
   ```

   New scripts must not append phase 2 runs to the old log: P1.3’s selector expects exactly one production F2 run per baseline family.

   I would use these interfaces:

   ```python
   load_recipe(path: Path) -> Recipe

   fit_recipe(
       recipe: Recipe,
       train: pd.DataFrame,
       y: np.ndarray,
       groups: np.ndarray,
       context: FitContext,
   ) -> FittedRecipe

   FittedRecipe.predict_high(raw_rows: pd.DataFrame) -> np.ndarray

   run_recipe_cv(
       recipe: Recipe,
       data: Dataset,
       folds: FoldSet,
       split_seeds: tuple[int, ...],
   ) -> PredictionBankV2

   blend_banks(
       banks: list[PredictionBankV2],
       weights: list[float],
   ) -> PredictionBankV2
   ```

   `fit_recipe` receives only the fitting population. It must not receive outer validation labels or reach back into the full dataset. `FitContext` contains permanent row IDs, deterministic seed information, and any permitted inner splits.

   Recipes should be JSON, not executable Python expressions. Include model parameters, feature blocks, preprocessing, training weights, estimator seeds, stopping policy, and deployment policy.

   Preserve the existing NPZ prediction arrays. Introduce `bank_v2` metadata for:

   - Explicit fold count and repeat count.
   - Evaluation population versus fitting population.
   - Feature recipe and source-column lineage.
   - Per-fold inner split hashes and selected iteration counts, where applicable.
   - Estimator seeds and parent bank hashes.
   - `prediction_kind`: fitted model, seed average, or model blend.
   - Search/stage identity and selection eligibility.

   Use the existing array validator with an explicit fold count; add v2 metadata validation separately. Do not silently weaken v1 validation.

   New verification must target an explicit run or manifest, rather than regenerate a filename from whichever code happens to be current:

   ```text
   python -m scripts.run_experiments --verify --run-id RUN_ID
   ```

   Record the executing code’s explicit dependency manifest. If a required source file has changed, report that historical replay requires its recorded Git revision. Never attach old source hashes to newly executed code. If a shared phase 1 file eventually must change, preserve a replay snapshot from its recorded revision and run its original verification entry points there.

   **Reason:** provenance should identify the computation that produced a bank, not force deletion and regeneration of approved historical evidence whenever development continues.

3. **Use bounded searches, mostly on all three split seeds, with fixed tree counts.**

   At 6–8 seconds per 15-fit run, restricting every cheap search to seed 155 saves little. It increases the chance of pursuing a configuration that happens to favor one partition.

   My initial budget is approximately **100 configurations across all families**, not hundreds per family. This is a cap, not a quota.

   | Family | Initial budget | Search space |
   |---|---:|---|
   | **LightGBM** | 32 configurations, including baseline | `num_leaves ∈ {7,15,31}`; `min_child_samples ∈ {10,20,40,80,120}`; learning rate `{0.02,0.05}`; trees `{300,600,1200}`; L1 `{0,0.1,1,5}`; L2 `{0,0.1,1,10,30}`; column fraction `{0.7,0.85,1}`; row fraction `{0.8,1}` |
   | **Random forest** | 18, including baseline | 500 trees; leaf size `{1,2,3,5,8,12,20}`; `max_features ∈ {"sqrt",0.5,0.75,1.0}`; depth `{None,12,24}`; bootstrap enabled; `max_samples ∈ {0.7,1.0}` |
   | **CatBoost** | 24 including anchors | Depth 3–7; L2 `{3,10,30}`; random strength `{0,0.5,1,2}`; learning rate `{0.03,0.06}`; iterations `{500,1000,1600}`; Plain and Ordered; fixed symmetric trees |
   | **Extra Trees** | 6 | 500 trees; leaf size `{2,5,12}`; feature fraction `{0.5,1}`; no bootstrap |
   | **XGBoost** | 8 | `hist`; depth 2–4; child weight `{3,10,30}`; L2 `{1,10,30}`; learning rate `{0.03,0.07}`; trees `{500,1000}`; row/column sampling 0.8 |
   | **HistGB** | 6 | Leaves `{7,15,31}`; minimum leaf `{20,50}`; learning rate 0.05; bounded fixed iteration counts; `early_stopping=False` |
   | **Spline logistic** | 6 | Knots `{4,7}`, degree 3, `C ∈ {0.1,1,10}`; fold-fitted imputation, splines, categorical encoding, scaling |
   | **EBM, optional** | 4 | Main effects versus three declared interactions; fixed rounds `{500,1500}` |

   Draw deterministic random configurations, save the ordered configuration list before fitting, and deduplicate identical recipes. Use one trial at a time with four model threads. There is little benefit to introducing asynchronous scheduling here.

   For CatBoost, first time one representative configuration. If a 15-fit run costs no more than about a minute, use all three repeats throughout. Otherwise screen on seed 155 and confirm the top three configurations plus an anchor on the remaining seeds. This is an explicitly declared runtime branch, not a decision based on favorable scores.

   **Iteration policy:** for LightGBM and CatBoost, I recommend fixed iteration counts as hyperparameters for this phase. Explicitly disable CatBoost’s best-model selection and supply no outer `eval_set`. Keep the stated LightGBM deterministic settings. Do not cache a globally constructed LightGBM `Dataset`; each fit must learn its binning from its fitting rows.

   This departs from §5.4’s planned 2,000–4,000-round inner early stopping. The reason is the measured runtime: a small iteration grid is inexpensive and removes a substantial implementation and review burden. It remains compliant with the principle that outer validation data cannot govern fitting.

   If fixed-count results consistently favor the largest count, allow **one predeclared local extension**, not an indefinite expansion. For a family earning further engineering time under the existing promotion rule, allow at most 12 local trials. Optuna is useful here, but optional.

   Treat every searched CV maximum as **selection-optimistic**. Three repeats improve stability; they are not three independent datasets. Bootstrap intervals do not correct the search. Log all attempted configurations, failures, and selection decisions. The Sunday evaluation assesses the restricted frozen menu, not the entire exploratory search.

4. **Install CatBoost; keep other installations conditional and preserve the pinned environment.**

   Benchmark evidence supports testing diversity, but does not identify the winner on this dataset. TabArena found strong tree performance, competitive tuned neural ensembles, and validation overfitting in cross-model ensembling. That favors a small measured menu over importing a large AutoML system. [TabArena, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)

   These are the versions I would propose for the installation manifests:

   | Package | Recommendation | What I verified; remaining uncertainty |
   |---|---|---|
   | **`catboost==1.2.10`** | Install | CPython 3.12 universal2 wheel exists. Its declared NumPy `<3` and pandas `<4` bounds admit the pinned versions; NumPy 2 support was added in 1.2.8. Runtime smoke test still required. |
   | **`optuna==4.9.0`** | Optional local refinement | Stable release from June 2026, Python ≥3.9, pure-Python wheel. No need to adopt the newer major version for this task. |
   | **`interpret-core==0.7.8`** | Optional EBM; use core package, without visualization extras | Declared requirements admit NumPy 2.5.4, pandas 3.1.0, sklearn 1.9.1. Verify native library loading and fitting on this Mac. |
   | **`torch==2.14.1`, `tabm==0.0.3`** | Only for P2.5 | Torch has a CPython 3.12 macOS arm64 wheel. TabM declares Torch `<3`; its numerical-embedding dependency must also be pinned in the resolved lock. |
   | **`pytabkit==1.7.3`** | Do not install in the main plan | Viable alternative for RealMLP, but adds Lightning/torchmetrics and wrapper behavior to audit. Do not install both neural routes. |
   | **SHAP** | Defer until after close | It contributes no model-development capability needed now. Recheck and pin then. |

   Sources: [CatBoost release files](https://pypi.org/project/catboost/1.2.10/), [declared dependencies](https://raw.githubusercontent.com/catboost/catboost/v1.2.10/catboost/python-package/setup.py), [NumPy support](https://raw.githubusercontent.com/catboost/catboost/v1.2.10/RELEASE.md), [Optuna 4.9.0](https://pypi.org/project/optuna/4.9.0/), [Interpret dependencies](https://raw.githubusercontent.com/interpretml/interpret/v0.7.8/python/interpret-core/setup.py), [Torch 2.14.1](https://pypi.org/project/torch/2.14.1/), [TabM dependencies](https://raw.githubusercontent.com/yandex-research/tabm/main/pyproject.toml), [PyTabKit](https://pypi.org/project/pytabkit/).

   **These are metadata-compatible proposals, not tested compatibility certifications.** None of these optional packages is currently installed.

   Before executing an installation:

   - Capture the complete installed distribution/version inventory, including transitive dependencies.
   - Resolve with that inventory constrained, using `pip install --dry-run --report ... --only-binary=:all:`.
   - Reject a plan that changes any protected installed version.
   - Record exact new versions, wheels and hashes.
   - Execute the reviewed plan in the separate environment.
   - Run `pip check`, imports, a small real-data fit with F2 NaNs, prediction/reload checks, and baseline compatibility checks.

   Pip documents the dry-run JSON report mechanism explicitly. [pip installation reports](https://pip.pypa.io/en/stable/cli/pip_install/)

   For **CatBoost**, begin with numeric F2. Use `thread_count=4`, explicit seed, `allow_writing_files=False`, CPU execution, explicit boosting type, and no automatic class weights. Do not feed exact-date strings into native target statistics by default. Ordered boosting does not establish that categorical target statistics exclude the entire repeated-track group. If low-cardinality musical fields are categorical, force one-hot treatment, which disables CTRs for those fields. [CatBoost parameter documentation](https://catboost.ai/docs/en/references/training-parameters/common)

   For **EBM**, disable its default random validation and bagging: `validation_size=0`, `outer_bags=1`, `inner_bags=0`, fixed rounds, four threads. Use main effects or explicit pairs such as year/loudness, year/acousticness, energy/loudness. The default EBM validation procedure is not suitable unchanged. [InterpretML API](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html)

   For **TabM**, use one deliberately modest architecture, initially two blocks, width 128, 16 members, dropout 0.1; at most one alternative preprocessing/configuration. Fit imputation, scaling and any numerical bins inside the fitting population. Train member losses separately and average probabilities at inference, as the official implementation instructs. [TabM implementation](https://github.com/yandex-research/tabm)

   Use group-safe inner early stopping for the neural model, then reinitialize and refit the outer-training population for the chosen epoch count. CPU first, no compilation, no mixed precision, deterministic seeds, zero data-loader workers. Stop the challenger if setup exceeds 30 minutes or its projected confirmation plus nested-evaluation cost is unreasonable.

   TabM and RealMLP are reasonable choices based on their papers; this is not evidence that either will help here. [TabM, ICLR 2025](https://arxiv.org/abs/2410.24210), [RealMLP, NeurIPS 2024](https://arxiv.org/abs/2407.04491)

   **Do not schedule** broad AutoML, FT-Transformer, TabNet, retrieval networks, or a second neural family. RBF SVM and kNN are optional cheap controls only after the main work, not prerequisites.

5. **Test compact feature blocks, with exact-date encoding as the most interesting new hypothesis.**

   Start each feature comparison with the reviewed LightGBM F2 recipe, then confirm a useful block with the tuned tree recipe. This separates feature effects from simultaneous parameter changes.

   | Order | Block | Exact proposal |
   |---|---|---|
   | 1 | **Date frequency** | `log1p(number of distinct fitting groups sharing the raw date string)`; unseen date gives zero |
   | 2 | **Date target encoding** | F2 plus grouped cross-fitted exact-date mean, smoothing strengths 10 and 30; compare with and without frequency |
   | 3 | **Musical categories and smooth preprocessing** | One-hot key/mode/time signature for spline, kernel or neural models; existing log transforms; missing indicators |
   | 4 | **Small interaction block** | Centered year × loudness; centered year × acousticness; energy × loudness; energy × danceability; acousticness × instrumentalness; `tempo * duration_ms / 60000` |
   | 5 | **Cyclic calendar representation** | Month and weekday sine/cosine pairs; optional day-of-year fraction using the actual year length; retain missingness for unobserved components |
   | 6 | **Era-relative audio** | Loudness/acousticness minus a shrunken fitting-population era mean |
   | 7 | **Training-distribution ablations** | Inverse-group-size loss weights; incomplete row only if time remains |

   Do not create one giant F3 containing every idea. Use names such as `F2_date_count`, `F2_date_te10`, `F2_interactions`. Advance only justified blocks.

   Monotone log transforms usually add little representational value to ordinary trees. Prioritize them for smooth, distance and neural models. Do not spend a large tree search on many arbitrary monotone transforms.

   **Exact-date signal is plausible, but album signal remains unverified.** Dates can be shared by unrelated releases. The new coverage probe suggests grouped CV retains a similar opportunity to transfer information across dates as the test set, though validation support is somewhat lower because each fold fits fewer tracks. That supports testing the feature under the existing track grouping. It does not justify grouping the primary CV by date.

   A date-held-out stress test would answer a different question, performance on unseen dates. If performed later, keep it diagnostic-only.

   **Leakage-safe target encoding:**

   For each outer training portion:

   1. Construct five inner stratified group folds using only its rows and labels.
   2. Pass the explicit local `(train_indices, validation_indices)` list to sklearn’s pinned `TargetEncoder`.
   3. Call `fit_transform` for outer-training features.
   4. Call the fitted encoder’s `transform` for outer validation and test.
   5. Preserve raw date strings, so `"1979"`, `"1979-06"` and full dates remain different categories.

   With fixed smoothing \(m\), the encoded value is:

   \[
   \operatorname{TE}(d)=
   \frac{\sum_{i:d_i=d}y_i+m\mu}{n_d+m},
   \]

   where every quantity comes from the appropriate inner fitting partition, or from the complete outer-training partition when transforming external rows. Unseen categories receive that fitting partition’s prior.

   Importantly, **sklearn 1.9 added support for explicit CV iterables and group metadata routing**. Older advice that `TargetEncoder` cannot do grouped cross-fitting is obsolete for this environment. Supplying explicit splits avoids introducing global metadata-routing settings. [TargetEncoder 1.9.1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)

   The same-group labels must be absent together, including conflicting-label duplicates. Never compute a global OOF date encoding once and reuse it inside another validation level.

   For frequency encoding, fit the dictionary on the permitted training rows only. A useful training representation subtracts the row’s own group from its date count; validation/test counts use the whole fitting dictionary. This reduces trivial self-inclusion differences without using labels. Record that convention explicitly.

   **Departure from strategy:** promote date frequency from D to a small scheduled experiment because it is cheap, its support can be audited, and it directly complements the target-encoding hypothesis. Do not promote identifier contents or automated feature search.

6. **Prefer small probability ensembles and evaluate deployment separately from model ranking.**

   Begin with the existing equal LightGBM/RF blend. The probe shows useful complementarity, but it does not warrant a fine weight search.

   Once models are confirmed, consider a small declared menu:

   - Best LightGBM, RF, CatBoost and one genuinely competitive challenger.
   - Equal LGB/RF and LGB/CatBoost averages.
   - Equal average of the three leading complementary families.
   - At most one or two declared 75/25 alternatives.

   Cap the final menu at eight predictor recipes. Do not enumerate every subset and every weight.

   Blend **probabilities**, separately for each repeat/fold. Recompute AUC from blended predictions. Never average member AUCs.

   Require identical row IDs, labels, groups, split seeds, folds and test IDs before blending. Record every parent bank’s hash, weights, and averaging order. Keep the original OOF-repeat axis.

   For finalists, use estimator seeds **0, 1, 2**, equally averaged. Average seeds **within a family first**, then apply family weights. Otherwise a family with three seed banks accidentally receives three times the weight of a family with one.

   Keep rank averaging excluded. Besides the strategy’s restriction on test ranks, probability averaging has a straightforward deployment interpretation. Calibration and a learned stacker add unnecessary nested procedures unless simple blends have clearly exhausted their value.

   **Default deployment:** retain the equal average of fold-model predictions. Ordinary single-fold OOF scores do not establish that this beats a full refit. Compare those deployment procedures in P2.7.

   Store a full-refit prediction vector in a distinct deployment artifact. Do not replace a bank’s `test_mean` while leaving its fold predictions unchanged. A deployment artifact should identify its recipe, fitting population, estimator seeds, parent evaluation evidence, predictions, and hash.

   The original submission validator assumes one globally constructed F0/F1/F2 frame. The phase 2 validator must handle learned fold-specific features and ensembles. Check duplicate consistency at each fitted member’s prediction step, then for the composite prediction. Retain the explicit check covering all 11 repeated test pairs.

7. **Make the restricted nested evaluation concrete and sufficiently small to finish Sunday.**

   Freeze no more than **five base recipes** and **eight predictor candidates** on Saturday. Include the original LightGBM F2 anchor. Other slots can hold tuned LightGBM, RF, CatBoost, and either a feature variant or a useful challenger. A blend references base recipes; it does not consume additional model fits.

   Use:

   - **Outer:** three `StratifiedGroupKFold` folds, one predeclared seed such as `20261101`.
   - **Inner:** five grouped folds with three repeats, seeds 155, 2026, 1029, rebuilt within each outer-training population.
   - The same complete-row evaluation population and primary groups.
   - All preprocessing and target encoding rebuilt at every required level.

   In each outer fold:

   1. Fit each frozen base recipe on the inner training partitions.
   2. Construct fresh inner OOF predictions and the declared blends.
   3. Select a candidate using the frozen selection rule.
   4. Predict the untouched outer validation rows with the average of that candidate’s inner fold models.
   5. Independently refit its constituent recipes on all outer-training rows and predict the same outer validation rows.
   6. Save both predictions and the selected recipe, fitting IDs, transformations, seeds, and stopping decisions.

   Step 4 directly evaluates the fold-average deployment procedure. Step 5 evaluates full refitting. For fixed-count trees, refitting is simple. For a neural recipe with early stopping, choose its refit epoch count using only the outer-training population.

   **Do not reuse the global phase 2 bank**, even for feature encodings or blend weights.

   A useful cost formula is:

   \[
   3 \times 15 \times \sum_j S_j
   \]

   base fits, where \(S_j\) is the number of estimator seeds in base recipe \(j\). Five recipes with three seeds each require at most **675 inner fits**, plus at most **45 full refits** if every base is refitted for comparison. Use measured family timings to project this before freezing. Restrict or remove an expensive challenger before the freeze if necessary.

   Report the three outer fold AUCs, their mean, pooled outer OOF AUC, selected candidates, and paired differences against the fixed baseline. Bootstrap the outer predictions by primary group if an interval is useful, retaining the existing conditional-uncertainty caveat.

   **Interpretation:** this evaluates a restricted selection procedure whose menu was informed by earlier exploration. It does not erase that earlier adaptation and does not estimate the benefit of rerunning the complete 100-configuration search.

   If nested evaluation disagrees with ordinary CV, first distinguish a common drop from smaller training populations from a reversal relative to the matched baseline. Do not respond by opening another search. A predeclared conservative rule could be:

   > If the selected procedure loses at least 0.003 mean AUC to the fixed baseline and loses on all three outer folds, recommend the baseline as the conservative candidate and retain the disagreement in the evidence.

   For deployment, keep fold averaging unless full refitting improves the paired outer mean by at least 0.003 with positive differences on all three outer folds. These are engineering rules, not significance tests. If the outer results are used for deployment selection, do not describe them afterward as an untouched estimate of the chosen final deployment.

8. **Defer interpretation outputs, but preserve the interfaces and evidence they will require.**

   Sam’s instruction means no report prose, Colab skeleton, SHAP analysis, importance ranking, or explanatory plot production during phase 2.

   Phase 2 should still preserve:

   - A predictor that accepts raw source columns.
   - Feature lineage from each derived feature to its original columns.
   - A reproducible way to rebuild or load every selected fold predictor.
   - Fold-fitted encoders and preprocessing state, or sufficient deterministic replay information.
   - Training/evaluation IDs and hashes.
   - Complete model/search/selection records and LLM integration evidence.

   These are development and reproducibility requirements. They also make the later source-column permutation analysis possible: permute one original column, rebuild its derivatives through the fitted pipeline, and predict without refitting.

   This is particularly important for date target encoding. Permuting the date later must apply the already-fitted date map; it must not learn a new map using evaluation labels.

   **Departure from strategy:** remove Sunday interpretation from §6 and move phase 2’s “ranked features with direction” output to phase 4. The master plan still contains that older wording and should be reconciled with Sam’s decision.

9. **Plan two phase 2 submissions, with an optional third only for a distinct purpose.**

   The already-reviewed P1.3 file should remain the initial upload. Its external pipeline check is pending; that does not prevent local development.

   | Submission | Timing | Purpose |
   |---|---|---|
   | Existing P1.3 LightGBM F2 | When Sam uploads | Establish that Kaggle accepts the schema and probability orientation |
   | Best materially improved, confirmed single recipe | Friday, if one exists | Exercise the phase 2 fitting and submission path |
   | Frozen validated blend | Saturday or Sunday, after its review | Exercise ensemble deployment |
   | Optional additional file | Only if warranted | A genuinely different, competitive frozen candidate or a deployment comparison already justified locally |

   If no new recipe is materially better, do not manufacture a submission just to meet the schedule. If a frozen candidate is already uploaded with identical bytes, reuse that submission rather than uploading it again.

   Each submission should have an explicit selection manifest listing exact run IDs and weights. `prepare_candidate` should consume that manifest, not search the whole experiment log for the current maximum.

   Record the intended purpose before Sam reports the public score. Public results may trigger investigation of an implementation or distribution mismatch. They should not drive a new weight grid, feature search, or repeated substitution among nearly tied models.

   This should keep the project near the strategy’s **four to six total uploads**, including final candidates. Ten per day remains a ceiling.

10. **Freeze a deterministic candidate-selection rule and explicit phase 2 exit criteria.**

   Keep the existing rule for allocating further engineering time:

   > At least +0.003 mean fold AUC, with positive mean differences on every repeat.

   Do not silently reinterpret that as either a significance threshold or a ban on retaining a small, cheap blend.

   For the frozen menu, I recommend this selection rule:

   1. Rank candidates by unrounded mean fold AUC.
   2. Treat candidates within **0.001** of the maximum as practically tied.
   3. Break those ties by fewer model families, then fewer supervised feature transformations, then lower fitting cost, then recipe ID.
   4. Apply the predeclared nested-evaluation fallback and deployment rule from item 7.

   The 0.001 band is a proposed concretization of “ties go to the simpler recipe.” It should be declared in the phase plan before the menu is evaluated, not selected after seeing which candidate it favors.

   Phase 2 exits when:

   - Mandatory tasks have converged; omitted optional tasks are explicitly recorded.
   - Final candidates have complete three-repeat banks on the common evaluation population.
   - Every learned transformation is demonstrably local to its permitted training population.
   - Banks, logs and submission recipes verify without writes.
   - The original phase 1 baseline and submission remain verifiable.
   - The restricted nested evaluation is complete, or its declared reduced fallback is documented honestly.
   - The candidate menu, selector, deployment rules, source revision, package lock and hashes are frozen.
   - The results are recorded as reviewed findings.

   **Saturday 18:00 PDT:** no new candidate families or feature hypotheses enter the menu.  
   **Sunday 18:00 PST:** no new model or parameter search.  
   **Monday:** reproduce, validate, prepare/upload candidates, and make final selections.

   For phase 3, propose:

   - **A:** the strongest stable candidate under the frozen rule.
   - **B:** a competitive alternative with demonstrated complementary errors.

   As a practical competitiveness screen, I would normally require B to be within about **0.005 ordinary-CV AUC** of A. This is a proposed engineering bound, not an uncertainty calculation. If no diverse model meets it, choose a closer strong alternative rather than sacrificing substantial validation performance for diversity.

11. **Address the likely formal-review blockers explicitly.**

   The current code is a solid phase 1 foundation, but several assumptions must not leak into phase 2 unchanged:

   | Risk | Required treatment |
   |---|---|
   | `make_features` is global and stateless | Learned features need a fitting interface; never construct global target encodings and pass them into the existing runner |
   | `run_cv` calls only `fit(X, y)` | New engine must route groups, weights and permitted inner splits explicitly |
   | Bank loading assumes five folds | New nested artifacts require explicit fold count and versioned validation |
   | Submission validation derives only F0/F1/F2 | Add recipe-aware validation for learned features, ensembles and refits |
   | P1.3 scans a log expecting three F2 runs | Freeze that log and selector; use explicit phase 2 selection manifests |
   | Source hashes change during development | Preserve old files; version changed recipes; never overwrite old banks or pretend new code is historical code |
   | CatBoost defaults can write files | Set `allow_writing_files=False`, including verification |
   | Library-internal validation can split tracks | Disable it or supply explicitly grouped inner partitions |
   | Search trials are selected on the same labels later used for comparisons | Report selection optimism and the limits of the restricted nested evaluation |
   | Date encoding is described as album identification | Call it exact-date association; album identity is unknown |
   | Training weights or incomplete-row treatment change score populations | Keep evaluation IDs identical for paired comparisons |
   | More model seeds accidentally change family weights | Average seeds within family before blending families |
   | GPU or library changes undermine replay | Pin the device/environment; establish repeatability before accepting a bank |

   New self-checks should directly test outer-label isolation, same-group exclusion in target encoding, nested fitting-ID containment, unseen dates, seed-subset alignment, ensemble identity cases, member-order invariance, and refit-versus-fold deployment validation.

   The biggest statistical risk is adaptive overfitting to a small dataset, not lack of another model family. Recent work still shows value in better search spaces, but those aggregate gains do not establish a benefit here. I would use that literature to motivate a few preprocessing and regularization choices, not to enlarge the search indefinitely. [Agentic Search Spaces for Tabular Machine Learning, September 2026 preprint](https://arxiv.org/abs/2609.16309)

   The main remaining uncertainties are **CatBoost/EBM runtime and runtime compatibility**, **whether exact-date encodings add anything beyond F2**, and **whether a neural model contributes enough to justify its confirmation and nested-evaluation cost**. Resolve these with bounded pilots.

Only Sam needs to decide:

- When to upload the reviewed files and report their public scores.
- Which two submissions receive final selection.
- Any policy exception, review escalation, or deadline-driven scope override that actually becomes necessary.
- Any change to the current prohibition on pretrained models or test-set fitting. This plan requires neither change.