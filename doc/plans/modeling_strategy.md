# Modeling strategy

Current extension, 2026-10-31: [P2.9: comprehensive development of other methods](phase2-p29-other-methods.md) adds neural, kernel, retrieval and statistical coverage after P2.7, on Sam's instructions. It is provisional until its R2 review. Existing course constraints and historical evidence remain in force.

v0.1, 2026-10-29. Owner: Sam Zhou. Status: approved in P0.2 after two review rounds (`../../reviews/2026-10-29-p0.2-modeling-strategy.md`) and committed 2026-10-29 (f29267b). Development was paused until Sam's go-ahead (Sam, 2026-10-29: "Okay, don't do development today. Yet. Just finish the plan and then pause.") and started the same day (phase 1, `phase1-data-baseline.md`). Phase 1 complete 2026-10-29. Amended in P2.0 with the phase 2 plan (`phase2-models-features.md` section 12; History), approved after two review rounds and committed (3e24fe6, df11816). Amended again in P2.0b on Sam's refinements (`phase2-models-features.md` section 14; History), approved in one review round and committed (ee557e2, 281a014). Amended in P2.8 (`phase2-models-features.md` section 16; History), in review.

This is the plan for building the best model under the course rules: the constraints, what the data shows so far, what is allowed, the validation protocol, the catalog of methods with priorities, the experiment order, and the decision rules. It covers phases 1 to 3 of `../project_plan.md` and feeds phase 4. It was built from a literature search and a scoping run by Claude, two consultation rounds with Codex (gpt-6-astra at reasoning effort xhigh, live web search; prompts and answers verbatim in `../llm_usage/2026-10-29-p0.2-consultation/`), and Sam's decision on pretrained models. Consultation advice is input to this plan, not approval of it.

Numbers marked "scoping" come from `../../analysis/scoping.py` (output `../../analysis/scoping_output.txt`); "Codex audit" marks the read-only checks of consultation round A. They set priorities here. Task P1.1 records each conclusion that later work relies on as a draft finding in `../findings.md`, with its code, row population, features, seeds and limitations, citing this evidence where the protocol is unchanged and rerunning it where the protocol changes.

## 1. Constraints

| Constraint | Consequence |
|---|---|
| Metric: ROC AUC on probabilities of High | Only the ranking matters. Fit with log loss, select on AUC. Map High to 1 explicitly; alphabetical class order would make Low the positive class |
| 3,864 training rows, 863 test rows | Small data: variance dominates. Strong regularization, repeated grouped CV, few and large decisions |
| Kaggle closes Monday 2026-11-02, 5:00 PM PST; 10 submissions a day; two final selections | About four working days from Sam's go-ahead. Submissions test stated hypotheses; they are not a search tool |
| Every task and every submission waits for a review round | Budgets use measured fitting, verification and review times, and the schedule reserves time for all three; no resource is assumed to be the bottleneck before it is measured (Sam's decision of 2026-10-29, `phase2-models-features.md` section 14) |
| Report (90 points) and Colab (10 points) due Tuesday 2026-11-03, 11:59 PM PST | Every experiment leaves evidence for the report (the experiment log, the findings); report and Colab work wait until development ends (Sam's decision of 2026-10-29) |
| Apple Silicon Mac, CPU only; Colab GPU available | Every scheduled model runs on CPU at this size; Colab is for the demo |
| Course rules (`../../CLAUDE.md`) and Sam's decision of 2026-10-29 (section 3) | Only models fitted from scratch on the provided training rows |

## 2. What the data shows so far

Scoping and the Codex audit, unreviewed until this task converges; P1.1 turns the relied-upon lines into findings.

| Observation | Value | Source |
|---|---|---|
| Target | `Popularity_Type` holds "High" and "Low" (the PDF says 1 and 0); 1,359 High, 2,505 Low, 35.2% High | Codex audit |
| Repeated tracks | 3,642 distinct `uri` values among the 3,863 rows with an identifier; 203 repeated groups (424 rows), at most 3 copies; features and date identical within a group; 32 groups with conflicting labels (66 rows) | Codex audit; scoping [1] |
| Repeated tracks in test | 11 pairs; no `uri` shared with train; two exact audio-only matches between a train and a test row, with different dates | Codex audit |
| Share of rows in repeated groups | 11.0% in train, 2.5% in test | Codex audit |
| Incomplete row | One training row has only a date (2024-09-08) and the label Low; the test file has no missing values | Codex audit |
| Dates | Train: 3,752 full dates, 111 year only, one year and month (1979-06). Test: 840 full, 23 year only | Codex audit |
| Identifiers | `track_href` and `analysis_url` end in the `uri` token; `type` is constant | Codex audit |
| Release era | High rate 1.00 before 1990 (134 rows), 0.50 in the 1990s, 0.52 in the 2000s, 0.42 in the 2010s, 0.25 in the 2020s | Codex audit |
| Univariate AUC (High positive) | loudness 0.68, energy 0.63, duration 0.61, valence 0.57; instrumentalness 0.33, acousticness 0.34, release year 0.36; the rest within 0.05 of 0.5 | Codex audit |
| Baselines: 5-fold `StratifiedGroupKFold` by `uri`, 3 repeats (split seeds 0, 1, 2), 13 audio features plus the scoping date fields (year, month, day, date precision, days since 1970 with a midyear fill for partial dates) | LightGBM with 600 trees, learning rate 0.02, 15 leaves 0.8268; random forest 0.8257; LightGBM default 0.8217; HistGB 0.8201; Extra Trees 0.8197; XGBoost default 0.8080; logistic regression on log features 0.7812. Naive SE about 0.003, which understates the uncertainty | scoping [2] |
| Value of the date | Audio only: 0.730 to 0.756. The scoping date fields add +0.05 (logistic) to +0.08 (trees) over audio only. In LightGBM on the same folds, the fields beyond the year add +0.0125 (audio + year 0.8143). Both gains are inside the 0.8268 baseline; further date engineering is unmeasured | scoping [2], [8] |
| Leakage through repeated tracks | Ungrouped folds scored +0.0158 higher for both random forest and LightGBM: observed protocol sensitivity, not a universal correction | scoping [3] |
| Train versus test | Release-year percentiles are broadly similar (median 2022 in both); the inspected marginal comparisons (Codex: KS on year and the 13 audio features) did not detect a large shift. This does not establish matching joint distributions or labeling. An earlier version of the scoping run also fitted a train-versus-test classifier; section 3 excludes that, the code was removed, and its result is not used | scoping [4] (descriptive); Codex audit |
| Row order and test ID | No marginal monotone association found by two checks (row-index AUC 0.506; correlation of test ID with year -0.03); nonlinear or conditional effects untested | scoping [5] |
| Exact release date as a shared category | 72% of test rows share an exact date with a train row (album identity is unknown). Its standalone out-of-fold target encoding (0.6311) is below the year alone (0.6405); its value next to audio and date features is unmeasured | scoping [6] |
| Public leaderboard noise | The standard error of one AUC near 0.83 would be 0.029 if 30% of the test rows are public and 0.022 at 50%. The public share is unknown, and these are not the uncertainty of a difference between two submissions | scoping [7] |

Reading: release date carries most of the signal beyond loudness and acoustic texture. One unverified hypothesis is collection: old songs that survive in a playlist-based sample tend to be popular. Train and test look drawn from a similar pool, so grouped random CV is the working proxy for the private leaderboard. The public leaderboard is a noisy secondary diagnostic.

## 3. What is allowed

Course rule (`../../Project_1.pdf` section 2): "you may only train and validate your models using the provided training set". Piazza ruling of 2026-10-28: every CSV column may be used, no external data. Sam's decision on pretrained models, 2026-10-29, verbatim: "pre-trained models certainly include data that is external to the project."

| Allowed | Excluded |
|---|---|
| Any model whose parameters are all fitted from scratch on the provided training rows: linear models, GAMs, trees, boosting, kernels, neighbors, neural networks from random initialization | Pretrained tabular foundation models and in-context learners of every kind, pretrained on synthetic or real data: TabPFN (2.x, 3, 3.5), TabICL and TabICLv2, TabDPT, Mitra, LimiX, EXAONE Tabular, TabFM, and the like |
| Open-source libraries, published algorithms, published default hyperparameters and static configuration lists (for example RealMLP's meta-tuned defaults): techniques, not learned weights | Anything learned on other data that transforms, teaches or selects: pretrained preprocessing, distillation from a pretrained teacher, meta-learned model selectors |
| LLMs as assistants for code, method choices and writing, disclosed in the report's LLM Usage section | Any pretrained model whose outputs enter the prediction pipeline: LLM-generated feature values, LLM guesses about songs, pretrained embeddings |
| Every column of `../../data/train.csv` and `../../data/test.csv` as model input, the release date and identifiers included | External data of any kind, lookups by identifier, existing code written for this Spotify dataset |
| Preprocessing fitted on training rows only, then applied to test rows; descriptive comparisons of the supplied feature distributions | Fitting any model or transformation on test rows, under the strict reading of the rule above: scalers, encoders, PCA or counts fitted on test features; pseudo-labeling and self-training; train-versus-test classifiers and domain weighting. Allowed only if the instructors say so, which this plan does not seek |
| AutoML only with an explicit whitelist of from-scratch models | AutoML presets that load excluded models: AutoGluon 1.5 `extreme`, and 1.6 `extreme` and `noncommercial` |

## 4. Validation protocol

1. Groups. Built without labels as connected components of two relations: the same nonmissing `uri`, and identical raw audio features plus the same release-date string. The incomplete row gets its own sentinel group. A stricter sensitivity grouping also joins exact audio-only matches. No approximate joins. These groups handle observable repeated tracks; independence between different tracks is an assumption, since artist and album are unknown.
2. Folds. `StratifiedGroupKFold`, 5 folds, split seeds 155, 2026 and 1029. The group mapping and fold ids are computed once (P1.2), saved, and reused by every model, so every comparison is paired. Class and era balance are checked per fold. Partitions are never chosen by model scores. Split seeds are distinct from estimator seeds.
3. Configurations run on all three split seeds. A subspace whose 15-fit run costs more than 120 s, timed before its scores, may be screened on split seed 155, with its finalists completed on all three; the neural challenger is screened that way (`phase2-models-features.md` sections 5.1 and 5.5). Several estimator seeds times three split repeats are for finalists only.
4. Everything learned from data is fitted inside the training part of each fold: imputation, scaling, encodings, feature selection, target encodings with inner group cross-fitting, neighbor statistics, early stopping. Early stopping uses an inner group-safe split of the outer training part, and the refit uses the chosen iteration count.
5. Internal splits inside libraries are audited: EBM bags (its defaults include a 15% validation subset and outer bags; pass group-safe bags or disable them), neural early stopping, SVM probability calibration, AutoML bagging and its `validation_structure` (AutoGluon 1.6), stacking, feature selection. A group-safe outer splitter does not make these group-safe; use explicit group-respecting inner assignments or disable the procedure.
6. Every model writes its out-of-fold predictions for every split seed and its test predictions to a bank with a fixed format, one file per recipe. Fixed ensembles are built from the bank. A learned combiner (constrained weights, a stacker) is fitted only on out-of-fold predictions rebuilt inside each training population, and the whole procedure is evaluated on held-out groups; a combiner is never fitted on the global bank and scored on it (`phase2-models-features.md` section 7.2).
7. Scores. Primary: mean fold AUC over the 15 folds. Secondary: pooled out-of-fold AUC per split seed. Diagnostics: AUC per era where both classes exist, row-weighted versus group-weighted AUC.
8. Paired comparison. For each of 2,000 bootstrap replicates, sample the training groups with replacement and give every row of a sampled group the group's multiplicity as its evaluation weight. Use the same multiplicities for both models and all three split repeats, keep the fold assignments, recompute each fold's weighted AUC and the paired difference in mean fold AUC. Report a percentile interval and flag replicates with a single-class fold. This is conditional uncertainty for fixed predictions: it excludes retraining variability, adaptive model selection, and dependence from overlapping training sets. It is not a private-leaderboard interval or a multiple-testing correction. The 15 fold scores are never bootstrapped as independent observations.
9. Allocation of further search. A gain of at least 0.003 overall, positive on all three repeats, is one route to further engineering time; phase 2 adds routes for repeat-stable smaller gains and for measured contribution to an ensemble (`phase2-models-features.md` section 5.6). These are practical thresholds for exploration, not statistical significance, and they are kept apart from final selection; the paired interval and the repeat-level results are reported with the decisions they drive. Ties go to the simpler recipe; for the phase 2 menu, the tie band and tie-breaks of `phase2-models-features.md` section 11.
10. Restricted nested evaluation (Sunday). Before it, freeze a small candidate menu and a deterministic selection rule. In each of three outer grouped folds, rebuild all inner predictions from the outer-training rows only; run feature and configuration selection, early stopping, calibration, blend weights and any stacker inside those rows; refit the selected pipeline on the outer-training rows and score the untouched outer fold. A learned combiner adds a third level: its meta-training predictions are rebuilt inside each inner training part. Never reuse the global out-of-fold bank inside this evaluation. Report it as an evaluation of that restricted selection procedure: the menu was informed by earlier exploration, so it does not remove all selection bias, and part of any drop from ordinary CV comes from smaller training sets. Use the small fixed menu, not a repeat of the full search. If even that is too expensive, evaluate the frozen recipes and state the limitation. The concrete design, the anchor fallback, the deployment comparison and the reduced path: `phase2-models-features.md` section 8.
11. Stress tests for the report, not for selection, run in phase 4 (Sam's decision of 2026-10-29): a chronological split (train through 2022, validate on 2023 and 2024, groups crossing the boundary removed), which measures extrapolation to newer releases since the release date is not the time popularity was measured; and the stricter grouping of step 1.
12. The public leaderboard. Its subset size, labels and paired uncertainty are unavailable. Public AUC is a noisy secondary diagnostic with no universal actionable difference threshold. Models are chosen on grouped validation. The purpose of each submission is recorded before its score is seen; a substantial disagreement with CV is investigated, without adapting repeatedly to the leaderboard.

## 5. Method catalog

Priority: A is scheduled, B is scheduled if time allows after A, D is documented but not scheduled before the close, X is excluded (section 3). Cost is for this data on the Mac.

### 5.1 Data preparation

| Item | Plan | Priority |
|---|---|---|
| Target | High = 1, Low = 0, explicitly | A |
| Constant and redundant columns | Drop `type`; keep `uri` only for grouping; drop `track_href` and `analysis_url`; P2.8 audits them and tests the identifiers (`phase2-models-features.md` section 16.2) | A |
| Incomplete row | Default: dropped from training. Ablation, with little time: keep it with native missing handling. The primary drop-versus-keep comparison scores the same 3,863 complete rows on the same folds; the keep variant uses the incomplete row only when its group is on the training side, and its held-out prediction is reported separately. A 3,863-row score is never compared with a 3,864-row score as an improvement | A (default), B (ablation) |
| Conflicting labels | Kept as observed; never relabeled by vote. They may reflect noise or unobserved context; no cause is asserted | A |
| Repeated tracks | Default row weights. Ablation: inverse group size, normalized to mean one within each training fold, as a distribution-sensitivity check, not a known correction. The metric stays row-weighted, as Kaggle scores rows | B |
| Label-quality diagnostics | Cross-fitted disagreement, to inspect, never to delete rows | D |

### 5.2 Features

Feature ladder, evaluated block by block:

| Set | Content |
|---|---|
| F0 | The 13 audio features |
| F1 | F0 + release year |
| F2 | F1 + every observed date component: month for full and year-month dates; day, day of year and weekday for full dates only; date precision as its own feature; days since 1970 for full dates, missing for partial dates (a midpoint fill is a separately named ablation, as in scoping) |
| F3 | F2 + a small documented set of transforms and interactions |

| Family | Plan | Priority |
|---|---|---|
| Date parts and era | As in F1 and F2; coarse eras or year splines for linear models | A |
| Feature stage of phase 2 | Screening, four declared cross-family date checks, preprocessing alternatives, fixed removal checks through drop lists, and re-tuning of up to four model-and-feature combinations (`phase2-models-features.md` section 6) | A |
| Skewed audio features | For linear, kernel and neural models: log1p(duration_ms / 1000), and raw versus log1p(100 x) for speechiness, liveness, instrumentalness and acousticness, plus a zero indicator for instrumentalness; quantile or robust scaling as an alternative. Imputation either in raw units before the transform or with statistics of the transformed column, always fitted in the fold | A |
| Musical categories | key, mode and time signature one-hot for linear and distance models; cyclic or circle-of-fifths key; key x mode. The `music` block (circle-of-fifths key and tempo octave, carried over from an earlier audio-only attempt, `../prior_project/`) is tested in P2.8 and on CatBoost and EBM in P2.4 (`phase2-models-features.md` section 16.3) | B |
| Interactions | year x loudness, year x acousticness, energy x loudness, energy x danceability, acousticness x instrumentalness, beat count = tempo x duration_ms / 60000 | B |
| Era-relative audio | Loudness and acousticness minus a shrunken era mean, fitted in folds | B |
| Target encoding | Of year, era or exact date, with inner group cross-fitting | B |
| Exact-date group count | The number of distinct training groups sharing a raw date string, fitted in the fold; scheduled in phase 2 (`phase2-models-features.md` section 6) | B |
| Other frequency encodings | Counts of a bin within the training rows only | D |
| Neighborhood features | kNN distances and out-of-fold neighbor label means within training folds, excluding the row's own group | D |
| Automated feature search | Polynomial screening or feature selection inside folds | D |
| Identifier contents | The bounded menu of P2.8 (`phase2-models-features.md` section 16.2): token shape, fold-fitted fragment counts and target encodings, an identifier-only linear model, with pseudotoken controls; anything further only after a reviewed positive result. Run in P2.8: no test met its follow-up rule (Finding 015) | B |
| Test-set statistics, external metadata, LLM-generated features | | X |

### 5.3 Models

| Family | Why here | Cost | Priority |
|---|---|---|---|
| Random forest | Close to the best boosting recipe in scoping; different errors from boosting; leaf size 2 to 30, 500 to 1,000 trees; no out-of-bag scores (repeats leak) | Minutes | A |
| LightGBM | Principal contender | Seconds per fit | A |
| CatBoost | Strong on small tables; Plain and Ordered boosting, depth 3 to 7 | Minutes | A |
| Smooth models: spline logistic regression and EBM (`interpret`) | Smooth shapes, ensemble diversity; EBM shape functions are report figures | Seconds to minutes | A |
| Extra Trees | About 0.007 below the best in scoping; diversity; a real search in phase 2 | Minutes | A |
| XGBoost (`hist`) | Regularized alternative and diversity; a real search in phase 2 | Seconds | A |
| sklearn HistGradientBoosting | Low-dependency control; its random internal validation disabled; four OpenMP threads | Seconds | B |
| One neural challenger from scratch: TabM, its budget set from its measured cost (`phase2-models-features.md` section 5.5) | Best current tabular networks; possible complementary errors; screened on split seed 155 | Minutes to hours on CPU after a timing pilot | A |
| kNN | Local neighborhoods, with the date weighted explicitly in the distance (`phase2-models-features.md` section 6.2) | Seconds | A |
| RBF SVM | Smooth boundary | Minutes | B |
| ModernNCA, TabR, FT-Transformer, ResNet, TabNet, NODE, SAINT | Easy to overfit at this size; little support for gains over TabM | Hours | D |
| BART, exact Gaussian-process classification | Extra dependencies and tuning; exact GPs scale poorly | Tens of minutes to hours | D |
| Naive Bayes, LDA, QDA, AdaBoost, RuleFit | Cheap diagnostics | Seconds | D |
| AutoML (AutoGluon, FLAML) with a whitelist | AutoGluon 1.6 requires `xgboost>=2.0,<3.4` against the pinned 3.4.1; broad AutoML comparisons are not worth the setup before the close | 1 to 2 hours plus environment | D |
| Pretrained tabular foundation models | | | X |

### 5.4 Tuning

- Seeded random search in batches: initial batches, expansion batches and ceilings per family set from measured costs, with expansion and stopping rules; local random refinement of at most four qualifying families; Optuna only through a reviewed installation, if a family stays productive. Fixed folds across trials. Failed trials are logged. The spaces, budgets and rules: `phase2-models-features.md` sections 5.1, 5.2 and 5.6, which replace the search regions of Codex round A and the budgets of P2.0.
- Boosted trees: fixed iteration counts as hyperparameters, with no early stopping. Neural networks: inner group-safe early stopping (section 4, step 4), then a refit for the chosen epoch count.
- Class weights: none by default; a modest positive weight only as an ablation. SMOTE, undersampling, focal loss and AUC surrogate losses are D.

### 5.5 Ensembling

1. Equal probability averages of complementary models from the bank, tried as soon as two models exist; in phase 2, over the declared pool of `phase2-models-features.md` section 7.1.
2. A small predeclared set of fixed weights, including contribution blends of a challenger with a reference predictor, judged on blended AUC on identical folds, so a challenger is not discarded for a lower standalone AUC alone.
3. Constrained learned weights (nonnegative, summing to one, mildly shrunk toward equal weights) and one L2 logistic stacker on member logits, each fitted on rebuilt inner out-of-fold predictions and evaluated as a whole procedure (section 4, step 6; `phase2-models-features.md` section 7). Within-family averages of a few close configurations. Hill-climbing selection and rank averaging remain D (rank averaging would also use test-set ranks).
4. Seed averaging of the members.
5. Deployment, predeclared per recipe before its final evaluation: the average of the fold models, or a refit on all training rows with the iteration count from CV. The single-estimator out-of-fold score does not compare the two by itself.

### 5.6 Interpretation for the report

Run in phase 4, after the Kaggle close (Sam's decision of 2026-10-29); phase 2 keeps what it needs (`phase2-models-features.md` section 9).

- Importance: on each held-out fold, permute a raw source column, rebuild every feature derived from it through the fitted fold pipeline without refitting, predict again, and rank columns by the decrease in held-out AUC, with repeated permutations and the spread across folds. The top 10 are 10 original input columns, so date parts never fill several places.
- Direction: model-agnostic permutation importance and ALE or dependence plots for the final predictor; TreeExplainer SHAP only for supported tree components, saying which component it explains; EBM shapes explain the EBM, not an ensemble. Directions are positive, negative, nonmonotonic or conditional, with the supported feature range and no causal reading.
- Overlap: source-column importances overlap and do not add up; correlated columns (loudness, energy, acousticness) are discussed together.
- The marginal directions in section 2 are hypotheses, not results.

## 6. Experiment plan

Every task is one reviewed unit under `../review_policy.md`, with its own record. An installation beyond `requirements.txt` (`catboost`, `optuna`, `interpret`, `shap`, `pytabkit` or `tabm` with `torch`) is reviewed before it is executed, with its exact command and dependency changes (`../review_policy.md` section 4.2). The baseline environment is preserved: a package with incompatible requirements is dropped (in phase 2, by `phase2-models-features.md` section 4.9; no isolated environment), and results affected by changed dependencies are revalidated. Submissions are gated actions, each reviewed with its file hash and description.

Day 1 is the day of Sam's go-ahead. The author-time caps of this table exclude review turnaround (phase 1 guidance); the phase 2 targets include review (`phase2-models-features.md`, Tasks).

| When | Tasks | Work | Cap and exit |
|---|---|---|---|
| Day 1 (Thursday evening or Friday morning) | P1.1, P1.2, P1.3 | P1.1: section 2 as draft findings. P1.2: the harness (loading, target map, groups, saved folds, features F0 to F2, the out-of-fold bank, metrics with the paired bootstrap, the submission writer with checks, the experiment log); first reproduce the scoping result with its exact rows, groups, features and seeds, then establish the baselines (LightGBM, random forest, logistic) under the production protocol; differences between the two protocols need not be noise. P1.3: the first submission | 2 to 3 hours. P1.3 exit: the submission is accepted; schema, orientation, prediction distribution and provenance are verified; the public score is recorded, with no fixed tolerance |
| Thursday afternoon to Sunday (phase 1 finished early) | P2.1, P2.0b, P2.2, P2.8, P2.3 to P2.5, P2.6a, P2.6b, P2.7 | `phase2-models-features.md`: infrastructure; the amendment; broad searches with the installed packages; the gap hunt (identifier probes, feature checks moved from P2.4, deterministic forest prediction); CatBoost and EBM; the feature stage and re-tuning; the neural challenger; the learned-procedure engine and nested evaluator; ensembles and the frozen menu; the restricted nested evaluation. No new model-family search on Sunday | Menu frozen Saturday 18:00 PDT; final freeze Sunday 18:00 PST |
| Monday, by 2 PM PST | P3.1 | Final refits and checks, two candidates proposed with their evidence, uploaded. No new feature or hyperparameter search | Uploaded by 2 PM PST |
| Monday, by 3 PM PST | P3.2 | Sam selects the two final submissions on Kaggle | Selected before 5 PM PST |
| Monday after the close, and Tuesday | P4.1 to P4.3 | Interpretation (section 5.6) and the stress tests (section 4, step 11); the Colab notebook run from a clean runtime; the report from the template; the Colab made public read-only; the Piazza note; the Gradescope upload | By Tuesday 11:59 PM PST |

Day 1 did not slip. Phase 2's cut order is in `phase2-models-features.md`, Tasks.

Submissions, about 4 to 6 in total: the first working baseline, a materially improved incumbent, a validated blend if it helps, and the final candidates. More only for a concrete unresolved question. The ceiling of 10 a day is never the target.

Documented but not scheduled before the close: everything marked D in section 5.

## 7. Ten highest-value ideas

Measured effects and speculative planning ranges are kept apart. Speculative ranges are rough scenarios, not confidence intervals or guarantees; losses outside them are possible. Effects are not additive. Comparators are named.

| Rank | Idea | Effect | Basis |
|---|---|---|---|
| 1 | Grouped, leakage-safe validation with a shared out-of-fold bank | Ungrouped CV scored +0.0158 higher for both tested models: observed protocol sensitivity, not a universal correction | Measured, scoping [3] |
| 2 | Release date as features | +0.05 to +0.08 over audio only; +0.0125 for the scoping date fields beyond year in LightGBM; further date gains unknown | Measured, scoping [2], [8] |
| 3 | Regularized, tuned boosting, CatBoost included | -0.005 to +0.010 against the 0.8268 scoping recipe on a matched protocol; +0.015 would be an optimistic outcome | Speculative |
| 4 | Small ensemble of 2 to 4 complementary models, seed averaged | -0.003 to +0.010 against its strongest member; more requires unusually complementary errors | Speculative |
| 5 | Tuned random forest and Extra Trees | -0.003 to +0.008 against each family's own baseline; whether either beats the incumbent is reported separately | Speculative |
| 6 | One smooth model (spline logistic or EBM) | -0.003 to +0.005 when added to the incumbent blend; standalone unmeasured; EBM shapes serve the report | Speculative |
| 7 | Normalized inverse-group-size weights; conflicts kept | -0.005 to +0.005 against the same unweighted recipe | Speculative |
| 8 | Further engineered features (F3) | -0.005 to +0.010 beyond the F2 baseline; smaller for trees | Speculative |
| 9 | One neural challenger trained from scratch | Standalone gain unknown and possibly clearly negative; budgeted for a complementary contribution of about 0 to +0.005, with no help a likely outcome | Speculative |
| 10 | Stop searching Sunday evening; spend Monday and Tuesday on confirmation, reproducibility and explanation | Protects the final choice and the 100 graded points | Process |

## 8. Submissions and the final selection

- Checks before every upload: 863 rows, the sample's column names and ID order, unique IDs, finite probabilities in [0, 1], High as the positive class, identical predictions for identical rows. The file is hashed; the recipe, versions, seeds and purpose are logged.
- Final selection A: the strongest stable validated recipe, a single model, a fixed ensemble or a learned ensemble procedure.
- Final selection B: a competitive alternative with evidenced complementary errors. Diversity alone does not justify a material validation loss.
- Both are chosen on grouped CV and the restricted nested evaluation, not on the public leaderboard. The stress tests are reported, not used for selection (section 4, step 11). Sam makes the choice.

## 9. Evidence for the report and the Colab

| Report section (template page limits) | Evidence produced on the way |
|---|---|
| Introduction (at most 1 page) | Team, Kaggle name, final rank and AUC, Colab and Piazza links, division of labor (Sam to supply) |
| Overview (half a page recommended, at most 1) | The experiment log (every recipe, its features, CV scores, public score, decision); the dated task history in `../../reviews/` |
| Approach (at most 1) | The findings; the feature ladder ablation; duplicate and incomplete-row handling; why each family was tried, kept or dropped |
| Model Selection (at most 1) | Section 4; log loss for fitting and AUC for selection; the CV table; the nested evaluation and stress tests; CV versus the public leaderboard |
| Conclusion (at most 1) | Top 10 original features with directions (section 5.6); what worked; challenges |
| Additional insights (optional) | Why AUC: population ROC AUC does not change with prevalence when the class-conditional score distributions stay fixed, while training behavior, finite-sample uncertainty and distribution changes can still depend on class balance; what parallelizes (folds, trees, trials, ensemble members); data worth collecting (artist, playlist and stream histories, the date popularity was measured) |
| LLM Usage | For every LLM-assisted component: the model identity from session evidence, the relevant prompt and adopted response excerpt, and how the response was integrated or changed; authoring (Claude Code), consultation and formal review (Codex) kept apart. The consultant's model comes from its session log, not from the reviewer script's configuration. Sources: `../llm_usage/` and `../../reviews/` |

The full experiment log stays outside the page-limited narrative. The Colab installs pinned packages, reads only the three provided CSVs, rebuilds the features and folds, reproduces the frozen recipe within a stated tolerance, draws several report figures and writes a checked submission. Before delivery it is run from a clean runtime, and its public read-only access is checked. The full search stays out of it.

## 10. Open questions

1. Team members and the Kaggle team name, for the report and the division of labor (Sam).
2. Review cost under the deadline: Sam may waive review for a named task (`../review_policy.md` section 3.3), for example a routine resubmission of an already reviewed recipe.
3. Whether the boundary in section 3 (published defaults and static configuration lists allowed; anything learned on other data that transforms, teaches, selects or predicts excluded) is how Sam reads the decision.

## Sources

Checked on 2026-10-29. "C" checked by Claude, "X" reported by Codex in the consultation with its own live search.

| Source | Use here |
|---|---|
| TabArena, NeurIPS 2025 Datasets and Benchmarks, `https://arxiv.org/abs/2506.16791` (C) | Trees, tuned networks and ensembles all competitive; validation overfitting in ensembles |
| TALENT, JMLR 26 (2025), `https://jmlr.org/papers/v26/25-0512.html` (X) | Unified implementations of tabular deep models |
| RealMLP, NeurIPS 2024, `https://arxiv.org/abs/2407.04491` (C) | Strong defaults for an MLP and for boosting |
| TabM, ICLR 2025, `https://arxiv.org/abs/2410.24210` (C) | Parameter-efficient MLP ensembles, the preferred neural challenger |
| ModernNCA, ICLR 2025, `https://proceedings.iclr.cc/paper_files/paper/2025/file/ecd294194d5155432aab533f7ce0f6fe-Paper-Conference.pdf` (X) | Learned neighborhood classifier, documented but not scheduled |
| The Kaggle Grandmasters Playbook, NVIDIA, 2025-09-18, `https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/` (C) | Group-aware CV, diverse baselines, blending, seed averaging, refit on all data |
| Stacking with cuML, NVIDIA, 2025, `https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/` (C) | Stacking from out-of-fold predictions, at a scale this plan does not attempt |
| Kaggle Playground S5E10 14th-place write-up, 2025-10-31, `https://www.kaggle.com/competitions/playground-series-s5e10/writeups/s5e10-14th-place-solution` (X) | Diverse out-of-fold models and ridge stacking; overfitting from aggressive optimization |
| Position: State-of-the-Art Claims Require State-of-the-Art Evidence, 2026-05, `https://arxiv.org/abs/2605.17273` (C) | Aggregate benchmark wins are fragile: measure on our own CV |
| LightGBM parameters `https://lightgbm.readthedocs.io/en/stable/Parameters.html`; CatBoost parameters `https://catboost.ai/docs/en/references/training-parameters/common` (X) | Tuning regions and bagging settings |
| InterpretML `ExplainableBoostingClassifier` API, `https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html` (X) | EBM internal validation and bags |
| AutoGluon 1.6 release notes `https://auto.gluon.ai/dev/whats_new/v1.6.0.html` (C: presets, `validation_structure`, `xgboost>=2.0,<3.4`); 1.5 notes `https://auto.gluon.ai/dev/whats_new/v1.5.0.html` (C); TabularPredictor API `https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html` (X) | Presets that load excluded models; the dependency conflict; the meaning of `groups` |
| TabPFN-2.5 `https://arxiv.org/abs/2511.08667`, TabPFN-3 `https://arxiv.org/pdf/2605.13986`, TabPFN-3.5 `https://arxiv.org/abs/2609.17895`, TabICLv2 `https://arxiv.org/abs/2602.11139` (C); Mitra-v2 `https://arxiv.org/abs/2609.04540`, EXAONE Tabular `https://arxiv.org/abs/2608.25774`, LimiX-2 `https://arxiv.org/abs/2609.17488` (X) | The pretrained models excluded in section 3, cited for the record |

## History

- v0.1, 2026-10-29: created with P0.2. Status line updated with P1.1; content unchanged. With P1.3, Sam's decision, verbatim: "No, from now on, don't worry about the collab or the report itself until the very end. Focus on development." Sections 1 and 6 changed accordingly: no report or Colab work before development ends.
- v0.1 amended, 2026-10-29, in P2.0, the amendments of `phase2-models-features.md` section 12: status line; section 4, step 9, a pointer to the phase 2 tie band; section 4, step 10, a pointer to the concrete nested design; section 4, step 11 and section 5.6, the stress tests and interpretation run in phase 4 under Sam's decision quoted above; section 5.2, the exact-date group count split from the other frequency encodings and promoted from D to B; section 5.4, 104 classical configurations on all three split seeds in place of the split-seed-155 search and its regions, Optuna only through a reviewed installation, and fixed iteration counts for boosted trees in place of inner early stopping; section 6, the Friday to Sunday rows replaced by the phase 2 plan, which starts Thursday, the Day 1 slip paragraph replaced, and an incompatible package dropped rather than isolated.
- v0.1 amended, 2026-10-29, in P2.0b, the amendments of `phase2-models-features.md` section 14, on Sam's refinements, verbatim in part: "This supersedes my earlier instruction to preserve the existing experiment budgets." Status line; section 1, measured budgeting in place of the assumed review bottleneck; section 4, steps 3, 6, 9 and 10, screening by measured cost, learned combiners fitted on rebuilt predictions, the allocation routes for exploration, the third nested level; section 5.2, the feature stage; section 5.3, real allocations for Extra Trees, XGBoost, the smooth models, kNN and TabM; section 5.4, batches, ceilings and stopping rules; section 5.5, contribution blends, constrained weights, one stacker and within-family averages; section 6, the re-cut tasks; section 8, learned procedures as finalists.
- v0.1 amended, 2026-10-30, in P2.8, the amendments of `phase2-models-features.md` section 16, on Sam's instructions, verbatim in part: "Prioritise gap hunt" and "Take what you can get out of the old project and then update your plan and your current working, and from then on just don't depend on the old project anymore." Status line; section 5.1, the identifier audit; section 5.2, the `music` block in the musical-categories row and identifier contents promoted from D to B as P2.8's bounded menu; section 6, P2.8 in the phase 2 row.
