# P2.9: comprehensive development of other methods

Version 1, 2026-10-31. Owner: Sam Zhou. Author: Codex. Status: provisional, awaiting R2 method review by Claude Fable 5.1 at xhigh. This is a plan, not a training launch or a claim of improved performance.

## 1. Scope and authority

Sam: "Can you extend a plan to do the other methods as well? Be comprehensive this time."

The objective is to improve validated ranking and useful ensemble diversity beyond the current tree ensemble. Cover different representations and training mechanisms before concluding that a family offers little. The goal is overall performance; no target public score is promised.

This amendment governs P2.9. It supersedes the old phase 2 prohibition on reopening exploration and its expired Saturday search cutoff, following Sam's decisions of 2026-10-31. The completed P2.6b menu and P2.7 remain historical baselines. This plan does not change their results, recipes, selection records or review status. It preserves the existing final selection tie rule, course constraints, owner control of uploads and final selections, and R1/R2/R3 review triggers. A new menu receives its own evaluation. Report and Colab development remain deferred until model development ends.

This request authorizes planning and its review. Implementation can proceed provisionally when directed; consequential method changes need R2 before dependent tuning. Large compute launches need Sam's approval of the measured schedule under AGENTS.md. No execution budget is inferred from this document's trial counts.

## 2. Starting evidence and comparison references

| Item | Existing evidence | Consequence for this plan |
|---|---|---|
| Trees | P2.2: 88 LightGBM, 48 random forest, 48 XGBoost, 56 Extra Trees and 12 HistGB configurations. P2.3: 48 CatBoost. P2.4: 128 further tree configurations and 96 feature comparisons (Findings 011, 019, 021 to 023) | Preserve the strong baseline; broad tree retuning is not the main new allocation |
| Smooth and distance models | P2.2: 48 logistic and 48 kNN configurations; later feature/contribution checks (Findings 011, 012, 022) | Test new representations and mechanisms, not another copy of those grids |
| TabM | Eight configurations, best 0.8034 CV, best blend with H 0.8320. No numerical embeddings; fitted date blocks excluded. Ordinary F2 calendar fields were present (Finding 025) | The tested setup was weak; numerical embeddings, architecture and feature availability remain separate questions |
| E, current ensemble | Equal average of four channels: three-recipe LightGBM average, XGBoost, CatBoost with music, Extra Trees with interactions. Mean fold AUC about 0.8411 (Finding 028) | Every new candidate gets a contribution comparison against E, not only the weaker historical H |
| H, original anchor | LightGBM F2, estimator seed 0, mean fold AUC about 0.8315 (Findings 007, 009) | Retain as a pipeline reproduction check and an interpretable historical baseline |
| P2.7 | Restricted nested mean 0.8398 against H 0.8306; fold averaging retained; build/replay agree (Finding 030, draft) | Reliability evidence for that menu, not an evaluation of this expanded search |
| Existing omitted nominees | Small exploratory contributions to E in the AUC handoff | Reuse as comparison controls; do not declare all kernels or neural retrieval equivalent to old kNN |

E is `p2_blend_eq_all_seeds155-2026-1029_375767dd7ee9`. Its constituent recipes, seeds and averaging order come from `configs/phase2/menu.json` and `configs/phase2/p2.6/procedures.json`. Resolve H from the final-source P2.6b menu. Record full bank, recipe, data, split and source hashes in the P2.9 reference manifest before runs. Prior scores above are rounded for orientation; computations use saved probabilities and unrounded scores. Findings 021 to 025 and 030 remain draft until their applicable release review.

## 3. Deliverables and task dependencies

| Task | Complete result | Depends on |
|---|---|---|
| P2.9a | Reviewed protocol, source audit, package-lock proposal, feature and search manifests, score-free cost pilots, measured launch proposal | This R2 plan; existing handoffs |
| P2.9b | Raw-row fitting adapters for neural stopping, kernel calibration and retrieval; rejection/isolation checks; historical baseline reproduction | P2.9a method contract |
| P2.9c | Kernel and statistical families, complete initial coverage and logged contribution checks | P2.9b relevant adapters |
| P2.9d | TabM, MLP, ResNet and FT-Transformer coverage, feature parity and optimization diagnostics | P2.9b neural adapter |
| P2.9e | TabR, metric-learning neighbours, and their own-group exclusion checks | P2.9b retrieval adapter |
| P2.9f | Hierarchical date and duration-rounding comparisons; nomination, seed confirmation and refinements | Safe features; corresponding family initial batches |
| P2.9g | Frozen expanded menu, complete production evaluation, compute reservation and R2 freeze review | Eligible completed candidates and final-source replay |
| P2.9h | Revised restricted nested evaluation, empty-cache replay, decision receipt and phase 3 handoff | P2.9g reviewed menu and approved launch |

P2.9c/d/e are logically independent after their adapters pass. The primary agent owns STATE and the plan. Separate processes may run only from fixed sources and within measured aggregate resource limits. This document does not require spawning helper agents. No package installation races another installation or running job.

## 4. Shared validation and reporting contract

Use the unchanged 3,863 eligible training rows, primary track groups, and saved five folds at split seeds 155, 2026, 1029. Keep the incomplete-row policy unchanged. All headline discovery results require all 15 folds. A failed fold is not silently dropped. A one-fold or small-population pilot establishes cost or correctness only; its scores do not nominate, reject or rank candidates.

For every completed configuration store standalone mean fold AUC, the three repeat means, fold predictions, training/stopping receipts, elapsed time and peak memory. With native probabilities or the calibrated probability procedure of section 6, evaluate the three fixed blends `(1-w) E + w candidate`, w in {0.10, 0.25, 0.50}, on exactly matching rows/folds. Record all three, even for a poor standalone model. Decision-score AUC is an additional diagnostic for margin models; the evaluated probability procedure is the candidate for blending and deployment.

Report paired differences against E, and against the family/feature control where appropriate. For decision-driving comparisons run the existing group bootstrap with 2,000 resamples, its saved seed and valid-resample count; intervals are conditional on fixed predictions. Repeated folds are correlated. Three repeats and multiple initialization seeds are stability checks, not independent datasets. Singleton-row AUC, era slices, date-coverage strata, log loss, Brier score and prediction correlation are diagnostics; none changes the selection metric. Do not adopt the inferred test-construction story as a proved fact.

The public leaderboard never determines trial expansion, features, weights or seeds. All labels have already been explored. New partition seeds and the new nested evaluation cannot erase whole-search selection optimism. Report the production search and conditional nested results separately, retaining every trial and allocation decision.

## 5. Comparable information and representation experiments

Feature names below are new P2.9 protocol arms, not replacements for historical F1/F2.

| Arm | Information | Required comparison |
|---|---|---|
| B | Existing F2 audio and calendar fields, exact existing definitions | Matched architecture controls; replay original TabM recipe |
| C | B with cyclic calendar/music representations; raw fields retained initially | Compare otherwise identical B/C recipes for smooth, kernel and neural families |
| D | B plus existing `date_count` and `date_te3` | A minimum feature-parity arm for each major family, built inside every relevant split |
| HIER | D with the exact-date prior shrunk through month and year, as defined below | Matched with D; it differs from adding another year-month column |
| ROUND | A binary `duration_ms % 1000 == 0` flag | Matched addition to E's strongest atomic LightGBM and to nominated non-tree recipes |

Fit imputation, log/robust/standard/quantile transformations, one-hot vocabularies, PCA, embedding bins and all data-derived transforms on the current permitted training population. Unknown categories have explicit handling. Missing dates retain precision/missing indicators. PLE bins use only permitted training feature values initially; any supervised binning is a separate reviewed recipe. Default bins apply to continuous measurements and continuous calendar coordinates, not indiscriminately to one-hot categories. Neural categorical embeddings versus fixed one-hot encoding are a declared ablation; no identifier vocabulary lookup outside the provided CSVs.

B/D coverage cannot be replaced by observing that a tree rejected a feature. For each major neural family, a minimum of four architecture/optimizer configurations is evaluated as matched B/D pairs. The kernel and retrieval family manifests have analogous pairs. C is tested at matched settings before a smooth/local family is declared adequately explored. Where a mechanism makes an arm unsafe or infeasible, record that arm as incomplete and the exact obstacle; do not label the family exhausted.

HIER formula, fitted only on each encoder-training partition: global prior p; year mean `(sum_y + m_year*p)/(n_year+m_year)`; month-within-year mean `(sum_month + m_month*p_year)/(n_month+m_month)`; exact-date mean `(sum_date + m_date*p_month)/(n_date+m_date)`. Counts/sums are row-weighted, as in current date TE. Unseen levels back off to the next observed parent. A partial date terminates at its observed precision; missing date backs off to p. Training features use the existing five group-held-out encoder folds, with the entire hierarchy rebuilt inside each fold. Initial tuples `(m_year,m_month,m_date)` are (30,10,3), (100,30,3), (30,10,10), (100,30,10). Counts remain distinct-group counts under the existing DateCount definition. Pair each HIER configuration with identical D hyperparameters.

Feature experiment allocation: 16 HIER comparisons (four tuples on four recipes: the strongest atomic LightGBM, best TabM, best SVM, best other neural recipe under the nomination rule); up to eight ROUND comparisons on the strongest atomic LightGBM and up to seven nominated non-tree recipes. If a named family is unfinished, its comparison remains unfinished. These 24 comparisons are separate from family initial counts. Additional combinations require a predeclared refinement batch. No duplicate-pair boosting, pseudo-labels or fitting on test features is included.

## 6. Isolation contracts that must precede tuning

### 6.1 Neural stopping and final refitting

Each fit receives only raw training rows D, labels, primary groups and a deterministic context. Choose one validation part from five stratified group folds of D using a fixed context seed. On S, the stopping-training part, rebuild feature transformations, including encoder cross-fitting and embedding bins. Transform V, the stopping-validation part, with S's state only. Optimize on S; choose the epoch using V. The outer validation rows are unavailable.

Discard the stopping model, optimizer, preprocessing, encoders, bins and retrieval memory. Rebuild all transforms on D; cross-fit D's supervised training features; initialize a fresh model and refit for the chosen epoch count. The exact epoch-to-update policy is saved. The default keeps the existing epoch-based refit policy and reports the larger final update count. A step-based variant is a separate recipe. Predictions on outer validation/query rows use only the state fitted on D.

This requires a new raw-row adapter. Passing D-wide precomputed target encodings to the old TabM stopping wrapper is forbidden. Preserve the old adapter for historical recipes. Apply the same contract inside calibration, stacking, and the revised nested evaluator. Receipts identify row sets, groups, transform hashes, stopping losses, chosen epoch and final updates. Structural receipt validation is implemented, including the deferred P2.5 receipt check.

### 6.2 Margin-model calibration

Disable SVC's internal probability fitting. Inside D, build explicit five-fold grouped calibration splits. For each split rebuild the complete base recipe, including all features and supervised encoders, on its calibration-training part; predict raw class-1 margins on its held-out part. Fit a fixed regularized sigmoid to those out-of-fold margins and D's labels. Refit the complete base recipe on D and apply that sigmoid to query margins. The resulting training-size mismatch is assessed by evaluating the entire procedure on outer folds. Initial sigmoid: L2 logistic regression on one margin feature, C=1, solver=lbfgs, tol=1e-8, max_iter=1000, unpenalized intercept, no class weights. Standardize the margin using only the calibration out-of-fold training margins and save that scaler. Record the slope and class orientation; reject nonconvergence. No isotonic search in the initial batch.

Use the same procedure for kernel-ridge scores and linear SVM when they enter probability blending. No global out-of-fold calibrator may be scored on the rows used to fit it. A fitted outer-training feature matrix must not be reused across calibration folds. Include the five calibration fits plus final fit in timing and recipe identity. Native probability models use their native probabilities initially. Never normalize scores using test or validation score distributions.

### 6.3 Retrieval and metric learning

Memory contains only the current permitted training population. During training, exclude every row of the query's primary group from retrieved candidates, not just the exact row. Stopping queries retrieve only S; outer queries retrieve only D. Fitted scaling, NCA/PCA, neighbour indices and neural retrieval parameters are rebuilt within their training boundary. When training query features or retrieved memory features use target encodings, exclude the query's whole group from every encoding dictionary that could reveal its label through the memory. A concrete safe implementation is group-held-out query blocks with their reference memory and all encoders fitted on the complementary groups. Freeze that policy, its inference memory rule and measured cost before the D-arm run. An unsafe shortcut is not an admissible D-arm.

A label-permutation or own-group test must detect training self-retrieval. Also test the indirect target-encoding route. Reject query rows or group IDs injected into a memory manifest. Record candidate exclusions and memory hashes. No retrieval from test rows, external song catalogs or pretrained embeddings.

### 6.4 Verification and resources

Required tests include held-out-label flips with fixed split assignments; poisoning a held-out feature to expose preprocessing fitted on the wrong population; wrong-group calibration; a deliberately leaky target-encoding path; self-group and indirect retrieval; stale or malformed stopping receipts; class-1 score orientation; fixed-seed replay; and end-to-end prediction through the deployment adapter. The negative tests must fail on the known-wrong variants. An outer-validation label flip must leave all training decisions and its predictions unchanged when assignments are fixed. A stopping-validation label flip may change the selected epoch and final fit; only the stopping-training features and fixed-epoch training path must remain unchanged. A retrieval self-label test holds model weights and other labels fixed and checks constructed query/memory inputs before the query loss is used; it does not demand that a retrained supervised model ignore training labels. Poisoning held-out feature values tests invariance of fitted training state, not invariance of that query's prediction. Test the complete path, not only a standalone transform.

Keep torch in a separate worker process because of the project's recorded OpenMP conflict. Existing installed package versions stay fixed. Planning inspection found scikit-learn 1.9.1, torch 2.14.1, tabm 0.0.3, rtdl-num-embeddings 0.0.12 and scipy 1.18.1; rtdl-revisiting-models is not installed. This inventory is saved with the consultation. Installed APIs still require inspection before implementation. Any added general-purpose library requires a pinned hashed route lock, dry-run dependency inspection and license/source record. A dependency conflict is a recorded feasibility issue, not permission to upgrade the baseline environment. Default to CPU with four threads and deterministic settings. A different device is a separately timed, validated and recorded route; no purchased compute is assumed.

## 7. Method coverage and search spaces

Initial counts below are minimum coverage targets for saying a family was explored under this plan. They are not runtime estimates or permission to launch. Every initial and refinement batch is materialized as an ordered manifest before its first fit. Use a fixed seed (20261031) and a stratified space-filling sample over declared discrete/log ranges, with mandatory matched pairs taking precedence. Deduplicate canonical recipes. If a failure reduces a target, replace it within the same declared space and log both; unresolved failures leave coverage incomplete. Never substitute one weak default run for a family floor.

| Family | Initial configurations | Initial coverage and refinement dimensions | Maximum without a new plan amendment |
|---|---:|---|---:|
| TabM | 32 | Raw scalar, piecewise-linear and periodic embeddings; B/C/D; k in {16,32}; blocks {2,3,4}, width {64,128,256,512}; bins {8,16,32,64}, embedding width {8,16,32}; learning rate log[1e-4,5e-3], weight decay {0} plus log[1e-6,1e-2], dropout {0,.1,.2,.3}; standard/quantile scaling | 96 |
| MLP | 24 | Plain and embedded inputs; B/C/D; layers {2,3,4}, width {64,128,256,512}; same optimizer/scaling ranges and matched embedding pairs | 64 |
| ResNet | 24 | Residual blocks {2,3,4}, width {64,128,256}; same information and optimizer coverage; two dropout locations declared separately | 64 |
| FT-Transformer | 24 | B/C/D; blocks {1,2,3}, token width {32,64,128}, heads {4,8} dividing width; attention/residual dropout {0,.1,.2}; learning rate log[1e-4,3e-3], weight decay {0} plus log[1e-6,1e-2]; standard/quantile scaling | 64 |
| TabR | 16 | Compact official architecture adapted to group exclusions; B/C/D; retrieval k {8,16,32,64}, width {64,128,256}, dropout {0,.1,.2}; learning rate log[1e-4,3e-3]; parameter sharing and all retrieval settings explicit | 48 |
| RBF SVM | 48 | Full 6 by 4 grid: C {0.01,.1,1,10,100,1000}, gamma multiplier {.01,.1,1,10} of 1/d after scaling, on B and D; every candidate evaluated as a calibrated procedure | 128 |
| Linear and polynomial SVM | 12 | Six linear C values above; six polynomial settings across degrees {2,3}, C {.1,1,10}, gamma 1/d, coef0=1; initial D, with B/C matched checks during refinement | 36 |
| Kernel approximations | 24 | Nyström RBF and random Fourier features with L2 logistic regression; components {128,256,512}, gamma {.1,1,10}/d, C {.1,1,10}; balanced B/D; fixed approximation seed 0 initially | 64 |
| Gaussian-process classifier | 12 | Isotropic RBF and Matern nu=1.5; length scale {.3,1,3} times sqrt(d); B/D; initial kernel amplitude 1, optimizer disabled, explicit convergence cap; later bounded length/amplitude optimization if useful | 24 |
| Smooth logistic and interactions | 24 | Existing spline-logistic control plus new cyclic/calendar and D arms; knots {4,8,12}, C log[1e-3,1e2]; explicitly listed low-order audio/date interactions, no arbitrary exhaustive expansion | 48 |
| Metric-learning neighbours | 16 | Regularized/shrunk whitening controls and NCA; dimensions {4,8,16} capped by input dimension, k {15,30,60,120}; B/C/D with standard scaling and group-safe calibration if needed | 40 |
| Statistical controls | 18 | Six shrinkage LDA, six regularized QDA, six Gaussian naive Bayes; balanced B/D; LDA shrinkage {auto,.1,.5}, QDA regularization {.1,.5,.9}, GaussianNB smoothing {1e-9,1e-6,1e-3} | 18 |

Initial family total: 274 configurations, plus the 24 feature comparisons of section 5. A feature comparison may require a new matched control run; count each actual fit in the cost manifest. Historical exact controls count toward coverage only if their complete recipe and source compatibility are verified. Do not label these 274 independent experiments or independent evidence.

Additional exact specifications before manifest generation:

- TabM's 32 include four matched old/no-embedding controls, at least eight PLE configurations, at least eight periodic configurations, the B/D matched pairs and one authors' default-inspired setting. PLE and periodic branches each receive their own outcome; one branch cannot eliminate the other. Validate compatibility with the installed package rather than copying newer API assumptions.
- MLP and ResNet each include at least eight plain and eight embedded settings, four B/D pairs and one default-inspired control. FT-Transformer includes at least four settings per depth and four B/D pairs. The residual and plain networks are separate tested families.
- TabR includes four B/D pairs and at least two retrieval sizes at each of the two smaller widths. A CPU exact-neighbour backend may replace an incompatible search dependency only after equivalence checks on fixed examples. Record the adaptation; do not silently call an arbitrary custom kNN network the published TabR model.
- RBF refinement must test C at matched settings, robust versus quantile preprocessing, and separate audio/calendar block weights {0.25,1,4}, before a negative family conclusion. The floor for these additional matched C/preprocessing checks is eight configurations within the 128 ceiling. Linear and polynomial controls are not prerequisites for continuing the RBF family.
- Kernel ridge with an RBF kernel is a named conditional extension: 12 settings across alpha {0.01,.1,1,10}, gamma {.1,1,10}/d on D, with grouped sigmoid calibration. It starts if the kernel approximation family contributes or RBF SVM is competitive, or if its measured full-route cost fits the broad coverage reserve. It is outside the 274 initial total, within an additional 24-configuration ceiling.
- GP is measured at full training size before rejecting it for cost. Exact and approximate kernels are different candidates. If the exact GPC route cannot fit the approved schedule, record "not completed: measured cost" and retain the approximate-kernel route; never treat that as a negative GP result. Initial fixed hyperparameters avoid hidden optimizer searches; later optimized versions receive new IDs.
- Neural binary cross entropy, AdamW, no class weighting, patience 20, tolerance 1e-6 and maximum 300 epochs are the initial defaults. Keep the original TabM loss and averaging order. Use the architecture authors' parameter-group defaults where applicable, explicitly recording whether biases/norm/embedding parameters receive decay. For each family include a fixed matched log-loss versus AUC stopping comparison on two configurations, inside its ceiling. No stopping on outer AUC. A max-epoch hit or obvious optimization pathology triggers diagnosis before a negative interpretation.
- Diagnostics: loss curves, stopping epochs, tiny-training-set overfit test, majority-prior/null control, dead/constant predictions, numerical scale and saturation, gradient health for neural adapters, sensitivity to learning rate and initialization. A training failure is not an empirical verdict on an architecture.

Scope boundaries: no pretrained foundation models, external audio/text embeddings, external song metadata, test pseudo-labeling or test duplicate boosts. No CNN/RNN is added solely because the table can be arranged as a sequence; no natural sequence structure has been established. TabNet, neural additive models, supervised metric variants beyond NCA, and learned era-dependent/pairwise-ranking combiners are reserve ideas, not secretly completed experiments. Reopen one only with a concrete mechanism, source, coverage proposal and measured cost. This inventory is comprehensive for the current handoffs and major underdeveloped mechanisms, not a claim to exhaust all algorithms.

## 8. Allocation, refinement and seed confirmation

Complete the initial coverage and required representation comparisons before performance-based pruning. Earlier P2.5 thresholds do not terminate this new neural search after a few raw-input trials. Do not demand +0.003 against E before allowing any investigation; that would hide smaller useful contributions.

After initial coverage, nominate two distinct recipes per family: highest standalone mean fold AUC, and greatest gain over E at one of the three declared blend weights. Exact duplicate nominations merge. Ties use unrounded scores then canonical recipe ID. A within-family rank is not an independent validation result.

Each family receives one diagnostic/refinement batch of eight if any of these holds: (a) its best blend improves E by at least 0.0005 on average and is positive on at least two repeats; (b) its best standalone is within 0.002 of the current best standalone across completed families; (c) an otherwise matched representation improves the family's prior control by at least 0.003 on average and is positive on at least two repeats; or (d) a specific optimization defect or search-boundary optimum requires a predefined rescue. For (d), state the evidence and corrective comparison before fitting; it permits one rescue batch, not unlimited retries. A confidence interval excluding zero is not required for this allocation decision.

After that first refinement batch, further batches of eight require at least 0.0005 improvement in the family's best contribution to E, or at least 0.001 in its best standalone while within 0.002 of the overall standalone leader. Two consecutive non-progress refinement batches stop expansion. Representation branches keep separate diagnostics. Ceilings in section 7 remain hard until amended, and the cost/schedule gate can stop earlier with an "incomplete" status. These are engineering allocation rules, not significance tests.

Confirm both nominees from each completed family with estimator seeds 0, 1, 2 (only one fit for deterministic estimators). The seed-0 run may be reused under identical sources; report each seed and the averaged recipe on all 15 folds. For random-feature methods seed also identifies the approximation. For neural models it controls initialization and batches via independent context-derived streams. A weak standalone recipe remains eligible through its ensemble contribution. Do not select the lucky initialization. Five-seed averaging is a named optional comparison for at most two neural nominees, nominated by three-seed contribution, within measured cost; if adopted, its complete seed policy is rebuilt at every validation/deployment level.

A complete negative finding states the configurations, feature arms, seeds, best standalone and contribution, optimization diagnostics, repeat variability and tested scope. Permitted outcomes: promising; tested under this protocol with no useful contribution; inconclusive; incomplete due to time; unavailable due to an identified dependency or implementation obstacle. "Neural methods do not work" is not an admissible conclusion from these experiments.

## 9. Expanded ensembles and candidate construction

First freeze the final three-seed nominee inventory and one final source revision. Re-evaluate E and all contributing historical atoms there before using a new bank with them. Canonical source/input identities must be compatible even if predictions happen to match. Address the deferred combiner-regularization metadata validation before any new learned-procedure bank is admitted.

Select at most four new atomic recipes: the strongest standalone nominee, then the three strongest contribution nominees from distinct remaining families, ranked by their best declared blend gain over E, exact ties by canonical ID. A standalone nominee enters only if within 0.002 of E or if its best declared blend gains at least 0.0005; remaining nominees require the latter gain. If none qualifies, retain E and report the completed exploration. This is a discovery nomination rule, not the final selection rule. Freeze the four-atom limit and nomination algorithm before nominee scores are examined.

Evaluate a deterministic small procedure set: E; each admitted atom alone; each `(1-w)E+w atom` at the three weights; and E blended at those weights with the equal average of all admitted atoms (omit duplicates). Historical H remains a comparator and menu candidate. No greedy weight search. All fixed blend scores reuse matching production banks; their nested predictions must be rebuilt locally. Raw scores cannot enter probability blends.

A new learned combiner is a separately costed extension, not a prerequisite for new models to contribute. If admitted before freeze, use the existing constrained weights {0,.01,.1} and logit stacker C {.01,.1,1}, on E as one channel and the admitted atoms. Rebuild every base channel and all preprocessing/calibration/stopping inside each meta-training fold. E is a six-atom predictor, not a free column from a global bank. Third-level cost includes every E atom and new calibrated/neural fit. Pairwise-ranking or era-dependent combiners need another method amendment, not an unlogged extra objective.

Construct a menu of at most 12 unique candidates: mandatory H and E; the best new standalone if any; the best new fixed blend if any; the best learned procedure if evaluated; then fill remaining places from the declared procedure set by descending unrounded production mean AUC, ties by ID. If fewer candidates exist, keep the smaller menu. Compute structural counts recursively from full procedures: unique atomic recipes, supervised transformations, learned combiners. Complexity ranks are generated from the existing canonical ordering before procedure evaluation. Preserve the final rule: maximum mean fold AUC; 0.001 practical tie band; lexicographic simplicity keys; candidate ID last. Runtime does not break a performance tie. A supervised NCA projection or fitted sigmoid contributes a supervised-transformation count; a probability calibrator is not counted as a multi-channel combiner. Learned ensemble combiners retain their separate count. Save and test these extensions to the complexity metadata before evaluating the procedure set.

Publish a new menu path `configs/phase2/p2.9/menu.json`, never overwrite `configs/phase2/menu.json`. Freeze recipes, weights, seeds, stopping/calibration/retrieval policy, source revision, package locks, permitted devices, selection function and evaluation plan together. Any cost-based omission happens before new outer scores and is reported. A changed ingredient means a new candidate and full production evaluation.

## 10. Revised restricted nested evaluation

Use the existing evaluator's three outer grouped folds and rebuilt 15 inner folds, with all fitted layers local. Save a new named outer assignment at seed 20261102 before any new outer predictions are generated. This is a fresh partition and conditional evaluation, not an untouched dataset. P2.7 outer results never tune new candidates. Do not try multiple outer seeds and report the best.

Repeat selection among the new frozen menu inside each outer-training population. For each outer fold generate predictions from the selected procedure, E and H, each with fold averaging. Regenerate all inner/meta/calibration/stopping/encoding/retrieval partitions inside their permitted populations. No global prediction banks or global combiner weights enter. Cache keys include exact training rows and labels, query rows, contexts, recipes, seeds, features, source and dependency identities. Reuse fits only for identical requests. Cache counts must expose both ordinary cross-fits and deeper adapter fits.

Default deployment remains fold averaging. Full refit is evaluated only if declared and costed before seeing outer scores. The prior fallback/deployment thresholds stay unchanged: fallback to H if the selected procedure loses at least 0.003 on average and loses on all three folds; a deployment switch needs +0.003 mean gain and gain on all three folds. If the full-refit comparison is omitted for cost, the evaluation cannot recommend a switch. Report the selected procedure's difference to E as well as H; Sam retains final-selection authority.

Run the complete new evaluation again in a new process with an empty cache. Compare assignments, selections, receipts and every prediction array under a prespecified tolerance. CPU deterministic routes retain the existing 1e-12 tolerance; any other route requires an approved numerical reproducibility criterion before scoring. Never loosen it after seeing a mismatch. Verify the exact artifact hashes and fresh-fit counts, then generate a receipt-bound decision. New correctness fixes invalidate all dependent banks and claims.

This evaluation does not nest the entire 274-configuration discovery search. It tests selection among a restricted menu informed by previous exploration. Its group bootstrap is conditional on the realized predictions/selections and excludes retraining/whole-search variability. Smaller inner fitting populations alter date support. Selection or deployment choices that use outer evidence consume that evidence; report these limits explicitly.

If the complete revised evaluation and replay cannot finish before release, retain the already evaluated E as the supported production choice. Newly promising candidates remain provisional and are not described as having completed nested validation. Any proposal to use a different reduced evaluation for final selection requires a separate R2 review and Sam's decision. The old P2.7 result cannot certify new models.

## 11. Cost, schedule and launch gates

Known historical measurements: TabM's eight configurations took 34.8 minutes in total; its score-free light/representative/expensive 15-fold pilots took about 102/156/759 seconds under concurrent load (Findings 024, 025). P2.7 build and replay each took about 1.75 hours for the old menu. These are context, not estimates for FT-Transformer, embeddings, calibration or retrieval.

P2.9a measures light, central and expensive representative configurations for every major family at actual training size. Include feature building, stopping plus refit, calibration subfits, inference, process startup, memory and output serialization. Record hardware, thread count, package versions and concurrent load. Separate score-free timing outputs from the score log. Include at least one smaller nested-population pilot for each new expensive adapter; do not extrapolate nested cost from only a top-level fit.

Create a deterministic operation-count graph for the planned menu: ordinary base fits, stopping-training/refit pairs, calibration folds plus refits, encoder folds, retrieval fitting, estimator seeds, meta folds, outer folds, and cache sharing. Compare projected versus actual counts in a small synthetic end-to-end case. Cost is the sum of measured route costs times non-shared counts, with conservative high/low ranges and at least 25% overhead margin; replay approximately doubles the uncached work and is budgeted explicitly. Existing P2.7 formulas alone do not cover the new adapters.

Before launching large work, present Sam with: full initial coverage cost; mandatory feature comparisons; nominee seed confirmation; at least one refinement allocation; historical replay; candidate construction; revised nested build/replay; storage/memory; and time for method/release review and submission preparation. Record the exact admitted manifests and budgets. If full coverage does not fit, present the measured shortfall and a prioritized feasible tranche. Do not silently redefine "comprehensive" by cutting every costly family.

Priority for admission: safe shared infrastructure; matched TabM/MLP/ResNet/FT-Transformer and RBF SVM coverage; TabR's implementation and full-size cost pilot followed by its coverage; approximate kernels and the remaining named families; refinements and additional ensembles. Cheap independent baselines may fill idle resource slots without delaying the core. Every named family gets a feasibility assessment; no family is labeled negative solely for missing the deadline. Reduce redundant expansions and extra combiners before removing initial core coverage or correctness checks.

Planning targets in America/Los_Angeles: Saturday 2026-10-31 evening, plan and method review; Sunday 2026-11-01 morning, pilot-based budget and implementation checkpoints; Sunday 12:00 PST, first coverage checkpoint; Sunday 16:00 PST, stop new expansions; Sunday 18:00 PST, target revised-menu freeze; Monday 2026-11-02 10:00 PST, target evaluation/replay completion; Monday 14:00 PST, release review complete; Monday 15:00 PST, verified files ready for Sam, leaving two hours before the 17:00 PST close. Daylight saving ends Sunday; use timezone-aware timestamps. These are provisional targets contingent on launch time and pilots, not assertions that every task fits.

A batch is admissible only if its conservative finish time plus all outstanding confirmation, replay, evaluation and review reservations precedes Monday 15:00 PST. At each checkpoint recompute this inequality from remaining work. If the projected revised evaluation requires an earlier freeze than Sunday 18:00, stop expansion earlier. If even initial core coverage cannot fit, ask Sam to resolve the measured scope/time conflict before launching an oversized job. Never cut isolation, bank validation or required review to meet a target. Report/Colab remain due Tuesday 2026-11-03 23:59 PST.

## 12. Artifacts, reviews and completion

Planned paths: `configs/phase2/p2.9/` for protocol, ordered batches, references and final menu; new family modules under `songpop/phase2/`; `analysis/p29_*.py` and `scripts/check_phase2.py` extensions; `artifacts/phase2/p29/` for timing, coverage, comparison tables, receipts and nested results. Preserve historical logs/banks. New runs have new recipe/source identities and a task ID. Record every source revision, raw-data hash, split hash, library/source lock, device, seed, failure and retry. Store source licenses and modification notes for general implementations, without downloaded benchmark datasets or weights.

A coverage table has one row per family/representation with target/completed counts, missing arms, failures, seed status, best standalone and E contribution, repeat differences, cost and outcome. Findings land in `doc/findings.md` before downstream use; draft evidence may support provisional work, reviewed evidence requires the named approval record. Add verbatim consultation prompts/responses and the integration account under `doc/llm_usage/`. STATE stays at most 80 lines; archive earlier state when refreshing it.

R2 review covers this plan's validation/selection/fit-boundary choices before tuning depends on them. A second R2 at P2.9g covers the concrete frozen menu and cost graph. R1 covers all findings, code, artifacts and claims used for final selection, plus every submission file. R3 and Sam's approval apply to costly launches or external/irreversible actions. Claude Fable 5.1 xhigh reviews in a separate fresh run through the cross-review skill; a planning consultation is not approval.

Plan-delivery checklist: reconcile owner decisions and old-plan overrides; retain Claude's positions before Codex's design/critique; primary-source technical references checked; counts and schedule checked; commit; fresh Claude review; answer findings and recheck material edits; update STATE and the LLM integration index. No training is part of this planning turn.

Execution completion: core coverage either complete or explicitly marked incomplete with Sam's resulting scope decision; all admitted nominees confirmed across seeds; whole fitting procedures leakage-tested; old deployment atoms reproduced; new menu frozen/reviewed; new nested build/replay verified; phase 3 evidence and submission provenance ready for R1. Useful negative results and unfinished methods are both reported honestly.

## 13. Sources and consultation

Project evidence: Findings 011, 019, 021 to 030; `analysis/p22_summary_output.txt`, `analysis/p25_summary_output.txt`; the three handoffs under `notes/`; current menu and P2.7 receipts. Consultation and source notes: `../llm_usage/2026-10-31-p2.9-plan/`.

Primary sources checked 2026-10-31. They motivate implementation choices; their benchmark results do not predict this project's improvement.

- [TabM official implementation](https://github.com/yandex-research/tabm): numerical embedding interfaces, defaults and training conventions.
- [Numerical embeddings official implementation](https://github.com/yandex-research/rtdl-num-embeddings): piecewise-linear and periodic input representations.
- [MLP, ResNet and FT-Transformer official implementation](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md): the separate architectures and reference modules.
- [TabR official implementation](https://github.com/yandex-research/tabular-dl-tabr) and [paper](https://arxiv.org/abs/2307.14338): learned retrieval, requiring project-specific group isolation.
- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [calibration](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html): margin models and calibration; project adapters must supply explicit grouped populations.
- [Kernel approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html) and [Gaussian-process classification](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html): alternative kernel implementations and controls.
- [Neighbour methods and NCA](https://scikit-learn.org/stable/modules/neighbors.html), [LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html), [QDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html): metric-learning and statistical controls. API behavior is checked against the pinned installed version before implementation.
