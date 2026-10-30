I recommend a substantially broader search, followed by a deliberately smaller ensemble menu. The largest new cost is validating learned combiners correctly, not running several hundred classical configurations.

These are proposed rules for Claude’s amendment, not approval of it. I made no file changes and ran no reviewers.

1. **Make P2.0b a separate amendment with explicit precedence.**

   Record that Sam’s second message supersedes both the old experiment ceilings and the requirement that an added experiment replace another. It does not supersede the freezes, grouped validation, installation rules or supplied-data-only restriction.

   Keep the P2.1 snapshot unchanged. Its current separation is appropriate: section 13 describes infrastructure changes, while sections 5 to 11 retain the approved P2.0 rules pending this amendment. Draft against an identified revision, then review the amendment against the actual merged document after P2.1 converges.

   Replace the strategy’s categorical statement that review turnaround is the bottleneck with:

   > Budgets use measured fitting, verification and review times. The schedule reserves time for all three. No resource is assumed to be the bottleneck before measurement.

   I computed the deadline intervals with Python’s `zoneinfo`: there are **55 hours** from Thursday 11:00 PDT to the menu freeze, followed by **25 hours** to the Sunday final freeze. The extra hour is the daylight-saving transition.

2. **Use the following initial allocations, with evidence-gated expansion.**

   Counts include anchors. The ceilings cover both broad search and local refinement on the initial feature representation. They are ceilings, not completion targets.

   | Family | Initial configurations | Expansion batch | Ceiling before feature retuning | Initial serial fitting cost |
   |---|---:|---:|---:|---|
   | LightGBM | 64 | 32 | 160 | About 2 to 51 min at the supplied reference timings |
   | Random forest | 48 | 24 | 120 | About 12 to 40 min |
   | XGBoost | 48 | 24 | 120 | About 15 min at the supplied depth-4 timing |
   | Extra Trees | 32 | 16 | 80 | About 10 min |
   | CatBoost | 48 | 24 | 120 | \(48t_C\), where \(t_C\) is the measured 15-fit cost |
   | Spline/logistic models | 24 | 12 | 48 | About 22 s |
   | EBM | 12 | 6 | 30 | \(12t_E\), measured after installation |
   | kNN | 48 | 16 | 80 | About 38 s |
   | HistGB | 12 | 12 | 36 | About 2.8 min at my probe’s settings |

   This is **436 initial classical configurations**, with a maximum of 894 before the separate feature-retuning allocation. Failures, unavailable packages and the stopping rules below can reduce those totals.

   One important correction to the timing assumptions: I ran a read-only probe through the current engine:

   ```text
   HistGB, F2, max_iter=300, learning_rate=0.05,
   max_leaf_nodes=15, min_samples_leaf=20,
   split seeds 155, 2026, 1029, estimator seed 0
   ```

   It took **13.765 seconds for 15 fits**. `threadpool_info()` showed OpenMP limits of 10 outside `thread_limit("hist_gb")` and 4 inside it. This is an **unreviewed timing**, not a new performance finding. I did not use its AUC. The supplied 158-second measurement should therefore not determine HistGB’s priority.

   Using the supplied timings and that probe, I calculated **44 to 119 minutes** for the initial families whose costs are known. This is a reference-cost scenario, not an upper bound on the enlarged spaces. If CatBoost costs 60 seconds and EBM 120 seconds per 15-fit run, they add 72 minutes.

   Before each family starts, time light, representative and expensive configurations. Include feature construction, prediction and internal stopping refits. Replace the projections with measured batch costs.

3. **Broaden the parameter spaces, while keeping their structure auditable.**

   Use seeded random sampling. Sample scale parameters logarithmically or from geometrically spaced lists. Save the complete configuration list before each batch.

   | Family | Proposed initial region |
   |---|---|
   | LightGBM | Leaves `{7,15,31,63}`; minimum child rows `{10,20,40,80,120,200}`; learning rate `0.01–0.10`; trees `{300,600,1000,1600}`; L1 `{0,0.01,0.1,1,5,10}`; L2 `{0,0.1,1,10,30,100}`; column fraction `{0.6,0.8,1}`; row fraction `{0.7,0.85,1}`. Set bagging frequency consistently with row sampling. |
   | Random forest | 500 trees initially; leaf size `{1,2,3,5,8,12,20,30}`; features `{"sqrt",0.3,0.5,0.75,1.0}`; depth `{None,8,16,24}`; bootstrap sample fraction `{0.6,0.8,1}`. |
   | Extra Trees | 500 trees; leaf size `{1,2,3,5,8,12,20}`; features `{"sqrt",0.3,0.5,0.75,1.0}`; depth `{None,12,24}`; no bootstrap initially. |
   | XGBoost | `hist`; depth `{2,3,4,5,6}`; child weight `{1,3,10,30}`; learning rate `0.015–0.10`; trees `{300,600,1000,1600}`; L1 `{0,0.1,1,10}`; L2 `{1,3,10,30,100}`; gamma `{0,0.1,0.5,1}`; row and column fractions `{0.65,0.8,1}`. |
   | CatBoost | Depth `{3,4,5,6,7}`; learning rate `0.015–0.10`; iterations `{400,800,1200,2000}`; L2 `1–100`; random strength `{0,0.1,0.5,1,2}`; Plain and Ordered boosting. Keep symmetric trees and numeric F2 initially. |
   | Smooth models | Spline knots `{3,4,5,7,10}`; cubic degree; C `0.001–100`. Include ordinary log-standard logistic controls. EBM: main effects or the three declared interactions; rounds `{300,700,1500}`; learning rate `{0.01,0.03,0.06}`; disable internal validation and bags as already planned. |
   | HistGB | Leaves `{7,15,31}`; minimum leaf `{10,20,40,80}`; learning rate `{0.02,0.05,0.1}`; iterations `{150,300,600}`; L2 `{0,1,10}`; no automatic early stopping. |

   These spaces require small factory extensions in the later tasks. For example, the current XGBoost allowlist lacks `reg_alpha` and `gamma`. That is future P2.2 work, not a P2.1 defect.

   For boosting, reject extremely small or large learning-rate/iteration combinations before fitting. An initial admissible region of `5 ≤ learning_rate × iterations ≤ 100` is reasonable. Preserve exact anchors even outside sampling constraints.

   **Keep fixed iteration counts for classical boosting.** They are cheap enough here, and this avoids introducing another split into every trial. Around a promising configuration, compare approximately `{0.67T, T, 1.5T}` while holding other parameters fixed. An optimum on a boundary permits one declared extension within the family’s ceiling. Do not inspect outer-fold learning curves to choose iterations.

4. **Replace the single promotion gate with explicit exploration-allocation rules.**

   Use named, frozen comparators for each batch:

   - The original LightGBM F2 anchor remains the project reference.
   - The family incumbent measures progress within a family.
   - A reference predictor \(H\), chosen from already reviewed results before the batch, measures ensemble contribution.

   For each configuration, calculate its standalone result and two fixed blends with \(H\): **90/10 and 75/25**, with the challenger receiving the smaller weight. Use matching folds and log these as search evaluations.

   After the initial allocation, authorize another batch through any of these routes:

   | Route | Proposed rule |
   |---|---|
   | Clear standalone progress | At least `+0.003` against the named comparator, positive on all three repeats |
   | Smaller useful progress | At least `+0.001` against the family incumbent, positive on all three repeats, and within `0.005` of the current strongest standalone recipe |
   | Competitive refinement | At least `+0.0005`, positive on all three repeats, and within `0.002` of the strongest standalone recipe |
   | Ensemble contribution | A declared blend improves \(H\) by at least `0.0005`, positive on all three repeats |

   These are **allocation heuristics**, not significance thresholds. I am uncertain about the optimal numerical cutoffs; no evidence here can establish them precisely. Their value is that they are explicit, paired and consistent.

   Stop expanding a family after two consecutive expansion batches produce neither:

   - A new qualifying standalone improvement of at least `0.0005`; nor
   - A new qualifying blend improvement of at least `0.0005`.

   Also stop at the ceiling or the scheduled search cutoff. Preserve already useful candidates when a family stops.

   Bootstrap only the decision-driving comparisons, rather than every trial. Report its conditional limitations. Do not make a confidence interval excluding zero a prerequisite for allocating another inexpensive batch.

   **This replaces the old rule that smaller gains authorize no further search. It does not replace section 11’s final selection rule.**

5. **Use all three split seeds for inexpensive trials; distinguish screening from confirmation.**

   My default is all three official split seeds, estimator seed 0, for the classical searches.

   For a predeclared expensive subspace whose representative 15-fit cost exceeds **120 seconds**, permit seed-155 screening. Determine this from timing, before performance results. Confirm the union of:

   - The best two standalone configurations.
   - The best two configurations by declared blend contribution.
   - A named family anchor.

   Deduplicate that list. Every retained finalist needs all three official repeats under its final recipe.

   Do not call the full three-repeat result an independent confirmation when those same repeats were used to choose the configuration. It is a complete matched evaluation and a stability check.

   For expensive screening, completing two additional repeats costs approximately twice the five-fit screen. Include that cost in the batch reservation before starting it.

6. **Use local random refinement first. Optuna is an optional second mechanism, not a prerequisite.**

   For up to four qualifying families, select two parents: the best standalone configuration and the best contributor to a blend. If they coincide, use the next distinct qualifying configuration.

   Run local refinement in batches of 12, up to 24 configurations per family, counted within the table’s ceilings:

   - Multiply positive scale parameters by factors sampled from roughly `[0.5,2]`.
   - Move discrete parameters to neighboring values.
   - Change one to three parameter dimensions per proposal.
   - Reserve a quarter of each batch for broader draws.
   - Deduplicate against all previous canonical recipes.

   This supports two objectives without forcing them into an arbitrary scalar score.

   I would **not install Optuna initially**. Its justification becomes stronger if a family remains productive after at least 64 evaluations and another 32 evaluations are warranted. At that point, a reviewed TPE route can replace a declared refinement batch.

   If used, freeze the objective, space, sampler seed and trial order. Save each adaptive proposal before fitting it and record the observations that produced it. Do not mix one-repeat and three-repeat objective values in one study. Do not prune on the first outer-fold AUC.

   Sequential execution is preferable for reproducibility. Optuna’s official documentation explicitly identifies parallel optimization as a source of nondeterminism. [Optuna reproducibility guidance](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results).

   No Optuna version is proposed here. Compatibility still requires the project’s reviewed additive lock process.

7. **Make feature engineering a second search stage, followed by retuning.**

   Give P2.4 an initial allocation of **32 comparisons**, expandable in batches of 16 to **96**. Save the exact recipes, comparators and allocation before each batch’s corresponding scores.

   Start with the nine existing anchor checks. Independently reserve these four comparisons, regardless of the LightGBM results:

   | Model | Comparison against the otherwise identical tuned F2 recipe |
   |---|---|
   | Random forest | Add date count |
   | Random forest | Add date TE30 plus count |
   | CatBoost | Add date count |
   | CatBoost | Add date TE30 plus count |

   If CatBoost is unavailable, use XGBoost and record that substitution before the comparisons run.

   Then allocate the remaining comparisons in this order:

   1. Date count and target encoding on the strongest tuned models.
   2. Selected combinations of individually useful blocks.
   3. Preprocessing alternatives for smooth, distance and neural models.
   4. Fixed feature-removal checks.
   5. Interactions, cyclic calendar and era-relative features.
   6. Training-weight sensitivity on promising recipes.

   Further date experiments can include smoothing `{3,10,30,100}`, selected in batches rather than an exhaustive block-by-family grid. Retain the existing cross-fitting, row-weighted target statistics, unseen-date prior and distinct-group count definitions.

   The **72.2% exact-date overlap is motivation**, not proof of album identity or predictive value. Measure coverage and support in the actual training dictionaries at each validation level. Do not group by date merely to test this feature.

   Suggested removal checks are fixed, named recipes:

   - Remove `days_since_1970`, retaining the other observed components.
   - Remove day-of-month and day-of-year together.
   - Remove weekday.
   - Remove the three musical categories.
   - Remove a newly added block from a successful combination.

   Do not remove audio features because their marginal AUC is near 0.5.

   After these comparisons, retune **up to four model-and-feature combinations**, in two batches of 16 each. Thus the post-feature allowance is at most **128 configurations**, separate from initial F2 tuning. Use the allocation rules in item 4.

   A fixed feature mask is a recipe hyperparameter. Its selection remains adaptive and must be acknowledged. Any data-derived selector, importance ranking, quantile transform or learned preprocessing must be fitted inside the relevant training population. I would omit an automated importance-based selector here and use explicit removal recipes.

   At 20 seconds per run, 96 feature comparisons plus 128 retuning configurations cost about **75 minutes**. Actual encodings and model choices will change that projection.

8. **Give kNN, the smooth models and TabM concrete, meaningful opportunities.**

   **kNN.** Start with F1, not seven redundant date coordinates in an unweighted distance.

   Fit the following representation inside every training population:

   - Apply the existing logarithms to the designated audio columns.
   - Median-impute and standardize the ten continuous audio variables.
   - Divide that audio block by \(\sqrt{10}\).
   - Standardize release year separately and multiply it by \(\alpha\).
   - One-hot key, mode and time signature. Multiply this block by \(0.5/\sqrt{3}\). Do not standardize rare one-hot columns to unit variance.

   The initial 48 configurations are the complete grid:

   ```text
   k = {15, 30, 60, 120, 240, 480}
   weights = {uniform, distance}
   alpha = {0.25, 0.5, 1, 2}
   metric = Euclidean
   ```

   This directly tests neighborhood scale and the importance of date in similarity. Refinement can test category weights, robust scaling, Manhattan distance with appropriate block normalization, and a cyclic-calendar alternative. Keep duplicate observations and primary groups as currently defined. Do not use inverse-group fitting weights with an estimator that does not support them.

   **Smooth models.** Give both spline logistic and EBM their initial allocations. Finding 008 concerns ordinary logistic regression; it does not establish that cyclic dates or nonlinear calendar effects are useless for a smooth model. Reserve selected F2/cyclic comparisons within P2.4.

   **TabM.** Replace the two-configuration and 90-minute author-time caps with a measured-cost policy:

   - Initial target: eight configurations.
   - Expand by four, to at most 16, only through item 4’s criteria.
   - Reserve three hours initially for fitting and confirmation.
   - Permit expansion to six hours if productive and if Sunday’s evaluation still fits.
   - If the pilot makes eight configurations plus confirmation exceed the initial reservation, start with four. Record that timing-based decision before inspecting the pilot’s score.

   Suggested region: two or three blocks; width `{64,128,256}`; 16 members initially; dropout `{0,0.1,0.2}`; AdamW learning rate `3e-4–3e-3`; weight decay `1e-6–1e-2`; batch size `{128,256}`. Include a limited numerical-embedding alternative only if its training-fold construction is audited.

   Predeclare maximum 300 epochs, patience 20 and inner validation log loss as the stopping metric. Choose the earliest best epoch under a fixed tolerance. Then discard the stopping model and refit preprocessing and the network on the complete permitted training population, as already planned.

   If a five-fit screen costs \(q\) minutes, eight screens plus completing three repeats for four configurations costs approximately **\(16q\)** minutes. At \(q=10\), that is 160 minutes before final estimator-seed averaging.

   Keep a neural candidate alive through measured blend contribution even if its standalone AUC trails substantially. Do not automatically put a costly neural model inside a learned combiner. It can contribute through a fixed probability blend.

9. **Separate the broad ensemble exploration pool from the frozen Sunday menu.**

   Before P2.6 ensemble scores are calculated, save a deterministic rule for constructing the pool from reviewed banks.

   I propose:

   - Three strongest distinct-family leaders by standalone AUC.
   - Up to two additional family leaders chosen by their declared blend contribution.
   - The original anchor.
   - Up to two additional configurations from the leading family, for a three-configuration family average.

   This gives **at most eight distinct atomic recipes**. An atomic recipe means an exact feature, preprocessing and model configuration. Estimator seeds do not count as separate atomic recipes.

   Within-family averaging should use equal weights. Include the leader and up to two configurations within `0.002` of it, chosen by a fixed ranking rule. Do not average every attempted configuration.

   Evaluate at most **32 declared ensemble procedures**, including:

   - Equal averages of the strongest two and three channels.
   - An equal average of all selected channels.
   - The existing 90/10 and 75/25 challenger blends.
   - The within-family average.
   - Three constrained-weight settings.
   - Three regularization settings for one stacker family.

   Freeze at most **12 predictor candidates** for Sunday. Retain the anchor, the selected standalone leaders, the family average, the best fixed blend, the best constrained-weight procedure and the best stacker. Fill any remaining slots by the existing selection rule.

   This is deliberately broader than the current five blends, but it avoids an unrestricted subset and weight search.

   Seed policy:

   - Use seeds `{0,1,2}` for stochastic classical finalists.
   - Do not fit deterministic kNN three times.
   - For TabM, declare one or three estimator seeds using its measured confirmation and nested-evaluation cost. Never choose the best individual estimator seed.
   - Every final policy still requires all three official split repeats.

10. **Implement learned combiners as fitted procedures with their own outer CV.**

   Let `fit_procedure(D)` receive only the permitted training rows \(D\).

   ```text
   Build one explicit five-fold grouped partition of D.
   For every base recipe:
       rebuild all features, preprocessing and encodings inside each inner fit;
       produce exactly one inner-OOF probability per row of D.
   Assemble the declared meta-feature matrix Z.
   Fit the combiner on Z and y_D.
   Refit each base recipe on all D.
   Predict new rows with those refitted bases, then the fitted combiner.
   ```

   Average estimator seeds within each base recipe before constructing its meta-feature. A within-family channel averages its declared configurations at that same point.

   Do not stack three repeat predictions as if they were three independent observations. One inner repeat gives one meta-training row per original row.

   Use two learned procedures:

   **Constrained probability weights**

   \[
   \min_{w\ge0,\ \sum w=1}
   \frac1n\sum_i \operatorname{LogLoss}(y_i,(Zw)_i)
   +\lambda\lVert w-u\rVert_2^2,
   \]

   where \(u\) is the equal-weight vector and \(\lambda\in\{0,0.01,0.1\}\).

   Use the installed SciPy optimizer. Check convergence, constraints and finite predictions. Clip probabilities only as specified for numerical evaluation of the loss. Zero weights are valid and must be represented correctly in the procedure artifact.

   **One regularized stacker**

   - Clip base probabilities to `[1e-6, 1-1e-6]`.
   - Convert them to logits.
   - Fit a scaler on the meta-training matrix only.
   - Fit L2 logistic regression with intercept and `C ∈ {0.01,0.1,1}`.
   - No original-feature passthrough and no class balancing.

   Select settings using the outer evaluation of the entire procedure. The combiner’s training AUC is not selection evidence.

   Evaluate each procedure on the official 15 outer folds by calling `fit_procedure(outer_training_rows)`. Store its held-out and test predictions. Default deployment averages those **15 complete fitted procedures**.

   Refitting bases on all \(D\) after generating OOF meta-features creates a familiar training-size mismatch. It is acceptable because the entire procedure is evaluated externally, and the rule is fixed. Scikit-learn’s stacking documentation describes this OOF-training/full-base-refit pattern and warns that its internal CV generates training predictions rather than evaluating the stack. [StackingClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html).

11. **Extend Sunday’s evaluation with the necessary third level, and budget shared fits.**

   Preserve the three outer folds and three-repeat inner selection protocol.

   For a learned candidate, there are now three relevant levels:

   | Level | Purpose |
   |---|---|
   | Sunday outer split | Evaluate the restricted selection procedure |
   | Inner selection CV | Compare candidate procedures without Sunday outer labels |
   | Combiner-training CV | Generate group-safe meta-training predictions within each inner training population |

   Target encoding and neural stopping remain local to the corresponding base-model fit. They are not shortcuts around these levels.

   In each Sunday outer training population:

   1. Build all candidate inner validation predictions.
   2. For every learned candidate, rebuild its meta-training predictions inside each inner training population.
   3. Apply section 11’s selector to those inner banks.
   4. Predict the Sunday outer validation rows with the selected inner-fold average.
   5. Separately refit the **complete selected procedure** on the outer training population.
   6. Produce the anchor’s fold-average prediction.

   Full refit of a learned candidate means regenerating permitted OOF meta-training predictions and fitting its combiner. It does not mean retaining globally learned weights and refitting only the bases.

   Cache base predictions across combiner penalties and stacker settings. The cache key must include fitting row IDs, validation row IDs, group/split identity, exact recipe, estimator seeds, source manifest and dependencies.

   For a transparent cost calculation, define:

   - \(C_{\rm all}\): summed 15-fit time of every unique atomic recipe in the frozen menu, at its final estimator-seed policy.
   - \(C_L\): the corresponding sum for recipes used by learned combiners.

   A conservative shared-cache estimate is:

   \[
   T_{\rm Sunday}\approx 3.2C_{\rm all}+16.2C_L.
   \]

   The first term covers the ordinary inner banks and outer refits. The second covers additional combiner-training cross-fitting and conservative refit overhead.

   I calculated these scenarios:

   | \(C_{\rm all}\) | \(C_L\) | Evaluation build | Build plus full replay, with 25% margin |
   |---:|---:|---:|---:|
   | 600 s | 300 s | 113 min | 4.71 h |
   | 1,200 s | 300 s | 145 min | 6.04 h |

   These are arithmetic projections, not measured nested runs. Confirm them with a representative pilot.

   Reserve **eight hours** for Sunday computation and verification, beginning Saturday night after P2.6 approval. If the projection exceeds this, first remove expensive members from the learned combiner while retaining useful fixed blends. Reduce duplicate configurations next.

   Retain the existing declared reduced path if necessary. Even a fixed-candidate check of a stack must rebuild its internal OOF predictions. Decide the path from timing before inspecting outer scores.

   Preserve the current fallback and deployment thresholds. Explicitly retain their limitation: once Sunday results trigger either decision, they are no longer an untouched estimate of the resulting choice.

12. **Treat the larger search’s optimism as a material limitation.**

   Hundreds of trials and their blend checks increase selection pressure. Three repeated splits reduce partition sensitivity; they do not supply three independent datasets.

   I would not create a purported “untouched holdout” now. All labels have already participated in exploration. Nor would I claim that a new split seed solves selection bias.

   Instead:

   - Log every trial, feature comparison, blend check and adaptive allocation.
   - Freeze the candidate-construction rule before its corresponding evaluations.
   - Freeze Sunday’s exact menu and procedure before outer results.
   - Report exploratory best scores separately from restricted nested results.
   - Describe bootstrap intervals as conditional on fixed predictions.
   - Preserve all three repeat-level differences for decision-driving comparisons.
   - Do not reopen search after Sunday reveals a disappointing result.

   New split seeds could be a partition-stability diagnostic, but I would not add them as another selection gate. Sunday already introduces new outer partitions.

   Also report the training-size limitation. Learned combiners’ innermost base fits can use substantially fewer rows than ordinary production CV. Date-support and neural-stopping effects can therefore contribute to a nested-score decline.

13. **Re-cut the schedule around independent branches and early ensemble infrastructure.**

   | Task | Revised scope and dependency | Target including review |
   |---|---|---|
   | P2.0b | Amendment only. Integrate after the fixed P2.1 snapshot is released. | Thursday afternoon |
   | P2.2 | Search extensions, kNN representation, broad installed-family searches and initial refinement. Depends on approved P2.1 and P2.0b. | Thursday evening to Friday morning |
   | P2.3 | Reviewed CatBoost/EBM installations, pilots and searches. Remove its unnecessary dependency on P2.2; use the approved anchor. | Thursday evening to Friday midday |
   | P2.4 | Feature comparisons, cross-family checks, preprocessing/removal checks and retuning. Requires the relevant reviewed family results. | Friday through Saturday morning |
   | P2.5 | TabM implementation, installation, pilot, search and confirmation. Independent of P2.2 results initially. | Start Thursday evening; conclude by Saturday morning |
   | P2.6a | Learned-procedure engine, cache, nested evaluator and isolation tests. Use synthetic inputs and the approved anchor for development. | Friday |
   | P2.6b | Ensemble evaluation, final seed policies, menu and deployment declarations. Depends on relevant completed tasks and P2.6a. | First review request Saturday 14:00 PDT; freeze 18:00 |
   | P2.7 | Execute frozen nested evaluation, verify and record decisions. | Saturday night to Sunday; review request by 14:00 PST, freeze 18:00 |

   P2.6a and P2.6b are distinct deliverables, each fully reviewed. This prevents Sunday from being the first use of a newly written nested evaluator.

   Parallelize independent work from approved source revisions. Benchmark at most two four-thread fitting processes against serial execution before adopting that arrangement. Serialize package installations. Do not mutate a source tree while a run or review fingerprints it. Use separate output shards and a checked merge rather than concurrent uncontrolled JSONL appends.

   Cut in this order:

   1. Expansion batches without recent measured progress.
   2. Additional weak feature combinations and redundant removal checks.
   3. Extra configurations within a family average.
   4. Expensive channels inside learned ensembles.
   5. Unpromising neural refinements.
   6. A learned ensemble procedure that cannot complete correct evaluation.

   Do not automatically cut XGBoost, Extra Trees or kNN before their initial meaningful evaluations. Their measured costs do not justify that old priority.

14. **Build later infrastructure explicitly, without pretending P2.1 already supplies it.**

   The current recipe engine supports independent model, feature and preprocessing choices. It does not yet implement the expanded procedures.

   | Later task | Addition |
   |---|---|
   | P2.2 | New factory parameters; kNN; named preprocessing alternatives; conditional sampling; local perturbation batches; timing and allocation records |
   | P2.4 | Additional smoothing values or preprocessing blocks; explicit removal recipes; post-feature retuning manifests |
   | P2.5 | TabM wrapper, stopping receipts, repeatability checks and runtime-based budget records |
   | P2.6a | `procedures.py`, `meta.py`, population-keyed prediction cache, generic fold contexts, learned-procedure bank metadata, recursive deployment and duplicate checks |
   | P2.6a/P2.7 | `nested.py`, frozen selector, outer result isolation, cost accounting and replay |
   | P2.6b | Frozen candidate manifest, exact ingredient graph, final seed policies and complexity ranks |

   A learned procedure needs its own schema. Record input channels, atomic recipes, averaging order, meta-fold assignments, combiner objective and regularization, clipping, fitted weights/coefficients, preprocessing statistics and fitting receipts. Do not disguise it as a fixed blend with one global weight vector.

   Keep the ten prediction arrays where useful, but extend metadata and validators deliberately. Internal nested banks must support their actual populations and fold counts. Do not force them through `Context.load()` and the production-only seed checks.

   Required new isolation tests include:

   - Changing held-out labels, with split assignments fixed, changes no fitted object or prediction for that held-out population.
   - Every base fit underlying a meta-training prediction excludes that row’s entire group.
   - Changing Sunday outer labels changes neither inner selection nor fitted combiner weights.
   - Cache entries from a larger population are rejected.
   - Different combiner regularization settings reuse base predictions but fit distinct combiners.
   - Full-refit and fold-average deployments remain distinct.
   - A corrupted member prediction cannot be hidden by another member’s opposite error.
   - Learned ensembles preserve duplicate consistency.

   Preserve historical banks and logs. Later source changes can legitimately make old phase 2 banks require replay at their matching source commit. Do not rewrite their source hashes or claim that a current-head mismatch is verification. Before freeze, regenerate the small finalist menu under one final compatible implementation revision. Phase 1 remains unchanged.

15. **Edit the strategy consistently, preserve the final rule, and make the exit criteria explicit.**

   Update these strategy locations:

   | Section | Required amendment |
   |---|---|
   | §1 constraints | Replace the assumed bottleneck with measured resource budgeting |
   | §4 step 3 | New cheap/expensive screening and confirmation policy |
   | §4 step 6 | Learned ensembles require population-local prediction banks; global banks alone are insufficient |
   | §4 step 9 | Replace the sole `0.003` allocation gate with item 4’s routes |
   | §4 step 10 | Explicit three-level evaluation for learned combiners |
   | §5.2 | Cross-family date checks, preprocessing alternatives, removals and retuning |
   | §5.3 | Meaningful allocations for XGBoost, Extra Trees, kNN, smooth models and TabM |
   | §5.4 | New batches, ceilings, stopping rules and optional adaptive optimization |
   | §5.5 | Constrained weights, within-family averages, one regularized stacker and their validation |
   | §6 | Revised tasks, dependencies, timing and cut order |
   | §8 | Learned procedures can be finalists, subject to unchanged final-selection principles |
   | History | Enumerate every departure and quote the superseding instruction |

   Keep phase-plan section 11’s **0.001 tie band**. Define its existing complexity terms for composites:

   - Count unique atomic recipes, not artifact nodes.
   - Count supervised feature transformations and the learned combiner.
   - Freeze deterministic complexity ranks before evaluating the menu.
   - Use no wall-clock tie-breaker.

   Update the submission rule so a useful confirmed improvement need not exceed `0.003` to justify a pipeline check. Keep roughly four to six total uploads, each with a predeclared purpose. The public 0.83597 remains a pipeline observation, not a search objective.

   Phase 2 exits only when the frozen candidates have complete matched banks, correct learned-procedure evaluation, verified deployment artifacts, reviewed findings and reproducible source associations. A large trial count is not an exit criterion.

   The most likely formal-review blockers are:

   - Treating a combiner fitted on the global bank as independently validated.
   - Omitting the third level from Sunday’s learned-candidate selection.
   - Counting only model fitting and omitting confirmation, stopping refits or replay.
   - Leaving the old budget, promotion or “no stacker” text elsewhere in the strategy.
   - Applying the amendment before its review or modifying the active P2.1 snapshot.
   - Presenting consultation timings, including mine, as reviewed findings.
   - Calling repeated use of the official folds independent confirmation.
   - Failing to distinguish fixed weights from weights learned separately in every fitting population.

**Decisions only Sam can make**

- The two final Kaggle selections and uploads.
- Any override of a freeze, installation constraint or review requirement if the schedule becomes infeasible.

No additional Sam decision is needed to specify the search spaces, allocation rules or implementation above. Those can be made concrete in P2.0b and subjected to the existing review process.