I recommend preserving P2.8’s decisions, spending the new initial comparisons mainly on date encodings, and running feature batches through a small manifest-based controller. The existing family-keyed wave planner is a poor fit for matched feature comparisons and for re-tuning two feature sets from one family.

I read the requested sources and existing results. I fitted nothing and computed no new score. I did not read Claude’s positions file. Cost estimates below are arithmetic projections, not new measurements.

1. **Fix the accounting and allocation rules in the new P2.4 execution section before running anything.**

   In [the phase plan](/Users/zwl0331/Desktop/Mini_project_1/doc/plans/phase2-models-features.md), add section 18 and update the task-table record path.

   Use these rules:

   > P2.4 starts with 32 comparison slots. Fourteen are imported from P2.8 and eighteen are new. Imported comparisons retain their original references and decisions. Support refits do not consume comparison slots, but their runtime is reserved. Feature expansion batches contain sixteen comparisons, up to ninety-six comparisons in total. Re-tuning has a separate ceiling of four selected combinations and thirty-two new configurations per combination.

   There is one ambiguity to resolve explicitly. Section 5.6 describes a **family counter**, while section 16.3 says the **initial feature allocation** progresses if any imported or new comparison qualifies.

   My proposed interpretation is:

   > The mixed-family feature stage has one allowance counter. Its initial progress flag is the OR of the route flags of its initial comparisons, including the fourteen imports. Later batches use only their newly evaluated comparisons to reset or decrement this counter. Route calculations still use each challenger’s actual model-family incumbent and previous best family blend.

   This preserves section 16.3 without allowing B12 and B14 to reset the counter repeatedly. It needs to be stated in the reviewed execution protocol, not inferred by the code.

   Keep re-tuning allocation separate, as described in item 7. Do not change final selection or deployment rules.

2. **Use the following initial 32 comparisons. Resolve every base by a deterministic rule.**

   Freeze these aliases from converged records before the new comparisons:

   | Alias | Rule |
   |---|---|
   | `H` | The submitted phase 1 LightGBM F2 recipe, unchanged. |
   | `L` | P2.2’s selected LightGBM F2 recipe, `p22_lightgbm_b3_001`. |
   | `C` | P2.3’s best complete CatBoost F2 recipe by unrounded mean fold AUC; ties by recipe ID. Use the specified XGBoost fallback only for a reviewed CatBoost omission. |
   | `E` | The Extra Trees contribution winner already identified in Finding 012, the Extra Trees anchor. |
   | `S` | P2.2’s best F1 spline logistic recipe. The existing log identifies `p22_logistic_b3_002`, with C 0.0531 and eight knots. |
   | `K` | P2.2’s best kNN recipe. The existing log identifies the anchor, with 60 distance-weighted Euclidean neighbors and alpha 1. |
   | `M` | P2.3’s best complete EBM F2 recipe, using the same score and tie rule as `C`. |
   | `R` | The strongest complete standalone recipe available before P2.4 that contains `days`, among the declared P2.2, P2.8 and P2.3 records. |

   Preserve the historically selected forest recipes. Do not rerank P2.2’s tie-sensitive records after deterministic prediction changes.

   | Slots | Matched base | Challenger |
   |---|---|---|
   | B1–B14 | Original P2.8 bases | Import unchanged, including route flags. |
   | B15 | `C` | Add `date_count`. |
   | B16 | `C` | Add `date_count`, `date_te30`. |
   | B17 | `C` | Add `music`. |
   | B18 | `C` | Add `interactions`. |
   | B19 | `M` | Add `music`. |
   | B20 | `L` | Add `date_te10`. |
   | B21 | `L` | Add `date_te30`. |
   | B22 | `L` | Add `date_count`, `date_te10`. |
   | B23 | `L` | Add `date_te3`. |
   | B24 | `L` | Add `date_count`, `date_te3`. |
   | B25 | `L` | Add `date_te100`. |
   | B26 | `L` | Add `date_count`, `date_te100`. |
   | B27 | `E` | Add `date_count`. |
   | B28 | `E` | Add `date_count`, `date_te30`. |
   | B29 | `S` | F2 with the selective spline preprocessing defined below. |
   | B30 | `S` | F1 plus `cyclic`, with that preprocessing. |
   | B31 | `K` | Replace audio standard scaling with the specified robust scaling. |
   | B32 | `R` | Drop `days`. |

   B15–B19 fulfill the outstanding reservations in sections 6.5 and 16.3. If EBM is omitted, follow section 16.3: record B19 as omitted. Do not silently invent a replacement to make the executed count equal 32.

   This allocation tests whether the useful LightGBM date result depends on smoothing or count inclusion. It also tests dates on the strongest established Extra Trees contributor. That is more valuable initially than repeating every weak anchor block across every family.

   Leave additional interactions, era-relative features, weighting changes and most removal checks for expansions. Leave identifier extensions, the incomplete-row ablation, neural work and interpretation outside this task.

3. **Use a manifest-based feature controller, while reusing the existing fitting and allocation functions.**

   Extend the pattern in [P2.8’s read-only judge](/Users/zwl0331/Desktop/Mini_project_1/analysis/p28_feature_checks.py). Do not force feature comparisons into `protocol.py`’s family-wave abstraction.

   Suggested additions:

   ```python
   # songpop/phase2/feature_protocol.py
   resolve_references(protocol, audited_records) -> dict
   plan_initial(protocol, references) -> dict
   plan_expansion(protocol, closed_batches) -> dict
   choose_retune_units(protocol, closed_batches) -> list[dict]
   plan_retune_batch(protocol, unit, batch_index, closed_batches) -> dict
   judge_feature_batch(manifest, audited_banks) -> dict
   audit_feature_task(context, protocol) -> dict
   ```

   These should be pure planning or read-only judging functions. Fitting stays in the existing engine.

   Add a protocol file at `configs/phase2/p2.4/protocol.json`. Save immutable generated manifests under an explicitly declared artifact directory. A manifest should contain:

   ```text
   schema, task_id, batch_id, batch_kind, protocol_sha256
   source_manifest_sha256, split_seeds, estimator_seed_policy
   prior_record_ids_and_hashes, import_receipt_ids
   ordered_comparisons:
       comparison_id
       comparator_recipe and expected_run_id
       challenger_recipe and expected_run_id
       changed_fields
       generation_rule
       budget_charge
   frozen_allocation_state:
       H, G
       family_incumbents
       previous_best_family_blends
       allowance_before
   proposal_seed, exclusions, timing_reservation, start_cutoff
   ```

   Save complete recipes, not only aliases or filenames. Resolve aliases to immutable run IDs and bank hashes.

   Give `scripts/run_experiments.py` explicit feature-plan, feature-run and feature-audit modes. Reuse `run_one`, comparison logging, bank validation and `allocation.routes`. Keep `analysis/p24_summary.py` read-only.

   Import the new execution modules before capturing the source manifest. Otherwise running and verifying through different import paths can produce different manifests even when the computations agree.

   **Imports need a receipt.** P2.8’s route results are partly represented by its saved output and judge, rather than an existing general feature-allocation record. Save an import receipt binding:

   - The P2.8 manifest, judge and output hashes.
   - The original comparator and challenger banks.
   - The original allocation references.
   - The original route flags and comparison IDs.

   Verify those original decisions against their original state. Do not evaluate them against P2.4’s newer incumbent.

4. **Separate matched feature effects from allocation decisions.**

   For every new comparison, report two distinct results:

   - **Matched effect:** challenger minus its otherwise identical base.
   - **Allocation result:** the section 5.6 routes against the frozen family incumbent, `G`, `H` and previous best family blend.

   Finding 016 illustrates why this matters. B12 qualified through ensemble contribution despite a worse standalone matched result. B14 qualified through several routes. Conversely, a positive matched change need not authorize further engineering.

   For new batches:

   - Keep `H` fixed at the submitted anchor throughout P2.4.
   - Define `G` as the strongest standalone recipe in the declared, completed reference pool.
   - Use the actual family incumbent, including earlier successful feature recipes.
   - Use the best previous declared blend of that family with the same `H`, with `H` itself as the floor.
   - Freeze all these references before any challenger in that batch runs.
   - Evaluate the two fixed challenger weights, 0.1 and 0.25.
   - Use all three official split seeds and estimator seed 0.

   Do not update an incumbent halfway through a batch. Serial execution order must not change its allocation outcome.

   Use the existing 2,000-draw paired cluster bootstrap for matched effects and decision-driving comparisons. Save its seed and all repeat differences. Its interval remains conditional on the saved predictions. It is neither a multiple-testing correction nor an independent confirmation.

   The imported initial progress grants two expansion opportunities under the proposed feature-stage counter. Later progress resets the allowance to two; failure consumes one. Stop at zero, the comparison ceiling, declared exhaustion or the runtime guard.

5. **Add the smoothing blocks and preprocessing alternatives without changing old recipe semantics.**

   In [features.py](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/features.py), register `date_te3` and `date_te100` in `STATEFUL` and `BLOCK_COLUMNS`. The existing `DateTargetEncoder(m)` already supports them.

   In [recipes.py](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/recipes.py), replace the special-case rejection of simultaneous `date_te10` and `date_te30` with:

   ```python
   DATE_TE_BLOCKS = {"date_te3", "date_te10", "date_te30", "date_te100"}
   if len(set(blocks) & DATE_TE_BLOCKS) > 1:
       raise ValueError("at most one date target encoding per recipe")
   ```

   Preserve the encoder’s existing inner-seed derivation. Different smoothing values should use identical inner splits for the same fitting population.

   For preprocessing, add named alternatives in [models.py](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/models.py). Do not silently change `spline` or `knn_blocks`.

   **`spline_calendar`:**

   - Keep the existing F1 core: phase 1 logs, cubic splines on the ten continuous audio fields and year, then scaling.
   - Keep musical categories one-hot encoded.
   - Treat F2’s additional numeric date fields as median-imputed, standardized linear inputs.
   - One-hot encode `date_precision`.
   - Treat cyclic coordinates as linear inputs, not spline inputs. Impute missing coordinates to zero and append one fixed missingness indicator per phase pair.
   - On F1 without added blocks, require the transformed matrix and predictions to reproduce the existing spline recipe.

   This avoids the current generic `spline` behavior of splining every non-musical numeric column, including any newly added sine and cosine coordinates.

   **`knn_blocks_v2`:**

   - Preserve the original ten-audio, year and categorical blocks.
   - Add `audio_scaler: "standard" | "robust"`.
   - For robust scaling, use `RobustScaler(quantile_range=(25, 75), unit_variance=True)` after the existing log transforms and median imputation.
   - Keep year standard-scaled. B31 should change audio scaling alone.
   - Retain the metric-specific block normalization.
   - Permit either no block or `cyclic`, with a declared `calendar_weight`.
   - For cyclic inputs, use six coordinates plus three fixed missingness indicators. Zero-impute coordinates. Scale the nine-coordinate block by `calendar_weight / sqrt(9)` for Euclidean distance or `/ 9` for Manhattan distance.
   - Continue to require one estimator seed, no training weights, and deterministic prediction.

   Validate the allowed combinations explicitly. Preserve the old recipe hashes and old factory behavior.

6. **Generate expansions from a finite, deterministic queue.**

   I would declare the following target allocation for each sixteen-comparison expansion:

   | Queue | Target slots | Candidate rule |
   |---|---:|---|
   | Date encodings | 6 | Further smoothing and count combinations on selected tree bases. |
   | Block combinations | 2 | Combine individually useful, compatible blocks. |
   | Smooth/kNN preprocessing | 2 | Follow the declared preprocessing menu. |
   | Removal checks | 4 | Apply section 6.6 to the strongest two eligible recipes. |
   | Other tree blocks | 1 | Interactions, cyclic or era-relative features on another tree family. |
   | Training weights | 1 | Inverse-group-size sensitivity on a promising non-kNN recipe. |

   Empty queues release slots to the first nonempty queue in this order. Deduplicate against all earlier declared computations. Do not pad a batch with a new name for an old recipe.

   Make candidate generation precise:

   - **Dates:** prioritize bases with a qualifying date-feature result. Within a base, complete untried smoothing values `{3,10,30,100}`, with and without count, nearest to the successful smoothing value on the log scale first. Use a fixed numeric tie rule. Admit at most two date bases per expansion.
   - **Combinations:** union two useful block sets on the same model configuration. Require evidence for each component under the declared rules. Never include two target encodings. Do not create an “all features” recipe.
   - **Smooth preprocessing:** F2 plus cyclic after the separate initial checks; then selected date-count or target-encoding additions as linear inputs. Each remains a named comparison.
   - **kNN:** robust versus standard scaling, the normalized Manhattan alternative, and cyclic weights `{0.5, 0.25, 1.0}` in that declared order. Do not tune every preprocessing dimension simultaneously.
   - **Removals:** without `days`; without `day` and `day_of_year`; without `weekday`; without the three musical categories; and remove an added block from a successful combination. Reject inapplicable or already-tested proposals.
   - **Other trees:** choose the best available contributor among the declared Extra Trees, XGBoost and HistGB representatives. Use a fixed block order. The negative anchor checks do not prove that these blocks cannot help another family.
   - **Weights:** test otherwise identical promising recipes. Keep evaluation row-weighted and target encoding row-weighted.

   Save the parent-selection ranking, queue order, generated candidates and rejected duplicates. The next batch may depend only on closed earlier batches, never on partial results from the batch being planned.

   The exact queue policy is a proposed implementation choice. It should appear in section 18 because section 6 does not currently specify it.

7. **Re-tune up to four feature combinations through separate search-unit IDs.**

   Do not rename model families to values such as `lightgbm_dates`. Use:

   ```text
   family = "lightgbm"
   unit_id = hash(fixed feature and preprocessing specification)
   ```

   A unit fixes:

   - Family, base, blocks and their order.
   - Drop list and training-weight policy.
   - Preprocessing method and nonsampled settings.
   - Estimator seed policy.

   Model parameters and explicitly declared preprocessing parameters, such as spline knots, remain searchable.

   Choose units once the feature stage closes:

   1. Consider only combinations with a documented section 5.6 qualifying result, including preserved P2.8 qualifications.
   2. Select up to two by standalone performance.
   3. Select up to two additional units by their best declared contribution to `H`.
   4. Deduplicate units; break ties by canonical ID.
   5. I recommend at most two units from one family. Record this diversity cap before scores.
   6. If fewer contribution-qualified units exist, fill remaining places from the standalone ranking, subject to qualification and runtime.

   This keeps B12 eligible without pretending it won standalone.

   Reuse [refine.py](/Users/zwl0331/Desktop/Mini_project_1/songpop/phase2/refine.py), with the feature specification fixed:

   - Sixteen configurations per batch.
   - Twelve local proposals and four broad draws.
   - One to three parameter changes per local proposal.
   - Use the family’s reviewed parameter types, admissible regions and hard bounds.
   - Include the two iteration variants in the first local slots when new and admissible.
   - Use the best standalone and best distinct contributor within the same unit as parents. If only one parent exists, use it for all local slots.
   - Exclude previous configurations with that feature specification.
   - Do not reset to F2 in the broad quarter.

   Qualification grants each selected unit up to two batches, within its thirty-two-configuration cap. Continue to calculate routes against the **actual family incumbent and family contribution history**, not a weaker unit-specific baseline.

   For fairness, freeze allocation state for all first retuning batches together. Update it only after that logical round closes, then freeze the second round. Execution can remain serial.

   No Optuna installation is needed. The typed local proposer and broad quarter already provide the required refinement mechanism.

8. **Reserve expensive-model costs explicitly; use all three repeats unless the plan declares a different procedure before scores.**

   For P2.4, I recommend all-three-repeat runs even for the slightly over-120-second CatBoost case. The stage measures relatively small matched changes. Screening and completing selected recipes would complicate the comparison and allocation records.

   Time the actual feature-bearing recipes without scoring before fixing reservations:

   - A representative date-encoding tree recipe.
   - The selected CatBoost base with the expensive feature combination.
   - The expensive end of any proposed CatBoost retuning space.
   - The selected EBM recipe with added features.
   - The new preprocessing paths.

   Do not use 133 seconds as an upper bound for deeper or longer CatBoost proposals.

   Illustrative costs using the supplied 133-second figure:

   | Work | Fitting | Fitting plus one full replay |
   |---|---:|---:|
   | Four initial CatBoost checks | About 9 minutes | About 18 minutes |
   | One 32-configuration CatBoost unit | About 71 minutes | About 142 minutes |
   | Three 32-configuration units at 20 s/run | About 32 minutes | About 64 minutes |
   | One expensive CatBoost unit plus those three units | About 103 minutes | About 206 minutes |

   These exclude support refits, EBM, feature expansions, bootstrap calculations and review.

   Start with capacity reserved for one expensive CatBoost unit. Admit a second only if measured costs leave room for its complete replay and completion review. This is a runtime decision, not a rule that CatBoost may contribute only one finalist.

   Use serial execution. Building log shards now would add infrastructure risk without a demonstrated need.

9. **Refit active bases under the P2.4 source manifest, while preserving historical artifacts.**

   Adding blocks or preprocessing changes the source manifest even when an old recipe’s computation is unchanged.

   Therefore:

   - Keep every P2.8 bank and comparison immutable.
   - Create fresh P2.4 support banks for the bases and allocation-reference recipes actually needed.
   - Record `rebase_of_run_id` and the old bank hash.
   - Bind new comparisons to these fresh bases.
   - Require `H` to match the phase 1/P2.1 anchor in all ten arrays.
   - Require unchanged deterministic paths to reproduce their prior arrays.
   - Treat pre-P2.8 forests under their documented historical replay policy. Prefer an existing deterministic P2.8 reference where available.
   - Do not rerank historical winners because a support refit changes a tie-sensitive AUC.

   Resolve and audit references before challengers run. A reference record must agree with its bank, recipe, data, folds and seed policy. Merely finding a matching name in the log is insufficient.

   Full verification should refit each distinct **new P2.4 run once**. Imported P2.8 results need their historical audit and replay association, not fourteen unnecessary new feature fits.

   Extend task-level audit dispatch for P2.4. The current family-wave audit cannot be assumed to understand a feature manifest.

10. **Include coverage evidence and self-checks that exercise the new failure modes.**

   Section 6.4 explicitly requires new coverage evidence. Produce it from the actual fitting populations:

   - Inner held-out training rows against their inner fitting dictionaries.
   - Outer validation rows against their outer fitting dictionaries.
   - Test rows against each outer fitting dictionary.

   Report exact-date coverage and distinct-group support, including counts with support 0, 1, 2–4 and at least 5. Distinguish row-weighted summaries from distinct-date summaries.

   These are descriptive checks. Do not fit anything on test rows or choose smoothing from test-distribution diagnostics. Describe the encoding as an exact-date association, not album identification.

   Add checks for:

   | Area | Necessary check |
   |---|---|
   | Smoothing | Hand-calculated values for m=3 and m=100; unseen dates use the permitted fitting prior. |
   | Group isolation | With fixed explicit inner splits, changing a held-out group’s labels cannot change its encoded values. |
   | Date count | Duplicate rows do not increase distinct-group support; the whole own group is excluded. |
   | Date precision | Year-only and year-month strings remain distinct categories; missing calendar components stay missing before preprocessing. |
   | Preprocessing | Extreme held-out values cannot alter fitted scalers, spline knots or categories. |
   | Compatibility | New spline preprocessing on plain F1 reproduces the old F1 matrix and predictions. |
   | kNN | Standard/no-calendar mode reproduces the original representation; metric normalization, zero distances and identical queries remain deterministic. |
   | Planning | Adding future records or permuting log order cannot alter an already-declared batch. |
   | Imports | A newer `G` cannot change an imported route flag; imports cannot reset later allowances. |
   | Retuning | Two units from one family have separate IDs and fixed features; neither silently becomes F2. |
   | Audit | Tampered comparator hashes, allocation state, proposal lists or route flags are rejected. |
   | Resume | An interrupted run resumes only against the same immutable declaration; a failed configuration is not silently skipped. |

   Synthetic or tiny fixtures suffice for readiness checks. Completion verification covers the actual banks.

11. **Use two checkpoints and protect the Saturday handoff.**

   Keep one P2.4 review record with readiness and completion checkpoints.

   **Readiness deliverables:** section 18, protocol and import receipt, initial comparison definitions, deterministic expansion and retuning rules, code, self-checks, timing reservations and source fingerprint.

   **Completion deliverables:** all declared outcomes, omissions, banks, comparisons, allocation decisions, verification output, findings and the handoff index.

   A workable latest-target schedule, assuming P2.3 converges around Friday 14:00:

   | Time, PDT | Work |
   |---|---|
   | Friday 14:00–16:00 | Implement and check against P2.3’s converged source. Resolve base aliases. |
   | 16:00–17:00 | Readiness review and correction buffer. |
   | 17:00–19:00 | Support refits, initial comparisons and audit. |
   | 19:00–21:00 | Productive feature expansions. |
   | 21:00–Saturday 04:00 | Selected retuning, with replay costs reserved. |
   | Saturday 04:00–06:00 | Finish verification, findings and completion record. |
   | By 06:00 | Request completion review. |
   | By 09:00, with 10:00 as the external guard | Converge or record reviewed omissions. |

   Use a batch-admission rule based on measured fitting plus replay time, a declared margin, and completion-review time. The Saturday 08:00 limit is the last permissible start, not a target for launching a large batch.

   Cut in this order:

   - Feature expansions without recent progress.
   - Redundant combinations and removal checks.
   - Second retuning batches and lower-priority retuning units.
   - Expensive additional CatBoost units whose replay would threaten convergence.

   Preserve the reserved initial comparisons, grouped fitting, reference checks and complete verification. Do not compress review to compensate for extra trials.

12. **Hand P2.6b a candidate inventory, not a prematurely selected winner.**

   Add reviewed findings for:

   - Matched feature effects and their repeat differences.
   - Date coverage and support under actual dictionaries.
   - Allocation decisions, including imported decisions.
   - Re-tuning outcomes, actual costs and omissions.

   Allocate finding numbers after P2.3’s ledger entries are settled.

   Export an index containing each usable recipe’s bank hash, source/replay revision, complete seed status, feature specification, standalone result, contribution results, comparison lineage and qualification history. Include useful lower-standalone contributors, not only the four retuned units.

   P2.6b must still establish its own `H*`, contribution comparisons, pool and estimator-seed policy. P2.4’s weights of 0.1 and 0.25 are exploration checks, not selected deployment weights.

   State the statistical limitation plainly: these are adaptively selected results on repeatedly used folds. More feature trials increase selection optimism. Neither three repeats nor conditional bootstrap intervals removes it.

   No report, Colab or interpretation work belongs here. After P2.4 converges, pause as Sam instructed.

**Decisions only Sam can make**

- Whether and when to resume after the requested P2.4 pause.
- Whether to upload an optional submission. I would not make one a P2.4 dependency.
- Any change to the scheduled freezes or the supplied-data-only restrictions.

The remaining choices above can be resolved in the P2.4 execution protocol and its existing review process.