Consultation complete. Nothing was written or executed. This is advice, not approval.

## Independent design

**Scope found in the code.** `engine.fit_recipe` is the single fit entry for production CV, the learned-procedure cache (`procedures.atomic_cv`) and the nested evaluator (`nested.refit_predict`). One dispatch there covers every population size. `recipes.validate_recipe` rejects any key set other than the twelve v1 keys before it reads the schema, so the v2 branch must be the first statement. `recipes.py` imports `songpop.artifacts`, which imports LightGBM and XGBoost, so it is not worker-safe. `banks.source_manifest` hashes only modules in the parent's `sys.modules`. `runs.check_sources` compares `packages`, in which torch reads 2.14.1 on both platforms. These four facts drive the design.

**1. v2 schema and dispatch.** New worker-safe module `songpop/phase2/recipes_v2.py` (stdlib only). Exact top-level keys:

```
schema="recipe_v2", name, family in {tabm_raw, mlp_raw, resnet_raw, ft_raw}
base, blocks, drop, estimator_seeds, deployment="fold_average"   # v1 semantics, so feature_columns() and
                                                                   # banks.validate_v2_metadata work unchanged
device: "cpu" | "cuda"
fit_policy: {kind:"stop_refit_v2", split_purpose, seed_namespace, stopping_folds:5, metric:"logloss"|"auc",
             metric_tolerance:1e-6, patience, max_epochs, refit:"epochs_v1", last_batch:"keep_short"|"merge_singleton"}
layout:  {kind:"legacy_v1", scaler}  |  {kind:"split_v2", continuous:[..], flags:[..], categorical:[..], scaler,
          unknown_category:"reserved_index", flag_path:"categorical2"|"passthrough"}
embeddings: {kind:"none"} | {kind:"ple", d_embedding, n_bins, version:"B", activation, bins:"quantile"}
            | {kind:"periodic", d_embedding, n_frequencies, frequency_init_scale, activation, lite}
arch: every constructor argument explicit (TabM: n_blocks,d_block,dropout,k,arch_type,activation,start_scaling_init;
      ResNet: n_blocks,d_block,d_hidden_multiplier,dropout1,dropout2; FT: n_blocks,d_block,attention_n_heads,
      attention_dropout,ffn_d_hidden_multiplier,ffn_dropout,residual_dropout)
optimizer: {name:"adamw", learning_rate, weight_decay, betas, eps, amsgrad:false, foreach:false, fused:false,
            batch_size, decay_groups:"all"|"authors_ft"|"no_bias_norm_embedding"}
```

Validator rules: `embeddings.kind != none` requires `layout.kind == split_v2`; `lite` requires `activation`; `d_block % attention_n_heads == 0`; `max_epochs` is a stored field, and the manifest validator (not the recipe validator) enforces the plan's 300/600 rule. A helper `legacy_tabm_to_v2(v1_recipe)` produces the gate recipes (legacy layout, purpose `tabm_stop`, namespace `tabm`, k 16, random-signs, patience 20, max 300, decay `all`, cpu); a predicate `is_legacy_equivalent` marks them non-scientific.

Dispatch edits, kept tiny:
- `recipes.validate_recipe`: first line delegates on `schema == "recipe_v2"`; the v1 path and `canonical` are untouched, so every v1 hash is preserved. Guard test: `recipe_hash` of every recipe under `configs/phase2/` and of `menu.json` `atom_recipes` equals the stored hash.
- `engine.fit_recipe`: after the raw-column and length checks, `if recipe["schema"] == "recipe_v2": return raw_adapters.fit_raw(recipe, rows, y, groups, ctx)`. `run_recipe_cv` is unchanged because the returned object exposes `features.columns_`, `features.lineage_`, `receipt`, `family`, `model_inputs`, `predict_each`, `predict_high`.
- `runs.fitted_config`: `resolved_parameters` branches to `raw_adapters.resolved_parameters(recipe, seed)`, which resolves every default and the derived seeds without fitting or torch. Today it calls `make_model`, which would construct an estimator.
- Also v2-blind today and needing a branch before any v2 candidate reaches R1: `models.family_packages` (`OPTIONAL_PACKAGES` has no v2 families, so identities would omit torch), and `submission.member_input_classes`, which reads `recipe["preprocessing"]`.

Import sides: `raw_adapters.py` (parent) imports `recipes_v2`, `raw_features`, `raw_splits`, `encoders` and `raw_worker` at module level, with torch imported only inside worker functions, exactly as `tabm.py` does. That is what puts the worker's modules into the parent's source manifest. The worker never imports `songpop.artifacts`, `phase2.models`, `recipes` or `engine`; it re-validates the recipe with `recipes_v2` and asserts that forbidden set after its own imports, reporting it in the hello.

**2. Smallest identity change.** One function `runs.runtime_route(root)` returning `{policy_id, policy_sha256, wheel_lock_sha256, bootstrap_lock_sha256, platform, machine, torch_runtime, implementation_sha256}`, where the last hashes the installed `tabm.py`, `rtdl_num_embeddings.py` and `rtdl_revisiting_models.py` files located with `importlib.util.find_spec` (no import). Then:
- add `"runtime_route"` to `fitted_config` and to the `blend_run` config; `make_run_id` changes for every new run automatically;
- `check_sources` compares `runtime_route` only when the stored config has one, so historical banks keep their current behaviour (replay at the named commit);
- `blend_run` refuses parents whose routes differ or are absent (plan: incompatible banks cannot blend);
- `procedures.atomic_request` becomes `atomic_v2` with an `identity` field; the `packages_of` callable threaded through `nested` returns `{"packages": ..., "runtime_route": ...}` via a new `runs.identity_for`. This is what stops a node nested run from hitting any Mac-era cache entry when sources happen to be unchanged;
- the worker hello must match the route (torch runtime string, CUDA, cuDNN, implementation hashes) and every worker module hash must equal the parent manifest entry, else the worker is refused;
- `compare_v2` is unchanged; comparisons with historical banks read only its array entries, as the plan already states.

**3. Worker protocol and state.** Protocol `raw-worker-v1`, same framing and request-id discipline as `tabm.Worker`, one worker per device, spawned with `device_policy.json["worker_environment"]` so `CUBLAS_WORKSPACE_CONFIG` precedes CUDA init. Ops: `fit` (recipe, raw D rows, labels, groups, context, all estimator seeds, optional `diagnostics`) and `predict` (net states, resolved arch, bins, Z). Inside `fit`: `partition` of the context-seeded or injected S/V; `RawFeaturePipeline` on S under `FitContext(ctx.split_seed, f"{ctx.fold}/S")`; layout and bins on S; per-seed stopping; everything from S discarded except digests; `RawFeaturePipeline`, layout and bins on D; per-seed final training for that seed's chosen epoch with the legacy epoch policy. The returned D state is `{features, layout, bins, arch_resolved, net_states: {seed: {name: ndarray}}, receipt}` with no Torch object. The parent builds Z for both `model_inputs` and `predict_each` from the same D objects, so the duplicate check and the predictions see one matrix. Shared transforms are structural: one request serves all seeds. Diagnostics inject the S/V split and S encoder folds together (D folds optionally), validated with `partition` and `crossfit` on both sides and flagged in the receipt; production passes none and the schema has no field for it.

Receipt validator `validate_raw_receipt`: `chosen_epoch == stop_rule(values)[1]` (AUC runs apply the existing `stop_rule` to the negated AUC series, which gives the plan's earliest-within-tolerance maximum without a second implementation); `stopped_by` equals the legacy rule; `stopping_updates == epochs_evaluated * batches(n_S)` and `final_updates == chosen_epoch * batches(n_D)` where `batches` is ceil for `keep_short` and ceil minus one when `n % b == 1` for `merge_singleton`; S plus V rows equal D rows; seeds equal the namespace derivation; worker block equals route and recipe device; `subfits` equals `expected_subfits(recipe)` = 2 feature fits, 2 layout fits, 0 or 2 bin fits, 2 trainings per seed. The legacy `tabm.validate_receipt` and the combiner `lambda`/`C` check follow the same pattern; run both read-only against the ten TabM banks and the P2.6b procedure banks before wiring them into `validate_v2_metadata`. `completed_fits` keeps its estimator meaning; `fitted_run` adds a summed `subfits` dict, and the cost graph multiplies cache misses by measured per-route cost that already contains its subfits.

Unavoidable per-family differences from the installed sources:

| Family | Installed fact | What it forces |
|---|---|---|
| TabM embedded | `num_embeddings` applies to all of `x_num`; `make` flips defaults to `start_scaling_init="normal"`, `n_blocks=2`; PLE needs `version="B"` | Flags and categories leave `x_num`; the scalar arm of a matched pair must use the same split layout, so the TabM family has two scalar layouts: legacy (gate runs, fresh controls) and split (pair arms) |
| TabM categorical | `_OneHotEncoding` needs indices in range, no unknown path | Reserved index fitted on S or D; one-hot and token arms differ for unseen categories |
| MLP, ResNet | Flat `d_in` only, no embedding or group helper | A project wrapper module (embed continuous slice, flatten, concat passthrough) that must be hashed and tested; decay-group policy is a project choice |
| ResNet | `BatchNorm1d` in each block and the head | `merge_singleton`; running statistics are buffers in the state; eval mode is row-independent so padding is safe |
| FT-Transformer | Continuous features are always `LinearEmbeddings` tokens; `make_parameter_groups` builds a `set`; `ffn_dropout` is a separate argument | No scalar-vs-embedded factor exists; one-hot arm adds 18 tokens; sort groups by parameter name for receipts (numerics are unaffected with `foreach=False`); declare `ffn_dropout` |
| PLE bins | `compute_bins` rejects constant columns and `n_bins >= len(X)`; `unique()` shrinks bins | Only declared continuous columns are binned; record per-column bin counts; a constant column in S is a recorded fit failure |

**4. Tests, duplicate invariance, legacy failures.** Existing: the eight boundary tests, v1 engine label and poisoning isolation, the P2.5 worker tests. Missing, with the known-wrong variant each must fail on: V-label flip with injected S/V and S folds, comparing S and D digests and the network state after a fixed horizon (fails on features fitted on D before the split); S-label positive control on a shared-date fixture; outer flip through `run_recipe_cv` for a D-arm v2 recipe; poisoning through the worker path; injection rejection on both sides; 400-row legacy equivalence bitwise, then old-adapter fresh-process repeat and the six 15-fold gate runs; three-seed run equals the mean of three one-seed runs bitwise; forged receipts; hello mismatch; route mismatch in `check_sources`, `blend_run` and the cache; identical rows at different padded-batch positions bitwise per seed on CPU and CUDA; ResNet sizes congruent to one modulo batch at S, D and nested sizes; unseen category per layout; hash regression guard. For neural adapters assert bitwise, not 1e-12, because every eval-mode forward here is row-independent.

The three legacy failures, with evidence only:
- Anchor: the check asserts exact equality against the Mac phase-1 bank. Finding 031 measured the node's H refit (same anchor recipe) against Mac at 5.6e-17 on `oof` and 1.1e-16 on `test_by_fold`, with `test_mean` and `fold_auc` equal and node repeats byte-equal. The failure names exactly `oof` and `test_by_fold`. Measurement: max absolute difference per array between `runs.fitted_run(anchor)` and `PHASE1_LGBM_F2`; expect at most about 2.3e-16 on those two, zero elsewhere, and a few last-digit differences in the P1.3 CSV bytes. Anything larger is a defect.
- kNN: a bare assert, line unknown. Bisect the asserts; the exact-equality ones at lines 312, 318 and 321 compare identical inputs in different batch compositions. Print the magnitude. A result near 1e-16 would be a BLAS summation-order effect, which matters because it means exact duplicate invariance is not guaranteed for brute kNN on this node while the engine's 1e-12 check still holds.
- Comparisons: failed in 0.16 seconds with no message in the progress log. Run the filtered runner with `comparisons:` to obtain the exception before touching anything. Candidates to discriminate by the message: a `SourceMismatch` from `save_run` (a file in the running suite's manifest edited on disk), a `load_logged` hash mismatch, or a bootstrap tolerance.

What to do with the anchor assertion once confirmed is a gate decision, not a tolerance tweak: plan section 12 requires legacy verification to pass under final sources, and an exact cross-platform assertion cannot. Re-scoping it to the v1.3 same-node wording needs R2.

**5. Bounded first unit.** Unit A, parent-side only, no torch: `recipes_v2.py`, `runtime_route` and `identity_for`, the dispatch branches listed above, `atomic_v2`, blend refusal, hash guard, the two deferred validators run read-only against existing banks, and a `check_p29_identity` script. Unit B: `raw_worker.py` and `raw_adapters.py` for scalar TabM in the legacy layout on CPU, with D-arm blocks allowed, the test list above, the 400-row equivalence and the six gate runs. Unit C: the CUDA route on that adapter with fresh-process, restart and post-unrelated-work repeats at full and nested sizes. Unit D: split layout, embeddings, MLP, ResNet, FT, AUC stopping, pilots, cost graph, then manifests. Calibration and retrieval later. R2 items hidden inside this work: the two-scalar-layout rule and which layout the TabM 32 controls use; the MLP/ResNet decay policy and FT `ffn_dropout`; the anchor check's status; any nonzero CUDA tolerance. Everything else is implementation.

## Critique of Codex's incremental route

1. **The first unit is too wide.** It bundles dispatch, CPU and CUDA routes, identity, receipt validators and injection. CUDA is not on the gate's critical path. Split as A, B, C above so each is reviewable in an hour and the gate runs start sooner.
2. **Editing legacy modules while the unchanged suite runs will corrupt its verdict.** `save_run` compares the on-disk file hashes with the manifest taken at import, so a disk edit to `engine.py`, `recipes.py`, `runs.py`, `procedures.py` or the uncommitted `tools/run_checks_filtered.py` (it is `sys.argv[0]`, so it is in the manifest) makes every writer-based check fail spuriously. Do Unit A in new files plus a staged patch, or in the `../Mini_project_1_p29` worktree, and check the runner's on-disk hash against the logged `d7cbdf80…`.
3. **"Delegate only on the version marker" is right but incomplete.** `fitted_config`, `family_packages`, `member_input_classes` and `make_preprocessing` all assume v1 keys. Without the submission branch a v2 atom could be scored but never released, and without the packages entry its identity omits torch.
4. **Worker-owned features need the predict path pinned.** The design does not say that Z for prediction comes from the parent's D objects. If the worker also transforms queries, `model_inputs` and `predict_each` can silently diverge. The hello should assert the forbidden-module set, not only hashes.
5. **Multi-seed consistency should be bitwise.** The design says "compare"; with context-derived seeds and shared transforms the mean of single-seed runs must equal the multi-seed run exactly, and any difference is a seed or ordering leak.
6. **Receipt validators need the arithmetic.** Update counts, the `stopped_by` rule and the singleton-batch adjustment are not stated. Wiring them into `validate_v2_metadata` changes how historical TabM banks load, so the read-only pass against the ten existing banks must come first.
7. **Route binding misses four consumers.** `check_sources`, `blend_run`, the worker hello check and the cache request schema. The route must also carry the installed implementation file hashes, since the wheel lock pins wheels and not the executing files.
8. **Family-shaping facts are absent.** Two scalar TabM layouts, FT's lack of a scalar arm, the MLP/ResNet wrapper being project code, and the FT group ordering being a receipt matter rather than a numerics matter. These change the manifest generator and the pair validator and should be settled before any manifest is called executable.
9. **Agreed.** No placeholders that can score an unimplemented recipe, partial adapters never count toward coverage, one GPU worker, and the rejection of the two-worker requirement.

## Failure modes and recommended sequence

Most consequential failure modes, in order: a disk edit during the suite run wasting the run and producing a wrong verdict; the v2 branch placed after the key check or altering `canonical`, changing every historical hash (the guard test catches it); the route left out of the cache request, letting a node nested run reuse Mac-era entries; worker code outside the manifest, so training-loop edits do not change run identity; flags or one-hot columns reaching `compute_bins`, failing only at nested sizes late in the schedule; ResNet singleton batches crashing only for particular population sizes; a CUDA route accepted on one-recipe pilot evidence; `max_epochs` implemented as a runtime rule instead of a stored field.

Sequence:
1. Leave legacy files alone until the suite exits; obtain the three failure messages with filtered runs (read-only, separate processes); record magnitudes in the suite verdict and Finding 031.
2. Unit A in new files plus a staged patch; hash guard; deferred validators run read-only; commit; review.
3. Unit B; old-adapter fresh-process repeat; 400-row equivalence; six gate runs; Finding; R2 on the equivalence outcome and any contract deviation.
4. Final-source refit of H and E's six atoms under the new route, arrays-only comparison with the Mac banks recorded, node E bank as the discovery comparator.
5. Unit C; freeze the device policy; R2 before any CUDA score.
6. Unit D with representative full and nested pilots and the subfit-based cost graph; manifests only when every field maps to an adapter and rejection tests exist.
7. The measured launch proposal to Sam.

Not checked: I did not run any code, so the three failure causes above are hypotheses tied to specific measurements, and the installed-package facts come from reading the site-packages sources, not from executing them.