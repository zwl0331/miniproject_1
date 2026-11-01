# P2.9a node migration and implementation contracts

2026-11-01. Draft. The reviewed P2.9 plan remains authoritative. This document specifies the next implementation and records the migration boundary. It does not authorize the 274-configuration launch or claim new model performance.

## Environment route

The node exposes an A100 MIG 2g.20gb device. The cgroup CPU quota is 32 core equivalents and the memory limit is 45 GiB. The 127 visible CPU IDs are not an allocation of 127 cores. Start one GPU fitting worker and use explicit four-thread limits for CPU fits. Concurrent work must be costed against the quota and measured memory, including process/library overhead.

Python 3.12.16 is installed through conda-forge. All historical constrained package versions present in the new environment are unchanged. The Linux transaction adds the CUDA dependencies and the authors' rtdl-revisiting-models 0.0.2 implementation. `configs/phase2/p2.9/install/linux-cuda.txt` pins each installed wheel URL and SHA-256. Its JSON receipt binds the resolver report, before/expected inventories and historical constraints. `analysis/p29_environment.py` verifies the complete installed inventory, not only the headline packages. Mac-only dependencies are not required merely to recreate their names.

The first installation exceeded the shared filesystem's quota after about 5.1 GiB. The generated environment was moved to `/tmp/miniproject_1-p29-venv-20261101`; `.venv` is a symlink to it. The same locked transaction completed there. This storage is node-local and disposable: after node loss, recreate the environment from the committed lock and Python installation receipt. Do not move the data, banks or evidence to scratch. The two failed install attempts are retained as logs; the explicit write probe reported errno 122. Dependency inventory verification and `pip check` passed after recovery.

The new route is not the old Mac binary environment, despite equal distribution versions. CPU/GPU and macOS/Linux reproducibility are different questions. Keep historical identities and artifacts immutable. P2.9 recipes will include a runtime-route identity covering device/backend policy, the Linux lock hash, source hashes and dependency versions. The current `runs.fitted_config` contains package versions but not wheel/build identities; matching version strings alone must not silently authorize cross-platform cache reuse.

## Source audit and dispatch boundary

| Current source | Observed behavior | Required P2.9 implementation |
|---|---|---|
| `engine.fit_recipe` | FeaturePipeline is fitted on all D before estimator.fit | Dispatch raw-row adapters before this step; retain the existing path for recipe_v1 |
| `recipes.validate_recipe` | Strict recipe_v1 keys; TabM rejects fitted blocks | Separate versioned adapter schema with strict fields; no removal of the legacy rejection |
| `features.FeaturePipeline` | Fold-fitted encoders support D transforms; encoder split injection is not exposed | P2.9 wrapper must accept validated diagnostic encoder assignments and rebuild within every S or calibration-training subset |
| `encoders.DateTargetEncoder` | Grouped encoder folds, row-weighted target smoothing | Retain these definitions; do not confuse them with DateCount's distinct-group counts |
| `tabm.stage_seeds`, `TabMPreprocessor`, `stop_rule` | Context-derived seeds, declared column layout, earliest epoch within loss tolerance | Reuse exactly for CPU B-equivalence controls; change mechanisms only in new declared recipes |
| `tabm.fit_state` | CPU stop/refit owns unsupervised preprocessing; no raw feature pipeline | New raw worker owns features, preprocessing, stopping and final refit; parent never imports torch |
| `tabm.logits` | Padded 256-row inference batches and float64 probability averaging | Preserve prediction independence from surrounding query batch size |
| `models.make_model`, `runs.fitted_config` | Constructible estimator parameters enter recipe/run identity | New adapters must expose complete declared parameters without fitting; include device and all implicit subfits |
| `banks.source_manifest` | Hashes imported project sources and running script | Import all executing adapter sources before freezing a run; include worker sources and external implementation hashes explicitly |
| `procedures.validate_procedure_metadata` | Validates kind and finite coefficient/weight arrays | Also verify regularization against the declared setting before new learned banks are admitted |
| `nested` | Nested whole-procedure refits and exact cache identities | New adapters must work unchanged at smaller populations and in full refits; record deeper fit counts |

## Raw-row neural adapter

The public fitting interface receives only raw D rows, labels, primary groups, immutable recipe and FitContext. No outer labels, global encodings, global bins or global stopping state enter. On S, build a fresh feature pipeline, group-held-out training target encodings, category vocabulary, imputation/scaling and numerical embedding bins. Transform V from S state. Choose the stopping epoch using only V's permitted metric. Discard all stopping objects and rebuild everything on D before a fresh final model/optimizer initialization. Query rows use only D state.

The parent process returns an object with the common CV protocol: predict_each, predict_high, model_inputs, feature columns/lineage and complete fitting receipts. Share the deterministic S and D feature/layout/bin states within a fold across estimator seeds; repeat the independent stopping and final network training for each seed, with chosen epochs recorded per seed. Features and bins do not depend on estimator initialization seeds. The GPU worker imports the existing safe feature/encoder modules and the neural modules, but never songpop.artifacts, phase2.models or another transitive LightGBM/XGBoost import. Child startup must include the numerical-policy environment before importing torch. The worker hello hashes all its project modules and the parent checks them against the frozen source manifest, including worker-only imports. Both ends validate diagnostic split assignments and record them in receipts.

Diagnostic APIs inject the S/V assignment and S's encoder assignment together. Validate dimensions, unique rows, full partition coverage and group disjointness. Production recipes cannot choose assignments after scores. The B controls retain exactly the old column layout, seed streams, AdamW settings, initialization and epoch-to-update policy on this node. The three timing-recipe old/new CPU comparisons cover all 15 folds after the old adapter repeats in fresh node processes. Historical-bank replay becomes a separately reported portability measurement only under the explicit version 1.3 R2 amendment to the main plan. Its failure excludes the old run as a scientific control and forbids claiming node reproduction of its result. This changes the cross-platform acceptance gate; the same-node 1e-12 gate is not loosened.

Numerical embeddings operate only on declared continuous columns. Category encodings and missing flags have explicit separate paths and an unseen-category test. Native FT categorical tokens reserve an unknown index fitted from S or D as appropriate. PLE bins use training features only. TabM embedded initialization, PLE version, periodic frequencies/scale/lite/activation, architecture dimensions, normalization, both ResNet dropout sites and optimizer decay groups are all explicit recipe fields. ResNet merges a singleton last batch into the preceding batch after the seeded permutation, retaining every row; test this at S, D and nested sizes. Legacy TabM keeps its short last batch. The FT helper's parameter sets are ordered by parameter name before optimization and receipt serialization; save the ordered group-name hashes. The AUC-stopping variant maximizes CPU float64 AUC of the deployed averaged probabilities with tolerance 1e-6 AUC units, using the earliest epoch within tolerance of the maximum; its hand-checked stop-rule tests precede tuning. New recipes explicitly name the stopping-seed purpose, cap and loss direction. B-equivalence controls use the legacy purpose unchanged.

## Calibration and retrieval

SVM probability fitting is disabled. Five grouped calibration splits of D each rebuild all raw features and preprocessing. A StandardScaler fitted to the resulting out-of-fold margins feeds the fixed sigmoid LogisticRegression(C=1, solver=lbfgs, tol=1e-8, max_iter=1000), with unpenalized intercept and no class weights. A fresh base recipe is fitted on D for queries. Save class orientation, slope, convergence, all assignments and scaler state. The whole procedure has six base fits per calibration, before any deeper base-adapter subfits. Kernel ridge and linear margins follow the same contract.

Retrieval remains a separate raw-row adapter. A training query excludes its full primary group from the candidate memory and every memory encoding capable of leaking its label. Group-held-out query blocks build reference memory and encoders on complementary groups. Stopping queries use only S, outer queries only D. Freeze the block count, seeds, batching, inference-memory rule and true fit cost before a D-arm recipe. No generic precomputed matrix shortcut is admitted.

## Proposed CUDA reproducibility policy

Before scientific scores, freeze deterministic algorithms with errors on unsupported operations; IEEE float32 training; no mixed precision, compilation or TF32; explicit independent seed streams; fixed padded inference; deterministic cuDNN with benchmarking disabled; cuBLAS workspace configuration set before startup. For installed torch 2.14 use its fp32_precision controls rather than mixing them with deprecated allow_tf32 settings. Keep the historical double-precision probability/loss arithmetic where applicable. Recipe identity distinguishes CPU and CUDA routes.

Target byte-identical states, receipts and predictions for identical GPU requests in fresh processes on this node. The synthetic probe demonstrated that property for scalar/PLE/periodic TabM, MLP, ResNet and FT-Transformer. It did not test full stopping adapters, all shapes or historical parity. Extend the check to full and nested populations before accepting the route. If it fails, diagnose the operation and propose a concrete criterion for R2 before scoring. Never relax the historical CPU 1e-12 gate in response to a failed comparison.

## Ordered manifests and cost gate

Generate complete canonical recipes from the reviewed domains at seed 20261031. Mandatory B/D and representation pairs come first, followed by remaining representation branches and then the space-filling remainder. Validate canonical uniqueness, family counts, match-only-the-declared-factor pairs, embedding/depth/batch-size floors and epoch caps. The six adapter-equivalence runs do not count toward 274 scientific configurations. The 48 core refinements, ten stopping variants and 24 feature comparisons retain the reviewed accounting. No manifests are called executable before every field maps to an adapter and rejection tests cover unsupported fields.

Cost full-size and nested-size complete raw fits, including stopping/refits, calibration, encoder rebuilding, memory construction, startup and serialization. Count base fits and deeper subfits separately to avoid multiplying the same work twice. Nominee confirmation, fresh final-policy seed averages, reference replay, proposed-menu construction, revised nested build/replay, at least 25 percent overhead and review/submission time are reserved before broad search admission. Synthetic probes and first-fold portability checks do not substitute for these pilots.

## Required verification and review

Known-wrong variants must fail fixed-assignment held-out-label tests, preprocessing poisoning, wrong-group calibration, leaky encoders, direct/indirect retrieval leakage, malformed receipts and class-orientation tests. Positive controls confirm that permitted training labels affect the expected features. Outer-label flips must leave training and prediction invariant; stopping-label flips may change losses, the selected epoch and final refit, but not S-fitted state or the common-horizon stopping trajectory.

The environment/source audit and actual results go to a fresh Fable 5.1 xhigh review, separate from consultation. Concrete raw-adapter contracts and any portability amendment receive R2 before dependent tuning. R1 for final artifacts and the measured large-job gate remain unchanged. Full P2.9a is incomplete until ordered executable manifests, safe adapters, representative pilots and a measured launch proposal exist.

## Implementation sources

The installed versions and their hashed sources govern the implementation. General references checked 2026-11-01: [PyTorch 2.14 reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html), [CUDA numerical precision](https://docs.pytorch.org/docs/2.14/notes/cuda.html), [TabM](https://github.com/yandex-research/tabm), [authors' MLP, ResNet and FT-Transformer package](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md), and [its 0.0.2 distribution](https://pypi.org/project/rtdl-revisiting-models/0.0.2/). No external dataset, weights or Spotify-specific model code were downloaded.
