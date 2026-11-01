# P2.9a independent implementation design

2026-11-01. Written by primary Codex after saving Claude's earlier positions verbatim as claude_prior_positions.md, before reading the new consultation. Provisional implementation design, not review.

## Environment and identities

Recreate Python 3.12 under .venv. Resolve exact historical package versions on Linux before installing them. The old additive locks contain Mac wheels and cannot be reused verbatim. Save the new resolver reports, wheel hashes and dependency inventory separately from all historical installation artifacts. A CUDA build of the pinned torch release is a new platform route; do not silently replace the scientific package versions to get CUDA. If the exact release is unavailable, present a measured dependency obstacle and the concrete alternative before any dependent scoring. The protected historical environment cannot be recreated by unconstrained pip install.

Keep the legacy TabM worker and recipe behavior intact. A new adapter/schema must declare the raw-row preprocessing/stopping boundary and device. Extend canonical source and dependency identities rather than silently writing new results to old IDs. Historical predictions remain immutable comparison evidence; reuse as a final-source control requires the plan's replay and compatibility checks.

## Fit boundary

The engine currently fits FeaturePipeline on the entire D population before invoking the estimator. That is safe for the old stateless-only TabM recipes but cannot support the new D feature arm. Dispatch new raw-row adapters before this feature step. A fitted raw adapter must expose prediction, input equivalence, feature lineage and a validated receipt so the common CV harness can preserve its current bank checks.

Stopping must build FeaturePipeline and the numerical/categorical preprocessing on S, transform V with S state, and discard all stopping state before fitting a fresh pipeline and network on D. Preserve existing context-derived stop/final initialization, batch seeds, column layout and epoch policy for B-equivalence controls. Fixed split injection should be available solely through a diagnostic API, validated as a complete group-disjoint partition. Target encoders need injected encoder folds as well as the fixed S/V split for the label-flip tests. Match legacy behavior first before adding numerical embeddings or CUDA.

Margin calibration similarly owns raw D, rebuilds feature and preprocessing state in all five grouped calibration fits, fits a standardized one-margin sigmoid on the held-out margins, and then fits a fresh base model on D. Preserve class-1 orientation and convergence failures. Record every deeper fit so measured calibration costs cannot be mistaken for one ordinary fit.

## GPU policy proposal

Start with deterministic float32 training, float64 ensemble-loss/probability arithmetic where the historical implementation uses it, no mixed precision or compilation, no TF32, deterministic CUDA algorithms, explicit cuBLAS workspace configuration before worker startup, independent fixed streams for initialization and batches, and fixed padded inference batches. One GPU fitting process initially. CPU processes get explicit four-thread limits; total runnable fit workers must stay within the 32-core quota and 45 GiB memory ceiling.

For same-device, same-package, same-input runs, target byte-identical learned states and predictions and identical selected epochs. Test in separate processes on synthetic and full-size score-free controls before freezing the route. Any nonzero tolerance needs a pre-scoring method review with evidence; do not infer CPU/GPU equality from same-device replay. Historical CPU and old/new CPU-adapter comparisons retain 1e-12. GPU predictions are a new recipe even if CPU parity happens to hold.

## Manifest and cost contracts

Generate immutable ordered batches from seed 20261031. Canonical computational identities exclude names and include device, architecture, preprocessing, stopping/calibration policy, seeds and feature-arm details. Validate exact family counts, block ordering, matched pairs and representation floors separately from gate-run counts. Generate mandatory pairs before filling space; every covered factor must differ alone. Neural stopping caps and final seed policies belong in the manifest and cost graph.

Cost observations include actual raw-row feature work, worker startup, stop/refit, calibration, encoder and retrieval subfits, prediction and serialization. Full-size and nested-size pilot populations are chosen independently of pilot scores. Timing output must not expose held-out AUC. Operation counts should describe one complete adapter fit with nested subcounts, then explicitly multiply only non-shared fits for seed confirmation, the frozen menu, nesting and empty-cache replay. Reserve at least 25 percent overhead and review/submission time.

## Immediate checks and critique

The new schema/interface could be too broad if designed for all 12 families before one complete recipe works. Implement and validate one raw TabM B control and one calibrated RBF SVM path first, then generalize. Retrieval D remains a separate adapter because query-group exclusion must reach memory encoders as well as neighbor rows. Do not add retrieval to an otherwise safe generic matrix adapter.

Bank receipts currently include TabM details but structural validation and combiner regularization checks are deferred; close those before new related banks are admitted. Known-wrong variants must fail held-out-label, preprocessing poisoning, calibration-group and indirect retrieval checks. The CPU source must remain reproductible under the new adapter and final-source manifests. Cross-platform floating differences cannot be fixed by editing saved evidence or loosening the old tolerance.

The broad plan does not guarantee that the actual node or available packages can finish every target. Avoid implementation based on a guessed GPU speedup. Complete environment/source audit and concrete contracts, consult Fable, then review consequential changes before dependent tuning. Large-job approval comes after actual timing and a concrete admitted schedule.
