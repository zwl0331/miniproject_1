# Independent implementation design before fresh Fable consultation

2026-11-01. Codex. Provisional implementation design under the approved v1.3
plan and node contracts. No new scientific scoring or large search is authorized.

Prior Claude positions are saved verbatim in the preceding task's
`../2026-11-01-p29a-node/claude_positions.md` and `claude_draft_critique.md`.
Their accepted requirements are the raw dispatch boundary, independent runtime
identity, strict same-node equivalence, shared seed-independent feature fits,
whole-procedure calibration, group-excluded retrieval, and complete fit counts.
These positions were read before this design. Consultation is not approval.

First complete unit: versioned raw dispatch and the scalar TabM stop/refit path,
including CPU/CUDA routes, runtime/source binding, structural receipt validation
and diagnostic split injection. Then add the nonlegacy representations and
architectures, calibration/statistical paths and retrieval under the same
interfaces. Keep every partial family explicitly unavailable for scientific
coverage until its actual adapter and gates exist. Do not create placeholders
that can silently score an unimplemented recipe.

Design:

1. Keep canonical recipe_v1 hashes and legacy fitting unchanged. A strict
   recipe_v2 validator in a worker-safe module owns raw-adapter fields. The
   existing validator delegates only on the version marker. `fit_recipe`
   dispatches v2 before constructing D-wide FeaturePipeline. A raw fitted object
   implements the current predict_each/predict_high/model_inputs and
   features.columns_/lineage_/receipt interfaces, so CV and nested fitting use
   the same whole procedure.
2. Separate semantic recipe fields from concrete runtime provenance. The
   adapter declares CPU or CUDA and its numerical policy. Every newly executed
   run, including v1 atoms, binds policy, wheel/bootstrap lock hashes and actual
   runtime builds in fitted_config and nested cache identity. Historical banks
   remain readable and immutable; new incompatible banks cannot blend or share
   cached fits. Source identity includes executing worker-only modules and
   installed architecture source hashes, checked against worker hello/replies.
3. Neural worker imports neither model factories nor artifacts. It owns the raw
   feature pipeline and all stop/refit transforms. S/V and S-encoder assignments
   are jointly injectable only through diagnostics, validated on parent and
   worker. D encoder assignments can also be injected for final-fit isolation.
   The returned serializable state contains the D feature/preprocessing objects,
   NumPy network state, column layout and per-seed receipts. It contains no Torch
   object. Shared S and D transforms are fitted once per requested seed policy.
4. Preserve scalar B layout, initialization, seed purpose strings, optimizer,
   batch order, epoch rule, inference padding and float64 probability arithmetic.
   CPU equivalence uses three old timing recipes on every saved fold and repeat.
   CUDA is separately deterministic and has independent identity; it need not
   reproduce CPU. Expanded embeddings keep binary flags/categories outside the
   continuous embedding operation. All constructor defaults become explicit
   resolved configuration fields before a run is admitted.
5. Receipt validators check partition identities, seed policy, lengths and finite
   stopping histories, chosen epoch and stopping reason against the declared
   rule, update counts, transform/weight/source hashes and runtime consistency.
   Legacy TabM receipts gain structural validation without requiring fields the
   historical schema never stored. Fix combiner regularization metadata checking
   at the same bounded compatibility revision, with known-wrong rejection tests.
6. Diagnostic tests freeze assignments and a common epoch horizon. V-label flips
   must preserve feature states and preselection training trajectory; permitted
   S labels must affect the expected encodings. Poisoned query features cannot
   alter fitted state. Test the actual worker/engine path and a deliberately
   wrong implementation, not only standalone features. Compare a fresh multi-seed
   fit with the average of separate seed fits under identical contexts.
7. All family manifests remain blocked until each field maps to an executable
   implementation. Representative full/nested pilots include all stopping,
   calibration and process costs. The proposal reserves confirmation, mandatory
   refinements, revised nested build/replay and review time before broad training.

Current verification issue: the unchanged full legacy suite was restarted with
progress instrumentation only. Its historical anchor exact-array assertion fails
on oof/test_by_fold; kNN has an unnamed AssertionError. The eight raw-boundary
checks pass again. Diagnose exact locations/magnitudes, do not silently change
tolerances or model behavior. The approved portability amendment already
distinguishes historical Mac comparisons from same-node gate requirements.

Critique of prior advice: an all-family implementation before any measurable
milestone creates unnecessary integration risk. Use complete incremental units,
without counting partial adapters toward scientific coverage. Two concurrent
GPU workers were suggested in the prior critique but the approved device policy
starts one; concurrency expansion is optional and requires separate measurement.
Do not add an unapproved concurrency requirement to the initial gate.
