# Bounded augmentation screen

2026-11-01 08:24 UTC. Complete before the 08:32:25 UTC cap. Sam authorized four mild
settings after the prior closeout. All 60 outer-fold runs completed; no extra
seed runs, new submission files or changed final selections.

The reference is the highest grouped-CV ensemble, x3_fixed_hybrid_retrieval5,
CV **0.844649232**. The 15 folds and split seeds 155/2026/1029 are unchanged.
These are exploratory results on reused folds, not independent significance tests.

| Setting | Standalone CV | Best checked ensemble CV | Delta | Positive split means |
|---|---:|---:|---:|---:|
| a1_tabm_mixup_a01 | 0.829259 | 0.844537 | -0.000112 | 1/3 |
| a1_tabm_noise002 | 0.829120 | 0.844608 | -0.000042 | 1/3 |
| a1_retrieval_noise002 | 0.789544 | 0.844483 | -0.000166 | 1/3 |
| a1_retrieval_noise005 | 0.791560 | 0.844595 | -0.000054 | 1/3 |

Individual predictions were tested as 5% and 10% probability additions and
10% logit additions. Retrieval variants also replaced the existing 5% retrieval
component. The matching original model at estimator seed0 was scored using
the same formulas, separating augmentation from a mere weight change.

Sam then asked, verbatim:

> Is it worth while to consider adding them to the ensemble or no?

One additional saved-bank check averaged all four augmentation models equally
and added that pool at 5%. CV was **0.844592452**, delta **-0.000056779**;
all three split means worsened. This required no retraining.

**Recommendation: leave these four models out.** No checked individual or
pooled addition improved the strongest CV ensemble. This is a conclusion about
these four settings and checked mixtures, not a claim that all possible
augmentation methods fail. The public regressions reported during this block
did not change the predeclared grid, reference metric or training scope.

Mixup matches categorical and missingness profiles and interpolates continuous
coordinates and labels only inside the current training population. Noise
affects observed audio coordinates after training-only preprocessing and is
clipped to training ranges. Validation/test rows are untouched. Retrieval
uses noise only and retains its entire-track exclusion from memory.

Three focused augmentation tests and three GPU route smokes passed. Four
imported legacy tests also ran once successfully; the test selector was then
narrowed to avoid unnecessary future reruns. Source code was frozen throughout
the scientific fits. Epoch-cap hits: 0 of 60 outer fit records.

Evidence: `artifacts/augmentation/queue.json`, `summary.csv`, `summary.json`,
`pooled_spec.json`, `pooled_comparison.json`, `decision.json`, `gpu.log` and
`queue_outcomes.json`; full per-model configs/OOF/test predictions and fit
records are in `artifacts/neural/a1_*`. The implementation checkpoint is
3997efa. LLM use and authorization: `doc/llm_usage/2026-11-01-augmentation/README.md`.
