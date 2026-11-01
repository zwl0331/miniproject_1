# Fast neural adapter

2026-11-01. Provisional development implementation. Sam authorized this session
to finish the neural adapter under fast-mode rules and then take ownership of
development. The 64-configuration extended architecture grid is complete;
remaining seed checks and ensemble status are in STATE.md.

## Use

```python
from songpop.phase2.encoders import FitContext
from songpop.phase2.raw_recipes import make_recipe
from songpop.phase2.raw_neural import fit_raw, Worker

recipe = make_recipe(
    "tabm", representation="ple", device="cuda",
    blocks=["date_count", "date_te3", "cyclic"],
    features={"year_relative": True, "year_standardize": True, "date_counts": True},
    layout="mixed_neural_v1", scaler="quantile", stopping_metric="auc",
    estimator_seeds=[0],
    params={"d_block": 128, "max_epochs": 200, "patience": 20},
)
fitted = fit_raw(recipe, raw_training_rows, y_train, track_groups, FitContext(155, 0))
p_high = fitted.predict_high(raw_validation_rows)
p_test = fitted.predict_high(raw_test_rows)
Worker.close_all()
```

Raw rows use the exact columns returned by
`analysis.fast_exploration.data_inputs()`. Do not prefit feature transforms.
`predict_proba` returns columns `[Low, High]`; `classes_` is `[0, 1]`.
Models and their fitted transforms can be pickled within this environment.

| Family | Representations | Notes |
|---|---|---|
| `tabm` | `scalar`, `ple`, `periodic` | 16 members by default; mean of member probabilities |
| `mlp` | `scalar`, `ple`, `periodic` | Single network; optional flattened numerical embeddings |
| `resnet` | `scalar`, `ple`, `periodic` | Residual network; hidden width is twice block width |
| `ft_transformer` | `scalar` | Native trainable feature tokenizer; four attention heads by default |
| `retrieval` | `scalar`, `ple`, `periodic` | TabR-inspired adaptation; learned keys and label values; excludes query track group |

`ple` means piecewise-linear embeddings. The default embedding dimension is 8
and bin count is 16. Periodic embeddings use 16 frequencies with initial scale
0.01. Override these with the `embedding` argument. Unknown fields are rejected.
The default feature representation is the existing numeric layout, including one-hot
categories and missing flags. Embeddings act on all of those coordinates.
Constant coordinates use a unit-width interval centered on their training value.
FT-Transformer uses all coordinates as numeric feature tokens.

With `layout="mixed_neural_v1"`, continuous columns receive the numerical
embeddings, missing flags stay binary and categories have explicit unknown codes.
TabM uses native categorical inputs; MLP/ResNet/retrieval use learned category
embeddings; FT-Transformer uses native category tokens. Extra year/count/round
features fit inside the same stopping and final-training boundaries. Optional
`stopping_metric="auc"` minimizes one minus the inner AUC. `pairwise_weight`
mixes BCE with pairwise logistic loss, from 0 (BCE) to 1 (pure pairwise).

Retrieval requires blocks without target encoding, for example
`blocks=["date_count", "cyclic"]`. Its training memory contains only the current
training partition. All candidates from the query's own track group are masked.
Query/test rows never enter memory fitting. This is a compact project adaptation
of the general TabR mechanism, not an exact reproduction of the published model.

## Running complete trials

```bash
.venv/bin/python -m analysis.neural_exploration --write-queue artifacts/neural/queue.json
.venv/bin/python -m analysis.neural_exploration --queue artifacts/neural/queue.json
```

The generated starter queue contains six configurations and does not launch
training. Edit it to the intended bounded comparison before executing it.
Each entry contains `id`, `recipe`, and `cutoff_utc`. The default cutoff is the
original experiment deadline, 2026-11-01 20:00 UTC. The runner checks the
cutoff before each outer fit; an already-started fit can finish. It never starts
a trial after the project experiment deadline, 2026-11-01 20:00 UTC.

One process uses one isolated GPU worker and four CPU threads. The extended
runner `analysis.extended_gpu --queue <path> --workers 4` coordinates four
independent workers. Do not edit worker-bound sources while they run. No packages
were installed. The pinned environment already contains Torch 2.14.1,
TabM 0.0.3, rtdl-num-embeddings 0.0.12, and rtdl-revisiting-models 0.0.2.

Results go to `artifacts/neural/<id>/`. A complete result includes its config,
OOF and test bank, all 15 fold scores, mean per split seed, seed/stopping details,
warnings, runtime and simple blend comparisons against existing references.
The arrays use the same `PredictionBank` schema as fast exploration. Extended
results also compare against follow-up trees (CV 0.843814).
Partial cutoff runs are not scored. The runner does not update submissions or
the shared fast-exploration notes table.

## Training boundary

The adapter receives only an outer training fold. It creates a track-grouped
inner stopping split. Date counts, cross-fitted target encoding, imputation,
scaling, category vocabularies and PLE bins fit only on the inner training rows
for stopping. Inner validation contributes only the stopping loss. After epoch
selection, the adapter discards the stopping networks, refits transformations
on the full outer training fold and trains fresh networks for the selected
epoch counts. Query rows never enter a fit. Estimator seeds use separate
initialization and minibatch streams and average probabilities.

ResNet retains a singleton last minibatch using training-derived BatchNorm
running moments for that batch. FT-Transformer uses the library's prescribed
zero-decay groups, sorted by parameter name for fixed-seed execution. All
networks start from random initialization. No pretrained weights or external
dataset inputs are used. The old scalar TabM route remains available through
`from_legacy`.

## Verification

The focused legacy checks and CPU/GPU architecture checks are recorded under
`doc/llm_usage/2026-11-01-neural-adapter/`. They cover training and probability
orientation, stopping isolation with changed held-out labels and features,
positive controls that alter allowed training data, rejection of overlapping
groups, query immutability, serialization, multiple seeds, scalar compatibility,
constant columns and ResNet singleton batches. The deliberately leaky control
in the preserved scalar test is detected.

Final checks: four advanced tests passed in 83.932 seconds, including all ten
architecture/representation combinations on CPU and CUDA. The scalar
compatibility recheck passed in 20.883 seconds. The full saved-fold runner smoke,
`adapter_smoke_mlp_ple_2epochs`, completed all 15 folds in 99.2 seconds and passed
bank validation. Its config, OOF/test arrays and fit information are committed
under `artifacts/neural/adapter_smoke_mlp_ple_2epochs/`. Its two-epoch cap is for
integration verification, not a nominated scientific configuration.

These are correctness checks. They do not establish that any architecture
improves AUC. Candidate selection still requires complete saved grouped folds
with split seeds 155, 2026 and 1029. Pretrained models are prohibited. The extended
checks in `doc/llm_usage/2026-11-01-extended-exploration/` cover stopping isolation,
unknown categories, continuous-only PLE bins and retrieval group exclusion, with
allowed-label positive controls. Three focused checks and six GPU smoke fits pass.

The completed extended grid contains 17 TabM, 17 MLP, 11 ResNet,
11 FT-Transformer and eight retrieval configurations, each evaluated on all
15 saved folds. None of these 960 outer fit records hit the maximum epoch cap.
Three launcher identifiers initially failed validation before fitting; their
recipes were retained and all three renamed jobs completed. The corrected roster
and original error records are preserved under `artifacts/extended/` and
`artifacts/neural/`. Model selection and seed confirmations are summarized in
`notes/2026-11-01-extended-closeout.md`.

Architecture APIs were checked against the installed packages and their primary
documentation: [TabM](https://github.com/yandex-research/tabm),
[numerical embeddings](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/README.md),
and [MLP, ResNet and FT-Transformer](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md).
