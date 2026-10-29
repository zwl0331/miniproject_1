# Claude's positions on P2.1 (written before reading Codex round G)

Written 2026-10-29 before the round G prompt was sent. Inputs: the committed project at df11816, in particular `doc/plans/phase2-models-features.md` (approved in P2.0) section 4, and the phase 1 code in `songpop/` and `scripts/`.

## Scope

1. P2.1 builds what P2.2 to P2.4 run: recipes, feature blocks and encoders, model factories for every family of section 5.1 (CatBoost and EBM imported lazily, runtime-checked in P2.3), the engine, bank_v2, ensembles, comparisons, the run, search and verify commands, the submission script for fold-average deployment of a single recipe or a declared blend, the self-checks, the installation locks, and the anchor reproduced through the new engine as the first phase 2 bank.
2. Staged to the tasks that first need them, with the plan's status recording the staging: `selection.py` (P2.6), `nested.py` and full-refit deployment artifacts (P2.7), the TabM wrapper and its early stopping (P2.5). Their self-checks of section 4.8 arrive with them. Building them now, before the menu exists, would be speculative code under review twice.

## Design

| Piece | Position |
|---|---|
| Feature pipeline | A recipe names a base set (F0, F1, F2 from `songpop.features.make_features`, stateless) and a list of blocks. Stateless blocks (interactions, cyclic) are functions of raw rows. Stateful blocks (date count, date target encoding, era-relative audio) are objects with `fit_transform(rows, y, groups, ctx)` for the fitting rows, which may cross-fit, and `transform(rows)` for any other rows. Output: a float64 frame with a fixed column order and lineage |
| Model preprocessing | Inside the estimator as a scikit-learn pipeline (imputation, log transforms, splines, one-hot, scaling), fitted on the training part only, as phase 1's logistic regression |
| Engine | `fit_recipe` returns a `FittedRecipe` (fitted feature blocks and estimator); `run_recipe_cv` mirrors `songpop.evaluate.run_cv`, with raw rows sliced per fold, and reuses `validate_cv_inputs` and `validate_bank` |
| Anchor | The recipe `lightgbm_F2_anchor`: base F2, no blocks, the phase 1 LightGBM parameters, seed 0. Its phase 2 bank must equal `production_lightgbm_F2_9c37e77d549e` array for array, exactly; LightGBM is deterministic with these settings |
| Bank identity | `recipe_id` = family and name plus the first 12 hex of the canonical recipe hash; `run_id` = recipe id, stage and the first 12 hex of the hash of the identity fields of plan section 4.2 |
| Source manifest | The project files among the loaded modules (`sys.modules`) plus the executing script, hashed after imports |
| Searches | A search JSON (family, base recipe, space, draws, sampler seed); the ordered, deduplicated configurations are written to `artifacts/phase2/searches/<search id>.json` before any fit; each configuration becomes a recipe stored in its bank's metadata |
| Verification | `--verify --run-id`: the recipe from the bank's metadata, the source manifest compared with the current files first (a mismatch exits nonzero), then recomputation and comparison at `atol=1e-12` |
| Comparisons | `songpop.metrics.paired_cluster_bootstrap`, unchanged, on phase 2 banks; records in the phase 2 log |
| Submission checks | Member-level duplicate checks need each fold's transformed test features, not the fitted models: refit the feature blocks per fold (cheap, deterministic) and compare `test_by_fold[r, k]` within groups of identical transformed vectors |
| Installation locks | A script reads the inventory and the three dry-run reports, writes `configs/phase2/install/<route>.txt` with `name==version --hash=sha256:...` lines for each route's additive closure, checks shared pins and unchanged installed versions, and has a `--check-before <route>` mode for step 4 of plan section 4.9 |

## Doubts

- Whether `--no-deps --require-hashes` lock files with only the additive closure are enough when a new package needs a newer version of an installed one (then the route is dropped under the plan).
- Whether the dry run downloads full wheels for torch (large) or only metadata.
- The right `TargetEncoder` smoothing parameter: scikit-learn's `smooth` is the m of the plan's formula, but the class shrinks toward the global mean of the fitting fold, which is what the plan says.
