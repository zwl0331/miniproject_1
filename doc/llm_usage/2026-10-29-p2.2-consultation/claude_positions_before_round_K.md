# Claude's positions on P2.2 (written before the round K prompt was sent)

Written 2026-10-29, about 21:00 PDT, at commit 281a014 (P2.0b committed). Inputs: `doc/plans/phase2-models-features.md` v0.3, sections 4.10, 5 and 6.2; the P2.1 code in `songpop/phase2/` and `scripts/`; Finding 010.

## Scope

P2.2, as the task table and section 4.10 define it: common search extensions (conditional sampling, admissible regions, local refinement batches, range-extension manifests, timing and allocation records, log shards and their merge); factory parameters of the installed families; the kNN family and its block-weighted representation; the broad searches and first refinements of LightGBM, random forest, XGBoost, Extra Trees, HistGB, logistic (`spline` and `log_standard`) and kNN; findings. Also the deferred housekeeping: Finding 010 to `reviewed`, the "in review" status lines of P2.0b.

## Two review stages in one record

1. Stage A: the code, the self-checks, the timing records (times only, no scores), the written P2.2 execution protocol, and the saved wave 1 manifests with their ordered configurations. No search score exists. Reviewed to `APPROVED`.
2. Then the waves run on that frozen source. Stage B: every bank, contribution and allocation record, the findings, the doc updates. Reviewed to `APPROVED`.

Why: a code defect found after 300 runs costs a full rerun; P2.3 needs the common search extensions approved before its searches; run ids carry the source manifest, so the code must not change during the waves. Both stages are rounds of one record; convergence is stated for each fingerprint.

## Code

| Piece | Position |
|---|---|
| Factory parameters | Extra Trees `max_depth`; XGBoost `reg_alpha`, `gamma`; HistGB `l2_regularization`. LightGBM and random forest already have every parameter of section 5.2 |
| kNN family | `knn` in `FAMILIES`; `KNeighborsClassifier` with `n_neighbors`, `weights`, `p` (2 Euclidean, 1 Manhattan); base F1 only; estimator seeds exactly `[0]`; training weights `none`; preprocessing must be `knn_blocks` |
| `knn_blocks` preprocessing | Fitted on the fitting rows: phase 1 log transforms; the ten continuous audio columns median-imputed, standardized and divided by sqrt(10) (by 10 when `p` is 1, the block normalization of Manhattan distance); year standardized times `alpha`; key, mode, time signature one-hot (unknown ignored) times `category_weight` divided by sqrt(3) (by 3 when `p` is 1), not standardized. Parameters `alpha`, `category_weight`; robust scaling and a cyclic block not built in P2.2 |
| Search schema v2 | Keeps v1 working unchanged. Adds: `derived` (declarative rules, for example `subsample_freq` 1 when `subsample` is below 1, absent otherwise); `admissible` (a declarative product constraint, `learning_rate` times trees within 5 to 100, anchors exempt); every rejected draw counted and saved |
| Refinement spec | Parents by run id; per-parameter types: `log` (positive, factor log-uniform in 0.5 to 2, integer rounding, hard bounds, zero handled as its own grid point), `ordinal` (move to a neighboring value), `categorical`; one to three dimensions changed; a quarter of the batch broad draws from the initial space; admissibility and canonical deduplication against every earlier configuration of the family; the proposal list saved before the first fit |
| Range extension | Only when the family incumbent's value of an ordered numeric parameter equals an end of its list: one new value one geometric step beyond, clipped to hard bounds, stated in the manifest before fitting; at most one per parameter in P2.2 |
| Batch manifest | `artifacts/phase2/batches/<batch id>.json`, written once before the first fit: family, wave, kind (initial, expansion, refinement), the pre-batch state (family anchor, family incumbent, strongest standalone recipe, reference H, each with run id and seed policy), the specs or proposals, the timing reservation |
| Contribution records | For every configuration, the 0.9 H + 0.1 c and 0.75 H + 0.25 c blends computed in memory and logged as records (both bank hashes, weights, unrounded mean fold AUC, repeat means); no bank saved; recomputed by verification |
| Allocation records | A pure function of logged records: the four routes of section 5.6 per batch, the allowance counter, the decision; logged; recomputed by verification |
| Timing records | Light, representative and expensive configuration per family, 15 fits each, feature construction and prediction included; elapsed seconds only, the score never computed into any output |
| Log shards | Not built: serial execution fits (below), so one process writes one log. Recorded as not needed, since section 5.1 allows at most two processes, not at least two |
| Verification | `--verify-all` refits every run, which becomes hours at 300 to 700 runs. Add `--audit-all`: every bank's hash against the log, bank validation, every record field recomputed from the bank arrays, every contribution, search expansion, manifest and allocation recomputed; refitting stays available per run id for a sample the reviewer chooses |

## Execution protocol (to be written into the plan as a P2.2 section before wave 1)

- H for all of P2.2: the LightGBM anchor, fixed. The strongest standalone recipe at the start: the anchor (0.8315).
- Family anchors, in each initial batch: LightGBM, the phase 1 anchor exactly; random forest, the phase 1 recipe (500 trees, leaf 3, F2); logistic, `log_standard` F2 with C 1 (phase 1 had no one-hot and `max_iter` 2000, so it is not reproducible exactly); XGBoost, Extra Trees and HistGB, the representative recipes of Finding 010, declared before their search scores; kNN, k 60, distance weights, alpha 1, a grid point declared now.
- Waves, all families together, each wave's manifests saved from the state after the previous wave, so the run order inside a wave changes nothing:
  1. Wave 1: the initial batches of section 5.1 (276 configurations). Logistic: `log_standard` F1 and F2 full grids (12) and 12 seeded draws of the 30 `spline` F1 configurations. kNN: the full 48 grid.
  2. Wave 2: a broad expansion batch for every family with allowance, with range extension where the incumbent sits at an edge.
  3. Waves 3 and 4: local refinement batches of 12 for at most four families with allowance, ranked by best standalone unrounded mean fold AUC (ties to the lower recipe id); two parents (best standalone, best contributor or the next distinct one); 3 broad draws and 9 local proposals per batch. For the boosting families the {0.67T, T, 1.5T} tree triplet of the incumbent runs once in wave 3, inside the ceiling. Other families with allowance get broad expansion.
  4. Later waves: broad expansion while allowance remains, until the ceiling or the cutoff.
- Cutoff: no P2.2 batch starts after Friday 2026-10-30, 10:00 PDT.
- Every configuration: all three split seeds, estimator seed 0, no 120 s subspace expected.
- Serial execution, four model threads. Projected wave 1 about 45 minutes, all waves about 2 hours.

## Storage

A bank is about 440 KB uncompressed. 300 to 700 banks make 130 to 300 MB, committed with the task in a repository with no remote. The bank format stays as reviewed.

## Findings

Finding 011: the broad search per family (counts, best standalone and best contribution to H with run ids, repeat differences against the anchor, allocation outcomes), labeled exploratory. A paired bootstrap only for comparisons that drive a decision. Possibly an optional pipeline submission under section 10 if a complete single recipe beats the anchor.

## Doubts

1. Whether two review stages in one record fit the policy as written, or whether the task should be re-cut as P2.2a and P2.2b in the plan.
2. Whether the refinement parent split and the wave order are what section 5.2 intends.
3. Whether family anchors from Finding 010, whose probe scores were seen, are acceptable as comparators.
4. kNN on F1 may find distance 0 between rows of different primary groups that share audio and year (the strict grouping joins identical audio-only signatures); a small optimism that trees share.
5. Repository growth from committed banks.
