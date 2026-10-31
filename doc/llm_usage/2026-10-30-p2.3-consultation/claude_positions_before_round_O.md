# Claude's positions on P2.3, written before consultation round O

2026-10-30, about 05:40 PDT. Author: Claude Code (Claude Opus 5.5). Written before the round O prompt was sent, so Codex's independent design can be compared with it. Sam, verbatim: "Okay, 2P2.3 and 2.4, and then pause after you finish." P2.8 is committed (5e2d5a0, cab0174).

## Scope

P2.3 of `doc/plans/phase2-models-features.md` (task table; sections 4.9, 4.10, 5.1 to 5.3, 5.6, 16.3): execute the reviewed CatBoost and EBM route locks; add the factory parameters; turn the two STAGED runtime checks into real checks; time each family; search both families on F2 under a wave protocol like P2.2's; verify; findings. No feature blocks (P2.4), no submission.

## Installation

Execute the two P2.1 locks unchanged (`scripts.install_locks check-before`, the pip command, `check-after`, `pip check`), CatBoost first. The commands were reviewed in P2.1, so I treat the installation as approved there and record it; the runtime checks below follow. If a route fails its checks, the family is omitted with a reviewed incompatibility finding (exit criterion 1).

## Factory changes (`songpop/phase2/models.py`)

- CatBoost: allow `bootstrap_type` (Bayesian, Bernoulli, MVS, No), `bagging_temperature` (only with Bayesian), `rsm` (a float in (0, 1]). Fixed: `thread_count=4`, `random_seed`, `allow_writing_files=False`, `use_best_model=False`, CPU, `SymmetricTree`, `verbose=False`; no eval set, no class weights, numeric F2 only (no `cat_features`).
- EBM: allow `learning_rate`; `interactions` becomes an enumerated recipe value, "none" (0) or "declared3", resolved to the column indices of (year, loudness), (year, acousticness), (energy, loudness) after the recipe's feature construction, so the factory needs the frame's columns. Fixed: `validation_size=0`, `outer_bags=1`, `inner_bags=0`, `n_jobs=4`, `random_state`. interpret's default `interactions=0.9` (automatic pairs) is never used.
- Runtime checks: fit on an F2 sample with its missing values; two fits with the same seed give identical predictions; a pickle round trip predicts identically; no file is written in the working directory; the fixed settings are in effect (CatBoost `get_all_params`, EBM boosting runs exactly `max_rounds`, its `best_iteration_`); P(class 1) in column 1.

## Protocol

Reuse P2.2's wave machinery (`songpop/phase2/protocol.py`, `scripts/run_experiments.py`) with a new protocol file `configs/phase2/p2.3/protocol.json`, task P2.3, seed base 2026103000. Code changes kept small:

1. The protocol path per task (P2.2 keeps its file).
2. H: P2.2 took H from its LightGBM family. For P2.3 I would declare a reference "family" `lightgbm` with only its anchor (initial 1, expansion 1, ceiling 1, no space): wave 0 fits H under P2.3's sources and checks it against the P2.1 anchor bank in all ten arrays; its wave 1 batch has no new configuration, so its allowance is 0 and it never plans again. No code change for H.
3. G, the strongest standalone recipe across families: P2.2's own G was H plus its closed waves. For P2.3 I would add an optional protocol entry naming P2.2's strongest logged run (`p22_lightgbm_b3_001`, base features, 0.8339) with its recipe, so G = the best of H, that run and P2.3's closed waves. P2.8's B14 (0.8354, with date encodings) is a feature-stage result and enters at P2.4.
4. Family anchors: CatBoost, the library defaults made explicit: depth 6, 1,000 iterations, learning rate 0.03, L2 3, random strength 1, Plain. EBM: main effects, 700 rounds, learning rate 0.03.
5. Initial batches (section 5.1): CatBoost 48 = anchor, candidate 31 as an explicit branch (section 16.3), 46 seeded draws over the section 5.2 space with learning rate times iterations in 5 to 100. EBM 12 = anchor and 11 draws over its 18-point space.
6. Refinement types: CatBoost depth ordinal, learning rate, iterations, L2, random strength (log, zero step 0.1), boosting type categorical, rsm ordinal, bagging temperature log; `bootstrap_type` never moved. EBM rounds and learning rate log, interactions categorical. Range extension as in P2.2.
7. Allocation exactly section 5.6, H fixed for the whole task, contributions and bootstraps as in section 15.5.
8. Cutoff: chosen from the timing so that P2.4 can still start its batches before Saturday 08:00 PDT; my first guess is Friday about 20:00 PDT.

## Timing and compute

Score-free timing of light, representative and expensive cases per family (CatBoost expensive: depth 7, 2,000 iterations, Ordered; EBM expensive: the declared pairs, 1,500 rounds) before the batches are sized; batch sizes confirmed or reduced, never raised. If CatBoost's representative 15-fit run exceeds 120 s, section 5.1 allows screening on split seed 155; I would rather keep all three seeds and reduce the batch, since screening needs new machinery. Serial, one process with four threads (the M5 has 4 performance cores of 10).

## Verification and records

One review record, two checkpoints like P2.2 (readiness, completion). `--audit-task P2.3` and a full refit `--verify-task P2.3`; CatBoost and EBM must replay exactly. Findings: 018 (installation and runtime compatibility, determinism), 019 (search results, contributions, allocation), exploratory. Findings 014 to 017 become `reviewed` in this ledger revision. P2.4 then takes P2.3's best CatBoost (unrounded mean fold AUC, ties to the lower recipe id) for the reserved checks of sections 6.5 and 16.3.

## Open points I am unsure of

- Whether G should include P2.2's best (stricter for the smaller-useful and competitive routes) or stay H plus P2.3's own waves as in P2.2.
- CatBoost's default bootstrap for Plain and Ordered on CPU, and whether recipes should always set it explicitly.
- How interpret 0.7.8 infers feature types for integer columns (key, mode, time signature, year, month) and whether to fix them.
- Whether the full refit should run concurrently with later waves in a second process (section 5.1 requires a benchmark first).
