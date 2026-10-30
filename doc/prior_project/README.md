# Evidence carried over from the earlier audio-only project

Copied 2026-10-30 in task P2.8 on Sam's instruction, verbatim: "Take what you can get out of the old project and then update your plan and your current working, and from then on just don't depend on the old project anymore. I may remove it later." The source was `/Users/zwl0331/Documents/Mini_Project_01`, never committed to git there. This folder holds everything the current project takes from it; no file, script or plan reads the source folder.

## What the earlier project was

| Aspect | Earlier project |
|---|---|
| Task and data | The same course task and the same 2026 data files, built 2026-10-27 and 2026-10-28 under the same review gate |
| Inputs | The 13 audio features only; the release date and the identifiers were excluded |
| Rows | The 3,863 complete rows, split into 3,090 development rows and a 773-row holdout (seed 20261027) |
| Groups | "Expanded" groups: the same `uri`, or audio features equal within declared tolerances (3,624 groups) |
| Selection metric | The "15-fold mean": mean fold AUC over three repeated 5-fold group splits of the development rows (`fold5_s51`, `fold5_s52`, `fold5_s53`). The same folds tuned and compared the settings, so its comparisons carry an unquantified selection bias, as that project disclosed |
| Adoption rule for a treatment | A mean gain of at least 0.003, positive on all three repeats ("gain test") |
| End | A nested evaluation over 10 outer folds and a holdout audit |

Its numbers come from a different validation setup (rows, groups, folds, features). They are not comparable with the current project's grouped CV and are never added to it. They are reasons to test, not expected effects.

## Files

Copied unchanged into `files/`; each sha256 equals the source file's at copy time.

| File | Content | sha256 |
|---|---|---|
| `PROJECT_CONTEXT.txt` | The earlier project's own status narrative (tasks, runs, nested and holdout results) | `72042b84ab1882a396dd8ce1c7b96e8193996f71fcde65e91fba604d1ba6cef8` |
| `split_summary.json` | Its split settings and group statistics | `3129c3e342cc49936b31d013fa83fc44922c7956d8122ab358ef7d5c93c74d36` |
| `p1.2_screen_summary.csv` | The screen of 15 families | `b842db3463427cb9dbc159d0ec4d84ba2ae0b20cffe12e492a5156c95609ae4b` |
| `p1.4_stage2_families.csv` | The tuned searches of 12 families | `b0479258c0c487e077dbb5294450da7d33929ffcb614d941dfc8411b16847422` |
| `p1.6_treatments.csv` | Every treatment of the four finalists against their tuned settings | `1c09ca36ed3f37e4a26f9e994a260d2abc8993ffebb975fd2bce72e82be83807` |
| `p1.7_treatments.csv` | Treatment combinations and iteration multipliers | `f998eb3b5ce3802a13304c6934a6c383208867eed947db6c4393603a371590dd` |
| `p1.7_finalists.json` | The finalists with their adopted treatments | `a46b0073c62d29a3f6e956d75b340f0a9106e2f22b8099aad67b8c67eedf08ab` |
| `p1.9_picks.json` | The final picks, with CatBoost candidate 31's settings | `5c44000a492206871494b420d13d6b1244703ef66b9f148872029ec35cbdb05a` |

No code was copied. The two feature bundles below are transcribed as formulas from the earlier project's `analysis/stage3.py` (sha256 `1b0981045b2aa4dccef68a85cd4f92e63fcb12e1c3dc74c33c7105e4c5c6ec60` at copy time); the current project defines its own blocks (plan section 16.3).

## Carried over

Earlier numbers are 15-fold development means, audio only, each against the same tuned setting without the treatment.

| Item | Earlier evidence | Use here |
|---|---|---|
| Music bundle | Four columns appended, raw key and tempo kept: sin and cos of 2π((7 key) mod 12)/12 (key on the circle of fifths) and sin and cos of 2π log2(tempo) (tempo octave: T and 2T coincide). CatBoost candidate 31: 0.760667 to 0.765095, +0.004428, repeats +0.006748, +0.004030, +0.002507. EBM +0.002583 (3 of 3 repeats up). LightGBM with linear leaves +0.003418 (2 of 3). Random forest +0.000081 (2 of 3) | The `music` block; the anchor check M1 in P2.8; reserved CatBoost and EBM checks for P2.4 (plan section 16.3) |
| Interactions bundle | energy x loudness, danceability x tempo, acousticness x instrumentalness. CatBoost +0.004287 (3 of 3), random forest +0.002973 (3 of 3), LightGBM linear +0.002721 (3 of 3), EBM +0.001007 (2 of 3). CatBoost interactions and music against music alone: +0.002995 (3 of 3, worst repeat +0.000911), short of the 0.003 rule | The current `interactions` block shares the first and third terms and replaces danceability x tempo with others (plan section 6.3); a CatBoost check with and without it is reserved for P2.4 |
| CatBoost candidate 31 | Depth 6, 1,944 iterations, learning rate 0.009535096982557112, L2 4.757135294733201, random strength 0, `rsm` 0.8880009208730744, Bayesian bootstrap with bagging temperature 1.793473187168845; the earlier single pick (with music) | One declared configuration of P2.3's initial CatBoost batch, on F2, with the factory parameters it needs added by P2.3; not assumed to be optimal with release dates |
| Early stopping | CatBoost -0.012313 (0 of 3 repeats up) | Consistent with the fixed iterations of plan section 5.3; no change |
| Log and logit transforms on trees | Random forest -0.001736, CatBoost -0.002251, EBM -0.003358, LightGBM linear -0.004915 | Consistent with transforms for the smooth, kNN and neural models only (plan section 6.2); no change |
| TabM and RealMLP | Best tuned settings on `fold5_s51`: TabM 0.741539 (150 warnings), RealMLP 0.740346 (860 warnings), random forest 0.765124 | P2.5 keeps TabM as a challenger with low expectations; RealMLP stays out of the plan |
| Ensembles | Nested evaluation (its Stage 5): the four-model ensemble procedure, compact tuning with medoid weights chosen from inner scores, 0.7626 against the single-model procedure 0.7594 (+0.0032; repeats +0.0014, +0.0050). Separately, the frozen fixed weights 0.3, 0.3, 0.2, 0.2: holdout +0.0027 over the single pick (interval -0.0054 to +0.0111) | Consistent with the planned ensembles (P2.6a, P2.6b); no change |

## Not carried over

- The earlier project's release-date phase stopped at a consultation. Its what-if numbers were unreviewed and scored on an opened holdout; none is used.
- Its descriptive statements about identifiers and row order were never reviewed; P2.8 tests them on the current protocol.
