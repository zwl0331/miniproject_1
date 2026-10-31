# Agent C: mechanism experiments on the date signal (matched, fixed folds)

Date: 2026-10-30, 22:32 to 23:35 PDT. Scratchpad: `/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/9f6b6eec-676c-4338-a75c-d6002d6d0d06/scratchpad/agentC_mechanics/`. Nothing under the project root was written; no git commands were run.

## Setup

| item | value |
|---|---|
| data | `data/train.csv`, row 425 excluded (no audio features): 3,863 rows, 1,359 High (35.2 %) |
| folds | `artifacts/splits/production_v1.csv`, `primary_fold_{155,2026,1029}` (5 stratified group folds each, groups `primary_group`); score = mean of the 15 fold AUCs, with the three per-seed means |
| F2 | 13 audio floats + year, month (NaN for YYYY), day, day_of_year, weekday, days since 1970-01-01 (NaN unless the full date is present), date_precision = string length; column order as listed; no scaling, no imputation |
| L | `LGBMClassifier(n_estimators=600, learning_rate=0.01, num_leaves=63, min_child_samples=10, colsample_bytree=0.7, subsample=0.7, subsample_freq=1, reg_alpha=0.0714, reg_lambda=1, random_state=0, n_jobs=2, deterministic=True, force_col_wise=True)` |
| L+ | L + `date_count` (log1p of the distinct training-fold groups sharing the exact date string, own group left out for training rows, 0 for an unseen date) + `date_te3` (smoothed exact-date mean label, m = 3, prior = training-fold High rate; training rows cross-fitted with 5-fold StratifiedGroupKFold inside the training fold, shuffle seed 0; held-out rows encoded with the whole training fold; unseen date = prior). Both columns appended after F2 in that order. |
| code | `pipe.py` (features, cross-fitted blocks, `evaluate`), `q1.py` to `q4.py` (job queues, run in that order, one fit at a time), `analyze2.py` (tables), `results.jsonl` (every run with all 15 fold AUCs and parameters), `oof/*.npz` (out-of-fold predictions per fold seed), `tables.md`, `derived.json`, `q*.log` |
| environment | project `.venv`: numpy, pandas, scikit-learn 1.9.1, lightgbm 4.7.0, catboost 1.2.10; `PYTHONDONTWRITEBYTECODE=1`; n_jobs / thread_count = 2 |

Every row below changes one thing against its named reference on the same 15 folds with estimator seed 0 (G is about estimator seeds). "d" columns are differences versus the reference; "all seeds up" says whether the difference is positive on all three fold seeds.

## Reproduction check

| recipe | mine | expected | note |
|---|---|---|---|
| L | 0.8340 (0.8355 / 0.8327 / 0.8339) | 0.8339 (0.8344 / 0.8331 / 0.8343) | mean within 0.0001; per-seed within 0.0011 |
| L+ | 0.8353 (0.8364 / 0.8367 / 0.8329) | 0.8367 (0.8386 / 0.8368 / 0.8346) | mean 0.0014 low; see caveat |
| L+, inner cross-fit seed 1 | 0.8352 (0.8378 / 0.8347 / 0.8331) | | the inner seed alone moves per-seed means by up to 0.002 |

Caveat on L+: the recipe leaves the inner cross-fit seed and the column order free, and the inner seed alone moves a per-seed mean by 0.002 while leaving the 15-fold mean unchanged. The 0.0014 gap to the team's number is of that size and was not resolved (the team's own code was not read, by instruction). In this reproduction L+ beats L by +0.0013, but not on every fold seed (+0.0009 / +0.0040 / -0.0010). All comparisons against L+ below use my L+ with inner seed 0, so they are matched.

## Reproduction

| run | mean | s155 | s2026 | s1029 | d mean | d155 | d2026 | d1029 | all seeds up |
|---|---|---|---|---|---|---|---|---|---|
| L (reference recipe) | 0.8340 | 0.8355 | 0.8327 | 0.8339 | ref |  |  |  |  |
| L+ (L + date_count + date_te3), inner cross-fit seed 0 | 0.8353 | 0.8364 | 0.8367 | 0.8329 | +0.0013 | +0.0009 | +0.0040 | -0.0010 | mixed |
| L+ with inner cross-fit seed 1 (noise calibration; d vs L+) | 0.8352 | 0.8378 | 0.8347 | 0.8331 | -0.0001 | +0.0014 | -0.0020 | +0.0003 | mixed |

### A. Histogram resolution (reference L)

| run | mean | s155 | s2026 | s1029 | d mean | d155 | d2026 | d1029 | all seeds up |
|---|---|---|---|---|---|---|---|---|---|
| max_bin=1023, min_data_in_bin=3 | 0.8337 | 0.8350 | 0.8331 | 0.8330 | -0.0003 | -0.0005 | +0.0004 | -0.0009 | mixed |
| max_bin=4095, min_data_in_bin=3 | 0.8342 | 0.8362 | 0.8327 | 0.8336 | +0.0002 | +0.0007 | +0.0001 | -0.0003 | mixed |
| max_bin=255, min_data_in_bin=1 | 0.8347 | 0.8355 | 0.8332 | 0.8354 | +0.0007 | +0.0000 | +0.0005 | +0.0016 | mixed |
| max_bin=4095, min_data_in_bin=1 | 0.8345 | 0.8370 | 0.8325 | 0.8338 | +0.0004 | +0.0015 | -0.0001 | -0.0001 | mixed |
| max_bin_by_feature days=4095 (others 255), min_data_in_bin=1 | 0.8349 | 0.8365 | 0.8326 | 0.8357 | +0.0009 | +0.0010 | -0.0001 | +0.0018 | mixed |
| best of the above (days=4095, mdb=1) applied to L+ (reference L+) | 0.8357 | 0.8358 | 0.8369 | 0.8344 | +0.0004 | -0.0006 | +0.0002 | +0.0015 | mixed |

### B. Leaf size (reference L unless stated)

| run | mean | s155 | s2026 | s1029 | d mean | d155 | d2026 | d1029 | all seeds up |
|---|---|---|---|---|---|---|---|---|---|
| min_child_samples=3 | 0.8351 | 0.8371 | 0.8343 | 0.8339 | +0.0011 | +0.0017 | +0.0016 | +0.0000 | yes |
| min_child_samples=5 | 0.8352 | 0.8369 | 0.8339 | 0.8347 | +0.0012 | +0.0014 | +0.0012 | +0.0009 | yes |
| min_child_samples=20 | 0.8331 | 0.8352 | 0.8320 | 0.8322 | -0.0009 | -0.0003 | -0.0007 | -0.0017 | no (all <=0) |
| min_child_samples=3 + days=4095/mdb=1 bins | 0.8351 | 0.8361 | 0.8343 | 0.8350 | +0.0011 | +0.0006 | +0.0016 | +0.0012 | yes |
| min_child_samples=5 + days=4095/mdb=1 bins | 0.8354 | 0.8369 | 0.8345 | 0.8347 | +0.0014 | +0.0014 | +0.0018 | +0.0008 | yes |
| L, mcs=5, estimator seed 1 (reference L seed 1) | 0.8349 | 0.8368 | 0.8334 | 0.8343 | +0.0009 | +0.0004 | +0.0010 | +0.0011 | yes |
| L, mcs=5, estimator seed 2 (reference L seed 2) | 0.8337 | 0.8365 | 0.8324 | 0.8321 | -0.0002 | +0.0006 | -0.0017 | +0.0005 | mixed |
| L, mcs=5, mean of estimator seeds {0,1,2} (reference L seeds {0,1,2}) | 0.8354 | 0.8376 | 0.8338 | 0.8346 | +0.0005 | +0.0008 | -0.0000 | +0.0008 | mixed |
| L+ with min_child_samples=5 (reference L+) | 0.8354 | 0.8367 | 0.8366 | 0.8328 | +0.0000 | +0.0004 | -0.0001 | -0.0001 | mixed |
| L+ with min_child_samples=3 (reference L+) | 0.8359 | 0.8380 | 0.8368 | 0.8330 | +0.0006 | +0.0016 | +0.0001 | +0.0002 | yes |
| L+, mcs=5, estimator seed 1 (reference L+ seed 1) | 0.8353 | 0.8360 | 0.8358 | 0.8339 | +0.0001 | +0.0006 | -0.0008 | +0.0004 | mixed |
| L+, mcs=5, estimator seed 2 (reference L+ seed 2) | 0.8358 | 0.8367 | 0.8362 | 0.8343 | +0.0009 | +0.0009 | +0.0013 | +0.0006 | yes |
| L+, mcs=5, mean of seeds {0,1,2} (reference L+ seeds {0,1,2}) | 0.8361 | 0.8370 | 0.8369 | 0.8342 | +0.0004 | +0.0007 | +0.0002 | +0.0003 | yes |

### C. Exact date as LightGBM categorical (min_data_in_bin=1, 4095 bins for the date column; reference L with min_data_in_bin=1 = A_L_mb255_mdb1 unless stated)

| run | mean | s155 | s2026 | s1029 | d mean | d155 | d2026 | d1029 | all seeds up |
|---|---|---|---|---|---|---|---|---|---|
| added: cat_smooth=1, min_data_per_group=5 | 0.8100 | 0.8119 | 0.8110 | 0.8072 | -0.0247 | -0.0236 | -0.0223 | -0.0282 | no (all <=0) |
| added: cat_smooth=1, min_data_per_group=20 | 0.8146 | 0.8170 | 0.8162 | 0.8106 | -0.0201 | -0.0185 | -0.0171 | -0.0249 | no (all <=0) |
| added: cat_smooth=10, min_data_per_group=5 | 0.8357 | 0.8368 | 0.8356 | 0.8347 | +0.0010 | +0.0013 | +0.0023 | -0.0007 | mixed |
| added: cat_smooth=10, min_data_per_group=20 | 0.8349 | 0.8361 | 0.8347 | 0.8340 | +0.0002 | +0.0006 | +0.0014 | -0.0015 | mixed |
| added: cat_smooth=30, min_data_per_group=5 | 0.8343 | 0.8348 | 0.8350 | 0.8331 | -0.0004 | -0.0007 | +0.0018 | -0.0023 | mixed |
| replacing days: cat_smooth=1, min_data_per_group=5 | 0.8107 | 0.8139 | 0.8106 | 0.8075 | -0.0241 | -0.0216 | -0.0227 | -0.0279 | no (all <=0) |
| added, vs plain L: cat_smooth=10, min_data_per_group=5 | 0.8357 | 0.8368 | 0.8356 | 0.8347 | +0.0017 | +0.0013 | +0.0029 | +0.0008 | yes |
| added, vs plain L: cat_smooth=10, min_data_per_group=20 | 0.8349 | 0.8361 | 0.8347 | 0.8340 | +0.0009 | +0.0006 | +0.0020 | +0.0001 | yes |
| L+ + categorical, cat_smooth=1, mdpg=5 (reference L+) | 0.8088 | 0.8122 | 0.8075 | 0.8068 | -0.0265 | -0.0242 | -0.0292 | -0.0260 | no (all <=0) |
| L+ + categorical, cat_smooth=10, mdpg=5 (reference L+) | 0.8354 | 0.8370 | 0.8358 | 0.8334 | +0.0001 | +0.0007 | -0.0009 | +0.0005 | mixed |

### D. CatBoost (depth 6, 1000 it, lr 0.03, l2 3), raw date string as categorical

| run | mean | s155 | s2026 | s1029 | d mean | d155 | d2026 | d1029 | all seeds up |
|---|---|---|---|---|---|---|---|---|---|
| CatBoost on F2 (reference) | 0.8283 | 0.8282 | 0.8292 | 0.8274 | ref |  |  |  |  |
| CatBoost on F2 + date string categorical | 0.8320 | 0.8334 | 0.8294 | 0.8333 | +0.0038 | +0.0052 | +0.0001 | +0.0059 | yes |
| CatBoost on F2 + date string, vs L | 0.8320 | 0.8334 | 0.8294 | 0.8333 | -0.0020 | -0.0020 | -0.0033 | -0.0006 | no (all <=0) |

### E. Within-date nearest-neighbour block (cross-fitted; 4 columns)

| run | mean | s155 | s2026 | s1029 | d mean | d155 | d2026 | d1029 | all seeds up |
|---|---|---|---|---|---|---|---|---|---|
| L + NN block (reference L) | 0.8320 | 0.8327 | 0.8326 | 0.8305 | -0.0021 | -0.0027 | -0.0001 | -0.0034 | no (all <=0) |
| L+ + NN block (reference L+) | 0.8329 | 0.8348 | 0.8327 | 0.8311 | -0.0025 | -0.0016 | -0.0040 | -0.0017 | no (all <=0) |

### F. Exact-date target-encoding smoothing (reference L+, m=3)

| run | mean | s155 | s2026 | s1029 | d mean | d155 | d2026 | d1029 | all seeds up |
|---|---|---|---|---|---|---|---|---|---|
| m=1 | 0.8356 | 0.8362 | 0.8370 | 0.8336 | +0.0003 | -0.0001 | +0.0002 | +0.0007 | mixed |
| m=0.5 | 0.8351 | 0.8351 | 0.8363 | 0.8340 | -0.0002 | -0.0013 | -0.0004 | +0.0012 | mixed |
| unsmoothed mean + raw count column | 0.8347 | 0.8353 | 0.8365 | 0.8322 | -0.0006 | -0.0010 | -0.0002 | -0.0007 | no (all <=0) |
| m=3, one label per (date, group) | 0.8349 | 0.8358 | 0.8343 | 0.8346 | -0.0004 | -0.0005 | -0.0025 | +0.0018 | mixed |

### G. Estimator-seed averaging (reference seed 0 of the same recipe)

| run | mean | s155 | s2026 | s1029 | d mean | d155 | d2026 | d1029 | all seeds up |
|---|---|---|---|---|---|---|---|---|---|
| L+ seed 1 alone | 0.8352 | 0.8354 | 0.8367 | 0.8335 | -0.0001 | -0.0009 | -0.0001 | +0.0006 | mixed |
| L+ seed 2 alone | 0.8348 | 0.8358 | 0.8349 | 0.8337 | -0.0005 | -0.0006 | -0.0018 | +0.0009 | mixed |
| L+ seed 3 alone | 0.8356 | 0.8365 | 0.8356 | 0.8348 | +0.0003 | +0.0001 | -0.0012 | +0.0019 | mixed |
| L+ seed 4 alone | 0.8350 | 0.8357 | 0.8358 | 0.8334 | -0.0003 | -0.0006 | -0.0010 | +0.0006 | mixed |
| L+ mean of seeds {0,1,2} | 0.8356 | 0.8363 | 0.8367 | 0.8339 | +0.0003 | -0.0001 | -0.0000 | +0.0011 | mixed |
| L+ mean of seeds {0..4} | 0.8359 | 0.8368 | 0.8366 | 0.8342 | +0.0006 | +0.0004 | -0.0001 | +0.0013 | mixed |
| L seed 1 alone | 0.8340 | 0.8364 | 0.8324 | 0.8332 | -0.0000 | +0.0009 | -0.0003 | -0.0007 | mixed |
| L seed 2 alone | 0.8339 | 0.8360 | 0.8341 | 0.8316 | -0.0001 | +0.0005 | +0.0014 | -0.0023 | mixed |
| L mean of seeds {0,1,2} | 0.8348 | 0.8368 | 0.8338 | 0.8338 | +0.0008 | +0.0014 | +0.0012 | -0.0001 | mixed |

### H. Blend geometry (reference L+ seed 0); weights are w_L+ / w_other, ranks within each validation fold

| run | mean | s155 | s2026 | s1029 | d mean | d155 | d2026 | d1029 | all seeds up |
|---|---|---|---|---|---|---|---|---|---|
| ExtraTrees(500, min_samples_leaf=2, max_features=1.0) alone | 0.8267 | 0.8272 | 0.8291 | 0.8238 | -0.0086 | -0.0092 | -0.0076 | -0.0090 | no (all <=0) |
| L+/ET probability 0.75/0.25 | 0.8378 | 0.8389 | 0.8391 | 0.8354 | +0.0025 | +0.0025 | +0.0024 | +0.0025 | yes |
| L+/ET rank 0.75/0.25 | 0.8382 | 0.8392 | 0.8398 | 0.8357 | +0.0029 | +0.0028 | +0.0031 | +0.0028 | yes |
| L+/ET logit 0.75/0.25 | 0.8379 | 0.8390 | 0.8392 | 0.8354 | +0.0026 | +0.0027 | +0.0025 | +0.0025 | yes |
| L+/ET probability 0.5/0.5 | 0.8378 | 0.8385 | 0.8395 | 0.8356 | +0.0025 | +0.0021 | +0.0028 | +0.0027 | yes |
| L+/ET rank 0.5/0.5 | 0.8376 | 0.8384 | 0.8395 | 0.8349 | +0.0023 | +0.0020 | +0.0027 | +0.0020 | yes |
| L+/ET logit 0.5/0.5 | 0.8385 | 0.8393 | 0.8401 | 0.8361 | +0.0032 | +0.0029 | +0.0034 | +0.0032 | yes |
| incidental: L+/L probability 0.75/0.25 | 0.8372 | 0.8384 | 0.8377 | 0.8354 | +0.0019 | +0.0020 | +0.0010 | +0.0026 | yes |
| incidental: L+/L rank 0.75/0.25 | 0.8371 | 0.8382 | 0.8377 | 0.8353 | +0.0018 | +0.0018 | +0.0010 | +0.0025 | yes |
| incidental: L+/L probability 0.5/0.5 | 0.8376 | 0.8388 | 0.8375 | 0.8364 | +0.0023 | +0.0025 | +0.0008 | +0.0035 | yes |
| incidental: L+/L logit 0.5/0.5 | 0.8376 | 0.8388 | 0.8375 | 0.8364 | +0.0023 | +0.0025 | +0.0008 | +0.0036 | yes |

## Interpretation per family

**A. Histogram resolution: nothing.** Five settings, all within -0.0003 to +0.0009 of L with mixed per-seed signs; the best on L (`days` at 4,095 bins, `min_data_in_bin=1`) is +0.0004 on L+, mixed. Finer bins on `days` do not let the trees find anything the 255-bin histogram misses. Not run: max_bin=1023 with min_data_in_bin=1.

**B. Leaf size: a small real effect on L that does not transfer to L+.** With estimator seed 0, `min_child_samples` 5 is +0.0012 and 3 is +0.0011 over L, each positive on all three fold seeds, and 20 is -0.0009 (negative on all seeds), with or without the bin change. Across estimator seeds the gain shrinks: mcs=5 is +0.0009 (seed 1, all seeds up) and -0.0002 (seed 2, mixed); the three-seed averages differ by +0.0005 (mixed). On L+ the same change is +0.0000 (mcs=5) and +0.0006 (mcs=3, all seeds up but two at +0.0001); on the three-seed averages of L+ it is +0.0004. Reading: small leaves and the cross-fitted date blocks pick up the same same-date clusters; once the blocks are in, the leaf size no longer matters. Worth at most 0.0005.

**C. Exact date as a LightGBM categorical: dangerous at cat_smooth=1, neutral at 10.** LightGBM's categorical split uses in-sample gradient statistics per category (no cross-fitting), so with about 1,400 date codes of 1 to 3 rows each, `cat_smooth=1` overfits catastrophically: -0.020 to -0.027 on every variant (added, replacing `days`, on L+), regardless of `min_data_per_group`. With `cat_smooth=10` the model barely uses the column (11 to 13 of 6,200 splits in a check fit) and the result is +0.0010 (mdpg=5) or +0.0002 (mdpg=20) against the matched reference (L with `min_data_in_bin=1`), mixed across seeds; against plain L it reads +0.0017 but part of that is the bin setting. On L+ it adds +0.0001. cat_smooth=30 (mdpg=5) is -0.0004 against the matched reference, mixed, so stronger smoothing only removes the column's small use. Also note the default `min_data_in_bin=3` drops every date with fewer than 3 rows from its own bin, so the default settings would silently ignore most dates.

**D. CatBoost ordered target statistics on the raw date string: +0.0038 inside CatBoost, still below L.** CatBoost on F2 scored 0.8283 (team: about 0.8306 with their parameters); adding the date string as a native categorical gave 0.8320, positive on all seeds (+0.0052 / +0.0001 / +0.0059). That is the cleanest evidence that a leakage-safe exact-date statistic carries real signal, but CatBoost with it is still 0.0020 below L and 0.0033 below L+. (An independent run reported by the coordinator collapsed to 0.8118; mine used CatBoost's default ordered statistics on CPU with `cat_features` passed at fit. I cannot reconcile the two without seeing that run.)

**E. Within-date nearest-neighbour block: hurts.** The four cross-fitted columns cost -0.0021 on L and -0.0025 on L+, non-positive on every seed. Label-free diagnostics: 69 % of held-out rows have a same-date neighbour; on those rows the kernel-weighted same-date mean alone has AUC 0.671 versus 0.665 for date_te3, so the block carries the same information as the target encoding plus the noise of single-neighbour labels, and the training rows see a noisier (80 % reference) version than the held-out rows. No leakage: the reference set never contains the query's group.

**F. Smoothing of the exact-date encoding: not a lever.** m=1 is +0.0003 (mixed), m=0.5 is -0.0002 (mixed), the unsmoothed mean with a count column is -0.0006 (non-positive on all seeds), one label per (date, group) is -0.0004 (mixed, with a 0.004 swing between seeds). All inside the inner-seed noise.

**G. Pure estimator-seed averaging: at most +0.0006.** Single L+ seeds span 0.8348 to 0.8356. The mean of seeds {0,1,2} is +0.0003 and of {0..4} +0.0006 over seed 0, mixed across fold seeds; for L the {0,1,2} mean is +0.0008, mixed. Seed averaging smooths but does not move the score.

**H. Blending L+ with ExtraTrees: the one repeat-stable gain, and the geometry is secondary.** ExtraTrees alone is 0.8267, yet every blend is +0.0023 to +0.0032 over L+ with all three seeds up and near-identical per-seed deltas. At 0.75/0.25: probability +0.0025, logit +0.0026, rank +0.0029. At 0.5/0.5: logit +0.0032, probability +0.0025, rank +0.0023. Differences between geometries are 0.0004 to 0.0009, within noise; the blend itself is the gain. Incidentally, blending L+ with plain L (no cross-fitted blocks) is also +0.0019 to +0.0023 with all seeds up, which says the cross-fitted date columns add variance that any second model averages out.

## Ranked list

Repeat-stable gains (positive on all three fold seeds):

| rank | change | gain over its reference | per seed |
|---|---|---|---|
| 1 | L+ blended with ExtraTrees (logit average 0.5/0.5) | +0.0032 over L+ (0.8385) | +0.0029 / +0.0034 / +0.0032 |
| 1 | same blend, rank average 0.75/0.25 | +0.0029 over L+ (0.8382) | +0.0028 / +0.0031 / +0.0028 |
| 1 | same blend, probability average 0.75/0.25 (the current geometry) | +0.0025 over L+ (0.8378) | +0.0025 / +0.0024 / +0.0025 |
| 2 | L+ blended with plain L, probability 0.5/0.5 (incidental) | +0.0023 over L+ (0.8376) | +0.0025 / +0.0008 / +0.0035 |
| 3 | CatBoost: raw date string as native categorical | +0.0038 over CatBoost F2 (0.8320), but -0.0020 below L | +0.0052 / +0.0001 / +0.0059 |
| 4 | L with min_child_samples=5 (seed 0; also +0.0009 at seed 1, -0.0002 at seed 2; +0.0000 on L+) | +0.0012 over L (0.8352) | +0.0014 / +0.0012 / +0.0009 |
| 5 | L+ with min_child_samples=3 | +0.0006 over L+ (0.8359) | +0.0016 / +0.0001 / +0.0002 |

Did nothing (within the ±0.001 noise of the inner cross-fit seed, mixed signs): histogram resolution (A, all five settings, and the best on L+ at +0.0004); target-encoding smoothing m=1 or 0.5 and the per-group variant (F); the LightGBM categorical date at cat_smooth=10 (+0.0001 on L+); estimator-seed averaging (G, +0.0003 to +0.0008).

Hurt: within-date nearest-neighbour block (E, -0.0021 on L, -0.0025 on L+, all seeds); LightGBM categorical date at cat_smooth=1 (C, -0.020 to -0.027, all seeds, every variant); unsmoothed exact-date mean plus count (F, -0.0006, all seeds); min_child_samples=20 (B, -0.0009, all seeds).

Not run: A with max_bin=1023 and min_data_in_bin=1; C replacing `days` at cat_smooth=10; bootstrap intervals (per-seed differences were used as the stability evidence instead).

Leakage review: every label-fitted statistic (date_te3 and its variants, the NN block) is cross-fitted inside the training fold with group folds, so a training row never sees its own group; held-out rows only see the training fold; date_count is label-free with the own group left out. The LightGBM categorical (C) is the one mechanism that is not cross-fitted by construction, which is why cat_smooth=1 collapses. The blends (H) and seed averages (G) are computed from out-of-fold predictions of the same folds, with ranks taken within each validation fold.
