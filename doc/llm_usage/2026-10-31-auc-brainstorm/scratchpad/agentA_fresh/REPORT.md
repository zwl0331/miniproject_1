# Fresh look at CS 155 Miniproject 1 (agent A, independent pass)

Date: 2026-10-30. All work in this scratchpad directory; nothing written in the project. Protocol: `artifacts/splits/production_v1.csv`, 3 seeds x 5 grouped stratified folds, eligible rows only (3,863 rows, the empty row 425 dropped). Score = mean of 15 fold AUCs. Every target-derived feature is fitted inside the training part of each fold and cross-fitted (5 inner GroupKFold splits) for the training rows. Models run with 2 threads, one fit at a time.

Two metrics are reported for every run:

- `mean15`: the mandated score (all eligible validation rows).
- `singleton15`: the same 15 folds, AUC restricted to validation rows whose exact (13 audio features + date string) key occurs once in train+test. This matters because the test set is 97.5% such rows (841 of 863) while train is 88.8% (3,431 of 3,863); see section (d).

## (a) Results table

Baseline `lgb_base`: LightGBM (400 trees, lr 0.03, 15 leaves, min_child_samples 20, subsample 0.8, colsample 0.8, lambda 1) on the 13 audio features + year, month, day, day-of-year, weekday, date precision, days-since-epoch. Deltas are against that baseline. `cnt` = `cnt_exact`, `datecnt` = `date_cnt_all`, `teDate` = cross-fitted date-string target encoding, `knn20` = cross-fitted 20-nearest-neighbour label mean (definitions in (b)).

| experiment | mean15 | seed155 | seed2026 | seed1029 | singleton15 | d mean | d 155 | d 2026 | d 1029 | d single | secs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lgb_base | 0.8311 | 0.8336 | 0.8297 | 0.8299 | 0.8273 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 24 |
| lgb_base+cnt | 0.8499 | 0.8524 | 0.8478 | 0.8495 | 0.8250 | +0.0188 | +0.0187 | +0.0181 | +0.0196 | -0.0023 | 25 |
| lgb_base+cnt+datecnt | 0.8501 | 0.8517 | 0.8475 | 0.8510 | 0.8257 | +0.0190 | +0.0181 | +0.0178 | +0.0211 | -0.0015 | 19 |
| lgb_base+cnt+datecnt+teDate | 0.8520 | 0.8523 | 0.8504 | 0.8533 | 0.8278 | +0.0209 | +0.0186 | +0.0207 | +0.0235 | +0.0005 | 25 |
| lgb+teDate w=1/cnt | 0.8510 | 0.8519 | 0.8495 | 0.8517 | 0.8273 | +0.0200 | +0.0183 | +0.0198 | +0.0218 | +0.0001 | 25 |
| lgb+teDate nl7 n600 | 0.8500 | 0.8512 | 0.8492 | 0.8497 | 0.8242 | +0.0190 | +0.0176 | +0.0195 | +0.0198 | -0.0031 | 18 |
| lgb+teDate nl31 mcs40 n300 | 0.8506 | 0.8516 | 0.8494 | 0.8509 | 0.8270 | +0.0195 | +0.0180 | +0.0197 | +0.0210 | -0.0003 | 33 |
| lgb+teDate lr.015 n800 | 0.8521 | 0.8527 | 0.8500 | 0.8536 | 0.8280 | +0.0211 | +0.0191 | +0.0203 | +0.0237 | +0.0007 | 43 |
| lgb+teDate extra_trees n800 | 0.8526 | 0.8530 | 0.8521 | 0.8526 | 0.8270 | +0.0215 | +0.0194 | +0.0224 | +0.0227 | -0.0003 | 44 |
| lgb+teDate colsample.5 | 0.8520 | 0.8519 | 0.8512 | 0.8527 | 0.8274 | +0.0209 | +0.0183 | +0.0215 | +0.0228 | +0.0001 | 28 |
| lgb+teDate mcs50 l2=5 | 0.8503 | 0.8508 | 0.8494 | 0.8506 | 0.8260 | +0.0192 | +0.0171 | +0.0197 | +0.0207 | -0.0013 | 27 |
| lgb+teDate+ENG | 0.8516 | 0.8513 | 0.8517 | 0.8517 | 0.8275 | +0.0205 | +0.0176 | +0.0220 | +0.0218 | +0.0002 | 28 |
| cat d6 +teDate | 0.8510 | 0.8531 | 0.8485 | 0.8515 | 0.8246 | +0.0200 | +0.0194 | +0.0188 | +0.0216 | -0.0026 | 56 |
| lgb+teDate+knn20 (oof) | 0.8527 | 0.8526 | 0.8527 | 0.8526 | 0.8287 | +0.0216 | +0.0190 | +0.0230 | +0.0227 | +0.0014 | 31 |
| lgb+teDate+knn20 bag5 | 0.8537 | 0.8541 | 0.8532 | 0.8537 | 0.8293 | +0.0226 | +0.0205 | +0.0235 | +0.0239 | +0.0021 | 125 |
| cat d4 +teDate+knn20+dateCat | 0.8118 | 0.8085 | 0.8158 | 0.8111 | 0.7792 | -0.0193 | -0.0251 | -0.0139 | -0.0187 | -0.0481 | 62 |
| cat d4 +teDate+knn20 | 0.8511 | 0.8528 | 0.8496 | 0.8509 | 0.8235 | +0.0200 | +0.0192 | +0.0199 | +0.0210 | -0.0038 | 33 |
| lgb+teDate+teYM+knn20 | 0.8498 | 0.8521 | 0.8497 | 0.8475 | 0.8250 | +0.0187 | +0.0185 | +0.0200 | +0.0176 | -0.0023 | 29 |
| lgb+teDate+knn x4 | 0.8504 | 0.8508 | 0.8501 | 0.8503 | 0.8261 | +0.0193 | +0.0171 | +0.0204 | +0.0204 | -0.0012 | 40 |
| lgb+teDate+knn20 | 0.8527 | 0.8526 | 0.8527 | 0.8526 | 0.8287 | +0.0216 | +0.0190 | +0.0230 | +0.0227 | +0.0014 | 29 |
| extratrees+teDate+knn20 | 0.8468 | 0.8480 | 0.8459 | 0.8467 | 0.8207 | +0.0158 | +0.0143 | +0.0162 | +0.0168 | -0.0066 | 38 |
| spline-logreg+teDate+knn20 | 0.8320 | 0.8336 | 0.8334 | 0.8291 | 0.8006 | +0.0010 | +0.0000 | +0.0037 | -0.0008 | -0.0267 | 5 |
| knn40clf+teDate | 0.8189 | 0.8196 | 0.8190 | 0.8182 | 0.7867 | -0.0121 | -0.0140 | -0.0107 | -0.0117 | -0.0406 | 5 |
| xgb d4 +teDate+knn20 | 0.8521 | 0.8518 | 0.8524 | 0.8523 | 0.8270 | +0.0211 | +0.0182 | +0.0226 | +0.0224 | -0.0003 | 18 |
| ebm +teDate+knn20 | 0.8419 | 0.8430 | 0.8428 | 0.8399 | 0.8111 | +0.0108 | +0.0093 | +0.0130 | +0.0100 | -0.0162 | 116 |
| lgb teDate m=1 knn20 | 0.8521 | 0.8517 | 0.8519 | 0.8526 | 0.8272 | +0.0210 | +0.0181 | +0.0222 | +0.0227 | -0.0001 | 28 |
| lgb teDate m=10 knn20 | 0.8522 | 0.8530 | 0.8519 | 0.8517 | 0.8280 | +0.0211 | +0.0194 | +0.0222 | +0.0218 | +0.0007 | 23 |
| lgb teDate m=3 knn10 | 0.8517 | 0.8528 | 0.8506 | 0.8518 | 0.8276 | +0.0206 | +0.0191 | +0.0209 | +0.0219 | +0.0003 | 22 |
| lgb teDate m=3 knn35 | 0.8514 | 0.8510 | 0.8523 | 0.8511 | 0.8274 | +0.0204 | +0.0174 | +0.0226 | +0.0212 | +0.0001 | 23 |
| lgb teDate knn20 NO cnt | 0.8351 | 0.8348 | 0.8364 | 0.8342 | 0.8293 | +0.0041 | +0.0012 | +0.0067 | +0.0043 | +0.0020 | 23 |
| lgb NO cnt (oof) | 0.8351 | 0.8348 | 0.8364 | 0.8342 | 0.8293 | +0.0041 | +0.0012 | +0.0067 | +0.0043 | +0.0020 | 22 |
| et NO cnt (oof) | 0.8249 | 0.8248 | 0.8268 | 0.8231 | 0.8204 | -0.0062 | -0.0088 | -0.0029 | -0.0068 | -0.0069 | 34 |

Rank-average blends of the saved out-of-fold predictions (per fold, ranks averaged with the given weights; all member models use cnt, datecnt, teDate, knn20):

| rank-average blend (weights) | mean15 | seed155 | seed2026 | seed1029 | singleton15 | d mean vs lgb alone | d single |
|---|---|---|---|---|---|---|---|
| {'lgb': 1} | 0.8527 | 0.8526 | 0.8527 | 0.8526 | 0.8287 | +0.0000 | +0.0000 |
| {'lgb': 1, 'et': 0.5} | 0.8550 | 0.8553 | 0.8549 | 0.8548 | 0.8308 | +0.0023 | +0.0021 |
| {'lgb': 1, 'et': 1} | 0.8546 | 0.8551 | 0.8541 | 0.8545 | 0.8301 | +0.0019 | +0.0014 |
| {'lgb': 1, 'lr': 0.3} | 0.8521 | 0.8525 | 0.8523 | 0.8513 | 0.8271 | -0.0006 | -0.0016 |
| {'lgb': 1, 'knn': 0.3} | 0.8537 | 0.8538 | 0.8538 | 0.8536 | 0.8290 | +0.0011 | +0.0003 |
| {'lgb': 1, 'et': 0.5, 'lr': 0.2} | 0.8544 | 0.8549 | 0.8541 | 0.8540 | 0.8297 | +0.0017 | +0.0011 |
| {'lgb': 1, 'et': 0.5, 'lr': 0.2, 'knn': 0.2} | 0.8544 | 0.8548 | 0.8541 | 0.8541 | 0.8295 | +0.0017 | +0.0008 |
| {'lgb': 1, 'xgb': 1} | 0.8535 | 0.8533 | 0.8538 | 0.8536 | 0.8291 | +0.0009 | +0.0004 |
| {'lgb': 1, 'xgb': 1, 'et': 0.5} | 0.8550 | 0.8550 | 0.8550 | 0.8550 | 0.8306 | +0.0023 | +0.0019 |
| {'lgb': 1, 'ebm': 1} | 0.8510 | 0.8512 | 0.8518 | 0.8499 | 0.8241 | -0.0017 | -0.0046 |
| {'lgb': 1, 'xgb': 1, 'ebm': 1} | 0.8526 | 0.8526 | 0.8533 | 0.8519 | 0.8264 | -0.0001 | -0.0022 |
| {'lgb': 1, 'xgb': 1, 'ebm': 1, 'et': 0.5} | 0.8540 | 0.8543 | 0.8543 | 0.8533 | 0.8281 | +0.0013 | -0.0006 |
| {'lgb': 1, 'xgb': 1, 'ebm': 1, 'et': 0.5, 'lr': 0.2} | 0.8536 | 0.8540 | 0.8540 | 0.8529 | 0.8276 | +0.0010 | -0.0011 |

Test-like AUC (per fold: all singleton validation rows plus duplicate rows subsampled to 2.55% of the subset, as in the test; mean over 15 folds x 20 draws):

| candidate | test-like AUC | singleton AUC |
|---|---|---|
| lgb with cnt | 0.8353 | 0.8287 |
| lgb without cnt | 0.8310 | 0.8293 |
| lgb + 0.5 ET, with cnt | 0.8372 | 0.8308 |
| lgb + 0.5 ET, without cnt | 0.8325 | 0.8311 |
| lgb + 0.5 ET without cnt, then rank boost of rows with cnt>1 | 0.8372 | 0.8311 |

Failed or not run: CatBoost with the raw date string as a native categorical ran but collapsed to 0.8118 (overfits the 1,629-level categorical even with ordered target statistics). Nothing else failed.

## (b) What gave real, repeat-stable gains

Stable means: positive on all three seeds, and in the same direction on `singleton15` where it is supposed to help singletons.

1. **Exact-duplicate count, `cnt_exact`** (+0.019 mean15 on every seed; singleton15 unchanged, -0.002). Definition: concatenate train and test, hash the 14-tuple (13 audio columns + the raw `track_album_release_date` string), `cnt_exact` = number of rows in train+test with that hash. In train: cnt 1 -> 29.7% High (n 3,431), cnt 2 -> 76.2% (374), cnt 3 -> 96.3% (54), cnt 4 -> 100% (4). It is label-free, so it needs no cross-fitting, and the rules allow it (derived from the provided files only). But it only pays on the test through the 22 within-test duplicate rows (11 pairs): on the test-like composition it is worth about +0.004 to +0.005, not +0.019. An equivalent and simpler alternative for the submission is to train without it and add a rank boost to the 22 duplicated test rows (table above: identical 0.8372). Do not read the 0.855 as a test expectation.

2. **Cross-fitted target encoding of the exact date string, `te_date`** (+0.0019 mean15; seeds +0.0006, +0.0029, +0.0023; singleton +0.002). Definition: for a validation/test row, `(sum of y over training rows with the same date string + m * prior) / (n + m)` with prior = training-fold High rate and m = 3; for training rows the same statistic from the other 4 of 5 inner GroupKFold splits (grouped by track). Also pass `n_date` (the count of training rows with that date). m in {1, 3, 10} is a wash. Why it works: among tracks sharing an exact date, a leave-one-out date mean has AUC 0.673 versus 0.581 for the leave-one-out year mean; 623 of 863 test rows share a date string with a training row (494 with at least two). Adding a year-month encoding on top hurts (-0.003).

3. **Cross-fitted kNN label mean, `knn20`** (+0.0007 mean15 on top of te_date, all three seeds +0.0003 to +0.0009; singleton +0.001). Definition: standardise the 13 audio features + year on the training fold, mean label of the 20 nearest training neighbours (Euclidean); cross-fitted with 5 inner grouped splits for training rows. k = 10, 20, 35 all within noise; adding several kNN variants at once hurts (-0.002).

4. **Rank-average blend LightGBM + 0.5 x ExtraTrees** (+0.0023 mean15: seeds +0.0027, +0.0022, +0.0022; singleton +0.0021). ExtraTrees: 500 trees, min_samples_leaf 3, max_features 0.5, median imputation, same features. Adding XGBoost is neutral, adding EBM, spline logistic regression or a kNN classifier hurts.

5. **Seed bagging** (5 LightGBM seeds, +0.001 mean15 on every seed, +0.0006 singleton). Cheap, stable, small.

Together: 0.8311 -> 0.8550 on the protocol metric (of which about 0.019 is the duplicate artifact), 0.8273 -> 0.8308 on singletons, and about 0.835 -> 0.837 expected on the test composition. Everything that is not in this list (LightGBM hyperparameters in a wide range, engineered audio ratios, log transforms, loudness relative to year median, CatBoost depth 4 and 6, XGBoost, EBM, 1/cnt sample weights) moved the score by less than the between-seed spread.

## (c) Promising, not done (ranked)

1. **Submit-side handling of the test duplicates.** Whatever the final model, make sure the 11 test pairs are ranked near the top (via `cnt_exact` or an explicit boost). Worth about +0.004 on the test, costs nothing. Check that `cnt_exact` for test rows is computed on train+test (no test row matches a train row; 22 test rows match another test row).
2. **A stacking meta-learner** (logistic regression on the rank-OOFs of LightGBM, ExtraTrees, XGBoost, kNN20, te_date, with era as an extra input) instead of hand weights. The 2015-2019 era is the weak spot (singleton OOF AUC 0.709 vs 0.885 before 2000 and 0.839 for 2023-24); era-dependent weights might buy 0.001-0.003.
3. **Pseudo-labelling / transductive use of the 863 test rows** (fit, label confident test rows, refit). Small data makes this plausible for +0.001-0.003; needs careful CV (pseudo-label only from the training fold's model).
4. **Monotone constraints** in LightGBM (loudness up, instrumentalness down, acousticness down, cnt_exact up) as regularisation. Untested here; typically 0 to +0.002 on 4k rows.
5. **Label-noise handling for the 33 conflicting-label duplicate groups** (same track, both labels): drop or soft-label them. Expected effect tiny.
6. **Checking whether the public LB cluster is real.** With 863 test rows at a 31% High rate the Hanley-McNeil standard error of an AUC near 0.84 is about 0.016; if the public LB is a 50% split it is about 0.023. A cluster at 0.86 versus 0.836 is 1 to 1.5 standard errors and public scores are max-selected over submissions. The team's own CV 0.8315 -> public 0.836 is consistent with that noise level. I would not chase 0.86 with changes that do not show on the 15-fold protocol.
7. **What the 0.86 teams may be doing that this pipeline would not:** TabPFN or another pretrained tabular model (commonly +0.01 to +0.02 AUC on 4k-row problems; forbidden here by the pretrained-model ruling), or simply picking the best public score. Nothing in the data suggests an unused legitimate signal of that size: identifiers are random across train/test, row order is shuffled (lag-1 label autocorrelation -0.007), the test `ID` is uncorrelated with year (-0.03).

## (d) Surprises in the data

- **The duplicate structure is a popularity signal and the test is depleted of it.** 432 train rows (11.2%) are exact copies of another row and 75% of them are High. Zero test rows match a train track exactly (2 match on audio only, with a different date: same recording, different release). The test IDs run 2 to 967 with 103 gaps, and the within-test duplicate rate is 2.5% versus 11.2% in train: consistent with a random row split followed by removal of test rows whose track also appears in train. So the test singletons are like the train singletons (about 30% High), and any CV that includes the duplicate rows overstates what duplicate-aware features will do on the leaderboard by about 0.015.
- Within train the `uri` is not random: all copies of a track share it. The randomness is only across train and test.
- **The same-date effect is a release-day cohort effect, not an album effect.** Label agreement of same-date pairs is 0.658 versus 0.582 for random same-year pairs, but it is flat across audio-distance quintiles (0.687, 0.648, 0.625, 0.652, 0.678), so "same date and similar audio" does not identify albums any better than the date alone.
- Weekday after 2015: Sunday releases 5.5% High (n 110), Wednesday 13.4% (217), Monday/Tuesday 21-22%, Thursday/Friday/Saturday 31%. Year-only dates (111 rows) are 59% High; January 1 dates 46%.
- Era is the dominant axis: the singleton OOF AUC is 0.885 before 2000, 0.81 for 2000-2009, 0.80 for 2010-2015, 0.71 for 2016-2019, 0.75 for 2020-2021, 0.79 for 2022-2023, 0.84 for 2024.
- Target-derived features saturate fast at this size: one date encoding plus one kNN feature helps, two date encodings or four kNN features hurt by 0.002-0.003, and CatBoost's native categorical on the date string loses 0.04.

## Files

- `pipe.py` (loader, features, cross-fitted TE, CV runner), `exp1.py` to `exp8.py` (experiments in order), `blend.py`, `results.csv` (all runs), `blend_results.csv`, `oof*.pkl` (out-of-fold predictions per seed), `eda1.py` to `eda4.py`.
