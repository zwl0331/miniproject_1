
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
