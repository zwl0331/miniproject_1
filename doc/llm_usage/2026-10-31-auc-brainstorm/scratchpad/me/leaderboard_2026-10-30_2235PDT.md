# Public leaderboard, read 2026-10-30 about 22:35 PDT (page is public, no login)
Public board = approximately 40% of test data (about 345 of 863 rows); private = other 60% (about 518 rows).
rank team score entries last
1 Bechtel 221 0.87092 6 2d
2 NullPointerException 0.87024 4 4m
3 Burgers 0.86729 13 1h
4 Yisong's Yues 0.86199 13 7h
5 (Chinese name) 0.86136 5 1d
6 AMO 0.85691 24 1h
7 Elon, Sam, Dario 0.85403 3 7m
8 LOATS 0.84699 10 1d
9 Weilong (Sam) Zhou 0.84343 4 5m   <- Sam; 4 entries; last submission 5 minutes before the read
10 Everything is linear 0.84059 5 1h
11 0.82648 1; 12 0.80482 1; 13 bechtel 214 0.80058 2; 14 Panda Express 0.71448 1
Notes: submissions/README.md logs only the first upload (0.83597); the 0.84343 and the other two uploads are not recorded in the project.
Project best grouped CV: B24 = tuned LightGBM + date_count + date_te3, 0.8367 (per seed 0.8386, 0.8368, 0.8346).

# Deployment check (me/refit_vs_foldavg.py, 2026-10-30 ~23:05 PDT), recipe L+ = tuned LightGBM F2 + date_count + date_te3, 2 threads
Within each outer fold: one model on the whole outer training part ("full") vs the average of 5 inner-fold models on 80% each ("fold_avg", the project's deployment mirrored one level down).
  full 0.8354 (0.8362, 0.8363, 0.8336); full 750 trees 0.8356; full 3 seeds 0.8360 (0.8368, 0.8365, 0.8348); fold_avg 0.8373 (0.8390, 0.8373, 0.8355)
  full minus fold_avg -0.0019 (-0.0028, -0.0009, -0.0019), 2/15 folds positive; 3-seed full minus fold_avg -0.0012 (6/15).
Reading: the fold average is a bag; it beats one model trained on 25% more rows. The project's fold_average deployment is right; a full refit would likely lose ~0.002. Bagging looks like a lever worth measuring (next script).
Note: my single-model L+ scores 0.8354 vs the project's 0.8367 for the same recipe; the difference is the TE inner-split seed and 2 vs 4 threads, i.e. ~0.001 of pure noise from the cross-fit realization.

# Agent B (structure forensics) key numbers, received ~23:40 PDT; full outputs in scratchpad/agentB_structure/*.out
- Test construction: 3864+863+103 = 4830; 3864/4830 = 0.8000. An 80/20 row split, then the 103 test rows whose uri occurred in train were deleted. Test = singleton-type tracks + 11 within-test duplicate pairs. Expected test High rate ~0.295.
- Duplicate-uri rows in train: 11% of rows, 79% High (vs 30% for singletons). OOF AUC all rows 0.8331/0.8273/0.8272 vs singleton rows 0.8263/0.8216/0.8232 (probe model): full-population CV overstates test-like AUC by ~0.005. Team's B24 0.8367 -> test-like ~0.832.
- Weekday: +0.0055 if absent (it is in F2). duration_ms % 1000 == 0: +0.0008 (6% of rows, High 0.22 vs 0.35).
- Within-test uri copies (n_copies in the same file): 11 pairs, 22 rows, expected ~0.77 High; pushing them up = +0.003 (sd 0.0025) on test. Transductive; needs Sam's rule call.
- Album-mate: same-date different-group pairs agree 0.629 vs 0.599 null; closest decile 0.741 vs 0.615; d<1 (2.4% of rows) 0.92-0.97. As a feature +0.003, redundant with date_te3 (both together +0.0002). Nearest-mate label alone +0.0007: do not build.
- LB noise: SE of one AUC on 345 rows 0.024; paired difference SE 0.013 (corr 0.9), 0.007 (0.97). Best-of-k inflation: k=4 +0.010..0.014; k=13 +0.016..0.021. The 0.025 gap: likely real 0.012 (to 0.861) to ~0.02 (to 0.87), +-0.013..0.025. Team's 0.843 is best-of-4, ~0.01 above truth.
- Nulls: ID gaps, YYYY precision within era, key/mode, loudness decimals, conflicting duplicates.

# Agent A (fresh eyes) key numbers, received ~23:45 PDT; report in scratchpad/agentA_fresh/REPORT.md
- Independently found: duplicate-count feature (train+test 14-tuple count) +0.019 on full CV, unchanged on singletons, ~+0.004 test-like; exact-date TE m=3 + count +0.0019; 20-NN label mean (audio+year) +0.0007; rank-average blend LightGBM + 0.5 ExtraTrees(500, leaf 3, max_features 0.5) +0.0023; 5-seed bagging +0.001. Combined 0.8550 full CV but singleton 0.8308 (from 0.8273); test-like ~0.837 vs 0.835.
- CatBoost with the raw date string as a native categorical collapsed to 0.8118. Hyperparameter sweeps, audio ratios, year-relative loudness, XGB/EBM/CatBoost: within seed spread.
- Weak era: 2016-2019 singleton OOF AUC 0.709 vs 0.885 pre-2000 and 0.839 for 2024.
- Suggests pseudo-labelling (+0.001..0.003, speculative), monotone constraints, stacking with era input.

# Re-scoring of all 576 phase 2 banks on singleton validation rows (me/rescore_singletons.py, ~23:50 PDT)
- Spearman(all-rows score, singleton score) = 0.993 over 576 banks: ranking robust. B24 top on both: all 0.8367, singleton 0.8335 (0.8356, 0.8324, 0.8326). Anchor 0.8315 -> 0.8279. L (b3_001) 0.8339 -> 0.8312.
- Gap all minus singleton by family: lightgbm 0.0029, catboost 0.0043, ebm 0.0048, hist_gb 0.0038, xgboost 0.0030, extra_trees 0.0021, random_forest 0.0010, logistic -0.0003, knn -0.0032.
- P2.4 re-tuning is producing runs (p2_p24_rt_*): best so far 0.8368 all rows / 0.8335 singleton, i.e. B24 within noise.

# Fixed blends of stored banks on singleton rows (me/blends_singletons.py, ~23:58 PDT); B24 alone: all 0.8367, singleton 0.8335
- B24 + 0.2 ET_anchor + 0.1 kNN_anchor + 0.1 CatBoost X2_04 (prob): all 0.8397 (+0.0030), singleton 0.8368 (+0.0033; +0.0029, +0.0035, +0.0034)
- B24 + 0.25 ET_anchor + 0.15 kNN_anchor (prob): singleton +0.0032 (all seeds +0.0028..+0.0034)
- B24 + 0.4 ET_b3_001 (prob): singleton +0.0031; B24 + 0.25 ET_anchor: +0.0023 (prob) / +0.0024 (rank)
- kNN_anchor at 0.25: all +0.0006, singleton +0.0017 (kNN complements more on test-like rows). CatBoost X2_04 alone: all 0.8353 but singleton 0.8298 (its gain was duplicate-driven).
- Rank vs probability averaging: within 0.0003 of each other; nothing to gain there.

# Era-aware recalibration of B24's OOF (diagnostic, me/era_recal.log, ~00:02 PDT 10-31): all rows -0.0019, singletons -0.0024, negative on every seed. Era does not belong in the stacker. Within-era fold AUC: 1990s 0.786, 2000s 0.818, 2010-15 0.819, 2016-19 0.734, 2020-22 0.757, 2023-24 0.834.
# Inverse-group-size weights (P2.8 B9) on singleton rows: 0.8266 vs anchor 0.8279 (-0.0013; all rows -0.0017). Re-weighting toward the test population does not help.
# Within-test duplicate pairs under B24's deployment output (me/test_duplicates.log, ~00:10 PDT 10-31): 22 rows, mean prediction 0.587 vs 0.289 for the others; percentile ranks 0.26,0.26,0.36,0.36,0.69..0.99; 4 of 22 below the median; 73% above the 70th percentile. A rank boost would move ~4-8 rows: expected test gain well under the agents' +0.003, and it is a transductive use of test.csv that needs Sam's rule call. Years of those rows: 1984, 1988, 2000, 2014, 2016 x2, 2019, 2022 x2, 2023, 2024.

# Bagging strength (me/bagging_strength.py, done ~00:18 PDT 10-31), recipe L+, same 15 outer folds; per seed 155, 2026, 1029
  full (one model on the outer training part)       0.8354  0.8362, 0.8363, 0.8336
  te3avg (3 TE cross-fit seeds, predictions averaged) 0.8362  +0.0008 (12/15 folds)
  fold_avg5 (5 inner-fold models on 80% each)         0.8373  +0.0019 (+0.0028, +0.0009, +0.0019; 13/15)
  fold_avg10 (two inner realizations)                 0.8374  +0.0020; vs fold_avg5 +0.0001 (saturated)
  bag10 (10 random 80% group subsamples)              0.8369  +0.0016
Reading: the project's fold_average deployment already banks this ~+0.002; nothing further to gain from more bags. CV single-model scores understate the deployed model by ~0.002. A full refit for deployment would lose it.
# decisions.md new row (2026-10-30): reviewer set to Claude Fable 5.1 xhigh in .agent/review.conf; Codex consultation unchanged.
