# Fresh look at CS 155 Miniproject 1 (agent A, independent pass)

Date: 2026-10-30. All work in this scratchpad directory; nothing written in the project. Protocol: `artifacts/splits/production_v1.csv`, 3 seeds x 5 grouped stratified folds, eligible rows only (3,863 rows, the empty row 425 dropped). Score = mean of 15 fold AUCs. Every target-derived feature is fitted inside the training part of each fold and cross-fitted (5 inner GroupKFold splits) for the training rows. Models run with 2 threads, one fit at a time.

Two metrics are reported for every run:

- `mean15`: the mandated score (all eligible validation rows).
- `singleton15`: the same 15 folds, AUC restricted to validation rows whose exact (13 audio features + date string) key occurs once in train+test. This matters because the test set is 97.5% such rows (841 of 863) while train is 88.8% (3,431 of 3,863); see section (d).

## (a) Results table

Baseline `lgb_base`: LightGBM (400 trees, lr 0.03, 15 leaves, min_child_samples 20, subsample 0.8, colsample 0.8, lambda 1) on the 13 audio features + year, month, day, day-of-year, weekday, date precision, days-since-epoch. Deltas are against that baseline. `cnt` = `cnt_exact`, `datecnt` = `date_cnt_all`, `teDate` = cross-fitted date-string target encoding, `knn20` = cross-fitted 20-nearest-neighbour label mean (definitions in (b)).

