# Improving the AUC: an independent look (Claude Fable 5.1, 2026-10-31, 00:30 PDT)

Read-only session. Nothing under the project root was created, modified or deleted; every number below comes
from scripts and outputs in this scratchpad (`me/`, `agentA_fresh/`, `agentB_structure/`, `agentC_mechanics/`).
Three fresh sub-agents worked independently: A built its own pipeline knowing nothing of the project's results;
B did data forensics with null comparisons; C ran matched experiments on mechanisms the project never tested.
All scores are mean fold AUC on the production folds (3 split seeds x 5 grouped folds) unless marked.

## 1. The leaderboard gap is smaller than it looks, but part of it is probably real

Public board, read at 22:35 PDT (the page is public): calculated on about 40% of the test rows (about 345).

| rank | score | entries | | rank | score | entries |
|---|---|---|---|---|---|---|
| 1 | 0.87092 | 6 | | 6 | 0.85691 | 24 |
| 2 | 0.87024 | 4 | | 7 | 0.85403 | 3 |
| 3 | 0.86729 | 13 | | 8 | 0.84699 | 10 |
| 4 | 0.86199 | 13 | | 9 (Sam) | 0.84343 | 4 |
| 5 | 0.86136 | 5 | | 10 | 0.84059 | 5 |

- Standard error of one AUC near 0.84 on 345 rows: 0.024 (Hanley-McNeil; 0.021 to 0.022 by bootstrap on real OOF
  predictions). Paired difference between two teams' models on the same rows: SE 0.013 at prediction correlation 0.9,
  0.018 at 0.8.
- Best-of-k selection on the public board: +0.010 to +0.014 at k = 4, +0.016 to +0.021 at k = 13. Sam's 0.843 is a
  best-of-4 and probably sits about 0.01 above its true public-subset value.
- Simulation (Agent B): with equal true AUC, P(observed gap >= 0.025 between a 4-entry and a 13-entry team) is 0.01 to
  0.10. Most likely real gap: about 0.012 to the 0.861 teams and 0.019 to 0.022 to the 0.87 teams, +-0.013 to 0.025.
- The private board is the other 60% (about 518 rows), SE 0.020 per AUC. Nothing below a 0.012 public improvement
  over the current submission should be believed.

## 2. What the test file is, and what it does to the CV numbers

- 3,864 + 863 + 103 = 4,830 rows and 3,864 / 4,830 = 0.8000 exactly: an 80/20 row split, then every test row whose
  track (uri) also occurred in train was deleted (103 rows; the ID gaps). Four observed counts match this simulation
  (train repeated uris 203, rows in them 424, removed 103, test repeated uris 11). The missing-ID positions are noise.
- So the test is singleton-type tracks plus 11 within-test duplicate pairs (22 rows). Expected test High rate about
  0.295, not 0.352.
- Duplicated tracks in train are 11% of rows and 79% High. They are easy to rank, so the all-rows CV overstates a
  test-like AUC. Re-scoring all 576 phase 2 banks on singleton validation rows (3,431 rows): B24 0.8367 -> 0.8335;
  anchor 0.8315 -> 0.8279. Gap by family: LightGBM 0.0029, XGBoost 0.0030, HistGB 0.0038, CatBoost 0.0043, EBM
  0.0048, Extra Trees 0.0021, random forest 0.0010, logistic -0.0003, kNN -0.0032 (kNN does relatively better on
  test-like rows).
- The ranking is robust: Spearman 0.993 between the two scores; B24 stays first. The P2.4 re-tuning's best run so far
  (0.8368) equals B24 within noise. Recommendation: keep the official metric, add the singleton score as a reported
  diagnostic, and treat CatBoost's 0.8353 (X2_04, singleton 0.8298) as duplicate-driven.

## 3. Mechanisms tested tonight, matched on the production folds

| mechanism | result | verdict |
|---|---|---|
| LightGBM histogram resolution: max_bin 1023, 4095, min_data_in_bin 1, days-only high bins | -0.0003 to +0.0009, mixed signs | null |
| min_child_samples 3, 5 (vs 10) | +0.0012 on L at seed 0, shrinks across seeds; +0.0000 to +0.0006 on L+ | at most 0.0005 |
| exact date as LightGBM categorical (cat_smooth 10, 30) | +0.0001 to +0.0010 mixed; cat_smooth 1 collapses (-0.02) | null |
| CatBoost with the raw date string as native categorical | 0.8320 (C) or 0.8118 (A, different config); both below L+ | null |
| within-date audio nearest-neighbour block (label, distance, kernel mean, count), cross-fitted | -0.0021 on L, -0.0025 on L+, non-positive on every seed | hurts |
| date TE smoothing below 3 (m = 1, 0.5, unsmoothed + count, per-group labels) | -0.0006 to +0.0003 | null |
| estimator-seed averaging of L+ (3 or 5 seeds) | +0.0003 to +0.0006, mixed | at most 0.0005 |
| blend geometry (probability vs rank vs logit) | within 0.0003 to 0.0009 of each other | nothing |
| Extra Trees partner (500 trees, leaf 2, all features) blended with L+ / B24 | +0.0025 to +0.0032 on all rows, +0.0023 to +0.0031 on singletons, every seed positive | real |
| L+ blended 0.5/0.5 with plain L (no cross-fitted blocks) | +0.0023 (+0.0025, +0.0008, +0.0035) | real, cheap |
| kNN anchor as a third partner (B24 + 0.25 ET + 0.15 kNN) | singleton +0.0032 vs +0.0023 for ET alone; kNN alone +0.0017 singleton, +0.0006 all rows | small, real on test-like rows |
| B24 + 0.2 ET + 0.1 kNN + 0.1 CatBoost X2_04 | all rows 0.8397 (+0.0030), singletons 0.8368 (+0.0033; +0.0029, +0.0035, +0.0034) | best fixed blend found |
| full refit on the whole training part vs the average of 5 fold models (deployment check, inside CV) | fold average +0.0019 over one model (13 of 15 folds); 10 models +0.0001 more; subsample bags +0.0016; TE cross-fit seed averaging alone +0.0008 | keep `fold_average`; already banked in deployment; a full refit would lose about 0.002 |
| era-aware recalibration of B24's scores (diagnostic) | -0.0019 all rows, -0.0024 singletons | era does not belong in the stacker |
| inverse-group-size weights on singleton rows | -0.0013 (all rows -0.0017) | null |
| within-test duplicate pairs ranked to the top (transductive) | B24 already gives them mean 0.587 vs 0.289, 4 of 22 below the median | well under +0.003; needs a rule call |
| `duration_ms % 1000 == 0` flag | +0.0008 +- 0.0005 (6% of rows, High 0.22 vs 0.35) | tiny, consistent |
| weekday | +0.0055 if absent; it is in F2 | already captured |
| 20-NN label mean in standardized audio + year (Agent A, cross-fitted) | +0.0007 on all seeds, +0.001 singletons | tiny |
| same-date Gaussian-weighted mate mean (Agent B) | +0.0028 alone, +0.0002 on top of date_te3 | redundant with date_te3 |
| year-month TE on top of date TE; two date encodings; four kNN variants | -0.002 to -0.003 | hurt |
| noise floor | the TE inner cross-fit seed alone moves a per-seed mean by up to 0.002 (mean unchanged) | differences under 0.001 per seed are not evidence |

## 4. Album-mate structure, settled

Same-date different-track pairs agree on the label 0.629 vs 0.599 for year-matched pairs; 0.741 vs 0.615 in the
closest audio decile; 0.92 to 0.97 for the 2.4% of rows with a mate at distance < 1. The excess is both-High pairs.
As a feature it is worth about +0.003 and the exact-date target encoding already has it (together +0.0002); single
nearest neighbours add noise. Coverage: 72% of test rows have a same-date train mate vs 69% of CV validation rows
(29% vs 25% within the close decile), so CV does not overstate it.

## 5. Recommendations, ranked by expected gain under the course rules

1. Blend B24 (or the re-tuning's best) with the Extra Trees anchor at 0.25 to 0.4: +0.003 on all rows and on
   test-like rows, every seed. P2.6b's pool already plans this; it is the one solid gain.
2. Add the kNN anchor (and the best CatBoost with date blocks) as small extra channels: +0.0005 to +0.001 more,
   mostly on test-like rows. Also consider plain L (no cross-fitted blocks) as a within-family channel even though
   it is 0.0028 below B24: its 0.5/0.5 blend with L+ gave +0.0023 because the cross-fitted date columns add variance
   that a second model averages out.
3. Keep `fold_average` deployment (15 fold models). The CV single-model numbers understate it by about 0.002 on the
   test; a full refit would give that back. Seed averaging adds at most 0.0005.
4. Report the singleton-row score beside the official one in the final selection; rank by the official metric
   (robust), but prefer the candidate whose singleton score is not lower when two are tied within 0.001.
5. Treat differences under 0.001 per seed as ties (the cross-fit noise floor). The P2.4 re-tuning's 64 configurations
   around B24 are unlikely to separate from it.
6. Expected result of 1 to 3: about 0.840 to 0.841 all-rows CV (0.837 singletons), roughly 0.845 to 0.85 on the public
   board after the deployment bonus and the subset offset seen so far (CV 0.8339 -> public 0.843).
7. Rule-sensitive, low value: a rank boost for the 22 within-test duplicate rows (B24 already ranks most of them
   high); pseudo-labelling of test rows (+0.001 to +0.003, speculative). Both are transductive uses of test.csv and
   need Sam's call and a line in the report.
8. Not worth building: histogram resolution, categorical dates, same-date neighbour features, smoothing below 3,
   era-aware stacking, inverse-group weights, year-month encodings.

## 6. What the 0.86 to 0.87 teams may have

Three independent looks found no unused legitimate signal of the size of the remaining 0.01 to 0.02. The remaining
explanations are model class (a pretrained tabular model of the TabPFN kind typically adds +0.01 to +0.02 on 4,000
rows and is excluded by Sam's ruling of 2026-10-29), a neural ensemble (P2.5's TabM will show what that is worth
here), public-board selection, or something not in the files.

## 7. Record-keeping notes (no action taken)

- The board shows 4 entries for Sam; the project logs two uploads (0.83597, 0.84343; the second is now in STATE.md).
- Agents A and C disagree on CatBoost with a categorical date (0.8118 vs 0.8320); different depths and extra blocks;
  irrelevant since both are below L+.

## Files

`me/`: `leaderboard_2026-10-30_2235PDT.md` (running notes with every number), `rescore_singletons.py/.csv/.log`,
`blends_singletons.py/.csv/.log`, `refit_vs_foldavg.py/.log`, `bagging_strength.py/.log`, `era_recal.log`,
`test_duplicates.log`. `agentA_fresh/REPORT.md` (results.csv, pipe.py, exp1 to exp8, blend.py).
`agentB_structure/*.out` (q1 to q7 scripts; its report is in its hand-back message, copied in the notes).
`agentC_mechanics/REPORT.md` (pipe.py, q1 to q4, results.jsonl, oof/).
