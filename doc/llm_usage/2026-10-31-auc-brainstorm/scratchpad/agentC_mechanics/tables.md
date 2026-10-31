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
