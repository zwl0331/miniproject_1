# Fast exploration, 2026-11-01

Owner-directed quick development. Codex primary. All new results are exploratory.
Rank by the mean of the 15 saved grouped-fold AUCs, not public scores. Split seeds
are 155, 2026 and 1029. A difference below about 0.002 is noise for selection.
New experiments stop at 2026-11-01 20:00 UTC. Files due 2026-11-02 20:00 UTC.

The first trials add duration-rounding information, hierarchical date statistics,
date-frequency summaries and audio transforms to the strongest earlier LightGBM
recipe. The brainstorm's previous negative results guide a small set of checks;
none of its test-duplicate or transductive proposals is adopted.

Every result saves the full config, OOF probabilities, 15 test-fold predictions,
and runtime under `artifacts/fast/`. Fitted encoders use only their training
population. New candidates are compared with fixed additions at weights 0.10,
0.25 and 0.50 to E and to the existing constrained_l0.1 predictions. This reuses
the existing properly cross-fitted constrained procedure; it does not fit blend
weights on validation labels. Report every weight, even when it loses.

## Results

| Model or feature change | CV mean | Seed 155 | Seed 2026 | Seed 1029 | Change vs E | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Historical blend_eq_all | 0.841062 | 0.842493 | 0.841728 | 0.838964 | +0.000000 | existing |
| Historical constrained_l0.1 | 0.841003 | 0.842426 | 0.841618 | 0.838964 | -0.000059 | existing |
| lgb_audio_extra | 0.835668 | 0.837419 | 0.836985 | 0.832601 | -0.005393 | 41.4 s |
| blend_eq_all + 0.10 lgb_audio_extra | 0.841095 | 0.842606 | 0.841787 | 0.838891 | +0.000033 | 41.4 s |
| blend_eq_all + 0.25 lgb_audio_extra | 0.840835 | 0.842360 | 0.841562 | 0.838585 | -0.000226 | 41.4 s |
| blend_eq_all + 0.50 lgb_audio_extra | 0.839798 | 0.841400 | 0.840662 | 0.837330 | -0.001264 | 41.4 s |
| constrained_l0.1 + 0.10 lgb_audio_extra | 0.840982 | 0.842474 | 0.841658 | 0.838815 | -0.000079 | 41.4 s |
| constrained_l0.1 + 0.25 lgb_audio_extra | 0.840767 | 0.842269 | 0.841513 | 0.838520 | -0.000294 | 41.4 s |
| constrained_l0.1 + 0.50 lgb_audio_extra | 0.839791 | 0.841406 | 0.840611 | 0.837357 | -0.001270 | 41.4 s |
| lgb_control | 0.836758 | 0.838972 | 0.836402 | 0.834901 | -0.004303 | 30.7 s |
| blend_eq_all + 0.10 lgb_control | 0.841054 | 0.842532 | 0.841679 | 0.838950 | -0.000008 | 30.7 s |
| blend_eq_all + 0.25 lgb_control | 0.840899 | 0.842372 | 0.841235 | 0.839090 | -0.000163 | 30.7 s |
| blend_eq_all + 0.50 lgb_control | 0.840040 | 0.841865 | 0.840124 | 0.838129 | -0.001022 | 30.7 s |
| constrained_l0.1 + 0.10 lgb_control | 0.841030 | 0.842464 | 0.841588 | 0.839037 | -0.000032 | 30.7 s |
| constrained_l0.1 + 0.25 lgb_control | 0.840862 | 0.842341 | 0.841175 | 0.839071 | -0.000199 | 30.7 s |
| constrained_l0.1 + 0.50 lgb_control | 0.840080 | 0.841877 | 0.840127 | 0.838235 | -0.000982 | 30.7 s |
| lgb_cyclic_music_round | 0.836334 | 0.838248 | 0.836982 | 0.833771 | -0.004728 | 42.3 s |
| blend_eq_all + 0.10 lgb_cyclic_music_round | 0.841204 | 0.842595 | 0.841990 | 0.839027 | +0.000142 | 42.3 s |
| blend_eq_all + 0.25 lgb_cyclic_music_round | 0.841061 | 0.842534 | 0.841806 | 0.838841 | -0.000001 | 42.3 s |
| blend_eq_all + 0.50 lgb_cyclic_music_round | 0.840280 | 0.841897 | 0.840834 | 0.838109 | -0.000782 | 42.3 s |
| constrained_l0.1 + 0.10 lgb_cyclic_music_round | 0.841135 | 0.842436 | 0.841979 | 0.838992 | +0.000074 | 42.3 s |
| constrained_l0.1 + 0.25 lgb_cyclic_music_round | 0.841075 | 0.842530 | 0.841805 | 0.838890 | +0.000013 | 42.3 s |
| constrained_l0.1 + 0.50 lgb_cyclic_music_round | 0.840261 | 0.841862 | 0.840860 | 0.838062 | -0.000800 | 42.3 s |
| lgb_date_counts | 0.837109 | 0.838454 | 0.836314 | 0.836559 | -0.003952 | 34.6 s |
| blend_eq_all + 0.10 lgb_date_counts | 0.841074 | 0.842479 | 0.841612 | 0.839131 | +0.000012 | 34.6 s |
| blend_eq_all + 0.25 lgb_date_counts | 0.840938 | 0.842218 | 0.841210 | 0.839385 | -0.000124 | 34.6 s |
| blend_eq_all + 0.50 lgb_date_counts | 0.840263 | 0.841498 | 0.840080 | 0.839212 | -0.000798 | 34.6 s |
| constrained_l0.1 + 0.10 lgb_date_counts | 0.841056 | 0.842407 | 0.841592 | 0.839169 | -0.000006 | 34.6 s |
| constrained_l0.1 + 0.25 lgb_date_counts | 0.840912 | 0.842154 | 0.841157 | 0.839423 | -0.000150 | 34.6 s |
| constrained_l0.1 + 0.50 lgb_date_counts | 0.840243 | 0.841482 | 0.840039 | 0.839208 | -0.000819 | 34.6 s |
| lgb_hier10 | 0.832001 | 0.834620 | 0.833316 | 0.828067 | -0.009061 | 41.3 s |
| blend_eq_all + 0.10 lgb_hier10 | 0.840867 | 0.842422 | 0.841597 | 0.838582 | -0.000195 | 41.3 s |
| blend_eq_all + 0.25 lgb_hier10 | 0.840288 | 0.842027 | 0.841006 | 0.837832 | -0.000773 | 41.3 s |
| blend_eq_all + 0.50 lgb_hier10 | 0.838624 | 0.840555 | 0.839670 | 0.835647 | -0.002437 | 41.3 s |
| constrained_l0.1 + 0.10 lgb_hier10 | 0.840811 | 0.842341 | 0.841549 | 0.838544 | -0.000250 | 41.3 s |
| constrained_l0.1 + 0.25 lgb_hier10 | 0.840232 | 0.841986 | 0.840935 | 0.837775 | -0.000830 | 41.3 s |
| constrained_l0.1 + 0.50 lgb_hier10 | 0.838637 | 0.840576 | 0.839591 | 0.835743 | -0.002425 | 41.3 s |
| lgb_hier3 | 0.833138 | 0.835941 | 0.834087 | 0.829387 | -0.007923 | 45.3 s |
| blend_eq_all + 0.10 lgb_hier3 | 0.841057 | 0.842557 | 0.841796 | 0.838818 | -0.000005 | 45.3 s |
| blend_eq_all + 0.25 lgb_hier3 | 0.840682 | 0.842292 | 0.841327 | 0.838428 | -0.000379 | 45.3 s |
| blend_eq_all + 0.50 lgb_hier3 | 0.839179 | 0.840973 | 0.840130 | 0.836433 | -0.001883 | 45.3 s |
| constrained_l0.1 + 0.10 lgb_hier3 | 0.841026 | 0.842456 | 0.841812 | 0.838811 | -0.000035 | 45.3 s |
| constrained_l0.1 + 0.25 lgb_hier3 | 0.840610 | 0.842224 | 0.841268 | 0.838339 | -0.000451 | 45.3 s |
| constrained_l0.1 + 0.50 lgb_hier3 | 0.839217 | 0.840983 | 0.840120 | 0.836548 | -0.001845 | 45.3 s |
| lgb_hier_replace | 0.832902 | 0.835144 | 0.833370 | 0.830191 | -0.008160 | 37.3 s |
| blend_eq_all + 0.10 lgb_hier_replace | 0.841036 | 0.842485 | 0.841762 | 0.838862 | -0.000025 | 37.3 s |
| blend_eq_all + 0.25 lgb_hier_replace | 0.840630 | 0.842049 | 0.841208 | 0.838633 | -0.000432 | 37.3 s |
| blend_eq_all + 0.50 lgb_hier_replace | 0.839101 | 0.840795 | 0.839811 | 0.836696 | -0.001961 | 37.3 s |
| constrained_l0.1 + 0.10 lgb_hier_replace | 0.840989 | 0.842375 | 0.841737 | 0.838856 | -0.000072 | 37.3 s |
| constrained_l0.1 + 0.25 lgb_hier_replace | 0.840573 | 0.841999 | 0.841149 | 0.838570 | -0.000489 | 37.3 s |
| constrained_l0.1 + 0.50 lgb_hier_replace | 0.839053 | 0.840685 | 0.839744 | 0.836731 | -0.002008 | 37.3 s |
| lgb_round | 0.835610 | 0.837327 | 0.836337 | 0.833166 | -0.005451 | 34.5 s |
| blend_eq_all + 0.10 lgb_round | 0.840900 | 0.842304 | 0.841647 | 0.838749 | -0.000162 | 34.5 s |
| blend_eq_all + 0.25 lgb_round | 0.840695 | 0.842113 | 0.841438 | 0.838535 | -0.000366 | 34.5 s |
| blend_eq_all + 0.50 lgb_round | 0.839506 | 0.840955 | 0.840200 | 0.837363 | -0.001556 | 34.5 s |
| constrained_l0.1 + 0.10 lgb_round | 0.840846 | 0.842251 | 0.841556 | 0.838731 | -0.000215 | 34.5 s |
| constrained_l0.1 + 0.25 lgb_round | 0.840590 | 0.841985 | 0.841338 | 0.838447 | -0.000472 | 34.5 s |
| constrained_l0.1 + 0.50 lgb_round | 0.839529 | 0.840983 | 0.840194 | 0.837411 | -0.001532 | 34.5 s |
| lgb_round_detail | 0.834505 | 0.836361 | 0.835260 | 0.831894 | -0.006557 | 34.8 s |
| blend_eq_all + 0.10 lgb_round_detail | 0.840966 | 0.842301 | 0.841623 | 0.838973 | -0.000096 | 34.8 s |
| blend_eq_all + 0.25 lgb_round_detail | 0.840658 | 0.841918 | 0.841367 | 0.838689 | -0.000403 | 34.8 s |
| blend_eq_all + 0.50 lgb_round_detail | 0.839360 | 0.840622 | 0.840209 | 0.837250 | -0.001701 | 34.8 s |
| constrained_l0.1 + 0.10 lgb_round_detail | 0.840889 | 0.842216 | 0.841560 | 0.838893 | -0.000172 | 34.8 s |
| constrained_l0.1 + 0.25 lgb_round_detail | 0.840618 | 0.841872 | 0.841310 | 0.838670 | -0.000444 | 34.8 s |
| constrained_l0.1 + 0.50 lgb_round_detail | 0.839352 | 0.840613 | 0.840211 | 0.837233 | -0.001709 | 34.8 s |
| lgb_round_leaf10_bins1023 | 0.836488 | 0.838959 | 0.836322 | 0.834184 | -0.004573 | 35.1 s |
| blend_eq_all + 0.10 lgb_round_leaf10_bins1023 | 0.841052 | 0.842576 | 0.841630 | 0.838949 | -0.000010 | 35.1 s |
| blend_eq_all + 0.25 lgb_round_leaf10_bins1023 | 0.840919 | 0.842678 | 0.841286 | 0.838793 | -0.000143 | 35.1 s |
| blend_eq_all + 0.50 lgb_round_leaf10_bins1023 | 0.840054 | 0.841994 | 0.840161 | 0.838008 | -0.001007 | 35.1 s |
| constrained_l0.1 + 0.10 lgb_round_leaf10_bins1023 | 0.840958 | 0.842454 | 0.841518 | 0.838902 | -0.000104 | 35.1 s |
| constrained_l0.1 + 0.25 lgb_round_leaf10_bins1023 | 0.840857 | 0.842574 | 0.841195 | 0.838803 | -0.000204 | 35.1 s |
| constrained_l0.1 + 0.50 lgb_round_leaf10_bins1023 | 0.840048 | 0.841961 | 0.840135 | 0.838048 | -0.001014 | 35.1 s |
| lgb_round_leaf3_bins255 | 0.836267 | 0.837329 | 0.837480 | 0.833991 | -0.004795 | 41.9 s |
| blend_eq_all + 0.10 lgb_round_leaf3_bins255 | 0.841016 | 0.842442 | 0.841732 | 0.838874 | -0.000046 | 41.9 s |
| blend_eq_all + 0.25 lgb_round_leaf3_bins255 | 0.840738 | 0.842140 | 0.841536 | 0.838539 | -0.000324 | 41.9 s |
| blend_eq_all + 0.50 lgb_round_leaf3_bins255 | 0.839975 | 0.841147 | 0.840958 | 0.837821 | -0.001086 | 41.9 s |
| constrained_l0.1 + 0.10 lgb_round_leaf3_bins255 | 0.840997 | 0.842319 | 0.841736 | 0.838936 | -0.000065 | 41.9 s |
| constrained_l0.1 + 0.25 lgb_round_leaf3_bins255 | 0.840729 | 0.842028 | 0.841568 | 0.838590 | -0.000333 | 41.9 s |
| constrained_l0.1 + 0.50 lgb_round_leaf3_bins255 | 0.839976 | 0.841140 | 0.840978 | 0.837810 | -0.001086 | 41.9 s |
| lgb_year_relative | 0.838242 | 0.839200 | 0.839704 | 0.835822 | -0.002819 | 31.7 s |
| blend_eq_all + 0.10 lgb_year_relative | 0.841335 | 0.842582 | 0.842179 | 0.839244 | +0.000273 | 31.7 s |
| blend_eq_all + 0.25 lgb_year_relative | 0.841616 | 0.842801 | 0.842467 | 0.839579 | +0.000554 | 31.7 s |
| blend_eq_all + 0.50 lgb_year_relative | 0.841323 | 0.842421 | 0.842325 | 0.839224 | +0.000261 | 31.7 s |
| constrained_l0.1 + 0.10 lgb_year_relative | 0.841260 | 0.842535 | 0.842028 | 0.839216 | +0.000198 | 31.7 s |
| constrained_l0.1 + 0.25 lgb_year_relative | 0.841553 | 0.842644 | 0.842400 | 0.839614 | +0.000491 | 31.7 s |
| constrained_l0.1 + 0.50 lgb_year_relative | 0.841324 | 0.842402 | 0.842273 | 0.839297 | +0.000263 | 31.7 s |

Runtime is the model's complete 15-fold fit, shared by its blend rows.

## Checkpoint

2026-11-01 04:20 UTC: P2.9 saved on `p2.9-paused` at 1b47c17; main pending
records committed at 8c03e05. The old suite has exited without a final verdict.
No further P2.9 work. Sam confirmed Codex owns development and the other session
is paused. No new experiment result yet.
