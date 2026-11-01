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
| Historical equal blend E | 0.841062 | 0.842493 | 0.841728 | 0.838964 | 0 | existing |
| Historical constrained_l0.1 | 0.841003 | 0.842426 | 0.841618 | 0.838964 | -0.000059 | existing |

## Checkpoint

2026-11-01 04:20 UTC: P2.9 saved on `p2.9-paused` at 1b47c17; main pending
records committed at 8c03e05. The old suite has exited without a final verdict.
No further P2.9 work. Sam confirmed Codex owns development and the other session
is paused. No new experiment result yet.
