# Claude's positions on P2.8 as a whole (written before the round N prompt was sent)

Written 2026-10-30, about 03:45 PDT, after P2.2 converged and was committed (810d1ce, 72a822b). Sam's instructions, verbatim: "Yeah, don't assume anything. Use tests and actual data to draw conclusions." (2026-10-29), "Prioritise gap hunt" (2026-10-30, answering the proposal to run the identifier probe and the date-encoding comparisons right after P2.2, ahead of the broad CatBoost and TabM searches) and "Yes, start P2.8" (2026-10-30). Round M (Codex's independent design of the identifier probe) has been read; these positions say what I keep, change and add.

## Why the task exists

Sam's public score is 0.836; Sam reports others near 0.86. Measured levers so far add thousandths (P2.2's best LightGBM +0.0024 over the anchor, the best blend +0.0036). If a large missing signal exists in the supplied columns, the two places it would show are the identifiers or row order, and the exact release date. The task answers both with declared tests before the broad CatBoost and TabM work.

## Scope: one task, three parts, two checkpoints

| Part | Content | Machinery |
|---|---|---|
| A | Identifier and order probes: round M's audit, blocks, menu I1 to I8, thresholds, controls and checks, nearly unchanged | A standalone runner in `analysis/identifier_probe/`, reusing phase 1 loading, F2 features, folds, model factories, metrics and bank validation, and the reviewed phase 2 encoders (`DateCount`, `DateTargetEncoder` logic, inner group splits) generalized to fragment columns |
| B | Feature anchor checks pulled forward from P2.4: the nine anchor checks of plan section 6.3 against the anchor H, and section 6.5's two random-forest checks on P2.2's best random forest by the logged scores (`p22_random_forest_b1_0_040`, section 15.7) | The phase 2 engine as it is: recipes with blocks or training weights, `--recipe`, `--compare`, logged as task P2.8 |
| C | Deterministic forest prediction (Finding 013's follow-up), needed before B's forest runs | `FittedRecipe.predict_each` predicts random forest and Extra Trees with one thread, fitting unchanged; a check refits each P2.2 family anchor under the new code: the five non-forest anchors must equal their P2.2 banks in every array, the two forest anchors within 1e-12 in predictions |

Two checkpoints in one record, as in P2.2: readiness (plan section 16, code, self-checks, the A manifest and B recipes saved before any score, a score-free timing), then completion (results, findings, verification).

## Positions on part A (against round M)

- Keep: the structural audit; `id_shape_v1`, `id_frequency_v1`, `id_te30_v1`, L, the audio-recurrence block; I1 to I8; the follow-up thresholds (a block: mean gain at least 0.001, positive on every repeat; L: a declared blend gains at least 0.0005 on every repeat); 10,000 shared group-bootstrap draws, one-sided Bonferroni bounds at 0.05/8; pseudotoken controls with three salts for at most two candidates; the implementation checks; the legality table; null wording; integration only through a later reviewed `raw_columns_for` extension.
- H and R through the probe runner must reproduce phase 1: H's arrays equal the phase 1 LightGBM F2 bank exactly; R (with part C's one-thread prediction applied locally in the runner) within 1e-12 in predictions, its AUC difference reported. This is the probe's equivalent of P2.2's H check.
- Outputs in `artifacts/p2.8/identifier_probe/` (banks, results JSON), not the phase 2 log, because the probe banks are not phase 2 recipes. The declared menu, salts, thresholds and seeds in `configs/phase2/p2.8/identifier_probe.json`, saved before any score.
- I7 (row position): test rows get a missing position; ineligible for deployment; diagnostic only.
- The controls apply to identifier candidates (I1 to I6) only.

## Positions on part B

- The 11 comparisons are P2.4's first 11 of its 32 initial comparisons. P2.8 reports them (mean fold AUC difference, by repeat, paired cluster bootstrap with 2,000 draws, descriptive) and takes no allocation decision; P2.4's declared section 5.6 routes judge them when P2.4 runs.
- The random-forest comparator is refit under the new code (deterministic prediction), not taken from P2.2's bank, so each pair differs only in the block.
- Open question for round N: whether B should also include the strongest P2.2 LightGBM (`p22_lightgbm_b3_001`) with `date_te30` and `date_count` (P2.4 item 1, "on the strongest tuned models"), two runs, to see whether the date signal survives tuning. Leaning yes, as two declared extra comparisons counted toward P2.4's 32.

## Positions on part C

- The change touches prediction only. The P2.2 forest runs then replay at commit 810d1ce, like the P2.1 anchor. Every later forest run is reproducible exactly.
- P2.3's searches start from P2.8's approved source revision, so they use deterministic forests too (P2.3 has no forest family, but the phase 2 source manifest is shared).

## Cost and schedule

About 35 fitted evaluations of 15 fits each (A: about 9 plus at most 6 controls; B: 13; C: 7), roughly 15 to 25 minutes of compute; authoring and review dominate. Target: readiness review by about 10:00 PDT Friday, completion by about 14:00, ahead of Saturday 08:00's exploratory cutoff.

## Findings

014: identifier structure and deployment availability (audit). 015: identifier predictive probes. 016: the feature anchor checks. The part C check goes into 016 or its own finding. Findings 011 to 013 change to `reviewed` in this ledger revision.
