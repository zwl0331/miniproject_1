# Claude's positions on P2.4, written before consultation round Q

2026-10-30, about 06:20 PDT. Author: Claude Code (Claude Opus 5.5). Written before the round Q prompt was sent, while P2.3 (CatBoost and EBM) was being prepared for its readiness review; P2.4 starts executing only after P2.3 converges. Sam, verbatim: "Okay, 2P2.3 and 2.4, and then pause after you finish."

## Scope

The feature stage of `doc/plans/phase2-models-features.md` section 6, as amended by section 16.3: 32 initial comparisons (the first 14 imported from P2.8 with their records and route flags, unchanged), expansion batches of 16 under section 5.6 up to 96 comparisons, then re-tuning of up to four model-and-feature combinations in two batches of 16 each (at most 128 configurations). Neural comparisons are omitted (P2.5 has not run). Saturday 08:00 PDT is the last batch start.

## The 18 new initial comparisons (declared before any P2.4 score)

| Group | Base | Addition | Count |
|---|---|---|---|
| Reserved (sections 6.5, 16.3) | P2.3's best CatBoost | `date_count`; `date_te30` with `date_count`; `music`; `interactions` | 4 |
| Reserved (16.3) | P2.3's best EBM | `music` (dropped with the reason if EBM is omitted) | 1 |
| Smoothing (6.1 item 1) | `p22_lightgbm_b3_001` | `date_count` with `date_te3`, with `date_te10`, with `date_te100` (B14 already holds `date_te30`) | 3 |
| Cross-family (6.1 item 1) | P2.2's best XGBoost, best Extra Trees, best logistic | `date_te30` with `date_count` | 3 |
| Combination (6.1 item 2) | B14's recipe | plus `music` | 1 |
| Removal (6.6) | B14's recipe | without `days`; without `day` and `day_of_year`; without `weekday`; without key, mode and time signature | 4 |
| Removal (6.6) | P2.3's best CatBoost with `date_te30` and `date_count` | without `days`; without key, mode and time signature | 2 |

Each challenger has a matched base: the identical recipe without the addition (or with the dropped columns kept), refitted under P2.4's sources when the sources change.

## Machinery

- New smoothing blocks `date_te3` and `date_te100` in `songpop/phase2/features.py` (the encoder is already parameterized by m).
- A P2.4 batch manifest saved before its scores: challengers, bases, matched pairs, and the allocation state of section 5.6 (H, family incumbents, G, previous best blends with H) from logged records, as P2.8 did. Runs through `scripts.run_experiments --recipe` (task P2.4). A read-only judge computes matched differences, contribution blends, routes, progress and the counter; P2.8's 14 are imported as records of the initial batch without re-judging.
- Expansion batches of 16: candidates enumerated by a declared priority rule from the logged results (6.1's order: more smoothing values on the strongest tuned models, combinations of blocks that met a route, preprocessing alternatives for logistic and kNN, interactions, cyclic and era-relative blocks on other tree families, training weights on promising recipes), so each expansion manifest is a function of earlier results.
- Re-tuning: up to four combinations chosen by section 5.6 (those meeting a route, the strongest first); for each, a local refinement batch of 16 around its recipe with the feature blocks fixed (the P2.2 refine code with a template that carries the blocks), and a second batch only if the first makes progress. The wave protocol keys families by name, so two LightGBM combinations need either separate lanes or one protocol per combination.
- Verification: `--verify --task P2.4` refits every P2.4 run; the judge's output reruns identically.

## Open points

- Whether to formalize the feature batches in the wave protocol (lanes keyed by combination rather than family) or keep P2.8's lighter manifest plus read-only judge.
- Whether the 13 non-reserved initial comparisons are the right ones, in particular the removal checks on B14 versus more smoothing and cross-family checks.
- How to keep CatBoost's re-tuning cost bounded if a CatBoost combination is chosen (Ordered boosting at depth 7 costs about 133 s per 15-fit run).
- Final estimator-seed policies stay in P2.6b.
