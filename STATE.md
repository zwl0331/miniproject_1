# State: CS 155 Miniproject 1

Updated 2026-11-01, 04:20 UTC. Fast exploration mode, owner-directed. Codex owns development; Sam confirmed the other session is paused. P2.9 is saved and stopped.

## Objective and rules

- Improve generalizing AUC with quick feature and model trials. Select on mean fold AUC over saved grouped splits, seeds 155, 2026, 1029. Treat gains below about 0.002 as noise and prefer simpler models in a tie.
- Only provided training data. No external data, pretrained models, Spotify lookups, dataset-specific external code, fitting on test features or pseudo-labels. Every fitted transform stays inside its training fold.
- Fast-mode override in `AGENTS.md`; full owner prompt: `doc/llm_usage/2026-11-01-fast-mode/prompt.md`. Exploration compute on this node is authorized. No P2.9 gates, consultation, replay, nested evaluation or coverage targets.
- New experiments stop 2026-11-01 20:00 UTC (12:00 PST). Files ready 2026-11-02 20:00 UTC (12:00 PST). Kaggle closes 2026-11-03 01:00 UTC (Monday 17:00 PST).
- Sam uploads, at most 10 per day, and decides the two final selections. Each file gets automatic ID/count/probability checks, correlation with the current best file, and a CV/hash/public-score log. One quick Fable review of the two owner-selected final files, leakage and file correctness only, one round.
- Report and public Colab follow development, due 2026-11-03 23:59 PST. Preserve brief LLM-use evidence.

## Baselines

| Candidate | Mean CV | Per seed: 155 / 2026 / 1029 | Public | Evidence |
|---|---:|---|---:|---|
| Current equal blend E | 0.841062 | 0.842493 / 0.841728 / 0.838964 | 0.85377 | `configs/phase2/menu.json`, `submissions/README.md` |
| Existing constrained blend, lambda 0.1 | 0.841003 | 0.842426 / 0.841618 / 0.838964 | 0.85415 | same menu, existing learned-procedure bank |
| Strongest earlier single-seed LightGBM | about 0.8368 | see P2.4 bank | not submitted alone | date_count + date_te3; Findings 022/023, draft |

P2.7 restricted nested result: selected procedure 0.8398 versus H 0.8306; saved replay passed. Historical evidence, still provisional. No nested reevaluation in fast mode.

## Saved work and environment

- Main pre-exploration snapshot: `8c03e05`, including pending records and the incomplete legacy-suite log.
- `p2.9-paused` protects `1b47c17`, including d7786f4, 369e762 and the standalone adapter at 7dbfc6d. `/tmp/miniproject_1-p29-adapters-20261101` is clean. Do not continue or integrate it.
- Legacy suite PID 25623 no longer exists. Its saved log has no final verdict and three diagnosed failures. No restart. The 135-fit scalar gate never ran and is abandoned with P2.9.
- `.venv` is node-local Python 3.12.16. Use it for project code. Node allocation: 32 CPU equivalents, 45 GiB RAM, A100 MIG 2g.20gb. No installs planned.
- Previous STATE is preserved in `notes/state-archive.md`. Historical banks, data and submission files stay unchanged.

## Running

Initial 12 feature trials complete. Best new standalone: year-relative audio LightGBM, 0.838242; best fixed addition 0.841616 (+0.000554 over E, below the noise threshold). Six feature follow-ups and the small model-family grid are next. Logs under `artifacts/fast/`.

## Next action

1. Read `notes/2026-11-01-fast-exploration.md`. Run feature additions on the best LightGBM recipe using the saved production folds and existing fold-local date encoders.
2. Try small grids of cheap non-menu families or distinct variants, using installed tools and the existing TabM adapter. Skip new framework/adapter work.
3. For every candidate report standalone CV and fixed-weight additions to E and the existing constrained blend; save OOF/test predictions, config and runtime. Keep one results table, commit code with results.
4. Propose 2 or 3 public-check files and two final recommendations. Sam decides. Apply the single final-file review only after his selections.
