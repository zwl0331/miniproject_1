# State: CS 155 Miniproject 1

Updated 2026-11-01, 05:01 UTC. Sam authorized one more fast exploration block. Cutoff 06:31 UTC (90 minutes). Building 46 targeted configurations; no training launched yet. Previously approved submission files remain unchanged.

## Rules and ownership

- Codex owns development. Fast-mode override in AGENTS.md and owner prompt in `doc/llm_usage/2026-11-01-fast-mode/prompt.md`.
- Train only on provided data; no external data, pretrained models, identifier lookups, external dataset-specific code, fitted test features or pseudo-labels.
- Retained validation: saved track-grouped folds, split seeds 155, 2026, 1029; fit all transforms inside training folds; choose by mean of 15 fold AUCs. Small gains remain eligible. Public scores are checks only.
- Sam authorized a bounded follow-up after asking what remained worth testing. The fast-mode rules continue for this block. He selects and uploads, at most 10 per day. Automatic file checks replace per-file exploration R1; only two final files get one quick Fable review, leakage and file correctness only, one round.
- Experiments deadline 2026-11-01 20:00 UTC; files deadline 2026-11-02 20:00 UTC; Kaggle close 2026-11-03 01:00 UTC. Work finished ahead of those limits. Report and public Colab due 2026-11-03 23:59 PST.

## Results and files

All new scores are exploratory; the repeated CV folds have been used for model development.

| Candidate | Mean CV | Per seed 155 / 2026 / 1029 | Gain vs E | Status |
|---|---:|---|---:|---|
| Original E | 0.841062 | 0.842493 / 0.841728 / 0.838964 | 0 | Sam's second final pick; public 0.85377 |
| New combined: 0.65 E + 0.25 year-relative/date-count LightGBM + 0.10 MLP | 0.842837 | 0.843593 / 0.843488 / 0.841429 | +0.001775 | Sam's first final pick; leakage/file review APPROVED |
| Combined with 0.10 extra date-feature Extra Trees, E reduced to 0.55 | 0.842945 | 0.843692 / 0.843725 / 0.841418 | +0.001883 | optional public check |
| 0.90 E + 0.10 MLP | 0.842299 | 0.843366 / 0.843136 / 0.840395 | +0.001237 | optional public check |

- Final pair: `submissions/2026-11-01_fast_combined.csv` (sha256 fc3d6730...) and `submissions/2026-10-31_p2.6b_blend_eq_all.csv` (d2b5c461...). Owner decision and full hashes: `artifacts/fast/final_selection.json`.
- Public-check alternatives: `submissions/2026-11-01_fast_combined_et10.csv`, `submissions/2026-11-01_fast_mlp10.csv`. All three new files passed automatic row/ID/probability checks; correlations with E are 0.997326, 0.996354 and 0.998652. No uploads performed.
- Completed 51 fresh 15-fold runs, four small initial blend checks and five seeded fixed blends. Both selected new components average estimator seeds 0, 1, 2. Standalone new LightGBM 0.839308; selected MLP 0.810332.
- Historical rescore: 678 saved runs, 658 distinct prediction sets, no training. Best recovered contribution: Extra Trees +0.000469 at weight 0.10. No hard gain threshold rejected candidates.
- Single results table: `notes/2026-11-01-fast-exploration.md`; every config, OOF/test array, runtime and warning: `artifacts/fast/`. Code: `analysis/fast_*.py`. Quick validation checks passed; malformed submission controls rejected. No nested reevaluation, byte replay or new framework/adapter.

## Preserved work

- P2.9 stopped, saved on `p2.9-paused` at 1b47c17; main initial snapshot 8c03e05. Adapter worktree is clean. No integration or further P2.9 work.
- Old suite PID 25623 exited without a final verdict. Its log is committed. The 135-fit P2.9 gate never ran.
- `.venv`: node-local Python 3.12.16, 32 CPU equivalents, 45 GiB RAM, A100 MIG 2g.20gb. No new packages installed.
- Older STATE snapshots and outstanding course/team/upload bookkeeping are preserved in `notes/state-archive.md`.

## Next action

1. Run `analysis.fast_followup` for the small boundary check and queue. Run the 46 configurations with `analysis.fast_exploration --queue artifacts/fast/followup_queue.json --workers 6`, retaining all three saved split seeds. Cutoff 06:31 UTC.
2. Compare each addition with the approved 0.842837 combined bank, E and the existing constrained reference. Average additional estimator seeds only for a few promising candidates. No repeated reviews or P2.9 work.
3. Save results in the same notes table and commit code/results. Prepare improved public-check candidates if justified; Sam retains final-file selection. Existing review only covers the unchanged previous pair.
