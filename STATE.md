# State: CS 155 Miniproject 1

Updated 2026-11-01, 05:41 UTC. The authorized follow-up is finished. Sam selected follow-up trees plus previous combined. One final leakage/file review is pending; all training jobs have exited.

## Rules and ownership

- Codex owns the project. Fast-mode override: AGENTS.md; owner prompts and steering: `doc/llm_usage/2026-11-01-fast-mode/`.
- Provided data only; no external data, pretrained models, identifier lookups, dataset-specific external code, fitted test features or pseudo-labels.
- Saved track-grouped folds; split seeds 155, 2026, 1029; all fitted transforms inside training folds. Rank by mean of 15 fold AUCs. Small gains remain eligible; public scores are checks only.
- Sam selects and uploads, at most 10 per day. Automatic file checks; one quick Fable review of the selected final pair, leakage and file correctness only, one round. No further modeling or review loops.
- Follow-up cap was 06:31 UTC; training finished about 05:40 UTC. Original experiment deadline 2026-11-01 20:00 UTC; file deadline 2026-11-02 20:00 UTC; Kaggle closes 2026-11-03 01:00 UTC. Report and Colab due 2026-11-03 23:59 PST.

## Final candidates

All scores are exploratory on repeatedly used grouped folds.

| File | CV mean | Per split 155 / 2026 / 1029 | Status |
|---|---:|---|---|
| `2026-11-01_fast_followup_trees.csv` | 0.843814 | 0.844708 / 0.844520 / 0.842215 | Sam's first final pick; review pending |
| `2026-11-01_fast_combined.csv` | 0.842837 | 0.843593 / 0.843488 / 0.841429 | Sam's second final pick; bytes unchanged |
| `2026-11-01_fast_followup_trees_cat.csv` | 0.843988 | 0.845335 / 0.844740 / 0.841890 | Optional public check; small mixed CatBoost gain |
| Original `2026-10-31_p2.6b_blend_eq_all.csv` | 0.841062 | 0.842493 / 0.841728 / 0.838964 | Earlier reference; public 0.85377 |

- Follow-up trees = 0.40 original E + 0.25 five-seed year-standardized/count LightGBM + 0.10 original three-seed MLP + 0.25 five-seed feature-transfer Extra Trees. Gain +0.000978 over previous combined, +0.002753 over E; all three split means improve.
- Authoritative selection: `artifacts/fast/final_selection.json`; previous pair: `final_selection_initial.json`. New files passed automatic 863-row/ordered-ID/finite-probability checks; hashes and CV logged in `submissions/README.md`. Pair correlation 0.997782. No uploads or pushes performed.
- Follow-up: 48 initial CPU configurations, six extra-seed runs, eight scientific GPU configurations, six CPU seed averages, two neural averages and small fixed probability/logit blend checks. All complete, no failures. Separate two-epoch adapter smoke is not a scientific candidate.
- Broader tests include four additional RBF SVMs, four fixed-kernel Gaussian processes, embedded TabM/MLP/ResNet and FT-Transformer. None of their tested additions, replacements or logit blends improved the stronger tree blend. Earlier two SVM additions did give small gains to the original E; they are retained, not erased by a threshold.
- Table: `notes/2026-11-01-fast-exploration.md`. Full records/configs/OOF/test arrays: `artifacts/fast/`, `artifacts/neural/`. Summary: `artifacts/fast/followup_summary.json`. Existing split/bank checks retained; focused new-feature checks in `followup_checks.txt`.

## Preserved work and environment

- P2.9 remains saved on `p2.9-paused` at 1b47c17; original worktree unchanged. Old suite PID 25623 exited; the old 135-fit gate never ran.
- Sam separately assigned adapter completion to Codex session `01a19580-8f99-7492-9bc9-409b9774fb23`. Its isolated commits bf9aa18/2b463c1 were integrated as e9eb1fe/a85b3ca. No old process gates resumed. See `notes/2026-11-01-neural-adapter.md` and its LLM-use directory.
- Node: Python 3.12.16, 32 CPU equivalents, 45 GiB RAM, A100 MIG 2g.20gb. No packages installed during fast exploration.
- Full prior states and open course/team/upload bookkeeping: `notes/state-archive.md`.

## Next action

1. Run and collect the one fresh Fable review of the selected pair: `reviews/2026-11-01-fast-followup-final-files.md`. Scope is leakage and file correctness. Do not launch more training or repeated reviews.
2. Deliver files and verdict. Sam uploads and reports public scores; log them without tuning models to the board. Final Kaggle selections remain his action.
3. Then report and Colab. Read `doc/README.md`, `Project_1.pdf`, `latex/project1_template.tex`, the results table and `doc/llm_usage/integration_index.md`. Model development is finished.
