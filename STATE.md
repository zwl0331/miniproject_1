# State: CS 155 Miniproject 1

Updated 2026-11-01 07:47 UTC. This session owns development after 32e7c5d. Sam reported public 0.86153 for follow-up trees and requested learned-ensemble tests. CPU exploration and all seed confirmations are complete. Core-five and expanded-seven learned ensembles are scored. Expanded-seven pairwise shrinkage, blended at 25% in logit space, improves CV by 0.000271 across all split seeds. The three-seed shallow L2=10 ranking channel adds 0.000456 with 10% probability blending, all split seeds positive. Final nine-channel pool includes that ranker and the confirmed five-seed neighborhood model. Neighborhood meta fitting completed; 15 ranking meta tasks launched at 07:30 UTC with six CPU workers; 12/15 complete at 07:47 UTC, last three expected near 07:54 (`stacking_ranking.log`). The corrected 64-case neural grid is complete, including all three ID-only repairs; no unresolved scientific run errors. One allowed late confirmation is the retrieval c16/w64/scalar model, which adds +0.000146 to the stronger hybrid (2/3 splits); two extra seeds launched with one GPU worker at 07:36 UTC (`gpu_late_confirm.log`). Worker-bound neural sources stay fixed. Original 20:00 UTC deadline applies; estimated modeling/file completion 07:45-08:30 UTC.

## Rules and ownership

- Receiving Codex session `01a19580-8f99-7492-9bc9-409b9774fb23` owns development, STATE.md and decisions.md after this handoff. Fast-mode override: AGENTS.md; owner prompts and steering: `doc/llm_usage/2026-11-01-fast-mode/`.
- Provided data only; no external data, pretrained models, identifier lookups, dataset-specific external code, fitted test features or pseudo-labels.
- Saved track-grouped folds; split seeds 155, 2026, 1029; all fitted transforms inside training folds. Rank by mean of 15 fold AUCs. Small gains remain eligible; public scores are checks only.
- Sam selects and uploads, at most 10 per day. Automatic file checks; one quick Fable review of the selected final pair, leakage and file correctness only, one round. The preceding session has finished. This receiving session continues the owner-authorized exploration, as recorded in decisions.md.
- Follow-up cap was 06:31 UTC; training finished about 05:40 UTC. Original experiment deadline 2026-11-01 20:00 UTC; file deadline 2026-11-02 20:00 UTC; Kaggle closes 2026-11-03 01:00 UTC. Report and Colab due 2026-11-03 23:59 PST.

## Final candidates

All scores are exploratory on repeatedly used grouped folds.

| File | CV mean | Per split 155 / 2026 / 1029 | Status |
|---|---:|---|---|
| `2026-11-01_fast_followup_trees.csv` | 0.843814 | 0.844708 / 0.844520 / 0.842215 | Sam's first final pick; APPROVED; public 0.86153 |
| `2026-11-01_fast_combined.csv` | 0.842837 | 0.843593 / 0.843488 / 0.841429 | Sam's second final pick; APPROVED at 91050e5 |
| `2026-11-01_fast_followup_trees_cat.csv` | 0.843988 | 0.845335 / 0.844740 / 0.841890 | Optional public check; small mixed CatBoost gain |
| Original `2026-10-31_p2.6b_blend_eq_all.csv` | 0.841062 | 0.842493 / 0.841728 / 0.838964 | Earlier reference; public 0.85377 |

- Follow-up trees = 0.40 original E + 0.25 five-seed year-standardized/count LightGBM + 0.10 original three-seed MLP + 0.25 five-seed feature-transfer Extra Trees. Gain +0.000978 over previous combined, +0.002753 over E; all three split means improve.
- Authoritative selection: `artifacts/fast/final_selection.json`; previous pair: `final_selection_initial.json`. New files passed automatic 863-row/ordered-ID/finite-probability checks; hashes and CV logged in `submissions/README.md`. Pair correlation 0.997782. Sam uploaded follow-up trees and reported public 0.86153; no agent upload or push.
- Follow-up: 48 initial CPU configurations, six extra-seed runs, eight scientific GPU configurations, six CPU seed averages, two neural averages and small fixed probability/logit blend checks. All complete, no failures. Separate two-epoch adapter smoke is not a scientific candidate.
- Broader tests include four additional RBF SVMs, four fixed-kernel Gaussian processes, embedded TabM/MLP/ResNet and FT-Transformer. None of their tested additions, replacements or logit blends improved the stronger tree blend. Earlier two SVM additions did give small gains to the original E; they are retained, not erased by a threshold.
- Table: `notes/2026-11-01-fast-exploration.md`. Full records/configs/OOF/test arrays: `artifacts/fast/`, `artifacts/neural/`. Summary: `artifacts/fast/followup_summary.json`. Existing split/bank checks retained; focused new-feature checks in `followup_checks.txt`.

## Preserved work and environment

- P2.9 remains saved on `p2.9-paused` at 1b47c17; original worktree unchanged. Old suite PID 25623 exited; the old 135-fit gate never ran.
- Sam separately assigned adapter completion to Codex session `01a19580-8f99-7492-9bc9-409b9774fb23`. Its isolated commits bf9aa18/2b463c1 were integrated as e9eb1fe/a85b3ca. No old process gates resumed. See `notes/2026-11-01-neural-adapter.md` and its LLM-use directory.
- Node: Python 3.12.16, 32 CPU equivalents, 45 GiB RAM, A100 MIG 2g.20gb. No packages installed during fast exploration.
- Full prior states and open course/team/upload bookkeeping: `notes/state-archive.md`.

## Next action

Prepared provisional hybrid candidate: `2026-11-01_fast_extended_rank_neighbors.csv`, CV 0.844494 (+0.000680, all splits positive), sha `e94065dd0834358511bf433e38115f5ae031a93743a58f0ce63bf29f839de975`; automatic checks passed. Details: `notes/2026-11-01-extended-closeout.md`. Existing final selections unchanged.

1. Running logs under `artifacts/extended/`: `gpu.log` (original grid), `gpu_id_repairs.log` (three original half-pairwise cases renamed after ID rejection; use `neural_queue_corrected.json` as the 64-trial completion roster), `stacking_cpu_a.log` (32 tasks, 1 worker), `stacking_cpu_b.log` (20 tasks, 1 worker), `stacking_cpu_c.log` (38 tasks, finished fits), then `final_cpu_confirm.log` (14 fits, 4 workers, SAME shell chain); `stacking_neural.log` (15 three-seed TabM meta tasks, 2 GPU workers). Do not duplicate C or launch superseded full-B/ranking-only queues. CPU initial, short, EBM confirmation and XGB confirmation queues are done.
2. Neighborhood meta fitting and confirmations finished: the selected five-seed c64 calendar-neighborhood LightGBM adds +0.000269 with 10% logit blending, all splits positive; c8 is nearly tied. Its 15-task meta queue is running (`stacking_neighbors.log`, two workers). Ranking confirmations finished: the L2=10 shallow ranker is selected by mean CV and consistency; its 15-task queue is `stacking_ranking_queue.json`. Core-five and expanded-seven scoring completed (`learned_core.log`, `learned_expanded.log`). Score nine combiners on core-five, expanded-seven and final expanded pool. Specs: `stacking_pool_xgb.json`, `stacking_pool_expanded.json`, `stacking_pool_final.json`. Global OOF arrays are outer validation only, never meta-training data. Data/split consistency and focused known-wrong leakage controls pass. No new nested model-selection re-evaluation.
3. Stop condition: finish the existing queues; allow extra seeds for at most two promising late neural configurations; compare a few fixed additions with the confirmed channels and learned ensembles; prepare/check/log up to 2-3 useful files and recommend two final selections. No new method families or broad grids. Read `doc/llm_usage/2026-11-01-extended-exploration/README.md`, `artifacts/extended/contributions.csv` and the results table. Other session is supervising read-only at Sam's request; send it queue/ensemble/file milestones, no new review gate.
4. Sam retains uploads/final selection. The existing selected pair is unchanged and APPROVED (`reviews/2026-11-01-fast-followup-final-files.md`). Any newly chosen final pair gets its one quick Fable leakage/file review; no review of exploration. Report/Colab are separate after development and final selections: `doc/README.md`, `Project_1.pdf`, `latex/project1_template.tex`, results and LLM integration index.
