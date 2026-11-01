# State: CS 155 Miniproject 1

Updated 2026-11-01 08:00 UTC. Prior extended block is complete at 275ae6a. Sam explicitly authorized a separate 30-minute augmentation screen; setup started 08:02:25 UTC, fit cutoff 08:32:25 UTC. Existing CSVs/selections stay preserved. This session owns development after 32e7c5d. Results are provisional under fast mode; Sam's existing selected pair remains unchanged.

## Rules and ownership

- Primary Codex session: `01a19580-8f99-7492-9bc9-409b9774fb23`. The preceding session supervised read-only at Sam's request; no new review gate.
- Fast-mode rules: AGENTS.md and decisions.md. Provided data only; no external data, pretrained weights, identifier lookups, fitted test features or pseudo-labels. Every fitted transform stays within its training partition.
- Saved track-grouped folds, split seeds 155/2026/1029. Select by mean of 15 fold AUCs; small gains remain eligible; public scores are checks only. These folds have been reused for exploration.
- Sam selects and uploads, at most 10 per day. A changed final pair gets one quick Fable leakage/file-correctness review, one round. No R1 review of each exploratory file.
- Original experiment deadline: 2026-11-01 20:00 UTC. Files: 2026-11-02 20:00 UTC. Kaggle closes 2026-11-03 01:00 UTC. Report/Colab due 2026-11-03 23:59 PST.

## Results and files

| File | CV mean | Split 155 / 2026 / 1029 | Status |
|---|---:|---|---|
| `2026-11-01_fast_followup_trees.csv` | 0.843814 | 0.844708 / 0.844520 / 0.842215 | Existing first final pick; APPROVED; Sam reported public 0.86153 |
| `2026-11-01_fast_extended_neural_blend.csv` | 0.844649 | 0.846180 / 0.844871 / 0.842897 | Highest new CV; checked, provisional, not owner-selected |
| `2026-11-01_fast_extended_rank_neighbors.csv` | 0.844494 | 0.845795 / 0.845047 / 0.842640 | Simpler checked alternative; provisional |
| `2026-11-01_fast_combined.csv` | 0.842837 | 0.843593 / 0.843488 / 0.841429 | Existing second final pick; APPROVED at 91050e5 |

- Recommendation by grouped CV: extended neural blend plus the simpler rank/neighborhood file. Preserve follow-up trees as the submitted reference. The neural increment over it is +0.000155 with mixed split and estimator-seed signs; both new candidates improve all split means versus submitted follow-up trees. No private-score guarantee.
- The hybrid combines confirmed five-seed calendar-neighborhood LightGBM and three-seed PairLogit with the submitted blend. The neural candidate adds 5% of a three-seed TabR-inspired project retrieval model. Exact expressions: `artifacts/extended/fixed_blend_specs.json`, `retrieval_blend_spec.json`.
- Learned ensembles were tested: 27 regularized combiners over core-five/expanded-seven/final-nine pools. Best standalone 0.843574; best addition to old incumbent 0.844333. Six learned additions to the stronger hybrid gave at most +0.000006, so no further file was proposed.
- Both new files pass 863-row, ordered unique ID and finite-[0,1] probability checks. Hashes/CV/correlations: `submissions/README.md` and CSV JSON sidecars. No agent upload. Live leaderboard access failed; 0.86153 is explicitly owner-reported, rank/private score unverified.
- Recommendation only: `artifacts/extended/recommendation.json`. Authoritative owner selections remain `artifacts/fast/final_selection.json`; prior review: `reviews/2026-11-01-fast-followup-final-files.md`. Both selected CSV hashes are unchanged.

## Completed work

- 96 CPU configurations and 68 neural configurations including estimator-seed confirmations, all on 15 folds. `artifacts/extended/scientific_completion.json`: 164/164 complete. The original 64-case neural grid spans TabM/MLP/ResNet/FT-Transformer/retrieval. None of its 960 outer fit records hit the epoch cap.
- Three invalid neural identifiers failed before fitting; ID-only repairs completed with unchanged recipes. Original errors retained, all resolved. Corrected roster: `neural_queue_corrected.json`. No unresolved run errors.
- All 135 unique outer-local meta-training banks completed. Meta predictions were rebuilt inside outer training populations; global OOF banks were outer validation only. Focused known-wrong leakage controls pass. No dropped nested model-selection re-evaluation resumed.
- Neural adapter is complete and documented: `notes/2026-11-01-neural-adapter.md`. Scalar/mixed layouts, quantile/year/count features, PLE/periodic embeddings, AUC stopping, pairwise objectives and group-excluding retrieval. No pretrained models; retrieval is a project adaptation, not exact published TabR.
- Focused adapter CPU/GPU checks, three extended boundary checks, six GPU smoke fits, three combiner/leakage checks and file checks passed. Full predictions/configs/runtime/fit details are saved and committed. No packages installed.
- Closeout: `notes/2026-11-01-extended-closeout.md`. Results table: `notes/2026-11-01-fast-exploration.md`; `artifacts/extended/summary.csv`, `contributions.csv`, `learned_comparison.json`. LLM-use record: `doc/llm_usage/2026-11-01-extended-exploration/README.md` and `integration_index.md`.
- P2.9 remains preserved on `p2.9-paused` at 1b47c17; old worktree/gates remain paused. Environment: Python 3.12.16, 32 CPU equivalents, 45 GiB RAM, A100 MIG 2g.20gb. Earlier state: `notes/state-archive.md`.

- Policy clarification from Sam via the supervising session: `doc/llm_usage/2026-11-01-policy-clarification/user_prompt.md` and `assessment.md`. Provided-data augmentation is allowed but synthetic-row augmentation was not tested. No new experiment block or relaxation of the no-test-fitting rule was authorized.

## Next action

1. Active owner-authorized augmentation screen: read `doc/llm_usage/2026-11-01-augmentation/README.md`. Implement/test 2-4 mild settings on existing neural models; all augmentation inside training folds, preserve categorical/missing flags. Start 08:02:25 UTC, fit cutoff 08:32:25 UTC. Compare marginal gain to x3_fixed_hybrid_retrieval5 (.844649232). No wider search. Prior closeout/recommendations remain preserved.
2. Sam chooses/uploads the public checks and decides any change to his two final selections. Keep `final_selection.json` unchanged until he does. If changed, perform the one quick Fable review of that concrete pair, limited to leakage and file correctness, one round.
3. Then report and Colab: `doc/README.md`, `Project_1.pdf`, `latex/project1_template.tex`, results and `doc/llm_usage/integration_index.md`. Do not claim new candidate public/private results before Sam reports them.
