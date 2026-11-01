# P2.9 GPU-node session handoff

2026-11-01. Sam: "Wrap up your work. I want to start a new session". Work paused; no jobs remain running. Read AGENTS.md, STATE.md, then this note and the next-action files before resuming. Codex remains primary; Claude Fable 5.1 xhigh is consultant and separate independent reviewer. Sam has explicitly approved sharing project context with Claude for those purposes; do not ask again.

## Saved checkpoints

| Commit | Scope and status |
|---|---|
| ff1d908, 74fc0b3 | Installed Linux environment, source/license audit, historical reference hashes, CUDA smoke probes and migration evidence; interpreter rebuild-lock header corrected and dry run passed |
| f420853 | Provisional raw-feature wrapper and strict population/split validation, eight focused tests pass; not wired into engine or neural worker |
| c6fa913 | Version 1.3 portability amendment, runtime/run/cache contracts and Finding 031; independent R2 round 2: `VERDICT: APPROVED` |
| aab2370 | Provisional score-free CPU/CUDA scalar TabM prototype and bounded pilot, Finding 033; no production banks or scientific scoring |
| 114c578 | Approval/status records; later handoff commit is found with git log |

Review: `reviews/2026-11-01-p29a-node-portability.md`. Fresh thread 5af7c77e-80a9-4ab2-85dd-13fd7f4b305c, explicit model claude-fable-5-1 and effort xhigh. All blocking findings resolved. F6 was a non-blocking index reference to the pilot before its commit; aab2370 resolves it. The review does not cover raw-feature code or the device prototype. Consultation prompts/answers are verbatim under `doc/llm_usage/2026-11-01-p29a-node/`.

## Environment and evidence

- A100 MIG 2g.20gb, 32-core-equivalent cgroup CPU quota, 45 GiB RAM. Visible CPU count is not the CPU allocation.
- Python 3.12.16; 81 hash-locked wheels plus four bootstrap distributions, 85 total. Exact installed inventory and pip check pass. All applicable historical version pins preserved. Historical requirements.txt is unchanged.
- The shared project filesystem hit its disk quota during installation. `.venv` is an ignored symlink to `/tmp/miniproject_1-p29-venv-20261101`. Node-local scratch is disposable. Rebuild from `configs/phase2/p2.9/install/python-linux-explicit.txt`, then `linux-cuda.txt` with pip --no-deps --require-hashes, and verify with `analysis.p29_environment verify` against the lock receipt. Rebuild parsing was dry-run checked; a second full install was not performed. REVIEW_PYTHON uses the project symlink.
- Nineteen foundation checks pass. Six tiny architecture CUDA probes repeat byte-identically in fresh processes. These are not full adapter gates.
- H plus six E atoms, all 15 saved folds, repeat byte-identically in two fresh node runs. H and LightGBM match Mac within 1e-12; XGBoost, CatBoost, Extra Trees and TabM differ. Causes remain untraced. Node E diagnostic AUC 0.8408963554 versus historical 0.8410616054; no new selection or submission. New discovery/menu E must be a fresh compatible node bank; old banks remain immutable.
- Raw boundary modules: `songpop/phase2/raw_splits.py`, `raw_features.py`; tests: `scripts/check_p29_boundaries.py`, eight passed. A deliberately leaky target encoder fails the isolation assertion, and allowed-label changes provide a positive control. Normal feature outputs/receipts match the legacy pipeline exactly on the fixture. No legacy module changed.
- `analysis/p29_tabm_device_pilot.py`: one fixed historical scalar B recipe, 3,091/2,061 fitting rows, two fresh processes per CPU/CUDA case. State/prediction/receipt/source hashes repeat exactly. Production CPU matches the old node adapter's first-fold predictions and full TabM receipt. Complete process times: production CPU 50 to 55 s, CUDA 23 to 24 s; smaller CPU 40 to 42 s, CUDA 17 to 18 s. Chosen epochs differ by device; this is about 2.24/2.37 times shorter for the complete procedure in these cases, not a general speedup. Peak Torch allocation about 83 MiB excludes other memory. No outer AUC, ranking or search budget.

## Interrupted verification

The full legacy `.venv/bin/python -m scripts.check_phase2` ran for about 40 minutes and was interrupted with SIGINT when Sam requested this handoff; tool exit 130. It did not complete, so no node-wide pass count is claimed. Its decorator-based runner prints the aggregate only at the end. The raw output is committed as `doc/llm_usage/2026-11-01-p29a-node/check_phase2.txt`; the traceback records the interruption, not an asserted model failure.

A brief read-only GDB stack sample located expensive grouped-bootstrap recomputation in the P2.4 feature-audit rejection checks (`phase2_stack_probe.txt`). GDB detached without changing state. The test parent and its joblib/resource-tracker helpers exited. No search or review process remains. A future full rerun needs enough time for these serial CPU checks; progress-printing in the check wrapper would help diagnose duration, without changing assertions or bootstrap counts. Do not label the interrupted suite passed.

## Next work, when resumed

1. Verify the scratch interpreter still exists and finish legacy verification with observable progress. Retain the interrupted log.
2. Implement strict recipe_v2 and raw adapter dispatch before D-wide FeaturePipeline fitting. Add runtime route to run configurations and nested cache keys independently of recipe schema, including unchanged recipe_v1 E atoms. Preserve legacy recipe hashes. Hash worker-only executing sources in the hello and parent identity. Add deferred combiner regularization/TabM receipt validation and tests.
3. Build the actual raw neural/calibration/retrieval adapters under the reviewed contracts. The boundary wrapper and scalar device prototype are starting components only. Complete three timing-recipe, all-15-fold old/new CPU comparisons, negative isolation tests and full/nested GPU repeatability before scientific scores. Match legacy B seeds/layout/optimizer; share seed-independent S/D transforms across estimator seeds.
4. Materialize executable ordered manifests, then representative full/nested route pilots across families and the fit-count graph. Present the measured full-search/reserve/evaluation/replay proposal to Sam before any large launch. The 274 initial configurations are coverage targets, not launch authorization.

The reviewed amendment preserves E's procedure/weights, the 0.003 fallback trigger and 0.001 simplicity band. Historical Mac runs do not count as P2.9 scientific coverage. The revised menu uses node E for comparisons. If E is selected, the already released historical file remains the default available artifact for Sam's final pick; a distinct node E file needs R1 and Sam's decision. No old artifact, final selection or Kaggle submission changed in this session. Report and Colab remain deferred until development ends.
