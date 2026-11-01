Critique below. Read-only; nothing executed. No verdict.

## Gate and amendment

**The reinterpretation needs an explicit R2 amendment, not a reading.** Plan 6.1 says: run the three timing recipes through both adapters on all 15 folds within 1e-12, and "also replay the old adapter against its historical banks", and "passing only the untouched old path is insufficient". The replay clause has no separate tolerance, so it inherits the project's 1e-12 verify criterion. The measured first-fold TabM replay fails that by seven orders of magnitude. Calling it "a separately reported diagnostic" changes what the gate requires, so it is a method change in a reviewed plan. Write the amendment, send it to the fresh reviewer, and do not run the six gate runs as a gate until it is approved. Running them as score-free pilots is fine.

**The amendment should say three things.** First, the hard gate is same-node: old adapter and new adapter on the node, all 15 folds, three recipes, within 1e-12, with bitwise agreement expected and any nonzero difference investigated. Second, the old adapter must itself be repeatable on the node in two fresh processes before the gate is read, as the P2.5 pilot did on the Mac. Third, the historical replay is a portability measurement with a recorded maximum difference, and its failure has the consequences below.

**The diagnosis the record should state.** Every platform-independent receipt field matched: split hashes, preprocessing digests in float64, seeds, update counts, chosen epoch. Only the float32 network trajectory diverged, from the ninth decimal at epoch one to the fourth decimal by epoch 87. That is float32 kernel arithmetic across arm64 and x86 builds, not a pipeline defect. Say so, and say that it is inferred, not traced.

**Claims to gate now.**

| Claim | Status after the measurement |
|---|---|
| P2.5 TabM results reproduced on the node | Not claimable. Findings 024 and 025 are Mac evidence. |
| P2.5 TabM runs count as P2.9 scientific controls | Excluded by plan 7's compatibility condition. The TabM 32 need fresh scalar controls, and the P2.5 best configuration should be one of them so the comparison is on-node. |
| P2.6b TabM banks verifiable under final source | Not claimable. They are not E atoms; label them nomination history. |
| E reproduced on the node | Open until the running full replay finishes. `references.json` records a blend reconstruction from historical member banks, not a node refit; its scope line says so. Do not cite it as reproduction. |
| H reproduced | First fold only. The 15-fold replay decides. |
| P2.7 comparisons in plan 10 | Now cross-platform for H and E. Label them so if the classical replay passes, and treat them as non-reproduction if it does not. |
| Historical CPU behaviour preserved | Re-scope to "preserved on the node across adapters". |

## Boundary, identities, environment, E

**Worker owns features.** The contracts document moves the feature pipeline and encoders into the worker. That works, but it widens the worker's import surface and the pipe payload. Two consequences must be handled. The parent's `source_manifest` hashes only modules loaded in the parent, so worker-only modules escape the run identity unless the parent imports them too; the document says so, and the hello should additionally report every worker-side module hash for the parent to check, because `source_digest` today covers one file. And the diagnostic S/V and encoder assignments now cross the pipe, so they belong in the request schema and the receipt, validated on both sides.

**Per-seed repetition is wrong.** "Recipes with several estimator seeds execute the entire stop/refit procedure per seed" repeats the S and D feature fits and encoder cross-fits for every seed. Those stages are label-driven but seed-independent. Share them within a fold and repeat only the network stages. This is not only cost: identical feature state across seeds is what makes the fresh final-policy run equal the mean of the single-seed runs, which is the plan's consistency check.

**Identities.** `packages_for` reports the distribution version, which is 2.14.1 and equals the Mac record, while the receipt reports the runtime string 2.14.1+cu130. My earlier statement that the PyPI build keeps the Mac version string was therefore half right and misleading; the integration note's correction stands. For v2 recipes and cache requests, add a declared route object with the runtime string, CUDA runtime, cuDNN version, the Linux lock hash and the device policy name. Keep hardware serials such as the MIG UUID in receipts, not in recipe hashes, or every recipe becomes node-specific.

**Single environment on scratch.** The environment is a symlink into node-local scratch after the quota failure. That is acceptable only if the committed lock and Python receipt recreate it exactly, which `verify` demonstrated once. Record the scratch path in `.agent/review.conf` for the reviewer's preflight, and treat node loss as a stop-and-rebuild event, not a continue.

**Thread environment.** The portability probe ran with the node's global `OMP_NUM_THREADS=2` and OpenBLAS at two threads; the node probe set four explicitly. The H match shows LightGBM ignored it, as expected with an explicit `n_jobs`. The running replay and every future fit should set the four thread variables explicitly and record them, because BLAS-bound families will otherwise run under a different, unrecorded setting.

**License record.** The probe stores `license: null` for all three neural packages because their METADATA License field is empty, and both rtdl packages share one LICENSE hash. A hash is provenance, not a license record. Add the SPDX identifier from the classifiers or the file's first line.

**E refit under final source.** Yes, it supplies a compatible discovery reference without redefining historical E, provided: the new bank gets its own run id and file and the historical bank, `menu.json` and `references.json` are untouched; it is compared to the historical bank with `compare_v2` at 1e-12 and the result recorded whichever way it falls; all P2.9 contribution blends use the final-source bank; and the nested evaluation rebuilds E from recipes, as it already does. It also serves a second purpose: if the classical replay under unchanged sources passes and the final-source refit then differs, the adapter commit changed v1 behaviour. Which E is the discovery comparator is a method choice affecting every P2.9 contribution number, so it is R2. Sam's decision that E is the fallback anchor is about the procedure, not bank bytes, and is unaffected. Sam should be told, not asked, if node-E differs from Mac-E, with the magnitude.

## Missing fields and tests before adapter work

| Field or test | What must exist first |
|---|---|
| BatchNorm final batch | A `last_batch` policy for ResNet that keeps every row, such as merging a singleton remainder into the previous batch, declared in the recipe, applied deterministically after the seeded permutation. Legacy TabM keeps the short last batch; the controls must not change. Test on a fixture whose size is one more than a batch multiple at S, D and nested sizes. |
| Category and flag layout | Explicit column lists for embedded continuous, flags and categoricals, with cardinalities and a reserved unknown index fitted on S or D. The probe passed `cat_cardinalities=None` for every TabM case, so TabM's categorical path is untested; MLP and ResNet take only `d_in`, so their categories need an external declared encoding. Test an unseen category at prediction. The one-hot versus token pair must differ only in that field. |
| Decay groups | A named policy per family: legacy all-parameter decay for controls, the authors' groups sorted by parameter name for FT-Transformer, with the ordered parameter-name list and its hash in the receipt. Test that two constructions give identical lists. |
| AUC stopping | `tolerance_auc` and the direction declared explicitly; the generalized stop rule tested by hand like the existing one; the AUC computed on CPU in float64 from the same averaged probability that is deployed, not from mean logits. |
| Epoch caps | 600 or 300 by learning rate in the recipe; `stopped_by: max_epochs` in receipts; the cap-only rescue rule referenced. |
| Embedding fields | PLE version, activation flags, bin count and the population the bins came from, with a bin digest; periodic frequencies, init scale and lite flag; start scaling init as the probe used. |
| Stopping-split purpose string | The v2 policy names it, so B controls reuse the legacy string and new families get their own. Test S/V equality for controls. |
| Deep fit accounting | A `subfits` receipt object counting feature fits, layout fits, network trainings, calibration base fits and sigmoid fits; the cost counter reads it; a synthetic end-to-end run's summed subfits must equal the predicted count. |
| GPU repeatability | The probe covers synthetic shapes with one seed call and no batch generator. Extend to the complete stopping adapter with the project's seed streams at full and nested fold sizes, two fresh processes, a restart, unrelated CUDA work before a repeat, and two concurrent workers. Bitwise. Freeze the policy document before any score. |

## Corrections to my earlier statements

- The probe script is complete, not empty; I read it mid-write.
- The interpreter now lives on scratch behind a symlink; my description of a conda prefix at `.venv` was the earlier state.
- The torch version-string expectation was wrong as stated; the metadata version matches the Mac, the runtime string does not, and the TabM predictions differ regardless.
- rtdl-revisiting-models is already in the single lock; my suggestion to add it as a route is moot.
- The 85-distribution inventory matches the lock and preserves every historical pin I listed; the environment is not the Mac inventory plus additions, and does not need to be.

Everything else in my first response stands, including the identity traps, the single adapter dispatch point, and the sequence.
