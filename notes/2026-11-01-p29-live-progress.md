# P2.9 live continuation index

Superseded at session wrap-up by `notes/2026-11-01-p29-session-handoff.md`.
The narrative below is preserved as history. Fable recheck is now APPROVED at
369e762, seven scalar and eight receipt tests passed, and the 135-fit CPU gate
did not start. Sam explicitly left its approval pending. Development is paused.

2026-11-01. Codex primary. Sam resumed development and requested Fable for
consultation and separate review, then asked to accelerate and optimize public
and private AUC. The reviewed selection protocol and measured large-job gate
remain in force. Do not treat status questions as a pause.

Main checkout: `/exp/mu2e/app/users/wzhou2/projects/miniproject_1`, master,
checkpoint 8dab4a6 (readiness progress wrapper, diagnostics, resume records).
The full old suite runs here with unchanged model sources and original test
assertions, session 87959, PID 25623, started 03:25:37 UTC. Log:
`doc/llm_usage/2026-11-01-p29a-readiness/check_phase2.txt`. It is currently in
the expensive P2.4 feature-audit fixtures. No complete verdict yet. It initially
failed kNN exact batch equality, historical anchor exact arrays, and a test
helper calling startswith on a null recipe_id. Isolated diagnostics measured
roundoff at 2.7756e-17 and at most 1.1102e-16 respectively. Never call this old
suite passed. Keep its sources fixed until it ends.

Development worktree: `/tmp/miniproject_1-p29-adapters-20261101`, detached
commit d7786f41cd478c037536daa84f0e28065a69528c. Shares this repository's git
object store. Run its code with the main project's `.venv/bin/python` and that
worktree as cwd. Its first unit corrects those three tests under the existing
1e-12 policy and adds deferred legacy TabM receipt validation and combiner
lambda/C validation. Five new receipt tests and nine affected legacy checks pass.
Full evidence and draft Finding 034 are committed there. Root's draft Finding
034 was created separately; reconcile documentary differences when integrating.
No v2 adapter exists yet. Do not lose this worktree's commits. Integration should
wait for the fixed-source original suite to finish.

Consultation: session 20358, Claude CLI with explicit model claude-fable-5-1,
effort xhigh, permission-mode plan, tools Read/Glob/Grep, max-turns 60. Prompt,
independent design and response JSON live in the MAIN checkout under
`doc/llm_usage/2026-11-01-p29b-adapters/`. JSON stays empty until completion.
Extract its verbatim `result` and actual session/model metadata after completion;
write integration notes. The prompt requests concrete v2 dispatch, runtime/cache
identities and neural-worker design, plus critique. A separate fresh review is
required for consequential method checkpoints.

Formal review: session 10748, cross-review runner from the DEVELOPMENT
worktree, record `reviews/2026-11-01-p29a-verification.md`, round 1 at d7786f4.
Reviewer is Fable 5.1 xhigh, read-only. It covers verification fixes and receipt
validators only. Read appended verdict and resolve findings under AGENTS.md.
The runner's review copy/log directory may use the development worktree's name.
The skill was read from `/home/wzhou2/.codex/skills/cross-review/SKILL.md`.

Completed diagnostic sessions: 17436/34380 isolated original failures,
34820 eight boundary tests passed; 98135 original kNN traceback; 39439 three
failure diagnostics; 36107 three corrected checks; 53167 five receipt tests;
8097 nine affected checks. Their outputs are in the readiness directory (some
only in the development worktree). No search fits or new production banks.

Environment: `.venv` survives at `/tmp/miniproject_1-p29-venv-20261101`,
Python 3.12.16, 32 CPU-core-equivalent quota, 45 GiB RAM, AMD EPYC 7543 host,
A100 MIG 2g.20gb. Historical Mac was Apple M5, 10 cores. More concurrency is
available; no controlled per-fit CPU speedup claim. The sandbox often fails
to mount devpts even for read-only commands. Retry important failures with
require_escalated and a scoped justification. Project-context sharing with
Claude was already authorized; do not ask again. Git commits need local
`-c user.name='Sam Zhou' -c user.email='zwl20000331@gmail.com'` because no
default identity exists. Do not set global config. No AI trailers.

Time estimate communicated to Sam: provisional engineering estimate 12–24 h
for a useful core expansion, 24–48+ h for comprehensive P2.9 including revised
nested validation/replay. This is not a measured forecast or resource approval.
Replace it with representative complete-route pilots and a feasible proposal.

## Later checkpoint in the same session

Consultation completed successfully, actual Fable session
2928bc8b-bd04-4142-b6ae-f28a911f16af, 81 reported turns, 920870 ms. Verbatim
extracted result and integration account are in the main consultation directory.
Review round 1 at d7786f4 returned VERDICT: APPROVED, fresh Fable thread
805d0fad-9d1f-44d2-a15a-4b86862201e6. The record remains in the development
worktree. Six non-blocking findings concern shared legacy defaults, missing
positive receipt cases, untraced kNN roundoff, receipt-validation cost, an
intermediate log source hash and stale STATE text. Corrections are being made
without delaying separate provisional adapter work. Material fixes need recheck.

Uncommitted development files now include raw_recipes.py (strict scalar-only v2),
neural_backend.py (raw S/D fitting and fixed-horizon trajectory diagnostic),
raw_neural.py (isolated worker, D-state parent prediction), and
scripts/check_p29_raw_neural.py (bounded 400-row/four-epoch CPU checks).
They are NOT connected to recipe/engine/run/cache dispatch yet and have not had
full CPU/CUDA acceptance gates. An initial startup failure on Torch's virtual
source filenames was fixed with is_file(). A second run passed six checks but
the deliberately wrong query-refitting control changed category dimensionality;
its fixture now preserves all categories. Current scalar run session 47509,
log check_raw_scalar_v3.txt in the development readiness directory. Earlier
failed logs are retained.

Review follow-ups modified tabm.py, models.py and receipts.py in the development
worktree only: one shared LEGACY_FIT_POLICY for factory/validator, and memoized
pure stopping splits keyed by complete labels/groups/fold count/seed with
immutable results. All receipt checks still run. Added max-epoch/multiple-seed
positive cases and cache-key rejection checks; check_receipts_v2.txt is running.
No raw-neural source should be edited while its test worker is active because
the protocol checks source hashes on each exchange. Root model/test sources
remain unchanged while the full original suite runs.
