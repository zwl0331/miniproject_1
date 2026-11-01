# Fast exploration mode: how the prompt was made

2026-11-01, about 04:00 to 04:12 UTC.

| Item | Fact |
|---|---|
| LLM | Claude Opus 5.5 (`claude-opus-5-5`) in Claude Code |
| Interaction | Sam asked for a read-only status report, then said the checks were slowing progress ("I'm just trying to get a quick and dirty test of the different methods and see if I can improve my AUC results a bit more"). Claude estimated that P2.9 at its planned pace would use the whole window to the Kaggle close, and that new model families were unlikely to close a 0.006 to 0.016 CV gap. Sam then asked: "Can you write a prompt that changes the current rule such that the development is faster and more faithful to the goal I stated in this session with you?" |
| Output | `prompt.md`, drafted by Claude for a new Codex session. |
| Integration | Sam pasted the prompt into the Claude Code session instead. The new Codex session had received only "Can you read in the project to get started?" and was idle, so Claude Code carries out the prompt as the primary agent. |
| Effect | `AGENTS.md` section "Fast exploration mode (Sam, 2026-11-01)"; one row in `decisions.md`; `STATE.md` rewritten short. |

## Codex takeover

Sam subsequently delivered the same prompt verbatim to this Codex session and
confirmed: "Codex owns development; other session paused". Codex preserved the
existing edits in commit 8c03e05, checked that `p2.9-paused` protects the complete
adapter chain at 1b47c17, and rewrote STATE for fast exploration. The earlier
Integration row describes the preceding session, not current ownership.
No design consultation or exploration review is required in this mode.

Codex wrote `analysis/fast_features.py`, `fast_exploration.py`, `check_fast.py`, `fast_reuse.py` and `fast_submission.py`. The runner reuses saved grouped splits and existing encoders. New features are fitted only on training rows. All 12 initial feature trials completed; results, configs and probabilities are saved under `artifacts/fast/` and summarized in the requested notes table. No consultation or review was run.

Completion: 51 fresh 15-fold runs and a 68.7-second rescore of 678 saved runs (658 distinct predictions), with no new packages. Small gains were retained after Sam clarified their value; his later instruction stopped search expansion. Three public-check CSVs were prepared with automatic checks. Sam chose the new combined CSV and original E for the one final Fable leakage/file review. Full results and limitations are in the requested notes table.

Final review: Claude Fable 5.1 xhigh returned `VERDICT: APPROVED` in one fresh,
read-only round at commit c04f340. The request and verbatim response are in
`reviews/2026-11-01-fast-final-files.md`. The reviewer inspected fold-local fitting,
group boundaries, class orientation, component arrays, CSVs, hashes and reported
CV. Its one non-blocking status-label finding was resolved by identifying
`artifacts/fast/final_selection.json` as authoritative in `submissions/README.md`.
No model, prediction, CSV or scientific claim changed after review. Development
closed 2026-11-01, about 04:49 UTC; report and Colab are next.
