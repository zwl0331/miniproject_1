# AUC brainstorm by a separate Claude session, 2026-10-30/31

Evidence for the report's LLM Usage section (`../../../Project_1.pdf` section 6). Sam ran a read-only brainstorm on how to raise the competition AUC in a separate Claude Code session (Claude Fable 5.1, titled "AUC score improvement brainstorm"), which used three sub-agents: A built its own pipeline knowing nothing of the project's results; B did data forensics with null comparisons; C ran matched experiments on mechanisms the project had not tested. That session wrote nothing into the project. Its findings reached the primary agent session (also Claude Fable 5.1) as a cross-session message on 2026-10-31 at about 03:35 PDT, with the words "Sam asked me to feed them to you". Everything it produced is provisional: none of it went through the project's engine, log or verification.

| File | What it is |
|---|---|
| `message_from_brainstorm_session.md` | The message, verbatim, with the time it was received |
| `scratchpad/me/REPORT.md` | The brainstorm's own report, with every number it cites |
| `scratchpad/me/leaderboard_2026-10-30_2235PDT.md` | Its running notes, including the public leaderboard as it read the page at 22:35 PDT on 2026-10-30 |
| `scratchpad/me/*.py`, `*.log`, `*.csv` | Its scripts and outputs: all 576 phase 2 banks rescored on singleton rows, stored-bank blends, full refit against fold average, bagging strength, era recalibration, the within-test duplicate pairs |
| `scratchpad/agentA_fresh/`, `scratchpad/agentB_structure/`, `scratchpad/agentC_mechanics/` | The sub-agents' reports, scripts, result tables and logs |

Copied on 2026-10-31 from `/private/tmp/claude-501/-Users-zwl0331-Desktop-Mini-project-1/9f6b6eec-676c-4338-a75c-d6002d6d0d06/scratchpad/` with rsync, text files only: 85 files, 672 KB. Binary arrays (`*.pkl`, `*.npy`, `*.npz`, `oof/`) and CatBoost's working directory were left out; the numbers they support are in the text outputs. The sub-agents' reports do not name their own model identities; the brainstorm session reported them as its sub-agents.

## What the primary session did with it

Done on 2026-10-31, during the pause after P2.6b (Sam, 2026-10-30), without launching any fitting job:

- Checked the two claims that the project's own data and banks can settle, in `../../../analysis/brainstorm_checks.py` with its output `brainstorm_checks_output.txt`: the test file's construction (an 80/20 row split from which every track that occurs in train was removed) and the frozen menu's scores on singleton validation rows. Both reproduce; the details are Finding 029 (`draft`) in `../../findings.md`.
- Recorded the brainstorm here and in `integration_index.md`. Nothing was adopted into the models, the frozen menu (`../../../configs/phase2/menu.json`), the selection rule, the nested protocol or the submission file.

| Brainstorm recommendation | Status in the project |
|---|---|
| 1. Blend the leading LightGBM with Extra Trees (+0.003) | Already in the frozen menu: the 0.25 Extra Trees contribution blend (0.8394) and the production selection, the equal average of four channels with Extra Trees at 0.25 (0.8411), Finding 028 |
| 2. Extra channels: the kNN anchor, CatBoost with date blocks, LightGBM without the cross-fitted date blocks | Not adopted. The pool and the menu are frozen and R2-approved; adding channels after seeing scores is the score-driven revision plan section 11 forbids. Sam's decision and an R2 review would be needed |
| 3. Keep `fold_average` deployment | Already the menu's deployment default; P2.7's deployment rule tests the full refit against it |
| 4. Report the singleton-row score beside the official metric; prefer the candidate whose singleton score is not lower within a tie | Computed for the 12 menu candidates (Finding 029): the production selection has the highest singleton score in its tie set. Whether phase 3 reports it is Sam's call; the frozen rule is unchanged |
| 5. Treat differences under 0.001 as ties | Already the frozen rule's tie band |
| 6. Expected public score 0.845 to 0.85 for the blend | The R1-approved file scored 0.85377 on the public leaderboard (Sam, 2026-10-31), a check only |
| 7. Transductive uses of `test.csv`: boost the 22 duplicate-pair rows; pseudo-labelling | Not adopted. A course-rule question for Sam; the brainstorm itself puts the value well under +0.003 |
| 8. Not worth building: histogram resolution, categorical dates, same-date neighbour features, smoothing below 3, era stacking, inverse weights, year-month encodings | Noted; none was planned |

Roles for the LLM Usage section: Claude Fable 5.1 (the brainstorm session and its three sub-agents) produced the advice, read-only; Claude Fable 5.1 (the primary session) wrote the check script, Finding 029 and these records; Codex did not take part in this record (no review has been requested; Finding 029 would be reviewed with the phase 3 selection if it is used there). Consultation advice is input, not approval.
