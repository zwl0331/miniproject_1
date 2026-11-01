Owner decision, 2026-11-01 (Sam): switch this project to fast exploration mode for the rest of development.

## Why
My words: "I'm just trying to get a quick and dirty test of the different methods and see if I can improve my AUC results a bit more." "I really don't want to waste more time on this." The goal is unchanged: the best AUC on the public and, above all, the private board. The P2.9 process (consultation and R2 before building, byte-exact replay, receipt validators, the 274-configuration plan, measured launch proposals) spends the remaining time on checks instead of models. It stops now.

## First, about 30 minutes
1. Read AGENTS.md, STATE.md and notes/2026-11-01-p29-live-progress.md for context only. Do not continue P2.9.
2. Stop the old legacy suite run (PID 25623) if it is still running. I have closed the previous Codex session.
3. Save, don't finish. Commit as is: the pending files in the main checkout on master, and the /tmp/miniproject_1-p29-adapters-20261101 worktree (commit d7786f4 plus its uncommitted files) on a new branch p2.9-paused. No further work on it.
4. Add a section "Fast exploration mode (Sam, 2026-11-01)" near the top of AGENTS.md. It overrides the Review section, the Codex consultation convention and the review-first findings rule until development ends. Add one row to decisions.md quoting this message. Save this prompt verbatim under doc/llm_usage/2026-11-01-fast-mode/. Rewrite STATE.md short, archiving the old one to notes/state-archive.md.

## Rules that stay
- Course rules: train only on the provided data. No external data, no lookups by Spotify identifier, no pretrained models, no code written for this dataset. Do not fit on test.csv features (no pseudo-labels or transductive tricks); that is an open course-rule question.
- Validation: folds grouped by track, with the same split seeds (155, 2026, 1029) and the same split code as the saved banks, so every new out-of-fold prediction is comparable with and blendable into the existing ones. Fit every preprocessing step and encoder inside the training folds only. This is the one thing not to cut.
- Choose by mean fold AUC over the three split seeds, as in STATE.md. The public board is a check only; it has run 0.012 to 0.015 above CV for every file.
- Every submission file passes a quick automatic check before I upload it: row count and ids match test.csv, probabilities finite and in [0, 1], and its correlation with the current best file printed. Log each file's CV, sha256 and, later, its public score in submissions/README.md. At most 10 uploads a day.
- Save each result's out-of-fold and test predictions plus the config that produced it. Fixed seeds. Commit code with results. Note LLM use briefly for the report.

## Rules dropped until development ends
R2 reviews of exploration; Codex or Fable consultation before building; R1 review of every submission file; byte-exact or cross-platform replay; receipt validators; nested re-evaluation; manifests and coverage targets; measured launch proposals; writing findings in doc/findings.md before code. Compute on this node is pre-authorized within the time box. Don't ask me to approve reversible steps.

## Time box
Stop new experiments at Sunday 2026-11-01, 12:00 PM PST. Final files ready by Monday 2026-11-02, 12:00 PM PST (Kaggle closes at 5:00 PM PST). Give me a few lines every 2 to 3 hours, or sooner if something beats the current best by more than 0.002 CV.

## What to try, in this order
Baseline to beat: blend_eq_all, CV 0.8411, public 0.85377 (configs/phase2/menu.json). Best standalone: LightGBM with date_count and date_te3, CV 0.8368.
1. Features first; they produced the biggest past gains. Try the ideas in doc/llm_usage/2026-10-31-auc-brainstorm/ and your own, each on the best LightGBM recipe, with a small re-tune if needed.
2. Cheap model families for blend diversity, only ones not already in the menu: for example sklearn MLP, RBF SVM, spline logistic regression, XGBoost or HistGB variants, and the existing TabM adapter. Small grids. Skip anything that needs a new framework or adapter.
3. Blending: for each new model, report the CV change when it is added to the current equal blend and to a simple constrained blend.
Use the existing phase 2 engine where it is faster; otherwise write a standalone script that reproduces the same grouped splits. Treat differences under about 0.002 CV as noise and prefer the simpler model in a tie.

Keep one results table in notes/2026-11-01-fast-exploration.md: model or feature change, CV mean, per-seed CV, change against the current best blend, runtime.

## At the end
Propose 2 or 3 files for public checks and your recommendation for the two final selections, each with its CV and reasons. I decide and upload. Then one quick Fable review of the two final files only, limited to leakage and file correctness, one round. After that, development is over and the work moves to the report and Colab.
