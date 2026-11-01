# CS 155 Miniproject 1: rules for AI agents

Owner: Weilong (Sam) Zhou. Workflow: agent-workflow v1, profile homework, set up 2026-10-30 by migration from the review-gated rules of 2026-10-28 (`decisions.md`). Codex reads this file as `AGENTS.md`; Claude Code reads it as `CLAUDE.md`, which is a link to it. By the owner's decision these rules replace any global review loop for this project, and `doc/review_policy.md`, under which the records of P0.1 to P2.4 round 4 were made.

## Fast exploration mode (Sam, 2026-11-01)

In force until development ends (`decisions.md`, 2026-11-01; the prompt verbatim in `doc/llm_usage/2026-11-01-fast-mode/prompt.md`). It overrides the Review section, the R1 rule for every submission file, the Codex consultation convention and the review-first findings rule below. Goal: the best AUC on the public and, above all, the private board, through quick tests of features and methods. P2.9 is paused on branch `p2.9-paused`.

- **Kept:** the course rules under "Binding constraints"; no fitting on `test.csv` features (no pseudo-labels or transductive tricks). Folds grouped by track, split seeds 155, 2026 and 1029, the same split code as the saved banks; every preprocessing step and encoder fitted inside the training folds. Choose by mean fold AUC over the three split seeds; the public board is a check only. Every submission file passes the automatic file check and is logged in `submissions/README.md` (CV, sha256, later the public score). Save each result's out-of-fold and test predictions with its config; fixed seeds; commit code with results; note LLM use briefly.
- **Dropped:** R2 reviews of exploration; consultation before building; R1 review of each submission file; byte-exact or cross-platform replay; receipt validators; nested re-evaluation; manifests and coverage targets; measured launch proposals; findings in `doc/findings.md` before code. Compute on this node is pre-authorized within the time box. Do not ask Sam to approve reversible steps.
- **Time box:** new experiments stop Sunday 2026-11-01, 12:00 PM PST; final files ready Monday 2026-11-02, 12:00 PM PST. A short update to Sam every 2 to 3 hours, or sooner for a gain above 0.002 CV.
- **End:** 2 or 3 files proposed for public checks and a recommendation for the two final selections; Sam decides and uploads; one quick Fable review of the two final files, limited to leakage and file correctness, one round. Then the report and Colab.
- **Records:** the results table is `notes/2026-11-01-fast-exploration.md`.
- **File checks and selection:** match every submission's row count and IDs to `test.csv`, require finite probabilities in [0, 1], print correlation with the current best file, and allow at most 10 uploads per day. Treat CV differences below about 0.002 as noise and prefer the simpler model in a tie. Try features first, then small cheap-family grids, then blend comparisons.

## Every session, and after a compaction

1. Read this file, then `STATE.md`, then the files named under "Next action".
2. If `STATE.md` disagrees with `decisions.md` or with the files, say so; do not pick silently. A dated owner decision beats an older file. Measured facts come from evidence files.
3. Project state lives only in this repository, never in a tool's own memory, which the other provider cannot read.

## What this project is

- Objective: Caltech CS/CNS/EE 155 Miniproject 1 (`Project_1.pdf`): predict from a song's features whether its popularity is low or high. Kaggle competition `cs-155-fa-26-miniproject-1`, scored by AUC on submitted probabilities. The brief is in `STATE.md`; the background, layout and course rulings in `doc/README.md`.
- Deliverables and deadlines: Kaggle submissions until Monday 2026-11-02, 5:00 PM PST, at most 10 a day per team, uploaded by Sam, with two final selections that are Sam's decision; a report (90 points) from `latex/project1_template.tex` and a public Colab demo (10 points), due Tuesday 2026-11-03, 11:59 PM PST, on Gradescope.
- Binding constraints, from `Project_1.pdf` section 2 and the instructors' Piazza answer of 2026-10-28:
  - Train and validate only on the provided data. Every column of `data/train.csv` and `data/test.csv` may be a model input, `track_album_release_date` and the Spotify identifiers included. Looking a song up by its identifiers, or using any other external data, is not allowed; pretrained models count as external data (`decisions.md`, 2026-10-29).
  - No existing code written to model this Spotify dataset. Open-source tools and general techniques are allowed. No collaboration between teams.
  - Every use of an LLM is reported in the report's LLM Usage section, in the format of `Project_1.pdf` section 6: the LLMs used, the components they touched, the interaction, and how the output was integrated.
- Pre-authorized routine actions (no review, no approval; never final selections, grades, or messages to people): none. Every Kaggle submission file gets an R1 review before Sam uploads it (`decisions.md`, 2026-10-30).

## How work runs

- One primary agent owns `STATE.md` and the plan. Helpers get bounded tasks with an output file. They do not edit `STATE.md`, plans or decisions.
- Work in small, complete, committed results. Label them provisional until they are reviewed. Do not ask the owner to approve reversible steps.
- Record owner decisions in `decisions.md` when they are made. Record job launches in `STATE.md` when they start. Refresh `STATE.md` at every checkpoint, and move old material to `notes/state-archive.md`.

## Review (cross-review skill; settings in .agent/review.conf)

- **Required:**
  - R1: anything that leaves the project or is used as final.
  - R2: a method choice that would invalidate a lot of later work, reviewed before that work.
  - R3: a costly or irreversible action, which also needs the owner's OK.
- **Not required while provisional:** exploration, reruns, refactors with unchanged tested behaviour, mechanical moves checked by checksum, updates to `STATE.md` and records, plans for reversible work. **The only exception to R1:** the pre-authorized routine actions named above.
- **Material edits.** An edit after review is material if it can change a number, claim, method, data selection, behaviour, grade, or the scope of a statement. A material edit needs a recheck that covers what depends on it. Non-blocking findings never block.
- **Stop and ask the owner only when:**
  - a blocking finding is still disputed after one exchange of evidence;
  - the same defect survives two fixes;
  - resolving a finding needs the owner's authority or information the agent lacks;
  - a finding implies that the objective or a decision is wrong.
- **Status:** provisional; reviewed (`APPROVED` at a commit, no blocking finding open); released (the owner approved the external use).
- **Older documents.** The plans in `doc/` call for readiness and completion reviews, convergence records and fingerprints under the old rules. Where they require a review, these triggers apply instead; the documents are not rewritten for it.

## Needs the owner

External or irreversible actions (sending messages, submitting, releasing grades, pushing to shared repositories, deleting data, large compute jobs), method choices with large consequences, disputes that evidence could not settle, and changes to this file.

## Homework profile

- **Course rules are binding:** allowed resources, collaboration, and how AI use must be disclosed. They are listed under "Binding constraints".
- **Method check (R2):** the validation scheme and leakage, the metric, before tuning.
- **Verification:** tests that fail on a known-wrong implementation; held-out validation; several seeds; simple baselines.
- **Release review (R1):** the report's claims checked against the results files; the submitted code; final selections; every Kaggle submission file (above). **AI-use record:** `reviews/`, `decisions.md` and `doc/llm_usage/` are the evidence for the LLM Usage section.

## Conventions

- **Layout:** `STATE.md` (snapshot, at most 80 lines), `decisions.md` (append-only), `reviews/` (one record per review unit), `notes/`. `doc/` holds the README (layout, course rulings), `project_plan.md`, `plans/`, `findings.md` (the evidence ledger) and `llm_usage/`; `data/` is the Kaggle data as downloaded, never modified; `doc/README.md` lists every directory.
- **Findings:** `doc/findings.md` keeps its review-first convention (Finding 000): a factual conclusion lands in the ledger before code, submissions, the notebook, or the report depend on it. A `draft` finding is provisional and may be built on provisionally; it becomes `reviewed` when an `APPROVED` record at a commit names it.
- **Codex consultation** (`decisions.md`, 2026-10-29): before building a task, write Claude's positions to a file, then ask Codex for an independent design and a hard critique of the draft. Prompts and answers are kept verbatim under `doc/llm_usage/<date>-<slug>/`. The reviewer is always a separate fresh run.
- **Python:** the environment `.venv` (Python 3.12), built from the pinned `requirements.txt` and ignored by git. Run project code with `.venv/bin/python`, as modules from the project root. **Review environment** (for the reviewer's preflight; `.agent/review.conf`): that interpreter of the live project, outside the review copy; the data is tracked in git.
- **Commits:** plain conventional style naming the task ID; no AI trailers. **Style:** short sentences, tables for parallel facts, dated status lines, no dashes as punctuation. **Reporting:** say what was checked, what was not, and what is still provisional.
