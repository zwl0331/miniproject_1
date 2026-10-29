# CS 155 Miniproject 1: song popularity from audio features

Owner: Weilong (Sam) Zhou. Team members and Kaggle team name: to be recorded (teams of two to four, `../Project_1.pdf` section 2).

Caltech CS/CNS/EE 155, Machine Learning & Data Mining, fall 2026. Miniproject 1 (`../Project_1.pdf`, released Tuesday 2026-10-27): predict whether a song is of low or high popularity from its audio features. Kaggle competition: `https://www.kaggle.com/competitions/cs-155-fa-26-miniproject-1`. The metric is AUC, so a submission carries the probability of high popularity for each test song, in the format of `../data/sample_submission.csv`. A TA benchmark sits on the public leaderboard. The private leaderboard decides a small extra credit (top 50% of groups, up to 5 points); the report and the Colab demo carry most of the grade.

## Layout

| Path | Content |
|---|---|
| `doc/` | This README; `project_plan.md` (master plan); `plans/modeling_strategy.md` (methods, validation protocol, experiment order); `findings.md` (evidence ledger, Finding 000); `review_policy.md` |
| `doc/llm_usage/` | Evidence for the report's LLM Usage section: verbatim prompts and answers of LLM consultations, one directory per task |
| `analysis/` | Analysis code: `scoping.py`, the scoping run behind the modeling strategy, with its output `scoping_output.txt` |
| `reviews/` | Review records, one file per task, required by `review_policy.md` |
| `tools/` | `codex_review.sh` runs the independent Codex reviewer non-interactively; `fingerprint.py` computes revision fingerprints |
| `data/` | `train.csv`, `test.csv`, `sample_submission.csv` as downloaded from Kaggle; never modified |
| `latex/` | The course's report template, `project1_template.tex` with `preamble.tex` |
| `Project_1.pdf` | The assignment |
| `requirements.txt` | Pinned Python packages for the ignored environment `.venv/` (Python 3.12) |
| `CLAUDE.md`, `AGENTS.md` | Rules for AI agents; `AGENTS.md` links to `CLAUDE.md` |

Ignored by git: `.venv/`, `.review_logs/` (reviewer output kept by `tools/codex_review.sh`), Python bytecode, notebook checkpoints, `.DS_Store`.

## Status

- 2026-10-27: assignment released; data downloaded into `data/`.
- 2026-10-28: a read-only first look at the data in conversation (not reviewed; its facts enter the ledger as findings in phase 1). Python environment `.venv` created from `requirements.txt`.
- 2026-10-28: Piazza ruling on the use of the release date and the identifiers (course rulings below).
- 2026-10-28: project made review-controlled on the framework of the Positron Conversion DIO project (P0.1, `../reviews/2026-10-28-p0.1-bootstrap.md`).
- 2026-10-29: P0.1 approved by Codex in one round and committed (4b724f6, 90b52f7), with Sam's approval of the policy and confirmation of the reviewer model.
- 2026-10-29: Sam's decision: no pretrained models of any kind, because "pre-trained models certainly include data that is external to the project."
- 2026-10-29: the modeling strategy drafted with two Codex consultation rounds and a scoping run (P0.2, `../reviews/2026-10-29-p0.2-modeling-strategy.md`), in review. Development starts on Sam's go-ahead.

## Key numbers

None yet. Every number here will cite its finding in `findings.md`.

## Course rulings

| Date | Source | Question | Ruling |
|---|---|---|---|
| 2026-10-28 | Piazza, instructors' answer by Harry Chen to an anonymous question | The assignment says popularity is predicted from audio features alone, without identifying features; may `track_album_release_date` and the Spotify identifiers (`uri`, `track_href`, `analysis_url`) still be used? | "You are allowed to use all columns of the provided csv file." External data is still not allowed, including looking a song up by its identifiers. |

## Deadlines and logistics

| Item | Rule |
|---|---|
| Kaggle | Closes Monday 2026-11-02, 5:00 PM PST; late hours cannot be used. 10 submissions a day per team. Two final selections for the private leaderboard. |
| Report and Colab | Due Tuesday 2026-11-03, 11:59 PM PST, on Gradescope, one submission per group; late hours are taken from every member. |
| Colab demo | Runs the best model and several of the report's figures; public read-only link posted in a public Piazza note with the team name; both links in the report. |
| Report sections (template) | Introduction (15 points, at most 1 page), Overview (15, half a page recommended, at most 1), Approach (20, at most 1), Model Selection (20, at most 1), Conclusion (20, at most 1; the top 10 features, with their direction for bonus points), Additional insights (optional, 5 extra credit), LLM Usage |
| Data | Only the provided training set for training and validation; no external data (course rulings above) |
