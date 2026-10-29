# CS 155 Miniproject 1: project rules for AI agents

Owner: Weilong (Sam) Zhou. This project is review-controlled. Full protocol: `doc/review_policy.md`. Read it before producing any work product.

## What this project is

- Caltech CS/CNS/EE 155, Machine Learning & Data Mining, Miniproject 1 (`Project_1.pdf`, released 2026-10-27): predict from a song's audio features whether its popularity is low or high. Kaggle competition `cs-155-fa-26-miniproject-1`, scored by AUC on submitted probabilities.
- Deliverables: Kaggle submissions until Monday 2026-11-02, 5:00 PM PST, with two final selections; a report (90 points) from `latex/project1_template.tex` and a public Colab demo (10 points), due Tuesday 2026-11-03, 11:59 PM PST, on Gradescope.
- Made review-controlled on 2026-10-28 on the framework of the Positron Conversion DIO project (task P0.1).
- Start with `doc/README.md` (what, layout, status, course rulings, deadlines), then `doc/project_plan.md` (phases, roles, risks) and `doc/findings.md` (the evidence ledger).

## Binding rules

1. Every work product must be reviewed by a second, independent agent and receive an explicit written `VERDICT: APPROVED` before it is delivered as final, committed, submitted as a job, or built upon. A work product is any change to project files, any produced data, plot, number, conclusion, or recommendation, and any state-changing action.
2. The reviewer is an agent from a different model family than the author: by default OpenAI Codex (gpt-6-astra, reasoning effort xhigh), run non-interactively in a read-only sandbox by `tools/codex_review.sh`. Only when Codex is unavailable, and with the reason recorded, a fresh Claude general-purpose subagent started with the Agent tool. Never a fork, never the author's own session. The author never self-approves.
3. On `CHANGES REQUESTED`, the author fixes or rebuts every finding in writing and resubmits to the same reviewer thread. Repeat until the reviewer issues `APPROVED` on the current revision fingerprint with no blocking finding open or contested, and every non-blocking finding fixed, deferred by agreement, or recorded as a disagreement. That is convergence.
4. Any edit after approval invalidates it. A new round is required.
5. After 3 rounds without convergence, or as soon as a blocking finding is contested (rebutted by the author, upheld by the reviewer), stop and escalate to Sam. A round cap, a timeout, or silence is never approval. If Sam says continue, the count restarts.
6. Every review cycle is recorded in `reviews/YYYY-MM-DD-<slug>.md` using the templates in `doc/review_policy.md` section 7, with revision fingerprints from `tools/fingerprint.py`. Quote the reviewer's verdict verbatim to Sam.
7. The reviewer inspects the actual files and outputs, verifies independently, and does not modify the work product. An `APPROVED` without verification evidence is invalid.
8. Sam is the final authority. Only Sam can waive review, explicitly and per task. Record the waiver.
9. `CLAUDE.md` and `doc/review_policy.md` change only with Sam's explicit approval.

## Course rules

Binding as well. From `Project_1.pdf` section 2 and the instructors' Piazza answer of 2026-10-28 (`doc/README.md`, course rulings).

- Train and validate only on the provided data. Every column of `data/train.csv` and `data/test.csv` may be a model input, `track_album_release_date` and the Spotify identifiers included. Looking a song up by its identifiers, or using any other external data, is not allowed.
- No existing code written to model this Spotify dataset. Open-source tools and general techniques are allowed.
- No collaboration between teams.
- Every use of an LLM is reported in the report's LLM Usage section, in the format of `Project_1.pdf` section 6: the LLMs used, the components they touched, the interaction, and how the output was integrated. The review records are the evidence for it.

## Exempt from review

- Read-only investigation and explanation that changes nothing and produces no new number, result, or recommendation.
- Writing the review record itself.
- Files in an agent's own scratchpad.

## Project conventions

- `AGENTS.md` is a link to this file, so Codex reads the same rules. When Codex is the author, Claude Code is the reviewer, run non-interactively in plan mode (`doc/review_policy.md` section 8.4).
- Documents live in `doc/`, review records in `reviews/`, review tooling in `tools/`, the assignment in `Project_1.pdf`, the Kaggle data as downloaded in `data/` (never modified), the report template in `latex/`. Code, notebooks, figures and submission files get their directories from the task that first needs them; `doc/README.md` lists every directory.
- Python: the environment `.venv` (Python 3.12), built from the pinned `requirements.txt` and ignored by git. Run project code with `.venv/bin/python`.
- Kaggle allows 10 submissions a day per team. Each submission is a gated action (`doc/review_policy.md` section 4.3). The two final selections are Sam's decision.
- `doc/findings.md` keeps its own convention (Finding 000, review-first): a factual conclusion about the data or the models lands in the ledger before code, submissions, the notebook, or the report depend on it. A finding changes status to `reviewed` only through a converged review record that names it.
- Style: short sentences, tables for parallel facts, dated status lines, no dashes as punctuation. Commit messages in the repository's plain conventional style, naming the task ID; no AI trailers.
