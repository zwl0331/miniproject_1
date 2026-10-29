Round B of the consultation. Thank you for round A; Claude has merged it with its own research and a scoping run.

Process clarification: in this consultation you are an advisor, not the author of a project work product. Claude Code is the author of the plan. A separate, fresh Codex session will review it formally under doc/review_policy.md. So do not run Claude or any other reviewer yourself, do not seek approvals, and do not modify any file. Your final message is advice to Claude.

Two new facts:
1. Sam's decision, 2026-10-29, verbatim: "pre-trained models certainly include data that is external to the project." All pretrained models are excluded: tabular foundation models and in-context learners pretrained on synthetic or real data, AutoML presets that load them, LLM-generated feature values, pretrained embeddings. Following the same strict reading of "you may only train and validate your models using the provided training set", the draft also excludes any training on test rows (fitting preprocessing on test features, pseudo-labeling, domain weighting). Do not argue for reinstating them; you may point out anything else in the plan that the decision or the strict reading should exclude, or anything excluded that should not be.
2. The scoping run: analysis/scoping.py, output analysis/scoping_output.txt. Key results: 5-fold StratifiedGroupKFold by uri, 3 repeats, audio plus date features: LightGBM (600 trees, lr 0.02, 15 leaves) 0.827, random forest 0.826, LightGBM default 0.822, HistGB 0.820, Extra Trees 0.820, XGBoost default 0.808, logistic regression on log features 0.781. Audio only: 0.73 to 0.76, so the date adds +0.05 to +0.08 (your round A range for this idea was +0.01 to +0.06). Ungrouped folds inflate AUC by +0.016. Train vs test adversarial AUC 0.529. Row order carries no signal. Exact-date target encoding 0.6311 versus year alone 0.6405.

The draft: doc/plans/modeling_strategy.md, plus the edits to doc/project_plan.md and doc/README.md (git diff shows them). Read it in full and critique it as hard as you can:
- Errors: wrong statements, numbers that do not match the data or the scoping output, wrong citations, internal contradictions, conflicts with AGENTS.md, the course rules, Sam's decision or the review policy.
- Gaps: important methods, risks, validation issues or report needs that are missing; anything from your round A that was dropped and should not have been.
- Cuts: anything that is not worth the time before Monday 2026-11-02, 5:00 PM PST, given that every task and submission waits for a review round.
- The validation protocol (section 4): is the grouping, the seed and fold design, the promotion rule (0.003 on all three seeds), the cluster bootstrap and the nested evaluation sound and feasible? Is the incomplete-row default (drop) right?
- The experiment plan (section 6): is the sequence and the time box realistic? What would you reorder?
- The ranked ideas (section 7): do the ranges now match the evidence?
You may run read-only checks with PYTHONDONTWRITEBYTECODE=1 .venv/bin/python (no file writes) and use web search to verify citations.

Deliver, in Markdown: a numbered list of concrete changes, each marked must-fix, should-fix or optional, with the section, the problem, and the exact replacement text or rule you propose. Then a short list of points where you disagree with the draft but would accept it. Be specific; do not restate what is fine.
