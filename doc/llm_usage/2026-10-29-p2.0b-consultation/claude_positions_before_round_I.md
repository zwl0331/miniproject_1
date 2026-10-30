# Claude's positions on P2.0b (written before reading Codex round I)

Written 2026-10-29, before the round I prompt was sent. Inputs: Sam's two messages asking for refinements to phase 2 (quoted in the round I prompt), the approved plan v0.1 and the P2.1 infrastructure in review, and the runtimes of `phase2_timing.py` (unreviewed until P2.0b converges).

## What changes

1. Budgets are set from measured runtimes, not the P2.0 caps. Compute is cheap (seconds to a minute per 15-fit run), so the binding limits are author and review time and, above all, selection optimism. More trials need more protection, not less.
2. All of Sam's refinements go into one amendment, P2.0b, reviewed before P2.2 relies on it: the three bounded refinements of his first message (kNN screen, cross-family date checks, contribution blends), now with the budgets of his second message, and the five items of the second (broader search, re-tuning after features, complementary models, learned ensembles, allocation rule).

## Search

| Stage | Position |
|---|---|
| Broad random search | All three split seeds. Batches of 50 for the boosted trees, 30 for the forests, 20 for the cheap smooth and kNN families. Initial caps: LightGBM 200, XGBoost 120, CatBoost 120, random forest 120, Extra Trees 60, HistGB 30, logistic with spline or log_standard 40, kNN 40, EBM 24. Wider ranges than v0.1 (for example LightGBM learning rate 0.01 to 0.1, leaves 4 to 63, minimum child samples 5 to 200, trees 200 to 3,000) |
| Expansion | A family's best within one grid step of a range edge extends that range once, by one step, in the next batch |
| Stopping | A family stops when two consecutive batches raise its best mean fold AUC by less than 0.0005, or at its cap |
| Focused refinement | For a family that qualifies under the allocation rule: up to 100 trials in batches of 25, around its best configurations. Optuna TPE only after its installation is reviewed (a new route lock); otherwise local random perturbation of the top five |
| Confirmation | The best five configurations of each family are rerun with estimator seeds 0, 1 and 2 averaged |

## Features

- Screening as in v0.1 on the anchor, then re-tuning of the two or three strongest model-and-feature combinations (up to 60 trials each).
- Removal checks on the best two recipes: drop each block, each group of date parts (calendar parts, `days`, `date_precision`), and each weak audio feature (key, mode, time signature, liveness), one at a time.
- Preprocessing alternatives for logistic, kNN and the network: log_standard against spline, and spline knots.
- The cross-family date checks of Sam's first message stay, declared now: `date_count` and `date_te30` on the tuned random forest and CatBoost; the four comparisons are a floor, extended under the allocation rule.

## Complementary models

- Each of XGBoost, Extra Trees, the smooth model, kNN and TabM gets a real search (above) and a contribution check against the current best blend.
- TabM: a timing pilot in P2.5 sets its budget; working assumption at most 40 configurations or 4 hours of compute, whichever comes first.

## Ensembles (P2.6)

- Candidates, declared before evaluation: equal averages of the best recipes of 2 to 5 families; top-k averages within a family (k = 3, 5); constrained weights (nonnegative, summing to one) fitted by minimizing log loss; one L2 logistic stacker on the logits of member probabilities, its strength fixed in advance.
- Fitting: for each outer repeat and fold, inner 5-fold grouped cross-validation on the outer training part rebuilds every member's out-of-fold predictions; the weights or the stacker are fitted on those only and applied to the outer held-out predictions of the members' outer fold models. Test predictions: each fold's combiner applied to that fold's member test predictions, averaged over folds.
- The restricted nested evaluation (Sunday) includes the learned combiner as a procedure, rebuilt inside each outer fold.

## Allocation rule (exploration only)

A family or block earns the next stage when, against its named comparator on identical folds: (a) at least +0.003 mean fold AUC and positive on every repeat; or (b) at least +0.001, positive on every repeat, and a bootstrap share of draws above zero of at least 0.9; or (c) adding it at 0.25 weight to the current best blend raises the blended mean fold AUC by at least 0.001, positive on every repeat. Final selection and deployment keep their own rules (plan sections 8 and 11).

## Schedule

Unchanged: menu freeze Saturday 18:00 PDT, final freeze Sunday 18:00 PST, Kaggle close Monday 17:00 PST. Two or three model processes of four threads each in parallel.

## Doubts

- Whether a fourth to sixth split seed (fresh folds) for confirming the top configurations is worth a new saved split file.
- How deep the nested evaluation can go once it includes a learned combiner (three levels of folds).
- Whether Optuna adds anything over local random perturbation at these budgets.
