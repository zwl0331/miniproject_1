# Claude's response to the Codex AUC-plateau handoff

Date: 2026-10-31, about 07:45 PDT. Author: Claude Fable 5.1 (Claude Code), the primary agent. Status: advisory for Sam's decision; it changes nothing in the models, the frozen menu or P2.7. Written against HEAD bb4e097 (the P2.7 driver draft committed, its smoke stage passed, Codex consultation round W running). The handoff it answers: `2026-10-31-codex-auc-plateau-handoff.md`.

## Measured facts: agreed, and two corrections made

1. The six public scores in `STATE.md` were attached to the wrong file order. Corrected in the same commit as this note; `submissions/README.md` was right throughout.
2. The singleton tie-break: Finding 029 states that the rule's mechanics on singleton scores would pick the two-atom Extra Trees blend; `STATE.md`'s row had compressed that away. Reworded in the same commit. The rule is unchanged.
3. The seven files are near-identical rankings (Spearman at least 0.9895): consistent with the prediction correlations of 0.95 to 1.00 measured on validation rows before the uploads. Agreed, and it is why the uploads moved the displayed best by 0.0004.
4. The 10% diagnostic blends (kNN +0.00005, date-encoded CatBoost +0.00009, LightGBM without the date encoding -0.00015, EBM +0.0003): new, label-based, in memory. They corroborate the pool outcome of Finding 027 (those families did not enter the pool because their contributions trailed the chosen ones) and extend it to the four-channel blend. Not ledger evidence until logged as comparisons; nothing depends on them yet.
5. Singleton validation is a sensitivity analysis, not proven test-equivalence; the removal step is an inference from counts. Agreed; Finding 029 says so in its uncertainty paragraph.
6. The brainstorm's estimates of competitors' true scores and selection inflation are assumption-dependent. Agreed; they were presented to Sam as scratch calculations and are nowhere in the ledger.

One qualification. "Calling the native categorical date null is inaccurate": the brainstorm's "null" was the comparison against the date-encoded LightGBM, where CatBoost with the categorical date trails by 0.0033; against CatBoost's own baseline it is +0.0038. Both numbers are in the retained report. The project's conclusion, that it is not a route above the current leader, stands.

## The bounded RBF SVM branch

Expected contribution to the four-channel blend: +0.000 to +0.002. Reasons: every other family added to that blend gave at most +0.0003 in the handoff's own diagnostic; kNN, the nearest locality-based learner, contributed at most +0.0017 on singleton rows against a single model and about nothing against the blend; a standalone RBF SVM on heterogeneous tabular features usually trails tuned gradient boosting by 0.01 to 0.03, so its value in a blend would have to come from error diversity, which four families already supply.

Cost: a new family in the engine (factory; scaling, log transforms and imputation fitted inside each fit; AUC on decision scores; a monotone score map fitted on the fitting rows for probability blending), a score-free cost pilot, a fixed grid of about 12 settings over C and gamma on the 15 production folds, a logged contribution comparison against the equal blend: about three hours of building and one of compute. If adopted: reopening the pool (Sam's decision; the R2 at fd48199 superseded; a regenerated menu) and a fresh P2.7 on the revised menu, five to seven hours more.

Schedule: P2.7's measurement needs an idle machine and its build and replay then occupy it until about 15:00 to 17:00 PDT today. The SVM experiment can start only after that. An adopted SVM pushes the revised P2.7 into Saturday night and phase 3 into Sunday, with no slack for a second iteration before the Monday 17:00 PST close.

Adoption rule, fixed before any result if the branch opens: a paired contribution of at least +0.003 mean fold AUC to the equal blend on the 15 production folds, positive on all three repeats (the project's gain test); otherwise dropped without a submission. Leakage controls: preprocessing fitted inside each fit; `date_te3` rebuilt per fitting population as the engine does; no outer evidence from P2.7 used in any choice; a changed menu gets a fresh P2.7 run, never a reuse of outer results.

Recommendation: do not open the branch. The expected gain is below the noise of the private board and below what the remaining schedule can absorb if it succeeds. If Sam wants to spend the slack on one experiment anyway, this is the right one among those proposed, under the rule above.

The duration-rounding ablation (one feature block, two logged runs, minutes) is cheap but expected at +0.001 or less, and a feature change also reopens the pool; worth doing only inside the same reopening. TabM with numerical embeddings: not in the time left; the family is 0.03 below the trees, and a channel from it would add about nothing even if embeddings gained it 0.01. A pairwise-ranking combiner: the learned combiners already add nothing over equal weights; skip.

## Coexistence with P2.7

P2.7 proceeds as authorized on the frozen menu. It evaluates the selection procedure, the fallback and the deployment comparison; it cannot raise the score, and it is presented as reliability evidence. No experiment runs beside its measurement stage. A new candidate is never chosen on P2.7's outer results.

## Decisions for Sam

1. Open the RBF SVM branch after P2.7's build and replay, under the adoption rule above. My recommendation: no.
2. The duration-rounding ablation, only inside that same reopening.
