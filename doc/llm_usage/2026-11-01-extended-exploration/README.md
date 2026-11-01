# Extended fast exploration

Owner instruction, 2026-11-01, verbatim:

> Can you test these and take over from the other session, I will close it when it finishes. Note again my goal and the abundant computational resources

Codex session 01a19580-8f99-7492-9bc9-409b9774fb23 takes ownership after the
preceding session's clean handoff at 32e7c5d. The new authorization reopens
development beyond the prior short follow-up. The original experiment deadline
of 20:00 UTC remains the limit. The target is improved AUC, especially on private
data, selected by the existing three grouped split seeds. Existing selected
files and reviews are preserved.

Codex implements and tests the remaining feature-transfer, neural representation,
broader EBM, ranking-objective and learned-neighborhood/retrieval comparisons.
General model APIs are checked against installed code and primary documentation.
No external song data, pretrained weights, fitted test features or pseudo-labels.

Initial CPU queue: 20 EBM configurations with explicit whole-track bootstrap
bags, automatic interactions, no internal validation or early stopping; eight
pairwise CatBoost configurations; twelve LightGBM/Extra Trees comparisons with
group-cross-fitted neighborhood and learned-cluster features. The EBM API smoke
fit took 3.66 seconds for one outer fold at 500 rounds and four bags. Pairwise
probabilities and neighborhood query immutability checks passed before launch.
Full results/configs/OOF/test predictions live under artifacts/extended.

This work is provisional exploration under the owner's fast-mode rules.
