# Neural model alternatives for Claude

Date: 2026-10-31, America/Los_Angeles. Author: Codex, in Sam's advisory chat. Status: provisional research advice within the project. Sam requested this companion handoff after discussing alternatives to SVM. The request authorizes this document, not new training or a revision of the frozen candidate menu.

## Recommendation

If Sam chooses to reopen model exploration, compare **TabM with numerical embeddings and a compact FT-Transformer**, with a small embedded MLP or ResNet as a simple reference. Keep an RBF SVM as an inexpensive alternative experiment. Consider TabR only after measuring the implementation and fitting cost of the first comparison.

The earlier [AUC plateau handoff](2026-10-31-codex-auc-plateau-handoff.md) put SVM first on cost and novelty. That was not evidence that SVM was the most likely route to 0.86. This note expands the neural options and specifies a neural-first branch if that is the direction Sam wants. No new experimental result has changed the measured ranking of the existing models.

The working hypothesis is that different representations or learning mechanisms could contribute information the current tree ensemble misses. There is no measured basis for promising that any named architecture closes the leaderboard gap.

## What the existing neural result establishes

[Finding 025](../doc/findings.md#finding-025-the-tabm-search) records eight TabM configurations with mean fold AUC from 0.8007 to 0.8034, against 0.8315 for the original LightGBM anchor. The best blend with that anchor reached 0.8320. Those are substantial negative results for the tested setup.

However, [the implementation](../songpop/phase2/tabm.py) sets `num_embeddings=None`. It also excludes fitted date blocks, including date counts and date target encodings, because the existing pipeline cannot safely pass them through the neural stopping split. The models did receive the ordinary F2 calendar features; this was not a comparison with no date information at all.

Every fit stopped by patience, and none reached the 300-epoch cap. Simply increasing that cap is therefore not an evidence-based first response. The matched quantile-scaling experiment was also inconsistent. The remaining proposals change representation or architecture, rather than repeating those particular changes.

Finding 025 is provisional and explicitly limits its conclusion to the tested protocol. The current equal ensemble scores about 0.8411 on the project's exploratory grouped CV. The best recorded public score is 0.85415 from a different combiner. Keep those metrics and candidates distinct. See [STATE.md](../STATE.md) and [the submission log](../submissions/README.md).

## Neural candidates

### TabM with numerical embeddings

An embedding converts a numeric value into several inputs before the main network combines features. Piecewise-linear embeddings represent where a value falls among intervals. Periodic embeddings provide another learned representation. Either could make useful nonlinear patterns easier to learn than supplying one standardized number.

This is a material extension of the existing TabM setup, although it is not a new model family. The [authors' implementation](https://github.com/yandex-research/tabm#creating-tabm-with-feature-embeddings) supports these representations. The [numerical-embeddings paper](https://arxiv.org/abs/2203.05556) found benefits across several neural architectures, including simple MLPs. These results support testing the idea; they do not establish its benefit for song popularity.

Suggested first test: a piecewise-linear variant against a matched no-embedding TabM control, using the same feature information, folds and training policy. Fit all bin boundaries within the relevant training population. Leave periodic embeddings as a possible follow-up rather than immediately multiplying every configuration by both options.

This has the strongest project-specific rationale among the neural proposals because an identifiable part of the existing setup was omitted. The large previous neural deficit still warrants caution about the likely size of any improvement.

### Compact FT-Transformer

FT-Transformer represents individual features as tokens and uses attention to combine them. It could learn interactions such as an audio pattern having different meaning in different release eras. That example is a hypothesis about this dataset, not an observed mechanism.

The architecture can be trained from scratch. It does not require an LLM or pretrained weights. The [original comparative study](https://arxiv.org/abs/2106.11959) establishes it as a serious tabular baseline and also finds no universally superior choice between neural models and boosted trees.

Suggested first test: a small number of compact configurations with explicit regularization and early stopping. With approximately 3,900 supplied training rows, start small. Compare against the embedded TabM on the same information and against the current ensemble through saved out-of-fold predictions.

This is the preferred genuinely different neural architecture in the first comparison. That priority reflects the useful contrast it provides, not evidence that attention will outperform TabM here.

### Embedded MLP or tabular ResNet

An MLP is a conventional feed-forward neural network. A tabular ResNet adds shortcut connections between layers. The [same comparative study](https://arxiv.org/abs/2106.11959) identifies a ResNet-style network as a strong baseline. Numerical embeddings can also benefit simpler networks, as discussed above.

Suggested first test: one small embedded MLP or ResNet as the reference, rather than a broad search over both. If promising, check several independent initialization seeds and their average. TabM's internal ensemble is not the same experiment as averaging independently trained networks.

This reference helps distinguish a useful numeric representation from an advantage specific to a sophisticated architecture. It can also reveal whether TabM's particular training setup was unusually unfavorable here. A simpler network is not automatically a better model for this dataset.

### TabR

[TabR](https://arxiv.org/abs/2307.14338) combines a neural network with learned retrieval of relevant training examples. For a new song, it can learn which supplied training songs are useful and combine their features and labels with its prediction.

This differs from ordinary kNN because similarity and aggregation are learned. The small contribution of the existing kNN nominee, documented in the earlier handoff, lowers confidence in a neighbor-based approach but does not settle learned retrieval.

Implementation would need careful isolation: retrieval candidates must come only from the appropriate training partition. During training, exclude the query's own track group from its retrieval candidates. During validation, never retrieve validation rows or labels, including from the stopping partition. No external song database is involved.

TabR is a later option because it introduces more integration work than changing TabM's input representation or adding a small feed-forward model. Measure cost before reserving a substantial run.

## Other model families

**RBF SVM:** retain as a small, distinct test of smooth feature interactions. Its priority in the earlier handoff was about an unexplored mechanism and likely manageable cost. It is not a high-confidence prediction of improvement. Evaluate decision scores for standalone AUC; use an explicitly group-safe calibration procedure if probability blending is needed. See the [SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

**Gaussian process classifier:** another kernel approach, capable of nonlinear classification with probabilistic predictions. Exact fitting can be expensive as the sample size grows. I would rank it below the first neural comparison and the SVM pilot because it explores partly overlapping kernel behavior with additional cost. Uncertainty estimates do not themselves imply better AUC. See the [official documentation](https://scikit-learn.org/stable/modules/gaussian_process.html).

Pretrained tabular foundation models remain excluded by the recorded course interpretation. All neural candidates above refer to random initialization and training solely on the supplied data. General architecture implementations are allowed; existing code written for this Spotify dataset is not.

## Proposed comparison and stopping decisions

1. **Freeze a small initial batch before reading its scores.** A reasonable starting comparison is a matched TabM control, one piecewise-linear TabM configuration, two compact FT-Transformer configurations and one embedded MLP or ResNet. These are proposed starting points, not a claim that one weak configuration eliminates a family. Set a total cost limit after timing representative fits, including preprocessing, stopping, refitting and verification.
2. **Separate architecture from feature availability.** First compare the architectures on the same F2 information. Then, if safe support for fitted date features is built, run a matched feature ablation on a nominated neural recipe. Do not attribute a gain to the architecture when the supplied information also changed.
3. **Preserve every training boundary.** Fit imputation, scaling, embedding bins and other learned transforms on the relevant training rows. For date target encoding, cross-fit the training features by track group and encode held-out rows from training labels only. Rebuild this process inside the neural stopping split; an outer-training encoding can still leak into inner stopping validation.
4. **Judge contribution to the current ensemble.** Save predictions on identical held-out groups. Report standalone mean fold AUC and the AUC change from one prespecified small blend with the current equal ensemble, for example 90% reference plus 10% candidate. A weaker standalone model can be useful if this contribution is reliable. Correlation alone is not a success criterion.
5. **Confirm before expanding.** Inspect paired changes by split repeat and use additional initialization seeds for promising neural recipes. Seek effects that persist beyond one favorable split or seed. Repeated folds are correlated, so do not treat their scores as independent evidence for an exaggerated significance claim. An isolated tiny gain should not trigger a large parameter search or a succession of public submissions.

The objective is a model that improves the existing ensemble reliably, not simply the highest new neural score. A useful handoff from the experiment would include its exact representation, stopping policy, runtime, standalone score, paired blend changes and seed sensitivity. If none contributes, retain the negative evidence and stop this branch rather than assuming a larger architecture must work.

## Integration with the current project

This assessment inspected HEAD `7e1e56f7cd7dc524ff501fa409ba9ad368929ee3`. There was active P2.7 work in the working tree. STATE's brief still says paused and its next-action text says no driver exists, while [decisions.md](../decisions.md) records Sam's later instruction to start P2.7 and [the driver](../analysis/p27_execute.py) exists. Follow the later owner decision and actual implementation state; this document does not assert that evaluation has completed.

The earlier handoff's clock references and snapshot discrepancies are historical. The current STATE has corrected the public-score mapping and the singleton-selection wording. This companion note does not reinstate the older errors.

P2.7 evaluates the frozen procedure described in [plan section 8](../doc/plans/phase2-models-features.md). Reopening the pool is a separate decision with the applicable R2 review. Do not tune these new candidates on P2.7 outer scores and then present those scores as independent validation of the expanded search. Preserve the authorized work and specify how any new exploration would coexist with it.

## Requested response from Claude

Please assess the proposed neural branch against the latest repository state. Identify which ideas have already been tested in an equivalent form, which remain meaningfully new, and which implementation can provide a fair comparison soonest. Recommend a measured cost budget and expansion rule if Sam chooses to proceed. Explain what the branch would displace before proposing a larger run.

This note records Codex's literature-informed proposals and its reading of the existing experiment evidence. No new model was fitted, dependency installed or submission generated. It is not an independent R1 or R2 approval. Retain it as AI-use evidence and integrate any adopted findings through the project's normal ledger and review process.
