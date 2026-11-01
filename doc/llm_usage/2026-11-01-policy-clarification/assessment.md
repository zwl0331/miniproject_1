# Policy clarification assessment

2026-11-01. Sam's prompt and the quoted instructor policy are preserved verbatim
in `user_prompt.md`, written by the supervising Codex session. This receiving
session records the clarification while closing the authorized experiments.

The pipeline uses only provided training labels/features, with held-out folds
from that same training set. It does not use internet training data, pretrained
models, foundation-model-sampled examples, external validation data or obtained
test labels. No transformations or models fit on test.csv features.

The coverage is broad, not exhaustive. Tree ensembles, SVMs, Gaussian processes,
EBM, several neural architectures, ranking objectives, retrieval/neighborhood
features and learned ensembles were tested. Feature engineering, bagging and
resampling, and pair sampling are present. Synthetic-row augmentation such as
mixup/SMOTE/CutMix, input jitter, denoising and teacher-student distillation were
not tested. Augmentation derived entirely from each permitted training partition
is an allowed avenue; this does not establish that it would improve the score.

The quoted policy does not explicitly resolve fitting on unlabeled test features.
Sam's stricter no-test-fitting instruction remains in force. This question is a
clarification and coverage request, not approval for a new experiment block.

Final recommendations follow grouped CV. An interim incumbent-as-hedge rationale
that referred to its public score was corrected: the recommended pair is the two
new files, while Sam's authoritative selections remain unchanged. The public
score is logged as an owner-reported check only. All configs, seeds, fitted-fold
records and prediction banks are saved; this is not a byte-exact cross-platform
replay claim.
