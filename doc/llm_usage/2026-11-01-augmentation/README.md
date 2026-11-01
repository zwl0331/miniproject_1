# Bounded augmentation screen

Owner acceptance, verbatim, relayed by the supervising Codex session:

> Okay, after it finishes, let it try

Accepted proposal, verbatim:

> Yes, worth one small, capped test. I'd budget about 30 minutes for 2-4 mild augmentation settings on existing neural models, measuring whether they improve the current ensemble.

Priorities: mild Mixup and small continuous-input perturbations, preserving
categorical validity; saved grouped folds and seeds; keep small consistent gains
eligible. No broader search, new framework, external data, generated examples
from foundation models, or fitted test features.

Prior block closed at 275ae6a with both candidate CSVs preserved. This separate
block started 2026-11-01 08:02:25 UTC; fit cutoff 08:32:25 UTC, including setup.
The supervising session communicated the rounded 08:03-08:33 window. Compare
against x3_fixed_hybrid_retrieval5, CV 0.8446492317. Primary session retains code,
STATE and experiment ownership; supervision is read-only.

General Mixup reference: https://arxiv.org/abs/1710.09412. No code or data specific
to this Spotify dataset is imported. Mixup is adapted to match categorical and
missing-indicator profiles, and is disabled for retrieval/pairwise-loss routes.
Noise acts on observed continuous audio coordinates only, after training-fold
preprocessing, with training-range clipping. Validation/test inputs are unchanged.

Implementation and focused checks completed at 08:07 UTC. Four settings: TabM
Mixup alpha=.1, TabM audio noise sigma=.02, retrieval audio noise sigma=.02/.05;
augmentation probability .5. Same base recipes and estimator seed0 as saved
controls. Three augmentation checks plus three GPU route smokes pass. The test
loader also ran four imported legacy checks successfully; future invocations
now explicitly select the augmentation class to avoid that unnecessary rerun.
Total validation time 43.979 seconds. Sources are frozen before scientific launch.
