# Findings

This file is the evidence ledger for the Miniproject 1 model. New factual conclusions about the data and the models land here before code, Kaggle submissions, the Colab notebook, or the report may depend on them. The convention is ported from the Positron Conversion DIO project (its Finding 000), which borrowed it from `DIO_Shape_Systematic`.

## Review Status Labels

- `draft`: written for discussion, not yet accepted.
- `reviewed`: named by a converged review record under `../reviews/` (`review_policy.md` section 4.1); the finding cites the record.
- `superseded`: replaced by a later finding.
- `rejected`: kept for audit history, but not used.

## Entry Template

```text
## Finding NNN: Short Title

Question:

Current conclusion:

Evidence:

Uncertainty:

Review status:

Downstream use:
```

## Finding 000: Review-First Convention

Question:

What rule keeps the models, the submissions, and the report auditable under a six-day deadline?

Current conclusion:

Every number that leaves this project, into a Kaggle submission, the Colab notebook, the report, or Piazza, must trace to findings in this ledger and to the reviewed code that produced it, with its inputs recorded: the data files and their hashes, the code revision, the random seeds, and the cross-validation split. If a code change would imply a new conclusion about the data or the models, the finding is written or updated first.

Evidence:

- Ported from the Positron Conversion DIO project's Finding 000 with the review framework, on Sam's instruction of 2026-10-28 (task P0.1).
- The failure surface here: a validation score inflated by leakage (repeated tracks across folds, preprocessing fitted on validation data) or by tuning to the public leaderboard looks like progress until the private leaderboard. Recording each number with its split and inputs makes such errors visible to the reviewer.

Uncertainty:

None. This is a working rule, not a measurement.

Review status:

`reviewed`, by the P0.1 review (`../reviews/2026-10-28-p0.1-bootstrap.md`). This ledger is committed only after that review converges.

Downstream use:

Every later finding.
