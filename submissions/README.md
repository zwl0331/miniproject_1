# Submissions

Every file here is written once by `../scripts/prepare_submission.py` (through `songpop.artifacts.write_submission`: exclusive create, read back, validated), next to a sidecar JSON with its provenance and its Kaggle description. Each submission is a gated action (`../doc/review_policy.md` section 4.3): it is reviewed before Sam uploads it. Sam uploads every file and reports the public score.

## Log

Events, newest last. A file reviewed and ready for Sam, a file uploaded, and a public score are separate events.

| Date | File | sha256 | Event |
|---|---|---|---|
| 2026-10-29 | `2026-10-29_p1.3_lightgbm_F2.csv` | 13f4f8b6… | Prepared in P1.3 (recipe `lightgbm_F2`, run `production_lightgbm_F2_9c37e77d549e`, mean fold AUC 0.8315); in review (`../reviews/2026-10-29-p1.3-first-submission.md`) |
| 2026-10-29 | `2026-10-29_p1.3_lightgbm_F2.csv` | 13f4f8b6… | Approved (P1.3, round 3) and committed (f6add21); ready for Sam's upload |
| 2026-10-29 | `2026-10-29_p1.3_lightgbm_F2.csv` | 13f4f8b6… | Uploaded by Sam at about 04:10 PDT (Sam at 08:10 PDT: "Uploaded 4 hours ago"), Kaggle description `2026-10-29_p1.3_lightgbm_F2`, as Sam reported it; the sidecar JSON keeps the prepared description unchanged. Accepted. Purpose, recorded before the score: verify the end-to-end submission pipeline |
| 2026-10-29 | `2026-10-29_p1.3_lightgbm_F2.csv` | 13f4f8b6… | Public leaderboard AUC 0.83597, reported by Sam. A pipeline check, not a selection signal: mean fold CV AUC 0.8315, and the public subset's size is unknown |
