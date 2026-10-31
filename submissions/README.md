# Submissions

Every file here is written once by `../scripts/prepare_submission.py` (through `songpop.artifacts.write_submission`: exclusive create, read back, validated), next to a sidecar JSON with its provenance and its Kaggle description. Each submission file gets an R1 review before Sam uploads it (`../AGENTS.md`; Sam's decision of 2026-10-30 in `../decisions.md`; before then, a gated action under `../doc/review_policy.md` section 4.3). Sam uploads every file and reports the public score.

## Log

Events, newest last. A file reviewed and ready for Sam, a file uploaded, and a public score are separate events.

| Date | File | sha256 | Event |
|---|---|---|---|
| 2026-10-29 | `2026-10-29_p1.3_lightgbm_F2.csv` | 13f4f8b6… | Prepared in P1.3 (recipe `lightgbm_F2`, run `production_lightgbm_F2_9c37e77d549e`, mean fold AUC 0.8315); in review (`../reviews/2026-10-29-p1.3-first-submission.md`) |
| 2026-10-29 | `2026-10-29_p1.3_lightgbm_F2.csv` | 13f4f8b6… | Approved (P1.3, round 3) and committed (f6add21); ready for Sam's upload |
| 2026-10-29 | `2026-10-29_p1.3_lightgbm_F2.csv` | 13f4f8b6… | Uploaded by Sam at about 04:10 PDT (Sam at 08:10 PDT: "Uploaded 4 hours ago"), Kaggle description `2026-10-29_p1.3_lightgbm_F2`, as Sam reported it; the sidecar JSON keeps the prepared description unchanged. Accepted. Purpose, recorded before the score: verify the end-to-end submission pipeline |
| 2026-10-29 | `2026-10-29_p1.3_lightgbm_F2.csv` | 13f4f8b6… | Public leaderboard AUC 0.83597, reported by Sam. A pipeline check, not a selection signal: mean fold CV AUC 0.8315, and the public subset's size is unknown |
| 2026-10-30 | `2026-10-30_p2.2_lightgbm_best.csv` | b35b54e6… | Prepared in P2.2 by `scripts.prepare_candidate` from `../artifacts/phase2/selections/p22_lightgbm_best.json` (run `p2_p22_lightgbm_b3_001_seeds155-2026-1029_3c98717cc491`, mean fold AUC 0.8339, Finding 011); in review (`../reviews/2026-10-29-p2.2-searches.md`). Purpose, recorded before any score: exercise the phase 2 fitting and submission path and calibrate how the public leaderboard moves with grouped CV |
| 2026-10-30 | `2026-10-30_p2.2_lightgbm_best.csv` | b35b54e6… | Uploaded by Sam (reported 2026-10-30, about 22:30 PDT); public leaderboard AUC 0.84343, as Sam reported it: "I submitted 2026-10-30_p2.2_lightgbm_best.csv and got 0.84343 on the public leadboard". Mean fold CV AUC 0.8339 against P1.3's 0.8315 (public 0.83597): the public score moved +0.0075 for a CV gain of +0.0024, in the same direction; the public subset's size is unknown, so this is a calibration observation, not a selection signal |
