# Claude's positions on the identifier probe (written before the round M prompt was sent)

Written 2026-10-29, about 23:42 PDT. Sam asked whether all features, identifiers included, are used, worried about leaving performance on the table; after my argued answer, Sam said, verbatim: "Yeah, don't assume anything. Use tests and actual data to draw conclusions."

## Questions, each answered by a test on the provided data

| Q | Question | Test |
|---|---|---|
| 1 | Do the `uri` tokens carry label signal? | Features derived from the token only (the 22 base-62 characters as position-wise codes; the decoded 128-bit integer's leading bits as a number); LightGBM with the anchor parameters on those features alone, the saved primary folds, three split seeds; mean fold AUC against a null from 20 group-level label permutations through the same pipeline. Then F2 plus the token features against F2 alone, paired cluster bootstrap |
| 2 | Are `track_href`, `analysis_url`, `type` redundant? | Re-check on every row that the two URLs are `https://api.spotify.com/v1/...` strings ending in the `uri` token and `type` is constant (Finding 002 did this; repeat it in the probe's own output) |
| 3 | Does row order carry signal, nonmonotone included? | The training row position as a feature: alone (LightGBM, same folds, permutation null) and added to F2 (paired bootstrap). Its relation to release year (to see whether the file is sorted). Test ID cannot be validated against labels; report its relation to the test release year and to the audio features only |
| 4 | Does repetition carry signal? | Within train, High rate and univariate AUC by `uri` repeat count and by primary group size; F2 plus the repeat count against F2 (grouped, so a group's rows share the value). Report the train and test distributions of the within-file count. Using the test file's own counts at prediction time is transductive and outside the strategy's "nothing fitted on test rows" rule; that decision stays Sam's |
| 5 | Is there any identifier overlap between the files? | `uri` overlap (Finding 002: none), audio-signature overlaps, exact-date overlap (72.2%, Finding 005) |

## Design rules

- Declared before any probe score. Grouped folds only; nothing fitted on test rows. The same LightGBM settings as the anchor, so differences come from the features.
- A null for every "alone" test (permutation), a paired comparison for every "added" test, repeat-level differences reported.
- Interim results in the scratchpad, labelled unreviewed; then a reviewed task (script in `analysis/`, its output, Finding 013), after P2.2's completion review so the two fingerprints stay separate.
- If any identifier feature shows signal that survives the null and the paired comparison, it becomes a proposed feature block for P2.4 through a reviewed amendment; the engine excludes identifiers by design today (`RAW_COLUMNS`), so that would be a reviewed engine change.

## Other assumptions in the plan worth testing (to flag to Sam)

- Section 6.6: "Audio features are not removed because their marginal AUC is near 0.5": a removal check is cheap.
- The exclusion of the incomplete training row (Finding 001), outside the default scope.
- kNN on F1 without date parts beyond the year.
