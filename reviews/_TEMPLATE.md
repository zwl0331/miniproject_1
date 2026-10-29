# Review record: <task slug>

Date opened: YYYY-MM-DD. Author: <model, session>. Reviewer: <model, agent>.

## Round 1: review request
Task: <what was asked, quoted or closely paraphrased>
Author: <model, session>
Revision: <fingerprint> (tools/fingerprint.py from the project root, or how else computed)
Files changed or created: <list>
Intended action: <exact command to be executed after approval, or none>
What was done: <short description>
How it was verified: <commands run and key output>
Known limitations and open questions: <list, or "none">

## Round 1: verdict
Reviewer: <model, agent>
Revision reviewed: <fingerprint>
What I checked and how: <list, with commands run and their results>
Findings:
  F1 [blocking | non-blocking] <file:line or location>. <problem>. <what would resolve it>
Notes: <deferred non-blocking items, recorded disagreements>
VERDICT: APPROVED | CHANGES REQUESTED

Reviewer run: <appended by tools/codex_review.sh: CLI version, model, effort, sandbox, thread id for later rounds>

## Round 2: review request
Revision: <new fingerprint>
Response to previous findings:
  F1: FIXED | REBUTTED | DEFERRED. <details>
<other fields as in round 1 if they changed>

## Round 2: verdict
<as in round 1>

## Convergence
Approved revision: <fingerprint>
Delivered revision: <fingerprint, must match>
Human decisions: <waivers or overrides, verbatim, or "none">

## Escalation (only if the loop stopped without convergence)
Task: ...
Rounds completed: <N>
Unresolved: <finding; author position; reviewer position>
Decision needed from Sam: ...

## Waiver (only if Sam waived review for this task)
Task: ...
Sam's wording, verbatim: "..."
Date: YYYY-MM-DD
