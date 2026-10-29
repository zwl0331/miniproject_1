#!/usr/bin/env bash
# Independent Codex reviewer for the CS 155 Miniproject 1 project
# (doc/review_policy.md, sections 8.1 and 8.2).
#
# Runs `codex exec` non-interactively in a read-only sandbox with the reviewer model
# pinned, validates the verdict, appends it verbatim to the review record, and prints
# the Codex thread id for later rounds.
#
# Usage:
#   tools/codex_review.sh <record.md> <round> <fingerprint> [<thread id>]
#   tools/codex_review.sh check <verdict file> <round> <fingerprint>    (validate only)
# Environment:
#   CODEX_MODEL      reviewer model, default gpt-6-astra
#   CODEX_EFFORT     reasoning effort, default xhigh
#   CODEX_SANDBOX    read-only (default) | workspace-write   (section 8.1)
#   REVIEW_POINTERS  optional neutral task-specific pointers appended to the prompt
#   CODEX_HOME       Codex configuration directory, default ~/.codex
#   REVIEW_LOG_DIR   where the reviewer output, event log and stderr are kept, default
#                    .review_logs in the project root, ignored by git (TMPDIR if it cannot be created)
set -euo pipefail

usage() { sed -n '2,20p' "$0" >&2; exit 2; }

# A verdict is accepted only if its first line is exactly the requested round heading,
# it has exactly one "Revision reviewed:" line whose value is the fingerprint under
# review, and exactly one well-formed VERDICT line, last (section 6.2). Mentions of a
# fingerprint anywhere else do not bind.
check_verdict() {
  local f="$1" r="$2" fp="$3" first nrev n strict last
  if ! printf '%s' "$fp" | grep -qE '^[0-9a-f]{7,64}$'; then
    echo "rejected: fingerprint must be lowercase hex (got: $fp)" >&2; return 1
  fi
  first=$(grep -v '^[[:space:]]*$' "$f" | head -1 || true)
  if [ "$first" != "## Round $r: verdict" ]; then
    echo "rejected: the first line must be exactly '## Round $r: verdict' (got: ${first:0:80})" >&2; return 1
  fi
  nrev=$(grep -c '^Revision reviewed:' "$f" || true)
  if [ "$nrev" -ne 1 ]; then
    echo "rejected: need exactly one 'Revision reviewed:' line (found $nrev)" >&2; return 1
  fi
  if ! grep -qE "^Revision reviewed:[[:space:]]*\`?${fp}(\`|[[:space:]]|[.,;:)]|$)" "$f"; then
    echo "rejected: the 'Revision reviewed:' value is not the fingerprint under review ($fp)" >&2; return 1
  fi
  n=$(grep -c '^VERDICT:' "$f" || true)
  strict=$(grep -cE '^VERDICT: (APPROVED|CHANGES REQUESTED)$' "$f" || true)
  last=$(grep -v '^[[:space:]]*$' "$f" | tail -1 || true)
  if [ "$n" -ne 1 ] || [ "$strict" -ne 1 ]; then
    echo "rejected: need exactly one line 'VERDICT: APPROVED' or 'VERDICT: CHANGES REQUESTED' (found $n VERDICT lines, $strict well formed)" >&2
    return 1
  fi
  if ! printf '%s\n' "$last" | grep -qE '^VERDICT: (APPROVED|CHANGES REQUESTED)$'; then
    echo "rejected: the VERDICT line is not the last line" >&2
    return 1
  fi
  return 0
}

if [ "${1:-}" = "check" ]; then
  [ $# -eq 4 ] || usage
  if check_verdict "$2" "$3" "$4"; then
    echo "verdict format ok: $(grep '^VERDICT:' "$2")"
    exit 0
  fi
  exit 3
fi
[ $# -ge 3 ] || usage

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REC="$1"; ROUND="$2"; FP="$3"; THREAD="${4:-}"
SANDBOX="${CODEX_SANDBOX:-read-only}"
CODEX_MODEL="${CODEX_MODEL:-gpt-6-astra}"
CODEX_EFFORT="${CODEX_EFFORT:-xhigh}"
case "$REC" in /*) RECPATH="$REC" ;; *) RECPATH="$ROOT/$REC" ;; esac
[ -f "$RECPATH" ] || { echo "no such record: $RECPATH" >&2; exit 2; }
grep -q "^## Round $ROUND: review request" "$RECPATH" \
  || { echo "record has no '## Round $ROUND: review request' heading" >&2; exit 2; }

export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
LOGBASE="${REVIEW_LOG_DIR:-$ROOT/.review_logs}"
mkdir -p "$LOGBASE" 2>/dev/null || LOGBASE="${TMPDIR:-/tmp}"
TMP="$(mktemp -d "$LOGBASE/codex_review.XXXXXX")"
OUT="$TMP/verdict.md"; EVENTS="$TMP/events.jsonl"

PROMPT="You are the independent reviewer for a task in the CS 155 Miniproject 1 project. Read AGENTS.md and doc/review_policy.md sections 6 and 7 first.

Your job is to find what is wrong, missing, unverified, or out of scope. You are not asked to confirm that the work is correct.

Review record: $REC. Read the \"Round $ROUND: review request\" there; it names the task, the files, and how the author verified the work. If this is not round 1, read the entire record: every finding from earlier rounds stands until you rule on it, fixed or upheld.
Revision under review: $FP. Recompute it with the command given in the request; if it differs, say so in the verdict.

Inspect the actual files and outputs, not only the request. Run the verification yourself where the sandbox allows: tests, scripts, recomputation. Do not modify any project file.

Your final message must be exactly a verdict in the format of section 7.2 of the policy: it starts with the line \"## Round $ROUND: verdict\", names the revision fingerprint, puts Notes before the verdict line, and its last line is exactly \"VERDICT: APPROVED\" or \"VERDICT: CHANGES REQUESTED\", with nothing after it. Each finding needs a severity (blocking or non-blocking), a location (file and line or section), the problem, and what would resolve it. An APPROVED must list what you checked and how."

if [ -n "${REVIEW_POINTERS:-}" ]; then
  PROMPT="$PROMPT

Areas the author suggests examining (neutral pointers, not conclusions):
$REVIEW_POINTERS"
fi

# `codex exec resume` accepts neither -C nor -s: run from the project root so the
# session lookup matches, and set the sandbox through a config override instead.
# The model and reasoning effort are pinned on both paths so the reviewer does not
# follow changes to the interactive Codex default.
cd "$ROOT"
set +e
if [ -n "$THREAD" ]; then
  codex exec resume "$THREAD" -m "$CODEX_MODEL" -c "model_reasoning_effort=\"$CODEX_EFFORT\"" \
    -c "sandbox_mode=\"$SANDBOX\"" --skip-git-repo-check --json \
    -o "$OUT" "$PROMPT" < /dev/null > "$EVENTS" 2> "$TMP/stderr.txt"
else
  codex exec -C "$ROOT" -s "$SANDBOX" -m "$CODEX_MODEL" -c "model_reasoning_effort=\"$CODEX_EFFORT\"" \
    --skip-git-repo-check --json \
    -o "$OUT" "$PROMPT" < /dev/null > "$EVENTS" 2> "$TMP/stderr.txt"
fi
STATUS=$?
set -e

if [ "$STATUS" -ne 0 ]; then
  echo "codex exec failed (exit $STATUS); see $TMP/stderr.txt" >&2
  tail -20 "$TMP/stderr.txt" >&2
  exit "$STATUS"
fi
NEWTHREAD="$(sed -n 's/.*"thread_id":"\([^"]*\)".*/\1/p' "$EVENTS" | head -1)"
[ -n "$NEWTHREAD" ] || NEWTHREAD="$THREAD"
if ! check_verdict "$OUT" "$ROUND" "$FP"; then
  echo "reviewer output rejected; nothing appended. Output kept at $OUT, events at $EVENTS" >&2
  exit 3
fi
{
  echo
  cat "$OUT"
  echo
  echo "Reviewer run: Codex CLI $(codex --version 2>/dev/null | awk '{print $2}'), model $CODEX_MODEL, effort $CODEX_EFFORT, sandbox $SANDBOX, thread $NEWTHREAD. Log: $EVENTS"
} >> "$RECPATH"
if [ "$SANDBOX" != "read-only" ]; then
  echo "NOTE: sandbox was $SANDBOX. Recompute the fingerprint and compare with $FP before accepting this verdict." >&2
fi
echo "Verdict appended to $REC (round $ROUND). Thread id: $NEWTHREAD"
grep '^VERDICT:' "$OUT"
