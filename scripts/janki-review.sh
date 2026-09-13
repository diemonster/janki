#!/usr/bin/env bash
#
# Code review at a git commit edge.
#
# Called by .git/hooks/post-commit (advisory, backgrounded) and
# .git/hooks/pre-push (blocking gate). Both hooks are thin shims; all of the
# behavior lives here so there is one tracked place to edit.
#
# Usage: janki-review.sh <advisory|gate> <git-range> <label>
#
# Writes a markdown report to .claude/reviews/<label>.md. In gate mode, exits
# non-zero when the review reports findings, which refuses the push.

set -uo pipefail

MODE="${1:?usage: janki-review.sh <advisory|gate> <range> <label>}"
RANGE="${2:?missing git range}"
LABEL="${3:?missing label}"

REPO_ROOT="$(git rev-parse --show-toplevel)" || exit 0
cd "$REPO_ROOT" || exit 0

REVIEW_DIR="$REPO_ROOT/.claude/reviews"
REPORT="$REVIEW_DIR/${LABEL}.md"

# Wall-clock cap, seconds, so a hung model call cannot wedge a push forever.
# JANKI_REVIEW_TIMEOUT overrides it so the timeout path can be exercised.
if [ "$MODE" = "gate" ]; then DEFAULT_TIMEOUT=900; else DEFAULT_TIMEOUT=1800; fi
TIMEOUT="${JANKI_REVIEW_TIMEOUT:-$DEFAULT_TIMEOUT}"

# --- guards: never break git over a review ------------------------------------

# A review is one selected-provider call per commit and per push. Keep the marker under
# .claude/ so disabling the hook remains per-clone and can never be committed.
DISABLED="$REPO_ROOT/.claude/hooks/DISABLED"
if [ -f "$DISABLED" ]; then
  echo "janki review: disabled ($(head -1 "$DISABLED" 2>/dev/null))"
  echo "  re-enable with: rm .claude/hooks/DISABLED"
  exit 0
fi

RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/janki-review.XXXXXX")" || {
  echo "janki review: NOTHING WAS REVIEWED — cannot create the review files." >&2
  exit 0
}
trap 'rm -rf "$RUN_DIR"' EXIT
PROMPT_FILE="$RUN_DIR/prompt.md"
DIFF_FILE="$RUN_DIR/diff.txt"
DIFF_ERROR_FILE="$RUN_DIR/diff-diagnostics.txt"
OUTPUT_FILE="$RUN_DIR/answer.txt"
ERROR_FILE="$RUN_DIR/diagnostics.txt"

# Code review and repository-content review are separate operations. A source,
# staged card, curated deck, generated media file, or operational ledger under
# data/ is never sent to the code reviewer merely because it shares a commit or
# push with implementation work. dist/ is generated output and equally outside
# this review. The leading `.` makes the exclusions a positive pathspec rather
# than relying on Git's implicit all-paths behaviour.
CODE_REVIEW_PATHSPEC=(
  "."
  ":(exclude)data/**"
  ":(exclude)dist/**"
)

# Supply the filtered diff ourselves: Claude's read-only role has no shell.
# Disable Git's external converters so preparing a review cannot run local code.
git --no-pager diff --no-ext-diff --no-textconv "$RANGE" -- \
  "${CODE_REVIEW_PATHSPEC[@]}" >"$DIFF_FILE" 2>"$DIFF_ERROR_FILE"
DIFF_STATUS=$?

# Empty commits, no-op pushes and content-only submissions never probe a model.
# A failed range lookup is an error, not an empty review scope.
if [ "$DIFF_STATUS" -eq 0 ] && [ ! -s "$DIFF_FILE" ]; then
  echo "janki code review: skipped ${RANGE} (repository-content-only change)"
  exit 0
fi

mkdir -p "$REVIEW_DIR"

# Provider selection is explicit; a refusal never tries another provider.
PROVIDER="${JANKI_REVIEW_PROVIDER:-codex}"
case "$PROVIDER" in
  claude) DEFAULT_MODEL="claude-opus-5" ;;
  codex) DEFAULT_MODEL="gpt-6-astra" ;;
  *) DEFAULT_MODEL="" ;;  # The launcher reports an unknown provider as a refusal.
esac
MODEL="${JANKI_REVIEW_MODEL:-$DEFAULT_MODEL}"
EFFORT="max"
LAUNCHER="$REPO_ROOT/scripts/llm.py"
PROMPT_TEMPLATE="$REPO_ROOT/prompts/development-code-review.md"

# --- run ----------------------------------------------------------------------

{
  echo "# Review: \`${RANGE}\`"
  echo
  echo "- Commit: \`$(git rev-parse --short HEAD 2>/dev/null)\` on \`$(git rev-parse --abbrev-ref HEAD 2>/dev/null)\`"
  echo "- Mode: ${MODE}"
  echo "- Provider: ${PROVIDER}"
  echo "- Model: ${MODEL}"
  echo "- Effort: ${EFFORT}"
  echo
  echo '---'
  echo
} >"$REPORT"

# Read the same tracked instructions for both providers on every run. The
# exact filtered diff is labelled input appended to the template.
run_review() {
  if [ "$DIFF_STATUS" -ne 0 ]; then
    cat "$DIFF_ERROR_FILE" >&2
    return "$DIFF_STATUS"
  fi
  cat "$PROMPT_TEMPLATE" >"$PROMPT_FILE" || return 1
  {
    printf '\n## Review input\n\nGit range: `%s`\n\n' "$RANGE"
    printf "Scoped diff command: \`git --no-pager diff --no-ext-diff --no-textconv %s -- . ':(exclude)data/**' ':(exclude)dist/**'\`\n" "$RANGE"
    printf '\nExact filtered diff:\n\n```diff\n'
    cat "$DIFF_FILE" || return 1
    printf '\n```\n'
  } >>"$PROMPT_FILE" || return 1
  # The launcher probes subscription auth and execs this provider's CLI under
  # the same guarded context. Read-only review permissions belong to it.
  exec "$LAUNCHER" --provider "$PROVIDER" --role review \
    --model "$MODEL" --effort "$EFFORT" --prompt-file "$PROMPT_FILE"
}
run_review >"$OUTPUT_FILE" 2>"$ERROR_FILE" </dev/null &
REVIEW_PID=$!

# Portable watchdog. macOS has no coreutils `timeout` by default. The trap is
# important: killing only the watchdog shell can leave its `sleep` child alive,
# holding a caller's output pipes open until the full timeout expires.
(
  SLEEP_PID=""
  stop_watchdog() {
    if [ -n "$SLEEP_PID" ]; then
      kill "$SLEEP_PID" 2>/dev/null
      wait "$SLEEP_PID" 2>/dev/null
    fi
    exit 0
  }
  trap stop_watchdog TERM INT
  sleep "$TIMEOUT" &
  SLEEP_PID=$!
  wait "$SLEEP_PID"
  kill -0 "$REVIEW_PID" 2>/dev/null && kill "$REVIEW_PID" 2>/dev/null
) >/dev/null 2>&1 </dev/null &
WATCHDOG_PID=$!
# Drop the watchdog from the job table, or killing it below makes the shell
# print "Terminated: 15 (sleep ...)" into the middle of a push.
disown "$WATCHDOG_PID" 2>/dev/null || true

# Reaped with stderr closed: when the watchdog kills the reviewer, the shell
# announces it as "Terminated: 15 (reviewer ...)" — the whole command line,
# printed into the middle of a git push.
wait "$REVIEW_PID" 2>/dev/null
STATUS=$?
kill "$WATCHDOG_PID" 2>/dev/null
wait "$WATCHDOG_PID" 2>/dev/null

# Keep complete diagnostics, but only the final-answer channel can decide the
# verdict. A CLI may echo the prompt (including example verdicts) on stderr.
if [ -s "$ERROR_FILE" ]; then
  {
    echo '## Reviewer diagnostics'
    echo
    cat "$ERROR_FILE"
    echo
    echo '---'
    echo
  } >>"$REPORT"
fi
cat "$OUTPUT_FILE" >>"$REPORT"

# Only a successful process with one exact final verdict completed a review.
# A partial reply may already contain CLEAN when the CLI subsequently fails.
VERDICT="$(awk '
  NF { last = $0 }
  /^VERDICT:/ { count++ }
  END {
    if (count == 1 && last ~ /^VERDICT: (CLEAN|FINDINGS [1-9][0-9]*)$/) print last
  }
' "$OUTPUT_FILE")"
REASON=""
if [ "$DIFF_STATUS" -ne 0 ]; then
  REASON="the requested Git range could not be read"
elif [ "$STATUS" -eq 143 ] || [ "$STATUS" -eq 124 ]; then
  REASON="timed out after ${TIMEOUT}s"
elif [ "$STATUS" -ne 0 ]; then
  REASON="the reviewer exited ${STATUS}"
elif [ -z "$VERDICT" ]; then
  REASON="the reviewer ended without one complete final verdict"
fi
if [ -n "$REASON" ]; then
  VERDICT="VERDICT: ERROR"
  {
    echo
    echo "_Review did not complete: ${REASON}._"
    echo
    echo "$VERDICT"
  } >>"$REPORT"
fi

# --- verdict ------------------------------------------------------------------

if [ "$MODE" != "gate" ]; then
  # Advisory. Say what happened and get out of the way; the commit already landed.
  if [ "$VERDICT" = "VERDICT: ERROR" ]; then
    echo "janki review [${RANGE}]: DID NOT RUN — ${REASON}. Re-run:" >&2
    echo "  scripts/janki-review.sh advisory ${RANGE} ${LABEL}" >&2
  elif [ -n "$VERDICT" ]; then
    echo "janki review [${RANGE}]: ${VERDICT#VERDICT: } -> ${REPORT#"$REPO_ROOT"/}"
  fi
  exit 0
fi

case "$VERDICT" in
  "VERDICT: CLEAN")
    echo "janki review: clean (${RANGE})"
    exit 0
    ;;
  "VERDICT: FINDINGS "*)
    echo
    echo "janki review: ${VERDICT#VERDICT: FINDINGS } finding(s) in ${RANGE} — push refused."
    echo "  ${REPORT#"$REPO_ROOT"/}"
    echo "  Push anyway with: git push --no-verify"
    echo
    exit 1
    ;;
  "VERDICT: ERROR")
    echo
    echo "janki review: NOTHING WAS REVIEWED (${RANGE}) — ${REASON}."
    echo "  ${REPORT#"$REPO_ROOT"/}"
    echo "  The push is allowed because a broken reviewer must not block one."
    echo "  To review it: scripts/janki-review.sh gate ${RANGE} ${LABEL}"
    echo
    exit 0
    ;;
  *)
    echo "janki review: no verdict in ${REPORT#"$REPO_ROOT"/}, allowing push"
    exit 0
    ;;
esac
