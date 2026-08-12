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

# Wall-clock cap, seconds. An Opus 5 max-effort review normally lands well
# inside this; the cap exists so a hung API call can never wedge a push forever.
# JANKI_REVIEW_TIMEOUT overrides it so the timeout path can be exercised.
if [ "$MODE" = "gate" ]; then DEFAULT_TIMEOUT=900; else DEFAULT_TIMEOUT=1800; fi
TIMEOUT="${JANKI_REVIEW_TIMEOUT:-$DEFAULT_TIMEOUT}"

# --- guards: never break git over a review ------------------------------------

# A review is a Claude call per commit and per push. Keep the local marker under
# .claude/ so disabling the hook remains per-clone and can never be committed.
DISABLED="$REPO_ROOT/.claude/hooks/DISABLED"
if [ -f "$DISABLED" ]; then
  echo "janki review: disabled ($(head -1 "$DISABLED" 2>/dev/null))"
  echo "  re-enable with: rm .claude/hooks/DISABLED"
  exit 0
fi

command -v claude >/dev/null 2>&1 || exit 0

# Nothing to look at. Also covers empty commits and no-op pushes.
if git diff --quiet "$RANGE" 2>/dev/null; then
  exit 0
fi

mkdir -p "$REVIEW_DIR"

# --- prompt -------------------------------------------------------------------

read -r -d '' PROMPT <<PROMPT_EOF
Review the changes in the git range \`${RANGE}\`.

Use \`git diff ${RANGE}\` and \`git log --oneline ${RANGE}\` to see them. Read
\`AGENTS.md\` and the docs relevant to the changed area before judging the
diff; this repository has non-obvious invariants around durable source data,
note identity, staging, the ledger, and generated media.

Prioritize concrete correctness defects, especially data loss or silently
dropped input; unstable note IDs/GUIDs; schema, ledger, or staging round-trip
breakage; content-addressed media mistakes; CLI/API contract mismatches;
Japanese encoding, reading, and furigana errors; and tests that would pass
against the pre-change code. Verify every finding against surrounding code and
call sites. Do not report style or naming preferences.

Output GitHub-flavored markdown:

- A one-line summary of what the range changes.
- Then each finding as its own section: a \`path/to/file.py:LINE\` heading, the
  concrete failure (inputs or state that produce the wrong result), and the fix.
  Order by severity, worst first.
- Report only defects you can trace to specific lines. No style notes, no
  praise, no "consider" suggestions, no summary of things that are fine.

End your output with exactly one final line, nothing after it:

VERDICT: CLEAN
  ...if you found no defects, or
VERDICT: FINDINGS <n>
  ...where <n> is the number of findings above.
PROMPT_EOF

# --- run ----------------------------------------------------------------------

{
  echo "# Review: \`${RANGE}\`"
  echo
  echo "- Commit: \`$(git rev-parse --short HEAD 2>/dev/null)\` on \`$(git rev-parse --abbrev-ref HEAD 2>/dev/null)\`"
  echo "- Mode: ${MODE}"
  echo
  echo '---'
  echo
} >"$REPORT"

# The model is named explicitly rather than left to a local agent or alias.
# Keeping the review instructions above in this tracked script means a fresh
# clone does not depend on an ignored .claude/agents file.
claude -p "$PROMPT" \
  --model claude-opus-5 \
  --effort max \
  --permission-mode dontAsk \
  --allowedTools Read Glob Grep "Bash(git *)" "Bash(rg *)" "Bash(ls *)" \
  >>"$REPORT" 2>&1 </dev/null &
CLAUDE_PID=$!

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
  kill -0 "$CLAUDE_PID" 2>/dev/null && kill "$CLAUDE_PID" 2>/dev/null
) >/dev/null 2>&1 </dev/null &
WATCHDOG_PID=$!
# Drop the watchdog from the job table, or killing it below makes the shell
# print "Terminated: 15 (sleep ...)" into the middle of a push.
disown "$WATCHDOG_PID" 2>/dev/null || true

# Reaped with stderr closed: when the watchdog kills the reviewer, the shell
# announces it as "Terminated: 15 (claude -p ...)" — the whole command line,
# printed into the middle of a git push.
wait "$CLAUDE_PID" 2>/dev/null
STATUS=$?
kill "$WATCHDOG_PID" 2>/dev/null
wait "$WATCHDOG_PID" 2>/dev/null

# A review that fails has to say so. Without this the file simply has no verdict
# line, which looks exactly like a review that is still running.
REASON=""
if ! grep -qE '^VERDICT: (CLEAN|FINDINGS [0-9]+)' "$REPORT"; then
  if [ "$STATUS" -eq 143 ] || [ "$STATUS" -eq 124 ]; then
    REASON="timed out after ${TIMEOUT}s"
  elif [ "$STATUS" -ne 0 ]; then
    REASON="the reviewer exited ${STATUS}"
  else
    REASON="the reviewer ended without a verdict"
  fi
  {
    echo
    echo "_Review did not complete: ${REASON}._"
    echo
    echo "VERDICT: ERROR"
  } >>"$REPORT"
fi

# --- verdict ------------------------------------------------------------------

VERDICT="$(grep -oE '^VERDICT: (CLEAN|FINDINGS [0-9]+|ERROR)' "$REPORT" | tail -1)"

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
