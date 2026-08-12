#!/usr/bin/env bash
# Install the repository's tracked review-hook shims without replacing custom
# hooks. JANKI_REPO_ROOT exists for the test harness; ordinary callers omit it.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${JANKI_REPO_ROOT:-$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)}"
HOOK_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-path hooks)" || exit 1
SOURCE_DIR="$SCRIPT_DIR/git-hooks"

mkdir -p "$HOOK_DIR"

CONFLICTS=0
for hook in post-commit pre-push; do
  target="$HOOK_DIR/$hook"
  [ -e "$target" ] || [ -L "$target" ] || continue
  cmp -s "$SOURCE_DIR/$hook" "$target" && continue
  grep -Fq 'Managed by scripts/install-review-hooks.sh.' "$target" 2>/dev/null && continue
  # Migrate the local-only hooks that existed before M6.7.
  grep -Fq '.claude/hooks/janki-review.sh' "$target" 2>/dev/null && continue

  echo "Review hook not installed: $target already contains an unmanaged hook." >&2
  echo "Merge $SOURCE_DIR/$hook into it by hand, then rerun this installer." >&2
  CONFLICTS=1
done

# Refuse the whole installation before changing either hook. A half-installed
# pair is harder to reason about than an explicit conflict.
[ "$CONFLICTS" -eq 0 ] || exit 1

for hook in post-commit pre-push; do
  install -m 0755 "$SOURCE_DIR/$hook" "$HOOK_DIR/$hook"
done

echo "Installed janki review hooks in $HOOK_DIR"
