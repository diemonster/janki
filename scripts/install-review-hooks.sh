#!/usr/bin/env bash
# Install the repository's tracked hook shims (Git LFS plus review) without
# replacing custom hooks. JANKI_REPO_ROOT exists for the test harness; ordinary
# callers omit it.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${JANKI_REPO_ROOT:-$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)}"
HOOK_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-path hooks)" || exit 1
SOURCE_DIR="$SCRIPT_DIR/git-hooks"
HOOKS="post-checkout post-commit post-merge pre-push"

mkdir -p "$HOOK_DIR"

# Exactly what `git lfs install` writes (git-lfs 3.7.1). Generated rather than
# stored so recognition stays byte-exact: a hook that merely looks like this
# one, or has anything appended to it, is somebody's own work and is kept.
stock_lfs_hook() {
  printf '%s\n' \
    '#!/bin/sh' \
    "command -v git-lfs >/dev/null 2>&1 || { printf >&2 \"\\n%s\\n\\n\" \"This repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting the '$1' file in the hooks directory (set by 'core.hookspath'; usually '.git/hooks').\"; exit 2; }" \
    "git lfs $1 \"\$@\""
}

CONFLICTS=0
for hook in $HOOKS; do
  target="$HOOK_DIR/$hook"
  [ -e "$target" ] || [ -L "$target" ] || continue
  cmp -s "$SOURCE_DIR/$hook" "$target" && continue
  grep -Fq 'Managed by scripts/install-review-hooks.sh.' "$target" 2>/dev/null && continue
  # Migrate the local-only hooks that existed before M6.7.
  grep -Fq '.claude/hooks/janki-review.sh' "$target" 2>/dev/null && continue
  # A stock `git lfs install` hook does strictly less than our shim does.
  stock_lfs_hook "$hook" | cmp -s - "$target" && continue

  echo "Review hook not installed: $target already contains an unmanaged hook." >&2
  echo "Merge $SOURCE_DIR/$hook into it by hand, then rerun this installer." >&2
  echo "  That shim runs both 'git lfs $hook' and janki's review step; combine" >&2
  echo "  the two deliberately so neither is silently dropped." >&2
  CONFLICTS=1
done

# Refuse the whole installation before changing any hook. A half-installed
# set is harder to reason about than an explicit conflict.
[ "$CONFLICTS" -eq 0 ] || exit 1

for hook in $HOOKS; do
  install -m 0755 "$SOURCE_DIR/$hook" "$HOOK_DIR/$hook"
done

echo "Installed janki hooks (Git LFS + review) in $HOOK_DIR"
