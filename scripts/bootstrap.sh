#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Media first: a clone whose audio is still pointer files builds broken decks,
# and the gates below read that audio.
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is required: this repository stores its audio in Git LFS." >&2
  echo "  macOS:          brew install git-lfs" >&2
  echo "  Debian/Ubuntu:  sudo apt-get install git-lfs" >&2
  echo "Install it, then rerun ./scripts/bootstrap.sh" >&2
  exit 1
fi

# Filters for this clone only. --skip-repo leaves the hooks alone; ours are
# tracked composites that run git-lfs and janki's review together.
git lfs install --local --skip-repo

./scripts/install-review-hooks.sh

# Materialize every tracked object even in a clone that was fetched under a
# narrower include/exclude filter, which would otherwise leave pointer files.
git lfs pull --include="" --exclude=""

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11+ is required." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ is required; found {sys.version.split()[0]}")
PY

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

make gates

cat <<'MSG'

Bootstrap complete.

Activate the environment with:
  source .venv/bin/activate

Try:
  janki inspect tests/fixtures/shirabe-sample.csv
  janki import-shirabe tests/fixtures/shirabe-sample.csv
  janki build data/decks/personal-vocabulary.yaml
MSG
