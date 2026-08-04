#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

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

ruff check .
pytest
janki validate data/decks/verbs.yaml
janki build data/decks/verbs.yaml --output dist/sample-verbs.apkg

cat <<'MSG'

Bootstrap complete.

Activate the environment with:
  source .venv/bin/activate

Try:
  janki inspect tests/fixtures/shirabe-sample.csv
  janki import-shirabe tests/fixtures/shirabe-sample.csv
  janki build data/decks/personal-vocabulary.yaml
MSG
