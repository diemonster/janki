"""Make pytest import *this* checkout, and never trust cached bytecode.

Two failure modes this file exists to make impossible. Both are silent —
they produce a green test run that proves nothing — so neither can be left
to whoever is running the suite to remember.

1. Wrong tree. The venv installs japanese_anki in editable mode pointing at
   the primary worktree. Parallel work here happens in linked worktrees, and
   without this file `pytest` inside one imports the *primary* worktree's
   source: the tests pass, and they tested code the branch never changed.
   Prepending this file's own src/ ties the import to the checkout the tests
   live in, whatever the venv believes.

2. Stale bytecode. CPython validates a .pyc against its source's mtime in
   whole seconds and its byte size. A mutation sweep's revert-then-reapply
   pair lands in the same second at the same size (`==` -> `!=`, `<` -> `<=`,
   swapped arguments), so Python reuses the previous mutant's bytecode: the
   new mutant never runs, its result is really the old one's, and an
   unguarded code path is reported as covered. Purging before the first
   import and refusing to write more means there is never a .pyc to go
   stale.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

sys.dont_write_bytecode = True

for _cache in _SRC.rglob("__pycache__"):
    shutil.rmtree(_cache, ignore_errors=True)

# Ahead of the editable install's path entry, not merely present in the list.
while str(_SRC) in sys.path:
    sys.path.remove(str(_SRC))
sys.path.insert(0, str(_SRC))

import japanese_anki  # noqa: E402  (must follow the sys.path edit above)

_imported = Path(japanese_anki.__file__).resolve().parent
_expected = (_SRC / "japanese_anki").resolve()
if _imported != _expected:
    raise RuntimeError(
        "pytest is about to test the wrong checkout: japanese_anki resolved to "
        f"{_imported}, expected {_expected}. Something imported the package "
        "before this conftest ran, so the whole suite would report on code that "
        "is not the code under test."
    )
