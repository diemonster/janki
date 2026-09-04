"""One YAML loader, named once.

libyaml parses the 570 KB staging files the Assistant catalog reads on every
turn 8x faster than PyYAML's pure-Python scanner, and the saving is worth
nothing if a single call site keeps the slow default. Two loaders would also be
two sets of parse semantics for the same repository files, so the rule enforced
here is stronger than "use the fast one": exactly one module names a loader.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from japanese_anki import io
from japanese_anki.io import load_structured

REPO_ROOT = Path(__file__).resolve().parents[1]

# `CSafeLoader` matches before `SafeLoader` would, so each name is reported as
# the one that was actually written.
LOADER_NAME = re.compile(r"safe_load\(|CSafeLoader|SafeLoader")


def test_structured_yaml_reads_use_the_c_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`load_structured` is the ~190-reads-per-catalog path; its default decides."""
    path = tmp_path / "deck.yaml"
    path.write_text("deck:\n  name: Sample\n", encoding="utf-8")
    seen: list[object] = []
    real = yaml.load

    def spy(stream: object, Loader: object) -> object:  # noqa: N803 - PyYAML's name
        seen.append(Loader)
        return real(stream, Loader=Loader)

    monkeypatch.setattr(yaml, "load", spy)

    assert load_structured(path) == {"deck": {"name": "Sample"}}
    assert seen == [yaml.CSafeLoader]


def test_a_pyyaml_without_libyaml_is_refused_at_import_with_the_fix() -> None:
    """The one loader is a requirement, and a missing one reads as a requirement.

    On a PyYAML built without libyaml, `yaml.CSafeLoader` is absent rather
    than slow, so every janki entry point would die with an `AttributeError`
    from a module-level line nobody wrote, naming a class no user has heard
    of. The guard runs at import for the same reason: `cli.py` reads this
    module long before `main()` could report anything.
    """
    with pytest.raises(SystemExit, match="libyaml"):
        io._require_libyaml(False)

    assert io._require_libyaml(True) is None


def test_src_names_one_yaml_loader() -> None:
    """io.py defines `YAML_LOADER`; nothing else in src/ may name a loader at all.

    A call site that reaches for `yaml.safe_load` is not a style lapse — it is a
    second parser for the same files, chosen by whoever typed fastest.
    """
    source_root = REPO_ROOT / "src" / "japanese_anki"
    named: dict[str, list[str]] = {}
    for path in sorted(source_root.rglob("*.py")):
        found = LOADER_NAME.findall(path.read_text(encoding="utf-8"))
        if found:
            named[path.relative_to(source_root).as_posix()] = sorted(set(found))

    assert named == {"io.py": ["CSafeLoader"]}
