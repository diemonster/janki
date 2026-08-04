from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("genanki")

from japanese_anki.config import ProjectConfig
from japanese_anki.exporters.anki import build_deck


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_builds_importable_package(tmp_path: Path) -> None:
    config = ProjectConfig.load(PROJECT_ROOT)
    output = tmp_path / "verbs.apkg"
    result = build_deck(PROJECT_ROOT / "data/decks/verbs.yaml", config, output)

    assert result.note_count == 3
    assert result.card_types == ("recognition", "production")
    assert output.exists()
    with ZipFile(output) as archive:
        assert "collection.anki2" in archive.namelist()
