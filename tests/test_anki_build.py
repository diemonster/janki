import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("genanki")

from japanese_anki.config import ProjectConfig
from japanese_anki.exporters.anki import build_deck
from japanese_anki.models import VocabularyRecord

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


def test_audio_is_resolved_against_media_dir_not_the_deck(tmp_path: Path) -> None:
    """`janki audio` stores paths relative to the project's media_dir, which is
    what the config names. Resolving them against the deck's own directory
    looked equivalent while nothing had audio and stopped being equivalent the
    moment a real clip existed — the exporter went looking in data/decks/audio/
    for a file living in data/media/audio/."""
    root = tmp_path
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n',
        encoding="utf-8",
    )
    shutil.copytree(
        PROJECT_ROOT / "templates" / "japanese-study",
        root / "templates" / "japanese-study",
    )
    clip = root / "media" / "audio" / "janki-abc.wav"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFF....WAVEfake")
    record = VocabularyRecord(
        id="word:橋:はし",
        expression="橋",
        reading="はし",
        meanings=["bridge"],
        audio="audio/janki-abc.wav",
    )
    (root / "vocabulary.json").write_text(
        json.dumps([record.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    (root / "decks").mkdir()
    (root / "decks" / "d.yaml").write_text(
        'name: D\ndeck:\n  source: "../vocabulary.json"\n', encoding="utf-8"
    )

    result = build_deck(
        root / "decks" / "d.yaml",
        ProjectConfig.load(root),
        root / "out.apkg",
    )

    assert result.media_count == 1, "the clip was found and packaged"
