from __future__ import annotations

from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

from japanese_anki import status
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters import anki

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(self, model_id, name, **kwargs):
        self.model_id = model_id
        self.name = name
        self.fields = kwargs["fields"]
        self.templates = kwargs["templates"]
        self.css = kwargs["css"]
        self.sort_field_index = kwargs["sort_field_index"]


class FakeDeck:
    def __init__(self, deck_id, name):
        self.deck_id = deck_id
        self.name = name
        self.description = ""
        self.notes = []

    def add_note(self, note) -> None:
        self.notes.append(note)


class FakeNote:
    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.fields = kwargs["fields"]
        self.tags = kwargs["tags"]
        self.guid = kwargs["guid"]


class FakePackage:
    def __init__(self, deck):
        self.deck = deck
        self.media_files = []

    def write_to_file(self, output: str) -> None:
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("collection.anki2", b"test-double")
            archive.writestr("media", "{}")


def test_builder_wires_templates_fields_and_stable_guids(tmp_path, monkeypatch) -> None:
    fake = SimpleNamespace(
        Model=FakeModel,
        Deck=FakeDeck,
        Note=FakeNote,
        Package=FakePackage,
        guid_for=lambda value: f"guid:{value}",
    )
    monkeypatch.setattr(anki, "genanki", fake)

    output = tmp_path / "verbs.apkg"
    config = ProjectConfig.load(PROJECT_ROOT)
    result = anki.build_deck(
        PROJECT_ROOT / "data/decks/verbs.yaml",
        config,
        output,
    )

    # From the deck's own membership — see the note in test_anki_build.py.
    _, expected = anki.resolve_deck_records(PROJECT_ROOT / "data/decks/verbs.yaml")
    assert result.note_count == len(expected)
    assert result.card_types == ("recognition", "production")
    assert output.exists()
    with ZipFile(output) as archive:
        assert "collection.anki2" in archive.namelist()


def test_word_decks_are_nonempty_and_do_not_share_stable_ids() -> None:
    config = ProjectConfig.load(PROJECT_ROOT)
    paths = [
        path
        for path in status.deck_files(config)
        if anki.deck_kind(path) in {"", "vocabulary"}
    ]
    assert len(paths) >= 2, [path.as_posix() for path in paths]
    memberships: dict[str, set[str]] = {}
    for path in paths:
        name = path.relative_to(config.root).as_posix()
        _, records = anki.resolve_deck_records(path)
        memberships[name] = {record.id for record in records}
        assert memberships[name], name

    for left, right in combinations(memberships, 2):
        assert memberships[left].isdisjoint(memberships[right]), f"{left} <> {right}"


def test_deck_files_discovers_nested_yaml(tmp_path: Path) -> None:
    deck_dir = tmp_path / "decks"
    nested = deck_dir / "pilots" / "camera.yaml"
    nested.parent.mkdir(parents=True)
    nested.write_text("name: Camera\n", encoding="utf-8")
    top_level = deck_dir / "verbs.yml"
    top_level.write_text("name: Verbs\n", encoding="utf-8")
    (deck_dir / "README.md").write_text("not a deck\n", encoding="utf-8")

    config = SimpleNamespace(deck_dir=deck_dir)

    assert status.deck_files(config) == [nested, top_level]
