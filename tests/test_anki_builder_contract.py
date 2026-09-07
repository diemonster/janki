from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from japanese_anki import status
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import anki, pattern_cards
from japanese_anki.io import load_records, load_structured

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


def test_public_build_seams_resolve_directions_and_configured_output(
    tmp_path: Path,
) -> None:
    config = replace(
        ProjectConfig.load(PROJECT_ROOT),
        dist_dir=tmp_path / "dist",
        default_cards={
            "recognition": True,
            "production": False,
            "reading": False,
        },
    )
    deck_path = tmp_path / "decks" / "lesson.yaml"

    assert anki.resolve_card_types(
        {"cards": {"recognition": False, "production": True, "reading": True}},
        config,
    ) == ["production", "reading"]
    assert anki.resolve_deck_output_path(
        deck_path, {"output": "lessons/../week-1.apkg"}, config
    ) == (tmp_path / "dist" / "week-1.apkg").resolve()
    assert anki.resolve_deck_output_path(deck_path, {}, config) == (
        tmp_path / "dist" / "lesson.apkg"
    ).resolve()


def test_builder_uses_the_public_direction_and_output_resolvers(
    tmp_path: Path, monkeypatch
) -> None:
    fake = SimpleNamespace(
        Model=FakeModel,
        Deck=FakeDeck,
        Note=FakeNote,
        Package=FakePackage,
        guid_for=lambda value: f"guid:{value}",
    )
    monkeypatch.setattr(anki, "genanki", fake)
    configured_output = tmp_path / "chosen" / "by-resolver.apkg"
    direction_calls: list[tuple[dict[str, object], ProjectConfig]] = []
    output_calls: list[tuple[Path, dict[str, object], ProjectConfig]] = []

    def directions(
        deck_config: dict[str, object], project_config: ProjectConfig
    ) -> list[str]:
        direction_calls.append((deck_config, project_config))
        return ["reading"]

    def output_path(
        deck_path: Path,
        deck_config: dict[str, object],
        project_config: ProjectConfig,
    ) -> Path:
        output_calls.append((deck_path, deck_config, project_config))
        return configured_output

    monkeypatch.setattr(anki, "resolve_card_types", directions)
    monkeypatch.setattr(anki, "resolve_deck_output_path", output_path)
    config = ProjectConfig.load(PROJECT_ROOT)
    deck_path = PROJECT_ROOT / "data/decks/verbs.yaml"

    result = anki.build_deck(deck_path, config)

    assert result.card_types == ("reading",)
    assert result.output_path == configured_output.resolve()
    assert len(direction_calls) == 1
    assert direction_calls[0][1] is config
    assert output_calls == [(deck_path.resolve(), direction_calls[0][0], config)]


def test_word_decks_are_nonempty_and_do_not_share_stable_ids() -> None:
    config = ProjectConfig.load(PROJECT_ROOT)
    paths = [
        path
        for path in status.deck_files(config)
        if anki.deck_kind(path) in {"", "vocabulary"}
    ]
    # Named, not counted. `>= 2` is satisfied by a subset, so a deck that
    # stopped being a word deck would leave this gate silently while the
    # assertion still passed. Not a *typo* in `kind:` — `deck_kind` already
    # refuses an unknown one — but the valid-but-wrong cases it cannot catch:
    # `verbs.yaml` gaining `kind: pattern`, or a deck blanking its kind. Both
    # measured. The deleted `deck-membership-partition` case pinned the exact
    # list; this is that half, kept, and the cost is one line to update when a
    # deck is legitimately added.
    assert {path.relative_to(config.root).as_posix() for path in paths} == {
        "data/decks/104-week-1-2.yaml",
        "data/decks/104-week-8.yaml",
        "data/decks/104-week-11.yaml",
        "data/decks/201-week-2.yaml",
        "data/decks/brandon-japanese-genki-ii-lesson-13-vocabulary-4689c50439.yaml",
        "data/decks/kanji-practice-112-123.yaml",
        "data/decks/medical-conditions-vocab.yaml",
        "data/decks/m7-camera-vertical-dialogue.yaml",
        "data/decks/m7-mixed-tsumori.yaml",
        "data/decks/m7-native-teform-table.yaml",
        "data/decks/verbs.yaml",
        "data/decks/yotsuba.yaml",
    }, [path.as_posix() for path in paths]
    memberships: dict[str, set[str]] = {}
    for path in paths:
        name = path.relative_to(config.root).as_posix()
        _, records = anki.resolve_deck_records(path)
        memberships[name] = {record.id for record in records}
        assert memberships[name], name

    for left, right in combinations(memberships, 2):
        assert memberships[left].isdisjoint(memberships[right]), f"{left} <> {right}"

    # And every record is in one. Disjointness alone was satisfied by decks
    # that between them cover nothing, and the collection was a partition only
    # because `verbs.yaml` was a catch-all: every record minus eight
    # exclusions, so anything untagged fell into it. That made "Starter Verbs"
    # the destination for the next noun imported without a source tag, which
    # is why it now selects by tag instead — and why the property it was
    # accidentally providing has to be asserted rather than arranged.
    #
    # A record in no deck is not an error janki can see: it builds, it
    # promotes, it gets audio, and it never reaches Anki.
    from japanese_anki.io import load_records

    everything = {record.id for record in load_records(config.normalized_file)}
    covered = set().union(*memberships.values())
    assert not (everything - covered), sorted(everything - covered)


def test_genki_lesson_13_potential_drill_has_its_exact_study_scope() -> None:
    config = ProjectConfig.load(PROJECT_ROOT)
    deck_path = (
        PROJECT_ROOT
        / "data/decks/brandon-japanese-genki-ii-lesson-13-potential-pr-221074117d.yaml"
    )
    records = load_records(pattern_cards.collection_for(deck_path, config))
    selected = {
        "word:話す:はなす",
        "word:まつ:まつ",
        "word:およぐ:およぐ",
        "word:行く:いく",
        "word:しぬ:しぬ",
        "word:読む:よむ",
        "word:あそぶ:あそぶ",
        "word:買う:かう",
        "word:見る:みる",
        "word:来る:くる",
        "word:する:する",
        "word:書く:かく",
        "word:飼う:かう",
        "word:弾く:ひく",
        "word:吹く:ふく",
        "word:たたく:たたく",
    }
    section = load_structured(deck_path)["deck"]

    shipped = {
        record.id for record in pattern_cards.shipping_records(deck_path, records)
    }

    assert set(section["include_ids"]) == selected
    assert set(section["drill_examples"]) == selected
    assert sum(len(examples) for examples in section["drill_examples"].values()) == 32
    for examples in section["drill_examples"].values():
        assert [example["register"] for example in examples] == ["polite", "casual"]
        assert all(example.get("audio", "").startswith("audio/janki-") for example in examples)
        assert all((config.media_dir / example["audio"]).is_file() for example in examples)
    assert shipped == selected
    assert pattern_cards.deck_problems(deck_path, {}, config) == []


def _real_word_selections(config: ProjectConfig) -> dict[str, anki.DeckSelection]:
    selections: dict[str, anki.DeckSelection] = {}
    for path in status.deck_files(config):
        if anki.deck_kind(path) not in {"", "vocabulary"}:
            continue
        deck_config, _records = anki.resolve_deck_records(path)
        name = path.relative_to(config.root).as_posix()
        selections[name] = anki.deck_selection(deck_config, path)
    return selections


def test_real_word_decks_define_unique_intake_tags() -> None:
    selections = _real_word_selections(ProjectConfig.load(PROJECT_ROOT))
    assignments = {
        name: selection.intake_tag for name, selection in selections.items()
    }

    assert all(tag is not None for tag in assignments.values()), assignments
    assert len(set(assignments.values())) == len(assignments), assignments


def test_each_real_intake_tag_selects_only_its_owner() -> None:
    selections = _real_word_selections(ProjectConfig.load(PROJECT_ROOT))

    for owner, owner_selection in selections.items():
        tag = owner_selection.intake_tag
        assert tag is not None, owner
        probe = SimpleNamespace(id="word:assignment:assignment", tags=[tag])
        claimers = {
            name for name, selection in selections.items() if selection.includes(probe)
        }
        assert claimers == {owner}, f"{tag!r} is selected by {sorted(claimers)}"


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


def test_deck_files_rejects_duplicate_stems(tmp_path: Path) -> None:
    deck_dir = tmp_path / "decks"
    left = deck_dir / "pilots" / "verbs.yaml"
    right = deck_dir / "archive" / "Verbs.yml"
    left.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    left.touch()
    right.touch()

    with pytest.raises(JankiError, match=r"duplicate stems: verbs") as error:
        status.deck_files(SimpleNamespace(deck_dir=deck_dir))

    assert str(left) in str(error.value)
    assert str(right) in str(error.value)
