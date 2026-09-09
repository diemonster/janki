from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from japanese_anki import status
from japanese_anki.application import deck_creation
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


def _discover_word_decks(config: ProjectConfig) -> list[Path]:
    """Every shipped deck file a *word* can land in, discovered rather than listed.

    `status.deck_files` plus `anki.deck_kind` is the same pair `janki status`
    and the builder use, so this cannot disagree with them about which files
    are word decks. That matters more than it sounds: the literal filename set
    this replaced had to be hand-edited whenever a deck was added, and what it
    asserted was the *names*. A deck that stopped being a word deck was
    indistinguishable from a deck that was legitimately renamed — both arrived
    as one line to update — so the gate was maintained rather than enforced.

    The properties asserted below are what the set was standing in for, and
    each of them survives a deck being added or renamed while still failing
    when a word deck silently leaves the collection: `verbs.yaml` gaining
    `kind: pattern` drops its records out of the coverage union. (A *typo* in
    `kind:` needs no test here — `deck_kind` refuses an unknown one.)
    """
    return [
        path
        for path in status.deck_files(config)
        if anki.deck_kind(path) in {"", "vocabulary"}
    ]


@pytest.fixture(scope="module")
def real_config() -> ProjectConfig:
    return ProjectConfig.load(PROJECT_ROOT)


@pytest.fixture(scope="module")
def word_deck_memberships(real_config: ProjectConfig) -> dict[str, set[str]]:
    """What each discovered word deck would actually ship, resolved once."""
    memberships: dict[str, set[str]] = {}
    for path in _discover_word_decks(real_config):
        _deck_config, records = anki.resolve_deck_records(path)
        name = path.relative_to(real_config.root).as_posix()
        memberships[name] = {record.id for record in records}
    return memberships


def test_every_shipped_word_deck_selects_at_least_one_record(
    word_deck_memberships: dict[str, set[str]],
) -> None:
    # Mutant: `DeckSelection.refusal` reads an empty tag set for every record
    # (`tags = set(record.tags)` becomes `tags: set[str] = set()`), so a deck
    # that selects by tag matches nothing, resolves to no records and writes an
    # empty package over its own output on exit 0.
    #
    # That mutant empties a tag-selecting deck structurally — it holds however
    # the shipped decks are tagged and whatever ids they list — which is what
    # this property is for. Inverting the include-tag test instead does not
    # empty a deck in general: `refusal` answers *why a deck declines a
    # record*, so dropping its `not` makes a deck take everything that lacks
    # its tag, and whether that empties anything depends on the collection.
    assert word_deck_memberships, "no word deck was discovered at all"
    empty = sorted(name for name, ids in word_deck_memberships.items() if not ids)
    assert not empty, empty


def test_no_two_word_decks_claim_the_same_stable_id(
    word_deck_memberships: dict[str, set[str]],
) -> None:
    # One record belongs to one word deck (DESIGN, "Compile"). The builder
    # filters each deck without looking at the others, so this convention is
    # only ever true because the deck files keep it — which is why it is
    # checked here over the real decks.
    #
    # Mutant: `DeckSelection.refusal` stops honouring `exclude_ids`, so two
    # decks whose tag filters overlap both claim the record one of them
    # explicitly gives up.
    for left, right in combinations(word_deck_memberships, 2):
        shared = word_deck_memberships[left] & word_deck_memberships[right]
        assert not shared, f"{left} <> {right}: {sorted(shared)}"


def test_every_canonical_record_reaches_a_word_deck(
    real_config: ProjectConfig, word_deck_memberships: dict[str, set[str]]
) -> None:
    # Disjointness alone was satisfied by decks that between them cover
    # nothing, and the collection was a partition only because `verbs.yaml`
    # was a catch-all: every record minus eight exclusions, so anything
    # untagged fell into it. That made "Starter Verbs" the destination for the
    # next noun imported without a source tag, which is why it now selects by
    # tag instead — and why the property it was accidentally providing has to
    # be asserted rather than arranged.
    #
    # A record in no deck is not an error janki can see: it builds, it
    # promotes, it gets audio, and it never reaches Anki.
    #
    # Mutant: `anki.deck_kind` answers something other than `""` for a deck
    # file that omits `kind:`, so every ordinary word deck drops out of
    # discovery and the records they ship are covered by nothing.
    everything = {record.id for record in load_records(real_config.normalized_file)}
    covered: set[str] = set().union(*word_deck_memberships.values())
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


def _real_word_selections(
    config: ProjectConfig,
) -> tuple[deck_creation._ExistingWordDeck, ...]:
    """The real word decks as deck creation itself reads them.

    Deck creation's own value type, so the probe helper below can be the
    production one rather than a copy of it.
    """
    decks: list[deck_creation._ExistingWordDeck] = []
    for path in _discover_word_decks(config):
        deck_config, _records = anki.resolve_deck_records(path)
        decks.append(
            deck_creation._ExistingWordDeck(
                path=path, selection=anki.deck_selection(deck_config, path)
            )
        )
    return tuple(decks)


def _deck_name(config: ProjectConfig, path: Path) -> str:
    return path.relative_to(config.root).as_posix()


def test_real_word_decks_define_unique_intake_tags(
    real_config: ProjectConfig,
) -> None:
    assignments = {
        _deck_name(real_config, deck.path): deck.selection.intake_tag
        for deck in _real_word_selections(real_config)
    }

    assert assignments, "no word deck was discovered at all"
    assert all(tag is not None for tag in assignments.values()), assignments
    assert len(set(assignments.values())) == len(assignments), assignments


def test_each_real_intake_tag_selects_only_its_owner(
    real_config: ProjectConfig,
) -> None:
    # One probe per deck, minted in that deck's own scope through the helper
    # deck creation uses when it refuses a new tag that would also select an
    # existing deck. The fixed `word:assignment:assignment` this replaced was
    # scope-blind, so a standalone deck refused its *own* probe — a shared
    # identity does not belong to a scoped collection at all — and the
    # exclusivity it was supposed to prove became "nobody claims it".
    #
    # Reusing `_cross_selection_probe` rather than restating the convention is
    # the point: a probe that drifts from the one creation uses would assert
    # exclusivity against an identity no assignment ever mints.
    #
    # Mutant: `_cross_selection_probe` stops passing `scope_id` to
    # `stable_record_id`, so the standalone deck's probe is a shared identity
    # its own scope refuses and no deck claims the tag.
    word_decks = _real_word_selections(real_config)
    assert word_decks, "no word deck was discovered at all"

    for owner in word_decks:
        tag = owner.selection.intake_tag
        assert tag is not None, owner.path
        probe = deck_creation._cross_selection_probe(
            word_decks, tag, owner.selection.scope_id
        )
        claimers = {
            _deck_name(real_config, deck.path)
            for deck in word_decks
            if deck.selection.includes(probe)
        }
        assert claimers == {_deck_name(real_config, owner.path)}, (
            f"{tag!r} is selected by {sorted(claimers)}"
        )


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
