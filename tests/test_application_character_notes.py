"""One confirmed batch of character notes: what it plans, and what it writes.

The split under test is the whole design. ``prepare_character_notes`` may look
a character up and may ask the provider for facts nobody saved yet, and it
writes **nothing**. ``execute_character_notes`` writes exactly the bytes it was
shown, asks nobody anything, and refuses outright when a target file is neither
the state the plan saw nor the state the plan proposes.
``verify_character_notes_applied`` reads the same plan after the apply and
accepts only the state it proposed, because what a build publishes is whatever
those files say when it reads them.

Nothing here reaches the network: both lookups are replaced at the seam the
service reads them through at call time, and every test asserts on what was
asked for as well as on what landed.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from japanese_anki import jpdb_kanji, kanji, kanji_notes
from japanese_anki.application.character_notes import (
    CharacterNotesError,
    CharacterNotesPlan,
    execute_character_notes,
    prepare_character_notes,
    verify_character_notes_applied,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.identifiers import character_record_id
from japanese_anki.jpdb_kanji import (
    BoundExample,
    CharacterReadings,
    ReadingGroup,
    ReadingUsage,
)

TEMPLATES = Path(__file__).resolve().parents[1] / "templates" / "japanese-study"

#: The owner's explicit targets for this milestone. Five characters, five
#: notes, and — with the default direction — five cards.
TARGETS = ("物", "特", "鳥", "料", "理")


# --- the world outside, as it really answers ---------------------------------


def info(character: str, *, strokes: tuple[str, ...] = ("M10,10 L20,20",)) -> kanji.KanjiInfo:
    """One reference answer, shaped exactly like a kanjiapi/KanjiVG lookup."""
    return kanji.KanjiInfo(
        character=character,
        stroke_count=11,
        meanings=("logic", "reason"),
        readings=(
            kanji.Reading(kind="on", reading="リ"),
            kanji.Reading(kind="kun", reading="ことわり"),
        ),
        strokes=strokes,
    )


def example(written: str, pronounced: str) -> BoundExample:
    return BoundExample(
        written=written,
        pronounced=pronounced,
        gloss="reason",
        furigana=f"{written}[{pronounced}]",
        source_url=f"https://jpdb.io/vocabulary/1/{written}/{pronounced}",
    )


def readings(
    character: str, *, examples: tuple[BoundExample, ...] = (), sha: str = "b" * 64
) -> CharacterReadings:
    """One provider answer, with the bound examples a reading card needs."""
    return CharacterReadings(
        character=character,
        source_url=f"https://jpdb.io/kanji/{character}",
        fetched_at_utc="2026-09-07T00:00:00Z",
        sha256=sha,
        groups=(
            ReadingGroup(
                source_class=jpdb_kanji.COMMON_CELL_CLASS,
                readings=(
                    ReadingUsage(
                        label="リ",
                        href=f"/kanji-reading/{character}/リ",
                        percent_text="(84%)",
                        percent=84,
                        percent_less_than=False,
                        examples=examples,
                    ),
                ),
            ),
        ),
    )


class Sources:
    """Both lookups, answering from a script and recording every request."""

    def __init__(self, *, refuse: frozenset[str] = frozenset()) -> None:
        self.looked_up: list[str] = []
        self.fetched: list[tuple[str, bool]] = []
        self.refuse = set(refuse)

    def fetch_kanji(self, character: str, **_kwargs: object) -> kanji.KanjiInfo:
        self.looked_up.append(character)
        if character in self.refuse:
            raise kanji.KanjiError(f"kanjiapi said 404 for {character}")
        return info(character)

    def fetch_character(
        self, character: str, *, html_cache: Path, refresh: bool = False, **_kw: object
    ) -> CharacterReadings:
        self.fetched.append((character, refresh))
        if character in self.refuse:
            raise jpdb_kanji.KanjiReadingsError(f"request for {character} failed")
        return readings(
            character,
            examples=(example("料理", "りょうり"), example("理由", "りゆう")),
            sha=("c" if refresh else "b") * 64,
        )


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> Sources:
    answers = Sources()
    monkeypatch.setattr(kanji, "fetch_kanji", answers.fetch_kanji)
    monkeypatch.setattr(jpdb_kanji, "fetch_character", answers.fetch_character)
    return answers


# --- a project on disk -------------------------------------------------------


def project(tmp_path: Path) -> ProjectConfig:
    root = tmp_path / "repo"
    (root / "data" / "decks").mkdir(parents=True)
    (root / "janki.toml").write_text(
        "[paths]\n"
        f'template_dir = "{TEMPLATES}"\n'
        'kanji_notes_file = "data/kanji_notes.json"\n'
        'kanji_file = "data/kanji.json"\n'
        'jpdb_readings_file = "data/jpdb_readings.json"\n'
        f'jpdb_html_cache = "{tmp_path / "cache"}"\n',
        encoding="utf-8",
    )
    return ProjectConfig.load(root)


def word_deck(config: ProjectConfig, stem: str = "words") -> Path:
    path = config.deck_dir / f"{stem}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "kind": "vocabulary",
                    "name": "Words",
                    "deck_id": 1234500001,
                    "output": f"{stem}.apkg",
                    "source": "../../vocabulary.json",
                    "intake_tag": "janki:deck:words",
                    "include_tags": ["janki:deck:words"],
                }
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (config.root / "vocabulary.json").write_text("[]", encoding="utf-8")
    return path


def apply(config: ProjectConfig, plan: CharacterNotesPlan) -> object:
    return execute_character_notes(config, plan, expected_fingerprint=plan.fingerprint)


def seed(
    config: ProjectConfig, characters: tuple[str, ...], name: str = "Kanji", **kwargs
) -> Path:
    """Create one character deck and its notes, and answer where it landed.

    The file stem is derived from the display name, so no test may spell it.
    """
    plan = prepare_character_notes(config, characters, deck_name=name, **kwargs)
    apply(config, plan)
    return plan.deck_path


def stored_notes(config: ProjectConfig) -> dict[str, kanji_notes.CharacterNote]:
    return kanji_notes.load_notes(config.kanji_notes_file)


def bound_bytes(plan: CharacterNotesPlan) -> dict[str, bytes | None]:
    """Exactly what every file this batch bound holds right now.

    ``None`` for a file that is not there, so an absence and an empty file are
    not the same observation.
    """
    return {
        str(item.path): item.path.read_bytes() if item.path.exists() else None
        for item in plan.files
    }


# --- one batch, one confirmation ---------------------------------------------


def test_a_new_deck_and_the_notes_it_ships_are_one_plan_and_one_apply(
    tmp_path: Path, sources: Sources
) -> None:
    """A deck naming characters that do not exist yet is not a state anyone
    should have to see, so the deck file and the notes land together."""
    config = project(tmp_path)

    plan = prepare_character_notes(config, TARGETS, deck_name="Genki II Kanji")

    assert plan.deck_created is True
    assert plan.characters == TARGETS
    assert plan.note_count == 5
    assert plan.card_count == 5, "recognition only, by default"
    assert {item.label for item in plan.changed_files} == {
        "character notes",
        "kanji reference store",
        "jpdb reading facts",
        "character deck",
    }

    result = apply(config, plan)

    assert sorted(stored_notes(config)) == sorted(TARGETS)
    assert plan.deck_path.exists()
    assert result.record_ids == tuple(
        character_record_id(character) for character in TARGETS
    )
    assert result.note_count == 5
    assert result.card_count == 5


def test_preparing_writes_nothing_at_all(tmp_path: Path, sources: Sources) -> None:
    """The whole point of the split: a preview that leaves a repository it
    looked things up for is a preview nobody can decline."""
    config = project(tmp_path)

    plan = prepare_character_notes(config, TARGETS, deck_name="Genki II Kanji")

    assert sources.fetched, "it really did ask the provider"
    for path in (
        config.kanji_notes_file,
        config.kanji_file,
        config.jpdb_readings_file,
        plan.deck_path,
    ):
        assert not path.exists(), f"{path} was written by a preparation"


def test_only_the_characters_nobody_has_are_looked_up(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    deck = seed(config, ("理",))
    sources.looked_up.clear()
    sources.fetched.clear()

    plan = prepare_character_notes(config, ("理", "料"), deck_path=deck)

    assert sources.looked_up == ["料"], "理 was already in the reference store"
    assert [character for character, _refresh in sources.fetched] == ["料"]
    assert plan.looked_up == ("料",)
    assert plan.fetched_readings == ("料",)


def test_an_ordinary_rerun_asks_for_nothing_and_proposes_nothing(
    tmp_path: Path, sources: Sources
) -> None:
    """Saved facts are reused. A second preparation over the same characters
    costs no request and proposes no change to any file."""
    config = project(tmp_path)
    deck = seed(config, TARGETS, name="Genki II Kanji")
    sources.looked_up.clear()
    sources.fetched.clear()

    plan = prepare_character_notes(config, TARGETS, deck_path=deck)

    assert sources.looked_up == []
    assert sources.fetched == []
    assert plan.changed_files == ()
    assert apply(config, plan).changed == ()


def test_refresh_is_the_only_thing_that_asks_again(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    deck = seed(config, ("理",))
    sources.fetched.clear()

    prepare_character_notes(config, ("理",), deck_path=deck, refresh_readings=True)

    assert sources.fetched == [("理", True)]


# --- the notes themselves ----------------------------------------------------


def test_a_new_note_credits_kanjivg_and_its_licence(
    tmp_path: Path, sources: Sources
) -> None:
    """KanjiVG is CC BY-SA 3.0, and a deck that ships its strokes has to say so
    where a learner can read it."""
    config = project(tmp_path)

    plan = prepare_character_notes(config, ("理",), deck_name="Kanji")

    credit = " ".join(plan.notes[0].sources)
    assert "https://kanjivg.tagaini.net/" in credit
    assert "CC BY-SA 3.0" in credit
    assert "https://creativecommons.org/licenses/by-sa/3.0/" in credit


def test_a_note_with_no_strokes_credits_nobody_for_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sources: Sources
) -> None:
    config = project(tmp_path)
    monkeypatch.setattr(kanji, "fetch_kanji", lambda character, **_: info(
        character, strokes=()
    ))

    plan = prepare_character_notes(config, ("理",), deck_name="Kanji")

    assert "KanjiVG" not in " ".join(plan.notes[0].sources)


def test_a_refresh_moves_the_evidence_and_leaves_the_curated_note_alone(
    tmp_path: Path, sources: Sources
) -> None:
    """Meanings and the fixed reading prompt are curated values. A refresh of a
    provider's page is not an instruction to replace what a card asks."""
    config = project(tmp_path)
    deck = seed(config, ("理",))
    curated = kanji_notes.load_notes(config.kanji_notes_file)["理"]
    edited = kanji_notes.CharacterNote(
        **{**{f: getattr(curated, f) for f in curated.__slots__}, "meanings": ("mine",)}
    )
    kanji_notes.save_notes(config.kanji_notes_file, {"理": edited})

    plan = prepare_character_notes(config, ("理",), deck_path=deck, refresh_readings=True)

    note = plan.notes[0]
    assert note.meanings == ("mine",), "a hand-edited meaning survives a refresh"
    assert note.reading_example == curated.reading_example, "the prompt is fixed"
    assert note.reading_evidence is not None
    assert note.reading_evidence.readings.sha256 == "c" * 64, "the evidence moved"


def test_the_reading_prompt_is_the_first_example_the_source_printed(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)

    plan = prepare_character_notes(config, ("理",), deck_name="Kanji")

    prompt = plan.notes[0].reading_example
    assert prompt is not None and prompt.written == "料理"


# --- directions --------------------------------------------------------------


def test_directions_are_stored_in_card_order_not_argument_order(
    tmp_path: Path, sources: Sources
) -> None:
    """A template's ordinal is its identity in a collection with review
    history, so the order a caller happened to type is not the order stored."""
    config = project(tmp_path)

    plan = prepare_character_notes(
        config,
        ("理",),
        deck_name="Kanji",
        directions=("production", "recognition", "reading"),
        production_cues={"理": "the 理 of 料理"},
    )

    assert plan.directions == ("recognition", "reading", "production")
    assert plan.card_count == 3


def test_production_without_a_cue_refuses_rather_than_writing_one(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)

    with pytest.raises(CharacterNotesError, match="production"):
        prepare_character_notes(
            config, ("理",), deck_name="Kanji", directions=("production",)
        )

    assert not config.kanji_notes_file.exists()


def test_a_deck_that_builds_other_cards_refuses_rather_than_changing_its_set(
    tmp_path: Path, sources: Sources
) -> None:
    """A deck's enabled card set is fixed once it has review history."""
    config = project(tmp_path)
    deck = seed(config, ("理",))

    with pytest.raises(CharacterNotesError, match="fixed once it"):
        prepare_character_notes(
            config, ("料",), deck_path=deck, directions=("recognition", "reading")
        )


def test_a_word_deck_is_not_a_destination_for_character_notes(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    path = word_deck(config)

    with pytest.raises(CharacterNotesError, match="not word cards"):
        prepare_character_notes(config, ("理",), deck_path=path)

    assert not config.kanji_notes_file.exists()


@pytest.mark.parametrize(
    "declared",
    [
        "kanji:理",
        7,
        True,
        {"kanji:理": "logic"},
    ],
    ids=["string", "integer", "boolean", "mapping"],
)
def test_an_include_ids_that_is_not_a_list_refuses_and_names_the_deck(
    tmp_path: Path, sources: Sources, declared: object
) -> None:
    """A deck names its characters as a list, and this is where that is proved.

    Anything else reaches an append that a scalar and a mapping cannot answer;
    a batch that dies there says nothing about which file is wrong.
    """
    config = project(tmp_path)
    deck_path = seed(config, ("理",))
    document = yaml.safe_load(deck_path.read_text(encoding="utf-8"))
    document["deck"]["include_ids"] = declared
    deck_path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    before = deck_path.read_bytes()

    with pytest.raises(CharacterNotesError, match="list") as refusal:
        prepare_character_notes(config, ("料",), deck_path=deck_path)

    assert deck_path.name in str(refusal.value)
    assert deck_path.read_bytes() == before


def test_an_added_id_is_appended_without_losing_the_owners_comments(
    tmp_path: Path, sources: Sources
) -> None:
    """A deck file is curated, so the append round-trips rather than re-dumps."""
    config = project(tmp_path)
    deck_path = seed(config, ("理",))
    deck_path.write_text(
        deck_path.read_text(encoding="utf-8").replace(
            "include_ids:", "# the characters this lesson teaches\n  include_ids:"
        ),
        encoding="utf-8",
    )

    plan = prepare_character_notes(config, ("料",), deck_path=deck_path)
    apply(config, plan)

    text = deck_path.read_text(encoding="utf-8")
    assert "# the characters this lesson teaches" in text
    assert plan.deck_record_ids == ("kanji:理", "kanji:料")
    assert yaml.safe_load(text)["deck"]["include_ids"] == ["kanji:理", "kanji:料"]


# --- counts: what was selected, and what the deck will hold ------------------


def test_the_plan_counts_the_selection_and_the_whole_resulting_deck(
    tmp_path: Path, sources: Sources
) -> None:
    """Two different numbers, and confusing them ships the wrong package. The
    batch adds one character; the deck it lands in will build three."""
    config = project(tmp_path)
    deck = seed(config, ("理", "料"))

    plan = prepare_character_notes(config, ("鳥",), deck_path=deck)

    assert plan.note_count == 1 and plan.card_count == 1
    assert plan.deck_note_count == 3
    assert plan.deck_card_count == 3
    assert plan.deck_record_ids == tuple(
        character_record_id(character) for character in ("理", "料", "鳥")
    )

    result = apply(config, plan)

    assert (result.note_count, result.card_count) == (1, 1)
    assert (result.deck_note_count, result.deck_card_count) == (3, 3)


def test_the_whole_deck_count_multiplies_by_every_enabled_direction(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)

    plan = prepare_character_notes(
        config,
        ("理", "料"),
        deck_name="Kanji",
        directions=("recognition", "reading"),
    )

    assert plan.deck_note_count == 2
    assert plan.deck_card_count == 4


def test_a_deck_naming_a_character_with_no_note_refuses_before_it_is_counted(
    tmp_path: Path, sources: Sources
) -> None:
    """The counts have to describe a package that will actually build. A deck
    declaring an id the store has no note for builds nothing at all."""
    config = project(tmp_path)
    deck_path = seed(config, ("理",))
    document = yaml.safe_load(deck_path.read_text(encoding="utf-8"))
    document["deck"]["include_ids"].append("kanji:雨")
    deck_path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(CharacterNotesError, match="雨"):
        prepare_character_notes(config, ("料",), deck_path=deck_path)


# --- the wire ----------------------------------------------------------------


def test_a_plan_survives_json_and_comes_back_identical(
    tmp_path: Path, sources: Sources
) -> None:
    """A finish receipt resumes from this in another process, so the snapshot
    has to be complete on its own — no store, no provider, no second plan."""
    config = project(tmp_path)
    plan = prepare_character_notes(
        config,
        TARGETS,
        deck_name="Genki II Kanji",
        directions=("recognition", "reading"),
    )

    wire = json.loads(json.dumps(plan.to_dict(), ensure_ascii=False))
    restored = CharacterNotesPlan.from_dict(wire)

    assert restored == plan
    assert wire["deck_note_count"] == 5
    assert wire["deck_card_count"] == 10


def test_a_plan_edited_on_the_wire_is_refused_by_its_own_fingerprint(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    plan = prepare_character_notes(config, ("理",), deck_name="Kanji")

    wire = plan.to_dict()
    wire["deck_name"] = "Something Else"

    with pytest.raises(CharacterNotesError, match="character-plan-stale"):
        CharacterNotesPlan.from_dict(wire)


def test_a_restored_plan_applies_exactly_what_it_was_prepared_with(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    plan = prepare_character_notes(config, TARGETS, deck_name="Genki II Kanji")
    sources.looked_up.clear()
    sources.fetched.clear()

    restored = CharacterNotesPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    result = execute_character_notes(
        config, restored, expected_fingerprint=restored.fingerprint
    )

    assert sources.looked_up == [] and sources.fetched == [], "execution asks nobody"
    assert result.note_count == 5
    assert sorted(stored_notes(config)) == sorted(TARGETS)


# --- refusing before writing -------------------------------------------------


def test_a_file_edited_after_preparation_stops_the_batch_before_any_write(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    plan = prepare_character_notes(config, TARGETS, deck_name="Genki II Kanji")
    kanji_notes.save_notes(config.kanji_notes_file, {})

    with pytest.raises(CharacterNotesError, match="character-batch-stale"):
        apply(config, plan)

    assert not plan.deck_path.exists(), "nothing further was written"
    assert not config.kanji_file.exists()


def test_a_fingerprint_that_does_not_match_the_plan_refuses(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    plan = prepare_character_notes(config, ("理",), deck_name="Kanji")

    with pytest.raises(CharacterNotesError, match="character-plan-stale"):
        execute_character_notes(config, plan, expected_fingerprint="0" * 64)

    assert not config.kanji_notes_file.exists()


def test_a_plan_from_another_repository_refuses(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    other = project(tmp_path / "elsewhere")
    plan = prepare_character_notes(config, ("理",), deck_name="Kanji")

    with pytest.raises(CharacterNotesError, match="different repository"):
        apply(other, plan)

    assert not other.kanji_notes_file.exists()
    assert not config.kanji_notes_file.exists()


def test_a_batch_writing_a_deck_outside_the_deck_directory_refuses(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    plan = prepare_character_notes(config, ("理",), deck_name="Kanji")
    moved = replace(plan, deck_path=config.root / "kanji.yaml")

    with pytest.raises(CharacterNotesError, match="outside"):
        execute_character_notes(config, moved, expected_fingerprint=moved.fingerprint)


# --- resuming ----------------------------------------------------------------


def test_an_interrupted_apply_finishes_from_where_it_stopped(
    tmp_path: Path, sources: Sources
) -> None:
    """Each file is either what the plan saw or what the plan proposes. The
    second means "already applied", and a resume writes only the rest."""
    config = project(tmp_path)
    plan = prepare_character_notes(config, TARGETS, deck_name="Genki II Kanji")
    landed = next(
        item for item in plan.changed_files if item.label == "character notes"
    )
    assert landed.after_text is not None
    landed.path.parent.mkdir(parents=True, exist_ok=True)
    landed.path.write_text(landed.after_text, encoding="utf-8")

    result = apply(config, plan)

    assert "character notes" not in result.changed, "it was already there"
    assert "character deck" in result.changed
    assert plan.deck_path.exists()


def test_applying_the_same_batch_twice_changes_nothing_the_second_time(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    plan = prepare_character_notes(config, TARGETS, deck_name="Genki II Kanji")
    first = apply(config, plan)

    second = apply(config, plan)

    assert first.changed and second.changed == ()
    assert first.record_ids == second.record_ids
    assert (second.deck_note_count, second.deck_card_count) == (5, 5)


# --- what a caller may ask for -----------------------------------------------


def test_naming_neither_a_deck_nor_a_deck_name_refuses(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)

    with pytest.raises(CharacterNotesError, match="either an existing"):
        prepare_character_notes(config, ("理",))


def test_a_word_is_not_a_character_target(tmp_path: Path, sources: Sources) -> None:
    config = project(tmp_path)

    with pytest.raises(CharacterNotesError, match="one character"):
        prepare_character_notes(config, ("料理",), deck_name="Kanji")

    assert sources.looked_up == [], "refused before anything was asked for"


def test_one_character_named_twice_is_refused(tmp_path: Path, sources: Sources) -> None:
    config = project(tmp_path)

    with pytest.raises(CharacterNotesError, match="more than once"):
        prepare_character_notes(config, ("理", "理"), deck_name="Kanji")


def test_a_cue_for_a_character_not_in_the_batch_is_refused(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)

    with pytest.raises(CharacterNotesError, match="not "):
        prepare_character_notes(
            config, ("理",), deck_name="Kanji", production_cues={"料": "a cue"}
        )


def test_a_failed_lookup_writes_nothing_and_says_which_character(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sources: Sources
) -> None:
    config = project(tmp_path)
    monkeypatch.setattr(kanji, "fetch_kanji", Sources(refuse=frozenset({"理"})).fetch_kanji)

    with pytest.raises(CharacterNotesError, match="理"):
        prepare_character_notes(config, ("理",), deck_name="Kanji")

    assert not config.kanji_file.exists()


def test_creating_a_deck_where_one_already_exists_refuses(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    seed(config, ("理",))

    with pytest.raises(CharacterNotesError, match="already uses the stable file stem"):
        prepare_character_notes(config, ("料",), deck_name="Kanji")


def test_the_facts_file_is_not_rewritten_when_nothing_was_fetched(
    tmp_path: Path, sources: Sources
) -> None:
    """A committed store nobody added to must not churn: a no-op preparation
    that rewrote it would show a diff for a run that learned nothing."""
    config = project(tmp_path)
    deck = seed(config, ("理",))
    before = config.jpdb_readings_file.read_bytes()

    plan = prepare_character_notes(config, ("理",), deck_path=deck)
    apply(config, plan)

    assert config.jpdb_readings_file.read_bytes() == before
    assert [item.label for item in plan.changed_files] == []


def test_the_saved_facts_are_exactly_what_the_provider_answered(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)

    seed(config, ("理",))

    saved = jpdb_kanji.load_readings(config.jpdb_readings_file)["理"]
    assert saved == readings(
        "理", examples=(example("料理", "りょうり"), example("理由", "りゆう"))
    )


# --- the applied state, checked before anything is built ---------------------
#
# Between planning a package and writing one, the only acceptable state is the
# one the batch applied. Every test here also asserts that the check itself
# left every bound file byte for byte as it found it.


def test_a_prepared_batch_that_was_never_applied_is_refused(
    tmp_path: Path, sources: Sources
) -> None:
    """The apply accepts the before state so an interrupted batch can finish.
    A build cannot: these notes do not exist yet, and no package may claim
    them."""
    config = project(tmp_path)
    plan = prepare_character_notes(config, TARGETS, deck_name="Genki II Kanji")
    before = bound_bytes(plan)

    with pytest.raises(CharacterNotesError, match="character-not-applied"):
        verify_character_notes_applied(config, plan)

    assert bound_bytes(plan) == before, "the check wrote nothing"
    assert not config.kanji_notes_file.exists()


def test_an_exactly_applied_batch_passes_and_touches_nothing(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    plan = prepare_character_notes(config, TARGETS, deck_name="Genki II Kanji")
    apply(config, plan)
    sources.looked_up.clear()
    sources.fetched.clear()
    before = bound_bytes(plan)

    assert verify_character_notes_applied(config, plan) is None

    assert sources.looked_up == [] and sources.fetched == [], "it asks nobody"
    assert bound_bytes(plan) == before, "and it writes nothing"


def test_the_plan_a_receipt_restores_is_the_plan_the_verifier_accepts(
    tmp_path: Path, sources: Sources
) -> None:
    """A resume holds a plan rebuilt from its wire form, never the object that
    was prepared, so that is the value this check has to accept."""
    config = project(tmp_path)
    plan = prepare_character_notes(config, TARGETS, deck_name="Genki II Kanji")
    apply(config, plan)
    restored = CharacterNotesPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    before = bound_bytes(plan)

    assert verify_character_notes_applied(config, restored) is None

    assert bound_bytes(plan) == before


def test_an_interrupted_apply_is_refused_rather_than_resumed(
    tmp_path: Path, sources: Sources
) -> None:
    """``execute_character_notes`` would finish this batch. The verifier will
    not: a package built from a half-applied batch ships whatever landed
    first."""
    config = project(tmp_path)
    plan = prepare_character_notes(config, TARGETS, deck_name="Genki II Kanji")
    landed = next(
        item for item in plan.changed_files if item.label == "character notes"
    )
    assert landed.after_text is not None
    landed.path.parent.mkdir(parents=True, exist_ok=True)
    landed.path.write_text(landed.after_text, encoding="utf-8")
    before = bound_bytes(plan)

    with pytest.raises(CharacterNotesError, match="character-not-applied"):
        verify_character_notes_applied(config, plan)

    assert bound_bytes(plan) == before
    assert not plan.deck_path.exists()


def test_notes_edited_after_the_apply_are_refused(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    plan = prepare_character_notes(config, TARGETS, deck_name="Genki II Kanji")
    apply(config, plan)
    stored = stored_notes(config)
    stored["理"] = replace(stored["理"], meanings=("nobody confirmed this",))
    kanji_notes.save_notes(config.kanji_notes_file, stored)
    before = bound_bytes(plan)

    with pytest.raises(
        CharacterNotesError, match=r"character-batch-stale.*kanji_notes\.json"
    ):
        verify_character_notes_applied(config, plan)

    assert bound_bytes(plan) == before


def test_a_deck_deleted_after_the_apply_is_refused(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    deck = seed(config, ("理",))
    plan = prepare_character_notes(config, ("料",), deck_path=deck)
    apply(config, plan)
    deck.unlink()
    before = bound_bytes(plan)

    with pytest.raises(CharacterNotesError, match=r"character-batch-stale.*\.yaml"):
        verify_character_notes_applied(config, plan)

    assert bound_bytes(plan) == before


def test_evidence_edited_after_the_apply_is_refused(
    tmp_path: Path, sources: Sources
) -> None:
    """The provider facts are bound too. The notes were written from them, and
    a file rewritten afterwards is not the evidence anyone confirmed."""
    config = project(tmp_path)
    plan = prepare_character_notes(config, ("理",), deck_name="Kanji")
    apply(config, plan)
    facts = config.jpdb_readings_file
    facts.write_text(facts.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    before = bound_bytes(plan)

    with pytest.raises(
        CharacterNotesError, match=r"character-batch-stale.*jpdb_readings\.json"
    ):
        verify_character_notes_applied(config, plan)

    assert bound_bytes(plan) == before


def test_a_file_this_batch_only_bound_is_checked_too(
    tmp_path: Path, sources: Sources
) -> None:
    """A rerun proposes no change to anything and still binds every file it
    read. The unchanged ones are checked at exactly the same strength: an edit
    to one of them is an edit to what a package would be built from."""
    config = project(tmp_path)
    deck = seed(config, ("理",))
    plan = prepare_character_notes(config, ("理",), deck_path=deck)
    assert plan.changed_files == (), "this batch proposes no change at all"
    reference = config.kanji_file
    reference.write_text(
        reference.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    before = bound_bytes(plan)

    with pytest.raises(
        CharacterNotesError, match=r"character-batch-stale.*kanji\.json"
    ):
        verify_character_notes_applied(config, plan)

    assert bound_bytes(plan) == before


def test_a_plan_from_another_repository_is_refused_by_the_verifier(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    other = project(tmp_path / "elsewhere")
    plan = prepare_character_notes(config, ("理",), deck_name="Kanji")
    apply(config, plan)
    before = bound_bytes(plan)

    with pytest.raises(CharacterNotesError, match="different repository"):
        verify_character_notes_applied(other, plan)

    assert bound_bytes(plan) == before
    assert not other.kanji_notes_file.exists()


def test_a_verified_plan_whose_deck_left_the_deck_directory_is_refused(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    plan = prepare_character_notes(config, ("理",), deck_name="Kanji")
    apply(config, plan)
    moved = replace(plan, deck_path=config.root / "kanji.yaml")
    before = bound_bytes(plan)

    with pytest.raises(CharacterNotesError, match="outside"):
        verify_character_notes_applied(config, moved)

    assert bound_bytes(plan) == before


def test_a_plan_altered_after_it_was_prepared_is_refused_by_the_verifier(
    tmp_path: Path, sources: Sources
) -> None:
    """Its fingerprint is what makes a saved plan the confirmed one. Restoring
    this from a receipt would refuse; so does handing the restored object on."""
    config = project(tmp_path)
    plan = prepare_character_notes(config, ("理",), deck_name="Kanji")
    apply(config, plan)
    altered = replace(plan, deck_name="Something Else")
    before = bound_bytes(plan)

    with pytest.raises(CharacterNotesError, match="character-plan-stale"):
        verify_character_notes_applied(config, altered)

    assert bound_bytes(plan) == before
