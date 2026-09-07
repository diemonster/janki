"""Plan-bound Assistant preparation of dedicated character notes.

These exercise Janki's own surface over the character-note application
service: what the owner is shown before confirming, which structurally
impossible requests refuse, and that confirmation applies the exact plan
preparation already produced rather than looking anything up again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import ai_schema, jpdb_kanji, kanji_notes
from japanese_anki.application import assistant_kanji_notes, character_notes
from japanese_anki.config import ProjectConfig


def _project(tmp_path: Path) -> ProjectConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'deck_dir = "data/decks"\n'
        'dist_dir = "dist"\n'
        'kanji_notes_file = "data/kanji_notes.json"\n',
        encoding="utf-8",
    )
    return ProjectConfig.load(tmp_path)


def _example(written: str, pronounced: str, gloss: str) -> jpdb_kanji.BoundExample:
    return jpdb_kanji.BoundExample(
        written=written,
        pronounced=pronounced,
        gloss=gloss,
        furigana=f"{written}[{pronounced}]",
        source_url=f"https://jpdb.io/vocabulary/1/{written}/{pronounced}",
    )


def _usage(
    character: str,
    label: str,
    percent_text: str | None,
    percent: int | None,
    *,
    examples: tuple[jpdb_kanji.BoundExample, ...] = (),
) -> jpdb_kanji.ReadingUsage:
    return jpdb_kanji.ReadingUsage(
        label=label,
        href=f"/kanji-reading/{character}/{label}",
        percent_text=percent_text,
        percent=percent,
        percent_less_than=None if percent is None else False,
        examples=examples,
    )


def _note(
    character: str,
    meanings: tuple[str, ...],
    *,
    stroke_count: int = 0,
    readings: tuple[jpdb_kanji.ReadingUsage, ...] = (),
    inventory: tuple[str, ...] = (),
    production_cue: str = "",
) -> kanji_notes.CharacterNote:
    """One prepared note, of the class the note module actually writes.

    A fake service plan keeps this surface isolated from the application
    service; a fake *note* would only pin a wire shape this surface invented,
    which is what these tests are here to stop.
    """

    evidence = (
        None
        if not readings
        else kanji_notes.evidence_from_readings(
            jpdb_kanji.CharacterReadings(
                character=character,
                source_url=jpdb_kanji.kanji_page_url(character),
                fetched_at_utc="2026-09-07T00:00:00Z",
                sha256="0" * 64,
                groups=(
                    jpdb_kanji.ReadingGroup(
                        source_class=jpdb_kanji.COMMON_CELL_CLASS,
                        readings=readings,
                    ),
                ),
            )
        )
    )
    return kanji_notes.CharacterNote(
        character=character,
        id=f"kanji:{character}",
        meanings=meanings,
        stroke_count=stroke_count,
        kanjidic_readings=tuple(
            kanji_notes.KanjidicReading(kind="on", reading=reading)
            for reading in inventory
        ),
        reading_evidence=evidence,
        reading_example=kanji_notes.first_bound_example(evidence),
        production_cue=production_cue,
    )


@dataclass(frozen=True)
class _Plan:
    """A stand-in for the pinned CharacterNotesPlan contract.

    ``deck_record_ids`` is what the deck holds once this batch lands, which is
    only the same as the selection when the destination was new. Keeping the
    two apart here is what lets a preview say "one note, six in the deck".
    """

    deck_path: Path
    output_path: Path
    deck_name: str
    characters: tuple[str, ...]
    directions: tuple[str, ...]
    notes: tuple[kanji_notes.CharacterNote, ...]
    deck_record_ids: tuple[str, ...]
    fingerprint: str = "a" * 64

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(note.id for note in self.notes)

    @property
    def note_count(self) -> int:
        return len(self.notes)

    @property
    def card_count(self) -> int:
        return len(self.notes) * len(self.directions)

    @property
    def deck_note_count(self) -> int:
        return len(self.deck_record_ids)

    @property
    def deck_card_count(self) -> int:
        return len(self.deck_record_ids) * len(self.directions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deck_path": self.deck_path.as_posix(),
            "output_path": self.output_path.as_posix(),
            "deck_name": self.deck_name,
            "characters": list(self.characters),
            "directions": list(self.directions),
            "deck_record_ids": list(self.deck_record_ids),
            "fingerprint": self.fingerprint,
            "notes": [note.to_dict() for note in self.notes],
        }


_FIVE = ("物", "特", "鳥", "料", "理")

_NOTES = (
    _note(
        "物",
        ("thing", "object"),
        stroke_count=8,
        readings=(
            _usage(
                "物",
                "ぶつ",
                "(56%)",
                56,
                examples=(_example("動物", "どうぶつ", "animal"),),
            ),
        ),
        inventory=("ブツ", "もの"),
    ),
    _note(
        "特",
        ("special",),
        stroke_count=10,
        readings=(_usage("特", "とく", "(99%)", 99),),
    ),
    _note("鳥", ("bird",)),
    _note("料", ("fee", "materials")),
    _note("理", ("logic", "reason")),
)


def _plan_for(
    config: ProjectConfig,
    *,
    characters: tuple[str, ...] = _FIVE,
    directions: tuple[str, ...] = ("recognition",),
    notes: tuple[kanji_notes.CharacterNote, ...] = _NOTES,
    deck_record_ids: tuple[str, ...] | None = None,
) -> _Plan:
    return _Plan(
        deck_path=config.deck_dir / "genki-ii-kanji.yaml",
        output_path=config.dist_dir / "genki-ii-kanji.apkg",
        deck_name="Genki II Kanji",
        characters=characters,
        directions=directions,
        notes=notes,
        deck_record_ids=(
            tuple(note.id for note in notes)
            if deck_record_ids is None
            else deck_record_ids
        ),
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    plan: _Plan,
    *,
    calls: list[dict[str, Any]] | None = None,
) -> None:
    def prepare(config: ProjectConfig, characters: Any, **options: Any) -> _Plan:
        if calls is not None:
            calls.append({"characters": tuple(characters), **options})
        return plan

    monkeypatch.setattr(character_notes, "prepare_character_notes", prepare)


def _request(**overrides: Any) -> assistant_kanji_notes.AssistantKanjiNotesRequest:
    values: dict[str, Any] = {
        "characters": _FIVE,
        "deck_path": None,
        "deck_name": "Genki II Kanji",
        "directions": ("recognition",),
        "refresh_readings": False,
        "production_cues": (),
        "instruction": "Add character notes for these five kanji.",
    }
    values.update(overrides)
    return assistant_kanji_notes.AssistantKanjiNotesRequest(**values)


def test_the_closed_schema_carries_character_targets_and_no_free_text() -> None:
    """The model may name characters and choices; it may not name a path."""

    import pydantic

    adapter = pydantic.TypeAdapter(ai_schema.assistant_agent_schema())

    parsed = adapter.validate_python(
        {
            "answer": "I can prepare those five character notes.",
            "action_intents": [
                {
                    "kind": "add_kanji_notes",
                    "resource_ids": [],
                    "record_ids": [],
                    "instruction": "Add character notes for these five kanji.",
                    "options": {
                        "study_type": "kanji",
                        "kanji_characters": list(_FIVE),
                        "deck_name": "Genki II Kanji",
                        "card_directions": ["recognition"],
                    },
                }
            ],
        }
    )

    intent = parsed.action_intents[0]
    assert intent.kind == "add_kanji_notes"
    assert intent.options.kanji_characters == list(_FIVE)
    assert intent.options.refresh_readings is None
    assert intent.options.production_cues == []
    with pytest.raises(pydantic.ValidationError):
        adapter.validate_python(
            {
                "answer": "I can prepare that.",
                "action_intents": [
                    {
                        "kind": "add_kanji_notes",
                        "resource_ids": [],
                        "record_ids": [],
                        "instruction": "Add these.",
                        "options": {"study_type": "grammar"},
                    }
                ],
            }
        )


def test_preparation_shows_the_actual_note_sides_and_exact_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Five explicit targets, recognition only: five notes and five cards."""

    config = _project(tmp_path)
    calls: list[dict[str, Any]] = []
    _install(monkeypatch, _plan_for(config), calls=calls)

    plan = assistant_kanji_notes.plan_kanji_notes(config, _request())

    assert calls == [
        {
            "characters": _FIVE,
            "deck_path": None,
            "deck_name": "Genki II Kanji",
            "directions": ("recognition",),
            "refresh_readings": False,
            "production_cues": None,
        }
    ]
    target = plan.projection["target"]
    assert target["note_count"] == 5
    assert target["card_count"] == 5
    assert target["deck_state"] == "new"
    assert target["characters"] == list(_FIVE)
    assert [note["cards"][0]["front"] for note in target["notes"]] == list(_FIVE)
    assert target["notes"][0]["cards"][0]["back"] == [
        "Meanings: thing, object",
        "Strokes: 8",
        "jpdb reported usage · ぶつ (56%)",
        "動物 どうぶつ — animal",
        "Additional readings: ブツ, もの",
    ]
    assert plan.projection["billing_class"] == "local"
    assert plan.projection["provider"] is None
    assert plan.projection["inputs"]["queried_characters"] == list(_FIVE)
    assert (
        plan.fingerprint
        == hashlib.sha256(plan.projection_wire.encode("utf-8")).hexdigest()
    )
    assert json.loads(plan.plan_wire)["fingerprint"] == "a" * 64


def test_a_reported_percentage_shows_even_without_a_bound_example(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reading with no example yet still has reported usage worth showing."""

    config = _project(tmp_path)
    _install(monkeypatch, _plan_for(config))

    plan = assistant_kanji_notes.plan_kanji_notes(config, _request())

    assert plan.projection["target"]["notes"][1]["cards"][0]["back"] == [
        "Meanings: special",
        "Strokes: 10",
        "jpdb reported usage · とく (99%)",
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"directions": ("listening",)}, "must be unique values chosen from"),
        ({"directions": ()}, "must be unique values chosen from"),
        ({"characters": ("料理",)}, "single-character targets"),
        ({"characters": ("理", "理")}, "exactly once"),
        (
            {"deck_path": Path("data/decks/kanji.yaml")},
            "not both and not neither",
        ),
        ({"deck_name": None}, "not both and not neither"),
    ],
    ids=(
        "unknown-direction",
        "no-direction",
        "multi-character-target",
        "repeated-target",
        "two-destinations",
        "no-destination",
    ),
)
def test_structurally_unsupported_requests_refuse_before_any_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    config = _project(tmp_path)
    monkeypatch.setattr(
        character_notes,
        "prepare_character_notes",
        lambda *_args, **_kwargs: pytest.fail(
            "an unsupported request must refuse before preparation"
        ),
    )

    with pytest.raises(assistant_kanji_notes.AssistantKanjiNotesError, match=message):
        assistant_kanji_notes.plan_kanji_notes(config, _request(**overrides))


def test_confirmation_applies_the_saved_plan_without_preparing_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-preparing would repeat the bounded lookup the owner already saw."""

    config = _project(tmp_path)
    service = _plan_for(config)
    calls: list[dict[str, Any]] = []
    _install(monkeypatch, service, calls=calls)
    plan = assistant_kanji_notes.plan_kanji_notes(config, _request())
    applied: list[tuple[Any, str]] = []

    def execute(
        _config: ProjectConfig,
        expected: Any,
        *,
        expected_fingerprint: str,
    ) -> Any:
        applied.append((expected, expected_fingerprint))
        return object()

    monkeypatch.setattr(character_notes, "execute_character_notes", execute)
    monkeypatch.setattr(
        character_notes,
        "prepare_character_notes",
        lambda *_args, **_kwargs: pytest.fail("confirmation must not re-prepare"),
    )

    execution = assistant_kanji_notes.execute_kanji_notes(config, plan)

    assert len(calls) == 1
    assert applied == [(service, "a" * 64)]
    assert execution.plan is plan


def test_a_plan_from_another_repository_refuses_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    _install(monkeypatch, _plan_for(config))
    plan = assistant_kanji_notes.plan_kanji_notes(config, _request())
    monkeypatch.setattr(
        character_notes,
        "execute_character_notes",
        lambda *_args, **_kwargs: pytest.fail("a foreign plan must not be applied"),
    )
    elsewhere = _project(tmp_path / "elsewhere")

    with pytest.raises(
        assistant_kanji_notes.AssistantKanjiNotesError,
        match="belongs to another repository",
    ):
        assistant_kanji_notes.execute_kanji_notes(elsewhere, plan)


def test_a_note_without_source_backed_meanings_refuses_rather_than_showing_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    _install(
        monkeypatch,
        _plan_for(
            config,
            characters=("理",),
            notes=(_note("理", ()),),
        ),
    )

    with pytest.raises(
        assistant_kanji_notes.AssistantKanjiNotesError,
        match="no exact source-backed character and meanings",
    ):
        assistant_kanji_notes.plan_kanji_notes(
            config,
            _request(characters=("理",)),
        )


def test_the_projection_names_every_file_the_confirmed_batch_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    _install(monkeypatch, _plan_for(config))

    plan = assistant_kanji_notes.plan_kanji_notes(config, _request())

    assert plan.projection["writes"] == {
        "character_notes": "data/kanji_notes.json",
        "deck_definition": "data/decks/genki-ii-kanji.yaml",
        "package": "dist/genki-ii-kanji.apkg",
    }


def test_an_existing_destination_writes_no_deck_definition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    _install(monkeypatch, _plan_for(config))

    plan = assistant_kanji_notes.plan_kanji_notes(
        config,
        _request(
            deck_name=None,
            deck_path=config.deck_dir / "genki-ii-kanji.yaml",
        ),
    )

    assert "deck_definition" not in plan.projection["writes"]
    assert plan.projection["target"]["deck_state"] == "existing"


_RI_EXAMPLE = jpdb_kanji.BoundExample(
    written="料理",
    pronounced="りょうり",
    gloss="cooking",
    furigana="料理[りょうり]",
    source_url="https://jpdb.io/vocabulary/1552120/料理/りょうり",
)

_RI_NOTE = kanji_notes.CharacterNote(
    character="理",
    id="kanji:理",
    meanings=("logic", "reason"),
    stroke_count=11,
    strokes=tuple(f"M{n}0,10L{n}0,70" for n in range(11)),
    # The on/kun inventory is deliberately spelled unlike anything jpdb
    # supplied, so a line that survives can only have come from KANJIDIC.
    kanjidic_readings=(
        kanji_notes.KanjidicReading(kind="on", reading="リ"),
        kanji_notes.KanjidicReading(kind="kun", reading="おさ.める"),
    ),
    reading_evidence=kanji_notes.ReadingEvidence(
        source=jpdb_kanji.SOURCE,
        metric=jpdb_kanji.METRIC,
        readings=jpdb_kanji.CharacterReadings(
            character="理",
            source_url="https://jpdb.io/kanji/理",
            fetched_at_utc="2026-09-07T00:00:00Z",
            sha256="0" * 64,
            groups=(
                jpdb_kanji.ReadingGroup(
                    source_class=jpdb_kanji.COMMON_CELL_CLASS,
                    readings=(
                        jpdb_kanji.ReadingUsage(
                            label="り",
                            href="/kanji-reading/理/り",
                            percent_text="(84%)",
                            percent=84,
                            percent_less_than=False,
                            examples=(_RI_EXAMPLE,),
                            detail_source_url="https://jpdb.io/kanji-reading/理/り",
                            detail_fetched_at_utc="2026-09-07T00:00:00Z",
                            detail_sha256="1" * 64,
                        ),
                    ),
                ),
                jpdb_kanji.ReadingGroup(
                    source_class="kanji-reading-list",
                    readings=(
                        jpdb_kanji.ReadingUsage(
                            label="ことわり",
                            href="/kanji-reading/理/ことわり",
                            percent_text=None,
                            percent=None,
                            percent_less_than=None,
                        ),
                    ),
                ),
            ),
        ),
    ),
    reading_example=_RI_EXAMPLE,
    created_at="2026-09-07T00:00:00Z",
    sources=("KANJIDIC", "KanjiVG", "jpdb"),
)


def test_preview_uses_real_character_note_contract() -> None:
    """The projection must read the note shape the note module actually writes.

    Everything above stands in a hand-written note whose ``to_dict`` this
    surface defined for itself. A real :class:`CharacterNote` files its
    identity under ``id`` and carries no ``character`` key at all, states its
    evidence as one object whose figures live at
    ``reading_evidence.readings.groups[].readings[]``, and binds its examples
    inside those readings rather than in any ``bound_examples`` list. A
    confirmation card built by guessing at those names shows the owner an
    empty or wrong side while the source-backed facts sit in the note.
    """

    projection = assistant_kanji_notes._note_projection(_RI_NOTE, ("recognition",))

    lines = projection["cards"][0]["back"]
    assert projection["character"] == "理"
    assert projection["record_id"] == "kanji:理"
    assert projection["cards"][0]["front"] == "理"
    assert projection["note"]["id"] == "kanji:理"
    assert any("logic" in line and "reason" in line for line in lines)
    assert any("11" in line for line in lines)
    # jpdb's own figure, beside the reading it was printed beside.
    assert any("り" in line and "84%" in line for line in lines)
    # A supplied reading jpdb quantified nothing for is still reported usage.
    assert any("ことわり" in line for line in lines)
    assert any(
        "料理" in line and "りょうり" in line and "cooking" in line for line in lines
    )
    # The KANJIDIC inventory, which no jpdb evidence line could supply.
    assert any("おさ.める" in line and "リ" in line for line in lines)


def test_a_curated_cue_already_on_the_note_satisfies_a_production_direction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cue the owner wrote earlier is still the owner's cue.

    Requiring a fresh cue for every target duplicates the core's own
    ``supported_directions`` gate and makes an already-curated note
    unusable, which is the opposite of what curation is for.
    """

    config = _project(tmp_path)
    curated = _note("理", ("logic",), production_cue="the one in 料理")
    _install(
        monkeypatch,
        _plan_for(
            config,
            characters=("理",),
            directions=("production",),
            notes=(curated,),
        ),
    )

    plan = assistant_kanji_notes.plan_kanji_notes(
        config,
        _request(characters=("理",), directions=("production",)),
    )

    assert plan.projection["target"]["directions"] == ["production"]


def test_each_selected_direction_shows_its_own_exact_card_sides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reading card does not ask the character, and neither does production.

    Showing the recognition front for every direction previews a card the
    owner is not making. Each front here is read off the note field its own
    template renders.
    """

    config = _project(tmp_path)
    note = _note(
        "物",
        ("thing", "object"),
        stroke_count=8,
        readings=(
            _usage(
                "物",
                "ぶつ",
                "(56%)",
                56,
                examples=(_example("動物", "どうぶつ", "animal"),),
            ),
        ),
        inventory=("ブツ", "もの"),
        production_cue="the counter-word one",
    )
    _install(
        monkeypatch,
        _plan_for(
            config,
            characters=("物",),
            directions=("recognition", "reading", "production"),
            notes=(note,),
        ),
    )

    plan = assistant_kanji_notes.plan_kanji_notes(
        config,
        _request(
            characters=("物",),
            directions=("recognition", "reading", "production"),
            production_cues=(("物", "the counter-word one"),),
        ),
    )

    cards = plan.projection["target"]["notes"][0]["cards"]
    assert [card["direction"] for card in cards] == [
        "recognition",
        "reading",
        "production",
    ]
    assert cards[0]["front"] == "物"
    assert cards[0]["back"] == [
        "Meanings: thing, object",
        "Strokes: 8",
        "jpdb reported usage · ぶつ (56%)",
        "動物 どうぶつ — animal",
        "Additional readings: ブツ, もの",
    ]
    assert cards[1]["front"] == "動物"
    assert cards[1]["back"] == [
        "動物[どうぶつ]",
        "animal",
        "物 · thing, object",
    ]
    assert cards[2]["front"] == "the counter-word one"
    assert cards[2]["back"] == ["物", "Meanings: thing, object"]


def test_a_reading_only_batch_previews_only_the_reading_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    note = _note(
        "物",
        ("thing",),
        readings=(
            _usage(
                "物",
                "ぶつ",
                None,
                None,
                examples=(_example("動物", "どうぶつ", "animal"),),
            ),
        ),
    )
    _install(
        monkeypatch,
        _plan_for(
            config,
            characters=("物",),
            directions=("reading",),
            notes=(note,),
        ),
    )

    plan = assistant_kanji_notes.plan_kanji_notes(
        config,
        _request(characters=("物",), directions=("reading",)),
    )

    cards = plan.projection["target"]["notes"][0]["cards"]
    assert len(cards) == 1
    assert cards[0]["direction"] == "reading"
    assert cards[0]["front"] == "動物"
    assert "物" not in cards[0]["front"] or cards[0]["front"] != "物"


def test_a_production_only_batch_previews_only_the_cue_front(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    note = _note("理", ("logic",), production_cue="the one in 料理")
    _install(
        monkeypatch,
        _plan_for(
            config,
            characters=("理",),
            directions=("production",),
            notes=(note,),
        ),
    )

    plan = assistant_kanji_notes.plan_kanji_notes(
        config,
        _request(characters=("理",), directions=("production",)),
    )

    cards = plan.projection["target"]["notes"][0]["cards"]
    assert len(cards) == 1
    assert cards[0]["direction"] == "production"
    assert cards[0]["front"] == "the one in 料理"


def test_the_projection_separates_selected_notes_from_the_whole_deck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding one to five is one note and six cards in the built package."""

    config = _project(tmp_path)
    note = _note("理", ("logic", "reason"))
    _install(
        monkeypatch,
        _plan_for(
            config,
            characters=("理",),
            notes=(note,),
            deck_record_ids=(
                "kanji:物",
                "kanji:特",
                "kanji:鳥",
                "kanji:料",
                "kanji:理",
                "kanji:説",
            ),
        ),
    )

    plan = assistant_kanji_notes.plan_kanji_notes(
        config,
        _request(characters=("理",)),
    )

    target = plan.projection["target"]
    assert target["note_count"] == 1
    assert target["card_count"] == 1
    assert target["deck_note_count"] == 6
    assert target["deck_card_count"] == 6
    assert target["deck_record_ids"] == [
        "kanji:物",
        "kanji:特",
        "kanji:鳥",
        "kanji:料",
        "kanji:理",
        "kanji:説",
    ]
