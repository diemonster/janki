"""The curated character store: identity, the wire, and the direction gate.

`tests/test_kanji.py` covers the *reference* cache one lookup fills. This
covers the store beside it — the notes a person curates, what survives a
round trip, and which card directions a note structurally supports. Nothing
here reads Japanese: every question is about presence and shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from japanese_anki import jpdb_kanji, kanji_notes
from japanese_anki.identifiers import IdentityError, character_record_id
from japanese_anki.jpdb_kanji import (
    BoundExample,
    CharacterReadings,
    ReadingGroup,
    ReadingUsage,
)
from japanese_anki.kanji_notes import (
    CharacterNote,
    KanjidicReading,
    KanjiNoteError,
)


def evidence(*, examples: tuple[BoundExample, ...] = ()) -> kanji_notes.ReadingEvidence:
    """One facts entry copied onto a note, exactly as preparation copies it."""
    return kanji_notes.evidence_from_readings(
        CharacterReadings(
            character="理",
            source_url="https://jpdb.io/kanji/%E7%90%86",
            fetched_at_utc="2026-09-07T00:00:00Z",
            sha256="a" * 64,
            groups=(
                ReadingGroup(
                    source_class="kanji-reading-list-common",
                    readings=(
                        ReadingUsage(
                            label="り",
                            href="https://jpdb.io/kanji/%E7%90%86%23り",
                            percent_text="(84%)",
                            percent=84,
                            percent_less_than=False,
                            examples=examples,
                        ),
                    ),
                ),
            ),
        )
    )


def bound(written: str = "料理", pronounced: str = "りょうり") -> BoundExample:
    return BoundExample(
        written=written,
        pronounced=pronounced,
        gloss="cooking",
        furigana="料理[りょうり]",
        source_url="https://jpdb.io/kanji/%E7%90%86",
    )


def note(character: str = "理", **overrides: object) -> CharacterNote:
    values: dict[str, object] = {
        "character": character,
        "id": character_record_id(character),
        "meanings": ("logic", "reason"),
        "stroke_count": 11,
        "strokes": ("M12,12L20,20",),
        "kanjidic_readings": (KanjidicReading(kind="on", reading="リ"),),
        "reading_evidence": evidence(examples=(bound(),)),
        "created_at": "2026-09-07T00:00:00Z",
        "sources": ("kanjiapi.dev", "KanjiVG"),
    }
    values.update(overrides)
    return CharacterNote(**values)  # type: ignore[arg-type]


def test_one_character_is_one_identity_and_a_word_is_refused() -> None:
    """N targets yield N notes, and a compound never becomes one of them."""
    assert character_record_id("理") == "kanji:理"

    with pytest.raises(IdentityError):
        character_record_id("料理")


def test_identity_folds_a_compatibility_spelling_of_the_same_character() -> None:
    """⼀ (U+2F00, the Kangxi radical) is a spelling of 一, not a second note."""
    assert character_record_id("⼀") == character_record_id("一")


def test_a_note_round_trips_through_its_wire_form() -> None:
    original = note(production_cue="reason, as in a principle", tags=("janki:kanji",))

    assert CharacterNote.from_dict("理", original.to_dict()) == original


def test_the_store_is_written_sorted_so_a_rerun_diffs_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "kanji_notes.json"

    kanji_notes.save_notes(path, {"理": note("理"), "物": note("物")})
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert list(payload["notes"]) == sorted(["理", "物"])
    assert path.read_text(encoding="utf-8").endswith("}\n")
    assert kanji_notes.load_notes(path) == {"物": note("物"), "理": note("理")}


def test_a_missing_store_reads_as_empty_rather_than_failing(tmp_path: Path) -> None:
    assert kanji_notes.load_notes(tmp_path / "absent.json") == {}


def test_an_entry_filed_under_the_wrong_character_is_refused(tmp_path: Path) -> None:
    """The key *is* the identity. A note whose id names another character would
    build a card under a GUID nothing else in the repository agrees with."""
    path = tmp_path / "kanji_notes.json"
    kanji_notes.save_notes(path, {"理": note("理")})
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["notes"]["理"]["id"] = "kanji:物"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(KanjiNoteError, match="kanji:理"):
        kanji_notes.load_notes(path)


def test_a_word_key_is_refused_by_the_store(tmp_path: Path) -> None:
    path = tmp_path / "kanji_notes.json"
    path.write_text(
        json.dumps({"schema_version": 1, "notes": {"料理": {"id": "kanji:料理"}}}),
        encoding="utf-8",
    )

    with pytest.raises(KanjiNoteError):
        kanji_notes.load_notes(path)


def test_two_keys_folding_to_one_identity_are_refused_rather_than_merged(
    tmp_path: Path,
) -> None:
    """塚 U+585A and 塚 U+FA10 are one identity, so they cannot be two entries.

    NFKC folds the compatibility ideograph onto the unified one, so both keys
    mint ``kanji:塚`` and each entry passes the key-is-the-identity check on its
    own. Loaded, they are two store entries for one Anki note: the exporter
    keys notes by id, so whichever came second silently replaces the first and
    a curated cue and fixed example vanish from the deck with nothing said.
    The store refuses instead, and names both keys — which print identically —
    by code point, so a person can tell which line of the file to delete.
    """
    unified, compatibility = chr(0x585A), chr(0xFA10)
    path = tmp_path / "kanji_notes.json"
    kanji_notes.save_notes(
        path,
        {
            unified: note(unified, production_cue="a mound over a grave"),
            compatibility: note(compatibility, reading_example=bound("塚穴", "つかあな")),
        },
    )
    before = path.read_bytes()

    with pytest.raises(KanjiNoteError) as raised:
        kanji_notes.load_notes(path)

    message = str(raised.value)
    assert "U+585A" in message and "U+FA10" in message
    assert unified in message and compatibility in message
    assert "kanji:塚" in message
    assert str(path) in message
    assert path.read_bytes() == before, "a refusal never rewrites the store"


def test_a_compatibility_key_on_its_own_still_loads_under_its_folded_identity(
    tmp_path: Path,
) -> None:
    """The refusal above is about a *collision*, not about the fold. One entry
    filed under 塚 U+FA10 keeps loading as ``kanji:塚``, under the key it was
    written with: the store does not rename or normalize what it reads."""
    compatibility = chr(0xFA10)
    path = tmp_path / "kanji_notes.json"
    kanji_notes.save_notes(path, {compatibility: note(compatibility)})

    loaded = kanji_notes.load_notes(path)

    assert list(loaded) == [compatibility]
    assert loaded[compatibility].id == "kanji:塚"


def test_recognition_is_the_only_direction_a_bare_note_supports() -> None:
    """The default: five characters, five notes, five cards."""
    assert kanji_notes.supported_directions(note(reading_example=None)) == frozenset(
        {"recognition"}
    )


def test_reading_needs_a_fixed_bound_example_and_evidence_alone_is_not_one() -> None:
    """Evidence can carry a percentage with no example attached to it. A
    reading card with nothing to read is not a card."""
    without = note(reading_evidence=evidence(), reading_example=None)
    assert "reading" not in kanji_notes.supported_directions(without)

    with_prompt = note(reading_example=bound())
    assert "reading" in kanji_notes.supported_directions(with_prompt)


def test_production_needs_a_nonblank_cue_nobody_wrote_for_the_owner() -> None:
    assert "production" not in kanji_notes.supported_directions(note(production_cue="  "))
    assert "production" in kanji_notes.supported_directions(note(production_cue="reason"))


def test_the_fixed_prompt_is_the_first_example_the_provider_bound() -> None:
    """Source order, and nothing else: choosing between two examples on any
    other basis would be janki deciding which Japanese teaches a reading."""
    first, second = bound("料理", "りょうり"), bound("理由", "りゆう")
    facts = evidence(examples=(first, second))

    assert kanji_notes.first_bound_example(facts) == first
    assert kanji_notes.first_bound_example(evidence()) is None
    assert kanji_notes.first_bound_example(None) is None


# --- the provider snapshot a note carries ------------------------------------


def snapshot(
    *, corpus_scope: str | None = None, denominator: str | None = None
) -> CharacterReadings:
    """One facts entry with every field the provider states filled in.

    Every value is distinct and none of them is a dataclass default, so a field
    the store drops on the way out, or fails to read on the way back, cannot
    land on the value it started with and pass unnoticed. The strings are
    opaque: what has to survive is *that* each one came back, not what it says.

    Groups, readings and examples are in an order no sort would reproduce.
    """
    return CharacterReadings(
        character="理",
        source_url="https://jpdb.io/kanji/%E7%90%86",
        fetched_at_utc="2026-09-07T00:00:00Z",
        sha256="c" * 64,
        corpus_scope=corpus_scope,
        denominator=denominator,
        groups=(
            ReadingGroup(
                source_class="kanji-reading-list-common",
                readings=(
                    ReadingUsage(
                        label="り",
                        href="/kanji/%E7%90%86#り",
                        percent_text="(84%)",
                        percent=84,
                        percent_less_than=False,
                        examples=(
                            BoundExample(
                                written="無理やり",
                                pronounced="むりやり",
                                gloss="forcibly",
                                furigana="無[む] 理[り]やり",
                                source_url="https://jpdb.io/vocabulary/1531030/a#b",
                            ),
                            BoundExample(
                                written="料理",
                                pronounced="りょうり",
                                gloss="cooking",
                                furigana="料[りょう] 理[り]",
                                source_url="https://jpdb.io/vocabulary/1550140/c#d",
                            ),
                        ),
                        detail_source_url="https://jpdb.io/kanji-reading/%E7%90%86/り",
                        detail_fetched_at_utc="2026-09-06T23:00:00Z",
                        detail_sha256="d" * 64,
                    ),
                    ReadingUsage(
                        label="ことわり",
                        href="/kanji/%E7%90%86#ことわり",
                        percent_text="(<1%)",
                        percent=1,
                        percent_less_than=True,
                        detail_source_url="https://jpdb.io/kanji-reading/%E7%90%86/ことわり",
                        detail_fetched_at_utc="2026-09-06T23:01:00Z",
                        detail_sha256="e" * 64,
                    ),
                ),
            ),
            ReadingGroup(
                source_class="kanji-reading-list",
                readings=(
                    ReadingUsage(
                        label="おさ",
                        href="/kanji/%E7%90%86#おさ",
                        percent_text=None,
                        percent=None,
                        percent_less_than=None,
                    ),
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("corpus_scope", "denominator"),
    [(None, None), ("the uses jpdb counted", "1487")],
    ids=["unknown", "supplied"],
)
def test_the_whole_provider_snapshot_survives_the_store(
    tmp_path: Path, corpus_scope: str | None, denominator: str | None
) -> None:
    """A note carries a *copy* of what the provider said, so a card can state
    where its figures came from and when. Every label, figure, bound example,
    URL, retrieval time and response hash therefore comes back exactly as it
    went in — including the printed bound on a figure, the reading the provider
    quantified nothing for, and the two fields it states nothing for at all,
    which are unknown rather than absent.
    """
    readings = snapshot(corpus_scope=corpus_scope, denominator=denominator)
    before = readings.to_dict()
    original = note(
        "理",
        reading_evidence=kanji_notes.evidence_from_readings(readings),
        reading_example=readings.groups[0].readings[0].examples[0],
    )
    path = tmp_path / "kanji_notes.json"

    kanji_notes.save_notes(path, {"理": original})
    reloaded = kanji_notes.load_notes(path)["理"]

    assert reloaded == original, "the whole note, field for field"
    assert reloaded.reading_evidence is not None
    assert reloaded.reading_evidence.readings.to_dict() == before
    assert reloaded.reading_evidence.source == jpdb_kanji.SOURCE == "jpdb"
    assert reloaded.reading_evidence.metric == jpdb_kanji.METRIC == "jpdb_reported_usage"
    assert path.read_text(encoding="utf-8") == kanji_notes.render_notes({"理": original})


def test_the_snapshots_source_order_is_the_order_the_store_returns(
    tmp_path: Path,
) -> None:
    """Which reading the provider printed first, and which word it listed first
    under one, are facts about the page. This fixture's order is one no sort
    would produce, so a store that tidied it up would say the page said
    something else — and the first bound example is the prompt a reading card
    is fixed to for the life of its review history."""
    original = note("理", reading_evidence=kanji_notes.evidence_from_readings(snapshot()))
    path = tmp_path / "kanji_notes.json"

    kanji_notes.save_notes(path, {"理": original})
    stored = kanji_notes.load_notes(path)["理"].reading_evidence
    assert stored is not None

    assert [group.source_class for group in stored.readings.groups] == [
        "kanji-reading-list-common",
        "kanji-reading-list",
    ]
    assert [
        usage.label for group in stored.readings.groups for usage in group.readings
    ] == ["り", "ことわり", "おさ"]
    assert [
        example.written for example in stored.readings.groups[0].readings[0].examples
    ] == ["無理やり", "料理"]
    first = kanji_notes.first_bound_example(stored)
    assert first is not None and first.written == "無理やり"


def test_the_stored_note_carries_the_providers_own_fields_and_no_others(
    tmp_path: Path,
) -> None:
    """The durable bytes, not only the objects rebuilt from them. A field the
    store stops writing is a figure a rebuilt card can no longer attribute; one
    it invents is a claim the provider never made. A value the provider left
    unknown is written as null rather than left out, because "asked, and it said
    nothing" is a different fact from "never asked"."""
    path = tmp_path / "kanji_notes.json"
    kanji_notes.save_notes(
        path,
        {
            "理": note(
                "理",
                reading_evidence=kanji_notes.evidence_from_readings(
                    snapshot(corpus_scope="the uses jpdb counted", denominator="1487")
                ),
            )
        },
    )
    stored = json.loads(path.read_text(encoding="utf-8"))["notes"]["理"]
    evidence_wire = stored["reading_evidence"]
    quantified = evidence_wire["readings"]["groups"][0]["readings"][0]
    unquantified = evidence_wire["readings"]["groups"][1]["readings"][0]

    assert set(evidence_wire) == {"source", "metric", "readings"}
    assert set(evidence_wire["readings"]) == {
        "character",
        "source_url",
        "fetched_at_utc",
        "sha256",
        "corpus_scope",
        "denominator",
        "groups",
    }
    assert set(evidence_wire["readings"]["groups"][0]) == {"source_class", "readings"}
    assert set(quantified) == {
        "label",
        "href",
        "percent_text",
        "percent",
        "percent_less_than",
        "examples",
        "detail_source_url",
        "detail_fetched_at_utc",
        "detail_sha256",
    }
    assert set(quantified["examples"][0]) == {
        "written",
        "pronounced",
        "gloss",
        "furigana",
        "source_url",
    }
    assert quantified["percent_text"] == "(84%)"
    assert quantified["detail_sha256"] == "d" * 64
    assert unquantified["percent_text"] is None
    assert unquantified["percent"] is None
    assert unquantified["percent_less_than"] is None
    assert unquantified["detail_source_url"] is None
