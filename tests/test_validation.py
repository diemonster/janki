import unicodedata
from dataclasses import replace

import pytest

from japanese_anki.models import VocabularyRecord
from japanese_anki.validation import ValidationIssue, has_errors, validate_records


def _issue_messages(record: VocabularyRecord) -> list[str]:
    return [issue.message for issue in validate_records([record])]


def test_formatted_validation_output_includes_the_stable_code_and_location() -> None:
    issue = ValidationIssue(
        "error",
        "reading is missing",
        record_id="word:話す:はなす",
        source="lesson.yaml",
        code="missing-reading",
    )

    assert issue.format() == (
        "[ERROR missing-reading] lesson.yaml:word:話す:はなす: reading is missing"
    )


def test_validation_requires_reading_for_kanji() -> None:
    record = VocabularyRecord(
        id="word:話す:",
        expression="話す",
        meanings=["to speak"],
    )
    issues = validate_records([record])
    assert has_errors(issues)
    assert any("reading is missing" in issue.message for issue in issues)
    assert any("staging review" in issue.message for issue in issues)
    # One fault, one error: the ID complaint would say the same thing here.
    assert not any("word:<expression>:" in issue.message for issue in issues)


def test_a_reading_less_id_is_an_error_even_once_the_reading_is_filled_in() -> None:
    # The reading was repaired by hand but the ID still carries the empty
    # reading slot Anki's GUID was derived from.
    record = VocabularyRecord(
        id="word:話す:",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
    )

    messages = _issue_messages(record)

    assert has_errors(validate_records([record]))
    assert any("word:<expression>:" in message for message in messages)
    assert any("staging review" in message for message in messages)
    # The reading is present, so only the ID complaint fires.
    assert not any("reading is missing" in message for message in messages)


def test_a_well_formed_id_is_not_flagged() -> None:
    record = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
    )

    assert _issue_messages(record) == []


def test_a_supplementary_plane_kanji_still_needs_a_reading() -> None:
    # 𠮟 is U+20B9F. A kanji test that stops at U+9FFF exempts this record from
    # the validator entirely, so the malformed id it carries is never reported.
    record = VocabularyRecord(
        id="word:𠮟る:",
        expression="𠮟る",
        meanings=["to scold"],
    )

    messages = _issue_messages(record)

    assert has_errors(validate_records([record]))
    assert any("reading is missing" in message for message in messages)


def test_a_reading_written_in_kanji_is_an_error() -> None:
    # The shape an importer mints from a row that supplied only one of the two
    # columns. The id looks well formed and is not: its reading slot holds a
    # spelling, and it is every bit as permanent as word:<expression>:.
    record = VocabularyRecord(
        id="word:話す:話す",
        expression="話す",
        reading="話す",
        meanings=["to speak"],
    )

    messages = _issue_messages(record)

    assert has_errors(validate_records([record]))
    assert any("reading is written in kanji" in message for message in messages)
    assert any("staging review" in message for message in messages)


def test_a_reading_written_in_supplementary_plane_kanji_is_an_error() -> None:
    record = VocabularyRecord(
        id="word:𠮟る:𠮟る",
        expression="𠮟る",
        reading="𠮟る",
        meanings=["to scold"],
    )

    assert any("reading is written in kanji" in message for message in _issue_messages(record))


def test_a_record_that_got_in_with_a_normalizing_kanji_reading_is_still_reported() -> None:
    # The id here is what an import that missed U+2F00 minted: both halves are
    # 一 after NFKC. Validation is the second line of defence and used to miss
    # it for the same reason the import gate did — one function, so one fix.
    record = VocabularyRecord(
        id="word:一:一",
        expression="一",
        reading="⼀",  # U+2F00 KANGXI RADICAL ONE
        meanings=["one"],
    )

    assert any("reading is written in kanji" in message for message in _issue_messages(record))


def test_the_staging_hint_says_what_to_do_without_sending_the_reader_elsewhere() -> None:
    # The remedy has to be readable from the error. Pointing at a document that
    # describes the review the reader just performed is how this check stopped
    # being actionable.
    record = VocabularyRecord(id="word:話す:", expression="話す", meanings=["to speak"])

    message = next(message for message in _issue_messages(record) if "staging review" in message)

    assert "id:" in message
    assert "README" not in message
    # No hardcoded path either: staging_dir is configurable, and a project
    # with staging_dir = "review" has no data/staging directory to look for.
    assert "data/staging" not in message


# --- accent patterns (M2.2's schema additions) ------------------------------


def _accented(pattern: str, *, audio_accent: str = "", reading: str = "はなす") -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:話す:{reading}",
        expression="話す",
        reading=reading,
        meanings=["to speak"],
        pitch_accent=[pattern] if pattern else [],
        audio_accent=audio_accent,
    )


def test_a_well_formed_accent_pattern_is_not_flagged() -> None:
    # One position per kana of はなす plus the particle slot that follows it.
    assert _issue_messages(_accented("LHHH")) == []


@pytest.mark.parametrize("pattern", ["LHH-", "L H H H", "0110", "HLLLx"])
def test_a_pattern_that_is_not_h_and_l_is_an_error(pattern: str) -> None:
    # Not a pattern at all: the converter reads it position by position, so
    # anything else is unusable rather than merely suspicious.
    record = _accented(pattern)

    assert has_errors(validate_records([record]))
    assert any("is not an accent pattern" in message for message in _issue_messages(record))


def test_a_pattern_of_the_wrong_length_warns_rather_than_erroring() -> None:
    # That the pattern covers the following particle is community-verified, not
    # documented, so a mismatch means "look at this", not "this file is wrong".
    record = _accented("LHH")  # 3 positions for a 3-kana reading; 4 expected

    issues = validate_records([record])

    assert not has_errors(issues)
    assert any(
        issue.level == "warning" and "4 were expected" in issue.message for issue in issues
    )


def test_the_audio_override_is_held_to_the_same_rules() -> None:
    # audio_accent is the pattern synthesis actually uses when it is set, so a
    # typo there is the one that reaches the engine.
    record = _accented("LHHH", audio_accent="HxL")

    messages = _issue_messages(record)

    assert has_errors(validate_records([record]))
    assert any("audio_accent" in message and "not an accent pattern" in message
               for message in messages)
    assert not any("pitch_accent[0]" in message for message in messages)


def test_the_length_warning_stays_quiet_while_the_reading_is_empty() -> None:
    # A record with no reading is already reported for that; measuring a pattern
    # against a reading nobody has typed adds a second line for one fix.
    record = VocabularyRecord(
        id="word:ありがとう:ありがとう",
        expression="ありがとう",
        reading="",
        meanings=["thank you"],
        pitch_accent=["LHHH"],
    )

    assert _issue_messages(record) == []


def test_every_pattern_on_the_record_is_checked_not_just_the_primary() -> None:
    record = _accented("LHHH")
    record.pitch_accent = ["LHHH", "not-a-pattern"]

    messages = _issue_messages(record)

    assert any("pitch_accent[1]" in message for message in messages)
    assert not any("pitch_accent[0]" in message for message in messages)


def test_a_kana_only_record_is_not_flagged_for_its_id() -> None:
    # Kana-only records default reading to the expression, so their IDs are
    # well formed; nothing here should look like the malformed shape.
    record = VocabularyRecord(
        id="word:ありがとう:ありがとう",
        expression="ありがとう",
        reading="ありがとう",
        meanings=["thank you"],
    )

    assert _issue_messages(record) == []


def test_a_lower_case_pattern_is_read_rather_than_refused() -> None:
    """`pitch.to_aquestalk` reads h/l deliberately — same data, different
    transcription habit. This check is an *error*, so disagreeing with it would
    make `janki build` refuse a whole deck over a pattern that converts and
    speaks correctly."""
    assert _issue_messages(_accented("lhhh")) == []


def test_a_decomposed_reading_is_counted_in_kana_not_codepoints() -> None:
    """が typed as か + U+3099 is two codepoints and one kana. Counting
    codepoints warns that audio will skip a record `to_aquestalk` handles fine,
    and sends the reviewer to 'fix' a correct pattern into a real mismatch."""
    decomposed = unicodedata.normalize("NFD", "がっこう")
    assert len(decomposed) == 5, "four kana, five codepoints"

    assert _issue_messages(_accented("LHHHH", reading=decomposed)) == []


_SENTENCE = "家族と城崎温泉に行きました。"


def _with_example(furigana: str, japanese: str = _SENTENCE) -> VocabularyRecord:
    from japanese_anki.models import ExampleSentence

    return VocabularyRecord(
        id="word:行く:いく",
        expression="行く",
        reading="いく",
        meanings=["to go"],
        examples=[ExampleSentence(japanese=japanese, furigana=furigana, english="x")],
    )


def test_the_notation_spaces_of_an_ordinary_field_are_not_reported() -> None:
    assert _issue_messages(_with_example("毎晩[まいばん]、 音楽[おんがく]を 聞[き]いて")) == []


def _record_with(**overrides: object) -> VocabularyRecord:
    return replace(
        VocabularyRecord(
            id="word:出発:しゅっぱつ",
            expression="出発",
            reading="しゅっぱつ",
            meanings=["departure"],
        ),
        **overrides,
    )


def test_the_anki_field_separator_in_a_value_is_an_error() -> None:
    """Anki stores a note's fields as one U+001F-joined string, so a separator
    inside a value adds a field: every value after it shifts one position, the
    usage note's tail lands in `Audio`, the audio tag lands in `Image`, and so
    on to the end of the notetype. The build succeeds and says nothing, and
    `janki status --rebuild` reads the collection back through the same split.

    Refused here rather than at the exporter because this is where a build
    already asks, and because an importer is what puts one there: `.strip()`
    takes a separator off either end of a CSV cell — U+001F is whitespace to
    Python — and leaves an interior one untouched."""
    issues = validate_records([_record_with(usage_notes="before\x1fafter")])

    assert has_errors(issues)
    [issue] = [issue for issue in issues if issue.code == "control-character"]
    assert "usage_notes" in issue.message
    assert "U+001F" in issue.message
    # Named as the separator, not as generic non-text: the remedy differs. Any
    # other control character is junk to strip, while this one has shifted
    # every field after it and the note has to be rebuilt.
    assert "separator" in issue.message


def test_a_preserved_source_column_is_not_refused() -> None:
    """`raw_fields` is the verbatim source row, kept so an unknown column is
    never silently discarded. It reaches no note field — the `Source` field is
    built from `type`, `imported_from`, and `row` — so a stray byte in a column
    janki does not map cannot corrupt a note.

    Refusing it would be worse than useless: `has_errors` gates `janki build`
    on the whole deck, so one unreadable byte in one preserved column would
    refuse every card, and the only remedies would be deleting the provenance
    or editing `data/inbox/`."""
    from japanese_anki.models import SourceReference

    record = _record_with(
        source=SourceReference(
            type="csv",
            imported_from="shirabe.csv",
            raw_fields={"DateAdded": "2024-01-01\x7f"},
        )
    )

    assert [
        issue
        for issue in validate_records([record])
        if issue.code == "control-character"
    ] == []


def test_a_control_character_inside_an_example_is_named_by_its_path() -> None:
    """Walked over the record's whole serialized shape, not a list of top-level
    fields: an example's Japanese reaches a note field just as directly, and a
    hand-written field list goes stale the next time one is added."""
    from japanese_anki.models import ExampleSentence

    record = _record_with(
        examples=[ExampleSentence(japanese="本を\x07読む。", english="x")]
    )

    [issue] = [
        issue
        for issue in validate_records([record])
        if issue.code == "control-character"
    ]

    assert "examples[0].japanese" in issue.message
    assert "U+0007" in issue.message


def test_ordinary_whitespace_is_not_a_control_character() -> None:
    """A usage note is prose and may be written across lines; refusing a
    newline would refuse the field's normal content."""
    record = _record_with(usage_notes="first line\nsecond\tline\r\nthird")

    assert [
        issue
        for issue in validate_records([record])
        if issue.code == "control-character"
    ] == []


def test_a_control_character_in_a_field_name_is_caught_too() -> None:
    """`conjugations` keys are rendered into the Conjugations field, so a form
    name carries into a note exactly as a value does. `from_dict` strips the
    key, and U+001F is whitespace to `str.strip` — a leading one is removed and
    an interior one survives.

    Without this the record validates clean and the *build* refuses it, which
    inverts the two checks: `janki validate` says the record is fine and then
    `janki build` will not ship it."""
    record = VocabularyRecord.from_dict(
        {
            "id": "word:出発:しゅっぱつ",
            "expression": "出発",
            "reading": "しゅっぱつ",
            "meanings": ["departure"],
            "conjugations": {"te\x1fform": "出発して"},
        }
    )

    [issue] = [
        issue
        for issue in validate_records([record])
        if issue.code == "control-character"
    ]

    assert "conjugations key" in issue.message
    assert "U+001F" in issue.message


