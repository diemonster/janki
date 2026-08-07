from japanese_anki.models import VocabularyRecord
from japanese_anki.validation import has_errors, validate_records


def _issue_messages(record: VocabularyRecord) -> list[str]:
    return [issue.message for issue in validate_records([record])]


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
