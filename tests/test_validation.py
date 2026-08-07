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
    assert any("data/staging" in issue.message for issue in issues)
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
    assert any("data/staging" in message for message in messages)
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
