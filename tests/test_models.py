"""``from_dict`` on malformed nested types: a clean error, never a traceback.

Every loader — the normalized file, deck inline notes, the ``--replace``
recovery count — funnels through these constructors. Before ``ModelError``,
a string ``examples`` or a scalar ``conjugations``/``source`` escaped as a
raw ``AttributeError`` from deep inside ``models.py``, killing ``janki
status`` (contracted to warn and skip), ``janki build``, and the one command
able to recover a broken vocabulary.json.
"""

from __future__ import annotations

from typing import Any

import pytest

from japanese_anki.errors import JankiError
from japanese_anki.models import ModelError, VocabularyRecord


def _raw(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "word:食べる:たべる",
        "expression": "食べる",
        "reading": "たべる",
        "meanings": ["to eat"],
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("examples", "毎日食べる。"),  # a string instead of a list
        ("examples", ["毎日食べる。"]),  # a list of strings instead of mappings
        ("examples", 5),
        ("conjugations", "dict form"),  # a scalar instead of a mapping
        ("conjugations", ["dict form"]),
        ("source", "shirabe"),  # a scalar instead of a mapping
        ("source", ["shirabe"]),
        ("source", {"type": "shirabe", "raw_fields": "vid=1"}),
    ],
)
def test_a_malformed_nested_field_is_a_clean_model_error(
    field_name: str, value: Any
) -> None:
    with pytest.raises(ModelError) as excinfo:
        VocabularyRecord.from_dict(_raw(**{field_name: value}))

    message = str(excinfo.value)
    assert field_name.split(".")[0] in message  # names the field
    assert "must be" in message  # says what shape was expected


def test_model_error_is_a_janki_error() -> None:
    # cli.main catches JankiError once; a ModelError that fell outside that
    # would be a traceback for the user again.
    assert issubclass(ModelError, JankiError)


@pytest.mark.parametrize("empty", [None, "", []])
def test_an_empty_nested_field_still_reads_as_absent(empty: Any) -> None:
    # The constructors have always read empty values as "no data"; only a
    # non-empty value of the wrong type is malformed.
    record = VocabularyRecord.from_dict(
        _raw(examples=empty, conjugations=empty, source=empty)
    )

    assert record.examples == []
    assert record.conjugations == {}
    assert record.source.type == "manual"


def test_a_single_example_mapping_is_still_wrapped_into_a_list() -> None:
    record = VocabularyRecord.from_dict(_raw(examples={"japanese": "毎日食べる。"}))

    assert [example.japanese for example in record.examples] == ["毎日食べる。"]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        # The natural YAML mistake: a mapping of languages under `meanings:`.
        ("meanings", {"en": "to run", "ja": "走ること"}),
        ("meanings", 5),
        ("meanings", True),
        ("tags", {"level": "N5"}),
        ("tags", 5),
    ],
)
def test_a_malformed_meanings_or_tags_list_is_a_clean_model_error(
    field_name: str, value: Any
) -> None:
    # The old fallback was `[str(value)]`, so this landed as one list entry
    # holding a Python repr — rendered onto the Anki card, written back to
    # vocabulary.json, and then treated by the next import as curated content
    # it must not overwrite.
    with pytest.raises(ModelError) as excinfo:
        VocabularyRecord.from_dict(_raw(**{field_name: value}))

    message = str(excinfo.value)
    assert field_name in message
    assert "must be a string or a list of strings" in message


@pytest.mark.parametrize("field_name", ["meanings", "tags"])
@pytest.mark.parametrize("value", ["one", ["one", "two"], ("one",), None, [], ""])
def test_the_shapes_meanings_and_tags_have_always_accepted_still_work(
    field_name: str, value: Any
) -> None:
    record = VocabularyRecord.from_dict(_raw(**{field_name: value}))

    assert all(isinstance(item, str) for item in getattr(record, field_name))


def test_a_model_error_excerpts_the_offending_value_rather_than_printing_it() -> None:
    # The realistic trigger is a long pasted block scalar written under
    # `examples:`; `janki build` prints the whole message on one stderr line
    # after naming the file, and an error as large as the malformed field is
    # not the clean error these constructors promise.
    with pytest.raises(ModelError) as excinfo:
        VocabularyRecord.from_dict(_raw(examples="Z" * 200_000))

    message = str(excinfo.value)
    assert len(message) < 300
    assert "str" in message
    assert message.endswith("...)")


# --- the M2.2 schema additions ----------------------------------------------


def test_the_new_schema_fields_default_to_empty() -> None:
    # Nothing writes them yet (jpdb enrichment does, in M2.6), so every record
    # loaded today gets the defaults — and each default has to read as *empty*
    # to the merge, or the first enrichment pass would find no holes to fill.
    record = VocabularyRecord.from_dict(_raw())

    assert record.pitch_accent == []
    assert record.audio_accent == ""
    assert record.frequency_rank is None
    assert VocabularyRecord.from_dict(
        _raw(examples=[{"japanese": "毎日食べる。"}])
    ).examples[0].audio == ""


def test_the_new_fields_round_trip_through_to_dict() -> None:
    # The one-time whole-file rewrite the design predicts: every stored record
    # gains these keys the next time vocabulary.json is saved.
    stored = VocabularyRecord.from_dict(
        _raw(
            pitch_accent=["LHHH", "LHHL"],
            audio_accent="LHHL",
            frequency_rank=1234,
            examples=[{"japanese": "毎日食べる。", "audio": "janki-abc.wav"}],
        )
    ).to_dict()

    assert stored["pitch_accent"] == ["LHHH", "LHHL"]
    assert stored["audio_accent"] == "LHHL"
    assert stored["frequency_rank"] == 1234
    assert stored["examples"][0]["audio"] == "janki-abc.wav"
    assert VocabularyRecord.from_dict(stored).to_dict() == stored


@pytest.mark.parametrize(
    ("value", "expected"),
    [("LHHH", ["LHHH"]), (["LHHH", "LHHL"], ["LHHH", "LHHL"]), (None, []), ("", [])],
)
def test_pitch_accent_accepts_one_pattern_or_several(value: Any, expected: list[str]) -> None:
    # A single pattern written as a bare string is the shape a hand-edited YAML
    # file takes; it reads as the one-entry list the field is documented as.
    assert VocabularyRecord.from_dict(_raw(pitch_accent=value)).pitch_accent == expected


@pytest.mark.parametrize("value", [{"primary": "LHHH"}, 5, True])
def test_a_malformed_pitch_accent_is_a_clean_model_error(value: Any) -> None:
    with pytest.raises(ModelError) as excinfo:
        VocabularyRecord.from_dict(_raw(pitch_accent=value))

    assert "pitch_accent" in str(excinfo.value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1234, 1234),
        # CSV columns are text and a JSON export can round-trip an int through
        # float; both are the same rank.
        ("1234", 1234),
        (1234.0, 1234),
        # Zero is a rank, not an absence: the merge would refill a hole.
        (0, 0),
        (None, None),
        ("", None),
    ],
)
def test_frequency_rank_coerces_the_shapes_a_source_actually_writes(
    value: Any, expected: int | None
) -> None:
    assert VocabularyRecord.from_dict(_raw(frequency_rank=value)).frequency_rank == expected


@pytest.mark.parametrize("value", ["very common", 12.5, [1234], True])
def test_a_frequency_rank_that_is_not_a_number_is_a_clean_model_error(value: Any) -> None:
    # Silently dropping it to None would be worse than refusing: a rank that
    # vanished on load looks exactly like one nobody has fetched yet, so the
    # next enrichment pass overwrites it instead of reporting it.
    with pytest.raises(ModelError) as excinfo:
        VocabularyRecord.from_dict(_raw(frequency_rank=value))

    message = str(excinfo.value)
    assert "frequency_rank" in message
    assert "whole number" in message


def test_a_null_raw_field_is_dropped_rather_than_stringified() -> None:
    # `str(None)` is the literal "None": a non-empty value nobody typed, which
    # `status --duplicates` would then group records on. An absent key is what
    # a JSON null in that column means.
    record = VocabularyRecord.from_dict(
        _raw(source={"type": "jpdb", "raw_fields": {"vid": None, "freq": 12}})
    )

    assert record.source.raw_fields == {"freq": "12"}
