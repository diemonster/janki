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


def test_a_null_raw_field_is_dropped_rather_than_stringified() -> None:
    # `str(None)` is the literal "None": a non-empty value nobody typed, which
    # `status --duplicates` would then group records on. An absent key is what
    # a JSON null in that column means.
    record = VocabularyRecord.from_dict(
        _raw(source={"type": "jpdb", "raw_fields": {"vid": None, "freq": 12}})
    )

    assert record.source.raw_fields == {"freq": "12"}
