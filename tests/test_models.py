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
