from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import stable_record_id


class ModelError(JankiError):
    """A record's raw data holds a nested field of the wrong type.

    Raised by the ``from_dict`` constructors instead of letting a stray
    ``AttributeError`` escape from deep inside them: every loader (the
    normalized file, deck inline notes, the ``--replace`` recovery count)
    funnels through these constructors, and each of those callers promises a
    clean error, a warning, or a skip — never a traceback. Defined here rather
    than reusing ``io.DataError`` because ``io`` imports this module; it
    subclasses ``JankiError``, so ``cli.main`` and the status warn-and-skip
    guard already handle it.
    """


_EXCERPT_LIMIT = 120


def _excerpt(value: Any) -> str:
    """``repr(value)``, short enough to read on one line.

    The realistic trigger for these errors is a long pasted block scalar
    written where a list or a mapping belongs, and an error message as large as
    the malformed field is not a clean error. The type name is the actionable
    half; the value is context.
    """
    text = repr(value)
    return text if len(text) <= _EXCERPT_LIMIT else text[: _EXCERPT_LIMIT - 3] + "..."


def _checked_mapping(value: Any, field_name: str, what: str) -> dict[str, Any]:
    """``value`` as the mapping ``field_name`` requires, or a clean error.

    Anything empty (``None``, ``""``, ``[]``) reads as an absent mapping, the
    way these constructors always read it; only a non-empty value of the wrong
    type is refused, which used to escape as an ``AttributeError`` traceback.
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    raise ModelError(
        f"'{field_name}' must be {what}, got {type(value).__name__} ({_excerpt(value)})"
    )


def _string_list(value: Any, field_name: str) -> list[str]:
    """``value`` as the list of strings ``field_name`` requires, or an error.

    The fallback used to be ``[str(value)]``, which turned a mapping written
    under ``meanings:`` into one list entry holding its Python repr — silent
    coercion of exactly the kind ``_checked_mapping`` exists to refuse, and
    worse here: the repr is rendered onto an Anki card and written back to
    vocabulary.json, where the next import treats it as curated content it must
    not overwrite.
    """
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ModelError(
        f"'{field_name}' must be a string or a list of strings, got "
        f"{type(value).__name__} ({_excerpt(value)})"
    )


@dataclass(slots=True)
class ExampleSentence:
    japanese: str = ""
    furigana: str = ""
    romaji: str = ""
    english: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExampleSentence:
        data = _checked_mapping(data, "examples", "a list of example mappings")
        return cls(
            japanese=str(data.get("japanese", "")).strip(),
            furigana=str(data.get("furigana", "")).strip(),
            romaji=str(data.get("romaji", "")).strip(),
            english=str(data.get("english", "")).strip(),
        )


@dataclass(slots=True)
class SourceReference:
    type: str = "manual"
    imported_from: str = ""
    row: int | None = None
    raw_fields: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SourceReference:
        data = _checked_mapping(data, "source", "a mapping")
        row_value = data.get("row")
        try:
            row = int(row_value) if row_value not in (None, "") else None
        except (TypeError, ValueError):
            row = None
        # A ``None`` is dropped rather than stringified: ``str(None)`` is the
        # literal ``"None"``, a non-empty value nobody typed, and downstream
        # readers group records by these strings (``status --duplicates`` on
        # ``vid``). An absent key is what a null column means.
        raw_fields = {
            str(key): str(value)
            for key, value in _checked_mapping(
                data.get("raw_fields"), "source.raw_fields", "a mapping"
            ).items()
            if value is not None
        }
        return cls(
            type=str(data.get("type", "manual")).strip() or "manual",
            imported_from=str(data.get("imported_from", "")).strip(),
            row=row,
            raw_fields=raw_fields,
        )


@dataclass(slots=True)
class VocabularyRecord:
    id: str
    expression: str
    reading: str = ""
    furigana: str = ""
    romaji: str = ""
    meanings: list[str] = field(default_factory=list)
    part_of_speech: str = ""
    verb_group: str = ""
    transitivity: str = ""
    examples: list[ExampleSentence] = field(default_factory=list)
    conjugations: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    usage_notes: str = ""
    audio: str = ""
    image: str = ""
    source: SourceReference = field(default_factory=SourceReference)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VocabularyRecord:
        expression = str(data.get("expression", "")).strip()
        reading = str(data.get("reading", "")).strip()
        record_id = str(data.get("id", "")).strip() or stable_record_id(expression, reading)
        examples_value = data.get("examples") or []
        if isinstance(examples_value, dict):
            examples_value = [examples_value]
        if not isinstance(examples_value, list | tuple):
            raise ModelError(
                "'examples' must be a list of example mappings, got "
                f"{type(examples_value).__name__} ({_excerpt(examples_value)})"
            )
        conjugations = {
            str(key).strip(): str(value).strip()
            for key, value in _checked_mapping(
                data.get("conjugations"), "conjugations", "a mapping of form names to text"
            ).items()
            if str(key).strip() and str(value).strip()
        }
        return cls(
            id=record_id,
            expression=expression,
            reading=reading,
            furigana=str(data.get("furigana", "")).strip(),
            romaji=str(data.get("romaji", "")).strip(),
            meanings=_string_list(data.get("meanings"), "meanings"),
            part_of_speech=str(data.get("part_of_speech", "")).strip(),
            verb_group=str(data.get("verb_group", "")).strip(),
            transitivity=str(data.get("transitivity", "")).strip(),
            examples=[ExampleSentence.from_dict(item) for item in examples_value],
            conjugations=conjugations,
            tags=_string_list(data.get("tags"), "tags"),
            usage_notes=str(data.get("usage_notes", "")).strip(),
            audio=str(data.get("audio", "")).strip(),
            image=str(data.get("image", "")).strip(),
            source=SourceReference.from_dict(data.get("source")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def first_example(self) -> ExampleSentence:
        return self.examples[0] if self.examples else ExampleSentence()
