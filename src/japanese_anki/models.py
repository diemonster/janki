from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji, short_fingerprint, stable_record_id


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


def _optional_int(value: Any, field_name: str) -> int | None:
    """``value`` as the optional whole number ``field_name`` requires.

    Empty (``None`` or ``""``) is the absent value, the way every other field
    here reads emptiness; ``0`` is a number and survives. A string is accepted
    because both of this project's readers hand one over — CSV columns are
    always text, and a JSON export that wrote ``1234.0`` round-trips through
    ``float``. Anything else is a ``ModelError`` rather than a silent ``None``:
    a rank that vanished on load looks exactly like a rank nobody has fetched
    yet, so the next enrichment pass would overwrite the value instead of
    reporting it. ``bool`` is refused explicitly — it is an ``int`` subclass in
    Python, and ``frequency_rank: true`` meaning rank 1 is nonsense.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ModelError(f"'{field_name}' must be a whole number or empty, got bool ({value!r})")
    if isinstance(value, int):
        return value
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        raise ModelError(
            f"'{field_name}' must be a whole number or empty, got "
            f"{type(value).__name__} ({_excerpt(value)})"
        ) from None
    if not number.is_integer():
        raise ModelError(
            f"'{field_name}' must be a whole number or empty, got {_excerpt(value)}"
        )
    return int(number)


@dataclass(slots=True)
class ExampleSentence:
    japanese: str = ""
    furigana: str = ""
    romaji: str = ""
    english: str = ""
    # Media-dir-relative filename of this sentence's generated audio (M5.3).
    audio: str = ""
    #: Optional human-written text sent to the sentence provider instead of
    #: ``japanese``. It is deliberately absent from model-generated example
    #: schemas: this is the escape hatch for a person who listened, not another
    #: answer for the model to invent. Janki never derives it from furigana.
    spoken_japanese: str = ""
    #: ``polite`` (〜ます/です) or ``casual`` (plain form), or empty for an
    #: example written before the distinction existed. A learner meets both and
    #: they are not interchangeable — a textbook teaches ます first and a friend
    #: never uses it — so a card that shows only one teaches half the word.
    register: str = ""

    def needs_ai_annotations(self) -> bool:
        """Whether a curated sentence still has visible holes.

        Completeness only — whether this example's text may be *pinned* for
        annotation is an authority question the sentence cannot answer about
        itself; ``models.example_accepted`` reads that off the record.
        """
        return bool(self.japanese) and (
            not self.english
            or self.register not in {"polite", "casual"}
            or (contains_kanji(self.japanese) and not self.furigana)
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        position: int | None = None,
    ) -> ExampleSentence:
        data = _checked_mapping(data, "examples", "a list of example mappings")
        spoken_value = data.get("spoken_japanese", "")
        if spoken_value is None:
            spoken_japanese = ""
        elif not isinstance(spoken_value, str):
            field_name = (
                f"examples[{position}].spoken_japanese"
                if position is not None
                else "examples.spoken_japanese"
            )
            raise ModelError(
                f"'{field_name}' must be text or empty, got "
                f"{type(spoken_value).__name__} ({_excerpt(spoken_value)})"
            )
        else:
            spoken_japanese = spoken_value.strip()
        return cls(
            japanese=str(data.get("japanese", "")).strip(),
            furigana=str(data.get("furigana", "")).strip(),
            romaji=str(data.get("romaji", "")).strip(),
            english=str(data.get("english", "")).strip(),
            audio=str(data.get("audio", "")).strip(),
            spoken_japanese=spoken_japanese,
            register=str(data.get("register", "")).strip().lower(),
        )


@dataclass(frozen=True, slots=True)
class SourceFormColumn:
    """One column a source printed, by stable identity and display label.

    ``id`` is the opaque identity the owner minted when they bound the table's
    layout; ``label`` is the display string they chose for it. Two columns may
    share a ``label`` precisely because the identities differ — the duplicate
    heading that made a printed label unusable as a key is the thing opaque
    identities remove.
    """

    id: str
    label: str


@dataclass(frozen=True, slots=True)
class SourceFormsTable:
    """The exact ordered columns a source printed, and the cells it filled.

    ``columns`` is printed order. ``cells`` is keyed by column identity and is
    preserved **verbatim**: a blank printed cell is a declared row with an
    empty value, and a column this word has no cell for stays absent. That
    distinction is what ``conjugations`` cannot carry — it trims its values and
    drops the blank ones — and it is the whole reason this field exists rather
    than being folded into that map.

    ``cells`` is an ordinary mutable ``dict`` inside a frozen wrapper, so
    ``frozen=True`` detaches nothing; ``io._copy_value``'s ``deepcopy`` is what
    keeps a merged record from aliasing an import the caller is still writing.
    """

    columns: tuple[SourceFormColumn, ...] = ()
    cells: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> SourceFormsTable | None:
        """The canonical wire ``{"columns": [{id,label}], "cells": {id: str}}``.

        ``None`` for an absent or null key **and** for the one deliberately
        defined empty table — no columns *and* no cells — because
        ``repairs._require_canonical_record`` requires
        ``to_dict → from_dict → to_dict`` to be equal, and an empty table that
        existed in memory as a present object would also look non-empty to
        ``io.merge_records``. A table that declares columns is never empty, so
        provided columns are never silently discarded.

        Structural refusals only. A duplicate identity makes the cell map
        ambiguous and a cell naming no declared column has no label binding and
        no witness: both are facts about the artifact, and neither reads the
        text of a cell.
        """
        if data is None:
            return None
        raw = _checked_mapping(
            data, "source_forms", "a mapping with 'columns' and 'cells'"
        )
        if not raw:
            return None
        unknown = set(raw) - {"columns", "cells"}
        if unknown:
            raise ModelError(
                "'source_forms' holds only 'columns' and 'cells', not "
                + ", ".join(sorted(repr(name) for name in unknown))
            )
        raw_columns = raw.get("columns")
        if raw_columns is None:
            raw_columns = []
        if not isinstance(raw_columns, list | tuple):
            raise ModelError(
                "'source_forms.columns' must be a list of {id, label} mappings, "
                f"got {type(raw_columns).__name__} ({_excerpt(raw_columns)})"
            )
        columns: list[SourceFormColumn] = []
        seen: set[str] = set()
        for position, entry in enumerate(raw_columns):
            where = f"source_forms.columns[{position}]"
            if not isinstance(entry, dict) or set(entry) != {"id", "label"}:
                raise ModelError(
                    f"'{where}' must be a mapping with exactly 'id' and 'label', "
                    f"got {_excerpt(entry)}"
                )
            column_id = entry["id"]
            label = entry["label"]
            if not isinstance(column_id, str) or not column_id:
                raise ModelError(
                    f"'{where}.id' must be a nonblank column identity, got "
                    f"{_excerpt(column_id)}"
                )
            if not isinstance(label, str):
                raise ModelError(
                    f"'{where}.label' must be text, got {type(label).__name__} "
                    f"({_excerpt(label)})"
                )
            if column_id in seen:
                raise ModelError(
                    f"'source_forms' declares column {column_id!r} twice; two "
                    "identities that are one identity make its cell ambiguous."
                )
            seen.add(column_id)
            # Preserved exactly as supplied. The owner authored this identity
            # and this label in their own editor; trimming or normalizing here
            # would change the thing they bound.
            columns.append(SourceFormColumn(id=column_id, label=label))
        raw_cells = _checked_mapping(
            raw.get("cells"), "source_forms.cells", "a mapping of column ids to text"
        )
        cells: dict[str, str] = {}
        for key, value in raw_cells.items():
            if not isinstance(key, str) or key not in seen:
                raise ModelError(
                    f"'source_forms.cells' holds {key!r}, which no declared "
                    "column identifies. A cell with no column has no label "
                    "binding and no witness."
                )
            if not isinstance(value, str):
                raise ModelError(
                    f"'source_forms.cells[{key!r}]' must be text, got "
                    f"{type(value).__name__} ({_excerpt(value)})"
                )
            # Verbatim: no strip, no blank-dropping. A printed blank is study
            # content and an absent column is a different fact.
            cells[key] = value
        if not columns and not cells:
            return None
        return cls(columns=tuple(columns), cells=cells)

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON containers, so both YAML writers and the JSON writer
        see the shapes they already write."""

        return {
            "columns": [
                {"id": column.id, "label": column.label} for column in self.columns
            ],
            "cells": dict(self.cells),
        }

    def is_blank(self) -> bool:
        """The one empty spelling: no columns and no cells."""

        return not self.columns and not self.cells

    def rows(self) -> list[tuple[str, str]]:
        """``(label, cell)`` for every declared column that has a cell.

        Printed order. A blank cell is a row whose value is empty; a column
        with no cell draws nothing. No code here matches a printed label or
        reads a cell's text to decide either.
        """

        return [
            (column.label, self.cells[column.id])
            for column in self.columns
            if column.id in self.cells
        ]


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
    # jpdb-style accent patterns over the reading's kana plus the following
    # particle slot ("LHHH"); possibly several, first entry primary. Written by
    # enrichment (M2.6), read by the pitch converter (M5.1) and the exporter
    # (M5.4) — no note field carries it before then.
    pitch_accent: list[str] = field(default_factory=list)
    # Per-record override of the pattern audio generation forces. Empty means
    # "use pitch_accent[0]".
    audio_accent: str = ""
    # jpdb corpus rank. ``None`` is "never looked up"; the rank itself is a
    # number, so 0 would be a value and not a hole.
    frequency_rank: int | None = None
    #: The exact table this record's source printed, when one was bound. It
    #: selects the rows of the same Anki ``Conjugations`` field the deck already
    #: ships; where it is absent the computed ``conjugations`` map still does.
    #: ``None`` on every record written before or without a bound layout, and
    #: omitted from ``to_dict`` there, so no existing record's bytes change.
    source_forms: SourceFormsTable | None = None
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
            examples=[
                ExampleSentence.from_dict(item, position=position)
                for position, item in enumerate(examples_value)
            ],
            conjugations=conjugations,
            tags=_string_list(data.get("tags"), "tags"),
            usage_notes=str(data.get("usage_notes", "")).strip(),
            audio=str(data.get("audio", "")).strip(),
            image=str(data.get("image", "")).strip(),
            pitch_accent=_string_list(data.get("pitch_accent"), "pitch_accent"),
            audio_accent=str(data.get("audio_accent", "")).strip(),
            frequency_rank=_optional_int(data.get("frequency_rank"), "frequency_rank"),
            source_forms=SourceFormsTable.from_dict(data.get("source_forms")),
            source=SourceReference.from_dict(data.get("source")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # The override is deliberately sparse. ``audio`` rewrites the complete
        # collection after changing one record, so ordinary dataclass
        # serialization must not add ``spoken_japanese: ""`` to every example.
        # Only the human-authored replacement input is stored.
        for example in payload["examples"]:
            if not str(example.get("spoken_japanese") or "").strip():
                example.pop("spoken_japanese", None)
        # Sparse for the same reason, and plain containers for another:
        # ``asdict`` would hand a tuple of column mappings through, which
        # ``json.dumps`` and ``yaml.safe_dump`` both accept and the next reader
        # would not. One canonical spelling on both sides is what keeps
        # ``to_dict → from_dict → to_dict`` equal for every repair pass.
        if self.source_forms is None:
            payload.pop("source_forms", None)
        else:
            payload["source_forms"] = self.source_forms.to_dict()
        return payload

    @property
    def first_example(self) -> ExampleSentence:
        return self.examples[0] if self.examples else ExampleSentence()

    def main_example(self) -> ExampleSentence:
        """The example the card shows unlabelled, or an empty one.

        The first that is not casual — by *register*, not by identity. Nothing
        orders `examples`: `enrich --ai` appends them as the model answered, so
        `examples[0]` is not the polite one just because it is first, and a
        record whose casual sentence leads used to show no main example at all.
        Excluding only the object `example_in("casual")` returned had the same
        shape of bug one step along: a record with two casual sentences put the
        second in the unlabelled slot, where the card presents it as the neutral
        form of the word.

        Empty when every example is casual, so the casual slot holds it and the
        card does not claim a contrast it does not have.

        Lives here because both readers must agree: `exporters.anki` builds the
        note and `preview` renders the same record for the browser, and a
        preview showing a different sentence than the deck it previews is not a
        preview.
        """
        for example in self.examples:
            if example.register.strip().lower() != "casual":
                return example
        return ExampleSentence()

    def example_in(self, register: str) -> ExampleSentence:
        """The first example written in this register, or an empty one.

        Falls back to nothing rather than to another register: a card slot
        labelled "casual" holding a ます sentence teaches the opposite of what
        it says. An example written before the field existed carries no
        register and answers neither.
        """
        wanted = register.strip().lower()
        for example in self.examples:
            if example.register.strip().lower() == wanted:
                return example
        return ExampleSentence()


# --- authority provenance ----------------------------------------------------
#
# The M7.6T trust keys and their machinery live beside the record type rather
# than in `staging` because every layer that moves records needs them without
# a cycle: `extract` and `enrich` write and resolve the marks, `audio` and
# `qc` read the holds, and `io`'s merge must carry them per *field* when an
# import fills a hole — and `io` is below `staging` in the import graph.
# Every consumer imports these names from here.

#: Field-level acceptance provenance for an extract-sourced record's examples.
#: The reviewer types the sentinel value (``staging-review``) into a staging
#: row's ``raw_fields``, or explicitly checks that row in the localhost review
#: panel, which writes the accepted sentences' content fingerprints directly.
#: ``promote._accept_examples`` replaces the manual sentinel with those same
#: fingerprints. A sentence added or rewritten later carries no covering
#: fingerprint and is simply not accepted; a stale stamp can never bless text
#: no reviewer read.
EXAMPLE_AUTHORITY_KEY = "example_authority"
EXAMPLE_AUTHORITY_STAGING = "staging-review"


def example_accepted(record: VocabularyRecord, example: ExampleSentence) -> bool:
    """Whether this example may be treated as accepted teaching content.

    The per-sentence trust rule the camera pilot forced into words: on an
    **extract**-sourced record a sentence is accepted only when the reviewer's
    promote-time acceptance covers its exact Japanese. Every other source type
    — a Shirabe export, an Anki import, a hand-written record — is the user's
    own data, curated by arrival.
    """
    if record.source.type != "extract":
        return True
    return short_fingerprint(example.japanese) in example_flags(
        record, EXAMPLE_AUTHORITY_KEY
    )


def flag_entries(record: VocabularyRecord, key: str) -> list[str]:
    """``key``'s fingerprints in stored order — the one parse of the wire form.

    Ordered, because the writers below re-serialize what this returns and a
    set-shaped read would make the stored value churn between runs.
    """
    raw = record.source.raw_fields.get(key, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def example_flags(record: VocabularyRecord, key: str) -> set[str]:
    """The example content-fingerprints ``key`` flags on this record.

    One parser for every comma-joined fingerprint flag, so a format change
    lands in one place instead of six.
    """
    return set(flag_entries(record, key))


def _write_flags(
    record: VocabularyRecord, key: str, fingerprints: Iterable[str]
) -> VocabularyRecord:
    """The one flag serializer: dedupe in order, remove the key when empty.

    Every public writer is a one-line policy over this core, so the wire form
    — and the no-empty-value, no-op-identity rules — cannot drift between
    them.
    """
    ordered = [item for item in dict.fromkeys(fingerprints) if item]
    raw_fields = dict(record.source.raw_fields)
    if ordered:
        raw_fields[key] = ",".join(ordered)
    else:
        raw_fields.pop(key, None)
    if raw_fields == record.source.raw_fields:
        return record
    return replace(record, source=replace(record.source, raw_fields=raw_fields))


def add_example_flags(
    record: VocabularyRecord, key: str, sentences: Iterable[str]
) -> VocabularyRecord:
    """Flag ``sentences`` under ``key``, by content fingerprint, keeping order.

    A fingerprint rather than an index, because an index stops meaning
    anything the moment a human deletes an example — and these keys are read
    much later, deciding whether to speak a sentence. Existing flags are kept
    and deduplicated, never replaced: a record collects them across passes.
    """
    return _write_flags(
        record,
        key,
        flag_entries(record, key)
        + [short_fingerprint(sentence) for sentence in sentences],
    )


def set_example_flags(
    record: VocabularyRecord, key: str, sentences: Iterable[str]
) -> VocabularyRecord:
    """Replace ``key`` with exactly ``sentences``' fingerprints, or remove it.

    The replace-semantics sibling of :func:`add_example_flags`, for writers
    whose statement is "this is the complete covered set" — promotion binding
    a reviewer's acceptance to the sentences they read. An empty set removes
    the key: a flag list naming nothing is a standing claim waiting to be
    misread.
    """
    return _write_flags(
        record, key, [short_fingerprint(sentence) for sentence in sentences]
    )

#: Authority state for semantic fields a model filled during extraction. The
#: marker is ``name:fingerprint`` pairs, comma-joined. The fingerprint binds
#: the mark to the *value* the model wrote: a human who edits the field
#: afterwards breaks the binding, and a broken binding reads as "curated" —
#: the mark must never authorize overwriting an edit a person made after
#: extraction.
PROVISIONAL_FIELDS_KEY = "provisional_fields"

#: The only fields extraction may mark provisional. The reading is pointedly
#: absent: it is half of the record ID, reviewed by a human at staging, and no
#: dictionary evidence is allowed to rewrite it.
PROVISIONAL_SEMANTIC_FIELDS: tuple[str, ...] = ("meanings", "part_of_speech")


def _provisional_fingerprint(value: Any) -> str:
    return short_fingerprint(json.dumps(value, ensure_ascii=False))


def mark_provisional(record: VocabularyRecord) -> VocabularyRecord:
    """Stamp the semantic fields a model filled as provisional claims.

    Written at extraction time, because that is the moment the values are
    known to be model output and nothing else: one step later they sit in a
    staging file beside human edits and the distinction is unrecoverable. A
    field the model left empty gets no mark — emptiness is not a claim.
    """
    entries = [
        (name, _provisional_fingerprint(getattr(record, name)))
        for name in PROVISIONAL_SEMANTIC_FIELDS
        if getattr(record, name)
    ]
    if not entries:
        return record
    raw_fields = dict(record.source.raw_fields)
    raw_fields[PROVISIONAL_FIELDS_KEY] = join_provisional_entries(entries)
    return replace(record, source=replace(record.source, raw_fields=raw_fields))


def join_provisional_entries(entries: Iterable[tuple[str, str]]) -> str:
    """``(name, fingerprint)`` pairs in the marker's wire form.

    The serialization counterpart of :func:`provisional_entries`, and the only
    place the wire form is spelled — three writers (extraction's mark, the
    clear, and ``io``'s merge carry) drifting on it would silently widen or
    lose dictionary-overwrite authority.
    """
    return ",".join(f"{name}:{fingerprint}" for name, fingerprint in entries)


def split_provisional(record: VocabularyRecord) -> tuple[list[str], list[str]]:
    """The marker split into ``(active, stale)`` by its value binding.

    One comparison site on purpose: active and stale are the two halves of a
    single question — does the field still hold the value the mark was bound
    to? — and answering it in two places would let the answers drift. A field
    is *active* (still the model's claim) while the fingerprint matches;
    an edit after extraction breaks the binding and makes the entry *stale* —
    curated content wearing a mark that must now be cleared, never obeyed.
    """
    active: list[str] = []
    stale: list[str] = []
    for name, fingerprint in provisional_entries(record):
        matches = _provisional_fingerprint(getattr(record, name)) == fingerprint
        (active if matches else stale).append(name)
    return active, stale


def provisional_fields(record: VocabularyRecord) -> list[str]:
    """The fields whose current value is still the model's provisional claim."""
    return split_provisional(record)[0]


def clear_provisional(
    record: VocabularyRecord, names: Iterable[str]
) -> VocabularyRecord:
    """Drop resolved or stale names from the marker, removing it when empty."""
    dropped = set(names)
    kept = [
        (name, fingerprint)
        for name, fingerprint in provisional_entries(record)
        if name not in dropped
    ]
    raw_fields = dict(record.source.raw_fields)
    if kept:
        raw_fields[PROVISIONAL_FIELDS_KEY] = join_provisional_entries(kept)
    else:
        raw_fields.pop(PROVISIONAL_FIELDS_KEY, None)
    if raw_fields == record.source.raw_fields:
        return record
    return replace(record, source=replace(record.source, raw_fields=raw_fields))


def provisional_entries(record: VocabularyRecord) -> list[tuple[str, str]]:
    """The marker parsed to ``(name, fingerprint)``, unknown names dropped.

    Unknown or malformed entries are ignored rather than errors: the marker
    rides in hand-editable YAML, and the failure mode to prevent is a stray
    edit *widening* what a dictionary may overwrite. Public because ``io``'s
    merge must carry a filled field's mark by name, not as an opaque blob.
    """
    raw = record.source.raw_fields.get(PROVISIONAL_FIELDS_KEY, "")
    entries: list[tuple[str, str]] = []
    for item in raw.split(","):
        name, _, fingerprint = item.strip().partition(":")
        if name in PROVISIONAL_SEMANTIC_FIELDS and fingerprint:
            entries.append((name, fingerprint))
    return entries
