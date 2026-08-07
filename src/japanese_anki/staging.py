"""Staging files: records a human still has to look at before they are real.

A staging file is ordinary YAML — a mapping with a ``records:`` list (exactly
the shape :func:`japanese_anki.io.load_records` already accepts, so
``janki validate`` works on it unchanged) plus metadata keys the record loader
ignores (``source_file``, ``extracted_at``, ``model``, ``review_notes``).

Per-record review annotations (``hold_reason``, ``already_known``,
``suggested_reading``) live in that record's ``source.raw_fields`` as strings,
so a staged record round-trips through ``VocabularyRecord.from_dict`` with no
schema additions and no bespoke loader.

``data/staging/`` is tracked (see AGENTS.md "Data lifecycle"), so a reading
typed in by hand is recoverable once it is committed — but only then, and a
review is usually mid-flight when the next import runs. :func:`write_staging`
therefore refuses to overwrite an existing file unless the caller explicitly
passes ``force=True``. In-place annotation of a file under review (M2.6's
``--staging``) is the case that legitimately passes it.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml import YAML, YAMLError

from japanese_anki.errors import JankiError
from japanese_anki.io import atomic_write_text, load_structured
from japanese_anki.models import VocabularyRecord


class StagingError(JankiError):
    pass


# Review annotations, stored stringified in ``source.raw_fields``.
ANNOTATION_KEYS: tuple[str, ...] = ("hold_reason", "already_known", "suggested_reading")

# Metadata keys that sit beside ``records:``; the record loader ignores them.
# Enforced by :func:`write_staging` as a warning, not a refusal: `read_staging`
# hands back every non-``records`` key it finds, so a note a reviewer added by
# hand has to survive a round trip. What the warning catches is a *writer*
# inventing a key nothing downstream reads.
META_KEYS: tuple[str, ...] = ("source_file", "extracted_at", "model", "review_notes")

_RECORDS_KEY = "records"

# ``read_staging`` parses by suffix (via ``load_structured``), so a staging file
# written under any other suffix would be write-only: the write succeeds and
# every read of it fails.
STAGING_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")


def _stringify(value: Any) -> str:
    """Render an annotation value as a string ``raw_fields`` can hold."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def annotate(record: VocabularyRecord, **values: Any) -> VocabularyRecord:
    """Return a copy of ``record`` carrying review annotations.

    Values are stringified into ``source.raw_fields``; passing ``None`` removes
    an annotation (how a resolved hold is cleared). Only names in
    ``ANNOTATION_KEYS`` are accepted — raw_fields is otherwise the importer's
    verbatim source row and must not become a junk drawer.
    """
    raw_fields = dict(record.source.raw_fields)
    for key, value in values.items():
        if key not in ANNOTATION_KEYS:
            raise StagingError(
                f"Unknown staging annotation '{key}'. Valid: {', '.join(ANNOTATION_KEYS)}"
            )
        if value is None:
            raw_fields.pop(key, None)
        else:
            raw_fields[key] = _stringify(value)
    return replace(record, source=replace(record.source, raw_fields=raw_fields))


def annotations(record: VocabularyRecord) -> dict[str, str]:
    """The review annotations present on ``record``, in ``ANNOTATION_KEYS`` order."""
    raw_fields = record.source.raw_fields
    return {key: raw_fields[key] for key in ANNOTATION_KEYS if key in raw_fields}


def write_staging(
    path: Path,
    records: Iterable[VocabularyRecord],
    meta: Mapping[str, Any] | None = None,
    force: bool = False,
) -> Path:
    """Write ``records`` and ``meta`` to a staging file at ``path``.

    Refuses to overwrite an existing file unless ``force`` is true: the file on
    disk may hold hand-edited readings not yet committed. Also refuses a suffix
    :func:`read_staging` could not parse — the content is YAML whatever the name
    says, so any other suffix produces a file only this function can make sense
    of. A metadata key outside :data:`META_KEYS` is written, with a warning: it
    is readable but nothing downstream looks at it.
    """
    path = Path(path)
    if path.suffix.lower() not in STAGING_SUFFIXES:
        raise StagingError(
            f"Staging files are YAML: {path} would be written as YAML under a "
            f"'{path.suffix}' name and read_staging parses by suffix, so nothing "
            f"could read it back. Use {' or '.join(STAGING_SUFFIXES)}."
        )
    if path.exists() and not force:
        raise StagingError(
            f"Staging file already exists: {path}. It may hold review edits you have "
            "not committed; move it aside or re-run with force to overwrite."
        )

    payload: dict[str, Any] = {}
    for key, value in (meta or {}).items():
        name = str(key)
        if name == _RECORDS_KEY:
            raise StagingError(f"Staging metadata cannot use the reserved key '{_RECORDS_KEY}'")
        if name not in META_KEYS:
            print(
                f"warning: staging metadata key '{name}' in {path} is not one of "
                f"{', '.join(META_KEYS)}; it will be written and read back, but no "
                "janki command looks at it",
                file=sys.stderr,
            )
        payload[name] = value
    payload[_RECORDS_KEY] = [record.to_dict() for record in records]

    text = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
    atomic_write_text(path, text)
    return path


def _parser() -> YAML:
    """The round-trip parser used to edit a file in place, configured once."""
    parser = YAML()  # round-trip mode: comments, order and quoting survive
    parser.preserve_quotes = True
    parser.allow_unicode = True
    parser.width = 100
    return parser


def _load_document(path: Path) -> Any:
    """A staging file as an editable round-trip document.

    Every ruamel failure becomes a :class:`StagingError` naming the path. Two
    are worth expecting: ordinary syntax errors, and a **duplicate key**, which
    PyYAML accepts silently (last value wins) and ruamel refuses. Refusing is
    the right answer for the writing path even though the reading path is
    lenient — dumping a document whose duplicate ruamel collapsed would delete
    one of the reviewer's two lines, and this function will not touch a file it
    cannot rewrite faithfully. :func:`check_rewritable` exists so a caller can
    find that out before it spends an API pass.
    """
    try:
        with Path(path).open(encoding="utf-8") as handle:
            document = _parser().load(handle)
    except YAMLError as exc:
        raise StagingError(f"Could not read {path} for rewriting: {exc}") from exc
    if not isinstance(document, MutableMapping) or _RECORDS_KEY not in document:
        raise StagingError(
            f"Expected a staging mapping with a '{_RECORDS_KEY}:' list in {path}"
        )
    return document


def check_rewritable(path: Path) -> None:
    """Raise :class:`StagingError` now if :func:`rewrite_staging` could not write.

    For callers that do expensive work — an API pass — between reading a
    staging file and writing it back. Failing at the end of that work would
    have spent it for nothing and would greet the user with an error after
    they had already confirmed the change.
    """
    _load_document(path)


def _apply_changes(
    target: MutableMapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    """Write only what actually changed into ``target``, recursing into mappings.

    Keys whose value is unchanged are never touched — not rewritten, and not
    *added* when the file left them out. That is what keeps a hand-written
    staging row from acquiring twenty empty schema fields the moment janki
    annotates it, and what leaves every key janki does not know about exactly
    where the reviewer put it.
    """
    for key, new_value in after.items():
        old_value = before.get(key)
        if old_value == new_value:
            continue
        current = target.get(key)
        if (
            isinstance(new_value, Mapping)
            and isinstance(old_value, Mapping)
            and isinstance(current, MutableMapping)
        ):
            _apply_changes(current, old_value, new_value)
            continue
        target[key] = new_value


def rewrite_staging(path: Path, records: Sequence[VocabularyRecord]) -> Path:
    """Update an existing staging file in place, preserving what janki does not own.

    :func:`write_staging` renders a file from records, which is right when it is
    creating one and destructive when it is not: a staging file under review is
    the one place in this repository holding work that exists nowhere else, and
    a load-then-dump round trip through the record schema silently deletes a
    reviewer's YAML comments and any key the schema has no field for. This
    reads the document, writes back only the values that actually changed, and
    leaves the rest of the file — comments, key order, quoting, unknown keys —
    byte-for-byte as it was.

    ``records`` must be the list :func:`read_staging` returned for this file,
    in order, with the same length; rows are matched positionally. A caller that
    adds or removes rows is not annotating a review, it is writing a new file,
    and should say so with :func:`write_staging`.

    **Both sides of the diff come from the same parser**, which is not a detail.
    The document is edited through ruamel (YAML 1.2) while ``records`` came from
    :func:`read_staging` (PyYAML, YAML 1.1), and the two dialects disagree about
    real values a reviewer types: 1.1 reads ``yes`` as a boolean and ``12:30``
    as the sexagesimal integer 750, 1.2 reads both as the strings they look
    like. Diffing a ruamel-parsed baseline against PyYAML-derived records would
    call every such value "changed" and write janki's reading of it over the
    reviewer's line — the exact thing this function exists to prevent. So the
    baseline is re-read with :func:`read_staging` too: the diff is then between
    two same-dialect readings, and only a key janki genuinely changed is ever
    written.
    """
    path = Path(path)
    if not path.exists():
        raise StagingError(
            f"No staging file to update at {path}; write_staging creates one."
        )

    original, _meta = read_staging(path)
    document = _load_document(path)
    raw_records = document[_RECORDS_KEY] or []
    if not (len(raw_records) == len(original) == len(records)):
        raise StagingError(
            f"{path} holds {len(raw_records)} row(s) but {len(records)} were given. "
            "rewrite_staging annotates the rows already in a file; use write_staging "
            "to write a different set."
        )

    for raw, before, after in zip(raw_records, original, records, strict=True):
        if not isinstance(raw, MutableMapping):
            raise StagingError(
                f"Record in {path} must be a mapping, got {type(raw).__name__}"
            )
        _apply_changes(raw, before.to_dict(), after.to_dict())

    buffer = io.StringIO()
    _parser().dump(document, buffer)
    atomic_write_text(path, buffer.getvalue())
    return path


def read_staging(path: Path) -> tuple[list[VocabularyRecord], dict[str, Any]]:
    """Read a staging file, returning ``(records, meta)``.

    ``meta`` is every top-level key except ``records``, so a hand-added note
    survives a read/write round trip.
    """
    path = Path(path)
    data = load_structured(path)
    if not isinstance(data, Mapping):
        raise StagingError(
            f"Expected a staging mapping with a '{_RECORDS_KEY}:' list in {path}, "
            f"got {type(data).__name__}"
        )
    if _RECORDS_KEY not in data:
        raise StagingError(f"Staging file {path} has no '{_RECORDS_KEY}:' list")

    raw_records = data[_RECORDS_KEY] or []
    if not isinstance(raw_records, list):
        raise StagingError(f"'{_RECORDS_KEY}' in {path} must be a list of records")

    records: list[VocabularyRecord] = []
    for index, item in enumerate(raw_records, start=1):
        if not isinstance(item, Mapping):
            raise StagingError(
                f"Record {index} in {path} must be a mapping, got {type(item).__name__}"
            )
        records.append(VocabularyRecord.from_dict(dict(item)))

    meta = {str(key): value for key, value in data.items() if key != _RECORDS_KEY}
    return records, meta
