"""Staging files: records a human still has to look at before they are real.

A staging file is ordinary YAML — a mapping with a ``records:`` list (exactly
the shape :func:`japanese_anki.io.load_records` already accepts, so
``janki validate`` works on it unchanged) plus metadata keys the record loader
ignores (``source_file``, ``extracted_at``, ``model``, ``review_notes``).

Per-record review annotations (``hold_reason``, ``already_known``,
``suggested_reading``) live in that record's ``source.raw_fields`` as strings,
so a staged record round-trips through ``VocabularyRecord.from_dict`` with no
schema additions and no bespoke loader.

Staging files hold un-committed human edits — the one thing in this repo git
cannot recover — so :func:`write_staging` refuses to overwrite one unless the
caller explicitly passes ``force=True``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from japanese_anki.errors import JankiError
from japanese_anki.io import atomic_write_text, load_structured
from japanese_anki.models import VocabularyRecord


class StagingError(JankiError):
    pass


# Review annotations, stored stringified in ``source.raw_fields``.
ANNOTATION_KEYS: tuple[str, ...] = ("hold_reason", "already_known", "suggested_reading")

# Metadata keys that sit beside ``records:``; the record loader ignores them.
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
    disk may hold hand-edited readings that exist nowhere else. Also refuses a
    suffix :func:`read_staging` could not parse — the content is YAML whatever
    the name says, so any other suffix produces a file only this function can
    make sense of.
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
            f"Staging file already exists: {path}. It may hold review edits that are "
            "not in git; move it aside or re-run with force to overwrite."
        )

    payload: dict[str, Any] = {}
    for key, value in (meta or {}).items():
        name = str(key)
        if name == _RECORDS_KEY:
            raise StagingError(f"Staging metadata cannot use the reserved key '{_RECORDS_KEY}'")
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
