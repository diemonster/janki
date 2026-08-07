"""Import a Shirabe Jisho CSV export.

Everything that is not Shirabe-specific lives in :mod:`csv_base`; what remains
here is the header vocabulary, the source label, the automatic tag and the
error type. If you are about to add behaviour to this module, check whether it
belongs to CSV import in general — the jpdb importer reads the same base.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from japanese_anki.errors import JankiError
from japanese_anki.importers import csv_base
from japanese_anki.importers.csv_base import (
    FIELD_ALIASES,
    ImportResult,
    InspectionResult,
)

__all__ = [
    "FIELD_ALIASES",
    "ImportResult",
    "InspectionResult",
    "ShirabeImportError",
    "detect_mapping",
    "import_file",
    "inspect_file",
]


class ShirabeImportError(JankiError):
    pass


# Read at call time, not baked into SHIRABE, so shrinking it makes the
# oversized-field path testable without writing a gigabyte fixture.
_CSV_FIELD_LIMIT = csv_base.CSV_FIELD_LIMIT

SHIRABE = csv_base.csv_format(
    source_type="shirabe",
    aliases=FIELD_ALIASES,
    tags=("shirabe",),
    error=ShirabeImportError,
)


def detect_mapping(headers: Iterable[str]) -> dict[str, str]:
    return csv_base.detect_mapping(headers, SHIRABE)


def inspect_file(path: Path, sample_size: int = 5) -> InspectionResult:
    return csv_base.inspect_file(
        path, SHIRABE, sample_size, field_limit=_CSV_FIELD_LIMIT
    )


def import_file(path: Path) -> ImportResult:
    return csv_base.import_file(path, SHIRABE, field_limit=_CSV_FIELD_LIMIT)
