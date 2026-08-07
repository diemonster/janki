from __future__ import annotations

import csv
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji, stable_record_id
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.staging import annotate


class ShirabeImportError(JankiError):
    pass


FIELD_ALIASES: dict[str, set[str]] = {
    "expression": {
        "word",
        "expression",
        "term",
        "japanese",
        "kanji",
        "vocabulary",
        "単語",
        "表記",
        "見出し",
    },
    "reading": {
        "reading",
        "kana",
        "pronunciation",
        "yomi",
        "読み",
        "よみ",
        "かな",
    },
    "furigana": {"furigana", "ruby", "振り仮名", "ふりがな"},
    "romaji": {"romaji", "romanization", "ローマ字"},
    "meanings": {
        "meaning",
        "meanings",
        "definition",
        "definitions",
        "english",
        "gloss",
        "意味",
        "英語",
    },
    "part_of_speech": {"partofspeech", "pos", "wordtype", "品詞"},
    "verb_group": {"verbgroup", "conjugationclass", "動詞グループ"},
    "transitivity": {"transitivity", "transitive", "自他", "他動詞自動詞"},
    "tags": {"tag", "tags", "folder", "category", "bookmark", "タグ", "フォルダ"},
    "notes": {"note", "notes", "memo", "comment", "メモ", "ノート"},
    "example_japanese": {"example", "examplejapanese", "sentence", "例文"},
    "example_english": {"exampleenglish", "translation", "例文英訳"},
}


@dataclass(frozen=True, slots=True)
class InspectionResult:
    delimiter: str
    headers: list[str]
    mapping: dict[str, str]
    unknown_headers: list[str]
    sample_rows: list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class ImportResult:
    records: list[VocabularyRecord]
    warnings: list[str]
    mapping: dict[str, str]
    # Rows whose reading slot is not usable kana — missing, or itself written in
    # kanji. Never importable, because the reading is part of the ID. They go to
    # a staging file for a human to fill in.
    needs_reading: list[VocabularyRecord] = field(default_factory=list)


def _normalize_header(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return re.sub(r"[^0-9a-zぁ-んァ-ヶ一-龯]+", "", normalized)


_NORMALIZED_ALIASES = {
    field: {_normalize_header(alias) for alias in aliases}
    for field, aliases in FIELD_ALIASES.items()
}


def detect_mapping(headers: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for header in headers:
        normalized = _normalize_header(header)
        for canonical, aliases in _NORMALIZED_ALIASES.items():
            if canonical not in mapping and normalized in aliases:
                mapping[canonical] = header
                break
    return mapping


def _open_reader(path: Path) -> tuple[csv.DictReader, object, str]:
    handle = path.open("r", encoding="utf-8-sig", newline="")
    sample = handle.read(8192)
    handle.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(handle, dialect=dialect)
    delimiter = getattr(dialect, "delimiter", ",")
    return reader, handle, delimiter


def inspect_file(path: Path, sample_size: int = 5) -> InspectionResult:
    try:
        reader, handle, delimiter = _open_reader(path)
    except (OSError, UnicodeDecodeError) as exc:
        raise ShirabeImportError(f"Could not read {path}: {exc}") from exc

    try:
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ShirabeImportError(f"CSV has no header row: {path}")
        mapping = detect_mapping(headers)
        rows = []
        for row in reader:
            rows.append({str(key): str(value or "") for key, value in row.items()})
            if len(rows) >= sample_size:
                break
    finally:
        handle.close()

    used_headers = set(mapping.values())
    return InspectionResult(
        delimiter=delimiter,
        headers=headers,
        mapping=mapping,
        unknown_headers=[header for header in headers if header not in used_headers],
        sample_rows=rows,
    )


def _split_meanings(value: str) -> list[str]:
    if not value.strip():
        return []
    parts = re.split(r"\s*(?:\r?\n|;|\||\s/\s)\s*", value.strip())
    return [part for part in parts if part]


def _split_tags(value: str) -> list[str]:
    if not value.strip():
        return []
    return [part for part in re.split(r"[,;\s]+", value.strip()) if part]


def _get(row: dict[str, str], mapping: dict[str, str], field: str) -> str:
    header = mapping.get(field)
    return str(row.get(header, "") if header else "").strip()


def import_file(path: Path) -> ImportResult:
    """Convert a Shirabe-style CSV into records.

    A row whose reading slot does not end up holding usable kana is *not*
    imported, in either of the two ways that happens: the reading is missing
    (the ID would be minted as ``word:<expression>:``), or the reading is
    itself kanji (``word:<kanji>:<kanji>``, which looks well formed and is
    not). The reading is ID-constitutive, so neither could be corrected later
    without orphaning Anki review history. Those rows come back on
    ``needs_reading`` (as minted, malformed ID included) for the caller to
    stage for human review.

    Kana-only rows are different: their reading is defaulted to the expression
    *before* the ID is minted, which is where it has always happened. Do not
    move it — every existing kana-only record's ID depends on it. The checks
    below run *after* both defaults, so a defaulted reading is tested too.
    """
    inspection = inspect_file(path)
    mapping = inspection.mapping
    if "expression" not in mapping and "reading" not in mapping:
        raise ShirabeImportError(
            "Could not identify a word/expression or reading column. "
            f"Headers were: {', '.join(inspection.headers)}"
        )

    reader, handle, _ = _open_reader(path)
    warnings: list[str] = []
    records: list[VocabularyRecord] = []
    needs_reading: list[VocabularyRecord] = []
    seen: set[str] = set()

    try:
        for row_number, source_row in enumerate(reader, start=2):
            row = {str(key): str(value or "") for key, value in source_row.items()}
            expression = _get(row, mapping, "expression")
            reading = _get(row, mapping, "reading")
            if not expression:
                # A row that filled in only the Reading column: that value is
                # the whole word. It may well be kanji, so it is re-tested
                # below rather than trusted as a reading.
                expression = reading
            if not reading and expression and not contains_kanji(expression):
                reading = expression
            if not expression:
                warnings.append(f"{path.name}:{row_number}: skipped empty row")
                continue

            record_id = stable_record_id(expression, reading)
            if record_id in seen:
                warnings.append(
                    f"{path.name}:{row_number}: duplicate row for {expression} [{reading}]"
                )
                continue
            seen.add(record_id)

            example_japanese = _get(row, mapping, "example_japanese")
            example_english = _get(row, mapping, "example_english")
            examples = []
            if example_japanese or example_english:
                examples.append(
                    ExampleSentence(japanese=example_japanese, english=example_english)
                )

            tags = sorted(set(["shirabe", *_split_tags(_get(row, mapping, "tags"))]))
            record = VocabularyRecord(
                id=record_id,
                expression=expression,
                reading=reading,
                furigana=_get(row, mapping, "furigana"),
                romaji=_get(row, mapping, "romaji"),
                meanings=_split_meanings(_get(row, mapping, "meanings")),
                part_of_speech=_get(row, mapping, "part_of_speech"),
                verb_group=_get(row, mapping, "verb_group"),
                transitivity=_get(row, mapping, "transitivity"),
                examples=examples,
                tags=tags,
                usage_notes=_get(row, mapping, "notes"),
                source=SourceReference(
                    type="shirabe",
                    imported_from=path.name,
                    row=row_number,
                    raw_fields=row,
                ),
            )
            # Reached with both defaults applied, so `not reading` means the
            # expression has kanji (a kana-only expression defaulted its own
            # reading) and a kanji reading means the row gave janki nothing but
            # the written form.
            if not reading:
                warnings.append(
                    f"{path.name}:{row_number}: {expression} contains kanji but has no "
                    "reading; held back for reading review"
                )
                needs_reading.append(annotate(record, hold_reason="missing reading"))
                continue
            if contains_kanji(reading):
                warnings.append(
                    f"{path.name}:{row_number}: the reading for {expression} is written "
                    f"in kanji ({reading}); held back for reading review"
                )
                needs_reading.append(annotate(record, hold_reason="reading contains kanji"))
                continue
            records.append(record)
    finally:
        handle.close()

    return ImportResult(
        records=records,
        warnings=warnings,
        mapping=mapping,
        needs_reading=needs_reading,
    )
