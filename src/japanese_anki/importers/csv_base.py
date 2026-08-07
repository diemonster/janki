"""CSV machinery shared by every vocabulary importer.

Shirabe exports and the JPDB-Export userscript are the same problem wearing
different headers: sniff a delimiter, map columns onto canonical field names,
turn each row into a :class:`VocabularyRecord`, and hold back the rows whose
reading cannot serve as part of an ID. Only the header vocabulary, the source
label, the automatic tags and the error type differ, so those are what a
:class:`CsvFormat` carries and everything else lives here once.

The rules encoded below are not cosmetic and are the reason this is shared
rather than copied:

* A reading is ID-constitutive. A row whose reading slot is missing, or is
  itself written in kanji, is *not* imported — it is returned on
  ``needs_reading`` for a human to fill in. See :func:`import_file`.
* A kana-only row defaults its reading to its expression **before** the ID is
  minted. Every existing kana-only record's ID depends on that order.
* Unknown columns are preserved in ``source.raw_fields`` rather than dropped.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji, stable_record_id
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.romaji import kana_to_romaji
from japanese_anki.staging import annotate

# csv.field_size_limit defaults to 128KB, which a long pasted article in a
# Notes cell exceeds. Raised generously (1 GiB fits a C long everywhere) so a
# big field imports instead of crashing; a field bigger than this is a broken
# file, reported through _rows. The limit counts *characters* and CPython's
# _csv buffers them, so reaching it takes roughly 12 GB of RAM — a file that
# large will hit MemoryError first, which is not a JankiError and is not this
# module's to catch.
#
# Every entry point takes it as an argument rather than reading this global, so
# a specialization can hold its own (and a test can shrink it to something a
# fixture can actually exceed).
CSV_FIELD_LIMIT = 2**30


# The header vocabulary every CSV importer starts from. Exporters disagree
# about what to call the same column, so a format specializes by *adding* to
# this rather than replacing it — a Shirabe export that happens to say
# "Spelling" should map as readily as a jpdb one.
FIELD_ALIASES: dict[str, set[str]] = {
    "expression": {
        "word",
        "expression",
        "term",
        "japanese",
        "kanji",
        "vocabulary",
        # jpdb's own name for the written form, via the JPDB-Export userscript.
        "spelling",
        "単語",
        "表記",
        "見出し",
    },
    "reading": {
        "reading",
        "kana",
        "pronunciation",
        "yomi",
        # Also the userscript's, whose reading column is furigana-annotated.
        # Unverified against a real export — see jpdb_import.
        "furigana reading",
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


def _normalize_header(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return re.sub(r"[^0-9a-zぁ-んァ-ヶ一-龯]+", "", normalized)


def normalize_aliases(aliases: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    """Pre-normalize an alias table so header matching is a set lookup.

    Normalization strips case, spacing and punctuation, which is why a format
    may spell an alias however it reads best — ``"furigana reading"`` and
    ``"Furigana_Reading"`` both arrive here as ``furiganareading``.
    """
    return {
        canonical: {_normalize_header(alias) for alias in names}
        for canonical, names in aliases.items()
    }


@dataclass(frozen=True, slots=True)
class CsvFormat:
    """One CSV dialect of vocabulary: its headers, label, tags and error type.

    ``aliases`` must already be normalized — build one with :func:`csv_format`
    rather than by hand, or header matching silently stops working for every
    alias whose spelling needed normalizing.
    """

    source_type: str
    aliases: Mapping[str, set[str]]
    tags: tuple[str, ...]
    error: type[JankiError]


def csv_format(
    *,
    source_type: str,
    aliases: Mapping[str, Iterable[str]],
    tags: Iterable[str],
    error: type[JankiError],
) -> CsvFormat:
    """Build a :class:`CsvFormat`, normalizing its alias table."""
    return CsvFormat(
        source_type=source_type,
        aliases=normalize_aliases(aliases),
        tags=tuple(tags),
        error=error,
    )


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


def _rows(
    reader: csv.DictReader, path: Path, error: type[JankiError]
) -> Iterable[dict[str, str | None]]:
    """Iterate CSV rows, turning parser explosions into janki's own error.

    ``csv.Error`` (oversized field, embedded NUL) and a bad byte past the
    sniffing sample both surface mid-iteration; without this they escape as
    raw tracebacks naming neither the file nor the row.

    ``reader.line_num`` is the last line the parser *finished*, so the offending
    line is one past it. The ``+ 1`` is what makes ``path:N`` here mean the same
    thing it means everywhere else in this module, where rows are numbered with
    ``enumerate(..., start=2)`` and the header is line 1 — without it every
    message pointed the user at a perfectly good row.
    """
    iterator = iter(reader)
    while True:
        try:
            row = next(iterator)
        except StopIteration:
            return
        except csv.Error as exc:
            raise error(
                f"{path.name}:{reader.line_num + 1}: could not parse the CSV: {exc}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise error(
                f"{path.name}: not valid UTF-8 after line {reader.line_num}: {exc}"
            ) from exc
        yield row


def detect_mapping(headers: Iterable[str], fmt: CsvFormat) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for header in headers:
        normalized = _normalize_header(header)
        for canonical, aliases in fmt.aliases.items():
            if canonical not in mapping and normalized in aliases:
                mapping[canonical] = header
                break
    return mapping


def _open_reader(path: Path, field_limit: int) -> tuple[csv.DictReader, object, str]:
    csv.field_size_limit(field_limit)
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


def inspect_file(
    path: Path,
    fmt: CsvFormat,
    sample_size: int = 5,
    *,
    field_limit: int = CSV_FIELD_LIMIT,
) -> InspectionResult:
    try:
        reader, handle, delimiter = _open_reader(path, field_limit)
    except (OSError, UnicodeDecodeError) as exc:
        raise fmt.error(f"Could not read {path}: {exc}") from exc

    try:
        try:
            headers = list(reader.fieldnames or [])
        except csv.Error as exc:
            raise fmt.error(
                f"{path.name}:1: could not parse the CSV header: {exc}"
            ) from exc
        if not headers:
            raise fmt.error(f"CSV has no header row: {path}")
        mapping = detect_mapping(headers, fmt)
        rows = []
        for row in _rows(reader, path, fmt.error):
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


def _get(row: dict[str, str], mapping: dict[str, str], name: str) -> str:
    header = mapping.get(name)
    return str(row.get(header, "") if header else "").strip()


def import_file(
    path: Path, fmt: CsvFormat, *, field_limit: int = CSV_FIELD_LIMIT
) -> ImportResult:
    """Convert a vocabulary CSV into records.

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
    inspection = inspect_file(path, fmt, field_limit=field_limit)
    mapping = inspection.mapping
    if "expression" not in mapping and "reading" not in mapping:
        raise fmt.error(
            "Could not identify a word/expression or reading column. "
            f"Headers were: {', '.join(inspection.headers)}"
        )

    reader, handle, _ = _open_reader(path, field_limit)
    warnings: list[str] = []
    records: list[VocabularyRecord] = []
    needs_reading: list[VocabularyRecord] = []
    seen: set[str] = set()

    try:
        for row_number, source_row in enumerate(_rows(reader, path, fmt.error), start=2):
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

            tags = sorted(set([*fmt.tags, *_split_tags(_get(row, mapping, "tags"))]))
            record = VocabularyRecord(
                id=record_id,
                expression=expression,
                reading=reading,
                furigana=_get(row, mapping, "furigana"),
                # Rule-computed, never guessed: romaji is a mechanical
                # transliteration of kana the file already gave us. Only for a
                # reading that is usable kana — a held row's reading is kanji or
                # absent, and transliterating either produces nonsense.
                romaji=_get(row, mapping, "romaji")
                or (kana_to_romaji(reading) if reading and not contains_kanji(reading) else ""),
                meanings=_split_meanings(_get(row, mapping, "meanings")),
                part_of_speech=_get(row, mapping, "part_of_speech"),
                verb_group=_get(row, mapping, "verb_group"),
                transitivity=_get(row, mapping, "transitivity"),
                examples=examples,
                tags=tags,
                usage_notes=_get(row, mapping, "notes"),
                source=SourceReference(
                    type=fmt.source_type,
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
