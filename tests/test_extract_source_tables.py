"""Extraction preserves the conjugation tables and chapter labels a source prints.

Both fields are pure transcription: janki copies what the page shows and never
decides which forms a Japanese verb has, which chapter teaches a word, or
whether a printed cell is right. Reading the source is the template's job, so
these tests check that the copy survives the schema, the record build, the
immutable accounting, and a staging round trip unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_extract import candidate, prepared, source_unit, table_extraction

from japanese_anki import extract, prompts
from japanese_anki.extract import build_records
from japanese_anki.staging import read_staging, write_staging

REPO_ROOT = Path(__file__).resolve().parents[1]

#: One arbitrary source's printed columns, in its printed order. Thirteen of
#: them, mixing label languages, out of any sortable order, with one blank cell
#: and one cell the scan clipped — all deliberate, and none of it a set janki
#: knows. A source may print any columns it likes; the point is that whatever
#: it prints comes back in that order.
SOURCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("dictionary", "話す"),
    ("ます", "話します"),
    ("ません", "話しません"),
    ("た", "話した"),
    ("ませんでした", "話しませんでした"),
    ("て", "話して"),
    ("ない", "話さない"),
    ("なかった", "話さなかった"),
    ("potential", "話せる"),
    ("volitional", "話そう"),
    ("imperative", ""),
    ("conditional ば", "話せば"),
    ("passive", "話され"),
)
SOURCE_CHAPTERS = ["16", "3", "第4課"]


def test_supplied_columns_keep_their_labels_values_and_printed_order() -> None:
    """A blank cell stays blank and a clipped cell stays clipped."""
    proposal = candidate(conjugations=dict(SOURCE_COLUMNS))

    assert list(proposal.conjugations.items()) == list(SOURCE_COLUMNS)
    assert proposal.conjugations["imperative"] == ""
    assert proposal.conjugations["passive"] == "話され"


def test_a_source_table_survives_the_parsed_extraction_shape() -> None:
    parsed = table_extraction(
        candidate(
            source_kind="table",
            section="verbs",
            ordinal=1,
            conjugations=dict(SOURCE_COLUMNS),
            source_chapters=["3", "16"],
        ),
        units=[
            source_unit(section="verbs", ordinal=1),
            source_unit(
                section="verbs",
                ordinal=2,
                disposition="duplicate",
                reason="repeat of page 12 verbs row 1 話す／はなす",
            ),
        ],
        count=2,
    )

    [proposal] = parsed.candidates
    assert list(proposal.conjugations.items()) == list(SOURCE_COLUMNS)
    assert proposal.source_chapters == ["3", "16"]
    assert [unit.ordinal for unit in parsed.source_units] == [1, 2]


def test_build_records_carries_every_supplied_column_into_the_record(
    tmp_path: Path,
) -> None:
    """The display map is the canonical one: every supplied column, in order.

    A canonical record has no empty conjugation cell — ``from_dict`` drops one
    and `repairs` refuses a record that does not survive that round trip — so
    the blank column is absent here and present in the witness beside it.
    """
    result = build_records(
        [candidate(conjugations=dict(SOURCE_COLUMNS))], prepared(tmp_path)
    )

    [record] = result.records
    assert list(record.conjugations.items()) == [
        (label, value) for label, value in SOURCE_COLUMNS if value
    ]
    assert list(
        json.loads(record.source.raw_fields["source_conjugations"]).items()
    ) == list(SOURCE_COLUMNS)


def test_the_raw_witness_keeps_the_entire_source_table_verbatim(
    tmp_path: Path,
) -> None:
    """``raw_fields`` is where the source's own table survives intact.

    ``VocabularyRecord.conjugations`` is the canonical display map, and
    ``from_dict`` trims it and drops blank cells on the way back in. That is
    established behaviour and stays. So the authoritative copy of what the page
    printed — every label, every value, the blank cell, the clipped cell, in
    printed order — is written beside the provenance as text.
    """
    result = build_records(
        [candidate(conjugations=dict(SOURCE_COLUMNS))], prepared(tmp_path)
    )

    [record] = result.records
    written = record.source.raw_fields["source_conjugations"]
    assert list(json.loads(written).items()) == list(SOURCE_COLUMNS)
    assert "話しませんでした" in written, "ensure_ascii=False keeps the forms readable"


def test_source_chapters_reach_raw_fields_as_exact_json(tmp_path: Path) -> None:
    result = build_records(
        [candidate(source_chapters=SOURCE_CHAPTERS)], prepared(tmp_path)
    )

    [record] = result.records
    written = record.source.raw_fields["source_chapters"]
    assert json.loads(written) == SOURCE_CHAPTERS
    assert "第4課" in written, "ensure_ascii=False keeps the printed label readable"
    assert record.tags == [], "extraction records provenance; tagging is a later decision"


def test_a_source_stating_neither_leaves_both_empty(tmp_path: Path) -> None:
    proposal = candidate()
    assert proposal.conjugations == {}
    assert proposal.source_chapters == []

    result = build_records([proposal], prepared(tmp_path))

    [record] = result.records
    assert record.conjugations == {}
    assert record.tags == []
    assert "source_chapters" not in record.source.raw_fields
    assert "source_conjugations" not in record.source.raw_fields


def test_the_parsed_proposal_shape_covers_both_new_fields() -> None:
    """The stdlib promote validator must move whenever the paid schema moves."""
    candidate_type = (
        extract.candidate_schema().model_fields["candidates"].annotation.__args__[0]
    )
    assert set(candidate_type.model_fields) == extract._PARSED_CANDIDATE_FIELDS
    assert {"conjugations", "source_chapters"} <= extract._PARSED_CANDIDATE_FIELDS

    dumped = candidate(
        conjugations=dict(SOURCE_COLUMNS), source_chapters=SOURCE_CHAPTERS
    ).model_dump(mode="json")

    assert list(dumped["conjugations"].items()) == list(SOURCE_COLUMNS)
    assert dumped["source_chapters"] == SOURCE_CHAPTERS
    assert extract._is_parsed_schema_proposal(dumped)


def test_accounting_written_before_these_fields_existed_stays_readable() -> None:
    """Paid accounting is immutable, so the older shape has to keep validating.

    Both fields default to empty, so a proposal that omits them is a complete
    proposal — not one to backfill, recompute, or run through an adapter.
    """
    older = candidate().model_dump(mode="json")
    del older["conjugations"]
    del older["source_chapters"]

    assert extract._is_parsed_schema_proposal(older)


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param({"conjugations": ["て", "話して"]}, id="table-is-a-list"),
        pytest.param({"conjugations": {"て": ["話して"]}}, id="cell-is-a-list"),
        pytest.param({"conjugations": {"て": None}}, id="cell-is-null"),
        pytest.param({"conjugations": {"て": 3}}, id="cell-is-a-number"),
        pytest.param({"source_chapters": "3"}, id="chapters-are-one-string"),
        pytest.param({"source_chapters": [3]}, id="chapter-is-a-number"),
        pytest.param({"source_chapters": {"0": "3"}}, id="chapters-are-a-mapping"),
    ],
)
def test_accounting_refuses_a_wrongly_shaped_source_table(
    broken: dict[str, Any],
) -> None:
    proposal = candidate().model_dump(mode="json")
    proposal.update(broken)

    assert not extract._is_parsed_schema_proposal(proposal)


def test_a_collision_group_keeps_both_source_tables_on_every_proposal(
    tmp_path: Path,
) -> None:
    """No Japanese-aware merge decides which row's forms are better."""
    first = candidate(conjugations={"て": "話して"}, source_chapters=["3"])
    second = candidate(
        page=13, conjugations={"volitional": "話そう"}, source_chapters=["16"]
    )

    result = build_records([first, second], prepared(tmp_path))

    [record] = result.records
    assert record.conjugations == {"て": "話して"}
    assert json.loads(record.source.raw_fields["source_chapters"]) == ["3"]

    [group] = result.candidate_accounting["collision_groups"]
    proposals = [item["parsed_schema_proposal"] for item in group["proposals"]]
    assert [item["conjugations"] for item in proposals] == [
        {"て": "話して"},
        {"volitional": "話そう"},
    ]
    assert [item["source_chapters"] for item in proposals] == [["3"], ["16"]]
    assert all(extract._is_parsed_schema_proposal(item) for item in proposals)
    assert result.candidate_accounting["candidate_accounting_fingerprint"] == (
        extract.candidate_accounting_fingerprint(result.candidate_accounting)
    )


def test_source_column_order_survives_a_staging_write_and_read(tmp_path: Path) -> None:
    """The order a reviewer sees is the order the source printed.

    Two things come back, and they are different on purpose. The display map
    is what ``from_dict`` makes of it — trimmed, without the blank cell — and
    keeps the source's column order. The raw witness is the page itself, blank
    cell and all, exactly as the model transcribed it.
    """
    result = build_records(
        [
            candidate(
                conjugations=dict(SOURCE_COLUMNS), source_chapters=SOURCE_CHAPTERS
            )
        ],
        prepared(tmp_path),
    )
    path = tmp_path / "staging" / "lesson.pdf.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_staging(path, result.records)

    records, _meta = read_staging(path)

    [staged] = records
    assert list(staged.conjugations.items()) == [
        (label, value) for label, value in SOURCE_COLUMNS if value
    ]
    assert list(
        json.loads(staged.source.raw_fields["source_conjugations"]).items()
    ) == list(SOURCE_COLUMNS)
    assert json.loads(staged.source.raw_fields["source_chapters"]) == SOURCE_CHAPTERS


# --- what the templates have to ask for ---------------------------------------


def _template(name: str) -> str:
    return " ".join(prompts.load(REPO_ROOT, name).split())


@pytest.mark.parametrize("name", ["extract-auto", "extract-table", "extract-prose"])
def test_every_template_asks_for_the_printed_columns_unchanged(name: str) -> None:
    text = _template(name)

    assert (
        "every printed column label as its key and that row's supplied cell as "
        "its value, in the source's printed order" in text
    )
    # However many columns a source prints, every one of them is kept. An
    # illustrative count is wording rather than the contract, so this asserts the
    # retention rule each template states instead of one template's example.
    lowered = text.lower()
    assert "keep every" in lowered
    assert "number" in lowered
    assert (
        "Do not substitute a familiar set of derived forms, drop a column whose "
        "label you do not recognise, reorder the columns, or fill a cell the "
        "source leaves blank" in text
    )
    assert "a blank cell stays an empty value under its label" in text
    assert (
        "explain the doubt in usage_notes and use low confidence rather than "
        "correcting it" in text
    )


@pytest.mark.parametrize("name", ["extract-auto", "extract-table", "extract-prose"])
def test_every_template_asks_for_the_printed_chapter_labels(name: str) -> None:
    text = _template(name)

    assert (
        "copy those labels into source_chapters exactly as printed and in "
        "printed order" in text
    )
    assert (
        "Never infer a chapter from the vocabulary itself or from a page number."
        in text
    )
    assert "Leave source_chapters empty when the source states none." in text


@pytest.mark.parametrize("name", ["extract-auto", "extract-table"])
def test_the_exhaustive_templates_consolidate_a_repeated_identity(name: str) -> None:
    text = _template(name)

    assert (
        "return one complete candidate for that identity carrying every sense, "
        "chapter, and supplied form the source teaches it" in text
    )
    assert (
        "The other rows remain their own source_units with the disposition "
        "duplicate, and each of those reasons names the retained candidate's "
        "page, section, and ordinal." in text
    )
    assert "Never join text from different rows into one context." in text
    assert (
        "keep each row's text verbatim in its own unit and say so in usage_notes "
        "rather than silently choosing one" in text
    )
