"""Reading vocabulary off PDFs and photos into staging files.

No network (IMPLEMENTATION_PLAN rule 6): ``parse_call`` is faked, so the tests
drive the command's own logic — the stop-reason discipline, the already-known
annotation, the staging shape — rather than the SDK's.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from japanese_anki import cli, extract, hardening
from japanese_anki.claude_client import CallResult, Refusal
from japanese_anki.extract import ExtractError, build_records, known_ids, system_prompt
from japanese_anki.inputs import PreparedInput
from japanese_anki.models import VocabularyRecord
from japanese_anki.staging import read_staging

PDF = b"%PDF-1.7 fake"


def candidate(**overrides: Any) -> Any:
    """One candidate in the shape the schema produces."""
    schema = extract.candidate_schema()
    fields: dict[str, Any] = {
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "part_of_speech": "verb",
        "example": "",
        "page": 12,
        "context": "話す　はなす　to speak",
        "confidence": "high",
        "inclusion_reason": "",
        "source_kind": "prose",
        "section": "",
        "ordinal": 0,
    }
    fields.update(overrides)
    return schema.model_fields["candidates"].annotation.__args__[0](**fields)


def extraction(*candidates: Any) -> Any:
    return extract.candidate_schema()(candidates=list(candidates))


def source_unit(**overrides: Any) -> Any:
    schema = extract.candidate_schema()
    fields: dict[str, Any] = {
        "page": 12,
        "section": "vocabulary",
        "ordinal": 1,
        "context": "話す　はなす　to speak",
        "disposition": "candidate",
        "reason": "",
    }
    fields.update(overrides)
    unit_type = schema.model_fields["source_units"].annotation.__args__[0]
    return unit_type(**fields)


def table_extraction(*candidates: Any, units: list[Any] | None = None, count: int = 0) -> Any:
    return extract.candidate_schema()(
        candidates=list(candidates),
        source_units=units or [],
        model_reported_unit_count=count,
    )


def prepared(tmp_path: Path, name: str = "lesson.pdf") -> PreparedInput:
    path = tmp_path / "inbox" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PDF)
    return PreparedInput(
        kind="document",
        media_type="application/pdf",
        data_b64="ZmFrZQ==",
        origin_path=path,
    )


class FakeCall:
    """Stands in for ``claude_client.parse_call`` and records what it was sent."""

    def __init__(self, *results: CallResult) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        model: str,
        system_blocks: Any,
        user_content: Any,
        schema: Any,
        client: Any = None,
        **kwargs: Any,
    ) -> CallResult:
        self.calls.append(
            {
                "model": model,
                "system": system_blocks,
                "content": user_content,
                "schema": schema,
            }
        )
        return self.results.pop(0)


def ok(*candidates: Any) -> CallResult:
    return CallResult(extraction(*candidates), "end_turn", None)


def table_ok(*candidates: Any, units: list[Any] | None = None, count: int = 0) -> CallResult:
    return CallResult(
        table_extraction(*candidates, units=units, count=count), "end_turn", None
    )


# --- the prompt --------------------------------------------------------------


def test_each_mode_gets_its_own_rules() -> None:
    table = system_prompt("table")
    prose = system_prompt("prose")
    auto = system_prompt(None)

    assert "Account for every row" in table
    assert "worth making a card for" in prose
    assert "Judge each page for itself" in auto
    # The rule that matters most is in all three: an invented reading becomes a
    # permanent, uncorrectable record ID.
    for text in (table, prose, auto):
        assert "Never invent a reading" in text


def test_context_normalization_is_one_stable_production_rule() -> None:
    assert extract.normalize_context(" 話す\r\n\tはなす  ") == "話す はなす"
    assert extract.context_fingerprint("話す  はなす") == extract.context_fingerprint(
        " 話す\nはなす "
    )


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ExtractError) as excinfo:
        system_prompt("poetry")

    assert "table" in str(excinfo.value)


def test_the_known_word_list_rides_in_the_user_turn_not_the_system_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # It changes every time the collection grows, and anything above the cache
    # breakpoint that changes invalidates the cached style guide.
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(extract.claude_client, "parse_call", call)

    extract.extract_candidates(
        prepared(tmp_path),
        model="claude-opus-5",
        style_guide="STYLE",
        mode="prose",
        known=["食べる"],
    )

    sent = call.calls[0]
    system_text = " ".join(block["text"] for block in sent["system"])
    assert "食べる" not in system_text
    assert "STYLE" in system_text
    user_text = " ".join(
        block["text"] for block in sent["content"] if block.get("type") == "text"
    )
    assert "食べる" in user_text


def test_the_file_is_sent_as_its_content_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = FakeCall(ok())
    monkeypatch.setattr(extract.claude_client, "parse_call", call)
    item = prepared(tmp_path)

    extract.extract_candidates(item, model="claude-opus-5", style_guide="S")

    assert call.calls[0]["content"][0] == item.content_block()


# --- stop-reason discipline --------------------------------------------------


def test_a_refusal_reports_the_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "It was declined" without a reason leaves the user nothing to act on.
    monkeypatch.setattr(
        extract.claude_client,
        "parse_call",
        FakeCall(CallResult(None, "refusal", Refusal("cyber", "policy decline"))),
    )

    with pytest.raises(ExtractError) as excinfo:
        extract.extract_candidates(
            prepared(tmp_path), model="claude-opus-5", style_guide="S"
        )

    message = str(excinfo.value)
    assert "cyber" in message and "policy decline" in message
    assert "Nothing was written" in message


def test_a_refusal_with_no_details_still_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        extract.claude_client, "parse_call", FakeCall(CallResult(None, "refusal", None))
    )

    with pytest.raises(ExtractError):
        extract.extract_candidates(
            prepared(tmp_path), model="claude-opus-5", style_guide="S"
        )


def test_a_truncated_answer_is_never_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The visible half would look like a complete extraction, and the words it
    # lost are the ones nobody would notice were missing.
    monkeypatch.setattr(
        extract.claude_client,
        "parse_call",
        FakeCall(CallResult(None, "max_tokens", None)),
    )

    with pytest.raises(ExtractError) as excinfo:
        extract.extract_candidates(
            prepared(tmp_path), model="claude-opus-5", style_guide="S"
        )

    message = str(excinfo.value)
    assert "cut off" in message
    # Actionable, not just a complaint.
    assert "split a long document" in message


def test_any_other_incomplete_stop_is_also_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        extract.claude_client,
        "parse_call",
        FakeCall(CallResult(None, "pause_turn", None)),
    )

    with pytest.raises(ExtractError) as excinfo:
        extract.extract_candidates(
            prepared(tmp_path), model="claude-opus-5", style_guide="S"
        )

    assert "pause_turn" in str(excinfo.value)


# --- source-unit accounting -------------------------------------------------


def table_result(
    *units: extract.SourceUnit,
    candidates: tuple[Any, ...] = (),
    reported: int = 0,
) -> extract.ExtractionResult:
    return extract.ExtractionResult(candidates, units, reported)


def normalized_unit(
    ordinal: int,
    *,
    context: str | None = None,
    disposition: str = "candidate",
    reason: str = "",
) -> extract.SourceUnit:
    text = context or f"row {ordinal}"
    return extract.SourceUnit(
        page=1,
        section="vocabulary",
        ordinal=ordinal,
        context=text,
        context_fingerprint=extract.context_fingerprint(text),
        disposition=disposition,
        reason=reason,
    )


def exhaustive_oracle(*units: extract.SourceUnit) -> Any:
    return SimpleNamespace(
        id="lesson-table",
        type="exhaustive",
        units=tuple(
            hardening.OracleUnit(
                page=unit.page,
                section=unit.section,
                ordinal=unit.ordinal,
                context_fingerprint=unit.context_fingerprint,
                disposition=unit.disposition,
            )
            for unit in units
        ),
    )


def test_exact_coverage_ignores_the_models_wrong_self_count() -> None:
    one = normalized_unit(1)
    two = normalized_unit(2, disposition="unreadable", reason="ink is hidden")

    block = extract.coverage_block(
        table_result(one, two, reported=99),
        source_sha256="a" * 64,
        mode="table",
        oracle=exhaustive_oracle(one, two),
        oracle_fingerprint="b" * 64,
    )

    assert block["status"] == "matched"
    assert block["blocking"] is False
    assert block["observed_unit_count"] == 2
    assert block["model_reported_unit_count"] == 99
    assert block["unreadable_units"] == [two.fact()]


def test_an_omitted_row_cannot_be_replaced_by_a_duplicate_or_invented_row() -> None:
    one = normalized_unit(1)
    two = normalized_unit(2)
    repeated_one = normalized_unit(1, disposition="duplicate", reason="repeated")
    invented = normalized_unit(3)

    block = extract.coverage_block(
        table_result(one, repeated_one, invented),
        source_sha256="a" * 64,
        mode="table",
        oracle=exhaustive_oracle(one, two),
        oracle_fingerprint="b" * 64,
    )

    assert block["status"] == "mismatch"
    assert block["missing_units"][0]["ordinal"] == 2
    assert block["unexpected_units"][0]["ordinal"] == 3
    assert block["duplicate_keys"] == [
        {"page": 1, "section": "vocabulary", "ordinal": 1}
    ]
    assert block["omission_units"][0]["ordinal"] == 2


def test_context_and_disposition_mismatches_are_exact_facts() -> None:
    expected = normalized_unit(1, context="話す はなす")
    observed = normalized_unit(
        1,
        context="話す はなし",
        disposition="unreadable",
        reason="last kana is hidden",
    )

    block = extract.coverage_block(
        table_result(observed),
        source_sha256="a" * 64,
        mode="table",
        oracle=exhaustive_oracle(expected),
        oracle_fingerprint="b" * 64,
    )

    assert block["context_mismatched_units"] == [
        {
            "page": 1,
            "section": "vocabulary",
            "ordinal": 1,
            "expected": expected.context_fingerprint,
            "observed": observed.context_fingerprint,
        }
    ]
    assert block["disposition_mismatched_units"][0]["observed"] == "unreadable"
    assert block["unreadable_units"] == [observed.fact()]


def test_table_candidates_must_link_to_source_units_one_to_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = FakeCall(
        table_ok(
            candidate(
                source_kind="table",
                section="vocabulary",
                ordinal=2,
            ),
            units=[source_unit(ordinal=1)],
        )
    )
    monkeypatch.setattr(extract.claude_client, "parse_call", call)

    with pytest.raises(ExtractError) as excinfo:
        extract.extract_candidates(
            prepared(tmp_path), model="claude-opus-5", style_guide="S", mode="table"
        )

    assert excinfo.value.code == "extract-candidate-unit-link"


def test_auto_mode_reports_table_and_prose_coverage_separately() -> None:
    unit = normalized_unit(1)
    result = table_result(
        unit,
        candidates=(candidate(inclusion_reason="new grammar", source_kind="prose"),),
    )

    block = extract.coverage_block(
        result, source_sha256="a" * 64, mode=None
    )

    assert block["status"] == "unmeasured"
    assert block["blocking"] is True
    assert block["prose_candidate_count"] == 1
    assert block["prose_coverage"] == "unmeasured"


def test_prose_coverage_is_unmeasured_even_when_no_candidate_is_selected() -> None:
    block = extract.coverage_block(
        table_result(), source_sha256="a" * 64, mode="prose"
    )

    assert block["status"] == "selection"
    assert block["prose_candidate_count"] == 0
    assert block["prose_coverage"] == "unmeasured"
    assert block["blocking"] is False


# --- candidates to records ---------------------------------------------------


def test_provenance_is_stringified_into_raw_fields(tmp_path: Path) -> None:
    item = prepared(tmp_path)

    [record] = build_records([candidate()], item)

    fields = record.source.raw_fields
    assert record.source.type == "extract"
    assert fields["page"] == "12"
    assert fields["confidence"] == "high"
    assert fields["context"] == "話す　はなす　to speak"
    assert fields["extracted_from"] == "lesson.pdf"
    assert all(isinstance(value, str) for value in fields.values())


def test_a_candidate_with_no_reading_keeps_its_malformed_id(tmp_path: Path) -> None:
    # The expected state of an extraction: validate reports it, and the review
    # supplies the reading. Inventing one would mint a permanent wrong ID.
    [record] = build_records([candidate(reading="")], prepared(tmp_path))

    assert record.id == "word:話す:"
    assert record.reading == ""


def test_an_example_sentence_becomes_an_example(tmp_path: Path) -> None:
    [record] = build_records(
        [candidate(example="日本語を話します。")], prepared(tmp_path)
    )

    assert [ex.japanese for ex in record.examples] == ["日本語を話します。"]


def test_a_candidate_with_no_expression_cannot_become_a_record(
    tmp_path: Path,
) -> None:
    assert build_records([candidate(expression="  ")], prepared(tmp_path)) == []


def test_already_known_candidates_are_marked_and_sorted_last(
    tmp_path: Path,
) -> None:
    # Kept, not dropped: a silent discard is a silent discard even for a
    # duplicate, and the reviewer may still want this page's example.
    item = prepared(tmp_path)
    records = build_records(
        [candidate(), candidate(expression="食べる", reading="たべる")],
        item,
        known_ids([VocabularyRecord(id="word:話す:はなす", expression="話す", reading="はなす")]),
    )

    assert [record.expression for record in records] == ["食べる", "話す"]
    assert "already_known" not in records[0].source.raw_fields
    assert records[0].romaji == "taberu"
    assert "record-romaji-from-reading" in records[0].source.raw_fields["janki_repairs"]
    assert records[1].source.raw_fields["already_known"] == "true"
    assert records[1].romaji == ""
    assert "janki_repairs" not in records[1].source.raw_fields


def test_repeated_candidate_rows_do_not_create_duplicate_canonical_records(
    tmp_path: Path,
) -> None:
    records = build_records([candidate(), candidate()], prepared(tmp_path))

    assert [record.id for record in records] == ["word:話す:はなす"]


def test_a_record_whose_stored_id_drifted_still_counts_as_known() -> None:
    # A hand-written record may carry an id that no longer matches what its
    # expression and reading would mint today; a candidate matching either is
    # a word janki already has.
    ids = known_ids(
        [VocabularyRecord(id="legacy-id", expression="話す", reading="はなす")]
    )

    assert "legacy-id" in ids
    assert "word:話す:はなす" in ids


# --- the CLI -----------------------------------------------------------------


def project(tmp_path: Path, records: list[VocabularyRecord] | None = None) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n'
        'scan_inbox = "inbox"\n',
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "JAPANESE_STYLE_GUIDE.md").write_text(
        "Prefer natural English.", encoding="utf-8"
    )
    if records is not None:
        (tmp_path / "vocabulary.json").write_text(
            json.dumps([item.to_dict() for item in records], ensure_ascii=False),
            encoding="utf-8",
        )
    return tmp_path


def source_pdf(tmp_path: Path, name: str = "lesson.pdf") -> Path:
    path = tmp_path / "desk" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PDF)
    return path


def write_approved_oracle(
    root: Path,
    *,
    source_sha256: str,
    units: tuple[extract.SourceUnit, ...],
    oracle_id: str = "lesson-table",
    approved: bool = True,
) -> Path:
    path = root / "quality" / "oracles" / f"{oracle_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    oracle_units = tuple(
        hardening.OracleUnit(
            unit.page,
            unit.section,
            unit.ordinal,
            unit.context_fingerprint,
            unit.disposition,
        )
        for unit in units
    )
    oracle = hardening.UnitOracle(
        path=path,
        relative_path=f"quality/oracles/{path.name}",
        id=oracle_id,
        source_fingerprint=source_sha256,
        type="exhaustive",
        case_ids=(),
        units=oracle_units,
        targets=(),
        selection_rubric=None,
        approval=None,
    )
    content_fingerprint = hardening.oracle_content_fingerprint(oracle)
    payload: dict[str, Any] = {
        "version": 1,
        "id": oracle_id,
        "source_fingerprint": source_sha256,
        "type": "exhaustive",
        "case_ids": [],
        "units": [
            {
                "page": unit.page,
                "section": unit.section,
                "ordinal": unit.ordinal,
                "context_fingerprint": unit.context_fingerprint,
                "disposition": unit.disposition,
            }
            for unit in oracle_units
        ],
    }
    if approved:
        payload["approval"] = {
            "authority": "repository-owner",
            "oracle_id": oracle_id,
            "source_fingerprint": source_sha256,
            "oracle_type": "exhaustive",
            "oracle_content_fingerprint": content_fingerprint,
            "approved_at": "2026-08-12",
        }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def test_extract_writes_one_staging_file_per_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    one, two = source_pdf(tmp_path), source_pdf(tmp_path, "lesson2.pdf")
    monkeypatch.setattr(
        cli.extract.claude_client, "parse_call", FakeCall(ok(candidate()), ok(candidate()))
    )

    code = cli.main(["--root", str(root), "extract", str(one), str(two)])

    assert code == 0
    assert (root / "staging" / "lesson.pdf.yaml").is_file()
    assert (root / "staging" / "lesson2.pdf.yaml").is_file()
    assert "Wrote 2 staging file(s)" in capsys.readouterr().out


def test_the_staging_file_is_the_pinned_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path))])

    written = yaml.safe_load((root / "staging" / "lesson.pdf.yaml").read_text(encoding="utf-8"))
    assert written["model"] == "claude-opus-5"
    assert "extracted_at" in written and "source_file" in written
    # Reads back through the staging loader, which is what `janki validate`
    # and the promote step will use.
    records, meta = read_staging(root / "staging" / "lesson.pdf.yaml")
    assert [record.expression for record in records] == ["話す"]
    # The basename, not an absolute path: this file is committed, and an
    # absolute path is stale on any other clone.
    assert meta["source_file"] == "lesson.pdf"


def test_an_approved_oracle_and_prompt_provenance_are_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    source = source_pdf(tmp_path)
    unit = normalized_unit(1, context="話す　はなす　to speak")
    # The model page is 12 in this fixture.
    unit = extract.SourceUnit(
        12,
        unit.section,
        unit.ordinal,
        unit.context,
        unit.context_fingerprint,
        unit.disposition,
        unit.reason,
    )
    oracle = write_approved_oracle(
        root,
        source_sha256=extract.source_fingerprint(source),
        units=(unit,),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-stored")
    monkeypatch.setattr(
        cli.extract.claude_client,
        "parse_call",
        FakeCall(
            table_ok(
                candidate(source_kind="table", section="vocabulary", ordinal=1),
                units=[source_unit()],
                count=17,
            )
        ),
    )

    code = cli.main(
        [
            "--root",
            str(root),
            "extract",
            str(source),
            "--mode",
            "table",
            "--coverage-oracle",
            str(oracle),
        ]
    )

    assert code == 0
    _records, meta = read_staging(root / "staging" / "lesson.pdf.yaml")
    assert meta["coverage"]["status"] == "matched"
    assert meta["coverage"]["oracle_id"] == "lesson-table"
    assert meta["coverage"]["model_reported_unit_count"] == 17
    provenance = meta["prompt_provenance"]
    assert provenance["provider"] == "anthropic"
    assert provenance["response_schema_version"] == extract.EXTRACTION_SCHEMA_VERSION
    serialized = json.dumps(meta, ensure_ascii=False)
    assert "must-not-be-stored" not in serialized
    assert str(root) not in serialized
    # Promotion reloads the approved oracle and recomputes these facts. A
    # hand-edited "matched" label cannot bypass that check.
    (root / "quality" / "oracles" / "unrelated.yaml").write_text(
        "not: a valid oracle\n", encoding="utf-8"
    )
    assert (
        cli.main(
            [
                "--root",
                str(root),
                "promote",
                str(root / "staging" / "lesson.pdf.yaml"),
                "--skip-reading-check",
            ]
        )
        == 0
    )
    _archived, archived_meta = read_staging(
        root / "staging" / "done" / "lesson.pdf.yaml"
    )
    assert archived_meta["coverage"]["oracle_id"] == "lesson-table"


def test_oracle_binding_errors_happen_before_the_first_paid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    one = source_pdf(tmp_path)
    two = source_pdf(tmp_path, "lesson2.pdf")
    # Both fixtures have the same bytes. One oracle would bind to two sources.
    oracle = write_approved_oracle(
        root,
        source_sha256=extract.source_fingerprint(one),
        units=(normalized_unit(1),),
    )
    call = FakeCall(ok(candidate()), ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(
        [
            "--root",
            str(root),
            "extract",
            str(one),
            str(two),
            "--coverage-oracle",
            str(oracle),
        ]
    )

    assert code == 1
    assert call.calls == []


def test_a_draft_oracle_is_refused_before_the_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    source = source_pdf(tmp_path)
    oracle = write_approved_oracle(
        root,
        source_sha256=extract.source_fingerprint(source),
        units=(normalized_unit(1),),
        approved=False,
    )
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(
        [
            "--root",
            str(root),
            "extract",
            str(source),
            "--coverage-oracle",
            str(oracle),
        ]
    )

    assert code == 1
    assert call.calls == []


def test_a_source_oracle_hash_mismatch_is_refused_before_the_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    source = source_pdf(tmp_path)
    oracle = write_approved_oracle(
        root,
        source_sha256="f" * 64,
        units=(normalized_unit(1),),
    )
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(
        [
            "--root",
            str(root),
            "extract",
            str(source),
            "--coverage-oracle",
            str(oracle),
        ]
    )

    assert code == 1
    assert call.calls == []


def test_prompt_fingerprints_change_with_the_prompt_not_the_path(tmp_path: Path) -> None:
    item = prepared(tmp_path)
    base = extract.prompt_provenance(
        item, model="m", style_guide="style", mode="prose", known=()
    )
    changed = extract.prompt_provenance(
        item, model="m", style_guide="style", mode="prose", known=("話す",)
    )

    assert base["system_prompt_fingerprint"] == changed["system_prompt_fingerprint"]
    assert base["user_prompt_fingerprint"] != changed["user_prompt_fingerprint"]
    assert str(tmp_path) not in json.dumps(base)


def test_the_input_is_copied_into_the_inbox_and_cited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A candidate has to stay checkable against the page it came from.
    root = project(tmp_path)
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path))])

    assert (root / "inbox" / "lesson.pdf").read_bytes() == PDF
    records, _ = read_staging(root / "staging" / "lesson.pdf.yaml")
    assert records[0].source.raw_fields["extracted_from"] == "lesson.pdf"


def test_an_existing_staging_file_is_not_overwritten_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Staging files hold review edits that exist nowhere else.
    root = project(tmp_path)
    (root / "staging").mkdir()
    (root / "staging" / "lesson.pdf.yaml").write_text(
        "records: []\nreview_notes: mid-review\n", encoding="utf-8"
    )
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    code = cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path))])

    assert code == 1
    assert "mid-review" in (root / "staging" / "lesson.pdf.yaml").read_text(encoding="utf-8")
    assert "already exists" in capsys.readouterr().err


def test_force_overwrites_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = project(tmp_path)
    (root / "staging").mkdir()
    (root / "staging" / "lesson.pdf.yaml").write_text("records: []\n", encoding="utf-8")
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    code = cli.main(
        ["--root", str(root), "extract", str(source_pdf(tmp_path)), "--force"]
    )

    assert code == 0
    records, _ = read_staging(root / "staging" / "lesson.pdf.yaml")
    assert [record.expression for record in records] == ["話す"]


def test_the_model_can_be_overridden_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    cli.main(
        [
            "--root",
            str(root),
            "extract",
            str(source_pdf(tmp_path)),
            "--model",
            "claude-haiku-4-5",
        ]
    )

    assert call.calls[0]["model"] == "claude-haiku-4-5"


def test_known_words_are_only_listed_for_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A table is transcribed row by row; telling the model to skip rows would
    # put holes in a faithful transcription.
    root = project(
        tmp_path, [VocabularyRecord(id="word:話す:はなす", expression="話す", reading="はなす")]
    )
    call = FakeCall(
        table_ok(
            candidate(source_kind="table", section="vocabulary", ordinal=1),
            units=[source_unit()],
            count=1,
        ),
        ok(candidate()),
    )
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path)), "--mode", "table"])
    cli.main(
        [
            "--root",
            str(root),
            "extract",
            str(source_pdf(tmp_path)),
            "--mode",
            "prose",
            "--force",
        ]
    )

    table_text = " ".join(
        block["text"] for block in call.calls[0]["content"] if block.get("type") == "text"
    )
    prose_text = " ".join(
        block["text"] for block in call.calls[1]["content"] if block.get("type") == "text"
    )
    assert "話す" not in table_text
    assert "話す" in prose_text


def test_an_already_known_candidate_is_annotated_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(
        tmp_path, [VocabularyRecord(id="word:話す:はなす", expression="話す", reading="はなす")]
    )
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path))])

    records, _ = read_staging(root / "staging" / "lesson.pdf.yaml")
    assert records[0].source.raw_fields["already_known"] == "true"
    assert "1 already known" in capsys.readouterr().out


def test_a_refusal_writes_no_staging_file_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    monkeypatch.setattr(
        cli.extract.claude_client,
        "parse_call",
        FakeCall(CallResult(None, "refusal", Refusal("bio", "declined"))),
    )

    code = cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path))])

    assert code == 1
    assert not (root / "staging" / "lesson.pdf.yaml").exists()
    assert "bio" in capsys.readouterr().err


def test_a_later_failure_keeps_the_earlier_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The work already paid for is kept, and the error names what is left.
    root = project(tmp_path)
    one, two = source_pdf(tmp_path), source_pdf(tmp_path, "lesson2.pdf")
    monkeypatch.setattr(
        cli.extract.claude_client,
        "parse_call",
        FakeCall(ok(candidate()), CallResult(None, "max_tokens", None)),
    )

    code = cli.main(["--root", str(root), "extract", str(one), str(two)])

    assert code == 1
    assert (root / "staging" / "lesson.pdf.yaml").is_file()
    assert not (root / "staging" / "lesson2.pdf.yaml").exists()


def test_nothing_reaches_the_normalized_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The guarantee the whole command rests on: extraction proposes, a human
    # accepts, and only `promote` writes.
    root = project(tmp_path, [])
    before = (root / "vocabulary.json").read_text(encoding="utf-8")
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path))])

    assert (root / "vocabulary.json").read_text(encoding="utf-8") == before


# --- one file per input, or none ---------------------------------------------


def test_the_same_file_twice_is_refused_rather_than_written_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # prepare_inputs keeps duplicates rather than discarding them silently;
    # naming the problem is how that stays true.
    root = project(tmp_path)
    source = source_pdf(tmp_path)
    call = FakeCall(ok(candidate()), ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(["--root", str(root), "extract", str(source), str(source)])

    assert code == 1
    assert call.calls == []


def test_a_scan_and_a_photo_of_the_same_page_keep_separate_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # worksheet.pdf and worksheet.png are a natural pairing and two different
    # sources. Keyed on the stem, the second silently overwrote the first under
    # --force and failed with a wrong diagnosis without it.
    root = project(tmp_path)
    scan = source_pdf(tmp_path, "worksheet.pdf")
    photo = tmp_path / "desk" / "worksheet.png"
    photo.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    monkeypatch.setattr(
        cli.extract.claude_client, "parse_call", FakeCall(ok(candidate()), ok(candidate()))
    )

    assert cli.main(["--root", str(root), "extract", str(scan), str(photo)]) == 0

    assert (root / "staging" / "worksheet.pdf.yaml").is_file()
    assert (root / "staging" / "worksheet.png.yaml").is_file()


def test_a_candidate_with_no_expression_is_counted_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # It cannot become a record — there is nothing to mint an id from — but a
    # reviewer checking the file against the page must be able to tell a lost
    # row from a page that had one fewer word.
    root = project(tmp_path)
    monkeypatch.setattr(
        cli.extract.claude_client,
        "parse_call",
        FakeCall(
            ok(
                candidate(),
                candidate(
                    expression="  ",
                    example="毎日日本語を話します。",
                    confidence="low",
                    inclusion_reason="new in this chapter",
                ),
            )
        ),
    )

    cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path))])

    assert "1 unusable" in capsys.readouterr().out
    # And durably, in the file a reviewer actually reads — a count that lives
    # only in scrollback is the same silent discard with an extra step.
    _records, meta = read_staging(root / "staging" / "lesson.pdf.yaml")
    note = meta["review_notes"]
    assert "could not be stored" in note
    # Everything the model did read about the row, so the page can be rechecked.
    assert "page: 12" in note
    assert "はなす" in note
    assert "to speak" in note
    # Every field the model filled in, not a hand-picked few: the example read
    # verbatim off the page and the low-confidence flag both matter to whoever
    # re-adds the word by hand.
    assert "毎日日本語を話します。" in note
    assert "verb" in note
    assert "low" in note
    assert "new in this chapter" in note


def test_a_held_back_row_whose_values_are_zero_is_still_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A row for 〇/ゼロ: the gloss and the source cell are both "0". Filtering
    # the *rendered text* to suppress the page sentinel swallowed them, which
    # is the drop this note exists to prevent.
    root = project(tmp_path)
    monkeypatch.setattr(
        cli.extract.claude_client,
        "parse_call",
        FakeCall(
            ok(candidate(expression="", reading="ゼロ", meanings=["0"], context="0", page=0))
        ),
    )

    cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path))])

    _records, meta = read_staging(root / "staging" / "lesson.pdf.yaml")
    note = meta["review_notes"]
    assert "meanings: 0" in note
    assert "context: 0" in note
    # The unknown-page sentinel is still suppressed rather than reported as 0.
    assert "page:" not in note
