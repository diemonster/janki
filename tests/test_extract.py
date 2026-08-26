"""Reading vocabulary off PDFs and photos into staging files.

No network (IMPLEMENTATION_PLAN rule 6): ``parse_call`` is faked, so the tests
drive the command's own logic — the stop-reason discipline, the already-known
annotation, the staging shape — rather than the SDK's.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_workbench_fixtures import _load, _prepared

from conftest import seed_prompts
from japanese_anki import cli, extract, operations, patterns, prompts
from japanese_anki.application import plan_extraction
from japanese_anki.application.extraction import (
    ANSWER_SAVED,
    OUTCOME_UNKNOWN,
    ExtractionTarget,
    authorize_dispatch,
    capture_hook,
    classify_dispatch_failure,
    complete_extraction,
    settle_dispatch,
)
from japanese_anki.claude_client import CallResult, Refusal
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.extract import ExtractError, build_records, known_ids, prompt_name
from japanese_anki.inputs import PreparedInput, prepare_inputs
from japanese_anki.models import VocabularyRecord, provisional_fields
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
        "examples": [
            {
                "japanese": "日本語を話します。",
                "speech_level": "polite",
                "furigana": "日本語[にほんご]を 話[はな]します。",
                "romaji": "nihongo o hanashimasu.",
                "english": "I speak Japanese.",
            },
            {
                "japanese": "あとで話そう。",
                "speech_level": "casual",
                "furigana": "あとで 話[はな]そう。",
                "romaji": "ato de hanasou.",
                "english": "Let's talk later.",
            },
        ],
        "usage_notes": "",
        "page": 12,
        "context": "話す　はなす　to speak",
        "confidence": "high",
        "inclusion_reason": "The lesson explicitly teaches this word.",
        "source_kind": "prose",
        "section": "",
        "ordinal": 0,
    }
    fields.update(overrides)
    return schema.model_fields["candidates"].annotation.__args__[0](**fields)


def test_persisted_parsed_proposal_shape_tracks_the_live_ai_schema() -> None:
    """The stdlib promote validator must move whenever the paid schema moves."""
    candidate_type = extract.candidate_schema().model_fields[
        "candidates"
    ].annotation.__args__[0]
    example_type = candidate_type.model_fields["examples"].annotation.__args__[0]

    assert set(candidate_type.model_fields) == extract._PARSED_CANDIDATE_FIELDS
    assert set(example_type.model_fields) == extract._PARSED_EXAMPLE_FIELDS
    with pytest.raises(ValueError):
        candidate(page=-1)


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


def test_each_mode_gets_its_own_shipped_prompt_file() -> None:
    """Read from `prompts/`, not from a constant.

    These are the files a person edits, so the assertions belong on the files
    themselves — a clause deleted from `extract-table.md` should fail here,
    which is the whole reason the prompts are a reviewable artifact rather
    than string literals in Python.
    """
    root = Path(__file__).resolve().parents[1]
    table = prompts.load(root, prompt_name("table"))
    prose = prompts.load(root, prompt_name("prose"))
    auto = prompts.load(root, prompt_name(None))
    auto_one_line = " ".join(auto.split())

    assert "Account for every row" in table
    assert "worth making a card for" in prose
    assert "Judge each page for its source shape when routing its contents." in (
        auto_one_line
    )
    assert (
        "Page-by-page routing does not require lexical evidence to appear "
        "on only one page"
    ) in auto_one_line
    # Each file stands alone — no shared block is concatenated at send time —
    # so the rule that matters most has to be present in all three. An invented
    # reading becomes a permanent, uncorrectable record ID.
    for text in (table, prose, auto):
        assert "Never invent a reading" in text
        assert text.startswith("You are reading Japanese study material")


def test_context_normalization_is_one_stable_production_rule() -> None:
    assert extract.normalize_context(" 話す\r\n\tはなす  ") == "話す はなす"
    assert extract.context_fingerprint("話す  はなす") == extract.context_fingerprint(
        " 話す\nはなす "
    )


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ExtractError) as excinfo:
        prompt_name("poetry")

    assert "table" in str(excinfo.value)


def test_the_response_schema_rejects_superseded_or_unknown_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="example"):
        extract.candidate_schema().model_validate(
            {"candidates": [{"expression": "話す", "example": "話します。"}]}
        )
    with pytest.raises(ValidationError, match="aside"):
        extract.candidate_schema().model_validate(
            {
                "candidates": [
                    {
                        "expression": "話す",
                        "examples": [
                            {"japanese": "話します。", "aside": "not in contract"}
                        ],
                    }
                ]
            }
        )


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
        system="S",
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

    extract.extract_candidates(item, model="claude-opus-5", style_guide="S", system="S")

    assert call.calls[0]["content"][0] == item.content_block()


def test_source_patterns_share_the_extraction_calls_prompt_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parsed = extract.candidate_schema()(
        document_kind="lesson",
        document_title="Week 11",
        patterns=[
            {
                "template": "〜んだ",
                "gloss": "explanation",
                "examples": ["大変だったんだ。"],
                "where": "page 2",
            }
        ],
    )
    monkeypatch.setattr(
        extract.claude_client,
        "parse_call",
        FakeCall(CallResult(parsed, "end_turn", None)),
    )

    result = extract.extract_candidates(
        prepared(tmp_path),
        model="claude-opus-5",
        style_guide="STYLE",
        system="SOURCE",
    )

    assert result.pattern_set.kind == "lesson"
    assert [item.template for item in result.pattern_set.patterns] == ["〜んだ"]
    assert result.pattern_set.prompt_provenance["model"] == "claude-opus-5"
    assert result.pattern_set.prompt_provenance["response_schema_version"] == 5


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
            prepared(tmp_path), model="claude-opus-5", style_guide="S",
        system="S")

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
            prepared(tmp_path), model="claude-opus-5", style_guide="S",
        system="S")


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
            prepared(tmp_path), model="claude-opus-5", style_guide="S",
        system="S")

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
            prepared(tmp_path), model="claude-opus-5", style_guide="S",
        system="S")

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


def test_coverage_records_both_counts_rather_than_trusting_the_models_own() -> None:
    """The model's self-count is stored as a claim, beside what was observed.

    M8.4 deleted the approved oracle that used to adjudicate between them, so
    nothing scores the difference now — but recording both is what lets a
    person see one, and collapsing them into a single number would destroy the
    evidence rather than the ceremony.
    """
    one = normalized_unit(1)
    two = normalized_unit(2, disposition="unreadable", reason="ink is hidden")

    block = extract.coverage_block(
        table_result(one, two, reported=99),
        source_sha256="a" * 64,
        mode="table",
    )

    assert block["observed_unit_count"] == 2
    assert block["model_reported_unit_count"] == 99
    assert block["unreadable_units"] == [two.fact()]
    assert block["status"] == "unmeasured", "nobody asserted what the page held"


def test_every_observed_unit_appears_in_exactly_one_disposition_list() -> None:
    """The disposition lists must partition `source_units`, not sample it.

    Building them from a key-deduplicated mapping instead of the observed list
    makes a repeated source row vanish from the record while
    `observed_unit_count` still counts it — the block then contradicts itself,
    and `promote._verify_coverage_facts` cannot notice because it re-derives
    with the same function, so the inconsistency is self-consistent. Measured
    after M8.4: green.
    """
    one = normalized_unit(1)
    repeated_one = normalized_unit(1, disposition="duplicate", reason="repeated")
    other = normalized_unit(3)

    block = extract.coverage_block(
        table_result(one, repeated_one, other),
        source_sha256="a" * 64,
        mode="table",
    )

    listed = sum(
        len(block[f"{name.replace('-', '_')}_units"])
        for name in extract.SOURCE_UNIT_DISPOSITIONS
    )
    assert listed == block["observed_unit_count"] == 3
    assert len(block["duplicate_units"]) == 1, "the repeat is filed, not dropped"


def test_a_repeated_source_unit_key_is_named_not_silently_collapsed() -> None:
    """The one internal inconsistency still detectable without an oracle.

    Two units under one key means the model returned the same row twice, which
    a plain `dict` build would swallow by overwriting. Nothing outside the
    response is needed to know it is wrong, which is why this half outlived the
    oracle comparisons deleted with M8.4.
    """
    one = normalized_unit(1)
    repeated_one = normalized_unit(1, disposition="duplicate", reason="repeated")
    other = normalized_unit(3)

    block = extract.coverage_block(
        table_result(one, repeated_one, other),
        source_sha256="a" * 64,
        mode="table",
    )

    assert block["duplicate_keys"] == [
        {"page": 1, "section": "vocabulary", "ordinal": 1}
    ]
    assert block["observed_unit_count"] == 3, "the repeat is counted, not dropped"


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
            prepared(tmp_path), model="claude-opus-5", style_guide="S", mode="table",
        system="S")

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


def test_coverage_v2_binds_the_complete_candidate_accounting_block(
    tmp_path: Path,
) -> None:
    candidates = (
        candidate(),
        candidate(page=13),
        candidate(expression="", reading="", page=14),
    )
    result = table_result(candidates=candidates)
    built = build_records(candidates, prepared(tmp_path))

    block = extract.coverage_block(
        result,
        source_sha256="a" * 64,
        mode="prose",
        candidate_accounting=built.candidate_accounting,
    )

    assert block["version"] == 2
    for name in (
        "parsed_candidate_count",
        "canonical_record_count",
        "unusable_candidate_count",
        "duplicate_candidate_count",
        "collision_group_count",
    ):
        assert block[name] == built.candidate_accounting[name]
    assert block["candidate_accounting_fingerprint"] == built.candidate_accounting[
        "candidate_accounting_fingerprint"
    ]
    assert block["prose_candidate_count"] == 3, (
        "source-kind accounting remains separate from record/collision counts"
    )


# --- candidates to records ---------------------------------------------------


def test_provenance_is_stringified_into_raw_fields(tmp_path: Path) -> None:
    item = prepared(tmp_path)

    [record] = build_records([candidate()], item).records

    fields = record.source.raw_fields
    assert record.source.type == "extract"
    assert fields["page"] == "12"
    assert fields["confidence"] == "high"
    assert fields["context"] == "話す　はなす　to speak"
    assert fields["extracted_from"] == "lesson.pdf"
    assert all(isinstance(value, str) for value in fields.values())


def test_the_models_reason_for_proposing_a_word_is_kept(tmp_path: Path) -> None:
    """`inclusion_reason` is why a prose candidate was proposed at all, and the
    prompt asks for it by name.

    It is persisted in exactly one place, and dropping it from that tuple left
    the suite green after M8.4 — the reviewer would lose the model's own
    argument for every prose candidate while every other provenance field kept
    working, which is the kind of gap nobody notices until they need it.
    """
    item = prepared(tmp_path)
    reasoned = candidate(inclusion_reason="Introduced in the dialogue on page 12.")

    [record] = build_records([reasoned], item).records

    assert (
        record.source.raw_fields["inclusion_reason"]
        == "Introduced in the dialogue on page 12."
    )


def test_a_candidate_with_no_reading_keeps_its_malformed_id(tmp_path: Path) -> None:
    # The expected state of an extraction: validate reports it, and the review
    # supplies the reading. Inventing one would mint a permanent wrong ID.
    [record] = build_records([candidate(reading="")], prepared(tmp_path)).records

    assert record.id == "word:話す:"
    assert record.reading == ""


def test_rich_candidate_content_becomes_a_reviewable_record(tmp_path: Path) -> None:
    [record] = build_records(
        [
            candidate(
                examples=[
                    {
                        "japanese": "日本語を話します。",
                        "speech_level": "polite",
                        "furigana": "日本語[にほんご]を 話[はな]します。",
                        "romaji": "nihongo o hanashimasu.",
                        "english": "I speak Japanese.",
                    },
                    {
                        "japanese": "あとで話そう。",
                        "speech_level": "casual",
                        "furigana": "あとで 話[はな]そう。",
                        "romaji": "ato de hanasou.",
                        "english": "Let's talk later.",
                    },
                ],
                usage_notes="Often takes と for the person spoken with.",
            )
        ],
        prepared(tmp_path),
    ).records

    assert [item.japanese for item in record.examples] == [
        "日本語を話します。",
        "あとで話そう。",
    ]
    assert record.examples[0].register == "polite"
    assert record.usage_notes == "Often takes と for the person spoken with."
    assert "example_authority" not in record.source.raw_fields


def test_a_candidate_without_examples_never_reaches_record_building() -> None:
    with pytest.raises(ValueError, match="at least 2 items"):
        candidate(examples=[])


def test_extracted_semantic_fields_are_marked_provisional(tmp_path: Path) -> None:
    # Marked at the only moment the values are known to be model output and
    # nothing else. The mark is value-bound, so `provisional_fields` reading
    # it back is also the proof the binding matches what the record holds.
    [record] = build_records([candidate()], prepared(tmp_path)).records

    assert provisional_fields(record) == ["meanings", "part_of_speech"]


def test_blank_optional_metadata_is_not_marked_as_a_semantic_claim(
    tmp_path: Path,
) -> None:
    [record] = build_records(
        [candidate(part_of_speech="")], prepared(tmp_path)
    ).records

    assert provisional_fields(record) == ["meanings"]


def test_a_candidate_with_no_expression_cannot_become_a_record(
    tmp_path: Path,
) -> None:
    assert build_records(
        [candidate(expression="  ")], prepared(tmp_path)
    ).records == ()


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
    ).records

    assert [record.expression for record in records] == ["食べる", "話す"]
    assert "already_known" not in records[0].source.raw_fields
    assert records[0].romaji == "taberu"
    assert "record-romaji-from-reading" in records[0].source.raw_fields["janki_repairs"]
    assert records[1].source.raw_fields["already_known"] == "true"
    assert records[1].romaji == "hanasu"
    assert "record-romaji-from-reading" in records[1].source.raw_fields[
        "janki_repairs"
    ]


def test_collision_accounting_preserves_every_parsed_schema_proposal_in_order(
    tmp_path: Path,
) -> None:
    first = candidate(meanings=["first"])
    second = candidate(meanings=["second"], page=13)
    third = candidate(meanings=["third"], page=14)

    result = build_records([first, second, third], prepared(tmp_path))

    assert [record.id for record in result.records] == ["word:話す:はなす"]
    block = result.candidate_accounting
    assert block["version"] == 1
    assert block["parsed_candidate_count"] == 3
    assert block["canonical_record_count"] == 1
    assert block["unusable_candidate_count"] == 0
    assert block["duplicate_candidate_count"] == 2
    assert block["collision_group_count"] == 1
    assert block["collision_groups"] == [
        {
            "stable_record_id": "word:話す:はなす",
            "canonical_candidate_index": 1,
            "proposals": [
                {
                    "candidate_index": 1,
                    "parsed_schema_proposal": first.model_dump(mode="json"),
                },
                {
                    "candidate_index": 2,
                    "parsed_schema_proposal": second.model_dump(mode="json"),
                },
                {
                    "candidate_index": 3,
                    "parsed_schema_proposal": third.model_dump(mode="json"),
                },
            ],
        }
    ]
    assert block["candidate_accounting_fingerprint"] == (
        extract.candidate_accounting_fingerprint(block)
    )


def test_collision_indices_survive_known_record_sorting(tmp_path: Path) -> None:
    known_first = candidate(context="known canonical")
    fresh = candidate(expression="食べる", reading="たべる", context="fresh")
    known_duplicate = candidate(context="known duplicate", page=14)

    result = build_records(
        [known_first, fresh, known_duplicate],
        prepared(tmp_path),
        known_ids=["word:話す:はなす"],
    )

    assert [record.id for record in result.records] == [
        "word:食べる:たべる",
        "word:話す:はなす",
    ]
    [group] = result.candidate_accounting["collision_groups"]
    assert group["canonical_candidate_index"] == 1
    assert [item["candidate_index"] for item in group["proposals"]] == [1, 3]


def test_collision_accounting_needs_one_canonical_record_per_group(
    tmp_path: Path,
) -> None:
    result = build_records(
        [candidate(), candidate(page=2), candidate(page=3)],
        prepared(tmp_path),
    )
    block = result.candidate_accounting
    block["canonical_record_count"] = 0
    block["unusable_candidate_count"] = 1
    block["candidate_accounting_fingerprint"] = (
        extract.candidate_accounting_fingerprint(block)
    )

    with pytest.raises(
        extract.ExtractError,
        match="collision_group_count cannot exceed canonical_record_count",
    ):
        extract.validate_candidate_accounting_block(block)


def test_duplicate_stable_id_proposals_are_preserved_and_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A repeated model proposal is evidence, even though it cannot be a note.

    The first proposal remains the one canonical staging record. Every member
    of that identity's collision group stays as its complete parsed schema
    value in metadata, without janki choosing between their Japanese content.
    """
    root = project(tmp_path)
    first = candidate(meanings=["first proposal"], context="first source line")
    duplicate = candidate(
        meanings=["second proposal"],
        part_of_speech="second part of speech",
        examples=[
            {
                "japanese": "二つ目の例です。",
                "speech_level": "polite",
                "furigana": "二[ふた]つ 目[め]の 例[れい]です。",
                "romaji": "futatsume no rei desu.",
                "english": "This is the second example.",
            },
            {
                "japanese": "これは二つ目の例だ。",
                "speech_level": "casual",
                "furigana": "これは 二[ふた]つ 目[め]の 例[れい]だ。",
                "romaji": "kore wa futatsume no rei da.",
                "english": "This is the second example.",
            },
        ],
        usage_notes="second usage note",
        page=13,
        context="second source line",
        confidence="low",
        inclusion_reason="second reason",
    )
    third = candidate(
        meanings=["third proposal"],
        part_of_speech="third part of speech",
        usage_notes="third usage note",
        page=14,
        context="third source line",
        confidence="medium",
        inclusion_reason="third reason",
    )
    monkeypatch.setattr(
        cli.extract.claude_client,
        "parse_call",
        FakeCall(ok(first, duplicate, third)),
    )

    code = cli.main(["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))])

    assert code == 0
    records, meta = read_staging(root / "staging" / "lesson.pdf.yaml")
    assert [record.meanings for record in records] == [["first proposal"]]
    accounting = meta["candidate_accounting"]
    [group] = accounting["collision_groups"]
    assert [item["parsed_schema_proposal"] for item in group["proposals"]] == [
        first.model_dump(mode="json"),
        duplicate.model_dump(mode="json"),
        third.model_dump(mode="json"),
    ]
    assert meta["coverage"]["version"] == 2
    assert meta["coverage"]["candidate_accounting_fingerprint"] == accounting[
        "candidate_accounting_fingerprint"
    ]
    captured = capsys.readouterr()
    assert "2 duplicate proposals preserved" in captured.out
    assert "candidate_accounting" not in captured.err


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
    seed_prompts(tmp_path)
    if records is not None:
        (tmp_path / "vocabulary.json").write_text(
            json.dumps([item.to_dict() for item in records], ensure_ascii=False),
            encoding="utf-8",
        )
    return tmp_path


def default_inbox_project(tmp_path: Path) -> Path:
    root = project(tmp_path)
    config = root / "janki.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            'scan_inbox = "inbox"\n', ""
        ),
        encoding="utf-8",
    )
    return root


def source_pdf(tmp_path: Path, name: str = "lesson.pdf") -> Path:
    path = tmp_path / "desk" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PDF)
    return path


def test_an_unattended_run_refuses_to_send_rather_than_assuming_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The opposite default from every other prompt in janki, on purpose.

    Elsewhere "nobody is watching" means proceed, because the worst case is a
    deck built with a gap. Here the worst case is a photograph of someone's own
    notebook sent to a vendor, which cannot be recalled — and AGENTS.md forbids
    an agent from answering an approval prompt as the user. So no person means
    stop. Note pytest is itself non-interactive, which is why every scripted
    test above passes `--yes`.
    """
    root = project(tmp_path)
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path))])

    assert code == 1
    assert call.calls == [], "nothing was sent"
    assert not (root / "staging" / "lesson.pdf.yaml").exists()
    captured = capsys.readouterr()
    assert "Refusing" in captured.err
    # And it says what it *kept*, by path. The inbox copy already happened,
    # into a tracked directory, so reporting only "Nothing was sent." would
    # read as "nothing happened" while a private document waits for the next
    # git add. Asserting the phrase alone let a message naming no file — or
    # the wrong one — pass, so the path is what is pinned.
    kept = root / "inbox" / "lesson.pdf"
    assert kept.exists(), "the copy really did happen"
    assert f"kept in the inbox: {kept}" in captured.out
    # One stream: `2>/dev/null` must not strip the qualification off
    # "Nothing was sent."
    assert "Nothing was sent." in captured.out


@pytest.mark.parametrize(
    "typed,sent",
    [("y", True), ("yes", True), ("Y", True), ("", False), ("n", False), ("q", False)],
    ids=["y", "yes", "Y", "bare-enter", "n", "typo"],
)
def test_only_an_explicit_yes_sends_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, typed: str, sent: bool
) -> None:
    """The prompt reads `[y/N]`, so silence must mean no.

    The interactive branch is the whole point of the gate and had no test at
    all: rewriting the answer check to `not in {"n", "no"}` left the suite
    green, which would make a bare Enter — and every stray keystroke — send
    someone's private document to a paid API.
    """
    root = project(tmp_path)
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": typed)

    code = cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path))])

    assert bool(call.calls) is sent
    assert code == (0 if sent else 1)


@pytest.mark.parametrize("abort", [EOFError, KeyboardInterrupt], ids=["eof", "ctrl-c"])
def test_aborting_the_prompt_is_a_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, abort: type[BaseException]
) -> None:
    """Ctrl-C at the prompt is the most natural way to say stop, and it was the
    one answer nothing pinned.

    Measured: flipping the `except (EOFError, KeyboardInterrupt)` handler's
    `return False` to `return True` left the whole suite green — so a person
    who saw the file list, thought better of it, and hit Ctrl-C would have sent
    the document anyway. An interrupt is not an explicit yes, and this gate
    accepts nothing less.
    """
    root = project(tmp_path)
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)

    def interrupted(_prompt: str = "") -> str:
        raise abort()

    monkeypatch.setattr("builtins.input", interrupted)

    code = cli.main(["--root", str(root), "extract", str(source_pdf(tmp_path))])

    assert call.calls == [], "nothing was sent"
    assert code == 1
    assert not (root / "staging" / "lesson.pdf.yaml").exists()


def test_a_file_already_in_the_inbox_is_not_announced_as_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal reports what this run *stored*, not what happens to sit
    under the inbox.

    Every branch of `_copy_into_inbox` returns a path inside the inbox —
    including the four that copy nothing — so the first version of this
    message, which asked "is this path under the inbox", announced
    `kept in the inbox: …` for a file the run never touched. Telling someone
    their own already-filed evidence was just deposited by a run they refused
    is a small lie in a message whose entire job is to be trusted.
    """
    root = default_inbox_project(tmp_path)
    source = root / "data" / "inbox" / "lesson.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(PDF)
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(["--root", str(root), "extract", str(source)])

    out = capsys.readouterr().out
    assert code == 1 and call.calls == []
    assert "Nothing was sent." in out
    assert "kept in the inbox" not in out, "this run stored nothing"
    assert source.exists(), "and it is still where it always was"


def test_the_consent_prompt_names_the_files_and_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both halves, because neither alone is something a person can consent to.

    "Send 3 files to the API?" does not say which files; naming the files
    without the model does not say where they go. The refusal path is used here
    only because it is the one that prints the notice without needing a TTY.
    """
    root = project(tmp_path)
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    cli.main([
        "--root", str(root), "extract", str(source_pdf(tmp_path)),
        "--model", "claude-haiku-4-5",
    ])

    out = capsys.readouterr().out
    assert "lesson.pdf" in out, "the file is named"
    assert "claude-haiku-4-5" in out, "and the model it would go to"
    assert "paid" in out


def test_extract_writes_one_staging_file_per_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    one, two = source_pdf(tmp_path), source_pdf(tmp_path, "lesson2.pdf")
    monkeypatch.setattr(
        cli.extract.claude_client, "parse_call", FakeCall(ok(candidate()), ok(candidate()))
    )

    code = cli.main(["--root", str(root), "extract", "--yes", str(one), str(two)])

    assert code == 0
    assert (root / "staging" / "lesson.pdf.yaml").is_file()
    assert (root / "staging" / "lesson2.pdf.yaml").is_file()
    _, first_meta = read_staging(root / "staging" / "lesson.pdf.yaml")
    _, second_meta = read_staging(root / "staging" / "lesson2.pdf.yaml")
    assert first_meta["review_run_id"] != second_meta["review_run_id"]
    assert "Wrote 2 staging file(s)" in capsys.readouterr().out


def test_one_source_call_writes_cards_and_patterns_with_the_same_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    parsed = extract.candidate_schema()(
        candidates=[candidate()],
        document_kind="lesson",
        document_title="Week 11",
        patterns=[
            {
                "template": "〜んだ",
                "gloss": "explanation",
                "examples": ["大変だったんだ。"],
                "where": "page 1",
            }
        ],
    )
    call = FakeCall(CallResult(parsed, "end_turn", None))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(
        ["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))]
    )

    assert code == 0
    assert len(call.calls) == 1
    _records, meta = read_staging(root / "staging" / "lesson.pdf.yaml")
    stored = cli.patterns.load_store(root / "data" / "patterns.json")["lesson.pdf"]
    assert [item.template for item in stored.patterns] == ["〜んだ"]
    assert stored.prompt_provenance == meta["prompt_provenance"]
    assert meta["pattern_set"]["patterns"][0]["template"] == "〜んだ"
    assert stored.review_run_id == meta["review_run_id"]
    assert meta["pattern_set"]["review_run_id"] == meta["review_run_id"]


def test_extract_preserves_reviewed_patterns_but_stages_the_new_rich_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = project(tmp_path)
    pattern_path = root / "data" / "patterns.json"
    reviewed = patterns.PatternSet(
        source="lesson.pdf",
        kind="lesson",
        title="Reviewed lesson",
        patterns=(patterns.Pattern("〜たことがある", "past experience"),),
        reviewed=True,
        prompt_provenance={"request_fingerprint": "reviewed-request"},
    )
    patterns.save_store(pattern_path, {reviewed.source: reviewed})
    parsed = extract.candidate_schema()(
        candidates=[candidate(usage_notes="fresh card answer")],
        document_kind="lesson",
        document_title="Fresh model answer",
        patterns=[
            {
                "template": "〜んだ",
                "gloss": "explanation",
                "examples": ["大変だったんだ。"],
                "where": "page 1",
            }
        ],
    )
    call = FakeCall(CallResult(parsed, "end_turn", None))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(
        ["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))]
    )

    assert code == 0
    assert len(call.calls) == 1, "reviewed patterns do not skip the rich source call"
    assert patterns.load_store(pattern_path)["lesson.pdf"] == reviewed
    staged, meta = read_staging(root / "staging" / "lesson.pdf.yaml")
    assert staged[0].usage_notes == "fresh card answer"
    assert meta["pattern_set"]["reviewed"] is False
    assert meta["pattern_set"]["patterns"][0]["template"] == "〜んだ"
    assert "kept the reviewed patterns" in capsys.readouterr().out


def test_extract_reloads_patterns_after_the_paid_call_before_deciding_what_to_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A review completed during the model call is newer than its preflight read.

    The early load still validates the store before money is spent. The
    decision after the call must use a fresh load under the writer lock, or it
    replaces that new human decision and also erases unrelated entries added
    while the call was running.
    """
    root = project(tmp_path)
    pattern_path = root / "data" / "patterns.json"
    initial = patterns.PatternSet(
        source="lesson.pdf",
        kind="lesson",
        title="Awaiting review",
        patterns=(patterns.Pattern("〜たことがある", "past experience"),),
        reviewed=False,
        prompt_provenance={"request_fingerprint": "original-request"},
    )
    patterns.save_store(pattern_path, {initial.source: initial})
    unrelated = patterns.PatternSet(
        source="other.pdf",
        kind="lesson",
        title="Other lesson",
        patterns=(patterns.Pattern("〜ながら", "while doing"),),
        reviewed=True,
        prompt_provenance={"request_fingerprint": "other-request"},
    )
    parsed = extract.candidate_schema()(
        candidates=[candidate()],
        document_kind="lesson",
        document_title="Fresh model answer",
        patterns=[
            {
                "template": "〜んだ",
                "gloss": "explanation",
                "examples": ["大変だったんだ。"],
                "where": "page 1",
            }
        ],
    )

    class ReviewDuringCall(FakeCall):
        def __call__(self, *args: Any, **kwargs: Any) -> CallResult:
            current = patterns.load_store(pattern_path)
            current[initial.source] = replace(current[initial.source], reviewed=True)
            current[unrelated.source] = unrelated
            patterns.save_store(pattern_path, current)
            return super().__call__(*args, **kwargs)

    call = ReviewDuringCall(CallResult(parsed, "end_turn", None))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(
        ["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))]
    )

    assert code == 0
    after = patterns.load_store(pattern_path)
    assert after[initial.source].reviewed is True
    assert after[initial.source].patterns == initial.patterns
    assert after[unrelated.source] == unrelated
    assert "kept the reviewed patterns" in capsys.readouterr().out


def test_extract_force_replaces_reviewed_patterns_as_unreviewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    pattern_path = root / "data" / "patterns.json"
    reviewed = patterns.PatternSet(
        source="lesson.pdf",
        kind="lesson",
        patterns=(patterns.Pattern("〜たことがある", "past experience"),),
        reviewed=True,
        prompt_provenance={"request_fingerprint": "reviewed-request"},
    )
    patterns.save_store(pattern_path, {reviewed.source: reviewed})
    parsed = extract.candidate_schema()(
        candidates=[candidate()],
        document_kind="lesson",
        document_title="Fresh model answer",
        patterns=[
            {
                "template": "〜んだ",
                "gloss": "explanation",
                "examples": ["大変だったんだ。"],
                "where": "page 1",
            }
        ],
    )
    call = FakeCall(CallResult(parsed, "end_turn", None))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(
        [
            "--root",
            str(root),
            "extract",
            "--yes",
            "--force",
            str(source_pdf(tmp_path)),
        ]
    )

    assert code == 0
    assert len(call.calls) == 1
    replacement = patterns.load_store(pattern_path)["lesson.pdf"]
    assert replacement.reviewed is False
    assert replacement.title == "Fresh model answer"
    assert [item.template for item in replacement.patterns] == ["〜んだ"]
    _records, meta = read_staging(root / "staging" / "lesson.pdf.yaml")
    assert replacement.prompt_provenance == meta["prompt_provenance"]
    assert replacement.review_run_id == meta["review_run_id"]
    assert meta["pattern_set"]["review_run_id"] == meta["review_run_id"]


def _is_operation_record(relative: Path) -> bool:
    """The paid-operation journal and its captured answers.

    Held apart from "durable artifacts" in byte-for-byte comparisons: these
    files exist precisely to change when a paid call happens, including —
    especially — when it fails.
    """
    parts = relative.parts
    return "operations.json" in parts or ".pending" in parts


def test_schema_validation_failure_preserves_forced_targets_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A paid but malformed answer cannot replace either durable artifact."""
    root = project(tmp_path)
    source = source_pdf(tmp_path)
    monkeypatch.setattr(
        cli.extract.claude_client,
        "parse_call",
        FakeCall(ok(candidate())),
    )
    assert cli.main(["--root", str(root), "extract", "--yes", str(source)]) == 0

    target = root / "staging" / "lesson.pdf.yaml"
    pattern_path = root / "data" / "patterns.json"
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not _is_operation_record(path.relative_to(root))
    }
    assert target.relative_to(root) in before
    assert pattern_path.relative_to(root) in before

    invalid_candidate = candidate().model_dump(mode="json")
    invalid_candidate["meanings"] = []
    invalid_candidate["examples"] = []
    response = type(
        "Response",
        (),
        {
            "stop_reason": "end_turn",
            "content": [
                type(
                    "TextBlock",
                    (),
                    {
                        "type": "text",
                        "text": json.dumps({"candidates": [invalid_candidate]}),
                    },
                )()
            ],
        },
    )()

    def return_invalid_response(
        model: str,
        _system: Any,
        _content: Any,
        schema: Any,
        _client: Any = None,
        **_kwargs: Any,
    ) -> CallResult:
        return cli.extract.claude_client._result_of(response, schema, model)

    monkeypatch.setattr(
        cli.extract.claude_client,
        "parse_call",
        return_invalid_response,
    )

    assert cli.main(
        ["--root", str(root), "extract", "--yes", "--force", str(source)]
    ) == 1
    assert "Nothing was written for this source." in capsys.readouterr().err
    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not _is_operation_record(path.relative_to(root))
    }
    assert after == before

    # The one thing a failed paid call *must* leave behind. Excluding it above
    # is not a loosening of the guarantee but the point of it: the durable
    # artifacts a reviewer reads are untouched, while the journal records that
    # money was spent and what came back, so the next run can tell "never sent"
    # from "sent and refused".
    journal = operations.OperationJournal.load(root / "data" / "operations.json")
    states = sorted(op.state for op in journal.operations.values())
    # The first run committed; the second was paid for and refused, and says so
    # rather than vanishing.
    assert states == ["committed", "outcome_unknown"], journal.operations
    failed = next(
        op for op in journal.operations.values() if op.state == "outcome_unknown"
    )
    assert failed.money_may_have_been_spent is True
    assert failed.detail


def test_extract_force_help_names_the_reviewed_pattern_reset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The flag replaces human-reviewed state, not only disposable output."""
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["extract", "--help"])

    assert excinfo.value.code == 0
    help_text = " ".join(capsys.readouterr().out.lower().split())
    assert "replace a reviewed pattern set" in help_text
    assert "unreviewed" in help_text


def test_cli_reuses_a_source_in_the_default_parent_inbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = default_inbox_project(tmp_path)
    source = root / "data" / "inbox" / "lesson.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(PDF)
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(["--root", str(root), "extract", "--yes", str(source)])

    assert code == 0
    assert len(call.calls) == 1
    assert not (root / "data" / "inbox" / "scans").exists()
    assert (root / "staging" / "lesson.pdf.yaml").is_file()


def test_cli_custom_scan_inbox_remains_its_own_durable_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    source = source_pdf(tmp_path)
    monkeypatch.setattr(
        cli.extract.claude_client, "parse_call", FakeCall(ok(candidate()))
    )

    assert cli.main(["--root", str(root), "extract", "--yes", str(source)]) == 0

    assert (root / "inbox" / "lesson.pdf").read_bytes() == PDF


def test_cli_refuses_cross_run_durable_basename_collisions_before_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = default_inbox_project(tmp_path)
    source = root / "data" / "inbox" / "lesson.pdf"
    existing = root / "data" / "inbox" / "scans" / "lesson.pdf"
    source.parent.mkdir(parents=True)
    existing.parent.mkdir(parents=True)
    source.write_bytes(PDF + b" root")
    existing.write_bytes(PDF + b" scan")
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(["--root", str(root), "extract", "--yes", "--force", str(source)])

    assert code == 1
    assert call.calls == []
    assert not (root / "staging" / "lesson.pdf.yaml").exists()
    assert "same basename" in capsys.readouterr().err


def test_cli_refuses_case_only_durable_name_collisions_before_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = default_inbox_project(tmp_path)
    first = root / "data" / "inbox" / "Lesson.pdf"
    source = root / "data" / "inbox" / "scans" / "lesson.pdf"
    first.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    first.write_bytes(PDF + b" root")
    source.write_bytes(PDF + b" scan")
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(["--root", str(root), "extract", "--yes", "--force", str(source)])

    assert code == 1
    assert call.calls == []
    assert not (root / "staging" / "Lesson.pdf.yaml").exists()
    error = capsys.readouterr().err
    assert "same basename" in error
    assert str(first) in error


def test_cli_does_not_duplicate_an_external_source_in_a_colliding_inbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = default_inbox_project(tmp_path)
    first = root / "data" / "inbox" / "lesson.pdf"
    existing = root / "data" / "inbox" / "scans" / "lesson.pdf"
    source = tmp_path / "desk" / "lesson.pdf"
    first.parent.mkdir(parents=True)
    existing.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    first.write_bytes(PDF + b" root")
    existing.write_bytes(PDF + b" scan")
    source.write_bytes(PDF + b" scan")
    call = FakeCall(ok(candidate()))
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", call)

    code = cli.main(["--root", str(root), "extract", "--yes", str(source)])

    assert code == 1
    assert call.calls == []
    assert list(existing.parent.iterdir()) == [existing]
    assert "same basename" in capsys.readouterr().err


def test_the_staging_file_is_the_pinned_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    cli.main(["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))])

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


def test_the_staging_file_records_its_provenance_and_never_the_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three assertions that rode on a deleted oracle test and had nothing to do
    with oracles.

    Staging files are committed. The `provider` field sits one string
    concatenation away from the environment variable holding the API key, so
    the leak guard belongs beside the thing it guards — measured after M8.4
    removed it: writing the key into `provider` left the whole suite green.

    The block's *presence* matters just as much. Drop it and every later
    `janki promote` dies with `[prompt-provenance-invalid]`, which turns paid
    extraction output into files nothing can consume — and that too was green.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-stored")
    root = project(tmp_path)
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    cli.main(["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))])

    _records, meta = read_staging(root / "staging" / "lesson.pdf.yaml")
    provenance = meta["prompt_provenance"]
    assert provenance["provider"] == "anthropic"
    assert provenance["response_schema_version"] == extract.EXTRACTION_SCHEMA_VERSION
    assert "must-not-be-stored" not in json.dumps(meta, ensure_ascii=False)
    assert str(root) not in json.dumps(meta, ensure_ascii=False)


def test_prompt_fingerprints_cover_the_anthropic_wire_schema_not_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = {"value": "wire-v1"}

    class FakeAnthropic:
        @staticmethod
        def transform_schema(schema: Any) -> dict[str, Any]:
            return {"wire": marker["value"], "name": schema.__name__}

    monkeypatch.setattr(extract.claude_client, "load_anthropic", lambda: FakeAnthropic)
    item = prepared(tmp_path)
    base = extract.prompt_provenance(
        item, model="m", style_guide="style", system="S", mode="prose", known=()
    )
    changed = extract.prompt_provenance(
        item, model="m", style_guide="style", system="S", mode="prose", known=("話す",)
    )
    assert base["system_prompt_fingerprint"] == changed["system_prompt_fingerprint"]
    assert base["user_prompt_fingerprint"] != changed["user_prompt_fingerprint"]
    assert base["response_schema_fingerprint"] == prompts.schema_fingerprint(
        {"wire": "wire-v1", "name": "Extraction"}
    )
    assert base["request_fingerprint"] != changed["request_fingerprint"]
    assert str(tmp_path) not in json.dumps(base)

    marker["value"] = "wire-v2"
    transformed = extract.prompt_provenance(
        item, model="m", style_guide="style", system="S", mode="prose", known=()
    )
    assert transformed["response_schema_fingerprint"] != base[
        "response_schema_fingerprint"
    ]
    assert transformed["request_fingerprint"] != base["request_fingerprint"]


def test_the_input_is_copied_into_the_inbox_and_cited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A candidate has to stay checkable against the page it came from.
    root = project(tmp_path)
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    cli.main(["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))])

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

    code = cli.main(["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))])

    assert code == 1
    assert "mid-review" in (root / "staging" / "lesson.pdf.yaml").read_text(encoding="utf-8")
    assert "already exists" in capsys.readouterr().err


def test_force_overwrites_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = project(tmp_path)
    (root / "staging").mkdir()
    (root / "staging" / "lesson.pdf.yaml").write_text("records: []\n", encoding="utf-8")
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    code = cli.main(
        ["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path)), "--force"]
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
            "--yes",
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

    cli.main([
        "--root", str(root), "extract", "--yes",
        str(source_pdf(tmp_path)), "--mode", "table",
    ])
    cli.main(
        [
            "--root",
            str(root),
            "extract",
            "--yes",
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

    cli.main(["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))])

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

    code = cli.main(["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))])

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

    code = cli.main(["--root", str(root), "extract", "--yes", str(one), str(two)])

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

    cli.main(["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))])

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

    code = cli.main(["--root", str(root), "extract", "--yes", str(source), str(source)])

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

    assert cli.main(["--root", str(root), "extract", "--yes", str(scan), str(photo)]) == 0

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
                    confidence="low",
                    inclusion_reason="new in this chapter",
                ),
            )
        ),
    )

    cli.main(["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))])

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
    # Every field the model filled in, not a hand-picked few: the proposed
    # example and low-confidence flag both matter to whoever re-adds the word.
    assert "日本語を話します。" in note
    assert "verb" in note
    assert "low" in note
    assert "new in this chapter" in note


def test_a_held_back_row_whose_text_values_are_zero_is_still_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A row for 〇/ゼロ: the gloss and the source cell are both "0". Filtering
    # rendered text by truthiness must not swallow them.
    root = project(tmp_path)
    monkeypatch.setattr(
        cli.extract.claude_client,
        "parse_call",
        FakeCall(
            ok(candidate(expression="", reading="ゼロ", meanings=["0"], context="0", page=1))
        ),
    )

    cli.main(["--root", str(root), "extract", "--yes", str(source_pdf(tmp_path))])

    _records, meta = read_staging(root / "staging" / "lesson.pdf.yaml")
    note = meta["review_notes"]
    assert "meanings: 0" in note
    assert "context: 0" in note
    assert "page: 1" in note


def test_an_interrupted_call_is_journaled_before_it_is_made(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The interval the journal exists for: the request is sent, the answer
    never arrives, and the process has to be able to say afterwards that money
    may have been spent. Before the journal there was nothing on disk that
    could tell "never sent" from "sent and lost"."""
    root = project(tmp_path)
    source = source_pdf(tmp_path)

    def connection_dies(*_args: object, **_kwargs: object) -> None:
        raise JankiError("connection reset while streaming the answer")

    monkeypatch.setattr(cli.extract.claude_client, "parse_call", connection_dies)

    assert cli.main(["--root", str(root), "extract", "--yes", str(source)]) == 1

    journal = operations.OperationJournal.load(root / "data" / "operations.json")
    assert len(journal.operations) == 1
    operation = next(iter(journal.operations.values()))
    assert operation.state == "outcome_unknown"
    assert operation.money_may_have_been_spent is True
    assert operation.source_file == "lesson.pdf"
    assert operation.request_fp
    assert "connection reset" in operation.detail
    # ...and it is what a person is told to look at.
    assert journal.needing_attention() == [operation]

    stderr = capsys.readouterr().err
    assert "may already have been billed" in stderr
    assert "second charge" in stderr
    staging_dir = root / "staging"
    assert staging_dir.is_dir()
    assert list(staging_dir.iterdir()) == []


def test_a_completed_extraction_leaves_a_committed_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished call is not something a person has to look at. It stays in
    the journal as history, and stops being a question."""
    root = project(tmp_path)
    source = source_pdf(tmp_path)
    monkeypatch.setattr(
        cli.extract.claude_client, "parse_call", FakeCall(ok(candidate()))
    )

    assert cli.main(["--root", str(root), "extract", "--yes", str(source)]) == 0

    journal = operations.OperationJournal.load(root / "data" / "operations.json")
    operation = next(iter(journal.operations.values()))
    assert operation.state == "committed"
    assert journal.needing_attention() == []
    assert operation.artifact is not None
    assert (root / "data" / operation.artifact.relative_name).exists()


def test_the_authority_is_recorded_before_the_request_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordering is the whole guarantee. If the journal entry appeared only
    after a successful call, the crash it protects against would take the
    record with it."""
    root = project(tmp_path)
    source = source_pdf(tmp_path)
    seen: list[str] = []

    def observe(*_args: object, **_kwargs: object) -> None:
        journal = operations.OperationJournal.load(
            root / "data" / "operations.json"
        )
        seen.extend(op.state for op in journal.operations.values())
        raise JankiError("stopped after the journal was written")

    monkeypatch.setattr(cli.extract.claude_client, "parse_call", observe)
    cli.main(["--root", str(root), "extract", "--yes", str(source)])

    # At the moment the provider was called, the journal already said so.
    assert seen == ["dispatching"]


def test_the_journal_names_the_request_that_was_actually_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The journal's whole purpose is that a crash between sending and parsing
    can still say which call was made. That is only true while the fingerprint
    it recorded describes the request the dispatch went on to build.

    Those are computed in two places — the plan journals one before dispatch,
    and `extract_candidates` records another into the staging file from the
    arguments it was actually handed — so they are independently derived and
    must agree. Nothing else compares them: asserting the journalled
    fingerprint is merely *present*, or merely *stable between two identical
    plans*, passes just as happily when the plan fingerprints a prompt with the
    known-words block and the dispatch sends one without it.

    Prose mode over a non-empty collection, because that is the only shape
    where the known list is non-empty and so the only shape where the two can
    disagree.
    """
    root = project(
        tmp_path,
        [VocabularyRecord(id="word:話す:はなす", expression="話す", reading="はなす")],
    )
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    assert cli.main([
        "--root", str(root), "extract", "--yes",
        str(source_pdf(tmp_path)), "--mode", "prose",
    ]) == 0

    journal = operations.OperationJournal.load(root / "data" / "operations.json")
    [operation] = journal.operations.values()
    staged = sorted((root / "staging").glob("*.yaml"))
    _records, meta = read_staging(staged[0])

    assert operation.request_fp
    assert operation.request_fp == meta["prompt_provenance"]["request_fingerprint"]


def test_the_known_list_is_ordered_so_a_rerun_has_one_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The known words go into the prompt the request fingerprint is computed
    over, so an unordered set gives the same corpus a different request
    identity on every run — and that identity is what the journal records and
    a retry compares.

    Two separate processes would be needed to see it through hash
    randomization, so this asserts the ordering directly instead.
    """
    root = project(
        tmp_path,
        [
            VocabularyRecord(id="word:話す:はなす", expression="話す", reading="はなす"),
            VocabularyRecord(id="word:食べる:たべる", expression="食べる", reading="たべる"),
            VocabularyRecord(id="word:飲む:のむ", expression="飲む", reading="のむ"),
        ],
    )
    monkeypatch.setattr(cli.extract.claude_client, "parse_call", FakeCall(ok(candidate())))

    plan = plan_extraction(
        ProjectConfig.load(root),
        prepare_inputs([source_pdf(tmp_path)], root / "inbox"),
        mode="prose",
        model="claude-opus-5",
        style_guide="style",
        system="system",
    )

    assert plan.skip_list == tuple(sorted(plan.skip_list))
    assert len(plan.skip_list) == 3


def test_an_answer_refused_after_it_arrived_says_it_was_paid_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The reply arrived, was stored, and something after it refused — a
    schema mismatch here. The bytes are on disk and have been billed, so the
    message has to name the safe exact reader rather than merely print a path
    whose current occupant it cannot prove.
    """
    root = project(tmp_path)
    source = source_pdf(tmp_path)

    def answer_then_refuse(*_args: object, capture: Any = None, **_kwargs: object) -> None:
        if capture is not None:
            capture({"content": [{"type": "text", "text": '{"candidates": []}'}]})
        raise JankiError("the answer did not match the schema")

    monkeypatch.setattr(cli.extract.claude_client, "parse_call", answer_then_refuse)

    assert cli.main(["--root", str(root), "extract", "--yes", str(source)]) == 1

    journal = operations.OperationJournal.load(root / "data" / "operations.json")
    [operation] = journal.operations.values()
    # Left where it is: calling a captured answer "unknown" hides one already
    # bought.
    assert operation.state == "result_captured"
    assert operation.artifact

    stderr = capsys.readouterr().err
    assert "arrived and was saved before it was refused" in stderr
    assert f"janki operations --show-reply {operation.operation_id}" in stderr
    assert operation.artifact.relative_name not in stderr
    for private_value in {
        operation.artifact.content_sha256,
        *(str(value) for value in operation.artifact.directory_identity),
        str(operation.artifact.entry_state[1]),
    }:
        assert private_value not in stderr
    assert "exact paid reply" in stderr


def test_a_reply_of_pure_reasoning_says_there_is_nothing_to_recover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A call that reaches `max_tokens` while still thinking returns a real,
    billed reply holding no answer. Telling someone it was saved sends them
    looking for cards in a file that has none — so this branch says the
    opposite, while still naming the safe exact reader."""
    root = project(tmp_path)
    source = source_pdf(tmp_path)

    def think_then_stop(*_args: object, capture: Any = None, **_kwargs: object) -> None:
        if capture is not None:
            capture({"content": [{"type": "thinking", "thinking": "reading the page"}]})
        raise JankiError("max_tokens reached before an answer")

    monkeypatch.setattr(cli.extract.claude_client, "parse_call", think_then_stop)

    assert cli.main(["--root", str(root), "extract", "--yes", str(source)]) == 1

    journal = operations.OperationJournal.load(root / "data" / "operations.json")
    [operation] = journal.operations.values()
    assert operation.state == "result_captured"

    stderr = capsys.readouterr().err
    assert "contains no answer — only the model's reasoning" in stderr
    assert f"janki operations --show-reply {operation.operation_id}" in stderr
    assert operation.artifact is not None
    assert operation.artifact.relative_name not in stderr
    for private_value in {
        operation.artifact.content_sha256,
        *(str(value) for value in operation.artifact.directory_identity),
        str(operation.artifact.entry_state[1]),
    }:
        assert private_value not in stderr
    assert "There is nothing in it to recover" in stderr


# --- what a paid answer becomes ---------------------------------------------


def _fixture_result(
    root: Path,
    scenario: str,
    name: str,
    extra: tuple = (),
    settle: bool = True,
    *,
    model: str = "claude-opus-5",
    mode: str | None = None,
):
    """One W0 fixture answer, parsed exactly as a live call would deliver it.

    `extra` appends candidates to the model's answer *before* normalization,
    which is where a real one would arrive — the fixtures are deliberately
    well-formed, so a malformed row has to be asked for.

    `settle=False` stops after `authorize_dispatch`, which is exactly where a
    caller whose provider wrapper never fired the capture hook ends up: the
    entry says the call is in flight and the answer is in hand anyway. Reached
    by leaving a step out rather than by winding the journal backwards, so the
    state under test is one the public API actually produces.
    """
    item = _prepared(root, name)
    parsed = _load(scenario)
    if extra:
        parsed = parsed.model_copy(
            update={"candidates": [*parsed.candidates, *extra]}
        )
    normalized = extract.normalize_response(parsed, mode, item.origin_path.name)
    provenance = extract.prompt_provenance(
        item, model=model, style_guide="s", system="y", mode=mode,
        source_sha256=extract.source_fingerprint(item.origin_path),
    )
    result = dataclasses.replace(
        normalized,
        pattern_set=patterns.with_prompt_provenance(normalized.pattern_set, provenance),
    )
    config = ProjectConfig.load(root)
    journal = operations.OperationJournal.load(config.operations_file)
    target = ExtractionTarget(
        item=item,
        staging_path=config.staging_dir / f"{item.origin_path.name}.yaml",
        patterns_path=config.patterns_file,
        source_sha256=str(provenance["source_sha256"]),
        provenance=provenance,
    )
    operation_id = authorize_dispatch(journal, target, model=model)
    if settle:
        journal.capture_result(
            operation_id,
            lambda: operations.capture_artifact(
                config.operations_file, operation_id, b'{"fixture": true}'
            ),
        )
    return config, journal, target, result, operation_id


def test_the_journal_commits_only_once_the_answer_is_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"Committed" means the exact answer became staging, and nothing else. A
    write that fails must leave the entry saying a paid reply is still waiting
    for somebody — not that it was already dealt with."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "japanese_anki.application.extraction.write_staging_under_lock", refuse
    )
    with pytest.raises(OSError):
        complete_extraction(
            config, journal, target, result, operation_id=operation_id,
            known=set(), mode=None, model="claude-opus-5",
        )

    entry = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert entry.state == "result_captured"


def test_forget_between_completion_preflight_and_write_changes_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final state check, staging write, and committed transition are one
    journal-locked decision. A forget landing after an earlier snapshot must
    win before any card or pattern output changes."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )
    pattern_before = (
        config.patterns_file.read_bytes()
        if config.patterns_file.exists()
        else None
    )
    entered_build = threading.Event()
    release_build = threading.Event()
    real_build = extract.build_records

    def pause_after_preflight(*args: Any, **kwargs: Any) -> Any:
        entered_build.set()
        if not release_build.wait(5):
            raise AssertionError("completion race was never released")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(extract, "build_records", pause_after_preflight)
    failures: list[BaseException] = []

    def complete() -> None:
        try:
            complete_extraction(
                config,
                journal,
                target,
                result,
                operation_id=operation_id,
                known=set(),
                mode=None,
                model="claude-opus-5",
            )
        except BaseException as exc:  # noqa: BLE001 - asserted across the thread
            failures.append(exc)

    worker = threading.Thread(target=complete)
    worker.start()
    assert entered_build.wait(5), "completion never reached its post-preflight work"

    operations.OperationJournal.load(config.operations_file).forget(
        [operation_id], force=True
    )
    release_build.set()
    worker.join(5)

    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], operations.OperationError)
    assert not target.staging_path.exists()
    assert (
        config.patterns_file.read_bytes()
        if config.patterns_file.exists()
        else None
    ) == pattern_before


def test_completion_of_a_cleanup_tombstone_changes_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A result in memory cannot revive a call whose final forget decision is
    already durable, even when cleanup has not yet retired its paid artifact."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )
    held = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert held.artifact is not None
    artifact = config.operations_file.parent / held.artifact.relative_name
    artifact_before = artifact.read_bytes()
    pattern_before = (
        config.patterns_file.read_bytes()
        if config.patterns_file.exists()
        else None
    )

    def leave_cleanup_pending(_binding: Any) -> None:
        raise operations.OperationError("injected cleanup pause")

    with monkeypatch.context() as cleanup_patch:
        cleanup_patch.setattr(
            operations, "_retire_artifact", leave_cleanup_pending
        )
        with pytest.raises(operations.OperationError, match="cleanup pause"):
            journal.forget([operation_id], force=True)

    tombstone = operations.OperationJournal.load(
        config.operations_file
    ).operations[operation_id]
    assert tombstone.state == "result_captured"
    assert tombstone.cleanup is not None

    with pytest.raises(operations.OperationError, match="being forgotten"):
        settle_dispatch(config, journal, operation_id, result)

    with pytest.raises(operations.OperationError, match="being forgotten"):
        complete_extraction(
            config,
            journal,
            target,
            result,
            operation_id=operation_id,
            known=set(),
            mode=None,
            model="claude-opus-5",
        )

    assert not target.staging_path.exists()
    assert artifact.read_bytes() == artifact_before
    assert (
        config.patterns_file.read_bytes()
        if config.patterns_file.exists()
        else None
    ) == pattern_before


def test_a_reviewed_pattern_answer_is_not_replaced_by_a_rerun(
    tmp_path: Path,
) -> None:
    """A person's grammar review outranks a fresh unreviewed answer. The new
    answer is not lost — the staging file carries a complete copy — but the
    store keeps the reviewed one until somebody asks otherwise."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )
    complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode=None, model="claude-opus-5",
    )
    store = patterns.load_store(config.patterns_file)
    source = result.pattern_set.source
    store[source] = dataclasses.replace(store[source], reviewed=True)
    patterns.save_store(config.patterns_file, store)

    # The review has been promoted and archived, so the second run has a clear
    # path to write — which is the ordinary way a source gets re-read.
    target.staging_path.unlink()

    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )
    outcome = complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode=None, model="claude-opus-5",
    )

    assert outcome.kept_reviewed_patterns is True
    assert patterns.load_store(config.patterns_file)[source].reviewed is True
    # The new answer is not lost: the staging file carries a complete copy of
    # the grammar this run proposed, whatever the store decided to keep.
    _records, meta = read_staging(outcome.target)
    assert meta["pattern_set"]["patterns"]


def test_forcing_a_rerun_replaces_a_reviewed_pattern_answer(
    tmp_path: Path,
) -> None:
    """The other half, and the reason the keep is not simply a refusal:
    `--force` is how somebody says they want this answer instead, and it must
    actually land."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )
    complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode=None, model="claude-opus-5",
    )
    store = patterns.load_store(config.patterns_file)
    source = result.pattern_set.source
    store[source] = dataclasses.replace(store[source], reviewed=True)
    patterns.save_store(config.patterns_file, store)

    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )
    outcome = complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode=None, model="claude-opus-5", force=True,
    )

    assert outcome.kept_reviewed_patterns is False
    assert patterns.load_store(config.patterns_file)[source].reviewed is False


def test_unusable_candidates_are_written_into_the_file_a_reviewer_reads(
    tmp_path: Path,
) -> None:
    """A count that lives only in scrollback is a silent discard with an extra
    step: the staging file is what somebody opens later."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root,
        "reading_holds",
        "holds.pdf",
        # A row the model read but could mint no ID from: a record's ID comes
        # from its expression, and this one has none.
        extra=(candidate(expression="  ", page=12, reading="およぐ",
                         meanings=["to swim"]),),
    )

    outcome = complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode=None, model="claude-opus-5",
    )

    assert outcome.unusable == 1
    _records, meta = read_staging(outcome.target)
    note = meta["review_notes"]
    assert "1 candidate(s) could not be stored" in note
    # Everything the model did read about the row, so the page can be
    # re-checked rather than merely known to be incomplete.
    assert "page: 12" in note
    assert "およぐ" in note
    assert "to swim" in note


def test_the_model_that_answered_is_the_one_written_down(tmp_path: Path) -> None:
    """`--model` picks who answers, and the staging file has to say who did.
    Every caller happens to pass the default, so nothing noticed that the
    parameter was threaded through but never read."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root,
        "lesson_with_grammar",
        "lesson.pdf",
        model="claude-haiku-4-5-20251001",
    )

    outcome = complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode=None, model="claude-haiku-4-5-20251001",
    )

    _records, meta = read_staging(outcome.target)
    assert meta["model"] == "claude-haiku-4-5-20251001"


def test_the_mode_a_run_was_read_under_reaches_its_coverage_verdict(
    tmp_path: Path,
) -> None:
    """`--mode table` is a claim about the source — it has rows to be
    exhaustive about — so an answer reporting no units is a hole that blocks
    rather than a selection nobody promised was complete. Drop the parameter
    on the way to `coverage_block` and that verdict silently goes lenient."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "pattern_only_chart", "chart.pdf", mode="table"
    )

    outcome = complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode="table", model="claude-opus-5",
    )

    _records, meta = read_staging(outcome.target)
    assert meta["coverage"]["status"] == "unmeasured"
    assert meta["coverage"]["blocking"] is True


def test_a_staging_file_is_not_overwritten_by_a_completion_without_force(
    tmp_path: Path,
) -> None:
    """The plan's collision check is stale by the time the answer arrives: the
    file can appear during the call — another dispatch, or somebody's hand
    edit. This is the last gate before a paid answer overwrites one."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )
    complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode=None, model="claude-opus-5",
    )
    kept = target.staging_path.read_bytes()

    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )
    with pytest.raises(JankiError) as caught:
        complete_extraction(
            config, journal, target, result, operation_id=operation_id,
            known=set(), mode=None, model="claude-opus-5",
        )

    assert "force" in str(caught.value)
    assert target.staging_path.read_bytes() == kept
    # And the entry still says a paid answer is waiting for somebody, which is
    # true: it was never turned into staging.
    entry = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert entry.state == "result_captured"


def test_a_dispatch_that_was_never_settled_is_completed_not_discarded(
    tmp_path: Path,
) -> None:
    """A caller that skipped `settle_dispatch` — a browser handler with a
    missing branch — arrives holding an answer somebody paid for, under an
    entry still claiming the call is in flight.

    Refusing there would be the worst of both: the journal made truthful by
    throwing away the very answer it exists to protect. The answer is recorded
    and the run completes."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf", settle=False
    )
    assert journal.operations[operation_id].state == "dispatching"

    outcome = complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode=None, model="claude-opus-5",
    )

    assert outcome.records == 2
    entry = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert entry.state == "committed"
    # And the answer is on disk under that entry, not merely counted. The
    # exact provider bytes were never captured — the hook is what captures
    # those — so this is `settle_dispatch`'s normalized fallback, which is
    # still what a person can read and re-enter by hand.
    assert entry.artifact is not None
    artifact = config.operations_file.parent / entry.artifact.relative_name
    assert "あげる" in artifact.read_text(encoding="utf-8")


def test_a_journal_that_never_saw_the_capture_still_completes(
    tmp_path: Path,
) -> None:
    """W3's shape: a request handler holds one journal while the client's
    capture hook advances another over the same file. The handler's copy is
    stale the moment the answer lands, so a pre-flight that asked it would
    refuse a run whose answer is captured and on disk — and refuse it while
    holding that answer."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf", settle=False
    )
    client_journal = operations.OperationJournal.load(config.operations_file)
    capture_hook(config, client_journal, operation_id)(
        {"content": [{"type": "text", "text": "the exact provider bytes"}]}
    )
    captured_receipt = client_journal.operations[operation_id].artifact
    assert captured_receipt is not None
    captured = (
        config.operations_file.parent / captured_receipt.relative_name
    ).read_bytes()

    # The handler's journal still says the call is in flight. Only the file
    # knows otherwise.
    assert journal.operations[operation_id].state == "dispatching"
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "result_captured"
    )

    outcome = complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode=None, model="claude-opus-5",
    )

    assert outcome.records == 2
    entry = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert entry.state == "committed"
    # The hook's own bytes, not the normalized fallback settling would have
    # written. Compared against a value read *before* the run: artifacts are
    # named from the operation id, so re-reading the path the entry names
    # would compare the file to itself and pass however it was clobbered.
    assert entry.artifact is not None
    assert (
        config.operations_file.parent / entry.artifact.relative_name
    ).read_bytes() == captured


def test_an_answer_that_arrives_while_the_call_reads_as_running_is_kept(
    tmp_path: Path,
) -> None:
    """`running` is where a caller that reports streaming sits, and an answer
    arriving there is exactly as paid for as one arriving from `dispatching`.
    Settling only the latter would refuse this run at the pre-flight while
    holding the answer — the inversion this whole path exists to avoid."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf", settle=False
    )
    journal.advance(operation_id, "running")

    outcome = complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode=None, model="claude-opus-5",
    )

    assert outcome.records == 2
    entry = operations.OperationJournal.load(config.operations_file).operations[
        operation_id
    ]
    assert entry.state == "committed"
    assert entry.artifact is not None
    assert "あげる" in (
        config.operations_file.parent / entry.artifact.relative_name
    ).read_text(encoding="utf-8")


def test_an_answer_is_not_recorded_against_another_run_s_operation(
    tmp_path: Path,
) -> None:
    """A caller that pairs one run's operation with another run's target.
    Nothing about the *states* is wrong, so a pre-flight checking only those
    passes both — and the entry ends up saying its exact answer became a
    staging file it never described.

    The two operations are sequential because the journal will not authorize a
    second paid call while one is unsettled; the confusion this guards against
    is a variable, not a race."""
    root = project(tmp_path)
    config, journal, first, first_result, other_operation = _fixture_result(
        root, "table_exhaustive", "table.pdf"
    )
    complete_extraction(
        config, journal, first, first_result, operation_id=other_operation,
        known=set(), mode=None, model="claude-opus-5",
    )
    config, journal, target, result, _own = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )

    with pytest.raises(operations.OperationError) as caught:
        complete_extraction(
            config, journal, target, result, operation_id=other_operation,
            known=set(), mode=None, model="claude-opus-5",
        )

    # Identity, not state: the mismatch is caught before the pre-flight would
    # have refused it for being committed.
    assert "different request" in str(caught.value)
    assert not target.staging_path.exists()
    # And the mismatched entry still describes its own run.
    entry = operations.OperationJournal.load(config.operations_file).operations[
        other_operation
    ]
    assert entry.state == "committed"
    assert entry.source_file == "table.pdf"


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("kind", "enrich"),
        ("source_file", "other.pdf"),
        ("source_sha256", "0" * 64),
        ("request_fp", "1" * 64),
        ("model", "claude-other-model"),
    ],
)
def test_completion_requires_the_journal_s_full_request_identity(
    tmp_path: Path,
    field: str,
    wrong: str,
) -> None:
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )
    current = operations.OperationJournal.load(config.operations_file)
    current.operations[operation_id] = dataclasses.replace(
        current.operations[operation_id],
        **{field: wrong},
    )
    current._write()

    with pytest.raises(operations.OperationError, match="different request"):
        complete_extraction(
            config,
            journal,
            target,
            result,
            operation_id=operation_id,
            known=set(),
            mode=None,
            model="claude-opus-5",
        )

    assert not target.staging_path.exists()


@pytest.mark.parametrize(
    "crosswire",
    [
        "prompt_provenance",
        "result_source_sha256",
        "result_model",
        "pattern_source",
        "target_source",
        "model",
        "mode",
    ],
)
def test_completion_binds_the_answer_and_arguments_to_the_target_request(
    tmp_path: Path,
    crosswire: str,
) -> None:
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )
    model = "claude-opus-5"
    mode = None
    if crosswire == "prompt_provenance":
        provenance = {
            **result.pattern_set.prompt_provenance,
            "request_fingerprint": "f" * 64,
        }
        result = dataclasses.replace(
            result,
            pattern_set=dataclasses.replace(
                result.pattern_set,
                prompt_provenance=provenance,
            ),
        )
    elif crosswire in {"result_source_sha256", "result_model"}:
        field, value = (
            ("source_sha256", "0" * 64)
            if crosswire == "result_source_sha256"
            else ("model", "claude-other-model")
        )
        result = dataclasses.replace(
            result,
            pattern_set=dataclasses.replace(
                result.pattern_set,
                prompt_provenance={
                    **result.pattern_set.prompt_provenance,
                    field: value,
                },
            ),
        )
    elif crosswire == "pattern_source":
        result = dataclasses.replace(
            result,
            pattern_set=dataclasses.replace(result.pattern_set, source="other.pdf"),
        )
    elif crosswire == "target_source":
        target = dataclasses.replace(target, source_sha256="0" * 64)
    elif crosswire == "model":
        model = "claude-other-model"
    elif crosswire == "mode":
        mode = "prose"
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(crosswire)

    with pytest.raises(operations.OperationError, match="answer does not describe"):
        complete_extraction(
            config,
            journal,
            target,
            result,
            operation_id=operation_id,
            known=set(),
            mode=mode,
            model=model,
        )

    assert not target.staging_path.exists()
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "result_captured"
    )


def test_completion_refuses_another_request_s_in_memory_result(
    tmp_path: Path,
) -> None:
    root = project(tmp_path)
    config, journal, first, other_result, first_operation = _fixture_result(
        root, "table_exhaustive", "table.pdf"
    )
    complete_extraction(
        config,
        journal,
        first,
        other_result,
        operation_id=first_operation,
        known=set(),
        mode=None,
        model="claude-opus-5",
    )
    config, journal, target, _target_result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )

    with pytest.raises(operations.OperationError, match="answer does not describe"):
        complete_extraction(
            config,
            journal,
            target,
            other_result,
            operation_id=operation_id,
            known=set(),
            mode=None,
            model="claude-opus-5",
        )

    assert not target.staging_path.exists()
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .state
        == "result_captured"
    )


def test_completing_an_operation_that_is_already_over_is_refused(
    tmp_path: Path,
) -> None:
    """The other half. A terminal entry cannot account for a new staging file,
    and the refusal has to land before the write rather than after it.

    Terminal by having finished, because that is the only way `result_captured`
    leaves: an answer on disk is never an unknown outcome, so the journal has
    no move from one to the other."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )
    complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known=set(), mode=None, model="claude-opus-5",
    )
    target.staging_path.unlink()

    with pytest.raises(operations.OperationError):
        complete_extraction(
            config, journal, target, result, operation_id=operation_id,
            known=set(), mode=None, model="claude-opus-5",
        )

    assert not target.staging_path.exists()


def test_completing_an_operation_the_journal_never_authorized_is_refused(
    tmp_path: Path,
) -> None:
    """Nothing to account against, so nothing is written."""
    root = project(tmp_path)
    config, journal, target, result, _operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf"
    )

    with pytest.raises(operations.OperationError) as caught:
        complete_extraction(
            config, journal, target, result, operation_id="op-that-never-was",
            known=set(), mode=None, model="claude-opus-5",
        )

    assert "op-that-never-was" in str(caught.value)
    assert not target.staging_path.exists()


def test_the_outcome_describes_the_file_it_wrote(tmp_path: Path) -> None:
    """Every field a caller renders. The CLI prints these and W3 puts them on a
    page; a count that does not match the file is a wrong answer in both."""
    root = project(tmp_path)
    config, journal, target, result, operation_id = _fixture_result(
        root,
        "lesson_with_grammar",
        "lesson.pdf",
        # Every count is a different number on purpose: equal ones could be
        # swapped, or hardcoded, without a test noticing.
        extra=(
            # Two more rows for a word already in this answer — same
            # expression, same reading, so the same minted ID both times.
            candidate(expression="あげる", reading="あげる", meanings=["to give"]),
            candidate(expression="あげる", reading="あげる", meanings=["to hand"]),
            candidate(expression="あげる", reading="あげる", meanings=["to raise"]),
            # And four the model read no expression for, so no ID can be
            # minted from them at all.
            candidate(expression="", reading="", meanings=["unreadable"]),
            candidate(expression="  ", reading="", meanings=["unreadable"]),
            candidate(expression="", reading="およぐ", meanings=["to swim"]),
            candidate(expression=" ", reading="", meanings=["unreadable"]),
        ),
    )

    outcome = complete_extraction(
        config, journal, target, result, operation_id=operation_id,
        known={"word:もらう:もらう"}, mode=None, model="claude-opus-5",
    )

    records, meta = read_staging(target.staging_path)
    assert outcome.target == target.staging_path
    assert outcome.records == len(records) == 2
    # The concrete verdict, and that it is the one the file carries: a lesson
    # teaches a selection, so it is not judged against a unit count the way an
    # exhaustive table is.
    assert outcome.coverage_status == "selection"
    assert outcome.coverage_status == meta["coverage"]["status"]
    # Deliberately four different numbers, records included: equal ones could
    # be swapped between fields, or hardcoded, without a test noticing.
    assert outcome.already_known == 1
    assert outcome.duplicates == 3
    assert outcome.unusable == 4
    assert outcome.source == result.pattern_set.source


def test_the_failure_handler_does_not_add_a_failure_of_its_own(
    tmp_path: Path,
) -> None:
    """Somebody ending a call by hand while it is still in flight is exactly
    when this runs. Advancing an already-terminal entry would refuse, replacing
    the provider's error with a journal error and telling the person nothing
    about either."""
    root = project(tmp_path)
    config, journal, _target, _result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf", settle=False
    )
    journal.end(operation_id)

    failure = classify_dispatch_failure(
        config, journal, operation_id, JankiError("the provider timed out")
    )

    assert failure.outcome == OUTCOME_UNKNOWN
    assert failure.operation_id == operation_id


def test_a_reply_captured_but_never_recorded_is_reported_as_saved(
    tmp_path: Path,
) -> None:
    """The blob is written before the journal names it. A crash in that gap —
    or an advance refused because the call was ended meanwhile — leaves the
    answer on disk under the operation's own id, and telling somebody it never
    came back sends them to buy it a second time."""
    root = project(tmp_path)
    config, journal, _target, _result, operation_id = _fixture_result(
        root, "lesson_with_grammar", "lesson.pdf", settle=False
    )
    operations.capture_artifact(
        config.operations_file,
        operation_id,
        b'{"content": [{"type": "text", "text": "the paid answer"}]}',
    )
    assert (
        operations.OperationJournal.load(config.operations_file)
        .operations[operation_id]
        .artifact
        is None
    )

    failure = classify_dispatch_failure(
        config, journal, operation_id, JankiError("connection reset")
    )

    assert failure.outcome == ANSWER_SAVED
    assert journal.read_reply(operation_id)
