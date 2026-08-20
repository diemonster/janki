"""The M7.6P boundary: two model-call paths, each returning a complete card.

These tests lead the implementation deliberately.  A source is read once by
``extract`` and a bare record is written once by ``enrich --ai``; meaning
polish and grammar-pattern discovery are parts of those answers, not later
paid passes.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, get_args

import pytest

from japanese_anki import cli, enrich, extract, patterns, prompts
from japanese_anki import coverage as coverage_module
from japanese_anki.ledger import Ledger
from japanese_anki.models import SourceReference, VocabularyRecord

REPO_ROOT = Path(__file__).resolve().parents[1]


def _list_item(model: Any, field: str) -> Any:
    """The Pydantic model inside one ``list[Model]`` field."""
    return get_args(model.model_fields[field].annotation)[0]


def _schema_descriptions(schema: dict[str, Any]) -> dict[str, str]:
    """Property path to every instruction-bearing schema description."""
    found: dict[str, str] = {}

    def visit(value: Any, path: str = "") -> None:
        if not isinstance(value, dict):
            return
        for name, field_schema in (value.get("properties") or {}).items():
            field_path = f"{path}.{name}" if path else name
            if "description" in field_schema:
                found[field_path] = field_schema["description"]
            visit(field_schema, field_path)
        for name, definition in (value.get("$defs") or {}).items():
            visit(definition, f"$defs.{name}")
        visit(value.get("items"), path + "[]")

    visit(schema)
    return found


def _rich_source_answer() -> dict[str, Any]:
    return {
        "candidates": [
            {
                "expression": "話す",
                "reading": "はなす",
                "meanings": ["to speak", "to talk"],
                "part_of_speech": "verb",
                "examples": [
                    {
                        "japanese": "先生と話します。",
                        "speech_level": "polite",
                        "furigana": "先生[せんせい]と 話[はな]します。",
                        "romaji": "sensei to hanashimasu.",
                        "english": "I speak with my teacher.",
                    },
                    {
                        "japanese": "あとで話そう。",
                        "speech_level": "casual",
                        "furigana": "あとで 話[はな]そう。",
                        "romaji": "ato de hanasou.",
                        "english": "Let's talk later.",
                    },
                ],
                "usage_notes": "Often takes と for the person spoken with.",
                "page": 1,
                "context": "先生と話します。",
                "confidence": "high",
                "source_kind": "prose",
            }
        ],
        "source_units": [],
        "model_reported_unit_count": 0,
        "document_kind": "lesson",
        "document_title": "Conversation practice",
        "patterns": [
            {
                "template": "〜と話す",
                "gloss": "speak with someone",
                "examples": ["先生と話します。"],
                "where": "page 1",
            }
        ],
    }


def test_one_source_answer_builds_a_complete_reviewable_card(tmp_path: Path) -> None:
    """No second enrichment call is needed after a rich extraction answer.

    The examples are still model proposals: putting them in reviewable staging
    must not fabricate the human acceptance stamp that promotion binds later.
    """
    parsed = extract.candidate_schema().model_validate(_rich_source_answer())
    result = extract.normalize_response(parsed, None, "lesson.pdf")
    prepared = type(
        "Prepared",
        (),
        {"origin_path": tmp_path / "lesson.pdf"},
    )()

    [record] = extract.build_records(result.candidates, prepared).records

    assert record.meanings == ["to speak", "to talk"]
    assert record.usage_notes == "Often takes と for the person spoken with."
    assert [item.register for item in record.examples] == ["polite", "casual"]
    assert record.examples[0].english == "I speak with my teacher."
    assert record.examples[0].furigana == "先生[せんせい]と 話[はな]します。"
    assert record.examples[0].romaji == "sensei to hanashimasu."
    assert Ledger.missing_enrichment([record]) == []
    assert "example_authority" not in record.source.raw_fields

    assert result.pattern_set.source == "lesson.pdf"
    assert result.pattern_set.kind == "lesson"
    assert result.pattern_set.reviewed is False
    assert [item.template for item in result.pattern_set.patterns] == ["〜と話す"]


def test_the_shared_rich_answer_has_one_application_semantics(tmp_path: Path) -> None:
    """Source/bare and known/fresh routing must not rewrite one answer differently."""
    raw = _rich_source_answer()
    candidate_raw = raw["candidates"][0]
    candidate_raw["meanings"] = ["to speak", "to speak"]
    candidate_raw["examples"][0]["furigana"] = (
        "先週[せんしゅう]、家族[かぞく]と 話[はな]します。"
    )
    candidate_raw["examples"][0]["romaji"] = "definitely wrong"
    parsed = extract.candidate_schema().model_validate(raw)
    candidate = parsed.candidates[0]
    prepared = type("Prepared", (), {"origin_path": tmp_path / "lesson.pdf"})()

    [fresh] = extract.build_records([candidate], prepared).records
    [known] = extract.build_records(
        [candidate], prepared, known_ids=["word:話す:はなす"]
    ).records
    bare_input = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        source=SourceReference(type="shirabe", imported_from="words.csv"),
    )
    bare = enrich.apply_ai_result(bare_input, candidate).record

    expected_furigana = candidate_raw["examples"][0]["furigana"]
    for item in (fresh, known, bare):
        assert item.meanings == ["to speak"]
        assert item.examples[0].furigana == expected_furigana
        assert item.examples[0].romaji != "definitely wrong"
        assert item.usage_notes == "Often takes と for the person spoken with."
    assert fresh.examples == known.examples == bare.examples


def test_source_and_bare_word_answers_share_the_rich_card_shape() -> None:
    """One card contract, reached through two input shapes.

    Keeping two independently described example schemas would let source cards
    and CSV cards drift back into different products.  The singular extraction
    ``example`` field is the superseded source-evidence-only path.
    """
    source_candidate = _list_item(extract.candidate_schema(), "candidates")
    bare = enrich.ai_schema()

    assert "example" not in source_candidate.model_fields
    assert "examples" in source_candidate.model_fields
    assert "usage_notes" in source_candidate.model_fields
    assert "meanings" in bare.model_fields

    source_example = _list_item(source_candidate, "examples")
    bare_example = _list_item(bare, "examples")
    assert source_example.model_json_schema() == bare_example.model_json_schema()


def test_the_shared_card_schema_allows_only_the_two_card_speech_levels() -> None:
    """The renderer has polite and casual slots, not an arbitrary register slot."""
    example_schema = _list_item(enrich.ai_schema(), "examples").model_json_schema()
    assert "speech_level" in example_schema["required"]
    with pytest.raises(ValueError):
        enrich.ai_schema().model_validate(
            {
                "examples": [
                    {
                        "japanese": "候ふ。",
                        "speech_level": "literary",
                    }
                ]
            }
        )


def test_response_schema_descriptions_are_only_the_terse_structural_labels() -> None:
    """Task policy belongs in Markdown, so changing a label is a visible contract."""
    assert _schema_descriptions(enrich.ai_schema().model_json_schema()) == {
        "meanings": "English glosses.",
        "usage_notes": "Usage note.",
        "$defs.GeneratedExample.japanese": "Japanese sentence.",
        "$defs.GeneratedExample.speech_level": "Speech level.",
        "$defs.GeneratedExample.furigana": "Sentence with Anki furigana.",
        "$defs.GeneratedExample.romaji": "Sentence in Hepburn romaji.",
        "$defs.GeneratedExample.english": "Natural English translation.",
    }
    assert _schema_descriptions(coverage_module.verdict_schema().model_json_schema()) == {
        "approved": "Coverage approval.",
        "reason": "Verdict rationale.",
    }
    assert _schema_descriptions(extract.candidate_schema().model_json_schema()) == {
        "model_reported_unit_count": "Reported source-unit count.",
        "document_kind": "Document kind.",
        "document_title": "Document title.",
        "$defs.CandidateRecord.meanings": "English glosses.",
        "$defs.CandidateRecord.usage_notes": "Usage note.",
        "$defs.CandidateRecord.expression": "The word as written, in Japanese.",
        "$defs.CandidateRecord.reading": "Kana reading.",
        "$defs.CandidateRecord.part_of_speech": "Part of speech, if known.",
        "$defs.CandidateRecord.page": "1-indexed page this was read from; 0 when unknown.",
        "$defs.CandidateRecord.context": "The line or cell this was read from, verbatim.",
        "$defs.CandidateRecord.confidence": "Confidence.",
        "$defs.CandidateRecord.inclusion_reason": (
            "In prose mode, why this word is worth a card."
        ),
        "$defs.CandidateRecord.source_kind": (
            "Whether this candidate comes from a table/list source unit or from "
            "prose selection."
        ),
        "$defs.CandidateRecord.section": (
            "Stable lowercase section slug. Required for a table candidate."
        ),
        "$defs.CandidateRecord.ordinal": (
            "One-based row ordinal within the section. Required for a table candidate."
        ),
        "$defs.GeneratedExample.japanese": "Japanese sentence.",
        "$defs.GeneratedExample.speech_level": "Speech level.",
        "$defs.GeneratedExample.furigana": "Sentence with Anki furigana.",
        "$defs.GeneratedExample.romaji": "Sentence in Hepburn romaji.",
        "$defs.GeneratedExample.english": "Natural English translation.",
        "$defs.SourcePattern.template": "Pattern template.",
        "$defs.SourcePattern.gloss": "English gloss.",
        "$defs.SourcePattern.examples": "Source examples.",
        "$defs.SourcePattern.where": "Source location.",
        "$defs.SourceUnitRecord.page": "One-based page number.",
        "$defs.SourceUnitRecord.section": (
            "Stable lowercase slug for the table or list section."
        ),
        "$defs.SourceUnitRecord.ordinal": "One-based row ordinal within the section.",
        "$defs.SourceUnitRecord.context": (
            "The complete source row or cell text, verbatim."
        ),
        "$defs.SourceUnitRecord.reason": "Disposition reason.",
    }


@pytest.mark.parametrize("mode", [None, "table", "prose"], ids=["auto", "table", "prose"])
def test_every_source_shape_asks_for_the_complete_card(mode: str | None) -> None:
    text = " ".join(prompts.load(REPO_ROOT, extract.prompt_name(mode)).lower().split())

    for clause in (
        "natural english",
        "few senses this candidate actually carries",
        "remove restatements, part-of-speech labels, and dictionary hedges",
        "contextual reading",
        "base contains exactly the characters that reading spells",
        "speech_level",
        "speech_level is only “polite” or “casual”",
        "polite example is everyday japanese using 〜ます or 〜です",
        "casual example is everyday casual plain-form japanese",
        "casual speech uses its own natural particles and sentence-final forms",
        "formal, literary, or otherwise outside those two slots",
        "leave it verbatim in context",
        "usage note",
        "patterns it teaches",
    ):
        assert clause in text, f"{extract.prompt_name(mode)} is missing {clause!r}"


def test_the_bare_word_template_asks_for_glosses_sentences_and_usage() -> None:
    text = prompts.load(REPO_ROOT, "enrich-bare-word").lower()

    assert "english gloss" in text
    assert "polite and casual example slots" in text
    assert "usage note" in text
    assert "reviewed lesson patterns" in text


def test_the_bare_word_template_completes_only_unoccupied_example_slots() -> None:
    """Pinned curation is preserved while the other card slot can still fill."""
    text = " ".join(
        prompts.load(REPO_ROOT, "enrich-bare-word").lower().split()
    )

    assert "return every pinned japanese string exactly" in text
    assert "only for a polite or casual slot that is not already occupied" in text


def test_the_shared_style_guide_agrees_that_rich_cards_have_two_examples() -> None:
    """The common block must not contradict every shape-specific task template."""
    text = " ".join(prompts.load(REPO_ROOT, "style-guide").lower().split())

    assert "two natural example sentences and translations" in text
    assert "one natural example sentence" not in text


def test_python_user_turns_are_labelled_data_not_hidden_instructions() -> None:
    record = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        source=SourceReference(type="shirabe", imported_from="words.csv"),
    )
    source_turn = extract.prompt_for("lesson.pdf", ["既知"])
    word_turn = enrich.ai_prompt(
        record,
        recent=["毎日話します。"],
        taught="Reviewed lesson patterns:\n* 〜んだ — explanation",
    )
    pattern_data = patterns.format_patterns([patterns.Pattern("〜んだ", "explanation")])

    for text in (source_turn, word_turn, pattern_data):
        lowered = text.lower()
        assert "do not" not in lowered
        assert "never " not in lowered
        assert "prefer " not in lowered
        assert "write " not in lowered
        assert "return " not in lowered
        assert "skip " not in lowered
    assert "Source file:" in source_turn
    assert "Known expressions:" in source_turn
    assert "Existing curated examples requiring annotations:" in word_turn
    assert "Recent examples from this run:" in word_turn
    assert pattern_data.startswith("Reviewed lesson patterns:")


def test_only_the_two_consolidated_model_call_paths_remain() -> None:
    on_disk = {path.stem for path in (REPO_ROOT / "prompts").glob("*.md")}

    assert "polish-meanings" not in on_disk
    assert "patterns" not in on_disk
    assert "enrich-bare-word" in on_disk

    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["enrich", "--polish-meanings"])
    with pytest.raises(SystemExit):
        parser.parse_args(["patterns", "handout.pdf"])
    with pytest.raises(SystemExit):
        parser.parse_args(["patterns", "--force"])


def test_request_fingerprint_covers_each_prompt_channel_and_schema() -> None:
    schema = enrich.ai_schema()
    def request_fp(style: str, task: str, user: str, response_schema: Any = schema) -> str:
        return prompts.request_fingerprint(
            provider="anthropic",
            style_guide=style,
            task_template=task,
            user_turn=user,
            transport_prompt={"system": [style, task], "user": user},
            schema=response_schema,
        )

    base = request_fp("style\r\n", "task", "record")

    assert base != request_fp("style\n", "task", "record")
    assert base != request_fp("style\r\n", "task!", "record")
    assert base != request_fp("style\r\n", "task", "record!")
    # Named JSON channels, rather than ambiguous string concatenation.
    assert request_fp("ab", "c", "") != request_fp("a", "bc", "")
    changed_schema = copy.deepcopy(schema.model_json_schema())
    changed_schema["properties"]["usage_notes"]["description"] = "Changed label."
    assert base != request_fp("style\r\n", "task", "record", changed_schema)
