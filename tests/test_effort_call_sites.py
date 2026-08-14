"""Every pass resolves reasoning depth from the model it actually sends.

`effort_for` deciding correctly is worth nothing if a call site resolves it
from the configured model while `--model` sends another — that is exactly how
the defect this file guards against shipped. The gating case
`model-request-effort-support` pins the helper and the body; these pin the
sites, which it cannot see.

Deleting `effort=` from any one of them used to pass the whole suite.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from japanese_anki import claude_client, enrich, extract, patterns, review
from japanese_anki.inputs import PreparedInput
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord

# A model the allow-list rejects, so "the site resolved it from this model" and
# "the site resolved it from the configured Opus" give opposite answers.
OLD = "claude-opus-4-5"

#: One it accepts, so a missing argument and a correctly-absent one differ.
NEW = "claude-opus-5"


def _record() -> VocabularyRecord:
    return VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        part_of_speech="v5s",
        examples=[ExampleSentence(japanese="毎日話します。", english="I speak daily.")],
        usage_notes="A common verb.",
        source=SourceReference(type="shirabe", imported_from="export.csv"),
    )


class _Recorder:
    """Stands in for `parse_call`, keeping the kwargs each site sent."""

    def __init__(self, result: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._result


@pytest.mark.parametrize("model,expected", [(NEW, "xhigh"), (OLD, None)])
def test_review_resolves_effort_from_the_model_it_reads_with(
    model: str, expected: str | None
) -> None:
    caller = _Recorder(claude_client.CallResult(None, "refusal", None))

    review.review_records(
        [_record()], model=model, style_guide="g", parse_call=caller
    )

    assert caller.calls, "review never called the model"
    # Both halves matter. The supported model catches the argument going
    # missing; the unsupported one catches it being sent regardless. Asserting
    # only the second passes when the site sends nothing at all.
    assert caller.calls[0].get("effort") == expected


@pytest.mark.parametrize("model,expected", [(NEW, "xhigh"), (OLD, None)])
def test_enrichment_resolves_effort_from_the_model_it_writes_with(
    model: str, expected: str | None
) -> None:
    caller = _Recorder(claude_client.CallResult(None, "refusal", None))

    enrich.enrich_ai(
        [_record()],
        model=model,
        style_guide="g",
        parse_call=caller,
        ids=["word:話す:はなす"],
        call_options={"effort": claude_client.effort_for(model)},
    )

    assert caller.calls, "enrichment never called the model"
    assert caller.calls[0].get("effort") == expected


@pytest.mark.parametrize("model,expected", [("claude-opus-5", "xhigh"), (OLD, None)])
def test_a_batch_entry_resolves_effort_from_its_own_model(
    model: str, expected: str | None
) -> None:
    """The batch sites matter most: an unsupported value there fails every row
    of a submitted batch at once, hours after anyone was watching."""
    # An unenriched record, so the pass has something to target.
    bare = replace(_record(), examples=[], usage_notes="")
    requests, _ = enrich.batch_requests([bare], model=model, style_guide="g")

    assert requests
    assert requests[0]["params"]["output_config"].get("effort") == expected


class _BodyCapture:
    """An injected SDK client that keeps the request body each pass sends.

    Checks the wire rather than the keyword, so a site that passes `effort`
    into something that drops it is caught too.
    """

    def __init__(self, text: str) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._text = text
        self.messages = self

    def stream(self, **kwargs: Any) -> Any:
        self.bodies.append(kwargs)
        return _Manager(
            SimpleNamespace(
                content=[SimpleNamespace(type="text", text=self._text)],
                stop_reason="end_turn",
                stop_details=None,
            )
        )


class _Manager:
    def __init__(self, answer: Any) -> None:
        self._answer = answer

    def __enter__(self) -> Any:
        return SimpleNamespace(get_final_message=lambda: self._answer)

    def __exit__(self, *_: Any) -> bool:
        return False


def _prepared(tmp_path: Path) -> Any:
    path = tmp_path / "lesson.pdf"
    path.write_bytes(b"%PDF-1.7 x")
    return PreparedInput(
        kind="document",
        media_type="application/pdf",
        data_b64="ZmFrZQ==",
        origin_path=path,
    )


@pytest.mark.parametrize("model,expected", [(NEW, "xhigh"), (OLD, None)])
def test_extraction_resolves_effort_from_the_model_it_reads_with(
    model: str, expected: str | None, tmp_path: Path
) -> None:
    client = _BodyCapture('{"candidates": [], "source_units": []}')

    extract.extract_candidates(
        _prepared(tmp_path), model=model, style_guide="g", client=client
    )

    assert client.bodies, "extraction never called the model"
    assert client.bodies[0]["output_config"].get("effort") == expected


@pytest.mark.parametrize("model,expected", [(NEW, "xhigh"), (OLD, None)])
def test_pattern_reading_resolves_effort_from_its_own_model(
    model: str, expected: str | None, tmp_path: Path
) -> None:
    client = _BodyCapture('{"kind": "grammar", "title": "t", "patterns": []}')

    patterns.extract_patterns(
        _prepared(tmp_path), model=model, style_guide="g", client=client
    )

    assert client.bodies, "pattern reading never called the model"
    assert client.bodies[0]["output_config"].get("effort") == expected


def test_asking_for_effort_also_asks_for_thinking() -> None:
    """The models that accept effort disagree about whether thinking is on by
    default: Opus 5 and Sonnet 5 think when the key is absent, Opus 4.8 and 4.7
    do not. Asking one of the latter for extra-high effort and then withholding
    thinking buys a fraction of what the budget was measured against, and does
    it without erroring."""
    with_effort = claude_client._request_body(
        NEW, [], "x", enrich.ai_schema(), 16000, claude_client.effort_for(NEW)
    )
    without = claude_client._request_body(
        OLD, [], "x", enrich.ai_schema(), 16000, claude_client.effort_for(OLD)
    )

    assert with_effort["thinking"] == {"type": "adaptive"}
    # And not sent otherwise: a model that takes no effort is not being asked
    # to reason harder, and older families reject the pairing differently.
    assert "thinking" not in without
