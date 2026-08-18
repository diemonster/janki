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

from japanese_anki import claude_client, enrich, extract
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
def test_enrichment_resolves_effort_from_the_model_it_writes_with(
    model: str, expected: str | None
) -> None:
    caller = _Recorder(claude_client.CallResult(None, "refusal", None))

    enrich.enrich_ai(
        [_record()],
        model=model,
        style_guide="g",
        instructions="I",
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
    plan = enrich.batch_requests(
        [bare], model=model, style_guide="g", instructions="I"
    )

    assert plan.requests
    assert plan.requests[0]["params"]["output_config"].get("effort") == expected


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
        _prepared(tmp_path), model=model, style_guide="g", client=client,
        system="S")

    assert client.bodies, "extraction never called the model"
    assert client.bodies[0]["output_config"].get("effort") == expected
    # Thinking travels with the model too, and per site — the gating case
    # builds the body directly, so only this reaches the call.
    assert ("thinking" in client.bodies[0]) is (expected is not None)


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
    # Not sent to models that reject it — but *is* sent to Opus 4.6 and Sonnet
    # 4.6, which take thinking and not the xhigh level. Keying the pairing on
    # effort left exactly those two silently thinking-off.
    assert "thinking" not in without
    pair = claude_client._request_body(
        "claude-opus-4-6", [], "x", enrich.ai_schema(), 16000,
        claude_client.effort_for("claude-opus-4-6"),
    )
    assert pair["output_config"].get("effort") is None
    assert pair["thinking"] == {"type": "adaptive"}


def test_refresh_skips_the_jpdb_stages_when_there_is_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keyed on the key rather than on `--no-jpdb`, and asserted with the
    environment controlled rather than inherited.

    The previous shape was pinned only when the key happened to be unset, so on
    a machine that exports it — the author's — deleting the branch entirely
    left the suite green.
    """
    from japanese_anki import cli

    monkeypatch.delenv("JPDB_API_KEY", raising=False)
    root = tmp_path / "p"
    (root / "data" / "decks").mkdir(parents=True)
    (root / "janki.toml").write_text("[paths]\n", encoding="utf-8")
    (root / "data" / "normalized").mkdir(parents=True, exist_ok=True)
    (root / "data" / "normalized" / "vocabulary.json").write_text("[]", encoding="utf-8")

    code = cli.main(
        ["--root", str(root), "refresh", "--no-audio", "--no-build"]
    )
    captured = capsys.readouterr()

    # stderr, not stdout: this skip is not something the caller asked for.
    assert "— jpdb: skipped (no JPDB_API_KEY" in captured.err
    # `--ai` is no longer one of them: M7.6V retired the sentence oracle, so the
    # AI pass writes and checks its examples without asking jpdb anything.
    #
    # Asserted as "it ran", not as "it was not skipped". The absence form is
    # satisfied by an *abort* — a run that dies at `ai` with "JPDB_API_KEY is
    # not set" also prints no "— ai: skipped" and also returns 1 — which is the
    # regression this test exists to catch and did not.
    assert "refresh: 1 stage(s) completed: ai" in captured.out
    # And non-zero, so cron cannot read "nothing was enriched" as success.
    assert code == 1



# ---------------------------------------------------------------------------
# The allow-lists themselves
# ---------------------------------------------------------------------------
#
# Ported from the gating case `model-request-effort-support` when M8.4 deleted
# the corpus. The tests above pin that each *site* resolves depth from the model
# it sends; these pin what the two allow-lists actually contain, which no site
# test can see. Measured before writing: deleting `mythos-5`, or `opus-4-8` and
# `opus-4-7`, or `sonnet-5` and `fable-5` from `_XHIGH_MODELS`, or trimming
# `sonnet-4-6` out of `_ADAPTIVE_THINKING_MODELS`, each left the whole suite
# green — so every pass would have quietly stopped asking for the depth it was
# configured for, on models nobody had a test for.

#: (model id, sent xhigh effort, sent adaptive thinking).
#: The two columns are deliberately not the same set: Opus 4.6 and Sonnet 4.6
#: think adaptively but reject the xhigh level, which is the pairing bug the
#: separate lists exist to prevent.
_DEPTH_TABLE = (
    ("claude-opus-5", True, True),
    ("claude-opus-4-8", True, True),
    ("claude-opus-4-7", True, True),
    ("claude-sonnet-5", True, True),
    ("claude-fable-5", True, True),
    ("claude-mythos-5", True, True),
    ("claude-opus-4-6", False, True),
    ("claude-sonnet-4-6", False, True),
    ("claude-opus-4-5", False, False),
    ("claude-sonnet-4-5", False, False),
    ("claude-haiku-4-5-20251001", False, False),
    ("some-unreleased-model", False, False),
)


@pytest.mark.parametrize("model,xhigh,thinks", _DEPTH_TABLE, ids=lambda v: str(v))
def test_the_request_body_asks_each_model_for_the_depth_it_accepts(
    model: str, xhigh: bool, thinks: bool
) -> None:
    """Both keys are checked for *absence*, never for a `None` value.

    A model that rejects `effort` answers the key with a 400 whatever its
    value, so `"effort": None` is not a softer version of not sending it — it
    is the same outage. `effort_for` returning `None` for such a model is
    already well pinned; what was not is which ids that covers.
    """
    body = claude_client._request_body(
        model, [], "x", enrich.ai_schema(), 100, claude_client.effort_for(model)
    )

    if xhigh:
        assert body["output_config"]["effort"] == claude_client.DEFAULT_EFFORT
    else:
        assert "effort" not in body["output_config"]
    if thinks:
        assert body["thinking"] == {"type": "adaptive"}
    else:
        assert "thinking" not in body
