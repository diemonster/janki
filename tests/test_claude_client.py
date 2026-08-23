"""The single owner of the Anthropic client.

No network (IMPLEMENTATION_PLAN rule 6): every call goes through an injected
fake client.

These tests **do** need the ``ai`` extra, which is why ``dev`` pulls it in.
The lazy-import promise they check is a *runtime* one — non-AI commands run
without the SDK — and it is verified by forcing the ImportError rather than by
the suite happening to run without the package. Skipping instead would report
green for the AI plumbing on every machine that had not installed it.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from japanese_anki import claude_client
from japanese_anki.claude_client import (
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    STYLE_GUIDE_PATH,
    ClaudeRequestError,
    batch_request,
    load_anthropic,
    parse_call,
    read_style_guide,
    system_blocks,
)
from japanese_anki.errors import JankiError


class Candidates(BaseModel):
    """A real schema — the module validates against it, so a stub will not do."""

    words: list[str]


class _Stream:
    """The manager ``messages.stream`` returns: dunder lookup is on the type,
    so this cannot be a SimpleNamespace."""

    def __init__(self, answer: Any) -> None:
        self._answer = answer

    def __enter__(self) -> Any:
        return SimpleNamespace(get_final_message=lambda: self._answer)

    def __exit__(self, *_: Any) -> bool:
        return False


class FakeMessages:
    """Records the one call it is given and answers with a canned response."""

    def __init__(
        self,
        text: str | None = None,
        stop_reason: str | None = "end_turn",
        details: Any = None,
    ) -> None:
        self.text = text
        self.stop_reason = stop_reason
        self.details = details
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        content = (
            [] if self.text is None else [SimpleNamespace(type="text", text=self.text)]
        )
        return SimpleNamespace(
            content=content, stop_reason=self.stop_reason, stop_details=self.details
        )

    def stream(self, **kwargs: Any) -> Any:
        """The shape the SDK returns: a manager whose ``__enter__`` yields the
        stream. ``get_final_message`` is on the stream, not on the manager."""
        return _Stream(self.create(**kwargs))

    def parse(self, **kwargs: Any) -> Any:  # pragma: no cover - must never run
        raise AssertionError(
            "parse_call used messages.parse(), which validates without checking "
            "stop_reason and therefore raises on a truncated answer"
        )


def fake_client(
    text: str | None = None, stop_reason: str | None = "end_turn", details: Any = None
) -> Any:
    return SimpleNamespace(messages=FakeMessages(text, stop_reason, details))


SCHEMA = Candidates


# --- the optional dependency -------------------------------------------------


def test_a_missing_sdk_names_the_extra_that_installs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Forced rather than assumed: this must fail the same way on a machine that
    # *has* installed the extra.
    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    with pytest.raises(JankiError) as excinfo:
        load_anthropic()

    assert "pip install -e '.[ai]'" in str(excinfo.value)


def test_importing_the_module_does_not_import_the_sdk() -> None:
    # The whole point of the lazy import: `janki build` must work on a checkout
    # that never installed the extra. Re-importing under a forced ImportError
    # proves the import is not at module scope.
    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise AssertionError("claude_client imported anthropic at module scope")
        return real_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "__import__", refuse)
        patch.delitem(sys.modules, "japanese_anki.claude_client", raising=False)
        import japanese_anki.claude_client as module

        assert module.DEFAULT_MAX_TOKENS == DEFAULT_MAX_TOKENS


@pytest.mark.allow_build_client
def test_build_client_leaves_key_resolution_to_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No api_key means no api_key argument — the SDK's own resolution order
    # covers more sources than janki should duplicate.
    seen: list[dict[str, Any]] = []
    fake_sdk = SimpleNamespace(
        Anthropic=lambda **kwargs: seen.append(kwargs) or SimpleNamespace(**kwargs)
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_sdk)

    claude_client.build_client()
    claude_client.build_client("sk-test")

    assert seen == [{}, {"api_key": "sk-test"}]


# --- system blocks and caching ----------------------------------------------


def test_the_cache_breakpoint_lands_on_the_last_block() -> None:
    # Caching is a prefix match, so the breakpoint has to be last or the blocks
    # before it are the only thing cached.
    blocks = system_blocks("style guide", "task instructions")

    assert [block["text"] for block in blocks] == ["style guide", "task instructions"]
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}


def test_a_batch_caller_can_ask_for_the_longer_ttl() -> None:
    blocks = system_blocks("style guide", cache_ttl="1h")

    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_empty_blocks_are_dropped_and_an_empty_prompt_is_an_error() -> None:
    assert len(system_blocks("guide", "", "task")) == 2
    with pytest.raises(JankiError):
        system_blocks("", "")


# --- the style guide ---------------------------------------------------------


def test_the_style_guide_is_read_from_the_project_root(tmp_path: Path) -> None:
    guide = tmp_path / STYLE_GUIDE_PATH
    guide.parent.mkdir(parents=True)
    guide.write_text("Prefer natural English.", encoding="utf-8")

    assert read_style_guide(tmp_path) == "Prefer natural English."


def test_a_missing_style_guide_stops_the_pass_rather_than_sending_nothing(
    tmp_path: Path,
) -> None:
    # Silently dropping it would produce plausible output that ignores every
    # convention this repository exists to enforce.
    with pytest.raises(JankiError) as excinfo:
        read_style_guide(tmp_path)

    # Names the file, which is now `prompts/style-guide.md` — the guide loads
    # through the prompt loader because that is what it is: text sent to a
    # model byte for byte.
    assert "prompts/style-guide.md" in str(excinfo.value)
    assert "will not run a pass without it" in str(excinfo.value)


def test_the_real_style_guide_is_where_this_module_looks_for_it() -> None:
    # A guard on the constant: the path is relative to the project root, and
    # every AI pass depends on it resolving.
    root = Path(__file__).resolve().parent.parent
    assert (root / STYLE_GUIDE_PATH).is_file()


# --- the call ----------------------------------------------------------------


VALID = '{"words": ["\u8a71\u3059"]}'
TRUNCATED = '{"words": ["\u8a71\u3059", "\u98df\u3079'


def test_a_call_sends_the_model_system_blocks_and_schema() -> None:
    client = fake_client(VALID)
    blocks = system_blocks("style guide")

    parse_call("claude-opus-5", blocks, "Extract the words.", SCHEMA, client)

    sent = client.messages.calls[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["system"] == blocks
    assert sent["max_tokens"] == DEFAULT_MAX_TOKENS
    assert sent["messages"] == [{"role": "user", "content": "Extract the words."}]
    # The schema goes as the strict JSON-schema form the API takes, transformed
    # by the SDK — the wire shape stays the SDK's business.
    schema = sent["output_config"]["format"]
    assert schema["type"] == "json_schema"
    assert schema["schema"]["additionalProperties"] is False
    assert schema["schema"]["required"] == ["words"]


def test_content_blocks_pass_through_untouched() -> None:
    # A document or image call hands over blocks; this module has no media
    # vocabulary and must not need one.
    client = fake_client(VALID)
    content = [
        {"type": "document", "source": {"type": "base64", "data": "..."}},
        {"type": "text", "text": "Transcribe this."},
    ]

    parse_call("claude-opus-5", system_blocks("guide"), content, SCHEMA, client)

    assert client.messages.calls[0]["messages"][0]["content"] == content


def test_a_complete_answer_is_validated_into_the_schema() -> None:
    client = fake_client(VALID)

    result, stop_reason, _refusal = parse_call(
        "claude-opus-5", system_blocks("guide"), "go", SCHEMA, client
    )

    assert isinstance(result, Candidates)
    assert result.words == ["\u8a71\u3059"]
    assert stop_reason == "end_turn"


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens", "pause_turn"])
def test_an_incomplete_response_is_returned_not_raised(stop_reason: str) -> None:
    # The whole point of the module. A truncated answer is *invalid JSON*, so
    # anything that validates before checking the stop reason raises here —
    # which is neither a JankiError nor recoverable into a stop reason.
    client = fake_client(TRUNCATED, stop_reason=stop_reason)

    result, reason, _refusal = parse_call(
        "claude-opus-5", system_blocks("guide"), "go", SCHEMA, client
    )

    assert result is None
    assert reason == stop_reason


def test_a_refusal_with_no_content_at_all_is_also_just_returned() -> None:
    client = fake_client(None, stop_reason="refusal")

    assert parse_call("claude-opus-5", system_blocks("g"), "go", SCHEMA, client) == (
        None,
        "refusal",
        None,
    )


def test_a_complete_answer_that_does_not_match_is_a_janki_error() -> None:
    # The provider enforces its transformed wire subset. The local Pydantic
    # contract can be stronger, and that mismatch still has to reach the user
    # as a formatted fail-closed error rather than a pydantic traceback.
    client = fake_client('{"words": "not a list"}')

    with pytest.raises(JankiError) as excinfo:
        parse_call("claude-opus-5", system_blocks("guide"), "go", SCHEMA, client)

    assert "did not match" in str(excinfo.value)


def test_a_complete_answer_with_no_text_block_is_a_janki_error() -> None:
    client = fake_client(None, stop_reason="end_turn")

    with pytest.raises(JankiError) as excinfo:
        parse_call("claude-opus-5", system_blocks("guide"), "go", SCHEMA, client)

    assert "no text" in str(excinfo.value)


def test_a_caller_can_raise_the_token_ceiling() -> None:
    client = fake_client(VALID)

    parse_call(
        "claude-opus-5", system_blocks("guide"), "go", SCHEMA, client, max_tokens=64000
    )

    assert client.messages.calls[0]["max_tokens"] == 64000


def test_an_injected_client_is_the_one_used(monkeypatch: pytest.MonkeyPatch) -> None:
    # The guard that keeps these tests off the network: with a client passed in,
    # nothing may build a real one.
    def explode() -> Any:
        raise AssertionError("parse_call built a client despite being given one")

    monkeypatch.setattr("japanese_anki.claude_client.build_client", explode)
    client = fake_client(VALID)

    parse_call("claude-opus-5", system_blocks("guide"), "go", SCHEMA, client)

    assert len(client.messages.calls) == 1


def test_an_sdk_request_error_becomes_a_clean_janki_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAPIError(Exception):
        pass

    class BrokenMessages:
        def create(self, **kwargs: Any) -> Any:
            raise FakeAPIError("output blocked by content filtering policy")

        def stream(self, **kwargs: Any) -> Any:
            raise FakeAPIError("output blocked by content filtering policy")

    monkeypatch.setattr(
        claude_client,
        "load_anthropic",
        lambda: SimpleNamespace(
            APIError=FakeAPIError,
            transform_schema=lambda schema: {"type": "object"},
        ),
    )
    client = SimpleNamespace(messages=BrokenMessages())

    with pytest.raises(ClaudeRequestError) as excinfo:
        parse_call("claude-opus-5", system_blocks("guide"), "go", SCHEMA, client)

    assert "output blocked by content filtering policy" in str(excinfo.value)


def test_the_sdk_helper_really_does_raise_on_a_truncated_answer() -> None:
    """Why this module does not call ``messages.parse()``.

    Run against the real SDK rather than a fake, because the whole point is
    that the fake cannot tell you this. If the SDK ever grows a stop-reason
    guard, this fails and the hand-rolled request in ``parse_call`` can go.
    """
    pytest.importorskip("anthropic", reason="the ai extra is not installed")
    from anthropic.lib._parse._response import parse_response
    from anthropic.types.message import Message

    truncated = Message.construct(
        id="msg_1",
        model="claude-opus-5",
        role="assistant",
        type="message",
        stop_reason="max_tokens",
        stop_sequence=None,
        usage={"input_tokens": 10, "output_tokens": 16000},
        content=[{"type": "text", "text": TRUNCATED}],
    )

    with pytest.raises(Exception) as excinfo:
        parse_response(output_format=Candidates, response=truncated)

    # Not a JankiError, so cli.main could not format it; and the stop reason is
    # not recoverable from it, so a caller could not tell refused from truncated.
    assert not isinstance(excinfo.value, JankiError)


def test_a_refusal_carries_the_category_a_caller_can_report() -> None:
    # M3.3's contract is "fail with the category in the message", so a bare
    # stop reason is not enough — the caller has to be able to say why.
    client = fake_client(
        None,
        stop_reason="refusal",
        details=SimpleNamespace(
            type="refusal", category="cyber", explanation="declined by policy"
        ),
    )

    result, stop_reason, refusal = parse_call(
        "claude-opus-5", system_blocks("guide"), "go", SCHEMA, client
    )

    assert result is None
    assert stop_reason == "refusal"
    assert refusal is not None
    assert refusal.category == "cyber"
    assert refusal.explanation == "declined by policy"


def test_a_normal_completion_carries_no_refusal() -> None:
    # The API fills stop_details for a refusal and nothing else, so this reads
    # as None without a special case.
    client = fake_client(VALID)

    assert parse_call(
        "claude-opus-5", system_blocks("guide"), "go", SCHEMA, client
    ).refusal is None


def test_effort_is_sent_when_a_caller_asks_for_it() -> None:
    client = fake_client(VALID)

    parse_call("m", system_blocks("g"), "hi", SCHEMA, client, effort=DEFAULT_EFFORT)

    assert client.messages.calls[0]["output_config"]["effort"] == "xhigh"


def test_effort_is_absent_when_a_caller_does_not_ask() -> None:
    """The adjudicator once ran on Haiku, which rejects the key with a 400 — and it
    catches every exception and returns "unsure", so an unconditional value
    would disable that pass permanently with nothing printed. The key is
    omitted rather than sent as None: a model that does not support effort
    rejects the key itself."""
    client = fake_client(VALID)

    parse_call("m", system_blocks("g"), "hi", SCHEMA, client)

    assert "effort" not in client.messages.calls[0]["output_config"]


def test_a_batch_entry_carries_the_same_effort_a_live_call_does() -> None:
    client = fake_client(VALID)
    blocks = system_blocks("g")

    parse_call("m", blocks, "hi", SCHEMA, client, effort=DEFAULT_EFFORT)
    batched = batch_request("r1", "m", blocks, "hi", SCHEMA, effort=DEFAULT_EFFORT)

    assert batched["params"] == client.messages.calls[0]


def test_the_output_ceiling_leaves_room_for_thinking_and_an_answer() -> None:
    """A regression with a receipt. At 16000, a real `janki extract` run on a
    two-page vocabulary chart spent the entire budget on extended thinking and
    returned `stop_reason: max_tokens` with a thinking block and no answer —
    a paid call that produced nothing.

    Thinking counts against `max_tokens`, and every call here runs at
    `DEFAULT_EFFORT`. The ceiling has to hold both. Claude Opus 5 accepts up to
    128000, and this module always streams, which is what makes a large value
    safe to send.
    """
    assert DEFAULT_MAX_TOKENS >= 64000
    assert DEFAULT_MAX_TOKENS <= 128000


def test_every_call_streams_so_a_large_ceiling_is_safe() -> None:
    """The SDK requires streaming for large `max_tokens`. If a non-streaming
    call ever appears here, the ceiling above stops being safe to send."""
    source = Path(claude_client.__file__).read_text(encoding="utf-8")
    assert "messages.stream(" in source
    assert "get_final_message()" in source
    assert "messages.create(" not in source
