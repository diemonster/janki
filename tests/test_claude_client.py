"""The single owner of the Anthropic client.

No network (IMPLEMENTATION_PLAN rule 6) and no dependency on the ``ai`` extra
being installed: the SDK is faked through ``sys.modules`` where a test needs
one, and the missing-dependency path is forced rather than assumed.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from japanese_anki.claude_client import (
    DEFAULT_MAX_TOKENS,
    STYLE_GUIDE_PATH,
    build_client,
    load_anthropic,
    parse_call,
    read_style_guide,
    system_blocks,
)
from japanese_anki.errors import JankiError


class FakeMessages:
    """Records the one call it is given and answers with a canned response."""

    def __init__(self, parsed: Any = None, stop_reason: str | None = "end_turn") -> None:
        self.parsed = parsed
        self.stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop_reason)


def fake_client(parsed: Any = None, stop_reason: str | None = "end_turn") -> Any:
    return SimpleNamespace(messages=FakeMessages(parsed, stop_reason))


SCHEMA = SimpleNamespace(__name__="Candidates")


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

    build_client()
    build_client("sk-test")

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

    assert "style guide" in str(excinfo.value)


def test_the_real_style_guide_is_where_this_module_looks_for_it() -> None:
    # A guard on the constant: the path is relative to the project root, and
    # every AI pass depends on it resolving.
    root = Path(__file__).resolve().parent.parent
    assert (root / STYLE_GUIDE_PATH).is_file()


# --- the call ----------------------------------------------------------------


def test_a_call_sends_the_model_system_blocks_and_schema() -> None:
    client = fake_client(parsed=SimpleNamespace(words=[]))
    blocks = system_blocks("style guide")

    parse_call("claude-opus-5", blocks, "Extract the words.", SCHEMA, client)

    sent = client.messages.calls[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["system"] == blocks
    assert sent["output_format"] is SCHEMA
    assert sent["max_tokens"] == DEFAULT_MAX_TOKENS
    assert sent["messages"] == [{"role": "user", "content": "Extract the words."}]


def test_content_blocks_pass_through_untouched() -> None:
    # A document or image call hands over blocks; this module has no media
    # vocabulary and must not need one.
    client = fake_client()
    content = [
        {"type": "document", "source": {"type": "base64", "data": "..."}},
        {"type": "text", "text": "Transcribe this."},
    ]

    parse_call("claude-opus-5", system_blocks("guide"), content, SCHEMA, client)

    assert client.messages.calls[0]["messages"][0]["content"] == content


def test_the_parsed_value_and_the_stop_reason_come_back_together() -> None:
    parsed = SimpleNamespace(words=["話す"])
    client = fake_client(parsed=parsed, stop_reason="end_turn")

    result, stop_reason = parse_call(
        "claude-opus-5", system_blocks("guide"), "go", SCHEMA, client
    )

    assert result is parsed
    assert stop_reason == "end_turn"


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_an_incomplete_response_is_returned_not_raised(stop_reason: str) -> None:
    # Neither is an exception from the SDK, and neither carries usable output.
    # Returning the pair is what stops a caller reading `parsed` without asking.
    client = fake_client(parsed=None, stop_reason=stop_reason)

    result, reason = parse_call(
        "claude-opus-5", system_blocks("guide"), "go", SCHEMA, client
    )

    assert result is None
    assert reason == stop_reason


def test_a_caller_can_raise_the_token_ceiling() -> None:
    client = fake_client()

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
    client = fake_client()

    parse_call("claude-opus-5", system_blocks("guide"), "go", SCHEMA, client)

    assert len(client.messages.calls) == 1
