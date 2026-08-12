from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from japanese_anki import codex_client
from japanese_anki.errors import JankiError


class Answer(BaseModel):
    examples: list[str]
    usage_notes: str = ""


class FakeRunner:
    def __init__(
        self,
        answer: dict[str, Any] | str,
        *,
        returncode: int = 0,
        events: list[dict[str, Any]] | None = None,
        write_answer: bool | None = None,
    ) -> None:
        self.answer = answer
        self.returncode = returncode
        self.events = events if events is not None else [{"type": "turn.completed"}]
        self.write_answer = returncode == 0 if write_answer is None else write_answer
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.schema: dict[str, Any] = {}

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, kwargs))
        schema = Path(command[command.index("--output-schema") + 1])
        self.schema = json.loads(schema.read_text(encoding="utf-8"))
        if self.write_answer:
            output = Path(command[command.index("--output-last-message") + 1])
            text = self.answer if isinstance(self.answer, str) else json.dumps(self.answer)
            output.write_text(text, encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            self.returncode,
            stdout="\n".join(json.dumps(event) for event in self.events),
            stderr="model unavailable" if self.returncode else "",
        )


def test_codex_call_pins_model_effort_and_validates_the_answer() -> None:
    runner = FakeRunner({"examples": ["話します。"], "usage_notes": "Polite."})

    result = codex_client.parse_call(
        "gpt-5.6-sol",
        [{"type": "text", "text": "Write beginner Japanese."}],
        "Use 話す.",
        Answer,
        reasoning_effort="ultra",
        runner=runner,
    )

    assert result.stop_reason == "end_turn"
    assert result.parsed == Answer(examples=["話します。"], usage_notes="Polite.")
    command, kwargs = runner.calls[0]
    assert command[command.index("--model") + 1] == "gpt-5.6-sol"
    assert command[command.index("--config") + 1] == 'model_reasoning_effort="ultra"'
    assert "--ignore-user-config" in command
    assert "--json" in command
    assert "--ignore-rules" not in command
    disabled = {
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--disable"
    }
    assert {
        "shell_tool",
        "unified_exec",
        "multi_agent",
        "image_generation",
        "workspace_dependencies",
    } <= disabled
    configs = {
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--config"
    }
    assert "tools.view_image=false" in configs
    assert "tools.web_search=false" in configs
    assert "--output-schema" in command
    assert kwargs["input"].endswith("USER REQUEST\nUse 話す.")
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False
    assert runner.schema["required"] == ["examples", "usage_notes"]
    assert runner.schema["additionalProperties"] is False
    assert "default" not in runner.schema["properties"]["usage_notes"]


def test_strict_schema_strips_nested_defaults_without_dropping_a_field_named_default() -> None:
    schema = {
        "type": "object",
        "properties": {
            "default": {"type": "string", "default": "value"},
            "nested": {
                "type": "object",
                "properties": {"value": {"type": "integer", "default": 1}},
            },
        },
    }

    strict = codex_client._strict_schema(schema)

    assert "default" in strict["properties"]
    assert "default" not in strict["properties"]["default"]
    assert "default" not in strict["properties"]["nested"]["properties"]["value"]


def test_model_and_effort_are_ordinary_call_configuration() -> None:
    runner = FakeRunner({"examples": [], "usage_notes": ""})

    codex_client.parse_call(
        "future-codex-model",
        [{"text": "Guide"}],
        "Request",
        Answer,
        reasoning_effort="high",
        runner=runner,
    )

    command = runner.calls[0][0]
    assert command[command.index("--model") + 1] == "future-codex-model"
    assert command[command.index("--config") + 1] == 'model_reasoning_effort="high"'


def test_missing_codex_cli_names_the_install_and_login_remedy() -> None:
    def missing(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("codex")

    with pytest.raises(JankiError) as caught:
        codex_client.parse_call(
            "gpt-5.6-sol", [{"text": "Guide"}], "Request", Answer, runner=missing
        )

    assert "Codex CLI is not installed" in str(caught.value)
    assert "codex login" in str(caught.value)


def test_codex_failure_is_a_clean_janki_error() -> None:
    runner = FakeRunner({}, returncode=2, events=[])

    with pytest.raises(JankiError) as caught:
        codex_client.parse_call(
            "gpt-5.6-sol", [{"text": "Guide"}], "Request", Answer, runner=runner
        )

    assert "exit 2" in str(caught.value)
    assert "model unavailable" in str(caught.value)


def test_a_structured_output_refusal_uses_the_shared_call_result() -> None:
    runner = FakeRunner(
        "I cannot help with that request.",
        events=[
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "agent_message",
                    "text": "I cannot help with that request.",
                },
            },
            {"type": "turn.completed"},
        ],
    )

    result = codex_client.parse_call(
        "gpt-5.6-sol", [{"text": "Guide"}], "Request", Answer, runner=runner
    )

    assert result.parsed is None
    assert result.stop_reason == "refusal"
    assert result.refusal is not None
    assert result.refusal.category == "codex"
    assert "cannot help" in result.refusal.explanation


def test_a_completed_but_malformed_answer_is_not_mislabeled_as_a_refusal() -> None:
    runner = FakeRunner(
        '{"examples": "not a list", "usage_notes": ""}',
        events=[{"type": "turn.completed"}],
    )

    with pytest.raises(JankiError) as caught:
        codex_client.parse_call(
            "gpt-5.6-sol", [{"text": "Guide"}], "Request", Answer, runner=runner
        )

    assert "did not match the expected shape" in str(caught.value)


def test_an_incomplete_codex_turn_uses_the_shared_call_result() -> None:
    runner = FakeRunner(
        {},
        returncode=1,
        write_answer=False,
        events=[
            {
                "type": "turn.failed",
                "error": {
                    "message": "incomplete response",
                    "reason": "max_output_tokens",
                },
            }
        ],
    )

    result = codex_client.parse_call(
        "gpt-5.6-sol", [{"text": "Guide"}], "Request", Answer, runner=runner
    )

    assert result.parsed is None
    assert result.stop_reason == "max_tokens"
    assert result.refusal is None


def test_a_failed_codex_turn_can_report_a_per_record_refusal() -> None:
    runner = FakeRunner(
        {},
        returncode=1,
        write_answer=False,
        events=[
            {
                "type": "turn.failed",
                "error": {"message": "model refusal for this request"},
            }
        ],
    )

    result = codex_client.parse_call(
        "gpt-5.6-sol", [{"text": "Guide"}], "Request", Answer, runner=runner
    )

    assert result.parsed is None
    assert result.stop_reason == "refusal"
    assert result.refusal is not None
