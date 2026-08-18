"""Structured-output calls through the locally authenticated Codex CLI.

The CLI owns authentication and model availability.  janki supplies the model
and reasoning effort explicitly so a user's global Codex defaults cannot make
an enrichment run use a different (and differently priced) model by accident.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from japanese_anki.claude_client import CallResult, Refusal
from japanese_anki.errors import JankiError

Runner = Callable[..., subprocess.CompletedProcess[str]]


def _type_adapter(schema: Any) -> Any:
    """Load the optional validator only when an AI command needs it."""
    try:
        import pydantic
    except ImportError as exc:  # pragma: no cover - covered through installation tests
        raise JankiError(
            "This command needs janki's AI support. Install it with: "
            "pip install -e '.[ai]'"
        ) from exc
    return pydantic.TypeAdapter(schema)


def _prompt(
    system_blocks: Sequence[Mapping[str, Any]],
    user_content: str | Iterable[Mapping[str, Any]],
) -> str:
    """Flatten janki's text-only enrichment request for ``codex exec``."""
    if not isinstance(user_content, str):
        raise JankiError(
            "The Codex enrichment provider currently accepts text prompts only."
        )
    system = "\n\n".join(
        str(block.get("text", "")).strip()
        for block in system_blocks
        if str(block.get("text", "")).strip()
    )
    if not system:
        raise JankiError("A Codex enrichment call needs non-empty system instructions.")
    return (
        "Follow the instructions below and answer the user request. Do not inspect "
        "files or call tools. Return only the JSON object required by the supplied "
        "output schema.\n\n"
        f"INSTRUCTIONS\n{system}\n\n"
        f"USER REQUEST\n{user_content}"
    )


def _strict_schema(value: Any, *, schema_node: bool = True) -> Any:
    """Make Pydantic's schema explicit enough for strict structured output.

    Pydantic expresses fields with defaults by leaving them out of ``required``.
    OpenAI strict schemas instead require every declared property; the values
    themselves may still be empty strings or arrays, which is exactly how the
    enrichment schema represents "nothing worth adding".
    """
    if isinstance(value, list):
        return [_strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    # ``default`` is annotation-only in ordinary JSON Schema, but it is outside
    # the strict Structured Outputs subset. Pydantic emits it for every field
    # with a Python default even though strict output makes every property
    # required, so retaining it makes the request itself invalid. Schema maps
    # such as ``properties`` and ``$defs`` are handled separately so a model may
    # still have a field or definition literally named ``default``.
    schema_maps = {"properties", "$defs", "definitions", "patternProperties"}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if schema_node and key == "default":
            continue
        if schema_node and key in schema_maps and isinstance(item, dict):
            result[key] = {
                name: _strict_schema(child) for name, child in item.items()
            }
        else:
            result[key] = _strict_schema(item)
    properties = result.get("properties")
    if isinstance(properties, dict):
        result["required"] = list(properties)
        result["additionalProperties"] = False
    return result


def wire_schema(schema: Any) -> Any:
    """The strict response schema written for ``codex exec``."""
    return _strict_schema(_type_adapter(schema).json_schema())


def wire_prompt(
    style_guide: str,
    task_template: str,
    user_turn: str,
) -> str:
    """The exact flattened prompt text sent through the Codex transport."""
    return _prompt(
        [{"type": "text", "text": style_guide}, {"type": "text", "text": task_template}],
        user_turn,
    )


def _failure_detail(completed: subprocess.CompletedProcess[str]) -> str:
    text = (completed.stderr or completed.stdout or "").strip()
    if not text:
        return "no diagnostic was returned"
    # Codex diagnostics can be long event streams. The tail contains the actual
    # error and never includes the prompt, which was sent on stdin.
    return text[-2000:]


def _events(text: str) -> list[Mapping[str, Any]]:
    """Best-effort parsing of the JSONL status channel from ``codex exec``."""
    events: list[Mapping[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            events.append(value)
    return events


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, Iterable):
        for item in value:
            yield from _strings(item)


def _agent_message(events: Sequence[Mapping[str, Any]]) -> str:
    for event in reversed(events):
        item = event.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _non_schema_outcome(
    events: Sequence[Mapping[str, Any]],
    *,
    fallback: str = "",
) -> CallResult | None:
    """Translate model-level failures without hiding CLI/infrastructure errors."""
    joined = (" ".join(_strings(events)) + " " + fallback).lower()
    incomplete_markers = (
        "max_output_tokens",
        "max output tokens",
        "max_tokens",
        "maximum context",
        "context window",
        "incomplete response",
    )
    if any(marker in joined for marker in incomplete_markers):
        return CallResult(None, "max_tokens", None)

    refusal_markers = (
        "refusal",
        "refused",
        "cannot help with",
        "can't help with",
        "cannot comply",
        "can't comply",
        "unable to assist with",
        "unable to comply",
    )
    if any(marker in joined for marker in refusal_markers):
        explanation = _agent_message(events) or fallback.strip() or "Codex declined"
        return CallResult(
            None,
            "refusal",
            Refusal(category="codex", explanation=explanation),
        )
    return None


def parse_call(
    model: str,
    system_blocks: Sequence[Mapping[str, Any]],
    user_content: str | Iterable[Mapping[str, Any]],
    schema: Any,
    client: Any | None = None,
    *,
    reasoning_effort: str = "ultra",
    runner: Runner = subprocess.run,
) -> CallResult:
    """Run one schema-constrained Codex call using the user's CLI login.

    ``client`` is accepted only to share the call shape used by the enrichment
    loop; Codex authentication belongs to the CLI and has no Python client.
    ``runner`` is injectable so tests never launch Codex or touch the network.
    """
    del client
    adapter = _type_adapter(schema)
    prompt = _prompt(system_blocks, user_content)

    with tempfile.TemporaryDirectory(prefix="janki-codex-") as temporary:
        workdir = Path(temporary)
        schema_path = workdir / "output-schema.json"
        answer_path = workdir / "answer.json"
        schema_path.write_text(
            json.dumps(wire_schema(schema), ensure_ascii=False),
            encoding="utf-8",
        )
        command = [
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--json",
            # Deck fields are untrusted prompt input.  A read-only sandbox stops
            # writes but still lets an agent read the host, so remove every
            # local-reading route instead of relying on the prompt's request
            # not to use tools.  Keep execpolicy rules enabled as another layer.
            "--disable",
            "shell_tool",
            "--disable",
            "unified_exec",
            "--disable",
            "multi_agent",
            "--disable",
            "apps",
            "--disable",
            "plugins",
            "--disable",
            "browser_use",
            "--disable",
            "computer_use",
            "--disable",
            "image_generation",
            "--disable",
            "skill_search",
            "--disable",
            "workspace_dependencies",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--model",
            model,
            "--config",
            f"model_reasoning_effort={json.dumps(reasoning_effort)}",
            "--config",
            "tools.view_image=false",
            "--config",
            "tools.web_search=false",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(answer_path),
            "--cd",
            str(workdir),
            "-",
        ]
        try:
            completed = runner(
                command,
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise JankiError(
                "The Codex CLI is not installed. Install it, run 'codex login', "
                "then retry janki enrich --ai."
            ) from exc
        except OSError as exc:
            raise JankiError(f"Could not start the Codex CLI: {exc}") from exc

        events = _events(completed.stdout or "")
        if completed.returncode != 0:
            detail = _failure_detail(completed)
            if outcome := _non_schema_outcome(events, fallback=detail):
                return outcome
            raise JankiError(
                f"Codex {model} failed (exit {completed.returncode}): "
                f"{detail}"
            )
        try:
            raw_answer = answer_path.read_text(encoding="utf-8")
        except OSError as exc:
            if outcome := _non_schema_outcome(events, fallback=str(exc)):
                return outcome
            raise JankiError(
                f"Codex {model} finished but did not write its structured answer: {exc}"
            ) from exc
        try:
            parsed = adapter.validate_json(raw_answer)
        except Exception as exc:
            if outcome := _non_schema_outcome(events, fallback=raw_answer):
                return outcome
            raise JankiError(
                f"Codex {model} finished but its answer did not match the expected "
                f"shape: {exc}"
            ) from exc
    return CallResult(parsed, "end_turn", None)
