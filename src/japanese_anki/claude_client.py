"""The single owner of the Anthropic client.

Every AI feature — ``extract`` (M3.3), ``enrich --ai`` (M4.2),
``--polish-meanings`` (M4.3), batch submit/fetch (M4.4) — calls
:func:`parse_call` rather than building its own client. One owner means one
place where the model is chosen, the API key is resolved, the style guide is
cached, and the **stop reason is handed back to the caller** — the last of
which is the point of the whole module.

``parse_call`` returns ``(parsed, stop_reason)`` and never just the parsed
value. A structured-output response is schema-valid *on normal completion*;
it is not schema-valid when the model refused or ran out of tokens, and both
of those arrive as an ordinary successful response rather than an exception.
Returning the pair makes that impossible to forget: a caller has to name the
stop reason to get at the data, so "check ``stop_reason`` before trusting the
output" is enforced by the signature instead of by everyone remembering. What
to *do* about each reason is the caller's — M3.3 turns ``refusal`` into an
``ExtractError`` and ``max_tokens`` into a per-page re-run — because a
truncated vocabulary table and a truncated meaning polish deserve different
answers.

The ``anthropic`` package is an optional dependency (``pip install -e '.[ai]'``)
and is imported lazily, so ``janki build`` works on a machine that has never
installed it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_MAX_TOKENS",
    "STYLE_GUIDE_PATH",
    "build_client",
    "load_anthropic",
    "parse_call",
    "read_style_guide",
    "system_blocks",
]

#: Where the API key comes from. Environment only — janki never reads a secret
#: from ``janki.toml`` (DESIGN_V2 "Secrets are environment-only").
API_KEY_ENV = "ANTHROPIC_API_KEY"

#: Below the SDK's non-streaming timeout ceiling. Going higher means streaming,
#: which is a caller's decision and a different call shape.
#:
#: Note this is a budget for **thinking plus response**, not response alone:
#: current models think by default, so a limit sized snugly around the expected
#: output can truncate mid-answer. That truncation arrives as
#: ``stop_reason == "max_tokens"``, which is exactly what this module makes
#: callers look at.
DEFAULT_MAX_TOKENS = 16000

#: The style guide, relative to the project root. Every AI pass sends it, which
#: is why it is worth a cache breakpoint.
STYLE_GUIDE_PATH = Path("docs") / "JAPANESE_STYLE_GUIDE.md"


def load_anthropic() -> Any:
    """The ``anthropic`` module, or a :class:`JankiError` saying how to get it.

    Imported here rather than at module scope so that every non-AI command —
    import, build, status — runs on a checkout that never installed the extra.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise JankiError(
            "This command needs the Anthropic SDK, which janki does not install "
            "by default. Install it with: pip install -e '.[ai]'"
        ) from exc
    return anthropic


def build_client(api_key: str | None = None) -> Any:
    """An ``anthropic.Anthropic``, with the key resolved the SDK's own way.

    Passing ``api_key`` is for tests and for a caller that genuinely holds a
    key; leaving it ``None`` lets the SDK resolve credentials from the
    environment, which covers ``ANTHROPIC_API_KEY`` and the other sources it
    knows about. Deliberately *not* checked here: a missing key is the SDK's
    error to raise, and pre-empting it would mean maintaining a second copy of
    its resolution order that drifts the first time it gains a source.
    """
    anthropic = load_anthropic()
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def read_style_guide(root: Path) -> str:
    """The project's Japanese style guide, for the system prompt.

    A missing guide is an error rather than an empty string: every AI pass is
    supposed to be writing to this project's conventions, and silently dropping
    them would produce plausible output that quietly ignores the rules the
    repository exists to enforce.
    """
    path = Path(root) / STYLE_GUIDE_PATH
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JankiError(
            f"Could not read the style guide at {path}: {exc}. Every AI pass sends "
            "it as system context, so janki will not run one without it."
        ) from exc


def system_blocks(
    *texts: str, cache_ttl: str | None = None
) -> list[dict[str, Any]]:
    """System text blocks with a cache breakpoint on the last one.

    Caching is a *prefix* match, so the breakpoint goes on the final block:
    everything before it — the style guide, any shared instructions — is then
    one cached prefix that repeated calls read instead of re-sending. Put the
    stable text first and anything that varies per call in the user content,
    not here; a single changed byte above the breakpoint invalidates it.

    ``cache_ttl`` selects the lifetime: the default (``None``) is the API's
    5-minute window, which suits an interactive run, and ``"1h"`` is what a
    batch pass wants so entries survive the gaps between submissions. The
    longer TTL costs more to write, so it earns its keep over more calls rather
    than fewer.

    A short prefix simply will not cache — the API has a per-model minimum and
    is silent about missing it — which is another reason the style guide leads:
    it is what makes the prefix long enough to be worth caching at all.
    """
    blocks = [{"type": "text", "text": text} for text in texts if text]
    if not blocks:
        raise JankiError("A system prompt needs at least one non-empty block")
    control: dict[str, Any] = {"type": "ephemeral"}
    if cache_ttl:
        control["ttl"] = cache_ttl
    blocks[-1]["cache_control"] = control
    return blocks


def parse_call(
    model: str,
    system_blocks: Sequence[dict[str, Any]],
    user_content: str | Iterable[dict[str, Any]],
    schema: Any,
    client: Any | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[Any, str | None]:
    """One structured-output call, returning ``(parsed, stop_reason)``.

    ``schema`` is a Pydantic model describing the shape the response must take;
    the SDK validates against it, so ``parsed`` is either an instance of that
    model or ``None``. ``user_content`` is a string or the content-block list a
    document or image call needs — this module does not care which, so
    :mod:`inputs` can build blocks without this one growing a media vocabulary.

    **Both halves of the return matter.** ``parsed`` is ``None`` whenever the
    model did not complete normally, and ``stop_reason`` says why:
    ``"refusal"`` means the request was declined, ``"max_tokens"`` means the
    answer was cut off mid-way, and neither raises. Deciding between them is
    the caller's job; surfacing them is this one's.

    ``client`` is injectable so tests never touch the network (IMPLEMENTATION_PLAN
    rule 6) — the default builds a real one.
    """
    api = client if client is not None else build_client()
    messages = [{"role": "user", "content": user_content}]
    response = api.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=list(system_blocks),
        messages=messages,
        output_format=schema,
    )
    return getattr(response, "parsed_output", None), getattr(response, "stop_reason", None)
