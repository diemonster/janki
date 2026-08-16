"""The single owner of the Anthropic client.

Every Anthropic-backed AI feature — ``extract`` (M3.3),
``--polish-meanings`` (M4.3), review, and batch submit/fetch (M4.4) — calls
:func:`parse_call` rather than building its own client. One owner means one
place where the model is chosen, the API key is resolved, the style guide is
cached, and the **stop reason is handed back to the caller** — the last of
which is the point of the whole module.

``parse_call`` returns a :class:`CallResult` — ``(parsed, stop_reason,
refusal)`` — and never just the parsed value. A structured-output response is
schema-valid *on normal completion*; it is not schema-valid when the model
refused or ran out of tokens, and both of those arrive as an ordinary
successful response rather than an exception. Returning a tuple makes that
impossible to forget: a caller has to name the stop reason to reach the data,
so "check ``stop_reason`` before trusting the output" is enforced by the
signature instead of by everyone remembering. What to *do* about each reason
is the caller's — M3.3 turns ``refusal`` into an ``ExtractError`` naming the
category and refuses truncated output outright — because a truncated
vocabulary table and a truncated meaning polish deserve different answers.

Immediate ``enrich --ai`` calls may instead use :mod:`codex_client`; both
providers return the same :class:`CallResult` shape.

The ``anthropic`` package is an optional dependency (``pip install -e '.[ai]'``)
and is imported lazily, so ``janki build`` works on a machine that has never
installed it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from japanese_anki import prompts
from japanese_anki.errors import JankiError

__all__ = [
    "API_KEY_ENV",
    "BATCH_ENDED",
    "COMPLETE_STOP_REASONS",
    "BatchEntry",
    "CallResult",
    "ClaudeRequestError",
    "DEFAULT_MAX_TOKENS",
    "effort_for",
    "STYLE_GUIDE_PATH",
    "Refusal",
    "batch_request",
    "batch_results",
    "batch_status",
    "build_client",
    "load_anthropic",
    "parse_call",
    "read_style_guide",
    "submit_batch",
    "system_blocks",
]

#: Where the API key comes from. Environment only — janki never reads a secret
#: from ``janki.toml`` (DESIGN_V2 "Secrets are environment-only").
API_KEY_ENV = "ANTHROPIC_API_KEY"

#: Below the SDK's non-streaming timeout ceiling. Going higher means streaming,
#: which is a caller's decision and a different call shape.
#:
#: Note this is a budget for **thinking plus response**, not response alone.
#: Every model janki sends adaptive thinking to spends part of it reasoning —
#: and that set is wider than the effort set, since Opus 4.6 and Sonnet 4.6
#: think without taking the ``xhigh`` level. A limit sized against the answer
#: alone truncates mid-thought. That truncation arrives as
#: ``stop_reason == "max_tokens"``, which is exactly what this module makes
#: callers look at.
DEFAULT_MAX_TOKENS = 16000

#: Reasoning depth for every pass. Inside ``output_config`` beside the schema,
#: not a top-level field.
#:
#: Every janki pass writes study content, so every one of them runs here:
#: extraction reads a photo and mints identities, pattern reading decides what
#: a handout teaches, enrichment writes the sentences a learner will study, and
#: meaning polish rewrites the glosses on the front of a card. There is no pass
#: whose answer is worth less than the others'.
DEFAULT_EFFORT = "xhigh"

#: Models that accept ``output_config.effort`` at :data:`DEFAULT_EFFORT`.
#:
#: **An allow-list, because the failure is asymmetric.** Sending the key to a
#: model that refuses it fails the call; withholding it from one that would
#: accept it costs some reasoning depth. The first aborts a run — enrichment's
#: call is not inside a try, so it dies on the first record — and the second is
#: merely a weaker answer. (The pass this was written for, the retired reading
#: adjudicator, swallowed the 400 into a permanent "unsure" with nothing
#: printed; M7.6V removed it, and the asymmetry still holds without it.)
#:
#: An earlier form listed the families that *reject* effort, which had the
#: direction wrong: ``xhigh`` arrived with Opus 4.7, so Opus 4.6 and Sonnet 4.6
#: take ``low``/``medium``/``high``/``max`` and refuse it, Opus 4.5 takes only
#: the first three, and Sonnet 4.5 and Haiku 4.5 refuse ``effort`` outright. A
#: deny-list naming one family sent an invalid level to five others.
#:
#: Matched on the family rather than the exact id because ids carry date
#: suffixes. An unrecognized model — a typo, an alias, one newer than this list
#: — gets no key, which is the safe direction.
_XHIGH_MODELS = (
    "opus-5",
    "opus-4-8",
    "opus-4-7",
    "sonnet-5",
    "fable-5",
    "mythos-5",
)

#: Models where adaptive thinking must be asked for by name.
#:
#: A separate list from the one above, because the two properties do not line
#: up. Opus 4.6 and Sonnet 4.6 take ``thinking`` but not the ``xhigh`` level,
#: so pairing the key with *effort* left exactly them thinking-off — the same
#: defect one family down from the one that motivated the pairing. Older
#: families are absent on purpose: they need ``budget_tokens``, which this
#: module never sends, so asking them for adaptive thinking would fail.
_ADAPTIVE_THINKING_MODELS = _XHIGH_MODELS + ("opus-4-6", "sonnet-4-6")


def _thinks_adaptively(model: str) -> bool:
    """Whether ``model`` should be sent ``thinking: {"type": "adaptive"}``."""
    name = model.strip().lower()
    return bool(name) and any(f in name for f in _ADAPTIVE_THINKING_MODELS)


def effort_for(model: str) -> str | None:
    """:data:`DEFAULT_EFFORT` when ``model`` takes it, otherwise ``None``.

    Resolved from the model rather than from the provider or the config,
    because ``--model`` changes the model alone: a guard on either of the other
    two guards something the caller did not just override — and pinning an
    older model is the usual reason to pass it.
    """
    name = model.strip().lower()
    if not name:
        return None
    return DEFAULT_EFFORT if any(family in name for family in _XHIGH_MODELS) else None

#: The style guide, relative to the project root. Every AI pass sends it, which
#: is why it is worth a cache breakpoint.
STYLE_GUIDE_PATH = prompts.DIRECTORY / "style-guide.md"

#: Stop reasons that mean the answer is whole and worth validating. Everything
#: else — ``refusal``, ``max_tokens``, ``pause_turn`` — leaves partial or absent
#: content, and is handed back to the caller instead.
#:
#: ``stop_sequence`` is deliberately absent: janki sends no stop sequences, so
#: one occurring at all would mean the response was cut at a boundary nobody
#: asked for, and a truncated answer is exactly what must not be parsed.
COMPLETE_STOP_REASONS: frozenset[str] = frozenset({"end_turn"})

#: The batch ``processing_status`` that means results can be read. Anything else
#: — ``in_progress``, ``canceling`` — means come back later.
BATCH_ENDED = "ended"


class ClaudeRequestError(JankiError):
    """The Anthropic service refused or failed a request before responding."""


@dataclass(frozen=True, slots=True)
class Refusal:
    """Why a request was declined, in janki's own vocabulary.

    The SDK's refusal object is translated here rather than handed onward, so
    that this module stays the only one that knows what an Anthropic response
    looks like. ``category`` is an open set — new ones appear without warning —
    so a caller should put it in a message rather than branch on it.
    """

    category: str
    explanation: str


class CallResult(NamedTuple):
    """What one call produced: the parsed value, why it stopped, and any refusal.

    A tuple so that unpacking still forces a caller to name ``stop_reason`` to
    reach ``parsed``, which is the discipline this module exists to impose.
    ``refusal`` is populated only when ``stop_reason == "refusal"`` — the API
    fills its details for nothing else — and is what lets a caller say *why* a
    request was declined instead of only that it was.
    """

    parsed: Any
    stop_reason: str | None
    refusal: Refusal | None


def _http_errors() -> tuple[type[BaseException], ...]:
    """Transport exceptions the SDK does not wrap, for the boundary below.

    ``httpx`` ships with the SDK, so this cannot fail where a call is possible;
    the empty tuple is for a checkout that has neither, where ``isinstance``
    against it is simply never true.
    """
    try:
        import httpx
    except ImportError:  # pragma: no cover - only without the AI extra
        return ()
    return (httpx.HTTPError,)


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


def _refusal_of(response: Any) -> Refusal | None:
    """The refusal details, if the API attached any.

    Populated only alongside ``stop_reason == "refusal"``; every other stop
    reason leaves it null, so this reads as ``None`` for them without a
    special case.
    """
    details = getattr(response, "stop_details", None)
    if details is None:
        return None
    return Refusal(
        category=str(getattr(details, "category", "") or ""),
        explanation=str(getattr(details, "explanation", "") or ""),
    )


def _type_adapter(schema: Any) -> Any:
    """A validator for ``schema``.

    ``pydantic`` is imported here rather than at module scope for the same
    reason ``anthropic`` is — it arrives with the ``ai`` extra, and every
    non-AI command has to run without it.
    """
    import pydantic

    return pydantic.TypeAdapter(schema)


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

    It lives in `prompts/` and loads like every other prompt, because that is
    what it is: text sent to a model byte for byte. It kept a named reader of
    its own only because it moved there later than the rest — every AI pass
    leads with it, so it is the one prompt with a fixed position rather than a
    pass that selects it.
    """
    return prompts.load(root, "style-guide")


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


def _request_body(
    model: str,
    system_blocks: Sequence[dict[str, Any]],
    user_content: str | Iterable[dict[str, Any]],
    schema: Any,
    max_tokens: int,
    effort: str | None = None,
) -> dict[str, Any]:
    """The Messages request both call shapes send.

    Shared so a live call and a batched one cannot drift: the whole promise of
    ``--batch-submit`` is that it is the same request at half price, and a batch
    that quietly sent a different ``max_tokens`` or lost the output format would
    return answers the synchronous path would never have produced.
    """
    anthropic = load_anthropic()
    output_config: dict[str, Any] = {
        "format": {
            "type": "json_schema",
            "schema": anthropic.transform_schema(schema),
        }
    }
    # Omitted rather than sent as None: a model that does not support effort
    # rejects the key itself, so the only safe way to not ask for it is to not
    # send it.
    if effort:
        output_config["effort"] = effort
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": list(system_blocks),
        "messages": [{"role": "user", "content": user_content}],
        "output_config": output_config,
    }
    if _thinks_adaptively(model):
        # Sent explicitly, not left to the default, because the default differs
        # across the models janki uses. Opus 5, Sonnet 5, Fable 5 and Mythos 5
        # think when `thinking` is absent; **Opus 4.8, 4.7, 4.6 and Sonnet 4.6
        # do not** — for them adaptive is the only on-mode and omitting the key
        # means no thinking at all. Keyed on the model rather than on `effort`:
        # the 4.6 pair takes thinking but not the `xhigh` level, so pairing it
        # with effort left exactly those two silently thinking-off, which is the
        # same defect one family down from the one that motivated the pairing.
        body["thinking"] = {"type": "adaptive"}
    return body


def _result_of(response: Any, schema: Any, model: str) -> CallResult:
    """A completed response as a :class:`CallResult`, stop reason checked first."""
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason not in COMPLETE_STOP_REASONS:
        # Refused, truncated, or paused: whatever text came back is not a whole
        # answer, so there is nothing worth validating and the caller decides.
        return CallResult(None, stop_reason, _refusal_of(response))

    text = next(
        (
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ),
        None,
    )
    if text is None:
        raise JankiError(
            f"{model} finished normally but returned no text to parse. "
            "Nothing was written."
        )
    try:
        return CallResult(_type_adapter(schema).validate_json(text), stop_reason, None)
    except Exception as exc:
        # Structured outputs are schema-valid on normal completion, so this is
        # an anomaly rather than an expected branch — but it still has to reach
        # the user as a janki error rather than a pydantic traceback.
        raise JankiError(
            f"{model} finished normally but its answer did not match the "
            f"expected shape: {exc}"
        ) from exc


def parse_call(
    model: str,
    system_blocks: Sequence[dict[str, Any]],
    user_content: str | Iterable[dict[str, Any]],
    schema: Any,
    client: Any | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str | None = None,
) -> CallResult:
    """One structured-output call, returning a :class:`CallResult`.

    ``schema`` is a Pydantic model describing the shape the response must take.
    ``user_content`` is a string or the content-block list a document or image
    call needs — this module does not care which, so :mod:`inputs` can build
    blocks without this one growing a media vocabulary.

    **Every field of the return matters.** ``parsed`` is ``None`` whenever the
    model did not complete normally, ``stop_reason`` says why — ``"refusal"``
    means declined, ``"max_tokens"`` means cut off mid-way — and **neither
    raises**. ``refusal`` carries the category and explanation the API attaches
    to a decline, so a caller can say *why* rather than only *that*. Deciding
    what to do is the caller's job; surfacing it is this one's.

    That last guarantee is why this builds the request itself instead of
    calling the SDK's ``messages.parse()`` helper. ``parse()`` validates every
    text block against the schema unconditionally — there is no stop-reason
    check anywhere in its path — so a truncated answer raises a
    ``pydantic`` ``ValidationError`` from inside the SDK. That is the wrong
    outcome twice over: it is not a :class:`JankiError`, so the CLI cannot
    format it, and the stop reason is not recoverable from the exception, so
    the caller cannot tell "declined" from "ran out of room" — which is the one
    thing this function exists to tell it. The schema transform is the SDK's
    (``anthropic.transform_schema``), so the wire format stays the SDK's
    business; only *when to validate* is ours.

    ``client`` is injectable so tests never touch the network (IMPLEMENTATION_PLAN
    rule 6) — the default builds a real one.
    """
    api = client if client is not None else build_client()
    try:
        # Streamed, and read whole with `get_final_message`: this module wants
        # one complete answer, not events. The reason is the budget, not the
        # events — thinking counts against `max_tokens`, so a review sized for
        # extra-high effort approaches the ceiling above which the SDK refuses
        # a non-streaming call outright, and refuses it with a bare ValueError
        # the boundary below would not convert. A stream is also never idle,
        # which is what a dropped connection at that length actually costs.
        #
        # `stream()` returns a manager, not the stream: `get_final_message`
        # exists on what `__enter__` yields. Both statements are inside the
        # try because the request is sent by the first and consumed by the
        # second, and either can fail.
        with api.messages.stream(
            **_request_body(
                model, system_blocks, user_content, schema, max_tokens, effort
            )
        ) as stream:
            response = stream.get_final_message()
    except Exception as exc:
        # The SDK's transport/status exceptions do not subclass JankiError, so
        # without this boundary the CLI prints a traceback. Keep the import
        # lazy and re-raise programming errors from an injected client.
        #
        # httpx is named separately because the SDK wraps only the initial
        # send: an error raised while iterating a stream's body arrives as a
        # bare httpx exception, which is precisely the mid-stream failure this
        # call shape makes possible.
        api_error = getattr(load_anthropic(), "APIError", ())
        if api_error and isinstance(exc, api_error):
            raise ClaudeRequestError(f"{model} request failed: {exc}") from exc
        if isinstance(exc, _http_errors()):
            raise ClaudeRequestError(
                f"{model} request failed mid-response: {exc}"
            ) from exc
        raise
    return _result_of(response, schema, model)


class BatchEntry(NamedTuple):
    """One batched request's fate.

    ``outcome`` is the API's own word — ``succeeded``, ``errored``, ``canceled``
    or ``expired`` — kept rather than flattened, because only the first of them
    carries a :class:`CallResult` and the others mean genuinely different things
    to a caller deciding whether to resubmit. ``result`` is that ``CallResult``
    and is ``None`` for the rest; ``detail`` carries whatever the API said about
    a failure.

    One value is **janki's own**, not the API's: ``invalid`` means the row
    succeeded and its text did not validate. It is separate from ``errored`` on
    purpose, and the separation is the point — an errored row has no answer
    anywhere and never will, while an invalid one's answer is complete and paid
    for and sitting on Anthropic's side, with janki's schema the only thing
    rejecting it. A caller may discard the first and must not discard the
    second.
    """

    custom_id: str
    outcome: str
    detail: str
    result: CallResult | None


def batch_request(
    custom_id: str,
    model: str,
    system_blocks: Sequence[dict[str, Any]],
    user_content: str | Iterable[dict[str, Any]],
    schema: Any,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str | None = None,
) -> dict[str, Any]:
    """One entry for :func:`submit_batch`, holding the same request as a live call.

    ``custom_id`` is what the results come back keyed by, and the API constrains
    it to a short ASCII identifier — a record id is neither — so the caller
    supplies a mapping rather than the id itself.
    """
    return {
        "custom_id": custom_id,
        "params": _request_body(
            model, system_blocks, user_content, schema, max_tokens, effort
        ),
    }


def submit_batch(requests: Sequence[dict[str, Any]], client: Any | None = None) -> str:
    """Send a batch and return its id.

    The id is the only thing worth keeping: it is what a later ``--batch-fetch``
    asks about, and losing it means paying for work nobody can collect. So an
    answer without one is an error here rather than an empty string stored in
    the ledger.
    """
    api = client if client is not None else build_client()
    batch = api.messages.batches.create(requests=list(requests))
    batch_id = str(getattr(batch, "id", "") or "")
    if not batch_id:
        raise JankiError(
            "The batch was accepted but came back without an id, so nothing can "
            "fetch its results later. Check the Anthropic console for a batch "
            "submitted just now before submitting again."
        )
    return batch_id


def batch_status(batch_id: str, client: Any | None = None) -> str:
    """The batch's ``processing_status``; compare against :data:`BATCH_ENDED`."""
    api = client if client is not None else build_client()
    batch = api.messages.batches.retrieve(batch_id)
    return str(getattr(batch, "processing_status", "") or "")


def batch_results(
    batch_id: str, schema: Any, model: str, client: Any | None = None
) -> Iterator[BatchEntry]:
    """Stream a finished batch's results, each validated the way a live call is.

    Streamed rather than collected: a batch is the shape janki reaches for when
    there are a thousand records, and holding a thousand parsed answers in
    memory to hand back a list would be a strange way to save on tokens.
    """
    api = client if client is not None else build_client()
    for entry in api.messages.batches.results(batch_id):
        custom_id = str(getattr(entry, "custom_id", "") or "")
        outcome = getattr(entry, "result", None)
        kind = str(getattr(outcome, "type", "") or "")
        if kind != "succeeded":
            # The SDK wraps the reason one level deeper than it looks: an
            # errored result carries an ``ErrorResponse`` whose own ``type`` is
            # the literal "error", and the thing worth printing — "Overloaded",
            # an invalid-request explanation — is inside its ``error``. Reading
            # the outer level yields the word "error" for every failure, which
            # tells nobody deciding whether to resubmit anything at all.
            error = getattr(outcome, "error", None)
            inner = getattr(error, "error", None) or error
            detail = str(
                getattr(inner, "message", None) or getattr(inner, "type", None) or ""
            )
            yield BatchEntry(custom_id, kind or "unknown", detail, None)
            continue
        try:
            parsed = _result_of(getattr(outcome, "message", None), schema, model)
        except JankiError as exc:
            # One row that completed normally and came back unparseable must not
            # take the batch down. A batch's results are immutable: raising here
            # would fail at the same row on every re-fetch, so a thousand good
            # answers beside one bad one would be uncollectible except by
            # forgetting the batch and throwing them all away. `parse_call`
            # keeps raising — a live call is one row, and there is nothing else
            # in it to save.
            yield BatchEntry(custom_id, "invalid", str(exc), None)
            continue
        yield BatchEntry(custom_id, kind, "", parsed)
