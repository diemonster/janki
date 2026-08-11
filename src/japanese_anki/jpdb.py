"""The jpdb.io API client — the one place janki talks to jpdb.

Three facts about this API decide the shape of everything here.

**Responses are column-oriented.** A request names the ``fields`` it wants and
the response answers with *positional arrays* in that order — ``[1000001,
"日本語", 42]``, not ``{"vid": ..., "spelling": ...}``. Every method in this
module zips those rows into dicts before returning them, so **no caller ever
sees a positional row**. A caller that indexes a row by number is one field-list
edit away from writing a frequency rank into a spelling, silently.

**Words are addressed by ``[vid, sid]`` pairs.** A vid is the dictionary word,
a sid the particular spelling of it; neither alone identifies anything. Pairs
go in, pairs come back, and :func:`as_pair` accepts every shape a caller might
hold one in.

**The API misspells "occurrences" as ``occurences``.** The request parameter is
``fetch_occurences`` and the response key is ``occurences`` — one 'r'. Both
spellings are load-bearing wire format: correcting either one silently stops
the parameter being read and the key being found. They are marked at every
occurrence below; do not "fix" them.

Everything is JSON-over-POST. The transport is injectable
(``(url, json_body, headers) -> (status, json)``) so tests drive canned
responses with no network — see ``tests/test_jpdb_client.py``.

This module also owns two tables the rest of the project imports rather than
rebuilds: the JMDict part-of-speech mapping (:func:`pos_to_part_of_speech`,
:func:`pos_to_verb_group`, :func:`pos_to_transitivity`) and the ``/parse``
request/response contract (:func:`furigana_to_anki`,
:func:`forced_furigana_span`).
"""

from __future__ import annotations

import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from japanese_anki.errors import JankiError

DEFAULT_BASE_URL = "https://jpdb.io/api/v1/"

# Secrets are never read from janki.toml (see config.py): the key comes from
# the environment or it does not come at all.
API_KEY_ENV = "JPDB_API_KEY"

DEFAULT_TIMEOUT = 30.0

# Retry budget. The limits are undocumented, so this is deliberately modest:
# five tries spanning a few seconds, not a client that hammers a rate limiter.
DEFAULT_MAX_TRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0

# jpdb's rate limits are undocumented; batch and ask for as little as possible.
# 200 pairs per lookup keeps a 2,000-word mining deck at ten requests.
DEFAULT_BATCH_SIZE = 200

#: Deck fields ``list_user_decks`` asks for by default (DESIGN_V2, deck sync).
DEFAULT_DECK_FIELDS: tuple[str, ...] = ("id", "name", "word_count")

#: Vocabulary fields ``lookup_vocabulary`` asks for by default (DESIGN_V2 step
#: 3). ``vid``/``sid`` are deliberately absent: the client already knows the
#: pair it asked about and attaches it to every returned entry, so requesting
#: them back would spend response width on data we hold.
DEFAULT_LOOKUP_FIELDS: tuple[str, ...] = (
    "spelling",
    "reading",
    "frequency_rank",
    "pitch_accent",
    "meanings_chunks",
    "meanings_part_of_speech",
    "part_of_speech",
    "card_state",
)

#: ``/parse`` token fields. Pinned by IMPLEMENTATION_PLAN M2.1 because M2.6 and
#: M4.1 code against this exact contract.
DEFAULT_TOKEN_FIELDS: tuple[str, ...] = ("vocabulary_index", "furigana")

#: ``/parse`` vocabulary fields. Pinned by IMPLEMENTATION_PLAN M2.1.
DEFAULT_VOCABULARY_FIELDS: tuple[str, ...] = (
    "vid",
    "sid",
    "spelling",
    "reading",
    "pitch_accent",
    "frequency_rank",
    "part_of_speech",
)

# Token fields whose values are byte/char offsets into the request text. Their
# presence is what makes `position_length_encoding` meaningful — see `parse`.
_POSITIONAL_TOKEN_FIELDS = frozenset({"position", "length"})

#: Accepted ``position_length_encoding`` values, mapped to a length function.
#: jpdb interprets every position and length in the request *and* the response
#: under this one setting, so the client sends it whenever either side involves
#: offsets.
_ENCODING_LENGTHS: dict[str, Callable[[str], int]] = {
    # UTF-16 code units: astral-plane kanji (𠮟, 𩸽 — ordinary Japanese words)
    # count as two. This is the encoding JavaScript, and therefore most of the
    # tooling around this API, measures strings in.
    "utf16": lambda text: sum(2 if ord(char) > 0xFFFF else 1 for char in text),
    "utf8": lambda text: len(text.encode("utf-8")),
    # UTF-32 code units are Python's own string indices.
    "utf32": len,
}

DEFAULT_ENCODING = "utf16"

# HTTP statuses and API error ids worth trying again. 429 is the documented
# rate-limit status; `api_unavailable` is jpdb's id for "down right now".
# Everything else (a bad key, an unknown deck) fails the same way on every try,
# so retrying it only delays the error the user needs to read.
RETRYABLE_STATUSES = frozenset({429})
RETRYABLE_ERRORS = frozenset({"api_unavailable"})

#: What one janki record calls each JMDict verb class. These are the values
#: ``VocabularyRecord.verb_group`` carries and M2.4's ``conjugate`` switches on.
GODAN = "godan"
ICHIDAN = "ichidan"
SURU = "suru"
KURU = "kuru"

Transport = Callable[[str, dict[str, Any], dict[str, str]], tuple[int, Any]]


class JpdbError(JankiError):
    """A jpdb request failed, or answered with something unusable.

    ``error`` carries the API's *stable* error id (``bad_key``,
    ``api_unavailable``, ...) rather than its prose, so calling code can branch
    on it without matching on a message that may be reworded. ``status`` is the
    HTTP status where there was one.
    """

    def __init__(self, message: str, *, error: str = "", status: int | None = None) -> None:
        super().__init__(message)
        self.error = error
        self.status = status


def api_key_from_env(env: Mapping[str, str] | None = None) -> str:
    """The jpdb API key, or a clean error explaining where it comes from."""
    value = (env if env is not None else os.environ).get(API_KEY_ENV, "").strip()
    if not value:
        raise JpdbError(
            f"{API_KEY_ENV} is not set. Copy your API key from the jpdb.io settings page "
            f"and export it: {API_KEY_ENV}=... (janki never reads secrets from janki.toml)."
        )
    return value


def _decode_body(raw: bytes, status: int, url: str) -> Any:
    """Decode a response body, or explain why it could not be."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        # An empty body is fine on an error status (the status is the message)
        # and never fine on a success status.
        if 200 <= status < 300:
            raise JpdbError(f"{url} answered HTTP {status} with an empty body", status=status)
        return {}
    try:
        return json.loads(text)
    except ValueError as exc:
        if 200 <= status < 300:
            excerpt = text if len(text) <= 200 else text[:197] + "..."
            raise JpdbError(
                f"{url} answered HTTP {status} with a body that is not JSON: {excerpt}",
                status=status,
            ) from exc
        # A non-JSON error body (an HTML 502 page from a proxy) is not a
        # parsing bug; the status alone is the report.
        return {}


def urllib_transport(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[int, Any]:
    """The default transport: one JSON POST via ``urllib.request``.

    Returns ``(status, decoded json)``. An HTTP error status is *returned*, not
    raised: the client decides what is retryable, and jpdb puts its stable
    error id in the body of a 4xx.
    """
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, _decode_body(response.read(), response.status, url)
    except urllib.error.HTTPError as exc:
        # Reading the error body is live socket I/O on a connection that has
        # already misbehaved — `urlopen` raises at the status line, with the
        # response unread — and it happens *inside* this handler, where the
        # sibling clauses below cannot reach it. The status is what jpdb's
        # retry logic needs; a body that will not come is not worth losing it.
        try:
            detail = exc.read()
        except (OSError, http.client.HTTPException):
            detail = b""
        return exc.code, _decode_body(detail, exc.code, url)
    except urllib.error.URLError as exc:
        raise JpdbError(f"Could not reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise JpdbError(f"{url} timed out after {timeout:g}s") from exc
    except (OSError, http.client.HTTPException) as exc:
        # `URLError` covers less than it looks: `urlopen` wraps only the
        # *request*, so a peer that accepts the connection and closes it
        # without answering arrives as a bare `RemoteDisconnected` — a
        # `ConnectionResetError` and a `BadStatusLine`, neither a `URLError`.
        # Escaping as a non-JankiError means a traceback where an error
        # message belongs.
        raise JpdbError(f"Could not reach {url}: {exc}") from exc


def _backoff_delay(attempt: int, base: float, cap: float, jitter: Callable[[], float]) -> float:
    """Seconds to wait before try ``attempt + 1``.

    Exponential with "equal jitter": half the capped delay is fixed, half is
    random. Full jitter can pick a near-zero wait and re-hit a rate limiter
    immediately; no jitter has several clients (or several janki commands)
    retrying in lockstep.
    """
    capped = min(cap, base * 2 ** (attempt - 1))
    return capped / 2 + capped / 2 * max(0.0, min(1.0, jitter()))


def as_pair(word: Any) -> list[int]:
    """``[vid, sid]`` for anything that identifies a jpdb word.

    Accepts the pair itself, a mapping with ``vid``/``sid`` keys (what
    :meth:`JpdbClient.list_deck_vocabulary` returns), or an object carrying
    those attributes. Callers pass around whichever they happen to hold; the
    wire only ever sees the pair.
    """
    if isinstance(word, Mapping):
        vid, sid = word.get("vid"), word.get("sid")
    elif isinstance(word, str) or not isinstance(word, Sequence):
        vid, sid = getattr(word, "vid", None), getattr(word, "sid", None)
    elif len(word) == 2:
        vid, sid = word
    else:
        raise JpdbError(f"A jpdb word is a [vid, sid] pair; got {len(word)} value(s): {word!r}")
    try:
        return [int(vid), int(sid)]  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise JpdbError(f"A jpdb word is a [vid, sid] pair of integers; got {word!r}") from exc


def _zip_rows(rows: Any, fields: Sequence[str], what: str) -> list[dict[str, Any]]:
    """Turn column-oriented rows into dicts, or say exactly how they did not fit.

    This is the function that keeps positional rows inside this module. A row
    whose width does not match the requested field list is never truncated or
    padded: the values would land under the wrong names and be written into
    records as if they were dictionary facts.
    """
    if not isinstance(rows, list):
        raise JpdbError(
            f"jpdb {what}: expected a list of rows, got {type(rows).__name__}",
        )
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if isinstance(row, str) or not isinstance(row, Sequence):
            raise JpdbError(
                f"jpdb {what}: row {index} is {type(row).__name__}, not a list of values"
            )
        if len(row) != len(fields):
            raise JpdbError(
                f"jpdb {what}: row {index} has {len(row)} value(s) but "
                f"{len(fields)} field(s) were requested ({', '.join(fields)}). "
                "Responses are column-oriented, so a width mismatch means the "
                "columns cannot be named."
            )
        entries.append(dict(zip(fields, row, strict=True)))
    return entries


def _is_three_column(rows: Any) -> bool:
    """Whether every row carries a third value beside the ``[vid, sid]`` pair."""
    return (
        isinstance(rows, list)
        and bool(rows)
        and all(isinstance(row, list) and len(row) == 3 for row in rows)
    )


def _clean_fields(fields: Iterable[str], what: str) -> tuple[str, ...]:
    cleaned = tuple(str(name).strip() for name in fields if str(name).strip())
    if not cleaned:
        raise JpdbError(f"jpdb {what}: at least one field must be requested")
    return cleaned


@dataclass(frozen=True, slots=True)
class ParseResult:
    """A ``/parse`` response, with every positional row already named.

    ``tokens`` are the tokens of the one text that was parsed, in order, each a
    dict of the requested ``token_fields``. ``vocabulary`` is the shared
    dictionary-entry table the tokens' ``vocabulary_index`` points into — use
    :meth:`vocabulary_for` rather than indexing it by hand.
    """

    tokens: list[dict[str, Any]]
    vocabulary: list[dict[str, Any]]

    def vocabulary_for(self, token: Mapping[str, Any]) -> dict[str, Any] | None:
        """The dictionary entry a token resolves to, or ``None``.

        ``None`` is a real answer, not a failure: jpdb returns tokens it has no
        entry for with a null ``vocabulary_index``.
        """
        index = token.get("vocabulary_index")
        if not isinstance(index, int) or isinstance(index, bool):
            return None
        if not 0 <= index < len(self.vocabulary):
            return None
        return self.vocabulary[index]


class JpdbClient:
    """Wraps the jpdb.io v1 API.

    ``transport`` is a callable ``(url, json_body, headers) -> (status, json)``;
    it defaults to :func:`urllib_transport` and tests inject a fake. ``sleep``
    and ``jitter`` are injectable for the same reason — a retry test that
    actually sleeps is a test nobody runs.
    """

    def __init__(
        self,
        api_key: str,
        transport: Transport | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        max_tries: int = DEFAULT_MAX_TRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        sleep: Callable[[float], Any] = time.sleep,
        jitter: Callable[[], float] = random.random,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        key = (api_key or "").strip()
        if not key:
            raise JpdbError(
                f"A jpdb API key is required. Set {API_KEY_ENV} in the environment "
                "(janki never reads secrets from janki.toml)."
            )
        self._api_key = key
        self._base_url = base_url.rstrip("/") + "/"
        self._transport: Transport = transport or partial(urllib_transport, timeout=timeout)
        self._max_tries = max(1, int(max_tries))
        self._base_delay = float(base_delay)
        self._max_delay = float(max_delay)
        self._sleep = sleep
        self._jitter = jitter

    # -- plumbing ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST one request, retrying the failures that are worth retrying.

        Any 2xx counts as success: the API documents 200 on some endpoints and
        201 on others, and which is which is not a thing this client should
        have an opinion about.
        """
        url = self._base_url + endpoint.lstrip("/")
        for attempt in range(1, self._max_tries + 1):
            status, payload = self._transport(url, body, self._headers())
            error, message = _error_of(payload)
            ok = 200 <= status < 300 and not error
            if ok:
                if not isinstance(payload, Mapping):
                    raise JpdbError(
                        f"jpdb {endpoint}: expected a JSON object, got "
                        f"{type(payload).__name__}",
                        status=status,
                    )
                return dict(payload)
            retryable = status in RETRYABLE_STATUSES or error in RETRYABLE_ERRORS
            if retryable and attempt < self._max_tries:
                self._sleep(
                    _backoff_delay(attempt, self._base_delay, self._max_delay, self._jitter)
                )
                continue
            raise JpdbError(
                _failure_message(endpoint, status, error, message, attempt),
                error=error,
                status=status,
            )
        raise AssertionError("unreachable: the retry loop always returns or raises")

    # -- endpoints --------------------------------------------------------

    def ping(self) -> dict[str, Any]:
        """Check that the API answers and the key is accepted."""
        return self._request("ping", {})

    def list_user_decks(
        self, fields: Sequence[str] = DEFAULT_DECK_FIELDS
    ) -> list[dict[str, Any]]:
        """The account's decks, one dict per deck in the requested fields."""
        names = _clean_fields(fields, "list-user-decks")
        payload = self._request("list-user-decks", {"fields": list(names)})
        return _zip_rows(payload.get("decks"), names, "list-user-decks")

    def list_deck_vocabulary(
        self, deck_id: int, fetch_occurences: bool = False
    ) -> list[dict[str, Any]]:
        """The ``[vid, sid]`` pairs in one deck, as ``{"vid", "sid"}`` dicts.

        With ``fetch_occurences=True`` each dict also carries ``occurences`` —
        how often the word appears in the deck's source material, useful for
        mining decks. **Both spellings are the API's own misspelling of
        "occurrences" and are wire format: do not correct them.**
        """
        payload = self._request(
            "deck/list-vocabulary",
            # `fetch_occurences` [sic] — the API's spelling. Correcting it here
            # silently turns the parameter off.
            {"id": int(deck_id), "fetch_occurences": bool(fetch_occurences)},
        )
        rows = payload.get("vocabulary")
        # `occurences` [sic] again. The community documents two shapes for where
        # the counts ride — a top-level list parallel to the words, or a third
        # column on each row — so accept both rather than lose them to a guess.
        # The third-column form has to be split off *before* the zip, which
        # rejects a row whose width does not match its fields.
        inline: list[Any] | None = None
        if fetch_occurences and _is_three_column(rows):
            inline = [row[2] for row in rows]
            rows = [row[:2] for row in rows]
        entries = _zip_rows(rows, ("vid", "sid"), "deck/list-vocabulary")
        if not fetch_occurences:
            return entries
        top_level = payload.get("occurences")
        counts = top_level if isinstance(top_level, list) else inline
        if counts is None:
            return entries
        if len(counts) != len(entries):
            raise JpdbError(
                f"jpdb deck/list-vocabulary: {len(counts)} occurence count(s) for "
                f"{len(entries)} word(s)"
            )
        for entry, count in zip(entries, counts, strict=True):
            entry["occurences"] = count
        return entries

    def lookup_vocabulary(
        self,
        pairs: Iterable[Any],
        fields: Sequence[str] = DEFAULT_LOOKUP_FIELDS,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """Dictionary data for ``[vid, sid]`` pairs, batched.

        Returns one dict per requested pair, in request order. Every dict
        carries ``vid`` and ``sid`` — the pair it was requested for — even when
        they were not among ``fields``, so a caller never has to re-zip the
        results against its own input to know which word it is holding.
        """
        names = _clean_fields(fields, "lookup-vocabulary")
        wanted = [as_pair(word) for word in pairs]
        size = max(1, int(batch_size))
        entries: list[dict[str, Any]] = []
        for start in range(0, len(wanted), size):
            batch = wanted[start : start + size]
            payload = self._request(
                "lookup-vocabulary", {"list": batch, "fields": list(names)}
            )
            rows = _zip_rows(
                payload.get("vocabulary_info"), names, "lookup-vocabulary"
            )
            if len(rows) != len(batch):
                raise JpdbError(
                    f"jpdb lookup-vocabulary: asked about {len(batch)} word(s), got "
                    f"{len(rows)} answer(s); the results are positional, so they "
                    "cannot be matched to the request."
                )
            for (vid, sid), row in zip(batch, rows, strict=True):
                row.setdefault("vid", vid)
                row.setdefault("sid", sid)
                entries.append(row)
        return entries

    def parse(
        self,
        text: str,
        *,
        token_fields: Sequence[str] = DEFAULT_TOKEN_FIELDS,
        vocabulary_fields: Sequence[str] = DEFAULT_VOCABULARY_FIELDS,
        forced_furigana: Sequence[Sequence[Any]] | None = None,
        encoding: str = DEFAULT_ENCODING,
    ) -> ParseResult:
        """Parse one text into tokens plus the dictionary entries they resolve to.

        ``forced_furigana`` disambiguates homographs: spans of the shape
        ``[position, length, reading]`` (see :func:`forced_furigana_span`) that
        tell jpdb which reading the text uses. Positions and lengths — in the
        request *and* in any ``position``/``length`` token fields — are counted
        in ``encoding``, which is sent as ``position_length_encoding`` whenever
        either is in play.

        One text per call, deliberately: every caller in this project parses a
        single expression or a single sentence, and the multi-text form's
        response indexing is one more thing to get wrong for no gain.
        """
        tokens = _clean_fields(token_fields, "parse")
        vocabulary = _clean_fields(vocabulary_fields, "parse")
        body: dict[str, Any] = {
            # A list of one: the response's `tokens` is then one token list per
            # input text, a shape that does not change when a caller batches.
            "text": [text],
            "token_fields": list(tokens),
            "vocabulary_fields": list(vocabulary),
        }
        if forced_furigana is not None or _POSITIONAL_TOKEN_FIELDS & set(tokens):
            body["position_length_encoding"] = _encoding_name(encoding)
        if forced_furigana is not None:
            body["furigana"] = [list(span) for span in forced_furigana]
        payload = self._request("parse", body)
        return ParseResult(
            tokens=_zip_rows(_tokens_of(payload.get("tokens")), tokens, "parse tokens"),
            vocabulary=_zip_rows(payload.get("vocabulary"), vocabulary, "parse vocabulary"),
        )


def _error_of(payload: Any) -> tuple[str, str]:
    """The API's stable error id and its prose, if the payload carries them."""
    if not isinstance(payload, Mapping):
        return "", ""
    error = payload.get("error")
    message = payload.get("error_message")
    return (
        str(error).strip() if isinstance(error, str) else "",
        str(message).strip() if isinstance(message, str) else "",
    )


def _failure_message(endpoint: str, status: int, error: str, message: str, tries: int) -> str:
    text = f"jpdb {endpoint} failed"
    if status:
        text += f" (HTTP {status})"
    detail = message or _ERROR_HINTS.get(error, "")
    if error and detail:
        text += f": {error} — {detail}"
    elif error or detail:
        text += f": {error or detail}"
    if tries > 1:
        text += f" after {tries} tries"
    if error == "bad_key":
        text += f". Check {API_KEY_ENV}."
    return text


# Prose for the error ids most likely to be hit with no server-supplied message.
_ERROR_HINTS = {
    "bad_key": "the API key was not accepted",
    "api_unavailable": "the API is temporarily unavailable",
}


def _tokens_of(raw: Any) -> Any:
    """The token rows for the single text we parsed.

    ``/parse`` answers a list of texts with a list of token lists. This client
    always sends one text, so the answer is normally ``[[token, ...]]`` — but
    accept a bare ``[token, ...]`` too, since which shape a single-text request
    gets back is exactly the kind of detail M2.1F may have to correct against a
    live capture, and a wrong guess here would look like a corrupt response
    rather than a nesting mismatch.
    """
    if not isinstance(raw, list) or not raw:
        return raw
    first = raw[0]
    if isinstance(first, list) and (not first or isinstance(first[0], list)):
        return first
    return raw


def _encoding_name(encoding: str) -> str:
    name = str(encoding).strip().lower().replace("-", "").replace("_", "")
    if name not in _ENCODING_LENGTHS:
        raise JpdbError(
            f"Unknown position_length_encoding {encoding!r}; jpdb understands "
            + ", ".join(sorted(_ENCODING_LENGTHS))
        )
    return name


def encoded_length(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    """Length of ``text`` in the units ``encoding`` counts positions in."""
    return _ENCODING_LENGTHS[_encoding_name(encoding)](text)


def forced_furigana_span(
    expression: str, reading: str, encoding: str = DEFAULT_ENCODING
) -> list[list[Any]]:
    """The forced-furigana spans that pin ``expression`` to ``reading``.

    The wire shape is ``[[position, length, reading]]`` with position and
    length counted in ``encoding`` — the same value the request sends as
    ``position_length_encoding``. This computes the *whole-expression* span,
    which is what every caller here needs: janki parses the expression on its
    own, so the span starts at 0 and covers all of it.
    """
    if not expression:
        raise JpdbError("Forced furigana needs an expression to attach to")
    if not reading:
        raise JpdbError(
            f"Forced furigana needs a reading for {expression!r}; parse without it instead"
        )
    return [[0, encoded_length(expression, encoding), reading]]


def furigana_to_anki(segments: Any) -> str:
    """Render jpdb token furigana as Anki furigana notation.

    jpdb gives a token's furigana as a list of segments, each either a plain
    kana string or a ``[text, reading]`` pair. Anki wants ``text[reading]``
    with **a space before every bracketed group except one at the very start**
    — that space is what separates a ruby group from the kana in front of it,
    and Anki renders the reading over the wrong characters without it::

        [["話", "はな"], "す"]                   -> 話[はな]す
        ["お", ["茶", "ちゃ"]]                   -> お 茶[ちゃ]
        [["日", "にっ"], ["本", "ぽん"], ["語", "ご"]] -> 日[にっ] 本[ぽん] 語[ご]

    Note the third case: jpdb segments 日本語 per kanji, not as the 日本 + 語
    compound a human would write, and reads it にっぽんご. Segmentation and
    reading are both the dictionary's to decide — this function only formats
    what it is handed.

    ``None`` or an empty list renders as ``""``: an all-kana token needs no
    furigana markup at all, and an empty string is what "nothing to write"
    looks like to the fill-empty enrichment rules. jpdb does send ``None``
    rather than a lone kana segment for all-kana tokens (confirmed by the
    M2.1F capture), and the note templates fall back to ``{{Reading}}`` when
    ``{{Furigana}}`` is empty, so nothing is lost on the card.

    The shape is confirmed against a real capture (``tests/fixtures/
    jpdb-parse-sample.json``), but still parsed defensively and never
    *silently*: a segment this cannot read raises rather than being dropped,
    because a dropped segment produces plausible-looking furigana with a kanji
    missing its reading.
    """
    if segments is None:
        return ""
    if isinstance(segments, str):
        # A whole furigana value given as a plain string is one kana segment.
        segments = [segments]
    if not isinstance(segments, Sequence):
        raise JpdbError(
            f"jpdb furigana: expected a list of segments, got {type(segments).__name__}"
        )
    out = ""
    for segment in segments:
        text, reading = _furigana_segment(segment)
        if not reading or reading == text:
            out += text
            continue
        if out and not out.endswith(" "):
            out += " "
        out += f"{text}[{reading}]"
    return out


def furigana_to_reading(segments: Any) -> str:
    """The kana a token's furigana spells out — the reading of the *surface* form.

    The companion to :func:`furigana_to_anki`, and the reason both exist: a
    token's furigana describes the text as written, while the dictionary entry
    it resolves to describes the lemma. Parse 行った and jpdb answers with the
    entry for 行く — ``entry["reading"]`` is ``いく``, which does not read 行った,
    but the token's furigana ``[["行", "い"], "った"]`` gives ``いった``, which
    does.

    Every segment contributes: its reading where it has one, its own text where
    it is plain kana. ``None`` (jpdb's answer for an all-kana token) gives
    ``""`` — the caller knows the text it asked about and this function does
    not, so "nothing stated" is the honest answer rather than a guess.

    Nothing here is segmentation janki invented: the split and every reading in
    it are the dictionary's, which is what keeps this on the right side of
    AGENTS.md's rule against guessing furigana for mixed kanji/kana words.
    """
    if segments is None:
        return ""
    if isinstance(segments, str):
        segments = [segments]
    if not isinstance(segments, Sequence):
        raise JpdbError(
            f"jpdb furigana: expected a list of segments, got {type(segments).__name__}"
        )
    out = ""
    for segment in segments:
        text, reading = _furigana_segment(segment)
        out += reading or text
    return out


def _furigana_segment(segment: Any) -> tuple[str, str]:
    """One furigana segment as ``(text, reading)``; reading ``""`` means plain."""
    if isinstance(segment, str):
        return segment, ""
    if isinstance(segment, Sequence):
        if len(segment) == 1 and isinstance(segment[0], str):
            return segment[0], ""
        if len(segment) == 2 and all(isinstance(part, str) for part in segment):
            return segment[0], segment[1]
    raise JpdbError(
        "jpdb furigana: a segment is either kana text or a [text, reading] pair; "
        f"got {segment!r}"
    )


# --- JMDict part-of-speech tables ------------------------------------------
#
# Single owner: M2.5 (import-jpdb) and M2.6 (enrich --jpdb) import these rather
# than each building a table, so that two importers can never disagree about
# what `v5k-s` is. jpdb returns JMDict codes as a list, most specific first.

_VERB_GROUPS: dict[str, str] = {
    # Godan, one entry per ending plus the irregular-ish subclasses.
    "v5u": GODAN,
    "v5u-s": GODAN,  # 問う, 請う
    "v5k": GODAN,
    "v5k-s": GODAN,  # 行く
    "v5g": GODAN,
    "v5s": GODAN,
    "v5t": GODAN,
    "v5n": GODAN,
    "v5b": GODAN,
    "v5m": GODAN,
    "v5r": GODAN,
    "v5r-i": GODAN,  # ある
    "v5aru": GODAN,  # ござる, いらっしゃる
    "v5uru": GODAN,  # 得る (うる)
    "v1": ICHIDAN,
    "v1-s": ICHIDAN,  # くれる
    # `vs` on a noun ("takes する") is what makes 勉強 conjugate, so it maps to
    # the suru group even though the word itself is a noun. The part-of-speech
    # table below still calls it a noun; the two answers are independent.
    "vs": SURU,
    "vs-s": SURU,
    "vs-i": SURU,
    "vs-c": SURU,
    "vk": KURU,
}

# Codes that name a verb class this project has no conjugation table for
# (archaic classes, ずる verbs). Deliberately absent from _VERB_GROUPS: an
# empty verb_group is flagged, a wrong one is silently mis-conjugated.

_PARTS_OF_SPEECH: dict[str, str] = {
    "adj-i": "i-adjective",
    "adj-ix": "i-adjective",  # いい/よい
    "adj-na": "na-adjective",
    "adj-no": "no-adjective",
    "adj-pn": "pre-noun adjectival",
    "adj-t": "taru adjective",
    "adj-f": "prenominal",
    "adj-nari": "adjective",
    "adj-ku": "adjective",
    "adj-shiku": "adjective",
    "n": "noun",
    "n-adv": "noun",
    "n-pr": "proper noun",
    "n-pref": "noun",
    "n-suf": "noun",
    "n-t": "noun",
    "pn": "pronoun",
    "adv": "adverb",
    "adv-to": "adverb",
    "aux": "auxiliary",
    "aux-v": "auxiliary verb",
    "aux-adj": "auxiliary adjective",
    "conj": "conjunction",
    "cop": "copula",
    "cop-da": "copula",
    "ctr": "counter",
    "exp": "expression",
    "int": "interjection",
    "num": "numeric",
    "pref": "prefix",
    "prt": "particle",
    "suf": "suffix",
}

# Every JMDict verb code: v1, v5k-s, v2a-s, v4r, vk, vs-i, vz, vr, vn. `vi` and
# `vt` are deliberately excluded — they say transitivity, not part of speech,
# and a word's POS list often starts with one of them.
_VERB_CODE = re.compile(r"^v(?:[0-9]|k|n|r|s|z)")

_TRANSITIVITY: dict[str, str] = {"vt": "transitive", "vi": "intransitive"}


def _codes(value: Any) -> list[str]:
    """JMDict codes from whatever the caller holds: one code, or a list of them.

    Unreadable entries are skipped rather than raised on: part of speech is
    metadata, and one odd value in a 2,000-word import must not stop the
    import.
    """
    if value is None:
        return []
    items: Iterable[Any] = [value] if isinstance(value, str) else value
    if not isinstance(items, Iterable):
        return []
    return [code for item in items if isinstance(item, str) and (code := item.strip().lower())]


def pos_to_verb_group(codes: Any) -> str:
    """``"godan"``/``"ichidan"``/``"suru"``/``"kuru"``, or ``""`` when unknown.

    ``""`` is a real answer: M2.4 conjugates only the classes it has tables
    for, and an empty verb group is flagged for a human rather than guessed at.
    """
    for code in _codes(codes):
        group = _VERB_GROUPS.get(code)
        if group:
            return group
    return ""


#: Labels that describe a *secondary* use of a word that is otherwise a verb.
#: jpdb puts these first often enough to matter: 見る comes back as
#: ``["aux-v", "vt", "v1"]`` and 分かる as ``["int", "vi", "v5", "v5r"]``, both
#: measured against the live API. Taking the first recognized code labelled six
#: of twenty ordinary verbs "auxiliary verb" or "interjection" on real cards,
#: each one contradicting the verb group on the same card.
_SECONDARY_TO_A_VERB = frozenset(
    {
        "auxiliary",
        "auxiliary verb",
        "auxiliary adjective",
        "interjection",
        # する arrives as ["aux-v", "vi", "suf", "vt", "vs"]: "suffix" is its
        # 勉強する use, not what the word is. A word that really is only a suffix
        # (〜的, 〜家) carries no verb code, so it keeps the label.
        "suffix",
    }
)


def pos_to_part_of_speech(codes: Any) -> str:
    """A human-readable part of speech for a JMDict code list, or ``""``.

    First recognized code wins, *except* that a verb code outranks the
    auxiliary and interjection labels wherever both appear. jpdb does not order
    these by significance the way the rest of this function assumes: 〜ている and
    分かる! are real but secondary uses, and they arrive ahead of the codes for
    the ordinary verb the card is about.

    Only that family is outranked. A noun that takes する (``["n", "vs"]``) is
    still a noun here, with a ``suru`` verb group — that ordering is jpdb saying
    something true about which word it is, not about which sense came first.

    A word that genuinely *is* primarily an auxiliary — た, られる — is labelled
    "verb" by this rule. That is the accepted cost: those are not words a
    vocabulary deck teaches as entries, while 見る, する, なる, いる, 来る and
    分かる are, and mislabelling those teaches a beginner the wrong category for
    six of the commonest verbs in the language.
    """
    first = ""
    for code in _codes(codes):
        label = _PARTS_OF_SPEECH.get(code)
        if label:
            if label not in _SECONDARY_TO_A_VERB:
                # `first or label`, not `label`: only a *verb* code outranks a
                # secondary one, which is what the rule above says. Returning
                # the later label demoted 〜的 to a noun for ["suf", "n"] and
                # turned ["aux-adj", "adj-i"] into an い-adjective — which then
                # reached `conjugate` and wrote a whole paradigm for ない.
                return first or label
            first = first or label
            continue
        if _VERB_CODE.match(code):
            return "verb"
    return first


def pos_to_transitivity(codes: Any) -> str:
    """``"transitive"``/``"intransitive"`` from ``vt``/``vi``, or ``""``.

    **A word tagged both gets neither.** する comes back as
    ``["aux-v", "vi", "suf", "vt", "vs"]`` — measured, not guessed — and taking
    the first match called it intransitive, on a card whose own example is
    仕事をします. Which one is true depends on the sense, so there is no single
    answer to state, and stating one anyway is exactly the guess this project
    does not make. The card then shows no transitivity rather than a wrong one,
    and `usage_notes` is where a real distinction belongs.
    """
    found = {
        transitivity
        for code in _codes(codes)
        if (transitivity := _TRANSITIVITY.get(code))
    }
    if len(found) != 1:
        return ""
    return found.pop()


# The remaining wire-value normalizers live here for the same reason the POS
# tables do: `import-jpdb` and `enrich --jpdb` read the same fields off the same
# endpoints, and a second definition of what `pitch_accent` or `frequency_rank`
# means is a second thing to keep in step.


def accent_patterns(value: Any) -> list[str]:
    """jpdb's ``pitch_accent`` as a list of patterns.

    One entry per pattern, first primary. A word with several accepted accents
    really does come back with several (confirmed by the M2.1F capture), so a
    lone string is the shape to widen, not the shape to expect.
    """
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, Sequence):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def frequency_rank(value: Any) -> int | None:
    """jpdb's ``frequency_rank`` as an int, or ``None``.

    ``None`` is "never looked up" and 0 would be a real rank, so an unparseable
    value must not become a number. ``bool`` is rejected explicitly because it
    is an ``int`` in Python and ``True`` would silently rank a word first.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
