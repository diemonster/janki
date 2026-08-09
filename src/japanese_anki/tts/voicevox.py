"""VOICEVOX, driven through the one endpoint that honours a forced accent.

**The endpoint choice is the whole feature.** VOICEVOX will happily synthesize
から katakana without being told it is kana, and the result sounds fine — it is
simply accented however the engine guessed. That is the 橋/箸 failure this
milestone exists to prevent, arriving as plausible audio rather than as an
error, which is the worst way for it to arrive.

``is_kana=true`` is accepted by ``/accent_phrases`` **and by nothing else**.
Passing it to ``/audio_query`` does not fail: FastAPI drops query parameters a
route does not declare, so the request succeeds and returns a normally-parsed
query whose accent is whatever the engine decided. So the flow is:

1. ``POST /accent_phrases?text=<kana>&is_kana=true&speaker=N`` — the forced
   accent phrases, and the only step that reads the accent mark.
2. ``POST /audio_query?text=<kana without the mark>&speaker=N`` — for the rest
   of the query: speed, pitch, pauses, sampling rate. Its accent phrases are
   the engine's guess and are about to be thrown away; what is wanted is the
   *envelope*, which is engine-versioned and must never be hand-constructed.
   Hand-built defaults are the other way this breaks, quietly, on an engine
   upgrade that adds a field.
3. Replace the query's ``accent_phrases`` with the forced ones.
4. ``POST /synthesis?speaker=N`` with that query.

A sentence — ``forced_accent=False`` — skips step 1 entirely and lets the engine
read naturally, which is what a sentence wants: forcing an accent on a whole
sentence would require accent data janki does not have for one.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from japanese_anki.pitch import ACCENT_MARK
from japanese_anki.tts import TtsError

__all__ = [
    "DEFAULT_TIMEOUT",
    "LAUNCH_HINT",
    "Transport",
    "VoicevoxProvider",
    "urllib_transport",
]

#: Seconds. Synthesis of one word is fast; the ceiling is for an engine that is
#: reachable but wedged, which must not hang a run over a few hundred records.
DEFAULT_TIMEOUT = 30.0

LAUNCH_HINT = (
    "Start the VOICEVOX engine and try again — open the VOICEVOX app, or run "
    "the engine directly (it listens on http://localhost:50021 by default). "
    "Set tts.voicevox_url in janki.toml if it listens somewhere else."
)

#: ``(method, url including query, json body or None) -> (status, raw bytes)``.
#:
#: Shaped for this API rather than reusing jpdb's: VOICEVOX puts its inputs in
#: the query string, varies the method, and returns audio — so a transport that
#: assumes "POST a JSON body, decode JSON back" describes the wrong service.
#: Decoding stays in the provider, because only the provider knows which of
#: these calls answers with JSON and which answers with a WAV.
Transport = Callable[[str, str, Any | None], tuple[int, bytes]]


def urllib_transport(
    method: str,
    url: str,
    body: Any | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[int, bytes]:
    """One HTTP call via ``urllib.request``, returning the raw body.

    An HTTP error status is *returned* rather than raised, the way jpdb's
    transport does it, so the provider decides what a given status means at a
    given step. A connection that cannot be made at all is different — the
    engine is not running, which is the common case — and raises with the
    launch hint attached.
    """
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    try:
        # Inside the try: `Request()` rejects a URL with no scheme before any
        # I/O, which is what an empty `tts.voicevox_url` produces, and that has
        # to arrive as a JankiError like every other way of not reaching the
        # engine.
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise TtsError(f"Could not reach VOICEVOX at {url}: {exc.reason}. {LAUNCH_HINT}") from exc
    except TimeoutError as exc:
        raise TtsError(f"VOICEVOX at {url} timed out after {timeout:g}s") from exc
    except ValueError as exc:
        # `Request()` rejects a URL with no scheme before any I/O — which is
        # what an empty `tts.voicevox_url` in janki.toml produces.
        raise TtsError(f"{url!r} is not a URL janki can request: {exc}") from exc
    except (OSError, http.client.HTTPException) as exc:
        # Everything else the stack can throw at us. `URLError` covers less than
        # it looks: `urlopen` wraps only the *request* in it, so a peer that
        # accepts the connection and closes it without answering surfaces as a
        # bare `RemoteDisconnected` — through an https-only port, another
        # process on 50021, or the engine still starting up. Escaping as a
        # non-JankiError means a traceback where the launch hint belongs.
        raise TtsError(f"Could not reach VOICEVOX at {url}: {exc}. {LAUNCH_HINT}") from exc


class VoicevoxProvider:
    """Word audio with the accent janki chose, not the one VOICEVOX guessed."""

    def __init__(
        self,
        base_url: str = "http://localhost:50021",
        speaker: int = 46,
        transport: Transport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._speaker = int(speaker)
        self._transport: Transport = transport or urllib_transport

    @property
    def name(self) -> str:
        return "voicevox"

    @property
    def voice(self) -> int:
        return self._speaker

    @property
    def launch_hint(self) -> str:
        return LAUNCH_HINT

    def _url(self, path: str, **params: Any) -> str:
        query = urllib.parse.urlencode(params, encoding="utf-8")
        return f"{self._base_url}{path}?{query}" if query else f"{self._base_url}{path}"

    def _call(self, method: str, url: str, body: Any | None = None) -> bytes:
        status, payload = self._transport(method, url, body)
        if status != 200:
            detail = payload.decode("utf-8", "replace").strip()
            raise TtsError(
                f"VOICEVOX answered {status} for {url}"
                + (f": {detail[:200]}" if detail else "")
            )
        return payload

    def _json_call(self, method: str, url: str, body: Any | None = None) -> Any:
        payload = self._call(method, url, body)
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TtsError(
                f"VOICEVOX answered {url} with something that is not JSON: {exc}"
            ) from exc

    def available(self) -> bool:
        """``GET /version``, answered as a plain yes or no.

        A local engine that is not running is the ordinary state of this system,
        not an exceptional one, so every way of failing to reach it — refused
        connection, a peer that hangs up, a timeout, a URL that is not one, an
        error status — is the same answer here. The caller prints
        :attr:`launch_hint`. :func:`urllib_transport` is what makes that true:
        it converts every transport failure to :class:`TtsError` so this does
        not have to enumerate them.
        """
        try:
            status, _ = self._transport("GET", self._url("/version"), None)
        except TtsError:
            return False
        return status == 200

    def synthesize(self, text_or_kana: str, *, forced_accent: bool) -> bytes:
        """WAV bytes, with the accent forced when ``forced_accent`` is set.

        The accent mark is stripped before ``/audio_query`` because that route
        does not read kana notation: left in, it is text to be pronounced. Step
        1 is the only place the mark means anything.
        """
        if not text_or_kana.strip():
            raise TtsError("Nothing to synthesize: the text is empty.")

        forced = None
        if forced_accent:
            forced = self._json_call(
                "POST",
                self._url(
                    "/accent_phrases",
                    text=text_or_kana,
                    is_kana="true",
                    speaker=self._speaker,
                ),
            )

        spoken = text_or_kana.replace(ACCENT_MARK, "") if forced_accent else text_or_kana
        query = self._json_call(
            "POST", self._url("/audio_query", text=spoken, speaker=self._speaker)
        )
        if not isinstance(query, dict):
            raise TtsError(
                f"VOICEVOX's /audio_query answered with {type(query).__name__}, not an "
                "object; janki will not guess at an AudioQuery it cannot read."
            )
        if forced_accent:
            # Branching on the flag, not on the payload: `forced is not None`
            # cannot tell "the engine answered null" from "we never asked", and
            # the first of those would fall through to the accent /audio_query
            # guessed — plausible audio, ledgered as forced, which is the whole
            # failure this provider exists to prevent. An empty list is refused
            # for the same reason: it is a schema-valid AudioQuery that
            # synthesizes to silence.
            if not isinstance(forced, list) or not forced:
                raise TtsError(
                    "VOICEVOX's /accent_phrases returned no usable accent "
                    f"phrases for {text_or_kana!r}; janki will not fall back to "
                    "a guessed accent."
                )
            # Only this key. The rest of the query is the engine's own envelope
            # — sampling rate, pauses, scales — and is versioned with it.
            query["accent_phrases"] = forced

        return self._call(
            "POST", self._url("/synthesis", speaker=self._speaker), query
        )
