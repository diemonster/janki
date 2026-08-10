"""Tests for the jpdb API client.

No network: every test drives the client through a fake transport with canned
responses (IMPLEMENTATION_PLAN rule 6), and the ``/parse`` sample is the
committed fixture, not a capture.
"""

from __future__ import annotations

import http.client
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from japanese_anki import cli, jpdb
from japanese_anki.jpdb import (
    API_KEY_ENV,
    DEFAULT_TOKEN_FIELDS,
    DEFAULT_VOCABULARY_FIELDS,
    GODAN,
    ICHIDAN,
    KURU,
    SURU,
    JpdbClient,
    JpdbError,
    api_key_from_env,
    as_pair,
    encoded_length,
    forced_furigana_span,
    furigana_to_anki,
    pos_to_part_of_speech,
    pos_to_transitivity,
    pos_to_verb_group,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "jpdb-parse-sample.json"

# The sentence the fixture stands in for; M2.1F captures this exact text.
FIXTURE_TEXT = "日本語を話す。お茶と食べ物をたべる。"


class FakeTransport:
    """A scripted ``(url, body, headers) -> (status, json)`` transport."""

    def __init__(self, *responses: tuple[int, Any]) -> None:
        self.responses = list(responses)
        self.calls: list[SimpleNamespace] = []

    def __call__(
        self, url: str, body: dict[str, Any], headers: dict[str, str]
    ) -> tuple[int, Any]:
        self.calls.append(SimpleNamespace(url=url, body=body, headers=headers))
        if not self.responses:
            raise AssertionError(f"unscripted request number {len(self.calls)} to {url}")
        return self.responses.pop(0)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [call.body for call in self.calls]


def _client(transport: FakeTransport, **kwargs: Any) -> JpdbClient:
    kwargs.setdefault("jitter", lambda: 0.0)
    kwargs.setdefault("sleep", lambda _seconds: None)
    return JpdbClient("test-key", transport, **kwargs)


def _fixture_response() -> dict[str, Any]:
    """The ``response`` half of the fixture — the only half the client sees."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["response"]


def _parse_the_fixture() -> Any:
    return _client(FakeTransport((200, _fixture_response()))).parse(FIXTURE_TEXT)


# --- transport, auth, retries ----------------------------------------------


def test_ping_posts_a_bearer_token_and_accepts_any_2xx() -> None:
    # 201 rather than 200: the API documents both, depending on the endpoint.
    transport = FakeTransport((201, {}))

    assert _client(transport).ping() == {}

    call = transport.calls[0]
    assert call.url == "https://jpdb.io/api/v1/ping"
    assert call.headers["Authorization"] == "Bearer test-key"
    assert call.headers["Content-Type"] == "application/json"


def test_an_empty_api_key_never_reaches_the_network() -> None:
    transport = FakeTransport((200, {}))

    with pytest.raises(JpdbError) as excinfo:
        JpdbClient("  ", transport)

    assert API_KEY_ENV in str(excinfo.value)
    assert transport.calls == []


def test_the_stable_error_id_travels_on_the_exception_and_is_not_retried() -> None:
    transport = FakeTransport((403, {"error": "bad_key", "error_message": "Invalid API key"}))

    with pytest.raises(JpdbError) as excinfo:
        _client(transport).ping()

    assert excinfo.value.error == "bad_key"
    assert excinfo.value.status == 403
    assert "Invalid API key" in str(excinfo.value)
    assert API_KEY_ENV in str(excinfo.value)
    assert len(transport.calls) == 1


def test_rate_limiting_is_retried_with_growing_backoff() -> None:
    transport = FakeTransport(
        (429, {"error": "too_many_requests"}),
        (429, {"error": "too_many_requests"}),
        (200, {}),
    )
    sleeps: list[float] = []

    _client(transport, sleep=sleeps.append).ping()

    assert len(transport.calls) == 3
    assert sleeps == [0.5, 1.0]  # jitter pinned to 0: half of 1s, then half of 2s


def test_jitter_lands_inside_the_capped_delay() -> None:
    transport = FakeTransport((429, {}), (200, {}))
    sleeps: list[float] = []

    _client(transport, sleep=sleeps.append, jitter=lambda: 1.0).ping()

    # Equal jitter: never below half the capped delay, never above it.
    assert sleeps == [1.0]


def test_api_unavailable_is_retried_until_the_budget_runs_out() -> None:
    transport = FakeTransport(*[(503, {"error": "api_unavailable"})] * 3)
    sleeps: list[float] = []

    with pytest.raises(JpdbError) as excinfo:
        _client(transport, sleep=sleeps.append, max_tries=3).ping()

    assert len(transport.calls) == 3
    assert len(sleeps) == 2
    assert excinfo.value.error == "api_unavailable"
    assert "after 3 tries" in str(excinfo.value)


def test_an_unknown_server_error_is_reported_once() -> None:
    transport = FakeTransport((500, {}))

    with pytest.raises(JpdbError) as excinfo:
        _client(transport).ping()

    assert len(transport.calls) == 1
    assert "HTTP 500" in str(excinfo.value)


def test_a_success_that_is_not_an_object_is_refused() -> None:
    transport = FakeTransport((200, [1, 2, 3]))

    with pytest.raises(JpdbError, match="JSON object"):
        _client(transport).ping()


# --- the default urllib transport -------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def test_the_default_transport_posts_json_and_decodes_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["body"] = json.loads(request.data.decode("utf-8"))
        seen["timeout"] = timeout
        return _FakeResponse(200, json.dumps({"decks": [[1, "Mining"]]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    decks = JpdbClient("test-key", timeout=5).list_user_decks(fields=("id", "name"))

    assert decks == [{"id": 1, "name": "Mining"}]
    assert seen["url"] == "https://jpdb.io/api/v1/list-user-decks"
    assert seen["method"] == "POST"
    assert seen["body"] == {"fields": ["id", "name"]}
    assert seen["timeout"] == 5


def test_the_default_transport_hands_an_http_error_back_for_the_retry_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error": "too_many_requests"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    sleeps: list[float] = []

    with pytest.raises(JpdbError) as excinfo:
        JpdbClient(
            "test-key", max_tries=2, sleep=sleeps.append, jitter=lambda: 0.0
        ).ping()

    assert excinfo.value.status == 429
    assert excinfo.value.error == "too_many_requests"
    assert len(sleeps) == 1  # the status was retried, not raised on the spot


def test_an_unreachable_api_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        raise urllib.error.URLError("nodename nor servname provided")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(JpdbError, match="Could not reach"):
        JpdbClient("test-key").ping()


def test_a_success_whose_body_is_not_json_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(200, b"<html>maintenance</html>")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(JpdbError, match="not JSON"):
        JpdbClient("test-key").ping()


# --- column-oriented responses ---------------------------------------------


def test_list_user_decks_names_the_columns() -> None:
    transport = FakeTransport((200, {"decks": [[1, "Mining", 1200], [7, "Genki I", 300]]}))

    decks = _client(transport).list_user_decks()

    assert decks == [
        {"id": 1, "name": "Mining", "word_count": 1200},
        {"id": 7, "name": "Genki I", "word_count": 300},
    ]
    assert transport.bodies[0] == {"fields": ["id", "name", "word_count"]}


def test_a_row_whose_width_misses_the_requested_fields_is_an_error() -> None:
    transport = FakeTransport((200, {"decks": [[1, "Mining"]]}))

    with pytest.raises(JpdbError) as excinfo:
        _client(transport).list_user_decks()

    message = str(excinfo.value)
    assert "2 value(s)" in message and "3 field(s)" in message
    assert "column-oriented" in message


def test_deck_vocabulary_comes_back_as_vid_sid_dicts() -> None:
    transport = FakeTransport((200, {"vocabulary": [[1000001, 1000001], [1000002, 5]]}))

    words = _client(transport).list_deck_vocabulary(42)

    assert words == [
        {"vid": 1000001, "sid": 1000001},
        {"vid": 1000002, "sid": 5},
    ]
    # The misspelling is the API's, and it is wire format: the request must
    # carry `fetch_occurences`, not `fetch_occurrences`.
    assert transport.bodies[0] == {"id": 42, "fetch_occurences": False}


def test_occurence_counts_ride_along_when_asked_for() -> None:
    transport = FakeTransport(
        (200, {"vocabulary": [[1, 1], [2, 2]], "occurences": [12, 3]}),
    )

    words = _client(transport).list_deck_vocabulary(42, fetch_occurences=True)

    assert words == [
        {"vid": 1, "sid": 1, "occurences": 12},
        {"vid": 2, "sid": 2, "occurences": 3},
    ]
    assert transport.bodies[0]["fetch_occurences"] is True


def test_occurence_counts_are_also_read_from_a_third_column() -> None:
    transport = FakeTransport((200, {"vocabulary": [[1, 1, 12], [2, 2, 3]]}))

    words = _client(transport).list_deck_vocabulary(42, fetch_occurences=True)

    assert words == [
        {"vid": 1, "sid": 1, "occurences": 12},
        {"vid": 2, "sid": 2, "occurences": 3},
    ]


def test_lookup_batches_and_keeps_the_request_order() -> None:
    pairs = [[vid, vid] for vid in range(1, 251)]
    transport = FakeTransport(
        (200, {"vocabulary_info": [[f"w{vid}"] for vid in range(1, 201)]}),
        (200, {"vocabulary_info": [[f"w{vid}"] for vid in range(201, 251)]}),
    )

    entries = _client(transport).lookup_vocabulary(pairs, fields=("spelling",))

    assert len(transport.calls) == 2
    assert len(transport.bodies[0]["list"]) == 200
    assert len(transport.bodies[1]["list"]) == 50
    assert transport.bodies[0]["fields"] == ["spelling"]
    assert [entry["spelling"] for entry in entries[:3]] == ["w1", "w2", "w3"]
    # vid/sid are attached from the request even though they were not asked
    # for, so a caller never has to re-zip results against its own input.
    assert entries[0] == {"spelling": "w1", "vid": 1, "sid": 1}
    assert entries[-1] == {"spelling": "w250", "vid": 250, "sid": 250}


def test_lookup_accepts_the_dicts_deck_vocabulary_returned() -> None:
    transport = FakeTransport((200, {"vocabulary_info": [["話す", "はなす"]]}))

    entries = _client(transport).lookup_vocabulary(
        [{"vid": 1000002, "sid": 1000002, "occurences": 4}], fields=("spelling", "reading")
    )

    assert transport.bodies[0]["list"] == [[1000002, 1000002]]
    assert entries[0]["vid"] == 1000002


def test_lookup_of_nothing_asks_jpdb_nothing() -> None:
    transport = FakeTransport()

    assert _client(transport).lookup_vocabulary([]) == []
    assert transport.calls == []


def test_lookup_refuses_an_answer_it_cannot_line_up_with_the_request() -> None:
    transport = FakeTransport((200, {"vocabulary_info": [["話す"]]}))

    with pytest.raises(JpdbError, match="asked about 2 word"):
        _client(transport).lookup_vocabulary([[1, 1], [2, 2]], fields=("spelling",))


@pytest.mark.parametrize(
    "word",
    [
        [1000002, 1000002],
        (1000002, 1000002),
        {"vid": 1000002, "sid": 1000002},
        {"vid": "1000002", "sid": "1000002"},
        SimpleNamespace(vid=1000002, sid=1000002),
    ],
)
def test_as_pair_takes_every_shape_a_word_id_is_held_in(word: Any) -> None:
    assert as_pair(word) == [1000002, 1000002]


@pytest.mark.parametrize("word", [[1], [1, 2, 3], "1000002", {"vid": 1}, None])
def test_as_pair_refuses_what_is_not_a_pair(word: Any) -> None:
    with pytest.raises(JpdbError):
        as_pair(word)


# --- the /parse contract ----------------------------------------------------


def test_parse_pins_the_request_contract() -> None:
    transport = FakeTransport((200, _fixture_response()))

    _client(transport).parse(FIXTURE_TEXT)

    body = transport.bodies[0]
    assert body["text"] == [FIXTURE_TEXT]
    assert body["token_fields"] == list(DEFAULT_TOKEN_FIELDS) == [
        "vocabulary_index",
        "furigana",
    ]
    assert body["vocabulary_fields"] == list(DEFAULT_VOCABULARY_FIELDS)
    # No positions in play, so no encoding is claimed.
    assert "position_length_encoding" not in body
    assert "furigana" not in body


def test_parse_sends_forced_furigana_with_the_encoding_it_is_measured_in() -> None:
    transport = FakeTransport((200, _fixture_response()))

    _client(transport).parse(
        "話す", forced_furigana=forced_furigana_span("話す", "はなす"), encoding="utf16"
    )

    body = transport.bodies[0]
    assert body["furigana"] == [[0, 2, "はなす"]]
    assert body["position_length_encoding"] == "utf16"


def test_parse_sends_the_encoding_when_positions_are_requested() -> None:
    transport = FakeTransport(
        (200, {"tokens": [[[0, 0, 3]]], "vocabulary": [["話す"]]}),
    )

    result = _client(transport).parse(
        "話す",
        token_fields=("vocabulary_index", "position", "length"),
        vocabulary_fields=("spelling",),
        encoding="utf8",
    )

    assert transport.bodies[0]["position_length_encoding"] == "utf8"
    assert result.tokens == [{"vocabulary_index": 0, "position": 0, "length": 3}]


def test_parse_names_every_column_and_resolves_a_token_to_its_entry() -> None:
    result = _parse_the_fixture()

    first = result.tokens[0]
    assert set(first) == set(DEFAULT_TOKEN_FIELDS)
    entry = result.vocabulary_for(first)
    assert entry is not None
    assert entry["spelling"] == "日本語"
    # にっぽんご, not the にほんご a human would write. jpdb's entry for this vid
    # picks the less common of JMDict's two readings, and the captured furigana
    # agrees (日[にっ]本[ぽん]語[ご]). Kept verbatim rather than "corrected":
    # a dictionary that disagrees with the curator is the exact case M2.6's
    # reading check is built for — it warns and declines to write, and it can
    # only do that if the client reports what jpdb actually said.
    assert entry["reading"] == "にっぽんご"
    assert entry["pitch_accent"] == ["LHHHHH"]
    assert entry["frequency_rank"] == 4800
    assert entry["part_of_speech"] == ["n"]
    assert entry["vid"] == 1464530
    assert entry["sid"] == 3361009543


def test_parse_accepts_a_token_list_that_is_not_nested_per_text() -> None:
    # Which shape a single-text request answers with is exactly what M2.1F may
    # have to correct; both read the same here.
    flat = {"tokens": [[0, ["たべる"]]], "vocabulary": [["たべる"]]}
    transport = FakeTransport((200, flat))

    result = _client(transport).parse("たべる", vocabulary_fields=("spelling",))

    assert result.tokens == [{"vocabulary_index": 0, "furigana": ["たべる"]}]


def test_a_token_jpdb_has_no_entry_for_resolves_to_none() -> None:
    transport = FakeTransport((200, {"tokens": [[[None, None]]], "vocabulary": []}))

    result = _client(transport).parse("〜", vocabulary_fields=("spelling",))

    assert result.vocabulary_for(result.tokens[0]) is None


# --- furigana_to_anki golden cases ------------------------------------------

# Anki's rule: `text[reading]`, with a space before every bracketed group that
# is not at the very start of the string.
# Two entries changed when the hand-written fixture was replaced by a live
# capture (M2.1F). Both were wrong guesses about jpdb, not wrong formatting —
# furigana_to_anki produced correct Anki notation from the real segments in
# every case, which is what this table is here to pin down.
GOLDEN_FURIGANA = {
    "話す": "話[はな]す",  # leading kanji + okurigana
    "お茶": "お 茶[ちゃ]",  # kanji after kana — the space is load-bearing
    # All kana: jpdb sends `null` furigana, not a single kana segment, so there
    # is nothing to annotate. Empty is the right field value — the note
    # templates fall back to {{Reading}} when {{Furigana}} is empty, so a
    # kana-only word still shows its reading on the card.
    "たべる": "",
    # Split per kanji, not as the 日本 + 語 compound a human would segment.
    "日本語": "日[にっ] 本[ぽん] 語[ご]",
    "食べ物": "食[た]べ 物[もの]",  # kanji, okurigana, kanji
}


def _golden_tokens() -> list[Any]:
    """Golden cases as ``(token, expected)`` pairs read out of the fixture.

    Parametrizing over the fixture rather than over hand-written segment lists
    is what makes M2.1F's acceptance gate mean something: once a live capture
    replaces the fixture, these tests exercise the captured furigana. Tokens
    with no golden expectation (a live capture will include particles) are
    skipped here and covered by the completeness test below.
    """
    result = _parse_the_fixture()
    cases = []
    for token in result.tokens:
        entry = result.vocabulary_for(token) or {}
        expected = GOLDEN_FURIGANA.get(entry.get("spelling", ""))
        if expected is not None:
            cases.append(pytest.param(token, expected, id=entry["spelling"]))
    return cases


@pytest.mark.parametrize(("token", "expected"), _golden_tokens())
def test_furigana_to_anki_golden(token: dict[str, Any], expected: str) -> None:
    assert furigana_to_anki(token["furigana"]) == expected


def test_the_fixture_carries_every_golden_case() -> None:
    result = _parse_the_fixture()
    spellings = {
        (result.vocabulary_for(token) or {}).get("spelling") for token in result.tokens
    }

    assert set(GOLDEN_FURIGANA) <= spellings


def test_furigana_of_an_all_kana_token_is_empty_when_jpdb_sends_nothing() -> None:
    # A token with no kanji may come back with a null furigana instead of a
    # single kana segment; "no markup needed" is the same answer either way.
    assert furigana_to_anki(None) == ""
    assert furigana_to_anki([]) == ""


def test_furigana_segments_can_arrive_as_a_bare_string() -> None:
    assert furigana_to_anki("たべる") == "たべる"


def test_a_segment_whose_reading_repeats_its_text_gets_no_brackets() -> None:
    assert furigana_to_anki([["ひらがな", "ひらがな"], ["語", "ご"]]) == "ひらがな 語[ご]"


def test_an_unreadable_furigana_segment_is_refused_rather_than_dropped() -> None:
    with pytest.raises(JpdbError, match="segment"):
        furigana_to_anki([["話", "はな", "extra"]])
    with pytest.raises(JpdbError, match="segment"):
        furigana_to_anki([{"text": "話", "reading": "はな"}])


# --- forced furigana spans ---------------------------------------------------


@pytest.mark.parametrize(
    ("encoding", "length"),
    [("utf16", 3), ("utf8", 7), ("utf32", 2)],
)
def test_a_span_over_a_surrogate_pair_is_measured_in_the_named_units(
    encoding: str, length: int
) -> None:
    # 𠮟る: 𠮟 is U+20B9F — two UTF-16 code units, four UTF-8 bytes, one code
    # point. Getting this wrong points the forced reading at the wrong span.
    assert forced_furigana_span("𠮟る", "しかる", encoding) == [[0, length, "しかる"]]


def test_an_ascii_safe_span_reads_the_same_in_every_encoding() -> None:
    spans = {encoding: forced_furigana_span("OK", "おっけー", encoding)[0][1] for encoding in
             ("utf8", "utf16", "utf32")}

    assert set(spans.values()) == {2}


def test_encoded_length_is_the_same_measure_the_span_uses() -> None:
    assert encoded_length("日本語", "utf16") == 3
    assert encoded_length("日本語", "utf8") == 9


def test_a_span_needs_a_reading_to_force() -> None:
    with pytest.raises(JpdbError, match="reading"):
        forced_furigana_span("話す", "")


def test_an_unknown_encoding_is_refused() -> None:
    with pytest.raises(JpdbError, match="position_length_encoding"):
        forced_furigana_span("話す", "はなす", "shift-jis")


# --- JMDict part-of-speech tables -------------------------------------------


@pytest.mark.parametrize(
    ("codes", "group"),
    [
        (["v5s", "vt"], GODAN),
        (["v5k-s"], GODAN),  # 行く
        (["v5r-i"], GODAN),  # ある
        (["v1", "vt"], ICHIDAN),
        (["v1-s"], ICHIDAN),  # くれる
        (["n", "vs"], SURU),
        (["vs-i"], SURU),
        (["vk"], KURU),
        ("v5u", GODAN),  # a bare code, not a list
        (["n"], ""),
        (["adj-i"], ""),
        (["vz"], ""),  # no conjugation table for ずる verbs: flagged, not guessed
        (None, ""),
    ],
)
def test_pos_to_verb_group(codes: Any, group: str) -> None:
    assert pos_to_verb_group(codes) == group


@pytest.mark.parametrize(
    ("codes", "label"),
    [
        (["v5s", "vt"], "verb"),
        (["vt", "v5k"], "verb"),  # vt says transitivity, not part of speech
        (["v1"], "verb"),
        (["vk"], "verb"),
        (["vs-i"], "verb"),
        (["n", "vs"], "noun"),  # a noun that takes する is still a noun
        (["adj-i"], "i-adjective"),
        (["adj-na"], "na-adjective"),
        (["adv", "n"], "adverb"),
        (["prt"], "particle"),
        (["exp"], "expression"),
        (["unc"], ""),
        ([], ""),
    ],
)
def test_pos_to_part_of_speech(codes: Any, label: str) -> None:
    assert pos_to_part_of_speech(codes) == label


def test_pos_to_transitivity() -> None:
    assert pos_to_transitivity(["v5s", "vt"]) == "transitive"
    assert pos_to_transitivity(["v5r", "vi"]) == "intransitive"
    assert pos_to_transitivity(["n"]) == ""


def test_pos_tables_skip_junk_instead_of_failing_an_import() -> None:
    assert pos_to_verb_group([None, 7, "V5R"]) == GODAN
    assert pos_to_part_of_speech(7) == ""


# --- CLI: janki jpdb ping ----------------------------------------------------


class _FakePingClient:
    instances: list[_FakePingClient] = []

    def __init__(self, api_key: str, transport: Any = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.pings = 0
        _FakePingClient.instances.append(self)

    def ping(self) -> dict[str, Any]:
        self.pings += 1
        return {}


def test_jpdb_ping_reports_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "key-from-the-environment")
    _FakePingClient.instances = []
    monkeypatch.setattr(cli.jpdb, "JpdbClient", _FakePingClient)

    assert cli.main(["jpdb", "ping"]) == 0

    assert _FakePingClient.instances[0].api_key == "key-from-the-environment"
    assert _FakePingClient.instances[0].pings == 1
    assert "ok" in capsys.readouterr().out


def test_jpdb_ping_without_a_key_says_which_variable_to_set(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    assert cli.main(["jpdb", "ping"]) == 1

    assert API_KEY_ENV in capsys.readouterr().err


def test_api_key_from_env_reads_and_trims() -> None:
    assert api_key_from_env({API_KEY_ENV: " abc \n"}) == "abc"
    with pytest.raises(JpdbError, match=API_KEY_ENV):
        api_key_from_env({})


@pytest.mark.parametrize(
    "boom",
    [
        http.client.IncompleteRead(b"partial", 5000),
        ConnectionResetError(54, "Connection reset by peer"),
    ],
    ids=["http-exception", "os-error"],
)
def test_a_status_survives_an_error_body_that_will_not_read(
    boom: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same defect the VOICEVOX transport had: `exc.read()` is live I/O inside
    the handler, where the sibling `except` clauses cannot catch it. A proxy
    answering 502 with a Content-Length it does not honour would escape as a
    raw IncompleteRead rather than a JpdbError."""

    class Stalling(io.BytesIO):
        def read(self, *args: object) -> bytes:
            raise boom

    def fake_urlopen(request: object, timeout: float = 0) -> object:
        raise urllib.error.HTTPError(
            "https://jpdb.io/api/v1/ping", 502, "Bad Gateway", {}, Stalling()
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    status, payload = jpdb.urllib_transport("https://jpdb.io/api/v1/ping", {}, {})

    assert status == 502


def test_a_peer_that_hangs_up_is_a_jpdb_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """`urlopen` wraps only the request in URLError, so a peer that accepts the
    connection and closes it without answering arrives as a bare
    RemoteDisconnected — neither a URLError nor a TimeoutError."""

    def fake_urlopen(request: object, timeout: float = 0) -> object:
        raise http.client.RemoteDisconnected("closed without response")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(jpdb.JpdbError):
        jpdb.urllib_transport("https://jpdb.io/api/v1/ping", {}, {})


@pytest.mark.parametrize(
    ("word", "codes"),
    [
        ("いる", ["aux-v", "vi", "v1"]),
        ("する", ["aux-v", "vi", "suf", "vt", "vs"]),
        ("なる", ["aux-v", "vi", "v5", "v5r"]),
        ("来る", ["aux-v", "vi", "vk"]),
        ("見る", ["aux-v", "vt", "v1"]),
        ("行く", ["aux-v", "vi", "v5", "v5k-s"]),
        ("分かる", ["int", "vi", "v5", "v5r"]),
    ],
)
def test_a_verb_code_outranks_an_auxiliary_or_interjection_label(
    word: str, codes: list[str]
) -> None:
    """These are the code lists jpdb really returns for these words, measured
    against the live API. It does not order them by significance: 〜ている and
    分かる! are real but secondary uses, and they arrive ahead of the verb code
    for the word the card is about. Taking the first recognized code labelled
    six of twenty ordinary verbs "auxiliary verb" or "interjection", each one
    contradicting the verb group on its own card."""
    assert pos_to_part_of_speech(codes) == "verb", word


@pytest.mark.parametrize(
    ("codes", "label"),
    [
        (["aux-v"], "auxiliary verb"),
        (["int"], "interjection"),
        (["suf"], "suffix"),
        (["n", "vs"], "noun"),
        (["adj-na", "n"], "na-adjective"),
    ],
    ids=["a-real-auxiliary", "a-real-interjection", "a-real-suffix", "suru-noun", "na-adj"],
)
def test_a_label_with_no_verb_code_behind_it_stands(codes: list[str], label: str) -> None:
    """Only the auxiliary/interjection/suffix family is outranked, and only when
    a verb code follows it. た really is an auxiliary verb, 〜的 really is a
    suffix, and a noun that takes する is still a noun — that ordering is jpdb
    saying which word it is, not which sense came first."""
    assert pos_to_part_of_speech(codes) == label
