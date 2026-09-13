"""The keyed dictionary fact book: record once, replay any number of times.

Contracts §7.3. Two free jpdb reads decide visible content — the promote
reading witness and dictionary enrichment — and a study finish has to be able
to show the owner exactly what they approved and then apply it without asking
the dictionary again. That needs a **keyed request → response mapping**, not a
consumptive sequence: the same expression is parsed once by the witness and
again by enrichment, and the promotion fold visits parts in an order the
preview does not repeat.

Nothing here reaches the network, and nothing here reads Japanese: the fake
below answers from a script and records what it was asked.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from japanese_anki import jpdb
from japanese_anki.enrich import (
    DictionaryFactBook,
    EnrichError,
    RecordingDictionaryClient,
    ReplayDictionaryClient,
    enrich_records,
)
from japanese_anki.models import SourceReference, VocabularyRecord

HANASU = {
    "vid": 1562350,
    "sid": 4280520068,
    "spelling": "話す",
    "reading": "はなす",
    "pitch_accent": ["LHLL"],
    "frequency_rank": 200,
    "part_of_speech": ["vt", "v5", "v5s"],
}
HANASU_TOKEN = {"vocabulary_index": 0, "furigana": [["話", "はな"], "す"]}


def parse_result(*entries: dict[str, Any]) -> jpdb.ParseResult:
    """One ``/parse`` answer, in the exact shape `JpdbClient.parse` returns."""
    return jpdb.ParseResult(
        tokens=[
            {"vocabulary_index": index, "furigana": []}
            for index, _entry in enumerate(entries)
        ],
        vocabulary=[dict(entry) for entry in entries],
    )


def hanasu_result() -> jpdb.ParseResult:
    return jpdb.ParseResult(
        tokens=[dict(HANASU_TOKEN)],
        vocabulary=[dict(HANASU)],
    )


class ScriptedDictionary:
    """A `DictionaryLookup` that answers from a script and logs every call.

    It records the arguments it actually received, so a test can prove that a
    generator was materialised *before* the transport saw it rather than
    consumed twice.
    """

    def __init__(
        self,
        parses: dict[str, list[jpdb.ParseResult]] | None = None,
        lookups: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self._parses = {key: list(value) for key, value in (parses or {}).items()}
        self._lookups = list(lookups or [])
        self.parse_calls: list[dict[str, Any]] = []
        self.lookup_calls: list[dict[str, Any]] = []

    def parse(
        self,
        text: str,
        *,
        token_fields: Any = jpdb.DEFAULT_TOKEN_FIELDS,
        vocabulary_fields: Any = jpdb.DEFAULT_VOCABULARY_FIELDS,
        forced_furigana: Any = None,
        encoding: str = jpdb.DEFAULT_ENCODING,
    ) -> jpdb.ParseResult:
        self.parse_calls.append(
            {
                "text": text,
                "token_fields": list(token_fields),
                "vocabulary_fields": list(vocabulary_fields),
                "forced_furigana": (
                    None
                    if forced_furigana is None
                    else [list(span) for span in forced_furigana]
                ),
                "encoding": encoding,
            }
        )
        answers = self._parses.get(text)
        if not answers:
            raise AssertionError(f"the script has no /parse answer for {text!r}")
        return answers[0] if len(answers) == 1 else answers.pop(0)

    def lookup_vocabulary(
        self,
        pairs: Any,
        fields: Any = jpdb.DEFAULT_LOOKUP_FIELDS,
        *,
        batch_size: int = jpdb.DEFAULT_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        self.lookup_calls.append(
            {
                "pairs": [list(pair) for pair in pairs],
                "fields": list(fields),
                "batch_size": batch_size,
            }
        )
        if not self._lookups:
            raise AssertionError("the script has no lookup-vocabulary answer left")
        return self._lookups[0] if len(self._lookups) == 1 else self._lookups.pop(0)


def record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "source": SourceReference(type="shirabe", imported_from="export.csv"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


# --- keyed, never consumptive -------------------------------------------------


def test_one_bound_query_answers_any_number_of_times_in_any_order() -> None:
    """The book is a mapping, not a queue: the witness parses an expression and
    enrichment parses it again, and the fold visits parts in an order the
    preview does not repeat."""
    live = ScriptedDictionary(
        parses={"話す": [hanasu_result()], "本": [parse_result({"spelling": "本"})]},
    )
    recorder = RecordingDictionaryClient(live)
    recorder.parse("話す")
    recorder.parse("本")

    book = recorder.freeze()
    replay = ReplayDictionaryClient(book)

    # Out of recording order, and each key several times.
    assert replay.parse("本").vocabulary[0]["spelling"] == "本"
    assert replay.parse("話す").vocabulary[0]["spelling"] == "話す"
    assert replay.parse("話す").vocabulary[0]["reading"] == "はなす"
    assert replay.parse("本").vocabulary[0]["spelling"] == "本"
    assert len(live.parse_calls) == 2, "replay never reaches the live client"


def test_the_recording_client_asks_the_live_dictionary_once_per_key() -> None:
    live = ScriptedDictionary(parses={"話す": [hanasu_result()]})
    recorder = RecordingDictionaryClient(live)

    recorder.parse("話す")
    recorder.parse("話す")
    recorder.parse("話す")

    assert len(live.parse_calls) == 3, "recording answers live; it is not a cache"
    assert len(recorder.freeze()) == 1, "one key, one fact"


def test_default_arguments_and_the_explicit_same_values_are_one_key() -> None:
    """A replay client can only canonicalise *effective* arguments because the
    Protocol writes the real defaults down."""
    live = ScriptedDictionary(parses={"話す": [hanasu_result()]})
    recorder = RecordingDictionaryClient(live)

    recorder.parse("話す")
    recorder.parse(
        "話す",
        token_fields=jpdb.DEFAULT_TOKEN_FIELDS,
        vocabulary_fields=jpdb.DEFAULT_VOCABULARY_FIELDS,
        forced_furigana=None,
        encoding=jpdb.DEFAULT_ENCODING,
    )

    assert len(recorder.freeze()) == 1


def test_a_positional_field_list_and_the_same_keyword_one_are_one_key() -> None:
    """`_readings_for` passes `fields` positionally, so the two forms have to
    canonicalise identically or the replay misses its own recording."""
    live = ScriptedDictionary(lookups=[[{"reading": "はなす", "alt_sids": []}]])
    recorder = RecordingDictionaryClient(live)

    recorder.lookup_vocabulary([[1, 2]], ("reading", "alt_sids"))
    recorder.lookup_vocabulary([[1, 2]], fields=("reading", "alt_sids"))

    assert len(recorder.freeze()) == 1


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda client: client.parse("話す", forced_furigana=[[0, 2, "はなす"]]),
            id="forced_furigana",
        ),
        pytest.param(
            lambda client: client.parse("話す", encoding="utf8"), id="encoding"
        ),
        pytest.param(
            lambda client: client.parse("話す", token_fields=("furigana",)),
            id="token_fields",
        ),
        pytest.param(
            lambda client: client.parse("話す", vocabulary_fields=("spelling",)),
            id="vocabulary_fields",
        ),
    ],
)
def test_every_explicit_parse_argument_is_part_of_the_key(call: Any) -> None:
    live = ScriptedDictionary(parses={"話す": [hanasu_result(), hanasu_result()]})
    recorder = RecordingDictionaryClient(live)

    recorder.parse("話す")
    call(recorder)

    assert len(recorder.freeze()) == 2, "a different effective request is a second fact"


def test_batch_size_and_fields_are_part_of_the_lookup_key() -> None:
    live = ScriptedDictionary(
        lookups=[[{"reading": "はなす"}], [{"reading": "はなす"}], [{"reading": "x"}]]
    )
    recorder = RecordingDictionaryClient(live)

    recorder.lookup_vocabulary([[1, 2]], ("reading",))
    recorder.lookup_vocabulary([[1, 2]], ("reading",), batch_size=1)
    recorder.lookup_vocabulary([[1, 2]], ("alt_sids",))

    assert len(recorder.freeze()) == 3


def test_an_effective_iterable_argument_is_snapshotted_before_the_transport() -> None:
    """A generator reaches the transport whole, and the key holds every pair.

    Recording after the transport consumed the iterable would key the request
    on an empty list and hand the live client nothing to look up.
    """
    live = ScriptedDictionary(lookups=[[{"reading": "はなす"}, {"reading": "ほん"}]])
    recorder = RecordingDictionaryClient(live)

    recorder.lookup_vocabulary(
        ([vid, sid] for vid, sid in ((1, 2), (3, 4))), ("reading",)
    )

    assert live.lookup_calls[0]["pairs"] == [[1, 2], [3, 4]]
    (method, arguments), _response = next(iter(recorder.freeze().items()))
    assert method == "lookup_vocabulary"
    assert json.loads(arguments)["pairs"] == [[1, 2], [3, 4]]


def test_a_tuple_pair_and_a_list_pair_are_the_same_effective_request() -> None:
    live = ScriptedDictionary(lookups=[[{"reading": "はなす"}]])
    recorder = RecordingDictionaryClient(live)

    recorder.lookup_vocabulary([[1, 2]], ("reading",))
    recorder.lookup_vocabulary([(1, 2)], ("reading",))

    assert len(recorder.freeze()) == 1


# --- the response wire is immutable against its callers -----------------------


def test_a_caller_that_mutates_a_recorded_answer_cannot_alter_the_book() -> None:
    live = ScriptedDictionary(parses={"話す": [hanasu_result()]})
    recorder = RecordingDictionaryClient(live)

    answer = recorder.parse("話す")
    answer.vocabulary[0]["spelling"] = "tampered"
    answer.tokens.clear()

    replayed = ReplayDictionaryClient(recorder.freeze()).parse("話す")
    assert replayed.vocabulary[0]["spelling"] == "話す"
    assert replayed.tokens == [HANASU_TOKEN]


def test_each_replay_hands_back_its_own_clone() -> None:
    live = ScriptedDictionary(parses={"話す": [hanasu_result()]})
    recorder = RecordingDictionaryClient(live)
    recorder.parse("話す")
    replay = ReplayDictionaryClient(recorder.freeze())

    first = replay.parse("話す")
    first.vocabulary[0]["reading"] = "tampered"
    second = replay.parse("話す")

    assert second.vocabulary[0]["reading"] == "はなす"
    assert first.vocabulary is not second.vocabulary


def test_the_recording_client_hands_back_exactly_what_replay_will() -> None:
    """The preview pass and every later pass see the same answer.

    The recorded answer is handed back as its own clone, so a caller editing
    what it got cannot reach the dictionary client's own state — and the pass
    that renders the preview and the pass that applies it therefore cannot
    diverge.
    """
    live = ScriptedDictionary(parses={"話す": [hanasu_result()]})
    recorder = RecordingDictionaryClient(live)

    recorded = recorder.parse("話す")
    recorded.vocabulary[0]["spelling"] = "tampered"
    replayed = ReplayDictionaryClient(recorder.freeze()).parse("話す")
    again = recorder.parse("話す")

    assert again == replayed
    assert again.vocabulary[0]["spelling"] == "話す"


def test_a_replayed_parse_is_a_real_parse_result_and_a_lookup_a_list_of_dicts() -> None:
    live = ScriptedDictionary(
        parses={"話す": [hanasu_result()]},
        lookups=[[{"vid": 1, "sid": 2, "reading": "はなす", "alt_sids": None}]],
    )
    recorder = RecordingDictionaryClient(live)
    recorder.parse("話す")
    recorder.lookup_vocabulary([[1, 2]], ("reading", "alt_sids"))
    replay = ReplayDictionaryClient(recorder.freeze())

    parsed = replay.parse("話す")
    rows = replay.lookup_vocabulary([[1, 2]], ("reading", "alt_sids"))

    assert isinstance(parsed, jpdb.ParseResult)
    assert parsed.vocabulary_for(parsed.tokens[0]) == HANASU
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["alt_sids"] is None, "a null field is a real answer"


def test_an_empty_result_is_recorded_and_replayed_as_an_empty_result() -> None:
    live = ScriptedDictionary(
        parses={"？": [jpdb.ParseResult(tokens=[], vocabulary=[])]}, lookups=[[]]
    )
    recorder = RecordingDictionaryClient(live)
    recorder.parse("？")
    recorder.lookup_vocabulary([[9, 9]], ("reading",))
    replay = ReplayDictionaryClient(recorder.freeze())

    assert replay.parse("？") == jpdb.ParseResult(tokens=[], vocabulary=[])
    assert replay.lookup_vocabulary([[9, 9]], ("reading",)) == []


# --- conflicts, and the evidence they keep ------------------------------------


def test_two_conflicting_answers_for_one_key_refuse_at_freeze_and_keep_both() -> None:
    """The recording client still holds both, so a person can see what moved."""
    live = ScriptedDictionary(
        parses={
            "話す": [
                hanasu_result(),
                jpdb.ParseResult(
                    tokens=[dict(HANASU_TOKEN)],
                    vocabulary=[{**HANASU, "frequency_rank": 999}],
                ),
            ]
        }
    )
    recorder = RecordingDictionaryClient(live)
    recorder.parse("話す")
    recorder.parse("話す")

    with pytest.raises(EnrichError, match="dictionary-fact-conflict"):
        recorder.freeze()

    (conflict,) = recorder.conflicts
    assert conflict.method == "parse"
    assert len(conflict.responses) == 2
    assert [json.loads(item)["vocabulary"][0]["frequency_rank"]
            for item in conflict.responses] == [200, 999]
    with pytest.raises(EnrichError, match="dictionary-fact-conflict"):
        recorder.freeze()


def test_an_identical_repeated_answer_is_not_a_conflict() -> None:
    live = ScriptedDictionary(parses={"話す": [hanasu_result(), hanasu_result()]})
    recorder = RecordingDictionaryClient(live)

    recorder.parse("話す")
    recorder.parse("話す")

    assert recorder.conflicts == ()
    assert len(recorder.freeze()) == 1


# --- replay refuses what it was never told ------------------------------------


def test_an_unrecorded_key_raises_rather_than_reaching_the_network() -> None:
    live = ScriptedDictionary(parses={"話す": [hanasu_result()]})
    recorder = RecordingDictionaryClient(live)
    recorder.parse("話す")
    replay = ReplayDictionaryClient(recorder.freeze())

    with pytest.raises(EnrichError, match="dictionary-fact-missing"):
        replay.parse("本")
    with pytest.raises(EnrichError, match="dictionary-fact-missing"):
        replay.parse("話す", forced_furigana=[[0, 2, "はなす"]])
    with pytest.raises(EnrichError, match="dictionary-fact-missing"):
        replay.lookup_vocabulary([[1, 2]], ("reading",))

    assert len(live.parse_calls) == 1, "no replay refusal fell through to the client"


# --- the frozen book is durable authority -------------------------------------


def test_a_frozen_book_round_trips_and_its_fingerprint_binds_every_answer() -> None:
    live = ScriptedDictionary(
        parses={"話す": [hanasu_result()]},
        lookups=[[{"vid": 1, "sid": 2, "reading": "はなす"}]],
    )
    recorder = RecordingDictionaryClient(live)
    recorder.parse("話す")
    recorder.lookup_vocabulary([[1, 2]], ("reading",))
    book = recorder.freeze()

    restored = DictionaryFactBook.from_dict(json.loads(json.dumps(book.to_dict())))

    assert restored == book
    assert len(restored) == 2
    assert restored.keys() == tuple(key for key, _response in restored.items())
    assert all(key in restored for key in tuple(restored.keys()))
    assert ("parse", "{}") not in restored
    assert restored.fingerprint == book.fingerprint
    assert len(book.fingerprint) == 64
    assert ReplayDictionaryClient(restored).parse("話す").vocabulary[0] == HANASU

    moved = DictionaryFactBook(
        facts=tuple(
            (method, arguments, response.replace("200", "201"))
            for method, arguments, response in book.facts
        )
    )
    assert moved.fingerprint != book.fingerprint


def test_a_book_refuses_two_answers_for_one_key_and_an_unknown_method() -> None:
    with pytest.raises(EnrichError, match="dictionary-fact-conflict"):
        DictionaryFactBook(facts=(("parse", "{}", "a"), ("parse", "{}", "b")))
    with pytest.raises(EnrichError, match="dictionary fact book"):
        DictionaryFactBook(facts=(("ping", "{}", "a"),))


# --- the whole enrichment pass runs off a replayed book -----------------------


def test_a_recorded_pass_replays_into_the_same_enrichment_result() -> None:
    """What the owner approved is what the apply writes, with no refetch."""
    live = ScriptedDictionary(parses={"話す": [hanasu_result()]})
    recorder = RecordingDictionaryClient(live)

    recorded = enrich_records(recorder, [record()])
    book = recorder.freeze()
    replayed = enrich_records(ReplayDictionaryClient(book), [record()])

    assert recorded.records[0].furigana == "話[はな]す"
    assert replayed.records[0] == recorded.records[0]
    assert replayed.changes == recorded.changes
    assert len(live.parse_calls) == 1, "the replayed pass asked nobody anything"


def test_both_clients_account_for_every_key_they_were_asked_for() -> None:
    """A finish reports what it fetched and what it replayed, and the two are
    the same keys — which is how "no phase refetches after review" is checked
    rather than asserted."""
    live = ScriptedDictionary(parses={"話す": [hanasu_result()]})
    recorder = RecordingDictionaryClient(live)
    recorder.parse("話す")
    recorder.parse("話す")
    book = recorder.freeze()
    replay = ReplayDictionaryClient(book)
    replay.parse("話す")

    assert len(recorder.calls) == 2
    assert set(recorder.calls) == set(replay.calls) == set(book.keys())
    assert len(live.parse_calls) == 2, "the replay added no live call"
