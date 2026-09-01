"""Planning contract for the deterministic per-record Realtime voice pool.

The pool resolves to ordinary concrete speech profiles before currency or WAL
logic sees an example.  No test in this module opens a provider connection.
"""

from __future__ import annotations

from pathlib import Path

from japanese_anki import audio_cmd
from japanese_anki import ledger as ledger_mod
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.tts.openai_realtime import OpenAiRealtimePool, voice_for_record


class WordVoice:
    name = "word-fake"
    voice = 13
    speed = 1.0
    suffix = ".wav"
    settings: dict[str, str] = {}


def _never_connect(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("audio planning must not call the Realtime transport")


def _record(
    record_id: str,
    expression: str,
    reading: str,
    examples: list[ExampleSentence],
) -> VocabularyRecord:
    return VocabularyRecord(
        id=record_id,
        expression=expression,
        reading=reading,
        meanings=["test"],
        examples=examples,
        source=SourceReference(type="manual", imported_from="test"),
    )


def _pool() -> OpenAiRealtimePool:
    return OpenAiRealtimePool(api_key="unused-test-key", transport=_never_connect)


def test_every_example_for_one_record_gets_the_same_concrete_profile() -> None:
    item = _record(
        "word:橋:はし",
        "橋",
        "はし",
        [
            ExampleSentence(japanese="橋を渡ります。", english="I cross the bridge."),
            ExampleSentence(japanese="この橋は長い。", english="This bridge is long."),
        ],
    )

    prepared = audio_cmd.prepare_example_audio_profiles(
        [item], sentence_provider=_pool()
    )

    profiles = prepared[item.id]
    assert len(profiles) == 2
    assert all(profile.voice == voice_for_record(item.id) for profile in profiles)
    assert all(profile.name == "openai-realtime" for profile in profiles)
    assert all(profile.suffix == ".wav" for profile in profiles)
    assert profiles[0] is profiles[1], "one record resolves once, not per example"


def test_selection_is_independent_of_record_and_example_order() -> None:
    first = _record(
        "word:橋:はし",
        "橋",
        "はし",
        [ExampleSentence(japanese="一。"), ExampleSentence(japanese="二。")],
    )
    second = _record(
        "word:飛行機:ひこうき",
        "飛行機",
        "ひこうき",
        [ExampleSentence(japanese="三。"), ExampleSentence(japanese="四。")],
    )
    pool = _pool()

    forward = audio_cmd.prepare_example_audio_profiles(
        [first, second], sentence_provider=pool
    )
    reversed_records = audio_cmd.prepare_example_audio_profiles(
        [second, first], sentence_provider=pool
    )
    reordered_first = _record(
        first.id,
        first.expression,
        first.reading,
        list(reversed(first.examples)),
    )
    reordered = audio_cmd.prepare_example_audio_profiles(
        [reordered_first], sentence_provider=pool
    )

    assert forward[first.id][0].voice == reversed_records[first.id][0].voice
    assert forward[second.id][0].voice == reversed_records[second.id][0].voice
    assert {profile.voice for profile in reordered[first.id]} == {
        voice_for_record(first.id)
    }
    assert forward[first.id][0].voice == voice_for_record(first.id)
    assert forward[second.id][0].voice == voice_for_record(second.id)


def test_current_pending_keys_use_each_exact_selected_profile(tmp_path: Path) -> None:
    first_example = ExampleSentence(japanese="橋を渡ります。")
    second_example = ExampleSentence(japanese="この橋は長い。")
    item = _record(
        "word:橋:はし",
        "橋",
        "",
        [first_example, second_example],
    )
    book = ledger_mod.Ledger(path=tmp_path / "ledger.json")
    prepared = audio_cmd.prepare_example_audio_profiles(
        [item], sentence_provider=_pool()
    )

    keys = audio_cmd.current_pending_audio_keys(
        [item],
        book=book,
        word_provider=WordVoice(),
        prepared_examples=prepared,
    )

    expected: set[str] = set()
    for example, selected in zip(
        item.examples, prepared[item.id], strict=True
    ):
        expected.add(
            book.pending_audio_key_for(
                item.id,
                of="example",
                target=(
                    "janki-"
                    f"{ledger_mod.example_audio_filename_fingerprint(item, example)}"
                    f"{selected.suffix}"
                ),
                request_input=ledger_mod.example_audio_request(example),
                forced_accent=False,
                content_fp=ledger_mod.example_audio_content_fingerprint(example),
                provider=selected.name,
                voice=selected.voice,
                speed=selected.speed,
                settings=selected.settings,
            )
        )
    assert keys == expected
    assert all(profile.voice == voice_for_record(item.id) for profile in prepared[item.id])
