"""Batched AI enrichment — ``janki enrich --ai --batch-submit`` / ``--batch-fetch``.

The Message Batches API is the same request at half price, answered within a day
instead of within seconds — so these tests are mostly about the gap in between:
what the ledger has to remember across it, what happens when the collection
moves while a batch is out, and what a result that never arrived costs. No
network (IMPLEMENTATION_PLAN rule 6): the batch endpoints are faked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import claude_client, cli, enrich, ledger
from japanese_anki.errors import JankiError
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.staging import read_staging


def record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "verb_group": "godan",
        "source": SourceReference(type="shirabe", imported_from="export.csv"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def many(count: int) -> list[VocabularyRecord]:
    return [
        record(id=f"word:話す{index}:はなす", expression=f"話す{index}")
        for index in range(count)
    ]


def ok(item: VocabularyRecord) -> Entry:
    """A succeeded result for one record, carrying an example of its own word."""
    tail = item.expression.removeprefix("話す")
    return Entry(
        key(item.id),
        Succeeded(message(f"毎日{item.expression}。", f"毎日[まいにち] 話[はな]す{tail}。")),
    )


def message(japanese: str, furigana: str = "") -> Any:
    """A Message the way the SDK hands one back: text blocks and a stop reason."""

    class Block:
        type = "text"
        text = json.dumps(
            {
                "examples": [
                    {"japanese": japanese, "furigana": furigana, "english": "", "romaji": ""}
                ],
                "usage_notes": "",
            },
            ensure_ascii=False,
        )

    class Message:
        stop_reason = "end_turn"
        content = [Block()]

    return Message()


class Entry:
    """One row of ``messages.batches.results``."""

    def __init__(self, custom_id: str, outcome: Any) -> None:
        self.custom_id = custom_id
        self.result = outcome


class Succeeded:
    type = "succeeded"

    def __init__(self, msg: Any) -> None:
        self.message = msg


class Failed:
    """A non-succeeded row, shaped the way the SDK shapes it.

    ``MessageBatchErroredResult.error`` is an ``ErrorResponse`` whose own
    ``type`` is the literal ``"error"``; the reason worth printing lives one
    level further in. A flatter fake would let janki read the outer level and
    still pass.
    """

    def __init__(self, kind: str, message: str = "") -> None:
        self.type = kind
        if kind != "errored":
            # MessageBatchExpiredResult and MessageBatchCanceledResult have a
            # type and nothing else. Giving them an error chain would let a
            # test assert on a reason the API never sends.
            return
        inner = type(
            "ErrorObject", (), {"type": "overloaded_error", "message": message}
        )()
        self.error = type("ErrorResponse", (), {"type": "error", "error": inner})()


class Batch:
    def __init__(self, batch_id: str, status: str) -> None:
        self.id = batch_id
        self.processing_status = status


class FakeBatches:
    """``client.messages.batches``, recording what it was asked to do."""

    def __init__(
        self,
        *,
        batch_id: str = "msgbatch_01",
        status: str = "ended",
        results: list[Entry] | None = None,
    ) -> None:
        self.batch_id = batch_id
        self.status = status
        self.results_rows = results or []
        self.submitted: list[list[dict[str, Any]]] = []
        self.retrieved: list[str] = []

    def create(self, requests: Any) -> Batch:
        self.submitted.append(list(requests))
        return Batch(self.batch_id, "in_progress")

    def retrieve(self, batch_id: str) -> Batch:
        self.retrieved.append(batch_id)
        return Batch(batch_id, self.status)

    def results(self, batch_id: str) -> list[Entry]:
        return self.results_rows


class FakeJpdb:
    """Answers /parse with nothing, the way a jpdb outage does — which means
    every example comes back unverified rather than confirmed."""

    def parse(self, text: str, **kw: Any) -> Any:
        raise JankiError(f"no parse for {text}")


def fake_sdk() -> Any:
    """Stands in for the ``anthropic`` module: only ``transform_schema`` is used."""
    return type(
        "A",
        (),
        {"transform_schema": staticmethod(lambda schema: {"of": schema.__name__})},
    )


def project(tmp_path: Path, records: list[VocabularyRecord]) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "JAPANESE_STYLE_GUIDE.md").write_text("Guide.", encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([item.to_dict() for item in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return tmp_path


def patch_all(monkeypatch: pytest.MonkeyPatch, batches: FakeBatches) -> None:
    monkeypatch.setenv("JPDB_API_KEY", "k")
    client = type("Client", (), {"messages": type("M", (), {"batches": batches})()})()
    monkeypatch.setattr(claude_client, "build_client", lambda *a, **kw: client)
    monkeypatch.setattr(
        cli.jpdb, "JpdbClient", lambda key, *a, **kw: FakeJpdb()
    )
    # transform_schema comes from the SDK; the fake never sends it anywhere.
    monkeypatch.setattr(claude_client, "load_anthropic", fake_sdk)


def stored(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload}


def book_of(root: Path) -> dict[str, Any]:
    return json.loads((root / "ledger.json").read_text(encoding="utf-8"))


def key(record_id: str) -> str:
    return enrich.batch_custom_id(record_id)


# --- keys --------------------------------------------------------------------


def test_a_record_id_is_not_a_custom_id() -> None:
    # The API wants a short ASCII identifier; 'word:話す:はなす' is neither.
    custom = key("word:話す:はなす")

    assert custom.isascii() and custom.isalnum()
    assert len(custom) <= 64


def test_the_key_map_refuses_a_collision_rather_than_resolving_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two words sharing a key would attach one's examples to the other."""
    monkeypatch.setattr(enrich, "batch_custom_id", lambda record_id: "rsame")

    with pytest.raises(enrich.EnrichError) as caught:
        enrich.batch_key_map(["word:話す:はなす", "word:聞く:きく"])

    assert "rsame" in str(caught.value)


# --- the requests ------------------------------------------------------------


def test_a_batch_request_is_the_same_request_a_live_call_makes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claude_client, "load_anthropic", fake_sdk)
    blocks = claude_client.system_blocks("guide", "instructions")
    sent: dict[str, Any] = {}

    class Client:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kw: Any) -> Any:
                sent.update(kw)
                return type("R", (), {"stop_reason": "refusal", "stop_details": None})()

    claude_client.parse_call("m", blocks, "hello", dict, Client(), max_tokens=99)
    batched = claude_client.batch_request("r1", "m", blocks, "hello", dict, max_tokens=99)

    assert batched["custom_id"] == "r1"
    assert batched["params"] == sent


def test_the_batch_prompt_has_no_variety_pressure(tmp_path: Path) -> None:
    # Every request is built before any answer exists, so there is nothing for
    # a later prompt to have seen. Said out loud because it is a real
    # difference in what the two paths produce.
    requests, ids = enrich.batch_requests(many(3), model="m", style_guide="guide")

    contents = [item["params"]["messages"][0]["content"] for item in requests]
    assert len(requests) == 3
    assert ids == [item.id for item in many(3)]
    assert not any("already written in this run" in text for text in contents)


def test_the_batch_asks_for_the_long_cache_window() -> None:
    # A five-minute window does not survive the span a batch's requests are read
    # over; the style guide leads every one of them.
    requests, _ = enrich.batch_requests(many(1), model="m", style_guide="guide")

    system = requests[0]["params"]["system"]
    assert system[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


# --- submit ------------------------------------------------------------------


def test_submitting_records_the_batch_and_the_records_it_covers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, many(3))
    batches = FakeBatches()
    patch_all(monkeypatch, batches)

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-submit"]) == 0

    assert len(batches.submitted[0]) == 3
    pending = book_of(root)["pending_batches"]
    assert list(pending) == ["msgbatch_01"]
    assert pending["msgbatch_01"]["kind"] == "ai"
    assert pending["msgbatch_01"]["pending_ids"] == [item.id for item in many(3)]
    assert "msgbatch_01" in capsys.readouterr().out


def test_a_second_submit_is_refused_while_one_is_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two batches at once leave two answers for one word and no way to say
    which is current."""
    root = project(tmp_path, many(3))
    batches = FakeBatches()
    patch_all(monkeypatch, batches)
    cli.main(["--root", str(root), "enrich", "--ai", "--batch-submit"])
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-submit"]) == 1

    assert "msgbatch_01" in capsys.readouterr().err
    assert len(batches.submitted) == 1


def test_nothing_to_submit_costs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    complete = record(usage_notes="note", examples=[ExampleSentence(japanese="話す。")])
    root = project(tmp_path, [complete])
    batches = FakeBatches()
    patch_all(monkeypatch, batches)

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-submit"]) == 0

    assert batches.submitted == []
    assert "Nothing to submit" in capsys.readouterr().out


def test_a_batch_id_that_never_arrived_is_an_error_not_a_blank_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoId:
        class messages:  # noqa: N801
            class batches:  # noqa: N801
                @staticmethod
                def create(requests: Any) -> Any:
                    return type("B", (), {"id": ""})()

    with pytest.raises(cli.JankiError) as caught:
        claude_client.submit_batch([{"custom_id": "r1"}], NoId())

    assert "cannot be collected" in str(caught.value) or "fetch" in str(caught.value)


# --- fetch -------------------------------------------------------------------


def submitted(tmp_path: Path, records: list[VocabularyRecord], monkeypatch, batches):
    root = project(tmp_path, records)
    patch_all(monkeypatch, batches)
    cli.main(["--root", str(root), "enrich", "--ai", "--batch-submit"])
    return root


def test_fetching_with_nothing_pending_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, many(1))
    patch_all(monkeypatch, FakeBatches())

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch"]) == 0

    assert "No batch is pending" in capsys.readouterr().out


def test_a_batch_still_running_reports_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """"Is it ready" is a question, and "not yet" is a successful answer to it."""
    batches = FakeBatches(status="in_progress")
    root = submitted(tmp_path, many(2), monkeypatch, batches)
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch"]) == 0

    out = capsys.readouterr().out
    assert "in_progress" in out
    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]


def test_a_finished_batch_writes_the_records_and_clears_the_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = many(2)
    batches = FakeBatches(
        results=[
            ok(item) for item in records
        ]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 0

    kept = stored(root)
    assert kept["word:話す0:はなす"]["examples"]
    assert kept["word:話す1:はなす"]["examples"]
    assert book_of(root)["pending_batches"] == {}
    entries = book_of(root)["records"]["word:話す0:はなす"]["enriched"]
    assert [item["kind"] for item in entries] == ["ai"]


def test_the_fetched_answers_go_through_the_same_qc_as_a_live_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sentence that does not contain the word is not an example of it,
    whether it arrived in two seconds or two hours."""
    records = [record()]
    batches = FakeBatches(
        results=[
            Entry(
                key("word:話す:はなす"),
                Succeeded(message("犬が走る。", "犬[いぬ]が走[はし]る。")),
            )
        ]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 0

    assert stored(root)["word:話す:はなす"]["examples"] == []
    assert "did not contain" in capsys.readouterr().err


def test_a_flagged_example_is_flagged_the_same_way_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [record()]
    batches = FakeBatches(
        results=[ok(record())]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"])

    raw = stored(root)["word:話す:はなす"]["source"]["raw_fields"]
    assert enrich.UNVERIFIED_KEY in raw


def test_an_errored_result_leaves_that_record_untouched_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = many(2)
    batches = FakeBatches(
        results=[
            Entry(key(records[0].id), Failed("errored", "overloaded")),
            ok(records[1]),
        ]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 0

    err = capsys.readouterr().err
    assert "word:話す0:はなす" in err
    assert "overloaded" in err
    assert stored(root)["word:話す0:はなす"]["examples"] == []
    assert stored(root)["word:話す1:はなす"]["examples"]


def test_an_expired_result_is_reported_as_expired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = many(2)
    batches = FakeBatches(
        results=[
            Entry(key(records[0].id), Failed("expired")),
            ok(records[1]),
        ]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"])

    assert "expired" in capsys.readouterr().err


def test_a_record_the_batch_never_mentioned_is_not_quietly_forgotten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = many(2)
    batches = FakeBatches(
        results=[ok(records[1])]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"])

    err = capsys.readouterr().err
    assert "word:話す0:はなす" in err
    assert "no result" in err


def test_a_record_deleted_while_the_batch_was_out_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A batch runs for up to a day; the collection does not hold still."""
    records = many(2)
    batches = FakeBatches(
        results=[
            ok(item) for item in records
        ]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    (root / "vocabulary.json").write_text(
        json.dumps([records[1].to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 0

    err = capsys.readouterr().err
    assert "no longer in the collection" in err
    assert "word:話す0:はなす" in err
    assert stored(root)["word:話す1:はなす"]["examples"]


def test_a_result_keyed_to_nothing_is_ignored_rather_than_guessed_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = many(1)
    batches = FakeBatches(
        results=[
            ok(records[0]),
            Entry("rdeadbeef0000", Succeeded(message("話します。", "話[はな]します。"))),
        ]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"])

    assert "rdeadbeef0000" in capsys.readouterr().err


def test_a_large_batch_lands_in_staging_the_way_a_large_live_run_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = many(enrich.STAGING_THRESHOLD)
    batches = FakeBatches(
        results=[
            ok(item) for item in records
        ]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 0

    staged, _ = read_staging(root / "staging" / "ai-enrichment.yaml")
    assert len(staged) == enrich.STAGING_THRESHOLD
    assert stored(root)["word:話す0:はなす"]["examples"] == []
    assert "janki promote" in capsys.readouterr().out


def test_declining_the_diff_leaves_the_batch_pending_so_it_can_be_fetched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Results live on Anthropic's side for weeks and fetching is free, so
    forgetting the id would be the only irreversible part of the run."""
    records = many(1)
    batches = FakeBatches(
        results=[ok(records[0])]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch"]) == 1

    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]
    assert stored(root)["word:話す0:はなす"]["examples"] == []
    assert "still recorded as pending" in capsys.readouterr().err


def test_a_batch_with_nothing_to_write_is_still_collected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = many(1)
    batches = FakeBatches(results=[Entry(key(records[0].id), Failed("errored", "boom"))])
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 0

    assert book_of(root)["pending_batches"] == {}


def test_the_batchs_own_model_is_used_not_the_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run submitted under one model was answered by that one, whatever the
    config says by the time it is collected."""
    records = many(1)
    batches = FakeBatches(
        results=[ok(records[0])]
    )
    root = project(tmp_path, records)
    patch_all(monkeypatch, batches)
    cli.main(
        ["--root", str(root), "enrich", "--ai", "--batch-submit", "--model", "old-model"]
    )

    cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"])

    entries = book_of(root)["records"]["word:話す0:はなす"]["enriched"]
    assert entries[0]["model"] == "old-model"


# --- the flags belong to the AI pass -----------------------------------------


def test_the_batch_flags_need_the_ai_pass(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["enrich", "--jpdb", "--batch-submit"]) == 1
    assert "needs --ai" in capsys.readouterr().err


def test_one_batch_action_at_a_time(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["enrich", "--ai", "--batch-submit", "--batch-fetch"]) == 1
    assert "one batch action at a time" in capsys.readouterr().err
    assert cli.main(["enrich", "--ai", "--batch-fetch", "--batch-forget"]) == 1
    assert "one batch action at a time" in capsys.readouterr().err


# --- the ledger --------------------------------------------------------------


def test_a_pending_batch_needs_records_to_be_pending_on(tmp_path: Path) -> None:
    book = ledger.Ledger(path=tmp_path / "ledger.json")

    with pytest.raises(ledger.LedgerError):
        book.record_batch("b1", kind="ai", model="m", pending_ids=[])


def test_recording_a_batch_replaces_rather_than_accumulates(tmp_path: Path) -> None:
    # A batch is one thing that is either in flight or collected; two entries
    # for one id would be the ledger disagreeing with itself.
    book = ledger.Ledger(path=tmp_path / "ledger.json")
    book.record_batch("b1", kind="ai", model="m", pending_ids=["a"])
    book.record_batch("b1", kind="ai", model="m", pending_ids=["a", "b"])

    assert book.pending_batches["b1"]["pending_ids"] == ["a", "b"]


def test_an_unknown_kind_is_refused(tmp_path: Path) -> None:
    book = ledger.Ledger(path=tmp_path / "ledger.json")

    with pytest.raises(ledger.LedgerError):
        book.record_batch("b1", kind="nonsense", model="m", pending_ids=["a"])


def test_a_pending_batch_round_trips_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    book = ledger.Ledger(path=path)
    book.record_batch("b1", kind="ai", model="m", pending_ids=["a", "b"])
    book.save()

    reloaded = ledger.load(path)

    assert reloaded.pending_batch() == ("b1", book.pending_batches["b1"])
    assert reloaded.clear_batch("b1") is True
    assert reloaded.pending_batch() is None


# --- what the review found ---------------------------------------------------


def test_the_reason_a_request_errored_survives_the_sdks_nesting() -> None:
    """The outer level's type is the literal "error" for every failure. Reading
    it tells nobody deciding whether to resubmit anything at all."""

    class Client:
        class messages:  # noqa: N801
            class batches:  # noqa: N801
                @staticmethod
                def results(batch_id: str) -> list[Entry]:
                    return [Entry("r1", Failed("errored", "Overloaded"))]

    (row,) = list(claude_client.batch_results("b1", dict, "m", Client()))

    assert row.outcome == "errored"
    assert row.detail == "Overloaded"


def test_a_collection_that_moved_leaves_the_batch_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-empty file holding none of the batch's records — a --replace
    import, a promote that re-minted ids after a reading fix, a restore from
    another revision. Clearing here would drop the id of a batch whose answers
    are alive on Anthropic's side, and exit 0 doing it."""
    records = many(2)
    batches = FakeBatches(results=[ok(item) for item in records])
    root = submitted(tmp_path, records, monkeypatch, batches)
    stranger = record(id="word:聞く:きく", expression="聞く")
    (root / "vocabulary.json").write_text(
        json.dumps([stranger.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1

    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]
    assert "fetch again" in capsys.readouterr().err


def test_force_fields_survives_the_gap_between_submit_and_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Submitting with --force-fields and fetching plainly would report
    "nothing to fill" and throw away an answer that was paid for."""
    curated = record(examples=[ExampleSentence(japanese="人と話す。")], usage_notes="note")
    batches = FakeBatches(results=[ok(curated)])
    root = project(tmp_path, [curated])
    patch_all(monkeypatch, batches)
    cli.main(
        [
            "--root", str(root), "enrich", "--ai", "--batch-submit",
            "--force-fields", "examples", curated.id,
        ]
    )
    assert book_of(root)["pending_batches"]["msgbatch_01"]["force_fields"] == ["examples"]
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 0

    landed = stored(root)["word:話す:はなす"]["examples"]
    assert [item["japanese"] for item in landed] == ["毎日話す。"]


def test_fetch_refuses_the_flags_that_were_decided_at_submit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, many(1))

    assert cli.main(
        ["--root", str(root), "enrich", "--ai", "--batch-fetch", "word:話す0:はなす"]
    ) == 1
    assert "recorded in the ledger" in capsys.readouterr().err
    assert cli.main(
        ["--root", str(root), "enrich", "--ai", "--batch-fetch", "--model", "m2"]
    ) == 1
    assert "recorded in the ledger" in capsys.readouterr().err


def test_a_failed_ledger_write_says_what_actually_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A large batch's answers sit in a staging file, so "those fields are
    filled, fetching again is a no-op" would send the reader at a re-fetch that
    refuses rather than repeats."""
    records = many(enrich.STAGING_THRESHOLD)
    batches = FakeBatches(results=[ok(item) for item in records])
    root = submitted(tmp_path, records, monkeypatch, batches)
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("disk full")),
    )
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1

    err = capsys.readouterr().err
    assert cli.STAGING_FILE_NAME in err
    assert "those fields are filled" not in err


# --- what the second review found --------------------------------------------


def test_a_one_record_batch_whose_word_was_deleted_has_a_way_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """For a one-record batch, "one record vanished" and "all of them vanished"
    are the same event, and janki cannot tell them apart — so it refuses rather
    than guess, and --batch-forget is what unsticks it. Without that command
    this would be a deadlock; with it, it is a question with an answer."""
    records = many(1)
    batches = FakeBatches(results=[ok(item) for item in records])
    root = submitted(tmp_path, records, monkeypatch, batches)
    survivor = record(id="word:聞く:きく", expression="聞く")
    (root / "vocabulary.json").write_text(
        json.dumps([survivor.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1
    assert "--batch-forget" in capsys.readouterr().err

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-forget"]) == 0
    assert book_of(root)["pending_batches"] == {}
    capsys.readouterr()
    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-submit"]) == 0


def test_an_empty_collection_is_a_wrong_root_and_keeps_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = many(2)
    batches = FakeBatches(results=[ok(item) for item in records])
    root = submitted(tmp_path, records, monkeypatch, batches)
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1

    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]
    assert "--batch-forget" in capsys.readouterr().err


def test_forgetting_is_the_supported_way_out_of_a_stranded_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without it the remedy would be hand-editing data/ledger.json, which
    AGENTS.md forbids for a file janki writes."""
    records = many(1)
    root = submitted(tmp_path, records, monkeypatch, FakeBatches())
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-forget"]) == 0

    out = capsys.readouterr().out
    assert "msgbatch_01" in out
    assert "console" in out
    assert book_of(root)["pending_batches"] == {}
    # And a submit is possible again.
    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-submit"]) == 0


def test_forgetting_nothing_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, many(1))
    patch_all(monkeypatch, FakeBatches())

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-forget"]) == 0

    assert "nothing to forget" in capsys.readouterr().out


def test_force_fields_at_fetch_overrides_and_says_it_is_doing_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A field can fill while the batch is out. Unlike the model and the ids,
    this one decides how an answer already in hand is applied, so it overrides
    rather than being refused — but never in silence."""
    plain = record()
    batches = FakeBatches(results=[ok(plain)])
    root = project(tmp_path, [plain])
    patch_all(monkeypatch, batches)
    cli.main(["--root", str(root), "enrich", "--ai", "--batch-submit"])
    # A promote filled the examples while the batch was out.
    filled = record(examples=[ExampleSentence(japanese="人と話す。")])
    (root / "vocabulary.json").write_text(
        json.dumps([filled.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    code = cli.main(
        [
            "--root", str(root), "enrich", "--ai", "--batch-fetch",
            "--force-fields", "examples", "--yes",
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "instead of the none this batch was submitted with" in out
    landed = stored(root)["word:話す:はなす"]["examples"]
    assert [item["japanese"] for item in landed] == ["毎日話す。"]


def test_a_batch_that_does_not_need_jpdb_does_not_need_a_jpdb_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Submitting builds requests and forgetting writes a ledger entry; neither
    asks jpdb anything, so neither may demand a key to run."""
    root = project(tmp_path, many(1))
    batches = FakeBatches()
    patch_all(monkeypatch, batches)
    monkeypatch.delenv("JPDB_API_KEY", raising=False)

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-submit"]) == 0
    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-forget"]) == 0


def test_an_expired_row_carries_no_reason_because_the_api_sends_none() -> None:
    """MessageBatchExpiredResult has a type and nothing else."""

    class Client:
        class messages:  # noqa: N801
            class batches:  # noqa: N801
                @staticmethod
                def results(batch_id: str) -> list[Entry]:
                    return [Entry("r1", Failed("expired"))]

    (row,) = list(claude_client.batch_results("b1", dict, "m", Client()))

    assert row.outcome == "expired"
    assert row.detail == ""


def test_the_force_fields_override_is_not_announced_before_it_can_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A batch still running applies nothing, so it must claim nothing — and
    the stored list is untouched, so a later plain fetch really does revert."""
    records = many(1)
    batches = FakeBatches(status="in_progress", results=[ok(item) for item in records])
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    code = cli.main(
        [
            "--root", str(root), "enrich", "--ai", "--batch-fetch",
            "--force-fields", "examples", "--yes",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "in_progress" in out
    assert "Applying with --force-fields" not in out


def test_forget_refuses_the_flags_it_would_otherwise_swallow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """"Forget this one record's slot" reads like something smaller than what
    the command does, which is drop the whole entry."""
    root = project(tmp_path, many(1))

    assert cli.main(
        ["--root", str(root), "enrich", "--ai", "--batch-forget", "word:話す0:はなす"]
    ) == 1
    assert "not something it could mean" in capsys.readouterr().err
    assert cli.main(
        ["--root", str(root), "enrich", "--ai", "--batch-forget", "--force-fields", "examples"]
    ) == 1
    assert "no field list to widen" in capsys.readouterr().err
    # Matched on the --force half of the message: "no field list to widen"
    # appears in the --force-fields refusal too, so it would pass either way.
    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-forget", "--force"]) == 1
    assert "no staging file to overwrite" in capsys.readouterr().err


def test_a_forget_that_could_not_be_saved_says_what_to_do_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """This is the command that exists so nobody hand-edits data/ledger.json.
    Failing without a next step invites exactly that edit."""
    root = submitted(tmp_path, many(1), monkeypatch, FakeBatches())
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("read-only")),
    )
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-forget"]) == 1

    err = capsys.readouterr().err
    assert "msgbatch_01" in err
    assert "still recorded as pending on disk" in err
    assert "--batch-forget again" in err
    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]


def test_one_errored_row_does_not_defeat_the_moved_collection_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A record that is both gone and errored is reported as failed, never as
    missing — so counting the missing would let one overloaded request clear
    the id of a batch none of whose records are here."""
    records = many(2)
    batches = FakeBatches(
        results=[
            Entry(key(records[0].id), Failed("errored", "Overloaded")),
            ok(records[1]),
        ]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    stranger = record(id="word:聞く:きく", expression="聞く")
    (root / "vocabulary.json").write_text(
        json.dumps([stranger.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1

    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]


def test_nothing_is_called_dropped_on_the_path_that_drops_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The entry survives and re-fetching recovers every answer, so telling the
    user their results were dropped one line before telling them to go get them
    is the wrong thing to say."""
    records = many(2)
    batches = FakeBatches(results=[ok(item) for item in records])
    root = submitted(tmp_path, records, monkeypatch, batches)
    stranger = record(id="word:聞く:きく", expression="聞く")
    (root / "vocabulary.json").write_text(
        json.dumps([stranger.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"])

    assert "were dropped" not in capsys.readouterr().err


def test_a_moved_collection_is_diagnosed_before_a_malformed_row_can_be(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Streaming the results first would schema-validate every succeeded row on
    the way to a verdict that needs none of them — and one bad row would send
    the user after a schema problem when the collection is what moved."""

    class Malformed:
        stop_reason = "end_turn"
        content = [type("B", (), {"type": "text", "text": "{not json"})()]

    records = many(2)
    batches = FakeBatches(
        results=[
            Entry(key(records[0].id), Succeeded(Malformed())),
            ok(records[1]),
        ]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    stranger = record(id="word:聞く:きく", expression="聞く")
    (root / "vocabulary.json").write_text(
        json.dumps([stranger.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1

    err = capsys.readouterr().err
    assert "any more" in err
    assert "did not match the expected shape" not in err
    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]
