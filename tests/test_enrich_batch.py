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
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import seed_prompts
from japanese_anki import claude_client, cli, enrich, ledger
from japanese_anki.errors import JankiError
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.staging import read_staging

REQUEST_A = "a" * 64
REQUEST_B = "b" * 64
INPUT_A = "c" * 64
INPUT_B = "d" * 64


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


def input_fingerprints(
    records: list[VocabularyRecord], *, taught: str = ""
) -> dict[str, str]:
    return {
        item.id: enrich.ai_input_fingerprint(item, taught=taught) for item in records
    }


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
                "meanings": ["to speak"],
                "examples": [
                    {
                        "japanese": japanese,
                        "furigana": furigana or japanese,
                        "english": "Fixture translation.",
                        "romaji": "fixture romaji",
                        "speech_level": "polite",
                    }
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
        'staging_dir = "staging"\n'
        "\n[ai]\n"
        'enrich_provider = "anthropic"\n'
        'enrich_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
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


def test_submit_explains_that_message_batches_need_anthropic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    config = (root / "janki.toml").read_text(encoding="utf-8")
    (root / "janki.toml").write_text(
        config.replace('enrich_provider = "anthropic"', 'enrich_provider = "codex"'),
        encoding="utf-8",
    )

    code = cli.main(
        ["--root", str(root), "enrich", "--ai", "--batch-submit"]
    )

    assert code == 1
    stderr = capsys.readouterr().err
    assert "Message Batches API" in stderr
    assert 'enrich_provider = "anthropic"' in stderr


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


class _Stream:
    """The manager ``messages.stream`` returns: dunder lookup is on the type,
    so this cannot be a SimpleNamespace."""

    def __init__(self, answer: Any) -> None:
        self._answer = answer

    def __enter__(self) -> Any:
        return SimpleNamespace(get_final_message=lambda: self._answer)

    def __exit__(self, *_: Any) -> bool:
        return False


def test_a_batch_request_is_the_same_request_a_live_call_makes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claude_client, "load_anthropic", fake_sdk)
    blocks = claude_client.system_blocks("guide", "instructions")
    sent: dict[str, Any] = {}

    class Client:
        class messages:  # noqa: N801
            @staticmethod
            def stream(**kw: Any) -> Any:
                sent.update(kw)
                return _Stream(
                    type("R", (), {"stop_reason": "refusal", "stop_details": None})()
                )

    claude_client.parse_call("m", blocks, "hello", dict, Client(), max_tokens=99)
    batched = claude_client.batch_request("r1", "m", blocks, "hello", dict, max_tokens=99)

    assert batched["custom_id"] == "r1"
    assert batched["params"] == sent


def test_the_batch_prompt_has_no_variety_pressure(tmp_path: Path) -> None:
    # Every request is built before any answer exists, so there is nothing for
    # a later prompt to have seen. Said out loud because it is a real
    # difference in what the two paths produce.
    plan = enrich.batch_requests(
        many(3), model="m", style_guide="guide", instructions="I"
    )

    contents = [item["params"]["messages"][0]["content"] for item in plan.requests]
    assert len(plan.requests) == 3
    assert plan.record_ids == [item.id for item in many(3)]
    assert all("Recent examples from this run:\n(none)" in text for text in contents)


def test_the_batch_carries_the_same_reviewed_patterns_as_the_immediate_path() -> None:
    """A batch is the same work at a different price, so it must be the same
    request. `batch_requests` had no way to accept the patterns block, so
    `enrich --ai` wrote 〜んだ sentences while `enrich --ai --batch-submit` on
    the same record wrote whatever the model reached for — and batch is the
    path used for bulk, so most of a collection got the unsteered version."""
    from japanese_anki.patterns import Pattern, format_patterns

    plan = enrich.batch_requests(
        many(2), model="m", style_guide="guide",
        taught=format_patterns([Pattern("〜んだ", "explains")]),
        instructions="I")

    contents = [item["params"]["messages"][0]["content"] for item in plan.requests]
    assert all("〜んだ" in text for text in contents)


def test_the_batch_asks_for_the_long_cache_window() -> None:
    # A five-minute window does not survive the span a batch's requests are read
    # over; the style guide leads every one of them.
    plan = enrich.batch_requests(
        many(1), model="m", style_guide="guide", instructions="I"
    )

    system = plan.requests[0]["params"]["system"]
    assert system[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_the_batch_plan_carries_full_request_and_record_input_provenance() -> None:
    [item] = many(1)

    plan = enrich.batch_requests(
        [item], model="m", style_guide="guide", instructions="task"
    )

    assert plan.input_fingerprints == {item.id: enrich.ai_input_fingerprint(item)}
    assert plan.request_fingerprints == {
        item.id: enrich.ai_request_fingerprint(
            item,
            provider="anthropic",
            style_guide="guide",
            instructions="task",
        )
    }


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


def test_nothing_to_submit_names_the_fields_that_define_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    complete = record(usage_notes="", examples=[ExampleSentence(japanese="話す。")])
    root = project(tmp_path, [complete])
    batches = FakeBatches()
    patch_all(monkeypatch, batches)

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-submit"]) == 0

    assert batches.submitted == []
    assert (
        capsys.readouterr().out
        == "Nothing to submit: every record already has meanings and examples.\n"
    )


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


@pytest.mark.parametrize(
    ("map_name", "bad_value"),
    [
        ("request_fingerprints", "tampered"),
        ("input_fingerprints", " " * 64),
    ],
)
def test_a_corrupt_batch_fingerprint_refuses_before_poll_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    map_name: str,
    bad_value: str,
) -> None:
    item = record(examples=[], usage_notes="")
    batches = FakeBatches(results=[ok(item)])
    root = submitted(tmp_path, [item], monkeypatch, batches)
    capsys.readouterr()
    payload = book_of(root)
    entry = payload["pending_batches"]["msgbatch_01"]
    entry[map_name][item.id] = bad_value
    (root / "ledger.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    before = (root / "vocabulary.json").read_text(encoding="utf-8")
    batches.retrieved.clear()

    code = cli.main(
        ["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]
    )

    assert code == 1
    assert batches.retrieved == []
    assert (root / "vocabulary.json").read_text(encoding="utf-8") == before
    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]
    assert "lowercase SHA-256" in capsys.readouterr().err


@pytest.mark.parametrize("damaged_model", [None, "", "   "])
def test_a_batch_with_no_exact_model_attribution_refuses_before_poll_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    damaged_model: str | None,
) -> None:
    item = record(examples=[], usage_notes="")
    batches = FakeBatches(results=[ok(item)])
    root = submitted(tmp_path, [item], monkeypatch, batches)
    capsys.readouterr()
    payload = book_of(root)
    entry = payload["pending_batches"]["msgbatch_01"]
    if damaged_model is None:
        entry.pop("model")
    else:
        entry["model"] = damaged_model
    (root / "ledger.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    before = (root / "vocabulary.json").read_text(encoding="utf-8")
    batches.retrieved.clear()

    code = cli.main(
        ["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]
    )

    assert code == 1
    assert batches.retrieved == []
    assert (root / "vocabulary.json").read_text(encoding="utf-8") == before
    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]
    assert "model" in capsys.readouterr().err


@pytest.mark.parametrize(
    "damaged_retry_ids",
    [
        "word:話す:はなす",
        [],
        ["word:話す:はなす", "word:話す:はなす"],
        [""],
        ["   "],
        [1],
        ["word:ghost:ghost"],
    ],
    ids=[
        "not-a-list",
        "empty-list",
        "duplicate",
        "blank",
        "whitespace",
        "not-text",
        "not-pending",
    ],
)
def test_malformed_retry_ids_refuse_before_poll_or_losing_the_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    damaged_retry_ids: Any,
) -> None:
    item = record(examples=[], usage_notes="")
    batches = FakeBatches(results=[ok(item)])
    root = submitted(tmp_path, [item], monkeypatch, batches)
    capsys.readouterr()
    payload = book_of(root)
    payload["pending_batches"]["msgbatch_01"]["retry_ids"] = damaged_retry_ids
    (root / "ledger.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    before = (root / "vocabulary.json").read_text(encoding="utf-8")
    batches.retrieved.clear()

    code = cli.main(
        ["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]
    )

    assert code == 1
    assert batches.retrieved == []
    assert (root / "vocabulary.json").read_text(encoding="utf-8") == before
    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]
    assert "retry_ids" in capsys.readouterr().err


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


@pytest.mark.parametrize("model", ["", "   "])
def test_an_ai_batch_needs_a_nonblank_model_before_it_is_recorded(
    tmp_path: Path, model: str
) -> None:
    book = ledger.Ledger(path=tmp_path / "ledger.json")

    with pytest.raises(ledger.LedgerError, match="model"):
        book.record_batch(
            "b1",
            kind="ai",
            model=model,
            provider="anthropic",
            pending_ids=["a"],
            request_fingerprints={"a": REQUEST_A},
            input_fingerprints={"a": INPUT_A},
        )

    assert book.pending_batches == {}


def test_recording_a_batch_replaces_rather_than_accumulates(tmp_path: Path) -> None:
    # A batch is one thing that is either in flight or collected; two entries
    # for one id would be the ledger disagreeing with itself.
    book = ledger.Ledger(path=tmp_path / "ledger.json")
    book.record_batch(
        "b1",
        kind="ai",
        model="m",
        provider="anthropic",
        pending_ids=["a"],
        request_fingerprints={"a": REQUEST_A},
        input_fingerprints={"a": INPUT_A},
    )
    book.record_batch(
        "b1",
        kind="ai",
        model="m",
        provider="anthropic",
        pending_ids=["a", "b"],
        request_fingerprints={"a": REQUEST_A, "b": REQUEST_B},
        input_fingerprints={"a": INPUT_A, "b": INPUT_B},
    )

    assert book.pending_batches["b1"]["pending_ids"] == ["a", "b"]


@pytest.mark.parametrize(
    ("request_fingerprints", "input_fingerprints", "detail"),
    [
        (None, {"a": INPUT_A}, "request fingerprints.*missing a"),
        ({"a": REQUEST_A}, None, "input fingerprints.*missing a"),
        (
            {"a": REQUEST_A, "unknown": REQUEST_B},
            {"a": INPUT_A},
            "request fingerprints.*unknown unknown",
        ),
        (
            {"a": REQUEST_A},
            {"a": INPUT_A, "unknown": INPUT_B},
            "input fingerprints.*unknown unknown",
        ),
        (
            {"a": "tampered"},
            {"a": INPUT_A},
            "request fingerprints.*lowercase SHA-256",
        ),
        (
            {"a": REQUEST_A},
            {"a": " " * 64},
            "input fingerprints.*lowercase SHA-256",
        ),
    ],
    ids=[
        "missing-request",
        "missing-input",
        "extra-request",
        "extra-input",
        "malformed-request",
        "malformed-input",
    ],
)
def test_an_ai_batch_requires_exact_request_and_input_fingerprint_coverage(
    tmp_path: Path,
    request_fingerprints: dict[str, str] | None,
    input_fingerprints: dict[str, str] | None,
    detail: str,
) -> None:
    book = ledger.Ledger(path=tmp_path / "ledger.json")

    with pytest.raises(ledger.LedgerError, match=detail):
        book.record_batch(
            "b1",
            kind="ai",
            model="m",
            provider="anthropic",
            pending_ids=["a"],
            request_fingerprints=request_fingerprints,
            input_fingerprints=input_fingerprints,
        )

    assert book.pending_batches == {}


def test_an_unknown_kind_is_refused(tmp_path: Path) -> None:
    book = ledger.Ledger(path=tmp_path / "ledger.json")

    with pytest.raises(ledger.LedgerError):
        book.record_batch("b1", kind="nonsense", model="m", pending_ids=["a"])


def test_a_pending_batch_round_trips_through_the_file(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    book = ledger.Ledger(path=path)
    book.record_batch(
        "b1",
        kind="ai",
        model="m",
        provider="anthropic",
        pending_ids=["a", "b"],
        request_fingerprints={"a": REQUEST_A, "b": REQUEST_B},
        input_fingerprints={"a": INPUT_A, "b": INPUT_B},
    )
    book.save()

    reloaded = ledger.load(path)

    assert reloaded.pending_batch() == ("b1", book.pending_batches["b1"])
    entry = reloaded.pending_batches["b1"]
    assert entry["request_fingerprints"] == {
        "a": REQUEST_A,
        "b": REQUEST_B,
    }
    assert entry["input_fingerprints"] == {"a": INPUT_A, "b": INPUT_B}
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


def test_force_fields_at_fetch_never_overrides_changed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Force widens writable fields; it does not make an old answer current."""
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
    captured = capsys.readouterr()
    out = captured.out
    assert "instead of the none this batch was submitted with" in out
    landed = stored(root)["word:話す:はなす"]["examples"]
    assert [item["japanese"] for item in landed] == ["人と話す。"]
    assert "changed after submission" in captured.err


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
    """Counting the missing would let the mix of outcomes decide the guard, so
    it asks the records instead. (The classification itself is pinned directly
    by test_a_terminal_row_keeps_its_reason_even_when_its_record_is_gone; this
    one is about the guard, which short-circuits before any row is read.)"""
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


def test_one_unparseable_row_costs_that_row_and_not_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A batch's results are immutable, so raising on a bad row would fail at
    the same row on every re-fetch — and the only way out would be forgetting
    the batch, throwing away every good answer beside it."""

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
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1

    kept = stored(root)
    assert kept["word:話す1:はなす"]["examples"], "the good answer landed"
    assert kept["word:話す0:はなす"]["examples"] == []
    err = capsys.readouterr().err
    assert "word:話す0:はなす" in err
    assert "did not match the expected shape" in err
    # The unreadable answer is intact on Anthropic's side and janki's schema is
    # what turned it away, so the id that reaches it is kept.
    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]


def test_the_override_announcement_names_both_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both halves come from parsed values now — the override from the flag
    command_enrich parsed, the submitted list from the ledger — so the line has
    to name what was asked for as well as what is being done instead.

    A regression guard rather than a red-before test: the output is what it
    always was, and what changed underneath is which variable each half reads.
    Crossing the two is the mistake this catches."""
    curated = record(usage_notes="note", examples=[ExampleSentence(japanese="人と話す。")])
    batches = FakeBatches(results=[ok(curated)])
    root = project(tmp_path, [curated])
    patch_all(monkeypatch, batches)
    cli.main(
        [
            "--root", str(root), "enrich", "--ai", "--batch-submit",
            "--force-fields", "usage_notes", curated.id,
        ]
    )
    capsys.readouterr()

    cli.main(
        [
            "--root", str(root), "enrich", "--ai", "--batch-fetch",
            "--force-fields", "examples", "--yes",
        ]
    )

    out = capsys.readouterr().out
    assert "Applying with --force-fields examples" in out
    assert "instead of the usage_notes this batch was submitted with" in out


def test_field_names_are_validated_in_one_place_and_it_is_not_the_fetch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pre-existing behavior, pinned because it became load-bearing: the fetch
    used to re-parse the flag and no longer does, so command_enrich's parse is
    now the only one. A misspelt field must still be caught before any batch
    state is read."""
    root = project(tmp_path, many(1))

    assert cli.main(
        [
            "--root", str(root), "enrich", "--ai", "--batch-fetch",
            "--force-fields", "exmaples",
        ]
    ) == 1

    assert "unknown field 'exmaples'" in capsys.readouterr().err


def test_a_batch_janki_could_not_read_at_all_is_not_forgotten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shape that makes the rule matter: a schema gains a field while a
    large batch is out, so every answer fails validation. They are complete and
    paid for; dropping the only id that reaches them would be janki discarding
    work over its own decision."""

    class Malformed:
        stop_reason = "end_turn"
        content = [type("B", (), {"type": "text", "text": "{not json"})()]

    records = many(3)
    batches = FakeBatches(
        results=[Entry(key(item.id), Succeeded(Malformed())) for item in records]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1

    captured = capsys.readouterr()
    assert "3 answer(s) in batch msgbatch_01 did not parse" in captured.err
    # Printed to stdout, so it has to be denied there — the "collected" line is
    # what this guard exists to suppress.
    assert "none of the answers had anything to fill" not in captured.out
    assert list(book_of(root)["pending_batches"]) == ["msgbatch_01"]
    # And the way out, for when they are not worth chasing.
    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-forget"]) == 0
    assert book_of(root)["pending_batches"] == {}


def test_a_terminal_row_does_not_hold_the_batch_the_way_an_unreadable_one_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An errored row has no answer anywhere and a re-fetch returns the same
    row, so it is terminal and the id may go.

    Green before the change as well as after — it guards the *other* side of a
    distinction this commit introduces, so that a later edit cannot collapse
    invalid back into failed without one of the pair going red."""
    records = many(2)
    batches = FakeBatches(
        results=[
            Entry(key(records[0].id), Failed("errored", "Overloaded")),
            ok(records[1]),
        ]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 0

    assert book_of(root)["pending_batches"] == {}


# F3: restored — the only thing pinning where the announcement sits.
def test_a_moved_collection_claims_no_override_it_never_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The announcement sits below the moved-collection guard on purpose: a run
    that collects nothing applies nothing and should claim nothing."""
    records = many(2)
    batches = FakeBatches(results=[ok(item) for item in records])
    root = submitted(tmp_path, records, monkeypatch, batches)
    stranger = record(id="word:聞く:きく", expression="聞く")
    (root / "vocabulary.json").write_text(
        json.dumps([stranger.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    code = cli.main(
        [
            "--root", str(root), "enrich", "--ai", "--batch-fetch",
            "--force-fields", "examples", "--yes",
        ]
    )

    assert code == 1
    output = capsys.readouterr()
    assert "any more" in output.err
    assert "Applying with --force-fields" not in output.out


def test_a_second_fetch_does_not_rewrite_what_the_first_one_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Holding the batch is what makes a second fetch reachable, and the gap
    between the two is exactly when a human corrects a sentence the first one
    wrote. Under the submitted --force-fields, re-applying would replace that
    correction with the model's original text."""

    class Malformed:
        stop_reason = "end_turn"
        content = [type("B", (), {"type": "text", "text": "{not json"})()]

    records = many(2)
    batches = FakeBatches(
        results=[
            ok(records[0]),
            Entry(key(records[1].id), Succeeded(Malformed())),
        ]
    )
    root = project(tmp_path, records)
    patch_all(monkeypatch, batches)
    cli.main(
        [
            "--root", str(root), "enrich", "--ai", "--batch-submit",
            "--force-fields", "examples",
        ]
    )
    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1
    # Narrowed to the row that failed, not to the one that landed.
    assert book_of(root)["pending_batches"]["msgbatch_01"]["retry_ids"] == [
        "word:話す1:はなす"
    ]

    # The human fixes the furigana on what landed.
    landed = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    for item in landed:
        if item["id"] == "word:話す0:はなす":
            item["examples"][0]["furigana"] = "毎日[まいにち] 話[はな]す0。 (checked)"
    (root / "vocabulary.json").write_text(
        json.dumps(landed, ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"])

    kept = stored(root)["word:話す0:はなす"]["examples"][0]
    assert kept["furigana"].endswith("(checked)"), "the hand correction survived"
    assert "settled by an earlier fetch" in capsys.readouterr().out


def test_a_held_staging_batch_says_the_file_is_where_the_answers_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A staging file *is* the answers, the same way extract's is. This batch
    will not offer them again, so "promote or discard" would present the one
    irreversible choice as a free alternative to the other."""

    class Malformed:
        stop_reason = "end_turn"
        content = [type("B", (), {"type": "text", "text": "{not json"})()]

    records = many(enrich.STAGING_THRESHOLD)
    batches = FakeBatches(
        results=[ok(item) for item in records[:-1]]
        + [Entry(key(records[-1].id), Succeeded(Malformed()))]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1

    err = capsys.readouterr().err
    assert cli.STAGING_FILE_NAME in err
    assert "Promote it to land them" in err
    assert "deleting it loses them" in err
    assert "discard" not in err
    assert "record(s) were written." not in err


def test_a_staged_batch_is_not_re_applied_over_what_was_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The case that made recording successes the wrong direction: the answers
    went to a staging file, a human corrected one and promoted it, and a second
    fetch under the submitted --force-fields would put the model's original
    text back over the correction."""

    class Malformed:
        stop_reason = "end_turn"
        content = [type("B", (), {"type": "text", "text": "{not json"})()]

    records = many(enrich.STAGING_THRESHOLD)
    batches = FakeBatches(
        results=[ok(item) for item in records[:-1]]
        + [Entry(key(records[-1].id), Succeeded(Malformed()))]
    )
    root = project(tmp_path, records)
    patch_all(monkeypatch, batches)
    cli.main(
        [
            "--root", str(root), "enrich", "--ai", "--batch-submit",
            "--force-fields", "examples",
        ]
    )
    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1
    target = root / "staging" / "ai-enrichment.yaml"
    assert target.is_file()
    # The reviewer fixes a segmentation and promotes.
    staged = target.read_text(encoding="utf-8").replace("毎日話す0。", "毎日話す0。 (checked)")
    target.write_text(staged, encoding="utf-8")
    assert cli.main(["--root", str(root), "promote", str(target), "--skip-reading-check"]) == 0
    capsys.readouterr()

    cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"])

    # The damage lands a promote later, so follow the second fetch all the way
    # through the route the first one took.
    again = root / "staging" / "ai-enrichment.yaml"
    if again.is_file():
        restaged, _ = read_staging(again)
        assert "word:話す0:はなす" not in {item.id for item in restaged}, (
            "the settled record was proposed for rewriting again"
        )
        cli.main(["--root", str(root), "promote", str(again), "--skip-reading-check"])

    landed = stored(root)["word:話す0:はなす"]["examples"]
    assert any("(checked)" in item["japanese"] for item in landed), (
        "the promoted correction survived the second fetch"
    )


def test_an_unreadable_answer_for_a_deleted_record_does_not_hold_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The plan's rule says a record deleted while the batch was out is
    deliberate and its answer moot. Deciding "unreadable" before "still here"
    would hold the id forever for a word nobody wants."""

    class Malformed:
        stop_reason = "end_turn"
        content = [type("B", (), {"type": "text", "text": "{not json"})()]

    records = many(2)
    batches = FakeBatches(
        results=[
            ok(records[0]),
            Entry(key(records[1].id), Succeeded(Malformed())),
        ]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    (root / "vocabulary.json").write_text(
        json.dumps([records[0].to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 0

    err = capsys.readouterr().err
    assert "no longer in the collection" in err
    assert "did not parse" not in err
    assert book_of(root)["pending_batches"] == {}


def test_retrying_one_row_of_a_large_batch_is_a_diff_not_a_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The staging threshold is about what this run asks a human to review, and
    a retry of one row is one row however large the batch was."""

    class Malformed:
        stop_reason = "end_turn"
        content = [type("B", (), {"type": "text", "text": "{not json"})()]

    records = many(enrich.STAGING_THRESHOLD)
    first = FakeBatches(
        results=[ok(item) for item in records[:-1]]
        + [Entry(key(records[-1].id), Succeeded(Malformed()))]
    )
    root = submitted(tmp_path, records, monkeypatch, first)
    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1
    (root / "staging" / "ai-enrichment.yaml").unlink()
    # The schema is fixed; the row parses this time.
    first.results_rows = [ok(item) for item in records]
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 0

    assert not (root / "staging" / "ai-enrichment.yaml").exists()
    assert stored(root)[records[-1].id]["examples"], "the retried row landed in the records"
    assert book_of(root)["pending_batches"] == {}


def test_a_terminal_row_keeps_its_reason_even_when_its_record_is_gone() -> None:
    """The API's word is the only signal that the API, rather than curation, is
    why a word went unenriched — so an errored row stays errored whether or not
    somebody deleted the record in the meantime. Only an *unreadable* answer
    goes through the presence test, because only that one could hold the id."""
    present = record(id="word:聞く:きく", expression="聞く")
    entries = [
        claude_client.BatchEntry(key("word:話す:はなす"), "errored", "Overloaded", None),
        claude_client.BatchEntry(key("word:見る:みる"), "invalid", "bad json", None),
    ]

    outcome = enrich.apply_batch_results(
        [present],
        entries,
        ["word:話す:はなす", "word:見る:みる"],
        model="m",
        input_fingerprints={},
    )

    reason = outcome.failed["word:話す:はなす"]
    assert "Overloaded" in reason
    # Not in `missing` — there is no answer for a later fetch to get, so it must
    # not hold the batch — but the collection having moved under it is reported
    # by nothing else, so the reason says both.
    assert "word:話す:はなす" not in outcome.missing
    assert "no longer in the collection" in reason
    # Unreadable and deleted: moot, and it must not hold the batch.
    assert outcome.missing == ["word:見る:みる"]
    assert not outcome.invalid


def test_an_unreadable_answer_for_a_record_still_here_is_held_not_dropped() -> None:
    """The other side of the presence branch. Green before the change too — it
    is here so that collapsing the two cases back together turns one of the
    pair red."""
    here = record(id="word:見る:みる", expression="見る")
    entries = [claude_client.BatchEntry(key(here.id), "invalid", "bad json", None)]

    outcome = enrich.apply_batch_results(
        [here],
        entries,
        [here.id],
        model="m",
        input_fingerprints=input_fingerprints([here]),
    )

    assert outcome.invalid == {here.id: "bad json"}
    assert not outcome.missing


def test_a_batch_answer_is_stale_when_the_record_data_changed_after_submission() -> None:
    submitted_record = record()
    edited_record = record(usage_notes="A human edited this while the batch ran.")
    entry = claude_client.BatchEntry(
        key(edited_record.id),
        "succeeded",
        "",
        claude_client.CallResult(
            enrich.ai_schema()(
                meanings=["to converse"], examples=[], usage_notes="model note"
            ),
            "end_turn",
            None,
        ),
    )

    outcome = enrich.apply_batch_results(
        [edited_record],
        [entry],
        [edited_record.id],
        model="m",
        input_fingerprints=input_fingerprints([submitted_record]),
        force_fields=("meanings", "usage_notes"),
    )

    assert outcome.stale == [edited_record.id]
    assert outcome.result.records == [edited_record]
    assert outcome.result.changes == {}
    assert outcome.result.looked_up == 0


def test_a_current_batch_answer_carries_its_submitted_request_provenance() -> None:
    item = record()
    plan = enrich.batch_requests(
        [item], model="m", style_guide="guide", instructions="task"
    )
    entry = claude_client.BatchEntry(
        key(item.id),
        "succeeded",
        "",
        claude_client.CallResult(
            enrich.ai_schema()(
                meanings=["to speak"], examples=[], usage_notes="model note"
            ),
            "end_turn",
            None,
        ),
    )

    outcome = enrich.apply_batch_results(
        [item],
        [entry],
        [item.id],
        model="m",
        input_fingerprints=plan.input_fingerprints,
        request_fingerprints=plan.request_fingerprints,
    )

    assert outcome.stale == []
    assert outcome.result.provenance == plan.request_fingerprints
    assert outcome.result.input_fingerprints == plan.input_fingerprints


def test_a_sub_threshold_retry_does_not_refuse_over_a_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The retry is one row, so it takes the diff route and never looks at the
    staging file. Green before the change: what this commit fixed is the held
    message, which promised a refusal this pins as not happening."""

    class Malformed:
        stop_reason = "end_turn"
        content = [type("B", (), {"type": "text", "text": "{not json"})()]

    records = many(enrich.STAGING_THRESHOLD)
    batches = FakeBatches(
        results=[ok(item) for item in records[:-1]]
        + [Entry(key(records[-1].id), Succeeded(Malformed()))]
    )
    root = submitted(tmp_path, records, monkeypatch, batches)
    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 1
    assert (root / "staging" / "ai-enrichment.yaml").is_file()
    batches.results_rows = [ok(item) for item in records]
    capsys.readouterr()

    # The staging file is still sitting there unpromoted, and this exits 0.
    assert cli.main(["--root", str(root), "enrich", "--ai", "--batch-fetch", "--yes"]) == 0

    assert stored(root)[records[-1].id]["examples"]
    assert book_of(root)["pending_batches"] == {}


def test_a_terminal_row_for_a_record_still_present_says_left_untouched() -> None:
    here = record(id="word:話す:はなす")
    entries = [claude_client.BatchEntry(key(here.id), "errored", "Overloaded", None)]

    outcome = enrich.apply_batch_results(
        [here],
        entries,
        [here.id],
        model="m",
        input_fingerprints=input_fingerprints([here]),
    )

    reason = outcome.failed[here.id]
    assert reason.endswith("left untouched")
    assert "no longer in the collection" not in reason
