"""Rewriting English glosses — ``janki enrich --polish-meanings``.

The one pass that replaces a full field rather than filling an empty one, so
these tests are mostly about what it refuses to do: empty a record's meanings,
write without being asked, or keep spending after the reviewer has stopped
reading. No network — the Claude call is faked (IMPLEMENTATION_PLAN rule 6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import claude_client, cli, enrich
from japanese_anki.claude_client import CallResult, Refusal
from japanese_anki.enrich import (
    apply_polish_result,
    polish_prompt,
    polish_schema,
    polish_targets,
)
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord


def answer(*meanings: str) -> Any:
    return polish_schema()(meanings=list(meanings))


def ok(*meanings: str) -> CallResult:
    return CallResult(answer(*meanings), "end_turn", None)


def record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:聞く:きく",
        "expression": "聞く",
        "reading": "きく",
        "meanings": ["to hear", "hearing"],
        "part_of_speech": "verb",
        "verb_group": "godan",
        "source": SourceReference(type="shirabe", imported_from="export.csv"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


class FakeCall:
    """Stands in for ``claude_client.parse_call``, counting what it was asked."""

    def __init__(self, *results: CallResult) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, model: str, blocks: Any, content: Any, schema: Any, client: Any = None, **kw: Any
    ) -> CallResult:
        self.calls.append({"model": model, "system": blocks, "content": content})
        return self.results.pop(0) if self.results else ok()


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


def patch_all(monkeypatch: pytest.MonkeyPatch, call: FakeCall) -> None:
    # Deliberately no JPDB_API_KEY: this pass rewrites English and asks jpdb
    # nothing, so needing a dictionary key to run it would be a bug.
    monkeypatch.delenv("JPDB_API_KEY", raising=False)
    monkeypatch.setattr(cli.enrich.claude_client, "parse_call", call)


def stored(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload}


def book_of(root: Path) -> dict[str, Any]:
    return json.loads((root / "ledger.json").read_text(encoding="utf-8"))


def answers(monkeypatch: pytest.MonkeyPatch, *replies: str) -> list[str]:
    """Drive the per-record prompt from a script, and record what it asked."""
    seen = list(replies)
    prompts: list[str] = []

    def fake_input(text: str = "") -> str:
        prompts.append(text)
        if not seen:
            raise EOFError
        return seen.pop(0)

    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


# --- targets and the prompt --------------------------------------------------


def test_every_record_is_a_target_because_meanings_are_never_empty() -> None:
    records = [record(), record(id="word:話す:はなす", expression="話す")]

    assert polish_targets(records) == records


def test_naming_ids_narrows_it() -> None:
    records = [record(), record(id="word:話す:はなす", expression="話す")]

    assert [item.id for item in polish_targets(records, ["word:話す:はなす"])] == [
        "word:話す:はなす"
    ]


def test_an_unknown_id_is_an_error() -> None:
    with pytest.raises(enrich.EnrichError) as caught:
        polish_targets([record()], ["word:nope:nope"])

    assert "word:nope:nope" in str(caught.value)


def test_the_prompt_carries_the_examples_that_say_which_sense_it_is() -> None:
    # 聞く is "to hear" and "to ask"; the sentence on file is the only evidence
    # janki has for which one this record was collected for.
    text = polish_prompt(
        record(examples=[ExampleSentence(japanese="先生に聞きました。")])
    )

    assert "Current meanings: to hear; hearing" in text
    assert "先生に聞きました。" in text


def test_the_prompt_fingerprint_covers_meanings_and_sense_defining_examples() -> None:
    original = record(examples=[ExampleSentence(japanese="音楽を聞きました。")])

    assert enrich.polish_prompt_fingerprint(original) != enrich.polish_prompt_fingerprint(
        record(
            meanings=["to ask"],
            examples=[ExampleSentence(japanese="音楽を聞きました。")],
        )
    )
    assert enrich.polish_prompt_fingerprint(original) != enrich.polish_prompt_fingerprint(
        record(examples=[ExampleSentence(japanese="先生に聞きました。")])
    )


def test_a_record_with_no_examples_still_prompts() -> None:
    text = polish_prompt(record())

    assert "聞く" in text
    assert "Current meanings: to hear; hearing" in text


# --- what an answer is allowed to do -----------------------------------------


def test_a_better_list_becomes_a_proposal() -> None:
    outcome = apply_polish_result(record(), answer("to hear", "to listen to", "to ask"))

    assert outcome.proposed is not None
    assert outcome.proposed.meanings == ["to hear", "to listen to", "to ask"]
    assert outcome.changes["meanings"] == (
        ["to hear", "hearing"],
        ["to hear", "to listen to", "to ask"],
    )


def test_an_empty_answer_means_the_glosses_are_already_right() -> None:
    outcome = apply_polish_result(record(), answer())

    assert outcome.proposed is None
    assert not outcome.changes


def test_an_answer_that_matches_what_is_on_file_is_not_a_change() -> None:
    outcome = apply_polish_result(record(), answer("to hear", "hearing"))

    assert outcome.proposed is None


def test_an_answer_of_only_blanks_never_empties_the_field() -> None:
    # A card with a Japanese side and no English one is worse than a clumsy
    # gloss, so this is refused rather than written.
    outcome = apply_polish_result(record(), answer("", "   "))

    assert outcome.proposed is None


def test_repeated_glosses_collapse() -> None:
    outcome = apply_polish_result(record(), answer("to hear", "to hear", "to ask"))

    assert outcome.proposed is not None
    assert outcome.proposed.meanings == ["to hear", "to ask"]


def test_nothing_but_meanings_is_touched() -> None:
    before = record(examples=[ExampleSentence(japanese="先生に聞きました。")])

    outcome = apply_polish_result(before, answer("to ask"))

    assert outcome.proposed is not None
    assert outcome.proposed.examples == before.examples
    assert outcome.proposed.id == before.id
    assert outcome.proposed.reading == before.reading


# --- the pass ----------------------------------------------------------------


def test_a_refusal_costs_that_record_not_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [record(), record(id="word:話す:はなす", expression="話す")]
    call = FakeCall(
        CallResult(None, "refusal", Refusal(category="other", explanation="no")),
        ok("to speak"),
    )
    monkeypatch.setattr(enrich.claude_client, "parse_call", call)

    outcomes = list(
        enrich.polish_meanings(records, model="m", style_guide="g")
    )

    assert outcomes[0].warning.startswith("word:聞く:きく")
    assert outcomes[0].proposed is None
    assert outcomes[1].proposed is not None


def test_a_truncated_answer_is_never_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    call = FakeCall(CallResult(None, "max_tokens", None))
    monkeypatch.setattr(enrich.claude_client, "parse_call", call)

    (outcome,) = list(enrich.polish_meanings([record()], model="m", style_guide="g"))

    assert outcome.proposed is None
    assert "max_tokens" in outcome.warning


def test_the_pass_is_lazy_so_walking_away_costs_what_was_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A generator, not a result object: the CLI's confirm decides whether the
    # next call is worth making.
    records = [record(id=f"word:聞く{index}:きく", expression=f"聞く{index}") for index in range(5)]
    call = FakeCall(*[ok("to ask") for _ in range(5)])
    monkeypatch.setattr(enrich.claude_client, "parse_call", call)

    stream = enrich.polish_meanings(records, model="m", style_guide="g")
    next(stream)
    next(stream)

    assert len(call.calls) == 2


# --- the command -------------------------------------------------------------


def test_the_command_shows_the_diff_confirms_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    patch_all(monkeypatch, FakeCall(ok("to hear", "to listen to", "to ask")))
    prompts = answers(monkeypatch, "y")

    assert cli.main(["--root", str(root), "enrich", "--polish-meanings"]) == 0

    out = capsys.readouterr().out
    assert "word:聞く:きく" in out
    assert "meanings: to hear; hearing -> to hear; to listen to; to ask" in out
    assert prompts and "[y/N/q]" in prompts[0]
    assert stored(root)["word:聞く:きく"]["meanings"] == [
        "to hear",
        "to listen to",
        "to ask",
    ]
    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    entries = book["records"]["word:聞く:きく"]["enriched"]
    assert [item["kind"] for item in entries] == ["polish"]
    assert entries[0]["fields"] == ["meanings"]


def test_declining_leaves_the_record_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    patch_all(monkeypatch, FakeCall(ok("to ask")))
    answers(monkeypatch, "n")

    assert cli.main(["--root", str(root), "enrich", "--polish-meanings"]) == 0

    assert stored(root)["word:聞く:きく"]["meanings"] == ["to hear", "hearing"]
    assert not (root / "ledger.json").exists()
    assert "Nothing written." in capsys.readouterr().out


def test_each_record_is_confirmed_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"These thirty are fine except the fourth" is not an answer one y/n takes."""
    root = project(
        tmp_path,
        [record(), record(id="word:話す:はなす", expression="話す", meanings=["to talk"])],
    )
    patch_all(monkeypatch, FakeCall(ok("to ask"), ok("to speak")))
    answers(monkeypatch, "n", "y")

    assert cli.main(["--root", str(root), "enrich", "--polish-meanings"]) == 0

    kept = stored(root)
    assert kept["word:聞く:きく"]["meanings"] == ["to hear", "hearing"]
    assert kept["word:話す:はなす"]["meanings"] == ["to speak"]


def test_quitting_keeps_what_was_already_accepted_and_stops_calling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(
        tmp_path,
        [
            record(),
            record(id="word:話す:はなす", expression="話す", meanings=["to talk"]),
            record(id="word:見る:みる", expression="見る", meanings=["to look"]),
        ],
    )
    call = FakeCall(ok("to ask"), ok("to speak"), ok("to see"))
    patch_all(monkeypatch, call)
    answers(monkeypatch, "y", "q")

    assert cli.main(["--root", str(root), "enrich", "--polish-meanings"]) == 0

    assert stored(root)["word:聞く:きく"]["meanings"] == ["to ask"]
    assert stored(root)["word:見る:みる"]["meanings"] == ["to look"]
    assert len(call.calls) == 2, "the third record was never asked about"
    assert "Stopped" in capsys.readouterr().out


def test_yes_accepts_every_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [record()])
    patch_all(monkeypatch, FakeCall(ok("to ask")))

    assert cli.main(["--root", str(root), "enrich", "--polish-meanings", "--yes"]) == 0

    assert stored(root)["word:聞く:きく"]["meanings"] == ["to ask"]


def test_ids_narrow_what_is_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(
        tmp_path,
        [record(), record(id="word:話す:はなす", expression="話す", meanings=["to talk"])],
    )
    call = FakeCall(ok("to speak"))
    patch_all(monkeypatch, call)

    code = cli.main(
        [
            "--root", str(root), "enrich", "--polish-meanings",
            "--yes", "word:話す:はなす",
        ]
    )

    assert code == 0
    assert len(call.calls) == 1
    assert stored(root)["word:聞く:きく"]["meanings"] == ["to hear", "hearing"]


def test_an_answer_that_changes_nothing_is_reported_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    patch_all(monkeypatch, FakeCall(ok()))

    assert cli.main(["--root", str(root), "enrich", "--polish-meanings", "--yes"]) == 0

    out = capsys.readouterr().out
    assert "1 record(s) already had the glosses" in out
    assert "Nothing written." in out
    assert not (root / "ledger.json").exists()


def test_the_run_says_what_it_will_cost_before_it_spends_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(
        tmp_path,
        [record(), record(id="word:話す:はなす", expression="話す", meanings=["to talk"])],
    )
    patch_all(monkeypatch, FakeCall(ok(), ok()))

    cli.main(["--root", str(root), "enrich", "--polish-meanings", "--yes"])

    assert "Polishing 2 record(s)" in capsys.readouterr().out


def test_the_model_can_be_overridden_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [record()])
    call = FakeCall(ok("to ask"))
    patch_all(monkeypatch, call)

    cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--yes", "--model", "m2"]
    )

    assert call.calls[0]["model"] == "m2"


# --- batch scale -------------------------------------------------------------


def _patch_polish_batch_submit(
    monkeypatch: pytest.MonkeyPatch, submitted: list[list[dict[str, Any]]]
) -> None:
    def request(
        custom_id: str, model: str, blocks: Any, content: str, schema: Any
    ) -> dict[str, Any]:
        return {"custom_id": custom_id, "model": model, "content": content}

    monkeypatch.setattr(enrich.claude_client, "batch_request", request)
    monkeypatch.setattr(
        cli.claude_client,
        "submit_batch",
        lambda requests: submitted.append(list(requests)) or "msgbatch_polish",
    )


def test_polish_can_be_submitted_as_one_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [record(), record(id="word:話す:はなす", expression="話す")]
    root = project(tmp_path, records)
    submitted: list[list[dict[str, Any]]] = []
    _patch_polish_batch_submit(monkeypatch, submitted)

    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-submit"]
    ) == 0

    assert len(submitted[0]) == 2
    pending = book_of(root)["pending_batches"]["msgbatch_polish"]
    assert pending["kind"] == "polish"
    assert pending["pending_ids"] == [item.id for item in records]
    assert set(pending["prompt_fingerprints"]) == {item.id for item in records}


def test_polish_submit_rebases_without_losing_a_parallel_ledger_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = record()
    root = project(tmp_path, [item])
    _patch_polish_batch_submit(monkeypatch, [])
    real_save = cli.ledger.Ledger.save
    first = True

    def race(book: cli.ledger.Ledger) -> None:
        nonlocal first
        if first:
            first = False
            parallel = cli.ledger.load(book.path)
            parallel.extra["parallel-agent"] = {"kept": True}
            real_save(parallel)
        real_save(book)

    monkeypatch.setattr(cli.ledger.Ledger, "save", race)

    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-submit"]
    ) == 0

    payload = book_of(root)
    assert payload["parallel-agent"] == {"kept": True}
    assert list(payload["pending_batches"]) == ["msgbatch_polish"]


def test_failed_polish_submit_keeps_a_complete_auto_recovery_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    item = record()
    root = project(tmp_path, [item])
    _patch_polish_batch_submit(monkeypatch, [])
    real_save = cli.ledger.Ledger.save
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("busy ledger")),
    )

    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-submit"]
    ) == 1

    recovery = cli.ledger.load_batch_recovery(root / "ledger.json")
    assert recovery is not None
    assert recovery.batch_id == "msgbatch_polish"
    assert recovery.entry["model"]
    assert recovery.entry["pending_ids"] == [item.id]
    assert recovery.entry["prompt_fingerprints"][item.id]
    assert "complete descriptor" in capsys.readouterr().err

    monkeypatch.setattr(cli.ledger.Ledger, "save", real_save)
    monkeypatch.setattr(cli.claude_client, "batch_status", lambda batch_id: "processing")
    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-fetch"]
    ) == 0

    assert list(book_of(root)["pending_batches"]) == ["msgbatch_polish"]
    assert cli.ledger.load_batch_recovery(root / "ledger.json") is None


def test_forget_can_resolve_two_batches_created_by_a_submission_race(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    item = record()
    root = project(tmp_path, [item])
    fingerprint = enrich.polish_prompt_fingerprint(item)
    book = cli.ledger.load(root / "ledger.json")
    book.record_batch(
        "msgbatch_in_ledger",
        kind="polish",
        model="claude-opus-5",
        pending_ids=[item.id],
        prompt_fingerprints={item.id: fingerprint},
        at="2026-08-11",
    )
    book.save()
    recovery_entry = {
        "kind": "polish",
        "model": "claude-opus-5",
        "submitted_at": "2026-08-11",
        "pending_ids": [item.id],
        "force_fields": [],
        "prompt_fingerprints": {item.id: fingerprint},
    }
    cli.ledger.write_batch_recovery(
        root / "ledger.json", "msgbatch_recovered", recovery_entry
    )

    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-forget"]
    ) == 0

    assert book_of(root)["pending_batches"] == {}
    recovery = cli.ledger.load_batch_recovery(root / "ledger.json")
    assert recovery is not None
    assert recovery.batch_id == "msgbatch_recovered"
    assert "remains intact" in capsys.readouterr().out


def test_polish_fetch_refuses_a_pending_ai_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    item = record()
    root = project(tmp_path, [item])
    book = cli.ledger.load(root / "ledger.json")
    book.record_batch(
        "msgbatch_ai", kind="ai", model="claude-opus-5", pending_ids=[item.id]
    )
    book.save()

    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-fetch"]
    ) == 1

    assert "AI-enrichment batch" in capsys.readouterr().err
    assert list(book_of(root)["pending_batches"]) == ["msgbatch_ai"]


def test_fetching_a_polish_batch_reviews_and_writes_its_proposals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = record()
    root = project(tmp_path, [item])
    _patch_polish_batch_submit(monkeypatch, [])
    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-submit"]
    ) == 0
    monkeypatch.setattr(cli.claude_client, "batch_status", lambda batch_id: "ended")
    monkeypatch.setattr(
        cli.claude_client,
        "batch_results",
        lambda batch_id, schema, model: iter(
            [
                claude_client.BatchEntry(
                    enrich.batch_custom_id(item.id),
                    "succeeded",
                    "",
                    ok("to ask"),
                )
            ]
        ),
    )

    assert cli.main(
        [
            "--root",
            str(root),
            "enrich",
            "--polish-meanings",
            "--batch-fetch",
            "--yes",
        ]
    ) == 0

    assert stored(root)[item.id]["meanings"] == ["to ask"]
    assert book_of(root)["pending_batches"] == {}
    assert book_of(root)["records"][item.id]["enriched"][0]["kind"] == "polish"


def test_fetch_recovers_provenance_after_meanings_land_but_ledger_does_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    item = record()
    root = project(tmp_path, [item])
    _patch_polish_batch_submit(monkeypatch, [])
    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-submit"]
    ) == 0
    monkeypatch.setattr(cli.claude_client, "batch_status", lambda batch_id: "ended")
    monkeypatch.setattr(
        cli.claude_client,
        "batch_results",
        lambda batch_id, schema, model: iter(
            [
                claude_client.BatchEntry(
                    enrich.batch_custom_id(item.id),
                    "succeeded",
                    "",
                    ok("to ask"),
                )
            ]
        ),
    )
    real_save = cli.ledger.Ledger.save
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("busy ledger")),
    )
    capsys.readouterr()

    assert cli.main(
        [
            "--root",
            str(root),
            "enrich",
            "--polish-meanings",
            "--batch-fetch",
            "--yes",
        ]
    ) == 1

    assert stored(root)[item.id]["meanings"] == ["to ask"]
    recovery = cli.ledger.load_batch_recovery(root / "ledger.json")
    assert recovery is not None
    assert recovery.phase == "applied"
    assert recovery.accepted_meanings == {item.id: ["to ask"]}
    assert "provenance" in capsys.readouterr().err

    monkeypatch.setattr(cli.ledger.Ledger, "save", real_save)
    monkeypatch.setattr(
        cli.claude_client,
        "batch_status",
        lambda batch_id: pytest.fail("recovery must finish before polling Anthropic"),
    )
    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-fetch"]
    ) == 0

    payload = book_of(root)
    assert payload["pending_batches"] == {}
    assert payload["records"][item.id]["enriched"][0]["kind"] == "polish"
    assert cli.ledger.load_batch_recovery(root / "ledger.json") is None


def test_fetch_ignores_an_answer_when_its_prompt_inputs_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    item = record()
    root = project(tmp_path, [item])
    _patch_polish_batch_submit(monkeypatch, [])
    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-submit"]
    ) == 0
    changed = record(meanings=["human correction"])
    (root / "vocabulary.json").write_text(
        json.dumps([changed.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(cli.claude_client, "batch_status", lambda batch_id: "ended")
    monkeypatch.setattr(
        cli.claude_client,
        "batch_results",
        lambda batch_id, schema, model: iter(
            [
                claude_client.BatchEntry(
                    enrich.batch_custom_id(item.id), "succeeded", "", ok("to ask")
                )
            ]
        ),
    )
    capsys.readouterr()

    assert cli.main(
        [
            "--root",
            str(root),
            "enrich",
            "--polish-meanings",
            "--batch-fetch",
            "--yes",
        ]
    ) == 0

    assert stored(root)[item.id]["meanings"] == ["human correction"]
    assert book_of(root)["pending_batches"] == {}
    stderr = capsys.readouterr().err
    assert "changed after submission" in stderr
    assert item.id in stderr


def test_fetch_refuses_an_older_polish_batch_without_input_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    item = record()
    root = project(tmp_path, [item])
    _patch_polish_batch_submit(monkeypatch, [])
    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-submit"]
    ) == 0
    book = cli.ledger.load(root / "ledger.json")
    book.pending_batches["msgbatch_polish"].pop("prompt_fingerprints")
    book.save()
    monkeypatch.setattr(cli.claude_client, "batch_status", lambda batch_id: "ended")
    capsys.readouterr()

    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-fetch"]
    ) == 1

    assert "predates polish input tracking" in capsys.readouterr().err
    assert list(book_of(root)["pending_batches"]) == ["msgbatch_polish"]


def test_an_invalid_answer_for_a_deleted_record_does_not_hold_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deleted = record()
    surviving = record(id="word:話す:はなす", expression="話す", reading="はなす")
    root = project(tmp_path, [deleted, surviving])
    _patch_polish_batch_submit(monkeypatch, [])
    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-submit"]
    ) == 0
    (root / "vocabulary.json").write_text(
        json.dumps([surviving.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(cli.claude_client, "batch_status", lambda batch_id: "ended")
    monkeypatch.setattr(
        cli.claude_client,
        "batch_results",
        lambda batch_id, schema, model: iter(
            [
                claude_client.BatchEntry(
                    enrich.batch_custom_id(deleted.id), "invalid", "bad shape", None
                ),
                claude_client.BatchEntry(
                    enrich.batch_custom_id(surviving.id), "succeeded", "", ok()
                ),
            ]
        ),
    )
    capsys.readouterr()

    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-fetch"]
    ) == 0

    assert book_of(root)["pending_batches"] == {}
    stderr = capsys.readouterr().err
    assert deleted.id in stderr
    assert "no longer in the collection" in stderr


def test_quitting_batch_review_persists_exactly_what_is_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [record(), record(id="word:話す:はなす", expression="話す")]
    root = project(tmp_path, records)
    _patch_polish_batch_submit(monkeypatch, [])
    cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-submit"]
    )
    monkeypatch.setattr(cli.claude_client, "batch_status", lambda batch_id: "ended")
    monkeypatch.setattr(
        cli.claude_client,
        "batch_results",
        lambda batch_id, schema, model: iter(
            [
                claude_client.BatchEntry(
                    enrich.batch_custom_id(records[0].id), "succeeded", "", ok("to ask")
                ),
                claude_client.BatchEntry(
                    enrich.batch_custom_id(records[1].id), "succeeded", "", ok("to speak")
                ),
            ]
        ),
    )
    answers(monkeypatch, "y", "q")
    capsys.readouterr()

    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-fetch"]
    ) == 0

    pending = book_of(root)["pending_batches"]["msgbatch_polish"]
    assert pending["retry_ids"] == [records[1].id]
    assert stored(root)[records[0].id]["meanings"] == ["to ask"]
    assert "fetching again resumes" in capsys.readouterr().out

    answers(monkeypatch, "y")
    assert cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--batch-fetch"]
    ) == 0

    assert stored(root)[records[1].id]["meanings"] == ["to speak"]
    assert book_of(root)["pending_batches"] == {}
    assert records[0].id not in capsys.readouterr().out


# --- the flag belongs to one pass --------------------------------------------


def test_polish_is_not_run_alongside_another_pass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["enrich", "--ai", "--polish-meanings"]) == 1
    assert "one pass at a time" in capsys.readouterr().err


def test_force_fields_has_nothing_to_widen_here(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])

    code = cli.main(
        [
            "--root", str(root), "enrich", "--polish-meanings",
            "--force-fields", "meanings",
        ]
    )

    assert code == 1
    assert "--polish-meanings writes 'meanings'" in capsys.readouterr().err


def test_the_staging_annotation_is_a_different_job(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    held = root / "held.yaml"
    held.write_text("records: []\n", encoding="utf-8")

    code = cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--staging", str(held)]
    )

    assert code == 1
    assert "--polish-meanings" in capsys.readouterr().err


def test_force_is_still_refused_because_it_writes_no_staging_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])

    code = cli.main(
        ["--root", str(root), "enrich", "--polish-meanings", "--force", "--yes"]
    )

    assert code == 1
    assert "--force" in capsys.readouterr().err


# --- what the review found ---------------------------------------------------


def test_ctrl_c_during_a_call_still_writes_what_was_already_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The call is the slow step, so it is where a Ctrl-C most often lands.
    Interrupting there must end the same way interrupting at the prompt does."""
    root = project(
        tmp_path,
        [record(), record(id="word:話す:はなす", expression="話す", meanings=["to talk"])],
    )

    class Interrupts(FakeCall):
        def __call__(self, *args: Any, **kw: Any) -> CallResult:
            if self.calls:
                raise KeyboardInterrupt
            return super().__call__(*args, **kw)

    patch_all(monkeypatch, Interrupts(ok("to ask")))
    answers(monkeypatch, "y")

    assert cli.main(["--root", str(root), "enrich", "--polish-meanings"]) == 0

    assert stored(root)["word:聞く:きく"]["meanings"] == ["to ask"]
    assert "Stopped" in capsys.readouterr().out


def test_a_failed_ledger_write_does_not_promise_a_recovery_that_never_happens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """'status --rebuild' reconstructs what the records prove, and a filled
    field does not say who filled it. Sending someone after that fix would
    report success while the entry stayed gone."""
    root = project(tmp_path, [record()])
    patch_all(monkeypatch, FakeCall(ok("to ask")))
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("disk full")),
    )

    assert cli.main(["--root", str(root), "enrich", "--polish-meanings", "--yes"]) == 1

    err = capsys.readouterr().err
    assert "'status --rebuild' cannot bring it back" in err
    assert "to recover the entries" not in err
    # This pass looks at every record every time, so the fill passes' "a re-run
    # finds nothing to do" is false here — and it would cost a call per record.
    assert "not a free repair" in err
    assert "skips these records" not in err
    # The records themselves are still written; only the note about them is not.
    assert stored(root)["word:聞く:きく"]["meanings"] == ["to ask"]


def test_the_model_flag_says_which_passes_honor_it() -> None:
    parser = cli.build_parser()
    enrich_parser = parser._subparsers._group_actions[0].choices["enrich"]  # noqa: SLF001
    (action,) = [item for item in enrich_parser._actions if item.dest == "model"]  # noqa: SLF001

    assert "--polish-meanings" in action.help
