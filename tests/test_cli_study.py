"""``janki study`` over the real services, with a faked provider.

Not a fake core: these run the same application services the Assistant's own
routes call, so a divergence between the two surfaces would show up here. What
is faked is the provider — the subscription transport is a canned stream, and
no Anthropic client is ever built.

The three things the command line owes:

1. **A local job write asks nothing.** Opening a job over an existing deck, or
   creating a deck and opening the job inside that same action, spends nothing
   and needs no consent prompt.
2. **`--yes` answers a prompt that already exists.** An unattended `extract`
   without it refuses after planning and before sending.
3. **Recovery is by exact id.** `study recover` refuses an operation that
   belongs to another job, and `study resume` finds an interrupted batch by
   the ids and hashes the job reserved.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_revision_provider import FakeClaudeRunner, _which
from test_workbench_fixtures import RESPONSES

from conftest import seed_prompts
from japanese_anki import card_preview, claude_client, cli
from japanese_anki.application import extraction_batch, study_job
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "table_exhaustive"
#: The one real PDF this suite renders, for the one command that renders:
#: hand-written, deterministic, no Japanese and no private source.
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "synthetic_table.pdf"


def _answer() -> dict[str, Any]:
    return json.loads((RESPONSES / f"{SCENARIO}.json").read_text(encoding="utf-8"))


def _stream(answer: Any) -> bytes:
    fixture = Path(__file__).parent / "fixtures" / "claude-code-stream.ndjson"
    lines = []
    for line in fixture.read_bytes().splitlines(keepends=True):
        payload = json.loads(line)
        if payload.get("type") == "result":
            payload["structured_output"] = answer
            line = json.dumps(payload).encode("utf-8") + b"\n"
        lines.append(line)
    return b"".join(lines)


def _no_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Anthropic API was reached")

    monkeypatch.setattr(claude_client, "prepare_paid_client", refuse)
    monkeypatch.setattr(claude_client, "parse_call", refuse)


def _project(tmp_path: Path) -> ProjectConfig:
    root = tmp_path / "project"
    root.mkdir()
    seed_prompts(root)
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n'
        'staging_dir = "staging"\n'
        'scan_inbox = "inbox"\n'
        'patterns_file = "patterns.json"\n'
        "[ai]\n"
        'extract_provider = "claude-code"\n'
        'extract_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    shutil.copytree(PROJECT_ROOT / "templates", root / "templates")
    (root / "decks").mkdir()
    (root / "media").mkdir()
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    return ProjectConfig.load(root)


def _deck(config: ProjectConfig, name: str = "lesson") -> Path:
    path = config.deck_dir / f"{name}.yaml"
    path.write_text(
        "deck:\n"
        f"  name: {name.title()} deck\n"
        "  source: ../vocabulary.json\n"
        "  include_tags: [lesson-intake]\n"
        "  intake_tag: lesson-intake\n"
        "  cards:\n"
        "    recognition: true\n"
        "    production: true\n",
        encoding="utf-8",
    )
    return path


def _source(config: ProjectConfig, name: str, body: bytes = b"page") -> Path:
    path = config.scan_inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 " + body)
    return path


def _run(config: ProjectConfig, *args: str) -> int:
    return cli.main(["--root", str(config.root), *args])


def _published_parts(
    config: ProjectConfig, job_id: str, names: tuple[str, ...]
) -> str:
    """One receipt over parts that are really in the corpus, bound by the job.

    Hand-built rather than rendered: what these tests are about is the job's
    binding to a publication, not PDFium's output, and a real render would
    make every one of them depend on an optional dependency.
    """

    recipe_id = "3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908"
    directory = config.operations_file.parent / "source_parts"
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    for ordinal, name in enumerate(names, start=1):
        data = b"%PDF-1.7 " + name.encode("ascii")
        _source(config, name, name.encode("ascii"))
        parts.append(
            {
                "ordinal": ordinal,
                "target_name": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "byte_length": len(data),
                "page_index": ordinal - 1,
                "page_rotate": 0,
                "page_size_pt": [612.0, 792.0],
                "pixel_rect": [0, 0, 100, 100],
                "regions": [],
            }
        )
    payload = (
        json.dumps(
            {
                "version": 1,
                "recipe_id": recipe_id,
                "parent_name": "verbs.pdf",
                "parent_sha256": "9f2c",
                "renderer": "pdfium",
                "renderer_version": "1",
                "encoder": "pillow",
                "encoder_version": "1",
                "render_dpi": 200,
                "recipe_sha256": "a" * 64,
                "plan_fingerprint": "f" * 64,
                "created_at": "2026-09-09T10:00:00+00:00",
                "parts": parts,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    (directory / f"{recipe_id}.json").write_text(payload, encoding="utf-8")

    job = study_job.load_study_job(config, job_id)
    intent = study_job.ActionIntent(
        intent_id=study_job.new_intent_id(),
        kind="source_parts",
        decided_at="2026-09-09T10:00:00+00:00",
        reserves={
            "recipe_id": recipe_id,
            "receipt_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        },
        bindings={},
    )
    saved = study_job.append_intent(
        config, job_id, intent, expected_revision=job.revision
    )
    study_job.append_outcome(
        config,
        job_id,
        study_job.IntentOutcome(
            intent_id=intent.intent_id,
            state="applied",
            at="2026-09-09T10:01:00+00:00",
        ),
        expected_revision=saved.revision,
    )
    return recipe_id


def _one_job(config: ProjectConfig) -> str:
    ids = study_job.list_study_jobs(config)
    assert len(ids) == 1
    return ids[0]


def test_study_new_binds_an_existing_deck_without_asking_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _project(tmp_path)
    source = _source(config, "verbs.pdf", b"verbs")
    deck = _deck(config)

    assert _run(config, "study", "new", "--source", "verbs.pdf", "--deck", "lesson") == 0

    job = study_job.load_study_job(config, _one_job(config))
    assert job.header.parent_source_name == "verbs.pdf"
    assert job.header.parent_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert job.header.deck_path == "decks/lesson.yaml"
    assert job.header.deck_sha256 == hashlib.sha256(deck.read_bytes()).hexdigest()
    assert job.intents == () and job.outcomes == () and job.choices == {}
    assert job.header.job_id in capsys.readouterr().out
    # Nothing paid, nothing journalled.
    assert not config.operations_file.exists()


def test_study_new_create_deck_makes_the_deck_and_the_job_in_one_action(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One deck-creation action; the local job write rides inside it."""

    config = _project(tmp_path)
    _source(config, "verbs.pdf", b"verbs")

    assert (
        _run(
            config,
            "study",
            "new",
            "--source",
            "verbs.pdf",
            "--create-deck",
            "201 Verbs",
            "--standalone",
            "--directions",
            "recognition",
            "production",
        )
        == 0
    )

    job = study_job.load_study_job(config, _one_job(config))
    deck = config.root / job.header.deck_path
    assert deck.is_file()
    # The hash bound is the one the deck's own writer produced, so a
    # concurrent edit between the write and a re-read cannot change it.
    assert job.header.deck_sha256 == hashlib.sha256(deck.read_bytes()).hexdigest()
    printed = capsys.readouterr().out
    assert str(deck) in printed
    assert job.header.job_id in printed


def test_study_new_refuses_directions_for_an_existing_deck(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A job never changes a deck's saved card set; §2.6 is not negotiable."""

    config = _project(tmp_path)
    _source(config, "verbs.pdf", b"verbs")
    _deck(config)

    code = _run(
        config,
        "study",
        "new",
        "--source",
        "verbs.pdf",
        "--deck",
        "lesson",
        "--directions",
        "reading",
    )

    assert code == 1
    assert study_job.list_study_jobs(config) == ()
    assert "settled when a deck is created" in capsys.readouterr().out


def test_study_extract_refuses_unattended_after_planning_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_api(monkeypatch)
    monkeypatch.setattr(
        "japanese_anki.cli_study.sys.stdin", type("_NoTty", (), {"isatty": lambda self: False})()
    )
    config = _project(tmp_path)
    _source(config, "verbs.pdf", b"verbs")
    _deck(config)
    _run(config, "study", "new", "--source", "verbs.pdf", "--deck", "lesson")
    job_id = _one_job(config)
    _published_parts(config, job_id, ("verbs-p1.pdf", "verbs-p2.pdf"))
    monkeypatch.setattr(
        extraction_batch,
        "plan_extraction_batch",
        _planner(config),
    )

    code = _run(config, "study", "extract", job_id)

    printed = capsys.readouterr().out
    assert code == 1
    assert "Nothing was sent." in printed
    assert "Consent fingerprint:" in printed
    # Planning journals nothing and records no intent.
    assert not config.operations_file.exists()
    assert study_job.load_study_job(config, job_id).intents[-1].kind == "source_parts"


def _planner(config: ProjectConfig) -> Any:
    """The real planner with this suite's faked subscription transport."""

    real = extraction_batch.plan_extraction_batch

    def plan(conf: Any, sources: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("provider_env", {})
        kwargs.setdefault("provider_runner", FakeClaudeRunner(reply=_stream(_answer())))
        kwargs.setdefault("provider_which", _which)
        return real(conf, sources, **kwargs)

    return plan


def _dispatcher() -> Any:
    real = extraction_batch.dispatch_extraction_batch

    def dispatch(conf: Any, plan: Any, **kwargs: Any) -> Any:
        runner = FakeClaudeRunner(reply=_stream(_answer()))
        kwargs.setdefault("provider_env", {})
        kwargs.setdefault("provider_runner", runner)
        kwargs.setdefault("provider_which", _which)
        kwargs.setdefault("provider_spawn", runner.spawn)
        return real(conf, plan, **kwargs)

    return dispatch


def test_study_extract_with_yes_sends_the_batch_and_records_its_backlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_api(monkeypatch)
    config = _project(tmp_path)
    _source(config, "verbs.pdf", b"verbs")
    _deck(config)
    _run(config, "study", "new", "--source", "verbs.pdf", "--deck", "lesson")
    job_id = _one_job(config)
    _published_parts(config, job_id, ("verbs-p1.pdf", "verbs-p2.pdf"))
    monkeypatch.setattr(extraction_batch, "plan_extraction_batch", _planner(config))
    monkeypatch.setattr(
        study_job.extraction_batch, "dispatch_extraction_batch", _dispatcher()
    )

    assert _run(config, "study", "extract", job_id, "--yes") == 0

    saved = study_job.load_study_job(config, job_id)
    batch_intent = next(
        intent for intent in saved.intents if intent.kind == "extract_batch"
    )
    batch_id = batch_intent.reserves["batch_id"]
    manifest = extraction_batch.batch_manifest_path(config, batch_id)
    assert json.loads(manifest.read_text(encoding="utf-8"))["job_id"] == job_id
    assert batch_intent.reserves["manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert [outcome.state for outcome in saved.outcomes] == ["applied", "applied"]

    status = study_job.study_job_status(config, job_id)
    assert status.batches[0].committed_count == 2
    assert "2 settled" in capsys.readouterr().out.replace(
        "2 settled, 0 in flight", "2 settled, 0 in flight"
    )


def _resumer() -> Any:
    """The real resume with this suite's faked subscription transport."""

    real = extraction_batch.resume_extraction_batch

    def resume(conf: Any, batch_id: str, **kwargs: Any) -> Any:
        runner = FakeClaudeRunner(reply=_stream(_answer()))
        kwargs.setdefault("provider_env", {})
        kwargs.setdefault("provider_runner", runner)
        kwargs.setdefault("provider_which", _which)
        kwargs.setdefault("provider_spawn", runner.spawn)
        return real(conf, batch_id, **kwargs)

    return resume


def test_study_resume_continues_a_paused_batch_and_agrees_with_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two surfaces of one milestone say the same thing about one job.

    One part's transport preflight refuses, so `extract` returns with that
    child reserved, unsent and unspent. `status` says an action has no outcome
    and points at `resume`; `resume` has to be able to finish exactly that
    recorded authority rather than reporting there is nothing to do.
    """

    _no_api(monkeypatch)
    config = _project(tmp_path)
    _source(config, "verbs.pdf", b"verbs")
    _deck(config)
    _run(config, "study", "new", "--source", "verbs.pdf", "--deck", "lesson")
    job_id = _one_job(config)
    _published_parts(config, job_id, ("verbs-p1.pdf", "verbs-p2.pdf"))
    monkeypatch.setattr(extraction_batch, "plan_extraction_batch", _planner(config))
    monkeypatch.setattr(
        study_job.extraction_batch, "dispatch_extraction_batch", _dispatcher()
    )
    real_prepare = extraction_batch.prepare_extraction_transport

    def refuse_second(call_plan: Any, target: Any, **kwargs: Any) -> Any:
        if target.name == "verbs-p2.pdf":
            raise JankiError("temporary login refusal")
        return real_prepare(call_plan, target, **kwargs)

    monkeypatch.setattr(
        extraction_batch, "prepare_extraction_transport", refuse_second
    )

    assert (
        _run(config, "study", "extract", job_id, "--yes", "--concurrency", "1") == 0
    )
    assert "1 settled, 1 in flight or unsent" in capsys.readouterr().out

    assert _run(config, "study", "status", job_id) == 0
    stalled = capsys.readouterr().out
    assert "1 recorded action(s) have no outcome" in stalled
    assert "verbs-p2.pdf — authorized" in stalled
    # Read from the batch's own durable eligibility, so the two surfaces of
    # this milestone cannot contradict each other over the same batch.
    assert "`janki study resume` can continue this batch" in stalled

    monkeypatch.setattr(
        extraction_batch, "prepare_extraction_transport", real_prepare
    )
    monkeypatch.setattr(
        study_job.extraction_batch, "resume_extraction_batch", _resumer()
    )
    before = json.loads(config.operations_file.read_text(encoding="utf-8"))

    assert _run(config, "study", "resume", job_id) == 0

    resumed = capsys.readouterr().out
    assert "Nothing in this job is waiting" not in resumed
    assert "2 settled, 0 in flight or unsent" in resumed
    # No new authority was bought to finish it: the same operation ids.
    after = json.loads(config.operations_file.read_text(encoding="utf-8"))
    assert set(after["operations"]) == set(before["operations"])

    assert _run(config, "study", "status", job_id) == 0
    settled = capsys.readouterr().out
    assert "recorded action(s) have no outcome" not in settled

    assert _run(config, "study", "resume", job_id) == 0
    assert "Nothing in this job is waiting" in capsys.readouterr().out


def _killed_dispatcher() -> Any:
    """The real dispatch, with the machine dying while the second call is out.

    The provider process is spawned only after `claim_batch_dispatch` has
    consumed that child's authority, so a `KeyboardInterrupt` out of the spawn
    is the hard kill that leaves a journal entry `dispatching` with no answer:
    money may already have gone, and what it bought is the owner's to settle.
    """

    real = extraction_batch.dispatch_extraction_batch

    def dispatch(conf: Any, plan: Any, **kwargs: Any) -> Any:
        runner = FakeClaudeRunner(reply=_stream(_answer()))
        sent: list[int] = []

        def spawn(command: list[str], **spawn_kwargs: Any) -> Any:
            sent.append(1)
            if len(sent) == 2:
                raise KeyboardInterrupt("killed with a call in flight")
            return runner.spawn(command, **spawn_kwargs)

        kwargs.setdefault("provider_env", {})
        kwargs.setdefault("provider_runner", runner)
        kwargs.setdefault("provider_which", _which)
        kwargs.setdefault("provider_spawn", spawn)
        return real(conf, plan, **kwargs)

    return dispatch


def test_study_resume_and_status_agree_that_an_in_flight_call_blocks_a_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One batch, one answer — including when the answer is "not yet".

    The process was killed between a child's dispatch claim and its
    settlement, so that call may have been paid for and the batch's own
    eligibility refuses to start another. `status` prints that refusal;
    `resume` may not print the opposite. Ending the call is the owner's
    decision, and janki takes it for nobody.
    """

    _no_api(monkeypatch)
    config = _project(tmp_path)
    _source(config, "verbs.pdf", b"verbs")
    _deck(config)
    _run(config, "study", "new", "--source", "verbs.pdf", "--deck", "lesson")
    job_id = _one_job(config)
    _published_parts(config, job_id, ("verbs-p1.pdf", "verbs-p2.pdf"))
    monkeypatch.setattr(extraction_batch, "plan_extraction_batch", _planner(config))
    monkeypatch.setattr(
        study_job.extraction_batch, "dispatch_extraction_batch", _killed_dispatcher()
    )

    with pytest.raises(KeyboardInterrupt):
        _run(config, "study", "extract", job_id, "--yes", "--concurrency", "1")
    capsys.readouterr()

    assert _run(config, "study", "status", job_id) == 0
    stalled = capsys.readouterr().out
    refusal = (
        "This batch cannot be resumed: a call may be in flight right now, so "
        "janki will not start another."
    )
    assert "verbs-p2.pdf — dispatching" in stalled
    assert "1 recorded action(s) have no outcome" in stalled
    assert refusal in stalled
    assert "`janki study resume` can continue this batch" not in stalled

    journalled = config.operations_file.read_bytes()
    revision = study_job.load_study_job(config, job_id).revision

    assert _run(config, "study", "resume", job_id) == 0

    resumed = capsys.readouterr().out
    assert refusal in resumed
    assert "still hold reserved authority" not in resumed
    # And the one step that actually unblocks it, which is a person's.
    assert "janki operations --end" in resumed
    # Nothing was ended, sent, or recorded to get that answer.
    assert config.operations_file.read_bytes() == journalled
    assert study_job.load_study_job(config, job_id).revision == revision
    assert len(study_job.study_job_status(config, job_id).open_intents) == 1


def test_study_parts_select_records_which_published_parts_the_batch_covers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI half of the one S4 job choice that has a writer.

    A reversible local preference through the same compare-and-swap service
    the Assistant's own control calls: no recipe, no render, no publication and
    no model call. `janki study extract` then covers exactly what it names.
    """

    _no_api(monkeypatch)
    config = _project(tmp_path)
    _source(config, "verbs.pdf", b"verbs")
    _deck(config)
    _run(config, "study", "new", "--source", "verbs.pdf", "--deck", "lesson")
    job_id = _one_job(config)
    _published_parts(config, job_id, ("verbs-p1.pdf", "verbs-p2.pdf"))
    capsys.readouterr()

    assert (
        _run(config, "study", "parts", job_id, "--select", "verbs-p2.pdf") == 0
    )

    printed = capsys.readouterr().out
    assert "covers 1 of 2 published part(s): verbs-p2.pdf" in printed
    saved = study_job.load_study_job(config, job_id)
    assert saved.choices == {"part_selections": ["verbs-p2.pdf"]}
    assert [
        path.name
        for path, _lineage in study_job.job_part_sources(config, job_id)
    ] == ["verbs-p2.pdf"]

    # A part this job never published refuses, and changes nothing.
    assert _run(config, "study", "parts", job_id, "--select", "verbs-p9.pdf") == 1
    assert "verbs-p9.pdf" in capsys.readouterr().out
    assert study_job.load_study_job(config, job_id).choices == {
        "part_selections": ["verbs-p2.pdf"]
    }
    # And this is local work only: nothing was journalled or published.
    assert not config.operations_file.exists()


def test_study_parts_publish_refuses_a_corrupt_receipt_with_an_error_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hand-edited receipt ends this command the ordinary way.

    `--publish` binds the receipt already on disk before it records anything,
    so a receipt corrupted down to `[]` is read by that binding. `cli.main`
    catches `JankiError` and nothing else, so a refusal that is not one leaves
    this command as a traceback instead of the `error:` line every other
    refusal prints. The render here is real, because the plan the publish is
    refused for has to be a real plan.
    """

    _no_api(monkeypatch)
    config = _project(tmp_path)
    config.scan_inbox.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE, config.scan_inbox / "synthetic_table.pdf")
    _deck(config)
    _run(config, "study", "new", "--source", "synthetic_table.pdf", "--deck", "lesson")
    job_id = _one_job(config)
    recipe_id = "9b4a7d21-6c3f-4e18-8a55-7d2c1b0e4f36"
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "version": 1,
                "recipe_id": recipe_id,
                "parent_name": "synthetic_table.pdf",
                "parent_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                "render_dpi": 150,
                "parts": [
                    {
                        "page_index": 0,
                        "page_rotate": 0,
                        "regions": [[0.05, 0.15, 0.95, 0.45]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    directory = config.operations_file.parent / "source_parts"
    directory.mkdir(parents=True, exist_ok=True)
    corrupt = directory / f"{recipe_id}.json"
    corrupt.write_text("[]", encoding="utf-8")
    capsys.readouterr()

    code = _run(config, "study", "parts", job_id, "--recipe", str(recipe), "--publish")

    captured = capsys.readouterr()
    assert code == 1
    assert f"error: The source-part receipt {corrupt}" in captured.err
    assert "Nothing was published." in captured.err
    assert "Traceback" not in captured.err
    # The corruption is evidence: it is read, never repaired or replaced.
    assert corrupt.read_text(encoding="utf-8") == "[]"
    assert sorted(path.name for path in config.scan_inbox.iterdir()) == [
        "synthetic_table.pdf"
    ]
    job = study_job.load_study_job(config, job_id)
    assert job.intents == () and job.outcomes == ()
    assert not config.operations_file.exists()


def test_study_status_derives_every_number_it_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_api(monkeypatch)
    config = _project(tmp_path)
    _source(config, "verbs.pdf", b"verbs")
    _deck(config)
    _run(config, "study", "new", "--source", "verbs.pdf", "--deck", "lesson")
    job_id = _one_job(config)
    recipe_id = _published_parts(config, job_id, ("verbs-p1.pdf",))
    capsys.readouterr()

    assert _run(config, "study", "status") == 0
    listed = capsys.readouterr().out
    assert job_id in listed and "verbs.pdf → decks/lesson.yaml" in listed

    assert _run(config, "study", "status", job_id) == 0
    printed = capsys.readouterr().out
    assert f"Parts {recipe_id}: 1 of 1 published from verbs.pdf" in printed
    assert "source_extraction" in printed


def test_study_recover_refuses_an_operation_from_another_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exact id, not a nearby one: a job never recovers a stranger's reply."""

    _no_api(monkeypatch)
    config = _project(tmp_path)
    _source(config, "verbs.pdf", b"verbs")
    _deck(config)
    _run(config, "study", "new", "--source", "verbs.pdf", "--deck", "lesson")
    mine = _one_job(config)
    _run(config, "study", "new", "--source", "verbs.pdf", "--deck", "lesson")
    theirs = next(job for job in study_job.list_study_jobs(config) if job != mine)

    _published_parts(config, theirs, ("verbs-p1.pdf",))
    monkeypatch.setattr(extraction_batch, "plan_extraction_batch", _planner(config))
    monkeypatch.setattr(
        study_job.extraction_batch, "dispatch_extraction_batch", _dispatcher()
    )
    _run(config, "study", "extract", theirs, "--yes")
    other = study_job.load_study_job(config, theirs)
    batch_id = next(
        intent.reserves["batch_id"]
        for intent in other.intents
        if intent.kind == "extract_batch"
    )
    plan = extraction_batch._load_batch_plan(config, batch_id)
    capsys.readouterr()

    code = _run(
        config,
        "study",
        "recover",
        mine,
        "--operation",
        plan.children[0].operation_id,
    )

    assert code == 1
    printed = capsys.readouterr().out
    assert theirs in printed and "Nothing was staged." in printed


@pytest.mark.skipif(
    card_preview.preview_unavailable() is not None,
    reason=str(card_preview.preview_unavailable()),
)
def test_study_preview_writes_the_cards_and_prints_their_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_api(monkeypatch)
    config = _project(tmp_path)
    _source(config, "verbs.pdf", b"verbs")
    _deck(config)
    _run(config, "study", "new", "--source", "verbs.pdf", "--deck", "lesson")
    job_id = _one_job(config)
    _published_parts(config, job_id, ("verbs-p1.pdf",))
    monkeypatch.setattr(extraction_batch, "plan_extraction_batch", _planner(config))
    monkeypatch.setattr(
        study_job.extraction_batch, "dispatch_extraction_batch", _dispatcher()
    )
    _run(config, "study", "extract", job_id, "--yes")
    capsys.readouterr()

    output = tmp_path / "preview.html"
    assert _run(config, "study", "preview", job_id, "--output", str(output)) == 0

    printed = capsys.readouterr().out
    fingerprint = next(
        line.split(": ", 1)[1].strip()
        for line in printed.splitlines()
        if line.startswith("Rendered content fingerprint:")
    )
    assert len(fingerprint) == 64
    # The fingerprint is the document's own hash: it binds the card fields and
    # the deck's real templates, and nothing about who looked at it.
    assert hashlib.sha256(output.read_bytes()).hexdigest() == fingerprint
    assert b"Show Answer" in output.read_bytes()
