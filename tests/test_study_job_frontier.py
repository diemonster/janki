"""One study job's effective extraction frontier, and the page it draws.

Contract §9.5 says readiness is computed over *effective children*, derived
from confirmed execution and never from file ordering. These tests build that
history for real: a two-part job whose first part settles and whose second
part fails, then an owner-confirmed retry of the failed part only, through the
same durable manifests, confirmed-execution receipts and journal the services
themselves write. Nothing is stubbed but the provider, which is the faked
subscription every other batch test uses.

What is deliberately *not* here: any check about Japanese. Two proposals
sharing an identity are disclosed with both sources named and one labelled
representative — which of two readings is right is a person's question, and
the frontier answers none of it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_revision_provider import FakeClaudeRunner, _which
from test_workbench_fixtures import RESPONSES

from conftest import seed_prompts
from japanese_anki import card_preview, claude_client, extract, operations, staging
from japanese_anki.application import extraction_batch, source_parts, study_job
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.workbench import assistant_adapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    card_preview.preview_unavailable() is not None,
    reason=str(card_preview.preview_unavailable()),
)

#: Two disjoint answers, so a combined page proves it drew both children
#: rather than one child twice.
PART_ONE_SCENARIO = "table_exhaustive"
PART_TWO_SCENARIO = "shared_word_source_a"

#: The owner's own frozen source form, exactly as `test_study_curation` holds
#: it: opaque column identities, printed positions, the printed strings they
#: recorded and the display labels they chose. Nothing derives any of it, and
#: it is built only through the real `extract.TableLayout.from_wire`.
LAYOUT_WIRE: dict[str, Any] = {
    "layout_id": "layout-7c1f2a",
    "revision": 1,
    "columns": [
        {
            "column_id": "col-9f3a71",
            "ordinal": 1,
            "label_witnesses": ["plain form "],
            "display_label": "Plain",
        },
        {
            "column_id": "col-2b8d04",
            "ordinal": 2,
            "label_witnesses": ["polite form", "polite （ます）"],
            "display_label": "Polite",
        },
    ],
}

#: One filled cell and one printed blank, keyed by the bound column ids, so a
#: layout-bound answer really carries the owner's table onto the page.
SUPPLIED_CELLS: dict[str, str] = {"col-9f3a71": "話す", "col-2b8d04": ""}


def _layout(**overrides: Any) -> extract.TableLayout:
    """A layout revision through the real deserializer, never a stand-in."""

    wire = json.loads(json.dumps(LAYOUT_WIRE))
    wire.update(overrides)
    return extract.TableLayout.from_wire(wire)


# --- the scratch repository ---------------------------------------------------


def _stream(scenario: str, *, cells: dict[str, str] | None = None) -> bytes:
    """One faked subscription reply carrying that scenario's answer.

    ``cells`` keys the answer's conjugations by the bound column ids, which is
    what a layout-bound call really returns; without it the answer supplies no
    keys at all, which `_require_bound_columns` reads as a valid all-absent
    table.
    """

    answer = json.loads((RESPONSES / f"{scenario}.json").read_text(encoding="utf-8"))
    if cells is not None:
        for entry in answer["candidates"]:
            entry["conjugations"] = dict(cells)
    fixture = Path(__file__).parent / "fixtures" / "claude-code-stream.ndjson"
    lines = []
    for line in fixture.read_bytes().splitlines(keepends=True):
        payload = json.loads(line)
        if payload.get("type") == "result":
            payload["structured_output"] = answer
            line = json.dumps(payload).encode("utf-8") + b"\n"
        lines.append(line)
    return b"".join(lines)


def _expressions(scenario: str) -> tuple[str, ...]:
    answer = json.loads((RESPONSES / f"{scenario}.json").read_text(encoding="utf-8"))
    return tuple(entry["expression"] for entry in answer.get("candidates") or ())


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No paid client, and no outbound connection, for every test here.

    The suite's own autouse guards cover the Claude CLI and the Anki
    collection. These are this module's: it renders real cards and runs a real
    batch, and neither may reach a network or the metered API.

    The refusal is at *connect* time rather than by replacing ``socket.socket``.
    Anki's transitive `socks` does ``class _BaseSocket(socket.socket)`` when it
    is imported, so a function in that slot makes the real renderer fail to
    import — a guard that hides the behaviour it is guarding.
    """

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Anthropic API was reached")

    monkeypatch.setattr(claude_client, "prepare_paid_client", refuse)
    monkeypatch.setattr(claude_client, "parse_call", refuse)

    import socket

    def no_connection(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a test opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", no_connection)
    monkeypatch.setattr(socket, "create_connection", no_connection)


def _project(tmp_path: Path) -> ProjectConfig:
    """A scratch repository with the repository's real templates in it."""

    seed_prompts(tmp_path)
    # Outside the repository, which the config reader insists on, and inside
    # the test's own temporary tree, which keeps a real user cache untouched.
    cache = tmp_path.parent / f"{tmp_path.name}-jpdb-cache"
    cache.mkdir(parents=True, exist_ok=True)
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n'
        'staging_dir = "staging"\n'
        'scan_inbox = "inbox"\n'
        'patterns_file = "patterns.json"\n'
        f'jpdb_html_cache = "{cache}"\n'
        "[ai]\n"
        'extract_provider = "claude-code"\n'
        'extract_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    shutil.copytree(PROJECT_ROOT / "templates", tmp_path / "templates")
    (tmp_path / "decks").mkdir()
    (tmp_path / "media").mkdir()
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _deck(config: ProjectConfig, stem: str = "201-verbs") -> Path:
    path = config.deck_dir / f"{stem}.yaml"
    path.write_text(
        "deck:\n"
        f"  name: {stem}\n"
        "  deck_id: 1500000001\n"
        "  source: ../vocabulary.json\n"
        "  include_tags: [lesson-intake]\n"
        "  intake_tag: lesson-intake\n"
        "  cards:\n"
        "    recognition: true\n"
        "    production: true\n",
        encoding="utf-8",
    )
    return path


def _parent(config: ProjectConfig) -> Path:
    path = config.scan_inbox / "verbs.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 verbs")
    return path


class _Part:
    """One rendered part of the parent document, exactly as a recipe holds it."""

    def __init__(self, ordinal: int, name: str, data: bytes) -> None:
        self.ordinal = ordinal
        self.target_name = name
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.byte_length = len(data)
        self.page_index = ordinal - 1
        self.page_rotate = 0
        self.page_size_pt = (612.0, 792.0)
        self.pixel_rect = (0, 0, 100, 100)
        self.regions: tuple[Any, ...] = ()
        self.thumbnail_png_base64 = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "target_name": self.target_name,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "page_index": self.page_index,
            "page_rotate": self.page_rotate,
            "page_size_pt": list(self.page_size_pt),
            "pixel_rect": list(self.pixel_rect),
            "regions": [],
        }


RECIPE_ID = "3f1a0c1e-2b8d-4c6a-9f0e-5d4c3b2a1908"


def _two_part_plan() -> Any:
    payloads = (b"rendered part one", b"rendered part two")
    parts = tuple(
        _Part(ordinal, f"verbs-p{ordinal}.png", data)
        for ordinal, data in enumerate(payloads, start=1)
    )
    return source_parts.SourcePartsPlan(
        recipe_id=RECIPE_ID,
        parent_name="verbs.pdf",
        parent_sha256="9f2c",
        renderer="pdfium",
        renderer_version="1",
        encoder="pillow",
        encoder_version="1",
        render_dpi=200,
        plan_fingerprint="f" * 64,
        parts=parts,
        recipe_sha256="a" * 64,
        payloads=payloads,
    )


def _job_with_two_published_parts(config: ProjectConfig) -> str:
    _parent(config)
    _deck(config)
    job = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=config.scan_inbox / "verbs.pdf",
        deck_path=config.deck_dir / "201-verbs.yaml",
    )
    plan = _two_part_plan()
    study_job.publish_job_source_parts(
        config, job.header.job_id, plan, publish_token=plan.plan_fingerprint
    )
    return job.header.job_id


def _select(config: ProjectConfig, job_id: str, *names: str) -> None:
    """The owner's own reversible narrowing of which parts a batch covers."""

    study_job.record_choice(
        config,
        job_id,
        {"part_selections": list(names)},
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )


def _bind_layout(
    config: ProjectConfig, job_id: str, *names: str
) -> extract.TableLayout:
    """Bind one frozen source form to these published parts, as the owner does.

    Every part of a batch must be bound or none of them is
    (`_batch_mode_and_layouts`), so this binds the whole selection in the one
    compare-and-swap `append_layout` already makes.
    """

    layout = _layout()
    study_job.append_layout(
        config,
        job_id,
        layout,
        bind=names or ("verbs-p1.png", "verbs-p2.png"),
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )
    return layout


#: A spawn that dies with the request already written: nothing was captured,
#: so what the call bought is genuinely unknown. This is how the journal really
#: reaches ``outcome_unknown`` — its transition table has no move there from
#: ``result_captured``, so a test that "advances" a captured child is inventing
#: a state janki refuses to record.
UNCAPTURED = object()

#: Part two's second answer in the ordinary two-part history: a paid reply that
#: reached the disk and could not be parsed, which leaves ``result_captured``.
UNUSABLE_ANSWER = b"not a stream at all\n"


class _ScriptedProvider:
    """One faked subscription whose reply depends on which call this is.

    The batch runs at concurrency one in these tests, so call order is the
    plan's own child order and a scripted list is exact rather than racy.
    """

    def __init__(self, replies: list[Any]) -> None:
        self.runner = FakeClaudeRunner(reply=b"")
        self.replies = list(replies)
        self.spawned = 0

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        return self.runner(command, **kwargs)

    def spawn(self, command: list[str], **kwargs: Any) -> Any:
        reply = self.replies[min(self.spawned, len(self.replies) - 1)]
        self.spawned += 1
        if reply is UNCAPTURED:
            raise RuntimeError("the CLI died with the request already written")
        self.runner.reply = reply
        return self.runner.spawn(command, **kwargs)


def _first_batch(
    config: ProjectConfig,
    job_id: str,
    *,
    second: Any = UNUSABLE_ANSWER,
    cells: dict[str, str] | None = None,
) -> Any:
    """Part one settles; part two does not. Retryable either way."""

    provider = _ScriptedProvider([_stream(PART_ONE_SCENARIO, cells=cells), second])
    plan = study_job.plan_job_extraction_batch(
        config,
        job_id,
        concurrency_limit=1,
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
    )
    outcome = study_job.dispatch_job_batch(
        config,
        job_id,
        plan,
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
        provider_spawn=provider.spawn,
    )
    states = {child.index: child.state for child in outcome.children}
    assert states[1] == "committed", states
    assert states[2] != "committed", states
    return plan


def _confirmed_retry(
    config: ProjectConfig,
    job_id: str,
    batch_id: str,
    *,
    scenario: str = PART_TWO_SCENARIO,
    cells: dict[str, str] | None = None,
) -> Any:
    """The owner's confirmed second attempt at part two, and nothing else.

    ``scenario`` is the answer that second attempt gets back. A layout-bound
    call has to be answered by a *table* answer — `table-layout` is one of the
    table modes, and `extract` refuses prose candidates under it — so a bound
    history sends the suite's one table fixture for both parts.
    """

    provider = _ScriptedProvider([_stream(scenario, cells=cells)])
    retry = study_job.plan_job_batch_retry(
        config,
        job_id,
        batch_id,
        [2],
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
    )
    outcome = study_job.dispatch_job_batch(
        config,
        job_id,
        retry,
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
        provider_spawn=provider.spawn,
    )
    assert outcome.committed_count == 1, outcome
    return retry


def _selected_batch(config: ProjectConfig, job_id: str, reply: Any) -> Any:
    """One real batch over whatever this job currently selects, and its outcome.

    The whole public journey for a job that sends its parts one at a time: the
    owner narrows the selection, this plans and dispatches exactly that, and
    the journal's own blocking rule still applies between them.
    """

    provider = _ScriptedProvider([reply])
    plan = study_job.plan_job_extraction_batch(
        config,
        job_id,
        concurrency_limit=1,
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
    )
    return plan, study_job.dispatch_job_batch(
        config,
        job_id,
        plan,
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
        provider_spawn=provider.spawn,
    )


def _digest_tree(config: ProjectConfig) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(config.root.rglob("*")):
        if path.is_file():
            found[str(path.relative_to(config.root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return found


# --- the frontier itself ------------------------------------------------------


def test_a_confirmed_retry_supersedes_only_the_child_it_discarded(
    tmp_path: Path,
) -> None:
    """One effective child per part, and the old attempt kept as history."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    retry = _confirmed_retry(config, job_id, first.batch_id)

    frontier = study_job.job_effective_frontier(config, job_id)

    assert [attempt.ref.source_name for attempt in frontier.effective] == [
        "verbs-p1.png",
        "verbs-p2.png",
    ]
    part_one, part_two = frontier.effective
    # Part one's settled child is the *first* batch's. A retry of its sibling
    # does not move it, and the job is not one batch.
    assert part_one.ref.batch_id == first.batch_id
    assert part_one.ref.index == 1
    assert part_one.state == "committed"
    assert part_two.ref.batch_id == retry.batch_id
    assert part_two.ref.index == 1
    assert part_two.state == "committed"
    # Equal indices from different batches, told apart by their own refs.
    assert part_one.ref != part_two.ref
    assert {ref.intent_id for ref in (part_one.ref, part_two.ref)} == {
        intent.intent_id
        for intent in study_job.load_study_job(config, job_id).intents
        if intent.kind in ("extract_batch", "retry")
    }

    # Each effective attempt names the document its own answer wrote, which is
    # what a finish reads rather than composing a path from a file name.
    for attempt, batch in ((part_one, first), (part_two, retry)):
        record = extraction_batch.read_batch_execution_record(config, batch.batch_id)
        assert attempt.staging_path == record.child(attempt.ref.index).staging_path
        assert attempt.staging_path.is_file()
    assert part_one.staging_path != part_two.staging_path
    assert all(attempt.complete for attempt in frontier.effective)
    assert frontier.unsettled == ()

    # The discarded attempt is history with its disclosed cost, not deleted.
    assert len(frontier.superseded) == 1
    (old,) = frontier.superseded
    assert old.ref.batch_id == first.batch_id
    assert old.ref.index == 2
    assert old.ref.source_name == "verbs-p2.png"
    assert old.superseded_by == part_two.ref
    assert old.retired_state not in ("", "committed")
    # A paid answer reached the disk before this attempt was retired, and the
    # receipt's snapshot is now the only record that money may have moved.
    assert old.retired_billed
    assert old.operation_id == first.children[1].operation_id
    assert frontier.prepared == ()
    assert frontier.unresolved == ()


def test_a_confirmed_retry_whose_receipt_vanished_refuses_by_name(
    tmp_path: Path,
) -> None:
    """A retry that really ran, with its confirmation gone, is not a reversal.

    The receipt is what says an execution was confirmed, and this one's is
    missing while the journal still shows the batch reserved and its old
    sibling retired. Quietly falling back on the old attempt would present a
    superseded child as current, so both batches are named instead.
    """

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)

    before = study_job.job_effective_frontier(config, job_id)
    assert before.effective[1].ref.batch_id == first.batch_id
    assert before.effective[1].state != "committed"

    retry = _confirmed_retry(config, job_id, first.batch_id)
    receipt = extraction_batch.execution_receipt_path(config, retry.batch_id)
    assert receipt.is_file()
    receipt.unlink()

    with pytest.raises(JankiError) as caught:
        study_job.job_effective_frontier(config, job_id)

    # The retry really ran, so its evidence being gone is a refusal rather than
    # a silent reversal: both children are named.
    assert first.batch_id in str(caught.value)
    assert retry.batch_id in str(caught.value)


def test_a_retry_manifest_written_before_its_receipt_is_merely_prepared(
    tmp_path: Path,
) -> None:
    """A planned retry that never reserved anything leaves the frontier alone."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)

    provider = _ScriptedProvider([_stream(PART_TWO_SCENARIO)])
    retry = study_job.plan_job_batch_retry(
        config,
        job_id,
        first.batch_id,
        [2],
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
    )
    # Exactly what a crash between `_write_batch_manifest` and
    # `_write_execution_receipt` leaves: the request, with nobody's
    # confirmation of it and nothing reserved.
    manifest = extraction_batch.batch_manifest_path(config, retry.batch_id)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_bytes(retry.manifest_bytes)
    study_job.append_intent(
        config,
        job_id,
        study_job.ActionIntent(
            intent_id=study_job.new_intent_id(),
            kind="retry",
            decided_at="2024-05-05T00:00:00+00:00",
            reserves={
                "batch_id": retry.batch_id,
                "manifest_sha256": retry.manifest_sha256,
                "child_operation_ids": [
                    child.operation_id for child in retry.children
                ],
            },
            bindings={"job_id": job_id, "retry_of": retry.retry_of},
        ),
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )

    frontier = study_job.job_effective_frontier(config, job_id)

    # Part two's effective child is still the first batch's failed one, and the
    # prepared request is disclosed rather than counted.
    assert frontier.effective[1].ref.batch_id == first.batch_id
    assert frontier.effective[1].ref.index == 2
    assert frontier.superseded == ()
    assert [ref.batch_id for ref in frontier.prepared] == [retry.batch_id]
    # The journal still holds the old attempt: nothing was retired.
    journal = operations.OperationJournal.load(config.operations_file)
    assert first.children[1].operation_id in journal.operations


def test_a_reserved_batch_whose_manifest_changed_refuses_by_name(
    tmp_path: Path,
) -> None:
    """An edited manifest for a batch that ran is not "not settled yet".

    The journal holds this retry's reservation, so it really ran and its
    retirement of part two's first attempt may already have been executed.
    Skipping it would leave the *retired* attempt standing as part two's
    effective child and report the reason as an unsettled source — a silent
    reversal wearing the words of ordinary waiting.
    """

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    retry = _confirmed_retry(config, job_id, first.batch_id)
    journal = operations.OperationJournal.load(config.operations_file)
    assert retry.batch_id in journal.batches

    manifest = extraction_batch.batch_manifest_path(config, retry.batch_id)
    manifest.write_bytes(manifest.read_bytes() + b"\n")

    with pytest.raises(JankiError) as caught:
        study_job.job_effective_frontier(config, job_id)

    message = str(caught.value)
    assert job_id in message
    assert retry.batch_id in message
    # The exact source the journal says this reservation covers, and the exact
    # reason its own artifact could not be matched.
    assert "verbs-p2.png" in message
    assert "hashes to" in message
    # The page refuses with it rather than drawing part one under a frontier
    # that quietly went back to the retired attempt.
    with pytest.raises(JankiError):
        study_job.render_job_preview(config, job_id)


def test_a_reserved_batch_whose_receipt_changed_refuses_by_name(
    tmp_path: Path,
) -> None:
    """The same invariant on the other branch: the owning read is what fails.

    The manifest still hashes to what this job bound, so the job's own
    resolution passes and the refusal comes out of the batch service's reader
    instead. Both failures mean the same thing about a reserved batch, so both
    have to end the same way.
    """

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    retry = _confirmed_retry(config, job_id, first.batch_id)

    receipt = extraction_batch.execution_receipt_path(config, retry.batch_id)
    receipt.write_bytes(receipt.read_bytes() + b"\n")
    # The job's own binding still matches: this is the batch service's reader
    # refusing, not `_resolve_batch`.
    manifest = extraction_batch.batch_manifest_path(config, retry.batch_id)
    assert (
        hashlib.sha256(manifest.read_bytes()).hexdigest() == retry.manifest_sha256
    )

    with pytest.raises(JankiError) as caught:
        study_job.job_effective_frontier(config, job_id)

    message = str(caught.value)
    assert job_id in message
    assert retry.batch_id in message
    assert "verbs-p2.png" in message
    assert "different action" in message


def test_a_confirmed_discard_of_a_committed_child_refuses_by_name(
    tmp_path: Path,
) -> None:
    """§9.5: a successful child is never superseded."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    retry = _confirmed_retry(config, job_id, first.batch_id)

    # A third confirmed record whose retirement names the *settled* child.
    journal = operations.OperationJournal.load(config.operations_file)
    settled = journal.operations[first.children[0].operation_id]
    assert settled.state == "committed"
    _save_confirmed_batch(
        config,
        job_id,
        _copy_of(
            config,
            retry.batch_id,
            batch_id=STRANGER_ID,
            discards=[settled],
            retry_of=first.batch_id,
        ),
        decided_at="2024-05-07T00:00:00+00:00",
    )

    with pytest.raises(JankiError) as caught:
        study_job.job_effective_frontier(config, job_id)

    message = str(caught.value)
    assert first.children[0].operation_id in message
    assert "verbs-p1.png" in message
    assert "committed" in message


def test_a_forked_lineage_refuses_with_both_children_named(
    tmp_path: Path,
) -> None:
    """Two confirmed retries of one attempt is not resolved by "latest"."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    retry = _confirmed_retry(config, job_id, first.batch_id)
    # A second confirmed record retiring exactly the attempt the real retry
    # already retired, built from that retry's own manifest.
    forked = extraction_batch.read_batch_execution_record(config, retry.batch_id)
    second = _save_confirmed_batch(
        config,
        job_id,
        _copy_of(
            config,
            retry.batch_id,
            batch_id=FORK_ID,
            discards=forked.discards,
        ),
        decided_at="2024-05-06T00:00:00+00:00",
    )

    with pytest.raises(JankiError) as caught:
        study_job.job_effective_frontier(config, job_id)

    message = str(caught.value)
    assert retry.batch_id in message
    assert second.batch_id in message


def test_a_retry_of_a_different_source_refuses_as_a_disjoint_chain(
    tmp_path: Path,
) -> None:
    """A discard whose replacement is not the same part is not a retry."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    retry = _confirmed_retry(config, job_id, first.batch_id)

    # A confirmation retiring part *two*'s failed attempt while sending part
    # one's request: an edge with no shared purpose. The retirement it names is
    # the real one the true retry saved, and neither end of this edge is a
    # success, so the only thing wrong with it is that it joins two parts.
    retired = extraction_batch.read_batch_execution_record(
        config, retry.batch_id
    ).discards
    _save_confirmed_batch(
        config,
        job_id,
        _copy_of(
            config,
            first.batch_id,
            batch_id=STRANGER_ID,
            indices=[1],
            discards=retired,
            retry_of=first.batch_id,
        ),
        decided_at="2024-05-07T00:00:00+00:00",
    )

    with pytest.raises(JankiError) as caught:
        study_job.job_effective_frontier(config, job_id)

    message = str(caught.value)
    assert "verbs-p1.png" in message
    assert "verbs-p2.png" in message
    assert STRANGER_ID in message
    # The part, not its ancestry or its form: each of §9.5's lineage facts is
    # its own refusal, so a test that accepted any of them would let the other
    # two answer for this one.
    assert "disjoint chain" in message


def test_a_retry_claiming_another_ancestry_refuses_as_a_disjoint_chain(
    tmp_path: Path,
) -> None:
    """The same file name is not the same part if it came from elsewhere."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    retry = _confirmed_retry(config, job_id, first.batch_id)

    # Part two's request again, retiring part two's real retired attempt, and
    # claiming the bytes were split out of somewhere else.
    retired = extraction_batch.read_batch_execution_record(
        config, retry.batch_id
    ).discards
    _save_confirmed_batch(
        config,
        job_id,
        _copy_of(
            config,
            retry.batch_id,
            batch_id=STRANGER_ID,
            discards=retired,
            descriptor="a recipe nobody in this job published, part 4",
        ),
        decided_at="2024-05-09T00:00:00+00:00",
    )

    with pytest.raises(JankiError) as caught:
        study_job.job_effective_frontier(config, job_id)

    message = str(caught.value)
    assert "verbs-p2.png" in message
    assert STRANGER_ID in message
    assert "different document" in message


# --- each lineage fact alone, with its neighbours held equal -------------------
#
# The two cases above are refused with *some* other lineage fact also differing
# and a real retry already retiring the same attempt, so disabling the clause
# they name leaves a neighbouring guard to refuse the same record. These three
# hold every neighbour equal by construction and run no competing retry, so the
# one comparison each names is the only rule left that can decide the record —
# and disabling it makes the frontier *accept*, not refuse differently.


def _live_row(config: ProjectConfig, operation_id: str) -> Any:
    """The journal's own row for an attempt nothing has retired yet."""

    journal = operations.OperationJournal.load(config.operations_file)
    return journal.operations[operation_id]


def test_only_a_disjoint_source_refuses_with_ancestry_and_form_held_equal(
    tmp_path: Path,
) -> None:
    """Part one's request, sent to replace part two, and nothing else differs."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)

    # Part two's failed attempt is still live: no real retry has retired it, so
    # there is no fork and no successful child anywhere in this edge.
    live = _live_row(config, first.children[1].operation_id)
    part_two = extraction_batch.read_batch_execution_record(
        config, first.batch_id
    ).child(2)
    _save_confirmed_batch(
        config,
        job_id,
        _copy_of(
            config,
            first.batch_id,
            batch_id=STRANGER_ID,
            indices=[1],
            discards=[live],
            retry_of=first.batch_id,
            descriptor=part_two.lineage.descriptor,
        ),
        decided_at="2024-05-11T00:00:00+00:00",
    )

    with pytest.raises(JankiError) as caught:
        study_job.job_effective_frontier(config, job_id)

    assert "disjoint chain" in str(caught.value)
    # What this record holds, proved rather than asserted in words: the part
    # ancestry and the frozen form are equal, and only the part differs.
    sent = extraction_batch.read_batch_execution_record(config, STRANGER_ID).child(1)
    assert sent.lineage == part_two.lineage
    assert sent.expectation.table_layout == part_two.expectation.table_layout
    assert sent.source.name != part_two.source.name


def test_only_a_disjoint_ancestry_refuses_with_the_source_held_equal(
    tmp_path: Path,
) -> None:
    """Part two's own request again, split out of a document nobody published."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)

    live = _live_row(config, first.children[1].operation_id)
    part_two = extraction_batch.read_batch_execution_record(
        config, first.batch_id
    ).child(2)
    _save_confirmed_batch(
        config,
        job_id,
        _copy_of(
            config,
            first.batch_id,
            batch_id=STRANGER_ID,
            indices=[2],
            discards=[live],
            retry_of=first.batch_id,
            descriptor="a recipe nobody in this job published, part 4",
        ),
        decided_at="2024-05-12T00:00:00+00:00",
    )

    with pytest.raises(JankiError) as caught:
        study_job.job_effective_frontier(config, job_id)

    assert "different document" in str(caught.value)
    sent = extraction_batch.read_batch_execution_record(config, STRANGER_ID).child(2)
    assert sent.source.name == part_two.source.name
    assert sent.source_sha256 == part_two.source_sha256
    assert sent.expectation.table_layout == part_two.expectation.table_layout
    assert sent.lineage != part_two.lineage


def test_a_bound_frozen_layout_survives_a_confirmed_retry_and_keeps_the_sibling(
    tmp_path: Path,
) -> None:
    """§9.5's third lineage fact, on a job that really carries one.

    Both parts are bound to one owner-authored revision, so every child is sent
    under `table-layout` and the retry planner restores the same frozen block
    from the saved child's own provenance. The join is then allowed *because*
    the forms match — which is only a statement about this clause if the block
    is really there, so both ends are read back and compared.

    Both children are answered by the suite's one table fixture, because a
    bound call may not be answered with prose candidates. The two parts
    therefore propose the same identities, which the page discloses as
    conflicts exactly as it does for any other pair of children.
    """

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    layout = _bind_layout(config, job_id)
    first = _first_batch(config, job_id, cells=SUPPLIED_CELLS)
    retry = _confirmed_retry(
        config,
        job_id,
        first.batch_id,
        scenario=PART_ONE_SCENARIO,
        cells=SUPPLIED_CELLS,
    )

    frontier = study_job.job_effective_frontier(config, job_id)

    assert [
        (attempt.ref.source_name, attempt.ref.batch_id, attempt.state)
        for attempt in frontier.effective
    ] == [
        ("verbs-p1.png", first.batch_id, "committed"),
        ("verbs-p2.png", retry.batch_id, "committed"),
    ]
    # Not vacuous: every child in this history carries the owner's block.
    for batch_id, index in (
        (first.batch_id, 1),
        (first.batch_id, 2),
        (retry.batch_id, 1),
    ):
        bound = (
            extraction_batch.read_batch_execution_record(config, batch_id)
            .child(index)
            .expectation.table_layout
        )
        assert bound is not None
        assert bound.to_wire() == layout.to_wire()
    # The retry keeps its sibling: part one's settled child is untouched.
    assert len(frontier.superseded) == 1
    (retired,) = frontier.superseded
    assert (retired.ref.batch_id, retired.ref.index) == (first.batch_id, 2)
    assert retired.superseded_by == frontier.effective[1].ref

    rendered = study_job.render_job_preview(config, job_id)

    assert [ref.source_name for ref in rendered.children] == [
        "verbs-p1.png",
        "verbs-p2.png",
    ]
    # The bound labels and the printed cell really reach the page.
    answers = "".join(card.answer_html for card in rendered.preview.cards)
    assert "Plain" in answers
    assert "話す" in answers


def test_only_a_changed_frozen_layout_refuses_the_retry_as_new_work(
    tmp_path: Path,
) -> None:
    """The same file, the same ancestry, another frozen form: not a retry.

    Planned through the real `plan_extraction_batch` over part two's own
    published path and its own lineage, so the source name and hash are
    identical by construction and the ancestry is carried verbatim. No real
    retry has run, so the attempt this retires is still live and unsuperseded:
    §9.5's frozen-form comparison is the only rule left to decide it.
    """

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    _bind_layout(config, job_id)
    first = _first_batch(config, job_id, cells=SUPPLIED_CELLS)

    live = _live_row(config, first.children[1].operation_id)
    path_two, lineage_two = study_job.job_part_sources(config, job_id)[1]
    provider = _ScriptedProvider([_stream(PART_TWO_SCENARIO, cells=SUPPLIED_CELLS)])
    planned = extraction_batch.plan_extraction_batch(
        config,
        [path_two],
        mode=extract.LAYOUT_MODE,
        layouts=(_layout(layout_id="layout-other", revision=2),),
        lineage=(lineage_two,),
        destination_deck=config.deck_dir / "201-verbs.yaml",
        job_id=job_id,
        concurrency_limit=1,
        provider_env={},
        provider_runner=provider,
        provider_which=_which,
    )
    _save_confirmed_batch(
        config,
        job_id,
        replace(
            planned,
            batch_id=STRANGER_ID,
            retry_of=first.batch_id,
            discards=(live,),
        ),
        decided_at="2024-05-13T00:00:00+00:00",
    )

    with pytest.raises(JankiError) as caught:
        study_job.job_effective_frontier(config, job_id)

    message = str(caught.value)
    assert "verbs-p2.png" in message
    assert STRANGER_ID in message
    assert "different frozen source form" in message
    # Everything a neighbouring guard could have answered with, held equal.
    sent = extraction_batch.read_batch_execution_record(config, STRANGER_ID).child(1)
    old = extraction_batch.read_batch_execution_record(config, first.batch_id).child(2)
    assert sent.source.name == old.source.name
    assert sent.source_sha256 == old.source_sha256
    assert sent.lineage == old.lineage
    assert sent.expectation.table_layout is not None
    assert old.expectation.table_layout is not None
    assert (
        sent.expectation.table_layout.to_wire()
        != old.expectation.table_layout.to_wire()
    )


def test_two_retries_retiring_each_other_refuse_rather_than_choosing_one(
    tmp_path: Path,
) -> None:
    """A cycle is not resolved by file order, receipt order or intent order.

    Two confirmed records, each claiming to retire the other's child. Every
    other rule passes — same part, same ancestry, same frozen layout, neither
    retiring a success — so what is left is a history with no oldest attempt,
    and picking one end of it would be janki inventing the answer.
    """

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    retry = _confirmed_retry(config, job_id, first.batch_id)

    journal = operations.OperationJournal.load(config.operations_file)
    live = journal.operations[retry.children[0].operation_id]
    one_child = f"{CYCLE_ONE_ID[:8]}-child-1"
    two_child = f"{CYCLE_TWO_ID[:8]}-child-1"
    for batch_id, retires, retry_of in (
        (CYCLE_ONE_ID, two_child, CYCLE_TWO_ID),
        (CYCLE_TWO_ID, one_child, CYCLE_ONE_ID),
    ):
        _save_confirmed_batch(
            config,
            job_id,
            _copy_of(
                config,
                retry.batch_id,
                batch_id=batch_id,
                discards=[
                    _snapshot(
                        config,
                        live,
                        operation_id=retires,
                        state="failed_before_send",
                        batch_id=retry_of,
                    )
                ],
                retry_of=retry_of,
            ),
            decided_at="2024-05-08T00:00:00+00:00",
        )

    with pytest.raises(JankiError) as caught:
        study_job.job_effective_frontier(config, job_id)

    message = str(caught.value)
    assert CYCLE_ONE_ID in message
    assert CYCLE_TWO_ID in message


def test_an_unknown_outcome_stays_the_effective_child_and_blocks(
    tmp_path: Path,
) -> None:
    """Nothing counts an outcome nobody knows as progress or as retired."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id, second=UNCAPTURED)
    journal = operations.OperationJournal.load(config.operations_file)
    assert journal.operations[first.children[1].operation_id].state == (
        "outcome_unknown"
    )

    frontier = study_job.job_effective_frontier(config, job_id)

    assert frontier.effective[1].state == "outcome_unknown"
    assert frontier.effective[1].ref.batch_id == first.batch_id
    assert frontier.superseded == ()
    assert not frontier.effective[1].settled


# --- one page, every settled effective child ----------------------------------


def test_the_job_page_draws_both_batches_and_keeps_each_source_reference(
    tmp_path: Path,
) -> None:
    """One real render of the whole effective frontier, across two batches."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    retry = _confirmed_retry(config, job_id, first.batch_id)
    before = _digest_tree(config)

    rendered = study_job.render_job_preview(config, job_id)

    assert [
        (ref.batch_id, ref.index, ref.source_name) for ref in rendered.children
    ] == [
        (first.batch_id, 1, "verbs-p1.png"),
        (retry.batch_id, 1, "verbs-p2.png"),
    ]
    assert rendered.pending == ()
    drawn = {card.record_id for card in rendered.preview.cards}
    for expression in (*_expressions(PART_ONE_SCENARIO), *_expressions(PART_TWO_SCENARIO)):
        assert any(expression in record_id for record_id in drawn), (
            expression,
            sorted(drawn),
        )
        assert expression.encode("utf-8") in rendered.preview.html
    assert rendered.rendering_fingerprint == rendered.preview.sha256
    # A presentation: it accepts nothing and writes nothing.
    assert _digest_tree(config) == before


def test_a_pending_job_draws_its_settled_subset_and_names_what_is_missing(
    tmp_path: Path,
) -> None:
    """Before the retry lands, looking at the settled half must still work.

    What is missing is named *off* the drawn document: in the typed references
    and in the derived sentences a surface prints, never in the page whose
    sha256 is this job's rendering fingerprint.
    """

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)

    rendered = study_job.render_job_preview(config, job_id)

    assert [ref.source_name for ref in rendered.children] == ["verbs-p1.png"]
    assert [ref.source_name for ref in rendered.pending] == ["verbs-p2.png"]
    assert [ref.batch_id for ref in rendered.pending] == [first.batch_id]
    assert b"verbs-p2.png" not in rendered.preview.html
    assert [line for line in rendered.disclosure if "verbs-p2.png" in line]
    assert first.batch_id in rendered.disclosure[0]
    assert "has not settled yet" in rendered.disclosure[0]
    for expression in _expressions(PART_ONE_SCENARIO):
        assert expression.encode("utf-8") in rendered.preview.html


def test_one_more_pending_part_changes_no_drawn_card_and_no_fingerprint(
    tmp_path: Path,
) -> None:
    """§9.5's fingerprint is the drawn cards, not how far the job has got.

    A wholly public journey: the owner sends part one, which settles, and then
    sends part two, which does not. Not one card on the page changes between
    the two looks, so the document — and the fingerprint an owner's saved
    review decisions are bound to — must be byte-identical. Hashing "which
    parts are missing" into it would tell the owner to decide again about cards
    nobody touched.
    """

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    _select(config, job_id, "verbs-p1.png")
    _selected_batch(config, job_id, _stream(PART_ONE_SCENARIO))

    before = study_job.render_job_preview(config, job_id)
    assert [ref.source_name for ref in before.children] == ["verbs-p1.png"]
    assert before.pending == ()

    _select(config, job_id, "verbs-p2.png")
    _second, outcome = _selected_batch(config, job_id, UNUSABLE_ANSWER)
    assert outcome.committed_count == 0

    after = study_job.render_job_preview(config, job_id)

    assert [ref.source_name for ref in after.children] == ["verbs-p1.png"]
    assert [ref.source_name for ref in after.pending] == ["verbs-p2.png"]
    assert after.preview.html == before.preview.html
    assert after.rendering_fingerprint == before.rendering_fingerprint
    assert after.rendering_fingerprint == after.preview.sha256
    assert b"verbs-p2.png" not in after.preview.html
    # The newly pending part is disclosed, off the page.
    assert before.disclosure == ()
    assert [line for line in after.disclosure if "verbs-p2.png" in line]


def test_a_part_that_lands_does_change_the_rendering_fingerprint(
    tmp_path: Path,
) -> None:
    """The control for the case above: drawn content still goes stale."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    before = study_job.render_job_preview(config, job_id)

    _confirmed_retry(config, job_id, first.batch_id)
    after = study_job.render_job_preview(config, job_id)

    assert after.pending == ()
    assert after.disclosure == ()
    assert [ref.source_name for ref in after.children] == [
        "verbs-p1.png",
        "verbs-p2.png",
    ]
    assert after.preview.html != before.preview.html
    assert after.rendering_fingerprint != before.rendering_fingerprint
    for expression in _expressions(PART_TWO_SCENARIO):
        assert expression.encode("utf-8") in after.preview.html


def test_an_unreserved_batch_nobody_can_match_is_disclosed_and_changes_nothing(
    tmp_path: Path,
) -> None:
    """An intent whose artifact is gone and which reserved nothing is a report.

    Nothing ran under it — the journal holds no reservation — so there is
    nothing to refuse and nothing to hide either. It is read-only disclosed
    state beside the pending part, and it moves no card and no fingerprint.
    """

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    before = study_job.render_job_preview(config, job_id)

    _unmatched_batch_intent(config, job_id, retry_of=first.batch_id)

    frontier = study_job.job_effective_frontier(config, job_id)
    assert [entry.split(":")[0] for entry in frontier.unresolved] == [STRANGER_ID]
    # The old attempt is still part two's effective child: nothing was retired.
    assert frontier.effective[1].ref.batch_id == first.batch_id
    assert frontier.superseded == ()

    after = study_job.render_job_preview(config, job_id)

    assert after.unresolved == frontier.unresolved
    assert [line for line in after.disclosure if STRANGER_ID in line]
    assert STRANGER_ID.encode("ascii") not in after.preview.html
    assert after.preview.html == before.preview.html
    assert after.rendering_fingerprint == before.rendering_fingerprint


def test_a_committed_child_whose_staging_vanished_refuses_by_part(
    tmp_path: Path,
) -> None:
    """A settled part that cannot be drawn is a refusal, never an omission."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    retry = _confirmed_retry(config, job_id, first.batch_id)
    record = extraction_batch.read_batch_execution_record(config, retry.batch_id)
    record.children[0].staging_path.unlink()

    with pytest.raises(JankiError) as caught:
        study_job.render_job_preview(config, job_id)

    assert "verbs-p2.png" in str(caught.value)


def test_two_effective_children_sharing_an_identity_disclose_both_batches(
    tmp_path: Path,
) -> None:
    """One representative card, both sources named, and no reading chosen."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    retry = _confirmed_retry(config, job_id, first.batch_id)

    # Part two now proposes an identity part one already proposed, differing
    # on one field. Its own metadata is kept exactly, so the document is still
    # provably that child's own answer.
    record = extraction_batch.read_batch_execution_record(config, retry.batch_id)
    staged = record.children[0].staging_path
    records, meta = staging.read_staging_text(
        staged.read_text(encoding="utf-8"), source=str(staged)
    )
    first_record = extraction_batch.read_batch_execution_record(
        config, first.batch_id
    ).children[0]
    theirs, _meta = staging.read_staging_text(
        first_record.staging_path.read_text(encoding="utf-8"),
        source=str(first_record.staging_path),
    )
    staging.write_staging(
        staged,
        [replace(theirs[0], meanings=["PART-TWO-GLOSS"]), *records],
        meta,
        force=True,
    )

    rendered = study_job.render_job_preview(config, job_id)

    disclosed = [line for line in rendered.conflicts if theirs[0].id in line]
    assert disclosed, rendered.conflicts
    detail = disclosed[0]
    assert "meanings" in detail
    assert "PART-TWO-GLOSS" in detail
    assert "verbs-p1.png" in detail and "verbs-p2.png" in detail
    # Equal child indices from two batches, told apart in the sentence itself.
    assert first.batch_id[:8] in detail and retry.batch_id[:8] in detail
    assert b"proposed twice" in rendered.preview.html


# --- the Assistant's own words ------------------------------------------------


def test_the_assistant_preview_names_the_source_parts_not_bare_indices(
    tmp_path: Path,
) -> None:
    """"sources 1, 1" says nothing; the parts a person published say it all."""

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    _confirmed_retry(config, job_id, first.batch_id)

    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config, deck_choices=(), _targets=()
    )
    offer = adapter.render_study_job_preview(job_id=job_id, deck_scope="")

    assert "verbs-p1.png" in offer.message
    assert "verbs-p2.png" in offer.message
    assert "sources 1, 1" not in offer.message


def test_the_assistant_preview_says_what_is_not_on_the_page_it_offers(
    tmp_path: Path,
) -> None:
    """The disclosure has to reach the person, since it is off the document.

    Both kinds: the part this job is still waiting on, and the batch intent
    whose own artifact nobody could match. Neither is on the page, so a surface
    that did not say them would offer a subset as the whole job.
    """

    config = _project(tmp_path)
    job_id = _job_with_two_published_parts(config)
    first = _first_batch(config, job_id)
    _unmatched_batch_intent(config, job_id, retry_of=first.batch_id)

    adapter = assistant_adapter.RevisionAssistantAdapter(
        config=config, deck_choices=(), _targets=()
    )
    offer = adapter.render_study_job_preview(job_id=job_id, deck_scope="")

    assert "verbs-p1.png" in offer.message
    assert "verbs-p2.png" in offer.message
    assert "has not settled yet" in offer.message
    assert STRANGER_ID in offer.message
    assert first.batch_id in offer.message


# --- structural helpers over the real saved receipts --------------------------


#: Batch identities for the corrupted-lineage records these tests save. Fixed
#: rather than minted, so a refusal message can be matched by name.
FORK_ID = "9c1d2e3f-4a5b-4c6d-8e9f-0a1b2c3d4e5f"
STRANGER_ID = "5b6c7d8e-9f0a-4b1c-8d2e-3f4a5b6c7d8e"
CYCLE_ONE_ID = "11112222-3333-4444-8555-666677778888"
CYCLE_TWO_ID = "99990000-1111-4222-8333-444455556666"


def _unmatched_batch_intent(
    config: ProjectConfig,
    job_id: str,
    *,
    batch_id: str = STRANGER_ID,
    retry_of: str = "",
) -> None:
    """A recorded batch intent whose manifest is not there, reserving nothing.

    Exactly what a plan that was recorded and whose manifest never landed —
    or was later removed — leaves in a job: the journal holds no reservation
    under this id, so nothing ran, nothing was retired and nothing paid.
    """

    study_job.append_intent(
        config,
        job_id,
        study_job.ActionIntent(
            intent_id=study_job.new_intent_id(),
            kind="retry",
            decided_at="2024-05-14T00:00:00+00:00",
            reserves={"batch_id": batch_id, "manifest_sha256": "b" * 64},
            bindings={"job_id": job_id, "retry_of": retry_of},
        ),
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )


def _snapshot(
    config: ProjectConfig, held: Any, *, operation_id: str = "", **changes: Any
) -> Any:
    """One journal row as a *saved* snapshot, with named fields moved.

    Built from the row the journal really holds and read back through the
    journal's own reader, so a state that cannot carry an artifact receipt —
    or any other combination the store refuses — cannot be smuggled into one
    of these receipts. That keeps the corruption these tests save limited to
    the lineage claim under test.
    """

    raw = dict(held.to_dict(), **changes)
    if raw.get("state") not in ("result_captured", "committed"):
        raw.pop("artifact", None)
    if operation_id and operation_id != held.operation_id:
        # A spool receipt names the operation's own pending file, so it cannot
        # be carried onto another identity — and the journal says so.
        raw.pop("response_spool", None)
    return operations.Operation.from_dict(
        config.operations_file, operation_id or held.operation_id, raw
    )


def _save_confirmed_batch(
    config: ProjectConfig,
    job_id: str,
    plan: Any,
    *,
    decided_at: str,
) -> Any:
    """Save one batch's real manifest and confirmed-execution receipt.

    Exactly the pair `dispatch_extraction_batch` writes, in its order, and
    exactly the state a machine that died between `_write_execution_receipt`
    and `_reserve_batch` leaves behind: somebody confirmed this execution, and
    the journal never got its reservation. Written through the owning module's
    own receipt serializer, so these are real saved receipts rather than a
    hand-typed shape — and the job records the intent, as its own dispatcher
    does, so nothing here is found by scanning a directory.
    """

    extraction_batch.batch_manifest_path(config, plan.batch_id).write_bytes(
        plan.manifest_bytes
    )
    extraction_batch.execution_receipt_path(config, plan.batch_id).write_bytes(
        extraction_batch._execution_receipt_bytes(plan)
    )
    study_job.append_intent(
        config,
        job_id,
        study_job.ActionIntent(
            intent_id=study_job.new_intent_id(),
            kind="retry",
            decided_at=decided_at,
            reserves={
                "batch_id": plan.batch_id,
                "manifest_sha256": plan.manifest_sha256,
                "child_operation_ids": [
                    child.operation_id for child in plan.children
                ],
            },
            bindings={"job_id": job_id, "retry_of": plan.retry_of},
        ),
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )
    return plan


def _copy_of(
    config: ProjectConfig,
    source_batch_id: str,
    *,
    batch_id: str,
    discards: Sequence[Any],
    retry_of: str | None = None,
    indices: Sequence[int] = (),
    descriptor: str = "",
) -> Any:
    """Another confirmed batch sending what that one sent, retiring these.

    The children are the real batch's own children — same sources, same frozen
    layouts, same part ancestry — under fresh operation ids, because two
    confirmed records can never be two claims on one authority. ``indices``
    narrows them to the parts this record claims to send, and ``descriptor``
    re-describes where those parts were split out of.
    """

    plan = extraction_batch._load_batch_plan(config, source_batch_id)
    children = tuple(
        replace(
            child,
            operation_id=f"{batch_id[:8]}-child-{child.index}",
            lineage=(
                child.lineage
                if not descriptor
                else replace(child.lineage, descriptor=descriptor)
            ),
        )
        for child in plan.children
        if not indices or child.index in indices
    )
    return replace(
        plan,
        batch_id=batch_id,
        children=children,
        discards=tuple(discards),
        retry_of=plan.retry_of if retry_of is None else retry_of,
    )
