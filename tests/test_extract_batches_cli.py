"""``janki extract`` over several sources, and the local ``extract-batch`` desk.

Every batch call here is a fake stand-in for
``japanese_anki.application.extraction_batch``: these tests are about what the
command line asks, shows, and refuses — not about how the core dispatches.
The fake is injected as the real module name so the surface's own import is
what gets exercised.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import cli

CORE_MODULE = "japanese_anki.application.extraction_batch"

BATCH_ID = "6f1c2b7a-0d3e-4a55-9c71-2f8e1b4d5a60"


@dataclass(frozen=True)
class FakeChild:
    index: int
    source: Path
    operation_id: str
    source_sha256: str = "a" * 64
    request_fingerprint: str = "b" * 64
    staging_path: Path = Path("data/staging/child.yaml")
    lineage: tuple[str, ...] = ()


@dataclass(frozen=True)
class FakeDiscard:
    """The shape ``plan.discards`` exposes: an ``operations.Operation`` snapshot."""

    operation_id: str
    source_file: str
    state: str
    request_fp: str
    model: str
    detail: str


@dataclass(frozen=True)
class FakePlan:
    batch_id: str = BATCH_ID
    children: tuple[FakeChild, ...] = ()
    concurrency_limit: int = 2
    provider: str = "anthropic-api"
    model: str = "claude-opus-5"
    scope_id: str = ""
    destination_deck: Path | None = None
    retry_of: str = ""
    manifest_bytes: bytes = b"{}"
    manifest_sha256: str = "c" * 64
    fingerprint: str = "d" * 64
    discards: tuple[FakeDiscard, ...] = ()


@dataclass(frozen=True)
class FakeChildOutcome:
    index: int
    operation_id: str
    source: str
    state: str
    staging_path: Path | None = None
    error: str = ""
    bookkeeping_complete: bool = True
    records: int = 0


@dataclass(frozen=True)
class FakeOutcome:
    batch_id: str = BATCH_ID
    children: tuple[FakeChildOutcome, ...] = ()
    concurrency_limit: int = 2
    committed_count: int = 0
    failed_count: int = 0
    unknown_count: int = 0
    pending_count: int = 0


@dataclass(frozen=True)
class FakeCardPreview:
    html: bytes = b"<html><body>combined</body></html>"
    sha256: str = "e" * 64
    card_count: int = 7
    content_security_policy: str = "default-src 'none'"


@dataclass(frozen=True)
class FakePreview:
    preview: FakeCardPreview = field(default_factory=FakeCardPreview)
    conflicts: tuple[str, ...] = ()
    child_indices: tuple[int, ...] = ()


@dataclass
class FakeCore:
    """Records exactly which core entry points the command line reached."""

    planned: list[tuple[tuple[Path, ...], dict[str, Any]]] = field(default_factory=list)
    dispatched: list[FakePlan] = field(default_factory=list)
    resumed: list[str] = field(default_factory=list)
    retry_planned: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    listed: int = 0
    previewed: list[str] = field(default_factory=list)
    plan: FakePlan | None = None
    retry_plan: FakePlan | None = None
    outcome: FakeOutcome | None = None
    status_outcome: FakeOutcome | None = None
    preview: FakePreview | None = None
    progress_events: tuple[tuple[int, str, str, str], ...] = ()

    def as_module(self) -> types.ModuleType:
        module = types.ModuleType(CORE_MODULE)
        module.plan_extraction_batch = self.plan_extraction_batch
        module.dispatch_extraction_batch = self.dispatch_extraction_batch
        module.resume_extraction_batch = self.resume_extraction_batch
        module.plan_extraction_batch_retry = self.plan_extraction_batch_retry
        module.extraction_batch_status = self.extraction_batch_status
        module.list_extraction_batches = self.list_extraction_batches
        module.render_extraction_batch_preview = self.render_extraction_batch_preview
        module.ExtractionBatchProgress = _Progress
        return module

    def plan_extraction_batch(
        self, config: Any, sources: Sequence[Path], **options: Any
    ) -> FakePlan:
        del config
        self.planned.append((tuple(sources), dict(options)))
        assert self.plan is not None, "the test did not stage an initial plan"
        return self.plan

    def dispatch_extraction_batch(
        self, config: Any, plan: FakePlan, *, progress: Any = None, **_: Any
    ) -> FakeOutcome:
        del config
        self.dispatched.append(plan)
        if progress is not None:
            for index, operation_id, state, message in self.progress_events:
                progress(_Progress(index, operation_id, state, message))
        assert self.outcome is not None, "the test did not stage an outcome"
        return self.outcome

    def resume_extraction_batch(
        self, config: Any, batch_id: str, *, progress: Any = None, **_: Any
    ) -> FakeOutcome:
        del config, progress
        self.resumed.append(batch_id)
        assert self.outcome is not None, "the test did not stage an outcome"
        return self.outcome

    def plan_extraction_batch_retry(
        self,
        config: Any,
        batch_id: str,
        child_indices: Sequence[int],
        **options: Any,
    ) -> FakePlan:
        del config, options
        self.retry_planned.append((batch_id, tuple(child_indices)))
        assert self.retry_plan is not None, "the test did not stage a retry plan"
        return self.retry_plan

    def extraction_batch_status(self, config: Any, batch_id: str) -> FakeOutcome:
        del config
        self.statuses.append(batch_id)
        assert self.status_outcome is not None, "the test did not stage a status"
        return self.status_outcome

    def list_extraction_batches(self, config: Any) -> tuple[FakeOutcome, ...]:
        del config
        self.listed += 1
        return () if self.status_outcome is None else (self.status_outcome,)

    def render_extraction_batch_preview(
        self, config: Any, batch_id: str, *, deck_path: Path | None = None
    ) -> FakePreview:
        del config, deck_path
        self.previewed.append(batch_id)
        assert self.preview is not None, "the test did not stage a preview"
        return self.preview


@dataclass(frozen=True)
class _Progress:
    index: int
    operation_id: str
    state: str
    message: str


@contextmanager
def installed(core: FakeCore) -> Iterator[FakeCore]:
    previous = sys.modules.get(CORE_MODULE)
    sys.modules[CORE_MODULE] = core.as_module()
    try:
        yield core
    finally:
        if previous is None:
            sys.modules.pop(CORE_MODULE, None)
        else:
            sys.modules[CORE_MODULE] = previous


def project(tmp_path: Path, *names: str) -> tuple[Path, list[Path]]:
    root = tmp_path / "project"
    root.mkdir()
    (root / "janki.toml").write_text("", encoding="utf-8")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    sources = []
    for name in names:
        path = incoming / name
        path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n" + name.encode("utf-8"))
        sources.append(path)
    return root, sources


def run(root: Path, *args: str) -> int:
    return cli.main(["--root", str(root), *args])


def two_child_plan(sources: Sequence[Path], **overrides: Any) -> FakePlan:
    children = tuple(
        FakeChild(index=index, source=path, operation_id=f"op-{index}")
        for index, path in enumerate(sources, start=1)
    )
    return FakePlan(children=children, **overrides)


def test_two_sources_reach_one_shared_batch_plan_and_send_nothing_unattended(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--yes`` an unattended run refuses after planning, before sending."""

    root, sources = project(tmp_path, "lesson-a.pdf", "lesson-b.pdf")
    core = FakeCore()
    core.plan = two_child_plan(sources)
    with installed(core):
        code = run(root, "extract", *[str(path) for path in sources])

    assert code == 1
    assert len(core.planned) == 1, "a multi-source run plans exactly one batch"
    assert core.dispatched == [], "nothing may be sent before an owner says yes"
    assert "Nothing was sent." in capsys.readouterr().out


def test_one_confirmation_names_every_child_the_provider_model_and_concurrency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, sources = project(tmp_path, "lesson-a.pdf", "lesson-b.pdf", "lesson-c.pdf")
    core = FakeCore()
    core.plan = two_child_plan(sources, concurrency_limit=3)
    core.outcome = FakeOutcome(
        children=(
            FakeChildOutcome(1, "op-1", "lesson-a.pdf", "committed", records=4),
            FakeChildOutcome(2, "op-2", "lesson-b.pdf", "committed", records=6),
            FakeChildOutcome(3, "op-3", "lesson-c.pdf", "failed_before_send"),
        ),
        concurrency_limit=3,
        committed_count=2,
        failed_count=1,
    )
    # Plain core-style messages: the numbering is the command's to render.
    core.progress_events = (
        (2, "op-2", "running", "reading lesson-b.pdf"),
        (2, "op-2", "committed", "saved 6 proposals"),
    )
    with installed(core):
        code = run(
            root,
            "extract",
            *[str(path) for path in sources],
            "--concurrency",
            "3",
            "--yes",
        )

    out = capsys.readouterr().out
    # One source failed, so the command says so with its exit status too.
    assert code == 1
    assert core.planned[0][1]["concurrency_limit"] == 3
    for position, name in enumerate(
        ("lesson-a.pdf", "lesson-b.pdf", "lesson-c.pdf"), start=1
    ):
        assert f"{position}. {name}" in out, out
    assert "claude-opus-5" in out
    assert "anthropic-api" in out
    assert "3 at a time" in out
    assert "Source 2 of 3: running" in out, out
    assert "reading lesson-b.pdf" in out, out
    assert len(core.dispatched) == 1


def test_one_source_keeps_the_existing_unbatched_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single file must not acquire a batch identity or a batch confirmation."""

    root, sources = project(tmp_path, "lesson-a.pdf")
    sentinel = RuntimeError("single-source planning reached")

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise sentinel

    monkeypatch.setattr(cli.claude_client, "read_style_guide", lambda _root: "guide")
    monkeypatch.setattr(cli.prompts, "load", lambda _root, _name: "system")
    monkeypatch.setattr(cli, "plan_extraction", refuse)
    core = FakeCore()
    with installed(core), pytest.raises(RuntimeError) as caught:
        run(root, "extract", str(sources[0]))

    assert caught.value is sentinel
    assert core.planned == []


@pytest.mark.parametrize("requested", ["0", "5"])
def test_concurrency_outside_one_through_four_is_refused_before_planning(
    tmp_path: Path, requested: str
) -> None:
    root, sources = project(tmp_path, "lesson-a.pdf", "lesson-b.pdf")
    core = FakeCore()
    with installed(core), pytest.raises(SystemExit):
        run(
            root,
            "extract",
            *[str(path) for path in sources],
            "--concurrency",
            requested,
            "--yes",
        )

    assert core.planned == []


def test_status_reads_children_without_planning_or_sending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    core = FakeCore()
    core.status_outcome = FakeOutcome(
        children=(
            FakeChildOutcome(1, "op-1", "lesson-a.pdf", "committed", records=4),
            FakeChildOutcome(2, "op-2", "lesson-b.pdf", "failed_before_send",
                            error="the provider refused"),
        ),
        committed_count=1,
        failed_count=1,
    )
    with installed(core):
        code = run(root, "extract-batch", "status", BATCH_ID)

    out = capsys.readouterr().out
    assert code == 0
    assert core.statuses == [BATCH_ID]
    assert core.planned == [] and core.dispatched == [] and core.retry_planned == []
    assert "1. lesson-a.pdf" in out
    assert "committed" in out
    assert "2. lesson-b.pdf" in out
    assert "failed_before_send" in out
    assert "the provider refused" in out


def test_status_without_an_identifier_lists_the_durable_batches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    core = FakeCore()
    core.status_outcome = FakeOutcome(
        children=(FakeChildOutcome(1, "op-1", "lesson-a.pdf", "committed"),),
        committed_count=1,
    )
    with installed(core):
        code = run(root, "extract-batch", "status")

    assert code == 0
    assert core.listed == 1
    assert core.statuses == []
    assert BATCH_ID in capsys.readouterr().out


def test_resume_continues_recorded_authority_without_replanning_or_asking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    core = FakeCore()
    core.outcome = FakeOutcome(
        children=(FakeChildOutcome(1, "op-1", "lesson-a.pdf", "committed", records=4),),
        committed_count=1,
    )
    core.status_outcome = core.outcome
    with installed(core):
        code = run(root, "extract-batch", "resume", BATCH_ID)

    out = capsys.readouterr().out
    assert code == 0
    assert core.resumed == [BATCH_ID]
    assert core.planned == [], "resume must never build a fresh plan"
    assert core.dispatched == [], "resume goes through the core's own resume path"
    assert "already authorized" in out.casefold()


def test_retry_needs_exact_children_and_never_takes_a_blind_flag(
    tmp_path: Path,
) -> None:
    root, _sources = project(tmp_path)
    core = FakeCore()
    with installed(core), pytest.raises(SystemExit):
        run(root, "extract-batch", "retry", BATCH_ID)

    assert core.retry_planned == []


def test_retry_confirmation_names_the_exact_discarded_evidence_and_new_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    core = FakeCore()
    core.retry_plan = FakePlan(
        children=(
            FakeChild(index=2, source=Path("data/inbox/lesson-b.pdf"), operation_id="op-9"),
        ),
        retry_of=BATCH_ID,
        discards=(
            FakeDiscard(
                operation_id="op-2",
                source_file="lesson-b.pdf",
                state="failed_before_send",
                request_fp="f" * 64,
                model="claude-opus-5",
                detail="the provider refused",
            ),
        ),
    )
    with installed(core):
        refused = run(root, "extract-batch", "retry", BATCH_ID, "--children", "2")

    refusal = capsys.readouterr().out
    assert refused == 1
    assert core.retry_planned == [(BATCH_ID, (2,))]
    assert core.dispatched == [], "a retry plan alone sends nothing"
    assert "op-2" in refusal
    assert "lesson-b.pdf" in refusal
    assert "failed_before_send" in refusal
    assert "the provider refused" in refusal
    assert "Nothing was sent." in refusal

    core.outcome = FakeOutcome(
        children=(FakeChildOutcome(2, "op-9", "lesson-b.pdf", "committed", records=5),),
        committed_count=1,
    )
    with installed(core):
        accepted = run(
            root, "extract-batch", "retry", BATCH_ID, "--children", "2", "--yes"
        )

    assert accepted == 0
    assert len(core.dispatched) == 1
    assert core.dispatched[0].retry_of == BATCH_ID


def test_preview_writes_the_core_rendered_card_html_and_names_conflicts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    output = tmp_path / "combined.html"
    core = FakeCore()
    core.preview = FakePreview(
        conflicts=("食べる appears in sources 1 and 2 with different readings",),
        child_indices=(1, 2),
    )
    with installed(core):
        code = run(
            root, "extract-batch", "preview", BATCH_ID, "--output", str(output)
        )

    out = capsys.readouterr().out
    assert code == 0
    assert core.previewed == [BATCH_ID]
    assert output.read_bytes() == b"<html><body>combined</body></html>"
    assert "7" in out
    assert "食べる appears in sources 1 and 2" in out
    assert core.dispatched == [] and core.planned == []


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("outcome_unknown", "its outcome and its cost are unknown"),
        ("result_captured", "already paid for and cannot be recovered"),
    ],
)
def test_retry_spells_out_a_captured_or_unknown_consequence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], state: str, expected: str
) -> None:
    root, _sources = project(tmp_path)
    core = FakeCore()
    core.retry_plan = FakePlan(
        children=(
            FakeChild(index=2, source=Path("data/inbox/lesson-b.pdf"), operation_id="op-9"),
        ),
        retry_of=BATCH_ID,
        discards=(
            FakeDiscard(
                operation_id="op-2",
                source_file="lesson-b.pdf",
                state=state,
                request_fp="f" * 64,
                model="claude-opus-5",
                detail="",
            ),
        ),
    )
    with installed(core):
        assert run(root, "extract-batch", "retry", BATCH_ID, "--children", "2") == 1

    out = capsys.readouterr().out
    assert state in out
    assert expected in out
    assert core.dispatched == []
