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
from japanese_anki.application import capture_recovery
from japanese_anki.errors import JankiError

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


# --- capture recovery, on the same command --------------------------------------
#
# The service is faked here exactly as the batch core is: what these prove is
# what the command line asks, prints and refuses. In particular that `--yes`
# answers the recovery question and never supplies a proposal, because choosing
# between two paid proposals is the owner's decision and not a flag's.

RECOVERY_MODULE = "japanese_anki.application.capture_recovery"
OPERATION = "0f7a4d31-6c5e-4a2b-9d18-7c3b5e2a4f60"


@dataclass(frozen=True)
class FakeLocation:
    frame_index: int
    block_index: int
    tool_use_id: str
    json_pointer: str
    envelope_shape: str


@dataclass(frozen=True)
class FakeGroup:
    proposal_sha256: str
    schema_verdict: str
    locations: tuple[FakeLocation, ...]


@dataclass(frozen=True)
class FakeSelection:
    proposal_sha256: str
    json_pointer: str


@dataclass(frozen=True)
class FakeCapturePlan:
    operation_id: str = OPERATION
    operation_state: str = "result_captured"
    request_fingerprint: str = "a" * 64
    capture_sha256: str = "b" * 64
    response_schema_fingerprint: str = "c" * 64
    source_sha256: str = "d" * 64
    staging_path: Path = Path("data/staging/lesson-a.pdf.yaml")
    patterns_path: Path = Path("data/patterns.json")
    destination_deck_sha256: str = ""
    groups: tuple[FakeGroup, ...] = ()
    valid_group_count: int = 0

    def select(
        self, proposal_sha256: str, *, json_pointer: str | None = None
    ) -> FakeSelection:
        """Exactly the dataclass's own semantics, including ``None``.

        ``json_pointer or …`` would accept ``""`` where the real ``select``
        refuses it, and a fake that normalizes that difference away reports
        green for a service contract it does not implement.
        """
        group = next(
            item for item in self.groups if item.proposal_sha256 == proposal_sha256
        )
        if json_pointer is None:
            return FakeSelection(proposal_sha256, group.locations[0].json_pointer)
        found = [
            item for item in group.locations if item.json_pointer == json_pointer
        ]
        if not found:
            raise JankiError(
                f"Proposal {proposal_sha256} is not at {json_pointer!r}."
            )
        return FakeSelection(proposal_sha256, found[0].json_pointer)


@dataclass(frozen=True)
class FakeExtractionOutcome:
    target: Path = Path("data/staging/lesson-a.pdf.yaml")
    records: int = 3
    already_known: int = 0
    unusable: int = 0
    duplicates: int = 0
    coverage_status: str = "unmeasured"
    kept_reviewed_patterns: bool = False
    source: str = "lesson-a.pdf"


@dataclass
class FakeRecovery:
    """Records exactly which recovery entry points the command line reached."""

    inspected: list[str] = field(default_factory=list)
    staged: list[tuple[str, Any]] = field(default_factory=list)
    rendered: list[tuple[str, Path]] = field(default_factory=list)
    #: A ``FakeCapturePlan`` or, where the selection semantics are what is
    #: under test, a real ``CaptureRecoveryPlan``.
    plan: Any = field(default_factory=FakeCapturePlan)
    outcome: FakeExtractionOutcome = field(default_factory=FakeExtractionOutcome)
    ambiguous: bool = False

    def as_module(self) -> types.ModuleType:
        module = types.ModuleType(RECOVERY_MODULE)
        module.inspect_capture_proposals = self.inspect_capture_proposals
        module.stage_capture_proposal = self.stage_capture_proposal
        module.render_capture_proposals = self.render_capture_proposals
        return module

    def inspect_capture_proposals(self, config: Any, operation_id: str) -> Any:
        del config
        self.inspected.append(operation_id)
        return self.plan

    def stage_capture_proposal(
        self, config: Any, operation_id: str, selection: Any = None
    ) -> FakeExtractionOutcome:
        del config
        self.staged.append((operation_id, selection))
        if self.ambiguous and selection is None:
            raise JankiError(
                "this capture holds 2 readable proposals, so which one is the "
                "answer is the owner's decision"
            )
        return self.outcome

    def render_capture_proposals(
        self, config: Any, operation_id: str, output_path: Path
    ) -> Path:
        del config
        self.rendered.append((operation_id, Path(output_path)))
        Path(output_path).write_text("<html>proposals</html>", encoding="utf-8")
        return Path(output_path)


@contextmanager
def installed_recovery(recovery: FakeRecovery) -> Iterator[FakeRecovery]:
    previous = sys.modules.get(RECOVERY_MODULE)
    sys.modules[RECOVERY_MODULE] = recovery.as_module()
    try:
        yield recovery
    finally:
        if previous is None:
            sys.modules.pop(RECOVERY_MODULE, None)
        else:
            sys.modules[RECOVERY_MODULE] = previous


def _readable_plan(**overrides: Any) -> FakeCapturePlan:
    return FakeCapturePlan(
        groups=(
            FakeGroup(
                proposal_sha256="e" * 64,
                schema_verdict="valid",
                locations=(
                    FakeLocation(7, 0, "toolu_01", "/message/content/0/input", "tool_use_input"),
                ),
            ),
        ),
        valid_group_count=1,
        **overrides,
    )


def _real_plan() -> capture_recovery.CaptureRecoveryPlan:
    """The service's own plan, so ``select`` is the real method the CLI calls.

    ``--proposal`` is a hash the owner types, and what the command line does
    with an omitted ``--at`` is a question only the real dataclass answers.
    """
    return capture_recovery.CaptureRecoveryPlan(
        operation_id=OPERATION,
        operation_state="result_captured",
        request_fingerprint="a" * 64,
        capture_sha256="b" * 64,
        response_schema_fingerprint="c" * 64,
        source_sha256="d" * 64,
        staging_path=Path("data/staging/lesson-a.pdf.yaml"),
        patterns_path=Path("data/patterns.json"),
        destination_deck_sha256="",
        groups=(
            capture_recovery.CaptureProposalGroup(
                proposal_sha256="e" * 64,
                schema_verdict="valid",
                locations=(
                    capture_recovery.CaptureProposalLocation(
                        frame_index=7,
                        block_index=0,
                        tool_use_id="toolu_01",
                        json_pointer="/message/content/0/input",
                        envelope_shape="tool_use_input",
                    ),
                ),
            ),
        ),
        valid_group_count=1,
    )


def test_inspect_capture_prints_pointers_and_verdicts_and_stages_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    recovery = FakeRecovery(plan=_readable_plan())

    with installed_recovery(recovery):
        code = run(root, "extract-batch", "inspect-capture", "--operation", OPERATION)

    out = capsys.readouterr().out
    assert code == 0
    assert recovery.inspected == [OPERATION]
    assert recovery.staged == [] and recovery.rendered == []
    assert "e" * 64 in out
    assert "/message/content/0/input" in out
    assert "frame 7" in out
    assert "tool_use_input" in out
    assert "changed nothing and cost nothing" in out


def test_inspect_capture_reports_an_unreadable_reply_without_repairing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    recovery = FakeRecovery(
        plan=FakeCapturePlan(
            groups=(
                FakeGroup(
                    proposal_sha256="f" * 64,
                    # The shape the service's own verdict has: a pointer, a
                    # structural location, a message and an error type — and
                    # none of the argument that failed.
                    schema_verdict=(
                        "/message/content/0/input: invented: Extra inputs are "
                        "not permitted [extra_forbidden]"
                    ),
                    locations=(
                        FakeLocation(
                            7, 0, "toolu_01", "/message/content/0/input",
                            "tool_use_input",
                        ),
                    ),
                ),
            ),
            valid_group_count=0,
        )
    )

    with installed_recovery(recovery):
        code = run(root, "extract-batch", "inspect-capture", "--operation", OPERATION)

    out = capsys.readouterr().out
    assert code == 1
    assert "Extra inputs are not permitted [extra_forbidden]" in out
    assert recovery.staged == []


def test_recover_stages_a_sole_readable_proposal_with_no_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    recovery = FakeRecovery(plan=_readable_plan())

    with installed_recovery(recovery):
        code = run(
            root, "extract-batch", "recover", "--operation", OPERATION, "--yes"
        )

    out = capsys.readouterr().out
    assert code == 0
    assert recovery.staged == [(OPERATION, None)]
    assert "Nothing is sent" in out
    assert "3 proposal(s)" in out


def test_yes_answers_the_recovery_question_and_never_picks_a_proposal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--yes` is consent in advance, not a choice between two paid answers."""

    root, _sources = project(tmp_path)
    recovery = FakeRecovery(
        plan=FakeCapturePlan(
            groups=(
                FakeGroup(
                    "1" * 64,
                    "valid",
                    (FakeLocation(7, 0, "toolu_01", "/message/content/0/input", "tool_use_input"),),
                ),
                FakeGroup(
                    "2" * 64,
                    "valid",
                    (FakeLocation(9, 0, "toolu_02", "/message/content/0/input", "tool_use_input"),),
                ),
            ),
            valid_group_count=2,
        ),
        ambiguous=True,
    )

    with installed_recovery(recovery):
        code = run(
            root, "extract-batch", "recover", "--operation", OPERATION, "--yes"
        )

    assert code == 1
    # Exactly one attempt, carrying no selection: the command did not retry
    # with a proposal of its own choosing.
    assert recovery.staged == [(OPERATION, None)]
    assert "owner's decision" in capsys.readouterr().err


def test_recover_passes_the_owners_typed_location_through_the_capture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    # The same answer in two places, which is the case `--at` exists for.
    recovery = FakeRecovery(
        plan=FakeCapturePlan(
            groups=(
                FakeGroup(
                    proposal_sha256="e" * 64,
                    schema_verdict="valid",
                    locations=(
                        FakeLocation(
                            7, 0, "toolu_01", "/message/content/0/input",
                            "tool_use_input",
                        ),
                        FakeLocation(
                            9, 1, "toolu_02", "/message/content/1/input/input",
                            "tool_use_input_wrapped",
                        ),
                    ),
                ),
            ),
            valid_group_count=1,
        )
    )

    with installed_recovery(recovery):
        code = run(
            root,
            "extract-batch",
            "recover",
            "--operation",
            OPERATION,
            "--proposal",
            "e" * 64,
            "--at",
            "/message/content/1/input/input",
            "--yes",
        )

    assert code == 0
    operation_id, selection = recovery.staged[0]
    assert operation_id == OPERATION
    assert selection == FakeSelection("e" * 64, "/message/content/1/input/input")
    assert "/message/content/1/input/input" in capsys.readouterr().out


def test_a_named_proposal_without_at_selects_it_through_the_real_service(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--proposal SHA256` alone is the documented resolution, and it works.

    §9.3 spells the flag `--proposal SHA256 [--at POINTER]`: the pointer is
    what narrows a hash held in more than one place, not a second requirement
    for naming the hash at all. The plan here is the real dataclass, whose
    ``select`` branches on ``None`` rather than on truthiness.
    """
    root, _sources = project(tmp_path)
    plan = _real_plan()
    recovery = FakeRecovery(plan=plan)

    with installed_recovery(recovery):
        code = run(
            root,
            "extract-batch",
            "recover",
            "--operation",
            OPERATION,
            "--proposal",
            "e" * 64,
            "--yes",
        )

    out = capsys.readouterr().out
    assert code == 0
    operation_id, selection = recovery.staged[0]
    assert operation_id == OPERATION
    assert selection == plan.select("e" * 64)
    assert selection.json_pointer == "/message/content/0/input"
    assert selection.frame_index == 7 and selection.tool_use_id == "toolu_01"
    assert "Staging proposal " + "e" * 64 in out


def test_a_typed_proposal_the_capture_does_not_hold_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hash is typed, so the real service still decides what it names."""

    root, _sources = project(tmp_path)
    recovery = FakeRecovery(plan=_real_plan())

    with installed_recovery(recovery):
        code = run(
            root,
            "extract-batch",
            "recover",
            "--operation",
            OPERATION,
            "--proposal",
            "9" * 64,
            "--yes",
        )

    assert code == 1
    assert recovery.staged == []
    assert "holds no proposal" in capsys.readouterr().err


def test_a_location_without_a_proposal_refuses_before_reaching_the_service(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    recovery = FakeRecovery(plan=_readable_plan())

    with installed_recovery(recovery):
        code = run(
            root,
            "extract-batch",
            "recover",
            "--operation",
            OPERATION,
            "--at",
            "/message/content/0/input",
        )

    assert code == 1
    assert recovery.inspected == [] and recovery.staged == []
    assert "--proposal SHA256" in capsys.readouterr().err


def test_recover_refuses_unattended_without_yes_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    recovery = FakeRecovery(plan=_readable_plan())

    with installed_recovery(recovery):
        code = run(root, "extract-batch", "recover", "--operation", OPERATION)

    assert code == 1
    assert recovery.staged == []
    assert "The captured reply is untouched." in capsys.readouterr().out


def test_render_proposals_writes_the_page_and_says_it_approves_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _sources = project(tmp_path)
    output = tmp_path / "proposals.html"
    recovery = FakeRecovery(plan=_readable_plan())

    with installed_recovery(recovery):
        code = run(
            root,
            "extract-batch",
            "render-proposals",
            "--operation",
            OPERATION,
            "--output",
            str(output),
        )

    out = capsys.readouterr().out
    assert code == 0
    assert recovery.rendered == [(OPERATION, output)]
    assert recovery.staged == []
    assert output.read_text(encoding="utf-8") == "<html>proposals</html>"
    assert "approves nothing" in out
