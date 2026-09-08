"""Many prepared corpus parts, sent once, under one consent.

A long document is already split before this module sees it: the parts are
durable files in the corpus, prepared by `inputs.prepare_corpus_input`, and
nothing here copies, slices, OCRs or looks for rows. What a batch adds is the
*authority* shape — one confirmation covering several sources, a bounded
number of calls in flight, and a durable record of which child is which.

Three rules decide almost everything below.

**Every child is revalidated before any child is reserved.** A batch is one
consent, so a single moved source, edited prompt, re-scoped deck or occupied
output means zero sends and zero discarded evidence — never "three of five
went". `revalidate_extraction_request` is the single-call service's own
revalidation, called per child, so the two surfaces cannot drift apart about
what "the request changed" means.

**The journal is the authority; the manifest is only a receipt.** The manifest
is written before the reservation precisely so a crash between them leaves
evidence, and that evidence is readable by status and preview. It never
permits a send: `authorize_batch` reserves all children or none, and
`claim_batch_dispatch` is what turns one reservation into one dispatch.

**A retry is a discard decision, not a repeat.** The exact old `Operation`
snapshots ride in the plan, the same confirmation that authorizes the new
calls authorizes retiring them, and the new calls take fresh operation ids
even when they carry the identical request. Committed work is never retried
automatically, and an unknown outcome is a person's decision first.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from japanese_anki import (
    card_preview,
    claude_client,
    extract,
    inputs,
    operations,
    patterns,
    prompts,
    staging,
)
from japanese_anki.application import assignment, revision_provider
from japanese_anki.application.extraction import (
    ExtractionDispatchError,
    ExtractionDispatchExpectation,
    ExtractionOutcome,
    ExtractionRevision,
    ExtractionTarget,
    RevalidatedExtraction,
    complete_extraction,
    destination_deck_facts,
    extraction_capture_parts,
    extraction_provider,
    extraction_replacement_revision,
    plan_extraction,
    prepare_extraction_transport,
    recover_extraction_from_capture,
    require_current_destination,
    revalidate_extraction_request,
    run_extraction_lifecycle,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    _parse_structured_text,
    atomic_write_text_bound,
    load_records,
    prepare_bound_directory,
    read_bytes_bound,
)
from japanese_anki.models import VocabularyRecord

__all__ = [
    "BATCH_DIR_NAME",
    "EXECUTION_SUFFIX",
    "MAX_CONCURRENCY",
    "RETIRED",
    "UNRESERVED",
    "ExtractionBatchChild",
    "ExtractionBatchChildOutcome",
    "ExtractionBatchOutcome",
    "ExtractionBatchPlan",
    "ExtractionBatchPreview",
    "ExtractionBatchProgress",
    "SourcePartLineage",
    "batch_manifest_path",
    "dispatch_extraction_batch",
    "extraction_batch_status",
    "list_extraction_batches",
    "plan_extraction_batch",
    "plan_extraction_batch_retry",
    "render_extraction_batch_preview",
    "resume_extraction_batch",
]

#: Where a batch's immutable manifest lives, beside the journal that is its
#: authority. Not under the journal's `.pending` directory: that is bound
#: private evidence for one operation's reply, and a manifest is a plain
#: reviewable record of a request nobody may have sent yet.
BATCH_DIR_NAME = "extraction_batches"

#: The receipt that says an owner confirmed this exact execution. Written
#: beside the request manifest, and the only thing that lets a public resume
#: finish an interrupted cleanup and reservation.
EXECUTION_SUFFIX = ".execution.json"

#: The most calls one batch may have in flight. Small on purpose: a batch
#: buys a person's long document a little parallelism, and both surfaces
#: document this cap. The journal's own primitive stays general; this is the
#: extraction service's own rule about its own runs.
MAX_CONCURRENCY = 4

#: No reservation exists for this child's batch at all. Not a lifecycle state:
#: it is what a manifest written before its reservation looks like.
UNRESERVED = "unreserved"

#: The batch still names this member, but its journal entry is gone — the
#: shape an ordinary `forget` leaves. Distinct from `unreserved`, because the
#: id here was spent: it may never be sent again under this membership.
RETIRED = "retired"


def batch_manifest_path(config: ProjectConfig, batch_id: str) -> Path:
    """The one durable manifest path for a batch id."""
    if not _valid_batch_id(batch_id):
        raise operations.OperationError(f"Not a batch identity: {batch_id!r}")
    return config.operations_file.parent / BATCH_DIR_NAME / f"{batch_id}.json"


def _valid_batch_id(batch_id: str) -> bool:
    """A batch id is a uuid, so it can never name a path outside its store."""
    try:
        return str(uuid.UUID(str(batch_id))) == str(batch_id)
    except (AttributeError, TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class SourcePartLineage:
    """Where one prepared part came from, said once and never re-derived.

    Documentary only. The part's own path and hash are the facts every
    request, journal entry and staging file is bound to; this says which
    document a person split it out of and how they described that split.
    ``descriptor`` is opaque text — nothing here parses it, and in particular
    nothing reads a page range out of it to decide what to send.
    """

    parent_name: str
    parent_sha256: str
    descriptor: str

    def to_dict(self) -> dict[str, str]:
        return {
            "parent_name": self.parent_name,
            "parent_sha256": self.parent_sha256,
            "descriptor": self.descriptor,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SourcePartLineage:
        return cls(
            parent_name=str(raw.get("parent_name", "")),
            parent_sha256=str(raw.get("parent_sha256", "")),
            descriptor=str(raw.get("descriptor", "")),
        )


def _frozen_provenance(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """The exact request identity, normalized once and then read-only.

    Round-tripped through JSON at the point it enters a child, so what the
    manifest holds and what the child carries cannot differ.
    """
    return MappingProxyType(
        json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
    )


@dataclass(frozen=True, slots=True)
class ExtractionBatchChild:
    """One source in a batch, and the exact call reserved for it.

    ``expectation`` is the single-call service's own confirmed-request value,
    which is what lets one child be revalidated by exactly the same code a
    single dispatch uses. ``provenance`` is the provider request as planned —
    manifest, channels, prompts and all — kept so a captured reply can be
    recovered later without asking today's prompt files what was sent.
    """

    index: int
    operation_id: str
    expectation: ExtractionDispatchExpectation
    provenance: Mapping[str, Any]
    lineage: SourcePartLineage | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _frozen_provenance(self.provenance))

    @property
    def source(self) -> Path:
        """This child's own durable corpus file, never its parent document."""
        return self.expectation.source

    @property
    def source_sha256(self) -> str:
        """This child's own bytes, never its parent's."""
        return self.expectation.source_sha256

    @property
    def request_fingerprint(self) -> str:
        return self.expectation.request_fingerprint

    @property
    def staging_path(self) -> Path:
        return self.expectation.staging_path

    def to_dict(self) -> dict[str, Any]:
        expectation = self.expectation
        value: dict[str, Any] = {
            "index": self.index,
            "operation_id": self.operation_id,
            "source": str(expectation.source),
            "source_sha256": expectation.source_sha256,
            "request_fingerprint": expectation.request_fingerprint,
            "provider": expectation.provider,
            "model": expectation.model,
            "mode": expectation.mode,
            "scope_id": expectation.scope_id,
            "staging_path": str(expectation.staging_path),
            "patterns_path": str(expectation.patterns_path),
            "operations_path": str(expectation.operations_path),
            "replacement_confirmed": expectation.replacement_confirmed,
            "provenance": dict(self.provenance),
        }
        if expectation.destination_deck is not None:
            value["destination_deck"] = str(expectation.destination_deck)
            value["destination_deck_sha256"] = expectation.destination_deck_sha256
        if expectation.replacement_revision is not None:
            revision = expectation.replacement_revision
            value["replacement_revision"] = {
                "staging_sha256": revision.staging_sha256,
                "pattern_entry_sha256": revision.pattern_entry_sha256,
                "pattern_reviewed": revision.pattern_reviewed,
                "pattern_has_patterns": revision.pattern_has_patterns,
            }
        if self.lineage is not None:
            value["lineage"] = self.lineage.to_dict()
        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExtractionBatchChild:
        revision_raw = raw.get("replacement_revision")
        revision = (
            ExtractionRevision(
                staging_sha256=revision_raw.get("staging_sha256"),
                pattern_entry_sha256=revision_raw.get("pattern_entry_sha256"),
                pattern_reviewed=revision_raw.get("pattern_reviewed"),
                pattern_has_patterns=revision_raw.get("pattern_has_patterns"),
            )
            if isinstance(revision_raw, Mapping)
            else None
        )
        deck = raw.get("destination_deck")
        expectation = ExtractionDispatchExpectation(
            source=Path(str(raw["source"])),
            provider=str(raw["provider"]),
            model=str(raw["model"]),
            mode=raw.get("mode"),
            source_sha256=str(raw["source_sha256"]),
            request_fingerprint=str(raw["request_fingerprint"]),
            replacement_revision=revision,
            replacement_confirmed=bool(raw.get("replacement_confirmed", False)),
            staging_path=Path(str(raw["staging_path"])),
            patterns_path=Path(str(raw["patterns_path"])),
            operations_path=Path(str(raw["operations_path"])),
            scope_id=str(raw.get("scope_id", "")),
            destination_deck=None if deck is None else Path(str(deck)),
            destination_deck_sha256=str(raw.get("destination_deck_sha256", "")),
        )
        lineage_raw = raw.get("lineage")
        return cls(
            index=int(raw["index"]),
            operation_id=str(raw["operation_id"]),
            expectation=expectation,
            provenance=raw.get("provenance") or {},
            lineage=(
                SourcePartLineage.from_dict(lineage_raw)
                if isinstance(lineage_raw, Mapping)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ExtractionBatchPlan:
    """What a batch would send, to whom, and what it would retire first.

    Immutable and exactly serializable: `manifest_bytes` is the durable
    record, and reading it back gives this value again. Nothing here has been
    reserved — a plan is what a surface renders for one confirmation.
    """

    batch_id: str
    children: tuple[ExtractionBatchChild, ...]
    concurrency_limit: int
    provider: str
    model: str
    scope_id: str = ""
    #: The deck whose scope this batch was planned for, when one was named.
    #: A consent binding rather than a destination this run writes to.
    destination_deck: Path | None = None
    #: The batch this one retries, when it is a retry.
    retry_of: str = ""
    #: The exact journal snapshots this confirmation would retire, as the
    #: journal itself holds them: they are both what a person is asked to throw
    #: away and the guard `forget` compares under its own lock.
    discards: tuple[operations.Operation, ...] = ()

    @property
    def manifest_bytes(self) -> bytes:
        """The exact durable bytes, newline-terminated, in one canonical order."""
        return (
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.manifest_bytes).hexdigest()

    @property
    def fingerprint(self) -> str:
        """The identity of the whole action somebody is about to confirm.

        Bound to the complete plan, not to the request text inside it. How many
        calls run at once, which reviews would be destroyed, which entries
        would be retired and which fresh authorities would be written are all
        part of what is being agreed to — a fingerprint that ignored them would
        let a surface carry one confirmation onto a different decision.

        A retry therefore has its own fingerprint even though every child
        request fingerprint inside it is unchanged. That is the point: sending
        the same request again under fresh authority, after retiring an old
        entry, is a different thing to agree to.
        """
        return hashlib.sha256(
            b"janki-extraction-batch-consent-v1\n" + self.manifest_bytes
        ).hexdigest()

    def child(self, index: int) -> ExtractionBatchChild:
        for candidate in self.children:
            if candidate.index == index:
                return candidate
        raise operations.OperationError(
            f"This batch has no child {index}; it has "
            f"{len(self.children)} of them."
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "version": 1,
            "batch_id": self.batch_id,
            "concurrency_limit": self.concurrency_limit,
            "provider": self.provider,
            "model": self.model,
            "scope_id": self.scope_id,
            "children": [child.to_dict() for child in self.children],
        }
        if self.destination_deck is not None:
            value["destination_deck"] = str(self.destination_deck)
        if self.retry_of:
            value["retry_of"] = self.retry_of
        if self.discards:
            value["discards"] = [
                {
                    "operation_id": discard.operation_id,
                    "operation": discard.to_dict(),
                }
                for discard in self.discards
            ]
        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExtractionBatchPlan:
        children = tuple(
            ExtractionBatchChild.from_dict(entry) for entry in raw["children"]
        )
        # The journal path a discarded snapshot was read against is the one
        # every child is bound to; a manifest whose children disagree about it
        # is not a batch this reader will reassemble.
        journal_paths = {child.expectation.operations_path for child in children}
        if len(journal_paths) != 1:
            raise operations.OperationError(
                "This batch manifest names more than one operations journal."
            )
        journal_path = next(iter(journal_paths))
        deck = raw.get("destination_deck")
        return cls(
            batch_id=str(raw["batch_id"]),
            children=children,
            concurrency_limit=int(raw["concurrency_limit"]),
            provider=str(raw["provider"]),
            model=str(raw["model"]),
            scope_id=str(raw.get("scope_id", "")),
            destination_deck=None if deck is None else Path(str(deck)),
            retry_of=str(raw.get("retry_of", "")),
            discards=tuple(
                operations.Operation.from_dict(
                    journal_path,
                    str(entry["operation_id"]),
                    entry["operation"],
                )
                for entry in raw.get("discards", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class ExtractionBatchProgress:
    """One best-effort note about one child, for display only.

    A callback that raises, blocks or disappears cannot change what the
    journal records: this is handed to `_report` and every exception from it
    is swallowed, because a closed browser tab must not strand an authority or
    an answer somebody paid for.
    """

    index: int
    operation_id: str
    state: str
    message: str


@dataclass(frozen=True, slots=True)
class ExtractionBatchChildOutcome:
    """What became of one child, read from the journal rather than guessed."""

    index: int
    operation_id: str
    source: Path
    #: The journal's own lifecycle state, or `UNRESERVED` / `RETIRED` — both of
    #: which are facts about the reservation rather than invented states.
    state: str
    staging_path: Path
    error: str = ""
    #: False when staging committed but this answer's separate pattern write
    #: did not. The proposals are safe; the grammar half owes a person.
    bookkeeping_complete: bool = True
    #: How many proposals this child's staging document holds. ``None`` when
    #: there is no committed document to count, which is a different statement
    #: from a document holding nothing.
    records: int | None = None

    @property
    def committed(self) -> bool:
        return self.state == "committed"


@dataclass(frozen=True, slots=True)
class ExtractionBatchOutcome:
    """Where a whole batch stands, at one moment, per child."""

    batch_id: str
    children: tuple[ExtractionBatchChildOutcome, ...]
    concurrency_limit: int
    #: Whether `resume_extraction_batch` has work it may legitimately do. Read
    #: from durable state here rather than guessed by each surface from the
    #: child states, so a button and the service agree about eligibility.
    resume_available: bool = False
    #: Why not, in a person's words. Empty exactly when resume is available.
    resume_refusal: str = ""

    @property
    def committed_count(self) -> int:
        return sum(1 for child in self.children if child.state == "committed")

    @property
    def unknown_count(self) -> int:
        """Children whose provider outcome nobody knows. A person's decision."""
        return sum(1 for child in self.children if child.state == "outcome_unknown")

    @property
    def pending_count(self) -> int:
        """Children still moving, or still holding unused authority.

        Includes a child with no journal entry: a manifest written before its
        reservation describes work nothing has been spent on yet.
        """
        return sum(
            1
            for child in self.children
            if child.state
            in (UNRESERVED, "authorized", "dispatching", "running")
        )

    @property
    def failed_count(self) -> int:
        """Everything else: settled without staging, or paid but not staged."""
        return (
            len(self.children)
            - self.committed_count
            - self.unknown_count
            - self.pending_count
        )


@dataclass(frozen=True, slots=True)
class ExtractionBatchPreview:
    """A whole batch's proposals, drawn as the deck's own real cards."""

    preview: card_preview.CardPreview
    #: Plain supplementary sentences about identities more than one child
    #: proposed, naming the exact differing fields and each source's literal
    #: value. Structural comparisons only: which of two proposed readings is
    #: right is a question about Japanese, and nothing here answers it.
    conflicts: tuple[str, ...]
    #: The children whose saved proposals this preview drew.
    child_indices: tuple[int, ...]


def _report(
    progress: Callable[[ExtractionBatchProgress], None] | None,
    lock: threading.Lock,
    event: ExtractionBatchProgress,
) -> None:
    if progress is None:
        return
    try:
        with lock:
            progress(event)
    except Exception:  # noqa: BLE001 - presentation cannot control a paid batch
        # Same rule as the single-call service: a tab, socket or event loop
        # can disappear after authority is journalled, and a display must
        # never be able to interrupt the lifecycle that follows.
        return


def _require_subscription_batch(config: ProjectConfig) -> str:
    """A batch runs on the owner's subscription, or it does not run.

    No fallback and no configured alternative: the metered API path builds its
    request inside the client, one call at a time, and quietly billing a key
    for a run somebody confirmed as a subscription batch is exactly the
    substitution the extraction service refuses everywhere else.
    """
    provider = extraction_provider(config)
    if provider != revision_provider.CLAUDE_CODE_PROVIDER:
        raise extract.ExtractError(
            f"A batch extraction runs on the {revision_provider.CLAUDE_CODE_PROVIDER} "
            f"subscription; this project is configured for {provider!r}, so send "
            "these sources one at a time instead.",
            code="extract-batch-provider-unsupported",
        )
    return provider


def _destination_binding(
    config: ProjectConfig,
    *,
    scope_id: str,
    destination_deck: Path | None,
) -> tuple[str, Path | None, str]:
    """The scope this batch suppresses known words in, and what bound it."""
    if destination_deck is None:
        return scope_id, None, ""
    deck_path = Path(os.path.realpath(destination_deck))
    deck_scope, deck_sha256 = destination_deck_facts(deck_path)
    if scope_id and scope_id != deck_scope:
        raise staging.StagingError(
            f"{deck_path.name} holds the {deck_scope or 'shared'} collection, not "
            f"{scope_id!r}, so it cannot be this batch's destination."
        )
    return deck_scope, deck_path, deck_sha256


def plan_extraction_batch(
    config: ProjectConfig,
    sources: Sequence[Path],
    *,
    mode: str | None = None,
    model: str | None = None,
    scope_id: str = "",
    destination_deck: Path | None = None,
    concurrency_limit: int = 2,
    force: bool = False,
    lineage: Sequence[SourcePartLineage] = (),
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
) -> ExtractionBatchPlan:
    """Resolve what a whole batch would send, before anybody agrees to it.

    Every source must already be a durable corpus file: adding a document and
    sending it are two separate actions, and splitting one is a third that
    happened before this was called. Each part is sent whole — there is no
    sampling, no first-page-only, and no row detection anywhere in here.

    Nothing is journalled, nothing is written, and no provider is called
    beyond the login probe every extraction plan already makes.
    """
    chosen_sources = tuple(Path(source) for source in sources)
    if not chosen_sources:
        raise extract.ExtractError(
            "A batch needs at least one source.", code="extract-batch-empty"
        )
    # Before the login probe and before anything is planned: a limit that is
    # not a small whole number is a mistake about the request, not a discovery
    # to make after touching the provider.
    if type(concurrency_limit) is not int or not (
        1 <= concurrency_limit <= MAX_CONCURRENCY
    ):
        raise extract.ExtractError(
            f"A batch runs between 1 and {MAX_CONCURRENCY} calls at a time, "
            f"not {concurrency_limit!r}.",
            code="extract-batch-concurrency",
        )
    parts = tuple(lineage)
    if parts and len(parts) != len(chosen_sources):
        raise extract.ExtractError(
            "Lineage describes each source in order, so there must be one entry "
            f"per source: {len(parts)} for {len(chosen_sources)} sources.",
            code="extract-batch-lineage",
        )
    seen: dict[Path, int] = {}
    for position, source in enumerate(chosen_sources, start=1):
        resolved = Path(os.path.realpath(source))
        if resolved in seen:
            raise extract.ExtractError(
                f"{source.name} is in this batch twice, as sources "
                f"{seen[resolved]} and {position}; one source makes one request.",
                code="extract-batch-duplicate-source",
            )
        seen[resolved] = position

    provider = _require_subscription_batch(config)
    chosen_model = model or config.extract_model
    batch_scope, deck_path, deck_sha256 = _destination_binding(
        config, scope_id=scope_id, destination_deck=destination_deck
    )

    prepared = [
        inputs.prepare_corpus_input(source, config.scan_inbox)
        for source in chosen_sources
    ]
    plan = plan_extraction(
        config,
        prepared,
        mode=mode,
        model=chosen_model,
        style_guide=claude_client.read_style_guide(config.root),
        system=prompts.load(config.root, extract.prompt_name(mode)),
        force=force,
        scope_id=batch_scope,
        provider=provider,
        provider_env=provider_env,
        provider_runner=provider_runner,
        provider_which=provider_which,
    )

    children: list[ExtractionBatchChild] = []
    fingerprints: dict[str, int] = {}
    for position, target in enumerate(plan.targets, start=1):
        fingerprint = str(target.provenance["request_fingerprint"])
        if fingerprint in fingerprints:
            # Equal source hashes are ordinary — two pages of one worksheet can
            # hold identical bytes. Two identical *requests* are not: they are
            # the same call, and one batch must not pay for it twice.
            raise extract.ExtractError(
                f"{target.name} makes the same request as source "
                f"{fingerprints[fingerprint]}, so this batch would send it "
                "twice.",
                code="extract-batch-duplicate-request",
            )
        fingerprints[fingerprint] = position
        revision = extraction_replacement_revision(config, target)
        children.append(
            ExtractionBatchChild(
                index=position,
                operation_id=str(uuid.uuid4()),
                expectation=ExtractionDispatchExpectation(
                    source=target.item.origin_path,
                    provider=plan.provider,
                    model=plan.model,
                    mode=plan.mode,
                    source_sha256=target.source_sha256,
                    request_fingerprint=fingerprint,
                    replacement_revision=revision,
                    replacement_confirmed=revision is not None,
                    staging_path=target.staging_path,
                    patterns_path=target.patterns_path,
                    operations_path=config.operations_file,
                    scope_id=plan.scope_id,
                    destination_deck=deck_path,
                    destination_deck_sha256=deck_sha256,
                ),
                provenance=target.provenance,
                lineage=parts[position - 1] if parts else None,
            )
        )

    return ExtractionBatchPlan(
        batch_id=str(uuid.uuid4()),
        children=tuple(children),
        concurrency_limit=concurrency_limit,
        provider=plan.provider,
        model=plan.model,
        scope_id=plan.scope_id,
        destination_deck=deck_path,
    )


def _load_batch_plan(config: ProjectConfig, batch_id: str) -> ExtractionBatchPlan:
    """Read one batch's immutable manifest, or say plainly that it is gone."""
    path = batch_manifest_path(config, batch_id)
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError as exc:
        raise operations.OperationError(
            f"No extraction batch {batch_id} is recorded at {path}."
        ) from exc
    except (JankiError, OSError) as exc:
        raise operations.OperationError(f"Could not read {path}: {exc}") from exc
    try:
        raw = json.loads(wire.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError) as exc:
        raise operations.OperationError(f"Could not read {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise operations.OperationError(f"{path} must hold an object")
    plan = ExtractionBatchPlan.from_dict(raw)
    if plan.batch_id != batch_id:
        raise operations.OperationError(
            f"{path} records batch {plan.batch_id}, not {batch_id}."
        )
    _require_authentic_reservation(config, plan)
    return plan


def _require_authentic_reservation(
    config: ProjectConfig, plan: ExtractionBatchPlan
) -> None:
    """Make the journal, not the file, say what this reservation covers.

    A manifest is a document on disk, and a document can be edited. Once a
    batch is reserved, the journal holds the hash of the manifest that was
    authorized, the exact membership and the limit — so anything read back
    under that batch id has to match all three before it may be recovered
    against, written from, or even reported as this batch's state. Otherwise
    an edited output path would be enough to redirect an answer somebody
    already paid for.

    An unreserved manifest is left alone here. It is provenance about a
    request nobody authorized, and it is never authority for anything.
    """
    journal = operations.OperationJournal.load(config.operations_file)
    batch = journal.batches.get(plan.batch_id)
    if batch is None:
        return
    complaint = ""
    if batch.manifest_sha256 != plan.manifest_sha256:
        complaint = (
            "its bytes are not the ones that were authorized"
        )
    elif tuple(batch.child_operation_ids) != tuple(
        child.operation_id for child in plan.children
    ):
        complaint = "it names different operations than the reservation does"
    elif batch.concurrency_limit != plan.concurrency_limit:
        complaint = "it names a different worker limit than the reservation does"
    if complaint:
        raise operations.OperationError(
            f"The manifest for batch {plan.batch_id} cannot be trusted: "
            f"{complaint}. janki will not recover, write or report anything "
            "against it."
        )
    _confirmed_execution(config, plan)


def _write_exactly_once(path: Path, payload: bytes, *, what: str) -> Path:
    """Publish these exact bytes, or prove they are already there.

    One batch id names one action. Writing the same bytes again is the same
    action retried after an interruption, so it is accepted; different bytes
    under the same name would be a second action wearing the first one's
    identity, and that is refused rather than resolved.
    """
    if path.exists():
        try:
            held = read_bytes_bound(path)
        except (JankiError, OSError) as exc:
            raise operations.OperationError(
                f"Could not read the existing {what} {path}: {exc}"
            ) from exc
        if held != payload:
            raise operations.OperationError(
                f"The {what} {path} records a different action than this one; "
                "its bytes changed, so nothing was retired, reserved or sent."
            )
        return path
    try:
        atomic_write_text_bound(
            path,
            payload.decode("utf-8"),
            expected_absent=True,
        )
    except JankiError as exc:
        raise operations.OperationError(
            f"Could not write the {what} {path}: {exc}"
        ) from exc
    return path


def _write_batch_manifest(config: ProjectConfig, plan: ExtractionBatchPlan) -> Path:
    """Persist the exact request manifest, before anything else happens.

    Deliberately first. The manifest is not authority — the journal is — so a
    crash after this and before the confirmed-execution receipt leaves a
    reviewable record of a request nothing was spent on, and that record can
    never on its own let anything be sent.
    """
    prepare_bound_directory(config.operations_file.parent / BATCH_DIR_NAME)
    return _write_exactly_once(
        batch_manifest_path(config, plan.batch_id),
        plan.manifest_bytes,
        what="batch manifest",
    )


def execution_receipt_path(config: ProjectConfig, batch_id: str) -> Path:
    """Where one batch's confirmed-execution receipt lives."""
    if not _valid_batch_id(batch_id):
        raise operations.OperationError(f"Not a batch identity: {batch_id!r}")
    return (
        config.operations_file.parent
        / BATCH_DIR_NAME
        / f"{batch_id}{EXECUTION_SUFFIX}"
    )


def _execution_receipt_bytes(plan: ExtractionBatchPlan) -> bytes:
    """The exact intent an owner confirmed, separate from any progress at it.

    Immutable and small on purpose: the manifest it names is the request, and
    this says *that* request was confirmed for execution, with these fresh
    operation ids and these exact retirements. Cleanup progress is the
    journal's own durable business and is deliberately not written here.
    """
    return (
        json.dumps(
            {
                "version": 1,
                "batch_id": plan.batch_id,
                "manifest_sha256": plan.manifest_sha256,
                "fingerprint": plan.fingerprint,
                "child_operation_ids": [
                    child.operation_id for child in plan.children
                ],
                "discards": [
                    {
                        "operation_id": discard.operation_id,
                        "operation": discard.to_dict(),
                    }
                    for discard in plan.discards
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_execution_receipt(config: ProjectConfig, plan: ExtractionBatchPlan) -> Path:
    """Record that this exact execution was confirmed, before any cleanup.

    The one thing that separates "a manifest exists" from "somebody said send
    this". Written after every current binding has been revalidated and before
    a single retirement or reservation, so an interrupted batch can be finished
    later without re-deriving what its owner agreed to.
    """
    prepare_bound_directory(config.operations_file.parent / BATCH_DIR_NAME)
    return _write_exactly_once(
        execution_receipt_path(config, plan.batch_id),
        _execution_receipt_bytes(plan),
        what="confirmed execution receipt",
    )


def _require_recorded_action(config: ProjectConfig, plan: ExtractionBatchPlan) -> None:
    """Prove what is already recorded under this batch id is this same action.

    Applies on every public entry that carries a plan of its own, whether or
    not there is work left: a batch id names one action, and a stored manifest
    that no longer matches the plan in hand is a disagreement to report rather
    than a difference to reconcile.
    """
    path = batch_manifest_path(config, plan.batch_id)
    if path.exists():
        try:
            held = read_bytes_bound(path)
        except (JankiError, OSError) as exc:
            raise operations.OperationError(
                f"Could not read the batch manifest {path}: {exc}"
            ) from exc
        if held != plan.manifest_bytes:
            raise operations.OperationError(
                f"The batch manifest {path} records a different action than "
                "this one; its bytes changed, so nothing was retired, reserved "
                "or sent."
            )
    _confirmed_execution(config, plan)


def _confirmed_execution(config: ProjectConfig, plan: ExtractionBatchPlan) -> bool:
    """Whether this exact plan was confirmed for execution and still matches."""
    path = execution_receipt_path(config, plan.batch_id)
    if not path.exists():
        return False
    try:
        held = read_bytes_bound(path)
    except (JankiError, OSError) as exc:
        raise operations.OperationError(
            f"Could not read the confirmed execution receipt {path}: {exc}"
        ) from exc
    if held != _execution_receipt_bytes(plan):
        raise operations.OperationError(
            f"The confirmed execution receipt {path} describes a different "
            "action than this batch's manifest; janki will not guess which "
            "one somebody agreed to."
        )
    return True


def _revalidate_children(
    config: ProjectConfig,
    plan: ExtractionBatchPlan,
    children: Sequence[ExtractionBatchChild],
    *,
    provider_env: Mapping[str, str] | None,
    provider_runner: Callable[..., Any],
    provider_which: Callable[..., str | None],
) -> dict[int, RevalidatedExtraction]:
    """Read every child's world again, and refuse the batch on any difference.

    All of them, before any of them. One consent covered these sources
    together, so a source that moved, a prompt that changed, an output that is
    now occupied or a destination deck that was re-scoped cancels the whole
    batch rather than the child that happened to notice.
    """
    revalidated: dict[int, RevalidatedExtraction] = {}
    for child in children:
        try:
            revalidated[child.index] = revalidate_extraction_request(
                config,
                child.expectation,
                provider_env=provider_env,
                provider_runner=provider_runner,
                provider_which=provider_which,
            )
        except ExtractionDispatchError as exc:
            # Which source moved is the whole message here: "the extraction
            # request changed" about an unnamed member of a five-part batch
            # tells a person nothing they can act on.
            label = (
                f"{child.source.name} (source {child.index} of "
                f"{len(plan.children)})"
            )
            try:
                cause: JankiError = type(exc.cause)(f"{label}: {exc}")
            except TypeError:
                cause = JankiError(f"{label}: {exc}")
            raise ExtractionDispatchError(cause, phase=exc.phase) from exc
    return revalidated


def _reserve_batch(config: ProjectConfig, plan: ExtractionBatchPlan) -> None:
    """Reserve every child, or none of them."""
    journal = operations.OperationJournal.load(config.operations_file)
    operations.prepare_artifact_store(journal.path)
    for child in plan.children:
        for parent in {
            child.expectation.staging_path.parent,
            child.expectation.patterns_path.parent,
        }:
            prepare_bound_directory(parent)
    journal.authorize_batch(
        plan.batch_id,
        tuple(
            operations.OperationAuthorization(
                operation_id=child.operation_id,
                kind="extract",
                source_file=child.source.name,
                source_sha256=child.source_sha256,
                request_fp=child.request_fingerprint,
                model=plan.model,
            )
            for child in plan.children
        ),
        concurrency_limit=plan.concurrency_limit,
        manifest_sha256=plan.manifest_sha256,
    )


def _require_discardable(
    config: ProjectConfig, plan: ExtractionBatchPlan
) -> None:
    """Prove every proposed retirement is still exactly what was rendered.

    Before the manifest, before the retirements themselves and before any
    reservation. `forget` makes the same comparison under the journal's lock,
    which is what actually closes the race; this is what makes a moved
    snapshot cost nothing at all rather than half a batch.
    """
    if not plan.discards:
        return
    journal = operations.OperationJournal.load(config.operations_file)
    for discard in plan.discards:
        held = journal.operations.get(discard.operation_id)
        if held != discard:
            raise operations.OperationError(
                f"Operation {discard.operation_id!r} changed after this retry "
                "was rendered, so nothing was retired and nothing was sent."
            )


#: What makes two journal rows the same paid decision. Deliberately not the
#: whole entry: a retirement in progress moves `updated_at`, records its
#: `cleanup` intent, and may bind an exact response-spool extension the
#: journal accepted. None of those change whose call it was.
_OPERATION_IDENTITY = (
    "operation_id",
    "kind",
    "source_file",
    "source_sha256",
    "request_fp",
    "model",
    "authorized_at",
    "batch_id",
)


def _same_operation_identity(
    held: operations.Operation, expected: operations.Operation
) -> bool:
    return all(
        getattr(held, field) == getattr(expected, field)
        for field in _OPERATION_IDENTITY
    )


def _execute_discards(config: ProjectConfig, plan: ExtractionBatchPlan) -> None:
    """Retire exactly the snapshots this confirmation named, one at a time.

    There is no atomic retire-and-reserve, and this does not pretend
    otherwise: a crash between two retirements, or between the last one and
    the reservation, leaves the journal's own durable cleanup evidence and a
    confirmed-execution receipt saying what was still owed. Running this again
    finishes the rest — a retirement already completed is recognized by its
    entry being gone, which is only safe *here*, inside the exact confirmed
    execution that named it.
    """
    journal = operations.OperationJournal.load(config.operations_file)
    for discard in plan.discards:
        held = journal.operations.get(discard.operation_id)
        if held is None:
            # This exact decision already landed. Not a fresh discovery about
            # an unrelated entry: it is named by the receipt being finished.
            continue
        if held == discard:
            journal.forget(
                [discard.operation_id],
                force=True,
                expected_operation=discard,
            )
            continue
        # The entry moved because `forget` itself was interrupted: it writes
        # its exact deletion decision before deleting anything, which stamps
        # `updated_at` and may legitimately adopt an exact spool extension.
        # Continue *that* decision under the journal's own compare-and-swap
        # rather than forcing a new one — and never by dropping the fields
        # that make it the same entry.
        if held.cleanup is None or not _same_operation_identity(held, discard):
            raise operations.OperationError(
                f"Operation {discard.operation_id!r} changed after this retry "
                "was rendered, so nothing was retired and nothing was sent."
            )
        journal.forget([discard.operation_id], expected_operation=held)


def _child_states(
    config: ProjectConfig, plan: ExtractionBatchPlan
) -> dict[int, str]:
    """Each child's actual journal state, through the batch's own membership.

    Membership comes from the journal's batch record rather than the manifest:
    a member the journal no longer holds was retired, and a manifest naming it
    is a receipt, never an authorization to send it again.
    """
    journal = operations.OperationJournal.load(config.operations_file)
    batch = journal.batches.get(plan.batch_id)
    if batch is None:
        return {child.index: UNRESERVED for child in plan.children}
    members = frozenset(batch.child_operation_ids)
    states: dict[int, str] = {}
    for child in plan.children:
        if child.operation_id not in members:
            # A reserved batch that does not name this child is not this
            # child's batch; treating it as unreserved would invite a resend.
            states[child.index] = RETIRED
            continue
        held = journal.operations.get(child.operation_id)
        states[child.index] = RETIRED if held is None else held.state
    return states


def _staging_record_count(path: Path) -> int:
    try:
        text = read_bytes_bound(path).decode("utf-8")
        records, _meta = staging.read_staging_text(text, source=str(path))
    except (JankiError, OSError, UnicodeError, ValueError):
        return 0
    return len(records)


def _pattern_entry_names(config: ProjectConfig) -> frozenset[str]:
    try:
        return frozenset(patterns.load_store(config.patterns_file))
    except JankiError:
        return frozenset()


#: Why a batch cannot be resumed, by what its children are doing.
_RESUME_REFUSALS: tuple[tuple[str, str], ...] = (
    (
        "dispatching",
        "a call may be in flight right now, so janki will not start another",
    ),
    ("running", "a call is with the provider right now"),
    (
        "outcome_unknown",
        "a call was sent and no answer was captured, so somebody has to decide "
        "what it bought before this batch moves again",
    ),
)


def _resume_eligibility(
    config: ProjectConfig,
    plan: ExtractionBatchPlan,
    states: Mapping[int, str],
) -> tuple[bool, str]:
    """Whether resuming this batch has anything it may legitimately do.

    Answered here rather than by each surface reading child states, because
    "may this be resumed" turns on a durable receipt that no list of states
    mentions: a request manifest alone is not a decision to send anything.
    """
    if not any(state == UNRESERVED for state in states.values()):
        if any(
            state in ("authorized", "result_captured") for state in states.values()
        ):
            return True, ""
        for state, why in _RESUME_REFUSALS:
            if state in states.values():
                return False, f"This batch cannot be resumed: {why}."
        return False, "Every source in this batch is already settled."
    if not _confirmed_execution(config, plan):
        return False, (
            "This batch has a request manifest but no confirmed execution, so "
            "nothing was ever authorized to be sent. Start it deliberately "
            "rather than resuming it."
        )
    return True, ""


def _outcome(
    config: ProjectConfig,
    plan: ExtractionBatchPlan,
    *,
    errors: Mapping[int, str] = MappingProxyType({}),
) -> ExtractionBatchOutcome:
    """Read the whole batch's current standing from durable state."""
    journal = operations.OperationJournal.load(config.operations_file)
    states = _child_states(config, plan)
    stored_patterns = _pattern_entry_names(config)
    children: list[ExtractionBatchChildOutcome] = []
    for child in plan.children:
        state = states[child.index]
        held = journal.operations.get(child.operation_id)
        committed = state == "committed"
        children.append(
            ExtractionBatchChildOutcome(
                index=child.index,
                operation_id=child.operation_id,
                source=child.source,
                state=state,
                staging_path=child.staging_path,
                error=errors.get(child.index, "" if held is None else held.detail),
                # A committed child owes its grammar half too. Read from the
                # store rather than remembered from this process, so a status
                # taken later says the same thing this run did.
                bookkeeping_complete=(
                    not committed or child.source.name in stored_patterns
                ),
                records=(
                    _staging_record_count(child.staging_path) if committed else None
                ),
            )
        )
    available, refusal = _resume_eligibility(config, plan, states)
    return ExtractionBatchOutcome(
        batch_id=plan.batch_id,
        children=tuple(children),
        concurrency_limit=plan.concurrency_limit,
        resume_available=available,
        resume_refusal=refusal,
    )


def _recover_child(
    config: ProjectConfig,
    child: ExtractionBatchChild,
) -> ExtractionOutcome:
    """Finish a child from the reply it already paid for, entirely offline.

    No provider, no login, and no prompt file is read. The captured artifact
    carries both halves — the request as planned and the reply as received —
    so recovery must not depend on today's prompts still existing or on a
    login still being valid. Somebody who logged out, or edited a prompt after
    the call, still owns that answer.

    What *is* checked is everything about now that the write depends on: the
    source artifact still being the one that was sent, the outputs still being
    the ones this request was bound to, the destination deck still being what
    the owner chose, and the review this answer is allowed to replace. Those
    are facts about the file system, not about a request, and none of them
    needs a provider to establish.
    """
    expectation = child.expectation
    journal = operations.OperationJournal.load(config.operations_file)
    payload = journal.read_reply(child.operation_id)
    _raw, provenance = extraction_capture_parts(payload)
    if provenance is None:
        raise ExtractionDispatchError(
            extract.ExtractError(
                f"The captured reply for {child.source.name} has no saved request "
                "beside it, so it cannot be recovered on its own.",
                code="extract-request-unavailable",
            ),
            phase="completion",
            operation_id=child.operation_id,
        )

    for current, bound, what in (
        (config.operations_file, expectation.operations_path, "operations journal"),
        (config.patterns_file, expectation.patterns_path, "pattern store"),
    ):
        if Path(os.path.realpath(current)) != bound:
            raise ExtractionDispatchError(
                operations.OperationError(
                    f"The {what} this answer was bound to is no longer the one "
                    "this project uses."
                ),
                phase="binding",
                operation_id=child.operation_id,
            )
    require_current_destination(expectation)

    try:
        item = inputs.prepare_corpus_input(child.source, config.scan_inbox)
    except JankiError as exc:
        raise ExtractionDispatchError(
            exc, phase="binding", operation_id=child.operation_id
        ) from exc
    fingerprint = item.source_sha256 or extract.source_fingerprint(item.origin_path)
    if fingerprint != child.source_sha256:
        raise ExtractionDispatchError(
            staging.StagingError(
                f"{child.source.name} changed after it was sent, so this answer "
                "no longer describes the file in the corpus."
            ),
            phase="binding",
            operation_id=child.operation_id,
        )

    # The answer's own provenance, never a rebuilt one: the completion's
    # identity check then compares the request that was made with itself.
    target = ExtractionTarget(
        item=item,
        staging_path=expectation.staging_path,
        patterns_path=expectation.patterns_path,
        source_sha256=child.source_sha256,
        provenance=provenance,
        replaces=(
            expectation.staging_path if expectation.staging_path.exists() else None
        ),
        provider_plan=None,
    )
    try:
        fresh_revision = extraction_replacement_revision(config, target)
    except JankiError as exc:
        raise ExtractionDispatchError(
            exc, phase="binding", operation_id=child.operation_id
        ) from exc
    rendered_revision = expectation.replacement_revision
    if fresh_revision is None or rendered_revision is None:
        if fresh_revision != rendered_revision:
            raise ExtractionDispatchError(
                staging.StagingError(
                    "The review this answer would replace changed after the "
                    "batch was confirmed."
                ),
                phase="binding",
                operation_id=child.operation_id,
            )
    elif (
        fresh_revision.staging_sha256 != rendered_revision.staging_sha256
        or fresh_revision.pattern_entry_sha256
        != rendered_revision.pattern_entry_sha256
    ):
        raise ExtractionDispatchError(
            staging.StagingError(
                "The review this answer would replace changed after the batch "
                "was confirmed."
            ),
            phase="binding",
            operation_id=child.operation_id,
        )

    saved_mode = provenance.get("mode")
    mode = None if saved_mode in (None, "auto") else str(saved_mode)
    result = recover_extraction_from_capture(
        payload, source_name=target.name, mode=mode
    )
    existing = (
        load_records(config.normalized_file)
        if config.normalized_file.exists()
        else []
    )
    try:
        return complete_extraction(
            config,
            journal,
            target,
            result,
            operation_id=child.operation_id,
            known=frozenset(
                extract.known_ids(existing, scope_id=expectation.scope_id)
            ),
            mode=mode,
            model=str(provenance.get("model") or expectation.model),
            force=expectation.replacement_confirmed,
            expected_revision=rendered_revision,
        )
    except JankiError as exc:
        raise ExtractionDispatchError(
            exc, phase="completion", operation_id=child.operation_id
        ) from exc


def _is_before_send_refusal(exc: ExtractionDispatchError) -> bool:
    """Whether this failure means nothing left the machine for that child.

    The phase alone cannot answer it: the lifecycle wraps a transport that
    died while being spawned as a `dispatch` failure, because from the
    journal's side a request may already have gone. So the journal's own
    classification decides — a failure that cost nothing is a local refusal
    that would meet every sibling identically, and queuing more sends behind
    it would just fail four more times.

    An answer that was captured, or an outcome nobody knows, is not that: the
    money moved, this child is that child's problem, and its siblings keep
    their reserved work.
    """
    if exc.phase in {"binding", "preparation", "authorization"}:
        return True
    if exc.journal_error is not None:
        # The journal could not even classify it. Stop rather than keep
        # spending against a record that cannot be trusted.
        return True
    failure = exc.failure
    if failure is None:
        return False
    return not failure.was_paid_for and not failure.money_may_have_been_spent


def _run_children(
    config: ProjectConfig,
    plan: ExtractionBatchPlan,
    revalidated: Mapping[int, RevalidatedExtraction],
    states: Mapping[int, str],
    *,
    progress: Callable[[ExtractionBatchProgress], None] | None,
    provider_env: Mapping[str, str] | None,
    provider_runner: Callable[..., Any],
    provider_which: Callable[..., str | None],
    provider_spawn: Any,
    errors: dict[int, str] | None = None,
) -> ExtractionBatchOutcome:
    """Work the reserved children, up to the batch's own worker limit.

    A bounded, finite set of calls that were all authorized together, so the
    executor is a plain `ThreadPoolExecutor` and not a job framework. Each
    worker holds its own journal handle; the journal's own locks are what
    order two children that finish at once, including the pattern-store
    read-modify-write inside the shared completion.
    """
    if errors is None:
        errors = {}
    lock = threading.Lock()
    # One refusal that is about the transport rather than about an answer —
    # no CLI, a logged-out login, an occupied slot — stops *queuing* more
    # work. The already reserved children keep their authority and a later
    # resume picks them up; nothing falls back to another provider.
    paused = threading.Event()

    def note(child: ExtractionBatchChild, state: str, message: str) -> None:
        _report(
            progress,
            lock,
            ExtractionBatchProgress(
                index=child.index,
                operation_id=child.operation_id,
                state=state,
                message=message,
            ),
        )

    def work(child: ExtractionBatchChild) -> None:
        if states[child.index] == "result_captured":
            # Recovering a paid answer is not a send, so a paused transport
            # does not stop it and a provider is never consulted.
            note(
                child,
                "result_captured",
                f"Reading the saved answer for {child.source.name}",
            )
            try:
                _recover_child(config, child)
            except JankiError as exc:
                with lock:
                    errors[child.index] = str(exc)
                note(child, "result_captured", str(exc))
                return
            note(child, "committed", f"Saved proposals for {child.source.name}")
            return

        if paused.is_set():
            return
        current = revalidated[child.index]
        journal = operations.OperationJournal.load(config.operations_file)
        try:
            transport = prepare_extraction_transport(
                current.plan,
                current.target,
                env=provider_env,
                runner=provider_runner,
                which=provider_which,
            )
        except JankiError as exc:
            paused.set()
            with lock:
                errors[child.index] = str(exc)
            note(child, states[child.index], str(exc))
            return

        try:
            if transport.prepared is not None:
                # Bound while the entry is still merely authorized, which is
                # the only window the journal accepts a spool in — and before
                # the claim, so an interrupted dispatch still leaves frames.
                journal.begin_response_capture(child.operation_id)
            journal.claim_batch_dispatch(
                child.operation_id,
                batch_id=plan.batch_id,
                request_fp=child.request_fingerprint,
                manifest_sha256=plan.manifest_sha256,
            )
        except JankiError as exc:
            # Either this reservation is no longer claimable or the batch's
            # slots are occupied. Both mean queued work waits for a person
            # rather than spinning: the authority is untouched.
            paused.set()
            with lock:
                errors[child.index] = str(exc)
            note(child, states[child.index], str(exc))
            return

        note(child, "dispatching", f"Sending {child.source.name}")
        try:
            run_extraction_lifecycle(
                config,
                journal,
                current,
                operation_id=child.operation_id,
                transport=transport,
                provider_spawn=provider_spawn,
            )
        except ExtractionDispatchError as exc:
            with lock:
                errors[child.index] = str(exc)
            if _is_before_send_refusal(exc):
                paused.set()
            note(child, exc.phase, str(exc))
            return
        note(child, "committed", f"Saved proposals for {child.source.name}")

    queued = [
        child
        for child in plan.children
        if states[child.index] == "result_captured"
        or (
            child.index in revalidated and states[child.index] == "authorized"
        )
    ]
    if queued:
        with ThreadPoolExecutor(
            max_workers=max(1, min(plan.concurrency_limit, len(queued))),
            thread_name_prefix="janki-extract-batch",
        ) as pool:
            for future in [pool.submit(work, child) for child in queued]:
                future.result()

    return _outcome(config, plan, errors=errors)


def dispatch_extraction_batch(
    config: ProjectConfig,
    plan: ExtractionBatchPlan,
    *,
    progress: Callable[[ExtractionBatchProgress], None] | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
    provider_spawn: Any = subprocess.Popen,
) -> ExtractionBatchOutcome:
    """Run one confirmed batch: revalidate all, reserve all, then send.

    The order is the whole design. Nothing is reserved until every child has
    been read again, nothing is retired until every retirement has been proved
    to be the one that was rendered, and the durable manifest lands before
    either — so every interruption leaves a state somebody can read rather
    than a batch half-authorized and half-explained.
    """
    # Whatever this batch already left on disk has to agree with the plan in
    # hand before anything else happens — including when every child is
    # settled, where there is nothing left to do but plenty left to be wrong
    # about. Divergent bytes are refused, never resolved.
    _require_recorded_action(config, plan)
    states = _child_states(config, plan)
    unsent = [
        child for child in plan.children if states[child.index] == UNRESERVED
    ]
    revalidated = _revalidate_children(
        config,
        plan,
        unsent,
        provider_env=provider_env,
        provider_runner=provider_runner,
        provider_which=provider_which,
    )
    if unsent:
        if not _confirmed_execution(config, plan):
            # A first confirmation. The snapshots it proposes to retire must
            # still be exactly the ones it was rendered against.
            _require_discardable(config, plan)
            _write_batch_manifest(config, plan)
            # Only now, with every current binding proved, is there something a
            # later resume may finish: the confirmed intent lands before the
            # first retirement, so an interruption never has to be guessed at.
            _write_execution_receipt(config, plan)
        # Otherwise this is the same confirmed action, sent again after an
        # interruption. Its own earlier retirements are why some snapshots are
        # already gone, so re-checking them against the journal would refuse
        # the batch for having done what it was asked to do.
        _execute_discards(config, plan)
        try:
            _reserve_batch(config, plan)
        except JankiError as exc:
            raise ExtractionDispatchError(exc, phase="authorization") from exc
        states = _child_states(config, plan)
    return _run_children(
        config,
        plan,
        revalidated,
        states,
        progress=progress,
        provider_env=provider_env,
        provider_runner=provider_runner,
        provider_which=provider_which,
        provider_spawn=provider_spawn,
    )


def resume_extraction_batch(
    config: ProjectConfig,
    batch_id: str,
    *,
    progress: Callable[[ExtractionBatchProgress], None] | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
    provider_spawn: Any = subprocess.Popen,
) -> ExtractionBatchOutcome:
    """Finish what an interrupted batch still has authority for, and no more.

    Only two kinds of child are touched. One still holding unused authority is
    dispatched; one whose exact reply is already on disk is recovered from
    those bytes without a second call. A child that is dispatching, running,
    of unknown outcome or already committed is reported, never reclaimed — the
    first two because something may be alive, the third because a person owes
    it a decision, and the last because it is done.

    A recorded execution may finish its interrupted reservation. A request
    manifest alone cannot authorize that work or widen the original consent.
    """
    plan = _load_batch_plan(config, batch_id)
    states = _child_states(config, plan)
    available, refusal = _resume_eligibility(config, plan, states)
    if not available:
        if any(state == UNRESERVED for state in states.values()):
            # A request manifest with no confirmed execution behind it. There
            # is no authority to finish, so this is a refusal rather than a
            # report: answering with a status would read as "nothing to do".
            raise ExtractionDispatchError(
                operations.OperationError(refusal), phase="authorization"
            )
        # Reserved, and nothing left that may be dispatched: in flight,
        # settled, or waiting on a person. The truthful current standing is
        # the answer, and no claim is touched to produce it.
        return _outcome(config, plan)

    errors: dict[int, str] = {}
    if any(state == UNRESERVED for state in states.values()):
        # A confirmed execution that never finished its cleanup or its
        # reservation. Everything it depends on is read again first: the
        # receipt says what was agreed to, not that the world stood still.
        unsent = [
            child for child in plan.children if states[child.index] == UNRESERVED
        ]
        revalidated = _revalidate_children(
            config,
            plan,
            unsent,
            provider_env=provider_env,
            provider_runner=provider_runner,
            provider_which=provider_which,
        )
        _execute_discards(config, plan)
        try:
            _reserve_batch(config, plan)
        except JankiError as exc:
            raise ExtractionDispatchError(exc, phase="authorization") from exc
        states = _child_states(config, plan)
    else:
        # Saved answers first, and on their own. Recovering one needs no
        # prompt file and no login, so a machine that has since logged out or
        # had its prompts edited must not be able to strand an answer that was
        # already paid for behind a preflight for a sibling that never sent.
        recovered = any(state == "result_captured" for state in states.values())
        if recovered:
            _run_children(
                config,
                plan,
                {},
                states,
                progress=progress,
                provider_env=provider_env,
                provider_runner=provider_runner,
                provider_which=provider_which,
                provider_spawn=provider_spawn,
                errors=errors,
            )
            states = _child_states(config, plan)
        unsent = [
            child for child in plan.children if states[child.index] == "authorized"
        ]
        if not unsent:
            return _outcome(config, plan, errors=errors)
        try:
            revalidated = _revalidate_children(
                config,
                plan,
                unsent,
                provider_env=provider_env,
                provider_runner=provider_runner,
                provider_which=provider_which,
            )
        except ExtractionDispatchError as exc:
            if not recovered:
                # Nothing was recovered either, so there is no partial result
                # to report and the refusal is the whole answer.
                raise
            # The preflight covers every unsent child before any of them is
            # sent, so one refusal means none of them go. Saying so per child
            # keeps the recovered work visible instead of raising it away.
            for child in unsent:
                errors[child.index] = str(exc)
            return _outcome(config, plan, errors=errors)
    return _run_children(
        config,
        plan,
        revalidated,
        states,
        progress=progress,
        provider_env=provider_env,
        provider_runner=provider_runner,
        provider_which=provider_which,
        provider_spawn=provider_spawn,
        errors=errors,
    )


#: Why one child cannot be retried, in the words a person needs to act on.
_RETRY_REFUSALS: dict[str, str] = {
    "committed": (
        "committed its proposals already, so retrying it would pay for work "
        "janki has. If its grammar half is missing, that is bookkeeping to "
        "finish rather than a call to make again"
    ),
    "dispatching": "may be in flight right now; a retry would risk a second charge",
    "running": "is with the provider right now; a retry would risk a second charge",
    "authorized": (
        "still holds unused authority and has not been sent, so resume this "
        "batch rather than retrying it"
    ),
}


def plan_extraction_batch_retry(
    config: ProjectConfig,
    batch_id: str,
    child_indices: Sequence[int],
    *,
    concurrency_limit: int | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
) -> ExtractionBatchPlan:
    """Plan a second attempt at named failed children, and name what it discards.

    Retrying is two decisions bound into one confirmation: retire these exact
    journal entries, and send these exact requests again under fresh
    identities. The old snapshots ride in `discards` so the surface can say
    what is being thrown away — including a captured reply somebody paid for,
    and including an outcome nobody could determine, both disclosed rather
    than quietly deleted.

    An `outcome_unknown` child is retryable *because* it is terminal: the
    journal's `end` already turned a live call into that state, and nothing
    can end it again. What the plan owes a person there is the truth that the
    old call may have been billed and that confirming this throws its record
    away. The old operation is never redispatched; the new call is a new
    authority with a new id.

    Nothing is retired, reserved or written here.
    """
    plan = _load_batch_plan(config, batch_id)
    wanted = tuple(int(index) for index in child_indices)
    if not wanted:
        raise operations.OperationError("Name at least one source to retry.")
    if len(set(wanted)) != len(wanted):
        raise operations.OperationError("Each retried source is named once.")

    journal = operations.OperationJournal.load(config.operations_file)
    batch = journal.batches.get(batch_id)
    if batch is None:
        raise operations.OperationError(
            f"Batch {batch_id} was never reserved, so there is nothing to retry; "
            "its manifest describes a run nothing was spent on."
        )
    members = frozenset(batch.child_operation_ids)

    selected: list[ExtractionBatchChild] = []
    discards: list[operations.Operation] = []
    for index in wanted:
        child = plan.child(index)
        held = (
            journal.operations.get(child.operation_id)
            if child.operation_id in members
            else None
        )
        if held is None:
            raise operations.OperationError(
                f"Source {index} ({child.source.name}) was retired: this batch "
                "still names it, but its journal entry is gone, so janki cannot "
                "say what its call cost. A retired member is never sent again "
                "under this batch; check its staging file instead."
            )
        refusal = _RETRY_REFUSALS.get(held.state)
        if refusal:
            raise operations.OperationError(
                f"Source {index} ({child.source.name}) {refusal}."
            )
        selected.append(child)
        discards.append(held)

    retry = plan_extraction_batch(
        config,
        [child.source for child in selected],
        mode=selected[0].expectation.mode,
        model=plan.model,
        scope_id=plan.scope_id,
        destination_deck=plan.destination_deck,
        concurrency_limit=(
            plan.concurrency_limit if concurrency_limit is None else concurrency_limit
        ),
        force=any(
            child.expectation.replacement_confirmed for child in selected
        ),
        # Carried only when every retried part has it: an invented empty
        # ancestry would be a claim about where a document came from.
        lineage=(
            tuple(child.lineage for child in selected)  # type: ignore[misc]
            if all(child.lineage is not None for child in selected)
            else ()
        ),
        provider_env=provider_env,
        provider_runner=provider_runner,
        provider_which=provider_which,
    )
    return replace(retry, retry_of=batch_id, discards=tuple(discards))


def extraction_batch_status(
    config: ProjectConfig, batch_id: str
) -> ExtractionBatchOutcome:
    """Where one batch stands right now. Reads only."""
    return _outcome(config, _load_batch_plan(config, batch_id))


def list_extraction_batches(
    config: ProjectConfig,
) -> tuple[ExtractionBatchOutcome, ...]:
    """Every recorded batch, oldest manifest first. Reads only."""
    directory = config.operations_file.parent / BATCH_DIR_NAME
    if not directory.is_dir():
        return ()
    found: list[tuple[float, str, ExtractionBatchOutcome]] = []
    for path in sorted(directory.glob("*.json")):
        batch_id = path.stem
        if not _valid_batch_id(batch_id):
            continue
        try:
            outcome = extraction_batch_status(config, batch_id)
            stamp = path.stat().st_mtime
        except (JankiError, OSError):
            # One unreadable manifest is not a reason to hide the others; the
            # batch it describes is still reachable by name, which is where
            # its exact error belongs.
            continue
        found.append((stamp, batch_id, outcome))
    return tuple(outcome for _stamp, _id, outcome in sorted(found, key=lambda row: row[:2]))


def _records_text(records: Sequence[VocabularyRecord]) -> str:
    """Serialize a prospective collection exactly as the canonical store holds it."""
    return (
        json.dumps(
            [record.to_dict() for record in sorted(records, key=lambda item: item.id)],
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _deck_include_overlay(
    deck_path: Path, new_record_ids: Sequence[str]
) -> card_preview.ProposedText | None:
    """Widen an explicit deck selection to the proposals, for display only.

    A deck that names its cards by id would draw none of these, which would
    make the preview truthfully empty and useless. The overlay lives in the
    preview's scratch mirror; the deck file on disk is not touched.
    """
    try:
        text = read_bytes_bound(deck_path).decode("utf-8")
    except (JankiError, OSError, UnicodeError) as exc:
        raise staging.StagingError(f"Could not read {deck_path}: {exc}") from exc
    raw = _parse_structured_text(deck_path, text)
    if not isinstance(raw, dict):
        raise staging.StagingError(f"Deck file must contain a mapping: {deck_path}")
    deck_config = raw.get("deck")
    if not isinstance(deck_config, dict):
        return None
    include = deck_config.get("include_ids")
    if not include:
        return None
    if deck_path.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise staging.StagingError(
            f"{deck_path.name} selects its cards by id, and janki can only "
            "render that selection widened for a JSON or YAML deck file."
        )
    known = {str(value) for value in include}
    widened = dict(raw)
    widened["deck"] = dict(deck_config) | {
        "include_ids": [*[str(value) for value in include],
                        *[value for value in new_record_ids if value not in known]]
    }
    # JSON is valid YAML, so one serializer covers both readers the preview
    # may use for this overlay.
    return card_preview.ProposedText(
        path=deck_path,
        text=json.dumps(widened, ensure_ascii=False, indent=2) + "\n",
    )


@dataclass(frozen=True, slots=True)
class _Proposal:
    """One staged record, and which child proposed it."""

    index: int
    source_name: str
    record: VocabularyRecord


def _conflict_sentences(group: Sequence[_Proposal]) -> str:
    """Say exactly how two children's proposals for one identity differ.

    Field names and each source's literal value, compared structurally. This
    never decides which is right: two pages of one worksheet can print the
    same word with different notes, and choosing between them is a judgement
    about Japanese that belongs to a person.
    """
    first = group[0]
    reference = first.record.to_dict()
    differences: list[str] = []
    for other in group[1:]:
        candidate = other.record.to_dict()
        for field in sorted(set(reference) | set(candidate)):
            left = reference.get(field)
            right = candidate.get(field)
            if left == right:
                continue
            differences.append(
                f"{field} ({first.source_name}: "
                f"{json.dumps(left, ensure_ascii=False)}; "
                f"{other.source_name}: {json.dumps(right, ensure_ascii=False)})"
            )
    named = ", ".join(
        f"source {item.index} ({item.source_name})" for item in group
    )
    if not differences:
        return (
            f"{first.record.id} was proposed by {named} with identical fields. "
            f"The card shown is source {first.index}'s proposal, standing for "
            "both; neither has been accepted."
        )
    return (
        f"{first.record.id} was proposed by {named}, differing on "
        + "; ".join(differences)
        + f". The card shown is source {first.index}'s proposal as a "
        "representative; nothing has been merged or accepted."
    )


def _scratch_deck_overlay(
    config: ProjectConfig, batch_id: str, record_ids: Sequence[str]
) -> tuple[Path, card_preview.ProposedText]:
    """A proposal-only deck that exists for the length of one render.

    Reviewing saved successes must not require choosing a destination first,
    and a review is not an assignment: this deck is never written, never
    configured and grants nothing. It selects exactly the proposed identities
    by id, reads the canonical collection like every word deck, and takes the
    project's own card defaults rather than inventing directions.
    """
    deck_path = Path(
        os.path.abspath(config.deck_dir / f"batch-{batch_id}.preview.yaml")
    )
    definition = {
        "deck": {
            "name": f"Batch {batch_id[:8]} proposals",
            "source": os.path.relpath(
                Path(os.path.realpath(config.normalized_file)),
                Path(os.path.realpath(config.deck_dir)),
            ),
            "include_ids": list(record_ids),
        }
    }
    # JSON is valid YAML, so the deck reader parses this overlay exactly.
    return deck_path, card_preview.ProposedText(
        path=deck_path,
        text=json.dumps(definition, ensure_ascii=False, indent=2) + "\n",
    )


def _assigned_for_destination(
    config: ProjectConfig,
    deck_path: Path,
    records: Sequence[VocabularyRecord],
) -> list[VocabularyRecord]:
    """What that destination would really own, by the assignment service.

    ``assigned_record`` and never ``prospective_record``: the second merges
    canonical content back over the proposal, which is exactly the content a
    reviewer is here to look at. The scoped identity and the ownership tags
    are assignment's rules, read rather than re-derived.
    """
    stem = deck_path.stem
    try:
        matrix = assignment.plan_deck_assignments(
            config, list(records), destination_stems=[stem]
        )
    except JankiError as exc:
        raise staging.StagingError(
            f"These proposals cannot be shown in {deck_path.name}: {exc}"
        ) from exc
    assigned: list[VocabularyRecord] = []
    for attempts, record in zip(matrix, records, strict=True):
        attempt = attempts[0]
        if attempt.plan is None:
            raise staging.StagingError(
                f"{record.id} cannot be shown in {deck_path.name}: "
                f"{attempt.refusal or 'that destination refused it'}"
            )
        assigned.append(attempt.plan.assigned_record)
    return assigned


def _require_own_staging(
    child: ExtractionBatchChild, meta: Mapping[str, Any]
) -> None:
    """Prove this document is the one that child's answer wrote.

    A path says where a file is, not what put it there. The document names the
    source it was extracted from and carries the exact request it came from,
    so both are compared before its rows are attributed to this child.

    This cannot separate two answers to the *same* request for the same
    source: a re-read that produced identical provenance is indistinguishable
    here, and staging carries nothing else that would tell them apart. That
    limit is deliberate — the alternative is a new store or an operation id in
    the staging document, and neither belongs to a review file.
    """
    named = str(meta.get("source_file") or "")
    if named != child.source.name:
        raise staging.StagingError(
            f"The staging document at {child.staging_path} was written for "
            f"{named or 'an unnamed source'}, not for {child.source.name}, so "
            "it is not source "
            f"{child.index}'s proposal."
        )
    stored = json.loads(
        json.dumps(dict(meta.get("prompt_provenance") or {}), sort_keys=True)
    )
    if stored != json.loads(json.dumps(dict(child.provenance), sort_keys=True)):
        raise staging.StagingError(
            f"The staging document at {child.staging_path} came from a "
            f"different request than source {child.index} "
            f"({child.source.name}) made, so it is not that child's proposal."
        )


def render_extraction_batch_preview(
    config: ProjectConfig,
    batch_id: str,
    *,
    deck_path: Path | None = None,
) -> ExtractionBatchPreview:
    """Draw a whole batch's saved proposals as real cards, and write nothing.

    Every child's staging document is read exactly as that child wrote it —
    there is no combined staging file, no merged metadata and no second
    provenance. The projection replaces the selected identities in an
    in-memory copy of the collection, so what renders is the *proposed*
    content rather than whatever the collection already says about those
    words; every unrelated canonical record is preserved untouched.

    With a destination deck, the identities and ownership tags come from the
    assignment service, because a standalone deck holds its own scoped copies
    and those rules are its own. With no destination, the proposals are drawn
    in a scratch deck that exists only for this render: reviewing saved work
    must not force a filing decision first, and nothing here assigns anything.

    Two children proposing one identity is reported in `conflicts` and shown
    as one labelled representative card. Nothing is merged, and no proposal is
    marked accepted.
    """
    plan = _load_batch_plan(config, batch_id)
    states = _child_states(config, plan)

    grouped: dict[str, list[_Proposal]] = {}
    drawn: list[int] = []
    for child in plan.children:
        # Only what the journal says this batch actually produced. A staging
        # file at a child's path may be an older review a forced re-read has
        # not replaced yet, and a child that never sent has proposed nothing —
        # drawing either as this batch's work would be a lie a reviewer acts
        # on. `bookkeeping_complete` is deliberately not required: a committed
        # answer whose pattern write failed still saved its cards.
        if states[child.index] != "committed":
            continue
        path = child.staging_path
        if not path.exists():
            continue
        try:
            text = read_bytes_bound(path).decode("utf-8")
            records, meta = staging.read_staging_text(text, source=str(path))
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise staging.StagingError(
                f"Could not read the proposals for source {child.index} "
                f"({child.source.name}): {exc}"
            ) from exc
        _require_own_staging(child, meta)
        drawn.append(child.index)
        for record in records:
            grouped.setdefault(record.id, []).append(
                _Proposal(
                    index=child.index,
                    source_name=child.source.name,
                    record=record,
                )
            )
    if not drawn:
        raise staging.StagingError(
            f"Batch {batch_id} has no saved proposals to draw yet."
        )

    # First occurrence wins the card, and says so. Choosing between two
    # proposed readings is not a decision this module is allowed to make.
    representatives = [group[0].record for group in grouped.values()]
    conflicts = tuple(
        _conflict_sentences(group)
        for _record_id, group in sorted(grouped.items())
        if len(group) > 1
    )

    target_deck = deck_path if deck_path is not None else plan.destination_deck
    overlay: list[card_preview.ProposedText] = []
    if target_deck is None:
        rendered = list(representatives)
        target_deck, scratch = _scratch_deck_overlay(
            config, batch_id, [record.id for record in rendered]
        )
        overlay.append(scratch)
    else:
        target_deck = Path(os.path.realpath(target_deck))
        rendered = _assigned_for_destination(config, target_deck, representatives)

    existing = (
        load_records(config.normalized_file)
        if config.normalized_file.exists()
        else []
    )
    known = {record.id for record in existing}
    # The proposal replaces what the collection holds for that identity, so
    # its new meanings, examples and notes are what a reviewer sees. Every
    # other canonical record is carried through exactly as it is.
    projected = {record.id: record for record in existing}
    for record in rendered:
        projected[record.id] = record
    new_record_ids = tuple(
        record.id for record in rendered if record.id not in known
    )
    canonical = Path(os.path.realpath(config.normalized_file))
    overlay.insert(
        0,
        card_preview.ProposedText(
            path=canonical, text=_records_text(list(projected.values()))
        ),
    )
    if deck_path is not None or plan.destination_deck is not None:
        # A real deck that names its cards by id would draw none of these.
        # The scratch deck already selects exactly the proposals.
        deck_overlay = _deck_include_overlay(
            target_deck, [record.id for record in rendered]
        )
        if deck_overlay is not None:
            overlay.append(deck_overlay)

    subtitle = "Proposed — nothing is promoted yet"
    if conflicts:
        subtitle += (
            f"; {len(conflicts)} identity proposed twice, showing one source's "
            "card as a representative"
            if len(conflicts) == 1
            else f"; {len(conflicts)} identities proposed twice, showing one "
            "source's card for each as a representative"
        )
    preview = card_preview.render_card_preview(
        config,
        target_deck,
        proposed=tuple(overlay),
        new_record_ids=new_record_ids,
        scope_record_ids=tuple(record.id for record in rendered),
        subtitle=subtitle,
    )
    return ExtractionBatchPreview(
        preview=preview,
        conflicts=conflicts,
        child_indices=tuple(drawn),
    )
