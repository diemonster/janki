"""What an extraction run would send, decided before anybody consents to it.

`WORKBENCH_PLAN.md` W1.1b, for W3. `command_extract` did all of this inline
between reading the config and asking the question, which meant the only way to
find out what a run would send was to start one.

W3 needs it as a value: *"Adding a file to the corpus and sending it to a model
are two separate actions, always"*, and the consent button has to name exactly
what leaves the computer — this file, this model, a paid API call. A dialog
cannot render any of that from a function that asks and dispatches in one
breath.

So the run splits where the consent does. `plan_extraction` resolves every
target, every fingerprint and every refusal that can be known without paying —
a staging collision, an unreadable pattern store — and returns them. Whether to
spend the money is then the caller's decision, made on a value it can show
someone, and the same value is what the paid half consumes.

**This is not a pure preview and must not be mistaken for one.** By the time it
is called the inputs are already copied into `data/inbox/`, because reading a
file is how its collision and its fingerprint are known at all. Nothing here
contacts a provider, journals an operation, or writes a staging file; the
inbox copy has already happened, and `kept_in_inbox` is what a caller tells
somebody who then declines.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from japanese_anki import (
    claude_client,
    extract,
    inputs,
    operations,
    patterns,
    prompts,
    staging,
)
from japanese_anki.application import revision_provider
from japanese_anki.application.journey import (
    GRAMMAR_NEEDS_REVIEW,
    GRAMMAR_REVIEWED,
    source_journeys,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.credential_safety import redact_environment_credentials
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import deck_selection
from japanese_anki.identifiers import record_scope_id
from japanese_anki.inputs import PreparedInput
from japanese_anki.io import (
    _parse_structured_text,
    exclusive_path_lock,
    load_records,
    prepare_bound_directory,
    read_bytes_bound,
)
from japanese_anki.models import VocabularyRecord
from japanese_anki.staging import (
    CANDIDATE_ACCOUNTING_KEY,
    CAPTURE_RECOVERY_KEY,
    new_review_run_id,
    write_staging_under_lock,
)

__all__ = [
    "ANSWER_EMPTY",
    "ANSWER_SAVED",
    "ANSWER_UNAVAILABLE",
    "FORGOTTEN",
    "OUTCOME_UNKNOWN",
    "DispatchFailure",
    "ExtractionCompletionError",
    "ExtractionConsent",
    "ExtractionDispatchError",
    "ExtractionDispatchExpectation",
    "ExtractionOutcome",
    "ExtractionPlan",
    "ExtractionRevision",
    "ExtractionTarget",
    "RevalidatedExtraction",
    "authorize_dispatch",
    "busy_refusal",
    "capture_hook",
    "classify_dispatch_failure",
    "complete_extraction",
    "describe_extraction",
    "dispatch_extraction",
    "durable_inbox_root",
    "extraction_capture_envelope",
    "extraction_capture_parts",
    "extraction_provider",
    "extraction_provider_plan",
    "prepare_extraction_transport",
    "recover_extraction",
    "require_current_destination",
    "recover_extraction_from_capture",
    "revalidate_extraction_request",
    "run_extraction_call",
    "run_extraction_lifecycle",
    "extraction_replacement_revision",
    "plan_corpus_extraction",
    "plan_extraction",
    "settle_dispatch",
]


@dataclass(frozen=True, slots=True)
class ExtractionTarget:
    """One input, and everything decided about it before the call."""

    item: PreparedInput
    #: The staging file this input's proposals would be written to.
    staging_path: Path
    #: The shared pattern store receiving this source's grammar proposals.
    patterns_path: Path
    source_sha256: str
    #: The exact request identity, including `request_fingerprint`. Journalled
    #: before dispatch so a crash between sending and parsing can still say
    #: which call was made — which is only true while it describes the request
    #: that is actually sent, so the dispatch reads it from here rather than
    #: rebuilding one of its own.
    provenance: Mapping[str, Any]
    #: A staging file this input would overwrite. Empty unless `force` was
    #: given: without it, planning refuses instead. W3 confirms a replacement
    #: separately and has to name the review it invalidates.
    replaces: Path | None = None
    #: The immutable request the shared provider registry planned for this
    #: source, on the subscription transport. Carried rather than re-planned
    #: at dispatch: preparing a second request would authenticate one call and
    #: send another. ``None`` on the explicit Anthropic API path, which builds
    #: its request inside the client.
    provider_plan: Any | None = None

    @property
    def name(self) -> str:
        """The permanent filename, which is what a person recognizes."""
        return self.item.origin_path.name

    @property
    def provider(self) -> str:
        """Which transport this request was planned for."""
        if self.provider_plan is None:
            return revision_provider.ANTHROPIC_API_PROVIDER
        return str(self.provider_plan.provider)

    @property
    def billing_display(self) -> str:
        """Who pays for this call, in the provider's own durable words."""
        if self.provider_plan is None:
            return revision_provider.billing_display(
                revision_provider.ANTHROPIC_API_PROVIDER,
                "anthropic-platform-api",
                {"auth_method": "environment-api-key"},
            )
        return str(self.provider_plan.billing_display)


@dataclass(frozen=True, slots=True)
class ExtractionRevision:
    """The exact review a checked replacement is allowed to destroy.

    The staging hash includes every byte, including hand-written comments.
    The pattern hash covers only this source's raw JSON entry so another
    lesson's review does not make an otherwise current consent stale.
    """

    staging_sha256: str | None
    pattern_entry_sha256: str | None
    pattern_reviewed: bool | None
    pattern_has_patterns: bool | None


@dataclass(frozen=True, slots=True)
class ExtractionDispatchExpectation:
    """The exact rendered extraction request one owner confirmed.

    This is deliberately only an expectation, never the plan that will be
    sent.  A surface binds these immutable values into its one-use consent
    capability; :func:`dispatch_extraction` re-plans from ``source`` at click
    time and refuses any difference before authorizing a paid call.

    ``replacement_confirmed`` is the sole source of ``force``.  The presence
    of ``replacement_revision`` says what the rendered page offered to
    destroy, not that the owner agreed to destroy it.
    """

    source: Path
    #: The transport the owner confirmed. Bound separately from the request
    #: fingerprint because "who is billed for this" is part of what they
    #: agreed to, and a config edit between render and click can change it.
    provider: str
    model: str
    mode: str | None
    source_sha256: str
    request_fingerprint: str
    replacement_revision: ExtractionRevision | None
    replacement_confirmed: bool
    staging_path: Path
    patterns_path: Path
    operations_path: Path
    #: The record namespace this request's known-word suppression was computed
    #: for: one deck's scope, or ``""`` for the shared collection.
    scope_id: str = ""
    #: The deck whose scope that is, when a surface let an owner choose one.
    #: A local consent binding rather than a destination this run writes to:
    #: extraction stages proposals, and assigning them to a deck is a separate
    #: decision. Optional, because a service caller may extract for a scope
    #: without a deck file in hand.
    destination_deck: Path | None = None
    #: That deck file's exact bytes when the batch was rendered. Bound
    #: separately from the request fingerprint because a re-scoped or edited
    #: destination can leave the prompt — and so the fingerprint — identical.
    destination_deck_sha256: str = ""
    #: The complete frozen layout this confirmation was rendered over, for a
    #: layout-bound request. The whole object rather than its identity pair:
    #: an identity resolved against today's mutable binding could name a
    #: different set of columns than the one somebody agreed to send.
    table_layout: extract.TableLayout | None = None

    def __post_init__(self) -> None:
        # These paths are authority and destination bindings, not display
        # conveniences. Resolve them when the expectation is minted so a
        # later config reload cannot redirect the confirmed action through a
        # different lexical spelling or symlink target.
        for field_name in ("staging_path", "patterns_path", "operations_path"):
            object.__setattr__(
                self,
                field_name,
                Path(os.path.realpath(getattr(self, field_name))),
            )
        if self.destination_deck is not None:
            object.__setattr__(
                self,
                "destination_deck",
                Path(os.path.realpath(self.destination_deck)),
            )
        # Mode and layout pair here as well as in the provenance, so a manifest
        # read back with one half of the pair missing refuses at the value
        # rather than at the request it would have rebuilt.
        if (self.mode == extract.LAYOUT_MODE) != (self.table_layout is not None):
            raise extract.ExtractError(
                f"A {extract.LAYOUT_MODE} request carries exactly one bound "
                f"layout and every other mode carries none; this one names "
                f"{self.mode or 'auto'} mode with "
                + ("a layout." if self.table_layout is not None else "no layout.")
                ,
                code="extract-layout-misaligned",
            )


@dataclass(frozen=True, slots=True)
class _PatternStoreSnapshot:
    store: dict[str, patterns.PatternSet]
    wire: bytes
    entry_sha256: str | None
    absent: bool


@contextlib.contextmanager
def _replacement_locks(staging_path: Path, patterns_path: Path):
    """Lock both replacement targets in the review subsystem's order."""
    real = {Path(os.path.realpath(path)) for path in (staging_path, patterns_path)}
    if len(real) != 2:
        raise operations.OperationError(
            "The staging file and pattern store must be distinct replacement targets"
        )
    with contextlib.ExitStack() as stack:
        for path in sorted(real, key=os.fspath):
            stack.enter_context(exclusive_path_lock(path))
        yield


def _staging_revision(path: Path) -> str | None:
    if path.is_symlink():
        raise staging.StagingError(f"Refusing a symlinked staging replacement: {path}")
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError:
        return None
    except (JankiError, OSError) as exc:
        raise staging.StagingError(f"Could not read staging review {path}: {exc}") from exc
    return hashlib.sha256(wire).hexdigest()


def _pattern_store_snapshot(path: Path, source_name: str) -> _PatternStoreSnapshot:
    if path.is_symlink():
        raise patterns.PatternError(f"Refusing a symlinked pattern store: {path}")
    if path.parent.is_symlink() or (
        path.parent.exists() and not path.parent.is_dir()
    ):
        raise patterns.PatternError(
            f"Refusing non-directory pattern-store parent: {path.parent}"
        )
    try:
        wire = read_bytes_bound(path)
    except FileNotFoundError:
        return _PatternStoreSnapshot({}, b"", None, True)
    except (JankiError, OSError) as exc:
        raise patterns.PatternError(f"Could not read {path}: {exc}") from exc
    try:
        text = wire.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise patterns.PatternError(f"Could not read {path}: {exc}") from exc
    store = patterns.load_store_text(text, source=str(path))
    return _PatternStoreSnapshot(
        store=store,
        wire=wire,
        entry_sha256=patterns.entry_fingerprint(
            text,
            source_name,
            source=str(path),
        ),
        absent=False,
    )


def _replacement_state_under_lock(
    config: ProjectConfig,
    target: ExtractionTarget,
) -> tuple[ExtractionRevision, _PatternStoreSnapshot]:
    pattern_snapshot = _pattern_store_snapshot(config.patterns_file, target.name)
    stored_patterns = pattern_snapshot.store.get(target.name)
    return (
        ExtractionRevision(
            staging_sha256=_staging_revision(target.staging_path),
            pattern_entry_sha256=pattern_snapshot.entry_sha256,
            pattern_reviewed=(
                stored_patterns.reviewed if stored_patterns is not None else None
            ),
            pattern_has_patterns=(
                bool(stored_patterns.patterns) if stored_patterns is not None else None
            ),
        ),
        pattern_snapshot,
    )


def extraction_replacement_revision(
    config: ProjectConfig,
    target: ExtractionTarget,
) -> ExtractionRevision | None:
    """Capture the exact card and grammar review a forced plan would replace."""
    if target.replaces is None:
        return None
    with _replacement_locks(target.staging_path, config.patterns_file):
        revision, _snapshot = _replacement_state_under_lock(config, target)
    if revision.staging_sha256 is None:
        raise staging.StagingError(
            f"The review for {target.name} changed while the consent page was loading"
        )
    return revision


@dataclass(frozen=True, slots=True)
class ExtractionPlan:
    """What a run would send, and to whom, before anyone agrees to it."""

    model: str
    mode: str | None
    targets: tuple[ExtractionTarget, ...]
    #: The instructions every input in this run is extracted under. Read once
    #: for the run: re-reading per file would let a mid-run edit split one
    #: command across two prompts.
    style_guide: str
    system: str
    #: Expressions the collection already has, sent only in prose mode — a
    #: table is transcribed row by row, and telling the model to skip rows
    #: would put holes in a faithful transcription.
    #:
    #: Sorted, and that is not cosmetic: this text goes into the prompt the
    #: request fingerprint is computed over, so an unordered set would give the
    #: same corpus a different request identity on every run — and that
    #: identity is what the journal records and a retry compares.
    skip_list: tuple[str, ...]
    known: frozenset[str]
    #: The record namespace both of those were computed in: one deck's scope,
    #: or ``""`` for the shared collection.
    scope_id: str = ""
    #: How this run reaches Claude: the owner's subscription by default, or
    #: the metered API when a project wrote that down.
    provider: str = revision_provider.CLAUDE_CODE_PROVIDER

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.targets)

    @property
    def kept_in_inbox(self) -> tuple[Path, ...]:
        """Files *this* run copied into the inbox, whether or not it sends them.

        The half a refusal message forgets. The copy happens before consent, so
        "nothing was sent" alone reads as "nothing happened" while a private
        document sits staged for the next `git add`.

        `item.copied`, not "is this path under the inbox": every branch of the
        copy returns a path under the inbox, including the ones that copy
        nothing, so a containment test announces files the run never touched.
        """
        return tuple(
            target.item.origin_path for target in self.targets if target.item.copied
        )


#: The transports an extraction may use. The subscription is the default; the
#: metered API is reached only by writing it down. Nothing in this module may
#: move between them because one was unavailable — a missing CLI, a logged-out
#: login or a raised exception is a refusal, never a switch to the paid key.
EXTRACTION_PROVIDERS = (
    revision_provider.CLAUDE_CODE_PROVIDER,
    revision_provider.ANTHROPIC_API_PROVIDER,
)

#: The durable capture wrapper is the journal's, so one decoder reads it.
EXTRACTION_CAPTURE_VERSION = operations.CAPTURED_REPLY_VERSION
EXTRACTION_CAPTURE_KEY = operations.CAPTURED_REPLY_KEY


def extraction_provider(config: ProjectConfig, provider: str | None = None) -> str:
    """Which transport this run uses: the configured one unless one is named.

    An explicit argument is a caller stating a transport, not inferring one.
    In particular the presence of a test client says nothing: a fake object
    must not be able to redirect a real request onto a billed account.
    """
    chosen = provider if provider else config.extract_provider
    if chosen not in EXTRACTION_PROVIDERS:
        choices = ", ".join(EXTRACTION_PROVIDERS)
        raise extract.ExtractError(
            f"Unknown extraction provider {chosen!r}; choose one of: {choices}.",
            code="extract-provider-unknown",
        )
    return chosen


def extraction_provider_plan(
    item: PreparedInput,
    *,
    provider: str,
    model: str,
    style_guide: str,
    system: str,
    known: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    runner: Callable[..., Any] = subprocess.run,
    which: Callable[..., str | None] = shutil.which,
    layout: extract.TableLayout | None = None,
) -> Any | None:
    """The exact provider request for one source, or None on the API path.

    The source rides as an input block, so the document is part of the
    request identity the owner confirms and the journal records. The explicit
    Anthropic API path keeps building its request inside the client, which is
    what its existing recovery and tests describe.

    ``layout`` is this one input's bound layout. It reaches the planned user
    turn here, so the saved manifest and channels describe the request that is
    actually sent rather than one that merely resembles it.
    """
    if provider == revision_provider.ANTHROPIC_API_PROVIDER:
        return None
    return revision_provider.plan_provider(
        provider,
        model=model,
        style_guide=style_guide,
        task_template=system,
        system_blocks=tuple(claude_client.system_blocks(style_guide, system)),
        user_turn=extract.prompt_for(item.origin_path.name, known, layout=layout),
        schema=extract.candidate_schema(),
        effort=claude_client.effort_for(model),
        input_blocks=(item.content_block(),),
        env=env,
        runner=runner,
        which=which,
    )


@dataclass(frozen=True, slots=True)
class ExtractionTransport:
    """A proved way to make one extraction call, before any authority exists."""

    provider: str
    #: The prepared subscription handle: a resolved executable and a checked
    #: login, both proved before the journal can say money may have gone.
    prepared: Any | None = None
    #: The Anthropic API client, on the explicit API path only.
    client: Any | None = None

    @property
    def billing_display(self) -> str:
        if self.prepared is not None:
            return str(self.prepared.plan.billing_display)
        return revision_provider.billing_display(
            revision_provider.ANTHROPIC_API_PROVIDER,
            "anthropic-platform-api",
            {"auth_method": "environment-api-key"},
        )


def prepare_extraction_transport(
    plan: ExtractionPlan,
    target: ExtractionTarget | None = None,
    *,
    client: Any | None = None,
    env: Mapping[str, str] | None = None,
    runner: Callable[..., Any] = subprocess.run,
    which: Callable[..., str | None] = shutil.which,
) -> ExtractionTransport:
    """Prove the selected transport can make this call, before authorizing it.

    Preparation precedes the journal on both paths, for the same reason: a
    local credential or login failure must not leave an entry saying a call
    may have been billed.
    """
    chosen = plan.provider
    if chosen not in EXTRACTION_PROVIDERS:
        choices = ", ".join(EXTRACTION_PROVIDERS)
        raise extract.ExtractError(
            f"Unknown extraction provider {chosen!r}; choose one of: {choices}.",
            code="extract-provider-unknown",
        )
    if chosen == revision_provider.ANTHROPIC_API_PROVIDER:
        return ExtractionTransport(
            provider=chosen,
            client=client if client is not None else claude_client.prepare_paid_client(),
        )
    if target is None:
        targets = plan.targets
        if len(targets) != 1:
            raise extract.ExtractError(
                "A subscription extraction is prepared one source at a time.",
                code="extract-provider-unprepared",
            )
        target = targets[0]
    if target.provider_plan is None:
        raise extract.ExtractError(
            f"{target.name}: no subscription request was planned for this source.",
            code="extract-provider-unprepared",
        )
    prepared = revision_provider.provider_for(chosen).prepare(
        target.provider_plan,
        env=env,
        runner=runner,
        which=which,
    )
    return ExtractionTransport(provider=chosen, prepared=prepared)


def run_extraction_call(
    transport: ExtractionTransport,
    plan: ExtractionPlan,
    target: ExtractionTarget,
    *,
    capture: Callable[[Any], None],
    frame: Callable[[str], None] | None = None,
    spawn: Any = subprocess.Popen,
) -> extract.ExtractionResult:
    """Make the one content call this target was planned for.

    The only place either transport is driven. The subscription sends the
    bytes it planned — not a request built again here — and both paths hand
    their answer to the same response adapter, so there is exactly one set of
    rules about what a complete extraction is.
    """
    if transport.provider == revision_provider.ANTHROPIC_API_PROVIDER:
        return extract.extract_candidates(
            target.item,
            model=plan.model,
            style_guide=plan.style_guide,
            system=plan.system,
            mode=plan.mode,
            known=plan.skip_list,
            client=transport.client,
            capture=capture,
            provenance=dict(target.provenance),
            # This transport builds its own turn inside the client, so the
            # layout has to travel with it. Read back from the target's own
            # saved provenance, never from a job document: the request that is
            # sent and the request that was recorded are then the same object.
            layout=extract.layout_from_provenance(target.provenance),
        )
    if transport.prepared is None:
        raise extract.ExtractError(
            f"{target.name}: the subscription transport was not prepared.",
            code="extract-provider-unprepared",
        )
    call = revision_provider.provider_for(transport.provider).dispatch(
        transport.prepared,
        capture=capture,
        spawn=spawn,
        frame=frame,
    )
    return extract.extraction_result_from_call(
        call,
        target.item,
        model=plan.model,
        mode=plan.mode,
        provenance=dict(target.provenance),
    )


def extraction_capture_envelope(
    provenance: Mapping[str, Any], raw: bytes
) -> bytes:
    """Wrap one paid subscription reply with the request that produced it.

    Written before anything reads the bytes, because the interval where an
    answer is worth most is the one before it has been understood. The reply
    is kept exactly, base64 so no decoder touches it, beside the manifest and
    channels needed to rebuild the same request. No new store: this is the
    operation's own captured artifact.
    """
    return json.dumps(
        {
            operations.CAPTURED_REPLY_KEY: operations.CAPTURED_REPLY_VERSION,
            "provenance": json.loads(json.dumps(provenance, ensure_ascii=False)),
            "reply_base64": base64.b64encode(raw).decode("ascii"),
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")


def extraction_capture_parts(
    payload: bytes,
) -> tuple[bytes, Mapping[str, Any] | None]:
    """The exact reply inside a captured artifact, and its request if present.

    One decoder, shared with the journal's own readers: anything that is not
    one of these wrappers is returned unchanged, which is what keeps every
    captured Anthropic API reply readable by the same code.
    """
    try:
        return operations.unwrap_captured_reply(payload)
    except operations.OperationError as exc:
        raise extract.ExtractError(
            str(exc), code="extract-capture-unreadable"
        ) from exc


def provider_plan_from_provenance(provenance: Mapping[str, Any]) -> Any:
    """Rebuild the exact planned request from what was saved beside a reply.

    Purely: no executable lookup, no login probe, no client, and no reading of
    today's prompt files. What was sent is read back from the saved manifest
    and channels, so an edited prompt cannot change the request this says was
    made.

    The response contract is a different matter. Decoding still runs through
    the current candidate schema, so a reply saved under an older contract is
    refused rather than reinterpreted: the bytes and the request stay exactly
    as captured, and a person is told the shape janki now expects is not the
    shape that was asked for.
    """
    manifest = provenance.get("provider_manifest")
    channels = provenance.get("provider_channels")
    if not isinstance(manifest, Mapping) or not isinstance(channels, Mapping):
        raise extract.ExtractError(
            "This extraction has no saved provider request to recover from.",
            code="extract-request-unavailable",
        )
    stored_contract = str(provenance.get("response_schema_fingerprint") or "")
    current_contract = prompts.schema_fingerprint(
        claude_client.wire_schema(extract.candidate_schema())
    )
    if stored_contract and stored_contract != current_contract:
        raise extract.ExtractError(
            "This captured answer and the request that produced it are "
            "preserved exactly, but janki's extraction response contract has "
            "changed since that call, so the current reader must not parse it "
            "under a different contract.",
            code="extract-response-contract-changed",
        )
    return revision_provider.provider_plan_from_manifest(
        manifest,
        model=str(manifest["model"]),
        style_guide="",
        task_template="",
        system_blocks=tuple(dict(block) for block in channels["system_blocks"]),
        user_turn=str(channels["user_turn"]),
        schema=extract.candidate_schema(),
        input_blocks=tuple(dict(block) for block in channels["input_blocks"]),
    )


@dataclass(frozen=True, slots=True)
class _RecoveredSource:
    """Just enough of a prepared input to name the file a reply belongs to."""

    origin_path: Path


def recover_extraction(
    target: ExtractionTarget,
    *,
    model: str,
    mode: str | None,
    raw_reply: bytes,
) -> extract.ExtractionResult:
    """Read an already-paid-for reply again, without making a second call."""
    plan = target.provider_plan
    if plan is None:
        plan = provider_plan_from_provenance(target.provenance)
    call = revision_provider.provider_for(plan.provider).recover(plan, raw_reply)
    return extract.extraction_result_from_call(
        call,
        target.item,
        model=model,
        mode=mode,
        provenance=dict(target.provenance),
    )


def recover_extraction_from_capture(
    payload: bytes,
    *,
    source_name: str,
    mode: str | None = None,
) -> extract.ExtractionResult:
    """Recover one extraction from its captured artifact alone.

    The wrapper carries both halves — the request as planned and the reply as
    received — so this needs no prompt file, no login and no second call. The
    saved mode is the mode: a caller may name it to be sure, but naming a
    different one is refused rather than silently restating the answer as
    something it was not read for.
    """
    raw, provenance = extraction_capture_parts(payload)
    if provenance is None:
        raise extract.ExtractError(
            f"The captured reply for {source_name} has no saved request beside "
            "it, so it cannot be recovered on its own.",
            code="extract-request-unavailable",
        )
    plan = provider_plan_from_provenance(provenance)
    saved_mode = provenance.get("mode")
    # "auto" is how an unforced run is written down; None is how it is asked
    # for. The same thing, said by two layers.
    effective = None if saved_mode in (None, "auto") else str(saved_mode)
    if mode is not None and mode != effective:
        raise extract.ExtractError(
            f"The captured reply for {source_name} was read in "
            f"{saved_mode or 'auto'} mode; it cannot be recovered as {mode}.",
            code="extract-recovery-mode-mismatch",
        )
    call = revision_provider.provider_for(plan.provider).recover(plan, raw)
    return extract.extraction_result_from_call(
        call,
        _RecoveredSource(origin_path=Path(source_name)),
        model=str(provenance.get("model") or plan.model),
        mode=effective,
        provenance=dict(provenance),
    )


def durable_inbox_root(config: ProjectConfig) -> Path:
    """The full provenance root for the configured scan directory.

    Shared rather than private to the CLI: the workbench asks the same
    question — is this file already in the corpus — and a second answer to it
    would be a second definition of where the corpus is.
    """
    standard = config.root / "data" / "inbox"
    try:
        if config.scan_inbox.resolve().is_relative_to(standard.resolve()):
            return standard
    except OSError:
        pass
    # A project can configure one standalone inbox instead of the standard
    # data/inbox/scans subtree. In that shape the scan directory is the root.
    return config.scan_inbox


def _aligned_layouts(
    prepared: Sequence[PreparedInput],
    *,
    mode: str | None,
    layouts: Sequence[extract.TableLayout | None],
) -> tuple[extract.TableLayout | None, ...]:
    """One layout per input under the layout mode, none under any other.

    Checked before anything is planned, exactly as the batch planner's
    ``lineage`` length check is: a misalignment is a mistake about the request,
    not a discovery to make after touching a provider.
    """
    supplied = tuple(layouts)
    if mode == extract.LAYOUT_MODE:
        if len(supplied) != len(prepared) or any(
            item is None for item in supplied
        ):
            raise extract.ExtractError(
                f"A {extract.LAYOUT_MODE} extraction sends one bound layout per "
                f"source, in order: {len(prepared)} source(s) need "
                f"{len(prepared)} layout(s), and this call supplied "
                f"{sum(1 for item in supplied if item is not None)} of "
                f"{len(supplied)}. Nothing was planned.",
                code="extract-layout-misaligned",
            )
        for item in supplied:
            if not isinstance(item, extract.TableLayout):
                raise extract.ExtractError(
                    "A bound source-form layout is an immutable TableLayout "
                    "revision, not an identity pair. Nothing was planned.",
                    code="extract-layout-misaligned",
                )
        return supplied
    if any(item is not None for item in supplied):
        raise extract.ExtractError(
            f"A source-form layout is only sent under {extract.LAYOUT_MODE}; "
            f"this call names {mode or 'auto'} mode. Nothing was planned.",
            code="extract-layout-misaligned",
        )
    return (None,) * len(prepared)


def plan_extraction(
    config: ProjectConfig,
    prepared: Sequence[PreparedInput],
    *,
    mode: str | None,
    model: str,
    style_guide: str,
    system: str,
    layouts: Sequence[extract.TableLayout | None] = (),
    force: bool = False,
    scope_id: str = "",
    provider: str | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
) -> ExtractionPlan:
    """Resolve everything a run can know before it spends anything.

    Every refusal that does not need a provider happens here, which is the
    point: a batch that would write two inputs to one staging file is refused
    now rather than after paying for both.

    ``scope_id`` names the collection this extraction is *for*. The default is
    the shared one, which is what every existing caller means. A standalone
    deck holds its own copies of words under its own scope, so "janki already
    has this" is a different question per destination, and both halves of the
    answer — the ids a candidate can match and the expressions prose mode is
    told to skip — are filtered to that one namespace. A shared run therefore
    stops suppressing words only a standalone deck has, and a scoped run stops
    suppressing the shared words it exists to copy.

    ``layouts`` is aligned to the complete ``prepared`` sequence: one entry per
    input, in order. Under ``table-layout`` every entry is a layout, because a
    layout-bound part with no bound layout is exactly the request nobody
    authored; under every other mode every entry is absent. A single-source
    caller supplies a one-element sequence. Both mismatches refuse **before**
    any provider request is planned, so a misaligned call cannot probe a login
    or send a page.
    """
    bound_layouts = _aligned_layouts(prepared, mode=mode, layouts=layouts)
    existing: list[VocabularyRecord] = (
        load_records(config.normalized_file)
        if config.normalized_file.exists()
        else []
    )
    known = frozenset(extract.known_ids(existing, scope_id=scope_id))
    skip_list = (
        tuple(
            sorted(
                {
                    record.expression
                    for record in existing
                    if record_scope_id(record.id) == scope_id
                }
            )
        )
        if mode == "prose"
        else ()
    )

    chosen_provider = extraction_provider(config, provider)
    targets = extract.staging_targets(config.staging_dir, prepared, force=force)
    fingerprints = [
        item.source_sha256 or extract.source_fingerprint(item.origin_path)
        for item in prepared
    ]
    # Validate the complete store before a paid call. The parsed mapping is
    # deliberately not carried: a human can finish reviewing a source while the
    # call is in flight, and the post-call decision must see that newer state
    # under the writer lock rather than replacing it from this stale snapshot.
    patterns.load_store(config.patterns_file)

    planned: list[ExtractionTarget] = []
    for item, staging_path, fingerprint, layout in zip(
        prepared, targets, fingerprints, bound_layouts, strict=True
    ):
        # One provider request per source, planned once. The subscription
        # transport probes the local login here — before consent, and long
        # before authority — so a logged-out machine is a refusal rather than
        # a discovery made with a source already sent.
        provider_plan = extraction_provider_plan(
            item,
            provider=chosen_provider,
            model=model,
            style_guide=style_guide,
            system=system,
            known=skip_list,
            env=provider_env,
            runner=provider_runner,
            which=provider_which,
            layout=layout,
        )
        planned.append(
            ExtractionTarget(
                item=item,
                staging_path=staging_path,
                patterns_path=config.patterns_file,
                source_sha256=fingerprint,
                replaces=staging_path if staging_path.exists() else None,
                provenance=extract.prompt_provenance(
                    item,
                    model=model,
                    style_guide=style_guide,
                    system=system,
                    mode=mode,
                    known=skip_list,
                    source_sha256=fingerprint,
                    provider_plan=provider_plan,
                    layout=layout,
                ),
                provider_plan=provider_plan,
            )
        )

    return ExtractionPlan(
        model=model,
        mode=mode,
        targets=tuple(planned),
        style_guide=style_guide,
        system=system,
        skip_list=skip_list,
        known=known,
        scope_id=scope_id,
        provider=chosen_provider,
    )


def plan_corpus_extraction(
    config: ProjectConfig,
    source: Path,
    *,
    mode: str | None,
    model: str,
    layout: extract.TableLayout | None = None,
    force: bool = False,
    scope_id: str = "",
    provider: str | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
) -> ExtractionPlan:
    """Freshly plan one source that must already be in the durable corpus.

    GET uses this with ``force=True`` solely so it can describe an existing
    review. POST calls it again with only the explicit replacement decision.
    Keeping the descriptor-bound corpus capture, prompt reload and input
    preparation here means the two surfaces describe and send by the same
    rules without ever handing the forced GET plan to dispatch.
    """
    prepared = [inputs.prepare_corpus_input(source, config.scan_inbox)]
    return plan_extraction(
        config,
        prepared,
        mode=mode,
        model=model,
        style_guide=claude_client.read_style_guide(config.root),
        system=prompts.load(config.root, extract.prompt_name(mode)),
        # One source, so the aligned sequence is one element long.
        layouts=(layout,),
        force=force,
        scope_id=scope_id,
        provider=provider,
        provider_env=provider_env,
        provider_runner=provider_runner,
        provider_which=provider_which,
    )


#: What became of a call that failed after the money may already have gone.
#:
#: Five states, and the difference between them is what the person is owed.
#: `answer_saved` — the bytes are on disk and something after them refused, so
#: there is paid content to look at. `answer_empty` — a reply arrived carrying
#: only the model's reasoning, so it was billed and holds nothing to recover;
#: calling that "the answer was saved" sends somebody hunting for cards in a
#: file with none. `answer_unavailable` — the journal records that a reply was
#: captured, but its exact recovery bytes can no longer be read; this is neither
#: an empty answer nor a vanished provider outcome. `outcome_unknown` — it was
#: sent and no answer was captured, which is the one where a retry risks a
#: second charge. `forgotten` — the user's final discard decision won the race,
#: so no stale handler may recover or advertise that operation's evidence again.
ANSWER_SAVED = "answer_saved"
ANSWER_EMPTY = "answer_empty"
ANSWER_UNAVAILABLE = "answer_unavailable"
OUTCOME_UNKNOWN = "outcome_unknown"
FORGOTTEN = "forgotten"


@dataclass(frozen=True, slots=True)
class DispatchFailure:
    """A dispatch that did not produce usable cards, and what it cost."""

    operation_id: str
    #: One of the five dispatch failure constants above.
    outcome: str
    #: Whether a retry risks paying twice. Read from the journal rather than
    #: guessed from the exception — it is the question a retry turns on.
    #:
    #: True for every `OUTCOME_UNKNOWN` today, and by construction rather than
    #: by accident: the transition table only allows that state from a
    #: dispatched one. It stays a field because it is the question W3's
    #: recovery surface asks, and answering it from the journal keeps it true
    #: if the table ever grows a not-billed terminal state.
    money_may_have_been_spent: bool = False
    #: The forget decision exists but exact evidence cleanup still needs its
    #: ordinary retry. False means the entry and its bound evidence are gone.
    cleanup_pending: bool = False

    @property
    def was_paid_for(self) -> bool:
        return self.outcome in (ANSWER_SAVED, ANSWER_EMPTY, ANSWER_UNAVAILABLE)


class ExtractionDispatchError(JankiError):
    """A shared extraction run that could not become staged proposals.

    ``phase`` lets each surface describe the same durable facts in its own
    presentation without reimplementing the operation.  ``binding``,
    ``preparation`` and ``authorization`` happen before provider dispatch;
    ``dispatch`` and ``completion`` happen after it.  A dispatch failure may
    carry the journal's exact recovery classification, while ``journal_error``
    records the rarer case where even that classification could not be made.
    """

    def __init__(
        self,
        cause: BaseException,
        *,
        phase: str,
        operation_id: str | None = None,
        failure: DispatchFailure | None = None,
        journal_error: JankiError | None = None,
    ) -> None:
        self.cause = cause
        self.phase = phase
        self.operation_id = operation_id
        self.failure = failure
        self.journal_error = journal_error
        message = str(cause).strip() or f"{type(cause).__name__} failed without details."
        super().__init__(message)

    @property
    def provider_dispatched(self) -> bool:
        return self.phase in {"dispatch", "completion"}


def busy_refusal(config: ProjectConfig) -> str:
    """Why janki will not start a paid run right now, in a person's words.

    W3: *"One mutating job at a time."* The answer comes from the journal
    rather than from a flag this process keeps, because a run started in a
    terminal is exactly as real as one started in a browser tab, and a guard
    that only knew about its own process would let the two overlap and bill
    twice.

    **This is a display, not the enforcement point.** It is read when a page
    renders and acted on when a button is clicked, so two callers can both
    pass it and both reach the writer. The rule itself lives in
    `OperationJournal.authorize`, which refuses under the journal's own lock;
    this exists so a page can say so in advance rather than let somebody click
    a button that was never going to work. It therefore reads the same set the
    gate does — a display that disagreed with the rule would be worse than no
    display at all.
    """
    journal = operations.OperationJournal.load(config.operations_file)
    blocking = journal.blocking()
    if not blocking:
        return ""
    first = blocking[0]
    where = "'janki operations' shows it"
    if first.state in operations.IN_FLIGHT:
        return (
            f"A call about {first.source_file} is still marked as running, so "
            "janki will not start another — that would risk a second charge. "
            f"If nothing is actually running, that call was interrupted; "
            f"{where}."
        )
    if first.state == "authorized":
        return (
            f"An earlier run wrote authority to read {first.source_file} and "
            f"never sent anything. janki will not start another until that is "
            f"cleared; {where}."
        )
    # `result_captured` and `outcome_unknown`. "may have been billed", not
    # "has been paid for": the second of those is in this list precisely
    # because nobody knows.
    return (
        f"A call about {first.source_file} may have been billed and has not "
        f"been dealt with. {where}."
    )


@dataclass(frozen=True, slots=True)
class ExtractionConsent:
    """One source, and exactly what sending it would mean, before anyone agrees.

    W3: *"the consent button names exactly what leaves the computer"*. That
    sentence cannot be written from a function that asks and dispatches in one
    breath, and it cannot be written from `plan_extraction` alone either —
    planning *raises* on the staging collision a consent page has to **show**,
    because replacing an extraction is a decision someone makes, not an error
    they hit. So this plans as though forced, and hands the collision back as
    something to confirm.

    Nothing here contacts a provider or journals anything.

    **A dispatch must re-plan rather than send what this describes.** Every
    field here is a snapshot taken when a page was rendered: staging can
    appear between the render and the click, and `replaces` is populated only
    because planning was forced, so deriving `force` from it would force a run
    nobody confirmed. This value is what a person is shown and what their
    confirmation is checked against — not what is sent.
    """

    #: The permanent filename, which is what a person recognizes.
    name: str
    model: str
    mode: str | None
    #: The single target this run would write, or None when it cannot be
    #: planned at all. Deliberately *not* the whole `ExtractionPlan`: that
    #: value carries the forced planning this page needs and a dispatch must
    #: not inherit, and a field nobody can reach is a better guarantee than a
    #: docstring asking them not to.
    target: ExtractionTarget | None = None
    #: A staging file this run would overwrite. When set, the page must
    #: confirm the replacement separately and name what it invalidates.
    replaces: Path | None = None
    #: That file's review, in the words the dashboard already uses for it —
    #: "Ready to add", "Cards need edits". A path and a row count do not tell
    #: somebody what they are about to throw away; its state does.
    replaces_state: str = ""
    replaces_cards: int = 0
    #: The grammar-track state the same force would reset.
    replaces_grammar: str = ""
    #: Exact card and grammar revisions behind the description above.
    replacement_revision: ExtractionRevision | None = None
    #: Why this cannot be sent at all, in a learner's words. Empty when it can.
    refusal: str = ""
    #: True when this run also sends the expressions already in the
    #: collection. Prose mode does; a table is transcribed row by row, so
    #: telling the model to skip rows would put holes in it.
    sends_known_words: bool = False
    #: Why janki will not start *any* paid run right now. Separate from
    #: `refusal`: this source is fine, the moment is not, and a page that
    #: merged them would tell somebody their lesson was the problem.
    busy: str = ""
    #: The collection this consent's known-word suppression was computed for:
    #: one deck's scope, or ``""`` for the shared one. Rendered, because "which
    #: words will janki treat as already had" is part of what is being agreed
    #: to — and it survives a refusal, so a page can still say which
    #: destination it was asking about.
    scope_id: str = ""

    @property
    def provider(self) -> str:
        """The transport this consent is about, or empty when nothing is."""
        return "" if self.target is None else self.target.provider

    @property
    def billing_display(self) -> str:
        """Who pays, taken from the planned request rather than written here."""
        return "" if self.target is None else self.target.billing_display

    @property
    def sendable(self) -> bool:
        return self.target is not None and not self.refusal and not self.busy


def describe_extraction(
    config: ProjectConfig,
    source: Path,
    *,
    mode: str | None = None,
    model: str | None = None,
    scope_id: str = "",
    provider: str | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
) -> ExtractionConsent:
    """What sending one corpus source to a model would mean, before agreeing.

    Reads the same style guide and system prompt the dispatch sends, so the
    plan behind the consent *is* the plan that would be dispatched rather than
    a description resembling one. A page that showed a different request
    identity from the one journalled would make the journal's whole promise —
    that it names the call that was made — untrue at the only moment anybody
    reads it.

    Refuses a source that is not already in the corpus. Adding a file and
    sending it are two separate actions, always; a consent page that copied
    its own subject in would have quietly done the first while asking about
    the second.
    """
    chosen = model or config.extract_model
    try:
        plan = plan_corpus_extraction(
            config,
            source,
            mode=mode,
            model=chosen,
            # Planned as though forced so a staging collision comes back as a
            # replacement to confirm rather than an exception. Consenting to
            # the replacement is what supplies `force` to the dispatch; this
            # value never sends anything.
            force=True,
            scope_id=scope_id,
            provider=provider,
            provider_env=provider_env,
            provider_runner=provider_runner,
            provider_which=provider_which,
        )
        replaces = plan.targets[0].replaces if plan.targets else None
        state, cards, grammar = "", 0, ""
        replacement_revision = None
        if replaces is not None:
            # Capture around the human-readable journey. If either exact
            # review moves while that description is being assembled, there
            # is no single state this page could truthfully ask to replace.
            before = extraction_replacement_revision(config, plan.targets[0])
            found = next(
                (
                    journey
                    for journey in source_journeys(config)[0]
                    if journey.staging_path == replaces
                ),
                None,
            )
            if found is not None:
                state, cards, grammar = found.state, found.card_count, found.grammar
            replacement_revision = extraction_replacement_revision(
                config, plan.targets[0]
            )
            if replacement_revision != before:
                raise staging.StagingError(
                    f"The review for {source.name} changed while the consent page "
                    "was loading; reload it"
                )
            # The dashboard intentionally has no grammar badge when staging's
            # embedded set is empty or missing. Force still replaces an
            # existing pattern-store entry, so consent must name that review
            # independently of whether this staging generation can badge it.
            if replacement_revision.pattern_reviewed is not None and (
                replacement_revision.pattern_reviewed
                or replacement_revision.pattern_has_patterns
            ):
                grammar = (
                    GRAMMAR_REVIEWED
                    if replacement_revision.pattern_reviewed
                    else GRAMMAR_NEEDS_REVIEW
                )
        # `data/operations.json` is exactly the file a killed paid call
        # leaves in interesting shapes, and the one page that manages
        # paid-call state must not be the one that dies when it cannot be
        # parsed. It lands in `busy` rather than `refusal` on purpose: not
        # knowing whether a call is already running is a fact about the
        # moment, and reporting it under "Send this file?" would tell
        # somebody their lesson was the problem.
        try:
            busy = busy_refusal(config)
        except JankiError as exc:
            busy = (
                "janki cannot tell whether a paid call is already running, so "
                f"it will not start one: {exc}"
            )
    except JankiError as exc:
        return ExtractionConsent(
            name=source.name,
            model=chosen,
            mode=mode,
            refusal=str(exc),
            scope_id=scope_id,
        )

    return ExtractionConsent(
        name=source.name,
        model=chosen,
        mode=mode,
        scope_id=plan.scope_id,
        target=plan.targets[0] if plan.targets else None,
        replaces=replaces,
        replaces_state=state,
        replaces_cards=cards,
        replaces_grammar=grammar,
        replacement_revision=replacement_revision,
        sends_known_words=bool(plan.skip_list),
        busy=busy,
    )


def authorize_dispatch(
    journal: operations.OperationJournal,
    target: ExtractionTarget,
    *,
    model: str,
    stream: bool = False,
) -> str:
    """Record the authority for one call and mark it dispatching.

    Before the request exists, not after it succeeds. The interval this
    protects is the one where the money is spent and nothing on disk
    remembers: a crash between the send and the parse otherwise leaves no way
    to tell "never sent" from "sent and lost".
    """
    # Refuse an unusable recovery store before authority is written or a
    # provider can be called. The bound artifact writer checks again at the
    # actual reply, but finding a static problem only after paying would lose
    # the answer this journal exists to preserve.
    operations.prepare_artifact_store(journal.path)
    for output_parent in {
        target.staging_path.parent,
        target.patterns_path.parent,
    }:
        prepare_bound_directory(output_parent)
    operation_id = str(uuid.uuid4())
    journal.authorize(
        operation_id,
        kind="extract",
        source_file=target.item.origin_path.name,
        source_sha256=target.source_sha256,
        request_fp=str(target.provenance["request_fingerprint"]),
        model=model,
    )
    if stream:
        # A streaming transport spools its exact frames as they arrive, so an
        # interrupted call still leaves what was received. Bound between the
        # authority and the dispatch state, which is the only window the
        # journal accepts one in.
        journal.begin_response_capture(operation_id)
    journal.advance(operation_id, "dispatching")
    return operation_id


def _capture_payload(
    response: Any, provenance: Mapping[str, Any] | None
) -> bytes:
    """The durable bytes for one captured answer.

    ``serialize_response`` is deliberately forgiving about unfamiliar objects,
    and what it makes of a ``bytes`` is that object's repr — quotes, escapes
    and all — which is not the reply anybody was billed for. A transport that
    hands over raw bytes has already given the exact answer, so it is written
    exactly, wrapped with the request that produced it when one is known.
    """
    if isinstance(response, bytes | bytearray):
        raw = bytes(response)
        return raw if provenance is None else extraction_capture_envelope(
            provenance, raw
        )
    return operations.serialize_response(response)


def capture_hook(
    config: ProjectConfig,
    journal: operations.OperationJournal,
    operation_id: str,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> Callable[[Any], None]:
    """The client's capture callback: persist the exact reply, then journal it.

    In that order. The bytes are what a person can still act on if parsing
    refuses, so they reach disk before anything says they arrived.

    ``provenance`` saves the request beside those bytes, so an interrupted or
    rejected answer can be read again later without asking today's prompts
    what was sent.
    """

    def _capture(response: Any) -> None:
        journal.capture_result(
            operation_id,
            lambda: operations.capture_artifact(
                config.operations_file,
                operation_id,
                _capture_payload(response, provenance),
            ),
        )

    return _capture


def settle_dispatch(
    config: ProjectConfig,
    journal: operations.OperationJournal,
    operation_id: str,
    result: Any,
) -> None:
    """Record an answer the capture hook never saw.

    The hook fires inside the client, before validation, which is where an
    unparseable-but-paid-for answer has to be caught. It is not the only path:
    a provider wrapper that never calls it would leave the entry at
    `dispatching` for ever. An answer did arrive — it is in `result` — so
    record that, using the normalized value as the artifact when the exact
    bytes were not captured.

    Both states a call can still be in, not just `dispatching`: a caller that
    reports streaming moves through `running`, and an answer that arrives
    there is exactly as paid for.

    Does nothing to an entry that already recorded an answer, so a caller may
    settle without first knowing whether the hook fired.

    The artifact this writes is janki's normalized value, not the provider's
    own reply, so provider-shaped content parsing finds no answer block in it.
    What it holds is still readable by a person, which is the promise; a
    surface classifying already-bound reply bytes must not call one of these
    empty.
    """
    journal.capture_result(
        operation_id,
        lambda: operations.capture_artifact(
            config.operations_file,
            operation_id,
            operations.serialize_response(result),
        ),
    )


def classify_dispatch_failure(
    config: ProjectConfig,
    journal: operations.OperationJournal,
    operation_id: str,
    exc: BaseException,
) -> DispatchFailure:
    """Decide what a failed call already cost, and leave the journal truthful.

    Read back from the file, not from `journal`. `advance` does update the
    object it is called on, so for a run that captured through *this* journal
    the two agree — but nothing makes that the only shape. A caller that holds
    one journal while the capture advances another (a request handler and a
    client, say) would be asking a copy that never saw the answer, and would
    then report a paid reply as an unknown outcome.

    An entry that reached `result_captured` is **left there**. That is the
    truthful state — the paid bytes are on disk and a person has to decide what
    to do with them — and calling it "unknown" would hide an answer already
    bought.

    This runs on the error path, so it does not add a failure of its own. An
    entry already in a terminal state is reported rather than moved: somebody
    ending the call by hand while it was still in flight is exactly when this
    function runs, and refusing `outcome_unknown → outcome_unknown` there would
    replace the provider's error with a journal error and tell the person
    nothing about either.
    """
    # Classification and any outcome transition share the journal lock with
    # forget. In particular, cleanup/missing short-circuit before probing the
    # artifact WAL: a stale validation frame must not recover or advertise
    # evidence after the user's durable discard decision.
    with exclusive_path_lock(config.operations_file):
        current = operations.OperationJournal.load(config.operations_file)
        held = current.operations.get(operation_id)
        if held is None:
            return DispatchFailure(
                operation_id=operation_id,
                outcome=FORGOTTEN,
            )
        if held.cleanup is not None:
            return DispatchFailure(
                operation_id=operation_id,
                outcome=FORGOTTEN,
                cleanup_pending=True,
            )

        # The exact bytes may be public or retained only by a bound private WAL
        # name. A journal path is historical state, never accessibility proof.
        reply = operations.reply_observation(config.operations_file, held)
        if reply.readable:
            answer = operations.response_answer_text(reply.payload)
            return DispatchFailure(
                operation_id=operation_id,
                outcome=ANSWER_SAVED if answer else ANSWER_EMPTY,
            )
        if reply.recorded:
            return DispatchFailure(
                operation_id=operation_id,
                outcome=ANSWER_UNAVAILABLE,
            )
        if held.state in operations.TERMINAL_STATES:
            return DispatchFailure(
                operation_id=operation_id,
                outcome=OUTCOME_UNKNOWN,
                money_may_have_been_spent=held.money_may_have_been_spent,
            )
        marked = journal._move_under_lock(
            current,
            operation_id=operation_id,
            state="outcome_unknown",
            detail=redact_environment_credentials(exc),
        )
        return DispatchFailure(
            operation_id=operation_id,
            outcome=OUTCOME_UNKNOWN,
            money_may_have_been_spent=marked.money_may_have_been_spent,
        )


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    """What one answer became: a staging file, and what is in it."""

    target: Path
    records: int
    already_known: int
    unusable: int
    duplicates: int
    coverage_status: str
    #: True when a *reviewed* pattern answer for this source was already
    #: stored and kept, so this run's grammar half is in the staging file
    #: only. Worth reporting: the reviewer's decision won, deliberately.
    kept_reviewed_patterns: bool
    source: str


class ExtractionCompletionError(JankiError):
    """Staging committed, but the answer's separate pattern write failed."""

    def __init__(self, cause: JankiError, staging_path: Path) -> None:
        self.detail = str(cause)
        self.staging_path = Path(staging_path)
        super().__init__(
            f"{cause} Proposals remain saved at {self.staging_path}, including "
            "the embedded pattern set."
        )


def _require_current_replacement(
    current: ExtractionRevision,
    expected: ExtractionRevision,
) -> None:
    if current.staging_sha256 != expected.staging_sha256:
        raise operations.OperationError(
            "The review changed while the source was being read; refusing to "
            "overwrite the newer work."
        )
    if current.pattern_entry_sha256 != expected.pattern_entry_sha256:
        raise operations.OperationError(
            "The grammar review changed while the source was being read; refusing "
            "to overwrite the newer work."
        )


def complete_extraction(
    config: ProjectConfig,
    journal: operations.OperationJournal,
    target: ExtractionTarget,
    result: extract.ExtractionResult,
    *,
    operation_id: str,
    known: Iterable[str],
    mode: str | None,
    model: str,
    force: bool = False,
    expected_revision: ExtractionRevision | None = None,
    capture_recovery: Mapping[str, Any] | None = None,
) -> ExtractionOutcome:
    """Turn one paid answer into a staging file, and close its journal entry.

    The half after the money. It runs in the caller's thread with the answer
    already in hand, so a browser and the command reach a staging file by the
    same route — which is the only way the two can agree about what an
    extraction produced.

    Ordering is load-bearing twice over. The journal is committed only once
    the staging file exists, because "committed" means *the exact answer
    became staging* and nothing else. And the pattern store is re-read inside
    its own lock after the call: `patterns --review` takes the same lock, so
    either a reviewer's complete decision wins and is preserved here, or this
    fresh unreviewed answer lands first and the reviewer sees it — neither can
    silently erase the other, nor an unrelated source added while the model
    was still answering.

    ``capture_recovery`` is the one caller-supplied metadata seam. It exists so
    that an answer salvaged out of a captured reply records *which* envelope in
    that reply it was read from, written by the sole staging writer rather than
    patched in beside it. Omitted — which is every ordinary dispatch and the
    ordinary batch recovery — the staging bytes are exactly what they were.
    """
    # Identity before anything else, because everything after it records this
    # answer against this entry. A caller running two extractions at once and
    # pairing one's operation with the other's target is the same variable
    # swap as a skipped settle, and it would otherwise commit entry A saying
    # its exact answer became B's staging file — silently, with both files
    # written and neither wrong on its face.
    #
    # Refusing does lose an in-memory answer, which this function otherwise
    # never does. The difference is that here there is no truthful entry to
    # record it against: the pairing itself is what cannot be believed.
    target_provenance = dict(target.provenance)
    normalized_mode = mode or "auto"
    if (
        target_provenance.get("source_sha256") != target.source_sha256
        or target_provenance.get("model") != model
        or target_provenance.get("mode") != normalized_mode
        or dict(result.pattern_set.prompt_provenance) != target_provenance
        or result.pattern_set.source != target.name
    ):
        raise operations.OperationError(
            f"The answer does not describe the request for {target.name!r}; "
            "refusing to record it against that target."
        )

    held = operations.OperationJournal.load(config.operations_file).operations.get(
        operation_id
    )
    expected_identity = {
        "kind": "extract",
        "source_file": target.name,
        "source_sha256": str(target_provenance["source_sha256"]),
        "request_fp": str(target_provenance["request_fingerprint"]),
        "model": str(target_provenance["model"]),
    }
    if held is not None and any(
        getattr(held, field) != value
        for field, value in expected_identity.items()
    ):
        raise operations.OperationError(
            f"Operation {operation_id!r} authorized a different request than "
            f"{target.name!r} describes; refusing to record this answer "
            "against it."
        )

    # This function is holding an answer somebody paid for, so a refusal here
    # must never be how that answer is lost. A caller that skipped
    # `settle_dispatch` — a browser handler with a missing branch — arrives
    # with the entry still saying the call is in flight; settling closes that
    # seam rather than guarding it, and puts the bytes on disk where
    # `janki status --operations` can find them either way.
    settle_dispatch(config, journal, operation_id, result)

    # Then asked before the staging write, though `advance` asks again after
    # it. The commit stays last on purpose — "committed" has to mean the file
    # exists — but a refusal after the write would leave a staging file under
    # an entry that never accounted for it. Read from the file: a caller may
    # hold a journal that never saw the capture.
    entry = operations.OperationJournal.load(config.operations_file).operations.get(
        operation_id
    )
    if entry is None:
        refusal = f"No operation {operation_id!r} to complete"
    else:
        refusal = operations.advance_refusal(
            operation_id, entry.state, "committed", entry.artifact
        )
    if refusal:
        raise operations.OperationError(refusal)

    item = target.item
    # The frozen layout from the saved request manifest, never the current
    # mutable job document: the record this writes describes the request that
    # was paid for, whatever the owner has since repointed.
    built = extract.build_records(
        result.candidates,
        item,
        known,
        layout=extract.layout_from_provenance(target_provenance),
    )
    records = built.records
    coverage = extract.coverage_block(
        result,
        source_sha256=target.source_sha256,
        mode=mode,
        candidate_accounting=built.candidate_accounting,
    )
    run_id = new_review_run_id()
    run_patterns = replace(result.pattern_set, review_run_id=run_id)
    meta: dict[str, Any] = {
        # The basename, like every other writer of this key. An absolute path
        # is stale on any other clone, and this file is committed.
        "source_file": item.origin_path.name,
        "extracted_at": date.today().isoformat(),
        "model": model,
        "review_run_id": run_id,
        "prompt_provenance": dict(result.pattern_set.prompt_provenance),
        # The same paid answer also inferred the document patterns. Keep a
        # complete copy beside the cards so a pattern-store write failure
        # cannot make that half of the answer unrecoverable.
        "pattern_set": run_patterns.to_dict(),
        "coverage": coverage,
    }
    # Held back into the file, not just onto the terminal: the staging file is
    # what a reviewer reads later, and a count that lives only in scrollback is
    # the same silent discard with an extra step.
    held = list(built.unusable_candidates)
    if held:
        meta["review_notes"] = extract.unusable_note(held)
    meta[CANDIDATE_ACCOUNTING_KEY] = built.candidate_accounting
    if capture_recovery is not None:
        # Round-tripped through JSON so the staging serializer receives plain
        # types, exactly as the accounting block above already is.
        meta[CAPTURE_RECOVERY_KEY] = json.loads(
            json.dumps(dict(capture_recovery), ensure_ascii=False)
        )
    kept_reviewed_patterns = False
    if expected_revision is None:
        # Staging's path lock is outside the journal lock, matching the
        # replacement transaction below. The authoritative state/cleanup
        # check, write, and committed transition then happen under the journal
        # lock, so forget cannot land in their seam.
        with exclusive_path_lock(target.staging_path):
            journal.commit_result(
                operation_id,
                lambda: write_staging_under_lock(
                    target.staging_path,
                    records,
                    meta,
                    force=force,
                ),
            )

        try:
            with exclusive_path_lock(config.patterns_file):
                pattern_snapshot = _pattern_store_snapshot(
                    config.patterns_file, run_patterns.source
                )
                previous = pattern_snapshot.store.get(run_patterns.source)
                if previous is not None and previous.reviewed and not force:
                    kept_reviewed_patterns = True
                else:
                    pattern_snapshot.store[run_patterns.source] = run_patterns
                    patterns.save_store_under_lock(
                        config.patterns_file,
                        pattern_snapshot.store,
                        expected_revision=(
                            None
                            if pattern_snapshot.absent
                            else hashlib.sha256(pattern_snapshot.wire).hexdigest()
                        ),
                        expected_absent=pattern_snapshot.absent,
                    )
        except JankiError as exc:
            raise ExtractionCompletionError(exc, target.staging_path) from exc
    else:
        if not force:
            raise operations.OperationError(
                "An exact replacement revision requires explicit replacement authority"
            )
        # The paid call cannot hold review locks while it waits on a provider.
        # Reacquire both only for the compare-and-swap and the two writes. The
        # per-source comparison catches cooperating review changes; the bound
        # writers close the final seam against an editor that ignores locks.
        with _replacement_locks(target.staging_path, config.patterns_file):
            current, pattern_snapshot = _replacement_state_under_lock(config, target)
            _require_current_replacement(current, expected_revision)
            assert expected_revision.staging_sha256 is not None
            journal.commit_result(
                operation_id,
                lambda: write_staging_under_lock(
                    target.staging_path,
                    records,
                    meta,
                    force=True,
                    expected_revision=expected_revision.staging_sha256,
                ),
            )
            try:
                previous = pattern_snapshot.store.get(run_patterns.source)
                if previous is not None and previous.reviewed and not force:
                    kept_reviewed_patterns = True
                else:
                    pattern_snapshot.store[run_patterns.source] = run_patterns
                    patterns.save_store_under_lock(
                        config.patterns_file,
                        pattern_snapshot.store,
                        expected_revision=(
                            None
                            if pattern_snapshot.absent
                            else hashlib.sha256(pattern_snapshot.wire).hexdigest()
                        ),
                        expected_absent=pattern_snapshot.absent,
                    )
            except JankiError as exc:
                raise ExtractionCompletionError(exc, target.staging_path) from exc

    return ExtractionOutcome(
        target=target.staging_path,
        records=len(records),
        already_known=sum(
            1 for record in records if "already_known" in record.source.raw_fields
        ),
        unusable=len(held),
        duplicates=built.candidate_accounting["duplicate_candidate_count"],
        coverage_status=coverage["status"],
        kept_reviewed_patterns=kept_reviewed_patterns,
        source=run_patterns.source,
    )


#: The deck kinds that can hold proposed vocabulary cards. A conjugation or
#: kanji deck is a real selected deck and the wrong destination for this
#: answer, so naming it is a refusal rather than a silent shared run.
VOCABULARY_DECK_KINDS = ("", "vocabulary")


def destination_deck_facts(deck_path: Path) -> tuple[str, str]:
    """One selected deck's record scope and the hash of its exact bytes.

    Structural only, and deliberately through the same readers the builder
    uses: the deck file's declared kind, its ``scope_id`` as
    :func:`deck_selection` parses it, and the bytes themselves. Nothing here
    looks at a card, and no second definition of what a scope is.
    """
    # One read, then parsed and hashed from that one snapshot. Two reads would
    # let an edit land between them and answer with one file's scope and the
    # other file's hash — the exact pair this consent binding exists to compare.
    try:
        wire = read_bytes_bound(deck_path)
    except FileNotFoundError as exc:
        raise staging.StagingError(f"Deck file not found: {deck_path}") from exc
    except OSError as exc:
        raise staging.StagingError(
            f"Could not read {deck_path}: {exc.strerror or exc}"
        ) from exc
    try:
        text = wire.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise staging.StagingError(f"Could not read {deck_path}: {exc}") from exc
    raw = _parse_structured_text(deck_path, text)
    if not isinstance(raw, dict):
        raise staging.StagingError(f"Deck file must contain a mapping: {deck_path}")
    deck_config = raw.get("deck") or {}
    if not isinstance(deck_config, dict):
        raise staging.StagingError(f"The deck section must be a mapping: {deck_path}")
    kind = str(deck_config.get("kind") or "").strip().lower()
    if kind not in VOCABULARY_DECK_KINDS:
        raise staging.StagingError(
            f"A {kind} deck cannot hold proposed vocabulary cards, so it cannot "
            f"be an extraction destination: {deck_path}"
        )
    selection = deck_selection(deck_config, deck_path)
    return selection.scope_id, hashlib.sha256(wire).hexdigest()


def require_current_destination(expected: ExtractionDispatchExpectation) -> None:
    """Refuse a paid call whose rendered destination deck has moved.

    Before authorization and before any provider call, because this is a
    consent binding rather than a display: the owner agreed to extract *for
    one named deck*, and its scope is what decided which words janki treated
    as already had.

    The scope is compared before the bytes on purpose. Every scope edit is
    also a byte edit, so a hash-only check would refuse correctly and explain
    it wrongly — "the deck changed" for a file that changed collections. It
    would also leave the scope binding untested and free to disappear.
    """
    if expected.destination_deck is None:
        if expected.destination_deck_sha256:
            raise ExtractionDispatchError(
                staging.StagingError(
                    "This extraction binds destination deck bytes without naming "
                    "the deck they belong to."
                ),
                phase="binding",
            )
        return
    try:
        scope, deck_sha256 = destination_deck_facts(expected.destination_deck)
    except (JankiError, OSError) as exc:
        raise ExtractionDispatchError(
            staging.StagingError(f"The destination deck could not be read: {exc}"),
            phase="binding",
        ) from exc
    if scope != expected.scope_id:
        raise ExtractionDispatchError(
            staging.StagingError(
                "The destination deck scope changed after this page was rendered."
            ),
            phase="binding",
        )
    if deck_sha256 != expected.destination_deck_sha256:
        raise ExtractionDispatchError(
            staging.StagingError(
                "The destination deck changed after this page was rendered."
            ),
            phase="binding",
        )


def _report_extraction_progress(
    progress: Callable[[str], None] | None,
    label: str,
) -> None:
    if progress is None:
        return
    try:
        progress(label)
    except Exception:  # noqa: BLE001 - presentation cannot control a paid transaction
        # A tab, socket, or event loop can disappear after authority is
        # journaled. Progress is best-effort display only: allowing its
        # callback to interrupt here would strand either an unused authority
        # or an already captured paid answer instead of completing the shared
        # recovery/staging transaction.
        return


@dataclass(frozen=True, slots=True)
class RevalidatedExtraction:
    """One freshly re-planned call, proved to be the one an owner confirmed.

    Every field is read again from disk at click time: nothing an expectation
    carried is dispatched.  Holding the four values together is what lets a
    single call and one child of a batch share the same revalidation instead
    of keeping two definitions of "the request has not moved".
    """

    plan: ExtractionPlan
    target: ExtractionTarget
    #: Explicit owner confirmation, which is the only force authority there is.
    force: bool
    #: The exact review that confirmation allows this answer to destroy.
    expected_revision: ExtractionRevision | None


def revalidate_extraction_request(
    config: ProjectConfig,
    expected: ExtractionDispatchExpectation,
    *,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
) -> RevalidatedExtraction:
    """Read everything again and refuse any difference, before any authority.

    Pure with respect to the journal and the provider: it plans, compares and
    raises.  A batch calls it for *every* child before it reserves anything,
    which is what makes "one page moved, so nothing was sent" a property of
    the batch rather than of whichever child happened to be checked first.
    """
    if expected.replacement_revision is not None and not expected.replacement_confirmed:
        raise ExtractionDispatchError(
            staging.StagingError(
                "Confirm that this re-read replaces the review named on the page."
            ),
            phase="binding",
        )
    if expected.replacement_revision is None and expected.replacement_confirmed:
        raise ExtractionDispatchError(
            staging.StagingError(
                "This extraction confirms a replacement the rendered action did "
                "not offer."
            ),
            phase="binding",
        )

    # Explicit owner confirmation is the only force authority.  In
    # particular, a replacement revision is a rendered snapshot, not consent.
    force = expected.replacement_confirmed
    try:
        plan = plan_corpus_extraction(
            config,
            expected.source,
            mode=expected.mode,
            model=expected.model,
            # The expectation's own frozen layout, never a fresh lookup: the
            # request-fingerprint comparison below is what refuses a layout
            # that has changed or gone missing since the confirmation, on
            # initial dispatch, on an unsent resume and on a selected retry
            # alike.
            layout=expected.table_layout,
            force=force,
            scope_id=expected.scope_id,
            provider_env=provider_env,
            provider_runner=provider_runner,
            provider_which=provider_which,
        )
    except JankiError as exc:
        raise ExtractionDispatchError(exc, phase="binding") from exc
    if len(plan.targets) != 1:
        cause = staging.StagingError(
            "That source no longer makes one extraction request."
        )
        raise ExtractionDispatchError(cause, phase="binding")

    target = plan.targets[0]
    if not expected.provider or plan.provider != expected.provider:
        cause = staging.StagingError(
            "The extraction transport changed after this page was rendered."
        )
        raise ExtractionDispatchError(cause, phase="binding")
    fresh_fingerprint = str(target.provenance["request_fingerprint"])
    if (
        fresh_fingerprint != expected.request_fingerprint
        or target.source_sha256 != expected.source_sha256
    ):
        cause = staging.StagingError(
            "The extraction request changed after this page was rendered."
        )
        raise ExtractionDispatchError(cause, phase="binding")

    fresh_staging_path = Path(os.path.realpath(target.staging_path))
    if fresh_staging_path != expected.staging_path:
        cause = staging.StagingError(
            "The extraction destination changed after this page was rendered."
        )
        raise ExtractionDispatchError(cause, phase="binding")
    fresh_patterns_path = Path(os.path.realpath(target.patterns_path))
    if fresh_patterns_path != expected.patterns_path:
        cause = staging.StagingError(
            "The extraction destination changed after this page was rendered."
        )
        raise ExtractionDispatchError(cause, phase="binding")
    fresh_operations_path = Path(os.path.realpath(config.operations_file))
    if fresh_operations_path != expected.operations_path:
        cause = operations.OperationError(
            "The extraction destination changed after this page was rendered."
        )
        raise ExtractionDispatchError(cause, phase="binding")

    try:
        fresh_revision = extraction_replacement_revision(config, target)
    except JankiError as exc:
        raise ExtractionDispatchError(exc, phase="binding") from exc
    rendered_revision = expected.replacement_revision
    if fresh_revision is None or rendered_revision is None:
        if fresh_revision != rendered_revision:
            cause = staging.StagingError(
                "The review changed after this page was rendered."
            )
            raise ExtractionDispatchError(cause, phase="binding")
    elif fresh_revision.staging_sha256 != rendered_revision.staging_sha256:
        cause = staging.StagingError(
            "The review changed after this page was rendered."
        )
        raise ExtractionDispatchError(cause, phase="binding")
    elif (
        fresh_revision.pattern_entry_sha256
        != rendered_revision.pattern_entry_sha256
    ):
        cause = patterns.PatternError(
            "The grammar review changed after this page was rendered."
        )
        raise ExtractionDispatchError(cause, phase="binding")

    require_current_destination(expected)

    return RevalidatedExtraction(
        plan=plan,
        target=target,
        force=force,
        expected_revision=rendered_revision,
    )


def run_extraction_lifecycle(
    config: ProjectConfig,
    journal: operations.OperationJournal,
    revalidated: RevalidatedExtraction,
    *,
    operation_id: str,
    transport: ExtractionTransport,
    progress: Callable[[str], None] | None = None,
    provider_spawn: Any = subprocess.Popen,
) -> ExtractionOutcome:
    """Make one already-authorized call and turn its answer into staging.

    The half after the authority, and the only one there is.  A single
    dispatch and one child of a batch differ in how they came by
    ``operation_id`` — an exclusive authorization or a reserved batch claim —
    and in nothing else, so the capture, the failure classification and the
    completion happen here once rather than in each caller.
    """
    plan = revalidated.plan
    target = revalidated.target
    _report_extraction_progress(progress, "Preparing pages")
    _report_extraction_progress(progress, "Reading the source")
    captured = capture_hook(
        config, journal, operation_id, provenance=target.provenance
    )
    shape_reported = False

    def frame(payload: str) -> None:
        # The exact streamed text, spooled as it arrives and before anything
        # parses it: an interrupted call still leaves what was received.
        journal.append_response_frame(operation_id, payload)

    def capture(response: object) -> None:
        nonlocal shape_reported
        captured(response)
        if not shape_reported:
            _report_extraction_progress(progress, "Checking the answer's shape")
            shape_reported = True

    try:
        result = run_extraction_call(
            transport,
            plan,
            target,
            capture=capture,
            frame=frame,
            spawn=provider_spawn,
        )
    except Exception as exc:  # noqa: BLE001 - settle every dispatched call
        try:
            failure = classify_dispatch_failure(config, journal, operation_id, exc)
        except JankiError as journal_error:
            raise ExtractionDispatchError(
                exc,
                phase="dispatch",
                operation_id=operation_id,
                journal_error=journal_error,
            ) from exc
        raise ExtractionDispatchError(
            exc,
            phase="dispatch",
            operation_id=operation_id,
            failure=failure,
        ) from exc

    # A provider wrapper is required to capture before it returns.  Keep the
    # user-facing state complete if a test double or future wrapper returns a
    # parsed result without calling the hook; complete_extraction will still
    # durably settle that normalized answer before any staging write.
    if not shape_reported:
        _report_extraction_progress(progress, "Checking the answer's shape")
    _report_extraction_progress(progress, "Saving proposals")
    try:
        return complete_extraction(
            config,
            journal,
            target,
            result,
            operation_id=operation_id,
            known=plan.known,
            mode=plan.mode,
            model=plan.model,
            force=revalidated.force,
            expected_revision=revalidated.expected_revision,
        )
    except JankiError as exc:
        raise ExtractionDispatchError(
            exc,
            phase="completion",
            operation_id=operation_id,
        ) from exc


def dispatch_extraction(
    config: ProjectConfig,
    expected: ExtractionDispatchExpectation,
    *,
    client: Any | None = None,
    progress: Callable[[str], None] | None = None,
    provider_env: Mapping[str, str] | None = None,
    provider_runner: Callable[..., Any] = subprocess.run,
    provider_which: Callable[..., str | None] = shutil.which,
    provider_spawn: Any = subprocess.Popen,
) -> ExtractionOutcome:
    """Re-plan and run one exact owner-confirmed corpus extraction.

    This is the shared paid boundary used by every surface.  The expectation
    is what the owner saw; none of its planned objects is dispatched.  The
    source, prompts, collection and replacement state are read again, the
    exact request identity is compared, and only then may the journal's locked
    authorization gate let a provider call begin.

    The progress callback receives state names only.  Providers do not report
    percentages, so this service never invents one.
    """
    revalidated = revalidate_extraction_request(
        config,
        expected,
        provider_env=provider_env,
        provider_runner=provider_runner,
        provider_which=provider_which,
    )

    # One selected transport, proved before the journal can record authority.
    # A local failure here — no CLI, a logged-out login, a missing key — is a
    # refusal on the transport that was chosen; it is never a reason to reach
    # the other one.
    try:
        transport = prepare_extraction_transport(
            revalidated.plan,
            revalidated.target,
            client=client,
            env=provider_env,
            runner=provider_runner,
            which=provider_which,
        )
    except JankiError as exc:
        raise ExtractionDispatchError(exc, phase="preparation") from exc

    try:
        journal = operations.OperationJournal.load(config.operations_file)
        # `busy_refusal` is a render-time display.  The real one-call gate is
        # OperationJournal.authorize inside authorize_dispatch, under its own
        # file lock, so two stale pages cannot both spend.
        operation_id = authorize_dispatch(
            journal,
            revalidated.target,
            model=revalidated.plan.model,
            stream=transport.prepared is not None,
        )
    except JankiError as exc:
        raise ExtractionDispatchError(exc, phase="authorization") from exc

    return run_extraction_lifecycle(
        config,
        journal,
        revalidated,
        operation_id=operation_id,
        transport=transport,
        progress=progress,
        provider_spawn=provider_spawn,
    )
