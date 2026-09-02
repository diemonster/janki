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

import contextlib
import hashlib
import os
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
from japanese_anki.application.journey import (
    GRAMMAR_NEEDS_REVIEW,
    GRAMMAR_REVIEWED,
    source_journeys,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.credential_safety import redact_environment_credentials
from japanese_anki.errors import JankiError
from japanese_anki.inputs import PreparedInput
from japanese_anki.io import (
    exclusive_path_lock,
    load_records,
    prepare_bound_directory,
    read_bytes_bound,
)
from japanese_anki.models import VocabularyRecord
from japanese_anki.staging import (
    CANDIDATE_ACCOUNTING_KEY,
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
    "authorize_dispatch",
    "busy_refusal",
    "capture_hook",
    "classify_dispatch_failure",
    "complete_extraction",
    "describe_extraction",
    "dispatch_extraction",
    "durable_inbox_root",
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

    @property
    def name(self) -> str:
        """The permanent filename, which is what a person recognizes."""
        return self.item.origin_path.name


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
    model: str
    mode: str | None
    source_sha256: str
    request_fingerprint: str
    replacement_revision: ExtractionRevision | None
    replacement_confirmed: bool
    staging_path: Path
    patterns_path: Path
    operations_path: Path

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


def plan_extraction(
    config: ProjectConfig,
    prepared: Sequence[PreparedInput],
    *,
    mode: str | None,
    model: str,
    style_guide: str,
    system: str,
    force: bool = False,
) -> ExtractionPlan:
    """Resolve everything a run can know before it spends anything.

    Every refusal that does not need a provider happens here, which is the
    point: a batch that would write two inputs to one staging file is refused
    now rather than after paying for both.
    """
    existing: list[VocabularyRecord] = (
        load_records(config.normalized_file)
        if config.normalized_file.exists()
        else []
    )
    known = frozenset(extract.known_ids(existing))
    skip_list = (
        tuple(sorted({record.expression for record in existing}))
        if mode == "prose"
        else ()
    )

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

    return ExtractionPlan(
        model=model,
        mode=mode,
        targets=tuple(
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
                ),
            )
            for item, staging_path, fingerprint in zip(
                prepared, targets, fingerprints, strict=True
            )
        ),
        style_guide=style_guide,
        system=system,
        skip_list=skip_list,
        known=known,
    )


def plan_corpus_extraction(
    config: ProjectConfig,
    source: Path,
    *,
    mode: str | None,
    model: str,
    force: bool = False,
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
        force=force,
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

    @property
    def sendable(self) -> bool:
        return self.target is not None and not self.refusal and not self.busy


def describe_extraction(
    config: ProjectConfig,
    source: Path,
    *,
    mode: str | None = None,
    model: str | None = None,
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
            name=source.name, model=chosen, mode=mode, refusal=str(exc)
        )

    return ExtractionConsent(
        name=source.name,
        model=chosen,
        mode=mode,
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
    journal.advance(operation_id, "dispatching")
    return operation_id


def capture_hook(
    config: ProjectConfig,
    journal: operations.OperationJournal,
    operation_id: str,
) -> Callable[[Any], None]:
    """The client's capture callback: persist the exact reply, then journal it.

    In that order. The bytes are what a person can still act on if parsing
    refuses, so they reach disk before anything says they arrived.
    """

    def _capture(response: Any) -> None:
        journal.capture_result(
            operation_id,
            lambda: operations.capture_artifact(
                config.operations_file,
                operation_id,
                operations.serialize_response(response),
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
    built = extract.build_records(result.candidates, item, known)
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


def dispatch_extraction(
    config: ProjectConfig,
    expected: ExtractionDispatchExpectation,
    *,
    client: Any | None = None,
    progress: Callable[[str], None] | None = None,
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
            force=force,
        )
    except JankiError as exc:
        raise ExtractionDispatchError(exc, phase="binding") from exc
    if len(plan.targets) != 1:
        cause = staging.StagingError(
            "That source no longer makes one extraction request."
        )
        raise ExtractionDispatchError(cause, phase="binding")

    target = plan.targets[0]
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

    if client is None:
        try:
            client = claude_client.prepare_paid_client()
        except JankiError as exc:
            raise ExtractionDispatchError(exc, phase="preparation") from exc

    try:
        journal = operations.OperationJournal.load(config.operations_file)
        # `busy_refusal` is a render-time display.  The real one-call gate is
        # OperationJournal.authorize inside authorize_dispatch, under its own
        # file lock, so two stale pages cannot both spend.
        operation_id = authorize_dispatch(journal, target, model=plan.model)
    except JankiError as exc:
        raise ExtractionDispatchError(exc, phase="authorization") from exc

    _report_extraction_progress(progress, "Preparing pages")
    _report_extraction_progress(progress, "Reading the source")
    captured = capture_hook(config, journal, operation_id)
    shape_reported = False

    def capture(response: object) -> None:
        nonlocal shape_reported
        captured(response)
        if not shape_reported:
            _report_extraction_progress(progress, "Checking the answer's shape")
            shape_reported = True

    try:
        result = extract.extract_candidates(
            target.item,
            model=plan.model,
            style_guide=plan.style_guide,
            system=plan.system,
            mode=plan.mode,
            known=plan.skip_list,
            client=client,
            capture=capture,
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
            force=force,
            expected_revision=rendered_revision,
        )
    except JankiError as exc:
        raise ExtractionDispatchError(
            exc,
            phase="completion",
            operation_id=operation_id,
        ) from exc
