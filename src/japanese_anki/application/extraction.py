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

import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from japanese_anki import claude_client, extract, inputs, operations, patterns, prompts
from japanese_anki.application.journey import source_journeys
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.inputs import PreparedInput
from japanese_anki.io import exclusive_path_lock, load_records
from japanese_anki.models import VocabularyRecord
from japanese_anki.staging import (
    CANDIDATE_ACCOUNTING_KEY,
    new_review_run_id,
    write_staging,
)

__all__ = [
    "ANSWER_EMPTY",
    "ANSWER_SAVED",
    "OUTCOME_UNKNOWN",
    "DispatchFailure",
    "ExtractionConsent",
    "ExtractionOutcome",
    "ExtractionPlan",
    "ExtractionTarget",
    "authorize_dispatch",
    "busy_refusal",
    "capture_hook",
    "classify_dispatch_failure",
    "complete_extraction",
    "describe_extraction",
    "durable_inbox_root",
    "plan_extraction",
    "settle_dispatch",
]


@dataclass(frozen=True, slots=True)
class ExtractionTarget:
    """One input, and everything decided about it before the call."""

    item: PreparedInput
    #: The staging file this input's proposals would be written to.
    staging_path: Path
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


#: What became of a call that failed after the money may already have gone.
#:
#: Three states, and the difference between them is what the person is owed.
#: `answer_saved` — the bytes are on disk and something after them refused, so
#: there is paid content to look at. `answer_empty` — a reply arrived carrying
#: only the model's reasoning, so it was billed and holds nothing to recover;
#: calling that "the answer was saved" sends somebody hunting for cards in a
#: file with none. `outcome_unknown` — it was sent and no answer was captured,
#: which is the one where a retry risks a second charge.
ANSWER_SAVED = "answer_saved"
ANSWER_EMPTY = "answer_empty"
OUTCOME_UNKNOWN = "outcome_unknown"


@dataclass(frozen=True, slots=True)
class DispatchFailure:
    """A dispatch that did not produce usable cards, and what it cost."""

    operation_id: str
    #: One of `ANSWER_SAVED`, `ANSWER_EMPTY`, `OUTCOME_UNKNOWN`.
    outcome: str
    #: The stored response, when one was captured.
    artifact: str = ""
    #: Whether a retry risks paying twice. Read from the journal rather than
    #: guessed from the exception — it is the question a retry turns on.
    #:
    #: True for every `OUTCOME_UNKNOWN` today, and by construction rather than
    #: by accident: the transition table only allows that state from a
    #: dispatched one. It stays a field because it is the question W3's
    #: recovery surface asks, and answering it from the journal keeps it true
    #: if the table ever grows a not-billed terminal state.
    money_may_have_been_spent: bool = False

    @property
    def was_paid_for(self) -> bool:
        return self.outcome in (ANSWER_SAVED, ANSWER_EMPTY)


#: States that mean a paid call is in flight right now. `authorized` is not
#: one of them: it says the authority was written and nothing was sent, which
#: is where a process that died before dispatching leaves an entry — and a
#: guard that counted it would wedge every later run behind a call that never
#: happened.
IN_FLIGHT: frozenset[str] = frozenset({"dispatching", "running"})


def busy_refusal(config: ProjectConfig) -> str:
    """Why janki will not start a paid run right now, in a person's words.

    W3: *"One mutating job at a time."* The answer comes from the journal
    rather than from a flag this process keeps, because a run started in a
    terminal is exactly as real as one started in a browser tab, and a guard
    that only knew about its own process would let the two overlap and bill
    twice.

    Two different refusals, deliberately. A call in flight is a *wait*. An
    answer that may already have been billed is a *decision*, and starting a
    second call would bury it.

    **This is a display, not the enforcement point.** It is read when a page
    renders and acted on when a button is clicked, so two callers can both
    pass it and both authorize. Whatever actually spends money has to refuse
    under the journal's own lock.
    """
    journal = operations.OperationJournal.load(config.operations_file)
    flying = [op for op in journal.unfinished() if op.state in IN_FLIGHT]
    if flying:
        first = flying[0]
        return (
            f"A call about {first.source_file} is still marked as running, so "
            "janki will not start another — that would risk a second charge. "
            "If nothing is actually running, that call was interrupted; "
            "'janki status --operations' shows it."
        )
    waiting = journal.needing_attention()
    if waiting:
        first = waiting[0]
        # "may have been billed", not "has been paid for": `outcome_unknown`
        # is in this list precisely because nobody knows, and a page that
        # asserted the charge would be guessing about someone's money in the
        # one place it must not.
        return (
            f"A call about {first.source_file} may have been billed and has "
            "not been dealt with. 'janki status --operations' shows what is "
            "known about it."
        )
    return ""


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
    root = durable_inbox_root(config)
    if not inputs.inside(source, root):
        return ExtractionConsent(
            name=source.name,
            model=chosen,
            mode=mode,
            refusal=(
                f"{source.name} is not in your corpus yet. Add it first — "
                "adding a file and sending it to a model are separate steps."
            ),
        )

    try:
        prepared = inputs.prepare_inputs(
            [source], config.scan_inbox, inbox_root=root
        )
        # A consent page must not be the thing that puts a file in the corpus.
        # The root check above says it is already there; this says the prepare
        # agreed — closing the sliver where the file is removed in between and
        # `_copy_into_inbox` helpfully writes it back.
        if prepared[0].copied:
            return ExtractionConsent(
                name=source.name,
                model=chosen,
                mode=mode,
                refusal=(
                    f"{source.name} moved while this page was loading. "
                    "Reload to see where it is now."
                ),
            )
        plan = plan_extraction(
            config,
            prepared,
            mode=mode,
            model=chosen,
            style_guide=claude_client.read_style_guide(config.root),
            system=prompts.load(config.root, extract.prompt_name(mode)),
            # Planned as though forced so a staging collision comes back as a
            # replacement to confirm rather than an exception. Consenting to
            # the replacement is what supplies `force` to the dispatch; this
            # value never sends anything.
            force=True,
        )
        replaces = plan.targets[0].replaces if plan.targets else None
        state, cards = "", 0
        if replaces is not None:
            found = next(
                (
                    journey
                    for journey in source_journeys(config)[0]
                    if journey.staging_path == replaces
                ),
                None,
            )
            if found is not None:
                state, cards = found.state, found.card_count
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
        relative = operations.capture_artifact(
            config.operations_file,
            operation_id,
            operations.serialize_response(response),
        )
        journal.advance(operation_id, "result_captured", artifact=relative)

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
    own reply, so `operations.answer_text` — which reads provider-shaped
    content blocks — finds nothing in it and reports no answer. What it holds
    is still readable by a person, which is the promise; a surface that grades
    artifacts through `answer_text` must not call one of these empty.
    """
    reloaded = operations.OperationJournal.load(config.operations_file)
    held = reloaded.operations.get(operation_id)
    if held is None or held.state not in ("dispatching", "running"):
        return
    journal.advance(
        operation_id,
        "result_captured",
        artifact=operations.capture_artifact(
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
    """
    held = operations.OperationJournal.load(config.operations_file).operations.get(
        operation_id
    )
    answer = (
        operations.answer_text(config.operations_file, held)
        if held is not None and held.artifact
        else ""
    )
    if held is not None and held.state == "result_captured":
        return DispatchFailure(
            operation_id=operation_id,
            outcome=ANSWER_SAVED if answer else ANSWER_EMPTY,
            artifact=held.artifact,
        )
    marked = journal.advance(operation_id, "outcome_unknown", detail=str(exc))
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
    held = operations.OperationJournal.load(config.operations_file).operations.get(
        operation_id
    )
    if held is not None and held.request_fp != str(
        target.provenance["request_fingerprint"]
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
    refusal = (
        f"No operation {operation_id!r} to complete"
        if entry is None
        else operations.advance_refusal(
            operation_id, entry.state, "committed", entry.artifact
        )
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
    target.staging_path.parent.mkdir(parents=True, exist_ok=True)
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
    write_staging(target.staging_path, records, meta, force=force)
    # The exact answer became staging, so the journal entry has done its job
    # and stops being something a person must look at.
    journal.advance(operation_id, "committed")

    kept_reviewed_patterns = False
    with exclusive_path_lock(config.patterns_file):
        store = patterns.load_store(config.patterns_file)
        previous = store.get(run_patterns.source)
        if previous is not None and previous.reviewed and not force:
            kept_reviewed_patterns = True
        else:
            store[run_patterns.source] = run_patterns
            patterns.save_store_under_lock(config.patterns_file, store)

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
