"""The Assistant's local extraction-batch desk.

Reads durable batch state through ``application.extraction_batch`` and turns it
into the rows, controls and confirmation sentences the ChatKit controller
renders. Nothing here writes to the journal, decides which source may be sent
again, retires evidence, or combines Japanese: the core owns all four.

Eligibility in particular is read, never inferred. ``resume_available`` and
``resume_refusal`` come from the core outcome as they are, so a control is
offered exactly when the core says continuing is possible and the reason is
shown when it is not.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki.workbench.assistant import (
    ExtractionBatchActionChoice,
    ExtractionBatchChildStatus,
    ExtractionBatchChoice,
    ExtractionBatchPreviewOffer,
)

# The states a source may be sent again from. Not "the call did not happen":
# a captured reply and an unknown outcome are both retry-eligible, and both may
# already have cost money. The core's ``plan_extraction_batch_retry`` remains
# the authority — this only decides which buttons are worth showing, and it
# touches no journal. A live child (authorized, dispatching, running), a
# committed one, and a retired or unreserved one are all absent on purpose.
RETRY_STATES = frozenset(
    {
        "failed_before_send",
        "canceled_before_send",
        "expired",
        "result_captured",
        "outcome_unknown",
    }
)
_RETRY_LABELS = {
    "result_captured": (
        "Discard the saved reply for source {index} and send again ({name})"
    ),
    "outcome_unknown": (
        "Send source {index} again ({name}) — its earlier outcome and cost "
        "are unknown"
    ),
}


def core() -> Any:
    return importlib.import_module("japanese_anki.application.extraction_batch")


def _short(batch_id: str) -> str:
    return batch_id.split("-")[0]


def _name(value: Any) -> str:
    return Path(str(value)).name


@dataclass(frozen=True, slots=True)
class PreparedExtractionBatch:
    """One exact core plan, held whole between rendering it and confirming it.

    The plan is stored rather than its ingredients: the core's fingerprint
    covers the complete manifest, including the batch and child operation ids
    it already minted. Re-planning at confirmation time would mint new ones and
    compare a fingerprint against a plan that no longer exists.
    """

    plan: Any
    fingerprint: str
    target: str
    effects: tuple[str, ...]
    disclosures: tuple[str, ...]
    confirm_label: str


def _child_lines(plan: Any) -> tuple[str, ...]:
    return tuple(
        f"{child.index}. {_name(child.source)}" for child in plan.children
    )


def plan_batch(
    config: Any,
    sources: Sequence[Path],
    *,
    mode: str | None = None,
    scope_id: str = "",
    destination_deck: Path | None = None,
    concurrency_limit: int = 2,
    job_id: str = "",
) -> PreparedExtractionBatch:
    """Plan one batch over exactly these preserved sources. Nothing is sent.

    This is the jobless Assistant route: it names sources the owner disclosed,
    not a study job's bound parts, so a layout-bound mode has nothing to send
    and is refused before the plan.
    """

    from japanese_anki import extract

    extract.refuse_unbound_layout_mode(mode, control="This extraction batch")
    return describe_batch(
        core().plan_extraction_batch(
            config,
            list(sources),
            mode=mode,
            scope_id=scope_id,
            destination_deck=destination_deck,
            concurrency_limit=concurrency_limit,
            job_id=job_id,
        )
    )


def _mode_line(plan: Any) -> str:
    """The one scalar mode this batch carries, named in its confirmation.

    Read off the children's own frozen expectations rather than recomputed:
    what the owner is agreeing to send is what the plan already froze. A batch
    whose children somehow disagreed would say so rather than name one of them.
    """

    modes = {child.expectation.mode or "auto" for child in plan.children}
    return f"Mode: {', '.join(sorted(modes))}"


def _layout_lines(plan: Any) -> tuple[str, ...]:
    """Each child's bound layout revision, or nothing when none is bound."""

    return tuple(
        f"{child.index}. {_name(child.source)} — layout "
        f"{child.expectation.table_layout.identity}"
        for child in plan.children
        if child.expectation.table_layout is not None
    )


def describe_batch(plan: Any) -> PreparedExtractionBatch:
    """Render one already-planned core batch as its owner confirmation.

    Separate from :func:`plan_batch` because a study job plans its batch over
    its own published parts through its own service, and re-planning here to
    describe it would mint a second set of operation ids and compare a
    confirmation against a plan that no longer exists.

    The pinned mode and every bound layout revision are named here, because
    they are part of what one confirmation buys: a layout-bound child asks for
    exactly the columns that revision declares, and nobody should confirm a
    request whose shape the confirmation did not state.
    """

    listed = _child_lines(plan)
    bound = _layout_lines(plan)
    effects = (
        f"Send these {len(plan.children)} whole sources to {plan.model}, "
        f"{plan.concurrency_limit} at a time:",
        *listed,
        "Propose vocabulary cards and grammar for owner review, one staging "
        "file per source",
    )
    disclosures = (
        f"These are {len(plan.children)} paid model calls through "
        f"{plan.provider}, not one.",
        _mode_line(plan),
        *(
            (
                "Each of these sources is sent under the printed-column layout "
                "revision you bound to it:",
                *bound,
            )
            if bound
            else ()
        ),
        "Each source is sent whole. Page ranges, row counts or other scope "
        "typed in chat are not applied.",
        "A source that fails leaves the ones that succeeded saved.",
    )
    return PreparedExtractionBatch(
        plan=plan,
        fingerprint=plan.fingerprint,
        target=f"{len(plan.children)} sources",
        effects=effects,
        disclosures=disclosures,
        confirm_label=f"Send these {len(plan.children)} sources — paid model calls",
    )


def plan_batch_retry(
    config: Any,
    batch_id: str,
    child_indices: Sequence[int],
    job_id: str = "",
) -> PreparedExtractionBatch:
    """Plan a retry of exactly these sources, disclosing both halves of it.

    One confirmation carries both what will be sent and what recorded evidence
    the core will retire to make room for it. This surface never retires
    anything itself and never mints a retry fingerprint.

    A retry inherits its job from the loaded manifest. ``job_id`` states which
    job the caller believes this batch belongs to, and the core refuses a
    disagreement; it never re-parents the retry.
    """

    return describe_batch_retry(
        core().plan_extraction_batch_retry(
            config, batch_id, list(child_indices), job_id=job_id
        )
    )


def describe_batch_retry(plan: Any) -> PreparedExtractionBatch:
    """Render one already-planned retry as its own fresh owner confirmation."""

    listed = _child_lines(plan)
    retired = tuple(
        f"Retire the earlier model call {discard.operation_id} for "
        f"{discard.source_file} — {discard.state}"
        + (f", {discard.detail}" if discard.detail else "")
        for discard in plan.discards
    )
    effects = (
        f"Send {len(plan.children)} source(s) again to {plan.model}, "
        f"{plan.concurrency_limit} at a time:",
        *listed,
        *retired,
    )
    states = {discard.state for discard in plan.discards}
    disclosures = []
    if "outcome_unknown" in states:
        disclosures.append(
            "One of those earlier calls may or may not have been made, so its "
            "outcome and its cost are unknown. Sending again discards what "
            "janki kept about it and makes a distinct new paid call with a new "
            "id; the old one is never sent again."
        )
    if "result_captured" in states:
        disclosures.append(
            "A reply janki already received and saved for that source is "
            "discarded. It was already paid for and cannot be recovered "
            "afterwards."
        )
    disclosures += [
        "Retiring the evidence above cannot be undone.",
        "Sources this batch already saved are not resent and not replaced.",
    ]
    return PreparedExtractionBatch(
        plan=plan,
        fingerprint=plan.fingerprint,
        # The batch being retried, which is what the owner is deciding about;
        # the fresh batch's own id is inside the plan this is bound to.
        target=f"batch {_short(plan.retry_of)}",
        effects=effects,
        disclosures=tuple(disclosures),
        confirm_label=(
            f"Retire that evidence and send {len(plan.children)} source(s) again"
        ),
    )


def _choice(outcome: Any) -> ExtractionBatchChoice:
    children = tuple(
        ExtractionBatchChildStatus(
            index=child.index,
            source_name=_name(child.source),
            state=child.state,
            detail=child.error or "",
        )
        for child in outcome.children
    )
    actions: list[ExtractionBatchActionChoice] = []
    # Recovery first. Where a reply was captured, continuing keeps the answer
    # that was already paid for, and the retry below throws it away.
    if outcome.resume_available:
        actions.append(
            ExtractionBatchActionChoice(
                action="resume",
                label="Continue this batch where it stopped",
            )
        )
    if outcome.committed_count:
        actions.append(
            ExtractionBatchActionChoice(
                action="preview",
                label="Show the cards these sources have saved",
            )
        )
    retryable = tuple(
        child for child in outcome.children if child.state in RETRY_STATES
    )
    for child in retryable:
        name = next(row.source_name for row in children if row.index == child.index)
        template = _RETRY_LABELS.get(
            child.state, "Send source {index} again ({name})"
        )
        actions.append(
            ExtractionBatchActionChoice(
                action="retry",
                label=template.format(index=child.index, name=name),
                child_indices=(child.index,),
            )
        )
    if len(retryable) > 1:
        indices = tuple(child.index for child in retryable)
        listed = ", ".join(str(index) for index in indices)
        states = {child.state for child in retryable}
        caveats = []
        if "result_captured" in states:
            caveats.append("discarding replies janki already saved")
        if "outcome_unknown" in states:
            caveats.append("where the earlier outcome and cost are unknown")
        label = f"Send sources {listed} again"
        if caveats:
            label = f"{label} — {' and '.join(caveats)}"
        actions.append(
            ExtractionBatchActionChoice(
                action="retry",
                label=label,
                child_indices=indices,
            )
        )
    summary = (
        f"{outcome.committed_count} saved, {outcome.failed_count} failed, "
        f"{outcome.unknown_count} unknown, {outcome.pending_count} waiting"
    )
    if outcome.resume_refusal:
        summary = f"{summary} — {outcome.resume_refusal}"
    return ExtractionBatchChoice(
        batch_id=outcome.batch_id,
        label=f"{len(outcome.children)}-source extraction {_short(outcome.batch_id)}",
        summary=summary,
        concurrency_limit=outcome.concurrency_limit,
        children=children,
        actions=tuple(actions),
    )


def list_batch_choices(config: Any) -> tuple[ExtractionBatchChoice, ...]:
    return tuple(_choice(outcome) for outcome in core().list_extraction_batches(config))


def batch_summary(outcome: Any) -> str:
    lines = [
        f"Batch {_short(outcome.batch_id)}: {outcome.committed_count} saved, "
        f"{outcome.failed_count} failed, {outcome.unknown_count} unknown, "
        f"{outcome.pending_count} still waiting."
    ]
    for child in outcome.children:
        line = f"{child.index}. {_name(child.source)} — {child.state}"
        if child.records is not None:
            line += f" ({child.records} proposals)"
        if child.error:
            line += f": {child.error}"
        lines.append(line)
    return "\n".join(lines)


def progress_label(event: Any, total: int) -> str:
    """The one live narration shape: which numbered source reached what."""

    return f"Source {event.index} of {total}: {event.state}"


def preview_offer(
    config: Any,
    batch_id: str,
    *,
    preview_store: Any,
    focus_scope: str,
    thread_id: str | None = None,
) -> ExtractionBatchPreviewOffer:
    """One rendered look at what this batch already saved. It accepts nothing."""

    rendered = core().render_extraction_batch_preview(config, batch_id)
    covered = ", ".join(str(index) for index in rendered.child_indices) or "none"
    conflicts = tuple(str(conflict) for conflict in rendered.conflicts)
    message = (
        f"These are the {rendered.preview.card_count} cards saved so far from "
        f"sources {covered}, drawn as Anki draws them. Looking at them changes "
        "nothing and accepts nothing."
    )
    if preview_store is None:
        return ExtractionBatchPreviewOffer(
            message=(
                f"{message} This workbench has no local preview address to serve "
                "them from, so there is nothing to open."
            ),
            preview_url=None,
            conflicts=conflicts,
        )
    offer = preview_store.offer(
        rendered.preview,
        label=f"Cards saved by extraction batch {_short(batch_id)}",
        focus_scope=focus_scope,
        thread_id=thread_id,
    )
    return ExtractionBatchPreviewOffer(
        message=message,
        preview_url=offer.url,
        conflicts=conflicts,
    )
