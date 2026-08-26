"""Read-only structural actions for one exact staged-record snapshot.

The workbench already has a byte-bound :class:`ReviewPanel`; this projection
accepts the records from that one snapshot rather than opening the staging
path again.  It combines janki's existing structural authorities into the
learner actions W4.1 renders: local validation, exact-example approval,
durable reading holds, and prospective ownership by the real deck selectors.

Nothing here reads Japanese, writes a file, or decides a review.  Learner
labels are paired with the stable codes and raw reasons that belong under the
page's Technical details disclosure.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from japanese_anki import staging, validation
from japanese_anki.application.assignment import (
    DeckOwnershipEvaluation,
    evaluate_prospective_deck_ownership,
)
from japanese_anki.application.authority import (
    example_authority_state,
    needs_example_review,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.models import VocabularyRecord

__all__ = [
    "CardCheck",
    "CardCheckAction",
    "CardCheckDetail",
    "CardCheckReport",
    "CheckActionKind",
    "check_cards",
]


CheckActionKind = Literal[
    "edit",
    "identity",
    "review",
    "deck",
    "hold",
    "repair",
]

_IDENTITY_ISSUES = frozenset(
    {
        "duplicate-id",
        "furigana-without-reading",
        "kanji-reading",
        "missing-expression",
        "missing-id",
        "missing-reading",
        "readingless-id",
    }
)


@dataclass(frozen=True, slots=True)
class CardCheckDetail:
    """One copyable machine fact supporting a learner-facing action."""

    code: str
    reason: str
    level: str


@dataclass(frozen=True, slots=True)
class CardCheckAction:
    """What the learner can do, with machinery kept in ``details``."""

    kind: CheckActionKind
    label: str
    details: tuple[CardCheckDetail, ...]


@dataclass(frozen=True, slots=True)
class CardCheck:
    """One supplied record and its actions, at its stable snapshot position."""

    index: int
    record: VocabularyRecord
    actions: tuple[CardCheckAction, ...]


@dataclass(frozen=True, slots=True)
class CardCheckReport:
    """The ordered card projection plus any source-wide validator facts."""

    source: str
    cards: tuple[CardCheck, ...]
    general: tuple[CardCheckDetail, ...] = ()


@dataclass(slots=True)
class _Action:
    kind: CheckActionKind
    label: str
    details: list[CardCheckDetail]


def _add_action(
    actions: list[_Action],
    kind: CheckActionKind,
    label: str,
    detail: CardCheckDetail,
) -> None:
    """Coalesce two mechanisms naming the same human action.

    A missing reading is both a durable hold and a validation error. Showing
    two **Add a reading** controls would pretend there are two decisions; the
    one action instead carries both technical facts in their discovered order.
    """
    for action in actions:
        if action.kind == kind and action.label == label:
            action.details.append(detail)
            return
    actions.append(_Action(kind=kind, label=label, details=[detail]))


def _validation_action(
    code: str,
    *,
    reidentifiable: bool,
) -> tuple[CheckActionKind, str]:
    if code in _IDENTITY_ISSUES and not reidentifiable:
        return "hold", "Keep this card for another identity decision"
    if code in {"missing-reading", "furigana-without-reading"}:
        return "identity", "Add a reading"
    if code in {"missing-expression", "missing-id"}:
        return "identity", "Re-identify this card"
    if code == "readingless-id":
        return "identity", "Re-identify this card"
    if code == "kanji-reading":
        return "identity", "Correct this reading"
    if code == "missing-meaning":
        return "edit", "Add a meaning"
    if code.startswith("example-"):
        return "edit", "Correct this example"
    if code == "duplicate-id":
        return "identity", "Re-identify or remove this duplicate"
    return "edit", "Correct this field"


def _hold_action(
    reason: str,
    *,
    reidentifiable: bool,
) -> tuple[CheckActionKind, str, str]:
    if not reidentifiable and reason in {
        staging.HOLD_MISSING_READING,
        staging.HOLD_READING_KANJI,
    }:
        return (
            "hold",
            "Keep this card for another identity decision",
            "reading-hold",
        )
    if reason == staging.HOLD_MISSING_READING:
        return "identity", "Add a reading", "reading-hold"
    if reason == staging.HOLD_READING_KANJI:
        return "identity", "Correct this reading", "reading-hold"
    if reason == staging.HOLD_UNKNOWN_READING:
        return (
            "hold",
            "This reading is not listed by jpdb; keep it for another decision.",
            "reading-hold",
        )
    if reason == staging.HOLD_UNVERIFIABLE_ID:
        return (
            "hold",
            "Keep this card for another identity decision",
            "identity-hold",
        )
    # Hand-written hold reasons deliberately count as reading holds unless
    # they are the one named non-reading reason. Preserve that default-deny
    # direction from staging.NON_READING_HOLDS.
    return "hold", "Keep this card for another reading decision", "reading-hold"


def _review_label(record: VocabularyRecord) -> str:
    count = sum(bool(example.japanese) for example in record.examples)
    if count == 1:
        return "Review the Japanese example"
    if count == 2:
        return "Review both Japanese examples"
    return f"Review all {count} Japanese examples"


def _ownership_action(
    evaluation: DeckOwnershipEvaluation,
) -> tuple[CheckActionKind, str, CardCheckDetail] | None:
    if evaluation.state == "exactly_one":
        return None
    if evaluation.state == "unassigned":
        return (
            "deck",
            "Choose a study deck",
            CardCheckDetail(
                code="deck-unassigned",
                reason="No configured word deck selects this card.",
                level="decision",
            ),
        )
    if evaluation.state == "multiple":
        owners = ", ".join(owner.name for owner in evaluation.owners)
        return (
            "deck",
            "Choose a study deck",
            CardCheckDetail(
                code="deck-overlap",
                reason=f"More than one word deck selects this card: {owners}.",
                level="decision",
            ),
        )
    return (
        "repair",
        "Repair the study deck configuration",
        CardCheckDetail(
            code="deck-ownership-unreadable",
            reason="; ".join(evaluation.unreadable_decks)
            or "The configured word decks could not be evaluated.",
            level="error",
        ),
    )


def check_cards(
    config: ProjectConfig,
    records: Sequence[VocabularyRecord],
    *,
    source: str | Path = "",
    reidentifiable: bool = True,
    approvable: bool = True,
) -> CardCheckReport:
    """Project actions from one caller-supplied, already-bound snapshot.

    ``records`` is never loaded again and is never mutated. Deck rules and the
    canonical collection are necessarily live inputs: ownership answers where
    these exact proposed records would land *now*, which is the same question
    the final promotion gate re-proves under its lock.
    """
    snapshot = tuple(records)
    source_text = str(source)
    issues = validation.validate_records(list(snapshot), source_text)
    by_index: dict[int, list[validation.ValidationIssue]] = {}
    general: list[CardCheckDetail] = []
    for issue in issues:
        if issue.record_index is not None:
            by_index.setdefault(issue.record_index, []).append(issue)
        else:
            general.append(
                CardCheckDetail(
                    code=issue.code,
                    reason=issue.message,
                    level=issue.level,
                )
            )

    id_counts = Counter(record.id for record in snapshot)
    hold_reasons = tuple(
        staging.annotations(record).get("hold_reason", "") for record in snapshot
    )
    identity_invalid = {
        index
        for index, row_issues in by_index.items()
        if any(issue.code in _IDENTITY_ISSUES for issue in row_issues)
    }
    eligible = tuple(
        (index, record)
        for index, record in enumerate(snapshot)
        if (
            record.id
            and id_counts[record.id] == 1
            and index not in identity_invalid
            and not hold_reasons[index]
        )
    )
    ownership_by_index: dict[int, DeckOwnershipEvaluation] = {}
    try:
        ownership = evaluate_prospective_deck_ownership(
            config,
            tuple(record for _index, record in eligible),
        )
    except JankiError as exc:
        ownership = tuple(
            DeckOwnershipEvaluation(
                record_id=record.id,
                state="unreadable",
                memberships=(),
                unreadable_decks=(str(exc),),
            )
            for _index, record in eligible
        )
    ownership_by_index.update(
        (index, evaluation)
        for (index, _record), evaluation in zip(eligible, ownership, strict=True)
    )

    cards: list[CardCheck] = []
    for index, record in enumerate(snapshot):
        actions: list[_Action] = []
        hold_reason = hold_reasons[index]
        if hold_reason:
            kind, label, code = _hold_action(
                hold_reason,
                reidentifiable=reidentifiable,
            )
            _add_action(
                actions,
                kind,
                label,
                CardCheckDetail(code=code, reason=hold_reason, level="decision"),
            )

        for issue in by_index.get(index, ()):
            kind, label = _validation_action(
                issue.code,
                reidentifiable=reidentifiable,
            )
            _add_action(
                actions,
                kind,
                label,
                CardCheckDetail(
                    code=issue.code,
                    reason=issue.message,
                    level=issue.level,
                ),
            )

        authority = example_authority_state(record)
        if needs_example_review(record):
            kind: CheckActionKind = "review" if approvable else "hold"
            label = (
                _review_label(record)
                if approvable
                else "Keep these examples for another review decision"
            )
            _add_action(
                actions,
                kind,
                label,
                CardCheckDetail(
                    code=f"example-review-{authority}",
                    reason=(
                        "The previous approval no longer covers every Japanese "
                        "example shown."
                        if authority == "stale"
                        else "These exact Japanese examples have not been approved."
                    ),
                    level="decision",
                ),
            )
        elif authority == "invalid":
            _add_action(
                actions,
                "repair",
                "Repair the example approval",
                CardCheckDetail(
                    code="example-authority-invalid",
                    reason="The stored example approval is not a recognized value.",
                    level="error",
                ),
            )

        deck = ownership_by_index.get(index)
        deck_action = _ownership_action(deck) if deck is not None else None
        if deck_action is not None:
            kind, label, detail = deck_action
            _add_action(actions, kind, label, detail)

        cards.append(
            CardCheck(
                index=index,
                record=record,
                actions=tuple(
                    CardCheckAction(
                        kind=action.kind,
                        label=action.label,
                        details=tuple(action.details),
                    )
                    for action in actions
                ),
            )
        )

    return CardCheckReport(
        source=source_text,
        cards=tuple(cards),
        general=tuple(general),
    )
