"""Typed services the CLI and the workbench both call.

`WORKBENCH_PLAN.md` W1.1. Nothing here renders HTML, parses `argv`, or prints;
nothing here is imported by a browser handler that then re-derives the same
answer its own way. A rule that decides whether a source is ready to promote
lives here once, so the page and the command can never disagree about it —
which is the only way the workbench can be "a safe view and controller over
the same files and operations" instead of a second opinion about them.
"""

from __future__ import annotations

from japanese_anki.application.authority import (
    AUTHORITY_STATES,
    example_authority_state,
    needs_example_review,
)
from japanese_anki.application.decks import (
    DeckMembership,
    deck_membership,
)
from japanese_anki.application.detail import (
    CardDetail,
    SourceDetail,
    source_detail,
)
from japanese_anki.application.extraction import (
    ExtractionPlan,
    ExtractionTarget,
    plan_extraction,
)
from japanese_anki.application.journey import (
    ADDED,
    CARDS_NEED_EDITS,
    COVERAGE_NEEDS_DECISION,
    EXAMPLES_NEED_REVIEW,
    GRAMMAR_NEEDS_REVIEW,
    GRAMMAR_NONE,
    GRAMMAR_ONLY,
    GRAMMAR_REVIEWED,
    GRAMMAR_UNKNOWN,
    JOURNEY_STATES,
    NOT_EXTRACTED,
    READING_HOLD,
    READY_TO_ADD,
    STAGING_UNREADABLE,
    SourceJourney,
    source_journeys,
)
from japanese_anki.application.promotion import (
    HeldCard,
    LandingCard,
    PromotionPlan,
    archive_for_run,
    archive_run_provenance,
    plan_promotion,
)

__all__ = [
    "ADDED",
    "AUTHORITY_STATES",
    "CardDetail",
    "DeckMembership",
    "CARDS_NEED_EDITS",
    "COVERAGE_NEEDS_DECISION",
    "EXAMPLES_NEED_REVIEW",
    "GRAMMAR_NEEDS_REVIEW",
    "GRAMMAR_NONE",
    "GRAMMAR_ONLY",
    "GRAMMAR_REVIEWED",
    "GRAMMAR_UNKNOWN",
    "ExtractionPlan",
    "ExtractionTarget",
    "HeldCard",
    "LandingCard",
    "PromotionPlan",
    "JOURNEY_STATES",
    "NOT_EXTRACTED",
    "READING_HOLD",
    "READY_TO_ADD",
    "STAGING_UNREADABLE",
    "SourceDetail",
    "SourceJourney",
    "archive_for_run",
    "archive_run_provenance",
    "deck_membership",
    "example_authority_state",
    "plan_extraction",
    "plan_promotion",
    "needs_example_review",
    "source_detail",
    "source_journeys",
]
