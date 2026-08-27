"""What each source is waiting for, derived from the repository alone.

`WORKBENCH_PLAN.md` W1.1. The dashboard needs one sentence per source — where
it is, and the single next thing to do — and that answer has to come from the
same files the CLI reads: `data/inbox/`, staging, the pattern store. Never from
a browser-side database, because refreshing the page or restarting the server
has to reconstruct the queue exactly. A state that only a running process knows
is a state that can claim a paid call or a review happened when it did not.

**Two tracks, in parallel, neither blocking the other.** Word cards move
staged → reviewed → promoted → built. Grammar moves extracted → reviewed. A
te-form chart teaches only patterns and has no cards to build, so it must not
sit forever in a deck-shaped queue; a vocabulary table teaches no grammar and
must not wait on a pattern review that will never come. Reporting one combined
state is what traps both.

**The priority is structural.** Holds, then edits, then example review, then
coverage, deck ownership, then add. It ranks by which gate stops the source
first, and never by reading the Japanese — picking "this looks like the
important word" is the rules-engine anti-pattern `docs/DESIGN.md` names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from japanese_anki import extract, patterns, staging, validation
from japanese_anki.application.assignment import (
    evaluate_prospective_deck_ownership,
)
from japanese_anki.application.authority import needs_example_review
from japanese_anki.application.finish import FinishScopeError, list_finish_receipts
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.models import VocabularyRecord

__all__ = [
    "ADDED",
    "CARDS_NEED_EDITS",
    "COVERAGE_NEEDS_DECISION",
    "DECK_NEEDS_DECISION",
    "EXAMPLES_NEED_REVIEW",
    "FINISH_ARCHIVE_UNREADABLE",
    "GRAMMAR_NEEDS_REVIEW",
    "GRAMMAR_NONE",
    "GRAMMAR_ONLY",
    "GRAMMAR_REVIEWED",
    "GRAMMAR_UNKNOWN",
    "JOURNEY_STATES",
    "NOT_EXTRACTED",
    "PATTERN_STORE_UNREADABLE",
    "READING_HOLD",
    "READY_TO_ADD",
    "STAGING_UNREADABLE",
    "SourceJourney",
    "source_journeys",
]

# --- the words on the dashboard ---------------------------------------------
#
# Learner-facing by contract (WORKBENCH_PLAN.md, "The words on the screen").
# Not `staged`, `promote`, `coverage-unresolved` or a validation code: those
# are the vocabulary of the machinery, and they belong under Technical details.

NOT_EXTRACTED = "In corpus — not yet extracted"
READING_HOLD = "Some cards held for a reading decision"
CARDS_NEED_EDITS = "Cards need edits"
EXAMPLES_NEED_REVIEW = "Examples need review"
COVERAGE_NEEDS_DECISION = "Coverage needs a decision"
DECK_NEEDS_DECISION = "Deck needs a decision"
READY_TO_ADD = "Ready to add"
GRAMMAR_ONLY = "Grammar saved — no word cards to build"
ADDED = "Added — dictionary/audio/build steps remain"
STAGING_UNREADABLE = "Needs attention — this source's file could not be read"
FINISH_ARCHIVE_UNREADABLE = (
    "Needs attention — completed-card archive could not be verified"
)
PATTERN_STORE_UNREADABLE = (
    "Needs attention — grammar history could not be verified"
)

#: Card-track states, in the structural priority order that picks one action.
JOURNEY_STATES: tuple[str, ...] = (
    FINISH_ARCHIVE_UNREADABLE,
    STAGING_UNREADABLE,
    PATTERN_STORE_UNREADABLE,
    NOT_EXTRACTED,
    READING_HOLD,
    CARDS_NEED_EDITS,
    EXAMPLES_NEED_REVIEW,
    COVERAGE_NEEDS_DECISION,
    DECK_NEEDS_DECISION,
    GRAMMAR_ONLY,
    READY_TO_ADD,
    ADDED,
)

# Grammar track. Independent badge; never folded into the card state above.
GRAMMAR_NONE = ""
GRAMMAR_NEEDS_REVIEW = "Grammar needs review"
GRAMMAR_REVIEWED = "Grammar reviewed"
GRAMMAR_UNKNOWN = "Grammar review state unknown"


@dataclass(frozen=True, slots=True)
class SourceJourney:
    """One source's place in both tracks, and the one thing to do next."""

    source: str
    state: str
    next_action: str
    grammar: str = GRAMMAR_NONE
    staging_path: Path | None = None
    card_count: int = 0
    held_count: int = 0
    example_review_count: int = 0
    invalid_count: int = 0
    deck_decision_count: int = 0
    detail: str = ""
    grammar_detail: str = ""
    finish_receipt_ids: tuple[str, ...] = ()

    @property
    def needs_a_person(self) -> bool:
        """Whether anything is still left for a human to do here.

        Every state carries a next action except one: a pattern-only source
        whose grammar has been read. `Ready to add` is waiting for someone to
        add the cards and `Added` is waiting for the build steps, so counting
        either as finished puts a reassuring number at the top of a page that
        still has work on it — which is the specific way a dashboard starts
        lying to the person reading it.
        """
        if self.state == GRAMMAR_ONLY:
            return self.grammar in {GRAMMAR_NEEDS_REVIEW, GRAMMAR_UNKNOWN}
        return True


def _held(record: VocabularyRecord) -> bool:
    """Whether a reading decision is what this row is waiting on.

    `promote` writes `hold_reason` and clears it on resolution, so its presence
    is the durable mark — not an inference from a blank reading, which would
    also catch rows nobody has run through the reading gate yet.
    """
    return bool(staging.annotations(record).get("hold_reason"))


def _grammar_state(
    meta: Mapping[str, Any],
    store: Mapping[str, patterns.PatternSet],
    store_issue: str,
) -> tuple[str, str]:
    """Whether this source's taught grammar is still waiting for a human.

    Deliberately narrower than `status._pattern_review_state`, which answers
    the *archival* question for a zero-record extraction and therefore folds in
    coverage. Coverage is a card-track gate; a lesson whose coverage is
    unresolved can still have had its grammar read and approved, and saying
    otherwise would park the grammar badge behind an unrelated decision.
    """
    raw = meta.get("pattern_set")
    if not isinstance(raw, Mapping):
        return GRAMMAR_NONE, ""
    listed = raw.get("patterns")
    if not isinstance(listed, list) or not listed:
        # A vocabulary table teaches no grammar. An empty set is not an
        # unreviewed set, and badging it would train the reader to ignore it.
        return GRAMMAR_NONE, ""
    source = meta.get("source_file")
    if not isinstance(source, str) or not source.strip():
        return (
            GRAMMAR_UNKNOWN,
            "this file names no source, so its grammar cannot be matched to the "
            "pattern store",
        )
    if store_issue:
        return GRAMMAR_UNKNOWN, store_issue
    stored = store.get(source)
    if stored is None:
        return GRAMMAR_UNKNOWN, "this source's grammar is not in the pattern store"
    if stored.review_run_id != meta.get("review_run_id"):
        # A different paid run wrote the stored entry. Its `reviewed` flag
        # answers a question about text this staging file no longer holds.
        return (
            GRAMMAR_UNKNOWN,
            "the pattern store holds a different extraction run for this source",
        )
    return (GRAMMAR_REVIEWED if stored.reviewed else GRAMMAR_NEEDS_REVIEW), ""


def _coverage_unresolved(meta: Mapping[str, Any]) -> str:
    """The coverage refusal `promote` would raise, or empty if it would not."""
    try:
        staging.require_resolved_coverage(meta)
    except JankiError as exc:
        return str(exc)
    return ""


def _deck_decisions(
    config: ProjectConfig, records: Sequence[VocabularyRecord]
) -> tuple[int, str, str]:
    """How many prospective cards lack one proven word-deck owner."""
    if not records:
        return 0, "", ""
    try:
        ownership = evaluate_prospective_deck_ownership(config, records)
    except JankiError as exc:
        # Deck-reading failures are represented by unreadable evaluations
        # below. An exception here instead means canonical or staged card data
        # could not be loaded or prospectively merged.
        return len(records), str(exc), "Repair the collection or staged cards"
    count = sum(item.state != "exactly_one" for item in ownership)
    unreadable = tuple(
        dict.fromkeys(
            problem
            for item in ownership
            for problem in item.unreadable_decks
        )
    )
    if unreadable:
        return count, "; ".join(unreadable), "Repair the study deck configuration"
    return (
        count,
        (
            "Every card must belong to exactly one word deck before it can be added."
            if count
            else ""
        ),
        "",
    )


def _card_state(
    records: Sequence[VocabularyRecord],
    meta: Mapping[str, Any],
    *,
    grammar: str,
    deck_decision_count: int,
    deck_detail: str,
    deck_repair_action: str,
) -> tuple[str, str, str]:
    """`(state, next_action, detail)` for the word-card track."""
    held = [record for record in records if _held(record)]
    invalid = [
        record
        for record in records
        if validation.has_errors(validation.validate_record(record))
    ]
    unreviewed = [record for record in records if needs_example_review(record)]

    if not records:
        # Zero cards is the pattern-only shape. Say so plainly rather than
        # reporting "Ready to add" over nothing, or trapping it in deck steps.
        if grammar in {GRAMMAR_NEEDS_REVIEW, GRAMMAR_UNKNOWN}:
            return GRAMMAR_ONLY, "Review this source's grammar", ""
        return GRAMMAR_ONLY, "Nothing left to do for this source", ""
    if held:
        return (
            READING_HOLD,
            f"Decide the reading for {_cards(len(held))}",
            "; ".join(
                sorted({staging.annotations(record)["hold_reason"] for record in held})
            ),
        )
    if invalid:
        return CARDS_NEED_EDITS, f"Fix {_cards(len(invalid))}", ""
    if unreviewed:
        return (
            EXAMPLES_NEED_REVIEW,
            f"Review the Japanese examples on {_cards(len(unreviewed))}",
            "",
        )
    coverage = _coverage_unresolved(meta)
    if coverage:
        return (
            COVERAGE_NEEDS_DECISION,
            "Decide whether the extraction covered this source",
            coverage,
        )
    if deck_repair_action:
        return (
            DECK_NEEDS_DECISION,
            deck_repair_action,
            deck_detail,
        )
    if deck_decision_count:
        return (
            DECK_NEEDS_DECISION,
            f"Choose a study deck for {_cards(deck_decision_count)}",
            deck_detail,
        )
    return READY_TO_ADD, f"Add {_cards(len(records))} to your collection", ""


def _cards(count: int) -> str:
    return "1 card" if count == 1 else f"{count} cards"


def _staged_journeys(
    config: ProjectConfig,
    store: Mapping[str, patterns.PatternSet],
    store_issue: str,
) -> tuple[list[SourceJourney], set[str], list[str]]:
    """Every live staging file, plus the source names they account for."""
    journeys: list[SourceJourney] = []
    accounted: set[str] = set()
    warnings: list[str] = []
    if not config.staging_dir.is_dir():
        return journeys, accounted, warnings

    parsed: list[tuple[Path, list[VocabularyRecord], dict[str, Any]]] = []
    for path in sorted(
        [*config.staging_dir.glob("*.yaml"), *config.staging_dir.glob("*.yml")]
    ):
        try:
            records, meta = staging.read_staging(path)
        except JankiError as exc:
            # An unreadable file is a visible state, not a dropped row. A queue
            # that silently omits it is how a source goes missing for a week.
            journeys.append(
                SourceJourney(
                    source=path.name,
                    state=STAGING_UNREADABLE,
                    next_action="Repair this file, or ask for help reading it",
                    staging_path=path,
                    detail=str(exc),
                )
            )
            warnings.append(f"skipping staging file {path}: {exc}")
            continue
        parsed.append((path, records, meta))

    for path, records, meta in parsed:
        named = meta.get("source_file")
        source = named if isinstance(named, str) and named.strip() else path.name
        accounted.add(source)
        grammar, grammar_detail = _grammar_state(meta, store, store_issue)
        deck_decision_count, deck_detail, deck_repair_action = _deck_decisions(
            config, records
        )
        state, action, detail = _card_state(
            records,
            meta,
            grammar=grammar,
            deck_decision_count=deck_decision_count,
            deck_detail=deck_detail,
            deck_repair_action=deck_repair_action,
        )
        journeys.append(
            SourceJourney(
                source=source,
                state=state,
                next_action=action,
                grammar=grammar,
                staging_path=path,
                card_count=len(records),
                held_count=sum(1 for record in records if _held(record)),
                example_review_count=sum(
                    1 for record in records if needs_example_review(record)
                ),
                invalid_count=sum(
                    1
                    for record in records
                    if validation.has_errors(validation.validate_record(record))
                ),
                deck_decision_count=deck_decision_count,
                detail=detail,
                grammar_detail=grammar_detail,
            )
        )
    return journeys, accounted, warnings


def _unextracted(
    config: ProjectConfig,
    accounted: set[str],
    store: Mapping[str, patterns.PatternSet],
    receipt_ids_by_source: Mapping[str, tuple[str, ...]],
    archive_issue: str,
    store_issue: str,
) -> list[SourceJourney]:
    """Inbox files nothing durable speaks for yet.

    "Nothing durable" is deliberately wider than "no staging file". A source
    can be read by a paid call and leave *only* a pattern-store entry behind —
    a chart teaches grammar and no vocabulary, and once its zero-record staging
    is archived and pruned the store is the sole surviving evidence the call
    happened. Checking staging alone reports such a source as unread and offers
    to read it again, which is an offer to spend money re-buying an answer the
    repository already holds. The store is consulted before calling the source
    unread for that reason.
    """
    if not config.scan_inbox.is_dir():
        return []
    found: list[SourceJourney] = []
    archive = config.staging_dir / "done"
    for path in sorted(config.scan_inbox.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.name in accounted:
            continue
        # `extract` owns the staging filename convention; deriving it here by
        # string concatenation would drift the moment that changes.
        live = extract.staging_path(config.staging_dir, path.name)
        if live.exists():
            # A live file is this source's current state; _staged_journeys
            # already spoke for it, and re-extraction after a promotion is
            # ordinary. The newer answer wins.
            continue
        archive_exists = (archive / live.name).exists()
        if archive_issue:
            # Receipt discovery validates the whole archive namespace so a
            # malformed file cannot conceal a duplicate durable handle. Once
            # that global proof fails, no source without live staging can be
            # called unread safely: the bad archive may be the paid answer for
            # this source even when its filename does not follow today's
            # convention.
            found.append(
                SourceJourney(
                    source=path.name,
                    state=FINISH_ARCHIVE_UNREADABLE,
                    next_action=(
                        "Reload this library; if this remains, repair the "
                        "completed-card archive"
                        if archive_issue.startswith("[finish-archive-scan]")
                        else "Repair the completed-card archive named below"
                    ),
                    detail=(
                        "janki could not verify the completed-card archive, so "
                        f"it will not offer another paid extraction for {path.name}: "
                        f"{archive_issue}"
                    ),
                )
            )
            continue
        finish_receipt_ids = receipt_ids_by_source.get(path.name, ())
        if finish_receipt_ids:
            # Promoted: its staging was archived and pruned. Before this
            # branch the source simply vanished from the queue, which reads
            # as "janki lost my lesson" rather than "that one is done".
            found.append(
                SourceJourney(
                    source=path.name,
                    state=ADDED,
                    next_action="Add dictionary facts, audio, and build the deck",
                    finish_receipt_ids=finish_receipt_ids,
                )
            )
            continue
        if archive_exists:
            # Historical promoted archives predate exact W5 finish receipts.
            # Their rows are still durable evidence that this source was read
            # and added, so never turn absence of a new receipt into an offer
            # to buy the same extraction again. The old archive cannot safely
            # manufacture the owner binding a bounded finish link would need.
            found.append(
                SourceJourney(
                    source=path.name,
                    state=ADDED,
                    next_action="Add dictionary facts, audio, and build the deck",
                )
            )
            continue
        if store_issue:
            # With no live review or completed-card archive, the grammar store
            # is the only place a pattern-only paid extraction can survive.
            # An unreadable store therefore cannot be treated as an empty one:
            # doing so would turn lost evidence into an offer to buy the same
            # answer again.
            found.append(
                SourceJourney(
                    source=path.name,
                    state=PATTERN_STORE_UNREADABLE,
                    next_action="Repair the grammar history named below",
                    detail=(
                        "janki could not verify whether this source was already "
                        "read for grammar, so it will not offer another paid "
                        f"extraction for {path.name}: {store_issue}"
                    ),
                )
            )
            continue
        stored = store.get(path.name)
        if stored is not None:
            # Read already, and all it taught was grammar.
            grammar = GRAMMAR_REVIEWED if stored.reviewed else GRAMMAR_NEEDS_REVIEW
            found.append(
                SourceJourney(
                    source=path.name,
                    state=GRAMMAR_ONLY,
                    next_action=(
                        "Nothing left to do for this source"
                        if stored.reviewed
                        else "Review this source's grammar"
                    ),
                    grammar=grammar,
                )
            )
            continue
        found.append(
            SourceJourney(
                source=path.name,
                state=NOT_EXTRACTED,
                next_action="Read this source to propose cards and grammar",
            )
        )
    return found


def source_journeys(
    config: ProjectConfig,
) -> tuple[list[SourceJourney], list[str]]:
    """Every source's state and next action, plus any warnings.

    Read-only: this opens files and writes nothing. Restarting the process and
    calling it again must produce the same answer, which is what lets the
    dashboard survive a refresh without a server-side session.

    Transient dispatch states are deliberately absent rather than guessed at.
    A queue that invents "Extraction running" from a file's mtime would be
    lying at exactly the moment it matters; journal recovery needs its own
    durable projection. Deck ownership, by contrast, is reconstructed from
    the staged records, canonical collection, and real deck selectors on every
    call.
    """
    store: dict[str, patterns.PatternSet] = {}
    store_issue = ""
    warnings: list[str] = []
    try:
        store = patterns.load_store(config.patterns_file)
    except JankiError as exc:
        store_issue = str(exc)
        warnings.append(
            f"could not read grammar review state in {config.patterns_file}: {exc}"
        )
    # Read live reviews first. Promotion publishes the done archive before it
    # prunes the live review; this order means a concurrent promotion can leave
    # a stale-but-safe review row or a fresh receipt row, never a false offer to
    # pay for extraction after both reads miss opposite sides of the handoff.
    staged, accounted, staged_warnings = _staged_journeys(config, store, store_issue)
    warnings.extend(staged_warnings)

    receipt_ids_by_source: dict[str, list[str]] = {}
    archive_issue = ""
    try:
        for receipt in list_finish_receipts(config):
            receipt_ids_by_source.setdefault(receipt.source_file, []).append(
                receipt.receipt_id
            )
    except FinishScopeError as exc:
        archive_issue = str(exc)
        warnings.append(
            f"could not read completed-card receipts in "
            f"{config.staging_dir / 'done'}: {exc}"
        )
    receipt_ids = {
        source: tuple(source_receipts)
        for source, source_receipts in receipt_ids_by_source.items()
    }
    staged = [
        replace(
            journey,
            finish_receipt_ids=receipt_ids.get(journey.source, ()),
        )
        for journey in staged
    ]
    journeys = [
        *staged,
        *_unextracted(
            config,
            accounted,
            store,
            receipt_ids,
            archive_issue,
            store_issue,
        ),
    ]
    journeys.sort(key=lambda journey: (JOURNEY_STATES.index(journey.state), journey.source))
    return journeys, warnings
