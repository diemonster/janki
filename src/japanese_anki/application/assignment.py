"""Read-only plans for assigning staged words to durable study decks.

The browser submits a deck *stem*, which is janki's stable handle for a deck
file.  The ``intake_tag`` remains configuration: this service resolves it,
shows the exact tag replacement, and then asks the deck selectors themselves
whether the prospective canonical record lands in exactly that word deck.

Nothing here writes staging or the collection.  A later promotion transaction
can bind and apply :class:`DeckTagDiff`; until then this is only a reviewable
plan.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from japanese_anki import status as status_module
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import (
    DeckSelection,
    deck_kind,
    deck_selection,
    project_deck_records,
    resolve_deck_records,
)
from japanese_anki.io import load_records, merge_records
from japanese_anki.models import VocabularyRecord

__all__ = [
    "AssignableWordDeck",
    "AssignmentError",
    "DeckAssignmentPlan",
    "DeckMembership",
    "DeckOwnershipEvaluation",
    "DeckOwnershipState",
    "DeckTagDiff",
    "ProposalOccurrence",
    "assignable_word_decks",
    "evaluate_deck_ownership",
    "evaluate_prospective_deck_ownership",
    "plan_deck_assignment",
    "require_exact_deck_ownership",
]


class AssignmentError(JankiError):
    """A requested assignment cannot truthfully produce one deck owner."""


@dataclass(frozen=True, slots=True)
class AssignableWordDeck:
    """A vocabulary deck with a configured assignment tag."""

    stem: str
    path: Path
    name: str
    intake_tag: str
    selection: DeckSelection


@dataclass(frozen=True, slots=True)
class _WordDeckRule:
    """Every vocabulary selector, including non-assignable exact-list decks."""

    stem: str
    path: Path
    name: str
    selection: DeckSelection
    source_path: Path | None
    resolved_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class ProposalOccurrence:
    """One source occurrence of a stable id, retained in response order."""

    imported_from: str
    row: int | None
    source_type: str


@dataclass(frozen=True, slots=True)
class DeckTagDiff:
    """The exact ownership-tag transform shown before any write."""

    before: tuple[str, ...]
    after: tuple[str, ...]
    removed: tuple[str, ...]
    added: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeckMembership:
    """One real vocabulary selector's verdict on the prospective record."""

    stem: str
    name: str
    takes: bool
    refusal: str | None


DeckOwnershipState = Literal[
    "exactly_one",
    "unassigned",
    "multiple",
    "unreadable",
]


@dataclass(frozen=True, slots=True)
class DeckOwnershipEvaluation:
    """The real word-deck partition result for one prospective record."""

    record_id: str
    state: DeckOwnershipState
    memberships: tuple[DeckMembership, ...]
    unreadable_decks: tuple[str, ...] = ()

    @property
    def owners(self) -> tuple[DeckMembership, ...]:
        return tuple(item for item in self.memberships if item.takes)

    @property
    def owner_stems(self) -> tuple[str, ...]:
        return tuple(item.stem for item in self.owners)


@dataclass(frozen=True, slots=True)
class DeckAssignmentPlan:
    """A chosen destination and the exact canonical record it would produce."""

    record_id: str
    destination: AssignableWordDeck
    proposals: tuple[ProposalOccurrence, ...]
    existing_owner: str | None
    tag_diff: DeckTagDiff
    assigned_record: VocabularyRecord
    prospective_record: VocabularyRecord
    memberships: tuple[DeckMembership, ...]


def _word_deck_census(
    config: ProjectConfig,
) -> tuple[tuple[_WordDeckRule, ...], tuple[str, ...]]:
    """Read every word-deck rule and retain anything that could not be read."""
    rules: list[_WordDeckRule] = []
    unreadable: list[str] = []
    try:
        paths = status_module.deck_files(config)
    except JankiError as exc:
        return (), (str(exc),)
    for path in paths:
        try:
            kind = deck_kind(path)
        except JankiError as exc:
            unreadable.append(f"{path}: {exc}")
            continue
        if kind not in ("", "vocabulary"):
            continue
        try:
            deck_config, records = resolve_deck_records(path)
            selection = deck_selection(deck_config, path)
        except JankiError as exc:
            unreadable.append(f"{path}: {exc}")
            continue
        source_value = deck_config.get("source")
        rules.append(
            _WordDeckRule(
                stem=path.stem,
                path=path,
                name=str(deck_config.get("name") or path.stem),
                selection=selection,
                source_path=(
                    (path.parent / str(source_value)).resolve()
                    if source_value
                    else None
                ),
                resolved_ids=frozenset(record.id for record in records),
            )
        )
    return tuple(rules), tuple(unreadable)


def _word_deck_rules(config: ProjectConfig) -> tuple[_WordDeckRule, ...]:
    """Read every real word-deck rule through the build's authorities."""
    rules, unreadable = _word_deck_census(config)
    if unreadable:
        raise AssignmentError(
            "Could not read every configured word deck: " + "; ".join(unreadable)
        )
    return rules


def _assignable_from_rules(
    config: ProjectConfig, rules: Sequence[_WordDeckRule]
) -> tuple[AssignableWordDeck, ...]:
    found: list[AssignableWordDeck] = []
    canonical = config.normalized_file.resolve()
    for rule in rules:
        if rule.selection.intake_tag is None:
            continue
        if rule.source_path != canonical:
            raise AssignmentError(
                f"Assignable word deck {rule.path} must read the canonical "
                f"collection {canonical}; it reads {rule.source_path or 'no source file'}."
            )
        found.append(
            AssignableWordDeck(
                stem=rule.stem,
                path=rule.path,
                name=rule.name,
                intake_tag=rule.selection.intake_tag,
                selection=rule.selection,
            )
        )
    return tuple(found)


def assignable_word_decks(config: ProjectConfig) -> tuple[AssignableWordDeck, ...]:
    """Vocabulary destinations that explicitly declare an ``intake_tag``."""
    return _assignable_from_rules(config, _word_deck_rules(config))


def _occurrence(record: VocabularyRecord) -> ProposalOccurrence:
    return ProposalOccurrence(
        imported_from=record.source.imported_from,
        row=record.source.row,
        source_type=record.source.type,
    )


def _prospective_collection(
    canonical: Sequence[VocabularyRecord], assigned: VocabularyRecord
) -> tuple[list[VocabularyRecord], VocabularyRecord]:
    records = _merge_prospective_collection(canonical, [assigned])
    return records, next(record for record in records if record.id == assigned.id)


def _merge_prospective_collection(
    canonical: Sequence[VocabularyRecord], incoming: Sequence[VocabularyRecord]
) -> list[VocabularyRecord]:
    try:
        records, _outcomes = merge_records(list(canonical), list(incoming))
    except JankiError as exc:
        raise AssignmentError(str(exc)) from exc
    return records


def _tag_diff(
    record: VocabularyRecord,
    *,
    known_intake_tags: frozenset[str],
    chosen_intake_tag: str,
) -> DeckTagDiff:
    before = tuple(record.tags)
    after_items = [
        tag for tag in before if tag not in known_intake_tags or tag == chosen_intake_tag
    ]
    if chosen_intake_tag not in after_items:
        after_items.append(chosen_intake_tag)
    after = tuple(after_items)
    return DeckTagDiff(
        before=before,
        after=after,
        removed=tuple(dict.fromkeys(tag for tag in before if tag not in after)),
        added=tuple(dict.fromkeys(tag for tag in after if tag not in before)),
    )


def _memberships(
    config: ProjectConfig,
    rules: Sequence[_WordDeckRule],
    record: VocabularyRecord,
    collection: Sequence[VocabularyRecord],
) -> tuple[DeckMembership, ...]:
    found: list[DeckMembership] = []
    canonical = config.normalized_file.resolve()
    for rule in rules:
        if rule.source_path == canonical:
            _deck_config, projected = project_deck_records(
                rule.path, canonical, collection
            )
            takes = any(item.id == record.id for item in projected)
        else:
            # Assigning a staged row changes only the canonical collection.
            # A deck reading another source is unaffected, so its currently
            # resolved record set is the build's exact answer.
            takes = record.id in rule.resolved_ids
        refusal = None if takes else rule.selection.refusal(record)
        found.append(
            DeckMembership(
                stem=rule.stem,
                name=rule.name,
                takes=takes,
                refusal=refusal or (
                    None
                    if takes
                    else "the deck's resolved source and inline notes do not include this card"
                ),
            )
        )
    return tuple(found)


def evaluate_deck_ownership(
    config: ProjectConfig,
    records: Sequence[VocabularyRecord],
    prospective_collection: Sequence[VocabularyRecord],
) -> tuple[DeckOwnershipEvaluation, ...]:
    """Evaluate exact word-deck ownership against a prospective collection.

    The caller supplies the records whose landing matters and the already
    merged collection they would land in. The records are resolved back out of
    that collection by stable id, so a stale pre-merge value cannot be tested
    accidentally. Deck paths, selectors, inline overrides, and static-source
    membership all remain configuration-owned and are read through the same
    resolver a build uses.
    """
    by_id: dict[str, VocabularyRecord] = {}
    for record in prospective_collection:
        if record.id in by_id:
            raise AssignmentError(
                f"The prospective collection has more than one {record.id}; "
                "deck ownership cannot be evaluated."
            )
        by_id[record.id] = record
    targets: list[VocabularyRecord] = []
    for record in records:
        target = by_id.get(record.id)
        if target is None:
            raise AssignmentError(
                f"The prospective collection does not contain {record.id}; "
                "deck ownership cannot be evaluated."
            )
        targets.append(target)

    rules, census_problems = _word_deck_census(config)
    configuration_problems = list(census_problems)
    try:
        _assignable_from_rules(config, rules)
    except JankiError as exc:
        configuration_problems.append(str(exc))

    evaluations: list[DeckOwnershipEvaluation] = []
    for record in targets:
        problems = list(configuration_problems)
        try:
            memberships = _memberships(
                config, rules, record, prospective_collection
            )
        except JankiError as exc:
            memberships = ()
            problems.append(str(exc))
        owner_count = sum(item.takes for item in memberships)
        if problems:
            state: DeckOwnershipState = "unreadable"
        elif owner_count == 0:
            state = "unassigned"
        elif owner_count == 1:
            state = "exactly_one"
        else:
            state = "multiple"
        evaluations.append(
            DeckOwnershipEvaluation(
                record_id=record.id,
                state=state,
                memberships=memberships,
                unreadable_decks=tuple(dict.fromkeys(problems)),
            )
        )
    return tuple(evaluations)


def evaluate_prospective_deck_ownership(
    config: ProjectConfig,
    incoming: Sequence[VocabularyRecord],
) -> tuple[DeckOwnershipEvaluation, ...]:
    """Load canonical state, merge ``incoming``, and evaluate its ownership."""
    try:
        canonical = (
            load_records(config.normalized_file)
            if config.normalized_file.exists()
            else []
        )
    except JankiError as exc:
        raise AssignmentError(str(exc)) from exc
    collection = _merge_prospective_collection(canonical, incoming)
    return evaluate_deck_ownership(config, incoming, collection)


def require_exact_deck_ownership(
    config: ProjectConfig,
    records: Sequence[VocabularyRecord],
    prospective_collection: Sequence[VocabularyRecord],
) -> tuple[DeckOwnershipEvaluation, ...]:
    """Refuse unless every named prospective record has exactly one owner."""
    evaluations = evaluate_deck_ownership(config, records, prospective_collection)
    for evaluation in evaluations:
        if evaluation.state == "exactly_one":
            continue
        if evaluation.state == "unreadable":
            raise AssignmentError(
                "[deck-ownership-unreadable] Could not prove exactly one word "
                f"deck owns {evaluation.record_id}: "
                + "; ".join(evaluation.unreadable_decks)
            )
        if evaluation.state == "unassigned":
            raise AssignmentError(
                f"[deck-unassigned] {evaluation.record_id} is not selected by "
                "any configured word deck. Choose a study deck before promotion."
            )
        names = ", ".join(item.name for item in evaluation.owners)
        raise AssignmentError(
            f"[deck-overlap] {evaluation.record_id} is selected by more than "
            f"one configured word deck ({names}). Choose one study deck before "
            "promotion."
        )
    return evaluations


def plan_deck_assignment(
    config: ProjectConfig,
    current: VocabularyRecord,
    destination_stem: str,
    *,
    sibling_proposals: Sequence[VocabularyRecord] = (),
) -> DeckAssignmentPlan:
    """Plan one stable id's exact ownership-tag replacement.

    ``assigned_record`` is a copy of ``current`` with only its ownership tags
    changed; that is the value a workbench may persist. ``prospective_record``
    is the canonical record merged with that copy and exists only to prove the
    real selector outcome. Duplicate proposals from other staged sources are
    display-only: every source occurrence remains separately present on the
    returned plan, but a sibling's tags or content cannot silently change this
    source's assignment.

    An existing record with one current word-deck owner may only keep that
    owner, and an existing record already selected by multiple word decks is
    refused as an invalid starting point. A zero-owner canonical record may be
    given its first owner by this explicit choice. Reassignment needs a future
    contract that binds review to exact old canonical tags; this service does
    not invent it.
    """
    rules = _word_deck_rules(config)
    destinations = _assignable_from_rules(config, rules)
    destination = next(
        (deck for deck in destinations if deck.stem == destination_stem), None
    )
    if destination is None:
        raise AssignmentError(f"No assignable word deck has deck stem {destination_stem!r}.")

    if any(proposal.id != current.id for proposal in sibling_proposals):
        raise AssignmentError(
            "Sibling proposals shown for a deck assignment must have the same "
            "stable id as the current staged card."
        )
    record_id = current.id
    try:
        canonical = (
            load_records(config.normalized_file)
            if config.normalized_file.exists()
            else []
        )
    except JankiError as exc:
        raise AssignmentError(str(exc)) from exc
    matching = [record for record in canonical if record.id == record_id]
    if len(matching) > 1:
        raise AssignmentError(
            f"The canonical collection has more than one {record_id}; resolve "
            "that duplicate before assignment."
        )
    existing = matching[0] if matching else None
    existing_owner: _WordDeckRule | None = None
    if existing is not None:
        current_memberships = _memberships(config, rules, existing, canonical)
        owner_stems = {item.stem for item in current_memberships if item.takes}
        owners = [rule for rule in rules if rule.stem in owner_stems]
        if len(owners) > 1:
            names = ", ".join(owner.name for owner in owners)
            raise AssignmentError(
                f"{record_id} already belongs to more than one word deck "
                f"({names}); resolve its existing membership before assignment."
            )
        if len(owners) == 1:
            existing_owner = owners[0]
            if existing_owner.stem != destination.stem:
                raise AssignmentError(
                    f"{record_id} already belongs to {existing_owner.name} "
                    f"({existing_owner.stem}); moving it to {destination.name} "
                    "requires an explicit reassignment review."
                )

    diff = _tag_diff(
        current,
        known_intake_tags=frozenset(deck.intake_tag for deck in destinations),
        chosen_intake_tag=destination.intake_tag,
    )
    assigned = replace(current, tags=list(diff.after))
    collection, prospective = _prospective_collection(canonical, assigned)
    memberships = _memberships(config, rules, prospective, collection)
    chosen_membership = next(item for item in memberships if item.stem == destination.stem)
    if not chosen_membership.takes:
        raise AssignmentError(
            f"{destination.name} does not select the card after assignment: "
            f"{chosen_membership.refusal or 'its deck rule refused the card'}."
        )
    other_owners = [item for item in memberships if item.takes and item.stem != destination.stem]
    if other_owners:
        names = ", ".join(item.name for item in other_owners)
        raise AssignmentError(
            f"The prospective record for {destination.name} would also select "
            f"{names}; change the deck rules or review the record tags first."
        )

    return DeckAssignmentPlan(
        record_id=record_id,
        destination=destination,
        proposals=tuple(_occurrence(record) for record in (current, *sibling_proposals)),
        existing_owner=existing_owner.stem if existing_owner is not None else None,
        tag_diff=diff,
        assigned_record=assigned,
        prospective_record=prospective,
        memberships=memberships,
    )
