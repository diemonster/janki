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
    "DeckTagDiff",
    "ProposalOccurrence",
    "assignable_word_decks",
    "plan_deck_assignment",
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


def _word_deck_rules(config: ProjectConfig) -> tuple[_WordDeckRule, ...]:
    """Read every real word-deck rule through the build's authorities."""
    rules: list[_WordDeckRule] = []
    for path in status_module.deck_files(config):
        kind = deck_kind(path)
        if kind not in ("", "vocabulary"):
            continue
        deck_config, records = resolve_deck_records(path)
        selection = deck_selection(deck_config, path)
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
    return tuple(rules)


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
    try:
        records, _outcomes = merge_records(list(canonical), [assigned])
    except JankiError as exc:
        raise AssignmentError(str(exc)) from exc
    return records, next(record for record in records if record.id == assigned.id)


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
