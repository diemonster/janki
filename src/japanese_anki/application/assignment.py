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

from collections import Counter
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
    "DeckAssignmentAttempt",
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
    "plan_deck_assignments",
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


@dataclass(frozen=True, slots=True)
class DeckAssignmentAttempt:
    """One destination's exact plan, or its exact refusal."""

    destination: AssignableWordDeck
    plan: DeckAssignmentPlan | None
    refusal: str | None


@dataclass(frozen=True, slots=True)
class _PreparedAssignment:
    index: int
    current: VocabularyRecord
    destination: AssignableWordDeck
    sibling_proposals: tuple[VocabularyRecord, ...]
    existing_owner: str | None
    tag_diff: DeckTagDiff
    assigned_record: VocabularyRecord


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


def _membership_ids(
    config: ProjectConfig,
    rules: Sequence[_WordDeckRule],
    collection: Sequence[VocabularyRecord],
) -> dict[Path, frozenset[str]]:
    """Project each deck once against one prospective collection."""
    projected: dict[Path, frozenset[str]] = {}
    canonical = config.normalized_file.resolve()
    for rule in rules:
        if rule.source_path == canonical:
            _deck_config, records = project_deck_records(
                rule.path, canonical, collection
            )
            projected[rule.path] = frozenset(item.id for item in records)
        else:
            # Assigning staged rows changes only the canonical collection.
            # A deck reading another source is unaffected.
            projected[rule.path] = rule.resolved_ids
    return projected


def _memberships_from_ids(
    rules: Sequence[_WordDeckRule],
    record: VocabularyRecord,
    membership_ids: dict[Path, frozenset[str]],
) -> tuple[DeckMembership, ...]:
    found: list[DeckMembership] = []
    for rule in rules:
        takes = record.id in membership_ids[rule.path]
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
    membership_ids: dict[Path, frozenset[str]] = {}
    membership_issue: str | None = None
    try:
        membership_ids = _membership_ids(config, rules, prospective_collection)
    except JankiError as exc:
        membership_issue = str(exc)

    evaluations: list[DeckOwnershipEvaluation] = []
    for record in targets:
        problems = list(configuration_problems)
        if membership_issue is not None:
            memberships = ()
            problems.append(membership_issue)
        else:
            memberships = _memberships_from_ids(rules, record, membership_ids)
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


def _finish_assignment_plan(
    rules: Sequence[_WordDeckRule],
    prepared: _PreparedAssignment,
    prospective: VocabularyRecord,
    membership_ids: dict[Path, frozenset[str]],
) -> DeckAssignmentPlan:
    memberships = _memberships_from_ids(rules, prospective, membership_ids)
    chosen = next(
        item for item in memberships if item.stem == prepared.destination.stem
    )
    if not chosen.takes:
        raise AssignmentError(
            f"{prepared.destination.name} does not select the card after assignment: "
            f"{chosen.refusal or 'its deck rule refused the card'}."
        )
    other_owners = [
        item
        for item in memberships
        if item.takes and item.stem != prepared.destination.stem
    ]
    if other_owners:
        names = ", ".join(item.name for item in other_owners)
        raise AssignmentError(
            f"The prospective record for {prepared.destination.name} would also "
            f"select {names}; change the deck rules or review the record tags first."
        )
    return DeckAssignmentPlan(
        record_id=prepared.current.id,
        destination=prepared.destination,
        proposals=tuple(
            _occurrence(record)
            for record in (prepared.current, *prepared.sibling_proposals)
        ),
        existing_owner=prepared.existing_owner,
        tag_diff=prepared.tag_diff,
        assigned_record=prepared.assigned_record,
        prospective_record=prospective,
        memberships=memberships,
    )


def _plan_prepared_group(
    config: ProjectConfig,
    rules: Sequence[_WordDeckRule],
    canonical: Sequence[VocabularyRecord],
    prepared: Sequence[_PreparedAssignment],
) -> dict[int, DeckAssignmentAttempt]:
    """Plan distinct ids together, falling back if a shared projection refuses."""
    try:
        collection = _merge_prospective_collection(
            canonical,
            [item.assigned_record for item in prepared],
        )
        by_id = {record.id: record for record in collection}
        membership_ids = _membership_ids(config, rules, collection)
    except JankiError as exc:
        if len(prepared) == 1:
            item = prepared[0]
            return {
                item.index: DeckAssignmentAttempt(
                    destination=item.destination,
                    plan=None,
                    refusal=str(exc),
                )
            }
        found: dict[int, DeckAssignmentAttempt] = {}
        for item in prepared:
            found.update(_plan_prepared_group(config, rules, canonical, [item]))
        return found

    found = {}
    for item in prepared:
        try:
            plan = _finish_assignment_plan(
                rules,
                item,
                by_id[item.current.id],
                membership_ids,
            )
        except JankiError as exc:
            found[item.index] = DeckAssignmentAttempt(
                destination=item.destination,
                plan=None,
                refusal=str(exc),
            )
        else:
            found[item.index] = DeckAssignmentAttempt(
                destination=item.destination,
                plan=plan,
                refusal=None,
            )
    return found


def plan_deck_assignments(
    config: ProjectConfig,
    records: Sequence[VocabularyRecord],
    *,
    destination_stems: Sequence[str] | None = None,
    sibling_proposals: Sequence[Sequence[VocabularyRecord]] | None = None,
) -> tuple[tuple[DeckAssignmentAttempt, ...], ...]:
    """Plan a record-by-destination matrix from one repository snapshot.

    Every cell remains the same independent assignment plan the single-record
    API returns. Distinct stable ids can share one prospective collection for a
    destination because merging, inline overrides, and deck selection are all
    record-local. Duplicate ids deliberately fall back to one-record groups so
    two separately rendered rows cannot influence one another's claim.
    """
    rules = _word_deck_rules(config)
    all_destinations = _assignable_from_rules(config, rules)
    if destination_stems is None:
        destinations = all_destinations
    else:
        by_stem = {deck.stem: deck for deck in all_destinations}
        selected: list[AssignableWordDeck] = []
        for stem in destination_stems:
            destination = by_stem.get(stem)
            if destination is None:
                raise AssignmentError(
                    f"No assignable word deck has deck stem {stem!r}."
                )
            selected.append(destination)
        destinations = tuple(selected)

    sibling_rows = (
        tuple(() for _record in records)
        if sibling_proposals is None
        else tuple(tuple(items) for items in sibling_proposals)
    )
    if len(sibling_rows) != len(records):
        raise AssignmentError(
            "Deck-assignment sibling proposals must align with the rendered records."
        )
    try:
        canonical = (
            load_records(config.normalized_file)
            if config.normalized_file.exists()
            else []
        )
    except JankiError as exc:
        raise AssignmentError(str(exc)) from exc

    canonical_matches: dict[str, list[VocabularyRecord]] = {}
    for record in canonical:
        canonical_matches.setdefault(record.id, []).append(record)
    existing_indices = [
        index
        for index, record in enumerate(records)
        if len(canonical_matches.get(record.id, ())) == 1
    ]
    current_membership_ids: dict[Path, frozenset[str]] = {}
    current_membership_issue: str | None = None
    if existing_indices:
        try:
            current_membership_ids = _membership_ids(config, rules, canonical)
        except JankiError as exc:
            current_membership_issue = str(exc)

    known_intake_tags = frozenset(
        destination.intake_tag for destination in all_destinations
    )
    rows: list[list[DeckAssignmentAttempt | None]] = [
        [None for _destination in destinations] for _record in records
    ]
    prepared_by_destination: list[list[_PreparedAssignment]] = [
        [] for _destination in destinations
    ]
    for index, (current, siblings) in enumerate(
        zip(records, sibling_rows, strict=True)
    ):
        problem: str | None = None
        existing_owner: _WordDeckRule | None = None
        if any(proposal.id != current.id for proposal in siblings):
            problem = (
                "Sibling proposals shown for a deck assignment must have the same "
                "stable id as the current staged card."
            )
        matching = canonical_matches.get(current.id, [])
        if problem is None and len(matching) > 1:
            problem = (
                f"The canonical collection has more than one {current.id}; resolve "
                "that duplicate before assignment."
            )
        if problem is None and matching:
            if current_membership_issue is not None:
                problem = current_membership_issue
            else:
                current_memberships = _memberships_from_ids(
                    rules,
                    matching[0],
                    current_membership_ids,
                )
                owner_stems = {
                    item.stem for item in current_memberships if item.takes
                }
                owners = [rule for rule in rules if rule.stem in owner_stems]
                if len(owners) > 1:
                    names = ", ".join(owner.name for owner in owners)
                    problem = (
                        f"{current.id} already belongs to more than one word deck "
                        f"({names}); resolve its existing membership before assignment."
                    )
                elif owners:
                    existing_owner = owners[0]

        for destination_index, destination in enumerate(destinations):
            refusal = problem
            if (
                refusal is None
                and existing_owner is not None
                and existing_owner.stem != destination.stem
            ):
                refusal = (
                    f"{current.id} already belongs to {existing_owner.name} "
                    f"({existing_owner.stem}); moving it to {destination.name} "
                    "requires an explicit reassignment review."
                )
            if refusal is not None:
                rows[index][destination_index] = DeckAssignmentAttempt(
                    destination=destination,
                    plan=None,
                    refusal=refusal,
                )
                continue
            diff = _tag_diff(
                current,
                known_intake_tags=known_intake_tags,
                chosen_intake_tag=destination.intake_tag,
            )
            prepared_by_destination[destination_index].append(
                _PreparedAssignment(
                    index=index,
                    current=current,
                    destination=destination,
                    sibling_proposals=siblings,
                    existing_owner=(
                        existing_owner.stem if existing_owner is not None else None
                    ),
                    tag_diff=diff,
                    assigned_record=replace(current, tags=list(diff.after)),
                )
            )

    for destination_index, prepared in enumerate(prepared_by_destination):
        if not prepared:
            continue
        ids = [item.current.id for item in prepared]
        counts = Counter(ids)
        unique = [item for item in prepared if counts[item.current.id] == 1]
        repeated = [item for item in prepared if counts[item.current.id] > 1]
        groups = ([unique] if unique else []) + [[item] for item in repeated]
        for group in groups:
            attempts = _plan_prepared_group(config, rules, canonical, group)
            for index, attempt in attempts.items():
                rows[index][destination_index] = attempt

    if any(attempt is None for row in rows for attempt in row):
        raise AssignmentError("Deck-assignment planning left an incomplete choice matrix.")
    return tuple(
        tuple(attempt for attempt in row if attempt is not None) for row in rows
    )


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
    attempts = plan_deck_assignments(
        config,
        [current],
        destination_stems=[destination_stem],
        sibling_proposals=[sibling_proposals],
    )
    attempt = attempts[0][0]
    if attempt.plan is None:
        raise AssignmentError(
            attempt.refusal
            if attempt.refusal is not None
            else "Deck assignment was refused."
        )
    return attempt.plan
