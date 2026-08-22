"""Changing which word a staged card *is*, deliberately.

`WORKBENCH_PLAN.md` W2d. Fixing an English gloss and deciding that `とまる`
is really `泊まる[とまる]` are different questions, and the second one has
consequences the first does not:

* the stable ID is `word:<expression>:<reading>`, so changing either changes
  the identity;
* the Anki GUID is `genanki.guid_for(record.id)`, so a card that has already
  shipped under the old ID becomes a **different note** under the new one —
  the old note stays in Anki with its review history and the new one starts
  from zero;
* the collection may already hold the new identity, in which case adding this
  card merges into that record rather than creating one.

None of that is recoverable by undo, so this module's job is to compute and
show it *before* anything is written. What it deliberately does not do is
suggest an identity. Deciding that a kana spelling "should" be a particular
kanji is reading Japanese, which `docs/DESIGN.md` reserves for the model and
the human — a rule that proposed `泊まる` for `とまる` would be wrong for the
next word that looks the same.

V1 re-identifies staged rows only. Migrating an identity that already reached
the canonical store is a different problem — it has to move review history and
rewrite what the ledger says shipped — and it is out of scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import normalize_identity_part, stable_record_id
from japanese_anki.models import VocabularyRecord

__all__ = [
    "Neighbour",
    "Reidentification",
    "ReidentifyError",
    "apply_reidentification",
    "plan_reidentification",
]


class ReidentifyError(JankiError):
    """A re-identification that cannot be applied as asked."""


@dataclass(frozen=True, slots=True)
class Neighbour:
    """An existing record the new identity would sit beside, or collide with."""

    record_id: str
    expression: str
    reading: str
    #: `same-id`, `same-reading` or `same-spelling`.
    relation: str
    #: Where it lives: `this source` or `your collection`.
    where: str


@dataclass(frozen=True, slots=True)
class Reidentification:
    """Everything a person needs to see before agreeing to this change."""

    index: int
    old_id: str
    new_id: str
    old_expression: str
    old_reading: str
    new_expression: str
    new_reading: str
    neighbours: tuple[Neighbour, ...]
    #: True when this card has already shipped in a built deck under `old_id`.
    was_exported: bool

    @property
    def is_change(self) -> bool:
        return self.old_id != self.new_id

    @property
    def collides_in_source(self) -> bool:
        """Whether another row in *this file* already holds the new identity.

        Genuinely impossible: one staging file cannot carry two rows claiming
        the same word, and promotion would not know which one to believe.
        """
        return any(
            n.relation == "same-id" and n.where == "this source"
            for n in self.neighbours
        )

    @property
    def matches_collection(self) -> bool:
        """Whether the collection already holds the new identity.

        Not an error — the opposite. Deciding that a staged card *is* the word
        you already have is the most useful thing this flow does, and promote's
        existing-wins merge is built for exactly it. Refusing it would block
        the case the feature exists for.
        """
        return any(
            n.relation == "same-id" and n.where == "your collection"
            for n in self.neighbours
        )

    def consequences(self) -> list[str]:
        """Plain sentences, in the order they matter."""
        said: list[str] = []
        if not self.is_change:
            return ["This is the same word it already was. Nothing would change."]
        if self.collides_in_source:
            said.append(
                f"Another card in this source is already {self.new_expression} "
                f"({self.new_reading}). Two cards cannot share one identity, so "
                "this change cannot be saved."
            )
        elif self.matches_collection:
            said.append(
                f"Your collection already has {self.new_expression} "
                f"({self.new_reading}). Adding this card would merge into that "
                "record, and the meanings already on it are kept."
            )
        if self.was_exported:
            said.append(
                "This word has already been built into a deck under its old "
                "name. Anki matches notes by that name, so the card you have "
                "been studying stays as it is and this becomes a new card with "
                "no review history."
            )
        else:
            said.append(
                "This card has not been built into a deck yet, so no review "
                "history is affected."
            )
        said.append(
            "The Japanese sentences on this card are not changed, and any "
            "approval you have already given them still stands."
        )
        return said


def _neighbours(
    new_id: str,
    expression: str,
    reading: str,
    *,
    index: int,
    staged: Sequence[VocabularyRecord],
    existing: Sequence[VocabularyRecord],
) -> tuple[Neighbour, ...]:
    """Records worth seeing before committing to this identity.

    Same identity first because it is a collision; then same reading and same
    spelling, because those are the two ways a person discovers they have just
    split or merged a word by accident.
    """
    found: list[Neighbour] = []
    for where, records in (("this source", staged), ("your collection", existing)):
        for position, record in enumerate(records):
            if where == "this source" and position == index:
                continue
            if record.id == new_id:
                relation = "same-id"
            elif record.reading == reading and record.expression != expression:
                relation = "same-reading"
            elif record.expression == expression and record.reading != reading:
                relation = "same-spelling"
            else:
                continue
            found.append(
                Neighbour(
                    record_id=record.id,
                    expression=record.expression,
                    reading=record.reading,
                    relation=relation,
                    where=where,
                )
            )
    order = {"same-id": 0, "same-reading": 1, "same-spelling": 2}
    found.sort(key=lambda n: (order[n.relation], n.where, n.record_id))
    return tuple(found)


def plan_reidentification(
    staged: Sequence[VocabularyRecord],
    index: int,
    expression: str,
    reading: str,
    *,
    existing: Sequence[VocabularyRecord] = (),
    exported_ids: frozenset[str] = frozenset(),
) -> Reidentification:
    """Work out what changing this card's identity would do. Writes nothing."""
    if not 0 <= index < len(staged):
        raise ReidentifyError("That card is not on this page.")
    new_expression = normalize_identity_part(expression)
    new_reading = normalize_identity_part(reading)
    if not new_expression:
        raise ReidentifyError("A card needs a Japanese expression.")
    if not new_reading:
        # The reading is half the identity, and a blank one is what the promote
        # gate holds a row back for. Minting an identity from it here would
        # walk straight into that refusal with a permanent ID already written.
        raise ReidentifyError(
            "A card needs a reading: the reading is part of what makes this "
            "word a distinct card."
        )
    record = staged[index]
    return Reidentification(
        index=index,
        old_id=record.id,
        new_id=stable_record_id(new_expression, new_reading),
        old_expression=record.expression,
        old_reading=record.reading,
        new_expression=new_expression,
        new_reading=new_reading,
        neighbours=_neighbours(
            stable_record_id(new_expression, new_reading),
            new_expression,
            new_reading,
            index=index,
            staged=staged,
            existing=existing,
        ),
        was_exported=record.id in exported_ids,
    )


def apply_reidentification(
    staged: Sequence[VocabularyRecord], plan: Reidentification
) -> tuple[VocabularyRecord, ...]:
    """The records with this card's identity changed. Pure."""
    if plan.collides_in_source:
        raise ReidentifyError(
            "Another card in this source already has that identity, so this "
            "change would make two cards claim one word."
        )
    updated = list(staged)
    record = updated[plan.index]
    updated[plan.index] = replace(
        record,
        id=plan.new_id,
        expression=plan.new_expression,
        reading=plan.new_reading,
    )
    return tuple(updated)
