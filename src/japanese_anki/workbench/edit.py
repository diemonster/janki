"""Editing a staged card's human-owned fields, and nothing else.

`WORKBENCH_PLAN.md` W2c. Two rules shape this module, and both are about what
a person is allowed to overwrite.

**Only human-owned fields are editable.** A staged card mixes two kinds of
value: what the model proposed as teaching content — meanings, examples, the
usage note — and the evidence for *why* it proposed it: the source page, the
sentence it was read from, the inclusion reason, the accounting the promote
gate checks. The first kind is a draft a person is meant to improve. The second
is a record of what happened, and a form that let someone retype it would let
them retype history. So the editable set is an allowlist, not a denylist: a
field nobody listed here cannot be edited by inventing a form field for it.

**Editing Japanese voids its approval, for free.** Approval binds the exact
sentence text by fingerprint, so a sentence someone rewrites no longer matches
the approval covering it and the card falls back to needing review. Nothing
here has to remember to do that — it is what fingerprinting the sentence
already means. Editing the *English* leaves approval intact, which is equally
correct: approval never covered the English.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from japanese_anki.errors import JankiError
from japanese_anki.models import VocabularyRecord

__all__ = [
    "EDITABLE_EXAMPLE_FIELDS",
    "EditError",
    "EditSubmission",
    "apply_edits",
    "changed_records",
    "parse_edit_form",
]


class EditError(JankiError):
    """A submitted edit that is malformed or names something uneditable."""


#: The example fields a person may retype, and the form prefix for each.
#: `japanese` is here deliberately: correcting a mis-transcribed sentence is
#: the whole point, and doing so voids that card's approval by fingerprint.
EDITABLE_EXAMPLE_FIELDS = {
    "ej": "japanese",
    "ef": "furigana",
    "er": "romaji",
    "ee": "english",
    "eg": "register",
}

# Cards are addressed by position, not by stable ID: an ID contains colons and
# Japanese text, and building a form-field name out of one invites a quoting
# bug at exactly the wrong layer. Position is unambiguous for the life of one
# render, and the byte snapshot is what proves the render is still current.
_CARD_FIELD = re.compile(r"\A(?P<kind>m|u)(?P<card>\d+)\Z")
_EXAMPLE_FIELD = re.compile(
    r"\A(?P<kind>ej|ef|er|ee|eg)(?P<card>\d+)_(?P<example>\d+)\Z"
)

_CONTROL_FIELDS = frozenset({"action", "csrf", "staging_snapshot"})


@dataclass(frozen=True, slots=True)
class EditSubmission:
    """One parsed edit form: its authority fields and the values it carries."""

    csrf: str
    staging_snapshot: str
    #: `(card index, field name) -> value`
    card_fields: dict[tuple[int, str], str]
    #: `(card index, example index, field name) -> value`
    example_fields: dict[tuple[int, int, str], str]


def parse_edit_form(pairs: Sequence[tuple[str, str]]) -> EditSubmission:
    """Validate an edit form's shape. Authority is the caller's to check."""
    grouped: dict[str, list[str]] = {}
    for key, value in pairs:
        grouped.setdefault(key, []).append(value)
    for key in _CONTROL_FIELDS:
        if len(grouped.get(key, [])) != 1:
            raise EditError(f"Form field {key!r} must appear exactly once")
    if grouped["action"][0] != "edit":
        raise EditError("The form action must be 'edit'")

    card_fields: dict[tuple[int, str], str] = {}
    example_fields: dict[tuple[int, int, str], str] = {}
    for key, values in grouped.items():
        if key in _CONTROL_FIELDS:
            continue
        if len(values) != 1:
            raise EditError(f"Form field {key!r} must appear exactly once")
        card = _CARD_FIELD.match(key)
        if card:
            name = "meanings" if card["kind"] == "m" else "usage_notes"
            card_fields[(int(card["card"]), name)] = values[0]
            continue
        example = _EXAMPLE_FIELD.match(key)
        if example:
            example_fields[
                (
                    int(example["card"]),
                    int(example["example"]),
                    EDITABLE_EXAMPLE_FIELDS[example["kind"]],
                )
            ] = values[0]
            continue
        # An unrecognized name is refused rather than ignored. Ignoring it
        # would let a form quietly carry a field this module does not own and
        # report success for an edit that never happened.
        raise EditError(f"Unknown edit field: {key}")
    return EditSubmission(
        csrf=grouped["csrf"][0],
        staging_snapshot=grouped["staging_snapshot"][0],
        card_fields=card_fields,
        example_fields=example_fields,
    )


def _meanings(value: str) -> list[str]:
    """One meaning per line, blanks dropped, order and wording kept."""
    return [line.strip() for line in value.splitlines() if line.strip()]


def apply_edits(
    records: Sequence[VocabularyRecord], submission: EditSubmission
) -> tuple[VocabularyRecord, ...]:
    """Return the records with the submitted values applied.

    Pure: it does not touch the filesystem, so the caller can diff the result
    against what is on disk before deciding whether anything needs writing at
    all. An index outside the rendered range is an error rather than a silent
    skip — it means the form and the file disagree about how many cards exist,
    and guessing which is right is how an edit lands on the wrong card.
    """
    updated = list(records)
    for (card_index, field), value in sorted(submission.card_fields.items()):
        if not 0 <= card_index < len(updated):
            raise EditError(f"Edit names card {card_index}, which is not on this page")
        record = updated[card_index]
        if field == "meanings":
            updated[card_index] = replace(record, meanings=_meanings(value))
        else:
            updated[card_index] = replace(record, usage_notes=value.strip())

    by_card: dict[int, dict[int, dict[str, str]]] = {}
    for (card_index, example_index, field), value in submission.example_fields.items():
        by_card.setdefault(card_index, {}).setdefault(example_index, {})[field] = value
    for card_index, examples in sorted(by_card.items()):
        if not 0 <= card_index < len(updated):
            raise EditError(f"Edit names card {card_index}, which is not on this page")
        record = updated[card_index]
        current = list(record.examples)
        for example_index, fields in sorted(examples.items()):
            if not 0 <= example_index < len(current):
                raise EditError(
                    f"Edit names example {example_index} on card {card_index}, "
                    "which is not on this page"
                )
            current[example_index] = replace(
                current[example_index],
                **{name: value.strip() for name, value in fields.items()},
            )
        updated[card_index] = replace(record, examples=tuple(current))
    return tuple(updated)


def changed_records(
    before: Sequence[VocabularyRecord], after: Sequence[VocabularyRecord]
) -> list[str]:
    """Which cards actually differ, by stable ID, for the saved message."""
    return [
        old.id
        for old, new in zip(before, after, strict=True)
        if old.to_dict() != new.to_dict()
    ]

