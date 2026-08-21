"""Where one staged record stands in the exact-example review, in one word.

The review panel grew this rule first, and the dashboard needs the identical
answer: a source may only be called **Ready to add** when every card on it has
human approval covering the exact Japanese sentences it currently carries. Two
implementations of that question is precisely the drift that would let a
browser page say *Ready to add* over cards `promote` then refuses — so the rule
lives here once and both callers import it.

`models.example_accepted` answers the narrower per-sentence question and is
what `promote` gates on. This module adds the two things a *reviewer-facing*
surface needs on top: that the hand-typed `staging-review` sentinel counts as
approval until promote replaces it with fingerprints, and that an approval
which no longer covers every current sentence is `stale` rather than simply
absent — a distinction a checkbox has to show, because re-approving is a
different act from approving.
"""

from __future__ import annotations

import re

from japanese_anki.models import (
    EXAMPLE_AUTHORITY_KEY,
    EXAMPLE_AUTHORITY_STAGING,
    VocabularyRecord,
    example_accepted,
)

__all__ = ["AUTHORITY_STATES", "example_authority_state", "needs_example_review"]

#: What `example_authority_state` may return.
#:
#: ``available``  — extract-sourced, has sentences, nobody has approved them.
#: ``stale``      — approved, but the approval does not cover every sentence
#:                  now on the card, so the card needs approving again.
#: ``existing``   — approved, covering every current sentence.
#: ``ineligible`` — not extract-sourced, or carries no Japanese sentence, so
#:                  there is nothing for this gate to approve.
#: ``invalid``    — the stored authority value is not a recognized shape; a
#:                  human has to correct it in the staging file.
AUTHORITY_STATES: tuple[str, ...] = (
    "available",
    "stale",
    "existing",
    "ineligible",
    "invalid",
)

# The shape `promote._accept_examples` and the review panel both write: one or
# more 12-hex-digit content fingerprints. Anything else in the field is a
# value no writer here produces and no reader may guess the meaning of.
_BOUND_AUTHORITY = re.compile(r"\s*[0-9a-f]{12}(?:\s*,\s*[0-9a-f]{12})*\s*\Z")


def example_authority_state(record: VocabularyRecord) -> str:
    """One of :data:`AUTHORITY_STATES` for this record's Japanese examples."""
    value = record.source.raw_fields.get(EXAMPLE_AUTHORITY_KEY, "")
    examples = [example for example in record.examples if example.japanese]
    if record.source.type != "extract" or not examples:
        return "ineligible"
    if value:
        if value == EXAMPLE_AUTHORITY_STAGING:
            return "existing"
        if _BOUND_AUTHORITY.fullmatch(value):
            if all(example_accepted(record, example) for example in examples):
                return "existing"
            return "stale"
        return "invalid"
    return "available"


def needs_example_review(record: VocabularyRecord) -> bool:
    """Whether a human still has to approve this card's exact sentences.

    ``invalid`` is deliberately not review-needing: a malformed authority value
    is a repair, not an approval, and offering a checkbox for it would let a
    reviewer overwrite evidence they never saw.
    """
    return example_authority_state(record) in {"available", "stale"}
