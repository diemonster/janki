"""One source, opened: its cards, its grammar, and what adding them would do.

`WORKBENCH_PLAN.md` W2. The dashboard answers "where is this source"; this
answers "show me the actual Japanese". It stays a projection — it reads files
and returns values, and writes nothing — so the same call backs a page refresh,
a CLI listing, and later the editor's before-picture.

The one piece of real reasoning here is the **merge preview**. When a source
proposes a word the collection already has, the person needs to see that
`promote`'s existing-wins rule will keep the meaning already on their card, not
the one this lesson proposed. Getting that from a second implementation of the
merge would be worse than not showing it at all: a preview that disagrees with
what promote does is a lie with a progress bar. So the preview calls
`promote.merge_staged_records` — the real one — against a copy, and reports
what it produced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from japanese_anki import patterns, promote, staging
from japanese_anki.application.authority import example_authority_state
from japanese_anki.application.journey import (
    GRAMMAR_NEEDS_REVIEW,
    GRAMMAR_REVIEWED,
    SourceJourney,
    source_journeys,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import load_records
from japanese_anki.models import VocabularyRecord
from japanese_anki.staging import LiveStaging

__all__ = ["CardDetail", "SourceDetail", "source_detail"]


@dataclass(frozen=True, slots=True)
class CardDetail:
    """One staged card, plus what the collection already knows about it."""

    record: VocabularyRecord
    authority: str
    hold_reason: str
    #: The canonical record sharing this stable ID, if the collection has one.
    existing: VocabularyRecord | None = None
    #: Meanings that would survive the merge. Empty when there is no conflict.
    merged_meanings: tuple[str, ...] = ()

    @property
    def is_new(self) -> bool:
        return self.existing is None

    @property
    def needs_example_review(self) -> bool:
        return self.authority in {"available", "stale"}

    @property
    def proposed_meanings_would_be_kept(self) -> bool:
        """Whether this lesson's meanings actually reach the collection.

        False is the interesting case and the reason the three panels exist:
        the card already has meanings, existing-wins keeps them, and the
        lesson's wording is recorded as source evidence rather than shown on
        the card. Nobody guesses that from a green checkmark.
        """
        if self.existing is None:
            return True
        return tuple(self.record.meanings) == self.merged_meanings


@dataclass(frozen=True, slots=True)
class SourceDetail:
    """Everything one source page needs, read from the repository."""

    journey: SourceJourney
    cards: tuple[CardDetail, ...]
    meta: dict[str, Any]
    pattern_set: patterns.PatternSet | None
    pattern_reviewed: bool
    #: Whether marking this grammar reviewed is even offerable. False when
    #: the store's lineage does not match this staging run — the panel
    #: refuses such a review, so offering the control would be a button
    #: that only ever produces an error.
    can_review_grammar: bool = False


def _merge_preview(
    existing: list[VocabularyRecord],
    staged: list[VocabularyRecord],
    meta: dict[str, Any],
) -> dict[str, VocabularyRecord]:
    """What `promote` would leave in the collection, without writing anything.

    Failures are swallowed on purpose. This is a preview beside a card, not a
    gate: if the real merge would refuse — a stale replacement binding, an
    unauthorized field — `promote` says so in its own words at the moment it
    matters. Showing a traceback here would replace a readable refusal with a
    scary one, on a page that changes nothing.
    """
    if not existing:
        return {}
    try:
        merged, _outcomes = promote.merge_staged_records(
            list(existing), list(staged), dict(meta)
        )
    except JankiError:
        return {}
    return {record.id: record for record in merged}


def source_detail(
    config: ProjectConfig,
    source: str,
    *,
    live: Sequence[LiveStaging] | None = None,
) -> SourceDetail | None:
    """Open one source by its dashboard name, or None if there is no such one.

    Lookup is by exact match against the names the dashboard itself computed,
    and the staging path comes from that journey — never from the caller. A
    request cannot name a path, so it cannot name a path outside the corpus.
    """
    journeys, _warnings = source_journeys(config, live=live)
    journey = next((one for one in journeys if one.source == source), None)
    if journey is None or journey.staging_path is None:
        return None

    try:
        records, meta = staging.read_staging(journey.staging_path)
    except JankiError:
        return None

    existing = (
        load_records(config.normalized_file)
        if config.normalized_file.exists()
        else []
    )
    by_id = {record.id: record for record in existing}
    merged = _merge_preview(existing, records, meta)

    cards = tuple(
        CardDetail(
            record=record,
            authority=example_authority_state(record),
            hold_reason=staging.annotations(record).get("hold_reason", ""),
            existing=by_id.get(record.id),
            merged_meanings=tuple(
                merged[record.id].meanings if record.id in merged else ()
            ),
        )
        for record in records
    )

    pattern_set = None
    raw = meta.get("pattern_set")
    if isinstance(raw, dict):
        try:
            pattern_set = patterns.PatternSet.from_dict(
                str(meta.get("source_file") or source), dict(raw)
            )
        except JankiError:
            pattern_set = None

    return SourceDetail(
        journey=journey,
        cards=cards,
        meta=meta,
        pattern_set=pattern_set,
        pattern_reviewed=journey.grammar == GRAMMAR_REVIEWED,
        can_review_grammar=journey.grammar
        in {GRAMMAR_REVIEWED, GRAMMAR_NEEDS_REVIEW},
    )
