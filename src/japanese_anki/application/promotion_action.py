"""Bind a rendered promotion preview to the decision that may execute it.

Promotion itself already has one shared writer.  This module supplies the
small read-only seam a long-lived browser page needs around that writer: a
stable identity for the exact offline preview, and the CLI's existing rule for
consulting jpdb only when a reading could change the answer.

The second offline pass after the dictionary lookup is deliberate.  A source,
collection, archive, or deck rule may change while that lookup is in flight.
The live decision may execute only when the repository still projects to the
preview the person confirmed; its own compare-and-swap checks then protect the
remaining interval through the write.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from japanese_anki.application.promotion import (
    POST_READING_GATES,
    PromotionDecision,
    decide_promotion,
    project_promotion,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.jpdb import JpdbClient

__all__ = [
    "PromotionActionError",
    "promotion_preview_fingerprint",
    "resolve_promotion_for_execution",
]


class PromotionActionError(JankiError):
    """A rendered promotion preview is no longer executable as shown."""


def _record(value: Any | None) -> dict[str, Any] | None:
    return value.to_dict() if value is not None else None


def _ownership(value: Any) -> dict[str, Any]:
    return {
        "record_id": value.record_id,
        "state": value.state,
        "memberships": [
            {
                "stem": membership.stem,
                "name": membership.name,
                "takes": membership.takes,
                "refusal": membership.refusal,
            }
            for membership in value.memberships
        ],
        "unreadable_decks": list(value.unreadable_decks),
    }


def _path(value: Path | None) -> str | None:
    return str(value) if value is not None else None


def promotion_preview_fingerprint(decision: PromotionDecision) -> str:
    """Fingerprint every durable or rendered fact behind one offline preview."""
    plan = project_promotion(decision)
    revision = decision.output_revision
    claim = {
        "version": 1,
        "source": decision.source,
        "staging_path": str(decision.staging_path),
        "staging_sha256": hashlib.sha256(decision.wire).hexdigest(),
        "state": decision.state,
        "gate": decision.gate,
        "blocked": plan.blocked,
        "repository": {
            "root": str(decision.repository.root),
            "normalized_file": str(decision.repository.normalized_file),
            "deck_dir": str(decision.repository.deck_dir),
            "ledger_file": str(decision.repository.ledger_file),
            "staging_dir": str(decision.repository.staging_dir),
            "patterns_file": str(decision.repository.patterns_file),
        },
        "archive_path": _path(decision.done),
        "archive_records": [_record(record) for record in decision.archived],
        "archive_sha256": (
            None
            if decision.archive_revision is None
            else hashlib.sha256(decision.archive_revision).hexdigest()
        ),
        "retry_flags": list(decision.retry_flags),
        "already_archived": list(plan.already_archived),
        "collection_revision": (
            None
            if revision is None
            else {
                "path": str(revision.path),
                "sha256": (
                    hashlib.sha256(revision.text.encode("utf-8")).hexdigest()
                    if revision.text is not None
                    else None
                ),
            }
        ),
        "stored_ids": sorted(decision.stored_ids),
        "unreadable_decks": list(decision.unreadable_decks),
        "deck_revision": decision.deck_revision,
        "ledger_sha256": (
            None
            if decision.ledger_revision is None
            else hashlib.sha256(decision.ledger_revision).hexdigest()
        ),
        "landing": [
            {
                "staged": _record(card.staged),
                "landing": _record(card.landing),
                "existing": _record(card.existing),
                "reminted_from": card.reminted_from,
            }
            for card in plan.landing
        ],
        "held": [
            {"record": _record(card.record), "reason": card.reason}
            for card in plan.held
        ],
        "warnings": list(plan.warnings),
        "deck_ownership": [_ownership(item) for item in plan.deck_ownership],
        "readings_unchecked": plan.readings_unchecked,
    }
    encoded = json.dumps(
        claim,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _needs_dictionary(decision: PromotionDecision) -> bool:
    return decision.state in {"nothing_lands", "lands"} or (
        decision.is_blocked and decision.gate in POST_READING_GATES
    )


def _stale() -> PromotionActionError:
    return PromotionActionError(
        "[promotion-preview-stale] this source, collection, archive, or study-deck "
        "plan changed after the preview. Nothing was promoted; reload and check "
        "the current result."
    )


def _prelookup_binding(decision: PromotionDecision) -> tuple[Any, ...]:
    """Durable inputs read before a potentially blocking dictionary lookup."""
    return (
        decision.wire,
        (
            decision.archive_revision,
            decision.archived,
            decision.archived_meta,
        ),
        decision.output_revision,
        decision.stored_ids,
        decision.unreadable_decks,
        decision.deck_revision,
        decision.ledger_revision,
    )


def resolve_promotion_for_execution(
    config: ProjectConfig,
    offline: PromotionDecision,
    *,
    client_factory: Callable[[], JpdbClient],
    expected_preview_fingerprint: str | None = None,
) -> PromotionDecision:
    """Refresh an offline preview and consult jpdb only when it can matter.

    ``offline`` may carry explicit-skip authority from the CLI.  The workbench
    never creates that state: its preview is incomplete until this function
    obtains the dictionary witness at click time.
    """
    if offline.reading_check == "explicit_skip":
        # This is CLI authority, supplied on the command being executed now.
        return offline
    if not _needs_dictionary(offline):
        return offline

    live = decide_promotion(
        config,
        offline.staging_path,
        source=offline.source,
        client=client_factory(),
        skip_reading_check=False,
    )
    if expected_preview_fingerprint is None:
        # A CLI decision is made and consumed in one command. Preserve its
        # established gate call count; execute_promotion supplies the CAS.
        return live
    after_lookup = decide_promotion(
        config,
        offline.staging_path,
        source=offline.source,
        skip_reading_check=None,
    )
    if (
        promotion_preview_fingerprint(after_lookup)
        != expected_preview_fingerprint
        or _prelookup_binding(live) != _prelookup_binding(after_lookup)
    ):
        raise _stale()
    return live
