"""Which exact deck-resolved records a preview was asked for.

The rendering itself lives in :mod:`japanese_anki.card_preview`, which draws
the real cards through the real exporters and Anki's own template engine. This
module answers the question that comes first and needs no Anki at all: given a
deck and a list of ids, which record *versions* would that deck build?
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from japanese_anki.errors import JankiError
from japanese_anki.exporters.anki import resolve_deck_records
from japanese_anki.models import VocabularyRecord


class PreviewError(JankiError):
    pass


def resolve_preview_records(
    deck_path: Path,
    record_ids: Sequence[str],
) -> tuple[VocabularyRecord, ...]:
    """Return exact deck-resolved record versions in requested order.

    Resolving the complete deck applies its real selectors and inline
    overrides. Missing IDs therefore refuse instead of previewing a canonical
    record that this deck would not actually build. No files are written.
    """
    if isinstance(record_ids, str | bytes | bytearray) or not isinstance(
        record_ids, Sequence
    ):
        raise PreviewError("Preview card IDs must be a sequence of IDs, not text.")
    requested_ids = tuple(record_ids)
    if not requested_ids:
        raise PreviewError("Preview needs at least one card ID.")
    if any(
        not isinstance(record_id, str) or not record_id.strip()
        for record_id in requested_ids
    ):
        raise PreviewError("Preview card IDs must be nonblank strings.")
    seen: set[str] = set()
    for record_id in requested_ids:
        if record_id in seen:
            raise PreviewError(f"Preview card ID {record_id!r} is repeated.")
        seen.add(record_id)

    _deck_config, resolved = resolve_deck_records(deck_path.resolve())
    by_id: dict[str, VocabularyRecord] = {}
    for record in resolved:
        if record.id in by_id:
            raise PreviewError(
                f"The deck resolves card ID {record.id!r} more than once."
            )
        by_id[record.id] = record
    missing = tuple(record_id for record_id in requested_ids if record_id not in by_id)
    if missing:
        names = ", ".join(repr(record_id) for record_id in missing)
        raise PreviewError(
            f"The requested card IDs are not in the resolved deck: {names}."
        )
    return tuple(by_id[record_id] for record_id in requested_ids)
