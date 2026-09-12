"""The shared promotion decision and execution service.

`WORKBENCH_PLAN.md` W1.1b. Two kinds of thing live here.

**Promote's own orchestration**, lifted out of `cli.py`: which
durable archive a staging run belongs to and which of its rows are already
there (`archive_for_run`), whether a path is the archive itself
(`inside_archive`), and the gates that refuse a file before anything is
written (`validate_record_archive`, `staged_ai_enrichment`,
`check_pattern_review`). `execute_promotion` is the one mutation path for both
surfaces: it writes canonical records and ledger state, completes pattern-only
reviews, and holds the live/archive locks through archive and pruning.

`plan_promotion` is the preview. It answers "what would adding this source do"
without doing it: which cards are new, which merge into a card you already
have and therefore keep the meanings already on it, which are held back and
why, and which would be filed under a different identity than the one they
were staged with.

**It is honest about what it cannot know offline**, which `readings_unchecked`
records — see that field for what the gap is and why it is left open.

Nothing in `plan_promotion` writes.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from japanese_anki import enrich, ledger, patterns, promote, staging
from japanese_anki import status as status_module
from japanese_anki.application.assignment import (
    DeckOwnershipEvaluation,
    require_exact_deck_ownership,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    MERGEABLE_FIELDS,
    DataError,
    MergeOutcome,
    RecordsRevision,
    atomic_unlink_bound,
    atomic_write_text_bound,
    exclusive_path_lock,
    load_records_snapshot,
    read_bytes_bound,
    records_json_text,
    save_records_json_locked,
)
from japanese_anki.models import (
    EXAMPLE_AUTHORITY_KEY,
    VocabularyRecord,
    set_example_flags,
)
from japanese_anki.promote import PromoteError
from japanese_anki.staging import (
    STAGING_SUFFIXES,
    StagingError,
    check_rewritable,
    finish_staging_under_lock,
    read_staging,
    render_staging_document,
    render_staging_finish,
    review_run_id,
    rich_extraction_review_run_id,
    validate_coverage_facts,
    write_staging_under_lock,
)

__all__ = [
    "PreparedSourcePromotion",
    "PromotionBatch",
    "PromotionExecutionResult",
    "PromotionRecovery",
    "HeldCard",
    "LandingCard",
    "PromotionDecision",
    "PromotionPlan",
    "DECISION_STATES",
    "POST_READING_GATES",
    "archive_for_run",
    "archive_run_provenance",
    "check_pattern_review",
    "record_review_snapshot",
    "staging_wire",
    "inside_archive",
    "apply_prepared_source_promotion",
    "decide_promotion",
    "execute_promotion",
    "prepare_source_promotion",
    "recover_promotion_intent",
    "recover_promotion_intent_under_guard",
    "plan_promotion",
    "promotion_batches",
    "project_promotion",
    "SourcePromotionFold",
    "SourcePromotionPart",
    "SourcePromotionProjection",
    "fold_source_extraction_promotions",
    "project_ai_enrichment_review_promotion",
    "project_card_revision_review_promotion",
    "project_source_extraction_review_promotion",
    "unreadable_deck_warning",
    "staged_ai_enrichment",
    "staged_card_revision",
    "validate_record_archive",
]


def archive_run_provenance(meta: Mapping[str, Any]) -> dict[str, Any]:
    """The model-run identity that decides whether an archive may be appended.

    New model-review files carry a persisted ``review_run_id`` as well as their
    request provenance. Partial promotion retries carry both byte-for-byte
    through the live staging file. A later invocation may reuse the staging
    basename and even the exact request, but its run id is different and its
    rows must not be folded into the earlier archive. Schema-v2 extraction
    staging is intentionally included: without a run id, its smaller
    ``prompt_provenance`` block remains the identity that version knew to record.

    Files older than model provenance retain their historical basename-based
    retry behavior through the explicit ``legacy`` identity.
    """
    run_id = review_run_id(meta)
    if staging.CARD_REVISION_KEY in meta:
        identity = {
            "kind": "card_revision",
            staging.CARD_REVISION_KEY: meta.get(staging.CARD_REVISION_KEY),
            "field_replacements": meta.get("field_replacements"),
        }
    elif staging.AI_ENRICHMENT_KEY in meta:
        identity = {
            "kind": "ai",
            staging.AI_ENRICHMENT_KEY: meta.get(staging.AI_ENRICHMENT_KEY),
            "field_replacements": meta.get("field_replacements"),
        }
    elif "prompt_provenance" in meta:
        identity = {
            "kind": "extract",
            "prompt_provenance": meta.get("prompt_provenance"),
        }
    else:
        identity = {"kind": "legacy"}
    if run_id is not None:
        identity["review_run_id"] = run_id
    return identity


def _archive_run_fingerprint(meta: Mapping[str, Any]) -> str:
    """Canonical digest of the full provenance used to select a done file."""
    try:
        encoded = json.dumps(
            archive_run_provenance(meta),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PromoteError(
            "This staging file has model-run provenance that cannot name a "
            f"durable archive: {exc}. Nothing was promoted."
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _select_archive_for_run(
    base: Path, meta: Mapping[str, Any], *, bind_read: bool
) -> tuple[Path, list[VocabularyRecord], dict[str, Any] | None, bytes | None]:
    """Choose this run's deterministic archive and any partial rows already there."""

    def read(path: Path) -> tuple[bytes | None, list[VocabularyRecord], dict[str, Any]]:
        if bind_read:
            return record_review_snapshot(path)
        records, archived_meta = read_staging(path)
        return None, records, archived_meta

    identity = archive_run_provenance(meta)
    if not base.exists():
        return base, [], None, None

    wire, previous, previous_meta = read(base)
    if archive_run_provenance(previous_meta) == identity:
        return base, list(previous), previous_meta, wire

    digest = _archive_run_fingerprint(meta)
    candidate = base.with_name(f"{base.stem}.{digest}{base.suffix}")
    if not candidate.exists():
        return candidate, [], None, None

    wire, previous, previous_meta = read(candidate)
    if archive_run_provenance(previous_meta) != identity:
        raise PromoteError(
            f"Archive provenance collision at {candidate}; nothing was promoted."
        )
    return candidate, list(previous), previous_meta, wire


def _archive_for_run_snapshot(
    base: Path, meta: Mapping[str, Any]
) -> tuple[Path, list[VocabularyRecord], dict[str, Any] | None, bytes | None]:
    """Choose and bind an archive for a read-only promotion decision."""
    return _select_archive_for_run(base, meta, bind_read=True)


def archive_for_run(
    base: Path, meta: Mapping[str, Any]
) -> tuple[Path, list[VocabularyRecord], dict[str, Any] | None]:
    """Choose this run's archive, including inside its already-held writer lock."""
    selected, records, archived_meta, _wire = _select_archive_for_run(
        base, meta, bind_read=False
    )
    return selected, records, archived_meta


def inside_archive(path: Path, archive_dir: Path) -> bool:
    """Is ``path`` the promoted archive, or inside it?

    Identity where the filesystem can answer it, because a lexical comparison
    is wrong on a case-insensitive filesystem: ``staging/Done/lesson.yaml``
    opens the real archive while comparing unequal to ``staging/done/...``, and
    promoting it doubles a committed file that is the only copy of a finished
    review. The lexical test stays as the fallback for a path that does not
    exist yet.
    """
    try:
        if archive_dir.exists() and path.parent.samefile(archive_dir):
            return True
    except OSError:
        pass
    return path.is_relative_to(archive_dir)


def _file_revision(path: Path) -> bytes | None:
    """Exact bytes at ``path``, with absence preserved as a distinct state."""
    try:
        return read_bytes_bound(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DataError(f"Could not read {path}: {exc.strerror or exc}") from exc


def _deck_configuration_revision(config: ProjectConfig) -> str:
    """Fingerprint the exact configured deck-file set and its bytes."""
    digest = hashlib.sha256()
    for path in status_module.deck_files(config):
        target = path.resolve()
        wire = _file_revision(target)
        if wire is None:
            raise DataError(f"Deck file vanished while it was read: {target}")
        encoded_path = str(target).encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(wire).to_bytes(8, "big"))
        digest.update(wire)
    return digest.hexdigest()


def _require_deck_inputs(
    config: ProjectConfig,
    existing_ids: Collection[str],
    *,
    stored_ids: set[str],
    unreadable_decks: Sequence[str],
    deck_revision: str,
) -> None:
    """Refuse deck/source rereads that differ from the pre-dictionary input.

    The collection enters as the ids it contributed rather than as its records,
    because that is all :func:`status.surviving_ids_from` reads and it is the
    part a prepared intent can carry: a resume has to reproduce the *original*
    collection's contribution, and the canonical file it would otherwise read
    may already hold the landing this very check is guarding.
    """
    revision_before = _deck_configuration_revision(config)
    current_ids, current_unreadable = status_module.surviving_ids_from(
        config, existing_ids
    )
    revision_after = _deck_configuration_revision(config)
    if (
        revision_before != deck_revision
        or revision_after != deck_revision
        or current_ids != stored_ids
        or tuple(current_unreadable) != tuple(unreadable_decks)
    ):
        raise PromoteError(
            "[promotion-input-stale] a configured deck or its declared records "
            "changed during the reading check. Reload the promotion preview."
        )


@dataclass(frozen=True, slots=True)
class PromotionBatch:
    """One exact, durable transaction recorded only in the done archive."""

    receipt_id: str
    archive_file: str
    archive_run_fingerprint: str
    archive_start_index: int
    source_file: str
    review_run_id: str | None
    promoted_ids: tuple[str, ...]
    owner_stems: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "archive_file": self.archive_file,
            "archive_run_fingerprint": self.archive_run_fingerprint,
            "archive_start_index": self.archive_start_index,
            "source_file": self.source_file,
            "review_run_id": self.review_run_id,
            "promoted_ids": list(self.promoted_ids),
            "owner_stems": {
                record_id: self.owner_stems[record_id]
                for record_id in self.promoted_ids
            },
        }


def _promotion_receipt_id(
    archive_file: str,
    archive_run_fingerprint: str,
    archive_start_index: int,
    source_file: str,
    promoted_ids: Sequence[str],
    owner_stems: Mapping[str, str],
) -> str:
    payload = {
        "archive_file": archive_file,
        "archive_run_fingerprint": archive_run_fingerprint,
        "archive_start_index": archive_start_index,
        "source_file": source_file,
        "promoted_ids": list(promoted_ids),
        "owner_stems": [
            [record_id, owner_stems[record_id]] for record_id in promoted_ids
        ],
    }
    wire = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def promotion_batches(
    meta: Mapping[str, Any],
    *,
    archived: Sequence[VocabularyRecord],
    archive_file: str,
) -> tuple[PromotionBatch, ...]:
    """Parse and validate the archive-only exact promotion receipts."""
    if staging.PROMOTION_BATCHES_KEY not in meta:
        return ()
    raw_batches = meta[staging.PROMOTION_BATCHES_KEY]
    if not isinstance(raw_batches, list):
        raise PromoteError(
            "[promotion-batches-invalid] promotion_batches must be a list"
        )
    if not raw_batches:
        raise PromoteError(
            "[promotion-batches-invalid] promotion_batches must be omitted when "
            "there are no receipts"
        )

    archive_source = meta.get("source_file")
    if "source_file" in meta and (
        not isinstance(archive_source, str) or not archive_source.strip()
    ):
        raise PromoteError(
            "[promotion-batches-invalid] archive source_file must be nonblank text"
        )
    try:
        archive_run_id = review_run_id(meta)
    except JankiError as exc:
        raise PromoteError(str(exc)) from exc
    archive_run_fingerprint = _archive_run_fingerprint(meta)

    parsed: list[PromotionBatch] = []
    seen_receipts: set[str] = set()
    seen_ids: set[str] = set()
    archived_ids = {record.id for record in archived}
    expected_keys = {
        "receipt_id",
        "archive_file",
        "archive_run_fingerprint",
        "archive_start_index",
        "source_file",
        "review_run_id",
        "promoted_ids",
        "owner_stems",
    }
    for position, raw in enumerate(raw_batches, start=1):
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise PromoteError(
                "[promotion-batches-invalid] each promotion batch must contain "
                f"exactly {', '.join(sorted(expected_keys))}; batch {position} does not"
            )
        receipt = raw.get("receipt_id")
        batch_archive_file = raw.get("archive_file")
        batch_run_fingerprint = raw.get("archive_run_fingerprint")
        batch_start_index = raw.get("archive_start_index")
        source_file = raw.get("source_file")
        batch_run_id = raw.get("review_run_id")
        ids = raw.get("promoted_ids")
        owners = raw.get("owner_stems")
        if (
            not isinstance(receipt, str)
            or len(receipt) != 64
            or any(character not in "0123456789abcdef" for character in receipt)
        ):
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} receipt_id must be "
                "lowercase SHA-256 text"
            )
        if (
            not isinstance(batch_archive_file, str)
            or not batch_archive_file.strip()
            or Path(batch_archive_file).name != batch_archive_file
            or batch_archive_file in {".", ".."}
        ):
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} archive_file must "
                "be one plain nonblank filename"
            )
        if batch_archive_file != archive_file:
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} names a different done archive file"
            )
        if parsed and batch_archive_file != parsed[0].archive_file:
            raise PromoteError(
                "[promotion-batches-invalid] every promotion batch in one archive "
                "must name that same archive file"
            )
        if batch_run_fingerprint != archive_run_fingerprint:
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} belongs to different "
                "archive-run provenance"
            )
        if (
            not isinstance(batch_start_index, int)
            or isinstance(batch_start_index, bool)
            or batch_start_index < 0
        ):
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} "
                "archive_start_index must be a nonnegative integer"
            )
        if parsed and batch_start_index != parsed[0].archive_start_index:
            raise PromoteError(
                "[promotion-batches-invalid] every promotion batch in one archive "
                "must preserve the same receipt start index"
            )
        if not isinstance(source_file, str) or not source_file.strip():
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} source_file must be nonblank text"
            )
        if batch_run_id is not None:
            try:
                batch_run_id = review_run_id({"review_run_id": batch_run_id})
            except JankiError as exc:
                raise PromoteError(str(exc)) from exc
        if archive_source is not None and source_file != archive_source.strip():
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} names a different "
                "source_file from its archive"
            )
        if batch_run_id != archive_run_id:
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} belongs to a "
                "different review run from its archive"
            )
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids)
        ):
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} promoted_ids must "
                "be a nonempty list of unique record ids"
            )
        if not isinstance(owners, Mapping) or set(owners) != set(ids):
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} owner_stems must "
                "exactly key its promoted_ids"
            )
        if any(
            not isinstance(owners[record_id], str) or not owners[record_id].strip()
            for record_id in ids
        ):
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} owner stems must be nonblank text"
            )
        if receipt != _promotion_receipt_id(
            batch_archive_file,
            batch_run_fingerprint,
            batch_start_index,
            source_file,
            ids,
            owners,
        ):
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} receipt_id does not "
                "bind its exact source, run, ids, and owners"
            )
        overlap = seen_ids.intersection(ids)
        if receipt in seen_receipts or overlap:
            raise PromoteError(
                "[promotion-batches-invalid] promotion batches must have distinct "
                "receipt ids and disjoint promoted ids"
            )
        if not set(ids) <= archived_ids:
            raise PromoteError(
                f"[promotion-batches-invalid] batch {position} names a record not "
                "present in the cumulative archive"
            )
        batch = PromotionBatch(
            receipt_id=receipt,
            archive_file=batch_archive_file,
            archive_run_fingerprint=batch_run_fingerprint,
            archive_start_index=batch_start_index,
            source_file=source_file,
            review_run_id=batch_run_id,
            promoted_ids=tuple(ids),
            owner_stems={record_id: str(owners[record_id]) for record_id in ids},
        )
        parsed.append(batch)
        seen_receipts.add(receipt)
        seen_ids.update(ids)
    archived_order = tuple(record.id for record in archived)
    if len(set(archived_order)) != len(archived_order):
        raise PromoteError(
            "[promotion-batches-invalid] the cumulative archive must contain unique record ids"
        )
    batch_order = tuple(
        record_id for batch in parsed for record_id in batch.promoted_ids
    )
    start_index = parsed[0].archive_start_index
    if archived_order[start_index:] != batch_order:
        raise PromoteError(
            "[promotion-batches-invalid] promotion batches must name every "
            "receipted archive row in exact append order"
        )
    return tuple(parsed)


def _latest_retry_batch(
    archived_meta: Mapping[str, Any] | None,
    archived: Sequence[VocabularyRecord],
    retry_ids: Sequence[str],
    *,
    empty_live: bool = False,
    archive_file: str,
) -> PromotionBatch | None:
    """Recover only the latest completed batch from an exact cleanup retry."""
    batches = promotion_batches(
        archived_meta or {}, archived=archived, archive_file=archive_file
    )
    if not batches:
        return None
    latest = batches[-1]
    if empty_live or set(latest.promoted_ids) <= set(retry_ids):
        return latest
    return None


def _new_promotion_batch(
    meta: Mapping[str, Any],
    *,
    source: str,
    promoted: Sequence[VocabularyRecord],
    ownership: Sequence[DeckOwnershipEvaluation],
    archive_file: str,
    archive_start_index: int,
) -> PromotionBatch:
    raw_source = meta.get("source_file") if "source_file" in meta else source
    if not isinstance(raw_source, str) or not raw_source.strip():
        raise PromoteError(
            "[promotion-batches-invalid] a promotion receipt needs a canonical source_file"
        )
    source_file = raw_source.strip()
    run_id = review_run_id(meta)
    run_fingerprint = _archive_run_fingerprint(meta)
    promoted_ids = tuple(record.id for record in promoted)
    if not promoted_ids or len(set(promoted_ids)) != len(promoted_ids):
        raise PromoteError(
            "[promotion-batches-invalid] a promotion receipt needs nonempty, unique promoted ids"
        )
    by_id = {item.record_id: item for item in ownership}
    if set(by_id) != set(promoted_ids):
        raise PromoteError(
            "[promotion-batches-invalid] re-proved deck ownership does not exactly "
            "cover the promoted ids"
        )
    owner_stems: dict[str, str] = {}
    for record_id in promoted_ids:
        stems = by_id[record_id].owner_stems
        if by_id[record_id].state != "exactly_one" or len(stems) != 1:
            raise PromoteError(
                f"[promotion-batches-invalid] {record_id} has no single re-proved study-deck owner"
            )
        stem = stems[0]
        if not isinstance(stem, str) or not stem.strip():
            raise PromoteError(
                f"[promotion-batches-invalid] {record_id}'s re-proved study-deck "
                "owner has no usable filename stem"
            )
        owner_stems[record_id] = stem
    receipt = _promotion_receipt_id(
        archive_file,
        run_fingerprint,
        archive_start_index,
        source_file,
        promoted_ids,
        owner_stems,
    )
    return PromotionBatch(
        receipt_id=receipt,
        archive_file=archive_file,
        archive_run_fingerprint=run_fingerprint,
        archive_start_index=archive_start_index,
        source_file=source_file,
        review_run_id=run_id,
        promoted_ids=promoted_ids,
        owner_stems=owner_stems,
    )


def validate_record_archive(
    live_meta: Mapping[str, Any],
    archived: Sequence[VocabularyRecord],
    archived_meta: Mapping[str, Any] | None,
    *,
    archive_file: str,
) -> None:
    """Prove a same-run record archive is intact before relying on it."""
    if archived_meta is None:
        if archived:
            raise PromoteError(
                "[record-archive-divergent] archive rows have no metadata"
            )
        return
    if not archived:
        # A zero-row same-run archive is still a completed review. In normal
        # operation it is the pattern-only archive, whose reviewed_pattern_set
        # exists only in the done copy. A later record-bearing live file must
        # not turn absence of rows into absence of an archive and overwrite it.
        # Exact pattern-only retries are routed to their stricter validator by
        # execute_promotion before the record-archive validator is called.
        raise PromoteError(
            "[record-archive-divergent] the same review run already has a "
            "completed zero-record archive; later record rows cannot be appended "
            "to it. Both the live review and done archive were kept."
        )

    # The done copy is durable paid evidence too. A retry may delete the only
    # live copy, so validate it independently rather than trusting the live
    # block or merely comparing the run-id subset used for archive selection.
    validate_coverage_facts(archived_meta)
    promote.check_coverage(dict(archived_meta))
    promote.check_candidate_accounting(
        live_meta, (), archived, archived_meta=archived_meta
    )
    promotion_batches(archived_meta, archived=archived, archive_file=archive_file)

    live_core = {
        key: value for key, value in live_meta.items() if key != "review_notes"
    }
    archive_core = {
        key: value
        for key, value in archived_meta.items()
        if key not in {"review_notes", staging.PROMOTION_BATCHES_KEY}
    }
    if archive_core != live_core:
        raise PromoteError(
            "[record-archive-divergent] the same-run archive metadata differs from the live review"
        )
    note = archived_meta.get("review_notes")
    suffix = f"Promoted {len(archived)} record(s) from this file."
    if not isinstance(note, str) or not (
        note.strip() == suffix or note.strip().endswith(f"\n\n{suffix}")
    ):
        raise PromoteError(
            "[record-archive-divergent] the same-run archive does not record "
            "its exact archived row count"
        )


class _AiLedgerHandoffIncomplete(Exception):
    """Keep the live review after records landed but AI attribution did not."""


def staged_card_revision(
    meta: Mapping[str, Any],
    records: Sequence[VocabularyRecord | str],
    *,
    archived_ids: Sequence[str] = (),
    archived_records: Sequence[VocabularyRecord] = (),
    require_owner_review: bool = True,
) -> tuple[str, str, dict[str, tuple[str, tuple[str, ...]]]] | None:
    """Validate provenance for a paid revision of canonical cards.

    The ordinary AI-enrichment pass may author only three prose fields.  An
    explicitly requested ``revise`` pass may author any mergeable card field,
    so it has a distinct metadata block while returning the same narrow facts
    the canonical promotion writer needs for ledger attribution.
    """

    raw = meta.get(staging.CARD_REVISION_KEY)
    if raw is None:
        return None
    if staging.AI_ENRICHMENT_KEY in meta:
        raise PromoteError(
            "A staging file cannot claim both ai_enrichment and "
            "card_revision provenance. Nothing was promoted."
        )
    required = {
        "version",
        "operation_id",
        "request_fingerprint",
        "provider",
        "attribution_provider",
        "model",
        "input_fingerprints",
        "fields",
    }
    allowed = required | {"focus_resource_id"}
    if (
        not isinstance(raw, Mapping)
        or not required.issubset(raw)
        or not set(raw) <= allowed
    ):
        raise PromoteError(
            "card_revision has invalid provenance fields. Nothing was promoted."
        )
    version = raw.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise PromoteError("card_revision must be a version 1 metadata block")
    operation_id = raw.get("operation_id")
    try:
        canonical_operation_id = str(uuid.UUID(operation_id))
    except (AttributeError, TypeError, ValueError) as exc:
        raise PromoteError(
            "card_revision operation_id must be a canonical UUIDv4"
        ) from exc
    if (
        not isinstance(operation_id, str)
        or canonical_operation_id != operation_id
        or uuid.UUID(operation_id).version != 4
    ):
        raise PromoteError("card_revision operation_id must be a canonical UUIDv4")
    request_fp = raw.get("request_fingerprint")
    if (
        not isinstance(request_fp, str)
        or len(request_fp) != 64
        or any(character not in "0123456789abcdef" for character in request_fp)
    ):
        raise PromoteError(
            "card_revision request_fingerprint must be a lowercase SHA-256"
        )
    transport = raw.get("provider")
    attribution = raw.get("attribution_provider")
    expected_attribution = {
        "anthropic-api": "anthropic",
        "claude-code": "anthropic",
    }.get(transport)
    if expected_attribution is None or attribution != expected_attribution:
        raise PromoteError("card_revision has an invalid provider attribution")
    if attribution not in ledger.AI_PROVIDERS:
        raise PromoteError("card_revision has an unsupported provider attribution")
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise PromoteError("card_revision needs a nonblank model")
    focused = raw.get("focus_resource_id")
    if "focus_resource_id" in raw and (
        not isinstance(focused, str) or not focused.strip()
    ):
        raise PromoteError("card_revision focus_resource_id must be nonblank text")
    inputs = raw.get("input_fingerprints")
    fields = raw.get("fields")
    replacement_block = meta.get(staging.FIELD_REPLACEMENTS_KEY)
    replacements = (
        replacement_block.get("records")
        if isinstance(replacement_block, Mapping)
        else None
    )
    if not all(isinstance(value, Mapping) for value in (inputs, fields, replacements)):
        raise PromoteError(
            "card_revision needs input, field, and replacement maps. Nothing was promoted."
        )
    assert isinstance(inputs, Mapping)
    assert isinstance(fields, Mapping)
    assert isinstance(replacements, Mapping)
    id_sets = [set(values) for values in (inputs, fields, replacements)]
    if any(
        any(not isinstance(record_id, str) or not record_id for record_id in values)
        for values in (inputs, fields, replacements)
    ) or any(ids != id_sets[0] for ids in id_sets[1:]):
        raise PromoteError(
            "card_revision input, field, and replacement maps must name "
            "identical record ids. Nothing was promoted."
        )
    provenance_ids = id_sets[0]
    current_ids = {
        record.id if isinstance(record, VocabularyRecord) else str(record)
        for record in records
    }
    known_ids = current_ids | {str(record_id) for record_id in archived_ids}
    missing = sorted(current_ids - provenance_ids)
    unknown = sorted(provenance_ids - known_ids)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing current " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise PromoteError(
            "card_revision provenance does not match this staging file "
            "or its done archive: " + "; ".join(details) + ". Nothing was promoted."
        )

    proven: dict[str, tuple[str, tuple[str, ...]]] = {}
    for record_id in sorted(provenance_ids):
        input_fp = inputs.get(record_id)
        raw_fields = fields.get(record_id)
        replacement_fields = replacements.get(record_id)
        if (
            not isinstance(input_fp, str)
            or len(input_fp) != 64
            or any(character not in "0123456789abcdef" for character in input_fp)
            or not isinstance(raw_fields, list)
            or not raw_fields
            or any(not isinstance(name, str) for name in raw_fields)
            or any(name not in MERGEABLE_FIELDS for name in raw_fields)
            or len(set(raw_fields)) != len(raw_fields)
            or not isinstance(replacement_fields, Mapping)
            or any(not isinstance(name, str) for name in replacement_fields)
            or set(raw_fields) != set(replacement_fields)
        ):
            raise PromoteError(
                f"card_revision has incomplete provenance for {record_id}"
            )
        proven[record_id] = (request_fp, tuple(raw_fields))
    if not require_owner_review:
        return str(attribution), model.strip(), proven
    review = meta.get(staging.CARD_REVISION_REVIEW_KEY)
    if not isinstance(review, Mapping) or set(review) != {
        "version",
        "authority",
        "accepted_record_ids",
        "content_fingerprint",
    }:
        raise PromoteError(
            "card_revision has no exact durable owner review. Nothing was promoted."
        )
    accepted = review.get("accepted_record_ids")
    review_fp = review.get("content_fingerprint")
    if (
        review.get("version") != staging.CARD_REVISION_REVIEW_VERSION
        or review.get("authority") != "repository-owner"
        or not isinstance(accepted, list)
        or any(not isinstance(item, str) for item in accepted)
        or len(accepted) != len(set(accepted))
        or set(accepted) != provenance_ids
        or not isinstance(review_fp, str)
    ):
        raise PromoteError(
            "card_revision has invalid durable owner review. Nothing was promoted."
        )
    review_records = [record for record in records if isinstance(record, VocabularyRecord)]
    archive_ids = {str(record_id) for record_id in archived_ids}
    archived_record_ids = {record.id for record in archived_records}
    if (
        len(review_records) != len(records)
        or archived_record_ids != archive_ids
        or len(archived_record_ids) != len(archived_records)
        or current_ids != set(accepted) - archive_ids
    ):
        raise PromoteError(
            "card_revision durable owner review cannot be revalidated against its "
            "exact live and archived rows. Nothing was promoted."
        )
    for record in [*review_records, *archived_records]:
        reviewed_fields = fields.get(record.id)
        if not isinstance(reviewed_fields, list) or "examples" not in reviewed_fields:
            continue
        expected = set_example_flags(
            record,
            EXAMPLE_AUTHORITY_KEY,
            (
                example.japanese
                for example in record.examples
                if record.source.type == "extract" and example.japanese
            ),
        ).source.raw_fields.get(EXAMPLE_AUTHORITY_KEY)
        if record.source.raw_fields.get(EXAMPLE_AUTHORITY_KEY) != expected:
            raise PromoteError(
                "card_revision exact example authority changed after owner review. "
                "Nothing was promoted."
            )
    try:
        actual_review_fp = staging.card_revision_review_fingerprint(
            meta, [*review_records, *archived_records]
        )
    except JankiError as exc:
        raise PromoteError(str(exc)) from exc
    if not hmac.compare_digest(review_fp, actual_review_fp):
        raise PromoteError(
            "card_revision changed after owner review. Nothing was promoted."
        )
    return str(attribution), model.strip(), proven


def staged_ai_enrichment(
    meta: Mapping[str, Any],
    records: Sequence[VocabularyRecord | str],
    *,
    archived_ids: Sequence[str] = (),
    archived_records: Sequence[VocabularyRecord] = (),
    require_owner_review: bool = False,
) -> tuple[str, str, dict[str, tuple[str, tuple[str, ...]]]] | None:
    """Validate AI provenance before a staged review can write anything.

    A partial promote keeps the original top-level metadata while moving some
    rows to the corresponding ``done`` archive.  Provenance may therefore name
    a row absent from the current staging file only when that exact id is
    already in the archive.  An older, completed run under the same basename is
    the opposite shape: archive-only ids need not appear in this run's maps.
    """
    revision = staged_card_revision(
        meta,
        records,
        archived_ids=archived_ids,
        archived_records=archived_records,
    )
    if revision is not None:
        return revision
    raw = meta.get(staging.AI_ENRICHMENT_KEY)
    if raw is None:
        if "field_replacements" in meta:
            raise PromoteError(
                "This staging file authorizes field replacements but has no "
                "ai_enrichment provenance. Nothing was promoted."
            )
        return None
    version = raw.get("version") if isinstance(raw, Mapping) else None
    if (
        not isinstance(raw, Mapping)
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version != 1
    ):
        raise PromoteError("ai_enrichment must be a version 1 metadata block")
    raw_model = raw.get("model")
    model = raw_model.strip() if isinstance(raw_model, str) else ""
    raw_provider = raw.get("provider")
    provider = raw_provider.strip() if isinstance(raw_provider, str) else ""
    focused = raw.get("focus_resource_id")
    if "focus_resource_id" in raw and (
        not isinstance(focused, str) or not focused.strip()
    ):
        raise PromoteError("ai_enrichment focus_resource_id must be nonblank text")
    requests = raw.get("request_fingerprints")
    inputs = raw.get("input_fingerprints")
    fields = raw.get("fields")
    if (
        not model
        or provider not in ledger.AI_PROVIDERS
        or not all(isinstance(value, Mapping) for value in (requests, inputs, fields))
    ):
        raise PromoteError(
            "ai_enrichment needs a provider, model, plus request, input, and field maps"
        )
    replacement_block = meta.get("field_replacements")
    replacement_records = (
        replacement_block.get("records")
        if isinstance(replacement_block, Mapping)
        else None
    )
    if not isinstance(replacement_records, Mapping):
        raise PromoteError(
            "ai_enrichment needs its field_replacements record map. Nothing was promoted."
        )

    maps = {
        "request_fingerprints": requests,
        "input_fingerprints": inputs,
        "fields": fields,
        "field_replacements": replacement_records,
    }
    id_sets: dict[str, set[str]] = {}
    for label, values in maps.items():
        if any(not isinstance(record_id, str) or not record_id for record_id in values):
            raise PromoteError(f"{label} record ids must be non-empty text")
        id_sets[label] = set(values)
    provenance_ids = id_sets["request_fingerprints"]
    if any(ids != provenance_ids for ids in id_sets.values()):
        raise PromoteError(
            "ai_enrichment request, input, field, and replacement maps must "
            "name identical record ids. Nothing was promoted."
        )

    current_ids = {
        record.id if isinstance(record, VocabularyRecord) else str(record)
        for record in records
    }
    known_ids = current_ids | {str(record_id) for record_id in archived_ids}
    missing = sorted(current_ids - provenance_ids)
    unknown = sorted(provenance_ids - known_ids)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing current " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise PromoteError(
            "ai_enrichment provenance does not match this staging file or its "
            "done archive: " + "; ".join(details) + ". Nothing was promoted."
        )

    proven: dict[str, tuple[str, tuple[str, ...]]] = {}
    for record_id in sorted(provenance_ids):
        request_fp = requests.get(record_id)
        input_fp = inputs.get(record_id)
        raw_fields = fields.get(record_id)
        replacement_fields = replacement_records.get(record_id)
        if (
            not isinstance(request_fp, str)
            or len(request_fp) != 64
            or any(character not in "0123456789abcdef" for character in request_fp)
            or not isinstance(input_fp, str)
            or len(input_fp) != 64
            or any(character not in "0123456789abcdef" for character in input_fp)
            or not isinstance(raw_fields, list)
            or not raw_fields
            or any(not isinstance(name, str) for name in raw_fields)
            or any(name not in enrich.AI_FIELDS for name in raw_fields)
            or len(set(raw_fields)) != len(raw_fields)
            or not isinstance(replacement_fields, Mapping)
            or any(not isinstance(name, str) for name in replacement_fields)
            or set(raw_fields) != set(replacement_fields)
        ):
            raise PromoteError(
                f"ai_enrichment has incomplete provenance for {record_id}"
            )
        proven[record_id] = (
            request_fp,
            tuple(dict.fromkeys(str(name) for name in raw_fields)),
        )
    if not require_owner_review:
        return provider, model, proven
    review = meta.get(staging.AI_ENRICHMENT_REVIEW_KEY)
    if not isinstance(review, Mapping) or set(review) != {
        "version",
        "authority",
        "accepted_record_ids",
        "content_fingerprint",
    }:
        raise PromoteError(
            "ai_enrichment has no exact durable owner review. Nothing was promoted."
        )
    accepted = review.get("accepted_record_ids")
    review_fp = review.get("content_fingerprint")
    if (
        review.get("version") != staging.AI_ENRICHMENT_REVIEW_VERSION
        or review.get("authority") != "repository-owner"
        or not isinstance(accepted, list)
        or any(not isinstance(item, str) for item in accepted)
        or len(accepted) != len(set(accepted))
        or set(accepted) != provenance_ids
        or not isinstance(review_fp, str)
    ):
        raise PromoteError(
            "ai_enrichment has invalid durable owner review. Nothing was promoted."
        )
    review_records = [record for record in records if isinstance(record, VocabularyRecord)]
    archive_ids = {str(record_id) for record_id in archived_ids}
    archived_record_ids = {record.id for record in archived_records}
    if (
        len(review_records) != len(records)
        or archived_record_ids != archive_ids
        or len(archived_record_ids) != len(archived_records)
        or current_ids != set(accepted) - archive_ids
    ):
        raise PromoteError(
            "ai_enrichment durable owner review cannot be revalidated against its "
            "exact live and archived rows. Nothing was promoted."
        )
    for record in [*review_records, *archived_records]:
        expected = set_example_flags(
            record,
            EXAMPLE_AUTHORITY_KEY,
            (
                example.japanese
                for example in record.examples
                if record.source.type == "extract" and example.japanese
            ),
        ).source.raw_fields.get(EXAMPLE_AUTHORITY_KEY)
        if record.source.raw_fields.get(EXAMPLE_AUTHORITY_KEY) != expected:
            raise PromoteError(
                "ai_enrichment exact example authority changed after owner review. "
                "Nothing was promoted."
            )
    try:
        actual_review_fp = staging.ai_enrichment_review_fingerprint(
            meta, [*review_records, *archived_records]
        )
    except JankiError as exc:
        raise PromoteError(str(exc)) from exc
    if not hmac.compare_digest(review_fp, actual_review_fp):
        raise PromoteError(
            "ai_enrichment changed after owner review. Nothing was promoted."
        )
    return provider, model, proven


@dataclass(frozen=True, slots=True)
class LandingCard:
    """One staged card that would reach the collection, and how."""

    staged: VocabularyRecord
    #: The record as it would be stored: re-minted identity, merged fields.
    landing: VocabularyRecord
    #: The canonical record it would merge into, if there is one.
    existing: VocabularyRecord | None
    #: The id it was staged under, when promoting would change it. Empty
    #: otherwise. A changed id is a different Anki note — worth saying out
    #: loud before it happens rather than after.
    reminted_from: str = ""

    @property
    def is_new(self) -> bool:
        return self.existing is None

    @property
    def keeps_existing_meanings(self) -> bool:
        """Whether the collection's wording wins over this source's.

        The case people are surprised by. `promote` is existing-wins, so a
        word you already have keeps the meaning already on it and this
        lesson's wording becomes source evidence rather than card text.
        """
        if self.existing is None or not self.staged.meanings:
            return False
        return list(self.landing.meanings) != list(self.staged.meanings)


@dataclass(frozen=True, slots=True)
class HeldCard:
    """One staged card that would stay in the staging file, and why."""

    record: VocabularyRecord
    reason: str


@dataclass(frozen=True, slots=True)
class _PromotionRepositoryBinding:
    """The repository paths one decision was derived from."""

    root: Path
    normalized_file: Path
    deck_dir: Path
    ledger_file: Path
    staging_dir: Path
    patterns_file: Path


def _repository_binding(config: ProjectConfig) -> _PromotionRepositoryBinding:
    return _PromotionRepositoryBinding(
        root=config.root.resolve(),
        normalized_file=config.normalized_file.resolve(),
        deck_dir=config.deck_dir.resolve(),
        ledger_file=config.ledger_file.resolve(),
        staging_dir=config.staging_dir.resolve(),
        patterns_file=config.patterns_file.resolve(),
    )


@dataclass(frozen=True, slots=True)
class PromotionPlan:
    """What adding this source would do. Computed without writing anything."""

    source: str
    staging_path: Path
    #: A refusal that stops the whole promote before any row is considered —
    #: the coverage gate, an unparseable file. Empty when nothing blocks it.
    blocked: str = ""
    landing: tuple[LandingCard, ...] = ()
    held: tuple[HeldCard, ...] = ()
    #: Rows this run already wrote to its durable archive. Promoting again
    #: removes them from the live review rather than adding them twice.
    already_archived: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    #: The exact configured word-deck owners proved for every landing card,
    #: in landing order. Safe to render; execution re-proves the same verdict
    #: under the deck-directory lock before the canonical save.
    deck_ownership: tuple[DeckOwnershipEvaluation, ...] = ()
    #: Whether the dictionary witness was skipped — load-bearing rather than
    #: decorative.
    #:
    #: The check costs a paid lookup per row, so a plan asked without a client
    #: declines it, and then a row listed in `landing` may still be held when
    #: the real promote asks jpdb whether its reading is one a dictionary
    #: lists. Handed a client, the plan makes the call and this says so, which
    #: is how a caller tells a complete decision from a partial one.
    #:
    #: True on a plan that was blocked before the check could run, and on the
    #: early returns that never reach it — in all of which the witness really
    #: was not consulted.
    readings_unchecked: bool = True

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked)

    @property
    def adding(self) -> tuple[LandingCard, ...]:
        return tuple(card for card in self.landing if card.is_new)

    @property
    def merging(self) -> tuple[LandingCard, ...]:
        return tuple(card for card in self.landing if not card.is_new)

    @property
    def reminted(self) -> dict[str, str]:
        return {
            card.reminted_from: card.landing.id
            for card in self.landing
            if card.reminted_from
        }


def unreadable_deck_warning(problem: str) -> str:
    """Why a row needing a new id stays put when a deck will not parse.

    One sentence, one place. The command prints it and the plan carries it, and
    two copies of a sentence that explains a *refusal* drift into explaining
    two different refusals.
    """
    return (
        f"{problem}; ids in that deck cannot be checked, so any row needing a "
        "new id stays in the staging file until it parses."
    )


def check_pattern_review(
    config: ProjectConfig,
    meta: Mapping[str, Any],
    run_id: str,
    *,
    pattern_store: Mapping[str, patterns.PatternSet] | None = None,
) -> None:
    """Refuse a zero-record rich extraction whose grammar nobody has reviewed.

    The completion contract for a file that proposed only patterns. Lifted out
    of `_complete_pattern_only_review`'s locked transaction so the preview can
    ask the same question without taking the lock or writing the archive — the
    checks themselves read files and decide, which is all either caller needs
    from them.

    ``pattern_store`` answers from a supplied mapping instead of the live file.
    A study finish prepares one aggregate post-review store for **every** part
    before it projects any of them, so a part whose grammar this same finish is
    about to mark reviewed projects `pattern_only` rather than the blocker a
    live read would still report. Ordinary callers pass nothing and read the
    file, and a decision built on an injection is `projected`.
    """
    source = meta.get("source_file")
    raw_pattern_set = meta.get("pattern_set")
    if (
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(raw_pattern_set, Mapping)
    ):
        raise PromoteError(
            "[pattern-review-invalid] a rich pattern-only extraction needs "
            "its source_file and pattern_set. Nothing was archived."
        )
    patterns.PatternSet.from_dict(source, dict(raw_pattern_set))
    stored_patterns = (
        patterns.load_store(config.patterns_file) if pattern_store is None else pattern_store
    ).get(source)
    if stored_patterns is None or not stored_patterns.reviewed:
        raise PromoteError(
            f"[patterns-unreviewed] {source} has not been reviewed. Run "
            f"'janki patterns --review {source}' first; nothing was archived."
        )
    provenance = meta.get("prompt_provenance")
    if (
        stored_patterns.review_run_id != run_id
        or not isinstance(provenance, Mapping)
        or stored_patterns.prompt_provenance != dict(provenance)
    ):
        raise PromoteError(
            f"[patterns-review-stale] the reviewed {source} pattern-store "
            "entry belongs to a different extraction run. Nothing was archived."
        )


def _pattern_only_archive_meta(
    meta: Mapping[str, Any], reviewed: patterns.PatternSet
) -> dict[str, Any]:
    """Keep the paid answer raw and add the corrected human-reviewed snapshot."""
    archived = dict(meta)
    archived["reviewed_pattern_set"] = reviewed.to_dict()
    note = "Reviewed pattern-only extraction; no records were promoted."
    existing = str(archived.get("review_notes") or "").strip()
    archived["review_notes"] = f"{existing}\n\n{note}" if existing else note
    return archived


def _complete_pattern_only_review(
    config: ProjectConfig,
    path: Path,
    done: Path,
    expected_meta: Mapping[str, Any],
    expected_wire: bytes,
    run_id: str,
    *,
    pattern_store: Mapping[str, patterns.PatternSet] | None = None,
    precheck: Callable[[Path], None] | None = None,
) -> tuple[Path, bool]:
    """Archive one reviewed zero-record v3 run as a locked CAS transaction.

    ``precheck`` runs once, with both the live review and the selected archive
    locked and before either is written, so a prepared intent can measure its
    whole component vector at the one moment nothing else can move it.
    """
    with exclusive_path_lock(path):
        current_wire = staging_wire(path)
        records, meta = read_staging(path)
        if (
            current_wire != expected_wire
            or records
            or dict(meta) != dict(expected_meta)
        ):
            raise PromoteError(
                "[staging-review-stale] the live staging file changed while its "
                "pattern review was being completed. The replacement was kept; "
                "nothing was archived."
            )

        # Re-check inside the lock. The first check happened before the CAS so
        # coverage acceptance could run without holding a file lock across a
        # paid call; this one binds the archive to the bytes about to be removed.
        promote.check_coverage(meta)
        if rich_extraction_review_run_id(meta) != run_id:
            raise PromoteError(
                "[staging-review-stale] the rich extraction run changed. Nothing was archived."
            )

        source = meta.get("source_file")
        raw_pattern_set = meta.get("pattern_set")
        if (
            not isinstance(source, str)
            or not source.strip()
            or not isinstance(raw_pattern_set, Mapping)
        ):
            raise PromoteError(
                "[pattern-review-invalid] a rich pattern-only extraction needs "
                "its source_file and pattern_set. Nothing was archived."
            )
        # Structural parsing only. Human corrections belong in the store and
        # deliberately need not equal this immutable paid proposal.
        patterns.PatternSet.from_dict(source, dict(raw_pattern_set))
        # Injected only by a preparation that has not yet written the store it
        # is about to write; the ordinary executor reads the live file, which
        # by apply time already holds the same aggregate after-store.
        stored_patterns = (
            patterns.load_store(config.patterns_file)
            if pattern_store is None
            else pattern_store
        ).get(source)
        if stored_patterns is None or not stored_patterns.reviewed:
            raise PromoteError(
                f"[patterns-unreviewed] {source} has not been reviewed. Run "
                f"'janki patterns --review {source}' first; nothing was archived."
            )
        provenance = meta.get("prompt_provenance")
        if (
            stored_patterns.review_run_id != run_id
            or not isinstance(provenance, Mapping)
            or stored_patterns.prompt_provenance != dict(provenance)
        ):
            raise PromoteError(
                f"[patterns-review-stale] the reviewed {source} pattern-store "
                "entry belongs to a different extraction run. Nothing was archived."
            )

        check_rewritable(path)
        if path.suffix.lower() not in STAGING_SUFFIXES:
            raise PromoteError(_unrewritable(path))
        archived_meta = _pattern_only_archive_meta(meta, stored_patterns)
        archive_base = done
        while True:
            selected, _archived, _archived_meta = archive_for_run(archive_base, meta)
            with exclusive_path_lock(selected):
                # Selection itself reads existing archives. Confirm after the
                # selected path is locked: another completion may have created
                # the base or candidate between those two operations.
                confirmed, _archived, _archived_meta = archive_for_run(
                    archive_base, meta
                )
                if confirmed != selected:
                    continue
                done = selected
                if precheck is not None:
                    precheck(done)
                retried = done.exists()
                if retried:
                    archive_records, existing_meta = read_staging(done)
                    if archive_records or existing_meta != archived_meta:
                        raise PromoteError(
                            f"[pattern-archive-divergent] {done} already names this "
                            "review run but is not the exact completed archive. "
                            "Both files were kept; nothing was overwritten."
                        )
                else:
                    # `selected` is already locked. The ordinary writer would
                    # acquire this non-reentrant lock again and deadlock.
                    write_staging_under_lock(done, [], archived_meta)
                    archive_records, written_meta = read_staging(done)
                    if archive_records or written_meta != archived_meta:
                        raise PromoteError(
                            f"[pattern-archive-divergent] {done} did not read "
                            "back as the exact completed archive. The live "
                            "review was kept."
                        )

                # Last, while both the live staging path and selected done path
                # remain locked. Neither a failed archive write, a concurrent
                # forced extraction, nor a force-write to the verified archive
                # can remove the only recoverable review artifact in between.
                try:
                    atomic_unlink_bound(
                        path,
                        expected_revision=hashlib.sha256(expected_wire).hexdigest(),
                    )
                except JankiError as exc:
                    raise PromoteError(
                        "[pattern-completion-incomplete] the reviewed pattern "
                        "archive completed, but janki could not prove that the "
                        "live staging review was retired unchanged. Inspect both "
                        "copies and rerun this exact promotion; do not delete "
                        "staging by hand."
                    ) from exc
                return done, retried


def _finish_record_review(
    path: Path,
    archive_base: Path,
    *,
    expected_wire: bytes,
    expected_meta: Mapping[str, Any],
    expected_archived: Sequence[VocabularyRecord],
    expected_archived_meta: Mapping[str, Any] | None,
    expected_archive_revision: bytes | None,
    promoted: Sequence[VocabularyRecord],
    retry_records: Sequence[VocabularyRecord],
    keep: Sequence[bool],
    held: Sequence[VocabularyRecord],
    canonical_commit: Callable[[Path, int], PromotionBatch | None] | None = None,
    precheck: Callable[[Path], None] | None = None,
) -> tuple[Path, int, PromotionBatch | None]:
    """Commit, archive, and retire rows as one live/done locked transaction.

    The optional canonical commit runs only after both snapshots have been
    revalidated and while the selected done path remains locked. This closes
    the window where a same-run zero-row completion could appear after
    preflight but before vocabulary and ledger writes.

    ``precheck`` runs once, with both paths locked and before **any** write, so
    a prepared intent can prove its whole component vector — including the
    frozen archive selection it is about to append to — at the one moment
    nothing else can move it.
    """
    if len(keep) - sum(keep) != len(promoted) + len(retry_records):
        raise PromoteError(
            "[record-promotion-invalid] row disposition does not match the archive transaction"
        )
    if sum(keep) != len(held):
        raise PromoteError(
            "[record-promotion-invalid] held rows do not match the live staging remainder"
        )

    with exclusive_path_lock(path):
        if staging_wire(path) != expected_wire:
            raise PromoteError(
                "[staging-review-stale] the live staging file changed while "
                "promotion was completing. The replacement was kept."
            )
        current_records, current_meta = read_staging(path)
        if dict(current_meta) != dict(expected_meta) or len(current_records) != len(
            keep
        ):
            raise PromoteError(
                "[staging-review-stale] the live staging review no longer "
                "matches the validated snapshot. The replacement was kept."
            )

        validate_coverage_facts(current_meta)
        promote.check_coverage(current_meta)
        while True:
            selected, _rows, _meta = archive_for_run(archive_base, current_meta)
            with exclusive_path_lock(selected):
                confirmed, archived, archived_meta = archive_for_run(
                    archive_base, current_meta
                )
                if confirmed != selected:
                    continue
                if (
                    _file_revision(confirmed) != expected_archive_revision
                    or (list(archived) != list(expected_archived))
                    or (None if archived_meta is None else dict(archived_meta))
                    != (
                        None
                        if expected_archived_meta is None
                        else dict(expected_archived_meta)
                    )
                ):
                    raise PromoteError(
                        "[record-archive-stale] the done archive changed while "
                        "promotion was completing. The live review was kept."
                    )
                validate_record_archive(
                    current_meta,
                    archived,
                    archived_meta,
                    archive_file=confirmed.name,
                )
                retry_flags = promote.check_candidate_accounting(
                    current_meta,
                    retry_records,
                    archived,
                    archived_meta=archived_meta,
                )
                if retry_records and not all(retry_flags):
                    raise PromoteError(
                        "[archive-retry-divergent] rows marked as archive retries "
                        "are not exact promoted rows in the done archive"
                    )
                if any(
                    promote.check_candidate_accounting(
                        current_meta,
                        promoted,
                        archived,
                        archived_meta=archived_meta,
                    )
                ):
                    raise PromoteError(
                        "[archive-retry-divergent] a pending row already exists in the done archive"
                    )

                if precheck is not None:
                    precheck(confirmed)

                prior_batches = promotion_batches(
                    archived_meta or {},
                    archived=archived,
                    archive_file=confirmed.name,
                )
                archive_start_index = (
                    prior_batches[0].archive_start_index
                    if prior_batches
                    else len(archived)
                )
                completed_batch = (
                    canonical_commit(confirmed, archive_start_index)
                    if canonical_commit is not None
                    else None
                )

                done = confirmed
                combined = list(archived) + list(promoted)
                if promoted:
                    try:
                        completed_meta = promote.archive_meta(
                            dict(current_meta), len(combined)
                        )
                        if completed_batch is not None:
                            if completed_batch.promoted_ids != tuple(
                                record.id for record in promoted
                            ):
                                raise PromoteError(
                                    "[promotion-batches-invalid] committed receipt "
                                    "does not exactly name this archive append"
                                )
                            prior_batches = (*prior_batches, completed_batch)
                        if prior_batches:
                            completed_meta[staging.PROMOTION_BATCHES_KEY] = [
                                batch.to_dict() for batch in prior_batches
                            ]
                            promotion_batches(
                                completed_meta,
                                archived=combined,
                                archive_file=confirmed.name,
                            )
                        write_staging_under_lock(
                            done,
                            combined,
                            completed_meta,
                            force=True,
                            expected_revision=(
                                hashlib.sha256(expected_archive_revision).hexdigest()
                                if expected_archive_revision is not None
                                else None
                            ),
                            expected_absent=expected_archive_revision is None,
                        )
                        written, written_meta = read_staging(done)
                        if written != combined or written_meta != completed_meta:
                            raise PromoteError(
                                f"[record-archive-divergent] {done} did not read "
                                "back as the exact completed archive. The live "
                                "review was kept."
                            )
                    except JankiError as exc:
                        if canonical_commit is not None:
                            raise PromoteError(
                                "[promotion-completion-incomplete] cards reached "
                                "the collection, but janki could not complete and "
                                "verify their archive. The live review was kept. "
                                "Inspect it and the archive, then rerun this exact "
                                "promotion; do not delete staging by hand."
                            ) from exc
                        raise

                live_revision = hashlib.sha256(expected_wire).hexdigest()
                try:
                    if held:
                        removed = finish_staging_under_lock(
                            path,
                            keep,
                            held,
                            expected_revision=live_revision,
                        )
                    else:
                        removed = len(keep) - sum(keep)
                        atomic_unlink_bound(
                            path,
                            expected_revision=live_revision,
                        )
                except JankiError as exc:
                    if canonical_commit is not None:
                        raise PromoteError(
                            "[promotion-completion-incomplete] cards reached the "
                            "collection and archive, but janki could not prove "
                            "that the live staging review was retired unchanged. "
                            "Inspect the live review and archive, then rerun this "
                            "exact promotion; do not delete staging by hand."
                        ) from exc
                    raise PromoteError(
                        "[staging-review-stale] janki could not prove that the "
                        "live staging review was retired unchanged. Inspect it "
                        "and retry; do not delete staging by hand."
                    ) from exc
                return done, removed, completed_batch


#: What a decide pass concluded, in one word.
#:
#: `blocked` — a gate refused; `error` carries it.
#: `nothing` — a legacy file holding no records; promote says so and exits 0.
#: `archive_retry` — an empty live file beside this run's own archive, or a
#: file whose every row is already in it: completion, not a new review.
#: `pattern_only` — a zero-record rich extraction whose grammar was reviewed.
#: `nothing_lands` — rows exist and every one is held back.
#: `lands` — rows would reach the collection.
#: Gates reached only *after* the readings pass, so what they refuse depends
#: on which rows survived it.
#:
#: An offline decision cannot settle these: it promotes every row a dictionary
#: would have held, so it judges a superset — and two rows can collide on one
#: landing id offline that a consulted run would never have brought together.
#: A caller that decides offline first has to come back with the client before
#: believing a refusal from here.
POST_READING_GATES = frozenset({"accounting", "deck", "merge", "ledger"})

#: What a decision did about the dictionary. `preview` consulted nothing and
#: cannot execute a landing; `explicit_skip` was authorized to skip; `consulted`
#: asked a real witness.
READING_CHECKS = ("preview", "explicit_skip", "consulted")

DECISION_STATES = (
    "blocked",
    "nothing",
    "archive_retry",
    "pattern_only",
    "nothing_lands",
    "lands",
)


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Everything `promote` works out before it writes a byte.

    The shared middle of the command and the preview. `plan_promotion`
    projects it into something safe to render; `execute_promotion` consumes it
    and performs the transaction for either surface.

    Why a projection rather than one public value: this carries the raw wire
    snapshot, the resolved merge and the compare-and-swap token — which is
    everything a caller would need to replay the write path *around* the gates
    that produced them. `docs/DESIGN.md` says the workbench may not weaken an
    authority gate the CLI enforces, so the boundary between this and
    `PromotionPlan` is a safety boundary and not a matter of taste.

    Constructing this decision writes nothing. The staging snapshot — the
    bytes and the parse — is taken under the writer lock, because the execution
    service's compare-and-swap trusts the two to describe each other. Nothing
    else is locked: the compare-and-swap is the protection for the collection,
    and holding a lock across the dictionary lookups would be worse than the
    race it closed.
    """

    source: str
    staging_path: Path
    state: str
    repository: _PromotionRepositoryBinding
    reading_check: Literal["preview", "explicit_skip", "consulted"]
    #: The refusal itself, not its text. The command re-raises it, so stderr
    #: and the exit code are what they always were, and its *type* survives —
    #: different gates raise different classes, and flattening them changes
    #: which handler catches a refusal.
    error: JankiError | None = None
    #: Which gate refused, when one did.
    #:
    #: Its own field because the type cannot answer this: `check_coverage`
    #: re-wraps whatever it catches into a plain `PromoteError`, so a coverage
    #: refusal, an accounting refusal and an archive-divergence refusal are
    #: indistinguishable by class. The only other discriminator is the
    #: `[coverage-unresolved]`-style tag in the message, and deciding whether
    #: to spend a model call by matching strings is exactly the fragility
    #: `--accept-coverage` must not rest on.
    gate: str = ""
    #: Whether the dictionary was actually consulted before this concluded.
    consulted: bool = False

    #: The staging file exactly as read, for the write path's compare-and-swap.
    wire: bytes = b""
    meta: Mapping[str, Any] = field(default_factory=dict)
    records: tuple[VocabularyRecord, ...] = ()

    #: This run's durable archive, its rows, and its metadata.
    done: Path | None = None
    archived: tuple[VocabularyRecord, ...] = ()
    archived_meta: Mapping[str, Any] | None = None
    #: Exact archive bytes from the same read that produced the rows and
    #: metadata. ``None`` preserves absence as a distinct preview state.
    archive_revision: bytes | None = None
    #: One flag per input row: already in this run's archive.
    retry_flags: tuple[bool, ...] = ()
    #: The ids this run has already archived — computed once, here, because
    #: two shapes reach `archive_retry` (an empty live file beside a full
    #: archive, and a full file whose every row is already in it) and they
    #: read it from different places. Deriving it in the projection *and*
    #: again in the write path is two chances to disagree about what already
    #: landed.
    already_archived: tuple[str, ...] = ()

    #: The rows this run may still act on, and what the readings pass decided.
    work: tuple[VocabularyRecord, ...] = ()
    readings: promote.PromoteResult | None = None
    #: One flag per promoted row, from the accounting call made *after* the
    #: readings pass. Carried rather than assumed: the first call already
    #: probes each row's resolved id and its stable re-mint, so these are
    #: provably all false by the time they exist — but the write path composes
    #: its prune flags from them, and "provably" is not a thing to build a
    #: staging prune on.
    promoted_retry_flags: tuple[bool, ...] = ()

    #: The collection as it was read, and the token proving it has not moved.
    existing: tuple[VocabularyRecord, ...] = ()
    output_path: Path | None = None
    output_revision: RecordsRevision | None = None
    #: Every id the reading gate treated as already present, including inline
    #: deck notes.  This is derived before a possibly blocking dictionary
    #: lookup and therefore belongs to the same snapshot as ``existing``.
    stored_ids: frozenset[str] = frozenset()
    #: Exact operational inputs sampled before the reading lookup. Coverage
    #: dispatch compares them after that blocking call: selector and ledger
    #: changes need not alter declared ids, parsing, or the first gate label.
    deck_revision: str = ""
    ledger_revision: bytes | None = None
    #: Decks whose ids could not be read, verbatim — the two warning streams
    #: stay separate because the command prints these *before* the readings
    #: warnings and the plan reports them after.
    unreadable_decks: tuple[str, ...] = ()

    ai_provenance: Any = None
    merged: tuple[VocabularyRecord, ...] = ()
    outcomes: Mapping[str, MergeOutcome] = field(default_factory=dict)
    #: The exact selector verdicts proved immediately before this decision.
    #: Execution re-proves them at the canonical commit seam, because a
    #: checked browser action may sit open while a deck file changes.
    deck_ownership: tuple[DeckOwnershipEvaluation, ...] = ()

    #: Whether any part of this decision was injected rather than read.
    #:
    #: True for a staging snapshot, a canonical snapshot or a pattern-store
    #: snapshot supplied by a caller. Such a decision describes a repository
    #: state that does not exist on disk yet — the next part of a chained
    #: projection, a review the writer has not performed — so it is a preview,
    #: not a transaction. `execute_promotion` refuses one at function entry.
    projected: bool = False

    @property
    def is_blocked(self) -> bool:
        return self.state == "blocked"

    @property
    def reading_warnings(self) -> tuple[str, ...]:
        return tuple(self.readings.warnings) if self.readings else ()

    @property
    def deck_warnings(self) -> tuple[str, ...]:
        return tuple(unreadable_deck_warning(one) for one in self.unreadable_decks)


PromotionExecutionState = Literal[
    "nothing",
    "pattern_only",
    "archive_retry",
    "nothing_lands",
    "landed",
    "landed_ledger_incomplete",
    "landed_ai_ledger_incomplete",
]


@dataclass(frozen=True, slots=True)
class PromotionExecutionResult:
    """The durable result of consuming one validated promotion decision.

    It contains facts, not terminal prose. The CLI formats these fields while
    the workbench can render the same committed transaction without parsing a
    command transcript.
    """

    state: PromotionExecutionState
    staging_path: Path
    archive_path: Path | None = None
    output_path: Path | None = None
    promoted: tuple[VocabularyRecord, ...] = ()
    held: tuple[VocabularyRecord, ...] = ()
    retry_records: tuple[VocabularyRecord, ...] = ()
    removed: int = 0
    empty_live_retry: bool = False
    archive_was_retry: bool = False
    reminted: Mapping[str, str] = field(default_factory=dict)
    outcomes: Mapping[str, MergeOutcome] = field(default_factory=dict)
    ledger_added: int = 0
    ledger_sources: int = 0
    ledger_error: ledger.LedgerError | None = None
    #: Opaque handle for the exact archive batch. Absent when canonical records
    #: landed but archive completion was deliberately deferred for AI-ledger
    #: recovery.
    receipt_id: str | None = None

    @property
    def promoted_ids(self) -> tuple[str, ...]:
        """The exact newly landed scope, in transaction order."""
        return tuple(record.id for record in self.promoted)

    def __post_init__(self) -> None:
        archive_required = self.state in {
            "pattern_only",
            "archive_retry",
            "landed",
            "landed_ledger_incomplete",
        } or (self.state == "nothing_lands" and bool(self.retry_records))
        if (self.archive_path is not None) != archive_required:
            raise ValueError(
                f"Promotion result {self.state!r} has an inconsistent archive path"
            )

        landed = self.state in {
            "landed",
            "landed_ledger_incomplete",
            "landed_ai_ledger_incomplete",
        }
        output_expected = landed or self.state == "nothing_lands"
        if landed != bool(self.promoted) or output_expected != (
            self.output_path is not None
        ):
            raise ValueError(
                f"Promotion result {self.state!r} has an inconsistent landing scope"
            )

        ledger_failed = self.state in {
            "landed_ledger_incomplete",
            "landed_ai_ledger_incomplete",
        }
        if (self.ledger_error is not None) != ledger_failed:
            raise ValueError(
                f"Promotion result {self.state!r} has an inconsistent ledger result"
            )
        receipt_required = self.state in {"landed", "landed_ledger_incomplete"}
        receipt_allowed = receipt_required or self.state in {
            "archive_retry",
            "nothing_lands",
        }
        if receipt_required and self.receipt_id is None:
            raise ValueError(
                f"Promotion result {self.state!r} has an inconsistent archive receipt"
            )
        if not receipt_allowed and self.receipt_id is not None:
            raise ValueError(
                f"Promotion result {self.state!r} has an inconsistent archive receipt"
            )
        if self.receipt_id is not None and (
            len(self.receipt_id) != 64
            or any(character not in "0123456789abcdef" for character in self.receipt_id)
        ):
            raise ValueError("Promotion receipt ids must be lowercase SHA-256 text")


def staging_wire(path: Path) -> bytes:
    """Read the exact live-review bytes used by pattern-only promotion's CAS."""
    try:
        return read_bytes_bound(path)
    except (DataError, OSError) as exc:
        raise PromoteError(
            f"[staging-review-stale] could not snapshot {path}: "
            f"{getattr(exc, 'strerror', None) or exc}. Nothing was archived."
        ) from exc


def record_review_snapshot(
    path: Path,
) -> tuple[bytes, list[VocabularyRecord], dict[str, Any]]:
    """Read one exact live-review snapshot while its writer lock is held."""
    with exclusive_path_lock(path):
        wire = staging_wire(path)
        try:
            text = wire.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise DataError(f"Could not parse staging file {path}: {exc}") from exc
        records, meta = staging.read_staging_text(text, source=str(path))
    return wire, records, meta


def project_ai_enrichment_review_promotion(
    config: ProjectConfig,
    staging_path: Path,
    record_ids: Sequence[str],
    *,
    expected_revision: str,
) -> PromotionDecision:
    """Project ordinary promotion after one exact pending enrichment review."""

    path = staging_path.resolve()
    wire, records, meta = record_review_snapshot(path)
    if hashlib.sha256(wire).hexdigest() != expected_revision:
        raise PromoteError(
            "[ai-enrichment-review-stale] the proposal changed while its "
            "aggregate finish was planned"
        )
    if staging.AI_ENRICHMENT_REVIEW_KEY in meta:
        raise PromoteError(
            "[ai-enrichment-review-invalid] this proposal already records owner review"
        )
    selected = tuple(record_ids)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(not isinstance(item, str) or not item for item in selected)
    ):
        raise PromoteError(
            "[ai-enrichment-review-invalid] accepted record ids must be nonempty "
            "and unique"
        )
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records) or any(item not in by_id for item in selected):
        raise PromoteError(
            "[ai-enrichment-review-invalid] accepted record ids must identify "
            "unique proposal rows"
        )
    staged_ai_enrichment(meta, records)
    selected_set = set(selected)
    chosen = [record for record in records if record.id in selected_set]
    reviewed = [
        set_example_flags(
            record,
            EXAMPLE_AUTHORITY_KEY,
            (
                example.japanese
                for example in record.examples
                if record.source.type == "extract" and example.japanese
            ),
        )
        for record in chosen
    ]
    projected = deepcopy(meta)
    enrichment = projected.get(staging.AI_ENRICHMENT_KEY)
    replacements = projected.get(staging.FIELD_REPLACEMENTS_KEY)
    if not isinstance(enrichment, dict) or not isinstance(replacements, dict):
        raise PromoteError(
            "[ai-enrichment-review-invalid] proposal metadata is not projectable"
        )
    for key in ("request_fingerprints", "input_fingerprints", "fields"):
        values = enrichment.get(key)
        if not isinstance(values, dict):
            raise PromoteError(
                f"[ai-enrichment-review-invalid] ai_enrichment.{key} is not projectable"
            )
        enrichment[key] = {
            record_id: value
            for record_id, value in values.items()
            if record_id in selected_set
        }
    replacement_records = replacements.get("records")
    if not isinstance(replacement_records, dict):
        raise PromoteError(
            "[ai-enrichment-review-invalid] field replacements are not projectable"
        )
    replacements["records"] = {
        record_id: value
        for record_id, value in replacement_records.items()
        if record_id in selected_set
    }
    projected[staging.AI_ENRICHMENT_REVIEW_KEY] = {
        "version": staging.AI_ENRICHMENT_REVIEW_VERSION,
        "authority": "repository-owner",
        "accepted_record_ids": [record.id for record in chosen],
        "content_fingerprint": staging.ai_enrichment_review_fingerprint(
            meta, chosen
        ),
    }
    source = projected.get("source_file")
    return decide_promotion(
        config,
        path,
        source=source if isinstance(source, str) else "",
        _record_snapshot=(wire, reviewed, projected),
    )


def project_card_revision_review_promotion(
    config: ProjectConfig,
    staging_path: Path,
    record_ids: Sequence[str],
    *,
    expected_revision: str,
) -> PromotionDecision:
    """Project promotion after one exact pending card-revision review.

    The owner needs to see review, promotion, audio, and package effects before
    a single confirmation.  This applies the review writer's deterministic
    selection and marker to an in-memory snapshot, then runs the ordinary
    promotion planner over that projected document.  Execution still invokes
    the real review writer and replans promotion from disk before anything
    canonical lands.
    """

    path = staging_path.resolve()
    wire, records, meta = record_review_snapshot(path)
    if hashlib.sha256(wire).hexdigest() != expected_revision:
        raise PromoteError(
            "[card-revision-review-stale] the proposal changed while its "
            "aggregate finish was planned"
        )
    if staging.CARD_REVISION_REVIEW_KEY in meta:
        raise PromoteError(
            "[card-revision-review-invalid] this proposal already records owner review"
        )
    selected = tuple(record_ids)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(not isinstance(item, str) or not item for item in selected)
    ):
        raise PromoteError(
            "[card-revision-review-invalid] accepted record ids must be nonempty "
            "and unique"
        )
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records) or any(item not in by_id for item in selected):
        raise PromoteError(
            "[card-revision-review-invalid] accepted record ids must identify "
            "unique proposal rows"
        )
    staged_card_revision(meta, records, require_owner_review=False)
    selected_set = set(selected)
    chosen = [record for record in records if record.id in selected_set]
    raw_revision = meta.get(staging.CARD_REVISION_KEY)
    declared = (
        raw_revision.get("fields") if isinstance(raw_revision, Mapping) else None
    )
    if not isinstance(declared, Mapping):
        raise PromoteError(
            "[card-revision-review-invalid] proposal fields are not projectable"
        )
    reviewed = [
        (
            set_example_flags(
                record,
                EXAMPLE_AUTHORITY_KEY,
                (
                    example.japanese
                    for example in record.examples
                    if record.source.type == "extract" and example.japanese
                ),
            )
            if isinstance(declared.get(record.id), list)
            and "examples" in declared[record.id]
            else record
        )
        for record in chosen
    ]
    projected = deepcopy(meta)
    revision = projected.get(staging.CARD_REVISION_KEY)
    replacements = projected.get(staging.FIELD_REPLACEMENTS_KEY)
    if not isinstance(revision, dict) or not isinstance(replacements, dict):
        raise PromoteError(
            "[card-revision-review-invalid] proposal metadata is not projectable"
        )
    for key in ("input_fingerprints", "fields"):
        values = revision.get(key)
        if not isinstance(values, dict):
            raise PromoteError(
                f"[card-revision-review-invalid] card_revision.{key} is not projectable"
            )
        revision[key] = {
            record_id: value
            for record_id, value in values.items()
            if record_id in selected_set
        }
    replacement_records = replacements.get("records")
    if not isinstance(replacement_records, dict):
        raise PromoteError(
            "[card-revision-review-invalid] field replacements are not projectable"
        )
    replacements["records"] = {
        record_id: value
        for record_id, value in replacement_records.items()
        if record_id in selected_set
    }
    projected[staging.CARD_REVISION_REVIEW_KEY] = {
        "version": staging.CARD_REVISION_REVIEW_VERSION,
        "authority": "repository-owner",
        "accepted_record_ids": [record.id for record in reviewed],
        "content_fingerprint": staging.card_revision_review_fingerprint(
            meta, reviewed
        ),
    }
    source = projected.get("source_file")
    return decide_promotion(
        config,
        path,
        source=source if isinstance(source, str) else "",
        _record_snapshot=(wire, reviewed, projected),
    )


def project_source_extraction_review_promotion(
    config: ProjectConfig,
    staging_path: Path,
    *,
    expected_revision: str,
    review_record_ids: Sequence[str],
    review_patterns: bool,
    pattern_store: Mapping[str, patterns.PatternSet],
    coverage_approval: Mapping[str, Any] | None,
    collection: Sequence[VocabularyRecord],
    collection_revision: RecordsRevision,
    witness: enrich.DictionaryLookup,
) -> PromotionDecision:
    """Project promotion after one part's exact review and coverage decisions.

    The owner sees review, coverage and promotion effects before one
    confirmation, so all three have to be modelled before any of them is
    written. This applies the review writer's own example flags and the
    owner's own coverage payload to an in-memory snapshot, then runs the
    ordinary planner over it against an injected canonical state and the one
    aggregate post-review pattern store every part in the batch sees.

    Every injection marks the result `projected`, which `execute_promotion`
    refuses at entry. Execution re-decides from the repository once the
    `reviewed` phase has actually written these bytes.
    """

    path = staging_path.resolve()
    wire, records, meta = record_review_snapshot(path)
    if hashlib.sha256(wire).hexdigest() != expected_revision:
        raise PromoteError(
            "[source-review-stale] the proposal changed while its aggregate "
            "finish was planned"
        )
    selected = tuple(review_record_ids)
    if len(selected) != len(set(selected)) or any(
        not isinstance(item, str) or not item for item in selected
    ):
        raise PromoteError(
            "[source-review-invalid] reviewed record ids must be unique nonblank text"
        )
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records):
        raise PromoteError(
            "[source-review-invalid] this proposal has duplicate record ids, so a "
            "review selection cannot identify one row"
        )
    missing = [item for item in selected if item not in by_id]
    if missing:
        raise PromoteError(
            "[source-review-invalid] reviewed record ids name rows this proposal "
            f"does not hold: {', '.join(sorted(missing))}"
        )

    # Exactly what `ReviewPanel.submit` writes, including its lack of a
    # `source.type == "extract"` filter. The two neighbouring projections do
    # filter, and copying either of them here would flag a different set from
    # the writer this finish actually runs.
    chosen = set(selected)
    reviewed = [
        (
            set_example_flags(
                record,
                EXAMPLE_AUTHORITY_KEY,
                (example.japanese for example in record.examples if example.japanese),
            )
            if record.id in chosen
            else record
        )
        for record in records
    ]

    projected_meta = deepcopy(meta)
    if coverage_approval is not None:
        block = projected_meta.get("coverage")
        if not isinstance(block, dict):
            raise PromoteError(
                "[source-review-invalid] this proposal carries no coverage block to "
                "approve, so the prepared approval describes another file"
            )
        block["approval"] = deepcopy(dict(coverage_approval))

    source_name = projected_meta.get("source_file")
    if review_patterns:
        # The one aggregate store must already carry this part's chosen mark;
        # otherwise the batch prepared a store that does not answer for the
        # decision this part is projecting under.
        entry = (
            pattern_store.get(source_name) if isinstance(source_name, str) else None
        )
        if entry is None or not entry.reviewed:
            raise PromoteError(
                "[source-review-invalid] the prepared pattern store does not carry "
                "this part's selected review mark"
            )

    return decide_promotion(
        config,
        path,
        source=source_name if isinstance(source_name, str) else "",
        client=witness,
        skip_reading_check=False,
        _record_snapshot=(wire, reviewed, projected_meta),
        _collection_snapshot=(collection, collection_revision),
        _pattern_store_snapshot=pattern_store,
    )


@dataclass(frozen=True, slots=True)
class SourcePromotionPart:
    """One part the fold projects, with the owner decisions taken over it."""

    part_name: str
    staging_path: Path
    #: sha256 of the staging bytes as they are **before** this finish's own
    #: review write. The projection refuses a file that moved since.
    expected_revision: str
    review_record_ids: tuple[str, ...] = ()
    review_patterns: bool = False
    coverage_approval: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SourcePromotionProjection:
    """What one part would do to the canonical collection, in fold order."""

    part_name: str
    staging_path: Path
    staging_sha256: str
    state: str
    gate: str
    #: sha256(text_{k-1}) and sha256(text_k). ``None`` means the canonical file
    #: is absent at that checkpoint, which a fresh repository really is.
    expected_before: str | None
    expected_after: str | None
    landed_ids: tuple[str, ...] = ()
    held_ids: tuple[str, ...] = ()
    excluded_ids: tuple[str, ...] = ()
    archive_retry_ids: tuple[str, ...] = ()
    #: In-memory only. It carries the injected snapshots and is never
    #: serialized into an authority.
    decision: PromotionDecision | None = None

    @property
    def lands(self) -> bool:
        return self.state == "lands"

    def to_dict(self) -> dict[str, Any]:
        """The part of this projection an authority may bind."""
        return {
            "part_name": self.part_name,
            "staging_path": str(self.staging_path),
            "staging_sha256": self.staging_sha256,
            "state": self.state,
            "gate": self.gate,
            "expected_before": self.expected_before,
            "expected_after": self.expected_after,
            "landed_ids": list(self.landed_ids),
            "held_ids": list(self.held_ids),
            "excluded_ids": list(self.excluded_ids),
            "archive_retry_ids": list(self.archive_retry_ids),
        }


@dataclass(frozen=True, slots=True)
class SourcePromotionFold:
    """The whole batch's chained canonical projection."""

    parts: tuple[SourcePromotionProjection, ...]
    #: sha256(text_0) … sha256(text_N), one more entry than there are parts.
    canonical_digests: tuple[str | None, ...]
    #: carry_N — the post-promotion collection every later phase reads.
    records_after: tuple[VocabularyRecord, ...]
    canonical_text_after: str | None


def _disclosed_ids(decision: PromotionDecision) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    """Landed, held, excluded and archive-retry ids for one decision.

    Held and excluded are different facts. A held row is one the reading
    witness or a structural hold kept in the live file with its reason; an
    excluded row is one this part proposed that no landing and no hold
    accounts for — a row already in this run's archive is disclosed under
    ``archive_retry`` instead. None of them is ever reported as landed.

    **One definition, used twice.** The fold discloses a projected part with
    it and `prepare_source_promotion` binds an intent with it, so §7.7's
    equality between the projected bindings and the intent's own is an
    identity rather than two derivations that have to be kept in step. They
    were not: a second derivation reported every already-archived row of an
    `archive_retry` part as *excluded* as well, which names a row that already
    landed as one needing an owner's exclusion decision.
    """
    readings = decision.readings
    landed: tuple[str, ...] = ()
    held: tuple[str, ...] = ()
    if readings is not None:
        exact = decision.promoted_retry_flags or (False,) * len(readings.promoted)
        landed = tuple(
            record.id
            for record, is_retry in zip(readings.promoted, exact, strict=True)
            if not is_retry
        )
        held = tuple(record.id for record in readings.held)
    retries = tuple(decision.already_archived)
    accounted = set(landed) | set(held) | set(retries)
    excluded = tuple(
        record.id for record in decision.records if record.id not in accounted
    )
    return landed, held, excluded, retries


def fold_source_extraction_promotions(
    config: ProjectConfig,
    parts: Sequence[SourcePromotionPart],
    *,
    collection: Sequence[VocabularyRecord],
    collection_revision: RecordsRevision,
    pattern_store: Mapping[str, patterns.PatternSet],
    witness: enrich.DictionaryLookup,
) -> SourcePromotionFold:
    """Chain every part's projection through one canonical collection.

    Parts run in one fixed order — published part name, ascending, by plain
    codepoint comparison — so the same batch folds the same way twice. The
    carry advances **only** for a part whose projected state is `lands`:
    `PromotionDecision.merged` is populated after every non-landing return has
    already left, so an empty ``merged`` is never read as the projected
    collection and the empty-collection digest never enters the chain.

    A projected `blocked` state is not a disposition. It stays a blocker, and
    the canonical state is carried past it unchanged.
    """

    ordered = sorted(parts, key=lambda part: part.part_name)
    names = [part.part_name for part in ordered]
    if len(set(names)) != len(names):
        raise PromoteError(
            "[source-fold-invalid] each part in one promotion fold has its own name"
        )
    canonical_path = config.normalized_file.resolve()
    carry: tuple[VocabularyRecord, ...] = tuple(collection)
    text = collection_revision.text
    digests: list[str | None] = [
        None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()
    ]
    projections: list[SourcePromotionProjection] = []
    for part in ordered:
        decision = project_source_extraction_review_promotion(
            config,
            part.staging_path,
            expected_revision=part.expected_revision,
            review_record_ids=part.review_record_ids,
            review_patterns=part.review_patterns,
            pattern_store=pattern_store,
            coverage_approval=part.coverage_approval,
            collection=carry,
            collection_revision=RecordsRevision(canonical_path, text),
            witness=witness,
        )
        before = digests[-1]
        if decision.state == "lands":
            carry = tuple(decision.merged)
            text = records_json_text(carry)
            after: str | None = hashlib.sha256(text.encode("utf-8")).hexdigest()
        else:
            # Nothing canonical is written for any other state, so the
            # checkpoint after this part is the checkpoint before it.
            after = before
        digests.append(after)
        landed, held, excluded, retries = _disclosed_ids(decision)
        projections.append(
            SourcePromotionProjection(
                part_name=part.part_name,
                staging_path=decision.staging_path,
                staging_sha256=part.expected_revision,
                state=decision.state,
                gate=decision.gate,
                expected_before=before,
                expected_after=after,
                landed_ids=landed,
                held_ids=held,
                excluded_ids=excluded,
                archive_retry_ids=retries,
                decision=decision,
            )
        )
    return SourcePromotionFold(
        parts=tuple(projections),
        canonical_digests=tuple(digests),
        records_after=carry,
        canonical_text_after=text,
    )


def _require_exact_review_landing(
    meta: Mapping[str, Any],
    existing: Sequence[VocabularyRecord],
    incoming: Sequence[VocabularyRecord],
    merged: Sequence[VocabularyRecord],
) -> None:
    """Refuse an Assistant-reviewed landing that changes undisplayed fields."""

    reviews_all_examples = staging.AI_ENRICHMENT_REVIEW_KEY in meta
    if staging.CARD_REVISION_REVIEW_KEY in meta:
        marker = meta.get(staging.CARD_REVISION_REVIEW_KEY)
        provenance = meta.get(staging.CARD_REVISION_KEY)
    elif staging.AI_ENRICHMENT_REVIEW_KEY in meta:
        marker = meta.get(staging.AI_ENRICHMENT_REVIEW_KEY)
        provenance = meta.get(staging.AI_ENRICHMENT_KEY)
    else:
        return
    fields = provenance.get("fields") if isinstance(provenance, Mapping) else None
    accepted = (
        marker.get("accepted_record_ids") if isinstance(marker, Mapping) else None
    )
    if not isinstance(fields, Mapping) or not isinstance(accepted, list):
        raise PromoteError(
            "[assistant-review-hidden-landing] exact review provenance is incomplete"
        )
    accepted_ids = set(accepted)
    before = {record.id: record for record in existing}
    after = {record.id: record for record in merged}
    for proposed in incoming:
        if proposed.id not in accepted_ids:
            raise PromoteError(
                "[assistant-review-hidden-landing] promotion selected a card outside "
                "the exact owner review"
            )
        current = before.get(proposed.id)
        landing = after.get(proposed.id)
        names = fields.get(proposed.id)
        if current is None or landing is None or not isinstance(names, list):
            raise PromoteError(
                "[assistant-review-hidden-landing] reviewed card identity changed "
                f"for {proposed.id}"
            )
        old_wire = current.to_dict()
        new_wire = landing.to_dict()
        hidden = sorted(
            name
            for name in old_wire
            if name != "source"
            and old_wire[name] != new_wire[name]
            and name not in names
        )
        old_source = dict(old_wire["source"])
        new_source = dict(new_wire["source"])
        old_raw = dict(old_source.pop("raw_fields", {}))
        new_raw = dict(new_source.pop("raw_fields", {}))
        old_authority = old_raw.pop(EXAMPLE_AUTHORITY_KEY, None)
        new_authority = new_raw.pop(EXAMPLE_AUTHORITY_KEY, None)
        examples_reviewed = reviews_all_examples or "examples" in names
        if old_authority != new_authority and not examples_reviewed:
            hidden.append("source.example_authority")
        if examples_reviewed:
            exact = set_example_flags(
                proposed,
                EXAMPLE_AUTHORITY_KEY,
                (
                    example.japanese
                    for example in proposed.examples
                    if proposed.source.type == "extract" and example.japanese
                ),
            ).source.raw_fields.get(EXAMPLE_AUTHORITY_KEY)
            if proposed.source.raw_fields.get(EXAMPLE_AUTHORITY_KEY) != exact:
                hidden.append("source.example_authority")
        if old_source != new_source or old_raw != new_raw:
            hidden.append("source")
        if hidden:
            raise PromoteError(
                "[assistant-review-hidden-landing] exact owner review did not display "
                f"landing changes to {proposed.id}: {', '.join(hidden)}"
            )


def _unrewritable(path: Path) -> str:
    return (
        f"{path} is not a staging file janki can rewrite: the promoted archive "
        f"is written under the same name, and that needs "
        f"{' or '.join(STAGING_SUFFIXES)}. Rename it and re-run."
    )


def _require_no_pending_curation(config: ProjectConfig, staging_path: Path) -> None:
    """Refuse this file while a durable curation decision over it is open.

    A job document janki cannot read is itself a refusal, exactly as in
    :func:`decide_promotion`: a skipped barrier is a lifted one, and an
    unreadable document is where an open intent would be invisible. The writers
    raise rather than returning a blocked decision, so the unreadable case is
    reported with promotion's own tag instead of the curation service's type.
    """

    try:
        curation = _pending_curation(config, staging_path)
    except JankiError as exc:
        raise PromoteError(str(exc)) from exc
    if curation:
        raise PromoteError(curation)


def _pending_curation(config: ProjectConfig, staging_path: Path) -> str:
    """Why a recorded cross-source curation decision blocks this file, if one does.

    Imported at call time rather than at module scope: the curation service
    reads the study job store, which reaches this package's own aggregate
    module, and promotion is below it in the import graph. The gate itself is
    inside :func:`decide_promotion`, so it applies identically to `janki
    promote`, the workbench promotion, the Assistant's typed promotion action
    and a study job's own finish.

    A job document that cannot be read refuses here rather than being skipped,
    because a skipped barrier is a lifted one.
    """

    from japanese_anki.application import study_curation

    return study_curation.pending_curation_refusal(config, staging_path)


def decide_promotion(
    config: ProjectConfig,
    staging_path: Path,
    *,
    source: str = "",
    client: enrich.DictionaryLookup | None = None,
    skip_reading_check: bool | None = None,
    _record_snapshot: tuple[
        bytes,
        Sequence[VocabularyRecord],
        Mapping[str, Any],
    ]
    | None = None,
    _collection_snapshot: tuple[Sequence[VocabularyRecord], RecordsRevision]
    | None = None,
    _pattern_store_snapshot: Mapping[str, patterns.PatternSet] | None = None,
) -> PromotionDecision:
    """Work out everything `janki promote` decides, and write nothing.

    Every decision comes from a function promote itself calls, **in the order
    promote calls it**. The order matters even though nothing is written: it
    decides which refusal a person is shown first, and the structural gates go
    ahead of coverage deliberately — sending someone to a coverage decision
    (whose other route is a paid completeness check) for a file no acceptance
    could make promotable is wasted work followed by the refusal they should
    have seen.

    **Gate refusals are returned, not raised** — `decision.error` holds the
    exception and `decision.gate` names what refused. The command re-raises it
    so its stderr and exit code are unchanged; a page reports it. A caller
    contract violation still raises, because a bad call is not a fact about
    the staging file and rendering it as "this source cannot be added" would
    say the wrong thing about somebody's work.

    `--accept-coverage` is deliberately outside this: it spends a model call
    and writes an approval into the staging file, so the command runs it
    between two decide passes rather than having a read-only function do it.
    """
    requested_skip = skip_reading_check
    if skip_reading_check is None:
        # Handing over a client and still getting the offline answer is the
        # shape a caller would never intend, so it is not spellable by
        # accident: the check follows the client unless someone says
        # otherwise, which is what `--skip-reading-check` does.
        skip_reading_check = client is None
    if not skip_reading_check and client is None:
        # Refused here, and unconditionally. `check_readings` refuses this too,
        # but from inside its per-record loop after the structural holds — so a
        # caller could decide a blocked or all-held file, see a decision come
        # back, and crash on the first clean row in production instead.
        raise PromoteError(
            "A reading check needs a dictionary client. Pass one, or leave "
            "skip_reading_check alone to get the offline plan."
        )

    reading_check: Literal["preview", "explicit_skip", "consulted"] = (
        "explicit_skip" if requested_skip is True else "preview"
    )
    # Any injection makes this a projection of a repository state that is not
    # on disk. One flag for all three, so a caller cannot supply the quiet one
    # and reach the executor.
    projected = (
        _record_snapshot is not None
        or _collection_snapshot is not None
        or _pattern_store_snapshot is not None
    )
    repository = _repository_binding(config)
    staging_path = staging_path.resolve()
    name = source or staging_path.name
    consulted = False
    # What has been read so far. A refusal carries it too: `--accept-coverage`
    # has to show the model the coverage block of the very file the gate
    # refused, and a blocked decision that dropped its snapshot left the
    # acceptance step with nothing to read — reporting "carries no coverage
    # block" about a file whose block is right there.
    snapshot: dict[str, Any] = {}

    def at(state: str, **fields: Any) -> PromotionDecision:
        # Enforced, not documented: `at` took any string, so a mislabelled
        # state was invisible until Step 2 dispatched on it.
        assert state in DECISION_STATES, state
        return PromotionDecision(
            source=name,
            staging_path=staging_path,
            state=state,
            repository=repository,
            reading_check=reading_check,
            consulted=consulted,
            projected=projected,
            **fields,
        )

    def blocked(reason: JankiError, gate: str) -> PromotionDecision:
        return at("blocked", error=reason, gate=gate, **snapshot)

    archive_base = (config.staging_dir / "done" / staging_path.name).resolve()
    if inside_archive(staging_path, archive_base.parent):
        return blocked(
            PromoteError(
                f"{staging_path} is inside the promoted archive. Those records "
                "are already in the collection; promoting the archive would "
                "only duplicate it."
            ),
            "archive",
        )

    try:
        curation = _pending_curation(config, staging_path)
    except JankiError as exc:
        # A job document a barrier could be hiding in, refused rather than
        # skipped: a skipped barrier is a lifted one. Returned, not raised,
        # because this function's contract is that a gate refusal comes back
        # in `decision.error` for the command to re-raise and a page to render.
        return blocked(PromoteError(str(exc)), "curation-pending")
    if curation:
        return blocked(PromoteError(curation), "curation-pending")

    try:
        # Both reads under the writer lock, so the bytes the write path's
        # compare-and-swap trusts provably describe the records the gates ran
        # on. Two unlocked reads leave a window where a write lands between
        # them: the benign direction fails safe at write time, but an
        # A-then-back-to-A edit defeats the check entirely, and the secondary
        # revalidation compares only metadata and a row count.
        #
        # It also wraps the read. `read_bytes` raises `OSError`, which is not
        # a `JankiError` and sailed straight out of this function — so a
        # staging file deleted between listing a page and previewing it
        # produced a traceback where the old code returned a refusal.
        if _record_snapshot is None:
            wire, records, meta = record_review_snapshot(staging_path)
        else:
            wire, supplied_records, supplied_meta = _record_snapshot
            records = list(supplied_records)
            meta = dict(supplied_meta)
        snapshot.update(wire=wire, records=tuple(records), meta=meta)
        if staging.PROMOTION_BATCHES_KEY in meta:
            raise PromoteError(
                "[promotion-batches-live] promotion_batches is archive-only; a live "
                "review cannot claim that rows were promoted"
            )
        if "source_file" in meta and (
            not isinstance(meta["source_file"], str) or not meta["source_file"].strip()
        ):
            raise PromoteError(
                "[promotion-source-invalid] source_file must be nonblank text when present"
            )
        promote.check_coverage_facts(meta)
        done, archived, archived_meta, archive_revision = _archive_for_run_snapshot(
            archive_base, meta
        )
        snapshot.update(
            done=done,
            archived=tuple(archived),
            archived_meta=archived_meta,
            archive_revision=archive_revision,
        )
        if records or archived:
            validate_record_archive(
                meta,
                archived,
                archived_meta,
                archive_file=done.name,
            )
        retry_targets = promote.candidate_archive_retry_ids(
            meta, records, archived, archived_meta=archived_meta
        )
        retry_flags = tuple(target is not None for target in retry_targets)
    except JankiError as exc:
        return blocked(exc, "structure")
    coverage_error: JankiError | None = None
    try:
        promote.check_coverage(meta)
    except JankiError as exc:
        # Named separately because this is the one `--accept-coverage` can
        # answer. Keep it while the remaining local gates run: a coverage
        # verdict cannot repair any of their refusals, so returning here would
        # spend a paid check before discovering a free, fatal answer.
        coverage_error = exc

    common = {
        "wire": wire,
        "meta": meta,
        "records": tuple(records),
        "done": done,
        "archived": tuple(archived),
        "archived_meta": archived_meta,
        "archive_revision": archive_revision,
        "retry_flags": tuple(retry_flags),
        "already_archived": (
            tuple(record.id for record in archived)
            if not records
            else tuple(target for target in retry_targets if target is not None)
        ),
    }
    # A refusal after this point still has to carry the exact repository
    # snapshots already read. Paid coverage preflight compares them after a
    # blocking dictionary lookup so a call cannot be bought for collection or
    # archive state that disappeared while jpdb was answering.
    snapshot.update(common)

    if not records:
        if archived:
            # An empty live file beside this run's own archive is a completed
            # partial promotion, not a new review.
            if coverage_error is not None:
                return blocked(coverage_error, "coverage")
            return at("archive_retry", **common)
        run_id = rich_extraction_review_run_id(meta)
        if run_id is None:
            if coverage_error is not None:
                return blocked(coverage_error, "coverage")
            return at("nothing", **common)
        # A zero-record rich extraction proposed grammar and nothing else. Its
        # completion contract is a different one, and it refuses a source
        # nobody has reviewed.
        try:
            check_pattern_review(
                config, meta, run_id, pattern_store=_pattern_store_snapshot
            )
        except JankiError as exc:
            return blocked(exc, "patterns")
        try:
            check_rewritable(staging_path)
        except JankiError as exc:
            return blocked(exc, "rewritable")
        if staging_path.suffix.lower() not in STAGING_SUFFIXES:
            return blocked(PromoteError(_unrewritable(staging_path)), "rewritable")
        if coverage_error is not None:
            return blocked(coverage_error, "coverage")
        return at("pattern_only", **common)

    # After the zero-record branch, exactly as in the command: a legacy file
    # holding no records is completed without either of these being asked, so
    # checking them earlier refuses a file promote finishes cleanly.
    try:
        check_rewritable(staging_path)
    except JankiError as exc:
        return blocked(exc, "rewritable")
    if staging_path.suffix.lower() not in STAGING_SUFFIXES:
        return blocked(PromoteError(_unrewritable(staging_path)), "rewritable")

    work = [
        record
        for record, is_retry in zip(records, retry_flags, strict=True)
        if not is_retry
    ]
    if not work:
        if coverage_error is not None:
            return blocked(coverage_error, "coverage")
        return at("archive_retry", **common)

    try:
        # Refuses a `field_replacements` block whose provenance does not line
        # up with the rows it claims to have enriched. The merge's own binding
        # check never reads `ai_enrichment`, so nothing downstream catches it.
        ai_provenance = staged_ai_enrichment(
            meta,
            work,
            archived_ids=[record.id for record in archived],
            archived_records=archived,
        )
        output_path = config.normalized_file.resolve()
        # One bound read supplies both the parsed collection and its CAS token.
        # Two opens can observe revision A and records B, then silently accept
        # a write if the path returns to A before execution.
        #
        # A chained source-review fold supplies part *k*'s canonical state
        # instead: the collection the parts before it would have produced,
        # which is not on disk and will not be until they actually land.
        if _collection_snapshot is None:
            existing, output_revision = load_records_snapshot(output_path)
        else:
            supplied_existing, output_revision = _collection_snapshot
            existing = list(supplied_existing)
        existing_ids = tuple(record.id for record in existing)
        stored_ids, unreadable = status_module.surviving_ids_from(config, existing_ids)
        deck_revision = _deck_configuration_revision(config)
        ledger_revision = _file_revision(config.ledger_file.resolve())
    except JankiError as exc:
        return blocked(exc, "collection")

    consulted = not skip_reading_check
    if consulted:
        reading_check = "consulted"
    readings = promote.check_readings(
        work,
        client=client,
        skip_reading_check=skip_reading_check,
        already_stored=stored_ids,
        remint_blocked=bool(unreadable),
    )
    common |= {
        "work": tuple(work),
        "readings": readings,
        "existing": tuple(existing),
        "output_path": output_path,
        "output_revision": output_revision,
        "stored_ids": frozenset(stored_ids),
        "deck_revision": deck_revision,
        "ledger_revision": ledger_revision,
        "unreadable_decks": tuple(unreadable),
        "ai_provenance": ai_provenance,
    }
    snapshot.update(common)

    try:
        # The *second* accounting call, the one made after `check_readings`.
        # Not a repeat: its duplicate test is over the ids rows would land
        # *under*, so two rows whose corrected readings mint the same id
        # collide only here.
        #
        # Its answer is carried rather than discarded. The first call already
        # probes each row's resolved id **and** its stable re-mint, and
        # `remint` mints nothing else, so a row the second call would flag was
        # excluded from `work` by the first — the plan has a test pinning
        # exactly that. The write path still composes its prune flags from
        # these, because a staging prune is not a place to spend an invariant.
        promoted_retry = promote.check_candidate_accounting(
            meta, readings.promoted, archived, archived_meta=archived_meta
        )
    except JankiError as exc:
        return blocked(exc, "accounting")

    common["promoted_retry_flags"] = tuple(promoted_retry)
    snapshot["promoted_retry_flags"] = tuple(promoted_retry)

    if not readings.promoted:
        # Nothing would land, so promote returns before it reads the ledger.
        # Reading it here would block a source on a corrupt ledger promote
        # never opens — telling someone their work is unusable when it is not.
        if coverage_error is not None:
            return blocked(coverage_error, "coverage")
        return at("nothing_lands", **common)

    try:
        ledger.load_snapshot(config.ledger_file, ledger_revision)
    except JankiError as exc:
        return blocked(exc, "ledger")
    try:
        merged, outcomes = promote.merge_staged_records(
            list(existing),
            list(readings.promoted),
            dict(meta),
            # Every live non-retry row in the form it arrived in, held ones
            # included. A reading hold narrows what may land, not what review
            # an old-value binding covers, so a stale binding on a held row
            # refuses the whole merge.
            validate_incoming=work,
        )
        _require_exact_review_landing(
            meta,
            existing,
            readings.promoted,
            merged,
        )
    except JankiError as exc:
        return blocked(exc, "merge")
    snapshot.update(merged=tuple(merged), outcomes=outcomes)
    try:
        newly_landing = [
            record
            for record, is_retry in zip(readings.promoted, promoted_retry, strict=True)
            if not is_retry
        ]
        _require_deck_inputs(
            config,
            existing_ids,
            stored_ids=stored_ids,
            unreadable_decks=unreadable,
            deck_revision=deck_revision,
        )
        deck_ownership = require_exact_deck_ownership(config, newly_landing, merged)
        _require_deck_inputs(
            config,
            existing_ids,
            stored_ids=stored_ids,
            unreadable_decks=unreadable,
            deck_revision=deck_revision,
        )
        try:
            confirmed_ownership = require_exact_deck_ownership(
                config, newly_landing, merged
            )
        except JankiError as exc:
            raise PromoteError(
                "[promotion-input-stale] study-deck ownership changed while "
                "it was being proved. Reload the promotion preview."
            ) from exc
        if confirmed_ownership != deck_ownership:
            raise PromoteError(
                "[promotion-input-stale] study-deck ownership changed while "
                "it was being proved. Reload the promotion preview."
            )
        _require_deck_inputs(
            config,
            existing_ids,
            stored_ids=stored_ids,
            unreadable_decks=unreadable,
            deck_revision=deck_revision,
        )
    except JankiError as exc:
        return blocked(exc, "deck")
    snapshot["deck_ownership"] = deck_ownership

    if coverage_error is not None:
        return blocked(coverage_error, "coverage")
    return at(
        "lands",
        **common,
        merged=tuple(merged),
        outcomes=outcomes,
        deck_ownership=deck_ownership,
    )


def _save_execution_ledger(book: ledger.Ledger) -> ledger.LedgerError | None:
    """Save the promotion ledger without hiding already-landed records.

    Called with this transaction's ledger path lock already held, so it uses
    the `_under_lock` seam rather than acquiring a lock it owns.
    """
    try:
        book.save_under_lock()
    except ledger.LedgerError as exc:
        return exc
    return None


#: Every date the promotion ledger component carries, frozen at prepare.
#:
#: `ledger._iso_date(None)` reads the clock, so a ledger payload recomputed at
#: apply time is different bytes at midnight and a resume on the following day
#: would find its own component at a third state and refuse. §7.1 fixes them
#: once, in the writer that owns the write.
LEDGER_DATE_KEYS: tuple[str, ...] = ("added_at", "seen_at", "enriched_at")


@dataclass(frozen=True, slots=True)
class _LandingPlan:
    """Row dispositions the executor derives once and the preparation reuses."""

    result: promote.PromoteResult
    retry_records: tuple[VocabularyRecord, ...]
    pending_promoted: tuple[VocabularyRecord, ...]
    pending_records: tuple[VocabularyRecord, ...]
    keep: tuple[bool, ...]
    reminted: Mapping[str, str]
    existing: tuple[VocabularyRecord, ...]
    output_path: Path
    output_revision: RecordsRevision
    ai_provenance: Any


def _landing_plan(decision: PromotionDecision) -> _LandingPlan:
    """Which rows land, which are exact retries, and which stay behind."""
    if decision.reading_check == "preview":
        raise PromoteError(
            "[reading-check-required] an offline promotion preview cannot be "
            "executed. Consult jpdb or explicitly authorize skipping the "
            "reading check. Nothing was promoted."
        )
    result = decision.readings
    if result is None:
        raise PromoteError(
            f"Promotion decision {decision.state!r} has no reading disposition"
        )
    output_path = decision.output_path
    output_revision = decision.output_revision
    if output_path is None or output_revision is None:
        raise PromoteError(
            f"Promotion decision {decision.state!r} has no collection snapshot"
        )

    records = decision.records
    raw_archive_retry = decision.retry_flags
    work_records = list(decision.work)
    retry_records: list[VocabularyRecord] = [
        record
        for record, is_retry in zip(records, raw_archive_retry, strict=True)
        if is_retry
    ]
    exact_promoted = decision.promoted_retry_flags
    pending_promoted = [
        record
        for record, is_retry in zip(result.promoted, exact_promoted, strict=True)
        if not is_retry
    ]
    retry_records.extend(
        record
        for record, is_retry in zip(result.promoted, exact_promoted, strict=True)
        if is_retry
    )
    promoted_flags = iter(exact_promoted)
    retry_by_work = [False if stays else next(promoted_flags) for stays in result.keep]
    work_keep = [
        stays and not is_retry
        for stays, is_retry in zip(result.keep, retry_by_work, strict=True)
    ]
    work_keep_iter = iter(work_keep)
    keep = [
        False if is_retry else next(work_keep_iter) for is_retry in raw_archive_retry
    ]
    pending_records = [
        record
        for record, is_retry in zip(work_records, retry_by_work, strict=True)
        if not is_retry
    ]
    pending_reminted: dict[str, str] = {}
    promoted_iter = iter(result.promoted)
    retry_iter = iter(exact_promoted)
    for original, stays in zip(work_records, result.keep, strict=True):
        if stays:
            continue
        transformed = next(promoted_iter)
        is_retry = next(retry_iter)
        if not is_retry and transformed.id != original.id:
            pending_reminted[original.id] = transformed.id

    return _LandingPlan(
        result=result,
        retry_records=tuple(retry_records),
        pending_promoted=tuple(pending_promoted),
        pending_records=tuple(pending_records),
        keep=tuple(keep),
        reminted=pending_reminted,
        existing=tuple(decision.existing),
        output_path=output_path,
        output_revision=output_revision,
        ai_provenance=decision.ai_provenance,
    )


def _digest(value: bytes | str | None) -> str | None:
    if value is None:
        return None
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _component(
    role: str,
    path: Path,
    *,
    before: str | None,
    after_text: str | None,
    removes: bool = False,
) -> staging.PreparedComponent:
    """One prepared component, with absence preserved on either side."""
    if removes:
        after: str | None = None
    elif after_text is None:
        after = before
    else:
        after = _digest(after_text)
    return staging.PreparedComponent(
        role=role,
        path=str(path),
        expected_before=before,
        expected_after=after,
        after_text=after_text,
    )


@dataclass(frozen=True, slots=True)
class PreparedSourcePromotion:
    """One part's complete promotion intent, frozen before its first write.

    Everything a resume cannot recompute is here: the selected archive name,
    the content-addressed receipt id, the archive metadata, every ledger date,
    and each distinct path's whole after-payload with its before binding. A
    crash between two of those writes recovers from this value alone —
    a returned result that was never persisted is not evidence, and neither is
    a writer's state label.
    """

    part_name: str
    source: str
    staging_path: str
    projected_state: str
    #: The dictionary disposition this part was decided under — a
    #: `PromotionDecision.reading_check` value. A resume re-decides with the
    #: same one: replaying `explicit_skip` as a consulted run, or the reverse,
    #: asks a different question from the one whose answer was recorded. It
    #: grants nothing; the decision it reproduces was already authorized, and
    #: the fresh outcome sets are compared to the intent regardless.
    reading_check: str = "consulted"
    landed_ids: tuple[str, ...] = ()
    held_ids: tuple[str, ...] = ()
    excluded_ids: tuple[str, ...] = ()
    archive_retry_ids: tuple[str, ...] = ()
    archive_name: str | None = None
    receipt_id: str | None = None
    archive_meta: Mapping[str, Any] | None = None
    ledger_dates: Mapping[str, str] = field(default_factory=dict)
    #: The exact rows this part prunes from its live review because they are
    #: already in its own archive. Frozen as rows, not only as ids, because
    #: nothing else in this intent carries them: the archive component is bound
    #: unwritten and the live remainder is what stays *behind*. The ordinary
    #: writer reports them, and a resumed finish reports the same ones.
    retry_rows: tuple[Mapping[str, Any], ...] = ()
    #: `commit_canonical_state`'s deck-input binding, frozen from the decision
    #: that was authorized: the ids the collection contributed **before** this
    #: landing, the resulting surviving-id set, the unreadable decks and the
    #: deck-configuration digest. A resume re-asks `_require_deck_inputs` with
    #: these at the same canonical seam the ordinary writer asks it at; without
    #: them a recovery would publish over deck inputs an ordinary promote
    #: refuses. Empty `deck_revision` means the decided state never read the
    #: decks at all (`nothing`, `pattern_only`, `archive_retry`).
    existing_ids: tuple[str, ...] = ()
    stored_ids: tuple[str, ...] = ()
    unreadable_decks: tuple[str, ...] = ()
    deck_revision: str = ""
    components: tuple[staging.PreparedComponent, ...] = ()

    def __post_init__(self) -> None:
        if self.projected_state not in DECISION_STATES or self.projected_state == "blocked":
            raise PromoteError(
                f"[promotion-intent-invalid] {self.projected_state!r} is not a "
                "promotable state"
            )
        if self.reading_check not in READING_CHECKS:
            raise PromoteError(
                f"[promotion-intent-invalid] {self.reading_check!r} is not a "
                "reading disposition"
            )
        roles = [component.role for component in self.components]
        if len(set(roles)) != len(roles):
            raise PromoteError(
                "[promotion-intent-invalid] a promotion intent binds each path once"
            )
        allowed = {"canonical", "ledger", "archive", "live_staging"}
        unknown = sorted(set(roles) - allowed)
        if unknown:
            raise PromoteError(
                "[promotion-intent-invalid] a promotion intent binds "
                f"{', '.join(sorted(allowed))}, not {', '.join(unknown)}"
            )

    def component(self, role: str) -> staging.PreparedComponent | None:
        for item in self.components:
            if item.role == role:
                return item
        return None

    @property
    def fingerprint(self) -> str:
        """One digest over every byte of this intent."""
        wire = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(wire.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_name": self.part_name,
            "source": self.source,
            "staging_path": self.staging_path,
            "projected_state": self.projected_state,
            "reading_check": self.reading_check,
            "landed_ids": list(self.landed_ids),
            "held_ids": list(self.held_ids),
            "excluded_ids": list(self.excluded_ids),
            "archive_retry_ids": list(self.archive_retry_ids),
            "archive_name": self.archive_name,
            "receipt_id": self.receipt_id,
            "archive_meta": (
                None if self.archive_meta is None else dict(self.archive_meta)
            ),
            "ledger_dates": dict(self.ledger_dates),
            "retry_rows": [dict(row) for row in self.retry_rows],
            "existing_ids": list(self.existing_ids),
            "stored_ids": list(self.stored_ids),
            "unreadable_decks": list(self.unreadable_decks),
            "deck_revision": self.deck_revision,
            "components": [component.to_dict() for component in self.components],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PreparedSourcePromotion:
        try:
            archive_meta = raw["archive_meta"]
            return cls(
                part_name=str(raw["part_name"]),
                source=str(raw["source"]),
                staging_path=str(raw["staging_path"]),
                projected_state=str(raw["projected_state"]),
                reading_check=str(raw["reading_check"]),
                landed_ids=tuple(str(item) for item in raw["landed_ids"]),
                held_ids=tuple(str(item) for item in raw["held_ids"]),
                excluded_ids=tuple(str(item) for item in raw["excluded_ids"]),
                archive_retry_ids=tuple(
                    str(item) for item in raw["archive_retry_ids"]
                ),
                archive_name=(
                    None if raw["archive_name"] is None else str(raw["archive_name"])
                ),
                receipt_id=(
                    None if raw["receipt_id"] is None else str(raw["receipt_id"])
                ),
                archive_meta=None if archive_meta is None else dict(archive_meta),
                ledger_dates={
                    str(key): str(value)
                    for key, value in dict(raw["ledger_dates"]).items()
                },
                retry_rows=tuple(dict(row) for row in raw["retry_rows"]),
                existing_ids=tuple(str(item) for item in raw["existing_ids"]),
                stored_ids=tuple(str(item) for item in raw["stored_ids"]),
                unreadable_decks=tuple(str(item) for item in raw["unreadable_decks"]),
                deck_revision=str(raw["deck_revision"]),
                components=tuple(
                    staging.PreparedComponent.from_dict(item)
                    for item in raw["components"]
                ),
            )
        except (KeyError, TypeError, ValueError, StagingError) as exc:
            raise PromoteError(
                f"[promotion-intent-invalid] a recorded promotion intent is "
                f"unreadable: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class PromotionRecovery:
    """What finishing one interrupted promotion from its intent actually did."""

    part_name: str
    state: PromotionExecutionState
    already_complete: tuple[str, ...]
    finished: tuple[str, ...]
    receipt_id: str | None = None
    archive_path: Path | None = None
    landed_ids: tuple[str, ...] = ()


def _reviewed_pattern_set(
    config: ProjectConfig,
    meta: Mapping[str, Any],
    *,
    pattern_store: Mapping[str, patterns.PatternSet] | None,
) -> patterns.PatternSet:
    """The reviewed store entry the pattern-only archive embeds."""
    source = meta.get("source_file")
    stored = (
        patterns.load_store(config.patterns_file)
        if pattern_store is None
        else pattern_store
    ).get(source if isinstance(source, str) else "")
    if stored is None or not stored.reviewed:
        raise PromoteError(
            f"[patterns-unreviewed] {source} has not been reviewed. Run "
            f"'janki patterns --review {source}' first; nothing was archived."
        )
    return stored


def prepare_source_promotion(
    config: ProjectConfig,
    decision: PromotionDecision,
    *,
    part_name: str = "",
    pattern_store: Mapping[str, patterns.PatternSet] | None = None,
    now: date | None = None,
) -> PreparedSourcePromotion:
    """Freeze everything one promotion would write, and write nothing.

    Side-effect-free on canonical and on every published target. It reads the
    collection, the ledger and this run's archive, renders each after-payload
    with the same pure serializers the writer uses, and fixes the archive name,
    the content-addressed receipt id and every ledger date. `execute_promotion`
    is this followed immediately by :func:`apply_prepared_source_promotion`, so
    the ordinary CLI and workbench exercise the prepared path on every run
    rather than a second copy of it.
    """
    if decision.projected:
        raise PromoteError(
            "[promotion-projection-not-executable] this promotion decision was "
            "projected from an injected staging, collection or pattern-store "
            "snapshot. Re-decide it from the repository before executing. "
            "Nothing was promoted."
        )
    if decision.repository != _repository_binding(config):
        raise PromoteError(
            "[promotion-config-mismatch] this promotion decision belongs to a "
            "different repository configuration. Nothing was promoted."
        )
    if decision.is_blocked:
        if decision.error is None:
            raise PromoteError("A blocked promotion decision has no refusal")
        raise decision.error

    frozen = (now or date.today()).isoformat()
    dates = dict.fromkeys(LEDGER_DATE_KEYS, frozen)
    path = decision.staging_path
    archive_base = (config.staging_dir / "done" / path.name).resolve()
    meta = decision.meta
    archived = decision.archived
    archived_meta = decision.archived_meta
    canonical_path = config.normalized_file.resolve()
    ledger_path = config.ledger_file.resolve()
    live_before = _digest(decision.wire)
    name = part_name or decision.source

    def built(
        state: str,
        *,
        components: tuple[staging.PreparedComponent, ...],
        archive_name: str | None = None,
        receipt_id: str | None = None,
        archive_meta_after: Mapping[str, Any] | None = None,
        retry_rows: Sequence[VocabularyRecord] = (),
    ) -> PreparedSourcePromotion:
        landed, held, excluded, retries = _disclosed_ids(decision)
        return PreparedSourcePromotion(
            part_name=name,
            source=decision.source,
            staging_path=str(path),
            projected_state=state,
            reading_check=decision.reading_check,
            landed_ids=landed,
            held_ids=held,
            excluded_ids=excluded,
            archive_retry_ids=retries,
            archive_name=archive_name,
            receipt_id=receipt_id,
            archive_meta=archive_meta_after,
            ledger_dates=dates,
            retry_rows=tuple(record.to_dict() for record in retry_rows),
            existing_ids=tuple(record.id for record in decision.existing),
            stored_ids=tuple(sorted(decision.stored_ids)),
            unreadable_decks=tuple(decision.unreadable_decks),
            deck_revision=decision.deck_revision,
            components=components,
        )

    if decision.state == "nothing":
        # The executor's one state with no effect at all: it reads no
        # collection, opens no archive and does not touch the live file. An
        # intent that bound those anyway would refuse for a file this state
        # never looks at.
        return built("nothing", components=())

    if decision.state == "pattern_only":
        run_id = rich_extraction_review_run_id(meta)
        if run_id is None:
            raise PromoteError(
                "A pattern-only promotion decision has no rich extraction run"
            )
        reviewed = _reviewed_pattern_set(config, meta, pattern_store=pattern_store)
        selected, _rows, _meta = archive_for_run(archive_base, meta)
        archive_after_meta = _pattern_only_archive_meta(meta, reviewed)
        archive_text = render_staging_document(
            [], archive_after_meta, source=str(selected)
        )
        return built(
            "pattern_only",
            archive_name=selected.name,
            archive_meta_after=archive_after_meta,
            components=(
                _component(
                    "archive",
                    selected,
                    before=_digest(_file_revision(selected)),
                    after_text=archive_text,
                ),
                _component(
                    "live_staging", path, before=live_before, after_text=None, removes=True
                ),
            ),
        )

    done = decision.done if decision.done is not None else archive_base
    archive_before = _digest(decision.archive_revision)

    if decision.state == "archive_retry":
        empty_live = not decision.records
        recovered = _latest_retry_batch(
            archived_meta,
            archived,
            decision.already_archived,
            empty_live=empty_live,
            archive_file=done.name,
        )
        return built(
            "archive_retry",
            archive_name=done.name,
            receipt_id=recovered.receipt_id if recovered is not None else None,
            # Exactly what `_finish_record_review` is given below: nothing when
            # the live file is already empty, and every live row otherwise.
            retry_rows=() if empty_live else decision.records,
            components=(
                # Bound, unwritten: `_finish_record_review` revalidates this
                # archive and appends nothing, and an external edit to it still
                # refuses the apply.
                _component("archive", done, before=archive_before, after_text=None),
                _component(
                    "live_staging", path, before=live_before, after_text=None, removes=True
                ),
            ),
        )

    plan = _landing_plan(decision)
    live_after: str | None = None
    live_removes = True
    if plan.result.held:
        live_after, _removed = render_staging_finish(
            decision.wire.decode("utf-8", errors="strict"),
            plan.keep,
            plan.result.held,
            source=str(path),
        )
        live_removes = False
    canonical_before = _digest(plan.output_revision.text)
    ledger_before = _digest(decision.ledger_revision)

    if not plan.pending_promoted:
        recovered = _latest_retry_batch(
            archived_meta,
            archived,
            decision.already_archived,
            archive_file=done.name,
        )
        return built(
            "nothing_lands",
            archive_name=done.name,
            receipt_id=recovered.receipt_id if recovered is not None else None,
            retry_rows=plan.retry_records,
            components=(
                _component(
                    "canonical", canonical_path, before=canonical_before, after_text=None
                ),
                _component(
                    "ledger", ledger_path, before=ledger_before, after_text=None
                ),
                _component("archive", done, before=archive_before, after_text=None),
                _component(
                    "live_staging",
                    path,
                    before=live_before,
                    after_text=live_after,
                    removes=live_removes,
                ),
            ),
        )

    merged, outcomes = promote.merge_staged_records(
        list(plan.existing),
        list(plan.pending_promoted),
        dict(meta),
        validate_incoming=list(plan.pending_records),
    )
    book = ledger.load_snapshot(ledger_path, decision.ledger_revision)
    _record_promotion_ledger(book, plan, outcomes, meta=meta, dates=dates)
    prior_batches = promotion_batches(
        archived_meta or {}, archived=archived, archive_file=done.name
    )
    archive_start_index = (
        prior_batches[0].archive_start_index if prior_batches else len(archived)
    )
    completed_batch = _new_promotion_batch(
        meta,
        source=decision.source,
        promoted=plan.pending_promoted,
        ownership=decision.deck_ownership,
        archive_file=done.name,
        archive_start_index=archive_start_index,
    )
    combined = list(archived) + list(plan.pending_promoted)
    completed_meta = promote.archive_meta(dict(meta), len(combined))
    completed_meta[staging.PROMOTION_BATCHES_KEY] = [
        batch.to_dict() for batch in (*prior_batches, completed_batch)
    ]
    promotion_batches(
        completed_meta, archived=combined, archive_file=done.name
    )
    return built(
        "lands",
        archive_name=done.name,
        receipt_id=completed_batch.receipt_id,
        archive_meta_after=completed_meta,
        retry_rows=plan.retry_records,
        components=(
            _component(
                "canonical",
                canonical_path,
                before=canonical_before,
                after_text=records_json_text(merged),
            ),
            _component(
                "ledger",
                ledger_path,
                before=ledger_before,
                after_text=book.serialized_text(),
            ),
            _component(
                "archive",
                done,
                before=archive_before,
                after_text=render_staging_document(
                    combined, completed_meta, source=str(done)
                ),
            ),
            _component(
                "live_staging",
                path,
                before=live_before,
                after_text=live_after,
                removes=live_removes,
            ),
        ),
    )


def _record_promotion_ledger(
    book: ledger.Ledger,
    plan: _LandingPlan,
    outcomes: Mapping[str, MergeOutcome],
    *,
    meta: Mapping[str, Any],
    dates: Mapping[str, str],
) -> tuple[int, int]:
    """Apply this promotion's ledger writes with every date frozen."""
    already_landed_fields = (
        promote.already_landed_staged_fields(
            list(plan.existing), list(plan.pending_promoted), dict(meta)
        )
        if plan.ai_provenance is not None
        else {}
    )
    added = sum(
        book.record_added(record_id, at=dates["added_at"])
        for record_id, outcome in outcomes.items()
        if outcome.label == "added"
    )
    seen = sum(
        book.record_source_seen(
            record_id, source_type, source_ref, seen_at=dates["seen_at"]
        )
        for record_id, source_type, source_ref in promote.source_references(
            plan.pending_promoted
        )
    )
    if plan.ai_provenance is not None:
        ai_provider, ai_model, provenance = plan.ai_provenance
        for record_id, outcome in outcomes.items():
            request_fp, proposed_fields = provenance[record_id]
            written_fields = (
                proposed_fields
                if outcome.label == "added"
                else tuple(
                    name
                    for name in proposed_fields
                    if name in outcome.filled_fields
                    or name in already_landed_fields.get(record_id, ())
                )
            )
            if written_fields:
                book.record_enriched(
                    record_id,
                    kind="ai",
                    model=ai_model,
                    provider=ai_provider,
                    fields=written_fields,
                    request_fingerprint=request_fp,
                    at=dates["enriched_at"],
                )
    return added, seen


#: How a finished intent's projected state reads as an execution result.
_INTENT_RESULT_STATES: Mapping[str, PromotionExecutionState] = {
    "nothing": "nothing",
    "pattern_only": "pattern_only",
    "archive_retry": "archive_retry",
    "nothing_lands": "nothing_lands",
    "lands": "landed",
}


def _observed_component(component: staging.PreparedComponent) -> str | None:
    """The digest at one bound path right now, or ``None`` when it is absent."""
    return _digest(_file_revision(Path(component.path)))


def _component_verdict(
    component: staging.PreparedComponent, observed: str | None
) -> str:
    """``"pending"``, ``"complete"``, or a refusal naming both bound digests."""
    if observed == component.expected_before:
        return "pending" if component.writes else "complete"
    if observed == component.expected_after:
        return "complete"
    raise PromoteError(
        f"[promotion-intent-stale] the prepared {component.role} component "
        f"{component.path} is at neither bound state: expected "
        f"{component.expected_before} before or {component.expected_after} "
        f"after, found {observed}. Nothing was written."
    )


def _component_path(
    config: ProjectConfig, prepared: PreparedSourcePromotion, role: str
) -> Path:
    """The path this repository's configuration gives that role.

    A closed role list resolved from the live configuration, so an altered
    intent cannot re-point a promotion at a file its phase never planned to
    touch. Configured paths are compared as they are configured — the
    collection and the ledger may legitimately sit outside the repository
    root, and assuming otherwise would refuse a valid project.
    """
    if role == "canonical":
        return config.normalized_file.resolve()
    if role == "ledger":
        return config.ledger_file.resolve()
    if role == "live_staging":
        return Path(prepared.staging_path)
    if role == "archive":
        if prepared.archive_name is None:
            raise PromoteError(
                "[promotion-intent-invalid] an archive component needs its "
                "frozen archive name"
            )
        return (config.staging_dir / "done" / prepared.archive_name).resolve()
    raise PromoteError(f"[promotion-intent-invalid] unknown component role {role!r}")


def _precheck_components(
    config: ProjectConfig,
    prepared: PreparedSourcePromotion,
    *,
    skip: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Measure the **entire** bound vector before any of it is written.

    One component at neither digest refuses the whole part and names the
    component and both digests, so a stale later target cannot be discovered
    after an earlier one has already landed.
    """
    verdicts: dict[str, str] = {}
    for component in prepared.components:
        if component.role in skip:
            continue
        expected_path = _component_path(config, prepared, component.role)
        if Path(component.path) != expected_path:
            raise PromoteError(
                f"[promotion-intent-stale] the prepared {component.role} component "
                f"names {component.path}, but this configuration resolves that "
                f"role to {expected_path}. Nothing was written."
            )
        verdicts[component.role] = _component_verdict(
            component, _observed_component(component)
        )
    return verdicts


def _require_unstarted(prepared: PreparedSourcePromotion, verdicts: Mapping[str, str]) -> None:
    """Refuse to re-run a part whose own writes have already begun."""
    started = sorted(
        role
        for role, verdict in verdicts.items()
        if verdict == "complete"
        and (component := prepared.component(role)) is not None
        and component.writes
    )
    if started:
        raise PromoteError(
            "[promotion-intent-partial] this promotion already wrote "
            f"{', '.join(started)}. Recover it from its recorded intent rather "
            "than deciding it again; nothing was written."
        )


def _assert_intent_matches(
    prepared: PreparedSourcePromotion, fresh: PreparedSourcePromotion
) -> None:
    """Compare a freshly decided part with exactly what the intent bound.

    Outcome sets first — state, the four disclosed id sets, the archive
    selection and the content-addressed receipt — and then **both** sides of
    every component binding. Comparing only the before-state would accept a
    repository that starts where the intent expected and finishes somewhere
    else, which is precisely the case a frozen ledger date or a frozen archive
    payload exists to rule out.
    """
    for label, expected, actual in (
        ("state", prepared.projected_state, fresh.projected_state),
        ("landed ids", prepared.landed_ids, fresh.landed_ids),
        ("held ids", prepared.held_ids, fresh.held_ids),
        ("excluded ids", prepared.excluded_ids, fresh.excluded_ids),
        ("archive-retry ids", prepared.archive_retry_ids, fresh.archive_retry_ids),
        ("archive name", prepared.archive_name, fresh.archive_name),
        ("receipt id", prepared.receipt_id, fresh.receipt_id),
        # The deck inputs `commit_canonical_state` revalidates. No component
        # digest covers them — a deck file is not one of this intent's paths —
        # so an unstarted re-decide would otherwise accept a deck-input change
        # the ordinary writer refuses with `[promotion-input-stale]`.
        ("deck configuration", prepared.deck_revision, fresh.deck_revision),
        ("surviving ids", prepared.stored_ids, fresh.stored_ids),
        ("collection ids", prepared.existing_ids, fresh.existing_ids),
        ("unreadable decks", prepared.unreadable_decks, fresh.unreadable_decks),
    ):
        if expected != actual:
            raise PromoteError(
                f"[promotion-intent-stale] {prepared.part_name} now promotes a "
                f"different {label}: the intent bound {expected!r} and the "
                f"repository decides {actual!r}. Nothing was promoted."
            )
    if {item.role for item in prepared.components} != {
        item.role for item in fresh.components
    }:
        raise PromoteError(
            f"[promotion-intent-stale] {prepared.part_name} now writes a "
            "different set of files from the one its intent bound. Nothing was "
            "promoted."
        )
    for component in prepared.components:
        current = fresh.component(component.role)
        assert current is not None  # proved by the role comparison above
        for side, expected_digest, actual_digest in (
            ("starts from", component.expected_before, current.expected_before),
            ("finishes at", component.expected_after, current.expected_after),
        ):
            if expected_digest != actual_digest:
                raise PromoteError(
                    f"[promotion-intent-stale] {prepared.part_name}'s "
                    f"{component.role} no longer {side} the bound state "
                    f"{expected_digest}; the repository decides {actual_digest}. "
                    "Nothing was promoted."
                )


def _intent_day(prepared: PreparedSourcePromotion) -> date:
    """The one day this intent froze, replayed by every later pass.

    Not `date.today()`: a part prepared before midnight and resumed after it
    would otherwise re-derive a ledger payload whose digest is neither of the
    two states its own component binds, and refuse its own writes.
    """
    frozen = {str(value) for value in prepared.ledger_dates.values()}
    if len(frozen) != 1:
        raise PromoteError(
            "[promotion-intent-invalid] a promotion intent freezes one date for "
            f"its whole ledger write, not {sorted(frozen)}"
        )
    try:
        return date.fromisoformat(next(iter(frozen)))
    except ValueError as exc:
        raise PromoteError(
            f"[promotion-intent-invalid] {next(iter(frozen))!r} is not an ISO "
            "date this promotion can replay"
        ) from exc


def _requires_deck_authority(prepared: PreparedSourcePromotion) -> bool:
    """Whether this intent's replay reaches the deck-ownership proof.

    Only a landing does: `commit_canonical_state` is the one place the ordinary
    writer takes the deck lock, and it runs for `lands` alone.
    """
    canonical = prepared.component("canonical")
    return canonical is not None and canonical.writes


def _intent_lock_paths(
    config: ProjectConfig, prepared: PreparedSourcePromotion
) -> list[Path]:
    """Every lock this intent's replay needs, in the ordinary writer's order.

    Live staging, then the archive, then the deck directory, then canonical,
    then the ledger — §7.5's order, and the order an ordinary promote acquires
    them in: `_finish_record_review` takes the first two and `precheck` joins
    the rest. Sorting these by name instead produced the *reverse*
    order on an ordinary layout, so a recovery holding canonical and waiting
    for the live review could meet a promote holding the live review and
    waiting for canonical.
    """
    ordered: list[Path] = []
    roles = ["live_staging", "archive"]
    if _requires_deck_authority(prepared):
        roles.append("deck_dir")
    roles += ["canonical", "ledger"]
    bound = {component.role for component in prepared.components}
    for role in roles:
        if role != "deck_dir" and role not in bound:
            continue
        target = (
            Path(os.path.realpath(config.deck_dir))
            if role == "deck_dir"
            else Path(os.path.realpath(_component_path(config, prepared, role)))
        )
        if target not in ordered:
            ordered.append(target)
    return ordered


def _writer_lock_paths(
    config: ProjectConfig, prepared: PreparedSourcePromotion
) -> list[Path]:
    """The locks the ordinary writer's `precheck` still has to join.

    :func:`_intent_lock_paths` minus the two `_finish_record_review` and
    `_complete_pattern_only_review` acquire themselves, so the whole vector is
    measured — and then written — under the same one order either entry takes.
    """
    held = {
        Path(os.path.realpath(_component_path(config, prepared, role)))
        for role in ("live_staging", "archive")
        if prepared.component(role) is not None
    }
    return [
        target
        for target in _intent_lock_paths(config, prepared)
        if target not in held
    ]


@contextlib.contextmanager
def _intent_locks(config: ProjectConfig, prepared: PreparedSourcePromotion):
    """Hold every lock this intent needs, through its proof and its effects.

    One acquisition, held to the end: the deck-ownership proof and the
    canonical replay it authorizes have to happen under the same deck lock, or
    a cooperating deck writer can move the rules in between and the proof
    describes a repository that no longer exists.
    """
    with ExitStack() as locks:
        for target in _intent_lock_paths(config, prepared):
            locks.enter_context(exclusive_path_lock(target))
        yield


def _intent_started(
    prepared: PreparedSourcePromotion, verdicts: Mapping[str, str]
) -> bool:
    """Whether any write this intent owns has already landed.

    One question, not three: a vector that is partly applied and one that is
    entirely applied are finished by the same replay — the components already
    at their after-state are skipped either way — and only "has this part begun"
    decides whether it may be planned again at all.
    """
    return any(
        component.writes and verdicts.get(component.role) == "complete"
        for component in prepared.components
    )


def _bound_intent(config: ProjectConfig, prepared: PreparedSourcePromotion) -> None:
    """Re-prove the configuration, source and role associations this intent claims.

    A durable intent names paths, and a path is exactly the part of it a later
    configuration or an editor can move. Every component's role is resolved
    from the live configuration by :func:`_component_path`; this proves the
    surrounding associations that resolution assumes — that the live review is
    a member of *this* configuration's active staging directory, that the
    frozen archive name is a plain staging filename rather than a route out of
    the archive directory, and that a landing carries the deck-input binding
    every later proof and comparison reads. An intent missing that binding is
    unreadable rather than exempt, and it is unreadable here — before the
    repository is consulted at all — so dropping the field is not a way around
    the check on either entry.
    """
    path = Path(prepared.staging_path)
    active = Path(os.path.abspath(os.fspath(config.staging_dir)))
    if Path(os.path.abspath(os.fspath(path))).parent != active:
        raise PromoteError(
            f"[promotion-intent-stale] the prepared live review {path} is not a "
            f"direct member of this configuration's staging directory {active}. "
            "Nothing was written."
        )
    if path.suffix.lower() not in STAGING_SUFFIXES:
        raise PromoteError(f"[promotion-intent-invalid] {_unrewritable(path)}")
    if not prepared.part_name.strip() or not prepared.source.strip():
        raise PromoteError(
            "[promotion-intent-invalid] a promotion intent names its part and "
            "its source"
        )
    name = prepared.archive_name
    if name is not None and (
        name != Path(name).name
        or "/" in name
        or "\\" in name
        or Path(name).suffix.lower() not in STAGING_SUFFIXES
    ):
        raise PromoteError(
            f"[promotion-intent-invalid] {name!r} is not a plain archive "
            "filename this promotion may write"
        )
    if (
        _requires_deck_authority(prepared)
        and prepared.landed_ids
        and not prepared.deck_revision
    ):
        raise PromoteError(
            "[promotion-intent-invalid] a landing intent carries the deck-input "
            "binding its decision was taken against"
        )


def _intent_collection(prepared: PreparedSourcePromotion) -> list[VocabularyRecord]:
    """The collection this intent froze, parsed from its own payload.

    Reads the intent, never the file: `records_json_text` wrote these bytes and
    this reverses exactly that, so a resume compares the plan against the
    repository rather than the repository against itself.
    """
    canonical = prepared.component("canonical")
    if canonical is None or canonical.after_text is None:
        return []
    try:
        payload = json.loads(canonical.after_text)
        if not isinstance(payload, list):
            raise ValueError("a collection payload is a list of records")
        return [VocabularyRecord.from_dict(item) for item in payload]
    except (ValueError, TypeError, JankiError) as exc:
        raise PromoteError(
            "[promotion-intent-invalid] the prepared canonical payload is not a "
            f"readable collection: {exc}"
        ) from exc


def _intent_landed_records(
    prepared: PreparedSourcePromotion,
) -> list[VocabularyRecord]:
    """The rows this intent says landed, read out of its own frozen payload."""
    merged = _intent_collection(prepared)
    if not merged:
        return []
    wanted = set(prepared.landed_ids)
    landed = [record for record in merged if record.id in wanted]
    if len(landed) != len(wanted):
        raise PromoteError(
            "[promotion-intent-invalid] the prepared canonical payload does not "
            "hold every row this intent says landed"
        )
    return landed


def _reprove_landing_authority_under_locks(
    config: ProjectConfig, prepared: PreparedSourcePromotion
) -> None:
    """Ask `commit_canonical_state`'s own questions before a pending landing.

    The canonical write is what puts a row in front of a study deck, so a
    resume that still has to perform it re-proves exactly one configured owner
    rather than replaying frozen bytes over a repository whose deck rules have
    since changed, and re-asks `_require_deck_inputs` with the bindings the
    authorized decision was taken against. The receipt is then re-derived from
    the freshly proved owners through the writer's own minter and **compared**
    — a different one describes a different transaction, and adopting it would
    let an intent stand in for authority it never carried.

    The caller holds the deck lock, and holds it through the canonical replay
    these proofs authorize: `io.exclusive_path_lock` is not re-entrant, and a
    proof taken under a lock that is released before the write it authorizes
    describes a repository a cooperating deck writer may already have changed.
    """
    canonical = prepared.component("canonical")
    if canonical is None or not canonical.writes:
        return
    landed = _intent_landed_records(prepared)
    if not landed:
        return
    merged = _intent_collection(prepared)
    ownership = require_exact_deck_ownership(config, landed, merged)
    # The binding is present: `_bound_intent` refuses a landing without one
    # before either entry reads the repository.
    _require_deck_inputs(
        config,
        prepared.existing_ids,
        stored_ids=set(prepared.stored_ids),
        unreadable_decks=prepared.unreadable_decks,
        deck_revision=prepared.deck_revision,
    )
    if prepared.receipt_id is None or prepared.archive_name is None:
        return
    batches = (prepared.archive_meta or {}).get(staging.PROMOTION_BATCHES_KEY)
    if not isinstance(batches, list) or not batches:
        raise PromoteError(
            "[promotion-intent-invalid] a landing intent carries its archive "
            "receipts"
        )
    frozen = batches[-1]
    if not isinstance(frozen, Mapping):
        raise PromoteError(
            "[promotion-intent-invalid] a promotion receipt must be a mapping"
        )
    reproved = _new_promotion_batch(
        prepared.archive_meta or {},
        source=prepared.source,
        promoted=landed,
        ownership=ownership,
        archive_file=prepared.archive_name,
        archive_start_index=int(frozen.get("archive_start_index", -1)),
    )
    if reproved.receipt_id != prepared.receipt_id:
        raise PromoteError(
            f"[promotion-intent-stale] the re-proved promotion receipt "
            f"{reproved.receipt_id} differs from the prepared "
            f"{prepared.receipt_id}. Nothing was written."
        )


def _finish_intent_under_locks(
    config: ProjectConfig,
    prepared: PreparedSourcePromotion,
    verdicts: Mapping[str, str],
) -> tuple[str, ...]:
    """Replay the components this intent still owes, in the writer's order.

    Canonical, then ledger, then archive, then the live review — the order
    `execute_promotion` writes in, so a second interruption leaves the same
    recoverable shapes this one is finishing. Components already at their
    after-state are left alone; a component at neither digest has already
    refused the whole vector in :func:`_precheck_components`.

    The caller holds every lock :func:`_intent_lock_paths` names, including the
    deck directory, from before the proof below until after the last write.
    """
    if verdicts.get("canonical") == "pending":
        _reprove_landing_authority_under_locks(config, prepared)
    finished: list[str] = []
    for role in ("canonical", "ledger", "archive", "live_staging"):
        component = prepared.component(role)
        if component is None or verdicts.get(role) != "pending":
            continue
        _write_component(component)
        finished.append(role)
    return tuple(finished)


def _intent_retry_records(
    prepared: PreparedSourcePromotion,
) -> tuple[VocabularyRecord, ...]:
    """The exact rows this intent prunes, read out of its own frozen payload."""
    try:
        return tuple(VocabularyRecord.from_dict(dict(row)) for row in prepared.retry_rows)
    except (ValueError, TypeError, JankiError) as exc:
        raise PromoteError(
            "[promotion-intent-invalid] the prepared archive-retry rows are not "
            f"readable records: {exc}"
        ) from exc


def _intent_held_records(
    prepared: PreparedSourcePromotion,
) -> tuple[VocabularyRecord, ...]:
    """The rows this intent leaves behind, read out of its own live remainder.

    The held remainder *is* the frozen live-staging payload, so nothing extra
    is bound to report it: the rows the writer would have returned are the rows
    that document holds. A removal leaves none, and a live component that
    writes nothing belongs to a part no resume can reach — every other
    component of such a part writes nothing either, so it is re-decided rather
    than finished from its intent.
    """
    live = prepared.component("live_staging")
    if live is None or live.after_text is None:
        return ()
    try:
        held, _meta = staging.read_staging_text(live.after_text, source=live.path)
    except JankiError as exc:
        raise PromoteError(
            "[promotion-intent-invalid] the prepared live remainder is not a "
            f"readable staging document: {exc}"
        ) from exc
    return tuple(held)


def _intent_execution_result(
    config: ProjectConfig, prepared: PreparedSourcePromotion
) -> PromotionExecutionResult:
    """What a completed intent reports, derived from the intent itself.

    Used where a resumed apply meets its own finished writes: the transaction
    the caller asked for is done, and saying so from the frozen plan is the
    only honest answer — the ordinary return value was never persisted, and a
    live file's presence proves nothing in either direction. Every field here
    is the value the ordinary writer would have returned for the same part:
    the pruned retry rows and the held remainder come out of the intent's own
    payloads, and ``removed`` is the writer's ``len(keep) - sum(keep)``, which
    its own invariant pins to the rows that leave the live file.

    Merge outcomes and the ledger counters are deliberately absent. They
    describe what the *writing* pass did to a ledger this pass may not have
    written, and inventing them would report bookkeeping nobody performed.
    """
    path = Path(prepared.staging_path)
    archive = (
        None
        if prepared.archive_name is None
        else (config.staging_dir / "done" / prepared.archive_name).resolve()
    )
    state = _INTENT_RESULT_STATES[prepared.projected_state]
    retries = _intent_retry_records(prepared)
    if state == "nothing":
        return PromotionExecutionResult(state="nothing", staging_path=path)
    if state == "pattern_only":
        return PromotionExecutionResult(
            state="pattern_only", staging_path=path, archive_path=archive
        )
    if state == "archive_retry":
        return PromotionExecutionResult(
            state="archive_retry",
            staging_path=path,
            archive_path=archive,
            retry_records=retries,
            removed=len(retries),
            # An empty live review beside a full archive is the writer's
            # `empty_live` shape, and it prunes nothing. Its already-archived
            # ids are every row in the archive, so counting *those* reported a
            # removal that never happened and denied the empty-live case.
            empty_live_retry=not retries,
            receipt_id=prepared.receipt_id,
        )
    canonical = prepared.component("canonical")
    removed = len(prepared.landed_ids) + len(retries)
    if state == "landed":
        return PromotionExecutionResult(
            state="landed",
            staging_path=path,
            archive_path=archive,
            output_path=None if canonical is None else Path(canonical.path),
            promoted=tuple(_intent_landed_records(prepared)),
            held=_intent_held_records(prepared),
            retry_records=retries,
            removed=removed,
            receipt_id=prepared.receipt_id,
        )
    return PromotionExecutionResult(
        state="nothing_lands",
        staging_path=path,
        # The archive exists for this state only when there were retries to
        # prune into it, which is exactly the result invariant's own rule.
        archive_path=archive if retries else None,
        output_path=None if canonical is None else Path(canonical.path),
        held=_intent_held_records(prepared),
        retry_records=retries,
        removed=removed,
        receipt_id=prepared.receipt_id,
    )


def _redecide_for_intent(
    config: ProjectConfig,
    prepared: PreparedSourcePromotion,
    *,
    witness: enrich.DictionaryLookup | None,
) -> PromotionDecision:
    """Decide an unstarted part again and compare it with what was bound.

    The dictionary disposition is the intent's own: replaying an
    ``explicit_skip`` decision as a consulted one, or the reverse, asks a
    different question from the one whose answer the owner approved. The frozen
    day is replayed too, so the comparison is between two plans that differ
    only where the repository really differs.
    """
    path = Path(prepared.staging_path)
    if prepared.reading_check == "explicit_skip":
        client: enrich.DictionaryLookup | None = None
        skip: bool | None = True
    elif prepared.reading_check == "consulted":
        if witness is None:
            raise PromoteError(
                "[promotion-intent-witness-required] this promotion was decided "
                "against a dictionary, so resuming it needs the recorded witness. "
                "Nothing was promoted."
            )
        client, skip = witness, False
    else:
        client, skip = None, None
    fresh = decide_promotion(
        config, path, source=prepared.source, client=client, skip_reading_check=skip
    )
    if fresh.is_blocked:
        if fresh.error is None:
            raise PromoteError("A blocked promotion decision has no refusal")
        raise fresh.error
    _assert_intent_matches(
        prepared,
        prepare_source_promotion(
            config,
            fresh,
            part_name=prepared.part_name,
            now=_intent_day(prepared),
        ),
    )
    return fresh


def execute_promotion(
    config: ProjectConfig, decision: PromotionDecision
) -> PromotionExecutionResult:
    """Consume a validated decision through the one promotion transaction.

    All mutation formerly in ``cli.command_promote`` lives here: canonical and
    ledger writes, exact archive retries, the live/archive CAS, held-row
    rewriting, and pattern-only completion. A caller may format the returned
    facts differently; it may not reproduce this writer.

    It is now exactly a preparation followed by its apply, so every ordinary
    CLI and workbench promote exercises the same frozen intent a study finish
    persists — there is one writer, not two.
    """
    prepared = prepare_source_promotion(config, decision, part_name=decision.source)
    return apply_prepared_source_promotion(config, prepared, decision=decision)


def apply_prepared_source_promotion(
    config: ProjectConfig,
    prepared: PreparedSourcePromotion,
    *,
    decision: PromotionDecision | None = None,
    witness: enrich.DictionaryLookup | None = None,
) -> PromotionExecutionResult:
    """Perform one prepared promotion through the writer that owns these files.

    ``decision`` is the decision the intent was prepared from, when the caller
    still holds it — the ordinary one-command path. Without it this is a
    resumed finish, and the order matters more than anything else here:

    1. the pending-curation barrier is rechecked under the caller's guard;
    2. the **entire** bound vector is classified while every one of its paths
       is locked, before anything is decided again;
    3. a part whose own writes have begun is finished from its intent — never
       re-planned, because a fresh decision reads those writes as somebody
       else's and either refuses or describes a different transaction;
    4. only a genuinely unstarted part is decided again, with the recorded
       dictionary disposition and the replay witness, and its fresh state,
       disclosed id sets, archive selection, receipt and *both* sides of every
       component binding must equal what the intent bound.

    An intent is a plan, never authority to write something else.

    The caller holds §6's shared curation guard; this rechecks that barrier
    under it and keeps it through every write. `io.exclusive_path_lock` is not
    re-entrant, so nothing here acquires the guard a second time — which is
    also why step 2's locks are released before the writer below takes its own.

    Every path this transaction touches is locked in §7.5's order — live
    staging, archive, deck directory, canonical, ledger — and held from before
    the vector is measured until after the last write. On the writer's path the
    first two are taken by `_finish_record_review` and the rest by `precheck`
    below, which is the one moment inside the writer when nothing else can move
    any of them; on the resumed path :func:`_intent_locks` takes the same set
    in the same order.
    """
    # Rechecked here, not only at planning. A curation intent can be published
    # while every bound staging file still holds its `sha256_before` bytes, so
    # the wire compare-and-swap below would pass for a decision taken *before*
    # that publication and consume a file a durable decision is about to
    # rewrite. Planning alone cannot close that race; this recheck runs under
    # the caller's coordination guard, which the intent's own publication also
    # takes, so the two cannot interleave.
    path = Path(prepared.staging_path)
    _require_no_pending_curation(config, path)

    resumed = decision is None
    if resumed:
        _bound_intent(config, prepared)
        with _intent_locks(config, prepared):
            verdicts = _precheck_components(config, prepared)
            if _intent_started(prepared, verdicts):
                # This part's own writes already began. Finish them from the
                # frozen payloads — same archive, same receipt, same dates —
                # rather than letting a fresh decision misclassify them.
                _finish_intent_under_locks(config, prepared, verdicts)
                return _intent_execution_result(config, prepared)
        decision = _redecide_for_intent(config, prepared, witness=witness)

    archive_base = (config.staging_dir / "done" / path.name).resolve()
    meta = decision.meta
    archived = decision.archived
    archived_meta = decision.archived_meta
    expected_wire = decision.wire
    dates = prepared.ledger_dates

    # Taken here rather than around the whole apply: the two the writer
    # itself owns — the live review and the selected archive — are acquired
    # inside it, and `precheck` is the one point after those and before any
    # effect where the rest of §7.5's order can still be joined.
    locks_taken = False

    with ExitStack() as writer_locks:
        def precheck(confirmed: Path) -> None:
            """Take the rest of §7.5's locks, then re-prove the whole vector.

            The classification above ran before any lock was taken and was
            released so this writer could take its own; this is the measurement
            that decides, at the one moment nothing else can move any of these
            files. The deck directory, canonical and the ledger are joined here
            — after the live review and the archive, which is the writer's own
            order — and stay held until this apply returns, so the deck proof
            below governs the canonical write it authorizes rather than a
            repository some other writer has since changed.

            Called once per transaction: the archive-selection loop repeats
            *before* this point, and the flag makes a second acquisition of a
            non-re-entrant lock impossible rather than merely unlikely.
            """
            nonlocal locks_taken
            if not locks_taken:
                for target in _writer_lock_paths(config, prepared):
                    writer_locks.enter_context(exclusive_path_lock(target))
                locks_taken = True
            if prepared.archive_name is not None and confirmed.name != prepared.archive_name:
                if resumed:
                    # A persisted intent's frozen archive is the only thing that
                    # proves which file its receipt describes, so a moved
                    # selection refuses rather than minting a second archive.
                    raise PromoteError(
                        f"[promotion-intent-stale] this promotion froze archive "
                        f"{prepared.archive_name!r} and the live selection is now "
                        f"{confirmed.name!r}. Nothing was promoted."
                    )
                # An in-process promote persisted nothing a moved name could
                # invalidate: another run occupied the base between this decision
                # and this transaction, and the writer's own
                # `expected_archive_revision` check is the authority for the file
                # it re-resolved to. The frozen archive payload describes a file
                # this promote is no longer writing, so it is not compared.
                _precheck_components(config, prepared, skip=frozenset({"archive"}))
                return
            verdicts = _precheck_components(config, prepared)
            if resumed:
                # A part that became partly applied while this pass was deciding
                # is somebody else's completion, not this one's to repeat.
                _require_unstarted(prepared, verdicts)

        if prepared.projected_state == "nothing":
            return PromotionExecutionResult(state="nothing", staging_path=path)

        if prepared.projected_state == "pattern_only":
            run_id = rich_extraction_review_run_id(meta)
            if run_id is None:
                raise PromoteError(
                    "A pattern-only promotion decision has no rich extraction run"
                )
            done, retried = _complete_pattern_only_review(
                config, path, archive_base, meta, expected_wire, run_id, precheck=precheck
            )
            return PromotionExecutionResult(
                state="pattern_only",
                staging_path=path,
                archive_path=done,
                archive_was_retry=retried,
            )

        if prepared.projected_state == "archive_retry":
            # The archive is written before the live review is pruned and deleted.
            # A crash after the prune leaves an empty extraction file; it is
            # completion evidence, not a new pattern-only review.
            empty_live = not decision.records
            done, removed, _batch = _finish_record_review(
                path,
                archive_base,
                expected_wire=expected_wire,
                expected_meta=meta,
                expected_archived=archived,
                expected_archived_meta=archived_meta,
                expected_archive_revision=decision.archive_revision,
                promoted=(),
                retry_records=() if empty_live else list(decision.records),
                keep=() if empty_live else [False] * len(decision.records),
                held=(),
                precheck=precheck,
            )
            return PromotionExecutionResult(
                state="archive_retry",
                staging_path=path,
                archive_path=done,
                retry_records=() if empty_live else decision.records,
                removed=removed,
                empty_live_retry=empty_live,
                receipt_id=prepared.receipt_id,
            )

        plan = _landing_plan(decision)
        if not plan.pending_promoted:
            # Held reasons still land in the live review; exact retries are pruned.
            done, removed, _batch = _finish_record_review(
                path,
                archive_base,
                expected_wire=expected_wire,
                expected_meta=meta,
                expected_archived=archived,
                expected_archived_meta=archived_meta,
                expected_archive_revision=decision.archive_revision,
                promoted=(),
                retry_records=list(plan.retry_records),
                keep=list(plan.keep),
                held=plan.result.held,
                precheck=precheck,
            )
            return PromotionExecutionResult(
                state="nothing_lands",
                staging_path=path,
                archive_path=done if plan.retry_records else None,
                output_path=plan.output_path,
                held=tuple(plan.result.held),
                retry_records=plan.retry_records,
                removed=removed,
                receipt_id=prepared.receipt_id,
            )

        # Deciding proved the ledger parses; load it here for the object this
        # transaction will mutate and attempt to save.
        book = ledger.load(config.ledger_file)
        merged, outcomes = promote.merge_staged_records(
            list(plan.existing),
            list(plan.pending_promoted),
            dict(meta),
            validate_incoming=list(plan.pending_records),
        )
        added, seen = _record_promotion_ledger(
            book, plan, outcomes, meta=meta, dates=dates
        )
        ledger_error: ledger.LedgerError | None = None

        def commit_canonical_state(
            archive_path: Path, archive_start_index: int
        ) -> PromotionBatch:
            nonlocal ledger_error
            # A checked decision may live in a browser capability while deck YAML
            # changes. Re-prove the exact selector verdict at the canonical commit
            # seam, under the deck lock `precheck` took before it measured
            # anything and holds through these writes, so a stale plan cannot
            # land zero-owner or overlapping rows and no deck writer can move the
            # rules between this proof and the canonical write it authorizes.
            fresh_ownership = require_exact_deck_ownership(
                config,
                list(plan.pending_promoted),
                merged,
            )
            if fresh_ownership != decision.deck_ownership:
                raise PromoteError(
                    "[deck-ownership-stale] study-deck rules changed after "
                    "these cards were checked. Nothing was promoted. Check "
                    "the current deck membership and try again."
                )
            _require_deck_inputs(
                config,
                [record.id for record in plan.existing],
                stored_ids=set(decision.stored_ids),
                unreadable_decks=decision.unreadable_decks,
                deck_revision=decision.deck_revision,
            )
            completed_batch = _new_promotion_batch(
                meta,
                source=decision.source,
                promoted=plan.pending_promoted,
                ownership=fresh_ownership,
                archive_file=archive_path.name,
                archive_start_index=archive_start_index,
            )
            # Re-proved, then compared — never adopted. The intent froze a
            # content-addressed receipt over this exact source, run, ids and
            # owners; a different one means the repository is describing a
            # different transaction from the one that was planned. The receipt
            # binds its archive file, so it is compared only for the archive
            # the intent actually froze — a same-run selection the writer
            # re-resolved under its lock is governed by `precheck` above.
            if (
                prepared.receipt_id is not None
                and archive_path.name == prepared.archive_name
                and completed_batch.receipt_id != prepared.receipt_id
            ):
                raise PromoteError(
                    "[promotion-intent-stale] the re-proved promotion receipt "
                    f"{completed_batch.receipt_id} differs from the prepared "
                    f"{prepared.receipt_id}. Nothing was promoted."
                )
            # Both saves are the `_locked` seams: `precheck` already owns the
            # canonical and ledger path locks, and these writers' own
            # acquisitions are not re-entrant.
            save_records_json_locked(
                plan.output_path, merged, expected=plan.output_revision
            )
            ledger_error = _save_execution_ledger(book)
            if ledger_error is not None and plan.ai_provenance is not None:
                # The reviewed proposal remains the only recoverable attribution.
                # Raising before archive/prune keeps it live while the outer locks
                # still guarantee no competing completion changed either copy.
                raise _AiLedgerHandoffIncomplete
            return completed_batch

        # Canonical writes, archive append, and live pruning/deletion share the
        # same live/done transaction. A same-run zero-row completion that wins the
        # lock is refused before `commit_canonical_state`; one that loses cannot
        # appear between validation and the canonical writes.
        try:
            done, removed, completed_batch = _finish_record_review(
                path,
                archive_base,
                expected_wire=expected_wire,
                expected_meta=meta,
                expected_archived=archived,
                expected_archived_meta=archived_meta,
                expected_archive_revision=decision.archive_revision,
                promoted=list(plan.pending_promoted),
                retry_records=list(plan.retry_records),
                keep=list(plan.keep),
                held=plan.result.held,
                canonical_commit=commit_canonical_state,
                precheck=precheck,
            )
        except _AiLedgerHandoffIncomplete:
            return PromotionExecutionResult(
                state="landed_ai_ledger_incomplete",
                staging_path=path,
                output_path=plan.output_path,
                promoted=plan.pending_promoted,
                held=tuple(plan.result.held),
                retry_records=plan.retry_records,
                reminted=dict(plan.reminted),
                outcomes=dict(outcomes),
                ledger_added=added,
                ledger_sources=seen,
                ledger_error=ledger_error,
            )

        return PromotionExecutionResult(
            state=("landed" if ledger_error is None else "landed_ledger_incomplete"),
            staging_path=path,
            archive_path=done,
            output_path=plan.output_path,
            promoted=plan.pending_promoted,
            held=tuple(plan.result.held),
            retry_records=plan.retry_records,
            removed=removed,
            reminted=dict(plan.reminted),
            outcomes=dict(outcomes),
            ledger_added=added,
            ledger_sources=seen,
            ledger_error=ledger_error,
            receipt_id=(
                completed_batch.receipt_id if completed_batch is not None else None
            ),
        )


def _write_component(component: staging.PreparedComponent) -> None:
    """Replay one frozen component payload through the bound low-level write."""
    target = Path(component.path)
    if component.removes:
        try:
            atomic_unlink_bound(target, expected_revision=component.expected_before)
        except JankiError as exc:
            raise PromoteError(
                f"[promotion-recovery-incomplete] could not retire {target} at its "
                f"bound revision: {exc}"
            ) from exc
        return
    try:
        atomic_write_text_bound(
            target,
            component.after_text or "",
            expected_revision=component.expected_before,
            expected_absent=component.expected_before is None,
        )
    except JankiError as exc:
        raise PromoteError(
            f"[promotion-recovery-incomplete] could not finish the {component.role} "
            f"write to {target}: {exc}"
        ) from exc


def _recovered(
    config: ProjectConfig,
    prepared: PreparedSourcePromotion,
    *,
    verdicts: Mapping[str, str],
    finished: tuple[str, ...],
) -> PromotionRecovery:
    """One recovery's report: what was already done, and what this pass did."""
    archive_name = prepared.archive_name
    return PromotionRecovery(
        part_name=prepared.part_name,
        state=_INTENT_RESULT_STATES[prepared.projected_state],
        already_complete=tuple(
            sorted(role for role, verdict in verdicts.items() if verdict == "complete")
        ),
        finished=finished,
        receipt_id=prepared.receipt_id,
        archive_path=(
            None
            if archive_name is None
            else (config.staging_dir / "done" / archive_name).resolve()
        ),
        landed_ids=prepared.landed_ids,
    )


def recover_promotion_intent_under_guard(
    config: ProjectConfig,
    prepared: PreparedSourcePromotion,
    *,
    witness: enrich.DictionaryLookup | None = None,
) -> PromotionRecovery:
    """Finish one interrupted promotion, the shared curation guard already held.

    From the intent, for everything it already wrote. A value that was returned
    but never persisted is not evidence, and neither is a writer's state label —
    so a **partially applied** part is never re-planned, which would misclassify
    its own writes as a new state. What this proves, every time, before any byte
    moves:

    * the pending-curation barrier, rechecked under the caller's guard, because
      a barrier published since the preparation is one a skip would lift;
    * the configuration, source and role associations the intent claims, so an
      altered target cannot borrow a valid vector;
    * the whole component vector, measured while every one of its paths is
      locked — a component at neither digest refuses the part and names both;
    * exactly one configured study-deck owner for every row a *pending*
      canonical write would land, the deck-input binding its decision was taken
      against, and the receipt re-derived from the freshly proved owners and
      compared. An intent is a plan, not the authority to publish a card, and
      recovery is not a way around a check the ordinary promote applies.

    Only then are the missing writes replayed from their frozen payloads, with
    the frozen archive name, receipt id and ledger dates — the same bytes on the
    day after the crash as on the day of it.

    An **unstarted** part is a different thing from an interrupted one, and
    §7.7 treats it differently: nothing this part owns has been written, so
    there is no half-finished transaction whose classification a fresh decision
    could corrupt, and the part goes back through the same re-decision a resumed
    apply performs — the recorded dictionary disposition, the replay
    ``witness``, the frozen day, and :func:`_assert_intent_matches` over the
    outcome sets and both sides of every component binding. Recovering such a
    part straight from its payloads was a way to apply a promotion whose owner
    judgements had since moved, by resuming it instead of applying it.

    Live-file presence proves nothing in either direction: a part that
    legitimately keeps its live file because rows were held recovers by the
    same rule as one whose file the writer removed.
    """
    _bound_intent(config, prepared)
    _require_no_pending_curation(config, Path(prepared.staging_path))
    with _intent_locks(config, prepared):
        verdicts = _precheck_components(config, prepared)
        if _intent_started(prepared, verdicts):
            return _recovered(
                config,
                prepared,
                verdicts=verdicts,
                finished=_finish_intent_under_locks(config, prepared, verdicts),
            )
    # Outside the component locks, exactly as the resumed apply re-decides
    # outside them: `decide_promotion` reads these same paths and
    # `io.exclusive_path_lock` is not re-entrant. Nothing has been written
    # between the two measurements, and the second one below is what decides —
    # a part that became partly applied in between is then finished from its
    # intent rather than written twice.
    _redecide_for_intent(config, prepared, witness=witness)
    with _intent_locks(config, prepared):
        verdicts = _precheck_components(config, prepared)
        return _recovered(
            config,
            prepared,
            verdicts=verdicts,
            finished=_finish_intent_under_locks(config, prepared, verdicts),
        )


def recover_promotion_intent(
    config: ProjectConfig,
    prepared: PreparedSourcePromotion,
    *,
    witness: enrich.DictionaryLookup | None = None,
) -> PromotionRecovery:
    """Take §6's shared guard, then finish one interrupted promotion.

    The guard is outermost, ahead of every component lock, for the reason §6.5
    gives: one global lock always taken first is what stops a curation holding
    file A and waiting for B from deadlocking a recovery holding B and waiting
    for A. A coordinator already inside the guard calls
    :func:`recover_promotion_intent_under_guard`; `io.exclusive_path_lock` is
    not re-entrant, so taking it twice from one thread deadlocks.

    ``witness`` is the recorded dictionary fact book §7.3 froze before the
    preview. It is needed only where an unstarted part has to be decided again
    and its recorded disposition is ``consulted``; supplying none there refuses
    rather than asking the offline question or fetching a fresh answer.
    """
    from japanese_anki.application import study_curation

    with study_curation.curation_guard(config):
        return recover_promotion_intent_under_guard(config, prepared, witness=witness)


def plan_promotion(
    config: ProjectConfig,
    staging_path: Path,
    *,
    source: str = "",
    client: enrich.DictionaryLookup | None = None,
    skip_reading_check: bool | None = None,
) -> PromotionPlan:
    """What `janki promote` would do to this staging file, without doing it.

    The public projection of `decide_promotion` — the cards, the holds, the
    refusal, and nothing a caller could replay the write path with.

    One gap, named rather than hidden: see `PromotionPlan.readings_unchecked`.
    It is a gap only by default. Pass a `client` — as the command does, having
    already decided to spend the lookups — and the dictionary witness runs
    here too, and this says so. A page refresh passes nothing and gets the
    offline answer; one code path either way, which is the point.

    Raises rather than reports for one thing only: asking for the dictionary
    check without handing over a client. See `decide_promotion`.
    """
    decision = decide_promotion(
        config,
        staging_path,
        source=source,
        client=client,
        skip_reading_check=skip_reading_check,
    )
    return project_promotion(decision)


def project_promotion(decision: PromotionDecision) -> PromotionPlan:
    """The safe view of a decision."""
    base = {
        "source": decision.source,
        "staging_path": decision.staging_path,
        "readings_unchecked": not decision.consulted,
    }
    if decision.is_blocked:
        return PromotionPlan(**base, blocked=str(decision.error))

    if decision.state in ("nothing", "pattern_only"):
        return PromotionPlan(**base)
    already = decision.already_archived

    readings = decision.readings
    if readings is None:
        return PromotionPlan(**base, already_archived=already)

    held = tuple(
        HeldCard(
            record=record,
            reason=staging.annotations(record).get("hold_reason", ""),
        )
        for record in readings.held
    )
    warnings = decision.reading_warnings + decision.deck_warnings
    if decision.state == "nothing_lands":
        return PromotionPlan(
            **base, held=held, already_archived=already, warnings=warnings
        )

    merged_by_id = {record.id: record for record in decision.merged}
    by_id = {record.id: record for record in decision.existing}
    landing: list[LandingCard] = []
    promoted = iter(readings.promoted)
    for original, stays in zip(decision.work, readings.keep, strict=True):
        if stays:
            continue
        transformed = next(promoted)
        landing.append(
            LandingCard(
                staged=original,
                landing=merged_by_id[transformed.id],
                existing=by_id.get(transformed.id),
                # `check_readings` already decided this and says so; deriving
                # it again from the ids is a second copy of the same rule.
                reminted_from=(original.id if original.id in readings.reminted else ""),
            )
        )
    return PromotionPlan(
        **base,
        landing=tuple(landing),
        held=held,
        already_archived=already,
        warnings=warnings,
        deck_ownership=decision.deck_ownership,
    )
