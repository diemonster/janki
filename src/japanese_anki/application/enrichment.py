"""Plan and commit one exact jpdb dictionary-enrichment pass.

The dictionary lookup is deliberately separate from the write.  The CLI shows
the resulting field diff before asking for confirmation, and the workbench can
render the same decision later without gaining a second implementation of the
operation.  A commit binds that rendered decision to the exact collection text
it read and refuses a stale or mutated plan before either durable file changes.

This service is only the dictionary pass and never widens into ``enrich --ai``.
The workbench ordinary finish path supplies no force fields; the shared CLI seam
accepts only its already-validated, documented dictionary field names.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from japanese_anki import enrich, jpdb, kanji, ledger
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import RecordsRevision, load_records_snapshot, save_records_json

__all__ = [
    "DictionaryEnrichmentCommit",
    "DictionaryEnrichmentDecision",
    "DictionaryEnrichmentError",
    "commit_dictionary_enrichment",
    "plan_all_dictionary_enrichment",
    "plan_dictionary_enrichment",
]


class DictionaryEnrichmentError(JankiError):
    """An exact dictionary pass cannot be planned or committed safely."""


DictionaryEnrichmentCommitState = Literal[
    "nothing",
    "committed",
    "committed_ledger_incomplete",
]


@dataclass(frozen=True, slots=True)
class DictionaryEnrichmentDecision:
    """The exact lookup result shown before a canonical write is authorized."""

    repository_root: Path
    output_path: Path
    ledger_path: Path
    kanji_path: Path
    record_ids: tuple[str, ...]
    force_fields: tuple[str, ...]
    output_revision: RecordsRevision
    result: enrich.EnrichResult
    fingerprint: str


@dataclass(frozen=True, slots=True)
class DictionaryEnrichmentCommit:
    """The durable outcome, including the records-landed/ledger-failed split."""

    state: DictionaryEnrichmentCommitState
    output_path: Path
    changed_record_ids: tuple[str, ...] = ()
    cleared_record_ids: tuple[str, ...] = ()
    ledger_error: ledger.LedgerError | None = None


def _repository_binding(config: ProjectConfig) -> tuple[Path, Path, Path, Path]:
    return (
        config.root.resolve(),
        config.normalized_file.resolve(),
        config.ledger_file.resolve(),
        config.kanji_file.resolve(),
    )


def _exact_ids(ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(ids, (str, bytes)):
        raise DictionaryEnrichmentError(
            "Dictionary enrichment needs a nonempty sequence of exact record IDs."
        )
    values = tuple(ids)
    if not values:
        raise DictionaryEnrichmentError(
            "Dictionary enrichment needs at least one exact record ID; an empty "
            "scope never means the whole collection."
        )
    if any(not isinstance(record_id, str) or not record_id for record_id in values):
        raise DictionaryEnrichmentError(
            "Dictionary enrichment record IDs must be nonempty text values."
        )
    return tuple(dict.fromkeys(values))


def _force_fields(fields: Sequence[str]) -> tuple[str, ...]:
    if isinstance(fields, (str, bytes)):
        raise DictionaryEnrichmentError(
            "Dictionary force fields must be a sequence of exact field names."
        )
    values = tuple(dict.fromkeys(fields))
    invalid = [name for name in values if name not in enrich.ENRICHABLE_FIELDS]
    if invalid:
        raise DictionaryEnrichmentError(
            "Dictionary enrichment cannot force "
            f"{', '.join(repr(name) for name in invalid)}."
        )
    return values


def _revision_identity(revision: RecordsRevision) -> dict[str, object]:
    text = revision.text
    return {
        "present": text is not None,
        "sha256": hashlib.sha256((text or "").encode("utf-8")).hexdigest(),
    }


def _decision_fingerprint(decision: DictionaryEnrichmentDecision) -> str:
    result = decision.result
    payload = {
        "version": 1,
        "repository_root": str(decision.repository_root),
        "output_path": str(decision.output_path),
        "ledger_path": str(decision.ledger_path),
        "kanji_path": str(decision.kanji_path),
        "record_ids": list(decision.record_ids),
        "force_fields": list(decision.force_fields),
        "output_revision": _revision_identity(decision.output_revision),
        # Bind the complete would-be write, not only the human-facing diff.  A
        # later controller may render a subset of these facts, but it can never
        # mutate an unrendered record and retain this decision's authority.
        "records": [record.to_dict() for record in result.records],
        "changes": result.changes,
        "cleared": result.cleared,
        "warnings": result.warnings,
        "skipped": result.skipped,
        "looked_up": result.looked_up,
    }
    wire = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _plan(
    config: ProjectConfig,
    client: jpdb.JpdbClient,
    record_ids: tuple[str, ...] | None,
    force_fields: tuple[str, ...],
) -> DictionaryEnrichmentDecision | None:
    repository_root, output_path, ledger_path, kanji_path = _repository_binding(config)
    records, output_revision = load_records_snapshot(output_path)
    if record_ids is None:
        if not records:
            return None
        targets = tuple(record.id for record in records)
    else:
        targets = record_ids

    # Preserve the CLI's pre-write refusal: an unreadable ledger is discovered
    # before any lookup and certainly before records can land.  Commit reloads
    # it because a browser decision may remain open after this read.
    ledger.load(ledger_path)
    result = enrich.enrich_records(
        client,
        records,
        force_fields=force_fields,
        ids=targets,
        kanji_store=kanji.load_store(kanji_path),
    )
    draft = DictionaryEnrichmentDecision(
        repository_root=repository_root,
        output_path=output_path,
        ledger_path=ledger_path,
        kanji_path=kanji_path,
        record_ids=targets,
        force_fields=force_fields,
        output_revision=output_revision,
        result=result,
        fingerprint="",
    )
    return replace(draft, fingerprint=_decision_fingerprint(draft))


def plan_dictionary_enrichment(
    config: ProjectConfig,
    client: jpdb.JpdbClient,
    ids: Sequence[str],
    *,
    force_fields: Sequence[str] = (),
) -> DictionaryEnrichmentDecision:
    """Look up one nonempty, exact record scope without writing anything."""
    decision = _plan(config, client, _exact_ids(ids), _force_fields(force_fields))
    assert decision is not None
    return decision


def plan_all_dictionary_enrichment(
    config: ProjectConfig,
    client: jpdb.JpdbClient,
    *,
    force_fields: Sequence[str] = (),
) -> DictionaryEnrichmentDecision | None:
    """Explicit CLI compatibility for its historical whole-collection mode."""
    return _plan(config, client, None, _force_fields(force_fields))


def commit_dictionary_enrichment(
    config: ProjectConfig,
    decision: DictionaryEnrichmentDecision,
    *,
    expected_fingerprint: str,
) -> DictionaryEnrichmentCommit:
    """Commit one confirmed decision, refusing mutation or collection drift.

    The records write remains authoritative when the later ledger save fails,
    matching the CLI's established partial-write contract.  That split is a
    returned state rather than an exception so every surface can say exactly
    what changed.
    """
    if _repository_binding(config) != (
        decision.repository_root,
        decision.output_path,
        decision.ledger_path,
        decision.kanji_path,
    ):
        raise DictionaryEnrichmentError(
            "[dictionary-config-mismatch] this enrichment decision belongs to a "
            "different repository configuration. Nothing was written."
        )
    current_fingerprint = _decision_fingerprint(decision)
    if (
        not isinstance(expected_fingerprint, str)
        or not secrets.compare_digest(expected_fingerprint, decision.fingerprint)
        or not secrets.compare_digest(current_fingerprint, decision.fingerprint)
    ):
        raise DictionaryEnrichmentError(
            "[dictionary-plan-stale] the exact dictionary result changed after it "
            "was prepared. Nothing was written; plan and review it again."
        )

    result = decision.result
    changed_ids = tuple(result.changes)
    cleared_ids = tuple(result.cleared)
    if not changed_ids and not cleared_ids:
        return DictionaryEnrichmentCommit(
            state="nothing",
            output_path=decision.output_path,
        )

    # Validate and prepare attribution before the canonical write.  It remains
    # in memory until that write wins its CAS, so drift cannot write the ledger.
    book = ledger.load(decision.ledger_path)
    for record_id, changed in result.changes.items():
        book.record_enriched(
            record_id,
            kind="jpdb",
            model="jpdb",
            fields=changed,
        )
    save_records_json(
        decision.output_path,
        result.records,
        expected=decision.output_revision,
    )

    ledger_error: ledger.LedgerError | None = None
    if changed_ids:
        try:
            book.save()
        except ledger.LedgerError as exc:
            ledger_error = exc
    return DictionaryEnrichmentCommit(
        state=(
            "committed_ledger_incomplete" if ledger_error is not None else "committed"
        ),
        output_path=decision.output_path,
        changed_record_ids=changed_ids,
        cleared_record_ids=cleared_ids,
        ledger_error=ledger_error,
    )
