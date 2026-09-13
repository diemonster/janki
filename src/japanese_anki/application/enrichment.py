"""Plan and commit one exact jpdb dictionary-enrichment pass.

The dictionary lookup is deliberately separate from the write.  The CLI shows
the resulting field diff before asking for confirmation, and the workbench can
render the same decision later without gaining a second implementation of the
operation.  A commit binds that rendered decision to the exact collection text
it read and refuses a stale or mutated plan before either durable file changes.

This service is only the dictionary pass and never widens into ``enrich --ai``.
The workbench ordinary finish path supplies no force fields; the shared CLI seam
accepts only its already-validated, documented dictionary field names.

Three seams beyond that ordinary pair, all contracts §7.8:

* :func:`plan_dictionary_enrichment_revision` decides over records a caller
  supplies, the revision they came from, and an **explicitly frozen** reference
  store — which is how a study finish models this phase over its projected
  post-promotion collection before anything has been written. Its decision is
  ``projected``, and :func:`commit_dictionary_enrichment` refuses a projected
  decision at its first statement: a chained canonical projection can never be
  replayed as a transaction.
* :func:`prepare_dictionary_enrichment` freezes the canonical and ledger
  components — both complete payloads, both digests, and the one date the
  ledger write carries — and publishes nothing.
* :func:`apply_prepared_dictionary_enrichment` and
  :func:`recover_prepared_dictionary_enrichment` precheck that whole vector
  under the canonical and ledger locks before any write, finish a part whose
  own writes have begun from its frozen payloads, and re-plan a genuinely
  unstarted one through the replay dictionary client only.

Every entry that writes takes §6's shared curation guard outermost, ahead of
the two path locks.  ``io.exclusive_path_lock`` is not re-entrant, so a
coordinator already holding the guard uses the ``_under_guard`` entry.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Literal

from japanese_anki import enrich, kanji, ledger, staging
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    RecordsRevision,
    atomic_write_text_bound,
    exclusive_path_lock,
    load_records_snapshot,
    read_bytes_bound,
    records_json_text,
    save_records_json_locked,
)

__all__ = [
    "DictionaryEnrichmentCommit",
    "DictionaryEnrichmentDecision",
    "DictionaryEnrichmentError",
    "DictionaryEnrichmentRecovery",
    "PreparedDictionaryEnrichment",
    "apply_prepared_dictionary_enrichment",
    "apply_prepared_dictionary_enrichment_under_guard",
    "commit_dictionary_enrichment",
    "commit_dictionary_enrichment_under_guard",
    "plan_all_dictionary_enrichment",
    "plan_dictionary_enrichment",
    "plan_dictionary_enrichment_revision",
    "prepare_dictionary_enrichment",
    "recover_prepared_dictionary_enrichment",
    "recover_prepared_dictionary_enrichment_under_guard",
]


class DictionaryEnrichmentError(JankiError):
    """An exact dictionary pass cannot be planned or committed safely."""


DictionaryEnrichmentCommitState = Literal[
    "nothing",
    "committed",
    "committed_ledger_incomplete",
]

#: The one refusal for a decision planned over a collection that is not on
#: disk. It is the first statement of every entry that could reach a write,
#: ahead of the repository compare and ahead of the early ``nothing`` return —
#: both of which a projection would otherwise sail straight through.
_PROJECTION_NOT_EXECUTABLE = (
    "[dictionary-projection-not-executable] this dictionary decision was "
    "planned over a projected collection rather than the one on disk. A "
    "chained canonical projection is a preview, never a transaction. Nothing "
    "was written; plan it again over the real collection."
)

#: The two files one dictionary pass writes, in §7.5's order: canonical, then
#: the ledger. Every lock, precheck and replay below follows this sequence.
_COMPONENT_ROLES: tuple[str, ...] = ("canonical", "ledger")


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
    #: True whenever the records, the revision or the reference store were
    #: supplied rather than read. Such a decision may be rendered and compared;
    #: it may never be committed.
    projected: bool = False


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
        "version": 2,
        "projected": decision.projected,
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


def _decided(
    config: ProjectConfig,
    *,
    record_ids: tuple[str, ...],
    force_fields: tuple[str, ...],
    output_revision: RecordsRevision,
    result: enrich.EnrichResult,
    projected: bool,
) -> DictionaryEnrichmentDecision:
    repository_root, output_path, ledger_path, kanji_path = _repository_binding(config)
    draft = DictionaryEnrichmentDecision(
        repository_root=repository_root,
        output_path=output_path,
        ledger_path=ledger_path,
        kanji_path=kanji_path,
        record_ids=record_ids,
        force_fields=force_fields,
        output_revision=output_revision,
        result=result,
        fingerprint="",
        projected=projected,
    )
    return replace(draft, fingerprint=_decision_fingerprint(draft))


def _plan(
    config: ProjectConfig,
    client: enrich.DictionaryLookup,
    record_ids: tuple[str, ...] | None,
    force_fields: tuple[str, ...],
) -> DictionaryEnrichmentDecision | None:
    _repository_root, output_path, ledger_path, kanji_path = _repository_binding(config)
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
    result = enrich.decide_enrichment(
        client,
        records,
        output_revision,
        ids=targets,
        force_fields=force_fields,
        kanji_store=kanji.load_store(kanji_path),
    )
    return _decided(
        config,
        record_ids=targets,
        force_fields=force_fields,
        output_revision=output_revision,
        result=result,
        projected=False,
    )


def plan_dictionary_enrichment(
    config: ProjectConfig,
    client: enrich.DictionaryLookup,
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
    client: enrich.DictionaryLookup,
    *,
    force_fields: Sequence[str] = (),
) -> DictionaryEnrichmentDecision | None:
    """Explicit CLI compatibility for its historical whole-collection mode."""
    return _plan(config, client, None, _force_fields(force_fields))


def plan_dictionary_enrichment_revision(
    config: ProjectConfig,
    client: enrich.DictionaryLookup,
    records: Sequence[Any],
    revision: RecordsRevision,
    ids: Sequence[str],
    *,
    force_fields: Sequence[str] = (),
    kanji_store: Any,
) -> DictionaryEnrichmentDecision:
    """Decide one pass over a supplied collection, revision and reference store.

    Modelled on ``audio.plan_targeted_audio_revision``: the caller has already
    resolved which records this phase sees, and hands them in with the exact
    revision they belong to.  For a study finish that collection is the
    post-promotion projection §7.2 chained — it is not on disk yet, so a read
    here would answer about a different repository state.

    ``kanji_store`` is **required and explicit** for the same reason.  It is
    §7.9's frozen projected store, the one the preview renders over, never a
    read of the live path before those reference facts have been written.
    Passing ``None`` is the ordinary "nobody has run `janki kanji`" state and
    still has to be said out loud.

    The ledger preflight ``_plan`` performs is kept: an unreadable ledger is
    discovered before any lookup.  The returned decision is ``projected``, so
    :func:`commit_dictionary_enrichment` refuses it at entry.
    """
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise DictionaryEnrichmentError(
            "A projected dictionary pass needs the sequence of records it "
            "decides over."
        )
    if not isinstance(revision, RecordsRevision):
        raise DictionaryEnrichmentError(
            "A projected dictionary pass needs the RecordsRevision its records "
            "came from, so its scope refusal names that revision's own path."
        )
    _repository_root, output_path, ledger_path, _kanji_path = _repository_binding(config)
    if Path(revision.path) != output_path:
        raise DictionaryEnrichmentError(
            f"[dictionary-projection-unbound] this projection carries the "
            f"revision of {revision.path}, and this configuration's collection "
            f"is {output_path}."
        )
    targets = _exact_ids(ids)
    fields = _force_fields(force_fields)
    ledger.load(ledger_path)
    result = enrich.decide_enrichment(
        client,
        list(records),
        revision,
        ids=targets,
        force_fields=fields,
        kanji_store=kanji_store,
    )
    return _decided(
        config,
        record_ids=targets,
        force_fields=fields,
        output_revision=revision,
        result=result,
        projected=True,
    )


# --- the one canonical -> ledger transaction ---------------------------------


def _require_executable(decision: DictionaryEnrichmentDecision) -> None:
    """The first statement of every entry a projection could otherwise reach."""
    if decision.projected:
        raise DictionaryEnrichmentError(_PROJECTION_NOT_EXECUTABLE)


def _require_repository(
    config: ProjectConfig, decision: DictionaryEnrichmentDecision
) -> None:
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


def _require_unmutated(
    decision: DictionaryEnrichmentDecision, expected_fingerprint: str | None
) -> None:
    current_fingerprint = _decision_fingerprint(decision)
    if not secrets.compare_digest(current_fingerprint, decision.fingerprint) or (
        expected_fingerprint is not None
        and (
            not isinstance(expected_fingerprint, str)
            or not secrets.compare_digest(expected_fingerprint, decision.fingerprint)
        )
    ):
        raise DictionaryEnrichmentError(
            "[dictionary-plan-stale] the exact dictionary result changed after it "
            "was prepared. Nothing was written; plan and review it again."
        )


def _digest(value: bytes | str | None) -> str | None:
    if value is None:
        return None
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _file_wire(path: Path) -> bytes | None:
    """One bound file's exact current bytes, or ``None`` when it is absent."""
    try:
        return read_bytes_bound(Path(path))
    except FileNotFoundError:
        return None
    except JankiError as exc:
        raise DictionaryEnrichmentError(f"Could not read {path}: {exc}") from exc


def _component(
    role: str, path: Path, *, before: str | None, after_text: str | None
) -> staging.PreparedComponent:
    """One prepared component, with absence preserved on either side."""
    after = before if after_text is None else _digest(after_text)
    return staging.PreparedComponent(
        role=role,
        path=str(path),
        expected_before=before,
        expected_after=after,
        after_text=after_text,
    )


def _component_path(config: ProjectConfig, role: str) -> Path:
    """The path this repository's configuration gives that role.

    A closed role list resolved from the live configuration, so an altered
    intent cannot re-point an enrichment at a file its phase never planned to
    touch.
    """
    if role == "canonical":
        return config.normalized_file.resolve()
    if role == "ledger":
        return config.ledger_file.resolve()
    raise DictionaryEnrichmentError(
        f"[enrichment-intent-invalid] unknown component role {role!r}"
    )


def _attributed_ledger(
    ledger_path: Path,
    wire: bytes | None,
    changes: Mapping[str, Mapping[str, Any]],
    *,
    at: str | None,
) -> ledger.Ledger:
    """The ledger with this pass's attribution applied, in memory only.

    ``at`` is §7.1's frozen day.  A resume on the day after the crash replays
    exactly these bytes instead of re-deriving a date, which is what keeps the
    bound ``expected_after`` reachable.
    """
    book = ledger.load_snapshot(ledger_path, wire)
    for record_id, changed in changes.items():
        book.record_enriched(
            record_id, kind="jpdb", model="jpdb", fields=changed, at=at
        )
    return book


def _commit_under_locks(
    decision: DictionaryEnrichmentDecision,
    *,
    at: str | None,
) -> DictionaryEnrichmentCommit:
    """Write canonical, then the ledger, with both path locks already held.

    The one writer.  The ordinary confirmed commit and a study finish's
    prepared apply both come through here, so there is no second transaction
    that could drift from this one.  The prepared apply needs no extra
    payload check here: `_assert_intent_matches` derives the fresh components
    from this same decision through `_prepared_from`, so the bytes compared and
    the bytes written are one derivation, under one lock.

    ``io.exclusive_path_lock`` is not re-entrant, which is why this calls
    ``save_records_json_locked`` and ``Ledger.save_under_lock`` rather than
    their lock-taking siblings.
    """
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
    book = _attributed_ledger(
        decision.ledger_path,
        _file_wire(decision.ledger_path),
        result.changes,
        at=at,
    )
    save_records_json_locked(
        decision.output_path,
        result.records,
        expected=decision.output_revision,
    )

    ledger_error: ledger.LedgerError | None = None
    if changed_ids:
        try:
            book.save_under_lock()
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


@contextlib.contextmanager
def _paths_locked(paths: Iterable[Path]):
    """Hold the given paths' locks, in the order given, deduplicated.

    §7.5's order for this phase is canonical, then the ledger. Sorting them by
    name instead would produce a different order on an ordinary layout, and two
    writers taking one pair of locks in two orders is the deadlock that order
    exists to prevent.
    """
    with ExitStack() as locks:
        held: set[Path] = set()
        for path in paths:
            target = Path(os.path.realpath(path))
            if target in held:
                continue
            held.add(target)
            locks.enter_context(exclusive_path_lock(target))
        yield


def commit_dictionary_enrichment(
    config: ProjectConfig,
    decision: DictionaryEnrichmentDecision,
    *,
    expected_fingerprint: str,
) -> DictionaryEnrichmentCommit:
    """Take §6's shared guard, then commit one confirmed decision.

    The guard is outermost, ahead of the canonical and ledger locks, for the
    reason §6.5 gives: one global lock always taken first is what stops a
    curation holding file A and waiting for B from deadlocking a write holding
    B and waiting for A.  A coordinator already inside the guard calls
    :func:`commit_dictionary_enrichment_under_guard`; ``exclusive_path_lock``
    is not re-entrant, so taking it twice from one thread deadlocks.
    """
    _require_executable(decision)
    from japanese_anki.application import study_curation

    with study_curation.curation_guard(config):
        return commit_dictionary_enrichment_under_guard(
            config, decision, expected_fingerprint=expected_fingerprint
        )


def commit_dictionary_enrichment_under_guard(
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
    _require_executable(decision)
    _require_repository(config, decision)
    _require_unmutated(decision, expected_fingerprint)
    with _paths_locked((decision.output_path, decision.ledger_path)):
        return _commit_under_locks(decision, at=None)


# --- the prepared intent -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedDictionaryEnrichment:
    """One dictionary pass's complete intent, frozen before its first write.

    Everything a resume cannot recompute is here: the exact field values this
    pass is authorized to write, the fact book they were decided from, the one
    day the ledger attribution carries, and each of the two paths' whole
    after-payload with its before binding.  A crash between the canonical write
    and the ledger save recovers from this value alone — a returned result that
    was never persisted is not evidence, and neither is a writer's state label.
    """

    repository_root: str
    output_path: str
    ledger_path: str
    kanji_path: str
    record_ids: tuple[str, ...]
    force_fields: tuple[str, ...]
    #: The ISO day §7.1 freezes, threaded into ``record_enriched(at=)`` so the
    #: ledger component's ``expected_after`` is the same digest on the day of
    #: the crash and the day after it.
    enriched_at: str
    #: ``record id -> {field: exact new value}``.  A resume re-plans and every
    #: one of these must come back identical, naming both values when it does
    #: not.  ``source_forms`` can never appear here: it is not enrichable.
    bound_values: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: ``record id -> (field, …)`` whose provisional mark this pass clears.
    #: A separate channel because a cleared-only pass writes canonical and no
    #: attribution row at all.
    cleared_fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: The fingerprint of the frozen fact book these values were decided from,
    #: or ``""`` where the caller bound none.  Compared at each apply.
    book_fingerprint: str = ""
    components: tuple[staging.PreparedComponent, ...] = ()

    @property
    def changed_record_ids(self) -> tuple[str, ...]:
        return tuple(self.bound_values)

    @property
    def cleared_record_ids(self) -> tuple[str, ...]:
        return tuple(self.cleared_fields)

    def component(self, role: str) -> staging.PreparedComponent | None:
        for item in self.components:
            if item.role == role:
                return item
        return None

    @property
    def projected_input_sha256(self) -> str | None:
        """The canonical bytes this pass starts from, for §7.4's authority."""
        canonical = self.component("canonical")
        return None if canonical is None else canonical.expected_before

    @property
    def projected_output_sha256(self) -> str | None:
        """The canonical bytes this pass finishes at, for §7.4's authority."""
        canonical = self.component("canonical")
        return None if canonical is None else canonical.expected_after

    @property
    def writes(self) -> bool:
        return any(item.writes for item in self.components)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "repository_root": self.repository_root,
            "output_path": self.output_path,
            "ledger_path": self.ledger_path,
            "kanji_path": self.kanji_path,
            "record_ids": list(self.record_ids),
            "force_fields": list(self.force_fields),
            "enriched_at": self.enriched_at,
            "bound_values": {
                record_id: dict(fields)
                for record_id, fields in self.bound_values.items()
            },
            "cleared_fields": {
                record_id: list(names)
                for record_id, names in self.cleared_fields.items()
            },
            "book_fingerprint": self.book_fingerprint,
            "components": [item.to_dict() for item in self.components],
        }
        payload["fingerprint"] = _intent_fingerprint(payload)
        return payload

    @property
    def fingerprint(self) -> str:
        """Derived, never stored: an altered intent cannot keep its own seal."""
        return str(self.to_dict()["fingerprint"])

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PreparedDictionaryEnrichment:
        try:
            if raw.get("schema_version") != 1:
                raise DictionaryEnrichmentError(
                    "Unknown prepared dictionary enrichment version "
                    f"{raw.get('schema_version')!r}."
                )
            prepared = cls(
                repository_root=str(raw["repository_root"]),
                output_path=str(raw["output_path"]),
                ledger_path=str(raw["ledger_path"]),
                kanji_path=str(raw["kanji_path"]),
                record_ids=tuple(str(item) for item in raw["record_ids"]),
                force_fields=tuple(str(item) for item in raw["force_fields"]),
                enriched_at=str(raw["enriched_at"]),
                bound_values={
                    str(record_id): dict(fields)
                    for record_id, fields in raw["bound_values"].items()
                },
                cleared_fields={
                    str(record_id): tuple(str(name) for name in names)
                    for record_id, names in raw["cleared_fields"].items()
                },
                book_fingerprint=str(raw["book_fingerprint"]),
                components=tuple(
                    staging.PreparedComponent.from_dict(item)
                    for item in raw["components"]
                ),
            )
        except DictionaryEnrichmentError:
            raise
        except (JankiError, KeyError, TypeError, ValueError) as exc:
            raise DictionaryEnrichmentError(
                f"This is not a complete prepared dictionary enrichment: {exc}"
            ) from exc
        recorded = raw.get("fingerprint")
        if recorded is not None and str(recorded) != prepared.fingerprint:
            raise DictionaryEnrichmentError(
                "[enrichment-intent-stale] the stored dictionary enrichment "
                "intent does not match its own fingerprint. Nothing was "
                "written; prepare it again."
            )
        return prepared


def _intent_fingerprint(payload: Mapping[str, Any]) -> str:
    wire = json.dumps(
        {key: value for key, value in payload.items() if key != "fingerprint"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


@dataclass(frozen=True, slots=True)
class DictionaryEnrichmentRecovery:
    """What one recovery found already done, and what this pass finished."""

    state: DictionaryEnrichmentCommitState
    output_path: Path
    already_complete: tuple[str, ...]
    finished: tuple[str, ...]
    changed_record_ids: tuple[str, ...] = ()
    cleared_record_ids: tuple[str, ...] = ()
    ledger_error: ledger.LedgerError | None = None


def _prepared_from(
    config: ProjectConfig,
    decision: DictionaryEnrichmentDecision,
    *,
    day: date,
    book_fingerprint: str,
) -> PreparedDictionaryEnrichment:
    """Freeze one decision's whole component vector without writing anything."""
    result = decision.result
    changed_ids = tuple(result.changes)
    cleared_ids = tuple(result.cleared)
    at = day.isoformat()

    canonical_before = _digest(decision.output_revision.text)
    # A pass with no effect binds canonical unwritten rather than proposing the
    # saver's normalisation of bytes nobody asked it to touch.
    canonical_after = (
        records_json_text(result.records) if (changed_ids or cleared_ids) else None
    )

    ledger_wire = _file_wire(decision.ledger_path)
    ledger_after = (
        _attributed_ledger(
            decision.ledger_path, ledger_wire, result.changes, at=at
        ).serialized_text()
        if changed_ids
        else None
    )

    return PreparedDictionaryEnrichment(
        repository_root=str(decision.repository_root),
        output_path=str(decision.output_path),
        ledger_path=str(decision.ledger_path),
        kanji_path=str(decision.kanji_path),
        record_ids=decision.record_ids,
        force_fields=decision.force_fields,
        enriched_at=at,
        bound_values={
            record_id: {name: new for name, (_old, new) in fields.items()}
            for record_id, fields in result.changes.items()
        },
        cleared_fields={
            record_id: tuple(names) for record_id, names in result.cleared.items()
        },
        book_fingerprint=book_fingerprint,
        components=(
            _component(
                "canonical",
                decision.output_path,
                before=canonical_before,
                after_text=canonical_after,
            ),
            _component(
                "ledger",
                decision.ledger_path,
                before=_digest(ledger_wire),
                after_text=ledger_after,
            ),
        ),
    )


def prepare_dictionary_enrichment(
    config: ProjectConfig,
    decision: DictionaryEnrichmentDecision,
    *,
    now: date | None = None,
    book: enrich.DictionaryFactBook | None = None,
) -> PreparedDictionaryEnrichment:
    """Freeze one decided pass, publishing nothing.

    Side-effect-free on canonical and on the ledger: it reads them, renders the
    exact bytes each would hold, and returns them for a coordinator to persist
    before the first mutation.  ``now`` defaults to today and is the one day
    the ledger attribution carries from here on.

    A projected decision is refused: its canonical before-digest would bind
    bytes that are not on disk, and an apply could then never reach either of
    its own bound states.
    """
    _require_executable(decision)
    _require_repository(config, decision)
    _require_unmutated(decision, None)
    return _prepared_from(
        config,
        decision,
        day=now or date.today(),
        book_fingerprint="" if book is None else book.fingerprint,
    )


# --- applying and recovering one prepared pass ---------------------------------


def _bound_intent(config: ProjectConfig, prepared: PreparedDictionaryEnrichment) -> None:
    """Re-prove the configuration and role associations this intent claims.

    A durable intent names paths, and a path is exactly the part of it a later
    configuration or an editor can move.  Proved before the repository is read,
    so an altered target cannot borrow a valid vector.
    """
    repository_root, output_path, ledger_path, kanji_path = _repository_binding(config)
    claimed = (
        prepared.repository_root,
        prepared.output_path,
        prepared.ledger_path,
        prepared.kanji_path,
    )
    if claimed != (
        str(repository_root),
        str(output_path),
        str(ledger_path),
        str(kanji_path),
    ):
        raise DictionaryEnrichmentError(
            "[enrichment-intent-invalid] this dictionary enrichment intent "
            "belongs to a different repository configuration. Nothing was "
            "written."
        )
    roles = tuple(item.role for item in prepared.components)
    if roles != _COMPONENT_ROLES:
        raise DictionaryEnrichmentError(
            "[enrichment-intent-invalid] a prepared dictionary pass binds "
            f"exactly {', '.join(_COMPONENT_ROLES)}, in that order, and this "
            f"one binds {', '.join(roles) or 'nothing'}."
        )
    if not prepared.record_ids:
        raise DictionaryEnrichmentError(
            "[enrichment-intent-invalid] a prepared dictionary pass names at "
            "least one exact record id; an empty scope never means the whole "
            "collection."
        )
    try:
        date.fromisoformat(prepared.enriched_at)
    except ValueError as exc:
        raise DictionaryEnrichmentError(
            f"[enrichment-intent-invalid] {prepared.enriched_at!r} is not an "
            "ISO date this pass can replay."
        ) from exc
    for item in prepared.components:
        expected = _component_path(config, item.role)
        if Path(item.path) != expected:
            raise DictionaryEnrichmentError(
                f"[enrichment-intent-stale] the prepared {item.role} component "
                f"names {item.path}, but this configuration resolves that role "
                f"to {expected}. Nothing was written."
            )


def _component_verdict(component: staging.PreparedComponent, observed: str | None) -> str:
    """``"pending"``, ``"complete"``, or a refusal naming both bound digests."""
    if observed == component.expected_before:
        return "pending" if component.writes else "complete"
    if observed == component.expected_after:
        return "complete"
    raise DictionaryEnrichmentError(
        f"[enrichment-intent-stale] the prepared {component.role} component "
        f"{component.path} is at neither bound state: expected "
        f"{component.expected_before} before or {component.expected_after} "
        f"after, found {observed}. Nothing was written."
    )


def _precheck_components(
    config: ProjectConfig, prepared: PreparedDictionaryEnrichment
) -> dict[str, str]:
    """Measure the **entire** bound vector before any of it is written.

    Every component, not only the pending writes: a bound file that changes
    nothing — a ledger a cleared-only pass leaves alone, a ledger that is
    absent and stays absent — is still evidence that this intent describes the
    repository it is about to write to.  One component at neither digest
    refuses the whole pass and names both digests, so a stale ledger stops the
    canonical write rather than being discovered after it.
    """
    verdicts: dict[str, str] = {}
    for component in prepared.components:
        verdicts[component.role] = _component_verdict(
            component, _digest(_file_wire(Path(component.path)))
        )
    return verdicts


def _intent_started(
    prepared: PreparedDictionaryEnrichment, verdicts: Mapping[str, str]
) -> bool:
    """Whether any write this intent owns has already landed."""
    return any(
        component.writes and verdicts.get(component.role) == "complete"
        for component in prepared.components
    )


def _write_component(component: staging.PreparedComponent) -> None:
    atomic_write_text_bound(
        Path(component.path),
        component.after_text or "",
        expected_revision=component.expected_before,
        expected_absent=component.expected_before is None,
    )


def _finish_intent_under_locks(
    prepared: PreparedDictionaryEnrichment, verdicts: Mapping[str, str]
) -> tuple[tuple[str, ...], ledger.LedgerError | None]:
    """Replay only the writes this intent still owes, in canonical→ledger order.

    A failed *ledger* replay is not an exception here: the records are
    committed and their attribution is not, which is exactly the
    ``committed_ledger_incomplete`` split the writer itself reports, reached
    from the intent alone when the process died before returning it.
    """
    finished: list[str] = []
    ledger_error: ledger.LedgerError | None = None
    for component in prepared.components:
        if not component.writes or verdicts.get(component.role) != "pending":
            continue
        try:
            _write_component(component)
        except JankiError as exc:
            if component.role == "ledger":
                ledger_error = ledger.LedgerError(str(exc))
                break
            raise DictionaryEnrichmentError(
                f"[enrichment-recovery-incomplete] could not finish the "
                f"{component.role} write to {component.path}: {exc}"
            ) from exc
        finished.append(component.role)
    return tuple(finished), ledger_error


def _intent_state(
    prepared: PreparedDictionaryEnrichment, ledger_error: ledger.LedgerError | None
) -> DictionaryEnrichmentCommitState:
    if ledger_error is not None:
        return "committed_ledger_incomplete"
    return "committed" if prepared.writes else "nothing"


def _intent_commit(
    prepared: PreparedDictionaryEnrichment, ledger_error: ledger.LedgerError | None
) -> DictionaryEnrichmentCommit:
    """A finished intent reports exactly what the writer would have."""
    return DictionaryEnrichmentCommit(
        state=_intent_state(prepared, ledger_error),
        output_path=Path(prepared.output_path),
        changed_record_ids=prepared.changed_record_ids,
        cleared_record_ids=prepared.cleared_record_ids,
        ledger_error=ledger_error,
    )


def _already_complete(
    prepared: PreparedDictionaryEnrichment, verdicts: Mapping[str, str]
) -> tuple[str, ...]:
    """The writes that had already landed before this pass touched anything."""
    return tuple(
        component.role
        for component in prepared.components
        if component.writes and verdicts.get(component.role) == "complete"
    )


def _redecide_for_intent(
    config: ProjectConfig,
    prepared: PreparedDictionaryEnrichment,
    *,
    client: enrich.DictionaryLookup | None,
) -> DictionaryEnrichmentDecision:
    """Plan this intent's exact scope again, from the replayed facts only.

    §7.8's apply re-plans with the ordinary planner over real canonical for
    exactly those ids, applying the saved fact book's values with no network
    refetch and reading the reference store §7.9 has already written.  That is
    this call: the live `plan_dictionary_enrichment`, handed a replay client.
    """
    if client is None:
        raise DictionaryEnrichmentError(
            "[enrichment-intent-client-required] resuming an unstarted "
            "dictionary pass re-plans its exact scope, and the only dictionary "
            "it may ask is the recorded fact book. Supply the replay client."
        )
    if prepared.book_fingerprint:
        actual = getattr(getattr(client, "book", None), "fingerprint", None)
        if actual != prepared.book_fingerprint:
            raise DictionaryEnrichmentError(
                "[enrichment-intent-book-mismatch] this intent was decided from "
                f"fact book {prepared.book_fingerprint} and the supplied client "
                f"replays {actual}. Nothing was written."
            )
    return plan_dictionary_enrichment(
        config,
        client,
        prepared.record_ids,
        force_fields=prepared.force_fields,
    )


def _assert_intent_matches(
    config: ProjectConfig,
    prepared: PreparedDictionaryEnrichment,
    decision: DictionaryEnrichmentDecision,
) -> None:
    """Compare a freshly decided pass with exactly what the intent bound.

    Every authorized value first — naming both the bound one and the one the
    repository now computes — and then **both** sides of each component
    binding.  Comparing only the before-state would accept a repository that
    starts where the intent expected and finishes somewhere else, which is
    precisely what a frozen ledger date exists to rule out.
    """
    fresh = _prepared_from(
        config,
        decision,
        day=date.fromisoformat(prepared.enriched_at),
        book_fingerprint=prepared.book_fingerprint,
    )
    for label, expected, actual in (
        ("record ids", prepared.record_ids, fresh.record_ids),
        ("force fields", prepared.force_fields, fresh.force_fields),
        (
            "cleared marks",
            {key: tuple(value) for key, value in prepared.cleared_fields.items()},
            {key: tuple(value) for key, value in fresh.cleared_fields.items()},
        ),
    ):
        if expected != actual:
            raise DictionaryEnrichmentError(
                f"[enrichment-intent-stale] this dictionary pass now writes a "
                f"different {label}: the intent bound {expected!r} and the "
                f"repository decides {actual!r}. Nothing was written."
            )
    bound_records = set(prepared.bound_values) | set(fresh.bound_values)
    for record_id in sorted(bound_records):
        expected_fields = dict(prepared.bound_values.get(record_id, {}))
        actual_fields = dict(fresh.bound_values.get(record_id, {}))
        for name in sorted(set(expected_fields) | set(actual_fields)):
            if expected_fields.get(name) != actual_fields.get(name):
                raise DictionaryEnrichmentError(
                    f"[enrichment-intent-stale] {record_id}'s {name} no longer "
                    f"matches: the intent bound {expected_fields.get(name)!r} "
                    f"and the repository computes {actual_fields.get(name)!r}. "
                    "Nothing was written."
                )
    for component in prepared.components:
        current = fresh.component(component.role)
        assert current is not None  # proved by the role comparison in _bound_intent
        for side, expected_digest, actual_digest in (
            ("starts from", component.expected_before, current.expected_before),
            ("finishes at", component.expected_after, current.expected_after),
        ):
            if expected_digest != actual_digest:
                raise DictionaryEnrichmentError(
                    f"[enrichment-intent-stale] the {component.role} component "
                    f"no longer {side} the bound state {expected_digest}; the "
                    f"repository decides {actual_digest}. Nothing was written."
                )


def apply_prepared_dictionary_enrichment(
    config: ProjectConfig,
    prepared: PreparedDictionaryEnrichment,
    *,
    client: enrich.DictionaryLookup | None = None,
    decision: DictionaryEnrichmentDecision | None = None,
) -> DictionaryEnrichmentCommit:
    """Take §6's shared guard, then apply one prepared dictionary pass."""
    from japanese_anki.application import study_curation

    with study_curation.curation_guard(config):
        return apply_prepared_dictionary_enrichment_under_guard(
            config, prepared, client=client, decision=decision
        )


def apply_prepared_dictionary_enrichment_under_guard(
    config: ProjectConfig,
    prepared: PreparedDictionaryEnrichment,
    *,
    client: enrich.DictionaryLookup | None = None,
    decision: DictionaryEnrichmentDecision | None = None,
) -> DictionaryEnrichmentCommit:
    """Perform one prepared pass through the writer that owns these two files.

    ``decision`` is the decision the intent was prepared from, when the caller
    still holds it.  Without it this is a resumed finish, and the order is the
    whole of §7.8's recovery contract:

    1. the configuration, source and role associations this intent claims are
       re-proved before the repository is consulted;
    2. the **entire** bound vector is classified while both of its paths are
       locked, before anything is decided again;
    3. a pass whose own writes have begun is finished from its intent — never
       re-planned, because a fresh decision reads those writes as somebody
       else's and either refuses or describes a different transaction;
    4. only a genuinely unstarted pass is decided again, through the replay
       client alone, and every authorized value and both sides of every
       component binding must equal what the intent bound.

    An intent is a plan, never authority to write something else.

    Unlike promotion's resume, the re-plan here happens **inside** the held
    locks: `plan_dictionary_enrichment` reads canonical, the ledger and the
    reference store through the bound readers and takes neither of these two
    path locks, so there is no window between the classification, the decision
    and the write.
    """
    _bound_intent(config, prepared)
    with _paths_locked(Path(item.path) for item in prepared.components):
        verdicts = _precheck_components(config, prepared)
        if _intent_started(prepared, verdicts):
            _finished, ledger_error = _finish_intent_under_locks(prepared, verdicts)
            return _intent_commit(prepared, ledger_error)
        fresh = (
            decision
            if decision is not None
            else _redecide_for_intent(config, prepared, client=client)
        )
        _require_executable(fresh)
        _require_repository(config, fresh)
        _require_unmutated(fresh, None)
        _assert_intent_matches(config, prepared, fresh)
        return _commit_under_locks(fresh, at=prepared.enriched_at)


def recover_prepared_dictionary_enrichment(
    config: ProjectConfig,
    prepared: PreparedDictionaryEnrichment,
    *,
    client: enrich.DictionaryLookup | None = None,
) -> DictionaryEnrichmentRecovery:
    """Take §6's shared guard, then finish one interrupted dictionary pass."""
    from japanese_anki.application import study_curation

    with study_curation.curation_guard(config):
        return recover_prepared_dictionary_enrichment_under_guard(
            config, prepared, client=client
        )


def recover_prepared_dictionary_enrichment_under_guard(
    config: ProjectConfig,
    prepared: PreparedDictionaryEnrichment,
    *,
    client: enrich.DictionaryLookup | None = None,
) -> DictionaryEnrichmentRecovery:
    """Finish one interrupted dictionary pass from its intent alone.

    A **started** pass replays only the writes it still owes, from the frozen
    payloads and the frozen day — the same bytes on the day after the crash as
    on the day of it — and reports which of them had already landed.  A ledger
    replay that still cannot be written leaves the ``committed_ledger_incomplete``
    split, which blocks a job reaching ``complete`` on the same rule as
    promotion's.

    An **unstarted** pass is a different thing from an interrupted one: nothing
    it owns has been written, so there is no half-finished transaction whose
    classification a fresh decision could corrupt, and it goes back through the
    same re-decision a resumed apply performs.
    """
    _bound_intent(config, prepared)
    with _paths_locked(Path(item.path) for item in prepared.components):
        verdicts = _precheck_components(config, prepared)
        already = _already_complete(prepared, verdicts)
        if _intent_started(prepared, verdicts):
            finished, ledger_error = _finish_intent_under_locks(prepared, verdicts)
            return DictionaryEnrichmentRecovery(
                state=_intent_state(prepared, ledger_error),
                output_path=Path(prepared.output_path),
                already_complete=already,
                finished=finished,
                changed_record_ids=prepared.changed_record_ids,
                cleared_record_ids=prepared.cleared_record_ids,
                ledger_error=ledger_error,
            )
        fresh = _redecide_for_intent(config, prepared, client=client)
        _require_executable(fresh)
        _require_repository(config, fresh)
        _require_unmutated(fresh, None)
        _assert_intent_matches(config, prepared, fresh)
        commit = _commit_under_locks(fresh, at=prepared.enriched_at)
    finished = []
    if commit.state != "nothing":
        finished.append("canonical")
        if commit.changed_record_ids and commit.ledger_error is None:
            finished.append("ledger")
    return DictionaryEnrichmentRecovery(
        state=commit.state,
        output_path=commit.output_path,
        already_complete=already,
        finished=tuple(finished),
        changed_record_ids=commit.changed_record_ids,
        cleared_record_ids=commit.cleared_record_ids,
        ledger_error=commit.ledger_error,
    )
