"""Stage exact paid-revision changes to existing canonical cards.

This service is the boundary between a captured model answer and content
review.  It does not call a provider, approve Japanese, or write canonical
records.  Its only mutation is creating one new review artifact under the
configured staging directory.  The artifact uses the same old-field
fingerprints as ``enrich --ai`` so the shared promotion transaction can later
refuse a proposal whose canonical input changed.

``stage_card_change_staging`` applies a *staging plan*.  It deliberately
does not mean "apply these values to vocabulary.json": only promotion may do
that, after the owner has reviewed the staged proposal.
"""

from __future__ import annotations

import copy
import hashlib
import json
import secrets
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from japanese_anki import staging, validation
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import (
    MERGEABLE_FIELDS,
    exclusive_path_lock,
    load_records_snapshot,
    validate_prefer_incoming,
)
from japanese_anki.models import VocabularyRecord

__all__ = [
    "CardChangeStagingError",
    "CardChangeStagingPlan",
    "CardChangeStagingResult",
    "CardChangeStagingView",
    "CardFieldChange",
    "CardRevisionProvenance",
    "plan_card_change_staging",
    "project_card_change_staging",
    "stage_card_change_staging",
    "stage_card_change_staging_under_lock",
]


class CardChangeStagingError(JankiError):
    """A paid card-revision proposal cannot be staged safely."""


_SHA256_LENGTH = 64
_TRANSPORT_ATTRIBUTION = {
    "anthropic-api": "anthropic",
    "claude-code": "anthropic",
}
_PROTECTED_FIELDS = ("id", "expression", "reading", "source", "tags")
_TEXT_FIELDS = frozenset(
    {
        "furigana",
        "romaji",
        "part_of_speech",
        "verb_group",
        "transitivity",
        "usage_notes",
        "audio",
        "image",
        "audio_accent",
    }
)
_TEXT_LIST_FIELDS = frozenset({"meanings", "pitch_accent"})
_EXAMPLE_FIELDS = frozenset(
    {
        "japanese",
        "furigana",
        "romaji",
        "english",
        "audio",
        "spoken_japanese",
        "register",
    }
)


@dataclass(frozen=True, slots=True)
class CardRevisionProvenance:
    """The already-captured ``revise`` request that authored this proposal.

    This value is evidence supplied by the caller, not authority to inspect or
    mutate the operation journal.  The operation id locates the durable
    revision manifest; the request fingerprint binds the exact paid request.
    """

    operation_id: str
    request_fingerprint: str
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class CardFieldChange:
    """One proposed replacement, in durable record-field order."""

    record_id: str
    expression: str
    field: str
    old_value: Any
    proposed_value: Any


@dataclass(frozen=True, slots=True)
class CardChangeStagingPlan:
    """Exact read-only plan for creating one unapproved staging artifact."""

    repository_root: Path
    canonical_path: Path
    staging_root: Path
    staging_path: Path
    current_records: tuple[VocabularyRecord, ...]
    proposed_records: tuple[VocabularyRecord, ...]
    changes: tuple[CardFieldChange, ...]
    input_fingerprints: tuple[tuple[str, str], ...]
    provenance: CardRevisionProvenance
    focus_resource_id: str | None
    fingerprint: str


@dataclass(frozen=True, slots=True)
class CardChangeStagingView:
    """Safe UI projection of one exact staging plan."""

    plan_fingerprint: str
    staging_path: Path
    record_ids: tuple[str, ...]
    changes: tuple[CardFieldChange, ...]
    operation_id: str
    request_fingerprint: str
    provider: str
    model: str
    focus_resource_id: str | None

    @property
    def changed_fields(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for change in self.changes:
            grouped.setdefault(change.record_id, []).append(change.field)
        return {record_id: tuple(fields) for record_id, fields in grouped.items()}


@dataclass(frozen=True, slots=True)
class CardChangeStagingResult:
    """The sole durable result of applying a staging plan."""

    staging_path: Path
    record_ids: tuple[str, ...]
    plan_fingerprint: str
    state: Literal["staged", "already_staged"]


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CardChangeStagingError(
            f"Card revision values must be JSON data: {exc}"
        ) from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_provenance(
    value: CardRevisionProvenance,
) -> CardRevisionProvenance:
    if not isinstance(value, CardRevisionProvenance):
        raise CardChangeStagingError(
            "Card revisions need typed operation provenance."
        )
    try:
        operation_id = str(uuid.UUID(value.operation_id))
    except (AttributeError, ValueError) as exc:
        raise CardChangeStagingError(
            "Card revision operation_id must be canonical UUID text."
        ) from exc
    parsed = uuid.UUID(operation_id)
    if parsed.version != 4 or operation_id != value.operation_id:
        raise CardChangeStagingError(
            "Card revision operation_id must be a canonical UUIDv4."
        )
    if not _is_sha256(value.request_fingerprint):
        raise CardChangeStagingError(
            "Card revision request_fingerprint must be a lowercase SHA-256."
        )
    provider = value.provider.strip() if isinstance(value.provider, str) else ""
    if provider not in _TRANSPORT_ATTRIBUTION:
        choices = ", ".join(sorted(_TRANSPORT_ATTRIBUTION))
        raise CardChangeStagingError(
            f"Unknown Card revision provider {value.provider!r}; expected {choices}."
        )
    model = value.model.strip() if isinstance(value.model, str) else ""
    if not model:
        raise CardChangeStagingError(
            "Card revision provenance needs a nonblank model."
        )
    return CardRevisionProvenance(
        operation_id=operation_id,
        request_fingerprint=value.request_fingerprint,
        provider=provider,
        model=model,
    )


def _focus_resource(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CardChangeStagingError(
            "A focus resource id must be nonblank text when supplied."
        )
    return value.strip()


def _detached_record(record: VocabularyRecord, *, label: str) -> VocabularyRecord:
    if not isinstance(record, VocabularyRecord):
        raise CardChangeStagingError(f"{label} must be a VocabularyRecord.")
    raw = record.to_dict()
    try:
        detached = VocabularyRecord.from_dict(copy.deepcopy(raw))
    except JankiError as exc:
        raise CardChangeStagingError(f"{label} is not structurally valid: {exc}") from exc
    if detached.to_dict() != raw:
        raise CardChangeStagingError(
            f"{label} contains values that do not round-trip as a VocabularyRecord."
        )
    return detached


def _exact_current_records(
    config: ProjectConfig,
    supplied: Sequence[VocabularyRecord],
) -> tuple[VocabularyRecord, ...]:
    if isinstance(supplied, (str, bytes)):
        raise CardChangeStagingError(
            "Card revisions need a nonempty sequence of current records."
        )
    current = tuple(
        _detached_record(record, label=f"Current record {index + 1}")
        for index, record in enumerate(supplied)
    )
    if not current:
        raise CardChangeStagingError(
            "Card revisions need at least one exact current record."
        )
    ids = [record.id for record in current]
    if len(ids) != len(set(ids)):
        raise CardChangeStagingError(
            "Card revision current records contain a duplicate id."
        )
    _require_canonical_records(config, current)
    return current


def _require_canonical_records(
    config: ProjectConfig,
    expected: Sequence[VocabularyRecord],
) -> None:
    try:
        canonical, _revision = load_records_snapshot(config.normalized_file.resolve())
    except JankiError as exc:
        raise CardChangeStagingError(
            f"Could not read the canonical records for a card revision: {exc}"
        ) from exc
    by_id: dict[str, VocabularyRecord] = {}
    duplicates: set[str] = set()
    for record in canonical:
        if record.id in by_id:
            duplicates.add(record.id)
        by_id[record.id] = record
    if duplicates:
        raise CardChangeStagingError(
            "Canonical records contain duplicate selected ids: "
            + ", ".join(sorted(duplicates))
        )
    for record in expected:
        held = by_id.get(record.id)
        if held is None:
            raise CardChangeStagingError(
                f"Canonical record {record.id!r} is missing; nothing was staged."
            )
        if held.to_dict() != record.to_dict():
            raise CardChangeStagingError(
                f"Canonical record {record.id!r} changed after this revision "
                "proposal was prepared; nothing was staged."
            )


def _proposals_from_complete_records(
    current: tuple[VocabularyRecord, ...],
    proposed: Sequence[VocabularyRecord],
) -> tuple[VocabularyRecord, ...]:
    if isinstance(proposed, (str, bytes)):
        raise CardChangeStagingError(
            "Complete Revision proposals must be a sequence of VocabularyRecord values."
        )
    detached = tuple(
        _detached_record(record, label=f"Proposed record {index + 1}")
        for index, record in enumerate(proposed)
    )
    by_id: dict[str, VocabularyRecord] = {}
    for record in detached:
        if record.id in by_id:
            raise CardChangeStagingError(
                f"Complete Revision proposals repeat record id {record.id!r}."
            )
        by_id[record.id] = record
    expected_ids = {record.id for record in current}
    if set(by_id) != expected_ids:
        raise CardChangeStagingError(
            "Complete Revision proposals must name exactly the supplied current records."
        )
    return tuple(by_id[record.id] for record in current)


def _proposals_from_field_values(
    current: tuple[VocabularyRecord, ...],
    replacements: Mapping[str, Mapping[str, Any]],
) -> tuple[VocabularyRecord, ...]:
    if not isinstance(replacements, Mapping) or not replacements:
        raise CardChangeStagingError(
            "Revision field replacements must be a nonempty record mapping."
        )
    if any(not isinstance(record_id, str) or not record_id for record_id in replacements):
        raise CardChangeStagingError(
            "Revision field-replacement record ids must be nonempty text."
        )
    expected_ids = {record.id for record in current}
    if set(replacements) != expected_ids:
        raise CardChangeStagingError(
            "Revision field replacements must name exactly the supplied current records."
        )

    proposed: list[VocabularyRecord] = []
    for record in current:
        fields = replacements[record.id]
        if not isinstance(fields, Mapping) or not fields:
            raise CardChangeStagingError(
                f"Revision field replacements for {record.id!r} must name at "
                "least one field."
            )
        raw = record.to_dict()
        for field, value in fields.items():
            if not isinstance(field, str):
                raise CardChangeStagingError(
                    f"Revision field names for {record.id!r} must be text."
                )
            if field in _PROTECTED_FIELDS:
                raise CardChangeStagingError(
                    f"Card revisions cannot change protected field {field!r}."
                )
            try:
                validate_prefer_incoming((field,))
            except JankiError as exc:
                raise CardChangeStagingError(
                    f"Card revisions cannot replace {field!r}: {exc}"
                ) from exc
            _validate_replacement_shape(record.id, field, value)
            raw[field] = copy.deepcopy(value)
        try:
            candidate = VocabularyRecord.from_dict(raw)
        except JankiError as exc:
            raise CardChangeStagingError(
                f"Proposed record {record.id!r} is not structurally valid: {exc}"
            ) from exc
        proposed.append(candidate)
    return tuple(proposed)


def _validate_replacement_shape(record_id: str, field: str, value: Any) -> None:
    """Refuse JSON types the canonical constructor would silently coerce."""

    valid = True
    if field in _TEXT_FIELDS:
        valid = isinstance(value, str)
    elif field in _TEXT_LIST_FIELDS:
        valid = isinstance(value, list) and all(
            isinstance(item, str) for item in value
        )
    elif field == "conjugations":
        valid = isinstance(value, Mapping) and all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        )
    elif field == "frequency_rank":
        valid = value is None or (
            isinstance(value, int) and not isinstance(value, bool)
        )
    elif field == "examples":
        valid = isinstance(value, list) and all(
            isinstance(item, Mapping)
            and set(item) == _EXAMPLE_FIELDS
            and all(isinstance(item[name], str) for name in _EXAMPLE_FIELDS)
            for item in value
        )
    elif field == "source_forms":
        # Removing the table is not a replacement, and a revision proposal is
        # the wrong boundary for it. `SourceFormsTable.from_dict` canonicalizes
        # both `null` and the empty table to absence, and `io.merge_records`
        # reads absence as a hole to keep the existing value in — so a proposal
        # spelling either one would stage, review as a deletion and then
        # promote to no change at all. Refused by name, at the shape check,
        # rather than accepted and quietly ignored downstream. The route the
        # refusal names is scoped on purpose: `--drop-table` rewrites a job's
        # staged copies, and `_require_canonical_records` has already proved
        # every record a revision can name is the canonical one, so it is not
        # the remediation for the table this proposal is about.
        if value is None or (
            isinstance(value, Mapping)
            and not value.get("columns")
            and not value.get("cells")
        ):
            raise CardChangeStagingError(
                f"Proposed record {record_id!r} would remove its "
                f"{field!r} table, which a card revision cannot express: a "
                "revision replaces a field's value, and an absent or empty "
                "table is the same wire as leaving it alone. A revision names "
                "canonical records, and a table already stored canonically is "
                "not a revision's to take off. `janki study curate JOB "
                "--record ID --drop-table` drops the table from a study job's "
                "staged copies, so it reaches this card only while its "
                "proposal is still unpromoted. Nothing was staged."
            )
        # The complete canonical shape, not "something `from_dict` will take":
        # this field is outside `enrich.ENRICHABLE_FIELDS` and `AI_FIELDS`, so
        # no dictionary or AI pass writes it and it is not on the automatic
        # repair allow-list either. A replacement that arrived as a bare cell
        # map would be accepted by the constructor as an empty table and
        # silently discard the source's declared columns.
        valid = (
            isinstance(value, Mapping)
            and set(value) == {"columns", "cells"}
            and isinstance(value["columns"], list)
            and all(
                isinstance(column, Mapping)
                and set(column) == {"id", "label"}
                and all(isinstance(column[name], str) for name in ("id", "label"))
                for column in value["columns"]
            )
            and isinstance(value["cells"], Mapping)
            and all(
                isinstance(key, str) and isinstance(item, str)
                for key, item in value["cells"].items()
            )
        )
    if not valid:
        raise CardChangeStagingError(
            f"Proposed record {record_id!r} has a replacement for {field!r} that "
            "does not have its complete canonical JSON field shape."
        )


def _changes(
    current: tuple[VocabularyRecord, ...],
    proposed: tuple[VocabularyRecord, ...],
) -> tuple[CardFieldChange, ...]:
    result: list[CardFieldChange] = []
    for old, new in zip(current, proposed, strict=True):
        for field in _PROTECTED_FIELDS:
            if old.to_dict()[field] != new.to_dict()[field]:
                raise CardChangeStagingError(
                    f"Card revisions cannot change protected field {field!r} "
                    f"on {old.id!r}."
                )
        old_wire = old.to_dict()
        new_wire = new.to_dict()
        for field in MERGEABLE_FIELDS:
            # `.get`, because the canonical serialization is deliberately
            # sparse for the optional fields — an absent `source_forms` key is
            # the absent table, exactly as an unset `spoken_japanese` is absent
            # from an example — and `MERGEABLE_FIELDS` is derived from the
            # dataclass rather than from one record's emitted keys.
            old_value = old_wire.get(field)
            new_value = new_wire.get(field)
            if old_value == new_value:
                continue
            result.append(
                CardFieldChange(
                    record_id=old.id,
                    expression=old.expression,
                    field=field,
                    old_value=copy.deepcopy(old_value),
                    proposed_value=copy.deepcopy(new_value),
                )
            )
    if not result:
        raise CardChangeStagingError(
            "The Revision proposal changes no canonical card fields; nothing was staged."
        )
    changed_ids = {change.record_id for change in result}
    missing = [record.id for record in current if record.id not in changed_ids]
    if missing:
        raise CardChangeStagingError(
            "Every supplied current record needs at least one proposed field change; "
            "unchanged: " + ", ".join(missing)
        )
    return tuple(result)


def _validate_proposed_records(records: tuple[VocabularyRecord, ...], source: Path) -> None:
    issues = validation.validate_records(list(records), source)
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        detail = "; ".join(issue.message for issue in errors)
        raise CardChangeStagingError(
            f"Revision proposal fails structural validation: {detail}"
        )


def _plan_payload(plan: CardChangeStagingPlan) -> dict[str, Any]:
    return {
        "version": 1,
        "repository_root": str(plan.repository_root),
        "canonical_path": str(plan.canonical_path),
        "staging_root": str(plan.staging_root),
        "staging_path": str(plan.staging_path),
        "current_records": [record.to_dict() for record in plan.current_records],
        "proposed_records": [record.to_dict() for record in plan.proposed_records],
        "changes": [
            {
                "record_id": change.record_id,
                "expression": change.expression,
                "field": change.field,
                "old_value": change.old_value,
                "proposed_value": change.proposed_value,
            }
            for change in plan.changes
        ],
        "input_fingerprints": dict(plan.input_fingerprints),
        "provenance": {
            "operation_id": plan.provenance.operation_id,
            "request_fingerprint": plan.provenance.request_fingerprint,
            "provider": plan.provenance.provider,
            "model": plan.provenance.model,
        },
        "focus_resource_id": plan.focus_resource_id,
    }


def _plan_fingerprint(plan: CardChangeStagingPlan) -> str:
    return _sha256_json(_plan_payload(plan))


def plan_card_change_staging(
    config: ProjectConfig,
    current_records: Sequence[VocabularyRecord],
    *,
    provenance: CardRevisionProvenance,
    proposed_records: Sequence[VocabularyRecord] | None = None,
    field_replacements: Mapping[str, Mapping[str, Any]] | None = None,
    focus_resource_id: str | None = None,
) -> CardChangeStagingPlan:
    """Plan one exact unapproved staging proposal without writing anything.

    Exactly one proposal shape is accepted.  Complete records make schema
    output convenient; field replacements let a tool return only the fields it
    was asked to change.  Both become the same complete staged records and the
    same old-field authority block.
    """

    if (proposed_records is None) == (field_replacements is None):
        raise CardChangeStagingError(
            "Supply exactly one of proposed_records or field_replacements."
        )
    checked_provenance = _validated_provenance(provenance)
    focused = _focus_resource(focus_resource_id)
    current = _exact_current_records(config, current_records)
    if proposed_records is not None:
        proposed = _proposals_from_complete_records(current, proposed_records)
    else:
        assert field_replacements is not None
        proposed = _proposals_from_field_values(current, field_replacements)
    changes = _changes(current, proposed)
    _validate_proposed_records(proposed, config.normalized_file.resolve())

    input_fingerprints = tuple(
        (record.id, _sha256_json({"record": record.to_dict()})) for record in current
    )
    repository_root = config.root.resolve()
    staging_root = config.staging_dir.resolve()
    staging_path = staging_root / (
        f"card-revision-{checked_provenance.operation_id}.yaml"
    )
    draft = CardChangeStagingPlan(
        repository_root=repository_root,
        canonical_path=config.normalized_file.resolve(),
        staging_root=staging_root,
        staging_path=staging_path,
        current_records=current,
        proposed_records=proposed,
        changes=changes,
        input_fingerprints=input_fingerprints,
        provenance=checked_provenance,
        focus_resource_id=focused,
        fingerprint="",
    )
    return replace(draft, fingerprint=_plan_fingerprint(draft))


def project_card_change_staging(
    plan: CardChangeStagingPlan,
) -> CardChangeStagingView:
    """Project a plan without exposing a writer or hidden canonical scope."""

    return CardChangeStagingView(
        plan_fingerprint=plan.fingerprint,
        staging_path=plan.staging_path,
        record_ids=tuple(record.id for record in plan.proposed_records),
        changes=plan.changes,
        operation_id=plan.provenance.operation_id,
        request_fingerprint=plan.provenance.request_fingerprint,
        provider=plan.provenance.provider,
        model=plan.provenance.model,
        focus_resource_id=plan.focus_resource_id,
    )


def _repository_matches(config: ProjectConfig, plan: CardChangeStagingPlan) -> bool:
    expected_staging = config.staging_dir.resolve() / (
        f"card-revision-{plan.provenance.operation_id}.yaml"
    )
    return (
        plan.repository_root == config.root.resolve()
        and plan.canonical_path == config.normalized_file.resolve()
        and plan.staging_root == config.staging_dir.resolve()
        and plan.staging_path == expected_staging
    )


def _validate_plan_semantics(plan: CardChangeStagingPlan) -> None:
    provenance = _validated_provenance(plan.provenance)
    if provenance != plan.provenance or _focus_resource(plan.focus_resource_id) != (
        plan.focus_resource_id
    ):
        raise CardChangeStagingError(
            "The Card revision plan contains non-canonical provenance."
        )
    current = tuple(
        _detached_record(record, label=f"Planned current record {index + 1}")
        for index, record in enumerate(plan.current_records)
    )
    proposed = _proposals_from_complete_records(current, plan.proposed_records)
    changes = _changes(current, proposed)
    if changes != plan.changes:
        raise CardChangeStagingError(
            "The Card revision plan's field diff does not match its records."
        )
    expected_inputs = tuple(
        (record.id, _sha256_json({"record": record.to_dict()})) for record in current
    )
    if expected_inputs != plan.input_fingerprints:
        raise CardChangeStagingError(
            "The Card revision plan's input fingerprints do not match its "
            "current records."
        )
    _validate_proposed_records(proposed, plan.canonical_path)


def _change_map(plan: CardChangeStagingPlan) -> dict[str, dict[str, tuple[Any, Any]]]:
    result: dict[str, dict[str, tuple[Any, Any]]] = {}
    for change in plan.changes:
        result.setdefault(change.record_id, {})[change.field] = (
            copy.deepcopy(change.old_value),
            copy.deepcopy(change.proposed_value),
        )
    return result


def _staging_meta(plan: CardChangeStagingPlan) -> dict[str, Any]:
    fields = project_card_change_staging(plan).changed_fields
    attribution_provider = _TRANSPORT_ATTRIBUTION[plan.provenance.provider]
    revision_provenance: dict[str, Any] = {
        "version": 1,
        "operation_id": plan.provenance.operation_id,
        "request_fingerprint": plan.provenance.request_fingerprint,
        "provider": plan.provenance.provider,
        "attribution_provider": attribution_provider,
        "model": plan.provenance.model,
        "input_fingerprints": dict(plan.input_fingerprints),
        "fields": {
            record.id: list(fields[record.id]) for record in plan.current_records
        },
    }
    if plan.focus_resource_id is not None:
        revision_provenance["focus_resource_id"] = plan.focus_resource_id
    return {
        "source_file": plan.canonical_path.name,
        "model": plan.provenance.model,
        "provider": plan.provenance.provider,
        # revision operation ids are canonical UUIDv4 values and one operation
        # authors one review artifact, so a second identity would add no truth.
        "review_run_id": plan.provenance.operation_id,
        staging.CARD_REVISION_KEY: revision_provenance,
        staging.FIELD_REPLACEMENTS_KEY: staging.field_replacement_block(
            plan.current_records,
            _change_map(plan),
        ),
        "review_notes": (
            f"{len(plan.current_records)} existing canonical record(s) were "
            "proposed for change by a revision operation. Review the exact "
            "field diff before promotion. Nothing in this file is approved."
        ),
    }


def _validated_stage(
    config: ProjectConfig,
    plan: CardChangeStagingPlan,
    *,
    expected_fingerprint: str,
) -> tuple[dict[str, Any], CardChangeStagingResult]:
    if not isinstance(plan, CardChangeStagingPlan):
        raise CardChangeStagingError("Expected a CardChangeStagingPlan.")
    if not _repository_matches(config, plan):
        raise CardChangeStagingError(
            "This Card revision plan belongs to different repository paths; "
            "nothing was staged."
        )
    actual_fingerprint = _plan_fingerprint(plan)
    if (
        not isinstance(expected_fingerprint, str)
        or not secrets.compare_digest(expected_fingerprint, plan.fingerprint)
        or not secrets.compare_digest(actual_fingerprint, plan.fingerprint)
    ):
        raise CardChangeStagingError(
            "The Card revision plan fingerprint is stale or was changed; "
            "nothing was staged."
        )
    _validate_plan_semantics(plan)
    expected_meta = _staging_meta(plan)
    result = CardChangeStagingResult(
        staging_path=plan.staging_path,
        record_ids=tuple(record.id for record in plan.proposed_records),
        plan_fingerprint=plan.fingerprint,
        state="staged",
    )
    return expected_meta, result


def _stage_under_lock(
    config: ProjectConfig,
    plan: CardChangeStagingPlan,
    expected_meta: Mapping[str, Any],
    result: CardChangeStagingResult,
) -> CardChangeStagingResult:
    if plan.staging_path.exists() or plan.staging_path.is_symlink():
        try:
            existing_records, existing_meta = staging.read_staging(plan.staging_path)
        except (JankiError, OSError, UnicodeError) as exc:
            raise CardChangeStagingError(
                "Card revision staging target already exists but "
                f"is not this exact review: {exc}"
            ) from exc
        if (
            existing_records == list(plan.proposed_records)
            and existing_meta == expected_meta
        ):
            return replace(result, state="already_staged")
        raise CardChangeStagingError(
            "Card revision staging target already exists with different content; "
            "it may contain owner review work and was not overwritten."
        )

    _require_canonical_records(config, plan.current_records)
    staging.write_staging_under_lock(
        plan.staging_path,
        plan.proposed_records,
        expected_meta,
        force=False,
        expected_absent=True,
    )
    return result


def stage_card_change_staging_under_lock(
    config: ProjectConfig,
    plan: CardChangeStagingPlan,
    *,
    expected_fingerprint: str,
) -> CardChangeStagingResult:
    """Stage while the caller holds ``plan.staging_path``'s application lock."""

    expected_meta, result = _validated_stage(
        config,
        plan,
        expected_fingerprint=expected_fingerprint,
    )
    try:
        return _stage_under_lock(config, plan, expected_meta, result)
    except CardChangeStagingError:
        raise
    except (JankiError, OSError) as exc:
        raise CardChangeStagingError(
            f"Could not create the card revision review: {exc}"
        ) from exc


def stage_card_change_staging(
    config: ProjectConfig,
    plan: CardChangeStagingPlan,
    *,
    expected_fingerprint: str,
) -> CardChangeStagingResult:
    """Create the exact planned review file, and nothing canonical.

    The plan and selected canonical records are revalidated immediately before
    the staging CAS. Later promotion rechecks every changed field's old-value
    fingerprint, closing the remaining interval without this service gaining a
    second canonical merge implementation.
    """

    expected_meta, result = _validated_stage(
        config,
        plan,
        expected_fingerprint=expected_fingerprint,
    )
    try:
        with exclusive_path_lock(plan.staging_path):
            return _stage_under_lock(config, plan, expected_meta, result)
    except CardChangeStagingError:
        raise
    except (JankiError, OSError) as exc:
        raise CardChangeStagingError(
            f"Could not create the card revision review: {exc}"
        ) from exc
