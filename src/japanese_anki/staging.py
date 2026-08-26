"""Staging files: records a human still has to look at before they are real.

A staging file is ordinary YAML — a mapping with a ``records:`` list (exactly
the shape :func:`japanese_anki.io.load_records` already accepts, so
``janki validate`` works on it unchanged) plus metadata keys the record loader
ignores (``source_file``, ``extracted_at``, ``model``, ``review_notes``, and
the other keys in :data:`META_KEYS`).

Per-record review annotations (``hold_reason``, ``already_known``,
``suggested_reading``) live in that record's ``source.raw_fields`` as strings,
so a staged record round-trips through ``VocabularyRecord.from_dict`` with no
schema additions and no bespoke loader.

``data/staging/`` is tracked (see AGENTS.md "Data lifecycle"), so a reading
typed in by hand is recoverable once it is committed — but only then, and a
review is usually mid-flight when the next import runs. :func:`write_staging`
therefore refuses to overwrite an existing file unless the caller explicitly
passes ``force=True``. In-place annotation of a file under review (M2.6's
``--staging``) is the case that legitimately passes it.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import io
import json
import re
import sys
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import yaml
from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from japanese_anki.errors import JankiError
from japanese_anki.io import (
    DataError,
    atomic_write_text_bound,
    exclusive_path_lock,
    read_text_bound,
    validate_prefer_incoming,
)
from japanese_anki.models import EXAMPLE_AUTHORITY_KEY, VocabularyRecord


class StagingError(JankiError):
    pass


def _path_locked(function: Callable[..., Any]) -> Callable[..., Any]:
    """Run a complete staging read-modify-write pass under its path lock."""

    @functools.wraps(function)
    def locked(path: Path, *args: Any, **kwargs: Any) -> Any:
        with exclusive_path_lock(Path(path)):
            return function(path, *args, **kwargs)

    return locked


# Review annotations, stored stringified in ``source.raw_fields``.
ANNOTATION_KEYS: tuple[str, ...] = ("hold_reason", "already_known", "suggested_reading")

# The values ``hold_reason`` takes. They live here rather than in the module
# that writes them because a second module has to *read* them: janki's reading
# assistant offers a reading for a held row, and one of these holds is not about
# the reading at all. Whoever tests the value needs the vocabulary, and only one
# module can own it without a cycle.
HOLD_MISSING_READING = "missing reading"
HOLD_READING_KANJI = "reading contains kanji"
HOLD_UNKNOWN_READING = "reading not in the dictionary"

#: A row whose id would change, on a run that cannot see the whole collection.
#: The odd one out: the reading is fine and it is the *id* that could not be
#: checked. Promoting it under the id it arrived with would write an id nothing
#: can repair — ``promote.remint`` is the only thing that fixes a stored id, and
#: a stored id is exempt from it — so the row waits in ``data/staging/``, which
#: is committed, until a run can prove the id is free.
HOLD_UNVERIFIABLE_ID = "cannot check this id against the whole collection"

#: The holds that are *not* about the reading — a deny-list, not an allow-list,
#: and the direction matters. A staging file is hand-edited: a reviewer may type
#: ``hold_reason: check the okurigana`` into one, and the importers write their
#: reasons as bare literals that could drift from the constants above. Under an
#: allow-list every one of those would silently mean "not a reading hold", and
#: janki's reading assistant would report a file with held rows as having none —
#: while ``status --staged``, which reads the raw value, still lists them. So an
#: unrecognised reason means what a reason has always meant, and only the one
#: hold that is genuinely about something else is named here.
NON_READING_HOLDS: frozenset[str] = frozenset({HOLD_UNVERIFIABLE_ID})

# Metadata keys that sit beside ``records:``; the record loader ignores them.
# Enforced by :func:`write_staging` as a warning, not a refusal: `read_staging`
# hands back every non-``records`` key it finds, so a note a reviewer added by
# hand has to survive a round trip. What the warning catches is a *writer*
# inventing a key nothing downstream reads.
CANDIDATE_ACCOUNTING_KEY = "candidate_accounting"

#: The block an `enrich --ai` review writes beside its rows. Named once: three
#: modules ask whether a staging file carries it, and the question decides
#: whether its rows may be re-identified and whether they are evidence about
#: any source.
AI_ENRICHMENT_KEY = "ai_enrichment"

META_KEYS: tuple[str, ...] = (
    "source_file",
    "extracted_at",
    "model",
    "provider",
    "review_run_id",
    "review_notes",
    "coverage",
    "prompt_provenance",
    "pattern_set",
    "reviewed_pattern_set",
    AI_ENRICHMENT_KEY,
    "field_replacements",
    CANDIDATE_ACCOUNTING_KEY,
)

_RECORDS_KEY = "records"
_COVERAGE_KEY = "coverage"

#: Whether a staging file's rows are a model's answer about words janki
#: already held, rather than a reading of some page.
#:
#: `data/staging/done/` looks like one uniform record of what janki has read,
#: and it is not. Most archives are a model reading a source document, and
#: their rows are the best surviving statement of what that page taught. An
#: `enrich --ai` pass is not: it answers about the collection's own words, so
#: its rows are model output about janki's data with no page behind them.
#:
#: The distinction has already cost something. When `enrich --jpdb` overwrote
#: taught meanings with dictionary glosses, the repair read the archives to
#: recover what each source said — the right instinct, and it worked. But
#: `data/staging/done/ai-enrichment.yaml` snapshots ninety records *as they
#: stood while damaged*, and sixty-eight of its rows still carry the bad
#: glosses. A sweep that treated every archive alike would have restored the
#: damage it was written to undo.
#:
#: `tests/test_source_evidence.py` pins that file's classification against the
#: shipped corpus, so this stays a fact about the data rather than a warning
#: somebody has to read.
def is_model_pass(meta: Mapping[str, Any], *, collection_name: str = "") -> bool:
    """Whether these rows are a model's answer about words janki already held.

    Two signals, because the marker arrived later than the files. A modern
    `enrich --ai` review carries an `ai_enrichment` block. An older one
    carries nothing but its `source_file`, which names the collection rather
    than any document — and naming the collection as your source is what it
    means to be a pass over what janki already had.

    That second signal is narrowed by `model`, and the narrowing matters: a
    review somebody wrote *by hand* against the collection names it the same
    way, and calling that a paid model pass would refuse re-identification —
    the one repair its rows are most likely to need — with a sentence that is
    not true of it. No importer writes `model`, and no hand edit does either.

    `collection_name` is the configured normalized file's name, not a literal:
    a project may call its collection anything, and a rule that only knew
    `vocabulary.json` would be right about this repository and wrong about the
    next one.
    """
    # Presence, not truthiness: a pass that recorded an empty block is still a
    # pass, and `promote`'s own gate already reads the key this way.
    if AI_ENRICHMENT_KEY in meta:
        return True
    named = str(meta.get("source_file") or "").strip()
    return bool(collection_name) and named == collection_name and bool(meta.get("model"))


#: Metadata written beside a large AI-enrichment review.  Ordinary extraction
#: staging has no such block: those rows fill holes and keep existing curation.
#: A rich AI answer may also propose replacing a non-empty meaning, so the file
#: has to carry proof of the exact old value the proposal was made against.
FIELD_REPLACEMENTS_KEY = "field_replacements"
FIELD_REPLACEMENTS_VERSION = 1

# Staging suffixes remain an explicit writer contract so review files are
# recognizable as YAML to people and ordinary tools as well as to janki.
STAGING_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COVERAGE_DISPOSITIONS = (
    "candidate_units",
    "duplicate_units",
    "non_vocabulary_units",
    "unreadable_units",
)
#: The one internal inconsistency a coverage block can still report. The other
#: five — missing, unexpected, context- and disposition-mismatched, and omitted
#: units — were differences against a human's approved inventory of the page,
#: and went with the oracle apparatus in M8.4. A repeated key needs nothing
#: outside the response to be wrong.
_COVERAGE_MISMATCHES = ("duplicate_keys",)
_COVERAGE_V1_REQUIRED = {
    "version",
    "status",
    "blocking",
    "source_fingerprint",
    "model_reported_unit_count",
    "observed_unit_count",
    "prose_candidate_count",
    "prose_coverage",
    "source_units",
    *_COVERAGE_DISPOSITIONS,
    *_COVERAGE_MISMATCHES,
}
_COVERAGE_V2_CANDIDATE_FIELDS = {
    "parsed_candidate_count",
    "canonical_record_count",
    "unusable_candidate_count",
    "duplicate_candidate_count",
    "collision_group_count",
    "candidate_accounting_fingerprint",
}
_UNIT_DISPOSITIONS = {"candidate", "duplicate", "non-vocabulary", "unreadable"}
#: Who may accept an unmeasured coverage block.
#:
#: ``repository-owner`` is a person deciding. ``model`` is `janki promote
#: --accept-coverage` recording that a model read the page and the record
#: together and found the record complete — the owner's decision moved up a
#: level, from "is this page accounted for" to "is a model allowed to answer
#: that". The distinction is kept in the data because a card promoted on a
#: model's word should say so forever.
COVERAGE_AUTHORITIES = frozenset({"repository-owner", "model"})
_SECTION = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_PROMPT_PROVENANCE_V2_FIELDS = {
    "source_sha256",
    "mode",
    "provider",
    "model",
    "response_schema_version",
    "system_prompt_fingerprint",
    "style_guide_fingerprint",
    "user_prompt_fingerprint",
}
_PROMPT_PROVENANCE_FIELDS = {
    *_PROMPT_PROVENANCE_V2_FIELDS,
    "response_schema_fingerprint",
    "request_fingerprint",
}


def new_review_run_id() -> str:
    """Mint the persisted identity of one newly written model-review artifact."""
    return str(uuid4())


def review_run_id(meta: Mapping[str, Any]) -> str | None:
    """Return a canonical UUIDv4 review-run id, allowing legacy absence."""
    if "review_run_id" not in meta:
        return None
    value = meta.get("review_run_id")
    if not isinstance(value, str):
        raise StagingError(
            "[review-run-id-invalid] review_run_id must be canonical lowercase "
            "UUIDv4 text"
        )
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise StagingError(
            "[review-run-id-invalid] review_run_id must be canonical lowercase "
            "UUIDv4 text"
        ) from exc
    if parsed.version != 4 or str(parsed) != value:
        raise StagingError(
            "[review-run-id-invalid] review_run_id must be canonical lowercase "
            "UUIDv4 text"
        )
    return value


def replacement_fingerprint(record: VocabularyRecord, field: str) -> str:
    """Bind one replaceable field's old wire value to its record and name.

    Including all three parts prevents a digest being moved between two words,
    or between two same-valued fields on one word.  The wire value comes from
    :meth:`VocabularyRecord.to_dict`, not directly from the dataclass, so nested
    examples are ordinary JSON mappings and the digest describes what is
    actually durable in ``vocabulary.json``.
    """
    try:
        (name,) = validate_prefer_incoming((str(field),))
    except JankiError as exc:
        raise StagingError(
            f"Cannot authorize staged replacement of {field!r}: {exc}"
        ) from exc
    value = record.to_dict()[name]
    encoded = json.dumps(
        {"record_id": record.id, "field": name, "old_value": value},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def field_replacement_block(
    records: Iterable[VocabularyRecord],
    changes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the metadata authorizing a reviewed staged replacement.

    ``changes`` is the enrichment result's existing shape: record id to field
    name to ``(old, proposed)``.  The proposed half is deliberately not part of
    the digest because the point of staging is that a reviewer may improve it.
    The old half must equal the supplied record, though: accepting a change map
    computed from a different collection revision would create authority for a
    value the model never saw.
    """
    by_id: dict[str, VocabularyRecord] = {}
    for record in records:
        if record.id in by_id:
            raise StagingError(
                f"Cannot fingerprint replacements: {record.id} appears more than "
                "once in the records."
            )
        by_id[record.id] = record

    bound: dict[str, dict[str, str]] = {}
    for record_id, field_changes in changes.items():
        key = str(record_id)
        record = by_id.get(key)
        if record is None:
            raise StagingError(
                f"Cannot fingerprint replacements for {key}: there is no old record "
                "with that id."
            )
        if not isinstance(field_changes, Mapping) or not field_changes:
            raise StagingError(
                f"Cannot fingerprint replacements for {key}: its changes must be "
                "a non-empty field mapping."
            )
        if any(not isinstance(name, str) for name in field_changes):
            raise StagingError(
                f"Cannot fingerprint replacements for {key}: field names must be text."
            )
        try:
            names = validate_prefer_incoming(str(name) for name in field_changes)
        except JankiError as exc:
            raise StagingError(
                f"Cannot fingerprint replacements for {key}: {exc}"
            ) from exc
        fingerprints: dict[str, str] = {}
        for name in names:
            change = field_changes[name]
            if not isinstance(change, (tuple, list)) or len(change) != 2:
                raise StagingError(
                    f"Cannot fingerprint replacement {key}.{name}: its change must "
                    "be an (old, proposed) pair."
                )
            old_value = getattr(record, name)
            if change[0] != old_value:
                raise StagingError(
                    f"Cannot fingerprint replacement {key}.{name}: the change says "
                    "its old value is different from the record supplied."
                )
            fingerprints[name] = replacement_fingerprint(record, name)
        bound[key] = fingerprints

    if not bound:
        raise StagingError(
            "A field-replacement block needs at least one record and field."
        )
    return {
        "version": FIELD_REPLACEMENTS_VERSION,
        "records": {record_id: bound[record_id] for record_id in sorted(bound)},
    }


def _replacement_records(meta: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Parse the exact field-replacement metadata shape, or return absent."""
    if FIELD_REPLACEMENTS_KEY not in meta:
        return None
    raw = meta[FIELD_REPLACEMENTS_KEY]
    if not isinstance(raw, Mapping) or set(raw) != {"version", "records"}:
        raise StagingError(
            "[field-replacements-invalid] field_replacements must contain exactly "
            "version and records"
        )
    version = raw.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != FIELD_REPLACEMENTS_VERSION
    ):
        raise StagingError(
            "[field-replacements-invalid] field_replacements version must be 1"
        )
    records = raw.get("records")
    if not isinstance(records, Mapping) or not records:
        raise StagingError(
            "[field-replacements-invalid] field_replacements.records must be a "
            "non-empty mapping"
        )
    for raw_id, raw_fields in records.items():
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise StagingError(
                "[field-replacements-invalid] every replacement record id must be "
                "non-empty text"
            )
        if not isinstance(raw_fields, Mapping) or not raw_fields:
            raise StagingError(
                f"[field-replacements-invalid] {raw_id} must name at least one field"
            )
        try:
            names = validate_prefer_incoming(str(name) for name in raw_fields)
        except JankiError as exc:
            raise StagingError(
                f"[field-replacements-invalid] {raw_id}: {exc}"
            ) from exc
        if set(names) != set(raw_fields):
            # A non-string mapping key can stringify to a valid field name.
            # Accepting it here would make the block mean something other than
            # what the YAML visibly says.
            raise StagingError(
                f"[field-replacements-invalid] {raw_id} field names must be text"
            )
        for name in names:
            digest = raw_fields[name]
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise StagingError(
                    f"[field-replacements-invalid] {raw_id}.{name} must be a SHA-256"
                )
    return records


def authorized_field_replacements(
    meta: Mapping[str, Any],
    current: Sequence[VocabularyRecord],
    incoming: Sequence[VocabularyRecord],
) -> dict[str, tuple[str, ...]]:
    """Fields a staged row may replace after old-value or retry proof matches.

    Validation is deliberately a complete first pass.  A stale second row must
    refuse before a matching first row is merged, otherwise one reviewed file
    can land half of its replacements against a collection revision it did not
    describe.  Entries for rows absent from ``incoming`` are allowed: partial
    promotion leaves held rows in the same file and its top-level metadata is
    not pruned row by row.

    A field already equal to the exact staged wire value is the one retry case:
    it proves the records write completed before a ledger failure. A value that
    matches neither the bound old value nor the reviewed proposal stays stale.

    No block means no replacements.  That is the permanent compatibility rule
    for extraction schema-v2 staging and importer hold-backs, both of which
    predate this metadata and retain ordinary existing-wins merge behavior.
    """
    authorized, _already_landed = _replacement_authorization(
        meta, current, incoming
    )
    return authorized


def already_landed_field_replacements(
    meta: Mapping[str, Any],
    current: Sequence[VocabularyRecord],
    incoming: Sequence[VocabularyRecord],
) -> dict[str, tuple[str, ...]]:
    """Reviewed fields already equal to the proposal after a failed handoff.

    The old-value digest no longer matches after ``vocabulary.json`` lands but
    the ledger save fails.  Exact equality with the reviewed staging wire value
    proves that this proposal is already present and makes a local retry safe.
    The classifier still refuses a third value, so this recovery cannot turn a
    later human edit into replacement authority.
    """
    _authorized, already_landed = _replacement_authorization(
        meta, current, incoming
    )
    return already_landed


def _replacement_authorization(
    meta: Mapping[str, Any],
    current: Sequence[VocabularyRecord],
    incoming: Sequence[VocabularyRecord],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Classify fresh and demonstrably already-landed replacement fields."""
    records = _replacement_records(meta)
    if records is None:
        return {}, {}

    by_id: dict[str, VocabularyRecord] = {}
    for record in current:
        if record.id in by_id:
            raise StagingError(
                f"[field-replacements-invalid] current records contain duplicate "
                f"id {record.id}"
            )
        by_id[record.id] = record

    incoming_by_id = {record.id: record for record in incoming}
    incoming_ids = set(incoming_by_id)
    authorized: dict[str, tuple[str, ...]] = {}
    already_landed: dict[str, tuple[str, ...]] = {}
    stale: list[str] = []
    for record_id in incoming_ids:
        raw_fields = records.get(record_id)
        if raw_fields is None:
            continue
        old = by_id.get(record_id)
        if old is None:
            stale.append(f"{record_id} (no current record)")
            continue
        names = tuple(str(name) for name in raw_fields)
        authorized[record_id] = names
        landed: list[str] = []
        for name in names:
            expected = str(raw_fields[name])
            actual = replacement_fingerprint(old, name)
            if hmac.compare_digest(expected, actual):
                continue
            proposed = incoming_by_id[record_id]
            if old.to_dict()[name] == proposed.to_dict()[name]:
                landed.append(name)
                continue
            stale.append(f"{record_id}.{name}")
        if landed:
            already_landed[record_id] = tuple(landed)
    if stale:
        raise StagingError(
            "[field-replacements-stale] the collection changed after this review "
            "was staged: "
            + ", ".join(sorted(stale))
            + ". Nothing was promoted; regenerate the proposal against the current "
            "records."
        )
    return authorized, already_landed


def coverage_block_fingerprint(block: Mapping[str, Any]) -> str:
    """Fingerprint coverage facts without the fingerprint or owner approval."""
    payload = {
        str(key): value
        for key, value in block.items()
        if key not in {"coverage_block_fingerprint", "approval"}
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def coverage_acceptance_requirements(block: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact coverage facts an owner approval must repeat."""
    return {
        "accepted_dispositions": {
            key: block.get(key, []) for key in _COVERAGE_DISPOSITIONS
        },
        "accepted_mismatches": {
            key: block.get(key, []) for key in _COVERAGE_MISMATCHES
        },
        "unmeasured": block.get("status") == "unmeasured",
    }


def _positive_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _coverage_fact(
    value: Any,
    *,
    fields: set[str],
    where: str,
    disposition: str | None = None,
) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise StagingError(
            f"[coverage-block-invalid] {where} must have exactly: "
            + ", ".join(sorted(fields))
        )
    if not _positive_integer(value.get("page")) or not _positive_integer(
        value.get("ordinal")
    ):
        raise StagingError(
            f"[coverage-block-invalid] {where} page and ordinal must be positive integers"
        )
    section = value.get("section")
    if not isinstance(section, str) or not _SECTION.fullmatch(section):
        raise StagingError(
            f"[coverage-block-invalid] {where} section must be a lowercase slug"
        )
    if "context_fingerprint" in fields:
        item = value.get("context_fingerprint")
        if not isinstance(item, str) or not _SHA256.fullmatch(item):
            raise StagingError(
                f"[coverage-block-invalid] {where}.context_fingerprint must be SHA-256"
            )
    if "disposition" in fields:
        actual = value.get("disposition")
        if actual not in _UNIT_DISPOSITIONS or (
            disposition is not None and actual != disposition
        ):
            raise StagingError(
                f"[coverage-block-invalid] {where} has an invalid disposition"
            )


def _validate_coverage_block(block: Mapping[str, Any]) -> None:
    version = block.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in {1, 2}
    ):
        raise StagingError(
            "[coverage-block-invalid] coverage version must be integer 1 or 2"
        )
    schema_fields = (
        _COVERAGE_V1_REQUIRED | _COVERAGE_V2_CANDIDATE_FIELDS
        if version == 2
        else _COVERAGE_V1_REQUIRED
    )
    fields = set(block)
    allowed = schema_fields | {"coverage_block_fingerprint", "approval"}
    required = schema_fields | {"coverage_block_fingerprint"}
    if fields - allowed or required - fields:
        raise StagingError(
            f"[coverage-block-invalid] coverage fields do not match schema version {version}"
        )
    if not isinstance(block.get("blocking"), bool):
        raise StagingError("[coverage-block-invalid] coverage blocking must be boolean")
    count_fields = [
        "model_reported_unit_count",
        "observed_unit_count",
        "prose_candidate_count",
    ]
    if version == 2:
        count_fields.extend(
            sorted(
                _COVERAGE_V2_CANDIDATE_FIELDS
                - {"candidate_accounting_fingerprint"}
            )
        )
        accounting_fingerprint = block.get("candidate_accounting_fingerprint")
        if not isinstance(accounting_fingerprint, str) or not _SHA256.fullmatch(
            accounting_fingerprint
        ):
            raise StagingError(
                "[coverage-block-invalid] candidate_accounting_fingerprint must be SHA-256"
            )
    for name in count_fields:
        value = block.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StagingError(
                f"[coverage-block-invalid] coverage {name} must be a non-negative integer"
            )
    for name in (*_COVERAGE_DISPOSITIONS, *_COVERAGE_MISMATCHES, "source_units"):
        if not isinstance(block.get(name), list):
            raise StagingError(
                f"[coverage-block-invalid] coverage {name} must be a list"
            )
    if block.get("version") == 2 and block.get("parsed_candidate_count") != (
        block.get("prose_candidate_count") + len(block["candidate_units"])
    ):
        raise StagingError(
            "[candidate-accounting-stale] parsed_candidate_count must equal "
            "prose_candidate_count plus table candidate source units"
        )
    if block.get("observed_unit_count") != len(block["source_units"]):
        raise StagingError(
            "[coverage-block-invalid] observed_unit_count must match source_units"
        )
    if block.get("prose_coverage") not in {"unmeasured", "not-applicable"}:
        raise StagingError(
            "[coverage-block-invalid] prose_coverage must be unmeasured or not-applicable"
        )
    if block.get("prose_candidate_count") > 0 and block.get(
        "prose_coverage"
    ) != "unmeasured":
        raise StagingError(
            "[coverage-block-invalid] prose coverage does not match its candidate count"
        )

    fact_fields = {
        "page",
        "section",
        "ordinal",
        "context_fingerprint",
        "disposition",
    }
    for name in _COVERAGE_DISPOSITIONS:
        disposition = name.removesuffix("_units").replace("_", "-")
        for index, value in enumerate(block[name]):
            _coverage_fact(
                value,
                fields=fact_fields,
                where=f"coverage.{name}[{index}]",
                disposition=disposition,
            )
    key_fields = {"page", "section", "ordinal"}
    for index, value in enumerate(block["duplicate_keys"]):
        _coverage_fact(
            value, fields=key_fields, where=f"coverage.duplicate_keys[{index}]"
        )
    source_fields = fact_fields | {"context"}
    for index, value in enumerate(block["source_units"]):
        if not isinstance(value, Mapping):
            raise StagingError(
                f"[coverage-block-invalid] coverage.source_units[{index}] must be a mapping"
            )
        allowed_source = source_fields | {"reason"}
        if not source_fields <= set(value) or set(value) - allowed_source:
            raise StagingError(
                f"[coverage-block-invalid] coverage.source_units[{index}] fields are invalid"
            )
        _coverage_fact(
            {key: value[key] for key in fact_fields},
            fields=fact_fields,
            where=f"coverage.source_units[{index}]",
        )
        context = value.get("context")
        reason = value.get("reason", "")
        if not isinstance(context, str) or not context.strip():
            raise StagingError(
                f"[coverage-block-invalid] coverage.source_units[{index}] needs context"
            )
        if not isinstance(reason, str) or (
            value.get("disposition") != "candidate" and not reason.strip()
        ):
            raise StagingError(
                f"[coverage-block-invalid] coverage.source_units[{index}] needs a reason"
            )


def _validate_prompt_provenance(
    meta: Mapping[str, Any], block: Mapping[str, Any]
) -> None:
    provenance = meta.get("prompt_provenance")
    if not isinstance(provenance, Mapping):
        raise StagingError(
            "[prompt-provenance-invalid] an M7.4 coverage block needs the exact "
            "prompt provenance schema"
        )
    version = provenance.get("response_schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise StagingError(
            "[prompt-provenance-invalid] response schema version must be a positive integer"
        )
    expected = (
        _PROMPT_PROVENANCE_FIELDS
        if version >= 3
        else _PROMPT_PROVENANCE_V2_FIELDS
    )
    if set(provenance) != expected:
        raise StagingError(
            "[prompt-provenance-invalid] an M7.4 coverage block needs the exact "
            "prompt provenance schema"
        )
    if provenance.get("source_sha256") != block.get("source_fingerprint"):
        raise StagingError(
            "[prompt-provenance-stale] prompt provenance and coverage name different sources"
        )
    if provenance.get("mode") not in {"auto", "table", "prose"}:
        raise StagingError(
            "[prompt-provenance-invalid] extraction mode must be auto, table, or prose"
        )
    for name in ("provider", "model"):
        value = provenance.get(name)
        if not isinstance(value, str) or not value.strip():
            raise StagingError(
                f"[prompt-provenance-invalid] {name} must be non-empty text"
            )
    fingerprint_fields = [
        "system_prompt_fingerprint",
        "style_guide_fingerprint",
        "user_prompt_fingerprint",
    ]
    if version >= 3:
        fingerprint_fields.extend(
            ["response_schema_fingerprint", "request_fingerprint"]
        )
    for name in fingerprint_fields:
        value = provenance.get(name)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise StagingError(
                f"[prompt-provenance-invalid] {name} must be SHA-256"
            )
    if version >= 3:
        run_id = review_run_id(meta)
        if run_id is None:
            raise StagingError(
                "[review-run-id-invalid] a rich extraction needs its "
                "review_run_id"
            )
        pattern_set = meta.get("pattern_set")
        if not isinstance(pattern_set, Mapping):
            raise StagingError(
                "[prompt-provenance-invalid] a rich extraction needs its pattern "
                "answer beside the staged cards"
            )
        pattern_provenance = pattern_set.get("prompt_provenance")
        if not isinstance(pattern_provenance, Mapping):
            raise StagingError(
                "[prompt-provenance-invalid] the rich pattern answer needs exact "
                "prompt provenance"
            )
        if dict(pattern_provenance) != dict(provenance):
            raise StagingError(
                "[prompt-provenance-stale] the staged cards and pattern answer "
                "name different model requests"
            )
        try:
            pattern_run_id = review_run_id(pattern_set)
        except StagingError as exc:
            raise StagingError(
                "[review-run-id-invalid] pattern_set.review_run_id must be "
                "canonical lowercase UUIDv4 text"
            ) from exc
        if pattern_run_id is None:
            raise StagingError(
                "[review-run-id-invalid] a rich pattern answer needs its "
                "review_run_id"
            )
        if pattern_run_id != run_id:
            raise StagingError(
                "[review-run-id-invalid] the staged cards and pattern answer "
                "name different review runs"
            )

        reviewed_pattern_set = meta.get("reviewed_pattern_set")
        if reviewed_pattern_set is not None:
            if not isinstance(reviewed_pattern_set, Mapping):
                raise StagingError(
                    "[pattern-review-invalid] reviewed_pattern_set must be a mapping"
                )
            if reviewed_pattern_set.get("reviewed") is not True:
                raise StagingError(
                    "[pattern-review-invalid] reviewed_pattern_set must record "
                    "reviewed: true"
                )
            reviewed_provenance = reviewed_pattern_set.get("prompt_provenance")
            if not isinstance(reviewed_provenance, Mapping) or dict(
                reviewed_provenance
            ) != dict(provenance):
                raise StagingError(
                    "[prompt-provenance-stale] the reviewed pattern snapshot and "
                    "staged answer name different model requests"
                )
            try:
                reviewed_run_id = review_run_id(reviewed_pattern_set)
            except StagingError as exc:
                raise StagingError(
                    "[review-run-id-invalid] reviewed_pattern_set.review_run_id "
                    "must be canonical lowercase UUIDv4 text"
                ) from exc
            if reviewed_run_id != run_id:
                raise StagingError(
                    "[review-run-id-invalid] the reviewed pattern snapshot and "
                    "staged answer name different review runs"
                )


def validate_coverage_facts(
    meta: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, str, str] | None:
    """Everything about a coverage block except whether anyone accepted it.

    Returns the facts an acceptance check would need when one is required, or
    ``None`` when the block needs no acceptance at all (a prose-only
    extraction) or carries no coverage.

    Split out so a caller that is about to *buy* an acceptance can find out
    first whether the block is even valid. Running it afterwards meant a
    staging file with a stale fingerprint or malformed provenance — all of it
    detectable offline and for free — spent a paid model call, recorded an
    approval, and only then refused, leaving a file nobody could promote
    without hand-deleting the approval it had just written.
    """
    provenance = meta.get("prompt_provenance")
    schema_version = (
        provenance.get("response_schema_version")
        if isinstance(provenance, Mapping)
        else None
    )
    if "coverage" not in meta:
        if (
            isinstance(schema_version, int)
            and not isinstance(schema_version, bool)
            and schema_version >= 3
        ):
            raise StagingError(
                "[coverage-block-invalid] a rich extraction needs its coverage block"
            )
        return None
    block = meta["coverage"]
    if not isinstance(block, Mapping):
        raise StagingError("[coverage-block-invalid] coverage must be a mapping")
    _validate_coverage_block(block)
    coverage_version = block.get("version")
    if (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version >= 4
        and coverage_version != 2
    ):
        raise StagingError(
            "[candidate-accounting-invalid] response schema version 4 or newer "
            "needs coverage v2 and candidate_accounting"
        )
    if coverage_version == 2 and (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 4
    ):
        raise StagingError(
            "[candidate-accounting-invalid] coverage v2 belongs to response "
            "schema version 4 or newer"
        )
    if coverage_version == 2:
        accounting = meta.get(CANDIDATE_ACCOUNTING_KEY)
        if not isinstance(accounting, Mapping):
            raise StagingError(
                "[candidate-accounting-invalid] coverage v2 needs candidate_accounting"
            )
        if accounting.get("candidate_accounting_fingerprint") != block.get(
            "candidate_accounting_fingerprint"
        ):
            raise StagingError(
                "[candidate-accounting-stale] coverage and candidate_accounting "
                "fingerprints differ"
            )
        for name in _COVERAGE_V2_CANDIDATE_FIELDS - {
            "candidate_accounting_fingerprint"
        }:
            if accounting.get(name) != block.get(name):
                raise StagingError(
                    "[candidate-accounting-stale] coverage and "
                    f"candidate_accounting disagree on {name}"
                )
    _validate_prompt_provenance(meta, block)
    fingerprint = block.get("coverage_block_fingerprint")
    expected_fingerprint = coverage_block_fingerprint(block)
    if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
        raise StagingError(
            "[coverage-block-invalid] coverage has no valid block fingerprint"
        )
    if fingerprint != expected_fingerprint:
        raise StagingError(
            "[coverage-block-stale] coverage facts changed; the coverage-block "
            f"fingerprint must be {expected_fingerprint}"
        )
    source_fingerprint = block.get("source_fingerprint")
    if not isinstance(source_fingerprint, str) or not _SHA256.fullmatch(
        source_fingerprint
    ):
        raise StagingError(
            "[coverage-block-invalid] coverage has no valid source fingerprint"
        )
    status = block.get("status")
    # Two values, not four. `matched` and `mismatch` were verdicts against an
    # approved oracle, and no producer can reach them since M8.4 deleted it —
    # keeping them accepted here would let a hand-edited file claim a
    # measurement nothing performs.
    if status not in {"unmeasured", "selection"}:
        raise StagingError(
            "[coverage-block-invalid] coverage status must be unmeasured or selection"
        )
    should_block = status == "unmeasured"
    if block.get("blocking") is not should_block:
        raise StagingError(
            "[coverage-block-invalid] coverage blocking state does not match its status"
        )
    if not should_block:
        return None
    # The three values the acceptance check needs, already derived and already
    # validated here. Returned rather than recomputed so the two halves cannot
    # disagree about which block they are talking about.
    return block, source_fingerprint, fingerprint, status


def rich_extraction_review_run_id(meta: Mapping[str, Any]) -> str | None:
    """Return a validated rich extraction's run id, or ``None`` for v2/legacy."""
    provenance = meta.get("prompt_provenance")
    if not isinstance(provenance, Mapping):
        return None
    version = provenance.get("response_schema_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version < 3
    ):
        return None
    # Callers first pass through validate_coverage_facts, which requires this
    # for rich staging and binds the nested pattern answer to it.
    value = review_run_id(meta)
    if value is None:
        raise StagingError(
            "[review-run-id-invalid] a rich extraction needs its review_run_id"
        )
    return value


def require_resolved_coverage(meta: Mapping[str, Any]) -> None:
    """Refuse an unresolved M7.4 coverage block; allow legacy files."""
    pending = validate_coverage_facts(meta)
    if pending is not None:
        _require_acceptance(*pending)


def _require_acceptance(
    block: Mapping[str, Any],
    source_fingerprint: str,
    fingerprint: str,
    status: str,
) -> None:
    approval = block.get("approval")
    if not isinstance(approval, Mapping):
        raise StagingError(
            f"[coverage-unresolved] coverage is {status}; repository-owner approval "
            f"must name source {source_fingerprint} and coverage block {fingerprint}"
        )
    required = {
        "authority",
        "source_fingerprint",
        "coverage_block_fingerprint",
        "accepted_dispositions",
        "accepted_mismatches",
        "unmeasured",
        "reason",
        "approved_at",
    }
    authority = approval.get("authority")
    if authority not in COVERAGE_AUTHORITIES:
        raise StagingError(
            "[coverage-approval-invalid] coverage approval authority must be "
            + " or ".join(sorted(COVERAGE_AUTHORITIES))
        )
    # A model's approval carries who answered and what it was asked, because
    # those are the two things that make it reviewable later: the same page can
    # be accepted by one prompt and refused by the next, and a reader deciding
    # whether to trust this line needs to know which asking produced it. An
    # owner's approval carries neither — a person is their own provenance.
    if authority == "model":
        required |= {"model", "prompt_fingerprint"}
    if set(approval) != required:
        raise StagingError(
            "[coverage-approval-invalid] coverage approval fields do not match the "
            f"schema for a {authority} approval"
        )
    if authority == "model":
        for name in ("model", "prompt_fingerprint"):
            value = approval.get(name)
            if not isinstance(value, str) or not value.strip():
                raise StagingError(
                    f"[coverage-approval-invalid] a model approval needs {name}"
                )
    if (
        approval.get("source_fingerprint") != source_fingerprint
        or approval.get("coverage_block_fingerprint") != fingerprint
    ):
        raise StagingError(
            "[coverage-approval-stale] coverage approval does not match the source "
            "and coverage-block fingerprints"
        )
    requirements = coverage_acceptance_requirements(block)
    for key, value in requirements.items():
        if approval.get(key) != value:
            raise StagingError(
                f"[coverage-approval-stale] coverage approval must repeat {key} exactly"
            )
    reason = approval.get("reason")
    approved_at = approval.get("approved_at")
    if not isinstance(reason, str) or not reason.strip():
        raise StagingError(
            "[coverage-approval-invalid] coverage approval needs a non-empty reason"
        )
    if not isinstance(approved_at, str):
        raise StagingError(
            "[coverage-approval-invalid] coverage approval needs an ISO approval date"
        )
    try:
        from datetime import date

        if date.fromisoformat(approved_at).isoformat() != approved_at:
            raise ValueError
    except ValueError as exc:
        raise StagingError(
            "[coverage-approval-invalid] coverage approval date must use YYYY-MM-DD"
        ) from exc


def _stringify(value: Any) -> str:
    """Render an annotation value as a string ``raw_fields`` can hold."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def annotate(record: VocabularyRecord, **values: Any) -> VocabularyRecord:
    """Return a copy of ``record`` carrying review annotations.

    Values are stringified into ``source.raw_fields``; passing ``None`` removes
    an annotation (how a resolved hold is cleared). Only names in
    ``ANNOTATION_KEYS`` are accepted — raw_fields is otherwise the importer's
    verbatim source row and must not become a junk drawer.
    """
    raw_fields = dict(record.source.raw_fields)
    for key, value in values.items():
        if key not in ANNOTATION_KEYS:
            raise StagingError(
                f"Unknown staging annotation '{key}'. Valid: {', '.join(ANNOTATION_KEYS)}"
            )
        if value is None:
            raw_fields.pop(key, None)
        else:
            raw_fields[key] = _stringify(value)
    return replace(record, source=replace(record.source, raw_fields=raw_fields))


def annotations(record: VocabularyRecord) -> dict[str, str]:
    """The review annotations present on ``record``, in ``ANNOTATION_KEYS`` order."""
    raw_fields = record.source.raw_fields
    return {key: raw_fields[key] for key in ANNOTATION_KEYS if key in raw_fields}


def _write_staging_unlocked(
    path: Path,
    records: Iterable[VocabularyRecord],
    meta: Mapping[str, Any] | None = None,
    force: bool = False,
    *,
    expected_revision: str | None = None,
) -> Path:
    """Implementation shared by the ordinary and already-locked writers."""
    path = Path(path)
    review_run_id(meta or {})
    if path.suffix.lower() not in STAGING_SUFFIXES:
        raise StagingError(
            f"Staging review files use YAML names: {path} has the unsupported "
            f"suffix '{path.suffix}'. Use {' or '.join(STAGING_SUFFIXES)}."
        )
    if (path.exists() or path.is_symlink()) and not force:
        raise StagingError(
            f"Staging file already exists: {path}. It may hold review edits you have "
            "not committed; move it aside or re-run with force to overwrite."
        )
    if expected_revision is not None and not force:
        raise StagingError(
            "A staging revision can guard only an explicitly forced replacement"
        )

    payload: dict[str, Any] = {}
    for key, value in (meta or {}).items():
        name = str(key)
        if name == _RECORDS_KEY:
            raise StagingError(f"Staging metadata cannot use the reserved key '{_RECORDS_KEY}'")
        if name not in META_KEYS:
            print(
                f"warning: staging metadata key '{name}' in {path} is not one of "
                f"{', '.join(META_KEYS)}; it will be written and read back, but no "
                "janki command looks at it",
                file=sys.stderr,
            )
        payload[name] = value
    payload[_RECORDS_KEY] = [record.to_dict() for record in records]

    text = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=STAGING_YAML_WIDTH,
    )
    # Even a first write is a compare-and-swap against absence. A staging
    # target can appear after the preflight above, and the ordinary atomic
    # writer follows symlinks; neither may turn a paid answer into an
    # overwrite outside the staging directory. Forced writes without an exact
    # revision still use the bound writer so a final-seam symlink swap refuses.
    atomic_write_text_bound(
        path,
        text,
        expected_revision=expected_revision,
        expected_absent=not force,
    )
    return path


@_path_locked
def write_staging(
    path: Path,
    records: Iterable[VocabularyRecord],
    meta: Mapping[str, Any] | None = None,
    force: bool = False,
    *,
    expected_revision: str | None = None,
) -> Path:
    """Write ``records`` and ``meta`` to a staging file at ``path``.

    Refuses to overwrite an existing file unless ``force`` is true: the file on
    disk may hold hand-edited readings not yet committed. A forced caller that
    already showed exact bytes to a person may supply their SHA-256 revision,
    making the comparison and replacement one transaction. Also refuses a suffix
    :func:`read_staging` could not parse — the content is YAML whatever the name
    says, so any other suffix produces a file only this function can make sense
    of. A metadata key outside :data:`META_KEYS` is written, with a warning: it
    is readable but nothing downstream looks at it.
    """
    return _write_staging_unlocked(
        path,
        records,
        meta,
        force,
        expected_revision=expected_revision,
    )


def write_staging_under_lock(
    path: Path,
    records: Iterable[VocabularyRecord],
    meta: Mapping[str, Any] | None = None,
    force: bool = False,
    *,
    expected_revision: str | None = None,
) -> Path:
    """Write staging when the caller already holds this exact path's lock.

    This narrow seam exists for a transaction spanning a live staging path and
    its done archive. Calling :func:`write_staging` while holding the done lock
    would try to acquire that non-reentrant lock again and deadlock; calling an
    unlocked writer without the outer lock would reopen the data-loss race.
    ``expected_revision`` closes the final seam against an editor that ignores
    that advisory lock.
    """
    return _write_staging_unlocked(
        path,
        records,
        meta,
        force,
        expected_revision=expected_revision,
    )


#: Line width for every writer that touches a staging file — and the reason
#: there is a constant rather than two literals.
#:
#: These files are *created* by PyYAML and *edited* by ruamel. Both used to
#: wrap at 100, but they choose different break points, so re-emitting a long
#: plain scalar re-folded it and left a trailing space at the break. A bare
#: load-and-dump with nothing edited changed 54 lines of a real staging file,
#: which made `rewrite_staging`'s promise to leave the rest "byte-for-byte as
#: it was" untrue, and churned every promotion's diff.
#:
#: Folding is what the two disagree about, so neither folds. Long scalars go
#: on one line: less pretty in a terminal diff, and exactly stable, which is
#: what a file holding work that exists nowhere else needs. Keep both writers
#: on this constant.
STAGING_YAML_WIDTH = 1 << 30


def _parser() -> YAML:
    """The round-trip parser used to edit a file in place, configured once."""
    parser = YAML()  # round-trip mode: comments, order and quoting survive
    parser.preserve_quotes = True
    parser.allow_unicode = True
    parser.width = STAGING_YAML_WIDTH
    # Write `null` where PyYAML wrote `null`. Every staging file is *created*
    # by `write_staging` through PyYAML, which spells an absent value `null`;
    # ruamel's round-trip representer spells it as an empty scalar. The two
    # mean the same thing to a parser and nothing to a reader, but re-dumping
    # rewrote that line on *every* record in the file — so annotating one row
    # produced a diff touching rows nobody edited, and this function's own
    # promise to leave the rest "byte-for-byte as it was" was not kept. One
    # edited field is now one changed line.
    parser.representer.add_representer(
        type(None),
        lambda representer, data: representer.represent_scalar(
            "tag:yaml.org,2002:null", "null"
        ),
    )
    return parser


def _load_document_snapshot(path: Path) -> tuple[Any, str, str]:
    """A staging file as an editable round-trip document.

    Every ruamel failure becomes a :class:`StagingError` naming the path. Two
    are worth expecting: ordinary syntax errors, and a **duplicate key**, which
    PyYAML accepts silently (last value wins) and ruamel refuses. Refusing is
    the right answer for the writing path even though the reading path is
    lenient — dumping a document whose duplicate ruamel collapsed would delete
    one of the reviewer's two lines, and this function will not touch a file it
    cannot rewrite faithfully. :func:`check_rewritable` exists so a caller can
    find that out before it spends an API pass.
    """
    try:
        text = read_text_bound(Path(path))
    except FileNotFoundError as exc:
        raise StagingError(
            f"Could not read {path} for rewriting: the staging file no longer exists"
        ) from exc
    except JankiError as exc:
        raise StagingError(f"Could not read {path} for rewriting: {exc}") from exc
    document = _load_document_text(text, source=str(path))
    revision = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return document, text, revision


def _load_document(path: Path) -> Any:
    """Document-only convenience for a read-only rewritability check."""
    document, _text, _revision = _load_document_snapshot(path)
    return document


def _load_document_text(text: str, *, source: str) -> Any:
    """Parse one captured staging wire value for a surgical in-memory edit."""
    try:
        document = _parser().load(io.StringIO(text))
    except YAMLError as exc:
        raise StagingError(f"Could not read {source} for rewriting: {exc}") from exc
    if not isinstance(document, MutableMapping) or _RECORDS_KEY not in document:
        raise StagingError(f"Expected a staging mapping with a '{_RECORDS_KEY}:' list in {source}")
    return document


def check_rewritable(path: Path) -> None:
    """Raise :class:`StagingError` now if :func:`rewrite_staging` could not write.

    For callers that do expensive work — an API pass — between reading a
    staging file and writing it back. Failing at the end of that work would
    have spent it for nothing and would greet the user with an error after
    they had already confirmed the change.
    """
    _load_document(path)


def _apply_changes(
    target: MutableMapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    """Write only what actually changed into ``target``, recursing into mappings.

    Keys whose value is unchanged are never touched — not rewritten, and not
    *added* when the file left them out. That is what keeps a hand-written
    staging row from acquiring twenty empty schema fields the moment janki
    annotates it, and what leaves every key janki does not know about exactly
    where the reviewer put it.
    """
    for key, new_value in after.items():
        old_value = before.get(key)
        if old_value == new_value:
            continue
        current = target.get(key)
        if (
            isinstance(new_value, Mapping)
            and isinstance(old_value, Mapping)
            and isinstance(current, MutableMapping)
        ):
            _apply_changes(current, old_value, new_value)
            continue
        target[key] = new_value


def _authority_update(
    before: VocabularyRecord,
    after: VocabularyRecord,
    *,
    row: int,
    source: str,
) -> str | None:
    """Return one authority replacement, proving it is the row's only change."""
    if before == after:
        return None
    before_raw = dict(before.source.raw_fields)
    after_raw = dict(after.source.raw_fields)
    before_authority = before_raw.pop(EXAMPLE_AUTHORITY_KEY, None)
    after_authority = after_raw.pop(EXAMPLE_AUTHORITY_KEY, None)
    expected_source = replace(before.source, raw_fields=dict(after.source.raw_fields))
    if (
        before_raw != after_raw
        or replace(before, source=expected_source) != after
        or not isinstance(after_authority, str)
        or not after_authority
    ):
        raise StagingError(
            f"{source} record {row}: captured-wire review may change only "
            f"source.raw_fields.{EXAMPLE_AUTHORITY_KEY} to non-empty text"
        )
    return None if before_authority == after_authority else after_authority


def _block_mapping(value: Any, *, where: str, source: str) -> MutableMapping[str, Any]:
    """Require the generated block mapping shape the surgical writer understands."""
    if (
        not isinstance(value, MutableMapping)
        or not hasattr(value, "lc")
        or not hasattr(value, "fa")
    ):
        raise StagingError(f"{source}: {where} must be a block-style mapping")
    if value.fa.flow_style() is True:
        raise StagingError(f"{source}: {where} must be a block-style mapping")
    return value


def _line_indent(line: str) -> int:
    content = line.rstrip("\r\n")
    if "\t" in content[: len(content) - len(content.lstrip())]:
        return -1
    return len(content) - len(content.lstrip(" "))


def render_example_authority_updates(
    captured_text: str,
    records: Sequence[VocabularyRecord],
    *,
    source: str = "<captured staging file>",
) -> str:
    """Surgically render only selected rows' example-authority wire lines.

    The review panel shows one exact byte snapshot. Re-serializing that entire
    YAML document to add a fingerprint line churns folded metadata, comments,
    omitted IDs, and null spellings that are unrelated to the human decision.
    This helper instead locates generated block-style ``source.raw_fields``
    mappings structurally with ruamel, inserts or replaces only the authority
    scalar line, and then proves that the result parses to exactly ``records``
    with unchanged metadata. Ambiguous flow or duplicate-key documents refuse.
    """
    original, original_meta = read_staging_text(captured_text, source=source)
    if len(original) != len(records):
        raise StagingError(f"{source} holds {len(original)} row(s) but {len(records)} were given")
    document = _load_document_text(captured_text, source=source)
    raw_records = document[_RECORDS_KEY] or []
    if not isinstance(raw_records, list) or len(raw_records) != len(original):
        raise StagingError(f"{source}: records must be one block-style sequence")

    lines = captured_text.splitlines(keepends=True)
    operations: list[tuple[str, int, str]] = []
    for index, (before, after) in enumerate(zip(original, records, strict=True)):
        authority = _authority_update(
            before,
            after,
            row=index + 1,
            source=source,
        )
        if authority is None:
            continue
        record_node = _block_mapping(raw_records[index], where=f"record {index + 1}", source=source)
        source_node = _block_mapping(
            record_node.get("source"),
            where=f"record {index + 1}.source",
            source=source,
        )
        raw_node = _block_mapping(
            source_node.get("raw_fields"),
            where=f"record {index + 1}.source.raw_fields",
            source=source,
        )
        try:
            raw_key_line, raw_key_column = source_node.lc.key("raw_fields")
        except (KeyError, TypeError, ValueError) as exc:
            raise StagingError(
                f"{source}: record {index + 1}.source.raw_fields has no unambiguous block location"
            ) from exc
        child_column = raw_key_column + 2
        for key in raw_node:
            try:
                _key_line, key_column = raw_node.lc.key(key)
            except (KeyError, TypeError, ValueError) as exc:
                raise StagingError(
                    f"{source}: record {index + 1}.source.raw_fields has an "
                    "unsupported key location"
                ) from exc
            if key_column != child_column:
                raise StagingError(
                    f"{source}: record {index + 1}.source.raw_fields must use "
                    "generated block indentation"
                )

        if EXAMPLE_AUTHORITY_KEY in raw_node:
            line_number, key_column = raw_node.lc.key(EXAMPLE_AUTHORITY_KEY)
            value_line, _value_column = raw_node.lc.value(EXAMPLE_AUTHORITY_KEY)
            if key_column != child_column or value_line != line_number:
                raise StagingError(
                    f"{source}: record {index + 1} has an unsupported "
                    f"{EXAMPLE_AUTHORITY_KEY} wire shape"
                )
            line = lines[line_number]
            content = line.rstrip("\r\n")
            ending = line[len(content) :]
            scalar = re.fullmatch(
                rf"(?P<prefix>[ ]{{{child_column}}}{EXAMPLE_AUTHORITY_KEY}"
                r"[ ]*:[ ]*)(?P<quote>['\"]?)(?P<value>[0-9a-f, ]+)"
                r"(?P=quote)(?P<suffix>[ ]*(?:#.*)?)",
                content,
            )
            if scalar is None:
                raise StagingError(
                    f"{source}: record {index + 1} has an unsupported "
                    f"{EXAMPLE_AUTHORITY_KEY} wire shape"
                )
            replacement = (
                scalar["prefix"]
                + scalar["quote"]
                + authority
                + scalar["quote"]
                + scalar["suffix"]
                + ending
            )
            operations.append(("replace", line_number, replacement))
            continue

        insertion = raw_key_line + 1
        while insertion < len(lines):
            content = lines[insertion].rstrip("\r\n")
            if not content.strip():
                insertion += 1
                continue
            indent = _line_indent(lines[insertion])
            if indent < 0:
                raise StagingError(f"{source}: tab-indented YAML is not reviewable")
            if indent <= raw_key_column:
                break
            insertion += 1
        if insertion == 0 or not lines[insertion - 1].endswith(("\n", "\r")):
            raise StagingError(
                f"{source}: generated staging must end authority insertion rows with a newline"
            )
        newline = "\r\n" if lines[insertion - 1].endswith("\r\n") else "\n"
        operations.append(
            (
                "insert",
                insertion,
                " " * child_column + f"{EXAMPLE_AUTHORITY_KEY}: {authority}" + newline,
            )
        )

    for operation, line_number, replacement in sorted(
        operations, key=lambda item: item[1], reverse=True
    ):
        if operation == "replace":
            lines[line_number] = replacement
        else:
            lines.insert(line_number, replacement)
    rendered = "".join(lines)
    reparsed, reparsed_meta = read_staging_text(rendered, source=source)
    _load_document_text(rendered, source=source)
    if reparsed != list(records) or reparsed_meta != original_meta:
        raise StagingError(
            f"{source}: surgical example-authority update did not preserve the "
            "captured staging structure"
        )
    return rendered


def _rewrite_staging_unlocked(
    path: Path, records: Sequence[VocabularyRecord]
) -> Path:
    """Update an existing staging file in place, preserving what janki does not own.

    :func:`write_staging` renders a file from records, which is right when it is
    creating one and destructive when it is not: a staging file under review is
    the one place in this repository holding work that exists nowhere else, and
    a load-then-dump round trip through the record schema silently deletes a
    reviewer's YAML comments and any key the schema has no field for. This
    reads the document, writes back only the values that actually changed, and
    leaves the rest of the file — comments, key order, quoting, unknown keys —
    byte-for-byte as it was.

    ``records`` must be the list :func:`read_staging` returned for this file,
    in order, with the same length; rows are matched positionally. A caller that
    adds or removes rows is not annotating a review, it is writing a new file,
    and should say so with :func:`write_staging`.

    **Both sides of the diff come from the same parser**, which is not a detail.
    The document is edited through ruamel (YAML 1.2) while ``records`` came from
    :func:`read_staging` (PyYAML, YAML 1.1), and the two dialects disagree about
    real values a reviewer types: 1.1 reads ``yes`` as a boolean and ``12:30``
    as the sexagesimal integer 750, 1.2 reads both as the strings they look
    like. Diffing a ruamel-parsed baseline against PyYAML-derived records would
    call every such value "changed" and write janki's reading of it over the
    reviewer's line — the exact thing this function exists to prevent. So the
    baseline is re-read with :func:`read_staging` too: the diff is then between
    two same-dialect readings, and only a key janki genuinely changed is ever
    written.
    """
    path = Path(path)

    document, captured_text, revision = _load_document_snapshot(path)
    original, _meta = read_staging_text(captured_text, source=str(path))
    raw_records = document[_RECORDS_KEY] or []
    if not (len(raw_records) == len(original) == len(records)):
        raise StagingError(
            f"{path} holds {len(raw_records)} row(s) but {len(records)} were given. "
            "rewrite_staging annotates the rows already in a file; use write_staging "
            "to write a different set."
        )

    for raw, before, after in zip(raw_records, original, records, strict=True):
        if not isinstance(raw, MutableMapping):
            raise StagingError(
                f"Record in {path} must be a mapping, got {type(raw).__name__}"
            )
        _apply_changes(raw, before.to_dict(), after.to_dict())

    buffer = io.StringIO()
    _parser().dump(document, buffer)
    atomic_write_text_bound(
        path,
        buffer.getvalue(),
        expected_revision=revision,
    )
    return path


def render_staging_update(path: Path, records: Sequence[VocabularyRecord]) -> str:
    """The text :func:`rewrite_staging` would write, without writing it.

    The workbench needs the same round-trip edit but a different *write*: a
    compare-and-swap bound to the exact bytes the browser rendered, so an
    approval or edit from a stale page cannot land on top of someone else's.
    Splitting render from write lets both callers share one implementation of
    "change only what changed" instead of growing a second one.
    """
    document, captured_text, _revision = _load_document_snapshot(path)
    original, _meta = read_staging_text(captured_text, source=str(path))
    raw_records = document[_RECORDS_KEY] or []
    if not (len(raw_records) == len(original) == len(records)):
        raise StagingError(
            f"{path} holds {len(raw_records)} row(s) but {len(records)} were given. "
            "render_staging_update annotates the rows already in a file."
        )
    for raw, before, after in zip(raw_records, original, records, strict=True):
        if not isinstance(raw, MutableMapping):
            raise StagingError(
                f"Record in {path} must be a mapping, got {type(raw).__name__}"
            )
        _apply_changes(raw, before.to_dict(), after.to_dict())
    buffer = io.StringIO()
    _parser().dump(document, buffer)
    return buffer.getvalue()


@_path_locked
def rewrite_staging(path: Path, records: Sequence[VocabularyRecord]) -> Path:
    """Update rows while preserving review-only YAML under the path lock."""
    return _rewrite_staging_unlocked(path, records)


def rewrite_staging_under_lock(
    path: Path, records: Sequence[VocabularyRecord]
) -> Path:
    """Update rows when the caller already holds this exact path's lock."""
    return _rewrite_staging_unlocked(path, records)


def coverage_already_resolved(meta: Mapping[str, Any]) -> bool:
    """Whether this file's coverage question is already answered.

    Answered, not merely *asked*: `validate_coverage_facts` reports what needs
    acceptance whether or not an approval exists, so it cannot be used to
    decide whether to spend. This runs the whole gate — fingerprints, repeated
    facts, schema — and says only whether it passes.
    """
    try:
        require_resolved_coverage(meta)
    except JankiError:
        return False
    return True


@_path_locked
def record_coverage_approval(
    path: Path, approval: Mapping[str, Any], *, replace_existing: bool = False
) -> Path:
    """Write a coverage approval into a staging file, changing nothing else.

    Through the same round-trip :func:`rewrite_staging` uses, and for the same
    reason: a staging file under review holds work that exists nowhere else,
    and a load-then-dump through the record schema would silently drop the
    reviewer's comments and any key janki has no field for. This sets exactly
    one key — ``coverage.approval`` — and leaves every byte around it alone.

    Refuses a file that already carries one unless ``replace_existing`` says
    otherwise. An approval is a decision about a specific coverage block, and
    quietly replacing it would let a second run overwrite a person's recorded
    reasoning with a model's — so replacing it is something a caller has to
    ask for in as many words.
    """
    path = Path(path)
    document, _captured_text, revision = _load_document_snapshot(path)
    block = document.get(_COVERAGE_KEY)
    if not isinstance(block, MutableMapping):
        raise StagingError(f"{path} carries no coverage block to approve.")
    if "approval" in block and not replace_existing:
        raise StagingError(
            f"{path} already carries a coverage approval. Remove it first if you "
            "mean to replace the recorded decision."
        )
    block["approval"] = _plain(approval)

    buffer = io.StringIO()
    _parser().dump(document, buffer)
    atomic_write_text_bound(
        path,
        buffer.getvalue(),
        expected_revision=revision,
    )
    return path


def _plain(value: Any) -> Any:
    """Ordinary containers, and strings quoted so both dialects agree.

    The hazard `rewrite_staging` documents, arriving from the other side. This
    writes through ruamel (YAML 1.2) into a file `read_staging` reads back with
    PyYAML (YAML 1.1), and the two disagree about bare words: 1.1 reads ``no``
    as ``False``, ``on`` as ``True`` and ``12:30`` as the sexagesimal integer
    750. A reason of "no", or a section slug of ``on`` inside the repeated
    disposition lists — both of which the field rules permit — would be written
    bare and read back as something else, failing the very approval just
    recorded and wedging the file until someone deleted it by hand.

    So every string is emitted double-quoted. Round-tripping through both
    dialects is then the identity, whatever the text says.
    """
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, str):
        return DoubleQuotedScalarString(value)
    return value


def _prune_staging_unlocked(path: Path, keep: Sequence[bool]) -> int:
    """Drop rows from a staging file, keeping the surviving ones verbatim.

    ``keep`` is one flag per row, in file order. Rows flagged ``False`` are
    removed; the surviving rows keep their own keys, quoting and inline
    comments, and the file keeps its metadata and header comments — for the
    same reason :func:`rewrite_staging` exists: this file holds a review, and
    re-rendering it from records deletes the notes the reviewer wrote in it.

    **One kind of comment does not survive:** a comment written on its own line
    *between* two rows. YAML attaches it to the row above, so removing that row
    takes it along even when the comment was about the row below. Editing the
    comment structure to compensate means reaching into the parser's internals
    for a case it does not model, which is a worse trade than the loss — and
    the loss is strictly smaller than re-rendering the file, which would take
    every comment in it. Notes written *inside* a row survive, which is where
    a per-row note belongs anyway.

    Returns how many rows were removed. Removing every row leaves an empty
    ``records:`` list rather than deleting the file — whether an emptied review
    is finished or wants keeping is the caller's call, not this function's.
    """
    path = Path(path)
    document, _captured_text, revision = _load_document_snapshot(path)
    raw_records = document[_RECORDS_KEY] or []
    if len(raw_records) != len(keep):
        raise StagingError(
            f"{path} holds {len(raw_records)} row(s) but {len(keep)} flag(s) were "
            "given; prune_staging needs one flag per row, in file order."
        )
    survivors = [raw for raw, wanted in zip(raw_records, keep, strict=True) if wanted]
    removed = len(raw_records) - len(survivors)
    if not removed:
        return 0
    # Assigned by slice so ruamel keeps the sequence object — and with it the
    # comments attached to the rows that stay.
    raw_records[:] = survivors
    document[_RECORDS_KEY] = raw_records
    buffer = io.StringIO()
    _parser().dump(document, buffer)
    atomic_write_text_bound(
        path,
        buffer.getvalue(),
        expected_revision=revision,
    )
    return removed


#: The maps an `enrich --ai` review keys on record id.
#:
#: `promote` demands they name exactly the rows the file holds plus the rows
#: its archive holds — no more, no less. So a row cannot leave the file on its
#: own: dropping the record and keeping its provenance makes the entry
#: *unknown*, and the next promote refuses the whole file with every other
#: paid answer still in it.
_AI_PROVENANCE_MAPS = (
    (AI_ENRICHMENT_KEY, "request_fingerprints"),
    (AI_ENRICHMENT_KEY, "input_fingerprints"),
    (AI_ENRICHMENT_KEY, "fields"),
    ("field_replacements", "records"),
)


def _drop_provenance(document: Any, gone: Iterable[str]) -> None:
    """Remove departing record ids from every provenance map in place.

    Edited on the loaded document rather than rebuilt, so the review's own
    comments and key order survive — the same reason pruning assigns rows by
    slice.
    """
    departing = {str(record_id) for record_id in gone}
    if not departing:
        return
    for block_key, map_key in _AI_PROVENANCE_MAPS:
        block = document.get(block_key)
        if not isinstance(block, MutableMapping):
            continue
        values = block.get(map_key)
        if not isinstance(values, MutableMapping):
            continue
        for record_id in [key for key in values if str(key) in departing]:
            del values[record_id]


def render_staging_prune(path: Path, keep: Sequence[bool]) -> str | None:
    """The text :func:`prune_staging` would write, or None if nothing goes.

    The workbench needs the same pruning bound to a compare-and-swap, for the
    same reason the edit path does: the browser rendered a specific set of
    rows, and a removal aimed at that set must not land on a different one.

    Unlike :func:`prune_staging`, this also drops the departing rows from any
    AI provenance maps. The two are pruning for opposite reasons: promote
    prunes rows it has just *archived*, whose provenance must stay because the
    archive still holds them, while the workbench prunes a row somebody threw
    away, whose provenance would otherwise name a record nothing holds.
    """
    path = Path(path)
    document, _captured_text, _revision = _load_document_snapshot(path)
    raw_records = document[_RECORDS_KEY] or []
    if len(raw_records) != len(keep):
        raise StagingError(
            f"{path} holds {len(raw_records)} row(s) but {len(keep)} flag(s) were "
            "given; pruning needs one flag per row, in file order."
        )
    survivors = [raw for raw, wanted in zip(raw_records, keep, strict=True) if wanted]
    if len(survivors) == len(raw_records):
        return None
    gone = [
        str(raw.get("id", ""))
        for raw, wanted in zip(raw_records, keep, strict=True)
        if not wanted
    ]
    # Assigned by slice so ruamel keeps the sequence object — and with it the
    # comments attached to the rows that stay.
    raw_records[:] = survivors
    document[_RECORDS_KEY] = raw_records
    _drop_provenance(document, gone)
    buffer = io.StringIO()
    _parser().dump(document, buffer)
    return buffer.getvalue()


@_path_locked
def prune_staging(path: Path, keep: Sequence[bool]) -> int:
    """Drop selected rows under the path lock, preserving review-only YAML."""
    return _prune_staging_unlocked(path, keep)


def prune_staging_under_lock(path: Path, keep: Sequence[bool]) -> int:
    """Drop selected rows when the caller already holds this path's lock."""
    return _prune_staging_unlocked(path, keep)


def _read_staging_data(data: Any, *, source: str) -> tuple[list[VocabularyRecord], dict[str, Any]]:
    """Build records and metadata from one already-captured YAML value."""
    if not isinstance(data, Mapping):
        raise StagingError(
            f"Expected a staging mapping with a '{_RECORDS_KEY}:' list in {source}, "
            f"got {type(data).__name__}"
        )
    if _RECORDS_KEY not in data:
        raise StagingError(f"Staging file {source} has no '{_RECORDS_KEY}:' list")

    raw_records = data[_RECORDS_KEY] or []
    if not isinstance(raw_records, list):
        raise StagingError(f"'{_RECORDS_KEY}' in {source} must be a list of records")

    records: list[VocabularyRecord] = []
    for index, item in enumerate(raw_records, start=1):
        if not isinstance(item, Mapping):
            raise StagingError(
                f"Record {index} in {source} must be a mapping, got {type(item).__name__}"
            )
        records.append(VocabularyRecord.from_dict(dict(item)))

    meta = {str(key): value for key, value in data.items() if key != _RECORDS_KEY}
    review_run_id(meta)
    return records, meta


def read_staging_text(
    text: str, *, source: str = "<captured staging file>"
) -> tuple[list[VocabularyRecord], dict[str, Any]]:
    """Read records and metadata from captured staging text, without a path race."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DataError(f"Could not parse staging file {source}: {exc}") from exc
    return _read_staging_data(data, source=source)


def read_staging(path: Path) -> tuple[list[VocabularyRecord], dict[str, Any]]:
    """Read a staging file, returning ``(records, meta)``.

    ``meta`` is every top-level key except ``records``, so a hand-added note
    survives a read/write round trip.
    """
    path = Path(path)
    try:
        text = read_text_bound(path)
    except FileNotFoundError as exc:
        raise DataError(f"File not found: {path}") from exc
    return read_staging_text(text, source=str(path))
