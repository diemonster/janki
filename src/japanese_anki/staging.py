"""Staging files: records a human still has to look at before they are real.

A staging file is ordinary YAML — a mapping with a ``records:`` list (exactly
the shape :func:`japanese_anki.io.load_records` already accepts, so
``janki validate`` works on it unchanged) plus metadata keys the record loader
ignores (``source_file``, ``extracted_at``, ``model``, ``review_notes``).

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

import hashlib
import io
import json
import re
import sys
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml import YAML, YAMLError

from japanese_anki.errors import JankiError
from japanese_anki.io import atomic_write_text, load_structured
from japanese_anki.models import VocabularyRecord


class StagingError(JankiError):
    pass


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
META_KEYS: tuple[str, ...] = (
    "source_file",
    "extracted_at",
    "model",
    "review_notes",
    "coverage",
    "prompt_provenance",
)

_RECORDS_KEY = "records"

# ``read_staging`` parses by suffix (via ``load_structured``), so a staging file
# written under any other suffix would be write-only: the write succeeds and
# every read of it fails.
STAGING_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COVERAGE_DISPOSITIONS = (
    "candidate_units",
    "duplicate_units",
    "non_vocabulary_units",
    "unreadable_units",
)
_COVERAGE_MISMATCHES = (
    "missing_units",
    "unexpected_units",
    "duplicate_keys",
    "context_mismatched_units",
    "disposition_mismatched_units",
    "omission_units",
)
_COVERAGE_REQUIRED = {
    "version",
    "status",
    "blocking",
    "source_fingerprint",
    "oracle_id",
    "oracle_type",
    "oracle_content_fingerprint",
    "model_reported_unit_count",
    "observed_unit_count",
    "prose_candidate_count",
    "prose_coverage",
    "source_units",
    *_COVERAGE_DISPOSITIONS,
    *_COVERAGE_MISMATCHES,
}
_UNIT_DISPOSITIONS = {"candidate", "duplicate", "non-vocabulary", "unreadable"}
_SECTION = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


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
    for name in fields & {"context_fingerprint", "expected", "observed"}:
        item = value.get(name)
        # Disposition mismatch entries use disposition names in these two
        # fields. Context mismatch entries use SHA-256 values.
        if name in {"expected", "observed"} and item in _UNIT_DISPOSITIONS:
            continue
        if not isinstance(item, str) or not _SHA256.fullmatch(item):
            raise StagingError(
                f"[coverage-block-invalid] {where}.{name} must be SHA-256 or a disposition"
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
    fields = set(block)
    allowed = _COVERAGE_REQUIRED | {"coverage_block_fingerprint", "approval"}
    required = _COVERAGE_REQUIRED | {"coverage_block_fingerprint"}
    if fields - allowed or required - fields:
        raise StagingError(
            "[coverage-block-invalid] coverage fields do not match schema version 1"
        )
    version = block.get("version")
    if isinstance(version, bool) or version != 1:
        raise StagingError("[coverage-block-invalid] coverage version must be integer 1")
    if not isinstance(block.get("blocking"), bool):
        raise StagingError("[coverage-block-invalid] coverage blocking must be boolean")
    for name in (
        "model_reported_unit_count",
        "observed_unit_count",
        "prose_candidate_count",
    ):
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

    oracle_id = block.get("oracle_id")
    oracle_type = block.get("oracle_type")
    oracle_fingerprint = block.get("oracle_content_fingerprint")
    if oracle_id is None:
        if oracle_type is not None or oracle_fingerprint is not None:
            raise StagingError(
                "[coverage-block-invalid] coverage has partial oracle identity"
            )
    elif (
        not isinstance(oracle_id, str)
        or not _SECTION.fullmatch(oracle_id)
        or oracle_type not in {"exhaustive", "selection"}
        or not isinstance(oracle_fingerprint, str)
        or not _SHA256.fullmatch(oracle_fingerprint)
    ):
        raise StagingError("[coverage-block-invalid] coverage oracle identity is invalid")

    status = block.get("status")
    if status in {"matched", "mismatch"} and oracle_type != "exhaustive":
        raise StagingError(
            "[coverage-block-invalid] matched or mismatched coverage needs an exhaustive oracle"
        )
    if status == "selection" and oracle_type not in {None, "selection"}:
        raise StagingError(
            "[coverage-block-invalid] selection coverage cannot use an exhaustive oracle"
        )
    if status == "unmeasured" and oracle_type not in {None, "selection"}:
        raise StagingError(
            "[coverage-block-invalid] unmeasured coverage cannot use an exhaustive oracle"
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
    for name in ("missing_units", "unexpected_units"):
        for index, value in enumerate(block[name]):
            _coverage_fact(
                value, fields=fact_fields, where=f"coverage.{name}[{index}]"
            )
    key_fields = {"page", "section", "ordinal"}
    for name in ("duplicate_keys", "omission_units"):
        for index, value in enumerate(block[name]):
            _coverage_fact(
                value, fields=key_fields, where=f"coverage.{name}[{index}]"
            )
    mismatch_fields = {"page", "section", "ordinal", "expected", "observed"}
    for name in ("context_mismatched_units", "disposition_mismatched_units"):
        for index, value in enumerate(block[name]):
            _coverage_fact(
                value, fields=mismatch_fields, where=f"coverage.{name}[{index}]"
            )
            expected = value["expected"]
            observed = value["observed"]
            if name == "context_mismatched_units" and not (
                isinstance(expected, str)
                and _SHA256.fullmatch(expected)
                and isinstance(observed, str)
                and _SHA256.fullmatch(observed)
            ):
                raise StagingError(
                    f"[coverage-block-invalid] coverage.{name}[{index}] needs two SHA-256 values"
                )
            if name == "disposition_mismatched_units" and not (
                expected in _UNIT_DISPOSITIONS and observed in _UNIT_DISPOSITIONS
            ):
                raise StagingError(
                    f"[coverage-block-invalid] coverage.{name}[{index}] needs two dispositions"
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


def require_resolved_coverage(meta: Mapping[str, Any]) -> None:
    """Refuse an unresolved M7.4 coverage block; allow legacy files."""
    if "coverage" not in meta:
        return
    block = meta["coverage"]
    if not isinstance(block, Mapping):
        raise StagingError("[coverage-block-invalid] coverage must be a mapping")
    _validate_coverage_block(block)
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
    if status not in {"matched", "mismatch", "unmeasured", "selection"}:
        raise StagingError(
            "[coverage-block-invalid] coverage status must be matched, mismatch, "
            "unmeasured, or selection"
        )
    should_block = status in {"mismatch", "unmeasured"}
    if block.get("blocking") is not should_block:
        raise StagingError(
            "[coverage-block-invalid] coverage blocking state does not match its status"
        )
    if not should_block:
        return

    approval = block.get("approval")
    if not isinstance(approval, Mapping):
        raise StagingError(
            f"[coverage-unresolved] coverage is {status}; repository-owner approval "
            f"must name source {source_fingerprint} and coverage block {fingerprint}"
        )
    allowed = {
        "authority",
        "source_fingerprint",
        "coverage_block_fingerprint",
        "accepted_dispositions",
        "accepted_mismatches",
        "unmeasured",
        "reason",
        "approved_at",
    }
    if set(approval) != allowed:
        raise StagingError(
            "[coverage-approval-invalid] coverage approval fields do not match the "
            "required owner-approval schema"
        )
    if approval.get("authority") != "repository-owner":
        raise StagingError(
            "[coverage-approval-invalid] coverage approval authority must be "
            "repository-owner"
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


def write_staging(
    path: Path,
    records: Iterable[VocabularyRecord],
    meta: Mapping[str, Any] | None = None,
    force: bool = False,
) -> Path:
    """Write ``records`` and ``meta`` to a staging file at ``path``.

    Refuses to overwrite an existing file unless ``force`` is true: the file on
    disk may hold hand-edited readings not yet committed. Also refuses a suffix
    :func:`read_staging` could not parse — the content is YAML whatever the name
    says, so any other suffix produces a file only this function can make sense
    of. A metadata key outside :data:`META_KEYS` is written, with a warning: it
    is readable but nothing downstream looks at it.
    """
    path = Path(path)
    if path.suffix.lower() not in STAGING_SUFFIXES:
        raise StagingError(
            f"Staging files are YAML: {path} would be written as YAML under a "
            f"'{path.suffix}' name and read_staging parses by suffix, so nothing "
            f"could read it back. Use {' or '.join(STAGING_SUFFIXES)}."
        )
    if path.exists() and not force:
        raise StagingError(
            f"Staging file already exists: {path}. It may hold review edits you have "
            "not committed; move it aside or re-run with force to overwrite."
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
        width=100,
    )
    atomic_write_text(path, text)
    return path


def _parser() -> YAML:
    """The round-trip parser used to edit a file in place, configured once."""
    parser = YAML()  # round-trip mode: comments, order and quoting survive
    parser.preserve_quotes = True
    parser.allow_unicode = True
    parser.width = 100
    return parser


def _load_document(path: Path) -> Any:
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
        with Path(path).open(encoding="utf-8") as handle:
            document = _parser().load(handle)
    except YAMLError as exc:
        raise StagingError(f"Could not read {path} for rewriting: {exc}") from exc
    if not isinstance(document, MutableMapping) or _RECORDS_KEY not in document:
        raise StagingError(
            f"Expected a staging mapping with a '{_RECORDS_KEY}:' list in {path}"
        )
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


def rewrite_staging(path: Path, records: Sequence[VocabularyRecord]) -> Path:
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
    if not path.exists():
        raise StagingError(
            f"No staging file to update at {path}; write_staging creates one."
        )

    original, _meta = read_staging(path)
    document = _load_document(path)
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
    atomic_write_text(path, buffer.getvalue())
    return path


def prune_staging(path: Path, keep: Sequence[bool]) -> int:
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
    document = _load_document(path)
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
    atomic_write_text(path, buffer.getvalue())
    return removed


def read_staging(path: Path) -> tuple[list[VocabularyRecord], dict[str, Any]]:
    """Read a staging file, returning ``(records, meta)``.

    ``meta`` is every top-level key except ``records``, so a hand-added note
    survives a read/write round trip.
    """
    path = Path(path)
    data = load_structured(path)
    if not isinstance(data, Mapping):
        raise StagingError(
            f"Expected a staging mapping with a '{_RECORDS_KEY}:' list in {path}, "
            f"got {type(data).__name__}"
        )
    if _RECORDS_KEY not in data:
        raise StagingError(f"Staging file {path} has no '{_RECORDS_KEY}:' list")

    raw_records = data[_RECORDS_KEY] or []
    if not isinstance(raw_records, list):
        raise StagingError(f"'{_RECORDS_KEY}' in {path} must be a list of records")

    records: list[VocabularyRecord] = []
    for index, item in enumerate(raw_records, start=1):
        if not isinstance(item, Mapping):
            raise StagingError(
                f"Record {index} in {path} must be a mapping, got {type(item).__name__}"
            )
        records.append(VocabularyRecord.from_dict(dict(item)))

    meta = {str(key): value for key, value in data.items() if key != _RECORDS_KEY}
    return records, meta
