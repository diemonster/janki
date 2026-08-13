"""Named, constrained repairs and field-scoped review proposals.

The registry gives callbacks immutable projections, never complete records or
paths. Direct apply is limited to derived fields in the M7.1 allowlist. A
protected-content change can only become a fingerprinted proposal for a person
to review.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import stat
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import yaml
from ruamel.yaml import YAML
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from japanese_anki import qc
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji
from japanese_anki.io import (
    atomic_write_text_bound,
    exclusive_path_lock,
    exclusive_path_locks,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.romaji import kana_to_romaji
from japanese_anki.validation import has_errors, validate_records

REPAIR_MODES = ("ingest-safe", "proposal-only", "revoked")
AUTOMATIC_FIELDS = frozenset(
    {
        "furigana",
        "romaji",
        "examples[*].furigana",
        "examples[*].romaji",
        "audio",
        "image",
        "frequency_rank",
    }
)
IDENTITY_FIELDS = frozenset({"id", "expression", "reading"})
PROPOSAL_KIND = "repair-proposals"
ARCHIVE_KIND = "repair-proposal-archive"
JOURNAL_MARKER = "janki-repair-proposal-transaction"
SCHEMA_VERSION = 1
ABSENT_REVISION = "absent"

_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_PATH = re.compile(r"(?P<field>[a-z_]+)(?:\[(?P<index>\d+)\]\.(?P<nested>[a-z_]+))?\Z")
_PATTERN = re.compile(r"(?P<field>[a-z_]+)(?:\[\*\]\.(?P<nested>[a-z_]+))?\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RepairError(JankiError):
    pass


class _StrictLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _StrictLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def bytes_fingerprint(value: bytes | str) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class RepairDeclaration:
    code: str
    version: str
    phase: str
    mode: str
    allowed_fields: tuple[str, ...]
    input_fields: tuple[str, ...]
    evidence: Mapping[str, Any]
    precondition: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    transformation: Callable[
        [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
    ]
    postcondition: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], bool
    ]
    provenance: str


@dataclass(frozen=True, slots=True)
class RepairChange:
    code: str
    version: str
    record_id: str
    field: str
    old: Any
    new: Any
    evidence: dict[str, Any]
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "version": self.version,
            "record_id": self.record_id,
            "field": self.field,
            "old": self.old,
            "new": self.new,
            "evidence": self.evidence,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class RepairPlan:
    path: str
    input_revision: str
    declarations: tuple[tuple[str, str, str], ...]
    changes: tuple[RepairChange, ...]
    intended_text: str
    intended_output_fingerprint: str
    plan_fingerprint: str
    records: tuple[VocabularyRecord, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "input_revision": self.input_revision,
            "repairs": [
                {"code": code, "version": version, "mode": mode}
                for code, version, mode in self.declarations
            ],
            "changes": [change.to_dict() for change in self.changes],
            "intended_output_fingerprint": self.intended_output_fingerprint,
            "repair_plan_fingerprint": self.plan_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class SafeDocument:
    root: Path
    normalized_file: Path
    staging_dir: Path
    path: Path
    relative_path: str
    text: str
    revision: str
    identity: tuple[int, int, int, int, int]
    raw: Any
    records: tuple[VocabularyRecord, ...]
    records_container: str


@dataclass(frozen=True, slots=True)
class ProposalReview:
    proposal_document: SafeDocument
    records_document: SafeDocument
    entries: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ProposalAcceptance:
    accepted: int
    remaining: int
    archive_path: Path


def _validate_field_pattern(value: str, where: str) -> None:
    match = _PATTERN.fullmatch(value)
    if match is None:
        raise RepairError(f"{where} has invalid field path {value!r}")
    record_fields = set(VocabularyRecord(id="", expression="").to_dict())
    field_name = match.group("field")
    nested = match.group("nested")
    if field_name not in record_fields:
        raise RepairError(f"{where} has unknown record field {field_name!r}")
    if nested is None:
        return
    example_fields = {item.name for item in fields(ExampleSentence)}
    if field_name != "examples" or nested not in example_fields:
        raise RepairError(f"{where} has unknown nested field {value!r}")


class RepairRegistry:
    def __init__(self, declarations: Iterable[RepairDeclaration]) -> None:
        by_code: dict[str, RepairDeclaration] = {}
        for declaration in declarations:
            self._validate(declaration)
            if declaration.code in by_code:
                raise RepairError(f"Duplicate repair code {declaration.code!r}")
            by_code[declaration.code] = declaration
        self._by_code = MappingProxyType(dict(sorted(by_code.items())))

    @staticmethod
    def _validate(declaration: RepairDeclaration) -> None:
        if not _SLUG.fullmatch(declaration.code):
            raise RepairError("Repair code must be a lowercase slug")
        if not _VERSION.fullmatch(declaration.version):
            raise RepairError(f"Repair {declaration.code} has an invalid version")
        if declaration.mode not in REPAIR_MODES:
            raise RepairError(f"Repair {declaration.code} has an invalid mode")
        if not declaration.phase or not declaration.provenance.strip():
            raise RepairError(f"Repair {declaration.code} needs phase and provenance")
        if not declaration.allowed_fields or not declaration.input_fields:
            raise RepairError(f"Repair {declaration.code} needs declared fields")
        if len(set(declaration.allowed_fields)) != len(declaration.allowed_fields):
            raise RepairError(f"Repair {declaration.code} repeats an allowed field")
        if len(set(declaration.input_fields)) != len(declaration.input_fields):
            raise RepairError(f"Repair {declaration.code} repeats an input field")
        for name in (*declaration.allowed_fields, *declaration.input_fields):
            _validate_field_pattern(name, f"Repair {declaration.code}")
        if any(name in IDENTITY_FIELDS for name in declaration.allowed_fields):
            raise RepairError(f"Repair {declaration.code} cannot target identity")
        if declaration.mode == "ingest-safe" and not set(
            declaration.allowed_fields
        ) <= AUTOMATIC_FIELDS:
            raise RepairError(
                f"Repair {declaration.code} targets a field outside the automatic allowlist"
            )
        if not isinstance(declaration.evidence, Mapping):
            raise RepairError(f"Repair {declaration.code} evidence must be a mapping")
        for callback in (
            declaration.precondition,
            declaration.transformation,
            declaration.postcondition,
        ):
            if not callable(callback):
                raise RepairError(f"Repair {declaration.code} callback is not callable")

    def all(self) -> tuple[RepairDeclaration, ...]:
        return tuple(self._by_code.values())

    def select(self, codes: Sequence[str] = ()) -> tuple[RepairDeclaration, ...]:
        names = tuple(codes) if codes else tuple(self._by_code)
        if len(set(names)) != len(names):
            raise RepairError("Repair code list contains duplicates")
        unknown = [name for name in names if name not in self._by_code]
        if unknown:
            raise RepairError("Unknown repair code(s): " + ", ".join(unknown))
        return tuple(self._by_code[name] for name in names)

    def get(self, code: str) -> RepairDeclaration | None:
        return self._by_code.get(code)


def _path_matches(path: str, pattern: str) -> bool:
    path_match = _PATH.fullmatch(path)
    pattern_match = _PATTERN.fullmatch(pattern)
    if path_match is None or pattern_match is None:
        return False
    return (
        path_match.group("field") == pattern_match.group("field")
        and path_match.group("nested") == pattern_match.group("nested")
        and bool(path_match.group("index")) == bool(pattern_match.group("nested"))
    )


def _field_value(record: VocabularyRecord, path: str) -> Any:
    match = _PATH.fullmatch(path)
    if match is None:
        raise RepairError(f"Invalid record field path {path!r}")
    value = getattr(record, match.group("field"), None)
    if match.group("nested") is None:
        return copy.deepcopy(value)
    index = int(match.group("index"))
    if not isinstance(value, list) or index >= len(value):
        raise RepairError(f"Field path {path!r} is outside the record")
    return copy.deepcopy(getattr(value[index], match.group("nested")))


def _set_field(record: VocabularyRecord, path: str, value: Any) -> VocabularyRecord:
    match = _PATH.fullmatch(path)
    if match is None:
        raise RepairError(f"Invalid repair target {path!r}")
    field_name = match.group("field")
    if field_name in IDENTITY_FIELDS:
        raise RepairError(f"A repair cannot change identity field {field_name!r}")
    nested = match.group("nested")
    if nested is None:
        if not hasattr(record, field_name):
            raise RepairError(f"Unknown repair target {path!r}")
        return replace(record, **{field_name: copy.deepcopy(value)})
    items = list(getattr(record, field_name, []))
    index = int(match.group("index"))
    if index >= len(items) or not isinstance(items[index], ExampleSentence):
        raise RepairError(f"Repair target {path!r} is outside the record")
    items[index] = replace(items[index], **{nested: copy.deepcopy(value)})
    return replace(record, **{field_name: items})


def _projection(record: VocabularyRecord, fields: Sequence[str]) -> Mapping[str, Any]:
    values: dict[str, Any] = {}
    for pattern in fields:
        match = _PATTERN.fullmatch(pattern)
        if match is None:
            raise RepairError(f"Invalid declared input field {pattern!r}")
        nested = match.group("nested")
        if nested is None:
            values[pattern] = copy.deepcopy(getattr(record, match.group("field")))
        else:
            items = getattr(record, match.group("field"))
            values[pattern] = [copy.deepcopy(getattr(item, nested)) for item in items]
    return _freeze(values)


def _provenance_entry(
    declaration: RepairDeclaration, fields: Sequence[str]
) -> dict[str, Any]:
    return {
        "code": declaration.code,
        "version": declaration.version,
        "fields": sorted(fields),
        "evidence": copy.deepcopy(dict(declaration.evidence)),
        "provenance": declaration.provenance,
    }


def _annotate_repair(
    record: VocabularyRecord, declaration: RepairDeclaration, fields: Sequence[str]
) -> VocabularyRecord:
    raw_fields = dict(record.source.raw_fields)
    existing = raw_fields.get("janki_repairs", "")
    if existing:
        try:
            payload = json.loads(existing)
        except json.JSONDecodeError as exc:
            raise RepairError(
                f"{record.id}: source.raw_fields.janki_repairs is not valid JSON"
            ) from exc
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise RepairError(f"{record.id}: janki_repairs must be a JSON list")
    else:
        payload = []
    entry = _provenance_entry(declaration, fields)
    if entry not in payload:
        payload.append(entry)
    raw_fields["janki_repairs"] = _canonical(payload)
    return replace(record, source=replace(record.source, raw_fields=raw_fields))


def apply_declarations(
    records: Sequence[VocabularyRecord],
    declarations: Sequence[RepairDeclaration],
    *,
    modes: frozenset[str],
) -> tuple[list[VocabularyRecord], list[RepairChange]]:
    result = list(records)
    changes: list[RepairChange] = []
    for declaration in declarations:
        if declaration.mode == "revoked":
            raise RepairError(f"Repair {declaration.code}@{declaration.version} is revoked")
        if declaration.mode not in modes:
            raise RepairError(
                f"Repair {declaration.code}@{declaration.version} is "
                f"{declaration.mode}; this operation accepts {', '.join(sorted(modes))}"
            )
        evidence = _freeze(dict(declaration.evidence))
        for index, record in enumerate(result):
            before = _projection(record, declaration.input_fields)
            try:
                matched = declaration.precondition(before, evidence)
            except (KeyError, TypeError) as exc:
                raise RepairError(
                    f"Repair {declaration.code} precondition read or changed "
                    "undeclared input"
                ) from exc
            if not isinstance(matched, bool):
                raise RepairError(f"Repair {declaration.code} precondition must return bool")
            if not matched:
                continue
            try:
                planned_raw = declaration.transformation(before, evidence)
            except (KeyError, TypeError) as exc:
                raise RepairError(
                    f"Repair {declaration.code} transformation read or changed "
                    "undeclared input"
                ) from exc
            if not isinstance(planned_raw, Mapping) or not planned_raw:
                raise RepairError(
                    f"Repair {declaration.code} matched {record.id} but produced no diff"
                )
            planned = {str(path): copy.deepcopy(value) for path, value in planned_raw.items()}
            for path in planned:
                if not any(_path_matches(path, allowed) for allowed in declaration.allowed_fields):
                    raise RepairError(
                        f"Repair {declaration.code} wrote undeclared field {path!r}"
                    )
            try:
                valid_output = declaration.postcondition(
                    before, _freeze(planned), evidence
                )
            except (KeyError, TypeError) as exc:
                raise RepairError(
                    f"Repair {declaration.code} postcondition read or changed "
                    "undeclared input"
                ) from exc
            if not isinstance(valid_output, bool):
                raise RepairError(
                    f"Repair {declaration.code} postcondition must return bool"
                )
            if not valid_output:
                raise RepairError(
                    f"Repair {declaration.code} postcondition failed for {record.id}"
                )
            updated = record
            changed_fields: list[str] = []
            local: list[RepairChange] = []
            for path in sorted(planned):
                old = _field_value(updated, path)
                new = planned[path]
                if old == new:
                    continue
                updated = _set_field(updated, path, new)
                changed_fields.append(path)
                local.append(
                    RepairChange(
                        declaration.code,
                        declaration.version,
                        record.id,
                        path,
                        old,
                        copy.deepcopy(new),
                        copy.deepcopy(dict(declaration.evidence)),
                        declaration.provenance,
                    )
                )
            if not local:
                raise RepairError(
                    f"Repair {declaration.code} matched {record.id} but changed no field"
                )
            if updated.id != record.id:
                raise RepairError(f"Repair {declaration.code} changed record identity")
            updated = _annotate_repair(updated, declaration, changed_fields)
            result[index] = updated
            changes.extend(local)
    return result, changes


def apply_ingest_safe(
    records: Sequence[VocabularyRecord], registry: RepairRegistry | None = None
) -> tuple[list[VocabularyRecord], list[RepairChange]]:
    chosen = registry or REGISTRY
    declarations = tuple(
        declaration
        for declaration in chosen.all()
        if declaration.mode == "ingest-safe" and declaration.phase == "ingest"
    )
    return apply_declarations(records, declarations, modes=frozenset({"ingest-safe"}))


def _record_romaji_pre(before: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    del evidence
    reading = before["reading"]
    return (
        bool(reading)
        and not contains_kanji(reading)
        and before["romaji"] != kana_to_romaji(reading)
    )


def _record_romaji_transform(
    before: Mapping[str, Any], evidence: Mapping[str, Any]
) -> Mapping[str, Any]:
    del evidence
    return {"romaji": kana_to_romaji(before["reading"])}


def _record_romaji_post(
    before: Mapping[str, Any], planned: Mapping[str, Any], evidence: Mapping[str, Any]
) -> bool:
    del evidence
    return planned.get("romaji") == kana_to_romaji(before["reading"])


def _example_romaji_values(before: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for japanese, furigana, romaji in zip(
        before["examples[*].japanese"],
        before["examples[*].furigana"],
        before["examples[*].romaji"],
        strict=True,
    ):
        repaired = qc.regenerate_example_romaji(
            ExampleSentence(japanese=japanese, furigana=furigana, romaji=romaji)
        )
        values.append(repaired.romaji)
    return values


def _example_romaji_pre(before: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    del evidence
    return tuple(_example_romaji_values(before)) != before["examples[*].romaji"]


def _example_romaji_transform(
    before: Mapping[str, Any], evidence: Mapping[str, Any]
) -> Mapping[str, Any]:
    del evidence
    return {
        f"examples[{index}].romaji": value
        for index, value in enumerate(_example_romaji_values(before))
        if value != before["examples[*].romaji"][index]
    }


def _example_romaji_post(
    before: Mapping[str, Any], planned: Mapping[str, Any], evidence: Mapping[str, Any]
) -> bool:
    del evidence
    expected = _example_romaji_values(before)
    return all(
        planned.get(f"examples[{index}].romaji", old) == expected[index]
        for index, old in enumerate(before["examples[*].romaji"])
    )


def _punctuation_pre(before: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    del evidence
    return any(
        qc.repair_spilled_punctuation(value) != value
        for value in before["examples[*].furigana"]
    )


def _punctuation_transform(
    before: Mapping[str, Any], evidence: Mapping[str, Any]
) -> Mapping[str, Any]:
    del evidence
    return {
        f"examples[{index}].furigana": repaired
        for index, value in enumerate(before["examples[*].furigana"])
        if (repaired := qc.repair_spilled_punctuation(value)) != value
    }


def _punctuation_post(
    before: Mapping[str, Any], planned: Mapping[str, Any], evidence: Mapping[str, Any]
) -> bool:
    del evidence
    for index, old in enumerate(before["examples[*].furigana"]):
        value = planned.get(f"examples[{index}].furigana", old)
        if qc.repair_spilled_punctuation(value) != value:
            return False
    return True


REGISTRY = RepairRegistry(
    (
        RepairDeclaration(
            code="record-romaji-from-reading",
            version="1.0.0",
            phase="ingest",
            mode="ingest-safe",
            allowed_fields=("romaji",),
            input_fields=("reading", "romaji"),
            evidence={"algorithm": "japanese_anki.romaji.kana_to_romaji"},
            precondition=_record_romaji_pre,
            transformation=_record_romaji_transform,
            postcondition=_record_romaji_post,
            provenance="Derived Hepburn romaji from the stored kana reading.",
        ),
        RepairDeclaration(
            code="example-romaji-from-furigana",
            version="1.0.0",
            phase="ingest",
            mode="ingest-safe",
            allowed_fields=("examples[*].romaji",),
            input_fields=(
                "examples[*].japanese",
                "examples[*].furigana",
                "examples[*].romaji",
            ),
            evidence={"algorithm": "japanese_anki.qc.regenerate_example_romaji"},
            precondition=_example_romaji_pre,
            transformation=_example_romaji_transform,
            postcondition=_example_romaji_post,
            provenance="Derived example romaji from stored Japanese and furigana.",
        ),
        RepairDeclaration(
            code="example-furigana-punctuation-separator",
            version="1.0.0",
            phase="curated",
            mode="proposal-only",
            allowed_fields=("examples[*].furigana",),
            input_fields=("examples[*].furigana",),
            evidence={
                "algorithm": "japanese_anki.qc.repair_spilled_punctuation",
                "basis": "leading punctuation is not part of a ruby word",
            },
            precondition=_punctuation_pre,
            transformation=_punctuation_transform,
            postcondition=_punctuation_post,
            provenance=(
                "Proposed an Anki furigana separator before a ruby group that "
                "starts with punctuation."
            ),
        ),
    )
)


def _reject_symlink_components(path: Path, root: Path) -> None:
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise RepairError(f"Repair path escapes the repository: {path}") from exc
    current = root.absolute()
    for part in relative.parts:
        current = current / part
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RepairError(f"Could not inspect {current}: {exc.strerror or exc}") from exc
        if stat.S_ISLNK(details.st_mode):
            raise RepairError(f"Repair path uses symlink component: {current}")


def _under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _strict_data(text: str, path: Path) -> Any:
    try:
        if path.suffix.lower() == ".json":
            def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, value in pairs:
                    if key in result:
                        raise RepairError(f"{path} has duplicate key {key!r}")
                    result[key] = value
                return result

            return json.loads(text, object_pairs_hook=unique)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.load(text, Loader=_StrictLoader)
    except RepairError:
        raise
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise RepairError(f"Could not parse repair input {path}: {exc}") from exc
    raise RepairError(f"Repair input must be JSON or YAML: {path}")


def _validate_record_shape(raw: Any, where: str) -> None:
    if not isinstance(raw, Mapping):
        raise RepairError(f"{where} must be a record mapping")
    record_fields = set(VocabularyRecord(id="", expression="").to_dict())
    unknown = sorted(set(raw) - record_fields)
    if unknown:
        raise RepairError(f"{where} has unknown field(s): {', '.join(unknown)}")
    examples = raw.get("examples", []) or []
    if isinstance(examples, Mapping):
        examples = [examples]
    if not isinstance(examples, list):
        raise RepairError(f"{where}.examples must be a list")
    example_fields = {item.name for item in fields(ExampleSentence)}
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise RepairError(f"{where}.examples[{index}] must be a mapping")
        unknown_example = sorted(set(example) - example_fields)
        if unknown_example:
            raise RepairError(
                f"{where}.examples[{index}] has unknown field(s): "
                + ", ".join(unknown_example)
            )
    source = raw.get("source", {}) or {}
    if not isinstance(source, Mapping):
        raise RepairError(f"{where}.source must be a mapping")
    unknown_source = sorted(set(source) - {"type", "imported_from", "row", "raw_fields"})
    if unknown_source:
        raise RepairError(
            f"{where}.source has unknown field(s): {', '.join(unknown_source)}"
        )


def read_safe_document(
    root: Path,
    normalized_file: Path,
    staging_dir: Path,
    path: Path,
    *,
    allow_proposal: bool = False,
) -> SafeDocument:
    """Read a regular record file without following a symlink or identity swap."""
    root = root.resolve()
    requested = Path(path)
    if ".." in requested.parts:
        raise RepairError(f"Repair path cannot contain '..': {path}")
    candidate = requested if requested.is_absolute() else root / requested
    candidate = candidate.absolute()
    _reject_symlink_components(candidate, root)
    normalized_root = (root / "data" / "normalized").resolve()
    active_staging = staging_dir.resolve()
    resolved = Path(os.path.realpath(candidate))
    in_normalized = _under(resolved, normalized_root)
    in_staging = _under(resolved, active_staging)
    in_archive = _under(resolved, active_staging / "done")
    if not (in_normalized or (in_staging and not in_archive)):
        raise RepairError(
            f"Repair input must be under data/normalized or active staging: {path}"
        )
    try:
        initial_details = os.lstat(candidate)
    except OSError as exc:
        raise RepairError(f"Could not inspect repair input {path}: {exc}") from exc
    if not stat.S_ISREG(initial_details.st_mode):
        raise RepairError(f"Repair input is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise RepairError(f"Could not open repair input {path}: {exc.strerror or exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RepairError(f"Repair input is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_mode)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_mode)
    try:
        path_details = os.lstat(candidate)
    except OSError as exc:
        raise RepairError(f"Repair input changed while it was read: {path}") from exc
    if identity != after_identity or (path_details.st_dev, path_details.st_ino) != (
        before.st_dev,
        before.st_ino,
    ):
        raise RepairError(f"Repair input changed identity while it was read: {path}")
    raw_bytes = b"".join(chunks)
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError(f"Repair input is not UTF-8: {path}") from exc
    raw = _strict_data(text, candidate)
    if isinstance(raw, Mapping) and raw.get("kind") == PROPOSAL_KIND:
        if allow_proposal:
            return SafeDocument(
                root,
                normalized_file.resolve(),
                active_staging,
                resolved,
                resolved.relative_to(root).as_posix(),
                text,
                bytes_fingerprint(raw_bytes),
                identity,
                raw,
                (),
                "proposal",
            )
        raise RepairError(
            f"{path} is proposal-shaped. Use 'janki promote {path} --accept-proposals'."
        )
    if isinstance(raw, list):
        raw_records = raw
        container = "list"
    elif isinstance(raw, Mapping) and "records" in raw:
        raw_records = raw.get("records") or []
        container = "mapping"
    else:
        raise RepairError(f"Repair input is not record-shaped: {path}")
    if not isinstance(raw_records, list):
        raise RepairError(f"Records in {path} must be a list")
    records: list[VocabularyRecord] = []
    for index, item in enumerate(raw_records):
        _validate_record_shape(item, f"{path}:record {index + 1}")
        try:
            records.append(VocabularyRecord.from_dict(dict(item)))
        except JankiError as exc:
            raise RepairError(f"Could not read {path}:record {index + 1}: {exc}") from exc
    ids = [record.id for record in records]
    if len(set(ids)) != len(ids):
        raise RepairError(f"Repair input {path} has duplicate record IDs")
    return SafeDocument(
        root,
        normalized_file.resolve(),
        active_staging,
        resolved,
        resolved.relative_to(root).as_posix(),
        text,
        bytes_fingerprint(raw_bytes),
        identity,
        raw,
        tuple(records),
        container,
    )


def _patch_value(target: Any, before: Any, after: Any) -> Any:
    if before == after:
        return target
    if isinstance(target, Mapping) and isinstance(before, Mapping) and isinstance(after, Mapping):
        for key, value in after.items():
            if key in before and key in target:
                target[key] = _patch_value(target[key], before[key], value)
            else:
                target[key] = copy.deepcopy(value)
        for key in set(before) - set(after):
            target.pop(key, None)
        return target
    if (
        isinstance(target, list)
        and isinstance(before, list)
        and isinstance(after, list)
        and len(target) == len(before) == len(after)
    ):
        for index in range(len(target)):
            target[index] = _patch_value(target[index], before[index], after[index])
        return target
    return copy.deepcopy(after)


def render_records_document(document: SafeDocument, records: Sequence[VocabularyRecord]) -> str:
    payload = [record.to_dict() for record in records]
    if document.path.suffix.lower() == ".json":
        if document.records_container == "list":
            return (
                json.dumps(
                    sorted(payload, key=lambda item: item["id"]),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
        mapping = copy.deepcopy(dict(document.raw))
        mapping["records"] = sorted(payload, key=lambda item: item["id"])
        return json.dumps(mapping, ensure_ascii=False, indent=2) + "\n"
    parser = YAML()
    parser.preserve_quotes = True
    parser.allow_unicode = True
    parser.width = 100
    round_trip = parser.load(document.text)
    target_records = round_trip if document.records_container == "list" else round_trip["records"]
    if len(target_records) != len(document.records) or len(records) != len(document.records):
        raise RepairError("Repair rendering cannot add or remove records")
    for index, (before, after) in enumerate(zip(document.records, records, strict=True)):
        target_records[index] = _patch_value(
            target_records[index], before.to_dict(), after.to_dict()
        )
    buffer = io.StringIO()
    parser.dump(round_trip, buffer)
    return buffer.getvalue()


def _same_identity(path: Path, identity: tuple[int, int, int, int, int]) -> bool:
    try:
        details = os.lstat(path)
    except OSError:
        return False
    return (details.st_dev, details.st_ino) == identity[:2] and stat.S_ISREG(details.st_mode)


def write_safe_document(document: SafeDocument, text: str) -> None:
    with exclusive_path_lock(document.path):
        current = read_safe_document(
            document.root,
            document.normalized_file,
            document.staging_dir,
            document.path,
            allow_proposal=document.records_container == "proposal",
        )
        if current.revision != document.revision or not _same_identity(
            document.path, document.identity
        ):
            raise RepairError(
                f"Repair input {document.relative_path} changed after the plan was built"
            )
        atomic_write_text_bound(
            document.path,
            text,
            expected_revision=document.revision,
            expected_identity=document.identity[:2],
        )
        if bytes_fingerprint(document.path.read_bytes()) != bytes_fingerprint(text):
            raise RepairError(f"Could not verify repaired output {document.relative_path}")


def build_plan(
    document: SafeDocument,
    declarations: Sequence[RepairDeclaration],
) -> RepairPlan:
    updated, changes = apply_declarations(
        document.records, declarations, modes=frozenset({"ingest-safe"})
    )
    issues = validate_records(updated, document.relative_path)
    if has_errors(issues):
        codes = ", ".join(
            sorted(
                {
                    issue.code or "validation-error"
                    for issue in issues
                    if issue.level == "error"
                }
            )
        )
        raise RepairError(f"Repair plan fails full-file validation: {codes}")
    intended = render_records_document(document, updated)
    content = {
        "path": document.relative_path,
        "input_revision": document.revision,
        "repairs": [
            {
                "code": declaration.code,
                "version": declaration.version,
                "mode": declaration.mode,
            }
            for declaration in declarations
        ],
        "changes": [change.to_dict() for change in changes],
        "intended_output_fingerprint": bytes_fingerprint(intended),
    }
    plan_fingerprint = fingerprint(content)
    return RepairPlan(
        document.relative_path,
        document.revision,
        tuple((item.code, item.version, item.mode) for item in declarations),
        tuple(changes),
        intended,
        content["intended_output_fingerprint"],
        plan_fingerprint,
        tuple(updated),
    )


def build_check_plan(
    document: SafeDocument,
    declarations: Sequence[RepairDeclaration],
) -> RepairPlan:
    """Build one check-only plan that can include proposal-only repairs."""
    current = list(document.records)
    preview = list(document.records)
    changes: list[RepairChange] = []
    for declaration in declarations:
        if declaration.mode == "revoked":
            raise RepairError(
                f"Repair {declaration.code}@{declaration.version} is revoked"
            )
        updated, local = apply_declarations(
            current,
            [declaration],
            modes=frozenset({declaration.mode}),
        )
        changes.extend(local)
        if declaration.mode == "ingest-safe":
            current = updated
        preview, _preview_changes = apply_declarations(
            preview,
            [declaration],
            modes=frozenset({declaration.mode}),
        )
    issues = validate_records(preview, document.relative_path)
    if has_errors(issues):
        codes = ", ".join(
            sorted(
                {
                    issue.code or "validation-error"
                    for issue in issues
                    if issue.level == "error"
                }
            )
        )
        raise RepairError(f"Repair plan fails full-file validation: {codes}")
    intended = render_records_document(document, current)
    content = {
        "path": document.relative_path,
        "input_revision": document.revision,
        "repairs": [
            {
                "code": declaration.code,
                "version": declaration.version,
                "mode": declaration.mode,
            }
            for declaration in declarations
        ],
        "changes": [change.to_dict() for change in changes],
        "intended_output_fingerprint": bytes_fingerprint(intended),
    }
    return RepairPlan(
        document.relative_path,
        document.revision,
        tuple((item.code, item.version, item.mode) for item in declarations),
        tuple(changes),
        intended,
        content["intended_output_fingerprint"],
        fingerprint(content),
        tuple(current),
    )


def _expanded_basis_fields(
    record: VocabularyRecord,
    declaration: RepairDeclaration,
    target: str,
) -> tuple[str, ...]:
    target_match = _PATH.fullmatch(target)
    target_index = target_match.group("index") if target_match else None
    result: list[str] = []
    for pattern in declaration.input_fields:
        match = _PATTERN.fullmatch(pattern)
        if match is None:
            raise RepairError(f"Invalid basis field {pattern!r}")
        if match.group("nested") is None:
            result.append(match.group("field"))
            continue
        items = getattr(record, match.group("field"))
        if target_index is not None and target_match is not None and (
            target_match.group("field") == match.group("field")
        ):
            result.append(
                f"{match.group('field')}[{target_index}].{match.group('nested')}"
            )
        else:
            result.extend(
                f"{match.group('field')}[{index}].{match.group('nested')}"
                for index in range(len(items))
            )
    return tuple(dict.fromkeys(result))


def proposal_basis(
    record: VocabularyRecord,
    declaration: RepairDeclaration,
    target: str,
    old: Any,
) -> tuple[dict[str, Any], str]:
    fields_ = _expanded_basis_fields(record, declaration, target)
    basis = {name: _field_value(record, name) for name in fields_}
    payload = {
        "record_id": record.id,
        "target_field": target,
        "target_old": old,
        "basis": basis,
        "evidence": dict(declaration.evidence),
    }
    return basis, fingerprint(payload)


def proposal_entry(
    record: VocabularyRecord,
    declaration: RepairDeclaration,
    change: RepairChange,
) -> dict[str, Any]:
    basis, basis_fingerprint = proposal_basis(
        record, declaration, change.field, change.old
    )
    payload = {
        "code": declaration.code,
        "version": declaration.version,
        "record_id": record.id,
        "target_field": change.field,
        "old_value": change.old,
        "new_value": change.new,
        "basis_fields": list(basis),
        "basis": basis,
        "basis_fingerprint": basis_fingerprint,
        "evidence": copy.deepcopy(dict(declaration.evidence)),
        "provenance": declaration.provenance,
    }
    payload["proposal_entry_fingerprint"] = fingerprint(payload)
    return payload


def proposal_path(staging_dir: Path, source_relative: str) -> Path:
    name = Path(source_relative).name
    stamp = hashlib.sha256(source_relative.encode("utf-8")).hexdigest()[:10]
    return staging_dir / f"repair-proposals-{name}-{stamp}.yaml"


def create_proposals(
    document: SafeDocument,
    declarations: Sequence[RepairDeclaration],
    staging_dir: Path,
) -> tuple[Path, tuple[dict[str, Any], ...]]:
    if document.path != document.normalized_file.resolve():
        raise RepairError("Repair proposals can only target the normalized records file")
    changes: list[RepairChange] = []
    for declaration in declarations:
        updated, local_changes = apply_declarations(
            document.records,
            [declaration],
            modes=frozenset({"proposal-only"}),
        )
        issues = validate_records(updated, document.relative_path)
        if has_errors(issues):
            codes = ", ".join(
                sorted(
                    {
                        issue.code or "validation-error"
                        for issue in issues
                        if issue.level == "error"
                    }
                )
            )
            raise RepairError(f"Repair proposals fail full-file validation: {codes}")
        changes.extend(local_changes)
    by_id = {record.id: record for record in document.records}
    by_code = {declaration.code: declaration for declaration in declarations}
    entries = tuple(
        proposal_entry(by_id[change.record_id], by_code[change.code], change)
        for change in changes
    )
    keys = [(entry["record_id"], entry["target_field"]) for entry in entries]
    if len(set(keys)) != len(keys):
        raise RepairError("Proposal generation produced duplicate record/field entries")
    if not entries:
        raise RepairError("The selected proposal repairs produced no field changes")
    target = proposal_path(staging_dir.resolve(), document.relative_path)
    _reject_symlink_components(target.absolute(), document.root)
    if not _under(target.absolute(), document.staging_dir):
        raise RepairError("The derived proposal path is outside active staging")
    payload = {
        "version": SCHEMA_VERSION,
        "kind": PROPOSAL_KIND,
        "source_file": document.relative_path,
        "source_revision": document.revision,
        "created_at": date.today().isoformat(),
        "proposals": list(entries),
    }
    text = yaml.safe_dump(
        payload, allow_unicode=True, sort_keys=False, default_flow_style=False, width=100
    )
    with exclusive_path_lock(target):
        try:
            target_details = os.lstat(target)
        except FileNotFoundError:
            target_details = None
        except OSError as exc:
            raise RepairError(f"Could not inspect proposal target {target}: {exc}") from exc
        if target_details is not None:
            raise RepairError(
                f"Proposal file already exists: {target}. Review it before regenerating."
            )
        atomic_write_text_bound(target, text, expected_absent=True)
    return target, entries


_PROPOSAL_TOP = {
    "version",
    "kind",
    "source_file",
    "source_revision",
    "created_at",
    "proposals",
}
_ENTRY_REQUIRED = {
    "code",
    "version",
    "record_id",
    "target_field",
    "old_value",
    "new_value",
    "basis_fields",
    "basis",
    "basis_fingerprint",
    "evidence",
    "provenance",
    "proposal_entry_fingerprint",
}


def _relative_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise RepairError(f"{where} must be a repository-relative path")
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise RepairError(f"{where} must be a repository-relative path")
    return pure.as_posix()


def _sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RepairError(f"{where} must be SHA-256")
    return value


def _verify_entry_fingerprint(raw: Mapping[str, Any], where: str) -> None:
    content = {
        key: copy.deepcopy(raw[key])
        for key in _ENTRY_REQUIRED
        if key != "proposal_entry_fingerprint"
    }
    try:
        actual = fingerprint(content)
    except (TypeError, ValueError) as exc:
        raise RepairError(f"{where} has a value that is not canonical JSON") from exc
    if actual != raw["proposal_entry_fingerprint"]:
        raise RepairError(f"{where} proposal-entry fingerprint is stale")


def validate_proposal_payload(value: Any, where: str = "proposal file") -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PROPOSAL_TOP:
        raise RepairError(f"{where} does not match proposal schema version 1")
    data = dict(value)
    if data.get("version") != SCHEMA_VERSION or isinstance(data.get("version"), bool):
        raise RepairError(f"{where}.version must be integer 1")
    if data.get("kind") != PROPOSAL_KIND:
        raise RepairError(f"{where}.kind must be {PROPOSAL_KIND!r}")
    _relative_path(data.get("source_file"), f"{where}.source_file")
    _sha(data.get("source_revision"), f"{where}.source_revision")
    created_at = data.get("created_at")
    if isinstance(created_at, str):
        try:
            date.fromisoformat(created_at)
        except ValueError as exc:
            raise RepairError(f"{where}.created_at must be an ISO date") from exc
    elif not isinstance(created_at, date):
        raise RepairError(f"{where}.created_at must be an ISO date")
    raw_entries = data.get("proposals")
    if not isinstance(raw_entries, list):
        raise RepairError(f"{where}.proposals must be a list")
    keys: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_entries):
        entry_where = f"{where}.proposals[{index}]"
        if not isinstance(raw, Mapping):
            raise RepairError(f"{entry_where} must be a mapping")
        fields_ = set(raw)
        if not fields_ >= _ENTRY_REQUIRED or fields_ - (_ENTRY_REQUIRED | {"stale"}):
            raise RepairError(f"{entry_where} fields do not match the proposal schema")
        code = raw.get("code")
        version = raw.get("version")
        if not isinstance(code, str) or not _SLUG.fullmatch(code):
            raise RepairError(f"{entry_where}.code is invalid")
        if not isinstance(version, str) or not _VERSION.fullmatch(version):
            raise RepairError(f"{entry_where}.version is invalid")
        record_id = raw.get("record_id")
        target = raw.get("target_field")
        if not isinstance(record_id, str) or not record_id:
            raise RepairError(f"{entry_where}.record_id is invalid")
        if not isinstance(target, str) or _PATH.fullmatch(target) is None:
            raise RepairError(f"{entry_where}.target_field is invalid")
        key = (record_id, target)
        if key in keys:
            raise RepairError(f"{where} has duplicate proposal for {record_id}/{target}")
        keys.add(key)
        basis_fields = raw.get("basis_fields")
        basis = raw.get("basis")
        if (
            not isinstance(basis_fields, list)
            or not all(isinstance(item, str) for item in basis_fields)
            or len(set(basis_fields)) != len(basis_fields)
            or not isinstance(basis, Mapping)
            or list(basis) != basis_fields
        ):
            raise RepairError(f"{entry_where} basis fields are invalid")
        _sha(raw.get("basis_fingerprint"), f"{entry_where}.basis_fingerprint")
        _sha(
            raw.get("proposal_entry_fingerprint"),
            f"{entry_where}.proposal_entry_fingerprint",
        )
        if not isinstance(raw.get("evidence"), Mapping):
            raise RepairError(f"{entry_where}.evidence must be a mapping")
        if not isinstance(raw.get("provenance"), str) or not raw["provenance"].strip():
            raise RepairError(f"{entry_where}.provenance must be non-empty text")
        if "stale" in raw and (
            not isinstance(raw["stale"], str) or not raw["stale"].strip()
        ):
            raise RepairError(f"{entry_where}.stale must be non-empty text")
        _verify_entry_fingerprint(raw, entry_where)
    return data


def _proposal_parser() -> YAML:
    parser = YAML()
    parser.preserve_quotes = True
    parser.allow_unicode = True
    parser.width = 100
    return parser


def _render_proposal_changes(
    document: SafeDocument,
    *,
    accepted: frozenset[str] = frozenset(),
    stale: Mapping[str, str] | None = None,
) -> str:
    """Render proposal removals and stale marks without losing comments."""
    parser = _proposal_parser()
    try:
        payload = parser.load(document.text)
    except Exception as exc:  # ruamel has several scanner exception classes
        raise RepairError(f"Could not rewrite proposal file {document.path}: {exc}") from exc
    raw_entries = payload.get("proposals") if isinstance(payload, Mapping) else None
    if not isinstance(raw_entries, list):
        raise RepairError(f"Proposal file {document.path} has no proposals list")
    remaining = []
    stale = stale or {}
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise RepairError(f"Proposal file {document.path} has an invalid entry")
        entry_fingerprint = raw.get("proposal_entry_fingerprint")
        if entry_fingerprint in accepted:
            continue
        reason = stale.get(str(entry_fingerprint))
        if reason:
            raw["stale"] = reason
        remaining.append(raw)
    raw_entries[:] = remaining
    payload["proposals"] = raw_entries
    buffer = io.StringIO()
    parser.dump(payload, buffer)
    return buffer.getvalue()


def _entry_reason(
    entry: Mapping[str, Any],
    record: VocabularyRecord | None,
    registry: RepairRegistry,
) -> str | None:
    """Return why an entry is stale, or return ``None`` when it is current."""
    if "stale" in entry:
        return str(entry["stale"])
    if record is None:
        return "The target record no longer exists."
    declaration = registry.get(str(entry["code"]))
    if declaration is None:
        return "The repair declaration is no longer registered."
    if declaration.version != entry["version"]:
        return "The repair declaration version changed."
    if declaration.mode != "proposal-only":
        return f"The repair declaration mode is now {declaration.mode}."
    target = str(entry["target_field"])
    if target.split("[", 1)[0].split(".", 1)[0] in IDENTITY_FIELDS:
        return "A proposal cannot change an identity field."
    if not any(_path_matches(target, allowed) for allowed in declaration.allowed_fields):
        return "The target field is not declared by this repair."
    if dict(entry["evidence"]) != dict(declaration.evidence):
        return "The repair evidence changed."
    if entry["provenance"] != declaration.provenance:
        return "The repair provenance changed."
    expected_fields = list(_expanded_basis_fields(record, declaration, target))
    if entry["basis_fields"] != expected_fields:
        return "The declared basis fields changed."
    try:
        old = _field_value(record, target)
    except RepairError:
        return "The target field no longer exists."
    if old != entry["old_value"]:
        return "The target old value changed."
    basis, basis_fingerprint = proposal_basis(record, declaration, target, old)
    if dict(entry["basis"]) != basis or entry["basis_fingerprint"] != basis_fingerprint:
        return "A proposal basis value changed."
    try:
        _updated, changes = apply_declarations(
            [record], [declaration], modes=frozenset({"proposal-only"})
        )
    except RepairError as exc:
        return f"The repair no longer produces a valid change: {exc}"
    matching = [change for change in changes if change.field == target]
    if len(matching) != 1 or matching[0].new != entry["new_value"]:
        return "The repair now produces a different value."
    return None


def inspect_proposals(
    root: Path,
    normalized_file: Path,
    staging_dir: Path,
    proposal_file: Path,
    registry: RepairRegistry | None = None,
) -> tuple[ProposalReview, dict[str, str]]:
    """Load a proposal bundle and recompute each entry from current records."""
    chosen = registry or REGISTRY
    proposal = read_safe_document(
        root,
        normalized_file,
        staging_dir,
        proposal_file,
        allow_proposal=True,
    )
    payload = validate_proposal_payload(proposal.raw, str(proposal.path))
    records = read_safe_document(root, normalized_file, staging_dir, normalized_file)
    source_file = _relative_path(payload["source_file"], "proposal source_file")
    if source_file != records.relative_path:
        raise RepairError(
            f"Proposal source {source_file} is not the normalized file "
            f"{records.relative_path}"
        )
    by_id = {record.id: record for record in records.records}
    entries = tuple(copy.deepcopy(dict(item)) for item in payload["proposals"])
    reasons = {
        entry["proposal_entry_fingerprint"]: reason
        for entry in entries
        if (
            reason := _entry_reason(
                entry,
                by_id.get(entry["record_id"]),
                chosen,
            )
        )
    }
    return ProposalReview(proposal, records, entries), reasons


def mark_stale_proposals(
    review: ProposalReview,
    reasons: Mapping[str, str],
) -> None:
    """Write current stale reasons with a compare-and-swap guard."""
    if not reasons:
        return
    intended = _render_proposal_changes(review.proposal_document, stale=reasons)
    if intended != review.proposal_document.text:
        write_safe_document(review.proposal_document, intended)


def proposal_journal_path(staging_dir: Path) -> Path:
    return staging_dir.resolve() / ".repair-proposal-journal.json"


def proposal_archive_path(staging_dir: Path, proposal_file: Path) -> Path:
    return staging_dir.resolve() / "done" / proposal_file.name


def _read_file_state(path: Path, root: Path) -> tuple[str | None, str]:
    """Read one regular file and return text plus its content revision."""
    _reject_symlink_components(path.absolute(), root.resolve())
    try:
        details = os.lstat(path)
    except FileNotFoundError:
        return None, ABSENT_REVISION
    except OSError as exc:
        raise RepairError(f"Could not inspect {path}: {exc.strerror or exc}") from exc
    if not stat.S_ISREG(details.st_mode):
        raise RepairError(f"Transaction target is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RepairError(f"Could not open {path}: {exc.strerror or exc}") from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RepairError(f"Transaction target changed while it was read: {path}")
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError(f"Transaction target is not UTF-8: {path}") from exc
    return text, bytes_fingerprint(text)


def _cas_write(path: Path, root: Path, expected: str, intended: str) -> None:
    _current, revision = _read_file_state(path, root)
    if revision != expected:
        raise RepairError(
            f"Transaction target {path} changed: expected {expected}, found {revision}"
        )
    atomic_write_text_bound(
        path,
        intended,
        expected_revision=None if expected == ABSENT_REVISION else expected,
        expected_absent=expected == ABSENT_REVISION,
    )
    _written, written_revision = _read_file_state(path, root)
    intended_revision = bytes_fingerprint(intended)
    if written_revision != intended_revision:
        raise RepairError(f"Could not verify transaction target {path}")


def _archive_payload(text: str | None, path: Path) -> dict[str, Any]:
    if text is None:
        return {"version": SCHEMA_VERSION, "kind": ARCHIVE_KIND, "accepted": []}
    raw = _strict_data(text, path)
    expected = {"version", "kind", "accepted"}
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise RepairError(f"Proposal archive {path} does not match schema version 1")
    if raw.get("version") != SCHEMA_VERSION or raw.get("kind") != ARCHIVE_KIND:
        raise RepairError(f"Proposal archive {path} has an unsupported marker or version")
    if not isinstance(raw.get("accepted"), list) or not all(
        isinstance(item, Mapping) for item in raw["accepted"]
    ):
        raise RepairError(f"Proposal archive {path} has an invalid accepted list")
    archive_fields = _ENTRY_REQUIRED | {
        "source_file",
        "source_revision",
        "accepted_at",
    }
    for index, item in enumerate(raw["accepted"]):
        where = f"Proposal archive {path} accepted[{index}]"
        if set(item) != archive_fields:
            raise RepairError(f"{where} fields do not match the archive schema")
        _relative_path(item.get("source_file"), f"{where}.source_file")
        _sha(item.get("source_revision"), f"{where}.source_revision")
        if not isinstance(item.get("accepted_at"), str) or not item["accepted_at"]:
            raise RepairError(f"{where}.accepted_at must be non-empty text")
        _verify_entry_fingerprint(item, where)
    return copy.deepcopy(dict(raw))


def _archive_text(
    current_text: str | None,
    archive_path: Path,
    accepted: Sequence[Mapping[str, Any]],
    source_file: str,
    source_revision: str,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    payload = _archive_payload(current_text, archive_path)
    archived = payload["accepted"]
    by_fingerprint: dict[str, Mapping[str, Any]] = {}
    for item in archived:
        entry_fingerprint = item.get("proposal_entry_fingerprint")
        if not isinstance(entry_fingerprint, str) or not _SHA256.fullmatch(
            entry_fingerprint
        ):
            raise RepairError(f"Proposal archive {archive_path} has an invalid entry")
        if entry_fingerprint in by_fingerprint:
            raise RepairError(
                f"Proposal archive {archive_path} repeats {entry_fingerprint}"
            )
        by_fingerprint[entry_fingerprint] = item
    additions: list[dict[str, Any]] = []
    accepted_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    for proposal in accepted:
        entry = copy.deepcopy(dict(proposal))
        entry.pop("stale", None)
        entry.update(
            {
                "source_file": source_file,
                "source_revision": source_revision,
                "accepted_at": accepted_at,
            }
        )
        entry_fingerprint = entry["proposal_entry_fingerprint"]
        previous = by_fingerprint.get(entry_fingerprint)
        if previous is not None:
            comparable = dict(previous)
            comparable.pop("accepted_at", None)
            candidate = dict(entry)
            candidate.pop("accepted_at", None)
            if comparable != candidate:
                raise RepairError(
                    "The proposal archive has different data for fingerprint "
                    f"{entry_fingerprint}"
                )
            continue
        archived.append(entry)
        by_fingerprint[entry_fingerprint] = entry
        additions.append(entry)
    text = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
    return text, tuple(additions)


def _apply_accepted_entries(
    records: Sequence[VocabularyRecord],
    entries: Sequence[Mapping[str, Any]],
    registry: RepairRegistry,
) -> list[VocabularyRecord]:
    result = list(records)
    positions = {record.id: index for index, record in enumerate(result)}
    annotations: dict[tuple[str, str], list[str]] = {}
    for entry in entries:
        record_id = str(entry["record_id"])
        index = positions[record_id]
        record = result[index]
        field_name = str(entry["target_field"])
        if _field_value(record, field_name) != entry["old_value"]:
            raise RepairError(f"Proposal target {record_id}/{field_name} changed")
        result[index] = _set_field(record, field_name, entry["new_value"])
        annotations.setdefault((record_id, str(entry["code"])), []).append(field_name)
    for (record_id, code), changed_fields in sorted(annotations.items()):
        declaration = registry.get(code)
        if declaration is None:
            raise RepairError(f"Repair {code} is no longer registered")
        index = positions[record_id]
        result[index] = _annotate_repair(
            result[index], declaration, sorted(changed_fields)
        )
    return result


def _dependency_reasons(
    entries: Sequence[Mapping[str, Any]],
    accepted_fingerprints: frozenset[str],
) -> dict[str, str]:
    accepted_targets = {
        (str(entry["record_id"]), str(entry["target_field"]))
        for entry in entries
        if entry["proposal_entry_fingerprint"] in accepted_fingerprints
    }
    reasons: dict[str, str] = {}
    for entry in entries:
        fingerprint_ = str(entry["proposal_entry_fingerprint"])
        dependencies = {
            (str(entry["record_id"]), str(field_name))
            for field_name in entry["basis_fields"]
        }
        if fingerprint_ in accepted_fingerprints:
            dependencies.discard(
                (str(entry["record_id"]), str(entry["target_field"]))
            )
        changed = sorted(field for record_id, field in dependencies & accepted_targets)
        if not changed:
            continue
        if fingerprint_ in accepted_fingerprints:
            raise RepairError(
                f"Accepted proposal {entry['record_id']}/{entry['target_field']} "
                f"depends on accepted target {changed[0]}; regenerate it"
            )
        reasons[fingerprint_] = (
            "An accepted proposal changed basis field(s): " + ", ".join(changed)
        )
    return reasons


_JOURNAL_FIELDS = {
    "marker",
    "version",
    "paths",
    "input_revisions",
    "intended_revisions",
    "intended_text",
    "accepted_entries",
    "transaction_fingerprint",
}
_TARGET_NAMES = ("records", "archive", "staging")


def _journal_payload(
    root: Path,
    records_path: Path,
    archive_path: Path,
    staging_path: Path,
    input_revisions: Mapping[str, str],
    intended_text: Mapping[str, str],
    accepted_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paths = {
        "records": records_path.resolve().relative_to(root.resolve()).as_posix(),
        "archive": archive_path.resolve().relative_to(root.resolve()).as_posix(),
        "staging": staging_path.resolve().relative_to(root.resolve()).as_posix(),
    }
    payload: dict[str, Any] = {
        "marker": JOURNAL_MARKER,
        "version": SCHEMA_VERSION,
        "paths": paths,
        "input_revisions": dict(input_revisions),
        "intended_revisions": {
            name: bytes_fingerprint(intended_text[name]) for name in _TARGET_NAMES
        },
        "intended_text": dict(intended_text),
        "accepted_entries": [copy.deepcopy(dict(item)) for item in accepted_entries],
    }
    payload["transaction_fingerprint"] = fingerprint(payload)
    return payload


def _journal_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _validate_revision(value: Any, where: str, *, allow_absent: bool) -> str:
    if allow_absent and value == ABSENT_REVISION:
        return value
    return _sha(value, where)


def _load_journal(text: str, path: Path, root: Path) -> dict[str, Any]:
    raw = _strict_data(text, path)
    if not isinstance(raw, Mapping) or set(raw) != _JOURNAL_FIELDS:
        raise RepairError(f"Recovery journal {path} does not match schema version 1")
    data = copy.deepcopy(dict(raw))
    if data.get("marker") != JOURNAL_MARKER or data.get("version") != SCHEMA_VERSION:
        raise RepairError(f"Recovery journal {path} has an unsupported marker or version")
    for name in ("paths", "input_revisions", "intended_revisions", "intended_text"):
        value = data.get(name)
        if not isinstance(value, Mapping) or set(value) != set(_TARGET_NAMES):
            raise RepairError(f"Recovery journal {path} has invalid {name}")
    for name in _TARGET_NAMES:
        relative = _relative_path(data["paths"][name], f"journal paths.{name}")
        requested_target = root.resolve() / relative
        _reject_symlink_components(requested_target.absolute(), root.resolve())
        target = requested_target.resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise RepairError(f"Recovery journal path escapes the repository: {relative}") from exc
        _validate_revision(
            data["input_revisions"][name],
            f"journal input_revisions.{name}",
            allow_absent=True,
        )
        _validate_revision(
            data["intended_revisions"][name],
            f"journal intended_revisions.{name}",
            allow_absent=False,
        )
        intended = data["intended_text"][name]
        if not isinstance(intended, str):
            raise RepairError(f"Recovery journal intended_text.{name} must be text")
        if bytes_fingerprint(intended) != data["intended_revisions"][name]:
            raise RepairError(f"Recovery journal intended {name} bytes do not match")
    accepted = data.get("accepted_entries")
    if not isinstance(accepted, list) or not all(
        isinstance(item, Mapping) for item in accepted
    ):
        raise RepairError(f"Recovery journal {path} has invalid accepted entries")
    for index, entry in enumerate(accepted):
        if set(entry) != _ENTRY_REQUIRED:
            raise RepairError(
                f"Recovery journal accepted_entries[{index}] has invalid fields"
            )
        entry_fingerprint = entry.get("proposal_entry_fingerprint")
        _sha(entry_fingerprint, f"journal accepted_entries[{index}] fingerprint")
        _verify_entry_fingerprint(entry, f"journal accepted_entries[{index}]")
    transaction_fingerprint = data.pop("transaction_fingerprint", None)
    if transaction_fingerprint != fingerprint(data):
        raise RepairError(f"Recovery journal {path} fingerprint does not match")
    data["transaction_fingerprint"] = transaction_fingerprint
    return data


def _journal_targets(
    payload: Mapping[str, Any],
    root: Path,
    normalized_file: Path,
    staging_dir: Path,
) -> dict[str, Path]:
    targets = {
        name: (root.resolve() / payload["paths"][name]).resolve()
        for name in _TARGET_NAMES
    }
    expected_records = normalized_file.resolve()
    expected_staging_root = staging_dir.resolve()
    if targets["records"] != expected_records:
        raise RepairError("Recovery journal names a different normalized records file")
    if not _under(targets["staging"], expected_staging_root) or _under(
        targets["staging"], expected_staging_root / "done"
    ):
        raise RepairError("Recovery journal names a staging path outside active staging")
    expected_archive = proposal_archive_path(staging_dir, targets["staging"])
    if targets["archive"] != expected_archive:
        raise RepairError("Recovery journal archive path does not match its staging path")
    return targets


def _remove_journal(path: Path, root: Path, expected_text: str) -> None:
    current, _revision = _read_file_state(path, root)
    if current != expected_text:
        raise RepairError(f"Recovery journal changed before removal: {path}")
    try:
        path.unlink()
    except OSError as exc:
        raise RepairError(f"Could not remove recovery journal {path}: {exc}") from exc


def _advance_transaction(
    root: Path,
    journal_path: Path,
    journal_text: str,
    payload: Mapping[str, Any],
    targets: Mapping[str, Path],
) -> None:
    current_revisions = tuple(
        _read_file_state(targets[name], root)[1] for name in _TARGET_NAMES
    )
    input_revisions = tuple(payload["input_revisions"][name] for name in _TARGET_NAMES)
    intended_revisions = tuple(
        payload["intended_revisions"][name] for name in _TARGET_NAMES
    )
    valid_states = []
    for completed in range(4):
        expected = tuple(
            intended_revisions[index] if index < completed else input_revisions[index]
            for index in range(3)
        )
        if current_revisions == expected:
            valid_states.append(completed)
    if not valid_states:
        raise RepairError(
            f"Recovery journal {journal_path} does not match an ordered transaction state"
        )
    completed = max(valid_states)
    for index in range(completed, 3):
        name = _TARGET_NAMES[index]
        _current, revision = _read_file_state(targets[name], root)
        if revision == intended_revisions[index]:
            continue
        _cas_write(
            targets[name],
            root,
            input_revisions[index],
            payload["intended_text"][name],
        )
    final = tuple(_read_file_state(targets[name], root)[1] for name in _TARGET_NAMES)
    if final != intended_revisions:
        raise RepairError(f"Could not verify transaction for recovery journal {journal_path}")
    _remove_journal(journal_path, root, journal_text)


def recover_proposal_transaction(
    root: Path,
    normalized_file: Path,
    staging_dir: Path,
) -> bool:
    """Finish one interrupted proposal transaction from its exact journal bytes."""
    journal_path = proposal_journal_path(staging_dir)
    initial_text, revision = _read_file_state(journal_path, root)
    if revision == ABSENT_REVISION or initial_text is None:
        return False
    initial_payload = _load_journal(initial_text, journal_path, root)
    targets = _journal_targets(initial_payload, root, normalized_file, staging_dir)
    with exclusive_path_locks([journal_path, *targets.values()]):
        journal_text, locked_revision = _read_file_state(journal_path, root)
        if journal_text is None or locked_revision != revision or journal_text != initial_text:
            raise RepairError(f"Recovery journal changed before recovery: {journal_path}")
        payload = _load_journal(journal_text, journal_path, root)
        locked_targets = _journal_targets(payload, root, normalized_file, staging_dir)
        if locked_targets != targets:
            raise RepairError(f"Recovery journal target paths changed: {journal_path}")
        _advance_transaction(root, journal_path, journal_text, payload, targets)
    return True


def accept_proposals(
    review: ProposalReview,
    decisions: Mapping[str, bool],
    *,
    registry: RepairRegistry | None = None,
) -> ProposalAcceptance:
    """Apply reviewed proposal fields in one journaled transaction."""
    chosen = registry or REGISTRY
    root = review.records_document.root
    normalized_file = review.records_document.normalized_file
    staging_dir = review.proposal_document.staging_dir
    proposal_file = review.proposal_document.path
    archive_path = proposal_archive_path(staging_dir, proposal_file)
    journal_path = proposal_journal_path(staging_dir)
    known_fingerprints = {
        str(entry["proposal_entry_fingerprint"]) for entry in review.entries
    }
    unknown_decisions = sorted(set(decisions) - known_fingerprints)
    if unknown_decisions:
        raise RepairError(
            "Review decisions name unknown proposal fingerprint(s): "
            + ", ".join(unknown_decisions)
        )
    accepted_fingerprints = frozenset(
        entry_fingerprint
        for entry_fingerprint, accepted in decisions.items()
        if accepted
    )
    if not accepted_fingerprints:
        return ProposalAcceptance(0, len(review.entries), archive_path)

    lock_paths = [journal_path, normalized_file, archive_path, proposal_file]
    with exclusive_path_locks(lock_paths):
        journal_text, journal_revision = _read_file_state(journal_path, root)
        if journal_revision != ABSENT_REVISION or journal_text is not None:
            raise RepairError(
                f"Unfinished repair proposal journal blocks this command: {journal_path}"
            )
        current, reasons = inspect_proposals(
            root,
            normalized_file,
            staging_dir,
            proposal_file,
            chosen,
        )
        if (
            current.proposal_document.revision
            != review.proposal_document.revision
            or current.records_document.revision != review.records_document.revision
        ):
            raise RepairError("Records or proposals changed during review; run the command again")
        if reasons:
            details = "; ".join(
                f"{entry_fingerprint}: {reason}"
                for entry_fingerprint, reason in sorted(reasons.items())
            )
            raise RepairError(f"Proposal entries became stale during review: {details}")
        current_fingerprints = {
            str(entry["proposal_entry_fingerprint"]) for entry in current.entries
        }
        if not accepted_fingerprints <= current_fingerprints:
            raise RepairError("An accepted proposal entry changed during review")
        dependency_reasons = _dependency_reasons(
            current.entries, accepted_fingerprints
        )
        accepted_entries = tuple(
            entry
            for entry in current.entries
            if entry["proposal_entry_fingerprint"] in accepted_fingerprints
        )
        updated_records = _apply_accepted_entries(
            current.records_document.records,
            accepted_entries,
            chosen,
        )
        issues = validate_records(updated_records, current.records_document.relative_path)
        if has_errors(issues):
            codes = ", ".join(
                sorted(
                    {
                        issue.code or "validation-error"
                        for issue in issues
                        if issue.level == "error"
                    }
                )
            )
            raise RepairError(f"Accepted proposals fail full-file validation: {codes}")
        records_text = render_records_document(
            current.records_document, updated_records
        )
        archive_current, archive_revision = _read_file_state(archive_path, root)
        archive_text, _archive_additions = _archive_text(
            archive_current,
            archive_path,
            accepted_entries,
            current.records_document.relative_path,
            current.proposal_document.raw["source_revision"],
        )
        staging_text = _render_proposal_changes(
            current.proposal_document,
            accepted=accepted_fingerprints,
            stale=dependency_reasons,
        )
        input_revisions = {
            "records": current.records_document.revision,
            "archive": archive_revision,
            "staging": current.proposal_document.revision,
        }
        intended_text = {
            "records": records_text,
            "archive": archive_text,
            "staging": staging_text,
        }
        payload = _journal_payload(
            root,
            current.records_document.path,
            archive_path,
            current.proposal_document.path,
            input_revisions,
            intended_text,
            accepted_entries,
        )
        transaction_text = _journal_text(payload)
        _cas_write(journal_path, root, ABSENT_REVISION, transaction_text)
        targets = {
            "records": current.records_document.path,
            "archive": archive_path,
            "staging": current.proposal_document.path,
        }
        _advance_transaction(
            root,
            journal_path,
            transaction_text,
            payload,
            targets,
        )
    return ProposalAcceptance(
        len(accepted_entries),
        len(current.entries) - len(accepted_entries),
        archive_path,
    )
