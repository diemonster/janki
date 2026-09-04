"""Named, constrained repairs to derived fields.

The registry gives callbacks immutable projections, never complete records or
paths, and apply is limited to the derived fields in the M7.1 allowlist —
romaji reconstructed from a reading or a furigana field, and nothing a person
wrote.

A protected-content change used to become a fingerprinted proposal for a human
to accept field by field: a staging document, a two-phase journal, an archive,
and a `promote --accept-proposals` review loop, about a thousand lines of it.
It had exactly one producer, the furigana punctuation repair, and M8.3 deleted
that as model-audit logic — leaving a subsystem no input could reach. Deleted
2026-08-15 rather than kept warm for a hypothetical second instance. A repair
that needs human judgement is a template clause or a staging-file edit, both
of which already have a human in them.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import stat
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
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
    YAML_LOADER,
    atomic_write_text_bound,
    exclusive_path_lock,
    read_bytes_bound,
    read_bytes_bound_snapshot,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.romaji import kana_to_romaji
from japanese_anki.validation import has_errors, validate_records

REPAIR_MODES = ("ingest-safe", "revoked")
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
STRUCTURED_RECORD_FIELDS = frozenset({"examples", "source"})

_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_PATH = re.compile(r"(?P<field>[a-z_]+)(?:\[(?P<index>\d+)\]\.(?P<nested>[a-z_]+))?\Z")
_PATTERN = re.compile(r"(?P<field>[a-z_]+)(?:\[\*\]\.(?P<nested>[a-z_]+))?\Z")


class RepairError(JankiError):
    pass


class _StrictLoader(YAML_LOADER):
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


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite number {value}")


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _validate_json_value(value: Any, where: str) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RepairError(f"{where} cannot contain a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RepairError(f"{where} mapping keys must be text")
            _validate_json_value(item, f"{where}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{where}[{index}]")
        return
    raise RepairError(f"{where} must contain only canonical JSON values")


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
    identity: tuple[int, int, int, int]
    raw: Any
    records: tuple[VocabularyRecord, ...]
    records_container: str


def _validate_field_pattern(value: str, where: str) -> None:
    match = _PATTERN.fullmatch(value) if isinstance(value, str) else None
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
            by_code[declaration.code] = replace(
                declaration,
                allowed_fields=tuple(declaration.allowed_fields),
                input_fields=tuple(declaration.input_fields),
                evidence=_freeze(declaration.evidence),
            )
        self._by_code = MappingProxyType(dict(sorted(by_code.items())))

    @staticmethod
    def _validate(declaration: RepairDeclaration) -> None:
        if not isinstance(declaration.code, str) or not _SLUG.fullmatch(declaration.code):
            raise RepairError("Repair code must be a lowercase slug")
        if not isinstance(declaration.version, str) or not _VERSION.fullmatch(
            declaration.version
        ):
            raise RepairError(f"Repair {declaration.code} has an invalid version")
        if declaration.mode not in REPAIR_MODES:
            raise RepairError(f"Repair {declaration.code} has an invalid mode")
        if (
            not isinstance(declaration.phase, str)
            or not _SLUG.fullmatch(declaration.phase)
            or not isinstance(declaration.provenance, str)
            or not declaration.provenance.strip()
        ):
            raise RepairError(f"Repair {declaration.code} needs phase and provenance")
        if (
            not isinstance(declaration.allowed_fields, tuple)
            or not isinstance(declaration.input_fields, tuple)
            or not declaration.allowed_fields
            or not declaration.input_fields
        ):
            raise RepairError(f"Repair {declaration.code} needs declared fields")
        if len(set(declaration.allowed_fields)) != len(declaration.allowed_fields):
            raise RepairError(f"Repair {declaration.code} repeats an allowed field")
        if len(set(declaration.input_fields)) != len(declaration.input_fields):
            raise RepairError(f"Repair {declaration.code} repeats an input field")
        for name in (*declaration.allowed_fields, *declaration.input_fields):
            _validate_field_pattern(name, f"Repair {declaration.code}")
            if name in STRUCTURED_RECORD_FIELDS:
                raise RepairError(
                    f"Repair {declaration.code} must declare leaf fields, not {name!r}"
                )
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
        _validate_json_value(
            declaration.evidence, f"Repair {declaration.code} evidence"
        )
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
        """The named repairs, always in registry order.

        Not the caller's order. Repairs are not independent: derivations read
        fields other repairs correct, so
        `example-romaji-from-furigana example-furigana-space-after-punctuation`
        on one command line derived romaji from furigana that the *next*
        declaration was about to fix — both recorded as successful, with the
        romaji describing a sentence the record was one step from no longer
        holding.

        Registry order is the sorted code order, which puts each field's
        correction before the derivations that read it today. That is luck
        rather than design; what is not luck is that a caller can no longer
        invert it by typing the codes the other way round.
        """
        names = tuple(codes) if codes else tuple(self._by_code)
        if len(set(names)) != len(names):
            raise RepairError("Repair code list contains duplicates")
        unknown = [name for name in names if name not in self._by_code]
        if unknown:
            raise RepairError("Unknown repair code(s): " + ", ".join(unknown))
        wanted = set(names)
        return tuple(
            declaration
            for code, declaration in self._by_code.items()
            if code in wanted
        )

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
        "evidence": _thaw(declaration.evidence),
        "provenance": declaration.provenance,
    }


def _annotate_repair(
    record: VocabularyRecord, declaration: RepairDeclaration, fields: Sequence[str]
) -> VocabularyRecord:
    raw_fields = dict(record.source.raw_fields)
    existing = raw_fields.get("janki_repairs", "")
    if existing:
        try:
            payload = json.loads(
                existing,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
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


def _require_canonical_record(record: VocabularyRecord, repair_code: str) -> None:
    raw = record.to_dict()
    try:
        round_trip = VocabularyRecord.from_dict(raw).to_dict()
    except JankiError as exc:
        raise RepairError(
            f"Repair {repair_code} produced an invalid record: {exc}"
        ) from exc
    if round_trip != raw:
        raise RepairError(
            f"Repair {repair_code} produced a value outside the canonical schema"
        )


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
        evidence = declaration.evidence
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
                        _thaw(declaration.evidence),
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
            _require_canonical_record(updated, declaration.code)
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
        # `settle`, not `regenerate`: a romaji that verifies against the
        # reading is kept with its word spacing, so this repair no longer
        # flattens a correctly segmented value into an unsegmented one every
        # time it runs.
        repaired, _rejected = qc.settle_example_romaji(
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


#: A bracketed reading, for recovering the text underneath the annotation.
_RUBY_READING = re.compile(r"\[[^\]]*\]")

#: Japanese sentence punctuation immediately followed by a ruby group, which
#: means the space between them is missing. Anki delimits ruby groups by
#: spaces, so `先週[せんしゅう]、家族[かぞく]` makes `、家族` the base and draws
#: かぞく over the comma as well as the word — confirmed against Anki's own
#: renderer, not inferred.
#:
#: Full-width only. ASCII `,` and `.` are digit and decimal separators, not
#: sentence punctuation, so including them turned `1,000円[えん]` into
#: `1, 000円[えん]` — a *worse* base than it started with, and a space inside a
#: number that `furigana_reading`, the romaji derivation and TTS all go on to
#: read. `?` and `!` are out for the same reason: a Japanese sentence ends in
#: ？ or ！, and an ASCII one is likelier to sit inside quoted Latin.
#:
#: A closing quote or bracket goes with the punctuation, so `「はい。」と家族`
#: gains its space after the 」 rather than before it. Inserting before would
#: leave the base `」と家族` exactly as it was and add a stray space for
#: nothing.
#:
#: The lookahead crosses any run of non-space, non-bracket characters, so it
#: also fires on `は、いい 天気[てんき]`, where `いい` stays in the base either
#: way and only the comma comes out. That is strictly better and not a full
#: fix; a full one would need to know where words end.
_RUBY_AFTER_PUNCTUATION = re.compile(r"([、。？！，][」』）】〉》]*)(?=[^\s\[\]]*\[)")


def _spaced_furigana(value: str) -> str:
    """``value`` with the missing space after punctuation restored.

    Structural, not a judgement about Japanese: punctuation cannot carry a
    reading, so a ruby base that begins with one is a typing slip and there is
    nothing to decide. Idempotent — a space already there defeats the
    lookahead — so re-running changes nothing.
    """
    return _RUBY_AFTER_PUNCTUATION.sub(r"\1 ", value)


def _example_furigana_values(before: Mapping[str, Any]) -> list[str]:
    return [_spaced_furigana(value) for value in before["examples[*].furigana"]]


def _example_furigana_pre(before: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    del evidence
    return tuple(_example_furigana_values(before)) != before["examples[*].furigana"]


def _example_furigana_transform(
    before: Mapping[str, Any], evidence: Mapping[str, Any]
) -> Mapping[str, Any]:
    del evidence
    return {
        f"examples[{index}].furigana": value
        for index, value in enumerate(_example_furigana_values(before))
        if value != before["examples[*].furigana"][index]
    }


def _example_furigana_post(
    before: Mapping[str, Any], planned: Mapping[str, Any], evidence: Mapping[str, Any]
) -> bool:
    del evidence
    expected = _example_furigana_values(before)
    if any(
        planned.get(f"examples[{index}].furigana", old) != expected[index]
        for index, old in enumerate(before["examples[*].furigana"])
    ):
        return False
    # The sentence being annotated must not change — only where the ruby
    # brackets sit within it.
    #
    # Not a comparison of *readings*: this repair deliberately changes those.
    # A base that swallowed `、京都` also swallowed the `みに` before it, so
    # 休みに vanished from the reading entirely and the card drew きょうと over
    # five characters. Restoring the space gives both back. Comparing readings
    # would therefore fail on every real case, and the escape hatch that first
    # papered over it was wide enough to hide a dropped character.
    #
    # Stripping the readings out of the notation leaves the sentence it
    # annotates, and *that* is what may not move.
    return all(
        _annotated_text(old) == _annotated_text(expected[index])
        for index, old in enumerate(before["examples[*].furigana"])
    )


def _annotated_text(furigana: str) -> str:
    """The sentence a furigana field annotates: its readings and spacing gone."""
    return _RUBY_READING.sub("", furigana).replace(" ", "").replace("\u3000", "")



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
            code="example-furigana-space-after-punctuation",
            version="1.0.0",
            phase="ingest",
            mode="ingest-safe",
            allowed_fields=("examples[*].furigana",),
            input_fields=("examples[*].furigana",),
            evidence={"algorithm": "japanese_anki.repairs._spaced_furigana"},
            precondition=_example_furigana_pre,
            transformation=_example_furigana_transform,
            postcondition=_example_furigana_post,
            provenance=(
                "Restored the space between punctuation and a following ruby "
                "group, which Anki needs to keep the punctuation out of the base."
            ),
        ),
        RepairDeclaration(
            code="example-romaji-from-furigana",
            # 1.2.0, not 1.1.0: the algorithm changed under it. `version` is
            # written into each record's `janki_repairs` provenance and into
            # the pinned plan fingerprint, so leaving it would have two
            # different behaviours claiming the same name — 1.1.0 meaning
            # always-rebuild on records repaired before 2026-08-17, and
            # 1.1.0 meaning check-then-keep on records repaired after.
            version="1.2.0",
            phase="ingest",
            mode="ingest-safe",
            allowed_fields=("examples[*].romaji",),
            input_fields=(
                "examples[*].japanese",
                "examples[*].furigana",
                "examples[*].romaji",
            ),
            evidence={"algorithm": "japanese_anki.qc.settle_example_romaji"},
            precondition=_example_romaji_pre,
            transformation=_example_romaji_transform,
            postcondition=_example_romaji_post,
            provenance="Derived example romaji from stored Japanese and furigana.",
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

            return json.loads(
                text,
                object_pairs_hook=unique,
                parse_constant=_reject_json_constant,
            )
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.load(text, Loader=_StrictLoader)
    except RepairError:
        raise
    except (json.JSONDecodeError, ValueError, yaml.YAMLError) as exc:
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
        identity, revision, raw_bytes = read_bytes_bound_snapshot(candidate)
    except (JankiError, OSError) as exc:
        if "non-regular" in str(exc):
            raise RepairError(f"Repair input is not a regular file: {path}") from exc
        raise RepairError(f"Could not safely read repair input {path}: {exc}") from exc
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError(f"Repair input is not UTF-8: {path}") from exc
    raw = _strict_data(text, candidate)
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
        revision,
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
        if bytes_fingerprint(read_bytes_bound(document.path)) != bytes_fingerprint(text):
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
    """Build one check-only plan: what a repair would change, changing nothing."""
    current = list(document.records)
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
        current = updated
    issues = validate_records(current, document.relative_path)
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
    del target
    result: list[str] = []
    for pattern in declaration.input_fields:
        match = _PATTERN.fullmatch(pattern)
        if match is None:
            raise RepairError(f"Invalid basis field {pattern!r}")
        if match.group("nested") is None:
            result.append(match.group("field"))
            continue
        items = getattr(record, match.group("field"))
        result.extend(
            f"{match.group('field')}[{index}].{match.group('nested')}"
            for index in range(len(items))
        )
    return tuple(dict.fromkeys(result))
