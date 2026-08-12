"""Strict, read-only status for deck-driven hardening evidence.

M7.2 owns two reviewed document types: the systemic findings catalog and pilot
reports. Case and oracle documents arrive in M7.3. This module validates their
IDs now, but it does not guess their schemas or resolve them early.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from japanese_anki.errors import JankiError

SCHEMA_VERSION = 1
FINDING_STATES = ("open", "fixed", "deferred", "accepted-risk")
M7_SOURCE_ARCHETYPES = (
    "native-pdf-table",
    "mixed-layout-pdf",
    "scan",
    "camera",
    "vertical-text",
    "dense-annotation",
    "handwritten",
    "printed-ruby",
    "bilingual-layout",
)
PILOT_STEPS = (
    "extract",
    "coverage_review",
    "promote",
    "dictionary_enrich",
    "ai_enrich",
    "audio",
    "final_review",
    "build",
)

_SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HardeningError(JankiError):
    pass


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader with one essential difference: duplicate keys are errors."""


def _construct_unique_mapping(
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


_StrictLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    fingerprint: str
    locator: str


@dataclass(frozen=True, slots=True)
class RiskApproval:
    authority: str
    finding_id: str
    approved_at: str
    risk_content_fingerprint: str
    risk: str
    reason: str


@dataclass(frozen=True, slots=True)
class Finding:
    id: str
    state: str
    pipeline_stage: str
    source_archetypes: tuple[str, ...]
    symptom: str
    invariant: str
    evidence: tuple[EvidenceReference, ...]
    recurrences: tuple[EvidenceReference, ...]
    case_ids: tuple[str, ...]
    fix_ref: str | None = None
    deferral_reason: str | None = None
    risk: str | None = None
    reason: str | None = None
    approval: RiskApproval | None = None


@dataclass(frozen=True, slots=True)
class FindingsCatalog:
    path: Path
    findings: tuple[Finding, ...]


@dataclass(frozen=True, slots=True)
class CountSet:
    extracted: int
    omitted: int
    held: int
    promoted: int
    rejected: int


@dataclass(frozen=True, slots=True)
class CorrectionCounts:
    content_specific: int
    systemic: int


@dataclass(frozen=True, slots=True)
class RepairCount:
    code: str
    version: str
    count: int


@dataclass(frozen=True, slots=True)
class ModelFingerprint:
    phase: str
    provider: str
    model: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class PromptFingerprint:
    phase: str
    kind: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class Pilot:
    path: Path
    relative_path: str
    id: str
    source_fingerprint: str
    source_archetype: str
    redistributable: bool
    unit_oracle_id: str
    eval_case_id: str
    counts: CountSet
    corrections: CorrectionCounts
    repair_applications: tuple[RepairCount, ...]
    false_positive_repairs: tuple[RepairCount, ...]
    model_fingerprints: tuple[ModelFingerprint, ...]
    prompt_fingerprints: tuple[PromptFingerprint, ...]
    finding_ids: tuple[str, ...]
    pipeline: tuple[tuple[str, bool], ...]
    notes: str | None = None

    @property
    def complete(self) -> bool:
        return all(value for _, value in self.pipeline)

    @property
    def missing_steps(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.pipeline if not value)


@dataclass(frozen=True, slots=True)
class RepairFalsePositive:
    code: str
    version: str
    count: int
    pilot_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HardeningStatus:
    findings: tuple[Finding, ...]
    pilots: tuple[Pilot, ...]
    false_positives: tuple[RepairFalsePositive, ...]
    catalog_path: str = "quality/findings.yaml"
    pilots_path: str = "quality/pilots"


def _where(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _reject_symlink_components(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise HardeningError(f"Hardening path escapes the repository: {path}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise HardeningError(
                f"Hardening path {_where(path, root)} uses symlink component "
                f"{_where(current, root)}"
            )


def _read_yaml(path: Path, root: Path) -> Any:
    root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    _reject_symlink_components(candidate, root)
    try:
        before = candidate.stat()
    except FileNotFoundError as exc:
        raise HardeningError(f"Hardening file not found: {_where(candidate, root)}") from exc
    except OSError as exc:
        raise HardeningError(
            f"Could not inspect {_where(candidate, root)}: {exc.strerror or exc}"
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise HardeningError(
            f"Hardening path {_where(candidate, root)} is not a regular file"
        )
    try:
        text = candidate.read_text(encoding="utf-8")
        after = candidate.stat()
    except UnicodeDecodeError as exc:
        raise HardeningError(
            f"Could not decode {_where(candidate, root)} as UTF-8: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise HardeningError(
            f"Could not read {_where(candidate, root)}: {exc.strerror or exc}"
        ) from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise HardeningError(
            f"Hardening file {_where(candidate, root)} changed while it was read"
        )
    try:
        return yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise HardeningError(
            f"Could not parse {_where(candidate, root)}: {exc}"
        ) from exc


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HardeningError(f"{where} must be a mapping")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise HardeningError(f"{where} must be a list")
    return value


def _no_unknown(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = [key for key in data if not isinstance(key, str) or key not in allowed]
    if unknown:
        names = ", ".join(sorted(repr(key) for key in unknown))
        raise HardeningError(f"{where} has unknown field(s): {names}")


def _required(data: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise HardeningError(f"{where} is missing required field {key!r}")
    return data[key]


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HardeningError(f"{where} must be a non-empty string")
    return value.strip()


def _optional_text(data: Mapping[str, Any], key: str, where: str) -> str | None:
    if key not in data:
        return None
    return _text(data[key], f"{where}.{key}")


def _slug(value: Any, where: str) -> str:
    result = _text(value, where)
    if not _SLUG.fullmatch(result):
        raise HardeningError(
            f"{where} must be a lowercase slug such as 'missing-second-column'"
        )
    return result


def _fingerprint(value: Any, where: str) -> str:
    result = _text(value, where)
    if not _SHA256.fullmatch(result):
        raise HardeningError(f"{where} must be a lowercase SHA-256 fingerprint")
    return result


def _count(value: Any, where: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HardeningError(f"{where} must be an integer")
    floor = 1 if positive else 0
    if value < floor:
        word = "positive" if positive else "non-negative"
        raise HardeningError(f"{where} must be {word}")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise HardeningError(f"{where} must be true or false")
    return value


def _schema_version(value: Any, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != SCHEMA_VERSION:
        raise HardeningError(f"{where} must be integer {SCHEMA_VERSION}")


def _locator(value: Any, where: str) -> str:
    result = _text(value, where)
    normalized = result.replace("\\", "/")
    windows = PureWindowsPath(result)
    if (
        "\x00" in result
        or normalized.startswith(("/", "~"))
        or re.match(r"[A-Za-z][A-Za-z0-9+.-]*:/", normalized)
        or windows.is_absolute()
        or windows.drive
        or ".." in normalized.split("/")
    ):
        raise HardeningError(
            f"{where} must be a repository-relative locator without path traversal"
        )
    return normalized


def _unique(values: Sequence[str], where: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise HardeningError(f"{where} contains duplicate values")
    return tuple(values)


def _slug_list(value: Any, where: str, *, nonempty: bool = False) -> tuple[str, ...]:
    raw = _list(value, where)
    if nonempty and not raw:
        raise HardeningError(f"{where} must not be empty")
    return _unique(
        [_slug(item, f"{where}[{index}]") for index, item in enumerate(raw)],
        where,
    )


def _parse_evidence(
    value: Any, where: str, *, nonempty: bool
) -> tuple[EvidenceReference, ...]:
    raw = _list(value, where)
    if nonempty and not raw:
        raise HardeningError(f"{where} must not be empty")
    result: list[EvidenceReference] = []
    for index, item in enumerate(raw):
        item_where = f"{where}[{index}]"
        data = _mapping(item, item_where)
        _no_unknown(data, {"fingerprint", "locator"}, item_where)
        result.append(
            EvidenceReference(
                fingerprint=_fingerprint(
                    _required(data, "fingerprint", item_where),
                    f"{item_where}.fingerprint",
                ),
                locator=_locator(
                    _required(data, "locator", item_where), f"{item_where}.locator"
                ),
            )
        )
    pairs = [(item.fingerprint, item.locator) for item in result]
    if len(set(pairs)) != len(pairs):
        raise HardeningError(f"{where} contains duplicate evidence references")
    return tuple(result)


def _parse_approval(value: Any, where: str) -> RiskApproval:
    data = _mapping(value, where)
    _no_unknown(
        data,
        {
            "authority",
            "finding_id",
            "approved_at",
            "risk_content_fingerprint",
            "risk",
            "reason",
        },
        where,
    )
    authority = _text(_required(data, "authority", where), f"{where}.authority")
    if authority != "repository-owner":
        raise HardeningError(f"{where}.authority must be 'repository-owner'")
    raw_date = _required(data, "approved_at", where)
    if isinstance(raw_date, date):
        approved_at = raw_date.isoformat()
    else:
        approved_at = _text(raw_date, f"{where}.approved_at")
        try:
            parsed_date = date.fromisoformat(approved_at)
        except ValueError as exc:
            raise HardeningError(f"{where}.approved_at must be an ISO date") from exc
        if parsed_date.isoformat() != approved_at:
            raise HardeningError(f"{where}.approved_at must use YYYY-MM-DD")
    if len(approved_at) != 10:
        raise HardeningError(f"{where}.approved_at must use YYYY-MM-DD")
    return RiskApproval(
        authority=authority,
        finding_id=_slug(
            _required(data, "finding_id", where), f"{where}.finding_id"
        ),
        approved_at=approved_at,
        risk_content_fingerprint=_fingerprint(
            _required(data, "risk_content_fingerprint", where),
            f"{where}.risk_content_fingerprint",
        ),
        risk=_text(_required(data, "risk", where), f"{where}.risk"),
        reason=_text(_required(data, "reason", where), f"{where}.reason"),
    )


def _evidence_payload(items: Sequence[EvidenceReference]) -> list[dict[str, str]]:
    return [
        {"fingerprint": item.fingerprint, "locator": item.locator}
        for item in sorted(items, key=lambda item: (item.locator, item.fingerprint))
    ]


def risk_content_fingerprint(finding: Finding) -> str:
    """Fingerprint the accepted risk, without state, approval, or dates."""
    payload = {
        "id": finding.id,
        "pipeline_stage": finding.pipeline_stage,
        "source_archetypes": sorted(finding.source_archetypes),
        "symptom": finding.symptom,
        "invariant": finding.invariant,
        "evidence": _evidence_payload(finding.evidence),
        "recurrences": _evidence_payload(finding.recurrences),
        "case_ids": sorted(finding.case_ids),
        "risk": finding.risk,
        "reason": finding.reason,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_finding(value: Any, where: str) -> Finding:
    data = _mapping(value, where)
    _no_unknown(
        data,
        {
            "id",
            "state",
            "pipeline_stage",
            "source_archetypes",
            "symptom",
            "invariant",
            "evidence",
            "recurrences",
            "case_ids",
            "fix_ref",
            "deferral_reason",
            "risk",
            "reason",
            "approval",
        },
        where,
    )
    state = _text(_required(data, "state", where), f"{where}.state")
    if state not in FINDING_STATES:
        raise HardeningError(
            f"{where}.state must be one of {', '.join(FINDING_STATES)}"
        )
    approval = (
        _parse_approval(data["approval"], f"{where}.approval")
        if "approval" in data
        else None
    )
    finding = Finding(
        id=_slug(_required(data, "id", where), f"{where}.id"),
        state=state,
        pipeline_stage=_slug(
            _required(data, "pipeline_stage", where), f"{where}.pipeline_stage"
        ),
        source_archetypes=tuple(
            sorted(
                _slug_list(
                    _required(data, "source_archetypes", where),
                    f"{where}.source_archetypes",
                    nonempty=True,
                )
            )
        ),
        symptom=_text(_required(data, "symptom", where), f"{where}.symptom"),
        invariant=_text(_required(data, "invariant", where), f"{where}.invariant"),
        evidence=_parse_evidence(
            _required(data, "evidence", where), f"{where}.evidence", nonempty=True
        ),
        recurrences=_parse_evidence(
            data.get("recurrences", []), f"{where}.recurrences", nonempty=False
        ),
        case_ids=tuple(
            sorted(_slug_list(data.get("case_ids", []), f"{where}.case_ids"))
        ),
        fix_ref=(
            _locator(data["fix_ref"], f"{where}.fix_ref")
            if "fix_ref" in data
            else None
        ),
        deferral_reason=_optional_text(data, "deferral_reason", where),
        risk=_optional_text(data, "risk", where),
        reason=_optional_text(data, "reason", where),
        approval=approval,
    )
    initial_evidence = {
        (item.fingerprint, item.locator) for item in finding.evidence
    }
    repeated_evidence = {
        (item.fingerprint, item.locator) for item in finding.recurrences
    }
    if initial_evidence & repeated_evidence:
        raise HardeningError(
            f"{where}.recurrences repeats an initial evidence reference"
        )
    if state == "open":
        forbidden = (
            finding.fix_ref,
            finding.deferral_reason,
            finding.risk,
            finding.reason,
            approval,
        )
        if any(item is not None for item in forbidden):
            raise HardeningError(
                f"{where}: an open finding cannot have terminal-state fields"
            )
    elif state == "fixed":
        if not finding.case_ids or finding.fix_ref is None:
            raise HardeningError(f"{where}: fixed requires case_ids and fix_ref")
        if any(
            item is not None
            for item in (
                finding.deferral_reason,
                finding.risk,
                finding.reason,
                approval,
            )
        ):
            raise HardeningError(f"{where}: fixed cannot have deferral or risk fields")
    elif state == "deferred":
        if finding.deferral_reason is None:
            raise HardeningError(f"{where}: deferred requires deferral_reason")
        if any(
            item is not None
            for item in (finding.fix_ref, finding.risk, finding.reason, approval)
        ):
            raise HardeningError(f"{where}: deferred cannot have fix or risk fields")
    else:
        if finding.risk is None or finding.reason is None:
            raise HardeningError(f"{where}: accepted-risk requires risk and reason")
        if finding.fix_ref is not None or finding.deferral_reason is not None:
            raise HardeningError(f"{where}: accepted-risk cannot have fix or deferral fields")
        expected = risk_content_fingerprint(finding)
        if approval is None:
            raise HardeningError(
                f"{where}: accepted-risk requires repository-owner approval; "
                f"the risk content fingerprint is {expected}"
            )
        if (
            approval.finding_id != finding.id
            or approval.risk != finding.risk
            or approval.reason != finding.reason
        ):
            raise HardeningError(
                f"{where}.approval must name the finding ID, risk, and reason exactly"
            )
        if approval.risk_content_fingerprint != expected:
            raise HardeningError(
                f"{where}.approval is stale: risk content fingerprint must be {expected}"
            )
    return finding


def load_findings(root: Path) -> FindingsCatalog:
    root = root.resolve()
    path = root / "quality" / "findings.yaml"
    data = _mapping(_read_yaml(path, root), "quality/findings.yaml")
    _no_unknown(data, {"version", "findings"}, "quality/findings.yaml")
    version = _required(data, "version", "quality/findings.yaml")
    _schema_version(version, "quality/findings.yaml version")
    raw = _list(
        _required(data, "findings", "quality/findings.yaml"),
        "quality/findings.yaml.findings",
    )
    findings = tuple(
        _parse_finding(item, f"quality/findings.yaml.findings[{index}]")
        for index, item in enumerate(raw)
    )
    ids = [finding.id for finding in findings]
    if len(set(ids)) != len(ids):
        raise HardeningError("quality/findings.yaml has duplicate finding IDs")
    return FindingsCatalog(
        path=path,
        findings=tuple(sorted(findings, key=lambda item: item.id)),
    )


def _parse_counts(value: Any, where: str) -> CountSet:
    data = _mapping(value, where)
    fields = ("extracted", "omitted", "held", "promoted", "rejected")
    _no_unknown(data, set(fields), where)
    values = {
        name: _count(_required(data, name, where), f"{where}.{name}")
        for name in fields
    }
    return CountSet(**values)


def _parse_corrections(value: Any, where: str) -> CorrectionCounts:
    data = _mapping(value, where)
    allowed = {"content_specific", "systemic"}
    _no_unknown(data, allowed, where)
    return CorrectionCounts(
        content_specific=_count(
            _required(data, "content_specific", where), f"{where}.content_specific"
        ),
        systemic=_count(_required(data, "systemic", where), f"{where}.systemic"),
    )


def _parse_repairs(value: Any, where: str) -> tuple[RepairCount, ...]:
    raw = _list(value, where)
    result: list[RepairCount] = []
    for index, item in enumerate(raw):
        item_where = f"{where}[{index}]"
        data = _mapping(item, item_where)
        _no_unknown(data, {"code", "version", "count"}, item_where)
        version = _text(_required(data, "version", item_where), f"{item_where}.version")
        if not _VERSION.fullmatch(version):
            raise HardeningError(f"{item_where}.version has an invalid format")
        result.append(
            RepairCount(
                code=_slug(_required(data, "code", item_where), f"{item_where}.code"),
                version=version,
                count=_count(
                    _required(data, "count", item_where),
                    f"{item_where}.count",
                    positive=True,
                ),
            )
        )
    keys = [(item.code, item.version) for item in result]
    if len(set(keys)) != len(keys):
        raise HardeningError(f"{where} has duplicate repair code/version entries")
    return tuple(sorted(result, key=lambda item: (item.code, item.version)))


def _parse_models(value: Any, where: str) -> tuple[ModelFingerprint, ...]:
    raw = _list(value, where)
    result: list[ModelFingerprint] = []
    for index, item in enumerate(raw):
        item_where = f"{where}[{index}]"
        data = _mapping(item, item_where)
        _no_unknown(data, {"phase", "provider", "model", "fingerprint"}, item_where)
        result.append(
            ModelFingerprint(
                phase=_slug(_required(data, "phase", item_where), f"{item_where}.phase"),
                provider=_slug(
                    _required(data, "provider", item_where), f"{item_where}.provider"
                ),
                model=_text(_required(data, "model", item_where), f"{item_where}.model"),
                fingerprint=_fingerprint(
                    _required(data, "fingerprint", item_where),
                    f"{item_where}.fingerprint",
                ),
            )
        )
    keys = [(item.phase, item.provider, item.model) for item in result]
    if len(set(keys)) != len(keys):
        raise HardeningError(f"{where} has duplicate phase/provider/model entries")
    return tuple(sorted(result, key=lambda item: (item.phase, item.provider, item.model)))


def _parse_prompts(value: Any, where: str) -> tuple[PromptFingerprint, ...]:
    raw = _list(value, where)
    result: list[PromptFingerprint] = []
    for index, item in enumerate(raw):
        item_where = f"{where}[{index}]"
        data = _mapping(item, item_where)
        _no_unknown(data, {"phase", "kind", "fingerprint"}, item_where)
        result.append(
            PromptFingerprint(
                phase=_slug(_required(data, "phase", item_where), f"{item_where}.phase"),
                kind=_slug(_required(data, "kind", item_where), f"{item_where}.kind"),
                fingerprint=_fingerprint(
                    _required(data, "fingerprint", item_where),
                    f"{item_where}.fingerprint",
                ),
            )
        )
    keys = [(item.phase, item.kind) for item in result]
    if len(set(keys)) != len(keys):
        raise HardeningError(f"{where} has duplicate phase/kind entries")
    return tuple(sorted(result, key=lambda item: (item.phase, item.kind)))


def _parse_pipeline(value: Any, where: str) -> tuple[tuple[str, bool], ...]:
    data = _mapping(value, where)
    _no_unknown(data, set(PILOT_STEPS), where)
    return tuple(
        (name, _bool(_required(data, name, where), f"{where}.{name}"))
        for name in PILOT_STEPS
    )


def _parse_pilot(value: Any, path: Path, root: Path) -> Pilot:
    where = _where(path, root)
    data = _mapping(value, where)
    allowed = {
        "version",
        "id",
        "source_fingerprint",
        "source_archetype",
        "redistributable",
        "unit_oracle_id",
        "eval_case_id",
        "counts",
        "corrections",
        "repair_applications",
        "false_positive_repairs",
        "model_fingerprints",
        "prompt_fingerprints",
        "finding_ids",
        "pipeline",
        "notes",
    }
    _no_unknown(data, allowed, where)
    version = _required(data, "version", where)
    _schema_version(version, f"{where}.version")
    pilot_id = _slug(_required(data, "id", where), f"{where}.id")
    if path.stem != pilot_id:
        raise HardeningError(f"{where}.id must match its filename stem {path.stem!r}")
    pilot = Pilot(
        path=path,
        relative_path=where,
        id=pilot_id,
        source_fingerprint=_fingerprint(
            _required(data, "source_fingerprint", where), f"{where}.source_fingerprint"
        ),
        source_archetype=_slug(
            _required(data, "source_archetype", where), f"{where}.source_archetype"
        ),
        redistributable=_bool(
            _required(data, "redistributable", where), f"{where}.redistributable"
        ),
        unit_oracle_id=_slug(
            _required(data, "unit_oracle_id", where), f"{where}.unit_oracle_id"
        ),
        eval_case_id=_slug(
            _required(data, "eval_case_id", where), f"{where}.eval_case_id"
        ),
        counts=_parse_counts(_required(data, "counts", where), f"{where}.counts"),
        corrections=_parse_corrections(
            _required(data, "corrections", where), f"{where}.corrections"
        ),
        repair_applications=_parse_repairs(
            _required(data, "repair_applications", where),
            f"{where}.repair_applications",
        ),
        false_positive_repairs=_parse_repairs(
            _required(data, "false_positive_repairs", where),
            f"{where}.false_positive_repairs",
        ),
        model_fingerprints=_parse_models(
            _required(data, "model_fingerprints", where),
            f"{where}.model_fingerprints",
        ),
        prompt_fingerprints=_parse_prompts(
            _required(data, "prompt_fingerprints", where),
            f"{where}.prompt_fingerprints",
        ),
        finding_ids=_slug_list(
            _required(data, "finding_ids", where), f"{where}.finding_ids"
        ),
        pipeline=_parse_pipeline(
            _required(data, "pipeline", where), f"{where}.pipeline"
        ),
        notes=_optional_text(data, "notes", where),
    )
    if (
        pilot.corrections.systemic or pilot.false_positive_repairs
    ) and not pilot.finding_ids:
        raise HardeningError(
            f"{where}.finding_ids must link every systemic correction or repair false positive"
        )
    if pilot.complete and not pilot.model_fingerprints:
        raise HardeningError(f"{where}: a complete pilot needs model_fingerprints")
    if pilot.complete and not pilot.prompt_fingerprints:
        raise HardeningError(f"{where}: a complete pilot needs prompt_fingerprints")
    return pilot


def load_pilots(
    root: Path, *, known_finding_ids: set[str] | None = None
) -> tuple[Pilot, ...]:
    root = root.resolve()
    directory = root / "quality" / "pilots"
    _reject_symlink_components(directory, root)
    try:
        details = directory.stat()
    except FileNotFoundError as exc:
        raise HardeningError("Hardening pilot directory not found: quality/pilots") from exc
    except OSError as exc:
        raise HardeningError(
            f"Could not inspect quality/pilots: {exc.strerror or exc}"
        ) from exc
    if not stat.S_ISDIR(details.st_mode):
        raise HardeningError("Hardening path quality/pilots is not a directory")
    try:
        paths = sorted(
            [
                path
                for path in directory.iterdir()
                if path.suffix.lower() in {".yaml", ".yml"}
            ],
            key=lambda path: path.name,
        )
    except OSError as exc:
        raise HardeningError(
            f"Could not list quality/pilots: {exc.strerror or exc}"
        ) from exc
    pilots = tuple(_parse_pilot(_read_yaml(path, root), path, root) for path in paths)
    ids = [pilot.id for pilot in pilots]
    if len(set(ids)) != len(ids):
        raise HardeningError("quality/pilots has duplicate pilot IDs")
    if known_finding_ids is not None:
        for pilot in pilots:
            missing = sorted(set(pilot.finding_ids) - known_finding_ids)
            if missing:
                names = ", ".join(missing)
                raise HardeningError(
                    f"{pilot.relative_path} links unknown finding ID(s): {names}"
                )
    return tuple(sorted(pilots, key=lambda item: item.id))


def build_status(root: Path) -> HardeningStatus:
    root = root.resolve()
    catalog = load_findings(root)
    finding_ids = {finding.id for finding in catalog.findings}
    pilots = load_pilots(root, known_finding_ids=finding_ids)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    pilot_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pilot in pilots:
        for item in pilot.false_positive_repairs:
            key = (item.code, item.version)
            totals[key] += item.count
            pilot_ids[key].add(pilot.id)
    false_positives = tuple(
        RepairFalsePositive(
            code=code,
            version=version,
            count=totals[(code, version)],
            pilot_ids=tuple(sorted(pilot_ids[(code, version)])),
        )
        for code, version in sorted(totals)
    )
    return HardeningStatus(
        findings=catalog.findings,
        pilots=pilots,
        false_positives=false_positives,
    )


def status_payload(report: HardeningStatus) -> dict[str, Any]:
    states = Counter(finding.state for finding in report.findings)
    open_findings = [finding for finding in report.findings if finding.state == "open"]
    deferred = [finding for finding in report.findings if finding.state == "deferred"]
    recurrences = [finding for finding in report.findings if finding.recurrences]
    incomplete = [pilot for pilot in report.pilots if not pilot.complete]
    covered = sorted({pilot.source_archetype for pilot in report.pilots})
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog": report.catalog_path,
        "pilots_directory": report.pilots_path,
        "findings": {
            "total": len(report.findings),
            "by_state": {state: states.get(state, 0) for state in FINDING_STATES},
            "open": [
                {
                    "id": item.id,
                    "pipeline_stage": item.pipeline_stage,
                    "source_archetypes": list(item.source_archetypes),
                    "symptom": item.symptom,
                }
                for item in open_findings
            ],
            "deferred": [
                {
                    "id": item.id,
                    "pipeline_stage": item.pipeline_stage,
                    "reason": item.deferral_reason,
                }
                for item in deferred
            ],
            "recurrences": [
                {"id": item.id, "count": len(item.recurrences)} for item in recurrences
            ],
        },
        "pilots": {
            "total": len(report.pilots),
            "complete": sum(pilot.complete for pilot in report.pilots),
            "incomplete": [
                {
                    "id": pilot.id,
                    "path": pilot.relative_path,
                    "source_archetype": pilot.source_archetype,
                    "missing_steps": list(pilot.missing_steps),
                }
                for pilot in incomplete
            ],
            "corrections": {
                "content_specific": sum(
                    pilot.corrections.content_specific for pilot in report.pilots
                ),
                "systemic": sum(pilot.corrections.systemic for pilot in report.pilots),
            },
            "repair_applications": sum(
                item.count
                for pilot in report.pilots
                for item in pilot.repair_applications
            ),
            "repair_false_positives": [
                {
                    "code": item.code,
                    "version": item.version,
                    "count": item.count,
                    "pilot_ids": list(item.pilot_ids),
                }
                for item in report.false_positives
            ],
        },
        "source_archetypes": {
            "defined": list(M7_SOURCE_ARCHETYPES),
            "covered": covered,
            "empty": [item for item in M7_SOURCE_ARCHETYPES if item not in covered],
        },
    }


def format_status(report: HardeningStatus) -> list[str]:
    payload = status_payload(report)
    finding_data = payload["findings"]
    pilot_data = payload["pilots"]
    states = finding_data["by_state"]
    lines = [
        "Hardening status",
        (
            f"Findings: {finding_data['total']} "
            f"(open {states['open']}, fixed {states['fixed']}, "
            f"deferred {states['deferred']}, accepted-risk {states['accepted-risk']})"
        ),
    ]
    if finding_data["open"]:
        lines.append(
            "Open findings: "
            + ", ".join(item["id"] for item in finding_data["open"])
        )
    else:
        lines.append("Open findings: none")
    if finding_data["deferred"]:
        lines.append(
            "Deferred findings: "
            + ", ".join(item["id"] for item in finding_data["deferred"])
        )
    else:
        lines.append("Deferred findings: none")
    if finding_data["recurrences"]:
        lines.append(
            "Recurrences: "
            + ", ".join(
                f"{item['id']} {item['count']}" for item in finding_data["recurrences"]
            )
        )
    else:
        lines.append("Recurrences: none")
    lines.append(
        f"Pilots: {pilot_data['total']} "
        f"({pilot_data['complete']} complete, {len(pilot_data['incomplete'])} incomplete)"
    )
    if pilot_data["incomplete"]:
        lines.extend(
            f"  {item['id']}: missing {', '.join(item['missing_steps'])}"
            for item in pilot_data["incomplete"]
        )
    corrections = pilot_data["corrections"]
    lines.append(
        "Corrections: "
        f"content-specific {corrections['content_specific']}, systemic {corrections['systemic']}"
    )
    if pilot_data["repair_false_positives"]:
        lines.append(
            "Repair false positives: "
            + ", ".join(
                f"{item['code']}@{item['version']} {item['count']}"
                for item in pilot_data["repair_false_positives"]
            )
        )
    else:
        lines.append("Repair false positives: none")
    empty = payload["source_archetypes"]["empty"]
    lines.append(
        "Empty source-archetype cells: " + (", ".join(empty) if empty else "none")
    )
    return lines
