"""Strict loading and read-only status for deck-driven hardening evidence.

The module owns findings, pilot reports, human unit oracles, minimized cases,
their content fingerprints, and every reciprocal link between them. Offline
execution stays in :mod:`japanese_anki.hardening_replay`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
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
ORACLE_TYPES = ("exhaustive", "selection")
UNIT_DISPOSITIONS = ("candidate", "duplicate", "non-vocabulary", "unreadable")
CASE_PURPOSES = ("regression", "coverage")
RUNNER_BOUNDARIES = {
    "candidate-response": "extraction-normalization",
    "staging-promote": "staging-promote",
    "validation-qc": "validation-qc",
    "render-build": "render-build",
    "dictionary-enrichment": "dictionary-enrichment",
    "ai-enrichment": "ai-enrichment",
}

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
class OracleUnit:
    page: int
    section: str
    ordinal: int
    context_fingerprint: str
    disposition: str

    @property
    def key(self) -> tuple[int, str, int]:
        return self.page, self.section, self.ordinal


@dataclass(frozen=True, slots=True)
class SelectionTarget:
    identity: str
    locator: str


@dataclass(frozen=True, slots=True)
class OracleApproval:
    authority: str
    oracle_id: str
    source_fingerprint: str
    oracle_type: str
    oracle_content_fingerprint: str
    selection_rubric: str | None
    approved_at: str


@dataclass(frozen=True, slots=True)
class UnitOracle:
    path: Path
    relative_path: str
    id: str
    source_fingerprint: str
    type: str
    case_ids: tuple[str, ...]
    units: tuple[OracleUnit, ...]
    targets: tuple[SelectionTarget, ...]
    selection_rubric: str | None
    approval: OracleApproval | None

    @property
    def approved(self) -> bool:
        return self.approval is not None


@dataclass(frozen=True, slots=True)
class CaseFixture:
    path: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class RedistributionApproval:
    authority: str
    case_id: str
    artifact_fingerprint: str
    basis: str
    approved_at: str


@dataclass(frozen=True, slots=True)
class LiveEvalApproval:
    authority: str
    case_id: str
    source_fingerprint: str
    provider: str
    models: tuple[str, ...]
    purpose: str
    reason: str
    approved_at: str


@dataclass(frozen=True, slots=True)
class LiveEvalRequest:
    provider: str
    models: tuple[str, ...]
    purpose: str
    approval: LiveEvalApproval | None

    @property
    def consented(self) -> bool:
        return self.approval is not None


@dataclass(frozen=True, slots=True)
class CaseSource:
    kind: str
    path: str
    fingerprint: str
    basis: str | None = None
    redistribution_approval: RedistributionApproval | None = None
    live_eval: LiveEvalRequest | None = None


@dataclass(frozen=True, slots=True)
class HardeningCase:
    path: Path
    relative_path: str
    id: str
    purpose: str
    gating: bool
    runner: str
    pipeline_boundary: str
    source_archetype: str
    redistributable: bool
    fixture_basis: str
    fixtures: tuple[CaseFixture, ...]
    runner_input: str
    oracle: str
    finding_id: str | None
    pilot_id: str | None
    unit_oracle_id: str | None
    source: CaseSource | None


@dataclass(frozen=True, slots=True)
class HardeningRepository:
    catalog: FindingsCatalog
    pilots: tuple[Pilot, ...]
    oracles: tuple[UnitOracle, ...]
    cases: tuple[HardeningCase, ...]


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
    oracles: tuple[UnitOracle, ...] = ()
    cases: tuple[HardeningCase, ...] = ()
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


def _date_text(value: Any, where: str) -> str:
    if isinstance(value, date):
        result = value.isoformat()
    else:
        result = _text(value, where)
        try:
            parsed = date.fromisoformat(result)
        except ValueError as exc:
            raise HardeningError(f"{where} must be an ISO date") from exc
        if parsed.isoformat() != result:
            raise HardeningError(f"{where} must use YYYY-MM-DD")
    if len(result) != 10:
        raise HardeningError(f"{where} must use YYYY-MM-DD")
    return result


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


def _relative_file(value: Any, where: str) -> str:
    result = _locator(value, where)
    parts = PurePosixPath(result).parts
    if not parts or any(part in {"", "."} for part in parts):
        raise HardeningError(f"{where} must name a repository-relative file")
    return result


def _authority(value: Any, where: str) -> str:
    result = _text(value, where)
    if result != "repository-owner":
        raise HardeningError(f"{where} must be 'repository-owner'")
    return result


def _verified_file(
    path: Path,
    root: Path,
    *,
    expected: str | None = None,
    return_bytes: bool = False,
) -> tuple[str, bytes | None]:
    """Hash a regular file without following a final symlink."""
    root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    _reject_symlink_components(candidate, root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except FileNotFoundError as exc:
        raise HardeningError(f"Hardening file not found: {_where(candidate, root)}") from exc
    except OSError as exc:
        raise HardeningError(
            f"Could not open {_where(candidate, root)}: {exc.strerror or exc}"
        ) from exc
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HardeningError(
                f"Hardening path {_where(candidate, root)} is not a regular file"
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if return_bytes:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise HardeningError(
            f"Could not read {_where(candidate, root)}: {exc.strerror or exc}"
        ) from exc
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise HardeningError(
            f"Hardening file {_where(candidate, root)} changed while it was read"
        )
    fingerprint = digest.hexdigest()
    if expected is not None and fingerprint != expected:
        raise HardeningError(
            f"Hardening file {_where(candidate, root)} has SHA-256 {fingerprint}; "
            f"expected {expected}"
        )
    return fingerprint, b"".join(chunks) if return_bytes else None


def verified_file_fingerprint(path: Path, root: Path) -> str:
    return _verified_file(path, root)[0]


def read_verified_bytes(path: Path, root: Path, expected: str) -> bytes:
    return _verified_file(path, root, expected=expected, return_bytes=True)[1] or b""


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
    unknown_archetypes = sorted(
        set(finding.source_archetypes) - set(M7_SOURCE_ARCHETYPES)
    )
    if unknown_archetypes:
        raise HardeningError(
            f"{where}.source_archetypes has unknown value(s): "
            + ", ".join(unknown_archetypes)
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
    if pilot.source_archetype not in M7_SOURCE_ARCHETYPES:
        raise HardeningError(
            f"{where}.source_archetype must be one of "
            f"{', '.join(M7_SOURCE_ARCHETYPES)}"
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


def oracle_content_fingerprint(oracle: UnitOracle) -> str:
    """Fingerprint reviewed oracle content, but not its approval."""
    payload: dict[str, Any] = {
        "id": oracle.id,
        "source_fingerprint": oracle.source_fingerprint,
        "type": oracle.type,
        "case_ids": sorted(oracle.case_ids),
    }
    if oracle.type == "exhaustive":
        payload["units"] = [
            {
                "page": unit.page,
                "section": unit.section,
                "ordinal": unit.ordinal,
                "context_fingerprint": unit.context_fingerprint,
                "disposition": unit.disposition,
            }
            for unit in oracle.units
        ]
    else:
        payload["targets"] = [
            {"identity": target.identity, "locator": target.locator}
            for target in sorted(
                oracle.targets, key=lambda item: (item.identity, item.locator)
            )
        ]
        payload["selection_rubric"] = oracle.selection_rubric
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_oracle_approval(value: Any, where: str) -> OracleApproval:
    data = _mapping(value, where)
    allowed = {
        "authority",
        "oracle_id",
        "source_fingerprint",
        "oracle_type",
        "oracle_content_fingerprint",
        "selection_rubric",
        "approved_at",
    }
    _no_unknown(data, allowed, where)
    return OracleApproval(
        authority=_authority(
            _required(data, "authority", where), f"{where}.authority"
        ),
        oracle_id=_slug(
            _required(data, "oracle_id", where), f"{where}.oracle_id"
        ),
        source_fingerprint=_fingerprint(
            _required(data, "source_fingerprint", where),
            f"{where}.source_fingerprint",
        ),
        oracle_type=_text(
            _required(data, "oracle_type", where), f"{where}.oracle_type"
        ),
        oracle_content_fingerprint=_fingerprint(
            _required(data, "oracle_content_fingerprint", where),
            f"{where}.oracle_content_fingerprint",
        ),
        selection_rubric=_optional_text(data, "selection_rubric", where),
        approved_at=_date_text(
            _required(data, "approved_at", where), f"{where}.approved_at"
        ),
    )


def _parse_oracle(value: Any, path: Path, root: Path) -> UnitOracle:
    where = _where(path, root)
    data = _mapping(value, where)
    allowed = {
        "version",
        "id",
        "source_fingerprint",
        "type",
        "case_ids",
        "units",
        "targets",
        "selection_rubric",
        "approval",
    }
    _no_unknown(data, allowed, where)
    _schema_version(_required(data, "version", where), f"{where}.version")
    oracle_id = _slug(_required(data, "id", where), f"{where}.id")
    if path.stem != oracle_id:
        raise HardeningError(f"{where}.id must match its filename stem {path.stem!r}")
    oracle_type = _text(_required(data, "type", where), f"{where}.type")
    if oracle_type not in ORACLE_TYPES:
        raise HardeningError(f"{where}.type must be one of {', '.join(ORACLE_TYPES)}")
    units: list[OracleUnit] = []
    targets: list[SelectionTarget] = []
    rubric = _optional_text(data, "selection_rubric", where)
    if oracle_type == "exhaustive":
        if "targets" in data or rubric is not None:
            raise HardeningError(
                f"{where}: exhaustive cannot have targets or selection_rubric"
            )
        raw_units = _list(_required(data, "units", where), f"{where}.units")
        if not raw_units:
            raise HardeningError(f"{where}.units must not be empty")
        for index, item in enumerate(raw_units):
            item_where = f"{where}.units[{index}]"
            unit = _mapping(item, item_where)
            _no_unknown(
                unit,
                {"page", "section", "ordinal", "context_fingerprint", "disposition"},
                item_where,
            )
            disposition = _text(
                _required(unit, "disposition", item_where),
                f"{item_where}.disposition",
            )
            if disposition not in UNIT_DISPOSITIONS:
                raise HardeningError(
                    f"{item_where}.disposition must be one of "
                    f"{', '.join(UNIT_DISPOSITIONS)}"
                )
            units.append(
                OracleUnit(
                    page=_count(
                        _required(unit, "page", item_where),
                        f"{item_where}.page",
                        positive=True,
                    ),
                    section=_slug(
                        _required(unit, "section", item_where),
                        f"{item_where}.section",
                    ),
                    ordinal=_count(
                        _required(unit, "ordinal", item_where),
                        f"{item_where}.ordinal",
                        positive=True,
                    ),
                    context_fingerprint=_fingerprint(
                        _required(unit, "context_fingerprint", item_where),
                        f"{item_where}.context_fingerprint",
                    ),
                    disposition=disposition,
                )
            )
        keys = [unit.key for unit in units]
        if len(set(keys)) != len(keys):
            raise HardeningError(f"{where}.units has duplicate unit keys")
        if keys != sorted(keys):
            raise HardeningError(
                f"{where}.units must use page, section, ordinal order"
            )
    else:
        if "units" in data:
            raise HardeningError(f"{where}: selection cannot have units")
        if rubric is None:
            raise HardeningError(f"{where}: selection requires selection_rubric")
        raw_targets = _list(
            _required(data, "targets", where), f"{where}.targets"
        )
        if not raw_targets:
            raise HardeningError(f"{where}.targets must not be empty")
        for index, item in enumerate(raw_targets):
            item_where = f"{where}.targets[{index}]"
            target = _mapping(item, item_where)
            _no_unknown(target, {"identity", "locator"}, item_where)
            targets.append(
                SelectionTarget(
                    identity=_text(
                        _required(target, "identity", item_where),
                        f"{item_where}.identity",
                    ),
                    locator=_locator(
                        _required(target, "locator", item_where),
                        f"{item_where}.locator",
                    ),
                )
            )
        pairs = [(target.identity, target.locator) for target in targets]
        if len(set(pairs)) != len(pairs):
            raise HardeningError(f"{where}.targets has duplicate targets")
    oracle = UnitOracle(
        path=path,
        relative_path=where,
        id=oracle_id,
        source_fingerprint=_fingerprint(
            _required(data, "source_fingerprint", where),
            f"{where}.source_fingerprint",
        ),
        type=oracle_type,
        case_ids=tuple(
            sorted(_slug_list(data.get("case_ids", []), f"{where}.case_ids"))
        ),
        units=tuple(units),
        targets=tuple(targets),
        selection_rubric=rubric,
        approval=(
            _parse_oracle_approval(data["approval"], f"{where}.approval")
            if "approval" in data
            else None
        ),
    )
    if oracle.approval is not None:
        approval = oracle.approval
        expected = oracle_content_fingerprint(oracle)
        repeated = (
            approval.oracle_id == oracle.id
            and approval.source_fingerprint == oracle.source_fingerprint
            and approval.oracle_type == oracle.type
            and approval.selection_rubric == oracle.selection_rubric
        )
        if not repeated:
            raise HardeningError(
                f"{where}.approval must repeat the oracle ID, source fingerprint, "
                "type, and selection rubric exactly"
            )
        if approval.oracle_content_fingerprint != expected:
            raise HardeningError(
                f"{where}.approval is stale: oracle content fingerprint must be {expected}"
            )
    return oracle


def _yaml_directory(root: Path, relative: str) -> tuple[Path, ...]:
    directory = root / relative
    _reject_symlink_components(directory, root)
    try:
        details = directory.stat()
    except FileNotFoundError as exc:
        raise HardeningError(f"Hardening directory not found: {relative}") from exc
    except OSError as exc:
        raise HardeningError(
            f"Could not inspect {relative}: {exc.strerror or exc}"
        ) from exc
    if not stat.S_ISDIR(details.st_mode):
        raise HardeningError(f"Hardening path {relative} is not a directory")
    try:
        return tuple(
            sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.suffix.lower() in {".yaml", ".yml"}
                ),
                key=lambda path: path.name,
            )
        )
    except OSError as exc:
        raise HardeningError(
            f"Could not list {relative}: {exc.strerror or exc}"
        ) from exc


def load_oracles(root: Path) -> tuple[UnitOracle, ...]:
    root = root.resolve()
    oracles = tuple(
        _parse_oracle(_read_yaml(path, root), path, root)
        for path in _yaml_directory(root, "quality/oracles")
    )
    ids = [oracle.id for oracle in oracles]
    if len(set(ids)) != len(ids):
        raise HardeningError("quality/oracles has duplicate oracle IDs")
    return tuple(sorted(oracles, key=lambda item: item.id))


def _parse_redistribution_approval(
    value: Any, where: str
) -> RedistributionApproval:
    data = _mapping(value, where)
    allowed = {
        "authority",
        "case_id",
        "artifact_fingerprint",
        "basis",
        "approved_at",
    }
    _no_unknown(data, allowed, where)
    return RedistributionApproval(
        authority=_authority(
            _required(data, "authority", where), f"{where}.authority"
        ),
        case_id=_slug(_required(data, "case_id", where), f"{where}.case_id"),
        artifact_fingerprint=_fingerprint(
            _required(data, "artifact_fingerprint", where),
            f"{where}.artifact_fingerprint",
        ),
        basis=_text(_required(data, "basis", where), f"{where}.basis"),
        approved_at=_date_text(
            _required(data, "approved_at", where), f"{where}.approved_at"
        ),
    )


def _parse_live_approval(value: Any, where: str) -> LiveEvalApproval:
    data = _mapping(value, where)
    allowed = {
        "authority",
        "case_id",
        "source_fingerprint",
        "provider",
        "models",
        "purpose",
        "reason",
        "approved_at",
    }
    _no_unknown(data, allowed, where)
    models = tuple(
        _text(item, f"{where}.models[{index}]")
        for index, item in enumerate(
            _list(_required(data, "models", where), f"{where}.models")
        )
    )
    if not models:
        raise HardeningError(f"{where}.models must not be empty")
    return LiveEvalApproval(
        authority=_authority(
            _required(data, "authority", where), f"{where}.authority"
        ),
        case_id=_slug(_required(data, "case_id", where), f"{where}.case_id"),
        source_fingerprint=_fingerprint(
            _required(data, "source_fingerprint", where),
            f"{where}.source_fingerprint",
        ),
        provider=_slug(
            _required(data, "provider", where), f"{where}.provider"
        ),
        models=_unique(models, f"{where}.models"),
        purpose=_text(_required(data, "purpose", where), f"{where}.purpose"),
        reason=_text(_required(data, "reason", where), f"{where}.reason"),
        approved_at=_date_text(
            _required(data, "approved_at", where), f"{where}.approved_at"
        ),
    )


def _parse_live_request(value: Any, where: str) -> LiveEvalRequest:
    data = _mapping(value, where)
    _no_unknown(data, {"provider", "models", "purpose", "approval"}, where)
    models = tuple(
        _text(item, f"{where}.models[{index}]")
        for index, item in enumerate(
            _list(_required(data, "models", where), f"{where}.models")
        )
    )
    if not models:
        raise HardeningError(f"{where}.models must not be empty")
    purpose = _text(_required(data, "purpose", where), f"{where}.purpose")
    if purpose != "hardening-eval":
        raise HardeningError(f"{where}.purpose must be 'hardening-eval'")
    return LiveEvalRequest(
        provider=_slug(_required(data, "provider", where), f"{where}.provider"),
        models=_unique(models, f"{where}.models"),
        purpose=purpose,
        approval=(
            _parse_live_approval(data["approval"], f"{where}.approval")
            if "approval" in data
            else None
        ),
    )


def _parse_case_source(
    value: Any,
    where: str,
    *,
    case_id: str,
    case_dir: Path,
    root: Path,
    redistributable: bool,
    fixture_basis: str,
    fixture_paths: set[str],
) -> CaseSource:
    data = _mapping(value, where)
    allowed = {
        "kind",
        "path",
        "fingerprint",
        "basis",
        "redistribution_approval",
        "live_eval",
    }
    _no_unknown(data, allowed, where)
    kind = _text(_required(data, "kind", where), f"{where}.kind")
    fingerprint = _fingerprint(
        _required(data, "fingerprint", where), f"{where}.fingerprint"
    )
    if kind == "bundled_fixture":
        path = _relative_file(_required(data, "path", where), f"{where}.path")
        if path not in fixture_paths:
            raise HardeningError(f"{where}.path must name a declared fixture")
        if not redistributable:
            raise HardeningError(f"{where}: bundled_fixture requires redistributable true")
        if "live_eval" in data:
            raise HardeningError(f"{where}: bundled_fixture cannot have live_eval")
        basis = _text(_required(data, "basis", where), f"{where}.basis")
        _verified_file(case_dir / path, case_dir, expected=fingerprint)
        approval = (
            _parse_redistribution_approval(
                data["redistribution_approval"],
                f"{where}.redistribution_approval",
            )
            if "redistribution_approval" in data
            else None
        )
        if fixture_basis == "agent-created-synthetic":
            if basis != "agent-created-synthetic":
                raise HardeningError(
                    f"{where}.basis must be 'agent-created-synthetic' for a "
                    "synthetic fixture"
                )
            if approval is not None:
                raise HardeningError(
                    f"{where}: agent-created-synthetic cannot have redistribution approval"
                )
        elif basis == "agent-created-synthetic":
            raise HardeningError(
                f"{where}.basis must state the license or permission for "
                f"{fixture_basis} material"
            )
        elif approval is None:
            raise HardeningError(
                f"{where}: {basis} requires repository-owner redistribution approval"
            )
        elif (
            approval.case_id != case_id
            or approval.artifact_fingerprint != fingerprint
            or approval.basis != basis
        ):
            raise HardeningError(
                f"{where}.redistribution_approval must match the case, artifact, and basis"
            )
        return CaseSource(
            kind=kind,
            path=path,
            fingerprint=fingerprint,
            basis=basis,
            redistribution_approval=approval,
        )
    if kind != "private_inbox_ref":
        raise HardeningError(
            f"{where}.kind must be 'bundled_fixture' or 'private_inbox_ref'"
        )
    if redistributable:
        raise HardeningError(f"{where}: private_inbox_ref requires redistributable false")
    if "basis" in data or "redistribution_approval" in data:
        raise HardeningError(
            f"{where}: private_inbox_ref cannot have bundled redistribution fields"
        )
    path = _relative_file(_required(data, "path", where), f"{where}.path")
    if not PurePosixPath(path).is_relative_to(PurePosixPath("data/inbox")):
        raise HardeningError(f"{where}.path must be under data/inbox")
    _verified_file(root / path, root / "data" / "inbox", expected=fingerprint)
    live_eval = _parse_live_request(
        _required(data, "live_eval", where), f"{where}.live_eval"
    )
    if live_eval.approval is not None:
        approval = live_eval.approval
        if (
            approval.case_id != case_id
            or approval.source_fingerprint != fingerprint
            or approval.provider != live_eval.provider
            or approval.models != live_eval.models
            or approval.purpose != live_eval.purpose
        ):
            raise HardeningError(
                f"{where}.live_eval.approval must match the case, source, provider, "
                "models, and purpose"
            )
    return CaseSource(
        kind=kind,
        path=path,
        fingerprint=fingerprint,
        live_eval=live_eval,
    )


def _parse_case(value: Any, path: Path, root: Path) -> HardeningCase:
    where = _where(path, root)
    data = _mapping(value, where)
    allowed = {
        "version",
        "id",
        "purpose",
        "gating",
        "runner",
        "pipeline_boundary",
        "source_archetype",
        "redistributable",
        "fixture_basis",
        "fixtures",
        "runner_input",
        "oracle",
        "finding_id",
        "pilot_id",
        "unit_oracle_id",
        "source",
    }
    _no_unknown(data, allowed, where)
    _schema_version(_required(data, "version", where), f"{where}.version")
    case_id = _slug(_required(data, "id", where), f"{where}.id")
    if path.parent.name != case_id:
        raise HardeningError(
            f"{where}.id must match its directory name {path.parent.name!r}"
        )
    purpose = _text(_required(data, "purpose", where), f"{where}.purpose")
    if purpose not in CASE_PURPOSES:
        raise HardeningError(
            f"{where}.purpose must be one of {', '.join(CASE_PURPOSES)}"
        )
    runner = _text(_required(data, "runner", where), f"{where}.runner")
    if runner not in RUNNER_BOUNDARIES:
        raise HardeningError(f"{where}.runner names unknown runner {runner!r}")
    boundary = _slug(
        _required(data, "pipeline_boundary", where),
        f"{where}.pipeline_boundary",
    )
    if boundary != RUNNER_BOUNDARIES[runner]:
        raise HardeningError(
            f"{where}.pipeline_boundary must be {RUNNER_BOUNDARIES[runner]!r} "
            f"for runner {runner!r}"
        )
    fixtures: list[CaseFixture] = []
    raw_fixtures = _list(_required(data, "fixtures", where), f"{where}.fixtures")
    if not raw_fixtures:
        raise HardeningError(f"{where}.fixtures must not be empty")
    for index, item in enumerate(raw_fixtures):
        item_where = f"{where}.fixtures[{index}]"
        fixture = _mapping(item, item_where)
        _no_unknown(fixture, {"path", "fingerprint"}, item_where)
        fixtures.append(
            CaseFixture(
                path=_relative_file(
                    _required(fixture, "path", item_where), f"{item_where}.path"
                ),
                fingerprint=_fingerprint(
                    _required(fixture, "fingerprint", item_where),
                    f"{item_where}.fingerprint",
                ),
            )
        )
    fixture_paths = [fixture.path for fixture in fixtures]
    if len(set(fixture_paths)) != len(fixture_paths):
        raise HardeningError(f"{where}.fixtures has duplicate paths")
    if "case.yaml" in fixture_paths:
        raise HardeningError(f"{where}.fixtures cannot declare case.yaml")
    for fixture in fixtures:
        _verified_file(
            path.parent / fixture.path,
            path.parent,
            expected=fixture.fingerprint,
        )
    try:
        actual_files = {
            item.relative_to(path.parent).as_posix()
            for item in path.parent.rglob("*")
            if item.is_file() or item.is_symlink()
        }
    except OSError as exc:
        raise HardeningError(f"Could not list {path.parent}: {exc.strerror or exc}") from exc
    expected_files = {"case.yaml", *fixture_paths}
    if actual_files != expected_files:
        undeclared = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        details = []
        if undeclared:
            details.append("undeclared: " + ", ".join(undeclared))
        if missing:
            details.append("missing: " + ", ".join(missing))
        raise HardeningError(f"{where} fixture set does not match files ({'; '.join(details)})")
    runner_input = _relative_file(
        _required(data, "runner_input", where), f"{where}.runner_input"
    )
    oracle_path = _relative_file(
        _required(data, "oracle", where), f"{where}.oracle"
    )
    for name, value_ in (("runner_input", runner_input), ("oracle", oracle_path)):
        if value_ not in set(fixture_paths):
            raise HardeningError(f"{where}.{name} must name a declared fixture")
    redistributable = _bool(
        _required(data, "redistributable", where), f"{where}.redistributable"
    )
    fixture_basis = _text(
        _required(data, "fixture_basis", where), f"{where}.fixture_basis"
    )
    if fixture_basis not in {
        "agent-created-synthetic",
        "owner-provided",
        "source-derived",
    }:
        raise HardeningError(f"{where}.fixture_basis has an invalid value")
    source = (
        _parse_case_source(
            data["source"],
            f"{where}.source",
            case_id=case_id,
            case_dir=path.parent,
            root=root,
            redistributable=redistributable,
            fixture_basis=fixture_basis,
            fixture_paths=set(fixture_paths),
        )
        if "source" in data
        else None
    )
    if fixture_basis != "agent-created-synthetic" and (
        source is None or source.kind != "bundled_fixture"
    ):
        raise HardeningError(
            f"{where}: non-synthetic fixtures require an approved bundled source"
        )
    finding_id = (
        _slug(data["finding_id"], f"{where}.finding_id")
        if "finding_id" in data
        else None
    )
    pilot_id = (
        _slug(data["pilot_id"], f"{where}.pilot_id")
        if "pilot_id" in data
        else None
    )
    unit_oracle_id = (
        _slug(data["unit_oracle_id"], f"{where}.unit_oracle_id")
        if "unit_oracle_id" in data
        else None
    )
    if purpose == "regression":
        if finding_id is None or pilot_id is not None:
            raise HardeningError(
                f"{where}: regression requires one finding_id and no pilot_id"
            )
    elif finding_id is not None or pilot_id is None or unit_oracle_id is None:
        raise HardeningError(
            f"{where}: coverage requires pilot_id and unit_oracle_id, and no finding_id"
        )
    if source is not None and unit_oracle_id is None:
        raise HardeningError(f"{where}: a source case requires unit_oracle_id")
    source_archetype = _slug(
        _required(data, "source_archetype", where),
        f"{where}.source_archetype",
    )
    if source_archetype not in M7_SOURCE_ARCHETYPES:
        raise HardeningError(
            f"{where}.source_archetype must be one of "
            f"{', '.join(M7_SOURCE_ARCHETYPES)}"
        )
    return HardeningCase(
        path=path,
        relative_path=where,
        id=case_id,
        purpose=purpose,
        gating=_bool(_required(data, "gating", where), f"{where}.gating"),
        runner=runner,
        pipeline_boundary=boundary,
        source_archetype=source_archetype,
        redistributable=redistributable,
        fixture_basis=fixture_basis,
        fixtures=tuple(sorted(fixtures, key=lambda item: item.path)),
        runner_input=runner_input,
        oracle=oracle_path,
        finding_id=finding_id,
        pilot_id=pilot_id,
        unit_oracle_id=unit_oracle_id,
        source=source,
    )


def load_cases(root: Path) -> tuple[HardeningCase, ...]:
    root = root.resolve()
    directory = root / "quality" / "cases"
    _reject_symlink_components(directory, root)
    try:
        entries = tuple(sorted(directory.iterdir(), key=lambda path: path.name))
    except FileNotFoundError as exc:
        raise HardeningError("Hardening case directory not found: quality/cases") from exc
    except OSError as exc:
        raise HardeningError(
            f"Could not list quality/cases: {exc.strerror or exc}"
        ) from exc
    cases: list[HardeningCase] = []
    for entry in entries:
        if entry.name == "README.md":
            continue
        if entry.is_symlink():
            raise HardeningError(f"Hardening case path {_where(entry, root)} is a symlink")
        if not entry.is_dir():
            raise HardeningError(
                f"Hardening case directory has unexpected entry {_where(entry, root)}"
            )
        manifest = entry / "case.yaml"
        cases.append(_parse_case(_read_yaml(manifest, root), manifest, root))
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise HardeningError("quality/cases has duplicate case IDs")
    return tuple(sorted(cases, key=lambda item: item.id))


def load_repository(root: Path) -> HardeningRepository:
    root = root.resolve()
    catalog = load_findings(root)
    finding_ids = {finding.id for finding in catalog.findings}
    pilots = load_pilots(root, known_finding_ids=finding_ids)
    oracles = load_oracles(root)
    cases = load_cases(root)
    findings_by_id = {finding.id: finding for finding in catalog.findings}
    pilots_by_id = {pilot.id: pilot for pilot in pilots}
    oracles_by_id = {oracle.id: oracle for oracle in oracles}
    cases_by_id = {case.id: case for case in cases}

    for finding in catalog.findings:
        for case_id in finding.case_ids:
            case = cases_by_id.get(case_id)
            if case is None:
                raise HardeningError(
                    f"Finding {finding.id!r} links unknown case ID {case_id!r}"
                )
            if case.finding_id != finding.id:
                raise HardeningError(
                    f"Finding {finding.id!r} and case {case.id!r} do not link both ways"
                )
            compatible_stages = {
                case.pipeline_boundary,
                case.runner,
                "extraction" if case.pipeline_boundary == "extraction-normalization" else "",
            }
            if finding.pipeline_stage not in compatible_stages:
                raise HardeningError(
                    f"Case {case.id!r} boundary does not match finding {finding.id!r}"
                )
            if case.source_archetype not in finding.source_archetypes:
                raise HardeningError(
                    f"Case {case.id!r} source archetype does not match finding "
                    f"{finding.id!r}"
                )
            if finding.state == "fixed" and not case.gating:
                raise HardeningError(
                    f"Fixed finding {finding.id!r} links non-gating case {case.id!r}"
                )
            if finding.state != "fixed" and case.gating:
                raise HardeningError(
                    f"Non-fixed finding {finding.id!r} cannot link gating case {case.id!r}"
                )
    for case in cases:
        if case.finding_id is not None:
            finding = findings_by_id.get(case.finding_id)
            if finding is None:
                raise HardeningError(
                    f"{case.relative_path} links unknown finding ID {case.finding_id!r}"
                )
            if case.id not in finding.case_ids:
                raise HardeningError(
                    f"Case {case.id!r} and finding {finding.id!r} do not link both ways"
                )
        if case.unit_oracle_id is not None:
            oracle = oracles_by_id.get(case.unit_oracle_id)
            if oracle is None:
                raise HardeningError(
                    f"{case.relative_path} links unknown oracle ID {case.unit_oracle_id!r}"
                )
            if not oracle.approved:
                raise HardeningError(
                    f"{case.relative_path} cannot bind draft oracle {oracle.id!r}"
                )
            if case.id not in oracle.case_ids:
                raise HardeningError(
                    f"Case {case.id!r} and oracle {oracle.id!r} do not link both ways"
                )
            if case.source is not None and (
                case.source.fingerprint != oracle.source_fingerprint
            ):
                raise HardeningError(
                    f"Case {case.id!r} source fingerprint does not match oracle "
                    f"{oracle.id!r}"
                )
        if case.purpose == "coverage":
            pilot = pilots_by_id.get(case.pilot_id or "")
            if pilot is None:
                raise HardeningError(
                    f"{case.relative_path} links unknown pilot ID {case.pilot_id!r}"
                )
            if pilot.eval_case_id != case.id:
                raise HardeningError(
                    f"Case {case.id!r} and pilot {pilot.id!r} do not link both ways"
                )
            if pilot.unit_oracle_id != case.unit_oracle_id:
                raise HardeningError(
                    f"Case {case.id!r} and pilot {pilot.id!r} name different oracles"
                )
            if (
                pilot.source_archetype != case.source_archetype
                or pilot.redistributable != case.redistributable
                or case.source is None
                or pilot.source_fingerprint != case.source.fingerprint
            ):
                raise HardeningError(
                    f"Case {case.id!r} source does not match pilot {pilot.id!r}"
                )
    for oracle in oracles:
        for case_id in oracle.case_ids:
            case = cases_by_id.get(case_id)
            if case is None:
                raise HardeningError(
                    f"Oracle {oracle.id!r} links unknown case ID {case_id!r}"
                )
            if case.unit_oracle_id != oracle.id:
                raise HardeningError(
                    f"Oracle {oracle.id!r} and case {case.id!r} do not link both ways"
                )
        if not oracle.approved and oracle.case_ids:
            raise HardeningError(
                f"Draft oracle {oracle.id!r} cannot link a hardening case"
            )
    for pilot in pilots:
        oracle = oracles_by_id.get(pilot.unit_oracle_id)
        if oracle is None:
            raise HardeningError(
                f"{pilot.relative_path} links unknown oracle ID {pilot.unit_oracle_id!r}"
            )
        if not oracle.approved:
            raise HardeningError(
                f"{pilot.relative_path} cannot bind draft oracle {oracle.id!r}"
            )
        case = cases_by_id.get(pilot.eval_case_id)
        if case is None:
            raise HardeningError(
                f"{pilot.relative_path} links unknown case ID {pilot.eval_case_id!r}"
            )
        if case.pilot_id != pilot.id or case.unit_oracle_id != oracle.id:
            raise HardeningError(
                f"Pilot {pilot.id!r}, case {case.id!r}, and oracle {oracle.id!r} "
                "do not link both ways"
            )
        if pilot.source_fingerprint != oracle.source_fingerprint:
            raise HardeningError(
                f"Pilot {pilot.id!r} source fingerprint does not match oracle "
                f"{oracle.id!r}"
            )
    return HardeningRepository(
        catalog=catalog,
        pilots=pilots,
        oracles=oracles,
        cases=cases,
    )


def build_status(root: Path) -> HardeningStatus:
    repository = load_repository(root)
    catalog = repository.catalog
    pilots = repository.pilots
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
        oracles=repository.oracles,
        cases=repository.cases,
    )


def status_payload(report: HardeningStatus) -> dict[str, Any]:
    states = Counter(finding.state for finding in report.findings)
    open_findings = [finding for finding in report.findings if finding.state == "open"]
    deferred = [finding for finding in report.findings if finding.state == "deferred"]
    recurrences = [finding for finding in report.findings if finding.recurrences]
    incomplete = [pilot for pilot in report.pilots if not pilot.complete]
    covered = sorted({pilot.source_archetype for pilot in report.pilots})
    regression_case_ids = {
        case.id for case in report.cases if case.purpose == "regression"
    }
    linked_case_ids = {
        case_id for finding in report.findings for case_id in finding.case_ids
    }
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
        "oracles": {
            "total": len(report.oracles),
            "approved": sum(oracle.approved for oracle in report.oracles),
            "draft": [oracle.id for oracle in report.oracles if not oracle.approved],
        },
        "cases": {
            "total": len(report.cases),
            "gating": sum(case.gating for case in report.cases),
            "regression": sum(case.purpose == "regression" for case in report.cases),
            "coverage": sum(case.purpose == "coverage" for case in report.cases),
            "finding_coverage": {
                "linked": len(linked_case_ids),
                "unlinked_cases": sorted(regression_case_ids - linked_case_ids),
                "uncovered_findings": [
                    finding.id for finding in report.findings if not finding.case_ids
                ],
            },
        },
    }


def format_status(report: HardeningStatus) -> list[str]:
    payload = status_payload(report)
    finding_data = payload["findings"]
    pilot_data = payload["pilots"]
    oracle_data = payload["oracles"]
    case_data = payload["cases"]
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
    lines.append(
        f"Oracles: {oracle_data['total']} "
        f"({oracle_data['approved']} approved, {len(oracle_data['draft'])} draft)"
    )
    lines.append(
        f"Cases: {case_data['total']} "
        f"({case_data['gating']} gating, {case_data['regression']} regression, "
        f"{case_data['coverage']} coverage)"
    )
    uncovered = case_data["finding_coverage"]["uncovered_findings"]
    lines.append(
        "Findings without cases: " + (", ".join(uncovered) if uncovered else "none")
    )
    return lines
