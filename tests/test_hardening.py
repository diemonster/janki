from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from japanese_anki import cli, hardening

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _finding(finding_id: str = "missing-second-column") -> dict[str, Any]:
    return {
        "id": finding_id,
        "state": "open",
        "pipeline_stage": "extraction",
        "source_archetypes": ["native-pdf-table"],
        "symptom": "The second column is absent.",
        "invariant": "Each selected cell has one disposition.",
        "evidence": [
            {
                "fingerprint": HASH_A,
                "locator": f"quality/cases/{finding_id}/case.yaml",
            }
        ],
        "recurrences": [],
        "case_ids": [],
    }


def _pipeline(*, complete: bool = False) -> dict[str, bool]:
    return {name: complete for name in hardening.PILOT_STEPS}


def _pilot(pilot_id: str = "native-table-pilot") -> dict[str, Any]:
    return {
        "version": 1,
        "id": pilot_id,
        "source_fingerprint": HASH_B,
        "source_archetype": "native-pdf-table",
        "redistributable": False,
        "unit_oracle_id": "native-table-units",
        "eval_case_id": "native-table-eval",
        "counts": {
            "extracted": 20,
            "omitted": 1,
            "held": 2,
            "promoted": 17,
            "rejected": 0,
        },
        "corrections": {"content_specific": 3, "systemic": 0},
        "repair_applications": [],
        "false_positive_repairs": [],
        "model_fingerprints": [],
        "prompt_fingerprints": [],
        "finding_ids": [],
        "pipeline": _pipeline(),
    }


def _project(
    tmp_path: Path,
    *,
    findings: list[dict[str, Any]] | None = None,
    pilots: list[dict[str, Any]] | None = None,
) -> Path:
    (tmp_path / "janki.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    quality = tmp_path / "quality"
    pilot_dir = quality / "pilots"
    pilot_dir.mkdir(parents=True)
    oracle_dir = quality / "oracles"
    oracle_dir.mkdir()
    case_root = quality / "cases"
    case_root.mkdir()
    (quality / "findings.yaml").write_text(
        yaml.safe_dump(
            {"version": 1, "findings": findings or []},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for pilot in pilots or []:
        source = f"source for {pilot['id']}\n".encode()
        source_fp = hashlib.sha256(source).hexdigest()
        pilot["source_fingerprint"] = source_fp
        inbox = tmp_path / "data" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        source_path = inbox / f"{pilot['id']}.txt"
        source_path.write_bytes(source)
        oracle_model = hardening.UnitOracle(
            path=oracle_dir / f"{pilot['unit_oracle_id']}.yaml",
            relative_path=f"quality/oracles/{pilot['unit_oracle_id']}.yaml",
            id=pilot["unit_oracle_id"],
            source_fingerprint=source_fp,
            type="exhaustive",
            case_ids=(pilot["eval_case_id"],),
            units=(
                hardening.OracleUnit(
                    page=1,
                    section="main",
                    ordinal=1,
                    context_fingerprint=HASH_A,
                    disposition="candidate",
                ),
            ),
            targets=(),
            selection_rubric=None,
            approval=None,
        )
        oracle = {
            "version": 1,
            "id": oracle_model.id,
            "source_fingerprint": source_fp,
            "type": "exhaustive",
            "case_ids": [pilot["eval_case_id"]],
            "units": [
                {
                    "page": 1,
                    "section": "main",
                    "ordinal": 1,
                    "context_fingerprint": HASH_A,
                    "disposition": "candidate",
                }
            ],
            "approval": {
                "authority": "repository-owner",
                "oracle_id": oracle_model.id,
                "source_fingerprint": source_fp,
                "oracle_type": "exhaustive",
                "oracle_content_fingerprint": hardening.oracle_content_fingerprint(
                    oracle_model
                ),
                "approved_at": "2026-08-12",
            },
        }
        (oracle_dir / f"{pilot['unit_oracle_id']}.yaml").write_text(
            yaml.safe_dump(oracle, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        _write_test_case(
            case_root,
            case_id=pilot["eval_case_id"],
            purpose="coverage",
            gating=True,
            source_archetype=pilot["source_archetype"],
            pilot_id=pilot["id"],
            unit_oracle_id=pilot["unit_oracle_id"],
            source={
                "kind": "private_inbox_ref",
                "path": f"data/inbox/{pilot['id']}.txt",
                "fingerprint": source_fp,
                "live_eval": {
                    "provider": "anthropic",
                    "models": ["claude-test"],
                    "purpose": "hardening-eval",
                },
            },
            redistributable=False,
        )
        (pilot_dir / f"{pilot['id']}.yaml").write_text(
            yaml.safe_dump(pilot, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    for finding in findings or []:
        for case_id in finding.get("case_ids", []):
            _write_test_case(
                case_root,
                case_id=case_id,
                purpose="regression",
                gating=finding["state"] == "fixed",
                source_archetype=finding["source_archetypes"][0],
                finding_id=finding["id"],
            )
    return tmp_path


def _write_test_case(
    case_root: Path,
    *,
    case_id: str,
    purpose: str,
    gating: bool,
    source_archetype: str,
    finding_id: str | None = None,
    pilot_id: str | None = None,
    unit_oracle_id: str | None = None,
    source: dict[str, Any] | None = None,
    redistributable: bool = True,
) -> None:
    if hardening._SLUG.fullmatch(case_id) is None:
        return
    case_dir = case_root / case_id
    case_dir.mkdir()
    input_text = '{"candidates": [], "observe": []}\n'
    oracle_text = '{"records": [], "record_ids": [], "unusable": 0}\n'
    (case_dir / "input.json").write_text(input_text, encoding="utf-8")
    (case_dir / "oracle.json").write_text(oracle_text, encoding="utf-8")
    manifest: dict[str, Any] = {
        "version": 1,
        "id": case_id,
        "purpose": purpose,
        "gating": gating,
        "runner": "candidate-response",
        "pipeline_boundary": "extraction-normalization",
        "source_archetype": source_archetype,
        "redistributable": redistributable,
        "fixture_basis": "agent-created-synthetic",
        "fixtures": [
            {
                "path": "input.json",
                "fingerprint": hashlib.sha256(input_text.encode()).hexdigest(),
            },
            {
                "path": "oracle.json",
                "fingerprint": hashlib.sha256(oracle_text.encode()).hexdigest(),
            },
        ],
        "runner_input": "input.json",
        "oracle": "oracle.json",
    }
    if finding_id is not None:
        manifest["finding_id"] = finding_id
    if pilot_id is not None:
        manifest["pilot_id"] = pilot_id
    if unit_oracle_id is not None:
        manifest["unit_oracle_id"] = unit_oracle_id
    if source is not None:
        manifest["source"] = source
    (case_dir / "case.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _write_findings(root: Path, findings: list[dict[str, Any]]) -> None:
    (root / "quality" / "findings.yaml").write_text(
        yaml.safe_dump(
            {"version": 1, "findings": findings},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _run(root: Path, *extra: str) -> int:
    return cli.main(["--root", str(root), "harden", "status", *extra])


def test_empty_catalog_is_valid_and_status_is_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    assert _run(root) == 0

    assert "Findings: 0" in capsys.readouterr().out
    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_json_status_is_deterministic_and_uses_relative_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, findings=[_finding()], pilots=[_pilot()])

    assert _run(root, "--format", "json") == 0
    first = capsys.readouterr().out
    assert _run(root, "--format", "json") == 0
    second = capsys.readouterr().out

    assert second == first
    payload = json.loads(first)
    assert payload["catalog"] == "quality/findings.yaml"
    assert payload["pilots"]["incomplete"][0]["path"] == (
        "quality/pilots/native-table-pilot.yaml"
    )
    assert str(root) not in first


def test_json_status_normalizes_set_like_finding_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first_finding = _finding("first-finding")
    first_finding["source_archetypes"] = ["scan", "camera"]
    first_finding["case_ids"] = ["second-case", "first-case"]
    second_finding = _finding("second-finding")
    root = _project(tmp_path, findings=[second_finding, first_finding])

    assert _run(root, "--format", "json") == 0
    first = capsys.readouterr().out
    first_finding["source_archetypes"].reverse()
    first_finding["case_ids"].reverse()
    _write_findings(root, [first_finding, second_finding])
    assert _run(root, "--format", "json") == 0
    second = capsys.readouterr().out

    assert second == first
    assert json.loads(second)["findings"]["open"][0]["source_archetypes"] == [
        "camera",
        "scan",
    ]


def test_text_status_reports_all_m7_2_signals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recurring = _finding()
    recurring["recurrences"] = [
        {"fingerprint": HASH_C, "locator": "quality/evidence/repeat.json"}
    ]
    deferred = _finding("vertical-text-order")
    deferred["state"] = "deferred"
    deferred["deferral_reason"] = "No redistributable fixture exists."
    pilot = _pilot()
    pilot["corrections"]["systemic"] = 1
    pilot["false_positive_repairs"] = [
        {"code": "normalize-romaji", "version": "v1", "count": 2}
    ]
    pilot["finding_ids"] = ["missing-second-column"]
    root = _project(tmp_path, findings=[deferred, recurring], pilots=[pilot])

    assert _run(root) == 0

    output = capsys.readouterr().out
    assert "Open findings: missing-second-column" in output
    assert "Deferred findings: vertical-text-order" in output
    assert "Recurrences: missing-second-column 1" in output
    assert "native-table-pilot: missing" in output
    assert "Repair false positives: normalize-romaji@v1 2" in output
    assert "Empty source-archetype cells:" in output
    assert "mixed-layout-pdf" in output


@pytest.mark.parametrize(
    ("document", "field"),
    [
        ("finding", "unexpected"),
        ("evidence", "note"),
        ("pilot", "unexpected"),
        ("counts", "unknown_count"),
    ],
)
def test_unknown_fields_are_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    document: str,
    field: str,
) -> None:
    finding = _finding()
    pilot = _pilot()
    if document == "finding":
        finding[field] = "bad"
    elif document == "evidence":
        finding["evidence"][0][field] = "bad"
    elif document == "pilot":
        pilot[field] = "bad"
    else:
        pilot["counts"][field] = 1
    root = _project(tmp_path, findings=[finding], pilots=[pilot])

    assert _run(root) == 1

    assert "unknown field" in capsys.readouterr().err


@pytest.mark.parametrize("document", ["catalog", "pilot"])
@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_schema_version_must_be_the_exact_supported_integer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    document: str,
    version: object,
) -> None:
    pilot = _pilot()
    root = _project(tmp_path, pilots=[pilot])
    if document == "catalog":
        (root / "quality" / "findings.yaml").write_text(
            yaml.safe_dump(
                {"version": version, "findings": []}, sort_keys=False
            ),
            encoding="utf-8",
        )
    else:
        pilot["version"] = version
        (root / "quality" / "pilots" / "native-table-pilot.yaml").write_text(
            yaml.safe_dump(pilot, sort_keys=False), encoding="utf-8"
        )

    assert _run(root) == 1

    assert "version must be integer 1" in capsys.readouterr().err


@pytest.mark.parametrize(
    "locator",
    [
        "../private/source.pdf",
        "/tmp/source.pdf",
        "C:\\source.pdf",
        "\\rooted\\source.pdf",
        "https://x.test/a",
        "https:\\x.test\\a",
    ],
)
def test_evidence_locator_rejects_path_traversal_and_external_locations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    locator: str,
) -> None:
    finding = _finding()
    finding["evidence"][0]["locator"] = locator
    root = _project(tmp_path, findings=[finding])

    assert _run(root) == 1

    assert "repository-relative locator" in capsys.readouterr().err


def test_relative_locator_uses_canonical_separators(tmp_path: Path) -> None:
    finding = _finding()
    finding["evidence"][0]["locator"] = "quality\\evidence\\source.json"
    root = _project(tmp_path, findings=[finding])

    loaded = hardening.load_findings(root).findings[0]

    assert loaded.evidence[0].locator == "quality/evidence/source.json"


def test_fix_reference_rejects_path_traversal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    finding = _finding()
    finding.update(
        state="fixed",
        case_ids=["missing-second-column-case"],
        fix_ref="../outside",
    )
    root = _project(tmp_path, findings=[finding])

    assert _run(root) == 1

    assert "repository-relative locator" in capsys.readouterr().err


def test_duplicate_finding_ids_are_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, findings=[_finding(), _finding()])

    assert _run(root) == 1

    assert "duplicate finding IDs" in capsys.readouterr().err


def test_duplicate_pilot_ids_are_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pilot = _pilot()
    root = _project(tmp_path, pilots=[pilot])
    (root / "quality" / "pilots" / "native-table-pilot.yml").write_text(
        yaml.safe_dump(pilot, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    assert _run(root) == 1

    assert "duplicate pilot IDs" in capsys.readouterr().err


def test_duplicate_yaml_keys_are_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)
    (root / "quality" / "findings.yaml").write_text(
        "version: 1\nversion: 1\nfindings: []\n", encoding="utf-8"
    )

    assert _run(root) == 1

    assert "duplicate key" in capsys.readouterr().err


def test_invalid_utf_8_is_a_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)
    (root / "quality" / "findings.yaml").write_bytes(b"version: 1\n\xff")

    assert _run(root) == 1

    assert "Could not decode quality/findings.yaml as UTF-8" in capsys.readouterr().err


def test_pilot_directory_listing_failure_is_a_cli_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    pilot_dir = root / "quality" / "pilots"
    original_iterdir = Path.iterdir

    def fail_for_pilots(path: Path):
        if path == pilot_dir:
            raise PermissionError("test denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_for_pilots)

    assert _run(root) == 1

    assert "Could not list quality/pilots: test denied" in capsys.readouterr().err


def test_a_recurrence_cannot_repeat_initial_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    finding = _finding()
    finding["recurrences"] = [dict(finding["evidence"][0])]
    root = _project(tmp_path, findings=[finding])

    assert _run(root) == 1

    assert "recurrences repeats an initial evidence reference" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("case", "Case One"),
        ("oracle", "../oracle"),
        ("eval", "eval/case"),
        ("finding", "UPPERCASE"),
    ],
)
def test_link_ids_must_be_lowercase_slugs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    target: str,
    value: str,
) -> None:
    finding = _finding()
    pilot = _pilot()
    if target == "case":
        finding["case_ids"] = [value]
    elif target == "oracle":
        pilot["unit_oracle_id"] = value
    elif target == "eval":
        pilot["eval_case_id"] = value
    else:
        pilot["finding_ids"] = [value]
    root = _project(tmp_path, findings=[finding], pilots=[pilot])

    assert _run(root) == 1

    assert "lowercase slug" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("state", "updates", "message"),
    [
        ("open", {"fix_ref": "commit:abc"}, "open finding cannot"),
        ("fixed", {}, "fixed requires"),
        ("deferred", {}, "deferred requires"),
        ("accepted-risk", {}, "accepted-risk requires"),
    ],
)
def test_each_terminal_state_enforces_its_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    state: str,
    updates: dict[str, Any],
    message: str,
) -> None:
    finding = _finding()
    finding["state"] = state
    finding.update(updates)
    root = _project(tmp_path, findings=[finding])

    assert _run(root) == 1

    assert message in capsys.readouterr().err


def test_fixed_and_deferred_findings_can_be_valid(tmp_path: Path) -> None:
    fixed = _finding("fixed-layout")
    fixed.update(
        state="fixed",
        case_ids=["fixed-layout-case"],
        fix_ref="commit:abc123",
    )
    deferred = _finding("deferred-layout")
    deferred.update(state="deferred", deferral_reason="Needs a new fixture.")
    root = _project(tmp_path, findings=[fixed, deferred])

    report = hardening.build_status(root)

    assert [item.state for item in report.findings] == ["deferred", "fixed"]


def _accepted_risk() -> tuple[dict[str, Any], str]:
    raw = _finding("accepted-layout")
    raw.update(
        state="accepted-risk",
        risk="One rare annotation can remain held.",
        reason="No safe deterministic rule exists.",
    )
    model = hardening.Finding(
        id=raw["id"],
        state=raw["state"],
        pipeline_stage=raw["pipeline_stage"],
        source_archetypes=tuple(raw["source_archetypes"]),
        symptom=raw["symptom"],
        invariant=raw["invariant"],
        evidence=(
            hardening.EvidenceReference(
                fingerprint=raw["evidence"][0]["fingerprint"],
                locator=raw["evidence"][0]["locator"],
            ),
        ),
        recurrences=(),
        case_ids=(),
        risk=raw["risk"],
        reason=raw["reason"],
    )
    fingerprint = hardening.risk_content_fingerprint(model)
    raw["approval"] = {
        "authority": "repository-owner",
        "finding_id": raw["id"],
        "approved_at": "2026-08-12",
        "risk_content_fingerprint": fingerprint,
        "risk": raw["risk"],
        "reason": raw["reason"],
    }
    return raw, fingerprint


def test_accepted_risk_requires_a_current_owner_approval(tmp_path: Path) -> None:
    finding, fingerprint = _accepted_risk()
    root = _project(tmp_path, findings=[finding])

    report = hardening.build_status(root)

    assert report.findings[0].approval is not None
    assert report.findings[0].approval.risk_content_fingerprint == fingerprint


def test_accepted_risk_draft_reports_the_fingerprint_for_owner_approval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    finding, fingerprint = _accepted_risk()
    del finding["approval"]
    root = _project(tmp_path, findings=[finding])

    assert _run(root) == 1

    error = capsys.readouterr().err
    assert "requires repository-owner approval" in error
    assert fingerprint in error


def test_accepted_risk_approval_becomes_stale_after_content_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    finding, _ = _accepted_risk()
    finding["symptom"] = "The changed symptom is now part of the risk."
    root = _project(tmp_path, findings=[finding])

    assert _run(root) == 1

    assert "approval is stale" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["finding_id", "risk", "reason"])
def test_accepted_risk_approval_must_repeat_the_exact_decision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
) -> None:
    finding, _ = _accepted_risk()
    finding["approval"][field] = "different-value"
    root = _project(tmp_path, findings=[finding])

    assert _run(root) == 1

    assert "must name the finding ID, risk, and reason exactly" in (
        capsys.readouterr().err
    )


def test_unquoted_yaml_approval_date_is_valid(tmp_path: Path) -> None:
    finding, _ = _accepted_risk()
    root = _project(tmp_path)
    rendered = yaml.safe_dump(
        {"version": 1, "findings": [finding]}, allow_unicode=True, sort_keys=False
    ).replace("approved_at: '2026-08-12'", "approved_at: 2026-08-12")
    (root / "quality" / "findings.yaml").write_text(rendered, encoding="utf-8")

    assert hardening.load_findings(root).findings[0].approval is not None


def test_content_specific_corrections_do_not_create_findings(tmp_path: Path) -> None:
    pilot = _pilot()
    pilot["corrections"]["content_specific"] = 7
    root = _project(tmp_path, pilots=[pilot])

    payload = hardening.status_payload(hardening.build_status(root))

    assert payload["findings"]["total"] == 0
    assert payload["pilots"]["corrections"] == {
        "content_specific": 7,
        "systemic": 0,
    }


@pytest.mark.parametrize("kind", ["systemic", "false-positive"])
def test_systemic_pilot_events_need_a_finding_link(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    pilot = _pilot()
    if kind == "systemic":
        pilot["corrections"]["systemic"] = 1
    else:
        pilot["false_positive_repairs"] = [
            {"code": "normalize-romaji", "version": "v1", "count": 1}
        ]
    root = _project(tmp_path, pilots=[pilot])

    assert _run(root) == 1

    assert "finding_ids must link" in capsys.readouterr().err


def test_pilot_finding_links_must_name_catalog_entries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pilot = _pilot()
    pilot["finding_ids"] = ["not-in-catalog"]
    root = _project(tmp_path, pilots=[pilot])

    assert _run(root) == 1

    assert "links unknown finding ID" in capsys.readouterr().err


def test_complete_pilot_requires_model_and_prompt_fingerprints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pilot = _pilot()
    pilot["pipeline"] = _pipeline(complete=True)
    root = _project(tmp_path, pilots=[pilot])

    assert _run(root) == 1

    assert "complete pilot needs model_fingerprints" in capsys.readouterr().err


def test_complete_pilot_with_fingerprints_is_valid(tmp_path: Path) -> None:
    pilot = _pilot()
    pilot["pipeline"] = _pipeline(complete=True)
    pilot["model_fingerprints"] = [
        {
            "phase": "extraction",
            "provider": "anthropic",
            "model": "claude-sonnet",
            "fingerprint": HASH_A,
        }
    ]
    pilot["prompt_fingerprints"] = [
        {"phase": "extraction", "kind": "system-prompt", "fingerprint": HASH_C}
    ]
    root = _project(tmp_path, pilots=[pilot])

    report = hardening.build_status(root)

    assert report.pilots[0].complete is True


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _dump_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def test_draft_oracle_cannot_bind_a_pilot_or_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pilot = _pilot()
    root = _project(tmp_path, pilots=[pilot])
    path = root / "quality" / "oracles" / "native-table-units.yaml"
    oracle = _load_yaml(path)
    del oracle["approval"]
    _dump_yaml(path, oracle)

    assert _run(root) == 1
    assert "cannot bind draft oracle" in capsys.readouterr().err


def test_oracle_approval_is_stale_after_reviewed_content_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, pilots=[_pilot()])
    path = root / "quality" / "oracles" / "native-table-units.yaml"
    oracle = _load_yaml(path)
    oracle["units"][0]["disposition"] = "duplicate"
    _dump_yaml(path, oracle)

    assert _run(root) == 1
    assert "approval is stale" in capsys.readouterr().err


def test_selection_oracle_is_a_draft_with_targets_and_a_rubric(tmp_path: Path) -> None:
    root = _project(tmp_path)
    oracle = {
        "version": 1,
        "id": "selected-prose",
        "source_fingerprint": HASH_A,
        "type": "selection",
        "case_ids": [],
        "targets": [
            {"identity": "word:話す:はなす", "locator": "page-1/paragraph-2"}
        ],
        "selection_rubric": "Select words that the first Genki volume does not teach.",
    }
    _dump_yaml(root / "quality" / "oracles" / "selected-prose.yaml", oracle)

    loaded = hardening.load_oracles(root)[0]

    assert loaded.type == "selection"
    assert loaded.approved is False
    assert loaded.units == ()
    assert loaded.targets[0].identity == "word:話す:はなす"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [("duplicate", "duplicate unit keys"), ("out-of-order", "must use page")],
)
def test_exhaustive_oracle_unit_keys_are_unique_and_ordered(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    message: str,
) -> None:
    root = _project(tmp_path, pilots=[_pilot()])
    path = root / "quality" / "oracles" / "native-table-units.yaml"
    oracle = _load_yaml(path)
    del oracle["approval"]
    second = dict(oracle["units"][0])
    second["ordinal"] = 2
    oracle["units"].append(second)
    if mutation == "duplicate":
        oracle["units"][1]["ordinal"] = 1
    else:
        oracle["units"].reverse()
    _dump_yaml(path, oracle)

    assert _run(root) == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("version: 1\nversion: 1\n", "duplicate key"),
        (
            "version: 1\nid: bad-hash\nsource_fingerprint: nope\n"
            "type: exhaustive\ncase_ids: []\nunits:\n"
            "  - page: 1\n    section: main\n    ordinal: 1\n"
            f"    context_fingerprint: {HASH_A}\n    disposition: candidate\n",
            "lowercase SHA-256",
        ),
    ],
)
def test_oracle_duplicate_keys_and_malformed_hashes_are_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str,
    message: str,
) -> None:
    root = _project(tmp_path)
    (root / "quality" / "oracles" / "bad-hash.yaml").write_text(
        content, encoding="utf-8"
    )

    assert _run(root) == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hash", "expected"),
        ("runner", "unknown runner"),
        ("path", "path traversal"),
        ("undeclared", "undeclared"),
    ],
)
def test_case_manifest_rejects_unsafe_or_unpinned_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    message: str,
) -> None:
    finding = _finding()
    finding.update(
        state="fixed", case_ids=["missing-second-column-case"], fix_ref="commit:abc"
    )
    root = _project(tmp_path, findings=[finding])
    case_dir = root / "quality" / "cases" / "missing-second-column-case"
    manifest_path = case_dir / "case.yaml"
    manifest = _load_yaml(manifest_path)
    if mutation == "hash":
        manifest["fixtures"][0]["fingerprint"] = HASH_C
    elif mutation == "runner":
        manifest["runner"] = "os.system"
    elif mutation == "path":
        manifest["fixtures"][0]["path"] = "../input.json"
    else:
        (case_dir / "extra.txt").write_text("not declared\n", encoding="utf-8")
    _dump_yaml(manifest_path, manifest)

    assert _run(root) == 1
    assert message in capsys.readouterr().err


def test_private_source_consent_defaults_to_false(tmp_path: Path) -> None:
    root = _project(tmp_path, pilots=[_pilot()])

    repository = hardening.load_repository(root)

    source = repository.cases[0].source
    assert source is not None
    assert source.kind == "private_inbox_ref"
    assert source.live_eval is not None
    assert source.live_eval.consented is False


@pytest.mark.parametrize("field", ["provider", "models", "purpose"])
def test_live_consent_binds_case_source_provider_models_and_purpose(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str
) -> None:
    root = _project(tmp_path, pilots=[_pilot()])
    path = root / "quality" / "cases" / "native-table-eval" / "case.yaml"
    case = _load_yaml(path)
    source = case["source"]
    request = source["live_eval"]
    request["approval"] = {
        "authority": "repository-owner",
        "case_id": case["id"],
        "source_fingerprint": source["fingerprint"],
        "provider": request["provider"],
        "models": request["models"],
        "purpose": request["purpose"],
        "reason": "Run this exact hardening evaluation.",
        "approved_at": "2026-08-12",
    }
    _dump_yaml(path, case)

    assert _run(root) == 0
    capsys.readouterr()
    request["approval"][field] = {
        "provider": "codex",
        "models": ["different-model"],
        "purpose": "another-purpose",
    }[field]
    _dump_yaml(path, case)
    assert _run(root) == 1
    assert "must match the case, source, provider, models, and purpose" in (
        capsys.readouterr().err
    )


def test_changed_private_source_invalidates_the_pinned_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, pilots=[_pilot()])
    source = root / "data" / "inbox" / "native-table-pilot.txt"
    source.write_text("changed source\n", encoding="utf-8")

    assert _run(root) == 1
    assert "expected" in capsys.readouterr().err


def test_private_source_must_stay_under_data_inbox(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, pilots=[_pilot()])
    path = root / "quality" / "cases" / "native-table-eval" / "case.yaml"
    case = _load_yaml(path)
    case["source"]["path"] = "quality/findings.yaml"
    _dump_yaml(path, case)

    assert _run(root) == 1
    assert "must be under data/inbox" in capsys.readouterr().err


def test_owner_provided_bundled_source_needs_exact_redistribution_approval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pilot = _pilot()
    pilot["redistributable"] = True
    root = _project(tmp_path, pilots=[pilot])
    path = root / "quality" / "cases" / "native-table-eval" / "case.yaml"
    case = _load_yaml(path)
    case_dir = path.parent
    source_path = root / case["source"]["path"]
    source_bytes = source_path.read_bytes()
    bundled = case_dir / "source.txt"
    bundled.write_bytes(source_bytes)
    source_fp = hashlib.sha256(source_bytes).hexdigest()
    case["redistributable"] = True
    case["fixture_basis"] = "owner-provided"
    case["fixtures"].append({"path": "source.txt", "fingerprint": source_fp})
    case["source"] = {
        "kind": "bundled_fixture",
        "path": "source.txt",
        "fingerprint": source_fp,
        "basis": "The repository owner permits this test fixture.",
    }
    _dump_yaml(path, case)

    assert _run(root) == 1
    assert "requires repository-owner redistribution approval" in capsys.readouterr().err

    case["source"]["redistribution_approval"] = {
        "authority": "repository-owner",
        "case_id": case["id"],
        "artifact_fingerprint": source_fp,
        "basis": "The repository owner permits this test fixture.",
        "approved_at": "2026-08-12",
    }
    _dump_yaml(path, case)
    assert _run(root) == 0

    capsys.readouterr()
    case["source"]["redistribution_approval"]["artifact_fingerprint"] = HASH_C
    _dump_yaml(path, case)
    assert _run(root) == 1
    assert "must match the case, artifact, and basis" in capsys.readouterr().err


def test_case_fixture_symlink_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    finding = _finding()
    finding.update(state="fixed", case_ids=["fixed-case"], fix_ref="commit:abc")
    root = _project(tmp_path, findings=[finding])
    case_dir = root / "quality" / "cases" / "fixed-case"
    fixture = case_dir / "input.json"
    target = case_dir / "target.json"
    target.write_bytes(fixture.read_bytes())
    fixture.unlink()
    fixture.symlink_to(target.name)

    assert _run(root) == 1
    assert "symlink" in capsys.readouterr().err


def test_fixed_finding_refuses_a_missing_or_non_gating_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    finding = _finding()
    finding.update(state="fixed", case_ids=["fixed-case"], fix_ref="commit:abc")
    root = _project(tmp_path, findings=[finding])
    path = root / "quality" / "cases" / "fixed-case" / "case.yaml"
    case = _load_yaml(path)
    case["gating"] = False
    _dump_yaml(path, case)

    assert _run(root) == 1
    assert "links non-gating case" in capsys.readouterr().err

    case["gating"] = True
    _dump_yaml(path, case)
    case["finding_id"] = "unknown-finding"
    _dump_yaml(path, case)
    assert _run(root) == 1
    assert "do not link both ways" in capsys.readouterr().err
