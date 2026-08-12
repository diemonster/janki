from __future__ import annotations

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
    (quality / "findings.yaml").write_text(
        yaml.safe_dump(
            {"version": 1, "findings": findings or []},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for pilot in pilots or []:
        (pilot_dir / f"{pilot['id']}.yaml").write_text(
            yaml.safe_dump(pilot, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return tmp_path


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
