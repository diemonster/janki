from __future__ import annotations

from pathlib import Path

import pytest

from japanese_anki import enrich, hardening, hardening_replay, jpdb, kanji, qc

ROOT = Path(__file__).resolve().parents[1]
SEEDED_CASES = (
    "ai-no-writable-change",
    "impossible-character-furigana",
    "missing-furigana-separator",
    "pos-precedence",
)
DISCOVERED_CASES = hardening_replay.discover_cases(ROOT)


def test_default_discovery_is_stable_and_is_the_complete_gating_corpus() -> None:
    cases = hardening_replay.discover_cases(ROOT)

    assert tuple(case.id for case in cases) == SEEDED_CASES
    assert all(case.gating for case in cases)


@pytest.mark.parametrize("case", DISCOVERED_CASES, ids=lambda case: case.id)
def test_each_discovered_gating_case_replays_offline(
    case: hardening.HardeningCase,
) -> None:
    result = hardening_replay.replay_case(ROOT, case)

    assert result.passed


def test_runner_inputs_reject_unknown_fields() -> None:
    with pytest.raises(hardening.HardeningError, match="unknown field"):
        hardening_replay.RUNNERS["validation-qc"](
            {"records": [], "recrods": []}, ROOT
        )


@pytest.mark.parametrize(
    "content",
    [b'{"records": [], "records": []}', b'{"value": NaN}'],
)
def test_replay_json_rejects_duplicate_keys_and_non_finite_numbers(
    content: bytes,
) -> None:
    with pytest.raises(hardening.HardeningError):
        hardening_replay._json_bytes(content, "case input")


def test_every_registered_runner_calls_its_production_boundary() -> None:
    assert set(hardening_replay.RUNNERS) == set(hardening.RUNNER_BOUNDARIES)

    candidate = hardening_replay.RUNNERS["candidate-response"](
        {"candidates": [], "observe": []}, ROOT
    )
    staging = hardening_replay.RUNNERS["staging-promote"](
        {
            "records": [
                {
                    "id": "word:話す:はなす",
                    "expression": "話す",
                    "reading": "はなす",
                }
            ],
            "metadata": {"source_file": "synthetic.json"},
            "skip_reading_check": True,
        },
        ROOT,
    )
    validation = hardening_replay.RUNNERS["validation-qc"](
        {
            "records": [
                {
                    "id": "word:話す:はなす",
                    "expression": "話す",
                    "reading": "はなす",
                    "furigana": "話[はな]す",
                    "meanings": ["to speak"],
                }
            ]
        },
        ROOT,
    )
    render = hardening_replay.RUNNERS["render-build"](
        {
            "records": [
                {
                    "id": "word:話す:はなす",
                    "expression": "話す",
                    "reading": "はなす",
                    "furigana": "話[はな]す",
                    "meanings": ["to speak"],
                }
            ],
            "cards": {"recognition": True, "production": False, "reading": False},
        },
        ROOT,
    )
    dictionary = hardening_replay.replay(ROOT, ["pos-precedence"])[0]
    ai = hardening_replay.replay(ROOT, ["ai-no-writable-change"])[0]

    assert candidate["record_ids"] == []
    assert staging["promoted_ids"] == ["word:話す:はなす"]
    assert validation["errors"] == 0
    assert render["notes"] == 1
    assert render["cards"] == 1
    assert dictionary.passed
    assert ai.passed


def test_default_replay_never_opens_a_network_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def network_forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("offline replay attempted a network request")

    monkeypatch.setattr(jpdb, "urllib_transport", network_forbidden)

    assert all(result.passed for result in hardening_replay.replay(ROOT))


def test_replay_reads_only_declared_case_fixtures_not_a_private_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = hardening.read_verified_bytes
    read_paths: list[Path] = []

    def capture(path: Path, root: Path, expected: str) -> bytes:
        read_paths.append(path)
        assert "data/inbox" not in path.as_posix()
        return original(path, root, expected)

    monkeypatch.setattr(hardening, "read_verified_bytes", capture)

    assert hardening_replay.replay(ROOT, ["pos-precedence"])[0].passed
    assert {path.name for path in read_paths} == {"input.json", "oracle.json"}


@pytest.mark.parametrize(
    ("case_id", "target", "replacement"),
    [
        ("pos-precedence", jpdb, lambda _codes: "adverb"),
        ("impossible-character-furigana", kanji, lambda _info, _reading: True),
        ("missing-furigana-separator", qc, lambda _furigana: ()),
        ("ai-no-writable-change", enrich, lambda _result: []),
    ],
)
def test_each_seeded_case_kills_a_production_mutant(
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    target: object,
    replacement: object,
) -> None:
    names = {
        "pos-precedence": "pos_to_part_of_speech",
        "impossible-character-furigana": "assigns_a_known_reading",
        "missing-furigana-separator": "spilled_furigana_groups",
        "ai-no-writable-change": "format_ai_no_changes",
    }
    monkeypatch.setattr(target, names[case_id], replacement)

    assert not hardening_replay.replay(ROOT, [case_id])[0].passed


def test_explicit_discovery_can_run_a_non_gating_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = hardening.HardeningCase(
        path=ROOT / "quality/cases/open/case.yaml",
        relative_path="quality/cases/open/case.yaml",
        id="open-case",
        purpose="regression",
        gating=False,
        runner="validation-qc",
        pipeline_boundary="validation-qc",
        source_archetype="scan",
        redistributable=True,
        fixture_basis="agent-created-synthetic",
        fixtures=(),
        runner_input="input.json",
        oracle="oracle.json",
        finding_id="open-finding",
        pilot_id=None,
        unit_oracle_id=None,
        source=None,
    )
    repository = hardening.HardeningRepository(
        catalog=hardening.FindingsCatalog(ROOT / "quality/findings.yaml", ()),
        pilots=(),
        oracles=(),
        cases=(case,),
    )
    monkeypatch.setattr(hardening, "load_repository", lambda _root: repository)

    assert hardening_replay.discover_cases(ROOT) == ()
    assert hardening_replay.discover_cases(ROOT, ["open-case"]) == (case,)


def test_cli_returns_nonzero_and_reports_structural_diff_for_a_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(enrich, "format_ai_no_changes", lambda _result: [])

    from japanese_anki import cli

    arguments = [
        "--root",
        str(ROOT),
        "harden",
        "replay",
        "ai-no-writable-change",
        "--format",
        "json",
    ]
    assert cli.main(arguments) == 1
    output = capsys.readouterr().out
    assert '"passed": false' in output
    assert '"actual"' in output
    assert '"expected"' in output
