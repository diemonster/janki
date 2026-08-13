from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from japanese_anki import enrich, hardening, hardening_replay, jpdb, kanji, qc, repairs

ROOT = Path(__file__).resolve().parents[1]
SEEDED_CASES = (
    "ai-exact-headword-spelling",
    "ai-existing-example-annotations",
    "ai-no-writable-change",
    "ai-rejected-example-retry",
    "deck-membership-partition",
    "derived-romaji-repair",
    "enrichment-forced-kana-reading",
    "enrichment-invalid-pitch-length",
    "enrichment-suru-compound",
    "extraction-oracle-key-binding",
    "extraction-selection-target-binding",
    "extraction-unit-accounting",
    "impossible-character-furigana",
    "input-external-existing-name-collision",
    "input-parent-inbox-casefold-collision",
    "input-parent-inbox-name-collision",
    "input-parent-inbox-provenance",
    "m7-mixed-tsumori-coverage",
    "m7-native-teform-table-coverage",
    "missing-furigana-separator",
    "pos-precedence",
    "reading-check-forced-furigana",
    "reading-check-suru-compound",
    "review-pitch-fact-recheck",
    "review-pitch-source-authority",
    "review-pitch-unbound-remains-reviewable",
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
    "scan_source_name",
    ("../lesson.pdf", "/tmp/lesson.pdf", "sub/lesson.pdf"),
)
def test_input_provenance_runner_requires_one_scan_filename(
    scan_source_name: str,
) -> None:
    with pytest.raises(hardening.HardeningError, match="must be one filename"):
        hardening_replay.RUNNERS["input-provenance"](
            {
                "source_name": "lesson.pdf",
                "content": "%PDF-1.7 source",
                "scan_source_name": scan_source_name,
                "scan_content": "%PDF-1.7 scan",
            },
            ROOT,
        )


def test_input_provenance_runner_resolves_a_temporary_root_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "unrelated-real-name"
    real.mkdir()
    alias = tmp_path / "short-alias"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(
        hardening_replay.tempfile,
        "TemporaryDirectory",
        lambda: contextlib.nullcontext(str(alias)),
    )

    observed = hardening_replay.RUNNERS["input-provenance"](
        {
            "source_name": "lesson.pdf",
            "content": "%PDF-1.7 source",
            "scan_content": "%PDF-1.7 conflicting scan",
        },
        ROOT,
    )

    assert observed["error_named_files"] == [
        "data/inbox/lesson.pdf",
        "data/inbox/scans/lesson.pdf",
    ]


def test_input_provenance_runner_relativizes_a_copied_path_through_an_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "different-real-name"
    real.mkdir()
    alias = tmp_path / "copy-alias"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(
        hardening_replay.tempfile,
        "TemporaryDirectory",
        lambda: contextlib.nullcontext(str(alias)),
    )

    observed = hardening_replay.RUNNERS["input-provenance"](
        {
            "source_name": "lesson.pdf",
            "content": "%PDF-1.7 source",
            "external_content": "%PDF-1.7 external",
        },
        ROOT,
    )

    assert observed["exit_code"] == 0
    assert observed["origin_relative_path"].startswith(
        "data/inbox/scans/lesson-"
    )
    assert observed["origin_relative_path"].endswith(".pdf")


def test_candidate_runner_rejects_unused_oracle_fields() -> None:
    with pytest.raises(hardening.HardeningError, match="require observe_coverage"):
        hardening_replay.RUNNERS["candidate-response"](
            {
                "candidates": [],
                "oracle_units": [],
                "observe": [],
            },
            ROOT,
        )


def test_missing_structured_schema_dependency_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> object:
        raise ImportError("pydantic is absent")

    monkeypatch.setattr(hardening_replay.extract, "candidate_schema", missing)

    with pytest.raises(hardening.HardeningError, match=r"Install.*\.\[ai\]"):
        hardening_replay.RUNNERS["candidate-response"](
            {"candidates": [], "observe": []}, ROOT
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


def test_deck_membership_accepts_one_nonempty_word_deck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    deck_path = root / "data/decks/only.yaml"
    deck_path.parent.mkdir(parents=True)
    deck_path.touch()
    config = SimpleNamespace(root=root, deck_dir=deck_path.parent)
    monkeypatch.setattr(
        hardening_replay.ProjectConfig,
        "load",
        lambda _root: config,
    )
    monkeypatch.setattr(
        hardening_replay.status,
        "deck_files",
        lambda _config: [deck_path],
    )
    monkeypatch.setattr(hardening_replay, "deck_kind", lambda _path: "vocabulary")
    monkeypatch.setattr(
        hardening_replay,
        "resolve_deck_records",
        lambda _path: ({}, [SimpleNamespace(id="word:一:いち")]),
    )

    assert hardening_replay.RUNNERS["deck-membership"]({}, root) == {
        "decks": ["data/decks/only.yaml"],
        "empty_decks": [],
        "overlaps": {},
    }


def test_deck_membership_rejects_an_external_deck_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "shared-decks"
    outside.mkdir()
    config = SimpleNamespace(root=root, deck_dir=outside)
    monkeypatch.setattr(
        hardening_replay.ProjectConfig,
        "load",
        lambda _root: config,
    )

    with pytest.raises(
        hardening.HardeningError,
        match="configured deck directory must be inside the repository",
    ):
        hardening_replay.RUNNERS["deck-membership"]({}, root)


def test_deck_membership_rejects_two_paths_to_one_deck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    deck_dir = root / "data/decks"
    deck_dir.mkdir(parents=True)
    original = deck_dir / "only.yaml"
    original.touch()
    alias = deck_dir / "alias.yaml"
    alias.symlink_to(original.name)
    config = SimpleNamespace(root=root, deck_dir=deck_dir)
    monkeypatch.setattr(
        hardening_replay.ProjectConfig,
        "load",
        lambda _root: config,
    )

    with pytest.raises(
        hardening.HardeningError,
        match=r"two paths for the same deck file:.*alias\.yaml.*only\.yaml",
    ):
        hardening_replay.RUNNERS["deck-membership"]({}, root)


def test_every_registered_runner_calls_its_production_boundary() -> None:
    assert set(hardening_replay.RUNNERS) == set(hardening.RUNNER_BOUNDARIES)

    candidate = hardening_replay.RUNNERS["candidate-response"](
        {"candidates": [], "observe": []}, ROOT
    )
    deck_membership = hardening_replay.RUNNERS["deck-membership"](
        {},
        ROOT,
    )
    input_provenance = hardening_replay.RUNNERS["input-provenance"](
        {"source_name": "lesson.pdf", "content": "%PDF-1.7 synthetic"}, ROOT
    )
    extraction_prompt = hardening_replay.RUNNERS["extraction-prompt"](
        {
            "source_name": "lesson.pdf",
            "known": [],
            "unit_keys": [
                {"page": 1, "section": "lesson-table", "ordinal": 1}
            ],
        },
        ROOT,
    )
    staging = hardening_replay.RUNNERS["staging-promote"](
        {
            "records": [
                {
                    "id": "word:話す:はなす",
                    "expression": "話す",
                    "reading": "はなす",
                },
                {
                    "id": "word:読む:",
                    "expression": "読む",
                    "reading": "",
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
            "observe_fields": ["Expression", "Furigana"],
        },
        ROOT,
    )
    dictionary = hardening_replay.replay(ROOT, ["pos-precedence"])[0]
    ai = hardening_replay.replay(ROOT, ["ai-no-writable-change"])[0]

    assert candidate["record_ids"] == []
    assert deck_membership == {
        "decks": [
            "data/decks/m7-camera-vertical-dialogue.yaml",
            "data/decks/m7-mixed-tsumori.yaml",
            "data/decks/m7-native-teform-table.yaml",
            "data/decks/verbs.yaml",
            "data/decks/yotsuba.yaml",
        ],
        "empty_decks": [],
        "overlaps": {},
    }
    assert input_provenance == {
        "exit_code": 0,
        "origin_relative_path": "data/inbox/lesson.pdf",
        "scan_copy_exists": False,
        "stored_files": ["data/inbox/lesson.pdf"],
        "error_has_name_collision": False,
        "error_named_files": [],
    }
    assert extraction_prompt["prompt_has_unit_keys"] is True
    assert staging["promoted_ids"] == ["word:話す:はなす"]
    assert staging["round_trip_record_ids"] == ["word:話す:はなす", "word:読む:"]
    assert validation["errors"] == 0
    assert render["notes"] == 1
    assert render["cards"] == 1
    assert render["rendered"] == [
        {
            "record_id": "word:話す:はなす",
            "fields": {"Expression": "話す", "Furigana": "話[はな]す"},
        }
    ]
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
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "quality" / "cases" / "private-case"
    case_dir.mkdir(parents=True)
    input_bytes = b'{"candidates": [], "observe": []}\n'
    oracle_bytes = b'{"records": [], "record_ids": [], "unusable": 0}\n'
    (case_dir / "input.json").write_bytes(input_bytes)
    (case_dir / "oracle.json").write_bytes(oracle_bytes)
    private_source = tmp_path / "data" / "inbox" / "private.png"
    private_source.parent.mkdir(parents=True)
    private_source.write_bytes(b"private pixels")
    case = hardening.HardeningCase(
        path=case_dir / "case.yaml",
        relative_path="quality/cases/private-case/case.yaml",
        id="private-case",
        purpose="coverage",
        gating=True,
        runner="candidate-response",
        pipeline_boundary="extraction-normalization",
        source_archetype="scan",
        redistributable=False,
        fixture_basis="agent-created-synthetic",
        fixtures=(
            hardening.CaseFixture(
                "input.json", hashlib.sha256(input_bytes).hexdigest()
            ),
            hardening.CaseFixture(
                "oracle.json", hashlib.sha256(oracle_bytes).hexdigest()
            ),
        ),
        runner_input="input.json",
        oracle="oracle.json",
        finding_id=None,
        pilot_id="private-pilot",
        unit_oracle_id="private-oracle",
        source=hardening.CaseSource(
            kind="private_inbox_ref",
            path="data/inbox/private.png",
            fingerprint=hashlib.sha256(b"private pixels").hexdigest(),
            live_eval=hardening.LiveEvalRequest(
                provider="anthropic",
                models=("approved-model",),
                purpose="hardening-eval",
                approval=None,
            ),
        ),
    )
    original = hardening.read_verified_bytes
    read_paths: list[Path] = []

    def capture(path: Path, root: Path, expected: str) -> bytes:
        read_paths.append(path)
        assert "data/inbox" not in path.as_posix()
        return original(path, root, expected)

    monkeypatch.setattr(hardening, "read_verified_bytes", capture)

    assert hardening_replay.replay_case(tmp_path, case).passed
    assert {path.name for path in read_paths} == {"input.json", "oracle.json"}


@pytest.mark.parametrize(
    ("case_id", "target", "replacement"),
    [
        ("pos-precedence", jpdb, lambda _codes: "adverb"),
        ("impossible-character-furigana", kanji, lambda _info, _reading: True),
        ("missing-furigana-separator", qc, lambda furigana: furigana),
        ("ai-no-writable-change", enrich, lambda _result: []),
        (
            "derived-romaji-repair",
            repairs,
            lambda records, _declarations, **_kwargs: (list(records), []),
        ),
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
        "missing-furigana-separator": "repair_spilled_punctuation",
        "ai-no-writable-change": "format_ai_no_changes",
        "derived-romaji-repair": "apply_declarations",
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
