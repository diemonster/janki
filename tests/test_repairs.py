"""Constrained repairs, review proposals, and their recovery transaction."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import cli, repairs
from japanese_anki import io as data_io
from japanese_anki.io import DataError, load_records, save_records_json
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord


def record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:ねこ:ねこ",
        "expression": "ねこ",
        "reading": "ねこ",
        "meanings": ["cat"],
        "source": SourceReference(type="test", imported_from="case.yaml"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def project(tmp_path: Path, records: list[VocabularyRecord]) -> tuple[Path, Path, Path]:
    (tmp_path / "janki.toml").write_text("", encoding="utf-8")
    normalized = tmp_path / "data" / "normalized" / "vocabulary.json"
    staging = tmp_path / "data" / "staging"
    staging.mkdir(parents=True)
    save_records_json(normalized, records)
    return tmp_path, normalized, staging


def safe_document(tmp_path: Path, item: VocabularyRecord) -> repairs.SafeDocument:
    root, normalized, staging = project(tmp_path, [item])
    return repairs.read_safe_document(root, normalized, staging, normalized)


def test_seed_repair_has_two_near_misses_and_is_idempotent(tmp_path: Path) -> None:
    declaration = repairs.REGISTRY.select(["record-romaji-from-reading"])
    document = safe_document(tmp_path, record(romaji="wrong"))

    plan = repairs.build_plan(document, declaration)

    assert [(change.field, change.old, change.new) for change in plan.changes] == [
        ("romaji", "wrong", "neko")
    ]
    assert plan.records[0].id == document.records[0].id
    repaired_again, second_changes = repairs.apply_declarations(
        plan.records, declaration, modes=frozenset({"ingest-safe"})
    )
    assert repaired_again == list(plan.records)
    assert second_changes == []

    no_reading = record(reading="", romaji="kept")
    kanji_reading = record(reading="猫", romaji="kept")
    unchanged, changes = repairs.apply_declarations(
        [no_reading, kanji_reading],
        declaration,
        modes=frozenset({"ingest-safe"}),
    )
    assert unchanged == [no_reading, kanji_reading]
    assert changes == []


def test_example_romaji_repair_is_narrow_and_idempotent() -> None:
    declarations = repairs.REGISTRY.select(["example-romaji-from-furigana"])
    item = record(
        examples=[
            ExampleSentence(japanese="ねこはかわいい。", romaji="wrong")
        ]
    )

    repaired, changes = repairs.apply_declarations(
        [item], declarations, modes=frozenset({"ingest-safe"})
    )

    assert repaired[0].examples[0].romaji == "nekohakawaii."
    assert [change.field for change in changes] == ["examples[0].romaji"]
    assert repaired[0].id == item.id
    repeated, repeated_changes = repairs.apply_declarations(
        repaired, declarations, modes=frozenset({"ingest-safe"})
    )
    assert repeated == repaired
    assert repeated_changes == []
    near_misses = [record(), replace(item, examples=[repaired[0].examples[0]])]
    unchanged, near_changes = repairs.apply_declarations(
        near_misses, declarations, modes=frozenset({"ingest-safe"})
    )
    assert unchanged == near_misses
    assert near_changes == []


@pytest.mark.parametrize("callback_name", ["precondition", "transformation", "postcondition"])
def test_each_callback_is_denied_an_undeclared_read(callback_name: str) -> None:
    base = repairs.REGISTRY.get("record-romaji-from-reading")
    assert base is not None

    def precondition(before: Any, evidence: Any) -> bool:
        del evidence
        return bool(before["expression"]) if callback_name == "precondition" else True

    def transformation(before: Any, evidence: Any) -> dict[str, str]:
        del evidence
        if callback_name == "transformation":
            before["expression"]
        return {"romaji": "neko"}

    def postcondition(before: Any, planned: Any, evidence: Any) -> bool:
        del planned, evidence
        return bool(before["expression"]) if callback_name == "postcondition" else True

    declaration = replace(
        base,
        code=f"test-{callback_name}",
        input_fields=("romaji",),
        precondition=precondition,
        transformation=transformation,
        postcondition=postcondition,
    )
    registry = repairs.RepairRegistry([declaration])

    with pytest.raises(repairs.RepairError, match="undeclared input"):
        repairs.apply_declarations(
            [record()], registry.all(), modes=frozenset({"ingest-safe"})
        )


def test_registry_and_runtime_refuse_protected_or_undeclared_writes() -> None:
    base = repairs.REGISTRY.get("record-romaji-from-reading")
    assert base is not None
    with pytest.raises(repairs.RepairError, match="automatic allowlist"):
        repairs.RepairRegistry([replace(base, allowed_fields=("meanings",))])

    declaration = replace(
        base,
        code="test-undeclared-write",
        transformation=lambda _before, _evidence: {"meanings": ["changed"]},
    )
    with pytest.raises(repairs.RepairError, match="undeclared field"):
        repairs.apply_declarations(
            [record()], [declaration], modes=frozenset({"ingest-safe"})
        )

    with pytest.raises(repairs.RepairError, match="leaf fields"):
        repairs.RepairRegistry([replace(base, input_fields=("examples",))])
    with pytest.raises(repairs.RepairError, match="canonical JSON"):
        repairs.RepairRegistry([replace(base, evidence={"object": object()})])
    frozen = repairs.RepairRegistry(
        [replace(base, evidence={"nested": {"proof": "stable"}})]
    ).all()[0]
    with pytest.raises(TypeError):
        frozen.evidence["nested"]["proof"] = "changed"
    _updated, nested_changes = repairs.apply_declarations(
        [record(romaji="wrong")],
        [frozen],
        modes=frozenset({"ingest-safe"}),
    )
    assert nested_changes[0].evidence == {"nested": {"proof": "stable"}}


def test_a_failed_postcondition_blocks_the_repair() -> None:
    base = repairs.REGISTRY.get("record-romaji-from-reading")
    assert base is not None
    declaration = replace(
        base,
        code="test-failed-postcondition",
        postcondition=lambda _before, _planned, _evidence: False,
    )

    with pytest.raises(repairs.RepairError, match="postcondition failed"):
        repairs.apply_declarations(
            [record(romaji="wrong")],
            [declaration],
            modes=frozenset({"ingest-safe"}),
        )


def test_a_declared_write_still_has_to_match_the_canonical_field_type() -> None:
    base = repairs.REGISTRY.get("record-romaji-from-reading")
    assert base is not None
    declaration = replace(
        base,
        code="test-invalid-output-type",
        transformation=lambda _before, _evidence: {"romaji": ["neko"]},
        postcondition=lambda _before, _planned, _evidence: True,
    )

    with pytest.raises(repairs.RepairError, match="canonical schema"):
        repairs.apply_declarations(
            [record(romaji="wrong")],
            [declaration],
            modes=frozenset({"ingest-safe"}),
        )


def test_plan_fingerprint_covers_version_evidence_diff_and_output(tmp_path: Path) -> None:
    document = safe_document(tmp_path, record(romaji="wrong"))
    base = repairs.REGISTRY.get("record-romaji-from-reading")
    assert base is not None
    original = repairs.build_plan(document, [base])
    changed_version = repairs.build_plan(document, [replace(base, version="2.0.0")])
    changed_evidence = repairs.build_plan(
        document, [replace(base, evidence={"algorithm": "same-rule-new-proof"})]
    )
    changed_provenance = repairs.build_plan(
        document, [replace(base, provenance="A changed provenance statement.")]
    )
    example = repairs.REGISTRY.get("example-romaji-from-furigana")
    assert example is not None
    changed_order = repairs.build_plan(document, [example, base])
    changed_diff = repairs.build_plan(
        document,
        [
            replace(
                base,
                transformation=lambda _before, _evidence: {"romaji": "cat"},
                postcondition=lambda _before, planned, _evidence: (
                    planned.get("romaji") == "cat"
                ),
            )
        ],
    )

    assert len(
        {
            original.plan_fingerprint,
            changed_version.plan_fingerprint,
            changed_evidence.plan_fingerprint,
            changed_provenance.plan_fingerprint,
            changed_order.plan_fingerprint,
            changed_diff.plan_fingerprint,
        }
    ) == 6
    assert original.intended_output_fingerprint == repairs.bytes_fingerprint(
        original.intended_text
    )
    annotation = json.loads(original.records[0].source.raw_fields["janki_repairs"])
    assert annotation[0]["code"] == base.code
    assert annotation[0]["version"] == base.version


def test_mixed_check_plan_runs_each_callback_once_and_fingerprints_shown_bytes(
    tmp_path: Path,
) -> None:
    document = safe_document(tmp_path, record(romaji="wrong"))
    base = repairs.REGISTRY.get("record-romaji-from-reading")
    assert base is not None
    calls = 0

    def transform(before: Any, evidence: Any) -> Any:
        nonlocal calls
        calls += 1
        return base.transformation(before, evidence)

    declaration = replace(
        base,
        code="test-proposal-preview",
        mode="proposal-only",
        transformation=transform,
    )

    plan = repairs.build_check_plan(document, [declaration])

    assert calls == 1
    assert plan.records[0].romaji == "neko"
    assert '"romaji": "neko"' in plan.intended_text
    assert plan.intended_output_fingerprint == repairs.bytes_fingerprint(
        plan.intended_text
    )


def test_apply_refuses_an_identity_swap_after_its_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = safe_document(tmp_path, record(romaji="wrong"))
    plan = repairs.build_plan(
        document, repairs.REGISTRY.select(["record-romaji-from-reading"])
    )
    original_write = repairs.atomic_write_text_bound
    attacker_text = document.text.replace('"wrong"', '"attacker"')

    def swap_then_write(path: Path, text: str, **kwargs: Any) -> None:
        moved = path.with_suffix(".original")
        path.rename(moved)
        path.write_text(attacker_text, encoding="utf-8")
        original_write(path, text, **kwargs)

    monkeypatch.setattr(repairs, "atomic_write_text_bound", swap_then_write)

    with pytest.raises(DataError, match="changed identity"):
        repairs.write_safe_document(document, plan.intended_text)
    assert document.path.read_text(encoding="utf-8") == attacker_text


def test_bound_replace_rechecks_the_path_after_it_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.json"
    original = "old\n"
    attacker = "attacker\n"
    target.write_text(original, encoding="utf-8")
    details = target.stat()
    original_stat = data_io.os.stat
    calls = 0

    def swap_on_final_stat(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        if kwargs.get("dir_fd") is not None and kwargs.get("follow_symlinks") is False:
            calls += 1
            if calls == 2:
                target.rename(target.with_suffix(".original"))
                target.write_text(attacker, encoding="utf-8")
        return original_stat(*args, **kwargs)

    monkeypatch.setattr(data_io.os, "stat", swap_on_final_stat)

    with pytest.raises(DataError, match="changed before replace"):
        data_io.atomic_write_text_bound(
            target,
            "new\n",
            expected_revision=repairs.bytes_fingerprint(original),
            expected_identity=(details.st_dev, details.st_ino),
        )
    assert target.read_text(encoding="utf-8") == attacker


def test_safe_reader_rejects_outside_traversal_symlink_and_directory(tmp_path: Path) -> None:
    root, normalized, staging = project(tmp_path, [record()])
    outside = tmp_path.parent / "outside-repair.json"
    outside.write_text("[]", encoding="utf-8")
    try:
        with pytest.raises(repairs.RepairError, match="escapes the repository"):
            repairs.read_safe_document(root, normalized, staging, outside)
        with pytest.raises(repairs.RepairError, match="cannot contain"):
            repairs.read_safe_document(
                root, normalized, staging, Path("data/normalized/../normalized/vocabulary.json")
            )
        link = tmp_path / "data" / "normalized" / "link.json"
        link.symlink_to(normalized)
        with pytest.raises(repairs.RepairError, match="symlink"):
            repairs.read_safe_document(root, normalized, staging, link)
        with pytest.raises(repairs.RepairError, match="regular file"):
            repairs.read_safe_document(root, normalized, staging, staging)
        fifo = staging / "input.yaml"
        os.mkfifo(fifo)
        with pytest.raises(repairs.RepairError, match="regular file"):
            repairs.read_safe_document(root, normalized, staging, fifo)
    finally:
        outside.unlink(missing_ok=True)


def test_noninteractive_apply_needs_the_exact_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, normalized, _staging = project(tmp_path, [record(romaji="wrong")])
    before = normalized.read_bytes()
    monkeypatch.setattr(os, "isatty", lambda _fd: False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    result = cli.main(
        [
            "--root",
            str(root),
            "repair",
            str(normalized),
            "--apply",
            "record-romaji-from-reading",
        ]
    )

    assert result == 1
    assert "--expected-plan" in capsys.readouterr().err
    assert normalized.read_bytes() == before


def test_noninteractive_no_op_apply_still_needs_an_expected_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, normalized, _staging = project(tmp_path, [record(romaji="neko")])
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    result = cli.main(
        [
            "--root",
            str(root),
            "repair",
            str(normalized),
            "--apply",
            "record-romaji-from-reading",
        ]
    )

    assert result == 1
    assert "--expected-plan" in capsys.readouterr().err


def test_repair_value_output_is_never_shortened() -> None:
    value = "x" * 200

    assert cli._format_repair_value(value) == json.dumps(value)


def test_noninteractive_apply_writes_only_the_checked_exact_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, normalized, _staging = project(tmp_path, [record(romaji="wrong")])
    check_arguments = [
        "--root",
        str(root),
        "repair",
        str(normalized),
        "--check",
        "record-romaji-from-reading",
        "--format",
        "json",
    ]
    assert cli.main(check_arguments) == 0
    checked = json.loads(capsys.readouterr().out)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    result = cli.main(
        [
            "--root",
            str(root),
            "repair",
            str(normalized),
            "--apply",
            "record-romaji-from-reading",
            "--expected-plan",
            checked["repair_plan_fingerprint"],
            "--format",
            "json",
        ]
    )

    assert result == 0
    assert load_records(normalized)[0].romaji == "neko"


def test_dependency_check_rejects_two_accepted_dependent_fields() -> None:
    first = {
        "record_id": "word:ねこ:ねこ",
        "target_field": "meanings",
        "basis_fields": ["meanings"],
        "proposal_entry_fingerprint": "a" * 64,
    }
    second = {
        "record_id": "word:ねこ:ねこ",
        "target_field": "part_of_speech",
        "basis_fields": ["meanings"],
        "proposal_entry_fingerprint": "b" * 64,
    }

    with pytest.raises(repairs.RepairError, match="depends on accepted target"):
        repairs._dependency_reasons(
            [first, second], frozenset({"a" * 64, "b" * 64})
        )
    stale = repairs._dependency_reasons([first, second], frozenset({"a" * 64}))
    assert stale == {
        "b" * 64: "An accepted proposal changed basis field(s): meanings"
    }


def test_journal_removal_refuses_an_identity_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _normalized, staging = project(tmp_path, [record()])
    journal = repairs.proposal_journal_path(staging)
    text = '{"marker": "test"}\n'
    journal.write_text(text, encoding="utf-8")
    original_unlink = repairs.unlink_path_bound

    def swap_then_unlink(path: Path, **kwargs: Any) -> None:
        path.rename(path.with_suffix(".original"))
        path.write_text(text, encoding="utf-8")
        original_unlink(path, **kwargs)

    monkeypatch.setattr(repairs, "unlink_path_bound", swap_then_unlink)

    with pytest.raises(DataError, match="changed identity"):
        repairs._remove_journal(journal, root, text)
    assert journal.read_text(encoding="utf-8") == text


def _proposal_only(code: str = "test-proposal-only") -> repairs.RepairDeclaration:
    """A synthetic proposal-only declaration, so the proposal machinery keeps
    its tests now that the punctuation repair — its only real instance — is
    deleted. M8.3 left the machinery standing for M8.4 to delete or
    re-instance; until that decision, these guards stay pinned."""
    base = repairs.REGISTRY.get("record-romaji-from-reading")
    assert base is not None
    return replace(base, code=code, mode="proposal-only")


def test_direct_apply_refuses_a_proposal_only_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mode guard is what keeps `repair --apply` from writing a protected
    change straight into vocabulary.json. Its only registered instance died
    with M8.3, so it is pinned through a synthetic declaration — relaxing the
    guard must fail this, not pass silently for want of a producer."""
    monkeypatch.setattr(
        repairs, "REGISTRY", repairs.RepairRegistry([_proposal_only()])
    )
    root, normalized, _staging = project(tmp_path, [record(romaji="wrong")])
    before = normalized.read_bytes()

    result = cli.main(
        ["--root", str(root), "repair", str(normalized), "--apply", "test-proposal-only"]
    )

    assert result == 1
    assert "only ingest-safe" in capsys.readouterr().err
    assert normalized.read_bytes() == before


def test_a_changed_declaration_version_makes_a_proposal_stale(
    tmp_path: Path,
) -> None:
    """The staleness check is the accept transaction's precondition: a proposal
    written under version N must not be applied under version N+1, whose
    transform may produce something else entirely."""
    root, normalized, staging = project(tmp_path, [record(romaji="wrong")])
    document = repairs.read_safe_document(root, normalized, staging, normalized)
    declaration = _proposal_only()
    proposal_path, _entries = repairs.create_proposals(
        document, [declaration], staging
    )
    changed_registry = repairs.RepairRegistry(
        [replace(declaration, version="2.0.0")]
    )

    _review_state, stale = repairs.inspect_proposals(
        root,
        normalized,
        staging,
        proposal_path,
        changed_registry,
    )

    assert stale
