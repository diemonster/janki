"""Constrained repairs, review proposals, and their recovery transaction."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from japanese_anki import cli, repairs
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


def test_punctuation_proposal_has_two_near_misses_and_is_idempotent() -> None:
    declarations = repairs.REGISTRY.select(
        ["example-furigana-punctuation-separator"]
    )
    item = punctuation_record()

    repaired, changes = repairs.apply_declarations(
        [item], declarations, modes=frozenset({"proposal-only"})
    )

    assert repaired[0].examples[0].furigana == "週末[しゅうまつ]、 何[なに]するの？"
    assert repaired[0].id == item.id
    assert len(changes) == 1
    repeated, repeated_changes = repairs.apply_declarations(
        repaired, declarations, modes=frozenset({"proposal-only"})
    )
    assert repeated == repaired
    assert repeated_changes == []
    correct = replace(
        item,
        examples=[
            replace(item.examples[0], furigana="お茶[おちゃ]を 飲[の]む")
        ],
    )
    ambiguous = replace(
        item,
        examples=[
            replace(
                item.examples[0],
                furigana="毎日[まいにち]、妻と日本語[にほんご]を 話[はな]す",
            )
        ],
    )
    unchanged, near_changes = repairs.apply_declarations(
        [correct, ambiguous], declarations, modes=frozenset({"proposal-only"})
    )
    assert unchanged == [correct, ambiguous]
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


def punctuation_record() -> VocabularyRecord:
    return VocabularyRecord(
        id="word:週末:しゅうまつ",
        expression="週末",
        reading="しゅうまつ",
        meanings=["weekend"],
        examples=[
            ExampleSentence(
                japanese="週末、何するの？",
                furigana="週末[しゅうまつ]、何[なに]するの？",
                english="What will you do this weekend?",
            )
        ],
        source=SourceReference(type="test", imported_from="case.yaml"),
    )


def test_proposal_generation_does_not_touch_source_and_acceptance_is_field_scoped(
    tmp_path: Path,
) -> None:
    root, normalized, staging = project(tmp_path, [punctuation_record()])
    original = normalized.read_bytes()
    document = repairs.read_safe_document(root, normalized, staging, normalized)
    declarations = repairs.REGISTRY.select(
        ["example-furigana-punctuation-separator"]
    )

    proposal_path, entries = repairs.create_proposals(document, declarations, staging)

    assert normalized.read_bytes() == original
    assert len(entries) == 1
    review_state, stale = repairs.inspect_proposals(
        root, normalized, staging, proposal_path
    )
    assert stale == {}
    entry_fingerprint = entries[0]["proposal_entry_fingerprint"]
    result = repairs.accept_proposals(review_state, {entry_fingerprint: True})

    assert result.accepted == 1
    [updated] = load_records(normalized)
    assert updated.examples[0].furigana == "週末[しゅうまつ]、 何[なに]するの？"
    annotation = json.loads(updated.source.raw_fields["janki_repairs"])
    assert annotation[0]["code"] == "example-furigana-punctuation-separator"
    remaining = yaml.safe_load(proposal_path.read_text(encoding="utf-8"))
    assert remaining["proposals"] == []
    archive = yaml.safe_load(result.archive_path.read_text(encoding="utf-8"))
    assert [item["proposal_entry_fingerprint"] for item in archive["accepted"]] == [
        entry_fingerprint
    ]
    assert not repairs.proposal_journal_path(staging).exists()


def test_changed_basis_is_marked_stale_without_changing_records(tmp_path: Path) -> None:
    root, normalized, staging = project(tmp_path, [punctuation_record()])
    document = repairs.read_safe_document(root, normalized, staging, normalized)
    declaration = repairs.REGISTRY.select(
        ["example-furigana-punctuation-separator"]
    )
    proposal_path, _entries = repairs.create_proposals(document, declaration, staging)
    changed = punctuation_record()
    changed.examples[0].furigana = "週末[しゅうまつ]、 何[なに]するの？"
    save_records_json(normalized, [changed])

    review_state, stale = repairs.inspect_proposals(
        root, normalized, staging, proposal_path
    )

    assert len(stale) == 1
    repairs.mark_stale_proposals(review_state, stale)
    raw = yaml.safe_load(proposal_path.read_text(encoding="utf-8"))
    assert "stale" in raw["proposals"][0]
    assert load_records(normalized)[0].examples[0].furigana == changed.examples[0].furigana


def test_ordinary_promote_refuses_a_proposal_without_pruning_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, normalized, staging = project(tmp_path, [punctuation_record()])
    document = repairs.read_safe_document(root, normalized, staging, normalized)
    proposal_path, _entries = repairs.create_proposals(
        document,
        repairs.REGISTRY.select(["example-furigana-punctuation-separator"]),
        staging,
    )
    before = proposal_path.read_bytes()

    result = cli.main(["--root", str(root), "promote", str(proposal_path)])

    assert result == 1
    assert "--accept-proposals" in capsys.readouterr().err
    assert proposal_path.read_bytes() == before


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


def test_direct_apply_refuses_a_proposal_only_repair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, normalized, _staging = project(tmp_path, [punctuation_record()])
    before = normalized.read_bytes()

    result = cli.main(
        [
            "--root",
            str(root),
            "repair",
            str(normalized),
            "--apply",
            "example-furigana-punctuation-separator",
        ]
    )

    assert result == 1
    assert "only ingest-safe" in capsys.readouterr().err
    assert normalized.read_bytes() == before


def test_duplicate_proposal_targets_are_invalid(tmp_path: Path) -> None:
    root, normalized, staging = project(tmp_path, [punctuation_record()])
    document = repairs.read_safe_document(root, normalized, staging, normalized)
    proposal_path, _entries = repairs.create_proposals(
        document,
        repairs.REGISTRY.select(["example-furigana-punctuation-separator"]),
        staging,
    )
    payload = yaml.safe_load(proposal_path.read_text(encoding="utf-8"))
    payload["proposals"].append(dict(payload["proposals"][0]))

    with pytest.raises(repairs.RepairError, match="duplicate proposal"):
        repairs.validate_proposal_payload(payload)


def test_changed_declaration_version_makes_the_proposal_stale(tmp_path: Path) -> None:
    root, normalized, staging = project(tmp_path, [punctuation_record()])
    document = repairs.read_safe_document(root, normalized, staging, normalized)
    [declaration] = repairs.REGISTRY.select(
        ["example-furigana-punctuation-separator"]
    )
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

    assert list(stale.values()) == ["The repair declaration version changed."]


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


@pytest.mark.parametrize(
    "crash_after",
    ["records", "archive", "staging"],
)
def test_recovery_finishes_each_ordered_partial_write_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_after: str,
) -> None:
    root, normalized, staging = project(tmp_path, [punctuation_record()])
    document = repairs.read_safe_document(root, normalized, staging, normalized)
    proposal_path, entries = repairs.create_proposals(
        document,
        repairs.REGISTRY.select(["example-furigana-punctuation-separator"]),
        staging,
    )
    review_state, stale = repairs.inspect_proposals(
        root, normalized, staging, proposal_path
    )
    assert stale == {}
    original_write = repairs._cas_write
    crashed = False

    def crash_once(path: Path, root_: Path, expected: str, intended: str) -> None:
        nonlocal crashed
        original_write(path, root_, expected, intended)
        is_target = (
            (crash_after == "records" and path == normalized)
            or (crash_after == "staging" and path == proposal_path)
            or (crash_after == "archive" and path.parent.name == "done")
        )
        if is_target and not crashed:
            crashed = True
            raise RuntimeError("injected crash")

    monkeypatch.setattr(repairs, "_cas_write", crash_once)
    with pytest.raises(RuntimeError, match="injected crash"):
        repairs.accept_proposals(
            review_state,
            {entries[0]["proposal_entry_fingerprint"]: True},
        )
    assert repairs.proposal_journal_path(staging).exists()

    monkeypatch.setattr(repairs, "_cas_write", original_write)
    assert repairs.recover_proposal_transaction(root, normalized, staging)

    [updated] = load_records(normalized)
    assert updated.examples[0].furigana == "週末[しゅうまつ]、 何[なに]するの？"
    archive_path = repairs.proposal_archive_path(staging, proposal_path)
    archive = yaml.safe_load(archive_path.read_text(encoding="utf-8"))
    assert len(archive["accepted"]) == 1
    proposal = yaml.safe_load(proposal_path.read_text(encoding="utf-8"))
    assert proposal["proposals"] == []
    assert not repairs.proposal_journal_path(staging).exists()


def test_recovery_refuses_an_out_of_order_target_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, normalized, staging = project(tmp_path, [punctuation_record()])
    document = repairs.read_safe_document(root, normalized, staging, normalized)
    proposal_path, entries = repairs.create_proposals(
        document,
        repairs.REGISTRY.select(["example-furigana-punctuation-separator"]),
        staging,
    )
    review_state, stale = repairs.inspect_proposals(
        root, normalized, staging, proposal_path
    )
    assert stale == {}
    original_write = repairs._cas_write

    def crash_after_records(
        path: Path, root_: Path, expected: str, intended: str
    ) -> None:
        original_write(path, root_, expected, intended)
        if path == normalized:
            raise RuntimeError("injected crash")

    monkeypatch.setattr(repairs, "_cas_write", crash_after_records)
    with pytest.raises(RuntimeError, match="injected crash"):
        repairs.accept_proposals(
            review_state,
            {entries[0]["proposal_entry_fingerprint"]: True},
        )
    monkeypatch.setattr(repairs, "_cas_write", original_write)
    archive_path = repairs.proposal_archive_path(staging, proposal_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text("version: 1\nkind: repair-proposal-archive\naccepted: []\n")

    with pytest.raises(repairs.RepairError, match="ordered transaction state"):
        repairs.recover_proposal_transaction(root, normalized, staging)
    assert repairs.proposal_journal_path(staging).exists()
