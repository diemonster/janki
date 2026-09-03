"""Exact destructive Assistant actions over canonical cards and deck files."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from japanese_anki import ledger
from japanese_anki.application import assistant_deletion
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    assistant_record_value,
)
from japanese_anki.application.assistant_deletion import (
    AssistantDeletionError,
    execute_canonical_deletion,
    execute_deck_deletion,
    plan_canonical_deletion,
    plan_deck_deletion,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.io import load_records
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord


def _record(expression: str, reading: str, *, tag: str) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=[f"meaning of {expression}"],
        examples=[
            ExampleSentence(
                japanese=f"{expression}を使います。",
                furigana=f"{expression}[{reading}]を使[つか]います。",
                romaji="fixture o tsukaimasu",
                english="I use the fixture.",
                register="polite",
                audio=f"example-{reading}.wav",
            )
        ],
        tags=[tag],
        audio=f"word-{reading}.wav",
        source=SourceReference(type="manual", imported_from="fixture"),
    )


def _project(tmp_path: Path) -> tuple[ProjectConfig, tuple[VocabularyRecord, ...], Path]:
    (tmp_path / "janki.toml").write_text(
        '[project]\nname = "Deletion fixture"\n'
        "[paths]\n"
        'normalized_file = "data/normalized/vocabulary.json"\n'
        'deck_dir = "data/decks"\n'
        'dist_dir = "dist"\n'
        'ledger_file = "data/ledger.json"\n'
        'staging_dir = "data/staging"\n'
        'operations_file = "data/operations.json"\n'
        'assistant_dir = "data/assistant"\n'
        'patterns_file = "data/patterns.json"\n'
        'kanji_file = "data/kanji.json"\n'
        'scan_inbox = "data/inbox"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    records = (
        _record("食べる", "たべる", tag="lesson"),
        _record("読む", "よむ", tag="other"),
    )
    config.normalized_file.parent.mkdir(parents=True)
    config.normalized_file.write_text(
        json.dumps(
            [record.to_dict() for record in records],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    config.deck_dir.mkdir(parents=True)
    deck_path = config.deck_dir / "lesson.yaml"
    deck_path.write_text(
        "deck:\n"
        "  kind: vocabulary\n"
        "  name: Lesson deck\n"
        "  deck_id: 1900000001\n"
        "  output: lesson.apkg\n"
        "  source: ../normalized/vocabulary.json\n"
        "  include_tags: [lesson]\n"
        "  cards:\n"
        "    recognition: true\n"
        "    production: false\n"
        "    reading: false\n",
        encoding="utf-8",
    )
    config.dist_dir.mkdir(parents=True)
    (config.dist_dir / "lesson.apkg").write_bytes(b"existing generated package")
    config.staging_dir.mkdir(parents=True)
    config.scan_inbox.mkdir(parents=True)
    config.patterns_file.write_text('{"version": 1, "documents": {}}\n', encoding="utf-8")
    config.kanji_file.write_text("{}\n", encoding="utf-8")
    config.operations_file.write_text('{"version": 3, "operations": []}\n', encoding="utf-8")
    book = ledger.Ledger(config.ledger_file)
    book.record_added(records[0].id)
    book.record_source_seen(records[0].id, "manual", "fixture")
    book.save()
    return config, records, deck_path


def _resources(config: ProjectConfig, kind: str) -> list[dict[str, object]]:
    catalog = json.loads(AssistantContextBroker(config).catalog().wire)
    return [
        item
        for item in catalog["data"]["resources"]
        if item["kind"] == kind
    ]


def _card_resource(config: ProjectConfig, title_prefix: str) -> str:
    matches = [
        item
        for item in _resources(config, "card")
        if str(item["title"]).startswith(title_prefix)
    ]
    assert len(matches) == 1
    return str(matches[0]["resource_id"])


def _deck_resource(config: ProjectConfig, title: str = "Lesson deck") -> str:
    matches = [item for item in _resources(config, "deck") if item["title"] == title]
    assert len(matches) == 1
    return str(matches[0]["resource_id"])


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_canonical_deletion_binds_full_cards_and_visible_retained_consequences(
    tmp_path: Path,
) -> None:
    config, records, _deck = _project(tmp_path)
    before = _files(tmp_path)

    plan = plan_canonical_deletion(
        config,
        card_resource_ids=(_card_resource(config, "食べる"),),
        record_ids=(records[0].id,),
        instruction="Permanently delete the 食べる canonical card.",
    )

    assert _files(tmp_path) == before
    projection = plan.projection
    assert projection["kind"] == "delete_canonical_cards"
    assert projection["selection"]["record_ids"] == [records[0].id]
    assert projection["selection"]["cards"][0]["record"] == assistant_record_value(
        records[0]
    )
    assert projection["canonical_collection"]["rows_before"] == 2
    assert projection["canonical_collection"]["rows_after"] == 1
    assert projection["canonical_collection"]["before_sha256"] == hashlib.sha256(
        config.normalized_file.read_bytes()
    ).hexdigest()
    assert projection["affected_decks"] == [
        {
            "name": "Lesson deck",
            "configured_file": "data/decks/lesson.yaml",
            "selected_before": [records[0].id],
            "selected_after_rebuild": [],
            "notes_before": 1,
            "notes_after_rebuild": 0,
            "declared_package": "dist/lesson.apkg",
        }
    ]
    assert projection["retained"] == {
        "configured_deck_definitions": True,
        "generated_packages": ["dist/lesson.apkg"],
        "ledger_history_for_record_ids": [records[0].id],
        "media_references": ["example-たべる.wav", "word-たべる.wav"],
        "staging_proposals": True,
    }
    assert projection["writes"] == {
        "replace_canonical_collection": "data/normalized/vocabulary.json",
        "remove_record_ids": [records[0].id],
    }
    inputs = {
        item["repository_file"]: item
        for item in projection["inputs"]["repository_files"]
    }
    assert set(inputs) == {
        "data/decks/lesson.yaml",
        "data/ledger.json",
        "data/normalized/vocabulary.json",
        "data/patterns.json",
    }
    assert all(item["exists"] and item["sha256"] for item in inputs.values())
    assert str(tmp_path) not in plan.projection_wire

    result = execute_canonical_deletion(config, plan)

    assert result.removed_record_ids == (records[0].id,)
    assert [record.id for record in load_records(config.normalized_file)] == [
        records[1].id
    ]
    assert (config.dist_dir / "lesson.apkg").read_bytes() == b"existing generated package"
    assert records[0].id in ledger.load(config.ledger_file).records


def test_canonical_deletion_refuses_stale_bytes_and_preserves_the_new_collection(
    tmp_path: Path,
) -> None:
    config, records, _deck = _project(tmp_path)
    plan = plan_canonical_deletion(
        config,
        card_resource_ids=(_card_resource(config, "食べる"),),
        record_ids=(records[0].id,),
        instruction="Delete this canonical card.",
    )
    replacement = config.normalized_file.read_bytes() + b" \n"
    config.normalized_file.write_bytes(replacement)

    with pytest.raises(AssistantDeletionError, match="changed after it was displayed"):
        execute_canonical_deletion(config, plan)

    assert config.normalized_file.read_bytes() == replacement


def test_canonical_deletion_replans_affected_decks_before_writing(
    tmp_path: Path,
) -> None:
    config, records, deck_path = _project(tmp_path)
    plan = plan_canonical_deletion(
        config,
        card_resource_ids=(_card_resource(config, "食べる"),),
        record_ids=(records[0].id,),
        instruction="Delete this canonical card.",
    )
    deck_path.write_text(
        deck_path.read_text(encoding="utf-8") + "  description: changed later\n",
        encoding="utf-8",
    )
    before = config.normalized_file.read_bytes()

    with pytest.raises(AssistantDeletionError, match="changed after it was displayed"):
        execute_canonical_deletion(config, plan)

    assert config.normalized_file.read_bytes() == before


def test_canonical_deletion_binds_an_unchanged_custom_deck_source_fingerprint(
    tmp_path: Path,
) -> None:
    config, records, _deck = _project(tmp_path)
    source = config.normalized_file.parent / "custom.json"
    source.write_text(
        json.dumps([records[1].to_dict()], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (config.deck_dir / "custom.yaml").write_text(
        "deck:\n"
        "  kind: vocabulary\n"
        "  name: Custom source deck\n"
        "  deck_id: 1900000002\n"
        "  source: ../normalized/custom.json\n",
        encoding="utf-8",
    )
    plan = plan_canonical_deletion(
        config,
        card_resource_ids=(_card_resource(config, "食べる"),),
        record_ids=(records[0].id,),
        instruction="Delete this canonical card.",
    )
    source.write_text(source.read_text(encoding="utf-8") + " \n", encoding="utf-8")
    before = config.normalized_file.read_bytes()

    with pytest.raises(AssistantDeletionError, match="changed after it was displayed"):
        execute_canonical_deletion(config, plan)

    assert config.normalized_file.read_bytes() == before


def test_canonical_deletion_requires_matching_opaque_card_resources(
    tmp_path: Path,
) -> None:
    config, records, _deck = _project(tmp_path)

    with pytest.raises(AssistantDeletionError, match="must identify the same cards"):
        plan_canonical_deletion(
            config,
            card_resource_ids=(_card_resource(config, "食べる"),),
            record_ids=(records[1].id,),
            instruction="Delete a mismatched card.",
        )

    with pytest.raises(AssistantDeletionError, match="supplied twice"):
        plan_canonical_deletion(
            config,
            card_resource_ids=(
                _card_resource(config, "食べる"),
                _card_resource(config, "食べる"),
            ),
            record_ids=(records[0].id,),
            instruction="Delete it twice.",
        )

    with pytest.raises(AssistantDeletionError, match="same order"):
        plan_canonical_deletion(
            config,
            card_resource_ids=(
                _card_resource(config, "食べる"),
                _card_resource(config, "読む"),
            ),
            record_ids=(records[1].id, records[0].id),
            instruction="Delete these exact cards in a mismatched order.",
        )


def test_canonical_deletion_refuses_pending_audio_for_the_selected_card(
    tmp_path: Path,
) -> None:
    config, records, _deck = _project(tmp_path)
    book = ledger.load(config.ledger_file)
    pending_key = book.pending_audio_key_for(
        records[0].id,
        of="word",
        target="pending.wav",
        request_input=records[0].expression,
        forced_accent=False,
        content_fp="a" * 64,
        provider="voicevox",
        voice=1,
        speed=1.0,
        settings={},
    )
    book.record_pending_audio(
        records[0].id,
        of="word",
        target="pending.wav",
        request_input=records[0].expression,
        forced_accent=False,
        content_fp="a" * 64,
        provider="voicevox",
        voice=1,
        speed=1.0,
        settings={},
        staged_file=f".pending/{pending_key}-{'b' * 64}.stage",
        staged_sha256="b" * 64,
    )
    book.save()
    before = config.normalized_file.read_bytes()

    with pytest.raises(AssistantDeletionError, match="pending audio recovery"):
        plan_canonical_deletion(
            config,
            card_resource_ids=(_card_resource(config, "食べる"),),
            record_ids=(records[0].id,),
            instruction="Delete this canonical card.",
        )

    assert config.normalized_file.read_bytes() == before


def test_canonical_deletion_refuses_a_symlink_without_touching_its_target(
    tmp_path: Path,
) -> None:
    config, records, _deck = _project(tmp_path)
    resource_id = _card_resource(config, "食べる")
    external = tmp_path / "external-records.json"
    external.write_bytes(config.normalized_file.read_bytes())
    config.normalized_file.unlink()
    config.normalized_file.symlink_to(external)

    with pytest.raises(
        AssistantDeletionError,
        match="Could not resolve|safely|symlink|escapes",
    ):
        plan_canonical_deletion(
            config,
            card_resource_ids=(resource_id,),
            record_ids=(records[0].id,),
            instruction="Delete this canonical card.",
        )

    assert load_records(external)[0].id == records[0].id


def test_deletion_refuses_an_external_configured_path_before_catalog_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, records, _deck = _project(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_bytes(config.normalized_file.read_bytes())
    unsafe = replace(config, normalized_file=outside)
    monkeypatch.setattr(
        assistant_deletion,
        "AssistantContextBroker",
        lambda *_args, **_kwargs: pytest.fail(
            "an external configured path must refuse before catalog access"
        ),
    )

    with pytest.raises(AssistantDeletionError, match="escapes the configured repository"):
        plan_canonical_deletion(
            unsafe,
            card_resource_ids=("opaque-card",),
            record_ids=(records[0].id,),
            instruction="Delete this canonical card.",
        )

    assert outside.exists()


def test_deletion_refuses_a_symlinked_deck_directory_before_enumerating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, records, _deck = _project(tmp_path)
    linked_decks = config.root / "linked-decks"
    config.deck_dir.rename(linked_decks)
    config.deck_dir.symlink_to(linked_decks, target_is_directory=True)
    monkeypatch.setattr(
        assistant_deletion.status,
        "deck_files",
        lambda *_args, **_kwargs: pytest.fail(
            "a symlinked configured directory must refuse before enumeration"
        ),
    )

    with pytest.raises(AssistantDeletionError, match="symlink"):
        plan_canonical_deletion(
            config,
            card_resource_ids=("opaque-card",),
            record_ids=(records[0].id,),
            instruction="Delete this canonical card.",
        )


def test_canonical_deletion_refuses_a_symlinked_deck_source(
    tmp_path: Path,
) -> None:
    config, records, _deck = _project(tmp_path)
    resource_id = _card_resource(config, "食べる")
    external = tmp_path / "other-records.json"
    external.write_bytes(config.normalized_file.read_bytes())
    linked = config.normalized_file.parent / "linked.json"
    linked.symlink_to(external)
    (config.deck_dir / "linked.yaml").write_text(
        "deck:\n"
        "  kind: vocabulary\n"
        "  name: Linked source deck\n"
        "  deck_id: 1900000002\n"
        "  output: linked.apkg\n"
        "  source: ../normalized/linked.json\n"
        "  cards:\n"
        "    recognition: true\n"
        "    production: false\n"
        "    reading: false\n",
        encoding="utf-8",
    )

    with pytest.raises(AssistantDeletionError, match="symlink|safely|non-regular"):
        plan_canonical_deletion(
            config,
            card_resource_ids=(resource_id,),
            record_ids=(records[0].id,),
            instruction="Delete this canonical card.",
        )


def test_two_canonical_confirmations_can_remove_the_card_only_once(
    tmp_path: Path,
) -> None:
    config, records, _deck = _project(tmp_path)
    plan = plan_canonical_deletion(
        config,
        card_resource_ids=(_card_resource(config, "食べる"),),
        record_ids=(records[0].id,),
        instruction="Delete this canonical card.",
    )

    def run() -> str:
        try:
            execute_canonical_deletion(config, plan)
        except AssistantDeletionError:
            return "refused"
        return "deleted"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(lambda _index: run(), range(2)))

    assert outcomes == ["deleted", "refused"]
    assert [record.id for record in load_records(config.normalized_file)] == [
        records[1].id
    ]


def test_canonical_deletion_never_follows_a_target_swapped_in_after_replan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, records, _deck = _project(tmp_path)
    plan = plan_canonical_deletion(
        config,
        card_resource_ids=(_card_resource(config, "食べる"),),
        record_ids=(records[0].id,),
        instruction="Delete this canonical card.",
    )
    external = tmp_path / "external-records.json"
    external.write_bytes(config.normalized_file.read_bytes())
    expected_external = external.read_bytes()
    bound_write = assistant_deletion.atomic_write_text_bound

    def race(path: Path, text: str, *, expected_revision: str) -> None:
        path.unlink()
        path.symlink_to(external)
        bound_write(path, text, expected_revision=expected_revision)

    monkeypatch.setattr(assistant_deletion, "atomic_write_text_bound", race)

    with pytest.raises(AssistantDeletionError, match="non-regular|safely|apply"):
        execute_canonical_deletion(config, plan)

    assert external.read_bytes() == expected_external
    assert config.normalized_file.is_symlink()


def test_deck_deletion_removes_only_exact_definition_and_retains_durable_content(
    tmp_path: Path,
) -> None:
    config, records, deck_path = _project(tmp_path)
    before_canonical = config.normalized_file.read_bytes()
    before_ledger = config.ledger_file.read_bytes()
    before_package = (config.dist_dir / "lesson.apkg").read_bytes()

    plan = plan_deck_deletion(
        config,
        deck_resource_id=_deck_resource(config),
        instruction="Permanently delete the Lesson deck definition.",
    )

    projection = plan.projection
    assert projection["kind"] == "delete_configured_deck"
    assert projection["target"] == {
        "resource_id": _deck_resource(config),
        "name": "Lesson deck",
        "kind": "vocabulary",
        "configured_file": "data/decks/lesson.yaml",
        "sha256": hashlib.sha256(deck_path.read_bytes()).hexdigest(),
    }
    assert projection["consequences"]["currently_selected_record_ids"] == [
        records[0].id
    ]
    assert projection["consequences"]["canonical_cards_removed"] == []
    assert projection["consequences"]["inline_only_record_ids_removed_from_library"] == []
    assert projection["retained"] == {
        "canonical_collection": "data/normalized/vocabulary.json",
        "generated_package": "dist/lesson.apkg",
        "ledger_history": True,
        "media": True,
        "staging_proposals": True,
    }
    assert projection["removals"] == ["data/decks/lesson.yaml"]
    assert {
        item["repository_file"]
        for item in projection["inputs"]["repository_files"]
    } == {
        "data/decks/lesson.yaml",
        "data/ledger.json",
        "data/normalized/vocabulary.json",
        "data/patterns.json",
    }

    result = execute_deck_deletion(config, plan)

    assert result.removed_deck_path == deck_path
    assert not deck_path.exists()
    assert config.normalized_file.read_bytes() == before_canonical
    assert config.ledger_file.read_bytes() == before_ledger
    assert (config.dist_dir / "lesson.apkg").read_bytes() == before_package


def test_deck_deletion_refuses_stale_or_replaced_target_without_unlinking_it(
    tmp_path: Path,
) -> None:
    config, _records, deck_path = _project(tmp_path)
    plan = plan_deck_deletion(
        config,
        deck_resource_id=_deck_resource(config),
        instruction="Delete this configured deck.",
    )
    replacement = deck_path.read_bytes().replace(b"Lesson deck", b"Changed deck")
    deck_path.write_bytes(replacement)

    with pytest.raises(AssistantDeletionError, match="changed after it was displayed"):
        execute_deck_deletion(config, plan)

    assert deck_path.read_bytes() == replacement


def test_deck_deletion_refuses_a_symlink_and_preserves_its_external_target(
    tmp_path: Path,
) -> None:
    config, _records, deck_path = _project(tmp_path)
    resource_id = _deck_resource(config)
    external = tmp_path / "external.yaml"
    external.write_bytes(deck_path.read_bytes())
    deck_path.unlink()
    deck_path.symlink_to(external)

    with pytest.raises(AssistantDeletionError, match="symlink|safely|regular"):
        plan_deck_deletion(
            config,
            deck_resource_id=resource_id,
            instruction="Delete this configured deck.",
        )

    assert external.exists()


def test_deck_deletion_binds_the_complete_configured_deck_set(
    tmp_path: Path,
) -> None:
    config, _records, deck_path = _project(tmp_path)
    plan = plan_deck_deletion(
        config,
        deck_resource_id=_deck_resource(config),
        instruction="Delete this configured deck.",
    )
    sibling = config.deck_dir / "new.yaml"
    sibling.write_text(
        "deck:\n  name: New deck\n  deck_id: 1900000002\n  notes: []\n",
        encoding="utf-8",
    )

    with pytest.raises(AssistantDeletionError, match="changed after it was displayed"):
        execute_deck_deletion(config, plan)

    assert deck_path.exists()
    assert sibling.exists()


def test_deck_deletion_names_inline_cards_that_leave_the_library(
    tmp_path: Path,
) -> None:
    config, _records, deck_path = _project(tmp_path)
    inline = _record("泳ぐ", "およぐ", tag="inline")
    deck_path.write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "kind": "vocabulary",
                    "name": "Lesson deck",
                    "deck_id": 1900000001,
                    "output": "lesson.apkg",
                },
                "notes": [inline.to_dict()],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    plan = plan_deck_deletion(
        config,
        deck_resource_id=_deck_resource(config),
        instruction="Delete this deck and its inline-only card.",
    )

    assert plan.projection["consequences"][
        "inline_only_record_ids_removed_from_library"
    ] == [inline.id]
    assert plan.projection["consequences"]["canonical_cards_removed"] == []


def test_deck_deletion_does_not_call_an_inline_id_lost_when_a_sibling_source_has_it(
    tmp_path: Path,
) -> None:
    config, _records, deck_path = _project(tmp_path)
    inline = _record("泳ぐ", "およぐ", tag="inline")
    deck_path.write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "kind": "vocabulary",
                    "name": "Lesson deck",
                    "deck_id": 1900000001,
                    "output": "lesson.apkg",
                },
                "notes": [inline.to_dict()],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    sibling_source = config.normalized_file.parent / "sibling.json"
    sibling_source.write_text(
        json.dumps([inline.to_dict()], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (config.deck_dir / "sibling.yaml").write_text(
        "deck:\n"
        "  kind: vocabulary\n"
        "  name: Sibling deck\n"
        "  deck_id: 1900000002\n"
        "  source: ../normalized/sibling.json\n",
        encoding="utf-8",
    )

    plan = plan_deck_deletion(
        config,
        deck_resource_id=_deck_resource(config),
        instruction="Delete this deck without misreporting its inline card.",
    )

    assert plan.projection["consequences"][
        "inline_only_record_ids_removed_from_library"
    ] == []


def test_deck_deletion_refuses_pending_audio_for_an_inline_only_card(
    tmp_path: Path,
) -> None:
    config, _records, deck_path = _project(tmp_path)
    inline = _record("泳ぐ", "およぐ", tag="inline")
    deck_path.write_text(
        yaml.safe_dump(
            {
                "deck": {
                    "kind": "vocabulary",
                    "name": "Lesson deck",
                    "deck_id": 1900000001,
                    "output": "lesson.apkg",
                },
                "notes": [inline.to_dict()],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    book = ledger.load(config.ledger_file)
    pending_key = book.pending_audio_key_for(
        inline.id,
        of="word",
        target="pending.wav",
        request_input=inline.expression,
        forced_accent=False,
        content_fp="c" * 64,
        provider="voicevox",
        voice=1,
        speed=1.0,
        settings={},
    )
    book.record_pending_audio(
        inline.id,
        of="word",
        target="pending.wav",
        request_input=inline.expression,
        forced_accent=False,
        content_fp="c" * 64,
        provider="voicevox",
        voice=1,
        speed=1.0,
        settings={},
        staged_file=f".pending/{pending_key}-{'d' * 64}.stage",
        staged_sha256="d" * 64,
    )
    book.save()

    with pytest.raises(AssistantDeletionError, match="pending.*audio recovery"):
        plan_deck_deletion(
            config,
            deck_resource_id=_deck_resource(config),
            instruction="Delete this configured deck.",
        )

    assert deck_path.exists()


def test_two_deck_confirmations_can_remove_the_definition_only_once(
    tmp_path: Path,
) -> None:
    config, _records, deck_path = _project(tmp_path)
    plan = plan_deck_deletion(
        config,
        deck_resource_id=_deck_resource(config),
        instruction="Delete this configured deck.",
    )

    def run() -> str:
        try:
            execute_deck_deletion(config, plan)
        except AssistantDeletionError:
            return "refused"
        return "deleted"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(lambda _index: run(), range(2)))

    assert outcomes == ["deleted", "refused"]
    assert not deck_path.exists()


def test_deck_deletion_never_follows_a_target_swapped_in_after_replan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _records, deck_path = _project(tmp_path)
    plan = plan_deck_deletion(
        config,
        deck_resource_id=_deck_resource(config),
        instruction="Delete this configured deck.",
    )
    external = tmp_path / "external.yaml"
    external.write_bytes(deck_path.read_bytes())
    expected_external = external.read_bytes()
    bound_unlink = assistant_deletion.atomic_unlink_bound

    def race(path: Path, *, expected_revision: str) -> None:
        path.unlink()
        path.symlink_to(external)
        bound_unlink(path, expected_revision=expected_revision)

    monkeypatch.setattr(assistant_deletion, "atomic_unlink_bound", race)

    with pytest.raises(AssistantDeletionError, match="safely|apply"):
        execute_deck_deletion(config, plan)

    assert external.read_bytes() == expected_external
    assert deck_path.is_symlink()
