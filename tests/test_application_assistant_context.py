"""Bounded repository context disclosed to the Assistant provider."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from japanese_anki import staging as staging_module
from japanese_anki.application import assistant_context as context_module
from japanese_anki.application import journey as journey_module
from japanese_anki.application.assistant_context import (
    AssistantContextBroker,
    AssistantContextError,
    ContextLimitError,
    ContextLimits,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.models import ExampleSentence

SOURCE_SECRET = "SOURCE-BYTES-MUST-NOT-BE-DISCLOSED"
RAW_SECRET = "RAW-FIELDS-MUST-NOT-BE-DISCLOSED"
PENDING_SECRET = "PENDING-REPLY-MUST-NOT-BE-DISCLOSED"
MEDIA_SECRET = "MEDIA-BYTES-MUST-NOT-BE-DISCLOSED"
OPERATION_SECRET = "OPERATION-DETAIL-MUST-NOT-BE-DISCLOSED"
WORD_MEDIA_PATH_SECRET = "WORD-MEDIA-PARENT-MUST-NOT-BE-DISCLOSED"
EXAMPLE_MEDIA_PATH_SECRET = "EXAMPLE-MEDIA-PARENT-MUST-NOT-BE-DISCLOSED"
HOST_PATH_SECRET = "HOST-PATH-MUST-NOT-BE-DISCLOSED"
PROPOSAL_FILE_SECRET = "PROPOSAL-RAW-BYTES-MUST-NOT-BE-DISCLOSED"
PACKAGE_BYTES_SECRET = "PACKAGE-BYTES-MUST-NOT-BE-DISCLOSED"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _project(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[project]\nname = \"Context fixture\"\n"
        "[paths]\nscan_inbox = \"data/inbox\"\n",
        encoding="utf-8",
    )
    records = [
        {
            "id": "word:食べる:たべる",
            "expression": "食べる",
            "reading": "たべる",
            "meanings": ["to eat"],
            "part_of_speech": "verb",
            "verb_group": "ichidan",
            "tags": ["lesson-1"],
            "usage_notes": "Use with を for the thing eaten.",
            "audio": f"/{WORD_MEDIA_PATH_SECRET}/word.wav",
            "image": f"/{WORD_MEDIA_PATH_SECRET}/food.png",
            "examples": [
                {
                    "japanese": "すしを食べます。",
                    "furigana": "すしを 食[た]べます。",
                    "english": "I eat sushi.",
                    "register": "polite",
                    "audio": f"C:\\{EXAMPLE_MEDIA_PATH_SECRET}\\sentence.mp3",
                }
            ],
            "source": {
                "type": "extract",
                "imported_from": "lesson.pdf",
                "row": 1,
                "raw_fields": {"private_note": RAW_SECRET},
            },
        },
        {
            "id": "word:遊ぶ:あそぶ",
            "expression": "遊ぶ",
            "reading": "あそぶ",
            "meanings": ["to play"],
            "part_of_speech": "verb",
            "verb_group": "godan",
            "tags": ["other"],
            "source": {"type": "manual"},
        },
    ]
    _write_json(tmp_path / "data/normalized/vocabulary.json", records)

    deck_dir = tmp_path / "data/decks"
    deck_dir.mkdir(parents=True)
    (deck_dir / "lesson.yaml").write_text(
        "deck:\n"
        "  name: Lesson One\n"
        "  description: Food from lesson one.\n"
        "  source: ../normalized/vocabulary.json\n"
        "  include_tags: [lesson-1]\n"
        "  cards:\n"
        "    recognition: true\n"
        "    production: false\n"
        f"  ignored_api_key: {RAW_SECRET}\n",
        encoding="utf-8",
    )
    (deck_dir / "potential.yaml").write_text(
        "deck:\n"
        "  kind: conjugation\n"
        "  form: potential\n"
        "  name: Potential practice\n"
        "  description: Practice saying what you can do.\n"
        "  source: ../normalized/vocabulary.json\n"
        "  include_ids: [word:食べる:たべる]\n",
        encoding="utf-8",
    )
    (deck_dir / "rules.yaml").write_text(
        "deck:\n"
        "  kind: pattern\n"
        "  name: Te-form rules\n"
        "  description: Rules from the reviewed chart.\n"
        "  document: rules.pdf\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "data/patterns.json",
        {
            "rules.pdf": {
                "kind": "pattern",
                "title": "Te-form chart",
                "reviewed": True,
                "patterns": [
                    {
                        "template": "う → って",
                        "gloss": "te-form ending",
                        "examples": ["かう ⇨ かって"],
                        "where": "row 1",
                    }
                ],
            }
        },
    )
    _write_json(
        tmp_path / "data/kanji.json",
        {
            "食": {
                "stroke_count": 9,
                "grade": 2,
                "jlpt": 5,
                "meanings": ["eat", "food"],
                "readings": [{"kind": "kun", "reading": "た(べる)"}],
                "strokes": ["M1 1L2 2"],
            }
        },
    )

    inbox = tmp_path / "data/inbox"
    inbox.mkdir(parents=True)
    (inbox / "lesson.pdf").write_bytes(SOURCE_SECRET.encode("utf-8"))
    pending = tmp_path / "data/.pending"
    pending.mkdir(parents=True)
    (pending / "reply.json").write_text(PENDING_SECRET, encoding="utf-8")
    media = tmp_path / "data/media/audio"
    media.mkdir(parents=True)
    (media / "clip.wav").write_bytes(MEDIA_SECRET.encode("utf-8"))

    _write_json(
        tmp_path / "data/operations.json",
        {
            "version": 1,
            "operations": {
                "00000000-0000-4000-8000-000000000001": {
                    "kind": "assistant_chat",
                    "state": "authorized",
                    "source_file": "lesson.pdf",
                    "source_sha256": "a" * 64,
                    "request_fp": "b" * 64,
                    "model": "claude-opus-5",
                    "authorized_at": "2026-09-02T00:00:00+00:00",
                    "updated_at": "2026-09-02T00:00:00+00:00",
                    "detail": OPERATION_SECRET,
                }
            },
        },
    )
    return ProjectConfig.load(tmp_path)


def _decoded(disclosure: object) -> dict[str, object]:
    return json.loads(disclosure.wire)  # type: ignore[attr-defined,no-any-return]


def _entry(catalog: dict[str, object], kind: str, title: str = "") -> dict[str, object]:
    entries = catalog["data"]["resources"]  # type: ignore[index]
    return next(
        item
        for item in entries  # type: ignore[union-attr]
        if item["kind"] == kind and (not title or item["title"] == title)
    )


def test_catalog_uses_opaque_ids_and_lists_every_supported_resource(
    tmp_path: Path,
) -> None:
    broker = AssistantContextBroker(_project(tmp_path))

    disclosure = broker.catalog()
    catalog = _decoded(disclosure)
    resources = catalog["data"]["resources"]

    assert {item["kind"] for item in resources} == {
        "card",
        "deck",
        "artifacts",
        "kanji",
        "operations",
        "patterns",
        "source",
        "status",
    }
    assert {item["title"] for item in resources if item["kind"] == "deck"} == {
        "Lesson One",
        "Potential practice",
        "Te-form rules",
    }
    for item in resources:
        resource_id = item["resource_id"]
        assert resource_id.startswith("resource_")
        assert len(resource_id) == len("resource_") + 32
        assert "/" not in resource_id
        assert "lesson" not in resource_id
    assert str(tmp_path) not in disclosure.wire
    assert "data/decks" not in disclosure.wire


def test_catalog_lists_sources_without_reading_the_completed_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog names sources; it must not pay to rank them.

    A `source` entry is an opaque id and a name. Ranking one reads every
    receipt in `staging/done/`, the pattern store, and deck ownership for every
    staged record — a second of every Assistant turn, disclosed nowhere in the
    catalog. The corrupt archive here makes the separation visible: it is
    enough to change the state and cannot change the name.
    """
    config = _project(tmp_path)
    archive = config.staging_dir / "done"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "unrelated.yaml").write_text(
        "records: []\npromotion_batches: broken\n", encoding="utf-8"
    )
    ranked = context_module.source_journeys
    monkeypatch.setattr(context_module, "source_journeys", pytest.fail)

    broker = AssistantContextBroker(config)
    source = _entry(_decoded(broker.catalog()), "source")

    assert source["title"] == "lesson.pdf"

    # The on-demand snapshot still discloses the full state, archive and all.
    monkeypatch.setattr(context_module, "source_journeys", ranked)
    value = _decoded(broker.snapshot(source["resource_id"]))  # type: ignore[arg-type]
    assert value["data"]["source"]["name"] == "lesson.pdf"


def test_vocabulary_deck_snapshot_contains_resolved_cards_and_teaching_config(
    tmp_path: Path,
) -> None:
    broker = AssistantContextBroker(_project(tmp_path))
    catalog = _decoded(broker.catalog())
    deck_id = _entry(catalog, "deck", "Lesson One")["resource_id"]

    disclosure = broker.snapshot(deck_id)  # type: ignore[arg-type]
    value = _decoded(disclosure)

    assert value["data"]["configuration"] == {
        "cards": {"production": False, "recognition": True},
        "description": "Food from lesson one.",
        "include_tags": ["lesson-1"],
        "kind": "vocabulary",
        "name": "Lesson One",
    }
    assert [card["expression"] for card in value["data"]["cards"]] == ["食べる"]
    assert value["data"]["cards"][0]["examples"][0]["japanese"] == "すしを食べます。"
    assert RAW_SECRET not in disclosure.wire


def test_trusted_deck_scope_returns_the_same_debited_context_without_wire_parsing(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    broker = AssistantContextBroker(config)

    resource_id = broker.resource_id_for_deck("data/decks/lesson.yaml")
    context = broker.deck_context(resource_id)

    assert context.resource_id == resource_id
    assert context.deck_kind == "vocabulary"
    assert context.configuration["name"] == "Lesson One"
    assert [record.expression for record in context.records] == ["食べる"]
    assert context.records[0].source.imported_from == "lesson.pdf"
    assert context.records[0].source.raw_fields == {}
    assert context.records[0].audio == "word.wav"
    assert context.records[0].image == "food.png"
    assert context.records[0].examples[0].audio == "sentence.mp3"
    assert broker.used_items == context.disclosure.item_count
    assert broker.used_utf8_bytes == context.disclosure.utf8_bytes


def test_project_status_helper_uses_the_same_snapshot_and_turn_budget(
    tmp_path: Path,
) -> None:
    broker = AssistantContextBroker(_project(tmp_path))

    disclosure = broker.project_status()

    assert disclosure.kind == "status"
    assert _decoded(disclosure)["data"]["records"]["total"] == 2
    assert broker.used_items == disclosure.item_count
    assert broker.used_utf8_bytes == disclosure.utf8_bytes


def test_source_snapshots_rank_from_the_broker_parse_not_a_second_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disclosing a source reuses the queue the broker already parsed.

    The catalog is built from one parse of the staging directory; a source
    snapshot that read the directory again would pay the half-megabyte parse a
    second time on the same turn and could rank a queue the catalog never saw.
    """
    config = _project(tmp_path)
    config.staging_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        config.staging_dir / "lesson.pdf.yaml",
        {
            "source_file": "lesson.pdf",
            "records": [
                {
                    "id": "word:食べる:たべる",
                    "expression": "食べる",
                    "meanings": ["placeholder"],
                    "source": {"type": "extract", "imported_from": "lesson.pdf"},
                }
            ],
        },
    )
    broker = AssistantContextBroker(config)
    catalog = _decoded(broker.catalog())
    [source] = [
        item
        for item in catalog["data"]["resources"]
        if item.get("kind") == "source" and "lesson.pdf" in item.values()
    ]
    monkeypatch.setattr(
        journey_module,
        "live_staging",
        lambda *_args, **_kwargs: pytest.fail("a source snapshot re-read the queue"),
    )

    disclosed = _decoded(broker.snapshot(source["resource_id"]))
    resolved = broker.source_path(source["resource_id"])

    assert "lesson.pdf" in json.dumps(disclosed, ensure_ascii=False)
    assert resolved.name.startswith("lesson.pdf")


def test_a_broker_build_parses_each_live_staging_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One Assistant turn, one parse of the review queue.

    Source names, proposal kinds and the staged counts in the project status
    are three projections of the same YAML, and a real staging file runs to
    half a megabyte — 90 ms a parse, paid three times over on every turn.
    Worse than the cost: three reads of one directory can disagree, so a
    catalog could name a proposal the status it disclosed had never seen.
    """
    config = _project(tmp_path)
    config.staging_dir.mkdir(parents=True, exist_ok=True)
    for name, source, record_id, expression in (
        ("lesson.pdf.yaml", "lesson.pdf", "word:食べる:たべる", "食べる"),
        ("week-2.pdf.yaml", "week-2.pdf", "word:遊ぶ:あそぶ", "遊ぶ"),
    ):
        _write_json(
            config.staging_dir / name,
            {
                "source_file": source,
                "records": [
                    {
                        "id": record_id,
                        "expression": expression,
                        "meanings": ["placeholder"],
                        "source": {"type": "extract", "imported_from": source},
                    }
                ],
            },
        )
    reads: list[str] = []
    real_read = staging_module.read_staging

    def counting_read(path: Path, *args: object, **kwargs: object) -> object:
        reads.append(Path(path).name)
        return real_read(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(staging_module, "read_staging", counting_read)

    broker = AssistantContextBroker(config)
    catalog = _decoded(broker.catalog())
    reported = _decoded(broker.project_status())

    assert sorted(reads) == ["lesson.pdf.yaml", "week-2.pdf.yaml"]
    # Each projection still answers for both files, from that one parse.
    resources = catalog["data"]["resources"]
    assert {
        item["title"] for item in resources if item["kind"] == "proposal"  # type: ignore[index,union-attr]
    } == {"lesson.pdf", "week-2.pdf"}
    assert {
        item["title"] for item in resources if item["kind"] == "source"  # type: ignore[index,union-attr]
    } == {"lesson.pdf", "week-2.pdf"}
    assert reported["data"]["staging"] == {"files": 2, "cards": 2}


def test_conjugation_snapshot_contains_the_resolved_drill_answer(
    tmp_path: Path,
) -> None:
    broker = AssistantContextBroker(_project(tmp_path))
    catalog = _decoded(broker.catalog())
    deck_id = _entry(catalog, "deck", "Potential practice")["resource_id"]

    value = _decoded(broker.snapshot(deck_id))  # type: ignore[arg-type]

    assert value["data"]["configuration"]["form"] == "potential"
    assert value["data"]["drills"] == [
        {
            "card": {
                "examples": ["ichidan"],
                "gloss": "to eat",
                "result": "食べられる",
                "trigger": "食べる（たべる）",
            },
            "record_id": "word:食べる:たべる",
        }
    ]
    assert value["data"]["cards"][0]["expression"] == "食べる"


def test_pattern_snapshot_contains_the_reviewed_patterns_and_resolved_cards(
    tmp_path: Path,
) -> None:
    broker = AssistantContextBroker(_project(tmp_path))
    catalog = _decoded(broker.catalog())
    deck_id = _entry(catalog, "deck", "Te-form rules")["resource_id"]

    value = _decoded(broker.snapshot(deck_id))  # type: ignore[arg-type]

    assert value["data"]["pattern_set"]["patterns"] == [
        {
            "examples": ["かう ⇨ かって"],
            "gloss": "te-form ending",
            "template": "う → って",
            "where": "row 1",
        }
    ]
    assert value["data"]["cards"] == [
        {
            "examples": ["かう ⇨ かって"],
            "gloss": "te-form ending",
            "result": "って",
            "trigger": "う",
        }
    ]


def test_search_is_a_case_sensitive_literal_substring_and_returns_real_cards(
    tmp_path: Path,
) -> None:
    broker = AssistantContextBroker(_project(tmp_path))

    matched = _decoded(broker.search_cards("遊ぶ"))
    case_mismatch = _decoded(broker.search_cards("Play"))
    field_name_is_not_content = _decoded(broker.search_cards("usage_notes"))

    assert matched["data"]["literal"] == "遊ぶ"
    assert matched["data"]["total_matches"] == 1
    assert matched["data"]["cards"][0]["card"]["meanings"] == ["to play"]
    assert case_mismatch["data"]["total_matches"] == 0
    assert field_name_is_not_content["data"]["total_matches"] == 0
    assert RAW_SECRET not in json.dumps(matched, ensure_ascii=False)


def test_card_snapshot_strips_importer_private_raw_fields_from_that_exact_card(
    tmp_path: Path,
) -> None:
    broker = AssistantContextBroker(_project(tmp_path))
    catalog = _decoded(broker.catalog())
    card_id = _entry(catalog, "card", "食べる（たべる）")["resource_id"]

    disclosure = broker.snapshot(card_id)  # type: ignore[arg-type]
    card = _decoded(disclosure)["data"]["card"]

    assert card["source"] == {"name": "lesson.pdf", "row": 1, "type": "extract"}
    assert "raw_fields" not in card["source"]
    assert RAW_SECRET not in disclosure.wire


def test_card_media_references_disclose_names_but_never_parent_paths(
    tmp_path: Path,
) -> None:
    broker = AssistantContextBroker(_project(tmp_path))
    catalog = _decoded(broker.catalog())
    card_id = _entry(catalog, "card", "食べる（たべる）")["resource_id"]

    disclosure = broker.snapshot(card_id)  # type: ignore[arg-type]
    card = _decoded(disclosure)["data"]["card"]

    assert card["audio"] == "word.wav"
    assert card["image"] == "food.png"
    assert card["examples"][0]["audio"] == "sentence.mp3"
    assert WORD_MEDIA_PATH_SECRET not in disclosure.wire
    assert EXAMPLE_MEDIA_PATH_SECRET not in disclosure.wire


def test_source_operation_and_other_excluded_bytes_never_enter_disclosures(
    tmp_path: Path,
) -> None:
    broker = AssistantContextBroker(_project(tmp_path))
    catalog = _decoded(broker.catalog())
    ids = {
        kind: _entry(catalog, kind)["resource_id"]
        for kind in ("source", "operations", "status")
    }

    wires = [
        broker.snapshot(resource_id).wire  # type: ignore[arg-type]
        for resource_id in ids.values()
    ]
    combined = "\n".join(wires)

    assert "lesson.pdf" in combined
    for secret in (
        SOURCE_SECRET,
        RAW_SECRET,
        PENDING_SECRET,
        MEDIA_SECRET,
        OPERATION_SECRET,
    ):
        assert secret not in combined
    operations = json.loads(wires[1])["data"]["operations"]
    assert operations[0]["has_captured_reply"] is False
    assert "detail" not in operations[0]
    assert "artifact" not in operations[0]


def test_pattern_and_kanji_stores_are_real_bounded_snapshots(tmp_path: Path) -> None:
    broker = AssistantContextBroker(_project(tmp_path))
    catalog = _decoded(broker.catalog())

    pattern_id = _entry(catalog, "patterns")["resource_id"]
    kanji_id = _entry(catalog, "kanji")["resource_id"]
    pattern_disclosure = broker.snapshot(pattern_id)  # type: ignore[arg-type]
    kanji_disclosure = broker.snapshot(kanji_id)  # type: ignore[arg-type]
    pattern_sets = _decoded(pattern_disclosure)["data"]["pattern_sets"]
    entries = _decoded(kanji_disclosure)["data"]["entries"]

    assert pattern_sets[0]["source_name"] == "rules.pdf"
    assert pattern_sets[0]["patterns"][0]["template"] == "う → って"
    assert entries == [
        {
            "character": "食",
            "grade": 2,
            "jlpt": 5,
            "meanings": ["eat", "food"],
            "readings": [{"kind": "kun", "reading": "た(べる)"}],
            "stroke_count": 9,
            "strokes": ["M1 1L2 2"],
        }
    ]
    assert pattern_disclosure.item_count == 1
    assert kanji_disclosure.item_count == 1


def test_kanji_guard_cannot_be_swapped_to_outside_symlink_before_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    broker = AssistantContextBroker(config)
    kanji_id = _entry(_decoded(broker.catalog()), "kanji")["resource_id"]
    original = config.kanji_file.with_suffix(".original")
    outside = tmp_path.parent / f"{tmp_path.name}-kanji-guard-race.json"
    _write_json(
        outside,
        {
            "秘": {
                "meanings": ["OUTSIDE-KANJI-MUST-NOT-CROSS-ASSISTANT-BOUNDARY"],
                "readings": [],
                "strokes": [],
            }
        },
    )
    real_load = context_module.kanji.load_store
    swapped = False

    def swap_after_guard(path: Path) -> object:
        nonlocal swapped
        swapped = True
        config.kanji_file.rename(original)
        config.kanji_file.symlink_to(outside)
        try:
            return real_load(path)
        finally:
            config.kanji_file.unlink()
            original.rename(config.kanji_file)

    monkeypatch.setattr(context_module.kanji, "load_store", swap_after_guard)

    with pytest.raises(AssistantContextError, match="symlink|Could not read"):
        broker.snapshot(kanji_id)  # type: ignore[arg-type]

    assert swapped


def test_unsafe_pattern_store_refuses_without_hiding_kanji(tmp_path: Path) -> None:
    config = _project(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-patterns.json"
    _write_json(outside, {"secret.pdf": {"patterns": []}})
    config.patterns_file.unlink()
    config.patterns_file.symlink_to(outside)
    broker = AssistantContextBroker(config)
    catalog = _decoded(broker.catalog())

    with pytest.raises(AssistantContextError, match="symlink"):
        broker.snapshot(_entry(catalog, "patterns")["resource_id"])  # type: ignore[arg-type]
    kanji = broker.snapshot(_entry(catalog, "kanji")["resource_id"])  # type: ignore[arg-type]
    assert _decoded(kanji)["data"]["entries"][0]["character"] == "食"


def test_readable_staging_proposals_are_individual_opaque_resources(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    config.staging_dir.mkdir(parents=True, exist_ok=True)
    record_id = "word:食べる:たべる"
    proposed = {
        "source_file": f"/{HOST_PATH_SECRET}/lesson.pdf",
        "model": "claude-opus-5",
        "coverage": {
            "version": 2,
            "status": "selection",
            "blocking": False,
            "source_units": [PROPOSAL_FILE_SECRET],
            "parsed_candidate_count": 1,
        },
        "pattern_set": {
            "kind": "lesson",
            "title": "Lesson grammar",
            "reviewed": False,
            "patterns": [
                {"template": "〜ます", "gloss": "polite", "examples": []}
            ],
            "prompt_provenance": {
                "request_fingerprint": "9" * 64,
                "raw_prompt": PROPOSAL_FILE_SECRET,
            },
        },
        "records": [
            {
                "id": record_id,
                "expression": "食べる",
                "reading": "たべる",
                "meanings": ["to eat"],
                "audio": f"/{HOST_PATH_SECRET}/word.wav",
                "source": {
                    "type": "extract",
                    "imported_from": f"/{HOST_PATH_SECRET}/lesson.pdf",
                    "raw_fields": {"private": PROPOSAL_FILE_SECRET},
                },
            }
        ],
    }
    _write_json(config.staging_dir / "lesson.pdf.yaml", proposed)
    (config.staging_dir / "broken.yaml").write_text("records: [", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-proposal.yaml"
    outside.write_text(PROPOSAL_FILE_SECRET, encoding="utf-8")
    (config.staging_dir / "linked.yaml").symlink_to(outside)

    broker = AssistantContextBroker(config)
    catalog = _decoded(broker.catalog())
    proposals = [
        item for item in catalog["data"]["resources"] if item["kind"] == "proposal"
    ]

    assert [(item["title"], item["proposal_kind"]) for item in proposals] == [
        ("lesson.pdf", "source_extraction")
    ]
    disclosure = broker.snapshot(proposals[0]["resource_id"])
    value = _decoded(disclosure)["data"]
    assert value["cards"][0]["expression"] == "食べる"
    assert value["cards"][0]["audio"] == "word.wav"
    assert value["metadata"]["source_name"] == "lesson.pdf"
    assert "source_units" not in value["metadata"]["coverage"]
    assert value["metadata"]["pattern_set"]["prompt_provenance"] == {
        "request_fingerprint": "9" * 64
    }
    assert HOST_PATH_SECRET not in disclosure.wire
    assert PROPOSAL_FILE_SECRET not in disclosure.wire


def test_generic_card_revision_and_deck_revision_have_safe_exact_projections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    config.staging_dir.mkdir(parents=True, exist_ok=True)
    operation_id = "12345678-1234-4234-8234-123456789abc"
    record_id = "word:食べる:たべる"
    _write_json(
        config.staging_dir / f"card-revision-{operation_id}.yaml",
        {
            "source_file": f"/{HOST_PATH_SECRET}/vocabulary.json",
            "model": "claude-opus-5",
            "provider": "claude-code",
            "card_revision": {
                "version": 1,
                "operation_id": operation_id,
                "request_fingerprint": "a" * 64,
                "provider": "claude-code",
                "attribution_provider": "anthropic",
                "model": "claude-opus-5",
                "input_fingerprints": {record_id: "b" * 64},
                "fields": {record_id: ["usage_notes"]},
            },
            "field_replacements": {
                "version": 1,
                "records": {record_id: {"usage_notes": "c" * 64}},
            },
            "records": [
                {
                    "id": record_id,
                    "expression": "食べる",
                    "reading": "たべる",
                    "meanings": ["to eat"],
                    "usage_notes": "Proposed note.",
                    "source": {"type": "extract"},
                }
            ],
        },
    )
    deck_revision = config.staging_dir / "revise-example.json"
    deck_revision.write_text(PROPOSAL_FILE_SECRET, encoding="utf-8")
    fake = SimpleNamespace(
        deck_relative_path=f"data/decks/{HOST_PATH_SECRET}/potential.yaml",
        state="proposed",
        selected_record_ids=(record_id,),
        current_form_note="Old note",
        current_drill_examples={
            record_id: (ExampleSentence(japanese="前です。", audio="/private/old.wav"),)
        },
        form_note="New note",
        drill_examples={
            record_id: (ExampleSentence(japanese="後です。", audio="/private/new.wav"),)
        },
        operation_id=operation_id,
        request_fingerprint="d" * 64,
        provider="claude-code",
        billing_class="subscription",
        request_bytes_sha256="e" * 64,
        proposal_sha256="f" * 64,
        plan_fingerprint="1" * 64,
    )
    monkeypatch.setattr(
        context_module.revision_apply,
        "plan_revision_apply",
        lambda received_config, path: fake,
    )

    broker = AssistantContextBroker(config)
    catalog = _decoded(broker.catalog())
    entries = {
        item["proposal_kind"]: item
        for item in catalog["data"]["resources"]
        if item["kind"] == "proposal"
    }
    card = broker.snapshot(entries["card_revision"]["resource_id"])
    deck = broker.snapshot(entries["deck_revision"]["resource_id"])
    card_value = _decoded(card)["data"]
    deck_value = _decoded(deck)["data"]

    assert card_value["cards"][0]["usage_notes"] == "Proposed note."
    assert card_value["metadata"]["card_revision"]["fields"] == {
        record_id: ["usage_notes"]
    }
    assert deck_value["target_name"] == "potential.yaml"
    assert deck_value["current"]["drill_examples"][record_id][0]["audio"] == "old.wav"
    assert deck_value["proposed"]["drill_examples"][record_id][0]["audio"] == "new.wav"
    assert deck_value["proposed"]["form_note"] == "New note"
    assert HOST_PATH_SECRET not in card.wire + deck.wire
    assert PROPOSAL_FILE_SECRET not in card.wire + deck.wire


def test_artifact_projection_counts_without_reading_bytes_or_pending_media(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    pending = config.media_dir / ".pending"
    pending.mkdir(parents=True)
    (pending / "private.stage").write_text(PENDING_SECRET, encoding="utf-8")
    config.dist_dir.mkdir(parents=True)
    (config.dist_dir / "lesson.apkg").write_text(
        PACKAGE_BYTES_SECRET, encoding="utf-8"
    )
    (config.dist_dir / "lesson-preview.html").write_text(
        PACKAGE_BYTES_SECRET, encoding="utf-8"
    )

    broker = AssistantContextBroker(config)
    catalog = _decoded(broker.catalog())
    disclosure = broker.snapshot(_entry(catalog, "artifacts")["resource_id"])  # type: ignore[arg-type]
    value = _decoded(disclosure)["data"]

    assert value["media"]["files"] == 1
    assert value["media"]["by_extension"] == {".wav": 1}
    assert value["packages"] == [
        {
            "bytes": len(PACKAGE_BYTES_SECRET),
            "kind": "preview",
            "name": "lesson-preview.html",
        },
        {
            "bytes": len(PACKAGE_BYTES_SECRET),
            "kind": "anki_package",
            "name": "lesson.apkg",
        },
    ]
    assert PENDING_SECRET not in disclosure.wire
    assert PACKAGE_BYTES_SECRET not in disclosure.wire
    assert str(tmp_path) not in disclosure.wire


def test_fingerprint_is_of_the_exact_disclosed_utf8_json(tmp_path: Path) -> None:
    broker = AssistantContextBroker(_project(tmp_path))

    disclosure = broker.catalog()

    exact = disclosure.wire.encode("utf-8")
    assert disclosure.utf8_bytes == len(exact)
    assert disclosure.sha256 == hashlib.sha256(exact).hexdigest()
    assert disclosure.item_count == len(_decoded(disclosure)["data"]["resources"])


def test_result_and_turn_budgets_refuse_without_partially_spending_the_budget(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    too_small = AssistantContextBroker(
        config,
        limits=ContextLimits(
            max_result_items=1,
            max_result_bytes=100_000,
            max_turn_items=100,
            max_turn_bytes=200_000,
        ),
    )

    with pytest.raises(ContextLimitError, match="result item limit"):
        too_small.catalog()
    assert too_small.used_items == 0
    assert too_small.used_utf8_bytes == 0

    probe = AssistantContextBroker(config)
    catalog_disclosure = probe.catalog()
    catalog = _decoded(catalog_disclosure)
    card_id = _entry(catalog, "card", "食べる（たべる）")["resource_id"]
    one_card = probe.snapshot(card_id)  # type: ignore[arg-type]

    bounded = AssistantContextBroker(
        config,
        limits=ContextLimits(
            max_result_items=catalog_disclosure.item_count,
            max_result_bytes=max(
                catalog_disclosure.utf8_bytes, one_card.utf8_bytes
            ),
            max_turn_items=catalog_disclosure.item_count + 1,
            max_turn_bytes=catalog_disclosure.utf8_bytes + one_card.utf8_bytes,
        ),
    )
    bounded_catalog = _decoded(bounded.catalog())
    bounded_id = _entry(bounded_catalog, "card", "食べる（たべる）")["resource_id"]
    bounded.snapshot(bounded_id)  # type: ignore[arg-type]
    used = (bounded.used_items, bounded.used_utf8_bytes)

    with pytest.raises(ContextLimitError, match="turn item limit"):
        bounded.snapshot(bounded_id)  # type: ignore[arg-type]
    assert (bounded.used_items, bounded.used_utf8_bytes) == used


def test_unknown_ids_are_not_interpreted_as_paths(tmp_path: Path) -> None:
    broker = AssistantContextBroker(_project(tmp_path))

    with pytest.raises(AssistantContextError, match="Unknown Assistant resource"):
        broker.snapshot("../../data/.pending/reply.json")


def test_unsafe_decks_refuse_individually_without_hiding_safe_decks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.yaml"
    outside.write_text("deck:\n  name: Outside\n", encoding="utf-8")
    linked = config.deck_dir / "linked.yaml"
    linked.symlink_to(outside)
    forbidden = {linked.absolute()}
    real_read = context_module.read_bytes_bound

    def read_only_safe_paths(path: Path) -> bytes:
        if Path(path).absolute() in forbidden:
            pytest.fail("an unsafe path reached the existing bound reader")
        return real_read(path)

    monkeypatch.setattr(context_module, "read_bytes_bound", read_only_safe_paths)

    broker = AssistantContextBroker(config)
    catalog = _decoded(broker.catalog())
    linked_entry = _entry(catalog, "deck", "linked")
    assert linked_entry["available"] is False
    safe_id = _entry(catalog, "deck", "Lesson One")["resource_id"]
    assert _decoded(broker.snapshot(safe_id))["data"]["cards"]  # type: ignore[arg-type]
    with pytest.raises(AssistantContextError, match="symlink"):
        broker.snapshot(linked_entry["resource_id"])  # type: ignore[arg-type]

    linked.unlink()
    outside_records = tmp_path.parent / f"{tmp_path.name}-outside.json"
    _write_json(outside_records, [])
    forbidden.add(outside_records.absolute())
    (config.deck_dir / "lesson.yaml").write_text(
        "deck:\n"
        "  name: Escaping deck\n"
        f"  source: ../../../{outside_records.name}\n",
        encoding="utf-8",
    )

    broker = AssistantContextBroker(config)
    catalog = _decoded(broker.catalog())
    escaping_entry = _entry(catalog, "deck", "Escaping deck")
    assert escaping_entry["available"] is False
    safe_id = _entry(catalog, "deck", "Potential practice")["resource_id"]
    assert _decoded(broker.snapshot(safe_id))["data"]["drills"]  # type: ignore[arg-type]
    with pytest.raises(AssistantContextError, match="escapes the project"):
        broker.snapshot(escaping_entry["resource_id"])  # type: ignore[arg-type]


def test_deck_guard_cannot_be_swapped_to_outside_symlink_before_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    deck = config.deck_dir / "lesson.yaml"
    original = deck.with_suffix(".original")
    outside = tmp_path.parent / f"{tmp_path.name}-guard-race.yaml"
    outside_marker = "OUTSIDE-DECK-MUST-NOT-CROSS-ASSISTANT-BOUNDARY"
    outside.write_text(f"deck:\n  name: {outside_marker}\n", encoding="utf-8")
    real_load = context_module.load_structured
    swapped = False

    def swap_after_guard(path: Path, **kwargs: object) -> object:
        nonlocal swapped
        if Path(path) == deck and not swapped:
            swapped = True
            deck.rename(original)
            deck.symlink_to(outside)
            try:
                return real_load(path, **kwargs)
            finally:
                deck.unlink()
                original.rename(deck)
        return real_load(path, **kwargs)

    monkeypatch.setattr(context_module, "load_structured", swap_after_guard)

    catalog = AssistantContextBroker(config).catalog()

    assert swapped
    assert outside_marker not in catalog.wire
    entry = _entry(_decoded(catalog), "deck", "lesson")
    assert entry["available"] is False


def test_malformed_deck_is_listed_but_does_not_disable_an_unrelated_deck(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    (config.deck_dir / "broken.yaml").write_text("deck: [", encoding="utf-8")

    broker = AssistantContextBroker(config)
    catalog = _decoded(broker.catalog())

    broken = _entry(catalog, "deck", "broken")
    assert broken["available"] is False
    safe_id = _entry(catalog, "deck", "Lesson One")["resource_id"]
    assert _decoded(broker.snapshot(safe_id))["data"]["cards"]  # type: ignore[arg-type]
    with pytest.raises(AssistantContextError, match="Could not read|parse"):
        broker.snapshot(broken["resource_id"])  # type: ignore[arg-type]


def test_unsafe_deck_does_not_erase_readable_cards_from_the_catalog(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    (config.deck_dir / "inline.yaml").write_text(
        "deck:\n"
        "  name: Inline deck\n"
        "notes:\n"
        "  - id: word:inline:inline\n"
        "    expression: inline\n"
        "    reading: inline\n"
        "    meanings: [readable inline card]\n"
        "    source: {type: manual}\n",
        encoding="utf-8",
    )
    outside_records = tmp_path.parent / f"{tmp_path.name}-outside-cards.json"
    _write_json(outside_records, [])
    (config.deck_dir / "lesson.yaml").write_text(
        "deck:\n"
        "  name: Escaping deck\n"
        f"  source: ../../../{outside_records.name}\n",
        encoding="utf-8",
    )

    broker = AssistantContextBroker(config)
    catalog = _decoded(broker.catalog())

    assert _entry(catalog, "deck", "Escaping deck")["available"] is False
    card_ids = {
        item["title"]: item["resource_id"]
        for item in catalog["data"]["resources"]  # type: ignore[index,union-attr]
        if item["kind"] == "card"
    }
    assert set(card_ids) == {"inline", "遊ぶ（あそぶ）", "食べる（たべる）"}
    card = _decoded(broker.snapshot(card_ids["食べる（たべる）"]))  # type: ignore[arg-type]
    assert card["data"]["card"]["expression"] == "食べる"


def test_unsafe_deck_does_not_break_project_status_or_reach_its_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    outside_records = tmp_path.parent / f"{tmp_path.name}-outside-status.json"
    _write_json(outside_records, [])
    unsafe_deck = config.deck_dir / "lesson.yaml"
    unsafe_deck.write_text(
        "deck:\n"
        "  name: Escaping deck\n"
        f"  source: ../../../{outside_records.name}\n",
        encoding="utf-8",
    )
    real_resolve = context_module.status.resolve_deck_records

    def reject_unsafe_deck(path: Path) -> object:
        if path == unsafe_deck:
            pytest.fail("the unavailable deck reached aggregate status collection")
        return real_resolve(path)

    monkeypatch.setattr(context_module.status, "resolve_deck_records", reject_unsafe_deck)

    broker = AssistantContextBroker(config)
    status_value = _decoded(broker.project_status())

    assert status_value["data"]["records"]["total"] == 2
    assert {deck["stem"] for deck in status_value["data"]["decks"]} == {
        "potential",
        "rules",
    }


def test_unrelated_unsafe_inbox_and_staging_entries_do_not_disable_reads(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    outside_source = tmp_path.parent / f"{tmp_path.name}-private-source.pdf"
    outside_source.write_text(SOURCE_SECRET, encoding="utf-8")
    (config.scan_inbox / "unsafe.pdf").symlink_to(outside_source)
    outside_stage = tmp_path.parent / f"{tmp_path.name}-private-stage.yaml"
    outside_stage.write_text(f"private: {PENDING_SECRET}\n", encoding="utf-8")
    config.staging_dir.mkdir(parents=True, exist_ok=True)
    (config.staging_dir / "unsafe.yaml").symlink_to(outside_stage)

    broker = AssistantContextBroker(config)
    catalog = _decoded(broker.catalog())
    safe_id = _entry(catalog, "deck", "Lesson One")["resource_id"]

    deck = broker.snapshot(safe_id)  # type: ignore[arg-type]
    project_status = broker.project_status()

    assert _decoded(deck)["data"]["cards"]
    assert _decoded(project_status)["data"]["records"]["total"] == 2
    assert SOURCE_SECRET not in deck.wire + project_status.wire
    assert PENDING_SECRET not in deck.wire + project_status.wire


def test_configured_directory_parent_traversal_is_rejected_before_scan(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-decks"
    outside.mkdir()
    (outside / "private.yaml").write_text(
        "deck:\n  name: Outside private deck\n", encoding="utf-8"
    )
    traversing = tmp_path / "data/decks/../../../" / outside.name

    with pytest.raises(AssistantContextError, match="escapes the project"):
        AssistantContextBroker(replace(config, deck_dir=traversing))


def test_a_deck_cannot_gain_path_traversal_after_its_opaque_id_was_issued(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    broker = AssistantContextBroker(config)
    resource_id = broker.resource_id_for_deck("data/decks/lesson.yaml")
    outside_records = tmp_path.parent / f"{tmp_path.name}-raced-outside.json"
    _write_json(outside_records, [])
    (config.deck_dir / "lesson.yaml").write_text(
        "deck:\n"
        "  name: Escaping deck\n"
        f"  source: ../../../{outside_records.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(AssistantContextError, match="escapes the project"):
        broker.snapshot(resource_id)


def test_a_standalone_decks_saved_scope_is_disclosed_and_secrets_stay_hidden(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    scoped = config.deck_dir / "lesson.yaml"
    scoped.write_text(
        scoped.read_text(encoding="utf-8") + f"  scope_id: {'a' * 64}\n",
        encoding="utf-8",
    )
    broker = AssistantContextBroker(config)

    disclosure = broker.snapshot(
        _entry(_decoded(broker.catalog()), "deck", "Lesson One")["resource_id"]  # type: ignore[arg-type]
    )
    value = _decoded(disclosure)

    assert value["data"]["configuration"]["scope_id"] == "a" * 64
    assert RAW_SECRET not in disclosure.wire
    assert "ignored_api_key" not in disclosure.wire

    # A deck that saved no scope is shared, and the snapshot says nothing at all
    # about one rather than inventing a default to report.
    (tmp_path / "shared").mkdir()
    plain = AssistantContextBroker(_project(tmp_path / "shared"))
    plain_value = _decoded(
        plain.snapshot(
            _entry(_decoded(plain.catalog()), "deck", "Lesson One")["resource_id"]  # type: ignore[arg-type]
        )
    )

    assert "scope_id" not in plain_value["data"]["configuration"]
