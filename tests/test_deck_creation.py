"""W4.1's explicit, preview-first study-deck creator."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from japanese_anki.application import deck_creation
from japanese_anki.application.deck_creation import (
    StudyDeckCreationError,
    create_study_deck,
    plan_study_deck,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.exporters.anki import deck_selection, resolve_deck_records

CONFIG = """
[paths]
normalized_file = "vocabulary.json"
deck_dir = "decks"
"""


def _project(tmp_path: Path) -> ProjectConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text("[]\n", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _existing_deck(
    config: ProjectConfig,
    stem: str,
    *,
    intake_tag: str | None = None,
    include_tags: list[str] | None = None,
    include_ids: list[str] | None = None,
    exclude_ids: list[str] | None = None,
    output: str | None = None,
    deck_id: int = 1_500_000_001,
    scope_id: str | None = None,
) -> Path:
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    section: dict[str, object] = {
        "name": stem,
        "deck_id": deck_id,
        "source": "../vocabulary.json",
    }
    if scope_id is not None:
        section["scope_id"] = scope_id
    if include_tags is not None:
        section["include_tags"] = include_tags
    if include_ids is not None:
        section["include_ids"] = include_ids
    if exclude_ids is not None:
        section["exclude_ids"] = exclude_ids
    if output is not None:
        section["output"] = output
    if intake_tag is not None:
        section["intake_tag"] = intake_tag
    path = config.deck_dir / f"{stem}.yaml"
    path.write_text(
        yaml.safe_dump({"deck": section}, sort_keys=False), encoding="utf-8"
    )
    return path


def test_plan_previews_exact_canonical_deck_and_create_is_explicit(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)

    plan = plan_study_deck(config, name="日本語", production=True)

    assert re.fullmatch(r"study-deck-[0-9a-f]{10}", plan.stem)
    assert plan.intake_tag == f"janki:deck:{plan.stem}"
    assert plan.path == config.deck_dir / f"{plan.stem}.yaml"
    assert plan.output_path == (config.dist_dir / f"{plan.stem}.apkg").resolve()
    assert not config.deck_dir.exists()
    document = yaml.safe_load(plan.yaml_bytes)
    assert document == {
        "deck": {
            "kind": "vocabulary",
            "name": "日本語",
            "deck_id": plan.deck_id,
            "output": f"{plan.stem}.apkg",
            "cards": {
                "recognition": True,
                "production": True,
                "reading": False,
            },
            "source": "../vocabulary.json",
            "intake_tag": plan.intake_tag,
            "include_tags": [plan.intake_tag],
        }
    }

    result = create_study_deck(config, plan)

    assert result.path == plan.path
    assert result.yaml_bytes == plan.yaml_bytes
    assert plan.path.read_bytes() == plan.yaml_bytes
    deck_config, records = resolve_deck_records(plan.path)
    selection = deck_selection(deck_config, plan.path)
    assert records == []
    assert selection.intake_tag == plan.intake_tag
    assert (plan.path.parent / deck_config["source"]).resolve() == config.normalized_file


#: The exact domain prefix a standalone vocabulary scope is derived under. Written
#: out here rather than imported so a changed derivation fails this test instead of
#: quietly moving every future scope with it.
SCOPE_DOMAIN = "janki:deck-scope:v1"


def _expected_scope(stem: str, counter: int) -> str:
    return hashlib.sha256(f"{SCOPE_DOMAIN}:{stem}:{counter}".encode("ascii")).hexdigest()


def _canonical_records(config: ProjectConfig, ids: list[str]) -> None:
    config.normalized_file.parent.mkdir(parents=True, exist_ok=True)
    config.normalized_file.write_text(
        json.dumps(
            [
                {"id": record_id, "expression": "one", "reading": "one"}
                for record_id in ids
            ]
        ),
        encoding="utf-8",
    )


def test_standalone_plan_derives_one_scope_and_still_writes_nothing(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)

    plan = plan_study_deck(config, name="Genki 1", standalone=True)

    assert plan.standalone is True
    assert re.fullmatch(r"[0-9a-f]{64}", plan.scope_id)
    assert plan.scope_id == _expected_scope(plan.stem, 0)
    assert not config.deck_dir.exists()
    assert yaml.safe_load(plan.yaml_bytes) == {
        "deck": {
            "kind": "vocabulary",
            "name": "Genki 1",
            "deck_id": plan.deck_id,
            "output": f"{plan.stem}.apkg",
            "cards": {
                "recognition": True,
                "production": False,
                "reading": False,
            },
            "source": "../vocabulary.json",
            "intake_tag": plan.intake_tag,
            "include_tags": [plan.intake_tag],
            "scope_id": plan.scope_id,
        }
    }
    assert plan == plan_study_deck(config, name="Genki 1", standalone=True)

    created = create_study_deck(config, plan)

    assert yaml.safe_load(created.yaml_bytes)["deck"]["scope_id"] == plan.scope_id
    assert plan.path.read_bytes() == plan.yaml_bytes


def test_shared_creation_keeps_its_definition_and_carries_no_scope(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)

    plan = plan_study_deck(config, name="Genki 1")

    assert plan.standalone is False
    assert plan.scope_id == ""
    assert "scope_id" not in yaml.safe_load(plan.yaml_bytes)["deck"]
    assert plan == plan_study_deck(config, name="Genki 1", standalone=False)


@pytest.mark.parametrize("value", [1, 0, "true", None], ids=("one", "zero", "text", "none"))
def test_standalone_must_be_an_explicit_boolean(tmp_path: Path, value: object) -> None:
    config = _project(tmp_path)

    with pytest.raises(StudyDeckCreationError, match="true or false"):
        plan_study_deck(config, name="Genki 1", standalone=value)  # type: ignore[arg-type]


def test_only_a_vocabulary_deck_can_hold_standalone_copies(tmp_path: Path) -> None:
    config = _project(tmp_path)

    with pytest.raises(StudyDeckCreationError, match="[Oo]nly a vocabulary deck"):
        plan_study_deck(
            config,
            name="Week 2 Characters",
            kind="kanji",
            include_ids=["kanji:理"],
            standalone=True,
        )


def test_scope_skips_ids_held_by_a_renamed_deck_or_a_surviving_record(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    stem = plan_study_deck(config, name="Genki 1", standalone=True).stem
    # A deck whose display name and file stem have nothing to do with the scope
    # it holds: this is what a rename leaves behind, and its Anki history is
    # still keyed on that scope.
    _existing_deck(
        config,
        "some-other-stem",
        intake_tag="janki:deck:some-other-stem",
        include_tags=["janki:deck:some-other-stem"],
        scope_id=_expected_scope(stem, 0),
    )
    # A canonical record that outlived the deck file it was created for.
    _canonical_records(config, [f"standalone:{_expected_scope(stem, 1)}:one:one"])

    plan = plan_study_deck(config, name="Genki 1", standalone=True)

    assert plan.scope_id == _expected_scope(stem, 2)


def test_a_fresh_scope_collision_stales_the_plan_before_any_write(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    plan = plan_study_deck(config, name="Genki 1", standalone=True)
    # Surviving canonical records do not move the deck-set fingerprint, so only a
    # repeated scope choice can catch this one.
    _canonical_records(config, [f"standalone:{plan.scope_id}:one:one"])

    with pytest.raises(StudyDeckCreationError, match="preview the study deck again"):
        create_study_deck(config, plan)

    assert not plan.path.exists()


def test_creating_a_standalone_deck_changes_no_canonical_identity(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    _canonical_records(config, ["word:one:one"])
    before = config.normalized_file.read_bytes()

    plan = plan_study_deck(config, name="Genki 1", standalone=True)
    create_study_deck(config, plan)

    assert config.normalized_file.read_bytes() == before

    document = yaml.safe_load(plan.path.read_text(encoding="utf-8"))
    document["deck"]["name"] = "Genki 1 Renamed"
    plan.path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    renamed = yaml.safe_load(plan.path.read_text(encoding="utf-8"))["deck"]

    assert renamed["scope_id"] == plan.scope_id
    assert config.normalized_file.read_bytes() == before


def test_the_cross_selection_probe_carries_the_scope_it_probes_for(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    plan = plan_study_deck(config, name="Genki 1", standalone=True)

    standalone_probe = deck_creation._cross_selection_probe(
        (), plan.intake_tag, plan.scope_id
    )
    shared_probe = deck_creation._cross_selection_probe((), plan.intake_tag, "")

    assert standalone_probe.tags == (plan.intake_tag,)
    assert standalone_probe.id.startswith(f"standalone:{plan.scope_id}:")
    assert shared_probe.id == "word:assignment:assignment"


def test_inputs_are_strict_and_name_derivation_is_stable(tmp_path: Path) -> None:
    config = _project(tmp_path)
    first = plan_study_deck(
        config,
        name="  Genki 1  ",
        recognition=False,
        reading=True,
    )
    again = plan_study_deck(
        config,
        name="Genki 1",
        recognition=False,
        reading=True,
    )

    assert first == again
    assert re.fullmatch(r"genki-1-[0-9a-f]{10}", first.stem)
    assert yaml.safe_load(first.yaml_bytes)["deck"]["cards"] == {
        "recognition": False,
        "production": False,
        "reading": True,
    }
    _existing_deck(
        config,
        "reserved-id",
        intake_tag="reserved-id",
        include_tags=["reserved-id"],
        deck_id=first.deck_id,
    )
    assert plan_study_deck(
        config, name="Genki 1", recognition=False, reading=True
    ).deck_id != first.deck_id
    with pytest.raises(StudyDeckCreationError, match="learner-facing name"):
        plan_study_deck(config, name=" \t")
    with pytest.raises(StudyDeckCreationError, match="at least one"):
        plan_study_deck(
            config,
            name="No cards",
            recognition=False,
            production=False,
            reading=False,
        )
    with pytest.raises(StudyDeckCreationError, match="true or false"):
        plan_study_deck(config, name="Wrong toggle", recognition=1)  # type: ignore[arg-type]


def test_creator_refuses_duplicate_or_cross_selecting_intake_tags(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    wanted = plan_study_deck(config, name="Lesson 12")
    _existing_deck(
        config,
        "other",
        intake_tag=wanted.intake_tag,
        include_tags=[wanted.intake_tag],
    )

    with pytest.raises(StudyDeckCreationError, match="already declares intake tag"):
        plan_study_deck(config, name="Lesson 12")

    config = _project(tmp_path / "exact-list")
    _existing_deck(
        config,
        "fixed-cards",
        include_ids=["word:fixed:fixed"],
    )
    assert plan_study_deck(config, name="Lesson 13").path.parent == config.deck_dir

    config = _project(tmp_path / "output-collision")
    wanted = plan_study_deck(config, name="Lesson 13")
    _existing_deck(
        config,
        "other-output",
        include_tags=["other-output"],
        output=f"nested/../{wanted.stem}.apkg",
    )
    with pytest.raises(StudyDeckCreationError, match="output.*already declared"):
        plan_study_deck(config, name="Lesson 13")

    config = _project(tmp_path / "broad")
    broad = _existing_deck(
        config,
        "everything",
        exclude_ids=["word:assignment:assignment"],
    )
    with pytest.raises(StudyDeckCreationError, match="would also select") as error:
        plan_study_deck(config, name="Lesson 13")
    assert str(broad) in str(error.value)


def test_execute_refuses_stale_deck_set_and_atomic_overwrite_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _project(tmp_path)
    existing = _existing_deck(
        config,
        "existing",
        intake_tag="existing",
        include_tags=["existing"],
    )
    stale = plan_study_deck(config, name="New lesson")
    original = existing.read_bytes()
    changed = original.replace(b"name: existing", b"name: EXISTING")
    assert len(changed) == len(original) and changed != original
    existing.write_bytes(changed)

    with pytest.raises(StudyDeckCreationError, match="deck set changed"):
        create_study_deck(config, stale)
    assert not stale.path.exists()

    added = plan_study_deck(config, name="New lesson")
    _existing_deck(
        config,
        "later",
        intake_tag="later",
        include_tags=["later"],
    )
    with pytest.raises(StudyDeckCreationError, match="deck set changed"):
        create_study_deck(config, added)
    assert not added.path.exists()

    current = plan_study_deck(config, name="New lesson")
    real_write = deck_creation.atomic_write_bytes_bound

    def race(path: Path, payload: bytes, *, expected_absent: bool = False) -> None:
        path.write_text("intruder\n", encoding="utf-8")
        real_write(path, payload, expected_absent=expected_absent)

    monkeypatch.setattr(deck_creation, "atomic_write_bytes_bound", race)
    with pytest.raises(StudyDeckCreationError, match="Could not create"):
        create_study_deck(config, current)
    assert current.path.read_text(encoding="utf-8") == "intruder\n"


def test_a_character_deck_names_its_characters_and_reads_the_curated_store(
    tmp_path: Path,
) -> None:
    """A kanji deck is not a word deck with a different source. It selects by
    exact character identity rather than by intake tag, pins its notetype id,
    and ships recognition only unless another direction is asked for."""
    config = _project(tmp_path)

    plan = plan_study_deck(
        config,
        name="Genki II Kanji",
        kind="kanji",
        include_ids=["kanji:理", "kanji:料"],
    )
    section = yaml.safe_load(plan.yaml_bytes.decode("utf-8"))["deck"]

    assert plan.kind == "kanji"
    assert section["kind"] == "kanji"
    assert section["include_ids"] == ["kanji:理", "kanji:料"]
    assert section["cards"] == {
        "recognition": True,
        "production": False,
        "reading": False,
    }
    assert section["source"] == "../data/kanji_notes.json"
    assert "intake_tag" not in section and "include_tags" not in section
    assert plan.intake_tag == ""
    assert section["model_id"] == plan.model_id == config.model_id_base + 9


def test_a_character_deck_needs_the_exact_characters_it_holds(
    tmp_path: Path,
) -> None:
    """No tag sweeps cards in later, so an empty list is a deck that can never
    hold anything — and a repeat would ship one character twice."""
    config = _project(tmp_path)

    with pytest.raises(StudyDeckCreationError, match="exact characters"):
        plan_study_deck(config, name="Empty", kind="kanji")
    with pytest.raises(StudyDeckCreationError, match="more than once"):
        plan_study_deck(
            config, name="Doubled", kind="kanji", include_ids=["kanji:理", "kanji:理"]
        )
    with pytest.raises(StudyDeckCreationError, match="not by an exact id list"):
        plan_study_deck(config, name="Words", include_ids=["word:橋:はし"])


def test_creating_a_character_deck_leaves_word_deck_arithmetic_alone(
    tmp_path: Path,
) -> None:
    """Its identities are characters, so no intake tag is minted and no word
    deck's selection is consulted or changed."""
    config = _project(tmp_path)
    word_deck = _existing_deck(
        config, "lesson", intake_tag="janki:deck:lesson", include_tags=["janki:deck:lesson"]
    )
    before = word_deck.read_bytes()

    created = create_study_deck(
        config,
        plan_study_deck(
            config, name="Kanji set", kind="kanji", include_ids=["kanji:理"]
        ),
    )

    assert word_deck.read_bytes() == before
    assert created.path.read_bytes() == created.yaml_bytes
    assert "intake_tag" not in created.yaml_bytes.decode("utf-8")
