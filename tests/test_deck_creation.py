"""W4.1's explicit, preview-first study-deck creator."""

from __future__ import annotations

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
) -> Path:
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    section: dict[str, object] = {
        "name": stem,
        "deck_id": deck_id,
        "source": "../vocabulary.json",
    }
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
