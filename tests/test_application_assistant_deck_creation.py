"""Plan-bound thematic deck creation for the repository Assistant."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import yaml

from japanese_anki.application import assistant_deck_creation, deck_creation
from japanese_anki.config import ProjectConfig


def _project(tmp_path: Path) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "data/normalized/vocabulary.json"\n'
        'deck_dir = "data/decks"\n'
        'dist_dir = "dist"\n',
        encoding="utf-8",
    )
    normalized = tmp_path / "data/normalized/vocabulary.json"
    normalized.parent.mkdir(parents=True)
    normalized.write_text("[]\n", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _request(
    *,
    name: str = "Genki II Lesson 14",
    recognition: bool = True,
    production: bool = False,
    reading: bool = True,
    instruction: str = "Create this thematic study deck.",
) -> assistant_deck_creation.AssistantDeckCreationRequest:
    return assistant_deck_creation.AssistantDeckCreationRequest(
        name=name,
        recognition=recognition,
        production=production,
        reading=reading,
        instruction=instruction,
    )


def _snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _existing_deck(config: ProjectConfig, name: str = "Existing") -> Path:
    config.deck_dir.mkdir(parents=True, exist_ok=True)
    path = config.deck_dir / f"{name.lower()}.yaml"
    path.write_text(
        "deck:\n"
        f"  name: {name}\n"
        "  deck_id: 1500000001\n"
        "  source: ../normalized/vocabulary.json\n"
        f"  intake_tag: existing:{name.lower()}\n"
        f"  include_tags: [existing:{name.lower()}]\n",
        encoding="utf-8",
    )
    return path


def test_plan_is_side_effect_free_and_exposes_exact_browser_safe_definition(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    request = _request(recognition=False, production=True, reading=False)
    before = _snapshot_files(tmp_path)

    plan = assistant_deck_creation.plan_deck_creation(config, request)

    assert _snapshot_files(tmp_path) == before
    assert plan.repository_root == config.root.resolve()
    assert plan.request == request
    assert plan.service_plan.name == request.name
    projection = plan.projection
    assert projection["schema_version"] == 1
    assert projection["kind"] == "create_deck"
    assert projection["instruction"] == request.instruction
    assert projection["target"] == {
        "name": request.name,
        "stable_stem": plan.service_plan.stem,
        "configured_file": f"data/decks/{plan.service_plan.stem}.yaml",
        "future_package": f"dist/{plan.service_plan.stem}.apkg",
        "deck_id": plan.service_plan.deck_id,
        "intake_tag": plan.service_plan.intake_tag,
        "cards": {
            "recognition": False,
            "production": True,
            "reading": False,
        },
    }
    assert projection["inputs"] == {
        "canonical_source": "data/normalized/vocabulary.json",
        "deck_set_sha256": plan.service_plan.deck_set_fingerprint,
    }
    assert projection["provider"] is None
    assert projection["billing_class"] == "local"
    assert projection["writes"] == {
        "deck_definition": f"data/decks/{plan.service_plan.stem}.yaml"
    }
    definition = projection["definition"]
    assert definition["encoding"] == "utf-8"
    assert definition["sha256"] == hashlib.sha256(
        plan.service_plan.yaml_bytes
    ).hexdigest()
    assert definition["yaml"].encode("utf-8") == plan.service_plan.yaml_bytes
    assert yaml.safe_load(definition["yaml"])["deck"]["cards"] == {
        "recognition": False,
        "production": True,
        "reading": False,
    }
    assert json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) == plan.projection_wire
    assert plan.fingerprint == hashlib.sha256(
        plan.projection_wire.encode("utf-8")
    ).hexdigest()
    assert str(config.root.resolve()) not in plan.projection_wire


@dataclass
class LooseRequest:
    name: object = "Lesson 15"
    recognition: object = True
    production: object = False
    reading: object = False
    instruction: object = "Create it."


def test_request_requires_exact_name_instruction_and_every_explicit_direction(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)

    with pytest.raises(
        assistant_deck_creation.AssistantDeckCreationError,
        match="explicit nonblank deck name",
    ):
        assistant_deck_creation.plan_deck_creation(
            config, LooseRequest(name=" ")  # type: ignore[arg-type]
        )
    with pytest.raises(
        assistant_deck_creation.AssistantDeckCreationError,
        match="must not begin or end",
    ):
        assistant_deck_creation.plan_deck_creation(
            config, LooseRequest(name=" Lesson 15")  # type: ignore[arg-type]
        )
    with pytest.raises(
        assistant_deck_creation.AssistantDeckCreationError,
        match="recognition.*explicitly true or false",
    ):
        assistant_deck_creation.plan_deck_creation(
            config, LooseRequest(recognition=1)  # type: ignore[arg-type]
        )
    with pytest.raises(
        assistant_deck_creation.AssistantDeckCreationError,
        match="enable at least one",
    ):
        assistant_deck_creation.plan_deck_creation(
            config,
            LooseRequest(recognition=False, production=False, reading=False),  # type: ignore[arg-type]
        )
    with pytest.raises(
        assistant_deck_creation.AssistantDeckCreationError,
        match="owner's nonblank instruction",
    ):
        assistant_deck_creation.plan_deck_creation(
            config, LooseRequest(instruction="\t")  # type: ignore[arg-type]
        )

    class MissingReading:
        name = "Lesson 15"
        recognition = True
        production = False
        instruction = "Create it."

    with pytest.raises(
        assistant_deck_creation.AssistantDeckCreationError,
        match="reading.*explicitly true or false",
    ):
        assistant_deck_creation.plan_deck_creation(
            config, MissingReading()  # type: ignore[arg-type]
        )


def test_plan_fingerprint_binds_name_directions_and_owner_instruction(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    first = assistant_deck_creation.plan_deck_creation(config, _request())
    changed_name = assistant_deck_creation.plan_deck_creation(
        config, _request(name="Genki II Lesson 15")
    )
    changed_directions = assistant_deck_creation.plan_deck_creation(
        config, _request(recognition=False, production=True)
    )
    changed_instruction = assistant_deck_creation.plan_deck_creation(
        config, _request(instruction="Create this other requested deck.")
    )

    assert changed_name.fingerprint != first.fingerprint
    assert changed_directions.fingerprint != first.fingerprint
    assert changed_instruction.fingerprint != first.fingerprint


def test_execute_freshly_replans_and_delegates_to_existing_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    plan = assistant_deck_creation.plan_deck_creation(config, _request())
    calls: list[deck_creation.StudyDeckCreationPlan] = []
    sentinel = deck_creation.CreatedStudyDeck(
        path=plan.service_plan.path,
        yaml_bytes=plan.service_plan.yaml_bytes,
    )

    def create(
        current: ProjectConfig,
        expected: deck_creation.StudyDeckCreationPlan,
    ) -> deck_creation.CreatedStudyDeck:
        assert current is config
        calls.append(expected)
        return sentinel

    monkeypatch.setattr(deck_creation, "create_study_deck", create)

    executed = assistant_deck_creation.execute_deck_creation(config, plan)

    assert executed.plan == plan
    assert executed.result is sentinel
    assert calls == [plan.service_plan]


@pytest.mark.parametrize(
    "changed_request",
    [
        _request(name="A different deck"),
        _request(recognition=False, production=True, reading=False),
    ],
)
def test_execute_refuses_changed_bound_name_or_directions_before_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_request: assistant_deck_creation.AssistantDeckCreationRequest,
) -> None:
    config = _project(tmp_path)
    displayed = assistant_deck_creation.plan_deck_creation(config, _request())
    tampered = replace(displayed, request=changed_request)
    called = False

    def create(
        _config: ProjectConfig,
        _expected: deck_creation.StudyDeckCreationPlan,
    ) -> deck_creation.CreatedStudyDeck:
        nonlocal called
        called = True
        raise AssertionError("a changed confirmation must not reach the writer")

    monkeypatch.setattr(deck_creation, "create_study_deck", create)

    with pytest.raises(
        assistant_deck_creation.AssistantDeckCreationError,
        match="changed after it was displayed",
    ):
        assistant_deck_creation.execute_deck_creation(config, tampered)

    assert called is False


def test_execute_refuses_fresh_deck_set_change_before_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _project(tmp_path)
    displayed = assistant_deck_creation.plan_deck_creation(config, _request())
    _existing_deck(config)
    called = False

    def create(
        _config: ProjectConfig,
        _expected: deck_creation.StudyDeckCreationPlan,
    ) -> deck_creation.CreatedStudyDeck:
        nonlocal called
        called = True
        raise AssertionError("a stale confirmation must not reach the writer")

    monkeypatch.setattr(deck_creation, "create_study_deck", create)

    with pytest.raises(
        assistant_deck_creation.AssistantDeckCreationError,
        match="changed after it was displayed",
    ):
        assistant_deck_creation.execute_deck_creation(config, displayed)

    assert called is False
