"""Plan-bound Assistant creation of one thematic vocabulary deck.

The ordinary Assistant turn must return the learner-facing deck name and all
three card-direction choices as structured data.  This broker validates that
closed input, asks :mod:`deck_creation` for the exact local plan, and exposes a
canonical path-free projection for the owner's confirmation.

Execution resolves the same request again and compares the complete fresh
service plan and browser projection before delegating to
``create_study_deck``.  The existing application service remains the sole
writer and performs its own locked re-plan immediately before publication.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from japanese_anki.application import deck_creation
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError

__all__ = [
    "AssistantDeckCreationError",
    "AssistantDeckCreationExecution",
    "AssistantDeckCreationPlan",
    "AssistantDeckCreationRequest",
    "AssistantDeckCreationRequestLike",
    "execute_deck_creation",
    "plan_deck_creation",
]


class AssistantDeckCreationError(JankiError):
    """An Assistant deck-creation request cannot be planned or executed."""


class AssistantDeckCreationRequestLike(Protocol):
    """The complete local input accepted from a closed Assistant intent."""

    name: str
    recognition: bool
    production: bool
    reading: bool
    instruction: str


@dataclass(frozen=True, slots=True)
class AssistantDeckCreationRequest:
    """An exact owner instruction and explicit card-direction choices."""

    name: str
    recognition: bool
    production: bool
    reading: bool
    instruction: str

    def __post_init__(self) -> None:
        _validate_request_values(
            name=self.name,
            recognition=self.recognition,
            production=self.production,
            reading=self.reading,
            instruction=self.instruction,
        )


@dataclass(frozen=True, slots=True)
class AssistantDeckCreationPlan:
    """One existing service plan plus the exact confirmation projection."""

    repository_root: Path
    request: AssistantDeckCreationRequest
    projection_wire: str
    fingerprint: str
    service_plan: deck_creation.StudyDeckCreationPlan

    def __post_init__(self) -> None:
        if not self.repository_root.is_absolute():
            raise ValueError("Assistant deck-creation root must be absolute")
        if self.repository_root != self.repository_root.resolve():
            raise ValueError("Assistant deck-creation root must be canonical")
        try:
            parsed = json.loads(self.projection_wire)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Assistant deck-creation projection must be JSON") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError(
                "Assistant deck-creation projection must be a JSON object"
            )
        if _canonical_json(parsed) != self.projection_wire:
            raise ValueError(
                "Assistant deck-creation projection must use canonical JSON"
            )
        if _sha256(self.projection_wire.encode("utf-8")) != self.fingerprint:
            raise ValueError(
                "Assistant deck-creation fingerprint does not bind its projection"
            )

    @property
    def projection(self) -> Mapping[str, Any]:
        """Return the parsed value safe for a browser confirmation card."""

        value = json.loads(self.projection_wire)
        assert isinstance(value, Mapping)
        return value


@dataclass(frozen=True, slots=True)
class AssistantDeckCreationExecution:
    """The fresh confirmed plan and the existing service's result."""

    plan: AssistantDeckCreationPlan
    result: deck_creation.CreatedStudyDeck


_MISSING = object()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AssistantDeckCreationError(
            f"Assistant deck-creation plan cannot be fingerprinted: {exc}"
        ) from exc


def _validate_request_values(
    *,
    name: object,
    recognition: object,
    production: object,
    reading: object,
    instruction: object,
) -> None:
    if not isinstance(name, str) or not name.strip():
        raise AssistantDeckCreationError(
            "Assistant deck creation needs an explicit nonblank deck name."
        )
    if name != name.strip():
        raise AssistantDeckCreationError(
            "Assistant deck name must not begin or end with whitespace."
        )
    directions = {
        "recognition": recognition,
        "production": production,
        "reading": reading,
    }
    for direction, enabled in directions.items():
        if not isinstance(enabled, bool):
            raise AssistantDeckCreationError(
                f"Assistant deck direction {direction!r} must be explicitly true "
                "or false."
            )
    if not any(directions.values()):
        raise AssistantDeckCreationError(
            "Assistant deck creation must enable at least one card direction."
        )
    if not isinstance(instruction, str) or not instruction.strip():
        raise AssistantDeckCreationError(
            "Assistant deck creation needs the owner's nonblank instruction."
        )


def _request(value: AssistantDeckCreationRequestLike) -> AssistantDeckCreationRequest:
    fields = {
        "name": getattr(value, "name", _MISSING),
        "recognition": getattr(value, "recognition", _MISSING),
        "production": getattr(value, "production", _MISSING),
        "reading": getattr(value, "reading", _MISSING),
        "instruction": getattr(value, "instruction", _MISSING),
    }
    _validate_request_values(**fields)
    name = fields["name"]
    recognition = fields["recognition"]
    production = fields["production"]
    reading = fields["reading"]
    instruction = fields["instruction"]
    assert isinstance(name, str)
    assert isinstance(recognition, bool)
    assert isinstance(production, bool)
    assert isinstance(reading, bool)
    assert isinstance(instruction, str)
    return AssistantDeckCreationRequest(
        name=name,
        recognition=recognition,
        production=production,
        reading=reading,
        instruction=instruction,
    )


def _relative_path(config: ProjectConfig, path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(config.root.resolve()).as_posix()
    except ValueError as exc:
        raise AssistantDeckCreationError(
            f"Assistant deck-creation {label} escapes the configured repository."
        ) from exc


def _projection(
    config: ProjectConfig,
    request: AssistantDeckCreationRequest,
    service: deck_creation.StudyDeckCreationPlan,
) -> dict[str, object]:
    try:
        yaml_text = service.yaml_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssistantDeckCreationError(
            "The planned deck definition is not valid UTF-8."
        ) from exc
    return {
        "schema_version": 1,
        "kind": "create_deck",
        "instruction": request.instruction,
        "target": {
            "name": service.name,
            "stable_stem": service.stem,
            "configured_file": _relative_path(
                config, service.path, label="definition"
            ),
            "future_package": _relative_path(
                config, service.output_path, label="package"
            ),
            "deck_id": service.deck_id,
            "intake_tag": service.intake_tag,
            "cards": {
                "recognition": service.recognition,
                "production": service.production,
                "reading": service.reading,
            },
        },
        "inputs": {
            "canonical_source": _relative_path(
                config, service.canonical_source, label="canonical source"
            ),
            "deck_set_sha256": service.deck_set_fingerprint,
        },
        "provider": None,
        "billing_class": "local",
        "writes": {
            "deck_definition": _relative_path(
                config, service.path, label="definition"
            )
        },
        "definition": {
            "encoding": "utf-8",
            "sha256": _sha256(service.yaml_bytes),
            "yaml": yaml_text,
        },
    }


def plan_deck_creation(
    config: ProjectConfig,
    request: AssistantDeckCreationRequestLike,
) -> AssistantDeckCreationPlan:
    """Return one side-effect-free exact plan for an Assistant confirmation."""

    exact = _request(request)
    try:
        service = deck_creation.plan_study_deck(
            config,
            name=exact.name,
            recognition=exact.recognition,
            production=exact.production,
            reading=exact.reading,
        )
        projection = _projection(config, exact, service)
    except AssistantDeckCreationError:
        raise
    except (JankiError, OSError, ValueError) as exc:
        raise AssistantDeckCreationError(
            f"Could not plan Assistant deck creation for {exact.name!r}: {exc}"
        ) from exc
    wire = _canonical_json(projection)
    return AssistantDeckCreationPlan(
        repository_root=config.root.resolve(),
        request=exact,
        projection_wire=wire,
        fingerprint=_sha256(wire.encode("utf-8")),
        service_plan=service,
    )


def _assert_fresh(
    expected: AssistantDeckCreationPlan,
    fresh: AssistantDeckCreationPlan,
) -> None:
    if (
        fresh.request != expected.request
        or fresh.service_plan != expected.service_plan
        or fresh.fingerprint != expected.fingerprint
        or fresh.projection_wire != expected.projection_wire
    ):
        raise AssistantDeckCreationError(
            "The Assistant deck-creation plan changed after it was displayed; "
            "reload and review the fresh plan before confirming it."
        )


def execute_deck_creation(
    config: ProjectConfig,
    expected: AssistantDeckCreationPlan,
) -> AssistantDeckCreationExecution:
    """Freshly re-plan a confirmed request and use the sole existing writer."""

    if expected.repository_root != config.root.resolve():
        raise AssistantDeckCreationError(
            "The Assistant deck-creation plan belongs to another repository."
        )
    fresh = plan_deck_creation(config, expected.request)
    _assert_fresh(expected, fresh)
    try:
        result = deck_creation.create_study_deck(config, fresh.service_plan)
    except (JankiError, OSError, ValueError) as exc:
        raise AssistantDeckCreationError(
            f"Could not create Assistant deck {fresh.request.name!r}: {exc}"
        ) from exc
    return AssistantDeckCreationExecution(plan=fresh, result=result)
