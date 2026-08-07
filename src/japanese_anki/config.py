from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError


class ConfigError(JankiError):
    pass


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "janki.toml").exists():
            return candidate
    raise ConfigError(
        "Could not find janki.toml. Run the command inside the project or pass --root."
    )


def _get(data: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return (data.get(section) or {}).get(key, default)


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    root: Path
    name: str
    raw_dir: Path
    normalized_file: Path
    deck_dir: Path
    template_dir: Path
    dist_dir: Path
    default_deck_name: str
    default_deck_id: int
    model_id_base: int
    default_cards: dict[str, bool]

    @classmethod
    def load(cls, root: Path | None = None) -> ProjectConfig:
        project_root = find_project_root(root)
        with (project_root / "janki.toml").open("rb") as handle:
            data = tomllib.load(handle)

        def project_path(value: str) -> Path:
            return (project_root / value).resolve()

        return cls(
            root=project_root,
            name=str(_get(data, "project", "name", "Japanese Anki")),
            raw_dir=project_path(str(_get(data, "paths", "raw_dir", "data/inbox/shirabe"))),
            normalized_file=project_path(
                str(_get(data, "paths", "normalized_file", "data/normalized/vocabulary.json"))
            ),
            deck_dir=project_path(str(_get(data, "paths", "deck_dir", "data/decks"))),
            template_dir=project_path(
                str(_get(data, "paths", "template_dir", "templates/japanese-study"))
            ),
            dist_dir=project_path(str(_get(data, "paths", "dist_dir", "dist"))),
            default_deck_name=str(
                _get(data, "anki", "default_deck_name", "Japanese Anki")
            ),
            default_deck_id=int(_get(data, "anki", "default_deck_id", 2059400110)),
            model_id_base=int(_get(data, "anki", "model_id_base", 1607392310)),
            default_cards={
                "recognition": bool(_get(data, "cards", "recognition", True)),
                "production": bool(_get(data, "cards", "production", True)),
                "reading": bool(_get(data, "cards", "reading", False)),
            },
        )
