from __future__ import annotations

import difflib
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError


class ConfigError(JankiError):
    pass


# Every key janki.toml understands, by section. Used to warn about typos —
# an unknown key is never an error, it is only silently useless without this.
KNOWN_KEYS: dict[str, tuple[str, ...]] = {
    "project": ("name",),
    "paths": (
        "raw_dir",
        "normalized_file",
        "deck_dir",
        "template_dir",
        "dist_dir",
        "ledger_file",
        "staging_dir",
        "media_dir",
        "scan_inbox",
    ),
    "anki": ("default_deck_name", "default_deck_id", "model_id_base"),
    "cards": ("recognition", "production", "reading"),
    "ai": ("extract_model", "enrich_model"),
    "tts": (
        "provider",
        "voicevox_url",
        "voicevox_speaker",
        "voicevox_speed",
        "azure_voice",
        "azure_region",
    ),
}


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


def _int(data: dict[str, Any], section: str, key: str, default: int) -> int:
    """Read an integer setting, refusing values that only look like one.

    A mistyped speaker id or deck id must fail loudly: silently coercing
    ``true`` to 1 or truncating 46.9 would point every card at the wrong voice
    or the wrong deck.
    """
    value = _get(data, section, key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"[{section}] {key} must be an integer, got {value!r}")
    return value


def _float(data: dict[str, Any], section: str, key: str, default: float) -> float:
    """Read a rate setting, refusing values that only look like one.

    ``bool`` is rejected for the same reason ``_int`` rejects it — ``True``
    would read as 1.0 and sound like nothing was set. An ``int`` *is* accepted:
    ``voicevox_speed = 1`` is a reasonable thing to write and means exactly
    1.0, unlike the cases where a float would silently truncate.

    The range is the engine's: VOICEVOX clamps ``speedScale`` to 0.5–2.0, and a
    value outside it would be quietly clamped into audio that is not what was
    asked for. Refused here so the number in the file is the number spoken.
    """
    value = _get(data, section, key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"[{section}] {key} must be a number, got {value!r}")
    if not 0.5 <= float(value) <= 2.0:
        raise ConfigError(
            f"[{section}] {key} must be between 0.5 and 2.0 (the engine's own "
            f"range for speech rate), got {value!r}"
        )
    return float(value)


def _bool(data: dict[str, Any], section: str, key: str, default: bool) -> bool:
    """Read a boolean setting, refusing values that only look like one.

    ``bool("false")`` is True: silently coercing a quoted flag would emit the
    exact cards the user turned off. The dual of ``_int``.
    """
    value = _get(data, section, key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"[{section}] {key} must be true or false, got {value!r}")
    return value


def _str(data: dict[str, Any], section: str, key: str, default: str) -> str:
    """Read a string setting, refusing values that only look like one.

    ``str(true)`` is ``"True"`` and ``str(123)`` is ``"123"`` — a plausible
    deck name or URL nobody typed. Same rule as ``_int`` and ``_bool``.
    """
    value = _get(data, section, key, default)
    if not isinstance(value, str):
        raise ConfigError(f"[{section}] {key} must be a string, got {value!r}")
    return value


def _closest(candidate: str, options: tuple[str, ...] | list[str]) -> str | None:
    matches = difflib.get_close_matches(candidate, list(options), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _warn(message: str) -> None:
    print(f"warning: janki.toml: {message}", file=sys.stderr)


def _warn_unknown_section(section: str) -> None:
    suggestion = _closest(section, list(KNOWN_KEYS))
    if suggestion:
        _warn(f"unknown section [{section}]; did you mean [{suggestion}]?")
        return
    valid = ", ".join(f"[{name}]" for name in sorted(KNOWN_KEYS))
    _warn(f"unknown section [{section}]; valid sections: {valid}")


_SECRET_WORDS = frozenset({"key", "keys", "token", "tokens", "secret", "password"})
# Credential names people spell without underscores; matched against the key
# with its underscores removed, so 'api_key', 'apikey' and 'API_KEY' all land.
_SECRET_FUSED = frozenset({"apikey", "authtoken", "passwd", "apitoken", "secretkey"})


def _warn_unknown_key(section: str, key: str) -> None:
    # A real typo hint beats the secrets lecture, so try both matches first.
    suggestion = _closest(key, KNOWN_KEYS[section])
    if suggestion:
        _warn(f"unknown key '{key}' in [{section}]; did you mean '{suggestion}'?")
        return
    for other, keys in KNOWN_KEYS.items():
        if other != section and key in keys:
            _warn(f"unknown key '{key}' in [{section}]; did you mean '{key}' in [{other}]?")
            return
    # Whole words only, so 'monkey_dir' is not a credential. The fused list
    # catches one-word spellings ('apikey') the word split misses. A compound
    # whose *own words* include one of these does get the lecture — 'deck_key'
    # splits to {'deck', 'key'} and is warned about — which is the intended
    # trade: a false lecture costs a line of stderr, a missed one costs a
    # secret committed to git.
    if (
        _SECRET_WORDS & set(key.lower().split("_"))
        or key.lower().replace("_", "") in _SECRET_FUSED
    ):
        _warn(
            f"'{key}' in [{section}] is ignored; secrets are never read from janki.toml. "
            "Set ANTHROPIC_API_KEY, JPDB_API_KEY, or AZURE_SPEECH_KEY in the environment."
        )
        return
    valid = ", ".join(f"'{name}'" for name in KNOWN_KEYS[section])
    _warn(f"unknown key '{key}' in [{section}]; valid keys: {valid}")


def check_unknown_keys(data: Mapping[str, Any]) -> None:
    """Warn (never raise) about sections and keys janki.toml does not understand."""
    for section, value in data.items():
        if section not in KNOWN_KEYS:
            if isinstance(value, Mapping):
                _warn_unknown_section(section)
            else:
                _warn(f"unknown top-level key '{section}'; expected a section such as [paths]")
            continue
        if not isinstance(value, Mapping):
            _warn(f"[{section}] should be a table; ignoring it")
            continue
        for key in value:
            if key not in KNOWN_KEYS[section]:
                _warn_unknown_key(section, key)


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    root: Path
    name: str
    raw_dir: Path
    normalized_file: Path
    deck_dir: Path
    template_dir: Path
    dist_dir: Path
    ledger_file: Path
    staging_dir: Path
    media_dir: Path
    scan_inbox: Path
    default_deck_name: str
    default_deck_id: int
    model_id_base: int
    default_cards: dict[str, bool]
    extract_model: str
    enrich_model: str
    tts_provider: str
    voicevox_url: str
    voicevox_speaker: int
    voicevox_speed: float
    azure_voice: str
    azure_region: str

    @classmethod
    def load(cls, root: Path | None = None) -> ProjectConfig:
        project_root = find_project_root(root)
        config_path = project_root / "janki.toml"
        try:
            with config_path.open("rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Could not parse {config_path}: {exc}") from exc
        except OSError as exc:
            raise ConfigError(
                f"Could not read {config_path}: {exc.strerror or exc}"
            ) from exc

        check_unknown_keys(data)

        # A section that is not a table would crash _get; the warning above
        # already reported it, so drop it and fall back to the defaults.
        data = {
            section: value
            for section, value in data.items()
            if not (section in KNOWN_KEYS and not isinstance(value, Mapping))
        }

        def project_path(value: str) -> Path:
            return (project_root / value).resolve()

        return cls(
            root=project_root,
            name=_str(data, "project", "name", "Japanese Anki"),
            # Every [paths] key goes through _str for the same reason the rest
            # do, and more urgently: these decide where records, the ledger and
            # the review queue are written. An unquoted path coerced to
            # `<root>/123`, `<root>/True` or `<root>/['a', 'b']` has janki
            # reading and writing under a Python repr while the real file sits
            # untouched and `janki status` reports 0 records.
            raw_dir=project_path(_str(data, "paths", "raw_dir", "data/inbox/shirabe")),
            normalized_file=project_path(
                _str(data, "paths", "normalized_file", "data/normalized/vocabulary.json")
            ),
            deck_dir=project_path(_str(data, "paths", "deck_dir", "data/decks")),
            template_dir=project_path(
                _str(data, "paths", "template_dir", "templates/japanese-study")
            ),
            dist_dir=project_path(_str(data, "paths", "dist_dir", "dist")),
            ledger_file=project_path(_str(data, "paths", "ledger_file", "data/ledger.json")),
            staging_dir=project_path(_str(data, "paths", "staging_dir", "data/staging")),
            media_dir=project_path(_str(data, "paths", "media_dir", "data/media")),
            scan_inbox=project_path(_str(data, "paths", "scan_inbox", "data/inbox/scans")),
            default_deck_name=_str(data, "anki", "default_deck_name", "Japanese Anki"),
            default_deck_id=_int(data, "anki", "default_deck_id", 2059400110),
            model_id_base=_int(data, "anki", "model_id_base", 1607392310),
            default_cards={
                "recognition": _bool(data, "cards", "recognition", True),
                "production": _bool(data, "cards", "production", True),
                "reading": _bool(data, "cards", "reading", False),
            },
            extract_model=_str(data, "ai", "extract_model", "claude-opus-5"),
            enrich_model=_str(data, "ai", "enrich_model", "claude-opus-5"),
            tts_provider=_str(data, "tts", "provider", "voicevox"),
            voicevox_url=_str(data, "tts", "voicevox_url", "http://localhost:50021"),
            voicevox_speaker=_int(data, "tts", "voicevox_speaker", 46),
            voicevox_speed=_float(data, "tts", "voicevox_speed", 1.0),
            azure_voice=_str(data, "tts", "azure_voice", "ja-JP-NanamiNeural"),
            azure_region=_str(data, "tts", "azure_region", "westus2"),
        )
