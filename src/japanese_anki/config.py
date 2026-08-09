from __future__ import annotations

import difflib
import math
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.tts import openai_tts


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
        "kanji_file",
        "scan_inbox",
    ),
    "anki": (
        "default_deck_name",
        "default_deck_id",
        "model_id_base",
        "collection",
        "profile",
    ),
    "cards": ("recognition", "production", "reading", "max_meanings"),
    "ai": ("extract_model", "enrich_model"),
    "tts": (
        "provider",
        "voicevox_url",
        "voicevox_speaker",
        "voicevox_sentence_speaker",
        "voicevox_speed",
        "sentence_provider",
        "openai_voice",
        "openai_model",
        "openai_instructions",
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

    Non-positive values are refused, and so are NaN and infinity; those bounds
    are arithmetic rather than taste: a rate of zero or below has no meaning,
    and VOICEVOX answers both with a 500. Everything above it is the engine's call, the same way
    ``voicevox_speaker`` is type-checked but not range-checked — janki cannot
    know which speakers an engine has, and it does not know which rates an
    engine will honour either. Verified against VOICEVOX ENGINE on 2026-08-08:
    ``speedScale`` is an unconstrained number in its schema, and 0.3 and 4.0
    both come back at exactly the durations they ask for. 0.5–2.0 is the range
    of the *editor's slider*, not a clamp, and enforcing it here rejected audio
    the engine would have produced correctly — while failing in
    ``ProjectConfig.load``, so an out-of-range rate took down ``janki status``
    and ``janki build``, neither of which synthesizes anything.
    """
    value = _get(data, section, key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"[{section}] {key} must be a number, got {value!r}")
    # `isfinite` first, because every comparison against NaN is False: a bare
    # `<= 0` waves `nan` and `inf` straight through, and both are ordinary TOML
    # float literals that `tomllib` hands back as floats. The rejected 0.5-2.0
    # bound caught them by accident, being two comparisons rather than one.
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ConfigError(
            f"[{section}] {key} must be a finite number greater than 0 (it "
            f"multiplies the rate of speech), got {value!r}"
        )
    return float(value)


def _int_or_none(data: dict[str, Any], section: str, key: str) -> int | None:
    """An optional integer setting, where *absent* is the only way to opt out.

    A sentinel would have to be a number, and every number here is a real
    VOICEVOX style id — 0 is 四国めたん・あまあま, which the audition page prints
    as a copyable value. Treating 0 as "unset" would silently discard a valid
    configuration, which is the failure ``_int`` exists to prevent.
    """
    value = _get(data, section, key, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"[{section}] {key} must be an integer, got {value!r}")
    return value


def _anki_collection(data: dict[str, Any], root: Path) -> str:
    """``[anki] collection`` as an absolute path, or ``""``.

    Resolved against the project root like every ``[paths]`` value. Left raw it
    resolved against the *working directory*, so a relative path either vanished
    or — worse — silently matched an unrelated `collection.anki2` that happened
    to sit wherever the command was run. ``~`` is expanded first, so the
    documented absolute form is unchanged.
    """
    value = _str(data, "anki", "collection", "").strip()
    if not value:
        return ""
    try:
        expanded = Path(value).expanduser()
    except RuntimeError as exc:
        # `~ghost/…`, or `~/…` where HOME is unset and the uid has no passwd
        # entry. `RuntimeError` is not a `JankiError`, and every command loads
        # the config — so an unexpandable path in a setting only `janki status`
        # reads would have taken down `build`, `import` and `audio` too.
        raise ConfigError(
            f"[anki] collection: could not expand {value!r}: {exc}"
        ) from exc
    return str(expanded if expanded.is_absolute() else (root / expanded).resolve())


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
    #: Looked-up reference data about characters, shared across records.
    kanji_file: Path
    scan_inbox: Path
    default_deck_name: str
    default_deck_id: int
    model_id_base: int
    #: Path to a `collection.anki2`, or empty to look under Anki's own
    #: directory. Only ever read, and only from a copy.
    anki_collection: str
    #: Which Anki profile to inspect when several exist. Empty means: the one,
    #: if there is exactly one; otherwise say so rather than guess.
    anki_profile: str
    default_cards: dict[str, bool]
    #: How many glosses a card shows. jpdb hands back every sense a word has
    #: — する has 17 — and a card is for recognising a word, not for holding
    #: its dictionary entry. 0 means show them all.
    max_meanings: int
    extract_model: str
    enrich_model: str
    tts_provider: str
    voicevox_url: str
    voicevox_speaker: int
    #: Which VOICEVOX voice reads example sentences. ``None`` — the key absent
    #: — means the same one that speaks the words. Not 0: that is a real style
    #: id, and a sentinel that collides with a valid value is how a correct
    #: configuration gets silently dropped.
    voicevox_sentence_speaker: int | None
    voicevox_speed: float
    #: Which engine reads example sentences: '' (the word engine), 'voicevox',
    #: or 'openai'. Words are never affected — only VOICEVOX can force an
    #: accent, which is what a word clip is for.
    sentence_provider: str
    openai_voice: str
    openai_model: str
    openai_instructions: str
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
            kanji_file=project_path(_str(data, "paths", "kanji_file", "data/kanji.json")),
            scan_inbox=project_path(_str(data, "paths", "scan_inbox", "data/inbox/scans")),
            default_deck_name=_str(data, "anki", "default_deck_name", "Japanese Anki"),
            default_deck_id=_int(data, "anki", "default_deck_id", 2059400110),
            model_id_base=_int(data, "anki", "model_id_base", 1607392310),
            anki_collection=_anki_collection(data, project_root),
            anki_profile=_str(data, "anki", "profile", ""),
            default_cards={
                "recognition": _bool(data, "cards", "recognition", True),
                "production": _bool(data, "cards", "production", True),
                "reading": _bool(data, "cards", "reading", False),
            },
            max_meanings=_int(data, "cards", "max_meanings", 4),
            extract_model=_str(data, "ai", "extract_model", "claude-opus-5"),
            enrich_model=_str(data, "ai", "enrich_model", "claude-opus-5"),
            tts_provider=_str(data, "tts", "provider", "voicevox"),
            voicevox_url=_str(data, "tts", "voicevox_url", "http://localhost:50021"),
            voicevox_speaker=_int(data, "tts", "voicevox_speaker", 46),
            voicevox_sentence_speaker=_int_or_none(data, "tts", "voicevox_sentence_speaker"),
            voicevox_speed=_float(data, "tts", "voicevox_speed", 1.0),
            sentence_provider=_str(data, "tts", "sentence_provider", ""),
            openai_voice=_str(data, "tts", "openai_voice", openai_tts.DEFAULT_VOICE),
            openai_model=_str(data, "tts", "openai_model", openai_tts.DEFAULT_MODEL),
            openai_instructions=_str(
                data, "tts", "openai_instructions", openai_tts.DEFAULT_INSTRUCTIONS
            ),
            azure_voice=_str(data, "tts", "azure_voice", "ja-JP-NanamiNeural"),
            azure_region=_str(data, "tts", "azure_region", "westus2"),
        )
