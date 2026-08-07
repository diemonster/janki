from __future__ import annotations

import dataclasses
from pathlib import Path
from textwrap import dedent

import pytest

from japanese_anki.config import ConfigError, ProjectConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_config(root: Path, text: str) -> Path:
    """Write a janki.toml into ``root`` and return the resolved project root."""
    (root / "janki.toml").write_text(dedent(text).lstrip(), encoding="utf-8")
    return root.resolve()


def test_defaults_apply_when_the_new_sections_are_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_config(
        tmp_path,
        """
        [project]
        name = "Test Project"
        """,
    )

    config = ProjectConfig.load(tmp_path)

    assert config.ledger_file == root / "data/ledger.json"
    assert config.staging_dir == root / "data/staging"
    assert config.media_dir == root / "data/media"
    assert config.scan_inbox == root / "data/inbox/scans"
    assert config.extract_model == "claude-opus-5"
    assert config.enrich_model == "claude-opus-5"
    assert config.tts_provider == "voicevox"
    assert config.voicevox_url == "http://localhost:50021"
    assert config.voicevox_speaker == 46
    assert config.azure_voice == "ja-JP-NanamiNeural"
    assert config.azure_region == "westus2"
    assert capsys.readouterr().err == ""


def test_new_sections_override_defaults_and_paths_resolve_against_the_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_config(
        tmp_path,
        """
        [paths]
        ledger_file = "state/ledger.json"
        staging_dir = "state/staging"
        media_dir = "assets"
        scan_inbox = "data/inbox/pages"

        [ai]
        extract_model = "claude-haiku-4-5"
        enrich_model = "claude-sonnet-5"

        [tts]
        provider = "azure"
        voicevox_url = "http://voice.local:1234"
        voicevox_speaker = 8
        azure_voice = "ja-JP-KeitaNeural"
        azure_region = "japaneast"
        """,
    )

    config = ProjectConfig.load(tmp_path)

    assert config.ledger_file == root / "state/ledger.json"
    assert config.staging_dir == root / "state/staging"
    assert config.media_dir == root / "assets"
    assert config.scan_inbox == root / "data/inbox/pages"
    assert config.extract_model == "claude-haiku-4-5"
    assert config.enrich_model == "claude-sonnet-5"
    assert config.tts_provider == "azure"
    assert config.voicevox_url == "http://voice.local:1234"
    assert config.voicevox_speaker == 8
    # Falsifiable now that _int refuses to coerce: a float or bool would raise
    # above, and a future coercion regression would fail here.
    assert type(config.voicevox_speaker) is int
    assert config.azure_voice == "ja-JP-KeitaNeural"
    assert config.azure_region == "japaneast"
    assert capsys.readouterr().err == ""


def test_unknown_key_warns_with_the_nearest_valid_key_and_still_loads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_config(
        tmp_path,
        """
        [paths]
        deck_dir = "decks"
        stageing_dir = "data/staging"
        """,
    )

    config = ProjectConfig.load(tmp_path)

    stderr = capsys.readouterr().err
    assert "stageing_dir" in stderr
    assert "staging_dir" in stderr
    assert "[paths]" in stderr
    # A warning is not an error: the rest of the file still takes effect and
    # the mistyped key falls back to its default.
    assert config.deck_dir == root / "decks"
    assert config.staging_dir == root / "data/staging"


def test_unknown_key_in_the_wrong_section_points_at_the_right_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(
        tmp_path,
        """
        [tts]
        media_dir = "assets"
        """,
    )

    ProjectConfig.load(tmp_path)

    stderr = capsys.readouterr().err
    assert "media_dir" in stderr
    assert "[paths]" in stderr


def test_unknown_key_with_no_near_match_lists_the_valid_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(
        tmp_path,
        """
        [ai]
        temperature = 0.5
        """,
    )

    ProjectConfig.load(tmp_path)

    stderr = capsys.readouterr().err
    assert "temperature" in stderr
    assert "extract_model" in stderr
    assert "enrich_model" in stderr


def test_unknown_section_warns_with_the_nearest_valid_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_config(
        tmp_path,
        """
        [path]
        deck_dir = "decks"
        """,
    )

    config = ProjectConfig.load(tmp_path)

    stderr = capsys.readouterr().err
    assert "[path]" in stderr
    assert "[paths]" in stderr
    assert config.deck_dir == root / "data/decks"


def test_secrets_in_toml_are_ignored_and_point_at_the_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(
        tmp_path,
        """
        [ai]
        anthropic_api_key = "sk-not-a-real-key"
        """,
    )

    config = ProjectConfig.load(tmp_path)

    stderr = capsys.readouterr().err
    assert "anthropic_api_key" in stderr
    assert "ANTHROPIC_API_KEY" in stderr
    # hasattr would be False for any undeclared name under slots=True, so
    # assert the secret reached no field at all.
    loaded = {str(value) for value in dataclasses.asdict(config).values()}
    assert not any("sk-not-a-real-key" in value for value in loaded)


def test_a_section_that_is_not_a_table_warns_instead_of_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(
        tmp_path,
        """
        tts = "voicevox"
        """,
    )

    config = ProjectConfig.load(tmp_path)

    assert "[tts]" in capsys.readouterr().err
    assert config.tts_provider == "voicevox"


def test_the_repository_config_loads_without_warnings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = ProjectConfig.load(PROJECT_ROOT)

    assert capsys.readouterr().err == ""
    assert config.name == "Brandon Japanese"
    assert config.deck_dir == PROJECT_ROOT / "data/decks"
    assert config.ledger_file == PROJECT_ROOT / "data/ledger.json"


@pytest.mark.parametrize(
    ("value", "why"),
    [
        ('"zundamon"', "a string that is not a number"),
        ("true", "a boolean, which int() would silently accept as 1"),
        ("46.9", "a float, which int() would silently truncate"),
    ],
)
def test_a_non_integer_speaker_is_refused_not_coerced(
    tmp_path: Path, value: str, why: str
) -> None:
    """A wrong speaker id would synthesize every card in the wrong voice."""
    _write_config(tmp_path, f"""
        [tts]
        voicevox_speaker = {value}
        """)

    with pytest.raises(ConfigError) as excinfo:
        ProjectConfig.load(tmp_path)

    assert "voicevox_speaker" in str(excinfo.value), why


def test_unparseable_toml_is_a_clean_error(tmp_path: Path) -> None:
    _write_config(tmp_path, "[paths\nraw_dir = 'x'")

    with pytest.raises(ConfigError) as excinfo:
        ProjectConfig.load(tmp_path)

    assert "janki.toml" in str(excinfo.value)


def test_an_unknown_key_containing_key_still_gets_its_typo_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """'monkey_dir' is not a credential; it is a typo for a path key."""
    _write_config(tmp_path, """
        [paths]
        monkey_dir = "data/x"
        """)

    ProjectConfig.load(tmp_path)

    stderr = capsys.readouterr().err
    assert "secrets are never read" not in stderr
    assert "valid keys" in stderr or "did you mean" in stderr


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("recognition", '"false"'),
        ("production", '"no"'),
        ("reading", '"0"'),
        ("reading", "1"),
    ],
)
def test_a_non_boolean_card_flag_is_refused_not_coerced(
    tmp_path: Path, key: str, value: str
) -> None:
    """bool("false") is True: coercing would emit the cards the user disabled."""
    _write_config(tmp_path, f"""
        [cards]
        {key} = {value}
        """)

    with pytest.raises(ConfigError) as excinfo:
        ProjectConfig.load(tmp_path)

    assert key in str(excinfo.value)


def test_real_booleans_still_configure_the_card_flags(tmp_path: Path) -> None:
    _write_config(tmp_path, """
        [cards]
        recognition = false
        reading = true
        """)

    config = ProjectConfig.load(tmp_path)

    assert config.default_cards == {
        "recognition": False,
        "production": True,
        "reading": True,
    }


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("anki", "default_deck_name", "123"),
        ("tts", "voicevox_url", "true"),
        ("ai", "extract_model", "5"),
    ],
)
def test_a_non_string_name_or_url_is_refused_not_coerced(
    tmp_path: Path, section: str, key: str, value: str
) -> None:
    """str(true) is "True" — a deck name or URL nobody typed."""
    _write_config(tmp_path, f"""
        [{section}]
        {key} = {value}
        """)

    with pytest.raises(ConfigError) as excinfo:
        ProjectConfig.load(tmp_path)

    assert key in str(excinfo.value)


def test_a_fused_credential_spelling_still_gets_the_secrets_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """'apikey' has no underscore for the whole-word match to split on."""
    _write_config(tmp_path, """
        [ai]
        apikey = "sk-not-a-real-key"
        """)

    ProjectConfig.load(tmp_path)

    stderr = capsys.readouterr().err
    assert "secrets are never read" in stderr
    assert "ANTHROPIC_API_KEY" in stderr


def test_an_unreadable_config_file_is_a_clean_error(tmp_path: Path) -> None:
    import os

    if os.geteuid() == 0:
        pytest.skip("root ignores file modes")
    config_path = _write_config(tmp_path, "[project]\nname = 'x'\n") / "janki.toml"
    config_path.chmod(0o000)
    try:
        with pytest.raises(ConfigError) as excinfo:
            ProjectConfig.load(tmp_path)
    finally:
        config_path.chmod(0o644)

    assert "janki.toml" in str(excinfo.value)
