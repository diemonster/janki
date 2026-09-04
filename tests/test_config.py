from __future__ import annotations

import dataclasses
from pathlib import Path
from textwrap import dedent

import pytest

from japanese_anki.claude_client import EFFORT_LEVELS
from japanese_anki.config import KNOWN_KEYS, ConfigError, ProjectConfig

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
    assert config.assistant_dir == root / "data/assistant"
    assert config.scan_inbox == root / "data/inbox/scans"
    assert config.extract_model == "claude-opus-5"
    # One model at one depth across the complete bare-word pass. Codex stays
    # selectable, while Anthropic remains the production default.
    assert config.enrich_provider == "anthropic"
    assert config.enrich_model == "claude-opus-5"
    assert config.enrich_reasoning_effort == "ultra"  # codex-only, inert here
    assert config.revise_provider == "anthropic-api"
    assert config.revise_model == "claude-opus-5"
    assert config.tts_provider == "voicevox"
    assert config.voicevox_url == "http://localhost:50021"
    assert config.voicevox_speaker == 46
    assert config.assistant_enabled is False
    assert config.assistant_provider == "claude-code"
    assert config.assistant_model == "claude-opus-5"
    assert config.assistant_effort == "medium"
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
        assistant_dir = "state/assistant"
        scan_inbox = "data/inbox/pages"

        [ai]
        extract_model = "claude-haiku-4-5"
        enrich_provider = "anthropic"
        enrich_model = "claude-sonnet-5"
        enrich_reasoning_effort = "high"
        revise_provider = "claude-code"
        revise_model = "claude-opus-5-20260801"

        [tts]
        provider = "voicevox"
        voicevox_url = "http://voice.local:1234"
        voicevox_speaker = 8

        [assistant]
        enabled = true
        provider = "anthropic-api"
        model = "claude-opus-5"
        effort = "high"
        """,
    )

    config = ProjectConfig.load(tmp_path)

    assert config.ledger_file == root / "state/ledger.json"
    assert config.staging_dir == root / "state/staging"
    assert config.media_dir == root / "assets"
    assert config.assistant_dir == root / "state/assistant"
    assert config.scan_inbox == root / "data/inbox/pages"
    assert config.extract_model == "claude-haiku-4-5"
    assert config.enrich_provider == "anthropic"
    assert config.enrich_model == "claude-sonnet-5"
    assert config.enrich_reasoning_effort == "high"
    assert config.revise_provider == "claude-code"
    assert config.revise_model == "claude-opus-5-20260801"
    assert config.tts_provider == "voicevox"
    assert config.voicevox_url == "http://voice.local:1234"
    assert config.voicevox_speaker == 8
    # Falsifiable now that _int refuses to coerce: a float or bool would raise
    # above, and a future coercion regression would fail here.
    assert type(config.voicevox_speaker) is int
    assert config.assistant_enabled is True
    assert config.assistant_provider == "anthropic-api"
    assert config.assistant_model == "claude-opus-5"
    assert config.assistant_effort == "high"
    assert capsys.readouterr().err == ""


def test_assistant_enabled_refuses_a_string_that_only_looks_boolean(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path,
        """
        [assistant]
        enabled = "true"
        """,
    )

    with pytest.raises(ConfigError, match=r"\[assistant\] enabled must be true or false"):
        ProjectConfig.load(tmp_path)


def test_an_unknown_assistant_provider_is_rejected(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
        [assistant]
        provider = "automatic"
        """,
    )

    with pytest.raises(ConfigError) as caught:
        ProjectConfig.load(tmp_path)

    message = str(caught.value)
    assert "[assistant] provider" in message
    assert "claude-code" in message
    assert "anthropic-api" in message


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-5", "claude-opus-5-20260801"],
)
def test_an_unpinned_assistant_model_is_rejected(
    tmp_path: Path,
    model: str,
) -> None:
    _write_config(
        tmp_path,
        f"""
        [assistant]
        model = "{model}"
        """,
    )

    with pytest.raises(ConfigError) as caught:
        ProjectConfig.load(tmp_path)

    message = str(caught.value)
    assert "[assistant] model" in message
    assert "claude-opus-5" in message


#: Every depth `claude --effort` prints, spelled out rather than imported. A
#: parametrization over the module's own tuple shrinks with it, so trimming a
#: level would silently stop testing it instead of failing.
CLI_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@pytest.mark.parametrize("level", CLI_EFFORT_LEVELS)
def test_assistant_effort_accepts_each_cli_level(tmp_path: Path, level: str) -> None:
    """Every depth `claude --effort` prints, and only those.

    The Assistant turn is the one pass whose depth is configured rather than
    resolved from the model, so the accepted set has to be the CLI's own — a
    level trimmed from it here is a level the owner can no longer ask for.
    """
    assert EFFORT_LEVELS == CLI_EFFORT_LEVELS
    _write_config(
        tmp_path,
        f"""
        [assistant]
        effort = "{level}"
        """,
    )

    assert ProjectConfig.load(tmp_path).assistant_effort == level
    # Absent, the measured default: one conversational turn is answered and
    # read immediately, at 8.5 s against 10.8 s for 'xhigh'.
    _write_config(tmp_path, "[assistant]\nenabled = true\n")
    assert ProjectConfig.load(tmp_path).assistant_effort == "medium"


def test_an_unknown_assistant_effort_is_rejected(tmp_path: Path) -> None:
    """A level the CLI would reject must fail here, not at dispatch."""
    _write_config(
        tmp_path,
        """
        [assistant]
        effort = "ultra"
        """,
    )

    with pytest.raises(ConfigError) as caught:
        ProjectConfig.load(tmp_path)

    message = str(caught.value)
    assert "[assistant] effort" in message
    for level in CLI_EFFORT_LEVELS:
        assert repr(level) in message


def test_assistant_effort_is_independent_from_enrich_reasoning_effort(
    tmp_path: Path,
) -> None:
    """Two unrelated settings that both spell 'effort'.

    `enrich_reasoning_effort` is Codex's vocabulary for the bare-word pass and
    accepts 'ultra', which the Claude CLI does not. Reading one from the other
    would either reject a valid Codex project or send Claude a level it refuses.
    """
    _write_config(
        tmp_path,
        """
        [ai]
        enrich_provider = "codex"
        enrich_reasoning_effort = "ultra"

        [assistant]
        effort = "high"
        """,
    )

    config = ProjectConfig.load(tmp_path)

    assert config.enrich_reasoning_effort == "ultra"
    assert config.assistant_effort == "high"


def test_assistant_transport_is_independent_from_revision_transport(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path,
        """
        [ai]
        revise_provider = "anthropic-api"
        revise_model = "claude-opus-5-api"

        [assistant]
        provider = "claude-code"
        model = "claude-opus-5"
        """,
    )

    config = ProjectConfig.load(tmp_path)

    assert config.revise_provider == "anthropic-api"
    assert config.revise_model == "claude-opus-5-api"
    assert config.assistant_provider == "claude-code"
    assert config.assistant_model == "claude-opus-5"


@pytest.mark.parametrize("retired", ["azure_voice", "azure_region"])
def test_retired_azure_settings_are_unknown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    retired: str,
) -> None:
    _write_config(tmp_path, f'[tts]\n{retired} = "retired"\n')

    ProjectConfig.load(tmp_path)

    assert retired in capsys.readouterr().err
    assert retired not in KNOWN_KEYS["tts"]


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


def test_the_retired_polish_model_key_is_unknown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(
        tmp_path,
        """
        [ai]
        polish_model = "claude-opus-5"
        """,
    )

    ProjectConfig.load(tmp_path)

    warning = capsys.readouterr().err
    assert "polish_model" in warning
    assert ("ai", "polish_model") not in KNOWN_KEYS


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


def test_an_unknown_enrichment_provider_is_rejected(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
        [ai]
        enrich_provider = "fable"
        """,
    )

    with pytest.raises(ConfigError) as caught:
        ProjectConfig.load(tmp_path)

    assert "enrich_provider" in str(caught.value)
    assert "codex" in str(caught.value)
    assert "anthropic" in str(caught.value)


def test_an_explicit_anthropic_provider_gets_an_anthropic_model_default(
    tmp_path: Path,
) -> None:
    _write_config(
        tmp_path,
        """
        [ai]
        enrich_provider = "anthropic"
        """,
    )

    config = ProjectConfig.load(tmp_path)

    assert config.enrich_provider == "anthropic"
    assert config.enrich_model == "claude-opus-5"


def test_an_unknown_revision_provider_is_rejected(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
        [ai]
        revise_provider = "automatic"
        """,
    )

    with pytest.raises(ConfigError) as caught:
        ProjectConfig.load(tmp_path)

    message = str(caught.value)
    assert "revise_provider" in message
    assert "claude-code" in message
    assert "anthropic-api" in message


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
    assert config.revise_provider == "claude-code"


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


@pytest.mark.parametrize("key", KNOWN_KEYS["paths"])
@pytest.mark.parametrize(
    "value",
    [
        "123",  # str(123) is "123": a directory named after a forgotten quote
        "true",  # str(True) is "True"
        '["a", "b"]',  # someone assuming several deck dirs are supported
        "{ a = 1 }",
    ],
)
def test_a_non_string_path_is_refused_not_coerced(
    tmp_path: Path, key: str, value: str
) -> None:
    """These nine decide where records, the ledger and staging are written.

    Coerced, they resolve to `<root>/123` or `<root>/['a', 'b']`: janki then
    reads and writes under a Python repr while the real file sits untouched and
    `janki status` reports 0 records, with no error and no warning.
    """
    _write_config(tmp_path, f"""
        [paths]
        {key} = {value}
        """)

    with pytest.raises(ConfigError) as excinfo:
        ProjectConfig.load(tmp_path)

    assert key in str(excinfo.value)


def test_real_string_paths_still_load(tmp_path: Path) -> None:
    root = _write_config(tmp_path, """
        [paths]
        normalized_file = "records/vocab.json"
        staging_dir = "queue"
        """)

    config = ProjectConfig.load(tmp_path)

    assert config.normalized_file == root / "records/vocab.json"
    assert config.staging_dir == root / "queue"


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


def test_voicevox_speed_defaults_to_normal(tmp_path: Path) -> None:
    _write_config(tmp_path, '[tts]\nprovider = "voicevox"\n')

    assert ProjectConfig.load(tmp_path).voicevox_speed == 1.0


def test_voicevox_speed_is_read(tmp_path: Path) -> None:
    _write_config(tmp_path, "[tts]\nvoicevox_speed = 0.85\n")

    assert ProjectConfig.load(tmp_path).voicevox_speed == 0.85


def test_a_whole_number_speed_is_accepted(tmp_path: Path) -> None:
    # `voicevox_speed = 1` is a reasonable thing to write and means 1.0 exactly.
    _write_config(tmp_path, "[tts]\nvoicevox_speed = 1\n")

    assert ProjectConfig.load(tmp_path).voicevox_speed == 1.0


@pytest.mark.parametrize(
    "value", ["0", "-1.0", "nan", "inf", "-inf", "true", '"0.85"']
)
def test_a_speed_that_is_not_a_rate_is_refused(tmp_path: Path, value: str) -> None:
    """Zero and below have no meaning as a multiplier, and VOICEVOX answers both
    with a 500. A quoted number and a bool are the `_int` cases: both would read
    as something plausible and sound like nothing was set.

    `nan` and `inf` are ordinary TOML float literals, and every comparison
    against NaN is False — so a bare `<= 0` accepts both, and the rate reaches
    the engine as the bare token `NaN` in a request body, failing mid-run on the
    first record instead of at load."""
    _write_config(tmp_path, f"[tts]\nvoicevox_speed = {value}\n")

    with pytest.raises(ConfigError):
        ProjectConfig.load(tmp_path)


@pytest.mark.parametrize("value", ["0.3", "4.0"])
def test_a_rate_outside_the_sliders_range_is_accepted(tmp_path: Path, value: str) -> None:
    """0.5-2.0 is the range of the VOICEVOX *editor's slider*, not a clamp: the
    engine's schema leaves speedScale unconstrained and returns exactly the
    durations 0.3 and 4.0 ask for (checked against a running engine 2026-08-08).
    Enforcing the slider's range here refused audio the engine makes correctly,
    and did it inside `ProjectConfig.load` — so it also took down `janki status`
    and `janki build`, which synthesize nothing."""
    _write_config(tmp_path, f"[tts]\nvoicevox_speed = {value}\n")

    assert ProjectConfig.load(tmp_path).voicevox_speed == float(value)
