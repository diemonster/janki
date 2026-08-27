"""Environment-only credentials never survive in text shown or stored."""

from __future__ import annotations

import pytest

from japanese_anki.credential_safety import redact_environment_credentials


@pytest.mark.parametrize(
    "name",
    ["ANTHROPIC_API_KEY", "JPDB_API_KEY", "OPENAI_API_KEY"],
)
def test_each_supported_environment_credential_is_redacted(name: str) -> None:
    secret = f"secret-for-{name.lower()}"

    rendered = redact_environment_credentials(
        f"provider echoed {secret}",
        {name: f"  {secret}  "},
    )

    assert secret not in rendered
    assert rendered == f"provider echoed [redacted {name}]"


def test_overlapping_credentials_are_redacted_longest_first() -> None:
    rendered = redact_environment_credentials(
        "header carried shared-secret-long and shared-secret",
        {
            "ANTHROPIC_API_KEY": "shared-secret",
            "OPENAI_API_KEY": "shared-secret-long",
        },
    )

    assert rendered == (
        "header carried [redacted OPENAI_API_KEY] and "
        "[redacted ANTHROPIC_API_KEY]"
    )
