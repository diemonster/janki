"""Keep environment-only provider credentials out of durable/user-visible text."""

from __future__ import annotations

import os
from collections.abc import Mapping

__all__ = ["CREDENTIAL_ENV_NAMES", "redact_environment_credentials"]


CREDENTIAL_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "JPDB_API_KEY",
    "OPENAI_API_KEY",
)


def redact_environment_credentials(
    value: object,
    env: Mapping[str, str] | None = None,
) -> str:
    """Replace exact configured key values before text is stored or rendered."""
    source = os.environ if env is None else env
    text = str(value)
    secrets = {
        secret.strip(): name
        for name in CREDENTIAL_ENV_NAMES
        if (secret := str(source.get(name, ""))).strip()
    }
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, f"[redacted {secrets[secret]}]")
    return text
