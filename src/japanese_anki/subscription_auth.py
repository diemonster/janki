"""What proves a Claude Code process is running on the owner's subscription.

Two callers share these facts: janki's revision transport
(``application/revision_provider.py``) and the tracked development launcher
``scripts/claude-subscription.py``.  The launcher has to work in a fresh clone
that has no venv yet, so this module imports nothing but the standard library
and nothing from the rest of the package — the launcher loads this file
directly by path rather than importing ``japanese_anki``.

The rule it encodes is one fact.  From the outside, a Claude Code process
billed to a Console API key is indistinguishable from one billed to a Pro or
Max subscription: same binary, same output, same exit code.  The only thing
that tells them apart is what ``claude auth status --json`` reports, and that
report only describes the process about to be started when the probe runs
under the same environment, working directory and settings sources as the
launch itself.  An allowlisted environment is what makes those two the same
process; ``ANTHROPIC_API_KEY`` inherited from a shell is what made them
different, silently, and at Console prices.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

__all__ = [
    "AUTH_STATUS_ARGUMENTS",
    "BASE_CONTROLLED_ENVIRONMENT",
    "HOST_ENV_ALLOWLIST",
    "SUBSCRIPTION_TYPES",
    "SubscriptionAuthError",
    "is_subscription_auth",
    "parse_auth_status",
    "sanitized_environment",
    "subscription_auth_reason",
]


HOST_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "TMPDIR",
        "USER",
        "XDG_CONFIG_HOME",
    }
)

# Every Claude Code process janki starts runs under at least these. Callers add
# their own controlled values on top; nothing removes one.
BASE_CONTROLLED_ENVIRONMENT: Mapping[str, str] = MappingProxyType(
    {
        "CLAUDE_CODE_SAFE_MODE": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
)

AUTH_STATUS_ARGUMENTS = ("auth", "status", "--json")

SUBSCRIPTION_TYPES = frozenset({"pro", "max"})


class SubscriptionAuthError(Exception):
    """``claude auth status --json`` did not return one readable status object."""


def sanitized_environment(
    source: Mapping[str, str],
    controlled: Mapping[str, str] = BASE_CONTROLLED_ENVIRONMENT,
) -> dict[str, str]:
    """The exact environment a subscription Claude Code process may inherit.

    Built by allowlist rather than by removing known-bad names: the set of
    variables that can reroute the CLI to a billed provider is the CLI's to
    grow, and a denylist is wrong the moment it does.
    """
    clean = {name: str(source[name]) for name in HOST_ENV_ALLOWLIST if name in source}
    clean.update({str(name): str(value) for name, value in dict(controlled).items()})
    return clean


def subscription_auth_reason(payload: Any) -> str | None:
    """Why ``payload`` is not a Claude Pro or Max login, or ``None`` when it is.

    The reasons never quote a value out of the payload.  ``auth status`` also
    reports the account's email, its organization, and the *name* of whatever
    API key source it found, and none of those belong in a terminal, a review
    report, or a commit.
    """
    if not isinstance(payload, Mapping):
        return "the CLI did not report one authentication object"
    if payload.get("loggedIn") is not True:
        return "the CLI reports no active login"
    if payload.get("authMethod") != "claude.ai":
        return "the login is not a claude.ai subscription login"
    if payload.get("apiProvider") != "firstParty":
        return (
            "the CLI is routed to a provider other than firstParty "
            "(API, Bedrock, Vertex, or Foundry)"
        )
    # Text first: a list or object here is unhashable, and testing it against a
    # frozenset raises where this function promises a refusal. A crash inside
    # the check that decides whose account pays is not a safe answer.
    subscription = payload.get("subscriptionType")
    if not isinstance(subscription, str) or subscription not in SUBSCRIPTION_TYPES:
        return "the login carries no Claude Pro or Max subscription"
    if payload.get("apiKeySource") is not None:
        return "an API key source is configured, so the call would be billed to the Console"
    return None


def is_subscription_auth(payload: Any) -> bool:
    """True only for a claude.ai first-party Pro/Max login with no API key source."""
    return subscription_auth_reason(payload) is None


def parse_auth_status(raw: bytes | str) -> dict[str, Any]:
    """Strictly decode one ``claude auth status --json`` reply.

    Strict because this decides whether money is about to be spent on the
    wrong account.  A repeated key, a JSON constant Python would happily turn
    into a float, trailing bytes, or anything that is not one object is a reply
    janki cannot read — not a reply that said nothing.  Raised messages never
    include the reply text, which carries the account details.
    """

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SubscriptionAuthError(f"the reply repeats the key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise SubscriptionAuthError(f"the reply contains the invalid JSON value {value!r}")

    if isinstance(raw, (bytes, bytearray)):  # noqa: UP038 - runs on a fresh clone's python3
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SubscriptionAuthError("the reply is not UTF-8") from exc
    else:
        text = str(raw)
    try:
        payload = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except SubscriptionAuthError:
        raise
    except json.JSONDecodeError as exc:
        raise SubscriptionAuthError(f"the reply is not valid JSON ({exc.msg})") from exc
    if not isinstance(payload, dict):
        raise SubscriptionAuthError("the reply is not one JSON object")
    return payload
