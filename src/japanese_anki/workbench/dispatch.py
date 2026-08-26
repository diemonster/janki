"""The one-use browser capability behind a paid extraction click.

The workbench's path token protects reading the page and its CSRF token proves
that a form came from this session. Neither says *which paid call* somebody
agreed to. A dispatch token does: it is minted only for a sendable consent
page, bound to the exact source/request/model/mode the page rendered, and
removed atomically on first use.

This is deliberately session memory, not repository state. It records an
unspent browser capability, not whether a call happened; the operation journal
remains the durable source of truth from authorization onward.
"""

from __future__ import annotations

import re
import secrets
import threading
from dataclasses import dataclass
from urllib.parse import parse_qsl

from japanese_anki import extract
from japanese_anki.application import ExtractionConsent, ExtractionRevision
from japanese_anki.errors import JankiError

__all__ = [
    "DispatchFormError",
    "DispatchSubmission",
    "ExtractionAction",
    "ExtractionActions",
    "parse_dispatch_form",
]

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_MAX_PENDING_ACTIONS = 256


class DispatchFormError(JankiError):
    """A paid-action form that is not one the consent page emits."""


@dataclass(frozen=True, slots=True)
class ExtractionAction:
    """The scalar consent snapshot one random capability is bound to."""

    name: str
    model: str
    mode: str | None
    source_sha256: str
    request_fingerprint: str
    replacement_revision: ExtractionRevision | None

    @property
    def replacement_offered(self) -> bool:
        return self.replacement_revision is not None


class ExtractionActions:
    """Thread-safe, bounded, one-use capabilities for one workbench session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, ExtractionAction] = {}

    def issue(self, consent: ExtractionConsent) -> str:
        """Bind a fresh capability to one rendered, sendable consent."""
        target = consent.target
        if not consent.sendable or target is None:
            return ""
        action = ExtractionAction(
            name=consent.name,
            model=consent.model,
            mode=consent.mode,
            source_sha256=target.source_sha256,
            request_fingerprint=str(target.provenance["request_fingerprint"]),
            replacement_revision=consent.replacement_revision,
        )
        with self._lock:
            # A tab can be refreshed forever. Bound the session-only memory;
            # evicting the oldest unused page merely makes that old page ask
            # for a reload, which is the safe direction for paid authority.
            while len(self._pending) >= _MAX_PENDING_ACTIONS:
                self._pending.pop(next(iter(self._pending)))
            token = secrets.token_urlsafe(32)
            while token in self._pending:  # fantastically unlikely, still exact
                token = secrets.token_urlsafe(32)
            self._pending[token] = action
        return token

    def consume(self, token: str) -> ExtractionAction | None:
        """Return and atomically spend `token`, or None for an old/unknown one."""
        if not token:
            return None
        with self._lock:
            return self._pending.pop(token, None)


@dataclass(frozen=True, slots=True)
class DispatchSubmission:
    csrf: str
    token: str
    model: str
    mode: str | None
    request_fingerprint: str
    replacement_offered: bool
    replacement_confirmed: bool


def parse_dispatch_form(body: bytes) -> DispatchSubmission:
    """Parse exactly the paid POST form rendered by `render_consent`."""
    try:
        pairs = parse_qsl(
            body.decode("utf-8", errors="strict"),
            keep_blank_values=True,
            strict_parsing=True,
            encoding="utf-8",
            errors="strict",
            max_num_fields=16,
        )
    except (UnicodeError, ValueError) as exc:
        raise DispatchFormError(f"The submitted form is malformed: {exc}") from exc

    fields: dict[str, list[str]] = {}
    for key, value in pairs:
        fields.setdefault(key, []).append(value)
    required = {
        "action",
        "csrf",
        "dispatch",
        "mode",
        "model",
        "replacement",
        "request_fingerprint",
    }
    if not required <= set(fields) or any(
        len(values) != 1 for values in fields.values()
    ):
        raise DispatchFormError("That paid-action form is not one this page offered.")
    unknown = set(fields) - required - {"replace"}
    if unknown:
        raise DispatchFormError(
            "That paid-action form carries an unexpected field: "
            + ", ".join(sorted(unknown))
        )
    if fields["action"][0] != "extract":
        raise DispatchFormError("The paid form action must be 'extract'.")

    raw_mode = fields["mode"][0]
    if raw_mode and raw_mode not in extract.MODES:
        raise DispatchFormError("That extraction mode is not one this page offered.")
    model = fields["model"][0].strip()
    if not model:
        raise DispatchFormError("That paid-action form names no model.")
    request_fingerprint = fields["request_fingerprint"][0]
    if not _FINGERPRINT.fullmatch(request_fingerprint):
        raise DispatchFormError("That paid-action form has no valid request identity.")
    replacement = fields["replacement"][0]
    if replacement not in {"0", "1"}:
        raise DispatchFormError("That paid-action form has an invalid replacement flag.")
    offered = replacement == "1"
    confirmation = fields.get("replace", [""])[0]
    if not offered and "replace" in fields:
        raise DispatchFormError(
            "That paid-action form confirms a replacement the page did not offer."
        )
    if confirmation not in {"", "confirmed"}:
        raise DispatchFormError("That replacement confirmation is invalid.")

    return DispatchSubmission(
        csrf=fields["csrf"][0],
        token=fields["dispatch"][0],
        model=model,
        mode=raw_mode or None,
        request_fingerprint=request_fingerprint,
        replacement_offered=offered,
        replacement_confirmed=offered and confirmation == "confirmed",
    )
