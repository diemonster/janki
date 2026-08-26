"""Exact browser authority for coverage and promotion actions.

Session CSRF proves a form came from this workbench.  A paid coverage click
needs one narrower fact too: which exact provider request the page described.
``CoverageActions`` keeps that scalar consent in bounded, one-use session
memory.  The repository journal remains the durable authority once dispatch
is authorized; this object never records whether a call happened.

The parser accepts only the five action shapes rendered by the add journey:
two coverage decisions, an ordinary promotion, and the separate check/confirm
pair needed when an offline refusal is provisional. Hidden fingerprints are
not trusted as current state -- the handler re-plans and compares them -- but
strict shapes keep a nearby form from silently acquiring the meaning of an
owner decision, paid call, reading check, or promotion.
"""

from __future__ import annotations

import re
import secrets
import threading
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qsl

from japanese_anki.application.coverage import CoverageDecision
from japanese_anki.errors import JankiError

__all__ = [
    "AdditionFormError",
    "CoverageAction",
    "CoverageActions",
    "CheckedPromotionSubmission",
    "ModelCoverageSubmission",
    "OwnerCoverageSubmission",
    "PromotionSubmission",
    "ReadingCheckSubmission",
    "parse_addition_form",
]

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_MAX_PENDING_ACTIONS = 256


class AdditionFormError(JankiError):
    """An add-page form that is not one the workbench rendered."""


@dataclass(frozen=True, slots=True)
class CoverageAction:
    """Only the scalar identity of one rendered paid completeness check."""

    source: str
    staging_fingerprint: str
    source_file: str
    source_fingerprint: str
    model: str
    request_fingerprint: str
    prompt_fingerprint: str


class CoverageActions:
    """Thread-safe, bounded, one-use paid-coverage capabilities."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, CoverageAction] = {}

    def issue(self, source: str, decision: CoverageDecision) -> str:
        """Bind a capability only when an exact paid request is ready."""
        if (
            decision.state != "ready"
            or decision.prepared is None
            or decision.request is None
        ):
            return ""
        action = CoverageAction(
            source=source,
            staging_fingerprint=decision.staging_revision,
            source_file=decision.source_file,
            source_fingerprint=decision.source_sha256,
            model=decision.model,
            request_fingerprint=decision.request_fingerprint,
            prompt_fingerprint=decision.prompt_fingerprint,
        )
        with self._lock:
            while len(self._pending) >= _MAX_PENDING_ACTIONS:
                self._pending.pop(next(iter(self._pending)))
            token = secrets.token_urlsafe(32)
            while token in self._pending:
                token = secrets.token_urlsafe(32)
            self._pending[token] = action
        return token

    def consume(self, token: str) -> CoverageAction | None:
        """Atomically spend a capability, or refuse an old/unknown value."""
        if not token:
            return None
        with self._lock:
            return self._pending.pop(token, None)


@dataclass(frozen=True, slots=True)
class OwnerCoverageSubmission:
    action: Literal["owner-coverage"]
    csrf: str
    staging_fingerprint: str
    reason: str


@dataclass(frozen=True, slots=True)
class ModelCoverageSubmission:
    action: Literal["model-coverage"]
    csrf: str
    coverage_action: str
    staging_fingerprint: str
    source_file: str
    source_fingerprint: str
    model: str
    request_fingerprint: str
    prompt_fingerprint: str


@dataclass(frozen=True, slots=True)
class PromotionSubmission:
    action: Literal["promote"]
    csrf: str
    staging_fingerprint: str
    preview_fingerprint: str


@dataclass(frozen=True, slots=True)
class ReadingCheckSubmission:
    action: Literal["check-readings"]
    csrf: str
    staging_fingerprint: str
    preview_fingerprint: str


@dataclass(frozen=True, slots=True)
class CheckedPromotionSubmission:
    action: Literal["promote-checked"]
    csrf: str
    staging_fingerprint: str
    offline_preview_fingerprint: str
    checked_preview_fingerprint: str


AdditionSubmission = (
    OwnerCoverageSubmission
    | ModelCoverageSubmission
    | PromotionSubmission
    | ReadingCheckSubmission
    | CheckedPromotionSubmission
)


def _fields(body: bytes) -> dict[str, str]:
    try:
        pairs = parse_qsl(
            body.decode("utf-8", errors="strict"),
            keep_blank_values=True,
            strict_parsing=True,
            encoding="utf-8",
            errors="strict",
            max_num_fields=12,
        )
    except (UnicodeError, ValueError) as exc:
        raise AdditionFormError(f"The submitted form is malformed: {exc}") from exc
    fields: dict[str, str] = {}
    for name, value in pairs:
        if name in fields:
            raise AdditionFormError("That add form repeats a field.")
        fields[name] = value
    return fields


def _require_exact(fields: dict[str, str], expected: set[str]) -> None:
    if set(fields) != expected:
        raise AdditionFormError("That add form is not one this page offered.")


def _fingerprint(fields: dict[str, str], name: str) -> str:
    value = fields[name]
    if not _FINGERPRINT.fullmatch(value):
        raise AdditionFormError(f"That add form has no valid {name.replace('_', ' ')}.")
    return value


def parse_addition_form(body: bytes) -> AdditionSubmission:
    """Parse exactly one form from the coverage/promotion journey."""
    fields = _fields(body)
    action = fields.get("action", "")
    if action == "owner-coverage":
        _require_exact(
            fields,
            {"action", "csrf", "staging_fingerprint", "compared", "reason"},
        )
        if fields["compared"] != "confirmed":
            raise AdditionFormError(
                "Confirm that you compared the source with the coverage account."
            )
        reason = fields["reason"].strip()
        if not reason:
            raise AdditionFormError("An owner coverage decision needs a reason.")
        return OwnerCoverageSubmission(
            action="owner-coverage",
            csrf=fields["csrf"],
            staging_fingerprint=_fingerprint(fields, "staging_fingerprint"),
            reason=reason,
        )
    if action == "model-coverage":
        expected = {
            "action",
            "csrf",
            "coverage_action",
            "staging_fingerprint",
            "source_file",
            "source_fingerprint",
            "model",
            "request_fingerprint",
            "prompt_fingerprint",
        }
        _require_exact(fields, expected)
        for name in ("coverage_action", "source_file", "model"):
            if not fields[name].strip():
                raise AdditionFormError(f"That paid coverage form names no {name}.")
        return ModelCoverageSubmission(
            action="model-coverage",
            csrf=fields["csrf"],
            coverage_action=fields["coverage_action"],
            staging_fingerprint=_fingerprint(fields, "staging_fingerprint"),
            source_file=fields["source_file"],
            source_fingerprint=_fingerprint(fields, "source_fingerprint"),
            model=fields["model"],
            request_fingerprint=_fingerprint(fields, "request_fingerprint"),
            prompt_fingerprint=_fingerprint(fields, "prompt_fingerprint"),
        )
    if action == "promote":
        _require_exact(
            fields,
            {"action", "csrf", "staging_fingerprint", "preview_fingerprint"},
        )
        return PromotionSubmission(
            action="promote",
            csrf=fields["csrf"],
            staging_fingerprint=_fingerprint(fields, "staging_fingerprint"),
            preview_fingerprint=_fingerprint(fields, "preview_fingerprint"),
        )
    if action == "check-readings":
        _require_exact(
            fields,
            {"action", "csrf", "staging_fingerprint", "preview_fingerprint"},
        )
        return ReadingCheckSubmission(
            action="check-readings",
            csrf=fields["csrf"],
            staging_fingerprint=_fingerprint(fields, "staging_fingerprint"),
            preview_fingerprint=_fingerprint(fields, "preview_fingerprint"),
        )
    if action == "promote-checked":
        _require_exact(
            fields,
            {
                "action",
                "csrf",
                "staging_fingerprint",
                "offline_preview_fingerprint",
                "checked_preview_fingerprint",
            },
        )
        return CheckedPromotionSubmission(
            action="promote-checked",
            csrf=fields["csrf"],
            staging_fingerprint=_fingerprint(fields, "staging_fingerprint"),
            offline_preview_fingerprint=_fingerprint(
                fields, "offline_preview_fingerprint"
            ),
            checked_preview_fingerprint=_fingerprint(
                fields, "checked_preview_fingerprint"
            ),
        )
    raise AdditionFormError("That add form names no available action.")
