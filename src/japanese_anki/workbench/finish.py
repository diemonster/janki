"""Exact browser forms and one-use dictionary decisions for W5.

The finish page has six current POST shapes.  Keeping them distinct prevents
one nearby button from silently acquiring the meaning of another.  Hidden
fingerprints are still only claims: the handler must resolve the promotion
receipt, re-plan where required, and compare current repository state before a
write.

Dictionary lookup is split from its confirmed write.  ``DictionaryActions``
holds the exact shared application decision that was rendered, in bounded
session memory, and atomically removes its random token on first use.  It is
not paid-call consent and records no durable outcome.
"""

from __future__ import annotations

import re
import secrets
import threading
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qsl

from japanese_anki.application.enrichment import DictionaryEnrichmentDecision
from japanese_anki.application.finish import (
    FinishScope,
    records_revision_fingerprint,
)
from japanese_anki.errors import JankiError

__all__ = [
    "AudioExamplesSubmission",
    "AudioWordsSubmission",
    "BuildSubmission",
    "DictionaryAction",
    "DictionaryActions",
    "DictionaryCommitSubmission",
    "DictionaryPlanSubmission",
    "FinishFormError",
    "KanjiAddSubmission",
    "parse_finish_form",
]

_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
# Each decision retains a complete prospective collection, not scalar identity.
_MAX_PENDING_ACTIONS = 8


class FinishFormError(JankiError):
    """A finish form that is not one the workbench rendered."""


@dataclass(frozen=True, slots=True)
class DictionaryAction:
    """One rendered dictionary result bound to its exact promoted scope."""

    receipt_id: str
    scope_fingerprint: str
    decision: DictionaryEnrichmentDecision


class DictionaryActions:
    """Thread-safe, bounded, one-use dictionary decision capabilities."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, DictionaryAction] = {}

    def issue(
        self,
        scope: FinishScope,
        decision: DictionaryEnrichmentDecision,
    ) -> str:
        """Bind a fresh random capability to the exact rendered decision."""
        if (
            decision.record_ids != scope.record_ids
            or decision.output_path != scope.canonical_path
            or decision.output_revision.path != scope.canonical_path
            or records_revision_fingerprint(decision.output_revision)
            != scope.canonical_revision
            or decision.force_fields
        ):
            raise FinishFormError(
                "A workbench dictionary decision must name the exact finish "
                "scope, canonical file, and no forced replacements."
            )
        action = DictionaryAction(
            receipt_id=scope.receipt_id,
            scope_fingerprint=scope.fingerprint,
            decision=decision,
        )
        with self._lock:
            while len(self._pending) >= _MAX_PENDING_ACTIONS:
                self._pending.pop(next(iter(self._pending)))
            token = secrets.token_urlsafe(32)
            while token in self._pending:
                token = secrets.token_urlsafe(32)
            self._pending[token] = action
        return token

    def consume(self, token: str) -> DictionaryAction | None:
        """Atomically spend one capability, refusing old or unknown tokens."""
        if not token:
            return None
        with self._lock:
            return self._pending.pop(token, None)


@dataclass(frozen=True, slots=True)
class DictionaryPlanSubmission:
    action: Literal["dictionary-plan"]
    csrf: str
    scope_fingerprint: str


@dataclass(frozen=True, slots=True)
class DictionaryCommitSubmission:
    action: Literal["dictionary-commit"]
    csrf: str
    scope_fingerprint: str
    dictionary_action: str
    plan_fingerprint: str


@dataclass(frozen=True, slots=True)
class KanjiAddSubmission:
    action: Literal["kanji-add"]
    csrf: str
    scope_fingerprint: str
    plan_fingerprint: str


@dataclass(frozen=True, slots=True)
class AudioWordsSubmission:
    action: Literal["audio-words"]
    csrf: str
    scope_fingerprint: str
    plan_fingerprint: str


@dataclass(frozen=True, slots=True)
class AudioExamplesSubmission:
    action: Literal["audio-examples"]
    csrf: str
    scope_fingerprint: str
    plan_fingerprint: str


@dataclass(frozen=True, slots=True)
class BuildSubmission:
    action: Literal["build"]
    csrf: str
    scope_fingerprint: str
    plan_fingerprint: str


FinishSubmission = (
    DictionaryPlanSubmission
    | DictionaryCommitSubmission
    | KanjiAddSubmission
    | AudioWordsSubmission
    | AudioExamplesSubmission
    | BuildSubmission
)


def _fields(body: bytes) -> dict[str, str]:
    try:
        pairs = parse_qsl(
            body.decode("utf-8", errors="strict"),
            keep_blank_values=True,
            strict_parsing=True,
            encoding="utf-8",
            errors="strict",
            max_num_fields=5,
        )
    except (UnicodeError, ValueError) as exc:
        raise FinishFormError(f"The submitted finish form is malformed: {exc}") from exc

    fields: dict[str, str] = {}
    for name, value in pairs:
        if name in fields:
            raise FinishFormError("That finish form repeats a field.")
        fields[name] = value
    return fields


def _require_exact(fields: dict[str, str], expected: set[str]) -> None:
    if set(fields) != expected:
        raise FinishFormError("That finish form is not one this page offered.")


def _fingerprint(fields: dict[str, str], name: str) -> str:
    value = fields[name]
    if not _FINGERPRINT.fullmatch(value):
        raise FinishFormError(f"That finish form has no valid {name.replace('_', ' ')}.")
    return value


def parse_finish_form(body: bytes) -> FinishSubmission:
    """Parse exactly one of the six finish-page POST forms."""
    fields = _fields(body)
    action = fields.get("action", "")
    if action == "dictionary-plan":
        _require_exact(fields, {"action", "csrf", "scope_fingerprint"})
        return DictionaryPlanSubmission(
            action="dictionary-plan",
            csrf=fields["csrf"],
            scope_fingerprint=_fingerprint(fields, "scope_fingerprint"),
        )
    if action == "dictionary-commit":
        _require_exact(
            fields,
            {
                "action",
                "csrf",
                "scope_fingerprint",
                "dictionary_action",
                "plan_fingerprint",
            },
        )
        return DictionaryCommitSubmission(
            action="dictionary-commit",
            csrf=fields["csrf"],
            scope_fingerprint=_fingerprint(fields, "scope_fingerprint"),
            dictionary_action=fields["dictionary_action"],
            plan_fingerprint=_fingerprint(fields, "plan_fingerprint"),
        )
    if action == "kanji-add":
        _require_exact(
            fields,
            {"action", "csrf", "scope_fingerprint", "plan_fingerprint"},
        )
        return KanjiAddSubmission(
            action="kanji-add",
            csrf=fields["csrf"],
            scope_fingerprint=_fingerprint(fields, "scope_fingerprint"),
            plan_fingerprint=_fingerprint(fields, "plan_fingerprint"),
        )
    if action == "audio-words":
        _require_exact(
            fields,
            {"action", "csrf", "scope_fingerprint", "plan_fingerprint"},
        )
        return AudioWordsSubmission(
            action="audio-words",
            csrf=fields["csrf"],
            scope_fingerprint=_fingerprint(fields, "scope_fingerprint"),
            plan_fingerprint=_fingerprint(fields, "plan_fingerprint"),
        )
    if action == "audio-examples":
        _require_exact(
            fields,
            {"action", "csrf", "scope_fingerprint", "plan_fingerprint"},
        )
        return AudioExamplesSubmission(
            action="audio-examples",
            csrf=fields["csrf"],
            scope_fingerprint=_fingerprint(fields, "scope_fingerprint"),
            plan_fingerprint=_fingerprint(fields, "plan_fingerprint"),
        )
    if action == "build":
        _require_exact(
            fields,
            {"action", "csrf", "scope_fingerprint", "plan_fingerprint"},
        )
        return BuildSubmission(
            action="build",
            csrf=fields["csrf"],
            scope_fingerprint=_fingerprint(fields, "scope_fingerprint"),
            plan_fingerprint=_fingerprint(fields, "plan_fingerprint"),
        )
    raise FinishFormError("That finish form names no available action.")
