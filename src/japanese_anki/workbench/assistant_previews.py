"""Offer one rendered card preview inside the same conversation.

A preview is a *view*, not an authority. The isolated ChatKit origin holds no
repository authority and never receives a filesystem path, so this store mints
an opaque server-side token for bytes that are already in hand: the renderer's
exact self-contained document and the exact content-security policy those bytes
were built for, kept as one immutable pair.

That makes this deliberately unlike ``assistant_packages``: there is nothing to
re-resolve, because nothing on disk backs a preview. Opening one consumes no
capability, writes no receipt, and changes no canonical file. Previews live
only in this running process — a browser reload finds the same link, a restart
does not, and no durable pre-confirmation state is created to make it.

The issuing plan fingerprint, deck focus and thread are retained as context for
the owner and the log, never as consent: they are recorded beside the snapshot
and are never compared against a confirmation.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, the renderer stays lazy
    from japanese_anki.card_preview import CardPreview

__all__ = [
    "MAX_PREVIEWS",
    "MAX_PREVIEW_BYTES",
    "PREVIEW_UNAVAILABLE_MESSAGE",
    "AssistantPreviewError",
    "AssistantPreviewOffer",
    "AssistantPreviewSnapshot",
    "LocalAssistantPreviewStore",
]

_TOKEN_BYTES = 32
#: Enough for a working conversation, small enough that a long session cannot
#: grow the workbench without bound. The oldest offer is evicted first.
MAX_PREVIEWS = 16
#: One self-contained document with inlined media. Over this it is refused with
#: an explicit message rather than truncated into a misleading page.
MAX_PREVIEW_BYTES = 24 * 1024 * 1024

PREVIEW_UNAVAILABLE_MESSAGE = (
    "This card preview link is unknown or has expired. Janki keeps previews "
    "only in memory while this workbench is running, and a bounded number of "
    "them. Ask Janki to preview those cards again — rendering a preview reads "
    "the repository locally, writes nothing and approves nothing, though "
    "asking for one is an ordinary Assistant turn like any other message."
)


class AssistantPreviewError(RuntimeError):
    """A safe refusal shown instead of serving preview bytes."""


@dataclass(frozen=True, slots=True)
class AssistantPreviewSnapshot:
    """One immutable rendered document and the policy it must be served under.

    ``plan_fingerprint``, ``focus_scope`` and ``thread_id`` are provenance for
    the owner: which prepared plan, deck focus and conversation this preview
    was rendered for. They are never authority, and nothing consults them to
    decide what a confirmation may do.
    """

    html: bytes
    sha256: str
    content_security_policy: str
    label: str
    card_count: int
    plan_fingerprint: str | None = None
    focus_scope: str | None = None
    thread_id: str | None = None


@dataclass(frozen=True, slots=True)
class AssistantPreviewOffer:
    """One opaque preview binding rendered into the thread as a link."""

    token: str
    url: str
    label: str
    card_count: int
    byte_count: int
    sha256: str


@dataclass(slots=True)
class LocalAssistantPreviewStore:
    """Hold a bounded number of exact rendered previews behind opaque tokens."""

    preview_prefix: str
    _snapshots: dict[str, AssistantPreviewSnapshot] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def offer(
        self,
        preview: CardPreview | Any,
        *,
        label: str,
        plan_fingerprint: str | None = None,
        focus_scope: str | None = None,
        thread_id: str | None = None,
    ) -> AssistantPreviewOffer:
        """Bind one opaque token to these exact bytes and their exact policy."""

        html = bytes(preview.html)
        policy = str(preview.content_security_policy)
        sha256 = str(preview.sha256)
        card_count = int(preview.card_count)
        if not html:
            raise AssistantPreviewError("The rendered card preview is empty.")
        if not policy.strip():
            raise AssistantPreviewError(
                "A card preview must carry the exact content-security policy its "
                "own bytes were rendered for."
            )
        if len(html) > MAX_PREVIEW_BYTES:
            limit_mib = MAX_PREVIEW_BYTES // (1024 * 1024)
            raise AssistantPreviewError(
                f"This card preview is too large to offer ({len(html)} bytes; "
                f"Janki holds up to {limit_mib} MiB). Preview fewer cards."
            )
        if hashlib.sha256(html).hexdigest() != sha256:
            raise AssistantPreviewError(
                "The rendered preview does not match its own SHA-256; Janki will "
                "not serve bytes the renderer does not vouch for."
            )
        snapshot = AssistantPreviewSnapshot(
            html=html,
            sha256=sha256,
            content_security_policy=policy,
            label=label,
            card_count=card_count,
            plan_fingerprint=plan_fingerprint,
            focus_scope=focus_scope,
            thread_id=thread_id,
        )
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        with self._lock:
            while len(self._snapshots) >= MAX_PREVIEWS:
                self._snapshots.pop(next(iter(self._snapshots)))
            self._snapshots[token] = snapshot
        return AssistantPreviewOffer(
            token=token,
            url=f"{self.preview_prefix}{token}",
            label=label,
            card_count=card_count,
            byte_count=len(html),
            sha256=sha256,
        )

    def read(self, token: str) -> AssistantPreviewSnapshot:
        """Return the exact retained snapshot, or refuse.

        Reading never consumes the token: a preview is not a capability, and a
        browser reload must find the same document.
        """

        with self._lock:
            snapshot = self._snapshots.get(token)
        if snapshot is None:
            raise AssistantPreviewError(PREVIEW_UNAVAILABLE_MESSAGE)
        return snapshot
