"""Offer one finished deck package for download inside the same thread.

The isolated ChatKit origin holds no repository authority, so it cannot be
given a path. A completed finish receipt is the only thing that can name a
downloadable package: this store mints an opaque one-thread token bound to
that receipt id, and every read re-resolves the receipt and re-verifies the
exact ``package_sha256`` it recorded. A replaced, stale, or unfinished package
refuses rather than being served as the confirmed artifact.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path

from japanese_anki.application import kanji_finish
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import DataError, read_bytes_bound

__all__ = [
    "AssistantPackageError",
    "AssistantPackageOffer",
    "LocalAssistantPackageStore",
]

_TOKEN_BYTES = 32
_MAX_OFFERS = 64
#: A download name reaches a response header, so it stays a plain package name.
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.apkg$")


class AssistantPackageError(RuntimeError):
    """A safe refusal shown instead of serving unproven package bytes."""


@dataclass(frozen=True, slots=True)
class AssistantPackageOffer:
    """One opaque download binding for an exact completed receipt."""

    token: str
    url: str
    filename: str
    byte_count: int
    sha256: str


@dataclass(slots=True)
class LocalAssistantPackageStore:
    """Serve only packages a complete finish receipt still proves."""

    config: ProjectConfig
    download_prefix: str
    _offers: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def offer(self, receipt_id: str) -> AssistantPackageOffer:
        """Bind one opaque token to the package that receipt records."""

        path, payload, sha256 = self._resolve(receipt_id)
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        with self._lock:
            while len(self._offers) >= _MAX_OFFERS:
                self._offers.pop(next(iter(self._offers)))
            self._offers[token] = receipt_id
        return AssistantPackageOffer(
            token=token,
            url=f"{self.download_prefix}{token}",
            filename=path.name,
            byte_count=len(payload),
            sha256=sha256,
        )

    def read(self, token: str) -> tuple[str, bytes]:
        """Return the exact bytes the receipt still proves, or refuse."""

        with self._lock:
            receipt_id = self._offers.get(token)
        if receipt_id is None:
            raise AssistantPackageError("This download link is unknown or expired.")
        path, payload, _sha256 = self._resolve(receipt_id)
        return path.name, payload

    def _resolve(self, receipt_id: str) -> tuple[Path, bytes, str]:
        try:
            receipt = kanji_finish.inspect_kanji_finish(self.config, receipt_id)
        except (JankiError, OSError, ValueError) as exc:
            raise AssistantPackageError(
                f"This finish receipt is no longer readable: {exc}"
            ) from exc
        if not receipt.succeeded or receipt.package_sha256 is None:
            raise AssistantPackageError(
                f"Finish receipt {receipt.receipt_id} is in state {receipt.state}; "
                "no completed package is bound to it yet."
            )
        if not _SAFE_FILENAME.fullmatch(receipt.output_path.name):
            raise AssistantPackageError(
                "The finished package name is not a plain .apkg download name."
            )
        try:
            payload = read_bytes_bound(receipt.output_path)
        except (DataError, OSError) as exc:
            raise AssistantPackageError(
                f"The built package could not be read safely: {exc}"
            ) from exc
        sha256 = hashlib.sha256(payload).hexdigest()
        if sha256 != receipt.package_sha256:
            raise AssistantPackageError(
                "The built package changed after it was finished; Janki will not "
                "serve bytes its receipt does not prove."
            )
        return receipt.output_path, payload, sha256
