"""Offer one finished deck package for download inside the same thread.

The isolated ChatKit origin holds no repository authority, so it cannot be
given a path. A completed finish receipt is the only thing that can name a
downloadable package, and more than one finish service now owns such a
receipt: character notes and a whole study job. So this store is a **closed
registry** over exactly those two services. Every entry states its kind, the
kind picks the one public reader that owns that receipt, and the minted opaque
one-thread token binds that exact ``(kind, receipt id)`` pair. Every read
re-resolves the receipt through its own service and re-verifies the exact
``package_sha256`` it recorded. A kind outside the table refuses without asking
either service, and a replaced, stale, or unfinished package refuses rather
than being served as the confirmed artifact.

Tokens are process-local convenience handles, never authority: after a restart
a fresh token is minted from the same durable receipt and the old one is gone.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from japanese_anki.application import kanji_finish, study_finish
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


class _FinishReceipt(Protocol):
    """What every finish service's public result already states.

    Read-only, and only the fields a download needs: the service owns its
    receipt, so this asks its result rather than opening the receipt JSON.
    """

    @property
    def receipt_id(self) -> str: ...

    @property
    def state(self) -> str: ...

    @property
    def succeeded(self) -> bool: ...

    @property
    def output_path(self) -> Path: ...

    @property
    def package_sha256(self) -> str | None: ...


#: The closed registry. Each entry calls the owning module's public reader by
#: name at dispatch time, so a kind names a *service* rather than a function
#: object captured at import. There is no default, no fallback and no alias: a
#: kind that is not a key here is refused before anything is read.
_RESOLVERS: Mapping[str, Callable[[ProjectConfig, str], _FinishReceipt]] = {
    "kanji_finish": lambda config, receipt_id: kanji_finish.inspect_kanji_finish(
        config, receipt_id
    ),
    "study_finish": lambda config, receipt_id: study_finish.inspect_study_finish(
        config, receipt_id
    ),
}


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
    #: token -> (kind, receipt id). The kind is half the binding: a token is
    #: resolvable only through the service that minted it.
    _offers: dict[str, tuple[str, str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def offer(self, *, kind: str, receipt_id: str) -> AssistantPackageOffer:
        """Bind one opaque token to the package that receipt records."""

        path, payload, sha256 = self._resolve(kind, receipt_id)
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        with self._lock:
            while len(self._offers) >= _MAX_OFFERS:
                self._offers.pop(next(iter(self._offers)))
            self._offers[token] = (kind, receipt_id)
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
            bound = self._offers.get(token)
        if bound is None:
            raise AssistantPackageError("This download link is unknown or expired.")
        kind, receipt_id = bound
        path, payload, _sha256 = self._resolve(kind, receipt_id)
        return path.name, payload

    def _resolve(self, kind: str, receipt_id: str) -> tuple[Path, bytes, str]:
        inspect = _RESOLVERS.get(kind)
        if inspect is None:
            raise AssistantPackageError(
                f"{kind!r} is not a finish kind Janki serves packages for."
            )
        try:
            receipt = inspect(self.config, receipt_id)
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
