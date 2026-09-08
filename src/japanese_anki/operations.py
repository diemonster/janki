"""What janki knows about a paid model call it has not finished.

`docs/DESIGN.md`, *Mechanisms the pipeline rests on*: every paid model call is
journaled durably **before dispatch**, and its exact response is persisted as a
pending artifact **before parsing**. This module is that journal.

The failure it exists for is specific. A paid provider receives a request and
janki then parses or decodes what comes back. Interrupt between those points —
a crash, a closed laptop, a killed terminal — and the money is spent while
nothing on disk remembers it. The next run has no way to tell "this was never
sent" from "this was sent and the answer is gone", and the only safe options
are to pay again or to give up. Recording the attempt before making it, and the
answer before trusting it, is what turns extraction, coverage and Realtime
sentence audio into resumable operations.

**Why its own file rather than the ledger.** The shape here is the one
`pending_audio` established — an exact-request key, a staged artifact, an
additive write under a lock — and DESIGN.md points at that shape deliberately.
The *file* is separate because the ledger is keyed by record, and an extraction
has no records yet: it happens before any exist. Keeping it apart also means a
crash mid-extraction can never damage the bookkeeping of what has already
shipped, which is the one thing in this repository that cannot be rebuilt.

**Authority is one-use.** An operation ID is authorized exactly once. Nothing
here re-dispatches: `outcome_unknown` is a resting state a person inspects, and
a fresh charge needs fresh authority. That is a rule about money, so it is
enforced by refusing the transition rather than documented as a convention.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import stat
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from japanese_anki.errors import JankiError
from japanese_anki.io import (
    DataError,
    _bound_capture_evidence,
    _bound_write_evidence,
    _BoundFileReceipt,
    _BoundWriteEvidence,
    _capture_bytes_bound,
    _cleanup_bound_snapshot,
    _finalize_bound_capture,
    _open_bound_directory,
    _open_cleanup_directory,
    _read_bound_write_evidence,
    _retire_bound_write_evidence,
    _retire_detached_entry,
    _retire_exact_entry,
    _validate_bound_directory,
    atomic_write_text_bound,
    exclusive_path_lock,
    prepare_bound_directory,
    read_bytes_bound,
)

__all__ = [
    "BATCH_OCCUPIED_STATES",
    "BLOCKS_SPENDING",
    "IN_FLIGHT",
    "PENDING_DIR",
    "LIVE_STATES",
    "STATES",
    "TERMINAL_STATES",
    "ArtifactReceipt",
    "Operation",
    "OperationAuthorization",
    "OperationBatch",
    "OperationError",
    "OperationJournal",
    "ReplyObservation",
    "ResponseSpoolObservation",
    "ResponseSpoolReceipt",
    "advance_refusal",
    "cancel_before_send",
    "capture_artifact",
    "prepare_artifact_store",
    "reply_observation",
    "response_spool_observation",
    "response_answer_text",
    "serialize_response",
]


class OperationError(JankiError):
    """A journal transition that is not allowed, or a journal that is unreadable."""


#: Any paid service's run error: `(message, *, operation_id, provider_dispatched)`.
_UnsentError = TypeVar("_UnsentError", bound=JankiError)


#: Every state an operation may hold, and the only legal moves between them.
#:
#: Read the values as answers to "what do we know about the money?":
#: `authorized` nothing was sent; `dispatching` we are about to send;
#: `running` it is with the provider; `result_captured` the exact answer is on
#: disk; `committed` that answer has been turned into staging and the entry may
#: be pruned. The rest are places an operation stops.
_TRANSITIONS: dict[str, frozenset[str]] = {
    "authorized": frozenset(
        {"dispatching", "canceled_before_send", "failed_before_send", "expired"}
    ),
    # `result_captured` directly: a caller that cannot observe streaming
    # never truthfully passes through `running`, and inventing the state
    # would put a moment in the journal that nobody witnessed.
    "dispatching": frozenset(
        {"running", "result_captured", "outcome_unknown", "failed_before_send"}
    ),
    "running": frozenset({"result_captured", "outcome_unknown", "failed_before_send"}),
    # Only `committed`. There is deliberately no move to `outcome_unknown`:
    # once the exact answer is on disk nothing about the outcome is unknown,
    # and the move existed only as a hole for a reply that landed while
    # somebody was declaring the call dead. `advance` re-checks this table
    # under the lock, so a state that must never be reachable has to be
    # unreachable *here* — a caller-side guard cannot close it.
    "result_captured": frozenset({"committed"}),
    # Terminal for dispatch. `outcome_unknown` may only strengthen to an exact
    # captured result through `capture_result`, whose operation-bound receipt
    # proves the answer was already durable. It never returns to a live state,
    # so one authorization can never become two charges.
    "committed": frozenset(),
    "outcome_unknown": frozenset({"result_captured"}),
    "failed_before_send": frozenset(),
    "canceled_before_send": frozenset(),
    "expired": frozenset(),
}

STATES: tuple[str, ...] = tuple(_TRANSITIONS)

#: States an operation is still moving through — authorized but not finished.
#: Not a statement about money: three of these four mean a request may already
#: have reached the provider. `money_may_have_been_spent` is that question.
LIVE_STATES: frozenset[str] = frozenset(
    {"authorized", "dispatching", "running", "result_captured"}
)

# Terminal means the provider must never be dispatched again. An unknown
# outcome remains terminal in that sense even though exact evidence already on
# disk may later strengthen it to `result_captured`.
TERMINAL_STATES: frozenset[str] = frozenset(
    {
        "committed",
        "outcome_unknown",
        "failed_before_send",
        "canceled_before_send",
        "expired",
    }
)

#: States that mean a paid call is in flight right now. `authorized` is not
#: one of them: it says the authority was written and nothing was sent, which
#: is where a process that died before dispatching leaves an entry — and
#: counting it would wedge every later run behind a call that never happened.
IN_FLIGHT: frozenset[str] = frozenset({"dispatching", "running"})

#: Where a captured reply is written, relative to the journal. One name so the
#: writer, the reader and the containment check cannot drift.
PENDING_DIR = ".pending"

#: States that stop janki starting another paid call: every live one, plus the
#: outcome nobody could determine. Either money is moving, or some already
#: moved and nobody has decided what it bought.
#:
#: This is the rule, not a warning about it: `authorize` refuses under the
#: journal's own lock. A surface that only *displayed* the condition would be
#: read when a page renders and acted on when a button is clicked, and two
#: callers could pass it and both spend.
#:
#: `authorized` is in the set even though nothing has been sent from it. It
#: has to be: authorizing and marking dispatched are two writes, so excluding
#: it leaves a window where two runs both authorize, both see the other
#: resting in a state that does not block, and both dispatch. That window is
#: exactly what this set exists to close, and the cost — an orphaned authority
#: from a process killed between the two writes — is paid by `end`, which
#: retires it as `canceled_before_send` because nothing was sent.
BLOCKS_SPENDING: frozenset[str] = LIVE_STATES | frozenset({"outcome_unknown"})

#: States in which one batch child still occupies a live dispatch slot.
#:
#: `outcome_unknown` is in the set and never leaves it. Nothing redispatches an
#: unknown outcome, so the slot it holds is spent, not recoverable — releasing
#: it would let a batch quietly run more calls than the person authorized while
#: the one nobody could account for is still unaccounted for. `result_captured`
#: is deliberately *out*: the exact reply is on disk, which is a call that
#: finished, and holding its slot would stall a batch on work already done.
BATCH_OCCUPIED_STATES: frozenset[str] = frozenset(
    {"dispatching", "running", "outcome_unknown"}
)

#: The only kinds a batch may reserve. A batch buys bounded parallelism for one
#: finite set of pages of one source document; every other paid call in janki is
#: a single decision and keeps the ordinary one-at-a-time rule.
_BATCH_KINDS: frozenset[str] = frozenset({"extract"})


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_batch_identifier(batch_id: Any) -> bool:
    """Whether a batch id is a usable stable key rather than a shape hazard."""
    return (
        isinstance(batch_id, str)
        and bool(batch_id)
        and batch_id.strip() == batch_id
        and "\x00" not in batch_id
        and all(character.isprintable() for character in batch_id)
    )


def advance_refusal(
    operation_id: str,
    state: str,
    to: str,
    artifact: ArtifactReceipt | None = None,
) -> str:
    """Why moving `operation_id` from `state` to `to` would be refused, or "".

    One function so a caller about to do something it cannot take back — write
    a staging file, spend money — can ask *first* and get the same answer
    :meth:`OperationJournal.advance` would give it afterwards. `advance` is
    still the authority and re-asks under the lock; this only lets a caller
    fail before it has made a mess rather than after.
    """
    if to not in _TRANSITIONS:
        return f"Unknown operation state {to!r}"
    # `.get`, because this is exported for callers holding a state they have
    # not validated. An unknown one is refused rather than raising KeyError:
    # the answer to "may this move?" is no, and a caller asking before it
    # spends or writes deserves that answer rather than a traceback.
    if to not in _TRANSITIONS.get(state, frozenset()):
        return f"Operation {operation_id!r} cannot move from {state!r} to {to!r}"
    if to == "committed" and not artifact:
        # Committing means "the exact answer became staging". Without a
        # captured artifact there is nothing that could have.
        return (
            f"Operation {operation_id!r} cannot be committed without a "
            "captured provider answer"
        )
    return ""


def serialize_response(response: Any) -> bytes:
    """The provider's answer as bytes, whatever object the SDK handed back.

    Deliberately forgiving. This runs at the one moment when losing the answer
    is most expensive — it has just been paid for and not yet parsed — so a
    serializer that raised on an unfamiliar object would turn "we have your
    answer" into "we had your answer". Anything it cannot model as JSON is
    written as its text, which is still the thing a person can read and a
    later run can replay by hand.
    """
    for attribute in ("model_dump_json", "to_json", "json"):
        method = getattr(response, attribute, None)
        if not callable(method):
            continue
        try:
            value = method()
            # Converted inside the guard, not after it. `.json()` returning
            # parsed data rather than text is the ordinary convention in half
            # the HTTP clients in existence, and `bytes(7)` does not raise —
            # it fabricates seven NUL bytes and stores them as somebody's
            # paid answer.
            if isinstance(value, str):
                return value.encode("utf-8")
            if isinstance(value, bytes | bytearray):
                return bytes(value)
            return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        except Exception:  # noqa: BLE001 - never lose a paid answer
            continue
    try:
        return json.dumps(response, ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return repr(response).encode("utf-8")


#: The versioned wrapper an extraction writes around one paid provider reply,
#: inside this operation's own captured artifact. The reply is kept exactly;
#: the request that produced it rides beside it so an answer that arrives
#: before anything can read it is still recoverable.
CAPTURED_REPLY_KEY = "janki_extraction_capture"
CAPTURED_REPLY_VERSION = 1


def unwrap_captured_reply(
    payload: bytes,
) -> tuple[bytes, Mapping[str, Any] | None]:
    """The exact provider reply inside a captured artifact, and its request.

    The one decoder. Anything that is not one of these wrappers — every
    Anthropic API reply ever captured — comes back byte for byte with no
    saved request, which is what keeps the older artifacts readable by the
    same readers.
    """
    try:
        decoded = json.loads(payload)
    except (UnicodeError, ValueError):
        return payload, None
    if not isinstance(decoded, Mapping) or CAPTURED_REPLY_KEY not in decoded:
        return payload, None
    if decoded.get(CAPTURED_REPLY_KEY) != CAPTURED_REPLY_VERSION:
        raise OperationError(
            "This captured reply was wrapped by a different version of janki "
            "and cannot be unwrapped safely."
        )
    encoded = decoded.get("reply_base64")
    if not isinstance(encoded, str):
        raise OperationError("This captured reply has no exact provider bytes.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise OperationError("This captured reply is unreadable.") from exc
    saved = decoded.get("provenance")
    return raw, saved if isinstance(saved, Mapping) else None


def _streamed_answer_text(payload_bytes: bytes) -> str:
    """The structured answer a streaming CLI reply carries, if it carries one.

    Only the terminal result frame counts. The prose frames before it are the
    model thinking aloud, and calling those an answer would report a reply
    that holds nothing as though it held cards.
    """
    answer = ""
    for line in payload_bytes.splitlines():
        if not line.strip():
            continue
        try:
            frame = json.loads(line)
        except (UnicodeError, ValueError):
            continue
        if not isinstance(frame, Mapping) or frame.get("type") != "result":
            continue
        structured = frame.get("structured_output")
        if structured is not None:
            answer = json.dumps(structured, ensure_ascii=False, sort_keys=True)
        elif isinstance(frame.get("result"), str):
            answer = str(frame["result"])
    return answer


def response_answer_text(payload_bytes: bytes | None) -> str:
    """Concatenate provider text blocks from already-bound response bytes."""
    if payload_bytes is None:
        return ""
    try:
        payload_bytes, _saved = unwrap_captured_reply(payload_bytes)
    except OperationError:
        return ""
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeError, ValueError):
        return _streamed_answer_text(payload_bytes)
    if not isinstance(payload, Mapping):
        return ""
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        return _streamed_answer_text(payload_bytes)
    return "".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "text"
    )


_ArtifactState = tuple[int, int, int, int, int]


def _artifact_state(details: os.stat_result) -> _ArtifactState:
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _digest_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _read_artifact_entry(
    directory_fd: int, name: str
) -> tuple[_ArtifactState, str]:
    """Bind one direct regular name to its inode and exact current bytes."""
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("Artifact is not a regular file")
        digest = _digest_descriptor(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(named.st_mode)
            or _artifact_state(before) != _artifact_state(after)
            or _artifact_state(before) != _artifact_state(named)
        ):
            raise OSError("Artifact changed while it was being bound")
        return _artifact_state(before), digest
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class _TerminalMarkerReceipt:
    """Exact terminal WAL entry bound by a durable artifact receipt."""

    name: str
    entry_state: _ArtifactState
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entry_state": list(self.entry_state),
            "sha256": self.content_sha256,
        }


def _terminal_marker_owned_identity(
    target_name: str,
    name: Any,
) -> tuple[int, int] | None:
    """Return the temp identity encoded by one exact validated marker name."""
    if not isinstance(name, str):
        return None
    try:
        encoded = os.fsencode(name)
    except (TypeError, UnicodeError, ValueError):
        return None
    prefix = f".{target_name}."
    suffix = ".janki-cas.validated"
    if (
        not encoded
        or b"\0" in encoded
        or Path(name).name != name
        or "/" in name
        or "\\" in name
        or not name.startswith(prefix)
        or not name.endswith(suffix)
    ):
        return None
    body = name[len(prefix) : -len(suffix)]
    try:
        token, identity = body.split(".", 1)
        device, inode = identity.split("-", 1)
    except ValueError:
        return None
    if (
        len(token) != 16
        or any(character not in "0123456789abcdef" for character in token)
        or not device
        or not inode
        or any(character not in "0123456789abcdef" for character in device)
        or any(character not in "0123456789abcdef" for character in inode)
    ):
        return None
    return int(device, 16), int(inode, 16)


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    """Durable proof of one exact published provider reply.

    A relative name keeps the committed repository portable; the directory
    identity, five-field entry state, and digest make that name evidence rather
    than a locator that could silently adopt a later occupant.
    """

    relative_name: str
    directory_identity: tuple[int, int]
    entry_state: _ArtifactState
    content_sha256: str
    terminal_marker: _TerminalMarkerReceipt | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_name": self.relative_name,
            "directory_identity": list(self.directory_identity),
            "entry_state": list(self.entry_state),
            "sha256": self.content_sha256,
            "terminal_marker": (
                self.terminal_marker.to_dict()
                if self.terminal_marker is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        operation_id: str,
        raw: Mapping[str, Any],
    ) -> ArtifactReceipt:
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_name",
            "directory_identity",
            "entry_state",
            "sha256",
            "terminal_marker",
        }:
            raise OperationError(
                f"Operation {operation_id!r} holds an invalid artifact receipt"
            )

        def integers(value: Any, length: int) -> tuple[int, ...]:
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(
                    not isinstance(item, int) or isinstance(item, bool) or item < 0
                    for item in value
                )
            ):
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid artifact receipt"
                )
            return tuple(value)

        relative_name = raw["relative_name"]
        expected_name = f"{PENDING_DIR}/{operation_id}.json"
        if (
            not isinstance(relative_name, str)
            or relative_name != expected_name
            or not _valid_pending_operation_id(operation_id)
        ):
            raise OperationError(
                f"Operation {operation_id!r} holds an invalid artifact receipt"
            )
        directory_identity = integers(raw["directory_identity"], 2)
        entry_state = integers(raw["entry_state"], 5)
        digest = raw["sha256"]
        if (
            entry_state[0] != directory_identity[0]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise OperationError(
                f"Operation {operation_id!r} holds an invalid artifact receipt"
            )
        terminal_marker_raw = raw["terminal_marker"]
        terminal_marker = None
        if terminal_marker_raw is not None:
            if not isinstance(terminal_marker_raw, Mapping) or set(
                terminal_marker_raw
            ) != {"name", "entry_state", "sha256"}:
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid artifact receipt"
                )
            marker_name = terminal_marker_raw["name"]
            marker_state = integers(terminal_marker_raw["entry_state"], 5)
            marker_digest = terminal_marker_raw["sha256"]
            marker_identity = _terminal_marker_owned_identity(
                f"{operation_id}.json", marker_name
            )
            if (
                marker_identity != (entry_state[0], entry_state[1])
                or marker_state[0] != directory_identity[0]
                or not isinstance(marker_digest, str)
                or len(marker_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in marker_digest
                )
            ):
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid artifact receipt"
                )
            terminal_marker = _TerminalMarkerReceipt(
                name=marker_name,
                entry_state=(
                    marker_state[0],
                    marker_state[1],
                    marker_state[2],
                    marker_state[3],
                    marker_state[4],
                ),
                content_sha256=marker_digest,
            )
        return cls(
            relative_name=relative_name,
            directory_identity=(directory_identity[0], directory_identity[1]),
            entry_state=(
                entry_state[0],
                entry_state[1],
                entry_state[2],
                entry_state[3],
                entry_state[4],
            ),
            content_sha256=digest,
            terminal_marker=terminal_marker,
        )


@dataclass(frozen=True, slots=True)
class ResponseSpoolReceipt:
    """Durable identity of one append-only provider response-frame spool.

    The initial empty-file state is retained only to prove the write-once
    preparation transaction and its marker.  Appends legitimately change the
    mutable state, so every later operation binds the directory plus the
    stable file identity and then proves the direct name still names it.
    """

    relative_name: str
    directory_identity: tuple[int, int]
    initial_state: _ArtifactState
    initial_sha256: str
    committed_size: int
    committed_sha256: str
    frame_count: int
    terminal_marker: _TerminalMarkerReceipt

    @property
    def file_identity(self) -> tuple[int, int]:
        return self.initial_state[:2]

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_name": self.relative_name,
            "directory_identity": list(self.directory_identity),
            "initial_state": list(self.initial_state),
            "initial_sha256": self.initial_sha256,
            "committed_size": self.committed_size,
            "committed_sha256": self.committed_sha256,
            "frame_count": self.frame_count,
            "terminal_marker": self.terminal_marker.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        operation_id: str,
        raw: Mapping[str, Any],
    ) -> ResponseSpoolReceipt:
        refusal = f"Operation {operation_id!r} holds an invalid response spool receipt"
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_name",
            "directory_identity",
            "initial_state",
            "initial_sha256",
            "committed_size",
            "committed_sha256",
            "frame_count",
            "terminal_marker",
        }:
            raise OperationError(refusal)

        def integers(value: Any, length: int) -> tuple[int, ...]:
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(
                    not isinstance(item, int) or isinstance(item, bool) or item < 0
                    for item in value
                )
            ):
                raise OperationError(refusal)
            return tuple(value)

        relative_name = raw["relative_name"]
        expected_name = f"{PENDING_DIR}/{operation_id}.frames"
        directory_identity = integers(raw["directory_identity"], 2)
        initial_state = integers(raw["initial_state"], 5)
        initial_sha256 = raw["initial_sha256"]
        committed_size = raw["committed_size"]
        committed_sha256 = raw["committed_sha256"]
        frame_count = raw["frame_count"]
        marker_raw = raw["terminal_marker"]
        if (
            not _valid_pending_operation_id(operation_id)
            or relative_name != expected_name
            or initial_state[0] != directory_identity[0]
            or initial_state[2] != 0
            or initial_sha256 != hashlib.sha256(b"").hexdigest()
            or not isinstance(committed_size, int)
            or isinstance(committed_size, bool)
            or committed_size < 0
            or not isinstance(frame_count, int)
            or isinstance(frame_count, bool)
            or frame_count < 0
            or not isinstance(committed_sha256, str)
            or len(committed_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in committed_sha256
            )
            or (committed_size == 0) != (frame_count == 0)
            or (
                committed_size == 0
                and committed_sha256 != hashlib.sha256(b"").hexdigest()
            )
            or not isinstance(marker_raw, Mapping)
            or set(marker_raw) != {"name", "entry_state", "sha256"}
        ):
            raise OperationError(refusal)
        marker_name = marker_raw["name"]
        marker_state = integers(marker_raw["entry_state"], 5)
        marker_sha256 = marker_raw["sha256"]
        if (
            _terminal_marker_owned_identity(
                f"{operation_id}.frames", marker_name
            )
            != initial_state[:2]
            or marker_state[0] != directory_identity[0]
            or not isinstance(marker_sha256, str)
            or len(marker_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in marker_sha256
            )
        ):
            raise OperationError(refusal)
        return cls(
            relative_name=relative_name,
            directory_identity=(directory_identity[0], directory_identity[1]),
            initial_state=(
                initial_state[0],
                initial_state[1],
                initial_state[2],
                initial_state[3],
                initial_state[4],
            ),
            initial_sha256=initial_sha256,
            committed_size=committed_size,
            committed_sha256=committed_sha256,
            frame_count=frame_count,
            terminal_marker=_TerminalMarkerReceipt(
                name=marker_name,
                entry_state=(
                    marker_state[0],
                    marker_state[1],
                    marker_state[2],
                    marker_state[3],
                    marker_state[4],
                ),
                content_sha256=marker_sha256,
            ),
        )


@dataclass(frozen=True, slots=True)
class _ArtifactBinding:
    path: Path
    directory_identity: tuple[int, int]
    entry_state: _ArtifactState
    content_sha256: str


@dataclass(frozen=True, slots=True)
class _CleanupIntent:
    """Exact deletion authority made durable before any evidence is retired."""

    artifact: _ArtifactBinding | None
    write_ahead: _BoundWriteEvidence | None
    response_spool: _ArtifactBinding | None
    response_spool_write_ahead: _BoundWriteEvidence | None
    forced: bool

    def to_dict(self) -> dict[str, Any]:
        artifact = None
        if self.artifact is not None:
            artifact = {
                "directory_identity": list(self.artifact.directory_identity),
                "entry_state": list(self.artifact.entry_state),
                "sha256": self.artifact.content_sha256,
            }
        return {
            "artifact": artifact,
            "write_ahead": (
                self.write_ahead.to_cleanup_dict()
                if self.write_ahead is not None
                else None
            ),
            "response_spool": (
                {
                    "directory_identity": list(
                        self.response_spool.directory_identity
                    ),
                    "entry_state": list(self.response_spool.entry_state),
                    "sha256": self.response_spool.content_sha256,
                }
                if self.response_spool is not None
                else None
            ),
            "response_spool_write_ahead": (
                self.response_spool_write_ahead.to_cleanup_dict()
                if self.response_spool_write_ahead is not None
                else None
            ),
            "forced": self.forced,
        }

    @classmethod
    def from_dict(
        cls,
        journal_path: Path,
        operation_id: str,
        raw: Mapping[str, Any],
    ) -> _CleanupIntent:
        if not isinstance(raw, Mapping) or set(raw) != {
            "artifact",
            "write_ahead",
            "response_spool",
            "response_spool_write_ahead",
            "forced",
        }:
            raise OperationError(
                f"Operation {operation_id!r} holds an invalid cleanup intent"
            )
        if not _valid_pending_operation_id(operation_id):
            raise OperationError(
                f"Operation {operation_id!r} cannot hold a cleanup intent"
            )

        def integers(value: Any, length: int) -> tuple[int, ...]:
            if (
                not isinstance(value, list)
                or len(value) != length
                or any(
                    not isinstance(item, int) or isinstance(item, bool) or item < 0
                    for item in value
                )
            ):
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid cleanup binding"
                )
            return tuple(value)

        def digest(value: Any) -> str:
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid cleanup digest"
                )
            return value

        target = (
            Path(journal_path).parent / PENDING_DIR / f"{operation_id}.json"
        ).absolute()
        artifact_raw = raw["artifact"]
        artifact = None
        if artifact_raw is not None:
            if not isinstance(artifact_raw, Mapping) or set(artifact_raw) != {
                "directory_identity",
                "entry_state",
                "sha256",
            }:
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid artifact cleanup"
                )
            directory_identity = integers(
                artifact_raw["directory_identity"], 2
            )
            entry_state = integers(artifact_raw["entry_state"], 5)
            if entry_state[0] != directory_identity[0]:
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid artifact "
                    "directory binding"
                )
            artifact = _ArtifactBinding(
                path=target,
                directory_identity=(
                    directory_identity[0],
                    directory_identity[1],
                ),
                entry_state=(
                    entry_state[0],
                    entry_state[1],
                    entry_state[2],
                    entry_state[3],
                    entry_state[4],
                ),
                content_sha256=digest(artifact_raw["sha256"]),
            )

        write_ahead_raw = raw["write_ahead"]
        write_ahead = None
        if write_ahead_raw is not None:
            try:
                write_ahead = _BoundWriteEvidence.from_cleanup_dict(
                    target, write_ahead_raw
                )
            except DataError as exc:
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid write-ahead "
                    f"cleanup: {exc}"
                ) from exc
        response_spool_target = (
            Path(journal_path).parent / PENDING_DIR / f"{operation_id}.frames"
        ).absolute()
        response_spool_raw = raw["response_spool"]
        response_spool = None
        if response_spool_raw is not None:
            if not isinstance(response_spool_raw, Mapping) or set(
                response_spool_raw
            ) != {"directory_identity", "entry_state", "sha256"}:
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid response "
                    "spool cleanup"
                )
            spool_directory_identity = integers(
                response_spool_raw["directory_identity"], 2
            )
            spool_entry_state = integers(
                response_spool_raw["entry_state"], 5
            )
            if spool_entry_state[0] != spool_directory_identity[0]:
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid response "
                    "spool directory binding"
                )
            response_spool = _ArtifactBinding(
                path=response_spool_target,
                directory_identity=(
                    spool_directory_identity[0],
                    spool_directory_identity[1],
                ),
                entry_state=(
                    spool_entry_state[0],
                    spool_entry_state[1],
                    spool_entry_state[2],
                    spool_entry_state[3],
                    spool_entry_state[4],
                ),
                content_sha256=digest(response_spool_raw["sha256"]),
            )
        response_spool_write_ahead_raw = raw[
            "response_spool_write_ahead"
        ]
        response_spool_write_ahead = None
        if response_spool_write_ahead_raw is not None:
            try:
                response_spool_write_ahead = (
                    _BoundWriteEvidence.from_cleanup_dict(
                        response_spool_target,
                        response_spool_write_ahead_raw,
                    )
                )
            except DataError as exc:
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid response "
                    f"spool write-ahead cleanup: {exc}"
                ) from exc
        forced = raw["forced"]
        if not isinstance(forced, bool):
            raise OperationError(
                f"Operation {operation_id!r} holds an invalid cleanup decision"
            )
        return cls(
            artifact=artifact,
            write_ahead=write_ahead,
            response_spool=response_spool,
            response_spool_write_ahead=response_spool_write_ahead,
            forced=forced,
        )


@dataclass(frozen=True, slots=True)
class _PendingAnswerEvidence:
    artifact: _ArtifactBinding | None
    write_ahead: _BoundWriteEvidence | None

    @property
    def reply_complete(self) -> bool:
        return self.artifact is not None or (
            self.write_ahead is not None
            and self.write_ahead.reply_complete
        )


@dataclass(frozen=True, slots=True)
class ReplyObservation:
    """What exact recovery bytes are readable for one journal operation now.

    ``artifact`` in the committed journal is a historical pointer, not proof
    that its lexical path still contains the paid reply.  Conversely, a
    complete write-ahead transaction may retain that reply only under its
    private bound name.  Keeping these facts separate prevents a status page
    from advertising a missing or replaced public file while still allowing a
    safe reader to recover the exact private answer.
    """

    #: Exact bytes read through their bound public or write-ahead evidence.
    payload: bytes | None = field(repr=False)
    #: The journal says a reply was captured, even if its bytes are unavailable.
    recorded: bool
    #: A write-ahead transaction exists but does not prove a complete reply.
    interrupted: bool

    @property
    def readable(self) -> bool:
        return self.payload is not None


@dataclass(frozen=True, slots=True)
class ResponseSpoolObservation:
    """Whether exact streaming frames are readable or only recorded now."""

    readable: bool
    recorded: bool
    recovery_pending: bool


def _valid_pending_operation_id(operation_id: str) -> bool:
    try:
        encoded = os.fsencode(operation_id)
    except (TypeError, UnicodeError, ValueError):
        return False
    return bool(
        operation_id
        and encoded
        and b"\0" not in encoded
        and Path(operation_id).name == operation_id
        and "/" not in operation_id
        and "\\" not in operation_id
    )


def _pending_answer_evidence(
    journal_path: Path,
    operation_id: str,
    receipt: ArtifactReceipt | None = None,
) -> _PendingAnswerEvidence:
    """Bind every exact on-disk form of one operation's pending answer."""
    if not _valid_pending_operation_id(operation_id):
        return _PendingAnswerEvidence(None, None)
    relative = f"{PENDING_DIR}/{operation_id}.json"
    target = Path(journal_path).parent / relative
    try:
        if receipt is None:
            write_ahead = _bound_write_evidence(target)
        elif receipt.terminal_marker is None:
            write_ahead = None
        else:
            write_ahead = _bound_capture_evidence(
                _receipt_bound_file(journal_path, receipt)
            )
    except DataError as exc:
        raise OperationError(
            f"Could not inspect recovery evidence for operation "
            f"{operation_id!r}: {exc}"
        ) from exc
    # A public name has authority only through a receipt already persisted in
    # the operation.  In particular, the absence of a WAL is never permission
    # to fresh-bind whatever now occupies the lexical operation-id name.
    binding = (
        _receipt_binding(journal_path, receipt, operation_id)
        if receipt is not None
        else None
    )
    return _PendingAnswerEvidence(binding, write_ahead)


def _receipt_binding(
    journal_path: Path,
    receipt: ArtifactReceipt,
    operation_id: str,
) -> _ArtifactBinding | None:
    """Translate a validated journal receipt without inspecting a live name."""
    expected = f"{PENDING_DIR}/{operation_id}.json"
    if receipt.relative_name != expected:
        return None
    return _ArtifactBinding(
        path=(Path(journal_path).parent / receipt.relative_name).absolute(),
        directory_identity=receipt.directory_identity,
        entry_state=receipt.entry_state,
        content_sha256=receipt.content_sha256,
    )


def _receipt_bound_file(
    journal_path: Path,
    receipt: ArtifactReceipt,
) -> _BoundFileReceipt:
    marker = receipt.terminal_marker
    if marker is None:
        raise DataError("Captured answer receipt holds no terminal WAL binding")
    return _BoundFileReceipt(
        target=(Path(journal_path).parent / receipt.relative_name).absolute(),
        directory_identity=receipt.directory_identity,
        entry_state=receipt.entry_state,
        content_sha256=receipt.content_sha256,
        marker_name=marker.name,
        marker_state=marker.entry_state,
        marker_revision=marker.content_sha256,
    )


def _read_artifact_binding(expected: _ArtifactBinding) -> bytes | None:
    """Read only the exact direct entry already bound by an observation."""
    try:
        with _open_bound_directory(expected.path.parent, create=False) as directory:
            parent = os.fstat(directory.descriptor)
            if (parent.st_dev, parent.st_ino) != expected.directory_identity:
                return None
            file_fd = os.open(
                expected.path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory.descriptor,
            )
            with os.fdopen(file_fd, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or _artifact_state(opened) != expected.entry_state
                ):
                    return None
                payload = handle.read()
                after = os.fstat(handle.fileno())
                entry = os.stat(
                    expected.path.name,
                    dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
                if (
                    _artifact_state(after) != expected.entry_state
                    or _artifact_state(entry) != expected.entry_state
                    or hashlib.sha256(payload).hexdigest()
                    != expected.content_sha256
                ):
                    return None
            _validate_bound_directory(directory)
            return payload
    except (DataError, OSError):
        return None


def _retire_artifact(expected: _ArtifactBinding) -> None:
    """Retire the exact answer through the private same-filesystem store.

    Moving first closes the public-name race; the exact private link is then
    removed without truncating its inode, so a legitimate hard-link backup
    remains intact. The public ``.pending`` name disappears only when the
    no-clobber move proves it is still the file ``forget`` bound.
    """
    try:
        with _open_cleanup_directory(
            expected.path.parent, expected.directory_identity
        ) as directory:
            if directory is None:
                _retire_detached_entry(
                    expected.directory_identity,
                    expected.path.name,
                    expected.entry_state[:4],
                    expected.content_sha256,
                    expected_ctime_ns=expected.entry_state[4],
                )
                return
            retired = _retire_exact_entry(
                directory,
                expected.path.name,
                expected.entry_state[:4],
                expected.content_sha256,
                expected_ctime_ns=expected.entry_state[4],
            )
            if not retired:
                current = _cleanup_bound_snapshot(
                    directory.descriptor, expected.path.name
                )
                if current == (expected.entry_state, expected.content_sha256):
                    raise DataError(
                        f"Exact artifact remained after cleanup: {expected.path}"
                    )
            _validate_bound_directory(directory)
    except (DataError, OSError) as exc:
        raise OperationError(
            "The operation's durable cleanup intent remains because its "
            f"artifact could not be retired safely: {exc}"
        ) from exc


def _retire_write_ahead(expected: _BoundWriteEvidence) -> None:
    """Retire exact operation-bound WAL evidence after its journal entry."""
    try:
        _retire_bound_write_evidence(expected)
    except (DataError, OSError) as exc:
        raise OperationError(
            "The operation's durable cleanup intent remains because its "
            f"write-ahead evidence could not be retired safely: {exc}"
        ) from exc


_FRAME_LENGTH_SIZE = 8
_FRAME_DIGEST_SIZE = hashlib.sha256().digest_size
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _response_spool_target(journal_path: Path, operation_id: str) -> Path:
    return (
        Path(journal_path).parent / PENDING_DIR / f"{operation_id}.frames"
    ).absolute()


def _response_spool_bound_file(
    journal_path: Path,
    receipt: ResponseSpoolReceipt,
) -> _BoundFileReceipt:
    marker = receipt.terminal_marker
    return _BoundFileReceipt(
        target=(Path(journal_path).parent / receipt.relative_name).absolute(),
        directory_identity=receipt.directory_identity,
        entry_state=receipt.initial_state,
        content_sha256=receipt.initial_sha256,
        marker_name=marker.name,
        marker_state=marker.entry_state,
        marker_revision=marker.content_sha256,
    )


def _response_spool_receipt_from_capture(
    operation_id: str,
    captured: _BoundFileReceipt,
) -> ResponseSpoolReceipt:
    receipt = ResponseSpoolReceipt(
        relative_name=f"{PENDING_DIR}/{operation_id}.frames",
        directory_identity=captured.directory_identity,
        initial_state=captured.entry_state,
        initial_sha256=captured.content_sha256,
        committed_size=0,
        committed_sha256=_EMPTY_SHA256,
        frame_count=0,
        terminal_marker=_TerminalMarkerReceipt(
            name=captured.marker_name,
            entry_state=captured.marker_state,
            content_sha256=captured.marker_revision,
        ),
    )
    return ResponseSpoolReceipt.from_dict(operation_id, receipt.to_dict())


def _response_spool_receipt_from_write_ahead(
    operation_id: str,
    evidence: _BoundWriteEvidence,
) -> ResponseSpoolReceipt | None:
    """Recover preparation that reached disk before its journal receipt."""
    public = evidence.public_answer_snapshot
    if (
        public is None
        or public[0][2] != 0
        or public[1] != _EMPTY_SHA256
        or evidence.generated_revision != _EMPTY_SHA256
    ):
        return None
    receipt = ResponseSpoolReceipt(
        relative_name=f"{PENDING_DIR}/{operation_id}.frames",
        directory_identity=evidence.directory_identity,
        initial_state=public[0],
        initial_sha256=public[1],
        committed_size=0,
        committed_sha256=_EMPTY_SHA256,
        frame_count=0,
        terminal_marker=_TerminalMarkerReceipt(
            name=evidence.marker_name,
            entry_state=evidence.marker_state,
            content_sha256=evidence.marker_revision,
        ),
    )
    return ResponseSpoolReceipt.from_dict(operation_id, receipt.to_dict())


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _read_response_spool_snapshot(
    journal_path: Path,
    receipt: ResponseSpoolReceipt,
) -> tuple[_ArtifactBinding, bytes] | None:
    """Read only the direct regular file bound by a durable spool receipt."""
    target = (Path(journal_path).parent / receipt.relative_name).absolute()
    try:
        with _open_bound_directory(target.parent, create=False) as directory:
            parent = os.fstat(directory.descriptor)
            if (parent.st_dev, parent.st_ino) != receipt.directory_identity:
                return None
            descriptor = os.open(
                target.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory.descriptor,
            )
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or (before.st_dev, before.st_ino) != receipt.file_identity
                ):
                    return None
                payload = _read_descriptor(descriptor)
                after = os.fstat(descriptor)
                named = os.stat(
                    target.name,
                    dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(named.st_mode)
                    or named.st_nlink != 1
                    or _artifact_state(before) != _artifact_state(after)
                    or _artifact_state(before) != _artifact_state(named)
                ):
                    return None
                binding = _ArtifactBinding(
                    path=target,
                    directory_identity=receipt.directory_identity,
                    entry_state=_artifact_state(before),
                    content_sha256=hashlib.sha256(payload).hexdigest(),
                )
            finally:
                os.close(descriptor)
            _validate_bound_directory(directory)
            return binding, payload
    except (DataError, OSError):
        return None


def _read_response_spool_state(
    journal_path: Path,
    receipt: ResponseSpoolReceipt,
) -> os.stat_result | None:
    """Bind the direct regular name of a spool without reading a byte of it.

    Every guarantee `_read_response_spool_snapshot` gets from the directory and
    the inode, for the questions that only need the file's size: a same-name
    replacement, a symlink, or a hard-linked name still answers `None` here.
    """
    target = (Path(journal_path).parent / receipt.relative_name).absolute()
    try:
        with _open_bound_directory(target.parent, create=False) as directory:
            parent = os.fstat(directory.descriptor)
            if (parent.st_dev, parent.st_ino) != receipt.directory_identity:
                return None
            named = os.stat(
                target.name,
                dir_fd=directory.descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(named.st_mode)
                or named.st_nlink != 1
                or (named.st_dev, named.st_ino) != receipt.file_identity
            ):
                return None
            _validate_bound_directory(directory)
            return named
    except (DataError, OSError):
        return None


def _decode_response_frames(payload: bytes) -> tuple[str, ...]:
    """Validate structural framing without interpreting any frame as JSON."""
    frames: list[str] = []
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < _FRAME_LENGTH_SIZE:
            raise OperationError("Response spool holds a torn frame length")
        header = payload[offset : offset + _FRAME_LENGTH_SIZE]
        length = struct.unpack(">Q", header)[0]
        offset += _FRAME_LENGTH_SIZE
        remaining = len(payload) - offset
        if remaining < length + _FRAME_DIGEST_SIZE:
            raise OperationError("Response spool holds a torn frame record")
        frame_bytes = payload[offset : offset + length]
        offset += length
        recorded_digest = payload[offset : offset + _FRAME_DIGEST_SIZE]
        offset += _FRAME_DIGEST_SIZE
        expected_digest = hashlib.sha256(header + frame_bytes).digest()
        if not hmac.compare_digest(recorded_digest, expected_digest):
            raise OperationError("Response spool frame checksum does not match")
        try:
            frames.append(frame_bytes.decode("utf-8", errors="strict"))
        except UnicodeError as exc:
            raise OperationError(
                "Response spool frame is not valid UTF-8"
            ) from exc
    return tuple(frames)


def _validated_response_frames(
    receipt: ResponseSpoolReceipt,
    payload: bytes,
) -> tuple[str, ...]:
    """Require the durable head, while allowing only a valid crash extension."""
    frames = _decode_response_frames(payload)
    if len(payload) < receipt.committed_size:
        raise OperationError("Response spool rolled back behind its durable head")
    committed_prefix = payload[: receipt.committed_size]
    if not hmac.compare_digest(
        hashlib.sha256(committed_prefix).hexdigest(),
        receipt.committed_sha256,
    ):
        raise OperationError("Response spool changed before its durable head")
    # The ordinary spool holds no crash extension, so its durable prefix is the
    # whole payload that was just decoded. Decoding those same bytes a second
    # time only to count them is what makes a streaming call quadratic.
    committed_frames = (
        frames
        if len(payload) == receipt.committed_size
        else _decode_response_frames(committed_prefix)
    )
    if len(committed_frames) != receipt.frame_count:
        raise OperationError(
            "Response spool durable frame count does not match its head"
        )
    uncommitted = len(frames) - receipt.frame_count
    if uncommitted > 1:
        raise OperationError(
            "Response spool holds more than one uncommitted frame"
        )
    return frames


def _response_spool_receipt_at_head(
    receipt: ResponseSpoolReceipt,
    payload: bytes,
    frames: tuple[str, ...],
) -> ResponseSpoolReceipt:
    return ResponseSpoolReceipt(
        relative_name=receipt.relative_name,
        directory_identity=receipt.directory_identity,
        initial_state=receipt.initial_state,
        initial_sha256=receipt.initial_sha256,
        committed_size=len(payload),
        committed_sha256=hashlib.sha256(payload).hexdigest(),
        frame_count=len(frames),
        terminal_marker=receipt.terminal_marker,
    )


def _confirm_response_spool_head(
    journal_path: Path,
    receipt: ResponseSpoolReceipt,
    expected_payload: bytes,
) -> None:
    if len(expected_payload) != receipt.committed_size:
        raise OperationError(
            "The exact response spool extended after its head was journalled"
        )
    observed = _read_response_spool_snapshot(journal_path, receipt)
    if observed is None or observed[1] != expected_payload:
        raise OperationError(
            "The exact response spool changed after its head was journalled"
        )
    _validated_response_frames(receipt, observed[1])


def _encode_response_frame(payload: str) -> bytes:
    if not isinstance(payload, str):
        raise OperationError("Response spool frames must be text")
    try:
        encoded = payload.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise OperationError("Response spool frame is not valid UTF-8") from exc
    header = struct.pack(">Q", len(encoded))
    return header + encoded + hashlib.sha256(header + encoded).digest()


def _response_spool_inspection_bytes(frames: tuple[str, ...]) -> bytes:
    """Render exact text-frame payloads without pretending they form a reply."""
    return (
        json.dumps(
            {
                "version": 1,
                "kind": "janki-response-frame-spool",
                "complete": False,
                "frames": [
                    {"encoding": "utf-8", "payload": payload}
                    for payload in frames
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _append_response_spool(
    journal_path: Path,
    receipt: ResponseSpoolReceipt,
    payload: str,
) -> ResponseSpoolReceipt:
    """Append and fsync one frame while proving the mutable file stayed bound."""
    observed = _read_response_spool_snapshot(journal_path, receipt)
    if observed is None:
        raise OperationError("The exact response spool is unavailable")
    binding, existing_payload = observed
    existing_frames = _validated_response_frames(receipt, existing_payload)
    if len(existing_payload) != receipt.committed_size:
        raise OperationError(
            "Response spool crash extension must be adopted before append"
        )
    record = _encode_response_frame(payload)
    target = binding.path
    try:
        with _open_bound_directory(target.parent, create=False) as directory:
            parent = os.fstat(directory.descriptor)
            if (parent.st_dev, parent.st_ino) != receipt.directory_identity:
                raise OperationError("The exact response spool is unavailable")
            descriptor = os.open(
                target.name,
                os.O_RDWR
                | os.O_APPEND
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory.descriptor,
            )
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or _artifact_state(before) != binding.entry_state
                ):
                    # The bound state carries dev, inode, size and both
                    # timestamps in nanoseconds, and no write can restore a
                    # ctime — so re-reading and rehashing the whole file here
                    # proves nothing the snapshot taken moments ago under this
                    # same lock did not. What the bytes are is proved after the
                    # append instead, against that snapshot.
                    raise OperationError(
                        "The exact response spool changed before append"
                    )
                written = 0
                while written < len(record):
                    count = os.write(descriptor, record[written:])
                    if count <= 0:
                        raise OSError("Could not append the complete response frame")
                    written += count
                # This is the durability boundary promised to the transport:
                # it may release the frame only after fsync has returned.
                os.fsync(descriptor)
                after = os.fstat(descriptor)
                named = os.stat(
                    target.name,
                    dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
                complete_payload = _read_descriptor(descriptor)
                confirmed = os.fstat(descriptor)
                # The validated snapshot already decoded the prefix, and this
                # exact record encodes exactly one frame. Proving the file is
                # that prefix followed by that record therefore proves the
                # frames without decoding and rechecksumming every earlier one.
                if (
                    not stat.S_ISREG(named.st_mode)
                    or named.st_nlink != 1
                    or (after.st_dev, after.st_ino) != receipt.file_identity
                    or _artifact_state(after) != _artifact_state(named)
                    or _artifact_state(after) != _artifact_state(confirmed)
                    or after.st_size != before.st_size + len(record)
                    or len(complete_payload) != before.st_size + len(record)
                    or not complete_payload.startswith(existing_payload)
                    or not complete_payload.endswith(record)
                ):
                    raise OperationError(
                        "The exact response spool changed during append"
                    )
                complete_frames = (*existing_frames, payload)
            finally:
                os.close(descriptor)
            _validate_bound_directory(directory)
            updated = _response_spool_receipt_at_head(
                receipt,
                complete_payload,
                complete_frames,
            )
    except OperationError:
        raise
    except (DataError, OSError) as exc:
        raise OperationError(f"Could not append response frame: {exc}") from exc
    return updated


def _retire_response_spool(expected: _ArtifactBinding) -> None:
    """Retire an exact final spool binding only from durable forget cleanup."""
    _retire_artifact(expected)


def _observe_reply(
    journal_path: Path,
    operation: Operation,
) -> tuple[ReplyObservation, _PendingAnswerEvidence]:
    """Bind and read one operation's exact reply without trusting a path string."""
    recorded = operation.artifact is not None or operation.state in {
        "result_captured",
        "committed",
    }
    if operation.cleanup is not None:
        # A durable discard decision won. Its evidence remains solely so exact
        # cleanup can finish; no stale status/error frame may expose it again.
        return ReplyObservation(None, recorded=recorded, interrupted=False), (
            _PendingAnswerEvidence(None, None)
        )
    evidence = _pending_answer_evidence(
        journal_path, operation.operation_id, operation.artifact
    )
    payload: bytes | None = None
    if (
        evidence.write_ahead is not None
        and evidence.write_ahead.reply_complete
    ):
        payload = _read_bound_write_evidence(evidence.write_ahead)
    if payload is None and evidence.artifact is not None:
        payload = _read_artifact_binding(evidence.artifact)
    recorded = recorded or evidence.reply_complete
    return (
        ReplyObservation(
            payload,
            recorded=recorded,
            interrupted=(
                operation.artifact is None
                and evidence.write_ahead is not None
                and not evidence.write_ahead.reply_complete
            ),
        ),
        evidence,
    )


def reply_observation(
    journal_path: Path,
    operation: Operation,
) -> ReplyObservation:
    """Observe readable reply bytes without making a lexical path into proof."""
    return _observe_reply(journal_path, operation)[0]


def response_spool_observation(
    journal_path: Path,
    operation: Operation,
) -> ResponseSpoolObservation:
    """Classify response frames without adopting, settling, or decoding JSON."""
    receipt = operation.response_spool
    if receipt is None or operation.cleanup is not None:
        return ResponseSpoolObservation(
            readable=False,
            recorded=False,
            recovery_pending=False,
        )
    observed = _read_response_spool_snapshot(journal_path, receipt)
    if observed is None:
        return ResponseSpoolObservation(
            readable=False,
            recorded=receipt.frame_count > 0,
            recovery_pending=False,
        )
    payload = observed[1]
    recorded = bool(payload) or receipt.frame_count > 0
    try:
        frames = _validated_response_frames(receipt, payload)
    except OperationError:
        return ResponseSpoolObservation(
            readable=False,
            recorded=recorded,
            recovery_pending=False,
        )
    # Inspection deliberately does not adopt a crash extension. Until `end`
    # or the exact provider recovery records that head, those bytes are known
    # evidence but are not yet an inspectable committed-frame view.
    recovery_pending = len(payload) > receipt.committed_size
    readable = not recovery_pending and bool(frames)
    return ResponseSpoolObservation(
        readable=readable,
        recorded=recorded,
        recovery_pending=recovery_pending,
    )


def prepare_artifact_store(journal_path: Path) -> Path:
    """Create and validate the direct recovery directory before dispatch."""
    directory = Path(journal_path).parent / PENDING_DIR
    try:
        return prepare_bound_directory(directory)
    except DataError as exc:
        raise OperationError(
            f"Could not prepare pending answer store {directory}: {exc}"
        ) from exc


def capture_artifact(
    journal_path: Path, operation_id: str, payload: bytes
) -> ArtifactReceipt:
    """Publish a paid reply and return exact proof for the journal.

    The operation-bound terminal WAL intentionally remains after this function
    returns.  :meth:`OperationJournal.capture_result` persists the receipt
    first and finalizes that WAL second, closing both sides of the crash seam.
    """
    if not _valid_pending_operation_id(operation_id):
        raise OperationError(f"Invalid operation ID for a pending answer: {operation_id!r}")
    target = Path(journal_path).parent / PENDING_DIR / f"{operation_id}.json"
    bound = _capture_bytes_bound(target, payload)
    return ArtifactReceipt(
        relative_name=f"{PENDING_DIR}/{target.name}",
        directory_identity=bound.directory_identity,
        entry_state=bound.entry_state,
        content_sha256=bound.content_sha256,
        terminal_marker=_TerminalMarkerReceipt(
            name=bound.marker_name,
            entry_state=bound.marker_state,
            content_sha256=bound.marker_revision,
        ),
    )


def _finalize_artifact(
    journal_path: Path,
    receipt: ArtifactReceipt,
) -> None:
    if receipt.terminal_marker is None:
        return
    bound = _receipt_bound_file(journal_path, receipt)
    try:
        _finalize_bound_capture(bound)
    except DataError as exc:
        raise OperationError(
            "The captured reply is journalled, but its write-ahead marker "
            f"could not be finalized safely: {exc}"
        ) from exc


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class OperationAuthorization:
    """One exact paid call a batch is asking to be allowed to make.

    The same six facts an ordinary `authorize` records, carried as a value so a
    whole childset can be validated before a single byte of authority is
    written. It is not itself authority: only a journalled row is that.
    """

    operation_id: str
    kind: str
    source_file: str
    source_sha256: str
    request_fp: str
    model: str


@dataclass(frozen=True, slots=True)
class OperationBatch:
    """The one finite childset a batch reserved, and how many may run at once.

    Durable and immutable. The membership outlives the rows: an ordinary
    `forget` removes a settled child's entry, and the id it used stays spent
    here so a replayed request can never ride on a decision already consumed.
    """

    batch_id: str
    child_operation_ids: tuple[str, ...]
    concurrency_limit: int
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "children": list(self.child_operation_ids),
            "concurrency_limit": self.concurrency_limit,
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_dict(cls, batch_id: str, raw: Mapping[str, Any]) -> OperationBatch:
        refusal = f"Operation batch {batch_id!r} is not a valid batch record"
        if not _valid_batch_identifier(batch_id):
            raise OperationError(refusal)
        if not isinstance(raw, Mapping) or set(raw) != {
            "children",
            "concurrency_limit",
            "manifest_sha256",
        }:
            raise OperationError(refusal)
        children = raw["children"]
        limit = raw["concurrency_limit"]
        if (
            not isinstance(children, list)
            or not children
            or any(
                not isinstance(child, str)
                or not _valid_pending_operation_id(child)
                for child in children
            )
            or len(set(children)) != len(children)
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
            or not _is_sha256_digest(raw["manifest_sha256"])
        ):
            raise OperationError(refusal)
        return cls(
            batch_id=batch_id,
            child_operation_ids=tuple(children),
            concurrency_limit=limit,
            manifest_sha256=raw["manifest_sha256"],
        )


def _validated_batch_children(
    batch_id: str,
    children: Iterable[OperationAuthorization],
) -> tuple[OperationAuthorization, ...]:
    """Materialize one finite childset and prove every request is exact.

    Everything here is structural and happens before the journal lock, so a
    childset janki cannot authorize in full never reaches the point of writing
    part of it.
    """
    authorizations = tuple(children)
    if not authorizations:
        raise OperationError(
            f"Operation batch {batch_id!r} must reserve at least one child "
            "operation"
        )
    for child in authorizations:
        if not isinstance(child, OperationAuthorization):
            raise OperationError(
                f"Operation batch {batch_id!r} needs exact extraction "
                f"authorizations, got {type(child).__name__}"
            )
        if (
            not _valid_pending_operation_id(child.operation_id)
            or child.kind not in _BATCH_KINDS
            or not isinstance(child.source_file, str)
            or not child.source_file
            or not _is_sha256_digest(child.source_sha256)
            or not _is_sha256_digest(child.request_fp)
            or not isinstance(child.model, str)
            or not child.model
        ):
            raise OperationError(
                f"Operation batch {batch_id!r} child {child.operation_id!r} is "
                "not an exact extraction authorization"
            )
    return authorizations


def _blocked_refusal(first: Operation) -> str:
    """What a person is told when an unsettled call stops the next one."""
    return (
        f"Operation {first.operation_id!r} for {first.source_file} "
        f"is {first.state!r} and janki will not start another paid "
        "call until it is settled. 'janki operations' shows it "
        "and the action that settles it."
    )


def _validate_batch_membership(
    path: Path,
    operations: Mapping[str, Operation],
    batches: Mapping[str, OperationBatch],
) -> None:
    """Prove every row and every batch agree about who belongs to what.

    Membership is the thing a claim counts slots against, so an entry that
    silently points at a missing or different batch — or a member row that
    quietly dropped its batch id — would be a way to dispatch outside the
    limit. A member with no row at all is ordinary: that is what `forget`
    leaves behind, and it is never recreated.
    """
    owner_of: dict[str, str] = {}
    for batch_id in sorted(batches):
        for child in batches[batch_id].child_operation_ids:
            owner = owner_of.setdefault(child, batch_id)
            if owner != batch_id:
                raise OperationError(
                    f"{path} has operation {child!r} in two operation batches "
                    f"{owner!r} and {batch_id!r}"
                )
    for operation_id in sorted(operations):
        held = operations[operation_id]
        owner = owner_of.get(operation_id)
        if not held.batch_id:
            if owner is not None:
                raise OperationError(
                    f"Operation {operation_id!r} holds no batch id but is a "
                    f"member of operation batch {owner!r}"
                )
            continue
        if held.batch_id not in batches:
            raise OperationError(
                f"Operation {operation_id!r} names missing operation batch "
                f"{held.batch_id!r}"
            )
        if owner != held.batch_id:
            raise OperationError(
                f"Operation {operation_id!r} is not a member of operation "
                f"batch {held.batch_id!r}"
            )


@dataclass(frozen=True, slots=True)
class Operation:
    """One paid call, and what is known about it."""

    operation_id: str
    kind: str
    state: str
    source_file: str
    source_sha256: str
    request_fp: str
    model: str
    authorized_at: str
    updated_at: str
    #: Exact public-answer proof, once capture is journalled.
    artifact: ArtifactReceipt | None = None
    #: Stable identity of a streaming response spool prepared before dispatch.
    response_spool: ResponseSpoolReceipt | None = None
    #: Why it stopped, for a terminal state.
    detail: str = ""
    #: Durable exact deletion authority while ``forget`` retires recovery data.
    cleanup: _CleanupIntent | None = None
    #: The batch that reserved this authority, for a batch child; "" otherwise.
    batch_id: str = ""

    @property
    def money_may_have_been_spent(self) -> bool:
        """Whether a provider may have billed for this.

        `dispatching` counts. The request may have reached the provider in the
        instant before the process died, and a journal that says "no" there
        would be guessing about someone's money.
        """
        return self.state in {
            "dispatching",
            "running",
            "result_captured",
            "committed",
            "outcome_unknown",
        }

    @property
    def needs_a_person(self) -> bool:
        return self.cleanup is None and self.state in {
            "outcome_unknown",
            "result_captured",
        }

    @property
    def blocks_spending(self) -> bool:
        """Whether this entry still holds an unsettled money decision."""
        return self.cleanup is None and self.state in BLOCKS_SPENDING

    def to_dict(self) -> dict[str, Any]:
        value = {
            "kind": self.kind,
            "state": self.state,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "request_fp": self.request_fp,
            "model": self.model,
            "authorized_at": self.authorized_at,
            "updated_at": self.updated_at,
        }
        if self.artifact is not None:
            value["artifact"] = self.artifact.to_dict()
        if self.response_spool is not None:
            value["response_spool"] = self.response_spool.to_dict()
        if self.detail:
            value["detail"] = self.detail
        if self.cleanup is not None:
            value["cleanup"] = self.cleanup.to_dict()
        # Only a batch child carries one, so an ordinary journal keeps exactly
        # the wire shape every released janki has written.
        if self.batch_id:
            value["batch_id"] = self.batch_id
        return value

    @classmethod
    def from_dict(
        cls,
        journal_path: Path,
        operation_id: str,
        raw: Mapping[str, Any],
    ) -> Operation:
        if not isinstance(raw, Mapping):
            raise OperationError(
                f"Operation {operation_id!r} must be an object, got "
                f"{type(raw).__name__}"
            )
        state = str(raw.get("state", ""))
        if state not in _TRANSITIONS:
            raise OperationError(
                f"Operation {operation_id!r} holds unknown state {state!r}"
            )
        cleanup = None
        if "cleanup" in raw:
            cleanup_raw = raw["cleanup"]
            if cleanup_raw is None:
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid cleanup intent"
                )
            cleanup = _CleanupIntent.from_dict(
                journal_path, operation_id, cleanup_raw
            )
        artifact = None
        if "artifact" in raw:
            artifact_raw = raw["artifact"]
            if artifact_raw is None:
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid artifact receipt"
                )
            artifact = ArtifactReceipt.from_dict(operation_id, artifact_raw)
        response_spool = None
        if "response_spool" in raw:
            response_spool_raw = raw["response_spool"]
            if response_spool_raw is None:
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid response "
                    "spool receipt"
                )
            response_spool = ResponseSpoolReceipt.from_dict(
                operation_id, response_spool_raw
            )
        batch_id = ""
        if "batch_id" in raw:
            batch_id = raw["batch_id"]
            if not _valid_batch_identifier(batch_id):
                raise OperationError(
                    f"Operation {operation_id!r} holds an invalid batch id"
                )
        receipt_states = {"result_captured", "committed"}
        if (state in receipt_states) != (artifact is not None):
            requirement = (
                "requires an artifact receipt"
                if state in receipt_states
                else "cannot hold an artifact receipt"
            )
            raise OperationError(
                f"Operation {operation_id!r} in state {state!r} {requirement}"
            )
        return cls(
            operation_id=operation_id,
            kind=str(raw.get("kind", "")),
            state=state,
            source_file=str(raw.get("source_file", "")),
            source_sha256=str(raw.get("source_sha256", "")),
            request_fp=str(raw.get("request_fp", "")),
            model=str(raw.get("model", "")),
            authorized_at=str(raw.get("authorized_at", "")),
            updated_at=str(raw.get("updated_at", "")),
            artifact=artifact,
            response_spool=response_spool,
            detail=str(raw.get("detail", "")),
            cleanup=cleanup,
            batch_id=batch_id,
        )


@dataclass
class OperationJournal:
    """The journal file, and the only writer of it.

    Every mutator re-reads under the path lock, applies one transition, and
    writes the whole file atomically. That is deliberately not the ledger's
    load-once/save-once pattern: two processes may legitimately be running
    extractions, and each transition has to land on whatever the other left
    rather than on a snapshot from before it started.
    """

    path: Path
    operations: dict[str, Operation] = field(default_factory=dict)
    #: Durable membership for every batch this journal has ever reserved. Empty
    #: for an ordinary journal, and read-only to everything but `authorize_batch`.
    batches: dict[str, OperationBatch] = field(default_factory=dict)
    _wire_revision: str | None = field(default=None, repr=False)
    _expected_absent: bool = field(default=True, repr=False)

    @classmethod
    def load(cls, path: Path) -> OperationJournal:
        path = Path(path)
        if path.is_symlink():
            raise OperationError(f"Refusing a symlinked operation journal: {path}")
        if path.parent.exists() and (
            path.parent.is_symlink() or not path.parent.is_dir()
        ):
            raise OperationError(
                f"Refusing non-directory operation journal parent: {path.parent}"
            )
        try:
            wire = read_bytes_bound(path)
            raw = json.loads(wire.decode("utf-8", errors="strict"))
        except FileNotFoundError:
            return cls(path=path)
        except (JankiError, OSError, UnicodeError, ValueError) as exc:
            raise OperationError(f"Could not read {path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise OperationError(f"{path} must hold an object")
        entries = raw.get("operations")
        if entries is None:
            entries = {}
        if not isinstance(entries, Mapping):
            raise OperationError(f"{path} operations must be an object")
        reserved = raw.get("batches")
        if reserved is None:
            reserved = {}
        if not isinstance(reserved, Mapping):
            raise OperationError(f"{path} batches must be an object")
        operations = {
            str(key): Operation.from_dict(path, str(key), value)
            for key, value in entries.items()
        }
        batches = {
            str(key): OperationBatch.from_dict(str(key), value)
            for key, value in reserved.items()
        }
        _validate_batch_membership(path, operations, batches)
        return cls(
            path=path,
            operations=operations,
            batches=batches,
            _wire_revision=hashlib.sha256(wire).hexdigest(),
            _expected_absent=False,
        )

    def _write(self) -> None:
        payload: dict[str, Any] = {
            "version": 1,
            "operations": {
                key: self.operations[key].to_dict() for key in sorted(self.operations)
            },
        }
        if self.batches:
            payload["batches"] = {
                key: self.batches[key].to_dict() for key in sorted(self.batches)
            }
        rendered = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=False
        ) + "\n"
        try:
            atomic_write_text_bound(
                self.path,
                rendered,
                expected_revision=self._wire_revision,
                expected_absent=self._expected_absent,
            )
        except DataError as exc:
            raise OperationError(
                f"Could not write operation journal {self.path}: {exc}"
            ) from exc
        self._wire_revision = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        self._expected_absent = False

    # --- transitions --------------------------------------------------------

    def authorize(
        self,
        operation_id: str,
        *,
        kind: str,
        source_file: str,
        source_sha256: str,
        request_fp: str,
        model: str,
    ) -> Operation:
        """Record one-use authority for a paid call that has not been sent.

        Refuses an ID that already exists in any state. An authorization is a
        person agreeing to spend money once; reusing the identity would let a
        replayed request ride on a decision already made.
        """
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            current._refuse_reused_identity(operation_id)
            blocking = current.blocking()
            if blocking:
                raise OperationError(_blocked_refusal(blocking[0]))
            now = _now()
            operation = Operation(
                operation_id=operation_id,
                kind=kind,
                state="authorized",
                source_file=source_file,
                source_sha256=source_sha256,
                request_fp=request_fp,
                model=model,
                authorized_at=now,
                updated_at=now,
            )
            current.operations[operation_id] = operation
            current.path = self.path
            current._write()
            self.operations = current.operations
            self.batches = current.batches
            return operation

    def _batch_retaining(self, operation_id: str) -> str | None:
        """The batch that still holds this operation id, if any holds it.

        Membership is immutable, so this answers "has this identity already
        been spent?" long after `forget` removed the row that used it.
        """
        for batch_id in sorted(self.batches):
            if operation_id in self.batches[batch_id].child_operation_ids:
                return batch_id
        return None

    def _refuse_reused_identity(self, operation_id: str) -> None:
        """Refuse one operation id that any authority has already consumed."""
        held = self.operations.get(operation_id)
        if held is not None:
            raise OperationError(
                f"Operation {operation_id!r} is already authorized and is "
                f"{held.state!r}; authority is one-use."
            )
        retained = self._batch_retaining(operation_id)
        if retained is not None:
            raise OperationError(
                f"Operation {operation_id!r} is retained by operation batch "
                f"{retained!r}; authority is one-use."
            )

    def authorize_batch(
        self,
        batch_id: str,
        children: Iterable[OperationAuthorization],
        *,
        concurrency_limit: int,
        manifest_sha256: str,
    ) -> OperationBatch:
        """Reserve one finite extraction childset in a single atomic write.

        The one exception to "one authority, one call at a time", and it is an
        exception only about *when* the authorities are written. Every child is
        still a one-use authority that has to be consumed by its own
        :meth:`claim_batch_dispatch`; the batch merely says a person agreed to
        this exact finite set of extraction requests, at this concurrency, over
        this exact manifest.

        Reserving them together is what makes bounded parallelism safe to
        account for: the whole childset and its limit reach disk before any
        request leaves, so a crash can never leave a batch whose real size
        nobody can reconstruct. It is all or nothing — a childset that cannot
        be authorized in full writes nothing at all.
        """
        if not _valid_batch_identifier(batch_id):
            raise OperationError(
                f"Operation batch {batch_id!r} needs a valid batch identifier"
            )
        if (
            not isinstance(concurrency_limit, int)
            or isinstance(concurrency_limit, bool)
            or concurrency_limit < 1
        ):
            raise OperationError(
                f"Operation batch {batch_id!r} needs a positive integer "
                "concurrency limit"
            )
        if not _is_sha256_digest(manifest_sha256):
            raise OperationError(
                f"Operation batch {batch_id!r} needs an exact manifest digest"
            )
        authorizations = _validated_batch_children(batch_id, children)
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            if batch_id in current.batches:
                raise OperationError(
                    f"Operation batch {batch_id!r} already exists; batch "
                    "authority is one-use."
                )
            blocking = current.blocking()
            if blocking:
                raise OperationError(_blocked_refusal(blocking[0]))
            named: set[str] = set()
            requested: set[str] = set()
            for child in authorizations:
                if child.operation_id in named:
                    raise OperationError(
                        f"Operation batch {batch_id!r} names operation "
                        f"{child.operation_id!r} twice"
                    )
                # Two children carrying one exact request are the same paid
                # call twice. Across *settled* batches that is a legitimate
                # retry; inside one reservation nothing has settled yet.
                if child.request_fp in requested:
                    raise OperationError(
                        f"Operation batch {batch_id!r} reserves one exact "
                        f"request twice: {child.request_fp!r}"
                    )
                named.add(child.operation_id)
                requested.add(child.request_fp)
                current._refuse_reused_identity(child.operation_id)
            batch = OperationBatch.from_dict(
                batch_id,
                OperationBatch(
                    batch_id=batch_id,
                    child_operation_ids=tuple(
                        child.operation_id for child in authorizations
                    ),
                    concurrency_limit=concurrency_limit,
                    manifest_sha256=manifest_sha256,
                ).to_dict(),
            )
            now = _now()
            for child in authorizations:
                current.operations[child.operation_id] = Operation(
                    operation_id=child.operation_id,
                    kind=child.kind,
                    state="authorized",
                    source_file=child.source_file,
                    source_sha256=child.source_sha256,
                    request_fp=child.request_fp,
                    model=child.model,
                    authorized_at=now,
                    updated_at=now,
                    batch_id=batch_id,
                )
            current.batches[batch_id] = batch
            current.path = self.path
            current._write()
            self.operations = current.operations
            self.batches = current.batches
            return batch

    def claim_batch_dispatch(
        self,
        operation_id: str,
        *,
        batch_id: str,
        request_fp: str,
        manifest_sha256: str,
    ) -> Operation:
        """Consume one child's authority, if the batch still has a free slot.

        This is the only door from `authorized` to `dispatching` for a batch
        child, which is what makes the stored limit a rule rather than advice:
        the count is taken from the journal's own member rows under the same
        lock that writes the transition, so two callers racing at one instant
        cannot both find the last slot free.

        A slot is occupied by every member that is dispatching, running, or
        resting in an unknown outcome. The unknown one never gives its slot
        back — nothing here redispatches it, and pretending the call ended
        would be inventing what somebody's money bought.
        """
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            batch = current.batches.get(batch_id)
            if batch is None:
                raise OperationError(
                    f"No operation batch {batch_id!r} to claim a dispatch slot"
                )
            held = current.operations.get(operation_id)
            if held is None:
                raise OperationError(
                    f"No operation {operation_id!r} to claim a dispatch slot"
                )
            if (
                operation_id not in batch.child_operation_ids
                or held.batch_id != batch_id
            ):
                raise OperationError(
                    f"Operation {operation_id!r} is not a member of operation "
                    f"batch {batch_id!r}"
                )
            if batch.manifest_sha256 != manifest_sha256:
                raise OperationError(
                    f"Operation batch {batch_id!r} was authorized for a "
                    "different exact manifest"
                )
            if held.request_fp != request_fp:
                raise OperationError(
                    f"Operation {operation_id!r} was authorized for a "
                    "different exact request"
                )
            if held.cleanup is not None:
                raise OperationError(
                    f"Operation {operation_id!r} is being forgotten; rerun "
                    f"'janki operations --forget {operation_id}' to finish cleanup"
                )
            if held.state != "authorized":
                raise OperationError(
                    f"Operation {operation_id!r} is {held.state!r} and cannot "
                    "claim a dispatch slot; batch authority is one-use"
                )
            occupied = sum(
                1
                for member in batch.child_operation_ids
                if (row := current.operations.get(member)) is not None
                and row.cleanup is None
                and row.state in BATCH_OCCUPIED_STATES
            )
            if occupied >= batch.concurrency_limit:
                raise OperationError(
                    f"Operation batch {batch_id!r} is at its concurrency limit "
                    f"of {batch.concurrency_limit}; another child must settle "
                    f"before {operation_id!r} may dispatch"
                )
            claimed = self._move_under_lock(
                current,
                operation_id,
                "dispatching",
                batched_claim=True,
            )
            self.batches = current.batches
            return claimed

    def advance(
        self,
        operation_id: str,
        state: str,
        *,
        detail: str = "",
    ) -> Operation:
        """Move one operation to `state`, or refuse if that move is not legal."""
        if state not in _TRANSITIONS:
            raise OperationError(f"Unknown operation state {state!r}")
        if state == "result_captured":
            raise OperationError(
                f"Operation {operation_id!r} cannot move to 'result_captured' "
                "through advance; capture_result must bind its exact receipt"
            )
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            return self._move_under_lock(
                current, operation_id, state, detail=detail
            )

    def begin_response_capture(
        self,
        operation_id: str,
    ) -> ResponseSpoolReceipt:
        """Prepare and durably bind an empty frame spool before transport.

        A preparation interrupted between file publication and the journal
        write leaves an operation-bound marker.  The exact retry adopts that
        marker; it never binds an unproven same-name file.
        """
        if not _valid_pending_operation_id(operation_id):
            raise OperationError(
                f"Invalid operation ID for response capture: {operation_id!r}"
            )
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            held = current.operations.get(operation_id)
            if held is None:
                raise OperationError(
                    f"No operation {operation_id!r} to prepare for response capture"
                )
            if held.cleanup is not None:
                raise OperationError(
                    f"Operation {operation_id!r} is being forgotten; rerun "
                    f"'janki operations --forget {operation_id}' to finish cleanup"
                )
            if held.response_spool is not None:
                observed = _read_response_spool_snapshot(
                    self.path, held.response_spool
                )
                if observed is None:
                    raise OperationError(
                        f"Operation {operation_id!r} cannot find its exact "
                        "response spool"
                    )
                _validated_response_frames(held.response_spool, observed[1])
                try:
                    _finalize_bound_capture(
                        _response_spool_bound_file(
                            self.path, held.response_spool
                        )
                    )
                except DataError as exc:
                    raise OperationError(
                        f"Operation {operation_id!r} response spool is "
                        f"journalled, but its preparation marker could not be "
                        f"finalized safely: {exc}"
                    ) from exc
                self.operations = current.operations
                return held.response_spool
            if held.state != "authorized":
                raise OperationError(
                    f"Operation {operation_id!r} must prepare response capture "
                    "before dispatch"
                )

            target = _response_spool_target(self.path, operation_id)
            try:
                write_ahead = _bound_write_evidence(target)
                if write_ahead is None:
                    captured = _capture_bytes_bound(target, b"")
                    receipt = _response_spool_receipt_from_capture(
                        operation_id, captured
                    )
                else:
                    receipt = _response_spool_receipt_from_write_ahead(
                        operation_id, write_ahead
                    )
                    if receipt is None:
                        raise OperationError(
                            f"Operation {operation_id!r} has interrupted or "
                            "foreign response spool preparation"
                        )
            except OperationError:
                raise
            except (DataError, OSError) as exc:
                raise OperationError(
                    f"Could not prepare response capture for operation "
                    f"{operation_id!r}: {exc}"
                ) from exc

            prepared = Operation(
                operation_id=held.operation_id,
                kind=held.kind,
                state=held.state,
                source_file=held.source_file,
                source_sha256=held.source_sha256,
                request_fp=held.request_fp,
                model=held.model,
                authorized_at=held.authorized_at,
                updated_at=_now(),
                artifact=held.artifact,
                response_spool=receipt,
                detail=held.detail,
                batch_id=held.batch_id,
            )
            current.operations[operation_id] = prepared
            current.path = self.path
            current._write()
            self.operations = current.operations
            try:
                _finalize_bound_capture(
                    _response_spool_bound_file(self.path, receipt)
                )
            except DataError as exc:
                raise OperationError(
                    f"Operation {operation_id!r} response spool is journalled, "
                    f"but its preparation marker could not be finalized safely: {exc}"
                ) from exc
            return receipt

    def _record_response_spool_head_under_lock(
        self,
        current: OperationJournal,
        held: Operation,
        receipt: ResponseSpoolReceipt,
    ) -> Operation:
        """Persist one fsynced spool head while the journal lock is held."""
        updated = Operation(
            operation_id=held.operation_id,
            kind=held.kind,
            state=held.state,
            source_file=held.source_file,
            source_sha256=held.source_sha256,
            request_fp=held.request_fp,
            model=held.model,
            authorized_at=held.authorized_at,
            updated_at=_now(),
            artifact=held.artifact,
            response_spool=receipt,
            detail=held.detail,
            batch_id=held.batch_id,
        )
        current.operations[held.operation_id] = updated
        current.path = self.path
        current._write()
        self.operations = current.operations
        return updated

    def _recover_response_spool_head_under_lock(
        self,
        current: OperationJournal,
        held: Operation,
        *,
        unavailable_ok: bool = False,
        invalid_ok: bool = False,
    ) -> Operation:
        """Adopt the sole complete frame left past a durable spool head.

        The provider callback fsyncs a frame before the journal records its new
        head.  If the process dies in that seam, the next operation holding the
        journal lock is allowed to adopt exactly that one validated extension.
        Missing, replaced, or malformed evidence may be preserved for an
        explicit discard path without preventing a person from ending a call.

        Only an extension is recovery's business, and only a size can make one:
        a spool still exactly as long as its durable head has nothing to adopt,
        which one bound stat settles.  Every caller that needs the frames
        themselves validates them for its own use — this must not become the
        place a streaming append pays to reread the whole file each frame.
        """
        receipt = held.response_spool
        if receipt is None:
            raise OperationError(
                f"Operation {held.operation_id!r} has no response spool"
            )
        state = _read_response_spool_state(self.path, receipt)
        if state is not None and state.st_size == receipt.committed_size:
            return held
        observed = _read_response_spool_snapshot(self.path, receipt)
        if observed is None:
            if unavailable_ok:
                return held
            raise OperationError("The exact response spool is unavailable")
        try:
            frames = _validated_response_frames(receipt, observed[1])
        except OperationError:
            if invalid_ok:
                return held
            raise
        if len(observed[1]) > receipt.committed_size:
            adopted_spool = _response_spool_receipt_at_head(
                receipt,
                observed[1],
                frames,
            )
            held = self._record_response_spool_head_under_lock(
                current, held, adopted_spool
            )
            _confirm_response_spool_head(
                self.path, adopted_spool, observed[1]
            )
        return held

    def append_response_frame(self, operation_id: str, payload: str) -> None:
        """Append one exact text frame and fsync it before returning."""
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            held = current.operations.get(operation_id)
            if held is None:
                raise OperationError(
                    f"No operation {operation_id!r} to append a response frame"
                )
            if held.cleanup is not None:
                raise OperationError(
                    f"Operation {operation_id!r} is being forgotten; its "
                    "response spool cannot be changed"
                )
            if held.response_spool is None:
                raise OperationError(
                    f"Operation {operation_id!r} has no response spool"
                )
            if held.state not in {"dispatching", "running"}:
                raise OperationError(
                    f"Operation {operation_id!r} in state {held.state!r} "
                    "cannot append a provider response frame"
                )
            held = self._recover_response_spool_head_under_lock(current, held)
            updated_spool = _append_response_spool(
                # ``held`` now carries the adopted exact head, if recovery was
                # needed; the low-level writer refuses any extension itself.
                self.path, held.response_spool, payload
            )
            updated = self._record_response_spool_head_under_lock(
                current, held, updated_spool
            )
            # The appended bytes were proved against this receipt two steps
            # ago, under this same lock, and nothing since has touched the
            # spool. All that is left to confirm is that the journal write did
            # not disturb the file it just described, which is a bound stat.
            recorded = _read_response_spool_state(self.path, updated_spool)
            if (
                recorded is None
                or recorded.st_size != updated_spool.committed_size
            ):
                raise OperationError(
                    "The exact response spool changed after its head was journalled"
                )
            self.operations[operation_id] = updated

    def read_response_frames(self, operation_id: str) -> tuple[str, ...]:
        """Return complete exact frames, refusing torn or corrupt records."""
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            held = current.operations.get(operation_id)
            if held is None:
                raise OperationError(
                    f"No operation {operation_id!r} to read response frames"
                )
            if held.cleanup is not None:
                raise OperationError(
                    f"Operation {operation_id!r} is being forgotten; its "
                    "response spool is no longer available to read"
                )
            if held.response_spool is None:
                raise OperationError(
                    f"Operation {operation_id!r} has no response spool"
                )
            held = self._recover_response_spool_head_under_lock(current, held)
            # Recovery only ever adopts a newer head, so the receipt checked
            # above is still there. It answers "is there an extension" from a
            # stat, which is what makes appending cheap, so the frames
            # themselves are read here — by the one caller that wants them.
            receipt = held.response_spool
            observed = (
                _read_response_spool_snapshot(self.path, receipt)
                if receipt is not None
                else None
            )
            if receipt is None or observed is None:
                raise OperationError("The exact response spool is unavailable")
            frames = _validated_response_frames(receipt, observed[1])
            self.operations = current.operations
            return frames

    def _move_under_lock(
        self,
        current: OperationJournal,
        operation_id: str,
        state: str,
        *,
        artifact: ArtifactReceipt | None = None,
        detail: str = "",
        batched_claim: bool = False,
    ) -> Operation:
        """The move itself, for a caller already holding this journal's lock.

        Separate from `advance` because the lock is not reentrant and `end`
        has to decide *and* write without letting go: it reads a state, picks
        a destination from it, and a reply landing in between would otherwise
        be recorded under a decision made about a different state.
        """
        if (state == "result_captured") != (artifact is not None):
            raise OperationError(
                f"Operation {operation_id!r} cannot attach or replace an "
                "artifact receipt in state {state!r}"
            )
        held = self._require_move_under_lock(
            current,
            operation_id,
            state,
            artifact=artifact,
            batched_claim=batched_claim,
        )
        moved = Operation(
            operation_id=held.operation_id,
            kind=held.kind,
            state=state,
            source_file=held.source_file,
            source_sha256=held.source_sha256,
            request_fp=held.request_fp,
            model=held.model,
            authorized_at=held.authorized_at,
            updated_at=_now(),
            artifact=artifact if artifact is not None else held.artifact,
            response_spool=held.response_spool,
            detail=detail or held.detail,
            batch_id=held.batch_id,
        )
        current.operations[operation_id] = moved
        current.path = self.path
        current._write()
        self.operations = current.operations
        return moved

    def _require_move_under_lock(
        self,
        current: OperationJournal,
        operation_id: str,
        state: str,
        *,
        artifact: ArtifactReceipt | None = None,
        batched_claim: bool = False,
    ) -> Operation:
        """Validate one transition against the locked journal snapshot."""
        held = current.operations.get(operation_id)
        if held is None:
            raise OperationError(f"No operation {operation_id!r} to advance")
        if held.cleanup is not None:
            raise OperationError(
                f"Operation {operation_id!r} is being forgotten; rerun "
                f"'janki operations --forget {operation_id}' to finish cleanup"
            )
        # A batch child leaves `authorized` through its claim or not at all.
        # Every other mover — `advance`, `end`, capture, commit — passes here,
        # so closing the door once closes it for all of them rather than
        # trusting each caller to remember the limit exists.
        if (
            not batched_claim
            and held.batch_id
            and held.state == "authorized"
            and state == "dispatching"
        ):
            raise OperationError(
                f"Operation {operation_id!r} belongs to operation batch "
                f"{held.batch_id!r} and must dispatch through its batch claim"
            )
        refusal = advance_refusal(
            operation_id,
            held.state,
            state,
            artifact if artifact is not None else held.artifact,
        )
        if refusal:
            raise OperationError(refusal)
        return held

    def capture_result(
        self,
        operation_id: str,
        capture: Callable[[], ArtifactReceipt],
    ) -> Operation:
        """Capture one arrived result and journal it under the same lock.

        The callback publishes the operation-bound recovery artifact. Running
        it only after the locked cleanup/state check prevents a durable forget
        decision from being followed by a newly recreated, unbound artifact.
        An already captured result never invokes the callback, but it retries
        terminal-WAL finalization in case the previous process died after the
        receipt became durable.
        """
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            held = current.operations.get(operation_id)
            if held is None or held.cleanup is not None:
                self._require_move_under_lock(
                    current, operation_id, "result_captured"
                )
            assert held is not None
            if held.state == "result_captured":
                self.operations = current.operations
                assert held.artifact is not None
                _finalize_artifact(self.path, held.artifact)
                return held
            self._require_move_under_lock(
                current, operation_id, "result_captured"
            )
            artifact = capture()
            if not isinstance(artifact, ArtifactReceipt):
                raise OperationError(
                    f"Operation {operation_id!r} capture returned no artifact receipt"
                )
            # Re-validate even an in-memory receipt and prove it against the
            # still-terminal operation-bound WAL before making it durable.
            artifact = ArtifactReceipt.from_dict(
                operation_id, artifact.to_dict()
            )
            capture_evidence = _pending_answer_evidence(
                self.path, operation_id
            ).write_ahead
            marker = artifact.terminal_marker
            if (
                marker is None
                or capture_evidence is None
                or capture_evidence.directory_identity
                != artifact.directory_identity
                or capture_evidence.public_answer_snapshot
                != (artifact.entry_state, artifact.content_sha256)
                or capture_evidence.marker_name != marker.name
                or capture_evidence.marker_state != marker.entry_state
                or capture_evidence.marker_revision != marker.content_sha256
            ):
                raise OperationError(
                    f"Operation {operation_id!r} capture returned a receipt "
                    "that is not proven by its terminal write-ahead record"
                )
            moved = self._move_under_lock(
                current,
                operation_id,
                "result_captured",
                artifact=artifact,
            )
            _finalize_artifact(self.path, artifact)
            return moved

    def commit_result(
        self,
        operation_id: str,
        persist: Callable[[], None],
    ) -> Operation:
        """Persist staging and mark its exact answer committed under one lock.

        The caller must acquire every output-path lock before entering this
        method. The callback then runs after the authoritative cleanup/state
        check but before the committed journal write, so cleanup cannot become
        durable in the seam and make staging change under a refused entry.
        """
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            self._require_move_under_lock(current, operation_id, "committed")
            persist()
            return self._move_under_lock(current, operation_id, "committed")

    def read_reply(self, operation_id: str) -> bytes:
        """Return the exact currently bound provider reply without settling it.

        The journal lock makes a durable forget decision and this read order
        themselves wholly before or after one another. The evidence reader may
        use a private WAL name when publication was safely refused; it never
        substitutes or exposes the lexical public occupant.
        """
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            held = current.operations.get(operation_id)
            if held is None:
                raise OperationError(f"No operation {operation_id!r} to read")
            if held.cleanup is not None:
                raise OperationError(
                    f"Operation {operation_id!r} is being forgotten; its "
                    "recovery reply is no longer available to read"
                )
            observation = reply_observation(self.path, held)
            if observation.payload is not None:
                self.operations = current.operations
                return observation.payload
            if observation.recorded:
                raise OperationError(
                    f"Operation {operation_id!r} records a captured reply, but "
                    "its exact recovery bytes are unavailable"
                )
            if observation.interrupted:
                raise OperationError(
                    f"Operation {operation_id!r} has an interrupted answer "
                    "capture, not a complete reply"
                )
            raise OperationError(
                f"Operation {operation_id!r} has no recoverable reply"
            )

    def read_inspectable_reply(
        self,
        operation_id: str,
        *,
        expected_operation: Operation | None = None,
    ) -> bytes:
        """Read a complete reply or a frame-preserving incomplete-stream view.

        Complete artifact bytes pass through unchanged. A streaming operation
        that has no complete artifact instead produces a deterministic JSON
        envelope carrying every exact committed text-frame payload and its
        boundary. This is an inspection surface only: it neither settles the
        operation nor adopts a crash extension whose head is not yet journalled.
        """
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            held = current.operations.get(operation_id)
            if expected_operation is not None and held != expected_operation:
                raise OperationError(
                    f"Operation {operation_id!r} changed after the action was rendered"
                )
            if held is None:
                raise OperationError(f"No operation {operation_id!r} to read")
            if held.cleanup is not None:
                raise OperationError(
                    f"Operation {operation_id!r} is being forgotten; its "
                    "recovery reply is no longer available to read"
                )
            observation = reply_observation(self.path, held)
            if observation.payload is not None:
                self.operations = current.operations
                return observation.payload
            if observation.recorded:
                raise OperationError(
                    f"Operation {operation_id!r} records a captured reply, but "
                    "its exact recovery bytes are unavailable"
                )
            if held.response_spool is not None:
                observed = _read_response_spool_snapshot(
                    self.path, held.response_spool
                )
                if observed is None:
                    raise OperationError("The exact response spool is unavailable")
                frames = _validated_response_frames(
                    held.response_spool, observed[1]
                )
                if len(observed[1]) != held.response_spool.committed_size:
                    raise OperationError(
                        "Response spool recovery must finish before its newly "
                        "durable frame can be inspected"
                    )
                if frames:
                    self.operations = current.operations
                    return _response_spool_inspection_bytes(frames)
            if observation.interrupted:
                raise OperationError(
                    f"Operation {operation_id!r} has an interrupted answer "
                    "capture, not a complete reply"
                )
            raise OperationError(
                f"Operation {operation_id!r} has no recoverable reply or response frames"
            )

    def end(
        self,
        operation_id: str,
        *,
        detail: str = "",
        expected_operation: Operation | None = None,
    ) -> Operation:
        """Say a call that will never finish is over, without inventing what it
        bought.

        The way out of a wedge that was otherwise permanent: a killed process
        leaves an entry saying a call is live, and janki will not spend again
        until something settles it. Nothing can settle it automatically — only
        a person can say the process is gone — so this is that statement.

        Where it lands depends on what is actually known, because a single
        destination would have to lie about one case or the other:

        * `authorized` — nothing was ever sent, so `canceled_before_send`.
          Calling this one an unknown outcome would invent a charge.
        * `dispatching` / `running` — the request left and no complete answer
          was sealed, so `outcome_unknown`. It never says the call failed and
          never says it succeeded. Exact terminal frames already on disk may
          later strengthen that state through `capture_result` without a new
          dispatch.

        A captured artifact is refused: nothing about it is unknown. It is on
        disk, and the choice there is to read it or to discard it deliberately.
        Partial streaming frames stay bound for inspection or exact local
        provider recovery.
        """
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            held = current.operations.get(operation_id)
            if expected_operation is not None and held != expected_operation:
                raise OperationError(
                    f"Operation {operation_id!r} changed after the action was rendered"
                )
            if held is None:
                raise OperationError(f"No operation {operation_id!r} to end")
            if held.cleanup is not None:
                raise OperationError(
                    f"Operation {operation_id!r} is being forgotten; rerun "
                    f"'janki operations --forget {operation_id}' to finish cleanup"
                )
            if held.state in TERMINAL_STATES:
                raise OperationError(
                    f"Operation {operation_id!r} is already {held.state!r} and "
                    "has nothing left to end"
                )
            if held.response_spool is not None:
                # A frame reaches durable storage before its new spool head is
                # journalled. Recover that one exact crash extension before
                # the terminal transition makes provider recovery ineligible.
                # Unreadable evidence is left bound for the explicit forced
                # discard path; it does not make the money outcome knowable.
                held = self._recover_response_spool_head_under_lock(
                    current,
                    held,
                    unavailable_ok=True,
                    invalid_ok=True,
                )
            # A path string is only history. Read through current bound
            # evidence so a private WAL answer remains recoverable without
            # advertising or adopting a missing/replaced public occupant.
            observation, evidence = _observe_reply(self.path, held)
            if observation.readable:
                raise OperationError(
                    f"Operation {operation_id!r} is not unfinished — its reply "
                    "arrived and is recoverable with "
                    f"'janki operations --show-reply {operation_id}'. Read it, then "
                    f"'janki operations --forget {operation_id} --force' to "
                    "drop it."
                )
            if observation.recorded:
                raise OperationError(
                    f"Operation {operation_id!r} records a captured reply, but "
                    "its exact recovery bytes are unavailable. If you accept "
                    "that loss, explicitly discard the record with "
                    f"'janki operations --forget {operation_id} --force'."
                )
            sent = held.state != "authorized"
            settlement_detail = detail or (
                "ended by hand; the process was gone"
                if sent
                else "ended by hand; nothing had been sent"
            )
            if (
                evidence.write_ahead is not None
                and not evidence.reply_complete
            ):
                settlement_detail += (
                    "; answer capture was interrupted; operation-bound "
                    "recovery evidence remains until this entry is forgotten"
                )
            return self._move_under_lock(
                current,
                operation_id,
                "outcome_unknown" if sent else "canceled_before_send",
                detail=settlement_detail,
            )

    def forget(
        self,
        operation_ids: Iterable[str],
        *,
        force: bool = False,
        expected_operation: Operation | None = None,
    ) -> int:
        """Drop finished operations, and only finished ones.

        Any terminal state, not just `committed`: an `outcome_unknown` a person
        has looked at and accepted is finished too, and refusing to drop it was
        how one lost call blocked every later one for ever.

        An entry still holding an answer nobody turned into staging is refused
        without `force`. That artifact is a reply somebody paid for, and this
        deletes it. The exact cleanup decision is journalled before deletion;
        an interrupted retry resumes those bindings without fresh force.
        """
        requested_ids = tuple(sorted({str(value) for value in operation_ids}))
        if expected_operation is not None and requested_ids != (
            expected_operation.operation_id,
        ):
            raise OperationError(
                "A guarded operation forget must target exactly its rendered operation"
            )
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            if expected_operation is not None and current.operations.get(
                expected_operation.operation_id
            ) != expected_operation:
                raise OperationError(
                    f"Operation {expected_operation.operation_id!r} changed after "
                    "the action was rendered"
                )
            cleanup: dict[str, _CleanupIntent] = {}
            newly_bound: set[str] = set()
            for operation_id in requested_ids:
                held = current.operations.get(operation_id)
                if held is None:
                    continue
                if held.cleanup is not None:
                    # The first call made its exact deletion decision durable.
                    # A retry resumes that decision without asking for force
                    # again and without rebinding a same-name replacement.
                    cleanup[operation_id] = held.cleanup
                    continue
                observation, evidence = _observe_reply(self.path, held)
                spool_binding: _ArtifactBinding | None = None
                spool_write_ahead: _BoundWriteEvidence | None = None
                spool_reply_readable = False
                spool_reply_recorded = False
                if held.response_spool is not None:
                    spool_observation = _read_response_spool_snapshot(
                        self.path, held.response_spool
                    )
                    if spool_observation is not None:
                        spool_binding = spool_observation[0]
                        spool_payload = spool_observation[1]
                        # Raw nonempty bytes are evidence even if a torn frame
                        # cannot be decoded. An ordinary forget must preserve
                        # them; force may deliberately retire their exact
                        # binding without pretending they were readable.
                        spool_reply_recorded = bool(spool_payload)
                        try:
                            spool_frames = _validated_response_frames(
                                held.response_spool, spool_payload
                            )
                        except OperationError:
                            spool_frames = ()
                        else:
                            spool_reply_readable = bool(spool_frames)
                            if (
                                held.state != "committed"
                                and len(spool_payload)
                                > held.response_spool.committed_size
                            ):
                                adopted_spool = _response_spool_receipt_at_head(
                                    held.response_spool,
                                    spool_payload,
                                    spool_frames,
                                )
                                held = self._record_response_spool_head_under_lock(
                                    current, held, adopted_spool
                                )
                                _confirm_response_spool_head(
                                    self.path, adopted_spool, spool_payload
                                )
                        spool_reply_recorded = (
                            spool_reply_recorded
                            or held.response_spool.frame_count > 0
                        )
                    else:
                        spool_reply_recorded = (
                            held.response_spool.frame_count > 0
                        )
                    spool_write_ahead = _bound_capture_evidence(
                        _response_spool_bound_file(
                            self.path, held.response_spool
                        )
                    )
                elif _valid_pending_operation_id(operation_id):
                    spool_target = _response_spool_target(
                        self.path, operation_id
                    )
                    try:
                        spool_write_ahead = _bound_write_evidence(spool_target)
                    except DataError as exc:
                        raise OperationError(
                            f"Could not inspect response spool recovery "
                            f"evidence for operation {operation_id!r}: {exc}"
                        ) from exc
                    if spool_write_ahead is not None:
                        public_spool = (
                            spool_write_ahead.public_answer_snapshot
                        )
                        if public_spool is not None:
                            spool_binding = _ArtifactBinding(
                                path=spool_target,
                                directory_identity=(
                                    spool_write_ahead.directory_identity
                                ),
                                entry_state=public_spool[0],
                                content_sha256=public_spool[1],
                            )
                actual_reply = observation.readable
                # `result_captured` too: its answer is on disk, so it is not
                # "unfinished" in the sense `end` means, and refusing it here
                # would leave the only entry holding a real reply with no way
                # out at all.
                if held.state not in TERMINAL_STATES | {"result_captured"} and not (
                    force and (actual_reply or observation.recorded)
                ):
                    if actual_reply:
                        raise OperationError(
                            f"Operation {operation_id!r} still holds a reply, "
                            "which was paid for and never became staging. Read it with "
                            f"'janki operations --show-reply {operation_id}', "
                            "then pass --force to drop it."
                        )
                    if spool_reply_readable:
                        raise OperationError(
                            f"Operation {operation_id!r} is still {held.state!r} "
                            "and holds provider response frames. Even --force "
                            "cannot prove its transport has stopped; first run "
                            f"'janki operations --end {operation_id}'."
                        )
                    if observation.recorded:
                        raise OperationError(
                            f"Operation {operation_id!r} records a captured reply, "
                            "but its exact recovery bytes are unavailable. Pass "
                            "--force only if you accept losing that reply record."
                        )
                    if spool_reply_recorded:
                        raise OperationError(
                            f"Operation {operation_id!r} is still {held.state!r} "
                            "and records provider response frames. Even --force "
                            "cannot prove its transport has stopped; first run "
                            f"'janki operations --end {operation_id}'."
                        )
                    raise OperationError(
                        f"Operation {operation_id!r} is {held.state!r}, which "
                        "is not finished; 'janki operations --end' ends a call "
                        "that will never finish"
                    )
                # Either name: a blob under this operation's own id is its
                # reply too, whether or not the entry ever got to record it.
                # Bind that recovery name exactly once. The same identity
                # decides whether a reply exists and is the only deletion
                # authority retained after the journal entry is removed.
                binding = evidence.artifact
                if (
                    held.money_may_have_been_spent
                    and held.state != "committed"
                    and not force
                ):
                    if actual_reply:
                        raise OperationError(
                            f"Operation {operation_id!r} still holds a reply, "
                            "which was paid for and never became staging. Read it with "
                            f"'janki operations --show-reply {operation_id}', "
                            "then pass --force to drop it."
                        )
                    if spool_reply_readable:
                        raise OperationError(
                            f"Operation {operation_id!r} still holds provider "
                            "response frames, which may contain paid output and "
                            "never became committed audio. Read them with "
                            f"'janki operations --show-reply {operation_id}', "
                            "then pass --force to drop them."
                        )
                    if observation.recorded:
                        raise OperationError(
                            f"Operation {operation_id!r} records a captured reply, "
                            "but its exact recovery bytes are unavailable. Pass "
                            "--force only if you accept losing that reply record."
                        )
                    if spool_reply_recorded:
                        raise OperationError(
                            f"Operation {operation_id!r} records provider "
                            "response frames, but their exact readable bytes are "
                            "unavailable. Pass --force only if you accept losing "
                            "that response record."
                        )
                # Persist the exact bindings before deleting anything.  This
                # is the cleanup tombstone: a crash may leave the entry here,
                # but the retry neither needs fresh force nor gains authority
                # over whatever later occupies one of these names.
                intent = _CleanupIntent(
                    artifact=binding,
                    write_ahead=evidence.write_ahead,
                    response_spool=spool_binding,
                    response_spool_write_ahead=spool_write_ahead,
                    forced=force,
                )
                cleanup[operation_id] = intent
                newly_bound.add(operation_id)

            if newly_bound:
                now = _now()
                for operation_id in newly_bound:
                    held = current.operations[operation_id]
                    current.operations[operation_id] = Operation(
                        operation_id=held.operation_id,
                        kind=held.kind,
                        state=held.state,
                        source_file=held.source_file,
                        source_sha256=held.source_sha256,
                        request_fp=held.request_fp,
                        model=held.model,
                        authorized_at=held.authorized_at,
                        updated_at=now,
                        artifact=held.artifact,
                        response_spool=held.response_spool,
                        detail=held.detail,
                        cleanup=cleanup[operation_id],
                        batch_id=held.batch_id,
                    )
                current.path = self.path
                current._write()
                self.operations = current.operations

            failures: dict[str, list[str]] = {}
            retired: list[str] = []
            for operation_id, intent in cleanup.items():
                errors: list[str] = []
                if intent.artifact is not None:
                    try:
                        _retire_artifact(intent.artifact)
                    except OperationError as exc:
                        errors.append(str(exc))
                if intent.write_ahead is not None:
                    try:
                        _retire_write_ahead(intent.write_ahead)
                    except OperationError as exc:
                        errors.append(str(exc))
                if intent.response_spool is not None:
                    try:
                        _retire_response_spool(intent.response_spool)
                    except OperationError as exc:
                        errors.append(str(exc))
                if intent.response_spool_write_ahead is not None:
                    try:
                        _retire_write_ahead(
                            intent.response_spool_write_ahead
                        )
                    except OperationError as exc:
                        errors.append(str(exc))
                if errors:
                    failures[operation_id] = errors
                else:
                    retired.append(operation_id)

            # Only successful tombstones disappear. A process death before
            # this write leaves all of them durable; cleanup is idempotent, so
            # the same ordinary forget resumes and proves missing exact names
            # as already retired.
            if retired:
                for operation_id in retired:
                    del current.operations[operation_id]
                current.path = self.path
                current._write()
                self.operations = current.operations

            if failures:
                detail = "; ".join(
                    f"{operation_id!r}: {'; '.join(errors)}"
                    for operation_id, errors in failures.items()
                )
                retries = "; ".join(
                    f"janki operations --forget {operation_id}"
                    for operation_id in failures
                )
                raise OperationError(
                    "Operation cleanup remains recorded in the journal; "
                    f"no fresh force decision is needed. {detail}. Retry: "
                    f"{retries}"
                )
            return len(retired)

    # --- reads --------------------------------------------------------------

    def unfinished(self) -> list[Operation]:
        """Live operations, oldest first — what a resumed run has to deal with."""
        return sorted(
            (
                op
                for op in self.operations.values()
                if op.cleanup is None and op.state in LIVE_STATES
            ),
            key=lambda op: (op.authorized_at, op.operation_id),
        )

    def blocking(self) -> list[Operation]:
        """Every operation stopping janki from starting another paid call.

        Wider than `needing_attention`, and it has to be: a call still marked
        in flight needs nobody's attention while it is genuinely running, but
        it is exactly what a person sees "janki will not start another" about.
        Listing only the ones that need a decision answered "why is this
        blocked?" with "nothing is blocked".
        """
        return sorted(
            (op for op in self.operations.values() if op.blocks_spending),
            key=lambda op: (op.authorized_at, op.operation_id),
        )

    def tracked(self) -> list[Operation]:
        """Blocking calls and unfinished recovery cleanup, oldest first.

        A durable cleanup intent means the money decision is settled, so it
        does not belong in :meth:`blocking`. It still needs one exact ordinary
        forget retry and must remain visible until that retirement succeeds.
        A terminal streaming call can likewise leave a bound response spool if
        the process dies just before its ordinary forget; it remains visible
        even when its before-send or committed state blocks no new spending.
        Reply recovery is observed separately at render time. A journal path
        string is historical state, not proof that the bytes are accessible.
        """
        return sorted(
            (
                op
                for op in self.operations.values()
                if op.blocks_spending
                or op.cleanup is not None
                or (
                    (
                        op.artifact is not None
                        or op.response_spool is not None
                    )
                    and op.state in TERMINAL_STATES
                )
            ),
            key=lambda op: (op.authorized_at, op.operation_id),
        )

    def needing_attention(self) -> list[Operation]:
        """Operations a person has to look at before janki spends again."""
        return sorted(
            (op for op in self.operations.values() if op.needs_a_person),
            key=lambda op: (op.authorized_at, op.operation_id),
        )


def cancel_before_send(
    journal_path: Path,
    operation_id: str,
    *,
    error: type[_UnsentError],
    label: str,
    detail: str,
    cause: BaseException,
) -> _UnsentError:
    """Retire a paid identity nothing was sent under, and say what happened.

    Every paid service allocates its operation *before* it publishes the exact
    request bytes it is about to send, so preparation can still fail with an
    authority already on disk. That authority blocks the next call until a
    person clears it, so it has to be retired here — and if retiring it also
    fails, the owner needs both failures in one sentence, because the second
    one is what leaves work for them.

    Returned rather than raised so the caller keeps `raise ... from cause` and
    the original traceback. `provider_dispatched` is always False: reaching
    this means the request never left, which is the fact a refusal has to carry
    for the caller to know its money was not spent.
    """
    try:
        OperationJournal.load(journal_path).advance(
            operation_id,
            "canceled_before_send",
            detail=detail,
        )
    except JankiError as journal_error:
        return error(
            f"{label} {operation_id} was not sent, and its authority could not "
            f"be retired: {detail} {cause}; {journal_error}. Inspect janki "
            "operations before retrying.",
            operation_id=operation_id,
            provider_dispatched=False,
        )
    return error(
        f"{label} {operation_id} was canceled before send: {detail} {cause}",
        operation_id=operation_id,
        provider_dispatched=False,
    )
