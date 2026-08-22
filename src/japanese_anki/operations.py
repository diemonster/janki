"""What janki knows about a paid model call it has not finished.

`docs/DESIGN.md`, *Mechanisms the pipeline rests on*: every paid model call is
journaled durably **before dispatch**, and its exact response is persisted as a
pending artifact **before parsing**. This module is that journal.

The failure it exists for is specific. `extract` sends a private source to a
paid provider and then parses what comes back. Interrupt it between those two
points — a crash, a closed laptop, a killed terminal — and the money is spent
while nothing on disk remembers it. The next run has no way to tell "this was
never sent" from "this was sent and the answer is gone", and the only safe
options are to pay again or to give up. Recording the attempt before making it,
and the answer before trusting it, is what turns that into a resumable state.

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

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.io import atomic_write_text, exclusive_path_lock

__all__ = [
    "LIVE_STATES",
    "STATES",
    "TERMINAL_STATES",
    "Operation",
    "OperationError",
    "OperationJournal",
]


class OperationError(JankiError):
    """A journal transition that is not allowed, or a journal that is unreadable."""


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
    "dispatching": frozenset({"running", "outcome_unknown", "failed_before_send"}),
    "running": frozenset({"result_captured", "outcome_unknown", "failed_before_send"}),
    "result_captured": frozenset({"committed", "outcome_unknown"}),
    # Terminal. Nothing leaves these, and in particular nothing leaves
    # `outcome_unknown` back into a live state: re-dispatching an operation
    # whose provider outcome is unknown is how one authorization becomes two
    # charges.
    "committed": frozenset(),
    "outcome_unknown": frozenset(),
    "failed_before_send": frozenset(),
    "canceled_before_send": frozenset(),
    "expired": frozenset(),
}

STATES: tuple[str, ...] = tuple(_TRANSITIONS)

#: States where no provider call has happened yet, so nothing has been spent.
LIVE_STATES: frozenset[str] = frozenset(
    {"authorized", "dispatching", "running", "result_captured"}
)

TERMINAL_STATES: frozenset[str] = frozenset(
    state for state, moves in _TRANSITIONS.items() if not moves
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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
    #: Relative path of the exact provider answer, once captured.
    artifact: str = ""
    #: Why it stopped, for a terminal state.
    detail: str = ""

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
        return self.state in {"outcome_unknown", "result_captured"}

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
        if self.artifact:
            value["artifact"] = self.artifact
        if self.detail:
            value["detail"] = self.detail
        return value

    @classmethod
    def from_dict(cls, operation_id: str, raw: Mapping[str, Any]) -> Operation:
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
            artifact=str(raw.get("artifact", "")),
            detail=str(raw.get("detail", "")),
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

    @classmethod
    def load(cls, path: Path) -> OperationJournal:
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise OperationError(f"Could not read {path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise OperationError(f"{path} must hold an object")
        entries = raw.get("operations")
        if entries is None:
            entries = {}
        if not isinstance(entries, Mapping):
            raise OperationError(f"{path} operations must be an object")
        return cls(
            path=path,
            operations={
                str(key): Operation.from_dict(str(key), value)
                for key, value in entries.items()
            },
        )

    def _write(self) -> None:
        payload = {
            "version": 1,
            "operations": {
                key: self.operations[key].to_dict() for key in sorted(self.operations)
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        )

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
            if operation_id in current.operations:
                held = current.operations[operation_id]
                raise OperationError(
                    f"Operation {operation_id!r} is already authorized and is "
                    f"{held.state!r}; authority is one-use."
                )
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
            return operation

    def advance(
        self,
        operation_id: str,
        state: str,
        *,
        artifact: str = "",
        detail: str = "",
    ) -> Operation:
        """Move one operation to `state`, or refuse if that move is not legal."""
        if state not in _TRANSITIONS:
            raise OperationError(f"Unknown operation state {state!r}")
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            held = current.operations.get(operation_id)
            if held is None:
                raise OperationError(f"No operation {operation_id!r} to advance")
            if state not in _TRANSITIONS[held.state]:
                raise OperationError(
                    f"Operation {operation_id!r} cannot move from {held.state!r} "
                    f"to {state!r}"
                )
            if state == "committed" and not held.artifact and not artifact:
                # Committing means "the exact answer became staging". Without a
                # captured artifact there is nothing that could have.
                raise OperationError(
                    f"Operation {operation_id!r} cannot be committed without a "
                    "captured provider answer"
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
                artifact=artifact or held.artifact,
                detail=detail or held.detail,
            )
            current.operations[operation_id] = moved
            current.path = self.path
            current._write()
            self.operations = current.operations
            return moved

    def forget(self, operation_ids: Iterable[str]) -> int:
        """Drop committed operations. Only committed ones: everything else is
        either live or a record of money whose outcome someone still needs."""
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            removed = 0
            for operation_id in {str(value) for value in operation_ids}:
                held = current.operations.get(operation_id)
                if held is None:
                    continue
                if held.state != "committed":
                    raise OperationError(
                        f"Operation {operation_id!r} is {held.state!r}, not "
                        "committed; only a committed operation may be forgotten"
                    )
                del current.operations[operation_id]
                removed += 1
            if removed:
                current.path = self.path
                current._write()
                self.operations = current.operations
            return removed

    # --- reads --------------------------------------------------------------

    def unfinished(self) -> list[Operation]:
        """Live operations, oldest first — what a resumed run has to deal with."""
        return sorted(
            (op for op in self.operations.values() if op.state in LIVE_STATES),
            key=lambda op: (op.authorized_at, op.operation_id),
        )

    def needing_attention(self) -> list[Operation]:
        """Operations a person has to look at before janki spends again."""
        return sorted(
            (op for op in self.operations.values() if op.needs_a_person),
            key=lambda op: (op.authorized_at, op.operation_id),
        )
