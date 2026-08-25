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

import contextlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.io import (
    atomic_write_bytes,
    atomic_write_text,
    exclusive_path_lock,
)

__all__ = [
    "BLOCKS_SPENDING",
    "IN_FLIGHT",
    "LIVE_STATES",
    "STATES",
    "TERMINAL_STATES",
    "Operation",
    "OperationError",
    "OperationJournal",
    "advance_refusal",
    "answer_text",
    "capture_artifact",
    "serialize_response",
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
    # `result_captured` directly: a caller that cannot observe streaming
    # never truthfully passes through `running`, and inventing the state
    # would put a moment in the journal that nobody witnessed.
    "dispatching": frozenset(
        {"running", "result_captured", "outcome_unknown", "failed_before_send"}
    ),
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

#: States that mean a paid call is in flight right now. `authorized` is not
#: one of them: it says the authority was written and nothing was sent, which
#: is where a process that died before dispatching leaves an entry — and
#: counting it would wedge every later run behind a call that never happened.
IN_FLIGHT: frozenset[str] = frozenset({"dispatching", "running"})

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


def advance_refusal(
    operation_id: str, state: str, to: str, artifact: str = ""
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
        if callable(method):
            try:
                value = method()
            except Exception:  # noqa: BLE001 - never lose a paid answer
                continue
            return value.encode("utf-8") if isinstance(value, str) else bytes(value)
    try:
        return json.dumps(response, ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return repr(response).encode("utf-8")


def answer_text(journal_path: Path, operation: Operation) -> str:
    """The answer inside a captured artifact, or empty when it holds none.

    A captured response is not the same thing as a captured *answer*. A call
    that reaches `max_tokens` while still reasoning returns a thinking block
    and no text at all — the reply is real, and paid for, and contains nothing
    to recover. Telling someone their answer was saved in that case sends them
    looking for cards in a file that has none.
    """
    if not operation.artifact:
        return ""
    path = Path(journal_path).parent / operation.artifact
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(payload, Mapping):
        return ""
    blocks = payload.get("content")
    if not isinstance(blocks, list):
        return ""
    return "".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "text"
    )


def capture_artifact(journal_path: Path, operation_id: str, payload: bytes) -> str:
    """Write the exact provider answer beside the journal, and name it.

    Returns the path relative to the journal's directory, which is what the
    entry stores: an absolute path is wrong on any other clone of a repository
    whose whole point is being portable.

    The bytes are written before the journal entry that references them moves
    to `result_captured`. That ordering is the guarantee — a crash between the
    two leaves an unreferenced file, which is litter, rather than an entry
    pointing at an answer that was never written, which is a lie.
    """
    directory = Path(journal_path).parent / ".pending"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{operation_id}.json"
    atomic_write_bytes(target, payload)
    return f".pending/{target.name}"


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
            blocking = sorted(
                (
                    op
                    for op in current.operations.values()
                    if op.state in BLOCKS_SPENDING
                ),
                key=lambda op: (op.authorized_at, op.operation_id),
            )
            if blocking:
                first = blocking[0]
                raise OperationError(
                    f"Operation {first.operation_id!r} for {first.source_file} "
                    f"is {first.state!r} and janki will not start another paid "
                    "call until it is settled. 'janki operations' shows it; "
                    "'janki operations --end' ends a call that will never "
                    "finish."
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
            refusal = advance_refusal(
                operation_id, held.state, state, artifact or held.artifact
            )
            if refusal:
                raise OperationError(refusal)
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

    def end(self, operation_id: str, *, detail: str = "") -> Operation:
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
        * `dispatching` / `running` — the request left and no answer came
          back, so `outcome_unknown`. It never says the call failed and never
          says it succeeded.

        A captured answer is refused: nothing about it is unknown. It is on
        disk, and the choice there is to read it or to discard it deliberately.
        """
        with exclusive_path_lock(self.path):
            held = OperationJournal.load(self.path).operations.get(operation_id)
            if held is None:
                raise OperationError(f"No operation {operation_id!r} to end")
            if held.state in TERMINAL_STATES:
                raise OperationError(
                    f"Operation {operation_id!r} is already {held.state!r} and "
                    "has nothing left to end"
                )
            if held.state == "result_captured":
                raise OperationError(
                    f"Operation {operation_id!r} is not unfinished — its reply "
                    f"arrived and is saved at {held.artifact}. Read it, then "
                    "'janki operations --forget --force' to drop it."
                )
            sent = held.state != "authorized"
        return self.advance(
            operation_id,
            "outcome_unknown" if sent else "canceled_before_send",
            detail=detail
            or (
                "ended by hand; the process was gone"
                if sent
                else "ended by hand; nothing had been sent"
            ),
        )

    def forget(self, operation_ids: Iterable[str], *, force: bool = False) -> int:
        """Drop finished operations, and only finished ones.

        Any terminal state, not just `committed`: an `outcome_unknown` a person
        has looked at and accepted is finished too, and refusing to drop it was
        how one lost call blocked every later one for ever.

        An entry still holding an answer nobody turned into staging is refused
        without `force`. That artifact is a reply somebody paid for, and this
        deletes it.
        """
        with exclusive_path_lock(self.path):
            current = OperationJournal.load(self.path)
            removed = 0
            for operation_id in {str(value) for value in operation_ids}:
                held = current.operations.get(operation_id)
                if held is None:
                    continue
                # `result_captured` too: its answer is on disk, so it is not
                # "unfinished" in the sense `end` means, and refusing it here
                # would leave the only entry holding a real reply with no way
                # out at all.
                if held.state not in TERMINAL_STATES | {"result_captured"}:
                    raise OperationError(
                        f"Operation {operation_id!r} is {held.state!r}, which "
                        "is not finished; 'janki operations --end' ends a call "
                        "that will never finish"
                    )
                if held.artifact and held.state != "committed" and not force:
                    raise OperationError(
                        f"Operation {operation_id!r} still holds the reply at "
                        f"{held.artifact}, which was paid for and never became "
                        "staging. Read it first, then pass --force to drop it."
                    )
                # The artifact goes with the entry. It is a recovery buffer,
                # and once the answer has become staging the archive under
                # `data/staging/done/` is the durable copy — leaving the blob
                # behind would accumulate an unreferenced megabyte per paid
                # call in a repository whose whole point is being portable.
                if held.artifact:
                    blob = Path(self.path).parent / held.artifact
                    with contextlib.suppress(OSError):
                        blob.unlink()
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

    def blocking(self) -> list[Operation]:
        """Every operation stopping janki from starting another paid call.

        Wider than `needing_attention`, and it has to be: a call still marked
        in flight needs nobody's attention while it is genuinely running, but
        it is exactly what a person sees "janki will not start another" about.
        Listing only the ones that need a decision answered "why is this
        blocked?" with "nothing is blocked".
        """
        return sorted(
            (op for op in self.operations.values() if op.state in BLOCKS_SPENDING),
            key=lambda op: (op.authorized_at, op.operation_id),
        )

    def needing_attention(self) -> list[Operation]:
        """Operations a person has to look at before janki spends again."""
        return sorted(
            (op for op in self.operations.values() if op.needs_a_person),
            key=lambda op: (op.authorized_at, op.operation_id),
        )
