"""Reading an answer out of a reply that was captured but never decoded.

A paid extraction is journalled before it is sent and its exact reply is
written to disk before anything parses it, so a reply the ordinary decoder
refuses is still an answer somebody was billed for. Today that decoder reads
one place — ``structured_output`` on the single successful result frame — and a
capture where the model put its answer somewhere else is money on the floor.

This module reads the *other* places, and only those. Four shapes are admitted,
listed in ``ENVELOPE_SHAPES`` below, and nothing else: no unknown-field
dropping, no stray-token deletion, no filling in a missing fact, and no parser
exception added per observed bad output. An extra property or an absent
``source_kind`` is a refusal naming the exact pointer, because the captured
reply is immutable evidence of what was paid for and a record built partly from
something a person typed would no longer describe it. **The template is the
first remedy**: the three extraction prompts ask for the bare schema object, and
that paragraph is what stops these shapes being produced at all.

Nothing here dispatches, redispatches, authorizes or bills. Recovering an
already-captured reply makes no new call and keeps its existing operation; a
replacement request needs a fresh confirmation and a fresh operation id, which
is ``extract-batch retry``'s job and not this module's.

**Reading is read-only.** :func:`inspect_capture_proposals` changes no journal
entry, writes no staging, reads no prompt file, probes no login and returns no
reply, candidate or source bytes — pointers, hashes and verdicts only, so an
inspection can never become an unmanifested disclosure.

**What ``frame_index`` counts.** A Claude Code capture is newline-delimited
stream frames, and ``frame_index`` is the 0-based index of the frame among the
capture's non-blank lines, in order — the same frames the ordinary decoder
walks. A capture that is one whole JSON document rather than a stream is frame
0. The value is written into staging metadata and typed on the command line, so
it is defined here rather than left to a reader.

**Which transports this can reach.** Only a capture that saved the request
beside its reply can be recovered at all, on this path or the ordinary one, so
these four shapes describe the subscription transport by construction. An
``anthropic-api`` capture has no saved provider manifest and refuses here
exactly as it already refuses in ``recover_extraction_from_capture``; there is
no fifth shape for it.
"""

from __future__ import annotations

import functools
import hashlib
import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from japanese_anki import card_preview, claude_client, extract, inputs, operations
from japanese_anki.application import extraction, extraction_batch
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.io import load_records

__all__ = [
    "ENVELOPE_SHAPES",
    "STRUCTURED_OUTPUT_TOOL",
    "VALID",
    "CaptureProposalGroup",
    "CaptureProposalLocation",
    "CaptureProposalSelection",
    "CaptureRecoveryError",
    "CaptureRecoveryPlan",
    "EnvelopeShape",
    "inspect_capture_proposals",
    "render_capture_proposals",
    "stage_capture_proposal",
]


class CaptureRecoveryError(JankiError):
    """A captured reply that cannot be read into staging, and exactly why."""


#: The tool the subscription CLI calls to return a ``--json-schema`` answer.
#: The same request disables every other tool, so a ``tool_use`` block naming
#: anything else is not this answer and is never read as one.
STRUCTURED_OUTPUT_TOOL = "StructuredOutput"

EnvelopeShape = Literal[
    "result_structured_output",
    "tool_use_input",
    "tool_use_input_wrapped",
    "tool_use_input_json_string",
]

#: The closed list, in the order the contract enumerates it.
ENVELOPE_SHAPES: tuple[EnvelopeShape, ...] = (
    # 1. ``structured_output`` on the single successful result frame — the one
    #    place the ordinary decoder already reads.
    "result_structured_output",
    # 2. the structured-output tool's ``input``, validating directly.
    "tool_use_input",
    # 3. shape 2 where ``input`` is an object whose *sole* key is ``input``.
    "tool_use_input_wrapped",
    # 4. shape 2 where ``input`` is a JSON string that parses totally and
    #    exactly to a validating object.
    "tool_use_input_json_string",
)

#: The verdict a proposal earns by validating against the current response
#: contract. Every other verdict is the exact refusal text.
VALID = "valid"

#: How a staged recovery says who chose the envelope it was read from.
SELECTED_BY_OWNER = "repository-owner"
SELECTED_BY_SOLE_PROPOSAL = "sole-valid-proposal"
SELECTED_BY_TERMINAL = "successful-terminal"

#: A result frame's structured output is not inside a content block, so it has
#: no block index and no tool-use id. Recorded as this rather than omitted, so
#: the location tuple has the same shape everywhere.
NO_BLOCK = -1


@dataclass(frozen=True, slots=True)
class CaptureProposalLocation:
    """Where in one capture a proposal was found. Never a content hash alone."""

    frame_index: int
    block_index: int
    tool_use_id: str
    #: RFC 6901. Every token here is a fixed name or an integer, so none of
    #: them needs the ``~0``/``~1`` escapes.
    json_pointer: str
    envelope_shape: EnvelopeShape

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "block_index": self.block_index,
            "tool_use_id": self.tool_use_id,
            "json_pointer": self.json_pointer,
            "envelope_shape": self.envelope_shape,
        }


@dataclass(frozen=True, slots=True)
class CaptureProposalGroup:
    """One proposed answer, and every place in the capture it appears.

    Grouped by content hash because two structurally identical proposals *are*
    the same answer — and kept as a list of locations because a hash cannot say
    which block it came from, and provenance that cannot say that is not
    provenance. A group is never collapsed to a representative.
    """

    proposal_sha256: str
    schema_verdict: str
    locations: tuple[CaptureProposalLocation, ...]

    @property
    def valid(self) -> bool:
        return self.schema_verdict == VALID


@dataclass(frozen=True, slots=True)
class CaptureRecoveryPlan:
    """What one captured reply holds: pointers, hashes and verdicts only."""

    operation_id: str
    operation_state: str
    request_fingerprint: str
    capture_sha256: str
    response_schema_fingerprint: str
    source_sha256: str
    staging_path: Path
    patterns_path: Path
    destination_deck_sha256: str
    groups: tuple[CaptureProposalGroup, ...]
    valid_group_count: int

    @property
    def valid_groups(self) -> tuple[CaptureProposalGroup, ...]:
        return tuple(group for group in self.groups if group.valid)

    @property
    def valid_location_count(self) -> int:
        return sum(len(group.locations) for group in self.valid_groups)

    @property
    def authoritative_location(self) -> CaptureProposalLocation | None:
        """The valid successful terminal, when this capture holds one.

        A capture that settled normally is authoritative: its answer is what
        the call returned, and an earlier tool argument that differs is an
        earlier draft rather than a competing answer. The reader never prefers
        the latest or the largest.
        """
        for group in self.valid_groups:
            for location in group.locations:
                if location.envelope_shape == "result_structured_output":
                    return location
        return None

    def group_for(self, location: CaptureProposalLocation) -> CaptureProposalGroup:
        for group in self.groups:
            if location in group.locations:
                return group
        raise CaptureRecoveryError(
            f"{location.json_pointer} is not a proposal in this capture."
        )

    def select(
        self, proposal_sha256: str, *, json_pointer: str | None = None
    ) -> CaptureProposalSelection:
        """The owner's one-use binding to an exact proposal in this capture.

        Naming the content selects the group and records every location in it.
        Naming a pointer as well narrows the selection to that one location;
        the group's other locations are still recorded, because they are part
        of what this capture says.
        """
        chosen = [
            group for group in self.groups
            if group.proposal_sha256 == proposal_sha256
        ]
        if not chosen:
            available = ", ".join(group.proposal_sha256 for group in self.groups)
            raise CaptureRecoveryError(
                f"This capture holds no proposal {proposal_sha256!r}. It holds: "
                f"{available or 'none'}."
            )
        group = chosen[0]
        if json_pointer is None:
            location = group.locations[0]
        else:
            found = [
                item for item in group.locations if item.json_pointer == json_pointer
            ]
            if not found:
                pointers = ", ".join(
                    item.json_pointer for item in group.locations
                )
                raise CaptureRecoveryError(
                    f"Proposal {proposal_sha256} is not at {json_pointer!r}. "
                    f"It is at: {pointers}."
                )
            location = found[0]
        return CaptureProposalSelection(
            operation_id=self.operation_id,
            capture_sha256=self.capture_sha256,
            response_schema_fingerprint=self.response_schema_fingerprint,
            frame_index=location.frame_index,
            block_index=location.block_index,
            tool_use_id=location.tool_use_id,
            json_pointer=location.json_pointer,
            proposal_sha256=group.proposal_sha256,
        )


@dataclass(frozen=True, slots=True)
class CaptureProposalSelection:
    """The exact proposal an owner chose, bound one-use to where it came from.

    Every component is compared before anything is staged. No part of it may
    be model-emittable: choosing between competing paid proposals is the
    owner's decision, made against a local rendering of the real cards.
    """

    operation_id: str
    capture_sha256: str
    response_schema_fingerprint: str
    frame_index: int
    block_index: int
    tool_use_id: str
    json_pointer: str
    proposal_sha256: str


# --- reading the capture --------------------------------------------------------


@functools.cache
def _pydantic() -> Any:
    try:
        import pydantic
    except ImportError as exc:  # pragma: no cover - installation boundary
        raise CaptureRecoveryError(
            "Reading a captured extraction needs janki's AI support. Install "
            "it with: pip install -e '.[ai]'"
        ) from exc
    return pydantic


@functools.cache
def _validator() -> Any:
    """One adapter for the current response contract, built once."""
    return _pydantic().TypeAdapter(extract.candidate_schema())


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _Proposal:
    """One location, the exact value found there, and what validation said."""

    location: CaptureProposalLocation
    value: Any
    proposal_sha256: str
    verdict: str
    parsed: Any | None


@dataclass(frozen=True, slots=True)
class _Capture:
    """Everything one read of a captured reply produced, including its bytes."""

    plan: CaptureRecoveryPlan
    batch: extraction_batch.ExtractionBatchPlan
    child: extraction_batch.ExtractionBatchChild
    proposals: tuple[_Proposal, ...]
    #: How many ``tool_use`` blocks named some other tool. Not proposals, and
    #: not silently dropped either: a refusal says how many were passed over.
    other_tool_blocks: int


def _frames(raw: bytes, *, operation_id: str) -> tuple[Any, ...]:
    """Every frame in one capture, parsed totally or not at all.

    A frame that is not complete JSON, or bytes that are not UTF-8, is a reply
    janki will not read rather than a reply it reads partially: salvaging the
    frames around a broken one would be exactly the stray-token deletion this
    module refuses to do.
    """
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CaptureRecoveryError(
            f"The captured reply for operation {operation_id} is not valid "
            f"UTF-8 ({exc}), so janki will not read it. The reply is preserved "
            "exactly."
        ) from exc
    lines = [line for line in text.split("\n") if line.strip()]
    frames: list[Any] = []
    for index, line in enumerate(lines):
        try:
            frames.append(json.loads(line))
        except ValueError as broken:
            # One whole JSON document rather than a stream: frame 0.
            try:
                return (json.loads(text),)
            except ValueError:
                raise CaptureRecoveryError(
                    f"Frame {index} of the captured reply for operation "
                    f"{operation_id} is not complete JSON ({broken}), so janki "
                    "will not read it. The reply is preserved exactly."
                ) from broken
    return tuple(frames)


def _schema_verdict(error: Any) -> str:
    """Where a validation failure is and what kind it is. Never its bytes.

    ``str(ValidationError)`` renders the offending value into every line, and
    a candidate's ``context`` is verbatim source text. A verdict becomes a
    field of the plan §5.1 designs as the model-safe projection and a line of
    the refusal an owner reads, so it is built from the structural fields
    pydantic reports separately — location, message and error type — with the
    input, the surrounding context and the documentation URL excluded.
    """
    parts = []
    for entry in error.errors(
        include_url=False, include_input=False, include_context=False
    ):
        where = "/".join(str(token) for token in entry["loc"]) or "the answer"
        parts.append(f"{where}: {entry['msg']} [{entry['type']}]")
    return "; ".join(parts)


def _validate(value: Any) -> tuple[Any | None, str]:
    """Validate one candidate object against the current response contract."""
    if not isinstance(value, Mapping):
        return None, f"the argument is {type(value).__name__}, not one JSON object"
    try:
        return _validator().validate_python(value), VALID
    except _pydantic().ValidationError as exc:
        return None, _schema_verdict(exc)
    except Exception as exc:  # noqa: BLE001 - anything else the adapter raises
        # The type only. An unexpected failure still refuses with a readable
        # verdict, and still without quoting the argument that caused it.
        return None, f"validating the argument raised {type(exc).__name__}"


def _proposal(
    location: CaptureProposalLocation, value: Any, verdict: str | None = None
) -> _Proposal:
    parsed, decided = (None, verdict) if verdict else _validate(value)
    if decided != VALID:
        decided = f"{location.json_pointer}: {decided}"
    return _Proposal(
        location=location,
        value=value,
        proposal_sha256=_content_hash(value),
        verdict=decided,
        parsed=parsed,
    )


def _result_frame_proposals(
    frames: Sequence[Any], *, operation_id: str
) -> list[_Proposal]:
    results = [
        (index, frame)
        for index, frame in enumerate(frames)
        if isinstance(frame, Mapping) and frame.get("type") == "result"
    ]
    if len(results) > 1:
        raise CaptureRecoveryError(
            f"The captured reply for operation {operation_id} carries "
            f"{len(results)} result frames; exactly one settles a call, so "
            "janki will not attribute an answer to it."
        )
    if not results:
        return []
    index, frame = results[0]
    if frame.get("is_error") is not False or frame.get("subtype") != "success":
        return []
    if "structured_output" not in frame:
        return []
    return [
        _proposal(
            CaptureProposalLocation(
                frame_index=index,
                block_index=NO_BLOCK,
                tool_use_id="",
                json_pointer="/structured_output",
                envelope_shape="result_structured_output",
            ),
            frame["structured_output"],
        )
    ]


def _tool_block_proposal(
    frame_index: int, block_index: int, block: Mapping[str, Any]
) -> _Proposal:
    tool_use_id = str(block.get("id") or "")
    base = f"/message/content/{block_index}/input"
    argument = block.get("input")

    def at(pointer: str, shape: EnvelopeShape) -> CaptureProposalLocation:
        return CaptureProposalLocation(
            frame_index=frame_index,
            block_index=block_index,
            tool_use_id=tool_use_id,
            json_pointer=pointer,
            envelope_shape=shape,
        )

    if isinstance(argument, str):
        # Shape 4. A total, exact parse: a valid prefix followed by anything
        # else is not this answer, so no partial decoder is used here.
        try:
            decoded = json.loads(argument)
        except ValueError as exc:
            return _proposal(
                at(base, "tool_use_input_json_string"),
                argument,
                verdict=f"the argument is a string that is not one JSON value ({exc})",
            )
        return _proposal(at(base, "tool_use_input_json_string"), decoded)
    if isinstance(argument, Mapping) and set(argument) == {"input"}:
        # Shape 3, and only for the sole key ``input``: any other key set is a
        # different object and is validated as itself.
        return _proposal(at(f"{base}/input", "tool_use_input_wrapped"), argument["input"])
    return _proposal(at(base, "tool_use_input"), argument)


def _tool_use_proposals(frames: Sequence[Any]) -> tuple[list[_Proposal], int]:
    """Proposals from assembled assistant messages, and other tools passed over.

    Only an assembled ``assistant`` frame counts. The partial
    ``content_block_delta`` frames before it are the argument arriving, not an
    argument, and reading one would be reconstructing an answer rather than
    finding it.
    """
    found: list[_Proposal] = []
    others = 0
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("type") != "assistant":
            continue
        message = frame.get("message")
        if not isinstance(message, Mapping):
            continue
        blocks = message.get("content")
        if not isinstance(blocks, list):
            continue
        for block_index, block in enumerate(blocks):
            if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                continue
            if block.get("name") != STRUCTURED_OUTPUT_TOOL:
                others += 1
                continue
            found.append(_tool_block_proposal(frame_index, block_index, block))
    return found, others


def _grouped(proposals: Sequence[_Proposal]) -> tuple[CaptureProposalGroup, ...]:
    order = sorted(
        proposals,
        key=lambda item: (
            item.location.frame_index,
            item.location.block_index,
            item.location.json_pointer,
        ),
    )
    groups: dict[str, list[_Proposal]] = {}
    for proposal in order:
        groups.setdefault(proposal.proposal_sha256, []).append(proposal)
    return tuple(
        CaptureProposalGroup(
            proposal_sha256=digest,
            schema_verdict=members[0].verdict,
            locations=tuple(member.location for member in members),
        )
        for digest, members in groups.items()
    )


_INSPECTABLE_STATES = ("result_captured", "committed")
_LIVE_STATES = ("authorized", "dispatching", "running")


def _require_inspectable(operation_id: str, state: str) -> None:
    if state in _INSPECTABLE_STATES:
        return
    if state in _LIVE_STATES:
        raise CaptureRecoveryError(
            f"Operation {operation_id} is still {state}: janki does not know "
            "whether that call has finished, and only a person can say a "
            "vanished process is gone. Settle it with "
            "`janki operations --end` first."
        )
    if state == "outcome_unknown":
        raise CaptureRecoveryError(
            f"Operation {operation_id} ended with an unknown outcome. Its "
            "evidence is never sent again automatically, and strengthening it "
            "to a captured result remains `janki operations --end`'s decision "
            "over the frames already on disk."
        )
    raise CaptureRecoveryError(
        f"Operation {operation_id} is {state}, so it has no captured reply to "
        "recover."
    )


def _read_capture(config: ProjectConfig, operation_id: str) -> _Capture:
    """One offline read of a captured reply. Writes nothing, settles nothing."""
    batch, child = extraction_batch.find_capture_child(config, operation_id)
    journal = operations.OperationJournal.load(config.operations_file)
    entry = journal.operations.get(operation_id)
    if entry is None:
        raise CaptureRecoveryError(
            f"No operation {operation_id!r} is recorded, so there is nothing "
            "to recover."
        )
    _require_inspectable(operation_id, entry.state)

    try:
        payload = journal.read_reply(operation_id)
    except JankiError as exc:
        raise CaptureRecoveryError(str(exc)) from exc
    raw, provenance = extraction.extraction_capture_parts(payload)
    if provenance is None:
        raise CaptureRecoveryError(
            f"The captured reply for {child.source.name} has no saved request "
            "beside it, so it cannot be recovered on its own."
        )
    # The same purely-local rebuild the ordinary recovery uses. It refuses a
    # capture whose response contract has changed since the call, and both
    # halves stay exactly as they were.
    extraction.provider_plan_from_provenance(provenance)

    frames = _frames(raw, operation_id=operation_id)
    proposals = _result_frame_proposals(frames, operation_id=operation_id)
    from_tools, others = _tool_use_proposals(frames)
    proposals.extend(from_tools)
    groups = _grouped(proposals)
    expectation = child.expectation
    plan = CaptureRecoveryPlan(
        operation_id=operation_id,
        operation_state=entry.state,
        request_fingerprint=child.request_fingerprint,
        capture_sha256=hashlib.sha256(payload).hexdigest(),
        response_schema_fingerprint=str(
            provenance.get("response_schema_fingerprint") or ""
        ),
        source_sha256=child.source_sha256,
        staging_path=expectation.staging_path,
        patterns_path=expectation.patterns_path,
        destination_deck_sha256=expectation.destination_deck_sha256,
        groups=groups,
        valid_group_count=sum(1 for group in groups if group.valid),
    )
    return _Capture(
        plan=plan,
        batch=batch,
        child=child,
        proposals=tuple(proposals),
        other_tool_blocks=others,
    )


def inspect_capture_proposals(
    config: ProjectConfig, operation_id: str
) -> CaptureRecoveryPlan:
    """What one captured reply holds, without disclosing any of its bytes.

    Pure and offline: no journal change, no staging, no paid call, no login
    probe, no prompt read. ``result_captured`` is the state this exists for;
    ``committed`` inspects read-only for diagnostics.
    """
    return _read_capture(config, operation_id).plan


# --- staging one of them --------------------------------------------------------


def _no_valid_proposal(capture: _Capture) -> CaptureRecoveryError:
    """The diagnostic, and the one thing an owner may do about it.

    Nobody hand-supplies a missing schema field: a staged record built partly
    from typed text would no longer describe the answer the journal says was
    billed. The settlement is a freshly authorized retry with a fresh operation
    id, which is a new decision about money rather than an edit to evidence.
    """
    lines = [
        f"The captured reply for {capture.child.source.name} holds no proposal "
        "janki can read under the current response contract. Nothing was "
        "written, and the request and the reply are preserved exactly."
    ]
    for group in capture.plan.groups:
        lines.append(f"  {group.schema_verdict}")
    if capture.other_tool_blocks:
        lines.append(
            f"  {capture.other_tool_blocks} tool call(s) in this reply are not "
            f"a call to the structured-output tool ({STRUCTURED_OUTPUT_TOOL}), "
            "so none of them is this answer."
        )
    lines.append(
        "A missing or malformed field cannot be supplied by hand. Ask for the "
        "source again with a fresh, separately authorized call: "
        f"`janki extract-batch retry {capture.batch.batch_id} --children "
        f"{capture.child.index}`, which retires this evidence by that same "
        "decision and mints a new operation id."
    )
    return CaptureRecoveryError("\n".join(lines))


def _validated_selection(
    plan: CaptureRecoveryPlan, selection: CaptureProposalSelection
) -> tuple[CaptureProposalLocation, CaptureProposalGroup]:
    """The exact readable location an owner's tuple names, or a refusal.

    All eight components are compared, and the tuple the owner supplied is the
    one compared — nothing here re-mints a selection from part of it and then
    checks that against itself. Matching content is not matching provenance:
    two identical proposals in one capture hash the same, so the frame, block,
    tool-use id and pointer are compared too.
    """
    if selection.operation_id != plan.operation_id:
        raise CaptureRecoveryError(
            f"That selection was made for operation {selection.operation_id}, "
            f"not {plan.operation_id}."
        )
    if selection.capture_sha256 != plan.capture_sha256:
        raise CaptureRecoveryError(
            "That selection was made against a different captured reply than "
            "the one this operation holds."
        )
    if selection.response_schema_fingerprint != plan.response_schema_fingerprint:
        raise CaptureRecoveryError(
            "That selection was made under a different response contract than "
            "this capture records."
        )
    for group in plan.valid_groups:
        for location in group.locations:
            if (
                group.proposal_sha256 == selection.proposal_sha256
                and location.frame_index == selection.frame_index
                and location.block_index == selection.block_index
                and location.tool_use_id == selection.tool_use_id
                and location.json_pointer == selection.json_pointer
            ):
                return location, group
    raise CaptureRecoveryError(
        "This capture holds no readable proposal at that exact location. A "
        "content hash alone is not a location: name the frame, block, "
        "tool-use id and pointer the inspection printed."
    )


def _resolve_selection(
    capture: _Capture, selection: CaptureProposalSelection | None
) -> tuple[CaptureProposalLocation, CaptureProposalGroup, str]:
    """Which proposal is being staged, and on whose authority it was chosen."""
    plan = capture.plan
    if not plan.valid_groups:
        raise _no_valid_proposal(capture)

    authoritative = plan.authoritative_location
    if authoritative is not None:
        group = plan.group_for(authoritative)
        if selection is not None:
            # Groups are keyed on content, and a settled Claude Code capture
            # holds the terminal's own answer twice — once as the assembled
            # tool argument and once as ``structured_output``. Naming either
            # location of the terminal's group is naming the terminal, so the
            # comparison is on the group rather than on the pointer. Only a
            # *different* group is the earlier draft §5.3 refuses.
            _location, chosen = _validated_selection(plan, selection)
            if chosen.proposal_sha256 != group.proposal_sha256:
                raise CaptureRecoveryError(
                    f"The captured reply for {capture.child.source.name} "
                    "settled normally, so its successful terminal answer is "
                    "authoritative and janki will not stage a different tool "
                    "argument from the same reply."
                )
        return authoritative, group, SELECTED_BY_TERMINAL

    if selection is None:
        if plan.valid_location_count != 1:
            raise CaptureRecoveryError(
                f"The captured reply for {capture.child.source.name} holds "
                f"{plan.valid_location_count} readable proposals, so which one "
                "is the answer is the owner's decision. Look at them with "
                "`janki extract-batch render-proposals` and name one with "
                "`--proposal SHA256 [--at POINTER]`. Consenting in advance "
                "cannot choose between them."
            )
        group = plan.valid_groups[0]
        return group.locations[0], group, SELECTED_BY_SOLE_PROPOSAL

    location, group = _validated_selection(plan, selection)
    return location, group, SELECTED_BY_OWNER


def _recovery_block(
    plan: CaptureRecoveryPlan,
    location: CaptureProposalLocation,
    group: CaptureProposalGroup,
    selected_by: str,
) -> dict[str, Any]:
    """The provenance a salvaged answer carries into its staging document."""
    return {
        "capture_sha256": plan.capture_sha256,
        "proposal_sha256": group.proposal_sha256,
        "envelope_shape": location.envelope_shape,
        "frame_index": location.frame_index,
        "block_index": location.block_index,
        "tool_use_id": location.tool_use_id,
        "json_pointer": location.json_pointer,
        # Every location in the selected group, so provenance never claims one
        # block when the capture held the same answer in two.
        "locations": [item.to_dict() for item in group.locations],
        "selected_by": selected_by,
        "selected_at": date.today().isoformat(),
    }


def _parsed_for(capture: _Capture, location: CaptureProposalLocation) -> Any:
    for proposal in capture.proposals:
        if proposal.location == location:
            return proposal.parsed
    raise CaptureRecoveryError(  # pragma: no cover - locations come from here
        f"{location.json_pointer} is not a proposal in this capture."
    )


def _result_from(
    capture: _Capture,
    binding: extraction_batch.CapturedChildBinding,
    location: CaptureProposalLocation,
) -> extract.ExtractionResult:
    """The normalized answer one chosen envelope yields.

    The terminal shape goes through the ordinary recovery so a capture that
    settled normally is read by exactly the code that reads every other settled
    capture. The salvaged shapes rejoin at ``extraction_result_from_call``,
    which is where all three transports already meet: there is one set of rules
    about what a complete extraction is, and this is not a second one.
    """
    if location.envelope_shape == "result_structured_output":
        return extraction.recover_extraction_from_capture(
            binding.payload, source_name=binding.target.name, mode=binding.mode
        )
    return extract.extraction_result_from_call(
        claude_client.CallResult(_parsed_for(capture, location), "end_turn", None),
        binding.target.item,
        model=binding.model,
        mode=binding.mode,
        provenance=dict(binding.provenance),
    )


def stage_capture_proposal(
    config: ProjectConfig,
    operation_id: str,
    selection: CaptureProposalSelection | None = None,
) -> extraction.ExtractionOutcome:
    """Write one already-paid-for captured answer to its own staging file.

    This is the operation's own answer reaching the durable destination its own
    confirmed call was for, so it needs no new gate — and it revalidates every
    binding the ordinary batch recovery revalidates before writing anything.
    ``result_captured → committed`` is a legal advance; no new writing path is
    created and no paid redispatch hides in here.

    ``selection`` is the owner's one-use binding to an exact proposal, needed
    whenever the capture holds more than one readable answer. Omitting it is
    the ungated single-proposal case, and it refuses when there is more than
    one.
    """
    capture = _read_capture(config, operation_id)
    if capture.plan.operation_state != "result_captured":
        raise CaptureRecoveryError(
            f"Operation {operation_id} is {capture.plan.operation_state}: its "
            "answer has already reached staging, and janki will not write it "
            "again."
        )
    location, group, selected_by = _resolve_selection(capture, selection)
    binding = extraction_batch.bind_captured_child(config, capture.child)
    result = _result_from(capture, binding, location)
    return extraction_batch.complete_captured_child(
        config,
        binding,
        result,
        capture_recovery=_recovery_block(
            capture.plan, location, group, selected_by
        ),
    )


# --- looking at them first ------------------------------------------------------


def _records_for(
    config: ProjectConfig,
    capture: _Capture,
    prepared: Any,
    location: CaptureProposalLocation,
) -> Sequence[Any]:
    """The canonical records one proposal would stage, built for display only.

    Through the same normalizer and record builder staging uses, in the mode
    the request was made under, so what a reviewer looks at is what they would
    get rather than a second rendering of it.
    """
    expectation = capture.child.expectation
    result = extract.normalize_response(
        _parsed_for(capture, location), expectation.mode, prepared.origin_path.name
    )
    existing = (
        load_records(config.normalized_file)
        if config.normalized_file.exists()
        else []
    )
    known = frozenset(extract.known_ids(existing, scope_id=expectation.scope_id))
    return extract.build_records(result.candidates, prepared, known).records


def _index_document(
    capture: _Capture, entries: Sequence[Mapping[str, Any]]
) -> bytes:
    """A plain local page naming every proposal this capture holds.

    Owner-facing bytes only, like any private unfinished paid reply: they are
    never ordinary model context. No script and no network reference — the card
    pages beside it are the rendered review, and this is the index to them.
    """
    plan = capture.plan
    rows = []
    for entry in entries:
        locations = "".join(
            "<li><code>{pointer}</code> — frame {frame}, block {block}, "
            "tool use {tool}, shape {shape}</li>".format(
                pointer=html.escape(item["json_pointer"]),
                frame=item["frame_index"],
                block=item["block_index"],
                tool=html.escape(item["tool_use_id"] or "—"),
                shape=html.escape(item["envelope_shape"]),
            )
            for item in entry["locations"]
        )
        link = (
            f'<p><a href="{html.escape(entry["cards"])}">Cards from this '
            "proposal</a></p>"
            if entry["cards"]
            else f"<p>Not readable: {html.escape(entry['verdict'])}</p>"
        )
        rows.append(
            "<section><h2><code>{digest}</code></h2><ul>{locations}</ul>"
            "{link}<p>Exact parsed argument: <code>{diagnostic}</code></p>"
            "</section>".format(
                digest=html.escape(entry["proposal_sha256"]),
                locations=locations,
                link=link,
                diagnostic=html.escape(entry["diagnostic"]),
            )
        )
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>Captured proposals for {html.escape(plan.operation_id)}</title>"
        "</head><body>"
        f"<h1>Captured proposals for operation {html.escape(plan.operation_id)}</h1>"
        f"<p>Capture <code>{html.escape(plan.capture_sha256)}</code>, request "
        f"<code>{html.escape(plan.request_fingerprint)}</code>. Nothing here is "
        "staged, promoted or approved; opening it changes nothing and spends "
        "nothing.</p>" + "".join(rows) + "</body></html>\n"
    ).encode("utf-8")


def render_capture_proposals(
    config: ProjectConfig, operation_id: str, output_path: Path
) -> Path:
    """Draw each proposal in one capture as the cards it would really make.

    Through the real exporters, the same way the batch preview draws saved
    proposals, so the owner chooses between cards rather than between opaque
    hashes. Competing proposals cannot share one projection — they may propose
    the same identity differently — so each readable proposal gets its own page
    beside the index, with the exact parsed argument written beside it as a
    JSON diagnostic.

    It writes no staging, mints no operation, makes no paid call and grants no
    approval, and its bytes are owner-facing: like any private unfinished paid
    reply they never enter ordinary model context.
    """
    capture = _read_capture(config, operation_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = None
    entries: list[dict[str, Any]] = []
    for group in capture.plan.groups:
        stem = f"{output_path.stem}-{group.proposal_sha256[:12]}"
        diagnostic = output_path.parent / f"{stem}.json"
        location = group.locations[0]
        value = next(
            proposal.value
            for proposal in capture.proposals
            if proposal.location == location
        )
        diagnostic.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cards = ""
        if group.valid:
            if prepared is None:
                try:
                    prepared = inputs.prepare_corpus_input(
                        capture.child.source, config.scan_inbox
                    )
                except JankiError as exc:
                    raise CaptureRecoveryError(
                        f"These proposals cannot be drawn: {exc}"
                    ) from exc
            records = _records_for(config, capture, prepared, location)
            projection = extraction_batch.project_proposals(
                config,
                records,
                deck_path=capture.child.expectation.destination_deck,
                scratch_slug=f"capture-{operation_id}-{group.proposal_sha256[:12]}",
                scratch_title=f"Captured proposal {group.proposal_sha256[:8]}",
            )
            preview = card_preview.render_card_preview(
                config,
                projection.deck_path,
                proposed=projection.overlay,
                new_record_ids=projection.new_record_ids,
                scope_record_ids=projection.scope_record_ids,
                subtitle=(
                    "Recovered from a captured reply — nothing is staged or "
                    "promoted yet"
                ),
            )
            written = card_preview.write_card_preview(
                preview, output_path.parent / f"{stem}.html"
            )
            cards = written.name
        entries.append(
            {
                "proposal_sha256": group.proposal_sha256,
                "verdict": group.schema_verdict,
                "locations": [item.to_dict() for item in group.locations],
                "cards": cards,
                "diagnostic": diagnostic.name,
            }
        )
    output_path.write_bytes(_index_document(capture, entries))
    return output_path
