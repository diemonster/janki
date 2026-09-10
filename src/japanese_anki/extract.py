"""Reading a PDF or photo into rich cards and source patterns for review.

``janki extract`` is the one command that *guesses*. Everything else in this
project is a rule over data someone already wrote down; this reads a textbook
page or a photograph of a whiteboard and proposes complete cards and what the
source teaches. Nothing it produces goes straight to ``vocabulary.json``:
every candidate lands in ``data/staging/`` with its page number, source line,
examples, and confidence, and every inferred pattern remains unreviewed.

That is also why the stop reason is checked before anything is written. A
refusal and a truncated answer both arrive as ordinary successful responses
(see :mod:`claude_client`), and a half-read vocabulary table is worse than no
file at all — it looks complete, and the words it lost are exactly the ones
nobody will notice are missing. Neither is ever written.

Candidates that janki already has a record for are kept, marked
``already_known``, and sorted last. Dropping them would be a silent discard;
mixing them in would bury the new words the extraction was run for.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from japanese_anki import ai_schema as shared_ai_schema
from japanese_anki import claude_client, patterns, prompts, repairs
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import (
    record_scope_id,
    stable_record_id,
    validate_scope_id,
)
from japanese_anki.inputs import PreparedInput
from japanese_anki.models import (
    SourceFormsTable,
    SourceReference,
    VocabularyRecord,
    mark_provisional,
)
from japanese_anki.staging import annotate

__all__ = [
    "CONFIDENCE_LEVELS",
    "EXTRACTION_SCHEMA_VERSION",
    "LAYOUT_MODE",
    "LAYOUT_MODE_REFUSAL",
    "MODES",
    "SOURCE_UNIT_DISPOSITIONS",
    "ExtractError",
    "ExtractionResult",
    "RecordBuild",
    "SourceUnit",
    "TABLE_MODES",
    "TableColumn",
    "TableLayout",
    "build_records",
    "candidate_accounting_fingerprint",
    "candidate_schema",
    "context_fingerprint",
    "coverage_block",
    "extract_candidates",
    "layout_from_provenance",
    "normalize_context",
    "normalize_response",
    "prompt_provenance",
    "prompt_for",
    "refuse_unbound_layout_mode",
    "source_fingerprint",
    "staging_path",
    "staging_targets",
    "prompt_name",
    "unusable_note",
    "validate_candidate_accounting_block",
]

#: The mode a source whose printed columns the owner has bound is sent under.
#: It is a complete additional template, not a branch inside `extract-table`,
#: and it is never chosen for a source: the owner binds a layout to a part, and
#: that binding is what makes this mode available for it.
LAYOUT_MODE = "table-layout"

#: ``--mode`` values. Omitting the flag lets the model judge each page for
#: itself, which DESIGN_V2 makes the default because one PDF often holds both.
MODES: tuple[str, ...] = ("table", "prose", LAYOUT_MODE)

#: The modes that read a printed table. Both are exhaustive over the rows the
#: source prints, so every mode branch that means "this page had a table"
#: names this set rather than the literal ``"table"`` — an unlisted mode would
#: otherwise skip the guard and the coverage block silently.
TABLE_MODES: frozenset[str] = frozenset({"table", LAYOUT_MODE})

#: What a candidate's ``confidence`` may say. Ordered worst-last so a reviewer
#: reading top to bottom meets the shakiest guesses first.
CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low")

SOURCE_UNIT_DISPOSITIONS: tuple[str, ...] = (
    "candidate",
    "duplicate",
    "non-vocabulary",
    "unreadable",
)

# Increment this when the structured response contract changes. It is stored
# with prompt provenance, so a later model drift report can separate a prompt
# change from a parser or schema change.
EXTRACTION_SCHEMA_VERSION = 5


class ExtractError(JankiError):
    """A deterministic extraction failure with a stable case identity."""

    def __init__(self, message: str, *, code: str = "extract-error") -> None:
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


@dataclass(frozen=True, slots=True)
class TableColumn:
    """One column of a layout the owner bound, exactly as they recorded it.

    ``column_id`` is minted locally and opaquely when the owner reviews the
    table; it is never a printed heading, which is what removes the
    duplicate-heading collapse and stops any code from matching a label.
    ``ordinal`` is printed order. ``label_witnesses`` are the exact printed
    strings, preserved verbatim — janki never reads them, and trimming or
    normalizing one would silently change what the request said. ``display_label``
    is the one owner-chosen display string, and two columns may share it because
    their identities differ.
    """

    column_id: str
    ordinal: int
    label_witnesses: tuple[str, ...]
    display_label: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "column_id": self.column_id,
            "ordinal": self.ordinal,
            "label_witnesses": list(self.label_witnesses),
            "display_label": self.display_label,
        }


@dataclass(frozen=True, slots=True)
class TableLayout:
    """One immutable owner-assigned layout revision.

    Immutable per ``(layout_id, revision)``: the study job's ``append_layout``
    is the only writer that may create one, and a dispatched child's request
    manifest freezes the whole object rather than the identity pair — all
    columns, witnesses and labels come back from the saved provenance so a
    retry sends the identical request without consulting a mutable binding.
    """

    layout_id: str
    revision: int
    columns: tuple[TableColumn, ...]

    @property
    def column_ids(self) -> tuple[str, ...]:
        return tuple(column.column_id for column in self.columns)

    @property
    def identity(self) -> str:
        """``layout_id revision N`` — how a refusal names this layout."""

        return f"{self.layout_id} revision {self.revision}"

    def to_wire(self) -> dict[str, Any]:
        """The exact frozen JSON a request manifest records.

        One canonical spelling, because the batch preview compares the staged
        ``prompt_provenance`` with the child's planned copy after canonical
        JSON: a layout serialized one way at plan time and another at staging
        time is a preview failure a long way from its cause.
        """

        return {
            "layout_id": self.layout_id,
            "revision": self.revision,
            "columns": [column.to_wire() for column in self.columns],
        }

    @classmethod
    def from_wire(cls, raw: Any, *, where: str = "table_layout") -> TableLayout:
        """Read one frozen layout back. Structural checks only."""

        if not isinstance(raw, Mapping) or set(raw) != {
            "layout_id",
            "revision",
            "columns",
        }:
            raise ExtractError(
                f"{where} is one object holding layout_id, revision and columns.",
                code="extract-layout-invalid",
            )
        layout_id = raw["layout_id"]
        revision = raw["revision"]
        if not isinstance(layout_id, str) or not layout_id.strip():
            raise ExtractError(
                f"{where}.layout_id is a nonblank identity.",
                code="extract-layout-invalid",
            )
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ExtractError(
                f"{where}.revision is a positive whole number, not {revision!r}.",
                code="extract-layout-invalid",
            )
        raw_columns = raw["columns"]
        if not isinstance(raw_columns, list) or not raw_columns:
            raise ExtractError(
                f"{where}.columns is a nonempty ordered list of columns.",
                code="extract-layout-invalid",
            )
        columns: list[TableColumn] = []
        seen: set[str] = set()
        for position, entry in enumerate(raw_columns, start=1):
            spot = f"{where}.columns[{position}]"
            if not isinstance(entry, Mapping) or set(entry) != {
                "column_id",
                "ordinal",
                "label_witnesses",
                "display_label",
            }:
                raise ExtractError(
                    f"{spot} holds exactly column_id, ordinal, label_witnesses "
                    "and display_label.",
                    code="extract-layout-invalid",
                )
            column_id = entry["column_id"]
            ordinal = entry["ordinal"]
            witnesses = entry["label_witnesses"]
            display_label = entry["display_label"]
            if not isinstance(column_id, str) or not column_id.strip():
                raise ExtractError(
                    f"{spot}.column_id is a nonblank opaque identity.",
                    code="extract-layout-invalid",
                )
            if column_id in seen:
                raise ExtractError(
                    f"{where} declares column {column_id!r} twice; one identity "
                    "names one column.",
                    code="extract-layout-invalid",
                )
            seen.add(column_id)
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
                raise ExtractError(
                    f"{spot}.ordinal is the column's one-based printed position.",
                    code="extract-layout-invalid",
                )
            if (
                not isinstance(witnesses, list)
                or not witnesses
                or any(not isinstance(item, str) for item in witnesses)
            ):
                raise ExtractError(
                    f"{spot}.label_witnesses are the exact printed strings the "
                    "owner recorded, at least one of them.",
                    code="extract-layout-invalid",
                )
            if not isinstance(display_label, str) or not display_label:
                raise ExtractError(
                    f"{spot}.display_label is the owner's display string.",
                    code="extract-layout-invalid",
                )
            columns.append(
                TableColumn(
                    column_id=column_id,
                    ordinal=ordinal,
                    # Verbatim, in the order they were recorded.
                    label_witnesses=tuple(witnesses),
                    display_label=display_label,
                )
            )
        return cls(
            layout_id=layout_id, revision=revision, columns=tuple(columns)
        )


def layout_from_provenance(provenance: Mapping[str, Any]) -> TableLayout | None:
    """The frozen layout one saved request was sent with, or ``None``.

    Pure, and over the saved provenance **and nothing else**: no job, no
    config, no store. The binding in a job's ``choices`` is mutable and
    repointable, so resolving one here would let today's choice redefine a
    request somebody already paid for. Replay therefore reads the frozen layout
    from the saved request manifest, exactly as it already reads the saved mode
    back rather than today's.

    Mode and layout must pair in both directions. A saved ordinary capture
    normalizes to *no layout* rather than to an invented empty one: the absence
    is the historical fact, and nothing is backfilled onto it.
    """
    mode = provenance.get("mode")
    raw = provenance.get("table_layout")
    if mode == LAYOUT_MODE:
        if raw is None:
            raise ExtractError(
                f"A {LAYOUT_MODE} request records the exact layout it was sent "
                "with; this provenance carries none, so janki cannot say which "
                "columns were asked for.",
                code="extract-layout-missing",
            )
        return TableLayout.from_wire(raw)
    if raw is not None:
        raise ExtractError(
            f"This provenance records a table layout under {str(mode)!r} mode. "
            f"Only {LAYOUT_MODE} sends one, so janki will not read the two as "
            "one request.",
            code="extract-layout-unexpected",
        )
    return None


@dataclass(frozen=True, slots=True)
class SourceUnit:
    """One table/list source row, including rows that do not make a card."""

    page: int
    section: str
    ordinal: int
    context: str
    context_fingerprint: str
    disposition: str
    reason: str

    @property
    def key(self) -> tuple[int, str, int]:
        return self.page, self.section, self.ordinal

    def fact(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "section": self.section,
            "ordinal": self.ordinal,
            "context_fingerprint": self.context_fingerprint,
            "disposition": self.disposition,
        }

    def staging_value(self) -> dict[str, Any]:
        value = self.fact()
        value["context"] = self.context
        if self.reason:
            value["reason"] = self.reason
        return value


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Normalized model output before candidates become records."""

    candidates: tuple[Any, ...]
    source_units: tuple[SourceUnit, ...]
    model_reported_unit_count: int
    pattern_set: patterns.PatternSet = field(
        default_factory=lambda: patterns.PatternSet(source="")
    )

    @property
    def prose_candidates(self) -> tuple[Any, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if str(getattr(candidate, "source_kind", "prose")) == "prose"
        )


@dataclass(frozen=True, slots=True)
class RecordBuild:
    """Canonical records plus a complete account of the parsed proposals.

    A deterministic record ID permits only one canonical staging row.  It does
    not make a later proposal disposable: the proposal may carry different
    source evidence or teaching content.  Callers therefore receive both
    products together so they cannot write the records and unknowingly lose
    the other paid output.
    """

    records: tuple[VocabularyRecord, ...]
    unusable_candidates: tuple[Any, ...]
    candidate_accounting: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CandidatePlan:
    canonical_candidates: tuple[Any, ...]
    unusable_candidates: tuple[Any, ...]
    accounting: dict[str, Any]


@functools.cache
def candidate_schema() -> Any:
    """The Pydantic model the response must match.

    Built on demand rather than at import time because it needs ``pydantic``,
    which arrives with the ``ai`` extra — the same lazy-import rule
    :mod:`claude_client` follows so that non-AI commands run without it. Cached
    so every call yields the *same* class: a fresh one each time would make an
    instance built by one call fail validation in another, and would defeat the
    API's own 24-hour schema cache by presenting an identical schema as new.

    Provenance lives *in the schema*: the model reports the page it read a word
    on, the surrounding line, and how sure it is. Asking for those alongside
    the word is what makes the staging file reviewable — a candidate a human
    cannot locate on the page is one they cannot check.
    """
    from typing import Annotated, Literal

    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        StringConstraints,
        model_validator,
    )

    RichCard = shared_ai_schema.extraction_rich_card_schema()
    SourcePattern = shared_ai_schema.source_pattern_schema()
    NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    class CandidateRecord(RichCard):
        model_config = ConfigDict(extra="forbid")

        expression: str = Field(description="The word as written, in Japanese.")
        reading: str = Field(
            description="Kana reading.",
        )
        part_of_speech: str = Field(description="Part of speech, if known.")
        page: int = Field(
            ge=1,
            description="1-indexed page this was read from.",
        )
        context: NonBlank = Field(
            description="The line or cell this was read from, verbatim."
        )
        confidence: Literal["high", "medium", "low"] = Field(
            description="Confidence.",
        )
        inclusion_reason: str = Field(
            description="In prose mode, why this word is worth a card.",
        )
        source_kind: Literal["table", "prose"] = Field(
            description=(
                "Whether this candidate comes from a table/list source unit or "
                "from prose selection."
            ),
        )
        section: str = Field(
            description=(
                "Stable lowercase section slug. Required for a table candidate."
            ),
        )
        ordinal: int = Field(
            ge=0,
            description=(
                "One-based row ordinal within the section. Required for a table "
                "candidate."
            ),
        )
        conjugations: dict[str, str] = Field(
            default_factory=dict,
            description=(
                "Source conjugation columns: each printed column label to that "
                "row's supplied form, in printed order. Empty when none."
            ),
        )
        source_chapters: list[str] = Field(
            default_factory=list,
            description=(
                "Chapter labels teaching this word, exactly as printed, in "
                "printed order. Empty when none."
            ),
        )

        @model_validator(mode="after")
        def has_source_kind_evidence(self) -> Any:
            if self.source_kind == "prose" and not self.inclusion_reason.strip():
                raise ValueError("a prose candidate needs a nonblank inclusion_reason")
            if self.source_kind == "table":
                if not self.section.strip():
                    raise ValueError("a table candidate needs a nonblank section")
                if self.ordinal < 1:
                    raise ValueError("a table candidate needs a one-based ordinal")
            return self

    class SourceUnitRecord(BaseModel):
        model_config = ConfigDict(extra="forbid")

        page: int = Field(ge=1, description="One-based page number.")
        section: str = Field(
            min_length=1,
            description="Stable lowercase slug for the table or list section.",
        )
        ordinal: int = Field(
            ge=1, description="One-based row ordinal within the section."
        )
        context: str = Field(
            min_length=1, description="The complete source row or cell text, verbatim."
        )
        disposition: Literal[
            "candidate", "duplicate", "non-vocabulary", "unreadable"
        ]
        reason: str = Field(
            default="",
            description="Disposition reason.",
        )

    class Extraction(BaseModel):
        model_config = ConfigDict(extra="forbid")

        candidates: list[CandidateRecord] = Field(default_factory=list)
        source_units: list[SourceUnitRecord] = Field(default_factory=list)
        model_reported_unit_count: int = Field(
            default=0,
            ge=0,
            description="Reported source-unit count.",
        )
        document_kind: Literal["pattern", "lesson", "vocabulary", "unknown"] = Field(
            default="unknown", description="Document kind."
        )
        document_title: str = Field(default="", description="Document title.")
        patterns: list[SourcePattern] = Field(default_factory=list)

    return Extraction


#: What a control that cannot carry a binding says when it is asked for the
#: layout-bound mode. One sentence, spelled once, so the CLI, the batch command
#: and both workbench routes refuse in the same words.
LAYOUT_MODE_REFUSAL = (
    f"{LAYOUT_MODE} extraction sends the exact layout the repository owner "
    "bound to that source part, and this control carries no binding. Bind one "
    "with `janki study layout JOB --part PART --layout FILE`, then send the "
    "part with `janki study extract JOB`. Nothing was sent."
)


def refuse_unbound_layout_mode(mode: str | None, *, control: str) -> None:
    """Refuse ``table-layout`` at an entry point that cannot bind a layout.

    Every consumer of :data:`MODES` calls this explicitly rather than
    inheriting the widened tuple: adding a mode to that tuple makes it typeable
    at the CLI, postable on the workbench's paid form and reachable by URL on
    the offered-mode route, and a request sent under it with no bound layout
    would ask the model to key its answer by identities nobody supplied.
    """
    if mode == LAYOUT_MODE:
        raise ExtractError(
            f"{control}: {LAYOUT_MODE_REFUSAL}",
            code="extract-layout-unbound",
        )


def prompt_name(mode: str | None) -> str:
    """The template one extraction mode sends.

    A name, not a text: the modes are three separate files under `prompts/`
    rather than one file with three rule blocks, because the work they ask for
    genuinely differs and a reader of `extract-table.md` should not have to
    mentally delete the prose paragraphs. Mode validation stays here, in the
    module that owns what a mode means.
    """
    if mode is None:
        return "extract-auto"
    if mode in MODES:
        return f"extract-{mode}"
    raise ExtractError(
        f"Unknown mode '{mode}'. Use one of: {', '.join(MODES)}, or omit "
        "--mode to let the model judge each page.",
        code="extract-mode-unknown",
    )


def _layout_block(layout: TableLayout) -> str:
    """The labelled data turn one bound layout adds to the ordinary ask.

    Data, in Python, rather than prose in a template: AGENTS keeps a branching
    instruction out of the prompt files, and a template that described columns
    would be a second, drifting definition of what was sent. Every part of the
    layout the owner authored rides here — identity, revision, each column's
    opaque id, its printed position, the exact printed headings recorded for it
    and the owner's display label — because all of it is part of what the model
    was told, so any of it differing is a different request.
    """
    lines = [
        "",
        "Printed source-form columns for this page, bound by the repository "
        f"owner as layout {layout.layout_id} revision {layout.revision}:",
    ]
    for column in layout.columns:
        witnesses = " | ".join(column.label_witnesses)
        lines.append(
            f"  {column.ordinal}. id={column.column_id}"
            f"  display label: {column.display_label}"
            f"  printed heading(s): {witnesses}"
        )
    lines.append(
        "Key each row's `conjugations` map by these exact `id` values, never by "
        "a printed heading and never by an id that is not listed above. Copy the "
        "cell that row prints under that column as the value: an empty string "
        "where the page printed a blank cell, and no key at all where that row "
        "has no cell for that column."
    )
    return "\n".join(lines)


def prompt_for(
    source_name: str,
    known: Sequence[str] = (),
    *,
    layout: TableLayout | None = None,
) -> str:
    """The user-turn text for one file.

    The known-word list rides here rather than in the system blocks on purpose:
    it changes every time the collection grows, and anything above the cache
    breakpoint that changes invalidates the cached style guide for every run.

    Two other blocks used to ride here — approved source-unit keys, and
    approved prose-selection targets with their rubric — binding a human's
    inventory of a page into the prompt so coverage could be scored against it.
    That was the pilot programme's question, and it was cancelled with it
    (M8.4). What is left is the ordinary ask, plus the owner's bound layout
    when there is one.
    """
    lines = [f"Source file: {source_name}"]
    if known:
        lines.append("\nKnown expressions:\n" + "、".join(known))
    if layout is not None:
        lines.append(_layout_block(layout))
    return "\n".join(lines)


_SPACE = re.compile(r"\s+")
_SECTION = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


def normalize_context(value: str) -> str:
    """Normalize source context once for both model output and human oracles."""
    normalized = unicodedata.normalize("NFC", str(value)).replace("\r\n", "\n")
    return _SPACE.sub(" ", normalized).strip()


def context_fingerprint(value: str) -> str:
    return hashlib.sha256(normalize_context(value).encode("utf-8")).hexdigest()


def source_fingerprint(path: Path) -> str:
    """Return the SHA-256 of one prepared, immutable inbox source."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plain_provenance(value: Any) -> Any:
    """Plain JSON types for a durable record.

    The provider hands back read-only mappings and tuples, which describe the
    request exactly and serialize to nothing. Converted once here so both the
    staged provenance and the captured envelope hold the same plain shape.
    """
    if isinstance(value, Mapping):
        return {str(key): _plain_provenance(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_provenance(item) for item in value]
    return value


def prompt_provenance(
    prepared: PreparedInput,
    *,
    model: str,
    style_guide: str,
    system: str,
    mode: str | None,
    known: Sequence[str] = (),
    source_sha256: str | None = None,
    provider_plan: Any | None = None,
    layout: TableLayout | None = None,
) -> dict[str, Any]:
    """The stable inputs needed to explain a later model-output change.

    ``provider_plan`` is the immutable request the shared provider registry
    planned for this source. When it is given, its identity *is* this
    provenance's identity — provider, model, schema and request fingerprints
    are copied from the object that will be sent rather than recomputed
    beside it, because a provenance that describes a second, similar request
    cannot explain the call that was actually made. Its durable manifest and
    exact planned channels ride along so the same request can be rebuilt,
    and its answer re-read, without re-planning from today's prompts.

    ``table_layout`` is recorded **only when a layout is given**, so no
    existing capture's provenance bytes change and an old capture still
    normalizes to no layout. The response contract is untouched: the model
    returns the supplied identities as keys of the existing conjugation map, so
    ``response_schema_fingerprint`` is identical and only the request moves.
    """
    user = prompt_for(prepared.origin_path.name, known, layout=layout)
    schema = candidate_schema()
    wire_schema = claude_client.wire_schema(schema)
    provenance: dict[str, Any] = {
        "source_sha256": source_sha256 or source_fingerprint(prepared.origin_path),
        "mode": mode or "auto",
        "provider": "anthropic",
        "model": model,
        "response_schema_version": EXTRACTION_SCHEMA_VERSION,
        "response_schema_fingerprint": prompts.schema_fingerprint(wire_schema),
        "system_prompt_fingerprint": _text_fingerprint(system),
        "style_guide_fingerprint": _text_fingerprint(style_guide),
        "user_prompt_fingerprint": _text_fingerprint(user),
        "request_fingerprint": prompts.request_fingerprint(
            provider="anthropic",
            style_guide=style_guide,
            task_template=system,
            user_turn=user,
            transport_prompt={"system": [style_guide, system], "user": user},
            schema=wire_schema,
        ),
    }
    if layout is not None:
        provenance["table_layout"] = layout.to_wire()
    if provider_plan is None:
        return provenance
    provenance["provider"] = provider_plan.provider
    provenance["model"] = provider_plan.model
    # `response_schema_fingerprint` stays janki's own, computed above over the
    # wire schema: it is what a later reader compares its current decoder
    # against. The provider's neutral fingerprint for the same contract is
    # kept inside the manifest, where its own validation uses it.
    provenance["request_fingerprint"] = provider_plan.request_fingerprint
    provenance["provider_manifest"] = _plain_provenance(
        provider_plan.persistent_manifest()
    )
    # The channels the manifest alone cannot rebuild. Larger than the rest of
    # this record — a document rides in `input_blocks` — and kept anyway: a
    # recovery that regenerated them from today's prompt would reconstruct a
    # different request and call it the one that was paid for.
    provenance["provider_channels"] = _plain_provenance(
        {
            "system_blocks": list(provider_plan.system_blocks),
            "user_turn": provider_plan.user_turn,
            "input_blocks": list(provider_plan.input_blocks),
        }
    )
    return provenance


def extract_candidates(
    prepared: PreparedInput,
    *,
    model: str,
    style_guide: str,
    system: str,
    mode: str | None = None,
    known: Sequence[str] = (),
    client: Any | None = None,
    capture: Any | None = None,
    provenance: dict[str, Any] | None = None,
    layout: TableLayout | None = None,
) -> ExtractionResult:
    """The normalized response one file yields, or a stable diagnostic.

    Both incomplete outcomes are refused rather than salvaged. A refusal is
    reported with the category the API gave, because "it was declined" without
    a reason leaves a user with nothing to act on. A ``max_tokens`` stop means
    the answer was cut mid-word: the visible half would look like a complete
    extraction and the rest would be lost silently, which is the one failure
    this whole command is arranged to avoid.

    This transport builds its own user turn, so ``layout`` has to reach it:
    a layout-bearing provenance beside a turn that never carried the block
    would describe a request that was not sent.
    """
    try:
        call = claude_client.parse_call(
            model,
            claude_client.system_blocks(style_guide, system),
            [
                prepared.content_block(),
                {
                    "type": "text",
                    "text": prompt_for(
                        prepared.origin_path.name, known, layout=layout
                    ),
                },
            ],
            candidate_schema(),
            client,
            effort=claude_client.effort_for(model),
            # Called with the exact answer before it is validated, so a
            # caller can persist what was paid for even when the schema
            # then rejects it.
            capture=capture,
        )
    except ExtractError:
        raise
    except ImportError as exc:
        raise ExtractError(
            f"{prepared.origin_path.name}: extraction needs the AI schema "
            "dependencies. Install the project with '.[ai]'.",
            code="extract-schema-dependency-missing",
        ) from exc
    except JankiError as exc:
        detail = str(exc).rstrip()
        if "nothing was written" not in detail.casefold():
            detail += " Nothing was written for this source."
        raise ExtractError(
            f"{prepared.origin_path.name}: {detail}",
            code="extract-model-call-failed",
        ) from exc

    if provenance is None:
        provenance = prompt_provenance(
            prepared,
            model=model,
            style_guide=style_guide,
            system=system,
            mode=mode,
            known=known,
            source_sha256=prepared.source_sha256 or None,
            layout=layout,
        )
    return extraction_result_from_call(
        call, prepared, model=model, mode=mode, provenance=provenance
    )


def extraction_result_from_call(
    call: Any,
    prepared: PreparedInput,
    *,
    model: str,
    mode: str | None,
    provenance: dict[str, Any],
) -> ExtractionResult:
    """The normalized result one already-paid-for answer yields.

    Shared by every transport: the subscription's streamed reply, its pure
    recovery from captured bytes, and the explicit Anthropic API call all
    arrive here with an answer in hand. Splitting it out is what keeps them
    one content pass — a second reader would be a second set of rules about
    what counts as a complete answer.

    ``provenance`` is passed in rather than computed, so what is written
    beside the proposals describes the request that was actually sent.
    """
    parsed, stop_reason, refusal = call.parsed, call.stop_reason, call.refusal
    if stop_reason == "refusal":
        detail = ""
        if refusal is not None:
            detail = f" ({refusal.category}" + (
                f": {refusal.explanation})" if refusal.explanation else ")"
            )
        raise ExtractError(
            f"{model} declined to read {prepared.origin_path.name}{detail}. "
            "Nothing was written.",
            code="extract-model-refusal",
        )
    if stop_reason == "max_tokens":
        raise ExtractError(
            f"{model} ran out of room part-way through {prepared.origin_path.name}, "
            "so the answer is cut off and janki will not write a half-read file. "
            "Give it less to read at once: split a long document and run the parts "
            "separately, or crop a dense photo to the section you want.",
            code="extract-response-truncated",
        )
    if parsed is None:
        raise ExtractError(
            f"{model} returned nothing usable for {prepared.origin_path.name} "
            f"(stop reason: {stop_reason}). Nothing was written.",
            code="extract-response-missing",
        )
    result = normalize_response(parsed, mode, prepared.origin_path.name)
    return ExtractionResult(
        candidates=result.candidates,
        source_units=result.source_units,
        model_reported_unit_count=result.model_reported_unit_count,
        pattern_set=patterns.with_prompt_provenance(result.pattern_set, provenance),
    )


def _candidate_key(candidate: Any) -> tuple[int, str, int]:
    return (
        int(getattr(candidate, "page", 0) or 0),
        str(getattr(candidate, "section", "") or "").strip(),
        int(getattr(candidate, "ordinal", 0) or 0),
    )


def normalize_response(
    parsed: Any, mode: str | None, source_name: str
) -> ExtractionResult:
    candidates = tuple(parsed.candidates)
    units: list[SourceUnit] = []
    for index, raw in enumerate(parsed.source_units, start=1):
        section = str(raw.section).strip()
        if not _SECTION.fullmatch(section):
            raise ExtractError(
                f"{source_name}: source unit {index} has section {section!r}; use a "
                "stable lowercase slug such as 'lesson-3-table'.",
                code="extract-unit-section-invalid",
            )
        context = str(raw.context)
        if not normalize_context(context):
            raise ExtractError(
                f"{source_name}: source unit {raw.page}/{section}/{raw.ordinal} "
                "has empty context.",
                code="extract-unit-context-missing",
            )
        disposition = str(raw.disposition)
        reason = str(raw.reason or "").strip()
        if disposition != "candidate" and not reason:
            raise ExtractError(
                f"{source_name}: source unit {raw.page}/{section}/{raw.ordinal} is "
                f"{disposition} but has no reason.",
                code="extract-unit-reason-missing",
            )
        units.append(
            SourceUnit(
                page=int(raw.page),
                section=section,
                ordinal=int(raw.ordinal),
                context=context,
                context_fingerprint=context_fingerprint(context),
                disposition=disposition,
                reason=reason,
            )
        )

    table_candidates = tuple(
        candidate
        for candidate in candidates
        if str(getattr(candidate, "source_kind", "prose")) == "table"
    )
    prose_candidates = tuple(
        candidate
        for candidate in candidates
        if str(getattr(candidate, "source_kind", "prose")) == "prose"
    )
    # Both table modes, named explicitly rather than inherited: an unlisted
    # mode string would silently skip this guard and the prose one below.
    if mode in TABLE_MODES and prose_candidates:
        raise ExtractError(
            f"{source_name}: {mode} mode returned {len(prose_candidates)} prose "
            "candidate(s).",
            code="extract-table-prose-candidate",
        )
    if mode == "prose" and (units or table_candidates):
        raise ExtractError(
            f"{source_name}: prose mode returned table source units or candidates.",
            code="extract-prose-table-unit",
        )

    candidate_units = [unit for unit in units if unit.disposition == "candidate"]
    candidate_keys = sorted(_candidate_key(candidate) for candidate in table_candidates)
    unit_keys = sorted(unit.key for unit in candidate_units)
    if candidate_keys != unit_keys:
        raise ExtractError(
            f"{source_name}: table candidate keys do not match candidate source-unit "
            "keys one-to-one.",
            code="extract-candidate-unit-link",
        )
    unit_contexts: dict[tuple[int, str, int], list[str]] = {}
    for unit in candidate_units:
        unit_contexts.setdefault(unit.key, []).append(unit.context_fingerprint)
    for candidate in table_candidates:
        key = _candidate_key(candidate)
        fingerprint = context_fingerprint(str(getattr(candidate, "context", "") or ""))
        expected = unit_contexts.get(key, [])
        if fingerprint not in expected:
            raise ExtractError(
                f"{source_name}: table candidate {key[0]}/{key[1]}/{key[2]} does "
                "not copy its source unit context.",
                code="extract-candidate-context-mismatch",
            )
        expected.remove(fingerprint)

    document_kind = str(
        getattr(parsed, "document_kind", "unknown") or "unknown"
    ).strip().lower()
    if document_kind not in patterns.DOCUMENT_KINDS:
        document_kind = "unknown"
    pattern_set = patterns.PatternSet(
        source=source_name,
        kind=document_kind,
        title=str(getattr(parsed, "document_title", "") or "").strip(),
        patterns=tuple(
            patterns.Pattern(
                template=str(getattr(item, "template", "") or "").strip(),
                gloss=str(getattr(item, "gloss", "") or "").strip(),
                examples=tuple(
                    text
                    for example in (getattr(item, "examples", []) or [])
                    if (text := str(example).strip())
                ),
                where=str(getattr(item, "where", "") or "").strip(),
            )
            for item in (getattr(parsed, "patterns", []) or [])
            if str(getattr(item, "template", "") or "").strip()
        ),
    )

    return ExtractionResult(
        candidates=candidates,
        source_units=tuple(units),
        model_reported_unit_count=int(parsed.model_reported_unit_count),
        pattern_set=pattern_set,
    )


def _unit_sort(value: dict[str, Any]) -> tuple[int, str, int, str]:
    return (
        int(value.get("page", 0)),
        str(value.get("section", "")),
        int(value.get("ordinal", 0)),
        str(value.get("disposition", "")),
    )


def _key_value(key: tuple[int, str, int]) -> dict[str, Any]:
    return {"page": key[0], "section": key[1], "ordinal": key[2]}


def _canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_CANDIDATE_ACCOUNTING_FIELDS = {
    "version",
    "parsed_candidate_count",
    "canonical_record_count",
    "unusable_candidate_count",
    "duplicate_candidate_count",
    "collision_group_count",
    "collision_groups",
    "candidate_accounting_fingerprint",
}
_CANDIDATE_ACCOUNTING_COUNT_FIELDS = (
    "parsed_candidate_count",
    "canonical_record_count",
    "unusable_candidate_count",
    "duplicate_candidate_count",
    "collision_group_count",
)
_PARSED_CANDIDATE_FIELDS = {
    "meanings",
    "examples",
    "usage_notes",
    "expression",
    "reading",
    "part_of_speech",
    "page",
    "context",
    "confidence",
    "inclusion_reason",
    "source_kind",
    "section",
    "ordinal",
    "conjugations",
    "source_chapters",
}
#: Fields a stored proposal may omit entirely. Candidate accounting is an
#: immutable record of what was paid for: a proposal written before these two
#: existed is complete as written, and both default to empty, so absence is
#: readable rather than something to backfill or adapt.
_OPTIONAL_PARSED_CANDIDATE_FIELDS = {"conjugations", "source_chapters"}
_PARSED_EXAMPLE_FIELDS = {
    "japanese",
    "speech_level",
    "furigana",
    "romaji",
    "english",
}


def _is_integer(value: Any, *, minimum: int | None = None) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and (minimum is None or value >= minimum)
    )


def _is_parsed_schema_proposal(value: Any) -> bool:
    """Validate the JSON shape without loading optional AI dependencies."""
    if not isinstance(value, Mapping):
        return False
    present = set(value)
    if not present <= _PARSED_CANDIDATE_FIELDS or not present >= (
        _PARSED_CANDIDATE_FIELDS - _OPTIONAL_PARSED_CANDIDATE_FIELDS
    ):
        return False
    conjugations = value.get("conjugations", {})
    chapters = value.get("source_chapters", [])
    if (
        not isinstance(conjugations, Mapping)
        or any(
            not isinstance(label, str) or not isinstance(form, str)
            for label, form in conjugations.items()
        )
        or not isinstance(chapters, list)
        or any(not isinstance(item, str) for item in chapters)
    ):
        return False
    text_fields = {
        "usage_notes",
        "expression",
        "reading",
        "part_of_speech",
        "context",
        "inclusion_reason",
        "section",
    }
    if any(not isinstance(value.get(name), str) for name in text_fields):
        return False
    meanings = value.get("meanings")
    examples = value.get("examples")
    if (
        not isinstance(meanings, list)
        or any(not isinstance(item, str) for item in meanings)
        or not isinstance(examples, list)
        or not _is_integer(value.get("page"), minimum=0)
        or not _is_integer(value.get("ordinal"), minimum=0)
        or value.get("confidence") not in CONFIDENCE_LEVELS
        or value.get("source_kind") not in {"table", "prose"}
    ):
        return False
    for example in examples:
        if (
            not isinstance(example, Mapping)
            or set(example) != _PARSED_EXAMPLE_FIELDS
            or any(
                not isinstance(example.get(name), str)
                for name in _PARSED_EXAMPLE_FIELDS
            )
            or example.get("speech_level") not in {"polite", "casual"}
        ):
            return False
    return True


def candidate_accounting_fingerprint(block: Mapping[str, Any]) -> str:
    """Fingerprint candidate accounting without its own digest field."""
    return _canonical_fingerprint(
        {
            key: value
            for key, value in block.items()
            if key != "candidate_accounting_fingerprint"
        }
    )


def _candidate_plan(candidates: Iterable[Any]) -> _CandidatePlan:
    """Partition parsed schema proposals without interpreting their Japanese."""
    parsed = tuple(candidates)
    canonical: list[Any] = []
    unusable_candidates: list[Any] = []
    first_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    groups_by_id: dict[str, dict[str, Any]] = {}

    for candidate_index, candidate in enumerate(parsed, start=1):
        expression = str(getattr(candidate, "expression", "") or "").strip()
        if not expression:
            unusable_candidates.append(candidate)
            continue
        reading = str(getattr(candidate, "reading", "") or "").strip()
        record_id = stable_record_id(expression, reading)
        parsed_value = candidate.model_dump(mode="json")
        proposal = {
            "candidate_index": candidate_index,
            "parsed_schema_proposal": parsed_value,
        }
        first = first_by_id.get(record_id)
        if first is None:
            first_by_id[record_id] = (candidate_index, parsed_value)
            canonical.append(candidate)
            continue
        group = groups_by_id.get(record_id)
        if group is None:
            first_index, first_value = first
            group = {
                "stable_record_id": record_id,
                "canonical_candidate_index": first_index,
                "proposals": [
                    {
                        "candidate_index": first_index,
                        "parsed_schema_proposal": first_value,
                    }
                ],
            }
            groups_by_id[record_id] = group
        group["proposals"].append(proposal)

    groups = sorted(
        groups_by_id.values(), key=lambda group: group["canonical_candidate_index"]
    )
    duplicate_count = sum(len(group["proposals"]) - 1 for group in groups)
    block: dict[str, Any] = {
        "version": 1,
        "parsed_candidate_count": len(parsed),
        "canonical_record_count": len(canonical),
        "unusable_candidate_count": len(unusable_candidates),
        "duplicate_candidate_count": duplicate_count,
        "collision_group_count": len(groups),
        "collision_groups": groups,
    }
    block["candidate_accounting_fingerprint"] = (
        candidate_accounting_fingerprint(block)
    )
    return _CandidatePlan(
        canonical_candidates=tuple(canonical),
        unusable_candidates=tuple(unusable_candidates),
        accounting=block,
    )


def _accounting_error(message: str, *, stale: bool = False) -> ExtractError:
    return ExtractError(
        message,
        code=("candidate-accounting-stale" if stale else "candidate-accounting-invalid"),
    )


def validate_candidate_accounting_block(block: Mapping[str, Any]) -> None:
    """Validate an immutable account of parsed schema proposals.

    This checks structure and internal identity only. Reviewed staging records
    remain editable; coverage v2 separately binds the original proposal counts.
    """
    if set(block) != _CANDIDATE_ACCOUNTING_FIELDS:
        raise _accounting_error(
            "candidate_accounting fields do not match schema version 1"
        )
    version = block.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise _accounting_error("candidate_accounting version must be integer 1")
    counts: dict[str, int] = {}
    for name in _CANDIDATE_ACCOUNTING_COUNT_FIELDS:
        value = block.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _accounting_error(
                f"candidate_accounting {name} must be a non-negative integer"
            )
        counts[name] = value
    fingerprint = block.get("candidate_accounting_fingerprint")
    try:
        expected_fingerprint = candidate_accounting_fingerprint(block)
    except (TypeError, ValueError) as exc:
        raise _accounting_error(
            "candidate_accounting is not canonical JSON"
        ) from exc
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise _accounting_error(
            "candidate_accounting needs a canonical SHA-256 fingerprint"
        )
    if fingerprint != expected_fingerprint:
        raise _accounting_error(
            "candidate_accounting changed; its fingerprint must be "
            f"{expected_fingerprint}",
            stale=True,
        )
    groups = block.get("collision_groups")
    if not isinstance(groups, list):
        raise _accounting_error("candidate_accounting collision_groups must be a list")
    if len(groups) != counts["collision_group_count"]:
        raise _accounting_error(
            "candidate_accounting collision_group_count does not match its groups"
        )
    if counts["collision_group_count"] > counts["canonical_record_count"]:
        raise _accounting_error(
            "candidate_accounting collision_group_count cannot exceed "
            "canonical_record_count"
        )

    group_ids: set[str] = set()
    proposal_indices: set[int] = set()
    duplicate_count = 0
    previous_group_index = 0
    for group_number, group in enumerate(groups, start=1):
        where = f"candidate_accounting.collision_groups[{group_number}]"
        if not isinstance(group, Mapping) or set(group) != {
            "stable_record_id",
            "canonical_candidate_index",
            "proposals",
        }:
            raise _accounting_error(f"{where} has invalid fields")
        record_id = group.get("stable_record_id")
        if not isinstance(record_id, str) or not record_id or record_id in group_ids:
            raise _accounting_error(f"{where} needs one unique stable_record_id")
        group_ids.add(record_id)
        canonical_index = group.get("canonical_candidate_index")
        if (
            isinstance(canonical_index, bool)
            or not isinstance(canonical_index, int)
            or canonical_index <= previous_group_index
        ):
            raise _accounting_error(
                f"{where}.canonical_candidate_index must preserve candidate order"
            )
        previous_group_index = canonical_index
        proposals = group.get("proposals")
        if not isinstance(proposals, list) or len(proposals) < 2:
            raise _accounting_error(f"{where}.proposals needs every collision member")
        duplicate_count += len(proposals) - 1
        previous_proposal_index = 0
        for proposal_number, proposal_entry in enumerate(proposals, start=1):
            proposal_where = f"{where}.proposals[{proposal_number}]"
            if not isinstance(proposal_entry, Mapping) or set(proposal_entry) != {
                "candidate_index",
                "parsed_schema_proposal",
            }:
                raise _accounting_error(f"{proposal_where} has invalid fields")
            candidate_index = proposal_entry.get("candidate_index")
            if (
                isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
                or candidate_index <= previous_proposal_index
                or candidate_index > counts["parsed_candidate_count"]
                or candidate_index in proposal_indices
            ):
                raise _accounting_error(
                    f"{proposal_where}.candidate_index is invalid or out of order"
                )
            previous_proposal_index = candidate_index
            proposal_indices.add(candidate_index)
            proposal = proposal_entry.get("parsed_schema_proposal")
            if not _is_parsed_schema_proposal(proposal):
                raise _accounting_error(
                    f"{proposal_where}.parsed_schema_proposal has invalid fields"
                )
            expression = str(proposal["expression"]).strip()
            reading = str(proposal["reading"]).strip()
            if not expression or stable_record_id(expression, reading) != record_id:
                raise _accounting_error(
                    f"{proposal_where} does not mint {record_id}"
                )
        if proposals[0]["candidate_index"] != canonical_index:
            raise _accounting_error(
                f"{where}.canonical_candidate_index must name its first proposal"
            )

    if duplicate_count != counts["duplicate_candidate_count"]:
        raise _accounting_error(
            "candidate_accounting duplicate_candidate_count does not match its groups"
        )
    if (
        counts["canonical_record_count"]
        + counts["unusable_candidate_count"]
        + counts["duplicate_candidate_count"]
        != counts["parsed_candidate_count"]
    ):
        raise _accounting_error(
            "candidate_accounting counts do not partition the parsed candidate list"
        )


def coverage_block(
    result: ExtractionResult,
    *,
    source_sha256: str,
    mode: str | None,
    candidate_accounting: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Durable staging metadata: what the model said the source contained.

    This used to compare that against a human's approved inventory of the same
    page — an "oracle" — and score the difference. M8.4 deleted the oracle
    apparatus with the pilot programme it served, so what is left is the record
    itself: the source units the model reported, their dispositions, the counts,
    and the keys it repeated.

    Still worth writing, and still checked. `promote._verify_coverage_facts`
    re-derives this block and refuses a staging file whose numbers no longer
    follow from the units beside them — which catches a hand-edit that changed
    one and not the other. What it no longer claims is that anything was
    *measured*: `status` is "unmeasured" whenever there are table units at all,
    because nobody is asserting what the page held.

    The extract writer always supplies ``candidate_accounting`` and emits v2.
    Omitting it reproduces v1 only to validate paid artifacts written before
    collision preservation existed; no missing parsed proposal is invented.
    """
    actual = list(result.source_units)
    by_key: dict[tuple[int, str, int], SourceUnit] = {}
    duplicate_keys: list[dict[str, Any]] = []
    for unit in actual:
        if unit.key in by_key:
            duplicate_keys.append(_key_value(unit.key))
        else:
            by_key[unit.key] = unit

    # A prose-only file has no table units to be exhaustive about; anything
    # else is unmeasured now that no oracle says what should have been there.
    #
    # Both table modes, inside the function rather than at a call site:
    # `promote._verify_coverage_facts` re-derives this block from the saved
    # mode, so a caller-side fix would make promotion regenerate a different
    # block and refuse the file the extraction had just written.
    has_table = bool(actual) or mode in TABLE_MODES
    status = "unmeasured" if has_table else "selection"

    dispositions = {
        name.replace("-", "_") + "_units": sorted(
            [unit.fact() for unit in actual if unit.disposition == name],
            key=_unit_sort,
        )
        for name in SOURCE_UNIT_DISPOSITIONS
    }
    block: dict[str, Any] = {
        "version": 2 if candidate_accounting is not None else 1,
        "status": status,
        "blocking": has_table,
        "source_fingerprint": source_sha256,
        "model_reported_unit_count": result.model_reported_unit_count,
        "observed_unit_count": len(actual),
        "prose_candidate_count": len(result.prose_candidates),
        "prose_coverage": (
            "unmeasured"
            if (mode == "prose" or bool(result.prose_candidates))
            else "not-applicable"
        ),
        "source_units": [unit.staging_value() for unit in actual],
        "duplicate_keys": sorted(duplicate_keys, key=_unit_sort),
        **dispositions,
    }
    if candidate_accounting is not None:
        validate_candidate_accounting_block(candidate_accounting)
        expected_parsed = len(result.prose_candidates) + len(
            dispositions["candidate_units"]
        )
        if candidate_accounting.get("parsed_candidate_count") != expected_parsed:
            raise _accounting_error(
                "candidate_accounting parsed_candidate_count does not match prose "
                "candidates plus table candidate source units",
                stale=True,
            )
        for name in _CANDIDATE_ACCOUNTING_COUNT_FIELDS:
            block[name] = candidate_accounting.get(name)
        block["candidate_accounting_fingerprint"] = candidate_accounting.get(
            "candidate_accounting_fingerprint"
        )
    block["coverage_block_fingerprint"] = _canonical_fingerprint(block)
    return block


def _raw_fields(
    candidate: Any, prepared: PreparedInput, *, layout: TableLayout | None = None
) -> dict[str, str]:
    """A candidate's provenance, stringified for ``raw_fields``.

    ``raw_fields`` is ``dict[str, str]`` and stays that way (DESIGN_V2: page
    and confidence are stringified into it rather than growing the model), so
    everything a reviewer needs to find the word again is written as text.
    """
    fields = {
        "extracted_from": prepared.origin_path.name,
        "confidence": str(getattr(candidate, "confidence", "") or ""),
    }
    page = getattr(candidate, "page", 0) or 0
    if page:
        fields["page"] = str(page)
    # Source evidence stays separate from the proposed teaching content in
    # ``record.examples``. Promotion's explicit example-acceptance boundary
    # decides whether those proposed sentences become curated content.
    for name in ("context", "inclusion_reason"):
        value = str(getattr(candidate, name, "") or "").strip()
        if value:
            fields[name] = value
    # The witness for what the source's own conjugation table said. The record
    # field beside it is the canonical *display* map, and reading a staging
    # file back through ``VocabularyRecord.from_dict`` trims its cells and
    # drops the blank ones — established behaviour that suits a card and loses
    # evidence. So the whole transcription is kept here as text: every printed
    # label, every supplied value including a blank or clipped one, in printed
    # order. Nothing is generated, sorted, or repaired on the way in.
    conjugations = dict(getattr(candidate, "conjugations", None) or {})
    if conjugations:
        fields["source_conjugations"] = json.dumps(conjugations, ensure_ascii=False)
    # And, when the owner bound one, the exact request metadata that produced
    # that map: the identity, the revision and each column's ordinal, printed
    # witnesses and display label. Written beside the witness rather than
    # derived later, so a reviewer can see which identity each key answered
    # without consulting a job document that may since have been repointed.
    if layout is not None:
        fields["source_form_layout"] = json.dumps(
            layout.to_wire(), ensure_ascii=False, sort_keys=True
        )
    # The chapters the source itself printed, copied in printed order. Written
    # only when there are some: an empty list is the ordinary case and says
    # nothing. No tag is minted here — which labels become deck tags is a
    # decision made later, over the exact labels this preserves.
    chapters = list(getattr(candidate, "source_chapters", None) or [])
    if chapters:
        fields["source_chapters"] = json.dumps(chapters, ensure_ascii=False)
    return fields


def _require_bound_columns(
    candidates: Sequence[Any], prepared: PreparedInput, layout: TableLayout
) -> None:
    """Every returned key names a column the request supplied. Nothing else.

    An artifact-identifier check, not a language read: it compares key sets and
    never looks at a cell, a heading or a word. An unbound key would otherwise
    become a card row with no label binding and no witness, so it refuses for a
    person to settle rather than being dropped — a dropped key is a silent
    discard of something already paid for.

    A duplicate identity cannot survive ``dict[str, str]``, and a non-string
    key or value already refuses at the serialization boundary, so this is the
    one structural rule left. An answer that supplies no keys at all is a valid
    all-absent table: the declared columns are still the layout's.
    """
    allowed = frozenset(layout.column_ids)
    for candidate in candidates:
        supplied = getattr(candidate, "conjugations", None) or {}
        unknown = sorted(key for key in supplied if key not in allowed)
        if unknown:
            raise ExtractError(
                f"{prepared.origin_path.name}: the answer returned source-form "
                f"column id(s) {', '.join(repr(key) for key in unknown)} that "
                f"layout {layout.identity} never supplied. janki matches no "
                "printed label and will not guess which column was meant; "
                "settle it against the page.",
                code="extract-layout-unknown-column",
            )


def _source_forms_for(candidate: Any, layout: TableLayout) -> SourceFormsTable | None:
    """The canonical table one layout-bound candidate's answer makes.

    Declared columns in the layout's printed order, and the supplied cells
    verbatim: a present empty string is a printed blank and an omitted
    identity is absent. Nothing infers which column was meant, and nothing
    inspects a cell to decide whether it counts.
    """
    supplied = dict(getattr(candidate, "conjugations", None) or {})
    return SourceFormsTable.from_dict(
        {
            "columns": [
                {"id": column.column_id, "label": column.display_label}
                for column in layout.columns
            ],
            "cells": {
                column.column_id: supplied[column.column_id]
                for column in layout.columns
                if column.column_id in supplied
            },
        }
    )


def build_records(
    candidates: Iterable[Any],
    prepared: PreparedInput,
    known_ids: Iterable[str] = (),
    *,
    layout: TableLayout | None = None,
) -> RecordBuild:
    """Build canonical records and account for every parsed schema proposal.

    A candidate janki already holds is kept rather than dropped — a silent
    discard is a silent discard even when the word is a duplicate, and the
    reviewer may still want the example sentence off this page. Marking and
    sinking it puts the new words where the review effort should go.

    Two candidates in one answer can also mint the same deterministic ID.
    Only the first can be a canonical record, but no Japanese-aware merge can
    decide which parts of the answers are better. Every member of a collision
    group, including the first, is therefore returned in original response
    order for a reviewer to compare.

    ``layout`` is the frozen layout this answer's own request was sent with,
    read back from the saved provenance. When there is one, every returned key
    is checked against the identities that request supplied — over the whole
    parsed list, before any partition, because a duplicate or unusable
    proposal's unknown key is the same structural fault.
    """
    parsed = tuple(candidates)
    if layout is not None:
        _require_bound_columns(parsed, prepared, layout)
    plan = _candidate_plan(parsed)
    known = set(known_ids)
    fresh: list[VocabularyRecord] = []
    seen: list[VocabularyRecord] = []

    for candidate in plan.canonical_candidates:
        expression = str(getattr(candidate, "expression", "") or "").strip()
        reading = str(getattr(candidate, "reading", "") or "").strip()
        candidate_id = stable_record_id(expression, reading)
        content = shared_ai_schema.adapt_rich_card(candidate)
        raw_fields = _raw_fields(candidate, prepared, layout=layout)
        if content.romaji_rejected:
            raw_fields["ai_warnings"] = json.dumps(
                list(content.romaji_rejected), ensure_ascii=False
            )
        record = VocabularyRecord(
            id=candidate_id,
            expression=expression,
            reading=reading,
            meanings=list(content.meanings),
            part_of_speech=str(getattr(candidate, "part_of_speech", "") or "").strip(),
            examples=list(content.examples),
            # The columns the source printed, in the order it printed them.
            # Deciding which forms a word *has* is reading Japanese; copying
            # the ones a page supplies is not.
            #
            # ``VocabularyRecord`` has one canonical spelling for this field —
            # labels and values trimmed, an empty cell absent — and `repairs`
            # refuses a record that does not survive that round trip. Read the
            # canonical spelling from the model layer that owns the rule
            # rather than restating it here; the untouched transcription,
            # blank and clipped cells included, stays in
            # ``raw_fields["source_conjugations"]``.
            #
            # Left empty for a layout-bound candidate: its keys are opaque
            # column identities, not display labels, and the canonical table
            # below carries them with their bound labels and their blanks. The
            # computed map still exists as its own separate field, and a later
            # dictionary pass may fill it without touching the printed table.
            conjugations=(
                {}
                if layout is not None
                else VocabularyRecord.from_dict(
                    {
                        "id": candidate_id,
                        "expression": expression,
                        "reading": reading,
                        "conjugations": dict(
                            getattr(candidate, "conjugations", None) or {}
                        ),
                    }
                ).conjugations
            ),
            source_forms=(
                None if layout is None else _source_forms_for(candidate, layout)
            ),
            usage_notes=content.usage_notes,
            source=SourceReference(
                type="extract",
                imported_from=prepared.origin_path.name,
                raw_fields=raw_fields,
            ),
        )
        # Marked here, at the only moment the values are known to be model
        # output and nothing else: one step later they sit in a staging file
        # beside human edits and the distinction is unrecoverable. Dictionary
        # reconciliation reads the mark to know what it may replace.
        record = mark_provisional(record)
        # The rich response does not carry word-level romaji.  That value is a
        # deterministic derivation from the candidate's kana reading, outside
        # the shared model-answer adapter.  Run only that declaration here:
        # applying the whole ingest registry used to rewrite example notation
        # for fresh IDs while leaving the identical known-ID answer untouched.
        derived, _changes = repairs.apply_declarations(
            [record],
            repairs.REGISTRY.select(("record-romaji-from-reading",)),
            modes=frozenset({"ingest-safe"}),
        )
        record = derived[0]
        if record.id in known:
            seen.append(annotate(record, already_known=True))
        else:
            fresh.append(record)
    return RecordBuild(
        records=tuple(fresh + seen),
        unusable_candidates=plan.unusable_candidates,
        candidate_accounting=plan.accounting,
    )


def unusable_note(candidates: Sequence[Any]) -> str:
    """A review note naming what was held back, and where to look for it.

    This goes into the staging file rather than only onto the terminal. The
    staging file is the committed artifact a reviewer reads later, possibly on
    another clone; a count that exists only in scrollback is the same silent
    discard with an extra step. Everything the model *did* read about the row —
    its page, the verbatim line, the reading it managed — is recorded so the
    page can be re-checked rather than merely known to be incomplete.
    """
    if not candidates:
        return ""
    lines = [
        f"{len(candidates)} candidate(s) could not be stored: the model read no "
        "expression for them, and a record's ID is minted from its expression. "
        "Nothing was lost from the source — check these against the page and add "
        "them by hand if they are real."
    ]
    for candidate in candidates:
        lines.append("  - " + "; ".join(_describe(candidate)))
    return "\n".join(lines)


def _describe(candidate: Any) -> list[str]:
    """Every field the model filled in, as ``name: value`` parts.

    Read off the schema rather than a hand-written list of field names, so a
    field added to :func:`candidate_schema` later cannot start being silently
    dropped from these notes — which is the whole failure this note exists to
    prevent, one level down. ``expression`` is skipped because it is empty by
    definition here; empty fields are skipped because they say nothing.
    """
    parts: list[str] = []
    # Off the class, not the instance: pydantic deprecated the instance form.
    fields = getattr(type(candidate), "model_fields", None) or {}
    for name in fields:
        if name == "expression":
            continue
        value = getattr(candidate, name, None)
        if isinstance(value, list | tuple):
            text = ", ".join(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value if value is not None else "").strip()
        if text:
            parts.append(f"{name}: {text}")
    return parts or ["nothing but an empty row"]


def known_ids(
    records: Iterable[VocabularyRecord], *, scope_id: str = ""
) -> set[str]:
    """Every id a candidate could match, by identity as well as stored id.

    Both, because a hand-written record may carry an id that no longer matches
    what its expression and reading would mint today, and a candidate matching
    either one is a word janki already has.

    ``scope_id`` names *which* collection is being asked. The default is the
    shared one, and it deliberately does not see deck-scoped copies: a
    standalone deck holding 話す does not make 話す a word the shared collection
    already has. A caller extracting for one standalone deck passes that deck's
    scope and gets that scope's records instead.

    The reconstructed key stays an ordinary ``word:`` identity in both cases.
    Extraction has not chosen a destination when it mints candidate ids, so
    ordinary keys are what a candidate is actually compared against; scoping the
    reconstruction would compare a scoped key with a candidate that can never
    carry one, and every word would look new.
    """
    if scope_id:
        # Refused here rather than silently answering "nothing is known": a
        # scope nothing can match would mark every word in the source as new.
        validate_scope_id(scope_id)
    ids: set[str] = set()
    for record in records:
        if record_scope_id(record.id) != scope_id:
            continue
        ids.add(record.id)
        ids.add(stable_record_id(record.expression, record.reading))
    return ids


def staging_path(staging_dir: Path, source_name: str) -> Path:
    """Where one file's candidates land: ``<staging_dir>/<source name>.yaml``.

    The whole name, suffix included (``worksheet.pdf.yaml``), not the stem.
    ``worksheet.pdf`` and ``worksheet.jpg`` — a scan and a photo of the same
    page, a natural pairing — are two different sources with two different sets
    of candidates, and keying on the stem would have the second silently
    overwrite the first. DESIGN_V2 says ``<source-name>.yaml``; this is that.
    """
    return Path(staging_dir) / f"{Path(source_name).name}.yaml"


def staging_targets(
    staging_dir: Path,
    prepared: Sequence[PreparedInput],
    *,
    force: bool = False,
) -> list[Path]:
    """Every input's staging file, refusing a batch where two would collide.

    Checked up front, before a single API call, because the alternatives are
    both bad: with ``--force`` the second write silently destroys the first
    file's candidates, and without it the second fails with a diagnosis about
    "review edits you have not committed" that is wrong — the file it is
    refusing to touch was written seconds ago by this same run — after the
    extraction has already been paid for.

    The same file listed twice lands here too. :func:`inputs.prepare_inputs`
    keeps duplicates rather than discarding them silently; naming the problem
    is how that stays true without one input overwriting the other.
    """
    staging_root = Path(staging_dir)
    if staging_root.is_symlink() or (
        staging_root.exists() and not staging_root.is_dir()
    ):
        raise ExtractError(
            f"Refusing non-directory staging parent: {staging_root}. Move it "
            "aside before extracting a source.",
            code="extract-staging-parent-not-directory",
        )
    targets = [staging_path(staging_root, item.origin_path.name) for item in prepared]
    seen: dict[Path, str] = {}
    for target, item in zip(targets, prepared, strict=True):
        if target in seen:
            raise ExtractError(
                f"{seen[target]} and {item.origin_path.name} would both be written "
                f"to {target}. Extract them separately, or rename one — janki will "
                "not overwrite one file's candidates with another's.",
                code="extract-staging-target-collision",
            )
        seen[target] = item.origin_path.name
        occupied = target.exists() or target.is_symlink()
        if occupied and (target.is_symlink() or not target.is_file()):
            raise ExtractError(
                f"Refusing non-regular staging target: {target}. Move it aside "
                "before extracting this source.",
                code="extract-staging-target-not-regular",
            )
        if occupied and not force:
            raise ExtractError(
                f"Staging file already exists: {target}. It may hold review edits "
                "you have not committed; move it aside or re-run with force to "
                "overwrite.",
                code="extract-staging-exists",
            )
    return targets
