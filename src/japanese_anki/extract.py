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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from japanese_anki import ai_schema as shared_ai_schema
from japanese_anki import claude_client, patterns, prompts, repairs
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import stable_record_id
from japanese_anki.inputs import PreparedInput
from japanese_anki.models import (
    SourceReference,
    VocabularyRecord,
    mark_provisional,
)
from japanese_anki.staging import annotate

__all__ = [
    "CONFIDENCE_LEVELS",
    "EXTRACTION_SCHEMA_VERSION",
    "MODES",
    "SOURCE_UNIT_DISPOSITIONS",
    "ExtractError",
    "ExtractionResult",
    "SourceUnit",
    "build_records",
    "candidate_schema",
    "context_fingerprint",
    "coverage_block",
    "extract_candidates",
    "normalize_context",
    "normalize_response",
    "prompt_provenance",
    "prompt_for",
    "source_fingerprint",
    "staging_path",
    "staging_targets",
    "prompt_name",
    "unusable",
    "unusable_note",
]

#: ``--mode`` values. Omitting the flag lets the model judge each page for
#: itself, which DESIGN_V2 makes the default because one PDF often holds both.
MODES: tuple[str, ...] = ("table", "prose")

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
EXTRACTION_SCHEMA_VERSION = 3


class ExtractError(JankiError):
    """A deterministic extraction failure with a stable case identity."""

    def __init__(self, message: str, *, code: str = "extract-error") -> None:
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()}"


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
    from typing import Literal

    from pydantic import BaseModel, ConfigDict, Field

    RichCard = shared_ai_schema.rich_card_schema()
    SourcePattern = shared_ai_schema.source_pattern_schema()

    class CandidateRecord(RichCard):
        model_config = ConfigDict(extra="forbid")

        expression: str = Field(description="The word as written, in Japanese.")
        reading: str = Field(
            default="",
            description="Kana reading.",
        )
        part_of_speech: str = Field(default="", description="Part of speech, if known.")
        page: int = Field(default=0, description="1-indexed page this was read from.")
        context: str = Field(
            default="", description="The line or cell this was read from, verbatim."
        )
        confidence: Literal["high", "medium", "low"] = Field(
            default="medium",
            description="Confidence.",
        )
        inclusion_reason: str = Field(
            default="",
            description="In prose mode, why this word is worth a card.",
        )
        source_kind: Literal["table", "prose"] = Field(
            default="prose",
            description=(
                "Whether this candidate comes from a table/list source unit or "
                "from prose selection."
            ),
        )
        section: str = Field(
            default="",
            description=(
                "Stable lowercase section slug. Required for a table candidate."
            ),
        )
        ordinal: int = Field(
            default=0,
            ge=0,
            description=(
                "One-based row ordinal within the section. Required for a table "
                "candidate."
            ),
        )

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


def prompt_for(
    source_name: str,
    known: Sequence[str] = (),
) -> str:
    """The user-turn text for one file.

    The known-word list rides here rather than in the system blocks on purpose:
    it changes every time the collection grows, and anything above the cache
    breakpoint that changes invalidates the cached style guide for every run.

    Two other blocks used to ride here — approved source-unit keys, and
    approved prose-selection targets with their rubric — binding a human's
    inventory of a page into the prompt so coverage could be scored against it.
    That was the pilot programme's question, and it was cancelled with it
    (M8.4). What is left is the ordinary ask.
    """
    lines = [f"Source file: {source_name}"]
    if known:
        lines.append("\nKnown expressions:\n" + "、".join(known))
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


def prompt_provenance(
    prepared: PreparedInput,
    *,
    model: str,
    style_guide: str,
    system: str,
    mode: str | None,
    known: Sequence[str] = (),
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """The stable inputs needed to explain a later model-output change."""
    user = prompt_for(prepared.origin_path.name, known)
    schema = candidate_schema()
    wire_schema = claude_client.wire_schema(schema)
    return {
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


def extract_candidates(
    prepared: PreparedInput,
    *,
    model: str,
    style_guide: str,
    system: str,
    mode: str | None = None,
    known: Sequence[str] = (),
    client: Any | None = None,
) -> ExtractionResult:
    """The normalized response one file yields, or a stable diagnostic.

    Both incomplete outcomes are refused rather than salvaged. A refusal is
    reported with the category the API gave, because "it was declined" without
    a reason leaves a user with nothing to act on. A ``max_tokens`` stop means
    the answer was cut mid-word: the visible half would look like a complete
    extraction and the rest would be lost silently, which is the one failure
    this whole command is arranged to avoid.
    """
    try:
        parsed, stop_reason, refusal = claude_client.parse_call(
            model,
            claude_client.system_blocks(style_guide, system),
            [
                prepared.content_block(),
                {
                    "type": "text",
                    "text": prompt_for(prepared.origin_path.name, known),
                },
            ],
            candidate_schema(),
            client,
            effort=claude_client.effort_for(model),
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
        raise ExtractError(
            f"{prepared.origin_path.name}: {exc}",
            code="extract-model-call-failed",
        ) from exc

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
    provenance = prompt_provenance(
        prepared,
        model=model,
        style_guide=style_guide,
        system=system,
        mode=mode,
        known=known,
        source_sha256=prepared.source_sha256 or None,
    )
    return ExtractionResult(
        candidates=result.candidates,
        source_units=result.source_units,
        model_reported_unit_count=result.model_reported_unit_count,
        pattern_set=patterns.with_prompt_provenance(
            result.pattern_set, provenance
        ),
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
    if mode == "table" and prose_candidates:
        raise ExtractError(
            f"{source_name}: table mode returned {len(prose_candidates)} prose "
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
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def coverage_block(
    result: ExtractionResult,
    *,
    source_sha256: str,
    mode: str | None,
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
    has_table = bool(actual) or mode == "table"
    status = "unmeasured" if has_table else "selection"

    dispositions = {
        name.replace("-", "_") + "_units": sorted(
            [unit.fact() for unit in actual if unit.disposition == name],
            key=_unit_sort,
        )
        for name in SOURCE_UNIT_DISPOSITIONS
    }
    block: dict[str, Any] = {
        "version": 1,
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
    block["coverage_block_fingerprint"] = _canonical_fingerprint(block)
    return block


def _raw_fields(candidate: Any, prepared: PreparedInput) -> dict[str, str]:
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
    return fields


def build_records(
    candidates: Iterable[Any],
    prepared: PreparedInput,
    known_ids: Iterable[str] = (),
) -> list[VocabularyRecord]:
    """Candidates as records, already-known ones marked and sorted last.

    A candidate janki already holds is kept rather than dropped — a silent
    discard is a silent discard even when the word is a duplicate, and the
    reviewer may still want the example sentence off this page. Marking and
    sinking it puts the new words where the review effort should go.
    """
    known = set(known_ids)
    fresh: list[VocabularyRecord] = []
    seen: list[VocabularyRecord] = []
    produced: set[str] = set()

    for candidate in candidates:
        expression = str(getattr(candidate, "expression", "") or "").strip()
        if not expression:
            continue
        reading = str(getattr(candidate, "reading", "") or "").strip()
        content = shared_ai_schema.adapt_rich_card(candidate)
        raw_fields = _raw_fields(candidate, prepared)
        if content.romaji_rejected:
            raw_fields["ai_warnings"] = json.dumps(
                list(content.romaji_rejected), ensure_ascii=False
            )
        record = VocabularyRecord(
            id=stable_record_id(expression, reading),
            expression=expression,
            reading=reading,
            meanings=list(content.meanings),
            part_of_speech=str(getattr(candidate, "part_of_speech", "") or "").strip(),
            examples=list(content.examples),
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
        # Repeated source rows stay visible as separate source units. They do
        # not become duplicate canonical notes with the same deterministic ID.
        if record.id in produced:
            continue
        produced.add(record.id)
        if record.id in known:
            seen.append(annotate(record, already_known=True))
        else:
            fresh.append(record)
    return fresh + seen


def unusable(candidates: Iterable[Any]) -> list[Any]:
    """Candidates that cannot become records: nothing to mint an id from.

    A record's id is minted from its expression, so a candidate without one has
    no identity and cannot be stored — even when the model read a reading, a
    gloss, and a page number off the row. That happens for real: a table row
    whose kanji cell is smudged still yields its kana column and its English.
    """
    return [
        candidate
        for candidate in candidates
        if not str(getattr(candidate, "expression", "") or "").strip()
    ]


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
        # `page` uses 0 for "unknown", so suppress the sentinel by *field*, not
        # by rendered text: a filter on the string "0" would also swallow a
        # meaning of "0" or a context cell reading "0", which is the silent
        # drop this note exists to prevent.
        if name == "page" and not value:
            continue
        if isinstance(value, list | tuple):
            text = ", ".join(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value if value is not None else "").strip()
        if text:
            parts.append(f"{name}: {text}")
    return parts or ["nothing but an empty row"]


def known_ids(records: Iterable[VocabularyRecord]) -> set[str]:
    """Every id a candidate could match, by identity as well as stored id.

    Both, because a hand-written record may carry an id that no longer matches
    what its expression and reading would mint today, and a candidate matching
    either one is a word janki already has.
    """
    ids: set[str] = set()
    for record in records:
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
    targets = [staging_path(staging_dir, item.origin_path.name) for item in prepared]
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
        if target.exists() and not force:
            raise ExtractError(
                f"Staging file already exists: {target}. It may hold review edits "
                "you have not committed; move it aside or re-run with force to "
                "overwrite.",
                code="extract-staging-exists",
            )
    return targets
