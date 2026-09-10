"""Shared structured-output shapes for janki's model-backed passes.

``extract`` reads source material and ``enrich --ai`` reads a bare vocabulary
record. ``revise`` reads an explicitly selected existing card or deck and the
owner's exact requested change. Extraction returns a complete
candidate card, including for an already-known identity, and therefore requires
both card slots; enrichment may preserve a reviewed example and return only the
unoccupied slot, so only its example-list cardinality is flexible. Revision's
conjugation-deck shape uses the drill artifact's narrower example fields and
requires both slots for every selected record, but cannot express identity,
deck, source, approval, audio, or romaji decisions.
Keeping the model classes behind functions preserves the project's optional
``ai`` dependency — importing janki for a build must not import Pydantic.

The field descriptions are intentionally terse.  They are serialized into the
JSON schema and therefore reach the model, but the substantive instructions
belong in ``prompts/`` where a person can read and edit the whole asking.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "assistant_agent_schema",
    "card_revision_schema",
    "conjugation_deck_revision_schema",
    "generated_example_schema",
    "adapt_rich_card",
    "extraction_rich_card_schema",
    "RichCardContent",
    "rich_card_schema",
    "source_pattern_schema",
]


@dataclass(frozen=True, slots=True)
class RichCardContent:
    """Canonical card content decoded from either rich model response."""

    meanings: tuple[str, ...]
    examples: tuple[Any, ...]
    usage_notes: str
    romaji_rejected: tuple[str, ...]


@functools.cache
def assistant_agent_schema() -> Any:
    """Repository-aware Assistant prose plus closed application intents."""
    from typing import Annotated

    from pydantic import BaseModel, ConfigDict, Field, StringConstraints

    NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    class AssistantActionOptions(BaseModel):
        """Closed scalar choices needed by action-specific local planners."""

        model_config = ConfigDict(extra="forbid")

        deck_name: NonBlank | None = Field(
            default=None,
            description=(
                "Exact learner-facing name for create_deck, or for a new "
                "destination deck in add_kanji_notes."
            ),
        )
        card_directions: list[
            Literal["recognition", "production", "reading"]
        ] = Field(
            default_factory=list,
            description=(
                "Explicit owner-selected directions for create_deck and "
                "add_kanji_notes; never infer an omitted direction."
            ),
        )
        deck_scope: Literal["shared", "standalone"] | None = Field(
            default=None,
            description=(
                "create_deck only: 'standalone' when the owner asked for their "
                "own independent copies of the words, 'shared' when they asked "
                "to reuse the existing cards; null when they have not chosen."
            ),
        )
        study_type: Literal["vocabulary", "kanji"] | None = Field(
            default=None,
            description=(
                "Content type the owner named; null when explicit targets "
                "already settle it."
            ),
        )
        kanji_characters: list[NonBlank] = Field(
            default_factory=list,
            description=(
                "Single characters copied exactly from the owner's message for "
                "add_kanji_notes; never expanded, ordered, or invented."
            ),
        )
        refresh_readings: Literal[True, False] | None = Field(
            default=None,
            description=(
                "True only when the owner's current message explicitly asks to "
                "refresh saved reading facts; never infer it."
            ),
        )
        production_cues: list[NonBlank] = Field(
            default_factory=list,
            description=(
                "Owner-written 'character=cue' entries for a production "
                "direction; never compose a cue for them."
            ),
        )
        review_patterns: bool | None = Field(
            default=None,
            description=(
                "Explicit owner choice for source-extraction review_staging; false "
                "or null for card-revision staging, which has no pattern review."
            ),
        )
        destination_resource_id: NonBlank | None = Field(
            default=None,
            description="Exact destination deck resource for assign_cards only.",
        )
        audio_words: Literal[True, False] | None = Field(
            default=None,
            description=(
                "Explicit word-clip choice for generate_audio. Supply together with "
                "audio_examples when the owner names clip classes; otherwise null."
            ),
        )
        audio_examples: Literal[True, False] | None = Field(
            default=None,
            description=(
                "Explicit example-clip choice for generate_audio. Supply together "
                "with audio_words when the owner names clip classes; otherwise null."
            ),
        )
        audio_force: Literal[True, False] | None = Field(
            default=None,
            description=(
                "True only when the owner's current message explicitly requests "
                "regeneration of selected audio; never infer it."
            ),
        )
        audio_prune: Literal[True, False] | None = Field(
            default=None,
            description=(
                "True only when the owner's current message explicitly requests "
                "repository-wide deletion of unreferenced janki audio; never infer it."
            ),
        )
        operation_action: Literal[
            "recover", "show_reply", "end", "forget"
        ] | None = Field(
            default=None,
            description="Explicit manage_operation sub-action.",
        )
        operation_id: NonBlank | None = Field(
            default=None,
            description=(
                "Exact disclosed paid-operation id for manage_operation, for "
                "inspecting or recovering one child of the named study job's "
                "own batch, or — with retry_study_parts — any one call of the "
                "batch whose children the owner asked to send again. Never for "
                "any other action."
            ),
        )
        accept_paid_output_loss: bool | None = Field(
            default=None,
            description=(
                "Explicit owner acceptance for a forced operation forget; null "
                "when not stated."
            ),
        )
        deletion_kind: Literal["staged_cards", "canonical_cards", "deck"] | None = (
            Field(
                default=None,
                description="Exact delete_content class; never infer from a target.",
            )
        )
        new_expression: NonBlank | None = Field(
            default=None,
            description=(
                "Exact owner-supplied expression for reidentify_staged_card only."
            ),
        )
        new_reading: NonBlank | None = Field(
            default=None,
            description=(
                "Exact owner-supplied reading for reidentify_staged_card only."
            ),
        )
        coverage_reason: NonBlank | None = Field(
            default=None,
            description=(
                "Owner's explicit reason for approve_coverage; never generate one."
            ),
        )
        search_literal: NonBlank | None = Field(
            default=None,
            description="Exact case-sensitive literal for search_cards only.",
        )
        search_limit: int | None = Field(
            default=None,
            ge=1,
            le=20,
            description="Maximum search_cards results; null uses Janki's default.",
        )
        concurrency_limit: int | None = Field(
            default=None,
            ge=1,
            le=4,
            description=(
                "How many sources extract_batch or extract_study_parts reads "
                "at once; null uses 2. Never set for any other action."
            ),
        )
        retry_child_indices: list[int] = Field(
            default_factory=list,
            description=(
                "retry_study_parts only: the exact 1-based child numbers the "
                "owner asked to send again. Never inferred, never widened, and "
                "never a successful or in-flight child."
            ),
        )

    class AssistantActionIntent(BaseModel):
        model_config = ConfigDict(extra="forbid")

        kind: Literal[
            "enrich_cards",
            "revise_cards",
            "revise_deck",
            "create_deck",
            "add_kanji_notes",
            "assign_cards",
            "review_staging",
            "reidentify_staged_card",
            "approve_coverage",
            "promote_staging",
            "generate_audio",
            "build_deck",
            "extract_source",
            "extract_batch",
            "open_source_part_editor",
            # Study jobs. Each of these reads, plans for one owner
            # confirmation, or dispatches work the owner already authorized
            # exactly. None of them mints or selects an owner decision:
            # creating a job, publishing regions, editing a choice, choosing
            # among competing captured proposals and every review, coverage or
            # disposition decision are owner controls that carry no
            # model-emittable field at all.
            "study_job_status",
            "inspect_capture_proposals",
            "extract_study_parts",
            "retry_study_parts",
            "stage_capture_proposal",
            "resume_study_job",
            "delete_content",
            "manage_operation",
            "inspect_resources",
            "search_cards",
            "preview_cards",
        ] = Field(description="One closed Janki application operation to plan.")
        resource_ids: list[NonBlank] = Field(
            description="Exact opaque repository resource ids needed by the operation."
        )
        record_ids: list[NonBlank] = Field(
            description="Exact canonical record ids needed by the operation."
        )
        instruction: NonBlank = Field(
            description="The owner's requested outcome, without invented approval."
        )
        options: AssistantActionOptions = Field(
            default_factory=AssistantActionOptions,
            description="Closed action-specific owner choices; empty when unused.",
        )

    class AssistantAgentAnswer(BaseModel):
        model_config = ConfigDict(extra="forbid")

        answer: NonBlank = Field(description="Concise Markdown answer to the owner.")
        action_intents: list[AssistantActionIntent] = Field(
            max_length=1,
            description=(
                "At most one typed application plan explicitly requested by the "
                "owner; empty for questions or unavailable exact targets."
            )
        )

    return AssistantAgentAnswer


@functools.cache
def card_revision_schema() -> Any:
    """Complete field replacements for one explicit existing-card revision."""
    from typing import Annotated

    from pydantic import BaseModel, ConfigDict, Field, StringConstraints

    NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    class CardFieldUpdate(BaseModel):
        model_config = ConfigDict(extra="forbid")

        field: Literal[
            "furigana",
            "romaji",
            "meanings",
            "part_of_speech",
            "verb_group",
            "transitivity",
            "examples",
            "conjugations",
            "usage_notes",
            "audio",
            "image",
            "pitch_accent",
            "audio_accent",
            "frequency_rank",
        ] = Field(description="One non-identity canonical card field.")
        value_json: NonBlank = Field(
            description="JSON for the complete proposed canonical field value."
        )

    class CardChange(BaseModel):
        model_config = ConfigDict(extra="forbid")

        record_id: NonBlank = Field(description="Exact selected canonical record id.")
        reason: NonBlank = Field(description="Short owner-facing reason for the change.")
        updates: list[CardFieldUpdate] = Field(
            min_length=1,
            description="Complete replacement values for changed fields only.",
        )

    class CardRevisionAnswer(BaseModel):
        model_config = ConfigDict(extra="forbid")

        summary: NonBlank = Field(description="Concise summary for owner review.")
        card_changes: list[CardChange] = Field(
            min_length=1,
            description="One or more explicit existing-card field changes.",
        )

    return CardRevisionAnswer


def adapt_rich_card(value: Any) -> RichCardContent:
    """Apply the one structural decoding contract shared by both AI paths.

    This does not judge Japanese.  It trims wire strings, removes duplicate
    gloss entries, maps the schema's speech-level label to the card field, and
    settles the mechanically derived romaji against the reading stated by the
    answer's furigana.  Source routing and whether an identity is already known
    must not change any of those results.
    """
    from japanese_anki import qc
    from japanese_anki.models import ExampleSentence

    meanings = tuple(
        dict.fromkeys(
            text
            for item in (getattr(value, "meanings", []) or [])
            if (text := str(item or "").strip())
        )
    )
    examples: list[ExampleSentence] = []
    rejected: list[str] = []
    for item in getattr(value, "examples", []) or []:
        japanese = str(getattr(item, "japanese", "") or "").strip()
        if not japanese:
            continue
        example = ExampleSentence(
            japanese=japanese,
            furigana=str(getattr(item, "furigana", "") or "").strip(),
            romaji=str(getattr(item, "romaji", "") or "").strip(),
            english=str(getattr(item, "english", "") or "").strip(),
            register=str(getattr(item, "speech_level", "") or "").strip().lower(),
        )
        settled, warning = qc.settle_example_romaji(example)
        examples.append(settled)
        if warning:
            rejected.append(warning)
    return RichCardContent(
        meanings=meanings,
        examples=tuple(examples),
        usage_notes=str(getattr(value, "usage_notes", "") or "").strip(),
        romaji_rejected=tuple(rejected),
    )


@functools.cache
def generated_example_schema() -> Any:
    """The vocabulary example shape shared by extraction and enrichment."""
    from typing import Annotated

    from pydantic import BaseModel, ConfigDict, Field, StringConstraints

    NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    class GeneratedExample(BaseModel):
        model_config = ConfigDict(extra="forbid")

        japanese: NonBlank = Field(description="Japanese sentence.")
        speech_level: Literal["polite", "casual"] = Field(
            description="Speech level."
        )
        furigana: NonBlank = Field(description="Sentence with Anki furigana.")
        romaji: NonBlank = Field(description="Sentence in Hepburn romaji.")
        english: NonBlank = Field(description="Natural English translation.")

    return GeneratedExample


@functools.cache
def rich_card_schema() -> Any:
    """Complete values common to source and bare-word card answers."""
    from typing import Annotated

    from pydantic import BaseModel, ConfigDict, Field, StringConstraints

    GeneratedExample = generated_example_schema()
    NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    class RichCard(BaseModel):
        model_config = ConfigDict(extra="forbid")

        meanings: list[NonBlank] = Field(
            min_length=1,
            description="English glosses.",
        )
        examples: list[GeneratedExample]
        usage_notes: str = Field(description="Usage note.")

    return RichCard


@functools.cache
def extraction_rich_card_schema() -> Any:
    """The fixed-cardinality variant used only while extracting a source.

    The shared shape already requires nonblank meanings and complete returned
    example values. Extraction additionally returns a complete candidate card,
    whether its identity is fresh or already known, so it must fill exactly the
    two polite/casual slots the note type renders. These are artifact
    constraints, not Japanese judgements; the Markdown template still does all
    language work, including choosing and writing those values.

    This is separate from :func:`rich_card_schema` because ``enrich --ai`` can
    receive an already-reviewed polite or casual example that it must preserve.
    Its legitimate response may therefore contain only the unoccupied slot.
    """
    from pydantic import Field, model_validator

    SharedRichCard = rich_card_schema()
    GeneratedExample = generated_example_schema()

    class RichCard(SharedRichCard):
        examples: list[GeneratedExample] = Field(min_length=2, max_length=2)

        @model_validator(mode="after")
        def has_one_example_for_each_card_slot(self) -> Any:
            levels = [example.speech_level for example in self.examples]
            if levels.count("polite") != 1 or levels.count("casual") != 1:
                raise ValueError(
                    "examples must contain exactly one polite and one casual value"
                )
            return self

    return RichCard


@functools.cache
def conjugation_deck_revision_schema() -> Any:
    """Owner-requested rich examples for selected conjugation-drill cards.

    The response contract exposes only the two content fields this pass may
    propose: a deck-wide form note and the polite/casual examples for named
    record IDs. The application layer compares those IDs with the exact
    selection sent in the request; keeping paths, identities, approvals and
    audio out of this schema makes it impossible for a model answer to decide
    any of them.
    """
    from typing import Annotated

    from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

    NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

    class RevisionExample(BaseModel):
        model_config = ConfigDict(extra="forbid")

        japanese: NonBlank = Field(description="Japanese sentence.")
        speech_level: Literal["polite", "casual"] = Field(
            description="Speech level."
        )
        furigana: NonBlank = Field(description="Sentence with Anki furigana.")
        english: NonBlank = Field(description="Natural English translation.")

    class RevisionCard(BaseModel):
        model_config = ConfigDict(extra="forbid")

        record_id: NonBlank = Field(description="Selected record identifier.")
        examples: list[RevisionExample] = Field(min_length=2, max_length=2)

        @model_validator(mode="after")
        def has_one_example_for_each_card_slot(self) -> Any:
            levels = [example.speech_level for example in self.examples]
            if levels.count("polite") != 1 or levels.count("casual") != 1:
                raise ValueError(
                    "examples must contain exactly one polite and one casual value"
                )
            return self

    class ConjugationDeckRevision(BaseModel):
        model_config = ConfigDict(extra="forbid")

        form_note: str = Field(description="Proposed deck-wide form note.")
        cards: list[RevisionCard] = Field(min_length=1)

        @model_validator(mode="after")
        def has_unique_record_ids(self) -> Any:
            identifiers = [card.record_id for card in self.cards]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("cards must contain each record_id exactly once")
            return self

    return ConjugationDeckRevision


@functools.cache
def source_pattern_schema() -> Any:
    """One grammar or usage pattern reported while a source is already open."""
    from pydantic import BaseModel, ConfigDict, Field

    class SourcePattern(BaseModel):
        model_config = ConfigDict(extra="forbid")

        template: str = Field(description="Pattern template.")
        gloss: str = Field(default="", description="English gloss.")
        examples: list[str] = Field(default_factory=list, description="Source examples.")
        where: str = Field(default="", description="Source location.")

    return SourcePattern
