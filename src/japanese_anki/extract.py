"""Reading vocabulary off a PDF or a photo, into a file a human then reviews.

``janki extract`` is the one command that *guesses*. Everything else in this
project is a rule over data someone already wrote down; this reads a textbook
page or a photograph of a whiteboard and proposes what the words might be. So
nothing it produces goes near ``vocabulary.json``: every candidate lands in
``data/staging/`` with its page number, the line it was found on, and the
model's own confidence, and a person decides what is real.

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
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from japanese_anki import claude_client
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import stable_record_id
from japanese_anki.inputs import PreparedInput
from japanese_anki.models import SourceReference, VocabularyRecord
from japanese_anki.staging import annotate

__all__ = [
    "CONFIDENCE_LEVELS",
    "MODES",
    "ExtractError",
    "build_records",
    "candidate_schema",
    "extract_candidates",
    "prompt_for",
    "staging_path",
    "staging_targets",
    "system_prompt",
    "unusable",
    "unusable_note",
]

#: ``--mode`` values. Omitting the flag lets the model judge each page for
#: itself, which DESIGN_V2 makes the default because one PDF often holds both.
MODES: tuple[str, ...] = ("table", "prose")

#: What a candidate's ``confidence`` may say. Ordered worst-last so a reviewer
#: reading top to bottom meets the shakiest guesses first.
CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low")


class ExtractError(JankiError):
    pass


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

    from pydantic import BaseModel, Field

    class CandidateRecord(BaseModel):
        expression: str = Field(description="The word as written, in Japanese.")
        reading: str = Field(
            default="",
            description=(
                "The reading in kana. Leave empty if the source does not give "
                "one and you are not certain — do not guess."
            ),
        )
        meanings: list[str] = Field(
            default_factory=list, description="English meanings, one per sense."
        )
        part_of_speech: str = Field(default="", description="Part of speech, if known.")
        example: str = Field(
            default="", description="An example sentence from the source, if there is one."
        )
        page: int = Field(default=0, description="1-indexed page this was read from.")
        context: str = Field(
            default="", description="The line or cell this was read from, verbatim."
        )
        confidence: Literal["high", "medium", "low"] = Field(
            default="medium",
            description="How sure you are that this reading and meaning are right.",
        )
        inclusion_reason: str = Field(
            default="",
            description="In prose mode, why this word is worth a card.",
        )

    class Extraction(BaseModel):
        candidates: list[CandidateRecord] = Field(default_factory=list)

    return Extraction


_TABLE_RULES = """\
This page is a vocabulary list or table. Transcribe it faithfully: every row is
a candidate, in the order it appears. Do not add words that are not on the
page, do not merge rows, and do not correct what the source says — if a reading
looks wrong, transcribe it and mark the candidate low confidence."""

_PROSE_RULES = """\
This page is running text. Pick out the vocabulary worth making a card for and
say why in inclusion_reason. Skip words that are trivially common, and skip any
word in the known-words list below. Quote the sentence you found each word in
as its context."""

_AUTO_RULES = """\
Judge each page for itself: a vocabulary list or table is transcribed
faithfully row by row, and running text is mined for the words worth a card.
One document may contain both."""

_ALWAYS = """\
Report the page number and the verbatim line each candidate came from, so a
human can find it again. Never invent a reading: if the source does not give
one and you are not certain, leave reading empty and let the review supply it —
an invented reading becomes a permanent, uncorrectable record ID. Mark your own
confidence honestly; "low" is a useful answer and a wrong "high" is not."""


def system_prompt(mode: str | None) -> str:
    """The instructions for one extraction mode."""
    if mode is None:
        rules = _AUTO_RULES
    elif mode == "table":
        rules = _TABLE_RULES
    elif mode == "prose":
        rules = _PROSE_RULES
    else:
        raise ExtractError(
            f"Unknown mode '{mode}'. Use one of: {', '.join(MODES)}, or omit "
            "--mode to let the model judge each page."
        )
    return (
        "You are reading Japanese study material and proposing vocabulary "
        "records for a human to review.\n\n" + rules + "\n\n" + _ALWAYS
    )


def prompt_for(source_name: str, known: Sequence[str] = ()) -> str:
    """The user-turn text for one file.

    The known-word list rides here rather than in the system blocks on purpose:
    it changes every time the collection grows, and anything above the cache
    breakpoint that changes invalidates the cached style guide for every run.
    """
    lines = [f"Extract vocabulary from {source_name}."]
    if known:
        lines.append(
            "\nWords janki already has — skip these unless the page says "
            "something new about them:\n" + "、".join(known)
        )
    return "\n".join(lines)


def extract_candidates(
    prepared: PreparedInput,
    *,
    model: str,
    style_guide: str,
    mode: str | None = None,
    known: Sequence[str] = (),
    client: Any | None = None,
) -> list[Any]:
    """The candidates one file yields, or an error explaining why none did.

    Both incomplete outcomes are refused rather than salvaged. A refusal is
    reported with the category the API gave, because "it was declined" without
    a reason leaves a user with nothing to act on. A ``max_tokens`` stop means
    the answer was cut mid-word: the visible half would look like a complete
    extraction and the rest would be lost silently, which is the one failure
    this whole command is arranged to avoid.
    """
    parsed, stop_reason, refusal = claude_client.parse_call(
        model,
        claude_client.system_blocks(style_guide, system_prompt(mode)),
        [prepared.content_block(), {"type": "text", "text": prompt_for(
            prepared.origin_path.name, known
        )}],
        candidate_schema(),
        client,
    )

    if stop_reason == "refusal":
        detail = ""
        if refusal is not None:
            detail = f" ({refusal.category}" + (
                f": {refusal.explanation})" if refusal.explanation else ")"
            )
        raise ExtractError(
            f"{model} declined to read {prepared.origin_path.name}{detail}. "
            "Nothing was written."
        )
    if stop_reason == "max_tokens":
        raise ExtractError(
            f"{model} ran out of room part-way through {prepared.origin_path.name}, "
            "so the answer is cut off and janki will not write a half-read file. "
            "Give it less to read at once: split a long document and run the parts "
            "separately, or crop a dense photo to the section you want."
        )
    if parsed is None:
        raise ExtractError(
            f"{model} returned nothing usable for {prepared.origin_path.name} "
            f"(stop reason: {stop_reason}). Nothing was written."
        )
    return list(parsed.candidates)


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

    for candidate in candidates:
        expression = str(getattr(candidate, "expression", "") or "").strip()
        if not expression:
            continue
        reading = str(getattr(candidate, "reading", "") or "").strip()
        example = str(getattr(candidate, "example", "") or "").strip()
        record = VocabularyRecord(
            id=stable_record_id(expression, reading),
            expression=expression,
            reading=reading,
            meanings=[
                text
                for item in getattr(candidate, "meanings", []) or []
                if (text := str(item).strip())
            ],
            part_of_speech=str(getattr(candidate, "part_of_speech", "") or "").strip(),
            examples=_examples(example),
            source=SourceReference(
                type="extract",
                imported_from=prepared.origin_path.name,
                raw_fields=_raw_fields(candidate, prepared),
            ),
        )
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


def _examples(japanese: str) -> list[Any]:
    if not japanese:
        return []
    from japanese_anki.models import ExampleSentence

    return [ExampleSentence(japanese=japanese)]


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
    staging_dir: Path, prepared: Sequence[PreparedInput]
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
                "not overwrite one file's candidates with another's."
            )
        seen[target] = item.origin_path.name
    return targets
