"""Asking a model whether an extraction accounted for the page it read.

The bookkeeping question, and only that one: does janki's record of what a page
contained match the page? Every vocabulary entry present and filed under a
defensible disposition, or something missing.

**Why this is not the review subsystem coming back.** M8.2 deleted a pass that
read finished cards and judged their Japanese — whether a sentence was natural,
whether a gloss was right — because quality belongs to the templates that ask
for it. This asks nothing about Japanese. It counts: is everything on the page
in the record. DESIGN.md puts artifact structure — identifiers, counts, file
shape, provenance — squarely in janki's business, and "did we read the whole
page" is a count that happens to need eyes. `prompts/approve-coverage.md` says
so in the prompt itself, and says what it is *not* judging, because a model
handed a page will otherwise volunteer opinions about the glosses.

**The model sees the page.** Not a transcription of it — the same base64
image or PDF the extraction pass was given, re-prepared from the durable inbox
and checked against the fingerprint recorded at extraction time. Asking a model
to confirm a reading of a page it cannot see would be asking it to agree with
itself.

**Whose approval this is.** The owner's, moved up a level: from "is this page
accounted for" to "may a model answer that". The approval it writes records
`authority: model` with the model id and the prompt's fingerprint, so a card
promoted on a model's word says so permanently and a reader can tell which
asking produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from japanese_anki import claude_client, prompts
from japanese_anki.errors import JankiError
from japanese_anki.inputs import PreparedInput

__all__ = [
    "CoverageVerdict",
    "CoverageReviewError",
    "format_account",
    "review_coverage",
    "verdict_schema",
]


class CoverageReviewError(JankiError):
    """The coverage pass could not produce a usable verdict."""


@dataclass(frozen=True, slots=True)
class CoverageVerdict:
    approved: bool
    reason: str
    model: str
    prompt_fingerprint: str


def verdict_schema() -> Any:
    """The structured answer: a decision and the sentence explaining it."""
    from pydantic import BaseModel, Field

    class Verdict(BaseModel):
        approved: bool = Field(
            description=(
                "True when every vocabulary entry visible on the page appears "
                "in the record under a defensible disposition. False when "
                "something is missing or plainly misfiled."
            )
        )
        reason: str = Field(
            description=(
                "Two or three sentences for a person reading this months "
                "later: what the page showed, what the record claimed, and "
                "where they met or did not. If refusing, name the specific "
                "entry or location at fault first."
            )
        )

    return Verdict


def format_account(block: dict[str, Any]) -> str:
    """janki's record of the page, as the model will read it.

    One line per source unit in page order, carrying exactly what the
    extraction claimed: where it is, what it says, how it was filed, and why
    when the filing needs a reason. The verbatim text is what lets the model
    line the record up against what it sees, so it is never truncated here.
    """
    units = block.get("source_units") or []
    lines = [
        f"janki recorded {len(units)} item(s) on this page, "
        f"{sum(1 for u in units if u.get('disposition') == 'candidate')} of which "
        "became flashcards.",
        "",
    ]
    for unit in units:
        where = f"page {unit.get('page')} · {unit.get('section')} · #{unit.get('ordinal')}"
        lines.append(f"[{unit.get('disposition')}] {where}")
        lines.append(f"    text: {unit.get('context', '')}")
        if unit.get("reason"):
            lines.append(f"    janki's reason: {unit['reason']}")
    return "\n".join(lines)


def review_coverage(
    prepared: PreparedInput,
    block: dict[str, Any],
    *,
    model: str,
    instructions: str,
    client: Any | None = None,
    parse_call: Any | None = None,
) -> CoverageVerdict:
    """Show a model the page and janki's account of it, and take its verdict.

    A refusal or a truncated answer is an error rather than a "no": both mean
    the question was never really answered, and recording either as a refusal
    would tell a reader the page was examined and found wanting when it was
    not examined at all.
    """
    caller = parse_call or claude_client.parse_call
    parsed, stop_reason, refusal = caller(
        model,
        claude_client.system_blocks(instructions),
        [
            prepared.content_block(),
            {"type": "text", "text": format_account(block)},
        ],
        verdict_schema(),
        client,
        max_tokens=claude_client.DEFAULT_MAX_TOKENS,
        effort=claude_client.effort_for(model),
    )
    if refusal:
        raise CoverageReviewError(
            f"The model declined to check coverage for {prepared.origin_path.name}: "
            f"{refusal}. Nothing was approved."
        )
    if stop_reason == "max_tokens":
        raise CoverageReviewError(
            f"The coverage check for {prepared.origin_path.name} was cut off "
            "mid-answer, so its verdict is unknown. Nothing was approved."
        )
    if parsed is None:
        raise CoverageReviewError(
            f"The coverage check for {prepared.origin_path.name} returned no "
            "verdict. Nothing was approved."
        )
    reason = str(getattr(parsed, "reason", "") or "").strip()
    if not reason:
        raise CoverageReviewError(
            "The coverage verdict carried no reason. A reason is the whole of "
            "what a person reads later, so an approval without one is refused."
        )
    return CoverageVerdict(
        approved=bool(getattr(parsed, "approved", False)),
        reason=reason,
        model=model,
        prompt_fingerprint=prompts.fingerprint(instructions),
    )
