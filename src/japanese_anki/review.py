"""The last gate before a deck ships: a model reads the finished card.

Every other check in this project is deterministic. `validate` knows the shape a
record must have, `qc` knows what Anki will draw from a furigana field,
`conjugation` knows which forms exist, `pitch` knows which patterns fit a
reading. Between them they catch everything that can be *stated as a rule* —
which is most of what goes wrong, and all of what goes wrong the same way twice.

What they cannot catch is a card that is well-formed and wrong. An example
sentence that uses the word in a sense the meanings do not list. A gloss that
says "to see" for 見える. A casual sentence that is not actually casual. A usage
note contradicting the sentence beside it. Those need someone to *read the card*,
and a model is the only reader available on every build.

So this is a reader, not a rule, and it is treated the way this project treats
every inferred thing:

* **It never has the last word.** A finding blocks the build, and only a human
  clears it — by fixing the card, or by saying `janki review --accept <id>`,
  which records that a person overruled the model and why. A model can stop a
  deck; it cannot pass one, and it cannot decide it was wrong about itself.
* **It is fingerprinted.** What was reviewed is recorded as a fingerprint of
  everything the card shows, so re-reviewing costs one request per *changed*
  card rather than one per build. Change a sentence and that card is unreviewed
  again; change nothing and the whole deck is free. The same shape
  `record_audio` uses for a clip.
* **An accepted finding is accepted for that version of the card.** The
  fingerprint covers the acceptance too, so editing the card afterwards brings
  the question back rather than carrying an old judgement forward onto new text.

The store is keyed by that fingerprint rather than by record id, because a
review is of a *card version* and one record can ship as more than one card: a
deck may carry an inline `notes:` entry overriding a field, or read a different
`source:` entirely, so `janki build` and `janki review` were looking at
different text under the same id. Keyed by id, that was an unbreakable deadlock
— the build refused a card the review said was clean, and no flag reached it.
Keyed by version, both sides ask the same question, and reverting an edit
restores its review for free instead of buying it again.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from japanese_anki import claude_client
from japanese_anki.errors import JankiError
from japanese_anki.identifiers import short_fingerprint
from japanese_anki.io import atomic_write_text
from japanese_anki.models import VocabularyRecord

__all__ = [
    "CardReview",
    "Finding",
    "ReviewError",
    "SEVERITIES",
    "card_fingerprint",
    "load_store",
    "open_findings",
    "review_records",
    "save_store",
    "unreviewed",
]


class ReviewError(JankiError):
    """A card review could not be run, or its answer could not be used."""


#: What a finding is worth. Only ``error`` blocks a build: a model asked to find
#: fault will always find some, and a gate that stops on "could be more natural"
#: is a gate someone learns to bypass.
SEVERITIES: tuple[str, ...] = ("error", "note")


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing wrong with one card."""

    #: Which part of the card: ``meanings``, ``examples[0]``, ``usage_notes``.
    where: str
    #: What is wrong, in a sentence.
    problem: str
    #: ``error`` blocks the build; ``note`` is reported and does not.
    severity: str = "note"
    #: What it should say instead, when the model can name it.
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "where": self.where,
            "problem": self.problem,
            "severity": self.severity,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Finding:
        severity = str(raw.get("severity") or "note").strip().lower()
        return cls(
            where=str(raw.get("where") or "").strip(),
            problem=str(raw.get("problem") or "").strip(),
            # An invented severity becomes the blocking one on purpose. A
            # finding janki cannot classify is not a finding it may wave through.
            severity=severity if severity in SEVERITIES else "error",
            suggestion=str(raw.get("suggestion") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class CardReview:
    """What the reader said about one card, and which version of it."""

    record_id: str
    #: Fingerprint of everything the card shows. A different card is unreviewed.
    content_fp: str
    at: str = ""
    findings: tuple[Finding, ...] = ()
    #: Set when a human overruled the findings for *this* fingerprint.
    accepted: bool = False
    #: Why they overruled it. Required — an acceptance with no reason is
    #: indistinguishable next month from one nobody thought about.
    accepted_because: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "at": self.at,
            "accepted": self.accepted,
            "accepted_because": self.accepted_because,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, content_fp: str, raw: dict[str, Any]) -> CardReview:
        listed = raw.get("findings") or []
        if not isinstance(listed, list):
            raise ReviewError(
                f"{content_fp}: findings must be a list, got {type(listed).__name__}"
            )
        for item in listed:
            if not isinstance(item, dict):
                raise ReviewError(
                    f"{content_fp}: each finding must be an object, got "
                    f"{type(item).__name__}"
                )
        return cls(
            record_id=str(raw.get("record_id") or ""),
            content_fp=content_fp,
            at=str(raw.get("at") or ""),
            findings=tuple(Finding.from_dict(item) for item in listed),
            accepted=bool(raw.get("accepted", False)),
            accepted_because=str(raw.get("accepted_because") or ""),
        )

    def blocking(self) -> tuple[Finding, ...]:
        """Findings that stop a build: errors nobody has accepted."""
        if self.accepted:
            return ()
        return tuple(f for f in self.findings if f.severity == "error")


def card_fingerprint(record: VocabularyRecord) -> str:
    """Everything the finished card shows, as one fingerprint.

    Every field a reader would form an opinion about, and nothing else: two
    records differing only in ledger state or audio filename are the same card
    to review, and re-reviewing them would cost a request to reach the same
    answer. Audio *content* is already covered by the sentence it says.

    Deliberately not `record.to_dict()`: that would fold in `source`, tags and
    the unverified-furigana bookkeeping, so re-importing the same word from a
    second deck would invalidate a review that is still perfectly good.
    """
    parts: list[str] = [
        record.expression,
        record.reading,
        record.furigana,
        "|".join(record.meanings),
        record.part_of_speech,
        # Transitivity is exactly the kind of claim only a reader can check —
        # the gate's first real run flagged a verb "glossed as intransitive" —
        # so a card whose transitivity changed must be read again.
        record.transitivity,
        record.verb_group,
        record.usage_notes,
        "|".join(record.pitch_accent),
    ]
    for example in record.examples:
        parts.extend(
            [
                example.japanese,
                example.furigana,
                example.english,
                example.register,
            ]
        )
    return short_fingerprint(*parts)


INSTRUCTIONS = """\
You are the last reader of a finished flashcard before it ships to a learner.
Every mechanical check has already passed: the fields are well-formed, the
furigana matches the sentence, the conjugations are computed, the pitch pattern
fits the reading. You are looking for the thing a rule cannot state — a card
that is well-formed and *wrong*.

Report a finding only when you can name what is wrong and why. In particular:

* an example sentence that does not use the word in one of the senses the
  card's meanings list, or uses it unnaturally;
* an English gloss that misstates the word — a transitive verb glossed as
  intransitive, a sense that belongs to a homograph;
* a sentence whose stated register is wrong: the "casual" one written in
  〜ます/です, or the "polite" one in plain form;
* a usage note that contradicts the sentences beside it, or states a rule that
  is not true;
* an example far above the learner's level, or one that teaches a different
  grammar point than the word it is for.

Severity:

* "error" — the card would teach something false. This stops the deck being
  built until a human fixes or overrules it.
* "note" — a real improvement, but nothing on the card is wrong. Reported and
  does not stop anything.

**Report nothing when nothing is wrong.** An empty list is the expected answer
for most cards, and it is a better answer than a manufactured "could be more
natural". You are a gate, not a critic: every false finding costs a person the
time to read it and dismiss it, and a gate that cries wolf is one they learn to
wave through — which is worse than no gate at all.

Judge only the card in front of you. Do not report a field as missing: whether
a card needs an image or a fourth example is the deck's decision, not yours."""


@functools.cache
def review_schema() -> Any:
    """The shape a card review must take."""
    from pydantic import BaseModel, Field

    class ReviewFinding(BaseModel):
        where: str = Field(description="Which part of the card, e.g. examples[0].")
        problem: str = Field(description="What is wrong, in a sentence.")
        severity: str = Field(
            default="note", description='"error" if the card teaches something false.'
        )
        suggestion: str = Field(
            default="", description="What it should say instead, if you can name it."
        )

    class CardVerdict(BaseModel):
        findings: list[ReviewFinding] = Field(
            default_factory=list,
            description="Empty when nothing is wrong, which is the usual answer.",
        )

    return CardVerdict


def card_prompt(record: VocabularyRecord) -> str:
    """The card as a reader sees it."""
    lines = [
        f"Expression: {record.expression}",
        f"Reading: {record.reading}",
    ]
    if record.furigana:
        lines.append(f"Furigana: {record.furigana}")
    lines.append(f"Meanings: {'; '.join(record.meanings)}")
    if record.part_of_speech:
        lines.append(f"Part of speech: {record.part_of_speech}")
    if record.transitivity:
        lines.append(f"Transitivity: {record.transitivity}")
    if record.verb_group:
        lines.append(f"Verb group: {record.verb_group}")
    if record.pitch_accent:
        lines.append(f"Pitch accent: {', '.join(record.pitch_accent)}")
    for index, example in enumerate(record.examples):
        register = example.register or "unstated"
        lines.append(f"examples[{index}] ({register}): {example.japanese}")
        if example.furigana:
            lines.append(f"    furigana: {example.furigana}")
        if example.english:
            lines.append(f"    english: {example.english}")
    if record.usage_notes:
        lines.append(f"Usage notes: {record.usage_notes}")
    return "\n".join(lines)


def review_records(
    records: Sequence[VocabularyRecord],
    *,
    model: str,
    style_guide: str,
    card_design: str = "",
    client: Any | None = None,
) -> tuple[dict[str, CardReview], list[str]]:
    """Read each card, and say what is wrong with it.

    Returns what was read and the cards that could not be, so the caller can
    save the first and report the second. One card failing must not discard
    nineteen answers already paid for — the same reason `janki audio` writes as
    it goes rather than at the end.

    A card that fails is simply *absent* from the result, which leaves it
    unreviewed: the build still refuses it, and nothing has to remember that a
    failure means "not clean". A truncated answer looks exactly like "nothing
    wrong", and this is the check standing between a wrong card and a learner.

    One request per card rather than one for the batch: a model given twenty
    cards at once returns findings keyed by a position it has to keep track of,
    and a dropped or shifted index attaches a finding to the wrong word — which
    is worse than no finding, because someone would edit a card that was right.
    """
    blocks = claude_client.system_blocks(
        *(text for text in (style_guide, card_design, INSTRUCTIONS) if text),
        cache_ttl="1h",
    )
    now = datetime.now(UTC).strftime("%Y-%m-%d")
    reviewed: dict[str, CardReview] = {}
    failures: list[str] = []
    for record in records:
        try:
            call = claude_client.parse_call(
                model,
                blocks,
                [{"type": "text", "text": card_prompt(record)}],
                review_schema(),
                client,
                # Generous, because the model reasons before answering and the
                # answer itself is short: 2000 cut off a card mid-verdict, and a
                # budget that truncates turns a clean read into a failed one.
                max_tokens=8000,
            )
        except JankiError as exc:
            failures.append(f"{record.id}: {exc}")
            continue
        if call.stop_reason == "refusal":
            failures.append(
                f"{record.id}: the model declined to read it"
                + (f" ({call.refusal})" if call.refusal else "")
            )
            continue
        if call.parsed is None:
            failures.append(
                f"{record.id}: no usable answer (stopped: {call.stop_reason})"
            )
            continue
        findings = tuple(
            Finding.from_dict(
                {
                    "where": getattr(item, "where", ""),
                    "problem": getattr(item, "problem", ""),
                    "severity": getattr(item, "severity", "note"),
                    "suggestion": getattr(item, "suggestion", ""),
                }
            )
            for item in (getattr(call.parsed, "findings", []) or [])
        )
        reviewed[card_fingerprint(record)] = CardReview(
            record_id=record.id,
            content_fp=card_fingerprint(record),
            at=now,
            # Kept even with no text. Dropping it recorded the card as read
            # and shipped it, which is the opposite of what this module does
            # with a finding it cannot classify two screens up.
            findings=tuple(
                f if f.problem else replace(
                    f, problem=f.suggestion or "(the reader gave no reason)"
                )
                for f in findings
                if f.problem or f.suggestion or f.severity == "error"
            ),
        )
    return reviewed, failures


def unreviewed(
    records: Iterable[VocabularyRecord], store: dict[str, CardReview]
) -> list[VocabularyRecord]:
    """Records whose current content nobody has read.

    Looked up by fingerprint, so a record that has been edited is unreviewed
    even though an entry under its id exists: that entry describes a card which
    no longer exists.
    """
    return [
        record for record in records if card_fingerprint(record) not in store
    ]


def open_findings(
    records: Iterable[VocabularyRecord], store: dict[str, CardReview]
) -> list[tuple[str, Finding]]:
    """Blocking findings against the current version of each record."""
    blocking: list[tuple[str, Finding]] = []
    for record in records:
        entry = store.get(card_fingerprint(record))
        if entry is None:
            continue
        blocking.extend((record.id, finding) for finding in entry.blocking())
    return blocking


def accept(
    store: dict[str, CardReview], record_id: str, because: str
) -> dict[str, CardReview]:
    """Record that a human overruled a card's findings, and why.

    Named by record id rather than by fingerprint, because that is what a person
    has in front of them — the id is what the refusal printed. Every version of
    that card currently carrying a finding is accepted: they were all read, and
    someone saying "this word is fine" means the word, not one hash of it.
    """
    if not because.strip():
        raise ReviewError(
            "An acceptance needs a reason: next month it is indistinguishable "
            "from one nobody thought about."
        )
    matching = [
        fingerprint
        for fingerprint, entry in store.items()
        if entry.record_id == record_id
    ]
    if not matching:
        raise ReviewError(f"No review on record for {record_id}")
    updated = dict(store)
    for fingerprint in matching:
        updated[fingerprint] = replace(
            updated[fingerprint], accepted=True, accepted_because=because.strip()
        )
    return updated


def load_store(path: Path) -> dict[str, CardReview]:
    """Every card version janki has read, keyed by its content fingerprint."""
    file = Path(path)
    if not file.exists():
        return {}
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReviewError(f"Could not read {file}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ReviewError(
            f"{file} must hold a JSON object keyed by content fingerprint"
        )
    # Refused, not skipped: `save_store` rewrites the whole file from what was
    # loaded, so an entry the loader dropped would be erased from the committed
    # store on the next write, and its card would silently become unreviewed.
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ReviewError(
                f"{file}: entry for {str(name)!r} must be an object, got "
                f"{type(value).__name__}"
            )
    return {
        str(name): CardReview.from_dict(str(name), value)
        for name, value in raw.items()
    }


def save_store(path: Path, store: dict[str, CardReview]) -> None:
    """Write the review file, sorted, so a re-run produces no diff by itself."""
    payload = {name: store[name].to_dict() for name in sorted(store)}
    atomic_write_text(
        Path(path), json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
