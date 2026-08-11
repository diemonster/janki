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
import re
from collections.abc import Iterable, Mapping, Sequence
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
    "carry_acceptances",
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


def _marks_of(
    content_fp: str, raw: dict[str, Any], findings: tuple[Finding, ...]
) -> tuple[str, ...]:
    """The marks an entry's acceptance answers, migrating one written before them.

    A store written by any earlier janki carries `accepted: true` with no
    `accepted_marks`, and reading that as "answers nothing" would silently stop
    a human's acceptance working the moment the code was upgraded — the build
    refusing a card they had already cleared, with nothing saying why. Before
    marks existed the acceptance covered whatever errors the entry held, so that
    is what it is read as.
    """
    listed = raw.get("accepted_marks") or []
    if not isinstance(listed, list):
        # Refused, not skipped — the rule `load_store` states and the sibling
        # `findings` field already follows. A scalar iterated into seventeen
        # single-character marks, which matched nothing, suppressed the
        # migration below by being truthy, and was written back to the store.
        #
        # Named, like every other error on this load path: `load_store` refuses
        # rather than skips, so one bad entry takes the review gate down for the
        # whole repository, and "accepted_marks must be a list" alone left
        # someone bisecting a few hundred entries of JSON by hand.
        raise ReviewError(
            f"{content_fp}: accepted_marks must be a list, got "
            f"{type(listed).__name__}"
        )
    for mark in listed:
        # The elements too, as `findings` checks its own. Two different
        # failures, both silent about their real cause:
        #
        # A truthy scalar — `1`, `true` — is kept by the filter below, so
        # `stored` is non-empty and suppresses the migration this entry needs;
        # it then matches no `finding_mark`, so the card's own error blocks the
        # build, `carry_acceptances` writes `accepted: false` back, and
        # `to_dict` returns the junk to the store. The acceptance is
        # unrecoverable at that point, because the migration only runs while
        # `accepted` is true.
        #
        # An unhashable one — a nested list, an object — never gets that far:
        # `set(accepted_marks)` in `blocking` and `carry_acceptances` raises
        # `TypeError`, which `cli.main` does not catch, so the review gate exits
        # on a raw traceback rather than a named error.
        if not isinstance(mark, str):
            raise ReviewError(
                f"{content_fp}: each accepted mark must be a string, got "
                f"{type(mark).__name__}"
            )
    stored = tuple(mark for mark in listed if mark)
    if stored or not raw.get("accepted"):
        return stored
    return tuple(
        dict.fromkeys(
            finding_mark(f) for f in findings if f.severity == "error"
        )
    )


def finding_mark(finding: Finding) -> str:
    """A finding's identity for the purpose of "has this been answered".

    Where it is and how bad it is, never the model's prose: a re-read of an
    unchanged card rewrites its sentences freely. Underscores, spacing and case
    are folded because the same field comes back as ``pitch_accent``,
    ``Pitch accent`` and ``Meanings`` across runs.

    Severity is pinned to ``error`` by every caller, so in practice a mark *is*
    the field name — an acceptance says "this field is right on this card",
    which is what a person overruling a finding means. The cost, stated: a
    genuinely different objection to the same field is covered by it. Including
    the prose to separate them was the first attempt and it did not work at all,
    since the model rewrites its sentences on every read; a field is the
    coarsest key that is stable, and a stable key is what an acceptance needs.
    """
    place = " ".join(finding.where.replace("_", " ").split()).lower()
    return f"{place}|{finding.severity}"


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
    #: Which findings the acceptance answers, as `finding_mark` strings. Held
    #: separately from `findings` because a re-read replaces those: an entry
    #: carrying `accepted` with an empty finding list had forgotten what was
    #: agreed, so the next run that resurfaced the same error demanded a fresh
    #: answer to a question already settled.
    accepted_marks: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "at": self.at,
            "accepted": self.accepted,
            "accepted_because": self.accepted_because,
            "accepted_marks": list(self.accepted_marks),
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
        findings = tuple(Finding.from_dict(item) for item in listed)
        return cls(
            record_id=str(raw.get("record_id") or ""),
            content_fp=content_fp,
            at=str(raw.get("at") or ""),
            findings=findings,
            accepted=bool(raw.get("accepted", False)),
            accepted_because=str(raw.get("accepted_because") or ""),
            accepted_marks=_marks_of(content_fp, raw, findings),
        )

    def blocking(self) -> tuple[Finding, ...]:
        """Findings that stop a build: errors nobody has accepted.

        Per finding rather than per entry. A re-read of the same card version
        can surface an error the acceptance never covered, and an all-or-nothing
        flag would wave that through on the strength of an answer to a different
        question.
        """
        # The marks are consulted whatever `accepted` says. They outlive a
        # lapse now, so an entry that went un-accepted because a re-read raised
        # something *new* still holds the record that the old finding was
        # answered — and re-listing it under "overrule it: janki review
        # --accept" asks again for a decision the same entry proves was made.
        # `accepted` is a summary of the marks, not a gate on them.
        answered = set(self.accepted_marks)
        return tuple(
            f
            for f in self.findings
            if f.severity == "error" and finding_mark(f) not in answered
        )


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


def card_prompt(record: VocabularyRecord, max_meanings: int = 0) -> str:
    """The card as a reader sees it.

    ``max_meanings`` is the deck's cap, and the meanings are shown capped —
    with the same "+N more senses" note the card carries — because that is what
    a learner is looking at. Sending the whole stored list made the reader
    object, correctly, that nineteen senses of する including JMdict's own
    metalanguage are unreadable on a card; but the card shows four. Every one of
    those findings was about text that never ships, which is the one thing a
    gate must not spend a person's attention on.
    """
    lines = [
        f"Expression: {record.expression}",
        f"Reading: {record.reading}",
    ]
    if record.furigana:
        lines.append(f"Furigana: {record.furigana}")
    shown = [value for value in record.meanings if value]
    hidden = 0
    if 0 < max_meanings < len(shown):
        hidden, shown = len(shown) - max_meanings, shown[:max_meanings]
    more = f" (+{hidden} more senses, not shown on the card)" if hidden else ""
    lines.append(f"Meanings: {'; '.join(shown)}{more}")
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
    max_meanings: int = 0,
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
                [{"type": "text", "text": card_prompt(record, max_meanings)}],
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


def carry_acceptances(
    store: Mapping[str, CardReview], fresh: Mapping[str, CardReview]
) -> dict[str, CardReview]:
    """Fresh reviews, keeping an acceptance that still answers what was found.

    An acceptance is of a *card version*, and a re-read under `--force` produces
    the same version — so dropping it made someone re-type a reason they had
    already given about text that had not changed. It is also of the findings
    that were *there*: a re-read turning up a new problem is new information,
    and clearing that with a sentence written about the old one is the silent
    pass this module exists to prevent.

    Two things follow, and the first is absolute.

    **The reason is never destroyed.** It is the only human-written sentence in
    this file and nothing can reconstruct it. It carries over even when the
    acceptance does not, so someone answering a new finding can see what was
    decided last time instead of a blank.

    **The acceptance carries when every blocking finding was already accepted**,
    compared by *where* and *severity* rather than by the model's prose. Prose
    was the first attempt and it was the wrong key: a re-read of an unchanged
    card rewrites its sentences freely, so the comparison almost never held, and
    the branch that ran was the one that threw the acceptance away. An empty
    finding set carries too — there is nothing left to answer.
    """

    carried: dict[str, CardReview] = {}
    for fingerprint, entry in fresh.items():
        previous = store.get(fingerprint)
        if previous is None:
            carried[fingerprint] = entry
            continue
        # Against what was *accepted*, not against whatever the last read
        # happened to return. Comparing findings meant a clean re-read wrote an
        # accepted entry with none, and the run after that — when the model
        # resurfaced the error, as it does — found an empty set to check against
        # and demanded the answer again.
        #
        # The marks and the reason carry whether or not the entry is currently
        # accepted, so an acceptance that lapsed for one generation — a re-read
        # raised something new, nobody has answered it yet — comes back when the
        # new finding goes away, instead of starting from nothing.
        marks = set(previous.accepted_marks)
        # `previous.accepted` counts as well as the marks. An entry written
        # before marks existed has none to migrate from when its findings are
        # empty, and so does an acceptance made on a card whose findings were
        # all notes — and withdrawing those on the next clean re-read erased a
        # person's decision from the file this repo treats as source of truth,
        # leaving their reason orphaned beside it. A card nobody accepted still
        # does not become accepted by being clean.
        answered = (bool(marks) or previous.accepted) and {
            finding_mark(f) for f in entry.findings if f.severity == "error"
        } <= marks
        carried[fingerprint] = replace(
            entry,
            accepted=answered,
            accepted_because=previous.accepted_because,
            accepted_marks=previous.accepted_marks,
        )
    return carried


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
    store: dict[str, CardReview],
    record_id: str,
    because: str,
    shipping: Iterable[str] = (),
) -> dict[str, CardReview]:
    """Record that a human overruled a card's findings, and why.

    Named by record id rather than by fingerprint, because that is what a person
    has in front of them — the id is what the refusal printed.

    ``shipping`` is the fingerprints the decks currently resolve, and the
    acceptance is scoped to those. The store keeps every version ever reviewed,
    so without it a reason written about today's text also cleared a finding on
    a version no deck ships — and anything that brought that text back (a
    revert, a re-import, a merge) shipped a card carrying a finding nobody read,
    annotated with a reason about different words. It also crossed decks: two
    decks shipping one record differently are two cards, and clearing one is not
    an answer about the other.
    """
    if not because.strip():
        raise ReviewError(
            "An acceptance needs a reason: next month it is indistinguishable "
            "from one nobody thought about."
        )
    current = {str(fingerprint) for fingerprint in shipping}
    matching = [
        fingerprint
        for fingerprint, entry in store.items()
        if entry.record_id == record_id and (not current or fingerprint in current)
    ]
    if not matching:
        if any(entry.record_id == record_id for entry in store.values()):
            raise ReviewError(
                f"{record_id} has changed since it was last read, so there is "
                f"nothing to accept on the version the decks now ship. Run "
                f"`janki review` first."
            )
        raise ReviewError(f"No review on record for {record_id}")
    updated = dict(store)
    for fingerprint in matching:
        entry = updated[fingerprint]
        # What is being answered, recorded now. A later re-read replaces
        # `findings`, so an acceptance that only pointed at them forgot what it
        # had agreed to the first time the model returned a clean read.
        marks = tuple(
            dict.fromkeys(
                [*entry.accepted_marks]
                + [finding_mark(f) for f in entry.findings if f.severity == "error"]
            )
        )
        updated[fingerprint] = replace(
            entry,
            accepted=True,
            accepted_because=because.strip(),
            accepted_marks=marks,
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
        str(fingerprint): entry
        for fingerprint, entry in (
            _entry_from(file, str(name), value) for name, value in raw.items()
        )
    }


#: A store key is a `short_fingerprint`, which is twelve hex digits. A record id
#: is not, which is what tells an old file from a current one — the *key* says
#: it, and looking inside the value only recognised one of the two old shapes.
_FINGERPRINT = re.compile(r"^[0-9a-f]{12}$")


def _entry_from(file: Path, key: str, raw: dict[str, Any]) -> tuple[str, CardReview]:
    """One store entry, migrating a record-id-keyed one where that is possible.

    The store was keyed by record id before a review became a review of a card
    *version*. Read under the current schema such a file loads every entry with
    ``content_fp`` set to a record id and ``record_id`` empty — matching no card,
    so every card reads as unreviewed, the build refuses the deck, and the next
    `save_store` writes the mistake back. That is the silent erasure the
    non-object check above exists to stop, arriving through the key.

    Two old shapes existed and only one is recoverable. The first kept
    ``content_fp`` in the value, so the real key is right there and the entry is
    rekeyed. The second was written by the reader that had *already* misread the
    first: its values carry ``record_id: ""`` and no fingerprint at all, so
    nothing on disk says which card version it describes. That one is refused by
    name rather than loaded as garbage and rewritten forever — the same
    "refused, not skipped" rule, for the same reason.
    """
    if _FINGERPRINT.match(key):
        return key, CardReview.from_dict(key, raw)
    stored = str(raw.get("content_fp") or "")
    if stored:
        return stored, CardReview.from_dict(stored, {**raw, "record_id": key})
    raise ReviewError(
        f"{file}: the entry for {key!r} is keyed by record id and carries no "
        f"content_fp, so which version of the card it describes cannot be "
        f"recovered. Delete it and re-run `janki review`; an acceptance on it "
        f"has to be made again."
    )


def save_store(path: Path, store: dict[str, CardReview]) -> None:
    """Write the review file, sorted, so a re-run produces no diff by itself."""
    payload = {name: store[name].to_dict() for name in sorted(store)}
    atomic_write_text(
        Path(path), json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
