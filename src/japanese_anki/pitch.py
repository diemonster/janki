"""Pitch accent: jpdb's pattern into what an engine and a card can use.

Two consumers, one source of truth. VOICEVOX needs AquesTalk kana notation —
katakana with a single ``'`` marking the accent nucleus — to force the accent
rather than guess it, which is the whole reason janki bothers: 橋 and 箸 are the
minimal pair a flashcard exists to teach, and an engine left to guess renders
them alike. The card needs the same pattern drawn, which is
:func:`render_pitch_html`.

**The naive rule is wrong for the largest accent class**, which is why this is a
module and not a lambda. jpdb's pattern has one character per *kana* plus one
trailing position for the following particle, so ``話す`` → ``LHLL``. Accent is
a property of *morae*, not kana — ``びょ`` is one mora written with two kana —
so the pattern has to be regrouped before it is read. And the accent nucleus is
the mora *after which* H falls to L, which means the particle position is the
only thing distinguishing odaka (``橋`` ``LHL``, the drop lands on the particle)
from heiban (``端`` ``LHH``, no drop anywhere).

**Heiban and odaka render identically here, and that is correct.** AquesTalk
notation carries exactly one accent mark per phrase, and the engine's own
convention writes heiban with the mark on the final mora — the same place odaka
puts it. The two patterns differ only in the pitch of a particle, and janki
speaks a word on its own, with no particle after it. So the distinction has
nothing to land on in isolated word audio, and encoding it would be inventing a
difference the notation cannot express. :func:`render_pitch_html` *does* keep
them apart, because a card shows the pattern rather than speaking it.

Nothing here guesses. A pattern whose length does not match its reading is a
:class:`PitchError` rather than a best effort, because the alternative is
silently speaking the wrong accent onto a card built to teach that accent.
"""

from __future__ import annotations

import html
import unicodedata
from collections.abc import Sequence

from japanese_anki.errors import JankiError
from japanese_anki.models import VocabularyRecord

__all__ = [
    "ACCENT_MARK",
    "PitchError",
    "morae",
    "render_pitch_html",
    "select_pattern",
    "to_aquestalk",
]


class PitchError(JankiError):
    """A pattern that cannot be read against its reading."""


#: AquesTalk's accent nucleus mark, written after the accented mora.
ACCENT_MARK = "'"

#: Kana that attach to the mora before them rather than forming one of their
#: own. ``っ`` is deliberately absent — it *is* a mora, and so is ``ん``, which
#: is the fact that makes 学校 four morae and not three.
_ATTACHING = frozenset("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")

#: The two levels jpdb writes. Accepted in either case and nothing else: an
#: unexpected character means the pattern is not what this function knows how to
#: read, and reading it anyway would produce a confident wrong accent.
_LEVELS = {"H": "H", "L": "L", "h": "H", "l": "L"}

_HIRAGANA_START = "ぁ"
_HIRAGANA_END = "ゖ"


def _to_katakana(text: str) -> str:
    """Hiragana to katakana, leaving everything else alone.

    A straight block shift rather than a table: the two blocks are
    codepoint-aligned across their whole range, ``ゔ`` → ``ヴ`` included. Kana
    already in katakana — a loanword reading — pass through untouched.
    """
    return "".join(
        chr(ord(char) + 0x60) if _HIRAGANA_START <= char <= _HIRAGANA_END else char
        for char in text
    )


def morae(reading: str) -> list[str]:
    """``reading`` grouped into morae.

    The unit accent is counted in. A small ``ゃゅょ`` joins the kana before it,
    so ``びょういん`` is four morae written with five kana; ``っ`` and ``ん``
    stand alone, which is why ``がっこう`` is four and not three.

    A leading attaching kana has nothing to attach to. It is kept as its own
    mora rather than dropped — this function's job is to group what it was
    given, not to judge it, and a reading that starts with ``ょ`` is a data
    problem the caller's length check will surface with better context.
    """
    grouped: list[str] = []
    for char in unicodedata.normalize("NFC", reading):
        if char in _ATTACHING and grouped:
            grouped[-1] += char
        else:
            grouped.append(char)
    return grouped


def _levels(reading: str, pattern: str) -> tuple[list[str], str]:
    """Per-mora levels plus the particle's, validated against the reading.

    The length check is against *kana*, which is the unit jpdb writes one
    character of pattern for. Community clients hard-assert this; janki raises,
    because a mismatch means the two halves describe different words and there
    is no safe way to read one against the other.
    """
    kana = unicodedata.normalize("NFC", reading)
    if not kana:
        raise PitchError("A pitch pattern needs a reading to describe; got none.")
    if len(pattern) != len(kana) + 1:
        raise PitchError(
            f"Pitch pattern {pattern!r} does not fit the reading {reading!r}: "
            f"expected {len(kana) + 1} characters (one per kana, plus one for "
            f"the following particle), got {len(pattern)}."
        )
    try:
        levels = [_LEVELS[char] for char in pattern]
    except KeyError as exc:
        raise PitchError(
            f"Pitch pattern {pattern!r} contains {exc.args[0]!r}; only H and L "
            "describe a pitch, and guessing at anything else would put a "
            "confident wrong accent on a card."
        ) from None

    # One level per mora, taken from the mora's *first* kana. Every kana of a
    # mora carries the same pitch, so which one is read does not matter for
    # well-formed data — but the first is the one that cannot be a small kana,
    # and so the one whose value is unambiguous if a source ever disagrees with
    # itself across a 拗音.
    per_mora: list[str] = []
    index = 0
    for mora in morae(kana):
        per_mora.append(levels[index])
        index += len(mora)
    return per_mora, levels[-1]


def _accent_position(per_mora: Sequence[str], particle: str) -> int:
    """The 1-based mora after which the pitch drops, per AquesTalk's convention.

    Three cases, and the third is the one the naive rule gets wrong:

    * the drop is inside the word — atamadaka or nakadaka — so the mark goes
      after the mora it falls from;
    * the drop lands on the particle — odaka — so the mark goes after the last
      mora, which is where the fall begins;
    * there is no drop at all — heiban — and AquesTalk still requires one mark,
      which the engine's convention puts on the final mora. Same output as
      odaka, and see the module docstring for why that is right.
    """
    levels = [*per_mora, particle]
    for index in range(len(per_mora)):
        if levels[index] == "H" and levels[index + 1] == "L":
            return index + 1
    return len(per_mora)


def to_aquestalk(reading: str, pattern: str) -> str:
    """``reading`` in AquesTalk kana notation, with ``pattern``'s accent forced.

    Katakana with exactly one :data:`ACCENT_MARK` after the accented mora, which
    is what ``/accent_phrases?is_kana=true`` takes. Raises :class:`PitchError`
    rather than returning something plausible for a pattern it cannot read.
    """
    per_mora, particle = _levels(reading, pattern)
    position = _accent_position(per_mora, particle)
    units = [_to_katakana(mora) for mora in morae(reading)]
    units[position - 1] += ACCENT_MARK
    return "".join(units)


def select_pattern(record: VocabularyRecord) -> str | None:
    """Which pattern this record's audio should use, or ``None`` for no answer.

    ``audio_accent`` first, because it exists for the reader who listened and
    disagreed; then jpdb's primary, which is the first entry in its own
    ordering. ``None`` when the record carries no pattern at all — the caller
    decides what to do about that, and DESIGN_V2 says skip and flag rather than
    let an engine guess, since the homographs a guess gets wrong are exactly the
    ones a pitch card exists for.

    **Upper-cased**, which is not cosmetic. The ledger's word-audio *content*
    fingerprint is ``fp(reading + this)``, and it is defined as covering what
    was spoken — so retyping ``LHLL`` as ``lhll`` would report perfectly good
    audio as stale, over two strings :func:`to_aquestalk` renders identically.
    Canonicalising here rather than at the loaders covers ``audio_accent`` and
    records built in memory too, and every already-upper pattern hashes
    unchanged, so no committed ledger entry moves.
    """
    if record.audio_accent.strip():
        return record.audio_accent.strip().upper()
    for pattern in record.pitch_accent:
        if pattern.strip():
            return pattern.strip().upper()
    return None


def render_pitch_html(reading: str, patterns: Sequence[str]) -> str:
    """The pattern drawn over the reading, one ``<span>`` per mora.

    A high mora gets a line above it and the mora the pitch falls from gets one
    down its right-hand side, matching the compact two-level diagram used by
    jpdb. Classes rather than inline styles let the note template own the look,
    so restyling a card later does not require every note to be regenerated.

    Every pattern is rendered, primary first, because a word with two accepted
    accents has two and showing only one would teach that the other is wrong.
    Unlike :func:`to_aquestalk` this keeps heiban and odaka apart — there is no
    single-mark notation forcing them together here, and the difference is
    exactly what a learner is looking at the diagram to see.

    The reading is escaped; a pattern that does not fit it raises
    :class:`PitchError`, on the same reasoning as everywhere else in this module.
    """
    if isinstance(patterns, str):
        raise PitchError(
            f"render_pitch_html takes a list of patterns; got the string "
            f"{patterns!r}, which would be read one character at a time. Pass "
            f"[{patterns!r}]."
        )
    rendered: list[str] = []
    for pattern in patterns:
        if not pattern.strip():
            continue
        per_mora, particle = _levels(reading, pattern.strip())
        units = morae(reading)
        levels = [*per_mora, particle]
        spans: list[str] = []
        for index, mora in enumerate(units):
            classes = ["mora", "high" if per_mora[index] == "H" else "low"]
            if per_mora[index] == "H" and levels[index + 1] == "L":
                classes.append("drop")
            # A rise as well as a fall: jpdb's notation draws the vertical at
            # both transitions, and without it a heiban word is a flat line a
            # reader takes for "no accent recorded" rather than "no drop".
            if index > 0 and per_mora[index] == "H" and per_mora[index - 1] == "L":
                classes.append("rise")
            spans.append(
                f'<span class="{" ".join(classes)}">{html.escape(mora)}</span>'
            )
        # The particle slot is drawn as an empty mora so odaka is visible: the
        # fall happens after the word, and a diagram that stops at the last kana
        # has nowhere to show it.
        particle_classes = ["mora", "particle", "high" if particle == "H" else "low"]
        if particle == "H" and per_mora[-1] == "L":
            particle_classes.append("rise")
        spans.append(f'<span class="{" ".join(particle_classes)}"></span>')
        rendered.append(f'<span class="pitch">{"".join(spans)}</span>')
    return "".join(rendered)
