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

import hashlib
import html
import unicodedata
from collections.abc import Sequence
from dataclasses import replace

from japanese_anki.errors import JankiError
from japanese_anki.models import VocabularyRecord

__all__ = [
    "ACCENT_MARK",
    "PitchError",
    "SOURCE_BINDING_KEY",
    "bind_source",
    "has_source_binding",
    "morae",
    "pattern_facts",
    "render_pitch_html",
    "select_pattern",
    "to_aquestalk",
]


class PitchError(JankiError):
    """A pattern that cannot be read against its reading."""


#: AquesTalk's accent nucleus mark, written after the accented mora.
ACCENT_MARK = "'"

#: A content-bound statement that the stored pattern came from a named source.
#: It lives with the record rather than in the operational ledger so a manual
#: edit to the pattern can be detected even when old enrichment history remains.
SOURCE_BINDING_KEY = "janki_pitch_accent_source"

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


#: Half-width prolonged sound mark. The same character to a reader, a
#: different codepoint to `NFC` — which `morae` uses, and which folds
#: compatibility variants nowhere. Left alone it survives to the engine as
#: itself and answers 400, the same failure `ー` does.
_HALFWIDTH_LONG_VOWEL = "\uff70"


def _to_katakana(text: str) -> str:
    """Hiragana to katakana, leaving everything else alone.

    A straight block shift rather than a table: the two blocks are
    codepoint-aligned across their whole range, ``ゔ`` → ``ヴ`` included. Kana
    already in katakana — a loanword reading — pass through untouched.

    The one exception is the half-width prolonged sound mark, folded to the
    full-width one so :func:`_spell_long_vowels` sees a single character to
    respell rather than two spellings of it.
    """
    return "".join(
        chr(ord(char) + 0x60)
        if _HIRAGANA_START <= char <= _HIRAGANA_END
        else ("ー" if char == _HALFWIDTH_LONG_VOWEL else char)
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


#: The vowel each kana ends on, for spelling out a long vowel **to VOICEVOX**.
#:
#: This is an input convention of one engine, not a respelling of Japanese.
#: `エスカレーター` is written with `ー` and the record keeps it: `reading` is
#: half of a record's permanent id. Only the string handed to the engine is
#: respelled, and only in transit.
#:
#: VOICEVOX's kana mode rejects `ー` outright — `/accent_phrases?is_kana=true`
#: answers 400 `UNKNOWN_TEXT` for `エスカレーター` with or without an accent
#: mark, and for a bare `オー`. AquesTalk notation writes a long vowel as the
#: vowel it lengthens (`オオ`, `コオヒイ`), so a reading carrying `ー` has to be
#: respelled before it is sent. Measured against the running engine, not
#: inferred: `オオ'`, `コオヒイ'` and `エスカレエタア'` all answer 200.
#:
#: The vowel each kana ends on, where repeating it is the right thing. `ヴ`
#: is here —
#: `_to_katakana` advertises `ゔ`→`ヴ` and `ヴウ'` answers 200 — as are `ヰ`,
#: `ヱ` and `ヶ`, measured at 200 the same way.
#:
#: Three kinds of kana are deliberately absent, and for each the fallback is a
#: refusal rather than a guess:
#:
#: - `ン` and `ッ` end on no vowel, so there is no answer to lengthen them
#:   with. The engine is no help here: it accepts `ンオ'` and `ッウ'` happily
#:   and pronounces the invented mora. `ッ` was briefly mapped to `ウ` for
#:   exactly that reason — it looked like it worked.
#: - `ヵ` has a vowel and the engine refuses it anyway: `ヵ'` and `ヵア'` both
#:   answer 400. A row here would trade one failed request for another, where
#:   refusing gets a real clip in the engine's own accent.
#: - Everything `_to_katakana` passes through without converting — `ヷヸヹヺ`,
#:   the half-width katakana block, anything not kana at all. `_to_katakana`
#:   shifts the hiragana block and folds U+FF70; it is not a general
#:   normalizer, and this table covers what it produces from hiragana.
_VOWEL_OF_ROW = {
    "ア": "アカサタナハマヤラワガザダバパャァヮ",
    "イ": "イキシチニヒミリギジヂビピィヰ",
    "ウ": "ウクスツヌフムユルグズヅブプュゥヴ",
    "エ": "エケセテネヘメレゲゼデベペェヶヱ",
    "オ": "オコソトノホモヨロヲゴゾドボポョォ",
}
_LONG_VOWEL_FOR = {
    kana: vowel for vowel, row in _VOWEL_OF_ROW.items() for kana in row
}


def _spell_long_vowels(units: list[str]) -> list[str]:
    """Replace each `ー` with the vowel of the mora it lengthens.

    Refuses rather than emitting a `ー` the engine will 400 on, wherever the
    kana before it has no row in :data:`_LONG_VOWEL_FOR`: a `ー` that opens a
    reading with nothing before it at all, a `ー` after `ン` or `ッ`, which end
    on no vowel, and a `ー` after anything with no row at all: `ヵ`, which the
    engine refuses either way, and everything :func:`_to_katakana` passes
    through without converting — `ヷヸヹヺ`, `・ヽヾヿ`, half-width katakana,
    and anything that is not kana.

    What a :class:`PitchError` here means depends on who asked. `audio_cmd`
    and the ledger voice the word with the engine's own accent — the same
    fallback as a record with no pattern at all, a worse clip than a forced one
    and a far better outcome than a failed request or an invented mora. jpdb
    enrichment asks a different question: it uses this to test whether a
    pattern converts, and files the refused ones as *unusable*. That is a
    split, not a rejection — jpdb often offers several and the usable ones are
    written — so the cost is only total when every offered pattern is refused
    and the record had no accent of its own. It is warned in every case.
    """
    spelled: list[str] = []
    for unit in units:
        # `startswith`, not `==`: a small kana attaches to the mora before it,
        # so `morae("かーょ")` is `['か', 'ーょ']` and the `ー` arrives heading a
        # two-character unit. Matching the bare mark alone left `カ'ーョ` — 400,
        # measured — reaching the engine through the one path this claims to
        # have closed.
        if not unit.startswith("ー"):
            spelled.append(unit)
            continue
        previous = spelled[-1] if spelled else ""
        vowel = _LONG_VOWEL_FOR.get(previous[-1:])
        if not vowel:
            raise PitchError(
                f"cannot spell out the long vowel in {''.join(units)!r} for "
                "VOICEVOX: "
                + (
                    "it opens with 'ー', which has nothing to lengthen"
                    if not previous
                    else f"'{previous}' has no vowel to repeat"
                )
                + ". The word is voiced with the engine's own accent instead."
            )
        spelled.append(vowel + unit[1:])
    return spelled


def to_aquestalk(reading: str, pattern: str) -> str:
    """``reading`` in AquesTalk kana notation, with ``pattern``'s accent forced.

    Katakana with exactly one :data:`ACCENT_MARK` after the accented mora, which
    is what ``/accent_phrases?is_kana=true`` takes. Raises :class:`PitchError`
    rather than returning something plausible for a pattern it cannot read.
    """
    per_mora, particle = _levels(reading, pattern)
    position = _accent_position(per_mora, particle)
    units = _spell_long_vowels([_to_katakana(mora) for mora in morae(reading)])
    units[position - 1] += ACCENT_MARK
    return "".join(units)


def select_pattern(record: VocabularyRecord) -> str | None:
    """Which pattern this record's audio should use, or ``None`` for no answer.

    ``audio_accent`` first, because it exists for the reader who listened and
    disagreed; then jpdb's primary, which is the first entry in its own
    ordering. ``None`` when the record carries no pattern at all — the caller
    decides what to do about that. It voices the word with the engine's own
    accent and marks the clip: the homographs a guess gets wrong are exactly
    the ones a pitch card exists for, so the guess is recorded rather than
    trusted, and it is replaced the moment a pattern arrives.

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


def _source_binding(reading: str, patterns: Sequence[str], source: str) -> str:
    values = [source.strip().lower(), unicodedata.normalize("NFC", reading)]
    values.extend(pattern.strip().upper() for pattern in patterns if pattern.strip())
    digest = hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()
    return f"{values[0]}:{digest}"


def bind_source(
    record: VocabularyRecord, source: str = "jpdb"
) -> VocabularyRecord:
    """Bind the current pitch value to the source that supplied it."""
    if not record.pitch_accent:
        return record
    raw_fields = dict(record.source.raw_fields)
    raw_fields[SOURCE_BINDING_KEY] = _source_binding(
        record.reading, record.pitch_accent, source
    )
    return replace(record, source=replace(record.source, raw_fields=raw_fields))


def has_source_binding(
    record: VocabularyRecord, source: str = "jpdb"
) -> bool:
    """Whether source metadata proves the current pitch value came from it."""
    if not record.pitch_accent:
        return False
    expected = _source_binding(record.reading, record.pitch_accent, source)
    return record.source.raw_fields.get(SOURCE_BINDING_KEY, "") == expected


def pattern_facts(reading: str, patterns: Sequence[str]) -> str:
    """Return the exact mora-level interpretation of stored raw patterns."""
    lines: list[str] = []
    for pattern in patterns:
        raw = pattern.strip().upper()
        if not raw:
            continue
        per_mora, particle = _levels(reading, raw)
        units = morae(reading)
        rendered = ", ".join(
            f"{unit}={level}" for unit, level in zip(units, per_mora, strict=True)
        )
        lines.append(f"raw {raw}; morae {rendered}; following particle={particle}")
    return "\n".join(lines)


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
