"""Furigana notation arithmetic: reading Anki's ruby format, never judging it.

M8.3 deleted the checks this module used to hold — headword containment,
teaching-suitability, the KANJIDIC comparison, the spill rules — because
janki's logic enriches the card and never audits the model (DESIGN.md). What
remains reads notation:

* :func:`furigana_pairs` — the ``(base, reading)`` groups a field states.
* :func:`furigana_reading` — the kana a field spells, notation spaces dropped.
* :func:`settle_example_romaji` — romaji checked against that reading.
  Model-supplied romaji is discarded outright: romaji is a mechanical
  transliteration, so there is no reason to accept a guess at one.

Pure functions, no CLI and no network.
"""

from __future__ import annotations

import re
from dataclasses import replace

from japanese_anki.identifiers import (
    normalize_identity_part,
)
from japanese_anki.models import (
    ExampleSentence,
)
from japanese_anki.romaji import accepting_pattern, kana_to_romaji

__all__ = [
    "furigana_pairs",
    "furigana_reading",
    "settle_example_romaji",
]

# One bracketed group in Anki furigana notation: the run of characters before
# ``[``, up to a space, and the reading inside the brackets. The space is what
# separates a ruby group from kana in front of it, which is why it terminates
# the text run.
#
# **The ASCII space, and only that.** Anki's own furigana filter separates on
# it alone, so a full-width space — ordinary in Japanese text, and what a model
# may well write — does not end the run there either. Treating it as a
# separator here would verify ``お　茶[ちゃ]`` while Anki renders ちゃ over both
# characters: the same wrong-ruby, truncated-audio failure a missing space
# causes, arriving through a different character.
_GROUP = re.compile(r"([^ \[\]]+)\[([^\[\]]+)\]")


def furigana_pairs(furigana: str) -> tuple[tuple[str, str], ...]:
    """The ``(text, reading)`` groups in Anki furigana notation, in order.

    Normalized, like everything else these are compared against: a reading
    written with a decomposed dakuten would otherwise fail against jpdb's
    composed one and report ``jpdb reads 語 as ご, not ご`` — two strings that
    render identically, so the rejection cannot be diagnosed at all.
    """
    return tuple(
        (normalize_identity_part(match.group(1)), normalize_identity_part(match.group(2)))
        for match in _GROUP.finditer(furigana)
    )


def furigana_reading(furigana: str) -> str:
    """The kana a whole sentence's Anki furigana spells out.

    Bracketed groups contribute their reading, everything else contributes
    itself — so ``話[はな]すを 食[た]べる`` reads ``はなすをたべる``.

    Only the single ASCII space Anki's notation requires *immediately before a
    ruby group* is dropped. Every other space is content: a sentence quoting
    ``「Hello World」`` would otherwise come back as ``HelloWorld``, since
    :func:`romaji.kana_to_romaji` passes Latin text through verbatim. A space
    between two ASCII words is provably not notation.
    """
    out: list[str] = []
    position = 0
    for match in _GROUP.finditer(furigana):
        chunk = furigana[position : match.start()]
        if chunk.endswith(" "):
            chunk = chunk[:-1]
        out.append(chunk)
        out.append(match.group(2))
        position = match.end()
    out.append(furigana[position:])
    return "".join(out)


def settle_example_romaji(example: ExampleSentence) -> tuple[ExampleSentence, str]:
    """The example with its romaji settled, and why, if a supplied one was
    thrown away.

    **Checked, not regenerated.** Romaji word spacing *is* word segmentation,
    and segmentation is parsing — the model's job, not janki's (`DESIGN.md`,
    "the LLM parses, the dictionaries enrich"). This function used to discard
    whatever romaji arrived and rebuild it from the furigana, on the reasoning
    that a wrong romaji is invisible to a learner who is reading it *because*
    they cannot yet read the kana. The reasoning is right; the conclusion was
    wrong. Furigana spacing is Anki's ruby notation rather than word
    boundaries, so rebuilding could only ever produce `hahanidenwao` for
    母に電話を, and it spelled the topic particle は as `ha`. It traded an
    occasionally wrong romaji for a reliably wrong one.

    So a supplied romaji is *verified* against the reading janki already has.
    :func:`romaji.accepting_pattern` allows word spaces anywhere, は as `wa`,
    へ as `e`, and an optional apostrophe in `n'`, and requires every other
    letter to agree. A romaji that matches says what the kana says and is
    kept, spacing and all. One that does not is replaced by the mechanical
    transliteration and *named*: a rejection means the sentence and its romaji
    disagree, which is worth a human's attention rather than a silent repair.

    An example with no romaji is transliterated with no complaint — the
    ordinary path for records that predate the prompt asking for one, and for
    every pass that sends no model at all.
    """
    reading = (
        furigana_reading(example.furigana) if example.furigana else example.japanese
    )
    mechanical = kana_to_romaji(reading) if reading else ""
    supplied = (example.romaji or "").strip()
    if not supplied:
        return replace(example, romaji=mechanical), ""

    pattern = accepting_pattern(reading) if reading else ""
    if not pattern:
        # Nothing to check against: the reading still holds kanji, so janki
        # does not know what it says either. Keeping the supplied value would
        # be trusting it for precisely the reason it cannot be trusted.
        return replace(example, romaji=mechanical), ""
    if re.fullmatch(pattern, supplied):
        return replace(example, romaji=" ".join(supplied.split())), ""
    return replace(example, romaji=mechanical), (
        f"romaji {supplied!r} does not transliterate {reading!r}; "
        f"replaced with {mechanical!r}"
    )
