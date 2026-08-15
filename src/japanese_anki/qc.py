"""Furigana notation arithmetic: reading Anki's ruby format, never judging it.

M8.3 deleted the checks this module used to hold — headword containment,
teaching-suitability, the KANJIDIC comparison, the spill rules — because
janki's logic enriches the card and never audits the model (DESIGN.md). What
remains reads notation:

* :func:`furigana_pairs` — the ``(base, reading)`` groups a field states.
* :func:`furigana_reading` — the kana a field spells, notation spaces dropped.
* :func:`regenerate_example_romaji` — romaji rebuilt from that reading.
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
from japanese_anki.romaji import kana_to_romaji

__all__ = [
    "furigana_pairs",
    "furigana_reading",
    "regenerate_example_romaji",
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


def regenerate_example_romaji(example: ExampleSentence) -> ExampleSentence:
    """The example with its romaji rebuilt from its furigana.

    Whatever romaji the example arrived with is discarded rather than checked.
    Romaji is a mechanical transliteration of a reading janki already has, so
    there is nothing a model could contribute but an opportunity to be wrong —
    and a wrong romaji is invisible to a learner who is reading it *because*
    they cannot yet read the kana.

    A sentence with no furigana is romanized from its own text, which is right
    when it holds no kanji and refuses to guess when it does: transliterating
    kanji is exactly the invention this function exists to remove.

    **The ASCII space before a ruby group is dropped; everything else stays.**
    That space is Anki's notation, and jpdb segments per kanji, so the verified
    furigana for 日本語 — ``日[にっ] 本[ぽん] 語[ご]`` — romanizes as
    ``nippongo``. Keeping those spaces would give ``ni pon go``: one word split
    into three, with the っ deleted because a sokuon at the end of a run has
    nothing to geminate.

    The rule is positional, not intentional, so a space someone *typed* right
    before a ruby group goes too — ``本を 食[た]べる`` reads ``本をたべる``,
    with the typed space gone. In that position a word space and a notation
    space are indistinguishable in the field. Everywhere else content spacing
    survives into the romaji: quoted Latin keeps its words apart, and so does a
    full-width space between two runs.

    What this does *not* do is insert word boundaries that were not already
    there. Real ones need the parse's tokens, which this function is not given;
    a caller that has one (M4.2) can do better, and :mod:`romaji` is built to
    accept it. The other inherited limit is :mod:`romaji`'s own: は and へ
    romanize as ``ha`` and ``he`` even as particles, because telling a particle
    from a syllable needs segmentation that module deliberately does not have.
    """
    reading = (
        furigana_reading(example.furigana) if example.furigana else example.japanese
    )
    return replace(example, romaji=kana_to_romaji(reading) if reading else "")
