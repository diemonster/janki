from __future__ import annotations

import hashlib
import re
import unicodedata

from japanese_anki.errors import JankiError


class IdentityError(JankiError):
    """A durable identity cannot be minted from what was supplied."""


def normalize_identity_part(value: str) -> str:
    """Normalize text used in durable record identities."""
    return unicodedata.normalize("NFKC", value).strip()


# Every Unicode range that holds Han characters a Japanese record can carry,
# inclusive. Kanji live in blocks spread across three planes (0, 2 and 3), not
# one contiguous BMP run: 𠮟 (U+20B9F, the JIS2004 form of しかる) and 𩸽
# (U+29E3D, hokke) are ordinary Japanese words that Shirabe exports, and both
# sit above U+FFFF.
#
# These ranges list ideographs only. The Han blocks that are *spellings* of an
# ideograph — Kangxi Radicals ⼀ U+2F00, CJK Radicals Supplement, the Hangzhou
# numerals, circled and squared forms ㊀ U+3280 / ㍻ U+337B — are deliberately
# absent: ``contains_kanji`` normalizes first, which folds every one of them
# into a code point that is already here. Adding them would be a second, drifting
# answer to a question NFKC has already settled.
_HAN_RANGES: tuple[tuple[int, int], ...] = (
    (0x3005, 0x3005),  # 々, the iteration mark: kanji for identity purposes
    (0x3007, 0x3007),  # 〇, ideographic number zero: a numeral kanji (れい/まる)
    (0x303B, 0x303B),  # 〻, the vertical iteration mark: Script=Han like 々
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2EE5F),  # CJK Unified Ideographs Extensions B through I (plane 2)
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    (0x30000, 0x3347F),  # CJK Unified Ideographs Extensions G, H and J (plane 3)
)


def han_character_class() -> str:
    """The Han ranges as a regex character-class body, without the brackets.

    So a caller that needs to *match* kanji inside a larger pattern gets the
    same definition :func:`contains_kanji` decides with, instead of restating a
    narrower one. `patterns._WORD` had `一-龥` (U+4E00–U+9FA5) written out by
    hand, which misses 𠮟 — named in the note above as a word real exports
    carry — and every Extension-A ideograph.

    Callers matching raw document text should normalize it first, as
    `contains_kanji` does: these ranges are ideographs only, and the 450 code
    points that merely *spell* an ideograph fold into them under NFKC.
    """
    return "".join(
        re.escape(chr(low)) if low == high else f"{re.escape(chr(low))}-{re.escape(chr(high))}"
        for low, high in _HAN_RANGES
    )


def contains_kanji(value: str) -> bool:
    """Whether ``value`` holds a Han character.

    One definition for the whole project: the importer decides with it whether
    a row may be imported at all, and the validator reports with it what got in
    anyway. Two copies would drift, and a copy that is too narrow does not miss
    a warning — it mints ``word:<kanji>:<kanji>``, an ID that can never be
    corrected without orphaning Anki review history.

    **The test is on the normalized form**, the same string
    :func:`stable_record_id` would put in the ID. That is what makes the gate
    and the ID-minter incapable of disagreeing: 450 assigned code points —
    Kangxi Radicals, the CJK Radicals Supplement, Hangzhou numerals, circled
    and squared ideographs — are not ideographs themselves but NFKC-fold into
    one, so a raw-string test called ⼀ (U+2F00) non-kanji while the minter
    turned it into 一 (U+4E00) and stamped ``word:一:一``. Widening the ranges
    instead has been tried twice and recurred twice; normalizing ends the class.

    Iteration is over code points (Python strings iterate that way, so a
    supplementary-plane kanji is one character here rather than two surrogate
    halves) and the ranges cover every plane kanji live in.
    """
    return any(
        any(low <= code <= high for low, high in _HAN_RANGES)
        for code in map(ord, normalize_identity_part(value))
    )


#: The kana a *reading* is written in, plus the marks that ride along with one:
#: the prolonged sound mark, the iteration marks, and the combining dakuten a
#: decomposed が is spelled with. Katakana is here because a reading copied off a
#: dictionary is sometimes written in it, and half-width katakana because that is
#: what an old export carries; NFKC folds the latter before this ever sees it.
_KANA_RANGES: tuple[tuple[int, int], ...] = (
    (0x3041, 0x309F),  # hiragana, with ゝゞ and the combining voiced marks
    (0x30A0, 0x30FF),  # katakana, with ・ー and ヽヾ
    (0x31F0, 0x31FF),  # katakana phonetic extensions (ㇰ, small kana for Ainu)
)


def is_kana(value: str) -> bool:
    """Whether every character of ``value`` is kana.

    Not the negation of :func:`contains_kanji`: "no kanji" is true of `to speak`
    and of an empty string, and a caller asking this question — which line of a
    shared deck's HTML is the reading, and which is the English — needs the
    positive test. Empty is false: nothing is not a reading.

    Normalized first, for the same reason `contains_kanji` is: half-width
    katakana ｶﾅ folds to katakana, and a raw test would call a real reading
    something else.
    """
    text = normalize_identity_part(value)
    return bool(text) and all(
        any(low <= code <= high for low, high in _KANA_RANGES) for code in map(ord, text)
    )


def stable_record_id(expression: str, reading: str = "") -> str:
    expression_part = normalize_identity_part(expression)
    reading_part = normalize_identity_part(reading)
    return f"word:{expression_part}:{reading_part}"


def character_record_id(character: str) -> str:
    """The durable identity of one character note: ``kanji:理``.

    Beside :func:`stable_record_id`, normalized the same way, and deliberately
    *not* built from a reading: 理 is one character whatever it is read as, and
    a note whose identity moved when its readings were refreshed would strand
    the review history the GUID exists to keep.

    Exactly one character, refused rather than truncated or split. ``料理`` is
    two characters and so two notes; minting ``kanji:料理`` would put a word
    under a character identity, which is the one thing this store must not
    hold. Nothing here reads the character — a single code point is the whole
    contract.
    """
    part = normalize_identity_part(character)
    if len(part) != 1:
        raise IdentityError(
            f"A character note is one character; {character!r} is "
            f"{len(part)} after normalization."
        )
    return f"kanji:{part}"


def short_fingerprint(*values: str, length: int = 12) -> str:
    payload = "\x1f".join(normalize_identity_part(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
