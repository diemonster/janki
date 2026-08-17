"""Kana to Hepburn romaji: one table, no dictionary, no network.

Romaji is rule-based work (``DESIGN_V2.md``, "Division of labor"), so it is
plain code: digraphs, gemination, and syllabic-``n`` assimilation are a table,
not a judgment call. This module is pure — no I/O, no config, no state — and
uses the standard library only.

The dialect, and why each choice was made:

* **Macron-free Hepburn.** Long vowels are written out (``ou``, ``uu``,
  ``oo``, ``ii``), never ``ō``/``ū``. That is the convention Shirabe Jisho
  exports use and the one already sitting in this repository's curated
  records (``ryouri``, ``Kinosaki Onsen``), so a generated value and a
  hand-written one look alike. It is also the only spelling that survives a
  round trip through ASCII-only tooling.
* **Modern Hepburn for ``ん``**: always ``n``, including before ``b``/``p``/``m``
  (しんぶん → ``shinbun``, こんばん → ``konban``), with ``n'`` before a vowel or
  ``y`` (きんえん → ``kin'en``, ほんや → ``hon'ya``). The apostrophe is what keeps
  ``kin'en`` from being read as ``ki-ne-n``.

  Traditional JR-station Hepburn spells that ``m`` — ``shimbun``, ``sampo`` —
  and this module did until 2026-08-17. The owner reads ``konban``, and a
  learner typing a word back into an IME gets ``ん`` from ``n`` and nothing
  from ``m``, so ``n`` is the spelling that round-trips.
* **``を`` is always ``o``.** Hepburn romanizes both the word-internal kana and
  the object particle as ``o`` (``hon o yomu``), which is what this repository's
  existing example romaji already does, so the mapping is unambiguous.
* **``は`` and ``へ`` are always ``ha`` and ``he``.** As particles they are read
  ``wa`` and ``e``, but telling a particle from a syllable needs word
  segmentation, which this module deliberately does not have. こんにちは comes
  out ``konnichiha``. A caller that knows its word boundaries (M4.2 feeds this
  from furigana segments) can convert segment by segment; whitespace in the
  input is preserved, so joined-up segments keep their spaces.

Everything else the scanner handles:

* katakana is folded to hiragana first, after NFKC (so halfwidth ｺｰﾋｰ and
  decomposed dakuten both work);
* ``ー`` repeats the previous vowel letter (コーヒー → ``koohii``);
* ``っ`` doubles the following consonant, becomes ``t`` before ``ch``
  (いっち → ``itchi``), and is dropped when there is no consonant to double;
* a small vowel that is not part of a known combination lengthens its host
  (ねぇ → ``nee``), while the known ones are spelled as loanwords are
  (フィルム → ``firumu``, チェック → ``chekku``).

**Anything it cannot convert confidently, it refuses to guess**: a kanji, an
iteration mark, a stray combining mark, ``ヶ`` (read ``ka``, ``ga`` or ``ko``
depending on the word) anywhere in the input makes the whole call return ``""``.
Empty means "no romaji available" and a caller can flag it; a half-converted
string looks like an answer and would be stored as one. Punctuation and ASCII
pass through, because neither carries a reading that could be got wrong.
"""

from __future__ import annotations

import unicodedata

__all__ = ["kana_to_romaji"]

_SOKUON = "っ"
_SYLLABIC_N = "ん"
_LONG_VOWEL_MARK = "ー"
# Sets, not strings: the lookahead is "" at the end of a run, and "" is a
# substring of every string, so `"" in "bpm"` would spell a word-final ん as m.
_VOWELS = frozenset("aeiou")
_LABIALS = frozenset("bpm")

# Katakana maps onto hiragana by a fixed offset over ァ..ヶ (U+30A1..U+30F6).
# The four code points just past that run — ヷヸヹヺ — must not be shifted:
# they would land on U+3097/U+3098 (unassigned) and U+3099/U+309A (the
# *combining* voiced sound marks), silently turning a word into a diacritic.
# They are spelled out instead.
_KATAKANA_FIRST = 0x30A1
_KATAKANA_LAST = 0x30F6
_KANA_OFFSET = 0x60
_KATAKANA_WITHOUT_HIRAGANA = {
    "ヷ": "ゔぁ",
    "ヸ": "ゔぃ",
    "ヹ": "ゔぇ",
    "ヺ": "ゔぉ",
}

# Two-kana units. Palatal digraphs first, then the combinations loanwords use
# to write sounds the gojūon has no kana for.
_DIGRAPHS: dict[str, str] = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "ぢゃ": "ja", "ぢゅ": "ju", "ぢょ": "jo",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "いぇ": "ye",
    "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "くぁ": "kwa", "くぃ": "kwi", "くぇ": "kwe", "くぉ": "kwo", "くゎ": "kwa",
    "ぐぁ": "gwa", "ぐぃ": "gwi", "ぐぇ": "gwe", "ぐぉ": "gwo", "ぐゎ": "gwa",
    "しぇ": "she", "じぇ": "je", "ちぇ": "che",
    "つぁ": "tsa", "つぃ": "tsi", "つぇ": "tse", "つぉ": "tso",
    "てぃ": "ti", "てゅ": "tyu", "でぃ": "di", "でゅ": "dyu",
    "とぅ": "tu", "どぅ": "du",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo", "ふゅ": "fyu",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo", "ゔゅ": "vyu",
}

# Single kana. The small kana are here as well, which is what makes a small
# vowel with no combination of its own lengthen the kana before it: ねぇ finds
# no digraph, so it reads as ね + ぇ -> "nee".
#
# Deliberately absent, each because it needs more than a table: ん and っ (both
# depend on what follows and are scanned separately); ゕ/ゖ, the small ka/ke of
# 一ヶ月 and 関ヶ原, which are read ka, ga or ko depending on the word; and the
# iteration marks ゝゞ, which repeat a kana this scanner would have to remember.
# Each of them makes the call return "" rather than pick a reading.
_MONOGRAPHS: dict[str, str] = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "i", "ゑ": "e", "を": "o",
    "ゔ": "vu",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ゎ": "wa",
}

# Marks with an unambiguous Latin equivalent. Anything else Unicode calls
# punctuation or a space passes through as itself (see _is_transparent): a
# bracket carries no reading, so copying it cannot produce a wrong one.
_PUNCTUATION: dict[str, str] = {
    "、": ", ",
    "。": ". ",
    "・": " ",
    "〜": "~",
    "…": "...",
    "「": '"',
    "」": '"',
    "『": '"',
    "』": '"',
}

_TRANSPARENT_CATEGORIES = frozenset({"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", "Zs"})


def _to_hiragana(text: str) -> str:
    """Fold katakana onto hiragana so one table serves both scripts."""
    folded: list[str] = []
    for char in text:
        spelled_out = _KATAKANA_WITHOUT_HIRAGANA.get(char)
        if spelled_out is not None:
            folded.append(spelled_out)
        elif _KATAKANA_FIRST <= ord(char) <= _KATAKANA_LAST:
            folded.append(chr(ord(char) - _KANA_OFFSET))
        else:
            folded.append(char)
    return "".join(folded)


def _read_unit(text: str, index: int) -> tuple[str, int] | None:
    """Romaji for the plain kana unit at ``index``, plus the index after it.

    Plain means table-driven: everything except ん, っ and ー, which depend on
    their neighbours. Returns ``None`` when nothing in the table starts here —
    which is also how the ん and っ rules learn that there is no next syllable.
    That includes the end of the string: ``index`` is a lookahead position, so
    a word ending in ん asks about the character after its last one.
    """
    if index >= len(text):
        return None
    digraph = _DIGRAPHS.get(text[index : index + 2])
    if digraph is not None:
        return digraph, index + 2
    monograph = _MONOGRAPHS.get(text[index])
    if monograph is not None:
        return monograph, index + 1
    return None


def _geminate(following: str) -> str:
    """The consonant っ contributes before ``following``."""
    if following.startswith("ch"):
        # Traditional Hepburn spells っち as tchi, not cchi: いっち -> itchi.
        return "t"
    head = following[:1]
    if head and head not in _VOWELS:
        return head
    # A sokuon before a vowel, or trailing the string (あっ), is a glottal stop
    # with no consonant to double. Hepburn drops it.
    return ""


def _syllabic_n(following: str) -> str:
    """How ん is spelled before ``following`` (empty at the end of a run).

    Never ``m``: see the module docstring. The apostrophe before a vowel or
    ``y`` is not cosmetic — without it ``kin'en`` reads as ``ki-ne-n``, which
    is a different word.
    """
    head = following[:1]
    if head in _VOWELS or head == "y":
        return "n'"
    return "n"


def _is_transparent(char: str) -> bool:
    return unicodedata.category(char) in _TRANSPARENT_CATEGORIES or (
        char.isascii() and (char.isprintable() or char.isspace())
    )


def kana_to_romaji(kana: str) -> str:
    """Romanize ``kana`` in macron-free Hepburn, or return ``""``.

    Hiragana, katakana (including halfwidth), punctuation and ASCII convert.
    Anything else — a kanji above all — means the reading is not knowable from
    a table, and the whole call returns ``""`` rather than a partly-guessed
    string. See the module docstring for the dialect and its known limits
    (particle は/へ, and no word segmentation).
    """
    text = _to_hiragana(unicodedata.normalize("NFKC", kana))
    romanized: list[str] = []
    last_vowel: str | None = None
    index = 0

    while index < len(text):
        char = text[index]

        if char == _SOKUON:
            following = _read_unit(text, index + 1)
            romanized.append(_geminate(following[0] if following else ""))
            index += 1
            continue

        if char == _SYLLABIC_N:
            following = _read_unit(text, index + 1)
            romanized.append(_syllabic_n(following[0] if following else ""))
            # ん ends on a consonant, so a ー after it has no vowel to repeat.
            last_vowel = None
            index += 1
            continue

        if char == _LONG_VOWEL_MARK:
            if last_vowel is None:
                return ""
            romanized.append(last_vowel)
            index += 1
            continue

        unit = _read_unit(text, index)
        if unit is not None:
            syllable, index = unit
            romanized.append(syllable)
            last_vowel = syllable[-1] if syllable[-1] in _VOWELS else None
            continue

        replacement = _PUNCTUATION.get(char)
        if replacement is not None:
            romanized.append(replacement)
            last_vowel = None
            index += 1
            continue

        if _is_transparent(char):
            romanized.append(char)
            last_vowel = None
            index += 1
            continue

        return ""

    # Whitespace is a word separator a caller supplied (furigana segments), so
    # it survives; the runs the punctuation table introduces are collapsed.
    return " ".join("".join(romanized).split())
