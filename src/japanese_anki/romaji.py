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

import re
import unicodedata

__all__ = ["accepting_pattern", "kana_to_romaji"]

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


#: What each ambiguous kana may legitimately be spelled as. Both are particles
#: read differently from the syllable they are written with, and which one a
#: given は is cannot be known without word segmentation — so a *verifier*
#: accepts either, where the generator has to pick one and picks the kana.
#:
#: Accepted **only where the answer is shaped the way the prompt asked** — the
#: spoken spelling as its own token. Keyed on the kana alone, every は in the
#: language could be spelled `wa`, so 花がきれい verified as `wana ga kirei`:
#: a wrong romaji reaching the one reader who cannot check it against the
#: kana. What remains uncaught is a `wa` standing alone that should have been
#: `ha`, which no check without a parse can see, and janki does not parse.
_PARTICLE_ALTERNATIVES = {"は": ("ha", "wa"), "へ": ("he", "e")}

#: `prompts/romaji.md` asks for particles as standalone tokens, so the spoken
#: spelling is accepted only where the answer is in that shape: a separator,
#: punctuation or the end after it.
#:
#: This is a check on the *format this project asked for*, not a claim about
#: Japanese. janki does not know which は is a particle and must not pretend
#: to — an earlier version of this also refused a sentence-initial `wa` on the
#: grounds that "a particle attaches to what precedes it", which is grammar
#: reasoned out in a Python file, exactly what DESIGN.md says does not belong
#: here.
_TOKEN_END = r"(?=[-\s,.?!;:'\"]|$)"

#: What may sit between two pieces without changing what the letters say: a
#: word space, or the hyphen romaji conventionally uses for a suffix.
_SEPARATOR = r"[-\s]*"


def accepting_pattern(kana: str) -> str:
    """A regex matching every spelling of ``kana`` a romanizer may defensibly
    produce, ignoring word spacing.

    For checking someone *else's* romaji rather than writing janki's own. The
    caller is a model that can segment — which is the thing :func:`kana_to_romaji`
    deliberately cannot do — so its output may legitimately differ from this
    module's in exactly three ways, and no others:

    * **word separators** — a space or a hyphen — anywhere, which is the whole
      reason to ask. A hyphen because `Tanaka-san` and `Yamada-kun` are how
      romaji conventionally joins a suffix, and rejecting them would mean the
      prompt asking for a spelling the verifier refuses;
    * **letter case**, since a proper noun takes a capital (`Nagoya`) and the
      kana does not record one;
    * **は as `wa` and へ as `e`**, but only written as a whole token with a
      separator before it — which is what a particle always is, and which
      `wana` for はな is not. This is a *guard*, not a parse: janki still
      cannot tell which は is a particle, so a particle written with no space
      around it is refused and a non-particle は spelled `wa` between two
      spaces would still slip through;
    * the **apostrophe** in ``n'``, which a writer may or may not type.

    Everything else — every consonant, every vowel, every geminate, every long
    vowel — has to agree exactly. A romaji that matches this therefore says
    what the kana says *up to* the slack above: the apostrophe is optional, so
    `kinen` passes for きんえん as well as きねん, and a `wa` standing alone
    between separators is taken on trust. Those are the two ways a matching
    romaji can still be wrong, and both are narrow. Everything outside them is
    rejected rather than silently kept. That is the guarantee `qc` cares about: a learner reading
    romaji is reading it *because* they cannot yet read the kana, so they
    cannot catch it being wrong.

    Returns ``""`` for kana this module cannot romanize at all, which callers
    must treat as "cannot verify" rather than as "matches nothing".
    """
    scanned = _scan(kana)
    if scanned is None:
        return ""
    parts = [_SEPARATOR]
    for source, piece in scanned:
        alternatives = _PARTICLE_ALTERNATIVES.get(source)
        if alternatives:
            written, spoken = alternatives
            parts.append(f"(?:{written}|{spoken}{_TOKEN_END})")
        elif piece == "n'":
            parts.append("n'?")
        elif not piece.strip():
            # Whitespace the *furigana* carried. It is ruby notation, not a
            # word boundary, so it neither has to be there nor has to be
            # absent — the separator between every piece already allows both.
            continue
        else:
            # Stripped before escaping: the punctuation table spells 。 as
            # ". " and 、 as ", ", and escaping that trailing space would make
            # it mandatory — so a sentence ending in "." rather than ". "
            # would fail to match itself.
            parts.append(re.escape(piece.strip()))
        parts.append(_SEPARATOR)
    return "".join(parts)


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
    scanned = _scan(kana)
    if scanned is None:
        return ""
    pieces = [piece for _source, piece in scanned]
    # Whitespace is a word separator a caller supplied (furigana segments), so
    # it survives; the runs the punctuation table introduces are collapsed.
    return " ".join("".join(pieces).split())


def _scan(kana: str) -> list[tuple[str, str]] | None:
    """``kana`` as ``(source unit, romaji piece)`` pairs, or ``None``.

    Split out from :func:`kana_to_romaji` so :func:`accepting_pattern` can walk
    the same scan and widen a single decision per unit, rather than keeping a
    second copy of the gemination, prolongation and syllabic-``n`` rules that
    would drift from this one. The source unit rides along because the two
    ambiguous kana — は and へ — are only identifiable before romanization:
    ``ha`` in the output could have come from は or from a は inside a word,
    and by then they are the same three letters.
    """
    text = _to_hiragana(unicodedata.normalize("NFKC", kana))
    romanized: list[tuple[str, str]] = []
    last_vowel: str | None = None
    index = 0

    while index < len(text):
        char = text[index]

        if char == _SOKUON:
            following = _read_unit(text, index + 1)
            romanized.append((char, _geminate(following[0] if following else "")))
            index += 1
            continue

        if char == _SYLLABIC_N:
            following = _read_unit(text, index + 1)
            romanized.append((char, _syllabic_n(following[0] if following else "")))
            # ん ends on a consonant, so a ー after it has no vowel to repeat.
            last_vowel = None
            index += 1
            continue

        if char == _LONG_VOWEL_MARK:
            if last_vowel is None:
                return None
            romanized.append((char, last_vowel))
            index += 1
            continue

        unit = _read_unit(text, index)
        if unit is not None:
            syllable, next_index = unit
            romanized.append((text[index:next_index], syllable))
            index = next_index
            last_vowel = syllable[-1] if syllable[-1] in _VOWELS else None
            continue

        replacement = _PUNCTUATION.get(char)
        if replacement is not None:
            romanized.append((char, replacement))
            last_vowel = None
            index += 1
            continue

        if _is_transparent(char):
            romanized.append((char, char))
            last_vowel = None
            index += 1
            continue

        return None

    return romanized
