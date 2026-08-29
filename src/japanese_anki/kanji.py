"""What a card can say about the *characters* in a word.

Stroke order, on/kun readings, and a common word per reading — the three blocks
of a paper kanji card. janki's records are words (``word:使う:つかう``), so this
is deliberately *not* record content: 前 is the same 前 in 名前 and 前線, and
storing its readings on every word that contains it would be the same lookup
repeated and the same edit needed in several places. It lives in its own file,
keyed by character, and the exporter reads it when it builds a card.

**Two sources, both looked up rather than generated.**

* `kanjiapi.dev` serves KANJIDIC2: stroke count, grade, JLPT level, meanings,
  and on/kun readings — plus the words that use a character.
* KanjiVG supplies the stroke *paths*, in order, which is what makes a
  stroke-by-stroke diagram possible rather than a single glyph.

**Example words need ranking, badly.** The word list for 前 is 740 entries and
arrives in an order that opens 一歩前進, 前官礼遇, 前駆体 — accurate and useless
to a learner. JMdict's priority tags are the signal: of those 740, 138 carry
one, and the three a commercial paper card chose (前線, 名前, 目の前) are all in
that set with ``nf08``, ``nf02`` and ``nf07``. So candidates are filtered to
tagged entries and ranked by the ``nfXX`` band, which is a frequency decile.

**Attribution.** KANJIDIC2 and JMdict are CC BY-SA 4.0 (EDRDG), and KanjiVG is
CC BY-SA 3.0 (Ulrich Apel). A personal deck is fine; a deck that is *shared*
has to credit all three, the same way VOICEVOX's per-character terms apply only
on sharing.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji

__all__ = [
    "Example",
    "KanjiError",
    "KanjiInfo",
    "Reading",
    "assigns_a_known_reading",
    "fetch_kanji",
    "kanji_in",
    "urllib_transport",
]

KANJIAPI = "https://kanjiapi.dev/v1"
KANJIVG = "https://raw.githubusercontent.com/KanjiVG/kanjivg/master/kanji"

#: Politeness, and a practical need: the default urllib agent gets a 403 from
#: kanjiapi.dev.
USER_AGENT = "janki (personal Japanese deck builder)"

DEFAULT_TIMEOUT = 30.0

#: How many example words to keep per reading. Two is what fits a card without
#: turning the section into a dictionary — the thing the meanings cap exists to
#: prevent one block higher up.
EXAMPLES_PER_READING = 2

#: Rows of example words per character. KANJIDIC lists every reading a
#: character has, including rendaku variants (使: つか.い *and* -づか.い), and
#: all of them is a dictionary entry rather than a reminder.
MAX_EXAMPLE_ROWS = 4

#: ``(url) -> bytes``. The seam tests replace, so no test reaches the network.
Transport = Callable[[str], bytes]


class KanjiError(JankiError):
    """A kanji source refused, or answered with something unusable."""


@dataclass(frozen=True, slots=True)
class Example:
    """One word that uses the character, for one of its readings."""

    written: str
    pronounced: str
    gloss: str


@dataclass(frozen=True, slots=True)
class Reading:
    """One reading of a character, with the words that show it in use."""

    kind: str  # "on" or "kun"
    reading: str
    examples: tuple[Example, ...] = ()


@dataclass(frozen=True, slots=True)
class KanjiInfo:
    """Everything a card shows about one character."""

    character: str
    stroke_count: int = 0
    grade: int | None = None
    jlpt: int | None = None
    meanings: tuple[str, ...] = ()
    readings: tuple[Reading, ...] = ()
    #: SVG path data, one per stroke, in writing order.
    strokes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "stroke_count": self.stroke_count,
            "grade": self.grade,
            "jlpt": self.jlpt,
            "meanings": list(self.meanings),
            "readings": [
                {
                    "kind": reading.kind,
                    "reading": reading.reading,
                    "examples": [
                        {"written": e.written, "pronounced": e.pronounced, "gloss": e.gloss}
                        for e in reading.examples
                    ],
                }
                for reading in self.readings
            ],
            "strokes": list(self.strokes),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> KanjiInfo:
        character = str(raw.get("character") or "")
        if not character:
            raise KanjiError("A kanji entry needs its character")
        readings = []
        for item in raw.get("readings") or []:
            examples = tuple(
                Example(
                    written=str(e.get("written") or ""),
                    pronounced=str(e.get("pronounced") or ""),
                    gloss=str(e.get("gloss") or ""),
                )
                for e in (item.get("examples") or [])
            )
            readings.append(
                Reading(
                    kind=str(item.get("kind") or ""),
                    reading=str(item.get("reading") or ""),
                    examples=examples,
                )
            )
        return cls(
            character=character,
            stroke_count=int(raw.get("stroke_count") or 0),
            grade=raw.get("grade"),
            jlpt=raw.get("jlpt"),
            meanings=tuple(str(m) for m in (raw.get("meanings") or [])),
            readings=tuple(readings),
            strokes=tuple(str(s) for s in (raw.get("strokes") or [])),
        )


def kanji_in(text: str) -> list[str]:
    """The distinct kanji of a string, in the order they appear.

    Order matters: a card shows them left to right as the word is written, and
    a set would put 前線 in whichever order the hash landed.
    """
    seen: dict[str, None] = {}
    for character in text:
        if contains_kanji(character):
            seen.setdefault(character, None)
    return list(seen)


def urllib_transport(url: str, timeout: float = DEFAULT_TIMEOUT) -> bytes:
    """One GET. Anything that goes wrong arrives as a :class:`KanjiError`."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise KanjiError(f"{url} answered {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise KanjiError(f"Could not reach {url}: {exc}") from exc


def _to_hiragana(text: str) -> str:
    return "".join(
        chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text
    )


def _match_key(reading: str) -> str:
    """The kana a word must contain to be an example of this reading.

    The **whole** reading, okurigana included: ``つか.う`` and ``つか.い`` are two
    readings of 使 that differ only after the dot, and matching on the stem
    alone makes 使う and 使い方 examples of both. It also let 教科書 (きょうかしょ)
    answer to 書's か.く, because か is inside きょうかしょ.

    The dot is a boundary marker rather than a sound, and ``-`` marks a suffix
    position, so both are dropped; on'yomi are katakana while words are
    hiragana, so the result is converted.
    """
    return _to_hiragana(reading.replace(".", "").replace("-", "").strip())


def _dedupe_key(reading: str) -> str:
    """What makes two KANJIDIC readings the same reading.

    Position and okurigana-boundary markers are notation rather than sounds.
    Thus ``-い.き`` and ``-いき`` are one reading written twice, while
    ``つか.う`` and ``つか.い`` remain distinct because their full readings
    differ. Collapsing those two labelled 使う as つか.い, which is simply the
    wrong reading for that word.
    """
    return _match_key(reading)


def assigns_a_known_reading(info: KanjiInfo | None, reading: str) -> bool:
    """Could this character be read this way, per KANJIDIC?

    Asked of a *per-character* furigana split, which jpdb hands back for every
    compound: `明日` arrives as ``[["明","あ"], ["日","した"]]``. あ is a real
    reading of 明 (あ.かり, あ.くる); した is no reading of 日 at all, because
    あした is *jukujikun* — the reading belongs to the compound and not to its
    characters. A card built from that split teaches two readings that do not
    exist.

    True when the store has never heard of the character: silence is not
    disagreement, and a lookup nobody has run must not start rejecting furigana
    the dictionary supplied.

    Matched on the stem, since a per-character reading carries no okurigana:
    the split gives 明→あ where the store holds あ(かり), and requiring the whole
    reading would reject every ordinary kun reading in the language. The store
    keeps readings in display form — ``あ(かり)``, ``〜あ(け)`` — so the stem is
    what precedes the parenthesis, position markers removed. Common compound
    sound changes also count: initial rendaku (口[ぐち] from くち) and a final
    sokuon that replaces a listed final mora (学[がっ] from がく).
    """
    if info is None:
        return True
    wanted = _to_hiragana(reading.strip())
    if not wanted:
        return True
    unvoiced = {
        "が": "か", "ぎ": "き", "ぐ": "く", "げ": "け", "ご": "こ",
        "ざ": "さ", "じ": "し", "ず": "す", "ぜ": "せ", "ぞ": "そ",
        "だ": "た", "ぢ": "ち", "づ": "つ", "で": "て", "ど": "と",
        "ば": "は", "び": "ひ", "ぶ": "ふ", "べ": "へ", "ぼ": "ほ",
        "ぱ": "は", "ぴ": "ひ", "ぷ": "ふ", "ぺ": "へ", "ぽ": "ほ",
    }
    wanted_forms = {wanted}
    if wanted[0] in unvoiced:
        wanted_forms.add(unvoiced[wanted[0]] + wanted[1:])
    for item in info.readings:
        text = item.reading.replace("〜", "").replace("-", "").strip()
        stem = _to_hiragana(text.split("(")[0].split(".")[0].strip())
        for known in {_match_key(text), stem}:
            if known in wanted_forms:
                return True
            if any(
                form.endswith("っ")
                and len(form) == len(known)
                and known.startswith(form[:-1])
                for form in wanted_forms
            ):
                return True
    return False


def _display_reading(reading: str) -> str:
    """A reading for a card: ``つか.う`` becomes ``つか(う)``.

    KANJIDIC's dot is machine notation for where the kanji stops and the
    okurigana begins. The information is worth keeping — it is why 使う is
    written with one kana after the character — but a bare dot on a card reads
    as a typo, which is exactly how it was reported. A leading ``-`` is not
    discarded: it becomes ``〜`` so a suffix-only reading such as ``-づか.い``
    is not presented as though the character were read づかい on its own.
    """
    text = reading.strip()
    position = "〜" if text.startswith("-") else ""
    text = text.removeprefix("-")
    if "." not in text:
        return f"{position}{text}"
    stem, _, okurigana = text.partition(".")
    shown = f"{stem}({okurigana})" if okurigana else stem
    return f"{position}{shown}"


def _priority_rank(priorities: Iterable[str]) -> int:
    """Lower is commoner. ``None``-ish entries sort last.

    ``nfXX`` is a frequency decile band in JMdict — ``nf01`` is the commonest —
    and ``ichi1``/``news1``/``spec1`` mark the common lists. Without this the
    word list for 前 opens on 前官礼遇 and 前駆体.
    """
    best = 999
    for tag in priorities:
        match = re.fullmatch(r"nf(\d+)", tag)
        if match:
            best = min(best, int(match.group(1)))
        elif tag in {"ichi1", "news1", "spec1", "gai1"}:
            best = min(best, 50)
    return best


def _shows_reading(character: str, written: str, pronounced: str, stem: str) -> bool:
    """Can this word *prove* that this character is read this way?

    A substring test cannot. 図書館 contains か, so 書's か.く claimed it as an
    example — but that か is 館's, and the card then asserted a reading nobody
    verified. 教科書 answered the same way (か from 科), 部分 answered 分's ブ
    (it is ブン), and 絵を描く answered 書 without containing 書 at all, because
    JMdict lists the spellings together.

    What *is* provable is position. If the character opens the word, its reading
    opens the pronunciation; if it closes the word, its reading closes the
    pronunciation. A character in the middle of a compound cannot be pinned to
    any run of kana without knowing how its neighbours are read, so those are
    refused rather than guessed at — the rule this project applies to every
    other reading.
    """
    # No separate "is the character even in the word" guard: a word the
    # character neither opens nor closes cannot be proved either way, and that
    # includes a word it is absent from — 絵を描く, which JMdict lists under 書
    # because the spellings share an entry.
    if written.startswith(character):
        return pronounced.startswith(stem)
    if written.endswith(character):
        return pronounced.endswith(stem)
    return False


def _ranked_examples_for(
    character: str,
    reading: str,
    words: list[dict[str, Any]],
    rivals: Iterable[str] = (),
) -> tuple[tuple[int, Example], ...]:
    """Ranked words that show this character being read this way.

    ``rivals`` are the character's other readings. A word goes to the *longest*
    reading that fits it, because one reading is often a prefix of another: 分's
    ブ and ブン both open 分野 (ぶんや), and offering 分野 as an example of ブ
    teaches a reading the word does not use.
    """
    stem = _match_key(reading)
    if not stem:
        return ()
    longer = sorted(
        (other for other in {_match_key(r) for r in rivals} if len(other) > len(stem)),
        key=len,
        reverse=True,
    )
    scored: list[tuple[int, Example]] = []
    for entry in words:
        glosses = entry.get("meanings") or [{}]
        gloss = (glosses[0].get("glosses") or [""])[0]
        for variant in entry.get("variants") or []:
            priorities = variant.get("priorities") or []
            if not priorities:
                # Dropped rather than merely ranked last. The sort below would
                # bury them anyway, but only while something tagged exists to
                # bury them under — for a rare character with no common words
                # at all, ranking alone would surface 前官礼遇-grade entries as
                # though they were the ones to learn.
                continue
            pronounced = _to_hiragana(str(variant.get("pronounced") or ""))
            written = str(variant.get("written") or "")
            if not _shows_reading(character, written, pronounced, stem):
                continue
            if any(
                _shows_reading(character, written, pronounced, other) for other in longer
            ):
                # A longer reading of the same character also fits, so this word
                # is that reading's example rather than this one's.
                continue
            scored.append((
                _priority_rank(priorities),
                Example(
                    written=str(variant.get("written") or ""),
                    pronounced=pronounced,
                    gloss=str(gloss),
                ),
            ))
    scored.sort(key=lambda pair: (pair[0], len(pair[1].written)))
    seen: set[str] = set()
    kept: list[tuple[int, Example]] = []
    for rank, example in scored:
        if example.written in seen:
            continue
        seen.add(example.written)
        kept.append((rank, example))
        if len(kept) >= EXAMPLES_PER_READING:
            break
    return tuple(kept)


def _examples_for(
    character: str,
    reading: str,
    words: list[dict[str, Any]],
    rivals: Iterable[str] = (),
) -> tuple[Example, ...]:
    """The capped examples stored for one reading, commonest first."""
    ranked = _ranked_examples_for(character, reading, words, rivals)
    return tuple(example for _, example in ranked[:EXAMPLES_PER_READING])


def fetch_kanji(character: str, *, transport: Transport | None = None) -> KanjiInfo:
    """Look one character up, from both sources.

    A missing stroke diagram is not a failure: KanjiVG does not cover every
    character, and readings without strokes are still most of the card.
    """
    if not character or not contains_kanji(character):
        raise KanjiError(f"Not a kanji: {character!r}")
    send = transport or urllib_transport

    quoted = urllib.parse.quote(character)
    try:
        info = json.loads(send(f"{KANJIAPI}/kanji/{quoted}"))
    except (ValueError, TypeError) as exc:
        raise KanjiError(f"kanjiapi gave no readable answer for {character}") from exc
    try:
        words = json.loads(send(f"{KANJIAPI}/words/{quoted}"))
    except KanjiError:
        words = []
    except (ValueError, TypeError):
        words = []
    if not isinstance(words, list):
        words = []

    # Deduplicated by the complete spoken reading: KANJIDIC sometimes writes
    # the same sound twice with different boundary notation (``-い.き`` and
    # ``-いき``). The plain-position form wins when both it and a suffix form
    # exist, while genuinely different okurigana remain separate readings.
    ranked_readings: list[tuple[int, int, Reading]] = []
    seen_stems: set[tuple[str, str]] = set()
    all_readings = [
        str(value)
        for key in ("on_readings", "kun_readings")
        for value in (info.get(key) or [])
    ]
    for kind, key in (("on", "on_readings"), ("kun", "kun_readings")):
        for value in sorted((info.get(key) or []), key=lambda r: ("-" in r, r)):
            text = str(value)
            marker = (kind, _dedupe_key(text))
            if not marker[1] or marker in seen_stems:
                continue
            seen_stems.add(marker)
            ranked_examples = _ranked_examples_for(
                character, text, words, all_readings
            )
            reading = Reading(
                kind=kind,
                reading=_display_reading(text),
                examples=tuple(
                    example
                    for _, example in ranked_examples[:EXAMPLES_PER_READING]
                ),
            )
            best_rank = ranked_examples[0][0] if ranked_examples else 999
            ranked_readings.append((best_rank, len(ranked_readings), reading))

    # The renderer has a deliberately small row budget. KANJIDIC's kana order
    # does not express usefulness, so let the best JMdict priority tag decide
    # which readings reach that budget; source order is the stable tie-breaker.
    readings = [reading for _, _, reading in sorted(ranked_readings)]

    try:
        svg = send(f"{KANJIVG}/{ord(character):05x}.svg").decode("utf-8")
        strokes = tuple(re.findall(r'<path\b[^>]*\bd="([^"]+)"', svg))
    except KanjiError:
        strokes = ()

    return KanjiInfo(
        character=character,
        stroke_count=int(info.get("stroke_count") or 0),
        grade=info.get("grade"),
        jlpt=info.get("jlpt"),
        meanings=tuple(str(m) for m in (info.get("meanings") or [])),
        readings=tuple(readings),
        strokes=strokes,
    )


#: KanjiVG draws on a 109x109 canvas.
CANVAS = 109


def render_kanji_html(
    entries: Iterable[KanjiInfo],
    *,
    record_expression: str = "",
    record_reading: str = "",
) -> str:
    """The collapsible block a card shows, or ``""`` when there is nothing.

    One ``<details>`` per character so a two-kanji word does not force the
    reader to open both, and so the summary can carry the character itself.
    When the shared reference entry contains the card's exact dictionary pair,
    that row is reserved before the display cap.  Exact text equality is a
    structural match; this renderer does not decide how any Japanese is read.
    """
    import html as html_mod

    blocks: list[str] = []
    for block_index, info in enumerate(entries):
        parts: list[str] = []
        header = html_mod.escape("、".join(info.meanings[:3]))
        tags = []
        if info.jlpt:
            tags.append(f"N{info.jlpt}")
        if info.stroke_count:
            tags.append(f"{info.stroke_count}画")
        if info.grade:
            tags.append(f"grade {info.grade}")
        parts.append(
            f'<div class="kanji-head"><span class="kanji-gloss">{header}</span>'
            f'<span class="kanji-tags">{html_mod.escape(" · ".join(tags))}</span></div>'
        )

        if info.strokes:
            # Define every (usually long) path once. Each stage recursively
            # references the completed stage before it, and each visible cell
            # adds only its newest stroke. The old cumulative-path markup
            # repeated n(n+1)/2 path strings for an n-stroke character.
            definitions = []
            cells = []
            for index, path in enumerate(info.strokes):
                stroke_id = f"kanji-{block_index}-stroke-{index}"
                stage_id = f"kanji-{block_index}-stage-{index}"
                definitions.append(
                    f'<path id="{stroke_id}" d="{html_mod.escape(path)}"/>'
                )
                prior = (
                    f'<use href="#kanji-{block_index}-stage-{index - 1}"/>'
                    if index
                    else ""
                )
                definitions.append(
                    f'<g id="{stage_id}">{prior}<use href="#{stroke_id}"/></g>'
                )
                drawn = prior + f'<use class="new" href="#{stroke_id}"/>'
                cells.append(
                    f'<span class="stroke-cell"><svg viewBox="0 0 {CANVAS} {CANVAS}" '
                    f'xmlns="http://www.w3.org/2000/svg">{drawn}</svg></span>'
                )
            parts.append(
                '<svg class="stroke-defs" aria-hidden="true" width="0" height="0" '
                f'xmlns="http://www.w3.org/2000/svg"><defs>{"".join(definitions)}'
                f'</defs></svg><div class="stroke-order">{"".join(cells)}</div>'
            )

        # Round robin: one example from every reading before any reading gets a
        # second. Filling reading by reading spent the whole row budget on the
        # first two — 使's card showed つか(い) twice and left out つか(う),
        # which is the reading of 使う, the word the card is about.
        #
        # A reading with no example at all is placed only after every example
        # has been placed, whatever order the caller's readings arrived in. The
        # renderer cannot assume the sort `fetch_kanji` applies: `data/kanji.json`
        # is committed, hand-editable, and older copies are in KANJIDIC's kana
        # order. In that order 来's example-less き(たす)/き(たる)/きた(す)/きた(る)
        # sat ahead of く(る), so the four-row budget went to 出来, 来年, 上出来
        # and one blank — and 来る, the reading of the word the card is about,
        # never rendered. That is the same loss the round robin exists to stop.
        with_examples = [r for r in info.readings if r.examples]
        without_examples = [r for r in info.readings if not r.examples]
        rows: list[tuple[Reading, Example | None]] = []
        depth_limit = max((len(r.examples) for r in with_examples), default=0)
        for depth in range(depth_limit):
            for reading in with_examples:
                if depth >= len(reading.examples):
                    continue
                example = reading.examples[depth]
                rows.append((reading, example))

        # The fetcher orders readings and examples using JMdict's word-priority
        # evidence. Preserve that evidence-derived order for the ordinary
        # stream, but first reserve the exact pair this particular vocabulary
        # card teaches. Both strings are required: spelling alone would select
        # the wrong member of a homograph pair.
        focus_index = next(
            (
                index
                for index, (_reading, example) in enumerate(rows)
                if example is not None
                and example.written == record_expression
                and example.pronounced == record_reading
            ),
            None,
        )
        if focus_index is not None:
            rows.insert(0, rows.pop(focus_index))

        for reading in without_examples:
            rows.append((reading, None))

        rendered_rows = []
        for reading, example in rows[:MAX_EXAMPLE_ROWS]:
            label = "音" if reading.kind == "on" else "訓"
            written = html_mod.escape(example.written) if example else ""
            pronounced = html_mod.escape(example.pronounced) if example else ""
            gloss = html_mod.escape(example.gloss) if example else ""
            rendered_rows.append(
                '<div class="kanji-example">'
                f'<span class="kanji-kind">{label}</span>'
                f'<span class="kanji-reading">{html_mod.escape(reading.reading)}</span>'
                f'<span class="kanji-word">{written}</span>'
                f'<span class="kanji-kana">{pronounced}</span>'
                f'<span class="kanji-gloss">{gloss}</span>'
                "</div>"
            )
        if rendered_rows:
            parts.append(
                f'<div class="kanji-examples">{"".join(rendered_rows)}</div>'
            )

        blocks.append(
            f'<details class="kanji"><summary>{html_mod.escape(info.character)}</summary>'
            f'{"".join(parts)}</details>'
        )
    return "".join(blocks)


@dataclass(slots=True)
class KanjiStore:
    """Every character janki has looked up, keyed by the character."""

    entries: dict[str, KanjiInfo] = field(default_factory=dict)

    def missing(self, characters: Iterable[str]) -> list[str]:
        return [c for c in dict.fromkeys(characters) if c not in self.entries]

    def for_text(self, text: str) -> list[KanjiInfo]:
        return [self.entries[c] for c in kanji_in(text) if c in self.entries]


def load_store(path: Any) -> KanjiStore:
    """Read the kanji file. A missing one is an empty store, not an error."""
    from pathlib import Path

    file = Path(path)
    if not file.exists():
        return KanjiStore()
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KanjiError(f"Could not read {file}: {exc}") from exc
    if not isinstance(raw, dict):
        raise KanjiError(f"{file} must hold a JSON object keyed by character")
    entries = {}
    for character, value in raw.items():
        if not isinstance(value, dict):
            raise KanjiError(f"{file}: entry for {character!r} must be an object")
        entries[str(character)] = KanjiInfo.from_dict({"character": character, **value})
    return KanjiStore(entries=entries)


def save_store(path: Any, store: KanjiStore) -> None:
    """Write the kanji file, sorted, so a re-fetch produces no diff by itself."""
    from pathlib import Path

    from japanese_anki.io import atomic_write_text

    payload = {}
    for character in sorted(store.entries):
        entry = store.entries[character].to_dict()
        entry.pop("character", None)
        payload[character] = entry
    atomic_write_text(
        Path(path), json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    )
