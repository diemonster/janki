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

**Attribution.** KANJIDIC2 is CC BY-SA 4.0 (EDRDG) and KanjiVG is CC BY-SA 3.0
(Ulrich Apel). A personal deck is fine; a deck that is *shared* has to credit
both, the same way VOICEVOX's per-character terms apply only on sharing.
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

    Only the position marker. ``まえ`` and ``-まえ`` are one reading written
    twice, so a row each prints the same example twice — but ``つか.う`` and
    ``つか.い`` are genuinely different, and collapsing them labelled 使う as
    つか.い, which is simply the wrong reading for that word.
    """
    return _to_hiragana(reading.replace("-", "").strip())


def _display_reading(reading: str) -> str:
    """A reading as a person writes it: ``つか.う`` becomes ``つか(う)``.

    KANJIDIC's dot is machine notation for where the kanji stops and the
    okurigana begins. The information is worth keeping — it is why 使う is
    written with one kana after the character — but a bare dot on a card reads
    as a typo, which is exactly how it was reported.
    """
    text = reading.replace("-", "").strip()
    if "." not in text:
        return text
    stem, _, okurigana = text.partition(".")
    return f"{stem}({okurigana})" if okurigana else stem


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


def _examples_for(reading: str, words: list[dict[str, Any]]) -> tuple[Example, ...]:
    stem = _match_key(reading)
    if not stem:
        return ()
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
            if stem not in pronounced:
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
    kept: list[Example] = []
    for _, example in scored:
        if example.written in seen:
            continue
        seen.add(example.written)
        kept.append(example)
        if len(kept) >= EXAMPLES_PER_READING:
            break
    return tuple(kept)


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

    # Deduplicated by stem: KANJIDIC lists まえ and -まえ separately — the same
    # reading, marked for a suffix position — and both match the same words, so
    # a row each prints the same examples twice. The plain form is kept.
    readings: list[Reading] = []
    seen_stems: set[tuple[str, str]] = set()
    for kind, key in (("on", "on_readings"), ("kun", "kun_readings")):
        for value in sorted((info.get(key) or []), key=lambda r: ("-" in r, r)):
            text = str(value)
            marker = (kind, _dedupe_key(text))
            if not marker[1] or marker in seen_stems:
                continue
            seen_stems.add(marker)
            readings.append(
                Reading(
                    kind=kind,
                    reading=_display_reading(text),
                    examples=_examples_for(text, words),
                )
            )

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


def _stroke_path(path: str, *, newest: bool) -> str:
    """One stroke. The newest is marked so a cell shows what *this* step adds.

    Written as a helper rather than inline: an f-string cannot carry a
    backslash-escaped quote before Python 3.12, and this project targets 3.11.
    """
    marker = ' class="new"' if newest else ""
    return f'<path d="{path}"{marker}/>' 


def render_kanji_html(entries: Iterable[KanjiInfo]) -> str:
    """The collapsible block a card shows, or ``""`` when there is nothing.

    One ``<details>`` per character so a two-kanji word does not force the
    reader to open both, and so the summary can carry the character itself.
    """
    import html as html_mod

    blocks: list[str] = []
    for info in entries:
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
            cells = []
            for index in range(len(info.strokes)):
                drawn = "".join(
                    _stroke_path(html_mod.escape(path), newest=step == index)
                    for step, path in enumerate(info.strokes[: index + 1])
                )
                cells.append(
                    f'<span class="stroke-cell"><svg viewBox="0 0 {CANVAS} {CANVAS}" '
                    f'xmlns="http://www.w3.org/2000/svg">{drawn}</svg></span>'
                )
            parts.append(f'<div class="stroke-order">{"".join(cells)}</div>')

        # Round robin: one example from every reading before any reading gets a
        # second. Filling reading by reading spent the whole row budget on the
        # first two — 使's card showed つか(い) twice and left out つか(う),
        # which is the reading of 使う, the word the card is about.
        rows = []
        with_examples = [r for r in info.readings if r.examples]
        for depth in range(EXAMPLES_PER_READING):
            for reading in with_examples:
                if depth >= len(reading.examples):
                    continue
                example = reading.examples[depth]
                label = "音" if reading.kind == "on" else "訓"
                rows.append(
                    '<div class="kanji-example">'
                    f'<span class="kanji-kind">{label}</span>'
                    f'<span class="kanji-reading">{html_mod.escape(reading.reading)}</span>'
                    f'<span class="kanji-word">{html_mod.escape(example.written)}</span>'
                    f'<span class="kanji-kana">{html_mod.escape(example.pronounced)}</span>'
                    f'<span class="kanji-gloss">{html_mod.escape(example.gloss)}</span>'
                    "</div>"
                )
        if rows:
            kept = "".join(rows[:MAX_EXAMPLE_ROWS])
            parts.append(f'<div class="kanji-examples">{kept}</div>')

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
