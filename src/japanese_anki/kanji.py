"""What a card can say about the *characters* in a word.

Stroke order and the readings KANJIDIC lists — the reference half of a paper
kanji card. janki's records are words (``word:使う:つかう``), so this
is deliberately *not* record content: 前 is the same 前 in 名前 and 前線, and
storing its readings on every word that contains it would be the same lookup
repeated and the same edit needed in several places. It lives in its own file,
keyed by character, and the exporter reads it when it builds a card.

**Two sources, both looked up rather than generated.**

* `kanjiapi.dev` serves KANJIDIC2: stroke count, grade, JLPT level, meanings,
  and on/kun readings.
* KanjiVG supplies the stroke *paths*, in order, which is what makes a
  stroke-by-stroke diagram possible rather than a single glyph.

**Example words are not this module's to choose.** They used to be: JMdict
priority tags ranked kanjiapi's word list, and a positional rule decided which
character in a word a run of kana belonged to. Both are janki deciding how
Japanese is read, and both are gone. A word reaches a card only where a
provider bound it to a reading itself — :mod:`japanese_anki.jpdb_kanji` carries
jpdb's own reading pages, whose links state which reading each word shows. What
KANJIDIC supplies here is an *inventory*: every reading it lists, in its order,
with no ranking and no examples attached.

**Attribution.** KANJIDIC2 is CC BY-SA 4.0 (EDRDG) and KanjiVG is CC BY-SA 3.0
(Ulrich Apel). A personal deck is fine; a deck that is *shared* has to credit
both, the same way VOICEVOX's per-character terms apply only on sharing.
"""

from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from japanese_anki.errors import JankiError
from japanese_anki.identifiers import contains_kanji
from japanese_anki.io import read_text_bound

if TYPE_CHECKING:  # the renderer reads these; it never builds one
    from japanese_anki.jpdb_kanji import BoundExample, CharacterReadings, ReadingUsage

__all__ = [
    "KanjiError",
    "KanjiInfo",
    "Reading",
    "assigns_a_known_reading",
    "fetch_kanji",
    "kanji_in",
    "render_furigana",
    "render_kanji_html",
    "render_stroke_strip",
    "urllib_transport",
]

KANJIAPI = "https://kanjiapi.dev/v1"
KANJIVG = "https://raw.githubusercontent.com/KanjiVG/kanjivg/master/kanji"

#: Politeness, and a practical need: the default urllib agent gets a 403 from
#: kanjiapi.dev.
USER_AGENT = "janki (personal Japanese deck builder)"

DEFAULT_TIMEOUT = 30.0

#: ``(url) -> bytes``. The seam tests replace, so no test reaches the network.
Transport = Callable[[str], bytes]


class KanjiError(JankiError):
    """A kanji source refused, or answered with something unusable."""


@dataclass(frozen=True, slots=True)
class Reading:
    """One reading KANJIDIC lists for a character.

    Inventory, not evidence: nothing here says how often the character is read
    this way or which words show it. Both of those are a provider's to state,
    and jpdb's reading pages are where janki gets them.
    """

    kind: str  # "on" or "kun"
    reading: str


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
                {"kind": reading.kind, "reading": reading.reading}
                for reading in self.readings
            ],
            "strokes": list(self.strokes),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> KanjiInfo:
        character = str(raw.get("character") or "")
        if not character:
            raise KanjiError("A kanji entry needs its character")
        readings = [
            Reading(
                kind=str(item.get("kind") or ""),
                reading=str(item.get("reading") or ""),
            )
            for item in raw.get("readings") or []
        ]
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


def fetch_kanji(character: str, *, transport: Transport | None = None) -> KanjiInfo:
    """Look one character up, from both sources.

    A missing stroke diagram is not a failure: KanjiVG does not cover every
    character, and the readings and meanings are still most of the card.

    The readings come back as KANJIDIC lists them — on'yomi then kun'yomi, each
    in the source's own order, nothing dropped and nothing reordered. まえ and
    -まえ are two entries there and stay two entries here: whether they are "the
    same reading" is a question about Japanese, and the answer this module used
    to give (equal once the boundary markers come off) was a rule of its own
    making. An inventory that is exactly what the dictionary published needs no
    such rule.
    """
    if not character or not contains_kanji(character):
        raise KanjiError(f"Not a kanji: {character!r}")
    send = transport or urllib_transport

    quoted = urllib.parse.quote(character)
    try:
        info = json.loads(send(f"{KANJIAPI}/kanji/{quoted}"))
    except (ValueError, TypeError) as exc:
        raise KanjiError(f"kanjiapi gave no readable answer for {character}") from exc

    readings = [
        Reading(kind=kind, reading=_display_reading(str(value)))
        for kind, key in (("on", "on_readings"), ("kun", "kun_readings"))
        for value in (info.get(key) or [])
        if _display_reading(str(value))
    ]

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

#: Explicit Anki furigana notation — ``word[reading]`` — as a provider supplied
#: it. This reads *markup*, not Japanese: the displayed run and its reading are
#: both already named, so rendering it decides nothing about how a word is read.
#: `exporters.pattern_cards` holds the same expression for deck-authored drill
#: examples; the two cannot share one, because this module is imported by the
#: exporter that module imports.
_FURIGANA = re.compile(r" ?([^>\s\[\]]+?)\[([^\[\]\r\n]+?)\]")


def render_stroke_strip(info: KanjiInfo, *, prefix: str = "kanji") -> str:
    """The graph-paper strip of one character, or ``""`` when it has no paths.

    Each cell shows every stroke drawn so far with the newest picked out, and it
    does so by *reference*: every (usually long) path is defined once, each
    stage points at the completed stage before it, and a cell adds only its own
    newest stroke. The markup this replaced repeated n(n+1)/2 path strings for
    an n-stroke character.

    Those references are ids, and ids are document-wide. ``prefix`` is how a
    caller keeps two strips on one card apart; composing a unique one is the
    caller's job, since only the caller knows what else is on the page.
    """
    if not info.strokes:
        return ""
    safe_prefix = html.escape(prefix, quote=True)
    definitions = []
    cells = []
    for index, path in enumerate(info.strokes):
        stroke_id = f"{safe_prefix}-stroke-{index}"
        stage_id = f"{safe_prefix}-stage-{index}"
        definitions.append(f'<path id="{stroke_id}" d="{html.escape(path)}"/>')
        prior = f'<use href="#{safe_prefix}-stage-{index - 1}"/>' if index else ""
        definitions.append(f'<g id="{stage_id}">{prior}<use href="#{stroke_id}"/></g>')
        drawn = prior + f'<use class="new" href="#{stroke_id}"/>'
        cells.append(
            f'<span class="stroke-cell"><svg viewBox="0 0 {CANVAS} {CANVAS}" '
            f'xmlns="http://www.w3.org/2000/svg">{drawn}</svg></span>'
        )
    return (
        '<svg class="stroke-defs" aria-hidden="true" width="0" height="0" '
        f'xmlns="http://www.w3.org/2000/svg"><defs>{"".join(definitions)}'
        f'</defs></svg><div class="stroke-order">{"".join(cells)}</div>'
    )


def render_furigana(value: str) -> str:
    """Render explicit ``word[reading]`` notation as safe ruby HTML.

    Anki does not apply its ``furigana:`` filter to notation stored inside
    another field's HTML, so a card showing a provider's ruby has to render it
    here or the brackets reach the learner as text.

    Only the brackets are structure, and the single space Anki writes in front
    of an annotated run is its boundary marker — that one is consumed with the
    run, and a second is text. Every base, reading and unannotated run is
    escaped and kept exactly as the notation wrote it, repeated annotations
    included: nothing here merges, reorders, or infers a segment.

    An empty value renders as ``""``. What a card shows instead is the caller's
    to decide, because only the caller knows the word.
    """
    parts: list[str] = []
    cursor = 0
    for match in _FURIGANA.finditer(value):
        parts.append(html.escape(value[cursor:match.start()]))
        parts.append(
            "<ruby><rb>"
            f"{html.escape(match.group(1))}</rb><rt>"
            f"{html.escape(match.group(2))}</rt></ruby>"
        )
        cursor = match.end()
    parts.append(html.escape(value[cursor:]))
    return "".join(parts)


def render_kanji_html(
    entries: Iterable[KanjiInfo],
    *,
    reading_evidence: Mapping[str, CharacterReadings] | None = None,
) -> str:
    """The collapsible block a card shows, or ``""`` when there is nothing.

    One ``<details>`` per character so a two-kanji word does not force the
    reader to open both, and so the summary can carry the character itself.

    Opening one shows what a provider *reported*: every reading jpdb printed a
    percentage beside, in jpdb's order, with its label and that percentage
    exactly as printed and the words jpdb's own page for that reading listed
    under it. A quantified reading with no listed word still shows its
    percentage — "84% of uses, and no example on file" is the fact, and hiding
    the figure until an example exists would report a different one.

    Everything else is one step further in. jpdb's unquantified readings and
    KANJIDIC's inventory each get their own disclosure, labelled for what they
    are: a jpdb reading group is not an on/kun reading, so neither list is
    merged into the other and no on/kun badge is ever printed beside a jpdb
    label. Without jpdb facts for a character the inventory is all there is, and
    it stays reference — nothing here promotes a KANJIDIC reading to evidence.
    """
    evidence = reading_evidence or {}
    blocks: list[str] = []
    for block_index, info in enumerate(entries):
        parts: list[str] = []
        header = html.escape("、".join(info.meanings[:3]))
        tags = []
        if info.jlpt:
            tags.append(f"N{info.jlpt}")
        if info.stroke_count:
            tags.append(f"{info.stroke_count}画")
        if info.grade:
            tags.append(f"grade {info.grade}")
        parts.append(
            f'<div class="kanji-head"><span class="kanji-gloss">{header}</span>'
            f'<span class="kanji-tags">{html.escape(" · ".join(tags))}</span></div>'
        )
        parts.append(render_stroke_strip(info, prefix=f"kanji-{block_index}"))

        reported = evidence.get(info.character)
        quantified, unquantified = _split_reported(reported)
        parts.append(_reported_usage_html(quantified))
        parts.append(
            _further_disclosure(
                "Other JPDB readings", "kanji-other", _labels_html(unquantified)
            )
        )
        parts.append(
            _further_disclosure(
                "KANJIDIC readings", "kanji-inventory", _inventory_html(info.readings)
            )
        )

        blocks.append(
            f'<details class="kanji"><summary>{html.escape(info.character)}</summary>'
            f'{"".join(parts)}</details>'
        )
    return "".join(blocks)


def _split_reported(
    reported: CharacterReadings | None,
) -> tuple[list[ReadingUsage], list[ReadingUsage]]:
    """jpdb's readings split by whether jpdb printed a quantity, in its order.

    The split is on the presence of a figure, which is jpdb's own doing. It is
    not a judgement about which readings matter.
    """
    quantified: list[ReadingUsage] = []
    unquantified: list[ReadingUsage] = []
    for group in reported.groups if reported else ():
        for usage in group.readings:
            (quantified if usage.percent_text is not None else unquantified).append(usage)
    return quantified, unquantified


def _reported_usage_html(readings: list[ReadingUsage]) -> str:
    """Every quantified reading, its printed figure, and its bound words.

    A bound word is shown with the provider's own ruby over it, which is the
    point of having followed the reading page at all: the whole-word reading and
    the English stay beside it as separate statements about the whole word.
    """
    if not readings:
        return ""
    rows = []
    for usage in readings:
        examples = "".join(
            '<div class="kanji-bound-example">'
            f'<span class="kanji-word">{_bound_word_html(example)}</span>'
            f'<span class="kanji-kana">{html.escape(example.pronounced)}</span>'
            f'<span class="kanji-gloss">{html.escape(example.gloss)}</span>'
            "</div>"
            for example in usage.examples
        )
        rows.append(
            '<div class="kanji-usage">'
            f'<span class="kanji-usage-reading">{html.escape(usage.label)}</span>'
            f'<span class="kanji-usage-percent">{html.escape(usage.percent_text or "")}'
            "</span>"
            f"{examples}</div>"
        )
    return (
        '<div class="kanji-evidence">'
        '<div class="kanji-evidence-label">JPDB reported usage</div>'
        f'{"".join(rows)}</div>'
    )


def _bound_word_html(example: BoundExample) -> str:
    """One bound word: the supplied ruby, or its plain spelling when there is none.

    The fallback is the spelling and never the whole-word reading laid over it.
    Which kana sit over which characters is what the provider's reading page
    states, and inventing it here would be janki reading the Japanese.
    """
    if not example.furigana:
        return html.escape(example.written)
    return render_furigana(example.furigana)


def _labels_html(readings: list[ReadingUsage]) -> str:
    """jpdb's unquantified readings: the labels, and nothing implied by them."""
    if not readings:
        return ""
    return "".join(
        f'<span class="kanji-usage-reading">{html.escape(usage.label)}</span>'
        for usage in readings
    )


def _inventory_html(readings: tuple[Reading, ...]) -> str:
    """KANJIDIC's list, in its order, each entry marked on or kun."""
    return "".join(
        '<div class="kanji-inventory-row">'
        f'<span class="kanji-kind">{"音" if reading.kind == "on" else "訓"}</span>'
        f'<span class="kanji-reading">{html.escape(reading.reading)}</span>'
        "</div>"
        for reading in readings
    )


def _further_disclosure(label: str, css_class: str, body: str) -> str:
    """One more ``<details>`` inside the character block, or nothing."""
    if not body:
        return ""
    return (
        f'<details class="kanji-more {css_class}">'
        f"<summary>{html.escape(label)}</summary>{body}</details>"
    )


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
    try:
        raw = json.loads(read_text_bound(file))
    except FileNotFoundError:
        return KanjiStore()
    except (JankiError, OSError, ValueError) as exc:
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
