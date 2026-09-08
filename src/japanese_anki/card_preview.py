"""Render real Anki cards into one self-contained interactive HTML preview.

The owner's card review is an interactive preview of the *actual* cards
(``docs/DESIGN.md``, ``docs/CARD_DESIGN.md``): the real notetype CSS, the real
templates, the real fields, navigation, Show Answer, and the answer's own
disclosures working independently.

Nothing here re-implements a card. The four existing exporters build a real
``.apkg`` into a scratch directory, a scratch Anki collection imports it, and
``card.render_output()`` — the same code path the desktop reviewer uses — draws
every card. That is the only thing that resolves ``{{furigana:}}``,
``{{#Field}}`` and ``{{FrontSide}}`` correctly, and writing a substitute would
be the "rules engine for Japanese from scratch" AGENTS.md names as this
project's defining anti-pattern.

A preview is read-only. It writes only inside its own ``TemporaryDirectory``,
never fetches, never generates, and never touches canonical files — proposed
content is supplied as exact in-memory text overlaying a mirrored snapshot of
just the inputs that deck actually reads.
"""

from __future__ import annotations

import base64
import hashlib
import html
import importlib.util
import io
import json
import mimetypes
import os
import re
import tempfile
import zipfile
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError

__all__ = [
    "CardPreview",
    "CardPreviewError",
    "PreviewCard",
    "ProposedText",
    "preview_unavailable",
    "render_card_preview",
    "write_card_preview",
]


class CardPreviewError(JankiError):
    """A preview could not be rendered, or was asked for something unsafe."""


@dataclass(frozen=True, slots=True)
class ProposedText:
    """One file the preview must read as proposed rather than as it is on disk.

    ``path`` is the exact absolute, project-contained path the proposal would
    eventually be written to; ``text`` is its complete proposed content. The
    preview never writes it — the overlay lives in a scratch mirror — so a
    proposed card renders without applying anything first.
    """

    path: Path
    text: str


@dataclass(frozen=True, slots=True)
class PreviewCard:
    """One card exactly as Anki drew it."""

    #: The note identity this card belongs to. A vocabulary or character note's
    #: own ``RecordID`` field; for a drill card the source record id its GUID
    #: was minted from; for a rule card its deterministic structural identity.
    record_id: str
    #: A short structural label from the notetype's sort field. Never a
    #: judgement about the Japanese.
    label: str
    #: The Anki template name, as the notetype spells it ("Recognition").
    template: str
    #: The deck's own direction key for that template ("recognition").
    direction: str
    question_html: str
    answer_html: str
    is_new: bool


@dataclass(frozen=True, slots=True)
class CardPreview:
    """Every rendered card, the counts around it, and the exact HTML bytes."""

    deck_name: str
    deck_kind: str
    directions: tuple[str, ...]
    #: Notes in the previewed selection.
    note_count: int
    #: Cards in the previewed selection — one per note per enabled direction.
    card_count: int
    #: How many selected notes are genuinely new identities.
    new_note_count: int
    #: Notes the whole prospective deck holds, selection or not.
    deck_note_count: int
    #: Cards the whole prospective deck holds.
    deck_card_count: int
    cards: tuple[PreviewCard, ...]
    #: The exact self-contained document. Paired with the policy below: serving
    #: these bytes under a different CSP breaks the viewer's script hash.
    html: bytes
    sha256: str
    content_security_policy: str

    @property
    def existing_note_count(self) -> int:
        """Selected notes that already exist — the complement of the new ones."""
        return self.note_count - self.new_note_count


def preview_unavailable() -> str | None:
    """Why a preview cannot be rendered here, or ``None`` when it can.

    Deliberately spec-only, and deliberately naming top-level packages:
    importing ``anki`` costs seconds and pulls in a Rust extension, and
    ``find_spec("anki.collection")`` would import the parent package to find
    the submodule — which is the very thing an ordinary CLI run and a base
    install must not do. The renderer imports it for real, lazily, at the
    moment it needs a collection.
    """
    missing = [
        name
        for name in ("genanki", "anki")
        if importlib.util.find_spec(name) is None
    ]
    if not missing:
        return None
    return (
        "Rendering real cards needs the optional preview dependencies "
        f"({', '.join(missing)}). Install them with: "
        "python -m pip install -e '.[preview]'"
    )


# --- bounds -------------------------------------------------------------------
#
# Explicit, and a refusal rather than a trim. A preview that quietly stopped at
# card 500 would be a review surface that lies about what a deck holds, which is
# worse than no preview at all.

#: Cards one document may hold before the answer is "ask for a narrower scope".
MAX_PREVIEW_CARDS = 2000
#: One inlined media file. Card audio is a few seconds of speech.
MAX_MEDIA_FILE_BYTES = 8 * 1024 * 1024
#: Every inlined media file together.
MAX_MEDIA_TOTAL_BYTES = 32 * 1024 * 1024
#: The finished document.
MAX_PREVIEW_BYTES = 48 * 1024 * 1024

#: The placeholder Anki leaves where a ``[sound:]`` tag was, naming the side and
#: the tag's index in that side's ``av_tags`` list. Replacing it in place is
#: what puts the control back at the card position the template chose.
_PLAY_TAG = re.compile(r"\[anki:play:(q|a):(\d+)\]")
_IMG_TAG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_TAG_SRC = re.compile(
    r"""\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""", re.IGNORECASE
)
_MARKUP = re.compile(r"<[^>]*>")
#: A structural test for "this would be fetched from somewhere else", not a
#: judgement about the URL. ``data:`` is already inline and stays.
_SCHEME_RELATIVE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.\-]*:)?//")

#: Config fields naming a store a proposal may replace wholesale. Each is an
#: absolute path in its own right, so a mirrored copy is repointed rather than
#: rebased.
_OVERLAYABLE_CONFIG_FILES = (
    "normalized_file",
    "kanji_notes_file",
    "kanji_file",
    "jpdb_readings_file",
    "patterns_file",
)


@dataclass(frozen=True, slots=True)
class _DeckPlan:
    """What one deck kind's builder produced, and how to read its notes back."""

    kind: str
    deck_name: str
    directions: tuple[str, ...]
    #: Note GUID -> record identity, for the two notetypes that carry none in a
    #: field. Empty for vocabulary and character decks, whose first field *is*
    #: the identity.
    identity_by_guid: dict[str, str]


@dataclass(frozen=True, slots=True)
class _CardRef:
    """One card the package holds, known before anything is drawn.

    The catalogue of these is the whole deck: what a scope selects from, what
    the deck totals count, and what a display cap is applied to *after* the
    selection narrows it.
    """

    card_id: int
    note_id: int
    ordinal: int
    record_id: str
    label: str
    template: str
    direction: str


@dataclass(frozen=True, slots=True)
class _Drawn:
    """One card as Anki drew it, before the viewer wraps it."""

    note_id: int
    ordinal: int
    record_id: str
    label: str
    template: str
    direction: str
    question_html: str
    answer_html: str


# --- resolving what to build --------------------------------------------------


def _deck_section(deck_path: Path) -> dict[str, Any]:
    """The ``deck:`` mapping, read the same way every other reader reads it."""
    from japanese_anki.io import load_structured

    raw = load_structured(deck_path)
    if not isinstance(raw, dict):
        raise CardPreviewError(f"Deck file must contain a mapping: {deck_path}")
    section = raw.get("deck") or {}
    if not isinstance(section, dict):
        raise CardPreviewError(f"The deck section must be a mapping: {deck_path}")
    return section


def _validated_overlay(
    config: ProjectConfig, proposed: Sequence[ProposedText]
) -> dict[Path, str]:
    """Exact proposed texts, keyed by the absolute path each one replaces.

    Paths are normalised lexically rather than through ``resolve()``: a
    proposal names the file it *would* be written to, which need not exist yet,
    and following a symlink here would let a link planted inside the project
    point the overlay at a file outside it.
    """
    if isinstance(proposed, str | bytes) or not isinstance(proposed, Sequence):
        raise CardPreviewError("Proposed preview content must be a sequence.")
    root = config.root.resolve()
    overlay: dict[Path, str] = {}
    for item in proposed:
        if not isinstance(item, ProposedText):
            raise CardPreviewError(
                f"Proposed preview content must be ProposedText values, got "
                f"{type(item).__name__}."
            )
        if not isinstance(item.text, str):
            raise CardPreviewError(
                f"Proposed content for {item.path} must be text, got "
                f"{type(item.text).__name__}."
            )
        path = Path(item.path)
        if not path.is_absolute():
            raise CardPreviewError(
                f"A proposed preview path must be absolute: {item.path}"
            )
        lexical = Path(os.path.abspath(os.fspath(path)))
        if not lexical.is_relative_to(root):
            raise CardPreviewError(
                f"A proposed preview path must be inside the project at {root}: "
                f"{item.path}"
            )
        if lexical.is_symlink():
            raise CardPreviewError(
                f"A proposed preview path must not be a symlink: {item.path}"
            )
        if lexical in overlay:
            raise CardPreviewError(
                f"Proposed preview path {lexical} is supplied more than once."
            )
        overlay[lexical] = item.text
    return overlay


def _mirrored_project(
    config: ProjectConfig,
    deck_path: Path,
    overlay: dict[Path, str],
    mirror_root: Path,
) -> tuple[ProjectConfig, Path]:
    """Snapshot only the inputs this deck reads, with the proposals applied.

    The mirror keeps the project's own directory layout, so a deck's
    ``source: ../normalized/vocabulary.json`` still names the same file after
    the move. That is the whole rebasing rule: nothing rewrites a path, and no
    relative reference changes meaning.

    Nothing else is copied. ``data/inbox``, the operation journal, the ledger,
    staging, generated media and ``.git`` are not inputs to a card, and a
    preview has no business duplicating them. Media is linked rather than
    copied, because a deck may name a clip relative to its own directory and
    the bytes are only ever read.
    """
    root = config.root.resolve()
    if not deck_path.is_relative_to(root):
        raise CardPreviewError(
            f"A previewed deck must live inside the project at {root}: {deck_path}"
        )
    used: set[Path] = set()

    def place(target: Path) -> Path:
        destination = mirror_root / target.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        used.add(target)
        if target in overlay:
            destination.write_text(overlay[target], encoding="utf-8")
            return destination
        try:
            destination.write_bytes(target.read_bytes())
        except OSError as exc:
            raise CardPreviewError(
                f"Could not read {target} for the preview: {exc}"
            ) from exc
        return destination

    mirror_deck = place(deck_path)
    source_value = _deck_section(mirror_deck).get("source")
    if source_value:
        declared = Path(
            os.path.abspath(os.fspath(deck_path.parent / str(source_value)))
        )
        if not declared.is_relative_to(root):
            raise CardPreviewError(
                f"{deck_path} reads {declared}, which is outside the project at "
                f"{root}, so a proposed preview cannot snapshot it."
            )
        mirrored = place(declared)
        # The rebase, stated as a check rather than as a comment: the deck's own
        # relative reference, read from where the mirror put it, still resolves
        # to the file the real deck names.
        rebased = Path(os.path.abspath(os.fspath(mirror_deck.parent / str(source_value))))
        if rebased != mirrored:
            # An *absolute* `source:` names the canonical file wherever the deck
            # sits, so moving the deck cannot move it. Rebind that one scalar to
            # the snapshot instead — nothing else in the file is touched, and
            # the canonical deck is never written at all.
            _rebind_deck_source(mirror_deck, mirrored, deck_path, source_value)

    changes: dict[str, Path] = {}
    for field in _OVERLAYABLE_CONFIG_FILES:
        target = Path(os.path.abspath(os.fspath(getattr(config, field))))
        if target in overlay:
            changes[field] = place(target)

    media_dir = Path(os.path.abspath(os.fspath(config.media_dir)))
    if media_dir.is_relative_to(root) and media_dir.is_dir():
        link = mirror_root / media_dir.relative_to(root)
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(media_dir, target_is_directory=True)
    _bind_deck_local_media(media_dir, deck_path.parent, mirror_deck, mirror_root)

    unread = sorted(str(path) for path in overlay if path not in used)
    if unread:
        raise CardPreviewError(
            "Proposed preview content names files this deck does not read, so "
            "the preview would silently ignore them: " + ", ".join(unread)
        )

    if config.deck_dir.is_relative_to(root):
        changes["deck_dir"] = mirror_root / config.deck_dir.relative_to(root)
    return (
        replace(config, root=mirror_root, dist_dir=mirror_root / "dist", **changes),
        mirror_deck,
    )


def _rebind_deck_source(
    mirror_deck: Path, snapshot: Path, deck_path: Path, source_value: Any
) -> None:
    """Point the snapshot deck's ``source:`` at the snapshot of that source.

    A round-trip edit of exactly one scalar: comments, quoting, anchors, the
    selectors and every inline note survive, because the alternative — dumping
    a re-parsed document — would rewrite the file the build then reads. The
    canonical deck is untouched; this writes only inside the scratch mirror.
    """
    from ruamel.yaml import YAML, YAMLError
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    parser = YAML()
    parser.preserve_quotes = True
    parser.allow_unicode = True
    parser.width = 1 << 30
    try:
        document = parser.load(mirror_deck.read_text(encoding="utf-8"))
    except YAMLError as exc:
        raise CardPreviewError(
            f"Could not read {deck_path} to rebind its source reference: {exc}"
        ) from exc
    section = (
        document.get("deck") if isinstance(document, MutableMapping) else None
    )
    if not isinstance(section, MutableMapping) or "source" not in section:
        raise CardPreviewError(
            f"Could not preserve {deck_path}'s source reference "
            f"{source_value!r} in a preview snapshot."
        )
    section["source"] = DoubleQuotedScalarString(str(snapshot))
    buffer = io.StringIO()
    parser.dump(document, buffer)
    mirror_deck.write_text(buffer.getvalue(), encoding="utf-8")

    rebound = _deck_section(mirror_deck).get("source")
    if Path(
        os.path.abspath(os.fspath(mirror_deck.parent / str(rebound)))
    ) != snapshot:
        raise CardPreviewError(
            f"Could not preserve {deck_path}'s source reference "
            f"{source_value!r} in a preview snapshot."
        )


def _declared_media_values(deck_path: Path, kind: str) -> list[str]:
    """Every media path this deck's own content names, and nothing more."""
    values: list[str] = []
    if kind == "vocabulary":
        from japanese_anki.exporters.anki import resolve_deck_records

        _deck_config, records = resolve_deck_records(deck_path)
        for record in records:
            values.extend((record.audio, record.image))
            values.extend(example.audio for example in record.examples)
    elif kind == "conjugation":
        raw = _deck_section(deck_path).get("drill_examples")
        if isinstance(raw, dict):
            for entries in raw.values():
                for entry in entries if isinstance(entries, list) else ():
                    if isinstance(entry, dict):
                        values.append(str(entry.get("audio") or ""))
    return [value for value in values if isinstance(value, str) and value]


def _bind_deck_local_media(
    media_dir: Path,
    original_deck_dir: Path,
    mirror_deck: Path,
    mirror_root: Path,
) -> None:
    """Keep the second half of the exporter's media rule after the deck moves.

    ``_resolve_media`` looks in the configured media directory first and then
    beside the deck. The first is linked whole above; the second is what the
    snapshot breaks, because the deck is no longer beside its own files. So
    each clip a deck actually names, that is *not* already in the configured
    directory and *is* beside the canonical deck, gets one link at the same
    relative place. Nothing else in that directory is exposed, no unrelated
    deck or clip is copied, and the precedence is unchanged: a name present in
    both still resolves to the configured one, because that candidate is tried
    first and never removed.

    Containment is the whole snapshot rather than the deck's own directory. A
    nested deck naming ``../clips/foo.wav`` is an ordinary deck-relative path,
    and measuring it against ``decks/nested/`` rejected a link that lands
    perfectly safely at ``decks/clips/`` inside the mirror — so the clip
    vanished from a proposal while the same deck plays it today. A value that
    would still reach outside the snapshot is skipped, and nothing is ever
    written into the project.
    """
    from japanese_anki.exporters.anki import deck_kind

    try:
        values = _declared_media_values(mirror_deck, deck_kind(mirror_deck) or "vocabulary")
    except JankiError:
        # A deck this cannot read is a deck the builder is about to refuse by
        # name. Guessing at its media here would replace that error with a
        # worse one.
        return
    mirror_deck_dir = mirror_deck.parent
    for value in dict.fromkeys(values):
        if value.startswith("[sound:"):
            continue
        if (media_dir / value).exists():
            continue
        origin = Path(os.path.abspath(os.fspath(original_deck_dir / value)))
        if not origin.is_file():
            continue
        link = Path(os.path.abspath(os.fspath(mirror_deck_dir / value)))
        if not link.is_relative_to(mirror_root) or link.exists():
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(origin)


def _build_package(
    config: ProjectConfig, deck_path: Path, output_path: Path
) -> _DeckPlan:
    """Build the real package for whichever of the four kinds this deck is."""
    from japanese_anki.exporters.anki import (
        deck_kind,
        resolve_card_types,
        resolve_deck_records,
    )

    kind = deck_kind(deck_path) or "vocabulary"
    if kind == "vocabulary":
        from japanese_anki.exporters.anki import build_deck

        deck_config, _records = resolve_deck_records(deck_path)
        directions = tuple(resolve_card_types(deck_config, config))
        build_deck(deck_path, config, output_path)
        return _DeckPlan(
            kind=kind,
            deck_name=str(deck_config.get("name", config.default_deck_name)),
            directions=directions,
            identity_by_guid={},
        )

    if kind == "kanji":
        from japanese_anki.exporters.kanji_cards import (
            build_kanji_deck,
            resolve_kanji_deck_notes,
        )

        resolved = resolve_kanji_deck_notes(deck_path, config)
        build_kanji_deck(deck_path, config, output_path)
        return _DeckPlan(
            kind=kind,
            deck_name=resolved.deck_name,
            directions=resolved.directions,
            identity_by_guid={},
        )

    import genanki

    from japanese_anki.exporters import pattern_cards

    section = _deck_section(deck_path)
    deck_name = str(section.get("name") or deck_path.stem)

    if kind == "pattern":
        from japanese_anki.patterns import load_store

        store = load_store(config.patterns_file)
        document = str(section.get("document") or "").strip()
        entry = store.get(document)
        if entry is None:
            known = ", ".join(sorted(store)) or "none"
            raise CardPreviewError(
                f"{deck_path}: no document has been read under {document!r}. "
                f"Known: {known}"
            )
        identities = {
            genanki.guid_for(
                f"pattern:{document}:{card.identity}"
            ): f"pattern:{document}:{card.identity}"
            for card in pattern_cards.cards_for(entry)
        }
        pattern_cards.build_pattern_deck(deck_path, config, store, output_path)
        return _DeckPlan(
            kind=kind,
            deck_name=deck_name,
            directions=("rule",),
            identity_by_guid=identities,
        )

    if kind == "conjugation":
        from japanese_anki.io import load_records

        form = str(section.get("form") or "te_form").strip()
        records = load_records(
            pattern_cards.collection_for(deck_path, config)
        )
        shipping = pattern_cards.shipping_records(deck_path, records, form)
        # A drill note's GUID is minted from the record and the form, and its
        # fields hold neither: the front is the conjugated word. Mapping the
        # GUID back is what lets an API scope stated in *vocabulary* ids select
        # the drill cards those records produced.
        identities = {
            genanki.guid_for(f"drill:{form}:{record_id}"): record_id
            for _card, record_id in pattern_cards.drill_cards(shipping, form)
        }
        pattern_cards.build_conjugation_deck(
            deck_path, config, records, output_path
        )
        return _DeckPlan(
            kind=kind,
            deck_name=deck_name,
            directions=("rule",),
            identity_by_guid=identities,
        )

    raise CardPreviewError(f"{deck_path}: janki cannot preview a {kind!r} deck.")


# --- media --------------------------------------------------------------------


def _package_media(package: Path, wanted: frozenset[str]) -> dict[str, bytes]:
    """The media the *selected* cards reference, by the name a card uses.

    Read out of the package rather than off disk: the package is what a learner
    would import, so its bytes are the truthful answer to "does this card have
    its audio". The archive is one janki just wrote, and the member checks
    below are structural belt-and-braces on that.

    Only ``wanted`` is opened, sized and capped. A clip on a card nobody asked
    to see is not in this page, so it cannot make this page too big — and
    reading it just to refuse it would be the cap deciding what a scope means.
    """
    if not wanted:
        return {}
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if "media" not in names:
            return {}
        try:
            mapping = json.loads(archive.read("media").decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise CardPreviewError(
                f"Could not read the media index of {package}: {exc}"
            ) from exc
        if not isinstance(mapping, dict):
            raise CardPreviewError(f"The media index of {package} is not a mapping.")
        collected: dict[str, bytes] = {}
        total = 0
        for member, filename in mapping.items():
            member = str(member)
            if str(filename) not in wanted:
                continue
            if member != Path(member).name or member in ("", ".", ".."):
                raise CardPreviewError(
                    f"Refused a package media member with a path in its name: "
                    f"{member!r}"
                )
            name = str(filename)
            if name != Path(name).name or name in ("", ".", ".."):
                raise CardPreviewError(
                    f"Refused a package media name with a path in it: {name!r}"
                )
            info = archive.getinfo(member)
            if info.file_size > MAX_MEDIA_FILE_BYTES:
                raise CardPreviewError(
                    f"{name} is {info.file_size} bytes, over the "
                    f"{MAX_MEDIA_FILE_BYTES}-byte per-file preview limit. Narrow "
                    "the preview rather than shipping a card without its media."
                )
            total += info.file_size
            if total > MAX_MEDIA_TOTAL_BYTES:
                raise CardPreviewError(
                    f"The selected cards' media exceeds the "
                    f"{MAX_MEDIA_TOTAL_BYTES}-byte preview limit. Preview a "
                    "narrower scope."
                )
            collected[name] = archive.read(member)
        return collected


def _data_uri(name: str, payload: bytes) -> str:
    kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return f"data:{kind};base64,{base64.b64encode(payload).decode('ascii')}"


def _missing_media(kind: str, name: str) -> str:
    """Say what is not here. Never draw a control that would play silence."""
    return (
        f'<span class="preview-media-missing">{html.escape(kind)} not packaged: '
        f"{html.escape(name)}</span>"
    )


def _audio_control(name: str, media: dict[str, bytes]) -> str:
    """A working clip is a control; only a missing one has to name itself.

    Generated audio is identity-addressed, so its filename is a hash — and a
    hash printed beside a play button is implementation detail on the face of a
    study card. A clip janki could not package is the opposite case: the name
    is the only way to go and find it.
    """
    payload = media.get(name)
    if payload is None or not payload:
        return _missing_media("Audio", name)
    return (
        '<span class="preview-audio">'
        f'<audio controls preload="none" src="{html.escape(_data_uri(name, payload), quote=True)}">'
        "</audio>"
        "</span>"
    )


def _inline_av_tags(text: str, av_tags: Sequence[Any], media: dict[str, bytes]) -> str:
    """Put a real, non-autoplaying control where the ``[sound:]`` tag was.

    Anki hands the tags back separately from the text and leaves an
    ``[anki:play:…]`` marker at the position the template chose; keeping the
    control there is what makes a word clip sit under the word and a sentence
    clip under its sentence.
    """

    def substitute(match: re.Match[str]) -> str:
        index = int(match.group(2))
        if index >= len(av_tags):
            return _missing_media("Audio", match.group(0))
        tag = av_tags[index]
        filename = getattr(tag, "filename", "")
        if not filename:
            # A text-to-speech tag asks the reviewer to synthesise speech. There
            # is no file, and a preview does not generate one.
            return _missing_media("Spoken audio", str(tag))
        return _audio_control(str(filename), media)

    return _PLAY_TAG.sub(substitute, text)


def _local_image_name(tag: str) -> str:
    """The packaged basename one ``<img>`` needs, or ``""`` for anything else.

    Shared by the collector below and the inliner, so the set of files a page
    opens is exactly the set it draws.
    """
    found = _TAG_SRC.search(tag)
    if not found:
        return ""
    source = (found.group(1) or found.group(2) or found.group(3) or "").strip()
    if not source or source.lower().startswith("data:"):
        return ""
    if _SCHEME_RELATIVE.match(source) or ":" in source.split("/", 1)[0]:
        return ""
    return Path(source).name


def _referenced_media(text: str, av_tags: Sequence[Any]) -> set[str]:
    """Every packaged file one drawn side would inline."""
    names = {
        str(name)
        for tag in av_tags
        if (name := getattr(tag, "filename", ""))
    }
    for match in _IMG_TAG.finditer(text):
        if name := _local_image_name(match.group(0)):
            names.add(name)
    return names


def _inline_images(text: str, media: dict[str, bytes]) -> str:
    """Inline packaged images; say so about any that would need the network."""

    def substitute(match: re.Match[str]) -> str:
        tag = match.group(0)
        found = _TAG_SRC.search(tag)
        if not found:
            return tag
        source = (found.group(1) or found.group(2) or found.group(3) or "").strip()
        if source.lower().startswith("data:"):
            return tag
        if _SCHEME_RELATIVE.match(source) or ":" in source.split("/", 1)[0]:
            return _missing_media("Remote image", source)
        payload = media.get(Path(source).name)
        if payload is None or not payload:
            return _missing_media("Image", source)
        replacement = f'src="{html.escape(_data_uri(source, payload), quote=True)}"'
        return tag[: found.start()] + replacement + tag[found.end():]

    return _IMG_TAG.sub(substitute, text)


# --- rendering ----------------------------------------------------------------


def _plain(value: str, limit: int = 60) -> str:
    """Field markup reduced to a short label.

    Structural only: tags out, entities decoded, whitespace collapsed. It reads
    no Japanese and decides nothing about the content — it is the notetype's own
    sort field, shortened enough to fit a menu.
    """
    text = " ".join(html.unescape(_MARKUP.sub(" ", value)).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _drawn_cards(
    package: Path,
    scratch: Path,
    plan: _DeckPlan,
    choose: Callable[[tuple[_CardRef, ...]], tuple[_CardRef, ...]],
) -> tuple[list[_Drawn], str]:
    """Import the package, enumerate every card, then draw only the chosen ones.

    Two passes on purpose. The catalogue is what the *whole deck* holds, and it
    is what a scope, a new-id list and the deck totals are validated against;
    only after that is anything rendered or any media opened. Selecting two
    cards out of a large deck therefore costs two renders and their own clips,
    and a display cap describes the page rather than the deck behind it.

    Drawing is the same code path the desktop reviewer uses, which is the only
    thing that resolves ``{{furigana:}}``, a ``{{#Field}}`` section,
    ``{{FrontSide}}`` and a ``[sound:]`` tag correctly. Returns the chosen
    cards in the order ``choose`` returned them, and the notetype's own CSS.
    """
    import anki.collection
    from anki.import_export_pb2 import ImportAnkiPackageRequest

    collection = anki.collection.Collection(str(scratch / "preview.anki2"))
    try:
        collection.import_anki_package(
            ImportAnkiPackageRequest(package_path=str(package))
        )
        catalogue: list[_CardRef] = []
        css = ""
        for card_id in collection.find_cards(""):
            card = collection.get_card(card_id)
            note = card.note()
            notetype = card.note_type()
            css = css or str(notetype.get("css") or "")
            if plan.identity_by_guid:
                record_id = plan.identity_by_guid.get(note.guid, "")
                if not record_id:
                    raise CardPreviewError(
                        f"A built note has no identity this preview can name "
                        f"(guid {note.guid!r}). Nothing was previewed."
                    )
            else:
                # The first field of both the vocabulary and the character
                # notetype *is* the record id, which is why they carry one.
                record_id = note.fields[0] if note.fields else ""
                if not record_id:
                    raise CardPreviewError(
                        "A built note carries no RecordID. Nothing was previewed."
                    )
            sort_index = int(notetype.get("sortf") or 0)
            label = _plain(
                note.fields[sort_index] if sort_index < len(note.fields) else ""
            )
            direction = (
                plan.directions[card.ord]
                if card.ord < len(plan.directions)
                else str(card.template().get("name", "")).lower()
            )
            catalogue.append(
                _CardRef(
                    card_id=card_id,
                    note_id=card.nid,
                    ordinal=card.ord,
                    record_id=record_id,
                    label=label,
                    template=str(card.template().get("name", "")),
                    direction=direction,
                )
            )
        # Note ids are the timestamps genanki minted in build order, so this is
        # the order the exporter wrote and the order the deck file implies.
        catalogue.sort(key=lambda item: (item.note_id, item.ordinal))

        chosen = choose(tuple(catalogue))

        rendered: list[tuple[_CardRef, Any]] = []
        wanted: set[str] = set()
        for ref in chosen:
            output = collection.get_card(ref.card_id).render_output()
            rendered.append((ref, output))
            wanted |= _referenced_media(output.question_text, output.question_av_tags)
            wanted |= _referenced_media(output.answer_text, output.answer_av_tags)
    finally:
        collection.close()

    media = _package_media(package, frozenset(wanted))
    drawn = [
        _Drawn(
            note_id=ref.note_id,
            ordinal=ref.ordinal,
            record_id=ref.record_id,
            label=ref.label,
            template=ref.template,
            direction=ref.direction,
            question_html=_inline_images(
                _inline_av_tags(output.question_text, output.question_av_tags, media),
                media,
            ),
            answer_html=_inline_images(
                _inline_av_tags(output.answer_text, output.answer_av_tags, media),
                media,
            ),
        )
        for ref, output in rendered
    ]
    return drawn, css


def _selected_ids(
    grouped: dict[str, list[_Drawn]],
    scope_record_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    """The identities to show, in the order the caller asked for them."""
    if scope_record_ids is None:
        return tuple(grouped)
    if isinstance(scope_record_ids, str | bytes | bytearray) or not isinstance(
        scope_record_ids, Sequence
    ):
        raise CardPreviewError("Preview card IDs must be a sequence of IDs, not text.")
    requested = tuple(scope_record_ids)
    if not requested:
        raise CardPreviewError("Preview needs at least one card ID.")
    seen: set[str] = set()
    for record_id in requested:
        if not isinstance(record_id, str) or not record_id.strip():
            raise CardPreviewError("Preview card IDs must be nonblank strings.")
        if record_id in seen:
            raise CardPreviewError(f"Preview card ID {record_id!r} is repeated.")
        seen.add(record_id)
    missing = tuple(record_id for record_id in requested if record_id not in grouped)
    if missing:
        names = ", ".join(repr(record_id) for record_id in missing)
        raise CardPreviewError(
            f"The requested card IDs are not among the cards this deck builds: "
            f"{names}."
        )
    return requested


def _new_ids(
    grouped: dict[str, list[_Drawn]], new_record_ids: Sequence[str]
) -> frozenset[str]:
    if isinstance(new_record_ids, str | bytes | bytearray) or not isinstance(
        new_record_ids, Sequence
    ):
        raise CardPreviewError("New card IDs must be a sequence of IDs, not text.")
    collected: set[str] = set()
    for record_id in new_record_ids:
        if not isinstance(record_id, str) or not record_id.strip():
            raise CardPreviewError("New card IDs must be nonblank strings.")
        if record_id in collected:
            raise CardPreviewError(f"New card ID {record_id!r} is repeated.")
        if record_id not in grouped:
            raise CardPreviewError(
                f"New card ID {record_id!r} is not among the cards this deck "
                "builds, so the preview cannot count it as an addition."
            )
        collected.add(record_id)
    return frozenset(collected)


# --- the document -------------------------------------------------------------


def _viewer_asset(config: ProjectConfig, name: str) -> str:
    """One outer viewer asset: the project's own copy, else the checkout's.

    The viewer is janki's tool rather than the deck's content, so the shipped
    copy under ``templates/card-preview/`` is the default; a project that keeps
    its own beside its notetype templates wins, which is the same precedence
    every other template here has.
    """
    candidates = (
        Path(config.template_dir).parent / "card-preview" / name,
        Path(__file__).resolve().parents[2] / "templates" / "card-preview" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise CardPreviewError(f"Missing preview viewer asset {name}. Tried: {tried}")


def _policy(script: str) -> str:
    """The exact policy these bytes are served under, script hash included.

    ``script-src`` carries only the viewer's hash: no ``unsafe-inline``, so an
    inline event handler in card content has no authority to run, and no host
    source, so nothing can be fetched. ``default-src 'none'`` with ``data:``
    for the three media kinds means every subresource is already in the file.
    ``style-src`` keeps ``unsafe-inline`` because the notetype stylesheet and
    the viewer's own are inlined and a stylesheet cannot execute — with
    ``img-src`` limited to ``data:`` a CSS ``url()`` cannot fetch either.

    ``form-action``, ``object-src`` and ``base-uri`` are denied outright. An
    ``<a href="https://…">`` dictionary link is navigation the reader chooses,
    not a subresource, and stays clickable.
    """
    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode(
        "ascii"
    )
    return (
        "default-src 'none'; "
        f"script-src 'sha256-{digest}'; "
        "style-src 'unsafe-inline'; "
        "img-src data:; "
        "media-src data:; "
        "font-src data:; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "object-src 'none'; "
        "frame-ancestors 'none'"
    )


def _counts_html(
    kind: str,
    directions: tuple[str, ...],
    note_count: int,
    card_count: int,
    new_note_count: int,
    deck_note_count: int,
    deck_card_count: int,
) -> str:
    """The three counts kept apart, because they answer different questions."""
    items = [
        f"{kind} deck",
        f"showing {note_count} note{'' if note_count == 1 else 's'} · "
        f"{card_count} card{'' if card_count == 1 else 's'}",
        f"whole deck {deck_note_count} note{'' if deck_note_count == 1 else 's'} · "
        f"{deck_card_count} card{'' if deck_card_count == 1 else 's'}",
        "directions: " + ", ".join(directions),
    ]
    rows = [f"<li>{html.escape(item)}</li>" for item in items]
    if new_note_count:
        rows.insert(
            2,
            '<li class="preview-count-new">'
            + html.escape(f"{new_note_count} new")
            + "</li>",
        )
    return '<ul class="preview-counts">' + "".join(rows) + "</ul>"


def _card_html(index: int, card: PreviewCard) -> str:
    """One card, labelled by what it is rather than by how it is addressed.

    The record id and the direction stay in the data attributes — that is
    where scope, selection and any structural reader want them — and off the
    face of the card, where they are noise beside the Japanese. The template
    name is the one label that tells a reader which card they are looking at,
    and the new mark is a fact about the batch rather than about the renderer.
    """
    badges = [f'<span class="preview-badge">{html.escape(card.template)}</span>']
    if card.is_new:
        badges.insert(0, '<span class="preview-badge preview-badge-new">new</span>')
    return (
        f'<section class="preview-card" data-preview-card '
        f'data-index="{index}" '
        f'data-record-id="{html.escape(card.record_id, quote=True)}" '
        f'data-direction="{html.escape(card.direction, quote=True)}" '
        f'data-template="{html.escape(card.template, quote=True)}" '
        f'data-new="{"true" if card.is_new else "false"}"'
        f"{'' if index == 0 else ' hidden'}>"
        f'<div class="preview-card-meta">{"".join(badges)}</div>'
        '<div class="preview-side" data-preview-question>'
        '<div class="preview-side-label">Question</div>'
        f'<div class="card">{card.question_html}</div>'
        "</div>"
        # Hidden in the bytes, not only by script: the preview opens on the
        # question the way the reviewer does, and a viewer that never ran must
        # not have given the answer away.
        '<div class="preview-side" data-preview-answer hidden>'
        '<div class="preview-side-label">Answer</div>'
        f'<div class="card">{card.answer_html}</div>'
        "</div>"
        "</section>"
    )


def _document(
    config: ProjectConfig,
    *,
    deck_name: str,
    kind: str,
    subtitle: str,
    directions: tuple[str, ...],
    cards: tuple[PreviewCard, ...],
    note_count: int,
    new_note_count: int,
    deck_note_count: int,
    deck_card_count: int,
    card_css: str,
) -> tuple[bytes, str]:
    """One self-contained page, and the exact policy it is paired with."""
    viewer_css = _viewer_asset(config, "viewer.css")
    viewer_js = _viewer_asset(config, "viewer.js")
    policy = _policy(viewer_js)
    title = f"{deck_name} — card preview"
    options = "".join(
        f'<option value="{index}">'
        + html.escape(
            f"{index + 1}. {card.label or card.template} — {card.template}"
        )
        + "</option>"
        for index, card in enumerate(cards)
    )
    subtitle_html = (
        f'<p class="preview-subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{html.escape(policy, quote=True)}">
<title>{html.escape(title)}</title>
<style>{card_css}</style>
<style>{viewer_css}</style>
</head>
<body class="preview-body">
<header class="preview-head">
<h1 class="preview-title">{html.escape(deck_name)}</h1>
{subtitle_html}
{_counts_html(kind, directions, note_count, len(cards), new_note_count,
              deck_note_count, deck_card_count)}
</header>
<nav class="preview-nav" aria-label="Card navigation">
<button type="button" id="preview-prev">Previous</button>
<button type="button" id="preview-next">Next</button>
<span class="preview-position" id="preview-position">Card 1 of {len(cards)}</span>
<label class="preview-jump-label" for="preview-jump">
<span class="preview-position">Go to</span>
<select class="preview-jump" id="preview-jump">{options}</select>
</label>
<button type="button" class="preview-flip" id="preview-flip" \
aria-pressed="false">Show Answer</button>
</nav>
<main class="preview-deck">
{"".join(_card_html(index, card) for index, card in enumerate(cards))}
</main>
<p class="preview-note">Rendered by Anki from this deck's own templates and \
stylesheet. Viewing a preview approves nothing.</p>
<script>{viewer_js}</script>
</body>
</html>
"""
    payload = document.encode("utf-8")
    if len(payload) > MAX_PREVIEW_BYTES:
        raise CardPreviewError(
            f"This preview is {len(payload)} bytes, over the "
            f"{MAX_PREVIEW_BYTES}-byte limit. Preview a narrower scope rather "
            "than shipping a page with cards left out."
        )
    return payload, policy


# --- the public renderer ------------------------------------------------------


def render_card_preview(
    config: ProjectConfig,
    deck_path: Path,
    *,
    proposed: Sequence[ProposedText] = (),
    new_record_ids: Sequence[str] = (),
    scope_record_ids: Sequence[str] | None = None,
    subtitle: str = "",
) -> CardPreview:
    """Render one deck's real cards into a self-contained interactive preview.

    Read-only. Everything is built inside a ``TemporaryDirectory`` that is
    removed whether this succeeds or fails; ``proposed`` supplies exact
    in-memory text for files that need not exist yet, so a proposal renders
    without being applied first.

    ``scope_record_ids`` selects exact identities in the caller's order and
    refuses an unknown or repeated one rather than quietly showing a different
    set. ``new_record_ids`` names genuine additions, which is what separates
    ``new_note_count`` from ``existing_note_count``; both stay distinct from
    the whole deck's ``deck_note_count``.
    """
    unavailable = preview_unavailable()
    if unavailable is not None:
        raise CardPreviewError(unavailable)
    if not isinstance(subtitle, str):
        raise CardPreviewError(
            f"A preview subtitle must be text, got {type(subtitle).__name__}."
        )

    target = Path(os.path.abspath(os.fspath(deck_path)))
    overlay = _validated_overlay(config, proposed)
    selection: dict[str, Any] = {}

    def choose(catalogue: tuple[_CardRef, ...]) -> tuple[_CardRef, ...]:
        """Validate the request against the whole deck, then narrow to it."""
        if not catalogue:
            raise CardPreviewError(
                f"{deck_path} builds no cards, so there is nothing to preview."
            )
        grouped: dict[str, list[_CardRef]] = {}
        for ref in catalogue:
            grouped.setdefault(ref.record_id, []).append(ref)
        new_ids = _new_ids(grouped, new_record_ids)
        selected = _selected_ids(grouped, scope_record_ids)
        chosen = tuple(ref for record_id in selected for ref in grouped[record_id])
        if len(chosen) > MAX_PREVIEW_CARDS:
            raise CardPreviewError(
                f"{deck_path} would show {len(chosen)} cards, over the "
                f"{MAX_PREVIEW_CARDS}-card preview limit. Preview an exact scope "
                "rather than a page with cards left out."
            )
        selection.update(
            new_ids=new_ids,
            selected=selected,
            deck_note_count=len(grouped),
            deck_card_count=len(catalogue),
        )
        return chosen

    with tempfile.TemporaryDirectory(prefix="janki-card-preview-") as temporary:
        scratch = Path(temporary)
        if overlay:
            mirror_root = scratch / "project"
            mirror_root.mkdir()
            build_config, build_deck_path = _mirrored_project(
                config, target, overlay, mirror_root
            )
        else:
            build_config, build_deck_path = config, target

        package = scratch / "preview.apkg"
        plan = _build_package(build_config, build_deck_path, package)
        drawn, card_css = _drawn_cards(package, scratch, plan, choose)

    new_ids: frozenset[str] = selection["new_ids"]
    selected: tuple[str, ...] = selection["selected"]
    deck_note_count: int = selection["deck_note_count"]
    deck_card_count: int = selection["deck_card_count"]

    cards = tuple(
        PreviewCard(
            record_id=item.record_id,
            label=item.label,
            template=item.template,
            direction=item.direction,
            question_html=item.question_html,
            answer_html=item.answer_html,
            is_new=item.record_id in new_ids,
        )
        for item in drawn
    )
    new_note_count = sum(1 for record_id in selected if record_id in new_ids)

    payload, policy = _document(
        config,
        deck_name=plan.deck_name,
        kind=plan.kind,
        subtitle=subtitle,
        directions=plan.directions,
        cards=cards,
        note_count=len(selected),
        new_note_count=new_note_count,
        deck_note_count=deck_note_count,
        deck_card_count=deck_card_count,
        card_css=card_css,
    )
    return CardPreview(
        deck_name=plan.deck_name,
        deck_kind=plan.kind,
        directions=plan.directions,
        note_count=len(selected),
        card_count=len(cards),
        new_note_count=new_note_count,
        deck_note_count=deck_note_count,
        deck_card_count=deck_card_count,
        cards=cards,
        html=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        content_security_policy=policy,
    )


def write_card_preview(preview: CardPreview, output_path: Path) -> Path:
    """Write the exact rendered bytes, and return where they landed."""
    target = Path(os.path.abspath(os.fspath(output_path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(preview.html)
    return target
