"""Prepare and apply one exact batch of character notes.

Two calls, and the split is the whole design. :func:`prepare_character_notes`
may look things up — the reference cache for a character nobody has fetched,
the provider's pages for facts that are missing or explicitly refreshed — and
turns what it found into the **exact bytes** every affected file would hold.
It writes none of them. :func:`execute_character_notes` writes exactly those
bytes and does nothing else: no request, no second plan, no model, no build.

That is what makes an interrupted apply recoverable. Each target file may be
either the exact state the plan saw or the exact state the plan proposes — the
first means "not applied yet", the second means "already applied", and a
resume finishes the rest without asking for the evidence again. Any third
state is somebody else's edit, and the batch refuses before writing anything
further rather than deciding whose change wins.

:func:`verify_character_notes_applied` reads that same plan once more, after
the apply and before a package is built from it, and accepts only the second
of those states. The apply has to tolerate both because an interrupted batch
must be able to finish; a build has no such excuse, because what it publishes
is whatever those files say at the moment it reads them. It writes nothing,
asks nobody anything, and prepares no second plan.

The plan carries the deck too, so creating a character deck and writing the
notes it ships are one confirmation rather than two: a deck naming characters
that do not exist yet is not a state anyone should have to see.

Nothing here reads Japanese. It chooses no readings, ranks no examples, and
writes no cue; those come from the sources, from the curated store, or from
the owner.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import secrets
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML, YAMLError

from japanese_anki import jpdb_kanji, kanji, kanji_notes
from japanese_anki.application import deck_creation
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.exporters import kanji_cards
from japanese_anki.exporters.anki import deck_kind
from japanese_anki.identifiers import character_record_id
from japanese_anki.io import (
    DataError,
    atomic_write_bytes_bound,
    exclusive_path_lock,
    load_structured,
    read_bytes_bound,
)
from japanese_anki.kanji_notes import CharacterNote

__all__ = [
    "CharacterNotesError",
    "CharacterNotesPlan",
    "CharacterNotesResult",
    "ProposedFile",
    "execute_character_notes",
    "prepare_character_notes",
    "verify_character_notes_applied",
]

#: One refusal for every way a plan can stop being the batch that was prepared.
#: Shared so the apply and the after-state check cannot drift into saying two
#: different things about the same condition.
_PLAN_STALE = (
    "[character-plan-stale] the exact character batch changed after it was "
    "prepared. Nothing was written; prepare it again."
)


class CharacterNotesError(JankiError):
    """A character-note batch cannot be prepared, or is no longer the one shown."""


@dataclass(frozen=True, slots=True)
class ProposedFile:
    """One file's exact before state and, when it changes, its exact after.

    ``after_text is None`` means the plan proposes no change and requires the
    file to still be what it was: the batch binds every file it read, not only
    the ones it rewrites, so an edit to any of them stops the apply.
    """

    label: str
    path: Path
    before_sha256: str | None
    after_text: str | None

    @property
    def changed(self) -> bool:
        return self.after_text is not None

    @property
    def after_sha256(self) -> str | None:
        if self.after_text is None:
            return self.before_sha256
        return hashlib.sha256(self.after_text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": str(self.path),
            "before_sha256": self.before_sha256,
            "after_text": self.after_text,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ProposedFile:
        return cls(
            label=str(raw["label"]),
            path=Path(str(raw["path"])),
            before_sha256=(
                None if raw["before_sha256"] is None else str(raw["before_sha256"])
            ),
            after_text=None if raw["after_text"] is None else str(raw["after_text"]),
        )


@dataclass(frozen=True, slots=True)
class CharacterNotesPlan:
    """Everything one confirmed character batch will do, and nothing it did."""

    project_root: Path
    deck_path: Path
    deck_name: str
    output_path: Path
    #: True when this batch also creates the deck. The deck and its notes land
    #: together; a deck naming characters that do not exist is not a state
    #: anyone should have to confirm their way out of.
    deck_created: bool
    characters: tuple[str, ...]
    directions: tuple[str, ...]
    notes: tuple[CharacterNote, ...]
    #: Every identity the deck names once this batch has landed, in the order
    #: the file lists them — the ones already there, then the ones added. The
    #: selection above is what this batch is *for*; this is what the package
    #: built afterwards will hold, and the two differ whenever the destination
    #: already had notes.
    deck_record_ids: tuple[str, ...]
    files: tuple[ProposedFile, ...]
    #: Characters whose reference entry this preparation looked up, and whose
    #: provider facts it requested. Named so a plan can say what it asked for.
    looked_up: tuple[str, ...]
    fetched_readings: tuple[str, ...]
    fingerprint: str

    @property
    def note_count(self) -> int:
        return len(self.notes)

    @property
    def card_count(self) -> int:
        return len(self.notes) * len(self.directions)

    @property
    def deck_note_count(self) -> int:
        return len(self.deck_record_ids)

    @property
    def deck_card_count(self) -> int:
        return len(self.deck_record_ids) * len(self.directions)

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(note.id for note in self.notes)

    @property
    def changed_files(self) -> tuple[ProposedFile, ...]:
        return tuple(item for item in self.files if item.changed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "project_root": str(self.project_root),
            "deck_path": str(self.deck_path),
            "deck_name": self.deck_name,
            "output_path": str(self.output_path),
            "deck_created": self.deck_created,
            "characters": list(self.characters),
            "directions": list(self.directions),
            "notes": [
                {"character": note.character, **note.to_dict()} for note in self.notes
            ],
            "deck_record_ids": list(self.deck_record_ids),
            # Derived, and written down anyway: a reader validating a built
            # package against this plan should not have to re-derive the
            # number it is checking, and a stored figure that disagrees with
            # the identities beside it is a corrupt plan rather than a
            # rounding difference.
            "note_count": self.note_count,
            "card_count": self.card_count,
            "deck_note_count": self.deck_note_count,
            "deck_card_count": self.deck_card_count,
            "files": [item.to_dict() for item in self.files],
            "looked_up": list(self.looked_up),
            "fetched_readings": list(self.fetched_readings),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CharacterNotesPlan:
        """Rebuild a plan in another process, exactly as it was prepared.

        A finish receipt resumes from this: the evidence a batch was confirmed
        against is in the plan, so recovery never has to ask a provider what
        it said.
        """
        try:
            if raw.get("schema_version") != 1:
                raise CharacterNotesError(
                    f"Unknown character-notes plan version {raw.get('schema_version')!r}."
                )
            notes = tuple(
                CharacterNote.from_dict(str(item["character"]), item)
                for item in raw["notes"]
            )
            plan = cls(
                project_root=Path(str(raw["project_root"])),
                deck_path=Path(str(raw["deck_path"])),
                deck_name=str(raw["deck_name"]),
                output_path=Path(str(raw["output_path"])),
                deck_created=bool(raw["deck_created"]),
                characters=tuple(str(item) for item in raw["characters"]),
                directions=tuple(str(item) for item in raw["directions"]),
                notes=notes,
                deck_record_ids=tuple(str(item) for item in raw["deck_record_ids"]),
                files=tuple(ProposedFile.from_dict(item) for item in raw["files"]),
                looked_up=tuple(str(item) for item in raw["looked_up"]),
                fetched_readings=tuple(str(item) for item in raw["fetched_readings"]),
                fingerprint=str(raw["fingerprint"]),
            )
            counted = {
                "note_count": plan.note_count,
                "card_count": plan.card_count,
                "deck_note_count": plan.deck_note_count,
                "deck_card_count": plan.deck_card_count,
            }
            wrong = [key for key, value in counted.items() if raw[key] != value]
            if wrong:
                raise CharacterNotesError(
                    f"This character-notes plan says {wrong[0]} is {raw[wrong[0]]!r}, "
                    f"and the identities it carries make it {counted[wrong[0]]}."
                )
        except CharacterNotesError:
            raise
        except (JankiError, KeyError, TypeError, ValueError) as exc:
            raise CharacterNotesError(
                f"This is not a complete character-notes plan: {exc}"
            ) from exc
        if _fingerprint(plan) != plan.fingerprint:
            raise CharacterNotesError(
                "[character-plan-stale] the stored character-notes plan does not "
                "match its own fingerprint. Nothing was written; plan it again."
            )
        return plan


@dataclass(frozen=True, slots=True)
class CharacterNotesResult:
    """What one apply landed, and which files it had to write to land it."""

    deck_path: Path
    output_path: Path
    #: The identities this batch selected, and what they cost in cards.
    record_ids: tuple[str, ...]
    note_count: int
    card_count: int
    #: What the deck now holds altogether, which is what a build of it will
    #: report. Equal to the pair above only when the destination was new.
    deck_record_ids: tuple[str, ...]
    deck_note_count: int
    deck_card_count: int
    #: The labels of the files this run wrote. Empty when every file already
    #: held the proposed bytes — an exact retry after an interrupted apply.
    changed: tuple[str, ...]


def _fingerprint(plan: CharacterNotesPlan) -> str:
    payload = plan.to_dict()
    payload["fingerprint"] = ""
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _exact_characters(characters: Sequence[str]) -> tuple[str, ...]:
    if isinstance(characters, str):
        raise CharacterNotesError(
            "Character targets are a list of single characters, not one string."
        )
    targets = tuple(characters)
    if not targets:
        raise CharacterNotesError("Name at least one character to prepare.")
    seen: dict[str, None] = {}
    for character in targets:
        if not isinstance(character, str):
            raise CharacterNotesError("Every character target must be text.")
        try:
            character_record_id(character)
        except JankiError as exc:
            raise CharacterNotesError(str(exc)) from exc
        if character in seen:
            raise CharacterNotesError(
                f"{character} was named more than once; one character is one note."
            )
        seen[character] = None
    return targets


def _exact_directions(directions: Sequence[str]) -> tuple[str, ...]:
    if isinstance(directions, str):
        raise CharacterNotesError("Card directions are a list, not one string.")
    wanted = tuple(directions)
    if not wanted:
        raise CharacterNotesError("Enable at least one card direction.")
    unknown = [name for name in wanted if name not in kanji_cards.KANJI_CARD_FILES]
    if unknown:
        raise CharacterNotesError(
            f"A character note has no {unknown[0]!r} card. Valid directions: "
            + ", ".join(kanji_cards.KANJI_CARD_FILES)
        )
    repeated = sorted({name for name in wanted if wanted.count(name) > 1})
    if repeated:
        raise CharacterNotesError(f"Direction {repeated[0]!r} was named twice.")
    # Card order, not argument order: a template's ordinal is its identity in a
    # collection with review history.
    return tuple(name for name in kanji_cards.KANJI_CARD_FILES if name in wanted)


def _exact_cues(
    cues: Mapping[str, str] | None, targets: tuple[str, ...]
) -> dict[str, str]:
    if cues is None:
        return {}
    if not isinstance(cues, Mapping):
        raise CharacterNotesError(
            "Production cues are a mapping of character to the owner's text."
        )
    exact: dict[str, str] = {}
    for character, text in cues.items():
        if character not in targets:
            raise CharacterNotesError(
                f"A production cue was supplied for {character!r}, which is not "
                "one of the characters being prepared."
            )
        if not isinstance(text, str) or not text.strip():
            raise CharacterNotesError(
                f"The production cue for {character} is blank. janki does not "
                "write one: a production card asks for a character from a hint, "
                "and that hint is study content."
            )
        exact[character] = text.strip()
    return exact


def _snapshot(path: Path) -> tuple[str | None, str | None]:
    """One file's exact current text and its hash, or ``(None, None)``."""
    try:
        payload = read_bytes_bound(path)
    except FileNotFoundError:
        return None, None
    except (DataError, OSError) as exc:
        raise CharacterNotesError(f"Could not read {path}: {exc}") from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise CharacterNotesError(f"{path} is not UTF-8 text: {exc}") from exc
    return text, hashlib.sha256(payload).hexdigest()


def _serialized(write: Any, store: Any) -> str:
    """The exact bytes a store writer would produce, without publishing them.

    Through the writer that owns the format rather than a second copy of it:
    the plan shows what the file will hold, and "what the file will hold" is
    whatever that module writes.
    """
    with tempfile.TemporaryDirectory(prefix="janki-character-plan-") as into:
        scratch = Path(into) / "store.json"
        write(scratch, store)
        return scratch.read_text(encoding="utf-8")


def _reference(
    character: str, store: kanji.KanjiStore, looked_up: list[str]
) -> kanji.KanjiInfo:
    existing = store.entries.get(character)
    if existing is not None:
        return existing
    try:
        info = kanji.fetch_kanji(character)
    except JankiError as exc:
        raise CharacterNotesError(
            f"Could not look up reference data for {character}: {exc}. Nothing "
            "was written."
        ) from exc
    if info.character != character:
        raise CharacterNotesError(
            f"The reference lookup for {character} answered for {info.character!r}."
        )
    store.entries[character] = info
    looked_up.append(character)
    return info


def _facts(
    config: ProjectConfig,
    character: str,
    store: dict[str, jpdb_kanji.CharacterReadings],
    fetched: list[str],
    *,
    refresh: bool,
) -> jpdb_kanji.CharacterReadings:
    existing = store.get(character)
    if existing is not None and not refresh:
        return existing
    try:
        readings = jpdb_kanji.fetch_character(
            character, html_cache=config.jpdb_html_cache, refresh=refresh
        )
    except JankiError as exc:
        raise CharacterNotesError(
            f"Could not read the published readings for {character}: {exc}. "
            "Nothing was written."
        ) from exc
    store[character] = readings
    fetched.append(character)
    return readings


def _new_note(
    character: str,
    info: kanji.KanjiInfo,
    readings: jpdb_kanji.CharacterReadings,
    cue: str,
) -> CharacterNote:
    evidence = kanji_notes.evidence_from_readings(readings)
    sources = [f"kanjiapi.dev (KANJIDIC2) · {character}"]
    if info.strokes:
        # KanjiVG is CC BY-SA 3.0 and a deck shipping its strokes has to credit
        # it where a learner can read the credit. Named in full on the note
        # rather than once in a README: the note is what leaves this repository.
        sources.append(
            "KanjiVG stroke order (https://kanjivg.tagaini.net/) · "
            "CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/)"
        )
    return CharacterNote(
        character=character,
        id=character_record_id(character),
        meanings=info.meanings,
        stroke_count=info.stroke_count,
        strokes=info.strokes,
        kanjidic_readings=tuple(
            kanji_notes.KanjidicReading(kind=item.kind, reading=item.reading)
            for item in info.readings
        ),
        reading_evidence=evidence,
        reading_example=kanji_notes.first_bound_example(evidence),
        production_cue=cue,
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sources=tuple(sources),
    )


def _updated_note(
    existing: CharacterNote,
    readings: jpdb_kanji.CharacterReadings,
    cue: str,
    *,
    refresh: bool,
) -> CharacterNote:
    """An existing note, with only what was explicitly asked for changed.

    A rerun over a character already prepared writes nothing: its meanings,
    its evidence and its fixed reading prompt are curated values, and a
    refresh of a cache is not an instruction to replace them. Only an explicit
    reading refresh moves the evidence, and even then the prompt stays where
    it is — the question a card with review history asks does not change
    because the source's page did.
    """
    note = existing
    if refresh:
        note = replace(note, reading_evidence=kanji_notes.evidence_from_readings(readings))
    if note.reading_example is None:
        prompt = kanji_notes.first_bound_example(note.reading_evidence)
        if prompt is not None:
            note = replace(note, reading_example=prompt)
    if cue:
        note = replace(note, production_cue=cue)
    return note


def _refuse_unsupported(note: CharacterNote, wanted: tuple[str, ...]) -> None:
    """Refuse a direction this note cannot carry, saying what it would need.

    By presence alone — a fixed bound example, a cue the owner wrote. Nothing
    here decides whether either is any good.
    """
    supported = kanji_notes.supported_directions(note)
    missing = [name for name in wanted if name not in supported]
    if not missing:
        return
    raise CharacterNotesError(
        f"{note.character} supports no {missing[0]} card. "
        + (
            "A reading card asks one example the source bound to a reading, "
            "and this character has none."
            if missing[0] == "reading"
            else "A production card needs a disambiguating cue, which only the "
            "owner writes: pass one for this character."
        )
    )


@dataclass(frozen=True, slots=True)
class _BoundDeck:
    """One deck's exact current bytes, its proposal, and what it will name."""

    path: Path
    name: str
    output_path: Path
    #: Every identity the file lists once this batch has landed, in file order.
    record_ids: tuple[str, ...]
    before_text: str | None
    before_sha256: str | None
    after_text: str | None
    created: bool


def _existing_deck(
    config: ProjectConfig,
    deck_path: Path,
    directions: tuple[str, ...],
    record_ids: tuple[str, ...],
) -> _BoundDeck:
    """Bind an existing character deck, and the ids it still has to name.

    Its exact current bytes come back with the proposal built from them, so
    the two cannot describe different versions of the file.
    """
    target = Path(deck_path).resolve()
    try:
        kind = deck_kind(target)
    except JankiError as exc:
        raise CharacterNotesError(f"Could not read {target}: {exc}") from exc
    if kind != "kanji":
        raise CharacterNotesError(
            f"{target} is a {kind or 'vocabulary'} deck, and character notes are "
            "not word cards. Name a deck with 'kind: kanji', or a new deck name."
        )
    raw = load_structured(target)
    section = raw.get("deck") if isinstance(raw, dict) else None
    if not isinstance(section, Mapping):
        raise CharacterNotesError(f"The deck section must be a mapping: {target}")
    current = kanji_cards.deck_directions(section, target)
    if current != directions:
        raise CharacterNotesError(
            f"{target} builds {'+'.join(current)} cards and this batch asks for "
            f"{'+'.join(directions)}. A deck's enabled card set is fixed once it "
            "has review history; create a separate deck instead."
        )
    text, sha = _snapshot(target)
    if text is None:  # pragma: no cover - deck_kind read it a moment ago
        raise CharacterNotesError(f"Deck file no longer exists: {target}")
    after, declared = _deck_text_with_ids(text, target, record_ids)
    filename = str(section.get("output", f"{target.stem}.apkg"))
    return _BoundDeck(
        path=target,
        name=str(section.get("name") or target.stem),
        output_path=(config.dist_dir / filename).resolve(),
        record_ids=declared,
        before_text=text,
        before_sha256=sha,
        after_text=after,
        created=False,
    )


def _new_deck(
    config: ProjectConfig,
    deck_name: str,
    directions: tuple[str, ...],
    record_ids: tuple[str, ...],
) -> _BoundDeck:
    """Bind the character deck this batch also creates, through the one creator.

    The deck file's exact bytes come from :mod:`deck_creation`, which owns
    stems, ids and the deck-set checks; nothing about a new deck is written
    twice.
    """
    try:
        plan = deck_creation.plan_study_deck(
            config,
            name=deck_name,
            kind="kanji",
            include_ids=list(record_ids),
            recognition="recognition" in directions,
            reading="reading" in directions,
            production="production" in directions,
        )
    except JankiError as exc:
        raise CharacterNotesError(str(exc)) from exc
    target = plan.path.resolve()
    before_text, sha = _snapshot(target)
    if before_text is not None:
        # `plan_study_deck` refuses a taken stem under the deck-directory lock,
        # so reaching this needs a file to appear in the moment after it let go.
        raise CharacterNotesError(
            f"A deck already exists at {target}; name the existing deck "
            "instead of creating another."
        )
    return _BoundDeck(
        path=target,
        name=plan.name,
        output_path=plan.output_path,
        record_ids=record_ids,
        before_text=None,
        before_sha256=sha,
        after_text=plan.yaml_bytes.decode("utf-8"),
        created=True,
    )


def _deck_text_with_ids(
    text: str, deck_path: Path, record_ids: tuple[str, ...]
) -> tuple[str | None, tuple[str, ...]]:
    """The deck text with any missing ids appended, and every id it then names.

    The text is ``None`` when nothing is missing. Round-tripped rather than
    re-rendered: a deck file is curated, and re-dumping it would drop the
    comments and quoting its owner wrote.
    """
    parser = YAML()
    parser.preserve_quotes = True
    parser.allow_unicode = True
    parser.indent(mapping=2, sequence=4, offset=2)
    try:
        document = parser.load(io.StringIO(text))
    except YAMLError as exc:
        raise CharacterNotesError(f"Could not read {deck_path} for rewriting: {exc}") from exc
    section = document.get("deck") if hasattr(document, "get") else None
    if section is None:
        raise CharacterNotesError(f"The deck section must be a mapping: {deck_path}")
    declared = section.get("include_ids")
    if declared is None:
        raise CharacterNotesError(
            f"{deck_path}: a character deck names the exact characters it holds, "
            "and this one has no include_ids list."
        )
    # Proved before anything iterates it. A scalar cannot be iterated and a
    # mapping cannot be appended to, so without this the batch dies on the
    # append with a bare AttributeError naming no file.
    if isinstance(declared, str | bytes) or not isinstance(declared, Sequence):
        raise CharacterNotesError(
            f"{deck_path}: a character deck names the exact characters it holds "
            f"as a list, and this one has {type(declared).__name__}."
        )
    present = [str(item) for item in declared]
    missing = [record_id for record_id in record_ids if record_id not in present]
    if not missing:
        return None, tuple(present)
    for record_id in missing:
        declared.append(record_id)
    stream = io.StringIO()
    parser.dump(document, stream)
    return stream.getvalue(), tuple(present + missing)


def prepare_character_notes(
    config: ProjectConfig,
    characters: Sequence[str],
    *,
    deck_path: Path | str | None = None,
    deck_name: str | None = None,
    directions: Sequence[str] = ("recognition",),
    refresh_readings: bool = False,
    production_cues: Mapping[str, str] | None = None,
) -> CharacterNotesPlan:
    """Look up what is missing and return the exact batch, writing nothing.

    Saved facts are reused: a character already in the stores costs no request
    and its curated note is preserved unchanged, so an ordinary rerun proposes
    no change at all. ``refresh_readings`` is the only path that replaces
    evidence already saved.
    """
    targets = _exact_characters(characters)
    wanted = _exact_directions(directions)
    cues = _exact_cues(production_cues, targets)
    if not isinstance(refresh_readings, bool):
        raise CharacterNotesError("refresh_readings must be true or false.")
    if (deck_path is None) == (deck_name is None):
        raise CharacterNotesError(
            "Name either an existing character deck or a name for a new one."
        )

    notes_path = config.kanji_notes_file.resolve()
    kanji_path = config.kanji_file.resolve()
    readings_path = config.jpdb_readings_file.resolve()

    notes_before, notes_sha = _snapshot(notes_path)
    kanji_before, kanji_sha = _snapshot(kanji_path)
    readings_before, readings_sha = _snapshot(readings_path)

    try:
        store = kanji_notes.load_notes(notes_path)
        reference = kanji.load_store(kanji_path)
        facts = jpdb_kanji.load_readings(readings_path)
    except JankiError as exc:
        raise CharacterNotesError(str(exc)) from exc

    looked_up: list[str] = []
    fetched: list[str] = []
    prepared: list[CharacterNote] = []
    for character in targets:
        info = _reference(character, reference, looked_up)
        readings = _facts(
            config, character, facts, fetched, refresh=refresh_readings
        )
        existing = store.get(character)
        note = (
            _new_note(character, info, readings, cues.get(character, ""))
            if existing is None
            else _updated_note(
                existing, readings, cues.get(character, ""), refresh=refresh_readings
            )
        )
        store[character] = note
        prepared.append(note)

    for note in prepared:
        _refuse_unsupported(note, wanted)

    record_ids = tuple(note.id for note in prepared)
    deck = (
        _existing_deck(config, Path(deck_path), wanted, record_ids)
        if deck_path is not None
        else _new_deck(config, str(deck_name), wanted, record_ids)
    )

    # Every note the deck will hold, not only the ones this batch selected:
    # the counts below claim what a build of it will report, and a build reads
    # the whole `include_ids` list. A destination already naming a character
    # nobody has a note for, or one that cannot carry an enabled direction,
    # builds nothing — and saying so here is the difference between a refusal
    # before the write and a broken package after it.
    by_id = {note.id: note for note in store.values()}
    for record_id in deck.record_ids:
        note = by_id.get(record_id)
        if note is None:
            raise CharacterNotesError(
                f"{deck.path} names {record_id}, and there is no note for it in "
                f"{notes_path}. Prepare that character too, or take it out of "
                "the deck: this batch cannot say what the deck would build."
            )
        _refuse_unsupported(note, wanted)

    notes_after = kanji_notes.render_notes(store)
    kanji_after = _serialized(kanji.save_store, reference) if looked_up else None
    readings_after = (
        _serialized(jpdb_kanji.save_readings, facts) if fetched else None
    )
    files = (
        ProposedFile(
            label="character notes",
            path=notes_path,
            before_sha256=notes_sha,
            after_text=None if notes_after == notes_before else notes_after,
        ),
        ProposedFile(
            label="kanji reference store",
            path=kanji_path,
            before_sha256=kanji_sha,
            after_text=None if kanji_after == kanji_before else kanji_after,
        ),
        ProposedFile(
            label="jpdb reading facts",
            path=readings_path,
            before_sha256=readings_sha,
            after_text=None if readings_after == readings_before else readings_after,
        ),
        ProposedFile(
            label="character deck",
            path=deck.path,
            before_sha256=deck.before_sha256,
            after_text=(
                None if deck.after_text == deck.before_text else deck.after_text
            ),
        ),
    )

    draft = CharacterNotesPlan(
        project_root=config.root.resolve(),
        deck_path=deck.path,
        deck_name=deck.name,
        output_path=deck.output_path,
        deck_created=deck.created,
        characters=targets,
        directions=wanted,
        notes=tuple(prepared),
        deck_record_ids=deck.record_ids,
        files=files,
        looked_up=tuple(looked_up),
        fetched_readings=tuple(fetched),
        fingerprint="",
    )
    return replace(draft, fingerprint=_fingerprint(draft))


def _expected_paths(config: ProjectConfig) -> dict[str, Path]:
    return {
        "character notes": config.kanji_notes_file.resolve(),
        "kanji reference store": config.kanji_file.resolve(),
        "jpdb reading facts": config.jpdb_readings_file.resolve(),
    }


def _assert_plan_binding(config: ProjectConfig, plan: CharacterNotesPlan) -> None:
    """Refuse a plan that is not this project's own, unaltered batch.

    Shared by the apply and the after-state check so the two cannot disagree
    about which repository a batch belongs to, which files it is allowed to
    name, or whether it is still the batch that was fingerprinted.
    """
    if plan.project_root != config.root.resolve():
        raise CharacterNotesError(
            "This character batch belongs to a different repository configuration."
        )
    expected = _expected_paths(config)
    for item in plan.files:
        if item.label in expected and item.path != expected[item.label]:
            raise CharacterNotesError(
                f"This character batch writes {item.path}, but the project's "
                f"{item.label} is {expected[item.label]}."
            )
    if plan.deck_path.parent != config.deck_dir.resolve():
        raise CharacterNotesError(
            f"This character batch writes a deck outside {config.deck_dir}."
        )
    if not secrets.compare_digest(_fingerprint(plan), plan.fingerprint):
        raise CharacterNotesError(_PLAN_STALE)


@contextlib.contextmanager
def _bound_files(
    config: ProjectConfig, plan: CharacterNotesPlan
) -> Iterator[tuple[ProposedFile, ...]]:
    """Hold every file this batch bound, in one fixed order, and hand them back.

    One order for the apply and for the after-state check: two callers taking
    these locks in different orders could wait on each other forever, and a
    state read without the lock is a state that may already have moved by the
    time its reader acts on it.
    """
    ordered = tuple(sorted(plan.files, key=lambda item: str(item.path)))
    with contextlib.ExitStack() as locks:
        locks.enter_context(exclusive_path_lock(config.deck_dir))
        for item in ordered:
            locks.enter_context(exclusive_path_lock(item.path))
        yield ordered


def verify_character_notes_applied(
    config: ProjectConfig, plan: CharacterNotesPlan
) -> None:
    """Refuse unless every file this batch bound holds exactly what it applied.

    Read-only, and deliberately separate from the apply. A package plan
    captures the bytes it is about to build from; this says whether those
    bytes are still the confirmed ones. The apply accepts the before state
    too, because an interrupted batch has to be able to finish — here that
    state means the notes a package would ship were never written, which is a
    refusal rather than a resume.

    Nothing here writes, prepares, looks anything up, or builds; a caller runs
    it between planning a package and publishing one, and the package executor
    still re-checks its own inputs under its own locks afterwards.
    """
    _assert_plan_binding(config, plan)
    # The same completeness gate a plan restored from a receipt passes through,
    # applied to this one: a build must not be authorized by a plan that is not
    # a whole, self-consistent, restorable batch.
    if CharacterNotesPlan.from_dict(plan.to_dict()) != plan:
        raise CharacterNotesError(_PLAN_STALE)

    with _bound_files(config, plan) as ordered:
        for item in ordered:
            _text, current = _snapshot(item.path)
            if current == item.after_sha256:
                continue
            if current == item.before_sha256:
                raise CharacterNotesError(
                    f"[character-not-applied] {item.path} still holds the state "
                    "this batch was prepared against, so its character notes "
                    "were never written. Nothing was built."
                )
            raise CharacterNotesError(
                f"[character-batch-stale] {item.path} is no longer the state "
                "this batch applied; something changed it afterwards. Nothing "
                "was built."
            )


def execute_character_notes(
    config: ProjectConfig,
    plan: CharacterNotesPlan,
    *,
    expected_fingerprint: str,
) -> CharacterNotesResult:
    """Write exactly the prepared bytes, or refuse before writing any of them.

    No lookup, no re-plan, no build. Every file must be either what the plan
    saw or what the plan proposes; the second is an apply that was already
    done, which is how an interrupted batch resumes without paying for the
    evidence again.
    """
    _assert_plan_binding(config, plan)
    if not isinstance(expected_fingerprint, str) or not secrets.compare_digest(
        expected_fingerprint, plan.fingerprint
    ):
        raise CharacterNotesError(_PLAN_STALE)

    written: list[str] = []
    with _bound_files(config, plan) as ordered:
        # Every file is checked before any is written, so a batch that would
        # refuse halfway leaves nothing half-applied.
        pending: list[ProposedFile] = []
        for item in ordered:
            _text, current = _snapshot(item.path)
            if current == item.after_sha256:
                continue
            if current != item.before_sha256:
                raise CharacterNotesError(
                    f"[character-batch-stale] {item.path} is neither the state "
                    "this batch was prepared against nor the state it proposes. "
                    "Nothing further was written; prepare it again."
                )
            if item.changed:
                pending.append(item)

        for item in pending:
            assert item.after_text is not None
            try:
                atomic_write_bytes_bound(
                    item.path,
                    item.after_text.encode("utf-8"),
                    expected_revision=item.before_sha256,
                    expected_absent=item.before_sha256 is None,
                )
            except (DataError, OSError) as exc:
                raise CharacterNotesError(
                    f"Could not write {item.path}: {exc}"
                ) from exc
            written.append(item.label)

    return CharacterNotesResult(
        deck_path=plan.deck_path,
        output_path=plan.output_path,
        record_ids=plan.record_ids,
        note_count=plan.note_count,
        card_count=plan.card_count,
        deck_record_ids=plan.deck_record_ids,
        deck_note_count=plan.deck_note_count,
        deck_card_count=plan.deck_card_count,
        changed=tuple(written),
    )
