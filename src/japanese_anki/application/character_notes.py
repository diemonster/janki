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
from japanese_anki.identifiers import character_record_id, contains_kanji
from japanese_anki.io import (
    DataError,
    atomic_write_bytes_bound,
    exclusive_path_lock,
    load_structured,
    read_bytes_bound,
)
from japanese_anki.kanji_notes import CharacterNote

__all__ = [
    "REFERENCE_FILE_LABELS",
    "CharacterNotesError",
    "CharacterNotesPlan",
    "CharacterNotesResult",
    "MissingReferenceFact",
    "ProposedFile",
    "ReferenceFactsPreparation",
    "ReferenceFactsResult",
    "apply_prepared_reference_facts",
    "execute_character_notes",
    "prepare_character_notes",
    "prepare_reference_facts",
    "recover_prepared_reference_facts",
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
        # From the text this batch bound above, not from a second read: the
        # digest and the store a proposal is built on have to come from one
        # read, or a write that is undone before apply lands as a silent loss.
        reference = kanji.parse_store(kanji_before, source=kanji_path)
        facts = jpdb_kanji.parse_readings(readings_before, source=readings_path)
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
def _bound_paths(
    files: Sequence[ProposedFile], *, deck_dir: Path | None = None
) -> Iterator[tuple[ProposedFile, ...]]:
    """Hold every file a prepared payload bound, in one fixed order.

    One order for every caller: two of them taking these locks in different
    orders could wait on each other forever, and a state read without the lock
    is a state that may already have moved by the time its reader acts on it.
    ``deck_dir`` is taken first where a batch also writes a deck; reference-only
    work names no deck and takes no deck lock.
    """
    ordered = tuple(sorted(files, key=lambda item: str(item.path)))
    with contextlib.ExitStack() as locks:
        if deck_dir is not None:
            locks.enter_context(exclusive_path_lock(deck_dir))
        for item in ordered:
            locks.enter_context(exclusive_path_lock(item.path))
        yield ordered


def _bound_files(
    config: ProjectConfig, plan: CharacterNotesPlan
) -> contextlib.AbstractContextManager[tuple[ProposedFile, ...]]:
    """The character batch's own lock set: its four files under the deck lock."""
    return _bound_paths(plan.files, deck_dir=config.deck_dir)


def _apply_proposed_files(
    ordered: Sequence[ProposedFile], *, tag: str, subject: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Precheck **every** bound file, then write only the pending ones.

    The one low-level writer for both the character batch and §7.9's reference
    preparation, so neither can drift into a different rule about which states
    are acceptable. A file already holding the proposed bytes is complete —
    that is how an interrupted apply resumes without paying for the evidence
    again — and any third state refuses before anything further is written,
    which is what stops a stale later target after an earlier one has landed.

    Returns ``(already_complete, written)`` by label.
    """
    already: list[str] = []
    pending: list[ProposedFile] = []
    for item in ordered:
        _text, current = _snapshot(item.path)
        if current == item.after_sha256:
            if item.changed:
                already.append(item.label)
            continue
        if current != item.before_sha256:
            raise CharacterNotesError(
                f"[{tag}] {item.path} is neither the state "
                f"this {subject} was prepared against nor the state it proposes. "
                "Nothing further was written; prepare it again."
            )
        if item.changed:
            pending.append(item)

    written: list[str] = []
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
            raise CharacterNotesError(f"Could not write {item.path}: {exc}") from exc
        written.append(item.label)
    return tuple(already), tuple(written)


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

    with _bound_files(config, plan) as ordered:
        # Every file is checked before any is written, so a batch that would
        # refuse halfway leaves nothing half-applied.
        _already, written = _apply_proposed_files(
            ordered, tag="character-batch-stale", subject="batch"
        )

    return CharacterNotesResult(
        deck_path=plan.deck_path,
        output_path=plan.output_path,
        record_ids=plan.record_ids,
        note_count=plan.note_count,
        card_count=plan.card_count,
        deck_record_ids=plan.deck_record_ids,
        deck_note_count=plan.deck_note_count,
        deck_card_count=plan.deck_card_count,
        changed=written,
    )


# --- reference facts the reviewed word cards already need ----------------------
#
# Contracts §7.9. A word card reads both reference stores at build time and
# never fetches, so a newly promoted verb bringing a character neither store
# covers ships a card with a hole. This prepares those two files and **only**
# those two: it mints no character note, no `kanji:` identity and no character
# deck, and it changes nothing in the curated `data/kanji_notes.json`.
#
# `kanji_addition` cannot do this as written — it binds canonical bytes, so it
# cannot describe a projection before the preview, and it fetches and writes in
# one call, so it cannot hand a prepared payload to a later apply.


#: The two labels a reference preparation may bind, in the order it holds them.
#: A closed list: a preparation that named anything else would be a route to
#: writing a file this phase has no business in.
REFERENCE_FILE_LABELS: tuple[str, str] = (
    "kanji reference store",
    "jpdb reading facts",
)

_REFERENCE_STALE = (
    "[reference-facts-stale] the exact reference preparation changed after it "
    "was prepared. Nothing was written; prepare it again."
)


@dataclass(frozen=True, slots=True)
class MissingReferenceFact:
    """One requested fact nobody could supply, disclosed exactly as it failed.

    §7.9: an unavailable lookup is a **missing state in the preview** for the
    owner to decide on. Nothing here retries in the background, ranks anything,
    or invents an entry — a character with no saved facts and no answer simply
    stays absent from the store, and this says so and why.
    """

    character: str
    store: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "store": self.store,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MissingReferenceFact:
        return cls(
            character=str(raw["character"]),
            store=str(raw["store"]),
            detail=str(raw["detail"]),
        )


@dataclass(frozen=True, slots=True)
class ReferenceFactsPreparation:
    """Both reference stores' exact after-bytes, and what they cost to get.

    Frozen before the preview and written by a later apply, so what the owner
    reviewed is what lands. The payloads come from the stores' own writers —
    `kanji.save_store` and `jpdb_kanji.save_readings` — rendered into a private
    scratch path, so a preparation publishes nothing a reader treats as content.
    """

    project_root: Path
    characters: tuple[str, ...]
    refresh_readings: bool
    #: Characters whose reference entry this preparation looked up, and whose
    #: provider facts it requested. Named so a plan can say what it asked for.
    looked_up: tuple[str, ...]
    fetched_readings: tuple[str, ...]
    missing: tuple[MissingReferenceFact, ...]
    files: tuple[ProposedFile, ...]
    #: Stored rather than derived, exactly as `CharacterNotesPlan` stores its
    #: own. Nothing re-derives these payloads at apply — the whole point is
    #: that they were frozen before the preview — so a seal computed from
    #: whatever the object currently holds would agree with any edit made to
    #: it. This one disagrees.
    fingerprint: str

    @property
    def changed_files(self) -> tuple[ProposedFile, ...]:
        return tuple(item for item in self.files if item.changed)

    def file(self, label: str) -> ProposedFile | None:
        for item in self.files:
            if item.label == label:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "project_root": str(self.project_root),
            "characters": list(self.characters),
            "refresh_readings": self.refresh_readings,
            "looked_up": list(self.looked_up),
            "fetched_readings": list(self.fetched_readings),
            "missing": [item.to_dict() for item in self.missing],
            "files": [item.to_dict() for item in self.files],
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ReferenceFactsPreparation:
        try:
            if raw.get("schema_version") != 1:
                raise CharacterNotesError(
                    "Unknown reference-facts preparation version "
                    f"{raw.get('schema_version')!r}."
                )
            preparation = cls(
                project_root=Path(str(raw["project_root"])),
                characters=tuple(str(item) for item in raw["characters"]),
                refresh_readings=bool(raw["refresh_readings"]),
                looked_up=tuple(str(item) for item in raw["looked_up"]),
                fetched_readings=tuple(str(item) for item in raw["fetched_readings"]),
                missing=tuple(
                    MissingReferenceFact.from_dict(item) for item in raw["missing"]
                ),
                files=tuple(ProposedFile.from_dict(item) for item in raw["files"]),
                fingerprint=str(raw["fingerprint"]),
            )
        except CharacterNotesError:
            raise
        except (JankiError, KeyError, TypeError, ValueError) as exc:
            raise CharacterNotesError(
                f"This is not a complete reference-facts preparation: {exc}"
            ) from exc
        if _reference_fingerprint(preparation) != preparation.fingerprint:
            raise CharacterNotesError(_REFERENCE_STALE)
        return preparation


@dataclass(frozen=True, slots=True)
class ReferenceFactsResult:
    """What one apply or recovery found already done, and what it wrote."""

    #: The labels of the stores that already held the proposed bytes.
    already_complete: tuple[str, ...]
    #: The labels this call wrote. Empty on an exact retry.
    changed: tuple[str, ...]
    #: Carried through unchanged, because a preview that disclosed a missing
    #: fact must keep disclosing it after the apply.
    missing: tuple[MissingReferenceFact, ...]


def _reference_fingerprint(preparation: ReferenceFactsPreparation) -> str:
    payload = preparation.to_dict()
    payload["fingerprint"] = ""
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _reference_characters(characters: Sequence[str]) -> tuple[str, ...]:
    """The exact single kanji this preparation is for, deduplicated in order.

    Deduplicated rather than refused, unlike a character-note batch: the caller
    applies `kanji.kanji_in` per record over a whole collection, and a character
    two words share arrives twice. One character is still one lookup.

    Nothing here mints an identity. The check is only that each target is one
    kanji, which is what both providers require of a request.
    """
    if isinstance(characters, (str, bytes)):
        raise CharacterNotesError(
            "Reference characters are a list of single characters, not one string."
        )
    targets = tuple(dict.fromkeys(characters))
    if not targets:
        raise CharacterNotesError("Name at least one character to prepare facts for.")
    for character in targets:
        if not isinstance(character, str):
            raise CharacterNotesError("Every reference character must be text.")
        if len(character) != 1 or not contains_kanji(character):
            raise CharacterNotesError(
                f"Not a single kanji character: {character!r}"
            )
    return targets


def _reference_paths(config: ProjectConfig) -> tuple[Path, Path]:
    """This configuration's two reference stores, proved to be two files."""
    kanji_path = config.kanji_file.resolve()
    readings_path = config.jpdb_readings_file.resolve()
    if kanji_path == readings_path:
        raise CharacterNotesError(
            "The kanji reference store and the jpdb reading facts must be two "
            f"different files; this configuration points both at {kanji_path}."
        )
    return kanji_path, readings_path


def _optional_reference(
    character: str,
    store: kanji.KanjiStore,
    looked_up: list[str],
    missing: list[MissingReferenceFact],
) -> None:
    """Fill one character's inventory entry, or disclose why it is missing.

    Through the same `_reference` the character-note batch uses, so there is
    one lookup rule. The difference is only what an unavailable answer means:
    a character *note* cannot be written without facts and refuses, while a
    word card that already exists simply keeps its hole, and the owner is told
    which character and why.
    """
    try:
        _reference(character, store, looked_up)
    except CharacterNotesError as exc:
        missing.append(
            MissingReferenceFact(
                character=character,
                store=REFERENCE_FILE_LABELS[0],
                detail=str(exc.__cause__ or exc),
            )
        )


def _optional_facts(
    config: ProjectConfig,
    character: str,
    store: dict[str, jpdb_kanji.CharacterReadings],
    fetched: list[str],
    missing: list[MissingReferenceFact],
    *,
    refresh: bool,
) -> None:
    """Fill one character's reported-usage entry, or disclose the failure.

    A refresh that fails over a character the store already covers is a
    different state from having no facts at all, and the disclosure says which.
    The saved entry is untouched — `_facts` refuses before it writes one — so
    the qualifier is a plain statement of what the file still holds, added only
    when a refresh was asked for and there was something to keep.
    """
    kept = refresh and character in store
    try:
        _facts(config, character, store, fetched, refresh=refresh)
    except CharacterNotesError as exc:
        detail = str(exc.__cause__ or exc)
        if kept:
            detail += (
                " The refresh failed; the saved reading facts for this "
                "character were kept."
            )
        missing.append(
            MissingReferenceFact(
                character=character,
                store=REFERENCE_FILE_LABELS[1],
                detail=detail,
            )
        )


def prepare_reference_facts(
    config: ProjectConfig,
    characters: Sequence[str],
    *,
    refresh_readings: bool = False,
) -> ReferenceFactsPreparation:
    """Fill the two reference stores for ``characters``, writing nothing.

    Saved facts are reused: a character both stores already cover costs no
    request at all, and a rerun over a settled collection proposes no change.
    Only the characters that are missing are fetched, through `kanji.fetch_kanji`
    and `jpdb_kanji.fetch_character` under the on-demand lookup preference, and
    the provider's raw HTML goes to the private cache outside the repository.

    **Refresh is explicit only**, and refreshes only the JPDB reading pages:
    `refresh_readings` re-requests exactly the named characters' published
    usage. The KANJIDIC inventory is reused whatever it says — a refresh of one
    provider's pages is not an instruction to re-ask another.

    This mints no character note, no ``kanji:`` identity and no character deck,
    and never reads or writes the curated notes store. A lookup nobody can
    answer becomes an exact disclosed missing state rather than a guess or a
    refusal: the word cards already exist, and the hole is the owner's to
    decide about.
    """
    targets = _reference_characters(characters)
    if not isinstance(refresh_readings, bool):
        raise CharacterNotesError("refresh_readings must be true or false.")
    kanji_path, readings_path = _reference_paths(config)

    kanji_before, kanji_sha = _snapshot(kanji_path)
    readings_before, readings_sha = _snapshot(readings_path)
    try:
        # The bytes bound above are the ones parsed here. A second read could
        # answer from content that is replaced and restored before the apply's
        # compare-and-swap, which would then accept a proposal built on a store
        # nobody bound and drop the entries the bound one held.
        reference = kanji.parse_store(kanji_before, source=kanji_path)
        facts = jpdb_kanji.parse_readings(readings_before, source=readings_path)
    except JankiError as exc:
        raise CharacterNotesError(str(exc)) from exc

    looked_up: list[str] = []
    fetched: list[str] = []
    missing: list[MissingReferenceFact] = []
    for character in targets:
        _optional_reference(character, reference, looked_up, missing)
        _optional_facts(
            config, character, facts, fetched, missing, refresh=refresh_readings
        )

    # Through the writers that own the formats, into a private scratch path.
    # What the file will hold is whatever those modules write, and nothing a
    # reader treats as content is published here.
    kanji_after = _serialized(kanji.save_store, reference) if looked_up else None
    readings_after = _serialized(jpdb_kanji.save_readings, facts) if fetched else None

    draft = ReferenceFactsPreparation(
        project_root=config.root.resolve(),
        characters=targets,
        refresh_readings=refresh_readings,
        looked_up=tuple(looked_up),
        fetched_readings=tuple(fetched),
        missing=tuple(missing),
        files=(
            ProposedFile(
                label=REFERENCE_FILE_LABELS[0],
                path=kanji_path,
                before_sha256=kanji_sha,
                after_text=None if kanji_after == kanji_before else kanji_after,
            ),
            ProposedFile(
                label=REFERENCE_FILE_LABELS[1],
                path=readings_path,
                before_sha256=readings_sha,
                after_text=(
                    None if readings_after == readings_before else readings_after
                ),
            ),
        ),
        fingerprint="",
    )
    return replace(draft, fingerprint=_reference_fingerprint(draft))


def _assert_reference_binding(
    config: ProjectConfig, preparation: ReferenceFactsPreparation
) -> None:
    """Refuse a preparation that is not this project's own, unaltered payload.

    The repository root is not the binding on its own: two configurations in
    one root can name different reference stores, and a substituted target
    holding byte-identical content would pass every digest check. Each label is
    re-resolved through the *live* configuration and compared lexically, so an
    alias cannot stand in for the bound file.
    """
    if preparation.project_root != config.root.resolve():
        raise CharacterNotesError(
            "This reference preparation belongs to a different repository "
            "configuration."
        )
    labels = tuple(item.label for item in preparation.files)
    if labels != REFERENCE_FILE_LABELS:
        raise CharacterNotesError(
            "A reference preparation binds exactly "
            f"{' and '.join(REFERENCE_FILE_LABELS)}, in that order, and this "
            f"one binds {', '.join(labels) or 'nothing'}."
        )
    expected = dict(zip(REFERENCE_FILE_LABELS, _reference_paths(config), strict=True))
    for item in preparation.files:
        if item.path != expected[item.label]:
            raise CharacterNotesError(
                f"This reference preparation writes {item.path}, but the "
                f"project's {item.label} is {expected[item.label]}."
            )
    if not secrets.compare_digest(
        _reference_fingerprint(preparation), preparation.fingerprint
    ):
        raise CharacterNotesError(_REFERENCE_STALE)
    # The same completeness gate a preparation restored from a receipt passes
    # through, applied to this one: an apply must not be authorized by a
    # payload that is not a whole, self-consistent, restorable preparation.
    if ReferenceFactsPreparation.from_dict(preparation.to_dict()) != preparation:
        raise CharacterNotesError(_REFERENCE_STALE)


def _replay_reference_facts(
    config: ProjectConfig, preparation: ReferenceFactsPreparation
) -> ReferenceFactsResult:
    """Write exactly the prepared bytes for both stores, or refuse first.

    No lookup, no re-plan, no refetch. Both paths are measured under their own
    locks before either is written, so a stale second store stops the first
    store's effect instead of being discovered after it landed.
    """
    _assert_reference_binding(config, preparation)
    with _bound_paths(preparation.files) as ordered:
        already, written = _apply_proposed_files(
            ordered, tag="reference-facts-stale", subject="reference preparation"
        )
    return ReferenceFactsResult(
        already_complete=already, changed=written, missing=preparation.missing
    )


def apply_prepared_reference_facts(
    config: ProjectConfig, preparation: ReferenceFactsPreparation
) -> ReferenceFactsResult:
    """Publish one prepared pair of reference stores, and nothing else.

    A store already holding the proposed bytes is safe rather than a refusal:
    this write is a pure compare-and-swap over frozen bytes, so an exact retry
    after an interrupted apply is the same operation again. That is also why
    :func:`recover_prepared_reference_facts` is this same replay — there is no
    classification an interrupted pair could corrupt, only writes it still owes.
    """
    return _replay_reference_facts(config, preparation)


def recover_prepared_reference_facts(
    config: ProjectConfig, preparation: ReferenceFactsPreparation
) -> ReferenceFactsResult:
    """Finish one interrupted reference write from its preparation alone.

    From the preparation, and from nothing else: a returned value that was
    never persisted is not evidence. Each bound path is re-measured against its
    own before/after pair, the missing write is finished, and a path at neither
    digest refuses. Nothing is looked up again — the facts were fetched once,
    before the preview.
    """
    return _replay_reference_facts(config, preparation)
