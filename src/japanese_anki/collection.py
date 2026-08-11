"""Reading an Anki collection, to say when an import silently did not land.

**Read-only, always, and never against the live file.** Anki holds an exclusive
lock on `collection.anki2` while it runs — verified, not assumed: an ordinary
read-only open answers `database is locked` the moment Anki is up. So the file
is copied to a temporary directory and the *copy* is opened, which works whether
Anki is running or not and cannot touch what the user is studying. Nothing here
writes, and janki never will through this module: the collection is Anki's, and
a tool that reads it to file a report has no business editing it.

Standard library only. The `anki` package would give a full API and costs a
large version-coupled dependency for what is three SQL queries; AnkiConnect
would give live access and costs an add-on install plus a running Anki. Neither
is worth it to answer a question with a yes-or-no answer. If janki ever needs to
*write*, that trade changes — see M5.8 in docs/IMPLEMENTATION_PLAN.md.

What it is looking for is the failure `docs/NOTETYPE_UPGRADE.md` documents:
importing a deck whose notetype gained a field, with **Merge Notetypes** left
off (Anki's default), leaves every existing note on the old notetype and files a
clone beside it named with a trailing `+`. Nothing errors, no card is lost, and
none of the new fields ever reach a note. The user finds out when a diagram they
built never appears.

Two things make that detectable from outside:

* a notetype whose name is another one's name plus `+`, and
* a notetype with fewer fields than :data:`FIELD_NAMES`.

Three wrinkles, all small, all found by trying it:

* Anki registers a custom ``unicase`` collation, so any query ordering by a
  collated column fails with ``no such collation sequence`` until one is
  registered. Ordering happens in Python here instead, and the collation is
  registered anyway because a schema change could put one in a view.
* The collection is WAL-mode, so committed-but-uncheckpointed changes live in
  ``collection.anki2-wal``. Copying the main file alone can read a stale
  snapshot, so the sidecars are copied with it.
* ``notetypes.config`` is protobuf rather than JSON. Nothing here needs it —
  names and field counts are in plain relational tables.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from japanese_anki.errors import JankiError

__all__ = [
    "CollectionError",
    "DeckNote",
    "Notetype",
    "clone_suffix_of",
    "default_anki_root",
    "find_profiles",
    "read_deck_notes",
    "read_notetypes",
]


class CollectionError(JankiError):
    """The collection could not be read. Never raised for "there isn't one"."""


#: What Anki appends to a notetype name when an import brings one whose name
#: matches an existing notetype but whose fields do not.
CLONE_SUFFIX = "+"

#: Files SQLite keeps beside the database. Copied with it so the copy is not a
#: stale snapshot of a WAL-mode collection.
_SIDECARS = ("-wal", "-shm")


@dataclass(frozen=True, slots=True)
class Notetype:
    """One notetype, as a report needs it."""

    id: int
    name: str
    field_count: int
    note_count: int

    @property
    def is_clone(self) -> bool:
        """Does this name look like Anki's import clone of another?

        Stripped before testing, because names carry stray whitespace: a real
        collection here holds ``'（コピー)Simple Model+++++++++++++ '`` — thirteen
        plus signs and then a space — which a bare ``endswith`` misses entirely.
        """
        return self.name.strip().endswith(CLONE_SUFFIX)


@dataclass(frozen=True, slots=True)
class DeckNote:
    """One note in a deck, as an importer needs it.

    ``fields`` is keyed by the notetype's own field names, so a caller maps
    "Front"/"Back" rather than counting positions — a note carries its fields as
    one `\x1f`-joined string, and the names live in another table entirely.
    """

    id: int
    guid: str
    notetype: str
    fields: dict[str, str]
    tags: str


def default_anki_root() -> Path | None:
    """Where Anki keeps its profiles on this platform, if that directory exists.

    ``None`` rather than a guess when it does not: a wrong path reported as a
    finding is worse than silence, and every caller here treats "no collection"
    as an ordinary state rather than an error.
    """
    if sys.platform == "darwin":
        candidate = Path.home() / "Library" / "Application Support" / "Anki2"
    elif os.name == "nt":
        base = os.environ.get("APPDATA")
        candidate = Path(base) / "Anki2" if base else Path.home() / "AppData" / "Roaming" / "Anki2"
    else:
        base = os.environ.get("XDG_DATA_HOME")
        candidate = Path(base) / "Anki2" if base else Path.home() / ".local" / "share" / "Anki2"
    return candidate if candidate.is_dir() else None


def find_profiles(root: Path | None = None) -> dict[str, Path]:
    """``profile name -> collection.anki2``, for every profile that has one.

    Sorted by name so a report reads the same twice. Directories that Anki keeps
    beside the profiles — ``addons21``, ``logs`` — have no collection and drop
    out by that test rather than by being named.
    """
    base = root if root is not None else default_anki_root()
    if base is None or not base.is_dir():
        return {}
    found = {}
    for entry in sorted(base.iterdir()):
        collection = entry / "collection.anki2"
        if entry.is_dir() and collection.is_file():
            found[entry.name] = collection
    return found


def _unicase(left: str, right: str) -> int:
    """Anki's collation, close enough for the ordering this module does.

    Anki's own is Unicode case-folding; this is Python's. They can disagree on
    exotic scripts, which would change a sort order and nothing else — no query
    here compares for equality under it.
    """
    left, right = left.casefold(), right.casefold()
    return (left > right) - (left < right)


def read_notetypes(collection: Path) -> list[Notetype]:
    """Every notetype in the collection, with its field and note counts.

    Copies first. Anki holds an exclusive lock while it runs, so opening the
    real file read-only fails outright with the user's collection open — which
    is exactly when someone would run ``janki status``.
    """
    source = Path(collection)
    if not source.is_file():
        raise CollectionError(f"No Anki collection at {source}")
    with tempfile.TemporaryDirectory(prefix="janki-collection-") as scratch:
        work = Path(scratch) / "collection.anki2"
        try:
            shutil.copy2(source, work)
            for suffix in _SIDECARS:
                sidecar = source.with_name(source.name + suffix)
                if sidecar.is_file():
                    shutil.copy2(sidecar, work.with_name(work.name + suffix))
        except OSError as exc:
            raise CollectionError(f"Could not read {source}: {exc.strerror or exc}") from exc
        return _read_copy(work, source)


def _read_copy(work: Path, source: Path) -> list[Notetype]:
    try:
        # `mode=ro` on a copy is belt and braces: the copy is disposable, and
        # the flag makes "this code cannot write to a collection" checkable by
        # reading one line rather than by auditing every query.
        connection = sqlite3.connect(f"file:{work}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise CollectionError(f"Could not open a copy of {source}: {exc}") from exc
    try:
        connection.create_collation("unicase", _unicase)
        rows = connection.execute(
            """
            select nt.id,
                   nt.name,
                   (select count(*) from fields f where f.ntid = nt.id),
                   (select count(*) from notes n where n.mid = nt.id)
            from notetypes nt
            """
        ).fetchall()
    except sqlite3.Error as exc:
        # The "older format" hypothesis only when that is what the error says.
        # A collection predating the `notetypes` table keeps them as JSON in
        # `col.models` — but `database disk image is malformed` (a torn copy, if
        # Anki checkpoints between the main file and its sidecars) is a
        # different problem entirely, and sending someone with a corrupt
        # collection off to think it is merely old is the opposite of this
        # command's job.
        hint = (
            " This may be an older collection format than janki knows how to "
            "inspect."
            if "no such table" in str(exc)
            else ""
        )
        raise CollectionError(f"Could not read notetypes from {source}: {exc}.{hint}") from exc
    finally:
        connection.close()
    return sorted(
        (Notetype(id=int(i), name=str(n), field_count=int(f), note_count=int(c))
         for i, n, f, c in rows),
        key=lambda notetype: notetype.name.casefold(),
    )


def read_deck_notes(collection: Path, deck: str) -> list[DeckNote]:
    """Every note with at least one card in ``deck``, oldest first.

    Same copy-then-open-read-only path as :func:`read_notetypes`, for the same
    reason: Anki holds an exclusive lock while it runs, and this is a thing
    somebody would run with Anki open.

    Matched on the deck's exact name. A note whose cards sit in two decks is
    returned once — the caller wants the note, not the cards — and a card
    parked in a filtered deck is found through ``odid``, which is where Anki
    keeps its real home while it is borrowed.
    """
    source = Path(collection)
    if not source.is_file():
        raise CollectionError(f"No Anki collection at {source}")
    with tempfile.TemporaryDirectory(prefix="janki-collection-") as scratch:
        work = Path(scratch) / "collection.anki2"
        try:
            shutil.copy2(source, work)
            for suffix in _SIDECARS:
                sidecar = source.with_name(source.name + suffix)
                if sidecar.is_file():
                    shutil.copy2(sidecar, work.with_name(work.name + suffix))
        except OSError as exc:
            raise CollectionError(f"Could not read {source}: {exc.strerror or exc}") from exc
        return _read_deck_copy(work, source, deck)


def _read_deck_copy(work: Path, source: Path, deck: str) -> list[DeckNote]:
    try:
        connection = sqlite3.connect(f"file:{work}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise CollectionError(f"Could not open a copy of {source}: {exc}") from exc
    try:
        # Registered before any query touches a collated index. Without it
        # SQLite cannot plan against Anki's own tables and answers "no query
        # solution" — which reads like a broken query rather than a missing
        # collation, and cost an hour to recognise once already.
        connection.create_collation("unicase", _unicase)
        # Anki separates the components of a nested deck name with \x1f;
        # `Parent::Child` is what a person types and what the UI shows.
        wanted = deck.replace("::", "\x1f")
        deck_ids = [
            int(row[0])
            for row in connection.execute("select id, name from decks").fetchall()
            if str(row[1]) == deck or str(row[1]) == wanted
        ]
        if not deck_ids:
            known = sorted(
                str(name).replace("\x1f", "::")
                for (_, name) in connection.execute("select id, name from decks")
            )
            raise CollectionError(
                f"No deck named {deck!r} in {source}. Known: {', '.join(known)}"
            )
        placeholders = ",".join("?" * len(deck_ids))
        note_ids = [
            int(row[0])
            for row in connection.execute(
                f"select distinct nid from cards where did in ({placeholders})"
                f" or odid in ({placeholders})",
                (*deck_ids, *deck_ids),
            ).fetchall()
        ]
        names: dict[int, list[str]] = {}
        for ntid, name in connection.execute(
            "select ntid, name from fields order by ntid, ord"
        ):
            names.setdefault(int(ntid), []).append(str(name))
        notetypes = {
            int(i): str(n) for i, n in connection.execute("select id, name from notetypes")
        }
        notes = []
        for nid in sorted(note_ids):
            row = connection.execute(
                "select guid, mid, flds, tags from notes where id = ?", (nid,)
            ).fetchone()
            if row is None:  # a card whose note is gone: Anki's own repair territory
                continue
            guid, mid, flds, tags = row
            field_names = names.get(int(mid), [])
            values = str(flds).split("\x1f")
            notes.append(
                DeckNote(
                    id=nid,
                    guid=str(guid),
                    notetype=notetypes.get(int(mid), str(mid)),
                    # Zipped rather than indexed: a notetype whose field list
                    # disagrees with a note's stored values is a real state, and
                    # dropping the extras beats raising over somebody else's deck.
                    fields=dict(zip(field_names, values, strict=False)),
                    tags=str(tags or "").strip(),
                )
            )
        return notes
    except sqlite3.Error as exc:
        raise CollectionError(f"Could not read notes from {source}: {exc}") from exc
    finally:
        connection.close()


def clone_suffix_of(name: str, notetypes: list[Notetype]) -> list[Notetype]:
    """Notetypes that look like ``name`` with one or more ``+`` appended.

    More than one because the suffix accumulates: every import that takes the
    default adds another, so a name can end up carrying a dozen. Matched by
    stripping the suffix rather than by ``startswith``, so ``Basic`` does not
    claim ``Basic (and reversed card)+``.
    """
    wanted = name.strip()
    return [
        notetype
        for notetype in notetypes
        if notetype.is_clone and notetype.name.strip().rstrip(CLONE_SUFFIX) == wanted
    ]
