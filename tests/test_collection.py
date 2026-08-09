"""Reading an Anki collection, and what `janki status` says about it — M5.8.

The fixtures build a collection with plain `sqlite3`, creating only the three
tables this module reads. That keeps the suite free of the `anki` package, which
is a large version-coupled dependency janki deliberately does not take — and it
lets a test build the *broken* state on purpose, which the real library refuses
to do (it will not accept a hand-set notetype id).

Shapes here were taken from a real collection, not invented: the `unicase`
collation, the WAL sidecars, and the trailing space on a clone's name are all
things that were found by reading one.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from japanese_anki import cli, status
from japanese_anki.collection import (
    CollectionError,
    Notetype,
    clone_suffix_of,
    find_profiles,
    read_notetypes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _collection(path: Path, notetypes: list[tuple[int, str, int, int]]) -> Path:
    """A collection holding `(id, name, field count, note count)` notetypes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(
        """
        create table notetypes (id integer primary key, name text not null,
                                mtime_secs integer not null default 0,
                                usn integer not null default 0, config blob);
        create table fields (ntid integer not null, ord integer not null,
                             name text not null, config blob);
        create table notes (id integer primary key, guid text not null,
                            mid integer not null, mod integer not null default 0,
                            usn integer not null default 0, tags text default '',
                            flds text default '', sfld text default '',
                            csum integer default 0, flags integer default 0,
                            data text default '');
        """
    )
    note_id = 1
    for ntid, name, fields, notes in notetypes:
        con.execute("insert into notetypes (id, name, config) values (?, ?, ?)",
                    (ntid, name, b"\x08\x01"))  # config is protobuf; unread here
        for ord_ in range(fields):
            con.execute("insert into fields (ntid, ord, name) values (?, ?, ?)",
                        (ntid, ord_, f"Field{ord_}"))
        for _ in range(notes):
            con.execute("insert into notes (id, guid, mid, mod) values (?, ?, ?, 0)",
                        (note_id, f"g{note_id}", ntid))
            note_id += 1
    con.commit()
    con.close()
    return path


JANKI = "Japanese Study (recognition+production)"


# --- reading ----------------------------------------------------------------


def test_notetypes_come_back_with_their_field_and_note_counts(tmp_path: Path) -> None:
    path = _collection(tmp_path / "collection.anki2", [
        (100, JANKI, 22, 3),
        (200, "Basic", 2, 0),
    ])

    types = read_notetypes(path)

    assert [(t.name, t.field_count, t.note_count) for t in types] == [
        ("Basic", 2, 0), (JANKI, 22, 3),
    ]


def test_the_live_file_is_never_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anki holds an exclusive lock while it runs — an ordinary read-only open
    answers `database is locked` the moment it is up, which is exactly when
    someone runs `janki status`. So the file is copied and the copy is read."""
    path = _collection(tmp_path / "collection.anki2", [(100, JANKI, 22, 1)])
    opened: list[str] = []
    real = sqlite3.connect

    def watch(target: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        opened.append(str(target))
        return real(target, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", watch)

    read_notetypes(path)

    assert opened, "something was opened"
    assert not any(str(path) in target for target in opened), "never the real file"


def test_a_wal_sidecar_is_copied_with_the_database(tmp_path: Path) -> None:
    """A real collection is WAL-mode, so committed-but-uncheckpointed rows live
    in `collection.anki2-wal`. Copying the main file alone reads a stale
    snapshot of a collection the user just changed."""
    import shutil

    path = _collection(tmp_path / "collection.anki2", [(100, JANKI, 22, 1)])
    # The connection stays open: a clean close checkpoints the WAL back into the
    # main file and deletes it, which is exactly the state this is *not* about.
    # Anki holds its collection open the same way.
    con = sqlite3.connect(path)
    con.execute("pragma journal_mode=wal")
    con.execute("insert into notetypes (id, name, config) values (?, ?, ?)",
                (101, JANKI + "+", b""))
    con.commit()          # committed, but living in the -wal until a checkpoint
    try:
        assert path.with_name(path.name + "-wal").exists(), "the sidecar is real"

        # What copying the main file alone would have seen.
        main_only = tmp_path / "main-only.anki2"
        shutil.copy2(path, main_only)
        blind = sqlite3.connect(f"file:{main_only}?mode=ro", uri=True)
        missed = blind.execute("select count(*) from notetypes").fetchone()[0]
        blind.close()

        names = [t.name for t in read_notetypes(path)]

        assert JANKI + "+" in names, "the sidecar's row came across"
        assert missed < len(names), "copying the main file alone would have missed it"
    finally:
        con.close()


def test_a_missing_collection_is_an_error_not_a_crash(tmp_path: Path) -> None:
    with pytest.raises(CollectionError, match="No Anki collection at"):
        read_notetypes(tmp_path / "nowhere.anki2")


def test_an_unreadable_schema_says_so_rather_than_raising_sqlite(tmp_path: Path) -> None:
    """A collection older than the `notetypes` table keeps them as JSON in
    `col.models`. That is "cannot tell", not "janki is broken"."""
    path = tmp_path / "collection.anki2"
    con = sqlite3.connect(path)
    con.execute("create table col (models text)")
    con.commit()
    con.close()

    with pytest.raises(CollectionError, match="older collection format"):
        read_notetypes(path)


# --- what counts as a clone -------------------------------------------------


def test_a_trailing_space_does_not_hide_a_clone() -> None:
    """Found in a real collection: `'（コピー)Simple Model+++++++++++++ '` — thirteen
    plus signs and then a space. A bare `endswith('+')` misses it entirely."""
    assert Notetype(1, "Basic+", 2, 5).is_clone
    assert Notetype(1, "（コピー)Simple Model+++++++++++++ ", 8, 216).is_clone
    assert not Notetype(1, "Basic", 2, 5).is_clone


def test_a_clone_is_matched_to_its_base_and_not_to_a_neighbour() -> None:
    """`startswith` would let `Basic` claim `Basic (and reversed card)+`, which
    would blame the wrong deck for someone else's import."""
    types = [
        Notetype(1, "Basic", 2, 0),
        Notetype(2, "Basic+", 6, 458),
        Notetype(3, "Basic (and reversed card)+", 2, 7),
        Notetype(4, "Basic+++", 6, 1),
    ]

    names = [t.name for t in clone_suffix_of("Basic", types)]

    assert names == ["Basic+", "Basic+++"], "every depth of clone, and only its own"


def test_profiles_skip_directories_with_no_collection(tmp_path: Path) -> None:
    """Anki keeps `addons21` and `logs` beside the profiles; they are excluded
    by having no collection rather than by being named."""
    _collection(tmp_path / "User 1" / "collection.anki2", [(100, JANKI, 22, 1)])
    _collection(tmp_path / "janki-test" / "collection.anki2", [(100, JANKI, 22, 1)])
    (tmp_path / "addons21").mkdir()
    (tmp_path / "logs").mkdir()

    assert sorted(find_profiles(tmp_path)) == ["User 1", "janki-test"]


def test_no_anki_at_all_is_an_empty_answer_not_an_error(tmp_path: Path) -> None:
    assert find_profiles(tmp_path / "nowhere") == {}


# --- the findings -----------------------------------------------------------


def test_a_clone_beside_the_deck_notetype_is_reported(tmp_path: Path) -> None:
    """The failure `docs/NOTETYPE_UPGRADE.md` documents: Merge Notetypes left
    off leaves every note on the old notetype and files a `+` clone beside it.
    Nothing errors and no field ever reaches a card."""
    path = _collection(tmp_path / "collection.anki2", [
        (100, JANKI, 19, 3),
        (101, JANKI + "+", 22, 0),
    ])

    findings, warnings = status.check_collection(path, [("verbs", 100, JANKI, 22)])

    assert warnings == []
    messages = " ".join(f.message for f in findings)
    assert "19 fields where this deck writes 22" in messages
    assert "Merge Notetypes" in messages
    assert all(f.deck_stem == "verbs" for f in findings)


def test_another_decks_clones_are_never_mentioned(tmp_path: Path) -> None:
    """A real collection is full of shared decks with their own `+` clones —
    1,815 notes' worth in the one this was written against. janki did not create
    them and cannot fix them, and listing them would bury the one actionable
    line."""
    path = _collection(tmp_path / "collection.anki2", [
        (100, JANKI, 22, 3),
        (200, "Basic", 2, 0),
        (201, "Basic+", 6, 458),
        (202, "（コピー)Simple Model+++++++++++++ ", 8, 216),
    ])

    findings, warnings = status.check_collection(path, [("verbs", 100, JANKI, 22)])

    assert (findings, warnings) == ([], [])


def test_a_notetype_that_was_never_imported_is_not_a_finding(tmp_path: Path) -> None:
    """A deck built and not yet imported is the ordinary state, not a failure."""
    path = _collection(tmp_path / "collection.anki2", [(200, "Basic", 2, 0)])

    assert status.check_collection(path, [("verbs", 100, JANKI, 22)]) == ([], [])


def test_more_fields_than_the_deck_writes_is_not_a_finding(tmp_path: Path) -> None:
    """A collection ahead of this janki — someone else's newer build, or a hand
    -added field — is not the failure being looked for."""
    path = _collection(tmp_path / "collection.anki2", [(100, JANKI, 24, 3)])

    assert status.check_collection(path, [("verbs", 100, JANKI, 22)]) == ([], [])


def test_an_unreadable_collection_is_a_warning_not_a_finding(tmp_path: Path) -> None:
    """A collection janki cannot read says nothing about the user's decks, and
    must not be presented as though it did."""
    path = tmp_path / "collection.anki2"
    con = sqlite3.connect(path)
    con.execute("create table col (models text)")
    con.commit()
    con.close()

    findings, warnings = status.check_collection(path, [("verbs", 100, JANKI, 22)])

    assert findings == []
    assert len(warnings) == 1 and "older collection format" in warnings[0]


def test_a_renamed_notetype_is_reported_against_its_id(tmp_path: Path) -> None:
    path = _collection(tmp_path / "collection.anki2", [(100, "My Own Name", 22, 3)])

    findings, _ = status.check_collection(path, [("verbs", 100, JANKI, 22)])

    assert len(findings) == 1
    assert "is named 'My Own Name' in Anki" in findings[0].message


# --- choosing a collection --------------------------------------------------


def _config(tmp_path: Path, anki: str = "") -> object:
    from japanese_anki.config import ProjectConfig

    (tmp_path / "janki.toml").write_text(
        f"[anki]\n{anki}\n" if anki else "[project]\nname = 'x'\n", encoding="utf-8"
    )
    return ProjectConfig.load(tmp_path)


def test_a_configured_collection_wins(tmp_path: Path) -> None:
    path = _collection(tmp_path / "elsewhere" / "collection.anki2", [(100, JANKI, 22, 1)])
    config = _config(tmp_path, f'collection = "{path}"')

    assert status.resolve_collection(config) == (path, "")


def test_a_configured_collection_that_is_missing_says_so(tmp_path: Path) -> None:
    config = _config(tmp_path, f'collection = "{tmp_path / "nope.anki2"}"')

    found, note = status.resolve_collection(config)

    assert found is None and "does not exist" in note


def test_several_profiles_are_not_guessed_between(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reporting findings about a collection the user never meant is worse than
    reporting nothing."""
    root = tmp_path / "Anki2"
    _collection(root / "User 1" / "collection.anki2", [(100, JANKI, 19, 3)])
    _collection(root / "second" / "collection.anki2", [(100, JANKI, 22, 3)])
    monkeypatch.setattr(status, "find_profiles", lambda: find_profiles(root))
    config = _config(tmp_path)

    found, note = status.resolve_collection(config)

    assert found is None
    assert "several Anki profiles" in note and "User 1" in note


def test_a_named_profile_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "Anki2"
    _collection(root / "User 1" / "collection.anki2", [(100, JANKI, 19, 3)])
    wanted = _collection(root / "second" / "collection.anki2", [(100, JANKI, 22, 3)])
    monkeypatch.setattr(status, "find_profiles", lambda: find_profiles(root))
    config = _config(tmp_path, 'profile = "second"')

    assert status.resolve_collection(config) == (wanted, "")


def test_an_unknown_profile_lists_the_real_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "Anki2"
    _collection(root / "User 1" / "collection.anki2", [(100, JANKI, 19, 3)])
    monkeypatch.setattr(status, "find_profiles", lambda: find_profiles(root))
    config = _config(tmp_path, 'profile = "typo"')

    found, note = status.resolve_collection(config)

    assert found is None and "Available: User 1" in note


def test_no_anki_installed_says_nothing_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`janki status` is the command people run *because* something is
    confusing. Not having Anki is not a problem it should report."""
    monkeypatch.setattr(status, "find_profiles", dict)
    config = _config(tmp_path)

    assert status.resolve_collection(config) == (None, "")


# --- through the CLI --------------------------------------------------------


def test_status_reports_the_finding_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json
    import shutil

    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'template_dir = "templates/japanese-study"\n'
        'ledger_file = "ledger.json"\n'
        f'[anki]\ncollection = "{tmp_path / "collection.anki2"}"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    shutil.copytree(
        PROJECT_ROOT / "templates" / "japanese-study", tmp_path / "templates" / "japanese-study"
    )
    (tmp_path / "decks").mkdir()
    (tmp_path / "decks" / "verbs.yaml").write_text(
        json.dumps({"deck": {"name": "T", "model_id": 100}, "notes": []}), encoding="utf-8"
    )
    _collection(tmp_path / "collection.anki2", [(100, JANKI, 19, 3), (101, JANKI + "+", 22, 0)])

    assert cli.main(["--root", str(tmp_path), "status"]) == 0

    err = capsys.readouterr().err
    assert "Anki: verbs:" in err
    assert "Merge Notetypes" in err


def test_status_still_works_with_no_collection_configured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "janki.toml").write_text(
        '[paths]\nnormalized_file = "vocabulary.json"\ndeck_dir = "decks"\n', encoding="utf-8"
    )
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    (tmp_path / "decks").mkdir()
    monkeypatch.setattr(status, "find_profiles", dict)

    assert cli.main(["--root", str(tmp_path), "status"]) == 0

    assert "Records: 0" in capsys.readouterr().out
