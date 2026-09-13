"""Reference facts the reviewed word cards already need.

Contracts §7.9. A word card reads `config.kanji_file` and
`config.jpdb_readings_file` at build time and never fetches, so a newly
promoted verb bringing a character neither store covers ships a card with a
hole. This prepares those two stores — and **only** those two.

It is reference enrichment of existing word cards: it mints no character note,
no ``kanji:`` identity and no character deck, and it changes nothing in the
curated ``data/kanji_notes.json``.

Nothing here reaches the network: both lookups are replaced at the seam the
service reads them through at call time.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import io as janki_io
from japanese_anki import jpdb_kanji, kanji, kanji_notes
from japanese_anki.application import character_notes
from japanese_anki.application.character_notes import (
    CharacterNotesError,
    ReferenceFactsPreparation,
    apply_prepared_reference_facts,
    prepare_reference_facts,
    recover_prepared_reference_facts,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.jpdb_kanji import CharacterReadings, ReadingGroup, ReadingUsage

TEMPLATES = Path(__file__).resolve().parents[1] / "templates" / "japanese-study"


def info(character: str) -> kanji.KanjiInfo:
    return kanji.KanjiInfo(
        character=character,
        stroke_count=13,
        meanings=("talk", "speak"),
        readings=(
            kanji.Reading(kind="on", reading="ワ"),
            kanji.Reading(kind="kun", reading="はな.す"),
        ),
        strokes=("M10,10 L30,10",),
    )


def readings(character: str, *, sha: str = "b" * 64) -> CharacterReadings:
    return CharacterReadings(
        character=character,
        source_url=f"https://jpdb.io/kanji/{character}",
        fetched_at_utc="2026-09-07T00:00:00Z",
        sha256=sha,
        groups=(
            ReadingGroup(
                source_class=jpdb_kanji.COMMON_CELL_CLASS,
                readings=(
                    ReadingUsage(
                        label="はな",
                        href=f"/kanji-reading/{character}/はな",
                        percent_text="(84%)",
                        percent=84,
                        percent_less_than=False,
                    ),
                ),
            ),
        ),
    )


class Sources:
    """Both lookups, answering from a script and recording every request."""

    def __init__(self, *, refuse: frozenset[str] = frozenset()) -> None:
        self.looked_up: list[str] = []
        self.fetched: list[tuple[str, bool, Path]] = []
        self.refuse = set(refuse)

    def fetch_kanji(self, character: str, **_kwargs: object) -> kanji.KanjiInfo:
        self.looked_up.append(character)
        if character in self.refuse:
            raise kanji.KanjiError(f"kanjiapi said 404 for {character}")
        return info(character)

    def fetch_character(
        self, character: str, *, html_cache: Path, refresh: bool = False, **_kw: object
    ) -> CharacterReadings:
        self.fetched.append((character, refresh, Path(html_cache)))
        if character in self.refuse:
            raise jpdb_kanji.KanjiReadingsError(f"the request for {character} failed")
        return readings(character, sha=("c" if refresh else "b") * 64)


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> Sources:
    answers = Sources()
    monkeypatch.setattr(kanji, "fetch_kanji", answers.fetch_kanji)
    monkeypatch.setattr(jpdb_kanji, "fetch_character", answers.fetch_character)
    return answers


def project(tmp_path: Path) -> ProjectConfig:
    root = tmp_path / "repo"
    (root / "data" / "decks").mkdir(parents=True)
    (root / "janki.toml").write_text(
        "[paths]\n"
        f'template_dir = "{TEMPLATES}"\n'
        'kanji_notes_file = "data/kanji_notes.json"\n'
        'kanji_file = "data/kanji.json"\n'
        'jpdb_readings_file = "data/jpdb_readings.json"\n'
        f'jpdb_html_cache = "{tmp_path / "cache"}"\n',
        encoding="utf-8",
    )
    return ProjectConfig.load(root)


def seed_stores(config: ProjectConfig, characters: tuple[str, ...]) -> None:
    """Save both stores for ``characters``, the way an earlier run would have."""
    kanji.save_store(
        config.kanji_file, kanji.KanjiStore(entries={c: info(c) for c in characters})
    )
    jpdb_kanji.save_readings(
        config.jpdb_readings_file, {c: readings(c) for c in characters}
    )


def bound_bytes(config: ProjectConfig) -> dict[str, bytes | None]:
    """Exactly what every file this service could touch holds right now."""
    return {
        str(path): path.read_bytes() if path.exists() else None
        for path in (
            config.kanji_file.resolve(),
            config.jpdb_readings_file.resolve(),
            config.kanji_notes_file.resolve(),
        )
    }


def saved_characters(config: ProjectConfig, label: str) -> list[str]:
    """What a store still holds, read back through the loader that owns it."""
    if label == "kanji reference store":
        return sorted(kanji.load_store(config.kanji_file).entries)
    return sorted(jpdb_kanji.load_readings(config.jpdb_readings_file))


def proposed(preparation: ReferenceFactsPreparation, label: str) -> Any:
    for item in preparation.files:
        if item.label == label:
            return item
    raise AssertionError(f"no {label!r} file")


KANJI_STORE = "kanji reference store"
READING_FACTS = "jpdb reading facts"


def written_bytes(scratch: Path, write: Any, store: Any) -> bytes:
    """Exactly what a store's own writer produces, outside any bound path."""
    write(scratch, store)
    return scratch.read_bytes()


@contextmanager
def replaced_after_the_first_read(
    monkeypatch: pytest.MonkeyPatch, target: Path, substitute: bytes
) -> Iterator[list[Path]]:
    """Let the first bound read of ``target`` see the file, every later one see
    ``substitute``, and the original bytes come back before anything applies.

    An ABA window at the seam every bound read goes through, wherever a caller
    reaches it from. A preparation that hashes one read and decodes another
    gets the substitute, and the apply's compare-and-swap cannot notice: the
    file it measures is byte-for-byte the one that was bound.
    """
    original = target.read_bytes()
    real = janki_io.read_bytes_bound_snapshot
    reads: list[Path] = []

    def counting(path: Path, *args: Any, **kwargs: Any) -> Any:
        answer = real(path, *args, **kwargs)
        if Path(path).resolve() == target.resolve():
            reads.append(Path(path))
            if len(reads) == 1:
                target.write_bytes(substitute)
        return answer

    monkeypatch.setattr(janki_io, "read_bytes_bound_snapshot", counting)
    try:
        yield reads
    finally:
        monkeypatch.setattr(janki_io, "read_bytes_bound_snapshot", real)
        target.write_bytes(original)


# --- preparation looks things up and writes nothing ---------------------------


def test_preparation_fetches_only_the_missing_characters_and_writes_nothing(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    seed_stores(config, ("話", "走"))
    before = bound_bytes(config)

    preparation = prepare_reference_facts(config, ["話", "走", "泳", "買"])

    assert sources.looked_up == ["泳", "買"], "saved facts are reused"
    assert [character for character, _refresh, _cache in sources.fetched] == ["泳", "買"]
    assert preparation.looked_up == ("泳", "買")
    assert preparation.fetched_readings == ("泳", "買")
    assert preparation.missing == ()
    assert bound_bytes(config) == before, "preparation publishes nothing"


def test_preparation_freezes_exactly_what_each_store_writer_would_produce(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    seed_stores(config, ("話",))

    preparation = prepare_reference_facts(config, ["話", "泳"])

    assert [item.label for item in preparation.files] == [KANJI_STORE, READING_FACTS]
    assert proposed(preparation, KANJI_STORE).path == config.kanji_file.resolve()
    assert proposed(preparation, READING_FACTS).path == (
        config.jpdb_readings_file.resolve()
    )
    apply_prepared_reference_facts(config, preparation)
    assert config.kanji_file.read_text(encoding="utf-8") == (
        proposed(preparation, KANJI_STORE).after_text
    )
    assert sorted(kanji.load_store(config.kanji_file).entries) == ["泳", "話"]
    assert sorted(jpdb_kanji.load_readings(config.jpdb_readings_file)) == ["泳", "話"]


def test_a_rerun_over_saved_characters_proposes_no_change_at_all(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    seed_stores(config, ("話", "走"))

    preparation = prepare_reference_facts(config, ["話", "走"])

    assert sources.looked_up == [] and sources.fetched == []
    assert proposed(preparation, KANJI_STORE).after_text is None
    assert proposed(preparation, READING_FACTS).after_text is None
    assert proposed(preparation, KANJI_STORE).changed is False
    assert preparation.changed_files == ()


def test_an_absent_store_is_bound_as_absence_not_as_an_empty_document(
    tmp_path: Path, sources: Sources
) -> None:
    """A missing store and a present ``{}`` are different states, and only the
    second one is a file somebody wrote."""
    config = project(tmp_path)

    missing = prepare_reference_facts(config, ["話"])

    assert proposed(missing, KANJI_STORE).before_sha256 is None
    assert proposed(missing, READING_FACTS).before_sha256 is None

    apply_prepared_reference_facts(config, missing)
    empty = kanji.KanjiStore()
    kanji.save_store(config.kanji_file, empty)
    present = prepare_reference_facts(config, ["話"])

    assert proposed(present, KANJI_STORE).before_sha256 == hashlib.sha256(
        config.kanji_file.read_bytes()
    ).hexdigest()
    assert proposed(present, KANJI_STORE).before_sha256 is not None


def test_readings_refresh_only_when_it_is_asked_for_and_never_the_inventory(
    tmp_path: Path, sources: Sources
) -> None:
    """KANJIDIC facts are reused; only the JPDB reading pages are re-requested,
    and only on an explicit refresh."""
    config = project(tmp_path)
    seed_stores(config, ("話",))

    preparation = prepare_reference_facts(config, ["話"], refresh_readings=True)

    assert sources.looked_up == [], "the character inventory is not re-fetched"
    assert sources.fetched == [("話", True, config.jpdb_html_cache)]
    assert preparation.fetched_readings == ("話",)
    assert preparation.looked_up == ()
    assert proposed(preparation, KANJI_STORE).after_text is None
    assert proposed(preparation, READING_FACTS).after_text is not None
    assert preparation.refresh_readings is True


def test_the_lookup_uses_only_the_private_html_cache(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)

    prepare_reference_facts(config, ["話"])

    assert [cache for _c, _r, cache in sources.fetched] == [config.jpdb_html_cache]
    assert config.jpdb_html_cache not in config.root.resolve().parents
    assert not str(config.jpdb_html_cache).startswith(str(config.root.resolve()))


def test_reference_preparation_mints_no_note_no_identity_and_no_deck(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    curated = {"話": "a curated note nobody asked this to touch"}
    config.kanji_notes_file.write_text(json.dumps(curated), encoding="utf-8")

    preparation = prepare_reference_facts(config, ["話"])
    apply_prepared_reference_facts(config, preparation)

    assert [item.label for item in preparation.files] == [KANJI_STORE, READING_FACTS]
    assert config.kanji_notes_file.read_text(encoding="utf-8") == json.dumps(curated)
    assert list(config.deck_dir.iterdir()) == []
    assert not hasattr(preparation, "notes")
    assert not hasattr(preparation, "deck_path")


# --- the store it binds is the store it proposes from -------------------------


def test_a_kanji_store_replaced_and_restored_mid_preparation_loses_nothing(
    tmp_path: Path, sources: Sources, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved character survives a store that changed and changed back while
    the preparation was reading it.

    The digest apply compares comes from one read. If the entries came from a
    second one, a write that is undone before the apply passes every check —
    the file is byte-identical to what was bound — and a request that only adds
    泳 silently drops the 話 the bound store held.
    """
    config = project(tmp_path)
    seed_stores(config, ("話",))
    original = config.kanji_file.read_bytes()
    substitute = written_bytes(
        tmp_path / "substitute-kanji.json",
        kanji.save_store,
        kanji.KanjiStore(entries={"走": info("走")}),
    )

    with replaced_after_the_first_read(
        monkeypatch, config.kanji_file, substitute
    ) as reads:
        preparation = prepare_reference_facts(config, ["泳"])

    assert reads, "the store was never read, so nothing was substituted"
    assert config.kanji_file.read_bytes() == original, "the window closed again"
    apply_prepared_reference_facts(config, preparation)
    saved = kanji.load_store(config.kanji_file).entries
    assert sorted(saved) == ["泳", "話"]
    assert "走" not in saved, "a store nobody bound reached the proposal"


def test_reading_facts_replaced_and_restored_mid_preparation_lose_nothing(
    tmp_path: Path, sources: Sources, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same window over the other store, which has its own parser."""
    config = project(tmp_path)
    seed_stores(config, ("話",))
    original = config.jpdb_readings_file.read_bytes()
    substitute = written_bytes(
        tmp_path / "substitute-readings.json",
        jpdb_kanji.save_readings,
        {"走": readings("走")},
    )

    with replaced_after_the_first_read(
        monkeypatch, config.jpdb_readings_file, substitute
    ) as reads:
        preparation = prepare_reference_facts(config, ["泳"])

    assert reads, "the facts were never read, so nothing was substituted"
    assert config.jpdb_readings_file.read_bytes() == original
    apply_prepared_reference_facts(config, preparation)
    saved = jpdb_kanji.load_readings(config.jpdb_readings_file)
    assert sorted(saved) == ["泳", "話"]
    assert "走" not in saved, "facts nobody bound reached the proposal"


# --- a fact nobody can supply is disclosed, never guessed ----------------------


def test_an_unavailable_lookup_becomes_an_exact_disclosed_missing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = Sources(refuse=frozenset({"泳"}))
    monkeypatch.setattr(kanji, "fetch_kanji", answers.fetch_kanji)
    monkeypatch.setattr(jpdb_kanji, "fetch_character", answers.fetch_character)
    config = project(tmp_path)

    preparation = prepare_reference_facts(config, ["話", "泳"])

    assert [(item.character, item.store) for item in preparation.missing] == [
        ("泳", KANJI_STORE),
        ("泳", READING_FACTS),
    ]
    assert "kanjiapi said 404 for 泳" in preparation.missing[0].detail
    assert "the request for 泳 failed" in preparation.missing[1].detail
    assert "saved reading facts" not in preparation.missing[1].detail, (
        "there are none to keep"
    )
    assert preparation.looked_up == ("話",)

    apply_prepared_reference_facts(config, preparation)

    assert sorted(kanji.load_store(config.kanji_file).entries) == ["話"]
    assert sorted(jpdb_kanji.load_readings(config.jpdb_readings_file)) == ["話"]


@pytest.mark.parametrize(
    ("label", "unreadable"),
    [
        ("kanji reference store", "{ this is not a store\n"),
        (
            "jpdb reading facts",
            json.dumps(
                {
                    "schema_version": 99,
                    "source": "jpdb",
                    "metric": "jpdb_reported_usage",
                    "characters": {},
                }
            )
            + "\n",
        ),
    ],
    ids=["malformed kanji json", "jpdb store that is not this wire"],
)
def test_a_saved_store_it_cannot_read_refuses_before_any_lookup(
    tmp_path: Path, sources: Sources, label: str, unreadable: str
) -> None:
    """A store janki cannot read stops the preparation, naming the file.

    Both cases are a saved store somebody has to look at: the first is not
    JSON, the second is JSON this module did not write. Neither is a hole to
    fill by fetching, and neither is a file to overwrite with an empty one, so
    nothing is requested and nothing is proposed.
    """
    config = project(tmp_path)
    seed_stores(config, ("話",))
    paths = {
        "kanji reference store": config.kanji_file,
        "jpdb reading facts": config.jpdb_readings_file,
    }
    paths[label].write_text(unreadable, encoding="utf-8")
    before = bound_bytes(config)
    intact = READING_FACTS if label == KANJI_STORE else KANJI_STORE

    with pytest.raises(CharacterNotesError, match=re.escape(str(paths[label]))):
        prepare_reference_facts(config, ["泳"])

    assert sources.looked_up == [] and sources.fetched == [], "nothing was requested"
    assert bound_bytes(config) == before, "a refusal writes nothing"
    assert saved_characters(config, intact) == ["話"], "the other store is untouched"


def test_a_failed_refresh_discloses_that_the_saved_facts_were_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refresh that fails over a character janki already has facts for is a
    different state from having none, and the disclosure says which.

    The saved entry stays exactly as it was — the refusal happens before the
    store is touched — so nothing is proposed and nothing is written. Without
    the qualifier the owner reads a missing-fact line for a character whose
    card will still show the facts saved earlier.
    """
    answers = Sources(refuse=frozenset({"話"}))
    monkeypatch.setattr(kanji, "fetch_kanji", answers.fetch_kanji)
    monkeypatch.setattr(jpdb_kanji, "fetch_character", answers.fetch_character)
    config = project(tmp_path)
    seed_stores(config, ("話",))
    before = bound_bytes(config)

    preparation = prepare_reference_facts(config, ["話"], refresh_readings=True)

    [fact] = preparation.missing
    assert (fact.character, fact.store) == ("話", READING_FACTS)
    assert "the request for 話 failed" in fact.detail
    assert "saved reading facts" in fact.detail and "kept" in fact.detail
    assert preparation.changed_files == ()
    apply_prepared_reference_facts(config, preparation)
    assert bound_bytes(config) == before, "the saved facts are still the saved facts"
    assert saved_characters(config, READING_FACTS) == ["話"]


def test_a_missing_fact_never_becomes_a_guessed_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = Sources(refuse=frozenset({"泳"}))
    monkeypatch.setattr(kanji, "fetch_kanji", answers.fetch_kanji)
    monkeypatch.setattr(jpdb_kanji, "fetch_character", answers.fetch_character)
    config = project(tmp_path)

    preparation = prepare_reference_facts(config, ["泳"])

    assert proposed(preparation, KANJI_STORE).after_text is None
    assert proposed(preparation, READING_FACTS).after_text is None
    assert preparation.missing != ()


def test_a_character_that_is_not_one_kanji_refuses_before_any_lookup(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)

    with pytest.raises(CharacterNotesError):
        prepare_reference_facts(config, ["はな"])
    with pytest.raises(CharacterNotesError):
        prepare_reference_facts(config, "話")
    with pytest.raises(CharacterNotesError):
        prepare_reference_facts(config, [])

    assert sources.looked_up == [] and sources.fetched == []


def test_one_character_named_twice_is_one_lookup(
    tmp_path: Path, sources: Sources
) -> None:
    """`kanji.kanji_in` is applied per record, so a shared character arrives
    more than once over a whole collection."""
    config = project(tmp_path)

    preparation = prepare_reference_facts(config, ["話", "話"])

    assert preparation.characters == ("話",)
    assert sources.looked_up == ["話"]


# --- apply and recovery -------------------------------------------------------


def test_apply_prechecks_both_paths_before_it_writes_either(
    tmp_path: Path, sources: Sources
) -> None:
    """A stale *later* target stops the earlier store's effect.

    The lock and precheck order is by path, so `data/jpdb_readings.json` is
    measured before `data/kanji.json`. Making the second one stale is the case
    that distinguishes "precheck everything, then write" from "check and write
    each in turn" — the first store would already be on disk.
    """
    config = project(tmp_path)
    preparation = prepare_reference_facts(config, ["話"])
    ordered = sorted(str(item.path) for item in preparation.files)
    assert ordered[-1] == str(config.kanji_file.resolve())
    config.kanji_file.write_text("{}\n", encoding="utf-8")
    before = bound_bytes(config)

    with pytest.raises(CharacterNotesError, match="reference-facts-stale"):
        apply_prepared_reference_facts(config, preparation)

    assert bound_bytes(config) == before, "the reading facts were never written"
    assert not config.jpdb_readings_file.exists()


def test_apply_is_idempotent_and_an_already_written_store_is_safe(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    preparation = prepare_reference_facts(config, ["話"])

    first = apply_prepared_reference_facts(config, preparation)
    landed = bound_bytes(config)
    second = apply_prepared_reference_facts(config, preparation)

    assert sorted(first.changed) == sorted([KANJI_STORE, READING_FACTS])
    assert first.already_complete == ()
    assert second.changed == ()
    assert sorted(second.already_complete) == sorted([KANJI_STORE, READING_FACTS])
    assert bound_bytes(config) == landed


def test_a_crash_after_the_first_store_recovers_only_the_second(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    preparation = prepare_reference_facts(config, ["話"])
    first = proposed(preparation, KANJI_STORE)
    first.path.parent.mkdir(parents=True, exist_ok=True)
    first.path.write_text(first.after_text, encoding="utf-8")
    landed = first.path.read_bytes()

    recovered = recover_prepared_reference_facts(config, preparation)

    assert recovered.already_complete == (KANJI_STORE,)
    assert recovered.changed == (READING_FACTS,)
    assert first.path.read_bytes() == landed, "the first store was not written again"
    assert sorted(jpdb_kanji.load_readings(config.jpdb_readings_file)) == ["話"]


def test_recovery_refetches_nothing(tmp_path: Path, sources: Sources) -> None:
    config = project(tmp_path)
    preparation = prepare_reference_facts(config, ["話"])
    asked = (list(sources.looked_up), list(sources.fetched))

    recover_prepared_reference_facts(config, preparation)

    assert (sources.looked_up, sources.fetched) == asked


def test_a_substituted_store_path_refuses_however_familiar_its_bytes(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    preparation = prepare_reference_facts(config, ["話"])
    elsewhere = replace(
        preparation,
        files=(
            replace(
                proposed(preparation, KANJI_STORE),
                path=config.root / "data" / "other-kanji.json",
            ),
            proposed(preparation, READING_FACTS),
        ),
    )
    before = bound_bytes(config)

    with pytest.raises(CharacterNotesError, match="kanji reference store"):
        apply_prepared_reference_facts(config, elsewhere)

    assert bound_bytes(config) == before


def test_a_preparation_from_another_repository_refuses(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    other = project(tmp_path / "second")
    preparation = prepare_reference_facts(config, ["話"])
    before = bound_bytes(other)

    with pytest.raises(CharacterNotesError):
        apply_prepared_reference_facts(other, preparation)

    assert bound_bytes(other) == before


def test_a_preparation_round_trips_and_its_fingerprint_binds_the_payloads(
    tmp_path: Path, sources: Sources
) -> None:
    config = project(tmp_path)
    preparation = prepare_reference_facts(config, ["話"])

    restored = ReferenceFactsPreparation.from_dict(
        json.loads(json.dumps(preparation.to_dict()))
    )

    assert restored == preparation
    assert restored.fingerprint == preparation.fingerprint
    assert len(preparation.fingerprint) == 64
    assert restored.file(KANJI_STORE) == proposed(preparation, KANJI_STORE)
    assert restored.file("curated notes") is None

    with pytest.raises(CharacterNotesError):
        ReferenceFactsPreparation.from_dict(
            {**preparation.to_dict(), "fingerprint": "0" * 64}
        )


def test_a_tampered_payload_refuses_at_apply(tmp_path: Path, sources: Sources) -> None:
    config = project(tmp_path)
    preparation = prepare_reference_facts(config, ["話"])
    tampered = replace(
        preparation,
        files=(
            replace(proposed(preparation, KANJI_STORE), after_text="{}\n"),
            proposed(preparation, READING_FACTS),
        ),
    )
    before = bound_bytes(config)

    with pytest.raises(CharacterNotesError, match="reference-facts-stale"):
        apply_prepared_reference_facts(config, tampered)

    assert bound_bytes(config) == before


def test_the_two_stores_may_not_be_one_file(tmp_path: Path, sources: Sources) -> None:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "janki.toml").write_text(
        "[paths]\n"
        f'template_dir = "{TEMPLATES}"\n'
        'kanji_file = "data/reference.json"\n'
        'jpdb_readings_file = "data/reference.json"\n'
        f'jpdb_html_cache = "{tmp_path / "cache"}"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(root)

    with pytest.raises(CharacterNotesError, match="two different files"):
        prepare_reference_facts(config, ["話"])


# --- the existing character-note batch is unchanged ---------------------------


def test_the_character_note_batch_still_refuses_a_failed_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disclosure above is reference-only. Creating a character note from a
    lookup that failed would be inventing study content, and still refuses."""
    answers = Sources(refuse=frozenset({"泳"}))
    monkeypatch.setattr(kanji, "fetch_kanji", answers.fetch_kanji)
    monkeypatch.setattr(jpdb_kanji, "fetch_character", answers.fetch_character)
    config = project(tmp_path)

    with pytest.raises(CharacterNotesError, match="Could not look up reference data"):
        character_notes.prepare_character_notes(config, ["泳"], deck_name="Kanji")

    assert not config.kanji_notes_file.exists()
    assert kanji_notes.load_notes(config.kanji_notes_file) == {}


def test_the_character_note_batch_preserves_bound_kanji_entries_during_replacement(
    tmp_path: Path, sources: Sources, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The character-note batch also preserves the kanji entries it bound."""
    config = project(tmp_path)
    seed_stores(config, ("話",))
    original = config.kanji_file.read_bytes()
    substitute = written_bytes(
        tmp_path / "substitute-kanji.json",
        kanji.save_store,
        kanji.KanjiStore(entries={"走": info("走")}),
    )

    with replaced_after_the_first_read(
        monkeypatch, config.kanji_file, substitute
    ) as reads:
        plan = character_notes.prepare_character_notes(
            config, ["泳"], deck_name="Kanji"
        )

    assert reads, "the store was never read, so nothing was substituted"
    assert config.kanji_file.read_bytes() == original
    character_notes.execute_character_notes(
        config, plan, expected_fingerprint=plan.fingerprint
    )
    saved = kanji.load_store(config.kanji_file).entries
    assert sorted(saved) == ["泳", "話"]
    assert "走" not in saved
