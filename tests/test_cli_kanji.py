"""``janki kanji`` — the command, not the lookup.

`tests/test_kanji.py` covers what one character's data turns into. This covers
the loop around it: which characters get asked for, what happens when some
lookups fail, and what the shell sees afterwards. Every test drives
`kanji.urllib_transport`, the same seam `fetch_kanji` reads at call time, so the
real fetch-and-parse path runs and nothing reaches the network.

The command asks two sources per character now, so the second is stubbed at
`jpdb_kanji.fetch_character` — one page's markup only makes sense for the one
character it is about, and `tests/test_jpdb_kanji.py` parses real fixtures.
Every project here also points the raw page cache at its own tmp directory: a
test that fell through to the real lookup would otherwise read whatever the
developer's machine had cached, and pass or fail accordingly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from japanese_anki import cli, jpdb_kanji, kanji
from japanese_anki.jpdb_kanji import CharacterReadings, KanjiReadingsError
from japanese_anki.kanji import KANJIAPI, KANJIVG, KanjiError
from japanese_anki.models import VocabularyRecord


def project(tmp_path: Path, records: list[VocabularyRecord]) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'kanji_file = "kanji.json"\n'
        'jpdb_readings_file = "jpdb_readings.json"\n'
        f'jpdb_html_cache = "{tmp_path / "cache"}"\n',
        encoding="utf-8",
    )
    (root / "vocabulary.json").write_text(
        json.dumps([r.to_dict() for r in records], ensure_ascii=False), encoding="utf-8"
    )
    return root


def record(expression: str, reading: str) -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:{expression}:{reading}",
        expression=expression,
        reading=reading,
        meanings=["something"],
    )


class Transport:
    """Answers kanjiapi and KanjiVG, and records every character asked for.

    `refuse` names characters whose *info* call fails — the ordinary shape of a
    partial outage, since that is the request that carries the readings.
    """

    def __init__(self, refuse: set[str] = frozenset()) -> None:
        self.asked: list[str] = []
        self.refuse = set(refuse)

    def __call__(self, url: str) -> bytes:
        if url.startswith(f"{KANJIVG}/"):
            raise KanjiError("no stroke data")
        if "/words/" in url:
            return b"[]"
        if url.startswith(f"{KANJIAPI}/kanji/"):
            character = url.rsplit("/", 1)[-1]
            from urllib.parse import unquote

            character = unquote(character)
            self.asked.append(character)
            if character in self.refuse:
                raise KanjiError(f"kanjiapi said 403 for {character}")
            return json.dumps(
                {"stroke_count": 9, "on_readings": ["ゼン"], "kun_readings": ["まえ"]}
            ).encode("utf-8")
        raise AssertionError(f"unexpected url {url}")


class Provider:
    """jpdb's kanji pages, answering from a script and recording every ask."""

    def __init__(self, refuse: set[str] = frozenset()) -> None:
        self.asked: list[tuple[str, bool]] = []
        self.refuse = set(refuse)

    def __call__(
        self, character: str, *, html_cache: Path, refresh: bool = False, **_kw: object
    ) -> CharacterReadings:
        self.asked.append((character, refresh))
        if character in self.refuse:
            raise KanjiReadingsError(f"request for {character} failed")
        return CharacterReadings(
            character=character,
            source_url=f"https://jpdb.io/kanji/{character}",
            fetched_at_utc="2026-09-07T00:00:00Z",
            sha256=("c" if refresh else "b") * 64,
            groups=(),
        )

    @property
    def characters(self) -> list[str]:
        return [character for character, _refresh in self.asked]


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Provider:
    answers = Provider()
    monkeypatch.setattr(jpdb_kanji, "fetch_character", answers)
    return answers


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch, provider: Provider) -> Transport:
    send = Transport()
    monkeypatch.setattr(kanji, "urllib_transport", send)
    return send


def test_each_character_is_asked_for_once_however_many_words_use_it(
    tmp_path: Path, transport: Transport, capsys: pytest.CaptureFixture[str]
) -> None:
    """前 is the same 前 in 名前 and 前線. Fetching per record would ask for it
    twice and pay twice for one answer."""
    root = project(tmp_path, [record("名前", "なまえ"), record("前線", "ぜんせん")])

    assert cli.main(["--root", str(root), "kanji"]) == 0

    assert transport.asked == ["名", "前", "線"], "once each, in the order met"
    assert "Looked up 3 character(s)" in capsys.readouterr().out


def test_exact_record_ids_use_the_same_bounded_kanji_scope_as_the_workbench(
    tmp_path: Path, transport: Transport
) -> None:
    first = record("名前", "なまえ")
    second = record("学校", "がっこう")
    outside = record("前線", "ぜんせん")
    root = project(tmp_path, [first, second, outside])

    assert cli.main(["--root", str(root), "kanji", first.id, second.id]) == 0

    assert transport.asked == ["名", "前", "学", "校"]


def test_only_the_missing_ones_are_fetched_on_a_second_run(
    tmp_path: Path, transport: Transport, capsys: pytest.CaptureFixture[str]
) -> None:
    """Adding one word to a collection of hundreds must cost one request, not a
    re-download of everything."""
    root = project(tmp_path, [record("名前", "なまえ")])
    assert cli.main(["--root", str(root), "kanji"]) == 0
    transport.asked.clear()

    (root / "vocabulary.json").write_text(
        json.dumps(
            [r.to_dict() for r in (record("名前", "なまえ"), record("前線", "ぜんせん"))],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert cli.main(["--root", str(root), "kanji"]) == 0
    assert transport.asked == ["線"], "名 and 前 were already known"


def test_nothing_missing_says_so_and_asks_for_nothing(
    tmp_path: Path, transport: Transport, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record("名前", "なまえ")])
    assert cli.main(["--root", str(root), "kanji"]) == 0
    transport.asked.clear()

    assert cli.main(["--root", str(root), "kanji"]) == 0

    assert transport.asked == []
    assert "already looked up" in capsys.readouterr().out


def test_refresh_asks_again_for_everything(
    tmp_path: Path, transport: Transport
) -> None:
    """The point of the flag: a fixed upstream record only reaches the cards if
    something re-asks."""
    root = project(tmp_path, [record("名前", "なまえ")])
    assert cli.main(["--root", str(root), "kanji"]) == 0
    transport.asked.clear()

    assert cli.main(["--root", str(root), "kanji", "--refresh"]) == 0

    assert transport.asked == ["名", "前"], "both, though both were on disk"


def test_a_word_with_no_kanji_asks_for_nothing(
    tmp_path: Path, transport: Transport, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record("する", "する")])

    assert cli.main(["--root", str(root), "kanji"]) == 0

    assert transport.asked == []
    assert "already looked up" in capsys.readouterr().out


def test_an_empty_collection_is_not_an_error(
    tmp_path: Path, transport: Transport, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    (root / "kanji.json").write_text("not valid JSON", encoding="utf-8")

    assert cli.main(["--root", str(root), "kanji"]) == 0

    assert "No records to read characters from" in capsys.readouterr().out


# --- the second source -------------------------------------------------------


def test_the_provider_is_asked_for_the_same_characters_and_its_facts_are_saved(
    tmp_path: Path, transport: Transport, provider: Provider,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = project(tmp_path, [record("名前", "なまえ")])

    assert cli.main(["--root", str(root), "kanji"]) == 0

    assert provider.asked == [("名", False), ("前", False)]
    saved = json.loads((root / "jpdb_readings.json").read_text(encoding="utf-8"))
    assert sorted(saved["characters"]) == sorted(["名", "前"])
    assert "Read 2 character(s)' published readings" in capsys.readouterr().out


def test_a_character_kanjidic_already_covers_is_still_asked_of_the_provider(
    tmp_path: Path, transport: Transport, provider: Provider
) -> None:
    """The two stores are separately complete. A collection whose reference
    data predates the provider store has every character in one file and none
    in the other, and it is that character's word card that shows no reported
    readings."""
    root = project(tmp_path, [record("名前", "なまえ")])
    assert cli.main(["--root", str(root), "kanji"]) == 0
    (root / "jpdb_readings.json").unlink()
    transport.asked.clear()
    provider.asked.clear()

    assert cli.main(["--root", str(root), "kanji"]) == 0

    assert transport.asked == [], "KANJIDIC was complete and cost nothing"
    assert provider.characters == ["名", "前"], "jpdb had never been asked"


def test_saved_provider_facts_are_reused_and_only_refresh_asks_again(
    tmp_path: Path, transport: Transport, provider: Provider
) -> None:
    root = project(tmp_path, [record("名前", "なまえ")])
    assert cli.main(["--root", str(root), "kanji"]) == 0
    provider.asked.clear()

    assert cli.main(["--root", str(root), "kanji"]) == 0
    assert provider.asked == [], "saved facts are reused"

    assert cli.main(["--root", str(root), "kanji", "--refresh"]) == 0
    assert provider.asked == [("名", True), ("前", True)]


def test_an_unrelated_saved_character_survives_a_later_run(
    tmp_path: Path, transport: Transport, provider: Provider
) -> None:
    """The merge is additive: a character this scope never named keeps its
    saved facts."""
    root = project(tmp_path, [record("名前", "なまえ")])
    assert cli.main(["--root", str(root), "kanji"]) == 0
    (root / "vocabulary.json").write_text(
        json.dumps([record("学校", "がっこう").to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )

    assert cli.main(["--root", str(root), "kanji"]) == 0

    saved = json.loads((root / "jpdb_readings.json").read_text(encoding="utf-8"))
    assert sorted(saved["characters"]) == sorted(["名", "前", "学", "校"])


def test_a_provider_failure_is_a_non_zero_exit_and_names_the_character(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    refusing = Provider(refuse={"前"})
    monkeypatch.setattr(jpdb_kanji, "fetch_character", refusing)
    root = project(tmp_path, [record("名前", "なまえ")])

    assert cli.main(["--root", str(root), "kanji"]) == 1

    assert "前: request for 前 failed" in capsys.readouterr().err
    saved = json.loads((root / "jpdb_readings.json").read_text(encoding="utf-8"))
    assert list(saved["characters"]) == ["名"], "the one that worked was written"


# --- when a lookup fails -----------------------------------------------------


def test_one_failure_does_not_cost_the_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A character kanjiapi has no entry for must not stop the rest: every other
    card still gets its section."""
    send = Transport(refuse={"前"})
    monkeypatch.setattr(kanji, "urllib_transport", send)
    root = project(tmp_path, [record("名前", "なまえ")])

    cli.main(["--root", str(root), "kanji"])

    stored = json.loads((root / "kanji.json").read_text(encoding="utf-8"))
    assert list(stored) == ["名"], "the one that worked was written"
    assert "403 for 前" in capsys.readouterr().err


def test_a_partial_failure_is_a_non_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: Provider
) -> None:
    """The exit code used to be 1 only when *every* lookup failed. There is no
    retry or backoff here, so one 403 part-way through leaves most characters
    unlooked-up — and a scripted `janki kanji && janki build` carried on and
    shipped cards with no kanji section, on a zero exit and a stderr warning."""
    send = Transport(refuse={"前"})
    monkeypatch.setattr(kanji, "urllib_transport", send)
    root = project(tmp_path, [record("名前", "なまえ")])

    assert cli.main(["--root", str(root), "kanji"]) == 1


def test_a_total_failure_writes_nothing_and_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: Provider
) -> None:
    send = Transport(refuse={"名", "前"})
    monkeypatch.setattr(kanji, "urllib_transport", send)
    root = project(tmp_path, [record("名前", "なまえ")])

    assert cli.main(["--root", str(root), "kanji"]) == 1

    stored = json.loads((root / "kanji.json").read_text(encoding="utf-8"))
    assert stored == {}, "the file is written, and honestly empty"
