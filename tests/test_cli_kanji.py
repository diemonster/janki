"""``janki kanji`` — the command, not the lookup.

`tests/test_kanji.py` covers what one character's data turns into. This covers
the loop around it: which characters get asked for, what happens when some
lookups fail, and what the shell sees afterwards. Every test drives
`kanji.urllib_transport`, the same seam `fetch_kanji` reads at call time, so the
real fetch-and-parse path runs and nothing reaches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from japanese_anki import cli, kanji
from japanese_anki.kanji import KANJIAPI, KANJIVG, KanjiError
from japanese_anki.models import VocabularyRecord


def project(tmp_path: Path, records: list[VocabularyRecord]) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'kanji_file = "kanji.json"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([r.to_dict() for r in records], ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


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


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> Transport:
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


# --- when a lookup fails -----------------------------------------------------


def test_one_failure_does_not_cost_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    send = Transport(refuse={"名", "前"})
    monkeypatch.setattr(kanji, "urllib_transport", send)
    root = project(tmp_path, [record("名前", "なまえ")])

    assert cli.main(["--root", str(root), "kanji"]) == 1

    stored = json.loads((root / "kanji.json").read_text(encoding="utf-8"))
    assert stored == {}, "the file is written, and honestly empty"
