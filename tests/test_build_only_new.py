"""`build --only-new` and `janki refresh` — M5.6.

The two commands share one fact: the ledger's export history is what makes an
incremental build correct, so a build that ships a record without recording it
makes that record look new forever, and one that records a record it did not
ship makes it invisible forever. Both directions are pinned here.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import cli

CONFIG = """
[paths]
normalized_file = "vocabulary.json"
deck_dir = "decks"
media_dir = "media"
template_dir = "templates/japanese-study"
dist_dir = "dist"
ledger_file = "ledger.json"

# These tests are about export bookkeeping — which records a build ships and
# what the ledger remembers — not about the correctness gate. Leaving it on
# would make every one of them fail at "no card has been read yet", testing
# `janki review` over and over instead of the thing each name claims.
# `tests/test_review.py` covers the gate itself, on both settings.
[review]
require = false
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _record(expression: str, reading: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"word:{expression}:{reading}",
        "expression": expression,
        "reading": reading,
        "meanings": ["x"],
        **extra,
    }


def _project(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    import shutil

    (tmp_path / "janki.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copytree(
        PROJECT_ROOT / "templates" / "japanese-study",
        tmp_path / "templates" / "japanese-study",
    )
    (tmp_path / "decks").mkdir()
    (tmp_path / "decks" / "verbs.yaml").write_text(
        "deck:\n  name: Test\n  source: ../vocabulary.json\nnotes: []\n",
        encoding="utf-8",
    )
    return tmp_path


def _run(root: Path, *argv: str) -> int:
    return cli.main(["--root", str(root), *argv])


def _notes(apkg: Path) -> int:
    import sqlite3
    import tempfile

    with zipfile.ZipFile(apkg) as package:
        name = (
            "collection.anki21"
            if "collection.anki21" in package.namelist()
            else "collection.anki2"
        )
        db = Path(tempfile.mkdtemp()) / "c.db"
        db.write_bytes(package.read(name))
    con = sqlite3.connect(db)
    count = con.execute("select count(*) from notes").fetchone()[0]
    con.close()
    return count


def _exports(root: Path) -> dict[str, dict[str, str]]:
    payload = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    return {rid: entry.get("exports", {}) for rid, entry in payload["records"].items()}


# --- what --only-new includes ----------------------------------------------


def test_only_new_ships_the_records_no_build_has_shipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_record("橋", "はし"), _record("話す", "はなす")])

    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0

    assert _notes(root / "dist" / "verbs.apkg") == 2
    assert set(_exports(root)) == {"word:橋:はし", "word:話す:はなす"}
    assert all(stems == {"verbs": stems["verbs"]} for stems in _exports(root).values())


def test_the_second_run_ships_only_what_arrived_since(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point: a deck that has shipped 400 records and gained 3 builds
    a package of 3, so importing it is not a 400-note no-op that Anki still has
    to reconcile note by note."""
    root = _project(tmp_path, [_record("橋", "はし")])
    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0
    records = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    records.append(_record("話す", "はなす"))
    (root / "vocabulary.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0

    assert _notes(root / "dist" / "verbs.apkg") == 1, "only the new record"
    assert set(_exports(root)) == {"word:橋:はし", "word:話す:はなす"}


def test_nothing_new_writes_no_package_at_all(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty incremental build must not overwrite the last real package with
    a zero-note one — the file on disk is what the user imports, and a deck
    that silently became empty is worse than one that is out of date."""
    root = _project(tmp_path, [_record("橋", "はし")])
    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0
    before = (root / "dist" / "verbs.apkg").read_bytes()
    capsys.readouterr()

    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0

    out = capsys.readouterr().out
    assert "nothing new to build" in out
    assert (root / "dist" / "verbs.apkg").read_bytes() == before, "left untouched"


def test_a_plain_build_records_its_exports_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Otherwise the first `--only-new` after a full build ships everything
    again: `unexported` is the only thing that knows, and a full build that
    stayed quiet would make every record it shipped look new forever."""
    root = _project(tmp_path, [_record("橋", "はし"), _record("話す", "はなす")])

    assert _run(root, "build", "verbs") == 0

    assert set(_exports(root)) == {"word:橋:はし", "word:話す:はなす"}
    capsys.readouterr()
    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0
    assert "nothing new to build" in capsys.readouterr().out


def test_exports_record_only_what_reached_the_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The opposite failure, and the worse one: a record marked exported that
    the package does not contain is invisible to every later `--only-new`, so
    it never reaches a card and nothing reports it."""
    root = _project(tmp_path, [_record("橋", "はし"), _record("話す", "はなす")])
    # 橋 was shipped by a build on a *different day*, which is what makes this
    # test able to fail: with both builds dated today, a wrongly-recorded export
    # is byte-identical to a correctly-preserved one.
    (root / "ledger.json").write_text(
        json.dumps({
            "version": 1,
            "records": {
                "word:橋:はし": {
                    "added_at": "2026-01-01",
                    "sources": [], "enriched": [], "audio": [],
                    "exports": {"verbs": "2026-01-01"},
                }
            },
        }),
        encoding="utf-8",
    )
    capsys.readouterr()

    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0

    assert _notes(root / "dist" / "verbs.apkg") == 1, "only 話す is in the package"
    # 橋 was not in it, so its entry keeps the date of the build that was.
    assert _export_date(root, "word:橋:はし") == "2026-01-01"
    # Through `_export_date`, because 話す ships with gaps so its value is now a
    # dict — and `dict != "2026-01-01"` is unconditionally true, which made this
    # assertion pass even when the date it guards was wrong.
    assert _export_date(root, "word:話す:はなす") != "2026-01-01"


# --- the gaps it warns about ------------------------------------------------


def test_the_gaps_are_reported_with_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A card with no audio behaves differently from its neighbours for a reason
    that is invisible on the card, so the counts come before the build rather
    than during study."""
    root = _project(
        tmp_path,
        [
            _record("橋", "はし", pitch_accent=["LHL"], audio="audio/x.wav"),
            _record("話す", "はなす"),
            _record("食べる", "たべる"),
        ],
    )
    (root / "media" / "audio").mkdir(parents=True)
    (root / "media" / "audio" / "x.wav").write_bytes(b"clip")

    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0

    err = capsys.readouterr().err
    assert "2 of 3 new records have no word audio" in err
    assert "3 of 3 new records have no example sentence" in err
    assert "2 of 3 new records have no pitch accent" in err


def test_a_no_at_the_prompt_builds_nothing_and_records_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path, [_record("橋", "はし")])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    assert _run(root, "build", "verbs", "--only-new") == 0

    assert not (root / "dist").exists(), "no package"
    assert _exports(root) == {} or all(not e for e in _exports(root).values())
    assert "not built" in capsys.readouterr().out


def test_a_non_tty_proceeds_without_asking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unattended runs ship and say so. `janki refresh` drives this, and a
    pipeline that blocks on an unanswerable question is worse than one that
    builds a card missing its audio and prints the count."""
    root = _project(tmp_path, [_record("橋", "はし")])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    def refuse(_prompt: str) -> str:
        raise AssertionError("a non-TTY run must not prompt")

    monkeypatch.setattr("builtins.input", refuse)

    assert _run(root, "build", "verbs", "--only-new") == 0

    assert _notes(root / "dist" / "verbs.apkg") == 1


def test_a_deck_with_no_gaps_asks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(
        tmp_path,
        [
            _record(
                "橋", "はし",
                pitch_accent=["LHL"],
                audio="audio/x.wav",
                examples=[{"japanese": "橋を渡る。", "english": "Cross the bridge."}],
            )
        ],
    )
    (root / "media" / "audio").mkdir(parents=True)
    (root / "media" / "audio" / "x.wav").write_bytes(b"clip")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)

    def refuse(_prompt: str) -> str:
        raise AssertionError("nothing is missing, so nothing should be asked")

    monkeypatch.setattr("builtins.input", refuse)

    assert _run(root, "build", "verbs", "--only-new") == 0


# --- naming a deck ----------------------------------------------------------


def test_a_bare_deck_name_resolves_under_the_deck_dir(tmp_path: Path) -> None:
    root = _project(tmp_path, [_record("橋", "はし")])

    assert _run(root, "build", "verbs") == 0

    assert (root / "dist" / "verbs.apkg").exists()


def test_a_bare_deck_name_resolves_in_a_nested_deck_directory(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path, [_record("橋", "はし")])
    top_level = root / "decks/verbs.yaml"
    nested = root / "decks/pilots/verbs.yaml"
    nested.parent.mkdir()
    nested.write_text(
        top_level.read_text(encoding="utf-8").replace(
            "../vocabulary.json", "../../vocabulary.json"
        ),
        encoding="utf-8",
    )
    top_level.unlink()

    assert _run(root, "build", "verbs") == 0

    assert (root / "dist" / "verbs.apkg").exists()


def test_a_path_that_exists_wins_over_a_same_named_deck(tmp_path: Path) -> None:
    """A file the user can see and point at is never shadowed by one under
    `deck_dir` that happens to share its name."""
    root = _project(tmp_path, [_record("橋", "はし")])
    (root / "verbs.yaml").write_text(
        "deck:\n  name: Local\n  output: local.apkg\n  source: vocabulary.json\nnotes: []\n",
        encoding="utf-8",
    )

    assert cli.main(["--root", str(root), "build", str(root / "verbs.yaml")]) == 0

    assert (root / "dist" / "local.apkg").exists()
    assert not (root / "dist" / "verbs.apkg").exists()


def test_an_unknown_deck_name_says_where_it_looked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [_record("橋", "はし")])

    assert _run(root, "build", "nouns") == 1

    err = capsys.readouterr().err
    assert "No such deck: nouns" in err
    assert str(root / "decks") in err, "and names the directory it searched"


# --- refresh ----------------------------------------------------------------


def test_refresh_runs_the_stages_in_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order is the whole value: jpdb fills the readings and accents `audio`
    needs to force a pitch, `--ai` writes the examples `audio` then voices, and
    the build ships what those finished. A build before `audio` ships silent
    cards *and marks them exported*, so the next `--only-new` never revisits
    them."""
    # Pin the non-TTY branch. Under `pytest -s` from a terminal
    # `sys.stdin.isatty()` is True and this record has every gap, so the
    # build stage would block on its prompt and hang the suite.
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    root = _project(tmp_path, [_record("橋", "はし")])
    called: list[str] = []

    def fake_enrich(args: Any) -> int:
        called.append("enrich --jpdb" if args.jpdb else "enrich --ai")
        return 0

    def fake_audio(args: Any) -> int:
        called.append(f"audio words={args.words} examples={args.examples}")
        return 0

    monkeypatch.setattr(cli, "command_enrich", fake_enrich)
    monkeypatch.setattr(cli, "command_audio", fake_audio)
    # The jpdb-dependent stages skip without a key, so the order this test is
    # about only exists when one is present. Set rather than inherited: the
    # developer's shell has one and a clean checkout does not, and a test whose
    # subject appears only on one of those is not pinning the subject.
    monkeypatch.setenv("JPDB_API_KEY", "k")

    assert _run(root, "refresh") == 0

    out = capsys.readouterr().out
    assert called == [
        "enrich --jpdb",
        "enrich --ai",
        "audio words=True examples=True",
    ]
    order = ["— jpdb", "— ai", "— audio", "— build"]
    positions = [out.index(stage) for stage in order]
    assert positions == sorted(positions), "each stage needs what the one before produced"
    assert "refresh: 4 stage(s) completed" in out


def test_a_skipped_stage_is_named_not_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pin the non-TTY branch. Under `pytest -s` from a terminal
    # `sys.stdin.isatty()` is True and this record has every gap, so the
    # build stage would block on its prompt and hang the suite.
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    root = _project(tmp_path, [_record("橋", "はし")])

    assert _run(root, "refresh", "--no-jpdb", "--no-ai", "--no-audio") == 0

    out = capsys.readouterr().out
    assert "— jpdb: skipped (--no-jpdb)" in out
    assert "refresh: 1 stage(s) completed: build" in out


def test_a_failing_stage_stops_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every later stage depends on what the failed one was supposed to produce,
    so continuing would build a package from half-enriched records — and mark
    those records exported, which is the state `--only-new` cannot recover
    from."""
    # No isatty pin here: the run stops at the jpdb stage, so the build stage —
    # the only one that prompts — is never reached.
    root = _project(tmp_path, [_record("橋", "はし")])
    monkeypatch.setattr(cli, "command_enrich", lambda args: 3)
    monkeypatch.setenv("JPDB_API_KEY", "k")

    assert _run(root, "refresh") == 3

    captured = capsys.readouterr()
    assert "refresh stopped at 'jpdb' (exit 3)" in captured.err
    assert not (root / "dist").exists(), "nothing was built"
    assert not (root / "ledger.json").exists(), "and nothing was marked exported"


def test_refresh_builds_only_the_named_deck(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path, [_record("橋", "はし")])
    (root / "decks" / "nouns.yaml").write_text(
        "deck:\n  name: Nouns\n  source: ../vocabulary.json\nnotes: []\n",
        encoding="utf-8",
    )

    assert _run(root, "refresh", "--no-jpdb", "--no-ai", "--no-audio", "--deck", "verbs") == 0

    assert (root / "dist" / "verbs.apkg").exists()
    assert not (root / "dist" / "nouns.apkg").exists()


# --- the package is written atomically --------------------------------------


def test_an_interrupted_build_leaves_the_previous_package_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """genanki writes the zip incrementally, so a build that dies partway
    through used to leave a truncated `.apkg` where a good one was — and a
    truncated package does not look broken until Anki refuses it."""
    root = _project(tmp_path, [_record("橋", "はし")])
    assert _run(root, "build", "verbs") == 0
    good = (root / "dist" / "verbs.apkg").read_bytes()

    from japanese_anki.exporters import anki as anki_module

    real = anki_module.genanki.Package.write_to_file

    def die(self: Any, path: str) -> None:
        real(self, path)  # write it, so there is something to leave behind
        raise KeyboardInterrupt("^C during the write")

    monkeypatch.setattr(anki_module.genanki.Package, "write_to_file", die)

    with pytest.raises(KeyboardInterrupt):
        _run(root, "build", "verbs")

    assert (root / "dist" / "verbs.apkg").read_bytes() == good, "the old one survived"
    leftovers = [p.name for p in (root / "dist").iterdir() if p.name != "verbs.apkg"]
    assert leftovers == [], "and no partial file was left behind"


# --- a throwaway package must not consume the deck's new-ness ---------------


def test_an_output_build_does_not_claim_the_deck_shipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--output` is a package to eyeball, or `make gates` proving the exporter
    still runs. Recording those as exports consumes the records' new-ness: the
    next `--only-new` skips them, so they never reach a card and nothing says
    so. This is not hypothetical — `make gates` built the real deck and
    committed the export entries it created."""
    root = _project(tmp_path, [_record("橋", "はし")])

    assert _run(root, "build", "verbs", "--output", str(tmp_path / "throwaway.apkg")) == 0

    assert (tmp_path / "throwaway.apkg").exists(), "the package was still written"
    assert not (root / "ledger.json").exists() or _exports(root) == {}
    capsys.readouterr()
    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0
    assert "nothing new" not in capsys.readouterr().out, "still new, as it should be"


def _gates_recipe() -> list[str]:
    """The `gates:` target's own recipe lines, by target rather than substring.

    The first version of this filtered lines containing "gates", which the
    recipe itself now contains — `dist/.gates-check.apkg` — so it silently
    inspected the `build-sample` target instead and would have failed naming
    the wrong one the day that target was renamed.
    """
    lines = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("gates:"))
    recipe = []
    for line in lines[start + 1:]:
        if not line.startswith("\t") and not line.startswith("    ") and line.strip():
            break
        recipe.append(line)
    return recipe


def test_the_gate_builds_a_throwaway_package(tmp_path: Path) -> None:
    """`gates` builds the real deck in the real project root, and every build
    records exports — so the mandatory pre-commit gate consumed the new-ness of
    every record it touched. `--output` makes it a throwaway, which records
    nothing."""
    # $(JANKI), not "build": the target's own echo line says "sample deck
    # builds" and would match a naive substring.
    build_lines = [ln for ln in _gates_recipe() if "$(JANKI)" in ln and " build " in ln]

    assert build_lines, "the gate still builds a deck"
    for line in build_lines:
        assert "--output" in line, f"gate build must be a throwaway: {line.strip()}"


def test_a_throwaway_build_leaves_the_ledger_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The behaviour the Makefile check stands for, asserted directly rather
    than inferred from a recipe line."""
    root = _project(tmp_path, [_record("橋", "はし")])
    assert _run(root, "build", "verbs") == 0
    before = (root / "ledger.json").read_bytes()
    capsys.readouterr()

    assert _run(root, "build", "verbs", "--output", str(tmp_path / "gate.apkg")) == 0

    assert (root / "ledger.json").read_bytes() == before, "not one byte moved"


# --- an incremental build is still a build ----------------------------------


def test_only_new_refuses_a_broken_deck_even_with_nothing_new(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deck whose already-shipped records are broken is a broken deck. An
    incremental build that exits 0 on one a full build refuses hides that until
    the next full build — and `janki refresh` reports "4 stages completed" on a
    deck that cannot be built at all."""
    root = _project(tmp_path, [_record("橋", "はし")])
    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0
    records = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    records[0]["furigana"] = "橋[はし"  # unbalanced: an error, not a warning
    (root / "vocabulary.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    assert _run(root, "build", "verbs", "--only-new", "--yes") == 1

    # The shipping path's readiness gate reports it (M7.6T): local validation
    # first, on its own, before any review-state answer.
    assert "fails local validation" in capsys.readouterr().err


# --- work the deck will never see -------------------------------------------


def test_audio_added_after_a_record_shipped_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`unexported` asks only whether a stem key exists, so a shipped record is
    invisible to `--only-new` forever — including when a later run gives it the
    clip it was shipped without. Silent, and the pipeline reports success."""
    root = _project(tmp_path, [_record("橋", "はし")])
    (root / "ledger.json").write_text(
        json.dumps({
            "version": 1,
            "records": {
                "word:橋:はし": {
                    "added_at": "2026-01-01",
                    "sources": [], "enriched": [],
                    "audio": [{"file": "janki-x.wav", "of": "word", "provider": "voicevox",
                               "voice": 13, "speed": 0.7, "content_fp": "x",
                               "at": "2026-02-01"}],
                    "exports": {"verbs": "2026-01-01"},
                }
            },
        }),
        encoding="utf-8",
    )
    capsys.readouterr()

    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0

    err = capsys.readouterr().err
    assert "1 record(s) shipped without audio" in err


def test_work_finished_the_same_day_is_not_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dates are days. A warning that fires on every same-day pipeline run —
    which is what `janki refresh` is — is one nobody reads."""
    root = _project(tmp_path, [_record("橋", "はし")])
    (root / "ledger.json").write_text(
        json.dumps({
            "version": 1,
            "records": {
                "word:橋:はし": {
                    "added_at": "2026-01-01", "sources": [], "enriched": [],
                    "audio": [{"file": "janki-x.wav", "of": "word", "provider": "voicevox",
                               "voice": 13, "speed": 0.7, "content_fp": "x",
                               "at": "2026-01-01"}],
                    "exports": {"verbs": "2026-01-01"},
                }
            },
        }),
        encoding="utf-8",
    )
    capsys.readouterr()

    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0

    assert "gained audio" not in capsys.readouterr().err


# --- refresh does not force past the gate -----------------------------------


def test_refresh_still_asks_at_a_terminal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refresh forced `--yes`, so the quick path shipped bare cards *and* marked
    them exported — hiding them from every later `--only-new`, which DESIGN_V2
    calls the worst-case version of "quickly build". Refresh is interactive: the
    enrich stages ahead of it prompt too."""
    root = _project(tmp_path, [_record("橋", "はし")])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    asked: list[str] = []

    def answer(prompt: str) -> str:
        asked.append(prompt)
        return "n"

    monkeypatch.setattr("builtins.input", answer)

    assert _run(root, "refresh", "--no-jpdb", "--no-ai", "--no-audio") == 0

    assert asked, "the gap prompt was reached"
    assert not (root / "dist").exists(), "and declining built nothing"


# --- flag combinations and the save failure ---------------------------------


def test_all_with_output_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--output` names one file. Silently ignoring it while building several
    decks leaves the user looking for a package that was never written."""
    root = _project(tmp_path, [_record("橋", "はし")])

    assert _run(root, "build", "--all", "--output", str(tmp_path / "x.apkg")) == 1

    assert "cannot be combined with --all" in capsys.readouterr().err
    assert not (tmp_path / "x.apkg").exists()


def test_a_failed_ledger_save_says_the_package_was_still_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path, [_record("橋", "はし")])

    def refuse(self: Any) -> None:
        raise cli.ledger.LedgerError("the ledger directory is read-only")

    monkeypatch.setattr(cli.ledger.Ledger, "save", refuse)

    assert _run(root, "build", "verbs") == 1

    captured = capsys.readouterr()
    assert (root / "dist" / "verbs.apkg").exists(), "the package is real"
    assert "The package(s) above were written" in captured.err


def test_a_failed_save_after_building_nothing_claims_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The message asserted packages were written and records would be revisited
    — on a run that wrote no package and had no export entry pending."""
    root = _project(tmp_path, [_record("橋", "はし")])
    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0

    def refuse(self: Any) -> None:
        raise cli.ledger.LedgerError("the ledger directory is read-only")

    monkeypatch.setattr(cli.ledger.Ledger, "save", refuse)
    capsys.readouterr()

    assert _run(root, "build", "verbs", "--only-new", "--yes") == 1

    err = capsys.readouterr().err
    assert "the ledger directory is read-only" in err
    assert "package(s) above were written" not in err


# --- the same-day blind spot, closed by recording what shipped --------------


def test_a_record_shipped_silent_and_voiced_the_same_day_is_still_caught(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dates are days on both sides, so no comparison of them can see this:
    ship at 09:00 with the gap warning, voice at 10:00, and the export date and
    the audio date are the same string. The build records what the record was
    *missing* instead, so "it went out silent and has a clip now" is a fact
    about two states rather than an inference from two dates."""
    root = _project(tmp_path, [_record("橋", "はし")])
    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0
    assert _entry_gaps(root, "word:橋:はし") == ["accent", "audio", "examples"]

    # The same day: give it the clip it shipped without.
    records = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    records[0]["audio"] = "audio/janki-x.wav"
    (root / "vocabulary.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    (root / "media" / "audio").mkdir(parents=True)
    (root / "media" / "audio" / "janki-x.wav").write_bytes(b"clip")
    capsys.readouterr()

    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0

    assert "shipped without audio" in capsys.readouterr().err


def test_a_record_that_shipped_complete_is_never_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The export value stays a bare date when there is nothing to record, so
    no existing ledger changes shape and a complete record cannot trip this."""
    root = _project(
        tmp_path,
        [_record("橋", "はし", pitch_accent=["LHL"], audio="audio/x.wav",
                 examples=[{"japanese": "橋を渡る。", "english": "Cross."}])],
    )
    (root / "media" / "audio").mkdir(parents=True)
    (root / "media" / "audio" / "x.wav").write_bytes(b"clip")
    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0
    capsys.readouterr()

    exports = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    value = exports["records"]["word:橋:はし"]["exports"]["verbs"]
    assert isinstance(value, str), "a plain date, the shape it has always had"

    assert _run(root, "build", "verbs", "--only-new", "--yes") == 0
    assert "shipped without" not in capsys.readouterr().err


def _export_date(root: Path, record_id: str, stem: str = "verbs") -> str:
    """The date out of an export value, whichever of its two forms it is."""
    value = _exports(root)[record_id][stem]
    return value["at"] if isinstance(value, dict) else value


def _entry_gaps(root: Path, record_id: str) -> list[str]:
    payload = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    return sorted(payload["records"][record_id]["exports"]["verbs"]["missing"])


# --- a failing deck must not discard what earlier decks recorded ------------


def test_one_broken_deck_does_not_lose_another_decks_exports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`build --all` sorts its decks, so an abort in a later one used to unwind
    past `_finish_build` and drop everything the earlier ones recorded in
    memory. Export state is reconstructible by nothing, so those records ship
    again on every future --only-new while `status` calls them unexported."""
    root = _project(tmp_path, [_record("橋", "はし"), _record("話す", "はなす")])
    # `nouns` sorts first and is fine; `verbs` carries a record with an error.
    # Inline notes, so `nouns` does not read the file `verbs` is about to
    # choke on — otherwise both decks fail and the test proves nothing.
    (root / "decks" / "nouns.yaml").write_text(
        "deck:\n  name: Nouns\n  output: nouns.apkg\n"
        "notes:\n"
        "  - id: word:本:ほん\n    expression: 本\n    reading: ほん\n"
        "    meanings: [book]\n",
        encoding="utf-8",
    )
    records = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    records[0]["furigana"] = "橋[はし"
    (root / "vocabulary.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )

    assert _run(root, "build", "--all", "--only-new", "--yes") == 1

    assert "fails local validation" in capsys.readouterr().err
    assert (root / "ledger.json").exists(), "the ledger was still saved"
    assert _exports(root).get("word:本:ほん", {}).get("nouns"), (
        "the deck that built before the failure kept its export entry"
    )


def test_a_crash_in_a_later_deck_does_not_lose_an_earlier_one_s_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half, and the one the `finally` is now load-bearing for. A deck
    that refuses is caught per deck so the sweep continues, so only something
    that is *not* a `JankiError` — a bug in the builder, an interrupt at the gap
    prompt — still unwinds through the loop. Export state is reconstructible by
    nothing, so what earlier decks recorded must survive that too."""
    from japanese_anki.exporters import anki as anki_module

    root = _project(tmp_path, [_record("橋", "はし")])
    (root / "decks" / "nouns.yaml").write_text(
        "deck:\n  name: Nouns\n  output: nouns.apkg\n"
        "notes:\n"
        "  - id: word:本:ほん\n    expression: 本\n    reading: ほん\n"
        "    meanings: [book]\n",
        encoding="utf-8",
    )
    real = anki_module.build_deck
    calls: list[int] = []

    def crash(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("a bug in the builder")
        return real(*args, **kwargs)

    monkeypatch.setattr(anki_module, "build_deck", crash)
    monkeypatch.setattr(cli, "build_deck", crash)

    with pytest.raises(RuntimeError):
        _run(root, "build", "--all", "--only-new", "--yes")

    assert (root / "ledger.json").exists(), "the ledger was still saved"
    assert _exports(root).get("word:本:ほん", {}).get("nouns"), (
        "the deck that built before the crash kept its export entry"
    )


# --- an --output build says it recorded nothing -----------------------------


def test_an_output_build_says_it_recorded_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Otherwise `--only-new --output` rebuilds the identical package every day
    — the records never leave "new" — and never says why."""
    root = _project(tmp_path, [_record("橋", "はし")])

    assert _run(root, "build", "verbs", "--only-new", "--yes",
                "--output", str(tmp_path / "x.apkg")) == 0

    assert "not recorded as exported" in capsys.readouterr().err


def test_a_failed_save_after_an_output_build_claims_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package *was* written, but no export entry was pending, so the "next
    --only-new will include those records again" line is false here too."""
    root = _project(tmp_path, [_record("橋", "はし")])

    def refuse(self: Any) -> None:
        raise cli.ledger.LedgerError("read-only")

    monkeypatch.setattr(cli.ledger.Ledger, "save", refuse)

    assert _run(root, "build", "verbs", "--output", str(tmp_path / "x.apkg")) == 1

    err = capsys.readouterr().err
    assert "read-only" in err
    assert "package(s) above were written" not in err


def test_all_returns_the_ledger_failure_rather_than_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `finally` that keeps earlier decks' exports discarded the exit code
    with it, so a failed save exited 0 — and `refresh`, or a cron gating on the
    code, recorded a clean build while the export history was lost."""
    root = _project(tmp_path, [_record("橋", "はし")])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    def refuse(self: Any) -> None:
        raise cli.ledger.LedgerError("the ledger directory is read-only")

    monkeypatch.setattr(cli.ledger.Ledger, "save", refuse)

    assert _run(root, "build", "--all", "--only-new") == 1

    assert "read-only" in capsys.readouterr().err


def test_refresh_reports_a_failed_save_from_the_build_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path, [_record("橋", "はし")])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    def refuse(self: Any) -> None:
        raise cli.ledger.LedgerError("read-only")

    monkeypatch.setattr(cli.ledger.Ledger, "save", refuse)

    assert _run(root, "refresh", "--no-jpdb", "--no-ai", "--no-audio") == 1

    assert "refresh stopped at 'build'" in capsys.readouterr().err
