"""Promoting a reviewed staging file into the records janki builds from.

No network (IMPLEMENTATION_PLAN rule 6): jpdb is driven through a fake
transport, so the three-outcome reading check is exercised for real rather
than stubbed at the decision.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from japanese_anki import cli, promote
from japanese_anki.jpdb import JpdbClient
from japanese_anki.models import SourceReference, VocabularyRecord
from japanese_anki.promote import (
    HOLD_MISSING_READING,
    HOLD_READING_KANJI,
    HOLD_UNKNOWN_READING,
    check_readings,
    remint,
)
from japanese_anki.staging import read_staging, write_staging

# One vocabulary row, in the order /parse answers its default fields in.
HANASU = [1562350, 4280520068, "話す", "はなす", ["LHLL"], 200, ["vt", "v5s"]]
ICHINICHI = [1579110, 111, "一日", "いちにち", ["LHHH"], 900, ["n"]]


class FakeJpdb:
    """Answers /parse and lookup-vocabulary from canned dictionary data."""

    def __init__(
        self,
        parses: dict[str, list[Any]] | None = None,
        senses: dict[tuple[int, int], dict[str, Any]] | None = None,
    ) -> None:
        self.parses = parses or {}
        self.senses = senses or {}

    def __call__(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> Any:
        endpoint = url.rsplit("/api/v1/", 1)[-1]
        if endpoint == "parse":
            text = body["text"][0]
            row = self.parses.get(text)
            if row is None:
                # jpdb resolved nothing — a real answer, not a failure.
                return 200, {"tokens": [[]], "vocabulary": []}
            return 200, {"tokens": [[[0, None]]], "vocabulary": [row]}
        if endpoint == "lookup-vocabulary":
            fields = body["fields"]
            rows = []
            for vid, sid in body["list"]:
                sense = self.senses.get((vid, sid), {})
                rows.append([sense.get(name) for name in fields])
            return 200, {"vocabulary_info": rows}
        raise AssertionError(f"unexpected request to {url}")


def client_for(api: FakeJpdb) -> JpdbClient:
    return JpdbClient("test-key", api, sleep=lambda _s: None, jitter=lambda: 0.0)


def hanasu_jpdb() -> FakeJpdb:
    return FakeJpdb(
        {"話す": HANASU}, {(1562350, 4280520068): {"reading": "はなす", "alt_sids": []}}
    )


def homograph_jpdb() -> FakeJpdb:
    """一日 parses as いちにち; ついたち is a real alternate sense."""
    return FakeJpdb(
        {"一日": ICHINICHI},
        {
            (1579110, 111): {"reading": "いちにち", "alt_sids": [222]},
            (1579110, 222): {"reading": "ついたち", "alt_sids": [111]},
        },
    )


def record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "source": SourceReference(type="extract", imported_from="lesson.pdf"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def held_reason(item: VocabularyRecord) -> str:
    return item.source.raw_fields["hold_reason"]


# --- the reading check -------------------------------------------------------


def test_a_primary_reading_passes_without_comment() -> None:
    result = check_readings([record()], client=client_for(hanasu_jpdb()))

    assert [item.id for item in result.promoted] == ["word:話す:はなす"]
    assert result.held == []
    assert result.warnings == []


def test_a_real_alternate_reading_passes_with_a_warning() -> None:
    # A homograph is a real thing and the reviewer chose it; jpdb's preference
    # is not evidence they were wrong.
    tsuitachi = record(id="word:一日:ついたち", expression="一日", reading="ついたち")

    result = check_readings([tsuitachi], client=client_for(homograph_jpdb()))

    assert [item.id for item in result.promoted] == ["word:一日:ついたち"]
    assert "homograph" in result.warnings[0]


def test_a_reading_no_entry_lists_is_held_back() -> None:
    # Far likelier a transcription slip than a discovery, and the reading is
    # half of an ID that cannot be corrected later.
    typo = record(id="word:話す:はなし", reading="はなし")

    result = check_readings([typo], client=client_for(hanasu_jpdb()))

    assert result.promoted == []
    assert held_reason(result.held[0]) == HOLD_UNKNOWN_READING
    assert result.keep == [True]
    assert "はなす" in result.warnings[0]


def test_a_spelling_jpdb_cannot_resolve_is_promoted_unchecked() -> None:
    # Silence is not disagreement. Holding these back would punish exactly the
    # uncommon words a textbook is most worth extracting.
    obscure = record(id="word:黌:こう", expression="黌", reading="こう")

    result = check_readings([obscure], client=client_for(FakeJpdb()))

    assert [item.id for item in result.promoted] == ["word:黌:こう"]
    assert "could not be checked" in result.warnings[0]


def test_the_check_can_be_skipped_offline() -> None:
    typo = record(id="word:話す:はなし", reading="はなし")

    result = check_readings([typo], skip_reading_check=True)

    assert [item.id for item in result.promoted] == ["word:話す:はなし"]


def test_the_check_needs_a_client_unless_skipped() -> None:
    with pytest.raises(promote.PromoteError) as excinfo:
        check_readings([record()])

    assert "--skip-reading-check" in str(excinfo.value)


# --- the structural holds ----------------------------------------------------


def test_a_missing_reading_is_held_whatever_else_is_right() -> None:
    result = check_readings([record(id="word:話す:", reading="")], skip_reading_check=True)

    assert result.promoted == []
    assert held_reason(result.held[0]) == HOLD_MISSING_READING


def test_a_reading_written_in_kanji_is_held_too() -> None:
    result = check_readings(
        [record(id="word:話す:話す", reading="話す")], skip_reading_check=True
    )

    assert result.promoted == []
    assert held_reason(result.held[0]) == HOLD_READING_KANJI


def test_skipping_the_dictionary_check_does_not_skip_the_kana_rule() -> None:
    # That rule is about whether an ID can exist at all, not about whether a
    # dictionary agrees, so no flag turns it off.
    result = check_readings(
        [record(id="word:話す:", reading=""), record(id="word:話す:話す", reading="話す")],
        skip_reading_check=True,
    )

    assert result.promoted == []
    assert len(result.held) == 2


# --- the sanctioned ID re-mint ----------------------------------------------


@pytest.mark.parametrize("stale", ["word:話す:", "word:話す:話す", "word:話す:はなし"])
def test_a_malformed_id_is_re_minted_from_expression_and_reading(stale: str) -> None:
    # Keyed off the id, not the shape of the reading: by promote time a human
    # has replaced the kanji reading with kana, so contains_kanji is False in
    # exactly the case the re-mint exists for.
    assert remint(record(id=stale)).id == "word:話す:はなす"


def test_a_correct_id_is_left_alone() -> None:
    original = record()

    assert remint(original) is original


def test_a_reviewed_row_keeps_its_old_id_only_until_promote() -> None:
    # The reviewer supplied the reading but left the malformed id in place.
    reviewed = record(id="word:話す:話す", reading="はなす")

    result = check_readings([reviewed], client=client_for(hanasu_jpdb()))

    assert [item.id for item in result.promoted] == ["word:話す:はなす"]
    assert result.reminted == {"word:話す:話す": "word:話す:はなす"}


# --- the CLI -----------------------------------------------------------------


def project(tmp_path: Path, records: list[VocabularyRecord] | None = None) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    if records is not None:
        (tmp_path / "vocabulary.json").write_text(
            json.dumps([item.to_dict() for item in records], ensure_ascii=False),
            encoding="utf-8",
        )
    return tmp_path


def staging_file(root: Path, text: str, name: str = "lesson.pdf.yaml") -> Path:
    path = root / "staging" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def patch_jpdb(monkeypatch: pytest.MonkeyPatch, api: FakeJpdb) -> None:
    monkeypatch.setenv("JPDB_API_KEY", "test-key")
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda key, *a, **kw: client_for(api))


def stored(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload}


ONE_GOOD = """\
source_file: lesson.pdf
records:
  - id: 'word:話す:はなす'
    expression: 話す
    reading: はなす
    meanings: [to speak]
    source:
      type: extract
      imported_from: lesson.pdf
"""


def test_promote_lands_records_the_archive_and_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path = staging_file(root, ONE_GOOD)
    patch_jpdb(monkeypatch, hanasu_jpdb())

    code = cli.main(["--root", str(root), "promote", str(path)])

    assert code == 0
    assert "word:話す:はなす" in stored(root)
    # The staging file is finished, so it is gone and archived.
    assert not path.exists()
    archived, _ = read_staging(root / "staging" / "done" / "lesson.pdf.yaml")
    assert [item.id for item in archived] == ["word:話す:はなす"]
    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    entry = book["records"]["word:話す:はなす"]
    assert [source["type"] for source in entry["sources"]] == ["extract"]
    assert "Promoted 1 record(s)" in capsys.readouterr().out


def test_the_ledger_reference_is_the_records_own_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same contract every writer honours: status --rebuild reconstructs the
    # reference from the record's own source, and a mismatch doubles it.
    root = project(tmp_path, [])
    patch_jpdb(monkeypatch, hanasu_jpdb())

    cli.main(["--root", str(root), "promote", str(staging_file(root, ONE_GOOD))])

    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    source = book["records"]["word:話す:はなす"]["sources"][0]
    assert (source["type"], source["ref"]) == ("extract", "lesson.pdf")


def test_held_rows_stay_in_the_file_with_their_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path = staging_file(
        root,
        ONE_GOOD
        + """\
  - id: 'word:食べ物:'
    expression: 食べ物
    reading: ''
    source:
      type: extract
      imported_from: lesson.pdf
""",
    )
    patch_jpdb(monkeypatch, hanasu_jpdb())

    cli.main(["--root", str(root), "promote", str(path)])

    assert "word:話す:はなす" in stored(root)
    assert "word:食べ物:" not in stored(root)
    survivors, _ = read_staging(path)
    assert [item.expression for item in survivors] == ["食べ物"]
    assert "1 row(s) still held back" in capsys.readouterr().out


def test_a_reviewers_notes_survive_the_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The staging file holds a review; pruning the promoted rows out of it must
    # not re-render the ones that stay. A comment on its own line *between*
    # rows is the documented exception — YAML attaches it to the row above, so
    # it goes when that row does.
    root = project(tmp_path, [])
    path = staging_file(
        root,
        """\
# checked against the textbook on 2026-08-07
source_file: lesson.pdf
records:
  - id: 'word:話す:はなす'
    expression: 話す
    reading: はなす
    source: {type: extract, imported_from: lesson.pdf}
  - id: 'word:食べ物:'
    expression: 食べ物
    reading: ''  # inline note kept with the row
    my_note: unresolved
    source: {type: extract, imported_from: lesson.pdf}
""",
    )
    patch_jpdb(monkeypatch, hanasu_jpdb())

    cli.main(["--root", str(root), "promote", str(path)])

    text = path.read_text(encoding="utf-8")
    assert "# checked against the textbook" in text
    assert "my_note: unresolved" in text
    assert "# inline note kept with the row" in text
    assert "話す" not in text


def test_re_promoting_a_half_done_file_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reviewer fills in the held row and runs it again; the first pass's
    # archived rows must not be lost.
    root = project(tmp_path, [])
    path = staging_file(
        root,
        ONE_GOOD
        + """\
  - id: 'word:食べ物:'
    expression: 食べ物
    reading: ''
    source:
      type: extract
      imported_from: lesson.pdf
""",
    )
    patch_jpdb(monkeypatch, FakeJpdb())
    cli.main(["--root", str(root), "promote", str(path), "--skip-reading-check"])

    text = path.read_text(encoding="utf-8").replace("reading: ''", "reading: たべもの")
    path.write_text(text, encoding="utf-8")
    cli.main(["--root", str(root), "promote", str(path), "--skip-reading-check"])

    assert set(stored(root)) == {"word:話す:はなす", "word:食べ物:たべもの"}
    archived, _ = read_staging(root / "staging" / "done" / "lesson.pdf.yaml")
    assert {item.id for item in archived} == {"word:話す:はなす", "word:食べ物:たべもの"}
    assert not path.exists()


def test_an_enrichment_shaped_file_merges_as_updates_not_adds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # M4.2's staging files hold records that already exist; promoting them
    # fills empty fields rather than adding rows.
    existing = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        source=SourceReference(type="extract"),
    )
    root = project(tmp_path, [existing])
    path = staging_file(
        root,
        """\
source_file: enrichment
records:
  - id: 'word:話す:はなす'
    expression: 話す
    reading: はなす
    part_of_speech: verb
    source: {type: extract, imported_from: lesson.pdf}
""",
    )
    patch_jpdb(monkeypatch, hanasu_jpdb())

    cli.main(["--root", str(root), "promote", str(path)])

    assert stored(root)["word:話す:はなす"]["part_of_speech"] == "verb"
    out = capsys.readouterr().out
    assert "0 added" in out and "1 filled" in out
    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    assert book["records"]["word:話す:はなす"]["added_at"]


def test_the_id_re_mint_is_reported_and_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path = staging_file(
        root,
        """\
source_file: lesson.pdf
records:
  - id: 'word:話す:話す'
    expression: 話す
    reading: はなす
    source: {type: extract, imported_from: lesson.pdf}
""",
    )
    patch_jpdb(monkeypatch, hanasu_jpdb())

    cli.main(["--root", str(root), "promote", str(path)])

    assert "word:話す:はなす" in stored(root)
    assert "word:話す:話す" not in stored(root)
    assert "word:話す:話す -> word:話す:はなす" in capsys.readouterr().out


def test_a_file_where_nothing_passes_still_records_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path = staging_file(
        root,
        """\
source_file: lesson.pdf
records:
  - id: 'word:食べ物:'
    expression: 食べ物
    reading: ''
    source: {type: extract, imported_from: lesson.pdf}
""",
    )
    patch_jpdb(monkeypatch, hanasu_jpdb())

    code = cli.main(["--root", str(root), "promote", str(path)])

    assert code == 0
    assert not (root / "vocabulary.json").exists() or stored(root) == {}
    survivors, _ = read_staging(path)
    assert held_reason(survivors[0]) == HOLD_MISSING_READING
    assert "Nothing promoted" in capsys.readouterr().out


def test_an_empty_staging_file_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path = staging_file(root, "records: []\n")

    assert cli.main(["--root", str(root), "promote", str(path)]) == 0

    assert "no records" in capsys.readouterr().out
    assert path.exists()


def test_skip_reading_check_never_calls_jpdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [])

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("promote built a jpdb client despite --skip-reading-check")

    monkeypatch.setattr(cli.jpdb, "JpdbClient", explode)

    code = cli.main(
        [
            "--root",
            str(root),
            "promote",
            str(staging_file(root, ONE_GOOD)),
            "--skip-reading-check",
        ]
    )

    assert code == 0


def test_the_archive_records_where_the_rows_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [])
    patch_jpdb(monkeypatch, hanasu_jpdb())

    cli.main(["--root", str(root), "promote", str(staging_file(root, ONE_GOOD))])

    archived = yaml.safe_load(
        (root / "staging" / "done" / "lesson.pdf.yaml").read_text(encoding="utf-8")
    )
    assert archived["source_file"] == "lesson.pdf"
    assert "Promoted 1 record(s)" in archived["review_notes"]


# --- what the file says after a partial promote ------------------------------


def test_a_held_row_is_rewritten_with_the_reason_promote_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HOLD_UNKNOWN_READING is a verdict only promote can reach — it needs the
    # jpdb cross-check. `status --staged` reads the reason off the file, not
    # from this run's scrollback, so it has to be written down.
    root = project(tmp_path, [])
    path = staging_file(
        root,
        ONE_GOOD
        + """\
  - id: 'word:話す:はなし'
    expression: 話す
    reading: はなし
    source: {type: extract, imported_from: lesson.pdf}
""",
    )
    patch_jpdb(monkeypatch, hanasu_jpdb())

    cli.main(["--root", str(root), "promote", str(path)])

    survivors, _ = read_staging(path)
    assert [item.reading for item in survivors] == ["はなし"]
    assert held_reason(survivors[0]) == HOLD_UNKNOWN_READING


def test_a_promoted_record_carries_no_review_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reviewer fixes a hold by typing the reading in, not by tidying the
    # annotations. Left in place they would land in vocabulary.json, where a
    # merge keeps the first record's source forever and every reader that
    # treats hold_reason as "still held" would go on believing it.
    root = project(tmp_path, [])
    path = staging_file(
        root,
        """\
source_file: lesson.pdf
records:
  - id: 'word:食べ物:'
    expression: 食べ物
    reading: たべもの
    source:
      type: extract
      imported_from: lesson.pdf
      raw_fields:
        hold_reason: missing reading
        suggested_reading: たべもの
""",
    )
    patch_jpdb(monkeypatch, FakeJpdb())

    cli.main(["--root", str(root), "promote", str(path)])

    fields = stored(root)["word:食べ物:たべもの"]["source"]["raw_fields"]
    assert "hold_reason" not in fields
    assert "suggested_reading" not in fields


def test_a_file_that_cannot_be_rewritten_is_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A duplicate key is an easy slip while hand-editing. PyYAML accepts it
    # silently, so the whole promote would land and only the final rewrite
    # would fail — leaving the promoted rows in the file, so the re-run
    # appends them to the archive a second time.
    root = project(tmp_path, [])
    duplicated = ONE_GOOD.replace(
        "    reading: はなす", "    reading: はなす\n    reading: はなす"
    )
    path = staging_file(root, duplicated)
    patch_jpdb(monkeypatch, hanasu_jpdb())

    code = cli.main(["--root", str(root), "promote", str(path)])

    assert code == 1
    assert not (root / "vocabulary.json").exists() or stored(root) == {}
    assert not (root / "staging" / "done").exists()
    assert not (root / "ledger.json").exists()


def test_promoting_the_archive_itself_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A tab-completion slip. The archive is valid staging YAML, so it would
    # promote cleanly, double itself in place, and then die on a length
    # mismatch that names no cause.
    root = project(tmp_path, [])
    patch_jpdb(monkeypatch, hanasu_jpdb())
    cli.main(["--root", str(root), "promote", str(staging_file(root, ONE_GOOD))])
    done = root / "staging" / "done" / "lesson.pdf.yaml"
    before = done.read_text(encoding="utf-8")

    code = cli.main(["--root", str(root), "promote", str(done)])

    assert code == 1
    assert done.read_text(encoding="utf-8") == before
    assert "already in the collection" in capsys.readouterr().err


def test_the_archive_keeps_a_hand_written_review_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On a fully promoted file the source is deleted, so the archive is the
    # only copy left of a note the reviewer wrote by hand.
    root = project(tmp_path, [])
    path = staging_file(root, "review_notes: chapter 3, checked with the teacher\n" + ONE_GOOD)
    patch_jpdb(monkeypatch, hanasu_jpdb())

    cli.main(["--root", str(root), "promote", str(path)])

    archived = yaml.safe_load(
        (root / "staging" / "done" / "lesson.pdf.yaml").read_text(encoding="utf-8")
    )
    assert "chapter 3, checked with the teacher" in archived["review_notes"]
    assert "Promoted 1 record(s)" in archived["review_notes"]


def test_the_archive_guard_is_not_fooled_by_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # macOS filesystems are case-insensitive, so staging/Done/lesson.yaml opens
    # the real archive while comparing unequal to staging/done/... — and
    # promoting it doubles a committed file that is the only copy of a
    # finished review.
    root = project(tmp_path, [])
    patch_jpdb(monkeypatch, hanasu_jpdb())
    cli.main(["--root", str(root), "promote", str(staging_file(root, ONE_GOOD))])
    done = root / "staging" / "done" / "lesson.pdf.yaml"
    before = done.read_text(encoding="utf-8")
    mixed_case = root / "staging" / "Done" / "lesson.pdf.yaml"
    if not mixed_case.exists():
        pytest.skip("case-sensitive filesystem: the lexical guard already covers it")

    code = cli.main(["--root", str(root), "promote", str(mixed_case)])

    assert code == 1
    assert done.read_text(encoding="utf-8") == before


def test_a_staging_file_the_archive_could_not_be_written_as_is_refused_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # read_staging accepts .json and JSON is valid YAML, so this got all the
    # way to the archive write before failing — after the records and ledger
    # landed, leaving a review that could never be finished however often it
    # was retried.
    root = project(tmp_path, [])
    path = root / "staging" / "lesson.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "id": "word:話す:はなす",
                        "expression": "話す",
                        "reading": "はなす",
                        "source": {"type": "extract", "imported_from": "lesson.pdf"},
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    patch_jpdb(monkeypatch, hanasu_jpdb())

    code = cli.main(["--root", str(root), "promote", str(path)])

    assert code == 1
    assert not (root / "vocabulary.json").exists() or stored(root) == {}
    assert not (root / "ledger.json").exists()
    assert not (root / "staging" / "done").exists()
    assert "Rename it" in capsys.readouterr().err


def test_promote_does_not_advertise_a_flag_it_does_not_have(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Promote merges existing-wins with no way to change it, so pointing at
    --prefer-incoming hands the reader a command that exits with "unrecognized
    arguments" — output worse than silence, because it reads as instruction."""
    existing = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        usage_notes="hand written",
        source=SourceReference(type="shirabe", imported_from="export.csv"),
    )
    root = project(tmp_path, [existing])
    incoming = replace(existing, usage_notes="the model's note")
    staged = root / "staging" / "in.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(staged, [incoming], {"source_file": "x", "review_notes": "n"})

    assert cli.main(["--root", str(root), "promote", str(staged), "--skip-reading-check"]) == 0

    out = capsys.readouterr().out
    assert "usage_notes" in out, "the conflict is reported"
    assert "--prefer-incoming" not in out
    assert "resolve these by hand" in out
