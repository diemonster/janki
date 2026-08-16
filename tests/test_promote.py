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

from japanese_anki import cli, enrich, extract, promote
from japanese_anki.jpdb import JpdbClient
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.promote import (
    HOLD_MISSING_READING,
    HOLD_READING_KANJI,
    HOLD_UNKNOWN_READING,
    HOLD_UNVERIFIABLE_ID,
    check_readings,
    remint,
)
from japanese_anki.staging import (
    annotate,
    coverage_acceptance_requirements,
    coverage_block_fingerprint,
    read_staging,
    write_staging,
)

# One vocabulary row, in the order /parse answers its default fields in.
HANASU = [1562350, 4280520068, "話す", "はなす", ["LHLL"], 200, ["vt", "v5s"]]
ICHINICHI = [1579110, 111, "一日", "いちにち", ["LHHH"], 900, ["n"]]
BENKYOU = [1512670, 1424808594, "勉強", "べんきょう", ["LHHHHH"], 1000, ["n", "vs"]]


class FakeJpdb:
    """Answers /parse and lookup-vocabulary from canned dictionary data."""

    def __init__(
        self,
        parses: dict[str, list[Any]] | None = None,
        senses: dict[tuple[int, int], dict[str, Any]] | None = None,
    ) -> None:
        self.parses = parses or {}
        self.senses = senses or {}
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> Any:
        self.bodies.append(body)
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


def test_the_reading_check_forces_the_reviewed_reading() -> None:
    api = hanasu_jpdb()

    check_readings([record()], client=client_for(api))

    assert api.bodies[0]["furigana"] == [[[0, 2, "はなす"]]]


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


def test_a_suru_compound_passes_when_jpdb_lists_the_exact_stem_as_vs() -> None:
    studying = record(
        id="word:勉強する:べんきょうする",
        expression="勉強する",
        reading="べんきょうする",
    )
    api = FakeJpdb(
        {"勉強する": BENKYOU},
        {(1512670, 1424808594): {"reading": "べんきょう", "alt_sids": []}},
    )

    result = check_readings([studying], client=client_for(api))

    assert [item.id for item in result.promoted] == [studying.id]
    assert result.held == []


def test_a_suru_suffix_does_not_pass_without_the_vs_dictionary_marker() -> None:
    invented = record(
        id="word:本する:ほんする", expression="本する", reading="ほんする"
    )
    noun = [1, 2, "本", "ほん", ["HL"], 10, ["n"]]
    api = FakeJpdb(
        {"本する": noun},
        {(1, 2): {"reading": "ほん", "alt_sids": []}},
    )

    result = check_readings([invented], client=client_for(api))

    assert result.promoted == []
    assert held_reason(result.held[0]) == HOLD_UNKNOWN_READING


def test_a_suru_compound_does_not_pass_on_a_different_stem_spelling() -> None:
    """The stem jpdb resolved has to be *this* word's stem. jpdb answers a
    single lexeme per spelling and picks it itself, so a near neighbour with the
    same reading and the same `vs` marker is exactly what it returns when its
    split is wrong — and accepting one promotes an identity no dictionary
    confirmed."""
    studying = record(
        id="word:勉強する:べんきょうする",
        expression="勉強する",
        reading="べんきょうする",
    )
    neighbour = [1512670, 1424808594, "勉学", "べんきょう", ["LHHHHH"], 1000, ["n", "vs"]]
    api = FakeJpdb(
        {"勉強する": neighbour},
        {(1512670, 1424808594): {"reading": "べんきょう", "alt_sids": []}},
    )

    result = check_readings([studying], client=client_for(api))

    assert result.promoted == []
    assert held_reason(result.held[0]) == HOLD_UNKNOWN_READING


def test_a_suru_compound_does_not_pass_on_a_different_stem_reading() -> None:
    """The other half of the same identity. 勉強 read べんがく is a different
    word from 勉強 read べんきょう, and only the spelling agrees."""
    studying = record(
        id="word:勉強する:べんきょうする",
        expression="勉強する",
        reading="べんきょうする",
    )
    misread = [1512670, 1424808594, "勉強", "べんがく", ["LHHHHH"], 1000, ["n", "vs"]]
    api = FakeJpdb(
        {"勉強する": misread},
        {(1512670, 1424808594): {"reading": "べんがく", "alt_sids": []}},
    )

    result = check_readings([studying], client=client_for(api))

    assert result.promoted == []
    assert held_reason(result.held[0]) == HOLD_UNKNOWN_READING


def test_a_word_with_no_suru_suffix_is_not_answered_for_by_its_first_character() -> None:
    """The reading check reaches the same allowance, so a word that is not a
    する compound must not be promoted on a stem entry that happens to match its
    opening characters.

    Constructed rather than drawn from a real word: the comparison is against
    the last two characters, so reaching it without the suffix check needs an
    expression of at least three characters whose first character is a word in
    its own right. 図書館 is held because としょかん is not a reading jpdb lists
    for it — without the suffix gate, 図/としょ would answer for it instead.

    Each suffix conjunct is pinned separately on the predicate itself, in
    `tests/test_enrich_jpdb.py`; through this path only the pair is
    observable, because either one alone still refuses."""
    library = record(
        id="word:図書館:としょかん", expression="図書館", reading="としょかん"
    )
    fragment = [1, 2, "図", "としょ", ["LH"], 500, ["n", "vs"]]
    api = FakeJpdb({"図書館": fragment}, {(1, 2): {"reading": "としょ", "alt_sids": []}})

    result = check_readings([library], client=client_for(api))

    assert result.promoted == []
    assert held_reason(result.held[0]) == HOLD_UNKNOWN_READING


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


# --- field-level example acceptance ------------------------------------------


def test_a_typed_acceptance_is_bound_to_the_sentences_the_reviewer_saw() -> None:
    # The acceptance is the reviewer *typing* the sentinel into the row —
    # never inferred from an example being present, because the AI staging
    # route puts model sentences on extract rows too. Promotion binds the
    # sentinel to the accepted sentences' fingerprints, so the durable stamp
    # covers exactly the Japanese the reviewer read.
    from japanese_anki.identifiers import short_fingerprint
    from japanese_anki.models import example_accepted

    sentence = "友達と日本語を話します。"
    reviewed = record(
        examples=[ExampleSentence(japanese=sentence)],
        source=SourceReference(
            type="extract",
            imported_from="lesson.pdf",
            raw_fields={"example_authority": "staging-review"},
        ),
    )

    result = check_readings([reviewed], skip_reading_check=True)

    [promoted] = result.promoted
    assert promoted.source.raw_fields["example_authority"] == short_fingerprint(
        sentence
    )
    assert example_accepted(promoted, promoted.examples[0])
    # The binding is per sentence: text added later carries no coverage.
    assert not example_accepted(promoted, ExampleSentence(japanese="別の文。"))


def test_an_example_without_the_typed_sentinel_is_never_stamped() -> None:
    # The AI staging route's shape: model-generated sentences sitting on an
    # extract-type row. Presence proves nothing about who wrote them, so
    # promotion must not mint acceptance out of it — the sentences promote
    # preserved but unaccepted.
    machine = record(
        examples=[ExampleSentence(japanese="友達と日本語を話します。")]
    )

    result = check_readings([machine], skip_reading_check=True)

    [promoted] = result.promoted
    assert "example_authority" not in promoted.source.raw_fields


def test_a_sentinel_with_no_sentences_accepts_nothing() -> None:
    # Left behind, it would be a standing claim waiting for text nobody
    # reviewed.
    empty = record(
        source=SourceReference(
            type="extract",
            imported_from="lesson.pdf",
            raw_fields={"example_authority": "staging-review"},
        ),
    )

    result = check_readings([empty], skip_reading_check=True)

    [promoted] = result.promoted
    assert "example_authority" not in promoted.source.raw_fields


def test_a_non_extract_row_is_left_exactly_as_typed() -> None:
    # A Shirabe export's example is the user's own data, curated by arrival;
    # the acceptance machinery has nothing to add or bind there.
    imported = record(
        source=SourceReference(type="shirabe", imported_from="export.csv"),
        examples=[ExampleSentence(japanese="友達と日本語を話します。")],
    )

    result = check_readings([imported], skip_reading_check=True)

    [promoted] = result.promoted
    assert "example_authority" not in promoted.source.raw_fields


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


# --- the extraction coverage gate ------------------------------------------


def unmeasured_coverage_with_units() -> dict[str, Any]:
    """A coverage block carrying one real source unit, so the per-unit shape
    checks have something to read."""
    unit = extract.SourceUnit(
        page=1,
        section="vocabulary",
        ordinal=1,
        context="話す　はなす　to speak",
        context_fingerprint=extract.context_fingerprint("話す　はなす　to speak"),
        disposition="candidate",
        reason="",
    )
    result = extract.ExtractionResult(
        candidates=(), source_units=(unit,), model_reported_unit_count=1
    )
    return extract.coverage_block(result, source_sha256="a" * 64, mode="table")


def unmeasured_coverage() -> dict[str, Any]:
    result = extract.ExtractionResult(candidates=(), source_units=(), model_reported_unit_count=0)
    return extract.coverage_block(
        result, source_sha256="a" * 64, mode="table"
    )


def approve_coverage(block: dict[str, Any]) -> None:
    requirements = coverage_acceptance_requirements(block)
    block["approval"] = {
        "authority": "repository-owner",
        "source_fingerprint": block["source_fingerprint"],
        "coverage_block_fingerprint": block["coverage_block_fingerprint"],
        **requirements,
        "reason": "I checked every row against the source page.",
        "approved_at": "2026-08-12",
    }


def extraction_meta(coverage: dict[str, Any], *, mode: str = "table") -> dict[str, Any]:
    return {
        "source_file": "lesson.pdf",
        "coverage": coverage,
        "prompt_provenance": {
            "source_sha256": coverage["source_fingerprint"],
            "mode": mode,
            "provider": "anthropic",
            "model": "test-model",
            "response_schema_version": 2,
            "system_prompt_fingerprint": "b" * 64,
            "style_guide_fingerprint": "c" * 64,
            "user_prompt_fingerprint": "d" * 64,
        },
    }


def test_unresolved_coverage_blocks_promotion_before_any_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    staged = root / "staging" / "blocked.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    coverage = unmeasured_coverage()
    write_staging(staged, [record()], extraction_meta(coverage))
    before = staged.read_bytes()

    code = cli.main(
        ["--root", str(root), "promote", str(staged), "--skip-reading-check"]
    )

    assert code == 1
    assert staged.read_bytes() == before
    assert not (root / "ledger.json").exists()
    assert not (root / "staging" / "done").exists()
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []
    assert "coverage-unresolved" in capsys.readouterr().err


def test_exact_reasoned_coverage_acceptance_survives_in_the_archive(
    tmp_path: Path,
) -> None:
    root = project(tmp_path, [])
    staged = root / "staging" / "accepted.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    coverage = unmeasured_coverage()
    approve_coverage(coverage)
    write_staging(staged, [record()], extraction_meta(coverage))

    code = cli.main(
        ["--root", str(root), "promote", str(staged), "--skip-reading-check"]
    )

    assert code == 0
    _records, meta = read_staging(root / "staging" / "done" / "accepted.yaml")
    assert meta["coverage"]["approval"]["reason"] == (
        "I checked every row against the source page."
    )
    assert meta["coverage"]["coverage_block_fingerprint"] == coverage[
        "coverage_block_fingerprint"
    ]


def test_a_coverage_approval_for_an_old_block_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    staged = root / "staging" / "stale.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    coverage = unmeasured_coverage()
    approve_coverage(coverage)
    coverage["approval"]["source_fingerprint"] = "b" * 64
    write_staging(staged, [record()], extraction_meta(coverage))

    code = cli.main(
        ["--root", str(root), "promote", str(staged), "--skip-reading-check"]
    )

    assert code == 1
    assert "coverage-approval-stale" in capsys.readouterr().err
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "damage,expected",
    [
        (lambda unit: unit.__setitem__("context_fingerprint", "not-a-sha"),
         "context_fingerprint"),
        (lambda unit: unit.__setitem__("disposition", "maybe"), "disposition"),
        (lambda unit: unit.pop("section"), "source_units[0] fields are invalid"),
    ],
    ids=["bad-fingerprint", "bad-disposition", "missing-field"],
)
def test_a_malformed_source_unit_is_named_field_by_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], damage: Any, expected: str
) -> None:
    """`_coverage_fact` is a live promote gate that had no test of its own.

    Measured: disabling its field-set check, its fingerprint check or its
    disposition check each left the whole suite green — before this change and
    after it, so it was already blind and M8.4 narrowed a blind validator. The
    surviving `coverage-facts-stale` test exercises a different function
    (`promote._verify_coverage_facts`), which re-derives the block rather than
    checking the shape of what it reads.

    A staging file is hand-editable YAML by design, so the shape check is what
    stands between a typo in a source unit and a promote that reads it as
    something else.
    """
    root = project(tmp_path, [])
    staged = root / "staging" / "damaged.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    coverage = unmeasured_coverage_with_units()
    damage(coverage["source_units"][0])
    coverage["coverage_block_fingerprint"] = coverage_block_fingerprint(coverage)
    write_staging(staged, [record()], extraction_meta(coverage))

    code = cli.main(
        ["--root", str(root), "promote", str(staged), "--skip-reading-check"]
    )

    assert code == 1
    assert expected in capsys.readouterr().err


def test_a_disposition_entry_missing_a_field_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`_coverage_fact`'s own exact-field-set check.

    `source_units` entries never reach it — they have a separate field check
    one line earlier — so the disposition lists are the only route, and
    deleting `set(value) != fields` was invisible without this. A disposition
    entry short a field is a hand-edit that silently changes which unit the
    list is talking about.
    """
    root = project(tmp_path, [])
    staged = root / "staging" / "damaged-fact.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    coverage = unmeasured_coverage_with_units()
    coverage["candidate_units"][0].pop("ordinal")
    coverage["coverage_block_fingerprint"] = coverage_block_fingerprint(coverage)
    write_staging(staged, [record()], extraction_meta(coverage))

    code = cli.main(
        ["--root", str(root), "promote", str(staged), "--skip-reading-check"]
    )

    assert code == 1
    assert "candidate_units[0] must have exactly" in capsys.readouterr().err


def test_a_hand_edited_coverage_status_cannot_recompute_its_way_past_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The block hash is not the gate; re-deriving the facts is.

    This used to prove that a bare `status: matched` could not bypass the
    approved oracle. M8.4 deleted the oracle, and what survives is the stronger
    half: promote regenerates the coverage block from the source units in the
    file and refuses when the stored one disagrees. Recomputing
    `coverage_block_fingerprint` after the edit — which any careful hand-editor
    would do — does not help, because the fingerprint covers what the editor
    wrote rather than what the units imply.
    """
    root = project(tmp_path, [])
    staged = root / "staging" / "fake-matched.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    coverage = unmeasured_coverage()
    # Schema-valid on its face — `selection` is a real status and `blocking`
    # agrees with it — but the units in the file are a table's, so regenerating
    # the block yields `unmeasured`. The schema alone cannot tell.
    coverage["status"] = "selection"
    coverage["blocking"] = False
    coverage["coverage_block_fingerprint"] = coverage_block_fingerprint(coverage)
    write_staging(staged, [record()], extraction_meta(coverage))

    code = cli.main(
        ["--root", str(root), "promote", str(staged), "--skip-reading-check"]
    )

    assert code == 1
    assert "coverage-facts-stale" in capsys.readouterr().err
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "damage",
    [
        lambda block: block.pop("duplicate_keys"),
        lambda block: block.__setitem__("model_reported_unit_count", True),
    ],
)
def test_an_incomplete_or_mistyped_coverage_block_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    damage: Any,
) -> None:
    root = project(tmp_path, [])
    staged = root / "staging" / "invalid-block.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    coverage = unmeasured_coverage()
    damage(coverage)
    coverage["coverage_block_fingerprint"] = coverage_block_fingerprint(coverage)
    write_staging(staged, [record()], extraction_meta(coverage))

    code = cli.main(
        ["--root", str(root), "promote", str(staged), "--skip-reading-check"]
    )

    assert code == 1
    assert "coverage-block-invalid" in capsys.readouterr().err
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []


def test_owner_approval_cannot_make_inconsistent_coverage_facts_valid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    staged = root / "staging" / "inconsistent.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    coverage = unmeasured_coverage()
    coverage["candidate_units"] = [
        {
            "page": 1,
            "section": "vocabulary",
            "ordinal": 1,
            "context_fingerprint": "e" * 64,
            "disposition": "candidate",
        }
    ]
    coverage["coverage_block_fingerprint"] = coverage_block_fingerprint(coverage)
    approve_coverage(coverage)
    write_staging(staged, [record()], extraction_meta(coverage))

    code = cli.main(
        ["--root", str(root), "promote", str(staged), "--skip-reading-check"]
    )

    assert code == 1
    assert "coverage-facts-stale" in capsys.readouterr().err
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []


def test_an_m7_4_block_without_prompt_provenance_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    staged = root / "staging" / "no-provenance.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    coverage = unmeasured_coverage()
    approve_coverage(coverage)
    write_staging(staged, [record()], {"source_file": "lesson.pdf", "coverage": coverage})

    code = cli.main(
        ["--root", str(root), "promote", str(staged), "--skip-reading-check"]
    )

    assert code == 1
    assert "prompt-provenance-invalid" in capsys.readouterr().err


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


def test_an_identity_conflict_on_promote_still_says_it_is_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Promote re-mints ids from the NFKC-normalized expression and reading, so
    a half-width row lands on a full-width record and they disagree about the
    expression. That is not one more field to settle by hand — it is the two
    copies disagreeing about which word this is, and the id comes from them."""
    existing = VocabularyRecord(
        id="word:ATM:エーティーエム",
        expression="ＡＴＭ",
        reading="エーティーエム",
        meanings=["ATM"],
        source=SourceReference(type="shirabe", imported_from="export.csv"),
    )
    root = project(tmp_path, [existing])
    staged_row = VocabularyRecord(
        id="word:ATM:エーティーエム",
        expression="ATM",
        reading="エーティーエム",
        meanings=["ATM"],
        source=SourceReference(type="extract", imported_from="page.png"),
    )
    staged = root / "staging" / "in.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(staged, [staged_row], {"source_file": "x", "review_notes": "n"})

    assert cli.main(["--root", str(root), "promote", str(staged), "--skip-reading-check"]) == 0

    # Asserted on the conflict line, not on the whole capture: pytest names its
    # tmp_path after the test, promote echoes that path, and a bare
    # `"identity" in out` is therefore satisfied by this test's own name.
    (line,) = [
        item
        for item in capsys.readouterr().out.splitlines()
        if item.startswith("  word:ATM:")
    ]
    assert "expression" in line
    assert "the two copies disagree about which word this is" in line
    assert "--prefer-incoming" not in line


def test_promote_never_re_mints_an_id_the_collection_already_holds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A record whose id was minted from a wrong reading keeps that id when the
    reading is corrected — the id is uncorrectable by design. M4.2's staging
    route then sends such a record back through promote, and re-minting there
    added a second record beside the curated one: the original kept its Anki
    history and never received the change, while the copy carried it."""
    curated = VocabularyRecord(
        id="word:辛い:からい",
        expression="辛い",
        reading="つらい",
        meanings=["painful"],
        source=SourceReference(type="shirabe", imported_from="export.csv"),
    )
    root = project(tmp_path, [curated])
    staged = root / "staging" / "ai.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        staged,
        [replace(curated, usage_notes="written by a model")],
        {"source_file": "vocabulary.json", "model": "m", "review_notes": "n"},
    )

    assert cli.main(["--root", str(root), "promote", str(staged), "--skip-reading-check"]) == 0

    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in stored] == ["word:辛い:からい"], "no second copy"
    assert stored[0]["usage_notes"] == "written by a model", "the change landed on it"
    assert "Re-minted" not in capsys.readouterr().out


def test_a_genuinely_new_row_is_still_re_minted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other side: a held row the collection has never seen still gets its
    id minted from the reading the reviewer supplied. That is the one sanctioned
    ID change, and gating on presence must not take it away."""
    root = project(tmp_path, [])
    staged = root / "staging" / "held.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        staged,
        [
            VocabularyRecord(
                id="word:辛い:辛い",
                expression="辛い",
                reading="からい",
                meanings=["spicy"],
                source=SourceReference(type="shirabe", imported_from="export.csv"),
            )
        ],
        {"source_file": "export.csv", "review_notes": "n"},
    )

    assert cli.main(["--root", str(root), "promote", str(staged), "--skip-reading-check"]) == 0

    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in stored] == ["word:辛い:からい"]
    assert "Re-minted" in capsys.readouterr().out


def test_an_id_that_lives_only_in_a_deck_is_still_the_collection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A record inside a deck YAML has the same stale-id problem and the same
    exported GUID as one in the normalized file. Reading only the normalized
    file would re-mint it just as happily."""
    root = tmp_path
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'deck_dir = "decks"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    (root / "decks").mkdir()
    (root / "decks" / "verbs.yaml").write_text(
        "name: Verbs\n"
        "notes:\n"
        "  - id: word:辛い:からい\n"
        "    expression: 辛い\n"
        "    reading: つらい\n"
        "    meanings: [painful]\n",
        encoding="utf-8",
    )
    staged = root / "staging" / "ai.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        staged,
        [
            VocabularyRecord(
                id="word:辛い:からい",
                expression="辛い",
                reading="つらい",
                meanings=["painful"],
                usage_notes="written by a model",
                source=SourceReference(type="shirabe", imported_from="export.csv"),
            )
        ],
        {"source_file": "vocabulary.json", "model": "m", "review_notes": "n"},
    )

    assert cli.main(["--root", str(root), "promote", str(staged), "--skip-reading-check"]) == 0

    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in stored] == ["word:辛い:からい"]
    assert "Re-minted" not in capsys.readouterr().out


def test_an_unreadable_deck_declines_every_re_mint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Its ids are unknown, so no staged id can be *proved* absent — and a
    re-mint decided on a set janki knows is incomplete is the guess this whole
    gate exists to refuse."""
    root = tmp_path
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'deck_dir = "decks"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    (root / "decks").mkdir()
    (root / "decks" / "broken.yaml").write_text("name: [unclosed\n", encoding="utf-8")
    staged = root / "staging" / "held.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        staged,
        [
            VocabularyRecord(
                id="word:辛い:辛い",
                expression="辛い",
                reading="からい",
                meanings=["spicy"],
                source=SourceReference(type="shirabe", imported_from="export.csv"),
            )
        ],
        {"source_file": "export.csv", "review_notes": "n"},
    )

    assert cli.main(["--root", str(root), "promote", str(staged), "--skip-reading-check"]) == 0

    captured = capsys.readouterr()
    assert "cannot be checked" in captured.err
    # Held, not promoted. Writing word:辛い:辛い would have put an id nothing can
    # repair into the store — a stored id is exempt from the re-mint that fixes
    # it — so the row waits in the staging file, which is committed.
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []
    assert staged.is_file()
    held, _ = read_staging(staged)
    assert [item.id for item in held] == ["word:辛い:辛い"]
    assert "Re-minted" not in captured.out


def test_a_note_a_filter_drops_is_still_in_the_collection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deck's include/exclude filters answer "what does this deck build",
    which is not "does this id exist". A note the deck declares and a filter
    drops is still in the file, still carries what a human wrote into it, and
    its GUID may already be in Anki."""
    root = tmp_path
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'deck_dir = "decks"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    (root / "decks").mkdir()
    (root / "decks" / "verbs.yaml").write_text(
        "name: Verbs\n"
        "deck:\n"
        "  exclude_ids:\n"
        "    - word:辛い:からい\n"
        "notes:\n"
        "  - id: word:辛い:からい\n"
        "    expression: 辛い\n"
        "    reading: つらい\n"
        "    meanings: [painful]\n"
        "    furigana: 辛[つら]い\n",
        encoding="utf-8",
    )
    staged = root / "staging" / "ai.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        staged,
        [
            VocabularyRecord(
                id="word:辛い:からい",
                expression="辛い",
                reading="つらい",
                meanings=["painful"],
                usage_notes="written by a model",
                source=SourceReference(type="shirabe", imported_from="export.csv"),
            )
        ],
        {"source_file": "vocabulary.json", "model": "m", "review_notes": "n"},
    )

    assert cli.main(["--root", str(root), "promote", str(staged), "--skip-reading-check"]) == 0

    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in stored] == ["word:辛い:からい"]
    assert "Re-minted" not in capsys.readouterr().out


def test_a_note_with_no_id_of_its_own_is_still_in_the_collection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The deck already builds it under the id minted from its expression and
    reading, and that id is what its GUID came from. Skipping it here would
    re-mint a staged row onto a second id beside the curated note."""
    root = tmp_path
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'deck_dir = "decks"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    (root / "decks").mkdir()
    (root / "decks" / "verbs.yaml").write_text(
        "name: Verbs\n"
        "notes:\n"
        "  - expression: 辛い\n"
        "    reading: からい\n"
        "    meanings: [spicy]\n",
        encoding="utf-8",
    )
    staged = root / "staging" / "ai.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        staged,
        [
            VocabularyRecord(
                id="word:辛い:からい",
                expression="辛い",
                reading="つらい",
                meanings=["painful"],
                source=SourceReference(type="shirabe", imported_from="export.csv"),
            )
        ],
        {"source_file": "vocabulary.json", "review_notes": "n"},
    )

    assert cli.main(["--root", str(root), "promote", str(staged), "--skip-reading-check"]) == 0

    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in stored] == ["word:辛い:からい"]
    assert "Re-minted" not in capsys.readouterr().out


def test_a_source_backed_decks_filtered_out_ids_still_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deck with `source:` and `include_ids` contributes every id its source
    declares, not the ones it happens to build.

    Green before the change as well as after: the `source:` branch was already
    unfiltered. It had no test at all, and it is where the shipped deck layout's
    behavior changed, so it gets one now."""
    root = tmp_path
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'deck_dir = "decks"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    stale = VocabularyRecord(
        id="word:辛い:からい",
        expression="辛い",
        reading="つらい",
        meanings=["painful"],
        source=SourceReference(type="shirabe", imported_from="export.csv"),
    )
    other = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        source=SourceReference(type="shirabe", imported_from="export.csv"),
    )
    (root / "records.json").write_text(
        json.dumps([stale.to_dict(), other.to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    (root / "decks").mkdir()
    (root / "decks" / "verbs.yaml").write_text(
        "name: Verbs\n"
        "deck:\n"
        '  source: "../records.json"\n'
        "  include_ids:\n"
        "    - word:話す:はなす\n",
        encoding="utf-8",
    )
    staged = root / "staging" / "ai.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        staged, [stale], {"source_file": "vocabulary.json", "review_notes": "n"}
    )

    assert cli.main(["--root", str(root), "promote", str(staged), "--skip-reading-check"]) == 0

    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in stored] == ["word:辛い:からい"]
    assert "Re-minted" not in capsys.readouterr().out


def test_an_id_hold_is_not_a_reading_hold(tmp_path: Path) -> None:
    """The reading assistant proposes a reading for held rows. This hold is
    about the id — proposing a reading would spend a call on a settled question
    and, for a homograph the reviewer chose, suggest the one they rejected."""
    held = annotate(
        VocabularyRecord(
            id="word:辛い:からい",
            expression="辛い",
            reading="からい",
            meanings=["spicy"],
            source=SourceReference(type="shirabe", imported_from="export.csv"),
        ),
        hold_reason=HOLD_UNVERIFIABLE_ID,
    )

    assert enrich.needs_reading(held) is False
    assert enrich.needs_reading(replace(held, reading="")) is True


def test_a_deck_janki_calls_broken_is_broken_here_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deck this reader accepts while build, validate and status refuse it is
    the worst of both: its source file is never opened, its ids vanish from the
    set, and the caller acts on a set it believes is complete."""
    root = tmp_path
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'deck_dir = "decks"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    (root / "decks").mkdir()
    (root / "decks" / "verbs.yaml").write_text("name: Verbs\ndeck: Verbs\n", encoding="utf-8")
    staged = root / "staging" / "held.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        staged,
        [
            VocabularyRecord(
                id="word:辛い:辛い",
                expression="辛い",
                reading="からい",
                meanings=["spicy"],
                source=SourceReference(type="shirabe", imported_from="export.csv"),
            )
        ],
        {"source_file": "export.csv", "review_notes": "n"},
    )

    assert cli.main(["--root", str(root), "promote", str(staged), "--skip-reading-check"]) == 0

    captured = capsys.readouterr()
    assert "deck section must be a mapping" in captured.err
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []
    assert staged.is_file(), "the row waits rather than taking an unrepairable id"


def test_a_proved_id_is_promoted_even_when_another_deck_will_not_parse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`remint_blocked` means the id set is incomplete, not wrong. An id in it
    was positively proved present, so holding that row would block a run over
    something nothing was ever uncertain about — and record the reason as
    "cannot check this id", which is false for it."""
    root = tmp_path
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'deck_dir = "decks"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    curated = VocabularyRecord(
        id="word:辛い:からい",
        expression="辛い",
        reading="つらい",
        meanings=["painful"],
        source=SourceReference(type="shirabe", imported_from="export.csv"),
    )
    (root / "vocabulary.json").write_text(
        json.dumps([curated.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    (root / "decks").mkdir()
    (root / "decks" / "broken.yaml").write_text("name: [unclosed\n", encoding="utf-8")
    staged = root / "staging" / "ai.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        staged,
        [replace(curated, usage_notes="written by a model")],
        {"source_file": "vocabulary.json", "model": "m", "review_notes": "n"},
    )

    assert cli.main(["--root", str(root), "promote", str(staged), "--skip-reading-check"]) == 0

    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in stored] == ["word:辛い:からい"]
    assert stored[0]["usage_notes"] == "written by a model", "the run was not blocked"
    assert "Re-minted" not in capsys.readouterr().out
    assert not staged.exists(), "the row was promoted, not held"


def test_a_hold_reason_janki_does_not_recognise_still_needs_a_reading(
    tmp_path: Path,
) -> None:
    """Staging files are hand-edited. Under an allow-list a reviewer's own
    wording would silently mean "not a reading hold", and the reading assistant
    would report a file with held rows as having none — while `status --staged`
    went on listing them."""
    typed_by_hand = annotate(
        VocabularyRecord(
            id="word:話す:はなす",
            expression="話す",
            reading="はなす",
            meanings=["to speak"],
            source=SourceReference(type="shirabe", imported_from="export.csv"),
        ),
        hold_reason="check the okurigana",
    )

    assert enrich.needs_reading(typed_by_hand) is True


def test_the_near_miss_warning_stays_off_non_extract_rows() -> None:
    # _accept_examples deliberately ignores non-extract rows, so a sentinel
    # typed there survives verbatim — warning about it would tell the user to
    # retype the exact value they typed.
    imported = record(
        source=SourceReference(
            type="shirabe",
            imported_from="export.csv",
            raw_fields={"example_authority": "staging-review"},
        ),
        examples=[ExampleSentence(japanese="友達と日本語を話します。")],
    )

    result = check_readings([imported], skip_reading_check=True)

    assert result.warnings == []


def test_a_typoed_sentinel_on_an_extract_row_is_named() -> None:
    typoed = record(
        examples=[ExampleSentence(japanese="友達と日本語を話します。")],
        source=SourceReference(
            type="extract",
            imported_from="lesson.pdf",
            raw_fields={"example_authority": "staging_review"},
        ),
    )

    result = check_readings([typoed], skip_reading_check=True)

    assert any("accepts nothing" in warning for warning in result.warnings)
