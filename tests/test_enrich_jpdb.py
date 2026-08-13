"""Dictionary enrichment from jpdb — ``janki enrich --jpdb``.

No network (IMPLEMENTATION_PLAN rule 6): the client is driven through a fake
transport that answers ``/parse`` and ``/lookup-vocabulary`` from canned
dictionary data. Rows are built in the client's default field order, so a
change to ``DEFAULT_VOCABULARY_FIELDS`` breaks these tests loudly rather than
silently shifting a column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from japanese_anki import cli, enrich, jpdb, pitch
from japanese_anki.enrich import (
    EnrichError,
    enrich_records,
    format_field_diff,
    parse_force_fields,
    suggest_readings,
)
from japanese_anki.jpdb import JpdbClient
from japanese_anki.models import SourceReference, VocabularyRecord

# The vocabulary row order jpdb answers a default `/parse` in.
assert jpdb.DEFAULT_VOCABULARY_FIELDS == (
    "vid",
    "sid",
    "spelling",
    "reading",
    "pitch_accent",
    "frequency_rank",
    "part_of_speech",
)


def vocab(
    vid: int,
    sid: int,
    spelling: str,
    reading: str,
    accent: list[str],
    rank: int,
    pos: list[str],
) -> list[Any]:
    return [vid, sid, spelling, reading, accent, rank, pos]


def parse_response(*entries: tuple[Any, list[Any]]) -> dict[str, Any]:
    """A ``/parse`` response: one ``(token_furigana, vocabulary_row)`` per word."""
    tokens = [[index, furigana] for index, (furigana, _) in enumerate(entries)]
    return {"tokens": [tokens], "vocabulary": [row for _, row in entries]}


HANASU = vocab(1562350, 4280520068, "話す", "はなす", ["LHLL"], 200, ["vt", "v5", "v5s"])
HANASU_FURIGANA = [["話", "はな"], "す"]

# 一日: the homograph the forced-furigana path exists for.
ICHINICHI = vocab(1579110, 111, "一日", "いちにち", ["LHHH"], 900, ["n"])
TSUITACHI = vocab(1579110, 222, "一日", "ついたち", ["LHHL"], 1400, ["n"])
ICHINICHI_FURIGANA = [["一日", "いちにち"]]
TSUITACHI_FURIGANA = [["一日", "ついたち"]]


class FakeApi:
    """Answers ``/parse`` and ``/lookup-vocabulary`` from canned data.

    ``/parse`` is keyed on whether the request forced a reading, because that
    is the distinction the enrichment turns on: the unforced answer is what
    janki's stored reading is *compared* to, and the forced one is what it may
    be written from. A fake that returned the same thing either way could not
    fail the homograph test.
    """

    def __init__(
        self,
        unforced: dict[str, dict[str, Any]],
        forced: dict[tuple[str, str], dict[str, Any]] | None = None,
        senses: dict[tuple[int, int], dict[str, Any]] | None = None,
    ) -> None:
        self.unforced = unforced
        self.forced = forced or {}
        self.senses = senses or {}
        self.bodies: list[tuple[str, dict[str, Any]]] = []

    def __call__(
        self, url: str, body: dict[str, Any], headers: dict[str, str]
    ) -> tuple[int, Any]:
        endpoint = url.rsplit("/api/v1/", 1)[-1]
        self.bodies.append((endpoint, body))
        if endpoint == "parse":
            return 200, self._parse(body)
        if endpoint == "lookup-vocabulary":
            return 200, self._lookup(body)
        raise AssertionError(f"unexpected request to {url}")

    def _parse(self, body: dict[str, Any]) -> dict[str, Any]:
        text = body["text"][0]
        if "furigana" in body:
            reading = body["furigana"][0][0][2]
            key = (text, reading)
            if key not in self.forced:
                raise AssertionError(f"fake has no forced parse for {key}")
            return self.forced[key]
        if text not in self.unforced:
            raise AssertionError(f"fake has no parse for {text!r}")
        return self.unforced[text]

    def _lookup(self, body: dict[str, Any]) -> dict[str, Any]:
        fields = body["fields"]
        rows = []
        for vid, sid in body["list"]:
            sense = self.senses.get((vid, sid))
            if sense is None:
                raise AssertionError(f"fake has no sense for {(vid, sid)}")
            rows.append([sense.get(name) for name in fields])
        return {"vocabulary_info": rows}

    def calls(self, endpoint: str) -> list[dict[str, Any]]:
        return [body for name, body in self.bodies if name == endpoint]


def client_for(api: FakeApi) -> JpdbClient:
    return JpdbClient("test-key", api, sleep=lambda _s: None, jitter=lambda: 0.0)


def record(**overrides: Any) -> VocabularyRecord:
    """A Shirabe-shaped record: identity filled in, dictionary fields empty."""
    values: dict[str, Any] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "source": SourceReference(type="shirabe", imported_from="export.csv"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def hanasu_api() -> FakeApi:
    return FakeApi({"話す": parse_response((HANASU_FURIGANA, HANASU))})


# --- filling ----------------------------------------------------------------


def test_every_field_jpdb_states_or_janki_computes_is_filled() -> None:
    api = hanasu_api()

    result = enrich_records(client_for(api), [record()])

    enriched = result.records[0]
    assert enriched.furigana == "話[はな]す"
    assert enriched.romaji == "hanasu"
    assert enriched.part_of_speech == "verb"
    assert enriched.verb_group == "godan"
    assert enriched.pitch_accent == ["LHLL"]
    assert pitch.has_source_binding(enriched)
    assert enriched.frequency_rank == 200
    assert enriched.conjugations["negative"] == "話さない"
    assert result.looked_up == 1
    assert result.warnings == []


def test_the_reading_is_never_written_even_when_jpdb_states_one() -> None:
    # The reading is half of stable_record_id: writing it would re-ID the
    # record and orphan its Anki review history.
    api = hanasu_api()

    result = enrich_records(client_for(api), [record()])

    assert result.records[0].reading == "はなす"
    assert result.records[0].id == "word:話す:はなす"
    assert "reading" not in result.changes["word:話す:はなす"]


def test_meanings_are_not_touched_by_a_dictionary_pass() -> None:
    api = hanasu_api()

    result = enrich_records(client_for(api), [record(meanings=[])])

    assert result.records[0].meanings == []
    assert "meanings" not in result.changes.get("word:話す:はなす", {})


def test_a_populated_field_is_left_exactly_as_the_human_wrote_it() -> None:
    api = hanasu_api()
    curated = record(part_of_speech="godan verb (u-verb)", pitch_accent=["HLLL"])

    result = enrich_records(client_for(api), [curated])

    enriched = result.records[0]
    assert enriched.part_of_speech == "godan verb (u-verb)"
    assert enriched.pitch_accent == ["HLLL"]
    assert not pitch.has_source_binding(enriched)
    changed = result.changes["word:話す:はなす"]
    assert "part_of_speech" not in changed
    assert "pitch_accent" not in changed
    assert "furigana" in changed


def test_a_record_with_nothing_to_fill_never_reaches_the_network() -> None:
    api = hanasu_api()
    full = record(
        furigana="話[はな]す",
        romaji="hanasu",
        part_of_speech="verb",
        verb_group="godan",
        conjugations={"polite": "話します"},
        pitch_accent=["LHLL"],
        frequency_rank=200,
    )

    result = enrich_records(client_for(api), [full])

    assert api.bodies == []
    assert result.skipped == 1
    assert result.looked_up == 0
    assert result.changes == {}


def test_a_field_jpdb_has_no_value_for_is_left_empty_not_blanked() -> None:
    api = FakeApi(
        {"本": parse_response(([["本", "ほん"]], vocab(1, 2, "本", "ほん", [], 0, ["n"])))}
    )

    result = enrich_records(
        client_for(api), [record(id="word:本:ほん", expression="本", reading="ほん")]
    )

    enriched = result.records[0]
    assert enriched.pitch_accent == []
    assert enriched.verb_group == ""
    assert enriched.conjugations == {}
    assert enriched.frequency_rank == 0


# --- --force-fields ---------------------------------------------------------


def test_force_fields_overwrites_only_the_fields_it_names() -> None:
    api = hanasu_api()
    curated = record(part_of_speech="godan verb (u-verb)", pitch_accent=["HLLL"])

    result = enrich_records(
        client_for(api), [curated], force_fields=("pitch_accent",)
    )

    enriched = result.records[0]
    assert enriched.pitch_accent == ["LHLL"]
    assert enriched.part_of_speech == "godan verb (u-verb)"


def test_force_fields_refuses_reading_by_name() -> None:
    with pytest.raises(EnrichError) as excinfo:
        parse_force_fields("furigana,reading")

    assert "record ID" in str(excinfo.value)


@pytest.mark.parametrize("value", ["", None])
def test_force_fields_absent_is_no_fields(value: str | None) -> None:
    assert parse_force_fields(value) == ()


def test_force_fields_of_separators_only_is_an_error_not_a_shrug() -> None:
    with pytest.raises(EnrichError):
        parse_force_fields(",,")


def test_force_fields_rejects_an_unknown_field() -> None:
    with pytest.raises(EnrichError) as excinfo:
        parse_force_fields("furrigana")

    assert "furigana" in str(excinfo.value)


def test_forcing_a_field_jpdb_has_nothing_for_does_not_blank_it() -> None:
    api = FakeApi(
        {"本": parse_response(([["本", "ほん"]], vocab(1, 2, "本", "ほん", [], 0, ["n"])))}
    )
    curated = record(
        id="word:本:ほん", expression="本", reading="ほん", pitch_accent=["LHH"]
    )

    result = enrich_records(client_for(api), [curated], force_fields=("pitch_accent",))

    assert result.records[0].pitch_accent == ["LHH"]


# --- readings ---------------------------------------------------------------


def homograph_api() -> FakeApi:
    """一日 parses as いちにち unless told otherwise; ついたち is its other sense."""
    return FakeApi(
        unforced={"一日": parse_response((ICHINICHI_FURIGANA, ICHINICHI))},
        forced={("一日", "ついたち"): parse_response((TSUITACHI_FURIGANA, TSUITACHI))},
        senses={
            (1579110, 111): {"reading": "いちにち", "alt_sids": [222]},
            (1579110, 222): {"reading": "ついたち", "alt_sids": [111]},
        },
    )


def test_a_homograph_is_re_parsed_with_the_stored_reading_forced() -> None:
    api = homograph_api()
    tsuitachi = record(id="word:一日:ついたち", expression="一日", reading="ついたち")

    result = enrich_records(client_for(api), [tsuitachi])

    parses = api.calls("parse")
    assert len(parses) == 2
    assert "furigana" not in parses[0]
    # The whole-expression span, in the encoding the same request declares.
    assert parses[1]["furigana"] == [[[0, 2, "ついたち"]]]
    assert parses[1]["position_length_encoding"] == "utf16"
    enriched = result.records[0]
    assert enriched.furigana == "一日[ついたち]"
    assert enriched.pitch_accent == []
    assert enriched.frequency_rank == 1400
    assert len(result.warnings) == 1
    assert "do not fit reading ついたち" in result.warnings[0]


def test_an_unrelated_kana_parse_retries_with_the_stored_reading() -> None:
    suru = vocab(1157170, 460825390, "する", "する", ["LHH"], 100, ["vs"])
    shinu = vocab(1310730, 851331686, "しぬ", "しぬ", ["LHH"], 30800, ["v5n"])
    api = FakeApi(
        unforced={"しぬ": parse_response((None, suru))},
        forced={("しぬ", "しぬ"): parse_response((None, shinu))},
    )
    item = record(id="word:しぬ:しぬ", expression="しぬ", reading="しぬ")

    result = enrich_records(client_for(api), [item])

    assert result.records[0].frequency_rank == 30800
    assert result.warnings == []
    parses = api.calls("parse")
    assert len(parses) == 2
    assert "furigana" not in parses[0]
    assert parses[1]["furigana"] == [[[0, 2, "しぬ"]]]


def test_the_reading_set_is_gathered_across_the_entrys_other_senses() -> None:
    api = homograph_api()
    tsuitachi = record(id="word:一日:ついたち", expression="一日", reading="ついたち")

    enrich_records(client_for(api), [tsuitachi])

    lookups = api.calls("lookup-vocabulary")
    assert lookups[0]["fields"] == ["reading", "alt_sids"]
    assert lookups[0]["list"] == [[1579110, 111]]
    # The alternate sense is asked about by vid, which is how the other reading
    # is found at all — it is not on the sense /parse picked.
    assert lookups[1]["list"] == [[1579110, 222]]


def test_a_reading_jpdb_does_not_list_is_warned_and_nothing_is_written() -> None:
    api = FakeApi(
        unforced={"話す": parse_response((HANASU_FURIGANA, HANASU))},
        senses={(1562350, 4280520068): {"reading": "はなす", "alt_sids": []}},
    )
    typo = record(id="word:話す:はなし", reading="はなし")

    result = enrich_records(client_for(api), [typo])

    assert result.changes == {}
    assert result.records[0] == typo
    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert "はなし" in warning and "はなす" in warning
    assert "never auto-fixed" in warning
    # Warned, not forced: asking jpdb to parse with a reading it does not have
    # would return something shaped like agreement.
    assert len(api.calls("parse")) == 1


def test_a_record_with_no_reading_at_all_is_enriched_from_the_first_parse() -> None:
    # Post-M1.5 no *imported* record has an empty reading; a hand-written one
    # can. There is nothing to compare, so the unforced answer is used as-is.
    api = hanasu_api()
    handwritten = VocabularyRecord(id="word:話す:", expression="話す", reading="")

    result = enrich_records(client_for(api), [handwritten])

    assert result.records[0].furigana == "話[はな]す"
    assert result.records[0].reading == ""
    # No reading means no romaji — the transliteration has nothing to read from.
    # The conjugation table does not need one: it inflects the expression, and
    # the reading only ever guards against the two disagreeing.
    assert result.records[0].romaji == ""
    assert result.records[0].conjugations["negative"] == "話さない"
    assert len(api.calls("parse")) == 1


def test_an_expression_jpdb_splits_into_several_words_is_warned() -> None:
    api = FakeApi(
        {
            "食べ物屋": parse_response(
                (
                    [["食", "た"], "べ", ["物", "もの"]],
                    vocab(1, 1, "食べ物", "たべもの", [], 2600, ["n"]),
                ),
                ([["屋", "や"]], vocab(2, 2, "屋", "や", [], 5000, ["suf"])),
            )
        }
    )
    compound = record(id="word:食べ物屋:たべものや", expression="食べ物屋", reading="たべものや")

    result = enrich_records(client_for(api), [compound])

    assert result.changes == {}
    assert "one word" in result.warnings[0]


def test_one_token_matching_the_whole_expression_wins_over_its_neighbours() -> None:
    # jpdb parses a trailing particle as its own token; the entry whose
    # spelling *is* the expression is still unambiguous.
    api = FakeApi(
        {
            "話す": parse_response(
                (HANASU_FURIGANA, HANASU),
                (None, vocab(2029010, 1, "を", "を", ["HL"], 100, ["prt"])),
            )
        }
    )

    result = enrich_records(client_for(api), [record()])

    assert result.records[0].frequency_rank == 200


# --- target selection -------------------------------------------------------


def test_only_the_named_ids_are_enriched() -> None:
    api = hanasu_api()
    other = record(id="word:本:ほん", expression="本", reading="ほん")

    result = enrich_records(client_for(api), [record(), other], ids=["word:話す:はなす"])

    assert set(result.changes) == {"word:話す:はなす"}
    assert result.records[1] == other


def test_an_unknown_id_is_an_error_before_any_api_call() -> None:
    api = hanasu_api()

    with pytest.raises(EnrichError) as excinfo:
        enrich_records(client_for(api), [record()], ids=["word:無い:ない"])

    assert "word:無い:ない" in str(excinfo.value)
    assert api.bodies == []


# --- the field diff ---------------------------------------------------------


def test_the_field_diff_is_a_record_header_and_one_line_per_field() -> None:
    lines = format_field_diff(
        {
            "word:話す:はなす": {
                "furigana": ("", "話[はな]す"),
                "frequency_rank": (None, 200),
            }
        }
    )

    assert lines == [
        "word:話す:はなす",
        "  furigana: (empty) -> 話[はな]す",
        "  frequency_rank: (none) -> 200",
    ]


# --- staging ----------------------------------------------------------------


def held(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:話す:",
        "expression": "話す",
        "reading": "",
        "source": SourceReference(
            type="shirabe",
            imported_from="export.csv",
            raw_fields={"hold_reason": "missing reading"},
        ),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def test_a_held_row_gets_the_reading_jpdb_proposes_as_an_annotation() -> None:
    api = hanasu_api()

    result = suggest_readings(client_for(api), [held()])

    annotated = result.records[0]
    assert annotated.source.raw_fields["suggested_reading"] == "はなす"
    # A proposal, not a fix: the row stays held until a human types it in.
    assert annotated.reading == ""
    assert annotated.source.raw_fields["hold_reason"] == "missing reading"
    assert result.suggested == {"word:話す:": "はなす"}
    assert result.held == 1
    assert result.warnings == []


def test_the_second_hold_class_is_annotated_too() -> None:
    # `reading contains kanji` mints word:<kanji>:<kanji> — it has a reading,
    # so selecting held rows on emptiness alone would skip it.
    api = hanasu_api()
    kanji_reading = held(
        id="word:話す:話す",
        reading="話す",
        source=SourceReference(raw_fields={"hold_reason": "reading contains kanji"}),
    )

    result = suggest_readings(client_for(api), [kanji_reading])

    assert result.records[0].source.raw_fields["suggested_reading"] == "はなす"
    assert result.held == 1


def test_a_row_that_is_not_held_is_left_alone() -> None:
    api = hanasu_api()

    result = suggest_readings(client_for(api), [record()])

    assert result.records[0] == record()
    assert result.held == 0
    assert api.bodies == []


# --- the CLI ----------------------------------------------------------------


def project(tmp_path: Path, records: list[VocabularyRecord]) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([item.to_dict() for item in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return tmp_path


def patch_api(monkeypatch: pytest.MonkeyPatch, api: FakeApi) -> None:
    monkeypatch.setenv("JPDB_API_KEY", "test-key")
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda key, *a, **kw: client_for(api))


def stored(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload}


def test_enrich_without_a_source_says_so_rather_than_guessing() -> None:
    assert cli.main(["enrich"]) == 1


def test_enrich_takes_one_source_at_a_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Each pass shows its own diff; running both at once would merge two
    # unrelated sets of proposals into one y/n.
    assert cli.main(["enrich", "--ai", "--jpdb"]) == 1

    assert "one pass at a time" in capsys.readouterr().err


def test_enrich_jpdb_writes_the_records_the_diff_and_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    patch_api(monkeypatch, hanasu_api())

    assert cli.main(["--root", str(root), "enrich", "--jpdb", "--yes"]) == 0

    out = capsys.readouterr().out
    assert "word:話す:はなす" in out
    assert "furigana: (empty) -> 話[はな]す" in out
    assert "Enriched 1 record(s)" in out
    assert stored(root)["word:話す:はなす"]["furigana"] == "話[はな]す"

    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    passes = book["records"]["word:話す:はなす"]["enriched"]
    assert [item["kind"] for item in passes] == ["jpdb"]
    assert "furigana" in passes[0]["fields"]


def test_enrich_jpdb_writes_nothing_when_there_is_nothing_to_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = hanasu_api()
    root = project(tmp_path, [record()])
    patch_api(monkeypatch, api)
    cli.main(["--root", str(root), "enrich", "--jpdb", "--yes"])
    before = (root / "vocabulary.json").read_text(encoding="utf-8")
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--jpdb", "--yes"]) == 0

    assert "Nothing to fill" in capsys.readouterr().out
    assert (root / "vocabulary.json").read_text(encoding="utf-8") == before


def test_a_declined_confirmation_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [record()])
    patch_api(monkeypatch, hanasu_api())
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    assert cli.main(["--root", str(root), "enrich", "--jpdb"]) == 1

    assert stored(root)["word:話す:はなす"]["furigana"] == ""
    assert not (root / "ledger.json").exists()


def test_enrich_staging_annotates_the_file_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    staging = root / "staging" / "shirabe-export-needs-reading.yaml"
    staging.parent.mkdir(parents=True)
    staging.write_text(
        yaml.safe_dump(
            {"records": [held().to_dict()], "source_file": "export.csv"},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    patch_api(monkeypatch, hanasu_api())

    code = cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--staging", str(staging), "--yes"]
    )

    assert code == 0
    written = yaml.safe_load(staging.read_text(encoding="utf-8"))
    assert written["records"][0]["source"]["raw_fields"]["suggested_reading"] == "はなす"
    # Metadata the reviewer's file carried survives the rewrite.
    assert written["source_file"] == "export.csv"
    assert "Suggested a reading for 1 of 1 held row(s)" in capsys.readouterr().out


def test_enrich_staging_takes_neither_force_fields_nor_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    patch_api(monkeypatch, hanasu_api())

    code = cli.main(
        [
            "--root",
            str(root),
            "enrich",
            "--jpdb",
            "--staging",
            str(root / "any.yaml"),
            "--force-fields",
            "furigana",
        ]
    )

    assert code == 1
    assert "--force-fields" in capsys.readouterr().err


def test_enrich_reports_a_reading_mismatch_on_stderr_and_still_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeApi(
        unforced={"話す": parse_response((HANASU_FURIGANA, HANASU))},
        senses={(1562350, 4280520068): {"reading": "はなす", "alt_sids": []}},
    )
    root = project(tmp_path, [record(id="word:話す:はなし", reading="はなし")])
    patch_api(monkeypatch, api)

    code = cli.main(["--root", str(root), "enrich", "--jpdb", "--yes"])

    captured = capsys.readouterr()
    assert code == 0
    assert "はなし" in captured.err
    assert "Nothing to fill" in captured.out
    assert stored(root)["word:話す:はなし"]["furigana"] == ""


def test_enrich_can_be_pointed_at_single_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = hanasu_api()
    other = record(id="word:本:ほん", expression="本", reading="ほん")
    root = project(tmp_path, [record(), other])
    patch_api(monkeypatch, api)

    code = cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--yes", "word:話す:はなす"]
    )

    assert code == 0
    assert stored(root)["word:本:ほん"]["furigana"] == ""
    assert stored(root)["word:話す:はなす"]["furigana"] == "話[はな]す"


def test_enrichable_fields_can_never_include_the_reading() -> None:
    # A guard, not a tautology: this tuple is what --force-fields validates
    # against, so an addition to it is an addition to what may be overwritten.
    assert "reading" not in enrich.ENRICHABLE_FIELDS
    assert "id" not in enrich.ENRICHABLE_FIELDS


# --- the lemma trap ---------------------------------------------------------


# 行った parses to one token whose entry is the *lemma* 行く. The token's
# furigana describes the surface form; the entry's reading does not.
ITTA_FURIGANA = [["行", "い"], "った"]
IKU = vocab(1578850, 777, "行く", "いく", ["LHL"], 300, ["vi", "v5k-s"])


def itta_api() -> FakeApi:
    return FakeApi(
        unforced={"行った": parse_response((ITTA_FURIGANA, IKU))},
        senses={(1578850, 777): {"reading": "いく", "alt_sids": []}},
    )


def test_a_held_inflected_form_is_suggested_its_own_reading_not_the_lemmas() -> None:
    # The bug this guards: suggesting いく for 行った would have a reviewer mint
    # word:行った:いく permanently — the exact unrecoverable outcome the staging
    # review exists to prevent.
    api = itta_api()
    row = held(id="word:行った:", expression="行った", reading="")

    result = suggest_readings(client_for(api), [row])

    assert result.suggested == {"word:行った:": "いった"}
    assert result.records[0].source.raw_fields["suggested_reading"] == "いった"


def test_an_inflected_form_with_no_reading_is_not_enriched_from_the_lemma() -> None:
    # Same trap in the main pass: 行く's pitch accent and frequency rank
    # describe a different word, and this record has no reading to prove the
    # entry is the right one.
    api = itta_api()
    handwritten = VocabularyRecord(id="word:行った:", expression="行った", reading="")

    result = enrich_records(client_for(api), [handwritten])

    assert result.changes == {}
    assert result.records[0].pitch_accent == []
    assert "no reading to confirm" in result.warnings[0]


def test_an_all_kana_expression_still_falls_back_to_the_entry_reading() -> None:
    # jpdb sends null furigana for an all-kana token, and there the entry is
    # the same surface form, so its reading can be trusted.
    api = FakeApi(
        {"たべる": parse_response((None, vocab(3, 4, "たべる", "たべる", [], 13600, ["v1"])))}
    )
    row = held(id="word:たべる:話す", expression="たべる", reading="話す")

    result = suggest_readings(client_for(api), [row])

    assert result.suggested == {"word:たべる:話す": "たべる"}


def test_furigana_to_reading_spells_out_the_surface_form() -> None:
    assert jpdb.furigana_to_reading(ITTA_FURIGANA) == "いった"
    assert jpdb.furigana_to_reading(HANASU_FURIGANA) == "はなす"
    assert jpdb.furigana_to_reading([["日", "にっ"], ["本", "ぽん"], ["語", "ご"]]) == "にっぽんご"
    assert jpdb.furigana_to_reading(None) == ""


# --- the staging file is a reviewer's working document ----------------------


STAGING_WITH_REVIEW_NOTES = """\
# Held out of shirabe-export.csv on 2026-08-07 — check these with my teacher.
source_file: export.csv
records:
  # not sure this one is even worth a card
  - id: 'word:話す:'
    expression: 話す
    reading: ''
    my_note: ask about the intransitive pair
    source:
      type: shirabe
      imported_from: export.csv
      raw_fields:
        hold_reason: missing reading
"""


def test_annotating_a_staging_file_keeps_the_reviewers_comments_and_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [])
    staging = root / "review.yaml"
    staging.write_text(STAGING_WITH_REVIEW_NOTES, encoding="utf-8")
    patch_api(monkeypatch, hanasu_api())

    code = cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--staging", str(staging), "--yes"]
    )

    assert code == 0
    text = staging.read_text(encoding="utf-8")
    # The annotation landed...
    assert "suggested_reading: はなす" in text
    # ...and nothing the reviewer wrote was rewritten out from under them.
    assert "# Held out of shirabe-export.csv on 2026-08-07" in text
    assert "# not sure this one is even worth a card" in text
    assert "my_note: ask about the intransitive pair" in text
    written = yaml.safe_load(text)
    assert written["records"][0]["my_note"] == "ask about the intransitive pair"
    assert written["source_file"] == "export.csv"


def test_annotating_adds_no_empty_schema_fields_the_file_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A row a human wrote by hand stays as short as they wrote it.
    root = project(tmp_path, [])
    staging = root / "review.yaml"
    staging.write_text(STAGING_WITH_REVIEW_NOTES, encoding="utf-8")
    patch_api(monkeypatch, hanasu_api())

    cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--staging", str(staging), "--yes"]
    )

    row = yaml.safe_load(staging.read_text(encoding="utf-8"))["records"][0]
    assert "audio" not in row
    assert "examples" not in row
    assert "pitch_accent" not in row


def test_a_row_jpdb_could_not_help_with_is_reported_before_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # One row jpdb helps with and one it does not: the prompt asks about the
    # first, and answering "no" must not be the reason the second was never
    # mentioned. Pre-fix the warnings printed after the gate, so a declined
    # run discarded them entirely.
    root = project(tmp_path, [])
    staging = root / "review.yaml"
    staging.write_text(
        yaml.safe_dump(
            {
                "records": [
                    held().to_dict(),
                    held(id="word:謎:", expression="謎", reading="").to_dict(),
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    api = FakeApi(
        {"話す": parse_response((HANASU_FURIGANA, HANASU)), "謎": parse_response()}
    )
    patch_api(monkeypatch, api)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    code = cli.main(["--root", str(root), "enrich", "--jpdb", "--staging", str(staging)])

    captured = capsys.readouterr()
    assert code == 1
    assert "one word" in captured.err
    # Declined, so the file is untouched.
    assert "suggested_reading" not in staging.read_text(encoding="utf-8")


STAGING_WITH_YAML_11_ISMS = """\
records:
  - id: 'word:話す:'
    expression: 話す
    reading: ''
    source:
      type: shirabe
      raw_fields:
        hold_reason: missing reading
        already_known: yes
        checked_at: 12:30
"""


def test_annotating_does_not_rewrite_values_in_the_reviewers_own_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The two YAML dialects disagree about these: 1.1 reads `yes` as a boolean
    # and `12:30` as the sexagesimal integer 750, 1.2 reads both as strings.
    # Diffing one dialect's reading against the other's calls them "changed"
    # and writes janki's version over the reviewer's line.
    root = project(tmp_path, [])
    staging = root / "review.yaml"
    staging.write_text(STAGING_WITH_YAML_11_ISMS, encoding="utf-8")
    patch_api(monkeypatch, hanasu_api())

    code = cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--staging", str(staging), "--yes"]
    )

    assert code == 0
    text = staging.read_text(encoding="utf-8")
    assert "suggested_reading: はなす" in text
    assert "already_known: yes" in text
    assert "checked_at: 12:30" in text
    assert "'True'" not in text and "'750'" not in text


def test_a_file_that_cannot_be_rewritten_fails_before_the_api_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A duplicate key is an easy slip while hand-editing. PyYAML accepts it
    # silently, so the read and the whole pass would succeed and the write
    # would then fail — after the user had already confirmed it.
    root = project(tmp_path, [])
    staging = root / "review.yaml"
    staging.write_text(
        STAGING_WITH_YAML_11_ISMS.replace(
            "    expression: 話す", "    expression: 話す\n    expression: 話す"
        ),
        encoding="utf-8",
    )
    api = hanasu_api()
    patch_api(monkeypatch, api)

    code = cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--staging", str(staging), "--yes"]
    )

    assert code == 1
    assert "review.yaml" in capsys.readouterr().err
    # Nothing was spent, and nothing was touched.
    assert api.bodies == []
    assert "suggested_reading" not in staging.read_text(encoding="utf-8")


def test_a_field_jpdb_has_no_answer_for_is_looked_up_again_next_run() -> None:
    # Not a bug, but the README makes a cost claim about it: a noun has no verb
    # group and no conjugation table, so those stay empty and the record stays
    # fillable. Nothing records "asked, and there was nothing".
    api = FakeApi(
        {"本": parse_response(([["本", "ほん"]], vocab(1, 2, "本", "ほん", ["LH"], 500, ["n"])))}
    )
    noun = VocabularyRecord(id="word:本:ほん", expression="本", reading="ほん")

    first = enrich_records(client_for(api), [noun])
    calls = len(api.bodies)
    second = enrich_records(client_for(api), first.records)

    assert second.looked_up == 1
    assert second.changes == {}
    assert len(api.bodies) == calls + 1


def test_a_failed_ledger_write_says_a_re_run_would_skip_these_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A record jpdb filled has nothing left it can fill, so a re-run reaches
    nothing whether it skips the record (the dictionary had every field) or
    looks it up again and proposes nothing (a noun, whose conjugations never
    fill). The message covers both, and is not the polish one."""
    root = project(tmp_path, [record()])
    patch_api(monkeypatch, hanasu_api())
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("disk full")),
    )

    assert cli.main(["--root", str(root), "enrich", "--jpdb", "--yes"]) == 1

    err = capsys.readouterr().err
    assert "'status --rebuild' cannot bring it back" in err
    assert "either skips these records or looks them up and proposes nothing" in err
    assert "not a free repair" not in err
    assert stored(root)["word:話す:はなす"]["furigana"] == "話[はな]す"
