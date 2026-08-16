"""Importing vocabulary from jpdb — the API path and the userscript CSV path.

No network (IMPLEMENTATION_PLAN rule 6): every API test drives the client
through a routing fake transport whose answers come from committed fixtures.
The lookup fixture is a real captured response; the deck fixtures are
hand-authored from real response *shapes* (their provenance notes say which,
and why).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from japanese_anki import cli
from japanese_anki.importers.jpdb_import import (
    JpdbImportError,
    deck_tag,
    import_csv,
    import_deck,
    record_from_entry,
    select_decks,
)
from japanese_anki.jpdb import JpdbClient

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LOOKUP_FIXTURE = FIXTURES / "jpdb-lookup-sample.json"
DECKS_FIXTURE = FIXTURES / "jpdb-decks-sample.json"


def _fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class RoutingTransport:
    """Answers by endpoint, the way the real API does.

    A scripted list of responses would encode the *order* the importer happens
    to call endpoints in today, so a refactor that reordered two harmless calls
    would fail every test here for no reason. Routing on the URL tests what the
    importer asks for, not when.
    """

    def __init__(self, decks: dict[str, Any], lookup: dict[str, Any]) -> None:
        self.decks = decks
        self.lookup = lookup
        self.calls: list[SimpleNamespace] = []

    def __call__(
        self, url: str, body: dict[str, Any], headers: dict[str, str]
    ) -> tuple[int, Any]:
        self.calls.append(SimpleNamespace(url=url, body=body, headers=headers))
        if url.endswith("/list-user-decks"):
            return 200, self.decks["list_user_decks"]
        if url.endswith("/deck/list-vocabulary"):
            return 200, self.decks["deck_vocabulary"][str(body["id"])]
        if url.endswith("/lookup-vocabulary"):
            return 200, self._lookup(body)
        raise AssertionError(f"unexpected request to {url}")

    def _lookup(self, body: dict[str, Any]) -> dict[str, Any]:
        """Return the fixture's rows for exactly the pairs asked about.

        Positional, like the real endpoint: the client matches answers to
        requests by index, so a fake that returned rows in fixture order would
        hide a mis-zip rather than expose it.
        """
        by_pair = {
            tuple(pair): row
            for pair, row in zip(
                self.lookup["request_list"], self.lookup["response"]["vocabulary_info"], strict=True
            )
        }
        rows = []
        for pair in body["list"]:
            row = by_pair.get(tuple(pair))
            if row is None:
                raise AssertionError(f"fixture has no entry for {pair}")
            rows.append(row)
        return {"vocabulary_info": rows}

    def endpoints(self) -> list[str]:
        return [call.url.rsplit("/api/v1/", 1)[-1] for call in self.calls]


def _client(transport: RoutingTransport) -> JpdbClient:
    return JpdbClient("test-key", transport, sleep=lambda _s: None, jitter=lambda: 0.0)


def _api() -> tuple[JpdbClient, RoutingTransport]:
    transport = RoutingTransport(_fixture(DECKS_FIXTURE), _fixture(LOOKUP_FIXTURE))
    return _client(transport), transport


def _entry(**overrides: Any) -> dict[str, Any]:
    """One lookup entry in the shape the client hands over — real values."""
    values: dict[str, Any] = {
        "spelling": "話す",
        "reading": "はなす",
        "frequency_rank": 200,
        "pitch_accent": ["LHLL"],
        "meanings_chunks": [["to talk", "to speak"], ["to tell"]],
        "meanings_part_of_speech": [["vt", "v5", "v5s"], ["vt", "v5", "v5s"]],
        "part_of_speech": ["vt", "v5", "v5s"],
        "card_state": ["locked", "new"],
        "vid": 1562350,
        "sid": 4280520068,
    }
    values.update(overrides)
    return values


def _record(**overrides: Any) -> Any:
    return record_from_entry(_entry(**overrides), deck_name="Lesson 1", source_ref="Lesson 1")


# --- the fixture is the API's shape, not ours -------------------------------


def test_the_lookup_fixture_still_matches_the_fields_the_client_asks_for() -> None:
    # The response is positional: if DEFAULT_LOOKUP_FIELDS and the captured
    # request_fields ever drift apart, every row here would be zipped under the
    # wrong names and the tests below would assert confidently about nonsense.
    from japanese_anki.jpdb import DEFAULT_LOOKUP_FIELDS

    fixture = _fixture(LOOKUP_FIXTURE)
    assert fixture["request_fields"] == list(DEFAULT_LOOKUP_FIELDS)
    for row in fixture["response"]["vocabulary_info"]:
        assert len(row) == len(DEFAULT_LOOKUP_FIELDS)


# --- mapping one entry ------------------------------------------------------


def test_an_entry_becomes_a_record_with_every_field_jpdb_stated() -> None:
    record = _record()

    assert record.id == "word:話す:はなす"
    assert record.expression == "話す"
    assert record.reading == "はなす"
    assert record.part_of_speech == "verb"
    assert record.verb_group == "godan"
    assert record.transitivity == "transitive"
    assert record.pitch_accent == ["LHLL"]
    assert record.frequency_rank == 200
    assert record.source.type == "jpdb"
    assert record.source.imported_from == "Lesson 1"


def test_a_suru_noun_is_imported_without_a_transitivity() -> None:
    """The importer is the *other* writer of this field, and it was unpinned.

    JMdict tags a noun that takes する with `vt`/`vi` too — 仕事 is
    `["n", "vs", "vi"]` — so a bare dictionary answer puts "intransitive" on a
    record whose own part of speech says "noun". `enrich --jpdb` was fixed for
    this; `import-jpdb` writes the same field from the same codes and had no
    test, so the two could drift apart silently.
    """
    record = _record(part_of_speech=["n", "vs", "vi"])

    assert record.part_of_speech == "noun"
    assert record.transitivity == "", "a noun states no transitivity"


def test_meanings_keep_one_line_per_sense_rather_than_one_per_gloss() -> None:
    # The exporter renders one <br> line per meanings entry and the production
    # template uses the whole field as the prompt, so 話す's three senses must
    # stay three lines — flattening spills eleven fragments onto the card.
    record = _record(
        meanings_chunks=[
            ["to talk", "to speak", "to converse", "to chat"],
            ["to tell", "to explain"],
            ["to speak (a language)"],
        ]
    )

    assert record.meanings == [
        "to talk, to speak, to converse, to chat",
        "to tell, to explain",
        "to speak (a language)",
    ]


@pytest.mark.parametrize(
    "chunks",
    [None, [], "to speak", [[]], [["", "  "]], [None], 7],
)
def test_meanings_survive_every_shape_jpdb_could_send(chunks: Any) -> None:
    # Meanings are metadata; one odd value must not stop a 2,000-word import.
    record = _record(meanings_chunks=chunks)
    assert all(isinstance(meaning, str) and meaning for meaning in record.meanings)


def test_romaji_is_computed_from_the_reading() -> None:
    assert _record().romaji == "hanasu"


def test_conjugations_are_computed_for_a_verb_jpdb_gave_a_class_for() -> None:
    record = _record()
    assert record.conjugations["negative"] == "話さない"
    assert record.conjugations["te_form"] == "話して"


def test_an_i_adjective_conjugates_even_though_jpdb_has_no_verb_class_for_it() -> None:
    # `adj-i` is not a verb code, so verb_group is empty and only the part of
    # speech says how the word inflects.
    record = _record(spelling="高い", reading="たかい", part_of_speech=["adj-i"])

    assert record.verb_group == ""
    assert record.part_of_speech == "i-adjective"
    assert record.conjugations["negative"] == "高くない"


def test_a_noun_gets_no_conjugation_table_invented_for_it() -> None:
    record = _record(spelling="日本語", reading="にほんご", part_of_speech=["n"])

    assert record.part_of_speech == "noun"
    assert record.verb_group == ""
    assert record.conjugations == {}


def test_an_unknown_verb_class_leaves_the_group_empty_rather_than_guessing() -> None:
    # `v2a-s` is a real archaic JMDict class this project has no table for.
    record = _record(part_of_speech=["vt", "v2a-s"])

    assert record.part_of_speech == "verb"
    assert record.verb_group == ""
    assert record.conjugations == {}


def test_raw_fields_keep_the_identifiers_the_record_cannot_carry() -> None:
    record = _record()

    assert record.source.raw_fields == {
        "vid": "1562350",
        "sid": "4280520068",
        "deck": "Lesson 1",
        "card_state": "locked,new",
    }


def test_a_word_in_no_deck_has_no_card_state_key_at_all() -> None:
    # `null` must not become the string "None": status --duplicates groups
    # records by these strings, and "None" is a value nobody typed.
    record = _record(card_state=None)

    assert "card_state" not in record.source.raw_fields


def test_occurence_counts_are_not_stored() -> None:
    # Deliberately deferred by M2.5. A count that changes as a deck is mined
    # would rewrite the record on every import for no card-visible gain.
    record = record_from_entry(
        _entry(occurences=12), deck_name="Lesson 1", source_ref="Lesson 1"
    )

    assert "occurences" not in record.source.raw_fields


def test_a_missing_frequency_rank_is_none_not_zero() -> None:
    # None means "never looked up"; 0 would be a real rank.
    assert _record(frequency_rank=None).frequency_rank is None
    assert _record(frequency_rank="not a number").frequency_rank is None
    assert _record(frequency_rank=0).frequency_rank == 0


def test_a_kana_only_word_defaults_its_reading_before_the_id_is_minted() -> None:
    record = _record(spelling="たべる", reading="", part_of_speech=["vt", "v1"])

    assert record.id == "word:たべる:たべる"
    assert record.reading == "たべる"
    assert record.romaji == "taberu"


# --- deck tags --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Lesson 1", "jpdb:lesson-1"),
        # Anki nests tags on "::" and splits them on whitespace, so a colon and
        # the spaces around it cannot survive into the tag.
        ("Textbook Vol. 1: Lesson 1", "jpdb:textbook-vol-1-lesson-1"),
        ("語彙・基礎", "jpdb:語彙-基礎"),  # Japanese survives; the separator does not
        ("  padded  ", "jpdb:padded"),
        ("New deck #1", "jpdb:new-deck-1"),
        ("ＡＢＣ", "jpdb:abc"),  # NFKC, like every other header/name in janki
        ("!!!", "jpdb"),  # nothing survived; still a usable tag
    ],
)
def test_deck_tag_flattens_a_deck_name_into_something_anki_accepts(
    name: str, expected: str
) -> None:
    tag = deck_tag(name)

    assert tag == expected
    assert " " not in tag and "\t" not in tag
    assert "::" not in tag


def test_a_record_is_tagged_jpdb_and_its_deck() -> None:
    assert _record().tags == ["jpdb", "jpdb:lesson-1"]


# --- choosing decks ---------------------------------------------------------


def test_decks_are_matched_by_name_ignoring_case_and_padding() -> None:
    decks = [{"id": 1, "name": "Lesson 1"}, {"id": 2, "name": "語彙"}]

    assert select_decks(decks, ["  lesson 1 "]) == [{"id": 1, "name": "Lesson 1"}]
    assert select_decks(decks, ["語彙", "Lesson 1"]) == [
        {"id": 2, "name": "語彙"},
        {"id": 1, "name": "Lesson 1"},
    ]


def test_naming_a_deck_that_does_not_exist_lists_the_ones_that_do() -> None:
    decks = [{"id": 1, "name": "Lesson 1"}, {"id": 2, "name": "Lesson 2"}]

    with pytest.raises(JpdbImportError) as excinfo:
        select_decks(decks, ["Lesson 3"])

    message = str(excinfo.value)
    assert "'Lesson 3'" in message
    assert "Lesson 1, Lesson 2" in message


def test_the_same_deck_named_twice_is_imported_once() -> None:
    decks = [{"id": 1, "name": "Lesson 1"}]

    assert select_decks(decks, ["Lesson 1", "lesson 1"]) == [{"id": 1, "name": "Lesson 1"}]


# --- importing a deck over the API ------------------------------------------


def test_importing_a_deck_walks_decks_then_pairs_then_a_batched_lookup() -> None:
    client, transport = _api()
    deck = {"id": 1, "name": "Textbook Vol. 1: Lesson 1"}

    result = import_deck(client, deck)

    assert transport.endpoints() == ["deck/list-vocabulary", "lookup-vocabulary"]
    assert [record.id for record in result.records] == [
        "word:日本語:にほんご",
        "word:話す:はなす",
        "word:たべる:たべる",
    ]
    assert result.warnings == []
    assert result.needs_reading == []


def test_every_imported_record_carries_its_deck_as_the_source_ref() -> None:
    # run_import's cross-module contract: source_ref must equal what the
    # importer wrote as source.imported_from, or `status --rebuild` appends a
    # near-duplicate reference to every record this import touched.
    client, _ = _api()
    deck = {"id": 1, "name": "Textbook Vol. 1: Lesson 1"}

    result = import_deck(client, deck)

    assert {record.source.imported_from for record in result.records} == {
        "Textbook Vol. 1: Lesson 1"
    }
    assert {record.source.type for record in result.records} == {"jpdb"}


def test_a_lookup_is_batched_and_asks_only_about_the_deck_it_was_given() -> None:
    client, transport = _api()

    import_deck(client, {"id": 1, "name": "Lesson 1"}, batch_size=2)

    lookups = [call.body for call in transport.calls if call.url.endswith("lookup-vocabulary")]
    assert [len(body["list"]) for body in lookups] == [2, 1]


def test_a_pair_listed_twice_in_a_deck_is_looked_up_once() -> None:
    decks = _fixture(DECKS_FIXTURE)
    decks["deck_vocabulary"]["1"]["vocabulary"] = [
        [1464530, 3361009543],
        [1464530, 3361009543],
        [1562350, 4280520068],
    ]
    transport = RoutingTransport(decks, _fixture(LOOKUP_FIXTURE))

    result = import_deck(_client(transport), {"id": 1, "name": "Lesson 1"})

    lookup = next(call for call in transport.calls if call.url.endswith("lookup-vocabulary"))
    assert lookup.body["list"] == [[1464530, 3361009543], [1562350, 4280520068]]
    assert len(result.records) == 2


def test_an_empty_deck_imports_nothing_without_asking_about_no_words() -> None:
    client, transport = _api()

    result = import_deck(client, {"id": 3, "name": "New deck #1"})

    assert result.records == []
    # An empty `list` would still be a request, and jpdb counts requests.
    assert "lookup-vocabulary" not in transport.endpoints()


def test_a_deck_with_no_name_is_an_error_rather_than_an_unlabelled_import() -> None:
    client, _ = _api()

    with pytest.raises(JpdbImportError):
        import_deck(client, {"id": 1, "name": "   "})


# --- both hold classes, on the API path too ---------------------------------


def _held_deck(reading: str) -> tuple[Any, dict[str, Any]]:
    """A one-word deck whose single entry has the given reading."""
    decks = {
        "list_user_decks": {"decks": [[1, "Lesson 1", 1]]},
        "deck_vocabulary": {"1": {"vocabulary": [[1562350, 4280520068]]}},
    }
    lookup = {
        "request_list": [[1562350, 4280520068]],
        "response": {
            "vocabulary_info": [
                [
                    "話す",
                    reading,
                    200,
                    ["LHLL"],
                    [["to speak"]],
                    [["vt"]],
                    ["vt", "v5", "v5s"],
                    None,
                ]
            ]
        },
    }
    transport = RoutingTransport(decks, lookup)
    return _client(transport), {"id": 1, "name": "Lesson 1"}


def test_an_entry_with_no_reading_is_held_not_imported() -> None:
    client, deck = _held_deck("")

    result = import_deck(client, deck)

    assert result.records == []
    assert [record.id for record in result.needs_reading] == ["word:話す:"]
    assert result.needs_reading[0].source.raw_fields["hold_reason"] == "missing reading"
    assert "no reading" in result.warnings[0]


def test_an_entry_whose_reading_is_itself_kanji_is_held_too() -> None:
    # word:話す:話す looks well formed and is permanently invalid, so the
    # empty-reading check cannot be the only one.
    client, deck = _held_deck("話す")

    result = import_deck(client, deck)

    assert result.records == []
    assert [record.id for record in result.needs_reading] == ["word:話す:話す"]
    assert result.needs_reading[0].source.raw_fields["hold_reason"] == "reading contains kanji"
    assert "written in kanji" in result.warnings[0]


def test_a_held_entry_gets_no_romaji_transliterated_from_its_kanji() -> None:
    client, deck = _held_deck("話す")

    result = import_deck(client, deck)

    assert result.needs_reading[0].romaji == ""


# --- the userscript CSV path ------------------------------------------------

USERSCRIPT_CSV = "spelling,furigana reading,meaning\n話す,はなす,to speak\n電話,でんわ,telephone\n"


def test_the_userscript_csv_columns_map_without_any_jpdb_specific_reader(
    tmp_path: Path,
) -> None:
    source = tmp_path / "jpdb-export.csv"
    source.write_text(USERSCRIPT_CSV, encoding="utf-8")

    result = import_csv(source)

    assert [record.id for record in result.records] == ["word:話す:はなす", "word:電話:でんわ"]
    assert result.records[0].meanings == ["to speak"]
    assert result.records[0].romaji == "hanasu"
    assert result.records[0].source.type == "jpdb"
    assert result.records[0].source.imported_from == "jpdb-export.csv"
    assert result.records[0].tags == ["jpdb"]


def test_a_plain_reading_header_maps_as_well_as_the_furigana_one(tmp_path: Path) -> None:
    # The userscript's exact header could not be verified from this
    # environment, so both spellings map and neither is load-bearing.
    source = tmp_path / "jpdb-export.csv"
    source.write_text("spelling,reading\n話す,はなす\n", encoding="utf-8")

    assert [record.id for record in import_csv(source).records] == ["word:話す:はなす"]


def test_a_csv_row_with_no_usable_reading_is_held_like_any_other(tmp_path: Path) -> None:
    source = tmp_path / "jpdb-export.csv"
    source.write_text("spelling,reading\n話す,\n", encoding="utf-8")

    result = import_csv(source)

    assert result.records == []
    assert [record.id for record in result.needs_reading] == ["word:話す:"]


# --- the CLI ----------------------------------------------------------------


def _project(tmp_path: Path) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    return tmp_path


def _stored(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    return {record["id"]: record for record in payload}


def _patch_api(monkeypatch: pytest.MonkeyPatch, transport: RoutingTransport) -> None:
    monkeypatch.setenv("JPDB_API_KEY", "test-key")
    monkeypatch.setattr(
        cli.jpdb, "JpdbClient", lambda key, *a, **kw: _client(transport)
    )


def test_import_jpdb_deck_lands_records_the_ledger_and_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)
    transport = RoutingTransport(_fixture(DECKS_FIXTURE), _fixture(LOOKUP_FIXTURE))
    _patch_api(monkeypatch, transport)

    code = cli.main(
        ["--root", str(root), "import-jpdb", "--deck", "Textbook Vol. 1: Lesson 1"]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "Imported 3 jpdb entries" in out
    assert "3 added" in out
    stored = _stored(root)
    assert set(stored) == {"word:日本語:にほんご", "word:話す:はなす", "word:たべる:たべる"}
    assert stored["word:話す:はなす"]["source"]["imported_from"] == "Textbook Vol. 1: Lesson 1"
    assert stored["word:話す:はなす"]["tags"] == ["jpdb", "jpdb:textbook-vol-1-lesson-1"]


def test_the_ledger_reference_matches_the_records_own_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The two must agree, or `status --rebuild` appends a second near-duplicate
    # reference to every record this import just wrote.
    root = _project(tmp_path)
    transport = RoutingTransport(_fixture(DECKS_FIXTURE), _fixture(LOOKUP_FIXTURE))
    _patch_api(monkeypatch, transport)

    cli.main(["--root", str(root), "import-jpdb", "--deck", "Textbook Vol. 1: Lesson 1"])

    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    entry = book["records"]["word:話す:はなす"]
    assert [
        {"type": source["type"], "ref": source["ref"]} for source in entry["sources"]
    ] == [{"type": "jpdb", "ref": "Textbook Vol. 1: Lesson 1"}]
    stored = _stored(root)["word:話す:はなす"]
    assert stored["source"]["imported_from"] == entry["sources"][0]["ref"]


def test_all_decks_imports_each_deck_under_its_own_source_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)
    decks = _fixture(DECKS_FIXTURE)
    lookup = _fixture(LOOKUP_FIXTURE)
    # Deck 2 holds two more words; give the fixture entries for them.
    lookup["request_list"] += [[1358340, 1564309720], [1002430, 3202573393]]
    lookup["response"]["vocabulary_info"] += [
        ["食べ物", "たべもの", 2600, ["LHLLL"], [["food"]], [["n"]], ["n"], None],
        ["お茶", "おちゃ", 1600, ["LHHH"], [["tea"]], [["n"]], ["n"], None],
    ]
    _patch_api(monkeypatch, RoutingTransport(decks, lookup))

    assert cli.main(["--root", str(root), "import-jpdb", "--all-decks"]) == 0

    out = capsys.readouterr().out
    assert "jpdb deck: Textbook Vol. 1: Lesson 1" in out
    assert "jpdb deck: 語彙・基礎" in out
    stored = _stored(root)
    assert stored["word:話す:はなす"]["source"]["imported_from"] == "Textbook Vol. 1: Lesson 1"
    assert stored["word:お茶:おちゃ"]["source"]["imported_from"] == "語彙・基礎"
    assert stored["word:お茶:おちゃ"]["tags"] == ["jpdb", "jpdb:語彙-基礎"]


def test_a_word_in_two_decks_earns_a_ledger_reference_for_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    decks = _fixture(DECKS_FIXTURE)
    # The same word in both decks — the ordinary case for a jpdb account.
    decks["deck_vocabulary"]["2"] = {"vocabulary": [[1562350, 4280520068]]}
    _patch_api(monkeypatch, RoutingTransport(decks, _fixture(LOOKUP_FIXTURE)))

    assert cli.main(["--root", str(root), "import-jpdb", "--all-decks"]) == 0

    entry = json.loads((root / "ledger.json").read_text(encoding="utf-8"))["records"][
        "word:話す:はなす"
    ]
    assert [source["ref"] for source in entry["sources"]] == [
        "Textbook Vol. 1: Lesson 1",
        "語彙・基礎",
    ]
    # The record itself keeps the first deck that supplied it; the merge is
    # existing-wins, and the ledger is where later sightings live.
    stored = _stored(root)["word:話す:はなす"]
    assert stored["source"]["imported_from"] == "Textbook Vol. 1: Lesson 1"


def test_held_entries_go_to_a_staging_file_named_after_the_deck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)
    decks = {
        "list_user_decks": {"decks": [[1, "Textbook Vol. 1: Lesson 1", 1]]},
        "deck_vocabulary": {"1": {"vocabulary": [[1562350, 4280520068]]}},
    }
    lookup = {
        "request_list": [[1562350, 4280520068]],
        "response": {
            "vocabulary_info": [
                ["話す", "", 200, ["LHLL"], [["to speak"]], [["vt"]], ["vt", "v5", "v5s"], None]
            ]
        },
    }
    _patch_api(monkeypatch, RoutingTransport(decks, lookup))

    assert cli.main(["--root", str(root), "import-jpdb", "--all-decks"]) == 0

    staged_files = list((root / "staging").glob("*.yaml"))
    assert len(staged_files) == 1
    staged = staged_files[0]
    # Readable-deck-name prefix, then a fingerprint of the full name so two
    # decks whose names flatten to the same slug cannot share one file.
    assert staged.name.startswith("jpdb-textbook-vol-1-lesson-1-")
    assert staged.name.endswith("-needs-reading.yaml")
    held = yaml.safe_load(staged.read_text(encoding="utf-8"))
    assert [record["id"] for record in held["records"]] == ["word:話す:"]
    out = capsys.readouterr().out
    # "row(s)" would be a lie about an API result.
    assert "entr(y/ies)" in out
    assert str(staged) in out


def test_import_jpdb_csv_routes_through_the_same_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)
    source = tmp_path / "jpdb-export.csv"
    source.write_text(USERSCRIPT_CSV, encoding="utf-8")

    assert cli.main(["--root", str(root), "import-jpdb", str(source)]) == 0

    assert "Imported 2 source rows" in capsys.readouterr().out
    stored = _stored(root)
    assert stored["word:話す:はなす"]["source"]["imported_from"] == "jpdb-export.csv"


@pytest.mark.parametrize(
    "extra",
    [
        [],  # no source at all
        ["--all-decks", "--deck", "Lesson 1"],
        ["FILE.csv", "--all-decks"],
    ],
)
def test_import_jpdb_needs_exactly_one_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    root = _project(tmp_path)

    assert cli.main(["--root", str(root), "import-jpdb", *extra]) == 1

    assert "exactly one source" in capsys.readouterr().err


def test_a_named_deck_that_does_not_exist_fails_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)
    transport = RoutingTransport(_fixture(DECKS_FIXTURE), _fixture(LOOKUP_FIXTURE))
    _patch_api(monkeypatch, transport)

    assert cli.main(["--root", str(root), "import-jpdb", "--deck", "Lesson 9"]) == 1

    assert "No jpdb deck named 'Lesson 9'" in capsys.readouterr().err
    assert not (root / "vocabulary.json").exists()


# --- --replace across several decks -----------------------------------------


def _replace_project(tmp_path: Path) -> Path:
    """A project holding one record --replace would discard."""
    root = _project(tmp_path)
    (root / "vocabulary.json").write_text(
        json.dumps(
            [{"id": "word:古い:ふるい", "expression": "古い", "reading": "ふるい"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def test_replace_discards_the_existing_records_on_a_single_deck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _replace_project(tmp_path)
    _patch_api(monkeypatch, RoutingTransport(_fixture(DECKS_FIXTURE), _fixture(LOOKUP_FIXTURE)))

    code = cli.main(
        [
            "--root",
            str(root),
            "import-jpdb",
            "--deck",
            "Textbook Vol. 1: Lesson 1",
            "--replace",
            "--yes",
        ]
    )

    assert code == 0
    assert "word:古い:ふるい" not in _stored(root)


def test_replace_applies_to_the_first_deck_only_so_deck_two_keeps_deck_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The rule that makes --replace mean "replace the collection with these
    # decks": honoured on every pass, deck two would erase deck one.
    root = _replace_project(tmp_path)
    decks = _fixture(DECKS_FIXTURE)
    lookup = _fixture(LOOKUP_FIXTURE)
    lookup["request_list"] += [[1358340, 1564309720], [1002430, 3202573393]]
    lookup["response"]["vocabulary_info"] += [
        ["食べ物", "たべもの", 2600, ["LHLLL"], [["food"]], [["n"]], ["n"], None],
        ["お茶", "おちゃ", 1600, ["LHHH"], [["tea"]], [["n"]], ["n"], None],
    ]
    _patch_api(monkeypatch, RoutingTransport(decks, lookup))

    code = cli.main(
        ["--root", str(root), "import-jpdb", "--all-decks", "--replace", "--yes"]
    )

    assert code == 0
    stored = _stored(root)
    assert "word:古い:ふるい" not in stored
    assert "word:話す:はなす" in stored
    assert "word:お茶:おちゃ" in stored


def test_declining_the_replace_prompt_imports_no_deck_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # "No" means no. Continuing to deck two would leave a collection missing
    # exactly the deck the user was asked about, under a summary that reads
    # like a complete import.
    root = _replace_project(tmp_path)
    _patch_api(monkeypatch, RoutingTransport(_fixture(DECKS_FIXTURE), _fixture(LOOKUP_FIXTURE)))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    code = cli.main(
        ["--root", str(root), "import-jpdb", "--all-decks", "--replace"]
    )

    assert code == 1
    assert _stored(root) == {
        "word:古い:ふるい": _stored(root)["word:古い:ふるい"]
    }
    err = capsys.readouterr().err
    assert "2 deck(s) not imported" in err


def test_two_decks_whose_names_differ_only_in_case_are_not_guessed_between(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Names are matched case-insensitively, so this collision is invisible to
    # the matcher; picking either would import the wrong deck's words.
    root = _project(tmp_path)
    decks = _fixture(DECKS_FIXTURE)
    decks["list_user_decks"]["decks"].append([4, "textbook vol. 1: lesson 1", 0])
    decks["deck_vocabulary"]["4"] = {"vocabulary": []}
    _patch_api(monkeypatch, RoutingTransport(decks, _fixture(LOOKUP_FIXTURE)))

    code = cli.main(
        ["--root", str(root), "import-jpdb", "--deck", "Textbook Vol. 1: Lesson 1"]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "matches more than one jpdb deck" in err
    assert not (root / "vocabulary.json").exists()


def test_two_decks_whose_names_flatten_alike_get_separate_staging_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 'Lesson 1' and 'Lesson: 1' both slug to lesson-1. Sharing one staging
    # file makes the second deck's held rows permanently unstageable: it is
    # told the file exists and to re-run, and re-running hands it to the first
    # deck again.
    root = _project(tmp_path)
    decks = {
        "list_user_decks": {"decks": [[1, "Lesson 1", 1], [2, "Lesson: 1", 1]]},
        "deck_vocabulary": {
            "1": {"vocabulary": [[1562350, 4280520068]]},
            "2": {"vocabulary": [[1358340, 1564309720]]},
        },
    }
    lookup = {
        "request_list": [[1562350, 4280520068], [1358340, 1564309720]],
        "response": {
            "vocabulary_info": [
                ["話す", "", 200, ["LHLL"], [["to speak"]], [["vt"]], ["v5s"], None],
                ["食べ物", "", 2600, ["LHLLL"], [["food"]], [["n"]], ["n"], None],
            ]
        },
    }
    _patch_api(monkeypatch, RoutingTransport(decks, lookup))

    assert cli.main(["--root", str(root), "import-jpdb", "--all-decks"]) == 0

    staged = sorted((root / "staging").glob("*.yaml"))
    assert len(staged) == 2
    held = [
        yaml.safe_load(path.read_text(encoding="utf-8"))["records"][0]["expression"]
        for path in staged
    ]
    assert sorted(held) == ["話す", "食べ物"]


def test_the_staging_filename_is_stable_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fingerprint is of the deck name alone, so the same deck finds the
    # same file next month rather than littering staging with near-duplicates.
    root = _project(tmp_path)
    decks = {
        "list_user_decks": {"decks": [[1, "Lesson 1", 1]]},
        "deck_vocabulary": {"1": {"vocabulary": [[1562350, 4280520068]]}},
    }
    lookup = {
        "request_list": [[1562350, 4280520068]],
        "response": {
            "vocabulary_info": [
                ["話す", "", 200, ["LHLL"], [["to speak"]], [["vt"]], ["v5s"], None]
            ]
        },
    }
    _patch_api(monkeypatch, RoutingTransport(decks, lookup))

    cli.main(["--root", str(root), "import-jpdb", "--all-decks"])
    first = [path.name for path in (root / "staging").glob("*.yaml")]
    cli.main(["--root", str(root), "import-jpdb", "--all-decks"])

    assert [path.name for path in (root / "staging").glob("*.yaml")] == first
