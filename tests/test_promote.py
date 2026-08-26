"""Promoting a reviewed staging file into the records janki builds from.

No network (IMPLEMENTATION_PLAN rule 6): jpdb is driven through a fake
transport, so the three-outcome reading check is exercised for real rather
than stubbed at the decision.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread, current_thread
from typing import Any

import pytest
import yaml

from japanese_anki import cli, enrich, extract, patterns, promote
from japanese_anki import staging as staging_module
from japanese_anki.application import promotion as promotion_application
from japanese_anki.config import ProjectConfig
from japanese_anki.io import load_records
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


def parsed_candidate(**overrides: Any) -> Any:
    """One complete value as returned by extraction's Pydantic schema."""
    values: dict[str, Any] = {
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "part_of_speech": "verb",
        "examples": [
            {
                "japanese": "日本語を話します。",
                "speech_level": "polite",
                "furigana": "日本語[にほんご]を 話[はな]します。",
                "romaji": "nihongo o hanashimasu.",
                "english": "I speak Japanese.",
            },
            {
                "japanese": "あとで話そう。",
                "speech_level": "casual",
                "furigana": "あとで 話[はな]そう。",
                "romaji": "ato de hanasou.",
                "english": "Let's talk later.",
            },
        ],
        "usage_notes": "",
        "page": 1,
        "context": "話す　はなす　to speak",
        "confidence": "high",
        "inclusion_reason": "introduced here",
        "source_kind": "prose",
        "section": "",
        "ordinal": 0,
    }
    values.update(overrides)
    candidate_type = extract.candidate_schema().model_fields[
        "candidates"
    ].annotation.__args__[0]
    return candidate_type(**values)


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


def test_an_unspeakable_pitch_pattern_is_named_rather_than_promoted_silently() -> None:
    """The third door, and the one nothing was watching.

    `enrich --jpdb` refuses a pattern it cannot render and `import-jpdb` names
    one, but `pitch_accent` is a first-class staging field: a hand-written row
    could put an unspeakable pattern into the collection, and the first anyone
    hears of it is the next `janki audio` run.

    Named, not held: an accent is not identity, and holding a whole row over
    one would be out of proportion to a clip that still gets made.
    """
    # はし is two kana, so three characters is the right length. `LHLL` is four.
    staged = record(
        id="word:はし:はし", expression="はし", reading="はし", pitch_accent=["LHLL"]
    )

    result = check_readings([staged], skip_reading_check=True)

    assert [item.id for item in result.promoted] == ["word:はし:はし"]
    [warning] = result.warnings
    assert "pitch_accent[0] LHLL" in warning, "named by the field and entry"
    assert "engine's own accent" in warning


def test_the_length_check_is_not_the_whole_test_at_promote_either() -> None:
    """`_unspeakable_patterns` renders; it does not measure.

    `LHHHHH` fits かんーぱい exactly — five kana, six characters — and still
    cannot be spoken, because `ン` ends on no vowel for the `ー` to repeat. A
    length-only check passes it, and the first version of the test above used a
    four-kana reading by mistake, so a length-only mutation survived the entire
    suite.
    """
    staged = record(
        id="word:かんーぱい:かんーぱい",
        expression="かんーぱい",
        reading="かんーぱい",
        pitch_accent=["LHHHHH"],
    )

    result = check_readings([staged], skip_reading_check=True)

    assert result.promoted, "named, not held"
    [warning] = result.warnings
    assert "LHHHHH" in warning
    assert "cannot be spoken" in warning


def test_promote_checks_the_pattern_that_actually_reaches_the_synthesizer() -> None:
    """`audio_accent` first, because `select_pattern` reads it first.

    Two failures came from checking `pitch_accent` alone. An unspeakable
    `audio_accent` — the *more* hand-written field, since no importer writes it
    and enrichment does not touch it, so a staging file is its only door —
    passed promote with nothing said at all. And a record with a good
    `audio_accent` and a bad `pitch_accent` was told its clip fell back to the
    engine's own accent when the clip was forced perfectly well: the same
    defect, in the same words, that the round before this one fixed in the
    importer.
    """
    hashi = {"id": "word:はし:はし", "expression": "はし", "reading": "はし"}
    unspeakable_choice = record(**hashi, audio_accent="LHLL")
    good_choice = record(**hashi, audio_accent="LHL", pitch_accent=["LHLL"])

    silent = check_readings([unspeakable_choice], skip_reading_check=True)
    misblamed = check_readings([good_choice], skip_reading_check=True)

    [silent_warning] = silent.warnings
    assert "audio_accent LHLL" in silent_warning, "checked, and named by its field"
    assert "engine's own accent" in silent_warning

    [misblamed_warning] = misblamed.warnings
    assert "pitch_accent[0] LHLL" in misblamed_warning, "still reported"
    assert "uses LHL, which is fine" in misblamed_warning
    assert "engine's own" not in misblamed_warning


def test_a_lower_case_accent_is_compared_in_its_canonical_form() -> None:
    """`pitch._LEVELS` accepts `h`/`l`, and `select_pattern` upper-cases.

    So a record whose accent is typed in lower case renders perfectly well, and
    the message about *which* pattern the clip uses has to compare the two in
    the same form. Comparing raw strings reported the chosen pattern as fine
    when it was the broken one — measured, and it survived the whole suite.
    """
    staged = record(
        id="word:はし:はし", expression="はし", reading="はし", pitch_accent=["lhll"]
    )

    result = check_readings([staged], skip_reading_check=True)

    [warning] = result.warnings
    assert "lhll" in warning, "reported as the curator typed it"
    assert "engine's own accent" in warning, "and recognised as the chosen one"


def test_one_pattern_in_two_spellings_is_reported_once() -> None:
    """`audio_accent: lhll` with `pitch_accent: [LHLL]` is one pattern.

    The dedup was an exact-string comparison while everything around it — the
    selection, the fingerprints, the chosen-pattern check — normalises, so the
    same pattern twice read as two problems.
    """
    staged = record(
        id="word:はし:はし",
        expression="はし",
        reading="はし",
        audio_accent="lhll",
        pitch_accent=["LHLL"],
    )

    result = check_readings([staged], skip_reading_check=True)

    [warning] = result.warnings
    assert warning.count("LHLL") + warning.count("lhll") == 1, warning
    # And it is `audio_accent`'s spelling that survives, not `pitch_accent`'s.
    # Building the list the other way round leaves the count at one and sends
    # the curator to the field the synthesizer does not use — which is the
    # defect this whole check exists to fix, one level down.
    assert "audio_accent lhll" in warning, warning


def test_every_unusable_pattern_is_named_not_just_the_first() -> None:
    """Two broken patterns in two fields, and both have to be said.

    Reporting only the first left the second silent: the curator fixes
    `audio_accent`, re-promotes, and hears about `pitch_accent` on the next
    pass — or not at all, if the row has moved on. The index is there because
    `pitch_accent` is a list and "pitch_accent is wrong" does not say which
    entry.
    """
    staged = record(
        id="word:はし:はし",
        expression="はし",
        reading="はし",
        audio_accent="LHLL",
        pitch_accent=["LHL", "LHHH"],
    )

    result = check_readings([staged], skip_reading_check=True)

    [warning] = result.warnings
    assert "audio_accent LHLL" in warning
    assert "pitch_accent[1] LHHH" in warning, "the second entry, named by index"
    assert "pitch_accent[0]" not in warning, "the usable one is not reported"


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


def test_the_cli_calls_the_shared_promotion_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command may format the result; it may not own a second writer."""
    root = project(tmp_path, [])
    path = staging_file(root, ONE_GOOD)
    calls: list[promotion_application.PromotionExecutionResult] = []
    real_execute = promotion_application.execute_promotion

    def observe(
        config: ProjectConfig, decision: promotion_application.PromotionDecision
    ) -> promotion_application.PromotionExecutionResult:
        result = real_execute(config, decision)
        calls.append(result)
        return result

    monkeypatch.setattr(promotion_application, "execute_promotion", observe)

    assert cli.main([
        "--root", str(root), "promote", str(path), "--skip-reading-check",
    ]) == 0

    assert len(calls) == 1
    assert calls[0].state == "landed"
    assert calls[0].promoted_ids == ("word:話す:はなす",)


def test_the_shared_writer_and_cli_land_identical_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A browser service call and the command commit the same transaction."""
    direct_root = tmp_path / "direct"
    cli_root = tmp_path / "cli"
    direct_root.mkdir()
    cli_root.mkdir()
    project(direct_root, [])
    project(cli_root, [])
    direct_path = staging_file(direct_root, ONE_GOOD)
    cli_path = staging_file(cli_root, ONE_GOOD)

    config = ProjectConfig.load(direct_root)
    decision = promotion_application.decide_promotion(
        config, direct_path, skip_reading_check=True
    )
    result = promotion_application.execute_promotion(config, decision)
    assert result.state == "landed"
    assert result.promoted_ids == ("word:話す:はなす",)

    assert cli.main([
        "--root", str(cli_root), "promote", str(cli_path),
        "--skip-reading-check",
    ]) == 0
    capsys.readouterr()

    direct_records = load_records(direct_root / "vocabulary.json")
    assert [item.id for item in direct_records] == ["word:話す:はなす"]
    assert direct_records == load_records(cli_root / "vocabulary.json")
    assert json.loads((direct_root / "ledger.json").read_text(encoding="utf-8")) == (
        json.loads((cli_root / "ledger.json").read_text(encoding="utf-8"))
    )
    assert read_staging(direct_root / "staging" / "done" / direct_path.name) == (
        read_staging(cli_root / "staging" / "done" / cli_path.name)
    )
    assert not direct_path.exists()
    assert not cli_path.exists()


def test_the_shared_writer_refuses_a_decision_from_another_project(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    project(first_root, [])
    project(second_root, [])
    path = staging_file(first_root, ONE_GOOD)
    first_config = ProjectConfig.load(first_root)
    decision = promotion_application.decide_promotion(
        first_config, path, skip_reading_check=True
    )
    staging_before = path.read_bytes()
    records_before = (first_root / "vocabulary.json").read_bytes()

    with pytest.raises(promote.PromoteError, match="promotion-config-mismatch"):
        promotion_application.execute_promotion(
            ProjectConfig.load(second_root), decision
        )

    assert path.read_bytes() == staging_before
    assert (first_root / "vocabulary.json").read_bytes() == records_before
    assert not (first_root / "ledger.json").exists()
    assert not (second_root / "ledger.json").exists()
    assert not (second_root / "staging" / "done").exists()


def test_the_shared_writer_refuses_an_unchecked_preview(
    tmp_path: Path,
) -> None:
    root = project(tmp_path, [])
    path = staging_file(root, ONE_GOOD)
    config = ProjectConfig.load(root)
    decision = promotion_application.decide_promotion(config, path)
    staging_before = path.read_bytes()
    records_before = (root / "vocabulary.json").read_bytes()

    with pytest.raises(promote.PromoteError, match="reading-check-required"):
        promotion_application.execute_promotion(config, decision)

    assert path.read_bytes() == staging_before
    assert (root / "vocabulary.json").read_bytes() == records_before
    assert not (root / "ledger.json").exists()
    assert not (root / "staging" / "done").exists()


def test_the_shared_result_names_a_partial_ledger_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [])
    path = staging_file(root, ONE_GOOD)
    config = ProjectConfig.load(root)
    decision = promotion_application.decide_promotion(
        config, path, skip_reading_check=True
    )
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("disk full")),
    )

    result = promotion_application.execute_promotion(config, decision)

    assert result.state == "landed_ledger_incomplete"
    assert str(result.ledger_error) == "disk full"
    assert result.archive_path is not None and result.archive_path.exists()
    assert result.promoted_ids == ("word:話す:はなす",)
    assert not path.exists()


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


def _rich_prompt_provenance(request_fingerprint: str) -> dict[str, Any]:
    return {
        "source_sha256": "1" * 64,
        "mode": "prose",
        "provider": "anthropic",
        "model": "claude-opus-5",
        "response_schema_version": extract.EXTRACTION_SCHEMA_VERSION,
        "system_prompt_fingerprint": "2" * 64,
        "style_guide_fingerprint": "3" * 64,
        "user_prompt_fingerprint": "4" * 64,
        "response_schema_fingerprint": "5" * 64,
        "request_fingerprint": request_fingerprint,
    }


def _pattern_only_meta(
    pattern_set: patterns.PatternSet,
    *,
    run_id: str = "11111111-1111-4111-8111-111111111111",
    request_fingerprint: str = "a" * 64,
    candidates: tuple[Any, ...] = (),
    candidate_accounting: dict[str, Any] | None = None,
    response_schema_version: int = extract.EXTRACTION_SCHEMA_VERSION,
) -> dict[str, Any]:
    provenance = _rich_prompt_provenance(request_fingerprint)
    provenance["response_schema_version"] = response_schema_version
    if candidate_accounting is None and response_schema_version >= 4:
        prepared = type("Prepared", (), {"origin_path": Path(pattern_set.source)})()
        candidate_accounting = extract.build_records(
            candidates, prepared
        ).candidate_accounting
    result = extract.ExtractionResult(
        candidates=candidates, source_units=(), model_reported_unit_count=0
    )
    coverage_kwargs = (
        {"candidate_accounting": candidate_accounting}
        if candidate_accounting is not None
        else {}
    )
    coverage = extract.coverage_block(
        result,
        source_sha256=provenance["source_sha256"],
        mode="prose",
        **coverage_kwargs,
    )
    bound = replace(
        patterns.with_prompt_provenance(pattern_set, provenance),
        review_run_id=run_id,
    )
    meta = {
        "source_file": pattern_set.source,
        "review_run_id": run_id,
        "prompt_provenance": provenance,
        "pattern_set": bound.to_dict(),
        "coverage": coverage,
    }
    if candidate_accounting is not None:
        meta["candidate_accounting"] = candidate_accounting
    return meta


def _completed_pattern_archive_meta(meta: dict[str, Any]) -> dict[str, Any]:
    source = str(meta["source_file"])
    proposed = patterns.PatternSet.from_dict(source, meta["pattern_set"])
    return promotion_application._pattern_only_archive_meta(
        meta, replace(proposed, reviewed=True)
    )


def accounted_extract(
    tmp_path: Path, *candidates: Any
) -> tuple[list[VocabularyRecord], dict[str, Any]]:
    prepared = type("Prepared", (), {"origin_path": tmp_path / "lesson.pdf"})()
    built = extract.build_records(candidates, prepared)
    meta = _pattern_only_meta(
        patterns.PatternSet(source="lesson.pdf", kind="lesson"),
        candidates=tuple(candidates),
        candidate_accounting=built.candidate_accounting,
    )
    return list(built.records), meta


REVIEW_RUN_A = "11111111-1111-4111-8111-111111111111"
REVIEW_RUN_B = "22222222-2222-4222-8222-222222222222"


def test_a_corrected_current_run_pattern_review_is_archived_beside_the_raw_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path = root / "staging" / "teform_song.pdf.yaml"
    path.parent.mkdir(parents=True)
    proposed = patterns.PatternSet(
        source="teform_song.pdf",
        kind="pattern",
        title="Te-form song",
        patterns=(patterns.Pattern("う・つ・る → って", "te-form rule"),),
    )
    meta = _pattern_only_meta(proposed)
    write_staging(path, [], meta)
    staged = patterns.PatternSet.from_dict(proposed.source, meta["pattern_set"])
    reviewed = replace(
        staged,
        title="Human-corrected Te-form song",
        patterns=(patterns.Pattern("う・つ・る → って", "reviewed rule"),),
        reviewed=True,
    )
    patterns.save_store(
        root / "data" / "patterns.json",
        {reviewed.source: reviewed},
    )

    code = cli.main(["--root", str(root), "promote", str(path)])

    assert code == 0
    assert not path.exists()
    archived, archived_meta = read_staging(
        root / "staging" / "done" / "teform_song.pdf.yaml"
    )
    assert archived == []
    assert archived_meta["pattern_set"] == meta["pattern_set"]
    assert archived_meta["reviewed_pattern_set"] == reviewed.to_dict()
    assert archived_meta["pattern_set"]["title"] == "Te-form song"
    assert archived_meta["reviewed_pattern_set"]["title"] == (
        "Human-corrected Te-form song"
    )
    assert "Reviewed pattern-only extraction; no records were promoted." in (
        archived_meta["review_notes"]
    )
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []
    assert not (root / "ledger.json").exists()
    output = capsys.readouterr().out
    assert "Completed reviewed pattern-only extraction" in output
    assert "Promoted 0 record" not in output


def test_a_pattern_only_extraction_waits_for_review_without_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path = root / "staging" / "teform_song.pdf.yaml"
    path.parent.mkdir(parents=True)
    proposed = patterns.PatternSet(
        source="teform_song.pdf",
        kind="pattern",
        patterns=(patterns.Pattern("う・つ・る → って", "te-form rule"),),
    )
    meta = _pattern_only_meta(proposed)
    write_staging(path, [], meta)
    staged = patterns.PatternSet.from_dict(proposed.source, meta["pattern_set"])
    patterns.save_store(
        root / "data" / "patterns.json",
        {proposed.source: staged},
    )
    before = path.read_bytes()

    code = cli.main(["--root", str(root), "promote", str(path)])

    assert code == 1
    assert path.read_bytes() == before
    assert not (root / "staging" / "done").exists()
    assert "patterns-unreviewed" in capsys.readouterr().err


def test_an_identical_older_reviewed_run_cannot_complete_a_fresh_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path = root / "staging" / "teform_song.pdf.yaml"
    path.parent.mkdir(parents=True)
    proposed = patterns.PatternSet(
        source="teform_song.pdf",
        kind="pattern",
        title="Fresh answer",
        patterns=(patterns.Pattern("う・つ・る → って", "te-form rule"),),
    )
    meta = _pattern_only_meta(proposed)
    write_staging(path, [], meta)
    staged = patterns.PatternSet.from_dict(proposed.source, meta["pattern_set"])
    old = replace(
        staged,
        reviewed=True,
        review_run_id=REVIEW_RUN_B,
    )
    patterns.save_store(root / "data" / "patterns.json", {old.source: old})
    before = path.read_bytes()

    code = cli.main(["--root", str(root), "promote", str(path)])

    assert code == 1
    assert path.read_bytes() == before
    assert not (root / "staging" / "done").exists()
    assert "patterns-review-stale" in capsys.readouterr().err


def test_an_empty_schema_v2_extraction_keeps_the_legacy_noop_boundary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path = root / "staging" / "legacy.yaml"
    path.parent.mkdir(parents=True)
    provenance = _rich_prompt_provenance("a" * 64)
    provenance["response_schema_version"] = 2
    del provenance["response_schema_fingerprint"]
    del provenance["request_fingerprint"]
    write_staging(
        path,
        [],
        {"source_file": "legacy.pdf", "prompt_provenance": provenance},
    )

    assert cli.main(["--root", str(root), "promote", str(path)]) == 0

    assert path.exists()
    assert not (root / "staging" / "done").exists()
    assert "holds no records" in capsys.readouterr().out


@pytest.mark.parametrize(
    "defect",
    [
        "missing-coverage",
        "missing-run",
        "missing-pattern-run",
        "mismatched-pattern-run",
        "missing-request-fingerprint",
    ],
)
def test_incomplete_schema_v3_pattern_only_staging_is_never_archived(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    defect: str,
) -> None:
    root = project(tmp_path, [])
    path = root / "staging" / "teform_song.pdf.yaml"
    path.parent.mkdir(parents=True)
    proposed = patterns.PatternSet(source="teform_song.pdf", kind="pattern")
    meta = _pattern_only_meta(proposed)
    if defect == "missing-coverage":
        del meta["coverage"]
    elif defect == "missing-run":
        del meta["review_run_id"]
    elif defect == "missing-pattern-run":
        del meta["pattern_set"]["review_run_id"]
    elif defect == "mismatched-pattern-run":
        meta["pattern_set"]["review_run_id"] = REVIEW_RUN_B
    elif defect == "missing-request-fingerprint":
        del meta["prompt_provenance"]["request_fingerprint"]
        del meta["pattern_set"]["prompt_provenance"]["request_fingerprint"]
    write_staging(path, [], meta)
    before = path.read_bytes()

    assert cli.main(["--root", str(root), "promote", str(path)]) == 1

    assert path.read_bytes() == before
    assert not (root / "staging" / "done").exists()
    assert "invalid" in capsys.readouterr().err


def test_a_malformed_v3_run_id_is_refused_before_archiving(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path = root / "staging" / "teform_song.pdf.yaml"
    path.parent.mkdir(parents=True)
    meta = _pattern_only_meta(
        patterns.PatternSet(source="teform_song.pdf", kind="pattern")
    )
    meta["review_run_id"] = "not-a-uuid"
    path.write_text(
        yaml.safe_dump({**meta, "records": []}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    before = path.read_bytes()

    assert cli.main(["--root", str(root), "promote", str(path)]) == 1

    assert path.read_bytes() == before
    assert not (root / "staging" / "done").exists()
    assert "review-run-id-invalid" in capsys.readouterr().err


def _stage_reviewed_pattern_run(
    root: Path,
    *,
    run_id: str,
    request_fingerprint: str = "a" * 64,
    title: str = "Te-form song",
) -> tuple[Path, dict[str, Any], patterns.PatternSet]:
    path = root / "staging" / "teform_song.pdf.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    proposed = patterns.PatternSet(
        source="teform_song.pdf",
        kind="pattern",
        title=title,
        patterns=(patterns.Pattern("う・つ・る → って", "te-form rule"),),
    )
    meta = _pattern_only_meta(
        proposed,
        run_id=run_id,
        request_fingerprint=request_fingerprint,
    )
    write_staging(path, [], meta)
    reviewed = replace(
        patterns.PatternSet.from_dict(proposed.source, meta["pattern_set"]),
        reviewed=True,
    )
    patterns.save_store(
        root / "data" / "patterns.json", {reviewed.source: reviewed}
    )
    return path, meta, reviewed


def test_two_zero_record_runs_with_one_basename_get_separate_archives(
    tmp_path: Path,
) -> None:
    root = project(tmp_path, [])
    path, _first_meta, _first_review = _stage_reviewed_pattern_run(
        root, run_id=REVIEW_RUN_A
    )
    command = ["--root", str(root), "promote", str(path)]
    assert cli.main(command) == 0

    path, _second_meta, _second_review = _stage_reviewed_pattern_run(
        root,
        run_id=REVIEW_RUN_B,
        # Same request on purpose: the run id, not model inputs, separates reviews.
        request_fingerprint="a" * 64,
    )
    assert cli.main(command) == 0

    archives = list((root / "staging" / "done").glob("teform_song.pdf*.yaml"))
    assert len(archives) == 2
    runs = {read_staging(archive)[1]["review_run_id"] for archive in archives}
    assert runs == {REVIEW_RUN_A, REVIEW_RUN_B}


def test_an_exact_pattern_archive_retry_does_not_overwrite_the_archive(
    tmp_path: Path,
) -> None:
    root = project(tmp_path, [])
    path, meta, _reviewed = _stage_reviewed_pattern_run(
        root, run_id=REVIEW_RUN_A
    )
    command = ["--root", str(root), "promote", str(path)]
    assert cli.main(command) == 0
    archive = root / "staging" / "done" / path.name
    before = archive.read_bytes()

    # The archive landed but the process died before deleting the live review.
    write_staging(path, [], meta)
    assert cli.main(command) == 0

    assert not path.exists()
    assert archive.read_bytes() == before


def test_a_divergent_same_run_pattern_archive_is_refused_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    path, meta, _reviewed = _stage_reviewed_pattern_run(
        root, run_id=REVIEW_RUN_A
    )
    command = ["--root", str(root), "promote", str(path)]
    assert cli.main(command) == 0
    archive = root / "staging" / "done" / path.name
    rows, archived_meta = read_staging(archive)
    archived_meta["review_notes"] += "\nA note added after completion."
    write_staging(archive, rows, archived_meta, force=True)
    archive_before = archive.read_bytes()
    write_staging(path, [], meta)
    live_before = path.read_bytes()

    assert cli.main(command) == 1

    assert path.read_bytes() == live_before
    assert archive.read_bytes() == archive_before
    assert "pattern-archive-divergent" in capsys.readouterr().err


def test_a_completed_schema_v3_pattern_archive_cannot_grow_record_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty done archive is present evidence, not an absent record archive."""
    root = project(tmp_path, [])
    path = root / "staging" / "teform_song.pdf.yaml"
    path.parent.mkdir(parents=True)
    proposed = patterns.PatternSet(
        source="teform_song.pdf",
        kind="pattern",
        patterns=(patterns.Pattern("う・つ・る → って", "te-form rule"),),
    )
    meta = _pattern_only_meta(proposed, response_schema_version=3)
    write_staging(path, [], meta)
    reviewed = replace(
        patterns.PatternSet.from_dict(proposed.source, meta["pattern_set"]),
        reviewed=True,
    )
    patterns.save_store(root / "data" / "patterns.json", {reviewed.source: reviewed})
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]

    assert cli.main(command) == 0
    archive = root / "staging" / "done" / path.name
    archive_before = archive.read_bytes()

    # A later hand edit recreated the live half of the same paid run but added
    # a record. It must not turn the already-completed zero-row review into a
    # record archive or erase its reviewed_pattern_set snapshot.
    write_staging(path, [record()], meta)
    live_before = path.read_bytes()

    assert cli.main(command) == 1

    assert path.read_bytes() == live_before
    assert archive.read_bytes() == archive_before
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []
    assert not (root / "ledger.json").exists()
    assert "record-archive-divergent" in capsys.readouterr().err


def test_a_pattern_archive_write_failure_keeps_the_live_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = project(tmp_path, [])
    path, _meta, _reviewed = _stage_reviewed_pattern_run(
        root, run_id=REVIEW_RUN_A
    )
    before = path.read_bytes()
    def fail_archive(*args: Any, **kwargs: Any) -> Path:
        target = Path(args[0])
        if target.parent.name == "done":
            raise cli.StagingError("archive write failed")
        raise AssertionError(f"unexpected unlocked archive target: {target}")

    monkeypatch.setattr(
        promotion_application, "write_staging_under_lock", fail_archive
    )

    assert cli.main(["--root", str(root), "promote", str(path)]) == 1

    assert path.read_bytes() == before
    assert not (root / "staging" / "done" / path.name).exists()
    assert "archive write failed" in capsys.readouterr().err


def test_a_concurrent_forced_replacement_is_not_deleted_as_the_review_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = project(tmp_path, [])
    path, _meta, _reviewed = _stage_reviewed_pattern_run(
        root, run_id=REVIEW_RUN_A
    )
    replacement = _pattern_only_meta(
        patterns.PatternSet(source="teform_song.pdf", kind="pattern"),
        run_id=REVIEW_RUN_B,
        request_fingerprint="b" * 64,
    )
    real_complete = promotion_application._complete_pattern_only_review

    def replace_before_completion(*args: Any, **kwargs: Any) -> tuple[Path, bool]:
        write_staging(path, [], replacement, force=True)
        return real_complete(*args, **kwargs)

    monkeypatch.setattr(
        promotion_application,
        "_complete_pattern_only_review",
        replace_before_completion,
    )

    assert cli.main(["--root", str(root), "promote", str(path)]) == 1

    _rows, surviving_meta = read_staging(path)
    assert surviving_meta["review_run_id"] == REVIEW_RUN_B
    assert not (root / "staging" / "done").exists()
    assert "staging-review-stale" in capsys.readouterr().err


def test_the_done_archive_lock_is_held_through_live_staging_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real force writer must wait until the recoverable live file is gone.

    The first transaction locked only the live path. Its archive writer released
    the done-path lock after readback, so a second real ``write_staging`` could
    replace the verified archive before the live unlink. Both operations then
    reported success while neither copy of the reviewed archive survived.
    """
    root = project(tmp_path, [])
    path, _meta, _reviewed = _stage_reviewed_pattern_run(
        root, run_id=REVIEW_RUN_A
    )
    archive = root / "staging" / "done" / path.name
    start_writer = Event()
    writer_attempting = Event()
    writer_acquired = Event()
    writer_finished = Event()
    writer_errors: list[BaseException] = []
    acquired_before_unlink: list[bool] = []
    writer_thread: Thread
    real_staging_lock = staging_module.exclusive_path_lock

    @contextmanager
    def observed_staging_lock(target: Path) -> Any:
        if current_thread() is writer_thread and Path(target) == archive:
            writer_attempting.set()
        with real_staging_lock(target):
            if current_thread() is writer_thread and Path(target) == archive:
                writer_acquired.set()
            yield

    monkeypatch.setattr(staging_module, "exclusive_path_lock", observed_staging_lock)

    def replace_archive() -> None:
        try:
            assert start_writer.wait(5)
            rows, meta = read_staging(archive)
            meta["review_notes"] += "\nConcurrent forced replacement."
            # The public writer and its real path lock, not a test double.
            staging_module.write_staging(archive, rows, meta, force=True)
        except BaseException as exc:  # pragma: no cover - asserted in parent thread
            writer_errors.append(exc)
        finally:
            writer_finished.set()

    writer_thread = Thread(target=replace_archive)
    writer_thread.start()
    real_unlink = Path.unlink

    def observe_live_unlink(target: Path, *args: Any, **kwargs: Any) -> None:
        if target == path:
            start_writer.set()
            assert writer_attempting.wait(5)
            # Once the writer has reached the real lock, it must remain blocked
            # until this unlink completes and the outer done lock is released.
            acquired_before_unlink.append(writer_acquired.wait(0.5))
        real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", observe_live_unlink)

    assert cli.main(["--root", str(root), "promote", str(path)]) == 0
    assert writer_finished.wait(5)
    writer_thread.join(timeout=5)

    assert writer_errors == []
    assert acquired_before_unlink == [False]
    assert writer_acquired.is_set(), "the real writer proceeds after both locks release"


def test_completed_rich_extractions_with_one_staging_name_keep_separate_archives(
    tmp_path: Path,
) -> None:
    first = record()
    second = record(
        id="word:聞く:きく",
        expression="聞く",
        reading="きく",
        meanings=["to hear"],
    )
    root = project(tmp_path, [])
    path = root / "staging" / "lesson.yaml"
    path.parent.mkdir(parents=True)
    pattern_set = patterns.PatternSet(source="lesson.pdf", kind="lesson")

    write_staging(
        path,
        [first],
        _pattern_only_meta(
            pattern_set,
            run_id=REVIEW_RUN_A,
            request_fingerprint="a" * 64,
            candidates=(parsed_candidate(),),
        ),
    )
    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 0

    write_staging(
        path,
        [second],
        _pattern_only_meta(
            pattern_set,
            run_id=REVIEW_RUN_B,
            request_fingerprint="b" * 64,
            candidates=(
                parsed_candidate(expression="聞く", reading="きく", meanings=["to hear"]),
            ),
        ),
    )
    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 0

    archives = sorted((root / "staging" / "done").glob("lesson*.yaml"))
    assert len(archives) == 2
    archived_runs = {}
    for archive in archives:
        rows, meta = read_staging(archive)
        archived_runs[meta["prompt_provenance"]["request_fingerprint"]] = rows
    assert [item.id for item in archived_runs["a" * 64]] == [first.id]
    assert [item.id for item in archived_runs["b" * 64]] == [second.id]


def test_completed_rich_extractions_with_the_same_request_are_distinct_runs(
    tmp_path: Path,
) -> None:
    first = record()
    second = record(
        id="word:聞く:きく", expression="聞く", reading="きく", meanings=["to hear"]
    )
    root = project(tmp_path, [])
    path = root / "staging" / "lesson.yaml"
    path.parent.mkdir(parents=True)
    pattern_set = patterns.PatternSet(source="lesson.pdf", kind="lesson")

    write_staging(
        path,
        [first],
        _pattern_only_meta(
            pattern_set,
            run_id=REVIEW_RUN_A,
            request_fingerprint="a" * 64,
            candidates=(parsed_candidate(),),
        ),
    )
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    assert cli.main(command) == 0
    write_staging(
        path,
        [second],
        _pattern_only_meta(
            pattern_set,
            run_id=REVIEW_RUN_B,
            request_fingerprint="a" * 64,
            candidates=(
                parsed_candidate(expression="聞く", reading="きく", meanings=["to hear"]),
            ),
        ),
    )
    assert cli.main(command) == 0

    archives = list((root / "staging" / "done").glob("lesson*.yaml"))
    assert len(archives) == 2
    runs = {}
    for archive in archives:
        rows, meta = read_staging(archive)
        runs[meta["review_run_id"]] = rows
    assert [item.id for item in runs[REVIEW_RUN_A]] == [first.id]
    assert [item.id for item in runs[REVIEW_RUN_B]] == [second.id]


def test_review_run_id_survives_a_partial_promotion_retry(tmp_path: Path) -> None:
    first = record()
    held = record(id="word:聞く:", expression="聞く", reading="", meanings=["to hear"])
    root = project(tmp_path, [])
    path = root / "staging" / "lesson.yaml"
    path.parent.mkdir(parents=True)
    meta = _pattern_only_meta(
        patterns.PatternSet(source="lesson.pdf", kind="lesson"),
        run_id=REVIEW_RUN_A,
        request_fingerprint="a" * 64,
        candidates=(
            parsed_candidate(),
            parsed_candidate(
                expression="聞く",
                reading="",
                meanings=["to hear"],
                page=2,
            ),
        ),
    )
    write_staging(
        path,
        [first, held],
        meta,
    )
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]

    assert cli.main(command) == 0
    survivors, meta = read_staging(path)
    assert meta["review_run_id"] == REVIEW_RUN_A
    write_staging(path, [replace(survivors[0], reading="きく")], meta, force=True)
    assert cli.main(command) == 0

    archives = list((root / "staging" / "done").glob("lesson*.yaml"))
    assert len(archives) == 1
    archived, archived_meta = read_staging(archives[0])
    assert archived_meta["review_run_id"] == REVIEW_RUN_A
    assert {item.id for item in archived} == {first.id, "word:聞く:きく"}


def test_candidate_accounting_survives_partial_promotion_and_human_review(
    tmp_path: Path,
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(
        tmp_path,
        parsed_candidate(),
        parsed_candidate(
            expression="聞く",
            reading="",
            meanings=["to hear"],
            page=2,
            context="聞く　to hear",
        ),
    )
    path = root / "staging" / "lesson.yaml"
    write_staging(path, records, meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]

    assert cli.main(command) == 0
    survivors, live_meta = read_staging(path)
    assert len(survivors) == 1
    assert live_meta["candidate_accounting"] == meta["candidate_accounting"]
    write_staging(
        path,
        [replace(survivors[0], reading="きく")],
        live_meta,
        force=True,
    )

    assert cli.main(command) == 0
    assert not path.exists()
    archived, archived_meta = read_staging(
        root / "staging" / "done" / "lesson.yaml"
    )
    assert {item.id for item in archived} == {
        "word:話す:はなす",
        "word:聞く:きく",
    }
    assert archived_meta["candidate_accounting"] == meta["candidate_accounting"]


def test_exact_archive_retry_prunes_live_row_without_appending_it_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(tmp_path, parsed_candidate())
    path = root / "staging" / "lesson.yaml"
    write_staging(path, records, meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    real_prune = promotion_application.prune_staging_under_lock

    def fail_after_archive(*_args: Any, **_kwargs: Any) -> int:
        raise staging_module.StagingError("simulated prune failure")

    monkeypatch.setattr(
        promotion_application, "prune_staging_under_lock", fail_after_archive
    )
    assert cli.main(command) == 1
    archive = root / "staging" / "done" / "lesson.yaml"
    assert path.exists() and archive.exists()
    first_archive = archive.read_bytes()

    monkeypatch.setattr(
        promotion_application, "prune_staging_under_lock", real_prune
    )
    assert cli.main(command) == 0

    assert not path.exists()
    archived, archived_meta = read_staging(archive)
    assert [item.id for item in archived] == ["word:話す:はなす"]
    assert archived_meta["candidate_accounting"] == meta["candidate_accounting"]
    assert archive.read_bytes() == first_archive, "an exact retry need not rewrite evidence"


def test_candidate_accounting_caps_new_rows_across_an_exact_archive_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One parsed proposal can be reidentified, but it cannot become two rows."""
    root = project(tmp_path, [])
    records, meta = accounted_extract(tmp_path, parsed_candidate())
    path = root / "staging" / "lesson.yaml"
    write_staging(path, records, meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    real_prune = promotion_application.prune_staging_under_lock

    def fail_after_archive(*_args: Any, **_kwargs: Any) -> int:
        raise staging_module.StagingError("simulated prune failure")

    monkeypatch.setattr(
        promotion_application, "prune_staging_under_lock", fail_after_archive
    )
    assert cli.main(command) == 1
    monkeypatch.setattr(
        promotion_application, "prune_staging_under_lock", real_prune
    )

    archive = root / "staging" / "done" / "lesson.yaml"
    archived_before = archive.read_bytes()
    normalized_before = (root / "vocabulary.json").read_bytes()
    ledger_before = (root / "ledger.json").read_bytes()
    live, live_meta = read_staging(path)
    appended = record(
        id="word:聞く:きく",
        expression="聞く",
        reading="きく",
        meanings=["to hear"],
    )
    # Keep the archived row as an exact retry and append one distinct row. The
    # retry consumes no new proposal slot; the appended row would be a second
    # accepted row from an account that proves the model parsed only one.
    write_staging(path, [*live, appended], live_meta, force=True)
    live_before = path.read_bytes()

    assert cli.main(command) == 1

    assert path.read_bytes() == live_before
    assert archive.read_bytes() == archived_before
    assert (root / "vocabulary.json").read_bytes() == normalized_before
    assert (root / "ledger.json").read_bytes() == ledger_before
    assert "candidate-accounting-population" in capsys.readouterr().err


def test_candidate_accounting_counts_archived_and_nonretry_live_rows_together(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A live population within the ceiling can still exceed it with the archive."""
    root = project(tmp_path, [])
    records, meta = accounted_extract(
        tmp_path,
        parsed_candidate(),
        parsed_candidate(
            expression="聞く",
            reading="",
            meanings=["to hear"],
            page=2,
        ),
    )
    path = root / "staging" / "lesson.yaml"
    write_staging(path, records, meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]

    assert cli.main(command) == 0
    archive = root / "staging" / "done" / "lesson.yaml"
    archived_before = archive.read_bytes()
    normalized_before = (root / "vocabulary.json").read_bytes()
    ledger_before = (root / "ledger.json").read_bytes()
    [held], live_meta = read_staging(path)
    repaired = replace(held, reading="きく")
    appended = record(
        id="word:食べる:たべる",
        expression="食べる",
        reading="たべる",
        meanings=["to eat"],
    )
    # Two live rows do not exceed the two parsed proposals by themselves. The
    # already-archived row makes three unique accepted rows for that same run.
    write_staging(path, [repaired, appended], live_meta, force=True)
    live_before = path.read_bytes()

    assert cli.main(command) == 1

    assert path.read_bytes() == live_before
    assert archive.read_bytes() == archived_before
    assert (root / "vocabulary.json").read_bytes() == normalized_before
    assert (root / "ledger.json").read_bytes() == ledger_before
    assert "candidate-accounting-population" in capsys.readouterr().err


def test_candidate_accounting_ceiling_keeps_collision_and_unusable_slots_editable(
    tmp_path: Path,
) -> None:
    """The ceiling is parsed proposals, not only machine-canonical records."""
    root = project(tmp_path, [])
    records, meta = accounted_extract(
        tmp_path,
        parsed_candidate(meanings=["canonical"]),
        parsed_candidate(meanings=["colliding proposal"], page=2),
        parsed_candidate(expression="", reading="", page=3),
    )
    assert meta["candidate_accounting"]["parsed_candidate_count"] == 3
    assert meta["candidate_accounting"]["canonical_record_count"] == 1
    assert meta["candidate_accounting"]["duplicate_candidate_count"] == 1
    assert meta["candidate_accounting"]["unusable_candidate_count"] == 1
    reviewed = [
        records[0],
        record(
            id="word:聞く:きく",
            expression="聞く",
            reading="きく",
            meanings=["to hear"],
        ),
        record(
            id="word:食べる:たべる",
            expression="食べる",
            reading="たべる",
            meanings=["to eat"],
        ),
    ]
    path = root / "staging" / "lesson.yaml"
    write_staging(path, reviewed, meta)

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 0

    archived, _archived_meta = read_staging(
        root / "staging" / "done" / "lesson.yaml"
    )
    assert {item.id for item in archived} == {item.id for item in reviewed}


def test_schema_v3_exact_archive_retry_does_not_append_the_row_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archive recovery predates candidate accounting and is not a v2 feature."""
    root = project(tmp_path, [])
    row = record()
    meta = _pattern_only_meta(
        patterns.PatternSet(source="lesson.pdf", kind="lesson"),
        candidates=(parsed_candidate(),),
        response_schema_version=3,
    )
    path = root / "staging" / "lesson.yaml"
    write_staging(path, [row], meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    real_prune = promotion_application.prune_staging_under_lock

    monkeypatch.setattr(
        promotion_application,
        "prune_staging_under_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            staging_module.StagingError("simulated prune failure")
        ),
    )
    assert cli.main(command) == 1
    monkeypatch.setattr(
        promotion_application, "prune_staging_under_lock", real_prune
    )

    assert cli.main(command) == 0

    archived, _archived_meta = read_staging(
        root / "staging" / "done" / "lesson.yaml"
    )
    assert [item.id for item in archived] == [row.id]


def test_schema_v4_still_requires_coverage_v2_candidate_accounting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The v5 bump must not reopen v4's authenticated-accounting boundary."""
    root = project(tmp_path, [])
    parsed = parsed_candidate()
    built = extract.build_records(
        [parsed], type("Prepared", (), {"origin_path": tmp_path / "lesson.pdf"})()
    )
    meta = _pattern_only_meta(
        patterns.PatternSet(source="lesson.pdf", kind="lesson"),
        candidates=(parsed,),
        response_schema_version=4,
    )
    del meta["candidate_accounting"]
    meta["coverage"] = extract.coverage_block(
        extract.ExtractionResult(
            candidates=(parsed,), source_units=(), model_reported_unit_count=0
        ),
        source_sha256="1" * 64,
        mode="prose",
    )
    path = root / "staging" / "lesson.yaml"
    write_staging(path, list(built.records), meta)
    before = path.read_bytes()

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 1
    assert path.read_bytes() == before
    assert "candidate-accounting-invalid" in capsys.readouterr().err


def test_exact_archive_retry_matches_the_post_remint_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(
        tmp_path,
        parsed_candidate(expression="聞く", reading=""),
    )
    reviewed = replace(records[0], reading="きく")
    path = root / "staging" / "lesson.yaml"
    write_staging(path, [reviewed], meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    real_prune = promotion_application.prune_staging_under_lock

    monkeypatch.setattr(
        promotion_application,
        "prune_staging_under_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            staging_module.StagingError("simulated prune failure")
        ),
    )
    assert cli.main(command) == 1
    monkeypatch.setattr(
        promotion_application, "prune_staging_under_lock", real_prune
    )

    assert cli.main(command) == 0

    archived, _archived_meta = read_staging(
        root / "staging" / "done" / "lesson.yaml"
    )
    assert [item.id for item in archived] == ["word:聞く:きく"]


def test_exact_remint_retry_is_independent_of_later_collection_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(
        tmp_path,
        parsed_candidate(expression="聞く", reading=""),
    )
    reviewed = replace(records[0], reading="きく")
    path = root / "staging" / "lesson.yaml"
    write_staging(path, [reviewed], meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    real_prune = promotion_application.prune_staging_under_lock

    monkeypatch.setattr(
        promotion_application,
        "prune_staging_under_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            staging_module.StagingError("simulated prune failure")
        ),
    )
    assert cli.main(command) == 1
    monkeypatch.setattr(
        promotion_application, "prune_staging_under_lock", real_prune
    )

    # Mutable collection state must not change what the already-archived
    # transaction means. This stale id appearing later makes ordinary remint()
    # preserve it, but the retry still names the stable-reminted archive row.
    normalized = root / "vocabulary.json"
    current = json.loads(normalized.read_text(encoding="utf-8"))
    normalized.write_text(
        json.dumps([*current, reviewed.to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )

    assert cli.main(command) == 0

    archived, _archived_meta = read_staging(
        root / "staging" / "done" / "lesson.yaml"
    )
    assert [item.id for item in archived] == ["word:聞く:きく"]


def test_exact_retry_does_not_delete_a_concurrent_forced_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(tmp_path, parsed_candidate())
    path = root / "staging" / "lesson.yaml"
    write_staging(path, records, meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    real_prune = promotion_application.prune_staging_under_lock

    monkeypatch.setattr(
        promotion_application,
        "prune_staging_under_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            staging_module.StagingError("simulated prune failure")
        ),
    )
    assert cli.main(command) == 1
    monkeypatch.setattr(
        promotion_application, "prune_staging_under_lock", real_prune
    )

    replacement_records, replacement_meta = accounted_extract(
        tmp_path,
        parsed_candidate(expression="食べる", reading="たべる"),
    )
    real_finish = promotion_application._finish_record_review

    def replace_before_finish(*args: Any, **kwargs: Any) -> tuple[Path, int]:
        write_staging(path, replacement_records, replacement_meta, force=True)
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(
        promotion_application, "_finish_record_review", replace_before_finish
    )

    assert cli.main(command) == 1

    surviving, _surviving_meta = read_staging(path)
    assert [item.id for item in surviving] == ["word:食べる:たべる"]
    assert "staging-review-stale" in capsys.readouterr().err


def test_normal_promotion_does_not_delete_a_concurrent_forced_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(tmp_path, parsed_candidate())
    path = root / "staging" / "lesson.yaml"
    write_staging(path, records, meta)
    replacement_records, replacement_meta = accounted_extract(
        tmp_path,
        parsed_candidate(expression="食べる", reading="たべる"),
    )
    real_finish = promotion_application._finish_record_review

    def replace_before_finish(*args: Any, **kwargs: Any) -> tuple[Path, int]:
        write_staging(path, replacement_records, replacement_meta, force=True)
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(
        promotion_application, "_finish_record_review", replace_before_finish
    )

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 1

    surviving, _surviving_meta = read_staging(path)
    assert [item.id for item in surviving] == ["word:食べる:たべる"]
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []
    assert not (root / "ledger.json").exists()
    assert not (root / "staging" / "done").exists()
    assert "staging-review-stale" in capsys.readouterr().err


def test_late_zero_row_archive_refuses_before_canonical_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The final archive recheck precedes vocabulary and ledger mutation."""
    root = project(tmp_path, [])
    meta = _pattern_only_meta(
        patterns.PatternSet(source="lesson.pdf", kind="lesson"),
        candidates=(parsed_candidate(),),
        response_schema_version=3,
    )
    path = root / "staging" / "lesson.yaml"
    write_staging(path, [record()], meta)
    live_before = path.read_bytes()
    archive = root / "staging" / "done" / "lesson.yaml"
    completed_pattern_meta = _completed_pattern_archive_meta(meta)
    real_finish = promotion_application._finish_record_review

    def complete_pattern_run_before_finish(
        *args: Any, **kwargs: Any
    ) -> tuple[Path, int]:
        write_staging(archive, [], completed_pattern_meta)
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(
        promotion_application,
        "_finish_record_review",
        complete_pattern_run_before_finish,
    )

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 1

    assert path.read_bytes() == live_before
    assert read_staging(archive) == ([], completed_pattern_meta)
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []
    assert not (root / "ledger.json").exists()
    assert "record-archive-stale" in capsys.readouterr().err


def test_record_promotion_holds_the_done_lock_through_canonical_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-row completer cannot enter after validation but before save."""
    root = project(tmp_path, [])
    meta = _pattern_only_meta(
        patterns.PatternSet(source="lesson.pdf", kind="lesson"),
        candidates=(parsed_candidate(),),
        response_schema_version=3,
    )
    path = root / "staging" / "lesson.yaml"
    write_staging(path, [record()], meta)
    archive = root / "staging" / "done" / "lesson.yaml"
    archive.parent.mkdir(parents=True)
    completed_pattern_meta = _completed_pattern_archive_meta(meta)
    start_writer = Event()
    writer_attempting = Event()
    writer_acquired = Event()
    writer_finished = Event()
    writer_errors: list[BaseException] = []

    def complete_zero_row_archive() -> None:
        try:
            assert start_writer.wait(5)
            writer_attempting.set()
            with cli.exclusive_path_lock(archive):
                writer_acquired.set()
                if not archive.exists():
                    promotion_application.write_staging_under_lock(
                        archive, [], completed_pattern_meta
                    )
        except BaseException as exc:  # pragma: no cover - asserted in parent thread
            writer_errors.append(exc)
        finally:
            writer_finished.set()

    writer = Thread(target=complete_zero_row_archive)
    writer.start()
    real_save_records = promotion_application.save_records_json
    acquired_before_save: list[bool] = []

    def observe_canonical_save(*args: Any, **kwargs: Any) -> None:
        start_writer.set()
        assert writer_attempting.wait(5)
        acquired_before_save.append(writer_acquired.wait(0.5))
        real_save_records(*args, **kwargs)

    monkeypatch.setattr(
        promotion_application, "save_records_json", observe_canonical_save
    )

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 0
    assert writer_finished.wait(5)
    writer.join(timeout=5)

    assert writer_errors == []
    assert acquired_before_save == [False]
    assert not path.exists()
    archived, archived_meta = read_staging(archive)
    assert [item.id for item in archived] == ["word:話す:はなす"]
    assert "reviewed_pattern_set" not in archived_meta
    assert set(stored(root)) == {"word:話す:はなす"}
    assert (root / "ledger.json").exists()


def test_record_archive_selection_is_rechecked_under_its_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(tmp_path, parsed_candidate())
    path = root / "staging" / "lesson.yaml"
    write_staging(path, records, meta)
    archive_base = root / "staging" / "done" / "lesson.yaml"
    other_records, other_meta = accounted_extract(
        tmp_path,
        parsed_candidate(expression="食べる", reading="たべる"),
    )
    other_meta["review_run_id"] = REVIEW_RUN_B
    other_meta["pattern_set"]["review_run_id"] = REVIEW_RUN_B
    real_finish = promotion_application._finish_record_review

    def occupy_base_before_finish(*args: Any, **kwargs: Any) -> tuple[Path, int]:
        write_staging(
            archive_base,
            other_records,
            promote.archive_meta(other_meta, len(other_records)),
        )
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(
        promotion_application, "_finish_record_review", occupy_base_before_finish
    )

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 0

    archives = list((root / "staging" / "done").glob("lesson*.yaml"))
    assert len(archives) == 2
    by_run = {}
    for archive in archives:
        rows, archived_meta = read_staging(archive)
        by_run[archived_meta["review_run_id"]] = rows
    assert [item.id for item in by_run[REVIEW_RUN_A]] == ["word:話す:はなす"]
    assert [item.id for item in by_run[REVIEW_RUN_B]] == ["word:食べる:たべる"]


def test_exact_retry_keeps_live_when_the_done_metadata_is_damaged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(tmp_path, parsed_candidate())
    path = root / "staging" / "lesson.yaml"
    write_staging(path, records, meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    real_prune = promotion_application.prune_staging_under_lock

    monkeypatch.setattr(
        promotion_application,
        "prune_staging_under_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            staging_module.StagingError("simulated prune failure")
        ),
    )
    assert cli.main(command) == 1
    archive = root / "staging" / "done" / "lesson.yaml"
    archived, archived_meta = read_staging(archive)
    del archived_meta["coverage"]
    write_staging(archive, archived, archived_meta, force=True)
    live_before = path.read_bytes()
    monkeypatch.setattr(
        promotion_application, "prune_staging_under_lock", real_prune
    )

    assert cli.main(command) == 1

    assert path.read_bytes() == live_before
    assert "coverage-block-invalid" in capsys.readouterr().err


def test_retry_removes_empty_live_file_left_after_archive_and_prune(
    tmp_path: Path,
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(tmp_path, parsed_candidate())
    path = root / "staging" / "lesson.yaml"
    write_staging(path, records, meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    assert cli.main(command) == 0
    archive = root / "staging" / "done" / "lesson.yaml"
    archive_before = archive.read_bytes()
    # Crash boundary: pruning committed its empty document, unlink did not.
    write_staging(path, [], meta)

    assert cli.main(command) == 0

    assert not path.exists()
    assert archive.read_bytes() == archive_before


def test_divergent_live_copy_of_an_archived_candidate_refuses_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(tmp_path, parsed_candidate())
    path = root / "staging" / "lesson.yaml"
    write_staging(path, records, meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    real_prune = promotion_application.prune_staging_under_lock

    monkeypatch.setattr(
        promotion_application,
        "prune_staging_under_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            staging_module.StagingError("simulated prune failure")
        ),
    )
    assert cli.main(command) == 1
    archive = root / "staging" / "done" / "lesson.yaml"
    archive_before = archive.read_bytes()
    live, live_meta = read_staging(path)
    write_staging(
        path,
        [replace(live[0], meanings=["human changed after archive"])],
        live_meta,
        force=True,
    )
    live_before = path.read_bytes()

    monkeypatch.setattr(
        promotion_application, "prune_staging_under_lock", real_prune
    )
    assert cli.main(command) == 1

    assert path.read_bytes() == live_before
    assert archive.read_bytes() == archive_before


def test_schema_v2_extraction_retry_reuses_its_partial_archive(tmp_path: Path) -> None:
    first = record()
    held = record(id="word:聞く:", expression="聞く", reading="", meanings=["to hear"])
    provenance = _rich_prompt_provenance("a" * 64)
    provenance["response_schema_version"] = 2
    del provenance["response_schema_fingerprint"]
    del provenance["request_fingerprint"]
    root = project(tmp_path, [])
    path = root / "staging" / "lesson.yaml"
    path.parent.mkdir(parents=True)
    write_staging(
        path,
        [first, held],
        {"source_file": "lesson.pdf", "prompt_provenance": provenance},
    )

    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    assert cli.main(command) == 0
    survivors, meta = read_staging(path)
    write_staging(path, [replace(survivors[0], reading="きく")], meta, force=True)
    assert cli.main(command) == 0

    archives = list((root / "staging" / "done").glob("lesson*.yaml"))
    assert len(archives) == 1
    archived, archived_meta = read_staging(archives[0])
    assert {item.id for item in archived} == {first.id, "word:聞く:きく"}
    assert archived_meta["prompt_provenance"] == provenance


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
    results: list[promotion_application.PromotionExecutionResult] = []
    real_execute = promotion_application.execute_promotion

    def capture(
        config: ProjectConfig, decision: promotion_application.PromotionDecision
    ) -> promotion_application.PromotionExecutionResult:
        result = real_execute(config, decision)
        results.append(result)
        return result

    monkeypatch.setattr(promotion_application, "execute_promotion", capture)

    code = cli.main(["--root", str(root), "promote", str(path)])

    assert code == 0
    assert not (root / "vocabulary.json").exists() or stored(root) == {}
    survivors, _ = read_staging(path)
    assert held_reason(survivors[0]) == HOLD_MISSING_READING
    assert results[0].state == "nothing_lands"
    assert results[0].archive_path is None
    assert not (root / "staging" / "done").exists()
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


def test_schema_v3_provenance_binds_the_schema_and_complete_request() -> None:
    block = unmeasured_coverage()
    approve_coverage(block)
    meta = extraction_meta(block)
    provenance = meta["prompt_provenance"]
    provenance["response_schema_version"] = 3
    provenance["response_schema_fingerprint"] = "e" * 64
    provenance["request_fingerprint"] = "f" * 64
    meta["review_run_id"] = REVIEW_RUN_A
    meta["pattern_set"] = {
        "kind": "lesson",
        "title": "Lesson",
        "reviewed": False,
        "patterns": [],
        "prompt_provenance": dict(provenance),
        "review_run_id": REVIEW_RUN_A,
    }

    promote.check_coverage(meta)

    del provenance["request_fingerprint"]
    with pytest.raises(promote.PromoteError, match="prompt-provenance-invalid"):
        promote.check_coverage(meta)


def test_schema_v3_pattern_answer_is_bound_to_the_same_request() -> None:
    block = unmeasured_coverage()
    approve_coverage(block)
    meta = extraction_meta(block)
    provenance = meta["prompt_provenance"]
    provenance["response_schema_version"] = 3
    provenance["response_schema_fingerprint"] = "e" * 64
    provenance["request_fingerprint"] = "f" * 64
    meta["review_run_id"] = REVIEW_RUN_A
    meta["pattern_set"] = {
        "kind": "lesson",
        "title": "Lesson",
        "reviewed": False,
        "patterns": [],
        "prompt_provenance": {
            **provenance,
            "request_fingerprint": "0" * 64,
        },
        "review_run_id": REVIEW_RUN_A,
    }

    with pytest.raises(promote.PromoteError, match="prompt-provenance-stale"):
        promote.check_coverage(meta)


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
    """The per-unit shape checks a hand-edited staging file has to pass.

    Measured: disabling `_coverage_fact`'s fingerprint or disposition check
    left the whole suite green — before this change and after it, so M8.4
    narrowed an already-blind validator. Those are the first two params. The
    third, a missing field, is caught one level up by
    `_validate_coverage_block`'s own `source_units` check rather than by
    `_coverage_fact`; its sibling test below covers `_coverage_fact`'s exact
    -field-set rule, which only an *unknown* key can reach.

    The surviving `coverage-facts-stale` test exercises a different function
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


def test_a_disposition_entry_with_an_unknown_key_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`_coverage_fact`'s own exact-field-set check.

    `source_units` entries never reach it — they have a separate field check
    one line earlier — so the disposition lists are the only route, and
    deleting `set(value) != fields` was invisible without this.
    """
    root = project(tmp_path, [])
    staged = root / "staging" / "damaged-fact.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    coverage = unmeasured_coverage_with_units()
    # An *extra* key, not a missing one. Every missing field is independently
    # caught (page/ordinal, section, fingerprint, disposition each have their
    # own check), so popping one proved the refusal but not this check. A
    # hand-typed key is what only the exact-field-set test refuses — and
    # without it the junk round-trips through `read_staging` into `validate`,
    # `status --staged` and everything else built on it.
    coverage["candidate_units"][0]["surprise"] = "typed by hand"
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


def test_the_archive_keeps_authenticated_collision_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Promotion consumes the canonical row, not the parsed proposals beside it."""
    root = project(tmp_path, [])
    records, meta = accounted_extract(
        tmp_path,
        parsed_candidate(meanings=["canonical proposal"]),
        parsed_candidate(meanings=["other proposal"], page=2),
    )
    path = root / "staging" / "lesson.pdf.yaml"
    write_staging(path, records, meta)
    patch_jpdb(monkeypatch, hanasu_jpdb())

    code = cli.main(["--root", str(root), "promote", str(path)])

    assert code == 0
    _, archived_meta = read_staging(
        root / "staging" / "done" / "lesson.pdf.yaml"
    )
    assert archived_meta["candidate_accounting"] == meta["candidate_accounting"]
    assert "candidate_accounting" not in capsys.readouterr().err


def test_candidate_accounting_promotion_does_not_load_the_optional_ai_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(
        tmp_path,
        parsed_candidate(),
        parsed_candidate(page=2),
    )
    path = root / "staging" / "lesson.pdf.yaml"
    write_staging(path, records, meta)

    monkeypatch.setattr(
        extract,
        "candidate_schema",
        lambda: (_ for _ in ()).throw(
            AssertionError("promote loaded optional AI schema dependencies")
        ),
    )

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 0


def _refresh_candidate_accounting_bindings(meta: dict[str, Any]) -> None:
    accounting = meta["candidate_accounting"]
    accounting["candidate_accounting_fingerprint"] = (
        extract.candidate_accounting_fingerprint(accounting)
    )
    coverage = meta["coverage"]
    coverage["candidate_accounting_fingerprint"] = accounting[
        "candidate_accounting_fingerprint"
    ]
    coverage["coverage_block_fingerprint"] = coverage_block_fingerprint(coverage)


@pytest.mark.parametrize(
    "damage",
    [
        "missing-block",
        "downgraded-to-v1",
        "float-accounting-version",
        "float-coverage-version",
        "extra-block-key",
        "stale-fingerprint",
        "noncanonical-json",
        "stable-id-disagrees",
        "proposal-shape",
        "missing-collision-group",
        "coverage-count",
        "source-candidate-count",
    ],
)
def test_candidate_accounting_damage_refuses_before_promotion_or_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    damage: str,
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(
        tmp_path,
        parsed_candidate(meanings=["canonical"]),
        parsed_candidate(meanings=["collision"], page=2),
        parsed_candidate(expression="食べる", reading="たべる", page=3),
    )
    accounting = meta["candidate_accounting"]
    coverage = meta["coverage"]
    if damage == "missing-block":
        del meta["candidate_accounting"]
    elif damage == "downgraded-to-v1":
        del meta["candidate_accounting"]
        meta["coverage"] = extract.coverage_block(
            extract.ExtractionResult(
                candidates=(
                    parsed_candidate(meanings=["canonical"]),
                    parsed_candidate(meanings=["collision"], page=2),
                    parsed_candidate(
                        expression="食べる", reading="たべる", page=3
                    ),
                ),
                source_units=(),
                model_reported_unit_count=0,
            ),
            source_sha256="1" * 64,
            mode="prose",
        )
    elif damage == "float-accounting-version":
        accounting["version"] = 1.0
        _refresh_candidate_accounting_bindings(meta)
    elif damage == "float-coverage-version":
        coverage["version"] = 2.0
        coverage["coverage_block_fingerprint"] = coverage_block_fingerprint(coverage)
    elif damage == "extra-block-key":
        accounting["invented"] = True
        _refresh_candidate_accounting_bindings(meta)
    elif damage == "stale-fingerprint":
        accounting["canonical_record_count"] = 99
    elif damage == "noncanonical-json":
        accounting["collision_groups"][0]["proposals"][0][
            "parsed_schema_proposal"
        ]["page"] = float("nan")
    elif damage == "stable-id-disagrees":
        accounting["collision_groups"][0]["stable_record_id"] = "word:聞く:きく"
        _refresh_candidate_accounting_bindings(meta)
    elif damage == "proposal-shape":
        accounting["collision_groups"][0]["proposals"][0][
            "parsed_schema_proposal"
        ]["invented"] = True
        _refresh_candidate_accounting_bindings(meta)
    elif damage == "missing-collision-group":
        accounting["collision_groups"] = []
        _refresh_candidate_accounting_bindings(meta)
    elif damage == "coverage-count":
        coverage["parsed_candidate_count"] += 1
        coverage["coverage_block_fingerprint"] = coverage_block_fingerprint(coverage)
    elif damage == "source-candidate-count":
        accounting["parsed_candidate_count"] += 1
        accounting["unusable_candidate_count"] += 1
        _refresh_candidate_accounting_bindings(meta)
        coverage["parsed_candidate_count"] = accounting["parsed_candidate_count"]
        coverage["unusable_candidate_count"] = accounting[
            "unusable_candidate_count"
        ]
        coverage["coverage_block_fingerprint"] = coverage_block_fingerprint(coverage)
    else:
        raise AssertionError(damage)

    path = root / "staging" / "lesson.pdf.yaml"
    write_staging(path, records, meta)
    before = path.read_bytes()

    code = cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    )

    assert code == 1
    assert path.read_bytes() == before
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []
    assert not (root / "ledger.json").exists()
    assert not (root / "staging" / "done").exists()
    error = capsys.readouterr().err
    if damage == "float-coverage-version":
        assert "coverage-block-invalid" in error
    else:
        assert "candidate-accounting" in error


def test_two_human_reidentified_rows_cannot_converge_on_one_canonical_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = project(tmp_path, [])
    records, meta = accounted_extract(
        tmp_path,
        parsed_candidate(),
        parsed_candidate(expression="食べる", reading="たべる"),
    )
    converged = replace(
        records[1],
        expression=records[0].expression,
        reading=records[0].reading,
    )
    path = root / "staging" / "lesson.pdf.yaml"
    write_staging(path, [records[0], converged], meta)
    before = path.read_bytes()

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 1

    assert path.read_bytes() == before
    assert json.loads((root / "vocabulary.json").read_text(encoding="utf-8")) == []
    assert not (root / "staging" / "done").exists()
    assert "canonical record id" in capsys.readouterr().err


def test_human_can_delete_a_bad_canonical_proposal_without_rewriting_accounting(
    tmp_path: Path,
) -> None:
    """Accounting records what the model proposed; it is not a keep-list."""
    root = project(tmp_path, [])
    records, meta = accounted_extract(
        tmp_path,
        parsed_candidate(),
        parsed_candidate(
            expression="食べる",
            reading="たべる",
            meanings=["not worth a card"],
            page=2,
        ),
    )
    path = root / "staging" / "lesson.pdf.yaml"
    write_staging(path, records[:1], meta)

    code = cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    )

    assert code == 0
    archived, archived_meta = read_staging(
        root / "staging" / "done" / "lesson.pdf.yaml"
    )
    assert [item.id for item in archived] == ["word:話す:はなす"]
    assert archived_meta["candidate_accounting"]["canonical_record_count"] == 2


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


# --- fingerprint-bound staged replacements ---------------------------------


def ai_staging_meta(
    originals: list[VocabularyRecord],
    changes: dict[str, dict[str, tuple[Any, Any]]],
) -> dict[str, Any]:
    ids = sorted(changes)
    return {
        "source_file": "vocabulary.json",
        "model": "claude-opus-5",
        "provider": "anthropic",
        "ai_enrichment": {
            "version": 1,
            "model": "claude-opus-5",
            "provider": "anthropic",
            "request_fingerprints": {record_id: "a" * 64 for record_id in ids},
            "input_fingerprints": {record_id: "b" * 64 for record_id in ids},
            "fields": {
                record_id: sorted(changes[record_id]) for record_id in ids
            },
        },
        "field_replacements": promote.field_replacement_block(originals, changes),
    }


def test_promote_lands_reviewed_ai_replacements_with_request_provenance(
    tmp_path: Path,
) -> None:
    original = record(
        meanings=["to speak"],
        examples=[ExampleSentence(japanese="古い例です。", english="Old example.")],
        usage_notes="old note",
    )
    proposal = replace(
        original,
        meanings=["to converse"],
        examples=[
            ExampleSentence(
                japanese="先生と話します。",
                furigana="先生[せんせい]と 話[はな]します。",
                english="I speak with my teacher.",
            )
        ],
        usage_notes="Often takes と for the person spoken with.",
    )
    changes = {
        original.id: {
            "meanings": (original.meanings, proposal.meanings),
            "examples": (original.examples, proposal.examples),
            "usage_notes": (original.usage_notes, proposal.usage_notes),
        }
    }
    root = project(tmp_path, [original])
    path = root / "staging" / "ai.yaml"
    path.parent.mkdir(parents=True)
    write_staging(path, [proposal], ai_staging_meta([original], changes))

    code = cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    )

    assert code == 0
    written = stored(root)[original.id]
    assert written["meanings"] == ["to converse"]
    assert [item["japanese"] for item in written["examples"]] == [
        "先生と話します。"
    ]
    assert written["usage_notes"] == "Often takes と for the person spoken with."
    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    entries = book["records"][original.id]["enriched"]
    assert [{key: value for key, value in entry.items() if key != "at"} for entry in entries] == [
        {
            "kind": "ai",
            "model": "claude-opus-5",
            "provider": "anthropic",
            "fields": ["examples", "meanings", "usage_notes"],
            "request_fingerprint": "a" * 64,
        }
    ]
    assert "at" in entries[0]


def test_promote_attributes_only_the_ai_fields_the_review_actually_wrote(
    tmp_path: Path,
) -> None:
    original = record(meanings=["to speak"], usage_notes="human wording")
    model_proposal = replace(
        original, meanings=["to converse"], usage_notes="model wording"
    )
    reviewed = replace(model_proposal, usage_notes=original.usage_notes)
    changes = {
        original.id: {
            "meanings": (original.meanings, model_proposal.meanings),
            "usage_notes": (original.usage_notes, model_proposal.usage_notes),
        }
    }
    root = project(tmp_path, [original])
    path = root / "staging" / "ai.yaml"
    path.parent.mkdir(parents=True)
    write_staging(path, [reviewed], ai_staging_meta([original], changes))

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 0

    written = stored(root)[original.id]
    assert written["meanings"] == ["to converse"]
    assert written["usage_notes"] == "human wording"
    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    [entry] = book["records"][original.id]["enriched"]
    assert entry["fields"] == ["meanings"]
    assert entry["request_fingerprint"] == "a" * 64


def test_completed_ai_reviews_with_one_staging_name_keep_separate_archives(
    tmp_path: Path,
) -> None:
    original = record(meanings=["to speak"])
    first_proposal = replace(original, meanings=["to converse"])
    root = project(tmp_path, [original])
    path = root / "staging" / "ai.yaml"
    path.parent.mkdir(parents=True)

    first_meta = ai_staging_meta(
        [original],
        {original.id: {"meanings": (original.meanings, first_proposal.meanings)}},
    )
    write_staging(path, [first_proposal], first_meta)
    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 0

    second_proposal = replace(first_proposal, meanings=["to chat"])
    second_meta = ai_staging_meta(
        [first_proposal],
        {
            original.id: {
                "meanings": (first_proposal.meanings, second_proposal.meanings)
            }
        },
    )
    second_meta["ai_enrichment"]["request_fingerprints"][original.id] = "c" * 64
    write_staging(path, [second_proposal], second_meta)
    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 0

    archives = sorted((root / "staging" / "done").glob("ai*.yaml"))
    assert len(archives) == 2
    archived_runs = {}
    for archive in archives:
        rows, meta = read_staging(archive)
        request = meta["ai_enrichment"]["request_fingerprints"][original.id]
        archived_runs[request] = rows
    assert archived_runs["a" * 64][0].meanings == ["to converse"]
    assert archived_runs["c" * 64][0].meanings == ["to chat"]


def test_completed_ai_reviews_with_the_same_request_are_distinct_runs(
    tmp_path: Path,
) -> None:
    original = record(meanings=["to speak"])
    proposal = replace(original, meanings=["to converse"])
    changes = {
        original.id: {"meanings": (original.meanings, proposal.meanings)}
    }
    root = project(tmp_path, [original])
    path = root / "staging" / "ai.yaml"
    path.parent.mkdir(parents=True)

    first_meta = ai_staging_meta([original], changes)
    first_meta["review_run_id"] = REVIEW_RUN_A
    write_staging(path, [proposal], first_meta)
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    assert cli.main(command) == 0

    second_meta = ai_staging_meta([original], changes)
    second_meta["review_run_id"] = REVIEW_RUN_B
    write_staging(path, [proposal], second_meta)
    assert cli.main(command) == 0

    archives = list((root / "staging" / "done").glob("ai*.yaml"))
    assert len(archives) == 2
    runs = {}
    for archive in archives:
        rows, meta = read_staging(archive)
        runs[meta["review_run_id"]] = rows
    assert runs[REVIEW_RUN_A][0].meanings == ["to converse"]
    assert runs[REVIEW_RUN_B][0].meanings == ["to converse"]


def test_failed_staged_ai_ledger_save_keeps_the_live_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = record(meanings=["to speak"])
    proposal = replace(original, meanings=["to converse"])
    root = project(tmp_path, [original])
    path = root / "staging" / "ai.yaml"
    path.parent.mkdir(parents=True)
    meta = ai_staging_meta(
        [original],
        {original.id: {"meanings": (original.meanings, proposal.meanings)}},
    )
    meta["review_run_id"] = REVIEW_RUN_A
    write_staging(
        path,
        [proposal],
        meta,
    )
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("disk full")),
    )

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 1

    assert stored(root)[original.id]["meanings"] == ["to converse"]
    assert path.exists(), "the only recoverable AI attribution stays live"
    _rows, live_meta = read_staging(path)
    assert live_meta["review_run_id"] == REVIEW_RUN_A
    assert not (root / "staging" / "done" / "ai.yaml").exists()
    err = capsys.readouterr().err
    assert "'status --rebuild' cannot bring it back" in err
    assert "staging" in err and "re-run" in err


def test_rerunning_a_failed_staged_ai_ledger_save_recovers_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = record(meanings=["to speak"])
    proposal = replace(original, meanings=["to converse"])
    root = project(tmp_path, [original])
    path = root / "staging" / "ai.yaml"
    path.parent.mkdir(parents=True)
    meta = ai_staging_meta(
        [original],
        {original.id: {"meanings": (original.meanings, proposal.meanings)}},
    )
    meta["review_run_id"] = REVIEW_RUN_A
    write_staging(
        path,
        [proposal],
        meta,
    )
    real_save = cli.ledger.Ledger.save
    calls = 0

    def fail_once(book: cli.ledger.Ledger) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise cli.ledger.LedgerError("disk full")
        real_save(book)

    monkeypatch.setattr(cli.ledger.Ledger, "save", fail_once)

    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    assert cli.main(command) == 1
    assert path.exists()
    _rows, live_meta = read_staging(path)
    assert live_meta["review_run_id"] == REVIEW_RUN_A
    assert cli.main(command) == 0

    [entry] = json.loads((root / "ledger.json").read_text(encoding="utf-8"))[
        "records"
    ][original.id]["enriched"]
    assert entry["kind"] == "ai"
    assert entry["fields"] == ["meanings"]
    assert entry["request_fingerprint"] == "a" * 64
    assert not path.exists()
    _rows, archived_meta = read_staging(root / "staging" / "done" / "ai.yaml")
    assert archived_meta["review_run_id"] == REVIEW_RUN_A


def test_failed_staged_ai_retry_refuses_a_third_field_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = record(meanings=["to speak"])
    proposal = replace(original, meanings=["to converse"])
    root = project(tmp_path, [original])
    path = root / "staging" / "ai.yaml"
    path.parent.mkdir(parents=True)
    write_staging(
        path,
        [proposal],
        ai_staging_meta(
            [original],
            {original.id: {"meanings": (original.meanings, proposal.meanings)}},
        ),
    )
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("disk full")),
    )
    command = ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    assert cli.main(command) == 1
    capsys.readouterr()

    human_edit = replace(proposal, meanings=["human correction"])
    (root / "vocabulary.json").write_text(
        json.dumps([human_edit.to_dict()], ensure_ascii=False), encoding="utf-8"
    )

    assert cli.main(command) == 1
    assert stored(root)[original.id]["meanings"] == ["human correction"]
    assert path.exists()
    assert not (root / "staging" / "done" / "ai.yaml").exists()
    assert "field-replacements-stale" in capsys.readouterr().err


@pytest.mark.parametrize(
    "damage",
    [
        "missing-ai-block",
        "float-version",
        "malformed-request-fingerprint",
        "numeric-request-fingerprint",
        "map-id-skew",
        "field-map-mismatch",
        "unknown-id-everywhere",
    ],
)
def test_ai_provenance_damage_refuses_before_records_change(
    tmp_path: Path, damage: str
) -> None:
    original = record(meanings=["to speak"])
    proposal = replace(original, meanings=["to converse"])
    changes = {
        original.id: {"meanings": (original.meanings, proposal.meanings)}
    }
    meta = ai_staging_meta([original], changes)
    block = meta["ai_enrichment"]
    if damage == "missing-ai-block":
        del meta["ai_enrichment"]
    elif damage == "float-version":
        block["version"] = 1.0
    elif damage == "malformed-request-fingerprint":
        block["request_fingerprints"][original.id] = "tampered"
    elif damage == "numeric-request-fingerprint":
        block["request_fingerprints"][original.id] = int("1" * 64)
    elif damage == "map-id-skew":
        block["request_fingerprints"]["word:ghost:ghost"] = "c" * 64
    elif damage == "field-map-mismatch":
        block["fields"][original.id] = ["examples"]
    else:
        ghost = "word:ghost:ghost"
        block["request_fingerprints"][ghost] = "c" * 64
        block["input_fingerprints"][ghost] = "d" * 64
        block["fields"][ghost] = ["meanings"]
        meta["field_replacements"]["records"][ghost] = {
            "meanings": meta["field_replacements"]["records"][original.id][
                "meanings"
            ]
        }
    root = project(tmp_path, [original])
    path = root / "staging" / "ai.yaml"
    path.parent.mkdir(parents=True)
    write_staging(path, [proposal], meta)
    before = (root / "vocabulary.json").read_text(encoding="utf-8")

    code = cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    )

    assert code == 1
    assert (root / "vocabulary.json").read_text(encoding="utf-8") == before
    assert path.exists(), "the reviewed proposal remains recoverable"
    assert not (root / "ledger.json").exists()
    assert not (root / "staging" / "done" / "ai.yaml").exists()


def test_ai_provenance_keeps_partial_promotion_retries_valid(tmp_path: Path) -> None:
    first = record(meanings=["to speak"])
    second = record(
        id="word:聞く:きく",
        expression="聞く",
        reading="きく",
        meanings=["to hear"],
    )
    proposals = [
        replace(first, meanings=["to converse"]),
        replace(second, meanings=["to ask"]),
    ]
    changes = {
        first.id: {"meanings": (first.meanings, proposals[0].meanings)},
        second.id: {"meanings": (second.meanings, proposals[1].meanings)},
    }
    root = project(tmp_path, [first, second])
    path = root / "staging" / "ai.yaml"
    path.parent.mkdir(parents=True)
    write_staging(
        path,
        [proposals[0], replace(proposals[1], reading="")],
        ai_staging_meta([first, second], changes),
    )

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 0
    after_first = stored(root)
    assert after_first[first.id]["meanings"] == ["to converse"]
    assert after_first[second.id]["meanings"] == ["to hear"]
    held, meta = read_staging(path)
    assert [item.id for item in held] == [second.id]
    write_staging(path, [replace(held[0], reading="きく")], meta, force=True)

    assert cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    ) == 0
    assert stored(root)[second.id]["meanings"] == ["to ask"]


def test_a_stale_held_ai_row_refuses_the_whole_staged_merge(tmp_path: Path) -> None:
    """Held rows remain part of the old-value binding for this review.

    A reading hold decides which rows can land today; it must not narrow the
    concurrency check.  Otherwise editing a held row after staging lets an
    unrelated row land under a review whose all-or-nothing binding is already
    stale.
    """
    first = record(meanings=["to speak"])
    second = record(
        id="word:聞く:きく",
        expression="聞く",
        reading="きく",
        meanings=["to hear"],
    )
    proposals = [
        replace(first, meanings=["to converse"]),
        replace(second, reading="", meanings=["to ask"]),
    ]
    changes = {
        first.id: {"meanings": (first.meanings, proposals[0].meanings)},
        second.id: {"meanings": (second.meanings, proposals[1].meanings)},
    }
    root = project(tmp_path, [first, second])
    path = root / "staging" / "ai.yaml"
    path.parent.mkdir(parents=True)
    write_staging(path, proposals, ai_staging_meta([first, second], changes))

    # Concurrent curation after the paid answer was staged.  The second row is
    # still held by its blank reading, but its binding is no less part of the
    # staged review than the first row's.
    current = [first, replace(second, meanings=["to listen"])]
    (root / "vocabulary.json").write_text(
        json.dumps([item.to_dict() for item in current], ensure_ascii=False),
        encoding="utf-8",
    )
    before = (root / "vocabulary.json").read_text(encoding="utf-8")

    code = cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    )

    assert code == 1
    assert (root / "vocabulary.json").read_text(encoding="utf-8") == before
    assert path.exists()
    assert not (root / "ledger.json").exists()
    assert not (root / "staging" / "done" / "ai.yaml").exists()


def test_an_unrelated_old_archive_cannot_explain_a_fresh_runs_unknown_id(
    tmp_path: Path,
) -> None:
    current = record(meanings=["to speak"])
    current_proposal = replace(current, meanings=["to converse"])
    current_changes = {
        current.id: {"meanings": (current.meanings, current_proposal.meanings)}
    }
    old = record(
        id="word:古い:ふるい",
        expression="古い",
        reading="ふるい",
        meanings=["old"],
    )
    old_proposal = replace(old, meanings=["aged"])
    old_changes = {old.id: {"meanings": (old.meanings, old_proposal.meanings)}}
    root = project(tmp_path, [current])
    done = root / "staging" / "done" / "ai.yaml"
    done.parent.mkdir(parents=True)
    write_staging(done, [old_proposal], ai_staging_meta([old], old_changes))

    fresh_meta = ai_staging_meta([current], current_changes)
    fresh_ai = fresh_meta["ai_enrichment"]
    fresh_ai["request_fingerprints"][old.id] = "c" * 64
    fresh_ai["input_fingerprints"][old.id] = "d" * 64
    fresh_ai["fields"][old.id] = ["meanings"]
    fresh_meta["field_replacements"]["records"][old.id] = {
        "meanings": fresh_meta["field_replacements"]["records"][current.id][
            "meanings"
        ]
    }
    path = root / "staging" / "ai.yaml"
    write_staging(path, [current_proposal], fresh_meta)
    before = (root / "vocabulary.json").read_text(encoding="utf-8")

    code = cli.main(
        ["--root", str(root), "promote", str(path), "--skip-reading-check"]
    )

    assert code == 1
    assert (root / "vocabulary.json").read_text(encoding="utf-8") == before
    assert path.exists()
    assert not (root / "ledger.json").exists()


def test_reviewed_replacements_land_only_on_the_record_and_fields_authorized() -> None:
    """The block is per record as well as per field; it is not a global force."""
    first = VocabularyRecord(
        id="word:話す:はなす", expression="話す", reading="はなす",
        meanings=["to speak"], usage_notes="human one",
        source=SourceReference(type="shirabe", imported_from="words.csv"),
    )
    second = VocabularyRecord(
        id="word:聞く:きく", expression="聞く", reading="きく",
        meanings=["to hear"], usage_notes="human two",
        source=SourceReference(type="shirabe", imported_from="words.csv"),
    )
    proposals = [
        replace(first, meanings=["to converse"], usage_notes="model one"),
        replace(second, meanings=["to ask"], usage_notes="model two"),
    ]
    block = promote.field_replacement_block(
        [first, second],
        {first.id: {"meanings": (first.meanings, proposals[0].meanings)}},
    )

    merged, outcomes = promote.merge_staged_records(
        [first, second],
        proposals,
        {promote.FIELD_REPLACEMENTS_KEY: block},
    )
    by_id = {item.id: item for item in merged}

    assert by_id[first.id].meanings == ["to converse"]
    assert by_id[first.id].usage_notes == "human one"
    assert by_id[second.id].meanings == ["to hear"]
    assert by_id[second.id].usage_notes == "human two"
    assert outcomes[first.id].filled_fields == ["meanings"]
    assert {name for name, _old, _new in outcomes[first.id].conflicts} == {
        "usage_notes"
    }
    assert {name for name, _old, _new in outcomes[second.id].conflicts} == {
        "meanings",
        "usage_notes",
    }


def test_a_reviewer_can_edit_the_new_value_because_the_binding_is_to_the_old() -> None:
    original = record(meanings=["to speak"])
    model_proposal = replace(original, meanings=["to converse"])
    reviewer_wording = replace(model_proposal, meanings=["to talk with someone"])
    block = promote.field_replacement_block(
        [original],
        {original.id: {"meanings": (original.meanings, model_proposal.meanings)}},
    )

    merged, _outcomes = promote.merge_staged_records(
        [original],
        [reviewer_wording],
        {promote.FIELD_REPLACEMENTS_KEY: block},
    )

    assert merged[0].meanings == ["to talk with someone"]


def test_one_stale_old_value_refuses_the_whole_staged_merge() -> None:
    """Validate every compare before merging even the records that still match.

    This is the mutation guard for the load-bearing comparison: comparing the
    stored digest with the staged *new* value, or reversing equality, must fail
    this test or its fresh-value sibling above.
    """
    first = record()
    second = record(
        id="word:聞く:きく", expression="聞く", reading="きく",
        meanings=["to hear"],
    )
    proposals = [
        replace(first, meanings=["to converse"]),
        replace(second, meanings=["to ask"]),
    ]
    block = promote.field_replacement_block(
        [first, second],
        {
            first.id: {"meanings": (first.meanings, proposals[0].meanings)},
            second.id: {"meanings": (second.meanings, proposals[1].meanings)},
        },
    )
    concurrently_edited = replace(second, meanings=["human correction"])

    with pytest.raises(promote.PromoteError, match="field-replacements-stale") as raised:
        promote.merge_staged_records(
            [first, concurrently_edited],
            proposals,
            {promote.FIELD_REPLACEMENTS_KEY: block},
        )

    message = str(raised.value)
    assert second.id in message and "meanings" in message
    assert first.meanings == ["to speak"], "the matching record was not merged first"
    assert concurrently_edited.meanings == ["human correction"]


def test_a_deleted_replacement_target_is_stale_not_a_new_record() -> None:
    original = record()
    proposed = replace(original, meanings=["to converse"])
    block = promote.field_replacement_block(
        [original],
        {original.id: {"meanings": (original.meanings, proposed.meanings)}},
    )

    with pytest.raises(promote.PromoteError, match="no current record"):
        promote.merge_staged_records(
            [], [proposed], {promote.FIELD_REPLACEMENTS_KEY: block}
        )


@pytest.mark.parametrize(
    "block,detail",
    [
        (None, "contain exactly"),
        ({"version": True, "records": {"word:話す:はなす": {"meanings": "a" * 64}}},
         "version"),
        ({"version": 1, "records": {"word:話す:はなす": {"meanings": "short"}}},
         "SHA-256"),
        ({"version": 1, "records": {"word:話す:はなす": {"reading": "a" * 64}}},
         "reading"),
    ],
    ids=["null-block", "boolean-version", "bad-digest", "identity-field"],
)
def test_malformed_replacement_authority_is_refused_before_merge(
    block: dict[str, Any] | None, detail: str
) -> None:
    existing = record()

    with pytest.raises(promote.PromoteError, match=detail):
        promote.merge_staged_records(
            [existing],
            [replace(existing, meanings=["new"])],
            {promote.FIELD_REPLACEMENTS_KEY: block},
        )


def test_schema_v2_extraction_staging_keeps_existing_wins_compatibility() -> None:
    """Old extraction archives have prompt provenance but no replacement block."""
    existing = record(meanings=["human meaning"])
    incoming = replace(existing, meanings=["model meaning"])
    old_meta = {
        "prompt_provenance": {
            "source_sha256": "a" * 64,
            "mode": "prose",
            "provider": "anthropic",
            "model": "claude-opus-5",
            "response_schema_version": 2,
            "system_prompt_fingerprint": "b" * 64,
            "style_guide_fingerprint": "c" * 64,
            "user_prompt_fingerprint": "d" * 64,
        }
    }

    merged, outcomes = promote.merge_staged_records(
        [existing], [incoming], old_meta
    )

    assert merged[0].meanings == ["human meaning"]
    assert outcomes[existing.id].label == "conflicting"


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


# --- what the shared decide step must not reorder ----------------------------


def test_a_refused_file_says_why_before_it_asks_for_a_dictionary_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gates decide without a dictionary, so a file they refuse must be
    refused in their words — not in a complaint about a missing API key.

    This is why the command decides offline first. Building the client up
    front would be simpler and would tell somebody with a broken staging file
    to go and find their jpdb key, which fixes nothing.
    """
    monkeypatch.delenv("JPDB_API_KEY", raising=False)
    root = project(tmp_path)
    staged = root / "staging" / "broken.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text("records: [oh dear\n", encoding="utf-8")

    assert cli.main(["--root", str(root), "promote", str(staged)]) == 1

    stderr = capsys.readouterr().err
    assert "JPDB_API_KEY" not in stderr
    assert "broken.yaml" in stderr


def test_no_dictionary_key_is_needed_for_a_file_that_holds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy file with no records asks the dictionary nothing."""
    monkeypatch.delenv("JPDB_API_KEY", raising=False)
    root = project(tmp_path)
    staged = root / "staging" / "empty.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text("records: []\nsource_file: legacy.pdf\n", encoding="utf-8")

    assert cli.main(["--root", str(root), "promote", str(staged)]) == 0


def test_no_dictionary_key_is_needed_to_finish_an_archive_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A completion writes no new record, so it asks the dictionary nothing.

    The state matters, not just the exit code: this has to be a real
    `archive_retry` — an emptied live review beside this run's own archive,
    the shape a crash between the prune and the unlink leaves — and it has to
    run *without* `--skip-reading-check`, or the condition under test is
    short-circuited before it is reached.
    """
    monkeypatch.delenv("JPDB_API_KEY", raising=False)
    root = project(tmp_path, [record()])
    staged = root / "staging" / "in.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    incoming = record(meanings=["to speak"])
    write_staging(staged, [incoming], {"source_file": "lesson.pdf"})
    _records, meta = read_staging(staged)
    done = staged.parent / "done"
    done.mkdir(parents=True, exist_ok=True)
    write_staging(done / staged.name, [incoming], promote.archive_meta(meta, 1))

    assert cli.main(["--root", str(root), "promote", str(staged)]) == 0

    out = capsys.readouterr().out
    assert "Completed exact archive retry" in out
    assert "already-archived row(s) were removed" in out


def test_an_emptied_live_review_says_it_was_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other archive-retry shape, and the other sentence.

    Both run through one branch now, so the branch is the only thing keeping
    the two messages apart — and neither message was asserted anywhere, which
    made the split unproven exactly where it is easiest to get wrong.
    """
    monkeypatch.delenv("JPDB_API_KEY", raising=False)
    root = project(tmp_path, [record()])
    staged = root / "staging" / "in.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    incoming = record(meanings=["to speak"])
    write_staging(staged, [incoming], {"source_file": "lesson.pdf"})
    _records, meta = read_staging(staged)
    done = staged.parent / "done"
    done.mkdir(parents=True, exist_ok=True)
    write_staging(done / staged.name, [incoming], promote.archive_meta(meta, 1))
    # The crash shape: the archive is written, the live rows are pruned, and
    # the file is never unlinked.
    write_staging(staged, [], meta, force=True)

    assert cli.main(["--root", str(root), "promote", str(staged)]) == 0

    out = capsys.readouterr().out
    assert "empty live review removed" in out
    assert "already-archived row(s) were removed" not in out


def test_accepting_coverage_is_refused_for_a_file_no_approval_could_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--accept-coverage` buys a model's verdict on whether a page is
    accounted for. A file refused by a *structural* gate cannot be made
    promotable by any verdict, so spending on it is money for nothing — and
    the refusal a person needs is the structural one."""
    sent: list[object] = []
    monkeypatch.setattr(
        cli.coverage, "review_coverage",
        lambda *a, **k: sent.append(1) or None,
    )
    root = project(tmp_path)
    staged = root / "staging" / "unreadable.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text("records: [oh dear\n", encoding="utf-8")

    assert cli.main([
        "--root", str(root), "promote", str(staged), "--accept-coverage",
    ]) == 1

    assert sent == [], "nothing should have been sent for a file this broken"
    assert "unreadable.yaml" in capsys.readouterr().err


def test_a_refusal_the_dictionary_could_change_is_not_settled_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deciding offline first judges a *superset* of the rows a consulted run
    promotes: without a dictionary, `check_readings` holds nothing for an
    unlisted reading. So rows can converge on one landing id offline that a
    consulted run never brings together — and believing that refusal exits 1
    on a file the command used to promote cleanly.

    Here the collection already holds `word:話す:はなし`, so that row keeps its
    id while the second re-mints onto it. Offline both land and collide. Asked,
    jpdb lists はなす and not はなし, so both are held, nothing converges, and
    the run ends by recording why.
    """
    monkeypatch.setenv("JPDB_API_KEY", "test-key")
    root = project(tmp_path, [record(id="word:話す:はなし", reading="はなし")])
    staged = root / "staging" / "in.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        staged,
        [
            record(id="word:話す:はなし", reading="はなし", meanings=["kept"]),
            record(id="word:話す:ふるい", reading="はなし", meanings=["reminted"]),
        ],
        {"source_file": "lesson.pdf"},
    )
    api = hanasu_jpdb()
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda *a, **kw: client_for(api))

    assert cli.main(["--root", str(root), "promote", str(staged)]) == 0

    # Held, not landed, and not refused: the collision existed only in the
    # answer nobody had paid for yet.
    assert api.bodies, "the dictionary was consulted"
    remaining, _meta = read_staging(staged)
    assert len(remaining) == 2
    assert all(staging_module.annotations(row).get("hold_reason") for row in remaining)
def test_a_row_the_second_accounting_call_flags_is_not_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`promoted_retry_flags` drives the staging prune, and it exists so the
    write path composes those flags rather than assuming they are all false.

    The assumption is true for every input the real accounting function can
    produce — the first call already probes each row's resolved id and its
    stable re-mint. That is exactly why it cannot be tested without forcing a
    flag: an invariant nothing can violate is indistinguishable from an
    assumption nobody checked.
    """
    root = project(tmp_path, [])
    staged = root / "staging" / "in.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    incoming = record(meanings=["to speak"])
    write_staging(staged, [incoming], {"source_file": "lesson.pdf"})

    from japanese_anki.application import promotion as promotion_module

    real = promotion_module.promote.check_candidate_accounting
    seen: list[int] = []

    def flag_the_second(meta, live, archived, **kwargs):
        answer = real(meta, live, archived, **kwargs)
        seen.append(1)
        # The first call partitions the live rows; the second judges what
        # would land. Only the second is forced.
        return [True] * len(answer) if len(seen) > 1 else answer

    monkeypatch.setattr(
        promotion_module.promote, "check_candidate_accounting", flag_the_second
    )

    assert cli.main([
        "--root", str(root), "promote", str(staged), "--skip-reading-check",
    ]) == 0

    # Flagged as already archived, so nothing landed — and the live review was
    # pruned of it rather than left behind.
    assert load_records(root / "vocabulary.json") == []


# --- saying what would happen, and doing none of it -------------------------


def test_a_dry_run_writes_nothing_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A question about what *would* happen must not spend and must not write.
    The whole tree is compared, because a preview that pruned or annotated
    would turn asking into a half-finished promote."""
    monkeypatch.delenv("JPDB_API_KEY", raising=False)
    root = project(tmp_path, [])
    staged = root / "staging" / "in.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(staged, [record()], {"source_file": "lesson.pdf"})
    before = {
        item: item.read_bytes() for item in sorted(root.rglob("*")) if item.is_file()
    }

    assert cli.main(["--root", str(root), "promote", str(staged), "--dry-run"]) == 0

    after = {
        item: item.read_bytes() for item in sorted(root.rglob("*")) if item.is_file()
    }
    assert after == before
    assert "話す" in capsys.readouterr().out


def test_a_dry_run_names_the_meanings_your_collection_keeps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The existing-wins surprise is the reason to look before adding: promote
    keeps the meaning already on the card, and this lesson's wording becomes
    source evidence rather than card text. A count alone does not warn anybody
    about that."""
    root = project(tmp_path, [record(meanings=["the wording already on my card"])])
    staged = root / "staging" / "in.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(staged, [record(meanings=["what this lesson says"])],
                  {"source_file": "lesson.pdf"})

    assert cli.main(["--root", str(root), "promote", str(staged), "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "keeps your meaning: the wording already on my card" in out
    assert "this source said:   what this lesson says" in out


def test_a_dry_run_exits_non_zero_when_the_source_cannot_be_added(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exit code answers "would this promote", so `--dry-run && promote`
    means what it looks like."""
    root = project(tmp_path, [])
    staged = root / "staging" / "broken.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text("records: [oh dear\n", encoding="utf-8")

    assert cli.main(["--root", str(root), "promote", str(staged), "--dry-run"]) == 1

    assert "cannot be added yet" in capsys.readouterr().out


def test_a_dry_run_admits_it_did_not_ask_the_dictionary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A row listed as landing may still be held when the source is really
    added, because the reading check costs a lookup per row and a preview does
    not spend one. Listing cards without saying so is the lie the flag exists
    to prevent."""
    monkeypatch.delenv("JPDB_API_KEY", raising=False)
    root = project(tmp_path, [])
    staged = root / "staging" / "in.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    write_staging(staged, [record()], {"source_file": "lesson.pdf"})

    assert cli.main(["--root", str(root), "promote", str(staged), "--dry-run"]) == 0

    assert "Readings were not checked" in capsys.readouterr().out


def test_a_dry_run_reports_rows_an_earlier_run_already_added(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Otherwise a retry looks like it would add the same cards twice."""
    root = project(tmp_path, [])
    staged = root / "staging" / "in.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    incoming = record()
    write_staging(staged, [incoming], {"source_file": "lesson.pdf"})
    _records, meta = read_staging(staged)
    done = staged.parent / "done"
    done.mkdir(parents=True, exist_ok=True)
    write_staging(done / staged.name, [incoming], promote.archive_meta(meta, 1))

    assert cli.main(["--root", str(root), "promote", str(staged), "--dry-run"]) == 0

    assert "Already added by an earlier run: 1" in capsys.readouterr().out
