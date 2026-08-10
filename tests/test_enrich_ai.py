"""Writing examples and usage notes — ``janki enrich --ai``.

No network (IMPLEMENTATION_PLAN rule 6): the Claude call is faked and jpdb is
driven through a fake transport, so the QC routing is exercised for real rather
than stubbed at the decision.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from japanese_anki import cli, enrich, qc
from japanese_anki.claude_client import CallResult, Refusal
from japanese_anki.enrich import (
    UNVERIFIED_KEY,
    ai_prompt,
    ai_schema,
    ai_targets,
    apply_ai_result,
    parse_force_fields,
)
from japanese_anki.identifiers import short_fingerprint
from japanese_anki.io import merge_records
from japanese_anki.jpdb import JpdbClient
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.staging import read_staging


def generated(japanese: str, furigana: str = "", english: str = "", romaji: str = "") -> Any:
    schema = ai_schema()
    item = schema.model_fields["examples"].annotation.__args__[0]
    return item(japanese=japanese, furigana=furigana, english=english, romaji=romaji)


def answer(*examples: Any, usage_notes: str = "") -> Any:
    return ai_schema()(examples=list(examples), usage_notes=usage_notes)


def ok(japanese: str, furigana: str = "", **kw: Any) -> CallResult:
    """A completed call carrying one generated example."""
    return CallResult(
        answer(generated(japanese, furigana=furigana), **kw), "end_turn", None
    )


def record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "verb_group": "godan",
        "source": SourceReference(type="shirabe", imported_from="export.csv"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


class FakeCall:
    """Stands in for ``claude_client.parse_call``."""

    def __init__(self, *results: CallResult) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, model: str, blocks: Any, content: Any, schema: Any, client: Any = None, **kw: Any
    ) -> CallResult:
        self.calls.append({"model": model, "system": blocks, "content": content})
        return self.results.pop(0) if self.results else CallResult(answer(), "end_turn", None)


class FakeJpdb:
    """Answers /parse from canned token data, and records what it was asked."""

    def __init__(self, parses: dict[str, list[Any]] | None = None) -> None:
        self.parses = parses or {}
        self.asked: list[str] = []

    def __call__(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> Any:
        if url.endswith("/parse"):
            text = body["text"][0]
            self.asked.append(text)
            tokens = self.parses.get(text)
            if tokens is None:
                return 200, {"tokens": [[]], "vocabulary": []}
            return 200, {"tokens": [tokens], "vocabulary": []}
        raise AssertionError(f"unexpected request to {url}")


def jpdb_for(api: FakeJpdb) -> JpdbClient:
    return JpdbClient("k", api, sleep=lambda _s: None, jitter=lambda: 0.0)


# 話します。 parsed the way jpdb would: one token, per-kanji furigana.
HANASHIMASU = [[0, [["話", "はな"], "します"]]]


# --- which records are targets ------------------------------------------------


def test_a_record_with_no_example_or_no_note_is_a_target() -> None:
    no_example = record(id="a", usage_notes="something")
    no_note = record(id="b", examples=[ExampleSentence(japanese="話す。")])
    complete = record(
        id="c", usage_notes="something", examples=[ExampleSentence(japanese="話す。")]
    )

    assert [r.id for r in ai_targets([no_example, no_note, complete])] == ["a", "b"]


def test_naming_ids_overrides_the_content_rule() -> None:
    # Re-running over a finished record is a legitimate ask; --force-fields is
    # what decides whether the answer may replace anything.
    complete = record(
        usage_notes="something", examples=[ExampleSentence(japanese="話す。")]
    )

    assert [r.id for r in ai_targets([complete], ["word:話す:はなす"])] == [
        "word:話す:はなす"
    ]


def test_an_unknown_id_is_an_error() -> None:
    with pytest.raises(enrich.EnrichError):
        ai_targets([record()], ["word:無い:ない"])


# --- --force-fields is per pass ----------------------------------------------


def test_the_ai_pass_has_its_own_writable_fields() -> None:
    assert parse_force_fields("examples,usage_notes", ai=True) == (
        "examples",
        "usage_notes",
    )


def test_naming_the_other_passs_field_says_which_pass_owns_it() -> None:
    with pytest.raises(enrich.EnrichError) as excinfo:
        parse_force_fields("pitch_accent", ai=True)

    assert "--jpdb field" in str(excinfo.value)


def test_reading_is_still_refused_by_name_for_the_ai_pass() -> None:
    with pytest.raises(enrich.EnrichError) as excinfo:
        parse_force_fields("reading", ai=True)

    assert "record ID" in str(excinfo.value)


# --- the prompt ---------------------------------------------------------------


def test_the_prompt_carries_what_janki_knows_about_the_word() -> None:
    # So the model writes about this word rather than a homograph.
    text = ai_prompt(record())

    assert "話す" in text and "はなす" in text and "to speak" in text
    assert "godan" in text


def test_recent_sentences_ride_along_as_variety_pressure() -> None:
    # Asked for twenty verbs in a row, a model writes twenty variations of
    # 毎日〜ます unless it can see that it already did.
    text = ai_prompt(record(), ["毎日話します。", "毎日食べます。"])

    assert "毎日話します。" in text
    assert "structurally different" in text


def test_the_patterns_the_learner_is_studying_ride_along_too() -> None:
    """The block reaching the prompt is the whole of what `janki patterns`
    does for `enrich --ai`; without it the command reads documents nothing
    consumes."""
    from japanese_anki.patterns import Pattern, format_patterns

    text = ai_prompt(record(), (), format_patterns([Pattern("〜んだ", "explains")]))

    assert "〜んだ" in text and "explains" in text


def test_no_reviewed_patterns_leaves_the_prompt_as_it_was() -> None:
    assert "currently studying" not in ai_prompt(record())


# --- the QC gate --------------------------------------------------------------


def test_an_example_without_the_word_is_rejected() -> None:
    outcome = apply_ai_result(record(), answer(generated("毎日言います。")))

    assert outcome.rejected == ["毎日言います。"]
    assert outcome.changes == {}


def test_a_conjugated_example_is_kept() -> None:
    outcome = apply_ai_result(
        record(),
        answer(generated("昨日話した。", furigana="昨日[きのう] 話[はな]した。")),
        parses={},
    )

    assert outcome.rejected == []
    assert [ex.japanese for ex in outcome.record.examples] == ["昨日話した。"]


def test_romaji_is_always_regenerated_never_taken() -> None:
    outcome = apply_ai_result(
        record(),
        answer(
            generated("話します。", furigana="話[はな]します。", romaji="totally wrong")
        ),
    )

    assert outcome.record.examples[0].romaji == "hanashimasu."


def test_furigana_jpdb_confirms_is_not_flagged(tmp_path: Path) -> None:
    parse = jpdb_for(FakeJpdb({"話します。": HANASHIMASU})).parse("話します。")

    outcome = apply_ai_result(
        record(),
        answer(generated("話します。", furigana="話[はな]します。")),
        parses={"話します。": parse},
    )

    assert outcome.unverified == []
    assert UNVERIFIED_KEY not in outcome.record.source.raw_fields


def test_a_furigana_mismatch_keeps_the_example_and_flags_it() -> None:
    # The sentence may be right where the segmentation is not, and a human
    # deciding that beats janki throwing away good Japanese.
    parse = jpdb_for(FakeJpdb({"話します。": HANASHIMASU})).parse("話します。")

    outcome = apply_ai_result(
        record(),
        answer(generated("話します。", furigana="話[か]します。")),
        parses={"話します。": parse},
    )

    assert [ex.japanese for ex in outcome.record.examples] == ["話します。"]
    assert outcome.unverified == ["話します。"]
    fingerprints = outcome.record.source.raw_fields[UNVERIFIED_KEY].split(",")
    assert fingerprints == [short_fingerprint("話します。")]


def test_an_unchecked_example_is_flagged_the_same_way() -> None:
    # No parse means nobody checked, which is what "unverified" means.
    outcome = apply_ai_result(
        record(), answer(generated("話します。", furigana="話[はな]します。")), parses={}
    )

    assert outcome.unverified == ["話します。"]


def test_the_flag_appends_rather_than_replacing() -> None:
    already = record(
        source=SourceReference(raw_fields={UNVERIFIED_KEY: "deadbeef"}),
        verb_group="godan",
    )

    outcome = apply_ai_result(
        already, answer(generated("話します。", furigana="話[か]します。")), parses={}
    )

    assert outcome.record.source.raw_fields[UNVERIFIED_KEY].startswith("deadbeef,")


def test_fields_that_already_have_content_are_left_alone() -> None:
    curated = record(
        examples=[ExampleSentence(japanese="curated")], usage_notes="curated note"
    )

    outcome = apply_ai_result(
        curated, answer(generated("話します。"), usage_notes="new note")
    )

    assert outcome.changes == {}
    assert outcome.record.usage_notes == "curated note"


def test_force_fields_lets_the_answer_replace_them() -> None:
    curated = record(
        examples=[ExampleSentence(japanese="curated")], usage_notes="curated note"
    )

    outcome = apply_ai_result(
        curated,
        answer(generated("話します。"), usage_notes="new note"),
        force_fields=("usage_notes",),
    )

    assert outcome.record.usage_notes == "new note"
    assert [ex.japanese for ex in outcome.record.examples] == ["curated"]


# --- the pass -----------------------------------------------------------------


def test_a_refusal_costs_that_record_not_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = FakeCall(
        CallResult(None, "refusal", Refusal("bio", "declined")),
        ok("食べます。", "食[た]べます。"),
    )
    monkeypatch.setattr(enrich.claude_client, "parse_call", call)
    first = record(id="word:話す:はなす")
    second = record(id="word:食べる:たべる", expression="食べる", reading="たべる",
                    verb_group="ichidan")

    result = enrich.enrich_ai([first, second], model="m", style_guide="S")

    assert "word:食べる:たべる" in result.changes
    assert "word:話す:はなす" not in result.changes
    assert "bio" in result.warnings[0]


def test_a_truncated_answer_is_never_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    # A half-written example is not a shorter example; it is a sentence that
    # stops mid-word, and accepting one would put it on a card.
    monkeypatch.setattr(
        enrich.claude_client, "parse_call", FakeCall(CallResult(None, "max_tokens", None))
    )

    result = enrich.enrich_ai([record()], model="m", style_guide="S")

    assert result.changes == {}
    assert "max_tokens" in result.warnings[0]


def test_every_generated_sentence_is_parsed_for_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        enrich.claude_client,
        "parse_call",
        FakeCall(
            ok("話します。", "話[はな]します。")
        ),
    )
    api = FakeJpdb({"話します。": HANASHIMASU})

    enrich.enrich_ai(
        [record()], model="m", style_guide="S", jpdb_client=jpdb_for(api)
    )

    assert api.asked == ["話します。"]


def test_variety_pressure_grows_as_the_run_goes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = FakeCall(
        ok("毎日話します。", "毎日[まいにち] 話[はな]します。"),
        ok("毎日食べます。", "毎日[まいにち] 食[た]べます。"),
    )
    monkeypatch.setattr(enrich.claude_client, "parse_call", call)
    records = [
        record(),
        record(id="word:食べる:たべる", expression="食べる", reading="たべる",
               verb_group="ichidan"),
    ]

    enrich.enrich_ai(records, model="m", style_guide="S")

    assert "毎日話します。" not in call.calls[0]["content"]
    assert "毎日話します。" in call.calls[1]["content"]


# --- the CLI ------------------------------------------------------------------


def project(tmp_path: Path, records: list[VocabularyRecord]) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "JAPANESE_STYLE_GUIDE.md").write_text("Guide.", encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([item.to_dict() for item in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return tmp_path


def patch_all(monkeypatch: pytest.MonkeyPatch, call: FakeCall, api: FakeJpdb) -> None:
    monkeypatch.setenv("JPDB_API_KEY", "k")
    monkeypatch.setattr(cli.enrich.claude_client, "parse_call", call)
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda key, *a, **kw: jpdb_for(api))


def stored(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload}


def test_enrich_ai_writes_records_the_diff_and_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    call = FakeCall(
        CallResult(
            answer(
                generated("話します。", furigana="話[はな]します。", english="I speak."),
                usage_notes="Polite present.",
            ),
            "end_turn",
            None,
        )
    )
    patch_all(monkeypatch, call, FakeJpdb({"話します。": HANASHIMASU}))

    code = cli.main(["--root", str(root), "enrich", "--ai", "--yes"])

    assert code == 0
    written = stored(root)["word:話す:はなす"]
    assert written["examples"][0]["japanese"] == "話します。"
    assert written["examples"][0]["romaji"] == "hanashimasu."
    assert written["usage_notes"] == "Polite present."
    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    passes = book["records"]["word:話す:はなす"]["enriched"]
    assert [item["kind"] for item in passes] == ["ai"]
    assert "Enriched 1 record(s)" in capsys.readouterr().out


def test_a_declined_diff_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [record()])
    patch_all(
        monkeypatch,
        FakeCall(ok("話します。", "話[はな]します。")),
        FakeJpdb({"話します。": HANASHIMASU}),
    )
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _p: "n")

    assert cli.main(["--root", str(root), "enrich", "--ai"]) == 1

    assert stored(root)["word:話す:はなす"]["examples"] == []


def test_a_large_run_goes_to_a_staging_file_instead_of_a_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A 500-record y/n is not review; a staging file is.
    many = [
        record(id=f"word:話す{index}:はなす", expression=f"話す{index}")
        for index in range(enrich.STAGING_THRESHOLD)
    ]
    root = project(tmp_path, many)
    call = FakeCall(
        *[
            CallResult(
                answer(generated(f"話す{index}。", furigana=f"話[はな]す{index}。")),
                "end_turn",
                None,
            )
            for index in range(enrich.STAGING_THRESHOLD)
        ]
    )
    patch_all(monkeypatch, call, FakeJpdb())

    code = cli.main(["--root", str(root), "enrich", "--ai", "--yes"])

    assert code == 0
    target = root / "staging" / "ai-enrichment.yaml"
    assert target.is_file()
    staged, meta = read_staging(target)
    assert len(staged) == enrich.STAGING_THRESHOLD
    # Nothing reached the records; promote is what lands them.
    assert stored(root)["word:話す0:はなす"]["examples"] == []
    assert "janki promote" in capsys.readouterr().out
    assert UNVERIFIED_KEY in meta["review_notes"]


def test_the_staging_route_records_already_exist_so_promote_updates_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    many = [
        record(id=f"word:話す{index}:はなす", expression=f"話す{index}")
        for index in range(enrich.STAGING_THRESHOLD)
    ]
    root = project(tmp_path, many)
    patch_all(
        monkeypatch,
        FakeCall(
            *[
                CallResult(
                    answer(generated(f"話す{index}。", furigana=f"話[はな]す{index}。")),
                    "end_turn",
                    None,
                )
                for index in range(enrich.STAGING_THRESHOLD)
            ]
        ),
        FakeJpdb(),
    )
    cli.main(["--root", str(root), "enrich", "--ai", "--yes"])

    staged, _ = read_staging(root / "staging" / "ai-enrichment.yaml")

    # Same ids as the records they came from, so promote merges rather than adds.
    assert {item.id for item in staged} <= {item.id for item in many}


def test_nothing_to_do_is_said_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    complete = record(
        usage_notes="note", examples=[ExampleSentence(japanese="話す。")]
    )
    root = project(tmp_path, [complete])
    call = FakeCall()
    patch_all(monkeypatch, call, FakeJpdb())

    assert cli.main(["--root", str(root), "enrich", "--ai", "--yes"]) == 0

    assert call.calls == []
    assert "already has examples" in capsys.readouterr().out


def test_the_model_can_be_overridden_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [record()])
    call = FakeCall(ok("話します。", "話[はな]します。"))
    patch_all(monkeypatch, call, FakeJpdb({"話します。": HANASHIMASU}))

    cli.main(
        ["--root", str(root), "enrich", "--ai", "--yes", "--model", "claude-haiku-4-5"]
    )

    assert call.calls[0]["model"] == "claude-haiku-4-5"


def test_a_flagged_example_is_reported_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    patch_all(
        monkeypatch,
        FakeCall(
            ok("話します。", "話[か]します。")
        ),
        FakeJpdb({"話します。": HANASHIMASU}),
    )

    cli.main(["--root", str(root), "enrich", "--ai", "--yes"])

    captured = capsys.readouterr()
    assert "jpdb did not confirm" in captured.err
    fields = stored(root)["word:話す:はなす"]["source"]["raw_fields"]
    assert fields[UNVERIFIED_KEY] == short_fingerprint("話します。")


def test_a_rejected_example_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    patch_all(
        monkeypatch,
        FakeCall(CallResult(answer(generated("毎日言います。")), "end_turn", None)),
        FakeJpdb(),
    )

    cli.main(["--root", str(root), "enrich", "--ai", "--yes"])

    assert "did not contain 話す" in capsys.readouterr().err


def test_the_staging_file_is_not_overwritten_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    many = [
        record(id=f"word:話す{index}:はなす", expression=f"話す{index}")
        for index in range(enrich.STAGING_THRESHOLD)
    ]
    root = project(tmp_path, many)
    (root / "staging").mkdir()
    (root / "staging" / "ai-enrichment.yaml").write_text(
        "records: []\nreview_notes: mid-review\n", encoding="utf-8"
    )
    patch_all(
        monkeypatch,
        FakeCall(
            *[
                CallResult(
                    answer(generated(f"話す{index}。", furigana=f"話[はな]す{index}。")),
                    "end_turn",
                    None,
                )
                for index in range(enrich.STAGING_THRESHOLD)
            ]
        ),
        FakeJpdb(),
    )

    assert cli.main(["--root", str(root), "enrich", "--ai", "--yes"]) == 1

    text = (root / "staging" / "ai-enrichment.yaml").read_text(encoding="utf-8")
    assert "mid-review" in text


def test_the_staging_file_reads_back_through_the_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    many = [
        record(id=f"word:話す{index}:はなす", expression=f"話す{index}")
        for index in range(enrich.STAGING_THRESHOLD)
    ]
    root = project(tmp_path, many)
    patch_all(
        monkeypatch,
        FakeCall(
            *[
                CallResult(
                    answer(
                        generated(f"話す{index}。", furigana=f"話[はな]す{index}。"),
                        usage_notes="note",
                    ),
                    "end_turn",
                    None,
                )
                for index in range(enrich.STAGING_THRESHOLD)
            ]
        ),
        FakeJpdb(),
    )
    cli.main(["--root", str(root), "enrich", "--ai", "--yes"])

    raw = yaml.safe_load(
        (root / "staging" / "ai-enrichment.yaml").read_text(encoding="utf-8")
    )

    assert raw["model"] == "claude-opus-5"
    assert raw["records"][0]["usage_notes"] == "note"


def test_a_polite_example_is_accepted(tmp_path: Path) -> None:
    # The style guide asks for beginner examples and a beginner textbook
    # teaches 〜ます first, so without the polite forms this check would reject
    # almost every good sentence a model writes.
    for sentence in ("話します。", "話しました。", "話しません。", "話しましょう。"):
        outcome = apply_ai_result(record(), answer(generated(sentence)))
        assert outcome.rejected == [], sentence


def test_the_polite_stem_alone_is_not_enough_to_count() -> None:
    # Matching the bare stem would let 食べ物 count as an example of 食べる.
    taberu = record(
        id="word:食べる:たべる", expression="食べる", reading="たべる", verb_group="ichidan"
    )

    outcome = apply_ai_result(taberu, answer(generated("食べ物が好きです。")))

    assert outcome.rejected == ["食べ物が好きです。"]


# --- what the review found ---------------------------------------------------


def test_the_unverified_flag_survives_the_staging_route_into_the_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The large-run route is the *only* one big runs may take, so the flag has
    to reach vocabulary.json through it. It rides in ``source.raw_fields``, and
    a merge keeps the existing record's source — which is right for provenance
    and wrong for a flag describing the examples arriving with it."""
    many = [
        record(id=f"word:話す{index}:はなす", expression=f"話す{index}")
        for index in range(enrich.STAGING_THRESHOLD)
    ]
    root = project(tmp_path, many)
    # FakeJpdb answers no parse, so every generated example is unverified.
    patch_all(
        monkeypatch,
        FakeCall(
            *[
                CallResult(
                    answer(generated(f"話す{index}。", furigana=f"話[はな]す{index}。")),
                    "end_turn",
                    None,
                )
                for index in range(enrich.STAGING_THRESHOLD)
            ]
        ),
        FakeJpdb(),
    )
    target = root / "staging" / "ai-enrichment.yaml"
    assert cli.main(["--root", str(root), "enrich", "--ai", "--yes"]) == 0
    staged, _ = read_staging(target)
    assert all(UNVERIFIED_KEY in item.source.raw_fields for item in staged)

    assert (
        cli.main(["--root", str(root), "promote", str(target), "--skip-reading-check"])
        == 0
    )

    landed = stored(root)["word:話す0:はなす"]
    assert landed["examples"], "the examples merged in"
    assert UNVERIFIED_KEY in landed["source"]["raw_fields"], (
        "M5.3 reads this key to decide whether to speak a sentence nobody checked"
    )


def test_the_flag_does_not_ride_along_when_the_examples_did_not_land() -> None:
    """A flag describes the examples it arrived with. If the merge kept the
    existing curated examples, saying they are unverified would be a lie."""
    curated = record(examples=[ExampleSentence(japanese="人と話す。")])
    incoming = replace(
        record(examples=[ExampleSentence(japanese="話しました。")]),
        source=replace(curated.source, raw_fields={UNVERIFIED_KEY: "abc123"}),
    )

    merged, _ = merge_records([curated], [incoming], ())

    assert merged[0].examples == curated.examples
    assert UNVERIFIED_KEY not in merged[0].source.raw_fields


def test_the_diff_shows_the_sentence_the_user_is_saying_yes_to() -> None:
    # Long enough that a dataclass repr would be truncated before the sentence
    # ended: the reviewer has to read the Japanese, not the field names.
    sentence = ExampleSentence(
        japanese="毎朝、友だちと日本語で話します。",
        furigana="毎朝[まいあさ]、友[とも]だちと日本語[にほんご]で話[はな]します。",
        english="Every morning I speak Japanese with my friend.",
    )
    lines = enrich.format_field_diff(
        {"word:話す:はなす": {"examples": ([], [sentence]), "usage_notes": ("", "Polite.")}}
    )

    assert lines[0] == "word:話す:はなす"
    assert any("毎朝、友だちと日本語で話します。" in line for line in lines[1:])
    assert any("Polite." in line for line in lines[1:])
    assert not any("ExampleSentence(" in line for line in lines)


def test_a_change_the_line_cannot_show_says_so() -> None:
    """Same sentence, different English: the rendered line is identical on both
    sides, so without this the user confirms an overwrite of curated text they
    were never shown."""
    japanese = "日本語を話します。"
    lines = enrich.format_field_diff(
        {
            "word:話す:はなす": {
                "examples": (
                    [ExampleSentence(japanese=japanese, english="I speak Japanese.")],
                    [ExampleSentence(japanese=japanese, english="I will speak Japanese.")],
                )
            }
        }
    )

    assert "english differ" in lines[1]


def test_one_visibly_new_sentence_does_not_hide_the_other_one_being_rewritten() -> None:
    """A list can change visibly in one element and invisibly in another. Only
    the invisible part needs saying, and it still needs saying."""
    kept = "日本語を話します。"
    lines = enrich.format_field_diff(
        {
            "word:話す:はなす": {
                "examples": (
                    [
                        ExampleSentence(japanese=kept, english="I speak Japanese."),
                        ExampleSentence(japanese="友だちと話しました。"),
                    ],
                    [
                        ExampleSentence(japanese=kept, english="I talk in Japanese."),
                        ExampleSentence(japanese="先生と話しました。"),
                    ],
                )
            }
        }
    )

    assert "先生と話しました。" in lines[1]
    assert "english differ" in lines[1]


def examples_diff(before: list[ExampleSentence], after: list[ExampleSentence]) -> str:
    return enrich.format_field_diff({"word:話す:はなす": {"examples": (before, after)}})[1]


def test_a_longer_answer_does_not_switch_the_check_off() -> None:
    """A model returning a different number of examples is the ordinary case,
    and it is exactly when the whole list is being rewritten."""
    kept = "日本語を話します。"
    line = examples_diff(
        [ExampleSentence(japanese=kept, english="I speak Japanese.")],
        [
            ExampleSentence(japanese=kept, english="I talk in Japanese."),
            ExampleSentence(japanese="先生と話しました。"),
        ],
    )

    assert "english differ" in line


def test_reordering_does_not_switch_the_check_off() -> None:
    kept = "日本語を話します。"
    line = examples_diff(
        [ExampleSentence(japanese="友だちと話す。"), ExampleSentence(japanese=kept, romaji="x")],
        [ExampleSentence(japanese=kept, romaji="y"), ExampleSentence(japanese="友だちと話す。")],
    )

    assert "romaji differ" in line


def test_an_example_that_was_simply_dropped_is_not_called_a_hidden_change() -> None:
    line = examples_diff(
        [ExampleSentence(japanese="友だちと話す。"), ExampleSentence(japanese="先生と話す。")],
        [ExampleSentence(japanese="友だちと話す。")],
    )

    assert "differ" not in line


def test_a_field_that_really_did_not_change_is_not_annotated() -> None:
    sentence = ExampleSentence(japanese="話します。", english="I speak.")
    lines = enrich.format_field_diff(
        {"word:話す:はなす": {"examples": ([], [sentence]), "usage_notes": ("", "Polite.")}}
    )

    assert not any("differ" in line for line in lines)


def test_a_changed_field_the_helper_does_not_know_still_shows() -> None:
    """A field diff that silently omits a change is the display version of
    discarding a row: the user confirms a write they were never shown."""
    lines = enrich.format_field_diff({"word:話す:はなす": {"transitivity": ("", "vi")}})

    assert any("transitivity" in line for line in lines[1:])


def test_ai_and_staging_are_not_the_same_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    staging_file = root / "held.yaml"
    staging_file.write_text("records: []\n", encoding="utf-8")

    assert cli.main(
        ["--root", str(root), "enrich", "--ai", "--staging", str(staging_file)]
    ) == 1
    assert "--ai" in capsys.readouterr().err


def test_an_existing_staging_file_is_refused_before_the_pass_is_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    many = [
        record(id=f"word:話す{index}:はなす", expression=f"話す{index}")
        for index in range(enrich.STAGING_THRESHOLD)
    ]
    root = project(tmp_path, many)
    (root / "staging").mkdir()
    (root / "staging" / "ai-enrichment.yaml").write_text("records: []\n", encoding="utf-8")
    call = FakeCall()
    patch_all(monkeypatch, call, FakeJpdb())

    assert cli.main(["--root", str(root), "enrich", "--ai", "--yes"]) == 1

    assert call.calls == [], "the file exists; nothing should have been generated"


def test_an_example_that_was_not_kept_is_not_reported_as_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A record with curated examples and no usage note is a target for the
    note. The example that comes back with it is discarded, so claiming it was
    flagged points a reviewer at a key that is not there."""
    curated = record(examples=[ExampleSentence(japanese="人と話す。")])
    root = project(tmp_path, [curated])
    patch_all(
        monkeypatch,
        FakeCall(
            CallResult(
                answer(
                    generated("話します。", furigana="話[はな]します。"),
                    usage_notes="Polite.",
                ),
                "end_turn",
                None,
            )
        ),
        FakeJpdb(),
    )

    assert cli.main(["--root", str(root), "enrich", "--ai", "--yes"]) == 0

    landed = stored(root)["word:話す:はなす"]
    assert landed["usage_notes"] == "Polite."
    assert UNVERIFIED_KEY not in landed["source"]["raw_fields"]
    output = capsys.readouterr()
    assert "flagged" not in (output.out + output.err)


def test_an_honorific_polite_example_is_accepted() -> None:
    example = ExampleSentence(japanese="先生は教室にいらっしゃいます。")

    assert qc.example_contains_target(example, "いらっしゃる", "godan")


def test_a_pass_flag_is_not_silently_ignored_by_the_other_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("JPDB_API_KEY", "k")
    root = project(tmp_path, [record()])

    assert cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--model", "claude-opus-5"]
    ) == 1
    assert "--model" in capsys.readouterr().err
    assert cli.main(["--root", str(root), "enrich", "--jpdb", "--force"]) == 1
    assert "--force" in capsys.readouterr().err


def test_a_failed_ledger_write_does_not_call_a_re_run_pointless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A note is written only when the model has one worth writing, so a record
    can land with examples and no note — and it is a target again next time.
    Telling the user a re-run skips it is both wrong and expensive to believe."""
    root = project(tmp_path, [record()])
    patch_all(monkeypatch, FakeCall(ok("話します。", furigana="話[はな]します。")), FakeJpdb())
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("disk full")),
    )

    assert cli.main(["--root", str(root), "enrich", "--ai", "--yes"]) == 1

    err = capsys.readouterr().err
    assert "'status --rebuild' cannot bring it back" in err
    assert "not a free repair" in err
    assert "either skips these records" not in err
    assert stored(root)["word:話す:はなす"]["examples"]


# --- the separator space, on the way in ---------------------------------------


def test_a_swallowed_comma_is_repaired_before_anything_reads_the_furigana() -> None:
    """A model writes `週末[しゅうまつ]、何[なに]` without the separator perhaps
    half the time. Anki then draws なに over `、何`, and the comma disappears
    from the reading the romaji and the sentence audio are built from — so the
    repair has to happen before the example is kept, not in a later pass over
    the file."""
    outcome = apply_ai_result(
        record(),
        answer(generated(
            "週末、何を話すの？", furigana="週末[しゅうまつ]、何[なに]を 話[はな]すの？"
        )),
    )

    kept = outcome.record.examples[0]
    assert kept.furigana == "週末[しゅうまつ]、 何[なに]を 話[はな]すの？"
    assert kept.romaji == "shuumatsu, naniohanasuno?", "the comma survives into the romaji"


def test_a_spill_needing_a_guess_is_kept_as_written() -> None:
    """`、妻と日本語[にほんご]` needs someone to decide where the word starts.
    It stays exactly as the model wrote it, for `janki validate` to report."""
    # No space before 日本語: with one there is no spill, and the test passed
    # for a reason unrelated to the behaviour it names.
    written = "毎日[まいにち]、妻と日本語[にほんご]を 話[はな]す"
    outcome = apply_ai_result(
        record(), answer(generated("毎日、妻と日本語を話す", furigana=written))
    )

    assert outcome.record.examples[0].furigana == written
