"""`enrich --recheck-furigana`: re-asking jpdb about examples already written.

The flag is written once, when an example is created, and read much later by
`janki audio` deciding whether to speak a sentence. Nothing else re-asks — so
before this pass existed, a flag left by a check that had since improved could
only be cleared by rewriting the sentence and paying for it again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import cli, enrich
from japanese_anki.errors import JankiError
from japanese_anki.jpdb import JpdbError
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord

SENTENCE = "橋を渡る。"
OTHER = "川に橋がある。"


class FakeJpdb:
    """Answers `parse` from a table, and records what it was asked."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def parse(self, sentence: str) -> Any:
        self.asked.append(sentence)
        answer = self.answers.get(sentence)
        if answer is None:
            raise JpdbError("jpdb has nothing for this")
        return answer


class Parse:
    def __init__(self, tokens: list[Any]) -> None:
        self.tokens = tokens


def parse_of(*segments: Any) -> Parse:
    return Parse([{"furigana": list(segments)}])


def record(*examples: ExampleSentence, flagged: str = "") -> VocabularyRecord:
    raw = {"furigana_unverified": flagged} if flagged else {}
    return VocabularyRecord(
        id="word:橋:はし",
        expression="橋",
        reading="はし",
        meanings=["bridge"],
        examples=list(examples),
        source=SourceReference(type="manual", raw_fields=raw),
    )


def fingerprint(sentence: str) -> str:
    from japanese_anki.identifiers import short_fingerprint

    return short_fingerprint(sentence)


# --- clearing ---------------------------------------------------------------


def test_a_flagged_example_jpdb_now_vouches_for_is_cleared() -> None:
    example = ExampleSentence(japanese=SENTENCE, furigana="橋[はし]を 渡[わた]る。")
    subject = record(example, flagged=fingerprint(SENTENCE))
    client = FakeJpdb({SENTENCE: parse_of(["橋", "はし"], "を", ["渡", "わた"], "る")})

    result = enrich.recheck_furigana([subject], jpdb_client=client)

    assert result.cleared == {"word:橋:はし": [SENTENCE]}
    raw = result.records[0].source.raw_fields
    assert "furigana_unverified" not in raw, "the last flag takes the key with it"


def test_one_of_two_flags_clearing_leaves_the_other() -> None:
    """Partial clearing has to keep the fingerprint of the example that still
    differs — dropping it would voice a sentence nobody vouched for."""
    good = ExampleSentence(japanese=SENTENCE, furigana="橋[はし]を 渡[わた]る。")
    bad = ExampleSentence(japanese=OTHER, furigana="川[かわ]に 橋[きょう]がある。")
    subject = record(good, bad, flagged=f"{fingerprint(SENTENCE)},{fingerprint(OTHER)}")
    client = FakeJpdb({
        SENTENCE: parse_of(["橋", "はし"], "を", ["渡", "わた"], "る"),
        OTHER: parse_of(["川", "かわ"], "に", ["橋", "はし"], "がある"),
    })

    result = enrich.recheck_furigana([subject], jpdb_client=client)

    assert result.cleared == {"word:橋:はし": [SENTENCE]}
    assert result.records[0].source.raw_fields["furigana_unverified"] == fingerprint(OTHER)
    assert result.differing["word:橋:はし"][0][0] == OTHER


def test_an_example_nobody_flagged_is_never_re_parsed() -> None:
    """"Only ever cleared": an example nobody doubted is not put in doubt by a
    parse that happens to fail today, and is not worth a request either."""
    wrong = ExampleSentence(japanese=SENTENCE, furigana="橋[きょう]を 渡[わた]る。")
    subject = record(wrong)
    client = FakeJpdb({SENTENCE: parse_of(["橋", "はし"], "を", ["渡", "わた"], "る")})

    result = enrich.recheck_furigana([subject], jpdb_client=client)

    assert client.asked == [], "not even asked about"
    assert result.cleared == {} and result.differing == {}


def test_a_parse_failure_clears_nothing_and_is_reported() -> None:
    """Nothing means unverified rather than fine — the same rule the AI pass
    applies to an absent client."""
    example = ExampleSentence(japanese=SENTENCE, furigana="橋[はし]を 渡[わた]る。")
    subject = record(example, flagged=fingerprint(SENTENCE))
    client = FakeJpdb({})

    result = enrich.recheck_furigana([subject], jpdb_client=client)

    assert result.unparsed == [SENTENCE]
    assert result.cleared == {}
    assert result.records[0].source.raw_fields["furigana_unverified"], "the flag survives"


def test_an_id_that_names_no_record_is_refused() -> None:
    """A typo read as a clean negative result: "nothing to confirm", exit 0.
    Every other ids-taking pass in this module refuses the same way."""
    subject = record(ExampleSentence(japanese=SENTENCE), flagged=fingerprint(SENTENCE))

    with pytest.raises(enrich.EnrichError, match="word:わかる:わかる"):
        enrich.recheck_furigana(
            [subject], jpdb_client=FakeJpdb({}), ids=["word:わかる:わかる"]
        )


# --- through the CLI --------------------------------------------------------


def _project(tmp_path: Path, records: list[VocabularyRecord]) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([r.to_dict() for r in records], ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


def test_the_command_writes_records_then_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    example = ExampleSentence(japanese=SENTENCE, furigana="橋[はし]を 渡[わた]る。")
    root = _project(tmp_path, [record(example, flagged=fingerprint(SENTENCE))])
    client = FakeJpdb({SENTENCE: parse_of(["橋", "はし"], "を", ["渡", "わた"], "る")})
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda *a, **k: client)
    monkeypatch.setattr(cli.jpdb, "api_key_from_env", lambda *a, **k: "test-key")

    assert cli.main(["--root", str(root), "enrich", "--recheck-furigana"]) == 0

    stored = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    assert "furigana_unverified" not in (stored[0]["source"].get("raw_fields") or {})
    entries = json.loads((root / "ledger.json").read_text(encoding="utf-8"))["records"]
    fields = entries["word:橋:はし"]["enriched"][0]["fields"]
    # Not "furigana": this pass writes no record field, it clears an example's
    # flag — and `record_enriched` dedups on (kind, model, fields), so claiming
    # "furigana" would let a later real --jpdb pass collapse into this entry.
    assert fields == ["furigana_unverified"]
    assert "Confirmed 1 example" in capsys.readouterr().out


def test_a_run_where_jpdb_answered_nothing_does_not_report_a_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An expired key or a downed API made every parse fail, and the command
    still printed "not vouched for yet" and exited 0 — asserting a judgment it
    never obtained."""
    example = ExampleSentence(japanese=SENTENCE, furigana="橋[はし]を 渡[わた]る。")
    root = _project(tmp_path, [record(example, flagged=fingerprint(SENTENCE))])
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda *a, **k: FakeJpdb({}))
    monkeypatch.setattr(cli.jpdb, "api_key_from_env", lambda *a, **k: "test-key")

    assert cli.main(["--root", str(root), "enrich", "--recheck-furigana"]) == 1

    assert "could not parse any" in capsys.readouterr().err


# `romaji` rather than an invalid field name, so what refuses is the guard
# under test and not `parse_force_fields`.
@pytest.mark.parametrize(
    "extra", [["--staging", "x.yaml"], ["--force-fields", "romaji"]]
)
def test_a_flag_this_pass_never_reads_is_refused(
    tmp_path: Path, extra: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """"A flag the running pass never reads is a typo, not a no-op" — the rule
    this file already enforces for --model and --force."""
    root = _project(tmp_path, [record(ExampleSentence(japanese=SENTENCE))])

    assert cli.main(["--root", str(root), "enrich", "--recheck-furigana", *extra]) == 1

    assert "have nothing to act on" in capsys.readouterr().err


def test_nothing_flagged_says_so_rather_than_blaming_jpdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path, [record(ExampleSentence(japanese=SENTENCE))])
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda *a, **k: FakeJpdb({}))
    monkeypatch.setattr(cli.jpdb, "api_key_from_env", lambda *a, **k: "test-key")

    assert cli.main(["--root", str(root), "enrich", "--recheck-furigana"]) == 0

    assert "no example carries an unverified-furigana flag" in capsys.readouterr().out


def test_a_missing_key_is_a_janki_error_not_a_traceback(tmp_path: Path) -> None:
    root = _project(tmp_path, [record(ExampleSentence(japanese=SENTENCE))])

    with pytest.raises(JankiError):
        cli.command_enrich(
            cli.build_parser().parse_args(
                ["--root", str(root), "enrich", "--recheck-furigana", "--staging", "x"]
            )
        )


# --- adjudication -----------------------------------------------------------


class FakeAdjudicator:
    """Stands in for the model, recording what it was asked to settle."""

    def __init__(self, verdict: str, why: str = "") -> None:
        self.verdict, self.why = verdict, why
        self.asked: list[tuple[str, str, str]] = []

    def __call__(self, sentence, dictionary_reading, writer_reading, **_kwargs):
        self.asked.append((sentence, dictionary_reading, writer_reading))
        return self.verdict, self.why


def _disagreeing() -> tuple[VocabularyRecord, FakeJpdb]:
    """A record whose example jpdb reads differently — the 日本語 case."""
    example = ExampleSentence(japanese="日本語", furigana="日本語[にほんご]")
    subject = VocabularyRecord(
        id="word:日本語:にほんご", expression="日本語", reading="にほんご",
        meanings=["Japanese"], examples=[example],
        source=SourceReference(type="manual", raw_fields={
            "furigana_unverified": fingerprint("日本語")}),
    )
    client = FakeJpdb({"日本語": parse_of(["日", "にっ"], ["本", "ぽん"], ["語", "ご"])})
    return subject, client


def test_the_adjudicator_can_overrule_jpdb(monkeypatch: pytest.MonkeyPatch) -> None:
    """jpdb's parse reads 日本語 as にっぽんご and the language is にほんご. Before
    this, the only ways out were writing a reading nobody uses onto a card or
    leaving a correct sentence unvoiced."""
    subject, client = _disagreeing()
    judge = FakeAdjudicator("writer", "にほんご is the standard reading")
    monkeypatch.setattr(enrich, "adjudicate_reading", judge)

    result = enrich.recheck_furigana(
        [subject], jpdb_client=client, adjudicate_model="test-model"
    )

    assert result.cleared == {"word:日本語:にほんご": ["日本語"]}
    assert result.adjudicated["word:日本語:にほんご"][0][1] == "にほんご is the standard reading"
    assert judge.asked, "it was actually consulted"


def test_the_adjudicator_siding_with_jpdb_leaves_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject, client = _disagreeing()
    monkeypatch.setattr(enrich, "adjudicate_reading", FakeAdjudicator("dictionary", "jpdb"))

    result = enrich.recheck_furigana(
        [subject], jpdb_client=client, adjudicate_model="test-model"
    )

    assert result.cleared == {}
    assert "adjudicator: dictionary" in result.differing["word:日本語:にほんご"][0][1]


def test_unsure_is_a_real_answer_and_keeps_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adjudicator that cannot answer must leave the flag alone. An unsure
    verdict costs a re-check; a confident wrong one puts a reading nobody uses
    onto a card."""
    subject, client = _disagreeing()
    monkeypatch.setattr(enrich, "adjudicate_reading", FakeAdjudicator("unsure", "rare word"))

    result = enrich.recheck_furigana(
        [subject], jpdb_client=client, adjudicate_model="test-model"
    )

    assert result.cleared == {}
    assert result.adjudicated == {}


def test_no_model_configured_means_no_adjudication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject, client = _disagreeing()
    judge = FakeAdjudicator("writer")
    monkeypatch.setattr(enrich, "adjudicate_reading", judge)

    result = enrich.recheck_furigana([subject], jpdb_client=client, adjudicate_model="")

    assert judge.asked == [], "not consulted at all"
    assert result.cleared == {}


def test_an_agreeing_example_is_never_adjudicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adjudication is for disagreements. Asking about one jpdb already
    confirmed spends a request to be told what is already known."""
    example = ExampleSentence(japanese=SENTENCE, furigana="橋[はし]を 渡[わた]る。")
    subject = record(example, flagged=fingerprint(SENTENCE))
    client = FakeJpdb({SENTENCE: parse_of(["橋", "はし"], "を", ["渡", "わた"], "る")})
    judge = FakeAdjudicator("writer")
    monkeypatch.setattr(enrich, "adjudicate_reading", judge)

    result = enrich.recheck_furigana(
        [subject], jpdb_client=client, adjudicate_model="test-model"
    )

    assert result.cleared and judge.asked == []
