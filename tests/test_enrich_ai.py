"""Writing examples and usage notes — ``janki enrich --ai``.

No network (IMPLEMENTATION_PLAN rule 6): the model call is faked, so the QC
routing is exercised for real rather than stubbed at the decision. The AI pass asks jpdb
nothing since M7.6V, so `patch_all`'s client swap and its JPDB_API_KEY are
vestigial — kept only because removing the key would be a second change, and
noted here because that key is what hid the pass's own key dependency until
`test_the_ai_pass_enriches_a_record_with_no_jpdb_key` was written without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import seed_prompts
from japanese_anki import cli, enrich, prompts
from japanese_anki.claude_client import CallResult, Refusal
from japanese_anki.enrich import (
    ai_prompt,
    ai_schema,
    ai_targets,
    apply_ai_result,
    parse_force_fields,
)
from japanese_anki.identifiers import short_fingerprint
from japanese_anki.jpdb import JpdbClient
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.staging import read_staging

REPO_ROOT = Path(__file__).resolve().parents[1]


def generated(
    japanese: str,
    furigana: str = "",
    english: str = "",
    romaji: str = "",
    **extra: Any,
) -> Any:
    """One generated example. ``extra`` reaches the schema item verbatim, which
    is how a test states a `speech_level` other than the default.

    Names checked against the schema first: the model config does not forbid
    extras, so pydantic *ignores* an unknown key. A misspelled `speech_level`
    would silently leave the default in place and turn a test asserting the
    register precedence into one comparing the default with itself."""
    schema = ai_schema()
    item = schema.model_fields["examples"].annotation.__args__[0]
    unknown = sorted(set(extra) - set(item.model_fields))
    if unknown:
        raise TypeError(f"not a field of the generated-example schema: {unknown}")
    return item(
        japanese=japanese, furigana=furigana, english=english, romaji=romaji, **extra
    )


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
    """Stands in for either provider's structured call."""

    def __init__(self, *results: CallResult) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, model: str, blocks: Any, content: Any, schema: Any, client: Any = None, **kw: Any
    ) -> CallResult:
        self.calls.append(
            {"model": model, "system": blocks, "content": content, **kw}
        )
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
    assert "Expression: 話す" in text


def test_the_instructions_preserve_the_exact_headword_spelling() -> None:
    """Read from `prompts/enrich-examples.md`, the file that is actually sent.

    Asserting against a Python constant was the bug M7.6P was supposed to end
    and did not: the constant survived the commit, byte-identical to the file,
    so these assertions guarded a copy nothing sends. Deleting the clause from
    the shipped file left the suite green.
    """
    shipped = prompts.load(REPO_ROOT, "enrich-examples")

    assert "Use the exact spelling shown in Expression" in shipped
    assert "do not\nreplace a kana-only expression with kanji" in shipped


def test_the_instructions_state_the_furigana_notation_contract() -> None:
    """The only thing standing between a malformed ruby field and a card.

    Ported from the gating case `furigana-full-width-separator` when M8.4
    deleted the corpus, and the one clause in this prompt nobody had written a
    test for. Nothing audits the notation afterwards — the M8 deletions retired
    the spill warning, its repair, and the paid review — while
    `regenerate_example_romaji` still *reads* the field, so a group whose
    separator Anki cannot read silently drops a word from the reconstructed
    reading, from the romaji, and from the audio. Asking is the whole of the
    protection, so the asking is what gets pinned.

    Two assertions, because the halves fail independently: without the base
    rule the base widens (話 becomes 毎日話), and without the separator rule the
    space goes missing — and either one alone produces the same silent drop.

    Read from the shipped file plus the assembled user turn. The constant this
    once read is gone — it survived M7.6P's first commit and made these
    assertions guard a copy nothing sends.
    """
    text = prompts.load(REPO_ROOT, "enrich-examples") + "\n" + ai_prompt(record())

    assert "The base is exactly the characters that reading" in text
    assert "an ASCII space separates each group" in text


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


def test_the_prompt_binds_an_incomplete_curated_example_exactly() -> None:
    # The default helper record is a Shirabe import: the user's own data,
    # curated by arrival, so its incomplete example is pinned for annotation.
    source_example = record(
        examples=[ExampleSentence(japanese="日本語を話します。")]
    )

    text = ai_prompt(source_example)

    assert "Existing curated examples need annotations" in text
    assert '"日本語を話します。"' in text
    assert "do not replace it" in text


def test_the_prompt_does_not_pin_an_uncurated_extracted_example() -> None:
    # The camera pilot's second trust failure: an example a model copied off
    # the page was described to the next model as reviewed. Without the
    # promote-time acceptance stamp, the prompt must not mention the sentence
    # at all — the model writes fresh pedagogic examples instead.
    excerpt = record(
        source=SourceReference(type="extract", imported_from="page.jpg"),
        examples=[ExampleSentence(japanese="やくそくのとおりに")],
    )

    text = ai_prompt(excerpt)

    assert "need annotations" not in text
    assert "やくそくのとおりに" not in text


def test_the_prompt_pins_an_extracted_example_the_reviewer_accepted() -> None:
    accepted = record(
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={"example_authority": short_fingerprint("日本語を話します。")},
        ),
        examples=[ExampleSentence(japanese="日本語を話します。")],
    )

    text = ai_prompt(accepted)

    assert "Existing curated examples need annotations" in text
    assert '"日本語を話します。"' in text


def test_acceptance_covers_sentences_not_the_record() -> None:
    # One accepted, one machine-era: only the covered sentence is pinned. A
    # record-level stamp would bless text the reviewer never read.
    accepted, stray = "日本語を話します。", "やくそくのとおりに話す"
    mixed = record(
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={"example_authority": short_fingerprint(accepted)},
        ),
        examples=[
            ExampleSentence(japanese=accepted),
            ExampleSentence(japanese=stray),
        ],
    )

    text = ai_prompt(mixed)

    assert f'"{accepted}"' in text
    assert stray not in text


# --- the QC gate --------------------------------------------------------------


def test_romaji_is_always_regenerated_never_taken() -> None:
    outcome = apply_ai_result(
        record(),
        answer(
            generated("話します。", furigana="話[はな]します。", romaji="totally wrong")
        ),
    )

    assert outcome.record.examples[0].romaji == "hanashimasu."


def test_an_unaccepted_extracted_example_is_preserved_not_replaced() -> None:
    # Nothing in data distinguishes a machine-era sentence from one a person
    # curated before the authority keys existed — 39 live records have exactly
    # this shape — so only the user may decide: without --force-fields the
    # stored sentence stays, unpinned, and the generated pair is not written.
    excerpt = record(
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={"example": "話すのとおりに"},
        ),
        examples=[ExampleSentence(japanese="話すのとおりに")],
    )

    outcome = apply_ai_result(
        excerpt,
        answer(
            generated(
                "毎日日本語を話します。",
                furigana="毎日[まいにち] 日本語[にほんご]を 話[はな]します。",
            ),
        ),
    )

    assert [ex.japanese for ex in outcome.record.examples] == ["話すのとおりに"]
    assert "examples" not in outcome.changes


def test_force_fields_examples_is_the_way_to_replace_an_unaccepted_one() -> None:
    excerpt = record(
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={"example": "話すのとおりに"},
        ),
        examples=[ExampleSentence(japanese="話すのとおりに")],
    )

    outcome = apply_ai_result(
        excerpt,
        answer(
            generated(
                "毎日日本語を話します。",
                furigana="毎日[まいにち] 日本語[にほんご]を 話[はな]します。",
            ),
        ),
        force_fields=("examples",),
    )

    assert [ex.japanese for ex in outcome.record.examples] == ["毎日日本語を話します。"]
    assert "examples" in outcome.changes
    assert outcome.record.source.raw_fields["example"] == "話すのとおりに"


def test_an_accepted_extracted_example_is_annotated_in_place_not_replaced() -> None:
    accepted = record(
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={"example_authority": short_fingerprint("日本語を話します。")},
        ),
        examples=[ExampleSentence(japanese="日本語を話します。")],
    )

    outcome = apply_ai_result(
        accepted,
        answer(
            generated(
                "日本語を話します。",
                furigana="日本語[にほんご]を 話[はな]します。",
                english="I speak Japanese.",
            ),
            generated(
                "昨日友達と話した。",
                furigana="昨日[きのう] 友達[ともだち]と 話[はな]した。",
            ),
        ),
    )

    assert [ex.japanese for ex in outcome.record.examples] == ["日本語を話します。"]
    assert outcome.record.examples[0].english == "I speak Japanese."


def test_an_in_place_fill_keeps_the_clip_and_regenerates_the_romaji() -> None:
    """The three halves of the in-place fill the sibling above does not reach.

    Ported from the gating case `ai-existing-example-annotations` when M8.4
    deleted the corpus. That case observed the whole merged example; the pytest
    twin asserted only the English, so three single edits in
    `_fill_existing_example_annotations` left the suite green:

    * dropping `or incoming.furigana`, so an empty furigana hole never fills —
      the field the card's ruby and the audio both come from;
    * writing `audio=""` into the replacement, which strands a paid clip and,
      because clips are addressed by content, orphans the file too;
    * skipping the romaji regeneration, which leaves romaji describing the
      sentence as it was before the furigana arrived.

    The romaji is the sharpest of the three: it is *derived*, so a stale value
    is not obviously wrong to read — it is simply not what the card now says.
    """
    accepted = record(
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={"example_authority": short_fingerprint("日本語を話します。")},
        ),
        examples=[
            ExampleSentence(
                japanese="日本語を話します。",
                audio="audio/janki-kept.mp3",
                # Deliberately present and wrong. With this empty, "regenerated"
                # and "populated because it was empty" are indistinguishable —
                # and the merge policy every neighbouring field in the same
                # `replace()` uses is `old.X or incoming.X`, so applying it
                # uniformly to a *derived* field would strand this value with
                # nothing objecting.
                romaji="stale-before-the-furigana-arrived",
            )
        ],
    )

    outcome = apply_ai_result(
        accepted,
        answer(
            generated(
                "日本語を話します。",
                furigana="日本語[にほんご]を 話[はな]します。",
                english="I speak Japanese.",
            )
        ),
    )

    [filled] = outcome.record.examples
    assert filled.audio == "audio/janki-kept.mp3", "the paid clip survives the fill"
    assert filled.furigana == "日本語[にほんご]を 話[はな]します。"
    # `o`, not `wo`: the particle を is romanized as it is said, which is also
    # proof the value came from the romaji module rather than a naive transliteration.
    assert filled.romaji == "nihongoohanashimasu.", "regenerated from that furigana"


def test_each_stored_annotation_wins_over_the_models_own() -> None:
    """The merge fills *holes*. A stored annotation is a reviewed one, and a
    model that disagrees with it is proposing a rewrite of curated content
    under the name of an empty-field fill.

    Per annotation, not once for the record: furigana and english are separate
    `or` expressions, so a precedence pinned only for furigana leaves english
    free to invert. Register is a conditional rather than an `or`, and has its
    own rule besides — a stored value outside {polite, casual} is not a reviewed
    label, so the model's answer fills that hole rather than being refused."""
    stored = VocabularyRecord(
        id="word:画面:がめん",
        expression="画面",
        reading="がめん",
        meanings=["screen"],
        examples=[
            ExampleSentence(
                japanese="画面を見ます。",
                furigana="画面[がめん]を 見[み]ます。",
                english="I look at the screen.",
                register="polite",
            )
        ],
    )

    outcome = apply_ai_result(
        stored,
        answer(
            generated(
                "画面を見ます。",
                furigana="画面[がめん]を 見[けん]ます。",
                english="A different gloss entirely.",
                speech_level="casual",
            )
        ),
    )

    kept = outcome.record.examples[0]
    assert kept.furigana == "画面[がめん]を 見[み]ます。"
    assert kept.english == "I look at the screen."
    assert kept.register == "polite"


@pytest.mark.parametrize("answered", ["formal", "neutral", ""])
def test_a_speech_level_janki_does_not_use_is_read_as_polite(answered: str) -> None:
    """`speech_level` is a plain string with a default, not a `Literal`, and the
    schema declares no validator — so the model can answer anything and pydantic
    passes it through. Only two labels select a slot on the card, and an
    unrecognised one would select neither, leaving the sentence in no half of
    the card at all.

    Polite rather than casual, because that is what the instructions ask for
    first and what every example written before the field existed actually is;
    guessing casual would put a ます sentence under a label saying it is not."""
    outcome = apply_ai_result(
        record(),
        answer(
            generated(
                "毎日話します。",
                furigana="毎日[まいにち] 話[はな]します。",
                english="I speak every day.",
                speech_level=answered,
            )
        ),
    )

    assert outcome.record.examples[0].register == "polite"


@pytest.mark.parametrize("stored_register", ["", "formal"])
def test_an_unreviewed_register_label_is_a_hole_the_model_may_fill(
    stored_register: str,
) -> None:
    """The other direction of the same rule, so the precedence above cannot be
    satisfied by never writing register at all.

    Both an empty label and a non-empty one janki does not use: only the two
    labels the card renders are reviewed answers, and a value outside them
    selects no slot, so keeping it would leave the sentence in neither the
    polite nor the casual half. Testing the empty case alone would be satisfied
    by a truthiness check, which is a different rule that happens to agree
    there."""
    stored = VocabularyRecord(
        id="word:画面:がめん",
        expression="画面",
        reading="がめん",
        meanings=["screen"],
        examples=[
            ExampleSentence(
                japanese="画面を見ます。",
                furigana="画面[がめん]を 見[み]ます。",
                english="I look at the screen.",
                register=stored_register,
            )
        ],
    )

    outcome = apply_ai_result(
        stored,
        answer(
            generated(
                "画面を見ます。",
                furigana="画面[がめん]を 見[み]ます。",
                english="I look at the screen.",
            )
        ),
    )

    assert outcome.record.examples[0].register == "polite"


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

    result = enrich.enrich_ai([first, second], model="m", style_guide="S", instructions="I")

    assert "word:食べる:たべる" in result.changes
    assert "word:話す:はなす" not in result.changes
    assert "bio" in result.warnings[0]


def test_a_truncated_answer_is_never_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    # A half-written example is not a shorter example; it is a sentence that
    # stops mid-word, and accepting one would put it on a card.
    monkeypatch.setattr(
        enrich.claude_client, "parse_call", FakeCall(CallResult(None, "max_tokens", None))
    )

    result = enrich.enrich_ai([record()], model="m", style_guide="S", instructions="I")

    assert result.changes == {}
    assert "max_tokens" in result.warnings[0]


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

    enrich.enrich_ai(records, model="m", style_guide="S", instructions="I")

    assert "毎日話します。" not in call.calls[0]["content"]
    assert "毎日話します。" in call.calls[1]["content"]


def project(tmp_path: Path, records: list[VocabularyRecord]) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([item.to_dict() for item in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return tmp_path


def patch_all(monkeypatch: pytest.MonkeyPatch, call: FakeCall, api: FakeJpdb) -> None:
    monkeypatch.setenv("JPDB_API_KEY", "k")
    # Both providers, whichever the config resolves to. Patching only one let a
    # default change route these through the real client — which does not fail,
    # it bills and blocks.
    monkeypatch.setattr(cli.codex_client, "parse_call", call)
    monkeypatch.setattr(cli.claude_client, "parse_call", call)
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


def test_a_visited_record_that_produces_no_change_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    patch_all(monkeypatch, FakeCall(CallResult(answer(), "end_turn", None)), FakeJpdb())

    assert cli.main(["--root", str(root), "enrich", "--ai", "--yes"]) == 0

    output = capsys.readouterr().out
    assert "No changes for 1 of 1 record(s)" in output
    assert "word:話す:はなす" in output


def test_the_model_can_be_overridden_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [record()])
    call = FakeCall(ok("話します。", "話[はな]します。"))
    patch_all(monkeypatch, call, FakeJpdb({"話します。": HANASHIMASU}))

    cli.main(
        ["--root", str(root), "enrich", "--ai", "--yes", "--model", "claude-opus-5"]
    )

    assert call.calls[0]["model"] == "claude-opus-5"
    # Reasoning depth follows the overridden model, not the configured one:
    # `--model` changes the model alone, so a depth resolved from anything else
    # is resolved from something the caller did not just override.
    assert call.calls[0]["effort"] == "xhigh"


@pytest.mark.parametrize(
    "model",
    # Every one of these answers `effort` — or the `xhigh` level — with a 400.
    # `xhigh` arrived with Opus 4.7, so the 4.6 pair and Opus 4.5 reject the
    # level, and Sonnet 4.5 and Haiku 4.5 reject the parameter outright. An
    # unrecognized id is treated the same way, which is the safe direction.
    (
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "some-model-this-list-has-never-heard-of",
    ),
)
def test_a_model_that_rejects_effort_is_not_sent_it(
    model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """These answer a request carrying `effort` with a 400, and enrichment's
    call is not inside a try — so sending it aborts the run on the first
    record. Pinning the *older* models matters most: pinning one is the usual
    reason to pass --model at all."""
    root = project(tmp_path, [record()])
    call = FakeCall(ok("話します。", "話[はな]します。"))
    patch_all(monkeypatch, call, FakeJpdb({"話します。": HANASHIMASU}))

    cli.main(
        [
            "--root", str(root), "enrich", "--ai", "--yes",
            "--model", model,
        ]
    )

    assert "effort" not in call.calls[0]


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


def test_the_preserve_warning_actually_fires() -> None:
    # The decision and its report live in different functions; this pins the
    # seam — a preserve that never sets the flag makes the user-decision nudge
    # silently disappear, which is the failure it exists to prevent.
    from japanese_anki.claude_client import CallResult as CR
    from japanese_anki.enrich import AiResult, absorb_ai_call

    excerpt = record(
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={"example": "話すのとおりに"},
        ),
        examples=[ExampleSentence(japanese="話すのとおりに")],
        usage_notes="a note",
    )
    result = AiResult(records=[excerpt], looked_up=1)

    absorb_ai_call(
        result,
        excerpt,
        CR(
            answer(
                generated(
                    "毎日日本語を話します。",
                    furigana="毎日[まいにち] 日本語[にほんご]を 話[はな]します。",
                )
            ),
            "end_turn",
            None,
        ),
        model="offline-model",
        positions={excerpt.id: 0},
        recent=[],
    )

    assert [ex.japanese for ex in result.records[0].examples] == ["話すのとおりに"]
    assert any(
        "stored examples were preserved" in warning for warning in result.warnings
    )


def test_the_codex_path_still_sends_its_own_reasoning_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex takes `reasoning_effort`, not `output_config.effort`, and remains a
    supported provider. Its only end-to-end pin went out with the test that was
    repointed at the Anthropic default."""
    call = FakeCall(ok("話します。", "話[はな]します。"))
    root = project(tmp_path, [record()])
    (root / "janki.toml").write_text(
        (root / "janki.toml").read_text(encoding="utf-8")
        + '\n[ai]\nenrich_provider = "codex"\nenrich_model = "gpt-5.6-sol"\n'
        'enrich_reasoning_effort = "ultra"\n',
        encoding="utf-8",
    )
    patch_all(monkeypatch, call, FakeJpdb({"話します。": HANASHIMASU}))

    cli.main(["--root", str(root), "enrich", "--ai", "--yes"])

    assert call.calls[0]["reasoning_effort"] == "ultra"
    assert "effort" not in call.calls[0]


def test_the_ai_pass_enriches_a_record_with_no_jpdb_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """M7.6V's headline: `--ai` asks jpdb nothing, so it must not demand a key.

    Driven over a record that really is enriched, and with `cli.jpdb.JpdbClient`
    left unpatched so any attempt to build one raises rather than being handed a
    fake. The first shape of the retirement narrowed `_JPDB_STAGES` but left the
    client built above the `--ai` dispatch, so the pass still died without a
    key; the only test covering it ran against an empty collection, where the
    stage no-ops before reaching the construction."""
    monkeypatch.delenv("JPDB_API_KEY", raising=False)
    root = project(tmp_path, [record()])
    call = FakeCall(
        CallResult(
            answer(
                generated(
                    "毎日話します。",
                    furigana="毎日[まいにち] 話[はな]します。",
                    english="I speak every day.",
                ),
                usage_notes="A common verb.",
            ),
            "end_turn",
            None,
        )
    )
    monkeypatch.setattr(cli.codex_client, "parse_call", call)
    monkeypatch.setattr(cli.claude_client, "parse_call", call)

    code = cli.main(["--root", str(root), "enrich", "--ai", "--yes"])

    assert code == 0, capsys.readouterr().err
    assert stored(root)["word:話す:はなす"]["examples"], "the pass wrote nothing"


def test_the_preserve_branch_still_fills_the_records_other_empty_fields() -> None:
    """A record that already has examples takes the merge branch, and that
    branch must still write the *other* empty AI fields — usage_notes above
    all, since `ai_targets` selects on "no example or no note" and
    有-examples/無-note is the largest real target class. M8.3's rewrite of
    `apply_ai_result` kept this behaviour but lost every test that pinned it:
    narrowing the writable list to force-fields-only left the whole suite
    green while silently never writing a note again."""
    stored = record(
        examples=[ExampleSentence(japanese="人と話す。", english="x", register="casual")]
    )

    outcome = apply_ai_result(
        stored,
        answer(
            generated("毎日話します。", furigana="毎日[まいにち] 話[はな]します。"),
            usage_notes="Casual speech often drops the particle.",
        ),
    )

    assert "usage_notes" in outcome.changes
    assert outcome.record.usage_notes == "Casual speech often drops the particle."
    # And the stored example was preserved, which is what routed us here.
    assert [ex.japanese for ex in outcome.record.examples] == ["人と話す。"]


def test_the_preserve_warning_fires_even_for_accepted_stored_examples() -> None:
    """M8.3 widened the preserve warning: it used to stay silent when every
    stored example carried reviewer acceptance, and now it reports the discard
    whenever generated sentences had nowhere to land. Deliberate — the warning
    is fill-discipline reporting ("your new sentences went nowhere; use
    --force-fields to replace"), which is true and useful regardless of how
    blessed the stored examples are."""
    from japanese_anki.identifiers import short_fingerprint
    from japanese_anki.models import EXAMPLE_AUTHORITY_KEY

    sentence = "人と話す。"
    accepted = record(
        source=SourceReference(
            type="extract",
            imported_from="page.jpg",
            raw_fields={EXAMPLE_AUTHORITY_KEY: short_fingerprint(sentence)},
        ),
        examples=[ExampleSentence(japanese=sentence, english="x", register="casual")],
    )
    from japanese_anki.models import example_accepted

    # The premise, asserted rather than assumed: this stored example really is
    # reviewer-accepted, so the warning firing below proves the widening.
    assert example_accepted(accepted, accepted.examples[0])
    result = enrich.AiResult(records=[accepted], looked_up=1)

    enrich.absorb_ai_call(
        result,
        accepted,
        CallResult(
            answer(generated("毎日話します。", furigana="毎日[まいにち] 話[はな]します。")),
            "end_turn",
            None,
        ),
        model="offline-model",
        positions={accepted.id: 0},
        recent=[],
    )

    assert any("stored examples were preserved" in w for w in result.warnings)


def test_a_sentence_whose_punctuation_supplied_a_space_is_still_unsegmented() -> None:
    """Counting spaces is not the test; equality with the machine output is.

    `konban, hahanidenwao kakerutsumoridesu.` holds two spaces — one from 、
    and one from the furigana's ruby notation — and four merged words. A
    space-counting heuristic read it as already segmented and the romaji pass
    skipped the record it exists to fix.
    """
    from japanese_anki.enrich import romaji_targets
    from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord

    record = VocabularyRecord(
        id="word:かける:かける",
        expression="かける",
        reading="かける",
        meanings=["to make (a call)"],
        source=SourceReference(type="extract", imported_from="lesson.pdf"),
        examples=[
            ExampleSentence(
                japanese="今晩、母に電話をかけるつもりです。",
                furigana="今晩[こんばん]、 母[はは]に 電話[でんわ]を かけるつもりです。",
                romaji="konban, hahanidenwao kakerutsumoridesu.",
            )
        ],
    )

    assert romaji_targets([record]) == [record]

    segmented = record.examples[0].__class__(
        japanese=record.examples[0].japanese,
        furigana=record.examples[0].furigana,
        romaji="konban, haha ni denwa o kakeru tsumori desu.",
    )
    from dataclasses import replace

    assert romaji_targets([replace(record, examples=[segmented])]) == []


def test_a_short_romaji_response_is_refused_whole_rather_than_zipped() -> None:
    """Lines are matched to sentences by position, so a missing one shifts
    every line after it onto the wrong sentence.

    That produces a perfectly well-formed romaji under Japanese it does not
    transliterate — and `settle_example_romaji` would catch each individual
    mismatch, but only after the damage of deciding which line belongs where.
    Refusing the response whole is the honest answer to "I cannot tell which
    sentence you left out".
    """
    from types import SimpleNamespace

    from japanese_anki.enrich import apply_romaji_result
    from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord

    record = VocabularyRecord(
        id="word:はし:はし",
        expression="はし",
        reading="はし",
        meanings=["bridge"],
        source=SourceReference(type="extract", imported_from="lesson.pdf"),
        examples=[
            ExampleSentence(japanese="はしをわたる。", furigana="", romaji="hashiowataru."),
            ExampleSentence(japanese="はしはながい。", furigana="", romaji="hashihanagai."),
        ],
    )

    updated, warnings = apply_romaji_result(
        record, SimpleNamespace(romaji=["hashi o wataru."])
    )

    assert updated == record, "nothing was written"
    assert len(warnings) == 1
    assert "asked for 2 romaji line(s) and got 1" in warnings[0]


def test_romaji_that_no_longer_matches_its_reading_is_targeted_again() -> None:
    """The third condition, and the one that is easy to leave out.

    A romaji can be segmented — so not equal to the mechanical output — and
    still be stale, because the *reading* changed underneath it. The furigana
    spacing repair does exactly that: it gives back a comma, and sometimes
    whole clauses, that a ruby base had swallowed. Eleven sentences were left
    describing Japanese the record no longer held, and neither the "empty" nor
    the "unsegmented" test could see them.
    """
    from dataclasses import replace

    from japanese_anki.enrich import romaji_targets
    from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord

    # Segmented, and correct for this reading.
    good = ExampleSentence(
        japanese="先週、家族と山に登りました。",
        furigana="先週[せんしゅう]、 家族[かぞく]と 山[やま]に 登[のぼ]りました。",
        romaji="senshuu, kazoku to yama ni noborimashita.",
    )
    record = VocabularyRecord(
        id="word:先週:せんしゅう",
        expression="先週",
        reading="せんしゅう",
        meanings=["last week"],
        source=SourceReference(type="extract", imported_from="lesson.pdf"),
        examples=[good],
    )
    assert romaji_targets([record]) == [], "nothing to do while they agree"

    # The same romaji against a reading that says something else.
    moved = replace(
        good, furigana="来年[らいねん]、 家族[かぞく]と 山[やま]に 登[のぼ]りました。"
    )
    stale = replace(record, examples=[moved])

    assert romaji_targets([stale]) == [stale], "stale, and neither empty nor unsegmented"


def _romaji_project(tmp_path: Path) -> Path:
    """A project holding one record whose romaji is the machine output."""
    import json

    from conftest import seed_prompts

    (tmp_path / "janki.toml").write_text(
        '[paths]\nnormalized_file = "vocabulary.json"\nstaging_dir = "staging"\n'
        'scan_inbox = "inbox"\npatterns_file = "patterns.json"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            [
                {
                    "id": "word:話す:はなす",
                    "expression": "話す",
                    "reading": "はなす",
                    "meanings": ["to speak"],
                    "source": {"type": "shirabe", "imported_from": "x.csv"},
                    "examples": [
                        {
                            "japanese": "毎日話します。",
                            "furigana": "毎日[まいにち] 話[はな]します。",
                            "romaji": "mainichihanashimasu.",
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
    return tmp_path


def test_the_romaji_pass_refuses_an_unknown_id_rather_than_reporting_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A typo'd id filters the target list to nothing, and "nothing to do" is
    indistinguishable from "already done".

    Every other pass raises on an unknown id. This one printed "Every example's
    romaji is already segmented" and exited 0, which is the answer a user gets
    for a job that never ran.
    """
    from japanese_anki import cli

    root = _romaji_project(tmp_path)

    code = cli.main(
        ["--root", str(root), "enrich", "--romaji", "--yes", "word:nope:nope"]
    )

    assert code == 1
    assert "No record with id" in capsys.readouterr().err


def test_the_romaji_pass_refuses_flags_it_would_ignore(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """"A flag the running pass never reads is a typo, not a no-op" — the rule
    `cli.py` states four lines above the gate that did not list `--romaji`.

    `--staging` named a file this pass never opens, and the run went ahead:
    one billed call per record, and the collection rewritten instead of the
    file the user named.
    """
    from japanese_anki import cli

    root = _romaji_project(tmp_path)
    staged = root / "staging" / "held.yaml"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text("records: []\n", encoding="utf-8")

    assert cli.main(
        ["--root", str(root), "enrich", "--romaji", "--staging", str(staged), "--yes"]
    ) == 1
    assert "Run them separately" in capsys.readouterr().err

    assert cli.main(
        ["--root", str(root), "enrich", "--romaji", "--force-fields", "examples", "--yes"]
    ) == 1
    assert "nothing for --force-fields" in capsys.readouterr().err


def test_the_romaji_pass_records_what_it_spent_in_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other model pass writes a ledger entry; this one wrote none.

    A billed rewrite of every example in the collection that leaves no
    operational trace is one `status --rebuild` cannot reconstruct and one
    nobody can audit after the fact.
    """
    from types import SimpleNamespace

    from japanese_anki import claude_client, cli, enrich, ledger

    root = _romaji_project(tmp_path)

    def call(_model, _blocks, _content, _schema, _client=None, **_options):
        return claude_client.CallResult(
            SimpleNamespace(romaji=["mainichi hanashimasu."]), "end_turn", None
        )

    for module in (claude_client, cli, enrich):
        monkeypatch.setattr(
            getattr(module, "claude_client", module), "parse_call", call, raising=False
        )
    monkeypatch.setattr(claude_client, "parse_call", call)

    cli.main(["--root", str(root), "enrich", "--romaji", "--yes"])

    book = ledger.load(root / "data" / "ledger.json")
    entries = book.enrichment_for("word:話す:はなす") if hasattr(book, "enrichment_for") else None
    raw = (root / "data" / "ledger.json").read_text(encoding="utf-8")
    assert "romaji" in raw, "the pass names itself in the ledger"
    del entries
