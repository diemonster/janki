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
from japanese_anki.kanji import KanjiInfo, KanjiStore, Reading
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.staging import read_staging


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
    assert "Use the exact spelling shown in Expression" in enrich.AI_INSTRUCTIONS
    assert "do not\nreplace a kana-only expression with kanji" in enrich.AI_INSTRUCTIONS


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


def test_an_example_without_the_word_is_rejected() -> None:
    outcome = apply_ai_result(record(), answer(generated("毎日言います。")))

    assert outcome.rejected == ["毎日言います。"]
    assert outcome.changes == {}


def test_a_conjugated_example_is_kept() -> None:
    outcome = apply_ai_result(
        record(),
        answer(generated("昨日話した。", furigana="昨日[きのう] 話[はな]した。")),
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


def test_furigana_that_spells_its_own_sentence_is_not_flagged() -> None:
    """The negative direction, and it is what the retired dictionary check kept
    getting wrong: with no parse it flagged this sentence, and with one it
    flagged whichever sentences jpdb segmented differently than the writer."""
    outcome = apply_ai_result(
        record(),
        answer(generated("話します。", furigana="話[はな]します。")),
    )

    assert outcome.unverified == []
    assert UNVERIFIED_KEY not in outcome.record.source.raw_fields


def test_a_flagged_example_is_kept_and_its_sentence_fingerprinted() -> None:
    # Kept, not dropped: the sentence may be right where the furigana is not,
    # and a human deciding that beats janki throwing away good Japanese. The
    # flag is keyed to the sentence, so it follows that example and no other.
    outcome = apply_ai_result(
        record(),
        answer(generated("話します。", furigana="話[はな]しました。")),
    )

    assert [ex.japanese for ex in outcome.record.examples] == ["話します。"]
    assert outcome.unverified == ["話します。"]
    fingerprints = outcome.record.source.raw_fields[UNVERIFIED_KEY].split(",")
    assert fingerprints == [short_fingerprint("話します。")]


def test_an_impossible_character_group_is_kept_flagged_and_reported() -> None:
    source = VocabularyRecord(
        id="word:画面:がめん",
        expression="画面",
        reading="がめん",
        meanings=["screen"],
        examples=[ExampleSentence(japanese="二本指で画面を広げます。")],
        usage_notes="A common noun.",
    )
    store = KanjiStore(
        entries={
            "指": KanjiInfo(
                character="指",
                readings=(Reading(kind="kun", reading="ゆび"),),
            )
        }
    )

    outcome = apply_ai_result(
        source,
        answer(
            generated(
                "二本指で画面を広げます。",
                furigana="二本 指[にほんゆび]で 画面[がめん]を 広[ひろ]げます。",
                english="Use two fingers to enlarge the screen.",
            )
        ),
        kanji_store=store,
    )

    kept = outcome.record.examples[0]
    assert kept.furigana == "二本 指[にほんゆび]で 画面[がめん]を 広[ひろ]げます。"
    assert kept.romaji == ""
    assert kept.english == "Use two fingers to enlarge the screen."
    assert outcome.unverified == ["二本指で画面を広げます。"]
    assert outcome.impossible_furigana == [
        (
            "二本指で画面を広げます。",
            "二本 指[にほんゆび]で 画面[がめん]を 広[ひろ]げます。",
            (("指", "にほんゆび"),),
        )
    ]


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


def test_an_impossible_group_is_not_reported_when_curated_furigana_wins() -> None:
    source = VocabularyRecord(
        id="word:画面:がめん",
        expression="画面",
        reading="がめん",
        meanings=["screen"],
        examples=[
            ExampleSentence(
                japanese="二本指で画面を広げます。",
                furigana="二本指[にほんゆび]で 画面[がめん]を 広[ひろ]げます。",
                romaji="nihon'yubidegamen'ohirogemasu.",
                english="Use two fingers to enlarge the screen.",
                register="polite",
            )
        ],
    )
    store = KanjiStore(
        entries={
            "指": KanjiInfo(
                character="指",
                readings=(Reading(kind="kun", reading="ゆび"),),
            )
        }
    )

    outcome = apply_ai_result(
        source,
        answer(
            generated(
                "二本指で画面を広げます。",
                furigana="二本 指[にほんゆび]で 画面[がめん]を 広[ひろ]げます。",
                english="Use two fingers to enlarge the screen.",
            ),
            usage_notes="Pinch gestures use two fingers.",
        ),
        kanji_store=store,
    )

    assert outcome.record.examples == source.examples
    assert outcome.impossible_furigana == []
    assert outcome.unverified == []
    assert outcome.record.usage_notes == "Pinch gestures use two fingers."


def test_a_stored_impossible_group_is_flagged_when_ai_fills_english() -> None:
    invalid = "二本 指[にほんゆび]で 画面[がめん]を 広[ひろ]げます。"
    source = VocabularyRecord(
        id="word:画面:がめん",
        expression="画面",
        reading="がめん",
        meanings=["screen"],
        examples=[
            ExampleSentence(
                japanese="二本指で画面を広げます。",
                furigana=invalid,
            )
        ],
        usage_notes="A common noun.",
    )
    store = KanjiStore(
        entries={
            "指": KanjiInfo(
                character="指",
                readings=(Reading(kind="kun", reading="ゆび"),),
            )
        }
    )

    outcome = apply_ai_result(
        source,
        answer(
            generated(
                "二本指で画面を広げます。",
                furigana=invalid,
                english="Use two fingers to enlarge the screen.",
            )
        ),
        kanji_store=store,
    )

    assert outcome.unverified == ["二本指で画面を広げます。"]
    assert outcome.impossible_furigana
    assert UNVERIFIED_KEY in outcome.record.source.raw_fields


def test_flags_accumulate_for_sentences_still_on_the_record() -> None:
    # A flag for a sentence the record still carries survives a later write;
    # writes append, never replace.
    first = apply_ai_result(
        record(), answer(generated("話します。", furigana="話[はな]しました。"))
    )

    second = apply_ai_result(
        first.record,
        answer(
            generated("話します。", furigana="話[はな]しました。"),
            generated("昨日話した。", furigana="昨日[きのう] 話[はな]しました。"),
        ),
        force_fields=("examples",),
    )

    flags = second.record.source.raw_fields[UNVERIFIED_KEY]
    assert short_fingerprint("話します。") in flags
    assert short_fingerprint("昨日話した。") in flags


def test_an_orphaned_flag_fingerprint_is_garbage_collected() -> None:
    # A fingerprint matching no current example refers to nothing — and for a
    # hold, keeping it would leave a sentence permanently refusable with no
    # way to un-hold it. Pruned at the write, when the orphan is created.
    already = record(
        source=SourceReference(raw_fields={UNVERIFIED_KEY: "deadbeef"}),
        verb_group="godan",
    )

    outcome = apply_ai_result(
        already, answer(generated("話します。", furigana="話[はな]しました。"))
    )

    assert "deadbeef" not in outcome.record.source.raw_fields[UNVERIFIED_KEY]
    assert short_fingerprint("話します。") in (
        outcome.record.source.raw_fields[UNVERIFIED_KEY]
    )


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


def test_all_rejected_examples_get_one_constrained_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waiting = record(
        id="word:まつ:まつ",
        expression="まつ",
        reading="まつ",
        meanings=["to wait"],
    )
    call = FakeCall(
        CallResult(
            answer(
                generated("友達を待ちます。"),
                usage_notes="The particle を can mark the person awaited.",
            ),
            "end_turn",
            None,
        ),
        CallResult(
            answer(
                generated("友達をまちます。"),
                generated("ここでまつの？"),
            ),
            "end_turn",
            None,
        ),
    )
    monkeypatch.setattr(enrich.claude_client, "parse_call", call)

    result = enrich.enrich_ai([waiting], model="m", style_guide="S")

    assert len(call.calls) == 2
    assert "Permitted written target forms:" in call.calls[1]["content"]
    assert "まちます" in call.calls[1]["content"]
    assert [example.japanese for example in result.records[0].examples] == [
        "友達をまちます。",
        "ここでまつの？",
    ]
    assert result.records[0].usage_notes == (
        "The particle を can mark the person awaited."
    )
    assert result.rejected == {}
    assert result.no_changes == []
    assert set(result.changes[waiting.id]) == {"examples", "usage_notes"}


def test_one_accepted_example_does_not_trigger_the_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = FakeCall(
        CallResult(
            answer(generated("話します。"), generated("友達に言います。")),
            "end_turn",
            None,
        )
    )
    monkeypatch.setattr(enrich.claude_client, "parse_call", call)

    result = enrich.enrich_ai([record()], model="m", style_guide="S")

    assert len(call.calls) == 1
    assert [example.japanese for example in result.records[0].examples] == [
        "話します。"
    ]
    assert result.rejected == {"word:話す:はなす": ["友達に言います。"]}


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


def test_a_flagged_example_is_reported_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    patch_all(
        monkeypatch,
        FakeCall(
            ok("話します。", "話[はな]しました。")
        ),
        FakeJpdb({"話します。": HANASHIMASU}),
    )

    cli.main(["--root", str(root), "enrich", "--ai", "--yes"])

    captured = capsys.readouterr()
    assert "a local check disagreed with" in captured.err
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
    # Each furigana field spells a different sentence than its example, which
    # is what flags them now that no dictionary is asked about a sentence.
    patch_all(
        monkeypatch,
        FakeCall(
            *[
                CallResult(
                    answer(generated(f"話す{index}。", furigana=f"話[はな]した{index}。")),
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
        "M5.3 reads this key to decide whether to speak a doubted sentence"
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
        "carry no reviewer acceptance" in warning for warning in result.warnings
    )


def test_both_furigana_warnings_say_what_was_kept_and_that_audio_is_blocked() -> None:
    """The flag is structured data; this is the sentence a person actually
    reads. Both messages have to say the same three things — which groups or
    strings disagree, that the example was kept rather than dropped, and that
    audio stays blocked — because a warning that reports only "unverified"
    sends someone looking for a card fault that is really a field fault.

    Driven through `absorb_ai_call`, where the text is built: the outcome
    assertions elsewhere in this file pin the tuples and never the string."""
    from japanese_anki.enrich import AiResult, absorb_ai_call

    store = KanjiStore(
        entries={
            "指": KanjiInfo(
                character="指", readings=(Reading(kind="kun", reading="ゆび"),)
            )
        }
    )
    impossible = record(id="word:画面:がめん", expression="画面", reading="がめん")
    result = AiResult(records=[impossible], looked_up=1)

    absorb_ai_call(
        result,
        impossible,
        CallResult(
            answer(
                generated(
                    "二本指で画面を広げます。",
                    furigana="二本 指[にほんゆび]で 画面[がめん]を 広[ひろ]げます。",
                    english="Use two fingers to enlarge the screen.",
                )
            ),
            "end_turn",
            None,
        ),
        model="offline-model",
        positions={impossible.id: 0},
        recent=[],
        kanji_store=store,
    )

    [warning] = [text for text in result.warnings if "need review" in text]
    assert "指" in warning and "にほんゆび" in warning
    assert "kept and marked unverified" in warning
    assert "sentence audio remains blocked" in warning

    rewriting = record(id="word:話:はなし", expression="話", reading="はなし")
    second = AiResult(records=[rewriting], looked_up=1)

    absorb_ai_call(
        second,
        rewriting,
        CallResult(
            answer(generated("話を話します。", furigana="話[はなし]が 話[はな]します。")),
            "end_turn",
            None,
        ),
        model="offline-model",
        positions={rewriting.id: 0},
        recent=[],
    )

    [named] = [
        text for text in second.warnings if "does not spell its own sentence" in text
    ]
    # The diagnosis, not just the verdict: this warning exists to say which two
    # strings disagree, which is the difference between "check this card" and
    # "check this field". Without it the message is the unverified count again.
    assert "話が話します。" in named and "話を話します。" in named
    assert "kept and marked unverified" in named
    assert "sentence audio remains blocked" in named


def test_a_furigana_field_that_rewrites_the_sentence_is_named() -> None:
    # The failure this catches is a model rewriting the sentence inside the
    # field that drives audio: the groups all read correctly, and the particle
    # is wrong. It is decidable without a dictionary, so it must be reported
    # even when no parse arrived — the case that used to short-circuit ahead of
    # it, leaving "unverified" as the only thing anyone was told.
    outcome = apply_ai_result(
        record(),
        answer(generated("話を話します。", furigana="話[はなし]が 話[はな]します。")),
    )

    assert outcome.rewritten_furigana == [
        (
            "話を話します。",
            "話[はなし]が 話[はな]します。",
            "the furigana spells 話が話します。, but the sentence is 話を話します。",
        )
    ]
    assert outcome.unverified == ["話を話します。"]


def test_a_rewritten_sentence_is_named_even_when_the_reading_is_impossible() -> None:
    # `impossible` short-circuits the verdict too, so a card carrying both
    # failures used to be told about only one of them.
    store = KanjiStore(
        entries={
            "話": KanjiInfo(
                character="話",
                readings=(Reading(kind="kun", reading="はなし"),),
            )
        }
    )

    outcome = apply_ai_result(
        record(),
        answer(generated("話を話します。", furigana="話[はなし]が 話[ざぶとん]します。")),
        kanji_store=store,
    )

    assert [item[2] for item in outcome.impossible_furigana] == [(("話", "ざぶとん"),)]
    assert [item[2] for item in outcome.rewritten_furigana] == [
        "the furigana spells 話が話します。, but the sentence is 話を話します。"
    ]


def test_a_furigana_field_that_matches_its_sentence_is_not_named() -> None:
    outcome = apply_ai_result(
        record(), answer(generated("話します。", furigana="話[はな]します。"))
    )

    assert outcome.rewritten_furigana == []


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


def test_an_impossible_reading_flags_a_record_that_had_no_examples() -> None:
    """The impossible-character route to the flag, on the branch where it is
    the only thing writing it.

    A record that already has examples takes the merge branch, which re-derives
    the flag list from `outcome.impossible_furigana` — so removing the check
    from the verdict there changes nothing and the two routes mask each other.
    A record with no stored example has no merge to re-derive from, and the
    verdict is the whole of it."""
    store = KanjiStore(
        entries={
            "指": KanjiInfo(
                character="指", readings=(Reading(kind="kun", reading="ゆび"),)
            )
        }
    )
    empty = record(id="word:画面:がめん", expression="画面", reading="がめん")

    outcome = apply_ai_result(
        empty,
        answer(
            generated(
                "二本指で画面を広げます。",
                furigana="二本 指[にほんゆび]で 画面[がめん]を 広[ひろ]げます。",
                english="Use two fingers to enlarge the screen.",
            )
        ),
        kanji_store=store,
    )

    assert outcome.record.examples, "the example is kept, not dropped"
    assert outcome.unverified == ["二本指で画面を広げます。"]


def test_accepting_clears_a_flag_a_person_has_vouched_for() -> None:
    """The only route left from a flagged example to a voiced one. M7.6V
    retired the dictionary re-check, and nothing else takes a flag off: a
    person can correct the furigana by hand and the flag stays, because it
    describes the sentence as it was when it was written.

    Without this, retiring the oracle would leave every already-flagged
    sentence permanently unvoiced with no way to un-hold it."""
    flagged = apply_ai_result(
        record(), answer(generated("話します。", furigana="話[はな]しました。"))
    ).record
    assert UNVERIFIED_KEY in flagged.source.raw_fields

    result = enrich.accept_furigana([flagged], [flagged.id])

    assert result.cleared == {flagged.id: ["話します。"]}
    assert UNVERIFIED_KEY not in result.records[0].source.raw_fields


def test_accepting_needs_the_ids_a_person_is_vouching_for() -> None:
    """Accepting everything unread is not a judgment — it is the one shape of
    this command that would clear a flag nobody looked at."""
    with pytest.raises(enrich.EnrichError) as excinfo:
        enrich.accept_furigana([record()], [])

    assert "not a judgment" in str(excinfo.value)


def test_accepting_an_id_no_record_has_is_a_typo_not_an_empty_result() -> None:
    with pytest.raises(enrich.EnrichError) as excinfo:
        enrich.accept_furigana([record()], ["word:無い:ない"])

    assert "word:無い:ない" in str(excinfo.value)


def test_a_stored_rewriting_furigana_is_flagged_when_ai_fills_english() -> None:
    """The merge branch re-derives the flag list, and it has to carry *both*
    routes. A stored example whose own furigana spells a different sentence is
    the mirror of the stored-impossible case: the model returns the sentence
    unchanged and fills an empty annotation, and the example that lands is
    still the one whose furigana is wrong.

    Pinned separately because the verdict above cannot stand in for it — the
    verdict runs on the *generated* example, and what reaches the record here
    is the stored one that the fill merged into."""
    stored = VocabularyRecord(
        id="word:本:ほん",
        expression="本",
        reading="ほん",
        meanings=["book"],
        examples=[ExampleSentence(japanese="本を読む。", furigana="本[ほん]が 読[よ]む。")],
    )

    outcome = apply_ai_result(
        stored,
        answer(
            generated(
                "本を読む。",
                furigana="本[ほん]が 読[よ]む。",
                english="I read a book.",
            )
        ),
    )

    assert outcome.record.examples[0].english == "I read a book."
    assert short_fingerprint("本を読む。") in (
        outcome.record.source.raw_fields[UNVERIFIED_KEY]
    )


def _ledger_entries(book: dict[str, Any]) -> list[dict[str, Any]]:
    """Every enrichment entry, whatever shape the ledger stores them in."""
    found: list[dict[str, Any]] = []
    stack: list[Any] = [book]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if {"kind", "fields"} <= set(node):
                found.append(node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


def test_accepting_through_the_cli_writes_the_records_and_the_human_ledger_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Driven through `cli.main`, because the unit tests above cannot see the
    command. The first shape of this handler cleared the flags on disk and then
    raised from `record_enriched` — `fields=[]` and no model are both refused —
    so it exited 1 having destroyed its own audit trail, and re-running found
    nothing left to accept. Every assertion here failed on that shape.

    Needs no jpdb key: accepting asks no dictionary anything."""
    monkeypatch.delenv("JPDB_API_KEY", raising=False)
    flagged = apply_ai_result(
        record(), answer(generated("話します。", furigana="話[はな]しました。"))
    ).record
    root = project(tmp_path, [flagged])

    code = cli.main(["--root", str(root), "enrich", "--accept", flagged.id])

    assert code == 0
    assert UNVERIFIED_KEY not in stored(root)[flagged.id]["source"]["raw_fields"]

    # The entry's exact shape, not just that the word "human" appears somewhere:
    # `record_enriched` dedups on (kind, model, fields), so naming the wrong
    # field here would let a later real --jpdb pass that fills `furigana`
    # collapse into this entry and inherit its date. That assertion existed in
    # the deleted test_recheck_furigana.py and has to survive the move.
    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    entries = [
        entry
        for record in book.get("records", {}).values()
        for entry in record.get("enriched", [])
    ] or _ledger_entries(book)
    assert entries, f"no enrichment entry was written: {book}"
    [entry] = entries
    assert entry["kind"] == "human", "who vouched is the whole of what this records"
    assert entry["model"] == "human"
    assert entry["fields"] == [UNVERIFIED_KEY]

    # Each accepted sentence is named, ahead of the count that summarises them.
    # That the naming also precedes `save_records_json` is real and deliberate
    # — a durable change should not be the first thing a reader learns about —
    # but stdout ordering cannot observe it, so this does not claim to.
    out = capsys.readouterr().out
    assert out.index("話します。") < out.index("Accepted 1 sentence")


@pytest.mark.parametrize(
    "extra",
    [["--force-fields", "examples"], ["--staging", "somewhere.yaml"]],
    ids=["force-fields", "staging"],
)
def test_accepting_refuses_the_flags_that_write_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    """`--accept` writes no field and reads no staging file, so neither flag
    has anything to act on. Without this the pass ran and ignored them, and
    `--force-fields examples` answered with the *jpdb* field list.

    Both rows, because they are separate conjuncts: the deleted
    test_recheck_furigana.py parametrized them and the staging row did not
    survive the move, so dropping it from the guard passed the whole suite."""
    root = project(tmp_path, [record()])

    code = cli.main(
        ["--root", str(root), "enrich", "--accept", *extra, "word:話す:はなす"]
    )

    assert code == 1
    assert "nothing to act on" in capsys.readouterr().err


@pytest.mark.parametrize(
    "furigana",
    ["毎日　話[はな]します。", "毎日話[はな]します。"],
    ids=["full-width-space", "no-separator"],
)
def test_a_full_width_separator_reaches_the_card_unflagged(furigana: str) -> None:
    """A known gap, pinned where it regressed. Before M7.6V every generated
    example was flagged when no parse arrived, so this shipped held back; now
    the two surviving checks are both silent on it and it reaches a card.

    Anki's furigana filter separates on the ASCII space alone, so the ruby in
    `毎日　話[はな]します。` covers 毎日 as well — the reading, the romaji and the
    sentence audio all lose it. Both rows, because they are one fault with two
    notations: a separator Anki cannot read, and no separator at all before a
    Han-initial run, which `spilled_furigana_groups` documents as its own gap.
    A fix for one that leaves the other silent has to fail something.

    Recorded as `furigana-full-width-separator` in quality/findings.yaml. This
    asserts the wrong behaviour on purpose and is expected to fail when the
    separator rule lands."""
    outcome = apply_ai_result(
        record(), answer(generated("毎日話します。", furigana=furigana))
    )

    assert outcome.unverified == []
    assert UNVERIFIED_KEY not in outcome.record.source.raw_fields
    assert outcome.record.examples[0].romaji == "hanashimasu."


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
