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
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from conftest import seed_prompts
from japanese_anki import ai_schema as ai_schema_module
from japanese_anki import claude_client, cli, codex_client, enrich, prompts, staging
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
from japanese_anki.ledger import Ledger
from japanese_anki.models import (
    PROVISIONAL_FIELDS_KEY,
    ExampleSentence,
    SourceReference,
    VocabularyRecord,
    mark_provisional,
    provisional_fields,
)
from japanese_anki.staging import read_staging

REPO_ROOT = Path(__file__).resolve().parents[1]


def generated(
    japanese: str,
    furigana: str | None = None,
    english: str | None = None,
    romaji: str | None = None,
    speech_level: str = "polite",
    **extra: Any,
) -> Any:
    """One generated example with an explicit card-slot label.

    Names checked against the schema first so a misspelled field in a fixture
    fails at the point where the fixture is built."""
    schema = ai_schema()
    item = schema.model_fields["examples"].annotation.__args__[0]
    unknown = sorted(set(extra) - set(item.model_fields))
    if unknown:
        raise TypeError(f"not a field of the generated-example schema: {unknown}")
    return item(
        japanese=japanese,
        furigana=furigana or japanese,
        english=english or "Fixture translation.",
        romaji=romaji or "fixture romaji",
        speech_level=speech_level,
        **extra,
    )


def test_request_provenance_tracks_provider_wire_prompt_and_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = record(examples=[], usage_notes="")
    marker = {"value": "wire-v1"}

    class FakeAnthropic:
        @staticmethod
        def transform_schema(schema: Any) -> dict[str, Any]:
            return {"wire": marker["value"], "name": schema.__name__}

    monkeypatch.setattr(claude_client, "load_anthropic", lambda: FakeAnthropic)
    anthropic_v1 = enrich.ai_request_fingerprint(
        item,
        provider="anthropic",
        style_guide="style\n",
        instructions="task\n",
    )
    marker["value"] = "wire-v2"
    anthropic_v2 = enrich.ai_request_fingerprint(
        item,
        provider="anthropic",
        style_guide="style\n",
        instructions="task\n",
    )

    assert anthropic_v1 != anthropic_v2

    codex_v1 = enrich.ai_request_fingerprint(
        item,
        provider="codex",
        style_guide="style\n",
        instructions="task\n",
    )
    original_prompt = codex_client._prompt
    monkeypatch.setattr(
        codex_client,
        "_prompt",
        lambda blocks, user: original_prompt(blocks, user) + "\ntransport-v2",
    )
    codex_v2 = enrich.ai_request_fingerprint(
        item,
        provider="codex",
        style_guide="style\n",
        instructions="task\n",
    )

    assert codex_v1 != codex_v2
    assert anthropic_v2 != codex_v1


def answer(
    *examples: Any,
    meanings: tuple[str, ...] | list[str] = ("to speak",),
    usage_notes: str = "",
) -> Any:
    return ai_schema()(
        meanings=list(meanings), examples=list(examples), usage_notes=usage_notes
    )


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


def test_a_record_with_no_meaning_or_example_is_a_target() -> None:
    no_meaning = record(
        id="z",
        meanings=[],
        usage_notes="something",
        examples=[ExampleSentence(japanese="話す。")],
    )
    no_example = record(id="a", usage_notes="something")
    no_note = record(id="b", examples=[ExampleSentence(japanese="話す。")])
    complete = record(
        id="c", usage_notes="something", examples=[ExampleSentence(japanese="話す。")]
    )

    assert [r.id for r in ai_targets([no_meaning, no_example, no_note, complete])] == [
        "z",
        "a",
    ]


def test_an_empty_usage_note_is_a_complete_answer_when_nothing_useful_applies() -> None:
    complete = record(
        meanings=["to speak"],
        examples=[ExampleSentence(japanese="話す。")],
        usage_notes="",
    )

    assert ai_targets([complete]) == []
    assert Ledger.missing_enrichment([complete]) == []


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
    assert parse_force_fields("meanings,examples,usage_notes", ai=True) == (
        "meanings",
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


def test_the_bare_word_path_uses_the_shared_rich_card_value_schema() -> None:
    assert ai_schema() is ai_schema_module.rich_card_schema()
    assert set(ai_schema().model_fields) == {"meanings", "examples", "usage_notes"}


def test_the_instructions_preserve_the_exact_headword_spelling() -> None:
    """Read from `prompts/enrich-bare-word.md`, the file that is actually sent.

    Asserting against a Python constant was the bug M7.6P was supposed to end
    and did not: the constant survived the commit, byte-identical to the file,
    so these assertions guarded a copy nothing sends. Deleting the clause from
    the shipped file left the suite green.
    """
    shipped = prompts.load(REPO_ROOT, "enrich-bare-word")

    assert "Use the exact\nspelling supplied in Expression" in shipped
    assert "a kana expression stays kana" in shipped


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
    text = prompts.load(REPO_ROOT, "enrich-bare-word") + "\n" + ai_prompt(record())

    assert "The base contains exactly the\ncharacters that reading spells" in text
    assert "an ASCII space separates each ruby group" in text


def test_recent_sentences_ride_along_as_variety_pressure() -> None:
    # Asked for twenty verbs in a row, a model writes twenty variations of
    # 毎日〜ます unless it can see that it already did.
    text = ai_prompt(record(), ["毎日話します。", "毎日食べます。"])

    assert "毎日話します。" in text
    assert "Recent examples from this run:" in text


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

    assert "Existing curated examples requiring annotations:" in text
    assert '"日本語を話します。"' in text
    assert "return" not in text.lower()


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

    assert "Existing curated examples requiring annotations:\n(none)" in text
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

    assert "Existing curated examples requiring annotations:" in text
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
                speech_level="casual",
            ),
        ),
    )

    assert [ex.japanese for ex in outcome.record.examples] == [
        "日本語を話します。",
        "昨日友達と話した。",
    ]
    assert [ex.register for ex in outcome.record.examples] == ["polite", "casual"]
    assert outcome.record.examples[0].english == "I speak Japanese."
    assert outcome.preserved is False


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
                instructions="Pronounce 日本語 as にほんご.",
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
    assert filled.instructions == "Pronounce 日本語 as にほんご."
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
        examples=[
            ExampleSentence(japanese="curated polite", register="polite"),
            ExampleSentence(japanese="curated casual", register="casual"),
        ],
        usage_notes="curated note",
    )

    outcome = apply_ai_result(
        curated, answer(generated("話します。"), usage_notes="new note")
    )

    assert outcome.changes == {}
    assert outcome.record.usage_notes == "curated note"


def test_a_rich_answer_fills_empty_meanings_with_examples_and_usage() -> None:
    bare = record(meanings=[])

    outcome = apply_ai_result(
        bare,
        answer(
            generated("話します。"),
            meanings=["to speak", "to talk"],
            usage_notes="Common in conversation.",
        ),
    )

    assert outcome.record.meanings == ["to speak", "to talk"]
    assert [example.japanese for example in outcome.record.examples] == ["話します。"]
    assert outcome.record.usage_notes == "Common in conversation."
    assert set(outcome.changes) == {"meanings", "examples", "usage_notes"}


def test_existing_meanings_are_curated_until_explicitly_forced() -> None:
    curated = record(meanings=["to speak"])
    parsed = answer(meanings=["to converse", "to address"])

    preserved = apply_ai_result(curated, parsed)
    replaced = apply_ai_result(curated, parsed, force_fields=("meanings",))

    assert preserved.record.meanings == ["to speak"]
    assert "meanings" not in preserved.changes
    assert replaced.record.meanings == ["to converse", "to address"]
    assert replaced.changes["meanings"] == (
        ["to speak"],
        ["to converse", "to address"],
    )


def test_duplicate_meanings_are_normalized_before_replacing_a_card() -> None:
    curated = record(meanings=["to speak"])

    deduplicated = apply_ai_result(
        curated,
        answer(meanings=[" to talk ", "to talk", "to address"]),
        force_fields=("meanings",),
    )

    assert deduplicated.record.meanings == ["to talk", "to address"]


def test_force_fields_lets_the_answer_replace_them() -> None:
    curated = record(
        examples=[ExampleSentence(japanese="curated", register="polite")],
        usage_notes="curated note",
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


def test_a_live_result_records_the_exact_request_and_input_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = record()
    monkeypatch.setattr(
        enrich.claude_client,
        "parse_call",
        FakeCall(ok("話します。", "話[はな]します。")),
    )

    result = enrich.enrich_ai(
        [item], model="m", style_guide="STYLE", instructions="TASK"
    )
    user_turn = ai_prompt(item)

    assert result.input_fingerprints == {
        item.id: prompts.fingerprint(user_turn),
    }
    assert result.provenance == {
        item.id: enrich.ai_request_fingerprint(
            item,
            provider="anthropic",
            style_guide="STYLE",
            instructions="TASK",
        )
    }


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
    assert passes[0]["provider"] == "anthropic"
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
    expected_inputs = {
        item.id: prompts.fingerprint(call.calls[index]["content"])
        for index, item in enumerate(many)
    }
    style_guide = prompts.load(root, "style-guide")
    task_template = prompts.load(root, "enrich-bare-word")
    expected_requests = {
        item.id: prompts.request_fingerprint(
            provider="anthropic",
            style_guide=style_guide,
            task_template=task_template,
            user_turn=call.calls[index]["content"],
            transport_prompt={
                "system": [style_guide, task_template],
                "user": call.calls[index]["content"],
            },
            schema=claude_client.wire_schema(ai_schema()),
        )
        for index, item in enumerate(many)
    }
    assert meta["ai_enrichment"] == {
        "version": 1,
        "model": "claude-opus-5",
        "provider": "anthropic",
        "request_fingerprints": expected_requests,
        "input_fingerprints": expected_inputs,
        "fields": {item.id: ["examples"] for item in many},
    }
    assert meta["provider"] == "anthropic"
    assert staging.review_run_id(meta) == meta["review_run_id"]
    assert meta["field_replacements"] == {
        "version": 1,
        "records": {
            item.id: {
                "examples": staging.replacement_fingerprint(item, "examples")
            }
            for item in sorted(many, key=lambda record: record.id)
        },
    }
    # Nothing reached the records; promote is what lands them.
    assert stored(root)["word:話す0:はなす"]["examples"] == []
    assert "janki promote" in capsys.readouterr().out


def test_two_large_ai_review_invocations_get_distinct_run_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            for _run in range(2)
            for index in range(enrich.STAGING_THRESHOLD)
        ]
    )
    patch_all(monkeypatch, call, FakeJpdb())
    command = ["--root", str(root), "enrich", "--ai", "--yes"]

    assert cli.main(command) == 0
    target = root / "staging" / "ai-enrichment.yaml"
    first = target.with_name("first-ai-enrichment.yaml")
    target.rename(first)
    assert cli.main(command) == 0

    _first_rows, first_meta = read_staging(first)
    _second_rows, second_meta = read_staging(target)
    assert first_meta["review_run_id"] != second_meta["review_run_id"]


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
    assert "already has meanings and examples" in capsys.readouterr().out


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


def test_a_failed_ledger_write_explains_that_a_re_run_cannot_restore_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid empty note leaves the record complete after its examples land.

    A default re-run therefore skips it. An explicit re-run can buy the same
    answer, but no changed field means there is still no ledger event to write.
    Neither route reconstructs attribution for the first accepted answer.
    """
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
    assert "default re-run skips" in err
    assert "identical answer has no field change" in err
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
    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    [entry] = book["records"]["word:話す:はなす"]["enriched"]
    assert entry["provider"] == "codex"


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
    branch must still write the *other* empty AI fields. A missing meaning makes
    this a default target; an explicitly named record can reach the same branch.
    M8.3's rewrite of `apply_ai_result` kept this behaviour but lost every test
    that pinned it: narrowing the writable list to force-fields-only left the
    whole suite green while silently never writing a note again."""
    stored = record(
        meanings=[],
        examples=[ExampleSentence(japanese="人と話す。", english="x", register="casual")]
    )

    outcome = apply_ai_result(
        stored,
        answer(
            generated("毎日話します。", furigana="毎日[まいにち] 話[はな]します。"),
            meanings=["to speak"],
            usage_notes="Casual speech often drops the particle.",
        ),
    )

    assert "meanings" in outcome.changes
    assert outcome.record.meanings == ["to speak"]
    assert "usage_notes" in outcome.changes
    assert outcome.record.usage_notes == "Casual speech often drops the particle."
    # The stored casual sentence survives and the empty polite slot fills.
    assert [ex.japanese for ex in outcome.record.examples] == [
        "人と話す。",
        "毎日話します。",
    ]
    assert [ex.register for ex in outcome.record.examples] == ["casual", "polite"]


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
            answer(
                generated(
                    "毎日話す。",
                    furigana="毎日[まいにち] 話[はな]す。",
                    speech_level="casual",
                )
            ),
            "end_turn",
            None,
        ),
        model="offline-model",
        positions={accepted.id: 0},
        recent=[],
    )

    assert any(
        "did not fill an unoccupied polite/casual slot" in warning
        for warning in result.warnings
    )


def test_the_ai_pass_reports_a_romaji_that_disagrees_with_its_own_sentence() -> None:
    """`--ai` asks for romaji and `settle_example_romaji` checks it, so the
    disagreement has to reach a human.

    The romaji the model sends is kept when every letter transliterates the
    reading its own furigana gives, because word spacing is segmentation and
    janki does not segment. When it does not agree, janki keeps the sentence
    and falls back to the mechanical transliteration — the safe half — but a
    sentence and a romaji that disagree mean the answer was not internally
    consistent, and that is worth a look rather than a quiet repair.
    """
    from types import SimpleNamespace

    from japanese_anki.enrich import apply_ai_result
    from japanese_anki.models import SourceReference, VocabularyRecord

    record = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        source=SourceReference(type="shirabe", imported_from="x.csv"),
    )
    parsed = SimpleNamespace(
        examples=[
            SimpleNamespace(
                japanese="毎日話します。",
                furigana="毎日[まいにち] 話[はな]します。",
                romaji="totally wrong",
                english="I speak every day.",
                speech_level="polite",
            )
        ],
        usage_notes="",
    )

    outcome = apply_ai_result(record, parsed)

    assert outcome.romaji_rejected, "the disagreement is carried, not swallowed"
    assert "does not transliterate" in outcome.romaji_rejected[0]
    [example] = outcome.record.examples
    assert example.romaji == "mainichihanashimasu.", "and the safe half is kept"


def test_the_models_romaji_reaches_the_record_when_it_verifies() -> None:
    """The other half, and the bug that hid behind the schema's own wording.

    `apply_ai_result` built its `ExampleSentence` without copying `romaji` from
    the model's answer, so `settle_example_romaji` was always handed an empty
    string and always fell back to the mechanical transliteration. The schema
    field said "Ignored; janki regenerates this from the furigana" — which that
    one missing line made true, long after the code around it had stopped
    intending it.

    Every example therefore arrived unsegmented, and the backfill pass built to
    segment them was paying a model to undo this.
    """
    from types import SimpleNamespace

    from japanese_anki.enrich import apply_ai_result
    from japanese_anki.models import SourceReference, VocabularyRecord

    record = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        source=SourceReference(type="shirabe", imported_from="x.csv"),
    )
    parsed = SimpleNamespace(
        examples=[
            SimpleNamespace(
                japanese="毎日話します。",
                furigana="毎日[まいにち] 話[はな]します。",
                romaji="mainichi hanashimasu.",
                english="I speak every day.",
                speech_level="polite",
            )
        ],
        usage_notes="",
    )

    outcome = apply_ai_result(record, parsed)

    [example] = outcome.record.examples
    assert example.romaji == "mainichi hanashimasu.", "the spacing survives"
    assert not outcome.romaji_rejected


# --- settling a provisional meaning -------------------------------------------
#
# `docs/DESIGN.md` names the AI pass as one of the two things that may settle a
# provisional meaning, because it reads the card. These pin that it actually
# does — the sentence was written into the design before the code honoured it.


def provisional(**overrides: Any) -> VocabularyRecord:
    """An extract-sourced record whose meanings are still a model claim."""
    values: dict[str, Any] = {
        "source": SourceReference(type="extract", imported_from="medical.pdf"),
        "meanings": ["mumps"],
        "examples": [ExampleSentence(japanese="子供がおたふくにかかりました。")],
    }
    values.update(overrides)
    return mark_provisional(record(**values))


def test_writing_a_meaning_settles_the_mark_it_overwrote() -> None:
    """Otherwise the mark outlives the write, and the next jpdb run reads the
    broken value binding as a human edit — reporting "edited since extraction
    ... kept as curated" about a value janki wrote itself, and rewriting the
    collection to say so."""
    marked = provisional()
    assert provisional_fields(marked) == ["meanings"]

    outcome = apply_ai_result(
        marked,
        answer(meanings=["mumps (infectious parotitis)"]),
        force_fields=("meanings",),
    )

    assert outcome.record.meanings == ["mumps (infectious parotitis)"]
    assert provisional_fields(outcome.record) == []
    # The mark is gone, not merely inactive: an inactive one still reads as a
    # stale edit to the next pass that looks.
    assert PROVISIONAL_FIELDS_KEY not in outcome.record.source.raw_fields


def test_a_model_that_restates_the_meaning_settles_it_too() -> None:
    """Agreement is evidence. A mark means nobody who can read this card has
    confirmed the claim, and here somebody just did.

    Without this the 135 meanings restored from source archives would keep a
    mark permanently whenever the model agreed with them: the value never
    changes, so no janki operation would ever have cause to touch the field
    again, and nothing surfaces the mark to a person either."""
    marked = provisional()

    outcome = apply_ai_result(
        marked, answer(meanings=["mumps"]), force_fields=("meanings",)
    )

    assert outcome.record.meanings == ["mumps"]
    assert "meanings" not in outcome.changes
    assert provisional_fields(outcome.record) == []


def test_an_answer_the_pass_discards_leaves_the_mark_alone() -> None:
    """No `--force-fields`, so a full field is not writable and the model's
    answer is dropped. Settling the mark on an answer that was never written
    would record a confirmation that did not happen."""
    marked = provisional()

    outcome = apply_ai_result(marked, answer(meanings=["homely woman"]))

    assert outcome.record.meanings == ["mumps"]
    assert provisional_fields(outcome.record) == ["meanings"]


def test_a_settle_survives_the_pass_that_wrote_no_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settle has to reach the caller's save gate, not just the record.

    `apply_ai_result` clearing the mark is worth nothing on its own: every
    save gate in `cli.py` reads `result.changes`, and an answer that restates
    the stored meaning writes no field. Carried only on `changes`, the settled
    record was built and then dropped — `absorb_ai_call` kept the original —
    so the mark stayed on disk, the record came back in `status --unsettled`,
    and piping that list into this pass bought the identical answer again, at
    the identical price, for as long as the model kept agreeing.

    So `cleared` is its own channel, exactly as `EnrichResult.cleared` is for
    the jpdb pass and for the same reason.
    """
    marked = mark_provisional(
        record(
            source=SourceReference(type="extract", imported_from="medical.pdf"),
            meanings=["mumps"],
            usage_notes="Commonly said in full as おたふく風邪.",
            examples=[
                ExampleSentence(japanese="子供がおたふくにかかりました。",
                                register="polite"),
                ExampleSentence(japanese="おたふくで一週間休んだ。", register="casual"),
            ],
        )
    )
    assert provisional_fields(marked) == ["meanings"]
    # The model is shown the claim and restates it. Nothing else is fillable:
    # both register slots are occupied and the usage note is written.
    monkeypatch.setattr(
        enrich.claude_client,
        "parse_call",
        FakeCall(CallResult(answer(meanings=["mumps"]), "end_turn", None)),
    )

    # Named explicitly, which is what the pipeline does: a complete record is
    # not a content-defined target, so `status --unsettled --format ids` is
    # how it gets selected at all.
    result = enrich.enrich_ai(
        [marked], model="m", style_guide="S", instructions="I",
        ids=[marked.id], force_fields=("meanings",),
    )

    assert result.changes == {}
    assert result.cleared == {marked.id: ["meanings"]}
    # The settled object is the one the caller will save, not the original.
    assert provisional_fields(result.records[0]) == []
    assert result.records[0].meanings == ["mumps"]


def test_an_answer_that_offers_nothing_confirms_nothing() -> None:
    """Settling on *writability* alone would be enough while the wire schema
    guarantees a non-empty `meanings` — and wrong the moment anything hands
    `apply_ai_result` an answer it did not validate, which its signature
    (`parsed: Any`, read via `getattr`) allows. An empty answer writes no
    field, so it has confirmed nothing; treating the field as settled would
    retire the mark on the strength of silence.
    """
    marked = mark_provisional(
        record(
            source=SourceReference(type="extract", imported_from="medical.pdf"),
            meanings=["mumps"],
            examples=[ExampleSentence(japanese="子供がおたふくにかかりました。")],
        )
    )
    silent = SimpleNamespace(meanings=[], examples=[], usage_notes="")

    outcome = apply_ai_result(marked, silent, force_fields=("meanings",))

    assert outcome.record.meanings == ["mumps"]
    assert outcome.cleared == []
    assert provisional_fields(outcome.record) == ["meanings"]


def test_a_stale_mark_on_an_untouched_field_is_left_for_its_reporter() -> None:
    """A mark whose binding is broken means the field really was edited after
    extraction. The jpdb pass clears it with a warning saying so, and that
    warning is the only notice the user gets — so a pass that did not write the
    field must not swallow it by clearing the mark first."""
    edited = replace(provisional(), meanings=["mumps (hand-checked)"])
    assert provisional_fields(edited) == []  # bound to the extraction value

    outcome = apply_ai_result(edited, answer(meanings=["mumps"]))

    assert outcome.record.meanings == ["mumps (hand-checked)"]
    assert PROVISIONAL_FIELDS_KEY in outcome.record.source.raw_fields


def test_a_stale_mark_on_a_field_this_pass_overwrote_is_cleared() -> None:
    """The mirror case, and the one that decides the rule.

    Here a person edited the meaning after extraction (breaking the binding)
    and then asked the AI pass to overwrite it anyway. The mark now describes a
    value two writes gone. Keeping it because it is "stale rather than active"
    arms exactly the misreport this settle exists to prevent: the next jpdb run
    would announce "edited since extraction ... kept as curated" about a value
    the model wrote seconds ago.

    So what decides is whether this pass acted on the field, not whether the
    mark still matched.
    """
    edited = replace(provisional(), meanings=["mumps (hand-checked)"])

    outcome = apply_ai_result(
        edited, answer(meanings=["mumps (infectious parotitis)"]),
        force_fields=("meanings",),
    )

    assert outcome.record.meanings == ["mumps (infectious parotitis)"]
    assert PROVISIONAL_FIELDS_KEY not in outcome.record.source.raw_fields


def test_a_confirming_answer_is_written_to_disk_not_reported_as_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point, at the layer the user meets it.

    `enrich --ai` bails early on `if not result.changes`, prints "Nothing
    written", and returns before saving. An answer that restates the stored
    meaning fills no field, so it took that exit — and the settle it had just
    computed never reached the file. The record kept its mark, came back in
    `status --unsettled`, and the next run down that pipeline paid for the same
    answer again.
    """
    marked = mark_provisional(
        record(
            source=SourceReference(type="extract", imported_from="medical.pdf"),
            meanings=["mumps"],
            usage_notes="Commonly said in full as おたふく風邪.",
            examples=[
                ExampleSentence(japanese="子供がおたふくにかかりました。",
                                register="polite"),
                ExampleSentence(japanese="おたふくで一週間休んだ。", register="casual"),
            ],
        )
    )
    root = project(tmp_path, [marked])
    assert "provisional_fields" in stored(root)[marked.id]["source"]["raw_fields"]
    patch_all(
        monkeypatch,
        FakeCall(CallResult(answer(meanings=["mumps"]), "end_turn", None)),
        FakeJpdb({}),
    )

    code = cli.main([
        "--root", str(root), "enrich", "--ai", "--yes",
        "--force-fields", "meanings", marked.id,
    ])

    assert code == 0
    written = stored(root)[marked.id]
    assert written["meanings"] == ["mumps"]
    # Settled on disk: the next `status --unsettled` no longer names it, and
    # nobody pays to ask the same question again.
    assert "provisional_fields" not in written["source"]["raw_fields"]
    assert "Nothing written" not in capsys.readouterr().out


def test_the_staging_route_carries_a_settle_with_no_field_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staging route writes only the records it thinks the run touched, and
    it is the *only* write that run makes — the normalized file is left for
    `promote`. So a record whose sole outcome is a settled mark has to be in
    that file or the settle is not deferred, it is lost, and the record comes
    back marked with another paid call behind it.
    """
    marked = mark_provisional(
        record(
            source=SourceReference(type="extract", imported_from="medical.pdf"),
            meanings=["mumps"],
            usage_notes="Commonly said in full as おたふく風邪.",
            examples=[
                ExampleSentence(japanese="子供がおたふくにかかりました。",
                                register="polite"),
                ExampleSentence(japanese="おたふくで一週間休んだ。", register="casual"),
            ],
        )
    )
    # The AI staging route is chosen by run size, not by a flag, so the settle
    # rides in a run big enough to take it.
    filler = [
        record(id=f"word:話す{index}:はなす", expression=f"話す{index}")
        for index in range(enrich.STAGING_THRESHOLD - 1)
    ]
    root = project(tmp_path, [marked, *filler])
    patch_all(
        monkeypatch,
        FakeCall(
            CallResult(answer(meanings=["mumps"]), "end_turn", None),
            *[
                CallResult(
                    answer(generated(f"話す{index}。", furigana=f"話[はな]す{index}。")),
                    "end_turn",
                    None,
                )
                for index in range(enrich.STAGING_THRESHOLD - 1)
            ],
        ),
        FakeJpdb(),
    )

    # Named explicitly, as `status --unsettled --format ids` would: a complete
    # record is not a content-defined target, so without the ids the run is one
    # short of the threshold and takes the diff route instead.
    code = cli.main([
        "--root", str(root), "enrich", "--ai", "--yes",
        "--force-fields", "meanings",
        marked.id, *[item.id for item in filler],
    ])

    assert code == 0
    staged, _meta = read_staging(root / "staging" / "ai-enrichment.yaml")
    settled = [item for item in staged if item.id == marked.id]
    assert settled, "the settle-only record was dropped from the staging file"
    assert provisional_fields(settled[0]) == []
