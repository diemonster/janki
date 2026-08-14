import unicodedata
from dataclasses import replace

import pytest

from japanese_anki.models import VocabularyRecord
from japanese_anki.validation import ValidationIssue, has_errors, validate_records


def _issue_messages(record: VocabularyRecord) -> list[str]:
    return [issue.message for issue in validate_records([record])]


def test_formatted_validation_output_includes_the_stable_code_and_location() -> None:
    issue = ValidationIssue(
        "error",
        "reading is missing",
        record_id="word:話す:はなす",
        source="lesson.yaml",
        code="missing-reading",
    )

    assert issue.format() == (
        "[ERROR missing-reading] lesson.yaml:word:話す:はなす: reading is missing"
    )


def test_validation_requires_reading_for_kanji() -> None:
    record = VocabularyRecord(
        id="word:話す:",
        expression="話す",
        meanings=["to speak"],
    )
    issues = validate_records([record])
    assert has_errors(issues)
    assert any("reading is missing" in issue.message for issue in issues)
    assert any("staging review" in issue.message for issue in issues)
    # One fault, one error: the ID complaint would say the same thing here.
    assert not any("word:<expression>:" in issue.message for issue in issues)


def test_a_reading_less_id_is_an_error_even_once_the_reading_is_filled_in() -> None:
    # The reading was repaired by hand but the ID still carries the empty
    # reading slot Anki's GUID was derived from.
    record = VocabularyRecord(
        id="word:話す:",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
    )

    messages = _issue_messages(record)

    assert has_errors(validate_records([record]))
    assert any("word:<expression>:" in message for message in messages)
    assert any("staging review" in message for message in messages)
    # The reading is present, so only the ID complaint fires.
    assert not any("reading is missing" in message for message in messages)


def test_a_well_formed_id_is_not_flagged() -> None:
    record = VocabularyRecord(
        id="word:話す:はなす",
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
    )

    assert _issue_messages(record) == []


def test_a_supplementary_plane_kanji_still_needs_a_reading() -> None:
    # 𠮟 is U+20B9F. A kanji test that stops at U+9FFF exempts this record from
    # the validator entirely, so the malformed id it carries is never reported.
    record = VocabularyRecord(
        id="word:𠮟る:",
        expression="𠮟る",
        meanings=["to scold"],
    )

    messages = _issue_messages(record)

    assert has_errors(validate_records([record]))
    assert any("reading is missing" in message for message in messages)


def test_a_reading_written_in_kanji_is_an_error() -> None:
    # The shape an importer mints from a row that supplied only one of the two
    # columns. The id looks well formed and is not: its reading slot holds a
    # spelling, and it is every bit as permanent as word:<expression>:.
    record = VocabularyRecord(
        id="word:話す:話す",
        expression="話す",
        reading="話す",
        meanings=["to speak"],
    )

    messages = _issue_messages(record)

    assert has_errors(validate_records([record]))
    assert any("reading is written in kanji" in message for message in messages)
    assert any("staging review" in message for message in messages)


def test_a_reading_written_in_supplementary_plane_kanji_is_an_error() -> None:
    record = VocabularyRecord(
        id="word:𠮟る:𠮟る",
        expression="𠮟る",
        reading="𠮟る",
        meanings=["to scold"],
    )

    assert any("reading is written in kanji" in message for message in _issue_messages(record))


def test_a_record_that_got_in_with_a_normalizing_kanji_reading_is_still_reported() -> None:
    # The id here is what an import that missed U+2F00 minted: both halves are
    # 一 after NFKC. Validation is the second line of defence and used to miss
    # it for the same reason the import gate did — one function, so one fix.
    record = VocabularyRecord(
        id="word:一:一",
        expression="一",
        reading="⼀",  # U+2F00 KANGXI RADICAL ONE
        meanings=["one"],
    )

    assert any("reading is written in kanji" in message for message in _issue_messages(record))


def test_the_staging_hint_says_what_to_do_without_sending_the_reader_elsewhere() -> None:
    # The remedy has to be readable from the error. Pointing at a document that
    # describes the review the reader just performed is how this check stopped
    # being actionable.
    record = VocabularyRecord(id="word:話す:", expression="話す", meanings=["to speak"])

    message = next(message for message in _issue_messages(record) if "staging review" in message)

    assert "id:" in message
    assert "README" not in message
    # No hardcoded path either: staging_dir is configurable, and a project
    # with staging_dir = "review" has no data/staging directory to look for.
    assert "data/staging" not in message


# --- accent patterns (M2.2's schema additions) ------------------------------


def _accented(pattern: str, *, audio_accent: str = "", reading: str = "はなす") -> VocabularyRecord:
    return VocabularyRecord(
        id=f"word:話す:{reading}",
        expression="話す",
        reading=reading,
        meanings=["to speak"],
        pitch_accent=[pattern] if pattern else [],
        audio_accent=audio_accent,
    )


def test_a_well_formed_accent_pattern_is_not_flagged() -> None:
    # One position per kana of はなす plus the particle slot that follows it.
    assert _issue_messages(_accented("LHHH")) == []


@pytest.mark.parametrize("pattern", ["LHH-", "L H H H", "0110", "HLLLx"])
def test_a_pattern_that_is_not_h_and_l_is_an_error(pattern: str) -> None:
    # Not a pattern at all: the converter reads it position by position, so
    # anything else is unusable rather than merely suspicious.
    record = _accented(pattern)

    assert has_errors(validate_records([record]))
    assert any("is not an accent pattern" in message for message in _issue_messages(record))


def test_a_pattern_of_the_wrong_length_warns_rather_than_erroring() -> None:
    # That the pattern covers the following particle is community-verified, not
    # documented, so a mismatch means "look at this", not "this file is wrong".
    record = _accented("LHH")  # 3 positions for a 3-kana reading; 4 expected

    issues = validate_records([record])

    assert not has_errors(issues)
    assert any(
        issue.level == "warning" and "4 were expected" in issue.message for issue in issues
    )


def test_the_audio_override_is_held_to_the_same_rules() -> None:
    # audio_accent is the pattern synthesis actually uses when it is set, so a
    # typo there is the one that reaches the engine.
    record = _accented("LHHH", audio_accent="HxL")

    messages = _issue_messages(record)

    assert has_errors(validate_records([record]))
    assert any("audio_accent" in message and "not an accent pattern" in message
               for message in messages)
    assert not any("pitch_accent[0]" in message for message in messages)


def test_the_length_warning_stays_quiet_while_the_reading_is_empty() -> None:
    # A record with no reading is already reported for that; measuring a pattern
    # against a reading nobody has typed adds a second line for one fix.
    record = VocabularyRecord(
        id="word:ありがとう:ありがとう",
        expression="ありがとう",
        reading="",
        meanings=["thank you"],
        pitch_accent=["LHHH"],
    )

    assert _issue_messages(record) == []


def test_every_pattern_on_the_record_is_checked_not_just_the_primary() -> None:
    record = _accented("LHHH")
    record.pitch_accent = ["LHHH", "not-a-pattern"]

    messages = _issue_messages(record)

    assert any("pitch_accent[1]" in message for message in messages)
    assert not any("pitch_accent[0]" in message for message in messages)


def test_a_kana_only_record_is_not_flagged_for_its_id() -> None:
    # Kana-only records default reading to the expression, so their IDs are
    # well formed; nothing here should look like the malformed shape.
    record = VocabularyRecord(
        id="word:ありがとう:ありがとう",
        expression="ありがとう",
        reading="ありがとう",
        meanings=["thank you"],
    )

    assert _issue_messages(record) == []


def test_a_lower_case_pattern_is_read_rather_than_refused() -> None:
    """`pitch.to_aquestalk` reads h/l deliberately — same data, different
    transcription habit. This check is an *error*, so disagreeing with it would
    make `janki build` refuse a whole deck over a pattern that converts and
    speaks correctly."""
    assert _issue_messages(_accented("lhhh")) == []


def test_a_decomposed_reading_is_counted_in_kana_not_codepoints() -> None:
    """が typed as か + U+3099 is two codepoints and one kana. Counting
    codepoints warns that audio will skip a record `to_aquestalk` handles fine,
    and sends the reviewer to 'fix' a correct pattern into a real mismatch."""
    decomposed = unicodedata.normalize("NFD", "がっこう")
    assert len(decomposed) == 5, "four kana, five codepoints"

    assert _issue_messages(_accented("LHHHH", reading=decomposed)) == []


_SENTENCE = "家族と城崎温泉に行きました。"


def _with_example(furigana: str, japanese: str = _SENTENCE) -> VocabularyRecord:
    from japanese_anki.models import ExampleSentence

    return VocabularyRecord(
        id="word:行く:いく",
        expression="行く",
        reading="いく",
        meanings=["to go"],
        examples=[ExampleSentence(japanese=japanese, furigana=furigana, english="x")],
    )


def test_furigana_missing_a_space_is_flagged() -> None:
    """Anki splits the field on spaces and draws the reading over everything
    back to the previous one — so an unspaced group spills onto the kana before
    it. Found on a real card: きのさきおんせん rendered across と城崎温泉."""
    record = _with_example("家族[かぞく]と城崎温泉[きのさきおんせん]に行[い]きました。")

    messages = _issue_messages(record)

    assert any("'と城崎温泉'" in m and "'に行'" in m for m in messages)
    # A warning, not an error: `has_errors` is what `build` refuses on, and a
    # renderable-but-wrong field must not stop a deck from building.
    assert not has_errors(validate_records([record]))


def test_correctly_spaced_furigana_is_not_flagged() -> None:
    record = _with_example("家族[かぞく]と 城崎温泉[きのさきおんせん]に 行[い]きました。")

    assert not any("missing a space" in m for m in _issue_messages(record))


def test_the_classic_ocha_case_is_flagged() -> None:
    """お茶[ちゃ] puts ちゃ over both characters; the correct form is お 茶[ちゃ].
    Told apart from お茶[おちゃ] — correct whole-word ruby — by whether the
    reading starts with the same kana the run does."""
    record = VocabularyRecord(
        id="word:お茶:おちゃ",
        expression="お茶",
        reading="おちゃ",
        meanings=["tea"],
        furigana="お茶[ちゃ]",
    )

    assert any("'お茶'" in m for m in _issue_messages(record))


def test_a_group_at_the_very_start_needs_no_space() -> None:
    record = VocabularyRecord(
        id="word:行く:いく",
        expression="行く",
        reading="いく",
        meanings=["to go"],
        furigana="行[い]く",
    )

    assert not any("missing a space" in m for m in _issue_messages(record))


def test_unbalanced_brackets_still_win_over_the_spacing_check() -> None:
    # A field janki cannot parse gets the error it deserves, not a confusing
    # second complaint derived from a broken parse. The input has to contain a
    # group the spacing check *would* flag — `に行[い]` — or the test passes
    # whether the branches are exclusive or not.
    record = _with_example("家族[かぞく]と 城崎温泉[きのさきおんせん]に行[い]きました[。")

    messages = _issue_messages(record)

    assert any("unbalanced" in m for m in messages)
    assert not any("missing a space" in m for m in messages)


def test_whole_word_ruby_over_leading_kana_is_not_flagged() -> None:
    """お茶[おちゃ] is correct: the ruby covers the word's own leading kana.
    Flagging it would be worse than useless — acting on the advice gives
    お 茶[おちゃ], whose reading is おおちゃ, which is the defect the check
    exists to catch."""
    record = _with_example(
        "毎日[まいにち] お茶[おちゃ]を 飲[の]みます。", japanese="毎日お茶を飲みます。"
    )

    assert not any("missing a space" in m for m in _issue_messages(record))


def test_a_spill_whose_run_starts_with_punctuation_is_caught() -> None:
    """Punctuation cannot be part of the word a ruby annotates, so a run
    beginning with one has swallowed the sentence in front of it."""
    record = _with_example(
        "毎日[まいにち]、妻と日本語[にほんご]で 話[はな]します。",
        japanese="毎日、妻と日本語で話します。",
    )

    assert any("'、妻と日本語'" in m for m in _issue_messages(record))


def test_per_kanji_furigana_without_spaces_is_not_flagged() -> None:
    # 日[にっ]本[ぽん]語[ご] abuts with no spaces and is correct: there is
    # nothing between the groups for a reading to spill onto.
    record = _with_example("日[にっ]本[ぽん]語[ご]", japanese="日本語")

    assert not any("missing a space" in m for m in _issue_messages(record))


def test_ruby_over_a_loanword_is_not_flagged() -> None:
    record = VocabularyRecord(
        id="word:ATM:エーティーエム",
        expression="ＡＴＭ",
        reading="エーティーエム",
        meanings=["ATM"],
        furigana="ＡＴＭ[エーティーエム]",
    )

    assert not any("missing a space" in m for m in _issue_messages(record))


def test_a_spill_whose_kana_matches_the_reading_is_still_caught() -> None:
    """は花[はな] is structurally identical to お茶[おちゃ] — leading kana that
    begins the reading — so a prefix test waves it through. It is a real spill:
    は vanishes from the reconstructed reading, and from the romaji and audio
    built on it."""
    record = _with_example(
        "庭[にわ]は花[はな]が きれいです。", japanese="庭は花がきれいです。"
    )

    assert any("'は花'" in m for m in _issue_messages(record))


def test_an_honorific_prefix_under_its_ruby_is_not_flagged() -> None:
    record = VocabularyRecord(
        id="word:ご飯:ごはん",
        expression="ご飯",
        reading="ごはん",
        meanings=["cooked rice"],
        furigana="ご飯[ごはん]",
    )

    assert not any("missing a space" in m for m in _issue_messages(record))


def test_a_correct_honorific_survives_decomposition() -> None:
    """A decomposed ご is こ plus a combining mark, so the honorific check has to
    look at the *normalized* head — `qc` takes the first character raw (to keep
    a leading U+3000 that NFKC would delete) and normalizes it before asking
    whether it is an honorific. Read raw, a decomposed ご is こ, which is not in
    the allowlist, and a correct ご飯[ごはん] reports as a spill."""
    composed = VocabularyRecord(
        id="word:ご飯:ごはん", expression="ご飯", reading="ごはん",
        meanings=["cooked rice"], furigana="ご飯[ごはん]",
    )
    decomposed = replace(
        composed, furigana=unicodedata.normalize("NFD", "ご飯[ごはん]")
    )

    assert not any("missing a space" in m for m in _issue_messages(composed))
    assert not any("missing a space" in m for m in _issue_messages(decomposed))


def test_a_real_spill_is_still_caught_when_decomposed() -> None:
    """The other direction, and the one the suite lost: a *detected* spill must
    survive decomposition too. Without it every composition test asserted only
    that nothing is flagged, which a checker that flags nothing at all passes.
    が decomposes to か — still kana, still not an honorific — so 課長[かちょう]
    with no space before it spills either way.

    The message is compared *after* normalizing, because the run is reported
    verbatim: in the decomposed field it really is か + U+3099 + 課長, and that
    is what the reader has to search their file for."""
    field = "彼[かれ]が課長[かちょう]です。"

    for furigana in (field, unicodedata.normalize("NFD", field)):
        messages = _issue_messages(_with_example(furigana))

        assert len(messages) == 1
        assert "missing a space before 'が課長'" in unicodedata.normalize(
            "NFC", messages[0]
        )


def test_a_spill_whose_run_starts_with_kanji_is_a_known_gap() -> None:
    """にほんご really is drawn across 妻と日本語 here, and this is not caught.

    It cannot be told from a legitimate 日[にっ]本[ぽん], or from whole-word ruby
    over a compound containing kana, without deciding where the word boundary
    is — the segmentation janki refuses to guess. Flagging the class would tell
    someone to add a space that breaks a correct field, which is the harm the
    check was rewritten to stop causing. Pinned so the gap is visible rather
    than mistaken for coverage."""
    record = _with_example(
        "毎日[まいにち]妻と日本語[にほんご]で 話[はな]します。",
        japanese="毎日妻と日本語で話します。",
    )

    assert not any("missing a space" in m for m in _issue_messages(record))


def test_a_full_width_space_does_not_separate_ruby_groups() -> None:
    """Ordinary in Japanese text, and Anki does not read it as a separator:
    `furigana_reading` drops only a single ASCII space, so the run before the
    group vanishes from the reading the romaji and audio are built on."""
    record = _with_example(
        "毎日[まいにち]　妻と日本語[にほんご]で 話[はな]します。",
        japanese="毎日　妻と日本語で話します。",
    )

    assert any("missing a space" in m for m in _issue_messages(record))


def test_the_warning_quotes_the_run_as_it_appears_in_the_field() -> None:
    """An NFKC-folded quote names a string the record does not contain, so
    nobody can find what to fix."""
    record = _with_example("毎日[まいにち]ﾆﾎﾝ語[にほんご]です。", japanese="毎日ﾆﾎﾝ語です。")

    assert any("'ﾆﾎﾝ語'" in m for m in _issue_messages(record))


def test_a_space_no_reading_annotates_is_reported() -> None:
    """A space in a furigana field means "the next group's run starts here".
    One before an unannotated katakana word survives into the reading, the
    romaji and the sentence audio, and draws on the card as a gap that the
    ExampleJapanese field beside it does not have. Nothing reported it: the
    spill check only inspects characters inside a group's run, and the jpdb
    comparison strips category-Z before comparing."""
    messages = _issue_messages(
        _with_example("日本語[にほんご]の ニュースが 少[すこ]し 分[わ]かります。")
    )

    assert len(messages) == 1
    assert "space before 'ニュースが'" in messages[0]


def test_the_notation_spaces_of_an_ordinary_field_are_not_reported() -> None:
    assert _issue_messages(_with_example("毎晩[まいばん]、 音楽[おんがく]を 聞[き]いて")) == []


# --- the teaching-content gate (M7.6T) ----------------------------------------


def _teaching_record(japanese: str, register: str = "") -> VocabularyRecord:
    from japanese_anki.models import ExampleSentence

    return VocabularyRecord(
        id="word:出発:しゅっぱつ",
        expression="出発",
        reading="しゅっぱつ",
        meanings=["departure"],
        examples=[ExampleSentence(japanese=japanese, english="x", register=register)],
    )


def _codes(record: VocabularyRecord) -> list[str]:
    return [issue.code for issue in validate_records([record])]


def test_a_topic_label_fragment_is_held() -> None:
    # The camera pilot's shape: a source line copied as an "example". について
    # is never a sentence-final predicate, so this is certain, not a guess.
    codes = _codes(_teaching_record("古い地図の出発について", register="polite"))

    assert codes == ["example-fragment"]


def test_a_dangling_conditional_is_held() -> None:
    codes = _codes(_teaching_record("もし出発がよければ", register="casual"))

    assert codes == ["example-fragment"]


def test_a_polite_sentence_labelled_casual_is_held() -> None:
    codes = _codes(_teaching_record("明日、九時に出発します。", register="casual"))

    assert codes == ["example-register-mismatch"]


def test_complete_polite_and_casual_sentences_pass() -> None:
    polite = _teaching_record("明日、九時に出発します。", register="polite")
    casual = _teaching_record("明日の出発、九時だよ。", register="casual")

    assert _codes(polite) == []
    assert _codes(casual) == []


def test_an_intentional_short_utterance_passes() -> None:
    # Short and casual is not the same thing as a fragment: a te-form request
    # is something a person actually says, and holding it would teach the gate
    # to reject real Japanese.
    codes = _codes(_teaching_record("ちょっと出発して。", register="casual"))

    assert codes == []


def test_an_elliptical_conditional_question_is_not_held() -> None:
    # 〜ば？ is a real elliptical suggestion; the punctuation is what separates
    # an intentional ellipsis from a clause cut off mid-thought.
    codes = _codes(_teaching_record("九時に出発すれば？", register="casual"))

    assert codes == []


def test_a_plain_ma_stem_verb_labelled_casual_is_not_held() -> None:
    # 励ました is the plain past of 励ます: it ends in the literal characters
    # ました, and the polite-form check must not read verb spelling as
    # politeness. A kanji before ます is undecidable, and undecidable means
    # unflagged.
    codes = _codes(_teaching_record("コーチが選手を励ました。", register="casual"))

    assert codes == []


def test_a_ba_final_noun_without_punctuation_is_not_held() -> None:
    # そば and ことば end in ば without being conditionals; the collection
    # legitimately holds complete sentences with no final punctuation.
    assert _codes(_teaching_record("好きな食べ物はそば", register="casual")) == []
    assert (
        _codes(_teaching_record("ありがとうは大切なことば", register="casual")) == []
    )


def test_an_elliptical_topic_question_is_not_held() -> None:
    # 何について？ is real spoken Japanese, structurally parallel to the
    # exempted 行けば？ — the punctuation marks the ellipsis as intentional.
    codes = _codes(_teaching_record("この本は何について？", register="casual"))

    assert codes == []


def test_a_lexicalized_polite_formula_labelled_casual_is_not_held() -> None:
    # すみません said to a friend is still すみません: a set phrase's polite
    # morphology carries no register information about the sentence around it.
    assert _codes(_teaching_record("あ、すみません", register="casual")) == []
    assert _codes(_teaching_record("お先に失礼します", register="casual")) == []


def test_the_sentence_final_tteba_is_not_a_conditional() -> None:
    # ば attaches to an e-stem, never a te-form: 〜ってば is the particle.
    assert _codes(_teaching_record("もういいってば", register="casual")) == []


def test_half_width_punctuation_cannot_blind_the_gate() -> None:
    # Camera transcription writes ｡ and ｣; the gate NFKC-normalizes first, so
    # the fragment and register checks see the same sentence a reader does.
    fragment = _codes(_teaching_record("古い地図の出発について｣", register="polite"))
    register = _codes(_teaching_record("明日、九時に出発します｡", register="casual"))

    assert fragment == ["example-fragment"]
    assert register == ["example-register-mismatch"]


def test_quoted_polite_speech_does_not_set_the_frames_register() -> None:
    # The quotation's politeness belongs to the speaker inside it; the frame
    # is judged by its own final form.
    codes = _codes(
        _teaching_record("彼が言ったのは「出発します」", register="casual")
    )

    assert codes == []


def test_a_kana_ma_stem_plain_form_is_not_read_as_polite() -> None:
    # はげます (plain) and 投げます (polite) share the [e-column]+ます surface,
    # so an e-column ending is undecidable — and undecidable means unflagged.
    codes = _codes(_teaching_record("ともだちをはげました", register="casual"))

    assert codes == []


def test_unambiguous_polite_negatives_and_volitionals_are_held() -> None:
    # No dictionary form ends in ません or ましょう, so they need no stem
    # guard — narrowing them alongside ます made 食べません escape while
    # 食べませんでした was held.
    assert _codes(_teaching_record("毎日、朝ご飯を食べません。", register="casual")) == [
        "example-register-mismatch"
    ]
    assert _codes(_teaching_record("一緒に食べましょう。", register="casual")) == [
        "example-register-mismatch"
    ]
    assert _codes(_teaching_record("九時に出発しますか。", register="casual")) == [
        "example-register-mismatch"
    ]


def test_the_super_polite_copula_is_not_a_greeting() -> None:
    # A bare ございます exemption swallowed でございます — the one です-family
    # form most in need of the check.
    assert _codes(
        _teaching_record("こちらは会議室でございます。", register="casual")
    ) == ["example-register-mismatch"]
    assert _codes(_teaching_record("おはようございます", register="casual")) == []


def test_the_middle_dot_ellipsis_cannot_blind_the_register_gate() -> None:
    # ・・・ is the trailing-off ellipsis camera transcription writes (and what
    # half-width ･･･ NFKC-folds into); it must strip like … does.
    assert _codes(_teaching_record("頑張ります・・・", register="casual")) == [
        "example-register-mismatch"
    ]


def test_formula_exemptions_cover_every_spelling_they_ship_in() -> None:
    # The exemption matches surface text while the rule matches morphology,
    # so each formula is listed in each spelling — the same rule the
    # conjugation honorific table follows.
    assert _codes(_teaching_record("じゃ、行ってきます", register="casual")) == []
    assert _codes(_teaching_record("お先にしつれいします", register="casual")) == []


def test_an_imperative_ni_tsuite_is_not_a_fragment() -> None:
    # について is also に+着いて: 席について is a real classroom imperative. The
    # fragment shape the camera wrote is the genitive 〜の〜について.
    assert _codes(_teaching_record("みんな、席について", register="casual")) == []
    assert _codes(_teaching_record("古い地図の出発について", register="polite")) == [
        "example-fragment"
    ]


def test_the_humble_auxiliary_is_not_a_meal_greeting() -> None:
    # いただきます standing alone is the set phrase; as a clause tail it is
    # the productive humble auxiliary — the most formal keigo there is.
    assert _codes(
        _teaching_record("本日は休業させていただきます", register="casual")
    ) == ["example-register-mismatch"]
    assert _codes(_teaching_record("じゃ、いただきます", register="casual")) == []


def test_stacked_final_particles_do_not_hide_politeness() -> None:
    assert _codes(_teaching_record("明日も行きますよね", register="casual")) == [
        "example-register-mismatch"
    ]
