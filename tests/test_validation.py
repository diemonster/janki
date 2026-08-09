import unicodedata

import pytest

from japanese_anki.models import VocabularyRecord
from japanese_anki.validation import has_errors, validate_records


def _issue_messages(record: VocabularyRecord) -> list[str]:
    return [issue.message for issue in validate_records([record])]


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


def test_the_verdict_does_not_depend_on_unicode_composition() -> None:
    """A decomposed が is か plus a combining mark, so an unnormalized compare
    reads the same field as correct in NFD and wrong in NFC."""
    composed = "彼[かれ]が課長[かちょう]です。"
    decomposed = unicodedata.normalize("NFD", composed)

    flagged_composed = _issue_messages(_with_example(composed, japanese="彼が課長です。"))
    flagged_decomposed = _issue_messages(
        _with_example(decomposed, japanese="彼が課長です。")
    )

    assert any("missing a space" in m for m in flagged_composed)
    assert any("missing a space" in m for m in flagged_decomposed)


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
