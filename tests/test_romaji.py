import pytest

from japanese_anki.romaji import kana_to_romaji


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("あいうえお", "aiueo"),
        ("かきくけこ", "kakikukeko"),
        ("がぎぐげご", "gagigugego"),
        ("なにぬねの", "naninuneno"),
        ("ぱぴぷぺぽ", "papipupepo"),
        ("やゆよ", "yayuyo"),
        ("わ", "wa"),
    ],
)
def test_the_gojuuon_is_a_straight_table(kana: str, expected: str) -> None:
    assert kana_to_romaji(kana) == expected


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("し", "shi"),  # not "si": Hepburn spells the sound, not the column
        ("ち", "chi"),
        ("つ", "tsu"),
        ("ふ", "fu"),
        ("じ", "ji"),
        ("ぢ", "ji"),
        ("づ", "zu"),
    ],
)
def test_hepburn_spells_the_irregular_syllables_by_sound(kana: str, expected: str) -> None:
    assert kana_to_romaji(kana) == expected


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("とうきょう", "toukyou"),
        ("ゆうびん", "yuubin"),
        ("おおきい", "ookii"),
        ("せんせい", "sensei"),
    ],
)
def test_long_vowels_are_written_out_never_macronned(kana: str, expected: str) -> None:
    # Macron-free is the Shirabe convention and the one the curated records in
    # this repository already use, so a generated value matches a typed one.
    assert kana_to_romaji(kana) == expected
    assert kana_to_romaji(kana).isascii()


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("きょ", "kyo"),
        ("しゃ", "sha"),
        ("じゅぎょう", "jugyou"),
        ("りょこう", "ryokou"),
        ("ちゃ", "cha"),
        ("びょういん", "byouin"),
    ],
)
def test_digraphs_are_one_syllable_not_two(kana: str, expected: str) -> None:
    assert kana_to_romaji(kana) == expected


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("がっこう", "gakkou"),
        ("きって", "kitte"),
        ("ざっし", "zasshi"),
        ("いっぱい", "ippai"),
        ("まっすぐ", "massugu"),
        ("いっしょ", "issho"),
    ],
)
def test_the_sokuon_doubles_the_consonant_that_follows_it(kana: str, expected: str) -> None:
    assert kana_to_romaji(kana) == expected


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("いっち", "itchi"),
        ("こっち", "kotchi"),
        ("まっちゃ", "matcha"),
    ],
)
def test_the_sokuon_before_ch_is_t_not_a_doubled_c(kana: str, expected: str) -> None:
    # Doubling the first letter of "chi" would give "cchi"; traditional Hepburn
    # writes "tchi", which is also how matcha reached English.
    assert kana_to_romaji(kana) == expected


@pytest.mark.parametrize(("kana", "expected"), [("あっ", "a"), ("あっあ", "aa")])
def test_a_sokuon_with_no_consonant_to_double_is_dropped(kana: str, expected: str) -> None:
    assert kana_to_romaji(kana) == expected


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("しんぶん", "shinbun"),
        ("さんぽ", "sanpo"),
        ("せんぱい", "senpai"),
        ("あんまり", "anmari"),
        ("こんばん", "konban"),
    ],
)
def test_n_stays_n_before_b_p_and_m(kana: str, expected: str) -> None:
    """Modern Hepburn, not the JR-station spelling.

    These were `shimbun`, `sampo`, `sempai`, `ammari` until 2026-08-17. Both
    dialects are correct romaji and the choice is the owner's; the argument
    that settled it is that a learner typing the word back into an IME gets
    `ん` from `n` and nothing at all from `m`, so `n` is the spelling that
    round-trips through the tool they will actually use it in.
    """
    assert kana_to_romaji(kana) == expected


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("きんえん", "kin'en"),
        ("ほんや", "hon'ya"),
        ("たんい", "tan'i"),
        ("げんいん", "gen'in"),
    ],
)
def test_n_takes_an_apostrophe_before_a_vowel_or_y(kana: str, expected: str) -> None:
    # Without it "kinen" reads as き-ね-ん, a different word.
    assert kana_to_romaji(kana) == expected


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("にほん", "nihon"),
        ("かんじ", "kanji"),
        ("けんか", "kenka"),
        ("こんにちは", "konnichiha"),  # ん before ni stays n; see the particle test for "ha"
    ],
)
def test_n_stays_plain_everywhere_else(kana: str, expected: str) -> None:
    assert kana_to_romaji(kana) == expected


def test_the_apostrophe_is_not_needed_when_a_space_already_separates() -> None:
    # A caller converting furigana segments supplies the word boundary, and the
    # lookahead only ever peeks at the very next character.
    assert kana_to_romaji("ほん や") == "hon ya"


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("カタカナ", "katakana"),
        ("ニッポン", "nippon"),
        ("ﾆﾎﾝ", "nihon"),  # halfwidth, folded by NFKC before anything else runs
        ("ｺｰﾋｰ", "koohii"),  # halfwidth with a halfwidth prolongation mark
        ("ヴァイオリン", "vaiorin"),
    ],
)
def test_katakana_is_folded_onto_hiragana_first(kana: str, expected: str) -> None:
    assert kana_to_romaji(kana) == expected


def test_the_va_row_is_spelled_out_rather_than_shifted() -> None:
    # ヷヸヹヺ sit just past ヶ. Shifting them by the katakana offset lands on
    # unassigned code points and on the *combining* voiced sound marks, which
    # would silently turn the word into a diacritic.
    assert kana_to_romaji("ヷヸヹヺ") == "vavivevo"


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("コーヒー", "koohii"),
        ("ラーメン", "raamen"),
        ("スーパー", "suupaa"),
        ("メール", "meeru"),
        ("パーティー", "paatii"),
    ],
)
def test_the_long_vowel_mark_repeats_the_vowel_before_it(kana: str, expected: str) -> None:
    assert kana_to_romaji(kana) == expected


@pytest.mark.parametrize("kana", ["ー", "ンー", "ーあ"])
def test_a_long_vowel_mark_with_no_vowel_to_repeat_is_not_guessed(kana: str) -> None:
    assert kana_to_romaji(kana) == ""


@pytest.mark.parametrize(
    ("kana", "expected"),
    [
        ("フィルム", "firumu"),
        ("ウェブ", "webu"),
        ("チェック", "chekku"),
        ("ジェット", "jetto"),
        ("ファン", "fan"),
        ("ティッシュ", "tisshu"),
        ("ツアー", "tsuaa"),
    ],
)
def test_small_vowels_after_non_palatal_kana_are_loanword_sounds(kana: str, expected: str) -> None:
    assert kana_to_romaji(kana) == expected


@pytest.mark.parametrize(("kana", "expected"), [("ねぇ", "nee"), ("なぁ", "naa")])
def test_a_small_vowel_with_no_combination_lengthens_its_host(kana: str, expected: str) -> None:
    assert kana_to_romaji(kana) == expected


def test_wo_is_o_whether_it_is_a_particle_or_not() -> None:
    # Hepburn romanizes both the kana and the object particle as "o" (hon o
    # yomu), which is what this repository's curated example romaji already
    # does, so the mapping needs no tokenizer to be right.
    assert kana_to_romaji("を") == "o"
    assert kana_to_romaji("ほんをよむ") == "hon'oyomu"
    assert kana_to_romaji("ほん を よむ") == "hon o yomu"


def test_particle_ha_and_he_are_not_special_cased() -> None:
    # Read aloud these are "wa" and "e", but telling a particle from a syllable
    # needs word segmentation this module deliberately does not have. Spelling
    # the kana is the answer that is never silently wrong about which it is.
    assert kana_to_romaji("こんにちは") == "konnichiha"
    assert kana_to_romaji("がっこうへ") == "gakkouhe"


def test_a_mixed_sentence_applies_every_rule_at_once() -> None:
    # kyou (digraph) / gakkou (sokuon + ou) / shinbun (n stays n before b) /
    # n'o (n before a vowel, here を) / koohii (katakana + prolongation marks)
    # / 、。
    assert (
        kana_to_romaji("きょうはがっこうでしんぶんをよみ、コーヒーをのみました。")
        == "kyouhagakkoudeshinbun'oyomi, koohiionomimashita."
    )


def test_furigana_style_segments_keep_their_spacing() -> None:
    assert kana_to_romaji("がっこう へ いきます") == "gakkou he ikimasu"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("あ、い。", "a, i."),
        ("「あ」", '"a"'),
        ("ＡＴＭをつかう", "ATMotsukau"),  # fullwidth Latin folds to ASCII and passes through
        ("にほん・ご", "nihon go"),
        ("あ  い", "a i"),  # whitespace runs collapse
        ("", ""),
    ],
)
def test_punctuation_and_ascii_pass_through(text: str, expected: str) -> None:
    assert kana_to_romaji(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "日本語",  # the whole point: a kanji has no table reading
        "べんきょう中",  # one kanji is enough — no half-converted string escapes
        "ゝ",  # kana iteration mark: repeats a kana this scanner does not track
        "ヶ",  # read ka, ga or ko depending on the word
        "ヵ",
        "はな゙",  # a combining voiced mark NFKC could not attach
        "привет",
        "🍣",
    ],
)
def test_what_cannot_be_converted_confidently_comes_back_empty(text: str) -> None:
    # The project rule: a wrong value is worse than an empty one. Empty is
    # "no romaji available", which a caller can flag; a partial string looks
    # like an answer and gets stored as one.
    assert kana_to_romaji(text) == ""


@pytest.mark.parametrize(
    "reading",
    ["はなす", "たべる", "べんきょうする", "きっぷ", "ジャンパー", "しんかんせん", "みっつ"],
)
def test_a_converted_reading_is_always_plain_ascii(reading: str) -> None:
    result = kana_to_romaji(reading)

    assert result
    assert result.isascii()
