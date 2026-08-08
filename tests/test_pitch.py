"""Golden tests for :mod:`japanese_anki.pitch` — the audio milestone's merge gate.

Every expected AquesTalk string below is written out by hand from the accent
class it names, not derived the way the module derives it: a test that rebuilds
the answer the same way proves only self-consistency, and the whole reason this
module exists is that the obvious rule is wrong for heiban.

The accent classes, with the reading and jpdb's pattern (one character per kana,
plus one for the following particle):

* **heiban** 端 はし ``LHH`` — no drop anywhere, including the particle
* **atamadaka** 箸 はし ``HLL`` — drops after the first mora
* **nakadaka** 卵 たまご ``LHLL`` — drops inside the word
* **odaka** 橋 はし ``LHL`` — drops on the particle, not inside the word
"""

from __future__ import annotations

import unicodedata

import pytest

from japanese_anki.models import SourceReference, VocabularyRecord
from japanese_anki.pitch import (
    PitchError,
    morae,
    render_pitch_html,
    select_pattern,
    to_aquestalk,
)


def record(**overrides: object) -> VocabularyRecord:
    values: dict[str, object] = {
        "id": "word:橋:はし",
        "expression": "橋",
        "reading": "はし",
        "meanings": ["bridge"],
        "source": SourceReference(type="jpdb", imported_from="deck"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Morae — the regrouping the whole conversion rests on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reading", "expected"),
    [
        ("はし", ["は", "し"]),
        # 拗音: two kana, one mora. This is the case that makes a per-kana
        # reading of the pattern wrong.
        ("びょういん", ["びょ", "う", "い", "ん"]),
        # っ and ん are morae in their own right — 学校 is four, not three.
        ("がっこう", ["が", "っ", "こ", "う"]),
        ("せんせい", ["せ", "ん", "せ", "い"]),
        # A long vowel mark is its own mora too.
        ("かーど", ["か", "ー", "ど"]),
        # Small vowels attach, in a loanword reading as much as a native one.
        ("ふぁん", ["ふぁ", "ん"]),
    ],
)
def test_morae_group_the_way_accent_is_counted(reading: str, expected: list[str]) -> None:
    assert morae(reading) == expected


def test_a_leading_small_kana_is_kept_rather_than_dropped() -> None:
    # Nothing to attach to. Grouping is not the place to judge the data — the
    # length check is, and it can say which reading and which pattern.
    assert morae("ょう") == ["ょ", "う"]


# ---------------------------------------------------------------------------
# The four accent classes
# ---------------------------------------------------------------------------


def test_heiban_puts_the_mark_on_the_final_mora() -> None:
    # 端 (はし) LHH: high from the second mora and never falls, particle
    # included. AquesTalk carries one mark per phrase and the engine writes
    # heiban with it on the last mora.
    assert to_aquestalk("はし", "LHH") == "ハシ'"


def test_atamadaka_drops_after_the_first_mora() -> None:
    # 箸 (はし) HLL → ハ'シ
    assert to_aquestalk("はし", "HLL") == "ハ'シ"


def test_nakadaka_drops_inside_the_word() -> None:
    # 卵 (たまご) LHLL → タマ'ゴ
    assert to_aquestalk("たまご", "LHLL") == "タマ'ゴ"


def test_odaka_drops_on_the_particle() -> None:
    # 橋 (はし) LHL → ハシ'. The word itself never falls; the fall lands on the
    # particle, so the mark goes after the last mora.
    assert to_aquestalk("はし", "LHL") == "ハシ'"


def test_heiban_and_odaka_are_the_same_string_and_that_is_correct() -> None:
    """端 and 橋 are the minimal pair this module exists for, and in *isolated
    word audio* they are genuinely identical: the difference is the pitch of a
    particle that is not spoken. AquesTalk has one mark per phrase and nowhere
    to put the distinction. The diagram keeps them apart; the audio cannot."""
    assert to_aquestalk("はし", "LHH") == to_aquestalk("はし", "LHL")
    assert render_pitch_html("はし", ["LHH"]) != render_pitch_html("はし", ["LHL"])


def test_atamadaka_and_odaka_are_not_the_same_string() -> None:
    # The pair that *must* stay apart, and does: 箸 ハ'シ vs 橋 ハシ'.
    assert to_aquestalk("はし", "HLL") != to_aquestalk("はし", "LHL")


# ---------------------------------------------------------------------------
# The shapes that break a per-kana reading
# ---------------------------------------------------------------------------


def test_a_youon_word_counts_two_kana_as_one_mora() -> None:
    # 病院 (びょういん) heiban: five kana, four morae. Per-kana the mark would
    # land a mora late.
    assert to_aquestalk("びょういん", "LLHHHH") == "ビョウイン'"


def test_the_moras_level_is_read_off_its_first_kana() -> None:
    """A 拗音's two kana carry one pitch. If a source ever writes the small kana
    with the *following* mora's level, the first kana is the one that cannot be
    the small one — so it is the one read."""
    assert to_aquestalk("びょういん", "LHHHHH") == "ビョウイン'"


def test_sokuon_is_a_mora() -> None:
    # 学校 (がっこう) heiban: four morae including っ.
    assert to_aquestalk("がっこう", "LHHHH") == "ガッコウ'"


def test_syllabic_n_is_a_mora() -> None:
    # 先生 (せんせい) nakadaka, accent 3: the drop is after the third mora, and
    # ん is the second — miscount it and the mark lands on ン.
    assert to_aquestalk("せんせい", "LHHLL") == "センセ'イ"


def test_a_long_vowel_mark_is_a_mora() -> None:
    # カード, atamadaka: カ'ード, not カー'ド.
    assert to_aquestalk("かーど", "HLLL") == "カ'ード"


def test_a_katakana_reading_passes_through() -> None:
    assert to_aquestalk("カード", "HLLL") == "カ'ード"


def test_a_decomposed_dakuten_is_one_kana() -> None:
    """が typed as か + U+3099 is two codepoints and one kana. Counting
    codepoints would reject a pattern that fits perfectly well."""
    decomposed = unicodedata.normalize("NFD", "がっこう")
    assert len(decomposed) > len("がっこう")

    assert to_aquestalk(decomposed, "LHHHH") == "ガッコウ'"


# ---------------------------------------------------------------------------
# Refusals — never a plausible answer
# ---------------------------------------------------------------------------


def test_a_pattern_that_does_not_fit_the_reading_is_refused() -> None:
    with pytest.raises(PitchError) as caught:
        to_aquestalk("はし", "LH")

    message = str(caught.value)
    assert "はし" in message and "'LH'" in message
    assert "3 characters" in message, "say what was expected, not just that it was wrong"


def test_a_pattern_one_character_short_is_refused_too() -> None:
    # The classic mistake: one per kana, forgetting the particle slot.
    with pytest.raises(PitchError):
        to_aquestalk("たまご", "LHL")


def test_a_level_that_is_not_a_level_is_refused() -> None:
    with pytest.raises(PitchError) as caught:
        to_aquestalk("はし", "LX H".replace(" ", ""))

    assert "'X'" in str(caught.value)


def test_lower_case_levels_are_read() -> None:
    # Same data, different transcription habit; refusing it would be pedantry
    # rather than safety.
    assert to_aquestalk("はし", "hll") == "ハ'シ"


def test_a_reading_with_nothing_in_it_is_refused() -> None:
    with pytest.raises(PitchError):
        to_aquestalk("", "L")


# ---------------------------------------------------------------------------
# Choosing a pattern
# ---------------------------------------------------------------------------


def test_the_override_wins_because_somebody_listened() -> None:
    chosen = record(pitch_accent=["LHH", "LHL"], audio_accent="LHL")

    assert select_pattern(chosen) == "LHL"


def test_otherwise_jpdbs_primary() -> None:
    assert select_pattern(record(pitch_accent=["LHL", "LHH"])) == "LHL"


def test_no_pattern_is_no_answer_rather_than_a_default() -> None:
    # The caller skips and flags. Letting an engine guess would get exactly the
    # homographs wrong that a pitch card exists to teach.
    assert select_pattern(record()) is None
    assert select_pattern(record(pitch_accent=["", "  "])) is None


def test_a_blank_override_falls_through_rather_than_blanking_the_answer() -> None:
    assert select_pattern(record(pitch_accent=["LHL"], audio_accent="   ")) == "LHL"


# ---------------------------------------------------------------------------
# The diagram
# ---------------------------------------------------------------------------


def test_every_mora_gets_a_span_marked_high_or_low() -> None:
    rendered = render_pitch_html("たまご", ["LHLL"])

    assert rendered.count("<span") == 5, "three morae, the particle slot, and the wrapper"
    assert '<span class="mora low">た</span>' in rendered
    assert '<span class="mora high drop">ま</span>' in rendered
    assert '<span class="mora low">ご</span>' in rendered


def test_the_drop_is_marked_on_the_mora_the_pitch_falls_from() -> None:
    assert "drop" in render_pitch_html("はし", ["HLL"]).split("</span>")[0]


def test_the_particle_slot_is_drawn_so_odaka_is_visible() -> None:
    """The fall happens after the word. A diagram that stops at the last kana
    has nowhere to show it, which is the whole difference from heiban."""
    heiban = render_pitch_html("はし", ["LHH"])
    odaka = render_pitch_html("はし", ["LHL"])

    assert 'class="mora particle high"' in heiban
    assert 'class="mora particle low"' in odaka
    assert "drop" not in heiban
    assert "drop" in odaka


def test_every_pattern_is_drawn_primary_first() -> None:
    # A word with two accepted accents has two; showing one would teach that the
    # other is wrong.
    rendered = render_pitch_html("はし", ["LHL", "HLL"])
    first, second = rendered.split('<span class="pitch">')[1:]

    assert rendered.count('<span class="pitch">') == 2
    # 橋 first: は low, し high and falling. Then 箸: は high and falling.
    assert '<span class="mora low">は</span>' in first
    assert '<span class="mora high drop">は</span>' in second


def test_a_blank_pattern_is_skipped_rather_than_drawn_empty() -> None:
    assert render_pitch_html("はし", ["", "LHL"]).count('<span class="pitch">') == 1


def test_no_patterns_render_nothing() -> None:
    assert render_pitch_html("はし", []) == ""


def test_the_reading_is_escaped() -> None:
    # Not a real reading, but the renderer must not be the thing that trusts it.
    rendered = render_pitch_html("<&", ["LHH"])

    assert "&lt;" in rendered and "&amp;" in rendered
    assert "<&" not in rendered


def test_one_pattern_passed_as_a_string_is_refused() -> None:
    # A str is a Sequence[str] that iterates as characters, so this would
    # quietly render a diagram per character — or fail far from the mistake.
    with pytest.raises(PitchError) as caught:
        render_pitch_html("はし", "LHL")  # type: ignore[arg-type]

    assert "['LHL']" in str(caught.value)


def test_a_pattern_that_does_not_fit_is_refused_here_too() -> None:
    with pytest.raises(PitchError):
        render_pitch_html("たまご", ["LH"])
