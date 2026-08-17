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
    bind_source,
    has_source_binding,
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


def test_a_source_binding_covers_the_exact_current_pitch_value() -> None:
    card = bind_source(record(pitch_accent=["LHH"]))

    assert has_source_binding(card)

    card.pitch_accent[0] = "HLL"

    assert not has_source_binding(card)


def test_an_unbound_pitch_value_is_not_assumed_to_be_from_jpdb() -> None:
    assert not has_source_binding(record(pitch_accent=["LHH"]))


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
    """授業 (じゅぎょう) atamadaka, ``HHLLLL`` → ``ジュ'ギョウ``.

    **The merge gate**, and it has to be a 拗音 word with the drop *inside* it:
    under heiban the mark goes on the last unit however the kana are grouped, so
    a heiban 拗音 word tells the mappings apart not at all.

    Precisely which mutations this catches, since claiming the wrong one is how
    a gate stops gating — both verified by making the change and watching this
    fail:

    * **consuming one pattern character per mora instead of per kana**
      (``index += 1`` in ``_levels``) reads the levels off the wrong offsets and
      yields ``ジュギョ'ウ``, which the string assertion catches;
    * **treating every kana as its own mora** (an empty ``_ATTACHING``) does
      *not* change the string — the mark is appended after the accented mora's
      last kana, which is where a per-kana scan puts it too — so the span count
      is what catches that one.
    """
    assert to_aquestalk("じゅぎょう", "HHLLLL") == "ジュ'ギョウ"
    # Three morae from five kana, drawn as three spans plus the particle slot.
    assert render_pitch_html("じゅぎょう", ["HHLLLL"]).count('class="mora ') == 4


def test_a_heiban_youon_word_too() -> None:
    # 病院 (びょういん): five kana, four morae. Kept for the class, but see
    # above — heiban is not where the grouping shows.
    assert to_aquestalk("びょういん", "LLHHHH") == "ビョウイン'"


def test_the_moras_level_is_read_off_its_first_kana() -> None:
    """A 拗音's two kana carry one pitch. If a source ever writes the small kana
    with the *following* mora's level, the first kana is the one that cannot be
    the small one — so it is the one read.

    ``HLLLLL`` is the case that discriminates: off the first kana the levels are
    ``[H,L,L,L]`` and the mark lands after mora 1 (``ビョ'ウイン``); off the last
    they are ``[L,L,L,L]``, no drop at all, and it lands on the final mora.
    """
    assert to_aquestalk("びょういん", "HLLLLL") == "ビョ'ウイン"


def test_sokuon_is_a_mora() -> None:
    # 学校 (がっこう) heiban: four morae including っ. The AquesTalk string is
    # the same either way here — heiban again — so the count is asserted where
    # it shows, on the diagram.
    assert to_aquestalk("がっこう", "LHHHH") == "ガッコウ'"
    assert render_pitch_html("がっこう", ["LHHHH"]).count('class="mora ') == 5


def test_syllabic_n_is_a_mora() -> None:
    # 先生 (せんせい) nakadaka, accent 3 → センセ'イ. Four morae plus the
    # particle slot; merge ん into せ and the diagram loses one.
    assert to_aquestalk("せんせい", "LHHLL") == "センセ'イ"
    assert render_pitch_html("せんせい", ["LHHLL"]).count('class="mora ') == 5


def test_a_drop_before_a_sokuon_moves_the_mark() -> None:
    """Where っ's mora-hood changes the *spoken* string: the drop has to fall
    before it, not after. ``いっき`` with ``HLLL`` is ``イ'ッキ``; merge っ into
    the kana before it and the mark moves to ``イッ'キ``.

    The pattern is the input here, not a claim about any word's dictionary
    accent — what this pins is the grouping, and asserting an accent class from
    memory in a golden file is how a wrong one gets copied forward.
    """
    assert to_aquestalk("いっき", "HLLL") == "イ'ッキ"


def test_a_long_vowel_mark_is_a_mora() -> None:
    # カード, atamadaka: the accent is on カ, not on the ー that follows it.
    # The ー is spelled out as the vowel it lengthens — VOICEVOX's kana mode
    # answers 400 for カ'ード and 200 for カ'アド, so the old expectation here
    # was pinning a string the engine has never accepted.
    assert to_aquestalk("かーど", "HLLL") == "カ'アド"


def test_a_katakana_reading_passes_through() -> None:
    assert to_aquestalk("カード", "HLLL") == "カ'アド"


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
    # `None`, not a default: the caller distinguishes "force this" from "let
    # the engine choose, and mark the clip", and a default here would erase
    # that difference for exactly the homographs a pitch card exists to teach.
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
    # `rise` as well as `drop`: jpdb's notation draws the vertical at both
    # transitions. Without the rise a heiban word is a flat line, which a
    # reader takes for "no accent recorded" rather than "no drop".
    assert '<span class="mora high drop rise">ま</span>' in rendered
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


def test_a_one_mora_heiban_word_rises_into_its_particle() -> None:
    assert '<span class="mora particle high rise"></span>' in render_pitch_html(
        "め", ["LH"]
    )


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


def test_the_chosen_pattern_is_upper_cased() -> None:
    """Not cosmetic: the ledger's word-audio content fingerprint is
    fp(reading + this) and is defined as covering what was *spoken*. Retyping
    LHLL as lhll would report good audio as stale, over two strings that render
    identically."""
    assert select_pattern(record(pitch_accent=["lhl"])) == "LHL"
    assert select_pattern(record(pitch_accent=["LHH"], audio_accent="lhl")) == "LHL"


def test_a_word_that_starts_high_has_no_rise_into_it() -> None:
    """Atamadaka begins high, so there is nothing below to climb from. Marking
    a rise there draws a vertical on the left edge of the first mora, out of
    nowhere."""
    rendered = render_pitch_html("なる", ["HLL"])

    assert '<span class="mora high drop">な</span>' in rendered
    assert "rise" not in rendered


def test_a_flat_word_still_marks_its_one_transition() -> None:
    """する is す low, る high — exactly what jpdb draws. The rise is the only
    feature a heiban word has, and it is what distinguishes "flat" from
    "nothing known"."""
    rendered = render_pitch_html("する", ["LHH"])

    assert '<span class="mora low">す</span>' in rendered
    assert '<span class="mora high rise">る</span>' in rendered
    assert '<span class="mora particle high"></span>' in rendered
    assert "drop" not in rendered, "nothing falls in a heiban word"


@pytest.mark.parametrize(
    "reading,pattern,expected",
    [
        ("エスカレーター", "LHHHLLLL", "エスカレ'エタア"),
        ("コーヒー", "LHHHL", "コオヒイ'"),
        ("おおきい", "LHHHH", "オオキイ'"),
    ],
    ids=["escalator", "coffee", "already-spelled"],
)
def test_a_long_vowel_is_spelled_out_rather_than_marked(
    reading: str, pattern: str, expected: str
) -> None:
    """VOICEVOX's kana mode rejects `ー` outright.

    Found by running the real pipeline: `janki audio --words` died on
    エスカレーター with `UNKNOWN_TEXT: ーター`. Measured against the engine
    afterwards — `オー'` answers 400 and `オオ'` answers 200, and a bare
    `エスカレーター` with no accent mark fails too, so it is the character and
    not the mark placement. AquesTalk writes a long vowel as the vowel it
    lengthens.

    The `already-spelled` row is the control: a reading that never uses `ー`
    must pass through untouched, or this would be rewriting readings rather
    than respelling one character.
    """
    assert to_aquestalk(reading, pattern) == expected


def test_a_reading_that_opens_with_a_long_vowel_is_left_for_the_engine() -> None:
    """`ー` first has nothing to lengthen. Inventing a vowel there would be
    guessing at a reading; the engine refusing it is the honest outcome."""
    assert to_aquestalk("ーん", "LHH").startswith("ー")
