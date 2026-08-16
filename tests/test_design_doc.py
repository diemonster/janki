"""`docs/DESIGN.md` says it wins when the code disagrees. This checks it doesn't.

The document opens by claiming precedence — "when code, plan, or another
document disagrees with it, this one wins and the other changes" — and nothing
enforced that. It drifted within ten hours of being written: an owner decision
about which engine voices a word was reversed the same evening, and DESIGN.md
kept the pre-reversal sentence. A reader following it would have re-migrated
word audio to an engine that cannot force a pitch accent, losing the one thing
橋/箸 cards exist to teach.

A prose document cannot be checked line by line. What *can* be checked is the
handful of load-bearing factual claims in it — which engine voices what, which
passes exist, which fields an external source fills. Those are the sentences a
reader acts on, and they are the ones that went stale.

Deliberately not asserted: the design *rules* ("janki never writes code that
reads Japanese"), which are judgements a test cannot make, and the prose
around each fact. If a claim here needs rewording, reword it and update the
substring — the test exists to make the drift visible, not to freeze the
sentence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN = (REPO_ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8")


def test_it_names_the_engine_that_actually_voices_a_word() -> None:
    """The claim that went stale, and the most expensive one to act on.

    `openai_tts` refuses a word outright — "橋 and 箸 would come out identical"
    — so a design document naming it as the word engine sends the next reader
    to undo a decision the owner made after measuring.
    """
    from japanese_anki.tts import openai_tts, voicevox

    assert "**VOICEVOX** — audio for words" in DESIGN
    assert "OpenAI TTS** — audio for example *sentences*" in DESIGN
    # And the code still refuses the pairing the doc now describes.
    assert hasattr(voicevox.VoicevoxProvider, "synthesize")
    with pytest.raises(openai_tts.TtsError, match="cannot force a pitch accent"):
        openai_tts.OpenAiSpeechProvider(api_key="x").synthesize("はし", forced_accent=True)


def test_it_names_the_fields_jpdb_actually_fills() -> None:
    """The doc listed two of eight, and called them "for card display" while
    jpdb is also the reading witness at the promote identity gate."""
    from japanese_anki.enrich import ENRICHABLE_FIELDS

    # Whitespace-normalised: the list is prose and wraps across lines.
    flat = " ".join(DESIGN.lower().split())
    for field in ENRICHABLE_FIELDS:
        assert field.replace("_", " ") in flat, field
    assert "reading witness" in flat


def test_it_does_not_promise_one_answer_when_there_are_four_passes() -> None:
    """Consolidating the passes is the open half of M7.6P. The doc described
    the destination as the present tense until a review caught it."""
    from japanese_anki import prompts

    sending = {
        name
        for name in ("extract-auto", "enrich-examples", "polish-meanings", "patterns")
        if prompts.load(REPO_ROOT, name)
    }
    assert len(sending) == 4, "four passes send templates"
    assert "four passes" in DESIGN, "and the doc says so rather than 'one answer'"


def test_it_attributes_stroke_order_to_the_source_that_licenses_it() -> None:
    """KANJIDIC gives stroke *count*; stroke *order* is KanjiVG under
    CC BY-SA 3.0, which a shared deck has to credit. Wrong attribution on a
    licensing-relevant fact is worse than a vague one."""
    from japanese_anki import kanji

    assert "KanjiVG" in DESIGN
    assert "CC BY-SA" in DESIGN
    assert "KANJIVG" in {name.upper() for name in dir(kanji)} | {
        value.upper() for value in vars(kanji) if isinstance(value, str)
    } or "kanjivg" in kanji.__doc__.lower()


def test_it_does_not_claim_note_ids_are_deterministic() -> None:
    """GUIDs are; note ids are timestamps genanki mints per build. The
    conclusion — rebuilds update rather than duplicate — is right, and the
    stated cause was wrong in three documents at once."""
    assert "Deterministic **GUIDs**" in DESIGN
    assert "Deterministic note IDs" not in DESIGN
