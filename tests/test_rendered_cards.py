"""What Anki actually draws — the templates rendered by Anki's own engine.

Every other template test in this suite reads the HTML *source* on disk. That
catches a missing link and a stray `<script>`, and it cannot catch any of the
things that only exist once Anki has processed the file: whether
``{{furigana:...}}`` puts the reading over the right characters, whether a
`{{#Field}}` section fires, whether a field name still resolves, whether
`[sound:...]` is consumed. Those are exactly the failures that reach a card
looking fine in the repository.

So this builds a real package, imports it into a scratch collection, and asks
Anki to render the cards — the same code path the desktop reviewer uses.

**It is not a device test.** It says nothing about AnkiMobile's or AnkiDroid's
rendering, about CSS or layout, or about whether the Shirabe app answers the
deep link. What it does prove is that the HTML janki ships turns into the HTML
it intends, which nothing checked before.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("genanki")
anki_collection = pytest.importorskip(
    "anki.collection",
    reason="the `anki` library renders these; it is in the dev extra",
)
from anki.import_export_pb2 import ImportAnkiPackageRequest  # noqa: E402

from japanese_anki.config import ProjectConfig  # noqa: E402
from japanese_anki.exporters.anki import AnkiBuildError, build_deck  # noqa: E402
from japanese_anki.models import ExampleSentence, VocabularyRecord  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project(root: Path) -> None:
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n',
        encoding="utf-8",
    )
    shutil.copytree(
        PROJECT_ROOT / "templates" / "japanese-study",
        root / "templates" / "japanese-study",
    )
    (root / "decks").mkdir()
    (root / "decks" / "d.yaml").write_text(
        'name: D\ndeck:\n  source: "../vocabulary.json"\n', encoding="utf-8"
    )


def kanji_data(root: Path) -> None:
    """A kanji entry for 話, so `{{#KanjiInfo}}` can fire.

    Without one `kanji.load_store` returns an empty store for a missing file and
    the block never renders — so the largest thing janki generates, a `<details>`
    with inline stroke SVG, went unchecked by the file whose whole subject is
    what Anki draws.
    """
    (root / "janki.toml").write_text(
        (root / "janki.toml").read_text(encoding="utf-8")
        + 'kanji_file = "kanji.json"\n',
        encoding="utf-8",
    )
    (root / "kanji.json").write_text(
        json.dumps({
            "話": {
                "stroke_count": 13,
                "meanings": ["talk", "speak"],
                "readings": [
                    {"kind": "on", "reading": "ワ", "examples": [
                        {"written": "会話", "pronounced": "かいわ", "gloss": "conversation"}
                    ]},
                    {"kind": "kun", "reading": "はな.す", "examples": [
                        {"written": "話す", "pronounced": "はなす", "gloss": "to speak"}
                    ]},
                ],
                "strokes": ["M1,1L2,2", "M3,3L4,4"],
            }
        }, ensure_ascii=False),
        encoding="utf-8",
    )


def render(
    records: list[VocabularyRecord], media: dict[str, bytes] | None = None
) -> list[dict[str, Any]]:
    """Build these records into a package and render every card Anki makes.

    Returns one ``{"question", "answer", "expression"}`` per card, in Anki's
    order. The collection is scratch and thrown away — nothing here can touch a
    real one, which `tests/conftest.py` also guards globally.
    """
    root = Path(tempfile.mkdtemp())
    # Opened around the *whole* body, not only the render. The build is what
    # raises here — a record that fails validation, or a missing template
    # directory — and a `finally` starting after it left the leak in place for
    # exactly the red-test loop that re-runs `render()` most.
    try:
        _project(root)
        kanji_data(root)
        for name, payload in (media or {}).items():
            path = root / "media" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        (root / "vocabulary.json").write_text(
            json.dumps([record.to_dict() for record in records], ensure_ascii=False),
            encoding="utf-8",
        )
        package = root / "out.apkg"
        build_deck(root / "decks" / "d.yaml", ProjectConfig.load(root), package)

        collection = anki_collection.Collection(str(root / "scratch.anki2"))
        try:
            collection.import_anki_package(
                ImportAnkiPackageRequest(package_path=str(package))
            )
            drawn: list[dict[str, Any]] = []
            for card_id in collection.find_cards(""):
                card = collection.get_card(card_id)
                output = card.render_output()
                drawn.append(
                    {
                        "question": output.question_text,
                        "answer": output.answer_text,
                        "expression": card.note().fields[1],
                        # What Anki pulled *out* of the HTML to play. A negative
                        # assertion on "[sound:" is satisfied by losing the audio
                        # entirely, which is the failure worth catching.
                        "sounds": [tag.filename for tag in output.answer_av_tags],
                    }
                )
            return drawn
        finally:
            collection.close()
    finally:
        # Actually thrown away. These live outside pytest's `tmp_path`, so its
        # retention policy never reaches them and every run left a copy of the
        # templates, a package and a collection behind for good.
        shutil.rmtree(root, ignore_errors=True)


def record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "furigana": "話[はな]す",
        "meanings": ["to speak"],
        "verb_group": "godan",
    }
    values.update(overrides)
    return VocabularyRecord(**values)


@pytest.fixture(scope="module")
def drawn() -> list[dict[str, Any]]:
    """One build and one import for the whole file — both are slow."""
    return render([
        record(
            pitch_accent=["LHHL"],
            usage_notes="Takes を.",
            examples=[
                ExampleSentence(
                    japanese="毎日妻と話します。",
                    furigana="毎日[まいにち] 妻[つま]と 話[はな]します。",
                    english="I speak with my wife every day.",
                    register="polite",
                ),
            ],
        ),
    ])


def backs(drawn: list[dict[str, Any]]) -> list[str]:
    return [card["answer"] for card in drawn]


# --- the furigana filter -------------------------------------------------------


def test_a_reading_is_drawn_over_its_own_kanji_and_no_further(
    drawn: list[dict[str, Any]],
) -> None:
    """The premise every furigana check in `qc.py` reasons from, verified
    against Anki rather than assumed: a correctly spaced field puts はな over
    話 alone, leaving す outside the ruby."""
    answer = backs(drawn)[0]

    assert "<ruby><rb>話</rb><rt>はな</rt></ruby>" in answer
    assert "<rb>話す</rb>" not in answer, "す is not part of the reading"


def test_each_group_in_a_sentence_gets_its_own_ruby(
    drawn: list[dict[str, Any]],
) -> None:
    answer = backs(drawn)[0]

    assert "<ruby><rb>毎日</rb><rt>まいにち</rt></ruby>" in answer
    assert "<ruby><rb>妻</rb><rt>つま</rt></ruby>" in answer


def test_a_missing_separator_really_does_spill_the_reading() -> None:
    """The failure `spilled_furigana_groups` exists to catch, demonstrated
    against Anki itself rather than described. Without the space before 妻, the
    ruby base becomes `、妻` — つま is drawn over the comma as well as the word.

    The comma is still *on* the card, under the reading; what loses it is
    `qc.furigana_reading`, which drops the swallowed run and so feeds a romaji
    field and a sentence audio missing it. This is why the repair in
    `qc.repair_spilled_punctuation` is not cosmetic.
    """
    spilled = render([
        record(examples=[ExampleSentence(
            japanese="毎日、妻と話します。",
            furigana="毎日[まいにち]、妻[つま]と 話[はな]します。",
        )])
    ])

    answer = backs(spilled)[0]
    assert "<rb>、妻</rb>" in answer, "the ruby swallowed the punctuation"
    assert "<ruby><rb>妻</rb><rt>つま</rt></ruby>" not in answer


# --- the lookup links ----------------------------------------------------------


def test_both_lookup_queries_render_percent_encoded(
    drawn: list[dict[str, Any]],
) -> None:
    """The encoding change nobody had tapped. 話す is three bytes per character,
    and both links have to carry it as a query rather than as raw text."""
    answer = backs(drawn)[0]
    quoted = "%E8%A9%B1%E3%81%99"

    assert f'href="shirabelookup://search?w={quoted}"' in answer
    assert f'href="https://jpdb.io/search?q={quoted}&amp;lang=english"' in answer


# --- what a template must not leave behind ------------------------------------


def test_no_field_reference_survives_rendering(drawn: list[dict[str, Any]]) -> None:
    """A renamed or mistyped field renders as literal `{{Whatever}}` — visible
    on the card, invisible to a test that reads the template source, and
    invisible to the build, which never resolves a field name."""
    for card in drawn:
        for side in ("question", "answer"):
            assert "{{" not in card[side], (card["expression"], side)


def test_a_sound_tag_is_consumed_rather_than_shown() -> None:
    """Anki replaces `[sound:...]` with a play button. A tag that survives into
    the rendered card is one Anki did not recognise, and the learner reads the
    filename instead of hearing it."""
    drawn = render(
        [record(audio="audio/janki-abc.mp3")],
        {"audio/janki-abc.mp3": b"ID3fake"},
    )

    assert drawn[0]["sounds"] == ["janki-abc.mp3"], "Anki took it to play"
    assert "[sound:" not in drawn[0]["answer"], "and left no tag behind"


# --- conditional sections ------------------------------------------------------


def test_a_section_with_no_content_does_not_draw_its_heading() -> None:
    """`{{#CasualJapanese}}` guards a "Casually" heading. A record with no
    casual sentence must render no heading at all, not an empty one — which is
    what the template source cannot tell you."""
    plain = render([record()])

    assert "Casually" not in plain[0]["answer"]


def test_a_casual_sentence_draws_its_section() -> None:
    """The other direction, so the test above cannot pass by the section being
    broken for everyone."""
    casual = render([
        record(examples=[
            ExampleSentence(japanese="毎日話します。", register="polite"),
            ExampleSentence(
                japanese="毎日話すよ。", furigana="毎日[まいにち] 話[はな]すよ。",
                register="casual",
            ),
        ])
    ])

    answer = casual[0]["answer"]
    assert "Casually" in answer
    assert "毎日話すよ。" in answer


# --- the blocks that only exist once rendered ---------------------------------


def test_the_pitch_diagram_and_kanji_block_reach_the_drawn_card(
    drawn: list[dict[str, Any]],
) -> None:
    answer = backs(drawn)[0]

    assert 'class="pitch"' in answer
    assert re.search(r'class="mora [^"]*drop', answer), "the fall is drawn"
    # The largest block janki generates, and the one most able to break a card:
    # a `<details>` carrying inline SVG per stroke.
    assert 'class="kanji-info"' in answer
    assert "会話" in answer, "the reading's example word"
    assert answer.count('class="stroke-cell"') == 2, "one cell per stroke"


def test_a_failed_build_leaves_no_scratch_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """The build is what raises in this helper — a record that fails validation,
    a missing audio file — and it runs *before* the collection exists. A cleanup
    that only covers the render leaves behind a copy of the templates, a deck
    and a `kanji.json` per attempt, in a bare `mkdtemp` that pytest's retention
    policy never reaps. It leaks worst in the red-test loop, which re-runs this
    most."""
    made: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*args: Any, **kwargs: Any) -> str:
        path = real_mkdtemp(*args, **kwargs)
        made.append(Path(path))
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", spy)

    with pytest.raises(AnkiBuildError):
        render([record(meanings=[])])

    assert made, "the helper did make a scratch tree"
    assert not made[0].exists(), f"and left {made[0]} behind"
