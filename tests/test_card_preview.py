"""What `janki preview` actually draws.

`tests/test_rendered_cards.py` and `tests/test_rendered_kanji_cards.py` ask
Anki what the *templates* draw. This file asks the same question of the
*preview*: it renders through those same exporters and the same scratch
collection, so the thing under test is everything the preview adds around
them — which cards it selects, what it counts, what it inlines, what it
refuses, and whether the exported page is safe to open.

Nothing here reads Japanese. The assertions are about identities, counts, HTML
structure and bytes; the Japanese in the fixtures is ordinary Genki-level
material used as *input*, never judged.

The scratch collection is thrown away by the renderer itself; `conftest.py`
also guards the developer's real one globally.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import html as html_module
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("genanki")
pytest.importorskip(
    "anki.collection",
    reason="the `anki` library renders these; it is the optional preview extra",
)

from japanese_anki import card_preview, cli, kanji_notes, patterns  # noqa: E402
from japanese_anki.card_preview import (  # noqa: E402
    CardPreview,
    CardPreviewError,
    ProposedText,
    preview_unavailable,
    render_card_preview,
    write_card_preview,
)
from japanese_anki.config import ProjectConfig  # noqa: E402
from japanese_anki.identifiers import character_record_id  # noqa: E402
from japanese_anki.models import ExampleSentence, VocabularyRecord  # noqa: E402
from japanese_anki.patterns import Pattern, PatternSet  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- fixtures -----------------------------------------------------------------


def _project(root: Path) -> ProjectConfig:
    """A scratch repository holding real templates and nothing else.

    The whole `templates/` tree is copied, so the viewer assets a project
    carries are the ones the preview uses — the same precedence every other
    template here has.
    """
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n'
        'kanji_notes_file = "kanji_notes.json"\n'
        'patterns_file = "patterns.json"\n',
        encoding="utf-8",
    )
    shutil.copytree(PROJECT_ROOT / "templates", root / "templates")
    (root / "decks").mkdir()
    (root / "media").mkdir()
    (root / "vocabulary.json").write_text("[]", encoding="utf-8")
    return ProjectConfig.load(root)


def _words(root: Path, records: list[VocabularyRecord], name: str = "vocabulary.json") -> Path:
    path = root / name
    path.write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _word_deck(root: Path, body: str = "", name: str = "lesson.yaml") -> Path:
    path = root / "decks" / name
    path.write_text(
        "deck:\n"
        "  name: Lesson\n"
        '  source: "../vocabulary.json"\n'
        "  cards:\n"
        "    recognition: true\n"
        "    production: true\n"
        "    reading: true\n" + body,
        encoding="utf-8",
    )
    return path


HANASU = VocabularyRecord(
    id="word:話す:はなす",
    expression="話す",
    reading="はなす",
    furigana="話[はな]す",
    meanings=["to speak"],
    verb_group="godan",
    tags=["lesson"],
    examples=[
        ExampleSentence(
            japanese="毎日妻と話します。",
            furigana="毎日[まいにち] 妻[つま]と 話[はな]します。",
            english="I speak with my wife every day.",
            register="polite",
        )
    ],
)
MIRU = VocabularyRecord(
    id="word:見る:みる",
    expression="見る",
    reading="みる",
    furigana="見[み]る",
    meanings=["to see"],
    verb_group="ichidan",
    tags=["lesson"],
)
KAU = VocabularyRecord(
    id="word:買う:かう",
    expression="買う",
    reading="かう",
    furigana="買[か]う",
    meanings=["to buy"],
    verb_group="godan",
    tags=["lesson"],
)


def _character(character: str, **overrides: Any) -> kanji_notes.CharacterNote:
    values: dict[str, Any] = {
        "character": character,
        "id": character_record_id(character),
        "meanings": ("logic", "reason"),
        "stroke_count": 2,
        "strokes": ("M1,1L2,2", "M3,3L4,4"),
        "kanjidic_readings": (
            kanji_notes.KanjidicReading(kind="on", reading="リ"),
        ),
        "sources": ("kanjiapi.dev (KANJIDIC2)",),
    }
    values.update(overrides)
    return kanji_notes.CharacterNote(**values)


def _kanji_deck_text(characters: tuple[str, ...], source: str | None = None) -> str:
    section: dict[str, Any] = {
        "kind": "kanji",
        "name": "Week 2 Kanji",
        "deck_id": 1600000001,
        "model_id": 1600000002,
        "include_ids": [character_record_id(c) for c in characters],
    }
    if source is not None:
        section["source"] = source
    return yaml.safe_dump({"deck": section}, allow_unicode=True, sort_keys=False)


# --- the four deck kinds ------------------------------------------------------


@pytest.fixture(scope="module")
def word_preview() -> CardPreview:
    """One vocabulary render, reused: opening a collection is the slow part."""
    root = Path(tempfile.mkdtemp())
    try:
        config = _project(root)
        _words(root, [HANASU, MIRU])
        deck = _word_deck(root)
        return render_card_preview(config, deck)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_vocabulary_deck_draws_every_enabled_direction_exactly_once(
    word_preview: CardPreview,
) -> None:
    """The whole enabled card set, and no card drawn twice.

    A preview that quietly showed recognition only would be a review surface
    for a deck the owner is not building.
    """
    assert word_preview.deck_kind == "vocabulary"
    assert word_preview.deck_name == "Lesson"
    assert word_preview.directions == ("recognition", "production", "reading")

    drawn = Counter((card.record_id, card.direction) for card in word_preview.cards)

    assert set(drawn.values()) == {1}, drawn
    assert {direction for _id, direction in drawn} == set(word_preview.directions)
    assert word_preview.card_count == 6 == len(word_preview.cards)
    assert word_preview.note_count == 2


def test_the_vocabulary_identity_is_the_notes_own_record_id(
    word_preview: CardPreview,
) -> None:
    assert {card.record_id for card in word_preview.cards} == {
        "word:話す:はなす",
        "word:見る:みる",
    }
    # The label is the notetype's sort field, reduced to text. Structural: it
    # is the field Anki itself sorts on, not a choice made about the Japanese.
    speaking = next(c for c in word_preview.cards if c.record_id == "word:話す:はなす")
    assert speaking.label == "話す"
    assert speaking.template in {"Recognition", "Production", "Reading"}


def test_the_source_furigana_reaches_the_preview_as_ruby(
    word_preview: CardPreview,
) -> None:
    """Anki's `furigana:` filter ran, so the preview shows what a card shows.

    A preview built from raw fields would print `話[はな]す` at the learner.
    """
    answers = "".join(card.answer_html for card in word_preview.cards)

    assert "<ruby><rb>話</rb><rt>はな</rt></ruby>" in answers
    assert "<ruby><rb>妻</rb><rt>つま</rt></ruby>" in answers
    assert "話[はな]す" not in answers, "no bracket notation reaches the learner"
    assert "{{" not in answers, "and no template reference survives"


def test_the_question_asks_and_the_answer_answers(word_preview: CardPreview) -> None:
    recognition = next(
        card for card in word_preview.cards
        if card.record_id == "word:話す:はなす" and card.direction == "recognition"
    )

    assert "to speak" not in recognition.question_html
    assert "to speak" in recognition.answer_html
    assert recognition.question_html != recognition.answer_html


def test_a_character_deck_renders_its_own_notetype_and_disclosures(
    tmp_path: Path,
) -> None:
    """A kanji deck is a second notetype, not a word deck with a template swap.

    Its identities are `kanji:理`, and its answer keeps KANJIDIC's inventory
    behind the deck's own `<details>` — which the preview must leave native and
    closed rather than flattening into the page.
    """
    config = _project(tmp_path)
    (tmp_path / "kanji_notes.json").write_text(
        kanji_notes.render_notes({"理": _character("理")}), encoding="utf-8"
    )
    deck = tmp_path / "decks" / "kanji.yaml"
    deck.write_text(_kanji_deck_text(("理",), source="../kanji_notes.json"), encoding="utf-8")

    preview = render_card_preview(config, deck)

    assert preview.deck_kind == "kanji"
    assert preview.directions == ("recognition",)
    assert [card.record_id for card in preview.cards] == ["kanji:理"]
    assert preview.cards[0].label == "理"
    answer = preview.cards[0].answer_html
    assert '<details class="kanji-more kanji-inventory">' in answer
    assert "KANJIDIC readings" in answer
    assert "理" in preview.cards[0].question_html


def test_a_pattern_deck_uses_its_deterministic_structural_identities(
    tmp_path: Path,
) -> None:
    """A rule note carries no record id in any field; its GUID is the identity.

    Mapping that GUID back is what lets the preview name a rule card at all,
    and it is the same `pattern:<document>:<trigger>` the build mints.
    """
    config = _project(tmp_path)
    patterns.save_store(
        tmp_path / "patterns.json",
        {
            "teform.pdf": PatternSet(
                "teform.pdf",
                "pattern",
                "Te-form",
                (
                    Pattern("う・つ・る → って", "godan て-form"),
                    Pattern("くる → きて / する → して", "the irregulars"),
                ),
                True,
            )
        },
    )
    deck = tmp_path / "decks" / "rules.yaml"
    deck.write_text(
        "deck:\n"
        "  kind: pattern\n"
        "  name: Te-form rules\n"
        "  deck_id: 2059400113\n"
        "  model_id: 1607392351\n"
        '  document: "teform.pdf"\n',
        encoding="utf-8",
    )

    preview = render_card_preview(config, deck)

    assert preview.deck_kind == "pattern"
    assert [card.record_id for card in preview.cards] == [
        "pattern:teform.pdf:う・つ・る",
        "pattern:teform.pdf:くる",
        "pattern:teform.pdf:する",
    ]
    assert preview.note_count == 3 and preview.card_count == 3
    assert "って" in preview.cards[0].answer_html


def test_a_conjugation_deck_maps_its_drill_cards_back_to_source_records(
    tmp_path: Path,
) -> None:
    """A drill note's GUID is `drill:<form>:<record id>` and its fields hold
    neither. An API scope is stated in *vocabulary* ids, so the preview has to
    map the GUID back or a caller could never select one of these cards.
    """
    config = _project(tmp_path)
    _words(tmp_path, [HANASU, KAU])
    deck = tmp_path / "decks" / "drill.yaml"
    deck.write_text(
        "deck:\n"
        "  kind: conjugation\n"
        "  form: te_form\n"
        "  name: Te-form drill\n"
        "  deck_id: 2059400114\n"
        "  model_id: 1607392351\n"
        '  source: "../vocabulary.json"\n',
        encoding="utf-8",
    )

    preview = render_card_preview(
        config, deck, scope_record_ids=["word:買う:かう"]
    )

    assert preview.deck_kind == "conjugation"
    assert [card.record_id for card in preview.cards] == ["word:買う:かう"]
    assert preview.deck_note_count == 2, "both verbs are drilled by the deck"
    assert "買って" in preview.cards[0].answer_html


# --- counts, scope and new identities -----------------------------------------


def test_counts_distinguish_the_selection_new_ids_and_the_whole_deck(
    tmp_path: Path,
) -> None:
    """Three different questions, three different numbers.

    "How many cards am I looking at", "how many of these are additions", and
    "how big is the deck this lands in" are answered separately — and a card
    is marked new only when its identity was named as an addition. Marking
    every selected card new is how a review of two changes to a 40-card deck
    starts reading as a 40-card creation.
    """
    config = _project(tmp_path)
    _words(tmp_path, [HANASU, MIRU, KAU])
    deck = _word_deck(tmp_path)

    preview = render_card_preview(
        config,
        deck,
        scope_record_ids=["word:見る:みる", "word:話す:はなす"],
        new_record_ids=["word:見る:みる"],
    )

    assert preview.note_count == 2
    assert preview.card_count == 6
    assert preview.new_note_count == 1
    assert preview.existing_note_count == 1
    assert preview.deck_note_count == 3
    assert preview.deck_card_count == 9
    assert {card.record_id for card in preview.cards if card.is_new} == {
        "word:見る:みる"
    }
    assert {card.record_id for card in preview.cards if not card.is_new} == {
        "word:話す:はなす"
    }
    page = preview.html.decode("utf-8")
    assert "showing 2 notes · 6 cards" in page
    assert "whole deck 3 notes · 9 cards" in page
    assert "1 new" in page


def test_a_scope_keeps_the_order_it_was_given(tmp_path: Path) -> None:
    config = _project(tmp_path)
    _words(tmp_path, [HANASU, MIRU, KAU])
    deck = _word_deck(tmp_path)

    preview = render_card_preview(
        config, deck, scope_record_ids=["word:買う:かう", "word:話す:はなす"]
    )

    assert [card.record_id for card in preview.cards][:3] == ["word:買う:かう"] * 3
    assert [card.record_id for card in preview.cards][3:] == ["word:話す:はなす"] * 3


@pytest.mark.parametrize(
    ("scope", "message"),
    [
        (["word:missing:missing"], "not among the cards this deck builds"),
        (["word:話す:はなす", "word:話す:はなす"], "is repeated"),
        ([], "at least one card ID"),
        (["  "], "nonblank strings"),
        ("word:話す:はなす", "not text"),
    ],
    ids=["unknown", "duplicate", "empty", "blank", "text"],
)
def test_an_unusable_scope_refuses_rather_than_showing_a_different_set(
    tmp_path: Path, scope: object, message: str
) -> None:
    config = _project(tmp_path)
    _words(tmp_path, [HANASU, MIRU])
    deck = _word_deck(tmp_path)

    with pytest.raises(CardPreviewError, match=message):
        render_card_preview(config, deck, scope_record_ids=scope)  # type: ignore[arg-type]


def test_a_new_id_the_deck_does_not_build_refuses(tmp_path: Path) -> None:
    """`new_record_ids` names actual additions. An id the deck never builds
    cannot be one of them, and counting it would inflate the new count against
    cards nobody can see."""
    config = _project(tmp_path)
    _words(tmp_path, [HANASU])
    deck = _word_deck(tmp_path)

    with pytest.raises(CardPreviewError, match="cannot count it as an addition"):
        render_card_preview(config, deck, new_record_ids=["word:nope:nope"])


def test_more_cards_than_the_limit_refuses_instead_of_truncating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _project(tmp_path)
    _words(tmp_path, [HANASU, MIRU])
    deck = _word_deck(tmp_path)
    monkeypatch.setattr(card_preview, "MAX_PREVIEW_CARDS", 3)

    with pytest.raises(CardPreviewError, match="over the 3-card preview limit"):
        render_card_preview(config, deck)


# --- proposed content ---------------------------------------------------------


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_proposed_text_renders_without_applying_or_touching_canonical_bytes(
    tmp_path: Path,
) -> None:
    """The whole point of a proposal preview: see the card before it exists.

    The deck reads an explicit *alternate* source, not the configured
    collection, and carries an inline override and a tag filter. All three have
    to survive the snapshot, or the preview shows cards this deck would not
    build.
    """
    config = _project(tmp_path)
    _words(tmp_path, [MIRU])  # the configured collection, deliberately not used
    alternate = _words(tmp_path, [HANASU], name="alternate.json")
    deck = tmp_path / "decks" / "lesson.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Lesson\n"
        '  source: "../alternate.json"\n'
        "  include_tags: [lesson]\n"
        "  cards:\n"
        "    recognition: true\n"
        "notes:\n"
        "  - id: word:話す:はなす\n"
        "    usage_notes: an inline override the deck keeps\n",
        encoding="utf-8",
    )
    before = _tree(tmp_path)
    proposed_records = [
        VocabularyRecord(
            id="word:話す:はなす",
            expression="話す",
            reading="はなす",
            furigana="話[はな]す",
            meanings=["to speak, to talk"],
            tags=["lesson"],
        ),
        VocabularyRecord(
            id="word:買う:かう",
            expression="買う",
            reading="かう",
            furigana="買[か]う",
            meanings=["to buy"],
            tags=["lesson"],
        ),
        VocabularyRecord(
            id="word:走る:はしる",
            expression="走る",
            reading="はしる",
            meanings=["to run"],
            tags=["another-deck"],
        ),
    ]

    preview = render_card_preview(
        config,
        deck,
        proposed=[
            ProposedText(
                path=alternate,
                text=json.dumps(
                    [record.to_dict() for record in proposed_records],
                    ensure_ascii=False,
                ),
            )
        ],
        new_record_ids=["word:買う:かう"],
    )

    assert _tree(tmp_path) == before, "a preview writes nothing canonical"
    assert {card.record_id for card in preview.cards} == {
        "word:話す:はなす",
        "word:買う:かう",
    }, "the deck's own include_tags filter still applies"
    answers = "".join(card.answer_html for card in preview.cards)
    assert "to speak, to talk" in answers, "the proposed meaning is what renders"
    assert "an inline override the deck keeps" in answers
    assert preview.new_note_count == 1 and preview.existing_note_count == 1


def test_a_proposed_deck_and_store_can_render_before_either_exists(
    tmp_path: Path,
) -> None:
    """The prepared-batch case: neither file is on disk yet.

    The deck names no `source:`, so its store comes from the project config —
    which the snapshot repoints at the proposed bytes rather than at the
    canonical file the batch has not written.
    """
    config = _project(tmp_path)
    deck = tmp_path / "decks" / "week-2-kanji.yaml"
    store_text = kanji_notes.render_notes(
        {"理": _character("理"), "料": _character("料", meanings=("fee",))}
    )

    preview = render_card_preview(
        config,
        deck,
        proposed=[
            ProposedText(path=deck, text=_kanji_deck_text(("理", "料"))),
            ProposedText(path=tmp_path / "kanji_notes.json", text=store_text),
        ],
        new_record_ids=["kanji:理", "kanji:料"],
        subtitle="Proposed batch — nothing written yet",
    )

    assert not deck.exists() and not (tmp_path / "kanji_notes.json").exists()
    assert sorted(card.record_id for card in preview.cards) == ["kanji:料", "kanji:理"]
    assert preview.new_note_count == 2 and preview.existing_note_count == 0
    assert "Proposed batch — nothing written yet" in preview.html.decode("utf-8")


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda root: ProposedText(path=Path("decks/lesson.yaml"), text="x"),
            "must be absolute",
        ),
        (
            lambda root: ProposedText(path=root.parent / "outside.json", text="x"),
            "must be inside the project",
        ),
        (
            lambda root: ProposedText(path=root / "unrelated.json", text="x"),
            "does not read",
        ),
    ],
    ids=["relative", "outside", "unread"],
)
def test_an_unusable_proposed_path_refuses(
    tmp_path: Path, factory: Any, message: str
) -> None:
    config = _project(tmp_path)
    _words(tmp_path, [HANASU])
    deck = _word_deck(tmp_path)

    with pytest.raises(CardPreviewError, match=message):
        render_card_preview(config, deck, proposed=[factory(tmp_path)])


def test_the_same_proposed_path_twice_refuses(tmp_path: Path) -> None:
    """Two texts for one file is a caller that does not know what it proposes;
    silently keeping the last one would preview bytes nobody chose."""
    config = _project(tmp_path)
    source = _words(tmp_path, [HANASU])
    deck = _word_deck(tmp_path)

    with pytest.raises(CardPreviewError, match="more than once"):
        render_card_preview(
            config,
            deck,
            proposed=[
                ProposedText(path=source, text="[]"),
                ProposedText(path=source, text="[]"),
            ],
        )


def test_a_failed_render_leaves_no_scratch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The build is what raises here, before any collection exists.

    A cleanup that only covered the render would leave a package and a mirrored
    snapshot behind per attempt, in a directory pytest's retention policy never
    reaps — worst in exactly the red-test loop that runs this most.
    """
    made: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def spy(*args: Any, **kwargs: Any) -> str:
        path = real_mkdtemp(*args, **kwargs)
        if Path(path).name.startswith("janki-card-preview-"):
            made.append(Path(path))
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", spy)
    config = _project(tmp_path)
    _words(tmp_path, [VocabularyRecord(id="word:x:x", expression="x", reading="x")])
    deck = _word_deck(tmp_path)

    with pytest.raises(Exception):  # noqa: B017 - the exporter's own refusal
        render_card_preview(config, deck)

    assert made, "the renderer did make a scratch tree"
    assert not made[0].exists(), f"and left {made[0]} behind"


# --- the exported page --------------------------------------------------------


def _viewer_asset(name: str) -> str:
    return (PROJECT_ROOT / "templates" / "card-preview" / name).read_text(
        encoding="utf-8"
    )


def test_the_viewer_opens_on_the_question_with_every_answer_hidden(
    word_preview: CardPreview,
) -> None:
    """The reviewer's own order, in the bytes rather than only in the script.

    A page whose answers were visible until JavaScript ran would give the
    answer away on every slow load — and to anyone who opened it with scripting
    off, which is exactly what the strict policy encourages.
    """
    page = word_preview.html.decode("utf-8")

    assert page.count("data-preview-answer hidden>") == word_preview.card_count
    assert "data-preview-answer>" not in page, "no answer starts visible"
    # The first card is the one on screen; the rest start hidden.
    assert page.count('data-preview-card data-index="0" ') == 1
    assert page.count(" hidden>") >= word_preview.card_count
    assert ">Show Answer</button>" in page


def test_every_rendered_card_has_navigation(word_preview: CardPreview) -> None:
    """Prev/Next alone reaches every card, and the jump list names each one, so
    a review of card 9 of 12 is one action rather than eight."""
    page = word_preview.html.decode("utf-8")

    assert page.count('<option value=') == word_preview.card_count
    for index in range(word_preview.card_count):
        assert f'<option value="{index}">' in page
    assert 'id="preview-prev"' in page and 'id="preview-next"' in page
    assert 'id="preview-jump"' in page and 'id="preview-flip"' in page
    assert "prefers-color-scheme: dark" in page, "readable in dark mode"
    assert 'name="viewport"' in page, "and on a phone"


def test_the_viewer_script_is_the_static_asset_and_its_policy_hashes_it(
    word_preview: CardPreview,
) -> None:
    """Hash-authorised, so `script-src` needs no `unsafe-inline` — which is the
    same setting that denies an inline event handler in untrusted card content
    any authority to run. The policy and these exact bytes are one artifact:
    serving them apart breaks the viewer."""
    script = _viewer_asset("viewer.js")
    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode()
    page = word_preview.html.decode("utf-8")

    assert f"'sha256-{digest}'" in word_preview.content_security_policy
    assert f"<script>{script}</script>" in page
    assert (
        html_module.escape(word_preview.content_security_policy, quote=True) in page
    ), "the exported page carries its own policy"
    assert word_preview.sha256 == hashlib.sha256(word_preview.html).hexdigest()


def test_the_policy_denies_remote_loading_forms_objects_and_a_base(
    word_preview: CardPreview,
) -> None:
    policy = word_preview.content_security_policy
    script_src = re.search(r"script-src ([^;]*)", policy).group(1)

    assert policy.startswith("default-src 'none';")
    assert "unsafe-inline" not in script_src and "unsafe-eval" not in script_src
    assert "unsafe-hashes" not in script_src
    assert "img-src data:;" in policy and "media-src data:;" in policy
    assert "form-action 'none'" in policy
    assert "object-src 'none'" in policy
    assert "base-uri 'none'" in policy
    assert "http://" not in policy and "https://" not in policy


def test_untrusted_card_text_reaches_the_page_inert(tmp_path: Path) -> None:
    """A record is data. Text that looks like markup stays text on the card,
    and stays text in the preview — the preview adds no way for it to run that
    the card did not already have, and the policy above removes several."""
    config = _project(tmp_path)
    _words(
        tmp_path,
        [
            VocabularyRecord(
                id="word:試す:ためす",
                expression="試す",
                reading="ためす",
                meanings=["<script>alert(1)</script>", '<img src="https://evil.example/p.png">'],
                usage_notes="<b onclick=\"steal()\">click</b>",
                tags=["lesson"],
            )
        ],
    )
    deck = _word_deck(tmp_path)

    preview = render_card_preview(config, deck)
    page = preview.html.decode("utf-8")
    body = page.split("</head>", 1)[1]

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "<script>alert(1)</script>" not in body
    # Shown as the text it is. What must not exist is a *live* element built
    # from it, so the assertion is on the tag rather than on the string.
    assert "&lt;img src=&quot;https://evil.example/p.png&quot;&gt;" in body
    assert '<img src="https://evil.example/p.png">' not in body
    assert "&lt;b onclick=&quot;steal()&quot;&gt;" in body
    assert "<b onclick=" not in body
    subresources = re.findall(
        r'<(?:img|audio|video|source|script|link|iframe|object|embed)\b'
        r'[^>]*?\b(?:src|href|data)\s*=\s*"([^"]*)"',
        body,
        re.IGNORECASE,
    )
    assert all(value.startswith("data:") for value in subresources), subresources
    # Exactly one script element, and it is the hashed viewer.
    assert body.count("<script>") == 1 and body.count("<script ") == 0


def test_every_image_and_audio_source_in_the_page_is_already_inline(
    tmp_path: Path,
) -> None:
    config = _project(tmp_path)
    (tmp_path / "media" / "janki-word.mp3").write_bytes(b"ID3fake")
    (tmp_path / "media" / "janki-pic.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    _words(
        tmp_path,
        [
            VocabularyRecord(
                id="word:話す:はなす",
                expression="話す",
                reading="はなす",
                meanings=["to speak"],
                audio="janki-word.mp3",
                image="janki-pic.png",
                tags=["lesson"],
            )
        ],
    )
    deck = _word_deck(tmp_path)

    preview = render_card_preview(config, deck)
    body = preview.html.decode("utf-8").split("</head>", 1)[1]

    sources = re.findall(r'<(?:img|audio|source)\b[^>]*\bsrc="([^"]*)"', body)
    assert sources, "the card really does carry media"
    assert all(source.startswith("data:") for source in sources), sources
    assert "data:audio/mpeg;base64," in body
    assert "data:image/png;base64," in body


def test_media_janki_could_not_package_says_so_instead_of_offering_a_control(
    tmp_path: Path,
) -> None:
    """A hand-written `[sound:]` tag packages no file, so the card is silent.

    A play button beside it would be the preview asserting audio that does not
    exist — the one thing a review surface must never do about media.
    """
    config = _project(tmp_path)
    (tmp_path / "media" / "janki-word.mp3").write_bytes(b"ID3fake")
    _words(
        tmp_path,
        [
            VocabularyRecord(
                id="word:話す:はなす", expression="話す", reading="はなす",
                meanings=["to speak"], audio="janki-word.mp3", tags=["lesson"],
            ),
            VocabularyRecord(
                id="word:見る:みる", expression="見る", reading="みる",
                meanings=["to see"], audio="[sound:by-hand.mp3]", tags=["lesson"],
            ),
        ],
    )
    deck = _word_deck(tmp_path)

    preview = render_card_preview(config, deck)
    packaged = "".join(
        card.answer_html for card in preview.cards if card.record_id == "word:話す:はなす"
    )
    hand_written = "".join(
        card.answer_html for card in preview.cards if card.record_id == "word:見る:みる"
    )

    # The working clip is a control and names nothing; only the absent one has
    # to say which file it is.
    assert "<audio controls" in packaged and "janki-word.mp3" not in packaged
    assert "<audio" not in hand_written, "nothing pretends to play a missing clip"
    assert "Audio not packaged: by-hand.mp3" in hand_written
    assert not re.search(
        r"<audio[^>]*\bautoplay", preview.html.decode("utf-8")
    ), "a preview never starts playing on its own"


def test_media_over_the_per_file_cap_refuses_rather_than_dropping_the_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _project(tmp_path)
    (tmp_path / "media" / "janki-word.mp3").write_bytes(b"ID3" * 100)
    _words(
        tmp_path,
        [
            VocabularyRecord(
                id="word:話す:はなす", expression="話す", reading="はなす",
                meanings=["to speak"], audio="janki-word.mp3", tags=["lesson"],
            )
        ],
    )
    deck = _word_deck(tmp_path)
    monkeypatch.setattr(card_preview, "MAX_MEDIA_FILE_BYTES", 8)

    with pytest.raises(CardPreviewError, match="per-file preview limit"):
        render_card_preview(config, deck)


def test_selecting_another_card_returns_the_viewer_to_its_question() -> None:
    """Moving to a card shows that card's question, never the last flip state.

    Asserted against the static asset because pytest has no browser; the
    initial-state and script-hash checks above are what the exported bytes
    prove, and the interaction itself is proven in the browser gate. Deleting
    the reset from `select` fails this.
    """
    script = _viewer_asset("viewer.js")
    body = script.split("function select(", 1)[1].split("\n  }", 1)[0]

    assert "showingAnswer = false;" in body
    assert "jump.addEventListener(\"change\"" in script
    assert "select(index - 1)" in script and "select(index + 1)" in script


def test_a_project_without_viewer_assets_falls_back_to_the_checkout(
    tmp_path: Path,
) -> None:
    """The viewer is janki's tool, not the deck's content.

    A scratch project that copied only its notetype templates still gets a
    working preview, which is what every fixture in this repository looks like.
    """
    config = _project(tmp_path)
    shutil.rmtree(tmp_path / "templates" / "card-preview")
    _words(tmp_path, [MIRU])
    deck = _word_deck(tmp_path)

    preview = render_card_preview(config, deck)

    assert f"<script>{_viewer_asset('viewer.js')}</script>" in preview.html.decode(
        "utf-8"
    )


def test_write_card_preview_writes_exactly_the_rendered_bytes(
    tmp_path: Path, word_preview: CardPreview
) -> None:
    target = tmp_path / "nested" / "preview.html"

    written = write_card_preview(word_preview, target)

    assert written == target.resolve()
    assert written.read_bytes() == word_preview.html


# --- the CLI ------------------------------------------------------------------


def test_the_cli_writes_a_preview_and_reports_both_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path)
    _words(tmp_path, [HANASU, MIRU])
    deck = _word_deck(tmp_path)

    code = cli.main(
        ["--root", str(tmp_path), "preview", str(deck), "--record-id", "word:見る:みる"]
    )

    out = capsys.readouterr().out
    written = tmp_path / "dist" / "lesson-preview.html"
    assert code == 0
    assert written.exists() and written.read_bytes().startswith(b"<!doctype html>")
    assert f"Wrote preview to {written}" in out
    assert "3 card(s) from 1 note(s) in Lesson (vocabulary;" in out
    assert "The whole deck holds 2 note(s) and 6 card(s)." in out


def test_the_cli_refuses_an_id_the_deck_does_not_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path)
    _words(tmp_path, [HANASU])
    deck = _word_deck(tmp_path)

    code = cli.main(
        ["--root", str(tmp_path), "preview", str(deck), "--record-id", "word:no:no"]
    )

    assert code == 1
    assert "not among the cards this deck builds" in capsys.readouterr().err
    assert not (tmp_path / "dist" / "lesson-preview.html").exists()


def test_the_preview_command_documents_its_scope_and_its_extra(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["preview", "--help"])

    out = capsys.readouterr().out
    assert raised.value.code == 0
    assert "--record-id" in out
    assert "'preview' extra" in out
    assert cli.build_parser().parse_args(["preview", "d.yaml"]).record_id is None


def test_the_ordinary_cli_never_imports_anki_to_answer_availability() -> None:
    """The base install has no Anki, and importing it costs seconds and a Rust
    extension. `preview_unavailable()` is the thing every other surface calls to
    find out — so it must answer from the module spec alone."""
    script = (
        "import sys\n"
        "import japanese_anki.cli\n"
        "from japanese_anki.card_preview import preview_unavailable\n"
        "assert preview_unavailable() is None, preview_unavailable()\n"
        "loaded = sorted(m for m in sys.modules if m == 'anki' or m.startswith('anki.'))\n"
        "assert not loaded, loaded\n"
        "print('lazy')\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lazy"


def test_preview_unavailable_names_the_extra_when_anki_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "anki" else real(name, *a, **k),
    )

    message = preview_unavailable()

    assert message is not None
    assert "anki" in message and ".[preview]" in message


# --- one visible side, in a real browser ---------------------------------------


def _launch_installed_chrome(playwright: Any) -> Any:
    """Use an installed browser without letting a test download one.

    Absence is a supported skip. A browser that is installed and will not
    launch is a failed browser check, not another spelling of absence — the
    same stance `tests/test_workbench_browser.py` takes.
    """
    chromium = playwright.chromium
    candidates: list[Path] = []
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    known = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    ]
    # No Playwright-installed browser is an ordinary absence, not an error.
    with contextlib.suppress(Exception):
        known.append(Path(chromium.executable_path))
    candidates.extend(path for path in known if path.is_file())
    installed = tuple(dict.fromkeys(path.resolve() for path in candidates))
    if not installed:
        pytest.skip("no installed Chrome/Chromium for the preview viewer check")
    failures = []
    for executable in installed:
        try:
            return chromium.launch(headless=True, executable_path=str(executable))
        except Exception as exc:  # noqa: BLE001 - reported below by name
            failures.append(f"{executable}: {exc}")
    pytest.fail("installed Chrome/Chromium could not launch:\n" + "\n".join(failures))


def test_the_browser_shows_exactly_one_card_side_at_a_time(tmp_path: Path) -> None:
    """A reviewer sees the question or the answer, never both stacked.

    Only `answer.hidden` used to move, so Show Answer left the question sitting
    above its own answer — the card read as already answered, which is the one
    thing a review surface must not do. Asserted in a real browser because it
    is the browser that decides what `hidden` means.
    """
    sync_api = pytest.importorskip("playwright.sync_api")

    config = _project(tmp_path)
    (tmp_path / "kanji_notes.json").write_text(
        kanji_notes.render_notes(
            {"理": _character("理"), "料": _character("料", meanings=("fee",))}
        ),
        encoding="utf-8",
    )
    deck = tmp_path / "decks" / "kanji.yaml"
    deck.write_text(
        _kanji_deck_text(("理", "料"), source="../kanji_notes.json"), encoding="utf-8"
    )
    page_path = write_card_preview(
        render_card_preview(config, deck), tmp_path / "preview.html"
    )

    with sync_api.sync_playwright() as playwright:
        browser = _launch_installed_chrome(playwright)
        try:
            page = browser.new_page()
            page.goto(page_path.as_uri())
            page.wait_for_selector("body[data-preview-ready='true']")
            cards = page.locator("[data-preview-card]")
            first = cards.nth(0)
            question = first.locator("[data-preview-question]")
            answer = first.locator("[data-preview-answer]")

            assert question.is_visible() and not answer.is_visible()

            page.click("#preview-flip")
            assert answer.is_visible(), "Show Answer reveals the answer"
            assert not question.is_visible(), "and puts the question away"

            # A native disclosure on the answer opens on its own, and opening it
            # changes neither side's visibility.
            disclosure = first.locator("details.kanji-inventory").first
            disclosure.locator("summary").click()
            assert disclosure.evaluate("node => node.open") is True
            assert answer.is_visible() and not question.is_visible()

            page.click("#preview-flip")
            assert question.is_visible() and not answer.is_visible()

            page.click("#preview-flip")
            page.click("#preview-next")
            second = cards.nth(1)
            assert second.locator("[data-preview-question]").is_visible()
            assert not second.locator("[data-preview-answer]").is_visible()
            assert not first.is_visible(), "one card at a time as well"
        finally:
            browser.close()


# --- content, not implementation identifiers -----------------------------------


def _visible_text(page: str) -> str:
    """What a reader actually sees: no script, no style, no markup."""
    body = page.split("</head>", 1)[1]
    body = re.sub(r"<script>.*?</script>", " ", body, flags=re.S)
    body = re.sub(r"<style>.*?</style>", " ", body, flags=re.S)
    return " ".join(html_module.unescape(re.sub(r"<[^>]*>", " ", body)).split())


def test_a_card_is_labelled_by_its_template_not_by_its_internal_identity(
    word_preview: CardPreview,
) -> None:
    """The badge row is for the reader, not for the renderer.

    A record id and a direction that only restates the template name are
    implementation detail on the face of a study card. They stay in the
    structural attributes, where scope and selection already read them.
    """
    page = word_preview.html.decode("utf-8")
    meta = re.findall(r'<div class="preview-card-meta">(.*?)</div>', page)

    assert len(meta) == word_preview.card_count
    for block in meta:
        assert block.count("preview-badge") == 1, block
        assert "word:" not in block
    visible = _visible_text(page)
    assert "word:話す:はなす" not in visible and "word:見る:みる" not in visible
    assert 'data-record-id="word:話す:はなす"' in page, "structural scope is kept"
    assert 'data-direction="recognition"' in page
    assert "Recognition" in visible, "the one label a reader needs"


def test_a_working_audio_control_shows_no_filename_and_a_missing_one_still_does(
    tmp_path: Path,
) -> None:
    """An identity-addressed clip name is a hash, and a hash is not content.

    A control that plays says so by being a control; a clip janki could not
    package has to name itself, because that name is the only way to find it.
    """
    config = _project(tmp_path)
    (tmp_path / "media" / "janki-a1b2c3d4.mp3").write_bytes(b"ID3fake")
    _words(
        tmp_path,
        [
            VocabularyRecord(
                id="word:話す:はなす", expression="話す", reading="はなす",
                meanings=["to speak"], audio="janki-a1b2c3d4.mp3", tags=["lesson"],
            ),
            VocabularyRecord(
                id="word:見る:みる", expression="見る", reading="みる",
                meanings=["to see"], audio="[sound:by-hand.mp3]", tags=["lesson"],
            ),
        ],
    )
    deck = _word_deck(tmp_path)

    preview = render_card_preview(config, deck)
    page = preview.html.decode("utf-8")
    visible = _visible_text(page)

    assert "<audio controls" in page
    assert 'class="preview-audio-name"' not in page
    assert "janki-a1b2c3d4.mp3" not in visible
    assert "Audio not packaged: by-hand.mp3" in visible


# --- an absolute alternate source ----------------------------------------------


def test_a_proposed_overlay_preserves_an_absolute_deck_source(tmp_path: Path) -> None:
    """A deck may name its collection by absolute path, and a snapshot moves
    the deck. Only that one reference is rebound, to the snapshot the overlay
    produced; the selectors, the inline override and every canonical byte are
    exactly what they were."""
    config = _project(tmp_path)
    alternate = _words(tmp_path, [HANASU], name="alternate.json")
    deck = tmp_path / "decks" / "lesson.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Lesson\n"
        f'  source: "{alternate}"\n'
        "  include_tags: [lesson]\n"
        "  cards:\n"
        "    recognition: true\n"
        "notes:\n"
        "  - id: word:話す:はなす\n"
        "    usage_notes: an inline override the deck keeps\n",
        encoding="utf-8",
    )
    before = _tree(tmp_path)

    preview = render_card_preview(
        config,
        deck,
        proposed=[
            ProposedText(
                path=alternate,
                text=json.dumps(
                    [
                        {**HANASU.to_dict(), "meanings": ["to speak, to talk"]},
                        {**KAU.to_dict()},
                        {**MIRU.to_dict(), "tags": ["another-deck"]},
                    ],
                    ensure_ascii=False,
                ),
            )
        ],
    )

    assert _tree(tmp_path) == before, "a preview writes nothing canonical"
    assert {card.record_id for card in preview.cards} == {
        "word:話す:はなす",
        "word:買う:かう",
    }, "the deck's own include_tags filter still applies"
    answers = "".join(card.answer_html for card in preview.cards)
    assert "to speak, to talk" in answers
    assert "an inline override the deck keeps" in answers


# --- deck-local fallback media --------------------------------------------------


def test_a_proposed_overlay_keeps_media_the_deck_names_beside_itself(
    tmp_path: Path,
) -> None:
    """The exporter looks in the configured media directory, then beside the
    deck. A snapshot moves the deck, so the second half of that rule has to
    move with it — otherwise a proposal loses a clip the same deck plays today.
    The configured directory still wins where both exist."""
    config = _project(tmp_path)
    (tmp_path / "decks" / "audio").mkdir()
    (tmp_path / "decks" / "audio" / "local.mp3").write_bytes(b"LOCALCLIP")
    (tmp_path / "media" / "shared.mp3").write_bytes(b"CONFIGURED")
    # A decoy beside the deck under the *same* name: the configured directory
    # is tried first, so these bytes must never reach a card.
    (tmp_path / "decks" / "shared.mp3").write_bytes(b"DECOYDECOY")
    source = _words(tmp_path, [MIRU])
    deck = _word_deck(tmp_path)

    preview = render_card_preview(
        config,
        deck,
        proposed=[
            ProposedText(
                path=source,
                text=json.dumps(
                    [
                        {**HANASU.to_dict(), "audio": "audio/local.mp3"},
                        {**MIRU.to_dict(), "audio": "shared.mp3"},
                    ],
                    ensure_ascii=False,
                ),
            )
        ],
    )

    page = preview.html.decode("utf-8")
    local = base64.b64encode(b"LOCALCLIP").decode("ascii")
    configured = base64.b64encode(b"CONFIGURED").decode("ascii")
    decoy = base64.b64encode(b"DECOYDECOY").decode("ascii")
    assert f"base64,{local}" in page, "the clip beside the deck still plays"
    assert f"base64,{configured}" in page, "and the configured one still wins"
    assert decoy not in page
    assert "Audio not packaged" not in page


# --- caps belong to the selection ----------------------------------------------


def test_the_card_cap_applies_to_the_selection_not_the_whole_deck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A display cap is about the page, so it is about what the page shows.

    Applied to every card the deck builds, an exact two-card review of a large
    deck refused for a size nobody asked to see. The whole-deck totals stay
    exact, because they are the other question the header answers.
    """
    config = _project(tmp_path)
    _words(tmp_path, [HANASU, MIRU, KAU])
    deck = _word_deck(tmp_path)
    monkeypatch.setattr(card_preview, "MAX_PREVIEW_CARDS", 3)

    with pytest.raises(CardPreviewError, match="over the 3-card preview limit"):
        render_card_preview(config, deck)

    preview = render_card_preview(config, deck, scope_record_ids=["word:見る:みる"])

    assert preview.card_count == 3 and preview.note_count == 1
    assert preview.deck_note_count == 3 and preview.deck_card_count == 9


def test_media_outside_the_selection_is_neither_read_nor_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversized clip on a card nobody asked for cannot refuse the page.

    The selected card's own clip still refuses accurately when it is the one
    over the limit — the cap is enforced, just against what is being shown.
    """
    config = _project(tmp_path)
    (tmp_path / "media" / "small.mp3").write_bytes(b"ID3")
    (tmp_path / "media" / "large.mp3").write_bytes(b"ID3" * 200)
    _words(
        tmp_path,
        [
            VocabularyRecord(
                id="word:話す:はなす", expression="話す", reading="はなす",
                meanings=["to speak"], audio="small.mp3", tags=["lesson"],
            ),
            VocabularyRecord(
                id="word:見る:みる", expression="見る", reading="みる",
                meanings=["to see"], audio="large.mp3", tags=["lesson"],
            ),
        ],
    )
    deck = _word_deck(tmp_path)
    monkeypatch.setattr(card_preview, "MAX_MEDIA_FILE_BYTES", 32)

    preview = render_card_preview(
        config, deck, scope_record_ids=["word:話す:はなす"]
    )

    assert "<audio controls" in preview.html.decode("utf-8")
    assert preview.deck_note_count == 2, "the whole-deck count is still exact"

    with pytest.raises(CardPreviewError, match="per-file preview limit"):
        render_card_preview(config, deck, scope_record_ids=["word:見る:みる"])


def test_a_proposed_overlay_keeps_media_a_deck_names_beside_its_parent(
    tmp_path: Path,
) -> None:
    """A nested deck may reach up to a sibling directory for its clips.

    ``../clips/foo.wav`` from ``decks/nested/`` is an ordinary deck-relative
    path, and the configured candidate for it — ``media/../clips/foo.wav`` —
    does not exist, which is asserted below so a primary-media hit cannot stand
    in for the fallback. The snapshot moves the deck, so the link that keeps
    that fallback alive lands beside the *snapshot's* copy of the deck rather
    than under the deck's own directory; containment is the whole snapshot.
    """
    config = _project(tmp_path)
    (tmp_path / "decks" / "nested").mkdir()
    (tmp_path / "decks" / "clips").mkdir()
    (tmp_path / "decks" / "clips" / "foo.wav").write_bytes(b"SIBLINGCLIP")
    assert not (tmp_path / "clips").exists(), (
        "the configured media candidate for this value must be absent, or a "
        "passing preview proves nothing about the deck-relative fallback"
    )
    source = tmp_path / "vocabulary.json"
    source.write_text(
        json.dumps(
            [{**HANASU.to_dict(), "audio": "../clips/foo.wav"}], ensure_ascii=False
        ),
        encoding="utf-8",
    )
    deck = tmp_path / "decks" / "nested" / "d.yaml"
    deck.write_text(
        "deck:\n"
        "  name: Nested\n"
        '  source: "../../vocabulary.json"\n'
        "  cards:\n"
        "    recognition: true\n",
        encoding="utf-8",
    )
    before = _tree(tmp_path)
    clip = base64.b64encode(b"SIBLINGCLIP").decode("ascii")

    canonical = render_card_preview(config, deck).html.decode("utf-8")

    assert f"base64,{clip}" in canonical
    assert _tree(tmp_path) == before

    proposed = render_card_preview(
        config,
        deck,
        proposed=[
            ProposedText(
                path=source,
                text=json.dumps(
                    [
                        {
                            **HANASU.to_dict(),
                            "audio": "../clips/foo.wav",
                            "meanings": ["to speak, to talk"],
                        }
                    ],
                    ensure_ascii=False,
                ),
            )
        ],
    ).html.decode("utf-8")

    assert "to speak, to talk" in proposed, "the overlay really is in force"
    assert f"base64,{clip}" in proposed, "and the same clip is still playable"
    assert "Audio not packaged" not in proposed
    assert _tree(tmp_path) == before
