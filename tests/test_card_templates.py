"""Contracts shared by the plain HTML card-back templates."""

from __future__ import annotations

from pathlib import Path

TEMPLATES = Path(__file__).parents[1] / "templates" / "japanese-study"
#: The *word* card backs. A pattern card is a rule with no word on it, so it
#: carries no dictionary lookup and is not held to that contract.
BACKS = tuple(
    path for path in sorted(TEMPLATES.glob("*-back.html"))
    if not path.name.startswith("pattern-")
)
#: Every card face, not only the backs, and pattern faces included. The
#: no-JavaScript rule is about what AnkiWeb will strip, and it strips a script
#: on a rule card just as silently.
ALL_FACES = tuple(sorted(TEMPLATES.glob("*.html")))


def test_there_are_three_word_card_backs() -> None:
    """Pinned, because every test below loops over this glob: rename the
    directory or the suffix and they all pass green having checked nothing,
    each one's name left standing for a thing no longer verified."""
    assert [path.name for path in BACKS] == [
        "production-back.html",
        "reading-back.html",
        "recognition-back.html",
    ]


def test_every_lookup_back_takes_its_query_already_encoded() -> None:
    """HTML entity escaping is not URL encoding, so the query cannot be the raw
    `{{Expression}}`: `Q&A` renders `?w=Q&amp;A`, the webview decodes it back to
    `?w=Q&A`, and the app receives `w=Q`. `ShirabeQuery` is percent-encoded at
    export time and serves both links."""
    for path in BACKS:
        template = path.read_text(encoding="utf-8")
        assert "shirabelookup://search?w={{ShirabeQuery}}" in template, path.name
        assert (
            "https://jpdb.io/search?q={{ShirabeQuery}}&amp;lang=english" in template
        ), path.name
        assert "?q={{Expression}}" not in template, path.name
        assert "?w={{Expression}}" not in template, path.name


def test_every_card_face_is_accounted_for() -> None:
    """Same reason as above, for the wider glob the no-JavaScript rule uses."""
    assert [path.name for path in ALL_FACES] == [
        "pattern-back.html",
        "pattern-front.html",
        "production-back.html",
        "production-front.html",
        "reading-back.html",
        "reading-front.html",
        "recognition-back.html",
        "recognition-front.html",
    ]


def test_no_card_face_carries_javascript() -> None:
    """A stated design rule, and not only a stylistic one: AnkiWeb's reviewer
    strips template scripts, so a link assembled in JS became a silent jump to
    an empty search page there — failing with no error, which is worse than the
    plain href it replaced. Over every face, since a front is stripped just as
    quietly as a back."""
    for path in ALL_FACES:
        template = path.read_text(encoding="utf-8")
        assert "<script" not in template, path.name
        assert "encodeURIComponent" not in template, path.name


def test_the_empty_pitch_particle_has_a_line_box() -> None:
    """The trailing particle slot carries no text, so without a zero-width
    space it has no line box and collapses — taking the overline that shows
    whether the pitch stays high after the word with it."""
    css = (TEMPLATES / "style.css").read_text(encoding="utf-8")

    assert ".pitch-accent .mora.particle::before" in css
    assert 'content: "\\200B"' in css


def test_every_field_a_template_names_exists_on_the_notetype() -> None:
    """A `{{Field}}` Anki cannot resolve does not degrade — it replaces *both
    sides of the whole card* with an error string. So a rename inside a
    `{{#Section}}` that a test fixture never fills is invisible until a learner
    meets a record that fills it, and then the card is gone rather than
    diminished.

    Checked at source level because the rendered harness can only cover the
    fields its fixtures happen to populate.

    Each face is held to *its own* notetype, not to the union of both. A union
    is satisfied by `{{Trigger}}` typed into a word-card back, or `{{KanjiInfo}}`
    into a rule card — each a field the notetype that template belongs to does
    not have, and so each one card-wide breakage the check would wave through."""
    import re

    from japanese_anki.exporters.anki import FIELD_NAMES

    # `FrontSide` is Anki's own, and `type:`/`furigana:`/`hint:` are filters
    # applied to a field named after the colon.
    from japanese_anki.exporters.pattern_cards import FIELDS as PATTERN_FIELDS

    # Taken from the exporters rather than restated, so a rename in either one
    # fails here instead of being blessed by a matching literal.
    for path in ALL_FACES:
        pattern_face = path.name.startswith("pattern-")
        known = {*(PATTERN_FIELDS if pattern_face else FIELD_NAMES), "FrontSide"}
        referenced: set[str] = set()
        for raw in re.findall(r"\{\{([^}]+)\}\}", path.read_text(encoding="utf-8")):
            name = raw.strip().lstrip("#^/").split(":")[-1].strip()
            if name:
                referenced.add(name)

        unknown = sorted(referenced - known)
        assert not unknown, f"{path.name} names field(s) its notetype lacks: {unknown}"
