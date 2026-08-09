"""Contracts shared by the plain HTML card-back templates."""

from __future__ import annotations

from pathlib import Path

TEMPLATES = Path(__file__).parents[1] / "templates" / "japanese-study"
BACKS = tuple(sorted(TEMPLATES.glob("*-back.html")))


def test_there_are_three_back_templates() -> None:
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


def test_no_card_carries_javascript() -> None:
    """A stated design rule, and not only a stylistic one: AnkiWeb's reviewer
    strips template scripts, so a link assembled in JS became a silent jump to
    an empty search page there — failing with no error, which is worse than the
    plain href it replaced."""
    for path in BACKS:
        template = path.read_text(encoding="utf-8")
        assert "<script" not in template, path.name
        assert "encodeURIComponent" not in template, path.name
