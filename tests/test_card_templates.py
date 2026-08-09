"""Contracts shared by the plain HTML card-back templates."""

from __future__ import annotations

from pathlib import Path

TEMPLATES = Path(__file__).parents[1] / "templates" / "japanese-study"
BACKS = tuple(TEMPLATES.glob("*-back.html"))


def test_every_lookup_back_percent_encodes_the_jpdb_query() -> None:
    """HTML entity escaping is not URL encoding: A&B and A#B need JS's URL
    encoding after the browser has decoded the field's attribute value."""
    for path in BACKS:
        template = path.read_text(encoding="utf-8")
        assert 'class="jpdb-link" data-query="{{Expression}}"' in template, path.name
        assert 'href="https://jpdb.io/search?lang=english"' in template, path.name
        assert "?q={{Expression}}" not in template, path.name
        assert "encodeURIComponent(link.dataset.query)" in template, path.name


def test_the_empty_pitch_particle_has_a_line_box() -> None:
    css = (TEMPLATES / "style.css").read_text(encoding="utf-8")

    assert ".pitch-accent .mora.particle::before" in css
    assert 'content: "\\200B"' in css
