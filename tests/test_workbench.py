"""W1.2: the read-only workbench dashboard and the boundary it sits behind.

Two things get proved here. The **boundary** — loopback origin, session token,
response headers, no write route — because the page renders private study
material on a port every process on this machine can reach. And the **honesty**
of what it renders: the plan forbids the word "Backed up" over files that exist
in exactly one place, and forbids machinery vocabulary on the main screen.

Nothing here starts a browser or touches the network beyond 127.0.0.1.
"""

from __future__ import annotations

import difflib
import http.client
import json
import re
import socket
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import pytest
import yaml
from test_application_journey import _approve_examples, _project, _stage

from japanese_anki.application import (
    ADDED,
    EXAMPLES_NEED_REVIEW,
    GRAMMAR_NEEDS_REVIEW,
    NOT_EXTRACTED,
    SourceJourney,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.workbench import (
    WorkbenchSession,
    make_server,
    render_dashboard,
    render_source,
)


def _session(tmp_path: Path) -> WorkbenchSession:
    _project(tmp_path)
    return WorkbenchSession.open(ProjectConfig.load(tmp_path))


def _running(session: WorkbenchSession):
    server = make_server(session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request(
    server: Any,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    port = server.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    response_headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, response_headers, payload


# --- the session token gates reads, not just writes -------------------------


def test_the_dashboard_requires_the_session_token(tmp_path: Path) -> None:
    """The whole point of W1.2's boundary over the review panel's. Loopback is
    not a permission: every local process can reach this port, so a bare `/`
    must not hand out the list of documents someone scanned."""
    session = _session(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, body = _request(server, "GET", "/")
        assert status == 404
        assert b"Your Japanese sources" not in body

        status, _headers, body = _request(server, "GET", f"/{session.token}/")
        assert status == 200
        assert b"Your Japanese sources" in body
    finally:
        server.shutdown()
        server.server_close()


def test_a_wrong_token_is_indistinguishable_from_no_such_page(
    tmp_path: Path,
) -> None:
    """A different status or wording would confirm a workbench is running here
    and let a local process wait for the real token."""
    session = _session(tmp_path)
    server, _thread = _running(session)
    try:
        wrong = _request(server, "GET", "/" + "a" * len(session.token) + "/")
        missing = _request(server, "GET", "/nope/")
        assert wrong[0] == missing[0] == 404
        assert wrong[2] == missing[2]
    finally:
        server.shutdown()
        server.server_close()


def test_the_stylesheet_is_behind_the_token_too(tmp_path: Path) -> None:
    session = _session(tmp_path)
    server, _thread = _running(session)
    try:
        assert _request(server, "GET", "/style.css")[0] == 404
        status, headers, _body = _request(
            server, "GET", f"/{session.token}/style.css"
        )
        assert status == 200
        assert headers["content-type"].startswith("text/css")
    finally:
        server.shutdown()
        server.server_close()


# --- the loopback boundary, inherited from the review panel ------------------


def test_a_foreign_host_header_is_refused(tmp_path: Path) -> None:
    session = _session(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, _body = _request(
            server,
            "GET",
            f"/{session.token}/",
            headers={"Host": "evil.example"},
        )
        assert status == 403
    finally:
        server.shutdown()
        server.server_close()


def test_a_cross_origin_header_is_refused(tmp_path: Path) -> None:
    session = _session(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, _body = _request(
            server,
            "GET",
            f"/{session.token}/",
            headers={
                "Host": server.expected_host,
                "Origin": "http://evil.example",
            },
        )
        assert status == 403
    finally:
        server.shutdown()
        server.server_close()


def test_every_response_carries_the_restrictive_headers(tmp_path: Path) -> None:
    session = _session(tmp_path)
    server, _thread = _running(session)
    try:
        for path in (f"/{session.token}/", "/nope/"):
            _status, headers, _body = _request(server, "GET", path)
            assert headers["content-security-policy"].startswith("default-src 'none'")
            assert headers["x-content-type-options"] == "nosniff"
            assert headers["cache-control"] == "no-store"
            assert headers["referrer-policy"] == "same-origin"
            assert headers["x-frame-options"] == "DENY"
    finally:
        server.shutdown()
        server.server_close()


def test_the_page_ships_no_script_and_no_remote_asset(tmp_path: Path) -> None:
    session = _session(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, body = _request(server, "GET", f"/{session.token}/")
        lowered = body.lower()
        assert b"<script" not in lowered
        assert b"http://" not in lowered.replace(b"http://127.0.0.1", b"")
        assert b"https://" not in lowered
    finally:
        server.shutdown()
        server.server_close()


def test_the_server_waits_for_in_flight_work_on_close(tmp_path: Path) -> None:
    session = _session(tmp_path)
    server, _thread = _running(session)
    try:
        assert server.daemon_threads is False
        assert server.block_on_close is True
    finally:
        server.shutdown()
        server.server_close()


# --- what the page actually says --------------------------------------------


def _render(journeys: list[SourceJourney], **kwargs: Any) -> str:
    return render_dashboard(journeys, **kwargs)


def test_it_says_saved_on_this_computer_never_backed_up(tmp_path: Path) -> None:
    """`WORKBENCH_PLAN.md` W1.2, verbatim. These files exist in exactly one
    place until a person copies or commits them, and a cheerful "Backed up"
    badge over a single copy is how someone loses a term's work."""
    html = _render([], root=tmp_path)

    assert "Saved on this computer" in html
    assert "Backed up" not in html
    assert "backed up" not in html.lower()


def test_it_renders_each_source_state_and_next_action(tmp_path: Path) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    journeys, warnings = session.journeys()

    html = _render(journeys, warnings=warnings, root=tmp_path, token=session.token)

    assert "lesson-8.pdf" in html
    assert EXAMPLES_NEED_REVIEW in html
    assert GRAMMAR_NEEDS_REVIEW in html
    assert "Review the Japanese examples on 2 cards" in html


def test_the_grammar_badge_is_visible_beside_the_card_state(tmp_path: Path) -> None:
    """Parallel tracks: a source whose cards need review must still show that
    its grammar does too, or one track hides the other."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    journeys, _warnings = session.journeys()

    html = _render(journeys, token=session.token)

    assert html.index(EXAMPLES_NEED_REVIEW) < html.index(GRAMMAR_NEEDS_REVIEW)


def test_a_promoted_source_is_shown_without_a_review_command(
    tmp_path: Path,
) -> None:
    """It has no live staging file, so it has no cards left to review."""
    journeys = [
        SourceJourney(
            source="week-8.pdf",
            state=ADDED,
            next_action="Add dictionary facts, audio, and build the deck",
        )
    ]

    html = _render(journeys)

    assert ADDED in html
    # No stale instruction to run a command that no longer exists.
    assert "review-panel" not in html


def test_a_source_filename_cannot_inject_markup(tmp_path: Path) -> None:
    """The name comes from whatever the person dropped in."""
    journeys = [
        SourceJourney(
            source='<img src=x onerror="alert(1)">.pdf',
            state=NOT_EXTRACTED,
            next_action="Read this source to propose cards and grammar",
        )
    ]

    html = _render(journeys)

    # The property that matters is that no *tag* survives: with `<` escaped
    # there is no element for an attribute to live on, so the literal word
    # "onerror" remaining as inert text content is fine and expected.
    assert '<img src=x onerror="alert(1)">' not in html
    assert "<img" not in html
    assert 'onerror="' not in html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;.pdf" in html


def test_an_unreadable_file_is_shown_not_silently_dropped(tmp_path: Path) -> None:
    _project(tmp_path)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(exist_ok=True)
    (staging_dir / "broken.pdf.yaml").write_text("records: [oops\n", encoding="utf-8")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    journeys, warnings = session.journeys()

    html = _render(journeys, warnings=warnings, token=session.token)

    assert "Could not be read" in html
    assert "broken.pdf.yaml" in html


def test_an_empty_corpus_explains_what_to_do(tmp_path: Path) -> None:
    html = _render([], root=tmp_path)

    assert "No sources yet" in html
    assert "inbox" in html


def test_the_page_never_shows_machinery_vocabulary(tmp_path: Path) -> None:
    """The learner-facing contract. `staging`/`promote`/`fingerprint` belong
    under Technical details, never in the states or actions themselves."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    journeys, _warnings = session.journeys()

    visible = "".join(
        f"{journey.state} {journey.next_action} {journey.grammar}"
        for journey in journeys
    ).lower()

    for word in ("staging", "promote", "fingerprint", "run id", "yaml", "json"):
        assert word not in visible, word


# --- the projection is recomputed, never cached -----------------------------


def test_a_change_on_disk_shows_up_without_a_restart(tmp_path: Path) -> None:
    """The dashboard reconstructs from the repository on every request, so a
    `promote` in another terminal is visible on refresh — and so nothing this
    process remembers can outlive what the files say."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    server, _thread = _running(session)
    try:
        _status, _headers, before = _request(server, "GET", f"/{session.token}/")
        assert GRAMMAR_NEEDS_REVIEW.encode() in before

        store_path = tmp_path / "patterns.json"
        store = json.loads(store_path.read_text(encoding="utf-8"))
        for entry in store.values():
            entry["reviewed"] = True
        store_path.write_text(
            json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        _status, _headers, after = _request(server, "GET", f"/{session.token}/")
        assert GRAMMAR_NEEDS_REVIEW.encode() not in after
        assert b"Grammar reviewed" in after
    finally:
        server.shutdown()
        server.server_close()


# --- W2a: one source, opened ------------------------------------------------


def _source_url(session: WorkbenchSession, name: str) -> str:
    return f"/{session.token}/source/{quote(name, safe='')}"


def test_a_source_page_needs_the_token_like_everything_else(
    tmp_path: Path,
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    server, _thread = _running(session)
    try:
        assert _request(server, "GET", "/source/lesson-8.pdf")[0] == 404
        assert _request(server, "GET", _source_url(session, "lesson-8.pdf"))[0] == 200
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "attempt",
    [
        "../../../../etc/passwd",
        "..%2f..%2fetc%2fpasswd",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "/etc/passwd",
    ],
)
def test_a_source_name_cannot_name_a_path(tmp_path: Path, attempt: str) -> None:
    """A traversal name matches no source the dashboard computed, so it 404s.

    Note what this does *not* prove: it passes even if the lookup joins the
    name onto `staging_dir`, because the name-match refuses first.
    `test_a_source_is_opened_by_name_not_by_filename` below is the one that
    fails under that change."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    server, _thread = _running(session)
    try:
        status, _headers, body = _request(
            server, "GET", f"/{session.token}/source/{attempt}"
        )
        assert status == 404
        assert b"root:" not in body
    finally:
        server.shutdown()
        server.server_close()


def test_an_unknown_source_is_a_plain_404(tmp_path: Path) -> None:
    session = _session(tmp_path)
    server, _thread = _running(session)
    try:
        assert _request(server, "GET", _source_url(session, "nope.pdf"))[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def test_a_promoted_source_has_no_page_to_open(tmp_path: Path) -> None:
    """It has no live staging file, so there are no staged cards to show."""
    _project(tmp_path)
    (tmp_path / "inbox" / "week-8.pdf").write_bytes(b"%PDF-1.7 fake")
    archive = tmp_path / "staging" / "done"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "week-8.pdf.yaml").write_text("records: []\n", encoding="utf-8")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    server, _thread = _running(session)
    try:
        assert _request(server, "GET", _source_url(session, "week-8.pdf"))[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def test_the_source_page_shows_japanese_before_machinery(tmp_path: Path) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    detail = session.detail("lesson-8.pdf")

    html = render_source(detail, token=session.token)

    assert "あげる" in html
    assert "Meaning in this lesson" in html
    assert "誕生日にプレゼントをあげます。" in html
    assert "How to use it" in html
    # The stable ID is evidence, not a headline: it lives after the learning
    # content, inside the collapsed details block.
    assert html.index("Meaning in this lesson") < html.index("word:あげる:あげる")
    assert "Source and technical details" in html


def test_the_grammar_section_is_display_only(tmp_path: Path) -> None:
    """Pattern content is extraction output. This page may show it and later
    mark the set reviewed; it never becomes a pattern editor."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))

    html = render_source(session.detail("lesson-8.pdf"), token=session.token)

    assert "Grammar from this lesson" in html
    assert "〜てあげる" in html
    assert "〜てもらう" in html
    # Display-only means the *content* cannot be edited here. Rendered without
    # a session there is no form at all; the only control W2b ever adds is the
    # "I read this" checkbox, which records a review rather than editing text.
    assert "<textarea" not in html
    assert "<input" not in html


def test_an_unapproved_card_states_what_approval_does_not_cover(
    tmp_path: Path,
) -> None:
    """Stated beside the control, not only in help text."""
    _stage(tmp_path, "table_exhaustive", filename="table.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))

    html = render_source(session.detail("table.pdf"), token=session.token)

    assert "Not yet approved" in html
    assert "not the meanings" in html
    assert "usage note" in html


def test_an_approved_card_names_the_exact_word_it_covers(tmp_path: Path) -> None:
    _stage(tmp_path, "table_exhaustive", filename="table.pdf")
    _approve_examples(tmp_path, "table.pdf.yaml")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))

    html = render_source(session.detail("table.pdf"), token=session.token)

    assert "Japanese examples approved for 走る (はしる)" in html


# --- the existing-wins merge, made visible ----------------------------------


def _with_existing(tmp_path: Path, meanings: list[str]) -> WorkbenchSession:
    """A collection that already holds あげる, with its own meanings."""
    _stage(tmp_path, "shared_word_source_a", filename="lesson-8.pdf")
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            [
                {
                    "id": "word:あげる:あげる",
                    "expression": "あげる",
                    "reading": "あげる",
                    "meanings": meanings,
                    "examples": [],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return WorkbenchSession.open(ProjectConfig.load(tmp_path))


def test_three_meaning_panels_show_what_the_merge_will_keep(
    tmp_path: Path,
) -> None:
    """`Meaning in this lesson` must not imply it replaces what is on the card.
    Existing-wins keeps the older populated meaning, and the panels say so."""
    session = _with_existing(tmp_path, ["to give (already curated)"])

    html = render_source(session.detail("lesson-8.pdf"), token=session.token)

    assert "Proposed by this lesson" in html
    assert "Currently on your card" in html
    assert "What will remain after adding" in html
    assert "to give (already curated)" in html
    assert "keeps them" in html


def test_a_new_word_says_so_instead_of_showing_empty_panels(
    tmp_path: Path,
) -> None:
    _stage(tmp_path, "shared_word_source_a", filename="lesson-8.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))

    html = render_source(session.detail("lesson-8.pdf"), token=session.token)

    assert "This word is new to your collection." in html
    assert "Currently on your card" not in html


def test_the_merge_preview_calls_the_real_merge(tmp_path: Path) -> None:
    """A preview from a second implementation of existing-wins would be a lie
    with a progress bar. This proves the shown result is promote's own."""
    session = _with_existing(tmp_path, ["to give (already curated)"])

    detail = session.detail("lesson-8.pdf")
    card = detail.cards[0]

    assert card.existing is not None
    assert card.merged_meanings == ("to give (already curated)",)
    assert card.proposed_meanings_would_be_kept is False


def test_a_source_is_opened_by_name_not_by_filename(tmp_path: Path) -> None:
    """Staging metadata may name a source differently from the file holding it
    — the real corpus's Yotsubato pack is `source_file: Yotsubato Volume 1
    Reading Pack Vocab` inside `anki-yotsubato-….yaml`. Opening it therefore
    cannot mean `staging_dir / (name + ".yaml")`; it means the path the journey
    already resolved. This is the test that catches that substitution."""
    _stage(tmp_path, "table_exhaustive", filename="week-8.pdf")
    staging_path = tmp_path / "staging" / "week-8.pdf.yaml"
    data = yaml.safe_load(staging_path.read_text(encoding="utf-8"))
    data["source_file"] = "Week 8 Handout"
    staging_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))

    detail = session.detail("Week 8 Handout")

    assert detail is not None
    assert [card.record.expression for card in detail.cards] == ["走る", "食べる", "飲む"]


# --- W2b: approval writes ---------------------------------------------------


def _form_fields(body: bytes) -> dict[str, str]:
    """The hidden fields the page rendered, as a browser would submit them."""
    text = body.decode()
    return {
        name: value
        for name, value in re.findall(
            r'<input type=hidden name=(\w+) value="([^"]*)">', text
        )
    }


def _approve(
    server: Any,
    session: WorkbenchSession,
    source: str,
    *,
    fields: dict[str, str],
    records: Sequence[str] = (),
    patterns_too: bool = False,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    pairs = [(key, value) for key, value in fields.items()]
    pairs += [("record", record) for record in records]
    if patterns_too:
        pairs.append(("patterns", "review"))
    return _request(
        server,
        "POST",
        f"/{session.token}/source/{quote(source, safe='')}/approve",
        headers={
            "Host": server.expected_host,
            "Content-Type": "application/x-www-form-urlencoded",
            **(headers or {}),
        },
        body=urlencode(pairs).encode(),
    )


def _staged(tmp_path: Path, name: str = "table.pdf") -> Any:
    _stage(tmp_path, "table_exhaustive", filename=name)
    return WorkbenchSession.open(ProjectConfig.load(tmp_path))


def test_approving_a_card_writes_the_exact_approval(tmp_path: Path) -> None:
    """The end-to-end W2b path: render, tick, submit, and the staging file on
    disk now carries authority for exactly that card."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "table.pdf")
        )
        fields = _form_fields(page)
        status, headers, _body = _approve(
            server, session, "table.pdf", fields=fields, records=["word:走る:はしる"]
        )
        assert status == 303  # post/redirect/get: a reload cannot resubmit
        assert "saved=1" in headers["location"]

        detail = session.detail("table.pdf")
        by_id = {card.record.id: card for card in detail.cards}
        assert by_id["word:走る:はしる"].authority == "existing"
        # ...and only that card.
        assert by_id["word:食べる:たべる"].authority == "available"
    finally:
        server.shutdown()
        server.server_close()


def test_an_approval_without_the_session_csrf_is_refused(tmp_path: Path) -> None:
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "table.pdf")
        )
        fields = _form_fields(page)
        fields["csrf"] = "not-the-session-token"
        status, _headers, _body = _approve(
            server, session, "table.pdf", fields=fields, records=["word:走る:はしる"]
        )
        assert status == 403
        assert session.detail("table.pdf").cards[0].authority == "available"
    finally:
        server.shutdown()
        server.server_close()


def test_an_approval_rendered_against_older_bytes_is_refused_whole(
    tmp_path: Path,
) -> None:
    """The compare-and-swap. Someone editing the staging file in another window
    must not have a stale page's approval land on top of their edit."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "table.pdf")
        )
        fields = _form_fields(page)

        staging_path = tmp_path / "staging" / "table.pdf.yaml"
        before = staging_path.read_bytes()
        staging_path.write_bytes(before + b"\n# a concurrent human edit\n")
        changed = staging_path.read_bytes()

        status, _headers, body = _approve(
            server, session, "table.pdf", fields=fields, records=["word:走る:はしる"]
        )
        assert status == 409
        assert b"Nothing was written" in body
        # The concurrent edit survives byte-identically.
        assert staging_path.read_bytes() == changed
    finally:
        server.shutdown()
        server.server_close()


def test_replaying_the_same_approval_cannot_land_twice(tmp_path: Path) -> None:
    """The first submit changes the file, so the second's snapshot no longer
    matches. A resend is refused rather than re-applied."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "table.pdf")
        )
        fields = _form_fields(page)
        first = _approve(
            server, session, "table.pdf", fields=fields, records=["word:走る:はしる"]
        )
        assert first[0] == 303
        after_first = (tmp_path / "staging" / "table.pdf.yaml").read_bytes()

        second = _approve(
            server, session, "table.pdf", fields=fields, records=["word:走る:はしる"]
        )
        assert second[0] == 409
        assert (tmp_path / "staging" / "table.pdf.yaml").read_bytes() == after_first
    finally:
        server.shutdown()
        server.server_close()


def test_grammar_is_approved_separately_from_cards(tmp_path: Path) -> None:
    """There is no Approve all: ticking cards leaves grammar unreviewed."""
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "lesson-8.pdf")
        )
        fields = _form_fields(page)
        status, _headers, _body = _approve(
            server,
            session,
            "lesson-8.pdf",
            fields=fields,
            records=["word:あげる:あげる"],
        )
        assert status == 303

        detail = session.detail("lesson-8.pdf")
        assert detail.pattern_reviewed is False
        assert detail.journey.grammar == GRAMMAR_NEEDS_REVIEW
    finally:
        server.shutdown()
        server.server_close()


def test_grammar_can_be_reviewed_without_approving_any_card(
    tmp_path: Path,
) -> None:
    _stage(tmp_path, "lesson_with_grammar", filename="lesson-8.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "lesson-8.pdf")
        )
        status, _headers, _body = _approve(
            server,
            session,
            "lesson-8.pdf",
            fields=_form_fields(page),
            patterns_too=True,
        )
        assert status == 303

        detail = session.detail("lesson-8.pdf")
        assert detail.pattern_reviewed is True
        assert all(card.authority == "available" for card in detail.cards)
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("headers", "body", "expected"),
    [
        ({"Transfer-Encoding": "chunked"}, b"0\r\n\r\n", 400),
        ({"Content-Type": "text/plain"}, b"action=save", 415),
    ],
)
def test_a_malformed_submission_is_refused(
    tmp_path: Path, headers: dict[str, str], body: bytes, expected: int
) -> None:
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, _body = _request(
            server,
            "POST",
            f"/{session.token}/source/table.pdf/approve",
            headers={
                "Host": server.expected_host,
                "Content-Type": "application/x-www-form-urlencoded",
                **headers,
            },
            body=body,
        )
        assert status == expected
    finally:
        server.shutdown()
        server.server_close()


def test_an_oversized_form_is_refused_before_it_is_read(tmp_path: Path) -> None:
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, _body = _request(
            server,
            "POST",
            f"/{session.token}/source/table.pdf/approve",
            headers={
                "Host": server.expected_host,
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(64 * 1024 + 1),
            },
            body=b"x" * 16,
        )
        assert status == 413
    finally:
        server.shutdown()
        server.server_close()


def test_an_approval_post_needs_the_path_token_too(tmp_path: Path) -> None:
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, _body = _request(
            server,
            "POST",
            "/source/table.pdf/approve",
            headers={
                "Host": server.expected_host,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body=urlencode([("action", "save")]).encode(),
        )
        assert status == 404
    finally:
        server.shutdown()
        server.server_close()


def test_a_cross_origin_approval_is_refused(tmp_path: Path) -> None:
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "table.pdf")
        )
        status, _headers, _body = _approve(
            server,
            session,
            "table.pdf",
            fields=_form_fields(page),
            records=["word:走る:はしる"],
            headers={"Origin": "http://evil.example"},
        )
        assert status == 403
        assert session.detail("table.pdf").cards[0].authority == "available"
    finally:
        server.shutdown()
        server.server_close()


def test_the_checkbox_states_what_it_does_not_approve(tmp_path: Path) -> None:
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "table.pdf")
        )
        text = page.decode()
        assert "Approve these Japanese example sentences" in text
        assert "走る (はしる)" in text
        assert "not the meanings" in text
        assert "usage note" in text
        # No single control that would blur separate decisions together.
        assert "Approve all" not in text
    finally:
        server.shutdown()
        server.server_close()


# --- boundary cases inherited from the deleted panel's adversarial suite ----


def test_a_null_origin_is_refused(tmp_path: Path) -> None:
    """Chrome serializes a same-origin form POST as `Origin: null` under
    no-referrer, which is why the boundary sends `Referrer-Policy:
    same-origin` rather than a stricter one. `null` itself is not this
    origin and must still be refused."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, _body = _request(
            server,
            "GET",
            _source_url(session, "table.pdf"),
            headers={"Host": server.expected_host, "Origin": "null"},
        )
        assert status == 403
    finally:
        server.shutdown()
        server.server_close()


def test_duplicate_authority_headers_are_refused(tmp_path: Path) -> None:
    """Two `Host` headers let a proxy and this server disagree about which
    authority the request was for. Exactly one is required."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        port = server.server_address[1]
        raw = (
            f"GET /{session.token}/ HTTP/1.1\r\n"
            f"Host: {server.expected_host}\r\n"
            f"Host: {server.expected_host}\r\n"
            "Connection: close\r\n\r\n"
        ).encode()
        with socket.create_connection(("127.0.0.1", port), timeout=3) as client:
            client.sendall(raw)
            response = client.recv(4096)
        assert b" 403 " in response.split(b"\r\n", 1)[0]
    finally:
        server.shutdown()
        server.server_close()


def test_a_duplicated_form_field_is_refused(tmp_path: Path) -> None:
    """Two `csrf` values would let a submission carry both a valid token and a
    forged one and hope the server reads the right index."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "table.pdf")
        )
        fields = _form_fields(page)
        pairs = list(fields.items()) + [("csrf", "second-value")]
        status, _headers, _body = _request(
            server,
            "POST",
            f"/{session.token}/source/table.pdf/approve",
            headers={
                "Host": server.expected_host,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body=urlencode(pairs).encode(),
        )
        assert status == 400
        assert session.detail("table.pdf").cards[0].authority == "available"
    finally:
        server.shutdown()
        server.server_close()


def test_two_concurrent_approvals_land_exactly_once(tmp_path: Path) -> None:
    """The workbench opens a *fresh* panel per request, so the panel's own
    in-process submission lock cannot serialize two requests the way it did for
    the single long-lived page. What protects the file here is the advisory
    lock plus the exact-byte snapshot: whichever request loses the race finds
    the bytes changed and is refused, rather than applying its approval on top
    of the other's write."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "table.pdf")
        )
        fields = _form_fields(page)
        results: list[int] = []
        barrier = threading.Barrier(2)

        def submit() -> None:
            barrier.wait()
            status, _h, _b = _approve(
                server,
                session,
                "table.pdf",
                fields=fields,
                records=["word:走る:はしる"],
            )
            results.append(status)

        threads = [threading.Thread(target=submit) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert sorted(results) == [303, 409], results
        # And exactly one approval is on disk.
        cards = {c.record.id: c for c in session.detail("table.pdf").cards}
        assert cards["word:走る:はしる"].authority == "existing"
        assert cards["word:食べる:たべる"].authority == "available"
    finally:
        server.shutdown()
        server.server_close()


# --- W2c: correcting a card's human-owned fields ----------------------------


def _edit_url(session: WorkbenchSession, name: str) -> str:
    return f"{_source_url(session, name)}?edit=1"


def _submit_edit(
    server: Any,
    session: WorkbenchSession,
    source: str,
    fields: dict[str, str],
    edits: Sequence[tuple[str, str]],
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    return _request(
        server,
        "POST",
        f"/{session.token}/source/{quote(source, safe='')}/edit",
        headers={
            "Host": server.expected_host,
            "Content-Type": "application/x-www-form-urlencoded",
            **(headers or {}),
        },
        body=urlencode(list(fields.items()) + list(edits)).encode(),
    )


def test_correcting_a_gloss_changes_only_that_line(tmp_path: Path) -> None:
    """W2c's ships-when: every byte you did not edit survives unchanged."""
    session = _staged(tmp_path)
    staging_path = tmp_path / "staging" / "table.pdf.yaml"
    before = staging_path.read_text(encoding="utf-8").split("\n")
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        status, headers, _body = _submit_edit(
            server,
            session,
            "table.pdf",
            _form_fields(page),
            [("ee0_0", "I jog through the park every morning.")],
        )
        assert status == 303
        assert "edited=1" in headers["location"]

        after = staging_path.read_text(encoding="utf-8").split("\n")
        changed = [
            line
            for line in difflib.unified_diff(before, after, lineterm="", n=0)
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]
        assert changed == [
            "-    english: I run through the park every morning.",
            "+    english: I jog through the park every morning.",
        ], changed
    finally:
        server.shutdown()
        server.server_close()


def test_correcting_the_japanese_voids_that_cards_approval(tmp_path: Path) -> None:
    """Approval binds the exact sentence by fingerprint, so rewriting the
    sentence makes the approval stop covering it. A tick must never end up
    standing over text nobody read."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        # Approve through the workbench, which binds each sentence's exact text
        # by fingerprint — not the hand-typed sentinel, which is a blanket mark
        # (see the test below).
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "table.pdf")
        )
        _approve(
            server,
            session,
            "table.pdf",
            fields=_form_fields(page),
            records=["word:走る:はしる", "word:食べる:たべる"],
        )
        assert session.detail("table.pdf").cards[0].authority == "existing"

        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        status, _headers, _body = _submit_edit(
            server,
            session,
            "table.pdf",
            _form_fields(page),
            [("ej0_0", "毎朝、公園を走っています。")],
        )
        assert status == 303

        cards = {c.record.id: c for c in session.detail("table.pdf").cards}
        assert cards["word:走る:はしる"].authority == "stale"
        # ...and only that card. The others were never touched.
        assert cards["word:食べる:たべる"].authority == "existing"
    finally:
        server.shutdown()
        server.server_close()


def test_correcting_the_english_leaves_approval_standing(tmp_path: Path) -> None:
    """Approval never covered the English, so changing it changes nothing
    about what a person approved."""
    session = _staged(tmp_path)
    _approve_examples(tmp_path, "table.pdf.yaml")
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        _submit_edit(
            server,
            session,
            "table.pdf",
            _form_fields(page),
            [("ee0_0", "I jog through the park each morning.")],
        )
        assert session.detail("table.pdf").cards[0].authority == "existing"
    finally:
        server.shutdown()
        server.server_close()


def test_source_evidence_has_no_edit_field(tmp_path: Path) -> None:
    """The page, the source sentence and the inclusion reason are a record of
    what happened. A form that let someone retype them would let them retype
    history, so no control exists for them at all."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        text = page.decode()
        names = set(re.findall(r'<textarea id="(\w+)"', text))
        # Only the allowlisted prefixes: meanings, usage, and the five example
        # fields. Nothing addressing source evidence or the accounting block.
        assert names
        assert all(re.match(r"\A(m|u|ej|ef|er|ee|eg)\d", name) for name in names), names
        assert "inclusion_reason" not in text.replace("Why it was included", "")
    finally:
        server.shutdown()
        server.server_close()


def test_an_unknown_edit_field_is_refused_not_ignored(tmp_path: Path) -> None:
    """Ignoring it would report a saved edit that never happened, and would be
    the seam through which an uneditable field became editable."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        status, _headers, body = _submit_edit(
            server,
            session,
            "table.pdf",
            _form_fields(page),
            [("inclusion_reason0", "because I said so")],
        )
        assert status == 400
        assert b"Unknown edit field" in body
    finally:
        server.shutdown()
        server.server_close()


def test_an_edit_naming_a_card_that_is_not_there_is_refused(
    tmp_path: Path,
) -> None:
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        status, _headers, _body = _submit_edit(
            server, session, "table.pdf", _form_fields(page), [("m99", "nope")]
        )
        assert status == 400
    finally:
        server.shutdown()
        server.server_close()


def test_an_edit_without_the_session_csrf_is_refused(tmp_path: Path) -> None:
    session = _staged(tmp_path)
    staging_path = tmp_path / "staging" / "table.pdf.yaml"
    before = staging_path.read_bytes()
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        fields = _form_fields(page)
        fields["csrf"] = "forged"
        status, _headers, _body = _submit_edit(
            server, session, "table.pdf", fields, [("ee0_0", "nope")]
        )
        assert status == 403
        assert staging_path.read_bytes() == before
    finally:
        server.shutdown()
        server.server_close()


def test_an_edit_from_a_stale_page_is_refused_whole(tmp_path: Path) -> None:
    session = _staged(tmp_path)
    staging_path = tmp_path / "staging" / "table.pdf.yaml"
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        fields = _form_fields(page)
        staging_path.write_bytes(
            staging_path.read_bytes() + b"\n# a concurrent human edit\n"
        )
        changed = staging_path.read_bytes()

        status, _headers, body = _submit_edit(
            server, session, "table.pdf", fields, [("ee0_0", "nope")]
        )
        assert status == 409
        assert b"Nothing was written" in body
        assert staging_path.read_bytes() == changed
    finally:
        server.shutdown()
        server.server_close()


def test_submitting_no_change_writes_nothing(tmp_path: Path) -> None:
    """Rewriting an identical file would still touch its mtime and its Git
    status for no reason, and would claim a save that changed nothing."""
    session = _staged(tmp_path)
    staging_path = tmp_path / "staging" / "table.pdf.yaml"
    before = staging_path.read_bytes()
    before_mtime = staging_path.stat().st_mtime_ns
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        status, headers, _body = _submit_edit(
            server, session, "table.pdf", _form_fields(page), []
        )
        assert status == 303
        assert "edited=0" in headers["location"]
        assert staging_path.read_bytes() == before
        assert staging_path.stat().st_mtime_ns == before_mtime
    finally:
        server.shutdown()
        server.server_close()


def test_edit_mode_offers_undo_and_is_separate_from_approving(
    tmp_path: Path,
) -> None:
    """Correcting and approving are different acts. One form carrying both
    would let a stray click approve sentences someone was only fixing."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, edit_page = _request(
            server, "GET", _edit_url(session, "table.pdf")
        )
        _status, _headers, read_page = _request(
            server, "GET", _source_url(session, "table.pdf")
        )
        edit_text, read_text = edit_page.decode(), read_page.decode()

        # Undo before save, with no JavaScript and no server round trip.
        assert "type=reset" in edit_text
        assert "Leave without saving" in edit_text
        # The edit view offers no approval checkbox...
        assert 'name=record' not in edit_text
        assert "Approve these Japanese example sentences" not in edit_text
        # ...and the read view offers no edit field.
        assert "<textarea" not in read_text
        assert "Approve these Japanese example sentences" in read_text
    finally:
        server.shutdown()
        server.server_close()


def test_the_hand_typed_sentinel_is_a_blanket_mark_not_a_sentence_binding(
    tmp_path: Path,
) -> None:
    """A trap worth naming. `example_authority: staging-review`, typed into
    YAML by hand, is not bound to any particular sentence — so editing the
    Japanese afterwards does *not* void it, and `promote` will later bind
    whatever sentences are in the file at that moment. The workbench's own
    approval writes per-sentence fingerprints instead, which is why editing
    through the tab does void it. Anyone hand-editing YAML should know these
    two marks are not interchangeable."""
    session = _staged(tmp_path)
    _approve_examples(tmp_path, "table.pdf.yaml")
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        _submit_edit(
            server,
            session,
            "table.pdf",
            _form_fields(page),
            [("ej0_0", "まったく違う文。")],
        )
        card = session.detail("table.pdf").cards[0]
        assert card.record.examples[0].japanese == "まったく違う文。"
        assert card.authority == "existing"
    finally:
        server.shutdown()
        server.server_close()


def test_an_uncertain_edit_write_never_claims_nothing_was_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`bound_replace` distinguishes two failures that must never be reported
    the same way. A refused compare-and-swap proves nothing was written. A
    write that failed *after* its snapshot stopped matching proves nothing at
    all — it may have landed. Saying "nothing was written" there is a lie at
    exactly the moment someone needs the truth."""
    from japanese_anki.workbench import review as review_module

    session = _staged(tmp_path)
    server, _thread = _running(session)

    def indeterminate(*_args: object, **_kwargs: object) -> None:
        raise review_module.IndeterminateWriteError(
            "The staging file write failed after its exact snapshot stopped "
            "matching; whether it landed is unknown.",
            intended_bytes_are_live=False,
        )

    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        monkeypatch.setattr(review_module, "bound_replace", indeterminate)
        status, _headers, body = _submit_edit(
            server,
            session,
            "table.pdf",
            _form_fields(page),
            [("ee0_0", "I jog through the park every morning.")],
        )
        assert status == 409
        text = body.decode()
        assert "whether it landed is unknown" in text
        assert "Nothing was written" not in text
        assert "check the cards" in text
    finally:
        server.shutdown()
        server.server_close()


def test_register_is_a_choice_not_a_text_box(tmp_path: Path) -> None:
    """Anything outside polite/casual makes the example incomplete, so a free
    text box would let a typo quietly degrade a card. Offer the whole domain
    instead: the two values and a blank."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        text = page.decode()
        assert '<select id="eg0_0"' in text
        assert '<option value="polite" selected>' in text
        assert '<option value="casual">' in text
        # ...and no textarea claiming to hold a register.
        assert '<textarea id="eg0_0"' not in text
    finally:
        server.shutdown()
        server.server_close()
