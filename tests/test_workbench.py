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
import html as html_module
import http.client
import json
import re
import socket
import threading
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import pytest
import yaml
from test_application_journey import _approve_examples, _project, _stage

from japanese_anki import promote, staging
from japanese_anki.application import (
    ADDED,
    EXAMPLES_NEED_REVIEW,
    GRAMMAR_NEEDS_REVIEW,
    NOT_EXTRACTED,
    SourceJourney,
)
from japanese_anki.application.promotion import staged_ai_enrichment
from japanese_anki.config import ProjectConfig
from japanese_anki.models import (
    ExampleSentence,
    SourceReference,
    VocabularyRecord,
)
from japanese_anki.staging import read_staging, write_staging
from japanese_anki.workbench import (
    STYLE,
    WorkbenchSession,
    make_server,
    render_dashboard,
    render_source,
    review,
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
    """The first form's hidden fields, as a browser would submit them.

    First-wins, not last-wins: an edit page also carries a removal form per
    card, and those repeat `action`, `csrf` and `staging_snapshot` with
    different values. Taking the last occurrence would silently build a
    removal submission while claiming to be an edit.
    """
    text = body.decode()
    # Only the first form. An edit page also carries one removal form per card,
    # each repeating `action`/`csrf`/`staging_snapshot` and adding `card` — and
    # a browser submits the fields of *one* form, never a union of all of them.
    end = text.find("</form>")
    if end != -1:
        text = text[:end]
    return dict(
        re.findall(r'<input type=hidden name=(\w+) value="([^"]*)">', text)
    )


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


def test_correcting_a_gloss_changes_only_that_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W2c's ships-when: every byte you did not edit survives unchanged."""
    session = _staged(tmp_path)
    staging_path = tmp_path / "staging" / "table.pdf.yaml"
    snapshot = staging_path.read_bytes()
    transient = snapshot + b"transient_private_note: must-not-return\n"
    real_render = staging.render_staging_update

    def render_during_transient_note(
        captured: bytes,
        updated: Sequence[VocabularyRecord],
        *,
        source: str,
    ) -> str:
        staging_path.write_bytes(transient)
        try:
            return real_render(captured, updated, source=source)
        finally:
            staging_path.write_bytes(snapshot)

    monkeypatch.setattr(
        staging,
        "render_staging_update",
        render_during_transient_note,
    )
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


def test_bound_replace_joins_the_staging_transaction_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from japanese_anki.workbench import review as review_module

    path = tmp_path / "review.yaml"
    path.write_text("old\n", encoding="utf-8")
    entered: list[Path] = []

    class ObservedLock:
        def __init__(self, target: Path) -> None:
            self.target = target

        def __enter__(self) -> None:
            entered.append(self.target)

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        review_module,
        "exclusive_path_lock",
        lambda target: ObservedLock(target),
    )

    review_module.bound_replace(path, "new\n", b"old\n", label="staging file")

    assert entered == [path]
    assert path.read_text(encoding="utf-8") == "new\n"


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


def test_a_register_the_select_cannot_represent_survives_being_edited(
    tmp_path: Path,
) -> None:
    """A `<select>` destroys anything outside its options: nothing matches, the
    browser falls back to the first entry, and merely opening the editor and
    saving erases what someone wrote by hand. An unrecognized value therefore
    gets its own option and comes back unchanged."""
    session = _staged(tmp_path)
    staging_path = tmp_path / "staging" / "table.pdf.yaml"
    data = yaml.safe_load(staging_path.read_text(encoding="utf-8"))
    data["records"][0]["examples"][0]["register"] = "formal"
    staging_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        assert '<option value="formal" selected>' in page.decode()

        # Round-trip through a save that changes something else entirely.
        status, _headers, _body = _submit_edit(
            server,
            session,
            "table.pdf",
            _form_fields(page),
            [("eg0_0", "formal"), ("ee0_0", "I jog through the park.")],
        )
        assert status == 303
        card = session.detail("table.pdf").cards[0]
        assert card.record.examples[0].register == "formal"
    finally:
        server.shutdown()
        server.server_close()


def test_resubmitting_every_field_verbatim_writes_nothing(tmp_path: Path) -> None:
    """The strongest form of "only what you edited changes": submit the whole
    editor back exactly as rendered and the file must be byte-identical.

    This is the test that catches a false positive in the changed-record
    comparison. One did exist — examples were rebuilt as a tuple while
    `to_dict` emits the declared list, so every card compared as changed and
    every save rewrote every row, re-folding long scalars the editor does not
    even expose, including `inclusion_reason`.
    """
    _stage(tmp_path, "lesson_with_grammar", filename="lesson.pdf")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    staging_path = tmp_path / "staging" / "lesson.pdf.yaml"
    before = staging_path.read_bytes()
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson.pdf"))
        text = page.decode()
        fields: list[tuple[str, str]] = [
            (name, html_module.unescape(value))
            for name, value in re.findall(
                r'<textarea id="(\w+)"[^>]*>(.*?)</textarea>', text, re.S
            )
        ]
        for name, options in re.findall(
            r'<select id="(\w+)"[^>]*>(.*?)</select>', text, re.S
        ):
            chosen = re.search(r'<option value="([^"]*)" selected>', options)
            fields.append((name, chosen.group(1) if chosen else ""))
        assert fields, "the editor rendered no fields"

        status, headers, _body = _submit_edit(
            server, session, "lesson.pdf", _form_fields(page), fields
        )
        assert status == 303
        assert "edited=0" in headers["location"]
        assert staging_path.read_bytes() == before
    finally:
        server.shutdown()
        server.server_close()


# --- W2c: removing a proposed card ------------------------------------------


def _remove_fields(body: bytes, index: int) -> dict[str, str]:
    """The hidden fields of one card's removal form."""
    match = re.search(
        rf'<form id="rm{index}"[^>]*>(.*?)</form>', body.decode(), re.S
    )
    assert match, f"no removal form for card {index}"
    return dict(
        re.findall(r'<input type=hidden name=(\w+) value="([^"]*)">', match.group(1))
    )


def _remove(
    server: Any,
    session: WorkbenchSession,
    source: str,
    fields: dict[str, str],
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    return _request(
        server,
        "POST",
        f"/{session.token}/source/{quote(source, safe='')}/remove",
        headers={
            "Host": server.expected_host,
            "Content-Type": "application/x-www-form-urlencoded",
            **(headers or {}),
        },
        body=urlencode(list(fields.items())).encode(),
    )


def test_removal_forms_are_never_nested_inside_the_editor(tmp_path: Path) -> None:
    """Forms cannot nest — a browser silently drops an inner one, so a per-card
    <form> inside the editor would render a button that does nothing at all.
    The buttons are associated by `form=` with forms declared after the
    editor's form closes.
    """
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        text = page.decode()
        editor_end = text.index("</form>")
        # No <form> opens between the editor's <form> and its closing tag.
        editor = text[text.index("<form method=post") : editor_end]
        assert "<form" not in editor[len("<form method=post") :]
        # ...and every removal form lives after it, with a button pointing at it.
        for index in range(3):
            assert f'<form id="rm{index}"' in text[editor_end:]
            assert f'form="rm{index}"' in editor
    finally:
        server.shutdown()
        server.server_close()


def test_removing_a_card_drops_only_that_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _staged(tmp_path)
    staging_path = tmp_path / "staging" / "table.pdf.yaml"
    snapshot = staging_path.read_bytes()
    records, meta = read_staging(staging_path)
    transient_path = tmp_path / "staging" / "transient.yaml"
    write_staging(transient_path, [records[1], records[0], records[2]], meta)
    transient = transient_path.read_bytes()
    transient_path.unlink()
    real_render = staging.render_staging_prune

    def render_during_transient_order(
        captured: bytes,
        keep: Sequence[bool],
        *,
        source: str,
    ) -> str | None:
        staging_path.write_bytes(transient)
        try:
            return real_render(captured, keep, source=source)
        finally:
            staging_path.write_bytes(snapshot)

    monkeypatch.setattr(
        staging,
        "render_staging_prune",
        render_during_transient_order,
    )
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        status, headers, _body = _remove(
            server, session, "table.pdf", _remove_fields(page, 1)
        )
        assert status == 303
        assert "removed=1" in headers["location"]

        remaining = [c.record.expression for c in session.detail("table.pdf").cards]
        assert remaining == ["走る", "飲む"]
    finally:
        server.shutdown()
        server.server_close()


def test_removal_deletes_only_the_removed_cards_lines(tmp_path: Path) -> None:
    """On this fixture every changed line is a deletion belonging to the
    removed card.

    Note what this does *not* prove. Any ruamel dump re-folds long plain
    scalars that PyYAML folded differently, so on real prose a write also
    re-wraps untouched text elsewhere in the file — see
    `test_a_bare_load_and_dump_already_reformats` in `tests/test_staging.py`.
    That is pre-existing and orthogonal to removal; this fixture's text is
    short enough never to fold, which is exactly why it isolates removal.
    """
    session = _staged(tmp_path)
    staging_path = tmp_path / "staging" / "table.pdf.yaml"
    before = staging_path.read_text(encoding="utf-8").split("\n")
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        _remove(server, session, "table.pdf", _remove_fields(page, 1))
        after = staging_path.read_text(encoding="utf-8").split("\n")
        changed = [
            line
            for line in difflib.unified_diff(before, after, lineterm="", n=0)
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]
        # Only deletions, and every one of them belongs to the removed card.
        assert changed, "nothing was removed"
        assert all(line.startswith("-") for line in changed), changed
        assert any("食べる" in line for line in changed)
        assert not any("走る" in line or "飲む" in line for line in changed)
    finally:
        server.shutdown()
        server.server_close()


def test_a_removal_without_the_session_csrf_is_refused(tmp_path: Path) -> None:
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        fields = _remove_fields(page, 1)
        fields["csrf"] = "forged"
        status, _headers, _body = _remove(server, session, "table.pdf", fields)
        assert status == 403
        assert len(session.detail("table.pdf").cards) == 3
    finally:
        server.shutdown()
        server.server_close()


def test_a_removal_from_a_stale_page_is_refused(tmp_path: Path) -> None:
    session = _staged(tmp_path)
    staging_path = tmp_path / "staging" / "table.pdf.yaml"
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        fields = _remove_fields(page, 1)
        staging_path.write_bytes(staging_path.read_bytes() + b"\n# concurrent edit\n")
        changed = staging_path.read_bytes()

        status, _headers, body = _remove(server, session, "table.pdf", fields)
        assert status == 409
        assert b"Nothing was removed" in body
        assert staging_path.read_bytes() == changed
    finally:
        server.shutdown()
        server.server_close()


def test_a_removal_naming_a_card_that_is_not_there_is_refused(
    tmp_path: Path,
) -> None:
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        fields = _remove_fields(page, 1)
        fields["card"] = "99"
        status, _headers, _body = _remove(server, session, "table.pdf", fields)
        assert status == 400
        assert len(session.detail("table.pdf").cards) == 3
    finally:
        server.shutdown()
        server.server_close()


def test_removal_says_it_cannot_be_undone(tmp_path: Path) -> None:
    """Corrections can be undone before saving; this cannot be undone at all,
    and re-reading the source would cost another paid call. The control says
    so rather than leaving someone to find out."""
    session = _staged(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "table.pdf"))
        text = page.decode()
        assert "Remove 走る (はしる) from this review" in text
        assert "cannot be undone" in text
        assert "another paid call" in text
    finally:
        server.shutdown()
        server.server_close()


# --- W2d: deliberate re-identification --------------------------------------


def _reidentify_fields(body: bytes, index: int) -> dict[str, str]:
    match = re.search(rf'<form id="ri{index}"[^>]*>(.*?)</form>', body.decode(), re.S)
    assert match, f"no re-identify form for card {index}"
    return dict(
        re.findall(r'<input type=hidden name=(\w+) value="([^"]*)">', match.group(1))
    )


def _reidentify(
    server: Any,
    session: WorkbenchSession,
    source: str,
    fields: dict[str, str],
) -> tuple[int, dict[str, str], bytes]:
    return _request(
        server,
        "POST",
        f"/{session.token}/source/{quote(source, safe='')}/reidentify",
        headers={
            "Host": server.expected_host,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        body=urlencode(list(fields.items())).encode(),
    )


def _held(tmp_path: Path) -> WorkbenchSession:
    _stage(tmp_path, "reading_holds", filename="lesson-9.pdf")
    return WorkbenchSession.open(ProjectConfig.load(tmp_path))


def test_the_first_submission_previews_and_writes_nothing(tmp_path: Path) -> None:
    """The whole point of the flow: see the consequence before causing it."""
    session = _held(tmp_path)
    staging_path = tmp_path / "staging" / "lesson-9.pdf.yaml"
    before = staging_path.read_bytes()
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        fields["expression"] = "泊まる"
        fields["reading"] = "とまる"

        status, _headers, body = _reidentify(server, session, "lesson-9.pdf", fields)

        assert status == 200
        text = body.decode()
        assert "word:泊まる:とまる" in text
        assert "Is this a different word?" in text
        assert staging_path.read_bytes() == before
    finally:
        server.shutdown()
        server.server_close()


def test_confirming_the_previewed_identity_applies_it(tmp_path: Path) -> None:
    session = _held(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        fields["expression"] = "泊まる"
        fields["reading"] = "とまる"
        fields["confirm"] = "word:泊まる:とまる"

        status, headers, _body = _reidentify(server, session, "lesson-9.pdf", fields)

        assert status == 303
        assert "reidentified=1" in headers["location"]
        ids = [c.record.id for c in session.detail("lesson-9.pdf").cards]
        assert ids == ["word:泊まる:とまる", "word:走る:わしる"]
    finally:
        server.shutdown()
        server.server_close()


def test_a_confirmation_for_a_different_identity_only_previews(
    tmp_path: Path,
) -> None:
    """The confirmation is bound to the exact identity the preview showed. A
    mismatch means the form drifted between the page someone read and the
    change they authorised, so it shows again rather than writing."""
    session = _held(tmp_path)
    staging_path = tmp_path / "staging" / "lesson-9.pdf.yaml"
    before = staging_path.read_bytes()
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        fields["expression"] = "泊まる"
        fields["reading"] = "とまる"
        fields["confirm"] = "word:止まる:とまる"  # not what the preview computed

        status, _headers, _body = _reidentify(server, session, "lesson-9.pdf", fields)

        assert status == 200
        assert staging_path.read_bytes() == before
    finally:
        server.shutdown()
        server.server_close()


def test_a_collision_is_shown_and_refused(tmp_path: Path) -> None:
    """Two cards cannot claim one word. The preview says so and offers no
    confirm button at all, rather than letting the write fail later."""
    session = _held(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        # card 1 is already 走る (わしる)
        fields["expression"] = "走る"
        fields["reading"] = "わしる"

        status, _headers, body = _reidentify(server, session, "lesson-9.pdf", fields)
        text = body.decode()

        assert status == 200
        assert "Already this exact word" in text
        assert "Two cards cannot" in text
        assert "name=confirm" not in text

        # ...and even a forced confirmation is refused.
        fields["confirm"] = "word:走る:わしる"
        status, _headers, _body = _reidentify(server, session, "lesson-9.pdf", fields)
        assert status == 409
        ids = [c.record.id for c in session.detail("lesson-9.pdf").cards]
        assert ids == ["word:泊まる:", "word:走る:わしる"]
    finally:
        server.shutdown()
        server.server_close()


def test_an_exported_card_is_warned_about_review_history(tmp_path: Path) -> None:
    """The Anki GUID derives from the record ID, so a card that already
    shipped becomes a *different* note under a new identity. Someone deciding
    this needs to know their review history stays with the old one."""
    session = _held(tmp_path)
    (tmp_path / "ledger.json").write_text(
        json.dumps(
            {
                "version": 1,
                "pending_batches": {},
                "records": {
                    "word:泊まる:": {
                        "added_at": "2026-08-01",
                        "exports": {"week-3": {"at": "2026-08-02"}},
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        fields["expression"] = "泊まる"
        fields["reading"] = "とまる"
        _status, _headers, body = _reidentify(server, session, "lesson-9.pdf", fields)
        text = body.decode()
        assert "already been built into a deck" in text
        assert "no review history" in text
    finally:
        server.shutdown()
        server.server_close()


def test_a_blank_reading_cannot_become_an_identity(tmp_path: Path) -> None:
    """A blank reading is exactly what the promote gate holds a row back for.
    Minting a permanent ID from one would walk into that refusal with the ID
    already written."""
    session = _held(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        fields["expression"] = "泊まる"
        fields["reading"] = "   "
        status, _headers, body = _reidentify(server, session, "lesson-9.pdf", fields)
        assert status == 400
        assert b"reading" in body
    finally:
        server.shutdown()
        server.server_close()


def test_re_identification_never_proposes_an_identity(tmp_path: Path) -> None:
    """Deciding a kana spelling "should" be particular kanji is reading
    Japanese, which this project reserves for the model and the person. The
    form is prefilled with what the card already says, never a guess."""
    session = _held(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        assert fields["expression"] == "泊まる"
        assert fields["reading"] == ""  # the card's own blank reading, not a guess
    finally:
        server.shutdown()
        server.server_close()


def test_re_identification_leaves_approval_and_sentences_alone(
    tmp_path: Path,
) -> None:
    """Approval covers the exact Japanese sentences, and those do not change
    when the card's identity does."""
    session = _held(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "lesson-9.pdf")
        )
        _approve(
            server,
            session,
            "lesson-9.pdf",
            fields=_form_fields(page),
            records=["word:泊まる:"],
        )
        before = session.detail("lesson-9.pdf").cards[0]
        assert before.authority == "existing"
        sentences = [e.japanese for e in before.record.examples]

        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        fields["expression"] = "泊まる"
        fields["reading"] = "とまる"
        fields["confirm"] = "word:泊まる:とまる"
        status, _headers, _body = _reidentify(server, session, "lesson-9.pdf", fields)
        assert status == 303

        after = session.detail("lesson-9.pdf").cards[0]
        assert after.record.id == "word:泊まる:とまる"
        assert [e.japanese for e in after.record.examples] == sentences
        assert after.authority == "existing"
    finally:
        server.shutdown()
        server.server_close()


def test_a_reidentification_without_the_session_csrf_is_refused(
    tmp_path: Path,
) -> None:
    session = _held(tmp_path)
    staging_path = tmp_path / "staging" / "lesson-9.pdf.yaml"
    before = staging_path.read_bytes()
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        fields["expression"] = "泊まる"
        fields["reading"] = "とまる"
        fields["confirm"] = "word:泊まる:とまる"
        fields["csrf"] = "forged"

        status, _headers, _body = _reidentify(server, session, "lesson-9.pdf", fields)

        assert status == 403
        assert staging_path.read_bytes() == before
    finally:
        server.shutdown()
        server.server_close()


def test_a_reidentification_from_a_stale_page_is_refused(tmp_path: Path) -> None:
    """The identity someone confirmed was computed against specific bytes. If
    the file moved under them, the card at that index may not be the card they
    were looking at."""
    session = _held(tmp_path)
    staging_path = tmp_path / "staging" / "lesson-9.pdf.yaml"
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        fields["expression"] = "泊まる"
        fields["reading"] = "とまる"
        fields["confirm"] = "word:泊まる:とまる"

        staging_path.write_bytes(staging_path.read_bytes() + b"\n# concurrent edit\n")
        changed = staging_path.read_bytes()

        status, _headers, body = _reidentify(server, session, "lesson-9.pdf", fields)

        assert status == 409
        assert b"Nothing was changed" in body
        assert staging_path.read_bytes() == changed
    finally:
        server.shutdown()
        server.server_close()


def test_a_reidentification_carrying_an_unknown_field_is_refused(
    tmp_path: Path,
) -> None:
    """`confirm` is the only optional field. Anything else means the form is
    not the one this page rendered, and guessing which parts to honour is how
    a field nobody offered becomes a field somebody can set."""
    session = _held(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        fields["expression"] = "泊まる"
        fields["reading"] = "とまる"
        fields["meanings"] = "sneaked in"

        status, _headers, body = _reidentify(server, session, "lesson-9.pdf", fields)

        assert status == 400
        assert b"meanings" in body
        assert [c.record.id for c in session.detail("lesson-9.pdf").cards] == [
            "word:泊まる:",
            "word:走る:わしる",
        ]
    finally:
        server.shutdown()
        server.server_close()


def test_re_identifying_into_a_word_you_already_own_is_allowed(
    tmp_path: Path,
) -> None:
    """The most useful thing this flow does: realising a staged card *is* the
    word already in your collection, and saying so. Promote's existing-wins
    merge is built for exactly that, so refusing it would block the case the
    feature exists for — even though the preview said it would merge."""
    session = _held(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            [
                {
                    "id": "word:泊まる:とまる",
                    "expression": "泊まる",
                    "reading": "とまる",
                    "meanings": ["to stay the night (already curated)"],
                    "examples": [],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", _edit_url(session, "lesson-9.pdf"))
        fields = _reidentify_fields(page, 0)
        fields["expression"] = "泊まる"
        fields["reading"] = "とまる"

        _status, _headers, preview = _reidentify(server, session, "lesson-9.pdf", fields)
        text = preview.decode()
        assert "would merge into that record" in text
        assert "name=confirm" in text, "the merge case must be confirmable"

        fields["confirm"] = "word:泊まる:とまる"
        status, _headers, _body = _reidentify(server, session, "lesson-9.pdf", fields)

        assert status == 303
        ids = [c.record.id for c in session.detail("lesson-9.pdf").cards]
        assert ids == ["word:泊まる:とまる", "word:走る:わしる"]
        # ...and the page now shows the merge it will make.
        _status, _headers, page = _request(
            server, "GET", _source_url(session, "lesson-9.pdf")
        )
        assert b"already curated" in page
    finally:
        server.shutdown()
        server.server_close()


# --- W3b: adding a source to the corpus -------------------------------------


def _multipart(
    filename: str | None, data: bytes, fields: dict[str, str], boundary: str = "ZZZ"
) -> bytes:
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
        f"{value}\r\n".encode()
        for name, value in fields.items()
    ]
    if filename is not None:
        # A quoted-string escapes `\\` and `"`. Sending them raw is malformed,
        # and the parser then eats the backslashes — see the test below.
        quoted = filename.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f'filename="{quoted}"\r\nContent-Type: application/pdf\r\n\r\n'.encode()
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts)


def _add_source(
    server: Any,
    session: WorkbenchSession,
    filename: str | None,
    data: bytes = b"%PDF-1.7 body",
    *,
    csrf: str | None = None,
    boundary: str = "ZZZ",
    content_type: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    body = _multipart(
        filename, data, {"csrf": session.csrf_token if csrf is None else csrf}, boundary
    )
    return _request(
        server,
        "POST",
        f"/{session.token}/add-source",
        headers={
            "Host": server.expected_host,
            "Content-Type": content_type
            or f'multipart/form-data; boundary="{boundary}"',
        },
        body=body,
    )


def _empty_project(tmp_path: Path) -> WorkbenchSession:
    _project(tmp_path)
    return WorkbenchSession.open(ProjectConfig.load(tmp_path))


def test_adding_a_source_copies_it_into_the_inbox(tmp_path: Path) -> None:
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        status, headers, _body = _add_source(server, session, "Genki Lesson 8.pdf")

        assert status == 303
        assert "added=1" in headers["location"]
        assert [p.name for p in (tmp_path / "inbox").iterdir()] == [
            "Genki Lesson 8.pdf"
        ]
    finally:
        server.shutdown()
        server.server_close()


def test_adding_a_source_sends_nothing_to_a_provider(tmp_path: Path) -> None:
    """The rule the whole intake design rests on: adding a file to the corpus
    and sending it to a model are two separate actions, always. The control
    says so, and there is no provider call behind this route to contradict it."""
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        _status, _headers, page = _request(server, "GET", f"/{session.token}/")
        text = page.decode()
        assert "sends it\nnowhere" in text or "sends it nowhere" in text
        assert "separate, paid step you choose" in text
        # The dashboard offers no way to spend money.
        assert "add-source" in text
        assert "extract" not in text.lower().replace("janki extract", "")
    finally:
        server.shutdown()
        server.server_close()


def test_the_same_file_twice_changes_nothing(tmp_path: Path) -> None:
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        _add_source(server, session, "lesson.pdf", b"%PDF-1.7 one")
        status, headers, _body = _add_source(
            server, session, "lesson.pdf", b"%PDF-1.7 one"
        )

        assert status == 303
        assert "added=0" in headers["location"]
        assert len(list((tmp_path / "inbox").iterdir())) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_one_name_cannot_mean_two_different_sources(tmp_path: Path) -> None:
    """The inbox is immutable, so a second file under a taken name is refused
    in words a person can act on rather than silently overwriting evidence."""
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        _add_source(server, session, "lesson-3.pdf", b"%PDF-1.7 one")
        status, _headers, body = _add_source(
            server, session, "lesson-3.pdf", b"%PDF-1.7 something else"
        )

        assert status == 409
        assert b"already in your corpus" in body
        assert b"Rename this file" in body
        assert (tmp_path / "inbox" / "lesson-3.pdf").read_bytes() == b"%PDF-1.7 one"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("sent", "lands_as"),
    [
        # A browser sends whatever the OS called the file, and some send a
        # full path. Everything before the last separator is discarded, so
        # these are ordinary uploads that succeed under a plain basename...
        ("C:\\Users\\me\\lesson.pdf", "lesson.pdf"),
        ("/home/me/scans/week-3.pdf", "week-3.pdf"),
        ("../../escape.pdf", "escape.pdf"),
        ("sub/dir/x.pdf", "x.pdf"),
    ],
)
def test_an_uploaded_name_lands_as_a_plain_basename(
    tmp_path: Path, sent: str, lands_as: str
) -> None:
    """...and none of them can choose where the file goes. The name is about to
    become a path under `data/inbox/`, so it is reduced to something that
    cannot contain a separator at all."""
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, _body = _add_source(server, session, sent)

        assert status == 303, "a full path is an ordinary upload, not an error"
        assert [p.name for p in (tmp_path / "inbox").rglob("*")] == [lands_as]
        assert not (tmp_path / "escape.pdf").exists()
        assert not (tmp_path.parent / "escape.pdf").exists()
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("hostile", ["", ".", "..", ".hidden.pdf", "a\x00b.pdf"])
def test_a_name_that_cannot_be_a_basename_is_refused(
    tmp_path: Path, hostile: str
) -> None:
    """The safety net under the basename reduction: anything that still could
    not be a plain filename is refused rather than guessed at."""
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, _body = _add_source(server, session, hostile)
        assert status in {400, 409}
        assert list((tmp_path / "inbox").iterdir()) == []
    finally:
        server.shutdown()
        server.server_close()


def test_an_upload_without_the_session_csrf_is_refused(tmp_path: Path) -> None:
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, _body = _add_source(
            server, session, "lesson.pdf", csrf="forged"
        )

        assert status == 403
        assert list((tmp_path / "inbox").iterdir()) == []
    finally:
        server.shutdown()
        server.server_close()


def test_an_upload_with_no_file_is_refused(tmp_path: Path) -> None:
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, body = _add_source(server, session, None)
        assert status == 400
        assert b"no file" in body
    finally:
        server.shutdown()
        server.server_close()


def test_an_unsupported_file_type_is_refused_by_name(tmp_path: Path) -> None:
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, body = _add_source(server, session, "notes.txt")
        assert status == 409
        assert b"cannot read" in body
        assert list((tmp_path / "inbox").iterdir()) == []
    finally:
        server.shutdown()
        server.server_close()


def test_an_upload_that_is_not_multipart_is_refused(tmp_path: Path) -> None:
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, _body = _add_source(
            server,
            session,
            "lesson.pdf",
            content_type="application/x-www-form-urlencoded",
        )
        assert status == 415
    finally:
        server.shutdown()
        server.server_close()


def test_an_oversized_upload_is_refused_before_it_is_read(tmp_path: Path) -> None:
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        status, _headers, _body = _request(
            server,
            "POST",
            f"/{session.token}/add-source",
            headers={
                "Host": server.expected_host,
                "Content-Type": 'multipart/form-data; boundary="Z"',
                "Content-Length": str(64 * 1024 * 1024 + 1),
            },
            body=b"x" * 16,
        )
        assert status == 413
    finally:
        server.shutdown()
        server.server_close()


def test_an_added_source_appears_on_the_dashboard_as_unread(tmp_path: Path) -> None:
    """The point of the whole slice: the source is in the corpus and the
    dashboard says plainly that nothing has read it yet."""
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        _add_source(server, session, "week-3.pdf")
        _status, _headers, page = _request(server, "GET", f"/{session.token}/")
        text = page.decode()
        assert "week-3.pdf" in text
        assert NOT_EXTRACTED in text
    finally:
        server.shutdown()
        server.server_close()


def test_a_sender_that_does_not_escape_its_filename_still_lands_safely(
    tmp_path: Path,
) -> None:
    """A backslash inside a quoted-string is an escape character. A sender that
    puts a raw Windows path in `filename="..."` is malformed, and the parser
    eats the separators before janki ever sees them — so the stored name is
    ugly rather than a path. Recorded because "why is my file called
    `C:Usersmelesson.pdf`" has an answer, and because the answer is *safe*:
    what arrives cannot contain a separator at all."""
    session = _empty_project(tmp_path)
    server, _thread = _running(session)
    try:
        raw = (
            b'--Z\r\nContent-Disposition: form-data; name="csrf"\r\n\r\n'
            + session.csrf_token.encode()
            + b'\r\n--Z\r\nContent-Disposition: form-data; name="file"; '
            b'filename="C:\\Users\\me\\lesson.pdf"\r\n'
            b"Content-Type: application/pdf\r\n\r\n%PDF-1.7\r\n--Z--\r\n"
        )
        status, _headers, _body = _request(
            server,
            "POST",
            f"/{session.token}/add-source",
            headers={
                "Host": server.expected_host,
                "Content-Type": 'multipart/form-data; boundary="Z"',
            },
            body=raw,
        )

        assert status == 303
        landed = [p.name for p in (tmp_path / "inbox").iterdir()]
        assert landed == ["C:Usersmelesson.pdf"]
        assert all("/" not in name and "\\" not in name for name in landed)
    finally:
        server.shutdown()
        server.server_close()


# --- sources that did not come from an extraction ----------------------------
#
# A staging file written by `janki extract` carries a review run id, prompt
# provenance and the pattern answer that run proposed. Every *approval* binds
# to those: accepting an example means accepting the sentence a particular
# model proposed on a particular run.
#
# A file that arrived another way — an Anki import, a hand-written review — has
# none of that, and is not broken for lacking it. Refusing to open it at all
# took the whole page down with the approvals: no CSRF token, so no forms, so
# not even a link to the editor. A reading typo in an imported deck was
# unfixable in a tool whose whole job is fixing them.


def _imported(tmp_path: Path, *, source: str = "Reading Pack Vocab") -> Path:
    """A staging file with ordinary rows and no extraction lineage at all."""
    _project(tmp_path)
    path = tmp_path / "staging" / "reading-pack.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        path,
        [
            VocabularyRecord(
                id="word:出来るだけ:だきるだけ",
                expression="出来るだけ",
                reading="だきるだけ",
                meanings=["As much as possible"],
            )
        ],
        {"source_file": source, "extracted_at": "2026-08-24"},
    )
    return path


def test_a_source_with_no_extraction_lineage_still_opens(tmp_path: Path) -> None:
    session = WorkbenchSession.open(ProjectConfig.load(_imported(tmp_path).parent.parent))

    panel = session.panel("Reading Pack Vocab")

    assert panel is not None
    assert panel.has_extraction_lineage is False
    assert [record.expression for record in panel.records] == ["出来るだけ"]


def test_such_a_source_offers_the_controls_that_do_not_need_a_run(
    tmp_path: Path,
) -> None:
    """Editing a gloss and correcting a misread word are about the rows, not
    about which model proposed them. The page that refuses to open offers
    neither, which is how a one-character reading typo became unfixable."""
    _imported(tmp_path)
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    panel = session.panel("Reading Pack Vocab")
    assert panel is not None

    html = render_source(
        session.detail("Reading Pack Vocab"),
        token=session.token,
        csrf=session.csrf_token,
        staging_snapshot=panel.staging_fingerprint,
        patterns_snapshot=panel.patterns_fingerprint,
        editing=True,
    )

    assert "<textarea" in html
    assert "This is a different word than it says" in html
    assert "from this review" in html


def test_such_a_source_says_why_its_grammar_cannot_be_reviewed(
    tmp_path: Path,
) -> None:
    """Not silence, and not a bare disabled control: the reason is the useful
    part, because "no model answer" is a fact about the source rather than a
    fault the reader should go looking for."""
    _imported(tmp_path)
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    panel = session.panel("Reading Pack Vocab")
    assert panel is not None

    assert panel.pattern_reviewable is False
    assert "did not come from an extraction" in panel.pattern_warning


def test_an_approval_is_refused_on_a_source_with_no_run_to_bind_it_to(
    tmp_path: Path,
) -> None:
    """The page never offers the control — but the request is what writes, and
    a request that arrives anyway must be refused rather than bound to a run
    that does not exist.

    The row has to be genuinely reviewable for this to test anything: an
    ineligible one is stopped by the earlier check and raises the same class
    for a different reason, which is how a missing guard would look guarded.
    """
    _project(tmp_path)
    path = tmp_path / "staging" / "reading-pack.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    reviewable = VocabularyRecord(
        id="word:出来るだけ:できるだけ",
        expression="出来るだけ",
        reading="できるだけ",
        meanings=["As much as possible"],
        # Extract-sourced with an unapproved example: exactly the shape whose
        # sentences a person would normally be asked to accept.
        source=SourceReference(type="extract", imported_from="pack.pdf"),
        examples=[ExampleSentence(japanese="出来るだけ早く来て。")],
    )
    write_staging(path, [reviewable], {"source_file": "Reading Pack Vocab"})
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    panel = session.panel("Reading Pack Vocab")
    assert panel is not None
    assert reviewable.id in panel.reviewable_record_ids, "the row must be approvable"

    with pytest.raises(review.PanelRequestError, match="no proposal to approve"):
        panel.submit(record_ids=[reviewable.id], review_patterns=False)


def test_a_malformed_lineage_is_still_refused(tmp_path: Path) -> None:
    """Absent is not invalid. A file that *claims* a coverage block and gets it
    wrong is broken, and opening it would show a review bound to nonsense."""
    path = _imported(tmp_path)
    records, meta = read_staging(path)
    meta["coverage"] = {"version": 2, "status": "not-a-status"}
    write_staging(path, records, meta, force=True)
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))

    assert session.panel("Reading Pack Vocab") is None


def test_a_path_shaped_source_name_is_refused_without_lineage_too(
    tmp_path: Path,
) -> None:
    """The source name is a pattern-store key. A path-shaped one escapes the
    store, and that is true however the rows arrived — so the check cannot sit
    behind the lineage branch."""
    _imported(tmp_path, source="../elsewhere/vocab")
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))

    assert session.panel("../elsewhere/vocab") is None


def test_a_project_that_never_extracted_can_still_review(tmp_path: Path) -> None:
    """A pattern store is created by extraction, so a project that has only
    ever imported a deck has none — and requiring one refused every review on
    exactly the projects most likely to need one. `patterns.load_store` already
    reads a missing store as empty; this matches it."""
    _imported(tmp_path)
    store = tmp_path / "patterns.json"
    if store.exists():
        store.unlink()

    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    panel = session.panel("Reading Pack Vocab")

    assert panel is not None
    assert panel.pattern_set is None


def test_a_pattern_store_that_exists_is_still_held_to_the_rule(
    tmp_path: Path,
) -> None:
    """Absent is allowed; a directory or a symlink where the store should be is
    not. Relaxing the missing case must not relax the rest."""
    _imported(tmp_path)
    store = tmp_path / "patterns.json"
    if store.exists():
        store.unlink()
    store.mkdir()

    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))

    assert session.panel("Reading Pack Vocab") is None


def test_both_card_actions_render_in_one_row(tmp_path: Path) -> None:
    """They are siblings, and were laid out as unrelated blocks: a full-width
    removal button that read as a disabled field, above a re-identify control
    with no rule at all that wrapped into its own caption. The grouping is what
    makes them read as two choices about the same card."""
    session = _staged(tmp_path)
    panel = session.panel("table.pdf")
    assert panel is not None

    html = render_source(
        session.detail("table.pdf"),
        token=session.token,
        csrf=session.csrf_token,
        staging_snapshot=panel.staging_fingerprint,
        patterns_snapshot=panel.patterns_fingerprint,
        editing=True,
    )

    assert '<div class="actions">' in html
    # One container per card, holding both controls.
    assert html.count('<div class="actions">') == len(session.detail("table.pdf").cards)
    assert html.count('<div class="reidentify">') == html.count('<div class="remove">')


def test_every_control_shows_that_it_is_a_control(tmp_path: Path) -> None:
    """A button that looks identical whether or not the pointer is over it
    reads as decoration — the removal control was reported as inert on that
    basis alone, while being perfectly functional. The states are asserted
    because losing them is invisible in every other test here."""
    del tmp_path
    # Rule *bodies*, not selectors. `cursor: pointer`, `:focus-visible` and
    # `prefers-reduced-motion` all appear elsewhere in this stylesheet already,
    # so asserting the bare substrings passed whether or not a button ever
    # gained a state.
    assert "button:hover { background: Highlight" in STYLE
    assert "button:active { transform:" in STYLE
    assert "cursor: pointer; border-radius:" in STYLE
    assert "button { transition: none; }" in STYLE


def _enrich_review(tmp_path: Path, records: list[VocabularyRecord]) -> Path:
    """A staging file shaped like one `enrich --ai` writes."""
    _project(tmp_path)
    path = tmp_path / "staging" / "ai-enrichment.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [record.id for record in records]
    write_staging(
        path,
        records,
        {
            "source_file": "vocabulary.json",
            "model": "claude-opus-5",
            "provider": "anthropic",
            "review_run_id": str(uuid.uuid4()),
            "ai_enrichment": {
                "version": 1,
                "model": "claude-opus-5",
                "provider": "anthropic",
                "request_fingerprints": {one: "a" * 64 for one in ids},
                "input_fingerprints": {one: "b" * 64 for one in ids},
                "fields": {one: ["meanings"] for one in ids},
            },
            # An `ai_enrichment` block is incomplete without this, and it is
            # keyed on record id too — so it is one of the maps a departing
            # row has to take with it.
            "field_replacements": promote.field_replacement_block(
                records,
                {
                    record.id: {
                        "meanings": (list(record.meanings), ["a new gloss"])
                    }
                    for record in records
                },
            ),
        },
    )
    return path


def test_a_paid_model_review_is_its_own_kind_of_source(tmp_path: Path) -> None:
    """`enrich --ai` writes a run id and an `ai_enrichment` block and never
    prompt provenance, so it is neither an extraction nor an import. Its rows
    are ordinary enough to edit and discard; what it cannot do is approve
    examples it never proposed, or claim the model spoke about a word it was
    never shown."""
    _enrich_review(
        tmp_path,
        [VocabularyRecord(id="word:話す:はなす", expression="話す", reading="はなす",
                          meanings=["to speak"])],
    )
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))

    panel = session.panel("vocabulary.json")

    assert panel is not None
    assert panel.provenance_kind == "model-pass"
    assert panel.has_extraction_lineage is False
    assert panel.reidentifiable is False


def test_removing_a_row_takes_its_provenance_with_it(tmp_path: Path) -> None:
    """The reason this file kind was unopenable at all.

    promote demands the provenance maps name exactly the rows the file holds
    plus the rows its archive holds. A row that leaves on its own becomes an
    *unknown* entry, and the next promote refuses the whole file — every other
    paid answer in it included. So the maps go where the row goes.
    """
    kept = VocabularyRecord(id="word:話す:はなす", expression="話す", reading="はなす",
                            meanings=["to speak"])
    doomed = VocabularyRecord(id="word:走る:はしる", expression="走る", reading="はしる",
                              meanings=["to run"])
    path = _enrich_review(tmp_path, [kept, doomed])
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    panel = session.panel("vocabulary.json")
    assert panel is not None

    text = staging.render_staging_prune(
        path.read_bytes(),
        [True, False],
        source=str(path),
    )
    assert text is not None
    path.write_text(text, encoding="utf-8")

    _records, meta = read_staging(path)
    block = meta["ai_enrichment"]
    for name in ("request_fingerprints", "input_fingerprints", "fields"):
        assert list(block[name]) == [kept.id], name

    # And promote's own gate agrees, which is the claim that matters.
    staged_ai_enrichment(meta, [kept.id])


def test_a_model_pass_row_cannot_be_told_it_is_a_different_word(
    tmp_path: Path,
) -> None:
    """The answer was produced for the word it was asked about, and its input
    fingerprint binds that turn. Re-keying it would record a model saying
    something it never said about a word it never saw — so the page does not
    offer it, and the request is refused because the page is not what writes.
    """
    path = _enrich_review(
        tmp_path,
        [VocabularyRecord(id="word:話す:はなす", expression="話す", reading="はなす",
                          meanings=["to speak"])],
    )
    del path
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    panel = session.panel("vocabulary.json")
    assert panel is not None
    detail = session.detail("vocabulary.json")

    offered = render_source(
        detail, token=session.token, csrf=session.csrf_token,
        staging_snapshot=panel.staging_fingerprint,
        patterns_snapshot=panel.patterns_fingerprint,
        editing=True,
        approvable=panel.has_extraction_lineage,
        reidentifiable=panel.reidentifiable,
    )
    assert "This is a different word than it says" not in offered
    # ...but it is still editable, which is the point of opening it at all.
    assert "<textarea" in offered
    assert "from this review" in offered

    server, _thread = _running(session)
    try:
        status, _headers, payload = _request(
            server, "POST", f"/{session.token}/source/vocabulary.json/reidentify",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=urlencode({
                "action": "reidentify", "csrf": session.csrf_token,
                "staging_snapshot": panel.staging_fingerprint, "card": "0",
                "expression": "話す", "reading": "はなす",
            }).encode("utf-8"),
        )
    finally:
        server.shutdown()

    assert status == 400
    assert b"cannot be moved to a different word" in payload
def test_a_page_that_cannot_approve_does_not_offer_approval(
    tmp_path: Path,
) -> None:
    """The guard refusing the write is not the same as the page not asking for
    it. An extract-typed row with an unapproved example renders a checkbox and
    a save button on any source — and on one with no model run every such
    submission is refused, which is the dead-end control this page already
    forbids for grammar review.
    """
    _project(tmp_path)
    path = tmp_path / "staging" / "reading-pack.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        path,
        [
            VocabularyRecord(
                id="word:走る:はしる", expression="走る", reading="はしる",
                meanings=["to run"],
                source=SourceReference(type="extract", imported_from="pack.pdf"),
                examples=[ExampleSentence(japanese="毎日走ります。")],
            )
        ],
        {"source_file": "Reading Pack Vocab"},
    )
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    panel = session.panel("Reading Pack Vocab")
    assert panel is not None
    detail = session.detail("Reading Pack Vocab")
    assert any(card.needs_example_review for card in detail.cards)

    offered = render_source(
        detail, token=session.token, csrf=session.csrf_token,
        staging_snapshot=panel.staging_fingerprint,
        patterns_snapshot=panel.patterns_fingerprint,
        approvable=panel.has_extraction_lineage,
    )

    assert "Save the approvals" not in offered
    assert 'type=checkbox' not in offered


def test_an_approval_lands_on_a_project_with_no_pattern_store(
    tmp_path: Path,
) -> None:
    """`open` accepts an absent store; `submit` used to refuse one, and said
    "a review target path changed after this page was rendered" — about a file
    that never existed and never changed. A card approval writes the staging
    file alone, so the store's absence is nothing to do with it."""
    _stage(tmp_path, "table_exhaustive", filename="table.pdf")
    store = tmp_path / "patterns.json"
    if store.exists():
        store.unlink()
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    panel = session.panel("table.pdf")
    assert panel is not None
    chosen = sorted(panel.reviewable_record_ids)[:1]
    assert chosen

    outcome = panel.submit(record_ids=chosen, review_patterns=False)

    assert outcome.accepted_record_ids == tuple(chosen)


def test_the_served_page_offers_controls_for_an_imported_source(
    tmp_path: Path,
) -> None:
    """Through the server, not the renderer. The bug that started this was the
    *wiring* — a refused panel meant an empty CSRF, and the renderer then
    emitted nothing at all, including the link to the editor. Every other test
    here calls `render_source` directly and would have passed throughout."""
    _imported(tmp_path)
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    server, _thread = _running(session)
    try:
        _status, _headers, read_view = _request(
            server, "GET", f"/{session.token}/source/Reading%20Pack%20Vocab"
        )
        _status, _headers, edit_view = _request(
            server, "GET", f"/{session.token}/source/Reading%20Pack%20Vocab?edit=1"
        )
        page = read_view.decode("utf-8")
        editor = edit_view.decode("utf-8")
    finally:
        server.shutdown()

    assert "edit=1" in page, "the read view must offer the way in"
    assert "<textarea" in editor
    assert "<form" in editor


def test_the_served_page_omits_approvals_for_a_source_that_cannot_approve(
    tmp_path: Path,
) -> None:
    """Through the server, with a row that *would* be offered for approval on
    an extraction source. Without such a row the page omits the controls for
    the ordinary reason and proves nothing about the wiring."""
    _project(tmp_path)
    path = tmp_path / "staging" / "reading-pack.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        path,
        [
            VocabularyRecord(
                id="word:走る:はしる", expression="走る", reading="はしる",
                meanings=["to run"],
                source=SourceReference(type="extract", imported_from="pack.pdf"),
                examples=[ExampleSentence(japanese="毎日走ります。")],
            )
        ],
        {"source_file": "Reading Pack Vocab"},
    )
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))
    panel = session.panel("Reading Pack Vocab")
    assert panel is not None
    assert panel.reviewable_record_ids, "the row must be one approval would offer"
    server, _thread = _running(session)
    try:
        _status, _headers, served = _request(
            server, "GET", f"/{session.token}/source/Reading%20Pack%20Vocab"
        )
    finally:
        server.shutdown()
    page = served.decode("utf-8")

    assert "Save the approvals" not in page
    assert "type=checkbox" not in page
    # ...but the page is still a working editor.
    assert "edit=1" in page


def test_promoting_a_row_leaves_its_provenance_where_the_archive_can_use_it(
    tmp_path: Path,
) -> None:
    """The other prune, pruning for the opposite reason.

    `prune_staging` drops rows promote has just *archived*. Their provenance
    must stay: the archive still holds those rows, promote's gate accepts
    archive-only ids, and `already_landed_staged_fields` reads the replacement
    entries to tell a field that already landed from one that did not. Only
    the workbench's prune — which throws a row away — takes the provenance
    with it.
    """
    kept = VocabularyRecord(id="word:話す:はなす", expression="話す", reading="はなす",
                            meanings=["to speak"])
    landed = VocabularyRecord(id="word:走る:はしる", expression="走る", reading="はしる",
                              meanings=["to run"])
    path = _enrich_review(tmp_path, [kept, landed])

    removed = staging.prune_staging(path, [True, False])

    assert removed == 1
    _records, meta = read_staging(path)
    block = meta["ai_enrichment"]
    for name in ("request_fingerprints", "input_fingerprints", "fields"):
        assert sorted(block[name]) == sorted([kept.id, landed.id]), name
    assert sorted(meta["field_replacements"]["records"]) == sorted(
        [kept.id, landed.id]
    )
    # And the gate accepts it, naming the landed row as archived.
    staged_ai_enrichment(meta, [kept.id], archived_ids=[landed.id])


def test_the_workbench_knows_what_its_collection_is_called(tmp_path: Path) -> None:
    """An `enrich --ai` review written before those files carried a marker
    names the collection as its source, and that is the only thing telling it
    apart from an import.

    The session has to hand that name to the panel. Without it the page offers
    "This is a different word than it says" over a model's answer, and moving
    it records that the model spoke about a word it was never shown.
    """
    _project(tmp_path)
    path = tmp_path / "staging" / "ai-enrichment.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_staging(
        path,
        [
            VocabularyRecord(
                id="word:走る:はしる",
                expression="走る",
                reading="はしる",
                meanings=["to run"],
                source=SourceReference(type="ai", imported_from="vocabulary.json"),
            )
        ],
        {"source_file": "vocabulary.json", "model": "claude-opus-5"},
    )
    session = WorkbenchSession.open(ProjectConfig.load(tmp_path))

    panel = session.panel("vocabulary.json")

    assert panel is not None
    assert panel.provenance_kind == "model-pass"
    assert panel.reidentifiable is False
