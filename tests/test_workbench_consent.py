"""W3: the page that asks whether to spend money, and says what on.

The consent GET is read-only by construction. Adding a file to the corpus and
sending it to a model are two separate actions, always — so arriving at this
page only *describes* a paid call, and the tests here are mostly about what it
says rather than what it computes. The wording is the feature:
somebody who believes a Claude Max subscription pays for this has not
consented to anything.

Nothing here contacts a provider. `extract_candidates` is replaced with a
function that fails the test if it is ever reached.
"""

from __future__ import annotations

import http.client
import threading
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

import pytest
from test_application_journey import _project, _stage

from conftest import seed_prompts
from japanese_anki import inputs, operations, patterns
from japanese_anki.application import busy_refusal, describe_extraction
from japanese_anki.config import ProjectConfig
from japanese_anki.staging import read_staging, write_staging
from japanese_anki.workbench import WorkbenchSession, make_server

PDF = b"%PDF-1.7 fake"


def _corpus(tmp_path: Path, name: str = "genki-8.pdf") -> Path:
    """A project with one source already in the corpus and nothing sent."""
    _project(tmp_path)
    seed_prompts(tmp_path)
    path = tmp_path / "inbox" / name
    path.write_bytes(PDF)
    return path


@pytest.fixture(autouse=True)
def _no_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing on this page may reach a provider, so make reaching one fail."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the consent page must not call a provider")

    monkeypatch.setattr("japanese_anki.extract.extract_candidates", refuse)


def _session(tmp_path: Path) -> WorkbenchSession:
    return WorkbenchSession.open(ProjectConfig.load(tmp_path))


def _get(session: WorkbenchSession, path: str) -> tuple[int, str]:
    server = make_server(session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        connection.close()
        return response.status, body
    finally:
        server.shutdown()
        server.server_close()


def _page(tmp_path: Path, name: str = "genki-8.pdf", query: str = "") -> str:
    session = _session(tmp_path)
    status, body = _get(
        session, f"/{session.token}/extract/{quote(name, safe='')}{query}"
    )
    assert status == 200, body
    return body


# --- what the page must say -------------------------------------------------


def test_the_page_names_the_file_the_model_and_the_charge(tmp_path: Path) -> None:
    """W3's sentence, and it has three parts. Which file leaves the computer,
    who answers it, and that answering costs money — a page missing any one of
    them is asking somebody to agree to something they cannot see."""
    _corpus(tmp_path)

    body = _page(tmp_path)

    assert "genki-8.pdf" in body
    assert "claude-opus-5" in body
    assert "paid API call" in body


def test_the_page_carries_every_disclosure(tmp_path: Path) -> None:
    """Each of these answers a different wrong belief somebody arrives with.
    They are separate sentences on the page for the same reason they are
    separate assertions here: a reader who skims one block absorbs none."""
    _corpus(tmp_path)

    body = _page(tmp_path)

    # Only this file — not "my corpus is being uploaded".
    assert "Only this one file is sent" in body
    # The copy stays. "Send" and "move" are the same word to some people.
    assert "Your copy stays where it is" in body
    # No page selection exists yet, so the page must not imply one.
    assert "The whole document is sent" in body
    assert "cannot yet send only some" in body
    # Whose money, and the subscription that is not it.
    assert "Anthropic API credits" in body
    assert "Max subscription is a different thing" in body


def test_the_page_says_nothing_has_been_sent(tmp_path: Path) -> None:
    """It describes a call. A person who lands here from a link must not be
    left wondering whether arriving already cost them something."""
    _corpus(tmp_path)

    assert "Nothing has been sent" in _page(tmp_path)


# --- the modes, in a learner's words ----------------------------------------


def test_the_mode_choice_defaults_to_letting_janki_decide(tmp_path: Path) -> None:
    """`table` and `prose` are janki's words for prompts, not descriptions of
    anybody's handout. The choice is advanced, collapsed, and defaulted."""
    _corpus(tmp_path)

    body = _page(tmp_path)

    assert "Choose automatically" in body
    assert "A vocabulary list" in body
    assert "A lesson, dialogue or exercise" in body
    # The internal words never reach the page as labels.
    assert ">table<" not in body
    assert ">prose<" not in body
    # Collapsed until somebody has chosen.
    assert "<details class=advanced>" in body


def test_each_label_is_bound_to_the_mode_it_names(tmp_path: Path) -> None:
    """The whole semantics of the control. Swap two labels and somebody picks
    "A vocabulary list", reads the plan beside it, and consents to a run
    described — and next, dispatched — under prose instructions. Asserting the
    labels exist and the values exist does not catch that; only the pairing
    does."""
    _corpus(tmp_path)

    body = _page(tmp_path)

    assert '"table"> A vocabulary list' in body
    assert '"prose"> A lesson, dialogue or exercise' in body
    assert '"" checked> Choose automatically' in body


def test_a_chosen_mode_is_kept_and_shown(tmp_path: Path) -> None:
    """Somebody who picked one must not have to find where their choice went."""
    _corpus(tmp_path)

    body = _page(tmp_path, query="?mode=table")

    assert 'value="table" checked' in body
    assert "<details class=advanced open>" in body


def test_a_mode_janki_does_not_know_is_ignored(tmp_path: Path) -> None:
    """An unrecognised mode must not reach `prompt_name` and select a prompt by
    accident: the mode decides which instructions are paid for."""
    _corpus(tmp_path)

    body = _page(tmp_path, query="?mode=../../etc/passwd")

    assert "Choose automatically" in body
    assert "passwd" not in body


# --- refusals ----------------------------------------------------------------


def test_a_file_that_is_not_in_the_corpus_is_refused(tmp_path: Path) -> None:
    """Adding and sending are separate actions. A consent page that copied its
    own subject in would have done the first while asking about the second."""
    _project(tmp_path)
    seed_prompts(tmp_path)
    outside = tmp_path / "elsewhere" / "secret.pdf"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(PDF)

    consent = describe_extraction(ProjectConfig.load(tmp_path), outside)

    assert not consent.sendable
    assert "not in your corpus yet" in consent.refusal
    # And it stayed out of the corpus.
    assert not (tmp_path / "inbox" / "secret.pdf").exists()


def test_a_source_the_corpus_does_not_have_is_a_404(tmp_path: Path) -> None:
    """The name is matched against janki's own view of the corpus, so a name
    that is not one of those has no page rather than a guessed path."""
    _corpus(tmp_path)
    session = _session(tmp_path)

    status, _body = _get(session, f"/{session.token}/extract/nothing.pdf")

    assert status == 404


def test_a_name_cannot_climb_out_of_the_inbox(tmp_path: Path) -> None:
    """The path comes from a match against the corpus listing, never from
    joining a request value onto a directory."""
    _corpus(tmp_path)
    session = _session(tmp_path)

    status, _body = _get(
        session, f"/{session.token}/extract/{quote('../janki.toml', safe='')}"
    )

    assert status == 404


# --- replacing an extraction someone has already reviewed --------------------


def test_replacing_a_review_names_what_it_would_destroy(tmp_path: Path) -> None:
    """"This will replace your review" is not a decision anybody can make.
    Which review, how much of it, and what state it had reached are the whole
    of the question."""
    _stage(tmp_path, "lesson_with_grammar", filename="genki-8.pdf")
    seed_prompts(tmp_path)
    (tmp_path / "inbox" / "genki-8.pdf").write_bytes(PDF)

    body = _page(tmp_path)

    assert "replaces what you already have" in body
    assert "2 cards" in body
    # The *state*, not only a count. "You will lose 2 cards" and "you will
    # lose 2 cards you have already reviewed the sentences for" are different
    # decisions, and the count alone cannot tell them apart.
    assert "Examples need review" in body
    assert "Grammar needs review" in body
    assert "not recoverable" in body


def test_replacing_reviewed_grammar_names_that_review_too(tmp_path: Path) -> None:
    """The replacement force applies to cards and patterns. Naming only the
    card track asks for less authority than completion exercises."""
    _stage(tmp_path, "lesson_with_grammar", filename="genki-8.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    store = patterns.load_store(config.patterns_file)
    store["genki-8.pdf"] = replace(store["genki-8.pdf"], reviewed=True)
    patterns.save_store(config.patterns_file, store)

    body = _page(tmp_path)

    assert "2 cards" in body
    assert "Examples need review" in body
    assert "Grammar reviewed" in body
    assert "not recoverable" in body


def test_an_unreviewed_empty_table_does_not_invent_a_grammar_review(
    tmp_path: Path,
) -> None:
    _stage(tmp_path, "table_exhaustive", filename="genki-8.pdf")
    seed_prompts(tmp_path)

    body = _page(tmp_path)

    assert "Grammar needs review" not in body
    assert "card and grammar reviews named above" not in body
    assert "the card review named above" in body


def test_stored_unreviewed_grammar_is_named_when_staging_does_not_badge_it(
    tmp_path: Path,
) -> None:
    """A nonempty store entry is still replacement scope even when the older
    staging copy no longer carries enough metadata for the dashboard badge."""
    _stage(tmp_path, "lesson_with_grammar", filename="genki-8.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    target = config.staging_dir / "genki-8.pdf.yaml"
    records, meta = read_staging(target)
    meta.pop("pattern_set")
    write_staging(target, records, meta, force=True)

    body = _page(tmp_path)

    assert "Grammar needs review" in body
    assert "card and grammar reviews named above" in body


@pytest.mark.parametrize("embedded_patterns", ["empty", "missing"])
def test_replacement_names_stored_grammar_even_when_staging_does_not_badge_it(
    tmp_path: Path,
    embedded_patterns: str,
) -> None:
    """Force replaces the source's store entry regardless of whether this
    staging generation has enough embedded pattern metadata for a badge."""
    _stage(tmp_path, "lesson_with_grammar", filename="genki-8.pdf")
    seed_prompts(tmp_path)
    config = ProjectConfig.load(tmp_path)
    target = config.staging_dir / "genki-8.pdf.yaml"
    records, meta = read_staging(target)
    if embedded_patterns == "empty":
        pattern_meta = dict(meta["pattern_set"])
        pattern_meta["patterns"] = []
        meta["pattern_set"] = pattern_meta
    else:
        meta.pop("pattern_set")
    write_staging(target, records, meta, force=True)
    store = patterns.load_store(config.patterns_file)
    store["genki-8.pdf"] = replace(
        store["genki-8.pdf"],
        patterns=(),
        reviewed=True,
    )
    patterns.save_store(config.patterns_file, store)

    body = _page(tmp_path)

    assert "Grammar reviewed" in body
    assert "card and grammar reviews named above" in body


def test_a_broken_symlink_cannot_be_offered_as_a_new_staging_target(
    tmp_path: Path,
) -> None:
    _corpus(tmp_path)
    target = tmp_path / "staging" / "genki-8.pdf.yaml"
    outside = tmp_path / "outside-review.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside)

    body = _page(tmp_path)

    assert "non-regular staging target" in body
    assert "<form method=post" not in body
    assert target.is_symlink()
    assert not outside.exists()


def test_a_first_reading_does_not_warn_about_a_replacement(tmp_path: Path) -> None:
    """The other half: a source nobody has read must not be dressed in a
    warning about losing work that does not exist."""
    _corpus(tmp_path)

    body = _page(tmp_path)

    assert "replaces what you already have" not in body


# --- one mutating job at a time ---------------------------------------------


def _authorize(tmp_path: Path, state: str, source: str = "other.pdf") -> None:
    config = ProjectConfig.load(tmp_path)
    journal = operations.OperationJournal.load(config.operations_file)
    journal.authorize(
        "op-1", kind="extract", source_file=source, source_sha256="a" * 64,
        request_fp="b" * 64, model="claude-opus-5",
    )
    for step in {
        "authorized": [],
        "dispatching": ["dispatching"],
        "running": ["dispatching", "running"],
        "result_captured": ["dispatching"],
    }[state]:
        journal.advance("op-1", step)
    if state == "result_captured":
        journal.capture_result(
            "op-1",
            lambda: operations.capture_artifact(
                config.operations_file, "op-1", b'{"content": []}'
            ),
        )


def test_a_call_in_flight_stops_another_from_starting(tmp_path: Path) -> None:
    """One mutating job at a time. The journal is the state of record, so a run
    started in a terminal blocks a browser click and the other way round —
    a guard that knew only its own process would let the two bill twice."""
    _corpus(tmp_path)
    _authorize(tmp_path, "dispatching")

    body = _page(tmp_path)

    assert "still marked as running" in body
    assert "other.pdf" in body
    assert "second charge" in body
    # And it says what to do when nothing is actually running, because a
    # killed process leaves this state and "wait" would be advice forever.
    assert "interrupted" in body
    assert "janki operations" in body


def test_a_call_reported_as_running_stops_another_from_starting(
    tmp_path: Path,
) -> None:
    """`running` is where a real dispatch sits longest — a caller that reports
    streaming moves through it. Guarding only `dispatching` would leave the
    page sendable through the whole middle of a paid call."""
    _corpus(tmp_path)
    _authorize(tmp_path, "running")

    assert busy_refusal(ProjectConfig.load(tmp_path))
    assert "still marked as running" in _page(tmp_path)


def test_an_outcome_nobody_knows_stops_another(tmp_path: Path) -> None:
    """The state a lost connection leaves. It is terminal and never
    auto-retried, so it blocks until a person deals with it — starting
    another call would be spending again on an unknown outcome."""
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    journal = operations.OperationJournal.load(config.operations_file)
    journal.authorize(
        "op-1", kind="extract", source_file="lost.pdf", source_sha256="a" * 64,
        request_fp="b" * 64, model="claude-opus-5",
    )
    journal.advance("op-1", "dispatching")
    journal.advance("op-1", "outcome_unknown", detail="connection lost")

    assert busy_refusal(config)
    assert "may have been billed" in _page(tmp_path)


def test_a_paid_answer_nobody_has_dealt_with_stops_another(tmp_path: Path) -> None:
    """A different refusal on purpose. A call in flight is a wait; an answer
    already paid for is a decision, and starting a second call buries it."""
    _corpus(tmp_path)
    _authorize(tmp_path, "result_captured")

    body = _page(tmp_path)

    # "may have been billed", never "has been paid for": `outcome_unknown`
    # sits in this same list precisely because nobody knows.
    assert "may have been billed" in body
    assert "still marked as running" not in body


def test_an_authority_that_never_sent_anything_is_named_as_that(
    tmp_path: Path,
) -> None:
    """It blocks — the gate has to count it, or two runs racing both pass —
    but it must not be described as a charge. Nothing was sent, and telling
    somebody their money may be gone when it demonstrably is not is the same
    class of lie as the reverse."""
    _corpus(tmp_path)
    _authorize(tmp_path, "authorized")

    assert busy_refusal(ProjectConfig.load(tmp_path))
    body = _page(tmp_path)
    assert "never sent anything" in body
    assert "still marked as running" not in body
    assert "may have been billed" not in body


def test_a_finished_run_blocks_nothing(tmp_path: Path) -> None:
    """The guard is about live work, not about history."""
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    journal = operations.OperationJournal.load(config.operations_file)
    journal.authorize(
        "op-1", kind="extract", source_file="done.pdf", source_sha256="a" * 64,
        request_fp="b" * 64, model="claude-opus-5",
    )
    journal.advance("op-1", "dispatching")
    journal.capture_result(
        "op-1",
        lambda: operations.capture_artifact(
            config.operations_file, "op-1", b'{"content": []}'
        ),
    )
    journal.advance("op-1", "committed")

    assert busy_refusal(config) == ""


# --- the dashboard's own link ------------------------------------------------


def test_the_dashboard_offers_the_consent_page_for_an_unread_source(
    tmp_path: Path,
) -> None:
    """And offers the page, never the call: the link goes somewhere that
    describes a charge rather than somewhere that makes one."""
    _corpus(tmp_path)
    session = _session(tmp_path)

    status, body = _get(session, f"/{session.token}/")

    assert status == 200
    assert f"/{session.token}/extract/genki-8.pdf" in body
    assert "See what reading this would send" in body


def test_the_dashboard_does_not_offer_to_re_read_a_source_under_review(
    tmp_path: Path,
) -> None:
    """A source with cards waiting is not offered a paid re-read from the
    dashboard: the next thing to do with it is the review, and the link would
    sit beside that advice contradicting it."""
    _stage(tmp_path, "table_exhaustive", filename="genki-8.pdf")
    seed_prompts(tmp_path)
    (tmp_path / "inbox" / "genki-8.pdf").write_bytes(PDF)
    session = _session(tmp_path)

    status, body = _get(session, f"/{session.token}/")

    assert status == 200
    assert "See what reading this would send" not in body


# --- the values a dispatch will trust ---------------------------------------


def test_a_describable_run_is_sendable_and_names_its_target(tmp_path: Path) -> None:
    """`sendable` is the gate the dispatch increment will ask, and `target`
    is what it would act on. Both are asserted positively here: a gate only
    ever tested on its refusals ships inverted and looks covered."""
    path = _corpus(tmp_path)

    consent = describe_extraction(ProjectConfig.load(tmp_path), path)

    assert consent.sendable is True
    assert consent.refusal == ""
    assert consent.busy == ""
    target = consent.target
    assert target is not None
    assert target.name == "genki-8.pdf"
    assert target.staging_path == tmp_path / "staging" / "genki-8.pdf.yaml"
    # The identity the journal would record, so a dispatch can prove the plan
    # it sends is the plan somebody was shown.
    assert target.provenance["request_fingerprint"]


def test_a_busy_janki_makes_a_fine_source_unsendable(tmp_path: Path) -> None:
    """`busy` has to gate `sendable` too. The source is fine; the moment is
    not — and a dispatch asking only about the source would spend anyway."""
    path = _corpus(tmp_path)
    _authorize(tmp_path, "dispatching")

    consent = describe_extraction(ProjectConfig.load(tmp_path), path)

    assert consent.busy
    assert consent.refusal == ""
    assert consent.sendable is False


# --- the page writes nothing -------------------------------------------------


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_rendering_the_page_leaves_the_repository_byte_identical(
    tmp_path: Path,
) -> None:
    """The headline property, asserted over the whole tree rather than over
    the one file somebody thought of. This page describes a paid call; a
    describe that journalled an operation or touched a staging file would be
    doing the thing it claims to be asking about."""
    _corpus(tmp_path)
    before = _tree(tmp_path)

    _page(tmp_path)
    _page(tmp_path, query="?mode=prose")

    assert _tree(tmp_path) == before


def test_rendering_a_replacement_page_does_not_touch_the_review(
    tmp_path: Path,
) -> None:
    """The riskiest version of the same property: this page plans as though
    forced, and forcing is what overwrites a staging file."""
    _stage(tmp_path, "table_exhaustive", filename="genki-8.pdf")
    seed_prompts(tmp_path)
    (tmp_path / "inbox" / "genki-8.pdf").write_bytes(PDF)
    before = _tree(tmp_path)

    body = _page(tmp_path)

    assert "replaces what you already have" in body
    assert _tree(tmp_path) == before


# --- the mode form actually submits ------------------------------------------


def test_the_mode_choice_is_a_form_that_can_be_submitted(tmp_path: Path) -> None:
    """Radios that submit nothing are a control that controls nothing — and
    here the cost is specific: somebody picks a kind, reads the plan beside
    it, and consents to a run described under instructions they did not
    choose."""
    _corpus(tmp_path)

    body = _page(tmp_path)

    start = body.index("<details class=advanced")
    end = body.index("</details>", start)
    mode_form = body[start:end]
    assert "<form method=get" in mode_form
    assert f"/extract/{quote('genki-8.pdf', safe='')}" in mode_form
    assert "Describe it as this kind" in mode_form
    # GET, because choosing how to describe a page changes nothing on disk.
    assert "method=post" not in mode_form.lower()


# --- what actually leaves the computer ---------------------------------------


def test_a_lesson_run_says_the_known_word_list_is_sent_too(tmp_path: Path) -> None:
    """"Only this one file" would otherwise be read as covering the skip
    list. Prose mode puts every expression already in the collection into the
    prompt, and a learner has no reason to distinguish their corpus from
    their collection."""
    _corpus(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        '[{"id": "word:走る:はしる", "expression": "走る", "reading": "はしる",'
        ' "meanings": ["to run"], "source": {"kind": "manual", "name": "seed"}}]',
        encoding="utf-8",
    )

    body = _page(tmp_path, query="?mode=prose")

    assert "the list of words you already have is sent too" in body


def test_a_table_run_does_not_claim_to_send_the_known_word_list(
    tmp_path: Path,
) -> None:
    """The other half. A table is transcribed row by row, so no skip list is
    sent — and saying otherwise would be an invented disclosure."""
    _corpus(tmp_path)
    (tmp_path / "vocabulary.json").write_text(
        '[{"id": "word:走る:はしる", "expression": "走る", "reading": "はしる",'
        ' "meanings": ["to run"], "source": {"kind": "manual", "name": "seed"}}]',
        encoding="utf-8",
    )

    body = _page(tmp_path, query="?mode=table")

    assert "the list of words you already have is sent too" not in body


# --- a damaged journal ------------------------------------------------------


def test_an_unreadable_journal_refuses_rather_than_killing_the_page(
    tmp_path: Path,
) -> None:
    """`data/operations.json` is exactly the file an interrupted paid call
    leaves in interesting shapes. The one page that manages paid-call state
    must not be the one that dies when it cannot be parsed."""
    _corpus(tmp_path)
    config = ProjectConfig.load(tmp_path)
    config.operations_file.parent.mkdir(parents=True, exist_ok=True)
    config.operations_file.write_text("{not json", encoding="utf-8")

    session = _session(tmp_path)
    status, body = _get(session, f"/{session.token}/extract/genki-8.pdf")

    # Not a crash, and not blamed on the source: the disclosures stay, and
    # the reason lands where "janki will not start a run right now" lives.
    assert status == 200
    assert "paid API call" in body
    assert "cannot tell whether a paid call is already running" in body


def test_a_sendable_page_offers_one_named_paid_action(tmp_path: Path) -> None:
    """The button is the consent: it repeats the file, model, purpose and
    charge rather than reducing the decision to an uninformative "Continue"."""
    _corpus(tmp_path)

    body = _page(tmp_path)

    assert "<form method=post" in body
    assert '<button type=submit name="dispatch"' in body
    assert '<input type=hidden name=dispatch' not in body
    assert "Send genki-8.pdf to claude-opus-5" in body
    assert "paid API call" in body
    assert "not built yet" not in body


def test_implicit_enter_targets_a_disabled_default_not_the_paid_action(
    tmp_path: Path,
) -> None:
    """Browser implicit submission clicks the first submit button. It must be
    disabled; intentional keyboard focus on the later paid button still works."""
    _corpus(tmp_path)

    body = _page(tmp_path)
    start = body.index("<form method=post")
    paid_form = body[start : body.index("</form>", start)]
    first = paid_form.index("<button")

    assert paid_form[first:].startswith(
        '<button type=submit disabled class="implicit-submit-guard"'
    )
    assert paid_form.index('name="dispatch"') > first


def test_a_page_that_cannot_send_does_not_offer_a_paid_action(tmp_path: Path) -> None:
    """A busy janki still explains the call but offers no POST capability."""
    _corpus(tmp_path)
    _authorize(tmp_path, "dispatching")

    body = _page(tmp_path)

    assert "<form method=post" not in body
    assert 'name="dispatch"' not in body


# --- a source the corpus hides ----------------------------------------------


def test_a_dotfile_has_no_consent_page(tmp_path: Path) -> None:
    """The dashboard's inbox walk skips dot-names. A route that rendered one
    would be a second, quieter answer to what is in somebody's corpus."""
    _corpus(tmp_path)
    (tmp_path / "inbox" / ".hidden.pdf").write_bytes(PDF)
    session = _session(tmp_path)

    status, _body = _get(session, f"/{session.token}/extract/.hidden.pdf")

    assert status == 404


def test_a_scan_root_replaced_by_a_symlink_exposes_no_external_source(
    tmp_path: Path,
) -> None:
    _corpus(tmp_path)
    session = _session(tmp_path)
    inbox = tmp_path / "inbox"
    inbox.rename(tmp_path / "original-inbox")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.pdf").write_bytes(b"private outside bytes")
    inbox.symlink_to(outside, target_is_directory=True)

    status, body = _get(session, f"/{session.token}/extract/secret.pdf")

    assert status == 404
    assert "paid API call" not in body


def test_a_symlinked_source_entry_has_no_consent_page(tmp_path: Path) -> None:
    source = _corpus(tmp_path)
    session = _session(tmp_path)
    source.rename(tmp_path / "original.pdf")
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"private outside bytes")
    source.symlink_to(outside)

    status, body = _get(session, f"/{session.token}/extract/genki-8.pdf")

    assert status == 404
    assert "paid API call" not in body


def test_a_source_swap_during_capture_cannot_change_the_rendered_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _corpus(tmp_path)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"private outside bytes")
    original = tmp_path / "original.pdf"
    real_read = inputs._read_fd_bytes
    swapped = False

    def swap_while_reading(descriptor: int) -> bytes:
        nonlocal swapped
        captured = real_read(descriptor)
        source.rename(original)
        source.symlink_to(outside)
        swapped = True
        return captured

    monkeypatch.setattr(inputs, "_read_fd_bytes", swap_while_reading)

    try:
        consent = describe_extraction(ProjectConfig.load(tmp_path), source)
    finally:
        if source.is_symlink():
            source.unlink()
        if original.exists():
            original.rename(source)

    assert swapped
    assert not consent.sendable
    assert "changed while it was being read" in consent.refusal
