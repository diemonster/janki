"""Letting a model answer the coverage question, and the record it leaves.

The owner's decision moved up a level — from "is this page accounted for" to
"may a model answer that" — so the tests here are mostly about *provenance and
refusal*: that a verdict nobody really gave is never recorded as one, and that
a card promoted on a model's word says so permanently.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from japanese_anki import coverage, prompts
from japanese_anki.claude_client import CallResult
from japanese_anki.errors import JankiError
from japanese_anki.inputs import PreparedInput
from japanese_anki.staging import StagingError, record_coverage_approval

REPO_ROOT = Path(__file__).resolve().parents[1]

BLOCK: dict[str, Any] = {
    "source_units": [
        {
            "page": 1,
            "section": "vocabulary",
            "ordinal": 1,
            "context": "話す　はなす　to speak",
            "disposition": "candidate",
        },
        {
            "page": 1,
            "section": "vocabulary",
            "ordinal": 2,
            "context": "Week 11 Homework",
            "disposition": "non-vocabulary",
            "reason": "A heading, not a vocabulary row.",
        },
    ],
}


def page() -> PreparedInput:
    return PreparedInput(
        kind="document",
        media_type="application/pdf",
        data_b64="ZmFrZQ==",
        origin_path=Path("lesson.pdf"),
    )


def answering(**fields: Any):
    """A `parse_call` returning one canned verdict, recording what it was sent."""
    sent: list[dict[str, Any]] = []

    def call(model, blocks, content, schema, client=None, **options):
        sent.append({"model": model, "blocks": blocks, "content": content})
        return CallResult(SimpleNamespace(**fields), "end_turn", None)

    call.sent = sent  # type: ignore[attr-defined]
    return call


# --- what the model is shown ---------------------------------------------------


def test_the_request_identity_binds_source_account_prompt_and_model() -> None:
    original = coverage.build_request(page(), BLOCK, model="m1", instructions="ASK")
    other_source = PreparedInput(
        kind="document",
        media_type="application/pdf",
        data_b64="ZGlmZmVyZW50",
        origin_path=Path("lesson.pdf"),
    )
    changed_block = {**BLOCK, "source_units": [*BLOCK["source_units"]]}
    changed_block["source_units"][0] = {
        **changed_block["source_units"][0],
        "context": "a different first row",
    }

    fingerprints = {
        original.request_fingerprint,
        coverage.build_request(
            other_source, BLOCK, model="m1", instructions="ASK"
        ).request_fingerprint,
        coverage.build_request(
            page(), changed_block, model="m1", instructions="ASK"
        ).request_fingerprint,
        coverage.build_request(
            page(), BLOCK, model="m1", instructions="ASK AGAIN"
        ).request_fingerprint,
        coverage.build_request(
            page(), BLOCK, model="m2", instructions="ASK"
        ).request_fingerprint,
    }

    assert len(fingerprints) == 5


def test_the_exact_reply_capture_is_forwarded_before_the_verdict_is_used() -> None:
    events: list[object] = []
    response = object()

    def call(*_args: Any, capture=None, **_kwargs: Any) -> CallResult:
        assert capture is not None
        capture(response)
        events.append("parsed")
        return CallResult(
            SimpleNamespace(approved=True, reason="Accounted for."),
            "end_turn",
            None,
        )

    coverage.review_coverage(
        page(),
        BLOCK,
        model="m",
        instructions="ASK",
        parse_call=call,
        capture=lambda value: events.append(value),
    )

    assert events == [response, "parsed"]


def test_the_model_is_shown_the_page_itself_not_a_description_of_it() -> None:
    """Asking a model to confirm a reading of a page it cannot see would be
    asking it to agree with itself. The same base64 block the extraction pass
    was given rides in the user turn."""
    call = answering(approved=True, reason="Every row is accounted for.")

    coverage.review_coverage(
        page(), BLOCK, model="m", instructions="I", parse_call=call
    )

    [block, text] = call.sent[0]["content"]
    assert block["source"]["data"] == "ZmFrZQ==", "the page rides along"
    assert block["source"]["media_type"] == "application/pdf"
    assert "話す　はなす　to speak" in text["text"], "and janki's account of it"


def test_the_account_carries_each_units_disposition_and_reason() -> None:
    """What lets the model line the record up against what it sees. The
    verbatim text is never truncated, and a disposition without its stated
    reason would ask the model to guess why something was filed as it was."""
    account = coverage.format_account(BLOCK)

    assert "[candidate]" in account and "[non-vocabulary]" in account
    assert "A heading, not a vocabulary row." in account
    assert "page 1 · vocabulary · #2" in account


# --- verdicts, and the ones that are not verdicts -------------------------------


def test_an_approval_carries_the_model_and_the_prompt_it_answered() -> None:
    """The two things that make a model's approval reviewable later: the same
    page can be accepted by one prompt and refused by the next."""
    call = answering(approved=True, reason="All four rows present.")

    verdict = coverage.review_coverage(
        page(), BLOCK, model="claude-opus-5", instructions="ASK", parse_call=call
    )

    assert verdict.approved is True
    assert verdict.model == "claude-opus-5"
    assert verdict.prompt_fingerprint == prompts.fingerprint("ASK")


def test_a_refusal_is_carried_with_its_reason() -> None:
    call = answering(approved=False, reason="Row 3 of the table is missing.")

    verdict = coverage.review_coverage(
        page(), BLOCK, model="m", instructions="I", parse_call=call
    )

    assert verdict.approved is False
    assert "Row 3" in verdict.reason


@pytest.mark.parametrize(
    "result,expected",
    [
        (CallResult(None, "end_turn", "policy"), "declined"),
        (CallResult(SimpleNamespace(approved=True, reason="ok"), "max_tokens", None),
         "cut off"),
        (CallResult(None, "end_turn", None), "no verdict"),
    ],
    ids=["refused", "truncated", "empty"],
)
def test_a_question_never_answered_is_an_error_not_a_no(
    result: CallResult, expected: str
) -> None:
    """The distinction that matters most here.

    Recording any of these as a refusal would tell a reader the page was
    examined and found wanting, when it was not examined at all — and recording
    them as an approval would be worse. Both are errors, and nothing is
    written.
    """
    def call(*_args: Any, **_kwargs: Any) -> CallResult:
        return result

    with pytest.raises(coverage.CoverageReviewError) as raised:
        coverage.review_coverage(
            page(), BLOCK, model="m", instructions="I", parse_call=call
        )

    assert expected in str(raised.value)
    assert "Nothing was approved" in str(raised.value) or "refused" in str(raised.value)


def test_an_approval_with_no_reason_is_refused() -> None:
    """The reason is the whole of what a person reads months later. An
    approval that says nothing is not an approval anyone can review."""
    call = answering(approved=True, reason="   ")

    with pytest.raises(coverage.CoverageReviewError, match="no reason"):
        coverage.review_coverage(
            page(), BLOCK, model="m", instructions="I", parse_call=call
        )


# --- the record it leaves -------------------------------------------------------


STAGING = """\
source_file: lesson.pdf
coverage:
  version: 1
  # A reviewer's note that must survive.
  status: unmeasured
records:
  - id: 'word:話す:はなす'
    expression: 話す
    reading: はなす
"""


def test_the_approval_lands_without_disturbing_the_reviewers_file(
    tmp_path: Path,
) -> None:
    """A staging file under review holds work that exists nowhere else. Writing
    an approval through a load-and-dump would silently drop the comments and
    any key janki has no field for."""
    path = tmp_path / "lesson.pdf.yaml"
    path.write_text(STAGING, encoding="utf-8")

    record_coverage_approval(path, {"authority": "model", "reason": "Accounted for."})

    written = path.read_text(encoding="utf-8")
    assert "# A reviewer's note that must survive." in written
    # Double-quoted, because ruamel writes YAML 1.2 and `read_staging` reads
    # PyYAML's 1.1: a bare `no`, `on` or `12:30` would round-trip into False,
    # True or 750 and fail the very approval just recorded.
    assert 'authority: "model"' in written
    assert 'reason: "Accounted for."' in written


def test_a_second_approval_does_not_overwrite_the_first(tmp_path: Path) -> None:
    """An approval is a decision about a specific block. Replacing one silently
    would let a re-run write a model's reasoning over a person's."""
    path = tmp_path / "lesson.pdf.yaml"
    path.write_text(STAGING, encoding="utf-8")
    record_coverage_approval(path, {"authority": "repository-owner", "reason": "Mine."})

    with pytest.raises(StagingError, match="already carries a coverage approval"):
        record_coverage_approval(path, {"authority": "model", "reason": "Theirs."})

    assert "Mine." in path.read_text(encoding="utf-8")


def test_the_shipped_checker_prompt_states_what_it_is_not_judging() -> None:
    """The clause keeping this from becoming the review subsystem again.

    A model handed a page will volunteer opinions about the glosses unless
    told not to; M8.2 deleted a whole subsystem for making that its job. If
    this clause ever leaves the file, the pass stops being about accounting.
    """
    text = prompts.load(REPO_ROOT, "approve-coverage")

    assert "What you are not judging" in text
    assert "Not whether a reading is correct" in text
    # And it knows the extraction's own contract: prose selection is not
    # exhaustive, so a sentence absent from the record is normal. Without this
    # the checker refuses every page carrying running text.
    assert "running text" in text
    assert "not exhaustively" in text or "not exhaustive" in text


# --- who is allowed to say yes --------------------------------------------------


def approved_block(**extra: Any) -> dict[str, Any]:
    """A coverage block carrying an approval, ready for the staging validator."""
    from japanese_anki.extract import ExtractionResult, coverage_block

    block = coverage_block(
        ExtractionResult(candidates=(), source_units=(), model_reported_unit_count=0),
        source_sha256="a" * 64,
        mode="table",
    )
    from japanese_anki.staging import coverage_acceptance_requirements

    block["approval"] = {
        "authority": "model",
        "model": "claude-opus-5",
        "prompt_fingerprint": "b" * 64,
        "source_fingerprint": block["source_fingerprint"],
        "coverage_block_fingerprint": block["coverage_block_fingerprint"],
        **coverage_acceptance_requirements(block),
        "reason": "Every row is accounted for.",
        "approved_at": "2026-08-16",
        **extra,
    }
    return block


def resolve(block: dict[str, Any]) -> None:
    from japanese_anki.staging import require_resolved_coverage

    require_resolved_coverage(
        {
            "coverage": block,
            "prompt_provenance": {
                "source_sha256": block["source_fingerprint"],
                "mode": "table",
                "provider": "anthropic",
                "model": "claude-opus-5",
                "response_schema_version": 2,
                "system_prompt_fingerprint": "c" * 64,
                "style_guide_fingerprint": "d" * 64,
                "user_prompt_fingerprint": "e" * 64,
            },
        }
    )


def test_a_model_approval_is_accepted_when_it_says_which_model_and_which_prompt() -> None:
    resolve(approved_block())


@pytest.mark.parametrize("missing", ["model", "prompt_fingerprint"])
def test_a_model_approval_without_its_provenance_is_refused(missing: str) -> None:
    """The two fields that make it reviewable. Without them the line says a
    model approved this page and gives a reader no way to find out which one,
    or what it was asked — which is the same as not recording it."""
    block = approved_block()
    del block["approval"][missing]

    with pytest.raises(StagingError, match="fields do not match"):
        resolve(block)


def test_an_owner_approval_still_needs_no_model_fields() -> None:
    """A person is their own provenance. Requiring a model id from a human
    approval would be asking them to name a model that never ran."""
    block = approved_block()
    approval = block["approval"]
    approval["authority"] = "repository-owner"
    del approval["model"]
    del approval["prompt_fingerprint"]

    resolve(block)


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
@pytest.mark.parametrize("field", ["model", "prompt_fingerprint"])
def test_a_model_approval_with_a_blank_provenance_field_is_refused(
    field: str, blank: str
) -> None:
    """Present-but-empty is the shape a hand-edit produces, and it satisfies a
    field-set check while carrying no information. The guard existed; nothing
    pinned it, so deleting it left the suite green."""
    with pytest.raises(StagingError, match=f"needs {field}"):
        resolve(approved_block(**{field: blank}))


def test_an_invented_authority_is_refused() -> None:
    """Only two things may accept a page: the owner, or a model on the owner's
    standing instruction. A third value would be an approval nobody can trace."""
    with pytest.raises(StagingError, match="authority must be"):
        resolve(approved_block(authority="looked-fine-to-me"))


# --- the CLI path, which had no test at all -------------------------------------


def project_with_source(tmp_path: Path, *, mode: str = "table") -> tuple[Path, Path]:
    """A project holding one inbox PDF and a staging file extracted from it."""
    import json

    from conftest import seed_promotion_deck, seed_prompts
    from japanese_anki.extract import ExtractionResult, SourceUnit, coverage_block
    from japanese_anki.staging import coverage_block_fingerprint

    (tmp_path / "janki.toml").write_text(
        '[paths]\nnormalized_file = "vocabulary.json"\n'
        'staging_dir = "staging"\nscan_inbox = "inbox"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    seed_prompts(tmp_path)
    seed_promotion_deck(tmp_path)
    source = tmp_path / "inbox" / "lesson.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.4\n%real\n")

    import hashlib

    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    units = () if mode == "prose" else tuple(
        SourceUnit(
            page=1, section="vocabulary", ordinal=n,
            context=f"row {n}",
            context_fingerprint=hashlib.sha256(f"row {n}".encode()).hexdigest(),
            disposition="candidate", reason="",
        )
        for n in (1, 2, 3)
    )
    block = coverage_block(
        ExtractionResult(
            candidates=(), source_units=units, model_reported_unit_count=len(units)
        ),
        source_sha256=sha,
        mode=mode,
    )
    staged = tmp_path / "staging" / "lesson.pdf.yaml"
    staged.parent.mkdir(parents=True)
    staged.write_text(
        "source_file: lesson.pdf\n"
        f"coverage: {json.dumps(block)}\n"
        "prompt_provenance:\n"
        f"  source_sha256: '{sha}'\n"
        f"  mode: {mode}\n  provider: anthropic\n  model: m\n"
        "  response_schema_version: 2\n"
        f"  system_prompt_fingerprint: '{'c' * 64}'\n"
        f"  style_guide_fingerprint: '{'d' * 64}'\n"
        f"  user_prompt_fingerprint: '{'e' * 64}'\n"
        "records:\n"
        "  - id: 'word:話す:はなす'\n    expression: 話す\n    reading: はなす\n"
        "    meanings: ['to speak']\n",
        encoding="utf-8",
    )
    assert coverage_block_fingerprint(block) == block["coverage_block_fingerprint"]
    return tmp_path, staged


def test_the_recorded_fingerprint_matches_the_block_that_was_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The field binding an approval to the account the model was shown.

    `coverage_acceptance_requirements` repeats the disposition lists but not
    `source_units`, and `format_account` builds what the model reads *from*
    `source_units` — so this fingerprint is the only thing tying the recorded
    verdict to the account that produced it. Truncate the units, approve the
    short account, restore the block, and a stale fingerprint would leave a
    permanent record claiming a model found a page complete that it never saw.

    **This test cannot tell a recompute from a copy, and does not claim to.**
    The sibling below is what closes that hole: validation now runs before the
    call, and it refuses any block whose stored fingerprint disagrees with its
    own facts — so by the time this value is written the two are equal by
    construction. `coverage_block_fingerprint(block)` is kept at the write
    anyway, as the form that stays correct if that ordering is ever changed.
    """
    from japanese_anki.staging import coverage_block_fingerprint, read_staging

    root, staged = project_with_source(tmp_path)
    monkeypatch.setattr(
        coverage, "review_coverage",
        lambda *a, **k: coverage.CoverageVerdict(True, "Accounted for.", "m", "f" * 64),
    )
    # A file whose stored fingerprint disagrees with its own facts is refused
    # outright now, so the honest check is that what we write matches the
    # recomputed value rather than whatever the file happened to carry.
    assert _accept(root, staged) is True

    _records, after = read_staging(staged)
    block = after["coverage"]
    assert block["approval"]["coverage_block_fingerprint"] == (
        coverage_block_fingerprint(block)
    )


def test_a_block_the_gate_would_reject_is_never_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate first, spend second.

    Even a self-consistent edited block is free to reject by regenerating its
    facts from the source units. Finding out after the call meant paying for a
    verdict, writing it into the file, and *then* refusing — leaving a staging
    file that could neither be promoted nor re-approved without hand-deleting
    the approval it had just been given.
    """
    from japanese_anki.staging import (
        coverage_block_fingerprint,
        read_staging,
        write_staging,
    )

    root, staged = project_with_source(tmp_path)
    records, meta = read_staging(staged)
    block = meta["coverage"]
    block["candidate_units"] = []
    block["coverage_block_fingerprint"] = coverage_block_fingerprint(block)
    write_staging(staged, records, meta, force=True)
    calls: list[object] = []
    monkeypatch.setattr(
        coverage, "review_coverage",
        lambda *a, **k: calls.append(1) or coverage.CoverageVerdict(True, "ok", "m", "f"),
    )

    with pytest.raises(JankiError, match="coverage-facts-stale"):
        _accept(root, staged)

    assert calls == [], "nothing was sent"
    assert "approval" not in staged.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "reason",
    # Measured, not guessed: each of these is emitted *bare* by ruamel's 1.2
    # resolver and read back by PyYAML 1.1 as a non-string, so each fails
    # without the quoting fix. Three candidates were tried and dropped for
    # failing that test — `y` (1.1 does not resolve it), `1.0` and `true`
    # (ruamel already quotes both, so they could never fail). Check a new one
    # the same way before adding it.
    ["no", "on", "off", "yes", "12:30"],
    ids=["no", "on", "off", "yes", "sexagesimal"],
)
def test_an_approval_survives_both_yaml_dialects(tmp_path: Path, reason: str) -> None:
    """The hazard `rewrite_staging` documents, arriving from the other side.

    This writes through ruamel (YAML 1.2) into a file `read_staging` reads back
    with PyYAML (YAML 1.1). Bare, `no` becomes False, `on` becomes True and
    `12:30` becomes 750 — so an approval whose reason is one of those words
    would fail the very check it was written to satisfy, and the file could not
    be promoted or re-approved without hand-deleting it. Every string is
    emitted double-quoted so the round trip is the identity.
    """
    from japanese_anki.staging import read_staging

    path = tmp_path / "lesson.pdf.yaml"
    path.write_text(STAGING, encoding="utf-8")

    record_coverage_approval(path, {"authority": "model", "reason": reason})

    _records, meta = read_staging(path)
    assert meta["coverage"]["approval"]["reason"] == reason


def test_a_block_that_needs_no_approval_is_not_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A prose-only extraction does not block, so there is nothing to accept.

    `validate_coverage_facts` says so by returning None. Discarding that return
    meant `--accept-coverage` bought a verdict and wrote an `authority: model`
    approval into a committed file that no gate would ever read — the same
    waste as spending before validation, one branch over.
    """
    root, staged = project_with_source(tmp_path, mode="prose")
    calls: list[object] = []
    monkeypatch.setattr(
        coverage, "review_coverage",
        lambda *a, **k: calls.append(1) or coverage.CoverageVerdict(True, "ok", "m", "f"),
    )

    accepted = _accept(root, staged)

    assert accepted is False
    assert calls == [], "nothing was sent"
    assert "approval" not in staged.read_text(encoding="utf-8")
    assert "does not need a coverage approval" in capsys.readouterr().err


def test_the_promote_command_sends_the_coverage_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the shared coverage service, not a direct unit call.

    The first attempt at this test called `review_coverage` directly and handed
    it the file — pinning only that the function forwards its own argument,
    which is exactly the mistake it was written to correct, applied to the site
    its own docstring called the worst one. Repointing `cli.py`'s loader at any
    other template stayed invisible.

    It matters here more than anywhere: `review_coverage` fingerprints the text
    it is handed, so a wrong wiring writes a permanent `authority: model`
    approval into a committed file naming a question that was never asked.
    """
    from japanese_anki import prompts

    root, staged = project_with_source(tmp_path)
    sent: list[str] = []

    def call(_model, blocks, _content, _schema, _client=None, **_options):
        from japanese_anki.claude_client import CallResult

        sent.append(
            "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in blocks
            )
        )
        return CallResult(
            SimpleNamespace(approved=True, reason="Accounted for."), "end_turn", None
        )

    monkeypatch.setattr(coverage.claude_client, "parse_call", call)

    assert _accept(root, staged) is True

    assert sent, "the command reached the model"
    assert prompts.load(REPO_ROOT, "approve-coverage") in sent[0]
    for other in (
        "extract-auto",
        "extract-table",
        "extract-prose",
        "enrich-bare-word",
    ):
        assert prompts.load(REPO_ROOT, other) not in sent[0], other


# --- asking again, and not asking again -------------------------------------


def _accept(root: Path, staged: Path, **kwargs: object) -> bool:
    from japanese_anki.application import coverage as coverage_application
    from japanese_anki.config import ProjectConfig

    decision = coverage_application.plan_model_coverage(
        ProjectConfig.load(root),
        staged,
        reaccept=bool(kwargs.get("reaccept", False)),
    )
    if decision.state == "not_required":
        print(decision.detail, file=sys.stderr)
        return False
    if decision.state == "already_resolved":
        print(decision.detail + " Use --reaccept-coverage to ask again and replace it.")
        return True
    return (
        coverage_application.run_model_coverage(
            ProjectConfig.load(root), decision
        ).state
        == "approved"
    )


def test_an_approval_that_already_stands_is_not_bought_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The approval is bound to the source and coverage-block fingerprints and
    repeats the accepted facts, so anything that could change the answer makes
    it *stale* rather than leaving it standing. A standing one therefore means
    the same question about the same bytes — and asking again bought the same
    verdict and then failed to write it, because replacing a recorded decision
    is refused.
    """
    root, staged = project_with_source(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(
        coverage, "review_coverage",
        lambda *a, **k: calls.append(1)
        or coverage.CoverageVerdict(True, "Accounted for.", "m", "f"),
    )
    assert _accept(root, staged) is True
    assert calls == [1]
    before = staged.read_text(encoding="utf-8")

    # Same file, same question, asked again.
    assert _accept(root, staged) is True

    assert calls == [1], "the second run must send nothing"
    assert staged.read_text(encoding="utf-8") == before
    assert "already approved" in capsys.readouterr().out


def test_reaccepting_asks_again_and_says_it_meant_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case worth paying for: changing the authority, or asking a better
    model than the one that answered before. The replacement records that it
    replaced something, because the file is the only place a later reader can
    see an earlier decision was set aside on purpose rather than lost.
    """
    from japanese_anki.staging import read_staging

    root, staged = project_with_source(tmp_path)
    verdicts = iter([
        coverage.CoverageVerdict(True, "First look.", "weak-model", "f1"),
        coverage.CoverageVerdict(True, "Second look.", "opus-5", "f2"),
    ])
    monkeypatch.setattr(
        coverage, "review_coverage", lambda *a, **k: next(verdicts)
    )

    assert _accept(root, staged) is True
    _records, first = read_staging(staged)
    assert first["coverage"]["approval"]["model"] == "claude-opus-5"

    assert _accept(root, staged, reaccept=True) is True

    _records, second = read_staging(staged)
    approval = second["coverage"]["approval"]
    assert approval["model"] == "claude-opus-5"
    assert "Second look." in approval["reason"]
    assert "replaced an earlier approval" in approval["reason"]


def test_reaccepting_a_file_with_no_approval_is_an_ordinary_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to replace, so nothing claims to have replaced anything — a
    reason that says it set aside an earlier decision would be a false record
    of one having existed."""
    from japanese_anki.staging import read_staging

    root, staged = project_with_source(tmp_path)
    monkeypatch.setattr(
        coverage, "review_coverage",
        lambda *a, **k: coverage.CoverageVerdict(True, "Accounted for.", "m", "f"),
    )

    assert _accept(root, staged, reaccept=True) is True

    _records, meta = read_staging(staged)
    assert meta["coverage"]["approval"]["reason"] == "Accounted for."


def test_a_structurally_refused_file_never_reaches_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--accept-coverage` buys a verdict on whether a page is accounted for.
    A file its *structural* gates refuse cannot be made promotable by any
    verdict, so the call is money for nothing — and the refusal a person needs
    is the structural one, not one about coverage.

    The file has to carry a real coverage block for this to mean anything: a
    file too broken to parse is stopped by the acceptance helper's own
    self-gating, which would make the guard look tested when it is not.
    """
    from japanese_anki import cli
    from japanese_anki.application import coverage as coverage_application
    from japanese_anki.config import ProjectConfig

    root, staged = project_with_source(tmp_path)
    # A valid review under a suffix the archive transaction cannot rewrite.
    # This gate comes after coverage in the ordinary promotion pipeline, so it
    # proves the pre-pay pass does not merely stop at the unresolved approval.
    refused = staged.with_suffix(".txt")
    staged.rename(refused)
    staged = refused
    sent: list[object] = []
    monkeypatch.setattr(
        coverage, "review_coverage", lambda *a, **k: sent.append(1) or None
    )

    with pytest.raises(JankiError, match="not a staging file"):
        coverage_application.plan_coverage(ProjectConfig.load(root), staged)

    assert cli.main([
        "--root", str(root), "promote", str(staged), "--accept-coverage",
    ]) == 1

    assert sent == [], "a structural refusal must not buy a coverage verdict"
    assert capsys.readouterr().err.strip()


def test_an_accepted_file_is_then_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the flag. The approval is written into the staging file,
    and the promote that follows has to see it — which means deciding again
    against the file as it now stands, not against the copy read before the
    approval existed."""
    from conftest import seed_promotion_deck
    from japanese_anki import cli, operations
    from japanese_anki.config import ProjectConfig
    from japanese_anki.io import load_records

    root, staged = project_with_source(tmp_path)
    seed_promotion_deck(root)
    monkeypatch.setattr(
        coverage, "review_coverage",
        lambda *a, **k: coverage.CoverageVerdict(True, "Accounted for.", "m", "f"),
    )

    assert cli.main([
        "--root", str(root), "promote", str(staged),
        "--accept-coverage", "--skip-reading-check",
    ]) == 0

    assert load_records(root / "vocabulary.json"), "the records should have landed"
    assert "approval" in staged.parent.joinpath("done", staged.name).read_text(
        encoding="utf-8"
    )
    [entry] = operations.OperationJournal.load(
        ProjectConfig.load(root).operations_file
    ).operations.values()
    assert (entry.kind, entry.state) == ("coverage", "committed")


# --- shared application service ------------------------------------------------


def test_owner_coverage_approval_is_bound_to_the_rendered_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owner's reason may approve only the local coverage bytes they saw."""
    from japanese_anki.application.coverage import (
        approve_coverage_as_owner,
        plan_coverage,
    )
    from japanese_anki.config import ProjectConfig
    from japanese_anki.staging import read_staging, require_resolved_coverage

    root, staged = project_with_source(tmp_path)
    config = ProjectConfig.load(root)
    monkeypatch.setattr(
        coverage,
        "build_request",
        lambda *_a, **_k: pytest.fail("owner approval loaded the optional AI route"),
    )
    stale = plan_coverage(config, staged)
    staged.write_text(
        staged.read_text(encoding="utf-8") + "# changed after preview\n",
        encoding="utf-8",
    )

    with pytest.raises(JankiError, match="changed|stale"):
        approve_coverage_as_owner(
            config,
            stale,
            reason="I compared all three source rows.",
        )

    _records, untouched = read_staging(staged)
    assert "approval" not in untouched["coverage"]

    fresh = plan_coverage(config, staged)
    approve_coverage_as_owner(
        config,
        fresh,
        reason="  I compared all three source rows.  ",
    )

    _records, approved = read_staging(staged)
    approval = approved["coverage"]["approval"]
    assert approval["authority"] == "repository-owner"
    assert approval["reason"] == "I compared all three source rows."
    require_resolved_coverage(approved)

    # Coverage sends the same private corpus bytes extraction did. A final
    # symlink must not let a staged basename read an outside file, even when
    # the target has identical bytes and would pass the source fingerprint.
    symlink_root = tmp_path / "coverage-symlink"
    symlink_root.mkdir()
    _root, symlink_staged = project_with_source(symlink_root)
    symlink_config = ProjectConfig.load(symlink_root)
    symlink_source = symlink_config.scan_inbox / "lesson.pdf"
    held_source = symlink_root / "held-source.pdf"
    outside = symlink_root / "outside.pdf"
    outside.write_bytes(symlink_source.read_bytes())
    symlink_source.rename(held_source)
    symlink_source.symlink_to(outside)
    try:
        with pytest.raises(JankiError, match="symlink|regular|safely read"):
            plan_coverage(symlink_config, symlink_staged)
    finally:
        symlink_source.unlink()
        held_source.rename(symlink_source)

    # Replacing the direct entry after its descriptor is read is also stale;
    # the captured original must not be silently described as the new path.
    from japanese_anki import inputs as inputs_module

    replacement_root = tmp_path / "coverage-replacement"
    replacement_root.mkdir()
    _root, replacement_staged = project_with_source(replacement_root)
    replacement_config = ProjectConfig.load(replacement_root)
    replacement_source = replacement_config.scan_inbox / "lesson.pdf"
    replacement_held = replacement_root / "held-source.pdf"
    replacement_outside = replacement_root / "outside.pdf"
    replacement_outside.write_bytes(replacement_source.read_bytes())
    real_read = inputs_module._read_fd_bytes
    swapped = False

    def replace_after_read(descriptor: int) -> bytes:
        nonlocal swapped
        captured = real_read(descriptor)
        if not swapped:
            replacement_source.rename(replacement_held)
            replacement_source.symlink_to(replacement_outside)
            swapped = True
        return captured

    monkeypatch.setattr(inputs_module, "_read_fd_bytes", replace_after_read)
    try:
        with pytest.raises(JankiError, match="changed while|safely read"):
            plan_coverage(replacement_config, replacement_staged)
        assert swapped
    finally:
        if replacement_source.is_symlink():
            replacement_source.unlink()
        if replacement_held.exists():
            replacement_held.rename(replacement_source)


def test_missing_key_refuses_coverage_before_journal_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from japanese_anki import operations
    from japanese_anki.application import coverage as coverage_application
    from japanese_anki.config import ProjectConfig

    root, staged = project_with_source(tmp_path)
    config = ProjectConfig.load(root)
    decision = coverage_application.plan_model_coverage(config, staged)

    def missing_key() -> object:
        raise JankiError("ANTHROPIC_API_KEY is not set; no provider was contacted")

    monkeypatch.setattr(
        coverage_application.claude_client,
        "prepare_paid_client",
        missing_key,
    )

    with pytest.raises(JankiError, match="no provider was contacted"):
        coverage_application.run_model_coverage(config, decision)

    assert not config.operations_file.exists()
    assert not (config.operations_file.parent / operations.PENDING_DIR).exists()


def test_paid_coverage_is_authorized_and_captured_before_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider sees dispatching; parsing sees a durable exact reply."""
    from japanese_anki import operations
    from japanese_anki.application import coverage as coverage_application
    from japanese_anki.application import promotion as promotion_application
    from japanese_anki.config import ProjectConfig
    from japanese_anki.staging import read_staging, require_resolved_coverage

    root, staged = project_with_source(tmp_path)
    config = ProjectConfig.load(root)
    decision = coverage_application.plan_model_coverage(config, staged)
    observed: list[str] = []

    def review(*_args: Any, capture=None, **_kwargs: Any) -> coverage.CoverageVerdict:
        [entry] = operations.OperationJournal.load(
            config.operations_file
        ).operations.values()
        observed.append(entry.state)
        assert capture is not None
        capture({"content": [{"type": "text", "text": "exact paid reply"}]})
        [captured] = operations.OperationJournal.load(
            config.operations_file
        ).operations.values()
        observed.append(captured.state)
        return coverage.CoverageVerdict(
            True,
            "Every source row is accounted for.",
            decision.model,
            decision.prompt_fingerprint,
        )

    monkeypatch.setattr(coverage_application.coverage, "review_coverage", review)

    prompt_path = prompts.path_for(root, "approve-coverage")
    prompt_wire = prompt_path.read_bytes()
    prompt_path.write_bytes(prompt_wire + b"\nChanged after consent.\n")
    with pytest.raises(JankiError, match="prompt|stale"):
        coverage_application.run_model_coverage(config, decision)
    assert operations.OperationJournal.load(config.operations_file).operations == {}
    assert observed == [], "a changed paid request must not be authorized or sent"
    prompt_path.write_bytes(prompt_wire)
    decision = coverage_application.plan_model_coverage(config, staged)

    outcome = coverage_application.run_model_coverage(config, decision)

    assert observed == ["dispatching", "result_captured"]
    assert outcome.state == "approved"
    entry = operations.OperationJournal.load(config.operations_file).operations[
        outcome.operation_id
    ]
    assert (entry.kind, entry.state) == ("coverage", "committed")
    assert entry.request_fp == decision.request_fingerprint
    _records, meta = read_staging(staged)
    require_resolved_coverage(meta)

    # A rejecting verdict is a paid answer, not an approval. Keep it in the
    # journal for recovery and leave the coverage gate unresolved.
    decline_root = tmp_path / "decline"
    decline_root.mkdir()
    _root, decline_staged = project_with_source(decline_root)
    decline_config = ProjectConfig.load(decline_root)
    decline_decision = coverage_application.plan_model_coverage(
        decline_config, decline_staged
    )

    def reject(
        *_args: Any, capture=None, **_kwargs: Any
    ) -> coverage.CoverageVerdict:
        assert capture is not None
        capture({"content": [{"type": "text", "text": "not complete"}]})
        return coverage.CoverageVerdict(
            False,
            "One source row is missing.",
            decline_decision.model,
            decline_decision.prompt_fingerprint,
        )

    monkeypatch.setattr(coverage_application.coverage, "review_coverage", reject)
    declined = coverage_application.run_model_coverage(
        decline_config, decline_decision
    )

    assert declined.state == "declined"
    decline_entry = operations.OperationJournal.load(
        decline_config.operations_file
    ).operations[declined.operation_id]
    assert decline_entry.state == "result_captured"
    _records, declined_meta = read_staging(decline_staged)
    assert "approval" not in declined_meta["coverage"]

    # A provisional deck refusal makes preflight call jpdb. Every repository
    # input read before that blocking lookup must still be current before the
    # separate paid model call is authorized; its eventual approval CAS would
    # be too late.
    from test_promote import FakeJpdb, client_for

    from japanese_anki import promote as promote_module
    from japanese_anki.staging import write_staging

    paid: list[object] = []
    monkeypatch.setattr(
        coverage_application.coverage,
        "review_coverage",
        lambda *_a, **_k: paid.append(1),
    )

    def race_project(
        name: str, *, reaccept: bool = False, partial_archive: bool = False
    ) -> tuple[ProjectConfig, Path, Path, object]:
        race_root = tmp_path / name
        race_root.mkdir()
        _root, race_staged = project_with_source(race_root)
        race_config = ProjectConfig.load(race_root)
        [race_deck] = race_config.deck_dir.glob("*.yaml")
        race_deck.write_text(
            "deck:\n"
            "  name: Other deck\n"
            '  source: "../../vocabulary.json"\n'
            "  intake_tag: other\n"
            "  include_tags: [other]\n",
            encoding="utf-8",
        )
        if partial_archive:
            records, meta = read_staging(race_staged)
            write_staging(
                race_staged,
                [
                    records[0],
                    replace(
                        records[0],
                        id="word:聞く:きく",
                        expression="聞く",
                        reading="きく",
                    ),
                ],
                meta,
                force=True,
            )
        if reaccept:
            coverage_application.approve_coverage_as_owner(
                race_config,
                coverage_application.plan_coverage(race_config, race_staged),
                reason="I compared all three source rows.",
            )
        race_decision = coverage_application.plan_model_coverage(
            race_config, race_staged, reaccept=reaccept
        )
        return race_config, race_staged, race_deck, race_decision

    def refuses_after(
        config: ProjectConfig,
        decision: object,
        change: Callable[[], None],
    ) -> None:
        class _ChangingJpdb(FakeJpdb):
            def __call__(self, *args: object, **kwargs: object) -> object:
                if not self.bodies:
                    change()
                return super().__call__(*args, **kwargs)  # type: ignore[arg-type]

        api = _ChangingJpdb(
            {
                "話す": [999_001, 999_002, "話す", "ちがう", [], 100, []],
                "聞く": [999_003, 999_004, "聞く", "ちがう", [], 100, []],
            },
            {
                (999_001, 999_002): {"reading": "ちがう", "alt_sids": []},
                (999_003, 999_004): {"reading": "ちがう", "alt_sids": []},
            },
        )
        with pytest.raises(
            coverage_application.CoverageApplicationError,
            match="preflight-stale",
        ):
            coverage_application.run_model_coverage(
                config,
                decision,  # type: ignore[arg-type]
                reading_client=client_for(api),
            )
        assert api.bodies
        assert paid == []
        assert operations.OperationJournal.load(config.operations_file).operations == {}

    def refuses_gate_swap(
        config: ProjectConfig,
        decision: object,
        *,
        during_lookup: Callable[[], None] = lambda: None,
        match: str = "promotion-input-stale",
    ) -> None:
        class _GateJpdb(FakeJpdb):
            def __call__(self, *args: object, **kwargs: object) -> object:
                if not self.bodies:
                    during_lookup()
                return super().__call__(*args, **kwargs)  # type: ignore[arg-type]

        api = _GateJpdb(
            {
                "話す": [999_001, 999_002, "話す", "はなす", [], 100, []],
                "聞く": [999_003, 999_004, "聞く", "きく", [], 100, []],
            },
            {
                (999_001, 999_002): {"reading": "はなす", "alt_sids": []},
                (999_003, 999_004): {"reading": "きく", "alt_sids": []},
            },
        )
        with pytest.raises(JankiError, match=match):
            coverage_application.run_model_coverage(
                config,
                decision,  # type: ignore[arg-type]
                reading_client=client_for(api),
            )
        assert api.bodies
        assert paid == []
        assert operations.OperationJournal.load(config.operations_file).operations == {}

    # Exact canonical bytes are the CAS identity even when they parse to the
    # same empty collection.
    canonical_config, _staged_path, _deck, canonical_decision = race_project(
        "preflight-canonical"
    )
    refuses_after(
        canonical_config,
        canonical_decision,
        lambda: canonical_config.normalized_file.write_text(
            "[]\n", encoding="utf-8"
        ),
    )

    staging_config, staging_path, _deck, staging_decision = race_project(
        "preflight-staging"
    )
    refuses_after(
        staging_config,
        staging_decision,
        lambda: staging_path.write_text(
            staging_path.read_text(encoding="utf-8") + "# changed during jpdb\n",
            encoding="utf-8",
        ),
    )

    stored_config, stored_staging, stored_deck, _stored_decision = race_project(
        "preflight-stored-ids"
    )
    stored_source = stored_config.root / "static.json"
    stored_source.write_text("[]", encoding="utf-8")
    stored_deck.write_text(
        "deck:\n"
        "  name: Static deck\n"
        '  source: "../../static.json"\n',
        encoding="utf-8",
    )
    stored_decision = coverage_application.plan_model_coverage(
        stored_config, stored_staging
    )
    refuses_after(
        stored_config,
        stored_decision,
        lambda: stored_source.write_text(
            '[{"id":"word:別:べつ","expression":"別","reading":"べつ"}]',
            encoding="utf-8",
        ),
    )

    # Keep the declared-id set empty while the deck becomes unreadable. A
    # dictionary hold means the consulted decision returns the coverage gate
    # before deck ownership is asked, so this pre-lookup warning state is the
    # only thing that can prevent a paid verdict for the now-broken project.
    unreadable_config, unreadable_staging, unreadable_deck, _unreadable_decision = (
        race_project("preflight-unreadable-deck")
    )
    unreadable_source = unreadable_config.root / "static.json"
    unreadable_source.write_text("[]", encoding="utf-8")
    unreadable_deck.write_text(
        "deck:\n"
        "  name: Static deck\n"
        '  source: "../../static.json"\n',
        encoding="utf-8",
    )
    unreadable_decision = coverage_application.plan_model_coverage(
        unreadable_config, unreadable_staging
    )
    refuses_after(
        unreadable_config,
        unreadable_decision,
        lambda: unreadable_source.write_text("{broken", encoding="utf-8"),
    )

    # Selector edits do not change declared ids or readability. They still
    # change the authoritative offline gate: this source goes from unassigned
    # to owned, so a fresh coverage refusal must not be mistaken for the old
    # provisional deck refusal.
    selector_config, _staged_path, selector_deck, selector_decision = race_project(
        "preflight-selector"
    )
    refuses_after(
        selector_config,
        selector_decision,
        lambda: selector_deck.write_text(
            "deck:\n"
            "  name: Now owns every card\n"
            '  source: "../../vocabulary.json"\n',
            encoding="utf-8",
        ),
    )

    # A selector can change only for the post-reading ownership gate and then
    # return to the rendered bytes before the final offline replan. Bind that
    # gate itself to the pre-jpdb deck revision; comparing the two outer plans
    # alone sees A on both sides and would buy a verdict for the transient B.
    aba_config, _staged_path, aba_deck, aba_decision = race_project(
        "preflight-selector-a-b-a"
    )
    deck_a = aba_deck.read_bytes()
    deck_b = (
        b"deck:\n"
        b"  name: Transient owner\n"
        b'  source: "../../vocabulary.json"\n'
    )
    original_ownership = promotion_application.require_exact_deck_ownership
    ownership_calls = 0

    def swap_during_ownership(*args: Any, **kwargs: Any) -> object:
        nonlocal ownership_calls
        ownership_calls += 1
        if ownership_calls == 3:
            aba_deck.write_bytes(deck_b)
            try:
                return original_ownership(*args, **kwargs)
            finally:
                aba_deck.write_bytes(deck_a)
        return original_ownership(*args, **kwargs)

    with monkeypatch.context() as race_patch:
        race_patch.setattr(
            promotion_application,
            "require_exact_deck_ownership",
            swap_during_ownership,
        )
        refuses_gate_swap(aba_config, aba_decision)
    assert aba_deck.read_bytes() == deck_a
    assert ownership_calls >= 4

    # The ledger is likewise first validated after jpdb. A corrupt A swapped
    # for valid B only during that load must not reach coverage and then pass
    # the final A-to-A outer comparison.
    ledger_aba_config, ledger_aba_staging, ledger_aba_deck, _decision = race_project(
        "preflight-ledger-a-b-a"
    )
    ledger_aba_deck.write_bytes(deck_b)
    corrupt_ledger = b"{broken"
    ledger_aba_config.ledger_file.write_bytes(corrupt_ledger)
    ledger_aba_decision = coverage_application.plan_model_coverage(
        ledger_aba_config, ledger_aba_staging
    )
    original_decide = coverage_application.decide_promotion
    decision_calls = 0

    def restore_ledger_after_consulted(*args: Any, **kwargs: Any) -> object:
        nonlocal decision_calls
        result = original_decide(*args, **kwargs)
        decision_calls += 1
        if decision_calls == 3:
            ledger_aba_config.ledger_file.write_bytes(corrupt_ledger)
        return result

    with monkeypatch.context() as race_patch:
        race_patch.setattr(
            coverage_application,
            "decide_promotion",
            restore_ledger_after_consulted,
        )
        refuses_gate_swap(
            ledger_aba_config,
            ledger_aba_decision,
            during_lookup=lambda: ledger_aba_config.ledger_file.write_bytes(b"{}\n"),
            match="Could not parse ledger",
        )
    assert ledger_aba_config.ledger_file.read_bytes() == corrupt_ledger
    assert decision_calls == 4

    ledger_config, _staged_path, _deck, ledger_decision = race_project(
        "preflight-ledger"
    )
    refuses_after(
        ledger_config,
        ledger_decision,
        lambda: ledger_config.ledger_file.write_text(
            '{"changed":"during jpdb"}\n', encoding="utf-8"
        ),
    )

    # Archive identity is exact wire, not only parsed rows and metadata. A
    # comment-only replacement during jpdb must stale the paid preflight too.
    archive_wire_config, archive_wire_staged, _deck, _decision = race_project(
        "preflight-archive-wire", reaccept=True, partial_archive=True
    )
    archive_wire_records, archive_wire_meta = read_staging(archive_wire_staged)
    archive_wire_done = archive_wire_config.staging_dir / "done"
    archive_wire_done.mkdir(parents=True, exist_ok=True)
    archive_wire_path = archive_wire_done / archive_wire_staged.name
    write_staging(
        archive_wire_path,
        [archive_wire_records[0]],
        promote_module.archive_meta(archive_wire_meta, 1),
    )
    archive_wire_decision = coverage_application.plan_model_coverage(
        archive_wire_config,
        archive_wire_staged,
        reaccept=True,
    )
    refuses_after(
        archive_wire_config,
        archive_wire_decision,
        lambda: archive_wire_path.write_bytes(
            archive_wire_path.read_bytes() + b"\n# exact-wire change\n"
        ),
    )

    # A re-check can start from already-approved coverage and a valid partial
    # archive can appear during jpdb. Bind both its rows and metadata before
    # buying the replacement verdict.
    archive_config, archive_staged, _deck, archive_decision = race_project(
        "preflight-archive", reaccept=True, partial_archive=True
    )

    def create_archive() -> None:
        records, meta = read_staging(archive_staged)
        done = archive_config.staging_dir / "done"
        done.mkdir(parents=True, exist_ok=True)
        write_staging(
            done / archive_staged.name,
            [records[0]],
            promote_module.archive_meta(meta, 1),
        )

    refuses_after(archive_config, archive_decision, create_archive)


def test_a_paid_answer_is_kept_when_the_review_changes_during_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale approval loses the write race, never the already-paid answer."""
    from japanese_anki import operations
    from japanese_anki.application import coverage as coverage_application
    from japanese_anki.config import ProjectConfig
    from japanese_anki.staging import read_staging

    root, staged = project_with_source(tmp_path)
    config = ProjectConfig.load(root)
    decision = coverage_application.plan_model_coverage(config, staged)

    def review(*_args: Any, capture=None, **_kwargs: Any) -> coverage.CoverageVerdict:
        assert capture is not None
        capture({"content": [{"type": "text", "text": "paid answer"}]})
        staged.write_text(
            staged.read_text(encoding="utf-8") + "# reviewer edit\n",
            encoding="utf-8",
        )
        return coverage.CoverageVerdict(
            True,
            "Every source row is accounted for.",
            decision.model,
            decision.prompt_fingerprint,
        )

    monkeypatch.setattr(coverage_application.coverage, "review_coverage", review)

    with pytest.raises(coverage_application.CoverageRunError) as raised:
        coverage_application.run_model_coverage(config, decision)

    [entry] = operations.OperationJournal.load(
        config.operations_file
    ).operations.values()
    assert entry.state == "result_captured"
    assert raised.value.operation_id == entry.operation_id
    assert f"operations --show-reply {entry.operation_id}" in str(raised.value)
    _records, meta = read_staging(staged)
    assert "approval" not in meta["coverage"]


def test_approval_write_is_reported_when_only_journal_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from japanese_anki import operations
    from japanese_anki.application import coverage as coverage_application
    from japanese_anki.config import ProjectConfig
    from japanese_anki.staging import read_staging

    root, staged = project_with_source(tmp_path)
    config = ProjectConfig.load(root)
    decision = coverage_application.plan_model_coverage(config, staged)

    def review(*_args: Any, capture=None, **_kwargs: Any) -> coverage.CoverageVerdict:
        assert capture is not None
        capture({"content": [{"type": "text", "text": "paid answer"}]})
        return coverage.CoverageVerdict(
            True,
            "Every source row is accounted for.",
            decision.model,
            decision.prompt_fingerprint,
        )

    real_move = operations.OperationJournal._move_under_lock

    def fail_committed_write(
        self: operations.OperationJournal,
        current: operations.OperationJournal,
        operation_id: str,
        state: str,
        **kwargs: Any,
    ) -> operations.Operation:
        if state == "committed":
            raise operations.OperationError("journal commit write failed")
        return real_move(self, current, operation_id, state, **kwargs)

    monkeypatch.setattr(coverage_application.coverage, "review_coverage", review)
    monkeypatch.setattr(
        operations.OperationJournal,
        "_move_under_lock",
        fail_committed_write,
    )

    with pytest.raises(coverage_application.CoverageRunError) as raised:
        coverage_application.run_model_coverage(config, decision, client=object())

    assert raised.value.provider_dispatched is True
    assert raised.value.approval_write == "written"
    assert raised.value.operation_write == "present"
    [entry] = operations.OperationJournal.load(config.operations_file).operations.values()
    assert entry.state == "result_captured"
    _records, meta = read_staging(staged)
    assert meta["coverage"]["approval"]["authority"] == "model"


def test_lost_committed_ack_reports_the_durable_committed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from japanese_anki import operations
    from japanese_anki.application import coverage as coverage_application
    from japanese_anki.config import ProjectConfig

    root, staged = project_with_source(tmp_path)
    config = ProjectConfig.load(root)
    decision = coverage_application.plan_model_coverage(config, staged)

    def review(*_args: Any, capture=None, **_kwargs: Any) -> coverage.CoverageVerdict:
        assert capture is not None
        capture({"content": [{"type": "text", "text": "paid answer"}]})
        return coverage.CoverageVerdict(
            True, "Accounted for.", decision.model, decision.prompt_fingerprint
        )

    real_move = operations.OperationJournal._move_under_lock

    def commit_then_fail(
        self: operations.OperationJournal,
        current: operations.OperationJournal,
        operation_id: str,
        state: str,
        **kwargs: Any,
    ) -> operations.Operation:
        result = real_move(self, current, operation_id, state, **kwargs)
        if state == "committed":
            raise operations.OperationError("commit acknowledgement lost")
        return result

    monkeypatch.setattr(coverage_application.coverage, "review_coverage", review)
    monkeypatch.setattr(
        operations.OperationJournal, "_move_under_lock", commit_then_fail
    )
    with pytest.raises(coverage_application.CoverageRunError) as raised:
        coverage_application.run_model_coverage(config, decision, client=object())

    assert raised.value.approval_write == "written"
    assert raised.value.operation_state == "committed"
    [entry] = operations.OperationJournal.load(config.operations_file).operations.values()
    assert entry.state == "committed"


def test_approval_write_is_unknown_when_its_acknowledgement_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from japanese_anki import operations, staging
    from japanese_anki.application import coverage as coverage_application
    from japanese_anki.config import ProjectConfig
    from japanese_anki.staging import read_staging

    root, staged = project_with_source(tmp_path)
    config = ProjectConfig.load(root)
    decision = coverage_application.plan_model_coverage(config, staged)

    def review(*_args: Any, capture=None, **_kwargs: Any) -> coverage.CoverageVerdict:
        assert capture is not None
        capture({"content": [{"type": "text", "text": "paid answer"}]})
        return coverage.CoverageVerdict(
            True,
            "Every source row is accounted for.",
            decision.model,
            decision.prompt_fingerprint,
        )

    real_record = staging.record_coverage_approval_under_lock

    def write_then_fail(*args: Any, **kwargs: Any) -> None:
        real_record(*args, **kwargs)
        raise staging.StagingError("approval write acknowledgement failed")

    monkeypatch.setattr(coverage_application.coverage, "review_coverage", review)
    monkeypatch.setattr(
        staging,
        "record_coverage_approval_under_lock",
        write_then_fail,
    )

    with pytest.raises(coverage_application.CoverageRunError) as raised:
        coverage_application.run_model_coverage(config, decision, client=object())

    assert raised.value.provider_dispatched is True
    assert raised.value.approval_write == "unknown"
    assert raised.value.operation_write == "present"
    [entry] = operations.OperationJournal.load(config.operations_file).operations.values()
    assert entry.state == "result_captured"
    _records, meta = read_staging(staged)
    assert meta["coverage"]["approval"]["authority"] == "model"


@pytest.mark.parametrize(
    ("phase", "durable_state"),
    [("authorize", "authorized"), ("dispatching", "dispatching")],
)
def test_journal_boundary_write_failures_preserve_possible_operation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    durable_state: str,
) -> None:
    from japanese_anki import operations
    from japanese_anki.application import coverage as coverage_application
    from japanese_anki.config import ProjectConfig

    root, staged = project_with_source(tmp_path)
    config = ProjectConfig.load(root)
    decision = coverage_application.plan_model_coverage(config, staged)
    sent: list[object] = []
    monkeypatch.setattr(
        coverage_application.coverage,
        "review_coverage",
        lambda *_args, **_kwargs: sent.append(object()),
    )

    if phase == "authorize":
        real_authorize = operations.OperationJournal.authorize

        def write_then_fail(
            self: operations.OperationJournal, *args: Any, **kwargs: Any
        ) -> operations.Operation:
            real_authorize(self, *args, **kwargs)
            raise operations.OperationError("authority write acknowledgement failed")

        monkeypatch.setattr(operations.OperationJournal, "authorize", write_then_fail)
    else:
        real_advance = operations.OperationJournal.advance

        def advance_then_fail(
            self: operations.OperationJournal,
            operation_id: str,
            state: str,
            **kwargs: Any,
        ) -> operations.Operation:
            result = real_advance(self, operation_id, state, **kwargs)
            if state == "dispatching":
                raise operations.OperationError(
                    "dispatch write acknowledgement failed"
                )
            return result

        monkeypatch.setattr(operations.OperationJournal, "advance", advance_then_fail)

    with pytest.raises(coverage_application.CoverageRunError) as raised:
        coverage_application.run_model_coverage(config, decision, client=object())

    assert raised.value.provider_dispatched is False
    assert raised.value.operation_write == "present"
    assert raised.value.approval_write == "not_written"
    assert sent == []
    [entry] = operations.OperationJournal.load(config.operations_file).operations.values()
    assert entry.state == durable_state
