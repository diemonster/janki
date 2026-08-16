"""Letting a model answer the coverage question, and the record it leaves.

The owner's decision moved up a level — from "is this page accounted for" to
"may a model answer that" — so the tests here are mostly about *provenance and
refusal*: that a verdict nobody really gave is never recorded as one, and that
a card promoted on a model's word says so permanently.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from japanese_anki import coverage, prompts
from japanese_anki.claude_client import CallResult
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

    from conftest import seed_prompts
    from japanese_anki.extract import ExtractionResult, SourceUnit, coverage_block
    from japanese_anki.staging import coverage_block_fingerprint

    (tmp_path / "janki.toml").write_text(
        '[paths]\nnormalized_file = "vocabulary.json"\n'
        'staging_dir = "staging"\nscan_inbox = "inbox"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    seed_prompts(tmp_path)
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
        "  mode: table\n  provider: anthropic\n  model: m\n"
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
    from japanese_anki import cli
    from japanese_anki.staging import coverage_block_fingerprint, read_staging

    root, staged = project_with_source(tmp_path)
    monkeypatch.setattr(
        cli.coverage, "review_coverage",
        lambda *a, **k: coverage.CoverageVerdict(True, "Accounted for.", "m", "f" * 64),
    )
    _records, meta = read_staging(staged)
    # A file whose stored fingerprint disagrees with its own facts is refused
    # outright now, so the honest check is that what we write matches the
    # recomputed value rather than whatever the file happened to carry.
    cli._model_accepts_coverage(  # noqa: SLF001
        cli._load_config(SimpleNamespace(root=root)), staged, meta
    )

    _records, after = read_staging(staged)
    block = after["coverage"]
    assert block["approval"]["coverage_block_fingerprint"] == (
        coverage_block_fingerprint(block)
    )


def test_a_block_the_gate_would_reject_is_never_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate first, spend second.

    A stale block fingerprint is free to detect and fatal either way. Finding
    out after the call meant paying for a verdict, writing it into the file,
    and *then* refusing — leaving a staging file that could neither be
    promoted nor re-approved without hand-deleting the approval it had just
    been given.
    """
    from japanese_anki import cli
    from japanese_anki.staging import StagingError, read_staging

    root, staged = project_with_source(tmp_path)
    staged.write_text(
        staged.read_text(encoding="utf-8").replace("row 1", "row 1 EDITED"),
        encoding="utf-8",
    )
    calls: list[object] = []
    monkeypatch.setattr(
        cli.coverage, "review_coverage",
        lambda *a, **k: calls.append(1) or coverage.CoverageVerdict(True, "ok", "m", "f"),
    )
    _records, meta = read_staging(staged)

    with pytest.raises(StagingError, match="coverage-block-stale"):
        cli._model_accepts_coverage(  # noqa: SLF001
            cli._load_config(SimpleNamespace(root=root)), staged, meta
        )

    assert calls == [], "nothing was sent"
    assert "approval" not in staged.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "reason",
    ["no", "on", "off", "yes", "12:30", "1.0"],
    ids=["no", "on", "off", "yes", "sexagesimal", "float"],
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
    from japanese_anki import cli
    from japanese_anki.staging import read_staging

    root, staged = project_with_source(tmp_path, mode="prose")
    calls: list[object] = []
    monkeypatch.setattr(
        cli.coverage, "review_coverage",
        lambda *a, **k: calls.append(1) or coverage.CoverageVerdict(True, "ok", "m", "f"),
    )
    _records, meta = read_staging(staged)

    accepted = cli._model_accepts_coverage(  # noqa: SLF001
        cli._load_config(SimpleNamespace(root=root)), staged, meta
    )

    assert accepted is False
    assert calls == [], "nothing was sent"
    assert "approval" not in staged.read_text(encoding="utf-8")
    assert "does not need a coverage approval" in capsys.readouterr().err
