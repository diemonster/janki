"""The last gate — ``janki review`` and the build that consults it.

Every test drives the client seam, so none reaches a model. The gate's whole
value is that it cannot be walked past, so most of what is pinned here is what
happens when someone tries: an unreviewed card, an edited card, an acceptance
that no longer applies to the text it was given for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import cli
from japanese_anki import review as review_module
from japanese_anki.claude_client import CallResult
from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.review import (
    CardReview,
    Finding,
    ReviewError,
    card_fingerprint,
    load_store,
    open_findings,
    save_store,
    unreviewed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = """
[paths]
normalized_file = "vocabulary.json"
deck_dir = "decks"
media_dir = "media"
template_dir = "templates/japanese-study"
dist_dir = "dist"
ledger_file = "ledger.json"
review_file = "review.json"
"""


def record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "verb_group": "godan",
        "source": SourceReference(type="shirabe", imported_from="export.csv"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def project(tmp_path: Path, records: list[VocabularyRecord], *, require: bool = True) -> Path:
    import shutil

    config = CONFIG if require else CONFIG + "\n[review]\nrequire = false\n"
    (tmp_path / "janki.toml").write_text(config, encoding="utf-8")
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([r.to_dict() for r in records], ensure_ascii=False), encoding="utf-8"
    )
    shutil.copytree(
        PROJECT_ROOT / "templates" / "japanese-study",
        tmp_path / "templates" / "japanese-study",
    )
    (tmp_path / "decks").mkdir()
    (tmp_path / "decks" / "verbs.yaml").write_text(
        "deck:\n  name: Test\n  source: ../vocabulary.json\nnotes: []\n", encoding="utf-8"
    )
    return tmp_path


class Verdict:
    def __init__(self, *findings: Any) -> None:
        self.findings = list(findings)


class Item:
    def __init__(self, where: str, problem: str, severity: str = "note",
                 suggestion: str = "") -> None:
        self.where, self.problem = where, problem
        self.severity, self.suggestion = severity, suggestion


def reader(*verdicts: Any, calls: list[str] | None = None):
    """Replaces `claude_client.parse_call`, one verdict per card in order."""
    queue = list(verdicts)

    def parse_call(_model, _blocks, content, *_args, **_kwargs) -> CallResult:
        if calls is not None:
            calls.append(
                "".join(b.get("text", "") for b in content if isinstance(b, dict))
            )
        answer = queue.pop(0) if queue else Verdict()
        return CallResult(parsed=answer, stop_reason="end_turn", refusal=None)

    return parse_call


@pytest.fixture(autouse=True)
def no_style_guide(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.claude_client, "read_style_guide", lambda _root: "guide")


def store_of(root: Path) -> dict:
    return json.loads((root / "review.json").read_text(encoding="utf-8"))


# --- what makes a card unreviewed ---------------------------------------------


def test_a_card_nobody_has_read_is_unreviewed() -> None:
    assert [r.id for r in unreviewed([record()], {})] == ["word:話す:はなす"]


def test_editing_a_card_makes_it_unreviewed_again() -> None:
    """The entry describes a card that no longer exists. Carrying the verdict
    forward would pass text nobody read."""
    original = record()
    store = {
        card_fingerprint(original): CardReview(
            original.id, card_fingerprint(original), "2026-08-09"
        )
    }
    assert unreviewed([original], store) == []

    edited = record(examples=[ExampleSentence(japanese="毎日話します。")])

    assert [r.id for r in unreviewed([edited], store)] == [original.id]


def test_a_field_the_card_does_not_show_does_not_invalidate_a_review() -> None:
    """Re-importing the same word from a second deck changes `source` and tags,
    and re-reviewing twenty cards to reach the same answer is a request each."""
    original = record()
    store = {
        card_fingerprint(original): CardReview(
            original.id, card_fingerprint(original), "2026-08-09"
        )
    }
    same_card = record(
        source=SourceReference(type="jpdb", imported_from="deck-4"),
        tags=["jpdb"],
        audio="audio/janki-abc.mp3",
    )

    assert unreviewed([same_card], store) == []


# --- which findings block ------------------------------------------------------


def test_only_an_error_blocks() -> None:
    """A model asked to find fault will always find some. A gate that stops on
    "could be more natural" is one people learn to bypass."""
    card = record()
    store = {card_fingerprint(card): CardReview(
        card.id, card_fingerprint(card),
        findings=(Finding("examples[0]", "could be more natural", "note"),),
    )}

    assert open_findings([card], store) == []


def test_an_error_blocks_until_accepted() -> None:
    card = record()
    entry = CardReview(card.id, card_fingerprint(card), findings=(
        Finding("meanings", "glossed as intransitive", "error"),
    ))

    assert len(open_findings([card], {card_fingerprint(card): entry})) == 1

    accepted = review_module.accept(
        {card_fingerprint(card): entry}, card.id, "the gloss is right"
    )

    assert open_findings([card], accepted) == []


def test_an_acceptance_does_not_survive_an_edit() -> None:
    """It was an acceptance of a finding about text that has since changed;
    carrying it forward would wave through words nobody agreed to."""
    card = record()
    accepted = review_module.accept(
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card), accepted=False,
            findings=(Finding("meanings", "wrong", "error"),),
        )},
        card.id,
        "checked it",
    )
    edited = record(meanings=["to speak", "to tell"])

    assert open_findings([edited], accepted) == [], "no finding against the new text"
    assert [r.id for r in unreviewed([edited], accepted)] == [card.id], "but unread"


def test_an_acceptance_needs_a_reason() -> None:
    card = record()
    store = {card_fingerprint(card): CardReview(card.id, card_fingerprint(card))}

    with pytest.raises(ReviewError, match="needs a reason"):
        review_module.accept(store, card.id, "   ")


def test_a_severity_the_model_invents_blocks() -> None:
    """A finding janki cannot classify is not one it may wave through."""
    assert Finding.from_dict({"where": "x", "problem": "y", "severity": "critical"}).severity == (
        "error"
    )


# --- the command ---------------------------------------------------------------


def test_a_clean_card_is_recorded_and_the_build_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    monkeypatch.setattr(review_module.claude_client, "parse_call", reader(Verdict()))

    assert cli.main(["--root", str(root), "review"]) == 0
    assert [e["findings"] for e in store_of(root).values()] == [[]]

    assert cli.main(["--root", str(root), "build", "verbs"]) == 0


def test_an_error_stops_the_build_and_says_what_and_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    monkeypatch.setattr(
        review_module.claude_client,
        "parse_call",
        reader(Verdict(Item("meanings", "glossed as intransitive", "error", "to speak (vt)"))),
    )
    assert cli.main(["--root", str(root), "review"]) == 1
    # Discarded, so the assertions below are about the *build's* output. The
    # review prints the same finding, so without this the test passed whether
    # or not the build said anything at all.
    capsys.readouterr()

    assert cli.main(["--root", str(root), "build", "verbs"]) == 1

    err = capsys.readouterr().err
    assert "glossed as intransitive" in err
    assert "janki review --accept" in err


def test_an_unreviewed_card_stops_the_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half: a card that failed the gate and one that never reached it
    are both unfit to ship, and the message says which, because the remedies
    differ."""
    root = project(tmp_path, [record()])

    assert cli.main(["--root", str(root), "build", "verbs"]) == 1

    err = capsys.readouterr().err
    assert "have not been read since they last changed" in err
    assert "janki review" in err


def test_accepting_lets_the_build_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [record()])
    monkeypatch.setattr(
        review_module.claude_client,
        "parse_call",
        reader(Verdict(Item("meanings", "glossed as intransitive", "error"))),
    )
    cli.main(["--root", str(root), "review"])

    assert cli.main([
        "--root", str(root), "review", "--accept", "word:話す:はなす",
        "--because", "the gloss matches jpdb",
    ]) == 0

    assert cli.main(["--root", str(root), "build", "verbs"]) == 0
    assert [e["accepted_because"] for e in store_of(root).values()] == [
        "the gloss matches jpdb"
    ]


def test_a_second_run_reads_nothing_that_has_not_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fingerprint is what makes this affordable: one request per changed
    card, not one per build."""
    calls: list[str] = []
    root = project(tmp_path, [record()])
    monkeypatch.setattr(
        review_module.claude_client, "parse_call", reader(Verdict(), calls=calls)
    )
    cli.main(["--root", str(root), "review"])
    assert len(calls) == 1

    assert cli.main(["--root", str(root), "review"]) == 0

    assert len(calls) == 1, "nothing changed, so nothing was re-read"
    assert "already read" in capsys.readouterr().out


def test_force_reads_them_all_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    root = project(tmp_path, [record()])
    monkeypatch.setattr(
        review_module.claude_client,
        "parse_call",
        reader(Verdict(), Verdict(), calls=calls),
    )
    cli.main(["--root", str(root), "review"])

    cli.main(["--root", str(root), "review", "--force"])

    assert len(calls) == 2


def test_the_card_reaches_the_model_as_a_reader_sees_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Including the register each example claims, which is one of the things
    only a reader can check."""
    calls: list[str] = []
    root = project(tmp_path, [record(examples=[
        ExampleSentence(japanese="毎日話します。", english="I speak every day.",
                        register="polite"),
    ])])
    monkeypatch.setattr(
        review_module.claude_client, "parse_call", reader(Verdict(), calls=calls)
    )

    cli.main(["--root", str(root), "review"])

    assert "毎日話します。" in calls[0]
    assert "(polite)" in calls[0]
    assert "to speak" in calls[0]


def test_a_truncated_answer_is_refused_not_read_as_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cut-off answer looks exactly like "nothing wrong", and this is the
    check standing between a wrong card and a learner."""
    root = project(tmp_path, [record()])

    def truncated(*_args: Any, **_kwargs: Any) -> CallResult:
        return CallResult(parsed=None, stop_reason="max_tokens", refusal=None)

    monkeypatch.setattr(review_module.claude_client, "parse_call", truncated)

    assert cli.main(["--root", str(root), "review"]) == 1


def test_a_throwaway_build_is_not_gated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--output` writes a scratch package and records nothing — `make gates`
    builds one on every run. Holding it to a review nobody asked for would make
    the gate something to work around."""
    root = project(tmp_path, [record()])

    assert cli.main([
        "--root", str(root), "build", str(root / "decks" / "verbs.yaml"),
        "--output", str(tmp_path / "scratch.apkg"),
    ]) == 0
    assert (tmp_path / "scratch.apkg").exists()


def test_a_project_that_turns_the_gate_off_builds_unreviewed(
    tmp_path: Path
) -> None:
    root = project(tmp_path, [record()], require=False)

    assert cli.main(["--root", str(root), "build", "verbs"]) == 0


# --- the store -----------------------------------------------------------------


def test_the_store_round_trips(tmp_path: Path) -> None:
    entry = CardReview(
        "word:話す:はなす", "abc123", "2026-08-09",
        findings=(Finding("meanings", "wrong", "error", "fix it"),),
        accepted=True, accepted_because="checked",
    )
    path = tmp_path / "review.json"

    save_store(path, {entry.content_fp: entry})

    assert load_store(path)[entry.content_fp] == entry


def test_a_missing_store_is_empty_not_an_error(tmp_path: Path) -> None:
    assert load_store(tmp_path / "nothing.json") == {}


def test_an_entry_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    """`save_store` rewrites the whole file from what was loaded, so an entry
    the loader dropped would be erased on the next write and its card would
    silently become unreviewed."""
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"abc123": None}), encoding="utf-8")

    with pytest.raises(ReviewError, match="must be an object"):
        load_store(path)


def test_a_finding_about_text_that_has_changed_is_not_reported_against_the_new_text() -> None:
    """The build refuses either way — an edited card is unreviewed — but it must
    refuse for the true reason. Reporting the old finding would send someone to
    fix a sentence that is no longer on the card."""
    card = record(examples=[ExampleSentence(japanese="毎日話します。")])
    store = {card_fingerprint(card): CardReview(
        card.id, card_fingerprint(card),
        findings=(Finding("examples[0]", "毎日話します。 is unnatural", "error"),),
    )}
    assert len(open_findings([card], store)) == 1

    edited = record(examples=[ExampleSentence(japanese="友だちと話しました。")])

    assert open_findings([edited], store) == [], "the finding was about other text"
    assert [r.id for r in unreviewed([edited], store)] == [card.id], "and it is unread"


def test_one_card_that_cannot_be_read_does_not_discard_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Twenty cards is twenty requests. Raising on the twentieth threw away
    nineteen answers already paid for — the "half a ledger" failure `janki
    audio` writes as it goes to avoid."""
    root = project(tmp_path, [
        record(),
        record(id="word:食べる:たべる", expression="食べる", reading="たべる",
               meanings=["to eat"], verb_group="ichidan"),
    ])
    answers = [
        CallResult(parsed=Verdict(), stop_reason="end_turn", refusal=None),
        CallResult(parsed=None, stop_reason="max_tokens", refusal=None),
    ]

    def flaky(*_args: Any, **_kwargs: Any) -> CallResult:
        return answers.pop(0)

    monkeypatch.setattr(review_module.claude_client, "parse_call", flaky)

    assert cli.main(["--root", str(root), "review"]) == 1

    stored = store_of(root)
    assert [e["record_id"] for e in stored.values()] == ["word:話す:はなす"], (
        "the one that worked was kept"
    )
    assert "could not read word:食べる:たべる" in capsys.readouterr().err


def test_a_card_that_could_not_be_read_still_stops_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent from the store means unreviewed, so nothing downstream has to
    remember that a failure is not a pass."""
    root = project(tmp_path, [record()])

    def truncated(*_args: Any, **_kwargs: Any) -> CallResult:
        return CallResult(parsed=None, stop_reason="max_tokens", refusal=None)

    monkeypatch.setattr(review_module.claude_client, "parse_call", truncated)
    cli.main(["--root", str(root), "review"])

    assert cli.main(["--root", str(root), "build", "verbs"]) == 1


# --- what the build actually ships --------------------------------------------


def test_a_deck_local_override_is_reviewed_as_its_own_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`janki build` resolves a deck's inline `notes:` over the normalized
    record; `janki review` used to read the normalized file. The two saw
    different text under the same id, so the build refused a card the review had
    just called clean — and no flag reached it. Not `--force`, which rewrote the
    normalized fingerprint; not `--accept`, which `unreviewed` ignores. The deck
    could never be built again."""
    root = project(tmp_path, [record()])
    (root / "decks" / "verbs.yaml").write_text(
        "deck:\n  name: Test\n  source: ../vocabulary.json\n"
        'notes:\n  - id: "word:話す:はなす"\n    usage_notes: "Deck-local note."\n',
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        review_module.claude_client, "parse_call", reader(Verdict(), calls=calls)
    )

    assert cli.main(["--root", str(root), "review"]) == 0

    assert "Deck-local note." in calls[0], "the card the build will ship"
    assert cli.main(["--root", str(root), "build", "verbs"]) == 0


def test_two_decks_shipping_one_record_differently_are_both_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keyed by record id, one deck's version overwrote the other's and the
    loser was refused forever. A review is of a card *version*."""
    root = project(tmp_path, [record()])
    (root / "decks" / "other.yaml").write_text(
        "deck:\n  name: Other\n  source: ../vocabulary.json\n"
        'notes:\n  - id: "word:話す:はなす"\n    usage_notes: "Other deck."\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        review_module.claude_client, "parse_call", reader(Verdict(), Verdict())
    )

    assert cli.main(["--root", str(root), "review"]) == 0

    assert len(store_of(root)) == 2, "one entry per version"
    assert cli.main(["--root", str(root), "build", "--all"]) == 0


def test_reverting_an_edit_restores_its_review_for_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pleasant consequence of keying by version: the old answer is still
    there, so undoing a change costs no request."""
    calls: list[str] = []
    root = project(tmp_path, [record()])
    monkeypatch.setattr(
        review_module.claude_client,
        "parse_call",
        reader(Verdict(), Verdict(), calls=calls),
    )
    cli.main(["--root", str(root), "review"])
    (root / "vocabulary.json").write_text(
        json.dumps([record(usage_notes="edited").to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )
    cli.main(["--root", str(root), "review"])
    assert len(calls) == 2

    (root / "vocabulary.json").write_text(
        json.dumps([record().to_dict()], ensure_ascii=False), encoding="utf-8"
    )

    assert cli.main(["--root", str(root), "review"]) == 0
    assert len(calls) == 2, "the first version's answer was still on record"


def test_transitivity_is_part_of_the_card_that_was_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate's first real run flagged a verb "glossed as intransitive", so it
    is exactly the kind of claim only a reader checks — and a card whose
    transitivity changed has not been read."""
    original = record(transitivity="transitive")
    store = {card_fingerprint(original): CardReview(
        original.id, card_fingerprint(original)
    )}

    assert unreviewed([original], store) == []
    assert [r.id for r in unreviewed([record(transitivity="intransitive")], store)] == [
        original.id
    ]


def test_an_accept_naming_an_unknown_id_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It printed "Accepted …" for the first id, then raised on the second
    before saving — so the card the user was just told was cleared was refused
    by the very next build."""
    root = project(tmp_path, [record()])
    monkeypatch.setattr(
        review_module.claude_client,
        "parse_call",
        reader(Verdict(Item("meanings", "wrong", "error"))),
    )
    cli.main(["--root", str(root), "review"])
    capsys.readouterr()

    assert cli.main([
        "--root", str(root), "review",
        "--accept", "word:話す:はなす", "--accept", "word:nope:nope",
        "--because", "checked",
    ]) == 1

    out = capsys.readouterr()
    assert "Accepted" not in out.out, "nothing was announced"
    assert cli.main(["--root", str(root), "build", "verbs"]) == 1, "and nothing saved"


def test_a_finding_with_no_text_is_not_dropped() -> None:
    """It was dropped *and* the card recorded as read, so the build shipped it —
    the opposite of what this module does with a severity it cannot classify."""
    from japanese_anki.review import Finding as F

    assert F.from_dict({"where": "meanings", "problem": "", "severity": "error"}).severity == (
        "error"
    )


# --- the refresh pipeline ------------------------------------------------------


def test_refresh_reads_the_cards_between_voicing_and_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Last before the build, because it reads the *finished* card: the
    sentences `--ai` wrote and the readings `--jpdb` filled. Reviewing earlier
    would read a card that does not exist yet and pass it."""
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    root = project(tmp_path, [record()])
    called: list[str] = []

    monkeypatch.setattr(cli, "command_enrich", lambda a: called.append("enrich") or 0)
    monkeypatch.setattr(cli, "command_audio", lambda a: called.append("audio") or 0)
    monkeypatch.setattr(cli, "command_review", lambda a: called.append("review") or 0)
    monkeypatch.setattr(cli, "command_build", lambda a: called.append("build") or 0)

    assert cli.main(["--root", str(root), "refresh"]) == 0

    assert called.index("review") > called.index("audio")
    assert called.index("review") < called.index("build")


def test_a_refusal_from_the_review_stage_stops_refresh_before_it_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the package is written *and marked exported*, and the next
    --only-new never revisits the card the gate objected to."""
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    root = project(tmp_path, [record()])
    built: list[str] = []

    monkeypatch.setattr(cli, "command_enrich", lambda a: 0)
    monkeypatch.setattr(cli, "command_audio", lambda a: 0)
    monkeypatch.setattr(cli, "command_review", lambda a: 1)
    monkeypatch.setattr(cli, "command_build", lambda a: built.append("built") or 0)

    assert cli.main(["--root", str(root), "refresh"]) != 0

    assert built == [], "nothing was packaged"


def test_no_review_skips_the_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    root = project(tmp_path, [record()])
    called: list[str] = []
    monkeypatch.setattr(cli, "command_enrich", lambda a: 0)
    monkeypatch.setattr(cli, "command_audio", lambda a: 0)
    monkeypatch.setattr(cli, "command_review", lambda a: called.append("review") or 0)
    monkeypatch.setattr(cli, "command_build", lambda a: 0)

    cli.main(["--root", str(root), "refresh", "--no-review"])

    assert called == []
    assert "--no-review" in capsys.readouterr().out


def test_a_project_with_the_gate_off_does_not_pay_for_the_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    root = project(tmp_path, [record()], require=False)
    called: list[str] = []
    monkeypatch.setattr(cli, "command_enrich", lambda a: 0)
    monkeypatch.setattr(cli, "command_audio", lambda a: 0)
    monkeypatch.setattr(cli, "command_review", lambda a: called.append("review") or 0)
    monkeypatch.setattr(cli, "command_build", lambda a: 0)

    cli.main(["--root", str(root), "refresh"])

    assert called == []
    assert "[review] require = false" in capsys.readouterr().out


def test_the_model_reported_is_the_model_billed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It printed the configured model and called the overridden one, with
    nothing on record saying which had actually read the card."""
    root = project(tmp_path, [record()])
    used: list[str] = []

    def note_model(model, *_args: Any, **_kwargs: Any) -> CallResult:
        used.append(model)
        return CallResult(parsed=Verdict(), stop_reason="end_turn", refusal=None)

    monkeypatch.setattr(review_module.claude_client, "parse_call", note_model)

    cli.main(["--root", str(root), "review", "--model", "claude-haiku-4-5-20251001"])

    assert used == ["claude-haiku-4-5-20251001"]
    assert "with claude-haiku-4-5-20251001" in capsys.readouterr().out
