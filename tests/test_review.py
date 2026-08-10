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
        original.id: CardReview(original.id, card_fingerprint(original), "2026-08-09")
    }
    assert unreviewed([original], store) == []

    edited = record(examples=[ExampleSentence(japanese="毎日話します。")])

    assert [r.id for r in unreviewed([edited], store)] == [original.id]


def test_a_field_the_card_does_not_show_does_not_invalidate_a_review() -> None:
    """Re-importing the same word from a second deck changes `source` and tags,
    and re-reviewing twenty cards to reach the same answer is a request each."""
    original = record()
    store = {
        original.id: CardReview(original.id, card_fingerprint(original), "2026-08-09")
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
    store = {card.id: CardReview(card.id, card_fingerprint(card), findings=(
        Finding("examples[0]", "could be more natural", "note"),
    ))}

    assert open_findings([card], store) == []


def test_an_error_blocks_until_accepted() -> None:
    card = record()
    entry = CardReview(card.id, card_fingerprint(card), findings=(
        Finding("meanings", "glossed as intransitive", "error"),
    ))

    assert len(open_findings([card], {card.id: entry})) == 1

    accepted = review_module.accept({card.id: entry}, card.id, "the gloss is right")

    assert open_findings([card], accepted) == []


def test_an_acceptance_does_not_survive_an_edit() -> None:
    """It was an acceptance of a finding about text that has since changed;
    carrying it forward would wave through words nobody agreed to."""
    card = record()
    accepted = review_module.accept(
        {card.id: CardReview(card.id, card_fingerprint(card), accepted=False, findings=(
            Finding("meanings", "wrong", "error"),
        ))},
        card.id,
        "checked it",
    )
    edited = record(meanings=["to speak", "to tell"])

    assert open_findings([edited], accepted) == [], "no finding against the new text"
    assert [r.id for r in unreviewed([edited], accepted)] == [card.id], "but unread"


def test_an_acceptance_needs_a_reason() -> None:
    card = record()
    store = {card.id: CardReview(card.id, card_fingerprint(card))}

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
    assert store_of(root)["word:話す:はなす"]["findings"] == []

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
    assert store_of(root)["word:話す:はなす"]["accepted_because"] == "the gloss matches jpdb"


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

    save_store(path, {entry.record_id: entry})

    assert load_store(path)[entry.record_id] == entry


def test_a_missing_store_is_empty_not_an_error(tmp_path: Path) -> None:
    assert load_store(tmp_path / "nothing.json") == {}


def test_an_entry_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    """`save_store` rewrites the whole file from what was loaded, so an entry
    the loader dropped would be erased on the next write and its card would
    silently become unreviewed."""
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"word:話す:はなす": None}), encoding="utf-8")

    with pytest.raises(ReviewError, match="must be an object"):
        load_store(path)


def test_a_finding_about_text_that_has_changed_is_not_reported_against_the_new_text() -> None:
    """The build refuses either way — an edited card is unreviewed — but it must
    refuse for the true reason. Reporting the old finding would send someone to
    fix a sentence that is no longer on the card."""
    card = record(examples=[ExampleSentence(japanese="毎日話します。")])
    store = {card.id: CardReview(card.id, card_fingerprint(card), findings=(
        Finding("examples[0]", "毎日話します。 is unnatural", "error"),
    ))}
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
    assert list(stored) == ["word:話す:はなす"], "the one that worked was kept"
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
