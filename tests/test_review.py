"""The last gate — ``janki review`` and the build that consults it.

Every test drives the client seam, so none reaches a model. The gate's whole
value is that it cannot be walked past, so most of what is pinned here is what
happens when someone tries: an unreviewed card, an edited card, an acceptance
that no longer applies to the text it was given for.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import cli
from japanese_anki import review as review_module
from japanese_anki.claude_client import CallResult
from japanese_anki.models import (
    LEARNER_LOAD_HOLD_KEY,
    ExampleSentence,
    SourceReference,
    VocabularyRecord,
    add_example_flags,
)
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
        # A finished card, because that is what review operates on: the
        # completeness gate refuses one an AI pass has not filled yet.
        "examples": [ExampleSentence(japanese="毎日話します。", english="I speak every day.")],
        "usage_notes": "A common verb.",
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


def test_local_validation_failures_are_reported_before_the_review_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """M7.6T build readiness: a deck that fails local validation *and* has
    unreviewed cards must name the local failure — it is fixable here for
    free, while the review gate's remedy can be a paid run that may only
    happen after every local gate passes."""
    from japanese_anki.models import ExampleSentence

    fragment = record(
        examples=[
            ExampleSentence(
                japanese="古い地図の話について", english="x", register="polite"
            )
        ]
    )
    root = project(tmp_path, [fragment])

    assert cli.main(["--root", str(root), "build", "verbs"]) == 1

    err = capsys.readouterr().err
    assert "example-fragment" in err
    assert "fails local validation" in err
    assert "have not been read" not in err


def test_a_saved_clean_review_cannot_answer_for_a_later_local_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from japanese_anki.models import ExampleSentence

    root = project(tmp_path, [record()])
    monkeypatch.setattr(review_module.claude_client, "parse_call", reader(Verdict()))
    assert cli.main(["--root", str(root), "review"]) == 0
    assert cli.main(["--root", str(root), "build", "verbs"]) == 0

    # A later edit introduces a local content failure. Whatever the review
    # store still says about this record, the build's answer must come from
    # validation, not from the saved result.
    broken = record(
        examples=[
            ExampleSentence(
                japanese="明日、九時に話します。", english="x", register="casual"
            )
        ]
    )
    (root / "vocabulary.json").write_text(
        json.dumps([broken.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    capsys.readouterr()

    assert cli.main(["--root", str(root), "build", "verbs"]) == 1

    err = capsys.readouterr().err
    assert "example-register-mismatch" in err


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
        "word:話す:はなす", "abc123def456", "2026-08-09",
        findings=(Finding("meanings", "wrong", "error", "fix it"),),
        accepted=True, accepted_because="checked",
        accepted_marks=("meanings|error",),
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


def test_a_finding_with_no_text_is_kept_and_given_some(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It was dropped *and* the card recorded as read, so the build shipped it —
    the opposite of what this module does with a severity it cannot classify.
    Asserted through `review_records`, which is where the dropping happened;
    `Finding.from_dict` never did it, so testing that pinned nothing."""
    monkeypatch.setattr(
        review_module.claude_client,
        "parse_call",
        reader(Verdict(Item("meanings", "", "error", "gloss it as transitive"))),
    )

    kept, failures = review_module.review_records(
        [record()], model="m", style_guide="guide"
    )
    entry = next(iter(kept.values()))

    assert failures == []
    assert [(f.severity, f.problem) for f in entry.findings] == [
        ("error", "gloss it as transitive"),
    ], "the suggestion stands in for the missing prose"


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


def test_an_old_store_keyed_by_record_id_is_migrated_not_misread(tmp_path: Path) -> None:
    """The store was keyed by record id before a review became a review of a
    card *version*. Read under the current schema every entry loaded with
    `content_fp` set to a record id and `record_id` empty — matching no card, so
    every card read as unreviewed, and the next save wrote the mistake back and
    dropped the real fingerprint. That is the erasure the non-object check
    exists to stop, arriving through the key instead of the value."""
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps({
            "word:話す:はなす": {
                "content_fp": "abc123def456",
                "at": "2026-08-09",
                "accepted": True,
                "accepted_because": "checked by hand",
                "findings": [{"where": "meanings", "problem": "wrong",
                              "severity": "error", "suggestion": ""}],
            }
        }),
        encoding="utf-8",
    )

    store = load_store(path)

    assert list(store) == ["abc123def456"], "keyed by the fingerprint it really had"
    assert store["abc123def456"].record_id == "word:話す:はなす"
    assert store["abc123def456"].accepted, "and the human's acceptance survived"


def test_a_migrated_store_survives_a_round_trip(tmp_path: Path) -> None:
    """Because `save_store` rewrites from what was loaded: a migration that only
    half worked would be committed as the new truth."""
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps({"word:話す:はなす": {"content_fp": "abc123def456", "findings": []}}),
        encoding="utf-8",
    )

    save_store(path, load_store(path))

    again = load_store(path)
    assert list(again) == ["abc123def456"]
    assert again["abc123def456"].record_id == "word:話す:はなす"


def test_an_acceptance_does_not_reach_a_version_no_deck_ships() -> None:
    """The store keeps every version ever read. Accepting by record id alone
    cleared findings on versions nothing ships, so a revert or a re-import
    brought back a card carrying a finding nobody read — annotated with a reason
    written about different words."""
    shipping = record()
    stale = record(usage_notes="an older version")
    store = {
        card_fingerprint(shipping): CardReview(
            shipping.id, card_fingerprint(shipping),
            findings=(Finding("meanings", "today's problem", "error"),),
        ),
        card_fingerprint(stale): CardReview(
            stale.id, card_fingerprint(stale),
            findings=(Finding("examples[0]", "an older problem", "error"),),
        ),
    }

    accepted = review_module.accept(
        store, shipping.id, "checked", {card_fingerprint(shipping)}
    )

    assert accepted[card_fingerprint(shipping)].accepted
    assert not accepted[card_fingerprint(stale)].accepted, "still unanswered"
    assert len(open_findings([stale], accepted)) == 1


def test_accepting_one_decks_version_does_not_clear_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two decks shipping one record differently are two cards — that is what
    `_shipping_records` promises — so clearing one is not an answer about the
    other."""
    root = project(tmp_path, [record()])
    (root / "decks" / "other.yaml").write_text(
        "deck:\n  name: Other\n  source: ../vocabulary.json\n"
        'notes:\n  - id: "word:話す:はなす"\n    usage_notes: "Other deck."\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        review_module.claude_client,
        "parse_call",
        reader(
            Verdict(Item("meanings", "problem A", "error")),
            Verdict(Item("meanings", "problem B", "error")),
        ),
    )
    cli.main(["--root", str(root), "review"])

    cli.main([
        "--root", str(root), "review", "--accept", "word:話す:はなす",
        "--because", "checked",
    ])

    accepted = [v["accepted"] for v in store_of(root).values()]
    assert sorted(accepted) == [True, True], (
        "both versions ship, so both are answered"
    )


def test_the_ready_to_ship_count_counts_cards_not_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`len(fresh) - len({ids with errors})` mixed two units: two versions of
    one record, both flagged, subtracted one from a total of two."""
    root = project(tmp_path, [record()])
    (root / "decks" / "other.yaml").write_text(
        "deck:\n  name: Other\n  source: ../vocabulary.json\n"
        'notes:\n  - id: "word:話す:はなす"\n    usage_notes: "Other deck."\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        review_module.claude_client,
        "parse_call",
        reader(
            Verdict(Item("meanings", "problem A", "error")),
            Verdict(Item("meanings", "problem B", "error")),
        ),
    )

    cli.main(["--root", str(root), "review"])

    assert "2 card(s): 0 ready to ship, 2 error(s)" in capsys.readouterr().out


def test_a_store_whose_fingerprints_are_gone_is_refused_by_name(tmp_path: Path) -> None:
    """Two old shapes existed. The first kept `content_fp` in the value and is
    rekeyed; the second was written by the reader that had already misread the
    first, so its values carry `record_id: ""` and no fingerprint at all.
    Nothing on disk says which card version that describes, and loading it
    anyway meant every card read as unreviewed, the build refused the deck, and
    `save_store` wrote the dead entries back forever with no diagnostic."""
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps({
            "word:なる:なる": {
                "record_id": "",
                "accepted": True,
                "accepted_because": "checked against jpdb",
                "findings": [{"where": "Pitch accent", "problem": "wrong",
                              "severity": "error", "suggestion": ""}],
            }
        }),
        encoding="utf-8",
    )

    with pytest.raises(ReviewError, match="cannot be recovered"):
        load_store(path)


def test_accepting_a_card_that_has_changed_says_so(tmp_path: Path) -> None:
    """Not "No review on record", which reads as "you mistyped the id" and sends
    the user hunting for the right one. The state is "the card changed since it
    was read", and the remedy is `janki review`, not a different id."""
    card = record()
    stale = record(usage_notes="the version that was read")
    store = {card_fingerprint(stale): CardReview(stale.id, card_fingerprint(stale))}

    with pytest.raises(ReviewError, match="has changed since it was last read"):
        review_module.accept(store, card.id, "checked", {card_fingerprint(card)})


def test_accepting_an_id_nothing_has_read_still_says_that(tmp_path: Path) -> None:
    """The sibling case has to keep its own message, or the new one swallows it."""
    card = record()

    with pytest.raises(ReviewError, match="No review on record"):
        review_module.accept({}, card.id, "checked", {card_fingerprint(card)})


def test_the_reader_is_shown_the_meanings_the_card_shows() -> None:
    """Not the stored list. Sending all nineteen senses of する — including
    JMdict's own metalanguage, "applies to nouns noted in this dictionary" —
    had the reader object that the card is unreadable, when the card shows four.
    Every one of those findings was about text that never ships, which is the
    one thing a gate must not spend a person's attention on."""
    from japanese_anki.review import card_prompt

    card = record(meanings=[f"sense {n}" for n in range(1, 18)])

    text = card_prompt(card, max_meanings=4)

    assert "sense 4" in text and "sense 5" not in text
    assert "+13 more senses, not shown on the card" in text


def test_no_cap_shows_everything() -> None:
    """`max_meanings = 0` means the card shows them all, so the reader must."""
    from japanese_anki.review import card_prompt

    card = record(meanings=[f"sense {n}" for n in range(1, 18)])

    assert "sense 17" in card_prompt(card, max_meanings=0)


def test_pitch_recheck_binds_source_authority_and_mechanical_facts() -> None:
    """A semantic reader cannot safely replace dictionary accent from memory."""
    from japanese_anki.pitch import bind_source
    from japanese_anki.review import pitch_recheck_prompt

    card = bind_source(record(reading="ねる", pitch_accent=["LHH"]))
    text = pitch_recheck_prompt(
        card,
        [Finding("pitch_accent", "ねる should use HLL", "error")],
    )

    assert "raw LHH; morae ね=L, る=H; following particle=H" in text
    assert "content-bound jpdb source marker" in text
    assert "do not replace this valid lexical pattern from model memory" in text


def test_pitch_recheck_does_not_claim_authority_for_an_unbound_value() -> None:
    from japanese_anki.review import pitch_recheck_prompt

    card = record(reading="ねる", pitch_accent=["LHH"])
    text = pitch_recheck_prompt(
        card,
        [Finding("pitch_accent", "ねる should use HLL", "error")],
    )

    assert "no valid source binding" in text
    assert "Do not assume that it came from jpdb" in text


def test_a_force_re_read_of_the_same_card_keeps_its_acceptance() -> None:
    """`--force` produces the same version, so dropping the acceptance made
    someone re-type a reason they had already given about text that had not
    changed."""
    card = record()
    finding = Finding("pitch_accent", "heiban", "error")
    before = review_module.accept(
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card), findings=(finding,)
        )},
        card.id,
        "jpdb is this deck's authority",
    )
    again = {card_fingerprint(card): CardReview(
        card.id, card_fingerprint(card), findings=(finding,)
    )}

    carried = review_module.carry_acceptances(before, again)

    assert carried[card_fingerprint(card)].accepted
    assert carried[card_fingerprint(card)].accepted_because == (
        "jpdb is this deck's authority"
    )


def test_a_re_read_that_finds_something_else_must_be_answered_again() -> None:
    """The acceptance is of the findings that were there. Carrying it onto a
    different finding clears it with a sentence written about another one."""
    card = record()
    before = review_module.accept(
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("pitch_accent", "heiban", "error"),),
        )},
        card.id,
        "checked",
    )
    again = {card_fingerprint(card): CardReview(
        card.id, card_fingerprint(card),
        findings=(Finding("meanings", "glossed as intransitive", "error"),),
    )}

    carried = review_module.carry_acceptances(before, again)

    assert not carried[card_fingerprint(card)].accepted


def test_the_reason_a_person_wrote_is_never_destroyed() -> None:
    """It is the only human-written sentence in the store and nothing can
    reconstruct it. It survived a re-read that found *nothing* — the fresh entry
    simply replaced the accepted one — and the sentence README quotes as its
    worked example was lost that way."""
    card = record()
    before = review_module.accept(
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("pitch_accent", "heiban", "error"),),
        )},
        card.id,
        "jpdb is this deck's authority",
    )
    found_nothing = {card_fingerprint(card): CardReview(
        card.id, card_fingerprint(card), findings=()
    )}

    carried = review_module.carry_acceptances(before, found_nothing)

    assert carried[card_fingerprint(card)].accepted_because == (
        "jpdb is this deck's authority"
    )


def test_an_acceptance_survives_the_model_rewording_its_finding() -> None:
    """Comparing the model's prose was the wrong key: a re-read of an unchanged
    card rewrites its sentences freely, so the carry branch almost never held
    and the destructive one is what ran. `where` and `severity` are stable."""
    card = record()
    before = review_module.accept(
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("pitch_accent", "なる is heiban (accent 0)", "error"),),
        )},
        card.id,
        "checked against jpdb",
    )
    reworded = {card_fingerprint(card): CardReview(
        card.id, card_fingerprint(card),
        findings=(Finding("Pitch accent", "the pattern marks it atamadaka", "error"),),
    )}

    carried = review_module.carry_acceptances(before, reworded)

    assert carried[card_fingerprint(card)].accepted, "same place, same severity"


def test_a_new_problem_still_has_to_be_answered() -> None:
    """And the old reason is kept beside it, so whoever answers can see what was
    decided last time rather than a blank."""
    card = record()
    before = review_module.accept(
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("pitch_accent", "heiban", "error"),),
        )},
        card.id,
        "checked against jpdb",
    )
    something_else = {card_fingerprint(card): CardReview(
        card.id, card_fingerprint(card),
        findings=(Finding("meanings", "glossed as intransitive", "error"),),
    )}

    carried = review_module.carry_acceptances(before, something_else)

    assert not carried[card_fingerprint(card)].accepted
    assert carried[card_fingerprint(card)].accepted_because == "checked against jpdb"


def test_a_clean_re_read_does_not_forget_what_was_accepted() -> None:
    """The hole in the first attempt. Carrying `accepted` with the *fresh*
    findings meant a re-read that found nothing wrote an accepted entry with an
    empty list — so the next run, when the model resurfaced the same error as it
    does, had nothing to check against and demanded the answer again."""
    card = record()
    accepted = review_module.accept(
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("pitch_accent", "heiban", "error"),),
        )},
        card.id,
        "jpdb is this deck's authority",
    )

    clean = review_module.carry_acceptances(
        accepted,
        {card_fingerprint(card): CardReview(card.id, card_fingerprint(card))},
    )
    resurfaced = review_module.carry_acceptances(
        clean,
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("Pitch accent", "worded differently", "error"),),
        )},
    )

    entry = resurfaced[card_fingerprint(card)]
    assert entry.accepted, "the same question, already answered"
    assert entry.blocking() == (), "so it does not stop a build"


def test_an_unanswered_error_still_blocks_on_an_accepted_card() -> None:
    """Per finding, not per entry: a re-read can surface an error the
    acceptance never covered, and an all-or-nothing flag would wave it through
    on the strength of an answer to a different question."""
    card = record()
    accepted = review_module.accept(
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("pitch_accent", "heiban", "error"),),
        )},
        card.id,
        "checked",
    )
    entry = accepted[card_fingerprint(card)]
    both = replace(
        entry,
        findings=(*entry.findings, Finding("meanings", "wrong gloss", "error")),
    )

    assert [f.where for f in both.blocking()] == ["meanings"]


def test_an_acceptance_written_before_marks_existed_keeps_working(tmp_path: Path) -> None:
    """Every store any earlier janki wrote carries `accepted: true` with no
    `accepted_marks`. Reading that as "answers nothing" would stop a human's
    acceptance working the moment the code was upgraded — the build refusing a
    card they had already cleared, with nothing on stderr saying why."""
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps({"abc123def456": {
            "record_id": "word:なる:なる",
            "accepted": True,
            "accepted_because": "jpdb is this deck's authority",
            "findings": [{"where": "Pitch accent", "problem": "heiban",
                          "severity": "error", "suggestion": ""}],
        }}),
        encoding="utf-8",
    )

    entry = load_store(path)["abc123def456"]

    assert entry.accepted_marks == ("pitch accent|error",)
    assert entry.blocking() == (), "so it still clears the build"


def test_a_lapsed_acceptance_still_remembers_what_was_decided() -> None:
    """An entry that lapsed once — a re-read raised something new, nobody has
    answered it yet — must not lose the record of what *was* decided, or the run
    after that starts from nothing."""
    card = record()
    accepted = review_module.accept(
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("pitch_accent", "heiban", "error"),),
        )},
        card.id,
        "jpdb is this deck's authority",
    )
    lapsed = review_module.carry_acceptances(accepted, {
        card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("meanings", "wrong gloss", "error"),),
        )
    })
    assert not lapsed[card_fingerprint(card)].accepted

    back_again = review_module.carry_acceptances(lapsed, {
        card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("pitch_accent", "heiban", "error"),),
        )
    })

    entry = back_again[card_fingerprint(card)]
    assert entry.accepted, "the pitch finding was answered two runs ago"
    assert entry.accepted_because == "jpdb is this deck's authority"


def test_a_clean_re_read_leaves_the_entry_accepted() -> None:
    """The value that path produces was pinned by nothing, and it is the state
    the whole `accepted_marks` design exists to make safe."""
    card = record()
    accepted = review_module.accept(
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("pitch_accent", "heiban", "error"),),
        )},
        card.id,
        "checked",
    )

    clean = review_module.carry_acceptances(
        accepted, {card_fingerprint(card): CardReview(card.id, card_fingerprint(card))}
    )

    entry = clean[card_fingerprint(card)]
    assert entry.accepted and entry.findings == ()
    assert entry.accepted_marks == ("pitch accent|error",), "and remembers why"


def test_a_card_nobody_accepted_does_not_become_accepted_by_being_clean() -> None:
    """`accepted` means a person overruled something. A card with no findings
    has nothing to overrule, and marking it accepted would put a human's word on
    a decision nobody made — and then carry it forward as one."""
    card = record()
    never = {card_fingerprint(card): CardReview(card.id, card_fingerprint(card))}

    carried = review_module.carry_acceptances(
        never, {card_fingerprint(card): CardReview(card.id, card_fingerprint(card))}
    )

    assert not carried[card_fingerprint(card)].accepted


def test_an_acceptance_with_no_derivable_marks_is_not_withdrawn(tmp_path: Path) -> None:
    """A store written before marks existed has none to migrate from when its
    findings are empty — the exact state the committed file was in. Withdrawing
    it on the next clean re-read erased a person's decision and left their
    reason orphaned beside it, with nothing printed."""
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps({"abc123def456": {
            "record_id": "word:なる:なる",
            "accepted": True,
            "accepted_because": "jpdb is this deck's authority",
            "findings": [],
        }}),
        encoding="utf-8",
    )
    store = load_store(path)
    assert store["abc123def456"].accepted_marks == (), "nothing to migrate from"

    carried = review_module.carry_acceptances(store, {
        "abc123def456": CardReview("word:なる:なる", "abc123def456")
    })

    assert carried["abc123def456"].accepted
    assert carried["abc123def456"].accepted_because == "jpdb is this deck's authority"


def test_a_lapsed_entry_does_not_re_ask_about_what_was_answered() -> None:
    """The marks outlive a lapse, so an entry that went un-accepted because a
    re-read raised something *new* still knows the old finding was answered.
    Re-listing it under "overrule it: janki review --accept" asks again for a
    decision the same entry proves was made."""
    card = record()
    accepted = review_module.accept(
        {card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(Finding("pitch_accent", "heiban", "error"),),
        )},
        card.id,
        "checked",
    )
    lapsed = review_module.carry_acceptances(accepted, {
        card_fingerprint(card): CardReview(
            card.id, card_fingerprint(card),
            findings=(
                Finding("pitch_accent", "heiban", "error"),
                Finding("meanings", "wrong gloss", "error"),
            ),
        )
    })

    entry = lapsed[card_fingerprint(card)]
    assert not entry.accepted, "the new finding is unanswered"
    assert [f.where for f in entry.blocking()] == ["meanings"], "and only that one"


def test_a_scalar_accepted_marks_is_refused(tmp_path: Path) -> None:
    """Iterated, a string became seventeen single-character marks — which
    matched nothing, suppressed the migration by being truthy, and was written
    back to the store. The sibling `findings` field is refused for exactly this."""
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps({"abc123def456": {
            "record_id": "word:なる:なる", "accepted": True,
            "accepted_marks": "pitch accent|error", "findings": [],
        }}),
        encoding="utf-8",
    )

    with pytest.raises(ReviewError, match="accepted_marks must be a list") as raised:
        load_store(path)

    # Named, like every other refusal on this path. `load_store` refuses rather
    # than skips, so one bad entry takes the gate down for every card — and the
    # message is the only thing pointing at which entry to open.
    assert "abc123def456" in str(raised.value)


@pytest.mark.parametrize("mark", [1, True, ["meanings|error"], {"a": 1}, None])
def test_a_mark_that_is_not_a_string_is_refused(tmp_path: Path, mark: object) -> None:
    """The container was checked and its elements were not, which is the same
    bug one level down — and it fails two different ways.

    A truthy *scalar* (`1`, `true`) survives the filter, so `stored` is
    non-empty and suppresses the migration this entry needs; it matches no
    `finding_mark`, so the card's error blocks the build; and
    `carry_acceptances` then writes `accepted: false` back, after which the
    acceptance cannot be recovered — the migration only runs while that flag is
    true.

    An *unhashable* one (a nested list, an object) never gets that far:
    `set(accepted_marks)` raises `TypeError`, which `cli.main` does not catch,
    so the gate exits on a traceback instead of a named error.

    `None` is listed last and is the easy case: it is falsy, so the filter alone
    already drops it."""
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps({"abc123def456": {
            "record_id": "word:なる:なる", "accepted": True,
            "accepted_marks": [mark],
            "findings": [
                {"where": "meanings", "problem": "wrong gloss", "severity": "error"}
            ],
        }}),
        encoding="utf-8",
    )

    with pytest.raises(ReviewError, match="each accepted mark must be a string") as raised:
        load_store(path)

    # Which entry, for the same reason as the sibling above: `load_store`
    # refuses rather than skips, so this prefix is the only thing pointing at
    # which of a few hundred entries to open.
    assert "abc123def456" in str(raised.value)


def test_a_card_with_no_sentence_is_not_paid_to_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a reader is paid for is the Japanese; meanings and headwords are
    checked by rules already. A card with no example has nothing to read."""
    calls: list[str] = []
    root = project(tmp_path, [record(examples=[], usage_notes="")])
    monkeypatch.setattr(
        review_module.claude_client, "parse_call", reader(Verdict(), calls=calls)
    )

    assert cli.main(["--root", str(root), "review"]) == 1
    assert calls == []


def test_one_unreadable_card_does_not_hold_up_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal exists to stop a wasted read, not to stop the run. Refusing
    everything would mean naming ninety-six ids to get past one."""
    calls: list[str] = []
    root = project(
        tmp_path,
        [
            record(id="word:話す:はなす", examples=[], usage_notes=""),
            record(id="word:読む:よむ", expression="読む", reading="よむ"),
        ],
    )
    monkeypatch.setattr(
        review_module.claude_client, "parse_call", reader(Verdict(), calls=calls)
    )

    assert cli.main(["--root", str(root), "review"]) == 0
    assert len(calls) == 1
    assert "読む" in calls[0]
    assert "held back: word:話す:はなす" in capsys.readouterr().err


def test_naming_an_unfinished_card_reads_it_anyway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal exists to stop accidental spend, not deliberate spend: an
    explicit id is a decision, as it is for enrich --ai."""
    calls: list[str] = []
    root = project(tmp_path, [record(examples=[], usage_notes="")])
    monkeypatch.setattr(
        review_module.claude_client, "parse_call", reader(Verdict(), calls=calls)
    )

    assert cli.main(["--root", str(root), "review", "word:話す:はなす"]) == 0
    assert len(calls) == 1


def test_a_card_with_a_local_error_is_not_paid_to_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local failures are fixable for free; a paid read cannot answer for one."""
    calls: list[str] = []
    root = project(tmp_path, [record(meanings=[])])
    monkeypatch.setattr(
        review_module.claude_client, "parse_call", reader(Verdict(), calls=calls)
    )

    assert cli.main(["--root", str(root), "review"]) == 1
    assert calls == []


def test_a_learner_load_hold_does_not_block_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hold is a warning, and a held card is still a finished card whose
    language a reader can judge. Refusing it would strand it between a build
    that calls it unreviewed and a review that calls it unready."""
    calls: list[str] = []
    # part_of_speech set so the learner-load hold is this record's *only*
    # warning: without it every fixture also carries verb-group-without-part-of-
    # speech, and a mutation that refused on warnings would fail this test for
    # that instead — passing the mutation check while proving nothing about
    # holds.
    held = add_example_flags(
        record(part_of_speech="v5s"), LEARNER_LOAD_HOLD_KEY, ["毎日話します。"]
    )
    root = project(tmp_path, [held])
    monkeypatch.setattr(
        review_module.claude_client, "parse_call", reader(Verdict(), calls=calls)
    )

    assert cli.main(["--root", str(root), "review"]) == 0
    assert len(calls) == 1
