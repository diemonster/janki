"""``janki patterns`` — the command around the reader.

`tests/test_patterns.py` covers what one document turns into. This covers the
loop: which documents get written to the store, what a partially failed run
reports and exits with, and the one piece of human-entered state in the file —
`reviewed` — which nothing may reset without saying so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import cli
from japanese_anki import patterns as patterns_module
from japanese_anki.claude_client import CallResult
from japanese_anki.errors import JankiError

#: A small collection, so a chart's verbs have a class on record. Without one
#: every row is held back — a conjugation chart teaches the outliers, so janki
#: will not guess a class to check one against.
COLLECTION = [
    {"id": "word:買う:かう", "expression": "買う", "reading": "かう",
     "meanings": ["to buy"], "verb_group": "godan"},
    {"id": "word:飲む:のむ", "expression": "飲む", "reading": "のむ",
     "meanings": ["to drink"], "verb_group": "godan"},
    {"id": "word:来る:くる", "expression": "来る", "reading": "くる",
     "meanings": ["to come"], "verb_group": "kuru"},
    {"id": "word:する:する", "expression": "する", "reading": "する",
     "meanings": ["to do"], "verb_group": "suru"},
]


def project(tmp_path: Path, records: list[dict] | None = None) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'patterns_file = "patterns.json"\n'
        'scan_inbox = "inbox"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(COLLECTION if records is None else records, ensure_ascii=False),
        encoding="utf-8",
    )
    return tmp_path


def document(root: Path, name: str) -> Path:
    """A file `prepare_inputs` will accept."""
    path = root / name
    path.write_bytes(b"%PDF-1.4\n%fake\n")
    return path


class Parsed:
    def __init__(self, kind: str, title: str, templates: list[str]) -> None:
        self.kind, self.title = kind, title
        self.patterns = [Item(t) for t in templates]


class Item:
    def __init__(self, template: str) -> None:
        self.template, self.gloss, self.examples, self.where = template, "", [], ""


def reader(answers: dict[str, Any]):
    """Answers per document name; a value that is an exception is raised."""

    def parse_call(_model, _blocks, content, *_args, **_kwargs) -> CallResult:
        text = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
        for name, answer in answers.items():
            if name in text:
                if isinstance(answer, Exception):
                    raise answer
                return CallResult(parsed=answer, stop_reason="end_turn", refusal=None)
        raise AssertionError(f"no answer prepared for {text!r}")

    return parse_call


@pytest.fixture(autouse=True)
def no_style_guide(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.claude_client, "read_style_guide", lambda _root: "")


def store_of(root: Path) -> dict:
    return json.loads((root / "patterns.json").read_text(encoding="utf-8"))


# --- reading documents -------------------------------------------------------


def test_a_document_is_read_into_the_store_unreviewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({"week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"])}),
    )

    assert cli.main(["--root", str(root), "patterns", str(document(root, "week11.pdf"))]) == 0

    entry = store_of(root)["week11.pdf"]
    assert entry["kind"] == "lesson" and entry["reviewed"] is False
    assert "--review 'week11.pdf'" in capsys.readouterr().out


def test_a_partial_failure_is_a_non_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`janki patterns *.pdf && janki patterns --review …` must not run on from
    a document that was never read."""
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({
            "week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"]),
            "week12.pdf": JankiError("the model declined"),
        }),
    )

    code = cli.main([
        "--root", str(root), "patterns",
        str(document(root, "week11.pdf")), str(document(root, "week12.pdf")),
    ])

    assert code == 1
    assert list(store_of(root)) == ["week11.pdf"], "and the one that worked was kept"


def test_a_partial_failure_still_names_the_next_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hint used to be gated on the whole run succeeding, so a run that read
    two of three documents said nothing about the two."""
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({
            "week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"]),
            "week12.pdf": JankiError("the model declined"),
        }),
    )

    cli.main([
        "--root", str(root), "patterns",
        str(document(root, "week11.pdf")), str(document(root, "week12.pdf")),
    ])

    out = capsys.readouterr()
    assert "--review 'week11.pdf'" in out.out
    assert "the model declined" in out.err


# --- the reviewed flag -------------------------------------------------------


def test_a_reviewed_document_is_skipped_rather_than_re_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`reviewed` is the only human-entered state in this file, and
    `extract_patterns` always returns False. Replacing the entry outright
    un-reviewed the document on any re-read — including the ordinary
    `janki patterns inbox/*.pdf` after adding one new handout — and the only
    signal was a line missing from the next enrich run.

    Skipping is not failing, so the run still exits 0: a glob over an inbox
    where one file has been reviewed must not stop
    `janki patterns *.pdf && janki patterns --review new.pdf` from reaching its
    second half."""
    root = project(tmp_path)
    calls: list[str] = []
    inner = reader({"week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"])})

    def counting(*args: Any, **kwargs: Any):
        calls.append("read")
        return inner(*args, **kwargs)

    monkeypatch.setattr(patterns_module.claude_client, "parse_call", counting)
    path = document(root, "week11.pdf")
    cli.main(["--root", str(root), "patterns", str(path)])
    cli.main(["--root", str(root), "patterns", "--review", "week11.pdf"])
    assert store_of(root)["week11.pdf"]["reviewed"] is True
    assert len(calls) == 1

    assert cli.main(["--root", str(root), "patterns", str(path)]) == 0

    assert store_of(root)["week11.pdf"]["reviewed"] is True, "still reviewed"
    # Before the read, not after it: checking afterwards billed a full document
    # read against every already-reviewed file in the inbox and discarded it.
    assert len(calls) == 1, "and no second document read was paid for"
    assert "already read and marked reviewed" in capsys.readouterr().out


def test_a_skip_does_not_mask_a_real_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two categories are kept apart, so a genuine extraction failure in the
    same run still exits non-zero — and each lands in its own stream. Asserting
    only the exit code pinned nothing: the 1 comes entirely from the failure, so
    the test passed both before the split and against a version counting skips
    as failures."""
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({
            "week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"]),
            "week12.pdf": JankiError("the model declined"),
        }),
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "week11.pdf"))])
    cli.main(["--root", str(root), "patterns", "--review", "week11.pdf"])

    code = cli.main([
        "--root", str(root), "patterns",
        str(root / "week11.pdf"), str(document(root, "week12.pdf")),
    ])

    assert code == 1
    out = capsys.readouterr()
    assert "already read and marked reviewed" in out.out, "the skip is a notice"
    assert "the model declined" in out.err, "the failure is a warning"
    assert store_of(root)["week11.pdf"]["reviewed"] is True


def test_force_re_reads_it_and_says_it_is_unreviewed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({"week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"])}),
    )
    path = document(root, "week11.pdf")
    cli.main(["--root", str(root), "patterns", str(path)])
    cli.main(["--root", str(root), "patterns", "--review", "week11.pdf"])

    assert cli.main(["--root", str(root), "patterns", "--force", str(path)]) == 0

    assert store_of(root)["week11.pdf"]["reviewed"] is False


def test_review_and_files_in_one_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--review` appends and `files` is nargs="*", so
    `janki patterns --review a.pdf b.pdf` binds b.pdf to `files` — a natural
    thing to type. The review branch returned before reading it, and the run
    reported only "Marked 1 document(s) reviewed"."""
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({"a.pdf": Parsed("lesson", "A", ["〜んだ"])}),
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "a.pdf"))])

    code = cli.main([
        "--root", str(root), "patterns",
        "--review", "a.pdf", str(document(root, "b.pdf")),
    ])

    assert code == 1
    assert "separate runs" in capsys.readouterr().err
    assert store_of(root)["a.pdf"]["reviewed"] is False, "and nothing was half-applied"


def test_reviewing_an_unread_document_names_what_is_known(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)

    assert cli.main(["--root", str(root), "patterns", "--review", "nope.pdf"]) == 1

    assert "Known: none" in capsys.readouterr().err


def test_listing_with_no_arguments_reports_what_has_been_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({"week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"])}),
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "week11.pdf"))])
    capsys.readouterr()

    assert cli.main(["--root", str(root), "patterns"]) == 0

    out = capsys.readouterr().out
    assert "week11.pdf — lesson, 1 pattern(s) [UNREVIEWED]" in out


def test_reviewing_a_chart_says_it_steers_no_sentences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only lesson documents steer example sentences, so "their patterns are now
    in use" was false for a te-form chart — eight human-reviewed patterns
    dropping out of the pipeline while four separate messages said they were
    working."""
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({"teform.pdf": Parsed("pattern", "Te-form Song", ["く → いて"])}),
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "teform.pdf"))])
    # Asserted before clearing: the read-time parenthetical is its own branch,
    # and inverting its condition left the whole suite green while the CLI told
    # you a lesson steers nothing and said nothing about a chart.
    assert "teform.pdf is a pattern document" in capsys.readouterr().out

    cli.main(["--root", str(root), "patterns", "--review", "teform.pdf"])

    err = capsys.readouterr().err
    assert "is a pattern document" in err
    assert "nothing uses its patterns yet" in err


def test_the_listing_marks_a_reviewed_chart_as_unused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({"teform.pdf": Parsed("pattern", "Te-form Song", ["く → いて"])}),
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "teform.pdf"))])
    cli.main(["--root", str(root), "patterns", "--review", "teform.pdf"])
    capsys.readouterr()

    cli.main(["--root", str(root), "patterns"])

    assert "[reviewed, not used for sentences]" in capsys.readouterr().out


def test_a_reviewed_lesson_is_marked_plainly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other side of it: the honest label must not appear on the documents
    that do steer, or it says nothing."""
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({"week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"])}),
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "week11.pdf"))])
    cli.main(["--root", str(root), "patterns", "--review", "week11.pdf"])
    capsys.readouterr()

    cli.main(["--root", str(root), "patterns"])

    out = capsys.readouterr().out
    assert "[reviewed]" in out
    assert "not used for sentences" not in out


def test_reading_a_chart_checks_its_rules_against_janki(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Automatic rather than a flag: a check nobody runs catches nothing."""
    root = project(tmp_path)
    parsed = Parsed("pattern", "Te-form Song", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かう ⇨ かって (to buy)"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"teform.pdf": parsed})
    )

    cli.main(["--root", str(root), "patterns", str(document(root, "teform.pdf"))])

    assert "checked 1/1 worked example(s)" in capsys.readouterr().out


def test_a_garbled_rule_is_named_when_the_chart_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    parsed = Parsed("pattern", "Te-form Song", [])
    parsed.patterns = [Item("う・つ・る → いて")]
    parsed.patterns[0].examples = ["かう ⇨ かいて"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"teform.pdf": parsed})
    )

    cli.main(["--root", str(root), "patterns", str(document(root, "teform.pdf"))])

    out = capsys.readouterr().out
    assert "checked 0/1" in out
    assert "かう ⇨ かいて is not what janki computes" in out
    assert "godan: かって" in out


def test_check_re_runs_over_the_store_without_reading_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    parsed = Parsed("pattern", "Te-form Song", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かう ⇨ かって"]
    calls: list[str] = []
    inner = reader({"teform.pdf": parsed})

    def counting(*args: Any, **kwargs: Any):
        calls.append("read")
        return inner(*args, **kwargs)

    monkeypatch.setattr(patterns_module.claude_client, "parse_call", counting)
    cli.main(["--root", str(root), "patterns", str(document(root, "teform.pdf"))])
    capsys.readouterr()

    assert cli.main(["--root", str(root), "patterns", "--check"]) == 0

    assert len(calls) == 1, "the store already holds what --check reads"
    assert "agree with janki" in capsys.readouterr().out


def test_check_exits_non_zero_on_a_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path)
    parsed = Parsed("pattern", "Te-form Song", [])
    parsed.patterns = [Item("う・つ・る → いて")]
    parsed.patterns[0].examples = ["かう ⇨ かいて"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"teform.pdf": parsed})
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "teform.pdf"))])

    assert cli.main(["--root", str(root), "patterns", "--check"]) == 1


def test_reading_a_lesson_says_nothing_about_steering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The negative half of the read-time parenthetical, so the condition is
    pinned in both directions rather than only where it fires."""
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({"week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"])}),
    )

    cli.main(["--root", str(root), "patterns", str(document(root, "week11.pdf"))])

    assert "steered only by" not in capsys.readouterr().out


def test_check_says_so_when_nothing_could_be_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The all-clear used to print whenever nothing disagreed, including when
    nothing was examined — false reassurance from the one command whose entire
    job is reassurance."""
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({"week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"])}),
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "week11.pdf"))])
    capsys.readouterr()

    assert cli.main(["--root", str(root), "patterns", "--check"]) == 0

    out = capsys.readouterr().out
    assert "Nothing in the store could be checked" in out
    assert "week11.pdf (lesson)" in out
    assert "agree with janki" not in out


def test_check_will_not_silently_ignore_files_passed_with_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--check` returns before the read loop, so a file passed alongside it was
    never read, never stored and never mentioned — on a zero exit. The same trap
    the --review guard exists for."""
    root = project(tmp_path)

    code = cli.main([
        "--root", str(root), "patterns", "--check", str(document(root, "teform.pdf"))
    ])

    assert code == 1
    assert "cannot be combined with" in capsys.readouterr().err


def test_check_names_the_documents_it_could_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One readable chart beside a document this cannot read printed an
    unqualified all-clear and never mentioned the other — the same false
    reassurance at document granularity."""
    root = project(tmp_path)
    chart = Parsed("pattern", "Te-form", [])
    chart.patterns = [Item("う・つ・る → って")]
    chart.patterns[0].examples = ["かう ⇨ かって"]
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({"teform.pdf": chart, "week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"])}),
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "teform.pdf"))])
    cli.main(["--root", str(root), "patterns", str(document(root, "week11.pdf"))])
    capsys.readouterr()

    assert cli.main(["--root", str(root), "patterns", "--check"]) == 0

    out = capsys.readouterr().out
    assert "Not checked" in out and "week11.pdf (lesson)" in out
    assert "All 1 worked example(s) agree" in out


def test_the_form_each_row_matched_is_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`のむ ⇨ のんだ` on a て-form chart agrees, as a past. Without naming the
    form, the likeliest garble on such a chart reads as a clean pass."""
    root = project(tmp_path)
    chart = Parsed("pattern", "Te-form", [])
    chart.patterns = [Item("む・ぶ・ぬ → んで")]
    chart.patterns[0].examples = ["のむ ⇨ のんだ"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"teform.pdf": chart})
    )

    cli.main(["--root", str(root), "patterns", str(document(root, "teform.pdf"))])

    assert "のむ ⇨ のんだ matched past" in capsys.readouterr().out


def test_the_check_uses_the_collections_verb_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """And holds back the verb it has no class for, rather than trying every
    class to see if one fits — a chart teaches the outliers, so a guessed class
    would contradict it exactly where it matters."""
    root = project(tmp_path, [{
        "id": "word:食べる:たべる", "expression": "食べる", "reading": "たべる",
        "meanings": ["to eat"], "verb_group": "ichidan",
    }])
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("potential")]
    parsed.patterns[0].examples = ["食べる ⇨ 食べれる", "およぐ ⇨ およいで"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])

    out = capsys.readouterr().out
    assert "食べる ⇨ 食べれる is not what janki computes" in out
    assert "およぐ ⇨ およいで not checked — no verb class on record" in out


def test_a_failed_class_lookup_does_not_discard_the_document_just_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The lookup used to sit inside the read loop, outside its try, and before
    the save — so an unset key, a timeout or a 429 threw away every document
    already read and paid for in the same run, and the model calls had to be
    made again."""
    root = project(tmp_path)
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("て")]
    parsed.patterns[0].examples = ["およぐ ⇨ およいで"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    def refuse(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        raise JankiError("JPDB_API_KEY is not set")

    monkeypatch.setattr(cli.patterns, "verb_groups_from_jpdb", refuse)

    cli.main([
        "--root", str(root), "patterns", "--ask-jpdb",
        str(document(root, "chart.pdf")),
    ])

    assert "chart.pdf" in store_of(root), "the read survived the lookup failure"
    assert "could not look up verb classes" in capsys.readouterr().err


def test_a_reading_key_finds_the_class_a_chart_writes_in_kana(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reason `_verb_groups` keys on both spelling and reading: a chart
    writes かう while the record is 買う. Nothing pinned it — the module-level
    test hands `check_pattern_rules` a dict that already has both keys, so it
    never builds the map."""
    root = project(tmp_path, [{
        "id": "word:買う:かう", "expression": "買う", "reading": "かう",
        "meanings": ["to buy"], "verb_group": "godan",
    }])
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かう ⇨ かいて"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])

    out = capsys.readouterr().out
    assert "かう ⇨ かいて is not what janki computes" in out, "checked, not held back"
    assert "godan: かって" in out


def test_a_kana_two_records_disagree_about_is_not_knowledge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """かえる is 変える (ichidan) and 帰る (godan). Taking whichever loaded first
    turned the correct row `かえる ⇨ かえって` into a confident failure against
    the wrong class."""
    root = project(tmp_path, [
        {"id": "word:変える:かえる", "expression": "変える", "reading": "かえる",
         "meanings": ["to change"], "verb_group": "ichidan"},
        {"id": "word:帰る:かえる", "expression": "帰る", "reading": "かえる",
         "meanings": ["to return"], "verb_group": "godan"},
    ])
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かえる ⇨ かえって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])

    out = capsys.readouterr().out
    assert "is not what janki computes" not in out, "no verdict on an ambiguous class"
    assert "no verb class on record" in out


def test_an_unreadable_collection_is_reported_rather_than_cancelling_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two things at once, and the second is what a warning alone lost.

    `janki patterns handout.pdf` does not otherwise touch the collection, so
    aborting before the PDF is read would suppress the output the command exists
    to produce — the document is still read, stored and reported.

    But the rule check *did not happen*, and exiting 0 says it did. The exit
    code is what `janki patterns *.pdf && janki patterns --review …` runs on.
    """
    root = project(tmp_path)
    (root / "vocabulary.json").write_text(
        '[{"id": "word:x:x", "expression": "x", "reading": "x", "examples": 3}]',
        encoding="utf-8",
    )
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    # A worked example, so the class map really would have been consulted. A
    # chart of bare endings names no verb, and reporting *that* run as failed
    # would be a claim about a check it never needed.
    parsed.patterns[0].examples = ["かう ⇨ かって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    code = cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])

    assert "chart.pdf" in store_of(root), "the document was still read"
    assert code == 1, "and the check that did not happen reached the exit code"
    assert "could not read the collection for verb classes" in capsys.readouterr().err


def test_a_document_with_no_verbs_does_not_fail_over_the_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`く → いて` names an ending, not a verb, so nothing in this run would have
    consulted the class map. Failing here breaks
    `janki patterns *.pdf && janki patterns --review …` over a file the run
    never needed — the same reason the jpdb failure is kept out of `failures`.
    Still said out loud, because the next document might need it."""
    root = project(tmp_path)
    (root / "vocabulary.json").write_text(
        '[{"id": "word:x:x", "expression": "x", "reading": "x", "examples": 3}]',
        encoding="utf-8",
    )
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("く → いて")]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    code = cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])

    assert code == 0
    assert "could not read the collection for verb classes" in capsys.readouterr().err


def test_a_missing_collection_is_refused_like_an_unreadable_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A renamed file or a typo'd `[paths]` entry. Returning an empty class map
    for it reached the same loss by the neighbouring branch: every row held back
    for want of a class, a pattern deck built with blank `Examples`, reported as
    built — and its GUIDs blank the examples of the rule cards already in
    Anki."""
    root = project(tmp_path)
    (root / "vocabulary.json").unlink()
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かう ⇨ かって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    code = cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])

    assert code == 1
    assert "vocabulary.json" in capsys.readouterr().err, "which file is missing"


def test_check_does_not_call_an_unreadable_collection_all_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--check` is a command whose exit code is its entire product. Swallowing
    the read error held every row back for want of a class, left `checked` at 0,
    printed "nothing could be checked" and returned 0 — a pass over a run that
    verified nothing, blaming the words instead of the file."""
    root = project(tmp_path)
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う → って")]
    parsed.patterns[0].examples = ["かう ⇨ かって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])
    (root / "vocabulary.json").write_text(
        '[{"id": "word:x:x", "expression": "x", "reading": "x", "examples": 3}]',
        encoding="utf-8",
    )
    capsys.readouterr()

    code = cli.main(["--root", str(root), "patterns", "--check"])

    assert code != 0
    out = capsys.readouterr()
    assert "could not read the collection" in out.err, "and says why"
    assert "Nothing in the store could be checked" not in out.out


def test_two_spellings_of_one_class_are_not_a_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`verb_group` is free text. A CSV import writes its column through
    verbatim, so `五段` sits beside another record's jpdb-written `godan`, and
    `enrich --jpdb` never repairs it because it only fills a field that is
    empty. Compared raw those two read as an ambiguous kana key, かう was
    dropped, and a correct row was held back saying no class was on record —
    when two were and both said godan. `conjugate` accepts either spelling."""
    root = project(tmp_path, [
        {"id": "word:買う:かう", "expression": "買う", "reading": "かう",
         "meanings": ["to buy"], "verb_group": "五段"},
        {"id": "word:飼う:かう", "expression": "飼う", "reading": "かう",
         "meanings": ["to keep an animal"], "verb_group": "godan"},
    ])
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かう ⇨ かって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])

    out = capsys.readouterr().out
    assert "no verb class on record" not in out, "both records say godan"
    assert "checked 1/1 worked example(s)" in out


def test_two_spellings_of_one_unrecognised_class_are_not_a_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`Group 3` and `group-3` are one class `conjugate` does not know, and the
    fold only covered names it does. Read as two, the word looked ambiguous and
    the reply blamed a missing class — sending someone to `enrich --jpdb`, which
    will not overwrite a non-empty `verb_group`. The right complaint names the
    class actually on the record."""
    root = project(tmp_path, [
        {"id": "word:買う:かう", "expression": "買う", "reading": "かう",
         "meanings": ["to buy"], "verb_group": "Group 3"},
        {"id": "word:飼う:かう", "expression": "飼う", "reading": "かう",
         "meanings": ["to keep an animal"], "verb_group": "group-3"},
    ])
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かう ⇨ かって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])

    out = capsys.readouterr().out
    assert "no verb class on record" not in out
    assert "does not recognise the verb class" in out
    assert "Group 3" in out or "group-3" in out, "quoted from the record"


def test_a_decomposed_record_still_matches_a_composed_chart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Records are stored as imported — `models.py` only strips — so a Shirabe
    export carrying decomposed dakuten writes およ + く + U+3099, while the
    chart's およぐ arrives composed. Keyed raw, the lookup missed and the row
    was held back as though the collection had never heard of the verb."""
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "泳ぐ")
    root = project(tmp_path, [{
        "id": "word:泳ぐ:およぐ", "expression": decomposed,
        "reading": unicodedata.normalize("NFD", "およぐ"),
        "meanings": ["to swim"], "verb_group": "godan",
    }])
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("ぐ → いで")]
    parsed.patterns[0].examples = ["およぐ ⇨ およいで"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])

    out = capsys.readouterr().out
    assert "checked 1/1 worked example(s)" in out, "the class was found"
    assert "no verb class on record" not in out


def test_one_jpdb_request_covers_every_document_in_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What `--ask-jpdb`'s help text promises. The lookup used to sit inside the
    read loop, so N files meant N requests, and moving it out was pinned by
    nothing."""
    root = project(tmp_path)
    a = Parsed("pattern", "A", [])
    a.patterns = [Item("う → って")]
    a.patterns[0].examples = ["まつ ⇨ まって"]
    b = Parsed("pattern", "B", [])
    b.patterns = [Item("ぐ → いで")]
    b.patterns[0].examples = ["およぐ ⇨ およいで"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"a.pdf": a, "b.pdf": b})
    )
    calls: list[list[str]] = []

    def one_lookup(verbs, _client):
        calls.append(list(verbs))
        return {"まつ": "godan", "およぐ": "godan"}

    monkeypatch.setattr(cli.patterns, "verb_groups_from_jpdb", one_lookup)
    monkeypatch.setattr(cli.jpdb, "api_key_from_env", lambda: "k")
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda _key: object())

    cli.main([
        "--root", str(root), "patterns", "--ask-jpdb",
        str(document(root, "a.pdf")), str(document(root, "b.pdf")),
    ])

    assert len(calls) == 1, "one request for the run, not one per file"
    assert sorted(calls[0]) == ["および", "まつ"] or sorted(calls[0]) == ["およぐ", "まつ"]


def test_jpdb_answering_for_every_verb_is_not_a_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--ask-jpdb` is documented as the way to check verbs the collection does
    not hold, and a collection that cannot be read is the limit case. Deciding
    the verdict before asking reported a run whose every verb jpdb resolved as
    unchecked — while stdout said `checked 1/1` — and exited 1 into the
    `janki patterns *.pdf && janki patterns --review …` chain."""
    root = project(tmp_path)
    (root / "vocabulary.json").unlink()
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かう ⇨ かって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )
    monkeypatch.setattr(
        cli.patterns, "verb_groups_from_jpdb", lambda *_a, **_k: {"かう": "godan"}
    )
    monkeypatch.setattr(cli.jpdb, "api_key_from_env", lambda: "k")
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda _key: object())

    code = cli.main([
        "--root", str(root), "patterns", "--ask-jpdb",
        str(document(root, "chart.pdf")),
    ])

    out = capsys.readouterr()
    assert "checked 1/1 worked example(s)" in out.out
    assert code == 0, "every verb was answered for"


def test_check_asks_jpdb_before_failing_over_the_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`janki patterns --check --ask-jpdb` — the command the README gives. It
    aborted before jpdb was asked, so it never discovered it needed nothing from
    the file it could not read."""
    root = project(tmp_path)
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かう ⇨ かって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])
    (root / "vocabulary.json").unlink()
    monkeypatch.setattr(
        cli.patterns, "verb_groups_from_jpdb", lambda *_a, **_k: {"かう": "godan"}
    )
    monkeypatch.setattr(cli.jpdb, "api_key_from_env", lambda: "k")
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda _key: object())
    capsys.readouterr()

    code = cli.main(["--root", str(root), "patterns", "--check", "--ask-jpdb"])

    assert code == 0
    assert "All 1 worked example(s) agree" in capsys.readouterr().out


def test_check_over_a_store_that_needs_no_class_is_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A store of bare-ending charts consults the class map for nothing, so an
    unreadable collection is not this run's problem — the same rule the read
    path follows."""
    root = project(tmp_path)
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("く → いて")]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])
    (root / "vocabulary.json").unlink()
    capsys.readouterr()

    code = cli.main(["--root", str(root), "patterns", "--check"])

    assert code == 0
    out = capsys.readouterr()
    assert "Nothing in the store could be checked" in out.out
    # Still said out loud, because the next document might need it — and this
    # is the only line that mentions the collection on a run that exits 0.
    assert "could not read the collection for verb classes" in out.err


def test_check_does_not_call_a_partly_stranded_run_all_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """jpdb places one of the two verbs, which is the ordinary outcome rather
    than a corner one — it answers for what it can place and omits the rest. The
    guard only covered a run where *nothing* was checked, so a mixed run printed
    an unqualified all-clear, and the row it held back blamed "no verb class on
    record" and pointed at `janki enrich --jpdb`, which reads the very file this
    run could not read."""
    root = project(tmp_path)
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かう ⇨ かって", "まつ ⇨ まって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])
    (root / "vocabulary.json").unlink()
    monkeypatch.setattr(
        cli.patterns, "verb_groups_from_jpdb", lambda *_a, **_k: {"かう": "godan"}
    )
    monkeypatch.setattr(cli.jpdb, "api_key_from_env", lambda: "k")
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda _key: object())
    capsys.readouterr()

    code = cli.main(["--root", str(root), "patterns", "--check", "--ask-jpdb"])

    out = capsys.readouterr()
    assert code == 1
    assert "All 1 worked example(s) agree" not in out.out, "not an all-clear"
    assert "まつ went unchecked" in out.out
    assert out.err.count("could not read the collection") == 1, "said once"
    assert "no class for まつ" in out.err, "and names what was stranded"


def test_check_names_the_stranded_verbs_even_when_a_row_disagrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The stdout line naming them is printed only when nothing disagreed, so on
    a run that is both stranded and wrong the stderr line is the only place the
    unresolved verbs are named."""
    root = project(tmp_path)
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かう ⇨ かいて", "まつ ⇨ まって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])
    (root / "vocabulary.json").unlink()
    monkeypatch.setattr(
        cli.patterns, "verb_groups_from_jpdb", lambda *_a, **_k: {"かう": "godan"}
    )
    monkeypatch.setattr(cli.jpdb, "api_key_from_env", lambda: "k")
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda _key: object())
    capsys.readouterr()

    code = cli.main(["--root", str(root), "patterns", "--check", "--ask-jpdb"])

    out = capsys.readouterr()
    assert code == 1
    assert "かう ⇨ かいて is not what janki computes" in out.out, "the row disagreed"
    assert "no class for まつ" in out.err


def test_check_says_the_collection_failed_even_when_jpdb_does_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two independent failures. Unguarded, an unset key or a 429 unwound past
    the line reporting the collection, so the user fixed the key, re-ran, and
    only then learned the collection was the real problem."""
    root = project(tmp_path)
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    parsed.patterns[0].examples = ["かう ⇨ かって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )
    cli.main(["--root", str(root), "patterns", str(document(root, "chart.pdf"))])
    (root / "vocabulary.json").unlink()

    def refuse():
        raise JankiError("JPDB_API_KEY is not set")

    monkeypatch.setattr(cli.jpdb, "api_key_from_env", refuse)
    capsys.readouterr()

    code = cli.main(["--root", str(root), "patterns", "--check", "--ask-jpdb"])

    err = capsys.readouterr().err
    assert code == 1
    assert "could not read the collection for verb classes" in err
    assert "JPDB_API_KEY is not set" in err, "and the other failure too"


def test_a_failed_lookup_keeps_the_offline_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`_classes_for` reads the collection first and only then asks jpdb, so
    discarding both on a 429 held back rows whose class *is* on record — and
    lost the garbled row the offline check would have caught."""
    root = project(tmp_path, [{
        "id": "word:買う:かう", "expression": "買う", "reading": "かう",
        "meanings": ["to buy"], "verb_group": "godan",
    }])
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う → って")]
    parsed.patterns[0].examples = ["かう ⇨ かいて", "およぐ ⇨ およいで"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    def refuse(*_args, **_kwargs):
        raise JankiError("jpdb said 429")

    monkeypatch.setattr(cli.patterns, "verb_groups_from_jpdb", refuse)
    monkeypatch.setattr(cli.jpdb, "api_key_from_env", lambda: "k")
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda _key: object())

    code = cli.main([
        "--root", str(root), "patterns", "--ask-jpdb",
        str(document(root, "chart.pdf")),
    ])

    out = capsys.readouterr()
    assert "かう ⇨ かいて is not what janki computes" in out.out, "offline class kept"
    assert "could not look up verb classes" in out.err
    assert code == 0, "nothing was lost, so the && chain still runs"


def test_each_documents_check_is_printed_under_its_own_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The check lines carry no filename, and two charts can share a template
    string — so one undifferentiated block at the end of the run says nothing
    about which chart holds the bad row."""
    root = project(tmp_path)
    a = Parsed("pattern", "A", [])
    a.patterns = [Item("く → いて")]
    b = Parsed("pattern", "B", [])
    b.patterns = [Item("ぐ → いで")]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"a.pdf": a, "b.pdf": b})
    )

    cli.main([
        "--root", str(root), "patterns",
        str(document(root, "a.pdf")), str(document(root, "b.pdf")),
    ])

    out = capsys.readouterr().out
    assert out.count("a.pdf — pattern") >= 1 and out.count("b.pdf — pattern") >= 1


def test_two_inputs_that_would_share_a_store_key_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The store is keyed by basename, so `scans/a/chart.pdf` and
    `scans/b/chart.pdf` claim one entry — the first read, paid for and printed
    as read, then overwritten when the store is saved. An input silently lost on
    a zero exit.

    Both already under `scan_inbox`, and with different bytes: that is the shape
    that reaches the guard as two paths. Two files *outside* the inbox are
    content-addressed on the way in, so different content already gets
    different inbox names and identical content is one document."""
    root = project(tmp_path)
    for folder, body in (("a", b"%PDF-1.4\n%one\n"), ("b", b"%PDF-1.4\n%two\n")):
        (root / "inbox" / folder).mkdir(parents=True)
        (root / "inbox" / folder / "chart.pdf").write_bytes(body)
    calls: list[str] = []
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        lambda *a, **k: calls.append("read") or (_ for _ in ()).throw(AssertionError),
    )

    code = cli.main([
        "--root", str(root), "patterns",
        str(root / "inbox" / "a" / "chart.pdf"), str(root / "inbox" / "b" / "chart.pdf"),
    ])

    assert code == 1
    assert calls == [], "refused before paying for a read"
    err = capsys.readouterr().err
    assert "would be stored under one name" in err
    # Not "read them in separate runs", which performs the loss it just refused:
    # run two replaces run one's entry with nothing said, or skips the second
    # document as "already read and marked reviewed" on exit 0.
    assert "separate runs" not in err


def test_one_file_named_twice_is_read_once_rather_than_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`janki patterns data/inbox/scans/chart.pdf ~/Downloads/chart.pdf` where
    the second is the copy the first came from — or one path caught by two
    overlapping globs. `prepare_inputs` content-addresses the inbox, so both
    resolve to one file: nothing can be lost, and refusing the batch threw away
    every other document in the run over it. Reading it twice would bill twice
    for one document."""
    root = project(tmp_path)
    inside = root / "inbox" / "chart.pdf"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_bytes(b"%PDF-1.4\n%fake\n")
    outside = root / "chart.pdf"
    outside.write_bytes(inside.read_bytes())
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("く → いて")]
    calls: list[str] = []
    real = reader({"chart.pdf": parsed})

    def counted(*args, **kwargs):
        calls.append("read")
        return real(*args, **kwargs)

    monkeypatch.setattr(patterns_module.claude_client, "parse_call", counted)

    code = cli.main(["--root", str(root), "patterns", str(inside), str(outside)])

    assert code == 0
    assert calls == ["read"], "one document, one read"
    assert "chart.pdf" in store_of(root)
    assert "named more than once" in capsys.readouterr().out, "and said so"


def test_two_spellings_of_one_inbox_path_are_the_same_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file already under the inbox is returned verbatim, so `scans/../scans/
    chart.pdf` is the same file under a second string. Keyed by spelling, both
    survived to the basename guard, which aborted the batch and offered "rename
    one" — i.e. edit a file under `data/inbox/`, which this project does not do
    to its own committed inputs."""
    root = project(tmp_path)
    scans = root / "inbox" / "scans"
    scans.mkdir(parents=True)
    (scans / "chart.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("く → いて")]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    code = cli.main([
        "--root", str(root), "patterns",
        str(scans / "chart.pdf"), str(scans / ".." / "scans" / "chart.pdf"),
    ])

    assert code == 0
    assert "would be stored under one name" not in capsys.readouterr().err


def test_an_unreadable_collection_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`_verb_groups` parses the whole collection, so recomputing it on the jpdb
    recovery path read the file twice and said the same thing twice. The jpdb
    failure is its own separate line — the two are different problems."""
    root = project(tmp_path)
    (root / "vocabulary.json").write_text(
        '[{"id": "word:x:x", "expression": "x", "reading": "x", "examples": 3}]',
        encoding="utf-8",
    )
    parsed = Parsed("pattern", "Chart", [])
    parsed.patterns = [Item("う・つ・る → って")]
    # A verb to look up: with no class on record, `--ask-jpdb` asks about かう,
    # which is what puts the second, separate failure on stderr.
    parsed.patterns[0].examples = ["かう ⇨ かって"]
    monkeypatch.setattr(
        patterns_module.claude_client, "parse_call", reader({"chart.pdf": parsed})
    )

    def refuse(*_args, **_kwargs):
        raise JankiError("jpdb said 429")

    monkeypatch.setattr(cli.patterns, "verb_groups_from_jpdb", refuse)
    monkeypatch.setattr(cli.jpdb, "api_key_from_env", lambda: "k")
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda _key: object())

    cli.main([
        "--root", str(root), "patterns", "--ask-jpdb",
        str(document(root, "chart.pdf")),
    ])

    err = capsys.readouterr().err
    assert err.count("could not read the collection for verb classes") == 1
    assert "could not look up verb classes: jpdb said 429" in err
