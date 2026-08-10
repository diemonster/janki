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
