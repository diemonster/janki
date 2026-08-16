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

from conftest import seed_prompts
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
    seed_prompts(tmp_path)
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


