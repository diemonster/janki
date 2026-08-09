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


def project(tmp_path: Path) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'patterns_file = "patterns.json"\n'
        'scan_inbox = "inbox"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
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


def test_re_reading_a_reviewed_document_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`reviewed` is the only human-entered state in this file, and
    `extract_patterns` always returns False. Replacing the entry outright
    un-reviewed the document on any re-read — including the ordinary
    `janki patterns inbox/*.pdf` after adding one new handout — and the only
    signal was a line missing from the next enrich run."""
    root = project(tmp_path)
    monkeypatch.setattr(
        patterns_module.claude_client,
        "parse_call",
        reader({"week11.pdf": Parsed("lesson", "Week 11", ["〜んだ"])}),
    )
    path = document(root, "week11.pdf")
    cli.main(["--root", str(root), "patterns", str(path)])
    cli.main(["--root", str(root), "patterns", "--review", "week11.pdf"])
    assert store_of(root)["week11.pdf"]["reviewed"] is True

    assert cli.main(["--root", str(root), "patterns", str(path)]) == 1

    assert store_of(root)["week11.pdf"]["reviewed"] is True, "still reviewed"
    assert "pass --force" in capsys.readouterr().err


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
