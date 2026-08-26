"""``janki patterns`` — local listing and review of extracted source patterns.

Source reading belongs to ``janki extract``. This command has no paid path: it
shows the durable pattern store and records the human decision that one or more
source entries have been reviewed.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from japanese_anki import cli
from japanese_anki import patterns as patterns_module
from japanese_anki.patterns import Pattern, PatternSet

REVIEW_RUN_A = "11111111-1111-4111-8111-111111111111"


def pattern_set(
    source: str,
    *,
    kind: str = "lesson",
    reviewed: bool = False,
    template: str = "〜んだ",
    gloss: str = "explains background",
    review_run_id: str | None = None,
) -> PatternSet:
    return PatternSet(
        source=source,
        kind=kind,
        title=source.removesuffix(".pdf"),
        patterns=(Pattern(template, gloss),),
        reviewed=reviewed,
        prompt_provenance={"request_fingerprint": f"request-for-{source}"},
        review_run_id=review_run_id,
    )


def project(
    tmp_path: Path, store: dict[str, PatternSet] | None = None
) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n" 'patterns_file = "patterns.json"\n',
        encoding="utf-8",
    )
    if store is not None:
        patterns_module.save_store(tmp_path / "patterns.json", store)
    return tmp_path


def stored(root: Path) -> dict[str, PatternSet]:
    return patterns_module.load_store(root / "patterns.json")


def test_listing_an_empty_store_names_the_command_that_populates_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)

    assert cli.main(["--root", str(root), "patterns"]) == 0

    assert (
        capsys.readouterr().out
        == "No patterns extracted yet. Run janki extract on source material.\n"
    )


def test_listing_shows_patterns_and_each_review_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(
        tmp_path,
        {
            "week12.pdf": pattern_set("week12.pdf"),
            "week11.pdf": pattern_set("week11.pdf", reviewed=True),
            "teform.pdf": pattern_set(
                "teform.pdf",
                kind="pattern",
                reviewed=True,
                template="く → いて",
                gloss="te-form",
            ),
        },
    )

    assert cli.main(["--root", str(root), "patterns"]) == 0

    out = capsys.readouterr().out
    assert "week11.pdf — lesson, 1 pattern(s) [reviewed]" in out
    assert "week12.pdf — lesson, 1 pattern(s) [UNREVIEWED]" in out
    assert "teform.pdf — pattern, 1 pattern(s) [reviewed, not used for sentences]" in out
    assert "    く → いて — te-form" in out
    assert out.index("teform.pdf") < out.index("week11.pdf") < out.index("week12.pdf")


def test_review_marks_every_named_source_without_losing_store_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = pattern_set("week11.pdf", review_run_id=REVIEW_RUN_A)
    second = pattern_set("week12.pdf", template="〜ながら")
    root = project(tmp_path, {first.source: first, second.source: second})

    assert (
        cli.main(
            [
                "--root",
                str(root),
                "patterns",
                "--review",
                first.source,
                "--review",
                second.source,
            ]
        )
        == 0
    )

    after = stored(root)
    assert after[first.source].reviewed is True
    assert after[second.source].reviewed is True
    assert after[first.source].patterns == first.patterns
    assert after[first.source].prompt_provenance == first.prompt_provenance
    assert after[first.source].review_run_id == REVIEW_RUN_A
    out = capsys.readouterr().out
    assert "Marked week11.pdf reviewed." in out
    assert "Marked week12.pdf reviewed." in out


def test_two_review_commands_cannot_clobber_each_others_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The path lock must cover load, mutation, and save as one transaction.

    Atomic rename alone only protects one write from being torn. Without a
    lock around the preceding read, both commands load the same old store,
    each marks its own source, and the writer that finishes last silently
    erases the other person's review.
    """
    first = pattern_set("week11.pdf")
    second = pattern_set("week12.pdf")
    root = project(tmp_path, {first.source: first, second.source: second})
    real_atomic_write = patterns_module.atomic_write_text_bound
    first_in_writer = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    results: dict[str, int] = {}

    def held_write(path: Path, text: str, **kwargs: object) -> None:
        payload = json.loads(text)
        if (
            payload[first.source]["reviewed"] is True
            and payload[second.source]["reviewed"] is False
            and not first_in_writer.is_set()
        ):
            first_in_writer.set()
            release_first.wait(timeout=2)
        real_atomic_write(path, text, **kwargs)

    def review(source: str, label: str) -> None:
        try:
            results[label] = cli.main(
                ["--root", str(root), "patterns", "--review", source]
            )
        finally:
            if label == "second":
                second_done.set()

    monkeypatch.setattr(patterns_module, "atomic_write_text_bound", held_write)
    first_thread = threading.Thread(target=review, args=(first.source, "first"))
    second_thread = threading.Thread(target=review, args=(second.source, "second"))
    first_thread.start()
    assert first_in_writer.wait(timeout=2)
    second_thread.start()
    second_finished_before_release = second_done.wait(timeout=0.1)
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert not second_finished_before_release
    assert results == {"first": 0, "second": 0}
    after = stored(root)
    assert after[first.source].reviewed is True
    assert after[second.source].reviewed is True


def test_the_already_locked_writer_does_not_reacquire_the_path_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller spanning staging and patterns needs a non-reentrant seam."""
    path = tmp_path / "patterns.json"
    locks: list[Path] = []

    @contextmanager
    def observed_lock(target: Path) -> Iterator[None]:
        locks.append(Path(target))
        yield

    monkeypatch.setattr(patterns_module, "exclusive_path_lock", observed_lock)
    entry = pattern_set("week11.pdf")

    patterns_module.save_store_under_lock(path, {entry.source: entry})
    assert locks == []

    patterns_module.save_store(path, {entry.source: entry})
    assert locks == [path]


def test_an_unknown_review_name_lists_known_sources_and_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    known = pattern_set("week11.pdf")
    root = project(tmp_path, {known.source: known})

    assert (
        cli.main(
            [
                "--root",
                str(root),
                "patterns",
                "--review",
                known.source,
                "--review",
                "missing.pdf",
            ]
        )
        == 1
    )

    assert "Known: week11.pdf" in capsys.readouterr().err
    assert stored(root)[known.source].reviewed is False


def test_an_unknown_review_name_says_when_the_store_is_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)

    assert cli.main(
        ["--root", str(root), "patterns", "--review", "missing.pdf"]
    ) == 1

    assert "Known: none" in capsys.readouterr().err


def test_reviewing_a_non_lesson_explains_that_it_does_not_steer_sentences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chart = pattern_set(
        "teform.pdf", kind="pattern", template="く → いて", gloss="te-form"
    )
    root = project(tmp_path, {chart.source: chart})

    assert cli.main(
        ["--root", str(root), "patterns", "--review", chart.source]
    ) == 0

    captured = capsys.readouterr()
    assert "Marked teform.pdf reviewed." in captured.out
    assert "only lesson documents steer example sentences" in captured.err
    assert "nothing uses its patterns yet" in captured.err
    assert stored(root)[chart.source].reviewed is True


def test_reviewing_a_lesson_has_no_unused_pattern_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lesson = pattern_set("week11.pdf")
    root = project(tmp_path, {lesson.source: lesson})

    assert cli.main(
        ["--root", str(root), "patterns", "--review", lesson.source]
    ) == 0

    captured = capsys.readouterr()
    assert "Marked week11.pdf reviewed." in captured.out
    assert captured.err == ""
    assert stored(root)[lesson.source].reviewed is True


def test_a_malformed_store_is_a_clean_command_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path)
    (root / "patterns.json").write_text("[]", encoding="utf-8")

    assert cli.main(["--root", str(root), "patterns"]) == 1

    assert "must hold a JSON object keyed by document name" in capsys.readouterr().err
