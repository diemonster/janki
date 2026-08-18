"""``janki patterns`` — local listing and review of extracted source patterns.

Source reading belongs to ``janki extract``. This command has no paid path: it
shows the durable pattern store and records the human decision that one or more
source entries have been reviewed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from japanese_anki import cli
from japanese_anki import patterns as patterns_module
from japanese_anki.patterns import Pattern, PatternSet


def pattern_set(
    source: str,
    *,
    kind: str = "lesson",
    reviewed: bool = False,
    template: str = "〜んだ",
    gloss: str = "explains background",
) -> PatternSet:
    return PatternSet(
        source=source,
        kind=kind,
        title=source.removesuffix(".pdf"),
        patterns=(Pattern(template, gloss),),
        reviewed=reviewed,
        prompt_provenance={"request_fingerprint": f"request-for-{source}"},
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
    first = pattern_set("week11.pdf")
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
    out = capsys.readouterr().out
    assert "Marked week11.pdf reviewed." in out
    assert "Marked week12.pdf reviewed." in out


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
