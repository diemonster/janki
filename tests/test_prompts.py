"""The prompt loader, and the files it ships.

Two things are worth pinning here. The loader's promises — read fresh, sent
byte for byte, a missing file is a clean error — because each of them is a
property someone editing `prompts/` is relying on. And the shipped files
themselves, because they are the deliverable: a person edits them without
opening Python, so a clause disappearing from one should fail a test rather
than quietly weaken a card.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from japanese_anki import extract, prompts

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every file a pass loads by name. `README.md` is deliberately absent — it is
#: the map for humans and nothing sends it.
SHIPPED = (
    "style-guide",
    "extract-auto",
    "extract-table",
    "extract-prose",
    "enrich-examples",
    "polish-meanings",
    "patterns",
    "approve-coverage",
)


# --- the loader ---------------------------------------------------------------


def test_a_prompt_is_sent_exactly_as_written(tmp_path: Path) -> None:
    """No stripping, no normalization, no trailing-newline tidying.

    The file and the request have to be the same bytes, or a person reading
    `prompts/patterns.md` is reading something subtly unlike what the model
    got — and the difference would be invisible from either side.
    """
    body = "  Leading spaces.\n\n\nThree blank lines above.\n\n"
    (tmp_path / prompts.DIRECTORY).mkdir()
    (tmp_path / prompts.DIRECTORY / "odd.md").write_text(body, encoding="utf-8")

    assert prompts.load(tmp_path, "odd") == body


def test_an_edit_takes_effect_with_no_rebuild(tmp_path: Path) -> None:
    """The property the whole directory exists for.

    Cache the read — the obvious optimization, since a prompt changes rarely —
    and a person edits a file, runs the pass, and gets the old text with no
    indication anything is stale. That failure is silent and produces
    plausible output, which is the expensive kind.
    """
    (tmp_path / prompts.DIRECTORY).mkdir()
    path = tmp_path / prompts.DIRECTORY / "shifting.md"
    path.write_text("First.\n", encoding="utf-8")
    assert prompts.load(tmp_path, "shifting") == "First.\n"

    path.write_text("Second.\n", encoding="utf-8")

    assert prompts.load(tmp_path, "shifting") == "Second.\n"


def test_a_missing_prompt_names_its_full_path(tmp_path: Path) -> None:
    """An error, never an empty string. A pass that ran with no instructions
    would return something that looks like an answer."""
    with pytest.raises(prompts.PromptError) as raised:
        prompts.load(tmp_path, "enrich-examples")

    message = str(raised.value)
    assert str(tmp_path / "prompts" / "enrich-examples.md") in message
    assert "will not run a pass without it" in message


def test_the_fingerprint_follows_the_text() -> None:
    """What ties a card to the exact asking that produced it, in staging
    archives and batch records. Equal text, equal fingerprint; one character
    apart, different."""
    assert prompts.fingerprint("same") == prompts.fingerprint("same")
    assert prompts.fingerprint("same") != prompts.fingerprint("same ")


# --- the shipped files ---------------------------------------------------------


@pytest.mark.parametrize("name", SHIPPED)
def test_every_shipped_prompt_loads_and_says_something(name: str) -> None:
    """A pass that resolved to an empty or missing file would send the model
    nothing but the user turn, and still look like it worked."""
    text = prompts.load(REPO_ROOT, name)

    assert text.strip(), name


def test_the_readme_lists_every_prompt_a_pass_sends() -> None:
    """The map has to stay true, because it is how someone finds the file to
    edit. A prompt added without a row here is one nobody knows exists."""
    readme = prompts.load(REPO_ROOT, "README")

    for name in SHIPPED:
        assert f"`{name}.md`" in readme, name


def test_the_directory_holds_no_file_nothing_sends() -> None:
    """The other direction: a leftover draft in here reads as a live prompt.

    Together with the test above this makes `prompts/` exactly the set of
    files that reach a model, plus the README that maps them.
    """
    on_disk = {path.stem for path in (REPO_ROOT / prompts.DIRECTORY).glob("*.md")}

    assert on_disk == {*SHIPPED, "README"}


def test_the_three_extraction_modes_are_three_complete_files() -> None:
    """No shared block is concatenated at send time.

    The trade the directory makes: a paragraph is duplicated across three
    files so that reading one of them requires reading no others. If that ever
    became one file plus three rule blocks, the duplication would vanish and
    so would the property.
    """
    texts = {
        mode: prompts.load(REPO_ROOT, extract.prompt_name(mode))
        for mode in (None, "table", "prose")
    }

    for mode, text in texts.items():
        assert text.startswith("You are reading Japanese study material"), mode
        assert "Never invent a reading" in text, mode
    assert len(set(texts.values())) == 3, "the modes ask for different work"


def test_a_staging_files_provenance_is_the_prompt_files_own_sha(tmp_path: Path) -> None:
    """What ties a card back to the asking that produced it.

    `prompt_provenance` fingerprints the text it was handed, and what it is
    handed is now the file's bytes — so a card extracted last month can be
    checked against `git log prompts/` to see exactly which version of
    `extract-table.md` read its page. If these two ever stopped agreeing, the
    recorded provenance would name a prompt that never ran.
    """
    from japanese_anki.inputs import PreparedInput

    item = PreparedInput(
        kind="document",
        media_type="application/pdf",
        data_b64="",
        origin_path=tmp_path / "lesson.pdf",
    )
    text = prompts.load(REPO_ROOT, "extract-table")

    recorded = extract.prompt_provenance(
        item, model="m", style_guide="g", system=text, mode="table",
        source_sha256="a" * 64,
    )

    assert recorded["system_prompt_fingerprint"] == prompts.fingerprint(text)
