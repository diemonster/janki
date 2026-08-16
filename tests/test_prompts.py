"""The prompt loader, and the files it ships.

Two things are worth pinning here. The loader's promises — read fresh, sent
byte for byte, a missing file is a clean error — because each of them is a
property someone editing `prompts/` is relying on. And the shipped files
themselves, because they are the deliverable: a person edits them without
opening Python, so a clause disappearing from one should fail a test rather
than quietly weaken a card.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

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


def test_crlf_is_not_silently_normalised(tmp_path: Path) -> None:
    """"Sent byte for byte" has to survive a checkout with `core.autocrlf` on.

    `read_text` translates CRLF to LF. That makes the request differ from the
    file invisibly, and — worse — makes the sha-256 recorded in a staging
    archive match no version of the file in `git log prompts/`, so a card's
    provenance names a prompt that never existed.
    """
    (tmp_path / prompts.DIRECTORY).mkdir()
    (tmp_path / prompts.DIRECTORY / "windows.md").write_bytes(b"One.\r\nTwo.\r\n")

    text = prompts.load(tmp_path, "windows")

    assert text == "One.\r\nTwo.\r\n"
    assert prompts.fingerprint(text) == hashlib.sha256(
        (tmp_path / prompts.DIRECTORY / "windows.md").read_bytes()
    ).hexdigest(), "the recorded sha is the file's own"


def test_an_empty_prompt_is_refused_like_a_missing_one(tmp_path: Path) -> None:
    """The same failure as a missing file, arriving by a different door: a
    truncated prompt buys a full paid pass with no instructions and returns
    something that looks like an answer."""
    (tmp_path / prompts.DIRECTORY).mkdir()
    (tmp_path / prompts.DIRECTORY / "blank.md").write_text("   \n\n", encoding="utf-8")

    with pytest.raises(prompts.PromptError, match="is empty"):
        prompts.load(tmp_path, "blank")


@pytest.mark.parametrize(
    "raw", [b"\xef\xbb\xbf", "\u200b".encode(), "\ufeff  \n".encode()],
    ids=["bom", "zero-width-space", "bom-and-space"],
)
def test_a_prompt_of_only_invisible_characters_is_refused(
    tmp_path: Path, raw: bytes
) -> None:
    """`'\ufeff'.strip()` is truthy, so a file truncated to its byte-order mark
    slipped past the empty guard and bought a full paid pass with no
    instructions — the exact failure that guard exists to refuse."""
    (tmp_path / prompts.DIRECTORY).mkdir()
    (tmp_path / prompts.DIRECTORY / "invisible.md").write_bytes(raw)

    with pytest.raises(prompts.PromptError, match="is empty"):
        prompts.load(tmp_path, "invisible")


def test_a_prompt_that_is_not_utf8_says_so(tmp_path: Path) -> None:
    """Rather than escaping as a raw UnicodeDecodeError the CLI cannot format."""
    (tmp_path / prompts.DIRECTORY).mkdir()
    (tmp_path / prompts.DIRECTORY / "sjis.md").write_bytes("日本語".encode("shift_jis"))

    with pytest.raises(prompts.PromptError, match="not UTF-8"):
        prompts.load(tmp_path, "sjis")


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


# --- the prompts actually reach the model ---------------------------------------
#
# The gap a review found after M7.6P shipped: every pass could have dropped its
# instructions, or been wired to the *wrong* file, with the whole suite green.
# The loader tests above prove a file is read; these prove the text is sent, and
# that each pass sends its own.


def _system_text(blocks: object) -> str:
    """Flatten whatever `system_blocks` produced into one searchable string."""
    if isinstance(blocks, str):
        return blocks
    out = []
    for block in blocks or ():
        out.append(block.get("text", "") if isinstance(block, dict) else str(block))
    return "\n".join(out)


def _recorder(result: object):
    sent: list[str] = []

    def call(_model, blocks, _content, _schema, _client=None, **_options):
        sent.append(_system_text(blocks))
        return result

    call.sent = sent  # type: ignore[attr-defined]
    return call


def test_the_enrichment_pass_sends_the_file_it_was_given() -> None:
    """Dropping `instructions` from the system blocks left the suite green, and
    so did wiring this pass to `polish-meanings.md`. A pass sending the wrong
    prompt asks the model for the wrong work and reports success."""
    from japanese_anki import enrich
    from japanese_anki.claude_client import CallResult
    from japanese_anki.models import SourceReference, VocabularyRecord

    call = _recorder(CallResult(None, "refusal", None))
    record = VocabularyRecord(
        id="word:話す:はなす", expression="話す", reading="はなす",
        meanings=["to speak"],
        source=SourceReference(type="shirabe", imported_from="x.csv"),
    )

    enrich.enrich_ai(
        [record], model="m", style_guide="G",
        instructions=prompts.load(REPO_ROOT, "enrich-examples"),
        ids=["word:話す:はなす"], parse_call=call,
    )

    [sent] = call.sent
    assert prompts.load(REPO_ROOT, "enrich-examples") in sent, "verbatim, not a paraphrase"
    assert "G" in sent, "and the style guide leads it"


def test_each_pass_would_notice_being_handed_another_passes_prompt() -> None:
    """The two enrichment prompts are different documents and must stay so.

    Swapping `enrich-examples.md` and `polish-meanings.md` at their call sites
    was undetectable. This does not pin the wiring by itself — the test above
    does — but it pins the premise that makes that test meaningful: if the two
    files ever became interchangeable, nothing downstream could tell.
    """
    examples = prompts.load(REPO_ROOT, "enrich-examples")
    polish = prompts.load(REPO_ROOT, "polish-meanings")

    assert examples != polish
    assert "gloss" in polish.lower(), "polish is about the English glosses"
    assert "example sentence" in examples.lower(), "examples is about sentences"


# --- the wiring: which file each command actually sends -------------------------
#
# The hole a verification review found in the first attempt at this section. The
# test above drives `enrich.enrich_ai` directly and passes the file itself, so it
# pins only that the function forwards its own argument. It says nothing about
# `cli.py` handing that function the *right* file — and repointing any of the
# seven `prompts.load` sites at another template was undetectable, including the
# coverage checker, which would then write a permanent approval naming a prompt
# it never sent.
#
# These drive the real commands and read what reached the model.


def _sent_system(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture the system text of every model call a command makes."""
    from japanese_anki import claude_client

    seen: list[str] = []

    def call(_model, blocks, _content, _schema, _client=None, **_options):
        seen.append(
            "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in blocks
            )
        )
        return claude_client.CallResult(None, "refusal", None)

    # Patched on every module that reads the name, not just the defining one.
    # `cli._enrich_ai` passes `parse_call=claude_client.parse_call` explicitly,
    # so the reference it resolves is `cli`'s — patching only the source module
    # left the real client reachable and the run tripped conftest's
    # build_client guard.
    from japanese_anki import cli, coverage, enrich, extract, patterns

    for module in (claude_client, cli, enrich, extract, patterns, coverage):
        target = getattr(module, "claude_client", module)
        monkeypatch.setattr(target, "parse_call", call, raising=False)
    monkeypatch.setattr(claude_client, "parse_call", call)
    return seen


def _wiring_project(tmp_path: Path) -> Path:
    import json

    from conftest import seed_prompts

    (tmp_path / "janki.toml").write_text(
        '[paths]\nnormalized_file = "vocabulary.json"\nstaging_dir = "staging"\n'
        'scan_inbox = "inbox"\npatterns_file = "patterns.json"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps(
            [
                {
                    "id": "word:話す:はなす",
                    "expression": "話す",
                    "reading": "はなす",
                    "meanings": ["to speak"],
                    "source": {"type": "shirabe", "imported_from": "x.csv"},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    seed_prompts(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["enrich", "--ai", "--yes"], "enrich-examples"),
        (["enrich", "--polish-meanings", "--yes"], "polish-meanings"),
    ],
    ids=["ai", "polish"],
)
def test_each_enrichment_command_sends_its_own_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: str
) -> None:
    """Repointing `cli.py`'s loader call at the other template was invisible.

    The two prompts ask for different work — one writes example sentences, the
    other rewrites English glosses — so a swap produces confidently wrong cards
    and reports success.
    """
    from japanese_anki import cli

    root = _wiring_project(tmp_path)
    seen = _sent_system(monkeypatch)

    cli.main(["--root", str(root), *argv])

    assert seen, "the command reached the model"
    wanted = prompts.load(REPO_ROOT, expected)
    assert any(wanted in text for text in seen), f"{expected}.md was not sent"
    others = {"enrich-examples", "polish-meanings"} - {expected}
    for other in others:
        assert not any(prompts.load(REPO_ROOT, other) in t for t in seen), (
            f"{other}.md was sent instead"
        )


def test_extraction_sends_the_template_for_the_mode_it_was_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--mode table` must send `extract-table.md` and not one of its siblings.

    The three files ask for genuinely different work — exhaustive row
    accounting versus selective prose reading — so a mode wired to the wrong
    template produces a coverage record that means something else entirely.
    """
    from japanese_anki import cli

    root = _wiring_project(tmp_path)
    source = root / "inbox" / "lesson.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"%PDF-1.4\n%x\n")
    seen = _sent_system(monkeypatch)

    cli.main(["--root", str(root), "extract", "--yes", "--mode", "table", str(source)])

    assert seen
    assert any(prompts.load(REPO_ROOT, "extract-table") in t for t in seen)
    for other in ("extract-prose", "extract-auto"):
        assert not any(prompts.load(REPO_ROOT, other) in t for t in seen), other


def test_the_coverage_checker_sends_the_prompt_it_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worst of the unpinned wirings.

    `review_coverage` records `prompt_fingerprint` from the text it was handed,
    so sending a different file writes a permanent `authority: model` approval
    naming a prompt that never ran — provenance that points at the wrong
    question.
    """
    from japanese_anki import coverage

    sent: list[str] = []

    def call(_model, blocks, _content, _schema, _client=None, **_options):
        from japanese_anki.claude_client import CallResult

        sent.append(
            "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in blocks
            )
        )
        return CallResult(
            SimpleNamespace(approved=True, reason="Accounted for."), "end_turn", None
        )

    text = prompts.load(REPO_ROOT, "approve-coverage")

    class _Page:
        origin_path = Path("lesson.pdf")

        def content_block(self) -> dict[str, object]:
            return {"type": "document"}

    verdict = coverage.review_coverage(
        _Page(), {"source_units": []}, model="m", instructions=text, parse_call=call
    )

    assert text in sent[0], "the file it fingerprints is the file it sent"
    assert verdict.prompt_fingerprint == prompts.fingerprint(text), (
        "and the fingerprint it records is that file's"
    )
