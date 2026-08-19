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
    "enrich-bare-word",
    "approve-coverage",
)

#: The complete card-writing contract has exactly these four input shapes.
#: The style guide is shared context and approve-coverage counts source units;
#: neither is a rich-card task template.
RICH_TEMPLATES = (
    "extract-auto",
    "extract-table",
    "extract-prose",
    "enrich-bare-word",
)


# --- the loader ---------------------------------------------------------------


def test_a_prompt_is_sent_exactly_as_written(tmp_path: Path) -> None:
    """No stripping, no normalization, no trailing-newline tidying.

    The file and the request have to be the same bytes, or a person reading
    `prompts/extract-table.md` is reading something subtly unlike what the model
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
    "raw",
    [
        b"\xef\xbb\xbf",
        "\u200b".encode(),
        "\ufeff  \n".encode(),
        # The two that a *two-pass* strip lets through: whitespace between the
        # marks survives the first pass, the marks survive the second. Without
        # these the guard could be reverted to two passes undetected.
        "\u200b \u200b".encode(),
        "\ufeff \u200b".encode(),
        # And the whitespace a hand-written space list omits. `str.strip`
        # removes these; a literal set of "the spaces I could think of" did
        # not, so a file holding one alone started passing a guard it had
        # failed the day before.
        "\u2029".encode(),
        "\u0085".encode(),
        b"\x1f",
    ],
    ids=["bom", "zwsp", "bom-and-space", "zwsp-space-zwsp", "bom-space-zwsp",
         "paragraph-separator", "next-line", "unit-separator"],
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
        prompts.load(tmp_path, "enrich-bare-word")

    message = str(raised.value)
    assert str(tmp_path / "prompts" / "enrich-bare-word.md") in message
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


def test_card_writing_has_exactly_four_rich_task_templates() -> None:
    """Three source shapes and one bare-record shape, with no extra pass."""
    assert set(SHIPPED) - {"style-guide", "approve-coverage"} == set(RICH_TEMPLATES)
    assert len(RICH_TEMPLATES) == 4


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


def test_auto_extraction_routes_sentence_grids_through_card_selection() -> None:
    """A grammar exercise can teach a pattern and still contain card material.

    The first rich-v3 run over the pair-work source fell between the old two
    nouns: it was neither a vocabulary table nor running prose, so the answer
    returned patterns and zero candidates.  The auto template owns that input
    shape; Python must not inspect the Japanese to repair the answer later.
    """
    text = " ".join(prompts.load(REPO_ROOT, "extract-auto").split())

    assert (
        "When a structured exercise, dialogue, or sentence grid contains "
        "complete Japanese sentences and is not an explicit vocabulary list or "
        "vocabulary table, treat those sentences as prose even if they are laid "
        "out in rows or columns."
    ) in text
    assert "Apply prose candidate selection to those sentences" in text
    assert "set those candidates' source_kind to prose" in text
    assert (
        "Keep source_units and model_reported_unit_count for explicit vocabulary "
        "lists and tables, using the exhaustive accounting above."
    ) in text
    assert "account for every row in source_units" in text
    assert "Link each candidate unit to exactly one candidate" in text
    assert (
        "A document can teach a grammar pattern and also yield vocabulary "
        "candidates."
    ) in text


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
    """Dropping `instructions` from the system blocks must be visible."""
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
        [record], model="m", style_guide="STYLE-GUIDE-MARKER",
        instructions=prompts.load(REPO_ROOT, "enrich-bare-word"),
        ids=["word:話す:はなす"], parse_call=call,
    )

    [sent] = call.sent
    assert prompts.load(REPO_ROOT, "enrich-bare-word") in sent, (
        "verbatim, not a paraphrase"
    )
    # A distinctive marker: the prompt itself contains "Genki" and "Give", so a
    # single capital G was satisfied by the instructions alone.
    assert "STYLE-GUIDE-MARKER" in sent, "and the style guide rides along"


def test_each_rich_input_shape_has_a_complete_distinct_template() -> None:
    """Each file stands alone and asks for the same complete card contract."""
    texts = {name: prompts.load(REPO_ROOT, name) for name in RICH_TEMPLATES}

    assert len(set(texts.values())) == 4
    for name, text in texts.items():
        lowered = text.lower()
        assert "gloss" in lowered, name
        assert "example" in lowered, name
        assert "usage note" in lowered, name


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


def _sent_system(
    monkeypatch: pytest.MonkeyPatch, parsed: object | None = None
) -> list[str]:
    """Capture the system text of every model call a command makes.

    The default answer is a refusal, which is enough to see what was sent and
    stops each command before it writes anything. Pass ``parsed`` when the test
    needs the command to run to completion — what a command *records* about the
    prompts it sent is only visible on the success path.
    """
    from japanese_anki import claude_client

    seen: list[str] = []

    def call(_model, blocks, _content, _schema, _client=None, **_options):
        seen.append(
            "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in blocks
            )
        )
        if parsed is not None:
            return claude_client.CallResult(parsed, "end_turn", None)
        return claude_client.CallResult(None, "refusal", None)

    # Patched on every module that holds a reference, not just the defining
    # one. The reason is not that the production code captures it early — it
    # does not — but that `tests/test_claude_client.py` deletes
    # `japanese_anki.claude_client` from `sys.modules` and re-imports it.
    # `MonkeyPatch` restores the `sys.modules` entry and not the package
    # attribute, so a later `from japanese_anki import claude_client` yields a
    # second module object while the production modules keep the first.
    # Patching only one of the two left the real client reachable and tripped
    # conftest's billed-client guard — in the full suite, not in isolation.
    from japanese_anki import cli, coverage, enrich, extract

    for module in (claude_client, cli, enrich, extract, coverage):
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


def test_enrichment_sends_the_bare_word_rich_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The only bare-record call asks for the whole card in one answer."""
    from japanese_anki import cli

    root = _wiring_project(tmp_path)
    seen = _sent_system(monkeypatch)

    cli.main(["--root", str(root), "enrich", "--ai", "--yes"])

    assert seen, "the command reached the model"
    expected = "enrich-bare-word"
    wanted = prompts.load(REPO_ROOT, expected)
    assert any(wanted in text for text in seen), f"{expected}.md was not sent"
    for other in set(RICH_TEMPLATES) - {expected}:
        assert not any(prompts.load(REPO_ROOT, other) in t for t in seen), (
            f"{other}.md was sent instead"
        )
    # The style guide too. `cli.py` hands it to six senders and to one
    # recorder; replacing it with "" at each of the seven, one at a time, is
    # caught at all seven now — six by assertions like this one and the
    # seventh by the provenance test above, which was the last to be written.
    # Dropping it leaves cards that are still cards, so nothing downstream
    # notices: they are simply no longer written to this project's
    # conventions.
    assert any(prompts.load(REPO_ROOT, "style-guide") in text for text in seen), (
        "the style guide was not sent"
    )


def test_the_batch_builder_sends_the_same_bare_word_rich_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch path builds the same request bodies without calling the model,
    so `parse_call` never fires and the interactive tests above miss it
    entirely. A batch wired to the wrong template bills a whole submission for
    the wrong work, hours before anyone reads a result."""
    from japanese_anki import claude_client, cli

    root = _wiring_project(tmp_path)
    submitted: list[list[dict[str, object]]] = []
    monkeypatch.setattr(
        claude_client, "submit_batch",
        lambda requests: submitted.append(requests) or "batch_test",
    )
    monkeypatch.setattr(cli.claude_client, "submit_batch",
                        lambda requests: submitted.append(requests) or "batch_test",
                        raising=False)

    cli.main(
        ["--root", str(root), "enrich", "--ai", "--batch-submit", "--yes"]
    )

    assert submitted, "a batch was built"
    # Read the system blocks, not a JSON dump of them: `json.dumps` escapes the
    # newlines, so the file's own text is never a substring of the encoded form.
    system = "\n".join(
        block.get("text", "")
        for batch in submitted
        for request in batch
        for block in request["params"]["system"]
    )
    expected = "enrich-bare-word"
    assert prompts.load(REPO_ROOT, expected) in system, f"{expected}.md was not sent"
    for other in set(RICH_TEMPLATES) - {expected}:
        assert prompts.load(REPO_ROOT, other) not in system, (
            f"{other}.md was sent instead"
        )
    assert prompts.load(REPO_ROOT, "style-guide") in system, "the style guide too"


@pytest.mark.parametrize("mode", ["table", "prose", None], ids=["table", "prose", "auto"])
def test_every_extraction_mode_sends_its_own_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str | None
) -> None:
    """All three, not just `--mode table`.

    Pinning one mode left the other two rewirable: `janki extract` with no
    mode, which is the ordinary invocation, could be sent the table rules and
    nothing would notice."""
    from japanese_anki import cli, extract

    root = _wiring_project(tmp_path)
    source = root / "inbox" / "lesson.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"%PDF-1.4\n%x\n")
    seen = _sent_system(monkeypatch)

    argv = ["--root", str(root), "extract", "--yes"]
    if mode is not None:
        argv += ["--mode", mode]
    cli.main([*argv, str(source)])

    assert seen
    wanted = extract.prompt_name(mode)
    assert any(prompts.load(REPO_ROOT, wanted) in t for t in seen), wanted
    for other in set(RICH_TEMPLATES) - {wanted}:
        assert not any(prompts.load(REPO_ROOT, other) in t for t in seen), other
    # The style guide too, and it matters more here than anywhere: the same
    # variable feeds `prompt_provenance`, which writes
    # `style_guide_fingerprint` unconditionally. Send nothing and the committed
    # staging file records the sha of the empty string — permanent provenance
    # naming a guide the run never sent.
    assert any(prompts.load(REPO_ROOT, "style-guide") in t for t in seen), (
        "the style guide was not sent"
    )


def test_the_recorded_provenance_names_the_guide_that_was_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staging file's `style_guide_fingerprint`, from a real run.

    `prompt_provenance` takes the guide as its own argument, so the send and
    the record are two separate hands on the same variable and only the send
    was pinned. Passing `""` to the record alone left the whole suite green
    while every staging file committed a permanent provenance naming the sha
    of the empty string — a card traceable to a guide no run ever sent, which
    is worse than no provenance at all because it reads as an answer.

    This drives `janki extract` to completion rather than checking the call,
    because the file is the artifact that outlives the run.
    """
    from test_extract import candidate, extraction

    from japanese_anki import cli
    from japanese_anki.staging import read_staging

    root = _wiring_project(tmp_path)
    source = root / "inbox" / "lesson.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"%PDF-1.4\n%x\n")
    _sent_system(monkeypatch, parsed=extraction(candidate()))

    cli.main(["--root", str(root), "extract", "--yes", "--mode", "prose", str(source)])

    [staged] = (root / "staging").glob("*.yaml")
    _records, meta = read_staging(staged)
    provenance = meta["prompt_provenance"]
    assert provenance["style_guide_fingerprint"] == prompts.fingerprint(
        prompts.load(REPO_ROOT, "style-guide")
    )
    assert provenance["system_prompt_fingerprint"] == prompts.fingerprint(
        prompts.load(REPO_ROOT, "extract-prose")
    )
