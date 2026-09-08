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
    "revise-conjugation-deck",
    "revise-cards",
    "assistant-agent",
    "approve-coverage",
)

#: The complete vocabulary-card contract has exactly these four input shapes.
#: Revision writes an existing conjugation drill's narrower card shape, while
#: the style guide is shared context and approve-coverage counts source units.
RICH_TEMPLATES = (
    "extract-auto",
    "extract-table",
    "extract-prose",
    "enrich-bare-word",
)

CARD_WRITING_TEMPLATES = (
    *RICH_TEMPLATES,
    "revise-conjugation-deck",
    "revise-cards",
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


def test_card_writing_has_exactly_six_task_templates() -> None:
    """Three source shapes, enrichment, and the two explicit revision shapes."""
    assert set(SHIPPED) - {
        "style-guide",
        "assistant-agent",
        "approve-coverage",
    } == set(
        CARD_WRITING_TEMPLATES
    )
    assert len(CARD_WRITING_TEMPLATES) == 6


def test_repository_agent_returns_only_prose_and_one_closed_intent() -> None:
    text = " ".join(prompts.load(REPO_ROOT, "assistant-agent").split())

    assert "at most one closed `action_intent`" in text
    assert "active deck is only a focus" in text
    assert "`revise_cards` changes canonical vocabulary-card fields" in text
    assert "`revise_deck` is only for specialized" in text
    assert "rich conjugation-practice deck" in text
    assert "`enrich_cards` fills missing rich fields" in text
    assert "Unfocused enrichment must include one exact opaque deck resource" in text
    assert "stops every unseen answer in ai-enrichment staging" in text
    assert "`canonical_cards` needs matching exact card resources and record ids" in text
    assert "`deck` needs one exact deck resource and no record ids" in text
    assert "`staged_cards` needs one exact staging-proposal resource" in text
    assert "generated packages, ledger history, and media are retained" in text
    assert "both `audio_words` and `audio_examples`" in text
    assert "leave both absent" in text
    assert "owner's current message explicitly requests" in text
    assert "repository-wide deletion of unreferenced janki-generated audio" in text
    assert "never infer `audio_force` or `audio_prune`" in text.casefold()
    assert "must literally say `prune` for `audio_prune`" in text
    assert "`recover` reuses an already-captured result without another provider call" in text
    assert "never reconstruct recovery from the current deck focus" in text.casefold()
    assert "never execution, Japanese-content authorship, or approval" in text
    assert "`extract`, `enrich --ai`, or `revise`" in text


def test_repository_agent_treats_explicit_kanji_targets_as_their_own_type() -> None:
    """Kanji is a distinct content type, so the asking has to say so.

    Each clause here is a decision the template carries alone: never minting a
    word record, never asking a settled type question, never inventing a
    character or a production cue, and never quietly folding a vocabulary
    request into the same operation.
    """
    text = " ".join(prompts.load(REPO_ROOT, "assistant-agent").split())

    assert "`add_kanji_notes` is the dedicated character-note operation" in text
    assert "one requested character becomes exactly one character note" in text
    assert "never a vocabulary record" in text
    assert "that settles the content type" in text
    assert "do not ask whether they mean words or kanji" in text
    assert "Copy each character into `kanji_characters` exactly as written" in text
    assert "never both" in text
    assert "`refresh_readings` is true only when the owner explicitly asks" in text
    assert "defaults to recognition alone" in text
    assert "requires the owner's own `production_cues`" in text
    assert "that is a separate operation" in text


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


def test_prose_extraction_requires_candidates_from_explicit_lexical_teaching() -> None:
    """A forced prose run must not turn a taught word into pattern-only output.

    Auto mode ignored the same requirement over the pair-work source.  Prose
    mode is the shape-specific escape hatch, so its complete template has to
    carry the lexical-teaching contract itself rather than rely on auto's file.
    """
    text = " ".join(prompts.load(REPO_ROOT, "extract-prose").split())

    assert (
        "Explicit lexical teaching may be a direct expression paired with its "
        "translation or a structured exercise that aligns a target-language "
        "answer or example with a translation, gloss, prompt, cue, or answer key."
    ) in text
    assert (
        "The alignment may appear in one place or in parallel versions of the "
        "same exercise on different pages."
    ) in text
    assert (
        "If any eligible explicitly taught items exist anywhere in the source, "
        "return at least one candidate total from those items."
    ) in text
    assert (
        "Then make one small, compact, high-value selection for the whole source; "
        "this is not one candidate per cue, page, or alignment."
    ) in text


def test_prose_explicit_teaching_has_a_bounded_elementary_word_exception() -> None:
    """The source's own callout wins only after a lexical boundary is proved."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-prose").split())

    assert (
        "Treat either alignment as lexical teaching only when it makes both a "
        "reusable word or lexicalized phrase and that item's meaning unambiguous."
    ) in text
    assert (
        "Do not select a whole sentence, grammar frame, generic prompt, or answer "
        "merely because it is paired, translated, or glossed."
    ) in text
    assert (
        "Choose the smallest source-supported lexical item that carries the "
        "aligned meaning."
    ) in text
    assert (
        "An explicitly taught item that meets the lexical boundary and passes the "
        "Known expressions rule remains eligible even when it is elementary."
    ) in text
    assert (
        "Include a listed expression only when the source itself explicitly "
        "foregrounds a distinct meaning, register, or construction as lesson content."
    ) in text


def test_prose_pattern_focus_cannot_replace_source_taught_vocabulary() -> None:
    """Cards and taught patterns are two outputs from the same paid answer."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-prose").split())

    assert (
        "Neither a grammar focus nor classifying the document as pattern may "
        "suppress those vocabulary candidates; reporting patterns is not a "
        "substitute for reporting taught vocabulary."
    ) in text
    assert "Do not inventory every cue or every word in an answer." in text


def test_prose_selection_keeps_exhaustive_coverage_fields_empty() -> None:
    """Forcing prose changes selection instructions, not its wire contract."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-prose").split())

    assert "Prose selection remains non-exhaustive." in text
    assert "source_units and model_reported_unit_count remain empty" in text


def test_prose_candidate_context_uses_one_exact_source_authority() -> None:
    """Context is evidence from the page, never an answer assembled by the model."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-prose").split())

    assert (
        "For each candidate, set context to the exact complete Japanese source "
        "sentence containing it when one is available."
    ) in text
    assert (
        "When no complete Japanese source sentence is available, use the exact "
        "verbatim callout or source line that teaches the candidate."
    ) in text
    assert "Do not paraphrase, translate, concatenate, or reconstruct context." in text


def test_prose_cross_page_alignment_keeps_page_and_context_local() -> None:
    """A parallel translation is provenance, not text to splice into context."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-prose").split())

    assert (
        "Set page to the page containing the Japanese candidate-bearing sentence, "
        "line, or callout copied into context."
    ) in text
    assert (
        "If its paired cue, translation, or answer key is on another page, "
        "inclusion_reason must name that page and the relationship."
    ) in text
    assert "Never merge text from different pages into one verbatim context." in text


def test_prose_every_candidate_explains_its_source_teaching() -> None:
    """The reviewer can tell why every selected identity crossed the boundary."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-prose").split())

    assert "Every candidate must have a non-empty inclusion_reason." in text
    assert (
        "For a candidate from aligned material, inclusion_reason must explain "
        "how the source explicitly teaches that item."
    ) in text


def test_auto_prose_candidates_explain_their_source_teaching() -> None:
    """Auto routing must not lose prose evidence that explicit prose mode requires."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-auto").split())

    assert (
        "For every candidate with source_kind set to prose, populate a non-empty "
        "inclusion_reason explaining why the source teaches or foregrounds that item."
    ) in text


def test_prose_known_expressions_are_a_decidable_exclusion() -> None:
    """The model receives identities, not the unseen contents of existing cards."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-prose").split())

    assert (
        "The Known expressions list names expressions janki already has. Normally "
        "exclude every expression it lists."
    ) in text
    assert (
        "Include a listed expression only when the source itself explicitly "
        "foregrounds a distinct meaning, register, or construction as lesson content."
    ) in text
    assert (
        "Do not speculate about an unseen existing card; the list establishes "
        "only that the identity exists."
    ) in text


def test_prose_explicit_teaching_requires_one_total_compact_selection() -> None:
    """The candidate floor is source-wide, not a rich-card explosion multiplier."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-prose").split())

    assert (
        "If any eligible explicitly taught items exist anywhere in the source, "
        "return at least one candidate total from those items."
    ) in text
    assert (
        "Then make one small, compact, high-value selection for the whole source; "
        "this is not one candidate per cue, page, or alignment."
    ) in text


@pytest.mark.parametrize("name", ("extract-auto", "extract-table", "extract-prose"))
def test_extraction_patterns_can_be_taught_without_a_heading(name: str) -> None:
    """Repeated evidence teaches only a bounded general construction."""
    text = " ".join(prompts.load(REPO_ROOT, name).split())

    assert (
        "An eligible pattern is a generalizable construction or rule the source "
        "teaches."
    ) in text
    assert (
        "Lexical alignment, mere use, contrast without a stated general rule, and "
        "a marked-wrong example without a source-stated valid rule are not eligible "
        "patterns."
    ) in text
    assert (
        "Repeated unheaded examples establish an eligible pattern only when the "
        "source labels or aligns them as instances of the same generalizable "
        "construction and makes its form and function unambiguous."
    ) in text
    assert (
        "Repeated source examples that are explicitly labelled or aligned can "
        "teach a construction"
    ) not in text


@pytest.mark.parametrize("name", ("extract-auto", "extract-table", "extract-prose"))
def test_extraction_document_kinds_name_the_source_shapes(name: str) -> None:
    """One whole-source primary purpose resolves mixed and embedded material."""
    text = " ".join(prompts.load(REPO_ROOT, name).split())

    assert (
        "Choose exactly one document_kind from the whole source's primary teaching "
        "purpose, not from an isolated page or whichever output arrays are non-empty."
    ) in text
    assert (
        "Use pattern for a standalone chart or reference whose primary purpose is "
        "teaching forms, conjugations, or transformations."
    ) in text
    assert (
        "Use lesson for a broader handout, slide set, dialogue, or exercise teaching "
        "sentence-level grammar or usage, including one that embeds a chart or "
        "reference."
    ) in text
    assert (
        "Use vocabulary when the source is primarily an explicit word list or "
        "vocabulary table."
    ) in text
    assert (
        "Use unknown only when no primary kind can be determined confidently."
    ) in text
    assert (
        "For a mixed source, choose its primary purpose; do not combine kinds or let "
        "an embedded section override the whole source."
    ) in text
    assert "Use unknown for anything else." not in text


@pytest.mark.parametrize("name", ("extract-auto", "extract-table", "extract-prose"))
def test_extraction_has_a_final_pattern_completeness_check(name: str) -> None:
    """Candidate completion cannot consume the pattern half of the paid answer."""
    text = " ".join(prompts.load(REPO_ROOT, name).split())

    assert (
        "Before returning the answer, perform a final pattern completeness check."
    ) in text
    assert (
        "Every eligible source-taught generalizable construction explicitly labelled "
        "by the source must appear exactly once in patterns."
    ) in text
    assert (
        "Every unheaded eligible construction taught unambiguously through repeated "
        "source-labelled or source-aligned instances of the same generalizable rule "
        "must also appear exactly once."
    ) in text
    assert (
        "Do not add a lexical pairing, mere use, contrast without a stated general "
        "rule, or a marked-wrong example without a source-stated valid rule to "
        "satisfy this check."
    ) in text
    assert (
        "Completing the candidate cards is not a reason to omit or postpone a "
        "required pattern."
    ) in text
    assert (
        "Every construction explicitly labelled by the source must appear exactly "
        "once in patterns."
    ) not in text
    assert (
        "Every construction taught unambiguously through repeated labelled or "
        "aligned source examples must also appear exactly once."
    ) not in text


@pytest.mark.parametrize("name", ("extract-auto", "extract-table", "extract-prose"))
def test_extraction_patterns_keep_marked_errors_out_of_templates(name: str) -> None:
    """The model transcribes source-taught corrections; it never performs one."""
    text = " ".join(prompts.load(REPO_ROOT, name).split())

    assert (
        "When the source marks an example wrong, never rewrite it or infer an "
        "unstated replacement."
    ) in text
    assert (
        "Create a corrected pattern only when the source explicitly states a "
        "generalizable valid replacement or rule."
    ) in text
    assert (
        "Its template must contain the source-stated valid form, and its gloss must "
        "state the source-stated scope."
    ) in text
    assert (
        "If the source states only why the marked example is wrong, use that fact "
        "only to bound another independently source-taught pattern, or omit it from "
        "patterns."
    ) in text
    assert (
        "Never put the marked-wrong form in a pattern template or in pattern examples."
    ) in text
    assert (
        "A valid pattern example must be ordinary text actually present in the source "
        "and copied verbatim, or a complete result unambiguously encoded by the "
        "source's visual layout and transcribed under the rule below."
    ) in text
    assert (
        "Otherwise leave the pattern's examples empty; never synthesize or correct "
        "an example."
    ) in text
    assert "source-derived valid example" not in text
    assert "create a pattern from the correction" not in text


@pytest.mark.parametrize("name", ("extract-auto", "extract-table", "extract-prose"))
def test_every_extraction_candidate_gets_a_final_completeness_check(name: str) -> None:
    """One structured object is a card, not a stub the reviewer must author."""
    text = " ".join(prompts.load(REPO_ROOT, name).split())

    assert (
        "Before returning the answer, check every emitted candidate for "
        "completeness."
    ) in text
    assert "It must contain at least one non-empty meaning." in text
    assert (
        "It must contain exactly two examples: one with speech_level “polite” and "
        "one with speech_level “casual”."
    ) in text
    assert (
        "Each example must have non-empty japanese, furigana, romaji, english, and "
        "speech_level fields."
    ) in text
    assert "Do not emit a partial or placeholder candidate." in text
    assert (
        "The selection and accounting rules above decide what must be emitted; "
        "incompleteness is not a reason to omit an otherwise required candidate or "
        "source unit."
    ) in text
    assert "Complete every required candidate before returning the answer." in text
    assert "finish it or omit it" not in text
    assert "usage_notes may remain empty when there is no useful nuance." in text


@pytest.mark.parametrize("name", ("extract-auto", "extract-table", "extract-prose"))
def test_every_extraction_prompt_prioritizes_complete_cards_before_patterns(
    name: str,
) -> None:
    """The primary task is prominent before the longer selection instructions."""
    text = " ".join(prompts.load(REPO_ROOT, name).split())
    priority = (
        "Selecting a candidate commits you to completing its entire card in this "
        "response. Populate every required card field, and complete all candidate "
        "cards before writing patterns."
    )

    assert priority in text
    assert text.index(priority) < text.index("Return a complete study card")


def test_style_guide_does_not_branch_on_a_card_writing_pass() -> None:
    text = " ".join(prompts.load(REPO_ROOT, "style-guide").split())

    assert "source-extraction" not in text
    assert "Bare-word enrichment" not in text
    preferred = text.split("## Preferred enrichments", 1)[1].split("## ", 1)[0]
    assert "Two natural example sentences" not in preferred


def test_prose_final_check_consolidates_identity_and_source_evidence() -> None:
    """Repeated evidence yields one complete card, with reviewable provenance."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-prose").split())

    assert (
        "Before returning prose candidates, perform a final evidence and identity "
        "check."
    ) in text
    assert (
        "Every candidate must have a source page, a non-empty inclusion_reason, "
        "and context copied as the exact complete source sentence or, only when "
        "none is available, the exact verbatim callout or source line."
    ) in text
    assert (
        "When several source passages support the same expression and reading, "
        "return one consolidated candidate for that lexical identity."
    ) in text
    assert (
        "Choose one exact candidate-bearing source location for page and context, "
        "and describe any cross-page support in inclusion_reason."
    ) in text
    assert "Never return duplicate or partial stubs for one identity." in text


def test_auto_extraction_routes_sentence_grids_through_card_selection() -> None:
    """A grammar exercise can teach a pattern and still contain card material.

    The first rich-v3 run over the pair-work source fell between the old two
    nouns: it was neither a vocabulary table nor running prose, so the answer
    returned patterns and zero candidates.  The auto template owns that input
    shape; Python must not inspect the Japanese to repair the answer later.
    """
    text = " ".join(prompts.load(REPO_ROOT, "extract-auto").split())

    # Narrowed after this rule cost 24 rows on a real source: it now applies to
    # grids that are *not* enumerated. An enumerated list is accounted for row
    # by row whatever its rows hold — see
    # `test_auto_accounts_for_enumerated_rows_even_when_they_are_sentences`.
    assert (
        "When a structured exercise, dialogue, or sentence grid is not "
        "enumerated in that way — running dialogue, a fill-in grid, parallel "
        "columns without one numbered item per row — treat its sentences as "
        "prose even if they are laid out in rows or columns."
    ) in text
    assert "Apply prose candidate selection to those sentences" in text
    assert "set those candidates' source_kind to prose" in text
    assert (
        "Keep source_units and model_reported_unit_count for every enumerated "
        "list and every vocabulary table, using the exhaustive accounting above."
    ) in text
    assert "account for every row in source_units" in text
    assert "Link each candidate unit to exactly one candidate" in text
    assert (
        "A document can teach a grammar pattern and also yield vocabulary "
        "candidates."
    ) in text


def test_auto_extraction_requires_candidates_from_explicit_lexical_teaching() -> None:
    """A translation callout or aligned cue is card material, not just layout.

    The second rich-v3 answer saw the structured exercise and quoted its
    sentences as pattern examples, but still returned no candidates.  Routing
    the sentences through prose selection was therefore not strong enough: the
    template must say what the source's explicit lexical alignment entails.
    """
    text = " ".join(prompts.load(REPO_ROOT, "extract-auto").split())

    assert (
        "Outside those sources, explicit lexical teaching may be an expression "
        "paired with its translation or a structured exercise that aligns a "
        "target-language answer or example with a translation, gloss, prompt, "
        "cue, or answer key."
    ) in text
    assert (
        "The alignment may appear in one place or in parallel versions of the "
        "same exercise on different pages."
    ) in text
    assert (
        "When either kind of source-taught alignment contains at least one item "
        "meeting that lexical boundary, return at least one candidate from that "
        "material."
    ) in text
    assert (
        "Keep the candidate selection compact and high-value, and set source_kind "
        "to prose on those candidates."
    ) in text
    assert "study-worthy" not in text


def test_auto_page_routing_does_not_reclassify_the_whole_document() -> None:
    """Source shape is page-local; document_kind has one whole-source value."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-auto").split())

    assert "Judge each page for its source shape when routing its contents." in text
    assert "page for its source shape and classification" not in text
    assert (
        "Page-by-page routing does not require lexical evidence to appear "
        "on only one page: parallel versions of the same exercise may jointly "
        "establish a lexical alignment under the bounded rules below."
    ) in text
    assert "Page-by-page classification" not in text
    assert (
        "The alignment may appear in one place or in parallel versions of the "
        "same exercise on different pages."
    ) in text
    assert (
        "Treat either alignment as lexical teaching only when it makes both a "
        "reusable word or lexicalized phrase and that item's meaning unambiguous."
    ) in text


def test_auto_aligned_material_never_overrides_the_table_contract() -> None:
    """Translated rows stay table rows when the source is a vocabulary table."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-auto").split())

    assert (
        "The aligned-material rules below apply only outside explicit vocabulary "
        "lists and vocabulary tables."
    ) in text
    assert (
        "Those list and table sources retain the exhaustive source_units contract "
        "above, and their candidates keep source_kind set to table."
    ) in text


def test_auto_alignment_identifies_a_lexical_item_not_a_paired_utterance() -> None:
    """Alignment proves a bounded lexical identity, not that every pair is one."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-auto").split())

    assert (
        "Treat either alignment as lexical teaching only when it makes both a "
        "reusable word or lexicalized phrase and that item's meaning unambiguous."
    ) in text
    assert (
        "Do not select a whole sentence, grammar frame, generic prompt, or answer "
        "merely because it is paired, translated, or glossed."
    ) in text
    assert (
        "Choose the smallest source-supported lexical item that carries the "
        "aligned meaning."
    ) in text


def test_auto_extraction_keeps_pattern_focus_from_suppressing_vocabulary() -> None:
    """Pattern reporting and bounded prose selection are independent outputs."""
    text = " ".join(prompts.load(REPO_ROOT, "extract-auto").split())

    assert (
        "Neither a grammar focus nor classifying the document as pattern may "
        "suppress those vocabulary candidates; reporting patterns is not a "
        "substitute for reporting taught vocabulary."
    ) in text
    assert "Do not inventory every cue or every word in an answer." in text
    assert (
        "This requirement does not make ordinary prose exhaustive; prose "
        "selection remains non-exhaustive."
    ) in text
    assert "Prose selection is not exhaustive." in text


@pytest.mark.parametrize(
    "name", ("extract-auto", "extract-table", "extract-prose")
)
def test_visual_pattern_charts_become_self_contained_plain_text(name: str) -> None:
    """Formatting cannot survive inside a pattern card's JSON string.

    The te-form song placed whole verbs in one column, highlighted only their
    final kana, and put replacement suffixes in another.  Flattening those
    cells produced misleading examples such as ``かく → いて``.  The prompt,
    not a Japanese-aware repair, owns that source-reading instruction in every
    complete extraction mode.
    """
    text = " ".join(prompts.load(REPO_ROOT, name).split())

    assert "Keep ordinary text examples verbatim." in text
    assert (
        "A pattern example whose meaning depends on visual layout must instead "
        "be a faithful, self-contained plain-text transcription."
    ) in text
    assert (
        "When a chart unambiguously aligns a complete input with a replacement "
        "suffix or other fragment, transcribe it as the complete input and "
        "complete transformed result, not as a whole-expression-to-fragment "
        "transformation."
    ) in text
    assert (
        "If the complete result is not unambiguously encoded by the source's "
        "own layout and labels, omit the example rather than guess."
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


def test_conjugation_revision_is_a_complete_narrow_content_prompt() -> None:
    """The third writing path is selected-card revision, never chat authority."""
    text = " ".join(
        prompts.load(REPO_ROOT, "revise-conjugation-deck").split()
    )

    assert "current deck content for only the selected cards" in text
    assert "every selected record ID exactly once, in the same order" in text
    assert "exactly two complete examples for each: one polite and one casual" in text
    assert "Both examples must actually demonstrate the deck's named" in text
    assert "preserve a current example exactly" in text
    assert "Do not preserve an audio field" in text
    assert "romaji" not in text.casefold()


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
        'scan_inbox = "inbox"\npatterns_file = "patterns.json"\n'
        # These tests fake the API `parse_call` to read back the exact bytes
        # sent. The default transport is the owner's subscription, which
        # would spawn the CLI instead of the seam they inspect.
        '[ai]\nextract_provider = "anthropic-api"\n',
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
    for other in set(CARD_WRITING_TEMPLATES) - {expected}:
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
    for other in set(CARD_WRITING_TEMPLATES) - {expected}:
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
    for other in set(CARD_WRITING_TEMPLATES) - {wanted}:
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


def test_auto_accounts_for_enumerated_rows_even_when_they_are_sentences() -> None:
    """A real extraction lost 24 rows to the clause this pins.

    `medical-conditions-vocab.pdf` page 3, section "(3) Other useful
    expressions", is a numbered list — 1. 熱があります through 24. ひりひりします.
    The auto template said to treat a structured thing containing complete
    Japanese sentences as prose, and prose selection is not exhaustive, so the
    whole section was skipped without appearing under any disposition. The
    coverage gate caught it; nothing else would have.

    The distinguishing feature is enumeration, not whether a row holds a word
    or a sentence. Accounting for a row is also not the same as making a card
    for it — a row that teaches no reusable item is still a unit, with a
    disposition and a reason.
    """
    text = " ".join(prompts.load(REPO_ROOT, "extract-auto").split())

    assert "An enumerated list is a list whatever its rows hold." in text
    assert "even when each row is a complete Japanese sentence" in text
    assert (
        "Never skip an enumerated row on the grounds that its content reads "
        "as prose."
    ) in text
    # The original intent survives: an unenumerated grid is still prose.
    assert (
        "running dialogue, a fill-in grid, parallel columns without one "
        "numbered item per row"
    ) in text
    # Accounting for a row is not the same as making a card for it.
    assert "Accounting for a row is not the same as making a card for it" in text
    assert "the disposition non-vocabulary and a reason" in text
