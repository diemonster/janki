"""The documented record schema, checked against the model it documents.

`docs/DATA_MODEL.md` is the one document a person copies *from* — it is what a
reader edits a staging file against, and what they hand-write a record from. A
doc that drifts from the model is worse than no doc: the example shipped for
months with `transitivity: "intransitive"` on 話す, which takes を, and nothing
could catch it because `transitivity` is free text with no validator.

So the example is parsed as a record here rather than proofread.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from japanese_anki.models import ExampleSentence, SourceReference, VocabularyRecord
from japanese_anki.validation import has_errors, validate_records

DOC = Path(__file__).parents[1] / "docs" / "DATA_MODEL.md"


def documented_record() -> dict:
    """The first YAML block in the schema doc."""
    block = re.search(r"```yaml\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)
    assert block, f"{DOC.name} no longer opens with a yaml block"
    return yaml.safe_load(block.group(1))


def test_the_documented_record_is_a_record_janki_would_accept() -> None:
    """Parsed and validated, not read. A reader who copies this and fills in
    their own word should not then be told by `janki build` that the shape they
    were handed is invalid."""
    record = VocabularyRecord.from_dict(documented_record())
    issues = validate_records([record], DOC)

    assert not has_errors(issues), [issue.message for issue in issues]


def test_the_documented_record_shows_every_field_a_record_can_carry() -> None:
    """The README bills this doc as exactly that. Left to prose it fell eight
    fields short — including `register`, whose absence silently costs a card its
    casual half, and `source.raw_fields`, which is where the columns janki does
    not recognise are preserved."""
    doc = documented_record()

    for cls, shown in (
        (VocabularyRecord, doc),
        (ExampleSentence, doc["examples"][0]),
        (SourceReference, doc["source"]),
    ):
        missing = [f.name for f in dataclasses.fields(cls) if f.name not in shown]
        assert not missing, f"{cls.__name__} fields undocumented: {missing}"


#: Fields the doc may legitimately differ from the collection on, because they
#: are illustration rather than fact: prose a person wrote, sentences that grow,
#: file paths, the import that happened to see this word first, and the table a
#: source printed — `source_forms` exists only where an owner bound one part's
#: printed columns, so the schema doc has to show its shape without claiming
#: this repository's own 話す came from such a page.
ILLUSTRATIVE = frozenset(
    {
        "usage_notes",
        "examples",
        "audio",
        "audio_accent",
        "image",
        "source",
        "source_forms",
        "tags",
    }
)


def test_the_documented_word_matches_the_collection_it_was_taken_from() -> None:
    """The example is this repository's own 話す. Where the two disagree one of
    them is wrong about Japanese, and the doc is the copy nobody validates.

    Compared field by field against the model rather than against a hand-listed
    tuple: a list written by hand covers what its author thought of, which is
    how the doc came to claim `LHHH` for a 中高 verb (the collection says
    `LHLL`) and rank 1042 for a word jpdb ranks 200. `audio_accent` is
    illustrative because it is deliberately empty here — it exists for the
    reader who listened and disagreed."""
    from japanese_anki.config import ProjectConfig
    from japanese_anki.io import load_records

    doc = VocabularyRecord.from_dict(documented_record())
    config = ProjectConfig.load(Path(__file__).parents[1])
    if not config.normalized_file.exists():
        pytest.skip("no collection in this checkout")
    real = next(
        (r for r in load_records(config.normalized_file) if r.id == doc.id), None
    )
    if real is None:  # the record may legitimately leave the collection
        pytest.skip(f"{doc.id} is no longer in the collection")

    differing = [
        field.name
        for field in dataclasses.fields(VocabularyRecord)
        if field.name not in ILLUSTRATIVE
        and getattr(doc, field.name) != getattr(real, field.name)
    ]

    assert not differing, (
        f"{doc.id} differs from the collection on {differing} — "
        "one of the two is wrong"
    )
