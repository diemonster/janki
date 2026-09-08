"""Deck-scoped standalone record identities and the deck-scope selector.

A standalone deck holds independent *copies* of words the shared collection may
already have. The copy is an ordinary canonical record under a reserved
namespace, so its GUID — still ``genanki.guid_for(record.id)`` — differs from
the shared word's and carries its own Anki progress.
"""

from __future__ import annotations

from pathlib import Path

import genanki
import pytest
import yaml

from japanese_anki.exporters.anki import DeckSelection, deck_selection
from japanese_anki.identifiers import IdentityError, record_scope_id, stable_record_id
from japanese_anki.io import DataError
from japanese_anki.models import VocabularyRecord

#: The GUIDs janki has always shipped for this word. Pinned as literals: the
#: whole point of a new namespace is that no existing note's identity moves.
SHARED_GUID = "DjM#j+WK=q"
SCOPED_GUID = "h#{)jW47!j"


def _record(record_id: str, *, tags: list[str] | None = None) -> VocabularyRecord:
    return VocabularyRecord(
        id=record_id,
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        tags=list(tags or []),
    )


def test_shared_identity_and_guid_are_unchanged() -> None:
    assert stable_record_id("話す", "はなす") == "word:話す:はなす"
    assert stable_record_id("話す") == "word:話す:"
    assert stable_record_id("話す", "はなす", scope_id="") == "word:話す:はなす"
    assert genanki.guid_for(stable_record_id("話す", "はなす")) == SHARED_GUID


def test_scoped_identity_uses_the_reserved_namespace_and_the_same_normalization() -> None:
    assert stable_record_id("話す", "はなす", scope_id="ab12") == "standalone:ab12:話す:はなす"
    # NFKC and strip, exactly as the shared minter does: the scope changes the
    # namespace, never how the Japanese is spelled into an identity.
    assert stable_record_id(" ｶﾅ ", " かな ", scope_id="ab12") == "standalone:ab12:カナ:かな"


def test_scope_id_is_validated_as_artifact_structure() -> None:
    for bad in ("AB12", "ab 12", "zz", "ab:12", "ab-12", " ab12"):
        with pytest.raises(IdentityError):
            stable_record_id("話す", "はなす", scope_id=bad)


def test_same_word_in_two_scopes_yields_three_distinct_guids() -> None:
    shared = stable_record_id("話す", "はなす")
    first = stable_record_id("話す", "はなす", scope_id="ab12")
    second = stable_record_id("話す", "はなす", scope_id="cd34")
    guids = {genanki.guid_for(value) for value in (shared, first, second)}
    assert len(guids) == 3
    assert genanki.guid_for(first) == SCOPED_GUID


def test_record_scope_id_reads_only_the_reserved_namespace() -> None:
    assert record_scope_id("word:話す:はなす") == ""
    assert record_scope_id("kanji:理") == ""
    assert record_scope_id("drill-audio:lesson:1") == ""
    assert record_scope_id("a hand written id") == ""
    assert record_scope_id("") == ""
    assert record_scope_id("standalone:ab12:話す:はなす") == "ab12"
    # The suffix is opaque: nothing here reads the Japanese half of an id.
    assert record_scope_id("standalone:ab12:anything at all:") == "ab12"
    assert record_scope_id(stable_record_id("話す", "はなす", scope_id="ab12")) == "ab12"


def test_malformed_reserved_namespace_is_refused_rather_than_read_as_shared() -> None:
    for bad in ("standalone:", "standalone:ab12", "standalone::話す:はなす", "standalone:AB12:x:y"):
        with pytest.raises(IdentityError):
            record_scope_id(bad)


def _deck(tmp_path: Path, section: dict[str, object]) -> DeckSelection:
    path = tmp_path / "deck.yaml"
    path.write_text(yaml.safe_dump({"deck": section}, allow_unicode=True), encoding="utf-8")
    return deck_selection(section, path)


def test_a_shared_deck_never_claims_a_standalone_copy(tmp_path: Path) -> None:
    selection = _deck(
        tmp_path,
        {"name": "Lesson", "include_tags": ["lesson"], "intake_tag": "lesson"},
    )
    assert selection.scope_id == ""

    shared = _record("word:話す:はなす", tags=["lesson"])
    scoped = _record("standalone:ab12:話す:はなす", tags=["lesson"])

    assert selection.includes(shared) is True
    assert selection.includes(scoped) is False
    assert "standalone" in (selection.refusal(scoped) or "")


def test_a_standalone_deck_takes_its_own_scope_and_still_applies_its_selectors(
    tmp_path: Path,
) -> None:
    selection = _deck(
        tmp_path,
        {
            "name": "Class verbs",
            "scope_id": "ab12",
            "include_tags": ["verbs"],
            "intake_tag": "verbs",
        },
    )
    assert selection.scope_id == "ab12"

    assert selection.includes(_record("standalone:ab12:話す:はなす", tags=["verbs"])) is True
    # Another deck's scope, the shared collection, and its own scope without the
    # intake tag: three separate refusals, each keeping its own reason.
    assert selection.includes(_record("standalone:cd34:話す:はなす", tags=["verbs"])) is False
    assert selection.includes(_record("word:話す:はなす", tags=["verbs"])) is False
    untagged = _record("standalone:ab12:話す:はなす", tags=[])
    assert selection.includes(untagged) is False
    assert "tagged" in (selection.refusal(untagged) or "")


def test_a_scoped_deck_is_not_an_unfiltered_deck(tmp_path: Path) -> None:
    assert _deck(tmp_path, {"name": "All", "scope_id": "ab12"}).takes_everything is False
    assert _deck(tmp_path, {"name": "All"}).takes_everything is True


def test_a_malformed_record_scope_is_refused_by_every_deck(tmp_path: Path) -> None:
    """One broken identity must not take down every deck's ownership answer."""
    selection = _deck(tmp_path, {"name": "Lesson", "include_tags": ["lesson"]})
    broken = _record("standalone:AB12:話す:はなす", tags=["lesson"])
    assert selection.includes(broken) is False
    assert "standalone" in (selection.refusal(broken) or "")


def test_deck_scope_id_must_be_a_hexadecimal_string(tmp_path: Path) -> None:
    for bad in ("AB12", "zz", 12, ["ab12"]):
        path = tmp_path / "deck.yaml"
        section = {"name": "Class verbs", "scope_id": bad}
        path.write_text(yaml.safe_dump({"deck": section}, allow_unicode=True), encoding="utf-8")
        with pytest.raises(DataError) as caught:
            deck_selection(section, path)
        assert "scope_id" in str(caught.value)
        assert str(path) in str(caught.value)
