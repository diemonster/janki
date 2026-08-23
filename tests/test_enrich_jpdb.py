"""Dictionary enrichment from jpdb — ``janki enrich --jpdb``.

No network (IMPLEMENTATION_PLAN rule 6): the client is driven through a fake
transport that answers ``/parse`` and ``/lookup-vocabulary`` from canned
dictionary data. Rows are built in the client's default field order, so a
change to ``DEFAULT_VOCABULARY_FIELDS`` breaks these tests loudly rather than
silently shifting a column.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from japanese_anki import cli, enrich, jpdb, pitch
from japanese_anki.enrich import (
    EnrichError,
    enrich_records,
    format_field_diff,
    parse_force_fields,
    suggest_readings,
)
from japanese_anki.jpdb import JpdbClient
from japanese_anki.models import (
    PROVISIONAL_FIELDS_KEY,
    SourceReference,
    VocabularyRecord,
    mark_provisional,
    provisional_fields,
)

# The vocabulary row order jpdb answers a default `/parse` in.
assert jpdb.DEFAULT_VOCABULARY_FIELDS == (
    "vid",
    "sid",
    "spelling",
    "reading",
    "pitch_accent",
    "frequency_rank",
    "part_of_speech",
)


def vocab(
    vid: int,
    sid: int,
    spelling: str,
    reading: str,
    accent: list[str],
    rank: int,
    pos: list[str],
) -> list[Any]:
    return [vid, sid, spelling, reading, accent, rank, pos]


def parse_response(*entries: tuple[Any, list[Any]]) -> dict[str, Any]:
    """A ``/parse`` response: one ``(token_furigana, vocabulary_row)`` per word."""
    tokens = [[index, furigana] for index, (furigana, _) in enumerate(entries)]
    return {"tokens": [tokens], "vocabulary": [row for _, row in entries]}


HANASU = vocab(1562350, 4280520068, "話す", "はなす", ["LHLL"], 200, ["vt", "v5", "v5s"])
HANASU_FURIGANA = [["話", "はな"], "す"]

# 凄い: jpdb states no verb class for an い-adjective, so its part of speech is
# what has to carry the inflection. The `adv` code leads, which is why the word
# class is decided by precedence rather than by taking the first code.
SUGOI = vocab(1379180, 333, "凄い", "すごい", ["LHHH"], 3100, ["adv", "adj-i"])
SUGOI_FURIGANA = [["凄", "すご"], "い"]

# 一日: the homograph the forced-furigana path exists for.
ICHINICHI = vocab(1579110, 111, "一日", "いちにち", ["LHHH"], 900, ["n"])
TSUITACHI = vocab(1579110, 222, "一日", "ついたち", ["LHHL"], 1400, ["n"])
ICHINICHI_FURIGANA = [["一日", "いちにち"]]
TSUITACHI_FURIGANA = [["一日", "ついたち"]]


class FakeApi:
    """Answers ``/parse`` and ``/lookup-vocabulary`` from canned data.

    ``/parse`` is keyed on whether the request forced a reading, because that
    is the distinction the enrichment turns on: the unforced answer is what
    janki's stored reading is *compared* to, and the forced one is what it may
    be written from. A fake that returned the same thing either way could not
    fail the homograph test.
    """

    def __init__(
        self,
        unforced: dict[str, dict[str, Any]],
        forced: dict[tuple[str, str], dict[str, Any]] | None = None,
        senses: dict[tuple[int, int], dict[str, Any]] | None = None,
    ) -> None:
        self.unforced = unforced
        self.forced = forced or {}
        self.senses = senses or {}
        self.bodies: list[tuple[str, dict[str, Any]]] = []

    def __call__(
        self, url: str, body: dict[str, Any], headers: dict[str, str]
    ) -> tuple[int, Any]:
        endpoint = url.rsplit("/api/v1/", 1)[-1]
        self.bodies.append((endpoint, body))
        if endpoint == "parse":
            return 200, self._parse(body)
        if endpoint == "lookup-vocabulary":
            return 200, self._lookup(body)
        raise AssertionError(f"unexpected request to {url}")

    def _parse(self, body: dict[str, Any]) -> dict[str, Any]:
        text = body["text"][0]
        if "furigana" in body:
            reading = body["furigana"][0][0][2]
            key = (text, reading)
            if key not in self.forced:
                raise AssertionError(f"fake has no forced parse for {key}")
            return self.forced[key]
        if text not in self.unforced:
            raise AssertionError(f"fake has no parse for {text!r}")
        return self.unforced[text]

    def _lookup(self, body: dict[str, Any]) -> dict[str, Any]:
        fields = body["fields"]
        rows = []
        for vid, sid in body["list"]:
            sense = self.senses.get((vid, sid))
            if sense is None:
                raise AssertionError(f"fake has no sense for {(vid, sid)}")
            rows.append([sense.get(name) for name in fields])
        return {"vocabulary_info": rows}

    def calls(self, endpoint: str) -> list[dict[str, Any]]:
        return [body for name, body in self.bodies if name == endpoint]


def client_for(api: FakeApi) -> JpdbClient:
    return JpdbClient("test-key", api, sleep=lambda _s: None, jitter=lambda: 0.0)


def record(**overrides: Any) -> VocabularyRecord:
    """A Shirabe-shaped record: identity filled in, dictionary fields empty."""
    values: dict[str, Any] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "source": SourceReference(type="shirabe", imported_from="export.csv"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def hanasu_api() -> FakeApi:
    return FakeApi({"話す": parse_response((HANASU_FURIGANA, HANASU))})


# --- filling ----------------------------------------------------------------


def test_every_field_jpdb_states_or_janki_computes_is_filled() -> None:
    api = hanasu_api()

    result = enrich_records(client_for(api), [record()])

    enriched = result.records[0]
    assert enriched.furigana == "話[はな]す"
    assert enriched.romaji == "hanasu"
    assert enriched.part_of_speech == "verb"
    assert enriched.verb_group == "godan"
    assert enriched.pitch_accent == ["LHLL"]
    assert pitch.has_source_binding(enriched)
    assert enriched.frequency_rank == 200
    assert enriched.conjugations["negative"] == "話さない"
    assert result.looked_up == 1
    assert result.warnings == []


def test_an_i_adjective_is_conjugated_from_its_part_of_speech() -> None:
    """jpdb has no verb class for an い-adjective, so `verb_group` is empty and
    the part of speech is the only thing that can drive the table.

    Narrowing the `verb_group or part_of_speech` fallback to `verb_group` alone
    leaves every jpdb-enriched い-adjective with `conjugations: {}` — a whole
    word class silently losing its drill forms — and, measured after M8.4
    deleted the case that pinned this, left the suite green. The label
    precedence has its own test; this pins that the label reaches `conjugate`.
    """
    api = FakeApi({"凄い": parse_response((SUGOI_FURIGANA, SUGOI))})

    result = enrich_records(
        client_for(api),
        [record(id="word:凄い:すごい", expression="凄い", reading="すごい",
                meanings=["amazing"])],
    )

    enriched = result.records[0]
    assert enriched.part_of_speech == "i-adjective"
    assert enriched.verb_group == "", "jpdb states no class for an い-adjective"
    assert enriched.conjugations["negative"] == "凄くない"
    assert enriched.conjugations["past"] == "凄かった"


def test_transitivity_is_filled_from_the_dictionarys_own_codes() -> None:
    """The last field with no filler, wired at last.

    `transitivity` was set at import and backfilled by nothing since the
    hand-paste era, while jpdb's `vt`/`vi` codes carried the dictionary's
    answer the whole time — 話す comes back `["vt", "v5", "v5s"]`.
    """
    result = enrich_records(client_for(hanasu_api()), [record()])

    assert result.records[0].transitivity == "transitive"
    assert "transitivity" in result.changes["word:話す:はなす"]


def test_a_noun_is_not_given_a_transitivity() -> None:
    """JMdict tags suru-nouns with vt/vi too — 仕事 is ["n", "vs", "vi"] — but
    transitivity is a property of a verb.

    Writing it on a record whose own part of speech says "noun" puts a
    contradiction on the card, which shows the field unconditionally and has no
    validation rule for it.
    """
    shigoto = vocab(1330910, 4, "仕事", "しごと", ["LHH"], 300, ["n", "vs", "vi"])
    api = FakeApi({"仕事": parse_response(([["仕事", "しごと"]], shigoto))})

    result = enrich_records(
        client_for(api),
        [record(id="word:仕事:しごと", expression="仕事", reading="しごと",
                meanings=["work"])],
    )

    enriched = result.records[0]
    assert enriched.part_of_speech == "noun"
    assert enriched.transitivity == "", "a noun has no transitivity to state"


def test_a_word_the_dictionary_tags_both_ways_gets_no_transitivity() -> None:
    """する is `["aux-v", "vi", "suf", "vt", "vs"]` — measured, not guessed.

    Which one is true depends on the sense, so there is no single answer to
    state, and stating one anyway is the guess this project does not make. An
    empty field is the honest card; `usage_notes` is where a real distinction
    belongs.
    """
    both = vocab(1157170, 9, "する", "する", ["LH"], 10, ["aux-v", "vi", "suf", "vt", "vs"])
    api = FakeApi({"する": parse_response(([["する", "する"]], both))})

    result = enrich_records(
        client_for(api),
        [record(id="word:する:する", expression="する", reading="する", meanings=["to do"])],
    )

    assert result.records[0].transitivity == "", "tagged both ways, so neither"


def test_the_reading_is_never_written_even_when_jpdb_states_one() -> None:
    # The reading is half of stable_record_id: writing it would re-ID the
    # record and orphan its Anki review history.
    api = hanasu_api()

    result = enrich_records(client_for(api), [record()])

    assert result.records[0].reading == "はなす"
    assert result.records[0].id == "word:話す:はなす"
    assert "reading" not in result.changes["word:話す:はなす"]


def test_meanings_are_not_touched_by_a_dictionary_pass() -> None:
    api = hanasu_api()

    result = enrich_records(client_for(api), [record(meanings=[])])

    assert result.records[0].meanings == []
    assert "meanings" not in result.changes.get("word:話す:はなす", {})


def test_a_populated_field_is_left_exactly_as_the_human_wrote_it() -> None:
    api = hanasu_api()
    curated = record(part_of_speech="godan verb (u-verb)", pitch_accent=["HLLL"])

    result = enrich_records(client_for(api), [curated])

    enriched = result.records[0]
    assert enriched.part_of_speech == "godan verb (u-verb)"
    assert enriched.pitch_accent == ["HLLL"]
    assert not pitch.has_source_binding(enriched)
    changed = result.changes["word:話す:はなす"]
    assert "part_of_speech" not in changed
    assert "pitch_accent" not in changed
    assert "furigana" in changed


def provisional_record(**overrides: Any) -> VocabularyRecord:
    """An extract-sourced record whose meanings and POS are model claims."""
    values: dict[str, Any] = {
        "source": SourceReference(type="extract", imported_from="page.jpg"),
        "meanings": ["to converse"],
        "part_of_speech": "noun",
    }
    values.update(overrides)
    return mark_provisional(record(**values))


def hanasu_senses() -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (1562350, 4280520068): {
            "reading": "はなす",
            "alt_sids": [],
            "meanings_chunks": [["to talk", "to speak"], ["to tell"]],
        }
    }


def test_an_exact_match_settles_provisional_pos_and_never_meanings() -> None:
    # A model's *grammatical* claim holds its seat only until the dictionary
    # states the word class — jpdb knows 話す is a verb no matter which sense
    # this source taught. Its *glosses* are a different kind of claim and are
    # refused: see `DICTIONARY_MAY_NOT_SETTLE`.
    api = FakeApi(
        unforced={"話す": parse_response((HANASU_FURIGANA, HANASU))},
        senses=hanasu_senses(),
    )

    result = enrich_records(client_for(api), [provisional_record()])

    [updated] = result.records
    assert updated.part_of_speech == "verb"
    assert updated.meanings == ["to converse"]
    changed = result.changes["word:話す:はなす"]
    assert "part_of_speech" in changed
    assert "meanings" not in changed
    # POS settled, meanings still awaiting a reader.
    assert provisional_fields(updated) == ["meanings"]
    # No gloss lookup was even attempted: the refusal is upstream of the call,
    # so it costs nothing rather than fetching an answer to discard.
    assert api.calls("lookup-vocabulary") == []


def test_a_human_edit_after_extraction_outranks_the_dictionary() -> None:
    # The value binding is the enforcement: editing the field after extraction
    # breaks it, and a broken binding reads as curated. The stale mark is
    # cleared so it can never authorize overwriting the edit later; the
    # untouched POS claim still reconciles normally.
    edited = replace(provisional_record(), meanings=["to chat (hand-checked)"])
    api = FakeApi(
        unforced={"話す": parse_response((HANASU_FURIGANA, HANASU))},
        senses=hanasu_senses(),
    )

    result = enrich_records(client_for(api), [edited])

    [updated] = result.records
    assert updated.meanings == ["to chat (hand-checked)"]
    assert updated.part_of_speech == "verb"
    assert "meanings" not in result.changes["word:話す:はなす"]
    assert PROVISIONAL_FIELDS_KEY not in updated.source.raw_fields
    assert any("edited since extraction" in warning for warning in result.warnings)


def test_a_different_reading_holds_provisional_fields() -> None:
    # The reviewed identity does not match what jpdb resolved, so no entry is
    # confirmed as this word: nothing is written and the claims stay marked.
    api = FakeApi(
        unforced={"話す": parse_response((HANASU_FURIGANA, HANASU))},
        forced={("話す", "はなし"): parse_response((HANASU_FURIGANA, HANASU))},
        senses={(1562350, 4280520068): {"reading": "はなす", "alt_sids": []}},
    )
    held = provisional_record(id="word:話す:はなし", reading="はなし")

    result = enrich_records(client_for(api), [held])

    assert result.changes == {}
    [updated] = result.records
    assert updated.meanings == ["to converse"]
    assert updated.part_of_speech == "noun"
    assert provisional_fields(updated) == ["meanings", "part_of_speech"]


def test_a_silent_dictionary_leaves_the_mark_and_says_so() -> None:
    # jpdb states no word class for this entry, so the provisional POS has
    # nothing to reconcile against: it keeps the model's value, keeps its
    # mark, and the run says which field is still waiting.
    silent = vocab(1562350, 4280520068, "話す", "はなす", ["LHLL"], 200, [])
    api = FakeApi(
        unforced={"話す": parse_response((HANASU_FURIGANA, silent))},
        senses={(1562350, 4280520068): {"reading": "はなす", "alt_sids": []}},
    )

    result = enrich_records(client_for(api), [provisional_record()])

    [updated] = result.records
    assert updated.part_of_speech == "noun"
    assert sorted(provisional_fields(updated)) == ["meanings", "part_of_speech"]
    assert any(
        "no answer for provisional part_of_speech" in warning
        for warning in result.warnings
    )
    # Load-bearing, and not a duplicate of the warning check: the warning is
    # the only thing above that distinguishes this from the pre-refusal code,
    # and it does so by a substring that a reword or a sort of the `unresolved`
    # join would quietly retire. This says the same thing structurally — the
    # meanings mark was never a candidate, so no gloss was fetched to judge it.
    assert api.calls("lookup-vocabulary") == []


def test_mark_clears_are_reported_for_persistence() -> None:
    # A cleared mark that only lives in memory comes back to make the same
    # network calls and print the same warning on every future run — the
    # caller's save gate reads `changes`, so clears need their own channel.
    edited = replace(provisional_record(), meanings=["to chat (hand-checked)"])
    api = FakeApi(
        unforced={"話す": parse_response((HANASU_FURIGANA, HANASU))},
        senses=hanasu_senses(),
    )

    result = enrich_records(client_for(api), [edited])

    assert result.cleared["word:話す:はなす"] == ["meanings"]


def test_a_confirming_dictionary_also_reports_its_clear() -> None:
    # The dictionary agreeing with the model claim settles the authority
    # question with no visible field change; the clear still has to reach disk.
    confirmed = provisional_record(part_of_speech="verb")
    api = FakeApi(
        unforced={"話す": parse_response((HANASU_FURIGANA, HANASU))},
        senses=hanasu_senses(),
    )

    result = enrich_records(client_for(api), [confirmed])

    assert result.cleared["word:話す:はなす"] == ["part_of_speech"]
    # `meanings` is never in that list: a dictionary cannot confirm a sense it
    # was never allowed to read, so the mark it leaves is not a clear.
    [updated] = result.records
    assert provisional_fields(updated) == ["meanings"]


def test_a_forced_write_settles_the_mark_it_overwrote() -> None:
    # --force-fields part_of_speech writes the dictionary's value over the
    # provisional claim. The mark is settled by janki's own write — leaving it
    # would misreport the next run's stale-clear as a human edit that never
    # happened, and rewrite the collection again to say so.
    api = FakeApi(
        unforced={"話す": parse_response((HANASU_FURIGANA, HANASU))},
        senses=hanasu_senses(),
    )

    result = enrich_records(
        client_for(api), [provisional_record()], force_fields=("part_of_speech",)
    )

    from japanese_anki.models import provisional_entries

    [updated] = result.records
    assert updated.part_of_speech == "verb"
    assert "part_of_speech" not in dict(provisional_entries(updated))
    assert not any("edited since extraction" in warning for warning in result.warnings)


def test_the_jpdb_pass_refuses_a_force_fields_meanings_flag() -> None:
    """The same refusal one layer up, and the layer a user actually touches.

    `DICTIONARY_MAY_NOT_SETTLE` stops the pass writing meanings from inside;
    this stops `--jpdb --force-fields meanings` being spelled at all, so the
    answer a user gets is a sentence rather than a silent no-op. Untested until
    now: if `meanings` were ever added to `ENRICHABLE_FIELDS` only the
    behaviour tests would have objected, and this is the cheaper alarm.
    """
    with pytest.raises(EnrichError, match="--ai"):
        parse_force_fields("meanings")

    # And it is a real field name to the pass that does own it, so the refusal
    # is about authority rather than a typo check.
    assert parse_force_fields("meanings", ai=True) == ("meanings",)


def test_a_kana_homograph_cannot_settle_a_provisional_claim() -> None:
    # あめ the candy tokenizes to 雨 the rain — spelling different, readings
    # agreeing all the way. Fine for filling empty fields; not authority to
    # overwrite a provisional claim about a different word.
    candy = mark_provisional(
        record(
            id="word:あめ:あめ",
            expression="あめ",
            reading="あめ",
            meanings=["candy"],
            part_of_speech="noun",
            source=SourceReference(type="extract", imported_from="page.jpg"),
        )
    )
    rain = vocab(1000090, 1, "雨", "あめ", ["HLL"], 250, ["n"])
    api = FakeApi(unforced={"あめ": parse_response(([["雨", "あめ"]], rain))})

    result = enrich_records(client_for(api), [candy])

    [updated] = result.records
    assert updated.meanings == ["candy"]
    assert updated.part_of_speech == "noun"
    assert provisional_fields(updated) == ["meanings", "part_of_speech"]
    assert any("stay provisional" in warning for warning in result.warnings)
    assert api.calls("lookup-vocabulary") == []


def test_a_kana_homograph_cannot_replace_the_meaning_its_source_taught() -> None:
    """The bug this rule exists for, with the real word that hit it.

    A Meguro Language Center medical sheet taught おたふく = "mumps". jpdb's
    entry for おたふく is お多福, the *other* word spelled that way: "homely
    woman (esp. one with a small low nose ...)". Reconciliation took it, and
    the shipped card read "homely woman" above its own example sentence,
    "My child came down with the mumps."

    Nothing earlier in the pass can catch this. The spelling matches exactly —
    both are おたふく — so the wrong-spelling guard passes; the reading matches
    too. Only the *sense* differs, and telling two senses apart is reading
    Japanese, which a gloss list keyed on spelling does not do. So the refusal
    has to be categorical, and this is the test that says so.

    63 of that sheet's 85 cards were overwritten this way before the rule
    existed, which is why the assertion is `== ["mumps"]` and not merely
    "unchanged": what makes the bug expensive is that the replacement is
    plausible prose, so nobody notices until they read the sentence under it.
    """
    otafuku = vocab(1000320, 1, "おたふく", "おたふく", ["LHHHH"], 92300, ["n"])
    api = FakeApi(
        unforced={"おたふく": parse_response(([["おたふく", "おたふく"]], otafuku))},
        senses={
            (1000320, 1): {
                "reading": "おたふく",
                "alt_sids": [],
                "meanings_chunks": [
                    ["homely woman (esp. one with a small low nose)", "plain woman"]
                ],
            }
        },
    )
    taught = mark_provisional(
        record(
            id="word:おたふく:おたふく",
            expression="おたふく",
            reading="おたふく",
            meanings=["mumps"],
            source=SourceReference(type="extract", imported_from="medical.pdf"),
        )
    )

    result = enrich_records(client_for(api), [taught])

    [updated] = result.records
    assert updated.meanings == ["mumps"]
    assert "meanings" not in result.changes.get("word:おたふく:おたふく", {})
    # Still a model claim, still marked, still waiting for a reader — the
    # dictionary declining to answer is not the same as it confirming.
    assert "meanings" in provisional_fields(updated)
    # And no gloss was fetched: the refusal is a rule, not a comparison, so
    # there is no request whose answer someone could later decide to trust.
    assert api.calls("lookup-vocabulary") == []
    # The rest of the pass still writes, and this pins the boundary rather
    # than endorsing it: these come from the same wrong-word entry, and for an
    # all-kana record no guard can say otherwise. DESIGN.md draws the line at
    # *sense* — accent and frequency describe the word either way — and says
    # why the residue is left to a person instead of refused wholesale.
    assert updated.frequency_rank == 92300
    assert updated.pitch_accent == ["LHHHH"]


def test_a_suru_stem_entry_cannot_settle_a_provisional_claim() -> None:
    # The suru-suffix allowance deliberately accepts 勉強's entry for a
    # 勉強する record, so the entry reached here spells something other than
    # the record does. The spelling guard is what stops that entry settling a
    # provisional claim: 勉強 is a noun, and letting it say so would file the
    # verb 勉強する under the stem's word class.
    compound = mark_provisional(
        record(
            id="word:勉強する:べんきょうする",
            expression="勉強する",
            reading="べんきょうする",
            meanings=["to study"],
            part_of_speech="verb",
            source=SourceReference(type="extract", imported_from="page.jpg"),
        )
    )
    benkyou = vocab(1512670, 1, "勉強", "べんきょう", ["LHHHHH"], 1000, ["n", "vs"])
    parse = parse_response(([["勉", "べん"], ["強", "きょう"]], benkyou))
    api = FakeApi(
        unforced={"勉強する": parse},
        forced={("勉強する", "べんきょうする"): parse},
        senses={(1512670, 1): {"reading": "べんきょう", "alt_sids": []}},
    )

    result = enrich_records(client_for(api), [compound])

    [updated] = result.records
    assert updated.meanings == ["to study"]
    assert updated.part_of_speech == "verb"
    assert sorted(provisional_fields(updated)) == ["meanings", "part_of_speech"]
    assert any("stay provisional" in warning for warning in result.warnings)
    # One pinned parse, not two. The surviving one is the kana-homograph retry,
    # which fires whenever jpdb's reading and spelling both differ from the
    # record's — measured, not assumed: a non-provisional record makes the same
    # three calls, so reconciliation is not what asks for it. What the suru
    # allowance skips is the *reading check's* own pinned re-ask on top of that,
    # redundant here because the allowance has already explained why 勉強's
    # entry spells differently from 勉強する. Removing the guard buys a second
    # request whose only possible answer is the one already in hand, and if it
    # fails to resolve the record is refused and nothing is written — the
    # symptom the allowance exists to prevent.
    forced = [body for endpoint, body in api.bodies
              if endpoint == "parse" and "furigana" in body]
    assert len(forced) == 1, "the reading check does not re-ask for a suru compound"


def test_a_record_without_a_reading_holds_reconciliation() -> None:
    # Empty fields still fill from the lemma parse, as they always have, but a
    # provisional claim is never settled against an unconfirmed identity.
    api = hanasu_api()
    unconfirmed = provisional_record(id="word:話す:", reading="")

    result = enrich_records(client_for(api), [unconfirmed])

    [updated] = result.records
    assert updated.meanings == ["to converse"]
    assert updated.part_of_speech == "noun"
    assert provisional_fields(updated) == ["meanings", "part_of_speech"]
    assert any("stay provisional" in warning for warning in result.warnings)


def test_a_record_with_nothing_to_fill_never_reaches_the_network() -> None:
    api = hanasu_api()
    full = record(
        furigana="話[はな]す",
        romaji="hanasu",
        part_of_speech="verb",
        verb_group="godan",
        transitivity="transitive",
        conjugations={"polite": "話します"},
        pitch_accent=["LHLL"],
        frequency_rank=200,
    )

    result = enrich_records(client_for(api), [full])

    assert api.bodies == []
    assert result.skipped == 1
    assert result.looked_up == 0
    assert result.changes == {}


def test_a_field_jpdb_has_no_value_for_is_left_empty_not_blanked() -> None:
    api = FakeApi(
        {"本": parse_response(([["本", "ほん"]], vocab(1, 2, "本", "ほん", [], 0, ["n"])))}
    )

    result = enrich_records(
        client_for(api), [record(id="word:本:ほん", expression="本", reading="ほん")]
    )

    enriched = result.records[0]
    assert enriched.pitch_accent == []
    assert enriched.verb_group == ""
    assert enriched.conjugations == {}
    assert enriched.frequency_rank == 0


# --- --force-fields ---------------------------------------------------------


def test_force_fields_overwrites_only_the_fields_it_names() -> None:
    api = hanasu_api()
    curated = record(part_of_speech="godan verb (u-verb)", pitch_accent=["HLLL"])

    result = enrich_records(
        client_for(api), [curated], force_fields=("pitch_accent",)
    )

    enriched = result.records[0]
    assert enriched.pitch_accent == ["LHLL"]
    assert enriched.part_of_speech == "godan verb (u-verb)"


def test_force_fields_refuses_reading_by_name() -> None:
    with pytest.raises(EnrichError) as excinfo:
        parse_force_fields("furigana,reading")

    assert "record ID" in str(excinfo.value)


@pytest.mark.parametrize("value", ["", None])
def test_force_fields_absent_is_no_fields(value: str | None) -> None:
    assert parse_force_fields(value) == ()


def test_force_fields_of_separators_only_is_an_error_not_a_shrug() -> None:
    with pytest.raises(EnrichError):
        parse_force_fields(",,")


def test_force_fields_rejects_an_unknown_field() -> None:
    with pytest.raises(EnrichError) as excinfo:
        parse_force_fields("furrigana")

    assert "furigana" in str(excinfo.value)


def test_forcing_a_field_jpdb_has_nothing_for_does_not_blank_it() -> None:
    api = FakeApi(
        {"本": parse_response(([["本", "ほん"]], vocab(1, 2, "本", "ほん", [], 0, ["n"])))}
    )
    curated = record(
        id="word:本:ほん", expression="本", reading="ほん", pitch_accent=["LHH"]
    )

    result = enrich_records(client_for(api), [curated], force_fields=("pitch_accent",))

    assert result.records[0].pitch_accent == ["LHH"]


# --- readings ---------------------------------------------------------------


def homograph_api() -> FakeApi:
    """一日 parses as いちにち unless told otherwise; ついたち is its other sense."""
    return FakeApi(
        unforced={"一日": parse_response((ICHINICHI_FURIGANA, ICHINICHI))},
        forced={("一日", "ついたち"): parse_response((TSUITACHI_FURIGANA, TSUITACHI))},
        senses={
            (1579110, 111): {"reading": "いちにち", "alt_sids": [222]},
            (1579110, 222): {"reading": "ついたち", "alt_sids": [111]},
        },
    )


def test_a_homograph_is_re_parsed_with_the_stored_reading_forced() -> None:
    api = homograph_api()
    tsuitachi = record(id="word:一日:ついたち", expression="一日", reading="ついたち")

    result = enrich_records(client_for(api), [tsuitachi])

    parses = api.calls("parse")
    assert len(parses) == 2
    assert "furigana" not in parses[0]
    # The whole-expression span, in the encoding the same request declares.
    assert parses[1]["furigana"] == [[[0, 2, "ついたち"]]]
    assert parses[1]["position_length_encoding"] == "utf16"
    enriched = result.records[0]
    assert enriched.furigana == "一日[ついたち]"
    assert enriched.pitch_accent == []
    assert enriched.frequency_rank == 1400
    assert len(result.warnings) == 1
    # The reason, not a generic "does not fit": this one really is a length
    # mismatch, and the sibling test below covers the pattern that fits the
    # reading exactly and still cannot be spoken.
    assert "cannot be used for reading ついたち" in result.warnings[0]
    assert "expected 5 characters" in result.warnings[0]


def test_same_spelling_homograph_uses_the_entry_selected_by_stored_reading() -> None:
    """The pinned parse may resolve to a different vid, not an alternate sense.

    分 is a same-spelling homograph whose ぶん and ふん entries have disjoint
    jpdb identities and reading sets.  The unforced entry therefore cannot
    advertise the stored reading through ``alt_sids``; only a parse pinned to
    that reading can find the dictionary facts for this record.
    """
    bun = vocab(100, 1, "分", "ぶん", ["LH"], 100, ["n"])
    fun = vocab(200, 2, "分", "ふん", ["HLL"], 1000, ["n"])
    api = FakeApi(
        unforced={"分": parse_response(([["分", "ぶん"]], bun))},
        forced={("分", "ふん"): parse_response(([["分", "ふん"]], fun))},
        senses={
            (100, 1): {"reading": "ぶん", "alt_sids": []},
            (200, 2): {"reading": "ふん", "alt_sids": []},
        },
    )
    item = record(id="word:分:ふん", expression="分", reading="ふん")

    result = enrich_records(client_for(api), [item])

    parses = api.calls("parse")
    assert len(parses) == 2
    assert "furigana" not in parses[0]
    assert parses[1]["furigana"] == [[[0, 1, "ふん"]]]
    enriched = result.records[0]
    assert enriched.pitch_accent == ["HLL"]
    assert enriched.frequency_rank == 1000
    assert result.warnings == []


def test_pinned_entrys_alternate_reading_can_confirm_the_stored_reading() -> None:
    """The pinned entry may declare the requested reading on another sense."""
    unforced = vocab(300, 1, "分", "わけ", ["LHL"], 300, ["n"])
    pinned = vocab(400, 10, "分", "ぶん", ["HLL"], 1000, ["n"])
    api = FakeApi(
        unforced={"分": parse_response(([["分", "わけ"]], unforced))},
        forced={("分", "ふん"): parse_response(([["分", "ふん"]], pinned))},
        senses={
            (300, 1): {"reading": "わけ", "alt_sids": []},
            (400, 10): {"reading": "ぶん", "alt_sids": [11]},
            (400, 11): {"reading": "ふん", "alt_sids": []},
        },
    )
    item = record(id="word:分:ふん", expression="分", reading="ふん")

    result = enrich_records(client_for(api), [item])

    enriched = result.records[0]
    assert enriched.furigana == "分[ふん]"
    assert enriched.pitch_accent == ["HLL"]
    assert enriched.frequency_rank == 1000
    assert result.warnings == []
    parses = api.calls("parse")
    assert len(parses) == 2
    assert parses[1]["furigana"] == [[[0, 1, "ふん"]]]


def test_failed_pinned_parse_does_not_fall_back_to_the_unforced_entry() -> None:
    """A known alternate reading cannot make an unresolved pinned parse usable."""
    api = FakeApi(
        unforced={"一日": parse_response((ICHINICHI_FURIGANA, ICHINICHI))},
        forced={("一日", "ついたち"): parse_response()},
        senses={
            (1579110, 111): {"reading": "いちにち", "alt_sids": [222]},
            (1579110, 222): {"reading": "ついたち", "alt_sids": [111]},
        },
    )
    item = record(id="word:一日:ついたち", expression="一日", reading="ついたち")

    result = enrich_records(client_for(api), [item])

    assert result.records[0] == item
    assert result.changes == {}
    assert len(result.warnings) == 1
    assert "when given the reading ついたち" in result.warnings[0]
    parses = api.calls("parse")
    assert len(parses) == 2
    assert parses[1]["furigana"] == [[[0, 2, "ついたち"]]]


def test_a_right_length_pattern_that_cannot_be_spoken_says_why() -> None:
    """Fitting the reading is not the whole test — rendering it is.

    かんーぱい is five kana, so a usable pattern is six characters, and
    `LHHHHH` is six: the length check passes and `to_aquestalk` still refuses,
    because `ン` ends on no vowel for the `ー` to repeat.

    The refusal itself is not new — `to_aquestalk` has behaved this way since
    the long-vowel work, and `test_a_long_vowel_with_no_vowel_to_repeat_is_
    refused` pins it directly. What is pinned here is the *message*: one
    warning covered both refusal shapes and told the curator the pattern did
    not fit the reading, which for this shape is false and sends them to check
    the one thing that is already right.

    And the consequence is larger than the audio: an unusable pattern is not
    written, so a record with no other accent gets no pitch diagram either.
    """
    kanpai = vocab(1000000, 555, "かんーぱい", "かんーぱい", ["LHHHHH"], 900, ["n"])
    api = FakeApi(unforced={"かんーぱい": parse_response((None, kanpai))})
    target = record(
        id="word:かんーぱい:かんーぱい", expression="かんーぱい", reading="かんーぱい"
    )

    result = enrich_records(client_for(api), [target])

    enriched = result.records[0]
    assert enriched.pitch_accent == [], "not written, so no diagram either"
    [warning] = result.warnings
    assert "LHHHHH" in warning
    assert "has no vowel to repeat" in warning, "the real reason"
    assert "expected" not in warning, "not the length — the length is correct"


def test_a_pattern_janki_can_use_survives_one_it_cannot() -> None:
    """A mixed list is not all-or-nothing, and the warning says so.

    `_compatible_pitch_patterns` splits rather than rejecting, so jpdb offering
    two patterns where one renders writes that one. The warning used to end
    "pitch accent was not written" in every case, including this one, where an
    accent *was* written.
    """
    hashi = vocab(1000001, 556, "はし", "はし", ["LHL", "LHLL"], 500, ["n"])
    api = FakeApi(unforced={"はし": parse_response((None, hashi))})
    target = record(id="word:はし:はし", expression="はし", reading="はし")

    result = enrich_records(client_for(api), [target])

    assert result.records[0].pitch_accent == ["LHL"], "the usable one was kept"
    [warning] = result.warnings
    assert "LHLL" in warning
    assert "the 1 that can was written" in warning
    assert "no usable pattern" not in warning


def test_the_warning_does_not_claim_a_write_that_did_not_happen() -> None:
    """`_wanted` decides, and it does not overwrite an accent already on file.

    The warning was emitted before `_apply` ran, so it reported the *split* —
    "kept the 1 that can" — as though the split were the outcome. For a record
    that already carries its own accent the split is discarded: jpdb fills
    empty fields, and a curator reading that line would believe jpdb's pattern
    had replaced the one they are looking at.
    """
    hashi = vocab(1000001, 556, "はし", "はし", ["LHL", "LHLL"], 500, ["n"])
    api = FakeApi(unforced={"はし": parse_response((None, hashi))})
    target = record(
        id="word:はし:はし", expression="はし", reading="はし", pitch_accent=["HLL"]
    )

    result = enrich_records(client_for(api), [target])

    assert result.records[0].pitch_accent == ["HLL"], "the record kept its own"
    [warning] = result.warnings
    assert "were not written" in warning or "not written" in warning
    assert "was written" not in warning, "nothing was"


def test_a_forced_rerun_that_lands_the_same_value_is_not_called_an_overwrite() -> None:
    """"Not written" covers two different things, and only one is a refusal.

    `_apply` records no change when the new value equals the old, so
    `--force-fields pitch_accent` on a record whose accent jpdb agrees with
    produces an empty diff — which the warning read as "jpdb does not overwrite
    one". The field was forced; jpdb simply had nothing different to say.
    """
    hashi = vocab(1000003, 558, "はし", "はし", ["LHL", "LHLL"], 500, ["n"])
    api = FakeApi(unforced={"はし": parse_response((None, hashi))})
    target = record(
        id="word:はし:はし", expression="はし", reading="はし", pitch_accent=["LHL"]
    )

    result = enrich_records(client_for(api), [target], force_fields=["pitch_accent"])

    [warning] = result.warnings
    assert "what the record already carried" in warning
    assert "does not overwrite" not in warning, "it was forced, not declined"


def test_each_unusable_pattern_gets_its_own_reason() -> None:
    """Two patterns, two different problems, and one of them named twice.

    `['LHHHHH', 'LHH']` against かんーぱい is a long-vowel refusal and a length
    refusal. Reporting the first one's reason for both is the same defect as
    reporting "does not fit the reading" for a pattern that fits it.
    """
    kanpai = vocab(1000002, 557, "かんーぱい", "かんーぱい", ["LHHHHH", "LHH"], 900, ["n"])
    api = FakeApi(unforced={"かんーぱい": parse_response((None, kanpai))})
    target = record(
        id="word:かんーぱい:かんーぱい", expression="かんーぱい", reading="かんーぱい"
    )

    result = enrich_records(client_for(api), [target])

    [warning] = result.warnings
    assert "LHHHHH ('ン' has no vowel to repeat)" in warning
    assert "LHH (expected 6 characters" in warning


def test_an_unrelated_kana_parse_retries_with_the_stored_reading() -> None:
    suru = vocab(1157170, 460825390, "する", "する", ["LHH"], 100, ["vs"])
    shinu = vocab(1310730, 851331686, "しぬ", "しぬ", ["LHH"], 30800, ["v5n"])
    api = FakeApi(
        unforced={"しぬ": parse_response((None, suru))},
        forced={("しぬ", "しぬ"): parse_response((None, shinu))},
    )
    item = record(id="word:しぬ:しぬ", expression="しぬ", reading="しぬ")

    result = enrich_records(client_for(api), [item])

    assert result.records[0].frequency_rank == 30800
    assert result.warnings == []
    parses = api.calls("parse")
    assert len(parses) == 2
    assert "furigana" not in parses[0]
    assert parses[1]["furigana"] == [[[0, 2, "しぬ"]]]


def test_the_reading_set_is_gathered_across_the_entrys_other_senses() -> None:
    api = homograph_api()
    tsuitachi = record(id="word:一日:ついたち", expression="一日", reading="ついたち")

    enrich_records(client_for(api), [tsuitachi])

    lookups = api.calls("lookup-vocabulary")
    assert lookups[0]["fields"] == ["reading", "alt_sids"]
    assert lookups[0]["list"] == [[1579110, 111]]
    # The alternate sense is asked about by vid, which is how the other reading
    # is found at all — it is not on the sense /parse picked.
    assert lookups[1]["list"] == [[1579110, 222]]


def test_a_reading_jpdb_does_not_list_is_warned_and_nothing_is_written() -> None:
    api = FakeApi(
        unforced={"話す": parse_response((HANASU_FURIGANA, HANASU))},
        # A forced request is only a request hint. The response must still
        # declare the stored reading before any of its facts are trusted.
        forced={("話す", "はなし"): parse_response((HANASU_FURIGANA, HANASU))},
        senses={(1562350, 4280520068): {"reading": "はなす", "alt_sids": []}},
    )
    typo = record(id="word:話す:はなし", reading="はなし")

    result = enrich_records(client_for(api), [typo])

    assert result.changes == {}
    assert result.records[0] == typo
    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert "はなし" in warning and "はなす" in warning
    assert "never auto-fixed" in warning
    parses = api.calls("parse")
    assert len(parses) == 2
    assert parses[1]["furigana"] == [[[0, 2, "はなし"]]]


def test_a_record_with_no_reading_at_all_is_enriched_from_the_first_parse() -> None:
    # Post-M1.5 no *imported* record has an empty reading; a hand-written one
    # can. There is nothing to compare, so the unforced answer is used as-is.
    api = hanasu_api()
    handwritten = VocabularyRecord(id="word:話す:", expression="話す", reading="")

    result = enrich_records(client_for(api), [handwritten])

    assert result.records[0].furigana == "話[はな]す"
    assert result.records[0].reading == ""
    # No reading means no romaji — the transliteration has nothing to read from.
    # The conjugation table does not need one: it inflects the expression, and
    # the reading only ever guards against the two disagreeing.
    assert result.records[0].romaji == ""
    assert result.records[0].conjugations["negative"] == "話さない"
    assert len(api.calls("parse")) == 1


def test_an_expression_jpdb_splits_into_several_words_is_warned() -> None:
    api = FakeApi(
        {
            "食べ物屋": parse_response(
                (
                    [["食", "た"], "べ", ["物", "もの"]],
                    vocab(1, 1, "食べ物", "たべもの", [], 2600, ["n"]),
                ),
                ([["屋", "や"]], vocab(2, 2, "屋", "や", [], 5000, ["suf"])),
            )
        }
    )
    compound = record(id="word:食べ物屋:たべものや", expression="食べ物屋", reading="たべものや")

    result = enrich_records(client_for(api), [compound])

    assert result.changes == {}
    assert "one word" in result.warnings[0]


def test_one_token_matching_the_whole_expression_wins_over_its_neighbours() -> None:
    # jpdb parses a trailing particle as its own token; the entry whose
    # spelling *is* the expression is still unambiguous.
    api = FakeApi(
        {
            "話す": parse_response(
                (HANASU_FURIGANA, HANASU),
                (None, vocab(2029010, 1, "を", "を", ["HL"], 100, ["prt"])),
            )
        }
    )

    result = enrich_records(client_for(api), [record()])

    assert result.records[0].frequency_rank == 200


# --- target selection -------------------------------------------------------


def test_only_the_named_ids_are_enriched() -> None:
    api = hanasu_api()
    other = record(id="word:本:ほん", expression="本", reading="ほん")

    result = enrich_records(client_for(api), [record(), other], ids=["word:話す:はなす"])

    assert set(result.changes) == {"word:話す:はなす"}
    assert result.records[1] == other


def test_an_unknown_id_is_an_error_before_any_api_call() -> None:
    api = hanasu_api()

    with pytest.raises(EnrichError) as excinfo:
        enrich_records(client_for(api), [record()], ids=["word:無い:ない"])

    assert "word:無い:ない" in str(excinfo.value)
    assert api.bodies == []


# --- the field diff ---------------------------------------------------------


def test_the_field_diff_is_a_record_header_and_one_line_per_field() -> None:
    lines = format_field_diff(
        {
            "word:話す:はなす": {
                "furigana": ("", "話[はな]す"),
                "frequency_rank": (None, 200),
            }
        }
    )

    assert lines == [
        "word:話す:はなす",
        "  furigana: (empty) -> 話[はな]す",
        "  frequency_rank: (none) -> 200",
    ]


# --- staging ----------------------------------------------------------------


def held(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:話す:",
        "expression": "話す",
        "reading": "",
        "source": SourceReference(
            type="shirabe",
            imported_from="export.csv",
            raw_fields={"hold_reason": "missing reading"},
        ),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def test_a_held_row_gets_the_reading_jpdb_proposes_as_an_annotation() -> None:
    api = hanasu_api()

    result = suggest_readings(client_for(api), [held()])

    annotated = result.records[0]
    assert annotated.source.raw_fields["suggested_reading"] == "はなす"
    # A proposal, not a fix: the row stays held until a human types it in.
    assert annotated.reading == ""
    assert annotated.source.raw_fields["hold_reason"] == "missing reading"
    assert result.suggested == {"word:話す:": "はなす"}
    assert result.held == 1
    assert result.warnings == []


def test_the_second_hold_class_is_annotated_too() -> None:
    # `reading contains kanji` mints word:<kanji>:<kanji> — it has a reading,
    # so selecting held rows on emptiness alone would skip it.
    api = hanasu_api()
    kanji_reading = held(
        id="word:話す:話す",
        reading="話す",
        source=SourceReference(raw_fields={"hold_reason": "reading contains kanji"}),
    )

    result = suggest_readings(client_for(api), [kanji_reading])

    assert result.records[0].source.raw_fields["suggested_reading"] == "はなす"
    assert result.held == 1


def test_a_row_that_is_not_held_is_left_alone() -> None:
    api = hanasu_api()

    result = suggest_readings(client_for(api), [record()])

    assert result.records[0] == record()
    assert result.held == 0
    assert api.bodies == []


# --- the CLI ----------------------------------------------------------------


def project(tmp_path: Path, records: list[VocabularyRecord]) -> Path:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    (tmp_path / "vocabulary.json").write_text(
        json.dumps([item.to_dict() for item in records], ensure_ascii=False),
        encoding="utf-8",
    )
    return tmp_path


def patch_api(monkeypatch: pytest.MonkeyPatch, api: FakeApi) -> None:
    monkeypatch.setenv("JPDB_API_KEY", "test-key")
    monkeypatch.setattr(cli.jpdb, "JpdbClient", lambda key, *a, **kw: client_for(api))


def stored(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload}


def test_enrich_without_a_source_says_so_rather_than_guessing() -> None:
    assert cli.main(["enrich"]) == 1


def test_enrich_takes_one_source_at_a_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Each pass shows its own diff; running both at once would merge two
    # unrelated sets of proposals into one y/n.
    assert cli.main(["enrich", "--ai", "--jpdb"]) == 1

    assert "one pass at a time" in capsys.readouterr().err


def test_enrich_jpdb_writes_the_records_the_diff_and_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [record()])
    patch_api(monkeypatch, hanasu_api())

    assert cli.main(["--root", str(root), "enrich", "--jpdb", "--yes"]) == 0

    out = capsys.readouterr().out
    assert "word:話す:はなす" in out
    assert "furigana: (empty) -> 話[はな]す" in out
    assert "Enriched 1 record(s)" in out
    assert stored(root)["word:話す:はなす"]["furigana"] == "話[はな]す"

    book = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    passes = book["records"]["word:話す:はなす"]["enriched"]
    assert [item["kind"] for item in passes] == ["jpdb"]
    assert "furigana" in passes[0]["fields"]


def test_enrich_jpdb_writes_nothing_when_there_is_nothing_to_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = hanasu_api()
    root = project(tmp_path, [record()])
    patch_api(monkeypatch, api)
    cli.main(["--root", str(root), "enrich", "--jpdb", "--yes"])
    before = (root / "vocabulary.json").read_text(encoding="utf-8")
    capsys.readouterr()

    assert cli.main(["--root", str(root), "enrich", "--jpdb", "--yes"]) == 0

    assert "Nothing to fill" in capsys.readouterr().out
    assert (root / "vocabulary.json").read_text(encoding="utf-8") == before


def test_a_marks_only_run_still_saves_the_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Nothing to fill, but a stale mark to clear: the run must persist the
    # clear or it repeats the same warning — and the same lookups — forever.
    marked = mark_provisional(
        record(
            source=SourceReference(type="extract", imported_from="page.jpg"),
            meanings=["a model gloss"],
            part_of_speech="verb",
            furigana="話[はな]す",
            romaji="hanasu",
            verb_group="godan",
            transitivity="transitive",
            conjugations={"plain": "話す"},
            pitch_accent=["LHLL"],
            frequency_rank=200,
        )
    )
    # Both bound fields edited, so both marks are stale, nothing is left to
    # reconcile, and the run never reaches the network.
    edited = replace(
        marked,
        meanings=["to speak (hand-checked)"],
        part_of_speech="verb (hand-checked)",
    )
    root = project(tmp_path, [edited])
    patch_api(monkeypatch, FakeApi({}))

    assert cli.main(["--root", str(root), "enrich", "--jpdb", "--yes"]) == 0

    out = capsys.readouterr().out
    assert "provisional-mark update(s)" in out
    raw_fields = stored(root)["word:話す:はなす"]["source"]["raw_fields"]
    assert "provisional_fields" not in raw_fields


def test_a_marks_only_run_still_asks_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Bookkeeping or not, the run rewrites vocabulary.json — and every write
    # this command makes goes through the same y/N.
    marked = mark_provisional(
        record(
            source=SourceReference(type="extract", imported_from="page.jpg"),
            meanings=["a model gloss"],
            part_of_speech="verb",
            furigana="話[はな]す",
            romaji="hanasu",
            verb_group="godan",
            transitivity="transitive",
            conjugations={"plain": "話す"},
            pitch_accent=["LHLL"],
            frequency_rank=200,
        )
    )
    edited = replace(
        marked,
        meanings=["to speak (hand-checked)"],
        part_of_speech="verb (hand-checked)",
    )
    root = project(tmp_path, [edited])
    patch_api(monkeypatch, FakeApi({}))
    before = (root / "vocabulary.json").read_text(encoding="utf-8")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    assert cli.main(["--root", str(root), "enrich", "--jpdb"]) == 1

    assert (root / "vocabulary.json").read_text(encoding="utf-8") == before


def test_a_declined_confirmation_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [record()])
    patch_api(monkeypatch, hanasu_api())
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    assert cli.main(["--root", str(root), "enrich", "--jpdb"]) == 1

    assert stored(root)["word:話す:はなす"]["furigana"] == ""
    assert not (root / "ledger.json").exists()


def test_enrich_staging_annotates_the_file_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    staging = root / "staging" / "shirabe-export-needs-reading.yaml"
    staging.parent.mkdir(parents=True)
    staging.write_text(
        yaml.safe_dump(
            {"records": [held().to_dict()], "source_file": "export.csv"},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    patch_api(monkeypatch, hanasu_api())

    code = cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--staging", str(staging), "--yes"]
    )

    assert code == 0
    written = yaml.safe_load(staging.read_text(encoding="utf-8"))
    assert written["records"][0]["source"]["raw_fields"]["suggested_reading"] == "はなす"
    # Metadata the reviewer's file carried survives the rewrite.
    assert written["source_file"] == "export.csv"
    assert "Suggested a reading for 1 of 1 held row(s)" in capsys.readouterr().out


def test_enrich_staging_takes_neither_force_fields_nor_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = project(tmp_path, [])
    patch_api(monkeypatch, hanasu_api())

    code = cli.main(
        [
            "--root",
            str(root),
            "enrich",
            "--jpdb",
            "--staging",
            str(root / "any.yaml"),
            "--force-fields",
            "furigana",
        ]
    )

    assert code == 1
    assert "--force-fields" in capsys.readouterr().err


def test_enrich_reports_a_reading_mismatch_on_stderr_and_still_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeApi(
        unforced={"話す": parse_response((HANASU_FURIGANA, HANASU))},
        forced={("話す", "はなし"): parse_response((HANASU_FURIGANA, HANASU))},
        senses={(1562350, 4280520068): {"reading": "はなす", "alt_sids": []}},
    )
    root = project(tmp_path, [record(id="word:話す:はなし", reading="はなし")])
    patch_api(monkeypatch, api)

    code = cli.main(["--root", str(root), "enrich", "--jpdb", "--yes"])

    captured = capsys.readouterr()
    assert code == 0
    assert "はなし" in captured.err
    assert "Nothing to fill" in captured.out
    assert stored(root)["word:話す:はなし"]["furigana"] == ""


def test_enrich_can_be_pointed_at_single_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = hanasu_api()
    other = record(id="word:本:ほん", expression="本", reading="ほん")
    root = project(tmp_path, [record(), other])
    patch_api(monkeypatch, api)

    code = cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--yes", "word:話す:はなす"]
    )

    assert code == 0
    assert stored(root)["word:本:ほん"]["furigana"] == ""
    assert stored(root)["word:話す:はなす"]["furigana"] == "話[はな]す"


def test_enrichable_fields_can_never_include_the_reading() -> None:
    # A guard, not a tautology: this tuple is what --force-fields validates
    # against, so an addition to it is an addition to what may be overwritten.
    assert "reading" not in enrich.ENRICHABLE_FIELDS
    assert "id" not in enrich.ENRICHABLE_FIELDS


# --- the lemma trap ---------------------------------------------------------


# 行った parses to one token whose entry is the *lemma* 行く. The token's
# furigana describes the surface form; the entry's reading does not.
ITTA_FURIGANA = [["行", "い"], "った"]
IKU = vocab(1578850, 777, "行く", "いく", ["LHL"], 300, ["vi", "v5k-s"])


def itta_api() -> FakeApi:
    return FakeApi(
        unforced={"行った": parse_response((ITTA_FURIGANA, IKU))},
        senses={(1578850, 777): {"reading": "いく", "alt_sids": []}},
    )


def test_a_held_inflected_form_is_suggested_its_own_reading_not_the_lemmas() -> None:
    # The bug this guards: suggesting いく for 行った would have a reviewer mint
    # word:行った:いく permanently — the exact unrecoverable outcome the staging
    # review exists to prevent.
    api = itta_api()
    row = held(id="word:行った:", expression="行った", reading="")

    result = suggest_readings(client_for(api), [row])

    assert result.suggested == {"word:行った:": "いった"}
    assert result.records[0].source.raw_fields["suggested_reading"] == "いった"


def test_an_inflected_form_with_no_reading_is_not_enriched_from_the_lemma() -> None:
    # Same trap in the main pass: 行く's pitch accent and frequency rank
    # describe a different word, and this record has no reading to prove the
    # entry is the right one.
    api = itta_api()
    handwritten = VocabularyRecord(id="word:行った:", expression="行った", reading="")

    result = enrich_records(client_for(api), [handwritten])

    assert result.changes == {}
    assert result.records[0].pitch_accent == []
    assert "no reading to confirm" in result.warnings[0]


def test_an_all_kana_expression_still_falls_back_to_the_entry_reading() -> None:
    # jpdb sends null furigana for an all-kana token, and there the entry is
    # the same surface form, so its reading can be trusted.
    api = FakeApi(
        {"たべる": parse_response((None, vocab(3, 4, "たべる", "たべる", [], 13600, ["v1"])))}
    )
    row = held(id="word:たべる:話す", expression="たべる", reading="話す")

    result = suggest_readings(client_for(api), [row])

    assert result.suggested == {"word:たべる:話す": "たべる"}


def test_furigana_to_reading_spells_out_the_surface_form() -> None:
    assert jpdb.furigana_to_reading(ITTA_FURIGANA) == "いった"
    assert jpdb.furigana_to_reading(HANASU_FURIGANA) == "はなす"
    assert jpdb.furigana_to_reading([["日", "にっ"], ["本", "ぽん"], ["語", "ご"]]) == "にっぽんご"
    assert jpdb.furigana_to_reading(None) == ""


# --- the staging file is a reviewer's working document ----------------------


STAGING_WITH_REVIEW_NOTES = """\
# Held out of shirabe-export.csv on 2026-08-07 — check these with my teacher.
source_file: export.csv
records:
  # not sure this one is even worth a card
  - id: 'word:話す:'
    expression: 話す
    reading: ''
    my_note: ask about the intransitive pair
    source:
      type: shirabe
      imported_from: export.csv
      raw_fields:
        hold_reason: missing reading
"""


def test_annotating_a_staging_file_keeps_the_reviewers_comments_and_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project(tmp_path, [])
    staging = root / "review.yaml"
    staging.write_text(STAGING_WITH_REVIEW_NOTES, encoding="utf-8")
    patch_api(monkeypatch, hanasu_api())

    code = cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--staging", str(staging), "--yes"]
    )

    assert code == 0
    text = staging.read_text(encoding="utf-8")
    # The annotation landed...
    assert "suggested_reading: はなす" in text
    # ...and nothing the reviewer wrote was rewritten out from under them.
    assert "# Held out of shirabe-export.csv on 2026-08-07" in text
    assert "# not sure this one is even worth a card" in text
    assert "my_note: ask about the intransitive pair" in text
    written = yaml.safe_load(text)
    assert written["records"][0]["my_note"] == "ask about the intransitive pair"
    assert written["source_file"] == "export.csv"


def test_annotating_adds_no_empty_schema_fields_the_file_left_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A row a human wrote by hand stays as short as they wrote it.
    root = project(tmp_path, [])
    staging = root / "review.yaml"
    staging.write_text(STAGING_WITH_REVIEW_NOTES, encoding="utf-8")
    patch_api(monkeypatch, hanasu_api())

    cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--staging", str(staging), "--yes"]
    )

    row = yaml.safe_load(staging.read_text(encoding="utf-8"))["records"][0]
    assert "audio" not in row
    assert "examples" not in row
    assert "pitch_accent" not in row


def test_a_row_jpdb_could_not_help_with_is_reported_before_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # One row jpdb helps with and one it does not: the prompt asks about the
    # first, and answering "no" must not be the reason the second was never
    # mentioned. Pre-fix the warnings printed after the gate, so a declined
    # run discarded them entirely.
    root = project(tmp_path, [])
    staging = root / "review.yaml"
    staging.write_text(
        yaml.safe_dump(
            {
                "records": [
                    held().to_dict(),
                    held(id="word:謎:", expression="謎", reading="").to_dict(),
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    api = FakeApi(
        {"話す": parse_response((HANASU_FURIGANA, HANASU)), "謎": parse_response()}
    )
    patch_api(monkeypatch, api)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    code = cli.main(["--root", str(root), "enrich", "--jpdb", "--staging", str(staging)])

    captured = capsys.readouterr()
    assert code == 1
    assert "one word" in captured.err
    # Declined, so the file is untouched.
    assert "suggested_reading" not in staging.read_text(encoding="utf-8")


STAGING_WITH_YAML_11_ISMS = """\
records:
  - id: 'word:話す:'
    expression: 話す
    reading: ''
    source:
      type: shirabe
      raw_fields:
        hold_reason: missing reading
        already_known: yes
        checked_at: 12:30
"""


def test_annotating_does_not_rewrite_values_in_the_reviewers_own_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The two YAML dialects disagree about these: 1.1 reads `yes` as a boolean
    # and `12:30` as the sexagesimal integer 750, 1.2 reads both as strings.
    # Diffing one dialect's reading against the other's calls them "changed"
    # and writes janki's version over the reviewer's line.
    root = project(tmp_path, [])
    staging = root / "review.yaml"
    staging.write_text(STAGING_WITH_YAML_11_ISMS, encoding="utf-8")
    patch_api(monkeypatch, hanasu_api())

    code = cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--staging", str(staging), "--yes"]
    )

    assert code == 0
    text = staging.read_text(encoding="utf-8")
    assert "suggested_reading: はなす" in text
    assert "already_known: yes" in text
    assert "checked_at: 12:30" in text
    assert "'True'" not in text and "'750'" not in text


def test_a_file_that_cannot_be_rewritten_fails_before_the_api_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A duplicate key is an easy slip while hand-editing. PyYAML accepts it
    # silently, so the read and the whole pass would succeed and the write
    # would then fail — after the user had already confirmed it.
    root = project(tmp_path, [])
    staging = root / "review.yaml"
    staging.write_text(
        STAGING_WITH_YAML_11_ISMS.replace(
            "    expression: 話す", "    expression: 話す\n    expression: 話す"
        ),
        encoding="utf-8",
    )
    api = hanasu_api()
    patch_api(monkeypatch, api)

    code = cli.main(
        ["--root", str(root), "enrich", "--jpdb", "--staging", str(staging), "--yes"]
    )

    assert code == 1
    assert "review.yaml" in capsys.readouterr().err
    # Nothing was spent, and nothing was touched.
    assert api.bodies == []
    assert "suggested_reading" not in staging.read_text(encoding="utf-8")


def test_a_field_jpdb_has_no_answer_for_is_looked_up_again_next_run() -> None:
    # Not a bug, but the README makes a cost claim about it: a noun has no verb
    # group and no conjugation table, so those stay empty and the record stays
    # fillable. Nothing records "asked, and there was nothing".
    api = FakeApi(
        {"本": parse_response(([["本", "ほん"]], vocab(1, 2, "本", "ほん", ["LH"], 500, ["n"])))}
    )
    noun = VocabularyRecord(id="word:本:ほん", expression="本", reading="ほん")

    first = enrich_records(client_for(api), [noun])
    calls = len(api.bodies)
    second = enrich_records(client_for(api), first.records)

    assert second.looked_up == 1
    assert second.changes == {}
    assert len(api.bodies) == calls + 1


def test_a_failed_ledger_write_says_a_re_run_would_skip_these_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A record jpdb filled has nothing left it can fill, so a re-run reaches
    nothing whether it skips the record (the dictionary had every field) or
    looks it up again and proposes nothing (a noun, whose conjugations never
    fill). The message covers both, and is not the polish one."""
    root = project(tmp_path, [record()])
    patch_api(monkeypatch, hanasu_api())
    monkeypatch.setattr(
        cli.ledger.Ledger,
        "save",
        lambda self: (_ for _ in ()).throw(cli.ledger.LedgerError("disk full")),
    )

    assert cli.main(["--root", str(root), "enrich", "--jpdb", "--yes"]) == 1

    err = capsys.readouterr().err
    assert "'status --rebuild' cannot bring it back" in err
    assert "either skips these records or looks them up and proposes nothing" in err
    assert "not a free repair" not in err
    assert stored(root)["word:話す:はなす"]["furigana"] == "話[はな]す"


def test_the_suru_allowance_needs_the_suffix_on_each_side_separately() -> None:
    """The allowance exists for `Xする` and nothing else, and it is gated on the
    suffix on *both* sides before it compares stems.

    Checked on the predicate rather than through a deck, because the two
    conjuncts are only separable on inputs a real record never has: drop the
    expression's suffix and the comparison is against `[:-2]` of a word that
    does not end in する, which needs a three-character expression whose first
    character is a word in its own right. Driven through a deck, only the pair
    is observable — either one alone still refuses, so the test would pass
    against a build that had lost one of them.
    """
    from japanese_anki.enrich import _supports_suru_suffix

    def entry(spelling: str, reading: str, *codes: str) -> dict[str, Any]:
        return {
            "spelling": spelling,
            "reading": reading,
            "part_of_speech": list(codes) or ["n", "vs"],
        }

    assert _supports_suru_suffix(
        "勉強する", "べんきょうする", entry("勉強", "べんきょう")
    )

    # The reading does not end in する, so this identity is 勉強する read
    # べんきょう — a spelling and a reading that disagree about the verb. The
    # entry answers for `reading[:-2]`, so that without the suffix conjunct
    # every remaining one holds and the identity would pass.
    assert not _supports_suru_suffix("勉強する", "べんきょう", entry("勉強", "べんき"))

    # The expression does not end in する. 図書館 read としょかんする is not a
    # する compound however its stem is spelled.
    assert not _supports_suru_suffix(
        "図書館", "としょかんする", entry("図", "としょかん")
    )

    # する itself is not `Xする`: there is no X, and an entry answering for the
    # empty stem would otherwise satisfy every remaining conjunct.
    assert not _supports_suru_suffix("する", "する", entry("", ""))


def test_a_stored_part_of_speech_cannot_unlock_a_transitivity() -> None:
    """The gate reads jpdb's derived label, never the record's own.

    A stored part of speech is uncontrolled text — a source column copied
    verbatim by an importer, extraction's contextual phrasing, a hand edit —
    and an exact test for "noun" lets "pronoun", "proper noun" and
    "noun, suru-verb" past it. Twenty-two live records carry a near-miss label
    like those with an empty transitivity, so the next `enrich --jpdb` would
    have written the contradiction the rule exists to prevent.
    """
    shigoto = vocab(1330910, 4, "仕事", "しごと", ["LHH"], 300, ["n", "vs", "vi"])
    api = FakeApi({"仕事": parse_response(([["仕事", "しごと"]], shigoto))})

    result = enrich_records(
        client_for(api),
        [record(id="word:仕事:しごと", expression="仕事", reading="しごと",
                meanings=["work"], part_of_speech="noun, suru-verb")],
    )

    assert result.records[0].transitivity == ""


@pytest.mark.parametrize(
    "codes,derived,expected",
    [
        (["v5u", "vt"], "verb", "transitive"),
        (["v1", "vi"], "verb", "intransitive"),
        (["exp", "v5r", "vt"], "expression", "transitive"),
        (["n", "vs", "vi"], "noun", ""),
        (["n", "vs", "vt"], "noun", ""),
        (["aux-v", "vi", "suf", "vt", "vs"], "verb", ""),
    ],
    ids=["godan-vt", "ichidan-vi", "expression", "suru-noun", "suru-stem", "both-ways"],
)
def test_transitivity_for_states_only_what_cannot_contradict_the_label(
    codes: list[str], derived: str, expected: str
) -> None:
    """The rule itself, called directly.

    It shipped with only indirect coverage through `enrich_records`, which
    could not distinguish "the gate refused" from "the dictionary said
    nothing" — the two produce the same empty field. Here they are separate
    rows: `suru-noun` is the gate refusing a real `vi`, `both-ways` is
    `pos_to_transitivity` declining to pick between `vt` and `vi` on a word
    tagged both.
    """
    from japanese_anki import jpdb

    assert jpdb.pos_to_part_of_speech(codes) == derived, "the premise"
    assert jpdb.transitivity_for(codes, derived) == expected
