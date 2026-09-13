"""The projected, prepared and recoverable dictionary-enrichment pass.

Contracts §7.8. A study finish has to model this phase before the owner
confirms it — over a canonical projection that is not on disk yet — and then
apply exactly what was approved, from an intent that survives the process
dying between the canonical write and the ledger save.

Three seams, tested here as three things:

* ``plan_dictionary_enrichment_revision`` decides over supplied records, a
  supplied revision and an **explicitly frozen** reference store, and marks its
  decision ``projected``. ``commit_dictionary_enrichment`` refuses a projected
  decision at its first statement, so a chained canonical projection can never
  be replayed as a transaction.
* ``prepare_dictionary_enrichment`` freezes the canonical and ledger components
  — both payloads, both digests and the ledger's own date — and writes nothing.
* ``apply_prepared_dictionary_enrichment`` and
  ``recover_prepared_dictionary_enrichment`` precheck that whole vector under
  the canonical and ledger locks, finish a started intent from its frozen
  payloads, and re-plan an unstarted one through the replay client only.

Nothing here reaches the network: every client below answers from a script.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from japanese_anki import enrich, jpdb, kanji, ledger
from japanese_anki import io as io_module
from japanese_anki.application import enrichment as enrichment_application
from japanese_anki.application import study_curation
from japanese_anki.application.enrichment import (
    DictionaryEnrichmentError,
    commit_dictionary_enrichment,
    plan_all_dictionary_enrichment,
    plan_dictionary_enrichment,
    plan_dictionary_enrichment_revision,
    prepare_dictionary_enrichment,
    recover_prepared_dictionary_enrichment,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.enrich import (
    DictionaryFactBook,
    RecordingDictionaryClient,
    ReplayDictionaryClient,
)
from japanese_anki.models import (
    SourceFormColumn,
    SourceFormsTable,
    SourceReference,
    VocabularyRecord,
    mark_provisional,
)

HANASU = {
    "vid": 1562350,
    "sid": 4280520068,
    "spelling": "話す",
    "reading": "はなす",
    "pitch_accent": ["LHLL"],
    "frequency_rank": 200,
    "part_of_speech": ["vt", "v5", "v5s"],
}
HANASU_FURIGANA = [["話", "はな"], "す"]

ASHITA = {
    "vid": 1111111,
    "sid": 2222222,
    "spelling": "明日",
    "reading": "あした",
    "pitch_accent": ["LHH"],
    "frequency_rank": 50,
    "part_of_speech": ["n"],
}
ASHITA_FURIGANA = [["明", "あ"], ["日", "した"]]

# A noun: jpdb states no verb class, no transitivity and no conjugation table,
# so an enriched noun is still fillable and a re-plan over it really does reach
# the dictionary again. That is what makes "resume before re-plan" observable.
HON = {
    "vid": 1234500,
    "sid": 99,
    "spelling": "本",
    "reading": "ほん",
    "pitch_accent": ["LH"],
    "frequency_rank": 30,
    "part_of_speech": ["n"],
}
HON_FURIGANA = [["本", "ほん"]]

#: The day a preparation freezes. Deliberately not today's: a frozen date that
#: happens to equal the clock proves nothing about freezing.
FROZEN = date(2026, 4, 1)
FROZEN_ISO = "2026-04-01"
LATER = date(2026, 4, 2)


class ScriptedDictionary:
    """A `DictionaryLookup` answering from a script, counting every call."""

    def __init__(self, entries: dict[str, tuple[Any, list[Any]]] | None = None) -> None:
        self._entries = dict(entries or {})
        self.parse_calls: list[str] = []
        self.lookup_calls: list[Any] = []

    def parse(
        self,
        text: str,
        *,
        token_fields: Any = jpdb.DEFAULT_TOKEN_FIELDS,
        vocabulary_fields: Any = jpdb.DEFAULT_VOCABULARY_FIELDS,
        forced_furigana: Any = None,
        encoding: str = jpdb.DEFAULT_ENCODING,
    ) -> jpdb.ParseResult:
        self.parse_calls.append(text)
        found = self._entries.get(text)
        if found is None:
            return jpdb.ParseResult(tokens=[], vocabulary=[])
        furigana, entry = found
        return jpdb.ParseResult(
            tokens=[{"vocabulary_index": 0, "furigana": furigana}],
            vocabulary=[dict(entry)],
        )

    def lookup_vocabulary(
        self,
        pairs: Any,
        fields: Any = jpdb.DEFAULT_LOOKUP_FIELDS,
        *,
        batch_size: int = jpdb.DEFAULT_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        wanted = [list(pair) for pair in pairs]
        self.lookup_calls.append(wanted)
        return [
            {"vid": vid, "sid": sid, "reading": "はなす", "alt_sids": []}
            for vid, sid in wanted
        ]


def hanasu_client() -> ScriptedDictionary:
    return ScriptedDictionary({"話す": (HANASU_FURIGANA, HANASU)})


def noun_client() -> ScriptedDictionary:
    return ScriptedDictionary({"本": (HON_FURIGANA, HON)})


def noun_record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:本:ほん",
        "expression": "本",
        "reading": "ほん",
        "meanings": ["book"],
    }
    values.update(overrides)
    return record(**values)


def record(**overrides: Any) -> VocabularyRecord:
    values: dict[str, Any] = {
        "id": "word:話す:はなす",
        "expression": "話す",
        "reading": "はなす",
        "meanings": ["to speak"],
        "source": SourceReference(type="shirabe", imported_from="export.csv"),
    }
    values.update(overrides)
    return VocabularyRecord(**values)


def settled(**overrides: Any) -> VocabularyRecord:
    """A record a dictionary pass has nothing left to fill."""
    values: dict[str, Any] = {
        "furigana": "話[はな]す",
        "romaji": "hanasu",
        "part_of_speech": "verb",
        "verb_group": "godan",
        "transitivity": "transitive",
        "conjugations": {"negative": "話さない"},
        "pitch_accent": ["LHLL"],
        "frequency_rank": 200,
    }
    values.update(overrides)
    return record(**values)


def edited_record() -> VocabularyRecord:
    """A model claim a human edited afterwards: the mark is stale, not active.

    Enriching it clears that mark and writes no attribution row, which is the
    cleared-only shape — canonical changes and the ledger does not.
    """
    marked = mark_provisional(
        record(
            source=SourceReference(type="extract", imported_from="page.jpg"),
            meanings=["to converse"],
        )
    )
    return replace(marked, meanings=["to chat (hand-checked)"])


def project(tmp_path: Path, records: list[VocabularyRecord]) -> ProjectConfig:
    root = tmp_path / "repo"
    (root / "staging").mkdir(parents=True)
    (root / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'kanji_file = "kanji.json"\n'
        'staging_dir = "staging"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(root)
    io_module.save_records_json(config.normalized_file.resolve(), records)
    return config


def canonical_text(config: ProjectConfig) -> str:
    return config.normalized_file.resolve().read_text(encoding="utf-8")


def digest(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def stored(config: ProjectConfig) -> dict[str, dict[str, Any]]:
    payload = json.loads(canonical_text(config))
    return {item["id"]: item for item in payload}


def component(prepared: Any, role: str) -> Any:
    for item in prepared.components:
        if item.role == role:
            return item
    raise AssertionError(f"no {role} component")


def files_now(config: ProjectConfig) -> dict[str, bytes | None]:
    return {
        str(path): path.read_bytes() if path.exists() else None
        for path in (
            config.normalized_file.resolve(),
            config.ledger_file.resolve(),
            config.kanji_file.resolve(),
        )
    }


def guarded_apply(config: ProjectConfig, prepared: Any, **kwargs: Any) -> Any:
    """Apply the way a finish coordinator does: the guard outermost."""
    with study_curation.curation_guard(config):
        return enrichment_application.apply_prepared_dictionary_enrichment_under_guard(
            config, prepared, **kwargs
        )


def recorded_book(config: ProjectConfig, client: ScriptedDictionary) -> DictionaryFactBook:
    """Fetch every fact once, before the preview, through the recording client."""
    recorder = RecordingDictionaryClient(client)
    plan_dictionary_enrichment(config, recorder, ["word:話す:はなす"])
    return recorder.freeze()


# --- the projection over a collection that is not on disk ---------------------


def test_the_revision_planner_decides_over_the_records_and_store_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-promotion records, their revision, and an explicitly frozen
    reference store — never a read of the live canonical or kanji path."""
    config = project(tmp_path, [])
    client = ScriptedDictionary({"明日": (ASHITA_FURIGANA, ASHITA)})
    projected = record(id="word:明日:あした", expression="明日", reading="あした")
    revision = io_module.RecordsRevision(
        config.normalized_file.resolve(), io_module.records_json_text([projected])
    )
    frozen_store = kanji.KanjiStore(
        entries={
            "日": kanji.KanjiInfo(
                character="日",
                stroke_count=4,
                meanings=("day",),
                readings=(kanji.Reading(kind="kun", reading="ひ"),),
            )
        }
    )

    def refuse(_path: Any) -> kanji.KanjiStore:
        raise AssertionError("the revision planner must not read the live kanji store")

    def refuse_canonical(_path: Any) -> Any:
        raise AssertionError("the revision planner must not read live canonical")

    monkeypatch.setattr(kanji, "load_store", refuse)
    monkeypatch.setattr(
        enrichment_application, "load_records_snapshot", refuse_canonical
    )

    decision = plan_dictionary_enrichment_revision(
        config,
        client,
        [projected],
        revision,
        ["word:明日:あした"],
        kanji_store=frozen_store,
    )

    assert decision.projected is True
    assert decision.record_ids == ("word:明日:あした",)
    assert decision.output_revision is revision
    # した is no reading of 日, so the supplied store rejects jpdb's split and
    # the whole-word form is proposed instead. With no store it would stand.
    assert decision.result.changes["word:明日:あした"]["furigana"][1] == "明日[あした]"


def test_the_revision_planner_keeps_the_pre_lookup_ledger_refusal(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [])
    config.ledger_file.resolve().write_text("{not json", encoding="utf-8")
    client = hanasu_client()
    target = record()
    revision = io_module.RecordsRevision(
        config.normalized_file.resolve(), io_module.records_json_text([target])
    )

    with pytest.raises(ledger.LedgerError):
        plan_dictionary_enrichment_revision(
            config,
            client,
            [target],
            revision,
            [target.id],
            kanji_store=None,
        )

    assert client.parse_calls == [], "the ledger is read before any lookup"


def test_a_projected_decision_is_refused_at_the_first_statement_of_commit(
    tmp_path: Path,
) -> None:
    """Ahead of the repository compare and ahead of the early nothing branch:
    those branches are where a caller could otherwise reach a write."""
    config = project(tmp_path, [record()])
    client = hanasu_client()
    target = record()
    revision = io_module.RecordsRevision(
        config.normalized_file.resolve(), io_module.records_json_text([target])
    )
    decision = plan_dictionary_enrichment_revision(
        config, client, [target], revision, [target.id], kanji_store=None
    )
    before = files_now(config)

    with pytest.raises(DictionaryEnrichmentError, match="dictionary-projection-not-executable"):
        commit_dictionary_enrichment(
            config, decision, expected_fingerprint=decision.fingerprint
        )

    # The nothing path: a projection that decides nothing still refuses.
    nothing = plan_dictionary_enrichment_revision(
        config,
        client,
        [settled()],
        revision,
        [target.id],
        kanji_store=None,
    )
    with pytest.raises(DictionaryEnrichmentError, match="dictionary-projection-not-executable"):
        commit_dictionary_enrichment(
            config, nothing, expected_fingerprint=nothing.fingerprint
        )

    # And ahead of the configuration compare, which would otherwise be the
    # first thing a projection from another repository met.
    elsewhere = replace(decision, repository_root=Path("/nowhere"))
    for entry in (
        commit_dictionary_enrichment,
        enrichment_application.commit_dictionary_enrichment_under_guard,
    ):
        with pytest.raises(
            DictionaryEnrichmentError, match="dictionary-projection-not-executable"
        ):
            entry(config, elsewhere, expected_fingerprint=decision.fingerprint)

    assert files_now(config) == before


def test_a_projected_decision_cannot_be_laundered_by_clearing_its_flag(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [record()])
    target = record()
    revision = io_module.RecordsRevision(
        config.normalized_file.resolve(), io_module.records_json_text([target])
    )
    decision = plan_dictionary_enrichment_revision(
        config, hanasu_client(), [target], revision, [target.id], kanji_store=None
    )

    with pytest.raises(DictionaryEnrichmentError, match="dictionary-plan-stale"):
        commit_dictionary_enrichment(
            config,
            replace(decision, projected=False),
            expected_fingerprint=decision.fingerprint,
        )


def test_the_ordinary_planners_are_unprojected_and_still_commit(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [record()])

    exact = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    everything = plan_all_dictionary_enrichment(config, hanasu_client())

    assert exact.projected is False
    assert everything is not None and everything.projected is False

    committed = commit_dictionary_enrichment(
        config, exact, expected_fingerprint=exact.fingerprint
    )

    assert committed.state == "committed"
    assert committed.changed_record_ids == ("word:話す:はなす",)
    assert stored(config)["word:話す:はなす"]["furigana"] == "話[はな]す"


# --- preparation freezes, and publishes nothing -------------------------------


def test_prepare_writes_nothing_and_binds_canonical_and_ledger(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    before = files_now(config)

    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)

    assert files_now(config) == before, "preparation publishes nothing"
    assert [item.role for item in prepared.components] == ["canonical", "ledger"]
    canonical = component(prepared, "canonical")
    assert canonical.expected_before == digest(canonical_text(config))
    assert canonical.after_text == io_module.records_json_text(
        decision.result.records
    )
    assert canonical.expected_after == digest(canonical.after_text)
    ledger_component = component(prepared, "ledger")
    assert ledger_component.expected_before is None, "the ledger does not exist yet"
    assert ledger_component.writes is True
    assert f'"{FROZEN_ISO}"' in ledger_component.after_text
    assert prepared.enriched_at == FROZEN_ISO
    assert prepared.changed_record_ids == ("word:話す:はなす",)
    assert prepared.cleared_record_ids == ()
    # §7.4's pair, so the authority binds them without re-deriving anything.
    assert prepared.projected_input_sha256 == canonical.expected_before
    assert prepared.projected_output_sha256 == canonical.expected_after
    assert prepared.writes is True


def test_prepare_refuses_a_projected_decision(tmp_path: Path) -> None:
    config = project(tmp_path, [record()])
    target = record()
    revision = io_module.RecordsRevision(
        config.normalized_file.resolve(), io_module.records_json_text([target])
    )
    decision = plan_dictionary_enrichment_revision(
        config, hanasu_client(), [target], revision, [target.id], kanji_store=None
    )

    with pytest.raises(DictionaryEnrichmentError, match="dictionary-projection-not-executable"):
        prepare_dictionary_enrichment(config, decision)


def test_a_prepared_intent_round_trips_through_its_own_wire(tmp_path: Path) -> None:
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])

    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    restored = enrichment_application.PreparedDictionaryEnrichment.from_dict(
        json.loads(json.dumps(prepared.to_dict()))
    )

    assert restored == prepared
    assert restored.fingerprint == prepared.fingerprint


# --- apply: ordinary, then the four result shapes -----------------------------


def test_apply_writes_canonical_then_the_ledger_and_reports_both(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)

    committed = guarded_apply(config, prepared, decision=decision)

    assert committed.state == "committed"
    assert committed.changed_record_ids == ("word:話す:はなす",)
    assert committed.cleared_record_ids == ()
    assert committed.ledger_error is None
    assert digest(canonical_text(config)) == component(prepared, "canonical").expected_after
    assert (
        digest(config.ledger_file.resolve().read_text(encoding="utf-8"))
        == component(prepared, "ledger").expected_after
    )
    entry = ledger.load(config.ledger_file.resolve()).records["word:話す:はなす"]
    assert entry["enriched"][0]["at"] == FROZEN_ISO


def test_a_cleared_only_pass_writes_canonical_and_no_attribution_row(
    tmp_path: Path,
) -> None:
    """Canonical changes and the ledger does not: `changed_ids` is not every
    effect, and a bound ledger component that writes nothing still has to be
    prechecked."""
    config = project(tmp_path, [edited_record()])
    config.ledger_file.resolve().write_text("{}\n", encoding="utf-8")
    ledger_before = config.ledger_file.resolve().read_text(encoding="utf-8")
    decision = plan_dictionary_enrichment(
        config, ScriptedDictionary(), ["word:話す:はなす"]
    )
    assert decision.result.cleared, "the stale mark is the only effect"
    assert not decision.result.changes

    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)

    assert component(prepared, "canonical").writes is True
    assert component(prepared, "ledger").writes is False
    assert component(prepared, "ledger").expected_before == digest(ledger_before)

    committed = guarded_apply(config, prepared, decision=decision)

    assert committed.state == "committed"
    assert committed.changed_record_ids == ()
    assert committed.cleared_record_ids == ("word:話す:はなす",)
    assert config.ledger_file.resolve().read_text(encoding="utf-8") == ledger_before


def test_a_mixed_change_and_clear_pass_writes_both_and_reports_both(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [edited_record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    assert decision.result.changes and decision.result.cleared

    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    committed = guarded_apply(config, prepared, decision=decision)

    assert committed.state == "committed"
    assert committed.changed_record_ids == ("word:話す:はなす",)
    assert committed.cleared_record_ids == ("word:話す:はなす",)
    assert stored(config)["word:話す:はなす"]["furigana"] == "話[はな]す"
    assert (
        digest(config.ledger_file.resolve().read_text(encoding="utf-8"))
        == component(prepared, "ledger").expected_after
    )


def test_a_pass_with_nothing_to_do_binds_both_components_and_writes_neither(
    tmp_path: Path,
) -> None:
    target = settled()
    config = project(tmp_path, [target])
    # Compact, not the saver's own indented form: a pass with nothing to do
    # must bind canonical unwritten rather than proposing a reformat nobody
    # asked for — which a collection already in canonical form would hide.
    config.normalized_file.resolve().write_text(
        json.dumps([target.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    decision = plan_dictionary_enrichment(config, hanasu_client(), [target.id])
    before = files_now(config)

    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)

    assert component(prepared, "canonical").writes is False
    assert component(prepared, "ledger").writes is False
    assert component(prepared, "ledger").expected_before is None
    assert component(prepared, "ledger").expected_after is None

    committed = guarded_apply(config, prepared, decision=decision)

    assert committed.state == "nothing"
    assert files_now(config) == before


def test_source_forms_survive_a_prepared_pass_byte_for_byte(tmp_path: Path) -> None:
    table = SourceFormsTable(
        columns=(SourceFormColumn(id="c1", label="ます形"),),
        cells={"c1": "話します"},
    )
    config = project(tmp_path, [record(source_forms=table)])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)

    guarded_apply(config, prepared, decision=decision)

    after = stored(config)["word:話す:はなす"]
    assert after["source_forms"] == {
        "columns": [{"id": "c1", "label": "ます形"}],
        "cells": {"c1": "話します"},
    }
    assert after["furigana"] == "話[はな]す", "the pass really did write something"


# --- the full-vector precheck -------------------------------------------------


def test_a_third_state_at_either_component_refuses_and_names_both_digests(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    config.ledger_file.resolve().write_text('{"version": 1, "records": {}}\n', encoding="utf-8")
    before = files_now(config)

    with pytest.raises(DictionaryEnrichmentError) as excinfo:
        guarded_apply(config, prepared, decision=decision)

    message = str(excinfo.value)
    assert "enrichment-intent-stale" in message
    assert "ledger" in message
    assert str(component(prepared, "ledger").expected_after) in message
    assert files_now(config) == before, "a stale ledger stopped the canonical write"


def test_a_bound_unwritten_component_refuses_a_third_state_too(
    tmp_path: Path,
) -> None:
    """Nochange and absence are bound too: the whole vector is prechecked, not
    only the pending writes."""
    config = project(tmp_path, [edited_record()])
    decision = plan_dictionary_enrichment(
        config, ScriptedDictionary(), ["word:話す:はなす"]
    )
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    assert component(prepared, "ledger").writes is False
    config.ledger_file.resolve().write_text('{"version": 1, "records": {}}\n', encoding="utf-8")
    canonical_before = canonical_text(config)

    with pytest.raises(DictionaryEnrichmentError, match="enrichment-intent-stale"):
        guarded_apply(config, prepared, decision=decision)

    assert canonical_text(config) == canonical_before


def test_an_intent_bound_to_another_configuration_refuses(tmp_path: Path) -> None:
    config = project(tmp_path, [record()])
    other = project(tmp_path / "second", [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    before = files_now(other)

    with pytest.raises(DictionaryEnrichmentError, match="enrichment-intent"):
        guarded_apply(other, prepared, decision=decision)

    assert files_now(other) == before


# --- resume before re-plan ----------------------------------------------------


def crash_after_canonical(config: ProjectConfig, prepared: Any) -> None:
    """Exactly what the writer does before the ledger save, and no more."""
    canonical = component(prepared, "canonical")
    io_module.atomic_write_text_bound(
        Path(canonical.path),
        canonical.after_text,
        expected_revision=canonical.expected_before,
        expected_absent=canonical.expected_before is None,
    )


def test_a_started_intent_is_finished_from_its_payloads_and_never_re_planned(
    tmp_path: Path,
) -> None:
    """The classification runs before any re-plan: a fresh decision would read
    this transaction's own canonical write as somebody else's.

    A noun, deliberately. jpdb states no verb class, transitivity or
    conjugations for one, so those fields stay empty and the record is still
    fillable after the canonical write — which means a re-plan over it really
    does reach the dictionary, and an empty book really would refuse.
    """
    target = noun_record()
    config = project(tmp_path, [target])
    decision = plan_dictionary_enrichment(config, noun_client(), [target.id])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    crash_after_canonical(config, prepared)
    assert enrich.enrich_records(noun_client(), [record()]).looked_up == 1

    # An empty book: any re-plan at all would refuse with dictionary-fact-missing.
    committed = guarded_apply(
        config, prepared, client=ReplayDictionaryClient(DictionaryFactBook())
    )

    assert committed.state == "committed"
    assert committed.changed_record_ids == (target.id,)
    assert (
        digest(config.ledger_file.resolve().read_text(encoding="utf-8"))
        == component(prepared, "ledger").expected_after
    )


def test_recovery_finishes_only_the_pending_write_and_says_which(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    crash_after_canonical(config, prepared)
    canonical_after = canonical_text(config)

    recovered = recover_prepared_dictionary_enrichment(
        config, prepared, client=ReplayDictionaryClient(DictionaryFactBook())
    )

    assert recovered.state == "committed"
    assert recovered.already_complete == ("canonical",)
    assert recovered.finished == ("ledger",)
    assert recovered.changed_record_ids == ("word:話す:はなす",)
    assert canonical_text(config) == canonical_after, "the canonical write was not repeated"


def test_a_next_day_recovery_replays_the_frozen_ledger_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    crash_after_canonical(config, prepared)

    class Tomorrow(date):
        @classmethod
        def today(cls) -> date:
            return LATER

    monkeypatch.setattr(ledger, "date", Tomorrow)

    recovered = recover_prepared_dictionary_enrichment(
        config, prepared, client=ReplayDictionaryClient(DictionaryFactBook())
    )

    assert recovered.finished == ("ledger",)
    assert (
        digest(config.ledger_file.resolve().read_text(encoding="utf-8"))
        == component(prepared, "ledger").expected_after
    )
    entry = ledger.load(config.ledger_file.resolve()).records["word:話す:はなす"]
    assert entry["enriched"][0]["at"] == FROZEN_ISO
    assert entry["added_at"] == FROZEN_ISO


def test_an_unstarted_intent_is_re_planned_through_the_replay_client(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [record()])
    client = hanasu_client()
    book = recorded_book(config, client)
    decision = plan_dictionary_enrichment(
        config, ReplayDictionaryClient(book), ["word:話す:はなす"]
    )
    prepared = prepare_dictionary_enrichment(
        config, decision, now=FROZEN, book=book
    )
    asked = len(client.parse_calls)

    committed = guarded_apply(config, prepared, client=ReplayDictionaryClient(book))

    assert committed.state == "committed"
    assert len(client.parse_calls) == asked, "the apply refetched nothing"
    assert stored(config)["word:話す:はなす"]["furigana"] == "話[はな]す"


def test_an_unstarted_resume_with_no_recorded_answer_refuses_rather_than_fetching(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    before = files_now(config)

    with pytest.raises(enrich.EnrichError, match="dictionary-fact-missing"):
        guarded_apply(
            config, prepared, client=ReplayDictionaryClient(DictionaryFactBook())
        )

    assert files_now(config) == before


def test_a_resume_whose_recomputed_value_moved_refuses_and_names_both(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [record()])
    client = hanasu_client()
    book = recorded_book(config, client)
    decision = plan_dictionary_enrichment(
        config, ReplayDictionaryClient(book), ["word:話す:はなす"]
    )
    prepared = prepare_dictionary_enrichment(
        config, decision, now=FROZEN, book=book
    )
    moved = replace(
        prepared,
        bound_values={"word:話す:はなす": {**prepared.bound_values["word:話す:はなす"],
                                        "furigana": "話[わ]す"}},
    )
    before = files_now(config)

    with pytest.raises(DictionaryEnrichmentError) as excinfo:
        guarded_apply(config, moved, client=ReplayDictionaryClient(book))

    message = str(excinfo.value)
    assert "enrichment-intent-stale" in message
    assert "話[わ]す" in message and "話[はな]す" in message
    assert files_now(config) == before


def test_a_resume_refuses_a_fact_book_that_is_not_the_one_it_bound(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [record()])
    client = hanasu_client()
    book = recorded_book(config, client)
    decision = plan_dictionary_enrichment(
        config, ReplayDictionaryClient(book), ["word:話す:はなす"]
    )
    prepared = prepare_dictionary_enrichment(
        config, decision, now=FROZEN, book=book
    )
    other = RecordingDictionaryClient(ScriptedDictionary({"本": ((), {"spelling": "本"})}))
    other.parse("本")
    before = files_now(config)

    with pytest.raises(DictionaryEnrichmentError, match="enrichment-intent-book"):
        guarded_apply(config, prepared, client=ReplayDictionaryClient(other.freeze()))

    assert files_now(config) == before


# --- the canonical/ledger split -----------------------------------------------


def test_a_failed_ledger_save_still_reports_the_records_as_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)

    def refuse(self: ledger.Ledger) -> None:
        raise ledger.LedgerError("the ledger is read-only")

    monkeypatch.setattr(ledger.Ledger, "save_under_lock", refuse)

    committed = guarded_apply(config, prepared, decision=decision)

    assert committed.state == "committed_ledger_incomplete"
    assert committed.changed_record_ids == ("word:話す:はなす",)
    assert str(committed.ledger_error) == "the ledger is read-only"
    assert stored(config)["word:話す:はなす"]["furigana"] == "話[はな]す"
    assert not config.ledger_file.resolve().exists()


def test_the_split_is_proven_from_the_intent_when_the_process_died_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No return value survived the crash, and none is needed.

    The canonical component is complete and the ledger component is pending —
    that pair is the evidence, and it is what a recovery reports when it still
    cannot finish the bookkeeping. The phase stays outstanding on the same rule
    as promotion's split.
    """
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    crash_after_canonical(config, prepared)
    landed = canonical_text(config)

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise io_module.DataError("the ledger is read-only")

    monkeypatch.setattr(enrichment_application, "atomic_write_text_bound", refuse)

    recovered = recover_prepared_dictionary_enrichment(
        config, prepared, client=ReplayDictionaryClient(DictionaryFactBook())
    )

    assert recovered.state == "committed_ledger_incomplete"
    assert recovered.already_complete == ("canonical",)
    assert recovered.finished == ()
    assert recovered.changed_record_ids == ("word:話す:はなす",)
    assert "read-only" in str(recovered.ledger_error)
    assert canonical_text(config) == landed, "the records stay committed"


# --- the guard ----------------------------------------------------------------


def test_the_ordinary_commit_takes_the_curation_guard_before_it_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    before = canonical_text(config)
    real = study_curation.staging_curation_guard
    reached = threading.Event()
    held = threading.Event()
    outcome: dict[str, Any] = {}

    @contextlib.contextmanager
    def watched(staging_dir: Path) -> Any:
        reached.set()
        with real(staging_dir):
            held.set()
            yield

    monkeypatch.setattr(study_curation, "staging_curation_guard", watched)

    def run() -> None:
        outcome["commit"] = commit_dictionary_enrichment(
            config, decision, expected_fingerprint=decision.fingerprint
        )

    with real(config.staging_dir):
        thread = threading.Thread(target=run)
        thread.start()
        assert reached.wait(5), "the commit reached §6's shared guard"
        assert not held.wait(0.5), "and waited there while another holder had it"
        assert canonical_text(config) == before, "nothing was written meanwhile"
    thread.join(10)

    assert outcome["commit"].state == "committed"
    assert canonical_text(config) != before


def test_a_caller_already_inside_the_guard_uses_the_under_guard_entry(
    tmp_path: Path,
) -> None:
    """`exclusive_path_lock` is not re-entrant, so a coordinator that already
    holds the guard must have an entry that does not take it again."""
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])

    with study_curation.curation_guard(config):
        committed = enrichment_application.commit_dictionary_enrichment_under_guard(
            config, decision, expected_fingerprint=decision.fingerprint
        )

    assert committed.state == "committed"
    assert stored(config)["word:話す:はなす"]["furigana"] == "話[はな]す"


# --- ordering against §7.9's reference write -----------------------------------


def test_an_unstarted_apply_on_a_later_day_still_lands_the_bound_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frozen day is threaded into the attribution the apply itself writes,
    not only into a replayed payload."""
    config = project(tmp_path, [record()])
    client = hanasu_client()
    book = recorded_book(config, client)
    decision = plan_dictionary_enrichment(
        config, ReplayDictionaryClient(book), ["word:話す:はなす"]
    )
    prepared = prepare_dictionary_enrichment(
        config, decision, now=FROZEN, book=book
    )

    class Tomorrow(date):
        @classmethod
        def today(cls) -> date:
            return LATER

    monkeypatch.setattr(ledger, "date", Tomorrow)

    committed = guarded_apply(config, prepared, client=ReplayDictionaryClient(book))

    assert committed.state == "committed"
    assert (
        digest(config.ledger_file.resolve().read_text(encoding="utf-8"))
        == component(prepared, "ledger").expected_after
    )
    entry = ledger.load(config.ledger_file.resolve()).records["word:話す:はなす"]
    assert entry["enriched"][0]["at"] == FROZEN_ISO


def test_the_re_plan_reads_the_reference_store_the_apply_order_has_written(
    tmp_path: Path,
) -> None:
    """§7.9's reference components land first within `enriched`, and §7.8's
    re-plan reads them. A store that appears between the preparation and the
    apply therefore changes what the re-plan computes — and refuses, naming
    both values, instead of quietly writing furigana the owner never saw."""
    projected = record(id="word:明日:あした", expression="明日", reading="あした")
    config = project(tmp_path, [projected])
    client = ScriptedDictionary({"明日": (ASHITA_FURIGANA, ASHITA)})
    recorder = RecordingDictionaryClient(client)
    decision = plan_dictionary_enrichment(config, recorder, [projected.id])
    book = recorder.freeze()
    prepared = prepare_dictionary_enrichment(
        config, decision, now=FROZEN, book=book
    )
    assert prepared.bound_values[projected.id]["furigana"] == "明[あ] 日[した]"

    # The reference facts §7.9 prepares, landing before the enrichment apply.
    kanji.save_store(
        config.kanji_file,
        kanji.KanjiStore(
            entries={
                "日": kanji.KanjiInfo(
                    character="日",
                    stroke_count=4,
                    meanings=("day",),
                    readings=(kanji.Reading(kind="kun", reading="ひ"),),
                )
            }
        ),
    )
    before = files_now(config)

    with pytest.raises(DictionaryEnrichmentError) as excinfo:
        guarded_apply(config, prepared, client=ReplayDictionaryClient(book))

    message = str(excinfo.value)
    assert "enrichment-intent-stale" in message
    assert "明[あ] 日[した]" in message and "明日[あした]" in message
    assert files_now(config) == before


def test_a_started_intent_refuses_when_its_bound_unwritten_ledger_moved(
    tmp_path: Path,
) -> None:
    """The precheck measures the whole vector, not only the pending writes.

    A cleared-only pass writes canonical and binds the ledger *unwritten*. If
    that ledger has since appeared, this intent no longer describes the
    repository it is finishing — and skipping bound-unwritten components would
    let the resume report success over a file somebody else created.
    """
    config = project(tmp_path, [edited_record()])
    decision = plan_dictionary_enrichment(
        config, ScriptedDictionary(), ["word:話す:はなす"]
    )
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    assert component(prepared, "ledger").writes is False
    crash_after_canonical(config, prepared)
    config.ledger_file.resolve().write_text(
        '{"version": 1, "records": {}}\n', encoding="utf-8"
    )
    before = files_now(config)

    with pytest.raises(DictionaryEnrichmentError) as excinfo:
        recover_prepared_dictionary_enrichment(
            config, prepared, client=ReplayDictionaryClient(DictionaryFactBook())
        )

    assert "enrichment-intent-stale" in str(excinfo.value)
    assert "ledger" in str(excinfo.value)
    assert files_now(config) == before


def test_the_guard_taking_apply_entry_is_the_same_transaction(
    tmp_path: Path,
) -> None:
    """The entry a caller outside the guard uses. It takes §6's lock itself and
    reaches the same writer, so a surface never has to know the difference."""
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)

    committed = enrichment_application.apply_prepared_dictionary_enrichment(
        config, prepared, decision=decision
    )

    assert committed.state == "committed"
    assert stored(config)["word:話す:はなす"]["furigana"] == "話[はな]す"
    assert (
        digest(config.ledger_file.resolve().read_text(encoding="utf-8"))
        == component(prepared, "ledger").expected_after
    )


def test_an_unstarted_resume_with_no_client_refuses_rather_than_guessing(
    tmp_path: Path,
) -> None:
    config = project(tmp_path, [record()])
    decision = plan_dictionary_enrichment(config, hanasu_client(), ["word:話す:はなす"])
    prepared = prepare_dictionary_enrichment(config, decision, now=FROZEN)
    before = files_now(config)

    with pytest.raises(
        DictionaryEnrichmentError, match="enrichment-intent-client-required"
    ):
        guarded_apply(config, prepared)

    assert files_now(config) == before
