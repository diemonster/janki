"""One owner-authorized apply-and-finish for a whole study job.

`docs/ASSISTANT_STUDY_JOBS_CONTRACTS.md` §7.6–§7.13 and §9.6. Everything here
runs the *real* writers — the staging review batch, the promotion fold and its
prepared intents, the reference/dictionary writers, the audio application's own
execution and completion proof, and the deck-package preparation/publication —
against a temporary repository.

**Exactly what is faked, and nothing else**, because each of these would
otherwise leave this machine:

1. the extraction subscription CLI (`FakeClaudeRunner`, plus `_no_api`'s refusal
   of the billed Anthropic path);
2. the jpdb word dictionary the plan and the enrichment pass consult
   (`FakeJpdb` through `client_for`);
3. the speech providers — `RecordingProvider` for words, and either the same
   recorder or the suite's `_RealtimeTransport` for sentences;
4. the two kanji **reference** lookups the finish's reference preparation makes,
   `kanji.fetch_kanji` (kanjiapi + KanjiVG) and `jpdb_kanji.fetch_character`
   (jpdb's kanji and reading-detail pages), answered by `_ReferenceAnswers`
   below with the transport under them replaced by a local refusal.

Nothing here reads Japanese. Every assertion is an artifact fact: which bytes
are at which digest, which intent was durable before which effect, and which
receipt a job's completion rests on.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_application_audio import RecordingProvider
from test_application_revision_finish import _RealtimeTransport
from test_promote import FakeJpdb, client_for
from test_revision_provider import FakeClaudeRunner, _which
from test_workbench_fixtures import RESPONSES

from conftest import seed_prompts
from japanese_anki import (
    card_preview,
    claude_client,
    jpdb_kanji,
    kanji,
    patterns,
    staging,
)
from japanese_anki.application import (
    assistant_assignment,
    deck_package,
    extraction_batch,
    study_curation,
    study_finish,
    study_job,
)
from japanese_anki.application import (
    coverage as coverage_application,
)
from japanese_anki.config import ProjectConfig
from japanese_anki.errors import JankiError
from japanese_anki.models import VocabularyRecord
from japanese_anki.tts import openai_realtime

BOUND = 20.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "table_exhaustive"

pytestmark = pytest.mark.skipif(
    card_preview.preview_unavailable() is not None,
    reason=str(card_preview.preview_unavailable()),
)

# The three rows `table_exhaustive` proposes, and the jpdb facts a reading check
# and an enrichment pass ask for. Ordinary Genki-level material used as input.
HASHIRU = [1310730, 1, "走る", "はしる", ["LHL"], 400, ["v5r", "vi"]]
TABERU = [1358280, 2, "食べる", "たべる", ["LHLL"], 300, ["v1", "vt"]]
NOMU = [1462610, 3, "飲む", "のむ", ["LHL"], 500, ["v5m", "vt"]]

SENSES = {
    (1310730, 1): {"reading": "はしる", "alt_sids": []},
    (1358280, 2): {"reading": "たべる", "alt_sids": []},
    (1462610, 3): {"reading": "のむ", "alt_sids": []},
}


def _answer(scenario: str = SCENARIO) -> dict[str, Any]:
    return json.loads((RESPONSES / f"{scenario}.json").read_text(encoding="utf-8"))


def _stream(answer: Any) -> bytes:
    fixture = Path(__file__).parent / "fixtures" / "claude-code-stream.ndjson"
    lines = []
    for line in fixture.read_bytes().splitlines(keepends=True):
        payload = json.loads(line)
        if payload.get("type") == "result":
            payload["structured_output"] = answer
            line = json.dumps(payload).encode("utf-8") + b"\n"
        lines.append(line)
    return b"".join(lines)


def _no_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the Anthropic API was reached")

    monkeypatch.setattr(claude_client, "prepare_paid_client", refuse)
    monkeypatch.setattr(claude_client, "parse_call", refuse)


class _FlakyTransport(_RealtimeTransport):
    """The suite's Realtime transport, with one switchable failure before send.

    Deliberately one class rather than two: the profile a confirmed plan binds
    names the transport implementation, so swapping in a different type between a
    run and its resume changes the authority itself.
    """

    #: ``"before_send"`` fails without connecting; ``"mid_stream"`` connects and
    #: then stops answering, which is the state that means a provider may already
    #: have billed; ``""`` answers normally.
    mode: str

    def __init__(self, mode: str = "before_send") -> None:
        super().__init__()
        self.mode = mode

    def __call__(self, endpoint: str, headers: Any, request: Any) -> Any:
        if self.mode == "before_send":
            self.calls.append(request)
            raise TimeoutError("the connection dropped before it was established")
        if self.mode == "mid_stream":
            self.calls.append(request)
            return [
                {
                    "type": "session.created",
                    "session": {"model": openai_realtime.MODEL},
                }
            ]
        return super().__call__(endpoint, headers, request)


def _dictionary() -> FakeJpdb:
    return FakeJpdb(
        {"走る": HASHIRU, "食べる": TABERU, "飲む": NOMU},
        SENSES,
    )


# --- the two reference services, answered here and never requested -----------
#
# `plan_study_finish` folds the accepted records' kanji and calls
# `character_notes.prepare_reference_facts`, which asks `kanji.fetch_kanji` and
# `jpdb_kanji.fetch_character` for every character neither reference store
# already holds. The temporary repository below starts with an empty vocabulary
# and no `kanji.json` or `jpdb_readings.json` at all, so every character is a
# miss — and both of those functions default to real urllib transport
# (kanjiapi.dev, KanjiVG's raw GitHub tree, jpdb.io, whose pages are published
# into the raw cache outside the repository). `tests/conftest.py` does not cover
# them: it guards the billed Anthropic client and the installed Claude CLI.
#
# So this module answers both lookups itself with fixed synthetic facts **and**
# replaces the transport underneath them with a local refusal, so a path that
# stops going through those two seams fails here rather than contacting a
# service. `test_the_plans_reference_lookups_are_answered_without_any_request`
# below is the fail-safe that keeps both halves honest.

_REAL_FETCH_KANJI = kanji.fetch_kanji
_REAL_FETCH_CHARACTER = jpdb_kanji.fetch_character

#: One fixed KANJIDIC/KanjiVG-shaped answer per character these fixtures accept.
#: The values are this fixture's own, deterministic and never derived from
#: anything the tests assert on; a character no fixture enumerates refuses
#: rather than being answered by a rule of this file's making.
_REFERENCE: dict[str, tuple[int, tuple[str, ...], tuple[tuple[str, str], ...]]] = {
    "走": (7, ("run",), (("on", "ソウ"), ("kun", "はし.る"))),
    "食": (9, ("eat", "food"), (("on", "ショク"), ("kun", "た.べる"))),
    "飲": (12, ("drink",), (("on", "イン"), ("kun", "の.む"))),
}


class _ReferenceAnswers:
    """Both reference lookups, answered from `_REFERENCE` and recorded."""

    def __init__(self) -> None:
        self.looked_up: list[str] = []
        self.fetched: list[str] = []

    def _known(
        self, character: str
    ) -> tuple[int, tuple[str, ...], tuple[tuple[str, str], ...]]:
        if character not in _REFERENCE:
            raise AssertionError(
                f"This module holds no fixed reference answer for {character!r}. "
                "Add one to _REFERENCE; a lookup must never fall through to a "
                "real service."
            )
        return _REFERENCE[character]

    def fetch_kanji(self, character: str, **_kwargs: Any) -> kanji.KanjiInfo:
        strokes, meanings, readings = self._known(character)
        self.looked_up.append(character)
        return kanji.KanjiInfo(
            character=character,
            stroke_count=strokes,
            meanings=meanings,
            readings=tuple(
                kanji.Reading(kind=kind, reading=reading) for kind, reading in readings
            ),
            strokes=("M10,10 L20,20",),
        )

    def fetch_character(
        self,
        character: str,
        *,
        html_cache: Path | None = None,
        refresh: bool = False,
        **_kwargs: Any,
    ) -> jpdb_kanji.CharacterReadings:
        self._known(character)
        self.fetched.append(character)
        return jpdb_kanji.CharacterReadings(
            character=character,
            source_url=f"https://jpdb.io/kanji/{character}",
            fetched_at_utc="2026-09-12T00:00:00Z",
            sha256=hashlib.sha256(character.encode("utf-8")).hexdigest(),
            groups=(),
        )


def _refuse_reference_transport(patch: pytest.MonkeyPatch) -> None:
    """Replace both reference transports with a local refusal.

    `fetch_kanji` and `fetch_character` both resolve `transport or
    urllib_transport` at call time, so rebinding that module attribute stops the
    request before anything is sent: the failure is an `AssertionError` in this
    process, never an HTTP request to a service.
    """

    def refuse(url: str, timeout: float = 0.0) -> bytes:
        raise AssertionError(
            f"This test reached a live reference service at {url}. Both kanji "
            "reference lookups are answered by _ReferenceAnswers; patch the seam "
            "the code path actually takes instead of letting the transport run."
        )

    patch.setattr(kanji, "urllib_transport", refuse)
    patch.setattr(jpdb_kanji, "urllib_transport", refuse)


@pytest.fixture(autouse=True)
def reference() -> Iterator[_ReferenceAnswers]:
    """Fixed reference facts and no reference transport, for every test here.

    Deliberately its own `MonkeyPatch` rather than the test's: several tests
    below call `monkeypatch.undo()` part-way through to drop a seam they
    installed, and that undoes *every* patch on the shared instance. Installing
    these four on a separate instance means an offline guarantee cannot be
    switched off half-way through a test that was undoing something else.
    """
    answers = _ReferenceAnswers()
    patch = pytest.MonkeyPatch()
    patch.setattr(kanji, "fetch_kanji", answers.fetch_kanji)
    patch.setattr(jpdb_kanji, "fetch_character", answers.fetch_character)
    _refuse_reference_transport(patch)
    try:
        yield answers
    finally:
        patch.undo()


def _project(tmp_path: Path) -> ProjectConfig:
    """A scratch repository with the checkout's real templates in it.

    `jpdb_html_cache` is named explicitly, at a scratch path beside this test's
    own root: the default is the developer's own `~/Library/Caches/janki/jpdb`,
    and a reference lookup that published or read that cache would make this
    module's results depend on what earlier real work left on the machine.
    """
    seed_prompts(tmp_path)
    cache = tmp_path.parent / f"{tmp_path.name}-jpdb-cache"
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        f'jpdb_html_cache = "{cache}"\n'
        'normalized_file = "vocabulary.json"\n'
        'ledger_file = "ledger.json"\n'
        'deck_dir = "decks"\n'
        'media_dir = "media"\n'
        'template_dir = "templates/japanese-study"\n'
        'dist_dir = "dist"\n'
        'staging_dir = "staging"\n'
        'scan_inbox = "inbox"\n'
        'patterns_file = "patterns.json"\n'
        'kanji_file = "kanji.json"\n'
        'jpdb_readings_file = "jpdb_readings.json"\n'
        "[ai]\n"
        'extract_provider = "claude-code"\n'
        'extract_model = "claude-opus-5"\n',
        encoding="utf-8",
    )
    shutil.copytree(PROJECT_ROOT / "templates", tmp_path / "templates")
    (tmp_path / "decks").mkdir()
    (tmp_path / "media").mkdir()
    (tmp_path / "vocabulary.json").write_text("[]", encoding="utf-8")
    return ProjectConfig.load(tmp_path)


def _deck(config: ProjectConfig, name: str = "lesson") -> Path:
    path = config.deck_dir / f"{name}.yaml"
    path.write_text(
        "deck:\n"
        f"  name: {name.title()} deck\n"
        "  deck_id: 1500000007\n"
        "  source: ../vocabulary.json\n"
        "  include_tags: [lesson-intake]\n"
        "  intake_tag: lesson-intake\n"
        "  cards:\n"
        "    recognition: true\n"
        "    production: true\n",
        encoding="utf-8",
    )
    return path


def _source(config: ProjectConfig, name: str, body: bytes) -> Path:
    path = config.scan_inbox / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 " + body)
    return path


def _settled_job(
    config: ProjectConfig,
    *,
    names: tuple[str, ...] = ("one.pdf",),
    assign: bool = True,
    scenario: str = SCENARIO,
    answer: dict[str, Any] | None = None,
) -> Any:
    """One study job whose extraction really ran and settled, per source."""
    deck = _deck(config)
    job = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=_source(config, "parent.pdf", b"parent"),
        deck_path=deck,
    )
    runner = FakeClaudeRunner(
        reply=_stream(_answer(scenario) if answer is None else answer)
    )
    sources = [_source(config, name, name.encode("ascii")) for name in names]
    plan = extraction_batch.plan_extraction_batch(
        config,
        sources,
        job_id=job.header.job_id,
        destination_deck=deck,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )
    outcome = study_job.dispatch_job_batch(
        config,
        job.header.job_id,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    assert [child.state for child in outcome.children] == ["committed"] * len(names)
    if assign:
        _assign(config, deck)
    return study_job.load_study_job(config, job.header.job_id)


def _assign(config: ProjectConfig, deck: Path) -> None:
    """The ordinary owner step between extraction and review.

    A staged row carries no deck until somebody puts it in one, and promotion's
    deck gate refuses a row no configured deck selects. This is the same writer
    `janki study assign` calls, so the intake tags these tests promote are the
    ones that flow really produces.
    """
    for path in sorted(config.staging_dir.glob("*.yaml")):
        records, _meta = staging.read_staging_text(
            path.read_text(encoding="utf-8"), source=str(path)
        )
        if not records:
            # A zero-record rich extraction proposes grammar and no card, so
            # there is nothing for an assignment to put in a deck.
            continue
        plan = assistant_assignment.plan_assignment_for_paths(
            config,
            proposal_path=path,
            destination_path=deck,
            record_ids=[record.id for record in records],
            instruction=f"Assign {path.name} to {deck.stem}.",
        )
        assistant_assignment.execute_assignment(config, plan)


def _owner_choices(
    config: ProjectConfig,
    job: Any,
    *,
    include_audio: bool | None = None,
    dispositions: dict[str, dict[str, Any]] | None = None,
    review_flags: dict[str, dict[str, Any]] | None = None,
    pattern_choices: dict[str, bool | None] | None = None,
) -> Any:
    """Save the owner's decisions, bound the way their editors bind them.

    Each per-part decision carries the job, the part, that part's current
    staging bytes and the fingerprint of the rendering it was taken over. The
    sentence-audio preference carries only the job and the revision it was saved
    against, because it is job-wide.

    `pattern_choices` writes the standalone `review_patterns` choice — the one
    place a part's pattern-set mark is stored — where `None` is the owner
    *withdrawing* that mark through the same control and the same
    compare-and-swap that recorded it.
    """
    job_id = job.header.job_id
    rendering = study_job.render_job_preview(config, job_id).rendering_fingerprint
    saved = study_job.load_study_job(config, job_id)
    for part_name, staging_path in _parts(config, job_id):
        bindings = {
            "job_id": job_id,
            "part": part_name,
            "staging_sha256": hashlib.sha256(
                staging_path.read_bytes()
            ).hexdigest(),
            "rendering_fingerprint": rendering,
        }
        choice: dict[str, Any] = {}
        # Only where the page really is offering one: a part whose coverage the
        # extraction already settled has nothing for an owner reason to approve,
        # and the coordinator refuses a saved reason there by name.
        if coverage_application.plan_coverage(config, staging_path).state == "ready":
            choice["coverage_reasons"] = {
                part_name: {
                    **bindings,
                    "reason": "I checked this page against the source myself.",
                }
            }
        if review_flags is not None and part_name in review_flags:
            choice["review_flags"] = {
                part_name: {**bindings, **review_flags[part_name]}
            }
        if pattern_choices is not None and part_name in pattern_choices:
            marked = pattern_choices[part_name]
            choice["review_patterns"] = {
                part_name: None if marked is None else {**bindings, "value": marked}
            }
        if dispositions is not None and part_name in dispositions:
            choice["dispositions"] = {
                part_name: {**bindings, **dispositions[part_name]}
            }
        if choice:
            saved = study_job.record_choice(
                config, job_id, choice, expected_revision=saved.revision
            )
    if include_audio is not None:
        saved = study_job.record_choice(
            config,
            job_id,
            {"include_example_audio": {"value": include_audio}},
            expected_revision=saved.revision,
        )
    return saved


def _parts(config: ProjectConfig, job_id: str) -> list[tuple[str, Path]]:
    status = study_job.study_job_status(config, job_id)
    found: list[tuple[str, Path]] = []
    for batch in status.batches:
        for child in batch.children:
            if child.state == "committed" and child.bookkeeping_complete:
                found.append(
                    (child.source_name, config.staging_dir / child.staging_name)
                )
    return sorted(found)


def _plan(
    config: ProjectConfig,
    job: Any,
    api: FakeJpdb,
    words: Any,
    sentences: Any = None,
) -> study_finish.StudyFinishPlan:
    return study_finish.plan_study_finish(
        config,
        job.header.job_id,
        client_factory=lambda: client_for(api),
        chosen_provider="voicevox",
        word_provider=words,
        sentence_provider=sentences if sentences is not None else words,
    )


def _record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(config: ProjectConfig) -> dict[str, Any]:
    return {
        item["id"]: item
        for item in json.loads(config.normalized_file.read_text(encoding="utf-8"))
    }


# --- the fixture's own offline guarantee --------------------------------------


def test_the_plans_reference_lookups_are_answered_without_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: _ReferenceAnswers
) -> None:
    """The fixed reference answers are load-bearing, not decoration.

    First half: the ordinary plan asks for exactly the three characters its
    accepted rows are written with, and this module answers all three. Second
    half: with the real `fetch_kanji` restored — which is what every test in this
    module did before this fixture existed — the same plan reaches
    `kanji.urllib_transport`, and the local refusal installed over it fails the
    test instead of a request leaving the machine. That refusal is what proves
    the first half is isolation rather than coincidence.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)

    plan = _plan(config, job, _dictionary(), words)

    assert sorted(plan.reference.characters) == ["走", "食", "飲"]
    assert sorted(reference.looked_up) == ["走", "食", "飲"]
    assert sorted(reference.fetched) == ["走", "食", "飲"]
    assert plan.missing_reference_facts == ()
    # Nothing published a raw provider page anywhere: the answers never touched a
    # cache, and the configured cache directory was never even created.
    assert not (tmp_path.parent / f"{tmp_path.name}-jpdb-cache").exists()

    # Restore this probe before the reference fixture tears down. The shared
    # monkeypatch fixture outlives it and would otherwise reinstall its fake.
    with monkeypatch.context() as probe_patch:
        probe_patch.setattr(kanji, "fetch_kanji", _REAL_FETCH_KANJI)
        probe_patch.setattr(jpdb_kanji, "fetch_character", _REAL_FETCH_CHARACTER)

        with pytest.raises(AssertionError, match="live reference service"):
            _plan(config, job, _dictionary(), words)


# --- the whole chain, with sentence audio left at its default -----------------


def test_one_authority_carries_a_whole_job_from_review_to_a_previewed_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain, end to end, over the real writers.

    Sentence audio is not chosen at all here: §7.10's default is to include it,
    so the plan discloses three word slots and six stored example slots without
    the owner having saved anything about audio. The completion rests on the
    package receipt *and* the preview receipt, and both are checked against the
    bytes on disk rather than against a phase label.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    api = _dictionary()
    words = RecordingProvider("voicevox", 7)

    plan = _plan(config, job, api, words)

    assert plan.audio.include_example_audio is True
    assert plan.audio.omitted_by_owner is False
    assert plan.audio.expected_word_slots == 3
    assert plan.audio.expected_example_slots == 6
    assert plan.audio.paid_provider_calls == 0
    assert sorted(plan.record_ids) == [
        "word:走る:はしる",
        "word:食べる:たべる",
        "word:飲む:のむ",
    ]
    assert plan.undisposed_parts == ()
    # The authority binds `deck_package`'s own complete projection wire, and it
    # restores to a plan that re-earns its own fingerprint.
    wire = plan.authority["package"]["projection"]
    restored = deck_package.plan_from_wire(config, wire)
    assert restored.fingerprint == wire["plan_fingerprint"]
    assert restored.output_path == plan.output_path
    assert {item.sha256 for item in restored.media_inputs} == {None}
    assert not plan.record_path.exists(), "planning wrote nothing"

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded and result.state == "complete"
    record = _record(plan.record_path)
    assert record["state"] == "complete"
    assert record["receipt_id"] == plan.fingerprint
    package = config.root / record["package_receipt"]["output_path"]
    assert (
        hashlib.sha256(package.read_bytes()).hexdigest()
        == record["package_receipt"]["package_sha256"]
    )
    preview = config.root / record["preview_receipt"]["preview_path"]
    assert (
        hashlib.sha256(preview.read_bytes()).hexdigest()
        == record["preview_receipt"]["preview_sha256"]
    )
    assert record["preview_receipt"]["package_sha256"] == (
        record["package_receipt"]["package_sha256"]
    )
    assert set(record["preview_receipt"]["render_assets"]) == {
        "notetype.css",
        "viewer.css",
        "viewer.js",
    }
    assert record["preview_receipt"]["audio_state"] == "packaged"

    # Every accepted card landed with its word clip and both sentence clips.
    canonical = _canonical(config)
    assert sorted(canonical) == sorted(plan.record_ids)
    for item in canonical.values():
        assert item["audio"]
        assert [example["audio"] for example in item["examples"]] == [
            example["audio"] for example in item["examples"] if example["audio"]
        ]
        assert len(item["examples"]) == 2
    proof = record["audio_receipt"]["proof"]
    assert record["audio_receipt"]["stored_slot_count"] == 9
    assert record["audio_receipt"]["paid_clip_count"] == 0
    assert {slot["origin"] for slot in proof["slots"]} == {"synthesized"}
    assert record["paid_clip_reservations"] == []

    from japanese_anki import ledger as ledger_module

    assert ledger_module.load(config.ledger_file).pending_audio == {}
    # The staging document really was reviewed and archived by the promotion
    # writer, so the job's own part is settled by artifact. No row was flagged
    # for review here and no pattern set was marked, which the receipt says
    # rather than leaving to inference.
    assert record["review_receipt"]["parts"]["one.pdf"] == {
        "record_ids": [],
        "pattern_reviewed": False,
    }
    assert record["promotion_receipt"]["parts"][0]["state"] == "landed"
    assert record["promotion_receipt"]["projection"] == [
        [record_id, "lesson", str(plan.deck_path)]
        for record_id in record["promotion_receipt"]["landed_ids"]
    ]
    archive = config.root / record["promotion_receipt"]["parts"][0]["archive_path"]
    assert archive.is_file()
    assert not (config.staging_dir / "one.pdf.yaml").exists()


def test_a_blocked_part_accepts_no_card_and_names_the_gate_that_refused_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate refusal is a blocker, not a set of cards this finish may claim.

    The fold still *discloses* which rows a blocked part proposed — that is what
    makes the refusal readable — but nothing canonical is written for it, so
    those ids are not this finish's accepted selection. Treating them as
    accepted would plan audio and a package over cards that do not exist.

    Here the staged rows were never assigned to a study deck, which is exactly
    the `deck` gate's refusal.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config, assign=False)
    _owner_choices(config, job)

    with pytest.raises(study_finish.StudyFinishError) as error:
        _plan(config, job, _dictionary(), RecordingProvider("voicevox", 7))

    message = str(error.value)
    assert "land no card at all" in message
    assert "one.pdf" in message and "deck" in message
    assert not list(study_finish.study_finish_directory(config).glob("*.json"))


# --- the owner's sentence-audio control ---------------------------------------


def test_the_owner_can_omit_sentence_audio_and_then_re_enable_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An opt-out binds zero example requests and is labelled as the owner's.

    It is never reported as successful sentence generation: the receipt records
    `omitted_by_owner`, the preview says so on its face, and the census expects
    no example slot. Re-enabling is the same control, and it produces a
    different authority because it authorizes different work.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job, include_audio=False)
    words = RecordingProvider("voicevox", 7)

    omitted = _plan(config, job, _dictionary(), words)

    assert omitted.audio.include_example_audio is False
    assert omitted.audio.omitted_by_owner is True
    assert omitted.audio.expected_example_slots == 0
    assert omitted.audio.expected_word_slots == 3
    assert omitted.audio.example_counts.total == 0

    # Re-enabling is the same control and the same CAS, and it authorizes
    # different work, so it is a different authority rather than an edit.
    reenabled_job = _owner_choices(config, job, include_audio=True)
    assert reenabled_job.choices["include_example_audio"]["value"] is True
    reenabled = _plan(config, job, _dictionary(), words)
    assert reenabled.audio.include_example_audio is True
    assert reenabled.audio.omitted_by_owner is False
    assert reenabled.audio.expected_example_slots == 6
    assert reenabled.fingerprint != omitted.fingerprint
    assert reenabled.record_path != omitted.record_path

    _owner_choices(config, job, include_audio=False)
    again = _plan(config, job, _dictionary(), words)
    assert again.fingerprint == omitted.fingerprint, (
        "the same saved choice is the same authority"
    )
    result = study_finish.execute_study_finish(
        config, again, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(again.record_path)
    assert record["audio_receipt"]["omitted_by_owner"] is True
    assert record["audio_receipt"]["stored_slot_count"] == 3
    assert record["preview_receipt"]["audio_state"] == "omitted"
    assert record["preview_receipt"]["notices"] == [
        "Sentence audio was omitted by owner for this job; existing clips and "
        "references were kept."
    ]
    page = (config.root / record["preview_receipt"]["preview_path"]).read_text(
        encoding="utf-8"
    )
    assert "omitted by owner" in page
    # The canonical rows kept both examples; none of them was voiced.
    for item in _canonical(config).values():
        assert item["audio"]
        assert [example.get("audio") or "" for example in item["examples"]] == [
            "",
            "",
        ]


# --- stored slots, unique clips and duplicate effective inputs ----------------


def test_expected_stored_slots_are_counted_apart_from_unique_clip_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nine stored slots, and the clips they resolve to, stay two numbers.

    The census is derived from the accepted records' own fields rather than from
    the plan's clip list, which is what stops an empty request list passing
    merely because every selected clip is already current.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)

    plan = _plan(config, job, _dictionary(), words)

    assert plan.audio.expected_word_slots + plan.audio.expected_example_slots == 9
    assert plan.audio.unique_clip_requests == 9
    assert plan.audio.word_counts.provider_required == 3
    assert plan.audio.example_counts.provider_required == 6
    # The whole-deck package projection is a different scope again: two card
    # directions over three notes, with the media the renderer really reads.
    restored = deck_package.plan_from_wire(
        config, plan.authority["package"]["projection"]
    )
    assert plan.note_count == 3
    assert plan.card_count == 6
    assert len(restored.media_inputs) == 9


def test_clips_this_finish_already_made_are_reused_without_a_second_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume after the audio landed re-proves it; it does not re-synthesize.

    The first run is stopped between the audio writer's return and the receipt —
    the bytes, the canonical references and the ledger rows are all on disk while
    the finish still says `enriched`. The resume re-plans over those exact
    records, finds every one of the nine slots current, and asks the provider for
    nothing at all. Reuse is evidence: the proof still accounts for all nine.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    plan = _plan(config, job, _dictionary(), words)

    real_advance = study_finish._advance

    def stop_before_the_audio_receipt(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("to_state") == "audio_complete":
            raise study_finish.StudyFinishError("stopped after the clips landed")
        return real_advance(*args, **kwargs)

    monkeypatch.setattr(study_finish, "_advance", stop_before_the_audio_receipt)
    with pytest.raises(
        study_finish.StudyFinishError, match="stopped after the clips landed"
    ):
        study_finish.execute_study_finish(
            config, plan, word_provider=words, sentence_provider=words
        )
    monkeypatch.undo()
    _no_api(monkeypatch)

    held = _record(plan.record_path)
    assert held["state"] == "enriched"
    assert held["audio_receipt"] is None
    spoken = len(words.said)
    assert spoken == 9, "the first run really did synthesize every slot"
    clips = sorted(
        path.name for path in (config.media_dir / "audio").glob("janki-*.wav")
    )
    assert len(clips) == 9

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a reused clip was requested from the provider again")

    monkeypatch.setattr(words, "synthesize", explode)
    result = study_finish.resume_study_finish(
        config, plan.fingerprint, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    assert len(words.said) == spoken
    final = _record(plan.record_path)
    assert final["audio_receipt"]["stored_slot_count"] == 9
    assert final["audio_receipt"]["paid_clip_count"] == 0
    # `origin` is what the *authority* classified the clip as, which is B's own
    # vocabulary: this finish was confirmed to synthesize these nine, and it did
    # — once. The no-second-request claim is the exploding provider above, not a
    # relabelling of the proof.
    assert {slot["origin"] for slot in final["audio_receipt"]["proof"]["slots"]} == {
        "synthesized"
    }
    assert (
        sorted(path.name for path in (config.media_dir / "audio").glob("janki-*.wav"))
        == clips
    ), "a second copy of a clip was written"


# --- durable authority, replay, and recovery ----------------------------------


def test_an_interrupted_finish_replays_its_bound_dictionary_and_recovers_its_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The authority is durable before the first effect, and a resume replays it.

    The first run is stopped between the promotion writer's return and the
    receipt, so the promotion intent is on disk and the archive exists while the
    finish still says `reviewed`. The resume finishes *from that intent* — it
    does not decide again — and makes no dictionary call at all, because every
    fact it needs was frozen into the authority at plan time.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    api = _dictionary()
    words = RecordingProvider("voicevox", 7)
    plan = _plan(config, job, api, words)
    asked_while_planning = len(api.bodies)
    assert asked_while_planning > 0, "the plan really did consult a dictionary"

    real_apply = study_finish.promotion.apply_prepared_source_promotion

    def stop_after_promoting(*args: Any, **kwargs: Any) -> Any:
        real_apply(*args, **kwargs)
        raise study_finish.StudyFinishError("stopped after the promotion returned")

    monkeypatch.setattr(
        study_finish.promotion, "apply_prepared_source_promotion", stop_after_promoting
    )
    with pytest.raises(
        study_finish.StudyFinishError, match="stopped after the promotion returned"
    ):
        study_finish.execute_study_finish(
            config, plan, word_provider=words, sentence_provider=words
        )
    monkeypatch.undo()
    _no_api(monkeypatch)

    interrupted = _record(plan.record_path)
    assert interrupted["state"] == "reviewed"
    assert interrupted["promotion_receipt"] is None
    assert set(interrupted["promotion_intents"]) == {"one.pdf"}
    frozen_intent = json.dumps(
        interrupted["promotion_intents"]["one.pdf"], sort_keys=True
    )
    assert len(api.bodies) == asked_while_planning, "a phase refetched a fact"

    recovered: list[str] = []
    real_recover = study_finish.promotion.recover_promotion_intent_under_guard

    def observe(*args: Any, **kwargs: Any) -> Any:
        recovered.append("recovered")
        return real_recover(*args, **kwargs)

    def never_decide(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a started promotion was decided again")

    monkeypatch.setattr(
        study_finish.promotion, "recover_promotion_intent_under_guard", observe
    )
    monkeypatch.setattr(study_finish.promotion, "decide_promotion", never_decide)

    result = study_finish.resume_study_finish(
        config, plan.fingerprint, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    assert recovered == ["recovered"], "recovery ran once, before any replan"
    assert len(api.bodies) == asked_while_planning, "the resume made a fresh call"
    final = _record(plan.record_path)
    assert final["state"] == "complete"
    assert (
        json.dumps(final["promotion_intents"]["one.pdf"], sort_keys=True)
        == frozen_intent
    ), "a recorded intent was rewritten"


def test_a_failed_preview_leaves_a_packaged_receipt_whose_resume_only_previews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§9.6.7: a preview that fails after the package is proven costs no rebuild.

    The retry draws the same proven bytes and calls no builder, no publisher and
    no provider — asserted by wiring all three to explode.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    plan = _plan(config, job, _dictionary(), words)

    def fail_the_preview(*_args: Any, **_kwargs: Any) -> Any:
        raise card_preview.CardPreviewError("the viewer asset went missing")

    monkeypatch.setattr(
        card_preview, "render_packaged_card_preview", fail_the_preview
    )
    with pytest.raises(study_finish.StudyFinishError, match="could not be drawn"):
        study_finish.execute_study_finish(
            config, plan, word_provider=words, sentence_provider=words
        )
    monkeypatch.undo()
    _no_api(monkeypatch)

    packaged = _record(plan.record_path)
    assert packaged["state"] == "packaged"
    assert packaged["preview_receipt"] is None
    package_sha256 = packaged["package_receipt"]["package_sha256"]
    intents = list(packaged["package_intents"])
    assert len(intents) == 1
    assert study_finish.inspect_study_finish(config, plan.fingerprint).state == (
        "packaged"
    )

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a preview-only resume did paid or building work")

    monkeypatch.setattr(study_finish.deck_package, "prepare_deck_package", explode)
    monkeypatch.setattr(
        study_finish.deck_package, "publish_prepared_deck_package", explode
    )
    monkeypatch.setattr(
        study_finish.audio_application, "execute_targeted_audio_locked", explode
    )
    monkeypatch.setattr(words, "synthesize", explode)

    result = study_finish.resume_study_finish(config, plan.fingerprint)

    assert result.succeeded
    final = _record(plan.record_path)
    assert final["package_receipt"]["package_sha256"] == package_sha256
    assert final["package_intents"] == intents, "the resume prepared a new package"
    assert final["preview_receipt"]["package_sha256"] == package_sha256


def test_a_vanished_private_stage_is_rebuilt_as_a_superseding_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§7.11: a missing scratch artifact mints a new intent, never edits the old.

    `dist/` is disposable, so the rebuild is free — but the preparation that
    replaces the vanished one is a separate immutable record that names what it
    supersedes, and the superseded intent's bytes are untouched.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    plan = _plan(config, job, _dictionary(), words)

    real_publish = study_finish.deck_package.publish_prepared_deck_package

    def stop_before_publishing(*_args: Any, **_kwargs: Any) -> Any:
        raise study_finish.StudyFinishError("stopped before publication")

    monkeypatch.setattr(
        study_finish.deck_package,
        "publish_prepared_deck_package",
        stop_before_publishing,
    )
    with pytest.raises(
        study_finish.StudyFinishError, match="stopped before publication"
    ):
        study_finish.execute_study_finish(
            config, plan, word_provider=words, sentence_provider=words
        )
    monkeypatch.undo()
    _no_api(monkeypatch)

    held = _record(plan.record_path)
    assert held["state"] == "audio_complete"
    assert len(held["package_intents"]) == 1
    first = held["package_intents"][0]
    staged = config.root / first["staged_path"]
    assert staged.is_file()
    staged.unlink()

    monkeypatch.setattr(
        study_finish.deck_package, "publish_prepared_deck_package", real_publish
    )
    result = study_finish.resume_study_finish(
        config, plan.fingerprint, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    final = _record(plan.record_path)
    assert len(final["package_intents"]) == 2
    assert final["package_intents"][0] == first, "the old intent was edited"
    assert final["package_intents"][1]["supersedes"] == first["preparation_id"]
    assert (
        final["package_intents"][1]["preparation_id"] != first["preparation_id"]
    )
    assert final["package_intents"][1]["inventory"] == first["inventory"]
    assert final["package_receipt"]["preparation_id"] == (
        final["package_intents"][1]["preparation_id"]
    )


def test_a_kana_only_job_finishes_with_a_no_op_reference_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepted cards that bring no kanji are a settled selection, not a hole.

    あげる and もらう are kana, so the character fold over this job's accepted
    collection names nothing. The finish binds the two reference stores' exact
    before-state anyway, proposes no change to either, and completes.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config, scenario="lesson_with_grammar")
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    api = FakeJpdb()

    plan = _plan(config, job, api, words)

    assert sorted(plan.record_ids) == ["word:あげる:あげる", "word:もらう:もらう"]
    assert plan.reference.characters == ()
    assert plan.missing_reference_facts == ()
    assert all(
        plan.reference.file(label).after_text is None
        for label in study_finish.character_notes.REFERENCE_FILE_LABELS
    )
    stores_before = {
        path: path.read_bytes() if path.exists() else None
        for path in (config.kanji_file.resolve(), config.jpdb_readings_file.resolve())
    }

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(plan.record_path)
    assert record["enrichment_receipt"]["reference"]["changed"] == []
    assert record["enrichment_receipt"]["reference"]["missing"] == []
    assert {
        path: path.read_bytes() if path.exists() else None
        for path in (config.kanji_file.resolve(), config.jpdb_readings_file.resolve())
    } == stores_before


# --- the owner's pattern mark, and withdrawing a decision ---------------------


def test_a_standalone_pattern_mark_reaches_the_pattern_store_this_finish_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§9.1's one pattern fact, carried into §7.2's real store write.

    `lesson_with_grammar` really teaches two patterns and its store entry is
    unreviewed. The owner marks it in the standalone `review_patterns` choice —
    the only place that decision is stored — and that single value is what the
    finish reads, what the review batch composes into the one captured store
    payload, what the promotion fold requires the prepared store to carry, and
    what the authority records. The end of the chain is `patterns.json` itself.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config, scenario="lesson_with_grammar")
    before = json.loads(config.patterns_file.read_text(encoding="utf-8"))
    assert before["one.pdf"]["reviewed"] is False
    assert len(before["one.pdf"]["patterns"]) == 2
    saved = _owner_choices(config, job, pattern_choices={"one.pdf": True})
    # The mark lives in one key, and no review selection was saved at all.
    assert saved.choices["review_patterns"]["one.pdf"]["value"] is True
    assert "review_flags" not in saved.choices
    words = RecordingProvider("voicevox", 7)

    plan = _plan(config, job, FakeJpdb(), words)

    assert [part.review_patterns for part in plan.parts] == [True]
    assert plan.authority["parts"][0]["review_patterns"] is True

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(plan.record_path)
    assert record["review_receipt"]["parts"]["one.pdf"] == {
        "record_ids": [],
        "pattern_reviewed": True,
    }
    after = json.loads(config.patterns_file.read_text(encoding="utf-8"))
    assert after["one.pdf"]["reviewed"] is True
    # Exactly that mark moved: the patterns are the extraction's own answer, and
    # this finish neither rewrote nor re-read them.
    assert after["one.pdf"]["patterns"] == before["one.pdf"]["patterns"]
    assert after["one.pdf"]["kind"] == before["one.pdf"]["kind"]


def test_a_false_withdrawn_or_stale_pattern_mark_is_never_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three ways the answer is "no", and none of them is inferred.

    `False` is the owner's stated "not this part's patterns". A withdrawal
    removes the decision through the same writer. A decision taken over a
    rendering that has since moved is *stale*, and the refusal names the control
    the owner acts in — checked here with **no review selection saved for this
    part at all**, so the freshness of the pattern choice is proven on its own
    rather than through a card-flag entry.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config, scenario="lesson_with_grammar")
    words = RecordingProvider("voicevox", 7)
    api = FakeJpdb()

    _owner_choices(config, job, pattern_choices={"one.pdf": False})
    assert [part.review_patterns for part in _plan(config, job, api, words).parts] == [
        False
    ]

    _owner_choices(config, job, pattern_choices={"one.pdf": True})
    assert [part.review_patterns for part in _plan(config, job, api, words).parts] == [
        True
    ]
    withdrawn = _owner_choices(config, job, pattern_choices={"one.pdf": None})
    assert withdrawn.choices["review_patterns"] == {}
    assert [part.review_patterns for part in _plan(config, job, api, words).parts] == [
        False
    ]
    assert json.loads(config.patterns_file.read_text(encoding="utf-8"))["one.pdf"][
        "reviewed"
    ] is False

    _owner_choices(config, job, pattern_choices={"one.pdf": True})
    live = config.staging_dir / "one.pdf.yaml"
    live.write_text(
        live.read_text(encoding="utf-8") + "# an edit nobody bound\n", encoding="utf-8"
    )

    with pytest.raises(study_finish.StudyFinishError) as error:
        _plan(config, job, api, words)

    message = str(error.value)
    assert "The saved review patterns for one.pdf" in message
    assert "janki study review" in message and "--patterns" in message
    assert "review_flags" not in study_job.load_study_job(
        config, job.header.job_id
    ).choices


def test_a_superfluous_coverage_reason_can_be_withdrawn_and_the_job_then_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "Clear it and plan again" route, over a real `not_required` fixture.

    `lesson_with_grammar` is a prose lesson, so `plan_coverage` answers
    `not_required` and a saved coverage reason has nothing to approve: the
    coordinator refuses by name. That refusal used to be unrecoverable — an empty
    map refuses, a blank reason refuses, and a later save omitting the part keeps
    it — so the job could only be finished by hand-editing its document. The same
    CAS writer now withdraws exactly that part's reason, and the job plans and
    completes.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config, scenario="lesson_with_grammar")
    job_id = job.header.job_id
    words = RecordingProvider("voicevox", 7)
    api = FakeJpdb()
    ((part_name, staging_path),) = _parts(config, job_id)
    assert (
        coverage_application.plan_coverage(config, staging_path).state == "not_required"
    )

    saved = study_job.record_choice(
        config,
        job_id,
        {
            "coverage_reasons": {
                part_name: {
                    "job_id": job_id,
                    "part": part_name,
                    "staging_sha256": hashlib.sha256(
                        staging_path.read_bytes()
                    ).hexdigest(),
                    "rendering_fingerprint": study_job.render_job_preview(
                        config, job_id
                    ).rendering_fingerprint,
                    "reason": "I checked this page against the source myself.",
                }
            }
        },
        expected_revision=study_job.load_study_job(config, job_id).revision,
    )

    with pytest.raises(study_finish.StudyFinishError) as error:
        _plan(config, job, api, words)
    message = str(error.value)
    assert "already not required" in message
    assert "Clear it and plan again" in message

    # The routes that are not a withdrawal stay refusals rather than becoming one.
    with pytest.raises(study_job.StudyJobError, match="keyed by published part name"):
        study_job.record_choice(
            config, job_id, {"coverage_reasons": {}}, expected_revision=saved.revision
        )

    cleared = study_job.record_choice(
        config,
        job_id,
        {"coverage_reasons": {part_name: None}},
        expected_revision=saved.revision,
    )

    assert cleared.choices["coverage_reasons"] == {}
    plan = _plan(config, job, api, words)
    assert [part.coverage_reason for part in plan.parts] == [""]
    assert [part.coverage_approved_at for part in plan.parts] == [None]
    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )
    assert result.succeeded
    recorded = _record(plan.record_path)["authority"]["parts"]
    assert [part["coverage_reason"] for part in recorded] == [""]
    assert [part["coverage_approved_at"] for part in recorded] == [None]


# --- held rows, a part that lands nothing, and the owner's disposition --------


class _ScriptedRunner(FakeClaudeRunner):
    """One fake subscription CLI that answers each child with its own reply.

    The suite's runner holds a single reply, and these histories need two
    different extractions in one batch: a page whose rows land and a page whose
    every row the reading witness holds back.
    """

    def __init__(self, replies: list[bytes]) -> None:
        super().__init__(reply=replies[0])
        self._replies = list(replies)

    def spawn(self, command: list[str], **kwargs: Any) -> Any:
        index = min(len(self.spawned), len(self._replies) - 1)
        self.reply = self._replies[index]
        return super().spawn(command, **kwargs)


def _mixed_job(config: ProjectConfig) -> Any:
    """A two-part job: `one.pdf` lands three rows, `two.pdf` lands none.

    `reading_holds` proposes 走る under わしる, a reading jpdb does not list, so the
    promotion writer keeps that row in the live staging file with its reason
    rather than landing or deleting it.
    """
    deck = _deck(config)
    job = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=_source(config, "parent.pdf", b"parent"),
        deck_path=deck,
    )
    # `reading_holds` also proposes 泊まる with no reading at all, which the deck
    # refuses to validate — so that row makes the job's own card preview
    # undrawable and there is no rendering for an owner decision to bind to.
    # This history is about a row jpdb *cannot place*, so only that candidate is
    # kept: 走る under わしる, a reading jpdb does not list.
    holds = _answer("reading_holds")
    holds["candidates"] = [
        candidate
        for candidate in holds["candidates"]
        if str(candidate.get("reading") or "").strip()
    ]
    assert [candidate["reading"] for candidate in holds["candidates"]] == ["わしる"]
    runner = _ScriptedRunner([_stream(_answer()), _stream(holds)])
    sources = [
        _source(config, "one.pdf", b"one"),
        _source(config, "two.pdf", b"two"),
    ]
    plan = extraction_batch.plan_extraction_batch(
        config,
        sources,
        job_id=job.header.job_id,
        destination_deck=deck,
        concurrency_limit=1,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )
    outcome = study_job.dispatch_job_batch(
        config,
        job.header.job_id,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    assert [child.state for child in outcome.children] == ["committed", "committed"]
    _assign(config, deck)
    return study_job.load_study_job(config, job.header.job_id)


def test_a_part_that_lands_nothing_needs_the_owners_disposition_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§9.6.2–3: held rows keep the job incomplete until the owner disposes of them.

    `two.pdf` lands nothing: its one row proposes a reading jpdb does not list,
    so the writer holds it in the live staging file with that reason. The finish
    refuses, names the control that records the decision, and authorizes nothing.
    After the owner defers that part — in their own words — the job completes over
    the three cards it did accept, and the held row is still on disk.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _mixed_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)

    plan = _plan(config, job, _dictionary(), words)

    by_part = {part.part_name: part for part in plan.parts}
    assert by_part["one.pdf"].projected_state == "lands"
    assert by_part["two.pdf"].projected_state == "nothing_lands"
    assert by_part["two.pdf"].held_ids == ("word:走る:わしる",)
    assert sorted(plan.record_ids) == [
        "word:走る:はしる",
        "word:食べる:たべる",
        "word:飲む:のむ",
    ]
    assert [part.part_name for part in plan.undisposed_parts] == ["two.pdf"]
    with pytest.raises(study_finish.StudyFinishError) as error:
        study_finish.execute_study_finish(
            config, plan, word_provider=words, sentence_provider=words
        )
    assert "janki study disposition" in str(error.value)
    assert "two.pdf" in str(error.value)
    assert not plan.record_path.exists(), "a refused finish authorized nothing"
    assert _canonical(config) == {}

    _owner_choices(
        config,
        job,
        dispositions={
            "two.pdf": {
                "action": "defer",
                "reason": "This page repeats lesson 12; I will read it with that one.",
            }
        },
    )
    disposed = _plan(config, job, _dictionary(), words)

    assert disposed.undisposed_parts == ()
    assert disposed.fingerprint != plan.fingerprint
    result = study_finish.execute_study_finish(
        config, disposed, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    assert result.held_ids == ("word:走る:わしる",)
    record = _record(disposed.record_path)
    outcomes = {
        entry["part_name"]: entry for entry in record["promotion_receipt"]["parts"]
    }
    assert outcomes["one.pdf"]["state"] == "landed"
    assert outcomes["two.pdf"]["landed_ids"] == []
    assert outcomes["two.pdf"]["disposition"]["action"] == "defer"
    assert outcomes["two.pdf"]["disposition"]["reason"].startswith("This page repeats")
    assert sorted(_canonical(config)) == [
        "word:走る:はしる",
        "word:食べる:たべる",
        "word:飲む:のむ",
    ]
    # Deferred, not deleted: the row is still in the live staging file with the
    # hold reason its own writer recorded.
    live = config.staging_dir / "two.pdf.yaml"
    assert live.is_file()
    held_records, _meta = staging.read_staging_text(
        live.read_text(encoding="utf-8"), source=str(live)
    )
    assert [item.id for item in held_records] == ["word:走る:わしる"]
    assert all(item.source.raw_fields.get("hold_reason") for item in held_records)


# --- a paid clip's two independent witnesses ----------------------------------


def test_a_paid_clip_saves_its_expected_request_fingerprint_before_it_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writers, two values, and the proof is where they meet.

    This coordinator derives the request fingerprint from the approved request
    and the resolved voice profile and saves it durably at the audio writer's own
    `before_paid_dispatch` seam — before the transport sees anything. The paid
    writer independently records its own attempt on the clip's ledger row. The
    completion proof matches one against the other; a reservation id on its own
    proves nothing, and neither does the journal's later silence.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    transport = _RealtimeTransport()
    sentences = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=transport,
        operations_path=config.operations_file,
    )

    plan = _plan(config, job, _dictionary(), words, sentences)

    assert plan.audio.paid_example_calls == 6
    assert plan.audio.paid_word_calls == 0
    assert plan.audio.example_provider is not None
    assert plan.audio.example_provider["access"] == "paid-network"

    saved_before_dispatch: list[tuple[int, int]] = []
    real_reserve = study_finish._reserve_paid_clip

    def observe(*args: Any, **kwargs: Any) -> Any:
        outcome = real_reserve(*args, **kwargs)
        # Durable *before* the call leaves: the record on disk already holds this
        # clip's reservation while the transport has seen fewer requests than
        # there are reservations.
        held = _record(plan.record_path)["paid_clip_reservations"]
        saved_before_dispatch.append((len(held), len(transport.calls)))
        return outcome

    monkeypatch.setattr(study_finish, "_reserve_paid_clip", observe)

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=sentences
    )

    assert result.succeeded
    assert len(transport.calls) == 6
    assert [reservations for reservations, _calls in saved_before_dispatch] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    for reservations, calls in saved_before_dispatch:
        assert calls == reservations - 1, "a reservation landed after its call"

    record = _record(plan.record_path)
    reserved = {
        item["target"]: item["expected_request_fp"]
        for item in record["paid_clip_reservations"]
    }
    assert len(reserved) == 6
    paid = {
        slot["target"]: slot["paid_operation"]
        for slot in record["audio_receipt"]["proof"]["slots"]
        if slot["paid_operation"] is not None
    }
    assert set(paid) == set(reserved)
    for target, attempt in paid.items():
        assert attempt["expected_request_fp"] == reserved[target]
        assert attempt["model"] == openai_realtime.MODEL
    assert record["audio_receipt"]["paid_clip_count"] == 6
    # The native writer's successful cleanup really did happen: no capture is
    # retained and the journal holds nothing for these clips.
    from japanese_anki import operations

    journal = operations.OperationJournal.load(config.operations_file)
    assert [
        entry
        for entry in journal.operations.values()
        if entry.kind == "audio-realtime"
    ] == []


def test_a_saved_expected_fingerprint_that_does_not_match_the_attempt_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The witness is the comparison, so a wrong saved value is not accepted.

    A single anchored mutation of the value this coordinator derives — one
    character of the request fingerprint — has to stop the dispatch. It does, at
    the reservation itself: the entry the paid provider authorized describes a
    different request from the one being saved, so nothing is billed and nothing
    is proven. If the saved value were decoration this would sail through.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    transport = _RealtimeTransport()
    sentences = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=transport,
        operations_path=config.operations_file,
    )
    plan = _plan(config, job, _dictionary(), words, sentences)
    real_expected = study_finish._expected_request_fp

    def wrong(*args: Any, **kwargs: Any) -> str:
        found = real_expected(*args, **kwargs)
        return ("0" if found[0] != "0" else "1") + found[1:]

    monkeypatch.setattr(study_finish, "_expected_request_fp", wrong)

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=sentences
    )

    # Refused at the last durable boundary before transport, so the provider was
    # never asked: a value that cannot be the witness stops the call rather than
    # being saved and reconciled afterwards.
    assert transport.calls == []
    assert not result.succeeded
    assert result.outstanding
    record = _record(plan.record_path)
    assert record["state"] == "enriched"
    assert record["audio_receipt"] is None
    assert record["paid_clip_reservations"] == []
    from japanese_anki import ledger as ledger_module

    assert ledger_module.load(config.ledger_file).pending_audio == {}


def test_a_confirmed_new_clip_another_run_supplied_refuses_with_a_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§7.10: this finish cannot claim media it never dispatched.

    The confirmation said these clips were new paid work. They are now on disk
    with no reservation of this finish's own, so nothing here proves *this* call
    produced them. The refusal names the route — plan again, and the fresh plan
    truthfully reports them as reused — rather than forcing a second call,
    inferring an approval, or attributing another run's bytes to this job.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    transport = _RealtimeTransport()
    sentences = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=transport,
        operations_path=config.operations_file,
    )
    plan = _plan(config, job, _dictionary(), words, sentences)

    # Another authorized run finishes the same clips first: the ordinary audio
    # writer, over the same canonical state, under its own lock.
    def land_the_clips_elsewhere() -> None:
        # The finish already holds `.janki-audio-operation`, which is exactly the
        # entry point this writer documents for a caller that owns that lock, so
        # nothing here takes it a second time.
        study_finish.audio_application.execute_targeted_audio_locked(
            config,
            plan.record_ids,
            words=True,
            examples=True,
            force=False,
            prune=False,
            chosen_provider="voicevox",
            word_provider=words,
            sentence_provider=sentences,
        )

    real_enriched = study_finish._apply_enriched

    def after_enrichment(*args: Any, **kwargs: Any) -> Any:
        outcome = real_enriched(*args, **kwargs)
        land_the_clips_elsewhere()
        return outcome

    monkeypatch.setattr(study_finish, "_apply_enriched", after_enrichment)

    with pytest.raises(study_finish.StudyFinishStaleAudioPlanError) as error:
        study_finish.execute_study_finish(
            config, plan, word_provider=words, sentence_provider=sentences
        )

    message = str(error.value)
    assert "Plan this finish again" in message
    assert "reused" in message
    held = _record(plan.record_path)
    assert held["state"] == "enriched"
    assert held["audio_receipt"] is None
    assert held["paid_clip_reservations"] == []


# --- the locks, and a component whose before-state moved ----------------------


def test_the_shared_curation_guard_is_held_before_any_finish_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.5: the curation guard comes first, and the finish really waits for it.

    While another staging mutation holds the guard, this finish writes no staging
    byte and advances no phase. The observable is the artifact: the part's staging
    document is untouched and the record still says `authorized`.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    plan = _plan(config, job, _dictionary(), words)
    live = config.staging_dir / "one.pdf.yaml"
    staged_before = live.read_bytes()

    finished = threading.Event()
    outcome: list[Any] = []

    def run() -> None:
        try:
            outcome.append(
                study_finish.execute_study_finish(
                    config, plan, word_provider=words, sentence_provider=words
                )
            )
        except BaseException as exc:  # noqa: BLE001 - reported through `outcome`
            outcome.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=run, daemon=True)
    try:
        with study_curation.staging_curation_guard(config.staging_dir):
            worker.start()
            assert not finished.wait(1.5), "the finish ran while a curation held it"
            assert live.read_bytes() == staged_before
            assert _record(plan.record_path)["state"] == "authorized"
        assert finished.wait(BOUND), "the finish never returned"
    finally:
        worker.join(BOUND)

    assert len(outcome) == 1 and not isinstance(outcome[0], BaseException), outcome
    assert outcome[0].succeeded


def test_a_component_whose_before_state_moved_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A third state is a refusal, not a merge.

    The authority binds the staged review's exact before-bytes. Something else
    edits that file after the authority is recorded, so what is on disk is
    neither the state this finish bound nor the state it would write. The review
    writer's compare-and-swap refuses, and the job stays exactly where it was.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    plan = _plan(config, job, _dictionary(), words)
    live = config.staging_dir / "one.pdf.yaml"
    third = live.read_text(encoding="utf-8") + "# an edit nobody bound\n"
    live.write_text(third, encoding="utf-8")

    with pytest.raises(JankiError):
        study_finish.execute_study_finish(
            config, plan, word_provider=words, sentence_provider=words
        )

    assert live.read_text(encoding="utf-8") == third
    assert _record(plan.record_path)["state"] == "authorized"
    assert _canonical(config) == {}


# --- what a study job's own store does with a finish -------------------------


def test_a_job_resolves_and_resumes_the_finish_it_reserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reserved receipt id resolves to the authority at that exact path.

    The finish record is named for the SHA-256 of the authority it carries, so
    the reserved id *is* the path. The job's resume hands the receipt back to the
    finish service — the only writer — and closes its intent only once that
    service reports a complete state.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    plan = _plan(config, job, _dictionary(), words)
    job_id = job.header.job_id

    def stop_before_packaging(*_args: Any, **_kwargs: Any) -> Any:
        raise study_finish.StudyFinishError("stopped before the package")

    monkeypatch.setattr(study_finish, "_apply_packaged", stop_before_packaging)
    with pytest.raises(
        study_finish.StudyFinishError, match="stopped before the package"
    ):
        study_finish.execute_study_finish(
            config, plan, word_provider=words, sentence_provider=words
        )
    monkeypatch.undo()
    _no_api(monkeypatch)

    current = study_job.load_study_job(config, job_id)
    intent = study_job.ActionIntent(
        intent_id=study_job.new_intent_id(),
        kind="finish",
        decided_at="2026-09-12T12:00:00+00:00",
        reserves={"receipt_id": plan.fingerprint},
        bindings={},
    )
    study_job.append_intent(
        config, job_id, intent, expected_revision=current.revision
    )

    references = study_job.discover_actions(config, job_id)
    finish_reference = next(item for item in references if item.kind == "finish")
    assert finish_reference.path == plan.record_path
    assert finish_reference.sha256 == hashlib.sha256(
        plan.record_path.read_bytes()
    ).hexdigest()
    assert finish_reference.reserves["receipt_id"] == plan.fingerprint

    actions = study_job.resume_job_actions(
        config,
        job_id,
        finish_options={
            "chosen_provider": "voicevox",
            "word_provider": words,
            "sentence_provider": words,
        },
    )

    finish_action = next(item for item in actions if item.kind == "finish")
    assert finish_action.closed is True
    assert plan.fingerprint[:12] in finish_action.detail
    # The number it reports is the package receipt's whole-deck expansion, so it
    # says so: this job contributed three notes, and the deck builds six cards.
    assert "whole deck" in finish_action.detail
    assert str(study_finish.inspect_study_finish(config, plan.fingerprint).card_count) in (
        finish_action.detail
    )
    closed = study_job.load_study_job(config, job_id)
    outcome = next(
        entry for entry in closed.outcomes if entry.intent_id == intent.intent_id
    )
    assert outcome.state == "applied"
    assert outcome.consequences == {"state": "complete", "resumed": True}
    assert study_finish.inspect_study_finish(config, plan.fingerprint).succeeded
    assert [
        item.receipt_id
        for item in study_finish.list_study_finishes(config, job_id=job_id)
    ] == [plan.fingerprint]


def test_two_stored_slots_that_speak_the_same_sentence_share_one_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several expected slots legitimately reach one clip; each still has to.

    One row proposes the same sentence twice, at two speech levels. That is two
    *stored* example slots and one *unique* clip request, and the census checks
    the slots against the plan rather than counting the plan's clips — which is
    exactly what stops a short request list passing for complete coverage.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    duplicated = _answer()
    first = duplicated["candidates"][0]
    first["examples"][1]["japanese"] = first["examples"][0]["japanese"]
    first["examples"][1]["furigana"] = first["examples"][0]["furigana"]
    job = _settled_job(config, answer=duplicated)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)

    plan = _plan(config, job, _dictionary(), words)

    assert plan.audio.expected_example_slots == 6
    assert plan.audio.example_counts.total == 5, "the repeated sentence is one clip"
    assert plan.audio.unique_clip_requests == 8

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(plan.record_path)
    # Nine stored slots, eight clips: the proof keeps the two scopes apart, and
    # the packaged media count is the renderer's own answer again.
    assert record["audio_receipt"]["stored_slot_count"] == 9
    assert record["audio_receipt"]["clip_count"] == 8
    assert record["package_receipt"]["media_count"] == 8
    assert record["package_receipt"]["audio_selection"]["stored_sentence_slots"] == 6
    assert record["package_receipt"]["audio_selection"]["sentence_clips"] == 5
    canonical = _canonical(config)
    repeated = canonical[plan.record_ids[0]]["examples"]
    assert repeated[0]["audio"] == repeated[1]["audio"]


def test_a_reservation_the_journal_proves_went_unsent_is_replaced_by_its_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One current attempt per target, and the journal is what says it never sent.

    The transport fails before the provider could connect, so the paid writer's
    own entry ends `failed_before_send` — terminal, and nothing that may have been
    billed. Only on that evidence may a resume reserve the same target again, and
    the record then carries the *successor*, not a history list: B's paid evidence
    rule takes the current reservation per target.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)

    # One transport class throughout: the confirmed authority binds the resolved
    # voice profile, and that profile names the transport implementation — so a
    # resume through a *different* transport type is refused as exceeding it,
    # which is a separate rule from the one under test here.
    transport = _FlakyTransport()
    sentences = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=transport,
        operations_path=config.operations_file,
    )
    plan = _plan(config, job, _dictionary(), words, sentences)

    first = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=sentences
    )

    assert not first.succeeded
    assert len(transport.calls) == 1
    held = _record(plan.record_path)
    assert held["state"] == "enriched"
    assert len(held["paid_clip_reservations"]) == 1
    stranded = held["paid_clip_reservations"][0]

    from japanese_anki import operations

    entry = operations.OperationJournal.load(config.operations_file).operations[
        stranded["operation_id"]
    ]
    assert entry.state == "failed_before_send"
    assert entry.state in operations.TERMINAL_STATES
    assert not entry.money_may_have_been_spent

    transport.mode = ""
    result = study_finish.resume_study_finish(
        config, plan.fingerprint, word_provider=words, sentence_provider=sentences
    )

    assert result.succeeded
    final = _record(plan.record_path)
    same_target = [
        item
        for item in final["paid_clip_reservations"]
        if item["target"] == stranded["target"]
    ]
    assert len(same_target) == 1, "the superseded attempt was kept as a duplicate"
    assert same_target[0]["operation_id"] != stranded["operation_id"]
    proven = {
        slot["target"]: slot["paid_operation"]
        for slot in final["audio_receipt"]["proof"]["slots"]
        if slot["paid_operation"] is not None
    }
    assert proven[stranded["target"]]["operation_id"] == (
        same_target[0]["operation_id"]
    )
    assert proven[stranded["target"]]["expected_request_fp"] == (
        same_target[0]["expected_request_fp"]
    )


def test_a_reservation_whose_outcome_is_unknown_is_never_re_sent_automatically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§7.10: a call that may already have been billed is a person's decision.

    The provider connects and then stops answering, so its own entry ends
    `outcome_unknown` — terminal for dispatch, and explicitly a state where money
    may have been spent. The resume refuses by name, names the state and the
    route, and sends nothing. The point of the read-only reconciliation is the
    last assertion: this coordinator does not advance or forget another writer's
    journal entry to get past it.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    transport = _FlakyTransport("mid_stream")
    sentences = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=transport,
        operations_path=config.operations_file,
    )
    plan = _plan(config, job, _dictionary(), words, sentences)

    first = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=sentences
    )

    assert not first.succeeded
    assert len(transport.calls) == 1
    held = _record(plan.record_path)
    assert held["state"] == "enriched"
    assert len(held["paid_clip_reservations"]) == 1
    stranded = held["paid_clip_reservations"][0]

    from japanese_anki import operations

    before = operations.OperationJournal.load(config.operations_file)
    entry = before.operations[stranded["operation_id"]]
    assert entry.state == "outcome_unknown"
    assert entry.money_may_have_been_spent

    transport.mode = ""
    with pytest.raises(study_finish.StudyFinishError) as error:
        study_finish.resume_study_finish(
            config,
            plan.fingerprint,
            word_provider=words,
            sentence_provider=sentences,
        )

    message = str(error.value)
    assert stranded["target"] in message
    assert "outcome_unknown" in message
    assert "janki operations" in message
    assert len(transport.calls) == 1, "the resume dispatched the clip again"
    after = operations.OperationJournal.load(config.operations_file)
    assert after.operations == before.operations, (
        "this coordinator kept another writer's books"
    )
    assert _record(plan.record_path)["paid_clip_reservations"] == [stranded]


# --- parts that write no canonical byte, and plural per-part receipts ---------


def _scripted_job(
    config: ProjectConfig, sources: tuple[tuple[str, str], ...]
) -> Any:
    """One job whose children really ran, each source with its own scenario."""
    deck = _deck(config)
    job = study_job.open_study_job(
        config,
        kind="source_extraction",
        parent_source=_source(config, "parent.pdf", b"parent"),
        deck_path=deck,
    )
    runner = _ScriptedRunner(
        [_stream(_answer(scenario)) for _name, scenario in sources]
    )
    paths = [_source(config, name, name.encode("ascii")) for name, _scenario in sources]
    plan = extraction_batch.plan_extraction_batch(
        config,
        paths,
        job_id=job.header.job_id,
        destination_deck=deck,
        concurrency_limit=1,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
    )
    outcome = study_job.dispatch_job_batch(
        config,
        job.header.job_id,
        plan,
        provider_env={},
        provider_runner=runner,
        provider_which=_which,
        provider_spawn=runner.spawn,
    )
    assert [child.state for child in outcome.children] == ["committed"] * len(sources)
    _assign(config, deck)
    return study_job.load_study_job(config, job.header.job_id)


def _finish_capturing_archived_wire(
    config: ProjectConfig,
    plan: study_finish.StudyFinishPlan,
    words: Any,
) -> tuple[study_finish.StudyFinishResult, dict[str, bytes]]:
    """Run one whole finish, keeping the exact bytes each part archived from.

    Those bytes are what §7.7's recoverable boundary leaves behind when the
    archive write succeeds and the live prune does not, so they are the only
    honest way to build an archive-retry history for this coordinator: the
    reviewed document the promotion writer really consumed, not a hand-made one.
    """
    captured: dict[str, bytes] = {}
    real_apply = study_finish.promotion.apply_prepared_source_promotion

    def capture(config_: Any, prepared: Any, **kwargs: Any) -> Any:
        captured[prepared.part_name] = Path(prepared.staging_path).read_bytes()
        return real_apply(config_, prepared, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(
        study_finish.promotion, "apply_prepared_source_promotion", capture
    )
    try:
        result = study_finish.execute_study_finish(
            config, plan, word_provider=words, sentence_provider=words
        )
    finally:
        patch.undo()
    return result, captured


def _withdraw_coverage_reason(config: ProjectConfig, job_id: str, part: str) -> None:
    """Drop a reason a restored review no longer offers, through the CAS writer.

    A review the promotion writer already approved carries that approval in its
    own metadata, so `plan_coverage` answers `already_resolved` for it and the
    saved reason has nothing left to approve. The owner's route out is the
    withdrawal their editor writes, never an edited job document.
    """
    saved = study_job.load_study_job(config, job_id)
    if part in (saved.choices.get("coverage_reasons") or {}):
        study_job.record_choice(
            config,
            job_id,
            {"coverage_reasons": {part: None}},
            expected_revision=saved.revision,
        )


def _restore_live_review(
    path: Path, wire: bytes, *, keep: tuple[str, ...] | None = None
) -> None:
    """Put back the live review its promotion archived from, optionally narrowed.

    Narrowing rewrites the same document through the staging writer with its own
    metadata: `_candidate_accounting_retry_targets` states in so many words that
    a reviewer may delete a proposal — what it refuses is a population larger
    than the immutable candidate account.
    """
    path.write_bytes(wire)
    if keep is None:
        return
    records, meta = staging.read_staging_text(
        path.read_text(encoding="utf-8"), source=str(path)
    )
    staging.write_staging(
        path, [record for record in records if record.id in keep], meta, force=True
    )


def _receipt_selection(record: dict[str, Any]) -> dict[str, list[tuple[str, list[str]]]]:
    """Part name → the (receipt, selected ids) pairs its promotion receipt binds."""
    return {
        str(entry["part_name"]): [
            (str(item["receipt_id"]), sorted(str(one) for one in item["selected_ids"]))
            for item in entry["receipts"]
        ]
        for entry in record["promotion_receipt"]["parts"]
    }


def test_a_pattern_only_part_promotes_beside_two_landing_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§7.6/§7.7: the states that write no canonical byte bind no canonical component.

    `pattern_only_chart` is a real zero-record rich extraction: its promotion
    writes an archive and retires the live review, and `prepare_source_promotion`
    binds exactly `(archive, live_staging)` for it. Demanding a canonical
    component of every part refuses this whole job, so the two parts that *do*
    land never land either. The part accepts no card, so §9.6.2 still wants the
    owner's disposition and fabricates no receipt for it.

    `nothing` is deliberately absent from this history and cannot be built from a
    settled study-job child: that state needs a zero-record file with **no** rich
    extraction run, and the extraction writer always records one — asserted below
    rather than asserted about.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _scripted_job(
        config,
        (
            ("one.pdf", SCENARIO),
            ("two.pdf", "pattern_only_chart"),
            ("three.pdf", "lesson_with_grammar"),
        ),
    )
    chart = config.staging_dir / "two.pdf.yaml"
    charted, chart_meta = staging.read_staging_text(
        chart.read_text(encoding="utf-8"), source=str(chart)
    )
    assert charted == []
    assert (
        study_finish.promotion.rich_extraction_review_run_id(chart_meta) is not None
    ), "a settled child always carries a rich extraction run, so `nothing` is unreachable"
    # The owner reviews this chart's grammar in the patterns control first: the
    # coverage editor's own promotion preflight blocks at the `patterns` gate
    # until they do, so this part cannot even be planned before that decision.
    store = patterns.load_store(config.patterns_file)
    assert store["two.pdf"].reviewed is False
    patterns.save_store(
        config.patterns_file,
        {**store, "two.pdf": replace(store["two.pdf"], reviewed=True)},
    )
    _owner_choices(
        config,
        job,
        dispositions={
            "two.pdf": {
                "action": "exclude",
                "reason": "This chart teaches grammar and proposes no card for this job.",
            }
        },
    )
    words = RecordingProvider("voicevox", 7)

    plan = _plan(config, job, _dictionary(), words)

    by_part = {part.part_name: part for part in plan.parts}
    assert by_part["one.pdf"].projected_state == "lands"
    assert by_part["two.pdf"].projected_state == "pattern_only"
    assert by_part["three.pdf"].projected_state == "lands"
    # The carry a part that writes nothing leaves untouched is its own binding.
    assert by_part["two.pdf"].expected_before == by_part["two.pdf"].expected_after
    assert by_part["two.pdf"].landed_ids == ()
    assert plan.undisposed_parts == ()

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(plan.record_path)
    outcomes = {
        entry["part_name"]: entry for entry in record["promotion_receipt"]["parts"]
    }
    assert outcomes["two.pdf"]["state"] == "pattern_only"
    assert outcomes["two.pdf"]["landed_ids"] == []
    assert outcomes["two.pdf"]["receipts"] == []
    assert outcomes["two.pdf"]["disposition"]["action"] == "exclude"
    assert (config.root / outcomes["two.pdf"]["archive_path"]).is_file()
    assert json.loads(config.patterns_file.read_text(encoding="utf-8"))["two.pdf"][
        "reviewed"
    ] is True

    # Each landing part binds its own one receipt, and the two selections
    # partition this job's accepted identities exactly.
    selections = _receipt_selection(record)
    assert len(selections["one.pdf"]) == 1 and len(selections["three.pdf"]) == 1
    assert sorted(
        record_id
        for entries in selections.values()
        for _receipt, ids in entries
        for record_id in ids
    ) == sorted(plan.record_ids)
    assert selections["one.pdf"][0][0] != selections["three.pdf"][0][0]
    assert sorted(_canonical(config)) == sorted(plan.record_ids)


def test_a_whole_archive_retry_part_binds_the_receipt_that_already_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§7.7/§9.6.2: an archive retry accepts cards it never landed itself.

    The first finish archives and lands; the live review it archived from is put
    back, which is the exact state a crash between the archive write and the
    prune leaves. The second finish's part therefore writes no canonical byte and
    lands nothing — every one of its accepted ids is already in its own archive —
    so its scope has to come from the **prior** receipt, resolved through the
    owning archive reader. Comparing `landed_ids` alone is vacuously true here.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    live = config.staging_dir / "one.pdf.yaml"

    first_plan = _plan(config, job, _dictionary(), words)
    first, captured = _finish_capturing_archived_wire(config, first_plan, words)

    assert first.succeeded
    assert not live.exists()
    prior = _record(first_plan.record_path)["promotion_receipt"]["parts"][0][
        "receipt_id"
    ]
    _restore_live_review(live, captured["one.pdf"])
    _withdraw_coverage_reason(config, job.header.job_id, "one.pdf")
    _owner_choices(config, job)

    plan = _plan(config, job, _dictionary(), words)

    part = plan.parts[0]
    assert part.projected_state == "archive_retry"
    assert part.landed_ids == ()
    assert sorted(part.archive_retry_ids) == sorted(first_plan.record_ids)
    assert part.expected_before == part.expected_after
    assert sorted(plan.record_ids) == sorted(first_plan.record_ids)

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(plan.record_path)
    entry = record["promotion_receipt"]["parts"][0]
    assert entry["state"] == "archive_retry"
    assert entry["landed_ids"] == []
    assert _receipt_selection(record)["one.pdf"] == [
        (prior, sorted(plan.record_ids))
    ]
    assert sorted(
        row[0] for row in record["promotion_receipt"]["projection"]
    ) == sorted(plan.record_ids)
    assert not live.exists()


def test_a_selection_strictly_smaller_than_a_prior_receipt_widens_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prior receipt may hold more ids than this job's live retry selects.

    The restored review keeps two of the three rows its own archive receipted —
    an ordinary reviewer deletion the candidate account permits. The third id is
    still in the collection and still in the whole deck, and it is **not** this
    job's: it may not enter the enrichment, audio, package or preview selection,
    and the receipt has to record which ids inside that prior receipt were taken.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    live = config.staging_dir / "one.pdf.yaml"

    first_plan = _plan(config, job, _dictionary(), words)
    first, captured = _finish_capturing_archived_wire(config, first_plan, words)

    assert first.succeeded
    prior = _record(first_plan.record_path)["promotion_receipt"]["parts"][0][
        "receipt_id"
    ]
    kept = tuple(sorted(first_plan.record_ids)[:2])
    left_out = sorted(first_plan.record_ids)[2]
    _restore_live_review(live, captured["one.pdf"], keep=kept)
    _withdraw_coverage_reason(config, job.header.job_id, "one.pdf")
    _owner_choices(config, job)

    plan = _plan(config, job, _dictionary(), words)

    assert plan.parts[0].projected_state == "archive_retry"
    assert sorted(plan.parts[0].archive_retry_ids) == sorted(kept)
    assert sorted(plan.record_ids) == sorted(kept)
    assert left_out not in plan.record_ids
    assert left_out not in plan.authority["audio"]["record_ids"]
    assert left_out not in plan.authority["enrichment"]["record_ids"]

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(plan.record_path)
    assert _receipt_selection(record)["one.pdf"] == [(prior, sorted(kept))]
    assert [row[0] for row in record["promotion_receipt"]["projection"]] == sorted(
        kept
    ) or sorted(row[0] for row in record["promotion_receipt"]["projection"]) == sorted(
        kept
    )
    assert left_out not in {row[0] for row in record["promotion_receipt"]["projection"]}
    # The card the selection left out is still a card of this deck, drawn by the
    # package but not by this job's preview.
    assert left_out in _canonical(config)
    preview = (config.root / record["preview_receipt"]["preview_path"]).read_text(
        encoding="utf-8"
    )
    assert f'data-record-id="{left_out}"' not in preview
    assert record["preview_receipt"]["note_count"] == 2
    assert record["preview_receipt"]["deck_note_count"] == 3


def test_one_part_binds_both_its_old_and_its_new_promotion_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§9.6.2's receipts are plural per part, and each one's selection is exact.

    The first finish promotes a two-row live review, so its archive holds one
    batch. The review is then put back with all three of its own rows, and the
    second finish lands the third while the first two are retries of that earlier
    batch. One part, two source-bound receipts, and the two selections together
    account for exactly the ids the authority accepted for it.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    words = RecordingProvider("voicevox", 7)
    live = config.staging_dir / "one.pdf.yaml"
    whole = live.read_bytes()
    every_id = sorted(
        record.id
        for record in staging.read_staging_text(
            whole.decode("utf-8"), source=str(live)
        )[0]
    )
    _restore_live_review(live, whole, keep=tuple(every_id[:2]))
    _owner_choices(config, job)

    first_plan = _plan(config, job, _dictionary(), words)
    first, captured = _finish_capturing_archived_wire(config, first_plan, words)

    assert first.succeeded
    assert sorted(first_plan.record_ids) == every_id[:2]
    old = _record(first_plan.record_path)["promotion_receipt"]["parts"][0]["receipt_id"]

    # The same document again, with the row the first pass never promoted: the
    # two archived rows are retries and the third is new work.
    _kept, reviewed_meta = staging.read_staging_text(
        captured["one.pdf"].decode("utf-8"), source=str(live)
    )
    records, _meta = staging.read_staging_text(whole.decode("utf-8"), source=str(live))
    staging.write_staging(live, records, reviewed_meta, force=True)
    _withdraw_coverage_reason(config, job.header.job_id, "one.pdf")
    _owner_choices(config, job)

    plan = _plan(config, job, _dictionary(), words)

    part = plan.parts[0]
    assert part.projected_state == "lands"
    assert sorted(part.landed_ids) == every_id[2:]
    assert sorted(part.archive_retry_ids) == every_id[:2]
    assert sorted(plan.record_ids) == every_id

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(plan.record_path)
    entry = record["promotion_receipt"]["parts"][0]
    assert entry["state"] == "landed"
    assert entry["landed_ids"] == every_id[2:]
    bound = dict(_receipt_selection(record)["one.pdf"])
    assert len(bound) == 2
    assert bound[old] == every_id[:2]
    new = next(receipt for receipt in bound if receipt != old)
    assert bound[new] == every_id[2:]
    assert sorted(row[0] for row in record["promotion_receipt"]["projection"]) == every_id

    # The union is not the claim. Crossing the two selections keeps every
    # accepted id accounted for once and still puts each of them in a receipt
    # that never promoted it, which §9.6.2's re-derivation refuses.
    study_finish._assert_complete(config, record)
    crossed = json.loads(json.dumps(record))
    entries = crossed["promotion_receipt"]["parts"][0]["receipts"]
    entries[0]["selected_ids"], entries[1]["selected_ids"] = (
        entries[1]["selected_ids"],
        entries[0]["selected_ids"],
    )
    with pytest.raises(study_finish.StudyFinishError) as error:
        study_finish._assert_complete(config, crossed)
    assert "does not promote them" in str(error.value)
    assert every_id[0] in str(error.value)


def test_an_unrelated_invalid_done_archive_leaves_this_source_outstanding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken done archive refuses, and it refuses as this part's diagnostic.

    `finish.list_finish_receipts` and `finish.resolve_finish_scope` both validate
    the *whole* `staging/done/` namespace on every call, and that global refusal
    stays: an archive janki cannot validate is not proven harmless by a lookup
    that happened not to need it. What this part's lookup owes is the **shape** of
    the refusal — the coordinator's own source-specific outstanding diagnostic,
    naming the part, its archive and what the owning reader refused, rather than a
    raw `FinishScopeError` escaping the promotion fold untyped.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)

    plan = _plan(config, job, _dictionary(), words)

    # Not this part's archive, not a receipt, and not readable as a staging
    # document: exactly the entry `finish._done_archive_batches` refuses. Planted
    # after the plan, because that is the only honest way an archive this job
    # never looked at becomes invalid between planning and promoting.
    broken = config.staging_dir / "done" / "unrelated.yaml"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("records: [\n", encoding="utf-8")

    with pytest.raises(study_finish.StudyFinishError) as error:
        study_finish.execute_study_finish(
            config, plan, word_provider=words, sentence_provider=words
        )

    message = str(error.value)
    assert not isinstance(
        error.value, study_finish.finish_application.FinishScopeError
    ), message
    assert "one.pdf" in message
    assert "[finish-archive-invalid]" in message and broken.name in message
    assert "The job stays outstanding." in message
    # Honest about what really happened: the promotion landed, and the receipt
    # stays at the phase whose accounting refused.
    record = _record(plan.record_path)
    assert record["state"] == "reviewed"
    assert sorted(_canonical(config)) == sorted(plan.record_ids)


def _lookup_resolutions(patch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """What each part's own receipt lookup paid a full resolve for.

    The spy delegates to `finish.resolve_finish_scope` itself — nothing here
    answers for the owning archive reader or fabricates a scope — and slices its
    calls by the part whose lookup was running, so the measurement is this part's
    own cost rather than a whole-run total the later phases also contribute to.
    Each resolve is a second full archive scan plus a canonical read plus a
    deck-ownership re-proof under the deck lock.
    """
    real_resolve = study_finish.finish_application.resolve_finish_scope
    real_lookup = study_finish._part_selections
    resolved: list[str] = []
    seen: dict[str, dict[str, Any]] = {}

    def resolve(config: Any, receipt_id: str) -> Any:
        resolved.append(receipt_id)
        return real_resolve(config, receipt_id)

    def lookup(config: Any, **kwargs: Any) -> Any:
        first = len(resolved)
        try:
            return real_lookup(config, **kwargs)
        finally:
            seen[str(kwargs["part_name"])] = {
                "primary": kwargs["primary"],
                "resolved": list(resolved[first:]),
            }

    patch.setattr(study_finish.finish_application, "resolve_finish_scope", resolve)
    patch.setattr(study_finish, "_part_selections", lookup)
    return seen


def _two_source_job(config: ProjectConfig) -> Any:
    """One job whose two children really ran, the unrelated archive sorting first.

    `earlier.pdf.yaml` sorts ahead of `later.pdf.yaml`, which is the order
    `finish.list_finish_receipts` enumerates in, so every receipt of the
    unrelated source is offered to `later.pdf`'s lookup before `later.pdf`'s own
    earlier batch. The two scenarios propose disjoint identities, so neither
    part's rows are the other's.
    """
    job = _scripted_job(
        config,
        (("earlier.pdf", "lesson_with_grammar"), ("later.pdf", SCENARIO)),
    )
    counts = {
        name: len(
            staging.read_staging_text(
                path.read_text(encoding="utf-8"), source=str(path)
            )[0]
        )
        for name, path in _parts(config, job.header.job_id)
    }
    assert counts == {"earlier.pdf": 2, "later.pdf": 3}
    return job


def test_a_parts_receipt_lookup_resolves_only_its_own_sources_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§7.12: binding a prior receipt costs one resolve per receipt of this source.

    `later.pdf` is the old+new history — its live review carries two rows its own
    archive already receipted and one it has not — and `earlier.pdf` is an
    unrelated part whose archive sorts first, so its receipt is offered to
    `later.pdf`'s lookup before that old batch. The part's own promotion already
    bound the source identity every receipt in its archive carries, which is
    enough to know the unrelated one cannot be one of this part's without
    resolving it. The receipts and their selections are the claim that must not
    move; the resolutions are the cost that must.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _two_source_job(config)
    words = RecordingProvider("voicevox", 7)
    earlier = config.staging_dir / "earlier.pdf.yaml"
    later = config.staging_dir / "later.pdf.yaml"
    whole = later.read_bytes()
    every_id = sorted(
        record.id
        for record in staging.read_staging_text(
            whole.decode("utf-8"), source=str(later)
        )[0]
    )
    # The first pass promotes two of `later.pdf`'s three rows, so the second has
    # one new row beside two retries of that first batch.
    _restore_live_review(later, whole, keep=tuple(every_id[:2]))
    _owner_choices(config, job)

    first_plan = _plan(config, job, _dictionary(), words)
    first, captured = _finish_capturing_archived_wire(config, first_plan, words)

    assert first.succeeded
    first_receipts = {
        str(entry["part_name"]): str(entry["receipt_id"])
        for entry in _record(first_plan.record_path)["promotion_receipt"]["parts"]
    }
    unrelated = first_receipts["earlier.pdf"]
    old = first_receipts["later.pdf"]

    _restore_live_review(earlier, captured["earlier.pdf"])
    # `later.pdf`'s own three rows under the metadata its promotion consumed:
    # the two archived ones are retries and the third is new work.
    _kept, reviewed_meta = staging.read_staging_text(
        captured["later.pdf"].decode("utf-8"), source=str(later)
    )
    records, _meta = staging.read_staging_text(
        whole.decode("utf-8"), source=str(later)
    )
    staging.write_staging(later, records, reviewed_meta, force=True)
    for name in ("earlier.pdf", "later.pdf"):
        _withdraw_coverage_reason(config, job.header.job_id, name)
    _owner_choices(config, job)

    plan = _plan(config, job, _dictionary(), words)

    by_part = {part.part_name: part for part in plan.parts}
    earlier_ids = sorted(by_part["earlier.pdf"].archive_retry_ids)
    assert by_part["earlier.pdf"].projected_state == "archive_retry"
    assert by_part["later.pdf"].projected_state == "lands"
    assert sorted(by_part["later.pdf"].landed_ids) == every_id[2:]
    assert sorted(by_part["later.pdf"].archive_retry_ids) == every_id[:2]

    resolutions = _lookup_resolutions(monkeypatch)
    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(plan.record_path)
    selections = _receipt_selection(record)
    bound = dict(selections["later.pdf"])
    new = next(receipt for receipt in bound if receipt != old)
    assert bound == {old: every_id[:2], new: every_id[2:]}
    assert selections["earlier.pdf"] == [(unrelated, earlier_ids)]

    # The measurement: this part paid for its own two receipts in candidate
    # order, and for nothing of the source whose archive sorts ahead of it.
    assert resolutions["later.pdf"] == {"primary": new, "resolved": [new, old]}
    assert resolutions["earlier.pdf"] == {
        "primary": unrelated,
        "resolved": [unrelated],
    }


def test_a_retry_that_binds_no_primary_still_skips_another_sources_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A strict-subset retry has no primary, and still pays for one source only.

    `promotion._latest_retry_batch` recovers nothing when the archive's latest
    batch holds more than the live review retries, so this part's prepared intent
    carries no receipt id at all and its lookup starts from the discovery listing
    with nothing already resolved. The identity its own promotion bound is what
    keeps that enumeration to its own archive: `earlier.pdf`'s receipt sorts
    first and is still never resolved for it, and the strictly smaller selection
    is unchanged.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _two_source_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    earlier = config.staging_dir / "earlier.pdf.yaml"
    later = config.staging_dir / "later.pdf.yaml"

    first_plan = _plan(config, job, _dictionary(), words)
    first, captured = _finish_capturing_archived_wire(config, first_plan, words)

    assert first.succeeded
    first_parts = {part.part_name: part for part in first_plan.parts}
    later_ids = sorted(first_parts["later.pdf"].landed_ids)
    first_receipts = {
        str(entry["part_name"]): str(entry["receipt_id"])
        for entry in _record(first_plan.record_path)["promotion_receipt"]["parts"]
    }
    unrelated = first_receipts["earlier.pdf"]
    prior = first_receipts["later.pdf"]

    _restore_live_review(earlier, captured["earlier.pdf"])
    _restore_live_review(later, captured["later.pdf"], keep=tuple(later_ids[:2]))
    left_out = later_ids[2]
    for name in ("earlier.pdf", "later.pdf"):
        _withdraw_coverage_reason(config, job.header.job_id, name)
    _owner_choices(config, job)

    plan = _plan(config, job, _dictionary(), words)

    by_part = {part.part_name: part for part in plan.parts}
    earlier_ids = sorted(by_part["earlier.pdf"].archive_retry_ids)
    assert by_part["later.pdf"].projected_state == "archive_retry"
    assert sorted(by_part["later.pdf"].archive_retry_ids) == later_ids[:2]
    assert left_out not in plan.record_ids

    resolutions = _lookup_resolutions(monkeypatch)
    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(plan.record_path)
    selections = _receipt_selection(record)
    assert selections["later.pdf"] == [(prior, later_ids[:2])]
    assert selections["earlier.pdf"] == [(unrelated, earlier_ids)]
    assert left_out not in {
        row[0] for row in record["promotion_receipt"]["projection"]
    }

    assert resolutions["later.pdf"] == {"primary": None, "resolved": [prior]}
    assert resolutions["earlier.pdf"] == {
        "primary": unrelated,
        "resolved": [unrelated],
    }


def test_completion_refuses_a_receipt_that_belongs_to_another_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt is bound to the archive, source and run of the part it covers.

    Both receipts here are real — one per landing part of one real job — so the
    swap is a genuine conflict rather than a fabricated handle, and §9.6.2's
    re-derivation refuses it instead of accepting an id that happens to be
    receipted somewhere.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _scripted_job(
        config, (("one.pdf", SCENARIO), ("three.pdf", "lesson_with_grammar"))
    )
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    plan = _plan(config, job, _dictionary(), words)

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(plan.record_path)
    study_finish._assert_complete(config, record)
    parts = record["promotion_receipt"]["parts"]
    swapped = json.loads(json.dumps(record))
    borrowed = parts[1]["receipts"][0]["receipt_id"]
    swapped["promotion_receipt"]["parts"][0]["receipts"][0]["receipt_id"] = borrowed

    with pytest.raises(study_finish.StudyFinishError) as error:
        study_finish._assert_complete(config, swapped)

    message = str(error.value)
    assert parts[0]["part_name"] in message
    assert borrowed in message
    # The refusal is about provenance, and names both archives by file.
    assert Path(parts[1]["archive_path"]).name in message
    assert Path(parts[0]["archive_path"]).name in message

    # A selection that drops one accepted id no longer accounts for the part.
    short = json.loads(json.dumps(record))
    dropped = short["promotion_receipt"]["parts"][0]["receipts"][0][
        "selected_ids"
    ].pop()
    with pytest.raises(study_finish.StudyFinishError) as narrowed:
        study_finish._assert_complete(config, short)
    assert dropped in str(narrowed.value)
    assert parts[0]["part_name"] in str(narrowed.value)

    # An id the receipt never promoted is not made this part's by claiming it,
    # even though the other part's archive really does receipt it.
    borrowed_id = parts[1]["receipts"][0]["selected_ids"][0]
    claimed = json.loads(json.dumps(record))
    claimed["promotion_receipt"]["parts"][0]["receipts"][0]["selected_ids"][0] = (
        borrowed_id
    )
    with pytest.raises(study_finish.StudyFinishError) as outside:
        study_finish._assert_complete(config, claimed)
    assert borrowed_id in str(outside.value)


# --- the final preview's two scopes -------------------------------------------


def test_the_final_preview_draws_only_this_jobs_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§7.12/§9.6.6: the job's selection and the whole deck stay two numbers.

    One kana-only card is already in this deck before the job runs, so the
    delivered package legitimately holds four notes while the job accepted three.
    The preview draws the three it accepted, counts them apart from the deck's
    own totals, and contains no card of the one it did not.
    """
    _no_api(monkeypatch)
    config = _project(tmp_path)
    outsider = VocabularyRecord(
        id="word:すし:すし",
        expression="すし",
        reading="すし",
        meanings=["sushi"],
        tags=["lesson-intake"],
    )
    config.normalized_file.write_text(
        json.dumps([outsider.to_dict()], ensure_ascii=False), encoding="utf-8"
    )
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)

    plan = _plan(config, job, _dictionary(), words)

    assert outsider.id not in plan.record_ids
    assert len(plan.record_ids) == 3
    # The whole-deck projection is the larger scope, and it is the package's.
    assert plan.note_count == 4
    assert plan.card_count == 8

    result = study_finish.execute_study_finish(
        config, plan, word_provider=words, sentence_provider=words
    )

    assert result.succeeded
    record = _record(plan.record_path)
    receipt = record["preview_receipt"]
    assert (receipt["note_count"], receipt["card_count"]) == (3, 6)
    assert (receipt["deck_note_count"], receipt["deck_card_count"]) == (4, 8)
    page = (config.root / receipt["preview_path"]).read_text(encoding="utf-8")
    assert page.count('data-record-id="') == 6
    assert f'data-record-id="{outsider.id}"' not in page
    assert "showing 3 notes · 6 cards" in page
    assert "whole deck 4 notes · 8 cards" in page
    # The package receipt keeps reporting the whole deck it delivered.
    assert record["package_receipt"]["note_count"] == 4
    assert record["package_receipt"]["card_count"] == 8


# --- an export ledger that moved on afterwards --------------------------------


def _fail_the_preview_once(
    config: ProjectConfig,
    plan: study_finish.StudyFinishPlan,
    words: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Leave one proven package at `packaged`, exactly as §9.6.7 describes."""

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise card_preview.CardPreviewError("the viewer asset went missing")

    monkeypatch.setattr(card_preview, "render_packaged_card_preview", fail)
    with pytest.raises(study_finish.StudyFinishError, match="could not be drawn"):
        study_finish.execute_study_finish(
            config, plan, word_provider=words, sentence_provider=words
        )
    monkeypatch.undo()
    _no_api(monkeypatch)
    packaged = _record(plan.record_path)
    assert packaged["state"] == "packaged"
    return packaged


def test_unrelated_later_ledger_history_still_lets_a_proven_package_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§9.6.6 wants this job's export entries recorded, not an immutable ledger file.

    The preview fails after the package is proven, and the owner then builds
    another deck — an ordinary `record_export` through the ledger's own writer,
    which changes the file's bytes and touches none of this job's rows. The
    preview-only resume completes and still builds, publishes and synthesizes
    nothing.
    """
    from japanese_anki import ledger as ledger_module

    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    plan = _plan(config, job, _dictionary(), words)

    packaged = _fail_the_preview_once(config, plan, words, monkeypatch)
    package_sha256 = packaged["package_receipt"]["package_sha256"]
    intents = list(packaged["package_intents"])
    delta = packaged["package_intents"][-1]["export_delta"]
    assert delta

    book = ledger_module.load(config.ledger_file)
    before = book.serialized_text()
    assert book.record_export(delta[0]["record_id"], "another-deck", at="2026-09-12")
    book.save()
    assert book.serialized_text() != before

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a preview-only resume did paid or building work")

    monkeypatch.setattr(study_finish.deck_package, "prepare_deck_package", explode)
    monkeypatch.setattr(
        study_finish.deck_package, "publish_prepared_deck_package", explode
    )
    monkeypatch.setattr(
        study_finish.audio_application, "execute_targeted_audio_locked", explode
    )
    monkeypatch.setattr(words, "synthesize", explode)

    result = study_finish.resume_study_finish(config, plan.fingerprint)

    assert result.succeeded
    final = _record(plan.record_path)
    assert final["package_intents"] == intents, "the resume prepared a new package"
    assert final["preview_receipt"]["package_sha256"] == package_sha256
    # Every frozen row is still exactly where the publication recorded it.
    after = ledger_module.load(config.ledger_file)
    for entry in delta:
        assert not after.record_export(
            entry["record_id"],
            entry["deck_stem"],
            gaps=entry["gaps"],
            at=entry["at"],
        )


def test_a_claimed_export_row_that_changed_still_refuses_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact frozen rows are the claim, so one of them moving refuses.

    Recorded through the ledger's own writer at a different day — the same
    `record_export` §7.11 froze — and never by editing the runtime file.
    """
    from japanese_anki import ledger as ledger_module

    _no_api(monkeypatch)
    config = _project(tmp_path)
    job = _settled_job(config)
    _owner_choices(config, job)
    words = RecordingProvider("voicevox", 7)
    plan = _plan(config, job, _dictionary(), words)

    packaged = _fail_the_preview_once(config, plan, words, monkeypatch)
    delta = packaged["package_intents"][-1]["export_delta"]
    claimed = delta[0]

    book = ledger_module.load(config.ledger_file)
    assert book.record_export(
        claimed["record_id"],
        claimed["deck_stem"],
        gaps=claimed["gaps"],
        at="2020-01-01",
    )
    book.save()

    with pytest.raises(study_finish.StudyFinishError) as error:
        study_finish.resume_study_finish(config, plan.fingerprint)

    message = str(error.value)
    assert claimed["record_id"] in message
    assert _record(plan.record_path)["state"] == "packaged"
