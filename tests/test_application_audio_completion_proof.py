"""Durable proof that one exact audio scope finished, and what it is worth.

Every proof here is minted by the real writer: the ordinary planner runs, the
ordinary executor voices the clips through offline fake providers, and the
proof is taken from the repository afterwards. No test hands
``prove_audio_completion`` a synthetic completion, because a synthetic one
would prove the assertions rather than the code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_application_audio import Provider, RecordingProvider
from test_application_revision_finish import _PCM, _RealtimeTransport

from japanese_anki import audio_cmd, operations
from japanese_anki import ledger as ledger_mod
from japanese_anki.application import audio as audio_application
from japanese_anki.config import ProjectConfig
from japanese_anki.io import (
    exclusive_path_lock,
    load_records,
    records_json_text,
)
from japanese_anki.models import ExampleSentence, VocabularyRecord
from japanese_anki.tts import openai_realtime, sentence_profile_for

SPEAK = "word:話す:はなす"
READ = "word:読む:よむ"


def speaking_record() -> VocabularyRecord:
    """One record whose stored, exported and unique clip counts all differ.

    Two identical polite sentences share one identity-addressed clip, and the
    second reaches no card field at all — so three stored slots become two
    clips and two exported slots, which is exactly the shape the package
    contract refuses to collapse into one number.
    """

    return VocabularyRecord(
        id=SPEAK,
        expression="話す",
        reading="はなす",
        meanings=["to speak"],
        part_of_speech="verb",
        verb_group="godan",
        examples=[
            ExampleSentence(
                japanese="日本語を話します。",
                english="I speak Japanese.",
                register="polite",
            ),
            ExampleSentence(
                japanese="日本語を話します。",
                english="I speak Japanese.",
                register="polite",
            ),
            ExampleSentence(
                japanese="日本語を話すよ。",
                english="I speak Japanese.",
                register="casual",
            ),
        ],
    )


def reading_record() -> VocabularyRecord:
    return VocabularyRecord(
        id=READ,
        expression="読む",
        reading="よむ",
        meanings=["to read"],
        examples=[
            ExampleSentence(
                japanese="本を読みます。",
                english="I read a book.",
                register="polite",
            )
        ],
    )


def project(tmp_path: Path, records: list[VocabularyRecord]) -> ProjectConfig:
    (tmp_path / "janki.toml").write_text(
        "[paths]\n"
        'normalized_file = "data/normalized/vocabulary.json"\n'
        'deck_dir = "data/decks"\n'
        'dist_dir = "dist"\n'
        'ledger_file = "data/ledger.json"\n'
        'media_dir = "data/media"\n'
        'operations_file = "data/operations.json"\n',
        encoding="utf-8",
    )
    config = ProjectConfig.load(tmp_path)
    config.normalized_file.parent.mkdir(parents=True, exist_ok=True)
    config.normalized_file.write_text(records_json_text(records), encoding="utf-8")
    return config


def word_engine() -> RecordingProvider:
    return RecordingProvider("voicevox", 7)


def sentence_engine() -> RecordingProvider:
    return RecordingProvider("voicevox", 3)


def paid_engine(config: ProjectConfig) -> openai_realtime.OpenAiRealtimePool:
    return openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=_RealtimeTransport(),
        operations_path=config.operations_file,
    )


class _RefuseFirstHandshake:
    """One fixture transport whose first attempt refuses before ``response.create``.

    A wrong-model session handshake is the supported proven-unsent refusal:
    the writer records ``failed_before_send`` and, because a dispatch callback
    is bound, leaves the entry for its caller to reconcile. Later attempts are
    the ordinary offline Realtime reply, so one provider profile — and
    therefore one confirmed enumeration — covers both runs.
    """

    def __init__(self) -> None:
        self.real = _RealtimeTransport()
        self.attempts = 0

    def __call__(
        self,
        endpoint: str,
        headers: dict[str, str],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self.attempts += 1
        if self.attempts == 1:
            return [
                {
                    "type": "session.created",
                    "session": {"model": "wrong-realtime-model"},
                }
            ]
        return self.real(endpoint, headers, request)


def _reserve(
    config: ProjectConfig,
    sentence_provider: object,
    attempts: list[audio_application.ReservedPaidAttempt],
):
    """The writer-owned reservation the finish records before a call leaves."""

    def before_paid_dispatch(dispatch: audio_application.PaidAudioDispatch) -> None:
        profile = sentence_profile_for(sentence_provider, dispatch.clip.record_id)
        attempts.append(
            audio_application.ReservedPaidAttempt(
                target=dispatch.clip.target,
                operation_id=dispatch.operation_id,
                expected_request_fp=profile.request_fingerprint(
                    dispatch.clip.request_input,
                    forced_accent=dispatch.clip.forced_accent,
                ),
            )
        )

    del config
    return before_paid_dispatch


def voice_and_prove(
    config: ProjectConfig,
    record_ids: list[str],
    *,
    words: bool = True,
    examples: bool = True,
    word_provider: object | None = None,
    sentence_provider: object | None = None,
) -> tuple[
    audio_application.AudioCompletionProof,
    audio_application.ConfirmedAudioPlan,
    list[audio_application.ReservedPaidAttempt],
]:
    """Plan, confirm, voice and then prove — the real order, the real writers."""

    word_provider = word_provider or word_engine()
    sentence_provider = sentence_provider or sentence_engine()
    plan = audio_application.plan_targeted_audio(
        config,
        record_ids,
        words=words,
        examples=examples,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )
    authority = audio_application.confirm_audio_plan(config, plan)
    attempts: list[audio_application.ReservedPaidAttempt] = []
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        outcome = audio_application.execute_targeted_audio_locked(
            config,
            record_ids,
            words=words,
            examples=examples,
            expected_fingerprint=plan.fingerprint,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
            before_paid_dispatch=_reserve(config, sentence_provider, attempts),
        )
        assert outcome.succeeded, outcome.state
        fresh = audio_application.plan_targeted_audio(
            config,
            record_ids,
            words=words,
            examples=examples,
            word_provider=word_provider,
            sentence_provider=sentence_provider,
        )
        proof = audio_application.prove_audio_completion(
            config,
            fresh,
            authority=authority,
            expected_slots=audio_application.expected_audio_slots(
                load_records(config.normalized_file.resolve()),
                record_ids,
                words=words,
                examples=examples,
            ),
            reservations=attempts,
        )
    return proof, authority, attempts


def replan(
    config: ProjectConfig,
    record_ids: list[str],
    *,
    word_provider: object,
    sentence_provider: object,
) -> audio_application.AudioPlan:
    return audio_application.plan_targeted_audio(
        config,
        record_ids,
        words=True,
        examples=True,
        word_provider=word_provider,
        sentence_provider=sentence_provider,
    )


def test_free_local_audio_proves_every_slot_with_no_paid_operation(
    tmp_path: Path,
) -> None:
    """VOICEVOX costs nothing, so no slot may claim a paid attempt.

    Mutation: require a reserved attempt for every ``provider-required`` clip
    rather than only for a paid-network one.
    """

    config = project(tmp_path, [speaking_record()])

    proof, _authority, attempts = voice_and_prove(config, [SPEAK])

    assert attempts == []
    assert proof.schema == audio_application.AUDIO_COMPLETION_SCHEMA
    assert all(slot.paid_operation is None for slot in proof.slots)
    assert all(slot.origin == "synthesized" for slot in proof.slots)
    assert {(slot.kind, slot.position) for slot in proof.slots} == {
        ("word", None),
        ("example", 0),
        ("example", 1),
        ("example", 2),
    }


def test_stored_slots_unique_clips_and_the_shared_target_stay_distinct(
    tmp_path: Path,
) -> None:
    """Two identical sentences share one clip and both slots still link to it.

    Mutation: derive the census from ``AudioPlan.clips`` instead of the stored
    example fields, which silently drops the duplicate slot.
    """

    config = project(tmp_path, [speaking_record()])

    proof, _authority, _attempts = voice_and_prove(config, [SPEAK])

    sentences = [slot for slot in proof.slots if slot.kind == "example"]
    assert len(sentences) == 3
    assert proof.stored_slot_count == 4
    assert proof.clip_count == 3
    assert len({slot.target for slot in sentences}) == 2
    shared = [slot for slot in sentences if slot.position in {0, 1}]
    assert shared[0].target == shared[1].target
    assert shared[0].media_sha256 == shared[1].media_sha256


def test_a_census_that_omits_a_stored_slot_refuses(tmp_path: Path) -> None:
    """A caller's disclosed count is an obligation, not an input.

    Mutation: drop the independent census cross-check and trust
    ``expected_slots``.
    """

    config = project(tmp_path, [speaking_record()])
    words, sentences = word_engine(), sentence_engine()
    plan = audio_application.plan_targeted_audio(
        config, [SPEAK], words=True, examples=True,
        word_provider=words, sentence_provider=sentences,
    )
    authority = audio_application.confirm_audio_plan(config, plan)
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        audio_application.execute_targeted_audio_locked(
            config, [SPEAK], words=True, examples=True,
            expected_fingerprint=plan.fingerprint,
            word_provider=words, sentence_provider=sentences,
        )
        fresh = replan(config, [SPEAK], word_provider=words, sentence_provider=sentences)
        full = audio_application.expected_audio_slots(
            load_records(config.normalized_file.resolve()),
            [SPEAK],
            words=True,
            examples=True,
        )
        short = tuple(slot for slot in full if slot.position != 1)

        with pytest.raises(audio_application.AudioProofError, match="slot census"):
            audio_application.prove_audio_completion(
                config, fresh, authority=authority, expected_slots=short
            )


def test_a_current_clip_in_another_voice_cannot_become_the_proof(
    tmp_path: Path,
) -> None:
    """Both states are ``current``; only one is the confirmed clip.

    Mutation: compare only the clips' states against the confirmed
    enumeration, dropping the provider comparison.
    """

    config = project(tmp_path, [speaking_record()])
    words, sentences = word_engine(), sentence_engine()
    # Confirmed in one voice, then actually voiced in another.
    authority = audio_application.confirm_audio_plan(
        config,
        audio_application.plan_targeted_audio(
            config, [SPEAK], words=True, examples=True,
            word_provider=words, sentence_provider=sentences,
        ),
    )
    other = RecordingProvider("voicevox", 11)
    other_plan = audio_application.plan_targeted_audio(
        config, [SPEAK], words=True, examples=True,
        word_provider=words, sentence_provider=other,
    )
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        audio_application.execute_targeted_audio_locked(
            config, [SPEAK], words=True, examples=True,
            expected_fingerprint=other_plan.fingerprint,
            word_provider=words, sentence_provider=other,
        )
        fresh = replan(config, [SPEAK], word_provider=words, sentence_provider=other)
        assert all(clip.state == "current" for clip in fresh.clips)

        with pytest.raises(
            audio_application.AudioProofError, match="example provider changed"
        ):
            audio_application.prove_audio_completion(
                config,
                fresh,
                authority=authority,
                expected_slots=audio_application.expected_audio_slots(
                    load_records(config.normalized_file.resolve()),
                    [SPEAK],
                    words=True,
                    examples=True,
                ),
            )


def test_a_ledger_entry_describing_another_profile_refuses(tmp_path: Path) -> None:
    """The voice is deliberately outside the content fingerprint.

    Mutation: ask ``audio_file_for`` for the content alone, dropping the
    provider/voice/speed/settings arguments.
    """

    config = project(tmp_path, [speaking_record()])
    words, sentences = word_engine(), sentence_engine()
    proof, _authority, _attempts = voice_and_prove(
        config, [SPEAK], word_provider=words, sentence_provider=sentences
    )
    book = ledger_mod.load(config.ledger_file)
    for entry in book.records[SPEAK]["audio"]:
        entry["voice"] = "someone-else"
    book.save()

    with exclusive_path_lock(config.root / ".janki-audio-operation"), pytest.raises(
        audio_application.AudioProofError, match="does not record"
    ):
        audio_application.revalidate_audio_completion(config, proof)


def test_edited_bytes_and_a_cleared_reference_each_refuse(tmp_path: Path) -> None:
    """A file on disk is not a canonical link, and a link is not bytes.

    Mutation: return the stored ``media_sha256`` instead of hashing the file.
    """

    config = project(tmp_path, [speaking_record()])
    proof, _authority, _attempts = voice_and_prove(config, [SPEAK])
    word = next(slot for slot in proof.slots if slot.kind == "word")
    media = config.media_dir.resolve() / "audio" / word.target
    original = media.read_bytes()
    media.write_bytes(b"a different recording entirely")

    with exclusive_path_lock(config.root / ".janki-audio-operation"), pytest.raises(
        audio_application.AudioProofError, match="no longer holds its proven bytes"
    ):
        audio_application.revalidate_audio_completion(config, proof)
    media.write_bytes(original)

    records = load_records(config.normalized_file.resolve())
    cleared = [replace(record, audio="") for record in records]
    config.normalized_file.write_text(records_json_text(cleared), encoding="utf-8")
    with exclusive_path_lock(config.root / ".janki-audio-operation"), pytest.raises(
        audio_application.AudioProofError, match="changed since audio completion"
    ):
        audio_application.revalidate_audio_completion(config, proof)


def test_a_removed_slot_refuses_even_with_a_recomputed_self_fingerprint(
    tmp_path: Path,
) -> None:
    """The census comes from the bound records, never from the payload's list.

    Mutation: validate only the slots a proof happens to carry.
    """

    config = project(tmp_path, [speaking_record()])
    proof, _authority, _attempts = voice_and_prove(config, [SPEAK])
    thinned = replace(
        proof,
        slots=tuple(slot for slot in proof.slots if slot.position != 1),
        fingerprint="0" * 64,
    )
    forged = replace(
        thinned, fingerprint=audio_application._proof_fingerprint(thinned)
    )
    # The forged payload passes its own integrity check, which is the point.
    audio_application.AudioCompletionProof.from_wire(config, forged.to_wire())

    with exclusive_path_lock(config.root / ".janki-audio-operation"), pytest.raises(
        audio_application.AudioProofError, match="carries 3 slots"
    ):
        audio_application.revalidate_audio_completion(config, forged)


def test_a_fresh_paid_clip_binds_its_reserved_attempt_and_expected_request(
    tmp_path: Path,
) -> None:
    """The paid slot names the attempt this finish reserved before dispatch.

    Mutation: read ``request_fp`` off the observed journal entry instead of
    comparing it to the expectation the dispatcher retained.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)

    proof, _authority, attempts = voice_and_prove(
        config, [READ], word_provider=word_engine(), sentence_provider=pool
    )

    assert len(attempts) == 1
    sentence = next(slot for slot in proof.slots if slot.kind == "example")
    assert sentence.paid_operation is not None
    assert sentence.origin == "synthesized"
    assert sentence.paid_operation.operation_id == attempts[0].operation_id
    assert (
        sentence.paid_operation.expected_request_fp == attempts[0].expected_request_fp
    )
    assert sentence.paid_operation.model == openai_realtime.MODEL
    assert sentence.provider.access == "paid-network"
    word = next(slot for slot in proof.slots if slot.kind == "word")
    assert word.paid_operation is None
    # A successful Realtime call commits and then forgets its entry; that
    # durable forget is what says the money was accounted for.
    assert sentence.paid_operation.state == "accounted"
    assert (
        operations.OperationJournal.load(config.operations_file).operations == {}
    )


def test_a_forgotten_failed_attempt_cannot_claim_another_run_s_paid_bytes(
    tmp_path: Path,
) -> None:
    """A reservation whose attempt produced nothing proves nothing about bytes.

    Every step is ordinary supported history, with no hand-edited file: a real
    ``before_paid_dispatch`` reservation **A**, a refusal proven before send,
    the ordinary journal ``forget`` a reconciler performs for a
    ``failed_before_send`` row, and then any other authorized run — a plain
    ``janki audio`` — voicing the same identity-addressed target under its own
    operation **B**. The first finish's proof must refuse rather than label
    B's bytes as A's accounted result.

    Mutation: treat an absent journal entry as an accounted attempt.
    """

    config = project(tmp_path, [reading_record()])
    transport = _RefuseFirstHandshake()
    pool = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=transport,
        operations_path=config.operations_file,
    )
    words = word_engine()
    first = audio_application.plan_targeted_audio(
        config, [READ], words=True, examples=True,
        word_provider=words, sentence_provider=pool,
    )
    authority = audio_application.confirm_audio_plan(config, first)
    attempts: list[audio_application.ReservedPaidAttempt] = []
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        stopped = audio_application.execute_targeted_audio_locked(
            config, [READ], words=True, examples=True,
            expected_fingerprint=first.fingerprint,
            word_provider=words, sentence_provider=pool,
            before_paid_dispatch=_reserve(config, pool, attempts),
        )
    assert stopped.state == "generation-stopped"
    assert len(attempts) == 1
    held = operations.OperationJournal.load(config.operations_file)
    assert held.operations[attempts[0].operation_id].state == "failed_before_send"

    # What a reconciler does with a reservation proven never to have been sent.
    operations.OperationJournal.load(config.operations_file).forget(
        [attempts[0].operation_id]
    )
    assert operations.OperationJournal.load(config.operations_file).operations == {}

    second = audio_application.plan_targeted_audio(
        config, [READ], words=True, examples=True,
        word_provider=words, sentence_provider=pool,
    )
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        done = audio_application.execute_targeted_audio_locked(
            config, [READ], words=True, examples=True,
            expected_fingerprint=second.fingerprint,
            word_provider=words, sentence_provider=pool,
        )
        assert done.succeeded, done.state
        fresh = replan(config, [READ], word_provider=words, sentence_provider=pool)

        with pytest.raises(
            audio_application.AudioProofError, match="did not produce"
        ):
            audio_application.prove_audio_completion(
                config,
                fresh,
                authority=authority,
                expected_slots=audio_application.expected_audio_slots(
                    load_records(config.normalized_file.resolve()),
                    [READ],
                    words=True,
                    examples=True,
                ),
                reservations=attempts,
            )


def test_a_paid_clip_with_no_reserved_attempt_refuses_by_name(
    tmp_path: Path,
) -> None:
    """A coordinator that never wires the dispatch seam gets no proof.

    Mutation: fall through to ``paid_operation = None`` when no reservation
    covers a confirmed paid clip.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)
    words = word_engine()
    plan = audio_application.plan_targeted_audio(
        config, [READ], words=True, examples=True,
        word_provider=words, sentence_provider=pool,
    )
    authority = audio_application.confirm_audio_plan(config, plan)
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        audio_application.execute_targeted_audio_locked(
            config, [READ], words=True, examples=True,
            expected_fingerprint=plan.fingerprint,
            word_provider=words, sentence_provider=pool,
        )
        fresh = replan(config, [READ], word_provider=words, sentence_provider=pool)

        with pytest.raises(
            audio_application.AudioProofError, match="no reserved attempt accounts"
        ):
            audio_application.prove_audio_completion(
                config,
                fresh,
                authority=authority,
                expected_slots=audio_application.expected_audio_slots(
                    load_records(config.normalized_file.resolve()),
                    [READ],
                    words=True,
                    examples=True,
                ),
            )


def test_an_attempt_bound_to_a_foreign_request_refuses(tmp_path: Path) -> None:
    """The journal must agree with the expectation, not merely with itself.

    Mutation: drop the ``request_fp`` comparison.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)
    words = word_engine()
    plan = audio_application.plan_targeted_audio(
        config, [READ], words=True, examples=True,
        word_provider=words, sentence_provider=pool,
    )
    authority = audio_application.confirm_audio_plan(config, plan)
    attempts: list[audio_application.ReservedPaidAttempt] = []
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        audio_application.execute_targeted_audio_locked(
            config, [READ], words=True, examples=True,
            expected_fingerprint=plan.fingerprint,
            word_provider=words, sentence_provider=pool,
            before_paid_dispatch=_reserve(config, pool, attempts),
        )
        fresh = replan(config, [READ], word_provider=words, sentence_provider=pool)
        clip = next(item for item in fresh.clips if item.kind == "example")
        # A live entry under that id, bound to a different request.
        journal = operations.OperationJournal.load(config.operations_file)
        journal.authorize(
            attempts[0].operation_id,
            kind="audio-realtime",
            source_file=f"{READ}#example:{clip.target}",
            source_sha256=clip.content_fingerprint,
            request_fp="a-completely-different-request",
            model=openai_realtime.MODEL,
        )

        with pytest.raises(
            audio_application.AudioProofError, match="does not match the exact"
        ):
            audio_application.prove_audio_completion(
                config,
                fresh,
                authority=authority,
                expected_slots=audio_application.expected_audio_slots(
                    load_records(config.normalized_file.resolve()),
                    [READ],
                    words=True,
                    examples=True,
                ),
                reservations=attempts,
            )


def test_an_unaccounted_live_attempt_refuses(tmp_path: Path) -> None:
    """`result_captured` and `outcome_unknown` are money nobody accounted for.

    Mutation: accept any terminal state rather than ``committed``.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)
    words = word_engine()
    plan = audio_application.plan_targeted_audio(
        config, [READ], words=True, examples=True,
        word_provider=words, sentence_provider=pool,
    )
    authority = audio_application.confirm_audio_plan(config, plan)
    attempts: list[audio_application.ReservedPaidAttempt] = []
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        audio_application.execute_targeted_audio_locked(
            config, [READ], words=True, examples=True,
            expected_fingerprint=plan.fingerprint,
            word_provider=words, sentence_provider=pool,
            before_paid_dispatch=_reserve(config, pool, attempts),
        )
        fresh = replan(config, [READ], word_provider=words, sentence_provider=pool)
        clip = next(item for item in fresh.clips if item.kind == "example")
        journal = operations.OperationJournal.load(config.operations_file)
        journal.authorize(
            attempts[0].operation_id,
            kind="audio-realtime",
            source_file=f"{READ}#example:{clip.target}",
            source_sha256=clip.content_fingerprint,
            request_fp=attempts[0].expected_request_fp,
            model=openai_realtime.MODEL,
        )
        operations.OperationJournal.load(config.operations_file).advance(
            attempts[0].operation_id, "dispatching"
        )

        with pytest.raises(
            audio_application.AudioProofError, match="is dispatching"
        ):
            audio_application.prove_audio_completion(
                config,
                fresh,
                authority=authority,
                expected_slots=audio_application.expected_audio_slots(
                    load_records(config.normalized_file.resolve()),
                    [READ],
                    words=True,
                    examples=True,
                ),
                reservations=attempts,
            )


def test_reuse_and_recovery_invent_no_reservation(tmp_path: Path) -> None:
    """A clip that was already current is proven by currency, not by money.

    Mutation: accept a reservation for a clip the confirmation classified as
    ``current``.
    """

    config = project(tmp_path, [speaking_record()])
    words, sentences = word_engine(), sentence_engine()
    voice_and_prove(config, [SPEAK], word_provider=words, sentence_provider=sentences)

    # A second finish over the same records finds everything current.
    plan = audio_application.plan_targeted_audio(
        config, [SPEAK], words=True, examples=True,
        word_provider=words, sentence_provider=sentences,
    )
    assert all(clip.state == "current" for clip in plan.clips)
    authority = audio_application.confirm_audio_plan(config, plan)
    slots = audio_application.expected_audio_slots(
        load_records(config.normalized_file.resolve()),
        [SPEAK],
        words=True,
        examples=True,
    )
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        proof = audio_application.prove_audio_completion(
            config, plan, authority=authority, expected_slots=slots
        )
        assert all(slot.origin == "reused" for slot in proof.slots)
        assert all(slot.paid_operation is None for slot in proof.slots)

        with pytest.raises(
            audio_application.AudioProofError, match="did not dispatch"
        ):
            audio_application.prove_audio_completion(
                config,
                plan,
                authority=authority,
                expected_slots=slots,
                reservations=[
                    audio_application.ReservedPaidAttempt(
                        target=plan.clips[0].target,
                        operation_id="invented",
                        expected_request_fp="invented",
                    )
                ],
            )


def test_revalidation_resolves_no_provider_and_sends_no_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request fingerprint was bound at dispatch; it is never recomputed.

    Mutation: recompute the expected request fingerprint from a live provider
    during revalidation.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)
    proof, _authority, _attempts = voice_and_prove(
        config, [READ], word_provider=word_engine(), sentence_provider=pool
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("revalidation resolved a provider")

    monkeypatch.setattr(audio_application, "resolve_sentence_provider", forbidden)
    monkeypatch.setattr(audio_application, "resolve_word_provider", forbidden)
    monkeypatch.setattr(audio_application, "sentence_profile_for", forbidden)

    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        audio_application.revalidate_audio_completion(config, proof)


def test_a_successful_run_leaves_no_write_ahead_row_and_that_is_expected(
    tmp_path: Path,
) -> None:
    """Cleanup after success is the finished state, never missing proof.

    Mutation: require a live ``pending_audio`` row for a recovered clip.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)

    proof, _authority, _attempts = voice_and_prove(
        config, [READ], word_provider=word_engine(), sentence_provider=pool
    )

    assert ledger_mod.load(config.ledger_file).pending_audio == {}
    stages = config.media_dir.resolve() / "audio" / ".pending"
    assert not stages.exists() or not list(stages.glob("*.stage"))
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        audio_application.revalidate_audio_completion(config, proof)


def test_the_proof_round_trips_and_refuses_a_tampered_payload(
    tmp_path: Path,
) -> None:
    """Self-fingerprints prove integrity; they are not authority.

    Mutation: skip the fingerprint recomputation in ``from_wire``.
    """

    config = project(tmp_path, [speaking_record()])
    proof, _authority, _attempts = voice_and_prove(config, [SPEAK])
    wire = proof.to_wire()

    restored = audio_application.AudioCompletionProof.from_wire(
        config, json.loads(json.dumps(wire, ensure_ascii=False))
    )

    assert restored == proof
    tampered = json.loads(json.dumps(wire, ensure_ascii=False))
    tampered["slots"][0]["media_sha256"] = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(audio_application.AudioProofError, match="own fingerprint"):
        audio_application.AudioCompletionProof.from_wire(config, tampered)
    unknown = json.loads(json.dumps(wire, ensure_ascii=False))
    unknown["schema"] = "janki-audio-completion-proof-v99"
    with pytest.raises(audio_application.AudioProofError, match="Unknown audio"):
        audio_application.AudioCompletionProof.from_wire(config, unknown)


def test_proving_is_idempotent_after_a_lost_execution_return(
    tmp_path: Path,
) -> None:
    """Nothing in the proof comes from a return value.

    Mutation: mint the fingerprint from a clock or a counter.
    """

    config = project(tmp_path, [speaking_record()])
    words, sentences = word_engine(), sentence_engine()
    first, authority, _attempts = voice_and_prove(
        config, [SPEAK], word_provider=words, sentence_provider=sentences
    )

    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        again = audio_application.prove_audio_completion(
            config,
            replan(config, [SPEAK], word_provider=words, sentence_provider=sentences),
            authority=authority,
            expected_slots=audio_application.expected_audio_slots(
                load_records(config.normalized_file.resolve()),
                [SPEAK],
                words=True,
                examples=True,
            ),
        )

    assert again.fingerprint == first.fingerprint


def test_a_deck_scoped_owner_is_refused_by_name(tmp_path: Path) -> None:
    """The packaged artifact a proof serves is a canonical vocabulary deck.

    Mutation: accept any owner path as the proof's canonical collection.
    """

    config = project(tmp_path, [speaking_record()])
    proof, _authority, _attempts = voice_and_prove(config, [SPEAK])
    elsewhere = replace(
        proof,
        canonical_path=config.root.resolve() / "data" / "decks" / "drills.yaml",
    )

    with exclusive_path_lock(config.root / ".janki-audio-operation"), pytest.raises(
        audio_application.AudioProofError, match="another canonical collection"
    ):
        audio_application.revalidate_audio_completion(config, elsewhere)


def test_duplicate_sentences_with_divergent_spoken_input_still_refuse(
    tmp_path: Path,
) -> None:
    """The existing structural collision check is preserved, not replaced.

    Mutation: deduplicate slots by displayed sentence inside the proof instead
    of leaving the refusal with ``prepare_example_audio_profiles``.
    """

    record = speaking_record()
    diverged = replace(
        record,
        examples=[
            record.examples[0],
            replace(record.examples[1], spoken_japanese="にほんごをはなします。"),
            record.examples[2],
        ],
    )
    config = project(tmp_path, [diverged])

    with pytest.raises(
        audio_application.AudioPlanError, match="different\\s+spoken Japanese"
    ):
        audio_application.plan_targeted_audio(
            config,
            [SPEAK],
            words=True,
            examples=True,
            word_provider=word_engine(),
            sentence_provider=sentence_engine(),
        )


def test_the_confirmed_enumeration_pins_an_already_current_clip_s_bytes(
    tmp_path: Path,
) -> None:
    """A clip confirmed as ``current`` must still hold those exact bytes.

    Mutation: compare only the clip states, not the initial media hash.
    """

    config = project(tmp_path, [speaking_record()])
    words, sentences = word_engine(), sentence_engine()
    voice_and_prove(config, [SPEAK], word_provider=words, sentence_provider=sentences)
    plan = replan(config, [SPEAK], word_provider=words, sentence_provider=sentences)
    authority = audio_application.confirm_audio_plan(config, plan)
    word = next(clip for clip in plan.clips if clip.kind == "word")
    media = config.media_dir.resolve() / "audio" / word.target
    media.write_bytes(b"replaced after the confirmation")

    with exclusive_path_lock(config.root / ".janki-audio-operation"), pytest.raises(
        audio_application.AudioProofError, match="already-current audio file"
    ):
        audio_application.prove_audio_completion(
            config,
            replan(
                config, [SPEAK], word_provider=words, sentence_provider=sentences
            ),
            authority=authority,
            expected_slots=audio_application.expected_audio_slots(
                load_records(config.normalized_file.resolve()),
                [SPEAK],
                words=True,
                examples=True,
            ),
        )


def test_an_unfinished_scope_is_not_a_completion(tmp_path: Path) -> None:
    """Nothing may be proven while a confirmed clip is still owed.

    Mutation: drop the every-clip-current gate.
    """

    config = project(tmp_path, [speaking_record()])
    words, sentences = word_engine(), sentence_engine()
    plan = audio_application.plan_targeted_audio(
        config, [SPEAK], words=True, examples=True,
        word_provider=words, sentence_provider=sentences,
    )
    authority = audio_application.confirm_audio_plan(config, plan)

    with exclusive_path_lock(config.root / ".janki-audio-operation"), pytest.raises(
        audio_application.AudioProofError, match="is not current"
    ):
        audio_application.prove_audio_completion(
            config,
            plan,
            authority=authority,
            expected_slots=audio_application.expected_audio_slots(
                load_records(config.normalized_file.resolve()),
                [SPEAK],
                words=True,
                examples=True,
            ),
        )


def test_the_proof_names_the_actual_spoken_input_it_bound(tmp_path: Path) -> None:
    """``request_input`` is the repository's own Japanese, not an opaque id.

    A reviewed ``spoken_japanese`` override is what the provider was sent, and
    the proof says so rather than pretending the wire carries no text. This is
    ordinary repository data under the existing disclosure rules; the proof
    grants no new disclosure.
    """

    record = speaking_record()
    overridden = replace(
        record,
        examples=[
            replace(record.examples[0], spoken_japanese="にほんごをはなします。"),
            replace(record.examples[1], spoken_japanese="にほんごをはなします。"),
            record.examples[2],
        ],
    )
    config = project(tmp_path, [overridden])

    proof, _authority, _attempts = voice_and_prove(config, [SPEAK])

    polite = next(slot for slot in proof.slots if slot.position == 0)
    assert polite.request_input == "にほんごをはなします。"
    assert polite.to_wire()["request_input"] == "にほんごをはなします。"


def test_a_pcm_backed_paid_clip_lands_real_bytes(tmp_path: Path) -> None:
    """The offline paid transport still produces a finite decoded WAV."""

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)

    proof, _authority, _attempts = voice_and_prove(
        config, [READ], word_provider=word_engine(), sentence_provider=pool
    )

    sentence = next(slot for slot in proof.slots if slot.kind == "example")
    media = config.media_dir.resolve() / "audio" / sentence.target
    payload = media.read_bytes()
    assert payload.startswith(b"RIFF")
    assert len(payload) > len(_PCM)
    assert hashlib.sha256(payload).hexdigest() == sentence.media_sha256


def test_a_local_provider_slot_keeps_its_provider_profile_on_the_wire(
    tmp_path: Path,
) -> None:
    """The profile is how the voice is proven, so it round-trips exactly."""

    config = project(tmp_path, [speaking_record()])
    proof, _authority, _attempts = voice_and_prove(config, [SPEAK])

    word = next(slot for slot in proof.slots if slot.kind == "word")
    assert isinstance(word.provider, audio_application.AudioProviderPlan)
    assert word.provider.voice == 7
    assert word.provider.access == "local-network"
    example = next(slot for slot in proof.slots if slot.kind == "example")
    assert example.provider.voice == 3
    assert Provider is not None  # the shared offline fake, imported above


def test_a_live_entry_under_a_settled_attempt_s_id_refuses(tmp_path: Path) -> None:
    """A settled attempt whose id now holds live authority is unaccounted money.

    Mutation: accept any journal state rather than an accounted committed call.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)
    proof, _authority, attempts = voice_and_prove(
        config, [READ], word_provider=word_engine(), sentence_provider=pool
    )
    sentence = next(slot for slot in proof.slots if slot.kind == "example")
    assert sentence.paid_operation.state == "accounted"
    operations.OperationJournal.load(config.operations_file).authorize(
        attempts[0].operation_id,
        kind="audio-realtime",
        source_file=sentence.paid_operation.source_file,
        source_sha256=sentence.content_fingerprint,
        request_fp=sentence.paid_operation.expected_request_fp,
        model=openai_realtime.MODEL,
    )

    with exclusive_path_lock(config.root / ".janki-audio-operation"), pytest.raises(
        audio_application.AudioProofError, match="is authorized, which is not an"
    ):
        audio_application.revalidate_audio_completion(config, proof)


def _ledger_witness(config: ProjectConfig, target: str) -> dict[str, object]:
    book = ledger_mod.load(config.ledger_file)
    for entry in book.records[READ]["audio"]:
        if entry.get("file") == target:
            return dict(entry[audio_cmd.PAID_ATTEMPT])
    raise AssertionError(f"no ledger entry for {target}")


def test_the_proven_paid_operation_comes_from_the_ledger_not_the_reservation(
    tmp_path: Path,
) -> None:
    """The writer's record of the call, compared against the dispatcher's.

    Two independently written values meet here: the attempt the paid writer
    recorded beside the bytes it persisted, and the expectation the dispatcher
    retained before the call left. Agreement between them is the proof; either
    one alone is a single source agreeing with itself.

    Mutation: read the witness fields out of the reservation instead of the
    ledger entry.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)

    proof, _authority, attempts = voice_and_prove(
        config, [READ], word_provider=word_engine(), sentence_provider=pool
    )

    sentence = next(slot for slot in proof.slots if slot.kind == "example")
    witness = _ledger_witness(config, sentence.target)
    assert witness["operation_id"] == attempts[0].operation_id
    assert witness["request_fp"] == attempts[0].expected_request_fp
    assert witness["model"] == openai_realtime.MODEL
    assert witness["audio_sha256"] == sentence.media_sha256
    assert sentence.paid_operation is not None
    assert sentence.paid_operation.operation_id == witness["operation_id"]
    # And the word clip, which cost nothing, records no attempt at all.
    word = next(slot for slot in proof.slots if slot.kind == "word")
    assert word.paid_operation is None
    word_entry = next(
        entry
        for entry in ledger_mod.load(config.ledger_file).records[READ]["audio"]
        if entry.get("file") == word.target
    )
    assert audio_cmd.PAID_ATTEMPT not in word_entry


def test_a_reservation_expecting_another_request_refuses_against_the_record(
    tmp_path: Path,
) -> None:
    """Structural validation: the two written values must be the same value.

    This checks object agreement, not owner authority — the finish record is
    owner-bound and never hand-edited, and a proof's self-hash is integrity
    only.

    Mutation: drop the recorded-vs-expected request fingerprint comparison.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)
    words = word_engine()
    plan = audio_application.plan_targeted_audio(
        config, [READ], words=True, examples=True,
        word_provider=words, sentence_provider=pool,
    )
    authority = audio_application.confirm_audio_plan(config, plan)
    attempts: list[audio_application.ReservedPaidAttempt] = []
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        audio_application.execute_targeted_audio_locked(
            config, [READ], words=True, examples=True,
            expected_fingerprint=plan.fingerprint,
            word_provider=words, sentence_provider=pool,
            before_paid_dispatch=_reserve(config, pool, attempts),
        )
        fresh = replan(config, [READ], word_provider=words, sentence_provider=pool)
        diverged = [
            replace(attempts[0], expected_request_fp=hashlib.sha256(b"other").hexdigest())
        ]

        with pytest.raises(
            audio_application.AudioProofError, match="recorded another request"
        ):
            audio_application.prove_audio_completion(
                config,
                fresh,
                authority=authority,
                expected_slots=audio_application.expected_audio_slots(
                    load_records(config.normalized_file.resolve()),
                    [READ],
                    words=True,
                    examples=True,
                ),
                reservations=diverged,
            )


def test_proven_bytes_must_be_the_bytes_the_recorded_attempt_rendered(
    tmp_path: Path,
) -> None:
    """A clip republished from elsewhere is not the attempt's own result.

    The ledger's currency answer covers the request and the voice, never the
    bytes, so a file replaced after the commit is still 'current' — and only
    the recorded render hash notices.

    Mutation: compare the recorded hash against the WAL's staged hash instead
    of the bytes read from disk.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)
    words = word_engine()
    plan = audio_application.plan_targeted_audio(
        config, [READ], words=True, examples=True,
        word_provider=words, sentence_provider=pool,
    )
    authority = audio_application.confirm_audio_plan(config, plan)
    attempts: list[audio_application.ReservedPaidAttempt] = []
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        audio_application.execute_targeted_audio_locked(
            config, [READ], words=True, examples=True,
            expected_fingerprint=plan.fingerprint,
            word_provider=words, sentence_provider=pool,
            before_paid_dispatch=_reserve(config, pool, attempts),
        )
        fresh = replan(config, [READ], word_provider=words, sentence_provider=pool)
        clip = next(item for item in fresh.clips if item.kind == "example")
        media = config.media_dir.resolve() / "audio" / clip.target
        media.write_bytes(media.read_bytes() + b"\x00\x00")

        with pytest.raises(
            audio_application.AudioProofError, match="rendered other bytes"
        ):
            audio_application.prove_audio_completion(
                config,
                fresh,
                authority=authority,
                expected_slots=audio_application.expected_audio_slots(
                    load_records(config.normalized_file.resolve()),
                    [READ],
                    words=True,
                    examples=True,
                ),
                reservations=attempts,
            )


@pytest.mark.parametrize("field", ["absent", "model"])
def test_a_paid_clip_whose_entry_cannot_name_its_call_refuses(
    tmp_path: Path,
    field: str,
) -> None:
    """An entry with no usable attempt proves currency, never money.

    A ledger written before this field existed, or one whose attribution was
    removed, is exactly as silent as the journal: it can still say the clip is
    current for the exact request and voice, and that is a different claim.

    Mutation: treat a missing or foreign witness as an accounted attempt.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)
    proof, _authority, _attempts = voice_and_prove(
        config, [READ], word_provider=word_engine(), sentence_provider=pool
    )
    sentence = next(slot for slot in proof.slots if slot.kind == "example")
    book = ledger_mod.load(config.ledger_file)
    for entry in book.records[READ]["audio"]:
        if entry.get("file") != sentence.target:
            continue
        if field == "absent":
            del entry[audio_cmd.PAID_ATTEMPT]
        else:
            entry[audio_cmd.PAID_ATTEMPT]["model"] = "gpt-realtime-legacy"
    book.save()

    expected = "carries no recorded attempt" if field == "absent" else "recorded model"
    with exclusive_path_lock(config.root / ".janki-audio-operation"), pytest.raises(
        audio_application.AudioProofError, match=expected
    ):
        audio_application.revalidate_audio_completion(config, proof)


def test_a_recovered_paid_clip_proves_without_any_fresh_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery costs nothing, so it reserves nothing and witnesses nothing.

    The clip was confirmed ``recoverable``: its bytes were already paid for and
    bound by the confirmation's fixed hash, and adopting them is not a new
    dispatch. Requiring a witness here would strand a job that has already
    spent its money.

    Mutation: require the recorded attempt for a recovered clip too.
    """

    config = project(tmp_path, [reading_record()])
    pool = paid_engine(config)
    words = word_engine()
    plan = audio_application.plan_targeted_audio(
        config, [READ], words=True, examples=True,
        word_provider=words, sentence_provider=pool,
    )
    real_promote = audio_cmd.promote_pending_audio
    monkeypatch.setattr(
        audio_application.audio_cmd,
        "promote_pending_audio",
        lambda *_a, **_k: (_ for _ in ()).throw(
            audio_application.JankiError("simulated publication interruption")
        ),
    )
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        interrupted = audio_application.execute_targeted_audio_locked(
            config, [READ], words=True, examples=True,
            expected_fingerprint=plan.fingerprint,
            word_provider=words, sentence_provider=pool,
        )
    assert interrupted.pending_recovery
    monkeypatch.setattr(
        audio_application.audio_cmd, "promote_pending_audio", real_promote
    )

    recovering = replan(config, [READ], word_provider=words, sentence_provider=pool)
    example = next(clip for clip in recovering.clips if clip.kind == "example")
    assert example.state == "recoverable"
    authority = audio_application.confirm_audio_plan(config, recovering)
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        adopted = audio_application.execute_targeted_audio_locked(
            config, [READ], words=True, examples=True,
            expected_fingerprint=recovering.fingerprint,
            word_provider=words, sentence_provider=pool,
        )
        assert adopted.succeeded, adopted.state
        proof = audio_application.prove_audio_completion(
            config,
            replan(config, [READ], word_provider=words, sentence_provider=pool),
            authority=authority,
            expected_slots=audio_application.expected_audio_slots(
                load_records(config.normalized_file.resolve()),
                [READ],
                words=True,
                examples=True,
            ),
        )
        sentence = next(slot for slot in proof.slots if slot.kind == "example")
        assert sentence.origin == "recovered"
        assert sentence.paid_operation is None
        audio_application.revalidate_audio_completion(config, proof)


def test_a_stage_interrupted_before_its_wal_row_still_proves_its_own_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interrupted paid writer, finished by the ordinary command.

    The call was reserved, dispatched and captured, and the process died inside
    the write-ahead persist: stage bytes on disk, the reply still in the
    journal, no row. What §7.10 promises for that state is that the exact same
    unforced command finishes it — no ``--force``, no second charge — and this
    finish can still prove the clip is the result of the call *it* paid for,
    under the enumeration it confirmed before any of it happened.

    No collaborator is replaced here: the executor wires its own persist
    callback, so this is the composition of recovery's attribution write with
    the real ledger merge. The spy on the resume only records what that
    callback asked the merge for and then performs it — recovery must swap the
    exact row it read, and must not reach for the blanket replace that
    ``--force`` means.

    Mutation: persist that attribution against the row being written, against
    nothing at all, or by replacing whatever is there.
    """

    config = project(tmp_path, [reading_record()])
    transport = _RealtimeTransport()
    pool = openai_realtime.OpenAiRealtimePool(
        api_key="test-key",
        transport=transport,
        operations_path=config.operations_file,
    )
    words = word_engine()
    plan = audio_application.plan_targeted_audio(
        config, [READ], words=False, examples=True,
        word_provider=words, sentence_provider=pool,
    )
    assert [clip.state for clip in plan.clips] == ["provider-required"]
    authority = audio_application.confirm_audio_plan(config, plan)
    reserved: list[audio_application.ReservedPaidAttempt] = []
    real_persist = audio_application._persist_audio_wal

    def die_in_the_wal(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt("simulated death inside the write-ahead persist")

    monkeypatch.setattr(audio_application, "_persist_audio_wal", die_in_the_wal)
    with exclusive_path_lock(config.root / ".janki-audio-operation"), pytest.raises(
        KeyboardInterrupt, match="inside the write-ahead persist"
    ):
        audio_application.execute_targeted_audio_locked(
            config, [READ], words=False, examples=True,
            expected_fingerprint=plan.fingerprint,
            word_provider=words, sentence_provider=pool,
            before_paid_dispatch=_reserve(config, pool, reserved),
        )
    monkeypatch.setattr(audio_application, "_persist_audio_wal", real_persist)

    assert len(transport.calls) == 1 and len(reserved) == 1
    assert ledger_mod.load(config.ledger_file).pending_audio == {}
    journal = operations.OperationJournal.load(config.operations_file)
    [captured] = list(journal.operations.values())
    assert captured.state == "result_captured"
    assert captured.operation_id == reserved[0].operation_id

    redispatched: list[audio_application.ReservedPaidAttempt] = []
    merges: list[tuple[bool, dict[str, Any] | None]] = []
    durable: list[dict[str, Any]] = []

    def watching(book: Any, keys: Any, **kwargs: Any) -> Any:
        merges.append(
            (
                bool(kwargs.get("replace")),
                None if kwargs.get("expected") is None else dict(kwargs["expected"]),
            )
        )
        outcome = real_persist(book, keys, **kwargs)
        durable.append(dict(ledger_mod.load(config.ledger_file).pending_audio))
        return outcome

    monkeypatch.setattr(audio_application, "_persist_audio_wal", watching)
    with exclusive_path_lock(config.root / ".janki-audio-operation"):
        recovering = audio_application.plan_targeted_audio(
            config, [READ], words=False, examples=True,
            word_provider=words, sentence_provider=pool,
        )
        assert [clip.state for clip in recovering.clips] == ["recoverable"]
        outcome = audio_application.execute_targeted_audio_locked(
            config, [READ], words=False, examples=True,
            expected_fingerprint=recovering.fingerprint,
            word_provider=words, sentence_provider=pool,
            before_paid_dispatch=_reserve(config, pool, redispatched),
        )
        monkeypatch.setattr(audio_application, "_persist_audio_wal", real_persist)
        assert outcome.succeeded, outcome.state
        assert len(transport.calls) == 1, "recovery must not re-dispatch the call"
        assert redispatched == [], "and must not reserve a second attempt"

        # Adoption asks for an additive merge; the attribution that follows asks
        # to swap exactly the row adoption made durable — not the value it is
        # writing, and not the blanket replace `--force` would have meant.
        adopted, attributed = durable
        assert merges == [(False, None), (False, adopted)]
        [(row_key, adopted_row)] = adopted.items()
        assert audio_cmd.PAID_ATTEMPT not in adopted_row["details"]
        assert attributed[row_key]["details"][audio_cmd.PAID_ATTEMPT] == {
            "operation_id": reserved[0].operation_id,
            "request_fp": reserved[0].expected_request_fp,
            "model": openai_realtime.MODEL,
            "audio_sha256": str(adopted_row["staged_sha256"]),
        }

        proof = audio_application.prove_audio_completion(
            config,
            audio_application.plan_targeted_audio(
                config, [READ], words=False, examples=True,
                word_provider=words, sentence_provider=pool,
            ),
            # The enumeration confirmed before the interrupted run, and the
            # reservation that run saved before its call left.
            authority=authority,
            expected_slots=audio_application.expected_audio_slots(
                load_records(config.normalized_file.resolve()),
                [READ],
                words=False,
                examples=True,
            ),
            reservations=reserved,
        )
        [sentence] = proof.slots
        assert sentence.origin == "synthesized"
        assert sentence.paid_operation is not None
        assert sentence.paid_operation.operation_id == reserved[0].operation_id
        audio_application.revalidate_audio_completion(config, proof)

    finished = ledger_mod.load(config.ledger_file)
    assert finished.pending_audio == {}
    entry = finished.audio_entry_for(
        READ,
        of="example",
        content_fp=sentence.content_fingerprint,
        provider=sentence.provider.name,
        voice=sentence.provider.voice,
        speed=sentence.provider.speed,
        settings=dict(sentence.provider.settings),
    )
    assert entry is not None
    assert entry[audio_cmd.PAID_ATTEMPT]["operation_id"] == reserved[0].operation_id
