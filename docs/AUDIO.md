# Audio

Two engines, because the two recordings do different jobs.

**Words are spoken by VOICEVOX with their pitch accent forced.** That is the
whole reason it is here: 橋 and 箸 are the pair a card exists to tell apart, and
an engine left to guess renders them identically — measured, not assumed.
Nothing else in this project can force an accent, so nothing else voices a word.

**Sentences are read naturally**, by VOICEVOX or by OpenAI. Nothing is forced
there — janki has no accent data for a whole sentence and does not pretend to —
so the choice is about which reads Japanese better, and you should listen rather
than take a recommendation.

## Getting VOICEVOX running

It is a local engine: no account, no key, no per-character cost, and it works
offline. The short way is:

```bash
make voicevox
```

which adopts an engine that is already answering, starts the `janki-voicevox`
container if there is one, and pulls the image if there is not — then waits for
it and fails with a real reason if it never comes up. `make audio-words` does
that and then voices the *words* — the local, free half. `make audio` adds the
example sentences, which go through OpenAI and are billed. `make voicevox-stop`
stops the container.

Otherwise, install the [VOICEVOX app](https://voicevox.hiroshiba.jp/) and leave
it open, or run the engine yourself:

```bash
docker run -d -p 50021:50021 --name janki-voicevox \
  voicevox/voicevox_engine:cpu-latest
```

Deliberately not `--rm`: `make voicevox` starts a container of this name rather
than creating a second one, and a `--rm` container disappears the moment it is
stopped — so with `--rm` every stop costs you the image's loaded state and the
next `make voicevox` has to create it again. A `--rm` container works fine
otherwise; nothing here deletes it, it deletes itself.

janki talks to `http://localhost:50021` by default; set `tts.voicevox_url` if
yours listens elsewhere. `janki audio` checks the engine is answering *before*
it synthesizes anything, so the ordinary startup failure costs no work. A
later provider or filesystem failure is recoverable per clip, as described
under [Interrupted audio and recovery](#interrupted-audio-and-recovery).

If you run it in a container, note that it loads a model per speaker on demand
and keeps them: a 2 GiB colima VM gets through about nine speakers before the
container is OOM-killed. That only bites when auditioning many voices at once —
`scripts/voice-samples.py` cycles the container itself to work around it.

## Generating the audio

```bash
janki audio --words --examples
```

Both kinds are off unless asked for. Every word gets a clip: when janki has
the pitch pattern it forces the accent, and when it does not the engine picks
one — those clips are reported and tagged `accent_unverified` in the ledger.
That is usually not permanent. Word audio currency binds the forced/natural
mode and **the exact one string sent to the provider** — the AquesTalk request
for a forced clip, or the bare reading for a natural one — rather than the
stored pattern. Once `janki enrich --jpdb` fills a *usable* accent, the guessed
clip reads as stale and the next `janki audio` replaces it with a forced one,
no `--force` needed.

The exception is a pattern janki cannot convert: a length that does not match
the reading, or a long vowel with no vowel to repeat. Those clips are voiced
with the engine's own accent, which is the same utterance as no pattern at
all — so they fingerprint the same, and the clip does not re-voice when the
pattern changes from one unusable value to another. `janki audio --force`
re-voices regardless.

`janki audio --words` names every one of these it voices, with the record id
and the reason, and that is the backstop for anything in the collection: a
merge that took `--prefer-incoming`, a hand edit of `vocabulary.json`, a
pattern that was fine until its reading was corrected. Two gaps in it are worth
knowing. A run without `--words` — `--examples` alone — names none of them,
because the check lives in the word-audio path. And it reads `vocabulary.json`
only, so a record living in a deck's inline `notes:` block is never voiced and
never checked; `janki status` and `janki validate` do count those records, so
they are not invisible, just unvoiced.

Three commands say it *earlier*, which is the point: before a clip exists to be
wrong. `janki enrich --jpdb` refuses to write an unusable pattern and names the
reason. `janki import-jpdb` keeps it — jpdb's answer is data you may want to
correct — and names it, saying whether it is the pattern the clip will actually
use. `janki promote` checks a staged row, and checks `audio_accent` as well as
`pitch_accent`, because `audio_accent` is the one that reaches the synthesizer
when it is set. No importer and no enrichment pass writes it, so a staged row
is the usual place a hand-typed one first appears — though not the only one: it
is a mergeable field, so `janki migrate-inline` can carry one in from a deck's
inline note without passing through promote at all.

This is still the one case where filling in an accent does not by itself
replace the clip.

`--prune` removes clips no record or pending audio transaction still owns,
taking their canonical ledger entries with them. It never deletes a staged
paid render waiting to be recovered.

## Choosing a voice

```toml
[tts]
voicevox_speaker = 53               # speaks the words, accent forced
voicevox_speed = 0.7                # below 1 slows delivery; the engine's own
                                    # time-stretch, so the pitch does not drop
```

VOICEVOX ships 40-odd speakers, most with several styles. To hear them rather
than read a list:

```bash
python3 scripts/voice-samples.py            # every speaker, plus a page to compare them
python3 scripts/voice-samples.py --male     # just the male voices
```

That writes clips and an `index.html` to `~/Desktop/janki-voice-samples`
(`--out` to put them elsewhere). Each one runs through janki's own forced-accent
path, so the sample word carries its real accent rather than the engine's guess
— what you hear is what a card will sound like. Use `--word/--reading/--pattern`
to audition with a word you care about.

Sentences can take a different voice, or a different engine:

```toml
[tts]
voicevox_sentence_speaker = 52      # another VOICEVOX voice for sentences
```

```toml
[tts]
sentence_provider = "openai"        # OpenAI reads the sentences instead
openai_voice = "onyx"               # alloy, ash, ballad, cedar, coral, echo,
                                    # fable, marin, nova, onyx, sage, shimmer,
                                    # verse. (Cove and the other ChatGPT app
                                    # voices are a different set — not this API's.)
openai_model = "gpt-4o-mini-tts"    # or a pinned snapshot like
                                    # gpt-4o-mini-tts-2025-12-15
```

OpenAI needs `OPENAI_API_KEY` in the environment — never in `janki.toml` — and
the default model bills for text-input and audio-output tokens. The speech API
also accepts a numeric `speed`, but janki deliberately leaves it at the API's
1.0 default: `voicevox_speed` belongs to the word engine, and letting that knob
reach OpenAI would re-render and re-bill every sentence when only the word pace
changed. Sentence pace and pronunciation steering are asked for in prose
through `openai_instructions`; the shipped default asks for a noticeably slower
delivery. Leave `sentence_provider` unset to keep sentences on VOICEVOX; unless
`voicevox_sentence_speaker` is set, one voice then does everything.

If one sentence still needs pronunciation help, add sparse, human-owned
instructions to that example in `vocabulary.json`:

```json
{
  "japanese": "毎日、日本語を話します。",
  "instructions": "Pronounce 毎日 as まいにち."
}
```

The clip-specific instruction is appended to the configured baseline,
separated by a blank line; it does not replace the learner pace and voice
guidance. This escape hatch requires `sentence_provider = "openai"`—a
VOICEVOX example run refuses before synthesizing anything rather than
pretending it honored prose steering. The ledger stores the exact effective
instructions sent for each clip, so an edit makes only that clip stale and
`janki audio --examples` rewrites it in place. Empty instructions stay absent
from the record and preserve the exact legacy render profile, so this feature
itself triggers no regeneration.

The current `/v1/audio/speech` schema lists `tts-1`, `tts-1-hd`,
`gpt-4o-mini-tts`, and its dated snapshot. The API does not apply
`instructions` to `tts-1` or `tts-1-hd`, so janki refuses those models while
instructions are configured rather than recording steering they ignored.
Those two older models also support only `alloy`, `ash`, `coral`, `echo`,
`fable`, `onyx`, `nova`, `sage`, and `shimmer`; the 13-name voice list above is
for the shipped GPT-4o mini TTS model. A blank model is refused locally. Both
the spoken input and the effective instructions must be valid UTF-8 and have a
4,096-character API limit; janki checks the whole selected run before
synthesis, so a deterministic request refusal cannot land earlier paid clips.
The
conversational audio models (`gpt-audio`, `gpt-audio-1.5`, `gpt-audio-mini`)
generate speech natively, which sounds like it should be better — but they
*respond* to text rather than reading it. Asked to read
「日本語を話しますか。」 they answer it; the measurements are in
M5.7 of `IMPLEMENTATION_PLAN.md`. The realtime family (`gpt-realtime-*`)
was not tested: it is a live speech-to-speech session, a different shape from
writing a file to disk.

## Changing a voice re-voices only what that voice said

The ledger records which engine, which voice, which rate and which style
settings made every clip, so changing any of them makes exactly those clips
stale and leaves the rest alone:

```bash
janki audio --examples       # after changing the sentence voice; words untouched
```

No `--force` needed. That flag remains for rewriting audio the settings did not
change.

Within one engine, filenames are identity-addressed—record id for a word, and
record id plus Japanese sentence text for an example. They are unchanged by a
voice or instruction edit, so clips are rewritten in place and Anki's media
sync picks up the new audio behind the same `[sound:]` references. Two records
that happen to share a sentence still receive two files; reordering examples
changes neither address.

> **Unverified, and worth one manual check.** That last sentence is true of
> AnkiWeb sync. janki delivers through `janki build` → `.apkg` → import, and
> nothing here pins what your Anki version does when an incoming package
> carries a media file whose *name* matches one already in `collection.media`
> but whose *bytes* differ. If it keeps the existing file, a re-voice is
> silently ignored on import — which matters most for a clip upgrading from a
> guessed accent to a forced one, since the note text does not change either.
> If a re-voiced clip sounds unchanged after an import, delete the file from
> `collection.media` and import again. **Switching engines changes the file
> extension** — VOICEVOX writes `.wav`, OpenAI `.mp3` — so the note's `[sound:]`
> reference is repointed and the old clip is left behind unreferenced. Follow
> that one with `janki audio --examples --prune`.

## Interrupted audio and recovery

Each paid result follows a small write-ahead transaction:

1. janki writes the bytes under `data/media/audio/.pending/` and records their
   exact request, provider profile, target filename, and SHA-256 in the sparse
   top-level `pending_audio` block of `data/ledger.json`.
2. It compare-and-swap saves `vocabulary.json`, refusing to overwrite a human
   or another command that changed the records during synthesis.
3. Only after that record write wins does it atomically publish the staged
   bytes at the canonical `janki-*` filename, install the ordinary per-record
   audio ledger entry, clear the pending row, and remove the stage.

The stage filename binds the exact request key and byte SHA. If the process is
interrupted in the narrow gap before its WAL row is merged, the exact rerun can
reconstruct that row from the self-verifying stage rather than paying again.
The pending row is otherwise durable after each completed clip, not merely at
the end of a batch. `janki status` reports recovery separately from missing or
stale audio, and a build refuses while one is unresolved: rerun the same
`janki audio` selection first. If the record content, target, and complete
render profile are still identical, that rerun verifies the staged SHA and
adopts the paid bytes without another provider call. A relevant edit to the
sentence, reading, instructions, voice, or model is a different request and is
never allowed to adopt them. A corrupt matching stage refuses before another
provider call; `--force` is the explicit authorization to replace it.
`--prune` protects both the pending target and its stage while recovery is
possible, and records ledger removals before deleting unreferenced bytes. The
audio transaction holds the normalized file, every deck definition, and every
deck source file it read through publication and prune; if that owner set
changes, the command refuses before moving canonical bytes. After a successful
run it removes self-verifying stages that neither a WAL row nor any current
exact request can claim, while preserving a recoverable stage for an
unselected current record.

This is why a re-voice interrupted part way — an OOM, a dropped connection, a
concurrent record edit, or a final ledger-write failure — is normally finished
simply by running the exact command again.

## If you share a deck

VOICEVOX voices are free to use, **but each character carries its own terms**,
and most ask to be credited. That is a question for a deck you publish, not for
one you study alone. Check the terms for the speaker you chose at
[voicevox.hiroshiba.jp](https://voicevox.hiroshiba.jp/) and credit it in the
deck description. OpenAI's
[text-to-speech guide](https://developers.openai.com/api/docs/guides/text-to-speech)
says end users must be clearly told that the voice is AI-generated rather than
human; that is a disclosure rather than character attribution, but it belongs
in a shared deck's description too.
