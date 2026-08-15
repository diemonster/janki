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
offline. Either install the [VOICEVOX app](https://voicevox.hiroshiba.jp/) and
leave it open, or run the engine on its own:

```bash
docker run --rm -p 50021:50021 --name janki-voicevox \
  voicevox/voicevox_engine:cpu-latest
```

janki talks to `http://localhost:50021` by default; set `tts.voicevox_url` if
yours listens elsewhere. `janki audio` checks the engine is answering *before*
it synthesizes anything, because a run that voices forty clips and then fails on
the forty-first has written forty files and half a ledger.

If you run it in a container, note that it loads a model per speaker on demand
and keeps them: a 2 GiB colima VM gets through about nine speakers before the
container is OOM-killed. That only bites when auditioning many voices at once —
`scripts/voice-samples.py` cycles the container itself to work around it.

## Generating the audio

```bash
janki audio --words --examples
```

Both kinds are off unless asked for. It skips — and reports — anything it is not
sure of: a record with no accent pattern is not voiced with a guess, and an
example carrying a furigana flag is not spoken at all —
`janki enrich --accept RECORD_ID` clears one once you have read the sentence. `--prune` removes
clips no record references any more, taking their ledger entries with them.

## Choosing a voice

```toml
[tts]
voicevox_speaker = 13               # speaks the words, accent forced
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
bills per character. That model has no rate parameter, so pace is asked for in
prose through `openai_instructions`; the shipped default asks for a noticeably
slower delivery. Leave `sentence_provider` unset and one voice does everything.

Only three model families on `/v1/audio/speech` work: `tts-1`, `tts-1-hd`, and
`gpt-4o-mini-tts` with its dated snapshots. The conversational audio models
(`gpt-audio`, `gpt-audio-1.5`, `gpt-audio-mini`) generate speech natively, which
sounds like it should be better — but they *respond* to text rather than reading
it. Asked to read 「日本語を話しますか。」 they answer it; the measurements are in
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

Within one engine, filenames are content-addressed and unchanged by a re-voice,
so clips are rewritten in place and Anki's media sync picks up the new audio
behind the same `[sound:]` references. **Switching engines changes the file
extension** — VOICEVOX writes `.wav`, OpenAI `.mp3` — so the note's `[sound:]`
reference is repointed and the old clip is left behind unreferenced. Follow that
one with `janki audio --examples --prune`.

This also means a re-voice interrupted part way — an OOM, a dropped connection —
is finished simply by running the command again.

## If you share a deck

VOICEVOX voices are free to use, **but each character carries its own terms**,
and most ask to be credited. That is a question for a deck you publish, not for
one you study alone. Check the terms for the speaker you chose at
[voicevox.hiroshiba.jp](https://voicevox.hiroshiba.jp/) and credit it in the
deck description. OpenAI audio has no such attribution requirement.
