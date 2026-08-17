.PHONY: bootstrap gates test lint build-sample build-all preview clean check \
        voicevox voicevox-stop audio audio-words

# Everything below is anchored to this Makefile's own directory, never to the
# shell's cwd and never to whatever `pytest`/`janki` happen to resolve to on
# PATH. Parallel work here happens in linked git worktrees, where the venv's
# editable install still points at the primary worktree: a bare `pytest` or
# `janki` there runs the *primary* tree's code and reports a green result for
# a branch it never executed. Deriving the paths makes that impossible rather
# than merely discouraged.
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# The venv lives in the primary worktree; linked worktrees borrow it. Ask git
# where the primary tree is rather than hardcoding a path that only works on
# one machine.
PY := $(shell \
	if [ -x "$(ROOT)/.venv/bin/python" ]; then \
		echo "$(ROOT)/.venv/bin/python"; \
	else \
		main=$$(dirname "$$(git -C "$(ROOT)" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"); \
		if [ -x "$$main/.venv/bin/python" ]; then echo "$$main/.venv/bin/python"; else echo python3; fi; \
	fi)

# PYTHONDONTWRITEBYTECODE: a .pyc is validated against its source's mtime in
# whole seconds and byte size, so same-size edits made in the same second
# (a mutation sweep) silently reuse the previous build. conftest.py enforces
# this for pytest; these exports cover the CLI invocations it cannot reach.
RUN := cd "$(ROOT)" && PYTHONPATH="$(ROOT)/src" PYTHONDONTWRITEBYTECODE=1 "$(PY)"
JANKI := $(RUN) -m japanese_anki

bootstrap:
	./scripts/bootstrap.sh

## gates: the definition of done. Run this, not its parts.
gates: lint test
	@$(JANKI) build data/decks/verbs.yaml --output dist/.gates-check.apkg >/dev/null
	@echo "gates: ruff clean, pytest green, sample deck builds  [$(ROOT)]"

lint:
	@$(RUN) -m ruff check "$(ROOT)"

test:
	@$(RUN) -m pytest

build-sample:
	@$(JANKI) build data/decks/verbs.yaml --output dist/sample-verbs.apkg

build-all:
	@$(JANKI) build --all

# The engine `janki audio` needs for words. Local, free, offline — but it has
# to be running, and the failure otherwise arrives *after* you have decided to
# voice a hundred clips.
#
# The container name is the one `docs/AUDIO.md` and `scripts/voice-samples.py`
# already use, so this adopts a running engine rather than fighting it.
VOICEVOX_URL ?= http://localhost:50021
VOICEVOX_CONTAINER ?= janki-voicevox
VOICEVOX_IMAGE ?= voicevox/voicevox_engine:cpu-latest

## voicevox: start the engine if it is not already answering, and wait for it.
#
# Every docker step's exit status is checked. The recipe is one backslash-joined
# shell with no `set -e`, and make only sees the last command, so an unchecked
# failure here does not stop the recipe — it falls through to the poll loop and
# reports a 60-second timeout for something that failed instantly. The two
# likeliest failures are both of that kind: the daemon not running (every
# `docker` call fails, and `docker ps -aq` returning empty then routes the
# *create* branch, which announces a container it never ran), and port 50021
# already bound by a VOICEVOX.app or a stale container.
voicevox:
	@if curl -sf -o /dev/null --max-time 2 "$(VOICEVOX_URL)/version"; then \
		echo "voicevox: already answering at $(VOICEVOX_URL)"; \
		exit 0; \
	fi; \
	if ! command -v docker >/dev/null 2>&1; then \
		echo "voicevox: not answering at $(VOICEVOX_URL), and docker is not installed." >&2; \
		echo "  Open the VOICEVOX app instead — https://voicevox.hiroshiba.jp/" >&2; \
		exit 1; \
	fi; \
	if ! docker info >/dev/null 2>&1; then \
		echo "voicevox: docker is installed but its daemon is not running." >&2; \
		echo "  Start Docker Desktop, or open the VOICEVOX app instead." >&2; \
		exit 1; \
	fi; \
	existing="$$(docker ps -aq -f name=^$(VOICEVOX_CONTAINER)$$)"; \
	if [ -n "$$existing" ]; then \
		echo "voicevox: restarting the existing $(VOICEVOX_CONTAINER) container..."; \
		if ! docker restart "$(VOICEVOX_CONTAINER)" >/dev/null; then \
			echo "voicevox: 'docker restart $(VOICEVOX_CONTAINER)' failed." >&2; \
			echo "  'docker logs $(VOICEVOX_CONTAINER)' says why; 'docker rm -f $(VOICEVOX_CONTAINER)' starts over." >&2; \
			exit 1; \
		fi; \
	else \
		echo "voicevox: running $(VOICEVOX_IMAGE) as $(VOICEVOX_CONTAINER)..."; \
		if ! docker run -d -p 50021:50021 --name "$(VOICEVOX_CONTAINER)" "$(VOICEVOX_IMAGE)" >/dev/null; then \
			echo "voicevox: could not start the container — the message above says why." >&2; \
			echo "  If port 50021 is taken, something is already serving it: close" >&2; \
			echo "  the VOICEVOX app, or 'docker rm -f $(VOICEVOX_CONTAINER)'." >&2; \
			docker rm -f "$(VOICEVOX_CONTAINER)" >/dev/null 2>&1 || true; \
			exit 1; \
		fi; \
	fi; \
	printf 'voicevox: waiting for the engine'; \
	for _ in $$(seq 1 60); do \
		if curl -sf -o /dev/null --max-time 2 "$(VOICEVOX_URL)/version"; then \
			echo " — up."; exit 0; \
		fi; \
		if [ -z "$$(docker ps -q -f name=^$(VOICEVOX_CONTAINER)$$)" ]; then \
			echo; \
			echo "voicevox: the container exited while starting up." >&2; \
			echo "  'docker logs $(VOICEVOX_CONTAINER)' says why." >&2; \
			exit 1; \
		fi; \
		printf '.'; sleep 1; \
	done; \
	echo; \
	echo "voicevox: 60 tries and the engine never answered. 'docker logs $(VOICEVOX_CONTAINER)' says why." >&2; \
	exit 1

## voicevox-stop: stop the container named $(VOICEVOX_CONTAINER).
#
# Not "the one this Makefile started" — it cannot know. `docs/AUDIO.md` gives a
# hand-run command using the same name and `--rm`, so stopping that one deletes
# it; and an engine you opened as the VOICEVOX app is not a container at all,
# which is the case that prints "nothing to stop" while the engine keeps
# answering.
voicevox-stop:
	@docker stop "$(VOICEVOX_CONTAINER)" >/dev/null 2>&1 && echo "voicevox: stopped $(VOICEVOX_CONTAINER)." || echo "voicevox: no $(VOICEVOX_CONTAINER) container to stop."

## audio: voice every record that needs it, starting VOICEVOX first.
#
# `--examples` goes to whichever `sentence_provider` janki.toml names, which is
# OpenAI — a paid API needing OPENAI_API_KEY. Only the `--words` half is the
# local engine `voicevox` guarantees, so this target is free for words and
# billed for sentences. `make audio-words` is the free half alone.
audio: voicevox
	@$(JANKI) audio --words --examples

## audio-words: voice words only — local, free, no API key.
audio-words: voicevox
	@$(JANKI) audio --words

preview:
	@$(JANKI) preview data/decks/verbs.yaml --output dist/verbs-preview.html

clean:
	rm -f "$(ROOT)"/dist/*.apkg "$(ROOT)"/dist/*.html

# Kept so existing muscle memory and docs keep working.
check: gates
