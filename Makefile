.PHONY: bootstrap gates test lint build-sample build-all preview clean check \
        voicevox voicevox-stop audio audio-words help

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

# The default goal, so a bare `make` lists what there is rather than running
# the first target it finds. `##` lines above a target are its help text; they
# were a convention with nothing reading them until this existed.
.DEFAULT_GOAL := help

help:
	@echo "janki — make targets:"
	@grep -hE '^## [a-z-]+:' "$(ROOT)/Makefile" | sed -e 's/^## /  /' | sort
	@echo
	@echo "Undocumented helpers: bootstrap, lint, test, build-sample, build-all, preview, clean."

bootstrap:
	@cd "$(ROOT)" && ./scripts/bootstrap.sh

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
# Every docker and curl step's exit status is checked, including the two command
# substitutions. The recipe is one backslash-joined shell with no `set -e`, and
# make only sees the last command, so an unchecked failure does not stop it — it
# falls through to the poll loop and reports a 60-second timeout for something
# that failed instantly. `docker info` succeeding does not make `docker ps`
# succeed: a draining daemon, a switched context or `DOCKER_HOST`, an
# API-version mismatch. An unchecked `ps` failing empty routes the *create*
# branch, which then announces a container it never ran — the original bug,
# reachable again through a different door.
#
# `docker start`, never `docker restart`. This is reached whenever curl failed,
# and "curl failed" includes a container that is running and simply not
# answering yet — VOICEVOX loads a speaker model on boot. `start` is a no-op on
# a running container and lets the poll loop do its job; `restart` would tear
# down the loaded-model cache on the common path to fix the rare one.
voicevox:
	@if ! command -v curl >/dev/null 2>&1; then \
		echo "voicevox: curl is not installed, so this cannot tell whether the engine is up." >&2; \
		echo "  Install curl, or start the engine yourself and run janki directly." >&2; \
		exit 1; \
	fi; \
	if curl -sf -o /dev/null --max-time 2 "$(VOICEVOX_URL)/version"; then \
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
	if ! existing="$$(docker ps -aq -f name=^$(VOICEVOX_CONTAINER)$$)"; then \
		echo "voicevox: 'docker ps' failed, so this cannot tell what is running." >&2; \
		echo "  The daemon answered 'docker info' — check DOCKER_HOST and 'docker context ls'." >&2; \
		exit 1; \
	fi; \
	if [ -n "$$existing" ]; then \
		echo "voicevox: starting the existing $(VOICEVOX_CONTAINER) container..."; \
		if ! docker start "$(VOICEVOX_CONTAINER)" >/dev/null; then \
			echo "voicevox: 'docker start $(VOICEVOX_CONTAINER)' failed — the message above says why." >&2; \
			echo "  Often port 50021 is taken by a VOICEVOX.app or another container." >&2; \
			echo "  'docker logs $(VOICEVOX_CONTAINER)' says more; 'docker rm -f $(VOICEVOX_CONTAINER)' starts over." >&2; \
			exit 1; \
		fi; \
	else \
		echo "voicevox: running $(VOICEVOX_IMAGE) as $(VOICEVOX_CONTAINER)..."; \
		if ! docker run -d -p 50021:50021 --name "$(VOICEVOX_CONTAINER)" "$(VOICEVOX_IMAGE)" >/dev/null; then \
			echo "voicevox: could not start the container — the message above says why." >&2; \
			echo "  If it names port 50021, something is already serving it: close the" >&2; \
			echo "  VOICEVOX app, or stop whatever holds it. If it names the container" >&2; \
			echo "  name, 'docker rm -f $(VOICEVOX_CONTAINER)' clears the leftover." >&2; \
			exit 1; \
		fi; \
	fi; \
	printf 'voicevox: waiting for the engine'; \
	for _ in $$(seq 1 60); do \
		if curl -sf -o /dev/null --max-time 2 "$(VOICEVOX_URL)/version"; then \
			echo " — up."; exit 0; \
		fi; \
		if running="$$(docker ps -q -f name=^$(VOICEVOX_CONTAINER)$$)" && [ -z "$$running" ]; then \
			echo; \
			echo "voicevox: the container exited while starting up." >&2; \
			echo "  'docker logs $(VOICEVOX_CONTAINER)' says why." >&2; \
			exit 1; \
		fi; \
		printf '.'; sleep 1; \
	done; \
	echo; \
	echo "voicevox: 60 tries and the engine never answered." >&2; \
	echo "  'docker logs $(VOICEVOX_CONTAINER)' says why. If it looks wedged rather" >&2; \
	echo "  than slow, 'docker restart $(VOICEVOX_CONTAINER)'." >&2; \
	exit 1

## voicevox-stop: stop the container named $(VOICEVOX_CONTAINER).
#
# Not "the one this Makefile started" — it cannot know that. An engine you
# opened as the VOICEVOX app is not a container at all, which is the case that
# reports no container to stop while the engine keeps answering. `docker stop`
# only stops; a container created with `--rm` also disappears, but that is a
# property of how it was created rather than of who stopped it.
voicevox-stop:
	@docker stop "$(VOICEVOX_CONTAINER)" >/dev/null 2>&1 && echo "voicevox: stopped $(VOICEVOX_CONTAINER)." || echo "voicevox: no $(VOICEVOX_CONTAINER) container to stop."

## audio: voice words locally and example sentences through OpenAI (billed).
#
# `--examples` goes to whichever `sentence_provider` janki.toml names, which is
# OpenAI — a paid API needing OPENAI_API_KEY. Only the `--words` half is the
# local engine `voicevox` guarantees. `make audio-words` is the free half.
audio: voicevox
	@$(JANKI) audio --words --examples

## audio-words: voice words only — local VOICEVOX, free, no API key.
audio-words: voicevox
	@$(JANKI) audio --words

preview:
	@$(JANKI) preview data/decks/verbs.yaml --output dist/verbs-preview.html

clean:
	rm -f "$(ROOT)"/dist/*.apkg "$(ROOT)"/dist/*.html

# Kept so existing muscle memory and docs keep working.
check: gates
