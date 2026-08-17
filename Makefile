.PHONY: bootstrap gates test lint build-sample build-all preview clean check \
        voicevox voicevox-stop audio

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
	if [ -n "$$(docker ps -aq -f name=^$(VOICEVOX_CONTAINER)$$)" ]; then \
		echo "voicevox: starting the existing $(VOICEVOX_CONTAINER) container..."; \
		docker start "$(VOICEVOX_CONTAINER)" >/dev/null; \
	else \
		echo "voicevox: running $(VOICEVOX_IMAGE) as $(VOICEVOX_CONTAINER)..."; \
		docker run -d -p 50021:50021 --name "$(VOICEVOX_CONTAINER)" "$(VOICEVOX_IMAGE)" >/dev/null; \
	fi; \
	printf 'voicevox: waiting for the engine'; \
	for _ in $$(seq 1 60); do \
		if curl -sf -o /dev/null --max-time 2 "$(VOICEVOX_URL)/version"; then \
			echo " — up."; exit 0; \
		fi; \
		printf '.'; sleep 1; \
	done; \
	echo; \
	echo "voicevox: the engine did not answer within 60s. 'docker logs $(VOICEVOX_CONTAINER)' says why." >&2; \
	exit 1

## voicevox-stop: stop the engine this Makefile started.
voicevox-stop:
	@docker stop "$(VOICEVOX_CONTAINER)" >/dev/null 2>&1 && echo "voicevox: stopped." || echo "voicevox: nothing to stop."

## audio: voice every record that needs it, starting the engine first.
audio: voicevox
	@$(JANKI) audio --words --examples

preview:
	@$(JANKI) preview data/decks/verbs.yaml --output dist/verbs-preview.html

clean:
	rm -f "$(ROOT)"/dist/*.apkg "$(ROOT)"/dist/*.html

# Kept so existing muscle memory and docs keep working.
check: gates
