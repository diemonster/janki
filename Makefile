.PHONY: bootstrap gates test lint build-sample build-all preview clean check

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
	@$(JANKI) build data/decks/verbs.yaml >/dev/null
	@echo "gates: ruff clean, pytest green, sample deck builds  [$(ROOT)]"

lint:
	@$(RUN) -m ruff check "$(ROOT)"

test:
	@$(RUN) -m pytest

build-sample:
	@$(JANKI) build data/decks/verbs.yaml --output dist/sample-verbs.apkg

build-all:
	@$(JANKI) build --all

preview:
	@$(JANKI) preview data/decks/verbs.yaml --output dist/verbs-preview.html

clean:
	rm -f "$(ROOT)"/dist/*.apkg "$(ROOT)"/dist/*.html

# Kept so existing muscle memory and docs keep working.
check: gates
