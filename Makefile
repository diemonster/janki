.PHONY: bootstrap test lint check build-sample build-all preview clean

bootstrap:
	./scripts/bootstrap.sh

test:
	pytest

lint:
	ruff check .

check: lint test
	janki validate data/decks/verbs.yaml

build-sample:
	janki build data/decks/verbs.yaml --output dist/sample-verbs.apkg

build-all:
	janki build --all

preview:
	janki preview data/decks/verbs.yaml --output dist/verbs-preview.html

clean:
	rm -f dist/*.apkg dist/*.html
