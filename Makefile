POETRY ?= poetry
PYTHON ?= $(POETRY) run python

.PHONY: install lint test format type-check build

install:
	$(POETRY) install

lint:
	$(POETRY) run ruff check src tests

format:
	$(POETRY) run ruff format src tests

type-check:
	$(POETRY) run mypy src

test:
	$(POETRY) run pytest

build:
	$(POETRY) build
