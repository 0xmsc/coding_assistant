.PHONY: test test-integration lint lint-check ci

all: ci

ci: lint-check test

test:
	uv run pytest -n auto -m "not slow"

test-integration:
	uv run pytest -m slow

lint:
	uv run ruff check --fix
	uv run ruff format
	uv run mypy .

lint-check:
	uv run ruff check
	uv run ruff format --check
	uv run mypy .
