.PHONY: test lint lint-check ci test-integration

# Default target
all: ci

# CI target (strictly checks, no auto-fixing)
ci: lint-check test

# Standard test runner
test:
	uv run pytest -n auto -m "not slow"

# Development linting (auto-fixes)
lint:
	uv run ruff check --fix
	uv run ruff format
	uv run mypy .

# CI linting (fail if not formatted/invalid)
lint-check:
	uv run ruff check
	uv run ruff format --check
	uv run mypy .

# Integration tests
test-integration:
	uv run coding-assistant \
		--model "google/gemini-3-flash-preview (medium)" \
		--trace \
		--no-ask-user \
		--task "Test the tools out your MCP server. Test all provided functionalities. Try to test corner cases that you think could fail. Test how ergonomic your tools are. Prepare a test report."
