#!/bin/bash
set -e

echo "Running tests..."
uv run pytest -n auto -m "not slow"

echo "Running ruff check..."
uv run ruff check src/coding_assistant

echo "Running ruff format check..."
uv run ruff format --check src/coding_assistant

echo "Running mypy..."
uv run mypy src/coding_assistant
