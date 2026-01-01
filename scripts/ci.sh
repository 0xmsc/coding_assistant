#!/bin/bash
set -e

# Support --fix flag for local development
FIX_ARG=""
if [[ "$1" == "--fix" ]]; then
    FIX_ARG="--fix"
fi

echo "Running ruff check..."
uv run ruff check $FIX_ARG src/coding_assistant

echo "Running ruff format..."
if [[ "$FIX_ARG" == "--fix" ]]; then
    uv run ruff format src/coding_assistant
else
    uv run ruff format --check src/coding_assistant
fi

echo "Running tests..."
uv run pytest -n auto -m "not slow"

echo "Running mypy..."
uv run mypy src/coding_assistant
