test:
    uv run pytest -n auto -m "not slow"
    uv run --directory packages/coding_assistant_mcp pytest -n auto


lint:
    uv run ruff check --fix src/coding_assistant
    uv run ruff format src/coding_assistant
    uv run mypy src/coding_assistant

    uv run --directory packages/coding_assistant_mcp ruff check --fix
    uv run --directory packages/coding_assistant_mcp ruff format
    uv run --directory packages/coding_assistant_mcp mypy .

