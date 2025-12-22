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

test-integration:
    uv run coding-assistant \
        --model "openrouter/google/gemini-3-flash-preview (medium)" \
        --trace \
        --task "Test the tools out your MCP server. Test all provided functionalities. Try to test corner cases that you think could fail. Test how ergonomic your tools are. Prepare a test report."
