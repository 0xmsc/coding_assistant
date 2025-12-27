test:
    uv run pytest -n auto -m "not slow"

lint:
    uv run ruff check --fix src/coding_assistant
    uv run ruff format src/coding_assistant
    uv run mypy src/coding_assistant

ci:
    #!/usr/bin/env -S parallel --shebang --jobs {{ num_cpus() }}
    just test
    just lint

test-integration:
    uv run coding-assistant \
        --model "openrouter/google/gemini-3-flash-preview (medium)" \
        --trace \
        --no-ask-user \
        --task "Test the tools out your MCP server. Test all provided functionalities. Try to test corner cases that you think could fail. Test how ergonomic your tools are. Prepare a test report."
