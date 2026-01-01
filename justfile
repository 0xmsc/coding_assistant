ci:
    ./scripts/ci.sh

lint:
    ./scripts/ci.sh --fix

test:
    ./scripts/ci.sh

test-integration:
    uv run coding-assistant \
        --model "google/gemini-3-flash-preview (medium)" \
        --trace \
        --no-ask-user \
        --task "Test the tools out your MCP server. Test all provided functionalities. Try to test corner cases that you think could fail. Test how ergonomic your tools are. Prepare a test report."
