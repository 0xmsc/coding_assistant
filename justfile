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

hello-world:
    ./run.fish \
      --task "Say 'Hello World'"

commit:
    ./run.fish \
    --task "Review the uncommitted changes and commit them, if they're okay. You can also create multiple commits if you want to. You do not necessarily have to run the tests."

review:
    ./run.fish \
    --task "Review changes in the current branch like a senior engineer would do. Stop if there are uncommitted changes, or the branch cannot be diffed against master. If you find issues, ask the client if you should fix them. If you fix something, commit the changes and continue reviewing. Check if there already exists a PR for the branch using the \`gh\` tool. If yes, check if the existing PR has a proper description and title. If not, change the title and description to something more appropriate, again using the \`gh\` tool."

fixlint:
    ./run.fish \
    --task "Run \`just lint\` and fix the problems. If a fix would be invasive, ask the client if you should do it."

