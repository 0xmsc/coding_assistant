test:
    uv run pytest -n auto -m "not slow"
    uv run --directory packages/coding_assistant_mcp pytest

hello-world:
    ./run.fish \
    --no-user-feedback \
      --task "Say 'Hello World'"

commit:
    ./run.fish \
    --task "Review the uncommitted changes and commit them, if they're okay. You can also create multiple commits if you want to. You do not necessarily have to run the tests."

review:
    ./run.fish \
    --task "Review changes in the current branch like a senior engineer would do. Stop if there are uncommitted changes, or the branch cannot be diffed against master. If you find issues, ask the client if you should fix them. If you fix something, commit the changes and continue reviewing. Check if there already exists a PR for the branch using the \`gh\` tool. If yes, check if the existing PR has a proper description and title. If not, change the title and description to something more appropriate, again using the \`gh\` tool."

pr:
    ./run.fish \
    --task "Create a PR with the \`gh\` tool. Create a proper PR title and description."
