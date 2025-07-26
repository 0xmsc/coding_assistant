test:
    uv run pytest -n auto

hello-world:
    ./run.fish \
      --no-user-feedback --no-feedback-agent \
      --task "Say 'Hello World'"

commit:
    ./run.fish \
    --task "Review the uncommited changes and commit them, if they're okay. You can also create multiple commits if you want to."

review:
    ./run.fish \
    --task "Review changes in the current branch like a senior engineer would do. Stop if there are uncommitted changes, or the branch cannot be diffed against master. If you find issues, ask the client if you should fix them. If you fix something, commit the changes and continue reviewing."

pr:
    ./run.fish \
    --task "Create a PR with the `gh` tool. Create a proper PR title and description."

