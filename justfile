test:
    uv run pytest -n auto

hello-world:
    ./run.fish \
      --no-user-feedback --no-feedback-agent \
      --task "Say 'Hello World'"

commit:
    ./run.fish \
    --task "Review the unstaged changes and commit them, if they're okay. You can also create multiple commits if you want to."
