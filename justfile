test:
    uv run pytest -n auto

hello-world:
    ./run.fish \
      --no-user-feedback --no-feedback-agent \
      --task "Say 'Hello World'"
