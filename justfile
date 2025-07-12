test:
    uv run pytest -n auto

hello-world:
    ./run.fish \
      --disable-user-feedback --disable-feedback-agent \
      --task "Say 'Hello World'"
