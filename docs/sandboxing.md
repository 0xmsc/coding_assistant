# External Sandboxing

Built-in sandboxing is no longer part of this project. If you want filesystem isolation, run the assistant inside an external sandbox such as `bubblewrap`.

This example keeps the normal home directory readable, makes the current working directory writable, and allows writable cache directories for `uv` and `coding_assistant`:

```bash
bwrap \
  --die-with-parent \
  --new-session \
  --ro-bind / / \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --tmpfs /dev/shm \
  --bind "$HOME/.cache/uv" "$HOME/.cache/uv" \
  --bind "$HOME/.cache/uv" "$HOME/.cache/coding_assistant" \
  --bind "$PWD" "$PWD" \
  --chdir "$PWD" \
  uv run coding-assistant --model 'minimax/minimax-m2.7'
```

The broad read-only mount comes first. The later `--bind` arguments create writable exceptions for the directories that must persist between runs.
