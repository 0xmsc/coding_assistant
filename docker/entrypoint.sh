#!/bin/sh
set -eu

uv --project /app run --no-sync python -m coding_assistant.manager.migrations
exec uv --project /app run --no-sync coding-assistant-manager
