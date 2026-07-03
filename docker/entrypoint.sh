#!/bin/sh
set -eu

CODING_ASSISTANT_MANAGER_DATA_DIR=/data uv --project /app run --no-sync alembic upgrade head
exec uv --project /app run --no-sync coding-assistant-manager
