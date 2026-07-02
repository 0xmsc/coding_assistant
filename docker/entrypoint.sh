#!/bin/sh
set -eu

uv --project /app run --no-sync alembic upgrade head
exec uv --project /app run --no-sync coding-assistant-manager
