#!/bin/sh
set -eu

export CODING_ASSISTANT_MANAGER_DATA_DIR=/data

if [ ! -f "$CODING_ASSISTANT_MANAGER_DATA_DIR/sessions.sqlite" ] ||
    ! uv --project /app run --no-sync alembic current --check-heads
then
    if [ -f "$CODING_ASSISTANT_MANAGER_DATA_DIR/sessions.sqlite" ]; then
        uv --project /app run --no-sync python -m coding_assistant.manager.migration_backup
    fi
    uv --project /app run --no-sync alembic upgrade head
fi

exec uv --project /app run --no-sync coding-assistant-manager
