test:
    make test

test-integration:
    make test-integration

lint:
    make lint

check-migrations:
    #!/usr/bin/env bash
    set -euo pipefail
    data_dir="$(mktemp -d -t coding-assistant-migrations-XXXXXX)"
    trap 'rm -rf "$data_dir"' EXIT
    CODING_ASSISTANT_MANAGER_DATA_DIR="$data_dir" uv run alembic upgrade head
    CODING_ASSISTANT_MANAGER_DATA_DIR="$data_dir" uv run alembic check

publish:
    ./docker/publish.sh

dev-manager:
    docker compose --env-file docker/.env -f docker/compose.manager.yml up --watch manager

dev-manager-down:
    docker compose --env-file docker/.env -f docker/compose.manager.yml down
