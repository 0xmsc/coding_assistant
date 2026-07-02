test:
    make test

test-integration:
    make test-integration

lint:
    make lint

check-migrations:
    tmp_dir="$(mktemp -d)"; trap 'rm -rf "$tmp_dir"' EXIT; CODING_ASSISTANT_MANAGER_DATA_DIR="$tmp_dir" uv run alembic upgrade head; CODING_ASSISTANT_MANAGER_DATA_DIR="$tmp_dir" uv run alembic check

publish:
    ./docker/publish.sh

dev-manager:
    docker compose --env-file docker/.env -f docker/compose.manager.yml watch manager

dev-manager-down:
    docker compose --env-file docker/.env -f docker/compose.manager.yml down
