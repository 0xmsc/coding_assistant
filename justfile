test:
    make test

test-integration:
    make test-integration

lint:
    make lint

lint-check:
    make lint-check

check-migrations:
    #!/usr/bin/env bash
    set -euo pipefail
    data_dir="$(mktemp -d -t coding-assistant-migrations-XXXXXX)"
    trap 'rm -rf "$data_dir"' EXIT
    CODING_ASSISTANT_MANAGER_DATA_DIR="$data_dir" uv run alembic upgrade head
    CODING_ASSISTANT_MANAGER_DATA_DIR="$data_dir" uv run alembic check

check: lint-check check-migrations test

publish:
    ./docker/publish.sh

deploy: check
    #!/usr/bin/env bash
    set -euo pipefail
    just publish
    ssh server <<'EOF'
    set -e
    cd ~/Server/containers
    docker pull forgejo.schneiderm.net/marcel/coding-assistant
    docker compose up -d coding-assistant-manager
    EOF

dev-manager:
    docker compose --env-file docker/.env -f docker/compose.manager.yml up --build --watch manager

dev-manager-down:
    docker compose --env-file docker/.env -f docker/compose.manager.yml down
