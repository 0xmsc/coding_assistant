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

deploy:
    #!/usr/bin/env bash
    set -euo pipefail
    git fetch --quiet origin 'refs/tags/deployed/*:refs/tags/deployed/*'
    if [[ -n "$(git tag --points-at HEAD --list 'deployed/*')" ]]; then
        echo "Current commit is already deployed."
        exit 0
    fi
    just check
    just publish
    ./docker/deploy.sh

dev-manager:
    docker compose --env-file docker/.env -f docker/compose.manager.yml up --build --watch manager

dev-manager-down:
    docker compose --env-file docker/.env -f docker/compose.manager.yml down
