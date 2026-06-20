test:
    make test

test-integration:
    make test-integration

lint:
    make lint

publish:
    ./docker/publish.sh

dev-manager:
    docker compose --env-file docker/.env -f docker/compose.manager.yml watch manager

dev-manager-down:
    docker compose --env-file docker/.env -f docker/compose.manager.yml down
