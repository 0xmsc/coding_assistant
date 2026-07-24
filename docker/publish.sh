#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="forgejo.schneiderm.net/marcel/coding-assistant:latest"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to publish: the Git worktree is not clean." >&2
    exit 1
fi

VCS_REF="$(git rev-parse HEAD)"
REVISION="$(git rev-parse --short=8 HEAD)"
CREATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "Building Docker image..."
docker build \
    --pull \
    --build-arg "CODING_ASSISTANT_REVISION=$REVISION" \
    --label "org.opencontainers.image.created=$CREATED_AT" \
    --label "org.opencontainers.image.revision=$VCS_REF" \
    --label "org.opencontainers.image.source=https://forgejo.schneiderm.net/marcel/coding_assistant" \
    -f docker/Dockerfile \
    -t "$IMAGE_TAG" \
    .

echo "Pushing Docker image..."
docker push "$IMAGE_TAG"

echo "Successfully published $IMAGE_TAG"
