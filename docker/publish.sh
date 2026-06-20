#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="forgejo.schneiderm.net/marcel/coding-assistant:latest"

echo "Building Docker image..."
docker build -f docker/Dockerfile -t "$IMAGE_TAG" .

echo "Pushing Docker image..."
docker push "$IMAGE_TAG"

echo "Successfully published $IMAGE_TAG"
