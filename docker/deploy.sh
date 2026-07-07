#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

just publish

ssh server <<'EOF'
set -e
cd ~/Server/containers
docker pull forgejo.schneiderm.net/marcel/coding-assistant
docker compose up -d coding-assistant-manager
EOF
