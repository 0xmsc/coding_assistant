#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

ssh server <<'EOF'
set -e
cd ~/Server/containers
docker pull forgejo.schneiderm.net/marcel/coding-assistant
docker compose up -d coding-assistant-manager
EOF

deployed_at="$(date -u +%Y%m%dT%H%M%SZ)"
git push origin "HEAD:refs/tags/deployed/$deployed_at"
