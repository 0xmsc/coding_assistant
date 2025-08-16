#!/usr/bin/env fish

uv --project (dirname (status filename)) run coding-assistant \
    --model "openai/gpt-5" \
    --expert-model "openai/gpt-5" \
    --readable-sandbox-directories /mnt/wsl ~/.ssh \
    --writable-sandbox-directories /tmp /dev/shm \
    --no-truncate-tools "^mcp_context7_get-library-docs" \
    --mcp-servers \
        '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem@2025.7.29", "/"]}' \
        '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch@2025.4.7"]}' \
        '{"name": "context7", "command": "npx", "args": ["-y", "@upstash/context7-mcp@1.0.14"], "env": []}' \
        '{"name": "tavily", "command": "npx", "args": ["-y", "tavily-mcp@0.2.9"], "env": ["TAVILY_API_KEY"]}' \
    $argv
