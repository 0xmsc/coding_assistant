#!/usr/bin/env fish

uv --project (dirname (status filename)) run coding-assistant \
    --model "openai/gpt-5" \
    --expert-model "openai/gpt-5" \
    --readable-sandbox-directories /mnt/wsl ~/.ssh \
    --writable-sandbox-directories /tmp /dev/shm \
    --mcp-servers \
        '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "{home_directory}"]}' \
        '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}' \
        '{"name": "context7", "command": "npx", "args": ["-y", "@upstash/context7-mcp"], "env": []}' \
        '{"name": "tavily", "command": "npx", "args": ["-y", "tavily-mcp"], "env": ["TAVILY_API_KEY"]}' \
    $argv
