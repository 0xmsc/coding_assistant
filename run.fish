#!/usr/bin/env fish

uv --project (dirname (status filename)) run coding-assistant \
    --model "gemini/gemini-2.5-flash" \
    --expert-model "gemini/gemini-2.5-pro" \
    --readable-sandbox-directories /mnt/wsl ~/.ssh \
    --writable-sandbox-directories /tmp /dev/shm \
    --shell-confirmation-patterns "^git rebase", "^gh" \
    --mcp-servers \
        '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "{working_directory}"]}' \
        '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}' \
        '{"name": "tavily", "command": "npx", "args": ["-y", "tavily-mcp"], "env": ["TAVILY_API_KEY"]}' \
    $argv
