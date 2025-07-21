#!/usr/bin/env fish

uv --project (dirname (status filename)) run coding-assistant \
    --model "gemini/gemini-2.5-flash" \
    --expert-model "gemini/gemini-2.5-pro" \
    --sandbox-directories /tmp /mnt/wsl \
    --mcp-servers \
        '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "{working_directory}"]}' \
        '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}' \
        '{"name": "tavily", "command": "npx", "args": ["-y", "tavily-mcp"], "env": ["TAVILY_API_KEY"]}' \
    $argv
