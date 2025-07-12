#!/usr/bin/env fish

set sandbox_dirs \
    /tmp \
    /mnt/wsl

set mcp_servers \
    '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "{working_directory}"], "env": []}' \
    '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"], "env": []}'

uv --project (dirname (status filename)) run coding-assistant \
    --model "gemini/gemini-2.5-flash" \
    --expert-model "gemini/gemini-2.5-pro" \
    --sandbox-directories $sandbox_dirs \
    --mcp-servers $mcp_servers \
    $argv
