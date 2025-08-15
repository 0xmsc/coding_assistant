#!/usr/bin/env fish

uv --project (dirname (status filename)) run coding-assistant \
    --model "gemini/gemini-2.5-flash" \
    --expert-model "gemini/gemini-2.5-pro" \
    --readable-sandbox-directories /mnt/wsl ~/.ssh \
    --writable-sandbox-directories /tmp /dev/shm ~/.serena \
    --instructions "- Prefer the tools from the `serena` MCP server." \
    --mcp-servers \
        '{"name": "serena", "command": "uvx", "args": ["--from", "git+https://github.com/oraios/serena@d5f90710676b6a7cacc450f59005b4934c49b6db", "serena", "start-mcp-server", "--project", "{working_directory}"], "env": []}' \
        '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch@2025.4.7"]}' \
        '{"name": "context7", "command": "npx", "args": ["-y", "@upstash/context7-mcp@1.0.14"], "env": []}' \
        '{"name": "tavily", "command": "npx", "args": ["-y", "tavily-mcp@0.2.9"], "env": ["TAVILY_API_KEY"]}' \
    $argv
