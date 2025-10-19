#!/usr/bin/env fish

# set fish_trace 1

set project_dir (dirname (status filename))
set mcp_project_dir $project_dir/packages/coding_assistant_mcp
set mcp_json_config (printf '{"name": "coding_assistant_mcp", "command": "uv", "args": ["--project", "%s", "run", "coding-assistant-mcp"], "env": []}' "$mcp_project_dir")

uv --project $project_dir run coding-assistant \
    --model "openai/gpt-5 (medium)" \
    --expert-model "openai/gpt-5 (high)" \
    --readable-sandbox-directories /mnt/wsl ~/.ssh ~/.rustup \
    --writable-sandbox-directories "$project_dir" /tmp /dev/shm \
    --mcp-servers \
        $mcp_json_config \
        '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "~"]}' \
    $argv
