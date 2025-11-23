#!/usr/bin/env fish

# set fish_trace 1

set project_dir (dirname (status filename))
set mcp_project_dir $project_dir/packages/coding_assistant_mcp
set mcp_json_config (printf '{"name": "coding_assistant_mcp", "command": "uv", "args": ["--project", "%s", "run", "coding-assistant-mcp"], "env": []}' "$mcp_project_dir")

uv --project $project_dir run coding-assistant \
    --model "openrouter/openai/gpt-5.1-codex (medium)" \
    --expert-model "openrouter/openai/gpt-5.1-codex (high)" \
    --readable-sandbox-directories /mnt/wsl ~/.ssh ~/.rustup \
    --writable-sandbox-directories "$project_dir" /tmp /dev/shm ~/.cache/coding_assistant \
    --mcp-servers \
        $mcp_json_config \
        '{"name": "context7", "command": "npx", "args": ["-y", "@upstash/context7-mcp"]}' \
    $argv
