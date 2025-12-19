#!/usr/bin/env fish

# set fish_trace 1

set -gx LITELLM_MODEL_COST_MAP_URL "https://raw.githubusercontent.com/0xmsc/litellm/ms/openrouter_reasoning_details/model_prices_and_context_window.json"

set project_dir (dirname (status filename))
set mcp_project_dir $project_dir/packages/coding_assistant_mcp
set mcp_json_config (printf '{"name": "coding_assistant_mcp", "command": "uv", "args": ["--project", "%s", "run", "coding-assistant-mcp"], "env": []}' "$mcp_project_dir")

uv --project $project_dir run coding-assistant \
    --trace \
    --model "openrouter/google/gemini-3-flash-preview (medium)" \
    --readable-sandbox-directories /mnt/wsl ~/.ssh ~/.rustup ~/.config ~/.local ~/.cache \
    --writable-sandbox-directories "$project_dir" /tmp /dev/shm ~/.cache/coding_assistant ~/.cache/nvim ~/.local/state/nvim \
    --mcp-servers \
        $mcp_json_config \
    $argv
