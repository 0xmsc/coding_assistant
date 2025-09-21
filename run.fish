#!/usr/bin/env fish

# set fish_trace 1

set project_dir (dirname (status filename))
set mcp_project_dir $project_dir/packages/coding_assistant_mcp
set mcp_json_config (printf '{"name": "coding_assistant_mcp", "command": "uv", "args": ["--project", "%s", "run", "coding-assistant-mcp"], "env": []}' "$mcp_project_dir")

set user_instructions (string collect "
- Prefer the tools from `coding_assistant_mcp` if other tools provide the same functionality.
- Use MCP shell tool `shell_execute` to execute shell commands.
    - Examples: eza/ls, git, fd/find, rg/grep, gh, pwd.
- Always manage a TODO list while working on your task.
    - Use the `todo_*` tools for managing the list.
")

uv --project $project_dir run coding-assistant \
    --model "openai/gpt-5 (medium)" \
    --expert-model "openai/gpt-5 (high)" \
    --readable-sandbox-directories /mnt/wsl ~/.ssh \
    --writable-sandbox-directories $project_dir /tmp /dev/shm \
    --instructions "$user_instructions" \
    --mcp-servers \
        $mcp_json_config \
        '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "{home_directory}"]}' \
        '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}' \
        '{"name": "context7", "command": "npx", "args": ["-y", "@upstash/context7-mcp"], "env": []}' \
        '{"name": "tavily", "command": "npx", "args": ["-y", "tavily-mcp"], "env": ["TAVILY_API_KEY"]}' \
    $argv
