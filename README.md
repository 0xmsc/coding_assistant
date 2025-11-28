# Coding Assistant

Coding Assistant is a Python-based, agent-orchestrated CLI that helps you automate and streamline coding tasks. It can plan, launch sub-agents, use MCP tools, run inside a sandbox, keep resumable history, and optionally emit OTLP traces for observability.

## Key Features

- Orchestrator agent that delegates to sub-agents and tools
- Resumable sessions and conversation summaries stored per-project
- Built-in MCP server with shell, Python, filesystem, and TODO tools
- Support for external MCP servers (filesystem, fetch, Context7, Tavily, etc.)
- Landlock-based filesystem sandbox with readable/writable allowlists
- Prompt-toolkit powered TUI with dense and regular output modes
- Shell/tool confirmation patterns to guard dangerous operations
- Chat mode enabled by default for interactive conversations
- Configurable via CLI flags (models, planning mode, instructions, etc.)
- Optional OTLP tracing (exporter over HTTP)

## Requirements

- Python 3.12+
- uv (recommended) or pip for running/installing
- Optional: fish shell if you want to use the provided `run.fish`
- Optional: External MCP servers if you want to extend functionality
  - Node.js/npm for `npx` (for NPM-based MCP servers)
  - Network access to fetch packages
- API keys as needed by your chosen model/tooling, e.g.:
  - `OPENAI_API_KEY` (or other LiteLLM-compatible provider keys)
  - Additional keys for external MCP servers (e.g., `TAVILY_API_KEY`)

## Installation

Using uv (recommended):

```bash
# In the repo root
uv sync  # or: uv pip install -e .
```

Using pip:

```bash
pip install -e .
```

## Quickstart

The easiest way to run is with the provided script, which preconfigures the built-in Coding Assistant MCP server and sensible defaults:

```bash
./run.fish --task "Say 'Hello World'"
```

A more realistic example:

```bash
./run.fish --task "Refactor all function names to snake_case."
```

Resume the last session:

```bash
./run.fish --task "Continue with the previous task." --resume
```

Show available options:

```bash
./run.fish --help
```

### Running without run.fish

You can invoke the CLI directly (e.g., using uv):

```bash
uv run coding-assistant \
  --task "Say 'Hello World'" \
  --model "openrouter/openai/gpt-4o-mini" \
  --expert-model "openrouter/anthropic/claude-3.5-sonnet"
```

Or with external MCP servers:

```bash
uv run coding-assistant \
  --task "Say 'Hello World'" \
  --model "openrouter/openai/gpt-4o-mini" \
  --mcp-servers \
    '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "{home_directory}"]}' \
    '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}'
```

Notes:
- Model names are routed via LiteLLM. Use any provider/model you have keys for; set the corresponding API key env vars.
- The `--mcp-servers` values are JSON strings; arguments support variable substitution for `{home_directory}` and `{working_directory}`.

## Usage Highlights

- `--task` The task for the orchestrator agent (required unless in chat mode).
- `--chat-mode` / `--no-chat-mode` Enable/disable open-ended chat mode (default: **enabled**).
- `--resume` / `--resume-file` Resume from the latest/specific orchestrator history in `.coding_assistant/history/`.
- `--model` / `--expert-model` Select models for general/expert tasks (default: `gpt-5` for both).
- `--instructions` Provide extra instructions that are composed with defaults.
- `--dense` / `--no-dense` Use dense output mode with compact formatting (default: **enabled**).
- `--print-chunks` / `--no-print-chunks` Control live model stream display (default: disabled, enabled in dense mode).
- `--print-reasoning` / `--no-print-reasoning` Display model reasoning (default: **enabled**).
- `--print-instructions` Print the final instruction bundle that will be given to the agent and exit.
- `--shell-confirmation-patterns` Ask for confirmation before running matching shell commands.
- `--tool-confirmation-patterns` Ask for confirmation before running matching tools.
- `--wait-for-debugger` Wait for a debugger (debugpy) to attach on port 1234.

Run `coding-assistant --help` to see all options.

## MCP Servers

Pass MCP servers with repeated `--mcp-servers` flags as JSON strings:

```json
{
  "name": "filesystem",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "{home_directory}"]
}
```

### Built-in: Coding Assistant MCP

This repository includes a built-in MCP server (package `packages/coding_assistant_mcp`) that provides:

- **shell**: `shell_execute` — Execute shell commands with timeout and output truncation
- **python**: `python_execute` — Execute Python code with timeout and output truncation
- **filesystem**: `filesystem_write_file`, `filesystem_edit_file` — Write new files or apply targeted edits
- **todo**: `todo_add`, `todo_list_todos`, `todo_complete` — Simple in-memory TODO list management

When connected, tools are exposed to the agent as fully-qualified names:
- `mcp_coding_assistant_mcp_shell_execute`
- `mcp_coding_assistant_mcp_python_execute`
- `mcp_coding_assistant_mcp_filesystem_write_file`
- `mcp_coding_assistant_mcp_filesystem_edit_file`
- `mcp_coding_assistant_mcp_todo_add`
- `mcp_coding_assistant_mcp_todo_list_todos`
- `mcp_coding_assistant_mcp_todo_complete`

The `run.fish` script starts this server automatically.

### External MCP Servers (Optional)

You can add external MCP servers to extend functionality. Examples include:

- **filesystem**: `@modelcontextprotocol/server-filesystem` (NPM) — Additional filesystem operations
- **fetch**: `mcp-server-fetch` (uvx) — Web fetching capabilities
- **context7**: `@upstash/context7-mcp` (NPM) — Context management
- **tavily**: `tavily-mcp` (needs `TAVILY_API_KEY`) — Web search

To use these, add them via `--mcp-servers` flags as shown in the examples above.

You can print all discovered tools from running MCP servers:

```bash
uv run coding-assistant --print-mcp-tools --mcp-servers '...'
```

## Sandbox

When enabled (default), the assistant applies Landlock restrictions. By default it adds:
- Readable directories: your active virtual environment and any passed via `--readable-sandbox-directories`.
- Writable directories: the current working directory and any passed via `--writable-sandbox-directories`.

Use these flags to widen access if needed when working across multiple directories or mounts.

Example from `run.fish`:
```bash
--readable-sandbox-directories /mnt/wsl ~/.ssh ~/.rustup \
--writable-sandbox-directories "$project_dir" /tmp /dev/shm ~/.cache/coding_assistant
```

## History and Resume

- Conversation history and summaries are kept under `.coding_assistant/` in your project.
- Use `--resume` to continue from the most recent session, or `--resume-file` to select a specific file.
- The assistant automatically trims old history once it's saved.

## Tracing (optional)

If `--trace-endpoint` is reachable (default `http://localhost:4318/v1/traces`), OTLP traces are exported using the HTTP exporter. If unreachable, tracing is disabled automatically.

## Shell and Python Execution

The built-in MCP tools `shell_execute` and `python_execute`:
- Support multi-line scripts
- Merge stderr into stdout and return plain text (no JSON envelope)
- Prefix output with `Returncode: N` only when the command/code exits non-zero
- Support `truncate_at` parameter to limit combined output size and append a note when truncation occurs
- Support `timeout` parameter (default: 30 seconds)
- Interactive commands (e.g., `git rebase -i`) are not supported and will block

## Development

- Run tests:

  ```bash
  just test
  ```

  or

  ```bash
  uv run pytest -n auto -m "not slow"
  uv run --directory packages/coding_assistant_mcp pytest -n auto
  ```

- Run linting/formatting/type-checking:

  ```bash
  just lint
  ```

- Handy `just` recipes: `hello-world`, `commit`, `review`, `fixlint` (see `justfile`).

## License

MIT
