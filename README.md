# Coding Assistant

Coding Assistant is a Python-based, agent-orchestrated CLI that helps you automate and streamline coding tasks. It can plan, launch sub-agents, use MCP tools (filesystem, web fetch/search, Context7, Tavily, etc.), run inside a sandbox, keep resumable history, and optionally emit OTLP traces for observability.

## Key Features

- Orchestrator agent that delegates to sub-agents and tools
- Resumable sessions and conversation summaries stored per-project
- MCP Server integration out of the box (built-in Coding Assistant MCP + filesystem, fetch, context7, tavily examples)
- Landlock-based filesystem sandbox with readable/writable allowlists
- Prompt-toolkit powered TUI with optional model chunks and reasoning display
- Shell/tool confirmation patterns to guard dangerous operations
- Configurable via CLI flags (models, planning mode, instructions, etc.)
- Optional OTLP tracing (exporter over HTTP)

## Requirements

- Python 3.12+
- uv (recommended) or pip for running/installing
- Optional: fish shell if you want to use the provided `run.fish`
- For the example MCP servers used in `run.fish`:
  - Node.js/npm for `npx`
  - Network access to fetch packages
- API keys as needed by your chosen model/tooling, e.g.:
  - `OPENAI_API_KEY` (or other LiteLLM-compatible provider keys)
  - `TAVILY_API_KEY` (for the Tavily MCP server)

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

The easiest way to run is with the provided script, which preconfigures the built-in Coding Assistant MCP server plus several external MCP servers and sensible defaults:

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
  --model "gpt-5" \
  --expert-model "gpt-5" \
  --mcp-servers \
    '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "{home_directory}"]}' \
    '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}'
```

Notes:
- Model names are routed via LiteLLM. Use any provider/model you have keys for; set the corresponding API key env vars.
- The `--mcp-servers` values are JSON strings; arguments support variable substitution for `{home_directory}` and `{working_directory}`.

## Usage Highlights

- `--task` The task for the orchestrator agent (required).
- `--resume`/`--resume-file` Resume from the latest/specific orchestrator history in `.coding_assistant/history/`.
- `--model`/`--expert-model` Select models for general/expert tasks.
- `--plan` Enable planning mode to build a stepwise plan before acting.
- `--instructions` Provide extra instructions that are composed with defaults.
- `--print-chunks`/`--print-reasoning` Control live stream and reasoning display in the TUI.
- `--print-instructions` Print the final instruction bundle that will be given to the agent and exit.
- `--user-feedback` Enable/disable user feedback prompts; `--ask-user` Allow the agent to ask the user questions.
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
- shell: `shell_execute` — execute shell commands (see behavior below)
- todo: `todo_add`, `todo_list_todos`, `todo_complete` — simple in-memory TODO list

When connected, tools are exposed to the agent as fully-qualified names:
- `mcp_coding_assistant_mcp_shell_execute`
- `mcp_coding_assistant_mcp_todo_add`
- `mcp_coding_assistant_mcp_todo_list_todos`
- `mcp_coding_assistant_mcp_todo_complete`

The `run.fish` script starts this server automatically. You can also add it manually as shown in the "Running without run.fish" example above.

### Other examples included in `run.fish`
- filesystem: `@modelcontextprotocol/server-filesystem`
- fetch: `mcp-server-fetch` (via `uvx`)
- context7: `@upstash/context7-mcp`
- tavily: `tavily-mcp` (needs `TAVILY_API_KEY`)

You can print discovered tools from running MCP servers:

```bash
uv run coding-assistant --print-mcp-tools ...
```

## Sandbox

When enabled (default), the assistant applies Landlock restrictions. By default it adds:
- Readable directories: your active virtual environment and any passed via `--readable-sandbox-directories`.
- Writable directories: the current working directory and any passed via `--writable-sandbox-directories`.

Use these flags to widen access if needed when working across multiple directories or mounts.

## History and Resume

- Conversation history and summaries are kept under `.coding_assistant/` in your project.
- Use `--resume` to continue from the most recent session, or `--resume-file` to select a specific file.
- The assistant automatically trims old history once it’s saved.

## Tracing (optional)

If `--trace-endpoint` is reachable (default `http://localhost:4318/v1/traces`), OTLP traces are exported using the HTTP exporter. If unreachable, tracing is disabled automatically.

## Shell command execution behavior

The MCP shell tool `shell_execute` (fully qualified in this project as `mcp_coding_assistant_mcp_shell_execute`):
- Merges stderr into stdout and returns plain text (no JSON envelope).
- Prefixes output with `Returncode: N` only when the command exits non-zero.
- Supports `truncate_at` to limit the combined output size and appends a note when truncation occurs.

## Development

- Run tests:

  ```bash
  just test
  ```

  or

  ```bash
  uv run pytest -n auto -m "not slow"
  ```

- Run linting/formatting/type-checking:

  ```bash
  just lint
  ```

- Handy `just` recipes: `hello-world`, `commit`, `review`, `fixlint` (see `justfile`).

## License

MIT
