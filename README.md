# Coding Assistant

Coding Assistant is a Python-based CLI for coding workflows. It combines a boundary-based agent loop, built-in local tools, and optional external MCP servers.

## Key Features

- Built-in local shell, Python, filesystem, TODO, and background-task tools.
- Support for external MCP servers over stdio or SSE.
- Prompt-toolkit powered interactive CLI.
- Bundled default instructions and skills, plus optional extra skill directories.
- No built-in sandboxing; run it inside an external sandbox if you need isolation.

## Requirements

- Python 3.12+.
- `uv` is recommended for installation and running.
- API keys for your chosen OpenAI-compatible provider, for example `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
- Optional dependencies for external MCP servers, such as Node.js/npm for NPM-based servers.

## Installation

Using `uv`:

```bash
uv tool install coding-assistant-cli
```

For local development:

```bash
uv sync
```

## Quickstart

Start an interactive session:

```bash
coding-assistant --model "openai/gpt-5-mini"
```

Show available options:

```bash
coding-assistant --help
```

The supported public interface is the CLI. Internal Python modules may change without notice.

## CLI Highlights

- `--model` selects the model to use. Required.
- `--instructions` appends custom instructions.
- `--mcp-servers` configures external MCP servers as JSON strings.
- `--trace` writes model request and response traces.
- `--wait-for-debugger` waits for a debugger on port `1234`.
- `--skills-directories` loads additional skill directories.

The CLI is interactive.
At startup it also prints a localhost websocket endpoint; remote clients can prompt the same live session when it is idle, stream updates, and cancel the current run.

The interactive CLI also supports:

- `/exit`
- `/help`
- `/compact`
- `/image <path-or-url>`

## MCP Servers

Pass MCP servers with repeated `--mcp-servers` flags as JSON strings. Both stdio-based servers and remote SSE servers are supported.

Example stdio server:

```json
{
  "name": "filesystem",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "{home_directory}"]
}
```

Example SSE server:

```json
{
  "name": "remote-mcp",
  "url": "http://localhost:8000/sse"
}
```

Arguments support variable substitution for `{home_directory}` and `{working_directory}`.

Example:

```bash
coding-assistant \
  --model "openrouter/openai/gpt-4o-mini" \
  --mcp-servers \
    '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "{home_directory}"]}' \
    '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}'
```

## Built-In Local Tools

The default CLI includes these local tools:

- `shell_execute` for running shell commands.
- `python_execute` for running Python snippets.
- `filesystem_write_file` and `filesystem_edit_file` for targeted file changes.
- `todo_add`, `todo_list_todos`, and `todo_complete` for TODO tracking.
- `tasks_list_tasks`, `tasks_get_status`, `tasks_get_output`, `tasks_kill_task`, and `tasks_remove_task` for background task management.

The core runtime also adds internal helper tools such as `compact_conversation` and `redirect_tool_call`.

## Skills

Coding Assistant currently ships one bundled example skill:

- `example`

It exists to demonstrate the packaged skill mechanism and expected `SKILL.md` layout. Replace or remove it once the project has real builtin skills.

You can add more skills with `--skills-directories`. Each additional skill directory should contain child directories with a `SKILL.md` file:

```text
skills-root/
├── my-skill/
│   ├── SKILL.md
│   ├── references/
│   └── scripts/
└── another-skill/
    └── SKILL.md
```

The default CLI exposes these skill tools:

- `skills_list_resources`
- `skills_read`

Those tools expose only the files inside the bundled and configured skill directories.

Skill names must be unique across bundled and user-provided skills. The CLI fails fast on collisions and reports the conflicting directories.

## External Sandboxing

Built-in sandboxing has been removed from this project. If you want filesystem isolation, run the assistant inside an external sandbox such as `bubblewrap`.

See [docs/sandboxing.md](docs/sandboxing.md) for a minimal `bubblewrap` example.

## Shell And Python Tool Behavior

The built-in `shell_execute` and `python_execute` tools:

- support multi-line scripts,
- merge stderr into stdout,
- prefix output with an exit-code header only for non-zero exits,
- support `truncate_at` to limit output size,
- support `timeout` with a default of 30 seconds,
- can hand long-running work off to the background task manager.

Interactive terminal programs such as `git rebase -i` are not supported.

## Development

Run tests:

```bash
just test
```

Run linting, formatting, and type checks:

```bash
just lint
```

There is also an integration test target:

```bash
just test-integration
```

## License

MIT
