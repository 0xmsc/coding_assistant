# Coding Assistant

## 🤖 Project Title: Coding Assistant

---

## Introduction

**Coding Assistant** is an advanced Python-based tool that leverages agent-based automation to streamline, support, and automate various coding tasks. It deploys multi-agent orchestration, research, automated development, feedback validation, and Model Context Protocol (MCP) tool integrations to tackle complex software engineering workflows efficiently and reliably. With agent roles specialized for distinct phases of software work, Coding Assistant optimizes developer productivity in real-world codebases.

---

## Key Features

- **Agent Orchestration**: The Orchestrator agent coordinates tasks by delegating work to specialized sub-agents as required.
- **Generic Agent System**: Flexible agent tool that can handle research, development, and analysis tasks based on provided instructions.
- **Feedback Agent**: Automatically reviews and validates agent results, ensuring they meet user requirements.
- **Flexible CLI**: Launch, control, and interact with agents/tasks from the command line.
- **MCP Server Integration**: Native support for MCP server toolchains (filesystem, fetch/web search, git, Tavily, etc). 
- **Configuration Management**: User configuration system with JSON-based settings and conversation history.
- **Tracing with OpenTelemetry**: Optional tracing and observability via OpenTelemetry compatible endpoints.
- **Sandbox Security**: Landlock-based filesystem sandbox for secure task execution with configurable directories.

---

## Software Architecture

The Coding Assistant is structured as a multi-agent system coordinated via an Orchestrator. The user provides configuration, CLI inputs, or explicit tasks, which flow into the Orchestrator. The Orchestrator directs requests to specialized agent roles (Generic Agent, Feedback Agent), each leveraging MCP server tools for file access, shell commands, git, web search/fetch, and more. All task results are stored in conversation history and returned to the user through the CLI.

```mermaid
flowchart TD
    subgraph User_Group
        User_Node[User]
    end
    subgraph CLI_Group
        CLI_Node[CLI / Terminal]
    end
    subgraph Config_Group
        CONFIG_Node[User Config]
        CACHE_Node[Conversation Cache]
    end

    subgraph Orchestrator_Group
        ORCH_Node[Orchestrator Agent]
    end

    subgraph Agents_Group
        GA_Node[Generic Agent]
        FA_Node[Feedback Agent]
    end

    subgraph MCP_Servers_Group [MCP Servers / Tooling]
        FS_Node[Filesystem]
        SH_Node[Shell]
        WEB_Node[Fetch / Tavily]
    end

    User_Node -->|"Inputs via"| CLI_Node
    CLI_Node -->|"Config / Task / Command"| ORCH_Node
    ORCH_Node <-->|"Loads/Saves"| CONFIG_Node
    ORCH_Node <-->|"History"| CACHE_Node

    ORCH_Node -->|"Delegates"| GA_Node
    ORCH_Node -->|"Sends for review"| FA_Node

    GA_Node -- "Uses tools" --> FS_Node
    GA_Node -- "Uses tools" --> SH_Node
    GA_Node -- "Uses" --> WEB_Node

    FA_Node -- "Uses tools" --> FS_Node
    FA_Node -- "Uses tools" --> SH_Node

    FS_Node -.->|"Provides Services"| MCP_Servers_Group
    SH_Node -.->|"Provides Services"| MCP_Servers_Group
    WEB_Node -.->|"Provides Services"| MCP_Servers_Group

    ORCH_Node -->|"Results / Outputs"| CLI_Node
```

**Explanation:**
- **User interacts with the Coding Assistant through the CLI.**
- **CLI collects user inputs and passes them to the Orchestrator Agent.**
- **Configuration and conversation history are managed automatically.**
- **The Orchestrator delegates tasks to specialist agents (Generic Agent, Feedback Agent).**
- **Agents interact with external services (MCP Servers: Filesystem, Shell, Fetch/Tavily) as needed for their task.**
- **Results flow back through the Orchestrator to the CLI, and are shown to the user.**

---

## Installation Guide

### System Requirements
- **Python**: 3.12+ (see `.python-version`)
- **Node.js**: For running MCP servers (filesystem, Tavily).
- **Additional tools**: `uv`, `npx`, and optionally `git`, `npm`, `pytest`.

### 1. Clone the repository & enter the directory
```bash
git clone <REPO_URL>
cd coding_assistant
```

### 2. Set up a virtual environment
```bash
uv venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv pip install -e .
```

### 4. Set environment variables
- Copy or edit `.envrc` to provide API keys:
    - `OPENAI_API_KEY` (for OpenAI model access)
    - `TAVILY_API_KEY` (for web research)
    - `DEEPSEEK_API_KEY` (optional, for DeepSeek models)
    - `GEMINI_API_KEY` (optional, for Google Gemini models)
    - `ANTHROPIC_API_KEY` (optional, for Anthropic Claude models)

If using direnv:
```bash
direnv allow
```

---

## Quickstart

### Launching the Orchestrator via CLI

#### Run an Orchestrator Task:
```bash
uv run coding-assistant --task "Refactor all function names to snake_case."
```

#### Print all available MCP tools:
```bash
uv run coding-assistant --print-mcp-tools
```

#### Disable feedback agent:
```bash
uv run coding-assistant --task "Update documentation" --disable-feedback-agent
```

#### Disable sandboxing (for development):
```bash
uv run coding-assistant --task "Debug issue" --disable-sandbox
```

---

## Configuration

### User Configuration File

The system automatically creates a configuration file at `~/.config/coding_assistant/config.json` with the following structure:

```json
{
  "model": "gpt-4o",
  "expert_model": "o3",
  "disable_feedback_agent": false,
  "disable_user_feedback": false,
  "instructions": null,
  "sandbox_directories": [
    "/tmp"
  ]
}
```

### Configuration Options

- **model**: Default LLM model for general agents
- **expert_model**: LLM model for complex tasks requiring expert-level reasoning
- **disable_feedback_agent**: Skip automatic feedback validation
- **disable_user_feedback**: Skip user feedback prompts
- **instructions**: Custom instructions to append to all agents
- **sandbox_directories**: Additional directories to allow in sandbox mode

### Environment Variables

Required/Optional environment variables:
- `OPENAI_API_KEY` (required for OpenAI models)
- `TAVILY_API_KEY` (required for web research functionality)
- `DEEPSEEK_API_KEY` (optional, for DeepSeek models)
- `GEMINI_API_KEY` (optional, for Google Gemini models)
- `ANTHROPIC_API_KEY` (optional, for Anthropic Claude models)

Place these in `.envrc` or export manually in your shell.

---

## Project Structure

```
coding_assistant/
├── src/
│   └── coding_assistant/
│       ├── __init__.py
│       ├── cache.py              # Conversation history management
│       ├── config.py             # User configuration system
│       ├── instructions.py       # Agent instruction handling
│       ├── llm/
│       │   └── model.py          # LLM model interface
│       ├── main.py               # CLI entry point
│       ├── sandbox.py            # Sandbox implementation (Landlock)
│       ├── tools.py              # MCP server & tool integration
│       ├── agents/
│       │   ├── agents.py         # All core agent/tool classes
│       │   ├── logic.py          # Agent orchestration logic
│       │   └── tests/            # Agent-specific tests
│       └── tests/                # Sandbox & integration tests
├── justfile                      # Development commands
├── pyproject.toml                # Dependencies and build configuration
├── README.md                     # This document
├── .envrc                        # Environment variables template
├── .python-version               # Python version specification
└── uv.lock                       # Dependency lock file
```

### Key Files

- **pyproject.toml**: Declares all dependencies and CLI entry point
- **justfile**: Development commands (`just test` for running tests)
- **.envrc**: Template for environment variables
- **cache.py**: Manages conversation history storage
- **config.py**: User configuration and settings management

---

## Agent Types

### Orchestrator Agent
The main coordination agent that:
- Receives tasks from the CLI
- Delegates work to sub-agents
- Manages conversation history
- Coordinates feedback loops

### Generic Agent (AgentTool)
A flexible agent that can handle:
- Research tasks and information gathering
- Code development and implementation
- Analysis and documentation tasks
- Any task with clear instructions and expected output

### Feedback Agent
Specialized agent for:
- Reviewing and validating agent outputs
- Ensuring results meet user requirements
- Providing quality control and improvement suggestions

### Utility Tools
- **AskClientTool**: Interactive user input
- **ExecuteShellCommandTool**: Shell command execution
- **FinishTaskTool**: Task completion signaling

---

## MCP Server Integration

The system automatically initializes MCP servers based on available tools and API keys:

### Filesystem Server
- **Command**: `npx @modelcontextprotocol/server-filesystem <working_directory>`
- **Purpose**: File system operations and management

### Fetch Server
- **Command**: `uvx mcp-server-fetch`
- **Purpose**: Web content fetching and HTTP requests

### Tavily Server (Optional)
- **Command**: `npx tavily-mcp@0.2.1`
- **Purpose**: Advanced web search and research
- **Requirement**: `TAVILY_API_KEY` environment variable

---

## Development

### Testing
```bash
# Run all tests
just test

# Or directly with pytest
uv run pytest -n auto
```

### Code Style
- **Formatter**: Black with 120-character line length (see pyproject.toml)
- **Type hints**: Used throughout for clarity
- **Structure**: Dataclasses and structured configuration

### Test Locations
- `src/coding_assistant/agents/tests/` - Agent-specific tests
- `src/coding_assistant/tests/` - Core system tests

---

## Advanced Features

### Conversation History
- Automatically stored in `~/.config/coding_assistant/cache/`
- Maintains context across sessions
- Used for agent learning and improvement

### Sandboxing
- Landlock-based filesystem restrictions
- Configurable allowed directories
- Can be disabled for development with `--disable-sandbox`

### Tracing and Observability
- OpenTelemetry integration
- Automatic detection of trace endpoint at `http://localhost:4318/v1/traces`
- Comprehensive span tracking for debugging

---

## Troubleshooting

### Common Issues

**Installation Problems:**
- Ensure Python 3.12+ is installed
- Use `uv pip install -e .` instead of requirements.txt
- Verify Node.js is available for MCP servers

**API Key Errors:**
- Check `.envrc` file exists and is loaded
- Verify API keys are valid and not expired
- Ensure direnv is installed and allowed if using `.envrc`

**Sandbox Errors:**
- Add required directories to `sandbox_directories` in config
- Use `--disable-sandbox` for debugging
- Check file permissions in working directory

**MCP Server Issues:**
- Verify Node.js installation
- Check `npx` is available in PATH
- Ensure network connectivity for package downloads

**Configuration Problems:**
- Check `~/.config/coding_assistant/config.json` exists
- Verify JSON syntax is valid
- Reset config by deleting file (will regenerate defaults)

### Getting Help
- Check agent logs for detailed error messages
- Review conversation history in cache directory
- Open issues on GitHub with reproduction steps

---

## License and Credits

- **License:** MIT
- **Contributions:** PRs and issues welcome. Please add tests for new functionality.
- **Architecture:** Built on MCP (Model Context Protocol) for tool integration
- **Security:** Landlock sandbox for safe code execution