# Coding Assistant

> An advanced Python-based project designed to streamline and automate various coding tasks using an agent-based architecture.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Orchestrator agent** to coordinate research, development, and feedback agents.
- **Research agent** for information gathering.
- **Developer agent** for implementing code based on a detailed plan.
- **Feedback agent** for evaluating agent outputs to ensure client satisfaction.
- Integration with Model Context Protocol (MCP) servers: filesystem, git, fetch, memory, shell, and optional Tavily.
- OpenTelemetry tracing support for observability.
- Rich prompts and interactive CLI for seamless workflows.

## Architecture

The project follows an agent-based architecture:

- **Orchestrator Tool**: Coordinates tasks among sub-agents.
- **Research Tool**: Launches a Researcher agent for gathering information.
- **Develop Tool**: Launches a Developer agent to write code per implementation plans.
- **Feedback Tool**: Launches a Feedback agent to review outputs.
- **Tools Module**: Manages connections to MCP servers for filesystem, git, shell, etc.
- **Configuration Module**: Centralizes configuration for model selection and working directory.
- **CLI Entry Point**: `coding-assistant` script initializes tracing, parses CLI args, and runs agents.

## Requirements

- Python 3.12 or higher
- OpenAI API Key
- Tavily API Key (for Tavily integration)
- Node.js and npm (for MCP servers via npx)
- `just` (optional, for task shortcuts)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/coding-assistant.git
   cd coding-assistant
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. Install development dependencies:
   ```bash
   pip install pytest pytest-asyncio pytest-xdist black
   ```

4. Optionally, install `just` for convenient task running:
   ```bash
   # On macOS with Homebrew:
   brew install just

   # On Linux:
   sudo apt-get install just
   ```

## Configuration

Configure environment variables in `.envrc` or your shell profile:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
export CODING_ASSISTANT_MODEL="o4-mini"       # default: o4-mini
export CODING_ASSISTANT_EXPERT_MODEL="o3"     # default: o3
```

## Usage

Use the `coding-assistant` CLI to run orchestrator and agents:

- Launch an orchestrator task:
  ```bash
  coding-assistant --task "Add a detailed README.md to the project"
  ```

- Ask a research question:
  ```bash
  coding-assistant --research "How to write unit tests for async code?"
  ```

- Invoke an expert agent:
  ```bash
  coding-assistant --expert "Generate SQLAlchemy models for my data schema"
  ```

## Development

- Code style enforced with [Black](https://black.readthedocs.io/) (line length: 120).
- Configuration stored in `pyproject.toml`.
- Source code under `src/coding_assistant/`.
- Build system: Hatchling (`[build-system]` in `pyproject.toml`).

## Testing

Run test suite with:
```bash
just test
# OR
pytest -n auto
```

## Contributing

Contributions welcome! Please open issues or pull requests. Follow standard GitHub flow:

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit changes and push: `git commit -am "Add some feature"`
4. Submit a pull request

## License

This project is currently unlicensed. Add a `LICENSE` file to choose a license (e.g., MIT, Apache 2.0).
