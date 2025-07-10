# Coding Assistant

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/pypi/v/coding-assistant)](https://pypi.org/project/coding-assistant)
[![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/coding-assistant/ci.yml?branch=main)](https://github.com/yourusername/coding-assistant/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Coding Assistant is an experimental, Python-based multi-agent system designed to automate and streamline coding workflows. It orchestrates specialized agents—Orchestrator, Researcher, and Developer—to collaboratively tackle complex tasks, from research and planning through code generation and review. Recent enhancements have further improved memory management, error handling, parameter validation, and overall code hygiene.

## Features

- **Modular & Extensible Multi-Agent Architecture**: Specializes tasks across Orchestrator, Researcher, and Developer agents for scalable workflows.
- **Versatile LLM Integration**: Seamless interaction with multiple LLM providers via `litellm`.
- **Automated Code Generation & Refactoring**: Rapidly generate, optimize, and refactor code based on context.
- **Deep Research & Knowledge Retrieval**: Built-in Researcher agent and `tavily-mcp` integration for real-time web search and content extraction.
- **Filesystem & Git Operations**: Robust file and version control manipulations using `mcp[cli]`.
- **Interactive CLI with Custom Flags**: Supports `--research` and `--expert` modes for tailored execution paths.
- **Configurable Environment Management**: Simple setup via environment variables to manage models, API keys, and behavior.
- **Telemetry & Monitoring**: Integrated OpenTelemetry for tracing and performance insights.
- **Comprehensive Error Handling & Logging**: Explicit exception handling and rich-formatted logs for clarity.
- **Memory Management**: Bounded conversation history trimming to optimize resource usage.
- **Pluggable Tooling & Extensions**: Easily extend functionality with custom tools and modules.

## Technologies & Dependencies

- **Python**: 3.12+
- **Node.js/UVX**: 16+ (Node.js runtime & UVX for fast installs)
- **requests**: HTTP requests library
- **mcp[cli]**: Filesystem and tool operations
- **tavily-mcp**: Web search and extraction
- **litellm**: Unified LLM interface
- **OpenTelemetry**: Tracing and telemetry instrumentation
- **rich**: ANSI and Markdown-rich console output

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/coding-assistant.git
cd coding-assistant

# (Optional) Python virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install .

# Install Node.js (v16+)
# Recommended: use nvm
nvm install 16
nvm use 16

# Install UVX globally (optional for faster package management)
npm install -g uvx

# Install Node.js dependencies via UVX or NPX
uvx install        # or
npx uvx install
```

## Configuration

Create a `.env` or `.envrc` file in the project root:

```bash
# OpenAI/Litellm Models
CODING_ASSISTANT_MODEL="o4-mini"
CODING_ASSISTANT_EXPERT_MODEL="o3"

# Tavily API Key for web search/extraction
TAVILY_API_KEY="your_tavily_api_key_here"
```

Default values if not specified:

- `CODING_ASSISTANT_MODEL`: `o4-mini`
- `CODING_ASSISTANT_EXPERT_MODEL`: `o3`

## Usage

```bash
# Basic task execution
coding-assistant --task "Describe your task here"

# Include research insights
coding-assistant --task "Describe your task here" --research

# Use expert model for advanced responses
coding-assistant --task "Describe your task here" --expert

# Combine flags
coding-assistant --task "Describe your task here" --research --expert
```

Or with Python module:

```bash
python -m coding_assistant.main --task "Your task" --research --expert
```

## Project Structure

```
.
├── src/coding_assistant/
│   ├── llm/            # LLM integration modules
│   ├── agents/         # Orchestrator, Researcher, Developer
│   ├── tools.py        # MCP-based utilities
│   ├── config.py       # Configuration settings
│   └── main.py         # CLI entry point
├── pyproject.toml      # Project metadata and dependencies
├── README.md           # Project documentation
└── .envrc              # direnv configuration (if used)
```

## Contributing

We welcome contributions! To contribute:

1. Fork this repository.
2. Create a feature branch:
   ```bash
git checkout -b feature/your-feature-name
```  
3. Make your changes and commit them.
4. Ensure your code passes any linters or style checks you use.
5. Push your branch:
   ```bash
git push origin feature/your-feature-name
```  
6. Open a Pull Request describing your changes.

Please follow the project's coding standards and update any relevant documentation.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
