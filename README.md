# Coding Assistant

## Introduction

Coding Assistant is an advanced Python-based project designed to streamline and automate various coding tasks. By leveraging different types of intelligent agents and powerful tools, Coding Assistant aims to enhance developer productivity and code quality.

## Project Overview

Coding Assistant utilizes a system of specialized agents (orchestrator, planner, and developer) working in tandem to automate coding processes. These agents, combined with a set of powerful tools, can assist in various tasks such as code generation, refactoring, and documentation.

## Features

- Multi-agent system (orchestrator, planner, developer) for intelligent task handling
- Automated code generation and refactoring
- Integrated file searching using ripgrep
- Modular and extensible architecture
- Strong typing with mypy for improved code quality
- Comprehensive test suite using pytest
- Custom logging for enhanced debugging and monitoring

## Installation

1. Ensure you have Python 3.x installed on your system.
2. Clone the repository:
   ```
   git clone https://github.com/yourusername/coding_assistant.git
   cd coding_assistant
   ```
3. Set up a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Set up direnv for environment management (optional):
   ```
   direnv allow
   ```

## Configuration

The project uses environment variables to configure the LLM backend and models:

```bash
CODING_ASSISTANT_BACKEND=OPENAI # or OPENROUTER
CODING_ASSISTANT_MODEL=gpt-4o
CODING_ASSISTANT_REASONING_MODEL=o1
```

Currently, the project only supports OpenAI and OpenRouter as LLM backends.

## Usage Guide

1. Start the Coding Assistant:
   ```
   python -m coding_assistant
   ```
2. Interact with the assistant through the command line interface.
3. Example commands or operations:
   - [Provide examples of how to use different features]

## Project Structure

```
coding_assistant/
├── agents/
│   ├── orchestrator.py
│   ├── planner.py
│   ├── developer.py
│   ├── expert.py
│   ├── agents.py
│   ├── researcher.py
│   ├── prompt.py
└── tools/
    ├── user.py
    ├── notebook.py
    ├── git_tools.py
    └── file.py
├── config/
│   └── config.py
├── tests/
├── __main__.py
├── logging.py
└── __init__.py
```

- `agents/`: Contains the different agent components (orchestrator, planner, developer, etc.)
- `tools/`: Includes utility tools and additional modules needed for various tasks
- `config/`: Manages configuration settings for the project
- `tests/`: Directory containing all the automated tests
- Other files support different functionalities and promote modularity

## Technologies Used

- Python: The primary programming language for the project
- langchain: A library for agent-based applications
- rich: A library for rich text and beautiful formatting in the terminal
- pytest: Testing framework for writing and running automated tests
- Git: Version control system for tracking changes and collaborating
- direnv: Tool for managing project-specific environment variables

## Contributing

We welcome contributions to the Coding Assistant project! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests to ensure everything is working (`pytest`)
5. Commit your changes (`git commit -am 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Create a new Pull Request

Please make sure to update tests as appropriate and adhere to the project's coding standards.

For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements

- Thanks to all contributors who have helped shape Coding Assistant
- Special thanks to the developers of ripgrep, langchain, rich, and pytest for their excellent tools
