import asyncio
import logging
import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction
from pathlib import Path

import debugpy

from coding_assistant.adapters.cli import run_cli
from coding_assistant.paths import get_log_file
from coding_assistant.trace import enable_tracing, get_default_trace_dir

logger = logging.getLogger("coding_assistant")
logger.setLevel(logging.INFO)


def setup_logging() -> None:
    """Setup logging to file only."""
    log_file = get_log_file()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Root logger for capture all logs in file
    root_logger = logging.getLogger()

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    # Set 'coding_assistant' logger to INFO
    logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the coding assistant executable."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description="Coding Assistant CLI")
    parser.add_argument(
        "--task", type=str, help="Task for the orchestrator agent. If provided, the agent runs in autonomous mode."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest orchestrator history file in .coding_assistant/history/.",
    )
    parser.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        help="Resume from a specific orchestrator history file.",
    )
    parser.add_argument("--print-mcp-tools", action="store_true", help="Print all available tools from MCP servers.")
    parser.add_argument(
        "--print-instructions",
        action="store_true",
        help="Print the instructions that will be given to the orchestrator agent and exit.",
    )
    parser.add_argument("--model", type=str, required=True, help="Model to use for the orchestrator agent.")
    parser.add_argument("--expert-model", type=str, default=None, help="Expert model to use.")
    parser.add_argument(
        "--instructions",
        nargs="*",
        default=[],
        help="Custom instructions for the agent.",
    )
    parser.add_argument(
        "--readable-sandbox-directories",
        nargs="*",
        default=[],
        help="Additional directories to include in the sandbox.",
    )
    parser.add_argument(
        "--writable-sandbox-directories",
        nargs="*",
        default=[],
        help="Additional directories to include in the sandbox.",
    )
    parser.add_argument(
        "--sandbox",
        action=BooleanOptionalAction,
        default=True,
        help="Enable sandboxing.",
    )
    parser.add_argument(
        "--mcp-servers",
        nargs="*",
        default=[],
        help='MCP server configurations as JSON strings. Format: \'{"name": "server_name", "command": "command", "args": ["arg1", "arg2"], "env": ["ENV_VAR1", "ENV_VAR2"]}\' or \'{"name": "server_name", "url": "http://localhost:8000/sse"}\'',
    )
    parser.add_argument(
        "--mcp-env",
        nargs="*",
        default=[],
        help="Environment variables to pass to the default MCP server.",
    )
    parser.add_argument(
        "--tool-confirmation-patterns",
        nargs="*",
        default=[],
        help="Ask for confirmation before executing a tool that matches any of the given patterns.",
    )
    parser.add_argument(
        "--shell-confirmation-patterns",
        nargs="*",
        default=[],
        help="Regex patterns that require confirmation before executing shell commands",
    )
    parser.add_argument(
        "--wait-for-debugger",
        action=BooleanOptionalAction,
        default=False,
        help="Wait for a debugger to attach.",
    )
    parser.add_argument(
        "--trace",
        action=BooleanOptionalAction,
        default=False,
        help="Enable tracing of model requests and responses to a session folder in $XDG_STATE_HOME/coding-assistant/traces.",
    )
    parser.add_argument(
        "--ask-user",
        action=BooleanOptionalAction,
        default=True,
        help="Enable/disable asking the user for input in runs started with --task.",
    )

    parser.add_argument(
        "--skills-directories",
        nargs="*",
        default=[],
        help="Paths to directories containing Agent Skills (with SKILL.md files).",
    )

    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    """Run the CLI and translate Ctrl-C into a clean shutdown."""
    logger.info(f"Starting Coding Assistant with arguments {args}")
    try:
        await run_cli(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


def main() -> None:
    """Entrypoint for the installed `coding_assistant` command."""
    args = parse_args()
    setup_logging()

    if args.trace:
        enable_tracing(get_default_trace_dir())

    if args.wait_for_debugger:
        logger.info("Waiting for debugger to attach on port 1234")
        debugpy.listen(1234)
        debugpy.wait_for_client()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
