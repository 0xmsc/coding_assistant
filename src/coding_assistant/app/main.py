import asyncio
import logging
import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction

import debugpy

from coding_assistant.app.cli import (
    CliController,
    CliOutput,
    _build_initial_system_message,
    _create_prompt_session,
    _prompt_with_session,
    build_default_agent_config,
    create_default_agent,
)
from coding_assistant.app.session_control import SessionController, SessionOutput
from coding_assistant.app.session_runtime import SessionRuntime
from coding_assistant.integrations.mcp_client import print_mcp_tools
from coding_assistant.remote.server import RemoteController
from coding_assistant.infra.paths import get_log_file
from coding_assistant.infra.trace import enable_tracing, get_default_trace_dir

logger = logging.getLogger("coding_assistant")
logger.setLevel(logging.INFO)


async def _main(args: argparse.Namespace) -> None:
    """Run the CLI and translate Ctrl-C into a clean shutdown."""
    logger.info(f"Starting Coding Assistant with arguments {args}")
    try:
        await run_session_runtime(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


async def run_session_runtime(args: argparse.Namespace) -> None:
    """Build the local session runtime plus its controllers and outputs, then run it."""

    config = build_default_agent_config(args)
    prompt_session = _create_prompt_session()

    async def prompt_user(words: list[str] | None = None) -> str:
        return await _prompt_with_session(prompt_session, words=words)

    async with create_default_agent(config=config) as bundle:
        if args.print_mcp_tools:
            await print_mcp_tools(bundle.mcp_servers)
            return

        system_message = _build_initial_system_message(instructions=bundle.instructions)
        cli_controller = CliController(prompt_user=prompt_user)
        remote_controller = RemoteController(
            cwd=config.working_directory,
            set_local_worker_endpoint=bundle.set_local_worker_endpoint,
        )
        runtime = SessionRuntime(
            history=[system_message],
            model=args.model,
            tools=bundle.tools,
            default_controller=cli_controller,
        )
        controllers: list[SessionController] = [
            cli_controller,
            remote_controller,
        ]
        outputs: list[SessionOutput] = [CliOutput(system_message=system_message)]

        try:
            await runtime.run(controllers=controllers, outputs=outputs)
        finally:
            await bundle.close_tools()


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
    parser.add_argument("--print-mcp-tools", action="store_true", help="Print all available tools from MCP servers.")
    parser.add_argument("--model", type=str, required=True, help="Model to use for the orchestrator agent.")
    parser.add_argument(
        "--instructions",
        nargs="*",
        default=[],
        help="Custom instructions for the agent.",
    )
    parser.add_argument(
        "--mcp-servers",
        nargs="*",
        default=[],
        help='MCP server configurations as JSON strings. Format: \'{"name": "server_name", "command": "command", "args": ["arg1", "arg2"], "env": ["ENV_VAR1", "ENV_VAR2"]}\' or \'{"name": "server_name", "url": "http://localhost:8000/sse"}\'',
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
        "--skills-directories",
        nargs="*",
        default=[],
        help="Paths to additional directories containing Agent Skills (with SKILL.md files).",
    )

    return parser.parse_args()


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
