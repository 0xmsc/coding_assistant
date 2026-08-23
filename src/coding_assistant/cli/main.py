import argparse
import asyncio
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction

import debugpy

from coding_assistant.cli.ui import run_cli
from coding_assistant.infra.logging import setup_logging
from coding_assistant.infra.trace import enable_tracing, get_default_trace_dir

logger = logging.getLogger("coding_assistant")
logger.setLevel(logging.INFO)


async def _main(args: argparse.Namespace) -> None:
    """Run the CLI and translate Ctrl-C into a clean shutdown."""
    logger.info(f"Starting Coding Assistant with arguments {args}")
    try:
        await run_cli(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the coding assistant executable."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description="Coding Assistant CLI")
    parser.add_argument("--model", type=str, required=True, help="Model to use for the orchestrator agent.")
    parser.add_argument(
        "--instructions",
        nargs="*",
        default=[],
        help="Custom instructions for the agent.",
    )
    parser.add_argument(
        "--wait-for-debugger",
        action=BooleanOptionalAction,
        default=False,
        help="Wait for a debugger to attach.",
    )
    parser.add_argument(
        "--bell",
        action=BooleanOptionalAction,
        default=True,
        help="Ring the terminal bell when a run finishes.",
    )
    parser.add_argument(
        "--trace",
        action=BooleanOptionalAction,
        default=True,
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
