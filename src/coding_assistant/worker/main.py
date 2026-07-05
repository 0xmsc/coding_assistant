from __future__ import annotations

import argparse
import asyncio
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from coding_assistant.infra.logging import setup_logging
from coding_assistant.worker.agent import WorkerAgentConfig, create_worker_agent
from coding_assistant.worker.server import WorkerRuntimeConfig, start_session_worker_server


TOOL_ENV_KEYS_ENV = "CODING_ASSISTANT_TOOL_ENV_KEYS"


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description="Coding Assistant worker")
    parser.add_argument("--model", required=True, help="Model to use for the worker agent.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the worker WebSocket server.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind the worker WebSocket server.")
    parser.add_argument("--workspace", default="/workspace", help="Mounted worker workspace path.")
    parser.add_argument("--instructions", nargs="*", default=[], help="Additional worker instructions.")
    parser.add_argument("--skills-directories", nargs="*", default=[], help="Additional Agent Skill directories.")
    return parser.parse_args()


def _tool_process_env_from_env() -> dict[str, str]:
    raw_keys = os.environ.get(TOOL_ENV_KEYS_ENV, "")
    keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
    return {key: os.environ[key] for key in keys if key in os.environ}


async def _main(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    config = WorkerAgentConfig(
        working_directory=workspace,
        skills_directories=tuple(args.skills_directories),
        user_instructions=tuple(args.instructions),
        process_env=_tool_process_env_from_env(),
    )
    async with create_worker_agent(config=config) as bundle:
        runtime = WorkerRuntimeConfig(
            model=args.model,
            tools=bundle.tools,
            finish_metadata_provider=bundle.session_title_state.finish_metadata,
        )
        async with start_session_worker_server(runtime=runtime, host=args.host, port=args.port) as server:
            print(f"Worker endpoint: {server.endpoint}", flush=True)
            await server.wait_finished()


def main() -> None:
    args = parse_args()
    setup_logging(console=True)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
