from __future__ import annotations

import argparse
import asyncio
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from coding_assistant.app.default_agent import DefaultAgentConfig, create_default_agent
from coding_assistant.app.main import setup_logging
from coding_assistant.tools.mcp_manager import MCPServerConfig
from coding_assistant.worker.server import WorkerRuntimeConfig, start_session_worker_server


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description="Coding Assistant worker")
    parser.add_argument("--model", required=True, help="Model to use for the worker agent.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the worker WebSocket server.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind the worker WebSocket server.")
    parser.add_argument("--workspace", default="/workspace", help="Mounted worker workspace path.")
    parser.add_argument("--instructions", nargs="*", default=[], help="Additional worker instructions.")
    parser.add_argument("--skills-directories", nargs="*", default=[], help="Additional Agent Skill directories.")
    parser.add_argument("--mcp-servers", nargs="*", default=[], help="MCP server configurations as JSON strings.")
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    config = DefaultAgentConfig(
        working_directory=workspace,
        mcp_server_configs=tuple(MCPServerConfig.model_validate_json(item) for item in args.mcp_servers),
        skills_directories=tuple(args.skills_directories),
        user_instructions=tuple(args.instructions),
    )
    async with create_default_agent(config=config) as bundle:
        runtime = WorkerRuntimeConfig(model=args.model, tools=bundle.tools)
        async with start_session_worker_server(runtime=runtime, host=args.host, port=args.port) as server:
            print(f"Worker endpoint: {server.endpoint}", flush=True)
            await asyncio.Event().wait()


def main() -> None:
    args = parse_args()
    setup_logging()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
