from __future__ import annotations

import argparse
import asyncio
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from coding_assistant.core.runtime import build_initial_system_message
from coding_assistant.infra.logging import setup_logging
from coding_assistant.llm.types import BaseMessage
from coding_assistant.manager.docker_worker import DockerWorkerConfig, DockerWorkerRunner
from coding_assistant.manager.server import start_manager_server
from coding_assistant.manager.service import ManagerService
from coding_assistant.manager.store import SessionStore
from coding_assistant.manager.workspace import WorkspacePaths
from coding_assistant.worker.agent import WorkerAgentConfig, build_worker_instructions


CODING_ASSISTANT_MANAGER_AUTH_SECRET_ENV = "CODING_ASSISTANT_MANAGER_AUTH_SECRET"
WORKER_PROVIDER_ENV_KEYS = ("OPENAI_API_KEY", "OPENAI_BASE_URL")


def _worker_environment_from_env() -> dict[str, str]:
    return {key: os.environ[key] for key in WORKER_PROVIDER_ENV_KEYS if key in os.environ}


def _manager_auth_secret_from_env() -> str:
    value = os.environ.get(CODING_ASSISTANT_MANAGER_AUTH_SECRET_ENV)
    if not value:
        raise ValueError(f"{CODING_ASSISTANT_MANAGER_AUTH_SECRET_ENV} must be set to start the manager.")
    return value


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description="Coding Assistant manager")
    parser.add_argument("--model", required=True, help="Model to use for worker agents.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the manager WebSocket server.")
    parser.add_argument("--port", type=int, default=8764, help="Port to bind the manager WebSocket server.")
    parser.add_argument("--database", default="/data/sessions.sqlite", help="SQLite database path.")
    parser.add_argument("--workspace-root", default="/data/workspaces", help="Host workspace root for sessions.")
    parser.add_argument("--worker-image", default="coding-assistant:latest", help="Docker image for worker containers.")
    parser.add_argument("--worker-network", default="coding-assistant", help="Docker network for worker containers.")
    parser.add_argument("--worker-port", type=int, default=8765, help="Worker container WebSocket port.")
    parser.add_argument("--worker-workspace", default="/workspace", help="Worker container workspace mount path.")
    parser.add_argument("--instructions", nargs="*", default=[], help="Additional worker instructions.")
    parser.add_argument("--skills-directories", nargs="*", default=[], help="Additional Agent Skill directories.")
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    workspace_root = Path(args.workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    store = SessionStore(
        database_path=Path(args.database),
        workspaces=WorkspacePaths(root=workspace_root),
    )
    worker_config = DockerWorkerConfig(
        image=args.worker_image,
        model=args.model,
        network=args.worker_network,
        worker_port=args.worker_port,
        workspace_mount=args.worker_workspace,
        environment=_worker_environment_from_env(),
        instructions=tuple(args.instructions),
        skills_directories=tuple(args.skills_directories),
    )
    service = ManagerService(store=store, worker_runner=DockerWorkerRunner(config=worker_config))
    agent_config = WorkerAgentConfig(
        working_directory=Path(args.worker_workspace),
        user_instructions=tuple(args.instructions),
        skills_directories=tuple(args.skills_directories),
    )
    instructions = build_worker_instructions(config=agent_config)
    initial_messages: list[BaseMessage] = [build_initial_system_message(instructions=instructions)]
    async with start_manager_server(
        service=service,
        initial_messages=initial_messages,
        host=args.host,
        port=args.port,
        auth_secret=_manager_auth_secret_from_env(),
    ) as server:
        print(f"Manager endpoint: {server.endpoint}", flush=True)
        await asyncio.Event().wait()


def main() -> None:
    args = parse_args()
    setup_logging()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
