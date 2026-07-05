from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from coding_assistant.infra.logging import setup_logging
from coding_assistant.manager.config import (
    MANAGER_HOST,
    MANAGER_PORT,
    WORKER_PORT,
    WORKER_WORKSPACE,
    ManagerConfig,
    _manager_config_from_env,
    _worker_environment_from_env,
)
from coding_assistant.manager.docker_worker import DockerWorkerConfig, DockerWorkerRunner
from coding_assistant.manager.server import start_manager_server
from coding_assistant.manager.service import ManagerService
from coding_assistant.manager.store import SessionStore
from coding_assistant.manager.workspace import WorkspacePaths


def _reject_cli_args(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise ValueError("coding-assistant-manager is configured only through environment variables.")


async def _main(config: ManagerConfig) -> None:
    config.session_root.mkdir(parents=True, exist_ok=True)
    store = SessionStore(
        database_path=config.database_path,
        workspaces=WorkspacePaths(root=config.session_root),
    )
    worker_config = DockerWorkerConfig(
        image=config.worker_image,
        network=config.worker_network,
        manager_session_root=str(config.session_root),
        worker_port=WORKER_PORT,
        session_source_root=str(config.session_source_root),
        workspace_mount=WORKER_WORKSPACE,
        environment=_worker_environment_from_env(),
        extra_hosts=config.worker_extra_hosts,
    )
    service = ManagerService(
        store=store,
        worker_runner=DockerWorkerRunner(config=worker_config),
        worker_workspace=Path(WORKER_WORKSPACE),
    )
    async with start_manager_server(
        service=service,
        host=MANAGER_HOST,
        port=MANAGER_PORT,
        auth_secret=config.auth_secret,
    ) as server:
        print(f"Manager endpoint: {server.endpoint}", flush=True)
        await asyncio.Event().wait()


def main() -> None:
    _reject_cli_args(sys.argv)
    setup_logging(console=True)
    asyncio.run(_main(_manager_config_from_env()))


if __name__ == "__main__":
    main()
