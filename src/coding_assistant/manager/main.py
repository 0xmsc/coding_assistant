from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from coding_assistant.infra.logging import setup_logging
from coding_assistant.manager.docker_worker import DockerWorkerConfig, DockerWorkerRunner
from coding_assistant.manager.server import start_manager_server
from coding_assistant.manager.service import ManagerService
from coding_assistant.manager.store import SessionStore
from coding_assistant.manager.workspace import WorkspacePaths


CODING_ASSISTANT_HOST_DATA_DIR_ENV = "CODING_ASSISTANT_HOST_DATA_DIR"
CODING_ASSISTANT_MANAGER_AUTH_SECRET_ENV = "CODING_ASSISTANT_MANAGER_AUTH_SECRET"
CODING_ASSISTANT_WORKER_IMAGE_ENV = "CODING_ASSISTANT_WORKER_IMAGE"
CODING_ASSISTANT_WORKER_NETWORK_ENV = "CODING_ASSISTANT_WORKER_NETWORK"
CODING_ASSISTANT_WORKER_EXTRA_HOSTS_ENV = "CODING_ASSISTANT_WORKER_EXTRA_HOSTS"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
MANAGER_DATA_DIR = Path("/data")
MANAGER_HOST = "0.0.0.0"
MANAGER_PORT = 8764
WORKER_PORT = 8765
WORKER_WORKSPACE = "/workspace"
WORKER_PROVIDER_ENV_KEYS = (OPENAI_API_KEY_ENV, OPENAI_BASE_URL_ENV)


@dataclass(frozen=True)
class ManagerConfig:
    auth_secret: str
    data_dir: Path
    host_data_dir: Path
    worker_image: str
    worker_network: str
    worker_extra_hosts: tuple[str, ...]
    session_source_root: Path

    @property
    def database_path(self) -> Path:
        return self.data_dir / "sessions.sqlite"

    @property
    def session_root(self) -> Path:
        return self.data_dir / "sessions"


def _worker_environment_from_env() -> dict[str, str]:
    return {key: os.environ[key] for key in WORKER_PROVIDER_ENV_KEYS if os.environ.get(key)}


def _worker_extra_hosts_from_env() -> tuple[str, ...]:
    raw_value = os.environ.get(CODING_ASSISTANT_WORKER_EXTRA_HOSTS_ENV, "")
    return tuple(host.strip() for host in raw_value.split(",") if host.strip())


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set to start the manager.")
    return value


def _absolute_path_env(name: str) -> Path:
    path = Path(_required_env(name))
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path.")
    return path


def _manager_auth_secret_from_env() -> str:
    return _required_env(CODING_ASSISTANT_MANAGER_AUTH_SECRET_ENV)


def _manager_config_from_env() -> ManagerConfig:
    _required_env(OPENAI_API_KEY_ENV)
    host_data_dir = _absolute_path_env(CODING_ASSISTANT_HOST_DATA_DIR_ENV)
    return ManagerConfig(
        auth_secret=_manager_auth_secret_from_env(),
        data_dir=MANAGER_DATA_DIR,
        host_data_dir=host_data_dir,
        worker_image=_required_env(CODING_ASSISTANT_WORKER_IMAGE_ENV),
        worker_network=_required_env(CODING_ASSISTANT_WORKER_NETWORK_ENV),
        worker_extra_hosts=_worker_extra_hosts_from_env(),
        session_source_root=host_data_dir / "sessions",
    )


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
    setup_logging()
    asyncio.run(_main(_manager_config_from_env()))


if __name__ == "__main__":
    main()
