from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from coding_assistant.llm.provider_config import require_provider_key, worker_provider_environment


CODING_ASSISTANT_HOST_DATA_DIR_ENV = "CODING_ASSISTANT_HOST_DATA_DIR"
CODING_ASSISTANT_MANAGER_AUTH_SECRET_ENV = "CODING_ASSISTANT_MANAGER_AUTH_SECRET"
CODING_ASSISTANT_WORKER_IMAGE_ENV = "CODING_ASSISTANT_WORKER_IMAGE"
CODING_ASSISTANT_WORKER_NETWORK_ENV = "CODING_ASSISTANT_WORKER_NETWORK"
CODING_ASSISTANT_WORKER_EXTRA_HOSTS_ENV = "CODING_ASSISTANT_WORKER_EXTRA_HOSTS"
MANAGER_DATA_DIR = Path("/data")
MANAGER_HOST = "0.0.0.0"
MANAGER_PORT = 8764
WORKER_PORT = 8765
WORKER_WORKSPACE = "/workspace"


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
    return worker_provider_environment()


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
    require_provider_key()
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
