from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from coding_assistant.infra.paths import get_app_runtime_dir


@dataclass(frozen=True, slots=True)
class RemoteRegistryEntry:
    pid: int
    port: int
    endpoint: str
    cwd: str
    started_at: str


def get_remote_registry_dir() -> Path:
    """Return the directory containing live remote registry entries."""
    return get_app_runtime_dir() / "remotes"


@asynccontextmanager
async def register_remote_instance(*, endpoint: str, cwd: str | None = None) -> AsyncIterator[None]:
    """Advertise one live remote endpoint for the current process."""
    pid = os.getpid()
    port = _endpoint_port(endpoint)
    registry_dir = get_remote_registry_dir()
    registry_dir.mkdir(parents=True, exist_ok=True)

    for existing_path in registry_dir.glob(f"{pid}-*.json"):
        _safe_unlink(existing_path)

    entry_path = registry_dir / f"{pid}-{port}.json"
    payload = {
        "pid": pid,
        "port": port,
        "endpoint": endpoint,
        "cwd": cwd or os.getcwd(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    entry_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        yield
    finally:
        _safe_unlink(entry_path)


def discover_remote_instances(*, current_pid: int | None = None) -> list[RemoteRegistryEntry]:
    """Return live registry entries and delete obviously stale ones."""
    registry_dir = get_remote_registry_dir()
    if not registry_dir.exists():
        return []

    discovered: list[RemoteRegistryEntry] = []
    for entry_path in sorted(registry_dir.glob("*.json")):
        entry = _load_registry_entry(entry_path)
        if entry is None:
            _safe_unlink(entry_path)
            continue
        if current_pid is not None and entry.pid == current_pid:
            continue
        if not _pid_is_running(entry.pid):
            _safe_unlink(entry_path)
            continue
        discovered.append(entry)
    return discovered


def _load_registry_entry(entry_path: Path) -> RemoteRegistryEntry | None:
    try:
        payload = json.loads(entry_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    pid = payload.get("pid")
    port = payload.get("port")
    endpoint = payload.get("endpoint")
    cwd = payload.get("cwd")
    started_at = payload.get("started_at")
    if not isinstance(pid, int) or not isinstance(port, int):
        return None
    if not isinstance(endpoint, str) or not isinstance(cwd, str) or not isinstance(started_at, str):
        return None
    return RemoteRegistryEntry(
        pid=pid,
        port=port,
        endpoint=endpoint,
        cwd=cwd,
        started_at=started_at,
    )


def _endpoint_port(endpoint: str) -> int:
    parsed = urlparse(endpoint)
    port = parsed.port
    if port is None:
        raise ValueError(f"Remote endpoint '{endpoint}' does not include a port.")
    return port


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _safe_unlink(path: Path) -> None:
    with suppress(FileNotFoundError):
        path.unlink()
