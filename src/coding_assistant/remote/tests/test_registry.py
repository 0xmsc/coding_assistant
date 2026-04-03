from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from coding_assistant.remote.registry import (
    discover_remote_instances,
    get_remote_registry_dir,
    register_remote_instance,
)


@pytest.mark.asyncio
async def test_register_remote_instance_writes_and_removes_registry_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))

    async with register_remote_instance(endpoint="ws://127.0.0.1:4312", cwd="/tmp/project"):
        registry_files = list(get_remote_registry_dir().glob("*.json"))
        assert len(registry_files) == 1
        payload = json.loads(registry_files[0].read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["port"] == 4312
        assert payload["endpoint"] == "ws://127.0.0.1:4312"
        assert payload["cwd"] == "/tmp/project"
        assert isinstance(payload["started_at"], str)

    assert list(get_remote_registry_dir().glob("*.json")) == []


def test_discover_remote_instances_skips_self_and_removes_dead_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    registry_dir = get_remote_registry_dir()
    registry_dir.mkdir(parents=True)

    self_pid = os.getpid()
    live_pid = self_pid + 1000
    dead_pid = self_pid + 2000

    (registry_dir / f"{self_pid}-4000.json").write_text(
        json.dumps(
            {
                "pid": self_pid,
                "port": 4000,
                "endpoint": "ws://127.0.0.1:4000",
                "cwd": "/self",
                "started_at": "2026-04-02T12:00:00+00:00",
            },
        ),
        encoding="utf-8",
    )
    live_path = registry_dir / f"{live_pid}-4001.json"
    live_path.write_text(
        json.dumps(
            {
                "pid": live_pid,
                "port": 4001,
                "endpoint": "ws://127.0.0.1:4001",
                "cwd": "/live",
                "started_at": "2026-04-02T12:01:00+00:00",
            },
        ),
        encoding="utf-8",
    )
    dead_path = registry_dir / f"{dead_pid}-4002.json"
    dead_path.write_text(
        json.dumps(
            {
                "pid": dead_pid,
                "port": 4002,
                "endpoint": "ws://127.0.0.1:4002",
                "cwd": "/dead",
                "started_at": "2026-04-02T12:02:00+00:00",
            },
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "coding_assistant.remote.registry._pid_is_running",
        lambda pid: pid in {self_pid, live_pid},
    )

    discovered = discover_remote_instances(current_pid=self_pid)

    assert [(entry.pid, entry.endpoint) for entry in discovered] == [(live_pid, "ws://127.0.0.1:4001")]
    assert live_path.exists()
    assert not dead_path.exists()
