from __future__ import annotations

import os
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
from collections.abc import AsyncIterator

from coding_assistant.infra.paths import get_app_state_dir


@dataclass(frozen=True)
class WorkerRecord:
    endpoint: str
    pid: int
    cwd: str
    started_at: str


def get_worker_records_dir() -> Path:
    return get_app_state_dir() / "workers"


def list_worker_records(*, exclude_endpoint: str | None = None) -> list[WorkerRecord]:
    records_dir = get_worker_records_dir()
    if not records_dir.exists():
        return []

    records: list[WorkerRecord] = []
    for record_path in records_dir.glob("*.json"):
        record = _read_record(record_path)
        if record is None:
            with suppress(FileNotFoundError):
                record_path.unlink()
            continue
        if exclude_endpoint is not None and record.endpoint == exclude_endpoint:
            continue
        records.append(record)

    return sorted(records, key=lambda item: item.endpoint)


@asynccontextmanager
async def advertise_worker(*, endpoint: str, cwd: Path) -> AsyncIterator[WorkerRecord]:
    records_dir = get_worker_records_dir()
    records_dir.mkdir(parents=True, exist_ok=True)
    record = WorkerRecord(
        endpoint=endpoint,
        pid=os.getpid(),
        cwd=str(cwd),
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    record_path = records_dir / f"{record.pid}.json"
    record_path.write_text(json.dumps(record.__dict__), encoding="utf-8")
    try:
        yield record
    finally:
        with suppress(FileNotFoundError):
            record_path.unlink()


def _read_record(record_path: Path) -> WorkerRecord | None:
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        record = WorkerRecord(
            endpoint=payload["endpoint"],
            pid=int(payload["pid"]),
            cwd=payload["cwd"],
            started_at=payload["started_at"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if not _process_is_alive(record.pid):
        return None
    return record


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
