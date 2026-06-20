from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from coding_assistant.llm.types import BaseMessage, message_from_dict, message_to_dict
from coding_assistant.manager.workspace import WorkspacePaths


class SessionNotFoundError(RuntimeError):
    pass


class StaleSessionCommitError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    scope_id: str
    version: int
    created_at: str
    updated_at: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedSession:
    record: SessionRecord
    messages: list[BaseMessage]
    workspace: Path


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_session_id() -> str:
    return f"sess_{uuid4().hex}"


def _message_to_json(message: BaseMessage) -> str:
    return json.dumps(message_to_dict(message), sort_keys=True)


def _message_from_json(payload: str) -> BaseMessage:
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("Stored message payload must be a JSON object.")
    return message_from_dict(decoded)


def _metadata_to_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True)


def _metadata_from_json(payload: str) -> dict[str, Any]:
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        return {}
    return decoded


def _record_from_row(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=str(row["session_id"]),
        scope_id=str(row["scope_id"]),
        title=row["title"] if isinstance(row["title"], str) else None,
        version=int(row["version"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        metadata=_metadata_from_json(str(row["metadata_json"])),
    )


class SessionStore:
    def __init__(self, *, database_path: Path, workspaces: WorkspacePaths) -> None:
        self.database_path = database_path
        self.workspaces = workspaces
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._init_schema(connection)

    def create_session(
        self,
        *,
        scope_id: str,
        messages: list[BaseMessage],
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LoadedSession:
        session_id = _new_session_id()
        now = _now_iso()
        workspace = self.workspaces.create_for_session(session_id)
        metadata_json = _metadata_to_json(metadata or {})
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    insert into sessions
                      (session_id, scope_id, title, version, created_at, updated_at, metadata_json)
                    values (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, scope_id, title, 0, now, now, metadata_json),
                )
                self._insert_messages(connection, session_id=session_id, version=0, messages=messages, created_at=now)

        record = SessionRecord(
            session_id=session_id,
            scope_id=scope_id,
            title=title,
            version=0,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        return LoadedSession(record=record, messages=list(messages), workspace=workspace)

    def list_sessions(self, *, scope_id: str) -> list[SessionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select session_id, scope_id, title, version, created_at, updated_at, metadata_json
                from sessions
                where scope_id = ?
                order by updated_at desc, session_id asc
                """,
                (scope_id,),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def load_session(self, *, scope_id: str, session_id: str) -> LoadedSession:
        with self._connect() as connection:
            record = self._get_session_record(connection, scope_id=scope_id, session_id=session_id)
            message_rows = connection.execute(
                """
                select payload_json
                from session_messages
                where session_id = ?
                order by id asc
                """,
                (session_id,),
            ).fetchall()
        workspace = self.workspaces.require_for_session(session_id)
        messages = [_message_from_json(str(row["payload_json"])) for row in message_rows]
        return LoadedSession(record=record, messages=messages, workspace=workspace)

    def rename_session(self, *, scope_id: str, session_id: str, title: str | None) -> SessionRecord:
        now = _now_iso()
        with self._connect() as connection:
            with connection:
                self._get_session_record(connection, scope_id=scope_id, session_id=session_id)
                connection.execute(
                    """
                    update sessions
                    set title = ?, updated_at = ?
                    where scope_id = ? and session_id = ?
                    """,
                    (title, now, scope_id, session_id),
                )
                return self._get_session_record(connection, scope_id=scope_id, session_id=session_id)

    def update_session_metadata(
        self,
        *,
        scope_id: str,
        session_id: str,
        metadata: dict[str, Any],
    ) -> SessionRecord:
        now = _now_iso()
        with self._connect() as connection:
            with connection:
                record = self._get_session_record(connection, scope_id=scope_id, session_id=session_id)
                next_metadata = {**record.metadata, **metadata}
                connection.execute(
                    """
                    update sessions
                    set updated_at = ?, metadata_json = ?
                    where scope_id = ? and session_id = ?
                    """,
                    (now, _metadata_to_json(next_metadata), scope_id, session_id),
                )
                return self._get_session_record(connection, scope_id=scope_id, session_id=session_id)

    def commit_messages(
        self,
        *,
        scope_id: str,
        session_id: str,
        base_version: int,
        messages: list[BaseMessage],
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = _now_iso()
        with self._connect() as connection:
            with connection:
                record = self._get_session_record(connection, scope_id=scope_id, session_id=session_id)
                if record.version != base_version:
                    raise StaleSessionCommitError(
                        f"Session {session_id} is at version {record.version}, not {base_version}.",
                    )
                new_version = base_version + 1
                self._insert_messages(
                    connection,
                    session_id=session_id,
                    version=new_version,
                    messages=messages,
                    created_at=now,
                )
                next_title = title if title is not None else record.title
                next_metadata = metadata if metadata is not None else record.metadata
                connection.execute(
                    """
                    update sessions
                    set version = ?, title = ?, updated_at = ?, metadata_json = ?
                    where session_id = ?
                    """,
                    (new_version, next_title, now, _metadata_to_json(next_metadata), session_id),
                )
        return new_version

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma foreign_keys = on")
        return connection

    def _init_schema(self, connection: sqlite3.Connection) -> None:
        with connection:
            connection.execute(
                """
                create table if not exists sessions (
                  session_id text primary key,
                  scope_id text not null,
                  title text null,
                  version integer not null default 0,
                  created_at text not null,
                  updated_at text not null,
                  metadata_json text not null default '{}'
                )
                """,
            )
            connection.execute(
                """
                create table if not exists session_messages (
                  id integer primary key autoincrement,
                  session_id text not null references sessions(session_id) on delete cascade,
                  version integer not null,
                  role text not null,
                  payload_json text not null,
                  created_at text not null
                )
                """,
            )
            connection.execute(
                "create index if not exists idx_sessions_scope_updated on sessions(scope_id, updated_at desc)",
            )
            connection.execute(
                "create index if not exists idx_session_messages_session on session_messages(session_id, id)",
            )

    def _get_session_record(self, connection: sqlite3.Connection, *, scope_id: str, session_id: str) -> SessionRecord:
        row = connection.execute(
            """
            select session_id, scope_id, title, version, created_at, updated_at, metadata_json
            from sessions
            where scope_id = ? and session_id = ?
            """,
            (scope_id, session_id),
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session {session_id} was not found.")
        return _record_from_row(row)

    def _insert_messages(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        version: int,
        messages: list[BaseMessage],
        created_at: str,
    ) -> None:
        connection.executemany(
            """
            insert into session_messages (session_id, version, role, payload_json, created_at)
            values (?, ?, ?, ?, ?)
            """,
            [(session_id, version, message.role, _message_to_json(message), created_at) for message in messages],
        )
