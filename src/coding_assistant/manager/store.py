from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from coding_assistant.core.session_updates import SessionAttachment
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
class StoredMessage:
    message_id: int
    message: BaseMessage
    created_at: str


@dataclass(frozen=True)
class StoredAttachment:
    attachment: SessionAttachment
    message_id: int
    sequence: int
    created_at: str


@dataclass(frozen=True)
class LoadedSession:
    record: SessionRecord
    messages: list[BaseMessage]
    message_records: list[StoredMessage]
    attachment_records: list[StoredAttachment]
    workspace: Path
    attachments: Path
    worker_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionWorkspaceReservation:
    """Session id and filesystem paths allocated before the session row is inserted."""

    session_id: str
    root: Path
    workspace: Path
    attachments: Path


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


def _worker_env_to_json(worker_env: dict[str, str]) -> str:
    return json.dumps(worker_env, sort_keys=True)


def _worker_env_from_json(payload: str) -> dict[str, str]:
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("Stored worker environment payload must be a JSON object.")
    result: dict[str, str] = {}
    for key, value in decoded.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Stored worker environment payload must contain only string keys and values.")
        result[key] = value
    return result


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

    def reserve_session_workspace(self) -> SessionWorkspaceReservation:
        """Allocate a session id and create its workspace before database insertion."""
        session_id = _new_session_id()
        paths = self.workspaces.create_for_session(session_id)
        return SessionWorkspaceReservation(
            session_id=session_id,
            root=paths.root,
            workspace=paths.workspace,
            attachments=paths.attachments,
        )

    def create_session(
        self,
        *,
        scope_id: str,
        messages: list[BaseMessage],
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        worker_env: dict[str, str] | None = None,
        reserved_workspace: SessionWorkspaceReservation | None = None,
    ) -> LoadedSession:
        """Insert a session, optionally using a workspace prepared before initial messages were built."""
        if reserved_workspace is None:
            reserved_workspace = self.reserve_session_workspace()
        session_id = reserved_workspace.session_id
        now = _now_iso()
        workspace = reserved_workspace.workspace
        session_metadata = metadata or {}
        metadata_json = _metadata_to_json(session_metadata)
        worker_env_json = _worker_env_to_json(worker_env or {})
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    insert into sessions
                      (session_id, scope_id, title, version, created_at, updated_at, metadata_json, worker_env_json)
                    values (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, scope_id, title, 0, now, now, metadata_json, worker_env_json),
                )
                message_records = self._insert_messages(
                    connection,
                    session_id=session_id,
                    version=0,
                    messages=messages,
                    created_at=now,
                )

        record = SessionRecord(
            session_id=session_id,
            scope_id=scope_id,
            title=title,
            version=0,
            created_at=now,
            updated_at=now,
            metadata=session_metadata,
        )
        return LoadedSession(
            record=record,
            messages=list(messages),
            message_records=message_records,
            attachment_records=[],
            workspace=workspace,
            attachments=reserved_workspace.attachments,
            worker_env=dict(worker_env or {}),
        )

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
            session_row = self._get_session_row(connection, scope_id=scope_id, session_id=session_id)
            record = _record_from_row(session_row)
            message_rows = connection.execute(
                """
                select id, payload_json, created_at
                from session_messages
                where session_id = ?
                order by id asc
                """,
                (session_id,),
            ).fetchall()
            attachment_rows = connection.execute(
                """
                select attachment_id, message_id, sequence, name, mime_type, size, path, sha256, created_at
                from session_attachments
                where session_id = ?
                order by message_id asc, sequence asc, attachment_id asc
                """,
                (session_id,),
            ).fetchall()
        paths = self.workspaces.require_for_session(session_id)
        message_records = [
            StoredMessage(
                message_id=int(row["id"]),
                message=_message_from_json(str(row["payload_json"])),
                created_at=str(row["created_at"]),
            )
            for row in message_rows
        ]
        messages = [record.message for record in message_records]
        attachment_records = [
            StoredAttachment(
                attachment=SessionAttachment(
                    attachment_id=str(row["attachment_id"]),
                    name=str(row["name"]),
                    mime_type=str(row["mime_type"]),
                    size=int(row["size"]),
                    path=str(row["path"]),
                    sha256=str(row["sha256"]),
                ),
                message_id=int(row["message_id"]),
                sequence=int(row["sequence"]),
                created_at=str(row["created_at"]),
            )
            for row in attachment_rows
        ]
        worker_env = _worker_env_from_json(str(session_row["worker_env_json"]))
        return LoadedSession(
            record=record,
            messages=messages,
            message_records=message_records,
            attachment_records=attachment_records,
            workspace=paths.workspace,
            attachments=paths.attachments,
            worker_env=worker_env,
        )

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
        attachments: list[SessionAttachment] | None = None,
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
                inserted_messages = self._insert_messages(
                    connection,
                    session_id=session_id,
                    version=new_version,
                    messages=messages,
                    created_at=now,
                )
                if attachments and not inserted_messages:
                    raise ValueError("Attachments must be linked to an inserted message.")
                self._insert_attachments(
                    connection,
                    session_id=session_id,
                    message_id=inserted_messages[-1].message_id if attachments else None,
                    attachments=attachments or [],
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
                  metadata_json text not null default '{}',
                  worker_env_json text not null default '{}'
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
                """
                create table if not exists session_attachments (
                  id integer primary key autoincrement,
                  attachment_id text not null,
                  session_id text not null references sessions(session_id) on delete cascade,
                  message_id integer not null references session_messages(id) on delete cascade,
                  sequence integer not null,
                  name text not null,
                  mime_type text not null,
                  size integer not null,
                  path text not null,
                  sha256 text not null,
                  created_at text not null,
                  unique(session_id, sequence),
                  unique(session_id, attachment_id)
                )
                """,
            )
            connection.execute(
                "create index if not exists idx_sessions_scope_updated on sessions(scope_id, updated_at desc)",
            )
            connection.execute(
                "create index if not exists idx_session_messages_session on session_messages(session_id, id)",
            )
            connection.execute(
                "create index if not exists idx_session_attachments_session on session_attachments(session_id, sequence)",
            )
            connection.execute(
                "create index if not exists idx_session_attachments_id on session_attachments(session_id, attachment_id)",
            )
            connection.execute(
                "create index if not exists idx_session_attachments_message on session_attachments(session_id, message_id)",
            )
            connection.execute(
                "create unique index if not exists idx_session_attachments_unique_id "
                "on session_attachments(session_id, attachment_id)",
            )

    def _get_session_row(self, connection: sqlite3.Connection, *, scope_id: str, session_id: str) -> sqlite3.Row:
        row = connection.execute(
            """
            select session_id, scope_id, title, version, created_at, updated_at, metadata_json, worker_env_json
            from sessions
            where scope_id = ? and session_id = ?
            """,
            (scope_id, session_id),
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session {session_id} was not found.")
        return cast(sqlite3.Row, row)

    def _get_session_record(self, connection: sqlite3.Connection, *, scope_id: str, session_id: str) -> SessionRecord:
        row = self._get_session_row(connection, scope_id=scope_id, session_id=session_id)
        return _record_from_row(row)

    def _insert_messages(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        version: int,
        messages: list[BaseMessage],
        created_at: str,
    ) -> list[StoredMessage]:
        stored_messages: list[StoredMessage] = []
        for message in messages:
            cursor = connection.execute(
                """
                insert into session_messages (session_id, version, role, payload_json, created_at)
                values (?, ?, ?, ?, ?)
                """,
                (session_id, version, message.role, _message_to_json(message), created_at),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Inserted session message did not return a row id.")
            stored_messages.append(
                StoredMessage(
                    message_id=cursor.lastrowid,
                    message=message,
                    created_at=created_at,
                )
            )
        return stored_messages

    def _insert_attachments(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        message_id: int | None,
        attachments: list[SessionAttachment],
        created_at: str,
    ) -> None:
        if not attachments:
            return
        if message_id is None:
            raise ValueError("Attachments must be linked to a message.")
        row = connection.execute(
            "select coalesce(max(sequence), 0) as sequence from session_attachments where session_id = ?",
            (session_id,),
        ).fetchone()
        next_sequence = int(row["sequence"]) + 1 if row is not None else 1
        connection.executemany(
            """
            insert into session_attachments
              (attachment_id, session_id, message_id, sequence, name, mime_type, size, path, sha256, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    attachment.attachment_id,
                    session_id,
                    message_id,
                    next_sequence + index,
                    attachment.name,
                    attachment.mime_type,
                    attachment.size,
                    attachment.path,
                    attachment.sha256,
                    created_at,
                )
                for index, attachment in enumerate(attachments)
            ],
        )
