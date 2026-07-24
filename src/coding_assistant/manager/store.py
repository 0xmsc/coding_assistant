from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlmodel import Session as SQLModelSession
from sqlmodel import col, select

from coding_assistant.core.session_updates import SessionAttachment
from coding_assistant.llm.types import BaseMessage, message_from_dict, message_to_dict
from coding_assistant.manager.db import create_manager_engine
from coding_assistant.manager.models import ManagerSession, SessionAttachmentRow, SessionMessageRow
from coding_assistant.manager.workspace import WorkspacePaths


class SessionNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    scope_id: str
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


def _record_from_row(row: ManagerSession) -> SessionRecord:
    return SessionRecord(
        session_id=row.session_id,
        scope_id=row.scope_id,
        title=row.title,
        created_at=row.created_at,
        updated_at=row.updated_at,
        metadata=_metadata_from_json(row.metadata_json),
    )


def _message_record_from_row(row: SessionMessageRow) -> StoredMessage:
    if row.id is None:
        raise ValueError("Stored message row is missing its id.")
    return StoredMessage(
        message_id=row.id,
        message=_message_from_json(row.payload_json),
        created_at=row.created_at,
    )


def _attachment_record_from_row(row: SessionAttachmentRow) -> StoredAttachment:
    return StoredAttachment(
        attachment=SessionAttachment(
            attachment_id=row.attachment_id,
            name=row.name,
            mime_type=row.mime_type,
            size=row.size,
            path=row.path,
            sha256=row.sha256,
        ),
        message_id=row.message_id,
        sequence=row.sequence,
        created_at=row.created_at,
    )


class SessionStore:
    def __init__(self, *, database_path: Path, workspaces: WorkspacePaths) -> None:
        self.database_path = database_path
        self.workspaces = workspaces
        self._engine: Engine = create_manager_engine(self.database_path)

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
        with SQLModelSession(self._engine, expire_on_commit=False) as session:
            with session.begin():
                session.add(
                    ManagerSession(
                        session_id=session_id,
                        scope_id=scope_id,
                        title=title,
                        created_at=now,
                        updated_at=now,
                        metadata_json=metadata_json,
                        worker_env_json=worker_env_json,
                    ),
                )
                message_records = self._insert_messages(
                    session,
                    session_id=session_id,
                    messages=messages,
                    created_at=now,
                )

        record = SessionRecord(
            session_id=session_id,
            scope_id=scope_id,
            title=title,
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
        with SQLModelSession(self._engine) as session:
            rows = session.exec(
                select(ManagerSession)
                .where(col(ManagerSession.scope_id) == scope_id)
                .order_by(col(ManagerSession.updated_at).desc(), col(ManagerSession.session_id).asc()),
            ).all()
        return [_record_from_row(row) for row in rows]

    def load_session(self, *, scope_id: str, session_id: str) -> LoadedSession:
        with SQLModelSession(self._engine) as session:
            session_row = self._get_session_row(session, scope_id=scope_id, session_id=session_id)
            record = _record_from_row(session_row)
            message_rows = session.exec(
                select(SessionMessageRow)
                .where(col(SessionMessageRow.session_id) == session_id)
                .order_by(col(SessionMessageRow.id).asc()),
            ).all()
            attachment_rows = session.exec(
                select(SessionAttachmentRow)
                .where(col(SessionAttachmentRow.session_id) == session_id)
                .order_by(
                    col(SessionAttachmentRow.message_id).asc(),
                    col(SessionAttachmentRow.sequence).asc(),
                    col(SessionAttachmentRow.attachment_id).asc(),
                ),
            ).all()
            worker_env = _worker_env_from_json(session_row.worker_env_json)
        paths = self.workspaces.require_for_session(session_id)
        message_records = [_message_record_from_row(row) for row in message_rows]
        messages = [record.message for record in message_records]
        attachment_records = [_attachment_record_from_row(row) for row in attachment_rows]
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
        with SQLModelSession(self._engine, expire_on_commit=False) as session:
            with session.begin():
                row = self._get_session_row(session, scope_id=scope_id, session_id=session_id)
                row.title = title
                row.updated_at = now
                session.flush()
                return _record_from_row(row)

    def update_session_metadata(
        self,
        *,
        scope_id: str,
        session_id: str,
        metadata: dict[str, Any],
    ) -> SessionRecord:
        now = _now_iso()
        with SQLModelSession(self._engine, expire_on_commit=False) as session:
            with session.begin():
                row = self._get_session_row(session, scope_id=scope_id, session_id=session_id)
                next_metadata = {**_metadata_from_json(row.metadata_json), **metadata}
                row.updated_at = now
                row.metadata_json = _metadata_to_json(next_metadata)
                session.flush()
                return _record_from_row(row)

    def delete_session(self, *, scope_id: str, session_id: str) -> None:
        """Delete a session, its messages/attachments, and the workspace tree.

        Raises ``SessionNotFoundError`` if the session does not exist or belongs
        to a different scope. The filesystem teardown runs after the row is
        deleted and is a no-op if the workspace directory is already gone.
        """
        with SQLModelSession(self._engine) as session:
            with session.begin():
                row = self._get_session_row(session, scope_id=scope_id, session_id=session_id)
                session.delete(row)
        self.workspaces.remove_for_session(session_id)

    def append_messages(
        self,
        *,
        scope_id: str,
        session_id: str,
        messages: list[BaseMessage],
        attachments: list[SessionAttachment] | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[StoredMessage]:
        now = _now_iso()
        with SQLModelSession(self._engine, expire_on_commit=False) as session:
            with session.begin():
                session_row = self._get_session_row(session, scope_id=scope_id, session_id=session_id)
                record = _record_from_row(session_row)
                inserted_messages = self._insert_messages(
                    session,
                    session_id=session_id,
                    messages=messages,
                    created_at=now,
                )
                if attachments and not inserted_messages:
                    raise ValueError("Attachments must be linked to an inserted message.")
                self._insert_attachments(
                    session,
                    session_id=session_id,
                    message_id=inserted_messages[-1].message_id if attachments else None,
                    attachments=attachments or [],
                    created_at=now,
                )
                next_title = title if title is not None else record.title
                next_metadata = metadata if metadata is not None else record.metadata
                session_row.title = next_title
                session_row.updated_at = now
                session_row.metadata_json = _metadata_to_json(next_metadata)
        return inserted_messages

    def _get_session_row(
        self,
        session: SQLModelSession,
        *,
        scope_id: str,
        session_id: str,
    ) -> ManagerSession:
        row = session.exec(
            select(ManagerSession).where(
                col(ManagerSession.scope_id) == scope_id,
                col(ManagerSession.session_id) == session_id,
            ),
        ).one_or_none()
        if row is None:
            raise SessionNotFoundError(f"Session {session_id} was not found.")
        return row

    def _insert_messages(
        self,
        session: SQLModelSession,
        *,
        session_id: str,
        messages: list[BaseMessage],
        created_at: str,
    ) -> list[StoredMessage]:
        stored_messages: list[StoredMessage] = []
        for message in messages:
            row = SessionMessageRow(
                session_id=session_id,
                role=message.role,
                payload_json=_message_to_json(message),
                created_at=created_at,
            )
            session.add(row)
            session.flush()
            if row.id is None:
                raise RuntimeError("Inserted session message did not return a row id.")
            stored_messages.append(
                StoredMessage(
                    message_id=row.id,
                    message=message,
                    created_at=created_at,
                )
            )
        return stored_messages

    def _insert_attachments(
        self,
        session: SQLModelSession,
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
        max_sequence = session.exec(
            select(func.max(SessionAttachmentRow.sequence)).where(col(SessionAttachmentRow.session_id) == session_id),
        ).one()
        next_sequence = int(max_sequence or 0) + 1
        for index, attachment in enumerate(attachments):
            session.add(
                SessionAttachmentRow(
                    attachment_id=attachment.attachment_id,
                    session_id=session_id,
                    message_id=message_id,
                    sequence=next_sequence + index,
                    name=attachment.name,
                    mime_type=attachment.mime_type,
                    size=attachment.size,
                    path=attachment.path,
                    sha256=attachment.sha256,
                    created_at=created_at,
                ),
            )
