from __future__ import annotations

from typing import Any, ClassVar

from sqlalchemy import Column, ForeignKey, Index, Integer, String, UniqueConstraint, text
from sqlmodel import Field, SQLModel


class ManagerSession(SQLModel, table=True):
    __tablename__: ClassVar[str] = "sessions"
    __table_args__: ClassVar[Any] = (Index("idx_sessions_scope_updated", "scope_id", "updated_at"),)

    session_id: str = Field(sa_column=Column(String, primary_key=True))
    scope_id: str = Field(sa_column=Column(String, nullable=False))
    title: str | None = Field(default=None, sa_column=Column(String, nullable=True))
    created_at: str = Field(sa_column=Column(String, nullable=False))
    updated_at: str = Field(sa_column=Column(String, nullable=False))
    metadata_json: str = Field(default="{}", sa_column=Column(String, nullable=False, server_default=text("'{}'")))
    worker_env_json: str = Field(default="{}", sa_column=Column(String, nullable=False, server_default=text("'{}'")))


class SessionMessageRow(SQLModel, table=True):
    __tablename__: ClassVar[str] = "session_messages"
    __table_args__: ClassVar[Any] = (Index("idx_session_messages_session", "session_id", "id"),)

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(
        sa_column=Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False),
    )
    role: str = Field(sa_column=Column(String, nullable=False))
    payload_json: str = Field(sa_column=Column(String, nullable=False))
    created_at: str = Field(sa_column=Column(String, nullable=False))


class SessionAttachmentRow(SQLModel, table=True):
    __tablename__: ClassVar[str] = "session_attachments"
    __table_args__: ClassVar[Any] = (
        UniqueConstraint("session_id", "sequence"),
        UniqueConstraint("session_id", "attachment_id"),
        Index("idx_session_attachments_session", "session_id", "sequence"),
        Index("idx_session_attachments_id", "session_id", "attachment_id"),
        Index("idx_session_attachments_message", "session_id", "message_id"),
        Index("idx_session_attachments_unique_id", "session_id", "attachment_id", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    attachment_id: str = Field(sa_column=Column(String, nullable=False))
    session_id: str = Field(
        sa_column=Column(String, ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False),
    )
    message_id: int = Field(
        sa_column=Column(Integer, ForeignKey("session_messages.id", ondelete="CASCADE"), nullable=False),
    )
    sequence: int = Field(sa_column=Column(Integer, nullable=False))
    name: str = Field(sa_column=Column(String, nullable=False))
    mime_type: str = Field(sa_column=Column(String, nullable=False))
    size: int = Field(sa_column=Column(Integer, nullable=False))
    path: str = Field(sa_column=Column(String, nullable=False))
    sha256: str = Field(sa_column=Column(String, nullable=False))
    created_at: str = Field(sa_column=Column(String, nullable=False))
