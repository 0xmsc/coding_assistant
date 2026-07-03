"""initial manager schema

Revision ID: 534287abf7fd
Revises:
Create Date: 2026-07-02 22:38:58.621369

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "534287abf7fd"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("version", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.Column("metadata_json", sa.String(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("worker_env_json", sa.String(), server_default=sa.text("'{}'"), nullable=False),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index("idx_sessions_scope_updated", "sessions", ["scope_id", "updated_at"])

    op.create_table(
        "session_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("payload_json", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_session_messages_session", "session_messages", ["session_id", "id"])

    op.create_table(
        "session_attachments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("attachment_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("mime_type", sa.String(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["session_messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "attachment_id"),
        sa.UniqueConstraint("session_id", "sequence"),
    )
    op.create_index("idx_session_attachments_id", "session_attachments", ["session_id", "attachment_id"])
    op.create_index("idx_session_attachments_message", "session_attachments", ["session_id", "message_id"])
    op.create_index("idx_session_attachments_session", "session_attachments", ["session_id", "sequence"])
    op.create_index(
        "idx_session_attachments_unique_id",
        "session_attachments",
        ["session_id", "attachment_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("idx_session_attachments_unique_id", table_name="session_attachments")
    op.drop_index("idx_session_attachments_session", table_name="session_attachments")
    op.drop_index("idx_session_attachments_message", table_name="session_attachments")
    op.drop_index("idx_session_attachments_id", table_name="session_attachments")
    op.drop_table("session_attachments")
    op.drop_index("idx_session_messages_session", table_name="session_messages")
    op.drop_table("session_messages")
    op.drop_index("idx_sessions_scope_updated", table_name="sessions")
    op.drop_table("sessions")
