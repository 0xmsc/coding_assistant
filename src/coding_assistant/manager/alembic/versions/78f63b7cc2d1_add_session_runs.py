"""add session runs

Revision ID: 78f63b7cc2d1
Revises: 534287abf7fd
Create Date: 2026-07-04 00:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "78f63b7cc2d1"
down_revision: str | None = "534287abf7fd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "session_runs",
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("started_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.Column("ended_at", sa.String(), nullable=True),
        sa.Column("stop_reason", sa.String(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("idx_session_runs_session", "session_runs", ["session_id", "started_at"])
    op.create_index("idx_session_runs_status", "session_runs", ["session_id", "status"])


def downgrade() -> None:
    op.drop_index("idx_session_runs_status", table_name="session_runs")
    op.drop_index("idx_session_runs_session", table_name="session_runs")
    op.drop_table("session_runs")
