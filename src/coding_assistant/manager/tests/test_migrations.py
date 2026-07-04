from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect

from coding_assistant.manager.db import create_manager_engine
from coding_assistant.manager.tests.migration_helpers import run_migrations


def test_alembic_upgrade_creates_manager_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.sqlite"

    run_migrations(database_path)

    engine = create_manager_engine(database_path)
    with engine.connect() as connection:
        inspector = inspect(connection)
        assert set(inspector.get_table_names()) == {
            "alembic_version",
            "session_attachments",
            "session_messages",
            "sessions",
        }
        assert {column["name"] for column in inspector.get_columns("sessions")} == {
            "session_id",
            "scope_id",
            "title",
            "version",
            "created_at",
            "updated_at",
            "metadata_json",
            "worker_env_json",
        }
