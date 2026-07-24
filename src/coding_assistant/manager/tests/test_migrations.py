from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, text

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
            "created_at",
            "updated_at",
            "metadata_json",
            "worker_env_json",
        }
        assert {column["name"] for column in inspector.get_columns("session_messages")} == {
            "id",
            "session_id",
            "role",
            "payload_json",
            "created_at",
        }


def test_alembic_upgrade_preserves_existing_messages_and_attachments(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.sqlite"
    run_migrations(database_path, "534287abf7fd")
    engine = create_manager_engine(database_path)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO sessions (
                    session_id, scope_id, title, version, created_at, updated_at, metadata_json, worker_env_json
                ) VALUES (
                    'sess_1', 'scope-a', 'Title', 1, 'created', 'updated', '{}', '{}'
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO session_messages (
                    id, session_id, version, role, payload_json, created_at
                ) VALUES (
                    1, 'sess_1', 1, 'user', '{"role": "user", "content": "hello"}', 'created'
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO session_attachments (
                    id, attachment_id, session_id, message_id, sequence, name,
                    mime_type, size, path, sha256, created_at
                ) VALUES (
                    1, 'att_1', 'sess_1', 1, 1, 'hello.txt',
                    'text/plain', 5, '/attachments/hello.txt', 'hash', 'created'
                )
                """
            )
        )

    run_migrations(database_path)

    with engine.connect() as connection:
        message = connection.execute(text("SELECT session_id, role FROM session_messages")).one()
        attachment = connection.execute(text("SELECT attachment_id, message_id FROM session_attachments")).one()
        foreign_key_errors = connection.execute(text("PRAGMA foreign_key_check")).all()

    assert tuple(message) == ("sess_1", "user")
    assert tuple(attachment) == ("att_1", 1)
    assert foreign_key_errors == []
