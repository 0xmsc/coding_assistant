from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from alembic import command
from sqlalchemy import inspect, text

from coding_assistant.llm.types import SystemMessage, message_to_dict
from coding_assistant.manager.db import create_manager_engine
from coding_assistant.manager.migrations import alembic_config
from coding_assistant.manager.store import SessionStore
from coding_assistant.manager.workspace import WorkspacePaths


def test_alembic_upgrade_creates_manager_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.sqlite"

    command.upgrade(alembic_config(database_path), "head")

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


def test_store_bootstraps_unversioned_legacy_database(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.sqlite"
    session_id = "sess_legacy"
    _create_legacy_database(database_path=database_path, session_id=session_id)
    WorkspacePaths(root=tmp_path / "sessions").create_for_session(session_id)

    store = SessionStore(
        database_path=database_path,
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    loaded = store.load_session(scope_id="scope-a", session_id=session_id)

    assert loaded.messages == [SystemMessage(content="system")]
    assert loaded.worker_env == {}
    engine = create_manager_engine(database_path)
    with engine.connect() as connection:
        assert inspect(connection).has_table("alembic_version")
        assert inspect(connection).has_table("session_attachments")
        assert connection.execute(text("select version_num from alembic_version")).scalar_one() is not None


def _create_legacy_database(*, database_path: Path, session_id: str) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            create table sessions (
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
            create table session_messages (
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
            insert into sessions
              (session_id, scope_id, title, version, created_at, updated_at, metadata_json)
            values (?, 'scope-a', null, 0, '2026-07-02T00:00:00+00:00', '2026-07-02T00:00:00+00:00', '{}')
            """,
            (session_id,),
        )
        connection.execute(
            """
            insert into session_messages (session_id, version, role, payload_json, created_at)
            values (?, 0, 'system', ?, '2026-07-02T00:00:00+00:00')
            """,
            (session_id, json.dumps(message_to_dict(SystemMessage(content="system")), sort_keys=True)),
        )
