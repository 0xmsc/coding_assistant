from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from coding_assistant.manager.db import create_manager_engine, database_url


DEFAULT_MANAGER_DATA_DIR = Path("/data")
CODING_ASSISTANT_MANAGER_DATABASE_PATH_ENV = "CODING_ASSISTANT_MANAGER_DATABASE_PATH"
CODING_ASSISTANT_MANAGER_DATA_DIR_ENV = "CODING_ASSISTANT_MANAGER_DATA_DIR"


def database_path_from_env() -> Path:
    database_path = os.environ.get(CODING_ASSISTANT_MANAGER_DATABASE_PATH_ENV)
    if database_path:
        return Path(database_path)
    data_dir = os.environ.get(CODING_ASSISTANT_MANAGER_DATA_DIR_ENV)
    if data_dir:
        return Path(data_dir) / "sessions.sqlite"
    return DEFAULT_MANAGER_DATA_DIR / "sessions.sqlite"


def alembic_config(database_path: Path) -> Config:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).with_name("alembic")))
    config.set_main_option("sqlalchemy.url", database_url(database_path))
    return config


def migrate_database(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    config = alembic_config(database_path)
    if _needs_legacy_bootstrap(database_path):
        _bootstrap_legacy_schema(database_path)
        command.stamp(config, "head")
        return
    command.upgrade(config, "head")


def _needs_legacy_bootstrap(database_path: Path) -> bool:
    if not database_path.exists():
        return False
    engine = create_manager_engine(database_path)
    with engine.connect() as connection:
        inspector = inspect(connection)
        return inspector.has_table("sessions") and not inspector.has_table("alembic_version")


def _bootstrap_legacy_schema(database_path: Path) -> None:
    engine = create_manager_engine(database_path)
    with engine.begin() as connection:
        connection.execute(
            text(
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
            ),
        )
        session_columns = {str(row.name) for row in connection.execute(text("pragma table_info(sessions)"))}
        if "worker_env_json" not in session_columns:
            connection.execute(text("alter table sessions add column worker_env_json text not null default '{}'"))
        connection.execute(
            text(
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
            ),
        )
        connection.execute(
            text(
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
            ),
        )
        connection.execute(
            text("create index if not exists idx_sessions_scope_updated on sessions(scope_id, updated_at)")
        )
        connection.execute(
            text("create index if not exists idx_session_messages_session on session_messages(session_id, id)"),
        )
        connection.execute(
            text(
                "create index if not exists idx_session_attachments_session "
                "on session_attachments(session_id, sequence)",
            ),
        )
        connection.execute(
            text(
                "create index if not exists idx_session_attachments_id "
                "on session_attachments(session_id, attachment_id)",
            ),
        )
        connection.execute(
            text(
                "create index if not exists idx_session_attachments_message "
                "on session_attachments(session_id, message_id)",
            ),
        )
        connection.execute(
            text(
                "create unique index if not exists idx_session_attachments_unique_id "
                "on session_attachments(session_id, attachment_id)",
            ),
        )


def main() -> None:
    migrate_database(database_path_from_env())


if __name__ == "__main__":
    main()
