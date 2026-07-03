from __future__ import annotations

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

from coding_assistant.manager import models  # noqa: F401
from coding_assistant.manager.db import database_url


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = SQLModel.metadata

CODING_ASSISTANT_MANAGER_DATABASE_PATH_ENV = "CODING_ASSISTANT_MANAGER_DATABASE_PATH"
CODING_ASSISTANT_MANAGER_DATA_DIR_ENV = "CODING_ASSISTANT_MANAGER_DATA_DIR"


def _configure_database_url() -> None:
    # Honor a URL already set on the config (e.g. by tests driving alembic.command
    # directly) so callers do not need to mutate the process environment.
    if config.get_main_option("sqlalchemy.url"):
        return
    database_path = os.environ.get(CODING_ASSISTANT_MANAGER_DATABASE_PATH_ENV)
    if database_path:
        path = Path(database_path)
    else:
        data_dir = os.environ.get(CODING_ASSISTANT_MANAGER_DATA_DIR_ENV)
        if not data_dir:
            raise RuntimeError(
                "CODING_ASSISTANT_MANAGER_DATABASE_PATH or "
                "CODING_ASSISTANT_MANAGER_DATA_DIR must be set to run migrations."
            )
        path = Path(data_dir) / "sessions.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    config.set_main_option("sqlalchemy.url", database_url(path))


def run_migrations_offline() -> None:
    _configure_database_url()
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    _configure_database_url()
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
