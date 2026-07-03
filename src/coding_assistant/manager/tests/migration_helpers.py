from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from coding_assistant.manager.db import database_url


def run_migrations(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    config = Config()
    config.set_main_option("script_location", "src/coding_assistant/manager/alembic")
    config.set_main_option("sqlalchemy.url", database_url(database_path))
    command.upgrade(config, "head")
