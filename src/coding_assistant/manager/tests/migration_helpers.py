from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config


CODING_ASSISTANT_MANAGER_DATABASE_PATH_ENV = "CODING_ASSISTANT_MANAGER_DATABASE_PATH"


def run_migrations(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    config = Config()
    config.set_main_option("script_location", "src/coding_assistant/manager/alembic")
    old_database_path = os.environ.get(CODING_ASSISTANT_MANAGER_DATABASE_PATH_ENV)
    os.environ[CODING_ASSISTANT_MANAGER_DATABASE_PATH_ENV] = str(database_path)
    try:
        command.upgrade(config, "head")
    finally:
        if old_database_path is None:
            os.environ.pop(CODING_ASSISTANT_MANAGER_DATABASE_PATH_ENV, None)
        else:
            os.environ[CODING_ASSISTANT_MANAGER_DATABASE_PATH_ENV] = old_database_path
