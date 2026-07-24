from __future__ import annotations

import sqlite3
from pathlib import Path

from coding_assistant.manager.migration_backup import BACKUP_LIMIT, create_backup


def write_value(database_path: Path, value: str) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS values_table (value TEXT)")
        connection.execute("DELETE FROM values_table")
        connection.execute("INSERT INTO values_table VALUES (?)", (value,))


def read_value(database_path: Path) -> str:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT value FROM values_table").fetchone()
    assert row is not None
    value = row[0]
    assert isinstance(value, str)
    return value


def test_create_backup_copies_database(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.sqlite"
    write_value(database_path, "before migration")

    backup_path = create_backup(tmp_path, "eb428987")

    assert backup_path == tmp_path / "backups" / "pre-migration-eb428987.sqlite3"
    assert read_value(backup_path) == "before migration"


def test_create_backup_does_not_overwrite_existing_revision(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.sqlite"
    write_value(database_path, "original")
    backup_path = create_backup(tmp_path, "eb428987")
    write_value(database_path, "changed")

    assert create_backup(tmp_path, "eb428987") == backup_path
    assert read_value(backup_path) == "original"


def test_create_backup_keeps_five_newest_snapshots(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.sqlite"

    for index in range(BACKUP_LIMIT + 1):
        write_value(database_path, str(index))
        create_backup(tmp_path, f"revision-{index}")

    backups = sorted((tmp_path / "backups").glob("*.sqlite3"))
    assert len(backups) == BACKUP_LIMIT
    assert tmp_path / "backups" / "pre-migration-revision-0.sqlite3" not in backups
