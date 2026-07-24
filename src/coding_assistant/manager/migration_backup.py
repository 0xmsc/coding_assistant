from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

DATABASE_FILENAME = "sessions.sqlite"
BACKUP_LIMIT = 5


def create_backup(data_dir: Path, revision: str) -> Path:
    database_path = data_dir / DATABASE_FILENAME
    backup_dir = data_dir / "backups"
    backup_path = backup_dir / f"pre-migration-{revision}.sqlite3"

    if backup_path.exists():
        return backup_path

    backup_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".pre-migration-{revision}-",
        suffix=".sqlite3",
        dir=backup_dir,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        with (
            closing(sqlite3.connect(database_path)) as source,
            closing(sqlite3.connect(temporary_path)) as destination,
        ):
            source.backup(destination)

        try:
            os.link(temporary_path, backup_path)
        except FileExistsError:
            pass
    finally:
        temporary_path.unlink(missing_ok=True)

    backups = sorted(
        backup_dir.glob("pre-migration-*.sqlite3"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_backup in backups[BACKUP_LIMIT:]:
        old_backup.unlink()

    return backup_path


def main() -> None:
    data_dir = Path(os.environ["CODING_ASSISTANT_MANAGER_DATA_DIR"])
    revision = os.environ["CODING_ASSISTANT_REVISION"]
    backup_path = create_backup(data_dir, revision)
    print(f"Database backup ready: {backup_path}")


if __name__ == "__main__":
    main()
