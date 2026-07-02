from __future__ import annotations

from pathlib import Path

from coding_assistant.manager.migrations import migrate_database
from coding_assistant.manager.store import SessionStore
from coding_assistant.manager.workspace import WorkspacePaths


def create_session_store(tmp_path: Path) -> SessionStore:
    database_path = tmp_path / "sessions.sqlite"
    migrate_database(database_path)
    return SessionStore(
        database_path=database_path,
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
