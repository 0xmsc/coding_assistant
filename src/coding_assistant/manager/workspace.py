from __future__ import annotations

from pathlib import Path


class WorkspaceMissingError(RuntimeError):
    pass


class WorkspacePaths:
    def __init__(self, *, root: Path) -> None:
        self.root = root

    def path_for_session(self, session_id: str) -> Path:
        return self.root / session_id

    def create_for_session(self, session_id: str) -> Path:
        path = self.path_for_session(session_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def require_for_session(self, session_id: str) -> Path:
        path = self.path_for_session(session_id)
        if not path.is_dir():
            raise WorkspaceMissingError(f"Workspace is missing for session {session_id}.")
        return path
