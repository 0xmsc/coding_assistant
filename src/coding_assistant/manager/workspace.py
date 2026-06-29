from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class WorkspaceMissingError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionPaths:
    root: Path
    workspace: Path
    attachments: Path


class WorkspacePaths:
    def __init__(self, *, root: Path) -> None:
        self.root = root

    def path_for_session(self, session_id: str) -> Path:
        return self.root / session_id

    def paths_for_session(self, session_id: str) -> SessionPaths:
        session_root = self.path_for_session(session_id)
        return SessionPaths(
            root=session_root,
            workspace=session_root / "workspace",
            attachments=session_root / "attachments",
        )

    def create_for_session(self, session_id: str) -> SessionPaths:
        paths = self.paths_for_session(session_id)
        paths.workspace.mkdir(parents=True, exist_ok=True)
        paths.attachments.mkdir(parents=True, exist_ok=True)
        return paths

    def require_for_session(self, session_id: str) -> SessionPaths:
        paths = self.paths_for_session(session_id)
        if not paths.workspace.is_dir() or not paths.attachments.is_dir():
            raise WorkspaceMissingError(f"Workspace is missing for session {session_id}.")
        return paths
