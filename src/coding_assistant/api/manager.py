import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from coding_assistant.api.bridge import WebSocketProgressCallbacks, WebSocketUI
from coding_assistant.config import Config
from coding_assistant.session import Session

logger = logging.getLogger(__name__)


class ActiveSession:
    def __init__(self, session: Session, ui: WebSocketUI, response_queue: "asyncio.Queue[Any]"):
        self.session = session
        self.ui = ui
        self.response_queue = response_queue
        self.task: Optional[asyncio.Task[Any]] = None


class SessionManager:
    def __init__(
        self,
        config: Config,
        coding_assistant_root: Path,
        completer: Optional[Any] = None,
        session_factory: Any = Session,
    ):
        self.config = config
        self.coding_assistant_root = coding_assistant_root
        self.active_sessions: Dict[str, ActiveSession] = {}
        self.completer = completer
        self.session_factory = session_factory

    def create_session(self, session_id: str, websocket: Any, working_directory: Path) -> ActiveSession:
        response_queue: asyncio.Queue[Any] = asyncio.Queue()
        ui = WebSocketUI(websocket, response_queue)
        callbacks = WebSocketProgressCallbacks(websocket)

        session_kwargs: Dict[str, Any] = {
            "config": self.config,
            "ui": ui,
            "callbacks": callbacks,
            "working_directory": working_directory,
            "coding_assistant_root": self.coding_assistant_root,
            "mcp_server_configs": [],
        }

        if self.completer:
            session_kwargs["completer"] = self.completer

        session = self.session_factory(**session_kwargs)

        active = ActiveSession(session, ui, response_queue)
        self.active_sessions[session_id] = active
        return active

    async def cleanup_session(self, session_id: str) -> None:
        if session_id in self.active_sessions:
            active = self.active_sessions[session_id]
            if active.task:
                active.task.cancel()

            del self.active_sessions[session_id]

    def get_session(self, session_id: str) -> Optional[ActiveSession]:
        return self.active_sessions.get(session_id)
