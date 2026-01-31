import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from coding_assistant.api.bridge import WebSocketProgressCallbacks, WebSocketUI
from coding_assistant.api.models import AnswerResponse, ConfirmationResponse
from coding_assistant.config import Config
from coding_assistant.session import Session

logger = logging.getLogger(__name__)

class ActiveSession:
    def __init__(self, session: Session, ui: WebSocketUI, response_queue: asyncio.Queue):
        self.session = session
        self.ui = ui
        self.response_queue = response_queue
        self.task: Optional[asyncio.Task] = None

class SessionManager:
    def __init__(self, config: Config, coding_assistant_root: Path):
        self.config = config
        self.coding_assistant_root = coding_assistant_root
        self.active_sessions: Dict[str, ActiveSession] = {}

    def create_session(
        self, 
        session_id: str, 
        websocket: Any, 
        working_directory: Path
    ) -> ActiveSession:
        response_queue = asyncio.Queue()
        ui = WebSocketUI(websocket, response_queue)
        callbacks = WebSocketProgressCallbacks(websocket)
        
        # Note: We can expand this to include custom instructions, etc.
        session = Session(
            config=self.config,
            ui=ui,
            callbacks=callbacks,
            working_directory=working_directory,
            coding_assistant_root=self.coding_assistant_root,
            mcp_server_configs=[], # Standard configs added by Session.__aenter__
        )
        
        active = ActiveSession(session, ui, response_queue)
        self.active_sessions[session_id] = active
        return active

    async def cleanup_session(self, session_id: str):
        if session_id in self.active_sessions:
            active = self.active_sessions[session_id]
            if active.task:
                active.task.cancel()
            
            # Clean up the session context (MCP servers, etc.)
            # This requires session.__aenter__ to have been called
            try:
                # Basic cleanup if needed
                pass
            except Exception as e:
                logger.error(f"Error during session cleanup: {e}")
            
            del self.active_sessions[session_id]

    def get_session(self, session_id: str) -> Optional[ActiveSession]:
        return self.active_sessions.get(session_id)
