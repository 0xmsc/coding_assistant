import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock
from coding_assistant.api.manager import SessionManager
from coding_assistant.config import Config

@pytest.mark.asyncio
async def test_session_manager_lifecycle():
    config = Config(
        model="gpt-4o", 
        expert_model="gpt-4o", 
        compact_conversation_at_tokens=200000
    )
    manager = SessionManager(config=config, coding_assistant_root=Path("/tmp"))
    
    mock_ws = MagicMock()
    session_id = "test-session"
    working_dir = Path("/tmp/work")
    
    # Create
    active = manager.create_session(session_id, mock_ws, working_dir)
    assert session_id in manager.active_sessions
    assert manager.get_session(session_id) == active
    
    # Cleanup logic (mocking task)
    active.task = MagicMock()
    
    await manager.cleanup_session(session_id)
    assert session_id not in manager.active_sessions
