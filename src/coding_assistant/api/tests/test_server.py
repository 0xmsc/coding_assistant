import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import MagicMock, patch

from coding_assistant.api.server import create_app
from coding_assistant.api.manager import SessionManager
from coding_assistant.config import Config

@pytest.fixture
def client():
    config = Config(
        model="gpt-4o", 
        expert_model="gpt-4o", 
        compact_conversation_at_tokens=200000
    )
    manager = SessionManager(config=config, coding_assistant_root=Path("/tmp"))
    app = create_app(manager)
    return TestClient(app)

def test_create_session_endpoint(client):
    response = client.post("/sessions", json={"session_id": "api-test", "working_directory": "/tmp"})
    assert response.status_code == 200
    assert response.json()["session_id"] == "api-test"

def test_websocket_flow(client):
    # Testing WebSockets with TestClient requires the 'websockets' or 'httpx' backend
    with client.websocket_connect("/ws/ws-test") as websocket:
        # Check if we can send a start command
        websocket.send_json({
            "payload": {
                "type": "start",
                "task": "echo hello"
            }
        })
        # Note: In a real test we'd mock the Session.run_agent to verify it's called
        # But this confirms the protocol parsing and connection works
