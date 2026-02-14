import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from coding_assistant.api.server import create_app
from coding_assistant.api.manager import SessionManager
from coding_assistant.config import Config


@pytest.fixture
def client() -> TestClient:
    config = Config(model="gpt-4o", expert_model="gpt-4o", compact_conversation_at_tokens=200000)
    manager = SessionManager(config=config, coding_assistant_root=Path("/tmp"))
    app = create_app(manager)
    return TestClient(app)


def test_create_session_endpoint(client: TestClient) -> None:
    response = client.post("/sessions", json={"session_id": "api-test", "working_directory": "/tmp"})
    assert response.status_code == 200
    assert response.json()["session_id"] == "api-test"


def test_websocket_flow(client: TestClient) -> None:
    with client.websocket_connect("/ws/ws-test") as websocket:
        websocket.send_json({"payload": {"type": "start", "task": "echo hello"}})
