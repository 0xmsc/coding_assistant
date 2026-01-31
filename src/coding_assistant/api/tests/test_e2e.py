import asyncio
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from coding_assistant.api.server import create_app
from coding_assistant.api.manager import SessionManager
from coding_assistant.config import Config
from coding_assistant.llm.types import AssistantMessage
from coding_assistant.session import Session


class MockSession(Session):
    """A minimal session that doesn't boot MCP or Sandbox."""

    async def __aenter__(self) -> "MockSession":
        self.tools = []
        self.instructions = "Mock instructions"
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def run_agent(self, task: str, history: Any = None) -> Any:
        self.callbacks.on_status_message(f"Starting task: {task}")
        # Use a mock response directy for the test
        msg = AssistantMessage(content="This is an E2E mock response.")
        self.callbacks.on_status_message(f"Final Result: {msg.content}")
        return msg


@pytest.fixture
def api_manager() -> SessionManager:
    config = Config(model="gpt-4o", expert_model="gpt-4o", compact_conversation_at_tokens=200000)
    return SessionManager(config=config, coding_assistant_root=Path("/tmp"))


@pytest.fixture
def client(api_manager: SessionManager) -> TestClient:
    app = create_app(api_manager)
    return TestClient(app)


async def mock_completer(*args: Any, **kwargs: Any) -> AssistantMessage:
    return AssistantMessage(content="This is an E2E mock response.")


@pytest.mark.asyncio
async def test_api_e2e_logic(client: TestClient, api_manager: SessionManager) -> None:
    # Use a mock session creator in the manager
    original_create = api_manager.create_session

    def mocked_create_session(*args: Any, **kwargs: Any) -> Any:
        active = original_create(*args, **kwargs)
        # Replace the real session with our lightweight mock
        active.session = MockSession(
            config=api_manager.config,
            ui=active.ui,
            callbacks=active.session.callbacks,
            working_directory=kwargs.get("working_directory", Path("/tmp")),
            coding_assistant_root=api_manager.coding_assistant_root,
            mcp_server_configs=[],
        )
        return active

    with patch.object(api_manager, "create_session", side_effect=mocked_create_session):
        with patch("coding_assistant.llm.openai.complete", new=mock_completer):
            with client.websocket_connect("/ws/test-e2e") as websocket:
                # Start
                websocket.send_json({"payload": {"type": "start", "task": "E2E Task"}})

                # Check for our mock response in messages
                found = False
                for _ in range(20):
                    try:
                        data = websocket.receive_json()
                        msg = str(data.get("payload", {}).get("message", ""))
                        if "This is an E2E mock response" in msg:
                            found = True
                            break
                    except Exception:
                        await asyncio.sleep(0.01)

                assert found, "Did not receive mock response over WebSocket"
