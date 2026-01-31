import asyncio
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

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
        # Use our injected completer

        completion = await self.completer([], model="mock", tools=[], callbacks=self.callbacks)
        self.callbacks.on_status_message(f"Final Result: {completion.message.content}")
        return completion.message


@pytest.fixture
def mock_completer() -> Any:
    async def _mock_completer(messages: Any, *, model: str, tools: Any, callbacks: Any) -> Any:
        from coding_assistant.llm.types import Completion

        return Completion(message=AssistantMessage(content="This is an E2E mock response."))

    return _mock_completer


@pytest.fixture
def mock_sandbox() -> Any:
    return AsyncMock()


@pytest.fixture
def api_manager(mock_completer: Any) -> SessionManager:
    config = Config(model="gpt-4o", expert_model="gpt-4o", compact_conversation_at_tokens=200000)
    return SessionManager(
        config=config,
        coding_assistant_root=Path("/tmp"),
        completer=mock_completer,
        session_factory=MockSession,
    )


@pytest.fixture
def client(api_manager: SessionManager) -> TestClient:
    app = create_app(api_manager)
    return TestClient(app)


@pytest.mark.asyncio
async def test_api_e2e_logic(client: TestClient, api_manager: SessionManager) -> None:
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
