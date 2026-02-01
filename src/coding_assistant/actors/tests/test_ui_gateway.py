import pytest

from coding_assistant.actors.system import ActorSystem
from coding_assistant.actors.ui_gateway import UIGatewayActor
from coding_assistant.actors.ui_bridge import ActorUIBridge
from coding_assistant.ui import UI


class MockUI(UI):
    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        return f"Answer to {prompt_text}"

    async def confirm(self, prompt_text: str) -> bool:
        return True

    async def prompt(self, words: list[str] | None = None) -> str:
        return "Generic prompt response"


@pytest.mark.asyncio
async def test_ui_gateway_roundtrip() -> None:
    system = ActorSystem()
    mock_ui = MockUI()

    # Postman (System) -> Gateway (Actor) -> Physical UI (Mock)
    gateway = UIGatewayActor("ui", system, mock_ui)
    system.register(gateway)
    await gateway.start()

    # Bridge (Legacy UI interface) -> Postman (System)
    bridge = ActorUIBridge(system, recipient="ui")

    # Test ask
    result = await bridge.ask("What is your name?")
    assert result == "Answer to What is your name?"

    # Test confirm
    result_confirm = await bridge.confirm("Are you sure?")
    assert result_confirm is True

    # Test prompt
    result_prompt = await bridge.prompt()
    assert result_prompt == "Generic prompt response"

    await system.shutdown()
