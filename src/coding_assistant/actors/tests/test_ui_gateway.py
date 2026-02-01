import pytest

from coding_assistant.actors.system import ActorSystem
from coding_assistant.actors.ui_gateway import UIGatewayActor
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

    from coding_assistant.messaging.envelopes import Envelope
    from coding_assistant.messaging.ui_messages import UserInputRequested, UserInputReceived
    from coding_assistant.messaging.messages import ActorMessage

    # Test ask
    envelope: Envelope[ActorMessage] = Envelope(
        sender="orchestrator",
        recipient="ui",
        payload=UserInputRequested(prompt="What is your name?", input_type="ask"),
    )
    response = await system.ask(envelope)
    assert isinstance(response.payload, UserInputReceived)
    assert response.payload.content == "Answer to What is your name?"

    # Test confirm
    envelope2: Envelope[ActorMessage] = Envelope(
        sender="orchestrator",
        recipient="ui",
        payload=UserInputRequested(prompt="Are you sure?", input_type="confirm"),
    )
    response2 = await system.ask(envelope2)
    assert isinstance(response2.payload, UserInputReceived)
    assert response2.payload.confirmed is True

    await system.shutdown()
