import pytest
import asyncio
from coding_assistant.actors.base import BaseActor
from coding_assistant.actors.system import ActorSystem
from coding_assistant.actors.observer import ObserverActor
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import DisplayMessage, StartTask, ActorMessage


class EchoActor(BaseActor):
    def __init__(self, address: str, system: "ActorSystem"):
        super().__init__(address)
        self.system = system

    async def receive(self, envelope: Envelope[ActorMessage]) -> None:
        if isinstance(envelope.payload, StartTask):
            # We explicitly cast or use type checking because ActorMessage is a Union
            payload: StartTask = envelope.payload
            reply: Envelope[ActorMessage] = Envelope(
                sender=self.address,
                recipient=envelope.sender,
                correlation_id=envelope.correlation_id,
                payload=DisplayMessage(content=f"Echo: {payload.prompt}"),
            )
            await self.system.send(reply)


@pytest.mark.asyncio
async def test_actor_registration_and_dispatch() -> None:
    system = ActorSystem()
    observer = ObserverActor("observer")
    system.register(observer)
    await observer.start()

    env: Envelope[ActorMessage] = Envelope(sender="test", recipient="observer", payload=DisplayMessage(content="Hello"))

    await system.send(env)
    await asyncio.sleep(0.1)  # Give it a moment to process

    assert len(observer.received_messages) == 1
    msg_payload = observer.received_messages[0].payload
    if isinstance(msg_payload, DisplayMessage):
        assert msg_payload.content == "Hello"
    else:
        pytest.fail(f"Expected DisplayMessage, got {type(msg_payload)}")

    await system.shutdown()


@pytest.mark.asyncio
async def test_actor_ask() -> None:
    system = ActorSystem()
    echo = EchoActor("echo", system)
    system.register(echo)
    await echo.start()

    env: Envelope[ActorMessage] = Envelope(sender="user", recipient="echo", payload=StartTask(prompt="ping"))

    # This should wait for the EchoActor to send a message back with the same correlation_id
    response = await system.ask(env, timeout=1.0)

    assert response.sender == "echo"
    res_payload = response.payload
    if isinstance(res_payload, DisplayMessage):
        assert "Echo: ping" in res_payload.content
    else:
        pytest.fail(f"Expected DisplayMessage, got {type(res_payload)}")

    await system.shutdown()
