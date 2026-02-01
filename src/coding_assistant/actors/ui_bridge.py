import logging
from coding_assistant.ui import UI
from coding_assistant.actors.system import ActorSystem
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import ActorMessage
from coding_assistant.messaging.ui_messages import UserInputRequested, UserInputReceived

logger = logging.getLogger(__name__)


class ActorUIBridge(UI):
    """
    A UI implementation that redirects all calls to the Actor System.
    This allows legacy code to use the UI while the execution is actually
    handled by a UIGatewayActor.
    """

    def __init__(
        self,
        system: ActorSystem,
        recipient: str = "ui_gateway",
        sender: str = "orchestrator",
    ):
        self.system = system
        self.recipient = recipient
        self.sender = sender

    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        envelope: Envelope[ActorMessage] = Envelope(
            sender=self.sender,
            recipient=self.recipient,
            payload=UserInputRequested(prompt=prompt_text, default=default, input_type="ask"),
        )
        response = await self.system.ask(envelope)
        if isinstance(response.payload, UserInputReceived):
            return response.payload.content
        raise RuntimeError(f"Unexpected response from UI: {type(response.payload)}")

    async def confirm(self, prompt_text: str) -> bool:
        envelope: Envelope[ActorMessage] = Envelope(
            sender=self.sender,
            recipient=self.recipient,
            payload=UserInputRequested(prompt=prompt_text, input_type="confirm"),
        )
        response = await self.system.ask(envelope)
        if isinstance(response.payload, UserInputReceived):
            return response.payload.confirmed if response.payload.confirmed is not None else False
        raise RuntimeError(f"Unexpected response from UI: {type(response.payload)}")

    async def prompt(self, words: list[str] | None = None) -> str:
        envelope: Envelope[ActorMessage] = Envelope(
            sender=self.sender,
            recipient=self.recipient,
            payload=UserInputRequested(
                prompt="Prompt",  # Generic
                input_type="prompt",
            ),
        )
        response = await self.system.ask(envelope)
        if isinstance(response.payload, UserInputReceived):
            return response.payload.content
        raise RuntimeError(f"Unexpected response from UI: {type(response.payload)}")
