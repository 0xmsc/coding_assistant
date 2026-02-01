import logging
from coding_assistant.actors.base import BaseActor
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import ActorMessage

logger = logging.getLogger(__name__)


class ObserverActor(BaseActor):
    """
    A passive actor that records all messages it receives.
    Used for shadow logging and verification.
    """

    def __init__(self, address: str):
        super().__init__(address)
        self.received_messages: list[Envelope[ActorMessage]] = []

    async def receive(self, envelope: Envelope[ActorMessage]) -> None:
        self.received_messages.append(envelope)
        logger.info(
            f"[Observer] Captured message from {envelope.sender} to {envelope.recipient}: {type(envelope.payload).__name__}"
        )
