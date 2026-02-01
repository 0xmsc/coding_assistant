import logging
from typing import Optional

from coding_assistant.actors.base import BaseActor
from coding_assistant.actors.system import ActorSystem
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import ActorMessage, DisplayMessage, Error
from coding_assistant.messaging.ui_messages import UserInputRequested, UserInputReceived
from coding_assistant.ui import UI

logger = logging.getLogger(__name__)


class UIGatewayActor(BaseActor):
    """
    Actor that bridges the messaging system with a physical UI implementation.
    """

    def __init__(self, address: str, system: ActorSystem, ui: UI):
        super().__init__(address)
        self.system = system
        self.ui = ui

    async def receive(self, envelope: Envelope[ActorMessage]) -> None:
        payload = envelope.payload

        if isinstance(payload, DisplayMessage):
            # In Phase 4, we just log/display via the UI.
            # In a real FSM system, the UI would be just one of many observers.
            logger.info(f"UI Display [{payload.role}]: {payload.content}")
            # Note: The current UI class doesn't have a 'display' method,
            # as it relies on print() or terminal output.
            # For now, we rely on the existing IO side-effects.
            pass

        elif isinstance(payload, UserInputRequested):
            logger.info(f"UI Prompt: {payload.prompt}")

            result_content = ""
            confirmed: Optional[bool] = None

            try:
                if payload.input_type == "ask":
                    result_content = await self.ui.ask(payload.prompt, payload.default)
                elif payload.input_type == "confirm":
                    confirmed = await self.ui.confirm(payload.prompt)
                    result_content = str(confirmed)
                elif payload.input_type == "prompt":
                    result_content = await self.ui.prompt()

                ok_reply: Envelope[ActorMessage] = Envelope(
                    sender=self.address,
                    recipient=envelope.sender,
                    correlation_id=envelope.correlation_id,
                    trace_id=envelope.trace_id,
                    payload=UserInputReceived(content=result_content, confirmed=confirmed),
                )
                await self.system.send(ok_reply)

            except Exception as e:
                logger.exception(f"Error getting user input: {e}")
                err_reply: Envelope[ActorMessage] = Envelope(
                    sender=self.address,
                    recipient=envelope.sender,
                    correlation_id=envelope.correlation_id,
                    trace_id=envelope.trace_id,
                    payload=Error(message=f"UI Error: {e}"),
                )
                await self.system.send(err_reply)
