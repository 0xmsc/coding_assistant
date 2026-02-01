import logging
from typing import Sequence

from coding_assistant.actors.base import BaseActor
from coding_assistant.actors.system import ActorSystem
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import (
    ActorMessage,
    ExecuteTool,
    ToolResult as ToolResultMessage,
)
from coding_assistant.llm.types import Tool

logger = logging.getLogger(__name__)


class ToolWorkerActor(BaseActor):
    """
    Actor responsible for executing tools.
    """

    def __init__(self, address: str, system: ActorSystem, tools: Sequence[Tool]):
        super().__init__(address)
        self.system = system
        self.tools = tools

    async def receive(self, envelope: Envelope[ActorMessage]) -> None:
        if isinstance(envelope.payload, ExecuteTool):
            payload: ExecuteTool = envelope.payload
            tool_name = payload.tool_name
            args = payload.arguments

            logger.info(f"[{self.address}] Executing tool: {tool_name}")

            try:
                # Find the tool
                target_tool = None
                for tool in self.tools:
                    if tool.name() == tool_name:
                        target_tool = tool
                        break

                if not target_tool:
                    raise ValueError(f"Tool {tool_name} not found.")

                # Execute it
                result = await target_tool.execute(args)

                # Send back the result
                # Note: We use to_dict() for the message payload to ensure serialization
                ok_reply: Envelope[ActorMessage] = Envelope(
                    sender=self.address,
                    recipient=envelope.sender,
                    correlation_id=envelope.correlation_id,
                    trace_id=envelope.trace_id,
                    payload=ToolResultMessage(result=result.to_dict(), is_error=False),
                )
                await self.system.send(ok_reply)

            except Exception as e:
                logger.exception(f"Error executing tool {tool_name}: {e}")
                err_reply: Envelope[ActorMessage] = Envelope(
                    sender=self.address,
                    recipient=envelope.sender,
                    correlation_id=envelope.correlation_id,
                    trace_id=envelope.trace_id,
                    payload=ToolResultMessage(result={"content": f"Error executing tool: {e}"}, is_error=True),
                )
                await self.system.send(err_reply)
