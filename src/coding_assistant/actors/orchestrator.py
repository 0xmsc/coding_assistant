import logging
import json
from enum import Enum, auto
from typing import Sequence, Optional

from coding_assistant.actors.base import BaseActor
from coding_assistant.actors.system import ActorSystem
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import (
    ActorMessage,
    StartTask,
    LLMResponse,
    ExecuteTool,
    ToolResult,
    Error,
    TaskCompleted,
)
from coding_assistant.llm.types import (
    Tool,
    ToolCall,
    ToolMessage,
    NullProgressCallbacks,
)
from coding_assistant.framework.history import append_assistant_message, append_tool_message
from coding_assistant.framework.types import AgentContext, Completer, AgentOutput
from coding_assistant.framework.results import (
    FinishTaskResult,
    result_from_dict,
)

logger = logging.getLogger(__name__)


class OrchestratorState(Enum):
    IDLE = auto()
    THINKING = auto()
    WAITING_FOR_TOOLS = auto()
    COMPLETED = auto()
    FAILED = auto()


class OrchestratorActor(BaseActor):
    """
    The main coordinator for an agent task.
    Implements a State Machine to handle the LLM loop.
    """

    def __init__(
        self,
        address: str,
        system: ActorSystem,
        context: AgentContext,
        completer: Completer,
        tools: Sequence[Tool],
        tool_worker_address: Optional[str] = None,
    ):
        super().__init__(address)
        self.system = system
        self.context = context
        self.completer = completer
        self.tools = tools
        # If no tool worker address is provided, we assume a private one
        self.tool_worker_address = tool_worker_address or f"{address}/tool_worker"

        self.state = OrchestratorState.IDLE
        self._pending_tool_calls: dict[str, ToolCall] = {}

    async def receive(self, envelope: Envelope[ActorMessage]) -> None:
        payload = envelope.payload

        if isinstance(payload, StartTask):
            await self._handle_start_task(envelope, payload)

        elif isinstance(payload, LLMResponse):
            await self._handle_llm_response(envelope, payload)

        elif isinstance(payload, ToolResult):
            await self._handle_tool_result(envelope, payload)

        elif isinstance(payload, Error):
            logger.error(f"orchestrator received error: {payload.message}")
            self.state = OrchestratorState.FAILED

    async def _handle_start_task(self, envelope: Envelope[ActorMessage], payload: StartTask) -> None:
        if self.state != OrchestratorState.IDLE:
            logger.warning("Agent already running; ignoring StartTask.")
            return

        self.state = OrchestratorState.THINKING
        await self._trigger_llm_step(envelope.trace_id)

    async def _trigger_llm_step(self, trace_id: str) -> None:
        try:
            completion = await self.completer(
                self.context.state.history,
                model=self.context.desc.model,
                tools=self.tools,
                callbacks=NullProgressCallbacks(),
            )

            # Wrap the result as if it came from a message
            msg = LLMResponse(completion=completion)
            env: Envelope[ActorMessage] = Envelope(
                sender=self.address,
                recipient=self.address,
                trace_id=trace_id,
                payload=msg,
            )
            await self.system.send(env)

        except Exception as e:
            logger.exception("LLM Step failed")
            await self.system.send(
                Envelope(
                    sender=self.address,
                    recipient=self.address,
                    trace_id=trace_id,
                    payload=Error(message=str(e)),
                )
            )

    async def _handle_llm_response(self, envelope: Envelope[ActorMessage], payload: LLMResponse) -> None:
        message = payload.completion.message

        # Update history
        append_assistant_message(
            self.context.state.history,
            context_name=self.context.desc.name,
            message=message,
        )

        if message.tool_calls:
            self.state = OrchestratorState.WAITING_FOR_TOOLS
            self._pending_tool_calls = {tc.id: tc for tc in message.tool_calls}

            for tool_call in message.tool_calls:
                # Dispatch to tool worker
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = {}

                exec_msg = ExecuteTool(tool_name=tool_call.function.name, arguments=args)
                await self.system.send(
                    Envelope(
                        sender=self.address,
                        recipient=self.tool_worker_address,
                        correlation_id=tool_call.id,
                        trace_id=envelope.trace_id,
                        payload=exec_msg,
                    )
                )
        else:
            # Handling no tool calls as a stall or completion
            self.state = OrchestratorState.IDLE

    async def _handle_tool_result(self, envelope: Envelope[ActorMessage], payload: ToolResult) -> None:
        if envelope.correlation_id not in self._pending_tool_calls:
            logger.warning(f"Received result for unknown tool call: {envelope.correlation_id}")
            return

        tool_call = self._pending_tool_calls.pop(envelope.correlation_id)

        # Convert dictionary result back to ToolResult object
        res_obj = result_from_dict(payload.result)

        # Check for task completion
        if isinstance(res_obj, FinishTaskResult):
            self.context.state.output = AgentOutput(result=res_obj.result, summary=res_obj.summary)
            self.state = OrchestratorState.COMPLETED
            await self.system.send(
                Envelope(
                    sender=self.address,
                    recipient=self.address,  # Could be sender of StartTask
                    trace_id=envelope.trace_id,
                    payload=TaskCompleted(result=res_obj.result),
                )
            )
            return

        try:
            args = json.loads(tool_call.function.arguments)
        except Exception:
            args = {}

        tool_message = ToolMessage(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            content=res_obj.content if hasattr(res_obj, "content") else str(payload.result),
        )

        append_tool_message(
            self.context.state.history,
            context_name=self.context.desc.name,
            message=tool_message,
            arguments=args,
        )

        # Check if we are finished with ALL tool calls for this step
        if not self._pending_tool_calls:
            self.state = OrchestratorState.THINKING
            await self._trigger_llm_step(envelope.trace_id)
