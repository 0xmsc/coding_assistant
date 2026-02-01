import logging
import json
import asyncio
from enum import auto, Enum
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
    DisplayMessage,
)
from coding_assistant.messaging.ui_messages import UserInputRequested, UserInputReceived
from coding_assistant.llm.types import (
    Tool,
    ToolCall,
    ToolMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    Usage,
    UserMessage,
)
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.history import (
    append_assistant_message,
    append_tool_message,
    append_user_message,
    clear_history,
)
from coding_assistant.framework.types import AgentContext, Completer, AgentOutput
from coding_assistant.framework.results import (
    FinishTaskResult,
    result_from_dict,
    CompactConversationResult,
)

logger = logging.getLogger(__name__)


class OrchestratorState(Enum):
    IDLE = auto()
    THINKING = auto()
    WAITING_FOR_TOOLS = auto()
    WAITING_FOR_USER = auto()
    COMPLETED = auto()
    FAILED = auto()


class OrchestratorActor(BaseActor):
    """
    The main coordinator for an agent task.
    Implements a State Machine to handle both autonomous and chat modes.
    """

    def __init__(
        self,
        address: str,
        system: ActorSystem,
        context: AgentContext,
        completer: Completer,
        tools: Sequence[Tool],
        is_chat_mode: bool = False,
        instructions: Optional[str] = None,
        tool_worker_address: Optional[str] = None,
        ui_gateway_address: str = "ui_gateway",
        compact_conversation_at_tokens: int = 200_000,
        progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
        tool_callbacks: ToolCallbacks = NullToolCallbacks(),
    ):
        super().__init__(address)
        self.system = system
        self.context = context
        self.completer = completer
        self.tools = list(tools)
        self.is_chat_mode = is_chat_mode
        self.instructions = instructions
        self.ui_gateway_address = ui_gateway_address
        self.compact_conversation_at_tokens = compact_conversation_at_tokens
        self.callbacks = progress_callbacks
        self.tool_callbacks = tool_callbacks

        # Signal for external observers to know when we change state
        self.state_event = asyncio.Event()

        # If no tool worker address is provided, we assume a private one
        self.tool_worker_address = tool_worker_address or f"{address}/tool_worker"

        self.state = OrchestratorState.IDLE
        self._pending_tool_calls: dict[str, ToolCall] = {}
        self.usage = Usage(tokens=0, cost=0.0)
        self.last_exception: Optional[BaseException] = None

    def _set_state(self, new_state: OrchestratorState) -> None:
        self.state = new_state
        self.state_event.set()
        self.state_event.clear()

    async def stop(self) -> None:
        await super().stop()
        self._set_state(OrchestratorState.COMPLETED)

    async def receive(self, envelope: Envelope[ActorMessage]) -> None:
        payload = envelope.payload

        if isinstance(payload, StartTask):
            await self._handle_start_task(envelope, payload)

        elif isinstance(payload, LLMResponse):
            await self._handle_llm_response(envelope, payload)

        elif isinstance(payload, ToolResult):
            await self._handle_tool_result(envelope, payload)

        elif isinstance(payload, UserInputReceived):
            await self._handle_user_input(envelope, payload)

        elif isinstance(payload, Error):
            await self._handle_error(envelope, payload)

    async def _handle_error(self, envelope: Envelope, payload: Error) -> None:
        logger.error(f"Orchestrator {self.address} received error: {payload.message}")
        
        if envelope.correlation_id in self._pending_tool_calls:
            # Error happened during tool execution.
            # Append error to history and CONTINUE (legacy behavior)
            tool_call = self._pending_tool_calls.pop(envelope.correlation_id)
            
            append_tool_message(
                self.context.state.history,
                callbacks=self.callbacks,
                context_name=self.context.desc.name,
                message=ToolMessage(
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    content=f"Error executing tool: {payload.message}",
                ),
                arguments={}
            )
            
            if not self._pending_tool_calls:
                self._set_state(OrchestratorState.THINKING)
                await self._trigger_llm_step(envelope.trace_id)
        else:
            if not self.is_chat_mode:
                self._set_state(OrchestratorState.FAILED)
            else:
                self._set_state(OrchestratorState.WAITING_FOR_USER)

    async def _handle_start_task(self, envelope: Envelope[ActorMessage], payload: StartTask) -> None:
        if self.state != OrchestratorState.IDLE:
            return

        # Initialize history if empty
        if not self.context.state.history:
            if self.is_chat_mode:
                from coding_assistant.framework.chat import _create_chat_start_message
                start_msg = _create_chat_start_message(self.instructions)
            else:
                from coding_assistant.framework.agent import _create_start_message
                start_msg = _create_start_message(desc=self.context.desc)

            append_user_message(
                self.context.state.history,
                callbacks=self.callbacks,
                context_name=self.context.desc.name,
                message=UserMessage(content=start_msg),
            )

        if self.is_chat_mode:
            self._set_state(OrchestratorState.WAITING_FOR_USER)
            await self._request_user_input(envelope.trace_id)
        else:
            self._set_state(OrchestratorState.THINKING)
            await self._trigger_llm_step(envelope.trace_id)

    async def _trigger_llm_step(self, trace_id: str) -> None:
        try:
            if self.usage.tokens > self.compact_conversation_at_tokens:
                append_user_message(
                    self.context.state.history,
                    callbacks=self.callbacks,
                    context_name=self.context.desc.name,
                    message=UserMessage(
                        content="Your conversation history has grown too large. Compact it immediately by using the `compact_conversation` tool."
                    ),
                )

            completion = await self.completer(
                self.context.state.history,
                model=self.context.desc.model,
                tools=self.tools,
                callbacks=self.callbacks,
            )

            if completion.usage:
                self.usage = Usage(
                    tokens=completion.usage.tokens,
                    cost=self.usage.cost + completion.usage.cost,
                )

            await self.system.send(
                Envelope(
                    sender=self.address,
                    recipient=self.address,
                    trace_id=trace_id,
                    payload=LLMResponse(completion=completion),
                )
            )

        except (AssertionError, KeyboardInterrupt) as e:
            self.last_exception = e
            self._set_state(OrchestratorState.FAILED)
            raise e
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

        append_assistant_message(
            self.context.state.history,
            callbacks=self.callbacks,
            context_name=self.context.desc.name,
            message=message,
        )

        if message.tool_calls:
            self._set_state(OrchestratorState.WAITING_FOR_TOOLS)
            self._pending_tool_calls = {tc.id: tc for tc in message.tool_calls}

            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = {}

                # Tool start callback
                self.callbacks.on_tool_start(self.context.desc.name, tool_call, args)

                await self.system.send(
                    Envelope(
                        sender=self.address,
                        recipient=self.tool_worker_address,
                        correlation_id=tool_call.id,
                        trace_id=envelope.trace_id,
                        payload=ExecuteTool(tool_name=tool_call.function.name, arguments=args),
                    )
                )
        else:
            if self.is_chat_mode:
                self._set_state(OrchestratorState.WAITING_FOR_USER)
                await self._request_user_input(envelope.trace_id)
            else:
                append_user_message(
                    self.context.state.history,
                    callbacks=self.callbacks,
                    context_name=self.context.desc.name,
                    message=UserMessage(
                        content="I detected a step from you without any tool calls. This is not allowed. If you are done with your task, please call the `finish_task` tool to signal that you are done. Otherwise, continue your work."
                    ),
                )
                self._set_state(OrchestratorState.THINKING)
                await self._trigger_llm_step(envelope.trace_id)

    async def _handle_tool_result(self, envelope: Envelope[ActorMessage], payload: ToolResult) -> None:
        if envelope.correlation_id not in self._pending_tool_calls:
            return

        tool_call = self._pending_tool_calls.pop(envelope.correlation_id)
        res_obj = result_from_dict(payload.result)

        if isinstance(res_obj, FinishTaskResult):
            from coding_assistant.framework.agent import _handle_finish_task_result
            _handle_finish_task_result(res_obj, state=self.context.state)
            
            try:
                args = json.loads(tool_call.function.arguments)
            except Exception:
                args = {}
            append_tool_message(
                self.context.state.history,
                callbacks=self.callbacks,
                context_name=self.context.desc.name,
                message=ToolMessage(tool_call_id=tool_call.id, name=tool_call.function.name, content="Agent output set."),
                arguments=args
            )

            self._set_state(OrchestratorState.COMPLETED)
            return

        if isinstance(res_obj, CompactConversationResult):
            from coding_assistant.framework.agent import _handle_compact_conversation_result
            _handle_compact_conversation_result(res_obj, desc=self.context.desc, state=self.context.state, progress_callbacks=self.callbacks, force=not self.is_chat_mode)
        else:
            try:
                args = json.loads(tool_call.function.arguments)
            except Exception:
                args = {}

            content = res_obj.content if hasattr(res_obj, "content") else str(payload.result)
            append_tool_message(
                self.context.state.history,
                callbacks=self.callbacks,
                context_name=self.context.desc.name,
                message=ToolMessage(
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    content=content,
                ),
                arguments=args,
            )

        if not self._pending_tool_calls:
            self._set_state(OrchestratorState.THINKING)
            await self._trigger_llm_step(envelope.trace_id)

    async def _request_user_input(self, trace_id: str) -> None:
        if self.is_chat_mode:
            status = f"💰 {self.usage.tokens} tokens • ${self.usage.cost:.2f}"
            self.callbacks.on_status_message(status)

        await self.system.send(
            Envelope(
                sender=self.address,
                recipient=self.ui_gateway_address,
                trace_id=trace_id,
                payload=UserInputRequested(prompt="> ", input_type="prompt"),
            )
        )

    async def _handle_user_input(self, envelope: Envelope[ActorMessage], payload: UserInputReceived) -> None:
        if self.state != OrchestratorState.WAITING_FOR_USER:
            return

        text = payload.content.strip()

        if text.startswith("/"):
            if text == "/exit":
                self._set_state(OrchestratorState.COMPLETED)
                return
            if text == "/clear":
                clear_history(self.context.state.history)
                from coding_assistant.framework.chat import _create_chat_start_message
                start_msg = _create_chat_start_message(self.instructions)
                append_user_message(self.context.state.history, callbacks=self.callbacks, context_name=self.context.desc.name, message=UserMessage(content=start_msg))
                await self._request_user_input(envelope.trace_id)
                return
            if text == "/compact":
                append_user_message(
                    self.context.state.history,
                    callbacks=self.callbacks,
                    context_name=self.context.desc.name,
                    message=UserMessage(
                        content="Immediately compact our conversation so far by using the `compact_conversation` tool."
                    ),
                    force=True,
                )
                self._set_state(OrchestratorState.THINKING)
                await self._trigger_llm_step(envelope.trace_id)
                return

        append_user_message(
            self.context.state.history,
            callbacks=self.callbacks,
            context_name=self.context.desc.name,
            message=UserMessage(content=payload.content),
        )
        self._set_state(OrchestratorState.THINKING)
        await self._trigger_llm_step(envelope.trace_id)
