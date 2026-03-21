from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, AsyncIterator, cast

from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from coding_assistant.runtime.engine import (
    RuntimeProgressCallbacks,
    complete_single_step,
    ensure_builtin_tools,
    execute_tool_call,
    normalize_tool_result,
    parse_tool_call_arguments,
)
from coding_assistant.runtime.events import (
    AssistantMessageEvent,
    CancelledEvent,
    FailedEvent,
    FinishedEvent,
    SessionEvent,
    ToolCallRequestedEvent,
    WaitingForUserEvent,
)
from coding_assistant.runtime.history import clear_history
from coding_assistant.runtime.persistence import HistoryStore
from coding_assistant.runtime.tool_spec import ToolSpec
from coding_assistant.tool_results import CompactConversationResult, FinishTaskResult


_QUEUE_SENTINEL = object()


@dataclass(slots=True)
class SessionOptions:
    compact_conversation_at_tokens: int = 200_000


@dataclass(slots=True)
class _PendingToolCall:
    tool_call: ToolCall
    arguments: dict[str, Any]


class AssistantSession:
    def __init__(
        self,
        *,
        tools: list[ToolSpec],
        options: SessionOptions | None = None,
        completer: Any = openai_complete,
        history_store: HistoryStore | None = None,
    ) -> None:
        self._tools = list(tools)
        self.options = options or SessionOptions()
        self._completer = completer
        self._history_store = history_store
        self._event_queue: asyncio.Queue[SessionEvent | object] = asyncio.Queue()
        self._history: list[BaseMessage] = []
        self._active_model: str | None = None
        self._waiting_for_user = False
        self._entered = False
        self._terminal = False
        self._run_task: asyncio.Task[None] | None = None
        self._pending_tool_calls: deque[_PendingToolCall] = deque()
        self._pending_external_tool_call: _PendingToolCall | None = None
        self._validate_tool_names()

    @property
    def history(self) -> list[BaseMessage]:
        return list(self._history)

    async def __aenter__(self) -> "AssistantSession":
        self._entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._run_task is not None and not self._run_task.done():
            await self.cancel()
        elif self._pending_external_tool_call is not None:
            await self.cancel()

        self._entered = False
        self._event_queue.put_nowait(_QUEUE_SENTINEL)

    async def start(
        self,
        *,
        history: Sequence[BaseMessage],
        model: str,
    ) -> None:
        self._ensure_entered()
        if self._run_task is not None or self._pending_external_tool_call is not None:
            raise RuntimeError("Session is already running.")
        if self._active_model is not None:
            raise RuntimeError("Session has already been started.")

        self._history = list(history)
        if not self._history:
            raise ValueError("Session start requires a non-empty history.")

        self._active_model = model
        if self._should_wait_for_user():
            self._waiting_for_user = True
            self._persist_history()
            self._emit(WaitingForUserEvent())
            return

        self._waiting_for_user = False
        self._run_task = asyncio.create_task(self._run_loop(), name="assistant-session-run")

    async def send_user_message(self, content: str | list[dict[str, Any]]) -> None:
        self._ensure_entered()
        if not self._waiting_for_user:
            raise RuntimeError("Session is not waiting for user input.")
        if self._terminal:
            raise RuntimeError("Session is already terminal.")

        self._waiting_for_user = False
        self._append_user_message(UserMessage(content=content))
        self._run_task = asyncio.create_task(self._run_loop(), name="assistant-session-run")

    async def submit_tool_result(self, tool_call_id: str, result: ToolResult | str) -> None:
        self._ensure_entered()
        if self._terminal:
            raise RuntimeError("Session is already terminal.")

        pending = self._require_pending_tool_call(tool_call_id)
        content = result if isinstance(result, str) else normalize_tool_result(result)
        self._append_tool_message(
            ToolMessage(
                tool_call_id=pending.tool_call.id,
                name=pending.tool_call.function.name,
                content=content,
            )
        )
        self._pending_external_tool_call = None
        self._run_task = asyncio.create_task(self._run_loop(), name="assistant-session-run")

    async def submit_tool_error(self, tool_call_id: str, error: str) -> None:
        self._ensure_entered()
        if self._terminal:
            raise RuntimeError("Session is already terminal.")

        pending = self._require_pending_tool_call(tool_call_id)
        self._append_tool_message(
            ToolMessage(
                tool_call_id=pending.tool_call.id,
                name=pending.tool_call.function.name,
                content=f"Error executing tool: {error}",
            )
        )
        self._pending_external_tool_call = None
        self._run_task = asyncio.create_task(self._run_loop(), name="assistant-session-run")

    async def cancel(self) -> None:
        if self._terminal:
            return

        if self._run_task is not None:
            task = self._run_task
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                if not self._terminal:
                    self._mark_cancelled()
            return

        if self._active_model is not None or self._pending_external_tool_call is not None:
            self._mark_cancelled()

    async def next_event(self) -> SessionEvent:
        item = await self._event_queue.get()
        if item is _QUEUE_SENTINEL:
            raise RuntimeError("Session is closed.")
        return cast(SessionEvent, item)

    async def events(self) -> AsyncIterator[SessionEvent]:
        while True:
            item = await self._event_queue.get()
            if item is _QUEUE_SENTINEL:
                break
            yield cast(SessionEvent, item)

    def _ensure_entered(self) -> None:
        if not self._entered:
            raise RuntimeError("Session must be entered before use.")

    def _validate_tool_names(self) -> None:
        names: set[str] = set()
        reserved_names = {"finish_task", "compact_conversation"}
        for tool in self._tools:
            name = tool.name()
            if name in names:
                raise ValueError(f"Duplicate tool name: {name}")
            if name in reserved_names:
                raise ValueError(f"Tool name '{name}' is reserved by the runtime.")
            names.add(name)

    def _require_pending_tool_call(self, tool_call_id: str) -> _PendingToolCall:
        pending = self._pending_external_tool_call
        if pending is None:
            raise RuntimeError("Session is not waiting for a tool result.")
        if pending.tool_call.id != tool_call_id:
            raise RuntimeError(f"Expected tool result for {pending.tool_call.id}, got {tool_call_id}.")
        return pending

    def _emit(self, event: SessionEvent) -> None:
        self._event_queue.put_nowait(event)

    def _append_user_message(self, message: UserMessage) -> None:
        self._history.append(message)

    def _append_assistant_message(self, message: AssistantMessage) -> None:
        self._history.append(message)

    def _append_tool_message(self, message: ToolMessage) -> None:
        self._history.append(message)

    def _persist_history(self) -> None:
        if self._history_store is not None:
            self._history_store.save(self._history)

    def _mark_cancelled(self) -> None:
        self._waiting_for_user = False
        self._pending_external_tool_call = None
        self._pending_tool_calls.clear()
        self._terminal = True
        self._persist_history()
        self._emit(CancelledEvent())

    def _should_wait_for_user(self) -> bool:
        last_message = self._history[-1]
        return last_message.role not in {"user", "tool"}

    def _build_builtin_tools(self) -> list[Tool]:
        return cast(list[Tool], ensure_builtin_tools(tools=[]))

    def _build_tool_definitions(self) -> list[ToolDefinition]:
        return [*self._tools, *self._build_builtin_tools()]

    async def _run_loop(self) -> None:
        try:
            while True:
                if self._pending_external_tool_call is not None:
                    return

                terminal = await self._drain_pending_tool_calls()
                if terminal or self._pending_external_tool_call is not None:
                    return

                tool_definitions = self._build_tool_definitions()
                progress_callbacks = RuntimeProgressCallbacks(self._emit)
                assert self._active_model is not None
                message, usage = await complete_single_step(
                    history=self._history,
                    model=self._active_model,
                    tools=tool_definitions,
                    progress_callbacks=progress_callbacks,
                    completer=self._completer,
                )
                self._append_assistant_message(message)
                self._emit(AssistantMessageEvent(message=message))

                if message.tool_calls:
                    self._queue_tool_calls(message.tool_calls)
                    terminal = await self._drain_pending_tool_calls()
                    if terminal or self._pending_external_tool_call is not None:
                        return
                else:
                    self._waiting_for_user = True
                    self._persist_history()
                    self._emit(WaitingForUserEvent())
                    return

                if usage is not None and usage.tokens > self.options.compact_conversation_at_tokens:
                    self._append_user_message(
                        UserMessage(
                            content="Your conversation history has grown too large. Compact it immediately by using the `compact_conversation` tool."
                        )
                    )
        except asyncio.CancelledError:
            if not self._terminal:
                self._mark_cancelled()
            raise
        except Exception as exc:
            self._terminal = True
            self._persist_history()
            error = str(exc) or exc.__class__.__name__
            self._emit(FailedEvent(error=error))
        finally:
            self._run_task = None

    def _queue_tool_calls(self, tool_calls: list[ToolCall]) -> None:
        if self._pending_tool_calls:
            raise RuntimeError("Cannot queue tool calls while others are still pending.")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if not function_name:
                raise RuntimeError(f"Tool call {tool_call.id} is missing function name.")

            arguments, parse_error = parse_tool_call_arguments(tool_call)
            if parse_error is not None:
                self._append_tool_message(
                    ToolMessage(
                        tool_call_id=tool_call.id,
                        name=function_name,
                        content=parse_error,
                    )
                )
                continue

            assert arguments is not None
            self._pending_tool_calls.append(_PendingToolCall(tool_call=tool_call, arguments=arguments))

    async def _drain_pending_tool_calls(self) -> bool:
        while self._pending_tool_calls:
            pending = self._pending_tool_calls.popleft()
            function_name = pending.tool_call.function.name
            if self._is_builtin_tool(function_name):
                terminal = await self._execute_builtin_tool(pending)
                if terminal:
                    self._pending_tool_calls.clear()
                    return True
                continue

            self._pending_external_tool_call = pending
            self._emit(ToolCallRequestedEvent(tool_call=pending.tool_call, arguments=pending.arguments))
            return False

        return False

    def _is_builtin_tool(self, function_name: str) -> bool:
        return function_name in {tool.name() for tool in self._build_builtin_tools()}

    async def _execute_builtin_tool(self, pending: _PendingToolCall) -> bool:
        function_name = pending.tool_call.function.name
        try:
            result = await execute_tool_call(
                function_name=function_name,
                function_args=pending.arguments,
                tools=self._build_builtin_tools(),
            )
        except Exception as exc:
            self._append_tool_message(
                ToolMessage(
                    tool_call_id=pending.tool_call.id,
                    name=function_name,
                    content=f"Error executing tool: {exc}",
                )
            )
            return False

        return self._apply_builtin_tool_result(
            tool_call=pending.tool_call,
            function_name=function_name,
            result=result,
        )

    def _apply_builtin_tool_result(self, *, tool_call: ToolCall, function_name: str, result: ToolResult) -> bool:
        if isinstance(result, FinishTaskResult):
            self._append_tool_message(
                ToolMessage(
                    tool_call_id=tool_call.id,
                    name=function_name,
                    content="Task finished.",
                )
            )
            self._terminal = True
            self._persist_history()
            self._emit(FinishedEvent(result=result.result, summary=result.summary))
            return True

        if isinstance(result, CompactConversationResult):
            clear_history(self._history)
            self._append_user_message(
                UserMessage(
                    content=(
                        "A summary of your conversation with the client until now:\n\n"
                        f"{result.summary}\n\nPlease continue your work."
                    )
                )
            )
            tool_summary = "Conversation compacted and history reset."
        else:
            tool_summary = normalize_tool_result(result)

        self._append_tool_message(
            ToolMessage(
                tool_call_id=tool_call.id,
                name=function_name,
                content=tool_summary,
            )
        )
        return False
