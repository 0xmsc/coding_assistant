from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, AsyncIterator, cast

from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    NullProgressCallbacks,
    ToolDefinition,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from coding_assistant.runtime.events import (
    AssistantDeltaEvent,
    AssistantMessageEvent,
    CancelledEvent,
    FailedEvent,
    InputRequestedEvent,
    SessionEvent,
    ToolCallRequestedEvent,
)


_QUEUE_SENTINEL = object()


@dataclass(slots=True)
class _PendingToolCall:
    tool_call: ToolCall
    arguments: dict[str, Any]


class _SessionProgressCallbacks(NullProgressCallbacks):
    def __init__(self, emit: Any) -> None:
        self._emit = emit

    def on_content_chunk(self, chunk: str) -> None:
        self._emit(AssistantDeltaEvent(delta=chunk))


def _parse_tool_call_arguments(tool_call: ToolCall) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(tool_call.function.arguments)
    except JSONDecodeError as exc:
        return None, f"Error: Tool call arguments `{tool_call.function.arguments}` are not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "Error: Tool call arguments must decode to a JSON object."
    return parsed, None


class AssistantSession:
    def __init__(
        self,
        *,
        tools: Sequence[ToolDefinition],
        completer: Any = openai_complete,
    ) -> None:
        self._tools = list(tools)
        self._completer = completer
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

    def replace_history(self, history: Sequence[BaseMessage]) -> None:
        self._ensure_entered()
        if self._terminal:
            raise RuntimeError("Session is already terminal.")
        if not history:
            raise ValueError("Replacement history must be non-empty.")
        self._history = list(history)

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
            self._emit(InputRequestedEvent())
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

    async def submit_tool_result(self, tool_call_id: str, result: str, *, resume: bool = True) -> None:
        self._ensure_entered()
        if self._terminal:
            raise RuntimeError("Session is already terminal.")

        pending = self._require_pending_tool_call(tool_call_id)
        self._record_tool_result(pending, result)
        if resume:
            self._run_task = asyncio.create_task(self._run_loop(), name="assistant-session-run")

    async def submit_tool_error(self, tool_call_id: str, error: str) -> None:
        self._ensure_entered()
        if self._terminal:
            raise RuntimeError("Session is already terminal.")

        pending = self._require_pending_tool_call(tool_call_id)
        self._record_tool_result(pending, f"Error executing tool: {error}")
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
        for tool in self._tools:
            name = tool.name()
            if name in names:
                raise ValueError(f"Duplicate tool name: {name}")
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

    def _record_tool_result(self, pending: _PendingToolCall, content: str) -> None:
        self._append_tool_message(
            ToolMessage(
                tool_call_id=pending.tool_call.id,
                name=pending.tool_call.function.name,
                content=content,
            )
        )
        self._pending_external_tool_call = None

    def _mark_cancelled(self) -> None:
        self._waiting_for_user = False
        self._pending_external_tool_call = None
        self._pending_tool_calls.clear()
        self._terminal = True
        self._emit(CancelledEvent())

    def _should_wait_for_user(self) -> bool:
        last_message = self._history[-1]
        return last_message.role not in {"user", "tool"}

    async def _run_loop(self) -> None:
        try:
            while True:
                if self._pending_external_tool_call is not None:
                    return

                terminal = await self._drain_pending_tool_calls()
                if terminal or self._pending_external_tool_call is not None:
                    return

                progress_callbacks = _SessionProgressCallbacks(self._emit)
                assert self._active_model is not None
                completion = await self._completer(
                    self._history,
                    model=self._active_model,
                    tools=self._tools,
                    callbacks=progress_callbacks,
                )
                message = completion.message
                self._append_assistant_message(message)
                self._emit(AssistantMessageEvent(message=message))

                if message.tool_calls:
                    self._queue_tool_calls(message.tool_calls)
                    terminal = await self._drain_pending_tool_calls()
                    if terminal or self._pending_external_tool_call is not None:
                        return
                else:
                    self._waiting_for_user = True
                    self._emit(InputRequestedEvent())
                    return
        except asyncio.CancelledError:
            if not self._terminal:
                self._mark_cancelled()
            raise
        except Exception as exc:
            self._terminal = True
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

            arguments, parse_error = _parse_tool_call_arguments(tool_call)
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
            self._pending_external_tool_call = pending
            self._emit(ToolCallRequestedEvent(tool_call=pending.tool_call, arguments=pending.arguments))
            return False

        return False
