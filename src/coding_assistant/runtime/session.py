from __future__ import annotations

import asyncio
import importlib.resources
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any, AsyncIterator, Literal, cast

from coding_assistant.config import MCPServerConfig
from coding_assistant.instructions import get_instructions
from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.llm.types import AssistantMessage, BaseMessage, Tool, ToolMessage, UserMessage
from coding_assistant.runtime.engine import (
    RuntimeProgressCallbacks,
    complete_single_step,
    create_agent_start_message,
    create_chat_start_message,
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
    WaitingForUserEvent,
)
from coding_assistant.runtime.history import clear_history
from coding_assistant.runtime.persistence import HistoryStore
from coding_assistant.tool_results import CompactConversationResult, FinishTaskResult, TextResult
from coding_assistant.tools.mcp import MCPServer, get_mcp_servers_from_config, get_mcp_wrapped_tools
from coding_assistant.tools.tools import LaunchAgentSchema, RedirectToolCallTool


SessionMode = Literal["chat", "agent"]
_QUEUE_SENTINEL = object()


@dataclass(slots=True)
class SessionOptions:
    model: str
    working_directory: Path
    expert_model: str | None = None
    compact_conversation_at_tokens: int = 200_000
    mcp_server_configs: tuple[MCPServerConfig, ...] = ()
    skills_directories: tuple[str, ...] = ()
    mcp_env: tuple[str, ...] = ()
    user_instructions: tuple[str, ...] = ()
    history_store: HistoryStore | None = None
    coding_assistant_root: Path | None = None

    def resolved_expert_model(self) -> str:
        return self.expert_model or self.model


@dataclass(slots=True)
class _SharedRuntimeState:
    instructions: str
    base_tools: list[Tool]
    mcp_servers: list[MCPServer]


class _LaunchAgentTool(Tool):
    def __init__(self, execute_child: Callable[[LaunchAgentSchema], Awaitable[TextResult]]) -> None:
        self._execute_child = execute_child

    def name(self) -> str:
        return "launch_agent"

    def description(self) -> str:
        return (
            "Launch a sub-agent to work on a well-scoped task. "
            "If the sub-agent cannot proceed, it will report what information is missing."
        )

    def parameters(self) -> dict[str, Any]:
        return LaunchAgentSchema.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
        validated = LaunchAgentSchema.model_validate(parameters)
        return await self._execute_child(validated)


class AssistantSession:
    def __init__(
        self,
        *,
        options: SessionOptions,
        completer: Any = openai_complete,
        shared_state: _SharedRuntimeState | None = None,
    ) -> None:
        self.options = options
        self._completer = completer
        self._shared_state = shared_state
        self._mcp_servers_cm: Any | None = None
        self._event_queue: asyncio.Queue[SessionEvent | object] = asyncio.Queue()
        self._history: list[BaseMessage] = []
        self._mode: SessionMode | None = None
        self._active_model: str | None = None
        self._waiting_for_user = False
        self._entered = False
        self._terminal = False
        self._run_task: asyncio.Task[None] | None = None

    @property
    def instructions(self) -> str:
        return self._require_shared_state().instructions

    @property
    def mcp_servers(self) -> list[MCPServer]:
        return self._require_shared_state().mcp_servers

    @property
    def history(self) -> list[BaseMessage]:
        return list(self._history)

    @staticmethod
    def get_default_mcp_server_config(
        root_directory: Path, skills_directories: list[str], env: list[str] | None = None
    ) -> MCPServerConfig:
        import sys

        args = ["-m", "coding_assistant.mcp"]
        if skills_directories:
            args.append("--skills-directories")
            args.extend(skills_directories)
        return MCPServerConfig(
            name="coding_assistant.mcp",
            command=sys.executable,
            args=args,
            env=env or [],
        )

    async def __aenter__(self) -> "AssistantSession":
        if self._shared_state is None:
            root = (
                self.options.coding_assistant_root
                or Path(str(importlib.resources.files("coding_assistant"))).parent.resolve()
            )
            configs = [
                *self.options.mcp_server_configs,
                self.get_default_mcp_server_config(
                    root,
                    list(self.options.skills_directories),
                    env=list(self.options.mcp_env),
                ),
            ]
            self._mcp_servers_cm = get_mcp_servers_from_config(
                configs, working_directory=self.options.working_directory
            )
            servers = await self._mcp_servers_cm.__aenter__()
            base_tools = await get_mcp_wrapped_tools(servers)
            instructions = get_instructions(
                working_directory=self.options.working_directory,
                user_instructions=list(self.options.user_instructions),
                mcp_servers=servers,
            )
            self._shared_state = _SharedRuntimeState(
                instructions=instructions,
                base_tools=base_tools,
                mcp_servers=servers,
            )

        self._entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._run_task is not None and not self._run_task.done():
            await self.cancel()

        if self._mcp_servers_cm is not None:
            await self._mcp_servers_cm.__aexit__(exc_type, exc_val, exc_tb)

        self._entered = False
        self._event_queue.put_nowait(_QUEUE_SENTINEL)

    async def start(
        self,
        *,
        mode: SessionMode,
        task: str | None = None,
        history: list[BaseMessage] | None = None,
        instructions: str | None = None,
        model: str | None = None,
    ) -> None:
        self._ensure_entered()
        if self._run_task is not None:
            raise RuntimeError("Session is already running.")
        if self._mode is not None:
            raise RuntimeError("Session has already been started.")
        if mode == "agent" and not task:
            raise ValueError("Agent mode requires a task.")

        self._history = list(history or [])
        self._mode = mode
        self._active_model = model or (self.options.resolved_expert_model() if mode == "agent" else self.options.model)
        start_message = self._create_start_message(mode=mode, task=task, extra_instructions=instructions)
        self._append_user_message(UserMessage(content=start_message))

        if mode == "chat":
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

    async def cancel(self) -> None:
        if self._run_task is None:
            return

        task = self._run_task
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            if not self._terminal:
                self._terminal = True
                self._persist_history()
                self._emit(CancelledEvent())

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

    def _require_shared_state(self) -> _SharedRuntimeState:
        if self._shared_state is None:
            raise RuntimeError("Shared runtime state is not initialized.")
        return self._shared_state

    def _emit(self, event: SessionEvent) -> None:
        self._event_queue.put_nowait(event)

    def _append_user_message(self, message: UserMessage) -> None:
        self._history.append(message)

    def _append_assistant_message(self, message: AssistantMessage) -> None:
        self._history.append(message)

    def _append_tool_message(self, message: ToolMessage) -> None:
        self._history.append(message)

    def _persist_history(self) -> None:
        if self.options.history_store is not None:
            self.options.history_store.save(self._history)

    def _create_start_message(self, *, mode: SessionMode, task: str | None, extra_instructions: str | None) -> str:
        instructions = self.instructions
        if extra_instructions and extra_instructions.strip():
            instructions = f"{instructions}\n\n# Run-specific instructions\n\n{extra_instructions.strip()}"
        if mode == "chat":
            return create_chat_start_message(instructions)
        assert task is not None
        return create_agent_start_message(task=task, instructions=instructions)

    def _build_tools(self) -> list[Tool]:
        shared = self._require_shared_state()
        launch_tool = _LaunchAgentTool(self._run_child_agent)
        session_tools: list[Tool] = [launch_tool, *shared.base_tools]
        session_tools = ensure_builtin_tools(tools=session_tools, include_finish_tool=self._mode == "agent")
        redirect_tool = RedirectToolCallTool(tools=session_tools)
        return [*session_tools, redirect_tool]

    async def _run_loop(self) -> None:
        try:
            while True:
                tools = self._build_tools()
                progress_callbacks = RuntimeProgressCallbacks(self._emit)
                assert self._active_model is not None
                message, usage = await complete_single_step(
                    history=self._history,
                    model=self._active_model,
                    tools=tools,
                    progress_callbacks=progress_callbacks,
                    completer=self._completer,
                )
                self._append_assistant_message(message)
                self._emit(AssistantMessageEvent(message=message))

                if message.tool_calls:
                    terminal = await self._handle_tool_calls(message=message, tools=tools)
                    if terminal:
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
            self._terminal = True
            self._persist_history()
            self._emit(CancelledEvent())
            raise
        except Exception as exc:
            self._terminal = True
            self._persist_history()
            error = str(exc) or exc.__class__.__name__
            self._emit(FailedEvent(error=error))
        finally:
            self._run_task = None

    async def _handle_tool_calls(self, *, message: AssistantMessage, tools: list[Tool]) -> bool:
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            if not function_name:
                raise RuntimeError(f"Tool call {tool_call.id} is missing function name.")

            function_args, parse_error = parse_tool_call_arguments(tool_call)
            if parse_error is not None:
                self._append_tool_message(
                    ToolMessage(
                        tool_call_id=tool_call.id,
                        name=function_name,
                        content=parse_error,
                    )
                )
                continue

            assert function_args is not None

            try:
                result = await execute_tool_call(function_name=function_name, function_args=function_args, tools=tools)
            except Exception as exc:
                result = TextResult(content=f"Error executing tool: {exc}")

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

    async def _run_child_agent(self, request: LaunchAgentSchema) -> TextResult:
        child_options = replace(self.options, history_store=None)
        model = self.options.resolved_expert_model() if request.expert_knowledge else self.options.model
        extra_instructions = self._compose_child_instructions(request)
        child = AssistantSession(
            options=child_options,
            completer=self._completer,
            shared_state=self._require_shared_state(),
        )

        last_assistant_message: AssistantMessage | None = None
        try:
            async with child:
                await child.start(
                    mode="agent",
                    task=request.task,
                    instructions=extra_instructions,
                    model=model,
                )
                async for event in child.events():
                    if isinstance(event, AssistantMessageEvent):
                        last_assistant_message = event.message
                        continue
                    if isinstance(event, FinishedEvent):
                        return TextResult(content=event.result)
                    if isinstance(event, WaitingForUserEvent):
                        message = last_assistant_message.content if last_assistant_message is not None else None
                        detail = message or "The sub-agent needs more information to proceed."
                        return TextResult(content=f"Sub-agent needs more information: {detail}")
                    if isinstance(event, FailedEvent):
                        return TextResult(content=f"Sub-agent failed: {event.error}")
                    if isinstance(event, CancelledEvent):
                        raise asyncio.CancelledError()
        except asyncio.CancelledError:
            await child.cancel()
            raise

        return TextResult(content="Sub-agent exited without producing a result.")

    def _compose_child_instructions(self, request: LaunchAgentSchema) -> str | None:
        parts: list[str] = []
        if request.instructions:
            parts.append(request.instructions.strip())
        if request.expected_output:
            parts.append(f"Expected output:\n{request.expected_output.strip()}")
        if not parts:
            return None
        return "\n\n".join(parts)
