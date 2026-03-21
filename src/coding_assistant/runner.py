from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, AsyncIterator, Literal

from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.llm.types import AssistantMessage, BaseMessage, SystemMessage, Tool, UserMessage
from coding_assistant.runtime import (
    AssistantMessageEvent,
    AssistantSession,
    CancelledEvent,
    CompletedEvent,
    FailedEvent,
    InputRequestedEvent,
    SessionEvent,
    SessionOptions,
    ToolCallRequestedEvent,
    ToolSpec,
)
from coding_assistant.runtime.persistence import HistoryStore
from coding_assistant.tool_policy import NullToolPolicy, ToolPolicy
from coding_assistant.tool_results import TextResult
from coding_assistant.tools.managed import (
    CompactConversationSchema,
    CompactConversationTool,
    FinishTaskSchema,
    FinishTaskTool,
    LaunchAgentSchema,
    LaunchAgentTool,
    RedirectToolCallTool,
)


CHAT_SYSTEM_PROMPT_TEMPLATE = """
## General

- You are an agent.
- You are in chat mode.
  - Use tools only when they materially advance the work.
  - When you want the user to reply, write a normal assistant message without tool calls.

{instructions_section}
""".strip()


AGENT_SYSTEM_PROMPT_TEMPLATE = """
## General

- You are an agent.
- You are working in agent mode.
  - Use tools when they materially advance the work.
  - When you are done, call the `finish_task` tool.
  - When you need more information, write a normal assistant message without tool calls.

{instructions_section}
""".strip()


class ManagedSession:
    def __init__(
        self,
        *,
        instructions: str,
        tools: list[Tool],
        model: str,
        expert_model: str | None = None,
        runtime_options: SessionOptions | None = None,
        completer: Any = openai_complete,
        history_store: HistoryStore | None = None,
        tool_policy: ToolPolicy | None = None,
        include_launch_agent: bool = True,
        include_redirect_tool_call: bool = True,
    ) -> None:
        self._instructions = instructions
        self._base_tools = list(tools)
        self._tool_policy = tool_policy or NullToolPolicy()
        self._model = model
        self._expert_model = expert_model
        self._runtime_options = runtime_options or SessionOptions()
        self._completer = completer
        self._history_store = history_store
        self._tools = self._build_tools(
            include_launch_agent=include_launch_agent,
            include_redirect_tool_call=include_redirect_tool_call,
        )
        self._tools_by_name = {tool.name(): tool for tool in self._tools}
        self._session = AssistantSession(
            tools=[ToolSpec.from_definition(tool) for tool in self._tools],
            options=self._runtime_options,
            completer=completer,
            history_store=history_store,
        )

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def history(self) -> list[Any]:
        return self._session.history

    async def __aenter__(self) -> "ManagedSession":
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._session.__aexit__(exc_type, exc_val, exc_tb)

    async def start(
        self,
        *,
        mode: Literal["chat", "agent"],
        task: str | None = None,
        history: list[BaseMessage] | None = None,
        instructions: str | None = None,
        model: str | None = None,
    ) -> None:
        resolved_model = model or self._resolve_model(mode=mode)
        start_history = self._build_start_history(
            mode=mode,
            task=task,
            history=history,
            instructions=instructions,
        )
        await self._session.start(
            history=start_history,
            model=resolved_model,
        )

    async def send_user_message(self, content: str | list[dict[str, Any]]) -> None:
        await self._session.send_user_message(content)

    async def cancel(self) -> None:
        await self._session.cancel()

    async def next_event(self) -> SessionEvent:
        while True:
            event = await self._session.next_event()
            if isinstance(event, ToolCallRequestedEvent):
                await self._handle_tool_call_requested(event)
                continue
            return event

    async def events(self) -> AsyncIterator[SessionEvent]:
        async for event in self._session.events():
            if isinstance(event, ToolCallRequestedEvent):
                await self._handle_tool_call_requested(event)
                continue
            yield event

    def _build_tools(self, *, include_launch_agent: bool, include_redirect_tool_call: bool) -> list[Tool]:
        managed_tools: list[Tool] = [
            FinishTaskTool(),
            CompactConversationTool(),
        ]
        if include_launch_agent:
            managed_tools.append(LaunchAgentTool(execute_child=self._run_child_agent))

        all_tools = [*managed_tools, *self._base_tools]
        if include_redirect_tool_call:
            all_tools.append(RedirectToolCallTool(tools=all_tools))

        names: set[str] = set()
        for tool in all_tools:
            name = tool.name()
            if name in names:
                raise ValueError(f"Duplicate tool name: {name}")
            names.add(name)

        return all_tools

    async def _handle_tool_call_requested(self, event: ToolCallRequestedEvent) -> None:
        tool_name = event.tool_call.function.name
        tool = self._tools_by_name.get(tool_name)
        if tool is None:
            await self._session.submit_tool_error(event.tool_call.id, f"Tool '{tool_name}' not found.")
            return

        if policy_result := await self._tool_policy.before_tool_execution(
            tool_call_id=event.tool_call.id,
            tool_name=tool_name,
            arguments=event.arguments,
        ):
            await self._session.submit_tool_result(event.tool_call.id, policy_result)
            return

        if tool_name == "finish_task":
            finish_request = FinishTaskSchema.model_validate(event.arguments)
            await self._session.submit_tool_result(event.tool_call.id, "Task finished.", resume=False)
            await self._session.complete(result=finish_request.result, summary=finish_request.summary)
            return

        if tool_name == "compact_conversation":
            compact_request = CompactConversationSchema.model_validate(event.arguments)
            self._session.compact_history(compact_request.summary)
            await self._session.submit_tool_result(event.tool_call.id, "Conversation compacted and history reset.")
            return

        try:
            result = await tool.execute(event.arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._session.submit_tool_error(event.tool_call.id, str(exc))
            return

        await self._session.submit_tool_result(event.tool_call.id, result)

    async def _run_child_agent(self, request: LaunchAgentSchema) -> TextResult:
        model = self._resolve_expert_model() if request.expert_knowledge else self._model
        child = ManagedSession(
            instructions=self._instructions,
            tools=self._base_tools,
            model=self._model,
            expert_model=self._expert_model,
            runtime_options=replace(self._runtime_options),
            completer=self._completer,
            history_store=None,
            tool_policy=self._tool_policy,
        )

        last_assistant_message: AssistantMessage | None = None
        try:
            async with child:
                await child.start(
                    mode="agent",
                    task=request.task,
                    instructions=self._compose_child_instructions(request),
                    model=model,
                )
                async for event in child.events():
                    if isinstance(event, AssistantMessageEvent):
                        last_assistant_message = event.message
                        continue
                    if isinstance(event, CompletedEvent):
                        return TextResult(content=event.result)
                    if isinstance(event, InputRequestedEvent):
                        detail = (
                            last_assistant_message.content
                            if last_assistant_message is not None and isinstance(last_assistant_message.content, str)
                            else "The sub-agent needs more information to proceed."
                        )
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

    def _resolve_model(self, *, mode: Literal["chat", "agent"]) -> str:
        if mode == "agent":
            return self._resolve_expert_model()
        return self._model

    def _resolve_expert_model(self) -> str:
        return self._expert_model or self._model

    def _build_start_history(
        self,
        *,
        mode: Literal["chat", "agent"],
        task: str | None,
        history: list[BaseMessage] | None,
        instructions: str | None,
    ) -> list[BaseMessage]:
        if mode == "agent" and not task:
            raise ValueError("Agent mode requires a task.")

        start_history = list(history or [])
        if not start_history:
            start_history.append(
                SystemMessage(
                    content=self._build_system_prompt(
                        mode=mode,
                        instructions=self._merge_instructions(extra_instructions=instructions),
                    )
                )
            )
        elif instructions and instructions.strip():
            start_history.append(SystemMessage(content=self._format_run_instructions(instructions)))

        if mode == "agent":
            assert task is not None
            start_history.append(UserMessage(content=task))

        return start_history

    def _merge_instructions(self, *, extra_instructions: str | None) -> str:
        if not extra_instructions or not extra_instructions.strip():
            return self._instructions
        return f"{self._instructions}\n\n# Run-specific instructions\n\n{extra_instructions.strip()}"

    def _build_system_prompt(self, *, mode: Literal["chat", "agent"], instructions: str) -> str:
        template = CHAT_SYSTEM_PROMPT_TEMPLATE if mode == "chat" else AGENT_SYSTEM_PROMPT_TEMPLATE
        return template.format(instructions_section=_render_instructions_section(instructions))

    def _format_run_instructions(self, instructions: str) -> str:
        return f"# Run-specific instructions\n\n{instructions.strip()}"


def _render_instructions_section(instructions: str) -> str:
    cleaned = instructions.strip()
    if not cleaned:
        return ""
    return f"## Instructions\n\n{cleaned}"
