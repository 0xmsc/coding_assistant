from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from json import JSONDecodeError
from typing import Any, Callable

from coding_assistant.agent_types import AgentRunResult
from coding_assistant.history import compact_history
from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    NullProgressCallbacks,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from coding_assistant.tool_policy import NullToolPolicy, ToolPolicy
from coding_assistant.tools.builtin import (
    CompactConversationTool,
    LaunchAgentSchema,
    LaunchAgentTool,
    RedirectToolCallTool,
)


class _ProgressCallbacks(NullProgressCallbacks):
    def __init__(self, on_delta: Callable[[str], None] | None) -> None:
        self._on_delta = on_delta
        self.saw_content = False

    def on_content_chunk(self, chunk: str) -> None:
        self.saw_content = True
        if self._on_delta is not None:
            self._on_delta(chunk)


def _parse_tool_call_arguments(tool_call: ToolCall) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(tool_call.function.arguments)
    except JSONDecodeError as exc:
        return None, f"Error: Tool call arguments `{tool_call.function.arguments}` are not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "Error: Tool call arguments must decode to a JSON object."
    return parsed, None


def _compose_child_instructions(request: LaunchAgentSchema) -> str | None:
    parts: list[str] = []
    if request.instructions:
        parts.append(request.instructions.strip())
    if request.expected_output:
        parts.append(f"Expected output:\n{request.expected_output.strip()}")
    if not parts:
        return None
    return "\n\n".join(parts)


def _format_run_instructions(instructions: str) -> str:
    return f"# Run-specific instructions\n\n{instructions.strip()}"


def _build_child_history(history: Sequence[BaseMessage], request: LaunchAgentSchema) -> list[BaseMessage]:
    child_history: list[BaseMessage] = []
    if history and isinstance(history[0], SystemMessage):
        child_history.append(history[0])

    if child_instructions := _compose_child_instructions(request):
        child_history.append(SystemMessage(content=_format_run_instructions(child_instructions)))

    child_history.append(UserMessage(content=request.task))
    return child_history


def _last_assistant_text(history: Sequence[BaseMessage]) -> str | None:
    for message in reversed(history):
        if isinstance(message, AssistantMessage) and isinstance(message.content, str):
            return message.content
    return None


def _build_tools(tools: Sequence[Tool], execute_child: Callable[[LaunchAgentSchema], Any]) -> list[Tool]:
    builtin_tools: list[Tool] = [
        CompactConversationTool(),
        LaunchAgentTool(execute_child=execute_child),
    ]
    redirectable_tools = [*builtin_tools, *tools]
    all_tools = [*redirectable_tools, RedirectToolCallTool(tools=redirectable_tools)]

    names: set[str] = set()
    for tool in all_tools:
        name = tool.name()
        if name in names:
            raise ValueError(f"Duplicate tool name: {name}")
        names.add(name)

    return all_tools


def _should_wait_for_user(history: Sequence[BaseMessage]) -> bool:
    return history[-1].role not in {"user", "tool"}


async def run_agent(
    *,
    history: Sequence[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    expert_model: str | None = None,
    tool_policy: ToolPolicy | None = None,
    completer: Any = openai_complete,
    on_delta: Callable[[str], None] | None = None,
) -> AgentRunResult:
    current_history = list(history)
    if not current_history:
        raise ValueError("run_agent requires a non-empty history.")

    resolved_tool_policy = tool_policy or NullToolPolicy()

    if _should_wait_for_user(current_history):
        return AgentRunResult(history=current_history, status="awaiting_user")

    async def execute_child(request: LaunchAgentSchema) -> str:
        child_result = await run_agent(
            history=_build_child_history(current_history, request),
            model=(expert_model or model) if request.expert_knowledge else model,
            tools=tools,
            expert_model=expert_model,
            tool_policy=resolved_tool_policy,
            completer=completer,
            on_delta=None,
        )
        if child_result.status == "failed":
            return f"Sub-agent failed: {child_result.error}"

        if content := _last_assistant_text(child_result.history):
            return content
        return "The sub-agent yielded control without a text reply."

    all_tools = _build_tools(tools, execute_child)
    tools_by_name = {tool.name(): tool for tool in all_tools}

    try:
        while True:
            progress_callbacks = _ProgressCallbacks(on_delta)
            completion = await completer(
                current_history,
                model=model,
                tools=all_tools,
                callbacks=progress_callbacks,
            )
            message = completion.message
            current_history.append(message)
            if progress_callbacks.saw_content and on_delta is not None:
                on_delta("\n")

            if not message.tool_calls:
                return AgentRunResult(history=current_history, status="awaiting_user")

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                if not tool_name:
                    raise RuntimeError(f"Tool call {tool_call.id} is missing function name.")

                arguments, parse_error = _parse_tool_call_arguments(tool_call)
                if parse_error is not None:
                    current_history.append(
                        ToolMessage(
                            tool_call_id=tool_call.id,
                            name=tool_name,
                            content=parse_error,
                        )
                    )
                    continue

                assert arguments is not None

                if policy_result := await resolved_tool_policy.before_tool_execution(
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    arguments=arguments,
                ):
                    result = policy_result
                else:
                    tool = tools_by_name.get(tool_name)
                    if tool is None:
                        result = f"Tool '{tool_name}' not found."
                    else:
                        try:
                            raw_result = await tool.execute(arguments)
                            if not isinstance(raw_result, str):
                                result = f"Error executing tool: Tool '{tool_name}' did not return text."
                            elif tool_name == "compact_conversation":
                                current_history = compact_history(current_history, raw_result)
                                result = "Conversation compacted and history reset."
                            else:
                                result = raw_result
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            result = f"Error executing tool: {exc}"

                current_history.append(
                    ToolMessage(
                        tool_call_id=tool_call.id,
                        name=tool_name,
                        content=result,
                    )
                )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = str(exc) or exc.__class__.__name__
        return AgentRunResult(history=current_history, status="failed", error=error)
