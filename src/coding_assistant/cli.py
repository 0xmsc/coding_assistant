from __future__ import annotations

import os
import sys
from argparse import Namespace
from contextlib import asynccontextmanager
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import AsyncIterator

from rich import print as rich_print
from rich.markdown import Markdown
from rich.panel import Panel

from coding_assistant.agent import (
    AwaitingToolsEvent,
    AwaitingUserEvent,
    ContentDeltaEvent,
    FailedEvent,
    ReasoningDeltaEvent,
    StatusEvent,
    run_agent,
)
from coding_assistant.history import build_system_prompt, compact_history
from coding_assistant.history_store import FileHistoryStore
from coding_assistant.image import get_image
from coding_assistant.instructions import get_instructions
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from coding_assistant.mcp import __name__ as mcp_package_name
from coding_assistant.sandbox import sandbox
from coding_assistant.tool_policy import ConfirmationToolPolicy, NullToolPolicy, ToolPolicy
from coding_assistant.tools.builtin import CompactConversationTool, RedirectToolCallTool, ToolExecutionResult
from coding_assistant.tools.mcp import (
    MCPServer,
    MCPServerConfig,
    get_mcp_servers_from_config,
    get_mcp_wrapped_tools,
    print_mcp_tools,
)
from coding_assistant.ui import DefaultAnswerUI, PromptToolkitUI, UI


@dataclass(slots=True)
class DefaultAgentConfig:
    """Configuration for the default CLI and embedding setup."""

    working_directory: Path
    mcp_server_configs: tuple[MCPServerConfig, ...] = ()
    skills_directories: tuple[str, ...] = ()
    mcp_env: tuple[str, ...] = ()
    user_instructions: tuple[str, ...] = ()


@dataclass(slots=True)
class DefaultAgentBundle:
    """Resolved defaults needed to run an agent in one place."""

    tools: list[Tool]
    instructions: str
    mcp_servers: list[MCPServer]
    history_store: FileHistoryStore


class DeltaRenderer:
    """Write streamed assistant text directly to stdout."""

    def on_delta(self, chunk: str) -> None:
        """Render one streamed content chunk without buffering."""
        sys.stdout.write(chunk)
        sys.stdout.flush()


def build_default_agent_config(args: Namespace) -> DefaultAgentConfig:
    """Translate CLI arguments into the default agent configuration."""
    working_directory = Path(os.getcwd())
    mcp_server_configs = tuple(MCPServerConfig.model_validate_json(item) for item in args.mcp_servers)
    return DefaultAgentConfig(
        working_directory=working_directory,
        mcp_server_configs=mcp_server_configs,
        skills_directories=tuple(args.skills_directories),
        mcp_env=tuple(args.mcp_env),
        user_instructions=tuple(args.instructions),
    )


def _get_default_mcp_server_config(
    skills_directories: tuple[str, ...],
    env: tuple[str, ...] = (),
) -> MCPServerConfig:
    """Build the bundled MCP server config used by default runs."""
    args = ["-m", mcp_package_name]
    if skills_directories:
        args.append("--skills-directories")
        args.extend(skills_directories)

    return MCPServerConfig(
        name="coding_assistant.mcp",
        command=sys.executable,
        args=args,
        env=list(env),
    )


@asynccontextmanager
async def create_default_agent(
    *,
    config: DefaultAgentConfig,
) -> AsyncIterator[DefaultAgentBundle]:
    """Resolve instructions, tools, and history storage for a default agent run."""
    history_store = FileHistoryStore(config.working_directory)
    server_configs = (
        *config.mcp_server_configs,
        _get_default_mcp_server_config(config.skills_directories, config.mcp_env),
    )

    async with get_mcp_servers_from_config(list(server_configs), working_directory=config.working_directory) as servers:
        tools = await get_mcp_wrapped_tools(servers)
        instructions = get_instructions(
            working_directory=config.working_directory,
            user_instructions=list(config.user_instructions),
            mcp_servers=servers,
        )
        yield DefaultAgentBundle(
            tools=tools,
            instructions=instructions,
            mcp_servers=servers,
            history_store=history_store,
        )


async def run_cli(args: Namespace) -> None:
    """Run the interactive or single-shot CLI entry point."""
    config = build_default_agent_config(args)

    if args.sandbox:
        _apply_sandbox(args=args, config=config)

    ui: UI
    if args.task is None or args.ask_user:
        ui = PromptToolkitUI()
    else:
        ui = DefaultAnswerUI()

    tool_policy: ToolPolicy
    if args.tool_confirmation_patterns or args.shell_confirmation_patterns:
        tool_policy = ConfirmationToolPolicy(
            ui=ui,
            tool_confirmation_patterns=list(args.tool_confirmation_patterns),
            shell_confirmation_patterns=list(args.shell_confirmation_patterns),
        )
    else:
        tool_policy = NullToolPolicy()

    async with create_default_agent(config=config) as bundle:
        if args.print_mcp_tools:
            await print_mcp_tools(bundle.mcp_servers)
            return

        if args.print_instructions:
            rich_print(Panel(Markdown(bundle.instructions), title="Instructions"))
            return

        history = _load_history(
            args=args,
            history_store=bundle.history_store,
            working_directory=config.working_directory,
        )
        current_history = _ensure_initial_history(history=history, instructions=bundle.instructions)
        if args.task is not None:
            current_history.append(UserMessage(content=args.task))

        await _drive_agent(
            history=current_history,
            model=args.model,
            tools=_build_agent_tools(raw_tools=bundle.tools, tool_policy=tool_policy),
            tool_policy=tool_policy,
            history_store=bundle.history_store,
            ui=ui,
            interactive=args.task is None or args.ask_user,
        )


def _load_history(
    *,
    args: Namespace,
    history_store: FileHistoryStore,
    working_directory: Path,
) -> list[BaseMessage] | None:
    """Load persisted history according to the selected resume flags."""
    if args.resume_file is not None:
        file_store = FileHistoryStore(working_directory, path=args.resume_file)
        return file_store.load()

    if args.resume:
        return history_store.load()

    return None


def _apply_sandbox(*, args: Namespace, config: DefaultAgentConfig) -> None:
    """Apply the CLI sandbox policy before starting the agent."""
    coding_assistant_root = Path(__file__).resolve().parent
    readable = [
        *[Path(path).resolve() for path in args.readable_sandbox_directories],
        *[Path(path).resolve() for path in config.skills_directories],
        coding_assistant_root,
    ]
    writable = [*([Path(path).resolve() for path in args.writable_sandbox_directories]), config.working_directory]
    sandbox(readable_paths=readable, writable_paths=writable, include_defaults=True)


def _ensure_initial_history(
    *,
    history: list[BaseMessage] | None,
    instructions: str,
) -> list[BaseMessage]:
    """Create the initial system prompt when no prior history exists."""
    if history:
        return list(history)
    return [SystemMessage(content=build_system_prompt(instructions=instructions))]


async def _drive_agent(
    *,
    history: list[BaseMessage],
    model: str,
    tools: list[Tool],
    tool_policy: ToolPolicy,
    history_store: FileHistoryStore,
    ui: UI,
    interactive: bool,
) -> None:
    """Drive one or more `run_agent` turns until the CLI should exit."""
    renderer = DeltaRenderer()
    command_names = ["/exit", "/help", "/compact", "/image"]
    tools_by_name = {tool.name(): tool for tool in tools}

    current_history = list(history)
    while True:
        saw_content = False
        boundary_event: AwaitingToolsEvent | AwaitingUserEvent | FailedEvent | None = None

        async for event in run_agent(
            history=current_history,
            model=model,
            tools=tools,
        ):
            if isinstance(event, ContentDeltaEvent):
                renderer.on_delta(event.content)
                saw_content = True
            elif isinstance(event, ReasoningDeltaEvent):
                continue
            elif isinstance(event, StatusEvent):
                continue
            else:
                boundary_event = event

        if boundary_event is None:
            print("Error: Agent stopped without yielding a terminal event.")
            return

        if saw_content:
            renderer.on_delta("\n")

        if isinstance(boundary_event, FailedEvent):
            print(f"Error: {boundary_event.error}")
            return

        if isinstance(boundary_event, AwaitingToolsEvent):
            current_history.append(boundary_event.message)
            current_history = await _apply_tool_calls(
                history=current_history,
                message=boundary_event.message,
                tools_by_name=tools_by_name,
                tool_policy=tool_policy,
            )
            history_store.save(current_history)
            continue

        if boundary_event.message is not None:
            current_history.append(boundary_event.message)
        history_store.save(current_history)

        if not interactive:
            return

        while True:
            answer = await ui.prompt(words=command_names)
            stripped = answer.strip()
            if stripped == "/exit":
                return
            if stripped == "/help":
                print("Available commands:\n  /exit\n  /help\n  /compact\n  /image <path-or-url>")
                continue
            if stripped == "/compact":
                current_history.append(
                    UserMessage(
                        content="Immediately compact our conversation so far by using the `compact_conversation` tool."
                    )
                )
                break
            if stripped.startswith("/image"):
                parts = stripped.split(maxsplit=1)
                if len(parts) < 2:
                    print("/image requires a path or URL argument.")
                    continue
                data_url = await get_image(parts[1])
                image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
                current_history.append(UserMessage(content=image_content))
                break

            current_history.append(UserMessage(content=answer))
            break


def _build_agent_tools(*, raw_tools: list[Tool], tool_policy: ToolPolicy) -> list[Tool]:
    """Build the complete tool list exposed to the model for CLI runs."""
    base_tools = [CompactConversationTool(), *raw_tools]
    base_tools_by_name = {tool.name(): tool for tool in base_tools}
    redirect_tool = RedirectToolCallTool(
        tools=base_tools,
        execute_tool=lambda tool_name, arguments: _execute_non_compacting_tool(
            tool_name=tool_name,
            arguments=arguments,
            tools_by_name=base_tools_by_name,
            tool_policy=tool_policy,
        ),
    )
    return [*base_tools, redirect_tool]


def _parse_tool_call_arguments(tool_call: ToolCall) -> tuple[dict[str, object] | None, str | None]:
    """Decode tool arguments into a JSON object or return a user-visible error."""
    import json

    try:
        parsed = json.loads(tool_call.function.arguments)
    except JSONDecodeError as exc:
        return None, f"Error: Tool call arguments `{tool_call.function.arguments}` are not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "Error: Tool call arguments must decode to a JSON object."
    return parsed, None


async def _apply_tool_calls(
    *,
    history: list[BaseMessage],
    message: AssistantMessage,
    tools_by_name: dict[str, Tool],
    tool_policy: ToolPolicy,
) -> list[BaseMessage]:
    """Execute tool calls requested by the assistant and append tool results."""
    current_history = list(history)
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
        current_history, result = await _execute_tool_call(
            history=current_history,
            tool_call_id=tool_call.id,
            tool_name=tool_name,
            arguments=arguments,
            tools_by_name=tools_by_name,
            tool_policy=tool_policy,
        )
        current_history.append(
            ToolMessage(
                tool_call_id=tool_call.id,
                name=tool_name,
                content=result,
            )
        )

    return current_history


async def _execute_tool_call(
    *,
    history: list[BaseMessage],
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, object],
    tools_by_name: dict[str, Tool],
    tool_policy: ToolPolicy,
) -> tuple[list[BaseMessage], str]:
    """Apply policy, execute one tool, and return any updated history plus result text."""
    if policy_result := await tool_policy.before_tool_execution(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments=arguments,
    ):
        return history, policy_result

    tool = tools_by_name.get(tool_name)
    if tool is None:
        return history, f"Tool '{tool_name}' not found."

    try:
        raw_result = await tool.execute(arguments)
    except Exception as exc:
        return history, f"Error executing tool: {exc}"

    if not isinstance(raw_result, str):
        return history, f"Error executing tool: Tool '{tool_name}' did not return text."

    if tool_name == "compact_conversation":
        return compact_history(history, raw_result), "Conversation compacted and history reset."

    return history, raw_result


async def _execute_non_compacting_tool(
    *,
    tool_name: str,
    arguments: dict[str, object],
    tools_by_name: dict[str, Tool],
    tool_policy: ToolPolicy,
) -> ToolExecutionResult:
    """Execute a redirectable tool while keeping policy checks above the runtime."""
    if tool_name == "compact_conversation":
        return False, "Error: Cannot redirect compact_conversation."

    if policy_result := await tool_policy.before_tool_execution(
        tool_call_id="",
        tool_name=tool_name,
        arguments=arguments,
    ):
        return False, policy_result

    tool = tools_by_name.get(tool_name)
    if tool is None:
        return False, f"Tool '{tool_name}' not found."

    try:
        raw_result = await tool.execute(arguments)
    except Exception as exc:
        return False, f"Error executing tool: {exc}"

    if not isinstance(raw_result, str):
        return False, f"Error: Tool '{tool_name}' did not return text."
    return True, raw_result
