from __future__ import annotations

import os
import sys
from argparse import Namespace
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from rich import print as rich_print
from rich.markdown import Markdown
from rich.panel import Panel

from coding_assistant.agent import run_agent
from coding_assistant.history import build_system_prompt
from coding_assistant.history_store import FileHistoryStore
from coding_assistant.image import get_image
from coding_assistant.instructions import get_instructions
from coding_assistant.llm.types import (
    BaseMessage,
    SystemMessage,
    Tool,
    UserMessage,
)
from coding_assistant.mcp import __name__ as mcp_package_name
from coding_assistant.sandbox import sandbox
from coding_assistant.tool_policy import (
    ConfirmationToolPolicy,
    DirectToolExecutor,
    NullToolPolicy,
    ToolApproved,
    ToolDenied,
    ToolExecutor,
    ToolPolicy,
)
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

    def __init__(self) -> None:
        self.saw_content = False

    def on_delta(self, chunk: str) -> None:
        """Render one streamed content chunk without buffering."""
        self.saw_content = True
        sys.stdout.write(chunk)
        sys.stdout.flush()


class PolicyToolExecutor:
    """Apply approval policy before delegating to direct tool execution."""

    def __init__(self, *, tool_policy: ToolPolicy) -> None:
        self._tool_policy = tool_policy
        self._delegate = DirectToolExecutor()

    async def execute(
        self,
        *,
        tool_call_id: str,
        tool: Tool,
        arguments: dict[str, Any],
    ) -> ToolApproved | ToolDenied:
        """Run approval checks before executing the resolved tool."""
        if policy_result := await self._tool_policy.before_tool_execution(
            tool_call_id=tool_call_id,
            tool_name=tool.name(),
            arguments=arguments,
        ):
            return ToolDenied(content=policy_result)

        return await self._delegate.execute(
            tool_call_id=tool_call_id,
            tool=tool,
            arguments=arguments,
        )


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
    tool_executor: ToolExecutor = PolicyToolExecutor(tool_policy=tool_policy)

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
            tools=bundle.tools,
            tool_executor=tool_executor,
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
    tool_executor: ToolExecutor,
    history_store: FileHistoryStore,
    ui: UI,
    interactive: bool,
) -> None:
    """Drive one or more `run_agent` turns until the CLI should exit."""
    command_names = ["/exit", "/help", "/compact", "/image"]

    current_history = list(history)
    while True:
        renderer = DeltaRenderer()
        current_history = await run_agent(
            history=current_history,
            model=model,
            tools=tools,
            tool_executor=tool_executor,
            on_content=renderer.on_delta,
        )

        if renderer.saw_content:
            renderer.on_delta("\n")
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
