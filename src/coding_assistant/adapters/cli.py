from __future__ import annotations

import importlib.resources
import os
import sys
from argparse import Namespace
from pathlib import Path

from rich import print as rich_print
from rich.markdown import Markdown
from rich.panel import Panel

from coding_assistant.agent import run_agent
from coding_assistant.config import MCPServerConfig
from coding_assistant.defaults import DefaultAgentConfig, create_default_agent
from coding_assistant.history_store import FileHistoryStore
from coding_assistant.history import build_system_prompt
from coding_assistant.image import get_image
from coding_assistant.sandbox import sandbox
from coding_assistant.tool_policy import ConfirmationToolPolicy, NullToolPolicy, ToolPolicy
from coding_assistant.tools.mcp import print_mcp_tools
from coding_assistant.ui import DefaultAnswerUI, PromptToolkitUI, UI
from coding_assistant.llm.types import BaseMessage, SystemMessage, Tool, UserMessage


class DeltaRenderer:
    def on_delta(self, chunk: str) -> None:
        sys.stdout.write(chunk)
        sys.stdout.flush()


def build_default_agent_config(args: Namespace) -> DefaultAgentConfig:
    working_directory = Path(os.getcwd())
    coding_assistant_root = Path(str(importlib.resources.files("coding_assistant"))).parent.resolve()
    mcp_server_configs = tuple(MCPServerConfig.model_validate_json(item) for item in args.mcp_servers)
    return DefaultAgentConfig(
        working_directory=working_directory,
        mcp_server_configs=mcp_server_configs,
        skills_directories=tuple(args.skills_directories),
        mcp_env=tuple(args.mcp_env),
        user_instructions=tuple(args.instructions),
        coding_assistant_root=coding_assistant_root,
    )


async def run_cli(args: Namespace) -> None:
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
            expert_model=args.expert_model,
            tools=bundle.tools,
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
    if args.resume_file is not None:
        file_store = FileHistoryStore(working_directory, path=args.resume_file)
        return file_store.load()

    if args.resume:
        return history_store.load()

    return None


def _apply_sandbox(*, args: Namespace, config: DefaultAgentConfig) -> None:
    coding_assistant_root = config.coding_assistant_root
    if coding_assistant_root is None:
        raise RuntimeError("coding_assistant_root must be resolved before sandboxing.")

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
    if history:
        return list(history)
    return [SystemMessage(content=build_system_prompt(instructions=instructions))]


async def _drive_agent(
    *,
    history: list[BaseMessage],
    model: str,
    expert_model: str | None,
    tools: list[Tool],
    tool_policy: ToolPolicy,
    history_store: FileHistoryStore,
    ui: UI,
    interactive: bool,
) -> None:
    renderer = DeltaRenderer()
    command_names = ["/exit", "/help", "/compact", "/image"]

    current_history = list(history)
    while True:
        result = await run_agent(
            history=current_history,
            model=model,
            expert_model=expert_model,
            tools=tools,
            tool_policy=tool_policy,
            on_delta=renderer.on_delta,
        )
        current_history = result.history
        history_store.save(current_history)

        if result.status == "failed":
            print(f"Error: {result.error}")
            return
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
