from __future__ import annotations

import importlib.resources
import os
import sys
from argparse import Namespace
from pathlib import Path

from rich import print as rich_print
from rich.markdown import Markdown
from rich.panel import Panel

from coding_assistant.config import MCPServerConfig
from coding_assistant.defaults import DefaultSessionConfig, create_default_session
from coding_assistant.history_store import FileHistoryStore
from coding_assistant.image import get_image
from coding_assistant.runner import ManagedSession
from coding_assistant.runtime import (
    AssistantDeltaEvent,
    AssistantMessageEvent,
    CancelledEvent,
    FailedEvent,
    InputRequestedEvent,
)
from coding_assistant.sandbox import sandbox
from coding_assistant.tool_policy import ConfirmationToolPolicy, NullToolPolicy, ToolPolicy
from coding_assistant.tools.mcp import print_mcp_tools
from coding_assistant.ui import DefaultAnswerUI, PromptToolkitUI, UI
from coding_assistant.llm.types import BaseMessage


class EventRenderer:
    def __init__(self) -> None:
        self._streaming = False

    def render(self, event: object) -> None:
        if isinstance(event, AssistantDeltaEvent):
            sys.stdout.write(event.delta)
            sys.stdout.flush()
            self._streaming = True
            return

        if isinstance(event, AssistantMessageEvent):
            if self._streaming:
                self._finish_stream()
            elif isinstance(event.message.content, str) and event.message.content:
                print(event.message.content)
            return

        if isinstance(event, InputRequestedEvent):
            self._finish_stream()
            return

        if isinstance(event, FailedEvent):
            self._finish_stream()
            print(f"Error: {event.error}")
            return

        if isinstance(event, CancelledEvent):
            self._finish_stream()
            print("Cancelled.")

    def _finish_stream(self) -> None:
        if not self._streaming:
            return
        if not sys.stdout.isatty():
            print()
        else:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._streaming = False


def build_default_session_config(args: Namespace) -> DefaultSessionConfig:
    working_directory = Path(os.getcwd())
    coding_assistant_root = Path(str(importlib.resources.files("coding_assistant"))).parent.resolve()
    mcp_server_configs = tuple(MCPServerConfig.model_validate_json(item) for item in args.mcp_servers)
    return DefaultSessionConfig(
        working_directory=working_directory,
        mcp_server_configs=mcp_server_configs,
        skills_directories=tuple(args.skills_directories),
        mcp_env=tuple(args.mcp_env),
        user_instructions=tuple(args.instructions),
        coding_assistant_root=coding_assistant_root,
    )


async def run_cli(args: Namespace) -> None:
    config = build_default_session_config(args)

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

    async with create_default_session(
        model=args.model,
        expert_model=args.expert_model,
        config=config,
        tool_policy=tool_policy,
    ) as bundle:
        session = bundle.session
        if args.print_mcp_tools:
            await print_mcp_tools(bundle.mcp_servers)
            return

        if args.print_instructions:
            rich_print(Panel(Markdown(session.instructions), title="Instructions"))
            return

        history = _load_history(
            args=args,
            history_store=bundle.history_store,
            working_directory=config.working_directory,
        )

        await session.start(
            history=history,
            initial_user_message=args.task,
            use_expert_model=args.task is not None,
        )
        await _drive_session(session=session, ui=ui, interactive=args.task is None or args.ask_user)


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


def _apply_sandbox(*, args: Namespace, config: DefaultSessionConfig) -> None:
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


async def _drive_session(*, session: ManagedSession, ui: UI, interactive: bool) -> None:
    renderer = EventRenderer()
    command_names = ["/exit", "/help", "/compact", "/image"]

    async for event in session.events():
        renderer.render(event)
        if isinstance(event, FailedEvent | CancelledEvent):
            return
        if not isinstance(event, InputRequestedEvent):
            continue
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
                await session.send_user_message(
                    "Immediately compact our conversation so far by using the `compact_conversation` tool."
                )
                break
            if stripped.startswith("/image"):
                parts = stripped.split(maxsplit=1)
                if len(parts) < 2:
                    print("/image requires a path or URL argument.")
                    continue
                data_url = await get_image(parts[1])
                image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
                await session.send_user_message(image_content)
                break

            await session.send_user_message(answer)
            break
