import asyncio
import logging
import os
import sys
import importlib.resources
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction
from pathlib import Path

import debugpy  # type: ignore[import-untyped]
from rich import print as rich_print
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel

from coding_assistant.framework.callbacks import ProgressCallbacks
from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.framework.chat import run_chat_loop
from coding_assistant.framework.types import Tool
from coding_assistant.callbacks import ConfirmationToolCallbacks, DenseProgressCallbacks
from coding_assistant.config import Config, MCPServerConfig
from coding_assistant.history import (
    get_latest_orchestrator_history_file,
    load_orchestrator_history,
    save_orchestrator_history,
)
from coding_assistant.instructions import get_instructions
from coding_assistant.sandbox import sandbox
from coding_assistant.trace import enable_tracing, get_default_trace_dir
from coding_assistant.tools.mcp import get_mcp_servers_from_config, get_mcp_wrapped_tools, print_mcp_tools
from coding_assistant.tools.mcp_server import start_mcp_server, get_free_port
from coding_assistant.tools.tools import AgentTool, AskClientTool
from coding_assistant.ui import PromptToolkitUI, DefaultAnswerUI

logging.basicConfig(level=logging.WARNING, handlers=[RichHandler()])
logger = logging.getLogger("coding_assistant")
logger.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description="Coding Assistant CLI")
    parser.add_argument(
        "--task", type=str, help="Task for the orchestrator agent. If provided, the agent runs in autonomous mode."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest orchestrator history file in .coding_assistant/history/.",
    )
    parser.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        help="Resume from a specific orchestrator history file.",
    )
    parser.add_argument("--print-mcp-tools", action="store_true", help="Print all available tools from MCP servers.")
    parser.add_argument(
        "--print-instructions",
        action="store_true",
        help="Print the instructions that will be given to the orchestrator agent and exit.",
    )
    parser.add_argument("--model", type=str, required=True, help="Model to use for the orchestrator agent.")
    parser.add_argument("--expert-model", type=str, default=None, help="Expert model to use.")
    parser.add_argument(
        "--instructions",
        nargs="*",
        default=[],
        help="Custom instructions for the agent.",
    )
    parser.add_argument(
        "--readable-sandbox-directories",
        nargs="*",
        default=[],
        help="Additional directories to include in the sandbox.",
    )
    parser.add_argument(
        "--writable-sandbox-directories",
        nargs="*",
        default=[],
        help="Additional directories to include in the sandbox.",
    )
    parser.add_argument(
        "--sandbox",
        action=BooleanOptionalAction,
        default=True,
        help="Enable sandboxing.",
    )
    parser.add_argument(
        "--mcp-servers",
        nargs="*",
        default=[],
        help='MCP server configurations as JSON strings. Format: \'{"name": "server_name", "command": "command", "args": ["arg1", "arg2"], "env": ["ENV_VAR1", "ENV_VAR2"]}\' or \'{"name": "server_name", "url": "http://localhost:8000/sse"}\'',
    )
    parser.add_argument(
        "--compact-conversation-at-tokens",
        type=int,
        default=200_000,
        help="Number of tokens after which conversation should be shortened.",
    )
    parser.add_argument(
        "--tool-confirmation-patterns",
        nargs="*",
        default=[],
        help="Ask for confirmation before executing a tool that matches any of the given patterns.",
    )
    parser.add_argument(
        "--shell-confirmation-patterns",
        nargs="*",
        default=[],
        help="Regex patterns that require confirmation before executing shell commands",
    )
    parser.add_argument(
        "--wait-for-debugger",
        action=BooleanOptionalAction,
        default=False,
        help="Wait for a debugger to attach.",
    )
    parser.add_argument(
        "--trace",
        action=BooleanOptionalAction,
        default=False,
        help="Enable tracing of model requests and responses to a session folder in $XDG_STATE_HOME/coding-assistant/traces.",
    )
    parser.add_argument(
        "--ask-user",
        action=BooleanOptionalAction,
        default=True,
        help="Enable/disable asking the user for input in agent mode.",
    )
    parser.add_argument(
        "--mcp-server",
        action=BooleanOptionalAction,
        default=True,
        help="Start an MCP server in the background exposing all available tools.",
    )
    parser.add_argument(
        "--mcp-server-port",
        type=int,
        default=0,
        help="Port for the background MCP server (using streamable-http transport).",
    )
    parser.add_argument(
        "--print-reasoning",
        action=BooleanOptionalAction,
        default=True,
        help="Print reasoning chunks from the model.",
    )
    parser.add_argument(
        "--skills-directories",
        nargs="*",
        default=[],
        help="Paths to directories containing Agent Skills (with SKILL.md files).",
    )

    return parser.parse_args()


def create_config_from_args(args) -> Config:
    return Config(
        model=args.model,
        expert_model=args.expert_model or args.model,
        compact_conversation_at_tokens=args.compact_conversation_at_tokens,
        enable_chat_mode=args.task is None,
        enable_ask_user=args.ask_user,
    )


async def run_root_agent(
    task: str,
    config: Config,
    tools: list[Tool],
    history: list | None,
    instructions: str | None,
    working_directory: Path,
    progress_callbacks: ProgressCallbacks,
    tool_callbacks: ConfirmationToolCallbacks,
):
    agent_ui = PromptToolkitUI() if config.enable_ask_user else DefaultAnswerUI()

    agent_mode_tools = [
        AskClientTool(ui=agent_ui),
        *tools,
    ]

    tool = AgentTool(
        model=config.model,
        expert_model=config.expert_model,
        compact_conversation_at_tokens=config.compact_conversation_at_tokens,
        enable_ask_user=config.enable_ask_user,
        tools=agent_mode_tools,
        history=history,
        progress_callbacks=progress_callbacks,
        ui=agent_ui,
        tool_callbacks=tool_callbacks,
        name="launch_orchestrator_agent",
        completer=openai_complete,
    )

    orchestrator_params = {
        "task": task,
        "instructions": instructions,
        "expert_knowledge": True,
    }

    try:
        result = await tool.execute(orchestrator_params)
    finally:
        save_orchestrator_history(working_directory, tool.history)

    print(f"🎉 Final Result\n\nResult:\n\n{result.content}")
    return result


async def run_chat_session(
    *,
    config: Config,
    tools: list[Tool],
    history: list | None,
    instructions: str | None,
    working_directory: Path,
    progress_callbacks: ProgressCallbacks,
    tool_callbacks: ConfirmationToolCallbacks,
):
    chat_history = history or []

    try:
        await run_chat_loop(
            history=chat_history,
            model=config.model,
            tools=tools,
            instructions=instructions,
            callbacks=progress_callbacks,
            tool_callbacks=tool_callbacks,
            completer=openai_complete,
            ui=PromptToolkitUI(),
            context_name="Orchestrator",
        )
    finally:
        save_orchestrator_history(working_directory, chat_history)


def get_default_mcp_server_config(
    root_directory: Path, skills_directories: list[str], mcp_url: str | None = None
) -> MCPServerConfig:
    args = [
        "-m",
        "coding_assistant.mcp",
    ]

    if skills_directories:
        args.append("--skills-directories")
        args.extend(skills_directories)

    if mcp_url:
        args.extend(["--mcp-url", mcp_url])

    return MCPServerConfig(
        name="coding_assistant.mcp",
        command=sys.executable,
        args=args,
    )


def enable_sandboxing(args, working_directory, root):
    if args.sandbox:
        logger.info("Sandboxing is enabled.")

        readable_sandbox_directories = [
            *[Path(d).resolve() for d in args.readable_sandbox_directories],
            *[Path(d).resolve() for d in args.skills_directories],
            root,
        ]

        writable_sandbox_directories = [
            *[Path(d).resolve() for d in args.writable_sandbox_directories],
            working_directory,
        ]

        sandbox(
            readable_paths=readable_sandbox_directories,
            writable_paths=writable_sandbox_directories,
            include_defaults=True,
        )
    else:
        logger.warning("Sandboxing is disabled")


async def _main(args):
    logger.info(f"Starting Coding Assistant with arguments {args}")

    config = create_config_from_args(args)
    logger.info(f"Using configuration from command line arguments: {config}")

    working_directory = Path(os.getcwd())
    logger.info(f"Running in working directory: {working_directory}")

    coding_assistant_root = Path(str(importlib.resources.files("coding_assistant"))).parent.resolve()
    logger.info(f"Coding Assistant root directory: {coding_assistant_root}")

    if args.resume_file:
        if not args.resume_file.exists():
            raise FileNotFoundError(f"Resume file {args.resume_file} does not exist.")
        logger.info(f"Resuming session from file: {args.resume_file}")
        resume_history = load_orchestrator_history(args.resume_file)
    elif args.resume:
        latest_history_file = get_latest_orchestrator_history_file(working_directory)
        if not latest_history_file:
            raise FileNotFoundError(
                f"No latest orchestrator history file found in {working_directory}/.coding_assistant/history."
            )
        logger.info(f"Resuming session from latest saved agent history: {latest_history_file}")
        resume_history = load_orchestrator_history(latest_history_file)
    else:
        resume_history = None

    enable_sandboxing(
        args,
        working_directory=working_directory,
        root=coding_assistant_root,
    )

    mcp_server_configs = [MCPServerConfig.model_validate_json(mcp_config_json) for mcp_config_json in args.mcp_servers]

    if args.mcp_server and args.mcp_server_port == 0:
        args.mcp_server_port = get_free_port()
        logger.info(f"Selected random port for background MCP server: {args.mcp_server_port}")

    mcp_url = f"http://localhost:{args.mcp_server_port}/mcp" if args.mcp_server else None
    mcp_server_configs.append(
        get_default_mcp_server_config(coding_assistant_root, args.skills_directories, mcp_url=mcp_url)
    )

    logger.info(f"Using MCP server configurations: {[s.name for s in mcp_server_configs]}")

    async with get_mcp_servers_from_config(mcp_server_configs, working_directory) as mcp_servers:
        if args.print_mcp_tools:
            await print_mcp_tools(mcp_servers)
            return

        tools = await get_mcp_wrapped_tools(mcp_servers)

        if args.mcp_server:
            await start_mcp_server(tools, args.mcp_server_port)

        instructions = get_instructions(
            working_directory=working_directory,
            user_instructions=args.instructions,
            mcp_servers=mcp_servers,
        )

        if args.print_instructions:
            rich_print(Panel(Markdown(instructions), title="Instructions"))
            return

        progress_callbacks = DenseProgressCallbacks(print_reasoning=args.print_reasoning)

        tool_callbacks = ConfirmationToolCallbacks(
            tool_confirmation_patterns=args.tool_confirmation_patterns,
            shell_confirmation_patterns=args.shell_confirmation_patterns,
        )

        if config.enable_chat_mode:
            await run_chat_session(
                config=config,
                tools=tools,
                history=resume_history,
                instructions=instructions,
                working_directory=working_directory,
                progress_callbacks=progress_callbacks,
                tool_callbacks=tool_callbacks,
            )
        else:
            await run_root_agent(
                task=args.task,
                config=config,
                tools=tools,
                history=resume_history,
                instructions=instructions,
                working_directory=working_directory,
                progress_callbacks=progress_callbacks,
                tool_callbacks=tool_callbacks,
            )


def main():
    args = parse_args()

    if args.trace:
        enable_tracing(get_default_trace_dir())

    if args.wait_for_debugger:
        logger.info("Waiting for debugger to attach on port 1234")
        debugpy.listen(1234)
        debugpy.wait_for_client()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
