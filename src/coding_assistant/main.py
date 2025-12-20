import asyncio
import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction
from pathlib import Path

import debugpy  # type: ignore[import-untyped]
from rich import print as rich_print
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel

from coding_assistant.agents.callbacks import ProgressCallbacks
from coding_assistant.llm.model import complete
from coding_assistant.agents.execution import run_chat_loop
from coding_assistant.agents.parameters import Parameter
from coding_assistant.agents.types import AgentContext, AgentDescription, AgentState, Tool
from coding_assistant.callbacks import ConfirmationToolCallbacks, DenseProgressCallbacks
from coding_assistant.config import Config, MCPServerConfig
from coding_assistant.history import (
    get_latest_orchestrator_history_file,
    load_orchestrator_history,
    save_orchestrator_history,
)
from coding_assistant.instructions import get_instructions
from coding_assistant.sandbox import sandbox
from coding_assistant.trace import enable_tracing
from coding_assistant.tools.mcp import get_mcp_servers_from_config, get_mcp_wrapped_tools, print_mcp_tools
from coding_assistant.tools.tools import AgentTool, CompactConversation
from coding_assistant.ui import PromptToolkitUI

logging.basicConfig(level=logging.WARNING, handlers=[RichHandler()])
logger = logging.getLogger("coding_assistant")
logger.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description="Coding Assistant CLI")
    parser.add_argument("--task", type=str, help="Task for the orchestrator agent.")
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
    parser.add_argument("--model", type=str, default="gpt-5", help="Model to use for the orchestrator agent.")
    parser.add_argument("--expert-model", type=str, default="gpt-5", help="Expert model to use.")
    parser.add_argument(
        "--chat-mode",
        action=BooleanOptionalAction,
        default=True,
        help="Enable open-ended chat mode for the root agent (no task, no finish_task).",
    )
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
        default=bool(os.getenv("CODING_ASSISTANT_TRACE")),
        help="Enable tracing of model requests and responses to $XDG_STATE_HOME/coding-assistant/traces.",
    )

    return parser.parse_args()


def create_config_from_args(args) -> Config:
    return Config(
        model=args.model,
        expert_model=args.expert_model,
        compact_conversation_at_tokens=args.compact_conversation_at_tokens,
        enable_chat_mode=args.chat_mode,
    )


async def run_orchestrator_agent(
    task: str,
    config: Config,
    tools: list[Tool],
    history: list | None,
    instructions: str | None,
    working_directory: Path,
    agent_callbacks: ProgressCallbacks,
    tool_callbacks: ConfirmationToolCallbacks,
):
    tool = AgentTool(
        config=config,
        tools=tools,
        history=history,
        callbacks=agent_callbacks,
        ui=PromptToolkitUI(),
        tool_callbacks=tool_callbacks,
        name="launch_orchestrator_agent",
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
    agent_callbacks: ProgressCallbacks,
    tool_callbacks: ConfirmationToolCallbacks,
):
    # Build a simple root agent for chat mode (no finish_task)
    params: list[Parameter] = []
    if instructions:
        params.append(
            Parameter(
                name="instructions",
                description="General instructions for the agent.",
                value=instructions,
            )
        )
    desc = AgentDescription(
        name="Orchestrator",
        model=config.model,
        parameters=params,
        tools=[
            CompactConversation(),
            *tools,
        ],
    )
    state = AgentState(history=history or [])
    ctx = history=state.history, model=desc.model, tools=desc.tools, parameters=desc.parameters, context_name=desc.name

    try:
        await run_chat_loop(
            history=state.history,
            model=config.model,
            tools=desc.tools,
            parameters=params,
            callbacks=agent_callbacks,
            tool_callbacks=tool_callbacks,
            completer=complete,
            ui=PromptToolkitUI(),
            context_name=desc.name,
        )
    finally:
        save_orchestrator_history(working_directory, state.history)


async def _main(args):
    logger.info(f"Starting Coding Assistant with arguments {args}")

    config = create_config_from_args(args)
    logger.info(f"Using configuration from command line arguments: {config}")

    working_directory = Path(os.getcwd())
    logger.info(f"Running in working directory: {working_directory}")

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

    # We'll assemble final instructions after initializing MCP servers, so we can include
    # any server-provided instruction banners.

    venv_directory = Path(os.environ["VIRTUAL_ENV"])
    logger.info(f"Using virtual environment directory: {venv_directory}")

    if args.sandbox:
        logger.info("Sandboxing is enabled.")

        readable_sandbox_directories = [
            *[Path(d).resolve() for d in args.readable_sandbox_directories],
            venv_directory,
        ]
        logger.info(f"Readable sandbox directories: {readable_sandbox_directories}")

        writable_sandbox_directories = [
            *[Path(d).resolve() for d in args.writable_sandbox_directories],
            working_directory,
        ]
        logger.info(f"Writable sandbox directories: {writable_sandbox_directories}")

        sandbox(readable_directories=readable_sandbox_directories, writable_directories=writable_sandbox_directories)
    else:
        logger.warning("Sandboxing is disabled")

    mcp_server_configs = [MCPServerConfig.model_validate_json(mcp_config_json) for mcp_config_json in args.mcp_servers]
    logger.info(f"Using MCP server configurations: {[s.name for s in mcp_server_configs]}")

    async with get_mcp_servers_from_config(mcp_server_configs, working_directory) as mcp_servers:
        if args.print_mcp_tools:
            await print_mcp_tools(mcp_servers)
            return

        tools = await get_mcp_wrapped_tools(mcp_servers)

        instructions = get_instructions(
            working_directory=working_directory,
            user_instructions=args.instructions,
            mcp_servers=mcp_servers,
        )

        if args.print_instructions:
            rich_print(Panel(Markdown(instructions), title="Instructions"))
            return

        agent_callbacks = DenseProgressCallbacks()

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
                callbacks=agent_callbacks,
                tool_callbacks=tool_callbacks,
            )
        else:
            if not args.task:
                raise ValueError("Task must be provided. Use --task to specify the task for the orchestrator agent.")
            await run_orchestrator_agent(
                task=args.task,
                config=config,
                tools=tools,
                history=resume_history,
                instructions=instructions,
                working_directory=working_directory,
                callbacks=agent_callbacks,
                tool_callbacks=tool_callbacks,
            )


def main():
    args = parse_args()

    if args.trace:
        enable_tracing()

    if args.wait_for_debugger:
        logger.info("Waiting for debugger to attach on port 1234")
        debugpy.listen(1234)
        debugpy.wait_for_client()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
