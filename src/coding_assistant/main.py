import asyncio
import json
import logging
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Optional

import requests
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from rich.panel import Panel
from rich import print as rich_print
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

from coding_assistant.tools.tools import OrchestratorTool
from coding_assistant.cache import (
    get_conversation_summaries,
    get_latest_orchestrator_history_file,
    save_conversation_summary,
    save_orchestrator_history,
    load_orchestrator_history,
    trim_orchestrator_history,
)
from coding_assistant.config import Config, MCPServerConfig
from coding_assistant.instructions import get_instructions
from coding_assistant.sandbox import sandbox
from coding_assistant.tools.mcp import get_mcp_servers_from_config, print_mcp_tools
from coding_assistant.agents.callbacks import AgentCallbacks, NullCallbacks, RichCallbacks

logging.basicConfig(level=logging.WARNING, handlers=[RichHandler()])
logger = logging.getLogger("coding_assistant")
tracer = trace.get_tracer(__name__)
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
    parser.add_argument("--model", type=str, default="gpt-4.1", help="Model to use for the orchestrator agent.")
    parser.add_argument("--expert-model", type=str, default="gpt-4.1", help="Expert model to use.")
    parser.add_argument(
        "--feedback-agent",
        action=BooleanOptionalAction,
        default=False,
        help="Enable the feedback agent.",
    )
    parser.add_argument(
        "--user-feedback",
        action=BooleanOptionalAction,
        default=True,
        help="Enable user feedback.",
    )
    parser.add_argument("--instructions", type=str, help="Custom instructions for the agent.")
    parser.add_argument(
        "--sandbox-directories",
        nargs="*",
        default=["/tmp"],
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
        help='MCP server configurations as JSON strings. Format: \'{"name": "server_name", "command": "command", "args": ["arg1", "arg2"], "env": ["ENV_VAR1", "ENV_VAR2"]}\'',
    )
    parser.add_argument(
        "--trace-endpoint",
        type=str,
        default="http://localhost:4318/v1/traces",
        help="Endpoint for OTLP trace exporter.",
    )

    return parser.parse_args()


def create_config_from_args(args) -> Config:
    sandbox_dirs = [Path(d) for d in args.sandbox_directories]

    # Expand user directories
    for i, directory in enumerate(sandbox_dirs):
        sandbox_dirs[i] = directory.expanduser()

    # Parse MCP server configurations from JSON strings
    mcp_servers = []
    for mcp_config_json in args.mcp_servers:
        config_dict = json.loads(mcp_config_json)
        mcp_server_config = MCPServerConfig(
            name=config_dict["name"],
            command=config_dict["command"],
            args=config_dict.get("args", []),
            env=config_dict.get("env", []),
        )
        mcp_servers.append(mcp_server_config)

    return Config(
        model=args.model,
        expert_model=args.expert_model,
        enable_feedback_agent=args.feedback_agent,
        enable_user_feedback=args.user_feedback,
        instructions=args.instructions,
        sandbox_directories=sandbox_dirs,
        mcp_servers=mcp_servers,
    )


def setup_tracing(args):
    TRACE_ENDPOINT = args.trace_endpoint

    try:
        requests.head(TRACE_ENDPOINT, timeout=0.2)
    except requests.RequestException as e:
        logger.info(f"Tracing endpoint {TRACE_ENDPOINT} not reachable. Tracing will be disabled. Error: {e}")
        return

    resource = Resource.create(attributes={SERVICE_NAME: "coding_assistant"})
    tracerProvider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=TRACE_ENDPOINT))
    tracerProvider.add_span_processor(processor)
    trace.set_tracer_provider(tracerProvider)
    logger.info(f"Tracing successfully enabled on endpoint {TRACE_ENDPOINT}.")


def get_additional_sandbox_directories(config: Config, working_directory: Path, venv_directory: Path):
    sandbox_directories = [
        working_directory,
        venv_directory,
    ]

    sandbox_directories.extend(config.sandbox_directories)

    return sandbox_directories


def get_resume_history(args, working_directory):
    if args.resume_file:
        if not args.resume_file.exists():
            raise FileNotFoundError(f"Resume file {args.resume_file} does not exist.")
        logger.info(f"Resuming session from file: {args.resume_file}")
        return load_orchestrator_history(args.resume_file)
    elif args.resume:
        latest_history_file = get_latest_orchestrator_history_file(working_directory)
        if not latest_history_file:
            raise FileNotFoundError(
                f"No latest orchestrator history file found in {working_directory}/.coding_assistant/history."
            )
        logger.info(f"Resuming session from latest saved agent history: {latest_history_file}")
        return load_orchestrator_history(latest_history_file)
    return None


async def run_orchestrator_agent(
    task: str,
    config: Config,
    mcp_servers: list,
    history: list | None,
    conversation_summaries: list[str],
    instructions: str | None,
    working_directory: Path,
    agent_callbacks: AgentCallbacks,
):
    with tracer.start_as_current_span("run_root_agent"):
        tool = OrchestratorTool(
            config,
            mcp_servers,
            history=history,
            agent_callbacks=agent_callbacks,
        )
        orchestrator_params = {
            "task": task,
            "summaries": conversation_summaries[-5:],
            "instructions": instructions,
        }

        try:
            result = await tool.execute(orchestrator_params)
        finally:
            save_orchestrator_history(working_directory, tool.history)

    summary = tool.summary
    save_conversation_summary(working_directory, summary)

    print(f"🎉 Final Result\n\nSummary:\n\n{summary}\n\nResult:\n\n{result.content}")
    return result


async def _main():
    args = parse_args()
    setup_tracing(args)

    config = create_config_from_args(args)
    logger.info(f"Using configuration from command line arguments: {config}")

    working_directory = Path(os.getcwd())
    logger.info(f"Running in working directory: {working_directory}")

    trim_orchestrator_history(working_directory)
    conversation_summaries = get_conversation_summaries(working_directory)

    instructions = get_instructions(working_directory, config)
    resume_history = get_resume_history(args, working_directory)

    venv_directory = Path(os.environ["VIRTUAL_ENV"])
    logger.info(f"Using virtual environment directory: {venv_directory}")

    if args.sandbox:
        logger.info(f"Sandboxing is enabled.")
        sandbox_directories = get_additional_sandbox_directories(config, working_directory, venv_directory)
        logger.info(f"Sandbox directories: {sandbox_directories}")
        sandbox(directories=sandbox_directories)
    else:
        logger.warning("Sandboxing is disabled")

    mcp_server_configs = config.mcp_servers
    logger.info(f"Using MCP server configurations: {[s.name for s in mcp_server_configs]}")

    async with get_mcp_servers_from_config(mcp_server_configs, working_directory) as mcp_servers:
        if args.print_mcp_tools:
            await print_mcp_tools(mcp_servers)
            return

        if not args.task:
            raise ValueError("Task must be provided. Use --task to specify the task for the orchestrator agent.")

        agent_callbacks = RichCallbacks()

        await run_orchestrator_agent(
            task=args.task,
            config=config,
            mcp_servers=mcp_servers,
            history=resume_history,
            conversation_summaries=conversation_summaries,
            instructions=instructions,
            working_directory=working_directory,
            agent_callbacks=agent_callbacks,
        )


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
