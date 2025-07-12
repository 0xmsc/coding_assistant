import asyncio
import json
import logging
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import requests
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from rich.console import Console
from rich.table import Table

from coding_assistant.agents.agents import OrchestratorTool
from coding_assistant.agents.logic import run_agent_loop
from coding_assistant.cache import get_cache_dir, get_conversation_history, save_conversation_history
from coding_assistant.config import Config, MCPServerConfig
from coding_assistant.instructions import get_instructions
from coding_assistant.sandbox import sandbox
from coding_assistant.tools import Tools, get_mcp_servers_from_config

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger("coding_assistant")
tracer = trace.get_tracer(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description="Coding Assistant CLI")
    parser.add_argument("--task", type=str, help="Task for the orchestrator agent.")
    parser.add_argument("--print-mcp-tools", action="store_true", help="Print all available tools from MCP servers.")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="Model to use for the orchestrator agent.")
    parser.add_argument("--expert-model", type=str, default="o3", help="Expert model to use.")
    parser.add_argument(
        "--disable-feedback-agent", action="store_true", default=False, help="Disable the feedback agent."
    )
    parser.add_argument("--disable-user-feedback", action="store_true", default=False, help="Disable user feedback.")
    parser.add_argument("--instructions", type=str, help="Custom instructions for the agent.")
    parser.add_argument(
        "--sandbox-directories",
        nargs="*",
        default=["/tmp"],
        help="Additional directories to include in the sandbox (default: /tmp).",
    )
    parser.add_argument("--disable-sandbox", action="store_true", default=False, help="Disable sandboxing.")
    parser.add_argument(
        "--mcp-servers",
        nargs="*",
        default=[],
        help='MCP server configurations as JSON strings. Format: \'{"name": "server_name", "command": "command", "args": ["arg1", "arg2"], "env": ["ENV_VAR1", "ENV_VAR2"]}\'',
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
        try:
            config_dict = json.loads(mcp_config_json)
            mcp_server_config = MCPServerConfig(
                name=config_dict["name"],
                command=config_dict["command"],
                args=config_dict.get("args", []),
                env=config_dict.get("env", []),  # Now expects a list of env var keys
            )
            mcp_servers.append(mcp_server_config)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Invalid MCP server configuration: {mcp_config_json}. Error: {e}")
            sys.exit(1)

    return Config(
        model=args.model,
        expert_model=args.expert_model,
        disable_feedback_agent=args.disable_feedback_agent,
        disable_user_feedback=args.disable_user_feedback,
        instructions=args.instructions,
        sandbox_directories=sandbox_dirs,
        mcp_servers=mcp_servers,
    )


async def print_mcp_tools(mcp_servers):
    console = Console()

    if not mcp_servers:
        console.print("[yellow]No MCP servers found.[/yellow]")
        return

    table = Table(show_header=True, show_lines=True)
    table.add_column("Server Name", style="magenta")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Parameters", style="yellow")

    for server in mcp_servers:
        tools_response = await server.session.list_tools()
        server_tools = tools_response.tools

        if not server_tools:
            logger.info(f"No tools found for MCP server: {server.name}")
            continue

        for tool in server_tools:
            name = tool.name
            description = tool.description
            parameters = tool.inputSchema
            parameters_str = ", ".join(parameters.get("properties", {}).keys()) if parameters else "None"
            table.add_row(server.name, name, description, parameters_str)

    console.print(table)


def setup_tracing():
    TRACE_ENDPOINT = "http://localhost:4318/v1/traces"

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
        get_cache_dir(),
    ]

    sandbox_directories.extend(config.sandbox_directories)

    return sandbox_directories


async def _main():
    setup_tracing()

    args = parse_args()

    config = create_config_from_args(args)
    logger.info(f"Using configuration from command line arguments: {config}")

    working_directory = Path(os.getcwd())
    logger.info(f"Running in working directory: {working_directory}")
    conversation_history = get_conversation_history(working_directory)

    venv_directory = Path(os.environ["VIRTUAL_ENV"])
    logger.info(f"Using virtual environment directory: {venv_directory}")

    if not args.disable_sandbox:
        logger.info(f"Sandboxing is enabled.")
        sandbox_directories = get_additional_sandbox_directories(config, working_directory, venv_directory)
        logger.info(f"Sandbox directories: {sandbox_directories}")
        sandbox(directories=sandbox_directories)
    else:
        logger.warning("Sandboxing is disabled")

    # Use configured MCP servers
    mcp_server_configs = config.mcp_servers
    logger.info(f"Using MCP server configurations: {[s.name for s in mcp_server_configs]}")

    async with get_mcp_servers_from_config(mcp_server_configs, working_directory) as mcp_servers:
        tools = Tools(mcp_servers=mcp_servers)

        if args.print_mcp_tools:
            await print_mcp_tools(mcp_servers)
            return

        result = None
        with tracer.start_as_current_span("run_root_agent"):
            tool = OrchestratorTool(config, tools)
            result = await tool.execute(
                {
                    "task": args.task,
                    "history": conversation_history[-5:],
                    "instructions": get_instructions(working_directory, config),
                }
            )
            summary = tool.summary

        print(f"Finished with: {result}")
        print(f"Summary: {summary}")

        save_conversation_history(working_directory, summary)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
