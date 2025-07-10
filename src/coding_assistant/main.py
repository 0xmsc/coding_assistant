import asyncio
import json
import logging
import os
import sys
from argparse import ArgumentParser
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
from coding_assistant.config import Config, get_config_file_path, create_default_config_file, load_user_config, merge_config_with_defaults
from coding_assistant.instructions import get_instructions
from coding_assistant.sandbox import sandbox
from coding_assistant.tools import Tools, get_all_mcp_servers

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger("coding_assistant")
tracer = trace.get_tracer(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, help="Task for the orchestrator agent.")
    parser.add_argument("--print_mcp_tools", action="store_true", help="Print all available tools from MCP servers.")
    parser.add_argument(
        "--disable-feedback-agent", action="store_true", default=False, help="Disable the feedback agent."
    )
    parser.add_argument("--disable-sandbox", action="store_true", default=False, help="Disable sandboxing.")
    return parser.parse_args()


def load_config(args) -> Config:
    # Load user configuration file (create if missing)
    config_path = get_config_file_path()
    if not config_path.exists():
        create_default_config_file(config_path)
    user_config = load_user_config(config_path)
    
    # Load defaults from user config with hardcoded fallbacks
    model_name = user_config.get("models", {}).get("default_model", "gemini/gemini-2.5-flash")
    expert_model_name = user_config.get("models", {}).get("expert_model", "gemini/gemini-2.5-pro")

    logger.info(f"Using model: {model_name}")
    logger.info(f"Using expert model: {expert_model_name}")

    # Create base config
    base_config = Config(
        working_directory=Path(os.getcwd()),
        model=model_name,
        expert_model=expert_model_name,
        disable_feedback_agent=args.disable_feedback_agent,
    )
    
    # Merge with user configuration
    return merge_config_with_defaults(user_config, base_config)


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


def get_additional_sandbox_directories(config: Config, working_directory, venv_directory):
    sandbox_directories = [
        working_directory,
        venv_directory,
        Path("/tmp"),
        get_cache_dir(),
    ]
    
    # Add user-configured sandbox directories
    sandbox_directories.extend(config.sandbox_directories)

    wsl_path = Path("/mnt/wsl")
    if wsl_path.exists():
        sandbox_directories.append(wsl_path)

    return sandbox_directories


async def _main():
    setup_tracing()

    args = parse_args()
    config = load_config(args)

    logger.info(f"Running in working directory: {config.working_directory}")
    conversation_history = get_conversation_history(config.working_directory)

    venv_directory = Path(os.environ["VIRTUAL_ENV"])
    logger.info(f"Using virtual environment directory: {venv_directory}")

    if not args.disable_sandbox:
        logger.info(f"Sandboxing is enabled.")
        sandbox_directories = get_additional_sandbox_directories(config, config.working_directory, venv_directory)
        sandbox(directories=sandbox_directories)
    else:
        logger.warning("Sandboxing is disabled")

    async with get_all_mcp_servers(config) as mcp_servers:
        tools = Tools(mcp_servers=mcp_servers)

        if args.print_mcp_tools:
            await print_mcp_tools(mcp_servers)
            return

        result = None
        with tracer.start_as_current_span("run_root_agent"):
            if args.task:
                tool = OrchestratorTool(config, tools)
                result = await tool.execute(
                    {
                        "task": args.task,
                        "history": conversation_history[-5:],
                        "instructions": get_instructions(config.working_directory),
                    }
                )
                summary = tool.summary
            else:
                print("No task or question specified.")
                sys.exit(1)

        print(f"Finished with: {result}")
        print(f"Summary: {summary}")

        save_conversation_history(config.working_directory, summary)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
