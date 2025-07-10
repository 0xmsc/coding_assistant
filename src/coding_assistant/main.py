import asyncio
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import requests

from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from rich.console import Console
from rich.table import Table

from coding_assistant.agents.logic import run_agent_loop
from coding_assistant.agents.tools import OrchestratorTool
from coding_assistant.config import Config
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
    model_name = os.environ.get("CODING_ASSISTANT_MODEL", "gpt-4.1")
    expert_model_name = os.environ.get("CODING_ASSISTANT_EXPERT_MODEL", "o3")

    logger.info(f"Using model: {model_name}")
    logger.info(f"Using expert model: {expert_model_name}")

    return Config(
        working_directory=Path(os.getcwd()),
        model=model_name,
        expert_model=expert_model_name,
        disable_feedback_agent=args.disable_feedback_agent,
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


async def _main():
    setup_tracing()

    args = parse_args()
    config = load_config(args)

    logger.info(f"Running in working directory: {config.working_directory}")

    venv_directory = Path(os.environ["VIRTUAL_ENV"])
    logger.info(f"Using virtual environment directory: {venv_directory}")

    if not args.disable_sandbox:
        logger.info(f"Sandboxing is enabled.")
        sandbox(working_directory=config.working_directory, venv_directory=venv_directory)
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
                result = await tool.execute({"task": args.task})
            else:
                print("No task or question specified.")
                sys.exit(1)

        print(f"Finished with: {result}")


def main():
    asyncio.run(_main(), debug=True)


if __name__ == "__main__":
    main()
