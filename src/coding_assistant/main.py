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

from coding_assistant.agents.logic import run_agent_loop
from coding_assistant.agents.agents import OrchestratorTool
from coding_assistant.config import Config
from coding_assistant.tools import Tools, get_all_mcp_servers


logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger("coding_assistant")
tracer = trace.get_tracer(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--research", type=str, help="Question to ask the research agent."
    )
    parser.add_argument("--task", type=str, help="Task for the orchestrator agent.")
    parser.add_argument("--expert", type=str, help="Task for the expert agent.")
    parser.add_argument(
        "-w",
        "--working_directory",
        type=Path,
        help="The working directory to use.",
        default=Path(os.getcwd()),
    )
    return parser.parse_args()


def load_config(args) -> Config:
    model_name = os.environ.get("CODING_ASSISTANT_MODEL", "o4-mini")
    expert_model_name = os.environ.get("CODING_ASSISTANT_EXPERT_MODEL", "o3")

    return Config(
        working_directory=args.working_directory,
        model=model_name,
        expert_model=expert_model_name,
    )


async def _main():
    args = parse_args()
    config = load_config(args)

    os.chdir(config.working_directory)
    logger.info(f"Running in working directory: {config.working_directory}")

    async with get_all_mcp_servers(config) as mcp_servers:
        tools = Tools(mcp_servers=mcp_servers)

        result = None
        with tracer.start_as_current_span("run_root_agent"):
            if args.task:
                tool = OrchestratorTool(config, tools)
                result = await tool.execute({"task": args.task})
            else:
                print("No task or question specified.")
                sys.exit(1)

        print(f"Finished with: {result}")


def setup_tracing():
    TRACE_ENDPOINT = "http://localhost:4318/v1/traces"

    try:
        requests.head(TRACE_ENDPOINT, timeout=0.2)
    except requests.RequestException as e:
        logger.info(
            f"Tracing endpoint {TRACE_ENDPOINT} not reachable. Tracing will be disabled. Error: {e}"
        )
        return

    resource = Resource.create(attributes={SERVICE_NAME: "coding_assistant"})
    tracerProvider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=TRACE_ENDPOINT))
    tracerProvider.add_span_processor(processor)
    trace.set_tracer_provider(tracerProvider)
    logger.info(f"Tracing successfully enabled on endpoint {TRACE_ENDPOINT}.")


def main():
    setup_tracing()
    asyncio.run(_main())


if __name__ == "__main__":
    main()
