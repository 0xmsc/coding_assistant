import argparse
import asyncio
import logging
import os
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from coding_assistant.agents.logic import do_single_step, run_agent_loop
from coding_assistant.agents.agents import create_orchestrator_agent
from coding_assistant.config import Config
from coding_assistant.tools import Tools, get_filesystem_server, get_git_server


logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger("coding_assistant")
tracer = trace.get_tracer(__name__)
logger.setLevel(logging.DEBUG)


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
    expert_model_name = os.environ.get("CODING_ASSISTANT_EXPERT_MODEL", "o1")

    return Config(
        working_directory=args.working_directory,
        model=model_name,
        expert_model=expert_model_name,
    )


@tracer.start_as_current_span("run_root_agent")
async def solve_task_using_agent(agent):
    return await run_agent_loop(agent)


async def _main():
    args = parse_args()
    config = load_config(args)

    os.chdir(config.working_directory)
    logger.info(f"Running in working directory: {config.working_directory}")

    async with (
        get_filesystem_server(config) as filesystem_server,
        get_git_server(config) as git_server,
    ):
        tools = Tools(mcp_servers=[filesystem_server, git_server])

        if args.task:
            agent = create_orchestrator_agent(args.task, config, tools)
        else:
            print("No task or question specified.")
            sys.exit(1)

        result = await solve_task_using_agent(agent)
        print("Finished with: ", result)


def setup_tracing():
    TRACE_ENDPOINT = "http://localhost:4318/v1/traces"
    resource = Resource.create(attributes={SERVICE_NAME: "coding_assistant"})
    tracerProvider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=TRACE_ENDPOINT))
    tracerProvider.add_span_processor(processor)
    trace.set_tracer_provider(tracerProvider)


def main():
    setup_tracing()
    asyncio.run(_main())


if __name__ == "__main__":
    main()
