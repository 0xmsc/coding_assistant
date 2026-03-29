from __future__ import annotations

import asyncio
from argparse import Namespace

from rich import print

from coding_assistant.app.default_agent import (
    build_default_agent_config,
    build_initial_system_message,
    create_default_agent,
)
from coding_assistant.app.output import DeltaRenderer, print_system_message, print_tool_calls
from coding_assistant.integrations.mcp_client import print_mcp_tools
from coding_assistant.llm.types import (
    CompletionEvent,
    ContentDeltaEvent,
    ReasoningDeltaEvent,
    StatusEvent,
    SystemMessage,
)
from coding_assistant.remote.server import start_worker_server
from coding_assistant.remote.worker_session import (
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    StateChangedEvent,
    ToolCallsEvent,
    WorkerSession,
)


async def _run_worker_output(*, session: WorkerSession, system_message: SystemMessage) -> None:
    """Render worker output locally without accepting local prompts."""

    renderer = DeltaRenderer()
    print_system_message(system_message)

    async with session.subscribe() as queue:
        try:
            while True:
                event = await queue.get()
                if isinstance(event, ContentDeltaEvent):
                    renderer.on_delta(event.content)
                    continue
                if isinstance(event, ToolCallsEvent):
                    renderer.finish(trailing_blank_line=False)
                    print_tool_calls(event.message)
                    continue
                if isinstance(event, RunFinishedEvent):
                    renderer.finish()
                    continue
                if isinstance(event, RunCancelledEvent):
                    renderer.finish()
                    continue
                if isinstance(event, RunFailedEvent):
                    renderer.finish()
                    print(f"[bold red]Run failed:[/bold red] {event.error}")
                    continue
                if isinstance(event, (StateChangedEvent, ReasoningDeltaEvent, StatusEvent, CompletionEvent)):
                    continue
        finally:
            renderer.finish()


async def run_worker(args: Namespace) -> None:
    """Run the process as a remote-controlled worker."""
    config = build_default_agent_config(args)
    async with create_default_agent(config=config, include_worker_tools=False) as bundle:
        if args.print_mcp_tools:
            await print_mcp_tools(bundle.mcp_servers)
            return

        system_message = build_initial_system_message(instructions=bundle.instructions)
        session = WorkerSession(
            history=[system_message],
            model=args.model,
            tools=bundle.tools,
        )
        output_task = asyncio.create_task(_run_worker_output(session=session, system_message=system_message))
        # Let the renderer subscribe before the worker starts accepting prompts.
        await asyncio.sleep(0)
        try:
            async with start_worker_server(session=session) as worker_server:
                print(f"Worker endpoint: {worker_server.endpoint}")
                await asyncio.Future()
        finally:
            output_task.cancel()
            await asyncio.gather(output_task, return_exceptions=True)
