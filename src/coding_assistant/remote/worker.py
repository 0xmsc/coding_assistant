from __future__ import annotations

import asyncio
from argparse import Namespace

from rich import print

from coding_assistant.app.default_agent import (
    build_default_agent_config,
    build_initial_system_message,
    create_default_agent,
)
from coding_assistant.app.output import run_session_output
from coding_assistant.core.agent_session import AgentSession
from coding_assistant.integrations.mcp_client import print_mcp_tools
from coding_assistant.remote.server import start_worker_server


async def run_worker(args: Namespace) -> None:
    """Run the process as a remote-controlled worker."""
    config = build_default_agent_config(args)
    async with create_default_agent(config=config) as bundle:
        if args.print_mcp_tools:
            await print_mcp_tools(bundle.mcp_servers)
            return

        system_message = build_initial_system_message(instructions=bundle.instructions)
        session = AgentSession(
            history=[system_message],
            model=args.model,
            tools=bundle.tools,
        )
        output_task = asyncio.create_task(
            run_session_output(session=session, system_message=system_message, show_state_updates=True)
        )
        # Let the renderer subscribe before the worker starts accepting prompts.
        await asyncio.sleep(0)
        try:
            async with start_worker_server(session=session) as worker_server:
                print(f"Worker endpoint: {worker_server.endpoint}")
                await asyncio.Future()
        finally:
            await session.close()
            output_task.cancel()
            await asyncio.gather(output_task, return_exceptions=True)
