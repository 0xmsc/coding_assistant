from __future__ import annotations

from argparse import Namespace

from rich import print

from coding_assistant.app.default_agent import (
    build_default_agent_config,
    build_initial_system_message,
    create_default_agent,
)
from coding_assistant.app.image import get_image
from coding_assistant.app.terminal_ui import PromptSubmitType, run_terminal_ui
from coding_assistant.core.agent_session import AgentSession
from coding_assistant.infra.paths import get_app_cache_dir
from coding_assistant.integrations.mcp_client import print_mcp_tools
from coding_assistant.remote.registry import register_remote_instance
from coding_assistant.remote.server import start_worker_server

CLI_COMMAND_NAMES = ["/exit", "/help", "/compact", "/image", "/priority", "/interrupt"]


async def run_cli(args: Namespace) -> None:
    """Run the interactive CLI entry point."""
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
        try:
            async with start_worker_server(session=session) as worker_server:
                async with register_remote_instance(endpoint=worker_server.endpoint):
                    print(f"Remote endpoint: {worker_server.endpoint}")
                    await run_terminal_ui(
                        session=session,
                        system_message=system_message,
                        history_path=get_app_cache_dir() / "history",
                        words=CLI_COMMAND_NAMES,
                        submit_handler=lambda answer, submit_type: _handle_prompt_submission(
                            session=session,
                            answer=answer,
                            submit_type=submit_type,
                        ),
                    )
        finally:
            await session.close()


async def _handle_prompt_submission(*, session: AgentSession, answer: str, submit_type: PromptSubmitType) -> bool:
    """Handle one prompt line and return true when the CLI should exit."""
    stripped = answer.strip()
    if stripped == "/exit":
        return True
    if stripped == "/help":
        print(
            "Available commands:\n"
            "  /exit\n"
            "  /help\n"
            "  /compact\n"
            "  /image <path-or-url>\n"
            "  /priority <prompt>\n"
            "  /interrupt <prompt>",
        )
        return False
    if stripped == "/compact":
        return not await session.enqueue_prompt(
            "Immediately compact our conversation so far by using the `compact_conversation` tool.",
        )
    if stripped.startswith("/image"):
        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            print("/image requires a path or URL argument.")
            return False
        data_url = await get_image(parts[1])
        image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
        return not await session.enqueue_prompt(image_content)
    if stripped.startswith("/priority"):
        priority_prompt = _extract_command_argument(answer=answer, command="/priority")
        if priority_prompt is None:
            print("/priority requires prompt text.")
            return False
        return not await session.enqueue_prompt(priority_prompt, priority=True)
    if stripped.startswith("/interrupt"):
        interrupt_prompt = _extract_command_argument(answer=answer, command="/interrupt")
        if interrupt_prompt is None:
            print("/interrupt requires prompt text.")
            return False
        return not await session.interrupt_and_enqueue(interrupt_prompt)

    # Handle steering vs queued submission types
    if submit_type == PromptSubmitType.STEERING:
        # Steering: inject this into the active agent loop at the next boundary.
        return not await session.enqueue_steering_prompt(answer)
    else:
        # Queued: always add to the queue
        return not await session.enqueue_prompt(answer)


def _extract_command_argument(*, answer: str, command: str) -> str | None:
    """Return the trailing argument text for one slash command."""
    if not answer.startswith(command):
        return None
    argument = answer[len(command) :].strip()
    if not argument:
        return None
    return argument
