import logging
import asyncio
from typing import Any
from coding_assistant.llm.types import (
    BaseMessage,
    ProgressCallbacks,
    NullProgressCallbacks,
    Tool,
)
from coding_assistant.framework.callbacks import (
    NullToolCallbacks,
    ToolCallbacks,
)
from coding_assistant.ui import UI
from coding_assistant.actors.system import ActorSystem
from coding_assistant.actors.orchestrator import OrchestratorActor, OrchestratorState
from coding_assistant.actors.tool_worker import ToolWorkerActor
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import StartTask
from coding_assistant.framework.types import AgentContext, AgentDescription, AgentState

logger = logging.getLogger(__name__)

CHAT_START_MESSAGE_TEMPLATE = """
## General

- You are an agent.
- You are in chat mode.
  - Use tools only when they materially advance the work.
  - When you have finished your task, reply without any tool calls to return control to the user.
  - When you want to ask the user a question, create a message without any tool calls to return control to the user.

{instructions_section}
""".strip()


def _create_chat_start_message(instructions: str | None) -> str:
    instructions_section = ""
    if instructions:
        instructions_section = f"## Instructions\n\n{instructions}"
    return CHAT_START_MESSAGE_TEMPLATE.format(
        instructions_section=instructions_section,
    )


async def run_chat_loop(
    *,
    history: list[BaseMessage],
    model: str,
    tools: list[Tool],
    instructions: str | None,
    callbacks: ProgressCallbacks = NullProgressCallbacks(),
    tool_callbacks: ToolCallbacks = NullToolCallbacks(),
    completer: Any,
    ui: UI,
    context_name: str,
    actor_system: ActorSystem | None = None,
) -> None:
    """
    Unified Actor-based Chat entry point.
    """
    should_shutdown = False
    if actor_system is None:
        actor_system = ActorSystem()
        should_shutdown = True

    # Setup dummy context for chat
    desc = AgentDescription(name=context_name, model=model, parameters=[], tools=tools)
    state = AgentState(history=history)
    ctx = AgentContext(desc=desc, state=state)

    import uuid
    run_id = str(uuid.uuid4())[:8]

    # Dispatch tools
    worker_addr = f"worker/{context_name}/{run_id}"
    tool_worker = ToolWorkerActor(worker_addr, actor_system, tools)
    actor_system.register(tool_worker)
    await tool_worker.start()

    # Register physical UI in the gateway
    from coding_assistant.actors.ui_gateway import UIGatewayActor

    ui_addr = f"ui/{context_name}/{run_id}"
    ui_gateway = UIGatewayActor(ui_addr, actor_system, ui)
    actor_system.register(ui_gateway)
    await ui_gateway.start()

    # Create unified orchestrator in chat mode
    agent_addr = f"agent/{context_name}/{run_id}"
    orchestrator = OrchestratorActor(
        address=agent_addr,
        system=actor_system,
        context=ctx,
        completer=completer,
        tools=tools,
        tool_worker_address=worker_addr,
        ui_gateway_address=ui_addr,
        is_chat_mode=True,
        instructions=instructions,
        progress_callbacks=callbacks,
        tool_callbacks=tool_callbacks,
    )
    actor_system.register(orchestrator)
    await orchestrator.start()

    # Start Chat Mode
    await actor_system.send(
        Envelope(
            sender="system", recipient=agent_addr, payload=StartTask(prompt="Enter chat mode")
        )
    )

    # Wait for session close (e.g. /exit)
    try:
        while orchestrator.state not in (OrchestratorState.COMPLETED, OrchestratorState.FAILED):
            try:
                await asyncio.wait_for(orchestrator.state_event.wait(), timeout=10.0)
                if orchestrator.last_exception:
                    raise orchestrator.last_exception
            except asyncio.TimeoutError:
                if orchestrator.state in (OrchestratorState.COMPLETED, OrchestratorState.FAILED):
                    break
                if orchestrator.state == OrchestratorState.WAITING_FOR_USER:
                    break
                logger.warning(f"Chat {context_name} wait timeout. State: {orchestrator.state}")
                break
    finally:
        if should_shutdown:
            await actor_system.shutdown()
        else:
            await orchestrator.stop()
            await tool_worker.stop()
            await ui_gateway.stop()
