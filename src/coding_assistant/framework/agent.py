import logging
import asyncio
from typing import Any

from coding_assistant.framework.builtin_tools import (
    CompactConversationTool,
    FinishTaskTool,
)
from coding_assistant.llm.types import (
    NullProgressCallbacks,
    ProgressCallbacks,
)
from coding_assistant.framework.callbacks import (
    NullToolCallbacks,
    ToolCallbacks,
)
from coding_assistant.framework.types import (
    AgentContext,
    AgentOutput,
)
from coding_assistant.ui import UI
from coding_assistant.actors.system import ActorSystem
from coding_assistant.actors.orchestrator import OrchestratorActor, OrchestratorState
from coding_assistant.actors.tool_worker import ToolWorkerActor
from coding_assistant.actors.ui_gateway import UIGatewayActor
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import StartTask

logger = logging.getLogger(__name__)

# Start Template remains for the Actor to use
START_MESSAGE_TEMPLATE = """
## General

- You are an agent named `{name}`.
- You are given a set of parameters by your client, among which are your task and your description.
  - It is of the utmost importance that you try your best to fulfill the task as specified.
  - The task shall be done in a way which fits your description.
- You must use at least one tool call in every step.
  - Use the `finish_task` tool when you have fully finished your task, no questions should still be open.

## Parameters

Your client has provided the following parameters for your task:

{parameters}
""".strip()


def _create_start_message(*, desc: Any) -> str:
    from coding_assistant.framework.parameters import format_parameters

    parameters_str = format_parameters(desc.parameters)
    return START_MESSAGE_TEMPLATE.format(
        name=desc.name,
        parameters=parameters_str,
    )


def _handle_finish_task_result(result: Any, *, state: Any) -> str:
    state.output = AgentOutput(result=result.result, summary=result.summary)
    return "Agent output set."


def _handle_compact_conversation_result(
    result: Any,
    *,
    desc: Any,
    state: Any,
    progress_callbacks: Any,
    force: bool = True,
) -> str:
    from coding_assistant.framework.history import clear_history, append_user_message
    from coding_assistant.llm.types import UserMessage

    clear_history(state.history)

    user_msg = UserMessage(
        content=f"A summary of your conversation with the client until now:\n\n{result.summary}\n\nPlease continue your work."
    )
    append_user_message(
        state.history,
        callbacks=progress_callbacks,
        context_name=desc.name,
        message=user_msg,
        force=force,
    )

    return "Conversation compacted and history reset."


async def run_agent_loop(
    ctx: AgentContext,
    *,
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
    tool_callbacks: ToolCallbacks = NullToolCallbacks(),
    completer: Any,
    ui: UI,
    actor_system: ActorSystem | None = None,
    compact_conversation_at_tokens: int = 200_000,
) -> None:
    """
    Unified Actor-based Agent entry point.
    """
    if ctx.state.output is not None:
        raise RuntimeError("Agent already has a result or summary.")

    should_shutdown = False
    if actor_system is None:
        actor_system = ActorSystem()
        should_shutdown = True

    tools = list(ctx.desc.tools)
    if not any(tool.name() == "finish_task" for tool in tools):
        tools.append(FinishTaskTool())
    if not any(tool.name() == "compact_conversation" for tool in tools):
        tools.append(CompactConversationTool())

    import uuid

    run_id = str(uuid.uuid4())[:8]

    # Spawn Tool Worker for this agent
    worker_addr = f"worker/{ctx.desc.name}/{run_id}"
    tool_worker = ToolWorkerActor(worker_addr, actor_system, tools)
    actor_system.register(tool_worker)
    await tool_worker.start()

    # Spawn UI Gateway
    ui_addr = f"ui/{ctx.desc.name}/{run_id}"
    ui_gateway = UIGatewayActor(ui_addr, actor_system, ui)
    actor_system.register(ui_gateway)
    await ui_gateway.start()

    # Spawn Orchestrator
    agent_addr = f"agent/{ctx.desc.name}/{run_id}"
    orchestrator = OrchestratorActor(
        address=agent_addr,
        system=actor_system,
        context=ctx,
        completer=completer,
        tools=tools,
        tool_worker_address=worker_addr,
        ui_gateway_address=ui_addr,
        is_chat_mode=False,
        compact_conversation_at_tokens=compact_conversation_at_tokens,
        progress_callbacks=progress_callbacks,
        tool_callbacks=tool_callbacks,
    )
    actor_system.register(orchestrator)
    await orchestrator.start()

    # Start Task

    await actor_system.send(
        Envelope(
            sender="system",
            recipient=agent_addr,
            payload=StartTask(prompt="Start autonomous task"),
        )
    )

    # Wait for completion
    try:
        while orchestrator.state not in (OrchestratorState.COMPLETED, OrchestratorState.FAILED):
            try:
                await asyncio.wait_for(orchestrator.state_event.wait(), timeout=10.0)
                if orchestrator.last_exception:
                    raise orchestrator.last_exception
            except asyncio.TimeoutError:
                if orchestrator.state in (OrchestratorState.COMPLETED, OrchestratorState.FAILED):
                    break
                logger.warning(f"Agent {ctx.desc.name} wait timeout. State: {orchestrator.state}")
                break
    finally:
        if should_shutdown:
            await actor_system.shutdown()
        else:
            await orchestrator.stop()
            await tool_worker.stop()
            await ui_gateway.stop()
