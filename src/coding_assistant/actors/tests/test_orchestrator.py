import pytest
import asyncio
from typing import Any
from unittest.mock import MagicMock

from coding_assistant.actors.system import ActorSystem
from coding_assistant.actors.orchestrator import OrchestratorActor, OrchestratorState
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import StartTask, LLMResponse, ToolResult, ActorMessage
from coding_assistant.framework.types import AgentContext, AgentDescription, AgentState
from coding_assistant.llm.types import AssistantMessage, Completion, Usage, ToolCall, FunctionCall


@pytest.mark.asyncio
async def test_orchestrator_fsm_transitions() -> None:
    system = ActorSystem()

    # Setup context
    desc = AgentDescription(name="test", model="m", parameters=[], tools=[])
    state = AgentState()
    ctx = AgentContext(desc=desc, state=state)

    # Mock completer
    mock_completer = MagicMock()

    async def dummy_completer(*args: Any, **kwargs: Any) -> Completion:
        return Completion(message=AssistantMessage(role="assistant"), usage=Usage(tokens=0, cost=0))

    mock_completer.side_effect = dummy_completer

    orchestrator = OrchestratorActor(address="agent", system=system, context=ctx, completer=mock_completer, tools=[])
    system.register(orchestrator)
    await orchestrator.start()

    # 1. Initially IDLE
    current_state: OrchestratorState = orchestrator.state
    assert current_state == OrchestratorState.IDLE

    # 2. Send StartTask -> Reaches IDLE again because dummy_completer has no tools
    await system.send(Envelope(sender="u", recipient="agent", payload=StartTask(prompt="run")))
    await asyncio.sleep(0.5)
    current_state = orchestrator.state
    assert current_state == OrchestratorState.IDLE

    # 3. Simulate LLM suggesting a tool -> Transition to WAITING_FOR_TOOLS
    tool_call = ToolCall(id="c1", function=FunctionCall(name="t1", arguments="{}"))
    completion = Completion(
        message=AssistantMessage(role="assistant", tool_calls=[tool_call]), usage=Usage(tokens=10, cost=0)
    )

    await system.send(Envelope(sender="agent", recipient="agent", payload=LLMResponse(completion=completion)))
    await asyncio.sleep(0.2)
    current_state = orchestrator.state
    assert current_state == OrchestratorState.WAITING_FOR_TOOLS

    # 4. Simulate Tool Result -> Should trigger _trigger_llm_step and eventually reach IDLE
    payload: ActorMessage = ToolResult(result={"content": "done"})
    await system.send(Envelope(sender="worker", recipient="agent", correlation_id="c1", payload=payload))
    await asyncio.sleep(0.5)
    current_state = orchestrator.state
    assert current_state == OrchestratorState.IDLE

    await system.shutdown()
