import pytest
import asyncio
from typing import Any
from unittest.mock import MagicMock

from coding_assistant.actors.system import ActorSystem
from coding_assistant.actors.orchestrator import OrchestratorActor, OrchestratorState
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import ToolResult, ActorMessage
from coding_assistant.framework.types import AgentContext, AgentDescription, AgentState
from coding_assistant.llm.types import AssistantMessage, Completion, Usage, ToolCall, FunctionCall


@pytest.mark.asyncio
async def test_multi_orchestrator_isolation() -> None:
    system = ActorSystem()

    # Create two agents
    def create_agent(name: str) -> OrchestratorActor:
        desc = AgentDescription(name=name, model="m", parameters=[], tools=[])
        ctx = AgentContext(desc=desc, state=AgentState())

        # We need an awaitable mock for completer
        mock_completer = MagicMock()

        async def dummy_completer(*args: Any, **kwargs: Any) -> Completion:
            return Completion(message=AssistantMessage(role="assistant"), usage=Usage(tokens=0, cost=0))

        mock_completer.side_effect = dummy_completer

        orchestrator = OrchestratorActor(
            address=f"agent/{name}", system=system, context=ctx, completer=mock_completer, tools=[]
        )
        return orchestrator

    agent1 = create_agent("one")
    agent2 = create_agent("two")

    system.register(agent1)
    system.register(agent2)
    await agent1.start()
    await agent2.start()

    # 1. Simulate agent one is waiting for a tool
    tc1 = ToolCall(id="c1", function=FunctionCall(name="t1", arguments="{}"))
    agent1.state = OrchestratorState.WAITING_FOR_TOOLS
    agent1._pending_tool_calls = {"c1": tc1}

    # 2. Simulate agent two is waiting for a tool
    tc2 = ToolCall(id="c2", function=FunctionCall(name="t2", arguments="{}"))
    agent2.state = OrchestratorState.WAITING_FOR_TOOLS
    agent2._pending_tool_calls = {"c2": tc2}

    # 3. Send result for agent one
    payload: ActorMessage = ToolResult(result={"content": "result-one"})
    await system.send(Envelope(sender="worker", recipient="agent/one", correlation_id="c1", payload=payload))
    # Using a slightly longer sleep to let the async mock complete
    await asyncio.sleep(0.3)

    # Verify agent one progressed (it went WAITING -> THINKING -> IDLE because dummy returns no tools)
    assert agent1.state == OrchestratorState.IDLE
    # Verify agent two is still waiting
    assert agent2.state == OrchestratorState.WAITING_FOR_TOOLS
    assert "c2" in agent2._pending_tool_calls

    await system.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_termination_on_finish_task() -> None:
    system = ActorSystem()
    desc = AgentDescription(name="test", model="m", parameters=[], tools=[])
    ctx = AgentContext(desc=desc, state=AgentState())

    orchestrator = OrchestratorActor(address="agent", system=system, context=ctx, completer=MagicMock(), tools=[])
    system.register(orchestrator)
    await orchestrator.start()

    # Simulate a tool call result that is a FinishTaskResult
    tc = ToolCall(id="finish", function=FunctionCall(name="finish_task", arguments="{}"))
    orchestrator.state = OrchestratorState.WAITING_FOR_TOOLS
    orchestrator._pending_tool_calls = {"finish": tc}

    # Send result with finish_task markers
    payload: ActorMessage = ToolResult(result={"result": "Success!", "summary": "Done"})
    await system.send(Envelope(sender="worker", recipient="agent", correlation_id="finish", payload=payload))
    await asyncio.sleep(0.2)

    assert orchestrator.state == OrchestratorState.COMPLETED
    assert ctx.state.output is not None
    assert ctx.state.output.result == "Success!"

    await system.shutdown()
