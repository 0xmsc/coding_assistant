import pytest

from coding_assistant.agents.callbacks import NullCallbacks
from coding_assistant.agents.execution import do_single_step, handle_tool_call, handle_tool_calls
from coding_assistant.agents.tests.helpers import FakeFunction, FakeToolCall, make_test_agent, make_ui_mock
from coding_assistant.agents.types import Agent, TextResult, Tool, ToolResult
from coding_assistant.tools.tools import FinishTaskTool, ShortenConversation
import asyncio, time


class FakeConfirmTool(Tool):
    def __init__(self):
        self.calls: list[dict] = []

    def name(self) -> str:
        return "execute_shell_command"

    def description(self) -> str:
        return "Pretend to execute a shell command"

    def parameters(self) -> dict:
        return {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}

    async def execute(self, parameters: dict) -> TextResult:
        self.calls.append(parameters)
        return TextResult(content=f"ran: {parameters['cmd']}")


@pytest.mark.asyncio
async def test_tool_confirmation_denied_and_allowed():
    tool = FakeConfirmTool()
    agent = make_test_agent(
        tools=[tool],
    )

    # Arguments will be parsed and shown as a Python dict in the confirm prompt
    args_json = '{"cmd": "echo 123"}'
    expected_prompt = "Execute tool `execute_shell_command` with arguments `{'cmd': 'echo 123'}`?"

    ui = make_ui_mock(confirm_sequence=[(expected_prompt, False), (expected_prompt, True)])

    # First: denied
    call1 = FakeToolCall(id="1", function=FakeFunction(name="execute_shell_command", arguments=args_json))
    await handle_tool_call(call1, agent, NullCallbacks(), tool_confirmation_patterns=[r"^execute_shell_command"], ui=ui)

    assert tool.calls == []  # should not run
    assert agent.history[-1] == {
        "tool_call_id": "1",
        "role": "tool",
        "name": "execute_shell_command",
        "content": "Tool execution denied.",
    }

    # Second: allowed
    call2 = FakeToolCall(id="2", function=FakeFunction(name="execute_shell_command", arguments=args_json))
    await handle_tool_call(call2, agent, NullCallbacks(), tool_confirmation_patterns=[r"^execute_shell_command"], ui=ui)

    assert tool.calls == [{"cmd": "echo 123"}]
    assert agent.history[-1] == {
        "tool_call_id": "2",
        "role": "tool",
        "name": "execute_shell_command",
        "content": "ran: echo 123",
    }


@pytest.mark.asyncio
async def test_unknown_result_type_raises():
    class WeirdResult(ToolResult):
        pass

    class WeirdTool(Tool):
        def name(self) -> str:
            return "weird"

        def description(self) -> str:
            return ""

        def parameters(self) -> dict:
            return {}

        async def execute(self, parameters: dict) -> ToolResult:
            return WeirdResult()

    agent = make_test_agent(model="TestModel", tools=[WeirdTool()])
    tool_call = FakeToolCall(id="1", function=FakeFunction(name="weird", arguments="{}"))
    with pytest.raises(KeyError, match=r"WeirdResult"):
        await handle_tool_call(tool_call, agent, NullCallbacks(), tool_confirmation_patterns=[], ui=make_ui_mock())


class ParallelSlowTool(Tool):
    def __init__(self, name: str, delay: float, events: list):
        self._name = name
        self._delay = delay
        self._events = events

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return f"Sleep for {self._delay}s then return its name"

    def parameters(self) -> dict:
        return {}

    async def execute(self, parameters: dict) -> TextResult:
        self._events.append(("start", self._name, time.monotonic()))
        await asyncio.sleep(self._delay)
        self._events.append(("end", self._name, time.monotonic()))
        return TextResult(content=f"done: {self._name}")


@pytest.mark.asyncio
async def test_multiple_tool_calls_are_parallel():
    # Two tools with equal delays: parallel run should take ~delay, sequential would take ~2*delay
    delay = 0.2
    events: list[tuple[str, str, float]] = []
    t1 = ParallelSlowTool("slow.one", delay, events)
    t2 = ParallelSlowTool("slow.two", delay, events)

    agent = make_test_agent(tools=[t1, t2])

    from coding_assistant.agents.tests.helpers import FakeMessage  # local import to avoid circulars

    msg = FakeMessage(
        tool_calls=[
            FakeToolCall(id="1", function=FakeFunction(name="slow.one", arguments="{}")),
            FakeToolCall(id="2", function=FakeFunction(name="slow.two", arguments="{}")),
        ]
    )

    start = time.monotonic()
    await handle_tool_call(msg.tool_calls[0], agent, NullCallbacks(), tool_confirmation_patterns=[], ui=make_ui_mock())
    await handle_tool_call(msg.tool_calls[1], agent, NullCallbacks(), tool_confirmation_patterns=[], ui=make_ui_mock())
    # Above would be sequential; now test real parallel variant using handle_tool_calls
    agent = make_test_agent(tools=[t1, t2])  # reset agent history
    events.clear()
    start = time.monotonic()
    await handle_tool_calls(msg, agent, NullCallbacks(), tool_confirmation_patterns=[], ui=make_ui_mock())
    elapsed = time.monotonic() - start

    # Assert total runtime significantly less than sequential (~0.4s)
    assert elapsed < delay + 0.1, f"Expected parallel execution (<~{delay+0.1:.2f}s) but took {elapsed:.2f}s"

    # Extract ordering: we expect both starts before at least one end (start1, start2, end?, end?) not start,end,start,end
    kinds = [k for (k, _, _) in events]
    # Find indices
    first_end_index = kinds.index("end")
    start_indices = [i for i, k in enumerate(kinds) if k == "start"]
    assert len(start_indices) == 2, "Both tools should have started"
    assert start_indices[1] < first_end_index, (
        "Second tool did not start before the first finished; tools likely executed sequentially. Events: " f"{events}"
    )

    # History should contain two tool messages (order may be any); validate both present
    tool_messages = [m for m in agent.history if m.get("role") == "tool"]
    names = sorted(m["name"] for m in tool_messages)
    assert names == ["slow.one", "slow.two"], f"Unexpected tool messages: {tool_messages}"
