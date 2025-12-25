from typing import cast
import asyncio
import time
import pytest

from coding_assistant.framework.callbacks import (
    ToolCallbacks,
    NullProgressCallbacks,
    NullToolCallbacks,
)
from coding_assistant.framework.execution import handle_tool_calls, execute_tool_call
from coding_assistant.framework.agent import _handle_finish_task_result
from coding_assistant.llm.types import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
    ToolMessage,
)
from coding_assistant.framework.tests.helpers import (
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.framework.types import (
    FinishTaskResult,
    TextResult,
    Tool,
    ToolResult,
)
from coding_assistant.callbacks import ConfirmationToolCallbacks


class FakeConfirmTool(Tool):
    def __init__(self) -> None:
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
async def test_execute_tool_call_regular_tool_and_not_found():
    class DummyTool(Tool):
        def __init__(self, name: str, result: str):
            self._name = name
            self._result = result

        def name(self) -> str:
            return self._name

        def description(self) -> str:
            return "desc"

        def parameters(self) -> dict:
            return {}

        async def execute(self, parameters: dict) -> TextResult:
            return TextResult(content=self._result)

    tool = DummyTool("echo", "ok")

    res = await execute_tool_call("echo", {}, tools=[tool])
    assert isinstance(res, TextResult)
    assert res.content == "ok"

    with pytest.raises(ValueError, match="Tool missing not found"):
        await execute_tool_call("missing", {}, tools=[tool])


@pytest.mark.asyncio
async def test_tool_confirmation_denied_and_allowed() -> None:
    tool = FakeConfirmTool()
    desc, state = make_test_agent(
        tools=[tool],
    )

    args_json = '{"cmd": "echo 123"}'
    expected_prompt = "Execute tool `execute_shell_command` with arguments `{'cmd': 'echo 123'}`?"

    ui = make_ui_mock(confirm_sequence=[(expected_prompt, False), (expected_prompt, True)])

    call1 = ToolCall(id="1", function=FunctionCall(name="execute_shell_command", arguments=args_json))
    msg1 = AssistantMessage(tool_calls=[call1])
    await handle_tool_calls(
        msg1,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        tool_callbacks=ConfirmationToolCallbacks(
            tool_confirmation_patterns=[r"^execute_shell_command"], shell_confirmation_patterns=[]
        ),
        ui=ui,
        context_name=desc.name,
    )

    assert tool.calls == []  # should not run
    assert state.history[-1] == ToolMessage(
        tool_call_id="1",
        name="execute_shell_command",
        content="Tool execution denied.",
    )

    call2 = ToolCall(id="2", function=FunctionCall(name="execute_shell_command", arguments=args_json))
    msg2 = AssistantMessage(tool_calls=[call2])
    await handle_tool_calls(
        msg2,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        tool_callbacks=ConfirmationToolCallbacks(
            tool_confirmation_patterns=[r"^execute_shell_command"], shell_confirmation_patterns=[]
        ),
        ui=ui,
        context_name=desc.name,
    )

    assert tool.calls == [{"cmd": "echo 123"}]
    assert state.history[-1] == ToolMessage(
        tool_call_id="2",
        name="execute_shell_command",
        content="ran: echo 123",
    )


@pytest.mark.asyncio
async def test_unknown_result_type_raises() -> None:
    class WeirdResult(ToolResult):
        def to_dict(self):
            return {}

    class WeirdTool(Tool):
        def name(self) -> str:
            return "weird"

        def description(self) -> str:
            return ""

        def parameters(self) -> dict:
            return {}

        async def execute(self, parameters: dict) -> ToolResult:
            return WeirdResult()

    desc, state = make_test_agent(model="TestModel", tools=[WeirdTool()])
    tool_call = ToolCall(id="1", function=FunctionCall(name="weird", arguments="{}"))
    msg = AssistantMessage(tool_calls=[tool_call])
    await handle_tool_calls(
        msg,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        NullToolCallbacks(),
        ui=make_ui_mock(),
        context_name=desc.name,
    )
    assert "WeirdResult" in (state.history[-1].content or "")


class ParallelSlowTool(Tool):
    def __init__(self, name: str, delay: float, events: list) -> None:
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
async def test_tool_call_malformed_arguments_records_error() -> None:
    # Tool name can be anything; malformed JSON should short-circuit before execution attempt
    desc, state = make_test_agent()
    bad_args = "{bad"  # invalid JSON
    call = ToolCall(id="bad1", function=FunctionCall(name="bad_tool", arguments=bad_args))
    msg = AssistantMessage(tool_calls=[call])

    await handle_tool_calls(
        msg,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        ui=make_ui_mock(),
        context_name=desc.name,
    )

    assert state.history, "Expected an error tool message appended to history"
    tool_msg = cast(ToolMessage, state.history[-1])
    assert tool_msg.role == "tool"
    assert tool_msg.name == "bad_tool"
    assert tool_msg.tool_call_id == "bad1"
    assert tool_msg.content.startswith("Error: Tool call arguments `{bad` are not valid JSON:"), tool_msg.content


@pytest.mark.asyncio
async def test_tool_execution_value_error_records_error() -> None:
    class ErrorTool(Tool):
        def __init__(self) -> None:
            self.executed = False

        def name(self) -> str:
            return "err_tool"

        def description(self) -> str:
            return "raises"

        def parameters(self) -> dict:
            return {}

        async def execute(self, parameters: dict) -> TextResult:
            self.executed = True
            raise ValueError("boom")

    tool = ErrorTool()
    desc, state = make_test_agent(tools=[tool])
    call = ToolCall(id="e1", function=FunctionCall(name="err_tool", arguments="{}"))
    msg = AssistantMessage(tool_calls=[call])

    await handle_tool_calls(
        msg,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        ui=make_ui_mock(),
        context_name=desc.name,
    )

    # Tool execute should have been invoked (setting executed True) then error captured
    assert tool.executed is True
    tool_msg = cast(ToolMessage, state.history[-1])
    assert tool_msg.role == "tool"
    assert tool_msg.name == "err_tool"
    assert tool_msg.content == "Error executing tool: boom"


@pytest.mark.asyncio
async def test_shell_tool_confirmation_denied_and_allowed() -> None:
    # Simulate the special shell tool name used by confirmation logic
    class FakeShellTool(Tool):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def name(self) -> str:
            return "shell_execute"

        def description(self) -> str:
            return "shell"

        def parameters(self) -> dict:
            return {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

        async def execute(self, parameters: dict) -> TextResult:
            self.calls.append(parameters)
            return TextResult(content=f"ran shell: {parameters['command']}")

    tool = FakeShellTool()
    desc, state = make_test_agent(tools=[tool])

    command = "rm -rf /tmp"
    args_json = '{"command": "rm -rf /tmp"}'
    expected_prompt = f"Execute shell command `{command}` for tool `shell_execute`?"
    ui = make_ui_mock(confirm_sequence=[(expected_prompt, False), (expected_prompt, True)])

    # First denied
    call1 = ToolCall(id="s1", function=FunctionCall(name="shell_execute", arguments=args_json))
    msg1 = AssistantMessage(tool_calls=[call1])

    await handle_tool_calls(
        msg1,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        tool_callbacks=ConfirmationToolCallbacks(
            shell_confirmation_patterns=[r"rm -rf"], tool_confirmation_patterns=[]
        ),
        ui=ui,
        context_name=desc.name,
    )
    assert tool.calls == []
    assert state.history[-1] == ToolMessage(
        tool_call_id="s1",
        name="shell_execute",
        content="Shell command execution denied.",
    )

    # Then allowed
    call2 = ToolCall(id="s2", function=FunctionCall(name="shell_execute", arguments=args_json))
    msg2 = AssistantMessage(tool_calls=[call2])
    await handle_tool_calls(
        msg2,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        tool_callbacks=ConfirmationToolCallbacks(
            shell_confirmation_patterns=[r"rm -rf"], tool_confirmation_patterns=[]
        ),
        ui=ui,
        context_name=desc.name,
    )
    assert tool.calls == [{"command": command}]
    assert state.history[-1] == ToolMessage(
        tool_call_id="s2",
        name="shell_execute",
        content=f"ran shell: {command}",
    )


@pytest.mark.asyncio
async def test_before_tool_execution_can_return_finish_task_result() -> None:
    # Callback should fabricate a FinishTaskResult and prevent underlying tool execution
    class RecordingFinishTaskTool(Tool):
        def __init__(self) -> None:
            self.executed = False

        def name(self) -> str:
            return "finish_task"

        def description(self) -> str:
            return "finish"

        def parameters(self) -> dict:
            return {
                "type": "object",
                "properties": {"result": {"type": "string"}, "summary": {"type": "string"}},
                "required": ["result", "summary"],
            }

        async def execute(self, parameters: dict) -> TextResult:
            self.executed = True
            return TextResult(content="should not run")

    finish_tool = RecordingFinishTaskTool()
    desc, state = make_test_agent(tools=[finish_tool])

    class FabricatingCallbacks(ToolCallbacks):
        async def before_tool_execution(self, context_name, tool_call_id, tool_name, arguments, *, ui):
            if tool_name == "finish_task":
                return FinishTaskResult(result="R", summary="S")
            return None

    call = ToolCall(
        id="f1", function=FunctionCall(name="finish_task", arguments='{"result": "ignored", "summary": "ignored"}')
    )
    msg = AssistantMessage(tool_calls=[call])

    def handle_tool_result(result: ToolResult) -> str:
        if isinstance(result, FinishTaskResult):
            return _handle_finish_task_result(result, state)
        if isinstance(result, TextResult):
            return result.content
        return f"Tool produced result of type {type(result).__name__}"

    await handle_tool_calls(
        msg,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        tool_callbacks=FabricatingCallbacks(),
        ui=make_ui_mock(),
        context_name=desc.name,
        handle_tool_result=handle_tool_result,
    )

    # Underlying tool not executed
    assert finish_tool.executed is False
    # Agent output set
    assert state.output is not None
    assert state.output.result == "R"
    assert state.output.summary == "S"
    # History appended with fabricated summary message
    assert state.history[-1] == ToolMessage(
        tool_call_id="f1",
        name="finish_task",
        content="Agent output set.",
    )


@pytest.mark.asyncio
async def test_multiple_tool_calls_are_parallel() -> None:
    # Two tools with equal delays: parallel run should take ~delay, sequential would take ~2*delay
    delay = 0.2
    events: list[tuple[str, str, float]] = []
    t1 = ParallelSlowTool("slow.one", delay, events)
    t2 = ParallelSlowTool("slow.two", delay, events)

    desc, state = make_test_agent(tools=[t1, t2])

    msg = AssistantMessage(
        tool_calls=[
            ToolCall(id="1", function=FunctionCall(name="slow.one", arguments="{}")),
            ToolCall(id="2", function=FunctionCall(name="slow.two", arguments="{}")),
        ],
    )

    start = time.monotonic()
    msg1 = AssistantMessage(tool_calls=[msg.tool_calls[0]])
    await handle_tool_calls(
        msg1,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        ui=make_ui_mock(),
        context_name=desc.name,
    )
    msg2 = AssistantMessage(tool_calls=[msg.tool_calls[1]])
    await handle_tool_calls(
        msg2,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        ui=make_ui_mock(),
        context_name=desc.name,
    )
    # Above would be sequential; now test real parallel variant using handle_tool_calls
    desc, state = make_test_agent(tools=[t1, t2])  # reset agent history
    events.clear()
    start = time.monotonic()
    await handle_tool_calls(
        msg,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        ui=make_ui_mock(),
        context_name=desc.name,
    )
    elapsed = time.monotonic() - start

    # Assert total runtime significantly less than sequential (~0.4s)
    assert elapsed < delay + 0.1, f"Expected parallel execution (<~{delay + 0.1:.2f}s) but took {elapsed:.2f}s"

    # Extract ordering: we expect both starts before at least one end (start1, start2, end?, end?) not start,end,start,end
    kinds = [k for (k, _, _) in events]
    # Find indices
    first_end_index = kinds.index("end")
    start_indices = [i for i, k in enumerate(kinds) if k == "start"]
    assert len(start_indices) == 2, "Both tools should have started"
    assert start_indices[1] < first_end_index, (
        f"Second tool did not start before the first finished; tools likely executed sequentially. Events: {events}"
    )

@pytest.mark.asyncio
async def test_tool_calls_process_as_they_arrive() -> None:
    # t1 completes quickly; t2 takes longer.
    # We check history immediately after first tool finishes (via a pseudo-tool or concurrent inspection).
    events: list[str] = []

    class FastTool(Tool):
        def name(self) -> str:
            return "fast"

        def description(self) -> str:
            return ""

        def parameters(self) -> dict:
            return {}

        async def execute(self, parameters: dict) -> TextResult:
            events.append("fast_start")
            return TextResult(content="fast_done")

    class SlowTool(Tool):
        def name(self) -> str:
            return "slow"

        def description(self) -> str:
            return ""

        def parameters(self) -> dict:
            return {}

        async def execute(self, parameters: dict) -> TextResult:
            events.append("slow_start")
            await asyncio.sleep(0.3)
            events.append("slow_done")
            return TextResult(content="slow_done")

    desc, state = make_test_agent(tools=[FastTool(), SlowTool()])

    msg = AssistantMessage(
        tool_calls=[
            ToolCall(id="s", function=FunctionCall(name="slow", arguments="{}")),
            ToolCall(id="f", function=FunctionCall(name="fast", arguments="{}")),
        ],
    )

    # We want to verify that the "fast" result is in history while "slow" is still running.
    # We can use a task_created_callback to get the tasks.
    slow_task = None

    def task_created(call_id, task):
        nonlocal slow_task
        if call_id == "s":
            slow_task = task

    async def checker():
        # Wait until fast tool is likely done but slow tool is still sleeping
        # We know fast tool has no sleep, so it should be done very quickly.
        # We'll poll history until "fast" result appears.
        for _ in range(10):
            if any(getattr(m, "tool_call_id", None) == "f" for m in state.history):
                # Verify that slow tool is NOT done yet
                assert not any(getattr(m, "tool_call_id", None) == "s" for m in state.history)
                events.append("check_passed")
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("Fast tool result never appeared in history while slow tool was running")

    # Run everything
    await asyncio.gather(
        handle_tool_calls(
            msg,
            desc.tools,
            state.history,
            NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            ui=make_ui_mock(),
            context_name=desc.name,
        ),
        checker(),
    )

    assert "check_passed" in events
    # Finally both should be in history
    ids = [getattr(m, "tool_call_id", None) for m in state.history if m.role == "tool"]
    assert "f" in ids
    assert "s" in ids
