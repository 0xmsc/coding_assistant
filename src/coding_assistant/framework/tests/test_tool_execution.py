from typing import Any, cast, Optional
import asyncio
import time
import pytest


from coding_assistant.framework.callbacks import ToolCallbacks
from coding_assistant.framework.actors.agent.actor import AgentActor
from coding_assistant.framework.actors.common.messages import HandleToolCallsRequest, HandleToolCallsResponse
from coding_assistant.framework.actors.tool_call.capabilities import register_tool_capabilities
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    FunctionCall,
    Tool,
    ToolCall,
    ToolMessage,
    ToolResult,
)
from coding_assistant.framework.tests.helpers import (
    append_tool_call_results_to_history,
    execute_tool_calls_via_messages,
    make_ui_mock,
    tool_call_actor_scope,
)
from coding_assistant.framework.types import AgentState
from coding_assistant.framework.results import FinishTaskResult, TextResult
from coding_assistant.callbacks import ConfirmationToolCallbacks
from coding_assistant.llm.types import NullProgressCallbacks


class FakeConfirmTool(Tool):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def name(self) -> str:
        return "execute_shell_command"

    def description(self) -> str:
        return "Pretend to execute a shell command"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
        self.calls.append(parameters)
        return TextResult(content=f"ran: {parameters['cmd']}")


@pytest.mark.asyncio
async def test_tool_execution_success_and_missing_tool() -> None:
    class DummyTool(Tool):
        def __init__(self, name: str, result: str):
            self._name = name
            self._result = result

        def name(self) -> str:
            return self._name

        def description(self) -> str:
            return "desc"

        def parameters(self) -> dict[str, Any]:
            return {}

        async def execute(self, parameters: dict[str, Any]) -> TextResult:
            return TextResult(content=self._result)

    tool = DummyTool("echo", "ok")
    history: list[BaseMessage] = []
    tools: list[Tool] = [tool]
    ui = make_ui_mock()

    msg_ok = AssistantMessage(tool_calls=[ToolCall(id="1", function=FunctionCall(name="echo", arguments="{}"))])
    msg_missing = AssistantMessage(tool_calls=[ToolCall(id="2", function=FunctionCall(name="missing", arguments="{}"))])

    async with tool_call_actor_scope(tools=tools, ui=ui, context_name="test") as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg_ok)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
        )
        assert history[-1] == ToolMessage(tool_call_id="1", name="echo", content="ok")

        response = await execute_tool_calls_via_messages(actor, message=msg_missing)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
        )
        assert history[-1] == ToolMessage(
            tool_call_id="2",
            name="missing",
            content="Error executing tool: Tool missing not found in available tool capabilities.",
        )


@pytest.mark.asyncio
async def test_tool_confirmation_denied_and_allowed() -> None:
    tool = FakeConfirmTool()
    tools: list[Tool] = [tool]
    history: list[BaseMessage] = []

    args_json = '{"cmd": "echo 123"}'
    expected_prompt = "Execute tool `execute_shell_command` with arguments `{'cmd': 'echo 123'}`?"

    ui = make_ui_mock(confirm_sequence=[(expected_prompt, False), (expected_prompt, True)])
    tool_callbacks = ConfirmationToolCallbacks(
        tool_confirmation_patterns=[r"^execute_shell_command"], shell_confirmation_patterns=[]
    )

    call1 = ToolCall(id="1", function=FunctionCall(name="execute_shell_command", arguments=args_json))
    msg1 = AssistantMessage(tool_calls=[call1])
    call2 = ToolCall(id="2", function=FunctionCall(name="execute_shell_command", arguments=args_json))
    msg2 = AssistantMessage(tool_calls=[call2])

    async with tool_call_actor_scope(
        tools=tools,
        ui=ui,
        context_name="test",
        tool_callbacks=tool_callbacks,
    ) as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg1)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
        )

        assert tool.calls == []  # should not run
        assert history[-1] == ToolMessage(
            tool_call_id="1", name="execute_shell_command", content="Tool execution denied."
        )

        response = await execute_tool_calls_via_messages(actor, message=msg2)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
        )

        assert tool.calls == [{"cmd": "echo 123"}]
        assert history[-1] == ToolMessage(tool_call_id="2", name="execute_shell_command", content="ran: echo 123")


@pytest.mark.asyncio
async def test_unknown_result_type_raises() -> None:
    class WeirdResult(ToolResult):
        def to_dict(self) -> Any:
            return {}

    class WeirdTool(Tool):
        def name(self) -> str:
            return "weird"

        def description(self) -> str:
            return ""

        def parameters(self) -> dict[str, Any]:
            return {}

        async def execute(self, parameters: dict[str, Any]) -> ToolResult:
            return WeirdResult()

    history: list[BaseMessage] = []
    tools: list[Tool] = [WeirdTool()]
    tool_call = ToolCall(id="1", function=FunctionCall(name="weird", arguments="{}"))
    msg = AssistantMessage(tool_calls=[tool_call])
    async with tool_call_actor_scope(tools=tools, ui=make_ui_mock(), context_name="test") as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
        )
        assert "WeirdResult" in (history[-1].content or "")


class ParallelSlowTool(Tool):
    def __init__(self, name: str, delay: float, events: list[Any]) -> None:
        self._name = name
        self._delay = delay
        self._events = events

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return f"Sleep for {self._delay}s then return its name"

    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
        self._events.append(("start", self._name, time.monotonic()))
        await asyncio.sleep(self._delay)
        self._events.append(("end", self._name, time.monotonic()))
        return TextResult(content=f"done: {self._name}")


@pytest.mark.asyncio
async def test_tool_call_actor_uses_request_scoped_tools() -> None:
    class EchoTool(Tool):
        def __init__(self, value: str, delay: float = 0.0) -> None:
            self._value = value
            self._delay = delay

        def name(self) -> str:
            return "echo"

        def description(self) -> str:
            return "Echoes a fixed value"

        def parameters(self) -> dict[str, Any]:
            return {}

        async def execute(self, parameters: dict[str, Any]) -> TextResult:
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            return TextResult(content=self._value)

    message = AssistantMessage(tool_calls=[ToolCall(id="x", function=FunctionCall(name="echo", arguments="{}"))])

    async with tool_call_actor_scope(tools=[], ui=make_ui_mock(), context_name="test") as actor:
        loop = asyncio.get_running_loop()
        future_a: asyncio.Future[HandleToolCallsResponse] = loop.create_future()
        future_b: asyncio.Future[HandleToolCallsResponse] = loop.create_future()
        assert actor._actor_directory is not None  # pyright: ignore[reportPrivateUsage]

        class ReplyActorA:
            async def send_message(self, response: HandleToolCallsResponse) -> None:
                future_a.set_result(response)

        class ReplyActorB:
            async def send_message(self, response: HandleToolCallsResponse) -> None:
                future_b.set_result(response)

        actor._actor_directory.register(  # pyright: ignore[reportPrivateUsage]
            uri="actor://test/reply/request-a",
            actor=ReplyActorA(),
        )
        actor._actor_directory.register(  # pyright: ignore[reportPrivateUsage]
            uri="actor://test/reply/request-b",
            actor=ReplyActorB(),
        )
        caps_a, actors_a = register_tool_capabilities(
            actor_directory=actor._actor_directory,  # pyright: ignore[reportPrivateUsage]
            tools=(EchoTool("from-a", delay=0.01),),
            context_name="test-a",
            uri_prefix="actor://test-a/capability",
        )
        caps_b, actors_b = register_tool_capabilities(
            actor_directory=actor._actor_directory,  # pyright: ignore[reportPrivateUsage]
            tools=(EchoTool("from-b"),),
            context_name="test-b",
            uri_prefix="actor://test-b/capability",
        )
        await asyncio.gather(
            actor.send_message(
                HandleToolCallsRequest(
                    request_id="request-a",
                    message=message,
                    reply_to_uri="actor://test/reply/request-a",
                    tool_capabilities=caps_a,
                )
            ),
            actor.send_message(
                HandleToolCallsRequest(
                    request_id="request-b",
                    message=message,
                    reply_to_uri="actor://test/reply/request-b",
                    tool_capabilities=caps_b,
                )
            ),
        )

        response_a = await future_a
        response_b = await future_b
        actor._actor_directory.unregister(uri="actor://test/reply/request-a")  # pyright: ignore[reportPrivateUsage]
        actor._actor_directory.unregister(uri="actor://test/reply/request-b")  # pyright: ignore[reportPrivateUsage]
        for cap_actor in (*actors_a, *actors_b):
            await cap_actor.stop()
        for capability in (*caps_a, *caps_b):
            actor._actor_directory.unregister(uri=capability.uri)  # pyright: ignore[reportPrivateUsage]

    assert len(response_a.results) == 1
    assert len(response_b.results) == 1
    assert isinstance(response_a.results[0].result, TextResult)
    assert isinstance(response_b.results[0].result, TextResult)
    assert response_a.results[0].result.content == "from-a"
    assert response_b.results[0].result.content == "from-b"


@pytest.mark.asyncio
async def test_tool_call_malformed_arguments_records_error() -> None:
    # Tool name can be anything; malformed JSON should short-circuit before execution attempt
    history: list[BaseMessage] = []
    tools: list[Tool] = []
    bad_args = "{bad"  # invalid JSON
    call = ToolCall(id="bad1", function=FunctionCall(name="bad_tool", arguments=bad_args))
    msg = AssistantMessage(tool_calls=[call])

    async with tool_call_actor_scope(tools=tools, ui=make_ui_mock(), context_name="test") as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
        )

    assert history, "Expected an error tool message appended to history"
    tool_msg = cast(ToolMessage, history[-1])
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

        def parameters(self) -> dict[str, Any]:
            return {}

        async def execute(self, parameters: dict[str, Any]) -> TextResult:
            self.executed = True
            raise ValueError("boom")

    tool = ErrorTool()
    history: list[BaseMessage] = []
    tools: list[Tool] = [tool]
    call = ToolCall(id="e1", function=FunctionCall(name="err_tool", arguments="{}"))
    msg = AssistantMessage(tool_calls=[call])

    async with tool_call_actor_scope(tools=tools, ui=make_ui_mock(), context_name="test") as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
        )

    # Tool execute should have been invoked (setting executed True) then error captured
    assert tool.executed is True
    tool_msg = cast(ToolMessage, history[-1])
    assert tool_msg.role == "tool"
    assert tool_msg.name == "err_tool"
    assert tool_msg.content == "Error executing tool: boom"


@pytest.mark.asyncio
async def test_shell_tool_confirmation_denied_and_allowed() -> None:
    # Simulate the special shell tool name used by confirmation logic
    class FakeShellTool(Tool):
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def name(self) -> str:
            return "shell_execute"

        def description(self) -> str:
            return "shell"

        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

        async def execute(self, parameters: dict[str, Any]) -> TextResult:
            self.calls.append(parameters)
            return TextResult(content=f"ran shell: {parameters['command']}")

    tool = FakeShellTool()
    history: list[BaseMessage] = []
    tools: list[Tool] = [tool]

    command = "rm -rf /tmp"
    args_json = '{"command": "rm -rf /tmp"}'
    expected_prompt = f"Execute shell command `{command}` for tool `shell_execute`?"
    ui = make_ui_mock(confirm_sequence=[(expected_prompt, False), (expected_prompt, True)])
    tool_callbacks = ConfirmationToolCallbacks(shell_confirmation_patterns=[r"rm -rf"], tool_confirmation_patterns=[])

    # First denied
    call1 = ToolCall(id="s1", function=FunctionCall(name="shell_execute", arguments=args_json))
    msg1 = AssistantMessage(tool_calls=[call1])
    call2 = ToolCall(id="s2", function=FunctionCall(name="shell_execute", arguments=args_json))
    msg2 = AssistantMessage(tool_calls=[call2])

    async with tool_call_actor_scope(
        tools=tools,
        ui=ui,
        context_name="test",
        tool_callbacks=tool_callbacks,
    ) as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg1)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
        )
        assert tool.calls == []
        assert history[-1] == ToolMessage(
            tool_call_id="s1", name="shell_execute", content="Shell command execution denied."
        )

        response = await execute_tool_calls_via_messages(actor, message=msg2)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
        )
        assert tool.calls == [{"command": command}]
        assert history[-1] == ToolMessage(tool_call_id="s2", name="shell_execute", content=f"ran shell: {command}")


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

        def parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {"result": {"type": "string"}, "summary": {"type": "string"}},
                "required": ["result", "summary"],
            }

        async def execute(self, parameters: dict[str, Any]) -> TextResult:
            self.executed = True
            return TextResult(content="should not run")

    finish_tool = RecordingFinishTaskTool()

    state = AgentState(history=[])
    tools: list[Tool] = [finish_tool]

    class FabricatingCallbacks(ToolCallbacks):
        async def before_tool_execution(
            self, context_name: Any, tool_call_id: Any, tool_name: Any, arguments: Any, *, ui: Any
        ) -> Optional[ToolResult]:
            if tool_name == "finish_task":
                return FinishTaskResult(result="R", summary="S")
            return None

    call = ToolCall(
        id="f1", function=FunctionCall(name="finish_task", arguments='{"result": "ignored", "summary": "ignored"}')
    )
    msg = AssistantMessage(tool_calls=[call])

    def handle_tool_result(result: ToolResult) -> str:
        if isinstance(result, FinishTaskResult):
            return AgentActor.handle_finish_task_result(result, state=state)
        if isinstance(result, TextResult):
            return result.content
        return f"Tool produced result of type {type(result).__name__}"

    async with tool_call_actor_scope(
        tools=tools,
        ui=make_ui_mock(),
        context_name="test",
        tool_callbacks=FabricatingCallbacks(),
    ) as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg)
        append_tool_call_results_to_history(
            history=state.history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
            handle_tool_result=handle_tool_result,
        )

    # Underlying tool not executed
    assert finish_tool.executed is False
    # Agent output set
    assert state.output is not None
    assert state.output.result == "R"
    assert state.output.summary == "S"
    # History appended with fabricated summary message
    assert state.history[-1] == ToolMessage(tool_call_id="f1", name="finish_task", content="Agent output set.")


@pytest.mark.asyncio
async def test_multiple_tool_calls_are_parallel() -> None:
    # Two tools with equal delays: parallel run should take ~delay, sequential would take ~2*delay
    delay = 0.2
    events: list[tuple[str, str, float]] = []
    t1 = ParallelSlowTool("slow.one", delay, events)
    t2 = ParallelSlowTool("slow.two", delay, events)

    history: list[BaseMessage] = []
    tools: list[Tool] = [t1, t2]

    msg = AssistantMessage(
        tool_calls=[
            ToolCall(id="1", function=FunctionCall(name="slow.one", arguments="{}")),
            ToolCall(id="2", function=FunctionCall(name="slow.two", arguments="{}")),
        ]
    )

    start = time.monotonic()
    msg1 = AssistantMessage(tool_calls=[msg.tool_calls[0]])
    msg2 = AssistantMessage(tool_calls=[msg.tool_calls[1]])
    async with tool_call_actor_scope(tools=tools, ui=make_ui_mock(), context_name="test") as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg1)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
        )
        response = await execute_tool_calls_via_messages(actor, message=msg2)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
        )
        # Above is sequential; now test real parallel variant with a single request.
        history = []
        events.clear()
        start = time.monotonic()
        response = await execute_tool_calls_via_messages(actor, message=msg)
        append_tool_call_results_to_history(
            history=history,
            execution_results=response.results,
            context_name="test",
            progress_callbacks=NullProgressCallbacks(),
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
    events: list[str] = []

    class FastTool(Tool):
        def name(self) -> str:
            return "fast"

        def description(self) -> str:
            return ""

        def parameters(self) -> dict[str, Any]:
            return {}

        async def execute(self, parameters: dict[str, Any]) -> TextResult:
            events.append("fast_start")
            return TextResult(content="fast_done")

    class SlowTool(Tool):
        def name(self) -> str:
            return "slow"

        def description(self) -> str:
            return ""

        def parameters(self) -> dict[str, Any]:
            return {}

        async def execute(self, parameters: dict[str, Any]) -> TextResult:
            events.append("slow_start")
            await asyncio.sleep(0.3)
            events.append("slow_done")
            return TextResult(content="slow_done")

    tools: list[Tool] = [FastTool(), SlowTool()]

    msg = AssistantMessage(
        tool_calls=[
            ToolCall(id="s", function=FunctionCall(name="slow", arguments="{}")),
            ToolCall(id="f", function=FunctionCall(name="fast", arguments="{}")),
        ]
    )

    async with tool_call_actor_scope(tools=tools, ui=make_ui_mock(), context_name="test") as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg)

    ids = [item.tool_call_id for item in response.results]
    assert "f" in ids
    assert "s" in ids
    assert ids.index("f") < ids.index("s")
