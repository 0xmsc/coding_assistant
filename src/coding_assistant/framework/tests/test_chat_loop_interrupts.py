import asyncio
import json
import os
import signal
import pytest
from unittest.mock import patch

from coding_assistant.framework.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.framework.chat import run_chat_loop
from coding_assistant.framework.interrupts import InterruptController
from coding_assistant.llm.types import UserMessage
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    FunctionCall,
    FakeMessage,
    ToolCall,
    make_ui_mock,
)
from coding_assistant.framework.types import TextResult, Tool


class InterruptibleTool(Tool):
    """A tool that can be interrupted during execution."""

    def __init__(self, delay: float = 0.5, interrupt_event: asyncio.Event | None = None):
        self.called = False
        self.completed = False
        self.cancelled = False
        self.delay = delay
        self.interrupt_event = interrupt_event

    def name(self) -> str:
        return "interruptible_tool"

    def description(self) -> str:
        return "A tool that can be interrupted"

    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, parameters: dict) -> TextResult:
        self.called = True
        try:
            # If we have an interrupt event, trigger it partway through
            if self.interrupt_event:
                await asyncio.sleep(self.delay / 2)
                self.interrupt_event.set()
                await asyncio.sleep(self.delay / 2)
            else:
                await asyncio.sleep(self.delay)
            self.completed = True
            return TextResult(content="completed")
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_interrupt_during_tool_execution_prompts_for_user_input():
    """Test that interrupting during tool execution returns to user prompt."""
    interrupt_event = asyncio.Event()
    tool = InterruptibleTool(delay=0.5, interrupt_event=interrupt_event)
    tool_call = ToolCall("1", FunctionCall("interruptible_tool", json.dumps({})))

    # Completer returns tool call, then response after interrupt
    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="Continuing after interrupt"),
        ]
    )

    history = [UserMessage(content="start")]
    tools = [tool]
    model = "test-model"
    instructions = None

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "Resume after interrupt"),
            ("> ", "/exit"),
        ]
    )

    captured_controller = []

    original_init = InterruptController.__init__

    def capture_init(self, loop):
        captured_controller.append(self)
        original_init(self, loop)

    with patch.object(InterruptController, "__init__", capture_init):
        # Run chat loop and trigger interrupt when tool starts
        async def run_with_interrupt():
            task = asyncio.create_task(
                run_chat_loop(
                    history=history,
                    model=model,
                    tools=tools,
                    instructions=instructions,
                    callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=completer,
                    ui=ui,
                    context_name="test",
                )
            )

            # Wait for interrupt signal from tool
            await interrupt_event.wait()

            # Request interrupt
            if captured_controller:
                captured_controller[0].request_interrupt()

            await task

        await run_with_interrupt()

    assert tool.called
    assert tool.cancelled

    user_messages = [m for m in history if m.role == "user"]
    resume_msg = next((m for m in user_messages if "Resume" in (m.content or "")), None)
    assert resume_msg is not None, "User should have been prompted after interrupt"


@pytest.mark.asyncio
async def test_interrupt_during_do_single_step():
    """Test that interrupting during LLM call (do_single_step) returns to user prompt."""
    interrupt_event = asyncio.Event()
    # Longer delay to ensure interrupt happens before completion
    tool = InterruptibleTool(delay=1.0, interrupt_event=interrupt_event)
    tool_call = ToolCall("1", FunctionCall("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="After interrupt"),
        ]
    )

    history = [UserMessage(content="start")]
    tools = [tool]
    model = "test-model"
    instructions = None

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "continue after interrupt"),
            ("> ", "/exit"),
        ]
    )

    captured_controller = []

    original_init = InterruptController.__init__

    def capture_init(self, loop):
        captured_controller.append(self)
        original_init(self, loop)

    with patch.object(InterruptController, "__init__", capture_init):

        async def run_with_interrupt():
            task = asyncio.create_task(
                run_chat_loop(
                    history=history,
                    model=model,
                    tools=tools,
                    instructions=instructions,
                    callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=completer,
                    ui=ui,
                    context_name="test",
                )
            )

            # Wait for interrupt event from tool (midway through execution)
            await interrupt_event.wait()

            # Trigger interrupt
            if captured_controller:
                captured_controller[0].request_interrupt()

            # Allow completion
            await task

        await run_with_interrupt()

    assert tool.cancelled

    user_messages = [m for m in history if m.role == "user"]
    assert len(user_messages) >= 2


@pytest.mark.asyncio
async def test_multiple_tool_calls_with_interrupt():
    """Test interrupting when multiple tool calls are in flight."""
    interrupt_event = asyncio.Event()
    tool1 = InterruptibleTool(delay=0.5, interrupt_event=interrupt_event)
    tool2 = InterruptibleTool(delay=0.5)

    tool_call1 = ToolCall("1", FunctionCall("interruptible_tool", json.dumps({})))
    tool_call2 = ToolCall("2", FunctionCall("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call1, tool_call2]),
            FakeMessage(content="After interrupt"),
        ]
    )

    history = [UserMessage(content="start")]
    tools = [tool1, tool2]
    model = "test-model"
    instructions = None

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "continue"),
            ("> ", "/exit"),
        ]
    )

    captured_controller = []

    original_init = InterruptController.__init__

    def capture_init(self, loop):
        captured_controller.append(self)
        original_init(self, loop)

    with patch.object(InterruptController, "__init__", capture_init):

        async def run_with_interrupt():
            task = asyncio.create_task(
                run_chat_loop(
                    history=history,
                    model=model,
                    tools=tools,
                    instructions=instructions,
                    callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=completer,
                    ui=ui,
                    context_name="test",
                )
            )

            await interrupt_event.wait()

            if captured_controller:
                captured_controller[0].request_interrupt()

            await task

        await run_with_interrupt()

    assert tool1.cancelled or tool2.cancelled


@pytest.mark.asyncio
async def test_chat_loop_without_interrupts_works_normally():
    """Test that chat loop works normally without any interrupts."""
    tool = InterruptibleTool(delay=0.05)
    tool_call = ToolCall("1", FunctionCall("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="Normal response"),
        ]
    )

    history = [UserMessage(content="start")]
    tools = [tool]
    model = "test-model"
    instructions = None

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "continue"),
            ("> ", "/exit"),
        ]
    )

    await run_chat_loop(
        history=history,
        model=model,
        tools=tools,
        instructions=instructions,
        callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        completer=completer,
        ui=ui,
        context_name="test",
    )

    assert tool.called
    assert tool.completed
    assert not tool.cancelled

    assert len(history) > 2


@pytest.mark.asyncio
async def test_interrupt_recovery_continues_conversation():
    """Test that after interrupt recovery, the conversation continues properly."""
    interrupt_event = asyncio.Event()
    tool = InterruptibleTool(delay=0.5, interrupt_event=interrupt_event)
    tool_call = ToolCall("1", FunctionCall("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="After recovery"),
            FakeMessage(content="Final message"),
        ]
    )

    history = [UserMessage(content="start")]
    tools = [tool]
    model = "test-model"
    instructions = None

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "recovered"),
            ("> ", "continue"),
            ("> ", "/exit"),
        ]
    )

    captured_controller = []

    original_init = InterruptController.__init__

    def capture_init(self, loop):
        captured_controller.append(self)
        original_init(self, loop)

    with patch.object(InterruptController, "__init__", capture_init):

        async def run_with_interrupt():
            task = asyncio.create_task(
                run_chat_loop(
                    history=history,
                    model=model,
                    tools=tools,
                    instructions=instructions,
                    callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=completer,
                    ui=ui,
                    context_name="test",
                )
            )

            await interrupt_event.wait()

            if captured_controller:
                captured_controller[0].request_interrupt()

            await task

        await run_with_interrupt()

    assert tool.cancelled

    user_messages = [m for m in history if m.role == "user"]
    assert len(user_messages) >= 2

    assistant_messages = [m for m in history if m.role == "assistant"]
    assert len(assistant_messages) >= 1


@pytest.mark.asyncio
async def test_interrupt_during_second_tool_call():
    """Test interrupting during handle_tool_calls with multiple concurrent tool calls."""
    interrupt_event = asyncio.Event()

    tool = InterruptibleTool(delay=0.5, interrupt_event=interrupt_event)

    call1 = ToolCall("1", FunctionCall("interruptible_tool", json.dumps({})))
    call2 = ToolCall("2", FunctionCall("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[call1, call2]),
            FakeMessage(content="After tool interrupt"),
        ]
    )

    history = [UserMessage(content="start")]
    tools = [tool]
    model = "test-model"
    instructions = None

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "after tool interrupt"),
            ("> ", "/exit"),
        ]
    )

    captured_controller = []

    original_init = InterruptController.__init__

    def capture_init(self, loop):
        captured_controller.append(self)
        original_init(self, loop)

    with patch.object(InterruptController, "__init__", capture_init):

        async def run_with_interrupt():
            task = asyncio.create_task(
                run_chat_loop(
                    history=history,
                    model=model,
                    tools=tools,
                    instructions=instructions,
                    callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=completer,
                    ui=ui,
                    context_name="test",
                )
            )

            await interrupt_event.wait()

            # Trigger interrupt
            if captured_controller:
                captured_controller[0].request_interrupt()

            await task

        await run_with_interrupt()

    assert tool.cancelled

    user_messages = [m for m in history if m.role == "user"]
    assert any("after tool interrupt" in (m.content or "") for m in user_messages)


@pytest.mark.asyncio
async def test_sigint_interrupts_tool_execution():
    """E2E test: SIGINT (CTRL-C) interrupts tool execution and returns to user prompt."""
    interrupt_event = asyncio.Event()
    tool = InterruptibleTool(delay=1.0, interrupt_event=interrupt_event)
    tool_call = ToolCall("1", FunctionCall("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="After SIGINT"),
        ]
    )

    history = [UserMessage(content="start")]
    tools = [tool]
    model = "test-model"
    instructions = None

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "recovered from SIGINT"),
            ("> ", "/exit"),
        ]
    )

    async def run_with_sigint():
        task = asyncio.create_task(
            run_chat_loop(
                history=history,
                model=model,
                tools=tools,
                instructions=instructions,
                callbacks=NullProgressCallbacks(),
                tool_callbacks=NullToolCallbacks(),
                completer=completer,
                ui=ui,
                context_name="test",
            )
        )

        await interrupt_event.wait()

        os.kill(os.getpid(), signal.SIGINT)

        await asyncio.sleep(0.1)

        await task

    await run_with_sigint()

    assert tool.cancelled

    user_messages = [m for m in history if m.role == "user"]
    assert any("recovered from SIGINT" in (m.content or "") for m in user_messages)


@pytest.mark.asyncio
async def test_interrupt_during_llm_call():
    """Test that CTRL-C during LLM call (not tool execution) cancels immediately."""

    llm_started = asyncio.Event()

    async def slow_completer(history, model, tools, callbacks):
        llm_started.set()
        await asyncio.sleep(2.0)
        return FakeCompleter([FakeMessage(content="Response from LLM")])._completions[0]

    history = [UserMessage(content="test")]
    tools = []
    model = "test-model"
    instructions = None

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "user input after interrupt"),
            ("> ", "/exit"),
        ]
    )

    captured_controller = []

    original_init = InterruptController.__init__

    def capture_init(self, loop):
        captured_controller.append(self)
        original_init(self, loop)

    with patch.object(InterruptController, "__init__", capture_init):

        async def run_with_interrupt():
            task = asyncio.create_task(
                run_chat_loop(
                    history=history,
                    model=model,
                    tools=tools,
                    instructions=instructions,
                    callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=slow_completer,
                    ui=ui,
                    context_name="test",
                )
            )

            await llm_started.wait()
            await asyncio.sleep(0.1)

            if captured_controller:
                captured_controller[0].request_interrupt()

            await task

        await run_with_interrupt()

    assistant_messages = [m for m in history if m.role == "assistant"]
    for msg in assistant_messages:
        assert "Response from LLM" not in (msg.content or "")

    user_messages = [m for m in history if m.role == "user"]
    assert any("user input after interrupt" in (m.content or "") for m in user_messages)
