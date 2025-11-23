import asyncio
import json
import pytest
from unittest.mock import patch

from coding_assistant.agents.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.agents.execution import run_chat_loop
from coding_assistant.agents.interrupts import InterruptController
from coding_assistant.agents.tests.helpers import (
    FakeCompleter,
    FakeFunction,
    FakeMessage,
    FakeToolCall,
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.agents.types import AgentContext, TextResult, Tool


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
    tool_call = FakeToolCall("1", FakeFunction("interruptible_tool", json.dumps({})))

    # Completer returns tool call, then response after interrupt
    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="Continuing after interrupt"),
        ]
    )

    desc, state = make_test_agent(tools=[tool], history=[{"role": "user", "content": "start"}])

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "Resume after interrupt"),
            ("> ", "/exit"),
        ]
    )

    ctx = AgentContext(desc=desc, state=state)

    # Capture interrupt controller via patch
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
                    ctx,
                    agent_callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=completer,
                    ui=ui,
                )
            )

            # Wait for interrupt signal from tool
            await interrupt_event.wait()

            # Request interrupt
            if captured_controller:
                captured_controller[0].request_interrupt()

            await task

        await run_with_interrupt()

    # Verify tool was cancelled
    assert tool.called
    assert tool.cancelled

    # Verify user was prompted after interrupt
    user_messages = [m for m in state.history if m.get("role") == "user"]
    resume_msg = next((m for m in user_messages if "Resume" in m.get("content", "")), None)
    assert resume_msg is not None, "User should have been prompted after interrupt"


@pytest.mark.asyncio
async def test_interrupt_during_do_single_step():
    """Test that interrupting during LLM call (do_single_step) returns to user prompt."""
    interrupt_event = asyncio.Event()
    # Longer delay to ensure interrupt happens before completion
    tool = InterruptibleTool(delay=1.0, interrupt_event=interrupt_event)
    tool_call = FakeToolCall("1", FakeFunction("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="After interrupt"),
        ]
    )

    desc, state = make_test_agent(tools=[tool], history=[{"role": "user", "content": "start"}])

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "continue after interrupt"),
            ("> ", "/exit"),
        ]
    )

    ctx = AgentContext(desc=desc, state=state)

    captured_controller = []

    original_init = InterruptController.__init__

    def capture_init(self, loop):
        captured_controller.append(self)
        original_init(self, loop)

    with patch.object(InterruptController, "__init__", capture_init):

        async def run_with_interrupt():
            task = asyncio.create_task(
                run_chat_loop(
                    ctx,
                    agent_callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=completer,
                    ui=ui,
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

    # Verify tool was cancelled due to interrupt
    assert tool.cancelled

    # Verify user was prompted after interrupt
    user_messages = [m for m in state.history if m.get("role") == "user"]
    assert len(user_messages) >= 2


@pytest.mark.asyncio
async def test_multiple_tool_calls_with_interrupt():
    """Test interrupting when multiple tool calls are in flight."""
    interrupt_event = asyncio.Event()
    tool1 = InterruptibleTool(delay=0.5, interrupt_event=interrupt_event)
    tool2 = InterruptibleTool(delay=0.5)

    tool_call1 = FakeToolCall("1", FakeFunction("interruptible_tool", json.dumps({})))
    tool_call2 = FakeToolCall("2", FakeFunction("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call1, tool_call2]),
            FakeMessage(content="After interrupt"),
        ]
    )

    desc, state = make_test_agent(tools=[tool1, tool2], history=[{"role": "user", "content": "start"}])

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "continue"),
            ("> ", "/exit"),
        ]
    )

    ctx = AgentContext(desc=desc, state=state)

    captured_controller = []

    original_init = InterruptController.__init__

    def capture_init(self, loop):
        captured_controller.append(self)
        original_init(self, loop)

    with patch.object(InterruptController, "__init__", capture_init):

        async def run_with_interrupt():
            task = asyncio.create_task(
                run_chat_loop(
                    ctx,
                    agent_callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=completer,
                    ui=ui,
                )
            )

            await interrupt_event.wait()

            if captured_controller:
                captured_controller[0].request_interrupt()

            await task

        await run_with_interrupt()

    # At least one tool should have been cancelled
    assert tool1.cancelled or tool2.cancelled


@pytest.mark.asyncio
async def test_chat_loop_without_interrupts_works_normally():
    """Test that chat loop works normally without any interrupts."""
    tool = InterruptibleTool(delay=0.05)
    tool_call = FakeToolCall("1", FakeFunction("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="Normal response"),
        ]
    )

    desc, state = make_test_agent(tools=[tool], history=[{"role": "user", "content": "start"}])

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "continue"),
            ("> ", "/exit"),
        ]
    )

    ctx = AgentContext(desc=desc, state=state)

    # Should complete without errors
    await run_chat_loop(
        ctx,
        agent_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        completer=completer,
        ui=ui,
    )

    # Verify tool completed successfully
    assert tool.called
    assert tool.completed
    assert not tool.cancelled

    # Verify conversation progressed normally
    assert len(state.history) > 2


@pytest.mark.asyncio
async def test_interrupt_recovery_continues_conversation():
    """Test that after interrupt recovery, the conversation continues properly."""
    interrupt_event = asyncio.Event()
    tool = InterruptibleTool(delay=0.5, interrupt_event=interrupt_event)
    tool_call = FakeToolCall("1", FakeFunction("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="After recovery"),
            FakeMessage(content="Final message"),
        ]
    )

    desc, state = make_test_agent(tools=[tool], history=[{"role": "user", "content": "start"}])

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "recovered"),
            ("> ", "continue"),
            ("> ", "/exit"),
        ]
    )

    ctx = AgentContext(desc=desc, state=state)

    captured_controller = []

    original_init = InterruptController.__init__

    def capture_init(self, loop):
        captured_controller.append(self)
        original_init(self, loop)

    with patch.object(InterruptController, "__init__", capture_init):

        async def run_with_interrupt():
            task = asyncio.create_task(
                run_chat_loop(
                    ctx,
                    agent_callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=completer,
                    ui=ui,
                )
            )

            await interrupt_event.wait()

            if captured_controller:
                captured_controller[0].request_interrupt()

            await task

        await run_with_interrupt()

    # Verify tool was cancelled
    assert tool.cancelled

    # Verify conversation continued with multiple user inputs
    user_messages = [m for m in state.history if m.get("role") == "user"]
    assert len(user_messages) >= 2

    # Verify assistant responded after recovery
    assistant_messages = [m for m in state.history if m.get("role") == "assistant"]
    assert len(assistant_messages) >= 1


@pytest.mark.asyncio
async def test_interrupt_during_second_tool_call():
    """Test interrupting during handle_tool_calls with multiple concurrent tool calls."""
    interrupt_event = asyncio.Event()

    # Create a tool that will signal when it's ready to be interrupted
    tool = InterruptibleTool(delay=0.5, interrupt_event=interrupt_event)

    # Two calls to the same tool
    call1 = FakeToolCall("1", FakeFunction("interruptible_tool", json.dumps({})))
    call2 = FakeToolCall("2", FakeFunction("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[call1, call2]),
            FakeMessage(content="After tool interrupt"),
        ]
    )

    desc, state = make_test_agent(tools=[tool], history=[{"role": "user", "content": "start"}])

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "after tool interrupt"),
            ("> ", "/exit"),
        ]
    )

    ctx = AgentContext(desc=desc, state=state)

    captured_controller = []

    original_init = InterruptController.__init__

    def capture_init(self, loop):
        captured_controller.append(self)
        original_init(self, loop)

    with patch.object(InterruptController, "__init__", capture_init):

        async def run_with_interrupt():
            task = asyncio.create_task(
                run_chat_loop(
                    ctx,
                    agent_callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=completer,
                    ui=ui,
                )
            )

            # Wait for at least one tool to start
            await interrupt_event.wait()

            # Trigger interrupt
            if captured_controller:
                captured_controller[0].request_interrupt()

            await task

        await run_with_interrupt()

    # Tool should have been cancelled
    assert tool.cancelled

    # User should have been prompted after interrupt
    user_messages = [m for m in state.history if m.get("role") == "user"]
    assert any("after tool interrupt" in m.get("content", "") for m in user_messages)


@pytest.mark.asyncio
async def test_sigint_interrupts_tool_execution():
    """E2E test: SIGINT (CTRL-C) interrupts tool execution and returns to user prompt."""
    import os
    import signal

    interrupt_event = asyncio.Event()
    tool = InterruptibleTool(delay=1.0, interrupt_event=interrupt_event)
    tool_call = FakeToolCall("1", FakeFunction("interruptible_tool", json.dumps({})))

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="After SIGINT"),
        ]
    )

    desc, state = make_test_agent(tools=[tool], history=[{"role": "user", "content": "start"}])

    ui = make_ui_mock(
        ask_sequence=[
            ("> ", "recovered from SIGINT"),
            ("> ", "/exit"),
        ]
    )

    ctx = AgentContext(desc=desc, state=state)

    async def run_with_sigint():
        task = asyncio.create_task(
            run_chat_loop(
                ctx,
                agent_callbacks=NullProgressCallbacks(),
                tool_callbacks=NullToolCallbacks(),
                completer=completer,
                ui=ui,
            )
        )

        # Wait for tool to start
        await interrupt_event.wait()

        # Send SIGINT to our own process (simulating CTRL-C)
        os.kill(os.getpid(), signal.SIGINT)

        # Allow signal to be processed
        await asyncio.sleep(0.1)

        await task

    await run_with_sigint()

    # Tool should have been cancelled
    assert tool.cancelled

    # User should have been prompted after SIGINT
    user_messages = [m for m in state.history if m.get("role") == "user"]
    assert any("recovered from SIGINT" in m.get("content", "") for m in user_messages)


# Note: Multiple SIGINT behavior is tested in test_interrupts.py::test_interruptible_section_handles_multiple_sigints
# With proper interrupt handling, multiple SIGINTs no longer cause sys.exit()


@pytest.mark.asyncio
async def test_interrupt_during_llm_call():
    """Test that CTRL-C during LLM call (not tool execution) cancels immediately."""
    
    # Create a slow completer that signals when it starts
    llm_started = asyncio.Event()
    
    async def slow_completer(history, model, tools, callbacks):
        llm_started.set()
        # Simulate slow LLM call
        await asyncio.sleep(2.0)
        return FakeCompleter([FakeMessage(content="Response from LLM")])._completions[0]
    
    desc, state = make_test_agent(history=[{"role": "user", "content": "test"}])
    ctx = AgentContext(desc=desc, state=state)
    
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
                    ctx,
                    agent_callbacks=NullProgressCallbacks(),
                    tool_callbacks=NullToolCallbacks(),
                    completer=slow_completer,
                    ui=ui,
                )
            )
            
            # Wait for LLM to start
            await llm_started.wait()
            await asyncio.sleep(0.1)
            
            # Send interrupt while LLM is processing
            if captured_controller:
                captured_controller[0].request_interrupt()
            
            await task
        
        await run_with_interrupt()
    
    # Verify LLM call was cancelled - no assistant message with "Response from LLM"
    assistant_messages = [m for m in state.history if m.get("role") == "assistant"]
    for msg in assistant_messages:
        assert "Response from LLM" not in msg.get("content", "")
    
    # Verify user was prompted after interrupt
    user_messages = [m for m in state.history if m.get("role") == "user"]
    assert any("user input after interrupt" in m.get("content", "") for m in user_messages)
