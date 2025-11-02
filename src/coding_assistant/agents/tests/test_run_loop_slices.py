import json

import pytest

from coding_assistant.agents.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.agents.execution import run_agent_loop
from coding_assistant.agents.tests.helpers import (
    FakeCompleter,
    FakeFunction,
    FakeMessage,
    FakeToolCall,
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.agents.types import AgentContext, TextResult, Tool, AgentOutput
from coding_assistant.tools.tools import FinishTaskTool, ShortenConversation


class FakeEchoTool(Tool):
    def __init__(self):
        self.called_with = None

    def name(self) -> str:
        return "fake.echo"

    def description(self) -> str:
        return "Echo a provided text"

    def parameters(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, parameters: dict) -> TextResult:
        self.called_with = parameters
        return TextResult(content=f"echo: {parameters['text']}")


@pytest.mark.asyncio
async def test_tool_selection_then_finish():
    echo_call = FakeToolCall("1", FakeFunction("fake.echo", json.dumps({"text": "hi"})))
    finish_call = FakeToolCall(
        "2",
        FakeFunction(
            "finish_task",
            json.dumps({"result": "done", "summary": "sum"}),
        ),
    )

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[echo_call]),
            FakeMessage(tool_calls=[finish_call]),
        ]
    )

    fake_tool = FakeEchoTool()
    agent = make_test_agent(tools=[fake_tool, FinishTaskTool(), ShortenConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        agent_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        shorten_conversation_at_tokens=200_000,
        enable_user_feedback=False,
        completer=completer,
        ui=make_ui_mock(),
    )

    assert state.output is not None
    assert state.output.result == "done"
    assert state.output.summary == "sum"
    assert fake_tool.called_with == {"text": "hi"}

    desc, state = agent
    assert state.history[1:] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "name": "fake.echo",
                        "arguments": '{"text": "hi"}',
                    },
                }
            ],
        },
        {
            "tool_call_id": "1",
            "role": "tool",
            "name": "fake.echo",
            "content": "echo: hi",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "2",
                    "function": {
                        "name": "finish_task",
                        "arguments": '{"result": "done", "summary": "sum"}',
                    },
                }
            ],
        },
        {
            "tool_call_id": "2",
            "role": "tool",
            "name": "finish_task",
            "content": "Agent output set.",
        },
    ]


@pytest.mark.asyncio
async def test_unknown_tool_error_then_finish(monkeypatch):
    unknown_call = FakeToolCall("1", FakeFunction("unknown.tool", "{}"))
    finish_call = FakeToolCall(
        "2",
        FakeFunction(
            "finish_task",
            json.dumps({"result": "ok", "summary": "s"}),
        ),
    )

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[unknown_call]),
            FakeMessage(tool_calls=[finish_call]),
        ]
    )

    agent = make_test_agent(tools=[FinishTaskTool(), ShortenConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        agent_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        shorten_conversation_at_tokens=200_000,
        enable_user_feedback=False,
        completer=completer,
        ui=make_ui_mock(),
    )

    desc, state = agent
    assert state.history[1:] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "name": "unknown.tool",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "tool_call_id": "1",
            "role": "tool",
            "name": "unknown.tool",
            "content": "Error executing tool: Tool unknown.tool not found in agent tools.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "2",
                    "function": {
                        "name": "finish_task",
                        "arguments": '{"result": "ok", "summary": "s"}',
                    },
                }
            ],
        },
        {
            "tool_call_id": "2",
            "role": "tool",
            "name": "finish_task",
            "content": "Agent output set.",
        },
    ]
    assert state.output is not None
    assert state.output.result == "ok"


@pytest.mark.asyncio
async def test_assistant_message_without_tool_calls_prompts_correction(monkeypatch):
    finish_call = FakeToolCall(
        "2",
        FakeFunction(
            "finish_task",
            json.dumps({"result": "r", "summary": "s"}),
        ),
    )
    completer = FakeCompleter(
        [
            FakeMessage(content="Hello"),
            FakeMessage(tool_calls=[finish_call]),
        ]
    )

    agent = make_test_agent(tools=[FinishTaskTool(), ShortenConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        agent_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        shorten_conversation_at_tokens=200_000,
        enable_user_feedback=False,
        completer=completer,
        ui=make_ui_mock(),
    )

    desc, state = agent
    assert state.history[1:] == [
        {
            "role": "assistant",
            "content": "Hello",
        },
        {
            "role": "user",
            "content": "I detected a step from you without any tool calls. This is not allowed. If you are done with your task, please call the `finish_task` tool to signal that you are done. Otherwise, continue your work.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "2",
                    "function": {
                        "name": "finish_task",
                        "arguments": '{"result": "r", "summary": "s"}',
                    },
                }
            ],
        },
        {
            "tool_call_id": "2",
            "role": "tool",
            "name": "finish_task",
            "content": "Agent output set.",
        },
    ]
    assert state.output is not None
    assert state.output.result == "r"


@pytest.mark.asyncio
async def test_interrupt_feedback_injected_and_loop_continues(monkeypatch):
    # Fake InterruptibleSection that signals one interruption, then none
    class FakeInterruptOnce:
        _count = 0

        def __enter__(self):
            type(self)._count += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @property
        def was_interrupted(self):
            return self._count == 1

    # Patch the execution module to use our fake
    from coding_assistant.agents import execution as execution_module

    monkeypatch.setattr(execution_module, "InterruptibleSection", FakeInterruptOnce)

    echo_call = FakeToolCall("1", FakeFunction("fake.echo", json.dumps({"text": "first"})))
    finish_call = FakeToolCall(
        "2",
        FakeFunction(
            "finish_task",
            json.dumps({"result": "done", "summary": "sum"}),
        ),
    )

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[echo_call]),  # interrupted after this step
            FakeMessage(tool_calls=[finish_call]),
        ]
    )

    echo_tool = FakeEchoTool()
    agent = make_test_agent(tools=[echo_tool, FinishTaskTool(), ShortenConversation()])
    desc, state = agent

    expected_feedback_text = (
        "Your client has provided the following feedback on your work:\n\n"
        "> Please refine\n\n"
        "Please rework your result to address the feedback.\n"
        "Afterwards, call the `finish_task` tool again to signal that you are done."
    )

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        agent_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        shorten_conversation_at_tokens=200_000,
        enable_user_feedback=False,
        completer=completer,
        ui=make_ui_mock(ask_sequence=[("Feedback: ", "Please refine")]),
        is_interruptible=True,
    )
    assert state.output is not None
    assert state.output.result == "done"
    desc, state = agent
    # Feedback should be injected between first tool result and the next assistant call
    assert expected_feedback_text in [m.get("content") for m in state.history if m.get("role") == "user"]

    @pytest.mark.asyncio
    async def test_interrupt_disabled_skips_feedback(monkeypatch):
        # Fake InterruptibleSection that would signal interruption, but we disable it
        class FakeInterruptAlways:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            @property
            def was_interrupted(self):
                return True

        from coding_assistant.agents import execution as execution_module

        monkeypatch.setattr(execution_module, "InterruptibleSection", FakeInterruptAlways)

        finish_call = FakeToolCall(
            "1",
            FakeFunction(
                "finish_task",
                json.dumps({"result": "done", "summary": "sum"}),
            ),
        )
        completer = FakeCompleter(
            [
                FakeMessage(tool_calls=[finish_call]),
            ]
        )

        desc, state = make_test_agent(tools=[FinishTaskTool(), ShortenConversation()])

        await run_agent_loop(
            AgentContext(desc=desc, state=state),
            agent_callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            shorten_conversation_at_tokens=200_000,
            enable_user_feedback=False,
            completer=completer,
            ui=make_ui_mock(),
            is_interruptible=False,
        )
        assert state.output is not None
        assert state.output.result == "done"
        # Ensure no feedback prompt injected
        assert not any(
            "Feedback on your work" in (m.get("content") or "") for m in state.history if m.get("role") == "user"
        )


@pytest.mark.asyncio
async def test_errors_if_output_already_set():
    desc, state = make_test_agent(tools=[FinishTaskTool(), ShortenConversation()])
    state.output = AgentOutput(result="r", summary="s")
    with pytest.raises(RuntimeError, match="Agent already has a result or summary."):
        await run_agent_loop(
            AgentContext(desc=desc, state=state),
            agent_callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            shorten_conversation_at_tokens=200_000,
            enable_user_feedback=False,
            completer=FakeCompleter([FakeMessage(content="irrelevant")]),
            ui=make_ui_mock(),
        )


@pytest.mark.asyncio
async def test_feedback_ok_does_not_reloop():
    finish_call = FakeToolCall(
        "1",
        FakeFunction(
            "finish_task",
            json.dumps({"result": "final", "summary": "sum"}),
        ),
    )
    completer = FakeCompleter([FakeMessage(tool_calls=[finish_call])])

    agent = make_test_agent(tools=[FinishTaskTool(), ShortenConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        agent_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        shorten_conversation_at_tokens=200_000,
        enable_user_feedback=True,
        completer=completer,
        ui=make_ui_mock(ask_sequence=[(f"Feedback for {desc.name}", "Ok")]),
    )
    assert state.output is not None
    assert state.output.result == "final"


@pytest.mark.asyncio
async def test_multiple_tool_calls_processed_in_order():
    call1 = FakeToolCall("1", FakeFunction("fake.echo", json.dumps({"text": "first"})))
    call2 = FakeToolCall("2", FakeFunction("fake.echo", json.dumps({"text": "second"})))
    finish_call = FakeToolCall(
        "3",
        FakeFunction(
            "finish_task",
            json.dumps({"result": "ok", "summary": "s"}),
        ),
    )

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[call1, call2]),
            FakeMessage(tool_calls=[finish_call]),
        ]
    )

    echo_tool = FakeEchoTool()
    agent = make_test_agent(tools=[echo_tool, FinishTaskTool(), ShortenConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        agent_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        shorten_conversation_at_tokens=200_000,
        enable_user_feedback=False,
        completer=completer,
        ui=make_ui_mock(),
    )
    assert state.output is not None
    assert state.output.result == "ok"
    desc, state = agent
    assert [m for m in state.history[1:4]] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "name": "fake.echo",
                        "arguments": '{"text": "first"}',
                    },
                },
                {
                    "id": "2",
                    "function": {
                        "name": "fake.echo",
                        "arguments": '{"text": "second"}',
                    },
                },
            ],
        },
        {
            "tool_call_id": "1",
            "role": "tool",
            "name": "fake.echo",
            "content": "echo: first",
        },
        {
            "tool_call_id": "2",
            "role": "tool",
            "name": "fake.echo",
            "content": "echo: second",
        },
    ]

    # Also verify the finish comes after both tool results
    desc, state = agent
    assert state.history[4] == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "3",
                "function": {
                    "name": "finish_task",
                    "arguments": '{"result": "ok", "summary": "s"}',
                },
            }
        ],
    }
    desc, state = agent
    assert state.history[5] == {
        "tool_call_id": "3",
        "role": "tool",
        "name": "finish_task",
        "content": "Agent output set.",
    }


@pytest.mark.asyncio
async def test_feedback_loop_then_finish():
    finish_call_1 = FakeToolCall(
        "1",
        FakeFunction(
            "finish_task",
            json.dumps({"result": "first", "summary": "s1"}),
        ),
    )

    finish_call_2 = FakeToolCall(
        "2",
        FakeFunction(
            "finish_task",
            json.dumps({"result": "second", "summary": "s2"}),
        ),
    )

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[finish_call_1]),
            FakeMessage(tool_calls=[finish_call_2]),
        ]
    )

    agent = make_test_agent(tools=[FinishTaskTool(), ShortenConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        agent_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        shorten_conversation_at_tokens=200_000,
        enable_user_feedback=True,
        completer=completer,
        ui=make_ui_mock(
            ask_sequence=[(f"Feedback for {desc.name}", "Please improve"), (f"Feedback for {desc.name}", "Ok")]
        ),
    )
    assert state.output is not None
    assert state.output.result == "second"
    assert state.output.summary == "s2"

    expected_feedback_text = (
        "Your client has provided the following feedback on your work:\n\n"
        "> Please improve\n\n"
        "Please rework your result to address the feedback.\n"
        "Afterwards, call the `finish_task` tool again to signal that you are done."
    )

    desc, state = agent
    assert state.history[1:] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "name": "finish_task",
                        "arguments": '{"result": "first", "summary": "s1"}',
                    },
                }
            ],
        },
        {
            "tool_call_id": "1",
            "role": "tool",
            "name": "finish_task",
            "content": "Agent output set.",
        },
        {
            "role": "user",
            "content": expected_feedback_text,
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "2",
                    "function": {
                        "name": "finish_task",
                        "arguments": '{"result": "second", "summary": "s2"}',
                    },
                }
            ],
        },
        {
            "tool_call_id": "2",
            "role": "tool",
            "name": "finish_task",
            "content": "Agent output set.",
        },
    ]
