import json

import pytest

from coding_assistant.agents.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.agents.execution import do_single_step, handle_tool_call
from coding_assistant.agents.tests.helpers import (
    FakeFunction,
    FakeMessage,
    FakeToolCall,
    FakeCompleter,
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.agents.types import AgentContext
from coding_assistant.tools.tools import FinishTaskTool, ShortenConversation


@pytest.mark.asyncio
async def test_shorten_conversation_resets_history():
    # Prepare agent with some existing history that should be cleared
    desc, state = make_test_agent(
        tools=[FinishTaskTool(), ShortenConversation()],
        history=[
            {"role": "user", "content": "old start"},
            {"role": "assistant", "content": "old reply"},
        ],
    )

    callbacks = NullProgressCallbacks()

    # Invoke shorten_conversation tool directly
    summary_text = "This is the summary of prior conversation."
    tool_call = FakeToolCall(
        id="shorten-1",
        function=FakeFunction(
            name="shorten_conversation",
            arguments=json.dumps({"summary": summary_text}),
        ),
    )

    ctx = AgentContext(desc=desc, state=state)
    await handle_tool_call(tool_call, ctx, callbacks, tool_callbacks=NullToolCallbacks(), ui=make_ui_mock())

    # History should be reset to a fresh start message + summary message, followed by the tool result message
    assert len(state.history) >= 3
    assert state.history[0]["role"] == "user"
    assert "You are an agent named" in state.history[0]["content"]

    assert state.history[1] == {
        "role": "user",
        "content": (
            f"A summary of your conversation with the client until now:\n\n{summary_text}\n\nPlease continue your work."
        ),
    }

    assert state.history[2] == {
        "tool_call_id": "shorten-1",
        "role": "tool",
        "name": "shorten_conversation",
        "content": "Conversation shortened and history reset.",
    }

    # Subsequent steps should continue from the new history
    finish_call = FakeToolCall(
        "finish-1",
        FakeFunction(
            "finish_task",
            json.dumps({"result": "done", "summary": "sum"}),
        ),
    )

    completer = FakeCompleter([FakeMessage(tool_calls=[finish_call])])

    await do_single_step(
        ctx,
        callbacks,
        shorten_conversation_at_tokens=200_000,
        completer=completer,
        ui=make_ui_mock(),
        tool_callbacks=NullToolCallbacks(),
    )

    # Verify the assistant tool call and finish result were appended after the reset messages
    assert state.history[-2] == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "finish-1",
                "function": {
                    "name": "finish_task",
                    "arguments": '{"result": "done", "summary": "sum"}',
                },
            }
        ],
    }
    assert state.history[-1] == {
        "tool_call_id": "finish-1",
        "role": "tool",
        "name": "finish_task",
        "content": "Agent output set.",
    }
