import json

import pytest

from coding_assistant.agents.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.agents.execution import do_single_step, handle_tool_calls
from coding_assistant.agents.tests.helpers import (
    FakeFunction,
    FakeMessage,
    FakeToolCall,
    FakeCompleter,
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.agents.types import AgentContext
from coding_assistant.tools.tools import FinishTaskTool, CompactConversation


@pytest.mark.asyncio
async def test_compact_conversation_resets_history():
    # Prepare agent with some existing history that should be cleared
    desc, state = make_test_agent(
        tools=[FinishTaskTool(), CompactConversation()],
        history=[
            {"role": "user", "content": "old start"},
            {"role": "assistant", "content": "old reply"},
        ],
    )

    callbacks = NullProgressCallbacks()

    # Invoke compact_conversation tool directly
    summary_text = "This is the summary of prior conversation."
    tool_call = FakeToolCall(
        id="shorten-1",
        function=FakeFunction(
            name="compact_conversation",
            arguments=json.dumps({"summary": summary_text}),
        ),
    )

    ctx = AgentContext(desc=desc, state=state)
    msg = FakeMessage(tool_calls=[tool_call])
    await handle_tool_calls(msg, ctx, callbacks, tool_callbacks=NullToolCallbacks(), ui=make_ui_mock())

    # History should be reset to keeping the first message + summary message, followed by the tool result message
    assert len(state.history) >= 3
    assert state.history[0] == {"role": "user", "content": "old start"}

    assert state.history[1] == {
        "role": "user",
        "content": (
            f"A summary of your conversation with the client until now:\n\n{summary_text}\n\nPlease continue your work."
        ),
    }

    assert state.history[2] == {
        "tool_call_id": "shorten-1",
        "role": "tool",
        "name": "compact_conversation",
        "content": "Conversation compacted and history reset.",
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

    msg, _ = await do_single_step(
        ctx,
        callbacks,
        completer=completer,
    )

    # Append assistant message to history
    from coding_assistant.agents.history import append_assistant_message

    append_assistant_message(state.history, callbacks, desc.name, msg)

    await handle_tool_calls(msg, ctx, callbacks, NullToolCallbacks(), ui=make_ui_mock())

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
