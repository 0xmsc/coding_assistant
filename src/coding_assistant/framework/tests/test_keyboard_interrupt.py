from unittest.mock import AsyncMock

import pytest

from coding_assistant.framework.chat import run_chat_loop
from coding_assistant.framework.builtin_tools import CompactConversationTool as CompactConversation
from coding_assistant.framework.tests.helpers import system_actor_scope_for_tests
from coding_assistant.llm.types import BaseMessage, Tool, UserMessage


@pytest.mark.asyncio
async def test_run_chat_loop_raises_keyboard_interrupt_at_prompt() -> None:
    """Test that run_chat_loop propagates KeyboardInterrupt raised during ui.prompt."""
    history: list[BaseMessage] = [UserMessage(content="start")]
    model = "test-model"
    tools: list[Tool] = []
    instructions = None

    # Mock UI to raise KeyboardInterrupt when prompt is called
    ui = AsyncMock()
    ui.prompt.side_effect = KeyboardInterrupt()

    # Verify that KeyboardInterrupt is raised
    tools_with_meta = list(tools)
    if not any(tool.name() == "compact_conversation" for tool in tools_with_meta):
        tools_with_meta.append(CompactConversation())

    with pytest.raises(KeyboardInterrupt):
        async with system_actor_scope_for_tests(tools=tools_with_meta, ui=ui, context_name="test") as actors:
            await run_chat_loop(
                history=history,
                model=model,
                tools=tools_with_meta,
                instructions=instructions,
                completer=AsyncMock(),
                ui=actors.user_actor,
                context_name="test",
                agent_actor=actors.agent_actor,
                tool_call_actor=actors.tool_call_actor,
                user_actor=actors.user_actor,
            )

    # Verify prompt was called
    ui.prompt.assert_called_once()
