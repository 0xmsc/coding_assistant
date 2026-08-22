import pytest

from coding_assistant.cli.tool_bundle import create_cli_tool_bundle, load_tool_instructions
from coding_assistant.llm.types import TextToolResult


@pytest.mark.asyncio
async def test_create_cli_tool_bundle_includes_remote_tools() -> None:
    bundle = create_cli_tool_bundle(skills_directories=[])
    try:
        tool_names = {tool.name() for tool in bundle.tools}

        assert bundle.instructions.startswith(load_tool_instructions())
        assert "advanced-tool-usage" in bundle.instructions
        assert {
            "skills_list_resources",
            "skills_read",
            "shell_execute",
            "python_execute",
            "remote_connect",
            "remotes_discover",
            "remotes_list",
            "remote_prompt",
            "remote_wait",
            "remotes_wait_any",
            "remote_cancel",
            "remote_disconnect",
        } <= tool_names
        assert {
            "todo_add",
            "todo_list_todos",
            "todo_complete",
        }.isdisjoint(tool_names)
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_cli_tool_bundle_close_terminates_background_tasks() -> None:
    bundle = create_cli_tool_bundle(skills_directories=[])
    shell_execute = next(tool for tool in bundle.tools if tool.name() == "shell_execute")
    result = await shell_execute.execute({"command": "sleep 10", "background": True})
    assert isinstance(result, TextToolResult)
    task = bundle._task_manager.get_task(1)
    assert task is not None
    assert task.handle.is_running

    await bundle.close()

    assert not task.handle.is_running
    assert bundle._task_manager.list_tasks() == []
