import pytest

from coding_assistant.cli.tool_bundle import create_cli_tool_bundle, load_tool_instructions
from coding_assistant.llm.types import TextToolResult


@pytest.mark.asyncio
async def test_create_cli_tool_bundle_tools() -> None:
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
            "filesystem_write_file",
            "filesystem_edit_file",
            "tasks_list_tasks",
            "tasks_get_output",
            "tasks_get_status",
            "tasks_kill_task",
            "tasks_remove_task",
        } == tool_names
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
