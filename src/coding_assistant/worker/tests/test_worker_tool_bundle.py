from pathlib import Path

from coding_assistant.worker.tool_bundle import create_worker_tool_bundle, load_tool_instructions


def test_create_worker_tool_bundle_excludes_cli_remote_tools() -> None:
    bundle = create_worker_tool_bundle(workspace=Path("/workspace"), skills_directories=[])
    tool_names = {tool.name() for tool in bundle.tools}

    assert bundle.instructions.startswith(load_tool_instructions())
    assert "advanced-tool-usage" in bundle.instructions
    assert "## Session title" in bundle.instructions
    assert "no previous `set_session_title` call" in bundle.instructions
    assert "## Remotes" not in bundle.instructions
    assert "## MCP" not in bundle.instructions
    assert {
        "skills_list_resources",
        "skills_read",
        "shell_execute",
        "python_execute",
        "filesystem_write_file",
        "filesystem_edit_file",
        "load_image",
        "set_session_title",
    } <= tool_names
    assert {
        "remote_connect",
        "remotes_discover",
        "remotes_list",
        "remote_prompt",
        "remote_wait",
        "remotes_wait_any",
        "remote_cancel",
        "remote_disconnect",
        "todo_add",
        "todo_list_todos",
        "todo_complete",
    }.isdisjoint(tool_names)
