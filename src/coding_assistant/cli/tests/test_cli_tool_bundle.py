import pytest

from coding_assistant.cli.tool_bundle import create_cli_tool_bundle, load_tool_instructions


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
            "remote_connect",
            "remotes_discover",
            "remotes_list",
            "remote_prompt",
            "remote_wait",
            "remotes_wait_any",
            "remote_cancel",
            "remote_disconnect",
        } <= tool_names
    finally:
        await bundle.close()
