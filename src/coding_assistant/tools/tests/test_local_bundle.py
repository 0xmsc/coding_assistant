from coding_assistant.tools.local_bundle import create_local_tool_bundle, load_tool_instructions


def test_create_local_tool_bundle_includes_builtin_tools() -> None:
    bundle = create_local_tool_bundle(skills_directories=[])
    tool_names = {tool.name() for tool in bundle.tools}

    assert bundle.instructions.startswith(load_tool_instructions())
    assert (
        "- **advanced-tool-usage**: Guidelines for multi-stage tool orchestration and handling large data using 'redirect_tool_call'. Use this when you need to process large amounts of data without exhausting the context window or when building complex data pipelines."
        in bundle.instructions
    )
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
