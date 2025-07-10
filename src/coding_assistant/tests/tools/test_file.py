import os
from coding_assistant.tools.file import RipgrepTool, RipgrepToolInput, read_only_file_tools


def test_ripgrep_tool_input():
    input_data = RipgrepToolInput(pattern="TODO", case_insensitive="true")
    assert input_data.pattern == "TODO"
    assert input_data.case_insensitive == True


def test_ripgrep_tool_run_no_mock():
    # Ensure ripgrep is available in the system
    tool = RipgrepTool()

    # Temporary test file
    test_filename = "test_ripgrep.txt"
    with open(test_filename, "w") as f:
        f.write("Find this line.")

    try:
        # Perform a search
        result = tool.invoke({"pattern": "Find this"})
        assert "Find this line." in result

        # Test case insensitivity
        result_ci = tool.invoke({"pattern": "find this", "case_insensitive": "true"})
        assert "Find this line." in result_ci
    finally:
        os.remove(test_filename)


def test_read_only_file_tools_integration():
    tools = read_only_file_tools()
    assert any(isinstance(tool, RipgrepTool) for tool in tools)
