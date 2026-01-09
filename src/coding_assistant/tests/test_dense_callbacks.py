from typing import cast, Any
from unittest.mock import patch, call
from coding_assistant import callbacks
from coding_assistant.callbacks import DenseProgressCallbacks, ReasoningState, ContentState, ToolState, IdleState
from coding_assistant.llm.types import ToolCall, ToolMessage, FunctionCall


def test_dense_callbacks_lifecycle() -> None:
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        assert cb._state is None

        cb.on_reasoning_chunk("Thinking...")
        assert isinstance(cb._state, ReasoningState)
        cb.on_reasoning_chunk("\n\nDone thinking.")

        cb.on_content_chunk("Hello")
        assert isinstance(cb._state, ContentState)
        cb.on_content_chunk(" world!\n\n")

        tool_call = ToolCall(id="call_1", function=FunctionCall(name="test_tool", arguments='{"arg": "val"}'))
        cb.on_tool_start("TestAgent", tool_call, {"arg": "val"})
        assert isinstance(cb._state, ToolState)
        tool_msg = ToolMessage(content="Tool result", tool_call_id="call_1", name="test_tool")
        cb.on_tool_message("TestAgent", tool_msg, "test_tool", {"arg": "val"})

        cb.on_content_chunk("Final bit")
        cb.on_chunks_end()
        assert isinstance(cb._state, IdleState)

    assert mock_print.called


def test_dense_callbacks_tool_formatting() -> None:
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        tool_call = ToolCall(id="call_1", function=FunctionCall(name="shell_execute", arguments='{"command": "ls"}'))
        cb.on_tool_start("TestAgent", tool_call, {"command": "ls"})
        tool_msg = ToolMessage(content="file1\nfile2", tool_call_id="call_1", name="shell_execute")
        cb.on_tool_message("TestAgent", tool_msg, "shell_execute", {"command": "ls"})

    assert mock_print.called


def test_dense_callbacks_paragraph_flushing() -> None:
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        cb.on_content_chunk("One")
        cb.on_content_chunk(" Two")

        cb.on_chunks_end()

    assert mock_print.called


def test_dense_callbacks_state_transition_flushes() -> None:
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        cb.on_reasoning_chunk("Thinking hard")
        cb.on_content_chunk("Actually here is the answer")
        cb.on_chunks_end()

    found_reasoning = False
    for call_args in mock_print.call_args_list:
        for arg in call_args.args:
            if (
                hasattr(arg, "renderable")
                and hasattr(arg.renderable, "markup")
                and "Thinking hard" in arg.renderable.markup
            ):
                found_reasoning = True
            if "Thinking hard" in str(arg):
                found_reasoning = True

    assert found_reasoning, "Reasoning should have been flushed when switching to content"


def test_dense_callbacks_empty_line_logic() -> None:
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        cb.on_reasoning_chunk("Thinking")

        cb.on_reasoning_chunk(" more")

        cb.on_content_chunk("Hello")

        tool_call = ToolCall(id="call_1", function=FunctionCall(name="test_tool", arguments='{"arg": 1}'))
        cb.on_tool_start("TestAgent", tool_call, {"arg": 1})

        cb.on_content_chunk("Result")

        print_calls = [c for c in mock_print.call_args_list]

        found_newline = False
        for i in range(len(print_calls) - 1):
            if print_calls[i] == call():
                found_newline = True
                break
        assert found_newline, "Expected newline when switching from reasoning to content"


def test_dense_callbacks_multiline_tool_formatting(capsys: Any) -> None:
    cb = DenseProgressCallbacks()
    callbacks.console.width = 200

    tool_call = ToolCall(id="call_1", function=FunctionCall(name="unknown_tool", arguments='{"arg": "line1\\nline2"}'))
    cb.on_tool_start("TestAgent", tool_call, {"arg": "line1\nline2"})
    captured = capsys.readouterr()
    assert 'unknown_tool(arg="line1\\nline2")' in captured.out

    tool_call = ToolCall(id="call_2", function=FunctionCall(name="shell_execute", arguments='{"command": "ls\\npwd"}'))
    cb.on_tool_start("TestAgent", tool_call, {"command": "ls\npwd"})
    captured = capsys.readouterr()
    assert "▶ shell_execute(command)" in captured.out
    assert "  command:" in captured.out
    assert "  ls" in captured.out
    assert "  pwd" in captured.out

    tool_call = ToolCall(id="call_3", function=FunctionCall(name="shell_execute", arguments='{"command": "ls"}'))
    cb.on_tool_start("TestAgent", tool_call, {"command": "ls"})
    captured = capsys.readouterr()
    assert 'shell_execute(command="ls")' in captured.out

    tool_call = ToolCall(
        id="call_4",
        function=FunctionCall(
            name="filesystem_write_file",
            arguments='{"path": "test.py", "content": "def hello():\\n    pass"}',
        ),
    )
    cb.on_tool_start("TestAgent", tool_call, {"path": "test.py", "content": "def hello():\n    pass"})
    assert cb._SPECIAL_TOOLS["filesystem_write_file"]["content"] == ""
    captured = capsys.readouterr()
    assert 'filesystem_write_file(path="test.py", content)' in captured.out
    assert "  content:" in captured.out
    assert "  def hello():" in captured.out

    tool_call = ToolCall(
        id="call_5",
        function=FunctionCall(
            name="filesystem_edit_file",
            arguments='{"path": "script.sh", "old_text": "line1\\nold", "new_text": "line1\\nline2"}',
        ),
    )
    cb.on_tool_start(
        "TestAgent",
        tool_call,
        {"path": "script.sh", "old_text": "line1\nold", "new_text": "line1\nline2"},
    )
    assert "old_text" in cb._SPECIAL_TOOLS["filesystem_edit_file"]
    assert "new_text" in cb._SPECIAL_TOOLS["filesystem_edit_file"]

    captured = capsys.readouterr()
    assert 'filesystem_edit_file(path="script.sh", old_text, new_text)' in captured.out
    assert "  old_text:" not in captured.out
    assert "  new_text:" not in captured.out

    tool_call = ToolCall(
        id="call_6",
        function=FunctionCall(
            name="python_execute",
            arguments='{"code": "import os\\nprint(os.getcwd())"}',
        ),
    )
    cb.on_tool_start(
        "TestAgent",
        tool_call,
        {"code": "import os\nprint(os.getcwd())"},
    )
    captured = capsys.readouterr()
    assert "▶ python_execute(code)" in captured.out
    assert "  code:" in captured.out

    tool_call = ToolCall(
        id="call_7",
        function=FunctionCall(
            name="todo_add",
            arguments='{"descriptions": ["task 1", "task 2"]}',
        ),
    )
    cb.on_tool_start(
        "TestAgent",
        tool_call,
        {"descriptions": ["task 1", "task 2"]},
    )
    captured = capsys.readouterr()
    assert "▶ todo_add(descriptions)" in captured.out
    assert "  descriptions:" in captured.out
    assert '"task 1"' in captured.out
    assert '"task 2"' in captured.out

    tool_call = ToolCall(
        id="call_8",
        function=FunctionCall(
            name="python_execute",
            arguments='{"not_code": "line1\\nline2"}',
        ),
    )
    cb.on_tool_start("TestAgent", tool_call, {"not_code": "line1\nline2"})
    captured = capsys.readouterr()
    assert 'python_execute(not_code="line1\\nline2")' in captured.out

    tool_call = ToolCall(
        id="call_9",
        function=FunctionCall(
            name="python_execute",
            arguments='{"code": "print(1)"}',
        ),
    )
    cb.on_tool_start(
        "TestAgent",
        tool_call,
        {"code": "print(1)"},
    )
    captured = capsys.readouterr()
    assert 'python_execute(code="print(1)")' in captured.out


def test_dense_callbacks_empty_arg_parentheses(capsys: Any) -> None:
    cb = DenseProgressCallbacks()
    tool_call = ToolCall(id="call_1", function=FunctionCall(name="tasks_list_tasks", arguments="{}"))
    cb.on_tool_start("TestAgent", tool_call, {})
    captured = capsys.readouterr()
    assert "▶ tasks_list_tasks()" in captured.out


def test_dense_callbacks_long_arg_parentheses(capsys: Any) -> None:
    cb = DenseProgressCallbacks()
    tool_call = ToolCall(
        id="call_1",
        function=FunctionCall(
            name="shell_execute",
            arguments='{"command": "echo line1\\necho line2", "background": false}',
        ),
    )
    cb.on_tool_start(
        "TestAgent",
        tool_call,
        {"command": "echo line1\necho line2", "background": False},
    )
    captured = capsys.readouterr()
    assert "▶ shell_execute(command, background=false)" in captured.out


def test_dense_callbacks_tool_result_stripping() -> None:
    cb = DenseProgressCallbacks()
    with patch("coding_assistant.callbacks.print") as mock_print:
        tool_msg = ToolMessage(
            content="--- test.py\n+++ test.py\n-old\n+new\n",
            tool_call_id="call_1",
            name="filesystem_edit_file",
        )
        cb.on_tool_message(
            "TestAgent",
            tool_msg,
            "filesystem_edit_file",
            {"path": "test.py", "old_text": "old", "new_text": "new"},
        )

        found_diff = False
        for call_args in mock_print.call_args_list:
            args = call_args.args
            if args and hasattr(args[0], "renderable"):
                renderable = args[0].renderable
                if hasattr(renderable, "markup") and "```diff" in renderable.markup:
                    found_diff = True
                    assert renderable.markup.endswith("\n````")
                    assert not renderable.markup.endswith("\n\n````")
        assert found_diff

        mock_print.reset_mock()

        tool_msg = ToolMessage(content="- [ ] Task 1\n", tool_call_id="call_2", name="todo_list_todos")
        cb.on_tool_message("TestAgent", tool_msg, "todo_list_todos", {})

        found_todo = False
        for call_args in mock_print.call_args_list:
            args = call_args.args
            if args and hasattr(args[0], "renderable"):
                renderable = args[0].renderable
                if hasattr(renderable, "markup") and "Task 1" in renderable.markup:
                    found_todo = True
                    assert renderable.markup == "- [ ] Task 1"
        assert found_todo


def test_dense_callbacks_tool_lang_extension(capsys: Any) -> None:
    cb = DenseProgressCallbacks()
    callbacks.console.width = 200

    with patch("coding_assistant.callbacks.Markdown", side_effect=cast(Any, callbacks).Markdown) as mock_markdown:
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(
                name="filesystem_write_file",
                arguments='{"path": "test.py", "content": "def hello():\\n    pass"}',
            ),
        )
        cb.on_tool_start(
            "TestAgent",
            tool_call,
            {"path": "test.py", "content": "def hello():\n    pass"},
        )
        found_py = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````py\ndef hello():" in arg:
                found_py = True
        assert found_py

        mock_markdown.reset_mock()

        tool_call = ToolCall(
            id="call_2",
            function=FunctionCall(
                name="filesystem_write_file",
                arguments='{"path": "script.sh", "content": "echo hello\\nls"}',
            ),
        )
        cb.on_tool_start(
            "TestAgent",
            tool_call,
            {"path": "script.sh", "content": "echo hello\nls"},
        )
        found_sh = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````sh\necho hello" in arg:
                found_sh = True
        assert found_sh

        mock_markdown.reset_mock()

        tool_call = ToolCall(
            id="call_3",
            function=FunctionCall(
                name="filesystem_edit_file",
                arguments='{"path": "index.js", "old_text": "const x = 1\\n", "new_text": "const x = 2\\n"}',
            ),
        )
        cb.on_tool_start(
            "TestAgent",
            tool_call,
            {"path": "index.js", "old_text": "const x = 1\n", "new_text": "const x = 2\n"},
        )
        found_js = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````js\nconst x = " in arg:
                found_js = True

        # Now that filesystem_edit_file excludes old_text/new_text from header
        # and doesn't print them as multiline params anymore, found_js should be False
        # as it was previously matching the multiline parameter printout which is now gone.
        # We verify that it is NOT found.
        assert not found_js

        mock_markdown.reset_mock()

        tool_call = ToolCall(
            id="call_4",
            function=FunctionCall(
                name="filesystem_write_file",
                arguments='{"path": "Dockerfile", "content": "FROM alpine\\nRUN ls"}',
            ),
        )
        cb.on_tool_start(
            "TestAgent",
            tool_call,
            {"path": "Dockerfile", "content": "FROM alpine\nRUN ls"},
        )
        found_default = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````\nFROM alpine" in arg:
                found_default = True
        assert found_default

        mock_markdown.reset_mock()

        tool_call = ToolCall(
            id="call_5",
            function=FunctionCall(
                name="filesystem_write_file",
                arguments='{"path": "dir.old/script", "content": "echo hello\\nline2"}',
            ),
        )
        cb.on_tool_start(
            "TestAgent",
            tool_call,
            {"path": "dir.old/script", "content": "echo hello\nline2"},
        )
        found_none = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````\necho hello" in arg:
                found_none = True
        assert found_none

        mock_markdown.reset_mock()

        tool_call = ToolCall(
            id="call_6",
            function=FunctionCall(
                name="filesystem_write_file",
                arguments='{"path": ".gitignore", "content": "node_modules/\\nline2"}',
            ),
        )
        cb.on_tool_start(
            "TestAgent",
            tool_call,
            {"path": ".gitignore", "content": "node_modules/\nline2"},
        )
        found_none = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````\nnode_modules/" in arg:
                found_none = True
        assert found_none

        mock_markdown.reset_mock()

        tool_call = ToolCall(
            id="call_7",
            function=FunctionCall(
                name="filesystem_write_file",
                arguments='{"path": "README.", "content": "content\\nline2"}',
            ),
        )
        cb.on_tool_start(
            "TestAgent",
            tool_call,
            {"path": "README.", "content": "content\nline2"},
        )
        found_none = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````\ncontent" in arg:
                found_none = True
        assert found_none
