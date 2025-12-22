from unittest.mock import patch, call
from coding_assistant import callbacks
from coding_assistant.callbacks import DenseProgressCallbacks, ReasoningState, ContentState, ToolState, IdleState


def test_dense_callbacks_lifecycle():
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        assert cb._state is None

        cb.on_reasoning_chunk("Thinking...")
        assert isinstance(cb._state, ReasoningState)
        cb.on_reasoning_chunk("\n\nDone thinking.")

        cb.on_content_chunk("Hello")
        assert isinstance(cb._state, ContentState)
        cb.on_content_chunk(" world!\n\n")

        cb.on_tool_start("TestAgent", "call_1", "test_tool", {"arg": "val"})
        assert isinstance(cb._state, ToolState)
        cb.on_tool_message("TestAgent", "call_1", "test_tool", {"arg": "val"}, "Tool result")

        cb.on_content_chunk("Final bit")
        cb.on_chunks_end()
        assert isinstance(cb._state, IdleState)

    assert mock_print.called


def test_dense_callbacks_tool_formatting():
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        cb.on_tool_start("TestAgent", "call_1", "shell_execute", {"command": "ls"})
        cb.on_tool_message("TestAgent", "call_1", "shell_execute", {"command": "ls"}, "file1\nfile2")

    assert mock_print.called


def test_dense_callbacks_paragraph_flushing():
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        cb.on_content_chunk("One")
        cb.on_content_chunk(" Two")

        cb.on_chunks_end()

    assert mock_print.called


def test_dense_callbacks_state_transition_flushes():
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


def test_dense_callbacks_empty_line_logic():
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        cb.on_reasoning_chunk("Thinking")

        cb.on_reasoning_chunk(" more")

        cb.on_content_chunk("Hello")

        cb.on_tool_start("TestAgent", "call_1", "test_tool", {"arg": 1})

        cb.on_content_chunk("Result")

        print_calls = [c for c in mock_print.call_args_list]

        found_newline = False
        for i in range(len(print_calls) - 1):
            if print_calls[i] == call():
                found_newline = True
                break
        assert found_newline, "Expected newline when switching from reasoning to content"


def test_dense_callbacks_multiline_tool_formatting(capsys):
    cb = DenseProgressCallbacks()
    callbacks.console.width = 200

    cb.on_tool_start("TestAgent", "call_1", "unknown_tool", {"arg": "line1\nline2"})
    captured = capsys.readouterr()
    assert 'unknown_tool(arg="line1\\nline2")' in captured.out

    cb.on_tool_start("TestAgent", "call_2", "shell_execute", {"command": "ls\npwd"})
    captured = capsys.readouterr()
    assert "▶ shell_execute(command)" in captured.out
    assert "  command:" in captured.out
    assert "  ls" in captured.out
    assert "  pwd" in captured.out

    cb.on_tool_start("TestAgent", "call_3", "shell_execute", {"command": "ls"})
    captured = capsys.readouterr()
    assert 'shell_execute(command="ls")' in captured.out

    cb.on_tool_start(
        "TestAgent",
        "call_4",
        "filesystem_write_file",
        {"path": "test.py", "content": "def hello():\n    pass"},
    )
    assert cb._SPECIAL_TOOLS["filesystem_write_file"]["content"] == ""
    captured = capsys.readouterr()
    assert 'filesystem_write_file(path="test.py", content)' in captured.out
    assert "  content:" in captured.out
    assert "  def hello():" in captured.out

    cb.on_tool_start(
        "TestAgent",
        "call_5",
        "filesystem_edit_file",
        {"path": "script.sh", "old_text": "line1\nold", "new_text": "line1\nline2"},
    )
    assert "old_text" in cb._SPECIAL_TOOLS["filesystem_edit_file"]
    assert "new_text" in cb._SPECIAL_TOOLS["filesystem_edit_file"]

    captured = capsys.readouterr()
    assert 'filesystem_edit_file(path="script.sh", old_text, new_text)' in captured.out
    assert "  old_text:" in captured.out
    assert "  new_text:" in captured.out

    cb.on_tool_start(
        "TestAgent",
        "call_6",
        "python_execute",
        {"code": "import os\nprint(os.getcwd())"},
    )
    captured = capsys.readouterr()
    assert "▶ python_execute(code)" in captured.out
    assert "  code:" in captured.out

    cb.on_tool_start(
        "TestAgent",
        "call_7",
        "todo_add",
        {"descriptions": ["task 1", "task 2"]},
    )
    captured = capsys.readouterr()
    assert "▶ todo_add(descriptions)" in captured.out
    assert "  descriptions:" in captured.out
    assert '"task 1"' in captured.out
    assert '"task 2"' in captured.out

    cb.on_tool_start(
        "TestAgent",
        "call_8",
        "python_execute",
        {"not_code": "line1\nline2"},
    )
    captured = capsys.readouterr()
    assert 'python_execute(not_code="line1\\nline2")' in captured.out

    cb.on_tool_start(
        "TestAgent",
        "call_9",
        "python_execute",
        {"code": "print(1)"},
    )
    captured = capsys.readouterr()
    assert 'python_execute(code="print(1)")' in captured.out


def test_dense_callbacks_empty_arg_parentheses(capsys):
    cb = DenseProgressCallbacks()
    cb.on_tool_start("TestAgent", "call_1", "tasks_list_tasks", {})
    captured = capsys.readouterr()
    assert "▶ tasks_list_tasks()" in captured.out


def test_dense_callbacks_long_arg_parentheses(capsys):
    cb = DenseProgressCallbacks()
    cb.on_tool_start(
        "TestAgent",
        "call_1",
        "shell_execute",
        {"command": "echo line1\necho line2", "background": False},
    )
    captured = capsys.readouterr()
    assert "▶ shell_execute(command, background=false)" in captured.out


def test_dense_callbacks_tool_result_stripping():
    cb = DenseProgressCallbacks()
    with patch("coding_assistant.callbacks.print") as mock_print:
        cb.on_tool_message(
            "TestAgent",
            "call_1",
            "filesystem_edit_file",
            {"path": "test.py", "old_text": "old", "new_text": "new"},
            "--- test.py\n+++ test.py\n-old\n+new\n",
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

        cb.on_tool_message("TestAgent", "call_2", "todo_list_todos", {}, "- [ ] Task 1\n")

        found_todo = False
        for call_args in mock_print.call_args_list:
            args = call_args.args
            if args and hasattr(args[0], "renderable"):
                renderable = args[0].renderable
                if hasattr(renderable, "markup") and "Task 1" in renderable.markup:
                    found_todo = True
                    assert renderable.markup == "- [ ] Task 1"
        assert found_todo


def test_dense_callbacks_tool_lang_extension(capsys):
    cb = DenseProgressCallbacks()
    callbacks.console.width = 200

    with patch("coding_assistant.callbacks.Markdown", side_effect=callbacks.Markdown) as mock_markdown:
        cb.on_tool_start(
            "TestAgent",
            "call_1",
            "filesystem_write_file",
            {"path": "test.py", "content": "def hello():\n    pass"},
        )
        found_py = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````py\ndef hello():" in arg:
                found_py = True
        assert found_py

        mock_markdown.reset_mock()

        cb.on_tool_start(
            "TestAgent",
            "call_2",
            "filesystem_write_file",
            {"path": "script.sh", "content": "echo hello\nls"},
        )
        found_sh = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````sh\necho hello" in arg:
                found_sh = True
        assert found_sh

        mock_markdown.reset_mock()

        cb.on_tool_start(
            "TestAgent",
            "call_3",
            "filesystem_edit_file",
            {"path": "index.js", "old_text": "const x = 1\n", "new_text": "const x = 2\n"},
        )
        found_js = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````js\nconst x = " in arg:
                found_js = True
        assert found_js

        mock_markdown.reset_mock()

        cb.on_tool_start(
            "TestAgent",
            "call_4",
            "filesystem_write_file",
            {"path": "Dockerfile", "content": "FROM alpine\nRUN ls"},
        )
        found_default = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````\nFROM alpine" in arg:
                found_default = True
        assert found_default

        mock_markdown.reset_mock()

        cb.on_tool_start(
            "TestAgent",
            "call_5",
            "filesystem_write_file",
            {"path": "dir.old/script", "content": "echo hello\nline2"},
        )
        found_none = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````\necho hello" in arg:
                found_none = True
        assert found_none

        mock_markdown.reset_mock()

        cb.on_tool_start(
            "TestAgent",
            "call_6",
            "filesystem_write_file",
            {"path": ".gitignore", "content": "node_modules/\nline2"},
        )
        found_none = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````\nnode_modules/" in arg:
                found_none = True
        assert found_none

        mock_markdown.reset_mock()

        cb.on_tool_start(
            "TestAgent",
            "call_7",
            "filesystem_write_file",
            {"path": "README.", "content": "content\nline2"},
        )
        found_none = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````\ncontent" in arg:
                found_none = True
        assert found_none
