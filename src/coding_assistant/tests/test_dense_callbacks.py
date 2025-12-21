from unittest.mock import patch, call, MagicMock
from coding_assistant import callbacks
from coding_assistant.callbacks import DenseProgressCallbacks, ReasoningState, ContentState, ToolState, IdleState


def test_dense_callbacks_lifecycle():
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        # 1. Idle
        assert cb._state is None

        # 2. Reasoning
        cb.on_reasoning_chunk("Thinking...")
        assert isinstance(cb._state, ReasoningState)
        cb.on_reasoning_chunk("\n\nDone thinking.")
        # ParagraphBuffer should have returned ["Thinking..."]
        # Then it should have printed:
        # 1. empty line (from on_reasoning_chunk start of state)
        # 2. empty line (before printing paragraph)
        # 3. Styled(Markdown("Thinking..."))

        # 3. Content
        cb.on_content_chunk("Hello")
        assert isinstance(cb._state, ContentState)
        cb.on_content_chunk(" world!\n\n")
        # ParagraphBuffer should have returned ["Hello world!"]

        # 4. Tool call
        cb.on_tool_start("TestAgent", "call_1", "test_tool", {"arg": "val"})
        assert isinstance(cb._state, ToolState)
        cb.on_tool_message("TestAgent", "call_1", "test_tool", {"arg": "val"}, "Tool result")

        # 5. End chunks (flushes remaining)
        cb.on_content_chunk("Final bit")
        cb.on_chunks_end()
        assert isinstance(cb._state, IdleState)

    # Verify that print was called
    assert mock_print.called


def test_dense_callbacks_tool_formatting():
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        # Test tool call with different result types
        cb.on_tool_start("TestAgent", "call_1", "mcp_coding_assistant_mcp_shell_execute", {"command": "ls"})
        cb.on_tool_message(
            "TestAgent", "call_1", "mcp_coding_assistant_mcp_shell_execute", {"command": "ls"}, "file1\nfile2"
        )

        # Verify it uses special handling for shell execute
        # We can check if Padding was called or just if print was called with certain arguments
        # Since we are mocking print, it's hard to see what's INSIDE the Padding/Panel/etc.
        # but we can at least ensure it doesn't crash.

    assert mock_print.called


def test_dense_callbacks_paragraph_flushing():
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        cb.on_content_chunk("One")
        cb.on_content_chunk(" Two")
        # No newline yet, so no paragraph printed yet (only the initial newline for state change)

        # Check that "One Two" hasn't been printed in a Markdown block yet
        # (Though it's hard with just mock_print)

        cb.on_chunks_end()
        # Now it should be flushed

    assert mock_print.called


def test_dense_callbacks_state_transition_flushes():
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        cb.on_reasoning_chunk("Thinking hard")
        # Switch to content - should flush reasoning?
        cb.on_content_chunk("Actually here is the answer")
        cb.on_chunks_end()

    # Let's see what was printed.
    # If it didn't flush "Thinking hard", then it's a bug.
    # We expect Styled(Markdown("Thinking hard"), "dim cyan") somewhere.

    found_reasoning = False
    for call_args in mock_print.call_args_list:
        for arg in call_args.args:
            if (
                hasattr(arg, "renderable")
                and hasattr(arg.renderable, "markup")
                and "Thinking hard" in arg.renderable.markup
            ):
                # This is a bit brittle as it depends on how Styled/Markdown are structured
                found_reasoning = True
            # Alternative check: just look for the string in any way
            if "Thinking hard" in str(arg):
                found_reasoning = True

    assert found_reasoning, "Reasoning should have been flushed when switching to content"


def test_dense_callbacks_empty_line_logic():
    cb = DenseProgressCallbacks()

    with patch("coding_assistant.callbacks.print") as mock_print:
        # 2. First reasoning chunk
        cb.on_reasoning_chunk("Thinking")

        # 3. Second reasoning chunk
        cb.on_reasoning_chunk(" more")

        # 4. Content chunk
        cb.on_content_chunk("Hello")

        # 5. Tool start
        cb.on_tool_start("TestAgent", "call_1", "test_tool", {"arg": 1})

        # 6. Content chunk after tool
        cb.on_content_chunk("Result")

        # Capture calls to print
        print_calls = [c for c in mock_print.call_args_list]

        # Verify newline when switching from reasoning to content
        found_newline = False
        for i in range(len(print_calls) - 1):
            if print_calls[i] == call():
                found_newline = True
                break
        assert found_newline, "Expected newline when switching from reasoning to content"


def test_dense_callbacks_multiline_tool_formatting(capsys):
    cb = DenseProgressCallbacks()
    # Force a wide terminal to avoid wrapping of long tool call lines
    callbacks.console.width = 200

    # 1. Unknown tool with multiline -> compact one-liner
    cb.on_tool_start("TestAgent", "call_1", "unknown_tool", {"arg": "line1\nline2"})
    captured = capsys.readouterr()
    assert 'unknown_tool(arg="line1\\nline2")' in captured.out

    # 2. Known special tool (shell_execute) with multiline -> fancy layout
    cb.on_tool_start("TestAgent", "call_2", "mcp_coding_assistant_mcp_shell_execute", {"command": "ls\npwd"})
    captured = capsys.readouterr()
    assert "▶ mcp_coding_assistant_mcp_shell_execute(command)" in captured.out
    assert "  command:" in captured.out
    assert "  ls" in captured.out
    assert "  pwd" in captured.out

    # 3. Known special tool (shell_execute) but SINGLE line -> compact one-liner
    cb.on_tool_start("TestAgent", "call_3", "mcp_coding_assistant_mcp_shell_execute", {"command": "ls"})
    captured = capsys.readouterr()
    assert 'mcp_coding_assistant_mcp_shell_execute(command="ls")' in captured.out

    # 4. Known special tool (filesystem_write_file) with multiline -> fancy layout
    cb.on_tool_start(
        "TestAgent",
        "call_4",
        "mcp_coding_assistant_mcp_filesystem_write_file",
        {"path": "test.py", "content": "def hello():\n    pass"},
    )
    # Verify it uses the extension
    assert cb._SPECIAL_TOOLS["mcp_coding_assistant_mcp_filesystem_write_file"]["content"] == ""
    # We can directly inspect the behavior via a mock of Markdown if needed,
    # but for now let's just make sure it runs without crashing and check captures.
    captured = capsys.readouterr()
    assert 'mcp_coding_assistant_mcp_filesystem_write_file(path="test.py", content)' in captured.out
    assert "  content:" in captured.out
    assert "  def hello():" in captured.out

    # 5. Known special tool (filesystem_edit_file) with multiline on one of the keys
    cb.on_tool_start(
        "TestAgent",
        "call_5",
        "mcp_coding_assistant_mcp_filesystem_edit_file",
        {"path": "script.sh", "old_text": "line1\nold", "new_text": "line1\nline2"},
    )
    # Verify it uses special handling for both keys
    assert "old_text" in cb._SPECIAL_TOOLS["mcp_coding_assistant_mcp_filesystem_edit_file"]
    assert "new_text" in cb._SPECIAL_TOOLS["mcp_coding_assistant_mcp_filesystem_edit_file"]

    captured = capsys.readouterr()
    assert 'mcp_coding_assistant_mcp_filesystem_edit_file(path="script.sh", old_text, new_text)' in captured.out
    assert "  old_text:" in captured.out
    assert "  new_text:" in captured.out

    # 6. Known special tool (python_execute) with multiline
    cb.on_tool_start(
        "TestAgent",
        "call_6",
        "mcp_coding_assistant_mcp_python_execute",
        {"code": "import os\nprint(os.getcwd())"},
    )
    captured = capsys.readouterr()
    assert "▶ mcp_coding_assistant_mcp_python_execute(code)" in captured.out
    assert "  code:" in captured.out

    # 7. Known special tool with non-string multiline (todo_add)
    cb.on_tool_start(
        "TestAgent",
        "call_7",
        "mcp_coding_assistant_mcp_todo_add",
        {"descriptions": ["task 1", "task 2"]},
    )
    captured = capsys.readouterr()
    assert "▶ mcp_coding_assistant_mcp_todo_add(descriptions)" in captured.out
    assert "  descriptions:" in captured.out
    assert '"task 1"' in captured.out
    assert '"task 2"' in captured.out

    # 8. Known special tool with key mismatch (multiline allowed but tool sends wrong key)
    # This should stay in header
    cb.on_tool_start(
        "TestAgent",
        "call_8",
        "mcp_coding_assistant_mcp_python_execute",
        {"not_code": "line1\nline2"},
    )
    captured = capsys.readouterr()
    assert 'mcp_coding_assistant_mcp_python_execute(not_code="line1\\nline2")' in captured.out

    # 9. Tool with key configured but NO multiline
    cb.on_tool_start(
        "TestAgent",
        "call_9",
        "mcp_coding_assistant_mcp_python_execute",
        {"code": "print(1)"},
    )
    captured = capsys.readouterr()
    assert 'mcp_coding_assistant_mcp_python_execute(code="print(1)")' in captured.out


def test_dense_callbacks_tool_result_stripping():
    cb = DenseProgressCallbacks()
    with patch("coding_assistant.callbacks.print") as mock_print:
        # Test filesystem_edit_file stripping
        cb.on_tool_message(
            "TestAgent",
            "call_1",
            "mcp_coding_assistant_mcp_filesystem_edit_file",
            {"path": "test.py", "old_text": "old", "new_text": "new"},
            "--- test.py\n+++ test.py\n-old\n+new\n",
        )

        # Verify the markup has no double newline
        found_diff = False
        for call_args in mock_print.call_args_list:
            args = call_args.args
            if args and hasattr(args[0], "renderable"):
                renderable = args[0].renderable
                if hasattr(renderable, "markup") and "```diff" in renderable.markup:
                    found_diff = True
                    # Should end with exactly one newline before the closing fence
                    assert renderable.markup.endswith("\n````")
                    assert not renderable.markup.endswith("\n\n````")
        assert found_diff

        # Reset mock
        mock_print.reset_mock()

        # Test todo stripping
        cb.on_tool_message("TestAgent", "call_2", "mcp_coding_assistant_mcp_todo_list_todos", {}, "- [ ] Task 1\n")

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
    # Force a wide terminal
    callbacks.console.width = 200

    with patch("coding_assistant.callbacks.Markdown", side_effect=callbacks.Markdown) as mock_markdown:
        # 1. Test .py extension
        cb.on_tool_start(
            "TestAgent",
            "call_1",
            "mcp_coding_assistant_mcp_filesystem_write_file",
            {"path": "test.py", "content": "def hello():\n    pass"},
        )
        found_py = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````py\ndef hello():" in arg:
                found_py = True
        assert found_py

        mock_markdown.reset_mock()

        # 2. Test .sh extension
        cb.on_tool_start(
            "TestAgent",
            "call_2",
            "mcp_coding_assistant_mcp_filesystem_write_file",
            {"path": "script.sh", "content": "echo hello\nls"},
        )
        found_sh = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````sh\necho hello" in arg:
                found_sh = True
        assert found_sh

        mock_markdown.reset_mock()

        # 3. Test .js extension
        cb.on_tool_start(
            "TestAgent",
            "call_3",
            "mcp_coding_assistant_mcp_filesystem_edit_file",
            {"path": "index.js", "old_text": "const x = 1\n", "new_text": "const x = 2\n"},
        )
        found_js = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````js\nconst x = " in arg:
                found_js = True
        assert found_js

        mock_markdown.reset_mock()

        # 4. Test no extension (fallback to default "")
        cb.on_tool_start(
            "TestAgent",
            "call_4",
            "mcp_coding_assistant_mcp_filesystem_write_file",
            {"path": "Dockerfile", "content": "FROM alpine\nRUN ls"},
        )
        found_default = False
        for call_args in mock_markdown.call_args_list:
            arg = call_args.args[0]
            if "````\nFROM alpine" in arg:
                found_default = True
        assert found_default
