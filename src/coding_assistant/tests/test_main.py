from unittest.mock import patch

import coding_assistant.main
from coding_assistant.main import parse_args, main, create_config_from_args


@patch("subprocess.run")
def test_cli_help_via_subprocess_mock(mock_run):
    """Mocked test for --help (replace with real CLI testing when setup)."""
    mock_run.return_value = type("Proc", (), {"returncode": 0, "stdout": "usage: ", "stderr": ""})()

    # Test parse_args exits cleanly on --help (SystemExit expected)
    with patch("sys.argv", ["coding-assistant", "--help", "--model", "test"]):
        with patch("sys.exit") as mock_exit:
            try:
                parse_args()
            except SystemExit:
                pass  # Expected for --help
            mock_exit.assert_called_once_with(0)  # --help exits with 0


@patch("coding_assistant.main.asyncio.run")
@patch("coding_assistant.main.enable_tracing")
def test_main_with_trace(mock_enable_tracing, mock_asyncio_run):
    """Test main enables tracing when --trace is True."""
    with patch("coding_assistant.main.parse_args") as mock_parse:
        mock_args = mock_parse.return_value
        mock_args.trace = True
        mock_args.wait_for_debugger = False
        main()
        mock_enable_tracing.assert_called_once()
        mock_asyncio_run.assert_called_once()


@patch("coding_assistant.main.debugpy.wait_for_client")
@patch("coding_assistant.main.debugpy.listen")
@patch("coding_assistant.main.asyncio.run")
def test_main_with_debugger(mock_asyncio_run, mock_listen, mock_wait):
    """Test main waits for debugger."""
    with patch("coding_assistant.main.parse_args") as mock_parse:
        mock_args = mock_parse.return_value
        mock_args.trace = False
        mock_args.wait_for_debugger = True
        main()
        mock_listen.assert_called_once_with(1234)
        mock_wait.assert_called_once()
        mock_asyncio_run.assert_called_once()


@patch("coding_assistant.main.asyncio.run")
def test_main_full_execution_mocked(mock_asyncio_run):
    """Test main function execution with real parse_args."""
    with patch("sys.argv", ["coding-assistant", "--model", "test-model"]):
        main()
        mock_asyncio_run.assert_called_once()


def test_parse_args_valid():
    """Test parse_args with valid arguments."""
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--task", "test"]):
        args = parse_args()
        assert args.model == "gpt-4"
        assert args.task == "test"
        assert args.sandbox is True


def test_parse_args_defaults():
    """Test default values."""
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4"]):
        args = parse_args()
        assert args.ask_user is True
        assert args.compact_conversation_at_tokens == 200000


def test_create_config_from_args():
    """Test Config object creation from args."""
    args = type("MockArgs", (), {})()
    args.model = "gpt-4"
    args.expert_model = None
    args.compact_conversation_at_tokens = 100000
    args.task = None
    args.ask_user = True
    config = create_config_from_args(args)
    assert config.model == "gpt-4"
    assert config.expert_model == "gpt-4"
    assert config.enable_chat_mode is True


# Note: Async functions like run_root_agent and run_chat_session can be tested separately with pytest-asyncio
# For now, focus on getting basic coverage up from 0%
