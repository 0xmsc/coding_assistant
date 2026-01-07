from typing import Any
import pytest
from unittest.mock import patch

from coding_assistant.main import parse_args, main, create_config_from_args


def test_parse_args_valid() -> None:
    """Test parse_args with valid arguments."""
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--task", "test"]):
        args = parse_args()
        assert args.model == "gpt-4"
        assert args.task == "test"
        assert args.sandbox is True


def test_parse_args_defaults() -> None:
    """Test default values."""
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4"]):
        args = parse_args()
        assert args.ask_user is True
        assert args.compact_conversation_at_tokens == 200000


def test_parse_args_with_multiple_flags() -> None:
    """Test parse_args with multiple boolean flags."""
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--trace", "--no-sandbox", "--no-ask-user"]):
        args = parse_args()
        assert args.trace is True
        assert args.sandbox is False
        assert args.ask_user is False


def test_create_config_from_args() -> None:
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


def test_create_config_with_expert_model() -> None:
    """Test Config creation with explicit expert_model."""
    args = type("MockArgs", (), {})()
    args.model = "gpt-4"
    args.expert_model = "gpt-4-turbo"
    args.compact_conversation_at_tokens = 100000
    args.task = None
    args.ask_user = True
    config = create_config_from_args(args)
    assert config.model == "gpt-4"
    assert config.expert_model == "gpt-4-turbo"
    assert config.enable_chat_mode is True


def test_create_config_fallback_to_model() -> None:
    """Test that expert_model falls back to model when None."""
    args = type("MockArgs", (), {})()
    args.model = "gpt-4o"
    args.expert_model = None
    args.compact_conversation_at_tokens = 100000
    args.task = "some task"
    args.ask_user = True
    config = create_config_from_args(args)
    assert config.model == "gpt-4o"
    assert config.expert_model == "gpt-4o"
    assert config.enable_chat_mode is False


@patch("coding_assistant.main._main")
@patch("coding_assistant.main.enable_tracing")
def test_main_enables_tracing_when_flag_set(mock_enable_tracing: Any, mock_main: Any) -> None:
    """Test main enables tracing when --trace is True."""
    with patch("sys.argv", ["coding-assistant", "--model", "test-model", "--trace"]):
        main()
        mock_enable_tracing.assert_called_once()
        mock_main.assert_called_once()


@patch("coding_assistant.main._main")
@patch("coding_assistant.main.debugpy.wait_for_client")
@patch("coding_assistant.main.debugpy.listen")
def test_main_waits_for_debugger(mock_listen: Any, mock_wait: Any, mock_main: Any) -> None:
    """Test main waits for debugger when --wait-for-debugger is True."""
    with patch("sys.argv", ["coding-assistant", "--model", "test-model", "--wait-for-debugger"]):
        main()
        mock_listen.assert_called_once_with(1234)
        mock_wait.assert_called_once()
        mock_main.assert_called_once()


@patch("coding_assistant.main._main")
def test_main_enters_chat_mode_by_default(mock_main: Any) -> None:
    """Test main enters chat mode when no --task is provided."""
    with patch("sys.argv", ["coding-assistant", "--model", "test-model"]):
        main()
        mock_main.assert_called_once()


def test_parse_args_mcp_env() -> None:
    """Test parse_args with --mcp-env."""
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--mcp-env", "VAR1", "VAR2"]):
        args = parse_args()
        assert args.mcp_env == ["VAR1", "VAR2"]


def test_help_exits_with_zero() -> None:
    """Test that --help exits cleanly with status 0."""
    with patch("sys.argv", ["coding-assistant", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            parse_args()
        assert exc_info.value.code == 0
