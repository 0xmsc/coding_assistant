import pytest
from unittest.mock import patch

from coding_assistant.worker.main import TOOL_ENV_KEYS_ENV, _tool_process_env_from_env, parse_args


def test_tool_process_env_from_env_includes_only_listed_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOOL_ENV_KEYS_ENV, " APPS_API_BASE_URL, APPS_API_TOKEN, MISSING ")
    monkeypatch.setenv("APPS_API_BASE_URL", "http://apps-api")
    monkeypatch.setenv("APPS_API_TOKEN", "prompt-token")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")

    assert _tool_process_env_from_env() == {
        "APPS_API_BASE_URL": "http://apps-api",
        "APPS_API_TOKEN": "prompt-token",
    }


def test_worker_parse_args_defaults() -> None:
    with patch("sys.argv", ["coding-assistant-worker", "--model", "gpt-4"]):
        args = parse_args()
        assert args.auto_compact is True
        assert args.auto_compact_token_budget is None


def test_worker_parse_args_auto_compact_flags() -> None:
    with patch(
        "sys.argv",
        [
            "coding-assistant-worker",
            "--model",
            "gpt-4",
            "--no-auto-compact",
            "--auto-compact-token-budget",
            "75000",
        ],
    ):
        args = parse_args()
        assert args.auto_compact is False
        assert args.auto_compact_token_budget == 75000
