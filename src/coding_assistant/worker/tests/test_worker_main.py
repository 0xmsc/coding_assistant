import pytest

from coding_assistant.worker.main import TOOL_ENV_KEYS_ENV, _tool_process_env_from_env


def test_tool_process_env_from_env_includes_only_listed_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOOL_ENV_KEYS_ENV, " APPS_API_BASE_URL, APPS_API_TOKEN, MISSING ")
    monkeypatch.setenv("APPS_API_BASE_URL", "http://apps-api")
    monkeypatch.setenv("APPS_API_TOKEN", "prompt-token")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")

    assert _tool_process_env_from_env() == {
        "APPS_API_BASE_URL": "http://apps-api",
        "APPS_API_TOKEN": "prompt-token",
    }
