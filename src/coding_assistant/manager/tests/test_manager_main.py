import pytest

from coding_assistant.manager.main import _manager_auth_secret_from_env, _worker_environment_from_env


def test_manager_auth_secret_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", raising=False)

    with pytest.raises(ValueError, match="CODING_ASSISTANT_MANAGER_AUTH_SECRET must be set"):
        _manager_auth_secret_from_env()


def test_manager_auth_secret_is_read_from_fixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", "secret-token")

    assert _manager_auth_secret_from_env() == "secret-token"


def test_worker_environment_forwards_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", "secret-token")
    monkeypatch.setenv("UNRELATED_ENV", "ignored")

    assert _worker_environment_from_env() == {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://example.invalid/v1",
    }


def test_worker_environment_omits_missing_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert _worker_environment_from_env() == {}
