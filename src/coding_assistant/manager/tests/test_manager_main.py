import pytest

from coding_assistant.manager.main import _environment_from_args, _manager_auth_secret_from_env


def test_manager_auth_secret_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", raising=False)

    with pytest.raises(ValueError, match="CODING_ASSISTANT_MANAGER_AUTH_SECRET must be set"):
        _manager_auth_secret_from_env()


def test_manager_auth_secret_is_read_from_fixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", "secret-token")

    assert _manager_auth_secret_from_env() == "secret-token"


def test_worker_env_rejects_manager_auth_secret() -> None:
    with pytest.raises(ValueError, match="CODING_ASSISTANT_MANAGER_AUTH_SECRET"):
        _environment_from_args(
            ["CODING_ASSISTANT_MANAGER_AUTH_SECRET=secret-token"],
            forbidden_keys={"CODING_ASSISTANT_MANAGER_AUTH_SECRET"},
        )
