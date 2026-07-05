from pathlib import Path

import pytest

from coding_assistant.manager.main import (
    _manager_auth_secret_from_env,
    _manager_config_from_env,
    _reject_cli_args,
    _worker_environment_from_env,
    _worker_extra_hosts_from_env,
)


MANAGER_ENV_KEYS = (
    "CODING_ASSISTANT_HOST_DATA_DIR",
    "CODING_ASSISTANT_MANAGER_AUTH_SECRET",
    "CODING_ASSISTANT_WORKER_EXTRA_HOSTS",
    "CODING_ASSISTANT_WORKER_IMAGE",
    "CODING_ASSISTANT_WORKER_NETWORK",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
)


def _clear_manager_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in MANAGER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_manager_auth_secret_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", raising=False)

    with pytest.raises(ValueError, match="CODING_ASSISTANT_MANAGER_AUTH_SECRET must be set"):
        _manager_auth_secret_from_env()


def test_manager_auth_secret_is_read_from_fixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", "secret-token")

    assert _manager_auth_secret_from_env() == "secret-token"


def test_manager_config_is_read_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_manager_env(monkeypatch)
    monkeypatch.setenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", "secret-token")
    monkeypatch.setenv("CODING_ASSISTANT_HOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CODING_ASSISTANT_WORKER_IMAGE", "coding-assistant:test")
    monkeypatch.setenv("CODING_ASSISTANT_WORKER_NETWORK", "test-network")
    monkeypatch.setenv("CODING_ASSISTANT_WORKER_EXTRA_HOSTS", "host.docker.internal:host-gateway")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = _manager_config_from_env()

    assert config.auth_secret == "secret-token"
    assert config.data_dir == Path("/data")
    assert config.database_path == Path("/data/sessions.sqlite")
    assert config.session_root == Path("/data/sessions")
    assert config.host_data_dir == tmp_path
    assert config.session_source_root == tmp_path / "sessions"
    assert config.worker_image == "coding-assistant:test"
    assert config.worker_network == "test-network"
    assert config.worker_extra_hosts == ("host.docker.internal:host-gateway",)


def test_manager_config_accepts_openrouter_provider_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_manager_env(monkeypatch)
    monkeypatch.setenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", "secret-token")
    monkeypatch.setenv("CODING_ASSISTANT_HOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CODING_ASSISTANT_WORKER_IMAGE", "coding-assistant:test")
    monkeypatch.setenv("CODING_ASSISTANT_WORKER_NETWORK", "test-network")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    config = _manager_config_from_env()

    assert config.auth_secret == "secret-token"


def test_manager_config_requires_provider_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_manager_env(monkeypatch)
    monkeypatch.setenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", "secret-token")
    monkeypatch.setenv("CODING_ASSISTANT_HOST_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CODING_ASSISTANT_WORKER_IMAGE", "coding-assistant:test")
    monkeypatch.setenv("CODING_ASSISTANT_WORKER_NETWORK", "test-network")

    with pytest.raises(ValueError, match="OPENAI_API_KEY or OPENROUTER_API_KEY must be set"):
        _manager_config_from_env()


def test_manager_config_requires_absolute_host_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_manager_env(monkeypatch)
    monkeypatch.setenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", "secret-token")
    monkeypatch.setenv("CODING_ASSISTANT_HOST_DATA_DIR", "relative-data")
    monkeypatch.setenv("CODING_ASSISTANT_WORKER_IMAGE", "coding-assistant:test")
    monkeypatch.setenv("CODING_ASSISTANT_WORKER_NETWORK", "test-network")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(ValueError, match="CODING_ASSISTANT_HOST_DATA_DIR must be an absolute path"):
        _manager_config_from_env()


def test_manager_rejects_cli_args() -> None:
    with pytest.raises(ValueError, match="configured only through environment variables"):
        _reject_cli_args(("coding-assistant-manager", "--model", "test-model"))


def test_worker_environment_forwards_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", "secret-token")
    monkeypatch.setenv("UNRELATED_ENV", "ignored")

    assert _worker_environment_from_env() == {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://example.invalid/v1",
        "OPENROUTER_API_KEY": "openrouter-key",
    }


def test_worker_environment_omits_missing_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert _worker_environment_from_env() == {}


def test_worker_environment_omits_blank_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert _worker_environment_from_env() == {"OPENAI_API_KEY": "test-key"}


def test_worker_extra_hosts_are_read_from_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CODING_ASSISTANT_WORKER_EXTRA_HOSTS",
        " host.docker.internal:host-gateway, apps.local:192.0.2.10 ,, ",
    )

    assert _worker_extra_hosts_from_env() == (
        "host.docker.internal:host-gateway",
        "apps.local:192.0.2.10",
    )
