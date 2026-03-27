from typing import Any
from coding_assistant.integrations.mcp_client import get_default_env


def test_get_default_env_includes_https_proxy(monkeypatch: Any) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    env = get_default_env()
    assert env.get("HTTPS_PROXY") == "http://proxy:8080"
