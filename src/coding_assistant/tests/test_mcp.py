from coding_assistant.tools.mcp import get_default_env


def test_get_default_env_includes_https_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    env = get_default_env()
    assert env.get("HTTPS_PROXY") == "http://proxy:8080"
