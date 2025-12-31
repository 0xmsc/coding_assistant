from typing import Any
import os
from pathlib import Path
from coding_assistant.paths import (
    get_cache_home,
    get_state_home,
    get_data_home,
    get_config_home,
    get_app_cache_dir,
    get_app_state_dir,
)


def test_get_cache_home_default() -> None:
    if "XDG_CACHE_HOME" in os.environ:
        del os.environ["XDG_CACHE_HOME"]
    expected = Path.home() / ".cache"
    assert get_cache_home() == expected


def test_get_cache_home_override(monkeypatch: Any) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/cache")
    assert get_cache_home() == Path("/tmp/cache")


def test_get_state_home_default() -> None:
    if "XDG_STATE_HOME" in os.environ:
        del os.environ["XDG_STATE_HOME"]
    expected = Path.home() / ".local" / "state"
    assert get_state_home() == expected


def test_get_state_home_override(monkeypatch: Any) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", "/tmp/state")
    assert get_state_home() == Path("/tmp/state")


def test_get_data_home_default() -> None:
    if "XDG_DATA_HOME" in os.environ:
        del os.environ["XDG_DATA_HOME"]
    expected = Path.home() / ".local" / "share"
    assert get_data_home() == expected


def test_get_data_home_override(monkeypatch: Any) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/data")
    assert get_data_home() == Path("/tmp/data")


def test_get_config_home_default() -> None:
    if "XDG_CONFIG_HOME" in os.environ:
        del os.environ["XDG_CONFIG_HOME"]
    expected = Path.home() / ".config"
    assert get_config_home() == expected


def test_get_config_home_override(monkeypatch: Any) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/config")
    assert get_config_home() == Path("/tmp/config")


def test_get_app_cache_dir(monkeypatch: Any) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/cache")
    assert get_app_cache_dir() == Path("/tmp/cache/coding_assistant")


def test_get_app_state_dir(monkeypatch: Any) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", "/tmp/state")
    assert get_app_state_dir() == Path("/tmp/state/coding_assistant")
