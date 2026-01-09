import os
from pathlib import Path
from datetime import datetime


SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


def get_cache_home() -> Path:
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home)
    return Path.home() / ".cache"


def get_state_home() -> Path:
    xdg_state_home = os.getenv("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home)
    return Path.home() / ".local" / "state"


def get_data_home() -> Path:
    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home)
    return Path.home() / ".local" / "share"


def get_config_home() -> Path:
    xdg_config_home = os.getenv("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home)
    return Path.home() / ".config"


def get_app_cache_dir() -> Path:
    return get_cache_home() / "coding_assistant"


def get_app_state_dir() -> Path:
    return get_state_home() / "coding_assistant"


def get_session_dir() -> Path:
    session_dir = get_app_cache_dir() / "sessions" / SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_log_file() -> Path:
    return get_session_dir() / "session.log"


def get_traces_dir() -> Path:
    return get_session_dir() / "traces"
