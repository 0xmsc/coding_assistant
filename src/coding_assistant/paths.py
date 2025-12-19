import os
from pathlib import Path


def get_cache_home() -> Path:
    """Return the XDG_CACHE_HOME directory, defaulting to ~/.cache."""
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home)
    return Path.home() / ".cache"


def get_state_home() -> Path:
    """Return the XDG_STATE_HOME directory, defaulting to ~/.local/state."""
    xdg_state_home = os.getenv("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home)
    return Path.home() / ".local" / "state"


def get_app_cache_dir() -> Path:
    """Return the application-specific cache directory."""
    return get_cache_home() / "coding_assistant"


def get_app_state_dir() -> Path:
    """Return the application-specific state directory."""
    return get_state_home() / "coding_assistant"
