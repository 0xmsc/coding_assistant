from datetime import datetime
from pathlib import Path
import logging

from coding_assistant.paths import get_app_cache_dir

_trace_dir_: Path | None = None
_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = logging.getLogger(__name__)


def get_default_trace_dir() -> Path:
    return get_app_cache_dir() / "traces" / _session_id


def enable_tracing(directory: Path, clear: bool = False) -> None:
    """
    Enable tracing globally.

    Args:
        directory: The directory to store traces in.
        clear: Whether to clear the traces directory.
    """
    global _trace_dir_

    _trace_dir_ = directory
    _trace_dir_.mkdir(parents=True, exist_ok=True)

    if clear:
        for f in _trace_dir_.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)

    logger.info(f"Tracing to {_trace_dir_}")


def trace_enabled() -> bool:
    return _trace_dir_ is not None


def trace_data(name: str, content: str) -> None:
    if _trace_dir_ is None:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    trace_file = _trace_dir_ / f"{timestamp}_{name}"
    with open(trace_file, "w") as f:
        f.write(content)
