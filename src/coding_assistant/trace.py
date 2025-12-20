from datetime import datetime
from pathlib import Path
import logging

from coding_assistant.paths import get_app_cache_dir

_trace_enabled = False

logger = logging.getLogger(__name__)


def _get_trace_dir() -> Path:
    return get_app_cache_dir() / "traces"


def enable_tracing() -> None:
    """Enable tracing globally."""
    global _trace_enabled
    _trace_enabled = True

    # Empty the traces directory
    trace_path = _get_trace_dir()
    if trace_path.exists():
        for f in trace_path.iterdir():
            if f.is_file():
                f.unlink()

    logger.info(f"Tracing to {trace_path}")


def trace_enabled() -> bool:
    """Check if tracing is enabled."""
    return _trace_enabled


def trace_data(name: str, content: str) -> None:
    """
    Write content to a trace file if tracing is enabled.

    Args:
        name: The base name for the trace file.
        content: The content to trace.
    """
    if not trace_enabled():
        return

    trace_path = _get_trace_dir()
    trace_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    trace_file = trace_path / f"{timestamp}_{name}"
    with open(trace_file, "w") as f:
        f.write(content)
