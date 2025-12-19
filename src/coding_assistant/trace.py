import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

_trace_enabled = False

def enable_tracing() -> None:
    """Enable tracing globally."""
    global _trace_enabled
    _trace_enabled = True

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

    xdg_state_home = os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state"))
    trace_path = Path(xdg_state_home) / "coding-assistant" / "traces"
    trace_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    trace_file = trace_path / f"{timestamp}_{name}.txt"
    with open(trace_file, "w") as f:
        f.write(content)
